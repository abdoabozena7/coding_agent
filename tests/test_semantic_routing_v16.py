from __future__ import annotations

import json
import tempfile
from dataclasses import replace
from pathlib import Path
from unittest import mock

import pytest

from agent.commands import parse_command
from agent.chat_runtime import SemanticTurnDecisionV2
from agent.config import RuntimeConfig
from agent.intake import RunMode
from agent.runtime import AgentRuntime, ProviderUnavailableError
from agent.store import StateStore
from agent.testing import (
    ScriptedProvider,
    semantic_goal_intake,
    semantic_turn,
)
from tests.test_runtime import inspect_call, plan_call, plan_pass


def config() -> RuntimeConfig:
    return replace(
        RuntimeConfig.from_env(),
        max_provider_retries=0,
        retry_base_ms=0,
        repository_index_warmup_files=0,
    )


def runtime(workspace: Path, turns, *, approval=lambda *_: True):
    store = StateStore(workspace)
    value = AgentRuntime(
        ScriptedProvider(turns), store, workspace,
        config=config(), approval=approval,
    )
    return value, store


@pytest.mark.parametrize(
    ("prompt", "response"),
    [
        ("Hello", "Hello! How can I help?"),
        ("Explain how a calculator app works", "A calculator maps input to arithmetic operations."),
        (
            "Explain a project website with multiple files and how an app can be tested; do not build anything.",
            "It can separate UI, logic, and tests without requesting a build.",
        ),
    ],
)
def test_natural_and_explanatory_turns_are_one_inference_chat(prompt: str, response: str) -> None:
    with tempfile.TemporaryDirectory() as directory:
        workspace = Path(directory)
        agent, store = runtime(
            workspace,
            [semantic_turn("chat", original=prompt, response=response)],
        )
        try:
            result = agent.chat(prompt)
            assert result.status == "chat"
            assert result.message == response
            assert len(agent.provider.calls) == 1
            assert agent.active_goal() is None
            assert store.list_session_actions(agent.session_id) == ()
        finally:
            agent.close(); store.close()


def test_provider_nested_semantic_transport_is_flattened_without_inventing_fields() -> None:
    original = "Create note.txt"
    decision = SemanticTurnDecisionV2.from_mapping(
        {
            "route": "action",
            "needs_workspace_tools": True,
            "direct_response": "",
            "authority_spans": {"write": [original]},
            "semantic_turn": {
                "interpretation": "Create the explicitly requested file.",
                "requested_effects": {"write": True, "read": False},
                "uncertainty": "clear",
            },
        },
        original_input=original,
    )
    assert decision.interpretation == "Create the explicitly requested file."
    assert [item.value for item in decision.requested_effects] == ["write"]


def test_exact_input_bytes_remain_the_semantic_source_of_truth() -> None:
    prompt = "  Explain the parser.\n"
    with tempfile.TemporaryDirectory() as directory:
        workspace = Path(directory)
        agent, store = runtime(workspace, [
            semantic_turn("chat", original=prompt, response="It parses input.")
        ])
        try:
            agent.route_input(prompt)
            state = store.get_workflow_session(agent.session_id)["state"]
            assert state["last_semantic_turn"]["original_input"] == prompt
            assert store.list_chat_messages(agent.session_id)[0]["content"] == prompt
        finally:
            agent.close(); store.close()


def test_simple_calculator_preview_is_one_bounded_action_without_goal_or_agents() -> None:
    prompt = "Create a simple calculator and preview it"
    html = "<!doctype html><title>Calculator</title><button>+</button>"
    preview = json.dumps({
        "status": "running", "preview_id": "preview-calculator",
        "url": "http://127.0.0.1:4567/token/index.html", "http_status": 200,
        "browser_opened": True, "verification": "passed",
        "console_errors": [], "page_errors": [], "network_errors": [],
    })
    with tempfile.TemporaryDirectory() as directory:
        workspace = Path(directory)
        agent, store = runtime(workspace, [
            semantic_turn("action", original=prompt, effects=("write", "preview")),
            {"tool_calls": [
                {"id": "write", "name": "write_file", "args": {"path": "index.html", "content": html}},
                {"id": "preview", "name": "preview_html", "args": {"path": "index.html"}},
            ]},
            "The calculator was created and previewed successfully.",
        ])
        try:
            with mock.patch("agent.tools.web_preview.create", return_value=preview):
                decision, result = agent.route_input(prompt)
            assert decision.kind.value == "action"
            assert result.status == "chat"
            assert agent.active_goal() is None
            assert (workspace / "index.html").read_text(encoding="utf-8") == html
            assert [item["tool_name"] for item in store.list_session_actions(agent.session_id)] == [
                "write_file", "preview_html",
            ]
            assert store.list_agent_registry() == ()
        finally:
            agent.close(); store.close()


def test_action_tool_contract_rejects_unrequested_category_before_execution() -> None:
    prompt = "Create note.txt"
    with tempfile.TemporaryDirectory() as directory:
        workspace = Path(directory)

        def disallowed_call(request):
            names = {item["function"]["name"] for item in request.tools}
            assert "write_file" in names
            assert "run_command" not in names
            return {"tool_calls": [{
                "id": "bad", "name": "run_command", "args": {"command": "echo unauthorized"},
            }]}

        agent, store = runtime(workspace, [
            semantic_turn("action", original=prompt, effects=("write",)),
            disallowed_call,
            {"tool_calls": [{
                "id": "write", "name": "write_file",
                "args": {"path": "note.txt", "content": "safe\n"},
            }]},
            "Created note.txt.",
        ])
        try:
            result = agent.chat(prompt)
            assert "Created note.txt" in result.message
            assert [item["tool_name"] for item in store.list_session_actions(agent.session_id)] == ["write_file"]
            assert (workspace / "note.txt").read_text(encoding="utf-8") == "safe\n"
        finally:
            agent.close(); store.close()


def test_complex_goal_preserves_exact_original_and_normal_is_not_silently_ultra() -> None:
    prompt = "Build a coordinated API, worker, and browser client with integration tests."
    intake = semantic_goal_intake(prompt)
    with tempfile.TemporaryDirectory() as directory:
        workspace = Path(directory)
        agent, store = runtime(workspace, [
            semantic_turn(
                "goal", original=prompt, goal_intake=intake,
                effects=("read", "write", "run"),
            ),
            inspect_call(), plan_call(), plan_pass(),
        ])
        try:
            decision, plan = agent.route_input(prompt)
            assert decision.kind.value == "goal"
            assert plan.goal_id == agent.active_goal().id
            assert agent.active_goal().objective == prompt
            assert agent.active_goal().metadata["execution_policy"]["mode"] == "normal"
            assert "ultra_run_id" not in agent.active_goal().metadata
        finally:
            agent.close(); store.close()


def test_schema_and_semantic_repair_budgets_are_independent_and_resumable() -> None:
    prompt = "Hello"
    malformed_semantic = semantic_turn("action", original=prompt, effects=("write",))
    malformed_semantic["tool_calls"][0]["args"]["authority_spans"] = {"write": ["not verbatim"]}
    with tempfile.TemporaryDirectory() as directory:
        workspace = Path(directory)
        agent, store = runtime(workspace, [{}, {}, {}])
        try:
            with pytest.raises(ProviderUnavailableError, match="could not be validated"):
                agent.route_input(prompt)
            pending = store.get_workflow_session(agent.session_id)["state"]["pending_semantic_turn"]
            assert pending["schema_attempts"] == 3
            assert pending["semantic_attempts"] == 0
            assert pending["status"] == "awaiting_provider"
        finally:
            agent.close(); store.close()

    with tempfile.TemporaryDirectory() as directory:
        workspace = Path(directory)
        agent, store = runtime(workspace, [malformed_semantic] * 3)
        try:
            with pytest.raises(ProviderUnavailableError, match="could not be validated"):
                agent.route_input(prompt)
            pending = store.get_workflow_session(agent.session_id)["state"]["pending_semantic_turn"]
            assert pending["schema_attempts"] == 0
            assert pending["semantic_attempts"] == 3
        finally:
            agent.close(); store.close()


def test_provider_outage_has_no_canned_reply_and_resume_reuses_exact_turn() -> None:
    prompt = "Hello after restart"
    with tempfile.TemporaryDirectory() as directory:
        workspace = Path(directory)
        first, first_store = runtime(workspace, [RuntimeError("provider offline")])
        try:
            with pytest.raises(ProviderUnavailableError, match="provider unavailable"):
                first.chat(prompt)
            pending = first_store.get_workflow_session(first.session_id)["state"]["pending_semantic_turn"]
            assert pending["original_input"] == prompt
            assert pending["status"] == "awaiting_provider"
            assert not any(item.get("role") == "assistant" for item in first_store.list_chat_messages(first.session_id))
        finally:
            first.close(); first_store.close()

        second_store = StateStore(workspace)
        second = AgentRuntime(
            ScriptedProvider([
                semantic_turn("chat", original=prompt, response="Welcome back.")
            ]),
            second_store,
            workspace,
            config=config(),
        )
        try:
            result = second.resume()
            assert result.message == "Welcome back."
            users = [
                item for item in second_store.list_chat_messages(second.session_id)
                if item.get("role") == "user" and item.get("content") == prompt
            ]
            assert len(users) == 1
            assert "pending_semantic_turn" not in second_store.get_workflow_session(second.session_id)["state"]
        finally:
            second.close(); second_store.close()


def test_restart_marks_mid_action_mutation_uncertain_and_never_replays_it() -> None:
    prompt = "Create note.txt"
    with tempfile.TemporaryDirectory() as directory:
        workspace = Path(directory)
        first, first_store = runtime(workspace, [
            semantic_turn("action", original=prompt, effects=("write",))
        ])
        semantic_turn_state, decision = first._semantic_preflight(prompt)
        semantic_turn_state["status"] = "dispatching"
        first._save_pending_semantic_turn(semantic_turn_state)
        action_id = first_store.begin_session_action(
            first.session_id,
            "write_file",
            {"path": "note.txt", "content": "not replayed"},
            risk="high",
            mutating=True,
        )
        first._record_semantic_action(
            str(semantic_turn_state["turn_id"]),
            action_id,
            category="write",
            mutating=True,
            status="running",
        )
        first.close(); first_store.close()

        second_store = StateStore(workspace)
        second = AgentRuntime(
            ScriptedProvider([]), second_store, workspace, config=config()
        )
        try:
            result = second.resume()
            assert result.status == "uncertain"
            assert result.needs_user
            assert second.provider.remaining == 0
            assert not (workspace / "note.txt").exists()
            actions = second_store.list_session_actions(second.session_id)
            assert actions[-1]["status"] == "uncertain"
        finally:
            second.close(); second_store.close()


def test_plan_mode_rejects_changing_action_and_model_repairs_to_goal() -> None:
    prompt = "Create note.txt"
    with tempfile.TemporaryDirectory() as directory:
        workspace = Path(directory)
        agent, store = runtime(workspace, [
            semantic_turn("action", original=prompt, effects=("write",)),
            semantic_turn(
                "goal", original=prompt, goal_intake=semantic_goal_intake(prompt), effects=("write",),
            ),
            inspect_call(), plan_call(), plan_pass(),
        ])
        try:
            agent.transition_mode(RunMode.PLAN.value)
            decision, plan = agent.route_input(prompt)
            assert decision.kind.value == "goal"
            assert agent.active_goal().metadata["execution_policy"]["mode"] == "plan"
            assert store.get_workflow_session(agent.session_id)["session_mode"] == "plan"
            assert not (workspace / "note.txt").exists()
            routed = [event for event in store.list_recent_events(limit=100) if event.event_type == "semantic_turn.routed"]
            assert routed[-1].payload["semantic_attempts"] == 1
        finally:
            agent.close(); store.close()


def test_goal_command_forces_goal_intake_without_general_route_guess() -> None:
    prompt = "Explain this as a durable project"
    with tempfile.TemporaryDirectory() as directory:
        workspace = Path(directory)

        def forced_goal(request):
            assert '"forced_route": "goal"' in request.conversation[0]["content"]
            return semantic_turn(
                "goal", original=prompt,
                goal_intake=semantic_goal_intake(prompt), effects=("read",),
            )

        agent, store = runtime(workspace, [forced_goal, inspect_call(), plan_call(), plan_pass()])
        try:
            plan = agent.apply_command(parse_command("/goal " + prompt))
            assert plan.goal_id == agent.active_goal().id
            events = store.list_recent_events(limit=100)
            routing = next(event for event in events if event.event_type == "semantic_turn.routing")
            assert routing.payload["forced_route"] == "goal"
        finally:
            agent.close(); store.close()
