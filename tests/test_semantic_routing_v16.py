from __future__ import annotations

import json
import tempfile
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from agent.commands import parse_command
from agent.chat_runtime import (
    RequestedEffectV2,
    SemanticRouteDecisionV3,
    SemanticTurnDecisionV2,
)
from agent.config import RuntimeConfig
from agent.intake import RunMode
from agent.runtime import AgentRuntime, ProviderUnavailableError, RuntimeStateError
from agent.store import StateStore
from agent.testing import (
    ScriptedProvider,
    semantic_goal_intake,
    semantic_goal_intake_turn,
    semantic_route,
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


def test_unquoted_external_effect_is_safely_contracted_without_blocking_local_work() -> None:
    original = "create the local app and run it"
    decision = SemanticRouteDecisionV3.from_mapping(
        {
            "route": "goal",
            "outcome_kind": "runnable_product",
            "interpretation": "Create and run the local app.",
            "requested_effects": {
                "write": True,
                "run": True,
                "external_side_effect": True,
            },
            "authority_spans": {
                "write": ["create the local app"],
                "run": ["run it"],
            },
            "needs_workspace_tools": True,
            "direct_response": "",
            "uncertainty": "clear",
            "clarification_question": "",
            "task_demand": {
                "reasoning": 2,
                "implementation": 3,
                "context_breadth": 2,
                "coordination": 2,
                "verification": 3,
                "visual_runtime": 2,
                "component_count": 2,
                "independently_parallelizable": False,
                "rationale": ["The model authored a local multi-step build."],
            },
        },
        original_input=original,
    )

    assert RequestedEffectV2.WRITE in decision.requested_effects
    assert RequestedEffectV2.RUN in decision.requested_effects
    assert RequestedEffectV2.EXTERNAL not in decision.requested_effects


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


def test_weak_model_scalar_and_descriptive_route_shapes_are_normalized() -> None:
    original = "Create note.txt"
    payload = semantic_route(
        "action", original=original, effects=("write",),
        outcome_kind="workspace_operation",
    )["tool_calls"][0]["args"]
    payload["requested_effects"] = "write"
    payload["authority_spans"]["write"] = original
    payload["needs_workspace_tools"] = "true"
    payload["uncertainty"] = "none"
    payload["task_demand"]["visual_runtime"] = "not applicable"
    payload["task_demand"]["independently_parallelizable"] = "no"
    payload["task_demand"]["rationale"] = "A model-authored reason"

    decision = SemanticTurnDecisionV2.from_mapping(payload, original_input=original)

    assert decision.uncertainty == "clear"
    assert decision.needs_workspace_tools is True
    assert decision.task_demand.visual_runtime == 1
    assert decision.task_demand.rationale == ("A model-authored reason",)


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


def test_simple_calculator_is_a_normal_goal_without_ultra_agents() -> None:
    prompt = "Create a simple calculator and preview it"
    with tempfile.TemporaryDirectory() as directory:
        workspace = Path(directory)
        agent, store = runtime(workspace, [
            semantic_route(
                "goal", original=prompt, outcome_kind="runnable_product",
                effects=("write", "preview"),
            ),
            semantic_goal_intake_turn(semantic_goal_intake(prompt)),
            inspect_call(), plan_call(), plan_pass(),
        ])
        try:
            decision, plan = agent.route_input(prompt)
            assert decision.kind.value == "goal"
            assert plan.goal_id == agent.active_goal().id
            assert agent.active_goal().metadata["execution_policy"]["mode"] == "normal"
            assert not (workspace / "index.html").exists()
            assert store.list_agent_registry() == ()
        finally:
            agent.close(); store.close()


def test_invalid_goal_intake_repairs_without_rerouting_the_goal() -> None:
    prompt = "Create a Three.js calculator and run it"
    malformed = semantic_goal_intake(prompt)
    malformed["component_count"] = "several"
    with tempfile.TemporaryDirectory() as directory:
        workspace = Path(directory)
        agent, store = runtime(workspace, [
            semantic_route(
                "goal", original=prompt, outcome_kind="runnable_product",
                effects=("write", "run"),
            ),
            semantic_goal_intake_turn(malformed),
            semantic_goal_intake_turn(semantic_goal_intake(prompt)),
            inspect_call(), plan_call(), plan_pass(),
        ])
        try:
            decision, _plan = agent.route_input(prompt)
            assert decision.kind.value == "goal"
            events = store.list_recent_events(limit=100)
            assert len([item for item in events if item.event_type == "semantic_turn.routing"]) == 1
            accepted = next(item for item in events if item.event_type == "semantic_turn.intake_accepted")
            assert accepted.payload["semantic_attempts"] == 1
            routed = next(item for item in events if item.event_type == "semantic_turn.routed")
            assert routed.payload["semantic_attempts"] == 0
        finally:
            agent.close(); store.close()


def test_contradictory_chat_build_decision_repairs_in_the_same_semantic_turn() -> None:
    prompt = "Create a runnable calculator"
    contradictory = semantic_route(
        "chat",
        original=prompt,
        response="Would you like me to build it?",
        outcome_kind="conversation",
        interpretation="The user requests a complete runnable calculator.",
        task_demand={
            "reasoning": 3,
            "implementation": 4,
            "context_breadth": 3,
            "coordination": 2,
            "verification": 3,
            "visual_runtime": 3,
            "component_count": 3,
            "independently_parallelizable": True,
            "rationale": ["A runnable product must be implemented and verified."],
        },
    )
    with tempfile.TemporaryDirectory() as directory:
        workspace = Path(directory)
        agent, store = runtime(workspace, [
            contradictory,
            semantic_route(
                "goal",
                original=prompt,
                outcome_kind="runnable_product",
                effects=("write", "run"),
            ),
            semantic_goal_intake_turn(semantic_goal_intake(prompt)),
            inspect_call(), plan_call(), plan_pass(),
        ])
        try:
            decision, plan = agent.route_input(prompt)
            assert decision.kind.value == "goal"
            assert plan.goal_id == agent.active_goal().id
            routed = next(
                event for event in store.list_recent_events(limit=100)
                if event.event_type == "semantic_turn.routed"
            )
            assert routed.payload["semantic_attempts"] == 1
            routing_events = [
                event for event in store.list_recent_events(limit=100)
                if event.event_type == "semantic_turn.routing"
            ]
            assert len(routing_events) == 1
            assert [
                message["role"]
                for message in store.list_chat_messages(agent.session_id)
            ] == ["user"]
        finally:
            agent.close(); store.close()


def test_clear_route_cannot_be_stopped_by_nonconsequential_intake_question() -> None:
    prompt = "Create a calculator and run it"
    questioned = semantic_goal_intake(prompt, questions=({
        "id": "style",
        "header": "Style",
        "question": "Which visual style should be used?",
        "reason": "A visual choice is available.",
        "options": [
            {"value": "a", "label": "A", "description": "A", "recommended": True},
            {"value": "b", "label": "B", "description": "B", "recommended": False},
            {"value": "c", "label": "C", "description": "C", "recommended": False},
        ],
    },))
    with tempfile.TemporaryDirectory() as directory:
        workspace = Path(directory)
        agent, store = runtime(workspace, [
            semantic_route(
                "goal", original=prompt, outcome_kind="runnable_product",
                effects=("write", "run"), uncertainty="clear",
            ),
            semantic_goal_intake_turn(questioned),
            semantic_goal_intake_turn(semantic_goal_intake(prompt)),
            inspect_call(), plan_call(), plan_pass(),
        ])
        try:
            decision, _plan = agent.route_input(prompt)
            assert decision.kind.value == "goal"
            accepted = next(
                item for item in store.list_recent_events(limit=100)
                if item.event_type == "semantic_turn.intake_accepted"
            )
            assert accepted.payload["semantic_attempts"] == 1
            assert agent.intake_questions() == ()
        finally:
            agent.close(); store.close()


def test_unadvertised_plan_question_is_rejected_without_spending_stage_budget() -> None:
    prompt = "Create a calculator and run it"
    malformed_question = {
        "tool_calls": [{
            "id": "bad-plan-question",
            "name": "request_plan_input",
            "args": {"questions": [{
                "id": "style",
                "header": "Style",
                "question": "Choose a style",
                "reason": "Visual direction",
                "options": [
                    {"value": "a", "label": "A", "description": "A", "recommended": True},
                    {"value": "b", "label": "B", "description": "B", "recommended": False},
                ],
            }]},
        }],
    }
    with tempfile.TemporaryDirectory() as directory:
        workspace = Path(directory)
        agent, store = runtime(workspace, [
            semantic_route(
                "goal", original=prompt, outcome_kind="runnable_product",
                effects=("write", "run"),
            ),
            semantic_goal_intake_turn(semantic_goal_intake(prompt)),
            inspect_call(), malformed_question, plan_call(), plan_pass(),
        ])
        try:
            decision, plan = agent.route_input(prompt)
            assert decision.kind.value == "goal"
            assert plan.revision == 1
            repairs = [
                item for item in store.list_recent_events(agent.active_goal().id, limit=100)
                if item.event_type == "workflow.retry"
                and item.payload.get("stage") == "plan_questions"
            ]
            rejected = [
                item for item in store.list_recent_events(agent.active_goal().id, limit=100)
                if item.event_type == "tool_contract.rejected"
                and "request_plan_input" in item.payload.get("received", ())
            ]
            assert repairs == []
            assert len(rejected) == 1
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


def test_retyping_exact_pending_request_resumes_same_turn_without_duplicate_message() -> None:
    prompt = "Hello after a temporary outage"
    with tempfile.TemporaryDirectory() as directory:
        workspace = Path(directory)
        agent, store = runtime(workspace, [
            RuntimeError("provider offline"),
            semantic_route("chat", original=prompt, response="Welcome back."),
        ])
        try:
            with pytest.raises(ProviderUnavailableError):
                agent.route_input(prompt)
            turn_id = store.get_workflow_session(agent.session_id)["state"][
                "pending_semantic_turn"
            ]["turn_id"]
            decision, result = agent.route_input(prompt)
            assert decision.kind.value == "chat"
            assert result.message == "Welcome back."
            last = store.get_workflow_session(agent.session_id)["state"]["last_semantic_turn"]
            assert last["turn_id"] == turn_id
            assert len(store.list_chat_messages(agent.session_id)) == 2
            routing = [
                event for event in store.list_recent_events(limit=100)
                if event.event_type == "semantic_turn.routing"
            ]
            assert len(routing) == 1
        finally:
            agent.close(); store.close()


def test_workflow_mode_is_locked_by_pending_semantic_turn_until_cancelled() -> None:
    prompt = "Create a calculator"
    with tempfile.TemporaryDirectory() as directory:
        workspace = Path(directory)
        agent, store = runtime(workspace, [RuntimeError("provider offline")])
        try:
            with pytest.raises(ProviderUnavailableError):
                agent.route_input(prompt)
            assert agent.dashboard().status == "needs_attention"
            assert agent.mode_transition_issue("normal") == ""
            assert "locked" in agent.mode_transition_issue("ultra").casefold()
            with pytest.raises(RuntimeStateError, match="locked"):
                agent.transition_mode("ultra")
            agent.cancel("CANCEL")
            assert agent.mode_transition_issue("ultra") == ""
            assert agent.transition_mode("ultra") == "normal"
            assert agent.interaction_mode.value == "working"
        finally:
            agent.close(); store.close()


def test_pending_route_resumes_in_a_separate_process_without_duplicate_input() -> None:
    prompt = "Hello across processes"
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as directory:
        workspace = Path(directory)
        first_code = f"""
from dataclasses import replace
from pathlib import Path
from agent.config import RuntimeConfig
from agent.runtime import AgentRuntime, ProviderUnavailableError
from agent.store import StateStore
from agent.testing import ScriptedProvider
w = Path({str(workspace)!r})
s = StateStore(w)
c = replace(RuntimeConfig.from_env(), max_provider_retries=0, retry_base_ms=0, repository_index_warmup_files=0)
r = AgentRuntime(ScriptedProvider([RuntimeError('offline')]), s, w, config=c)
try:
    try: r.route_input({prompt!r})
    except ProviderUnavailableError: pass
finally:
    r.close(); s.close()
"""
        first = subprocess.run(
            [sys.executable, "-c", first_code], cwd=root,
            capture_output=True, text=True, timeout=30,
        )
        assert first.returncode == 0, first.stderr

        second_code = f"""
import json
from dataclasses import replace
from pathlib import Path
from agent.config import RuntimeConfig
from agent.runtime import AgentRuntime
from agent.store import StateStore
from agent.testing import ScriptedProvider, semantic_route
w = Path({str(workspace)!r})
s = StateStore(w)
c = replace(RuntimeConfig.from_env(), max_provider_retries=0, retry_base_ms=0, repository_index_warmup_files=0)
r = AgentRuntime(ScriptedProvider([semantic_route('chat', original={prompt!r}, response='Recovered.')]), s, w, config=c)
try:
    result = r.resume()
    messages = s.list_chat_messages(r.session_id)
    print(json.dumps({{'message': result.message, 'user_count': sum(1 for x in messages if x.get('role') == 'user')}}))
finally:
    r.close(); s.close()
"""
        second = subprocess.run(
            [sys.executable, "-c", second_code], cwd=root,
            capture_output=True, text=True, timeout=30,
        )
        assert second.returncode == 0, second.stderr
        recovered = json.loads(second.stdout.strip().splitlines()[-1])
        assert recovered == {"message": "Recovered.", "user_count": 1}


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
            goal = agent.active_goal()
            assert goal.metadata["interaction_mode"] == "plan"
            assert goal.metadata["execution_policy"]["mode"] == "normal"
            assert goal.metadata["execution_strategy"] == "staged"
            session = store.get_workflow_session(agent.session_id)
            assert session["session_mode"] == "normal"
            assert session["state"]["interaction_mode"] == "plan"
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
