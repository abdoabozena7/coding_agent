from __future__ import annotations

import json
import os
import platform
import tempfile
import time
from pathlib import Path
from unittest import mock

import pytest

from agent.intake import normalize_question
from agent.chat_runtime import RequestedEffectV2, SemanticGoalIntakeV3
from agent.local_provider import normalize_generated_tool_payload
from agent.models import GoalStatus
from agent.providers.base import ToolCall
from agent.runtime import AgentRuntime
from agent.semantic import RequestedEffect
from agent.store import StateStore
from agent.testing import ScriptedProvider
from agent.ultra import UltraOrchestrator
from agent.ultra_models import (
    ExecutionClass,
    ResultPackageV1,
    TaskContractV1,
    UltraRun,
    UltraRunStatus,
    WorkNode,
    normalize_contract_path,
)
from agent import tools


def _save_execution_lease(store: StateStore, lease: dict) -> None:
    session = store.get_workflow_session("workspace-session")
    store.save_workflow_session(
        "workspace-session",
        goal_id=session.get("goal_id"),
        session_mode=str(session["session_mode"]),
        plan_state=str(session["plan_state"]),
        run_state=str(session["run_state"]),
        state={**dict(session.get("state") or {}), "execution_lease": lease},
    )


def test_contract_path_drops_only_the_conceptual_workspace_root() -> None:
    assert normalize_contract_path("workspace/src/App.js") == "src/App.js"
    assert normalize_contract_path("WORKSPACE/test/App.test.js") == "test/App.test.js"
    assert normalize_contract_path("src/workspace/App.js") == "src/workspace/App.js"
    with pytest.raises(ValueError):
        normalize_contract_path("workspace/../outside.txt")


def test_runtime_capability_report_contains_the_real_tool_contract() -> None:
    report = {item["name"]: item for item in tools.capability_report()}

    preview = report["preview_html"]
    assert "real browser" in preview["description"]
    assert preview["parameters"]["required"] == ["path"]
    assert set(preview["parameters"]["properties"]) == {
        "path", "open_browser", "verify", "settle_ms", "interactions",
    }
    assert "screenshot_path" in preview["result_contract"]["fields"]
    assert "console_errors" in preview["result_contract"]["fields"]

    process = report["start_process"]
    assert "must remain running" in process["description"]
    assert process["lifecycle"] == "managed"
    assert process["parameters"]["required"] == [
        "command", "readiness_type", "readiness_value",
    ]
    assert "process_id" in process["result_contract"]["fields"]


def test_goal_transitions_keep_the_workflow_session_projection_current() -> None:
    with tempfile.TemporaryDirectory() as directory:
        store = StateStore(directory)
        try:
            goal = store.create_goal("Build and verify the demo")
            session = store.get_workflow_session("workspace-session")
            store.save_workflow_session(
                "workspace-session",
                goal_id=goal.id,
                session_mode=str(session["session_mode"]),
                plan_state="none",
                run_state="idle",
                state=dict(session.get("state") or {}),
            )
            for status in (
                GoalStatus.DISCOVERING,
                GoalStatus.AWAITING_PLAN_APPROVAL,
                GoalStatus.RUNNING,
                GoalStatus.VERIFYING,
                GoalStatus.REVIEWING,
                GoalStatus.COMPLETED,
            ):
                store.transition_goal(goal.id, status)

            completed = store.get_workflow_session("workspace-session")
            assert completed["run_state"] == "completed"
            assert completed["plan_state"] == "approved"
        finally:
            store.close()


def test_semantic_authority_transport_reference_resolves_to_exact_input() -> None:
    original = "create a small calculator web app and run it"
    normalized, receipt = normalize_generated_tool_payload(
        "submit_semantic_route",
        {
            "authority_spans": {
                "write": ["exact_latest_user_input"],
                "run": ["${exact_latest_user_input}"],
            }
        },
        context={"exact_latest_user_input": original},
    )

    assert normalized["authority_spans"] == {
        "write": [original],
        "run": [original],
    }
    assert len(receipt.actions) == 2


def test_goal_only_fills_missing_internal_authority_citation() -> None:
    original = "create a calculator web app and run it"
    normalized, receipt = normalize_generated_tool_payload(
        "submit_semantic_route",
        {
            "route": "goal",
            "requested_effects": ["write", "run", "install", "external"],
            "authority_spans": {"write": [original], "run": [original]},
        },
        context={"exact_latest_user_input": original},
    )

    assert normalized["authority_spans"]["install"] == [original]
    assert "external" not in normalized["authority_spans"]
    assert any("approval-gated Goal" in action for action in receipt.actions)

    action, _receipt = normalize_generated_tool_payload(
        "submit_semantic_route",
        {
            "route": "action",
            "requested_effects": ["install"],
            "authority_spans": {},
        },
        context={"exact_latest_user_input": original},
    )
    assert action["authority_spans"] == {}


def test_repeated_equivalent_previews_do_not_count_as_new_durable_progress() -> None:
    with tempfile.TemporaryDirectory() as directory:
        store = StateStore(directory)
        runtime = AgentRuntime(ScriptedProvider(), store, directory)
        try:
            goal = store.create_goal("Verify one unchanged HTML artifact")

            def record(preview_id: str, screenshot: str) -> None:
                payload = json.dumps({
                    "status": "running",
                    "preview_id": preview_id,
                    "url": f"http://127.0.0.1/{preview_id}/index.html",
                    "http_status": 200,
                    "verification": "passed",
                    "console_errors": [],
                    "page_errors": [],
                    "network_errors": [],
                    "screenshot_path": screenshot,
                })
                action_id = store.begin_action(
                    goal.id,
                    "preview_html",
                    {"arguments": {"path": "index.html"}},
                    risk="high",
                    mutating=False,
                )
                store.complete_action(action_id, payload)
                store.add_evidence(
                    goal_id=goal.id,
                    kind="tool_result",
                    summary="preview_html completed with authoritative harness evidence",
                    data={"tool": "preview_html", "path": "index.html", "result": payload},
                    created_by="harness",
                    verified=True,
                )

            record("preview-one", "one.png")
            first = runtime._durable_progress_snapshot(goal.id)
            record("preview-two", "two.png")
            second = runtime._durable_progress_snapshot(goal.id)

            assert second == first
        finally:
            runtime.close()
            store.close()


def test_runtime_close_releases_its_execution_lease() -> None:
    with tempfile.TemporaryDirectory() as directory:
        store = StateStore(directory)
        runtime = AgentRuntime(ScriptedProvider(), store, directory)
        try:
            runtime._update_execution_lease(stage="working", state="active")
            runtime.close()
            lease = store.get_workflow_session("workspace-session")["state"][
                "execution_lease"
            ]
            assert lease["worker_id"] == runtime._worker_id
            assert lease["lease_state"] == "released"
            assert lease["stage"] == "runtime-closed"
        finally:
            runtime.close()
            store.close()


def test_execution_claim_ignores_legacy_planning_provider_heartbeat() -> None:
    with tempfile.TemporaryDirectory() as directory:
        store = StateStore(directory)
        runtime = AgentRuntime(ScriptedProvider(), store, directory)
        try:
            goal = store.create_goal("Run accepted work")
            _save_execution_lease(
                store,
                {
                    "goal_id": goal.id,
                    "worker_id": "old-planner",
                    "stage": "provider:plan-critic",
                    "action_id": "",
                    "heartbeat_at": time.time(),
                    "expires_at": time.time() + 300,
                    "lease_state": "active",
                },
            )

            assert runtime._claim_execution_lease(goal)
            lease = store.get_workflow_session("workspace-session")["state"][
                "execution_lease"
            ]
            assert lease["worker_id"] == runtime._worker_id
            assert lease["stage"] == "starting"
        finally:
            runtime.close()
            store.close()


def test_versioned_workflow_lease_protects_a_live_planner() -> None:
    with tempfile.TemporaryDirectory() as directory:
        store = StateStore(directory)
        owner = AgentRuntime(ScriptedProvider(), store, directory)
        companion = None
        try:
            _save_execution_lease(
                store,
                {
                    "lease_kind": "workflow",
                    "worker_id": "live-planner",
                    "process_id": os.getpid(),
                    "host": platform.node(),
                    "stage": "provider:master-plan",
                    "action_id": "",
                    "heartbeat_at": time.time(),
                    "expires_at": time.time() + 300,
                    "lease_state": "active",
                },
            )

            companion = AgentRuntime(
                ScriptedProvider(),
                store,
                directory,
                session_id=owner.session_id,
            )

            assert companion._foreign_execution_owner_live
            companion.close()
            lease = store.get_workflow_session(owner.session_id)["state"][
                "execution_lease"
            ]
            assert lease["worker_id"] == "live-planner"
            assert lease["lease_state"] == "active"
        finally:
            if companion is not None:
                companion.close()
            owner.close()
            store.close()


def test_execution_claim_respects_a_live_execution_process() -> None:
    with tempfile.TemporaryDirectory() as directory:
        store = StateStore(directory)
        runtime = AgentRuntime(ScriptedProvider(), store, directory)
        try:
            goal = store.create_goal("Run accepted work")
            _save_execution_lease(
                store,
                {
                    "goal_id": goal.id,
                    "worker_id": "other-worker",
                    "process_id": os.getpid(),
                    "host": platform.node(),
                    "stage": "working",
                    "action_id": "action-1",
                    "heartbeat_at": time.time() - 600,
                    "expires_at": time.time() - 300,
                    "lease_state": "active",
                },
            )

            assert not runtime._claim_execution_lease(goal)
        finally:
            runtime.close()
            store.close()


def test_read_only_companion_runtime_does_not_recover_a_live_worker() -> None:
    with tempfile.TemporaryDirectory() as directory:
        store = StateStore(directory)
        owner = AgentRuntime(ScriptedProvider(), store, directory)
        companion = None
        try:
            goal = store.create_goal(
                "Keep the live worker running",
                session_id=owner.session_id,
            )
            action_id = store.begin_action(
                goal.id,
                "write_file",
                {"path": "result.txt", "content": "working"},
                risk="medium",
                mutating=True,
            )
            _save_execution_lease(
                store,
                {
                    "goal_id": goal.id,
                    "worker_id": "live-owner",
                    "process_id": os.getpid(),
                    "host": platform.node(),
                    "stage": "tool:write_file",
                    "action_id": action_id,
                    "heartbeat_at": time.time(),
                    "expires_at": time.time() + 300,
                    "lease_state": "active",
                },
            )

            companion = AgentRuntime(
                ScriptedProvider(),
                store,
                directory,
                session_id=owner.session_id,
            )

            assert companion._foreign_execution_owner_live
            assert store.list_actions(goal.id, status="running")[0]["id"] == action_id
            assert store.get_goal(goal.id).status == goal.status
            companion.close()
            lease = store.get_workflow_session(owner.session_id)["state"][
                "execution_lease"
            ]
            assert lease["worker_id"] == "live-owner"
            assert lease["lease_state"] == "active"
        finally:
            if companion is not None:
                companion.close()
            if store.list_actions(goal.id, status="running"):
                store.complete_action(action_id, "test cleanup")
            owner.close()
            store.close()


def test_restart_reconciles_terminal_ultra_failure_with_stale_running_goal() -> None:
    with tempfile.TemporaryDirectory() as directory:
        store = StateStore(directory)
        first = AgentRuntime(ScriptedProvider(), store, directory)
        restored = None
        try:
            goal = store.create_goal(
                "Restore the failed recursive checkpoint",
                session_id=first.session_id,
            )
            store.transition_goal(goal.id, GoalStatus.AWAITING_PLAN_APPROVAL)
            store.transition_goal(goal.id, GoalStatus.RUNNING)
            run = store.create_ultra_run(
                UltraRun(
                    goal_id=goal.id,
                    provider="test",
                    model="weak-local",
                    execution_class=ExecutionClass.LOCAL,
                    concurrency=1,
                )
            )
            store.update_goal_metadata(goal.id, ultra_run_id=run.id)
            store.update_ultra_run(
                run.id,
                status=UltraRunStatus.BLOCKED,
                error="tester returned a malformed typed verdict",
            )
            first.close()

            restored = AgentRuntime(
                ScriptedProvider(),
                store,
                directory,
                session_id=first.session_id,
            )

            reconciled = store.get_goal(goal.id)
            assert reconciled.status is GoalStatus.BLOCKED
            assert reconciled.metadata["resume_action"] == "Retry"
            assert reconciled.metadata["waiting_on"] == "recovery"
            assert reconciled.metadata["waiting_question"] == ""
            assert reconciled.metadata["auto_retryable"] is True
            assert reconciled.metadata["terminal_ultra_failure"] == {
                "run_id": run.id,
                "status": UltraRunStatus.BLOCKED.value,
                "diagnostic": "tester returned a malformed typed verdict",
                "mutation_replayed": False,
            }
            assert store.get_ultra_run(run.id).status is UltraRunStatus.BLOCKED
        finally:
            if restored is not None:
                restored.close()
            else:
                first.close()
            store.close()


def test_restart_reconciles_revision_required_ultra_with_stale_running_goal() -> None:
    with tempfile.TemporaryDirectory() as directory:
        store = StateStore(directory)
        first = AgentRuntime(ScriptedProvider(), store, directory)
        restored = None
        try:
            goal = store.create_goal(
                "Restore the stopped quality revision",
                session_id=first.session_id,
            )
            store.transition_goal(goal.id, GoalStatus.AWAITING_PLAN_APPROVAL)
            store.transition_goal(goal.id, GoalStatus.RUNNING)
            run = store.create_ultra_run(
                UltraRun(
                    goal_id=goal.id,
                    provider="test",
                    model="weak-local",
                    execution_class=ExecutionClass.LOCAL,
                    concurrency=1,
                )
            )
            store.update_goal_metadata(goal.id, ultra_run_id=run.id)
            store.update_ultra_run(
                run.id,
                status=UltraRunStatus.REVISION_REQUIRED,
                error="browser runtime failed: THREE is not defined",
            )
            first.close()

            restored = AgentRuntime(
                ScriptedProvider(),
                store,
                directory,
                session_id=first.session_id,
            )

            reconciled = store.get_goal(goal.id)
            assert reconciled.status is GoalStatus.PAUSED
            assert reconciled.metadata["resume_action"] == "ultra_replan"
            assert reconciled.metadata["waiting_on"] == "diagnosis"
            assert "THREE is not defined" in reconciled.metadata["replan_feedback"]
            assert reconciled.metadata["terminal_ultra_failure"] == {
                "run_id": run.id,
                "status": UltraRunStatus.REVISION_REQUIRED.value,
                "diagnostic": "browser runtime failed: THREE is not defined",
                "mutation_replayed": False,
            }
        finally:
            if restored is not None:
                restored.close()
            else:
                first.close()
            store.close()


def test_restart_candidate_error_overrides_stale_harness_contract_label() -> None:
    with tempfile.TemporaryDirectory() as directory:
        store = StateStore(directory)
        first = AgentRuntime(ScriptedProvider(), store, directory)
        restored = None
        try:
            goal = store.create_goal(
                "Repair the broken browser application",
                session_id=first.session_id,
            )
            store.transition_goal(goal.id, GoalStatus.AWAITING_PLAN_APPROVAL)
            store.transition_goal(goal.id, GoalStatus.RUNNING)
            run = store.create_ultra_run(
                UltraRun(
                    goal_id=goal.id,
                    provider="test",
                    model="weak-local",
                    execution_class=ExecutionClass.LOCAL,
                    concurrency=1,
                )
            )
            store.update_goal_metadata(goal.id, ultra_run_id=run.id)
            store.create_work_node(
                WorkNode(
                    ultra_run_id=run.id,
                    title="Browser integration",
                    objective="Run the interactive application.",
                    contract=TaskContractV1(
                        objective="Run the interactive application.",
                        success_criteria=("The application starts without runtime errors.",),
                        write_paths=("index.html",),
                    ),
                    result=ResultPackageV1(
                        summary="Browser verification failed.",
                        issues=("THREE is not defined", "Required DOM elements were not found"),
                        metadata={
                            "component_package": {
                                "failure_diagnostic": {
                                    "failure_kind": "contract",
                                    "blocker_owner": "test_harness",
                                    "mutation_prohibited": True,
                                    "failure_fingerprint": "c" * 64,
                                }
                            }
                        },
                    ),
                )
            )
            store.update_ultra_run(
                run.id,
                status=UltraRunStatus.REVISION_REQUIRED,
                error="Browser quality gate failed.",
            )
            first.close()

            restored = AgentRuntime(
                ScriptedProvider(),
                store,
                directory,
                session_id=first.session_id,
            )

            reconciled = store.get_goal(goal.id)
            assert reconciled.status is GoalStatus.PAUSED
            assert reconciled.metadata["resume_action"] == "ultra_replan"
            assert reconciled.metadata["auto_retryable"] is True
            assert "THREE is not defined" in reconciled.metadata["replan_feedback"]
            assert "verification_blocker" not in reconciled.metadata
        finally:
            if restored is not None:
                restored.close()
            else:
                first.close()
            store.close()


@pytest.mark.parametrize(
    "options",
    [
        ["Use the existing scope (recommended)", "Expand the scope"],
        [
            {"label": "Keep scope", "value": "keep", "recommended": True},
            "Expand scope",
            {"label": "Cancel", "value": "cancel"},
        ],
    ],
)
def test_question_transport_normalizes_string_object_and_mixed_options(options) -> None:
    value = normalize_question(
        {
            "question": "Should this irreversible external change expand scope?",
            "options": options,
            "allow_free_form": True,
            "decision_need": {
                "impact": "Changes an external deployment boundary",
                "affected_scope": ["deployment"],
                "affected_effects": ["external_side_effect"],
                "reversible": False,
                "requires_user_authority": True,
                "reason": "The exact request does not authorize deployment.",
                "evidence_refs": [],
            },
        }
    )

    assert len(value.options) in {2, 3}
    assert len({item.value for item in value.options}) == len(value.options)
    assert value.options[0].recommended
    assert value.allow_freeform
    assert value.decision_need is not None
    assert value.decision_need.blocks_work


def test_ultra_drops_question_without_user_authority_proof() -> None:
    optional = {
        "id": "design-depth",
        "question": "What level of visual detail should the calculator use?",
        "options": ["Minimal", "Polished"],
        "allow_freeform": True,
    }
    assert UltraOrchestrator._validated_questions([optional]) == ()

    authority = {
        **optional,
        "id": "external-publish",
        "question": "May the workflow publish the result to the external service?",
        "decision_need": {
            "impact": "Publishes data outside the local workspace.",
            "affected_scope": ["deployment"],
            "affected_effects": ["external_side_effect"],
            "reversible": False,
            "requires_user_authority": True,
            "reason": "The original request did not authorize publication.",
            "evidence_refs": ["exact-user-request"],
        },
    }
    accepted = UltraOrchestrator._validated_questions([authority])
    assert len(accepted) == 1
    assert accepted[0]["decision_need"]["requires_user_authority"] is True


def test_goal_intake_rebinds_outer_model_authored_task_demand_facts() -> None:
    intake = SemanticGoalIntakeV3.from_mapping(
        {
            "objective": "Build the accepted app.",
            "deliverables": ["Runnable app"],
            "constraints": [],
            "exclusions": [],
            "acceptance_expectations": ["The app runs"],
            "assumptions": [],
            "risks": [],
            "component_count": 5,
            "parallelism_required": True,
            "coordination_summary": "Coordinate the model-authored components.",
            "uncertainty": "clear",
            "complexity_reasons": ["Creating and running the app is complex."],
            "task_demand": {
                "reasoning": 3,
                "implementation": 4,
                "context_breadth": 2,
                "coordination": 1,
                "verification": 4,
                "visual_runtime": 4,
            },
        }
    )

    assert intake.task_demand.component_count == 5
    assert intake.task_demand.independently_parallelizable is True
    assert intake.task_demand.rationale == (
        "Creating and running the app is complex.",
    )


def test_one_option_is_a_real_malformed_question() -> None:
    with pytest.raises(ValueError, match="two or three"):
        normalize_question(
            {"question": "Choose", "options": ["Only option"]}
        )


def test_dependency_lockfile_is_a_declared_audited_mutation_footprint() -> None:
    with tempfile.TemporaryDirectory() as directory:
        workspace = Path(directory)
        (workspace / "package.json").write_text(
            '{"dependencies":{"three":"1.0.0"}}', encoding="utf-8"
        )
        store = StateStore(workspace)
        runtime = AgentRuntime(
            ScriptedProvider(), store, workspace, approval=lambda *_: True
        )
        try:
            goal = store.create_goal("Install the accepted project dependencies")
            store.update_goal_metadata(
                goal.id,
                resource_claims=[
                    {
                        "supports_tasks": ["T001"],
                        "resolved_paths": ["package.json"],
                        "state": "resolved",
                    }
                ],
            )
            goal = store.get_goal(goal.id)

            def install(_name, _args):
                (workspace / "package-lock.json").write_text(
                    '{"lockfileVersion":3}', encoding="utf-8"
                )
                return json.dumps(
                    {"status": "installed", "manager": "npm", "exit_code": 0}
                )

            with mock.patch("agent.runtime.tools.run_tool", side_effect=install):
                result = runtime._execute_workspace_tool(
                    goal,
                    ToolCall(
                        id="install",
                        name="install_dependencies",
                        args={"directory": ".", "manager": "auto"},
                    ),
                    task_id="T001",
                    actor="planner",
                )

            assert not result.startswith("Error:")
            assert store.list_actions(goal.id)[-1]["status"] == "completed"
            footprint_events = [
                event
                for event in store.list_recent_events(goal.id, limit=100)
                if event.event_type == "mutation.footprint_derived"
            ]
            assert footprint_events[-1].payload["derived_paths"] == [
                "package-lock.json"
            ]
        finally:
            runtime.close()
            store.close()


def test_legacy_successful_lockfile_uncertainty_reconciles_without_replay() -> None:
    with tempfile.TemporaryDirectory() as directory:
        workspace = Path(directory)
        store = StateStore(workspace)
        runtime = AgentRuntime(ScriptedProvider(), store, workspace)
        try:
            goal = store.create_goal("Install dependencies")
            action_id = store.begin_action(
                goal.id,
                "install_dependencies",
                {"arguments": {"directory": ".", "manager": "auto"}},
                task_id="T001",
                risk="critical",
                mutating=True,
            )
            store.complete_action(
                action_id,
                json.dumps({"status": "installed", "exit_code": 0}),
            )
            store.mark_action_uncertain(
                action_id, "mutation escaped accepted resource leases"
            )
            store.update_goal_metadata(
                goal.id,
                uncertain_actions=[
                    {"action_id": action_id, "paths": ["package-lock.json"]}
                ],
                waiting_question="Inspect uncertain mutation",
            )

            reconciled = runtime._auto_reconcile_declared_tool_side_effects(
                store.get_goal(goal.id)
            )

            assert reconciled == (action_id,)
            assert store.list_actions(goal.id)[-1]["status"] == "resolved_applied"
            assert store.get_goal(goal.id).metadata["uncertain_actions"] == []
            event = next(
                item
                for item in store.list_recent_events(goal.id, limit=100)
                if item.event_type == "execution.reconciled"
            )
            assert event.payload["mutation_replayed"] is False
        finally:
            runtime.close()
            store.close()


def test_unknown_source_mutation_never_inherits_a_lockfile_footprint() -> None:
    footprint = tools.mutation_footprint(
        "install_dependencies",
        {"directory": ".", "manager": "npm"},
        ["package.json"],
    )
    assert "package-lock.json" in footprint.derived_paths
    assert "src/app.js" not in footprint.effective_paths


def test_shell_effect_aliases_are_mechanical_capability_equivalents() -> None:
    assert RequestedEffect.parse("execute_shell") is RequestedEffect.EXECUTE_CODE
    assert RequestedEffect.parse("run_shell") is RequestedEffect.EXECUTE_CODE
    assert RequestedEffectV2.parse("execute_shell") is RequestedEffectV2.RUN
    assert RequestedEffectV2.parse("shell") is RequestedEffectV2.RUN

    normalized, receipt = normalize_generated_tool_payload(
        "propose_semantic_goal",
        {
            "requested_effects": [
                "read_file", "write_file", "preview_html", "write_file"
            ]
        },
    )
    assert normalized["requested_effects"] == [
        "read_workspace", "mutate_workspace", "execute_code"
    ]
    assert receipt.actions


def test_plan_change_strips_only_harness_owned_lifecycle_fields() -> None:
    task = {
        "title": "Verify the accepted artifact",
        "description": "Run the accepted verification and repair concrete failures.",
        "acceptance_criteria": ["The verification exits successfully."],
        "verification": ["Run the accepted test command."],
        "depends_on": ["T001"],
        "risk": "medium",
        "status": "in_progress",
        "attempts": 3,
        "evidence": ["model-authored lifecycle claim"],
    }
    normalized, receipt = normalize_generated_tool_payload(
        "propose_plan_change",
        {"reason": "Verification exposed additional accepted-scope work.", "tasks": [task]},
    )

    clean = normalized["tasks"][0]
    assert clean == {
        key: value
        for key, value in task.items()
        if key not in {"status", "attempts", "evidence"}
    }
    assert any("/status removed" in action for action in receipt.actions)
    assert any("/attempts removed" in action for action in receipt.actions)
    assert any("/evidence removed" in action for action in receipt.actions)


def test_codex_add_file_patch_transport_is_scoped_and_applied() -> None:
    patch = """*** Begin Patch
*** Add File: src/example.txt
+first line
+second line
*** End Patch"""
    normalized, receipt = normalize_generated_tool_payload(
        "apply_patch", {"patch": patch, "base_path": ""}
    )
    assert normalized["base_path"] == "."
    assert any("base_path" in action for action in receipt.actions)
    assert tools.apply_patch.patch_paths(patch) == ("src/example.txt",)

    with tempfile.TemporaryDirectory() as directory:
        with tools.workspace_context(directory):
            result = tools.run_tool("apply_patch", normalized)
        assert result.startswith("Applied patch to 1 file")
        assert (Path(directory) / "src" / "example.txt").read_text(
            encoding="utf-8"
        ) == "first line\nsecond line\n"


@pytest.mark.skipif(os.name != "nt", reason="Windows shell contract")
def test_windows_shell_rejects_posix_heredoc_with_actionable_repair() -> None:
    from agent.tools.run_bash import run_with_timeout

    result = run_with_timeout("python - <<'PY'\nprint('x')\nPY")

    assert "POSIX heredoc syntax is not supported" in result
    assert "python -c" in result


def test_empty_workspace_accepts_model_selected_new_layout_provenance() -> None:
    with tempfile.TemporaryDirectory() as directory:
        store = StateStore(directory)
        runtime = AgentRuntime(ScriptedProvider(), store, directory)
        try:
            goal = store.create_goal("Create and run the requested application")
            task = store.coerce_task(
                {
                    "id": "T001",
                    "title": "Create the application",
                    "description": "Create the accepted runnable application.",
                    "acceptance_criteria": ["The application runs."],
                    "verification": ["Launch and exercise the application."],
                    "depends_on": [],
                    "risk": "medium",
                },
                goal.id,
                1,
                "agent",
            )
            runtime._validate_plan_applicability(
                {
                    "semantic_goal": {},
                    "applicability_evidence": [
                        {
                            "fact": "The workspace root was inspected and is empty.",
                            "source": "inspection:I001",
                            "supports_tasks": ["T001"],
                        }
                    ],
                    "expected_changes": [
                        {
                            "path": "index.html",
                            "intent": "Create the model-selected entry point.",
                            "basis": "model_selected_new_layout",
                            "evidence_refs": ["inspection:I001"],
                            "supports_tasks": ["T001"],
                        }
                    ],
                },
                (task,),
                successful_inspection_ids=frozenset({"I001"}),
                original_request="Create and run the requested application",
            )
        finally:
            runtime.close()
            store.close()


def test_fresh_ultra_foundation_plan_is_not_reclassified_as_legacy() -> None:
    with tempfile.TemporaryDirectory() as directory:
        store = StateStore(directory)
        runtime = AgentRuntime(ScriptedProvider(), store, directory)
        goal = store.create_goal("Create a calculator application")
        run_id = "ultra_fresh_foundation"
        store.update_goal_metadata(
            goal.id,
            ultra_run_id=run_id,
            execution_strategy="recursive",
        )
        store.transition_goal(goal.id, GoalStatus.DISCOVERING)
        plan = store.create_plan(
            goal.id,
            "Create and verify the calculator",
            [
                {
                    "id": "T001",
                    "title": "Create calculator",
                    "description": "Create the approved calculator application.",
                    "acceptance_criteria": ["The calculator works."],
                    "verification": ["preview_html index.html"],
                    "depends_on": [],
                    "risk": "medium",
                }
            ],
            applicability_evidence=[
                {
                    "fact": "ULTRA foundation was built after inspection.",
                    "source": f"ultra:{run_id}:foundation",
                    "supports_tasks": ["T001"],
                }
            ],
            execution_strategy="Execute the recursive approved module.",
            expected_changes=[
                {
                    "path": "index.html",
                    "intent": "Create the calculator.",
                    "supports_tasks": ["T001"],
                }
            ],
        )
        store.transition_goal(goal.id, GoalStatus.AWAITING_PLAN_APPROVAL)
        store.approve_plan(goal.id, plan.revision)
        runtime.close()
        store.close()

        restored_store = StateStore(directory)
        restored = AgentRuntime(ScriptedProvider(), restored_store, directory)
        try:
            assert restored_store.get_goal(goal.id).status is GoalStatus.RUNNING
        finally:
            restored.close()
            restored_store.close()


def _accepted_two_task_runtime(directory: str):
    store = StateStore(directory)
    runtime = AgentRuntime(ScriptedProvider(), store, directory)
    goal = store.create_goal("Create two accepted files")
    store.update_goal_metadata(goal.id, execution_strategy="staged")
    store.transition_goal(goal.id, GoalStatus.DISCOVERING)
    tasks = [
        {
            "id": "T001",
            "title": "Create first file",
            "description": "Create first.txt.",
            "acceptance_criteria": ["first.txt exists."],
            "verification": ["Read first.txt."],
            "depends_on": [],
            "risk": "low",
        },
        {
            "id": "T002",
            "title": "Create second file",
            "description": "Create second.txt.",
            "acceptance_criteria": ["second.txt exists."],
            "verification": ["Read second.txt."],
            "depends_on": ["T001"],
            "risk": "low",
        },
    ]
    plan = store.create_plan(
        goal.id,
        "Create both files",
        tasks,
        applicability_evidence=[
            {
                "fact": "The workspace was inspected.",
                "source": "inspection:I001",
                "supports_tasks": ["T001", "T002"],
            }
        ],
        execution_strategy="Create first.txt, then second.txt.",
        expected_changes=[
            {
                "path": "first.txt",
                "intent": "Create first file.",
                "supports_tasks": ["T001"],
            },
            {
                "path": "second.txt",
                "intent": "Create second file.",
                "supports_tasks": ["T002"],
            },
        ],
    )
    store.transition_goal(goal.id, GoalStatus.AWAITING_PLAN_APPROVAL)
    runtime.approve_plan(plan.revision)
    store.update_goal_metadata(goal.id, execution_strategy="staged")
    return runtime, store, goal.id


def test_staged_coordinator_preserves_task_scoped_resource_ownership() -> None:
    with tempfile.TemporaryDirectory() as directory:
        runtime, store, goal_id = _accepted_two_task_runtime(directory)
        try:
            result = runtime._execute_workspace_tool(
                store.get_goal(goal_id),
                ToolCall(
                    id="future-scope-write",
                    name="write_file",
                    args={"path": "second.txt", "content": "accepted\n"},
                ),
                task_id="T001",
                actor="coordinator",
            )
            assert "not covered" in result
            assert not (Path(directory) / "second.txt").exists()
        finally:
            runtime.close()
            store.close()


def test_apply_patch_absolute_workspace_base_is_normalized_before_scope_checks() -> None:
    with tempfile.TemporaryDirectory() as directory:
        runtime, store, goal_id = _accepted_two_task_runtime(directory)
        try:
            result = runtime._execute_workspace_tool(
                store.get_goal(goal_id),
                ToolCall(
                    id="absolute-base-patch",
                    name="apply_patch",
                    args={
                        "base_path": str(Path(directory).resolve()),
                        "patch": """*** Begin Patch
*** Add File: first.txt
+accepted
*** End Patch""",
                    },
                ),
                task_id="T001",
                actor="coordinator",
            )
            assert result.startswith("Applied patch")
            assert (Path(directory) / "first.txt").read_text(
                encoding="utf-8"
            ) == "accepted\n"
            normalized = [
                event for event in store.list_recent_events(goal_id, limit=100)
                if event.event_type == "tool_payload.normalized"
            ]
            assert any("absolute in-workspace" in " ".join(event.payload["actions"]) for event in normalized)
        finally:
            runtime.close()
            store.close()


def test_absolute_in_workspace_tool_paths_are_normalized_generically() -> None:
    with tempfile.TemporaryDirectory() as directory:
        runtime, store, goal_id = _accepted_two_task_runtime(directory)
        try:
            target = Path(directory) / "first.txt"
            target.write_text("read me\n", encoding="utf-8")
            result = runtime._execute_workspace_tool(
                store.get_goal(goal_id),
                ToolCall(
                    id="absolute-read",
                    name="read_file",
                    args={"path": str(target.resolve())},
                ),
                task_id="T001",
                actor="coordinator",
            )
            assert result.splitlines() == ["read me"]
            action = store.list_actions(goal_id)[-1]
            assert json.loads(action["args_json"])["arguments"]["path"] == "first.txt"
        finally:
            runtime.close()
            store.close()


def test_malformed_tool_arguments_are_rejected_before_permission_prompt() -> None:
    with tempfile.TemporaryDirectory() as directory:
        runtime, store, goal_id = _accepted_two_task_runtime(directory)
        approvals: list[tuple[str, dict, str]] = []
        runtime.approval = lambda name, args, risk: approvals.append(
            (name, args, risk)
        ) or True
        try:
            Path(directory, "index.html").write_text(
                "<!doctype html><title>safe</title>", encoding="utf-8"
            )
            before = len(store.list_actions(goal_id))
            result = runtime._execute_workspace_tool(
                store.get_goal(goal_id),
                ToolCall(
                    id="invalid-preview",
                    name="preview_html",
                    args={
                        "path": "index.html",
                        "capture_console_output": True,
                    },
                ),
                task_id="T001",
                actor="coordinator",
            )
            assert "unknown argument" in result
            assert approvals == []
            assert len(store.list_actions(goal_id)) == before
            rejected = next(
                event for event in store.list_recent_events(goal_id, limit=100)
                if event.event_type == "tool_contract.rejected"
            )
            assert rejected.payload["approval_requested"] is False
        finally:
            runtime.close()
            store.close()


def test_inapplicable_preview_is_rejected_before_permission_prompt() -> None:
    with tempfile.TemporaryDirectory() as directory:
        runtime, store, goal_id = _accepted_two_task_runtime(directory)
        approvals: list[tuple[str, dict, str]] = []
        runtime.approval = lambda name, args, risk: approvals.append(
            (name, args, risk)
        ) or True
        try:
            Path(directory, "component.js").write_text(
                "export const value = 1;", encoding="utf-8"
            )
            before = len(store.list_actions(goal_id))
            result = runtime._execute_workspace_tool(
                store.get_goal(goal_id),
                ToolCall(
                    id="wrong-preview-kind",
                    name="preview_html",
                    args={"path": "component.js", "open_browser": False},
                ),
                task_id="T001",
                actor="coordinator",
            )
            assert "requires an existing .html or .htm file" in result
            assert approvals == []
            assert len(store.list_actions(goal_id)) == before
        finally:
            runtime.close()
            store.close()


def test_plan_change_task_without_optional_lifecycle_fields_does_not_crash() -> None:
    with tempfile.TemporaryDirectory() as directory:
        runtime, store, _goal_id = _accepted_two_task_runtime(directory)
        try:
            revision = runtime.revise_plan(
                reason="Add accepted-scope verification work.",
                add=[
                    {
                        "title": "Verify both files",
                        "description": "Read both accepted files and record evidence.",
                        "acceptance_criteria": ["Both files are readable."],
                        "verification": ["Read first.txt and second.txt."],
                    }
                ],
                proposed_by="coordinator",
                inherit_approved_scope=True,
            )
            added = revision.tasks[-1]
            assert added.depends_on == ()
            assert added.risk == "medium"
        finally:
            runtime.close()
            store.close()
