from __future__ import annotations

import tempfile
import unittest
import io
import hashlib
import json
import subprocess
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from agent.config import RuntimeConfig, update_runtime_config
from agent.cli import _run_auto
from agent.hardware import HardwareProbeResult, probe_local_gpu
from agent.model_catalog import ExecutionClass, ModelDescriptor
from agent.models import DelegationStatus, GoalStatus, PlanStatus, TaskStatus
from agent.providers.base import ToolCall
from agent.runtime import (
    AgentRuntime,
    RuntimeStateError,
    SliceResult,
    _extract_explicit_workspace_paths,
)
from agent.sandbox import DockerSandbox, PermissionAdapter
from agent.store import StateStore
from agent.testing import ScriptedProvider, semantic_goal_intake, semantic_turn
from agent.local_provider import ModelCapabilityProfile
from agent import tools as agent_tools
from agent.ui import ConsoleUI


def plan_call():
    return {
        "tool_calls": [
            {
                "id": "plan",
                "name": "propose_plan",
                "args": {
                    "semantic_goal": {
                        "original_request": "",
                        "interpreted_outcome": "Implement and independently verify the requested durable behavior.",
                        "requested_effects": [
                            "read_workspace", "mutate_workspace", "execute_code"
                        ],
                        "required_outcomes": [
                            "The requested durable behavior is implemented and verified."
                        ],
                        "constraints": ["Preserve unrelated workspace state."],
                        "exclusions": [],
                        "acceptance_criteria": [
                            "The requested behavior survives restart and tests pass."
                        ],
                        "unresolved_decisions": [],
                        "repository_evidence_refs": [
                            "inspection:I001", "user:request"
                        ],
                    },
                    "summary": "Implement and independently verify durable behavior.",
                    "applicability_evidence": [
                        {
                            "fact": "The workspace was inspected and requires the requested durable behavior.",
                            "source": "tool:inspect-workspace",
                            "supports_tasks": ["T001"],
                        }
                    ],
                    "execution_strategy": "Inspect the implementation, edit the workspace, and run focused offline verification.",
                    "expected_changes": [
                        {
                            "path": "artifact.txt",
                            "intent": "Implement the durable behavior and its verification support.",
                            "basis": "repository_convention",
                            "evidence_refs": ["inspection:I001"],
                            "supports_tasks": ["T001"],
                        }
                    ],
                    "tasks": [
                        {
                            "id": "T001",
                            "title": "Implement durable behavior",
                            "description": "Make the requested durable change and cover edge cases.",
                            "acceptance_criteria": ["The requested behavior survives restart and tests pass."],
                            "verification": ["Run the focused offline tests and inspect the final state."],
                            "depends_on": [],
                            "risk": "high",
                        }
                    ],
                },
            },
        ]
    }


def inspect_call():
    return {"tool_calls": [{"id": "inspect-workspace", "name": "list_files", "args": {}}]}


def invalid_plan_call(index: int):
    return {
        "tool_calls": [
            {
                "id": f"invalid-plan-{index}",
                "name": "propose_plan",
                "args": {
                    "summary": "Attempted structured plan.",
                    "applicability_evidence": [{}],
                    "execution_strategy": "Inspect, implement, and verify the requested behavior.",
                    "expected_changes": [{}],
                    "tasks": [{}],
                },
            }
        ]
    }


def invalid_evidence_plan_call(index: int):
    value = plan_call()
    value["tool_calls"][0]["id"] = f"invalid-evidence-{index}"
    value["tool_calls"][0]["args"]["applicability_evidence"][0]["source"] = (
        "tool:missing-inspection"
    )
    return value


def plan_pass():
    return {
        "tool_calls": [
            {
                "id": "critic",
                "name": "submit_plan_review",
                "args": {"verdict": "pass", "summary": "Plan is complete and verifiable.", "issues": []},
            }
        ]
    }


def dependency_plan_call():
    first = plan_call()["tool_calls"][0]["args"]["tasks"][0]
    second = {
        "id": "T002",
        "title": "Integrate durable behavior",
        "description": "Integrate and verify the implementation from T001.",
        "acceptance_criteria": ["Integration behavior is directly evidenced."],
        "verification": ["Run the integration test."],
        "depends_on": ["T001"],
        "risk": "medium",
    }
    value = {
        "tool_calls": [
            {
                "id": "plan-deps",
                "name": "propose_plan",
                "args": {
                    "semantic_goal": {
                        **plan_call()["tool_calls"][0]["args"]["semantic_goal"],
                        "acceptance_criteria": [
                            "The requested behavior survives restart and tests pass.",
                            "Integration behavior is directly evidenced.",
                        ],
                    },
                    "summary": "Implement then integrate.",
                    "applicability_evidence": [
                        {
                            "fact": "The inspected workspace needs implementation and integration coverage.",
                            "source": "tool:inspect-workspace",
                            "supports_tasks": ["T001", "T002"],
                        }
                    ],
                    "execution_strategy": "Edit the implementation first, then integrate it and run both focused checks.",
                    "expected_changes": [
                        {
                            "path": "artifact.txt",
                            "intent": "Implement and integrate the durable behavior.",
                            "basis": "repository_convention",
                            "evidence_refs": ["inspection:I001"],
                            "supports_tasks": ["T001", "T002"],
                        }
                    ],
                    "tasks": [first, second],
                },
            },
        ]
    }
    return value


def task_update(status, evidence=(), note=""):
    return {
        "id": f"update-{status}",
        "name": "update_task",
        "args": {
            "task_id": "T001",
            "status": status,
            "note": note,
            "evidence": list(evidence),
        },
    }


def finish_call():
    return {
        "id": "finish",
        "name": "finish_goal",
        "args": {"summary": "Durable behavior implemented and verified.", "evidence": ["offline tests passed"]},
    }


def review_pass():
    return {
        "tool_calls": [
            {
                "id": "review",
                "name": "submit_review",
                "args": {
                    "verdict": "pass",
                    "summary": "Objective and criteria are directly evidenced.",
                    "issues": [],
                    "checked_task_ids": ["T001"],
                },
            }
        ]
    }


def stored_plan_basis(*task_ids: str):
    ids = list(task_ids)
    return {
        "applicability_evidence": [
            {
                "fact": "The inspected workspace requires the recovery task.",
                "source": "test workspace",
                "supports_tasks": ids,
            }
        ],
        "execution_strategy": "Apply the workspace change and inspect the resulting durable state.",
        "expected_changes": [
            {
                "path": "workspace/",
                "intent": "Create or verify the artifact required by the recovery task.",
                "supports_tasks": ids,
            }
        ],
    }


class RuntimeTestCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp.name)
        self.store = StateStore(self.workspace)
        self.config = RuntimeConfig(
            planning_steps=6,
            work_quantum_steps=8,
            review_steps=4,
            subagent_steps=4,
            max_delegation_depth=3,
            max_delegations_per_slice=4,
            max_provider_retries=0,
            repeated_action_limit=2,
            no_action_limit=2,
            conversation_chars=50_000,
            retry_base_ms=0,
        )

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def runtime(self, turns):
        provider = ScriptedProvider(turns)
        runtime = AgentRuntime(
            provider,
            self.store,
            self.workspace,
            config=self.config,
            sleeper=lambda _seconds: None,
            approval=lambda _name, _args, _risk: True,
        )
        return runtime, provider


class SleepModeTests(RuntimeTestCase):
    def test_runtime_recreates_a_missing_session_envelope_from_authority(self):
        runtime = AgentRuntime(
            ScriptedProvider([], model="local-test"),
            self.store,
            self.workspace,
            config=self.config,
            session_id="session-envelope-race",
        )
        try:
            with self.store.transaction() as connection:
                connection.execute(
                    "DELETE FROM workflow_sessions WHERE id=?",
                    (runtime.session_id,),
                )
            snapshot = runtime.workflow_runtime_snapshot()
            self.assertEqual(snapshot.phase, "ready")
            recreated = self.store.get_workflow_session(runtime.session_id)
            self.assertEqual(recreated["id"], runtime.session_id)
            self.assertTrue(
                any(
                    event.event_type == "workflow.session_recreated"
                    for event in self.store.list_events()
                )
            )
        finally:
            runtime.close()

    def test_active_goal_recursive_strategy_overrides_stale_staged_session_label(self):
        runtime = AgentRuntime(
            ScriptedProvider([], model="local-test"),
            self.store,
            self.workspace,
            config=self.config,
        )
        goal = self.store.create_goal(
            "Execute through recursive nodes",
            session_id=runtime.session_id,
        )
        self.store.update_goal_metadata(goal.id, execution_strategy="recursive")
        self.store.mutate_workflow_session(
            runtime.session_id,
            lambda current: {
                **current,
                "state": {
                    **dict(current.get("state") or {}),
                    "execution_strategy": "staged",
                },
            },
        )

        snapshot = runtime.workflow_runtime_snapshot()

        self.assertEqual(snapshot.execution_strategy, "recursive")

    def test_local_snapshot_rewrites_stale_cloud_boundary_wording(self):
        runtime = AgentRuntime(
            ScriptedProvider([], model="local-test"),
            self.store,
            self.workspace,
            config=self.config,
        )
        goal = self.store.create_goal(
            "Recover the local model",
            session_id=runtime.session_id,
        )
        self.store.transition_goal(
            goal.id,
            GoalStatus.AWAITING_PLAN_APPROVAL,
            reason="plan ready",
        )
        self.store.transition_goal(goal.id, GoalStatus.RUNNING, reason="planning")
        self.store.update_goal_metadata(
            goal.id,
            retry_reason="Internet/provider unavailable; saved stage unchanged.",
        )
        with runtime._live_activity_lock:
            runtime._live_provider_activity = {
                "state": "network_unavailable",
                "last_signal_at": time.time(),
            }
        snapshot = runtime.workflow_runtime_snapshot()
        self.assertEqual(snapshot.waiting_on, "model")
        self.assertIn("Local model runner unavailable", snapshot.reason)

    def test_sleep_mode_change_is_broadcast_for_other_ui_surfaces(self):
        runtime = AgentRuntime(
            ScriptedProvider([]), self.store, self.workspace, config=self.config
        )
        captured = []
        unsubscribe = runtime.events.subscribe(captured.append)
        try:
            runtime.set_sleep_mode(True, policy="full")
        finally:
            unsubscribe()
        changed = [event for event in captured if event.kind == "sleep.mode_changed"]
        self.assertEqual(len(changed), 1)
        self.assertTrue(changed[0].data["enabled"])
        self.assertEqual(changed[0].data["policy"], "full")
        self.assertEqual(changed[0].data["sleep_state"], "on")

    def test_pre_goal_local_continuation_updates_the_saved_semantic_turn(self):
        runtime = AgentRuntime(
            ScriptedProvider([], model="cloud-model"),
            self.store,
            self.workspace,
            config=self.config,
        )
        pending = {
            "turn_id": "turn-pre-goal-local",
            "original_input": "Build the saved project",
            "status": "awaiting_provider",
            "stage": "goal_intake",
            "last_error": "cloud quota exhausted",
            "model_capability_envelope": runtime.model_capability_envelope().to_dict(),
        }
        runtime._save_pending_semantic_turn(pending)
        descriptor = ModelDescriptor(
            provider="ollama",
            model="local-coder",
            execution_class="local",
            capabilities=("tools", "structured_output"),
            metadata={
                "parameter_count_billions": 32,
                "context_window_tokens": 65_536,
                "maximum_output_tokens": 8_192,
            },
        )
        local_provider = ScriptedProvider([], model="local-coder")

        result = runtime.continue_with_local_model(local_provider, descriptor)

        saved = self.store.get_workflow_session(runtime.session_id)["state"]
        saved_pending = saved["pending_semantic_turn"]
        self.assertEqual(saved_pending["model_capability_envelope"]["provider"], "ollama")
        self.assertEqual(saved_pending["model_capability_envelope"]["model"], "local-coder")
        self.assertEqual(saved_pending["last_error"], "")
        self.assertIsNone(saved_pending["retry_not_before"])
        self.assertTrue(
            saved_pending["local_continuation_policy"]["quality_floor"][
                "completion_gates_unchanged"
            ]
        )
        self.assertTrue(result["quality_gates_unchanged"])
        self.assertEqual(runtime.model_descriptor.provider, "ollama")
        self.assertEqual(runtime.model_name, "local-coder")

    def test_local_plan_reconciles_task_criteria_when_semantic_projection_is_empty(self):
        request = (
            "Create a small hello.txt file containing Hello from the local workflow "
            "and verify it."
        )
        semantic = {
            "original_request": request,
            "interpreted_outcome": request,
            "requested_effects": ["read_workspace"],
            "required_outcomes": ["existents:hello.txt"],
            "constraints": [],
            "exclusions": [],
            # This is the omission produced by the weak local model that used
            # to crash the post-critic transition.
            "acceptance_criteria": [],
            "requirement_anchors": [],
            "unresolved_decisions": [],
            "repository_evidence_refs": ["inspection:I001"],
            "status": "interpreted",
        }
        plan = {
            "semantic_goal": semantic,
            "summary": "Create and verify hello.txt.",
            "tasks": [{
                "title": "Create hello.txt",
                "description": "Create hello.txt in the workspace root.",
                "acceptance_criteria": ["hello.txt exists with the requested content."],
                "verification": ["Read hello.txt and compare its contents."],
                "depends_on": [],
                "risk": "low",
            }],
            "applicability_evidence": [{
                "fact": "The workspace is empty.",
                "source": "inspection:I001",
                "supports_tasks": ["1"],
            }],
            "execution_strategy": "Create the file and read it back for verification.",
            "expected_changes": [{
                "path": "hello.txt",
                "intent": "Create the requested file.",
                "basis": "explicit_user_requirement",
                "evidence_refs": ["user:request"],
                "supports_tasks": ["1"],
            }],
        }
        provider = ScriptedProvider([
            {"tool_calls": [{"name": "propose_semantic_goal", "args": semantic}]},
            {"tool_calls": [{"name": "propose_plan", "args": plan}]},
            {"tool_calls": [{
                "name": "submit_plan_review",
                "args": {"verdict": "pass", "summary": "Plan is complete.", "issues": []},
            }]},
        ], model="local-coder")
        runtime = AgentRuntime(
            provider,
            self.store,
            self.workspace,
            config=self.config,
            model_descriptor=ModelDescriptor(
                "ollama", "local-coder", ExecutionClass.LOCAL, capabilities=("tools",)
            ),
        )
        try:
            result = runtime.start_goal(request)
            self.assertIsNotNone(result)
            self.assertEqual(
                runtime.active_goal().status,
                GoalStatus.AWAITING_PLAN_APPROVAL,
            )
            self.assertEqual(
                result.tasks[0].acceptance_criteria,
                ("hello.txt exists with the requested content.",),
            )
            self.assertTrue(
                any(
                    event.event_type == "planning.semantic_criteria_reconciled"
                    for event in self.store.list_events(runtime.active_goal().id)
                )
            )
            self.assertTrue(
                any(
                    event.event_type == "planning.semantic_effects_reconciled"
                    for event in self.store.list_events(runtime.active_goal().id)
                )
            )
            self.assertIn(
                "mutate_workspace",
                runtime.active_goal().metadata["semantic_goal"]["requested_effects"],
            )
            provider.assert_exhausted()
        finally:
            runtime.close()

    def test_live_tool_resolution_is_durable_before_the_resolver_returns(self):
        runtime = AgentRuntime(
            ScriptedProvider([]), self.store, self.workspace, config=self.config
        )
        goal = self.store.create_goal("Approve a preview", session_id=runtime.session_id)
        fingerprint = "live-" + ("f" * 64)
        self.store.update_goal_metadata(
            goal.id,
            pending_tool_approval={
                "tool": "preview_html",
                "arguments": {"path": "index.html", "open_browser": True},
                "risk": "high",
                "action_fingerprint": fingerprint,
                "policy_group": "project_preview",
            },
        )
        observed: list[str] = []

        def resolver(_fingerprint, _decision):
            pending = self.store.get_goal(goal.id).metadata["pending_tool_approval"]
            observed.append(str(pending.get("decision") or ""))
            return True

        runtime.set_external_tool_approval_resolver(resolver)

        self.assertTrue(runtime.resolve_tool_approval(fingerprint, "allow_once"))
        self.assertEqual(observed, ["allow_once"])

    def test_callback_approval_clears_the_marker_it_just_created(self):
        runtime = AgentRuntime(
            ScriptedProvider([]),
            self.store,
            self.workspace,
            config=self.config,
            approval=lambda _name, _args, _risk: "allow_once",
        )
        goal = self.store.create_goal("Approve one command", session_id=runtime.session_id)

        self.assertTrue(
            runtime._approval_allowed(
                "run_command", {"command": "python -c \"print(1)\""}, "critical"
            )
        )
        self.assertEqual(
            self.store.get_goal(goal.id).metadata.get("pending_tool_approval"), {}
        )

    def test_allow_session_is_reused_for_matching_policy_group(self):
        decisions = iter(["allow_session"])
        runtime = AgentRuntime(
            ScriptedProvider([]),
            self.store,
            self.workspace,
            config=self.config,
            sleeper=lambda _seconds: None,
            approval=lambda _name, _args, _risk: next(decisions),
        )

        self.assertTrue(
            runtime._approval_allowed(
                "run_command", {"command": "npm install first-package"}, "critical"
            )
        )
        self.assertTrue(
            runtime._approval_allowed(
                "run_command", {"command": "npm install second-package"}, "critical"
            )
        )
        self.assertIn("dangerous_command", runtime._approval_session_groups())
        self.assertTrue(
            any(item.event_type == "approval.session_reused" for item in self.store.list_events())
        )

    def test_full_auto_approves_risky_tool_and_records_audit_event(self):
        runtime = AgentRuntime(
            ScriptedProvider([]),
            self.store,
            self.workspace,
            config=self.config,
            sleeper=lambda _seconds: None,
            approval=lambda _name, _args, _risk: (_ for _ in ()).throw(
                AssertionError("Full Auto must resolve before the manual approval UI")
            ),
        )
        runtime.set_sleep_mode(True, policy="full")

        self.assertTrue(
            runtime._approval_allowed(
                "run_command",
                {"command": "npm install an-example-package"},
                "critical",
            )
        )
        self.assertEqual(runtime.sleep_mode_policy(), "full")
        self.assertTrue(
            any(
                item.event_type == "sleep.full_auto_approval"
                for item in self.store.list_events()
            )
        )

    def test_full_auto_approves_critic_reviewed_plan_and_records_boundary(self):
        runtime, _provider = self.runtime([inspect_call(), plan_call(), plan_pass()])
        plan = runtime.start_goal("Build a small verified application")
        runtime.set_sleep_mode(True, policy="full")

        self.assertEqual(runtime.auto_resolve_full_auto_boundary(), ("plan",))
        goal = runtime.active_goal()
        self.assertIsNotNone(goal)
        assert goal is not None
        self.assertEqual(goal.status, GoalStatus.RUNNING)
        self.assertEqual(goal.active_plan_revision, plan.revision)
        events = self.store.list_recent_events(goal.id, limit=100)
        approval = next(
            item for item in events if item.event_type == "sleep.full_auto_plan_approval"
        )
        self.assertEqual(approval.payload["approved_by"], "sleep-full-auto")
        self.assertTrue(approval.payload["quality_gates_unchanged"])

    def test_continuous_controller_approves_plan_created_after_full_auto_was_enabled(self):
        runtime, _provider = self.runtime([inspect_call(), plan_call(), plan_pass()])
        runtime.set_sleep_mode(True, policy="full")
        plan = runtime.start_goal("Build a small verified application")
        self.assertEqual(runtime.active_goal().status, GoalStatus.AWAITING_PLAN_APPROVAL)

        with mock.patch.object(
            runtime,
            "_run_slice_impl",
            return_value=SliceResult(
                "running",
                "Execution reached a test checkpoint.",
                needs_user=True,
            ),
        ) as run_slice:
            runtime.continue_until_boundary()

        goal = runtime.active_goal()
        self.assertEqual(goal.status, GoalStatus.RUNNING)
        self.assertEqual(goal.active_plan_revision, plan.revision)
        run_slice.assert_called_once()
        self.assertTrue(
            any(
                event.event_type == "sleep.full_auto_plan_approval"
                for event in self.store.list_recent_events(goal.id, limit=100)
            )
        )

    def test_full_auto_plan_boundary_survives_runtime_restart(self):
        runtime, _provider = self.runtime([inspect_call(), plan_call(), plan_pass()])
        plan = runtime.start_goal("Resume unattended work after restart")
        runtime.set_sleep_mode(True, policy="full")
        self.store.close()

        self.store = StateStore(self.workspace)
        restarted = AgentRuntime(
            ScriptedProvider([]),
            self.store,
            self.workspace,
            config=self.config,
            sleeper=lambda _seconds: None,
            approval=lambda _name, _args, _risk: True,
        )
        self.assertEqual(restarted.sleep_mode_policy(), "full")
        self.assertEqual(restarted.auto_resolve_full_auto_boundary(), ("plan",))
        goal = self.store.get_goal(plan.goal_id)
        self.assertEqual(goal.status, GoalStatus.RUNNING)
        restarted.close()

    def test_full_auto_answers_intake_question_without_showing_a_hidden_prompt(self):
        runtime = AgentRuntime(
            ScriptedProvider([]),
            self.store,
            self.workspace,
            config=self.config,
        )
        runtime.set_sleep_mode(True, policy="full")
        question = {
            "id": "preview_target",
            "question": "Which preview should be used?",
            "options": [
                {"value": "browser", "label": "Browser", "recommended": True},
                {"value": "static", "label": "Static"},
            ],
        }
        with (
            mock.patch.object(runtime, "intake_questions", return_value=(question,)),
            mock.patch.object(runtime, "answer_intake_question") as answer,
        ):
            self.assertEqual(runtime.auto_resolve_full_auto_boundary(), ("question",))
        answer.assert_called_once_with("preview_target", "browser")
        self.assertTrue(
            any(
                item.event_type == "sleep.full_auto_question_answered"
                and item.payload["selection"] == "recommended"
                for item in self.store.list_events()
            )
        )

    def test_full_auto_uses_explicit_question_default_then_first_option(self):
        runtime = AgentRuntime(
            ScriptedProvider([]),
            self.store,
            self.workspace,
            config=self.config,
        )
        runtime.set_sleep_mode(True, policy="full")
        goal = self.store.create_goal("Choose a deterministic default", session_id=runtime.session_id)
        self.store.update_goal_metadata(
            goal.id,
            plan_questions=[
                {
                    "id": "q-default",
                    "question": "Which output?",
                    "default": "static",
                    "options": [
                        {"value": "browser", "label": "Browser"},
                        {"value": "static", "label": "Static"},
                    ],
                }
            ],
            plan_answers={},
        )
        with mock.patch.object(runtime, "answer_plan_question") as answer:
            self.assertEqual(runtime.auto_resolve_full_auto_boundary(), ("question",))
        answer.assert_called_once_with("q-default", "static")

    def test_full_auto_does_not_guess_when_recommendations_conflict(self):
        runtime = AgentRuntime(
            ScriptedProvider([]),
            self.store,
            self.workspace,
            config=self.config,
        )
        runtime.set_sleep_mode(True, policy="full")
        goal = self.store.create_goal("Do not guess", session_id=runtime.session_id)
        self.store.update_goal_metadata(
            goal.id,
            plan_questions=[
                {
                    "id": "q-conflict",
                    "question": "Which output?",
                    "options": [
                        {"value": "browser", "label": "Browser", "recommended": True},
                        {"value": "static", "label": "Static", "recommended": True},
                    ],
                }
            ],
            plan_answers={},
        )
        with mock.patch.object(runtime, "answer_plan_question") as answer:
            self.assertEqual(runtime.auto_resolve_full_auto_boundary(), ())
        answer.assert_not_called()
        self.assertTrue(
            any(item.event_type == "sleep.full_auto_question_blocked" for item in self.store.list_events())
        )

    def test_full_auto_resumes_durable_tool_boundary_after_restart(self):
        runtime, _provider = self.runtime([inspect_call(), plan_call(), plan_pass()])
        plan = runtime.start_goal("Resume an approved tool after restart")
        runtime.approve_plan(plan.revision)
        goal = runtime.active_goal()
        self.assertIsNotNone(goal)
        assert goal is not None
        fingerprint = "restart-tool-" + ("a" * 64)
        self.store.update_goal_metadata(
            goal.id,
            pending_tool_approval={
                "tool": "run_command",
                "arguments": {"command": "python -c pass"},
                "risk": "critical",
                "action_fingerprint": fingerprint,
                "policy_group": "dangerous_command",
            },
        )
        self.store.transition_goal(goal.id, GoalStatus.PAUSED, reason="tool approval checkpoint")
        runtime.set_sleep_mode(True, policy="full")

        self.assertEqual(runtime.auto_resolve_full_auto_boundary(), ("tool",))
        resumed = self.store.get_goal(goal.id)
        self.assertEqual(resumed.status, GoalStatus.RUNNING)
        self.assertEqual(
            resumed.metadata["pending_tool_approval"]["decision"],
            "allow_once",
        )

    def test_durable_pending_tool_approval_overrides_stale_heartbeat_projection(self):
        runtime = AgentRuntime(
            ScriptedProvider([]),
            self.store,
            self.workspace,
            config=self.config,
        )
        goal = self.store.create_goal(
            "Recover a preview approval",
            session_id=runtime.session_id,
        )
        self.store.transition_goal(
            goal.id,
            GoalStatus.AWAITING_PLAN_APPROVAL,
            reason="plan ready",
        )
        self.store.transition_goal(goal.id, GoalStatus.RUNNING, reason="execution")
        self.store.update_goal_metadata(
            goal.id,
            pending_tool_approval={
                "tool": "preview_html",
                "arguments": {"path": "public/index.html"},
                "risk": "high",
                "action_fingerprint": "stale-approval-" + ("a" * 64),
            },
        )
        stale = time.time() - 3_600
        self.store.append_event(
            "workflow.heartbeat",
            goal_id=goal.id,
            entity_type="worker",
            entity_id="coordinator",
            payload={"heartbeat_at": stale, "stage": "working"},
        )
        for index in range(60):
            self.store.append_event(
                "worker.heartbeat",
                goal_id=goal.id,
                entity_type="worker",
                entity_id="coordinator",
                payload={"sequence": index},
            )

        snapshot = runtime.workflow_runtime_snapshot()

        self.assertEqual(snapshot.phase, "waiting_for_approval")
        self.assertEqual(snapshot.waiting_on, "user")
        self.assertEqual(snapshot.liveness, "waiting")
        self.assertIn("preview_html", snapshot.reason)

    def test_durable_sleep_auto_approves_safe_preview_after_ui_restart(self):
        runtime = AgentRuntime(
            ScriptedProvider([]),
            self.store,
            self.workspace,
            config=self.config,
            sleeper=lambda _seconds: None,
            approval=lambda _name, _args, _risk: (_ for _ in ()).throw(
                AssertionError("safe durable Sleep action reached the UI")
            ),
        )
        runtime.set_sleep_mode(True)

        self.assertTrue(
            runtime._approval_allowed(
                "preview_html",
                {"path": "index.html", "open_browser": True},
                "high",
            )
        )
        self.assertTrue(runtime.sleep_mode_enabled())
        self.assertTrue(
            any(
                item.event_type == "sleep.auto_approval"
                for item in self.store.list_events()
            )
        )

    def test_general_sleep_survives_state_store_restart(self):
        runtime = AgentRuntime(
            ScriptedProvider([]), self.store, self.workspace, config=self.config
        )
        runtime.set_sleep_mode(True)
        self.store.close()

        self.store = StateStore(self.workspace)

        self.assertEqual(
            self.store.get_workflow_session("workspace-session")["sleep_state"],
            "on",
        )


class UltraQualityFeedbackTests(RuntimeTestCase):
    def test_quality_replan_feedback_excludes_successful_node_history(self):
        runtime, _provider = self.runtime([])
        result = SimpleNamespace(
            run=SimpleNamespace(id=""),
            node_results=(
                SimpleNamespace(
                    node_id="M001",
                    success=True,
                    findings=(
                        "Superseded canvas-missing finding.",
                    )
                ),
                SimpleNamespace(
                    node_id="M002",
                    success=False,
                    findings=("Calculator state does not update after button input.",),
                ),
            ),
            global_result=SimpleNamespace(
                node_id="__global__",
                success=False,
                findings=("Global browser behavior did not pass.",)
            ),
        )

        feedback = runtime._ultra_quality_feedback(result)

        self.assertIn("Calculator state does not update", feedback)
        self.assertIn("Global browser behavior did not pass", feedback)
        self.assertNotIn("Superseded canvas-missing", feedback)


class PlanningAndCompletionTests(RuntimeTestCase):
    def test_explicit_path_parser_does_not_treat_three_js_product_name_as_a_file(self):
        self.assertEqual(
            _extract_explicit_workspace_paths(
                "Create a complete Three.js 3D calculator in index.html"
            ),
            ("index.html",),
        )
        self.assertEqual(
            _extract_explicit_workspace_paths("Create the file named Three.js"),
            ("Three.js",),
        )

    def test_local_replan_projects_in_scope_repair_after_empty_transport(self):
        provider = ScriptedProvider(
            [
                inspect_call(),
                plan_call(),
                plan_pass(),
                "",
                "",
                plan_pass(),
            ],
            model="local-coder",
        )
        runtime = AgentRuntime(
            provider,
            self.store,
            self.workspace,
            config=replace(self.config, no_action_limit=1, planning_steps=4),
            model_descriptor=ModelDescriptor(
                "ollama", "local-coder", ExecutionClass.LOCAL,
                capabilities=("tools",),
            ),
        )
        first = runtime.start_goal("Repair and verify artifact.txt")

        revised = runtime.reject_plan(
            "Use a materially different repair strategy while preserving scope."
        )

        self.assertIsNotNone(revised)
        self.assertEqual(revised.revision, first.revision + 1)
        self.assertEqual(revised.status, PlanStatus.PENDING_APPROVAL)
        self.assertEqual(
            [item["path"] for item in revised.expected_changes],
            ["artifact.txt"],
        )
        self.assertIn("Repair and verify", revised.summary)
        self.assertTrue(
            any(
                event.event_type == "planning.deterministic_plan_projected"
                for event in self.store.list_recent_events(
                    revised.goal_id, limit=100
                )
            )
        )
        provider.assert_exhausted()

    def test_local_html_coordinator_is_constrained_to_missing_preview_evidence(self):
        request = "Create and verify an interactive Three.js calculator in index.html"
        semantic = {
            **plan_call()["tool_calls"][0]["args"]["semantic_goal"],
            "original_request": request,
            "interpreted_outcome": "Create and verify the interactive calculator in index.html.",
            "acceptance_criteria": [
                "The calculator buttons produce the correct visible result.",
                "The Three.js interaction passes in the managed preview.",
            ],
            "repository_evidence_refs": ["inspection:I001", "user:request"],
        }
        staged_plan = plan_call()
        staged_args = staged_plan["tool_calls"][0]["args"]
        staged_args.pop("semantic_goal", None)
        staged_args["semantic_fingerprint"] = "0" * 64
        staged_args["applicability_evidence"][0]["source"] = "inspection:I001"
        staged_args["expected_changes"][0].update(
            {
                "path": "index.html",
                "basis": "explicit_user_requirement",
                "evidence_refs": ["user:request"],
            }
        )
        staged_args["tasks"][0].update(
            {
                "title": "Implement and verify the interactive calculator",
                "description": "Implement every calculator button and the Three.js interaction.",
                "acceptance_criteria": list(semantic["acceptance_criteria"]),
                "verification": [
                    "Run deterministic button scenarios in the managed preview.",
                    "Verify the Three.js interaction in the browser.",
                ],
            }
        )

        def assert_preview_only(provider_request):
            advertised = {
                schema["function"]["name"] for schema in provider_request.tools
            }
            self.assertEqual(advertised, {"preview_html"})
            rendered = json.dumps(provider_request.conversation, ensure_ascii=False)
            self.assertIn('"required_next_tool": "preview_html"', rendered)
            self.assertIn("index.html", rendered)
            return "Preview action checkpointed for the next slice."

        def assert_failed_preview_requires_repair(provider_request):
            advertised = {
                schema["function"]["name"] for schema in provider_request.tools
            }
            self.assertEqual(
                advertised,
                {"read_file", "edit_file", "write_file"},
            )
            rendered = json.dumps(provider_request.conversation, ensure_ascii=False)
            self.assertIn('"required_next_tool": "repair_failed_preview"', rendered)
            self.assertIn("verification", rendered)
            self.assertIn("failed", rendered)
            return "Repair action checkpointed for the next slice."

        provider = ScriptedProvider(
            [
                {"tool_calls": [{"name": "propose_semantic_goal", "args": semantic}]},
                staged_plan,
                plan_pass(),
                assert_preview_only,
                assert_failed_preview_requires_repair,
            ],
            model="local-coder",
        )
        runtime = AgentRuntime(
            provider,
            self.store,
            self.workspace,
            config=self.config,
            model_descriptor=ModelDescriptor(
                "ollama", "local-coder", ExecutionClass.LOCAL, capabilities=("tools",)
            ),
        )

        plan = runtime.start_goal(request)
        runtime.approve_plan(plan.revision)
        legacy_changes = [
            *plan.expected_changes,
            {
                "path": "Three.js",
                "intent": "Legacy parser mistook a technology name for a file.",
                "basis": "explicit_user_requirement",
                "evidence_refs": ["user:request"],
                "supports_tasks": ["T001"],
            },
        ]
        with self.store.transaction() as connection:
            connection.execute(
                "UPDATE plans SET expected_changes_json=? WHERE goal_id=? AND revision=?",
                (
                    json.dumps(legacy_changes, separators=(",", ":")),
                    plan.goal_id,
                    plan.revision,
                ),
            )
        legacy_goal = self.store.get_goal(plan.goal_id)
        legacy_contract = dict(legacy_goal.metadata["goal_contract"])
        legacy_contract["artifact_expectations"] = ["index.html", "Three.js"]
        legacy_target = dict(legacy_goal.metadata["quality_target"])
        legacy_target["artifact_ids"] = ["index.html", "Three.js"]
        legacy_claims = [
            *legacy_goal.metadata.get("resource_claims", ()),
            {
                "purpose": "Legacy false path claim",
                "kind": "file",
                "supports_tasks": ["T001"],
                "inspection_refs": ["user:request"],
                "selector": "Three.js",
                "resolved_paths": ["Three.js"],
                "state": "resolved",
            },
        ]
        self.store.update_goal_metadata(
            plan.goal_id,
            goal_contract=legacy_contract,
            quality_target=legacy_target,
            resource_claims=legacy_claims,
        )
        (self.workspace / "index.html").write_text(
            "<!doctype html><html><body><button>1</button></body></html>",
            encoding="utf-8",
        )

        result = runtime.run_slice(steps=1)

        self.assertFalse(result.completed)
        repaired = self.store.get_goal(plan.goal_id)
        self.assertEqual(
            repaired.metadata["goal_contract"]["artifact_expectations"],
            ["index.html"],
        )
        self.assertEqual(
            repaired.metadata["quality_target"]["artifact_ids"],
            ["index.html"],
        )
        self.assertNotIn(
            "Three.js",
            [claim["selector"] for claim in repaired.metadata["resource_claims"]],
        )
        self.assertTrue(repaired.metadata["legacy_contract_projection_repaired"])

        self.store.add_evidence(
            goal_id=plan.goal_id,
            plan_revision=plan.revision,
            task_id="T001",
            kind="tool",
            summary="Managed preview failed its interaction assertion.",
            data={
                "tool": "preview_html",
                "verification": "failed",
                "failure_kind": "application",
                "mutation_sequence": int(
                    repaired.metadata.get("mutation_sequence", 0) or 0
                ),
                "interactions_passed": False,
            },
            created_by="harness",
            verified=False,
        )
        rejected = runtime._control_update_task(
            self.store.get_goal(plan.goal_id),
            runtime.latest_plan(),
            {
                "task_id": "T001",
                "status": "done",
                "note": "The preview was attempted.",
                "evidence": ["A preview action returned."],
            },
        )
        self.assertIn("fresh passing managed preview", rejected)

        action_id = self.store.begin_action(
            plan.goal_id,
            "preview_html",
            {
                "_harness_mutation_sequence": int(
                    runtime.active_goal().metadata.get("mutation_sequence", 0) or 0
                ),
                "arguments": {"path": "index.html", "verify": True},
            },
            task_id="T001",
            risk="medium",
        )
        self.store.complete_action(
            action_id,
            json.dumps(
                {
                    "verification": "failed",
                    "page_errors": ["interaction assertion failed"],
                    "interaction_results": [{"passed": False}],
                }
            ),
        )

        second = runtime.run_slice(steps=1)

        self.assertFalse(second.completed)
        provider.assert_exhausted()

    def test_repeated_action_circuit_breaker_checkpoints_the_slice_immediately(self):
        repeated = lambda index: {
            "tool_calls": [
                {"id": f"list-{index}", "name": "list_files", "args": {"path": "."}}
            ]
        }
        runtime, provider = self.runtime(
            [
                inspect_call(),
                plan_call(),
                plan_pass(),
                repeated(1),
                repeated(2),
                repeated(3),
                "this turn must remain unconsumed",
            ]
        )
        plan = runtime.start_goal("Build persistent behavior")
        runtime.approve_plan(plan.revision)

        result = runtime.run_slice(steps=8)

        self.assertEqual(result.steps, 3)
        self.assertEqual(provider.remaining, 1)
        self.assertTrue(
            any(
                event.event_type == "execution.repeated_action_recovery"
                for event in self.store.list_recent_events(plan.goal_id, limit=100)
            )
        )

    def test_repair_scope_allows_approved_capability_but_rejects_new_one(self):
        goal = self.store.create_goal("Create and verify the requested application.")
        task = {
            "id": "T001",
            "title": "Build application",
            "description": "Create package.json, run npm install, and verify the app.",
            "acceptance_criteria": ["The application runs."],
            "verification": ["Run npm test."],
            "depends_on": [],
            "risk": "medium",
        }
        basis = {
            "applicability_evidence": [
                {"fact": "Workspace inspected.", "source": "inspection:I001", "supports_tasks": ["T001"]}
            ],
            "execution_strategy": "Install the approved test dependency and run verification.",
            "expected_changes": [
                {"path": "package.json", "intent": "Add the approved dependency.", "supports_tasks": ["T001"]}
            ],
        }
        approved = self.store.create_plan(goal.id, "Initial plan", [task], **basis)
        proposed = self.store.create_plan(
            goal.id,
            "Repair plan",
            [{**task, "description": "Repair package.json, run npm install, and verify again."}],
            **basis,
        )
        self.assertTrue(
            AgentRuntime._repair_revision_is_in_scope(approved, proposed, [task])
        )

        prefixed_basis = {
            **basis,
            "expected_changes": [
                {
                    "path": "workspace/package.json",
                    "intent": "Repair the already approved dependency manifest.",
                    "supports_tasks": ["T001"],
                }
            ],
        }
        prefixed = self.store.create_plan(
            goal.id,
            "Transport-prefixed repair",
            [task],
            **prefixed_basis,
        )
        self.assertTrue(
            AgentRuntime._repair_revision_is_in_scope(approved, prefixed, [task])
        )

        network_task = {
            **task,
            "description": "Repair the package and call a new external service over the network.",
        }
        expanded = self.store.create_plan(
            goal.id,
            "Expanded repair",
            [network_task],
            **basis,
        )
        self.assertFalse(
            AgentRuntime._repair_revision_is_in_scope(
                approved,
                expanded,
                [network_task],
            )
        )

    def test_ultra_convergence_scopes_and_approves_persisted_plan_projection(self):
        runtime, _provider = self.runtime([])
        goal = self.store.create_goal(
            "Build and verify the calculator.",
            session_id=runtime.session_id,
        )
        self.store.update_goal_metadata(goal.id, ultra_run_id="ultra-test")
        self.store.transition_goal(goal.id, GoalStatus.AWAITING_PLAN_APPROVAL)
        task = {
            "id": "T001",
            "title": "Build calculator",
            "description": "Create the approved calculator artifact.",
            "acceptance_criteria": ["The calculator artifact exists."],
            "verification": ["Read the calculator artifact."],
            "depends_on": [],
            "risk": "medium",
        }
        basis = {
            "applicability_evidence": [
                {
                    "fact": "Workspace inspected.",
                    "source": "inspection:I001",
                    "supports_tasks": ["T001"],
                }
            ],
            "execution_strategy": "Build, inspect, and independently review.",
            "expected_changes": [
                {
                    "path": "index.html",
                    "intent": "Create the calculator.",
                    "supports_tasks": ["T001"],
                }
            ],
        }
        accepted = self.store.create_plan(goal.id, "Approved plan", [task], **basis)
        self.store.transition_goal(goal.id, GoalStatus.AWAITING_PLAN_APPROVAL)
        self.store.approve_plan(
            goal.id,
            accepted.revision,
            expected_fingerprint=accepted.fingerprint,
        )
        self.store.update_goal_metadata(
            goal.id,
            waiting_question="This stale approval message must not survive execution.",
        )
        self.assertEqual(runtime.dashboard().waiting_question, "")
        proposed = self.store.create_plan(goal.id, "Repair plan", [task], **basis)
        revision_required = SimpleNamespace(
            run=SimpleNamespace(phase="revision_required"),
            results=(),
            global_result=None,
        )
        stable_boundary = SimpleNamespace(
            run=SimpleNamespace(phase="awaiting_plan_approval"),
            results=(),
            global_result=None,
        )

        with mock.patch.object(runtime, "_ensure_ultra_session"), mock.patch.object(
            runtime,
            "wait_for_ultra",
            side_effect=[revision_required, stable_boundary],
        ), mock.patch.object(
            runtime, "replan_ultra", return_value=SimpleNamespace(revision=99)
        ) as replan, mock.patch.object(runtime, "approve_ultra") as approve:
            result = runtime.converge_ultra()

        self.assertIs(result, stable_boundary)
        approve.assert_called_once_with(
            proposed.revision,
            approved_by="risk-adaptive-policy",
        )
        replan.assert_called_once_with(
            mock.ANY,
            allow_scope_expansion=False,
        )

    def test_ultra_convergence_does_not_replan_a_test_harness_failure(self):
        runtime, _provider = self.runtime([])
        goal = self.store.create_goal(
            "Verify the browser artifact.",
            session_id=runtime.session_id,
        )
        self.store.update_goal_metadata(goal.id, ultra_run_id="ultra-contract")
        self.store.transition_goal(goal.id, GoalStatus.AWAITING_PLAN_APPROVAL)
        self.store.transition_goal(goal.id, GoalStatus.RUNNING)
        self.store.transition_goal(goal.id, GoalStatus.REVISING)
        revision_required = SimpleNamespace(
            run=SimpleNamespace(phase="revision_required"),
            node_results=(
                SimpleNamespace(
                    findings=("browser assertion schema failed",),
                    component_package={
                        "failure_diagnostic": {
                            "failure_kind": "contract",
                            "blocker_owner": "test_harness",
                            "mutation_prohibited": True,
                            "failure_fingerprint": "a" * 64,
                        }
                    },
                ),
            ),
            global_result=None,
        )

        with mock.patch.object(runtime, "_ensure_ultra_session"), mock.patch.object(
            runtime, "wait_for_ultra", return_value=revision_required
        ), mock.patch.object(runtime, "replan_ultra") as replan:
            result = runtime.converge_ultra()

        self.assertIs(result, revision_required)
        replan.assert_not_called()
        current = self.store.get_goal(goal.id)
        self.assertEqual(current.status, GoalStatus.PAUSED)
        self.assertEqual(
            current.metadata["verification_blocker"]["blocker_owner"],
            "test_harness",
        )

    def test_ultra_convergence_honors_strategy_repetition_limit(self):
        runtime, _provider = self.runtime([])
        goal = self.store.create_goal(
            "Repair the candidate without plan thrashing.",
            session_id=runtime.session_id,
        )
        self.store.update_goal_metadata(
            goal.id,
            ultra_run_id="ultra-repetition",
            in_scope_quality_revision_attempts=2,
        )
        self.store.transition_goal(goal.id, GoalStatus.AWAITING_PLAN_APPROVAL)
        self.store.transition_goal(goal.id, GoalStatus.RUNNING)
        self.store.transition_goal(goal.id, GoalStatus.REVISING)
        revision_required = SimpleNamespace(
            run=SimpleNamespace(phase="revision_required"),
            node_results=(
                SimpleNamespace(
                    findings=("same candidate failure",),
                    component_package={},
                ),
            ),
            global_result=None,
        )

        with mock.patch.object(runtime, "_ensure_ultra_session"), mock.patch.object(
            runtime, "wait_for_ultra", return_value=revision_required
        ), mock.patch.object(runtime, "replan_ultra") as replan:
            result = runtime.converge_ultra()

        self.assertIs(result, revision_required)
        replan.assert_not_called()
        current = self.store.get_goal(goal.id)
        self.assertEqual(current.status, GoalStatus.PAUSED)
        self.assertIn("strategy", current.metadata["waiting_question"].casefold())

    def test_resume_contracts_ultra_repair_from_accepted_foundation(self):
        runtime, _provider = self.runtime([])
        goal = self.store.create_goal(
            "Build and verify the calculator.",
            session_id=runtime.session_id,
        )
        task = {
            "id": "T001",
            "title": "Build calculator",
            "description": "Create the approved calculator artifact.",
            "acceptance_criteria": ["The calculator artifact exists."],
            "verification": ["Read the calculator artifact."],
            "depends_on": [],
            "risk": "medium",
        }
        basis = {
            "applicability_evidence": [
                {
                    "fact": "Workspace inspected.",
                    "source": "inspection:I001",
                    "supports_tasks": ["T001"],
                }
            ],
            "execution_strategy": "Build and verify.",
            "expected_changes": [
                {
                    "path": "src/App.js",
                    "intent": "Create the calculator.",
                    "supports_tasks": ["T001"],
                }
            ],
        }
        accepted = self.store.create_plan(goal.id, "Approved plan", [task], **basis)
        self.store.transition_goal(goal.id, GoalStatus.AWAITING_PLAN_APPROVAL)
        self.store.approve_plan(
            goal.id,
            accepted.revision,
            expected_fingerprint=accepted.fingerprint,
        )
        self.store.update_goal_metadata(goal.id, ultra_run_id="ultra-repair")
        self.store.transition_goal(goal.id, GoalStatus.REVISING)
        expanded = self.store.create_plan(
            goal.id,
            "Expanded repair",
            [task],
            **{
                **basis,
                "expected_changes": [
                    {
                        "path": "new/outside.js",
                        "intent": "Add an avoidable new path.",
                        "supports_tasks": ["T001"],
                    }
                ],
            },
        )
        self.store.transition_goal(goal.id, GoalStatus.AWAITING_PLAN_APPROVAL)

        def replan(feedback):
            self.assertIn("src/App.js", feedback)
            self.store.transition_goal(goal.id, GoalStatus.PAUSED)
            return "ultra-repair-plan"

        with mock.patch.object(
            runtime, "replan_ultra", side_effect=replan
        ) as ultra_replan, mock.patch.object(runtime, "generate_plan") as normal_plan:
            result = runtime.resume()

        self.assertEqual(result, "ultra-repair-plan")
        ultra_replan.assert_called_once()
        normal_plan.assert_not_called()
        self.assertEqual(self.store.get_plan(goal.id, expanded.revision).status, PlanStatus.REJECTED)

    def test_ultra_replan_contract_exhaustion_is_a_resumable_boundary(self):
        from agent.ultra import AgentProtocolError

        runtime, _provider = self.runtime([])
        goal = self.store.create_goal(
            "Repair the approved calculator.",
            session_id=runtime.session_id,
        )
        self.store.update_goal_metadata(
            goal.id,
            ultra_run_id="ultra-source",
            accepted_foundation_source_run_id="ultra-source",
        )
        self.store.transition_goal(goal.id, GoalStatus.AWAITING_PLAN_APPROVAL)
        self.store.transition_goal(goal.id, GoalStatus.REVISING)
        fake_session = SimpleNamespace(
            running=False,
            restart_plan_from_accepted_foundation=mock.Mock(
                side_effect=AgentProtocolError(
                    "repair plan paths exceed approved scope: new.css"
                )
            ),
        )

        with mock.patch.object(
            runtime, "active_ultra_run", return_value=SimpleNamespace(id="ultra-source")
        ), mock.patch.object(
            runtime, "_make_ultra_session", return_value=fake_session
        ), mock.patch.object(self.store, "update_ultra_run"):
            result = runtime.replan_ultra("Reuse only approved paths.")

        self.assertIsInstance(result, SliceResult)
        self.assertEqual(result.status, "paused")
        paused = self.store.get_goal(goal.id)
        self.assertEqual(paused.status, GoalStatus.PAUSED)
        self.assertEqual(paused.metadata["resume_action"], "ultra_replan")
        self.assertEqual(paused.metadata["boundary_kind"], "contract_incompatibility")

        with mock.patch.object(
            runtime, "replan_ultra", return_value="retried-ultra-plan"
        ) as retry, mock.patch.object(runtime, "generate_plan") as normal_plan:
            resumed = runtime.resume()

        self.assertEqual(resumed, "retried-ultra-plan")
        retry.assert_called_once()
        normal_plan.assert_not_called()

    def test_resume_rebuilds_an_interrupted_discovering_plan(self):
        runtime, _provider = self.runtime([])
        goal = self.store.create_goal(
            "Resume the exact interrupted planning request.",
            session_id=runtime.session_id,
        )
        self.store.transition_goal(
            goal.id,
            GoalStatus.DISCOVERING,
            reason="planning began before process interruption",
        )

        with mock.patch.object(
            runtime, "generate_plan", return_value="restored-plan"
        ) as generate:
            result = runtime.resume()

        self.assertEqual(result, "restored-plan")
        generate.assert_called_once()
        feedback = generate.call_args.args[0]
        self.assertIn("persisted semantic state", feedback)
        events = self.store.list_recent_events(goal.id, limit=20)
        self.assertTrue(
            any(item.event_type == "planning.resumed" for item in events)
        )

    def test_windows_gpu_probe_accepts_amd_display_adapter(self):
        def which(name):
            if name == "nvidia-smi":
                return None
            if name == "powershell":
                return "powershell"
            return None

        completed = subprocess.CompletedProcess(
            ["powershell"],
            0,
            stdout=(
                '{"Name":"AMD Radeon RX 7900 XTX",'
                '"AdapterCompatibility":"Advanced Micro Devices, Inc.",'
                '"DriverVersion":"31.0.24002.92",'
                '"AdapterRAM":4293918720,'
                '"Status":"OK"}'
            ),
            stderr="",
        )
        with mock.patch("agent.hardware.shutil.which", side_effect=which), mock.patch(
            "agent.hardware.subprocess.run",
            return_value=completed,
        ):
            probe = probe_local_gpu(environ={})

        self.assertTrue(probe.gpu_available)
        self.assertEqual(probe.source, "win32-video-controller")
        self.assertEqual(probe.devices[0]["name"], "AMD Radeon RX 7900 XTX")

    def test_windows_gpu_probe_rejects_basic_display_adapter(self):
        def which(name):
            if name == "nvidia-smi":
                return None
            if name == "powershell":
                return "powershell"
            return None

        completed = subprocess.CompletedProcess(
            ["powershell"],
            0,
            stdout=(
                '[{"Name":"Microsoft Basic Display Adapter",'
                '"AdapterCompatibility":"Microsoft",'
                '"DriverVersion":"10.0.0.0",'
                '"AdapterRAM":0,'
                '"Status":"OK"}]'
            ),
            stderr="",
        )
        with mock.patch("agent.hardware.shutil.which", side_effect=which), mock.patch(
            "agent.hardware.subprocess.run",
            return_value=completed,
        ):
            probe = probe_local_gpu(environ={})

        self.assertFalse(probe.gpu_available)
        self.assertEqual(probe.source, "win32-video-controller")
        self.assertIn("unsupported/basic/virtual", probe.message)

    def test_runtime_config_accepts_gpu_required_boolean_setting(self):
        enabled = update_runtime_config(self.config, "require_gpu", "on")
        self.assertTrue(enabled.require_local_gpu)
        disabled = update_runtime_config(enabled, "local_gpu", "cpu")
        self.assertFalse(disabled.require_local_gpu)

    def test_gpu_required_blocks_local_ultra_without_gpu_evidence(self):
        runtime = AgentRuntime(
            ScriptedProvider([]),
            self.store,
            self.workspace,
            config=replace(self.config, require_local_gpu=True),
            model_descriptor=ModelDescriptor(
                "ollama",
                "gemma4",
                ExecutionClass.LOCAL,
                capabilities=("tools",),
            ),
            permission_adapter=PermissionAdapter("normal", DockerSandbox()),
        )
        with mock.patch(
            "agent.runtime.probe_local_gpu",
            return_value=HardwareProbeResult(False, "test", message="no GPU in test"),
        ):
            with self.assertRaises(RuntimeStateError) as raised:
                runtime._require_ultra_setup()
        self.assertIn("GPU-required", str(raised.exception))
        self.assertIn("AGENT_REQUIRE_LOCAL_GPU=0", str(raised.exception))

    def test_gpu_required_records_probe_metadata_when_available(self):
        runtime = AgentRuntime(
            ScriptedProvider([]),
            self.store,
            self.workspace,
            config=replace(self.config, require_local_gpu=True),
            model_descriptor=ModelDescriptor(
                "ollama",
                "gemma4",
                ExecutionClass.LOCAL,
                capabilities=("tools",),
            ),
            permission_adapter=PermissionAdapter("normal", DockerSandbox()),
        )
        probe = HardwareProbeResult(
            True,
            "test",
            devices=({"name": "RTX Test", "driver": "555.0"},),
            message="ok",
        )
        with mock.patch("agent.runtime.probe_local_gpu", return_value=probe):
            descriptor, _permissions = runtime._require_ultra_setup()
        self.assertTrue(descriptor.metadata["gpu_required"])
        self.assertEqual(descriptor.metadata["hardware_probe"]["devices"][0]["name"], "RTX Test")

    def test_castle_goal_weak_first_result_refines_repairs_failure_and_stops_for_visual_review(self):
        castle_plan = plan_call()
        castle_plan["tool_calls"][0]["args"]["expected_changes"][0]["path"] = "index.html"
        weak_html = '<html><body><div id="castle">Castle</div></body></html>'
        broken_html = '<html><style>@keyframes ramStrike{to{transform:translateX(2px)}}</style><body><div id="castle">BROKEN siege tower arrows catapult</div></body></html>'
        final_html = '<html><style>@keyframes ramStrike{to{transform:translateX(2px)}}@media(max-width:600px){#castle{width:90%}}</style><body><main id="castle" aria-label="castle siege"><div id="gate">Gate</div><div id="siege-tower">Tower</div><script>function fireArrow(){} function launchCatapult(){}</script></main></body></html>'
        failed_review = {"tool_calls": [{"id": "weak-review", "name": "submit_review", "args": {
            "verdict": "fail", "summary": "The first castle is a weak static placeholder.",
            "issues": [{"severity": "high", "title": "Improve siege detail", "details": "Actors and motion are missing.",
                        "acceptance_criteria": ["Castle actors and animation components are structurally present."]}],
            "checked_task_ids": ["T001"],
        }}]}
        first_turn = {"tool_calls": [
            task_update("in_progress", note="creating weak candidate"),
            {"id": "weak-write", "name": "write_file", "args": {"path": "index.html", "content": weak_html}},
            {"id": "weak-read", "name": "read_file", "args": {"path": "index.html"}},
            {"id": "weak-preview", "name": "preview_html", "args": {"path": "index.html", "open_browser": False, "settle_ms": 0}},
            task_update("done", ["candidate read back"], "weak candidate created"), finish_call(),
        ]}
        broken_turn = {"tool_calls": [
            {"id": "repair-start", "name": "update_task", "args": {"task_id": "T002", "status": "in_progress", "note": "refining", "evidence": []}},
            {"id": "broken-write", "name": "write_file", "args": {"path": "index.html", "content": broken_html}},
            {"id": "broken-check", "name": "run_bash", "args": {"command": "python -c \"import pathlib,sys;sys.exit(1 if 'BROKEN' in pathlib.Path('index.html').read_text() else 0)\""}},
        ]}
        repaired_turn = {"tool_calls": [
            {"id": "fixed-write", "name": "write_file", "args": {"path": "index.html", "content": final_html}},
            {"id": "fixed-check", "name": "run_bash", "args": {"command": "python -c \"import pathlib,sys;sys.exit(1 if 'BROKEN' in pathlib.Path('index.html').read_text() else 0)\""}},
            {"id": "fixed-preview", "name": "preview_html", "args": {"path": "index.html", "open_browser": False, "settle_ms": 0}},
            {"id": "repair-done", "name": "update_task", "args": {"task_id": "T002", "status": "done", "note": "failure repaired", "evidence": ["fresh narrow check passed"]}},
            finish_call(),
        ]}
        final_review = {"tool_calls": [{"id": "final-review", "name": "submit_review", "args": {
            "verdict": "pass", "summary": "Structural requirements and the controlled runtime check pass.",
            "issues": [], "checked_task_ids": ["T001", "T002"],
        }}]}
        runtime, provider = self.runtime([
            inspect_call(), castle_plan, plan_pass(), first_turn, failed_review,
            broken_turn, repaired_turn, final_review,
        ])
        plan = runtime.start_goal("Create a detailed animated castle siege in one self-contained HTML file")
        goal_id = runtime.active_goal().id
        run_id = runtime.active_goal().metadata["run_id"]
        runtime.approve_plan(plan.revision)

        first = runtime.run_slice(steps=1)
        self.assertFalse(first.completed)
        self.assertEqual(self.store.get_latest_plan(goal_id).revision, 2)
        self.assertEqual(self.store.get_latest_plan(goal_id).status, PlanStatus.ACCEPTED)
        runtime.run_slice(steps=1)
        final = runtime.run_slice(steps=1)

        goal = self.store.get_goal(goal_id)
        self.assertTrue(
            final.completed,
            (
                final,
                goal.status,
                goal.metadata.get("waiting_question"),
                goal.metadata.get("convergence_state"),
                provider.remaining,
                [
                    (item["tool_name"], item["status"], item["result_summary"])
                    for item in self.store.list_actions(goal_id)
                ],
            ),
        )
        self.assertEqual(goal.metadata["run_id"], run_id)
        self.assertEqual(goal.metadata["convergence_state"], "converged_with_limitations")
        self.assertEqual(goal.metadata["completion_disposition"], "completed_with_limitations")
        self.assertTrue(goal.metadata["goal_change_sets"])

    def test_no_native_tools_uses_harness_generated_constrained_action(self):
        runtime, provider = self.runtime(['proposal: {"name":"read_file","args":{"path":"x.py"}}'])
        provider.capability_profile = ModelCapabilityProfile("weak", tool_call_support=False)
        goal = self.store.create_goal("Inspect one file")

        turn = runtime._call_provider(
            [{"role": "user", "content": "inspect"}], agent_tools.TOOL_SCHEMAS,
            "bounded worker", actor="worker", step=1,
        )

        self.assertEqual(turn.tool_calls[0].name, "read_file")
        self.assertTrue(turn.tool_calls[0].id.startswith("harness-worker-"))
        events = self.store.list_recent_events(goal.id, limit=20)
        self.assertTrue(any(event.event_type == "provider.request_adapter_selected" for event in events))

    def test_below_target_chat_candidate_escalates_into_same_goal_run_on_yes(self):
        chat_write = {"tool_calls": [{
            "id": "chat-write", "name": "write_file",
            "args": {"path": "candidate.txt", "content": "first draft"},
        }]}
        runtime, _provider = self.runtime([
            semantic_turn(
                "goal",
                original="Create a polished candidate",
                goal_intake=semantic_goal_intake("Create a polished candidate"),
                effects=("write",),
            ),
            inspect_call(), plan_call(), plan_pass(),
        ])

        chat_result = runtime.chat("Create a polished candidate")
        goal = runtime.active_goal()
        self.assertEqual(goal.objective, "Create a polished candidate")
        self.assertEqual(goal.metadata["execution_policy"]["entry_surface"], "chat")
        self.assertEqual(goal.status, GoalStatus.AWAITING_PLAN_APPROVAL)
        self.assertFalse((self.workspace / "candidate.txt").exists())
        self.assertEqual(chat_result.status, PlanStatus.PENDING_APPROVAL)

    def test_goal_runtime_persists_policy_contract_projection_and_plan_quality_target(self):
        runtime, provider = self.runtime([inspect_call(), plan_call(), plan_pass()])
        plan = runtime.start_goal("Build persistent behavior")
        goal = runtime.active_goal()
        self.assertEqual(goal.metadata["weak_model_policy"]["version"], 1)
        self.assertEqual(goal.metadata["goal_contract"]["original_objective"], "Build persistent behavior")
        self.assertTrue(goal.metadata["goal_contract_fingerprint"])
        self.assertTrue(all("GOAL_CONTRACT_PROJECTION" in call.conversation[0]["content"] for call in provider.calls))

        runtime.approve_plan(plan.revision)
        goal = runtime.active_goal()
        self.assertEqual(goal.metadata["quality_target"]["dimensions"][0]["description"], plan.tasks[0].acceptance_criteria[0])
        self.assertEqual(goal.metadata["convergence_state"], "not_evaluated")
        events = self.store.list_recent_events(goal.id, limit=100)
        self.assertTrue(any(event.event_type == "goal_contract.projected" for event in events))
        self.assertTrue(any(event.event_type == "quality_target.created" for event in events))

    def test_visual_goal_without_vision_evaluator_completes_with_limitations(self):
        html_plan = plan_call()
        html_plan["tool_calls"][0]["args"]["expected_changes"][0]["path"] = "index.html"
        runtime, provider = self.runtime([
            inspect_call(), html_plan, plan_pass(),
            {"tool_calls": [
                {"id": "preview", "name": "preview_html", "args": {"path": "index.html", "open_browser": False, "settle_ms": 0}},
                finish_call(),
            ]},
            review_pass(),
        ])
        plan = runtime.start_goal("Create a polished visual page")
        runtime.approve_plan(plan.revision)
        (self.workspace / "index.html").write_text(
            "<!doctype html><html><body>verified</body></html>",
            encoding="utf-8",
        )
        runtime.update_task_from_user("T001", "done", "Structural and runtime checks passed")

        result = runtime.run_slice(steps=1)

        self.assertTrue(result.completed)
        goal = self.store.get_goal(plan.goal_id)
        self.assertEqual(goal.status, GoalStatus.COMPLETED)
        self.assertEqual(goal.metadata["convergence_state"], "converged_with_limitations")
        self.assertEqual(goal.metadata["completion_disposition"], "completed_with_limitations")
        self.assertTrue(goal.metadata["completion_limitations"])
        self.assertTrue(goal.metadata["latest_evaluation"]["scores"])

    def test_mode_changes_are_locked_without_mutating_the_durable_run_contract(self):
        runtime, _provider = self.runtime([inspect_call(), plan_call(), plan_pass()])
        plan = runtime.start_goal("Build persistent behavior")
        runtime.approve_plan(plan.revision)
        before = runtime.active_goal()
        run_id = before.metadata["run_id"]
        fingerprint = before.metadata["goal_contract_fingerprint"]

        # Compatibility aliases that normalize to the already-bound Normal
        # mode are idempotent. Material changes are rejected while running.
        for mode in ("chat", "goal", "normal"):
            self.assertEqual(runtime.transition_mode(mode), "normal")
        for mode in ("plan", "ultra"):
            with self.assertRaisesRegex(RuntimeStateError, "locked"):
                runtime.transition_mode(mode)

        after = runtime.active_goal()
        session = self.store.get_workflow_session(runtime.session_id)
        self.assertEqual(after.id, before.id)
        self.assertEqual(after.metadata["run_id"], run_id)
        self.assertEqual(after.metadata["goal_contract_fingerprint"], fingerprint)
        self.assertEqual(session["session_mode"], "normal")
        transitions = [event for event in self.store.list_recent_events(after.id, limit=100) if event.event_type == "mode.transition"]
        self.assertEqual(transitions, [])

    def test_ultra_can_visit_plan_and_change_depth_without_replacing_the_run(self):
        runtime, _provider = self.runtime([])
        runtime.transition_mode("ultra")
        runtime.transition_mode("plan")
        session = self.store.get_workflow_session(runtime.session_id)
        self.assertEqual(session["session_mode"], "plan")
        self.assertEqual(session["state"]["interaction_mode"], "plan")
        self.assertEqual(session["state"]["minimum_strategy"], "recursive")

        runtime.transition_mode("normal")
        self.assertEqual(
            self.store.get_workflow_session(runtime.session_id)["session_mode"],
            "normal",
        )

        session = self.store.get_workflow_session(runtime.session_id)
        self.assertEqual(session["state"]["interaction_mode"], "working")
        self.assertEqual(session["state"]["minimum_strategy"], "recursive")

    def test_normal_can_visit_plan_and_return_to_normal(self):
        runtime, _provider = self.runtime([])
        runtime.transition_mode("plan")
        runtime.transition_mode("normal")
        self.assertEqual(
            self.store.get_workflow_session(runtime.session_id)["session_mode"],
            "normal",
        )

    def test_short_visual_feedback_creates_delta_refinement_on_same_run_and_index(self):
        html_plan = plan_call()
        html_plan["tool_calls"][0]["args"]["expected_changes"][0]["path"] = "index.html"
        runtime, _provider = self.runtime([inspect_call(), html_plan, plan_pass()])
        plan = runtime.start_goal("Create a detailed castle scene")
        runtime.approve_plan(plan.revision)
        (self.workspace / "index.html").write_text(
            '<section id="castle"><style>#castle{color:gray}@keyframes ramStrike{to{transform:translateX(2px)}}</style></section>',
            encoding="utf-8",
        )
        before = runtime.active_goal()

        runtime.add_guidance("The graphics are weak.")

        after = runtime.active_goal()
        self.assertEqual(after.id, before.id)
        self.assertEqual(after.active_plan_revision, plan.revision)
        self.assertEqual(after.metadata["convergence_state"], "refining")
        action = after.metadata["refinement_actions"][-1]
        self.assertEqual(action["affected_dimensions"], ["requirement_completeness"])
        self.assertEqual(action["affected_components"], [])
        self.assertIn("repository_context_slice", action)
        self.assertEqual(action["repository_context_slice"]["query"], "The graphics are weak.")
        self.assertEqual(action["repository_context_slice"]["size_chars"], 0)
        self.assertEqual(after.metadata["goal_contract"]["user_feedback"][-1], "The graphics are weak.")

    def test_fourth_equivalent_failed_tool_approach_is_blocked_by_persisted_policy(self):
        failures = {
            "tool_calls": [
                {"id": f"missing-{index}", "name": "read_file", "args": {"path": "missing.py"}}
                for index in range(4)
            ]
        }
        runtime, _provider = self.runtime([inspect_call(), plan_call(), plan_pass(), failures])
        plan = runtime.start_goal("Build persistent behavior")
        goal_id = runtime.active_goal().id
        runtime.approve_plan(plan.revision)

        runtime.run_slice(steps=1)

        attempts = self.store.list_actions(goal_id)
        self.assertEqual(len([item for item in attempts if item["tool_name"] == "read_file"]), 2)
        goal = self.store.get_goal(goal_id)
        self.assertEqual(len(goal.metadata["failed_attempts"]), 2)
        events = self.store.list_recent_events(goal_id, limit=100)
        self.assertTrue(any(event.event_type == "approach.change_forced" for event in events))

    def test_goal_mutation_creates_hash_bound_change_set_and_invalidates_evaluation(self):
        write_turn = {"tool_calls": [{
            "id": "write", "name": "write_file",
            "args": {"path": "artifact.txt", "content": "improved\n"},
        }]}
        runtime, _provider = self.runtime([inspect_call(), plan_call(), plan_pass(), write_turn])
        plan = runtime.start_goal("Build persistent behavior")
        goal_id = runtime.active_goal().id
        runtime.approve_plan(plan.revision)

        runtime.run_slice(steps=1)

        goal = self.store.get_goal(goal_id)
        self.assertEqual(goal.metadata["mutation_sequence"], 1)
        self.assertTrue(goal.metadata["latest_evaluation_stale"])
        change_set = goal.metadata["goal_change_sets"][-1]
        self.assertEqual(change_set["changed_files"], ["artifact.txt"])
        self.assertIsNone(change_set["pre_hashes"]["artifact.txt"])
        self.assertTrue(change_set["post_hashes"]["artifact.txt"])
        self.assertIn("+improved", change_set["diff"])
        self.assertEqual(change_set["review_status"], "pending")

    def test_harness_activates_ready_task_and_binds_tool_evidence_without_model_bookkeeping(self):
        write_turn = {"tool_calls": [{
            "id": "write-without-start",
            "name": "write_file",
            "args": {"path": "artifact.txt", "content": "proved\n"},
        }]}
        runtime, _provider = self.runtime([inspect_call(), plan_call(), plan_pass(), write_turn])
        plan = runtime.start_goal("Build persistent behavior")
        goal_id = runtime.active_goal().id
        runtime.approve_plan(plan.revision)

        runtime.run_slice(steps=1)

        task = self.store.list_tasks(goal_id, plan.revision)[0]
        self.assertEqual(task.status, TaskStatus.IN_PROGRESS)
        evidence = self.store.list_evidence(goal_id, task_id="T001")
        self.assertTrue(any(item.verified and item.data.get("tool") == "write_file" for item in evidence))
        events = self.store.list_recent_events(goal_id, limit=100)
        selected = [item for item in events if item.event_type == "execution.task_selected"]
        self.assertTrue(selected)
        self.assertTrue(selected[-1].payload["activated"])
        actions = self.store.list_actions(goal_id)
        write_action = next(
            item for item in actions if item["tool_name"] == "write_file"
        )
        self.assertEqual(
            json.loads(write_action["args_json"])["_harness_task_id"],
            "T001",
        )

    def test_denied_resource_call_does_not_poison_the_later_owning_task(self):
        candidate = dependency_plan_call()
        args = candidate["tool_calls"][0]["args"]
        args["expected_changes"] = [
            {
                "path": "first.txt",
                "intent": "Create the first task artifact.",
                "basis": "repository_convention",
                "evidence_refs": ["inspection:I001"],
                "supports_tasks": ["T001"],
            },
            {
                "path": "second.txt",
                "intent": "Create the second task artifact.",
                "basis": "repository_convention",
                "evidence_refs": ["inspection:I001"],
                "supports_tasks": ["T002"],
            },
        ]
        runtime, _provider = self.runtime(
            [inspect_call(), candidate, plan_pass()]
        )
        plan = runtime.start_goal("Build two dependency-ordered artifacts.")
        goal = runtime.active_goal()
        runtime.approve_plan(plan.revision)
        active_plan, active = runtime._activate_ready_task(
            self.store.get_goal(goal.id), runtime.latest_plan()
        )
        self.assertEqual(active.id, "T001")
        call = ToolCall(
            "write-second", "write_file",
            {"path": "second.txt", "content": "owned by T002\n"},
        )

        first_denial = runtime._execute_workspace_tool(
            self.store.get_goal(goal.id), call,
            task_id="T001", actor="coordinator",
        )
        second_denial = runtime._execute_workspace_tool(
            self.store.get_goal(goal.id), call,
            task_id="T001", actor="coordinator",
        )
        self.assertIn("not covered", first_denial)
        self.assertIn("not covered", second_denial)

        self.store.transition_task(
            goal.id, active_plan.revision, "T001", TaskStatus.COMPLETED,
            note="User verified the first task.",
            evidence=["User evidence: first task is complete."],
            actor="user",
        )
        _plan, active = runtime._activate_ready_task(
            self.store.get_goal(goal.id), runtime.latest_plan()
        )
        self.assertEqual(active.id, "T002")
        result = runtime._execute_workspace_tool(
            self.store.get_goal(goal.id), call,
            task_id="T002", actor="coordinator",
        )

        self.assertIn("Wrote", result)
        self.assertEqual(
            (self.workspace / "second.txt").read_text(encoding="utf-8"),
            "owned by T002\n",
        )

    def test_fresh_dependency_command_evidence_completes_verification_only_task(self):
        candidate = dependency_plan_call()
        args = candidate["tool_calls"][0]["args"]
        args["semantic_goal"]["acceptance_criteria"].append(
            "python -c \"print('verified')\" exits successfully."
        )
        args["applicability_evidence"][0]["supports_tasks"] = [
            "T001", "T002", "T003"
        ]
        args["expected_changes"] = [
            {
                "path": "first.txt",
                "intent": "Create the first implementation artifact.",
                "basis": "repository_convention",
                "evidence_refs": ["inspection:I001"],
                "supports_tasks": ["T001"],
            },
            {
                "path": "second.txt",
                "intent": "Create the second implementation artifact.",
                "basis": "repository_convention",
                "evidence_refs": ["inspection:I001"],
                "supports_tasks": ["T002"],
            },
        ]
        args["tasks"].append(
            {
                "id": "T003",
                "title": "Verify the integrated result",
                "description": "Run python -c \"print('verified')\".",
                "acceptance_criteria": [
                    "python -c \"print('verified')\" exits successfully."
                ],
                "verification": [
                    "Run python -c \"print('verified')\" and require exit code 0."
                ],
                "depends_on": ["T001", "T002"],
                "risk": "low",
            }
        )
        runtime, _provider = self.runtime(
            [inspect_call(), candidate, plan_pass()]
        )
        plan = runtime.start_goal("Build and verify two ordered artifacts.")
        goal = runtime.active_goal()
        runtime.approve_plan(plan.revision)

        active_plan, active = runtime._activate_ready_task(
            self.store.get_goal(goal.id), runtime.latest_plan()
        )
        self.assertEqual(active.id, "T001")
        self.store.transition_task(
            goal.id, active_plan.revision, "T001", TaskStatus.COMPLETED,
            note="User verified the first task.",
            evidence=["User evidence: first task is complete."],
            actor="user",
        )
        active_plan, active = runtime._activate_ready_task(
            self.store.get_goal(goal.id), runtime.latest_plan()
        )
        self.assertEqual(active.id, "T002")
        command = "python -c \"print('verified')\""
        command_result = runtime._execute_workspace_tool(
            self.store.get_goal(goal.id),
            ToolCall("verify-early", "run_command", {"command": command}),
            task_id="T002", actor="coordinator",
        )
        self.assertIn("exit code: 0", command_result)
        self.assertIn(
            "completed",
            runtime._control_update_task(
                self.store.get_goal(goal.id), runtime.latest_plan(),
                {
                    "task_id": "T002", "status": "done",
                    "note": "Dependency verification passed.",
                    "evidence": ["The command exited with code 0."],
                },
            ),
        )
        active_plan, active = runtime._activate_ready_task(
            self.store.get_goal(goal.id), runtime.latest_plan()
        )
        self.assertEqual(active.id, "T003")

        result = runtime._control_update_task(
            self.store.get_goal(goal.id), active_plan,
            {
                "task_id": "T003", "status": "done",
                "note": "The already-run exact verification is still fresh.",
                "evidence": ["The exact required command exited with code 0."],
            },
        )

        self.assertIn("completed", result)
        evidence = self.store.list_evidence(goal.id, task_id="T003")
        inherited = [
            item for item in evidence if item.kind == "inherited_verification"
        ]
        self.assertEqual(len(inherited), 1)
        self.assertEqual(inherited[0].data["source_task_id"], "T002")

    def test_model_cannot_complete_task_with_prose_before_bound_authoritative_evidence(self):
        done_without_tool = {"tool_calls": [task_update("done", ["I verified it"], "claim only")]}
        runtime, _provider = self.runtime([inspect_call(), plan_call(), plan_pass(), done_without_tool])
        plan = runtime.start_goal("Build persistent behavior")
        goal_id = runtime.active_goal().id
        runtime.approve_plan(plan.revision)

        runtime.run_slice(steps=1)

        task = self.store.list_tasks(goal_id, plan.revision)[0]
        self.assertEqual(task.status, TaskStatus.IN_PROGRESS)
        self.assertEqual(self.store.list_evidence(goal_id, task_id="T001"), ())

    def test_finish_goal_is_rejected_while_quality_target_is_below_target(self):
        runtime, provider = self.runtime([inspect_call(), plan_call(), plan_pass(), {"tool_calls": [finish_call()]}])
        plan = runtime.start_goal("Build persistent behavior")
        goal_id = runtime.active_goal().id
        runtime.approve_plan(plan.revision)
        runtime.update_task_from_user("T001", "done", "Focused verification failed the quality threshold")
        self.store.update_goal_metadata(goal_id, convergence_state="below_target")

        result = runtime.run_slice(steps=1)

        self.assertFalse(result.completed)
        self.assertEqual(self.store.get_goal(goal_id).status, GoalStatus.RUNNING)
        self.assertFalse(any("independent final reviewer" in call.system for call in provider.calls))
        events = self.store.list_recent_events(goal_id, limit=100)
        self.assertFalse(any(event.event_type == "quality_convergence.decided" for event in events))

    def test_repeated_identical_invalid_plan_stops_without_burning_repair_budget(self):
        runtime, provider = self.runtime(
            [inspect_call(), *(invalid_plan_call(index) for index in range(1, 5))]
        )

        plan = runtime.start_goal("Build persistent behavior")

        self.assertIsNone(plan)
        self.assertEqual(runtime.active_goal().status, GoalStatus.PAUSED)
        checkpoints = [
            event
            for event in self.store.list_recent_events(runtime.active_goal().id, limit=100)
            if event.event_type == "planning.checkpoint"
        ]
        self.assertEqual(checkpoints[-1].payload["format_attempts"], 1)
        self.assertIn("contract_incompatibility", checkpoints[-1].payload["reason"])
        self.assertEqual(provider.remaining, 2)

    def test_three_distinct_plan_contract_failures_use_three_independent_repairs(self):
        invalid_turns = []
        for index in range(1, 5):
            turn = invalid_plan_call(index)
            turn["tool_calls"][0]["args"]["summary"] = f"Distinct malformed attempt {index}"
            invalid_turns.append(turn)
        runtime, provider = self.runtime([inspect_call(), *invalid_turns])

        self.assertIsNone(runtime.start_goal("Exercise independent plan repairs"))

        checkpoints = [
            event
            for event in self.store.list_recent_events(runtime.active_goal().id, limit=100)
            if event.event_type == "planning.checkpoint"
        ]
        self.assertEqual(checkpoints[-1].payload["format_attempts"], 3)
        self.assertEqual(provider.remaining, 1)

    def test_repeated_identical_applicability_failure_stops_without_budget_burn(self):
        runtime, provider = self.runtime(
            [inspect_call(), *(invalid_evidence_plan_call(index) for index in range(1, 5))]
        )

        self.assertIsNone(runtime.start_goal("Build from inspected workspace evidence"))

        self.assertEqual(runtime.active_goal().status, GoalStatus.PAUSED)
        checkpoints = [
            event
            for event in self.store.list_recent_events(runtime.active_goal().id, limit=100)
            if event.event_type == "planning.checkpoint"
        ]
        self.assertEqual(checkpoints[-1].payload["format_attempts"], 0)
        self.assertEqual(checkpoints[-1].payload["applicability_attempts"], 1)
        self.assertIn("tool:missing-inspection", checkpoints[-1].payload["technical_detail"])
        self.assertEqual(provider.remaining, 2)

    def test_one_turn_with_duplicate_invalid_proposals_stops_at_first_repeat(self):
        burst = {
            "tool_calls": [
                invalid_plan_call(index)["tool_calls"][0]
                for index in range(1, 7)
            ]
        }
        runtime, provider = self.runtime([inspect_call(), burst])

        self.assertIsNone(runtime.start_goal("Bound malformed proposals in one response"))

        checkpoints = [
            event
            for event in self.store.list_recent_events(runtime.active_goal().id, limit=100)
            if event.event_type == "planning.checkpoint"
        ]
        self.assertEqual(checkpoints[-1].payload["format_attempts"], 1)
        provider.assert_exhausted()

    def test_plan_pauses_for_revision_bound_user_approval(self):
        runtime, provider = self.runtime([inspect_call(), plan_call(), plan_pass()])
        plan = runtime.start_goal("Build persistent behavior")
        goal = runtime.active_goal()
        self.assertEqual(goal.status, GoalStatus.AWAITING_PLAN_APPROVAL)
        self.assertEqual(plan.status, PlanStatus.PENDING_APPROVAL)
        self.assertIsNone(goal.active_plan_revision)
        planning_actions = self.store.list_actions(goal.id)
        self.assertEqual(len(planning_actions), 1)
        self.assertEqual(planning_actions[0]["tool_name"], "list_files")
        critic_request = next(call for call in provider.calls if "fresh-context critic" in call.system)
        critic_context = "\n".join(
            str(message.get("content", "")) for message in critic_request.conversation
        )
        self.assertIn("successful_workspace_inspections", critic_context)
        self.assertIn("inspect-workspace", critic_context)
        runtime.approve_plan(plan.revision)
        self.assertEqual(runtime.active_goal().status, GoalStatus.RUNNING)
        provider.assert_exhausted()

    def test_planner_must_see_prior_inspection_result_before_plan_is_accepted(self):
        same_turn = {
            "tool_calls": [
                inspect_call()["tool_calls"][0],
                plan_call()["tool_calls"][0],
            ]
        }
        runtime, provider = self.runtime([same_turn, plan_call(), plan_pass()])

        plan = runtime.start_goal("Build persistent behavior from real workspace evidence")

        self.assertIsNotNone(plan)
        self.assertEqual(runtime.active_goal().status, GoalStatus.AWAITING_PLAN_APPROVAL)
        retry_context = "\n".join(
            str(message.get("content", ""))
            for message in provider.calls[1].conversation
        )
        self.assertIn("semantic goal must cite successful repository inspection", retry_context)
        provider.assert_exhausted()

    def test_ollama_style_placeholder_source_binds_to_stable_inspection_reference(self):
        repeated_inspection = inspect_call()
        repeated_inspection["tool_calls"][0]["id"] = "inspect-again"
        placeholder_plan = plan_call()
        placeholder_plan["tool_calls"][0]["args"]["applicability_evidence"][0]["source"] = (
            "tool:CALL_ID"
        )
        runtime, provider = self.runtime(
            [inspect_call(), repeated_inspection, placeholder_plan, plan_pass()]
        )

        plan = runtime.start_goal("Create one verified file in the empty workspace")

        self.assertIsNotNone(plan)
        self.assertEqual(plan.applicability_evidence[0]["source"], "inspection:I001")
        self.assertEqual(len(self.store.list_actions(runtime.active_goal().id)), 1)
        inspection_events = [
            event
            for event in self.store.list_recent_events(runtime.active_goal().id, limit=100)
            if event.event_type == "planning.inspection_recorded"
        ]
        self.assertEqual(len(inspection_events), 1)
        self.assertEqual(inspection_events[0].payload["reference"], "inspection:I001")
        critic_request = next(call for call in provider.calls if "fresh-context critic" in call.system)
        critic_context = "\n".join(
            str(message.get("content", "")) for message in critic_request.conversation
        )
        self.assertIn("inspection:I001", critic_context)
        provider.assert_exhausted()

    def test_prose_done_never_completes_persistent_goal(self):
        runtime, provider = self.runtime([inspect_call(), plan_call(), plan_pass(), "Everything is done.", "Done for sure."])
        plan = runtime.start_goal("Build persistent behavior")
        runtime.approve_plan(plan.revision)
        result = runtime.run_slice(steps=3)
        self.assertFalse(result.completed)
        self.assertEqual(runtime.active_goal().status, GoalStatus.RUNNING)
        self.assertEqual(self.store.list_tasks(runtime.active_goal().id, 1)[0].status, TaskStatus.IN_PROGRESS)
        provider.assert_exhausted()

    def test_bounded_planner_failure_pauses_with_a_resumable_goal(self):
        provider = ScriptedProvider(
            ["I should inspect more.", "Here is a prose-only plan.", inspect_call(), plan_call(), plan_pass()]
        )
        runtime = AgentRuntime(
            provider,
            self.store,
            self.workspace,
            config=replace(self.config, planning_steps=2),
            sleeper=lambda _seconds: None,
        )

        plan = runtime.start_goal("Build a complex persistent system")

        goal = runtime.active_goal()
        self.assertIsNone(plan)
        self.assertEqual(goal.status, GoalStatus.PAUSED)
        self.assertEqual(goal.metadata["resume_status"], GoalStatus.DISCOVERING.value)
        self.assertIn("critic-approved structured plan", goal.metadata["waiting_question"])
        retried = runtime.reject_plan("Use one concise, directly verifiable task.")
        self.assertIsNotNone(retried)
        self.assertEqual(runtime.active_goal().status, GoalStatus.AWAITING_PLAN_APPROVAL)
        provider.assert_exhausted()

    def test_short_single_artifact_goal_recovers_from_prose_only_weak_planner(self):
        runtime, provider = self.runtime(
            [inspect_call(), "I will create the file after planning."]
        )

        plan = runtime.start_goal(
            "Create counter.html with a visible counter and verify it in a browser."
        )

        self.assertIsNone(plan)
        self.assertEqual(runtime.active_goal().status, GoalStatus.PAUSED)
        events = self.store.list_recent_events(runtime.active_goal().id, limit=100)
        self.assertTrue(any(event.event_type == "planning.checkpoint" for event in events))
        self.assertIsNone(runtime.active_goal().active_plan_revision)
        provider.assert_exhausted()

    def test_planning_provider_exhaustion_pauses_instead_of_stranding_phase(self):
        runtime, provider = self.runtime([])

        plan = runtime.start_goal("Build a system despite a temporary provider outage")

        goal = runtime.active_goal()
        self.assertIsNone(plan)
        self.assertEqual(goal.status, GoalStatus.PAUSED)
        self.assertEqual(goal.metadata["resume_status"], GoalStatus.DISCOVERING.value)
        self.assertIn("provider retries were exhausted", goal.metadata["waiting_question"])
        self.assertTrue(goal.metadata["auto_retryable"])
        provider.assert_exhausted()

    def test_provider_failure_schedules_durable_retry_then_recovers(self):
        def fail_once(_request):
            raise RuntimeError("temporary provider outage")

        provider = ScriptedProvider(
            [
                inspect_call(),
                plan_call(),
                plan_pass(),
                fail_once,
                {"tool_calls": [
                    {"id": "retry-verify", "name": "list_files", "args": {"path": "."}},
                    task_update("done", ["retry recovered"], "retry recovered"),
                ]},
            ]
        )
        runtime = AgentRuntime(
            provider,
            self.store,
            self.workspace,
            config=replace(
                self.config,
                max_provider_retries=0,
                goal_retry_base_ms=0,
                goal_retry_max_ms=0,
            ),
            sleeper=lambda _seconds: None,
            approval=lambda _name, _args, _risk: True,
        )
        plan = runtime.start_goal("Survive provider outages until the goal is reached")
        runtime.approve_plan(plan.revision)

        first = runtime.run_slice(steps=1)
        self.assertEqual(first.status, GoalStatus.RUNNING.value)
        self.assertFalse(first.needs_user)
        self.assertEqual(runtime.active_goal().metadata["goal_attempt"], 1)

        runtime.wait_for_scheduled_retry()
        second = runtime.run_slice(steps=1)
        self.assertEqual(second.status, GoalStatus.RUNNING.value)
        self.assertEqual(
            self.store.list_tasks(runtime.active_goal().id, plan.revision)[0].status,
            TaskStatus.COMPLETED,
        )
        self.assertEqual(runtime.active_goal().metadata["consecutive_retries"], 0)
        provider.assert_exhausted()

    def test_repeated_provider_failures_pause_with_actionable_recovery(self):
        def fail(_request):
            raise RuntimeError("bad model or credentials")

        provider = ScriptedProvider(
            [inspect_call(), plan_call(), plan_pass(), fail, fail]
        )
        runtime = AgentRuntime(
            provider,
            self.store,
            self.workspace,
            config=replace(
                self.config,
                max_provider_retries=0,
                provider_failure_limit=2,
                goal_retry_base_ms=0,
                goal_retry_max_ms=0,
            ),
            sleeper=lambda _seconds: None,
            approval=lambda _name, _args, _risk: True,
        )
        plan = runtime.start_goal("Pause when provider repair is required")
        runtime.approve_plan(plan.revision)

        first = runtime.run_slice(steps=1)
        self.assertEqual(first.status, GoalStatus.RUNNING.value)
        runtime.wait_for_scheduled_retry()
        second = runtime.run_slice(steps=1)

        self.assertEqual(second.status, GoalStatus.PAUSED.value)
        self.assertTrue(second.needs_user)
        goal = runtime.active_goal()
        self.assertFalse(goal.metadata["auto_retryable"])
        self.assertEqual(goal.metadata["retry_after_ms"], 0)
        self.assertIn("Check the selected model", goal.metadata["waiting_question"])
        provider.assert_exhausted()

    def test_auto_mode_retries_transient_planning_failures_until_plan_boundary(self):
        def fail(_request):
            raise RuntimeError("temporary planning outage")

        provider = ScriptedProvider(
            [fail, fail, inspect_call(), plan_call(), plan_pass()]
        )
        runtime = AgentRuntime(
            provider,
            self.store,
            self.workspace,
            config=replace(
                self.config,
                max_provider_retries=0,
                goal_retry_base_ms=0,
                goal_retry_max_ms=0,
            ),
            sleeper=lambda _seconds: None,
        )
        self.assertIsNone(runtime.start_goal("Keep planning through transient outages"))
        self.assertEqual(runtime.active_goal().status, GoalStatus.PAUSED)
        output = io.StringIO()

        _run_auto(runtime, ConsoleUI(stream=output, color=False))

        self.assertEqual(runtime.active_goal().status, GoalStatus.AWAITING_PLAN_APPROVAL)
        self.assertIn("durable retry policy", output.getvalue())
        provider.assert_exhausted()

    def test_repeated_no_progress_slices_self_reprompt_without_retry_limit(self):
        provider = ScriptedProvider([
            inspect_call(), plan_call(), plan_pass(),
            "done", "still done", "still no action",
        ])
        runtime = AgentRuntime(
            provider,
            self.store,
            self.workspace,
            config=replace(self.config, no_action_limit=1, stalled_slice_limit=2),
            sleeper=lambda _seconds: None,
            approval=lambda _name, _args, _risk: True,
        )
        plan = runtime.start_goal("Build persistent behavior")
        runtime.approve_plan(plan.revision)
        self.assertEqual(runtime.run_slice(steps=1).status, GoalStatus.RUNNING.value)
        second = runtime.run_slice(steps=1)
        self.assertEqual(second.status, GoalStatus.RUNNING.value)
        self.assertFalse(second.needs_user)
        third = runtime.run_slice(steps=1)
        self.assertEqual(third.status, GoalStatus.RUNNING.value)
        goal = runtime.active_goal()
        self.assertEqual(goal.metadata["goal_attempt"], 2)
        self.assertTrue(goal.metadata["auto_retryable"])
        second_attempt_context = "\n".join(
            str(message.get("content", ""))
            for message in provider.calls[-1].conversation
        )
        self.assertIn("SELF-RETRY ATTEMPT 1", second_attempt_context)
        provider.assert_exhausted()

    def test_completion_requires_task_evidence_and_independent_review(self):
        runtime, provider = self.runtime(
            [
                inspect_call(),
                plan_call(),
                plan_pass(),
                {"tool_calls": [task_update("in_progress", note="starting")]},
                {
                    "tool_calls": [
                        {"id": "verify-list", "name": "list_files", "args": {"path": "."}},
                        task_update("done", ["focused tests passed"], "implemented and tested"),
                        finish_call(),
                    ]
                },
                review_pass(),
            ]
        )
        plan = runtime.start_goal("Build persistent behavior")
        goal_id = runtime.active_goal().id
        runtime.approve_plan(plan.revision)
        result = runtime.run_slice()
        self.assertTrue(result.completed)
        self.assertEqual(runtime.store.get_goal(goal_id).status, GoalStatus.COMPLETED)
        self.assertTrue(
            any(item.kind == "final_review" and item.verified for item in self.store.list_evidence(goal_id))
        )
        evaluation = runtime.store.get_goal(goal_id).metadata["latest_evaluation"]
        self.assertEqual(evaluation["mutation_sequence"], 0)
        self.assertTrue(evaluation["artifact_hashes"])
        self.assertTrue(evaluation["scores"])
        self.assertTrue(all(score["evidence_ids"] for score in evaluation["scores"]))
        reviewer_request = next(call for call in provider.calls if "independent final reviewer" in call.system)
        reviewer_context = "\n".join(str(message.get("content", "")) for message in reviewer_request.conversation)
        self.assertIn("acceptance_criteria", reviewer_context)
        self.assertIn("survives restart and tests pass", reviewer_context)
        provider.assert_exhausted()

    def test_passing_review_must_explicitly_cover_every_accepted_task(self):
        complete_review = {
            "tool_calls": [
                {
                    "id": "review-complete",
                    "name": "submit_review",
                    "args": {
                        "verdict": "pass",
                        "summary": "Both accepted tasks are directly evidenced.",
                        "issues": [],
                        "checked_task_ids": ["T001", "T002"],
                    },
                }
            ]
        }
        runtime, provider = self.runtime(
            [
                inspect_call(),
                dependency_plan_call(),
                plan_pass(),
                {"tool_calls": [finish_call()]},
                review_pass(),
                complete_review,
            ]
        )
        plan = runtime.start_goal("Build and integrate persistent behavior")
        runtime.approve_plan(plan.revision)
        runtime.update_task_from_user("T001", "done", "Restart behavior passed focused tests.")
        runtime.update_task_from_user("T002", "done", "Integration behavior passed its test.")

        result = runtime.run_slice()

        self.assertTrue(result.completed)
        reviewer_calls = [
            call for call in provider.calls if "independent final reviewer" in call.system
        ]
        self.assertEqual(len(reviewer_calls), 2)
        rejection_context = "\n".join(
            str(message.get("content", ""))
            for message in reviewer_calls[1].conversation
        )
        self.assertIn("pass must explicitly cover every accepted task", rejection_context)
        provider.assert_exhausted()

    def test_final_review_reserves_its_last_turn_for_a_verdict(self):
        inspect_evidence = {
            "tool_calls": [
                {
                    "id": "inspect-evidence",
                    "name": "inspect_task",
                    "args": {
                        "task_id": "T001",
                        "evidence_offset": 0,
                        "evidence_limit": 10,
                    },
                }
            ]
        }
        runtime, provider = self.runtime(
            [
                inspect_call(),
                plan_call(),
                plan_pass(),
                {"tool_calls": [finish_call()]},
                inspect_evidence,
                inspect_evidence,
                inspect_evidence,
                review_pass(),
            ]
        )
        plan = runtime.start_goal("Build persistent behavior")
        runtime.approve_plan(plan.revision)
        runtime.update_task_from_user(
            "T001",
            "done",
            "The requested behavior and focused verification passed.",
        )

        result = runtime.run_slice()

        self.assertTrue(result.completed)
        reviewer_calls = [
            call for call in provider.calls
            if "independent final reviewer" in call.system
        ]
        self.assertEqual(len(reviewer_calls), 4)
        final_tools = {
            item["function"]["name"] for item in reviewer_calls[-1].tools
        }
        self.assertEqual(final_tools, {"submit_review"})
        final_context = "\n".join(
            str(message.get("content", ""))
            for message in reviewer_calls[-1].conversation
        )
        self.assertIn("FINAL REVIEW VERDICT TURN", final_context)
        provider.assert_exhausted()

    def test_user_can_update_checklist_mid_goal_and_new_revision_reapproval_is_mandatory(self):
        runtime, provider = self.runtime([inspect_call(), plan_call(), plan_pass()])
        first = runtime.start_goal("Build persistent behavior")
        goal_id = runtime.active_goal().id
        runtime.approve_plan(first.revision)
        runtime.update_task_from_user("T001", "done", "User ran the focused restart test.")
        second = runtime.add_user_task("Audit Windows interruption", "Interrupted writes are never replayed.")
        self.assertEqual(second.revision, 2)
        self.assertEqual(second.status, PlanStatus.PENDING_APPROVAL)
        self.assertEqual(self.store.get_goal(goal_id).status, GoalStatus.AWAITING_PLAN_APPROVAL)
        carried = [
            item
            for item in self.store.list_evidence(goal_id, task_id="T001")
            if item.plan_revision == 2
        ]
        self.assertTrue(carried)
        runtime.approve_plan(2)
        self.assertEqual(self.store.get_goal(goal_id).active_plan_revision, 2)
        self.assertEqual(self.store.get_latest_plan(goal_id).tasks[0].status, TaskStatus.COMPLETED)
        provider.assert_exhausted()

    def test_reopening_prerequisite_invalidates_completed_dependants(self):
        runtime, provider = self.runtime([inspect_call(), dependency_plan_call(), plan_pass()])
        plan = runtime.start_goal("Build dependency-aware behavior")
        runtime.approve_plan(plan.revision)
        runtime.update_task_from_user("T001", "done", "unit test passed")
        runtime.update_task_from_user("T002", "done", "integration test passed")
        runtime.update_task_from_user("T001", "pending")
        statuses = {task.id: task.status for task in runtime.latest_plan().tasks}
        self.assertEqual(statuses, {"T001": TaskStatus.PENDING, "T002": TaskStatus.PENDING})
        provider.assert_exhausted()

    def test_invalid_checklist_edit_does_not_strand_accepted_plan(self):
        runtime, provider = self.runtime([inspect_call(), plan_call(), plan_pass()])
        plan = runtime.start_goal("Build persistent behavior")
        runtime.approve_plan(plan.revision)
        with self.assertRaisesRegex(Exception, "task not found"):
            runtime.revise_plan(reason="bad user edit", edit=("MISSING", "new text"))
        self.assertEqual(runtime.active_goal().status, GoalStatus.RUNNING)
        self.assertEqual(runtime.latest_plan().revision, 1)
        self.assertEqual(runtime.latest_plan().status, PlanStatus.ACCEPTED)
        provider.assert_exhausted()

    def test_field_edit_updates_approval_bound_criteria(self):
        runtime, provider = self.runtime([inspect_call(), plan_call(), plan_pass()])
        plan = runtime.start_goal("Build persistent behavior")
        runtime.approve_plan(plan.revision)
        revised = runtime.revise_plan(
            reason="user strengthened evidence requirements",
            edit=("T001", "accept", "First proof || Second proof"),
        )
        self.assertEqual(revised.tasks[0].acceptance_criteria, ("First proof", "Second proof"))
        self.assertEqual(revised.status, PlanStatus.PENDING_APPROVAL)
        self.assertEqual(runtime.active_goal().status, GoalStatus.AWAITING_PLAN_APPROVAL)
        provider.assert_exhausted()

    def test_replan_while_running_preserves_old_accepted_plan_until_new_approval(self):
        runtime, provider = self.runtime(
            [inspect_call(), plan_call(), plan_pass(), inspect_call(), plan_call(), plan_pass()]
        )
        first = runtime.start_goal("Build persistent behavior")
        goal_id = runtime.active_goal().id
        runtime.approve_plan(first.revision)
        second = runtime.reject_plan("Split verification more clearly")
        self.assertEqual(second.revision, 2)
        self.assertEqual(second.status, PlanStatus.PENDING_APPROVAL)
        self.assertEqual(runtime.active_goal().status, GoalStatus.AWAITING_PLAN_APPROVAL)
        self.assertEqual(self.store.get_accepted_plan(goal_id).revision, 1)
        provider.assert_exhausted()

    def test_persistent_watchdog_survives_runtime_restart(self):
        (self.workspace / "public.txt").write_text("safe", encoding="utf-8")
        read_turn = {"tool_calls": [{"name": "read_file", "args": {"path": "public.txt"}}]}
        runtime, provider = self.runtime([inspect_call(), plan_call(), plan_pass(), read_turn, read_turn])
        plan = runtime.start_goal("Build persistent behavior")
        goal_id = runtime.active_goal().id
        runtime.approve_plan(plan.revision)
        runtime.run_slice(steps=2)
        self.assertEqual(
            len([item for item in self.store.list_actions(goal_id) if item["tool_name"] == "read_file"]),
            2,
        )
        provider.assert_exhausted()

        self.store.close()
        self.store = StateStore(self.workspace)
        second_provider = ScriptedProvider([read_turn])
        second_runtime = AgentRuntime(
            second_provider,
            self.store,
            self.workspace,
            config=self.config,
            sleeper=lambda _seconds: None,
            approval=lambda _name, _args, _risk: True,
        )
        second_runtime.run_slice(steps=1)
        self.assertEqual(
            len([item for item in self.store.list_actions(goal_id) if item["tool_name"] == "read_file"]),
            2,
        )
        second_provider.assert_exhausted()

    def test_post_semantic_planner_can_inspect_and_read_workspace_alias_is_normalized(self):
        request = "Create artifact.txt and verify it"
        semantic = {
            **plan_call()["tool_calls"][0]["args"]["semantic_goal"],
            "original_request": request,
            "interpreted_outcome": "Create and verify artifact.txt.",
            "repository_evidence_refs": ["inspection:I001"],
        }
        staged_plan = plan_call()
        staged_args = staged_plan["tool_calls"][0]["args"]
        staged_args.pop("semantic_goal", None)
        staged_args["semantic_fingerprint"] = "0" * 64
        staged_args["applicability_evidence"][0]["source"] = "inspection:I001"
        provider = ScriptedProvider(
            [
                {"tool_calls": [{"name": "propose_semantic_goal", "args": semantic}]},
                {"tool_calls": [{"name": "read_workspace", "args": {}}]},
                staged_plan,
                plan_pass(),
            ],
            model="local-coder",
        )
        runtime = AgentRuntime(
            provider,
            self.store,
            self.workspace,
            config=self.config,
            model_descriptor=ModelDescriptor(
                "ollama", "local-coder", ExecutionClass.LOCAL, capabilities=("tools",)
            ),
        )

        plan = runtime.start_goal(request)

        self.assertIsNotNone(plan)
        # Local Ollama tool schemas are transported as a streamable, bounded
        # JSON action contract instead of an atomic native-tool request.
        self.assertEqual(provider.calls[1].tools, [])
        self.assertIn("propose_plan", provider.calls[1].system)
        self.assertIn("list_files", provider.calls[1].system)
        self.assertTrue(
            any(
                event.event_type == "tool_contract.alias_normalized"
                for event in self.store.list_recent_events(plan.goal_id, limit=100)
            )
        )
        provider.assert_exhausted()

    def test_captured_local_plan_transport_mismatches_are_mechanically_bound(self):
        request = (
            "Create a complete Three.js 3D calculator in index.html, verify every "
            "button and the 3D interaction, and run deterministic checks."
        )
        anchors = [
            {
                "id": f"R{index:03d}",
                "kind": "interaction" if index in {3, 4} else "testing",
                "verbatim_span": label,
                "interpreted_requirement": label,
                "observable_implications": [label],
            }
            for index, label in enumerate(
                (
                    "index.html",
                    "Three.js 3D calculator",
                    "verify every button",
                    "3D interaction",
                    "deterministic checks",
                ),
                1,
            )
        ]
        semantic = {
            "original_request": request,
            "interpreted_outcome": request,
            "requested_effects": [
                "read_workspace",
                "mutate_workspace",
                "execute_code",
            ],
            "required_outcomes": ["A complete verified calculator."],
            "constraints": ["The final application entry point is index.html."],
            "exclusions": [],
            "acceptance_criteria": ["Every accepted requirement is verified."],
            "requirement_anchors": anchors,
            "unresolved_decisions": [],
            "repository_evidence_refs": ["inspection:I001"],
            "status": "interpreted",
        }
        proposal = {
            "semantic_fingerprint": "0" * 64,
            "summary": "Build and verify the complete Three.js calculator.",
            "applicability_evidence": [{
                "fact": "The local preflight inspected the workspace root.",
                # This is the exact source label present in the captured run.
                "source": "harness_preflight",
                "supports_tasks": ["1"],
            }],
            "execution_strategy": "Implement the page, preview it, and verify every interaction.",
            "expected_changes": [{
                "path": "index.html",
                "intent": "Create the calculator requested by the user.",
                # The captured model used this provenance enum even though the
                # exact path is verbatim in the request.
                "basis": "model_selected_new_layout",
                "evidence_refs": ["inspection:I001"],
                "supports_tasks": ["1"],
            }],
            "tasks": [{
                "id": "1",
                "title": "Implement and verify the calculator",
                "description": "Create index.html with Three.js and exercise every button and 3D interaction.",
                "requirement_refs": ["R001", "R002", "R003", "R004"],
                "acceptance_criteria": ["The complete calculator works in the browser preview."],
                # R005 was present only here in the real model response.
                "verification": "Run deterministic checks for arithmetic and interaction coverage (R005).",
                "depends_on": [],
                "risk": "high",
            }],
        }
        provider = ScriptedProvider(
            [
                {"tool_calls": [{"name": "propose_semantic_goal", "args": semantic}]},
                {"tool_calls": [{"name": "propose_plan", "args": proposal}]},
                plan_pass(),
            ],
            model="local-coder",
        )
        runtime = AgentRuntime(
            provider,
            self.store,
            self.workspace,
            config=self.config,
            model_descriptor=ModelDescriptor(
                "ollama", "local-coder", ExecutionClass.LOCAL, capabilities=("tools",)
            ),
        )

        plan = runtime.start_goal(request)

        self.assertIsNotNone(plan)
        self.assertEqual(runtime.active_goal().status, GoalStatus.AWAITING_PLAN_APPROVAL)
        self.assertIn("R005", plan.tasks[0].metadata["requirement_refs"])
        self.assertEqual(plan.applicability_evidence[0]["source"], "inspection:I001")
        self.assertEqual(plan.expected_changes[0]["basis"], "explicit_user_requirement")
        self.assertEqual(plan.expected_changes[0]["evidence_refs"], ["user:request"])
        self.assertFalse(
            any(
                event.event_type == "planning.validation_rejected"
                for event in self.store.list_recent_events(plan.goal_id, limit=100)
            )
        )
        provider.assert_exhausted()

    def test_semantic_assumption_is_saved_as_default_not_false_user_blocker(self):
        goal = self.store.create_goal(
            "Create a complete calculator with every standard button",
        )
        semantic = {
            "original_request": goal.objective,
            "interpreted_outcome": "Create and verify the calculator.",
            "requested_effects": ["mutate_workspace", "execute_code"],
            "required_outcomes": ["The calculator works."],
            "constraints": [],
            "exclusions": [],
            "acceptance_criteria": ["Every calculator button works."],
            "requirement_anchors": [],
            "unresolved_decisions": [
                "The mathematical scope was not enumerated; assuming basic arithmetic "
                "buttons based on standard practice."
            ],
            "repository_evidence_refs": [],
            "status": "interpreted",
        }

        validated = AgentRuntime._validate_semantic_stage(
            goal,
            semantic,
            successful_inspection_ids=frozenset({"I001"}),
        )

        self.assertEqual(validated.unresolved_decisions, ())
        self.assertTrue(
            any(item.startswith("Planner default:") for item in validated.constraints)
        )
        self.assertIn("inspection:I001", validated.repository_evidence_refs)
        self.assertIn("read_workspace", [item.value for item in validated.requested_effects])

    def test_structured_planner_progress_resets_consecutive_empty_turns(self):
        request = "Create artifact.txt and verify it"
        semantic = {
            **plan_call()["tool_calls"][0]["args"]["semantic_goal"],
            "original_request": request,
            "interpreted_outcome": "Create and verify artifact.txt.",
            "repository_evidence_refs": ["inspection:I001"],
        }
        staged_plan = plan_call()
        staged_args = staged_plan["tool_calls"][0]["args"]
        staged_args.pop("semantic_goal", None)
        staged_args["semantic_fingerprint"] = "0" * 64
        staged_args["applicability_evidence"][0]["source"] = "inspection:I001"
        provider = ScriptedProvider(
            [
                "No action yet.",
                "Still reasoning.",
                {"tool_calls": [{"name": "propose_semantic_goal", "args": semantic}]},
                "Preparing the structured plan.",
                staged_plan,
                plan_pass(),
            ],
            model="local-coder",
        )
        runtime = AgentRuntime(
            provider,
            self.store,
            self.workspace,
            config=replace(self.config, planning_steps=6, no_action_limit=3),
            model_descriptor=ModelDescriptor(
                "ollama", "local-coder", ExecutionClass.LOCAL, capabilities=("tools",)
            ),
        )

        plan = runtime.start_goal(request)

        self.assertIsNotNone(plan)
        recoveries = [
            event
            for event in self.store.list_recent_events(plan.goal_id, limit=200)
            if event.event_type == "planning.no_progress_recovery"
        ]
        self.assertEqual(recoveries, [])
        provider.assert_exhausted()

    def test_local_planner_no_progress_projects_accepted_semantic_into_reviewed_plan(self):
        request = "Create and verify index.html"
        semantic = {
            **plan_call()["tool_calls"][0]["args"]["semantic_goal"],
            "original_request": request,
            "interpreted_outcome": "Create and verify index.html.",
            "repository_evidence_refs": ["inspection:I001"],
            "acceptance_criteria": [
                "index.html exists with the requested implementation.",
                "Fresh deterministic verification passes.",
            ],
        }
        provider = ScriptedProvider(
            [
                {"tool_calls": [{"name": "propose_semantic_goal", "args": semantic}]},
                "No plan transport yet.",
                "Still no structured action.",
                "No action after the fresh packet.",
                "Still unable to emit the plan schema.",
                plan_pass(),
            ],
            model="local-coder",
        )
        runtime = AgentRuntime(
            provider,
            self.store,
            self.workspace,
            config=replace(self.config, planning_steps=6, no_action_limit=2),
            model_descriptor=ModelDescriptor(
                "ollama", "local-coder", ExecutionClass.LOCAL, capabilities=("tools",)
            ),
        )

        plan = runtime.start_goal(request)

        self.assertIsNotNone(plan)
        self.assertEqual(runtime.active_goal().status, GoalStatus.AWAITING_PLAN_APPROVAL)
        self.assertEqual(len(plan.tasks), 1)
        self.assertEqual(plan.expected_changes[0]["path"], "index.html")
        self.assertEqual(plan.expected_changes[0]["basis"], "explicit_user_requirement")
        self.assertTrue(
            any(
                event.event_type == "planning.deterministic_plan_projected"
                for event in self.store.list_recent_events(plan.goal_id, limit=200)
            )
        )
        provider.assert_exhausted()

    def test_planner_contract_failure_is_typed_instead_of_becoming_empty_turn(self):
        disallowed = {
            "tool_calls": [{"name": "delete_file", "args": {"path": "artifact.txt"}}]
        }
        provider = ScriptedProvider([disallowed, disallowed, disallowed], model="local-coder")
        runtime = AgentRuntime(
            provider,
            self.store,
            self.workspace,
            config=self.config,
            model_descriptor=ModelDescriptor(
                "ollama", "local-coder", ExecutionClass.LOCAL, capabilities=("tools",)
            ),
        )
        schemas = [
            schema
            for schema in runtime._planner_tools()
            if schema["function"]["name"] == "propose_plan"
        ]

        turn = runtime._call_provider(
            [], schemas, "planner", actor="planner", step=1
        )

        error = turn.native["tool_contract_error"]
        self.assertEqual(error["received"], ["delete_file"])
        self.assertEqual(error["allowed"], ["propose_plan"])
        self.assertEqual(error["physical_attempt"], 3)
        self.assertTrue(error["logical_request_id"].startswith("planner-1-"))
        self.assertIsNone(turn.text)
        self.assertEqual(turn.tool_calls, [])
        provider.assert_exhausted()

    def test_empty_response_after_rejected_tool_remains_a_typed_contract_failure(self):
        disallowed = {
            "tool_calls": [{"name": "propose_semantic_goal", "args": {}}]
        }
        provider = ScriptedProvider(
            [disallowed, {"text": ""}, {"text": ""}],
            model="local-coder",
        )
        runtime = AgentRuntime(
            provider,
            self.store,
            self.workspace,
            config=self.config,
            model_descriptor=ModelDescriptor(
                "ollama", "local-coder", ExecutionClass.LOCAL, capabilities=("tools",)
            ),
        )
        schemas = [
            schema
            for schema in runtime._planner_tools()
            if schema["function"]["name"] == "propose_plan"
        ]

        turn = runtime._call_provider([], schemas, "planner", actor="planner", step=3)

        error = turn.native["tool_contract_error"]
        self.assertEqual(error["received"], ["<no tool call after contract correction>"])
        self.assertEqual(error["attempts"], 3)
        self.assertEqual(error["physical_attempt"], 3)
        self.assertEqual(turn.tool_calls, [])
        self.assertIsNone(turn.text)
        provider.assert_exhausted()

    def test_planner_rotation_injects_full_authoritative_stage_checkpoint(self):
        request = "Create artifact.txt and verify it"
        semantic = {
            **plan_call()["tool_calls"][0]["args"]["semantic_goal"],
            "original_request": request,
            "interpreted_outcome": "Create and verify artifact.txt.",
            "repository_evidence_refs": ["inspection:I001"],
        }
        staged_plan = plan_call()
        staged_args = staged_plan["tool_calls"][0]["args"]
        staged_args.pop("semantic_goal", None)
        staged_args["semantic_fingerprint"] = "0" * 64
        staged_args["applicability_evidence"][0]["source"] = "inspection:I001"

        def assert_rotated_checkpoint(provider_request):
            rendered = json.dumps(provider_request.conversation, ensure_ascii=False)
            self.assertIn("PROVIDER_CONTEXT_CHECKPOINT", rendered)
            self.assertIn("planning_semantic_goal", rendered)
            self.assertIn("planning_semantic_fingerprint", rendered)
            self.assertIn("inspection:I001", rendered)
            self.assertIn("plan_generation", rendered)
            return staged_plan

        provider = ScriptedProvider(
            [
                {"tool_calls": [{"name": "propose_semantic_goal", "args": semantic}]},
                "I am still thinking about the plan. " + ("x" * 8_000),
                assert_rotated_checkpoint,
                plan_pass(),
            ],
            model="local-coder",
        )
        runtime = AgentRuntime(
            provider,
            self.store,
            self.workspace,
            config=replace(self.config, conversation_chars=4_000, no_action_limit=3),
            model_descriptor=ModelDescriptor(
                "ollama", "local-coder", ExecutionClass.LOCAL, capabilities=("tools",)
            ),
        )

        plan = runtime.start_goal(request)

        self.assertIsNotNone(plan)
        # Planner rotation must not make an unmonitored second provider call
        # before the visible/watchdog-protected planner request.
        self.assertEqual(provider.summary_calls, [])
        provider.assert_exhausted()

    def test_resume_reuses_persisted_semantic_and_inspection_checkpoint(self):
        request = "Create artifact.txt and verify it"
        semantic = {
            **plan_call()["tool_calls"][0]["args"]["semantic_goal"],
            "original_request": request,
            "interpreted_outcome": "Create and verify artifact.txt.",
            "repository_evidence_refs": ["inspection:I001"],
        }
        first_provider = ScriptedProvider(
            [
                {"tool_calls": [{"name": "propose_semantic_goal", "args": semantic}]},
                "No structured plan yet.",
            ],
            model="local-coder",
        )
        descriptor = ModelDescriptor(
            "ollama", "local-coder", ExecutionClass.LOCAL, capabilities=("tools",)
        )
        runtime = AgentRuntime(
            first_provider,
            self.store,
            self.workspace,
            config=replace(self.config, planning_steps=2, no_action_limit=2),
            model_descriptor=descriptor,
        )
        self.assertIsNone(runtime.start_goal(request))
        goal_id = runtime.active_goal().id
        inspection_count = len(
            [
                event
                for event in self.store.list_recent_events(goal_id, limit=100)
                if event.event_type == "planning.inspection_recorded"
            ]
        )
        runtime.close()

        staged_plan = plan_call()
        staged_args = staged_plan["tool_calls"][0]["args"]
        staged_args.pop("semantic_goal", None)
        staged_args["semantic_fingerprint"] = "0" * 64
        staged_args["applicability_evidence"][0]["source"] = "inspection:I001"
        second_provider = ScriptedProvider([staged_plan, plan_pass()], model="local-coder")
        resumed = AgentRuntime(
            second_provider,
            self.store,
            self.workspace,
            config=self.config,
            model_descriptor=descriptor,
        )

        resumed.resume()

        plan = resumed.latest_plan()
        self.assertIsNotNone(plan)
        self.assertEqual(resumed.active_goal().status, GoalStatus.AWAITING_PLAN_APPROVAL)
        self.assertEqual(
            len(
                [
                    event
                    for event in self.store.list_recent_events(goal_id, limit=200)
                    if event.event_type == "planning.inspection_recorded"
                ]
            ),
            inspection_count,
        )
        second_provider.assert_exhausted()
        resumed.close()

    def test_continuous_controller_retries_one_saved_planning_no_progress_boundary(self):
        provider = ScriptedProvider(
            [
                "No action yet.",
                "Still no action.",
                inspect_call(),
                plan_call(),
                plan_pass(),
            ]
        )
        runtime = AgentRuntime(
            provider,
            self.store,
            self.workspace,
            config=replace(self.config, planning_steps=2),
            sleeper=lambda _seconds: None,
        )
        self.assertIsNone(runtime.start_goal("Build persistent behavior"))
        self.assertTrue(runtime.active_goal().metadata["auto_retryable"])

        result = runtime.continue_until_boundary()

        self.assertIsNotNone(runtime.latest_plan())
        self.assertEqual(runtime.active_goal().status, GoalStatus.AWAITING_PLAN_APPROVAL)
        self.assertEqual(result.status, GoalStatus.AWAITING_PLAN_APPROVAL.value)
        self.assertEqual(result.phase, "awaiting_approval")
        provider.assert_exhausted()


class DelegationAndReviewTests(RuntimeTestCase):
    def test_dynamic_role_worker_has_isolated_context_and_structured_result(self):
        role = (
            "Crash-consistency investigator. Inspect transaction boundaries and report only "
            "restart evidence for this storage task."
        )
        delegate = {
            "id": "delegate",
            "name": "delegate_task",
            "args": {
                "task_id": "T001",
                "role": role,
                "task": "Inspect whether state writes survive interruption.",
                "success_criteria": ["Report a concrete restart finding."],
                "context": "Focus on the state store only.",
                "allowed_tools": ["read_file", "grep"],
            },
        }
        worker_return = {
            "tool_calls": [
                {
                    "id": "return",
                    "name": "return_work",
                    "args": {
                        "outcome": "success",
                        "summary": "Transactions are atomic and recovery is journaled.",
                        "evidence": ["Recovery test covers an interrupted in-flight action."],
                        "changed_paths": [],
                        "remaining_risks": [],
                        "proposed_subtasks": [],
                    },
                }
            ]
        }
        runtime, provider = self.runtime(
            [
                inspect_call(),
                plan_call(),
                plan_pass(),
                {"tool_calls": [task_update("in_progress", note="starting"), delegate]},
                worker_return,
                {"tool_calls": [
                    {"id": "verify-worker", "name": "list_files", "args": {"path": "."}},
                    task_update("done", ["worker evidence reviewed"], "verified"), finish_call()
                ]},
                review_pass(),
            ]
        )
        plan = runtime.start_goal("Build persistent behavior")
        goal_id = runtime.active_goal().id
        runtime.approve_plan(plan.revision)
        result = runtime.run_slice()
        self.assertTrue(result.completed)
        delegation = self.store.list_delegations(goal_id)[0]
        self.assertEqual(delegation.role.mission, role)
        self.assertEqual(delegation.status, DelegationStatus.COMPLETED)
        # The worker call is fresh: it sees one WORKER_BRIEF user message, not the
        # coordinator's full conversation.
        worker_request = next(call for call in provider.calls if "focused worker" in call.system)
        self.assertEqual(len(worker_request.conversation), 1)
        provider.assert_exhausted()

    def test_delegate_is_bound_to_explicit_plan_task(self):
        delegate = {
            "id": "delegate-t2",
            "name": "delegate_task",
            "args": {
                "task_id": "T002",
                "role": "Integration boundary investigator. Verify only the T002 integration contract.",
                "task": "Inspect the T002 integration boundary.",
                "success_criteria": ["Return direct T002 integration evidence."],
                "context": "T001 is already complete.",
                "allowed_tools": ["read_file"],
            },
        }
        worker_return = {
            "tool_calls": [
                {
                    "name": "return_work",
                    "args": {
                        "outcome": "success",
                        "summary": "T002 integration inspected.",
                        "evidence": ["T002 boundary matches the accepted contract."],
                        "changed_paths": [],
                        "remaining_risks": [],
                        "proposed_subtasks": [],
                    },
                }
            ]
        }
        runtime, provider = self.runtime(
            [
                inspect_call(),
                dependency_plan_call(),
                plan_pass(),
                {"tool_calls": [
                    {**task_update("in_progress", note="integrating"), "args": {**task_update("in_progress", note="integrating")["args"], "task_id": "T002"}},
                    delegate,
                ]},
                worker_return,
            ]
        )
        plan = runtime.start_goal("Build dependency-aware behavior")
        goal_id = runtime.active_goal().id
        runtime.approve_plan(plan.revision)
        runtime.update_task_from_user("T001", "done", "T001 unit test passed")
        runtime.run_slice(steps=1)
        delegation = self.store.list_delegations(goal_id)[0]
        self.assertEqual(delegation.task_id, "T002")
        t2_evidence = self.store.list_evidence(goal_id, task_id="T002")
        self.assertTrue(any(item.kind == "delegation" for item in t2_evidence))
        provider.assert_exhausted()

    def test_final_reviewer_cannot_execute_shell_even_with_user_approval(self):
        shell_attempt = {
            "tool_calls": [
                {
                    "id": "bad-review-shell",
                    "name": "run_bash",
                    "args": {"command": "echo mutated > reviewer.txt"},
                }
            ]
        }
        runtime, provider = self.runtime(
            [inspect_call(), plan_call(), plan_pass(), {"tool_calls": [finish_call()]}, shell_attempt, review_pass()]
        )
        plan = runtime.start_goal("Build persistent behavior")
        runtime.approve_plan(plan.revision)
        runtime.update_task_from_user("T001", "done", "focused tests passed")
        result = runtime.run_slice(steps=1)
        self.assertTrue(result.completed)
        self.assertFalse((self.workspace / "reviewer.txt").exists())
        reviewer_requests = [call for call in provider.calls if "independent final reviewer" in call.system]
        self.assertTrue(reviewer_requests)
        self.assertTrue(
            all(
                schema["function"]["name"] != "run_bash"
                for request in reviewer_requests
                for schema in request.tools
            )
        )
        provider.assert_exhausted()

    def test_request_user_prevents_later_finish_call_in_same_turn(self):
        pause_then_finish = {
            "tool_calls": [
                {
                    "id": "need-user",
                    "name": "request_user",
                    "args": {"question": "Which API contract is authoritative?", "reason": "Both local contracts conflict."},
                },
                finish_call(),
            ]
        }
        runtime, provider = self.runtime([inspect_call(), plan_call(), plan_pass(), pause_then_finish])
        plan = runtime.start_goal("Build persistent behavior")
        goal_id = runtime.active_goal().id
        runtime.approve_plan(plan.revision)
        runtime.update_task_from_user("T001", "done", "focused tests passed")
        result = runtime.run_slice(steps=1)
        self.assertFalse(result.completed)
        self.assertEqual(self.store.get_goal(goal_id).status, GoalStatus.PAUSED)
        self.assertFalse(any(item.kind == "final_review" for item in self.store.list_evidence(goal_id)))
        provider.assert_exhausted()

    def test_invalid_agent_plan_change_is_a_tool_error_not_a_runtime_crash(self):
        invalid_change = {
            "tool_calls": [
                {
                    "name": "propose_plan_change",
                    "args": {
                        "reason": "A new task was discovered.",
                        "tasks": [
                            {
                                "id": "T099",
                                "title": "Broken dependency",
                                "description": "This proposal references an unknown prerequisite.",
                                "acceptance_criteria": ["The proposal is valid."],
                                "verification": ["Validate the DAG."],
                                "depends_on": ["MISSING"],
                                "risk": "medium",
                            }
                        ],
                    },
                }
            ]
        }
        runtime, provider = self.runtime([inspect_call(), plan_call(), plan_pass(), invalid_change])
        plan = runtime.start_goal("Build persistent behavior")
        runtime.approve_plan(plan.revision)
        result = runtime.run_slice(steps=1)
        self.assertEqual(result.status, GoalStatus.RUNNING.value)
        self.assertEqual(runtime.latest_plan().revision, 1)
        provider.assert_exhausted()

    def test_coordinator_reopening_prerequisite_invalidates_completed_dependants(self):
        reopen = {
            "tool_calls": [
                {
                    "name": "update_task",
                    "args": {
                        "task_id": "T001",
                        "status": "pending",
                        "note": "A new edge case invalidated the earlier result.",
                        "evidence": [],
                    },
                }
            ]
        }
        runtime, provider = self.runtime([inspect_call(), dependency_plan_call(), plan_pass(), reopen])
        plan = runtime.start_goal("Build dependency-aware behavior")
        runtime.approve_plan(plan.revision)
        runtime.update_task_from_user("T001", "done", "unit proof")
        runtime.update_task_from_user("T002", "done", "integration proof")
        runtime.run_slice(steps=1)
        statuses = {task.id: task.status for task in runtime.latest_plan().tasks}
        self.assertEqual(statuses, {"T001": TaskStatus.PENDING, "T002": TaskStatus.PENDING})
        provider.assert_exhausted()


class RecoveryRuntimeTests(RuntimeTestCase):
    def test_execution_depth_can_increase_before_plan_approval(self):
        goal = self.store.create_goal("Saved planning objective")
        self.store.transition_goal(goal.id, GoalStatus.DISCOVERING)
        self.store.transition_goal(goal.id, GoalStatus.AWAITING_PLAN_APPROVAL)
        runtime = AgentRuntime(
            ScriptedProvider([]), self.store, self.workspace, config=self.config
        )
        with mock.patch.object(runtime, "generate_plan", return_value="revision") as generate:
            self.assertEqual(runtime.prepare_ultra_from_existing_goal(), "revision")

        current = self.store.get_latest_goal()
        self.assertEqual(current.id, goal.id)
        self.assertEqual(current.metadata["execution_strategy"], "recursive")
        self.assertFalse(current.metadata["strategy_locked"])
        self.assertIsNone(runtime.ultra_session)
        generate.assert_called_once()

    def test_failed_ultra_foundation_retry_reuses_saved_goal(self):
        goal = self.store.create_goal("Saved canonical objective")
        self.store.transition_goal(goal.id, GoalStatus.DISCOVERING)
        provider = ScriptedProvider([inspect_call(), plan_call(), plan_pass()])
        runtime = AgentRuntime(
            provider, self.store, self.workspace, config=self.config
        )

        result = runtime.retry_ultra_foundation()

        self.assertEqual(result.status, PlanStatus.PENDING_APPROVAL)
        self.assertEqual(
            self.store.get_goal(goal.id).status,
            GoalStatus.AWAITING_PLAN_APPROVAL,
        )
        self.assertEqual(self.store.get_latest_goal().id, goal.id)
        self.assertIsNone(runtime.ultra_session)
        provider.assert_exhausted()

    def test_interrupted_planning_resumes_planner_not_invalid_running_state(self):
        goal = self.store.create_goal("Resume an interrupted plan")
        self.store.transition_goal(goal.id, GoalStatus.DISCOVERING)
        provider = ScriptedProvider([inspect_call(), plan_call(), plan_pass()])
        runtime = AgentRuntime(provider, self.store, self.workspace, config=self.config, sleeper=lambda _s: None)
        self.assertEqual(runtime.active_goal().status, GoalStatus.PAUSED)
        self.assertEqual(runtime.active_goal().metadata["resume_status"], GoalStatus.DISCOVERING.value)
        runtime.resume()
        self.assertEqual(runtime.active_goal().status, GoalStatus.AWAITING_PLAN_APPROVAL)
        provider.assert_exhausted()

    def test_interrupted_review_resumes_as_running_for_a_fresh_finish_request(self):
        goal = self.store.create_goal("Resume an interrupted review")
        self.store.transition_goal(goal.id, GoalStatus.AWAITING_PLAN_APPROVAL)
        plan = self.store.create_plan(goal.id, "review", [
            {
                "id": "T001", "title": "Verify", "description": "Verify work",
                "acceptance_criteria": ["Work is proven"], "verification": ["Run tests"],
                "depends_on": [], "risk": "medium",
            }
        ], **stored_plan_basis("T001"))
        self.store.approve_plan(goal.id, plan.revision)
        self.store.transition_goal(goal.id, GoalStatus.VERIFYING)
        self.store.transition_goal(goal.id, GoalStatus.REVIEWING)
        runtime = AgentRuntime(ScriptedProvider([]), self.store, self.workspace, config=self.config)
        self.assertEqual(runtime.active_goal().status, GoalStatus.PAUSED)
        runtime.resume()
        self.assertEqual(runtime.active_goal().status, GoalStatus.RUNNING)

    def test_uncertain_action_can_be_reconciled_then_resumed(self):
        goal = self.store.create_goal("Recover a write")
        self.store.transition_goal(goal.id, GoalStatus.AWAITING_PLAN_APPROVAL)
        plan = self.store.create_plan(goal.id, "recover", [
            {
                "id": "T001", "title": "Write safely", "description": "Write safely",
                "acceptance_criteria": ["State is known"], "verification": ["Inspect file"],
                "depends_on": [], "risk": "high",
            }
        ], **stored_plan_basis("T001"))
        self.store.approve_plan(goal.id, plan.revision)
        self.store.transition_task(goal.id, 1, "T001", TaskStatus.IN_PROGRESS)
        action_id = self.store.begin_action(
            goal.id, "write_file", {"_harness_actor": "coordinator", "arguments": {"path": "x"}},
            task_id="T001", mutating=True,
        )
        runtime = AgentRuntime(ScriptedProvider([]), self.store, self.workspace, config=self.config)
        self.assertEqual(runtime.active_goal().status, GoalStatus.PAUSED)
        uncertain_set = runtime.active_goal().metadata["goal_change_sets"][-1]
        self.assertEqual(uncertain_set["integration_status"], "uncertain")
        self.assertEqual(uncertain_set["tool_action_ids"], [action_id])
        self.assertTrue(runtime.active_goal().metadata["latest_evaluation_stale"])
        runtime.add_guidance("Do not replay the interrupted write.")
        self.assertEqual(runtime.active_goal().status, GoalStatus.PAUSED)
        with self.assertRaises(RuntimeStateError):
            runtime.resume()
        runtime.resolve_action(action_id, "not-run", "Inspected workspace; x was not created.")
        self.assertEqual(self.store.list_actions(goal.id)[0]["status"], "resolved_not_run")
        self.assertEqual(self.store.list_tasks(goal.id, 1)[0].status, TaskStatus.IN_PROGRESS)
        runtime.resume()
        self.assertEqual(runtime.active_goal().status, GoalStatus.RUNNING)

    def test_uncertain_delegation_can_be_reconciled_then_resumed(self):
        goal = self.store.create_goal("Recover an interrupted delegated worker")
        self.store.transition_goal(goal.id, GoalStatus.AWAITING_PLAN_APPROVAL)
        plan = self.store.create_plan(goal.id, "recover worker", [
            {
                "id": "T001", "title": "Delegate safely", "description": "Delegate safely",
                "acceptance_criteria": ["Worker state is known"], "verification": ["Inspect workspace"],
                "depends_on": [], "risk": "high",
            }
        ], **stored_plan_basis("T001"))
        self.store.approve_plan(goal.id, plan.revision)
        self.store.transition_task(goal.id, 1, "T001", TaskStatus.IN_PROGRESS)
        delegation = self.store.create_delegation(
            goal_id=goal.id,
            task_id="T001",
            plan_revision=plan.revision,
            brief="Inspect and implement the task-specific change.",
        )
        self.store.transition_delegation(delegation.id, DelegationStatus.IN_PROGRESS)

        runtime = AgentRuntime(ScriptedProvider([]), self.store, self.workspace, config=self.config)

        self.assertEqual(runtime.active_goal().status, GoalStatus.PAUSED)
        self.assertEqual(
            self.store.list_delegations(goal.id)[0].status,
            DelegationStatus.UNCERTAIN,
        )
        resolved = runtime.resolve_action(
            delegation.id,
            "not-run",
            "Inspected the workspace; the worker made no durable change.",
        )
        self.assertEqual(resolved.status, DelegationStatus.FAILED)
        self.assertEqual(
            self.store.list_tasks(goal.id, plan.revision)[0].status,
            TaskStatus.IN_PROGRESS,
        )
        runtime.resume()
        self.assertEqual(runtime.active_goal().status, GoalStatus.RUNNING)

    def test_failed_review_creates_and_harness_approves_in_scope_repair_revision(self):
        failed_review = {
            "tool_calls": [
                {
                    "id": "review-fail",
                    "name": "submit_review",
                    "args": {
                        "verdict": "fail",
                        "summary": "Restart edge case is not proven.",
                        "issues": [
                            {
                                "severity": "high",
                                "title": "Prove restart recovery",
                                "details": "The crash window lacks direct verification.",
                                "acceptance_criteria": ["An interrupted write is restored or marked uncertain without replay."],
                            }
                        ],
                        "checked_task_ids": ["T001"],
                    },
                }
            ]
        }
        runtime, provider = self.runtime(
            [
                inspect_call(),
                plan_call(),
                plan_pass(),
                {"tool_calls": [task_update("in_progress", note="starting")]},
                {"tool_calls": [
                    {"id": "verify-before-review", "name": "list_files", "args": {"path": "."}},
                    task_update("done", ["initial test"], "tested"), finish_call()
                ]},
                failed_review,
            ]
        )
        plan = runtime.start_goal("Build persistent behavior")
        goal_id = runtime.active_goal().id
        runtime.approve_plan(plan.revision)
        result = runtime.run_slice()
        self.assertFalse(result.completed)
        goal = self.store.get_goal(goal_id)
        revised = self.store.get_latest_plan(goal_id)
        self.assertEqual(goal.status, GoalStatus.RUNNING)
        self.assertEqual(revised.revision, 2)
        self.assertEqual(revised.status, PlanStatus.ACCEPTED)
        self.assertIn("Prove restart recovery", [task.title for task in revised.tasks])
        self.assertEqual(goal.metadata["convergence_state"], "refining")
        self.assertEqual(goal.metadata["refinement_actions"][-1]["source"], "independent-reviewer")
        provider.assert_exhausted()
