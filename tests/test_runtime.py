from __future__ import annotations

import tempfile
import unittest
import io
import hashlib
import subprocess
from dataclasses import replace
from pathlib import Path
from unittest import mock

from agent.config import RuntimeConfig, update_runtime_config
from agent.cli import _run_auto
from agent.hardware import HardwareProbeResult, probe_local_gpu
from agent.model_catalog import ExecutionClass, ModelDescriptor
from agent.models import DelegationStatus, GoalStatus, PlanStatus, TaskStatus
from agent.runtime import AgentRuntime, RuntimeStateError
from agent.sandbox import DockerSandbox, PermissionAdapter
from agent.store import StateStore
from agent.testing import ScriptedProvider
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
                            "path": "agent/",
                            "intent": "Implement the durable behavior and its verification support.",
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
    return {
        "tool_calls": [
            {
                "id": "plan-deps",
                "name": "propose_plan",
                "args": {
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
                            "path": "agent/ and tests/",
                            "intent": "Implement and integrate the durable behavior.",
                            "supports_tasks": ["T001", "T002"],
                        }
                    ],
                    "tasks": [first, second],
                },
            },
        ]
    }


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


class PlanningAndCompletionTests(RuntimeTestCase):
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

        self.assertFalse(final.completed)
        goal = runtime.active_goal()
        self.assertEqual(goal.metadata["run_id"], run_id)
        self.assertEqual(goal.metadata["convergence_state"], "user_review_required")
        self.assertTrue(goal.metadata["failed_hypotheses"])
        self.assertIn("error_context_slice", goal.metadata)
        self.assertIn("query", goal.metadata["error_context_slice"])
        self.assertGreaterEqual(goal.metadata["error_context_slice"]["size_chars"], 0)
        self.assertGreaterEqual(len(goal.metadata["goal_change_sets"]), 3)
        self.assertEqual(goal.metadata["latest_evaluation"]["mutation_sequence"], goal.metadata["mutation_sequence"])
        self.assertEqual(goal.metadata["latest_evaluation"]["artifact_hashes"]["index.html"], hashlib.sha256(final_html.encode()).hexdigest())
        event_types = {event.event_type for event in self.store.list_recent_events(goal_id, limit=300)}
        self.assertTrue({"refinement_cycle.started", "error_signature.created", "quality_evaluation.invalidated", "quality_convergence.decided"} <= event_types)
        provider.assert_exhausted()

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
            chat_write, "Candidate created.", inspect_call(), plan_call(), plan_pass()
        ])

        chat_result = runtime.chat("Create a polished candidate")
        session = self.store.get_workflow_session(runtime.session_id)
        run_id = session["state"]["run_id"]
        self.assertIn("BELOW_TARGET", chat_result.message)

        plan = runtime.start_goal("yes")

        goal = runtime.active_goal()
        self.assertIsNotNone(plan)
        self.assertEqual(goal.objective, "Create a polished candidate")
        self.assertEqual(goal.metadata["run_id"], run_id)
        self.assertTrue(goal.metadata["continued_from_chat"])
        self.assertIn("candidate.txt", goal.metadata["goal_contract"]["artifact_expectations"])

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

    def test_visual_goal_without_vision_evaluator_stops_at_user_review_required(self):
        html_plan = plan_call()
        html_plan["tool_calls"][0]["args"]["expected_changes"][0]["path"] = "index.html"
        runtime, provider = self.runtime([
            inspect_call(), html_plan, plan_pass(), {"tool_calls": [finish_call()]}, review_pass()
        ])
        plan = runtime.start_goal("Create a polished visual page")
        runtime.approve_plan(plan.revision)
        runtime.update_task_from_user("T001", "done", "Structural and runtime checks passed")

        result = runtime.run_slice(steps=1)

        self.assertFalse(result.completed)
        goal = runtime.active_goal()
        self.assertEqual(goal.status, GoalStatus.PAUSED)
        self.assertEqual(goal.metadata["convergence_state"], "user_review_required")
        self.assertIn("Review the latest visual artifact", goal.metadata["waiting_question"])
        self.assertTrue(any(score["confidence"] == "low" for score in goal.metadata["latest_evaluation"]["scores"]))

        runtime.add_guidance("accept")
        completed = self.store.get_goal(goal.id)
        self.assertEqual(completed.status, GoalStatus.COMPLETED)
        self.assertEqual(completed.metadata["convergence_state"], "converged")
        self.assertTrue(completed.metadata["latest_evaluation"]["user_visual_acceptance_evidence_id"])

    def test_mode_changes_preserve_one_durable_run_contract_and_quality_state(self):
        runtime, _provider = self.runtime([inspect_call(), plan_call(), plan_pass()])
        plan = runtime.start_goal("Build persistent behavior")
        runtime.approve_plan(plan.revision)
        before = runtime.active_goal()
        run_id = before.metadata["run_id"]
        fingerprint = before.metadata["goal_contract_fingerprint"]

        for mode in ("chat", "plan", "goal", "chat", "goal", "ultra"):
            runtime.transition_mode(mode)

        after = runtime.active_goal()
        session = self.store.get_workflow_session(runtime.session_id)
        self.assertEqual(after.id, before.id)
        self.assertEqual(after.metadata["run_id"], run_id)
        self.assertEqual(after.metadata["goal_contract_fingerprint"], fingerprint)
        self.assertEqual(session["state"]["run_id"], run_id)
        self.assertEqual(session["session_mode"], "ultra")
        transitions = [event for event in self.store.list_recent_events(after.id, limit=100) if event.event_type == "mode.transition"]
        # Legacy chat/plan/goal aliases all normalize to one durable Normal
        # mode, so only the final Normal -> Ultra transition is material.
        self.assertGreaterEqual(len(transitions), 1)

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
        self.assertIn("visual_quality", action["affected_dimensions"])
        self.assertTrue(any(component["path"] == "index.html" for component in action["affected_components"]))
        self.assertIn("repository_context_slice", action)
        self.assertIn("style css color", action["repository_context_slice"]["query"])
        self.assertGreaterEqual(action["repository_context_slice"]["size_chars"], 1)
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
        self.assertEqual(len([item for item in attempts if item["tool_name"] == "read_file"]), 3)
        goal = self.store.get_goal(goal_id)
        self.assertEqual(len(goal.metadata["failed_attempts"]), 3)
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

    def test_repeated_invalid_plan_shapes_checkpoint_after_four_aggregated_retries(self):
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
        self.assertEqual(checkpoints[-1].payload["format_attempts"], 4)
        self.assertIn("invalid structured plan", checkpoints[-1].payload["reason"])
        provider.assert_exhausted()

    def test_repeated_invalid_plan_evidence_uses_the_same_four_attempt_cutoff(self):
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
        self.assertEqual(checkpoints[-1].payload["format_attempts"], 4)
        self.assertIn("tool:missing-inspection", checkpoints[-1].payload["technical_detail"])
        provider.assert_exhausted()

    def test_one_turn_with_many_invalid_proposals_stops_exactly_at_four(self):
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
        self.assertEqual(checkpoints[-1].payload["format_attempts"], 4)
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
        self.assertIn("must successfully inspect the workspace", retry_context)
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

        self.assertIsNotNone(plan)
        self.assertEqual(runtime.active_goal().status, GoalStatus.AWAITING_PLAN_APPROVAL)
        self.assertEqual(plan.expected_changes[0]["path"], "counter.html")
        self.assertEqual(plan.proposed_by, "harness-weak-model-fallback")
        events = self.store.list_recent_events(runtime.active_goal().id, limit=100)
        self.assertTrue(any(event.event_type == "planning.harness_fallback" for event in events))
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
                {"tool_calls": [task_update("in_progress", note="retry recovered")]},
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
            TaskStatus.IN_PROGRESS,
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
        provider = ScriptedProvider([inspect_call(), plan_call(), plan_pass(), "done", "still done"])
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
    def test_failed_ultra_foundation_retry_reuses_saved_goal(self):
        goal = self.store.create_goal("Saved canonical objective")
        self.store.transition_goal(goal.id, GoalStatus.DISCOVERING)
        runtime = AgentRuntime(
            ScriptedProvider([]), self.store, self.workspace, config=self.config
        )
        session = mock.Mock()
        session.running = False
        session.restart_foundation.return_value = "master-plan"

        with mock.patch.object(runtime, "_make_ultra_session", return_value=session):
            result = runtime.retry_ultra_foundation()

        self.assertEqual(result, "master-plan")
        session.restart_foundation.assert_called_once_with(
            goal.id, "Saved canonical objective"
        )
        self.assertEqual(self.store.get_goal(goal.id).status, GoalStatus.DISCOVERING)
        self.assertEqual(self.store.get_latest_goal().id, goal.id)

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
