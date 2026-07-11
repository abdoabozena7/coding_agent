from __future__ import annotations

import tempfile
import unittest
import io
from dataclasses import replace
from pathlib import Path

from agent.config import RuntimeConfig
from agent.cli import _run_auto
from agent.models import DelegationStatus, GoalStatus, PlanStatus, TaskStatus
from agent.runtime import AgentRuntime, RuntimeStateError
from agent.store import StateStore
from agent.testing import ScriptedProvider
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

    def test_prose_done_never_completes_persistent_goal(self):
        runtime, provider = self.runtime([inspect_call(), plan_call(), plan_pass(), "Everything is done.", "Done for sure."])
        plan = runtime.start_goal("Build persistent behavior")
        runtime.approve_plan(plan.revision)
        result = runtime.run_slice(steps=3)
        self.assertFalse(result.completed)
        self.assertEqual(runtime.active_goal().status, GoalStatus.RUNNING)
        self.assertEqual(self.store.list_tasks(runtime.active_goal().id, 1)[0].status, TaskStatus.PENDING)
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

    def test_provider_failure_schedules_unbounded_goal_retry_then_recovers(self):
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
        self.assertIn("unbounded retry loop", output.getvalue())
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
                {"tool_calls": [task_update("done", ["worker evidence reviewed"], "verified"), finish_call()]},
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

    def test_failed_review_creates_repair_revision_and_requires_approval(self):
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
                {"tool_calls": [task_update("done", ["initial test"], "tested"), finish_call()]},
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
        self.assertEqual(goal.status, GoalStatus.AWAITING_PLAN_APPROVAL)
        self.assertEqual(revised.revision, 2)
        self.assertEqual(revised.status, PlanStatus.PENDING_APPROVAL)
        self.assertIn("Prove restart recovery", [task.title for task in revised.tasks])
        provider.assert_exhausted()
