from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

from agent.chat_runtime import RouteKind
from agent.models import GoalStatus, PlanStatus
from agent.runtime import AgentRuntime
from agent.store import StateStore
from agent.testing import (
    ScriptedProvider,
    semantic_goal_intake,
    semantic_goal_intake_turn,
    semantic_turn,
)
from tests.test_runtime import inspect_call, plan_call, plan_pass, review_pass
from tests.test_store import plan_basis, task


def _execution_turn(content: str = "implemented\n") -> dict:
    return {
        "tool_calls": [
            {
                "id": "write",
                "name": "write_file",
                "args": {"path": "artifact.txt", "content": content},
            },
            {
                "id": "read",
                "name": "read_file",
                "args": {"path": "artifact.txt"},
            },
            {
                "id": "done",
                "name": "update_task",
                "args": {
                    "task_id": "T001",
                    "status": "done",
                    "note": "artifact written and read back",
                    "evidence": ["authoritative write and read-back evidence"],
                },
            },
        ]
    }


def _finish_turn() -> dict:
    return {
        "tool_calls": [
            {
                "id": "finish",
                "name": "finish_goal",
                "args": {
                    "summary": "Implementation and focused verification complete.",
                    "evidence": ["artifact.txt was written and read back"],
                },
            }
        ]
    }


def test_hybrid_router_keeps_explanation_in_chat_and_promotes_work_to_recursive_goal() -> None:
    with tempfile.TemporaryDirectory() as directory:
        workspace = Path(directory)
        store = StateStore(workspace)
        try:
            runtime = AgentRuntime(
                ScriptedProvider(
                    [
                        semantic_turn(
                            "chat",
                            original="Why does the workflow persist state?",
                            response="The workflow persists state so interrupted work can resume.",
                        ),
                        semantic_turn("action", original="create note.txt", effects=("write",)),
                        semantic_goal_intake_turn(
                            semantic_goal_intake("create note.txt")
                        ),
                        inspect_call(),
                        plan_call(),
                        plan_pass(),
                    ]
                ),
                store,
                workspace,
                approval=lambda *_: True,
            )
            explanation, chat_result = runtime.route_input(
                "Why does the workflow persist state?"
            )
            assert explanation.kind is RouteKind.CHAT
            assert chat_result.status == "chat"
            assert runtime.active_goal() is None

            action, action_result = runtime.route_input("create note.txt")
            assert action.kind is RouteKind.GOAL
            assert action_result.goal_id == runtime.active_goal().id
            assert runtime.active_goal().metadata["execution_strategy"] == "recursive"
            assert not (workspace / "note.txt").exists()
        finally:
            store.close()


def test_arabic_vague_input_checkpoints_consequential_intake_questions() -> None:
    with tempfile.TemporaryDirectory() as directory:
        workspace = Path(directory)
        store = StateStore(workspace)
        try:
            original = "اعمل ده"
            intake = semantic_goal_intake(original)
            intake["questions"] = [{
                "id": "outcome", "header": "Outcome",
                "question": "What outcome should be produced?",
                "reason": "The requested result is consequentially ambiguous.",
                "options": [
                    {"value": "implement", "label": "Implement", "description": "Create a working result.", "recommended": True},
                    {"value": "repair", "label": "Repair", "description": "Repair existing work.", "recommended": False},
                    {"value": "analyze", "label": "Analyze", "description": "Report without changes.", "recommended": False},
                ],
            }]
            runtime = AgentRuntime(
                ScriptedProvider([semantic_turn("goal", original=original, goal_intake=intake)]),
                store,
                workspace,
            )
            result = runtime.submit_intent(original)
            assert result.status == "awaiting_answers"
            assert result.needs_user
            assert runtime.active_goal() is None
            assert runtime.intake_questions()
        finally:
            store.close()


def test_pending_normal_approval_survives_restart_and_mutates_exactly_once() -> None:
    with tempfile.TemporaryDirectory() as directory:
        workspace = Path(directory)
        first_store = StateStore(workspace)
        first = AgentRuntime(
            ScriptedProvider([inspect_call(), plan_call(), plan_pass()]),
            first_store,
            workspace,
            approval=lambda *_: True,
        )
        plan = first.start_goal("Implement restart-safe durable behavior")
        fingerprint = plan.fingerprint
        goal_id = plan.goal_id
        assert plan.status is PlanStatus.PENDING_APPROVAL
        first.close()
        first_store.close()

        second_store = StateStore(workspace)
        try:
            second = AgentRuntime(
                ScriptedProvider(
                    [_execution_turn(), _finish_turn(), review_pass()]
                ),
                second_store,
                workspace,
                approval=lambda *_: True,
            )
            restored = second.latest_plan()
            assert restored.fingerprint == fingerprint
            second.approve_plan(restored.revision)
            result = second.continue_until_boundary()
            assert result.completed
            assert second_store.get_goal(goal_id).status is GoalStatus.COMPLETED
            writes = [
                action
                for action in second_store.list_actions(goal_id)
                if action["tool_name"] == "write_file"
            ]
            assert len(writes) == 1
            assert (workspace / "artifact.txt").read_text(
                encoding="utf-8"
            ) == "implemented\n"
            second.close()
        finally:
            second_store.close()


def test_pending_normal_approval_recovers_across_real_processes() -> None:
    with tempfile.TemporaryDirectory() as directory:
        workspace = Path(directory)
        repository = Path(__file__).resolve().parents[1]
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(repository) + os.pathsep + environment.get(
            "PYTHONPATH", ""
        )
        create = textwrap.dedent(
            """
            import json
            import sys
            from pathlib import Path
            from agent.runtime import AgentRuntime
            from agent.store import StateStore
            from agent.testing import ScriptedProvider
            from tests.test_runtime import inspect_call, plan_call, plan_pass

            workspace = Path(sys.argv[1])
            store = StateStore(workspace)
            runtime = AgentRuntime(
                ScriptedProvider([inspect_call(), plan_call(), plan_pass()]),
                store,
                workspace,
            )
            plan = runtime.start_goal("Implement restart-safe durable behavior")
            print(json.dumps({
                "goal_id": plan.goal_id,
                "revision": plan.revision,
                "fingerprint": plan.fingerprint,
                "status": plan.status.value,
            }))
            runtime.close()
            store.close()
            """
        )
        resume = textwrap.dedent(
            """
            import json
            import sys
            from pathlib import Path
            from agent.runtime import AgentRuntime
            from agent.store import StateStore
            from agent.testing import ScriptedProvider
            from tests.test_runtime import review_pass
            from tests.test_workflow_repair_v15 import _execution_turn, _finish_turn

            workspace = Path(sys.argv[1])
            store = StateStore(workspace)
            runtime = AgentRuntime(
                ScriptedProvider([_execution_turn(), _finish_turn(), review_pass()]),
                store,
                workspace,
                approval=lambda *_: True,
            )
            plan = runtime.latest_plan()
            runtime.approve_plan(plan.revision)
            result = runtime.continue_until_boundary()
            writes = [
                action for action in store.list_actions(plan.goal_id)
                if action["tool_name"] == "write_file"
            ]
            print(json.dumps({
                "fingerprint": plan.fingerprint,
                "completed": result.completed,
                "disposition": result.disposition,
                "write_count": len(writes),
                "write_statuses": [action["status"] for action in writes],
            }))
            runtime.close()
            store.close()
            """
        )
        first = subprocess.run(
            [sys.executable, "-c", create, str(workspace)],
            cwd=repository,
            env=environment,
            text=True,
            capture_output=True,
            timeout=60,
            check=True,
        )
        created = json.loads(first.stdout.strip().splitlines()[-1])
        assert created["status"] == PlanStatus.PENDING_APPROVAL.value

        second = subprocess.run(
            [sys.executable, "-c", resume, str(workspace)],
            cwd=repository,
            env=environment,
            text=True,
            capture_output=True,
            timeout=60,
            check=True,
        )
        resumed = json.loads(second.stdout.strip().splitlines()[-1])
        assert resumed["fingerprint"] == created["fingerprint"]
        assert resumed["completed"]
        assert resumed["disposition"] == "verified"
        assert resumed["write_count"] == 1
        assert resumed["write_statuses"] == ["completed"]


def test_legacy_pending_plan_is_semantically_enriched_before_approval() -> None:
    with tempfile.TemporaryDirectory() as directory:
        workspace = Path(directory)
        first_store = StateStore(workspace)
        goal = first_store.create_goal("Implement durable legacy behavior")
        first_store.transition_goal(
            goal.id,
            GoalStatus.AWAITING_PLAN_APPROVAL,
        )
        legacy = first_store.create_plan(
            goal.id,
            "Legacy combined proposal",
            [task("T001")],
            **plan_basis("T001"),
        )
        first_store.close()

        second_store = StateStore(workspace)
        try:
            runtime = AgentRuntime(
                ScriptedProvider([inspect_call(), plan_call(), plan_pass()]),
                second_store,
                workspace,
            )
            assert runtime.active_goal().metadata[
                "legacy_semantic_enrichment_required"
            ]

            enriched = runtime.approve_plan(legacy.revision)

            assert enriched.revision == legacy.revision + 1
            assert enriched.status is PlanStatus.PENDING_APPROVAL
            restored = runtime.active_goal()
            assert restored.status is GoalStatus.AWAITING_PLAN_APPROVAL
            assert restored.metadata["accepted_semantic_fingerprint"]
            assert not restored.metadata.get(
                "legacy_semantic_enrichment_required",
                False,
            )
            runtime.close()
        finally:
            second_store.close()


def test_independent_review_scope_expansion_requires_new_approval() -> None:
    failed_review = {
        "tool_calls": [
            {
                "id": "review",
                "name": "submit_review",
                "args": {
                    "verdict": "fail",
                    "summary": "The requested external integration is absent.",
                    "issues": [
                        {
                            "severity": "high",
                            "title": "Install external dependency",
                            "details": (
                                "Install a new package and call an external service."
                            ),
                            "acceptance_criteria": [
                                "The external dependency is installed and configured."
                            ],
                        }
                    ],
                    "checked_task_ids": ["T001"],
                },
            }
        ]
    }
    with tempfile.TemporaryDirectory() as directory:
        workspace = Path(directory)
        store = StateStore(workspace)
        try:
            runtime = AgentRuntime(
                ScriptedProvider(
                    [
                        inspect_call(),
                        plan_call(),
                        plan_pass(),
                        _execution_turn(),
                        _finish_turn(),
                        failed_review,
                    ]
                ),
                store,
                workspace,
                approval=lambda *_: True,
            )
            initial = runtime.start_goal("Implement bounded durable behavior")
            runtime.approve_plan(initial.revision)
            result = runtime.run_slice(steps=2)
            assert not result.completed
            goal = runtime.active_goal()
            assert goal.status is GoalStatus.AWAITING_PLAN_APPROVAL
            revised = runtime.latest_plan()
            assert revised.revision == initial.revision + 1
            assert revised.status is PlanStatus.PENDING_APPROVAL
            assert goal.metadata["convergence_state"] == (
                "scope_expansion_pending"
            )
        finally:
            store.close()


def test_repeated_successful_reads_are_not_durable_progress() -> None:
    with tempfile.TemporaryDirectory() as directory:
        workspace = Path(directory)
        (workspace / "artifact.txt").write_text("baseline\n", encoding="utf-8")
        store = StateStore(workspace)
        try:
            reads = [
                {
                    "tool_calls": [
                        {
                            "id": f"read-{index}",
                            "name": "read_file",
                            "args": {"path": "artifact.txt"},
                        }
                    ]
                }
                for index in range(3)
            ]
            runtime = AgentRuntime(
                ScriptedProvider(
                    [inspect_call(), plan_call(), plan_pass(), *reads]
                ),
                store,
                workspace,
                approval=lambda *_: True,
                sleeper=lambda _seconds: None,
            )
            plan = runtime.start_goal("Verify durable no-progress recovery")
            runtime.approve_plan(plan.revision)
            runtime.run_slice(steps=1)
            runtime.run_slice(steps=1)
            runtime.run_slice(steps=1)
            goal = runtime.active_goal()
            assert goal.metadata["no_progress_slices"] == 2
            assert goal.metadata["goal_attempt"] == 2
            assert goal.metadata["auto_retryable"]
        finally:
            store.close()


def test_operational_guidance_does_not_create_a_quality_refinement_blocker() -> None:
    with tempfile.TemporaryDirectory() as directory:
        workspace = Path(directory)
        store = StateStore(workspace)
        try:
            runtime = AgentRuntime(
                ScriptedProvider([inspect_call(), plan_call(), plan_pass()]),
                store,
                workspace,
                approval=lambda *_: True,
            )
            plan = runtime.start_goal("Implement durable behavior")
            runtime.approve_plan(plan.revision)
            store.update_goal_metadata(
                plan.goal_id,
                quality_target={"id": "quality-test", "artifact_ids": []},
                convergence_state="refining",
                refinement_actions=[
                    {
                        "id": "refinement-001",
                        "feedback": (
                            "Tool permissions are now available. Retry pytest "
                            "and continue."
                        ),
                        "status": "pending",
                    }
                ],
            )

            runtime.add_guidance(
                "The authoritative command is python -m pytest. Complete T001 "
                "and call finish_goal without changing scope."
            )

            goal = runtime.active_goal()
            assert goal.metadata["convergence_state"] == "reverifying"
            assert goal.metadata["refinement_actions"][0]["status"] == "resolved"
            assert any(
                event.event_type == "guidance.operational"
                for event in store.list_recent_events(goal.id, limit=30)
            )
            runtime.close()
        finally:
            store.close()


def test_real_python_project_fails_then_repairs_and_completes() -> None:
    request = "Create and verify a tiny arithmetic package."
    criterion = "The arithmetic package passes its executable pytest suite."
    proposal = {
        "tool_calls": [
            {
                "id": "plan",
                "name": "propose_plan",
                "args": {
                    "semantic_goal": {
                        "original_request": request,
                        "interpreted_outcome": request,
                        "requested_effects": [
                            "read_workspace",
                            "mutate_workspace",
                            "execute_code",
                        ],
                        "required_outcomes": [
                            "A tested arithmetic package exists."
                        ],
                        "constraints": ["Use only the Python standard library."],
                        "exclusions": [],
                        "acceptance_criteria": [criterion],
                        "unresolved_decisions": [],
                        "repository_evidence_refs": ["inspection:I001"],
                    },
                    "summary": "Create the package and prove it with pytest.",
                    "applicability_evidence": [
                        {
                            "fact": "The temporary workspace was inspected.",
                            "source": "inspection:I001",
                            "supports_tasks": ["T001"],
                        }
                    ],
                    "execution_strategy": (
                        "Create package and test files, run pytest, repair any "
                        "failure, rerun pytest, and request independent review."
                    ),
                    "expected_changes": [
                        {
                            "path": "tinycalc/__init__.py",
                            "intent": "Implement the package.",
                            "basis": "repository_convention",
                            "evidence_refs": ["inspection:I001"],
                            "supports_tasks": ["T001"],
                        },
                        {
                            "path": "tests/test_tinycalc.py",
                            "intent": "Add executable regression coverage.",
                            "basis": "repository_convention",
                            "evidence_refs": ["inspection:I001"],
                            "supports_tasks": ["T001"],
                        },
                    ],
                    "tasks": [
                        {
                            "title": "Implement and verify tinycalc",
                            "description": (
                                "Create the package, execute pytest, and repair "
                                "the injected first failure."
                            ),
                            "acceptance_criteria": [criterion],
                            "verification": ["Run python -m pytest -q."],
                            "risk": "medium",
                        }
                    ],
                },
            }
        ]
    }
    failing_implementation = {
        "tool_calls": [
            {
                "id": "bad-code",
                "name": "write_file",
                "args": {
                    "path": "tinycalc/__init__.py",
                    "content": "def add(a, b):\n    return a - b\n",
                },
            },
            {
                "id": "tests",
                "name": "write_file",
                "args": {
                    "path": "tests/test_tinycalc.py",
                    "content": (
                        "from tinycalc import add\n\n"
                        "def test_add():\n"
                        "    assert add(2, 3) == 5\n"
                    ),
                },
            },
            {
                "id": "red",
                "name": "run_command",
                "args": {"command": "python -m pytest -q"},
            },
        ]
    }
    repaired = {
        "tool_calls": [
            {
                "id": "fix",
                "name": "write_file",
                "args": {
                    "path": "tinycalc/__init__.py",
                    "content": "def add(a, b):\n    return a + b\n",
                },
            },
            {
                "id": "green",
                "name": "run_command",
                "args": {"command": "python -m pytest -q"},
            },
            {
                "id": "done",
                "name": "update_task",
                "args": {
                    "task_id": "T001",
                    "status": "done",
                    "note": "pytest passed after the focused repair",
                    "evidence": ["python -m pytest -q passed"],
                },
            },
            {
                "id": "finish",
                "name": "finish_goal",
                "args": {
                    "summary": "tinycalc is implemented and pytest passes.",
                    "evidence": ["green pytest action"],
                },
            },
        ]
    }
    with tempfile.TemporaryDirectory() as directory:
        workspace = Path(directory)
        store = StateStore(workspace)
        try:
            runtime = AgentRuntime(
                ScriptedProvider(
                    [
                        inspect_call(),
                        proposal,
                        plan_pass(),
                        failing_implementation,
                        repaired,
                        review_pass(),
                    ]
                ),
                store,
                workspace,
                approval=lambda *_: True,
            )
            plan = runtime.start_goal(request)
            runtime.approve_plan(plan.revision)
            result = runtime.continue_until_boundary()
            assert result.completed
            assert result.disposition == "verified"
            command_actions = [
                item
                for item in store.list_actions(plan.goal_id)
                if item["tool_name"] == "run_command"
            ]
            assert [item["status"] for item in command_actions] == [
                "failed",
                "completed",
            ]
            assert "return a + b" in (
                workspace / "tinycalc" / "__init__.py"
            ).read_text(encoding="utf-8")
            runtime.close()
        finally:
            store.close()
