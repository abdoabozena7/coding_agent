from __future__ import annotations

import tempfile
import unittest
import sqlite3
from pathlib import Path

from agent.models import (
    Delegation,
    DelegationStatus,
    GoalStatus,
    RoleProfile,
    TaskGraphError,
    TaskStatus,
)
from agent.store import CompletionGateError, StalePlanError, StateCorruptionError, StateStore


def task(task_id: str, *, depends_on=()):
    return {
        "id": task_id,
        "title": f"Task {task_id}",
        "description": f"Implement {task_id}",
        "depends_on": list(depends_on),
        "acceptance_criteria": [f"{task_id} behavior is proven"],
        "verification": [f"verify {task_id}"],
        "risk": "medium",
    }


def plan_basis(*task_ids: str):
    ids = list(task_ids)
    return {
        "applicability_evidence": [
            {
                "fact": "The inspected test workspace needs these tasks.",
                "source": "test workspace",
                "supports_tasks": ids,
            }
        ],
        "execution_strategy": "Apply the test changes and verify the resulting durable state.",
        "expected_changes": [
            {
                "path": "workspace/",
                "intent": "Create the implementation artifacts covered by the plan.",
                "supports_tasks": ids,
            }
        ],
    }


class StateStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp.name)
        self.store = StateStore(self.workspace)

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def _pending_goal(self):
        goal = self.store.create_goal("Build a durable harness")
        self.store.transition_goal(goal.id, GoalStatus.AWAITING_PLAN_APPROVAL)
        return goal

    def test_plan_approval_is_bound_to_latest_revision_and_fingerprint(self):
        goal = self._pending_goal()
        first = self.store.create_plan(goal.id, "first", [task("T001")], **plan_basis("T001"))
        second = self.store.create_plan(
            goal.id,
            "second",
            [task("T001"), task("T002", depends_on=("T001",))],
            **plan_basis("T001", "T002"),
        )
        with self.assertRaises(StalePlanError):
            self.store.approve_plan(goal.id, first.revision)
        with self.assertRaises(StalePlanError):
            self.store.approve_plan(goal.id, second.revision, expected_fingerprint="tampered")
        accepted, approval = self.store.approve_plan(
            goal.id, second.revision, expected_fingerprint=second.fingerprint
        )
        self.assertEqual(accepted.status.value, "accepted")
        self.assertEqual(approval.revision, 2)
        restored = self.store.get_goal(goal.id)
        self.assertEqual(restored.status, GoalStatus.RUNNING)
        self.assertEqual(restored.active_plan_revision, 2)

    def test_applicability_evidence_is_persisted_and_fingerprint_bound(self):
        goal = self._pending_goal()
        first_basis = plan_basis("T001")
        first = self.store.create_plan(goal.id, "same plan", [task("T001")], **first_basis)
        changed_basis = plan_basis("T001")
        changed_basis["applicability_evidence"][0]["fact"] = "A different inspected workspace fact supports the task."
        second = self.store.create_plan(goal.id, "same plan", [task("T001")], **changed_basis)

        self.assertNotEqual(first.fingerprint, second.fingerprint)
        restored = self.store.get_plan(goal.id, second.revision)
        self.assertEqual(
            restored.applicability_evidence[0]["fact"],
            "A different inspected workspace fact supports the task.",
        )
        self.assertEqual(restored.execution_strategy, changed_basis["execution_strategy"])
        self.assertEqual(restored.expected_changes[0]["path"], "workspace/")

    def test_v1_state_migrates_and_legacy_pending_plan_requires_replan(self):
        goal = self._pending_goal()
        plan = self.store.create_plan(
            goal.id,
            "legacy pending plan",
            [task("T001")],
            **plan_basis("T001"),
        )
        path = self.store.path
        self.store.close()
        connection = sqlite3.connect(path)
        try:
            connection.execute("ALTER TABLE plans DROP COLUMN expected_changes_json")
            connection.execute("ALTER TABLE plans DROP COLUMN execution_strategy")
            connection.execute("ALTER TABLE plans DROP COLUMN applicability_json")
            connection.execute("PRAGMA user_version=1")
            connection.commit()
        finally:
            connection.close()

        self.store = StateStore(self.workspace)

        migrated = self.store.get_plan(goal.id, plan.revision)
        self.assertEqual(migrated.applicability_evidence, ())
        with self.assertRaisesRegex(StalePlanError, "lacks fingerprinted applicability evidence"):
            self.store.approve_plan(goal.id, plan.revision)

    def test_v2_state_migrates_to_v4_without_losing_goal_or_plan(self):
        goal = self._pending_goal()
        plan = self.store.create_plan(
            goal.id,
            "v2 accepted shape",
            [task("T001")],
            **plan_basis("T001"),
        )
        path = self.store.path
        self.store.close()
        connection = sqlite3.connect(path)
        try:
            for table in (
                "brain_entries_fts",
                "memory_access",
                "resource_leases",
                "prompt_traces",
                "artifacts",
                "agent_runs",
                "brain_entries",
                "work_nodes",
                "ultra_runs",
            ):
                connection.execute(f"DROP TABLE IF EXISTS {table}")
            connection.execute("PRAGMA user_version=2")
            connection.commit()
        finally:
            connection.close()

        self.store = StateStore(self.workspace)

        self.assertEqual(self.store.get_goal(goal.id).id, goal.id)
        self.assertEqual(self.store.get_plan(goal.id, plan.revision).fingerprint, plan.fingerprint)
        migrated = sqlite3.connect(path)
        try:
            self.assertEqual(migrated.execute("PRAGMA user_version").fetchone()[0], 9)
            self.assertIsNotNone(
                migrated.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='ultra_runs'"
                ).fetchone()
            )
        finally:
            migrated.close()

    def test_invalid_dependency_graph_is_never_persisted(self):
        goal = self._pending_goal()
        with self.assertRaises(TaskGraphError):
            self.store.create_plan(
                goal.id,
                "cycle",
                [task("T001", depends_on=("T002",)), task("T002", depends_on=("T001",))],
                **plan_basis("T001", "T002"),
            )
        self.assertIsNone(self.store.get_latest_plan(goal.id))

    def test_plan_basis_cannot_claim_evidence_for_unknown_tasks(self):
        goal = self._pending_goal()
        basis = plan_basis("MISSING")
        with self.assertRaisesRegex(ValueError, "outside this plan"):
            self.store.create_plan(goal.id, "invalid basis", [task("T001")], **basis)
        self.assertIsNone(self.store.get_latest_plan(goal.id))

    def test_task_completion_requires_dependencies_and_evidence(self):
        goal = self._pending_goal()
        plan = self.store.create_plan(
            goal.id, "ordered", [task("T001"), task("T002", depends_on=("T001",))],
            **plan_basis("T001", "T002"),
        )
        self.store.approve_plan(goal.id, plan.revision)
        with self.assertRaises(CompletionGateError):
            self.store.transition_task(goal.id, 1, "T002", TaskStatus.IN_PROGRESS)
        self.store.transition_task(goal.id, 1, "T001", TaskStatus.IN_PROGRESS)
        with self.assertRaises(CompletionGateError):
            self.store.transition_task(goal.id, 1, "T001", TaskStatus.COMPLETED)
        self.store.transition_task(
            goal.id,
            1,
            "T001",
            TaskStatus.COMPLETED,
            evidence=["offline state test passed"],
        )
        self.store.transition_task(goal.id, 1, "T002", TaskStatus.IN_PROGRESS)

    def test_restart_recovery_marks_uncertain_work_and_never_replays(self):
        goal = self._pending_goal()
        plan = self.store.create_plan(goal.id, "recover", [task("T001")], **plan_basis("T001"))
        self.store.approve_plan(goal.id, 1)
        self.store.transition_task(goal.id, 1, "T001", TaskStatus.IN_PROGRESS)
        delegation = self.store.create_delegation(
            Delegation(
                goal_id=goal.id,
                task_id="T001",
                plan_revision=1,
                brief="Inspect the interrupted write",
                role=RoleProfile(name="recovery inspector", mission="Determine side-effect state"),
            )
        )
        self.store.transition_delegation(delegation.id, DelegationStatus.IN_PROGRESS)
        action_id = self.store.begin_action(
            goal.id, "write_file", {"path": "x.py", "content": "redacted"}, task_id="T001", mutating=True
        )
        self.store.close()
        self.store = StateStore(self.workspace)
        report = self.store.recover_inflight()
        self.assertEqual(report.task_ids, ("T001",))
        self.assertIn(delegation.id, report.delegation_ids)
        self.assertIn(action_id, report.action_ids)
        self.assertEqual(self.store.list_tasks(goal.id, 1)[0].status, TaskStatus.UNCERTAIN)
        self.assertEqual(self.store.list_delegations(goal.id)[0].status, DelegationStatus.UNCERTAIN)
        self.assertEqual(self.store.list_actions(goal.id)[0]["status"], "uncertain")
        self.assertEqual(self.store.get_goal(goal.id).status, GoalStatus.RECOVERING)
        self.assertEqual(self.store.list_actions(goal.id)[0]["id"], action_id)

    def test_active_goal_and_events_survive_reopen(self):
        goal = self.store.create_goal("A long goal")
        self.store.close()
        self.store = StateStore(self.workspace)
        self.assertEqual(self.store.load_active_goal().id, goal.id)
        self.assertEqual(self.store.list_events(goal.id)[0].event_type, "goal.created")

    def test_graceful_checkpoint_does_not_make_task_uncertain_on_reopen(self):
        goal = self._pending_goal()
        plan = self.store.create_plan(goal.id, "resume", [task("T001")], **plan_basis("T001"))
        self.store.approve_plan(goal.id, plan.revision)
        self.store.transition_task(goal.id, 1, "T001", TaskStatus.IN_PROGRESS)
        self.store.close()
        self.store = StateStore(self.workspace)
        report = self.store.recover_inflight()
        self.assertFalse(report.changed)
        self.assertEqual(self.store.list_tasks(goal.id, 1)[0].status, TaskStatus.IN_PROGRESS)
        self.assertEqual(self.store.get_goal(goal.id).status, GoalStatus.RUNNING)

    def test_recent_event_page_tracks_the_newest_events(self):
        goal = self.store.create_goal("Many durable events")
        for index in range(150):
            self.store.append_event("marker", goal_id=goal.id, payload={"summary": f"event-{index}"})
        recent = self.store.list_recent_events(goal.id, limit=10)
        self.assertEqual(len(recent), 10)
        self.assertEqual(recent[-1].payload["summary"], "event-149")
        self.assertLess(recent[0].sequence, recent[-1].sequence)


class CorruptionTests(unittest.TestCase):
    def test_corruption_is_reported_not_silently_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = root / ".coding-agent"
            state.mkdir()
            (state / "state.db").write_bytes(b"not sqlite")
            with self.assertRaises(StateCorruptionError):
                StateStore(root)
            self.assertEqual((state / "state.db").read_bytes(), b"not sqlite")

    def test_normal_git_repo_gets_local_exclude_without_tracked_file_edit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".git" / "info").mkdir(parents=True)
            exclude = root / ".git" / "info" / "exclude"
            exclude.write_text("# local excludes\n", encoding="utf-8")
            store = StateStore(root)
            store.close()
            content = exclude.read_text(encoding="utf-8")
            self.assertIn("/.coding-agent/", content.splitlines())
            self.assertFalse((root / ".gitignore").exists())


if __name__ == "__main__":
    unittest.main()
