from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

from fastapi.testclient import TestClient

from agent.config import RuntimeConfig
from agent.events import EventBus, UIEvent
from agent.models import GoalStatus
from agent.quality import ChangeSetStatus, ChangeSetV1
from agent.runtime import AgentRuntime
from agent.store import StalePlanError, StateStore
from agent.testing import ScriptedProvider
from agent.ultra_models import (
    AgentRun,
    AgentRunStatus,
    TaskContractV1,
    UltraRun,
    UltraRunStatus,
    WorkNode,
    WorkNodeStatus,
)
from agent.web_views.schemas import PlanPayload
from agent.web_views.security import SessionSecurity
from agent.web_views.server import LocalWebServer, create_app
from agent.web_views.service import CoreWebAdapter


def basis(task_id: str = "T001") -> dict[str, object]:
    return {
        "applicability_evidence": (
            {
                "source": "repository inspection",
                "fact": "The implementation area exists.",
                "supports_tasks": [task_id],
            },
        ),
        "execution_strategy": "Implement the task and run focused verification.",
        "expected_changes": (
            {
                "path": "agent/example.py",
                "intent": "Implement the task.",
                "supports_tasks": [task_id],
            },
        ),
    }


def task_value(task_id: str = "T001", title: str = "Implement the feature") -> dict[str, object]:
    return {
        "id": task_id,
        "title": title,
        "description": "Implement the bounded behavior.",
        "status": "pending",
        "depends_on": [],
        "acceptance_criteria": ["The behavior works."],
        "verification": ["Run the focused test."],
        "risk": "medium",
    }


class LocalWebViewTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temporary.name)
        self.store = StateStore(self.workspace)
        self.events = EventBus()
        self.runtime = AgentRuntime(
            ScriptedProvider([]),
            self.store,
            self.workspace,
            events=self.events,
            config=replace(RuntimeConfig(), repository_index_warmup_files=0),
            session_id="session-a82f",
        )
        self.goal = self.store.create_goal(
            "Build the local artifact workspace",
            session_id=self.runtime.session_id,
        )
        self.plan = self.store.create_plan(
            self.goal.id,
            "Implement the first vertical slice.",
            [task_value()],
            **basis(),
        )
        self.store.transition_goal(
            self.goal.id,
            GoalStatus.AWAITING_PLAN_APPROVAL,
            reason="test plan ready",
        )
        self.adapter = CoreWebAdapter(self.runtime)
        self.security = SessionSecurity(self.runtime.session_id)
        self.app = create_app(self.adapter, self.security)
        self.app.state.port = 43210
        self.client = TestClient(self.app, base_url="http://127.0.0.1:43210")
        response = self.client.get(
            f"/sessions/{self.runtime.session_id}/plan?token={self.security.token}"
        )
        self.assertEqual(response.status_code, 200)

    def tearDown(self) -> None:
        self.client.close()
        self.runtime.close()
        self.store.close()
        self.temporary.cleanup()

    def csrf_headers(self) -> dict[str, str]:
        return {"X-GA3BAD-CSRF": self.client.cookies.get("ga3bad_csrf")}

    def plan_payload(self, *, revision: int = 1, title: str = "Implement the edited feature"):
        snapshot = self.client.get(
            f"/api/sessions/{self.runtime.session_id}/plan"
        ).json()
        task = dict(snapshot["tasks"][0])
        for protected_field in ("status", "paused", "disabled"):
            task.pop(protected_field, None)
        task["title"] = title
        return {
            "base_revision": revision,
            "summary": "Implement the edited vertical slice.",
            "tasks": [task],
            "global_constraints": ["Do not modify the landing page."],
            "protected_paths": ["legacy/"],
            "change_note": "User clarified the task.",
        }


class PlanStudioIntegrationTests(LocalWebViewTestCase):
    def test_revision_and_approval_are_separate_explicit_actions(self):
        captured: list[UIEvent] = []
        unsubscribe = self.events.subscribe(captured.append)
        try:
            response = self.client.post(
                f"/api/sessions/{self.runtime.session_id}/plan/revision",
                headers=self.csrf_headers(),
                json=self.plan_payload(),
            )
        finally:
            unsubscribe()

        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["revision"], 2)
        self.assertFalse(body["approved"])
        latest = self.store.get_latest_plan(self.goal.id)
        self.assertEqual(latest.revision, 2)
        self.assertEqual(latest.tasks[0].title, "Implement the edited feature")
        self.assertIsNone(self.store.get_goal(self.goal.id).active_plan_revision)
        self.assertTrue(any(event.kind == "plan.revision.created" for event in captured))
        durable = self.store.list_recent_events(self.goal.id, limit=20)
        self.assertTrue(any(event.event_type == "plan.revision.created" for event in durable))

        approved = self.client.post(
            f"/api/sessions/{self.runtime.session_id}/plan/approve",
            headers=self.csrf_headers(),
            json={"revision": 2},
        )
        self.assertEqual(approved.status_code, 200, approved.text)
        accepted_goal = self.store.get_goal(self.goal.id)
        self.assertEqual(accepted_goal.active_plan_revision, 2)
        self.assertEqual(accepted_goal.status, GoalStatus.RUNNING)
        self.assertTrue(accepted_goal.metadata["strategy_locked"])
        self.assertEqual(accepted_goal.metadata["interaction_mode"], "working")
        self.assertEqual(accepted_goal.metadata["execution_strategy"], "staged")

    def test_plan_snapshot_exposes_model_aware_diagnostics_only_as_data(self):
        self.store.update_goal_metadata(
            self.goal.id,
            semantic_goal={
                "required_outcomes": ["Render the requested interactive artifact."],
                "acceptance_criteria": ["The artifact runs without browser errors."],
                "constraints": ["Keep the implementation inside the workspace."],
                "exclusions": [],
            },
            strategy_decision={
                "strategy": "staged",
                "reasons": ["task demand fits the selected model capability envelope"],
                "capability_fingerprint": "capability",
                "demand_fingerprint": "demand",
                "max_concurrency": 1,
                "locked": False,
                "version": 1,
            },
        )
        snapshot = self.client.get(
            f"/api/sessions/{self.runtime.session_id}/plan"
        ).json()
        self.assertEqual(snapshot["interaction_mode"], "working")
        self.assertEqual(snapshot["execution_strategy"], "staged")
        self.assertFalse(snapshot["strategy_locked"])
        self.assertIn("capability_envelope", snapshot)
        self.assertIn("task_demand", snapshot)
        self.assertEqual(
            snapshot["semantic_goal"]["required_outcomes"],
            ["Render the requested interactive artifact."],
        )
        self.assertEqual(snapshot["strategy_decision"]["strategy"], "staged")
        self.assertEqual(snapshot["execution_nodes"], [])

    def test_rejects_stale_plan_revision_without_overwriting_latest(self):
        first = self.client.post(
            f"/api/sessions/{self.runtime.session_id}/plan/revision",
            headers=self.csrf_headers(),
            json=self.plan_payload(title="First accepted edit"),
        )
        self.assertEqual(first.status_code, 200, first.text)
        stale = self.client.post(
            f"/api/sessions/{self.runtime.session_id}/plan/revision",
            headers=self.csrf_headers(),
            json=self.plan_payload(revision=1, title="Stale overwrite"),
        )
        self.assertEqual(stale.status_code, 409)
        self.assertEqual(stale.json()["current_revision"], 2)
        self.assertEqual(self.store.get_latest_plan(self.goal.id).tasks[0].title, "First accepted edit")

    def test_draft_is_central_but_does_not_change_active_plan(self):
        response = self.client.post(
            f"/api/sessions/{self.runtime.session_id}/plan/draft",
            headers=self.csrf_headers(),
            json=self.plan_payload(title="Draft only"),
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.store.get_latest_plan(self.goal.id).revision, 1)
        state = self.store.get_workflow_session(self.runtime.session_id)["state"]
        self.assertEqual(state["web_plan_draft"]["tasks"][0]["title"], "Draft only")

    def test_backend_cycle_and_task_validation_are_enforced(self):
        payload = self.plan_payload()
        first = dict(payload["tasks"][0])
        first["dependencies"] = ["T002"]
        second = dict(first)
        second.update(
            {
                "id": "T002",
                "title": "Second task",
                "dependencies": ["T001"],
            }
        )
        payload["tasks"] = [first, second]
        response = self.client.post(
            f"/api/sessions/{self.runtime.session_id}/plan/revision",
            headers=self.csrf_headers(),
            json=payload,
        )
        self.assertEqual(response.status_code, 422)
        self.assertIn("cycle", response.json()["error"])
        self.assertEqual(self.store.get_latest_plan(self.goal.id).revision, 1)

    def test_task_status_is_read_only_and_active_plan_cannot_be_mutated(self):
        forged = self.plan_payload()
        forged["tasks"][0]["status"] = "completed"
        rejected = self.client.post(
            f"/api/sessions/{self.runtime.session_id}/plan/revision",
            headers=self.csrf_headers(),
            json=forged,
        )
        self.assertEqual(rejected.status_code, 422)
        self.assertEqual(
            rejected.json()["details"][0]["loc"][-1],
            "status",
        )
        self.assertEqual(self.store.get_latest_plan(self.goal.id).tasks[0].status.value, "pending")

        self.runtime.approve_plan(1)
        active_edit = self.client.post(
            f"/api/sessions/{self.runtime.session_id}/plan/revision",
            headers=self.csrf_headers(),
            json=self.plan_payload(),
        )
        self.assertEqual(active_edit.status_code, 422)
        self.assertIn("read-only", active_edit.json()["error"])

    def test_workspace_context_routes_only_mandatory_gates(self):
        context = self.client.get(
            f"/api/sessions/{self.runtime.session_id}/workspace"
        )
        self.assertEqual(context.status_code, 200)
        self.assertEqual(context.json()["required_view"], "plan")
        self.assertEqual(context.json()["attention"]["action"]["label"], "Review plan")
        self.runtime.approve_plan(1)
        running = self.client.get(
            f"/api/sessions/{self.runtime.session_id}/workspace"
        ).json()
        self.assertIsNone(running["required_view"])
        self.assertEqual(running["attention"]["state"], "working")

    def test_future_queue_is_not_available_before_plan_approval(self):
        response = self.client.post(
            f"/api/sessions/{self.runtime.session_id}/queue",
            headers=self.csrf_headers(),
            json={"text": "Run this later"},
        )
        self.assertEqual(response.status_code, 422)
        self.assertEqual(
            self.store.list_queued_prompts(self.runtime.session_id),
            (),
        )

    def test_store_expected_parent_revision_is_atomic(self):
        with self.assertRaises(StalePlanError):
            self.store.create_plan(
                self.goal.id,
                "Stale plan",
                [task_value()],
                **basis(),
                expected_parent_revision=0,
            )
        self.assertEqual(self.store.get_latest_plan(self.goal.id).revision, 1)


class SecurityAndLifecycleTests(LocalWebViewTestCase):
    def test_session_token_csrf_and_cross_session_access(self):
        unauthenticated = TestClient(self.app, base_url="http://127.0.0.1:43210")
        try:
            self.assertEqual(
                unauthenticated.get(
                    f"/api/sessions/{self.runtime.session_id}/plan"
                ).status_code,
                401,
            )
        finally:
            unauthenticated.close()
        self.assertEqual(
            self.client.post(
                f"/api/sessions/{self.runtime.session_id}/plan/draft",
                json=self.plan_payload(),
            ).status_code,
            403,
        )
        self.assertEqual(
            self.client.get("/api/sessions/foreign-session/plan").status_code,
            404,
        )

    def test_event_serialization_has_timestamp_and_data(self):
        event = self.events.publish(
            "web.view.opened",
            "Plan Studio opened.",
            session_id=self.runtime.session_id,
            source="terminal",
        )
        self.assertEqual(event.kind, "web.view.opened")
        self.assertEqual(event.data["session_id"], self.runtime.session_id)
        self.assertIn("+00:00", event.timestamp)
        serialized = event.to_dict()
        self.assertTrue(serialized["event_id"].startswith("event_"))
        self.assertEqual(serialized["source"], "terminal")

    def test_local_server_stops_and_invalidates_session_with_runtime(self):
        server = LocalWebServer(self.runtime).start()
        self.runtime.local_web_server = server
        port = server.port
        self.assertTrue(server.running)
        self.assertGreater(port, 0)
        self.runtime.close()
        self.assertFalse(server.running)
        self.assertTrue(server.security.expired)

    def test_review_ready_event_opens_mandatory_view_without_blocking(self):
        server = LocalWebServer(self.runtime).start()
        self.runtime.local_web_server = server
        with mock.patch("agent.web_views.server.webbrowser.open", return_value=True) as opened:
            self.events.publish(
                "checkpoint.review_ready",
                "Checkpoint ready for review.",
                checkpoint_id="checkpoint-1",
                source="agent_runtime",
            )
            deadline = time.monotonic() + 2
            while not opened.called and time.monotonic() < deadline:
                time.sleep(0.01)
        self.assertTrue(opened.called)
        self.assertIn("/review?token=", opened.call_args.args[0])


class PromptQueueIntegrationTests(LocalWebViewTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.runtime.approve_plan(1)

    def _enqueue(self, text: str) -> dict[str, object]:
        response = self.client.post(
            f"/api/sessions/{self.runtime.session_id}/queue",
            headers=self.csrf_headers(),
            json={"text": text},
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()["item"]

    def test_pending_queue_reorders_without_moving_running_work(self):
        first = self._enqueue("First request")
        second = self._enqueue("Second request")
        third = self._enqueue("Third request")
        claimed = self.store.claim_next_prompt(self.runtime.session_id)
        self.assertEqual(claimed.id, first["id"])

        response = self.client.patch(
            f"/api/sessions/{self.runtime.session_id}/queue/order",
            headers=self.csrf_headers(),
            json={"ordered_ids": [third["id"], second["id"]]},
        )
        self.assertEqual(response.status_code, 200, response.text)
        items = response.json()["queue"]["items"]
        self.assertEqual(items[0]["id"], first["id"])
        self.assertEqual(items[0]["status"], "running")
        self.assertEqual(
            [item["id"] for item in items if item["status"] == "pending"],
            [third["id"], second["id"]],
        )

    def test_stale_queue_reorder_is_atomic(self):
        first = self._enqueue("One")
        second = self._enqueue("Two")
        before = [
            item.id
            for item in self.store.list_queued_prompts(self.runtime.session_id)
        ]
        response = self.client.patch(
            f"/api/sessions/{self.runtime.session_id}/queue/order",
            headers=self.csrf_headers(),
            json={"ordered_ids": [second["id"]]},
        )
        self.assertEqual(response.status_code, 409)
        after = [
            item.id
            for item in self.store.list_queued_prompts(self.runtime.session_id)
        ]
        self.assertEqual(after, before)
        self.assertIn(first["id"], after)


class ReviewAndAgentIntegrationTests(LocalWebViewTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.runtime.approve_plan(1)
        self.run = self.store.create_ultra_run(
            UltraRun(
                goal_id=self.goal.id,
                provider="test",
                model="offline",
                status=UltraRunStatus.RUNNING,
                plan_revision=1,
                master_plan_fingerprint=self.plan.fingerprint,
                master_approved=True,
                config={"max_depth": 5, "max_nodes": 50},
            )
        )
        self.node = self.store.create_work_node(
            WorkNode(
                ultra_run_id=self.run.id,
                title="Implement authentication",
                objective="Implement and verify authentication.",
                contract=TaskContractV1(
                    objective="Implement and verify authentication.",
                    success_criteria=("Authentication works.",),
                    write_paths=("agent/auth.py",),
                ),
                status=WorkNodeStatus.REVIEWING,
                assigned_role="backend",
            )
        )
        self.agent = self.store.create_agent_run(
            AgentRun(
                ultra_run_id=self.run.id,
                work_node_id=self.node.id,
                role="backend",
                provider="test",
                model="offline",
                phase="implementation",
                status=AgentRunStatus.COMPLETED,
            )
        )
        diff = (
            "diff --git a/agent/auth.py b/agent/auth.py\n"
            "--- a/agent/auth.py\n"
            "+++ b/agent/auth.py\n"
            "@@ -1,2 +1,3 @@\n"
            " def login(user):\n"
            "-    return False\n"
            "+    validate(user)\n"
            "+    return True\n"
        )
        self.checkpoint = self.store.save_change_set(
            ChangeSetV1(
                ultra_run_id=self.run.id,
                responsible_agent_id=self.agent.id,
                parent_id=self.node.id,
                status=ChangeSetStatus.REVIEWING,
                changed_files=("agent/auth.py",),
                diff=diff,
                metadata={"reason": "Implement the approved login flow."},
            )
        )

    def test_review_snapshot_includes_the_recorded_diff_and_hunks(self):
        response = self.client.get(
            f"/api/sessions/{self.runtime.session_id}/review"
        )
        self.assertEqual(response.status_code, 200, response.text)
        file = response.json()["files"][0]
        self.assertEqual(file["path"], "agent/auth.py")
        self.assertIn("+    return True", file["diff"])
        self.assertEqual(file["hunks"][0]["id"], "agent/auth.py:h1")
        self.assertIn("-    return False", file["hunks"][0]["content"])

    def test_submit_change_review_persists_decisions_and_starts_fixer(self):
        response = self.client.post(
            f"/api/sessions/{self.runtime.session_id}/review/submit",
            headers=self.csrf_headers(),
            json={
                "checkpoint_id": self.checkpoint.id,
                "decisions": [
                    {
                        "target_type": "hunk",
                        "file_path": "agent/auth.py",
                        "hunk_id": "agent/auth.py:h1",
                        "decision": "changes_requested",
                        "reason": "Add the invalid-user branch.",
                    }
                ],
                "comments": [
                    {
                        "file_path": "agent/auth.py",
                        "hunk_id": "agent/auth.py:h1",
                        "line": 2,
                        "body": "Cover the failure path here.",
                    }
                ],
                "summary": "One focused correction is required.",
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(response.json()["fixer_started"])
        saved = self.store.list_change_sets(self.run.id)[0]
        self.assertEqual(saved.status, ChangeSetStatus.BLOCKED)
        self.assertEqual(saved.metadata["latest_user_review"]["comments"][0]["line"], 2)
        fixer = self.store.get_work_node(self.node.id)
        self.assertEqual(fixer.status, WorkNodeStatus.FIXING)
        self.assertIn("Preserve accepted files and hunks", fixer.checkpoint)
        self.assertIn("Add the invalid-user branch", fixer.checkpoint)

    def test_review_validation_rejects_foreign_files_and_invalid_hunks(self):
        response = self.client.post(
            f"/api/sessions/{self.runtime.session_id}/review/submit",
            headers=self.csrf_headers(),
            json={
                "checkpoint_id": self.checkpoint.id,
                "decisions": [
                    {
                        "target_type": "hunk",
                        "file_path": "outside.py",
                        "hunk_id": "outside.py:h9",
                        "decision": "accepted",
                    }
                ],
            },
        )
        self.assertEqual(response.status_code, 422)
        self.assertEqual(self.store.list_change_sets(self.run.id)[0].status, ChangeSetStatus.REVIEWING)

    def test_review_cannot_submit_with_an_unresolved_file(self):
        second = self.store.save_change_set(
            replace(
                self.checkpoint,
                id="changeset-two-files",
                changed_files=("agent/auth.py", "agent/session.py"),
            )
        )
        response = self.client.post(
            f"/api/sessions/{self.runtime.session_id}/review/submit",
            headers=self.csrf_headers(),
            json={
                "checkpoint_id": second.id,
                "decisions": [
                    {
                        "target_type": "file",
                        "file_path": "agent/auth.py",
                        "decision": "accepted",
                    }
                ],
            },
        )
        self.assertEqual(response.status_code, 422)
        self.assertIn("agent/session.py", response.json()["error"])
        saved = next(
            item for item in self.store.list_change_sets(self.run.id)
            if item.id == second.id
        )
        self.assertEqual(saved.status, ChangeSetStatus.REVIEWING)

    def test_agent_tree_uses_real_nodes_agents_and_side_panel_data(self):
        response = self.client.get(
            f"/api/sessions/{self.runtime.session_id}/agents"
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["nodes"][0]["id"], self.node.id)
        self.assertEqual(body["agents"][0]["id"], self.agent.id)
        self.assertEqual(body["agents"][0]["current_file"], "agent/auth.py")
        self.assertIsNone(body["agents"][0]["progress"])


if __name__ == "__main__":
    unittest.main()
