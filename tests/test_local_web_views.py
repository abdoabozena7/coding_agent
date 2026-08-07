from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import asyncio
import tempfile
import time
import unittest
from unittest import mock

from fastapi.testclient import TestClient
from starlette.requests import Request

from agent.config import RuntimeConfig
from agent.events import EventBus, UIEvent
from agent.models import Evidence, GoalStatus
from agent.model_catalog import ExecutionClass
from agent.quality import ChangeSetStatus, ChangeSetV1
from agent.runtime import AgentRuntime, ProviderUnavailableError
from agent.store import StalePlanError, StateStore
from agent.testing import ScriptedProvider
from agent.ultra_models import (
    AgentRun,
    AgentRunStatus,
    Artifact,
    TaskContractV1,
    UltraRun,
    UltraRunStatus,
    WorkNode,
    WorkNodeStatus,
)
from agent.web_views.schemas import PlanPayload, WorkspaceActionRequest
from agent.web_views.security import SessionSecurity
from agent.web_views.server import LocalWebServer, _error_code, create_app
from agent.web_views.service import CoreWebAdapter, _public_task_description


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
    def test_real_local_build_surface_starts_goal_without_a_second_router_call(self):
        self.store.transition_goal(
            self.goal.id,
            GoalStatus.CANCELLED,
            reason="start a fresh local build request",
        )
        with (
            mock.patch.object(
                type(self.runtime),
                "execution_class",
                new_callable=mock.PropertyMock,
                return_value="local",
            ),
            mock.patch.object(
                type(self.runtime),
                "provider_name",
                new_callable=mock.PropertyMock,
                return_value="ollama",
            ),
            mock.patch.object(self.runtime, "start_goal", return_value=None) as start_goal,
            mock.patch.object(self.runtime, "route_input") as route_input,
        ):
            response = self.client.post(
                f"/api/sessions/{self.runtime.session_id}/plan/request",
                headers=self.csrf_headers(),
                json={"request": "Build the local project"},
            )
        self.assertEqual(response.status_code, 200, response.text)
        start_goal.assert_called_once_with(
            "Build the local project",
            planning_only=True,
            execution_mode="plan",
            entry_surface="build",
        )
        route_input.assert_not_called()

    def test_provider_failure_is_a_named_recoverable_http_boundary(self):
        self.store.transition_goal(
            self.goal.id,
            GoalStatus.CANCELLED,
            reason="start a fresh request for the boundary test",
        )
        with mock.patch.object(
            self.runtime,
            "route_input",
            side_effect=ProviderUnavailableError("cloud provider quota exhausted; saved stage unchanged"),
        ):
            response = self.client.post(
                f"/api/sessions/{self.runtime.session_id}/plan/request",
                headers=self.csrf_headers(),
                json={"request": "Build the saved project"},
            )
        self.assertEqual(response.status_code, 503, response.text)
        self.assertEqual(response.json()["code"], "quota_exhausted")
        self.assertTrue(response.json()["retryable"])
        self.assertTrue(response.json()["saved_stage"])

    def test_legacy_internal_repair_suffix_is_hidden_from_plan_copy(self):
        description = (
            "Implement the calculator logic. Accepted repair requirements: "
            "ULTRA foundation/phase browser_scenarios failed after three targeted "
            "typed-return repairs: browser_scenarios.modules must be a non-empty array"
        )

        self.assertEqual(
            _public_task_description(description),
            "Implement the calculator logic.",
        )

    def test_awaiting_plan_projects_one_coherent_runtime_state(self):
        context = self.client.get(
            f"/api/sessions/{self.runtime.session_id}/workspace"
        ).json()
        plan = self.client.get(
            f"/api/sessions/{self.runtime.session_id}/plan"
        ).json()

        self.assertEqual(context["goal"]["plan_revision"], 1)
        self.assertEqual(context["workflow_identity"]["plan_revision"], 1)
        self.assertEqual(context["runtime"]["execution_class"], "local")
        self.assertEqual(context["runtime"]["phase"], "awaiting_approval")
        self.assertEqual(context["runtime"]["liveness"], "waiting")
        self.assertEqual(context["runtime"]["waiting_on"], "user")
        self.assertEqual(context["runtime"]["reason"], "Plan r1 is ready for review.")
        self.assertEqual(plan["interaction_mode"], "working")

    def test_automatic_local_fallback_is_projected_from_durable_metadata(self):
        self.store.update_goal_metadata(
            self.goal.id,
            provider_recovery={
                "state": "switched_to_local",
                "automatic_fallback": True,
                "provider": "ollama",
                "model": "coder",
                "execution_class": "local",
            },
        )
        context = self.client.get(
            f"/api/sessions/{self.runtime.session_id}/workspace"
        ).json()
        self.assertEqual(context["provider_recovery"]["state"], "switched_to_local")
        self.assertTrue(context["provider_recovery"]["automatic_fallback"])
        self.assertEqual(context["provider_recovery"]["model"], "coder")

    def test_project_settings_exposes_the_saved_profile_for_reopening(self):
        settings = self.client.get(
            f"/api/sessions/{self.runtime.session_id}/project-settings"
        )

        self.assertEqual(settings.status_code, 200, settings.text)
        body = settings.json()
        self.assertTrue(body["safe_checkpoint"])
        self.assertEqual(body["model"]["provider"], self.runtime.provider_name)
        self.assertEqual(body["access_level"], "normal")
        self.assertGreaterEqual(body["concurrency"], 1)
        self.assertIn("configured_provider", body["protection"])
        self.assertIn("reopen_behavior", body)

    def test_unified_thread_projection_has_stable_items_and_incremental_cursor(self):
        response = self.client.get(
            f"/api/sessions/{self.runtime.session_id}/thread?after_sequence=0"
        )
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertIn("user_message", {item["type"] for item in body["items"]})
        self.assertIn("plan", {item["type"] for item in body["items"]})
        self.assertIn("workflow_status", {item["type"] for item in body["items"]})
        ids = [item["item_id"] for item in body["items"]]
        self.assertEqual(len(ids), len(set(ids)))
        for item in body["items"]:
            self.assertIn("content_revision", item)
            self.assertIn("payload", item)
        repeat = self.client.get(
            f"/api/sessions/{self.runtime.session_id}/thread?after_sequence=0"
        )
        self.assertEqual(repeat.status_code, 200, repeat.text)
        self.assertEqual(body["items"], repeat.json()["items"])
        cursor = body["next_sequence"]
        next_page = self.client.get(
            f"/api/sessions/{self.runtime.session_id}/thread?after_sequence={cursor}"
        )
        self.assertEqual(next_page.status_code, 200, next_page.text)
        self.assertFalse(next_page.json()["items"])

    def test_thread_cursor_does_not_drop_items_when_limit_splits_activity_groups(self):
        full = self.client.get(
            f"/api/sessions/{self.runtime.session_id}/thread?after_sequence=0&limit=500"
        )
        self.assertEqual(full.status_code, 200, full.text)
        expected = {item["item_id"] for item in full.json()["items"]}

        cursor = 0
        seen: list[str] = []
        for _ in range(20):
            page = self.client.get(
                f"/api/sessions/{self.runtime.session_id}/thread?after_sequence={cursor}&limit=1"
            )
            self.assertEqual(page.status_code, 200, page.text)
            body = page.json()
            seen.extend(item["item_id"] for item in body["items"])
            if not body["has_more"]:
                break
            self.assertGreater(body["next_sequence"], cursor)
            cursor = body["next_sequence"]
        else:
            self.fail("small thread pages did not converge")

        self.assertEqual(set(seen), expected)
        self.assertEqual(len(seen), len(set(seen)))

    def test_inspector_projection_groups_environment_and_workflow_details(self):
        response = self.client.get(
            f"/api/sessions/{self.runtime.session_id}/inspector"
        )
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        for section in ("environment", "model", "git", "access", "sleep", "changes", "processes", "agents", "tree", "sources"):
            self.assertIn(section, body)
        self.assertEqual(body["environment"]["session_id"], self.runtime.session_id)
        section = self.client.get(
            f"/api/sessions/{self.runtime.session_id}/inspector?section=changes"
        )
        self.assertEqual(section.status_code, 200, section.text)
        self.assertEqual(section.json()["selected_section"], "changes")
        self.assertIn("section", section.json())

    def test_project_session_index_is_safe_for_the_left_rail(self):
        response = self.client.get("/api/sessions")
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["session_id"], self.runtime.session_id)
        self.assertEqual(body["projects"][0]["id"], self.runtime.session_id)
        self.assertTrue(any(item["id"] == self.goal.id for item in body["tasks"]))

    def test_goal_without_first_plan_is_a_visible_saved_request_not_a_404(self):
        self.store.transition_goal(
            self.goal.id,
            GoalStatus.CANCELLED,
            reason="replace fixture goal",
        )
        planning = self.store.create_goal(
            "Create a Three.js calculator and run it",
            session_id=self.runtime.session_id,
        )

        response = self.client.get(
            f"/api/sessions/{self.runtime.session_id}/plan"
        )

        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["state"], "preparing_plan")
        self.assertEqual(body["goal_id"], planning.id)
        self.assertEqual(
            body["current_request"],
            "Create a Three.js calculator and run it",
        )
        self.assertEqual(body["tasks"], [])
        self.assertFalse(body["capabilities"]["can_submit_request"])

    def test_semantic_turn_without_goal_is_the_single_saved_request_truth(self):
        self.store.transition_goal(
            self.goal.id,
            GoalStatus.CANCELLED,
            reason="replace fixture goal with pre-goal intake",
        )
        self.runtime.transition_mode("plan")
        session = self.store.get_workflow_session(self.runtime.session_id)
        state = dict(session["state"])
        state["pending_semantic_turn"] = {
            "turn_id": "turn-saved-intake",
            "original_input": "Create a Three.js calculator and run it",
            "request_fingerprint": "saved-request-fingerprint",
            "status": "awaiting_provider",
            "stage": "goal_intake",
            "interaction_mode": "plan",
            "last_error": "Local provider is temporarily unavailable",
            "model_capability_envelope": self.runtime.model_capability_envelope().to_dict(),
        }
        self.store.save_workflow_session(
            self.runtime.session_id,
            goal_id=None,
            session_mode=str(session["session_mode"]),
            plan_state=str(session["plan_state"]),
            run_state="blocked",
            ultra_profile=str(session["ultra_profile"]),
            sleep_state=str(session["sleep_state"]),
            state=state,
        )

        context = self.client.get(
            f"/api/sessions/{self.runtime.session_id}/workspace"
        ).json()
        plan = self.client.get(
            f"/api/sessions/{self.runtime.session_id}/plan"
        ).json()

        self.assertEqual(context["required_view"], "plan")
        self.assertEqual(context["goal"]["id"], "turn-saved-intake")
        self.assertEqual(
            context["goal"]["objective"],
            "Create a Three.js calculator and run it",
        )
        self.assertEqual(context["attention"]["state"], "blocked")
        self.assertFalse(context["capabilities"]["can_submit_plan_request"])
        self.assertEqual(plan["state"], "preparing_plan")
        self.assertEqual(plan["goal_id"], "turn-saved-intake")
        self.assertEqual(
            plan["current_request"],
            "Create a Three.js calculator and run it",
        )
        self.assertFalse(plan["capabilities"]["can_submit_request"])

    def test_thread_coalesces_preplan_recovery_and_hides_internal_contract_name(self):
        self.store.transition_goal(
            self.goal.id,
            GoalStatus.CANCELLED,
            reason="replace fixture goal with a route contract boundary",
        )
        self.runtime.transition_mode("plan")
        session = self.store.get_workflow_session(self.runtime.session_id)
        state = dict(session["state"])
        state["pending_semantic_turn"] = {
            "turn_id": "turn-contract-boundary",
            "original_input": "Create a Three.js calculator and run it",
            "request_fingerprint": "contract-boundary-fingerprint",
            "status": "awaiting_provider",
            "stage": "route",
            "interaction_mode": "plan",
            "failure_kind": "contract",
            "last_error": "submit_semantic_route must be called exactly once",
            "model_capability_envelope": self.runtime.model_capability_envelope().to_dict(),
        }
        self.store.save_workflow_session(
            self.runtime.session_id,
            goal_id=None,
            session_mode=str(session["session_mode"]),
            plan_state=str(session["plan_state"]),
            run_state="blocked",
            ultra_profile=str(session["ultra_profile"]),
            sleep_state=str(session["sleep_state"]),
            state=state,
        )

        context = self.client.get(
            f"/api/sessions/{self.runtime.session_id}/workspace"
        ).json()
        thread = self.client.get(
            f"/api/sessions/{self.runtime.session_id}/thread?after_sequence=0"
        ).json()
        items = thread["items"]
        by_type = {item["type"]: item for item in items}

        self.assertNotIn("plan", {item["type"] for item in items})
        self.assertIn("recovery", by_type)
        self.assertIn("workflow_status", by_type)
        self.assertNotIn("submit_semantic_route", str(context))
        self.assertNotIn("submit_semantic_route", str(by_type["recovery"]))
        self.assertIn("targeted retry", by_type["recovery"]["payload"]["error"])
        self.assertIn("original request is preserved", by_type["workflow_status"]["payload"]["reason"])

    def test_durable_pending_approval_survives_noisy_event_history(self):
        fingerprint = "pending-" + ("a" * 64)
        self.store.update_goal_metadata(
            self.goal.id,
            pending_tool_approval={
                "tool": "run_command",
                "arguments": {"command": "npm install three"},
                "risk": "critical",
                "action_fingerprint": fingerprint,
            },
        )
        for index in range(100):
            self.store.append_event(
                "worker.heartbeat",
                goal_id=self.goal.id,
                entity_type="worker",
                entity_id="coordinator",
                payload={"sequence": index},
            )

        context = self.client.get(
            f"/api/sessions/{self.runtime.session_id}/workspace"
        ).json()

        self.assertEqual(context["tool_approval"]["action_fingerprint"], fingerprint)
        self.assertEqual(context["tool_approval"]["arguments"]["command"], "npm install three")
        self.assertEqual(context["required_action"]["kind"], "allow_tool")

    def test_durable_pending_approval_projects_waiting_runtime_after_restart(self):
        self.runtime.approve_plan(1)
        fingerprint = "stale-runtime-" + ("p" * 64)
        self.store.update_goal_metadata(
            self.goal.id,
            pending_tool_approval={
                "tool": "preview_html",
                "arguments": {"path": "public/index.html"},
                "risk": "high",
                "action_fingerprint": fingerprint,
            },
        )
        for index in range(80):
            self.store.append_event(
                "worker.heartbeat",
                goal_id=self.goal.id,
                entity_type="worker",
                entity_id="coordinator",
                payload={"sequence": index},
            )

        context = self.client.get(
            f"/api/sessions/{self.runtime.session_id}/workspace"
        ).json()

        self.assertEqual(context["runtime"]["phase"], "waiting_for_approval")
        self.assertEqual(context["runtime"]["waiting_on"], "user")
        self.assertEqual(context["runtime"]["liveness"], "waiting")
        self.assertIn("preview_html", context["runtime"]["reason"])

    def test_full_auto_requires_confirmation_and_resumes_pending_tool(self):
        fingerprint = "pending-" + ("b" * 64)
        self.store.update_goal_metadata(
            self.goal.id,
            pending_tool_approval={
                "tool": "run_command",
                "arguments": {"command": "npm install three"},
                "risk": "critical",
                "action_fingerprint": fingerprint,
            },
        )
        rejected = self.client.post(
            f"/api/sessions/{self.runtime.session_id}/actions",
            headers=self.csrf_headers(),
            json={"action": "sleep_full_on", "value": "yes", "source": "web"},
        )
        self.assertEqual(rejected.status_code, 422)

        enabled = self.client.post(
            f"/api/sessions/{self.runtime.session_id}/actions",
            headers=self.csrf_headers(),
            json={"action": "sleep_full_on", "value": "FULL AUTO", "source": "web"},
        )

        self.assertEqual(enabled.status_code, 200, enabled.text)
        self.assertIn("pending action was approved", enabled.json()["message"])
        context = self.client.get(
            f"/api/sessions/{self.runtime.session_id}/workspace"
        ).json()
        self.assertTrue(context["sleep_enabled"])
        self.assertEqual(context["sleep_policy"], "full")
        self.assertIsNone(context["tool_approval"])
        self.assertEqual(self.store.get_goal(self.goal.id).status, GoalStatus.RUNNING)
        self.assertTrue(
            any(
                item.event_type == "sleep.full_auto_plan_approval"
                for item in self.store.list_recent_events(self.goal.id, limit=100)
            )
        )

    def test_full_auto_tool_boundary_wakes_the_controller_after_restart(self):
        self.runtime.approve_plan(1)
        fingerprint = "paused-tool-" + ("d" * 64)
        self.store.update_goal_metadata(
            self.goal.id,
            pending_tool_approval={
                "tool": "run_command",
                "arguments": {"command": "python -c pass"},
                "risk": "critical",
                "action_fingerprint": fingerprint,
                "policy_group": "dangerous_command",
            },
        )
        self.store.transition_goal(self.goal.id, GoalStatus.PAUSED, reason="tool boundary")
        wake = mock.Mock()
        adapter = CoreWebAdapter(self.runtime, on_execution_requested=wake)

        receipt = adapter.apply_workspace_action(
            WorkspaceActionRequest(
                action="sleep_full_on",
                value="FULL AUTO",
                source="web",
            )
        )

        self.assertTrue(receipt["accepted"])
        wake.assert_called_once_with()
        self.assertEqual(self.store.get_goal(self.goal.id).status, GoalStatus.RUNNING)
        self.assertEqual(
            self.store.get_goal(self.goal.id).metadata["pending_tool_approval"]["decision"],
            "allow_once",
        )

    def test_error_taxonomy_distinguishes_quota_from_rate_limit(self):
        self.assertEqual(_error_code("Provider quota exhausted", 429), "quota_exhausted")
        self.assertEqual(_error_code("Too many requests", 429), "rate_limited")
        self.assertEqual(_error_code("Network connection timed out", 500), "runtime_unreachable")

    def test_model_catalog_is_on_demand_and_secret_free(self):
        descriptor = mock.Mock()
        descriptor.id = "ollama:coder@http://localhost:11434"
        descriptor.display_name = "Coder (ollama)"
        descriptor.to_dict.return_value = {
            "id": descriptor.id,
            "provider": "ollama",
            "model": "coder",
            "execution_class": "local",
            "host": "http://localhost:11434",
            "capabilities": ["tools"],
            "label": None,
            "source": "ollama",
            "metadata": {"context_window_tokens": 32768},
        }
        catalog = mock.Mock()
        catalog.discover.return_value = (descriptor,)
        catalog.diagnostics = ()
        with mock.patch("agent.web_views.service.ModelCatalog", return_value=catalog):
            response = self.client.get(
                f"/api/sessions/{self.runtime.session_id}/models"
            )
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["models"][0]["model"], "coder")
        self.assertTrue(body["safe_checkpoint"])
        self.assertNotIn("api_key", response.text.casefold())

    def test_switch_model_action_uses_runtime_checkpoint_authority(self):
        descriptor = mock.Mock()
        descriptor.id = "ollama:coder@http://localhost:11434"
        descriptor.provider = "ollama"
        descriptor.model = "coder"
        provider = ScriptedProvider([])
        descriptor.create_provider.return_value = provider
        catalog = mock.Mock()
        catalog.by_id.return_value = descriptor
        with (
            mock.patch("agent.web_views.service.ModelCatalog", return_value=catalog),
            mock.patch.object(self.runtime, "replace_provider") as replace_provider,
        ):
            response = self.client.post(
                f"/api/sessions/{self.runtime.session_id}/actions",
                headers=self.csrf_headers(),
                json={
                    "action": "switch_model",
                    "target_id": descriptor.id,
                    "value": descriptor.id,
                    "expected_sequence": self.store.latest_event_sequence(),
                    "source": "web",
                },
            )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["message"], "Model changed to ollama/coder.")
        replace_provider.assert_called_once_with(provider, descriptor)

    def test_continue_local_model_selects_the_strongest_local_candidate(self):
        weak = mock.Mock()
        weak.id = "ollama:weak"
        weak.provider = "ollama"
        weak.model = "weak"
        weak.execution_class = ExecutionClass.LOCAL
        weak.supports_tools = True
        weak_provider = ScriptedProvider([], model="weak")
        weak.create_provider.return_value = weak_provider
        strong = mock.Mock()
        strong.id = "ollama:strong"
        strong.provider = "ollama"
        strong.model = "strong"
        strong.execution_class = ExecutionClass.LOCAL
        strong.supports_tools = True
        strong_provider = ScriptedProvider([], model="strong")
        strong.create_provider.return_value = strong_provider
        cloud = mock.Mock()
        cloud.execution_class = ExecutionClass.CLOUD
        cloud.supports_tools = True
        catalog = mock.Mock()
        catalog.discover.return_value = (weak, cloud, strong)
        catalog.diagnostics = ()
        weak_envelope = mock.Mock(
            level=2, context_window_tokens=32_768, maximum_output_tokens=4_096,
            structured_output=True, thinking=False, parameter_count_billions=7,
        )
        strong_envelope = mock.Mock(
            level=3, context_window_tokens=65_536, maximum_output_tokens=8_192,
            structured_output=True, thinking=True, parameter_count_billions=32,
        )
        continuation = {
            "abstraction_level": "bounded",
            "max_cohesive_components_per_packet": 4,
        }
        with (
            mock.patch("agent.web_views.service.ModelCatalog", return_value=catalog),
            mock.patch.object(
                self.runtime,
                "_capability_envelope_for",
                side_effect=[weak_envelope, strong_envelope],
            ),
            mock.patch.object(
                self.runtime,
                "continue_with_local_model",
                return_value=continuation,
            ) as continue_local,
        ):
            response = self.client.post(
                f"/api/sessions/{self.runtime.session_id}/actions",
                headers=self.csrf_headers(),
                json={
                    "action": "continue_local_model",
                    "action_fingerprint": "continue-local-test",
                    "source": "web",
                },
            )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIn("Continuing with ollama/strong", response.json()["message"])
        self.assertIn("quality gates are unchanged", response.json()["message"])
        continue_local.assert_called_once_with(strong_provider, strong)

    def test_continue_local_model_resumes_a_pre_goal_semantic_boundary(self):
        self.store.transition_goal(
            self.goal.id,
            GoalStatus.CANCELLED,
            reason="replace fixture goal with pre-goal intake",
        )
        self.runtime.transition_mode("plan")
        session = self.store.get_workflow_session(self.runtime.session_id)
        state = dict(session["state"])
        state["pending_semantic_turn"] = {
            "turn_id": "turn-pre-goal-action",
            "original_input": "Build the saved project",
            "status": "awaiting_provider",
            "stage": "goal_intake",
            "last_error": "cloud quota exhausted",
        }
        self.store.save_workflow_session(
            self.runtime.session_id,
            goal_id=None,
            session_mode=str(session["session_mode"]),
            plan_state=str(session["plan_state"]),
            run_state="blocked",
            ultra_profile=str(session["ultra_profile"]),
            sleep_state=str(session["sleep_state"]),
            state=state,
        )
        descriptor = mock.Mock()
        descriptor.id = "ollama:strong-pre-goal"
        descriptor.provider = "ollama"
        descriptor.model = "strong-pre-goal"
        descriptor.execution_class = ExecutionClass.LOCAL
        descriptor.supports_tools = True
        provider = ScriptedProvider([], model="strong-pre-goal")
        descriptor.create_provider.return_value = provider
        catalog = mock.Mock()
        catalog.discover.return_value = (descriptor,)
        catalog.diagnostics = ()
        envelope = mock.Mock(
            level=3,
            context_window_tokens=65_536,
            maximum_output_tokens=8_192,
            structured_output=True,
            thinking=True,
            parameter_count_billions=32,
        )
        continuation = {
            "abstraction_level": "bounded",
            "max_cohesive_components_per_packet": 4,
        }

        with (
            mock.patch("agent.web_views.service.ModelCatalog", return_value=catalog),
            mock.patch.object(
                self.runtime, "_capability_envelope_for", return_value=envelope
            ),
            mock.patch.object(
                self.runtime,
                "continue_with_local_model",
                return_value=continuation,
            ) as continue_local,
            mock.patch.object(self.runtime, "resume") as resume,
        ):
            response = self.client.post(
                f"/api/sessions/{self.runtime.session_id}/actions",
                headers=self.csrf_headers(),
                json={
                    "action": "continue_local_model",
                    "target_id": "turn-pre-goal-action",
                    "action_fingerprint": "continue-local-pre-goal",
                    "source": "web",
                },
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertIn("Workflow resumed", response.json()["message"])
        continue_local.assert_called_once_with(provider, descriptor)
        resume.assert_called_once_with()

    def test_tool_approval_is_exposed_and_resolved_by_fingerprint(self):
        fingerprint = "a" * 64
        self.store.append_event(
            "approval.requested",
            goal_id=self.goal.id,
            entity_type="tool",
            entity_id="run_command",
            payload={
                "tool": "run_command",
                "risk": "risky",
                "action_fingerprint": fingerprint,
            },
        )
        context = self.client.get(f"/api/sessions/{self.runtime.session_id}/workspace").json()
        self.assertEqual(context["tool_approval"]["tool"], "run_command")
        self.assertEqual(context["tool_approval"]["action_fingerprint"], fingerprint)
        with mock.patch.object(self.runtime, "resolve_tool_approval", return_value=True) as resolver:
            response = self.client.post(
                f"/api/sessions/{self.runtime.session_id}/tool-approval",
                headers=self.csrf_headers(),
                json={"action_fingerprint": fingerprint, "decision": "allow_once"},
            )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["decision"], "allow")
        resolver.assert_called_once_with(fingerprint, "allow_once")

        with mock.patch.object(self.runtime, "resolve_tool_approval", return_value=True) as resolver:
            response = self.client.post(
                f"/api/sessions/{self.runtime.session_id}/tool-approval",
                headers=self.csrf_headers(),
                json={"action_fingerprint": fingerprint, "decision": "allow_session"},
            )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["decision"], "allow_session")
        resolver.assert_called_once_with(fingerprint, "allow_session")

    def test_workspace_action_can_allow_matching_tools_for_the_session(self):
        fingerprint = "s" * 64
        with mock.patch.object(self.runtime, "resolve_tool_approval", return_value=True) as resolver:
            response = self.client.post(
                f"/api/sessions/{self.runtime.session_id}/actions",
                headers=self.csrf_headers(),
                json={
                    "action": "allow_tool_session",
                    "target_id": "run_command",
                    "action_fingerprint": fingerprint,
                    "expected_sequence": self.store.latest_event_sequence(),
                    "source": "web",
                },
            )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIn("Allowed for this session", response.json()["message"])
        resolver.assert_called_once_with(fingerprint, "allow_session")

    def test_workspace_hides_a_pending_approval_that_was_already_accepted(self):
        fingerprint = "accepted-" + ("z" * 64)
        self.store.update_goal_metadata(
            self.goal.id,
            pending_tool_approval={
                "tool": "preview_html",
                "arguments": {"path": "index.html"},
                "risk": "high",
                "action_fingerprint": fingerprint,
            },
        )
        self.store.append_event(
            "workspace.action.accepted",
            goal_id=self.goal.id,
            entity_type="workspace",
            entity_id="preview_html",
            payload={
                "action": "allow_tool_session",
                "action_fingerprint": fingerprint,
                "source": "web",
                "actor": "user",
            },
        )

        context = self.client.get(
            f"/api/sessions/{self.runtime.session_id}/workspace"
        ).json()

        self.assertIsNone(context["tool_approval"])
        self.assertNotEqual(context["attention"]["eyebrow"], "Approval required")

    def test_new_identical_approval_is_not_hidden_by_an_older_acceptance(self):
        fingerprint = "repeat-" + ("r" * 64)
        self.store.append_event(
            "workspace.action.accepted",
            goal_id=self.goal.id,
            entity_type="workspace",
            entity_id="preview_html",
            payload={
                "action": "allow_tool",
                "action_fingerprint": fingerprint,
                "source": "web",
            },
        )
        request_event = self.store.append_event(
            "approval.requested",
            goal_id=self.goal.id,
            entity_type="tool",
            entity_id="preview_html",
            payload={"tool": "preview_html", "action_fingerprint": fingerprint},
        )
        self.store.update_goal_metadata(
            self.goal.id,
            pending_tool_approval={
                "tool": "preview_html",
                "arguments": {"path": "index.html"},
                "risk": "high",
                "action_fingerprint": fingerprint,
                "requested_sequence": request_event.sequence,
            },
        )

        context = self.client.get(
            f"/api/sessions/{self.runtime.session_id}/workspace"
        ).json()

        self.assertEqual(context["tool_approval"]["action_fingerprint"], fingerprint)

    def test_plan_revision_preserves_requirement_anchor_coverage(self):
        self.store.update_goal_metadata(
            self.goal.id,
            semantic_goal={
                "requirement_anchors": [
                    {
                        "id": "R001",
                        "verbatim_span": "local artifact workspace",
                        "interpreted_requirement": "Deliver the requested local workspace.",
                        "observable_implications": ["The workspace is locally runnable."],
                        "kind": "deliverable",
                    }
                ]
            },
        )
        payload = self.plan_payload()
        payload["tasks"][0]["requirement_refs"] = ["R001"]

        response = self.client.post(
            f"/api/sessions/{self.runtime.session_id}/plan/revision",
            headers=self.csrf_headers(),
            json=payload,
        )

        self.assertEqual(response.status_code, 200, response.text)
        task = self.store.get_latest_plan(self.goal.id).tasks[0]
        self.assertEqual(task.metadata["requirement_refs"], ["R001"])

    def test_plan_revision_cannot_drop_requirement_anchor(self):
        self.store.update_goal_metadata(
            self.goal.id,
            semantic_goal={
                "requirement_anchors": [
                    {
                        "id": "R001",
                        "verbatim_span": "local artifact workspace",
                        "interpreted_requirement": "Deliver the requested local workspace.",
                        "observable_implications": ["The workspace is locally runnable."],
                        "kind": "deliverable",
                    }
                ]
            },
        )

        response = self.client.post(
            f"/api/sessions/{self.runtime.session_id}/plan/revision",
            headers=self.csrf_headers(),
            json=self.plan_payload(),
        )

        self.assertEqual(response.status_code, 422, response.text)
        self.assertIn("drop user requirement anchors", response.text)

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
        self.assertEqual(context.json()["attention"]["action"]["label"], "Open plan")
        self.runtime.approve_plan(1)
        running = self.client.get(
            f"/api/sessions/{self.runtime.session_id}/workspace"
        ).json()
        self.assertIsNone(running["required_view"])
        self.assertEqual(running["attention"]["state"], "working")
        self.assertIn("activity_sequence", running["runtime"])
        self.assertIn("liveness", running["runtime"])
        self.assertIn("timeline_preview", running["runtime"])

    def test_unified_history_execution_aliases_and_required_action(self):
        context = self.client.get(
            f"/api/sessions/{self.runtime.session_id}/workspace"
        ).json()
        self.assertEqual(context["control_surface"], "web")
        self.assertEqual(context["required_action"]["kind"], "approve_plan")
        self.assertEqual(context["required_action"]["target_view"], "plan")

        history = self.client.get(
            f"/api/sessions/{self.runtime.session_id}/history"
        )
        self.assertEqual(history.status_code, 200)
        self.assertTrue(history.json()["items"])
        self.assertIn("goals", history.json())

        execution = self.client.get(
            f"/api/sessions/{self.runtime.session_id}/execution"
        )
        self.assertEqual(execution.status_code, 200)
        self.assertTrue(execution.json()["read_only"])
        self.assertIn("tree", execution.json())

        tree = self.client.get(
            f"/api/sessions/{self.runtime.session_id}/tree"
        )
        self.assertEqual(tree.status_code, 200)

        revisions = self.client.get(
            f"/api/sessions/{self.runtime.session_id}/plan/revisions"
        )
        self.assertEqual(revisions.status_code, 200)
        self.assertEqual(revisions.json()["current_revision"], 1)

    def test_unified_workspace_action_approval_is_idempotent(self):
        context = self.client.get(
            f"/api/sessions/{self.runtime.session_id}/workspace"
        ).json()
        fingerprint = "plan-action-" + ("a" * 32)
        payload = {
            "action": "approve_plan",
            "target_id": "1",
            "action_fingerprint": fingerprint,
            "expected_sequence": context["history_cursor"],
            "source": "web",
            "value": "1",
        }
        first = self.client.post(
            f"/api/sessions/{self.runtime.session_id}/actions",
            headers=self.csrf_headers(),
            json=payload,
        )
        self.assertEqual(first.status_code, 200)
        self.assertTrue(first.json()["accepted"])
        self.assertFalse(first.json()["duplicate"])

        second = self.client.post(
            f"/api/sessions/{self.runtime.session_id}/actions",
            headers=self.csrf_headers(),
            json=payload,
        )
        self.assertEqual(second.status_code, 200)
        self.assertTrue(second.json()["duplicate"])
        self.assertEqual(self.store.get_latest_plan(self.goal.id).status.value, "accepted")

    def test_terminal_fallback_is_rejected_while_web_is_connected(self):
        self.runtime.web_control_connected = True
        response = self.client.post(
            f"/api/sessions/{self.runtime.session_id}/actions",
            headers=self.csrf_headers(),
            json={
                "action": "retry",
                "source": "terminal_fallback",
            },
        )
        self.assertEqual(response.status_code, 422)
        self.assertIn("Web workspace is disconnected", response.json()["error"])

    def test_history_and_revision_queries_are_session_isolated(self):
        foreign = "goal-from-another-session"
        history = self.client.get(
            f"/api/sessions/{self.runtime.session_id}/history?goal_id={foreign}"
        )
        revisions = self.client.get(
            f"/api/sessions/{self.runtime.session_id}/plan/revisions?goal_id={foreign}"
        )
        self.assertEqual(history.status_code, 404)
        self.assertEqual(revisions.status_code, 404)

    def test_workspace_projects_a_pending_question_as_one_answer_action(self):
        self.store.update_goal_metadata(
            self.goal.id,
            plan_questions=[{
                "id": "q1",
                "question": "Which preview should be used?",
                "options": [{"label": "Browser", "value": "browser"}],
                "allow_freeform": False,
            }],
            plan_answers={},
        )
        self.store.transition_goal(self.goal.id, GoalStatus.PAUSED, reason="waiting for answer")
        context = self.client.get(
            f"/api/sessions/{self.runtime.session_id}/workspace"
        ).json()
        self.assertEqual(context["required_action"]["kind"], "answer")
        self.assertEqual(context["required_action"]["question"]["id"], "q1")

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

    def test_queue_rejects_duplicate_active_request_and_duplicate_follow_up(self):
        self.runtime.approve_plan(1)
        duplicate_active = self.client.post(
            f"/api/sessions/{self.runtime.session_id}/queue",
            headers=self.csrf_headers(),
            json={"text": "  BUILD   THE LOCAL ARTIFACT WORKSPACE  "},
        )
        self.assertEqual(duplicate_active.status_code, 200, duplicate_active.text)
        self.assertTrue(duplicate_active.json()["duplicate"])
        self.assertEqual(duplicate_active.json()["duplicate_of"], "active_request")
        self.assertEqual(self.store.list_queued_prompts(self.runtime.session_id), ())

        first = self.client.post(
            f"/api/sessions/{self.runtime.session_id}/queue",
            headers=self.csrf_headers(),
            json={"text": "Add keyboard navigation"},
        )
        repeated = self.client.post(
            f"/api/sessions/{self.runtime.session_id}/queue",
            headers=self.csrf_headers(),
            json={"text": "add   keyboard navigation"},
        )
        self.assertTrue(first.json()["queued"])
        self.assertTrue(repeated.json()["duplicate"])
        self.assertEqual(repeated.json()["duplicate_of"], "queued_prompt")
        self.assertEqual(len(self.store.list_queued_prompts(self.runtime.session_id)), 1)

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
        self.assertGreater(serialized["sequence"], 0)

    def test_sse_begins_with_a_runtime_snapshot_and_safe_activity_sequence(self):
        self.events.publish(
            "provider.activity",
            "Provider request sent",
            source_kind="MODEL",
            phase="planning",
            state="started",
            provider_state="request_sent",
            received_bytes=0,
            received_chunks=0,
        )
        route = next(
            item for item in self.app.routes
            if getattr(item, "path", "") == "/api/sessions/{session_id}/events"
        )
        request = Request(
            {
                "type": "http",
                "http_version": "1.1",
                "method": "GET",
                "scheme": "http",
                "path": f"/api/sessions/{self.runtime.session_id}/events",
                "raw_path": b"/events",
                "query_string": b"",
                "headers": [(b"last-event-id", b"0")],
                "client": ("127.0.0.1", 12345),
                "server": ("127.0.0.1", 43210),
            }
        )

        async def first_messages():
            response = await route.endpoint(request, self.runtime.session_id)
            iterator = response.body_iterator
            try:
                return await anext(iterator), await anext(iterator)
            finally:
                await iterator.aclose()

        retry, snapshot = asyncio.run(first_messages())
        self.assertEqual(retry, "retry: 1000\n")
        self.assertIn("event: snapshot", snapshot)
        self.assertIn('"activity_sequence"', snapshot)
        self.assertNotIn("partial_tool_args", snapshot)

    def test_workspace_assets_use_sse_and_incremental_live_regions(self):
        html = self.client.get("/assets/index.html").text
        script = self.client.get("/assets/app.js").text
        self.assertIn('id="liveWorkflow"', html)
        self.assertIn('id="liveTimeline"', html)
        self.assertIn("new EventSource", script)
        self.assertIn("applyLiveActivity", script)

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

    def test_execution_result_exposes_safe_artifacts_and_changed_files(self):
        self.store.add_artifact(
            Artifact(
                ultra_run_id=self.run.id,
                work_node_id=self.node.id,
                agent_run_id=self.agent.id,
                kind="preview",
                uri="http://127.0.0.1:4173/result?token=secret#panel",
                path="output/result.html",
                content_hash="verified-hash",
                evidence={"verified": True},
            )
        )
        self.store.add_artifact(
            Artifact(
                ultra_run_id=self.run.id,
                kind="external",
                uri="https://example.com/private?token=secret",
            )
        )

        response = self.client.get(
            f"/api/sessions/{self.runtime.session_id}/execution"
        )

        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()["result"]
        self.assertEqual(result["changed_files"], ["agent/auth.py"])
        preview = next(item for item in result["artifacts"] if item["kind"] == "preview")
        external = next(item for item in result["artifacts"] if item["kind"] == "external")
        self.assertEqual(preview["preview_url"], "http://127.0.0.1:4173/result")
        self.assertTrue(preview["verified"])
        self.assertEqual(external["preview_url"], "")
        self.assertNotIn("secret", response.text)

    def test_staged_completed_result_exposes_evidence_without_ultra_run(self):
        self.store.update_ultra_run(self.run.id, status=UltraRunStatus.COMPLETED)
        self.store.transition_goal(
            self.goal.id,
            GoalStatus.RUNNING,
            reason="run the staged coordinator",
        )
        self.store.update_goal_metadata(
            self.goal.id,
            completion_summary="Implementation and verification complete.",
            goal_change_sets=[{"changed_files": ["artifact.txt"]}],
        )
        self.store.add_evidence(
            Evidence(
                goal_id=self.goal.id,
                plan_revision=1,
                task_id="T001",
                kind="tool_result",
                summary="artifact.txt was written and read back",
                data={
                    "path": "artifact.txt",
                    "file_hash": "verified-hash",
                    "file_exists": True,
                },
                created_by="harness",
                verified=True,
            )
        )
        self.store.transition_goal(
            self.goal.id,
            GoalStatus.VERIFYING,
            reason="evidence recorded",
        )
        self.store.transition_goal(
            self.goal.id,
            GoalStatus.REVIEWING,
            reason="independent review started",
        )
        self.store.transition_goal(
            self.goal.id,
            GoalStatus.COMPLETED,
            reason="independent review passed",
        )

        response = self.client.get(
            f"/api/sessions/{self.runtime.session_id}/execution"
        )

        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()["result"]
        self.assertEqual(result["summary"], "Implementation and verification complete.")
        self.assertEqual(result["changed_files"], ["artifact.txt"])
        self.assertEqual(result["artifacts"][0]["path"], "artifact.txt")
        self.assertTrue(result["artifacts"][0]["verified"])


if __name__ == "__main__":
    unittest.main()
