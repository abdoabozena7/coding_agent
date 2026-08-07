from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import time
import unittest

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright

from agent.config import RuntimeConfig
from agent.models import GoalStatus
from agent.quality import ChangeSetStatus, ChangeSetV1
from agent.runtime import AgentRuntime
from agent.store import StateStore
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
from agent.web_views.server import LocalWebServer
from tests.test_local_web_views import basis, task_value


class LocalWebViewsBrowserTests(unittest.TestCase):
    """Exercise the unified workspace and its protected interactions."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.playwright = sync_playwright().start()
        try:
            cls.browser = cls.playwright.chromium.launch(headless=True)
        except PlaywrightError as exc:
            cls.playwright.stop()
            raise unittest.SkipTest(f"Playwright Chromium is unavailable: {exc}") from exc

    @classmethod
    def tearDownClass(cls) -> None:
        cls.browser.close()
        cls.playwright.stop()

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temporary.name)
        self.store = StateStore(self.workspace)
        self.runtime = AgentRuntime(
            ScriptedProvider([]),
            self.store,
            self.workspace,
            config=replace(RuntimeConfig(), repository_index_warmup_files=0),
            session_id="browser-session",
        )
        self.goal = self.store.create_goal(
            "Exercise Local Web Views",
            session_id=self.runtime.session_id,
        )
        self.plan = self.store.create_plan(
            self.goal.id,
            "Exercise the browser workspace.",
            [task_value()],
            **basis(),
        )
        self.store.transition_goal(
            self.goal.id,
            GoalStatus.AWAITING_PLAN_APPROVAL,
            reason="browser plan ready",
        )
        self.server = LocalWebServer(self.runtime).start()
        self.runtime.local_web_server = self.server
        self.context = self.browser.new_context(viewport={"width": 1440, "height": 1000})
        self.console_errors: list[str] = []
        self.page_errors: list[str] = []
        self.artifacts = Path("output/playwright")
        self.artifacts.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        self.context.close()
        self.runtime.close()
        self.store.close()
        self.temporary.cleanup()

    def tracked_page(self):
        page = self.context.new_page()
        page.on(
            "console",
            lambda message: self.console_errors.append(message.text)
            if message.type == "error" else None,
        )
        page.on("pageerror", lambda error: self.page_errors.append(str(error)))
        return page

    def create_review_fixture(self) -> tuple[WorkNode, AgentRun, ChangeSetV1]:
        self.runtime.approve_plan(1)
        run = self.store.create_ultra_run(
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
        node = self.store.create_work_node(
            WorkNode(
                ultra_run_id=run.id,
                title="Implement browser flow",
                objective="Implement and verify the browser flow.",
                contract=TaskContractV1(
                    objective="Implement and verify the browser flow.",
                    success_criteria=("The browser flow works.",),
                    write_paths=("agent/example.py",),
                ),
                status=WorkNodeStatus.REVIEWING,
                assigned_role="frontend",
            )
        )
        agent = self.store.create_agent_run(
            AgentRun(
                ultra_run_id=run.id,
                work_node_id=node.id,
                role="frontend",
                provider="test",
                model="offline",
                phase="implementation",
                status=AgentRunStatus.RUNNING,
                usage={"progress": 62, "tools": ["filesystem", "test_runner"]},
            )
        )
        checkpoint = self.store.save_change_set(
            ChangeSetV1(
                ultra_run_id=run.id,
                responsible_agent_id=agent.id,
                parent_id=node.id,
                status=ChangeSetStatus.REVIEWING,
                changed_files=("agent/example.py",),
                diff=(
                    "diff --git a/agent/example.py b/agent/example.py\n"
                    "--- a/agent/example.py\n"
                    "+++ b/agent/example.py\n"
                    "@@ -1,2 +1,3 @@\n"
                    " def value():\n"
                    "-    return 1\n"
                    "+    checked = True\n"
                    "+    return 2\n"
                ),
                metadata={"reason": "Implement the approved browser flow."},
            )
        )
        self.store.add_artifact(
            Artifact(
                ultra_run_id=run.id,
                work_node_id=node.id,
                agent_run_id=agent.id,
                kind="preview",
                uri="http://127.0.0.1:4173/result?token=hidden",
                path="output/result.html",
                content_hash="browser-verified",
                evidence={"verified": True},
            )
        )
        return node, agent, checkpoint

    def test_primary_navigation_uses_codex_workflow_labels(self):
        """The shell exposes the conversation flow, not the old internal phases."""
        page = self.tracked_page()
        page.goto(self.server.url_for("thread"))
        page.locator("#workspaceNav").wait_for()

        visible_labels = page.locator(
            "#workspaceNav .nav-button:visible > span:not(.nav-icon):not(.nav-badge)"
        ).all_text_contents()
        self.assertEqual(
            [label.strip() for label in visible_labels],
            ["Thread", "Plan", "Result", "History"],
        )
        self.assertEqual(page.locator("#workspaceNav [data-view='review']").count(), 0)
        self.assertEqual(page.locator("#workspaceNav [data-view='execution']").count(), 0)

        page.locator("#workspaceNav button[data-view='result']").click()
        page.get_by_role("heading", name="Result", exact=True).wait_for()
        self.assertEqual(page.locator("#workspaceNav [data-view='result'].selected").count(), 1)
        self.assertEqual(self.console_errors, [])
        self.assertEqual(self.page_errors, [])

    def test_plan_is_simple_read_first_and_approval_is_explicit(self):
        page = self.tracked_page()
        page.goto(self.server.url_for("plan"))
        page.locator("#liveNowTitle").wait_for()
        self.assertEqual(page.locator("#liveNowTitle").text_content(), "Plan ready to start")
        page.get_by_role("heading", name="Plan", exact=True).wait_for()
        page.locator("#attentionTitle").wait_for()
        self.assertEqual(page.get_by_text("No workflow is active.", exact=True).count(), 0)
        self.assertIn("plan r1", page.locator("#workflowIdentity").text_content())
        self.assertFalse(
            page.get_by_role("button", name="Open plan", exact=True).is_visible()
        )
        self.assertEqual(page.get_by_text("Revision 1 · 1 tasks", exact=True).count(), 1)
        page.get_by_role("heading", name="How the system will execute this plan").wait_for()
        self.assertEqual(page.get_by_text("Staged coordinator", exact=True).count(), 1)
        self.assertEqual(
            page.get_by_text("Specialist agents are not created for this plan.").count(),
            1,
        )
        self.assertEqual(page.locator("select").count(), 0)
        self.assertEqual(page.locator("text=completed").count(), 0)

        page.get_by_role("button", name="Advanced", exact=True).click()
        page.locator(".task-row").first.get_by_role("button", name="Edit", exact=True).click()
        page.get_by_label("Task ID").wait_for()
        description = page.get_by_label("What this task must do")
        description.fill("A longer explanation that wraps naturally without an inner scroll trap.")
        self.assertEqual(
            description.evaluate("element => getComputedStyle(element).overflowY"),
            "hidden",
        )
        page.get_by_role("button", name="Add task").click()
        page.locator("#task-1-title").fill("Verify the new flow")
        page.locator(".task-row").nth(1).get_by_role("button", name="Up").click()
        self.assertEqual(
            page.locator(".task-row").first.locator("h4").text_content(),
            "Verify the new flow",
        )
        page.get_by_role("button", name="Simple", exact=True).click()
        self.assertFalse(page.get_by_label("Task ID").is_visible())

        page.locator("[data-action='save-revision']").click()
        page.locator("#toastRegion").get_by_text(
            "Revision 2 saved. Nothing is running yet."
        ).wait_for()
        self.assertEqual(self.store.get_latest_plan(self.goal.id).revision, 2)
        self.assertIsNone(self.store.get_goal(self.goal.id).active_plan_revision)

        page.get_by_role("button", name="Approve & start").click()
        page.locator("#toastRegion").get_by_text(
            "Plan r2 approved. Work is starting."
        ).wait_for()
        self.assertEqual(self.store.get_goal(self.goal.id).active_plan_revision, 2)
        self.assertTrue(self.server.take_execution_request())
        self.assertFalse(self.server.take_execution_request())
        page.get_by_role("heading", name="Work is running").wait_for()

        page.wait_for_timeout(250)
        page.screenshot(path=self.artifacts / "workspace-unified-desktop.png", full_page=True)
        page.set_viewport_size({"width": 768, "height": 900})
        page.screenshot(path=self.artifacts / "workspace-unified-tablet.png", full_page=True)
        page.set_viewport_size({"width": 390, "height": 844})
        page.screenshot(path=self.artifacts / "workspace-unified-mobile.png", full_page=True)
        self.assertEqual(self.console_errors, [])
        self.assertEqual(self.page_errors, [])

    def test_unified_thread_inspector_reconnect_and_responsive_close_are_stable(self):
        page = self.tracked_page()
        page.goto(self.server.url_for("thread"))
        page.get_by_role("heading", name="Unified workflow thread", exact=True).wait_for()

        ids = page.locator("[data-thread-item]").evaluate_all(
            "nodes => nodes.map(node => node.dataset.threadItem)"
        )
        self.assertTrue(ids)
        self.assertEqual(len(ids), len(set(ids)))

        page.locator("#inspectorToggle").click()
        page.locator("#inspector.open").wait_for()
        self.assertEqual(page.locator(".inspector-tab").count(), 5, page.locator("#inspector").inner_text())
        page.locator(".inspector-tab[data-inspector-section='changes']").click()
        page.locator("#inspectorBody").get_by_text("Changes", exact=True).wait_for()
        self.assertTrue(page.locator("body.inspector-visible").count())

        # A mobile resize closes the side surface instead of leaving a fixed
        # panel over the composer and approval controls.
        page.set_viewport_size({"width": 390, "height": 844})
        page.locator("#inspector:not(.open)").wait_for(state="attached")
        self.assertEqual(page.locator("body.inspector-visible").count(), 0)

        page.reload()
        page.get_by_role("heading", name="Unified workflow thread", exact=True).wait_for()
        refreshed_ids = page.locator("[data-thread-item]").evaluate_all(
            "nodes => nodes.map(node => node.dataset.threadItem)"
        )
        self.assertEqual(refreshed_ids, ids)
        self.assertEqual(self.console_errors, [])
        self.assertEqual(self.page_errors, [])

    def test_preplan_workspace_shows_original_request_and_blocks_duplicate_cue(self):
        self.store.transition_goal(
            self.goal.id,
            GoalStatus.CANCELLED,
            reason="replace fixture goal",
        )
        objective = "Create a Three.js calculator with impressive design and run it"
        self.store.create_goal(objective, session_id=self.runtime.session_id)
        page = self.tracked_page()

        page.goto(self.server.url_for("plan"))

        page.get_by_role("heading", name="Your request is already running").wait_for()
        current = page.get_by_test_id("current-request")
        self.assertEqual(current.get_by_text(objective, exact=True).count(), 1)
        page.get_by_role("heading", name="Do not submit this request again").wait_for()
        self.assertIn(
            "Do not repeat the prompt",
            page.get_by_label("Request", exact=True).get_attribute("placeholder"),
        )
        self.assertEqual(page.get_by_text("The workspace could not load").count(), 0)
        self.assertEqual(self.console_errors, [])
        self.assertEqual(self.page_errors, [])

    def test_pre_goal_provider_boundary_never_renders_a_second_prompt_surface(self):
        self.store.transition_goal(
            self.goal.id,
            GoalStatus.CANCELLED,
            reason="replace fixture goal with pre-goal intake",
        )
        self.runtime.transition_mode("plan")
        session = self.store.get_workflow_session(self.runtime.session_id)
        state = dict(session["state"])
        objective = "Create a Three.js calculator with impressive design and run it"
        state["pending_semantic_turn"] = {
            "turn_id": "turn-browser-saved-intake",
            "original_input": objective,
            "request_fingerprint": "browser-saved-request-fingerprint",
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
        page = self.tracked_page()
        page.goto(self.server.url_for("execution"))

        page.get_by_role("heading", name="Your request is saved").wait_for()
        current = page.get_by_test_id("current-request")
        self.assertEqual(current.get_by_text(objective, exact=True).count(), 1)
        page.get_by_role("heading", name="Do not submit this request again").wait_for()
        self.assertIn(
            "Do not repeat the prompt",
            page.get_by_label("Request", exact=True).get_attribute("placeholder"),
        )
        composer = page.get_by_label("Request", exact=True)
        composer.fill("/retry")
        self.assertTrue(
            page.get_by_role("option", name="/retry", exact=False).is_enabled()
        )
        composer.fill("/continue with local model")
        self.assertTrue(
            page.get_by_role(
                "option", name="/continue with local model", exact=False
            ).is_enabled()
        )
        self.assertEqual(
            page.get_by_role("heading", name="What do you want to build or change?").count(),
            0,
        )
        self.assertEqual(self.console_errors, [])
        self.assertEqual(self.page_errors, [])

    def test_thread_provider_contract_boundary_has_one_recovery_item_and_public_copy(self):
        self.store.transition_goal(
            self.goal.id,
            GoalStatus.CANCELLED,
            reason="replace fixture goal with a route contract boundary",
        )
        self.runtime.transition_mode("plan")
        session = self.store.get_workflow_session(self.runtime.session_id)
        state = dict(session["state"])
        state["pending_semantic_turn"] = {
            "turn_id": "turn-browser-contract-boundary",
            "original_input": "Create a Three.js calculator and run it",
            "request_fingerprint": "browser-contract-boundary",
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
        page = self.tracked_page()
        page.goto(self.server.url_for("thread"))

        page.get_by_role("heading", name="Unified workflow thread", exact=True).wait_for()
        page.get_by_role("heading", name="The exact request is preserved", exact=True).wait_for()
        self.assertEqual(page.locator(".thread-plan").count(), 0)
        self.assertEqual(page.get_by_text("submit_semantic_route", exact=False).count(), 0)
        self.assertTrue(
            page.get_by_text("targeted retry", exact=False).first.is_visible()
        )
        self.assertEqual(page.get_by_role("heading", name="Saved request needs recovery", exact=True).count(), 1)
        self.assertEqual(self.console_errors, [])
        self.assertEqual(self.page_errors, [])

    def test_first_prompt_provider_failure_reopens_the_saved_request_without_duplicate_composer(self):
        # Use a genuinely empty session so the first prompt cannot be mistaken
        # for a follow-up to the fixture plan.
        self.context.close()
        self.runtime.close()
        self.store.close()
        self.temporary.cleanup()
        self.temporary = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temporary.name)
        self.store = StateStore(self.workspace)
        self.runtime = AgentRuntime(
            ScriptedProvider([]),
            self.store,
            self.workspace,
            config=replace(RuntimeConfig(), repository_index_warmup_files=0),
            session_id="first-prompt-browser-session",
        )
        self.server = LocalWebServer(self.runtime).start()
        self.runtime.local_web_server = self.server
        self.context = self.browser.new_context(viewport={"width": 1440, "height": 1000})
        self.console_errors = []
        self.page_errors = []
        page = self.tracked_page()
        page.goto(self.server.url_for("plan"))

        composer = page.locator("#globalPrompt")
        objective = "Create a verified calculator and run its tests"
        composer.fill(objective)
        page.get_by_role("button", name="Send request", exact=True).click()

        page.get_by_role("heading", name="Your request is saved", exact=True).wait_for()
        page.get_by_role("heading", name="Do not submit this request again", exact=True).wait_for()
        page.get_by_role("button", name="Retry saved stage", exact=True).wait_for()
        page.get_by_role("button", name="Try strongest available local model", exact=True).wait_for()
        page.get_by_role("button", name="Stop safely", exact=True).wait_for()
        self.assertEqual(composer.input_value(), "")
        self.assertEqual(
            page.get_by_test_id("current-request").get_by_text(objective, exact=True).count(),
            1,
        )
        self.assertEqual(page.get_by_role("heading", name="The workspace could not load").count(), 0)
        self.assertEqual(self.page_errors, [])

    def test_local_runner_boundary_is_named_instead_of_looking_like_generic_work(self):
        self.runtime.approve_plan(1)
        with self.runtime._live_activity_lock:
            self.runtime._live_provider_activity = {
                "state": "network_unavailable",
                "actor": "planner",
                "operation": "Waiting for provider connectivity",
                "last_signal_at": time.time(),
            }
        page = self.tracked_page()

        page.goto(self.server.url_for("execution"))

        page.get_by_role(
            "heading", name="Local model runner is unavailable"
        ).wait_for()
        self.assertIn(
            "saved stage unchanged",
            page.locator("#liveEvidence").text_content(),
        )
        self.assertEqual(self.console_errors, [])
        self.assertEqual(self.page_errors, [])

    def test_full_auto_recovery_card_explains_that_no_action_is_required(self):
        self.runtime.approve_plan(1)
        self.runtime.set_sleep_mode(True, policy="full")
        self.store.update_goal_metadata(
            self.goal.id,
            provider_recovery={
                "state": "waiting",
                "full_auto_retry": True,
                "retry_not_before": time.time() + 60,
                "error": "local model runner unavailable",
            },
        )
        with self.runtime._live_activity_lock:
            self.runtime._live_provider_activity = {
                "state": "network_unavailable",
                "actor": "planner",
                "operation": "Waiting for provider connectivity",
                "last_signal_at": time.time(),
            }
        page = self.tracked_page()
        page.goto(self.server.url_for("execution"))

        note = page.get_by_test_id("auto-recovery-note")
        note.wait_for()
        self.assertIn(
            "no action is required",
            note.text_content().lower(),
        )
        self.assertIn(
            "next retry in about",
            note.text_content().lower(),
        )
        self.assertEqual(self.console_errors, [])
        self.assertEqual(self.page_errors, [])

    def test_project_settings_drawer_is_visible_and_explains_reopen_behavior(self):
        page = self.tracked_page()
        page.goto(self.server.url_for("plan"))

        page.get_by_role("button", name="Project settings", exact=True).click()
        page.get_by_role("heading", name="Project settings", exact=True).wait_for()
        page.locator(".settings-summary").wait_for()
        self.assertIn(
            "reused when this project opens again",
            page.locator("#drawerBody").text_content().lower(),
        )
        self.assertTrue(page.get_by_role("button", name="Change model", exact=True).is_visible())
        self.assertEqual(self.console_errors, [])
        self.assertEqual(self.page_errors, [])

    def test_project_settings_names_an_older_owner_instead_of_showing_raw_404(self):
        page = self.tracked_page()
        page.route(
            "**/api/sessions/*/project-settings",
            lambda route: route.fulfill(
                status=404,
                content_type="application/json",
                body='{"detail":"Not Found"}',
            ),
        )
        page.goto(self.server.url_for("plan"))

        page.get_by_role("button", name="Project settings", exact=True).click()
        page.get_by_role(
            "heading", name="Project settings are unavailable on this owner", exact=True
        ).wait_for()
        self.assertIn(
            "restart the ga3bad owner once",
            page.locator("#drawerBody").text_content().lower(),
        )
        self.assertNotIn("Request failed (404)", page.locator("#drawerBody").text_content())
        self.assertTrue(any("404" in message for message in self.console_errors))
        self.assertEqual(self.page_errors, [])

    def test_quota_boundary_names_the_usage_limit_and_model_recovery(self):
        self.store.update_goal_metadata(
            self.goal.id,
            waiting_question="Provider quota exhausted for this model.",
        )
        self.store.transition_goal(
            self.goal.id,
            GoalStatus.PAUSED,
            reason="provider quota exhausted",
        )
        page = self.tracked_page()

        page.goto(self.server.url_for("execution"))

        page.get_by_role(
            "heading", name="This model has reached its usage limit"
        ).wait_for()
        self.assertTrue(
            page.get_by_text("change model", exact=False).first.is_visible()
        )
        self.assertEqual(self.console_errors, [])
        self.assertEqual(self.page_errors, [])

    def test_plan_polling_preserves_reader_scroll_position(self):
        page = self.tracked_page()
        page.set_viewport_size({"width": 768, "height": 420})
        page.goto(self.server.url_for("plan"))
        page.get_by_role("heading", name="Plan", exact=True).wait_for()
        page.evaluate("window.scrollTo(0, document.documentElement.scrollHeight)")
        before = page.evaluate("window.scrollY")
        self.assertGreater(before, 0)
        page.wait_for_timeout(4500)
        after = page.evaluate("window.scrollY")
        self.assertLessEqual(abs(after - before), 2)
        self.assertEqual(self.console_errors, [])
        self.assertEqual(self.page_errors, [])

    def test_sse_activity_is_immediate_and_preserves_focus_scroll_and_draft(self):
        page = self.tracked_page()
        page.set_viewport_size({"width": 768, "height": 500})
        page.goto(self.server.url_for("plan"))
        page.get_by_role("heading", name="Plan", exact=True).wait_for()
        page.get_by_role("button", name="Advanced", exact=True).click()
        page.locator(".task-row").first.get_by_role("button", name="Edit", exact=True).click()
        editor = page.get_by_label("What this task must do")
        draft = "Keep this unsaved draft while verified live activity arrives."
        editor.fill(draft)
        editor.focus()
        editor.scroll_into_view_if_needed()
        before_top = editor.evaluate("element => element.getBoundingClientRect().top")

        started = time.perf_counter()
        self.runtime.events.publish(
            "provider.activity",
            "Provider request sent",
            source_kind="MODEL",
            phase="planning",
            actor="planner",
            state="started",
            provider_state="request_sent",
            operation="Preparing the verified project plan",
            received_bytes=0,
            received_chunks=0,
        )
        page.get_by_text("Request open · no response bytes yet", exact=False).wait_for(timeout=1000)
        self.assertLess(time.perf_counter() - started, 1.0)

        self.runtime.events.publish(
            "provider.activity",
            "First response bytes received",
            source_kind="MODEL",
            phase="planning",
            actor="planner",
            state="receiving",
            provider_state="receiving",
            operation="Receiving the verified project plan",
            received_bytes=2048,
            received_chunks=2,
        )
        page.get_by_text("Receiving model output · 2.0 KB · 2 chunks", exact=False).wait_for(timeout=1000)
        self.assertEqual(editor.input_value(), draft)
        self.assertTrue(editor.evaluate("element => document.activeElement === element"))
        after_top = editor.evaluate("element => element.getBoundingClientRect().top")
        self.assertLessEqual(abs(after_top - before_top), 2)
        self.assertEqual(self.console_errors, [])
        self.assertEqual(self.page_errors, [])

    def test_sse_disconnect_is_truthful_and_stops_live_animation(self):
        page = self.tracked_page()
        page.route(
            "**/api/sessions/*/events",
            lambda route: route.fulfill(
                status=200,
                content_type="text/event-stream",
                body="",
            ),
        )
        page.goto(self.server.url_for("plan"))
        page.get_by_role("heading", name="Plan", exact=True).wait_for()
        page.get_by_text(
            "Live connection lost · saved state is unchanged · reconnecting",
            exact=True,
        ).wait_for(timeout=3000)
        self.assertFalse(page.locator("#liveWorkflow").evaluate(
            "element => element.classList.contains('is-active')"
        ))
        self.assertTrue(page.locator("#liveWorkflow").evaluate(
            "element => element.classList.contains('is-stalled')"
        ))
        self.assertIn(
            page.locator("#connection span").text_content(),
            {"Reconnecting", "Polling"},
        )
        self.assertEqual(self.console_errors, [])
        self.assertEqual(self.page_errors, [])

    def test_review_uses_inline_feedback_and_execution_has_no_fake_progress(self):
        node, agent, _checkpoint = self.create_review_fixture()
        page = self.tracked_page()
        page.goto(self.server.url_for("review"))
        page.locator("#attentionTitle").get_by_text(
            "Changes need your attention", exact=True
        ).wait_for()
        self.assertEqual(
            page.locator("#liveNowTitle").text_content().strip(),
            "Changes need your attention",
        )
        self.assertIn("Checking changes", page.locator("#workflowIdentity").text_content())
        self.assertIn("ready for review", page.locator("#liveEvidence").text_content().lower())
        page.get_by_role("heading", name="Recorded changes", exact=True).wait_for()
        page.wait_for_timeout(250)
        page.locator("#attention").scroll_into_view_if_needed()
        page.screenshot(path=self.artifacts / "workspace-unified-review.png")
        self.assertEqual(page.get_by_text("0 of 1 files reviewed").count(), 1)
        self.assertTrue(page.get_by_role("button", name="Submit review").is_disabled())

        page.get_by_role("button", name="View changes", exact=True).click()
        page.get_by_role("heading", name="Changes", exact=True).wait_for()
        self.assertEqual(page.locator(".diff-line.added").count(), 2)
        self.assertEqual(page.locator(".diff-line.deleted").count(), 1)
        self.assertTrue(page.get_by_text("+    return 2", exact=True).is_visible())
        self.assertTrue(page.get_by_text("-    return 1", exact=True).is_visible())
        self.assertEqual(page.get_by_role("button", name="Hide changes").count(), 1)
        self.assertFalse(page.get_by_text("Raw diff", exact=True).is_visible())
        page.screenshot(path=self.artifacts / "workspace-review-diff-simple.png", full_page=True)

        page.get_by_role("button", name="Advanced", exact=True).click()
        page.get_by_text("Raw diff", exact=True).wait_for()
        page.get_by_role("button", name="Simple", exact=True).click()

        page.get_by_role("button", name="Request changes").click()
        page.get_by_label("Required feedback").fill("Cover the invalid return path.")
        page.get_by_role("button", name="Save change request").click()
        self.assertEqual(page.get_by_text("1 of 1 files reviewed").count(), 1)
        page.get_by_role("button", name="Submit review").click()
        page.get_by_role("button", name="Confirm submit").click()
        page.locator("#toastRegion").get_by_text(
            "Changes submitted. A fixer has started."
        ).wait_for()
        self.assertEqual(self.store.get_work_node(node.id).status, WorkNodeStatus.FIXING)

        page.get_by_role("button", name="Result").click()
        page.get_by_role("heading", name="Result", exact=True).wait_for()
        page.get_by_role("button", name="Pause safely").wait_for()
        page.get_by_role("button", name="Change model").wait_for()
        page.get_by_role("link", name="Open preview").wait_for()
        self.assertEqual(page.get_by_text("agent/example.py", exact=True).count(), 1)
        self.assertNotIn("token=hidden", page.content())
        page.screenshot(path=self.artifacts / "workspace-execution-result.png", full_page=True)
        self.assertEqual(page.get_by_text("62% authoritative").count(), 0)
        self.assertEqual(
            page.get_by_text("activity reported without an authoritative percentage").count(),
            1,
        )
        page.get_by_role("button", name="Advanced", exact=True).click()
        page.get_by_text(agent.id, exact=True).wait_for()
        page.get_by_role("button", name="Ask for explanation").click()
        page.get_by_role("heading", name="Request an explanation").wait_for()
        self.assertEqual(page.locator("textarea#agentQuestion").count(), 1)
        page.screenshot(path=self.artifacts / "workspace-unified-execution.png")
        self.assertEqual(self.console_errors, [])
        self.assertEqual(self.page_errors, [])

    def test_history_collapses_routine_provider_noise_by_default(self):
        self.runtime.approve_plan(1)
        self.store.append_event(
            "provider.heartbeat",
            goal_id=self.goal.id,
            entity_type="provider",
            entity_id="scripted",
            payload={"summary": "Routine provider heartbeat", "actor": "provider"},
        )
        self.store.append_event(
            "workflow.blocked",
            goal_id=self.goal.id,
            entity_type="workflow",
            entity_id=self.goal.id,
            payload={"summary": "A decision is required", "actor": "harness"},
        )
        page = self.tracked_page()
        page.goto(self.server.url_for("history"))
        page.get_by_role("heading", name="History", exact=True).wait_for()
        self.assertFalse(page.get_by_text("Model activity recorded", exact=True).is_visible())
        self.assertTrue(page.get_by_text("A decision is required", exact=True).is_visible())
        page.get_by_label("Show").select_option("all")
        self.assertTrue(page.get_by_text("Model activity recorded", exact=True).is_visible())
        page.screenshot(path=self.artifacts / "workspace-history-filtered.png", full_page=True)
        self.assertEqual(self.console_errors, [])
        self.assertEqual(self.page_errors, [])

    def test_global_composer_slash_navigation_and_model_picker(self):
        page = self.tracked_page()
        page.route(
            "**/api/sessions/*/models",
            lambda route: route.fulfill(
                status=200,
                content_type="application/json",
                body=(
                    '{"current_id":"ollama:coder@http://localhost:11434",'
                    '"safe_checkpoint":true,"diagnostics":[],"models":['
                    '{"id":"ollama:coder@http://localhost:11434",'
                    '"provider":"ollama","model":"coder","execution_class":"local",'
                    '"capabilities":["tools"],"display_name":"Coder (ollama)",'
                    '"selected":true}]}'
                ),
            ),
        )
        page.goto(self.server.url_for("plan"))
        composer = page.get_by_label("Request", exact=True)
        composer.fill("/")
        page.get_by_role("option", name="/model", exact=False).wait_for()
        page.locator("#modelButton").click()
        page.get_by_role("heading", name="Choose the right model").wait_for()
        page.get_by_role("button", name="Coder (ollama)", exact=False).wait_for()
        self.assertFalse(page.locator("#commandMenu").is_visible())
        self.assertTrue(page.locator(".app-shell").evaluate("element => element.inert"))
        self.assertEqual(page.evaluate("document.activeElement.id"), "drawerClose")
        page.screenshot(path=self.artifacts / "workspace-model-picker.png")
        page.get_by_role("button", name="Close").click()
        self.assertFalse(page.locator(".app-shell").evaluate("element => element.inert"))
        self.assertEqual(page.evaluate("document.activeElement.id"), "modelButton")

        composer.fill("/")
        for command in ("/continue with local model", "/tree", "/agents", "/tools", "/activity", "/status", "/refresh", "/changes"):
            page.get_by_role("option", name=command, exact=False).wait_for()
        page.screenshot(path=self.artifacts / "workspace-command-menu.png")
        page.get_by_role("option", name="/tree", exact=False).click()
        page.get_by_role("heading", name="Result", exact=True).wait_for()
        self.assertEqual(page.url.split("/")[-1], "execution")
        timeline_row = page.locator("#liveTimeline .live-event").last
        timeline_row.wait_for()
        timeline_row.evaluate("element => { element.dataset.stabilityProbe = 'kept'; }")
        page.wait_for_timeout(300)
        page.evaluate("""
          () => {
            window.timelineMutationCount = 0;
            window.timelineObserver = new MutationObserver((items) => {
              window.timelineMutationCount += items.length;
            });
            window.timelineObserver.observe(document.querySelector('#liveTimeline'), {
              subtree: true, childList: true, characterData: true, attributes: true,
            });
          }
        """)
        page.wait_for_timeout(2200)
        self.assertEqual(page.locator('[data-stability-probe="kept"]').count(), 1)
        self.assertEqual(page.evaluate("window.timelineMutationCount"), 0)
        page.evaluate("window.timelineObserver.disconnect()")
        submitted_actions: list[dict] = []

        def accept_local_continuation(route):
            submitted_actions.append(route.request.post_data_json)
            route.fulfill(
                status=200,
                content_type="application/json",
                body=(
                    '{"accepted":true,"action":"continue_local_model","source":"web",'
                    '"duplicate":false,"next_view":"execution","next_phase":"running",'
                    '"event_sequence":42,"message":"Continuing with ollama/local. Quality gates unchanged."}'
                ),
            )

        page.route("**/api/sessions/*/actions", accept_local_continuation)
        composer = page.get_by_label("Request", exact=True)
        composer.fill("/continue with local model")
        composer.press("Enter")
        page.locator("#toastRegion").get_by_text("Continuing with ollama/local", exact=False).wait_for()
        self.assertEqual(submitted_actions[0]["action"], "continue_local_model")
        self.assertEqual(self.console_errors, [])
        self.assertEqual(self.page_errors, [])

    def test_workflow_status_names_automatic_local_fallback(self):
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
        page = self.tracked_page()
        page.goto(self.server.url_for("plan"))
        composer = page.get_by_label("Request", exact=True)
        composer.fill("/status")
        composer.press("Enter")
        page.get_by_role("heading", name="Workflow status", exact=True).wait_for()
        self.assertTrue(
            page.get_by_text("Full Auto switched to ollama/coder", exact=True).is_visible()
        )
        page.get_by_role("button", name="Close", exact=True).click()
        self.assertEqual(self.console_errors, [])
        self.assertEqual(self.page_errors, [])

    def test_pending_tool_approval_is_visible_and_stoppable_in_web(self):
        self.runtime.approve_plan(1)
        fingerprint = "browser-approval-" + ("c" * 64)
        self.store.update_goal_metadata(
            self.goal.id,
            pending_tool_approval={
                "tool": "run_command",
                "arguments": {"command": "npm install three"},
                "risk": "critical",
                "action_fingerprint": fingerprint,
                "policy_group": "dangerous_command",
            },
        )
        page = self.tracked_page()
        page.goto(self.server.url_for("execution"))
        page.get_by_role("heading", name="Allow run command?").wait_for()
        page.get_by_role("heading", name="Waiting for your approval").wait_for()
        page.get_by_role("button", name="Allow once", exact=True).wait_for()
        page.get_by_role("button", name="Always allow this session", exact=True).wait_for()
        page.get_by_role("button", name="Deny", exact=True).wait_for()
        page.get_by_role("button", name="Stop safely", exact=True).wait_for()
        self.assertEqual(
            page.locator(".view-head .status-badge").text_content().strip(),
            "Waiting",
        )
        self.assertEqual(page.get_by_role("button", name="Pause safely", exact=True).count(), 0)
        self.assertTrue(page.get_by_text("Approval required above", exact=True).is_visible())
        self.assertTrue(page.get_by_text("npm install three", exact=False).is_visible())
        page.screenshot(path=self.artifacts / "workspace-tool-approval.png", full_page=True)
        page.set_viewport_size({"width": 390, "height": 844})
        self.assertTrue(page.get_by_role("button", name="Stop safely", exact=True).is_visible())
        page.screenshot(path=self.artifacts / "workspace-tool-approval-mobile.png", full_page=True)

        self.assertTrue(page.get_by_text("Session access applies to: dangerous command", exact=False).is_visible())
        page.get_by_role("button", name="Always allow this session", exact=True).click()
        page.locator("#toolApprovalActions").wait_for(state="hidden")
        self.assertEqual(page.get_by_text("already accepted", exact=False).count(), 0)
        self.assertEqual(
            self.store.get_goal(self.goal.id).metadata["pending_tool_approval"]["decision"],
            "allow_session",
        )
        self.assertIn("dangerous_command", self.runtime._approval_session_groups())
        self.assertTrue(
            self.runtime._approval_allowed(
                "run_command", {"command": "npm install another-package"}, "critical"
            )
        )
        page.reload()
        page.get_by_role("heading", name="Result", exact=True).wait_for()
        self.assertEqual(page.locator("#toolApprovalActions:not(.hidden)").count(), 0)
        self.assertEqual(self.console_errors, [])
        self.assertEqual(self.page_errors, [])

    def test_sleep_drawer_enables_explicit_full_auto(self):
        page = self.tracked_page()
        page.goto(self.server.url_for("plan"))
        page.locator("#sleepToggle").click()
        page.get_by_role("heading", name="Sleep mode", exact=True).wait_for()
        enable = page.get_by_role("button", name="Enable Full Auto", exact=True)
        self.assertTrue(enable.is_disabled())
        page.get_by_label("Type FULL AUTO to confirm").fill("FULL AUTO")
        self.assertFalse(enable.is_disabled())
        page.screenshot(path=self.artifacts / "workspace-sleep-modes.png")
        page.set_viewport_size({"width": 390, "height": 844})
        self.assertTrue(enable.is_visible())
        page.screenshot(path=self.artifacts / "workspace-sleep-modes-mobile.png")
        enable.click()
        page.locator("#sleepTopbar[aria-label='Sleep mode: Full Auto']").wait_for()
        self.assertEqual(self.runtime.sleep_mode_policy(), "full")
        deadline = time.monotonic() + 3
        while self.store.get_goal(self.goal.id).status is not GoalStatus.RUNNING and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertEqual(self.store.get_goal(self.goal.id).status, GoalStatus.RUNNING)
        self.assertTrue(
            any(
                item.event_type == "sleep.full_auto_plan_approval"
                for item in self.store.list_recent_events(self.goal.id, limit=100)
            )
        )
        self.assertEqual(self.console_errors, [])
        self.assertEqual(self.page_errors, [])

    def test_sleep_action_failure_closes_drawer_and_exposes_runtime_recovery(self):
        page = self.tracked_page()
        page.goto(self.server.url_for("plan"))
        page.locator("#sleepToggle").click()
        page.get_by_role("heading", name="Sleep mode", exact=True).wait_for()
        page.get_by_label("Type FULL AUTO to confirm").fill("FULL AUTO")

        # Simulate the exact unattended edge case: the loopback runtime dies
        # after the drawer is open but before the action POST is sent.
        self.server.stop()
        page.get_by_role("button", name="Enable Full Auto", exact=True).click()
        page.get_by_role(
            "heading", name="The local runtime is not responding", exact=True
        ).wait_for()
        self.assertFalse(
            page.get_by_role("heading", name="Sleep mode", exact=True).is_visible()
        )
        self.assertTrue(page.get_by_role("button", name="Retry", exact=True).is_visible())
        self.assertTrue(
            all("ERR_CONNECTION_REFUSED" in message for message in self.console_errors)
        )
        self.assertEqual(self.page_errors, [])

    def test_terminal_fallback_tracks_a_real_browser_connection(self):
        self.assertFalse(bool(getattr(self.runtime, "web_control_connected", False)))
        ownership: list[bool] = []
        self.runtime.web_control_state_sink = ownership.append
        page = self.tracked_page()
        page.goto(self.server.url_for("plan"))
        deadline = time.monotonic() + 3
        while not bool(getattr(self.runtime, "web_control_connected", False)) and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertTrue(bool(getattr(self.runtime, "web_control_connected", False)))
        self.assertIn(True, ownership)

        page.close()
        deadline = time.monotonic() + 3
        while bool(getattr(self.runtime, "web_control_connected", False)) and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertFalse(bool(getattr(self.runtime, "web_control_connected", False)))
        self.assertEqual(ownership[-1], False)


if __name__ == "__main__":
    unittest.main()
