from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
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
        return node, agent, checkpoint

    def test_plan_is_simple_read_first_and_approval_is_explicit(self):
        page = self.tracked_page()
        page.goto(self.server.url_for("plan"))
        page.get_by_role("heading", name="Plan", exact=True).wait_for()
        page.get_by_role("heading", name="Review the plan before work starts").wait_for()
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
            "Plan r2 approved. Execution is starting in the terminal."
        ).wait_for()
        self.assertEqual(self.store.get_goal(self.goal.id).active_plan_revision, 2)
        self.assertTrue(self.server.take_execution_request())
        self.assertFalse(self.server.take_execution_request())
        page.get_by_role("heading", name="Work is running in the terminal").wait_for()

        page.wait_for_timeout(250)
        page.screenshot(path=self.artifacts / "workspace-unified-desktop.png", full_page=True)
        page.set_viewport_size({"width": 768, "height": 900})
        page.screenshot(path=self.artifacts / "workspace-unified-tablet.png", full_page=True)
        page.set_viewport_size({"width": 390, "height": 844})
        page.screenshot(path=self.artifacts / "workspace-unified-mobile.png", full_page=True)
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

    def test_review_uses_inline_feedback_and_execution_has_no_fake_progress(self):
        node, agent, _checkpoint = self.create_review_fixture()
        page = self.tracked_page()
        page.goto(self.server.url_for("review"))
        page.get_by_role("heading", name="Review the recorded changes", exact=True).wait_for()
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
            "Review submitted. A fixer has started."
        ).wait_for()
        self.assertEqual(self.store.get_work_node(node.id).status, WorkNodeStatus.FIXING)

        page.get_by_role("button", name="Execution").click()
        page.get_by_role("heading", name="Execution", exact=True).wait_for()
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


if __name__ == "__main__":
    unittest.main()
