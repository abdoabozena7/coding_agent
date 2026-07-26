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
    """Exercise the required artifact interactions in a real browser."""

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
        self.agent = self.store.create_agent_run(
            AgentRun(
                ultra_run_id=self.run.id,
                work_node_id=self.node.id,
                role="frontend",
                provider="test",
                model="offline",
                phase="implementation",
                status=AgentRunStatus.RUNNING,
                usage={"progress": 62, "tools": ["filesystem", "test_runner"]},
            )
        )
        self.checkpoint = self.store.save_change_set(
            ChangeSetV1(
                ultra_run_id=self.run.id,
                responsible_agent_id=self.agent.id,
                parent_id=self.node.id,
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
            )
        )
        self.server = LocalWebServer(self.runtime).start()
        self.runtime.local_web_server = self.server
        self.context = self.browser.new_context(viewport={"width": 1440, "height": 1000})

    def tearDown(self) -> None:
        self.context.close()
        self.runtime.close()
        self.store.close()
        self.temporary.cleanup()

    def test_plan_edit_reorder_modes_apply_and_conflict(self):
        page = self.context.new_page()
        page.goto(self.server.url_for("plan"))
        page.get_by_role("heading", name="Plan Studio").wait_for()
        page.wait_for_timeout(250)
        self.assertEqual(page.get_by_label("Task 1 runtime status").text_content(), "pending")
        self.assertEqual(page.locator('[data-field="status"]').count(), 0)
        self.assertEqual(
            page.locator("#viewRoot").evaluate(
                "element => getComputedStyle(element).opacity"
            ),
            "1",
        )
        self.assertEqual(
            page.get_by_role("heading", name="Plan Studio").evaluate(
                "element => getComputedStyle(element).color"
            ),
            "rgb(243, 244, 239)",
        )
        artifacts = Path("output/playwright")
        artifacts.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=artifacts / "local-web-plan-studio.png", full_page=True)

        page.get_by_role("button", name="Advanced", exact=True).click()
        self.assertTrue(page.get_by_label("Task ID").is_visible())
        page.get_by_role("button", name="+ Add task").click()
        task_titles = page.locator(".title-input")
        task_titles.nth(1).fill("Verify the new flow")
        page.locator(".task-row").nth(1).get_by_role("button", name="Move task up").click()
        self.assertEqual(page.locator(".title-input").first.input_value(), "Verify the new flow")
        page.get_by_role("button", name="Simple", exact=True).click()
        self.assertFalse(page.get_by_label("Task ID").first.is_visible())
        page.get_by_role("button", name="Apply to GA3BAD").click()
        page.get_by_text("Plan r1 → r2 applied.").wait_for()
        self.assertEqual(self.store.get_latest_plan(self.goal.id).revision, 2)

        stale = self.context.new_page()
        stale.goto(self.server.url_for("plan"))
        stale.get_by_role("heading", name="Plan Studio").wait_for()
        current = self.context.new_page()
        current.goto(self.server.url_for("plan"))
        current.get_by_role("heading", name="Plan Studio").wait_for()
        current.locator(".title-input").first.fill("Current revision edit")
        current.get_by_role("button", name="Apply to GA3BAD").click()
        current.get_by_text("Plan r2 → r3 applied.").wait_for()
        stale.locator(".title-input").first.fill("Stale revision edit")
        stale.get_by_role("button", name="Apply to GA3BAD").click()
        stale.get_by_text("This page was opened with Plan r2. The current plan is Plan r3.").wait_for()
        self.assertEqual(self.store.get_latest_plan(self.goal.id).tasks[0].title, "Current revision edit")

    def test_review_decisions_inline_comment_and_agent_panel(self):
        review = self.context.new_page()
        review.goto(self.server.url_for("review"))
        review.get_by_role("heading", name="Change Review").wait_for()
        review.wait_for_timeout(250)
        review.screenshot(
            path=Path("output/playwright/local-web-change-review.png"),
            full_page=True,
        )
        review.get_by_role("button", name="Accept file").click()
        review.once("dialog", lambda dialog: dialog.accept("Keep the validation, but adjust the return path."))
        review.get_by_role("button", name="Reject", exact=True).click()
        review.once("dialog", lambda dialog: dialog.accept("Explain why this line changes."))
        review.get_by_role("button", name="Comment on line 1").first.click()
        review.get_by_role("button", name="Submit review").click()
        review.get_by_text("Fixer started with your feedback.").wait_for()
        self.assertEqual(self.store.get_work_node(self.node.id).status, WorkNodeStatus.FIXING)

        agents = self.context.new_page()
        agents.goto(self.server.url_for("agents"))
        agents.get_by_role("heading", name="Execution Map").wait_for()
        agents.locator("[data-agent-id]").click()
        agents.get_by_text(self.agent.id, exact=True).wait_for()
        agents.get_by_role("button", name="Request explanation").wait_for()
        agents.screenshot(
            path=Path("output/playwright/local-web-agent-tree.png"),
            full_page=True,
        )


if __name__ == "__main__":
    unittest.main()
