from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import base64
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
    ResultPackageV1,
    TaskContractV1,
    UltraRun,
    UltraRunStatus,
    WorkNode,
    WorkNodeStatus,
)
from agent.web_views.server import LocalWebServer
from tests.test_local_web_views import basis, task_value


class LocalWebViewsBrowserTests(unittest.TestCase):
    """Verify the simple Plan/Live workspace and standalone developer trace."""

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
            "Exercise the Ultra workspace",
            session_id=self.runtime.session_id,
        )
        self.plan = self.store.create_plan(
            self.goal.id,
            "Implement the first browser slice.",
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
        self.context.add_init_script("window.close = () => { window.__closeRequested = true; };")
        self.console_errors: list[str] = []
        self.page_errors: list[str] = []
        self.artifacts = Path("output/playwright")
        self.artifacts.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        self.context.close()
        self.runtime.close()
        self.store.close()
        self.temporary.cleanup()

    def page(self):
        page = self.context.new_page()
        page.on(
            "console",
            lambda message: self.console_errors.append(message.text)
            if message.type == "error" else None,
        )
        page.on("pageerror", lambda error: self.page_errors.append(str(error)))
        return page

    def assert_browser_clean(self) -> None:
        self.assertEqual(self.console_errors, [])
        self.assertEqual(self.page_errors, [])

    def test_standalone_output_renders_copy_ready_text_and_images(self):
        image = self.workspace / "output" / "browser" / "screen.png"
        image.parent.mkdir(parents=True)
        image.write_bytes(base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
        ))
        self.runtime._publish_output_tool({
            "title": "Browser task complete",
            "message": "The browser opened and the requested screenshot was captured.",
            "copy_sections": [{"id": "copy-1", "label": "Copy ready", "text": "Reusable final text"}],
            "assets": [{
                "id": "image-1", "path": "output/browser/screen.png",
                "label": "Captured page", "kind": "image", "sha256": "a" * 64,
                "byte_size": image.stat().st_size,
            }],
        })
        page = self.page()
        page.goto(self.server.url_for("output"), wait_until="networkidle")

        self.assertEqual(page.get_by_role("heading", name="Result", exact=True).count(), 1)
        self.assertEqual(page.get_by_text("Browser task complete", exact=True).count(), 0)
        self.assertEqual(page.get_by_text("Reusable final text", exact=True).count(), 1)
        self.assertEqual(page.get_by_role("button", name="Copy response").count(), 1)
        self.assertEqual(page.get_by_role("button", name="Copy section").count(), 1)
        image_locator = page.locator(".asset img")
        image_locator.scroll_into_view_if_needed()
        page.wait_for_function(
            "image => image.complete && image.naturalWidth > 0",
            arg=image_locator.element_handle(),
        )
        self.assertGreater(image_locator.evaluate("image => image.naturalWidth"), 0)
        page.get_by_role("button", name="Copy section").click()
        page.get_by_role("button", name="Copied").wait_for(timeout=2_000)
        self.assertEqual(page.get_by_role("button", name="Copied").count(), 1)
        page.screenshot(path=self.artifacts / "output-page.png", full_page=True)
        self.assert_browser_clean()

    def start_running_fixture(self) -> tuple[WorkNode, AgentRun]:
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
                status=WorkNodeStatus.IN_PROGRESS,
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
                usage={"progress": 62},
            )
        )
        self.store.save_change_set(
            ChangeSetV1(
                ultra_run_id=run.id,
                responsible_agent_id=agent.id,
                parent_id=node.id,
                status=ChangeSetStatus.REVIEWING,
                changed_files=("agent/example.py",),
                diff=(
                    "diff --git a/agent/example.py b/agent/example.py\n"
                    "--- a/agent/example.py\n+++ b/agent/example.py\n"
                    "@@ -1 +1 @@\n-return 1\n+return 2\n"
                ),
                metadata={"reason": "Implement the approved browser flow."},
            )
        )
        return node, agent

    def test_shell_has_exactly_plan_and_live_with_reference_theme(self):
        page = self.page()
        page.goto(self.server.url_for("plan"))
        page.get_by_role("heading", name="Here is the complete plan.").wait_for()

        self.assertEqual(
            [value.strip() for value in page.locator(".primary-nav button").all_text_contents()],
            ["Plan", "Live"],
        )
        self.assertEqual(page.locator("select").count(), 0)
        self.assertEqual(page.get_by_text("Full Auto", exact=False).count(), 0)
        self.assertEqual(
            page.locator("body").evaluate("node => getComputedStyle(node).backgroundColor"),
            "rgb(251, 250, 246)",
        )

        page.get_by_role("button", name="Live", exact=True).click()
        page.get_by_role("heading", name="Live work", exact=True).wait_for()
        self.assertTrue(page.url.endswith("/live"))
        self.assertEqual(
            [value.strip() for value in page.locator(".live-tabs button").all_text_contents()],
            ["Overview", "Agents", "Timeline"],
        )
        page.wait_for_timeout(450)
        page.screenshot(path=self.artifacts / "ultra-live-desktop.png", full_page=True)
        self.assert_browser_clean()

    def test_advanced_tracing_is_standalone_keyboard_navigable_and_clean(self):
        _node, agent = self.start_running_fixture()
        page = self.page()
        page.goto(self.server.url_for("advanced-tracing"))
        page.get_by_role("heading", name="Overview", exact=True).wait_for()

        self.assertTrue(page.url.endswith("/advanced-tracing"))
        self.assertEqual(page.locator(".primary-nav, .live-tabs").count(), 0)
        self.assertEqual(
            page.locator("body").evaluate("node => getComputedStyle(node).backgroundColor"),
            "rgb(247, 244, 236)",
        )

        page.get_by_role("button", name="Files", exact=False).click()
        page.get_by_role("heading", name="Files", exact=True).wait_for()
        self.assertNotIn("null", page.locator("main").inner_text().split())

        page.get_by_role("button", name="Agents & models", exact=False).click()
        page.get_by_role("heading", name="Agents & models", exact=True).wait_for()
        page.get_by_text(agent.role, exact=False).first.wait_for()
        page.keyboard.press("/")
        search = page.get_by_role("searchbox", name="Find in trace")
        self.assertTrue(search.evaluate("node => node === document.activeElement"))
        search.fill(agent.role)
        self.assertGreaterEqual(page.locator(".trace-row").count(), 1)
        page.screenshot(
            path=self.artifacts / "advanced-tracing-desktop.png",
            full_page=True,
        )
        self.assert_browser_clean()

    def test_show_diff_is_standalone_exact_and_keyboard_navigable(self):
        self.start_running_fixture()
        page = self.page()
        page.goto(self.server.url_for("show-diff"))
        page.get_by_role("heading", name="agent/example.py", exact=True).wait_for()

        self.assertTrue(page.url.endswith("/show-diff"))
        self.assertEqual(page.locator(".primary-nav, .live-tabs").count(), 0)
        self.assertEqual(page.locator(".file-row").count(), 1)
        self.assertGreater(page.locator(".diff-line.added").count(), 0)
        self.assertGreater(page.locator(".diff-line.deleted").count(), 0)
        self.assertTrue(page.get_by_role("button", name="Copy full diff").is_enabled())

        page.keyboard.press("/")
        search = page.get_by_role("searchbox", name="Find a file")
        self.assertTrue(search.evaluate("node => node === document.activeElement"))
        search.fill("example.py")
        self.assertEqual(page.locator(".file-row").count(), 1)
        page.screenshot(path=self.artifacts / "workflow-diff-desktop.png", full_page=True)
        self.assert_browser_clean()

    def test_advanced_changes_offer_copyable_history_and_exact_diff(self):
        self.start_running_fixture()
        page = self.page()
        page.goto(self.server.url_for("advanced-tracing"))
        page.get_by_role("button", name="Changes", exact=False).click()
        page.get_by_role("heading", name="Changes", exact=True).wait_for()
        self.assertTrue(page.get_by_role("button", name="Copy section").is_visible())
        page.locator(".trace-row").first.click()
        self.assertTrue(page.get_by_role("button", name="Copy record").is_visible())
        self.assertTrue(page.get_by_role("button", name="Copy diff").is_visible())
        self.assertIn("+return 2", page.locator(".diff-view").inner_text())
        self.assert_browser_clean()

    def test_plan_document_requires_team_preview_then_second_approval(self):
        page = self.page()
        page.goto(self.server.url_for("plan"))
        editor = page.get_by_label("Editable plan document")
        editor.wait_for()
        editor.fill(editor.input_value() + "\n\n2. Verify the result\nRun the focused browser checks.")

        self.assertEqual(page.get_by_role("button", name="Start Execution").count(), 0)
        page.get_by_role("button", name="Update Execution preview").click()
        page.get_by_role("heading", name="Execution preview").wait_for()
        self.assertEqual(page.locator(".team-member").count(), 2)
        self.assertEqual(self.store.get_latest_plan(self.goal.id).revision, 2)
        self.assertIsNone(self.store.get_goal(self.goal.id).active_plan_revision)
        page.wait_for_timeout(3800)
        page.screenshot(path=self.artifacts / "ultra-plan-team-preview.png", full_page=True)

        page.get_by_role("button", name="Start Execution").click()
        page.get_by_role("heading", name="Execution is running.").wait_for()
        self.assertEqual(self.store.get_goal(self.goal.id).active_plan_revision, 2)
        page.wait_for_timeout(200)
        self.assertTrue(page.evaluate("window.__closeRequested"))
        self.assert_browser_clean()

    def test_plan_prompt_survives_live_refresh_and_reload(self):
        self.store.transition_goal(
            self.goal.id,
            GoalStatus.CANCELLED,
            reason="show the empty Ultra Plan composer",
        )
        self.runtime.transition_mode("plan")
        page = self.page()
        page.goto(self.server.url_for("plan"))
        editor = page.get_by_label("What should Ultra Plan prepare?")
        editor.wait_for()
        draft = "Keep this detailed prompt while live events refresh the page."
        editor.fill(draft)

        page.evaluate("refresh({force: true})")
        page.wait_for_timeout(500)
        self.assertEqual(editor.input_value(), draft)
        self.assertTrue(editor.evaluate("node => document.activeElement === node"))

        page.reload()
        restored = page.get_by_label("What should Ultra Plan prepare?")
        restored.wait_for()
        self.assertEqual(restored.input_value(), draft)
        self.assertLess(restored.bounding_box()["height"], 180)
        page.wait_for_timeout(400)
        self.assertEqual(
            page.locator("#planPage").evaluate("node => getComputedStyle(node).opacity"),
            "1",
        )
        page.screenshot(path=self.artifacts / "ultra-plan-composer.png", full_page=True)
        self.assert_browser_clean()

    def test_live_tree_timeline_and_diff_are_read_only(self):
        node, _agent = self.start_running_fixture()
        page = self.page()
        page.goto(self.server.url_for("live"))
        page.get_by_role("heading", name="Live work", exact=True).wait_for()

        page.get_by_role("tab", name="Agents", exact=True).click()
        page.get_by_role("heading", name="GA3BAD Core", exact=True).wait_for()
        page.locator(f"[data-node='{node.id}']").click()
        self.assertTrue(
            page.locator(".agent-inspector").get_by_text(
                "Implement and verify the browser flow.", exact=True
            ).is_visible()
        )
        self.assertTrue(page.get_by_text("Files changing", exact=True).is_visible())

        page.get_by_role("tab", name="Timeline", exact=True).click()
        page.locator(".timeline-toolbar").wait_for()
        page.get_by_role("button", name="Problems", exact=True).click()
        self.assertEqual(page.locator("textarea").count(), 0)

        page.get_by_role("tab", name="Overview", exact=True).click()
        page.get_by_role("button", name="agent/example.py", exact=True).click()
        page.get_by_role("heading", name="agent/example.py", exact=True).wait_for()
        self.assertTrue(page.get_by_text("This is an audit view.", exact=False).is_visible())
        self.assertEqual(page.locator("#drawerBody button").count(), 0)
        page.screenshot(path=self.artifacts / "ultra-live-diff.png", full_page=True)
        self.assert_browser_clean()

    def test_live_agents_renders_every_run_with_waiting_state_and_result(self):
        node, _running = self.start_running_fixture()
        run = self.store.get_active_ultra_run(self.goal.id)
        queued = self.store.create_agent_run(
            AgentRun(
                ultra_run_id=run.id,
                work_node_id=node.id,
                role="tester",
                provider="test",
                model="offline",
                phase="testing",
                status=AgentRunStatus.QUEUED,
            )
        )
        completed = self.store.create_agent_run(
            AgentRun(
                ultra_run_id=run.id,
                work_node_id=node.id,
                role="reviewer",
                provider="test",
                model="offline",
                phase="reviewing",
                status=AgentRunStatus.COMPLETED,
                result=ResultPackageV1(
                    summary="The implementation passed independent review.",
                    changed_files=("agent/example.py",),
                ),
            )
        )
        page = self.page()
        page.goto(self.server.url_for("live"))
        page.get_by_role("tab", name="Agents", exact=True).click()

        page.locator(".agent-node.agent-run").nth(2).wait_for()
        self.assertEqual(page.locator(".agent-node.agent-run").count(), 3)
        self.assertTrue(page.get_by_text("Waiting", exact=True).is_visible())

        page.locator(f"[data-node='agent:{queued.id}']").click()
        self.assertTrue(page.get_by_text("Waiting for its dependency", exact=False).is_visible())
        page.locator(f"[data-node='agent:{completed.id}']").click()
        self.assertTrue(
            page.locator(".agent-result p").get_by_text(
                "The implementation passed independent review.", exact=True
            ).is_visible()
        )
        self.assertTrue(page.get_by_text("Latest result", exact=True).is_visible())
        page.screenshot(path=self.artifacts / "ultra-live-all-agents.png", full_page=True)
        self.assert_browser_clean()

    def test_blocker_exposes_only_small_recovery_controls(self):
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
        page = self.page()
        page.goto(self.server.url_for("live"))
        blocker = page.locator("#blocker:not(.hidden)")
        blocker.wait_for()
        self.assertEqual(
            [value.strip() for value in blocker.locator("button").all_text_contents()],
            ["Allow once", "Deny", "Stop safely"],
        )
        self.assertEqual(page.get_by_text("Always allow", exact=False).count(), 0)
        page.set_viewport_size({"width": 390, "height": 844})
        self.assertTrue(page.get_by_role("button", name="Stop safely", exact=True).is_visible())
        page.wait_for_timeout(450)
        page.screenshot(path=self.artifacts / "ultra-live-blocker-mobile.png", full_page=True)
        self.assert_browser_clean()


if __name__ == "__main__":
    unittest.main()
