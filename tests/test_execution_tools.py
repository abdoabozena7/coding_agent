"""Security and lifecycle tests for patch, process, and HTML preview tools."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
from urllib.error import HTTPError
from urllib.request import urlopen

from agent import tools


class ExecutionToolTests(unittest.TestCase):
    def test_apply_patch_updates_matching_preimage(self):
        with tempfile.TemporaryDirectory() as directory, tools.workspace_context(directory):
            Path(directory, "a.txt").write_text("one\ntwo\n", encoding="utf-8")
            patch = "--- a/a.txt\n+++ b/a.txt\n@@ -1,2 +1,2 @@\n one\n-two\n+three\n"
            result = tools.run_tool("apply_patch", {"patch": patch})
            self.assertFalse(result.startswith("Error:"), result)
            self.assertEqual(Path(directory, "a.txt").read_text(encoding="utf-8"), "one\nthree\n")

    def test_apply_patch_accepts_codex_update_envelope_and_duplicate_end_marker(self):
        with tempfile.TemporaryDirectory() as directory, tools.workspace_context(directory):
            Path(directory, "style.css").write_text(
                "#canvas {\n  display: block;\n}\n\n#panel {\n  color: white;\n}\n",
                encoding="utf-8",
            )
            patch = """*** Begin Patch
*** Update File: style.css
@@
 #canvas {
   display: block;
+  position: absolute;
 }
@@
 #panel {
-  color: white;
+  color: lime;
 }
*** End Patch
*** End Patch"""

            result = tools.run_tool("apply_patch", {"patch": patch})

            self.assertFalse(result.startswith("Error:"), result)
            self.assertEqual(
                Path(directory, "style.css").read_text(encoding="utf-8"),
                "#canvas {\n  display: block;\n  position: absolute;\n}\n\n"
                "#panel {\n  color: lime;\n}\n",
            )

    def test_apply_patch_rejects_ambiguous_codex_update_without_mutation(self):
        with tempfile.TemporaryDirectory() as directory, tools.workspace_context(directory):
            target = Path(directory, "repeated.txt")
            target.write_text("same\nkeep\nsame\nkeep\n", encoding="utf-8")
            patch = """*** Begin Patch
*** Update File: repeated.txt
@@
-same
+changed
*** End Patch"""

            result = tools.run_tool("apply_patch", {"patch": patch})

            self.assertIn("preimage is ambiguous", result)
            self.assertEqual(target.read_text(encoding="utf-8"), "same\nkeep\nsame\nkeep\n")

    def test_apply_patch_accepts_codex_delete_envelope(self):
        with tempfile.TemporaryDirectory() as directory, tools.workspace_context(directory):
            target = Path(directory, "obsolete.txt")
            target.write_text("remove me\n", encoding="utf-8")

            result = tools.run_tool("apply_patch", {"patch": """*** Begin Patch
*** Delete File: obsolete.txt
*** End Patch"""})

            self.assertFalse(result.startswith("Error:"), result)
            self.assertFalse(target.exists())

    def test_apply_patch_accepts_multiple_codex_envelopes_in_one_call(self):
        with tempfile.TemporaryDirectory() as directory, tools.workspace_context(directory):
            patch = """*** Begin Patch
*** Add File: first.txt
+first
*** End Patch
*** Begin Patch
*** Add File: nested/second.txt
+second
*** End Patch
*** End Patch"""

            result = tools.run_tool("apply_patch", {"patch": patch})

            self.assertFalse(result.startswith("Error:"), result)
            self.assertEqual(Path(directory, "first.txt").read_text(encoding="utf-8"), "first\n")
            self.assertEqual(
                Path(directory, "nested", "second.txt").read_text(encoding="utf-8"),
                "second\n",
            )

    def test_apply_patch_rejects_sensitive_and_traversal_paths_without_partial_writes(self):
        with tempfile.TemporaryDirectory() as directory, tools.workspace_context(directory):
            Path(directory, "safe.txt").write_text("before\n", encoding="utf-8")
            patches = (
                "--- /dev/null\n+++ b/../escape.txt\n@@ -0,0 +1,1 @@\n+bad\n",
                "--- /dev/null\n+++ b/.env\n@@ -0,0 +1,1 @@\n+SECRET=bad\n",
            )
            for patch in patches:
                with self.subTest(patch=patch.splitlines()[1]):
                    result = tools.run_tool("apply_patch", {"patch": patch})
                    self.assertTrue(result.startswith("Error:"), result)
            self.assertEqual(Path(directory, "safe.txt").read_text(encoding="utf-8"), "before\n")
            self.assertFalse(Path(directory).parent.joinpath("escape.txt").exists())

    def test_managed_process_readiness_output_and_stop(self):
        with tempfile.TemporaryDirectory() as directory, tools.workspace_context(directory):
            command = f'"{sys.executable}" -u -c "import time;print(\'READY\');time.sleep(30)"'
            started = json.loads(tools.run_tool("start_process", {
                "command": command, "readiness_type": "log", "readiness_value": "READY", "timeout_seconds": 5,
            }))
            try:
                self.assertTrue(started["ready"])
                output = json.loads(tools.run_tool("read_process_output", {"process_id": started["process_id"]}))
                self.assertIn("READY", output["output"])
                stopped = json.loads(tools.run_tool("stop_process", {"process_id": started["process_id"]}))
                self.assertTrue(stopped["stopped"])
            finally:
                tools.process_manager.shutdown_workspace(directory)

    def test_preview_is_loopback_tokenized_and_hides_sensitive_files(self):
        with tempfile.TemporaryDirectory() as directory, tools.workspace_context(directory):
            Path(directory, "index.html").write_text("<!doctype html><title>safe</title>", encoding="utf-8")
            Path(directory, ".env").write_text("SECRET=yes", encoding="utf-8")
            Path(directory, "innocent.txt").write_text(
                "-----BEGIN PRIVATE KEY-----\n" + ("secret" * 12) + "\n-----END PRIVATE KEY-----", encoding="utf-8",
            )
            payload = json.loads(tools.run_tool("preview_html", {
                "path": "index.html", "open_browser": False, "verify": False,
            }))
            self.addCleanup(tools.web_preview.shutdown_workspace, directory)
            self.assertTrue(payload["url"].startswith("http://127.0.0.1:"))
            self.assertEqual(urlopen(payload["url"]).status, 200)
            base = payload["url"].rsplit("/", 1)[0]
            for suffix in ("/.env", "/../.env", "/%2e%2e/.env", "/innocent.txt"):
                with self.assertRaises(HTTPError):
                    urlopen(base + suffix)
            tools.run_tool("stop_preview", {"preview_id": payload["preview_id"]})

    def test_preview_reports_page_errors_and_screenshot_with_real_browser(self):
        capability = tools.web_preview.browser_capability()
        if not capability["available"] or not capability["playwright"]:
            self.skipTest("Playwright plus Chrome/Edge/Chromium is unavailable")
        with tempfile.TemporaryDirectory() as directory, tools.workspace_context(
            directory,
            session_id="session-ui",
            goal_id="goal-web",
            task_id="T009",
        ):
            Path(directory, "index.html").write_text(
                "<!doctype html><title>broken</title><script>throw new Error('boom')</script>", encoding="utf-8",
            )
            payload = json.loads(tools.run_tool("preview_html", {
                "path": "index.html", "open_browser": False, "verify": True, "settle_ms": 50,
            }))
            self.addCleanup(tools.web_preview.shutdown_workspace, directory)
            self.assertEqual(payload["verification"], "failed")
            self.assertTrue(any("boom" in item for item in payload["page_errors"]))
            self.assertTrue(Path(payload["screenshot_path"]).exists())
            self.assertEqual(
                Path(payload["screenshot_path"]).parent,
                Path(directory, "output", "playwright", "session-ui", "goal-web", "T009"),
            )
            tools.run_tool("stop_preview", {"preview_id": payload["preview_id"]})

    def test_preview_runs_typed_interactions_and_records_observed_values(self):
        capability = tools.web_preview.browser_capability()
        if not capability["available"] or not capability["playwright"]:
            self.skipTest("Playwright plus Chrome/Edge/Chromium is unavailable")
        with tempfile.TemporaryDirectory() as directory, tools.workspace_context(directory):
            Path(directory, "index.html").write_text(
                """<!doctype html><button aria-label='Add one'>+</button>
                <input aria-label='Result' readonly value='0'>
                <script>document.querySelector('button').onclick=()=>{
                document.querySelector('input').value='1'}</script>""",
                encoding="utf-8",
            )
            payload = json.loads(tools.run_tool("preview_html", {
                "path": "index.html",
                "open_browser": False,
                "verify": True,
                "settle_ms": 0,
                "interactions": [{
                    "name": "increment",
                    "steps": [{"action": "click", "role": "button", "name": "Add one"}],
                    "assertions": [{
                        "role": "textbox",
                        "name": "Result",
                        "property": "value",
                        "equals": "1",
                    }],
                }],
            }))
            self.addCleanup(tools.web_preview.shutdown_workspace, directory)
            self.assertEqual(payload["verification"], "passed")
            self.assertTrue(payload["interaction_results"][0]["passed"])
            self.assertEqual(
                payload["interaction_results"][0]["assertions"][0]["observed"],
                "1",
            )
            tools.run_tool("stop_preview", {"preview_id": payload["preview_id"]})

    def test_preview_supports_bounded_dom_identity_visibility_and_count_assertions(self):
        capability = tools.web_preview.browser_capability()
        if not capability["available"] or not capability["playwright"]:
            self.skipTest("Playwright plus Chrome/Edge/Chromium is unavailable")
        with tempfile.TemporaryDirectory() as directory, tools.workspace_context(directory):
            Path(directory, "index.html").write_text(
                """<!doctype html><div id='threeD-canvas' data-object-count='3'
                data-visual-state='5+3=8'>ready</div>
                <div class='object'>one</div><div class='object'>two</div>
                <input id='enabled' type='checkbox' checked>""",
                encoding="utf-8",
            )
            payload = json.loads(tools.run_tool("preview_html", {
                "path": "index.html",
                "open_browser": False,
                "verify": True,
                "settle_ms": 0,
                "interactions": [{
                    "name": "bounded DOM observables",
                    "steps": [],
                    "assertions": [
                        {
                            "selector": "#threeD-canvas",
                            "property": "id",
                            "equals": "threeD-canvas",
                        },
                        {
                            "selector": "#threeD-canvas",
                            "property": "visible",
                            "equals": "true",
                        },
                        {
                            "selector": ".object",
                            "property": "count",
                            "equals": "2",
                        },
                        {
                            "selector": ".object",
                            "property": "visibleCount",
                            "equals": "2",
                        },
                        {
                            "selector": "#enabled",
                            "property": "checked",
                            "equals": "true",
                        },
                        {
                            "selector": "#threeD-canvas",
                            "property": "dataObjectCount",
                            "equals": "3",
                        },
                        {
                            "selector": "#threeD-canvas",
                            "property": "dataVisualState",
                            "equals": "5+3=8",
                        },
                    ],
                }],
            }))
            self.addCleanup(tools.web_preview.shutdown_workspace, directory)
            self.assertEqual(payload["verification"], "passed")
            observations = payload["interaction_results"][0]["assertions"]
            self.assertEqual(
                [item["observed"] for item in observations],
                ["threeD-canvas", True, 2, 2, True, "3", "5+3=8"],
            )
            tools.run_tool("stop_preview", {"preview_id": payload["preview_id"]})

    def test_preview_normalizes_small_model_dom_id_and_textcontent_transport(self):
        capability = tools.web_preview.browser_capability()
        if not capability["available"] or not capability["playwright"]:
            self.skipTest("Playwright plus Chrome/Edge/Chromium is unavailable")
        with tempfile.TemporaryDirectory() as directory, tools.workspace_context(directory):
            Path(directory, "index.html").write_text(
                """<!doctype html><h1 id='counter'>Count: 0</h1>
                <button id='increment'>Click to Increment</button>
                <script>document.querySelector('#increment').onclick=()=>{
                document.querySelector('#counter').textContent='Count: 1'}</script>""",
                encoding="utf-8",
            )
            payload = json.loads(tools.run_tool("preview_html", {
                "path": "index.html",
                "open_browser": False,
                "verify": True,
                "settle_ms": 0,
                "interactions": [{
                    "name": "captured local-model scenario",
                    # The model placed the DOM id in the accessible-name slot.
                    "steps": [{"action": "click", "role": "button", "name": "increment"}],
                    # It also emitted a DOM property spelling and an invented
                    # label while preserving the exact asserted visible value.
                    "assertions": [{
                        "role": "text",
                        "name": "Visible Count After One Click",
                        "property": "textContent",
                        "equals": "Count: 1",
                    }],
                }],
            }))
            self.addCleanup(tools.web_preview.shutdown_workspace, directory)
            self.assertEqual(payload["verification"], "passed")
            self.assertEqual(
                payload["interaction_results"][0]["assertions"][0]["observed"],
                "Count: 1",
            )
            tools.run_tool("stop_preview", {"preview_id": payload["preview_id"]})

    def test_preview_reports_missing_model_selector_as_contract_failure_without_action_timeout(self):
        capability = tools.web_preview.browser_capability()
        if not capability["available"] or not capability["playwright"]:
            self.skipTest("Playwright plus Chrome/Edge/Chromium is unavailable")
        with tempfile.TemporaryDirectory() as directory, tools.workspace_context(directory):
            Path(directory, "index.html").write_text(
                "<!doctype html><button data-value='5'>5</button><input id='display' value='0'>",
                encoding="utf-8",
            )
            payload = json.loads(tools.run_tool("preview_html", {
                "path": "index.html",
                "open_browser": False,
                "verify": True,
                "settle_ms": 0,
                "interactions": [{
                    "name": "invented selector",
                    "steps": [{"action": "click", "selector": "#button-5"}],
                    "assertions": [{
                        "selector": "#display",
                        "property": "value",
                        "equals": "5",
                    }],
                }],
            }))
            self.addCleanup(tools.web_preview.shutdown_workspace, directory)
            self.assertEqual(payload["verification"], "failed")
            self.assertEqual(payload["failure_kind"], "contract")
            self.assertIn("matched no elements", payload["interaction_results"][0]["error"])
            self.assertTrue(
                any(item["data_value"] == "5" for item in payload["interaction_targets"])
            )
            tools.run_tool("stop_preview", {"preview_id": payload["preview_id"]})

    def test_preview_reports_missing_exact_id_assertion_as_application_failure(self):
        capability = tools.web_preview.browser_capability()
        if not capability["available"] or not capability["playwright"]:
            self.skipTest("Playwright plus Chrome/Edge/Chromium is unavailable")
        with tempfile.TemporaryDirectory() as directory, tools.workspace_context(directory):
            Path(directory, "index.html").write_text(
                "<!doctype html><button id='ac'>AC</button>",
                encoding="utf-8",
            )
            payload = json.loads(tools.run_tool("preview_html", {
                "path": "index.html",
                "open_browser": False,
                "verify": True,
                "settle_ms": 0,
                "interactions": [{
                    "name": "required canvas",
                    "steps": [{"action": "click", "selector": "#ac"}],
                    "assertions": [{
                        "selector": "#threeD-canvas",
                        "property": "id",
                        "equals": "threeD-canvas",
                    }],
                }],
            }))
            self.addCleanup(tools.web_preview.shutdown_workspace, directory)
            self.assertEqual(payload["verification"], "failed")
            self.assertEqual(payload["failure_kind"], "application")
            self.assertIn("matched no elements", payload["interaction_results"][0]["error"])
            tools.run_tool("stop_preview", {"preview_id": payload["preview_id"]})

    def test_preview_reports_zero_named_count_target_as_contract_failure(self):
        capability = tools.web_preview.browser_capability()
        if not capability["available"] or not capability["playwright"]:
            self.skipTest("Playwright plus Chrome/Edge/Chromium is unavailable")
        with tempfile.TemporaryDirectory() as directory, tools.workspace_context(directory):
            Path(directory, "index.html").write_text(
                "<!doctype html><button>2</button><canvas id='threeD-canvas'></canvas>",
                encoding="utf-8",
            )
            payload = json.loads(tools.run_tool("preview_html", {
                "path": "index.html",
                "open_browser": False,
                "verify": True,
                "settle_ms": 0,
                "interactions": [{
                    "name": "invented 3D object label",
                    "steps": [{"action": "click", "role": "button", "name": "2"}],
                    "assertions": [{
                        "role": "textbox",
                        "name": "3D Object State Check",
                        "property": "visibleCount",
                        "equals": "3",
                    }],
                }],
            }))
            self.addCleanup(tools.web_preview.shutdown_workspace, directory)
            self.assertEqual(payload["verification"], "failed")
            self.assertEqual(payload["failure_kind"], "contract")
            self.assertIn(
                "count target matched no elements",
                payload["interaction_results"][0]["error"],
            )
            tools.run_tool("stop_preview", {"preview_id": payload["preview_id"]})

    def test_dependency_install_auto_detects_npm_without_global_install(self):
        with tempfile.TemporaryDirectory() as directory, tools.workspace_context(directory):
            Path(directory, "package.json").write_text('{"name":"x"}', encoding="utf-8")
            completed = mock.Mock(returncode=0, stdout=b"ok")
            with mock.patch("agent.tools.install_dependencies.shutil.which", return_value="npm"), mock.patch(
                "agent.tools.install_dependencies.subprocess.run", return_value=completed
            ) as runner:
                payload = json.loads(tools.run_tool("install_dependencies", {"directory": "."}))
            self.assertEqual(payload["manager"], "npm")
            self.assertEqual(runner.call_args.args[0], ["npm", "install"])


if __name__ == "__main__":
    unittest.main()
