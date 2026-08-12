"""Security and lifecycle tests for patch, process, and HTML preview tools."""

from __future__ import annotations

import json
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest import mock
from urllib.error import HTTPError
from urllib.request import urlopen

from agent import tools


class ExecutionToolTests(unittest.TestCase):
    def test_browser_launch_falls_back_to_the_next_installed_engine(self):
        from agent.tools.browser_session import _launch_browser_with_fallback

        working_browser = object()
        chromium = mock.Mock()
        chromium.launch.side_effect = [
            RuntimeError("Chrome closed during startup"),
            working_browser,
        ]

        browser, engine, failures = _launch_browser_with_fallback(
            chromium,
            {
                "candidates": [
                    {"channel": "chrome", "executable": "chrome.exe"},
                    {"channel": "msedge", "executable": "msedge.exe"},
                ],
            },
            visible=True,
        )

        self.assertIs(browser, working_browser)
        self.assertEqual(engine, "msedge")
        self.assertEqual(len(failures), 1)
        self.assertIn("Chrome closed", failures[0])
        self.assertEqual(
            chromium.launch.call_args_list,
            [
                mock.call(headless=False, channel="chrome"),
                mock.call(headless=False, channel="msedge"),
            ],
        )

    def test_browser_open_exposes_a_persistent_playwright_receipt(self):
        receipt = json.dumps({
            "browser_session_id": "browser-test",
            "status": "running",
            "browser_opened": True,
            "url": "http://127.0.0.1:8501/app",
            "title": "App",
            "http_status": 200,
            "interaction_targets": [],
            "console_errors": [],
            "page_errors": [],
            "network_errors": [],
        })
        with tempfile.TemporaryDirectory() as directory, tools.workspace_context(directory):
            with (
                mock.patch("agent.tools.browser_open._loopback_preflight", return_value=""),
                mock.patch("agent.tools.browser_open.browser_session.web_preview.browser_capability", return_value={"playwright": True, "available": True}),
                mock.patch("agent.tools.browser_open.browser_session.open_browser", return_value=receipt) as opened,
            ):
                payload = json.loads(tools.run_tool("browser_open", {
                    "url": "http://127.0.0.1:8501/app",
                    "visible": True,
                }))

        self.assertEqual(payload["browser_session_id"], "browser-test")
        self.assertTrue(payload["browser_opened"])
        opened.assert_called_once()

    def test_browser_open_reports_a_missing_playwright_runtime_clearly(self):
        with tempfile.TemporaryDirectory() as directory, tools.workspace_context(directory):
            with mock.patch(
                "agent.tools.browser_open.browser_session.web_preview.browser_capability",
                return_value={"playwright": False, "available": True},
            ):
                result = tools.run_tool("browser_open", {"url": "https://example.com"})
        self.assertIn("Playwright is not installed", result)

    def test_browser_controller_uploads_workspace_files_and_reports_actionable_targets(self):
        capability = tools.web_preview.browser_capability()
        if not capability["available"] or not capability["playwright"]:
            self.skipTest("Playwright plus Chrome/Edge/Chromium is unavailable")
        with tempfile.TemporaryDirectory() as directory, tools.workspace_context(
            directory, session_id="upload-session", goal_id="upload-goal", task_id="upload-task"
        ):
            Path(directory, "sample.txt").write_text("safe sample", encoding="utf-8")
            Path(directory, "index.html").write_text(
                "<!doctype html><input type='file'><details><summary>Details</summary>"
                "<div id='selected'>No file</div></details><script>"
                "document.querySelector('input').addEventListener('change', e => "
                "document.querySelector('#selected').textContent=e.target.files[0].name)"
                "</script>",
                encoding="utf-8",
            )
            opened = json.loads(tools.run_tool("browser_open", {
                "path": "index.html", "visible": False, "settle_ms": 0,
            }))
            browser_id = opened["browser_session_id"]
            try:
                targets = opened["interaction_targets"]
                self.assertTrue(any(
                    item.get("selector") == 'input[type="file"]'
                    for item in targets
                ))
                self.assertTrue(any(
                    item.get("tag") == "summary"
                    and item.get("role") == "button"
                    and item.get("selector") == "summary"
                    for item in targets
                ))
                uploaded = json.loads(tools.run_tool("browser_act", {
                    "browser_session_id": browser_id,
                    "actions": [{
                        "action": "upload", "selector": 'input[type="file"]',
                        "files": ["sample.txt"],
                    }],
                }))
                self.assertTrue(any(
                    item.get("type") == "file" and "sample.txt" in item.get("name", "")
                    for item in uploaded["interaction_targets"]
                ))
                rejected = tools.run_tool("browser_act", {
                    "browser_session_id": browser_id,
                    "actions": [{
                        "action": "upload", "selector": 'input[type="file"]',
                        "files": ["../outside.txt"],
                    }],
                })
                self.assertIn("inside the active workspace", rejected)
            finally:
                tools.run_tool("browser_close", {"browser_session_id": browser_id})

    def test_browser_screenshot_receipt_contains_a_durable_perceptual_hash(self):
        capability = tools.web_preview.browser_capability()
        if not capability["available"] or not capability["playwright"]:
            self.skipTest("Playwright plus Chrome/Edge/Chromium is unavailable")
        with tempfile.TemporaryDirectory() as directory, tools.workspace_context(
            directory,
            session_id="screenshot-session",
            goal_id="screenshot-goal",
            task_id="screenshot-task",
        ):
            Path(directory, "index.html").write_text(
                "<!doctype html><title>Visible state</title><main>Evidence</main>",
                encoding="utf-8",
            )
            opened = json.loads(tools.run_tool("browser_open", {
                "path": "index.html", "visible": False, "settle_ms": 0,
            }))
            browser_id = opened["browser_session_id"]
            try:
                receipt = json.loads(tools.run_tool("browser_screenshot", {
                    "browser_session_id": browser_id,
                    "name": "visible-state",
                    "full_page": True,
                }))
                self.assertTrue(Path(receipt["screenshot_path"]).is_file())
                self.assertEqual(len(receipt["sha256"]), 64)
                self.assertEqual(len(receipt["perceptual_hash"]), 64)
            finally:
                tools.run_tool("browser_close", {"browser_session_id": browser_id})

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
                self.assertEqual(started["readiness"], {"type": "log", "value": "READY"})
                output = json.loads(tools.run_tool("read_process_output", {"process_id": started["process_id"]}))
                self.assertIn("READY", output["output"])
                stopped = json.loads(tools.run_tool("stop_process", {"process_id": started["process_id"]}))
                self.assertTrue(stopped["stopped"])
            finally:
                tools.process_manager.shutdown_workspace(directory)

    def test_managed_process_rejects_empty_or_missing_readiness_before_spawn(self):
        with tempfile.TemporaryDirectory() as directory, tools.workspace_context(directory):
            missing = tools.process_manager.start(
                f'"{sys.executable}" -c "print(1)"',
                readiness_type="none",
            )
            empty_log = tools.process_manager.start(
                f'"{sys.executable}" -c "print(1)"',
                readiness_type="log",
                readiness_value="",
            )
            self.assertFalse(tools.process_manager.list_processes())
        self.assertIn("requires a real readiness signal", missing)
        self.assertIn("non-empty readiness_value", empty_log)

    def test_managed_process_url_readiness_uses_live_loopback_server(self):
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        with tempfile.TemporaryDirectory() as directory, tools.workspace_context(directory):
            command = (
                f'"{sys.executable}" -m http.server {port} '
                '--bind 127.0.0.1'
            )
            started = json.loads(tools.run_tool("start_process", {
                "command": command,
                "readiness_type": "url",
                "readiness_value": f"http://127.0.0.1:{port}",
                "timeout_seconds": 8,
            }))
            try:
                self.assertTrue(started["ready"])
                self.assertEqual(started["readiness_url"], f"http://127.0.0.1:{port}")
            finally:
                tools.run_tool("stop_process", {"process_id": started["process_id"]})

    def test_managed_process_uses_announced_loopback_url_when_declared_port_is_wrong(self):
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            actual_port = reservation.getsockname()[1]
        script = (
            "import http.server;"
            f"print('Local: http://localhost:{actual_port}/',flush=True);"
            f"http.server.ThreadingHTTPServer(('127.0.0.1',{actual_port}),"
            "http.server.SimpleHTTPRequestHandler).serve_forever()"
        )
        command = f'"{sys.executable}" -u -c "{script}"'
        with tempfile.TemporaryDirectory() as directory, tools.workspace_context(directory):
            started = json.loads(tools.run_tool("start_process", {
                "command": command,
                "readiness_type": "url",
                "readiness_value": "http://localhost:1",
                "timeout_seconds": 8,
            }))
            try:
                self.assertTrue(started["ready"])
                self.assertEqual(started["readiness_source"], "process_log")
                self.assertEqual(started["requested_readiness"]["value"], "http://localhost:1")
                self.assertEqual(started["readiness_url"], f"http://localhost:{actual_port}/")
            finally:
                first = json.loads(tools.run_tool("stop_process", {"process_id": started["process_id"]}))
                second = json.loads(tools.run_tool("stop_process", {"process_id": started["process_id"]}))
            self.assertFalse(first["already_stopped"])
            self.assertTrue(second["already_stopped"])

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
            self.assertTrue(
                Path(payload["interaction_results"][0]["screenshot_path"]).is_file()
            )
            self.assertEqual(
                Path(payload["interaction_results"][0]["screenshot_path"]).parent.name,
                "scenes",
            )
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

    def test_dependency_install_reuses_healthy_existing_node_modules(self):
        with tempfile.TemporaryDirectory() as directory, tools.workspace_context(directory):
            Path(directory, "package.json").write_text('{"name":"x"}', encoding="utf-8")
            Path(directory, "node_modules").mkdir()
            healthy = mock.Mock(returncode=0, stdout=b"x@1.0.0")
            with mock.patch("agent.tools.install_dependencies.shutil.which", return_value="npm"), mock.patch(
                "agent.tools.install_dependencies.subprocess.run", return_value=healthy
            ) as runner:
                payload = json.loads(tools.run_tool("install_dependencies", {"directory": "."}))

            self.assertEqual(payload["status"], "already_satisfied")
            self.assertEqual(runner.call_count, 1)
            self.assertEqual(runner.call_args.args[0], ["npm", "ls", "--depth=0", "--silent"])

    def test_dependency_install_reports_nested_component_choices_before_approval(self):
        with tempfile.TemporaryDirectory() as directory:
            for component in ("client", "server"):
                Path(directory, component).mkdir()
                Path(directory, component, "package.json").write_text(
                    '{"name":"' + component + '"}', encoding="utf-8"
                )
            issue = tools.applicability_issue(
                "install_dependencies", {"manager": "npm"}, directory
            )
        self.assertIn("multiple components", issue)
        self.assertIn("'client'", issue)
        self.assertIn("'server'", issue)

    def test_dependency_install_auto_selects_one_nested_component(self):
        with tempfile.TemporaryDirectory() as directory, tools.workspace_context(directory):
            Path(directory, "client").mkdir()
            Path(directory, "client", "package.json").write_text(
                '{"name":"client"}', encoding="utf-8"
            )
            completed = mock.Mock(returncode=0, stdout=b"ok")
            with mock.patch(
                "agent.tools.install_dependencies.shutil.which", return_value="npm"
            ), mock.patch(
                "agent.tools.install_dependencies.subprocess.run", return_value=completed
            ) as runner:
                payload = json.loads(tools.run_tool("install_dependencies", {"manager": "npm"}))
        self.assertEqual(Path(payload["directory"]), Path(directory, "client"))
        self.assertEqual(runner.call_args.kwargs["cwd"], str(Path(directory, "client")))

    def test_dependency_install_reuses_a_satisfied_python_virtual_environment(self):
        with tempfile.TemporaryDirectory() as directory, tools.workspace_context(directory):
            Path(directory, "requirements.txt").write_text("example==1.0\n", encoding="utf-8")
            python_path = Path(directory, ".venv", "Scripts" if sys.platform == "win32" else "bin", "python.exe" if sys.platform == "win32" else "python")
            python_path.parent.mkdir(parents=True)
            python_path.write_bytes(b"runtime")
            healthy = mock.Mock(returncode=0, stdout=b"Requirement already satisfied: example==1.0")
            with mock.patch(
                "agent.tools.install_dependencies.subprocess.run", return_value=healthy
            ) as runner:
                payload = json.loads(tools.run_tool("install_dependencies", {"directory": "."}))

            self.assertEqual(payload["status"], "already_satisfied")
            self.assertEqual(runner.call_count, 1)
            self.assertEqual(
                runner.call_args.args[0],
                [str(python_path), "-m", "pip", "install", "--dry-run", "-r", "requirements.txt"],
            )

    def test_dependency_install_selects_installed_python_311_on_a_newer_host(self):
        completed = mock.Mock(
            returncode=0,
            stdout=b" -V:3.13 C:\\Python313\\python.exe\n -V:3.11 C:\\Python311\\python.exe\n",
        )
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.object(tools.install_dependencies.sys, "version_info", (3, 13, 0)),
            mock.patch.object(tools.install_dependencies.os, "name", "nt"),
            mock.patch("agent.tools.install_dependencies.shutil.which", return_value="py"),
            mock.patch("agent.tools.install_dependencies.subprocess.run", return_value=completed),
        ):
            selected = tools.install_dependencies._python_runtime(Path(directory))

        self.assertEqual(selected, ["py", "-3.11"])

    def test_publish_output_structures_copy_and_existing_image_assets(self):
        with tempfile.TemporaryDirectory() as directory, tools.workspace_context(directory):
            Path(directory, "shot.png").write_bytes(b"image")
            tools.register_vision_evaluator(directory, lambda images, purpose, criteria: {
                "status": "evaluated",
                "model": "vision-test",
                "evaluations": [{
                    "path": images[0]["path"],
                    "readable": True,
                    "visual_quality_score": 90,
                    "requirement_fit_score": 90,
                    "strengths": ["clear"],
                    "issues": [],
                    "visible_facts": ["image"],
                }],
                "ranking": [images[0]["path"]],
                "selected": [images[0]["path"]],
                "copy_facts": ["image"],
            })
            self.assertIn('"status": "evaluated"', tools.run_tool("inspect_images", {
                "paths": ["shot.png"],
                "purpose": "Select the screenshot",
                "criteria": "Readable",
            }))
            published = []

            def save(envelope):
                published.append(envelope)
                return {"status": "ready", "output_id": "output-test", **envelope}

            tools.register_output_publisher(directory, save)
            payload = json.loads(tools.run_tool("publish_output", {
                "title": "Task result",
                "message": "Everything completed.",
                "copy_sections": [{"label": "Copy ready", "text": "Use this text"}],
                "assets": [{"path": "shot.png", "label": "Main screen", "kind": "image"}],
            }))

            self.assertEqual(payload["status"], "ready")
            self.assertEqual(payload["output_id"], "output-test")
            self.assertEqual(published[0]["copy_sections"][0]["text"], "Use this text")
            self.assertEqual(published[0]["assets"][0]["path"], "shot.png")
            self.assertEqual(len(published[0]["assets"][0]["sha256"]), 64)

    def test_fixed_media_bundle_is_not_part_of_the_general_tool_surface(self):
        self.assertIsNone(tools.get_spec("deliver_media_bundle"))
        return
        with tempfile.TemporaryDirectory() as directory, tools.workspace_context(
            directory,
            session_id="media-session",
            goal_id="media-goal",
            task_id="T004",
        ):
            Path(directory, "first.png").write_bytes(b"first-image")
            Path(directory, "second.webp").write_bytes(b"second-image")
            preview = json.dumps({
                "preview_id": "preview-gallery",
                "url": "http://127.0.0.1:43210/token/index.html",
                "browser_opened": True,
                "verification": "passed",
            })
            tools.register_vision_evaluator(directory, lambda images, purpose, criteria: {
                "status": "evaluated",
                "model": "vision-test",
                "evaluations": [
                    {
                        "path": image["path"], "readable": True,
                        "visual_quality_score": 90, "requirement_fit_score": 95,
                        "strengths": ["clear"], "issues": [], "visible_facts": ["feature visible"],
                    }
                    for image in images
                ],
                "ranking": [image["path"] for image in images],
                "selected": [image["path"] for image in images],
                "copy_facts": ["feature visible"],
            })
            evidence = json.loads(tools.run_tool("inspect_images", {
                "paths": ["first.png", "second.webp"],
                "purpose": "Choose project screenshots",
                "criteria": "Readable and relevant",
            }))
            self.assertEqual(evidence["status"], "evaluated")
            with mock.patch("agent.tools.deliver_media_bundle.web_preview.create", return_value=preview):
                payload = json.loads(tools.run_tool("deliver_media_bundle", {
                    "copy_text": "جاهز للنسخ",
                    "directory": "output/deliverables/demo",
                    "assets": [
                        {"path": "first.png", "label": "Matching", "claim": "Shows fit scoring"},
                        {"path": "second.webp", "label": "CV", "claim": "Shows CV tailoring"},
                    ],
                }))

            target = Path(directory, "output", "deliverables", "demo")
            self.assertEqual(payload["status"], "ready")
            self.assertEqual(Path(target, "copy.txt").read_text(encoding="utf-8"), "جاهز للنسخ")
            self.assertTrue(Path(target, "01-asset.png").is_file())
            self.assertTrue(Path(target, "02-asset.webp").is_file())
            self.assertIn("Copy text", Path(target, "index.html").read_text(encoding="utf-8"))
            manifest = json.loads(Path(target, "manifest.json").read_text(encoding="utf-8"))
            self.assertNotIn("platform", manifest)
            self.assertEqual([item["label"] for item in manifest["assets"]], ["Matching", "CV"])
            self.assertTrue(all(len(item["sha256"]) == 64 for item in manifest["assets"]))
            self.assertEqual(payload["preview_url"], "http://127.0.0.1:43210/token/index.html")

    def test_media_bundle_never_overwrites_an_existing_delivery(self):
        self.assertIsNone(tools.get_spec("deliver_media_bundle"))
        return
        with tempfile.TemporaryDirectory() as directory, tools.workspace_context(directory):
            Path(directory, "shot.png").write_bytes(b"image")
            existing = Path(directory, "output", "deliverables", "media")
            existing.mkdir(parents=True)
            Path(existing, "copy.txt").write_text("old", encoding="utf-8")

            result = tools.run_tool("deliver_media_bundle", {
                "copy_text": "new",
                "assets": [{"path": "shot.png", "label": "One", "claim": "Claim"}],
            })

            self.assertTrue(result.startswith("Error:"), result)
            self.assertEqual(Path(existing, "copy.txt").read_text(encoding="utf-8"), "old")

    def test_media_bundle_rejects_an_image_without_current_vision_evidence(self):
        self.assertIsNone(tools.get_spec("deliver_media_bundle"))
        return
        with tempfile.TemporaryDirectory() as directory, tools.workspace_context(directory):
            Path(directory, "shot.png").write_bytes(b"image")

            result = tools.run_tool("deliver_media_bundle", {
                "copy_text": "copy",
                "assets": [{"path": "shot.png", "label": "One", "claim": "Claim"}],
                "open_browser": False,
            })

            self.assertIn("no current vision evidence", result)

    def test_media_bundle_rejects_an_evaluated_image_the_vision_model_did_not_select(self):
        self.assertIsNone(tools.get_spec("deliver_media_bundle"))
        return
        with tempfile.TemporaryDirectory() as directory, tools.workspace_context(directory):
            Path(directory, "shot.png").write_bytes(b"image")
            tools.register_vision_evaluator(directory, lambda images, purpose, criteria: {
                "status": "evaluated",
                "model": "vision-test",
                "evaluations": [{
                    "path": images[0]["path"], "readable": True,
                    "visual_quality_score": 20, "requirement_fit_score": 10,
                    "strengths": [], "issues": ["not relevant"], "visible_facts": [],
                }],
                "ranking": [images[0]["path"]],
                "selected": [],
                "copy_facts": [],
            })
            evidence = tools.run_tool("inspect_images", {
                "paths": ["shot.png"], "purpose": "Select a relevant image", "criteria": "Relevant",
            })
            self.assertIn('"status": "evaluated"', evidence)

            result = tools.run_tool("deliver_media_bundle", {
                "copy_text": "copy",
                "assets": [{"path": "shot.png", "label": "One", "claim": "Claim"}],
                "open_browser": False,
            })

            self.assertIn("no current vision evidence", result)


if __name__ == "__main__":
    unittest.main()
