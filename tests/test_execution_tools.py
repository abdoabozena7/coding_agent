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
        with tempfile.TemporaryDirectory() as directory, tools.workspace_context(directory):
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
