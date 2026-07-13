"""Regressions for executable ordinary Chat and durable generated artifacts."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

from agent.runtime import AgentRuntime
from agent.events import EventBus
from agent.store import StateStore
from agent.testing import ScriptedProvider


class ChatExecutionTests(unittest.TestCase):
    @contextmanager
    def runtime(self, directory: str, turns):
        store = StateStore(directory)
        provider = ScriptedProvider(turns)
        runtime = AgentRuntime(provider, store, directory, approval=lambda *_: True)
        try:
            yield runtime, store, provider
        finally:
            runtime.close()
            store.close()

    def test_background_thread_tool_execution_has_workspace_context(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.runtime(directory, [
                {"tool_calls": [{"id": "write", "name": "write_file", "args": {
                    "path": "index.html", "content": "<!doctype html><title>ok</title>",
                }}]},
                "Done.",
            ]) as (runtime, store, _provider):
                results = []
                thread = threading.Thread(target=lambda: results.append(runtime.chat("save it to index.html")))
                thread.start(); thread.join(timeout=10)

                self.assertFalse(thread.is_alive())
                self.assertTrue((Path(directory) / "index.html").exists())
                self.assertIn("write_file", results[0].message)
                action = store.list_session_actions(runtime.session_id)[0]
                self.assertEqual(action["status"], "completed")
                self.assertEqual(action["changed_paths"], ["index.html"])

    def test_toolless_run_refusal_is_reprompted_and_previewed(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "index.html").write_text("<!doctype html><title>ok</title>", encoding="utf-8")
            with self.runtime(directory, [
                "I am text-based and cannot launch a browser. Open it yourself.",
                {"tool_calls": [{"id": "preview", "name": "preview_html", "args": {"path": "index.html"}}]},
                "Done.",
            ]) as (runtime, _store, provider):
                payload = json.dumps({
                "status": "running", "preview_id": "preview-test",
                "url": "http://127.0.0.1:43210/token/index.html", "http_status": 200,
                "browser_opened": True, "verification": "passed",
                "console_errors": [], "page_errors": [], "network_errors": [],
                })
                with mock.patch("agent.tools.web_preview.create", return_value=payload) as preview:
                    result = runtime.chat("run index.html")

                preview.assert_called_once()
                self.assertEqual(len(provider.calls), 3)
                self.assertIn("http://127.0.0.1:43210", result.message)
                self.assertIn("verification passed", result.message)
                self.assertNotIn("Open it yourself", result.message)

    def test_failed_write_is_not_mutation_or_artifact_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.runtime(directory, [
                {"tool_calls": [{"id": "bad", "name": "write_file", "args": {
                    "path": "../escape.txt", "content": "bad",
                }}]},
                "Done.", "Done.", "Done.",
            ]) as (runtime, store, _provider):
                result = runtime.chat("save it")

                self.assertNotIn("BELOW_TARGET", result.message)
                action = store.list_session_actions(runtime.session_id)[0]
                self.assertEqual(action["status"], "failed")
                self.assertEqual(action["changed_paths"], [])
                self.assertFalse(Path(directory).parent.joinpath("escape.txt").exists())

    def test_large_generated_html_survives_restart_and_materializes_exactly(self):
        with tempfile.TemporaryDirectory() as directory:
            code = "<!doctype html><html><title>large</title><body>" + ("x" * 4_000) + "</body></html>\n"
            store = StateStore(directory)
            first = AgentRuntime(ScriptedProvider([f"```html\n{code}```"]), store, directory, approval=lambda *_: True)
            first.chat("show me the generated HTML")
            artifact = store.list_chat_artifacts(first.session_id)[0]
            first.close()

            second = AgentRuntime(ScriptedProvider([
                {"tool_calls": [{"id": "save", "name": "materialize_artifact", "args": {
                    "artifact_id": artifact["id"], "path": "index.html",
                    "expected_sha256": artifact["content_hash"],
                }}]},
                "Saved.",
            ]), store, directory, approval=lambda *_: True)
            try:
                result = second.chat("save it to index.html")
                self.assertEqual(Path(directory, "index.html").read_text(encoding="utf-8"), code)
                self.assertIn(artifact["id"], result.message)
                self.assertTrue(any("CHAT_ARTIFACT" in str(item.get("content")) for item in second._chat_conversation))
            finally:
                second.close(); store.close()

    def test_generated_html_save_and_run_recovers_from_exact_manual_refusal(self):
        with tempfile.TemporaryDirectory() as directory:
            code = "<!doctype html><html><title>game</title><body>" + ("game" * 800) + "</body></html>\n"
            store = StateStore(directory)
            provider = ScriptedProvider([f"```html\n{code}```"])
            runtime = AgentRuntime(provider, store, directory, approval=lambda *_: True)
            try:
                runtime.chat("show the generated HTML")
                artifact = store.list_chat_artifacts(runtime.session_id)[0]
                provider._turns.extend([
                    "The code block is the runnable artifact. Save it and open it yourself.",
                    {"tool_calls": [
                        {"id": "save", "name": "materialize_artifact", "args": {
                            "artifact_id": artifact["id"], "path": "index.html",
                            "expected_sha256": artifact["content_hash"],
                        }},
                        {"id": "preview", "name": "preview_html", "args": {"path": "index.html"}},
                    ]},
                    "Done.",
                ])
                preview_payload = json.dumps({
                    "status": "running", "preview_id": "preview-e2e",
                    "url": "http://127.0.0.1:45678/token/index.html", "http_status": 200,
                    "browser_opened": True, "verification": "passed",
                    "console_errors": [], "page_errors": [], "network_errors": [],
                })
                with mock.patch("agent.tools.web_preview.create", return_value=preview_payload):
                    result = runtime.chat("put it in index.html and run it")
                self.assertEqual(Path(directory, "index.html").read_text(encoding="utf-8"), code)
                self.assertIn("http://127.0.0.1:45678", result.message)
                self.assertNotIn("open it yourself", result.message.casefold())
                self.assertEqual(
                    [item["tool_name"] for item in store.list_session_actions(runtime.session_id)],
                    ["materialize_artifact", "preview_html"],
                )
            finally:
                runtime.close(); store.close()

    def test_explanatory_question_does_not_force_a_tool(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.runtime(directory, ["An HTML preview serves files over loopback."]) as (runtime, _store, provider):
                result = runtime.chat("How does an HTML preview run?")
                self.assertEqual(len(provider.calls), 1)
                self.assertIn("loopback", result.message)

    def test_chat_final_text_is_returned_once_and_not_streamed_as_a_duplicate(self):
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(directory)
            events = EventBus()
            seen = []
            events.subscribe(lambda event: seen.append(event))
            runtime = AgentRuntime(ScriptedProvider(["ONE UNIQUE RESPONSE"]), store, directory, events=events)
            try:
                result = runtime.chat("hello")
                self.assertEqual(result.message.count("ONE UNIQUE RESPONSE"), 1)
                self.assertFalse(any(event.kind == "model_text" for event in seen))
            finally:
                runtime.close(); store.close()

    def test_full_mode_chat_shell_routes_through_permission_adapter(self):
        class Access:
            value = "full"
        class Adapter:
            access_level = Access()
            calls = []
            def requires_approval(self, _normal=True): return False
            def run_shell(self, command, workspace, *, normal_runner):
                self.calls.append((command, str(workspace)))
                return "exit code: 0\nstdout:\nok"

        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(directory)
            adapter = Adapter()
            runtime = AgentRuntime(ScriptedProvider([
                {"tool_calls": [{"id": "run", "name": "run_command", "args": {"command": "echo ok"}}]},
                "Done.",
            ]), store, directory, permission_adapter=adapter)
            try:
                result = runtime.chat("run the command echo ok")
                self.assertEqual(adapter.calls, [("echo ok", str(Path(directory).resolve()))])
                self.assertIn("exit code: 0", result.message)
            finally:
                runtime.close(); store.close()


if __name__ == "__main__":
    unittest.main()
