"""Regressions for executable ordinary Chat and durable generated artifacts."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

from agent.runtime import AgentRuntime
from agent.chat_runtime import SemanticTurnDecisionV2
from agent import tools
from agent.events import EventBus
from agent.store import StateStore
from agent.testing import ScriptedProvider, semantic_turn


class ChatExecutionTests(unittest.TestCase):
    def test_completed_action_text_drops_stale_confirmation_request(self):
        draft = (
            "I captured three screenshots. To complete the final handoff, I need to visually evaluate them.\n\n"
            "Please confirm if you would like me to proceed with calling inspect_images."
        )
        cleaned = AgentRuntime._sanitize_completed_action_text(draft)
        self.assertEqual(
            cleaned,
            (
                "The requested task completed successfully. The verified result and "
                "attachments are included with this response."
            ),
        )

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

    def bounded_action(self, runtime, prompt: str, *, effects):
        """Exercise the legacy bounded executor below the recursive router."""
        raw = semantic_turn("action", original=prompt, effects=effects)
        decision = SemanticTurnDecisionV2.from_mapping(
            raw["tool_calls"][0]["args"],
            original_input=prompt,
            parse_goal_intake=False,
        )
        turn_id = "test-bounded-action"
        runtime._save_pending_semantic_turn(
            {
                "turn_id": turn_id,
                "original_input": prompt,
                "requested_mode": "normal",
                "interaction_mode": "working",
                "minimum_strategy": "recursive",
                "status": "dispatching",
                "stage": "action",
                "action_records": [],
            }
        )
        result = runtime.chat(
            prompt,
            _route_checked=True,
            semantic_decision=decision,
            semantic_turn_id=turn_id,
        )
        if result.status == "action_incomplete":
            runtime._hold_semantic_turn(
                turn_id,
                result_status=result.status,
                reason=result.reason or result.message,
                limitations=result.limitations,
            )
        else:
            runtime._complete_semantic_turn(turn_id, result_status=result.status)
        return result

    def test_missing_screenshot_delivery_is_not_reported_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            turns = [
                {"tool_calls": [{"id": "list-1", "name": "list_files", "args": {}}]},
                "Done.",
                {"tool_calls": [{"id": "list-2", "name": "list_files", "args": {}}]},
                "Done.",
                {"tool_calls": [{"id": "list-3", "name": "list_files", "args": {}}]},
                "Done.",
            ]
            with self.runtime(directory, turns) as (runtime, store, _provider):
                result = self.bounded_action(
                    runtime,
                    "Run this web project, open it, capture three distinct screenshots, evaluate them, and write a post.",
                    effects=("run", "preview"),
                )

                self.assertEqual(result.status, "action_incomplete")
                self.assertFalse(result.completed)
                self.assertTrue(
                    any("capture 3 distinct browser screenshots" in item for item in result.limitations)
                )
                self.assertIn("No missing screenshot", result.message)
                session = store.get_workflow_session(runtime.session_id)
                self.assertEqual(session["run_state"], "blocked")
                self.assertEqual(
                    session["state"]["pending_semantic_turn"]["status"],
                    "needs_evidence",
                )
                self.assertEqual(runtime.workflow_runtime_snapshot().phase, "paused")
                self.assertTrue(runtime.prepare_automatic_semantic_retry())
                self.assertFalse(runtime.prepare_automatic_semantic_retry())

    def test_restart_reconciles_interrupted_ephemeral_process_without_user_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            prompt = "Run this web project and open it."
            raw = semantic_turn(
                "action", original=prompt, effects=("read", "run", "preview")
            )
            decision = SemanticTurnDecisionV2.from_mapping(
                raw["tool_calls"][0]["args"],
                original_input=prompt,
                parse_goal_intake=False,
            )
            first_store = StateStore(directory)
            first = AgentRuntime(
                ScriptedProvider([]), first_store, directory,
                session_id="ephemeral-recovery",
            )
            first.set_session_tool_permissions(True)
            action_id = first_store.begin_session_action(
                first.session_id,
                "start_process",
                {
                    "command": "npm run dev",
                    "cwd": ".",
                    "readiness_type": "url",
                    "readiness_value": "http://localhost:3000",
                },
                risk="critical",
                mutating=True,
            )
            browser_action_id = first_store.begin_session_action(
                first.session_id,
                "browser_open",
                {"url": "http://localhost:3000", "visible": True},
                risk="high",
                mutating=False,
            )
            dependency_action_id = first_store.begin_session_action(
                first.session_id,
                "install_dependencies",
                {"directory": ".", "manager": "npm"},
                risk="critical",
                mutating=True,
            )
            first._save_pending_semantic_turn({
                "turn_id": "ephemeral-turn",
                "original_input": prompt,
                "requested_mode": "normal",
                "status": "needs_evidence",
                "stage": "dispatching",
                "route_decision": decision.to_dict(),
                "decision": decision.to_dict(),
                "action_records": [{
                    "action_id": action_id,
                    "tool_name": "start_process",
                    "category": "process",
                    "mutating": True,
                    "status": "running",
                    "args": {
                        "command": "npm run dev",
                        "cwd": ".",
                        "readiness_type": "url",
                        "readiness_value": "http://localhost:3000",
                    },
                }, {
                    "action_id": browser_action_id,
                    "tool_name": "browser_open",
                    "category": "preview",
                    "mutating": False,
                    "status": "running",
                    "args": {"url": "http://localhost:3000", "visible": True},
                }, {
                    "action_id": dependency_action_id,
                    "tool_name": "install_dependencies",
                    "category": "install",
                    "mutating": True,
                    "status": "running",
                    "args": {"directory": ".", "manager": "npm"},
                }],
            })
            first.close(); first_store.close()

            reopened_store = StateStore(directory)
            reopened = AgentRuntime(
                ScriptedProvider(["Done.", "Done.", "Done."]),
                reopened_store,
                directory,
                session_id="ephemeral-recovery",
            )
            try:
                result = reopened._resume_pending_semantic_turn()

                self.assertNotEqual(result.status, "uncertain")
                action = next(
                    item for item in reopened_store.list_session_actions(reopened.session_id)
                    if item["id"] == action_id
                )
                self.assertEqual(action["status"], "failed")
                browser_action = next(
                    item for item in reopened_store.list_session_actions(reopened.session_id)
                    if item["id"] == browser_action_id
                )
                self.assertEqual(browser_action["status"], "failed")
                dependency_action = next(
                    item for item in reopened_store.list_session_actions(reopened.session_id)
                    if item["id"] == dependency_action_id
                )
                self.assertEqual(dependency_action["status"], "failed")
                self.assertTrue(any(
                    event.event_type == "semantic_action.ephemeral_reconciled"
                    for event in reopened_store.list_recent_events(limit=100)
                ))
            finally:
                reopened.close(); reopened_store.close()

    def test_restart_reopens_legacy_false_completed_media_action(self):
        with tempfile.TemporaryDirectory() as directory:
            prompt = "Run this web project, open it, and capture three screenshots."
            raw = semantic_turn("action", original=prompt, effects=("run", "preview"))
            decision = SemanticTurnDecisionV2.from_mapping(
                raw["tool_calls"][0]["args"],
                original_input=prompt,
                parse_goal_intake=False,
            )
            store = StateStore(directory)
            runtime = AgentRuntime(ScriptedProvider([]), store, directory)
            session_id = runtime.session_id
            runtime.store.mutate_workflow_session(
                session_id,
                lambda current: {
                    "state": {
                        **dict(current.get("state") or {}),
                        "route": "action",
                        "last_semantic_turn": {
                            "turn_id": "legacy-false-complete",
                            "original_input": prompt,
                            "route_decision": decision.to_dict(),
                            "status": "completed",
                            "result_status": "chat",
                            "action_records": [],
                        },
                    },
                    "run_state": "idle",
                },
            )
            runtime.close()
            store.close()

            reopened_store = StateStore(directory)
            reopened = AgentRuntime(ScriptedProvider([]), reopened_store, directory)
            try:
                session = reopened_store.get_workflow_session(session_id)
                pending = session["state"]["pending_semantic_turn"]
                self.assertEqual(pending["status"], "needs_evidence")
                self.assertEqual(pending["result_status"], "action_incomplete")
                self.assertEqual(session["run_state"], "blocked")
                self.assertEqual(reopened.workflow_runtime_snapshot().phase, "paused")
            finally:
                reopened.close()
                reopened_store.close()

    def test_action_completes_only_after_browser_vision_and_output_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            image_paths = []
            for index in range(1, 4):
                path = Path(directory, f"shot-{index}.png")
                path.write_bytes(b"png-" + bytes([index]))
                image_paths.append(str(path))
            turns = [
                {"tool_calls": [{"id": "start", "name": "start_process", "args": {
                    "command": "streamlit run app.py", "readiness_type": "port", "readiness_value": "8501",
                }}]},
                {"tool_calls": [{"id": "open", "name": "browser_open", "args": {
                    "url": "http://127.0.0.1:8501", "visible": True,
                }}]},
                {"tool_calls": [{"id": "inspect", "name": "browser_inspect", "args": {
                    "browser_session_id": "browser-media",
                }}]},
            ]
            turns.extend([
                {"tool_calls": [{"id": f"shot-{index}", "name": "browser_screenshot", "args": {
                    "browser_session_id": "browser-media", "name": f"feature-{index}",
                }}]}
                for index in range(1, 4)
            ])
            turns.extend([
                {"tool_calls": [{"id": "vision", "name": "inspect_images", "args": {
                    "paths": image_paths, "purpose": "select the strongest project views", "criteria": "distinct and clear",
                }}]},
                {"tool_calls": [{"id": "output", "name": "publish_output", "args": {
                    "title": "Project result",
                    "message": "Everything is ready.",
                    "copy_sections": [{"label": "Copy ready", "text": "Ready-to-copy project post"}],
                    "assets": [
                        {"path": path, "label": f"Feature {index}", "kind": "image"}
                        for index, path in enumerate(image_paths, start=1)
                    ],
                }}]},
                "Everything is ready.",
            ])
            with self.runtime(directory, turns) as (runtime, _store, _provider):
                shot_index = {f"feature-{index}": path for index, path in enumerate(image_paths, start=1)}

                def run_tool(name, args):
                    if name == "start_process":
                        payload = {"process_id": "process-media", "pid": 321, "status": "running", "ready": True}
                    elif name == "browser_open":
                        payload = {
                            "browser_session_id": "browser-media", "status": "running",
                            "browser_opened": True, "url": "http://127.0.0.1:8501",
                            "title": "Project", "http_status": 200,
                        }
                    elif name == "browser_inspect":
                        payload = {
                            "browser_session_id": "browser-media", "status": "running",
                            "browser_opened": True, "url": "http://127.0.0.1:8501",
                            "title": "Project", "http_status": 200,
                            "interaction_targets": [],
                        }
                    elif name == "browser_screenshot":
                        path = shot_index[args["name"]]
                        payload = {
                            "browser_session_id": "browser-media", "status": "running",
                            "browser_opened": True, "url": "http://127.0.0.1:8501",
                            "screenshot_path": path, "workspace_path": Path(path).name,
                            "sha256": str(args["name"]).encode().hex().ljust(64, "0")[:64],
                            "image_width": 1440, "image_height": 900,
                        }
                    elif name == "inspect_images":
                        payload = {
                            "status": "evaluated", "model": "weak-vision",
                            "evaluations": [{"path": path, "readable": True} for path in image_paths],
                            "selected": image_paths,
                        }
                    elif name == "publish_output":
                        payload = {
                            "status": "ready", "output_id": "output-media", "title": "Project result",
                            "copy_sections": [{"label": "Copy ready", "text": "Ready-to-copy project post"}],
                            "assets": [{"path": path, "kind": "image"} for path in image_paths],
                        }
                    else:
                        raise AssertionError(name)
                    return tools.ToolExecutionResult(True, json.dumps(payload))

                with mock.patch("agent.runtime.tools.run_tool_detailed", side_effect=run_tool):
                    result = self.bounded_action(
                        runtime,
                        "Run this web project, open it, capture three distinct screenshots, evaluate them, and write a post.",
                        effects=("run", "preview"),
                    )

                self.assertEqual(result.status, "action_completed")
                self.assertTrue(result.completed)
                self.assertIn("Images attached: 3", result.message)
                self.assertIn("Copy-ready sections: 1", result.message)
                self.assertEqual(runtime.workflow_runtime_snapshot().phase, "completed")

    def test_action_stops_after_first_conclusive_vision_capability_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            image_paths = []
            for index in range(1, 4):
                path = Path(directory, f"vision-blocked-{index}.png")
                path.write_bytes(b"png-" + bytes([index]))
                image_paths.append(str(path))
            turns = [
                {"tool_calls": [{"id": "start", "name": "start_process", "args": {
                    "command": "streamlit run app.py", "readiness_type": "port", "readiness_value": "8501",
                }}]},
                {"tool_calls": [{"id": "open", "name": "browser_open", "args": {
                    "url": "http://127.0.0.1:8501", "visible": True,
                }}]},
                {"tool_calls": [{"id": "inspect", "name": "browser_inspect", "args": {
                    "browser_session_id": "browser-media",
                }}]},
            ]
            turns.extend([
                {"tool_calls": [{"id": f"shot-{index}", "name": "browser_screenshot", "args": {
                    "browser_session_id": "browser-media", "name": f"feature-{index}",
                }}]}
                for index in range(1, 4)
            ])
            turns.extend([
                {"tool_calls": [{"id": "vision-1", "name": "inspect_images", "args": {
                    "paths": image_paths, "purpose": "check the views", "criteria": "distinct and readable",
                }}]},
                {"tool_calls": [{"id": "vision-2", "name": "inspect_images", "args": {
                    "paths": image_paths, "purpose": "retry", "criteria": "retry",
                }}]},
            ])
            with self.runtime(directory, turns) as (runtime, _store, provider):
                shot_index = {
                    f"feature-{index}": path
                    for index, path in enumerate(image_paths, start=1)
                }
                vision_calls = 0

                def run_tool(name, args):
                    nonlocal vision_calls
                    if name == "start_process":
                        payload = {"process_id": "process-media", "status": "running", "ready": True}
                    elif name == "browser_open":
                        payload = {
                            "browser_session_id": "browser-media", "status": "running",
                            "browser_opened": True, "url": "http://127.0.0.1:8501", "http_status": 200,
                        }
                    elif name == "browser_inspect":
                        payload = {
                            "browser_session_id": "browser-media", "status": "running",
                            "browser_opened": True, "url": "http://127.0.0.1:8501", "http_status": 200,
                            "interaction_targets": [],
                        }
                    elif name == "browser_screenshot":
                        path = shot_index[args["name"]]
                        payload = {
                            "browser_session_id": "browser-media", "status": "running",
                            "browser_opened": True, "screenshot_path": path,
                            "workspace_path": Path(path).name,
                            "sha256": str(args["name"]).encode().hex().ljust(64, "0")[:64],
                        }
                    elif name == "inspect_images":
                        vision_calls += 1
                        return tools.ToolExecutionResult(
                            False,
                            "Error: vision evaluation unavailable: the model could not read the pixel-only OCR probe",
                        )
                    else:
                        raise AssertionError(name)
                    return tools.ToolExecutionResult(True, json.dumps(payload))

                with mock.patch("agent.runtime.tools.run_tool_detailed", side_effect=run_tool):
                    result = self.bounded_action(
                        runtime,
                        "Run this web project, capture three distinct screenshots, and evaluate them visually.",
                        effects=("run", "preview"),
                    )

                self.assertEqual(result.status, "action_incomplete")
                self.assertEqual(vision_calls, 1)
                self.assertEqual(provider.remaining, 1)
                self.assertIn("paused at the visual evidence gate", result.message)
                self.assertIn("Output was not published", result.message)

    def test_failed_vision_probe_is_cached_for_the_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.runtime(directory, ["not json"]) as (runtime, _store, provider):
                first = runtime._verify_vision_capability()
                second = runtime._verify_vision_capability()

                self.assertEqual(first, second)
                self.assertFalse(first[0])
                self.assertEqual(len(provider.calls), 1)

    def test_normal_access_never_installs_a_visual_model_implicitly(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.runtime(directory, []) as (runtime, _store, _provider):
                with (
                    mock.patch("agent.runtime.installed_vision_models", return_value=()),
                    mock.patch("agent.runtime.pull_ollama_vision_model") as pulled,
                ):
                    evaluator, reason = runtime._resolve_vision_provider()

                self.assertIsNone(evaluator)
                self.assertIn("vision", reason)
                pulled.assert_not_called()

    def test_session_wide_authority_installs_and_probes_visual_fallback_once(self):
        from agent.providers.ollama_provider import OllamaProvider

        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(directory)
            primary = OllamaProvider(model="offline")
            runtime = AgentRuntime(primary, store, directory)
            runtime.set_session_tool_permissions(True)
            fallback = ScriptedProvider(
                [{"text": '{"token":"VISION-731"}'}],
                model="vision-local",
            )
            fallback.capabilities = SimpleNamespace(supports_vision=True)
            descriptor = SimpleNamespace(
                model="vision-local",
                create_provider=lambda: fallback,
            )
            try:
                with (
                    mock.patch("agent.runtime.installed_vision_models", return_value=()),
                    mock.patch(
                        "agent.runtime.pull_ollama_vision_model",
                        return_value=descriptor,
                    ) as pulled,
                ):
                    first, reason = runtime._resolve_vision_provider()
                    second, second_reason = runtime._resolve_vision_provider()

                self.assertIs(first, fallback)
                self.assertIs(second, fallback)
                self.assertEqual(reason, "")
                self.assertEqual(second_reason, "")
                pulled.assert_called_once()
                self.assertEqual(len(fallback.calls), 1)
            finally:
                runtime.close()
                store.close()
        return
        with tempfile.TemporaryDirectory() as directory:
            image_paths = []
            for index in range(1, 4):
                path = Path(directory, f"shot-{index}.png")
                path.write_bytes(b"png-" + bytes([index]))
                image_paths.append(str(path))
            preview = json.dumps({
                "status": "running",
                "preview_id": "preview-media",
                "url": "http://127.0.0.1:8501",
                "http_status": 200,
                "browser_opened": True,
                "verification": "passed",
                "screenshot_path": image_paths[0],
                "interaction_results": [
                    {"name": "feature two", "passed": True, "screenshot_path": image_paths[1]},
                    {"name": "feature three", "passed": True, "screenshot_path": image_paths[2]},
                ],
                "console_errors": [], "page_errors": [], "network_errors": [],
            })
            inspected = json.dumps({
                "status": "evaluated",
                "model": "weak-vision",
                "evaluations": [{"path": path, "score": 8} for path in image_paths],
                "selected": image_paths,
            })
            delivered = json.dumps({
                "status": "ready",
                "directory": "output/deliverables/media-1",
                "copy_path": "output/deliverables/media-1/copy.txt",
                "gallery_path": "output/deliverables/media-1/index.html",
                "manifest_path": "output/deliverables/media-1/manifest.json",
                "assets": [
                    {"file": f"{index:02d}-asset.png", "source": path}
                    for index, path in enumerate(image_paths, start=1)
                ],
                "browser_opened": True,
                "verification": "passed",
            })
            receipts = {
                "start_process": json.dumps({
                    "process_id": "process-media", "pid": 321,
                    "status": "running", "ready": True,
                }),
                "preview_url": preview,
                "inspect_images": inspected,
                "deliver_media_bundle": delivered,
            }
            turns = [
                {"tool_calls": [{"id": "start", "name": "start_process", "args": {
                    "command": "streamlit run app.py", "readiness_type": "port", "readiness_value": "8501",
                }}]},
                {"tool_calls": [{"id": "preview", "name": "preview_url", "args": {
                    "url": "http://127.0.0.1:8501", "interactions": [],
                }}]},
                {"tool_calls": [{"id": "vision", "name": "inspect_images", "args": {
                    "paths": image_paths, "purpose": "select the strongest project views", "criteria": "distinct and clear",
                }}]},
                {"tool_calls": [{"id": "deliver", "name": "deliver_media_bundle", "args": {
                    "copy_text": "Ready-to-copy project post", "assets": [
                        {"path": path, "label": f"Feature {index}", "claim": "Verified feature"}
                        for index, path in enumerate(image_paths, start=1)
                    ], "directory": "output/deliverables/media-1", "open_browser": True,
                }}]},
                "Everything is ready.",
            ]
            with self.runtime(directory, turns) as (runtime, _store, _provider):
                def run_tool(name, _args):
                    return tools.ToolExecutionResult(True, receipts[name])

                with mock.patch("agent.runtime.tools.run_tool_detailed", side_effect=run_tool):
                    result = self.bounded_action(
                        runtime,
                        "Run this web project, open it, capture three distinct screenshots, evaluate them, and write a post.",
                        effects=("run", "preview"),
                    )

                self.assertEqual(result.status, "action_completed")
                self.assertTrue(result.completed)
                self.assertIn("Screenshots delivered: 3", result.message)
                self.assertIn("copy.txt", result.message)
                self.assertEqual(runtime.workflow_runtime_snapshot().phase, "completed")

    def test_background_thread_tool_execution_has_workspace_context(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.runtime(directory, [
                {"tool_calls": [{"id": "write", "name": "write_file", "args": {
                    "path": "index.html", "content": "<!doctype html><title>ok</title>",
                }}]},
                "Done.",
            ]) as (runtime, store, _provider):
                results = []
                thread = threading.Thread(target=lambda: results.append(
                    self.bounded_action(runtime, "save it to index.html", effects=("write",))
                ))
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
                    result = self.bounded_action(
                        runtime, "run index.html", effects=("preview",)
                    )

                preview.assert_called_once()
                self.assertEqual(len(provider.calls), 3)
                self.assertIn("http://127.0.0.1:43210", result.message)
                self.assertIn("verification passed", result.message)
                self.assertNotIn("Open it yourself", result.message)

    def test_advertised_native_tool_json_text_is_normalized_and_executed(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.runtime(directory, [
                '{"name":"write_file","arguments":{"path":"hello.txt","content":"hello"}}',
                "Done.",
            ]) as (runtime, store, provider):
                provider.capability_profile = type("Capabilities", (), {"tool_call_support": True})()
                result = self.bounded_action(
                    runtime, "create hello.txt", effects=("write",)
                )

                self.assertEqual(Path(directory, "hello.txt").read_text(encoding="utf-8"), "hello")
                self.assertIn("write_file", result.message)
                self.assertEqual(len(provider.calls), 2)
                self.assertEqual(store.list_session_actions(runtime.session_id)[0]["status"], "completed")

    def test_native_write_repairs_double_escaped_document_layout(self):
        escaped_html = (
            '<!DOCTYPE html>\\n<html>\\n<body>\\n<script>'
            'const label = "line\\nvalue";'
            '</script>\\n</body>\\n</html>'
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.runtime(directory, [
                {"tool_calls": [{"id": "write", "name": "write_file", "args": {
                    "path": "index.html", "content": escaped_html,
                }}]},
                "Done.",
            ]) as (runtime, _store, _provider):
                self.bounded_action(runtime, "create index.html", effects=("write",))

                written = Path(directory, "index.html").read_text(encoding="utf-8")
                self.assertIn("<!DOCTYPE html>\n<html>\n<body>", written)
                self.assertIn('"line\\nvalue"', written)
                self.assertNotIn(r"<!DOCTYPE html>\n<html>", written)

    def test_failed_write_is_not_mutation_or_artifact_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.runtime(directory, [
                {"tool_calls": [{"id": "bad", "name": "write_file", "args": {
                    "path": "../escape.txt", "content": "bad",
                }}]},
                "Done.", "Done.", "Done.",
            ]) as (runtime, store, _provider):
                result = self.bounded_action(runtime, "save it", effects=("write",))

                self.assertNotIn("BELOW_TARGET", result.message)
                action = store.list_session_actions(runtime.session_id)[0]
                self.assertEqual(action["status"], "failed")
                self.assertEqual(action["changed_paths"], [])
                self.assertFalse(Path(directory).parent.joinpath("escape.txt").exists())

    def test_large_generated_html_survives_restart_and_materializes_exactly(self):
        with tempfile.TemporaryDirectory() as directory:
            code = "<!doctype html><html><title>large</title><body>" + ("x" * 4_000) + "</body></html>\n"
            store = StateStore(directory)
            first = AgentRuntime(ScriptedProvider([
                semantic_turn("chat", original="show me the generated HTML", response=f"```html\n{code}```")
            ]), store, directory, approval=lambda *_: True)
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
                result = self.bounded_action(
                    second, "save it to index.html", effects=("write",)
                )
                self.assertEqual(Path(directory, "index.html").read_text(encoding="utf-8"), code)
                self.assertIn(artifact["id"], result.message)
                self.assertTrue(any("CHAT_ARTIFACT" in str(item.get("content")) for item in second._chat_conversation))
            finally:
                second.close(); store.close()

    def test_generated_html_save_and_run_recovers_from_exact_manual_refusal(self):
        with tempfile.TemporaryDirectory() as directory:
            code = "<!doctype html><html><title>game</title><body>" + ("game" * 800) + "</body></html>\n"
            store = StateStore(directory)
            provider = ScriptedProvider([
                semantic_turn("chat", original="show the generated HTML", response=f"```html\n{code}```")
            ])
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
                    result = self.bounded_action(
                        runtime,
                        "put it in index.html and run it",
                        effects=("write", "preview"),
                    )
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
            with self.runtime(directory, [
                semantic_turn("chat", original="How does an HTML preview run?", response="An HTML preview serves files over loopback.")
            ]) as (runtime, _store, provider):
                result = runtime.chat("How does an HTML preview run?")
                self.assertEqual(len(provider.calls), 1)
                self.assertIn("loopback", result.message)

    def test_chat_final_text_is_returned_once_and_not_streamed_as_a_duplicate(self):
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(directory)
            events = EventBus()
            seen = []
            events.subscribe(lambda event: seen.append(event))
            runtime = AgentRuntime(ScriptedProvider([
                semantic_turn("chat", original="hello", response="ONE UNIQUE RESPONSE")
            ]), store, directory, events=events)
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
                result = self.bounded_action(
                    runtime, "run the command echo ok", effects=("run",)
                )
                self.assertEqual(adapter.calls, [("echo ok", str(Path(directory).resolve()))])
                self.assertIn("exit code: 0", result.message)
            finally:
                runtime.close(); store.close()

    def test_full_mode_chat_can_install_or_start_declared_project_without_prompt(self):
        class Access:
            value = "full"

        class Adapter:
            access_level = Access()

            def requires_approval(self, _normal=True):
                return False

            def run_shell(self, *_args, **_kwargs):
                raise AssertionError("start_process is a managed host action, not a shell call")

        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(directory)
            runtime = AgentRuntime(
                ScriptedProvider([
                    {"tool_calls": [{
                        "id": "start",
                        "name": "start_process",
                        "args": {
                            "command": "npm start",
                            "readiness_type": "log",
                            "readiness_value": "ready",
                        },
                    }]},
                    "The project is running.",
                ]),
                store,
                directory,
                permission_adapter=Adapter(),
                approval=lambda *_: (_ for _ in ()).throw(
                    AssertionError("Full access must not prompt")
                ),
            )
            receipt = json.dumps({
                "process_id": "process-full-access",
                "pid": 321,
                "status": "running",
                "ready": True,
            })
            from agent import tools

            try:
                with mock.patch(
                    "agent.runtime.tools.run_tool_detailed",
                    return_value=tools.ToolExecutionResult(True, receipt),
                ) as executed:
                    result = self.bounded_action(
                        runtime,
                        "run this project",
                        effects=("run",),
                    )

                executed.assert_called_once_with(
                    "start_process",
                    {
                        "command": "npm start",
                        "readiness_type": "log",
                        "readiness_value": "ready",
                    },
                )
                self.assertIn("project is running", result.message.casefold())
            finally:
                runtime.close()
                store.close()

    def test_full_access_browser_open_runs_without_an_approval_question(self):
        class Access:
            value = "full"

        class Adapter:
            access_level = Access()

            def requires_approval(self, _normal=True):
                return False

        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(directory)
            runtime = AgentRuntime(
                ScriptedProvider([
                    {"tool_calls": [{"id": "browser", "name": "browser_open", "args": {
                        "url": "http://127.0.0.1:8501", "visible": True,
                    }}]},
                    "The local app is open.",
                ]),
                store,
                directory,
                permission_adapter=Adapter(),
                approval=lambda *_: (_ for _ in ()).throw(
                    AssertionError("Full access must not ask before opening the Playwright browser")
                ),
            )
            receipt = json.dumps({
                "status": "running", "browser_session_id": "browser-full",
                "url": "http://127.0.0.1:8501", "http_status": 200,
                "browser_opened": True,
                "console_errors": [], "page_errors": [], "network_errors": [],
            })
            try:
                with mock.patch(
                    "agent.runtime.tools.run_tool_detailed",
                    return_value=tools.ToolExecutionResult(True, receipt),
                ):
                    result = self.bounded_action(
                        runtime,
                        "Open the running local web app in the browser",
                        effects=("preview",),
                    )

                self.assertEqual(result.status, "action_completed")
                self.assertTrue(result.completed)
            finally:
                runtime.close()
                store.close()

    def test_weak_model_browser_workflow_recovers_from_duplicate_discovery_and_traces_phases(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            workspace.joinpath("README.md").write_text(
                "Run with: streamlit run app.py --server.port 8501",
                encoding="utf-8",
            )
            workspace.joinpath("requirements.txt").write_text("streamlit\n", encoding="utf-8")
            workspace.joinpath("app.py").write_text("print('app')\n", encoding="utf-8")
            image_paths = []
            for index in range(1, 4):
                path = workspace / f"shot-{index}.png"
                path.write_bytes(b"png-state-" + bytes([index]))
                image_paths.append(str(path))

            turns = [
                "Please provide the project path or start command.",
                {"tool_calls": [{"id": "list", "name": "list_files", "args": {}}]},
                "",
                {"tool_calls": [{"id": "duplicate", "name": "list_files", "args": {}}]},
                {"tool_calls": [{"id": "read", "name": "read_file", "args": {"path": "README.md"}}]},
                {"tool_calls": [{"id": "start", "name": "start_process", "args": {
                    "command": "streamlit run app.py --server.port 8501",
                    "readiness_type": "port", "readiness_value": "8501",
                }}]},
                {"tool_calls": [{"id": "open", "name": "browser_open", "args": {
                    "url": "http://127.0.0.1:8501", "visible": True,
                }}]},
                {"tool_calls": [{"id": "inspect", "name": "browser_inspect", "args": {
                    "browser_session_id": "browser-recovery",
                }}]},
                {"tool_calls": [
                    {"id": f"shot-{index}", "name": "browser_screenshot", "args": {
                        "browser_session_id": "browser-recovery", "name": f"feature-{index}",
                    }}
                    for index in range(1, 4)
                ]},
                {"tool_calls": [{"id": "vision", "name": "inspect_images", "args": {
                    "paths": image_paths, "purpose": "select feature evidence", "criteria": "distinct and readable",
                }}]},
                "The running project was inspected and three distinct feature screenshots were selected.",
            ]
            with self.runtime(directory, turns) as (runtime, store, provider):
                original_run_tool = tools.run_tool_detailed

                def run_tool(name, args):
                    if name in {"list_files", "read_file"}:
                        return original_run_tool(name, args)
                    if name == "start_process":
                        payload = {
                            "process_id": "process-recovery", "pid": 321,
                            "status": "running", "ready": True,
                            "readiness": {"type": "port", "value": "8501"},
                            "readiness_url": "http://127.0.0.1:8501",
                        }
                    elif name == "browser_open":
                        payload = {
                            "browser_session_id": "browser-recovery", "status": "running",
                            "browser_opened": True, "url": "http://127.0.0.1:8501", "http_status": 200,
                        }
                    elif name == "browser_inspect":
                        payload = {
                            "browser_session_id": "browser-recovery", "status": "running",
                            "browser_opened": True, "url": "http://127.0.0.1:8501",
                            "interaction_targets": [{"role": "button", "name": "Scan file"}],
                        }
                    elif name == "browser_screenshot":
                        index = int(str(args["name"]).rsplit("-", 1)[-1])
                        payload = {
                            "browser_session_id": "browser-recovery", "status": "running",
                            "browser_opened": True, "screenshot_path": image_paths[index - 1],
                            "workspace_path": Path(image_paths[index - 1]).name,
                            "sha256": f"{index:064x}", "image_width": 1440, "image_height": 900,
                        }
                    elif name == "inspect_images":
                        payload = {
                            "status": "evaluated", "model": "weak-vision",
                            "evaluations": [{"path": path, "readable": True} for path in image_paths],
                            "selected": image_paths,
                        }
                    else:
                        raise AssertionError(name)
                    return tools.ToolExecutionResult(True, json.dumps(payload))

                with mock.patch("agent.runtime.tools.run_tool_detailed", side_effect=run_tool):
                    result = self.bounded_action(
                        runtime,
                        "Run this web project, open it, capture three distinct screenshots, evaluate them visually, and show them.",
                        effects=("read", "run", "preview"),
                    )

                self.assertEqual(result.status, "action_completed")
                actions = [item["tool_name"] for item in store.list_session_actions(runtime.session_id)]
                self.assertEqual(actions.count("list_files"), 1)
                self.assertEqual(
                    actions,
                    [
                        "list_files", "read_file", "start_process", "browser_open", "browser_inspect",
                        "browser_screenshot", "browser_screenshot", "browser_screenshot", "inspect_images",
                    ],
                )
                phases = [
                    str(event.payload.get("workflow_phase") or "")
                    for event in store.list_events()
                    if event.event_type == "action.workflow_phase"
                ]
                self.assertEqual(
                    phases,
                    [
                        "discover", "inspect_startup", "start_runtime", "open_browser",
                        "inspect_browser", "capture", "visual_review", "publish_output",
                    ],
                )
                self.assertTrue(all(len(call.system) < 30_000 for call in provider.calls))


if __name__ == "__main__":
    unittest.main()
