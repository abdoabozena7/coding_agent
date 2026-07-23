from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace
from unittest import mock

from agent.cli import (
    choose_access_level,
    choose_interaction_mode,
    choose_project_protection,
    choose_workspace,
    execute_command,
    interactive_loop,
    main,
)
from agent.commands import CommandKind, CommandParseError, parse_command
from agent.config import InteractionMode, RuntimeConfig, SessionPreferences
from agent.models import GoalStatus
from agent.store import StateStore
from agent.ui import ConsoleUI, DashboardView, WorkspaceRefreshRequested
from agent.ui_state import WorkspaceUIStore


class _TTY(io.StringIO):
    encoding = "utf-8"

    def isatty(self) -> bool:
        return True


class CLITests(unittest.TestCase):
    def test_plan_mode_review_keeps_safe_default_and_never_offers_direct_start(self):
        from agent.cli import _plan_attention

        view = SimpleNamespace(
            goal_id="goal-1",
            plan_revision=2,
            plan_summary="A saved plan",
            tasks=(SimpleNamespace(title="Inspect the TUI"),),
        )
        request = _plan_attention(view, (), plan_only=True, ultra_available=True)

        values = {item.value for item in request.options}
        self.assertEqual(request.default_key, "cancel")
        self.assertIn("normal", values)
        self.assertIn("ultra", values)
        self.assertNotIn("start", values)
        self.assertEqual(sum(item.recommended for item in request.options), 1)

    def test_persistent_controller_contains_last_intake_provider_failure(self):
        import time

        from agent.cli import _persistent_interactive_loop
        from agent.local_provider import (
            ProviderDiagnostic,
            ProviderFailureKind,
            ProviderRequestError,
        )

        session = SimpleNamespace(
            source="intake",
            current={
                "id": "q-last",
                "question": "Use the recommended architecture?",
                "options": (
                    {"label": "Yes", "description": "Continue", "recommended": True},
                ),
            },
        )
        failure = ProviderRequestError(
            ProviderDiagnostic(
                True,
                ProviderFailureKind.MODEL_LOAD_FAILED,
                "parse_stream",
                provider_message="CUDA error: illegal memory access",
                endpoint="http://localhost:11434/api/chat",
            )
        )
        captured = {}

        class FakeApp:
            def __init__(self, store, *, on_input, on_interrupt, on_exit, **_kwargs):
                self.store = store
                self.on_exit = on_exit
                self.overlay_kind = ""
                captured["store"] = store

            def run(self):
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    request = self.store.active_attention()
                    if request is None:
                        time.sleep(0.01)
                        continue
                    if request.kind.value == "recovery":
                        self.store.resolve_attention("keep")
                        self.on_exit()
                        return
                    self.store.resolve_selected_attention()
                    time.sleep(0.01)
                self.on_exit()

            def stop(self):
                return None

            def open_details(self, *_args, **_kwargs):
                return None

            def open_swarm(self, *_args, **_kwargs):
                return None

            def update_swarm(self, *_args, **_kwargs):
                return None

        runtime = mock.Mock()
        runtime.workspace = Path("workspace")
        runtime.model_name = "ollama/test"
        runtime.dashboard.return_value = SimpleNamespace(
            status="idle", tasks=(), objective="", goal_id="", plan_revision=0
        )
        runtime.active_goal.return_value = None
        console = mock.Mock()
        console.stream = io.StringIO()
        console.color = False
        answer = mock.Mock(side_effect=failure)

        with mock.patch("agent.cli.PersistentWorkspaceApp", FakeApp), mock.patch(
            "agent.cli.TelemetrySampler"
        ) as telemetry, mock.patch(
            "agent.cli.question_session", return_value=session
        ), mock.patch(
            "agent.cli.answer_question", answer
        ), mock.patch(
            "agent.cli._show_runtime_state"
        ), mock.patch(
            "agent.cli._current_ultra_run", return_value=None
        ):
            telemetry.return_value.start.return_value = None
            telemetry.return_value.stop.return_value = None
            _persistent_interactive_loop(runtime, console, SessionPreferences())

        answer.assert_called_once()
        transcript = captured["store"].snapshot().transcript
        self.assertTrue(
            any("Local model stopped unexpectedly" in item.text for item in transcript)
        )

    def test_general_sleep_mode_is_available_outside_ultra(self):
        output = io.StringIO()
        console = ConsoleUI(stream=output, color=False)
        runtime = mock.Mock()
        preferences = SessionPreferences(mode=InteractionMode.NORMAL)

        self.assertTrue(
            execute_command(runtime, console, parse_command("/sleep on"), preferences)
        )
        self.assertTrue(console.sleep_enabled)
        runtime.sleep_profile.assert_not_called()
        self.assertTrue(
            execute_command(runtime, console, parse_command("/sleep status"), preferences)
        )
        self.assertIn("safe recommended choices only", output.getvalue())

    def test_ultra_sleep_gate_failure_does_not_disable_safe_ui_sleep(self):
        output = io.StringIO()
        console = ConsoleUI(stream=output, color=False)
        runtime = mock.Mock()
        runtime.sleep_profile.side_effect = RuntimeError("Docker is not ready")
        preferences = SessionPreferences(mode=InteractionMode.ULTRA)

        self.assertTrue(
            execute_command(runtime, console, parse_command("/sleep on"), preferences)
        )
        self.assertTrue(console.sleep_enabled)
        self.assertIn("deeper Ultra Sleep was not armed", output.getvalue())

    def test_plain_project_protection_defaults_to_local_git_when_gh_is_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            output = io.StringIO()
            with mock.patch("agent.version_control.shutil.which") as which:
                which.side_effect = lambda name: "git" if name == "git" else None
                status = choose_project_protection(
                    directory,
                    rich=False,
                    input_func=lambda _prompt: "",
                    output=output,
                )

            self.assertTrue(status.dedicated_repository)
            self.assertFalse(status.github_connected)
            self.assertIn("Recommended", output.getvalue())
            self.assertTrue((Path(directory) / ".git").is_dir())

    def test_plain_access_picker_cannot_select_full_when_docker_is_not_ready(self):
        answers = iter(("2", "1"))
        output = io.StringIO()
        sandbox = SimpleNamespace(
            status=lambda: SimpleNamespace(ready=False, reason="Docker is not ready.")
        )

        selected = choose_access_level(
            rich=False,
            input_func=lambda _prompt: next(answers),
            output=output,
            sandbox=sandbox,
        )

        self.assertEqual(selected.value, "normal")
        self.assertIn("Full access is unavailable", output.getvalue())

    def test_mode_picker_cannot_select_ultra_when_runtime_prerequisite_is_missing(self):
        answers = iter(("2", "1"))
        output = io.StringIO()

        selected = choose_interaction_mode(
            rich=False,
            input_func=lambda _prompt: next(answers),
            output=output,
            ultra_disabled_reason="A usable local GPU was not detected.",
        )

        self.assertEqual(selected, InteractionMode.NORMAL)
        self.assertIn("Ultra is unavailable", output.getvalue())
        self.assertIn("usable local GPU", output.getvalue())

    def test_status_command_is_offline_import_safe_and_creates_durable_state(self):
        with tempfile.TemporaryDirectory() as directory:
            output = io.StringIO()
            with redirect_stdout(output):
                code = main(
                    [
                        "--workspace",
                        directory,
                        "--provider",
                        "ollama",
                        "--command",
                        ":status",
                        "--no-color",
                    ]
                )
            self.assertEqual(code, 0)
            rendered = output.getvalue()
            self.assertIn("GA3BAD CODING AGENT", rendered)
            self.assertIn("STATUS IDLE", rendered)
            self.assertNotIn("\x1b", rendered)
            self.assertTrue((Path(directory) / ".coding-agent" / "state.db").is_file())

    def test_slash_menu_mode_and_settings_commands_are_parsed(self):
        self.assertEqual(parse_command("/").kind, CommandKind.MENU)
        self.assertEqual(parse_command("/   ").kind, CommandKind.MENU)
        self.assertEqual(parse_command("/thinking").kind, CommandKind.THINKING)
        self.assertEqual(parse_command("/reasoning hide").args["action"], "hide")
        self.assertEqual(parse_command("/details 42").kind, CommandKind.DETAILS)
        self.assertEqual(parse_command("/details 42").args["target"], "42")
        self.assertEqual(parse_command("/doctor").kind, CommandKind.DOCTOR)
        self.assertEqual(parse_command("/readiness").kind, CommandKind.DOCTOR)
        self.assertEqual(parse_command("/doctor --live").args, {"live": True})
        self.assertEqual(parse_command("/doctor --record").args, {"record": True})
        self.assertEqual(parse_command("/doctor --record --live").args, {"record": True, "live": True})
        self.assertEqual(parse_command("/skills").kind, CommandKind.SKILLS)
        self.assertEqual(parse_command("/processes").kind, CommandKind.PROCESSES)
        self.assertEqual(parse_command("/versions").kind, CommandKind.VERSIONS)
        self.assertEqual(parse_command("/diff").args, {"target": None})
        self.assertEqual(parse_command("/diff 2").args, {"target": "2"})
        self.assertEqual(parse_command("/undo").args, {"steps": 1})
        self.assertEqual(parse_command("/undo 3").args, {"steps": 3})
        stopped = parse_command("/stop-process preview-123")
        self.assertEqual(stopped.kind, CommandKind.STOP_PROCESS)
        self.assertEqual(stopped.args["resource_id"], "preview-123")
        self.assertEqual(parse_command("/model gemma4:e4b high").args, {"model": "gemma4:e4b", "effort": "high"})
        self.assertEqual(parse_command("/model xhigh").args, {"model": None, "effort": "xhigh"})
        self.assertEqual(parse_command("/keymap").kind, CommandKind.KEYMAP)
        for unsupported in (
            "/ide", "/vim on", "/sandbox-add-read-dir C:\\Users", "/experimental"
        ):
            with self.assertRaises(CommandParseError):
                parse_command(unsupported)
        self.assertEqual(parse_command("/chat").kind, CommandKind.CHAT)
        self.assertEqual(parse_command("/explorer").kind, CommandKind.EXPLORER)
        self.assertEqual(parse_command("/continue").kind, CommandKind.RESUME)
        self.assertEqual(parse_command("/plan edit").args["action"], "edit")

        mode_query = parse_command("/mode")
        self.assertEqual(mode_query.kind, CommandKind.MODE)
        self.assertIsNone(mode_query.args["mode"])
        self.assertEqual(parse_command("/mode GOAL").args["mode"], "normal")
        self.assertEqual(parse_command("/mode\tgoal").args["mode"], "normal")
        self.assertEqual(parse_command(":mode plan").args["mode"], "plan")

        settings_query = parse_command("/settings")
        self.assertEqual(settings_query.kind, CommandKind.SETTINGS)
        self.assertEqual(settings_query.args, {"key": None, "value": None})
        setting = parse_command("/settings work-quantum 7")
        self.assertEqual(
            setting.args,
            {"key": "work_quantum", "value": "7"},
        )
        self.assertEqual(
            parse_command("/settings\tcolor\toff").args,
            {"key": "color", "value": "off"},
        )

    def test_sequential_main_commands_share_goal_mode_with_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            output = io.StringIO()
            with redirect_stdout(output):
                code = main(
                    [
                        "--workspace",
                        directory,
                        "--provider",
                        "ollama",
                        "--command",
                        "/mode goal",
                        "--command",
                        "/settings",
                        "--no-color",
                    ]
                )

            self.assertEqual(code, 0)
            rendered = output.getvalue()
            self.assertIn("NORMAL mode active", rendered)
            self.assertIn("Session settings", rendered)
            self.assertIn("mode       = normal", rendered)
            self.assertNotIn("API_KEY", rendered)

    def test_doctor_record_command_persists_benchmark_history(self):
        with tempfile.TemporaryDirectory() as directory:
            output = io.StringIO()
            with mock.patch.dict("os.environ", {"AGENT_REQUIRE_LOCAL_GPU": "0"}, clear=False):
                with redirect_stdout(output):
                    code = main(
                        [
                            "--workspace",
                            directory,
                            "--provider",
                            "ollama",
                            "--model",
                            "gemma4:e4b",
                            "--command",
                            "/doctor --record",
                            "--no-color",
                            "--plain",
                        ]
                    )
            with StateStore(directory) as store:
                rows = store.list_benchmark_results(suite_name="agent-readiness", limit=10)
                trend_rows = store.list_benchmark_results(suite_name="benchmark-trend", limit=10)

        self.assertEqual(code, 0)
        rendered = output.getvalue()
        self.assertIn("Agent readiness", rendered)
        self.assertIn("Recorded benchmark runs", rendered)
        self.assertIn("Benchmark trends", rendered)
        scenarios = {row["scenario_name"]: row for row in rows}
        self.assertIn("structural", scenarios)
        self.assertIn("behavioral", scenarios)
        self.assertEqual(scenarios["structural"]["provider"], "ollama")
        self.assertEqual(scenarios["behavioral"]["model"], "gemma4:e4b")
        self.assertGreaterEqual(scenarios["behavioral"]["metrics"]["checks"], 4)
        trend_scenarios = {row["scenario_name"]: row for row in trend_rows}
        self.assertIn("agent-readiness/structural", trend_scenarios)
        self.assertIn("agent-readiness/behavioral", trend_scenarios)
        self.assertEqual(
            trend_scenarios["agent-readiness/structural"]["inputs"]["trend"]["verdict"],
            "insufficient_history",
        )

    def test_settings_mutate_and_query_runtime_config_for_this_session(self):
        output = io.StringIO()
        console = ConsoleUI(stream=output, color=False)
        preferences = SessionPreferences()
        runtime = mock.Mock()
        runtime.config = RuntimeConfig()
        runtime.provider_name = "scripted"
        runtime.model_name = "offline"
        runtime.workspace = Path("workspace")

        def replace_config(config):
            runtime.config = config

        runtime.replace_config.side_effect = replace_config

        self.assertTrue(
            execute_command(
                runtime,
                console,
                parse_command("/settings work-quantum 7"),
                preferences,
            )
        )
        self.assertEqual(runtime.config.work_quantum_steps, 7)
        runtime.replace_config.assert_called_once()

        self.assertTrue(
            execute_command(
                runtime,
                console,
                parse_command("/settings work-quantum"),
                preferences,
            )
        )
        self.assertIn("work_quantum_steps = 7", output.getvalue())

    def test_invalid_retry_range_and_terminal_control_model_are_rejected(self):
        with mock.patch.dict(
            "os.environ",
            {"AGENT_GOAL_RETRY_BASE_MS": "60000", "AGENT_GOAL_RETRY_MAX_MS": "0"},
            clear=False,
        ):
            with self.assertRaisesRegex(ValueError, "cannot exceed"):
                RuntimeConfig.from_env()

        output = io.StringIO()
        console = ConsoleUI(stream=output, color=False)
        runtime = mock.Mock()
        runtime.provider = SimpleNamespace(model="safe-model")
        runtime.provider_name = "scripted"
        runtime.model_name = "safe-model"
        with self.assertRaisesRegex(ValueError, "control characters"):
            execute_command(
                runtime,
                console,
                parse_command("/model evil\x1b[31mred"),
                SessionPreferences(),
            )
        self.assertNotIn("\x1b", output.getvalue())
        self.assertEqual(runtime.provider.model, "safe-model")

    def test_goal_mode_runs_auto_only_after_an_explicit_successful_approval(self):
        output = io.StringIO()
        console = ConsoleUI(stream=output, color=False)
        preferences = SessionPreferences(mode=InteractionMode.GOAL)
        console.set_mode(preferences.mode)
        runtime = mock.Mock()
        runtime.dashboard.return_value = DashboardView(
            status=GoalStatus.AWAITING_PLAN_APPROVAL.value
        )
        pending_goal = SimpleNamespace(status=GoalStatus.AWAITING_PLAN_APPROVAL)
        running_goal = SimpleNamespace(status=GoalStatus.RUNNING)

        with mock.patch("agent.cli._run_auto") as run_auto:
            runtime.active_goal.return_value = pending_goal
            self.assertTrue(
                execute_command(
                    runtime,
                    console,
                    parse_command("/mode goal"),
                    preferences,
                )
            )
            run_auto.assert_not_called()

            self.assertTrue(
                execute_command(
                    runtime,
                    console,
                    parse_command("/approve 1"),
                    preferences,
                )
            )
            run_auto.assert_not_called()

            runtime.active_goal.return_value = running_goal
            runtime.dashboard.return_value = DashboardView(status=GoalStatus.RUNNING.value)
            self.assertTrue(
                execute_command(
                    runtime,
                    console,
                    parse_command("/approve 1"),
                    preferences,
                )
            )
            run_auto.assert_called_once_with(runtime, console)

            run_auto.reset_mock()
            self.assertTrue(
                execute_command(
                    runtime,
                    console,
                    parse_command(""),
                    preferences,
                )
            )
            run_auto.assert_not_called()

            self.assertTrue(
                execute_command(
                    runtime,
                    console,
                    parse_command("Here is the answer you requested"),
                    preferences,
                )
            )
            run_auto.assert_called_once_with(runtime, console)

    def test_workspace_chooser_creates_next_numbered_workspace(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "project-001").mkdir()
            output = io.StringIO()
            selected = choose_workspace(root, input_func=lambda _prompt: "", output=output)
            self.assertEqual(selected.name, "project-002")
            self.assertTrue(selected.is_dir())

    def test_unprefixed_exit_keeps_legacy_repl_behavior(self):
        self.assertEqual(parse_command("exit").kind, CommandKind.QUIT)
        self.assertEqual(parse_command("quit").kind, CommandKind.QUIT)

    def test_common_setup_errors_return_friendly_exit_code(self):
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(["--workspace", "definitely-missing-workspace", "--command", ":status"])
        self.assertEqual(code, 2)
        self.assertIn("fatal:", output.getvalue())

        with tempfile.TemporaryDirectory() as directory:
            output = io.StringIO()
            with redirect_stdout(output):
                code = main([
                    "--workspace", directory, "--create-workspace", "--command", ":status"
                ])
            self.assertEqual(code, 2)
            self.assertIn("fatal:", output.getvalue())

    def test_invalid_provider_environment_is_reported_even_with_model_override(self):
        with tempfile.TemporaryDirectory() as directory:
            output = io.StringIO()
            with mock.patch.dict("os.environ", {"LLM_PROVIDER": " invalid provider "}, clear=False):
                with redirect_stdout(output):
                    code = main([
                        "--workspace", directory, "--model", "anything", "--command", ":status"
                    ])
            self.assertEqual(code, 2)
            self.assertIn("Unknown LLM_PROVIDER", output.getvalue())

    def test_noninteractive_approval_fails_closed(self):
        output = io.StringIO()
        console = ConsoleUI(stream=output, color=False, input_func=lambda _prompt: (_ for _ in ()).throw(EOFError()))
        self.assertFalse(console.confirm_action("write_file", {"path": "x"}, "high"))
        self.assertIn("Approval denied", output.getvalue())

    def test_persistent_simple_policy_allows_project_edits_without_interrupting(self):
        console = ConsoleUI(stream=io.StringIO(), color=False)
        store = WorkspaceUIStore()
        console.bind_workspace_store(store)
        self.assertTrue(console.confirm_action("write_file", {"path": "app.py"}, "risky"))
        self.assertIsNone(store.active_attention())

    def test_persistent_simple_workspace_keeps_multiline_results_and_errors_visible(self):
        console = ConsoleUI(stream=io.StringIO(), color=False)
        store = WorkspaceUIStore()
        console.bind_workspace_store(store)

        console.write("PLAN\n  1. Build UI\n  2. Run tests")
        console.write("error: provider is unavailable")

        transcript = store.snapshot().transcript
        self.assertEqual(len(transcript), 2)
        self.assertTrue(all(not item.technical for item in transcript))
        self.assertIn("Run tests", transcript[0].text)
        self.assertIn("provider is unavailable", transcript[1].text)

    def test_persistent_project_checks_ask_once_per_session(self):
        console = ConsoleUI(stream=io.StringIO(), color=False)
        store = WorkspaceUIStore()
        console.bind_workspace_store(store)
        results: list[bool] = []
        worker = Thread(
            target=lambda: results.append(
                console.confirm_action(
                    "run_bash", {"command": "python -m pytest -q"}, "risky"
                )
            )
        )
        worker.start()
        for _ in range(100):
            if store.active_attention() is not None:
                break
            Event().wait(0.01)
        self.assertIsNotNone(store.active_attention())
        self.assertTrue(store.resolve_attention("allow_session"))
        worker.join(1)
        self.assertEqual(results, [True])
        self.assertTrue(
            console.confirm_action(
                "run_bash", {"command": "python -m pytest tests/test_cli.py"}, "risky"
            )
        )
        self.assertIsNone(store.active_attention())

    def test_persistent_approval_shows_target_and_defaults_to_deny(self):
        console = ConsoleUI(stream=io.StringIO(), color=False)
        store = WorkspaceUIStore()
        console.bind_workspace_store(store)
        results = []
        worker = Thread(
            target=lambda: results.append(
                console.confirm_action("run_bash", {"command": "npm install package-x"}, "risky")
            )
        )
        worker.start()
        for _ in range(100):
            if store.active_attention() is not None:
                break
            Event().wait(0.01)
        request = store.active_attention()
        self.assertIsNotNone(request)
        self.assertIn("npm install package-x", request.message)
        primary = [item.key for item in request.options if item.primary]
        self.assertEqual(primary, ["deny"])
        store.resolve_selected_attention()
        worker.join(1)
        self.assertEqual(results, [False])

    def test_background_approval_is_handed_to_the_main_ui_thread(self):
        console = ConsoleUI(stream=io.StringIO(), color=False)
        console.set_background_working(True)
        ready = Event()
        result: list[bool] = []

        with mock.patch.object(console, "_modal_available", return_value=True), mock.patch.object(
            console, "_interrupt_composer", side_effect=ready.set
        ), mock.patch.object(console, "_select_approval", return_value=True):
            worker = Thread(
                target=lambda: result.append(
                    console.confirm_action("run_bash", {"command": "mkdir gui logic"}, "critical")
                ),
                daemon=True,
            )
            worker.start()
            self.assertTrue(ready.wait(1.0))
            self.assertTrue(console.has_pending_approval())
            self.assertTrue(console.resolve_pending_approval())
            worker.join(1.0)

        self.assertFalse(worker.is_alive())
        self.assertEqual(result, [True])

    def test_approval_screen_failure_is_not_reported_as_user_denial(self):
        console = ConsoleUI(stream=io.StringIO(), color=False)
        console.set_background_working(True)
        ready = Event()
        errors: list[BaseException] = []

        def request_approval() -> None:
            try:
                console.confirm_action("run_bash", {"command": "test"}, "critical")
            except BaseException as exc:
                errors.append(exc)

        with mock.patch.object(console, "_modal_available", return_value=True), mock.patch.object(
            console, "_interrupt_composer", side_effect=ready.set
        ), mock.patch.object(
            console,
            "_select_approval",
            side_effect=RuntimeError("approval UI unavailable"),
        ):
            worker = Thread(target=request_approval, daemon=True)
            worker.start()
            self.assertTrue(ready.wait(1.0))
            self.assertFalse(console.resolve_pending_approval())
            worker.join(1.0)

        self.assertFalse(worker.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertIn("approval UI unavailable", str(errors[0]))
        self.assertIn("Approval screen error", console.stream.getvalue())

    def test_modal_approval_survives_prompt_toolkit_patching_stdout(self):
        terminal = _TTY()
        console = ConsoleUI(stream=terminal, color=False)

        with mock.patch("agent.ui.sys.stdin", _TTY()), mock.patch(
            "agent.ui.sys.stdout", io.StringIO()
        ):
            self.assertTrue(console._modal_available())

    def test_background_checkpoint_wakes_the_composer(self):
        console = ConsoleUI(stream=io.StringIO(), color=False, input_func=lambda _prompt: "")
        console.wake_prompt()

        with self.assertRaises(WorkspaceRefreshRequested):
            console.prompt()

    def test_ctrl_c_at_approval_propagates_to_the_checkpoint_handler(self):
        output = io.StringIO()
        console = ConsoleUI(
            stream=output,
            color=False,
            input_func=lambda _prompt: (_ for _ in ()).throw(KeyboardInterrupt()),
        )
        with self.assertRaises(KeyboardInterrupt):
            console.confirm_action("run_bash", {"command": "test"}, "high")
        self.assertIn("checkpointing", output.getvalue())

    def test_prompt_ctrl_c_relies_on_the_single_checkpoint_event(self):
        runtime = mock.Mock()
        console = mock.Mock()
        console.prompt.side_effect = [KeyboardInterrupt(), EOFError()]
        with mock.patch("agent.cli._show_runtime_state"):
            interactive_loop(runtime, console, SessionPreferences())

        runtime.checkpoint_interrupt.assert_called_once_with()
        console.write.assert_called_once_with("\nInput closed. Durable goal state is saved.")

    def test_noninteractive_ctrl_c_does_not_dump_status_after_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), mock.patch(
            "agent.cli.execute_command", side_effect=KeyboardInterrupt()
        ), mock.patch(
            "agent.cli.AgentRuntime.checkpoint_interrupt"
        ) as checkpoint, mock.patch(
            "agent.cli._show_runtime_state"
        ) as show_state:
            code = main(
                [
                    "--workspace",
                    directory,
                    "--provider",
                    "ollama",
                    "--command",
                    "/status",
                    "--no-color",
                    "--plain",
                ]
            )

        self.assertEqual(code, 130)
        checkpoint.assert_called_once_with()
        show_state.assert_not_called()


if __name__ == "__main__":
    unittest.main()
