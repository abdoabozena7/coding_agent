from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
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
from agent.commands import CommandKind, parse_command
from agent.config import InteractionMode, RuntimeConfig, SessionPreferences
from agent.models import GoalStatus
from agent.store import StateStore
from agent.ui import ConsoleUI, DashboardView


class CLITests(unittest.TestCase):
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
        self.assertEqual(parse_command("/ide").kind, CommandKind.IDE)
        self.assertEqual(parse_command("/keymap").kind, CommandKind.KEYMAP)
        self.assertEqual(parse_command("/vim on").args, {"state": "on"})
        self.assertEqual(
            parse_command("/sandbox-add-read-dir C:\\Users").args,
            {"path": "C:\\Users"},
        )
        self.assertEqual(parse_command("/experimental").args, {"state": "status"})

        mode_query = parse_command("/mode")
        self.assertEqual(mode_query.kind, CommandKind.MODE)
        self.assertIsNone(mode_query.args["mode"])
        self.assertEqual(parse_command("/mode GOAL").args["mode"], "normal")
        self.assertEqual(parse_command("/mode\tgoal").args["mode"], "normal")
        self.assertEqual(parse_command(":mode plan").args["mode"], "normal")

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
