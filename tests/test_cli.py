from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from agent.cli import choose_workspace, execute_command, main
from agent.commands import CommandKind, parse_command
from agent.config import InteractionMode, RuntimeConfig, SessionPreferences
from agent.models import GoalStatus
from agent.ui import ConsoleUI, DashboardView


class CLITests(unittest.TestCase):
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

        mode_query = parse_command("/mode")
        self.assertEqual(mode_query.kind, CommandKind.MODE)
        self.assertIsNone(mode_query.args["mode"])
        self.assertEqual(parse_command("/mode GOAL").args["mode"], "goal")
        self.assertEqual(parse_command("/mode\tgoal").args["mode"], "goal")
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
            self.assertIn("GOAL mode active", rendered)
            self.assertIn("Session settings", rendered)
            self.assertIn("mode       = goal", rendered)
            self.assertNotIn("API_KEY", rendered)

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


if __name__ == "__main__":
    unittest.main()
