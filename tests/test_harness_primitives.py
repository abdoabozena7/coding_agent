from __future__ import annotations

import io
import unittest

from agent.commands import CommandKind, CommandParseError, parse_command
from agent.context import maybe_compact
from agent.control import ControlValidationError, validate_control_call
from agent.events import UIEvent
from agent.safety import ProgressWatchdog, redact_data, redact_text
from agent.ui import (
    SLASH_COMMANDS,
    ConsoleUI,
    DashboardView,
    SlashCommandCompleter,
    TaskView,
    WorkerView,
    render_agents,
    render_brand,
    render_dashboard,
    render_memory,
    render_plan,
    render_slash_menu,
    render_status,
)


class CommandTests(unittest.TestCase):
    def test_plain_text_and_plan_edit_commands(self):
        self.assertEqual(parse_command("build it").kind, CommandKind.TEXT)
        add = parse_command(":add Harden paths :: traversal tests pass")
        self.assertEqual(add.args["text"], "Harden paths")
        self.assertEqual(add.args["acceptance_criteria"], "traversal tests pass")
        edit = parse_command("/edit t002 Better title")
        self.assertEqual(
            edit.args,
            {"task_id": "T002", "field": "task", "value": "Better title"},
        )
        criteria = parse_command(":edit T002 accept First proof || Second proof")
        self.assertEqual(criteria.args["field"], "accept")

    def test_revision_and_slice_are_validated(self):
        self.assertEqual(parse_command(":approve 7").args["revision"], 7)
        self.assertEqual(parse_command(":run 3").args["steps"], 3)
        with self.assertRaises(CommandParseError):
            parse_command(":run 0")
        with self.assertRaises(CommandParseError):
            parse_command(":approve latest")
        resolved = parse_command(":resolve action_123 not-run inspected the workspace")
        self.assertEqual(resolved.args["resolution"], "not-run")

    def test_slash_errors_keep_the_slash_prefix(self):
        with self.assertRaisesRegex(CommandParseError, r"/reject FEEDBACK"):
            parse_command("/reject")
        with self.assertRaisesRegex(CommandParseError, r"/replan FEEDBACK"):
            parse_command("/replan")

    def test_v3_ultra_views_and_permission_commands_parse(self):
        self.assertEqual(parse_command("/mode ultra").args["mode"], "ultra")
        self.assertEqual(parse_command("/permissions full").args["level"], "full")
        self.assertEqual(parse_command("/tree M001").args["target"], "M001")
        self.assertEqual(parse_command("/agents --all").args["all"], True)
        self.assertEqual(parse_command("/trace latest").args["target"], "latest")
        self.assertEqual(
            parse_command("/answer platform Desktop").args,
            {"question_id": "platform", "value": "Desktop"},
        )


class ControlSchemaTests(unittest.TestCase):
    def test_plan_requires_verifiable_typed_tasks(self):
        valid = {
            "summary": "Implement and verify the requested behavior.",
            "applicability_evidence": [
                {
                    "fact": "The inspected repository lacks durable state.",
                    "source": "agent/runtime.py",
                    "supports_tasks": ["T001"],
                }
            ],
            "execution_strategy": "Edit the state layer and run the crash recovery test.",
            "expected_changes": [
                {
                    "path": "agent/store.py",
                    "intent": "Persist goal and task state transactionally.",
                    "supports_tasks": ["T001"],
                }
            ],
            "tasks": [
                {
                    "id": "T001",
                    "title": "Implement durable state",
                    "description": "Persist goal and tasks transactionally.",
                    "acceptance_criteria": ["Restart restores the same active goal."],
                    "verification": ["Run the crash recovery test."],
                    "depends_on": [],
                    "risk": "high",
                }
            ],
        }
        self.assertIs(validate_control_call("propose_plan", valid), valid)
        invalid = {**valid, "surprise": True}
        with self.assertRaisesRegex(ControlValidationError, "unknown fields"):
            validate_control_call("propose_plan", invalid)
        chat_only = dict(valid)
        del chat_only["expected_changes"]
        with self.assertRaisesRegex(ControlValidationError, "expected_changes is required"):
            validate_control_call("propose_plan", chat_only)

    def test_done_update_shape_requires_evidence_field(self):
        with self.assertRaisesRegex(ControlValidationError, "evidence is required"):
            validate_control_call(
                "update_task",
                {"task_id": "T001", "status": "done", "note": "done"},
            )


class ContextTests(unittest.TestCase):
    def test_compaction_keeps_recent_tool_call_pair(self):
        messages = []
        for index in range(8):
            messages.extend(
                [
                    {"role": "user", "content": f"turn {index} " + "x" * 80},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{"id": f"c{index}", "name": "read_file", "args": {"path": "a"}}],
                    },
                    {"role": "tool", "id": f"c{index}", "name": "read_file", "content": "result"},
                ]
            )
        compacted = maybe_compact(messages, lambda head: "facts", max_chars=200, keep_recent_user_turns=2)
        self.assertTrue(compacted[0]["content"].startswith("[HARNESS CONVERSATION SUMMARY"))
        tail_calls = {
            call["id"]
            for message in compacted
            for call in message.get("tool_calls", [])
        }
        tail_results = {message["id"] for message in compacted if message.get("role") == "tool"}
        self.assertEqual(tail_calls, tail_results)


class SafetyTests(unittest.TestCase):
    def test_secret_redaction_is_recursive(self):
        self.assertNotIn("sk-abcdefghijklmnopqrst", redact_text("key sk-abcdefghijklmnopqrst"))
        data = redact_data({"OPENAI_API_KEY": "secret", "nested": ["Bearer abcdefghijklmnop"]})
        self.assertEqual(data["OPENAI_API_KEY"], "[REDACTED]")
        self.assertNotIn("abcdefghijklmnop", data["nested"][0])

    def test_generic_credentials_and_private_keys_are_redacted_before_persistence(self):
        private_key = (
            "-----BEGIN PRIVATE KEY-----\n"
            + "A" * 80
            + "\n-----END PRIVATE KEY-----"
        )
        source = (
            "api_key=generic-secret-value "
            "aws=AKIAABCDEFGHIJKLMNOP "
            "jwt=eyJabcdefghijk.abcdefghijkl.abcdefghijkl "
            + private_key
        )

        redacted = redact_text(source)

        self.assertNotIn("generic-secret-value", redacted)
        self.assertNotIn("AKIAABCDEFGHIJKLMNOP", redacted)
        self.assertNotIn("eyJabcdefghijk", redacted)
        self.assertNotIn("A" * 40, redacted)

    def test_watchdog_rejects_repeated_action(self):
        watchdog = ProgressWatchdog(repeat_limit=2)
        args = {"path": "missing"}
        self.assertFalse(watchdog.check("read_file", args).stalled)
        watchdog.record("read_file", args, "error")
        watchdog.record("read_file", args, "error")
        self.assertTrue(watchdog.check("read_file", args).stalled)


class DashboardTests(unittest.TestCase):
    def test_dashboard_is_ascii_and_contains_control_state(self):
        output = render_dashboard(
            DashboardView(
                objective="Build a durable agent",
                status="running",
                plan_revision=3,
                approved_revision=3,
                tasks=[TaskView("T001", "Persist goals", "done"), TaskView("T002", "Review", "in_progress")],
                workers=[WorkerView("W1", "T002", "task-specific adversarial reviewer")],
                provider="fake",
                model="small",
                workspace="C:/repo",
            ),
            width=110,
        )
        output.encode("ascii")
        self.assertIn("GA3BAD CODING AGENT", output)
        self.assertIn("MODE PLAN", output)
        self.assertIn("PLAN r3 / r3", output)
        self.assertIn("[x] T001", output)
        self.assertIn("task-specific advers", output)

    def test_full_plan_shows_every_approval_bound_field(self):
        tasks = [
            TaskView(
                f"T{index:03d}",
                f"Task {index}",
                "pending",
                acceptance_criteria=[f"criterion {index}"],
                verification=[f"verification {index}"],
                depends_on=[f"T{index - 1:03d}"] if index > 1 else [],
                risk="high" if index == 16 else "medium",
            )
            for index in range(1, 17)
        ]
        output = render_plan(
            DashboardView(
                objective="A complex goal",
                status="awaiting_plan_approval",
                plan_revision=4,
                plan_summary="All approval-bound work.",
                plan_applicability=[
                    {
                        "fact": "The repository contains the implementation target.",
                        "source": "agent/runtime.py",
                        "supports_tasks": ["T001"],
                    }
                ],
                execution_strategy="Edit the runtime, then execute focused and full verification.",
                expected_changes=[
                    {
                        "path": "agent/runtime.py",
                        "intent": "Implement the accepted runtime behavior.",
                        "supports_tasks": ["T001"],
                    }
                ],
                tasks=tasks,
            ),
            width=100,
        )
        self.assertIn("T016", output)
        self.assertIn("criterion 16", output)
        self.assertIn("verification 16", output)
        self.assertIn("risk=high", output)
        self.assertIn("depends_on=T015", output)
        self.assertIn("APPLICABILITY EVIDENCE", output)
        self.assertIn("EXECUTION STRATEGY", output)
        self.assertIn("EXPECTED WORKSPACE CHANGES", output)
        self.assertIn("agent/runtime.py", output)

    def test_dashboard_respects_narrow_terminal_width(self):
        output = render_dashboard(
            DashboardView(
                objective="A very long objective that needs wrapping or clipping",
                status="running",
                tasks=[TaskView("T001", "A long task title that cannot fit")],
            ),
            width=36,
        )
        self.assertTrue(all(len(line) <= 36 for line in output.splitlines()))
        very_narrow = render_dashboard(DashboardView(objective="x"), width=20)
        self.assertTrue(all(len(line) == 20 for line in very_narrow.splitlines()))


class TerminalUITests(unittest.TestCase):
    def test_live_slash_completer_covers_every_palette_command_and_settings(self):
        if SlashCommandCompleter is None:
            self.skipTest("prompt-toolkit is not installed")
        from prompt_toolkit.document import Document

        completer = SlashCommandCompleter()
        top = list(completer.get_completions(Document("/"), None))
        self.assertEqual({item.text for item in top}, {command for command, _ in SLASH_COMMANDS})
        self.assertTrue(all(item.start_position == -1 for item in top))

        modes = list(completer.get_completions(Document("/mode g"), None))
        self.assertEqual([item.text for item in modes], ["goal"])
        tab_modes = list(completer.get_completions(Document("/mode\t\tg"), None))
        self.assertEqual([item.text for item in tab_modes], ["goal"])
        settings = {
            item.text
            for item in completer.get_completions(Document("/settings "), None)
        }
        self.assertTrue({"mode", "color", "provider", "model", "workspace"} <= settings)
        spaced_settings = {
            item.text
            for item in completer.get_completions(Document("/settings    "), None)
        }
        self.assertEqual(settings, spaced_settings)

    def test_brand_is_strict_ascii_with_centered_small_subtitle(self):
        rendered = render_brand()
        rendered.encode("ascii")
        lines = rendered.splitlines()

        self.assertEqual(len(lines), 6)
        self.assertEqual(
            lines[:5],
            [
                "  ____    _    _____ ____    _    ____ ",
                " / ___|  / \\  |___ /| __ )  / \\  |  _ \\",
                "| |  _  / _ \\   |_ \\|  _ \\ / _ \\ | | | |",
                "| |_| |/ ___ \\ ___) | |_) / ___ \\| |_| |",
                " \\____/_/   \\_\\____/|____/_/   \\_\\____/ ",
            ],
        )
        self.assertEqual(lines[-1].strip(), "coding agent")
        self.assertEqual(len(lines[-1]), max(len(line) for line in lines[:5]))
        self.assertEqual(lines[-1], "coding agent".center(len(lines[-1])))

    def test_slash_menu_lists_modes_settings_and_legacy_fallback(self):
        rendered = render_slash_menu()
        rendered.encode("ascii")

        self.assertIn("/mode", rendered)
        self.assertIn("/settings", rendered)
        self.assertIn("/model", rendered)
        self.assertIn("/mode plan", rendered)
        self.assertIn("/mode goal", rendered)
        self.assertIn("/mode ultra", rendered)
        self.assertIn("/trace", rendered)
        self.assertIn("Legacy :commands remain supported.", rendered)

    def test_show_brand_without_color_matches_plain_renderer(self):
        output = io.StringIO()
        ConsoleUI(stream=output, color=False).show_brand()

        self.assertEqual(output.getvalue(), render_brand() + "\n\n")

    def test_auto_color_tolerates_a_legacy_stdio_wrapper_without_isatty(self):
        class LegacyWrapper:
            def __init__(self):
                self.value = ""

            def write(self, text):
                self.value += text

            def flush(self):
                pass

        stream = LegacyWrapper()
        console = ConsoleUI(stream=stream, color=None, input_func=lambda _prompt: "/quit")
        console.show_brand()
        self.assertFalse(console.color)
        self.assertNotIn("\x1b", stream.value)

    def test_show_brand_uses_bold_green_art_and_dim_green_subtitle(self):
        output = io.StringIO()
        ConsoleUI(stream=output, color=True).show_brand()
        rendered = output.getvalue()

        self.assertTrue(rendered.startswith("\033[1m\033[32m"))
        self.assertIn("\033[2m\033[32m" + "coding agent".center(40), rendered)
        self.assertTrue(rendered.endswith("\033[0m\n\n"))

    def test_basic_prompt_fallback_includes_current_mode(self):
        prompts: list[str] = []
        console = ConsoleUI(
            stream=io.StringIO(),
            color=False,
            interaction_mode="goal",
            input_func=lambda prompt: prompts.append(prompt) or "/status",
        )

        self.assertEqual(console.prompt(), "/status")
        self.assertEqual(prompts, ["GA3BAD [GOAL]> "])

        console.set_mode("plan")
        self.assertEqual(console.prompt(), "/status")
        self.assertEqual(prompts[-1], "GA3BAD [PLAN]> ")

    def test_ultra_status_is_sparse_and_active_events_are_gold(self):
        view = DashboardView(
            objective="Build the game",
            status="running",
            provider="ollama",
            model="cloud-model",
            interaction_mode="ultra",
            workspace="workspace",
        )
        rendered = render_status(
            view,
            access_level="full",
            execution_class="cloud",
            active_agents=4,
        )
        self.assertNotIn("+---", rendered)
        self.assertIn("MODE ULTRA · STATUS RUNNING", rendered)
        self.assertIn("FULL · CLOUD · agents 4", rendered)

        output = io.StringIO()
        console = ConsoleUI(stream=output, color=True, interaction_mode="ultra")
        console.on_event(
            UIEvent(
                "ultra.agent_started",
                "coder",
                {"role": "coder", "phase": "implement", "node_id": "Physics"},
            )
        )
        self.assertIn("\033[38;5;220m", output.getvalue())
        self.assertIn("[Physics · coder]", output.getvalue())

    def test_agents_view_shows_role_phase_and_node_title(self):
        rendered = render_agents(
            [
                {
                    "role": "coder",
                    "status": "running",
                    "phase": "implement",
                    "work_node_id": "node-physics",
                    "model": "qwen",
                }
            ],
            node_titles={"node-physics": "Physics"},
        )
        self.assertIn("[coder] running · implement · Physics · qwen", rendered)

    def test_memory_view_shows_the_entry_content_without_cards(self):
        rendered = render_memory(
            [
                {
                    "section": "decision",
                    "title": "Physics units",
                    "content": "Use SI units so every module shares one contract.",
                }
            ]
        )
        self.assertIn("[decision] Physics units", rendered)
        self.assertIn("Use SI units", rendered)
        self.assertNotIn("+---", rendered)


if __name__ == "__main__":
    unittest.main()
