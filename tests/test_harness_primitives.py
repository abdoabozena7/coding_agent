from __future__ import annotations

import io
import unittest
from unittest import mock

from agent.commands import CommandKind, CommandParseError, parse_command
from agent.context import maybe_compact
from agent.config import ReasoningEffort
from agent.control import ControlValidationError, validate_control_call
from agent.events import UIEvent
from agent.safety import ProgressWatchdog, redact_data, redact_text
from agent.ui import (
    CODEX_SLASH_COMMANDS,
    SLASH_COMMANDS,
    ConsoleUI,
    DashboardView,
    SlashCommandCompleter,
    TaskView,
    WorkerView,
    prompt_receipt,
    render_agents,
    render_brand,
    render_dashboard,
    render_memory,
    render_plan,
    render_slash_menu,
    render_status,
)


class CommandTests(unittest.TestCase):
    def test_reasoning_effort_aliases_and_validation(self):
        self.assertIs(ReasoningEffort.parse("max"), ReasoningEffort.XHIGH)
        self.assertIs(ReasoningEffort.parse("MEDIUM"), ReasoningEffort.MEDIUM)
        with self.assertRaisesRegex(ValueError, "low, medium, high, or xhigh"):
            ReasoningEffort.parse("infinite")

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
        self.assertEqual(parse_command("/agent 12").kind, CommandKind.AGENT)
        self.assertEqual(parse_command("/agents node-api").args["target"], "node-api")
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

    def test_structured_task_evidence_is_canonicalized_without_a_retry(self):
        value = validate_control_call(
            "update_task",
            {
                "task_id": "T001",
                "status": "done",
                "note": "verified",
                "evidence": [
                    {
                        "summary": "hello.txt contains the requested text",
                        "path": "hello.txt",
                    }
                ],
            },
        )

        self.assertEqual(
            value["evidence"],
            ["hello.txt contains the requested text [source: hello.txt]"],
        )

    def test_control_validation_reports_independent_defects_together(self):
        with self.assertRaises(ControlValidationError) as caught:
            validate_control_call(
                "propose_plan",
                {
                    "summary": "A complete plan is required.",
                    "applicability_evidence": [{}],
                    "execution_strategy": "Inspect, implement, and verify the requested change.",
                    "expected_changes": [{}],
                    "tasks": [{}],
                },
            )

        message = str(caught.exception)
        self.assertIn("applicability_evidence[0].fact is required", message)
        self.assertIn("applicability_evidence[0].supports_tasks is required", message)
        self.assertIn("expected_changes[0].intent is required", message)
        self.assertIn("tasks[0].title is required", message)
        self.assertIn("tasks[0].verification is required", message)


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
        self.assertIn("MODE NORMAL", output)
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

    def test_approved_plan_groups_used_completed_and_unused_work(self):
        output = render_plan(
            DashboardView(
                objective="Ship the feature",
                status="running",
                plan_revision=2,
                approved_revision=2,
                plan_applicability=[{"fact": "internal planning detail"}],
                expected_changes=[{"path": "internal planning detail"}],
                tasks=[
                    TaskView("T001", "Active work", "in_progress"),
                    TaskView("T002", "Finished work", "done"),
                    TaskView("T003", "Excluded work", "skipped"),
                ],
            ),
            width=90,
        )

        self.assertIn("EXECUTION PLAN", output)
        self.assertIn("IN USE (1)", output)
        self.assertIn("COMPLETED (1)", output)
        self.assertIn("NOT USED (1)", output)
        self.assertNotIn("APPLICABILITY EVIDENCE", output)
        self.assertNotIn("EXPECTED WORKSPACE CHANGES", output)

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
    def test_long_prompt_receipt_is_compact_without_mutating_content(self):
        content = "requirement\n" * 1000
        receipt = prompt_receipt(content)

        self.assertEqual(receipt, f"[Pasted Content {len(content)} chars]")
        self.assertEqual(content, "requirement\n" * 1000)

    def test_short_prompt_receipt_stays_readable(self):
        self.assertEqual(prompt_receipt("fix the loader"), "fix the loader")

    def test_colored_prompt_style_uses_supported_color_values(self):
        console = ConsoleUI(stream=io.StringIO(), color=True)
        style = console._prompt_style()

        self.assertIsNotNone(style)

    def test_thoughts_stream_live_then_collapse_into_session_history(self):
        class TTY(io.StringIO):
            encoding = "utf-8"

            def isatty(self):
                return True

        output = TTY()
        console = ConsoleUI(stream=output, color=False, reduced_motion=True)
        with mock.patch.object(console._live_activity, "start") as start, mock.patch.object(
            console._live_activity, "stop"
        ), mock.patch.object(console._live_activity, "update_label") as update_label:
            console.on_event(UIEvent("step", data={"actor": "planner", "step": 1}))
            console.on_event(UIEvent("model_thought", "Inspect the empty workspace first."))
            console.on_event(
                UIEvent("model_text", "\n{\"tool\":\"list_files\"}", {"actor": "planner"})
            )
            console.on_event(UIEvent("usage", data={"input_tokens": 1, "output_tokens": 1}))

        start.assert_called_with("plan", "Planner · step 1")
        update_label.assert_any_call("Inspect the empty workspace first.")
        self.assertIn("Inspect the empty workspace first.", output.getvalue())
        self.assertIn("collapsed", output.getvalue())
        self.assertNotIn("Response", output.getvalue())
        self.assertIn("Inspect the empty workspace first.", console.thought_blocks()[0]["text"])
        self.assertIn("list_files", console.thought_blocks()[0]["text"])

    def test_visible_thinking_keeps_prose_rows_but_filters_code(self):
        output = io.StringIO()
        console = ConsoleUI(stream=output, color=False, plain=True)
        console.on_event(UIEvent("step", data={"actor": "planner", "step": 1}))
        console.on_event(UIEvent("model_thought", "Inspect files\nCompare results\n```python\nprint('hidden')\n```"))
        console.on_event(UIEvent("usage"))

        rendered = output.getvalue()
        self.assertIn("Inspect files", rendered)
        self.assertIn("Compare results", rendered)
        self.assertNotIn("print('hidden')", rendered)

    def test_live_thinking_uses_codex_working_row(self):
        class TTY(io.StringIO):
            encoding = "utf-8"

            def isatty(self):
                return True

        output = TTY()
        console = ConsoleUI(stream=output, color=True, reduced_motion=True)
        activity = console._live_activity
        activity._state = "plan"
        activity._label = "Inspecting the workspace"
        activity._started = 0.0
        with mock.patch("agent.ui.time.monotonic", return_value=2.0), mock.patch.object(
            activity, "_supports_esc_interrupt", return_value=True
        ):
            activity._draw(0)

        rendered = output.getvalue()
        self.assertIn("Working", rendered)
        self.assertIn("esc to interrupt", rendered)
        self.assertNotIn("Inspecting the workspace", rendered)
        self.assertEqual(rendered.count("\n"), 1)

    def test_chat_code_is_response_not_thinking(self):
        output = io.StringIO()
        console = ConsoleUI(stream=output, color=False, plain=True)
        console.on_event(UIEvent("step", data={"actor": "chat", "step": 1}))
        console.on_event(UIEvent("model_thought", "I will implement the function."))
        console.on_event(UIEvent("model_text", "```python\nprint('ok')\n```", {"actor": "chat"}))

        self.assertIn("Response", output.getvalue())
        self.assertIn("print('ok')", output.getvalue())
        self.assertNotIn("print('ok')", console.thought_blocks()[0]["text"])

    def test_live_activity_hides_unreviewed_plan_attempts_and_coalesces_duplicates(self):
        class TTY(io.StringIO):
            encoding = "utf-8"

            def isatty(self):
                return True

        output = TTY()
        console = ConsoleUI(stream=output, color=False, reduced_motion=True)
        with mock.patch.object(console._live_activity, "start"), mock.patch.object(
            console._live_activity, "stop"
        ):
            for call_id in ("one", "two"):
                console.on_event(
                    UIEvent(
                        "tool_call",
                        "list_files",
                        {"args": {"path": "."}, "actor": "planner", "id": call_id},
                    )
                )
                console.on_event(
                    UIEvent(
                        "tool_result",
                        "(no files under '.')",
                        {"tool": "list_files", "actor": "planner"},
                    )
                )

            console.on_event(
                UIEvent("tool_call", "propose_plan", {"args": {}, "actor": "planner"})
            )
            console.on_event(
                UIEvent(
                    "tool_result",
                    "Plan proposal captured for independent critique.",
                    {"tool": "propose_plan", "actor": "planner"},
                )
            )
            self.assertNotIn("Prepared plan", output.getvalue())

            console.on_event(UIEvent("plan", "Plan r1 is ready for approval."))

        rendered = output.getvalue()
        self.assertEqual(rendered.count("Inspected workspace"), 1)
        self.assertIn("Plan r1 is ready", rendered)

    def test_full_screen_modal_buffers_background_events_until_picker_closes(self):
        class TTY(io.StringIO):
            encoding = "utf-8"

            def isatty(self):
                return True

        output = TTY()
        console = ConsoleUI(stream=output, color=False, reduced_motion=True)
        with mock.patch.object(console._live_activity, "stop"):
            with console.full_screen_modal():
                console.on_event(UIEvent("warning", "Worker reached a checkpoint."))
                self.assertEqual(output.getvalue(), "")

        self.assertIn("Worker reached a checkpoint", output.getvalue())

    def test_full_screen_modal_preserves_every_buffered_event_in_order(self):
        console = ConsoleUI(stream=io.StringIO(), color=False)
        rendered: list[str] = []
        with mock.patch.object(
            console,
            "_render_event",
            side_effect=lambda event: rendered.append(event.message),
        ):
            with console.full_screen_modal():
                for index in range(600):
                    console.on_event(UIEvent("warning", f"event-{index:03d}"))

        self.assertEqual(rendered, [f"event-{index:03d}" for index in range(600)])

    def test_plain_activity_pairs_tools_and_suppresses_internal_plan_retries(self):
        output = io.StringIO()
        console = ConsoleUI(stream=output, color=False, plain=True, reduced_motion=True)
        for call_id in ("one", "two"):
            console.on_event(
                UIEvent(
                    "tool_call",
                    "list_files",
                    {"args": {"path": "."}, "actor": "planner", "id": call_id},
                )
            )
            console.on_event(
                UIEvent(
                    "tool_result",
                    "(no files under '.')",
                    {"tool": "list_files", "actor": "planner"},
                )
            )
        console.on_event(
            UIEvent("tool_call", "propose_plan", {"args": {}, "actor": "planner"})
        )
        console.on_event(
            UIEvent(
                "tool_result",
                "Plan proposal captured for independent critique.",
                {"tool": "propose_plan", "actor": "planner"},
            )
        )

        rendered = output.getvalue()
        self.assertEqual(rendered.count("Inspected workspace"), 1)
        self.assertNotIn("list_files", rendered)
        self.assertNotIn("Prepared plan", rendered)

    def test_concurrent_tool_results_keep_their_node_specific_details(self):
        class TTY(io.StringIO):
            encoding = "utf-8"

            def isatty(self):
                return True

        output = TTY()
        console = ConsoleUI(stream=output, color=False, reduced_motion=True)
        with mock.patch.object(console._live_activity, "start"), mock.patch.object(
            console._live_activity, "stop"
        ):
            for node, path in (("node-a", "a.py"), ("node-b", "b.py")):
                console.on_event(
                    UIEvent(
                        "tool_call",
                        "edit_file",
                        {
                            "args": {"path": path},
                            "actor": "implementer",
                            "node_id": node,
                        },
                    )
                )
            for node in ("node-a", "node-b"):
                console.on_event(
                    UIEvent(
                        "tool_result",
                        "Updated file.",
                        {
                            "tool": "edit_file",
                            "actor": "implementer",
                            "node_id": node,
                        },
                    )
                )

        rendered = output.getvalue()
        self.assertIn("Edited file · a.py", rendered)
        self.assertIn("Edited file · b.py", rendered)

    def test_terminal_plan_failure_resets_visual_retry_counter(self):
        class TTY(io.StringIO):
            encoding = "utf-8"

            def isatty(self):
                return True

        console = ConsoleUI(stream=TTY(), color=False, reduced_motion=True)
        with mock.patch.object(console._live_activity, "start"), mock.patch.object(
            console._live_activity, "stop"
        ):
            console.on_event(
                UIEvent("tool_call", "propose_plan", {"args": {}, "actor": "planner"})
            )
            console.on_event(
                UIEvent(
                    "tool_result",
                    "Error: invalid plan proposal; fields are missing",
                    {"tool": "propose_plan", "actor": "planner"},
                )
            )
            self.assertEqual(console._plan_format_retries, 1)
            console.on_event(
                UIEvent(
                    "error",
                    "Plan could not be prepared.",
                    {"attempts": 4},
                )
            )

        self.assertEqual(console._plan_format_retries, 0)

    def test_every_terminal_planning_pause_resets_visual_retry_state(self):
        class TTY(io.StringIO):
            encoding = "utf-8"

            def isatty(self):
                return True

        console = ConsoleUI(stream=TTY(), color=False, reduced_motion=True)
        with mock.patch.object(console._live_activity, "start"), mock.patch.object(
            console._live_activity, "stop"
        ):
            for terminal_event in (
                UIEvent(
                    "warning",
                    "The planner reached its bounded slice without a valid plan.",
                    {"planning_terminal": True},
                ),
                UIEvent("questions", "Planning needs one decision."),
            ):
                console._plan_format_retries = 2
                console._plan_recovered_retries = 1
                console.on_event(terminal_event)
                self.assertEqual(console._plan_format_retries, 0)
                self.assertEqual(console._plan_recovered_retries, 0)

    def test_live_slash_completer_covers_every_palette_command_and_settings(self):
        if SlashCommandCompleter is None:
            self.skipTest("prompt-toolkit is not installed")
        from prompt_toolkit.document import Document

        completer = SlashCommandCompleter()
        top = list(completer.get_completions(Document("/"), None))
        from agent.ui import ALL_SLASH_COMMANDS

        self.assertEqual({item.text for item in top}, {command for command, _ in ALL_SLASH_COMMANDS})
        self.assertTrue(all(item.start_position == -1 for item in top))

        modes = list(completer.get_completions(Document("/mode u"), None))
        self.assertEqual([item.text for item in modes], ["ultra"])
        tab_modes = list(completer.get_completions(Document("/mode\t\tu"), None))
        self.assertEqual([item.text for item in tab_modes], ["ultra"])
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

        self.assertIn("/model", rendered)
        self.assertIn("/ide", rendered)
        self.assertIn("/keymap", rendered)
        self.assertIn("/sandbox-add-read-dir", rendered)
        self.assertIn("/mode normal", rendered)
        self.assertNotIn("/mode goal", rendered)
        self.assertIn("/mode ultra", rendered)
        self.assertIn("/settings", rendered)
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
        self.assertEqual(prompts, ["GA3BAD [NORMAL]> "])

        console.set_mode("plan")
        self.assertEqual(console.prompt(), "/status")
        self.assertEqual(prompts[-1], "GA3BAD [NORMAL]> ")

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

    def test_sparse_status_keeps_long_goal_and_input_on_single_bounded_rows(self):
        view = DashboardView(
            objective="Build a detailed animation " + "with many requirements " * 20,
            status="paused",
            waiting_question="The planner needs guidance " + "before retrying " * 20,
        )
        with mock.patch("agent.ui.shutil.get_terminal_size") as terminal_size:
            terminal_size.return_value.columns = 64
            rendered = render_status(view)

        self.assertTrue(all(len(line) <= 64 for line in rendered.splitlines()))
        self.assertEqual(sum(line.startswith("goal") for line in rendered.splitlines()), 1)
        self.assertEqual(sum(line.startswith("input") for line in rendered.splitlines()), 1)

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

    def test_live_agents_view_is_read_only_and_uses_stable_node_numbers(self):
        rendered = render_agents(
            [
                {
                    "id": "agent-1",
                    "role": "coder",
                    "status": "running",
                    "phase": "implement",
                    "work_node_id": "node-api",
                    "model": "gemma4:e4b",
                }
            ],
            nodes=[
                {
                    "id": "node-domain",
                    "title": "Domain",
                    "status": "completed",
                    "depth": 1,
                },
                {
                    "id": "node-api",
                    "title": "Appointment API",
                    "status": "running",
                    "depth": 1,
                },
            ],
            node_titles={"node-api": "Appointment API"},
            run_id="ultra-backend",
        )
        self.assertIn("Swarm observer | READ ONLY", rendered)
        self.assertIn("[02] Appointment API", rendered)
        self.assertIn("/agent NUMBER|NODE_ID|AGENT_ID", rendered)

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
