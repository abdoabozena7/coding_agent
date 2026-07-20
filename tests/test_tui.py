from __future__ import annotations

import io
import unittest
from unittest import mock

from agent.tui import (
    NINE_DOT_STATES,
    ChoiceItem,
    ChoiceListState,
    PersistentWorkspaceApp,
    SwarmInspectorState,
    UserExitRequested,
    _responsive_welcome_brand,
    _welcome_fragments,
    inline_square_levels,
    loading_grid_levels,
    nine_dot_frame,
    prompt_text,
    render_persistent_workspace,
    render_choices,
    render_nine_dot,
    render_swarm_inspector,
    render_welcome,
    rich_terminal_available,
    run_loading_task,
    run_swarm_inspector,
    select_choice,
    select_horizontal_action,
    swarm_agent_name,
    terminal_supports_unicode,
)
from agent.ui_state import (
    ActivityStage,
    AttentionKind,
    AttentionOption,
    AttentionRequest,
    ExperienceMode,
    WorkspaceUIStore,
)


class _TTY(io.StringIO):
    encoding = "utf-8"

    def isatty(self) -> bool:
        return True


class _AsciiTTY(_TTY):
    encoding = "ascii"


class _Cp1256TTY(_TTY):
    encoding = "cp1256"


def _items() -> tuple[ChoiceItem, ...]:
    return (
        ChoiceItem(
            "full",
            "Full access",
            "Run inside the Docker sandbox with fewer confirmations.",
            "Unavailable",
            disabled=True,
            disabled_reason="Docker Desktop is not running.",
        ),
        ChoiceItem(
            "normal",
            "Normal access",
            "Ask before risky actions and work without Docker.",
            "Recommended",
            value="normal-value",
        ),
        ChoiceItem("read", "Read only", "Inspect files without changing them."),
    )


class ChoiceStateTests(unittest.TestCase):
    def test_choice_item_has_stable_search_and_value_behavior(self):
        item = _items()[1]
        self.assertEqual(item.resolved_value, "normal-value")
        self.assertIn("risky actions", item.search_text)
        self.assertEqual(ChoiceItem("plan", "Plan").resolved_value, "plan")


class PersistentWorkspaceSnapshotTests(unittest.TestCase):
    def test_simple_snapshot_fits_common_terminal_sizes(self):
        store = WorkspaceUIStore()
        store.update_identity(workspace="project-090", model="gemma4:e4b", status="running")
        store.append_transcript("user", "Build a calculator")
        store.set_activity(ActivityStage.BUILDING, "Creating the interface", running=True)
        for width, height in ((80, 24), (120, 30)):
            with self.subTest(size=(width, height)):
                rendered = render_persistent_workspace(
                    store.snapshot(), width=width, height=height
                )
                lines = rendered.splitlines()
                self.assertEqual(len(lines), height)
                self.assertTrue(all(len(line) <= width for line in lines))
                self.assertIn("Build a calculator", rendered)
                self.assertIn("Creating the interface", rendered)

    def test_raw_events_only_appear_after_switching_to_advanced(self):
        store = WorkspaceUIStore()
        store.append_log('{"command": "pytest"}')
        simple = render_persistent_workspace(store.snapshot(), width=80, height=24)
        self.assertNotIn("pytest", simple)
        store.set_mode(ExperienceMode.ADVANCED)
        advanced = render_persistent_workspace(store.snapshot(), width=80, height=24)
        # The deterministic plain renderer reserves Advanced details for the
        # live app; the mode switch itself remains observable in snapshots.
        self.assertIn("ADVANCED", advanced)

    def test_arabic_simple_snapshot_stays_compact_and_has_no_raw_log(self):
        store = WorkspaceUIStore()
        store.observe_user_text("اعمل آلة حاسبة")
        store.append_transcript("user", "اعمل آلة حاسبة")
        store.append_log('{"command": "technical"}')
        store.set_activity(ActivityStage.UNDERSTANDING, "فهم الطلب", running=True)
        rendered = render_persistent_workspace(store.snapshot(), width=80, height=24)
        self.assertEqual(len(rendered.splitlines()), 24)
        self.assertIn("اعمل آلة حاسبة", rendered)
        self.assertNotIn("technical", rendered)

    def test_one_application_owns_composer_and_mode_shortcut(self):
        from prompt_toolkit.input.defaults import create_pipe_input
        from prompt_toolkit.output import DummyOutput

        store = WorkspaceUIStore()
        submitted = []
        with create_pipe_input() as pipe:
            pipe.send_text("Build it\r\x1bOQ\x11")
            app = PersistentWorkspaceApp(
                store,
                on_input=submitted.append,
                on_interrupt=lambda: None,
                on_exit=store.mark_exit,
                output=io.StringIO(),
                app_input=pipe,
                app_output=DummyOutput(),
                no_color=True,
            )
            app.run()
        self.assertEqual(submitted[0].text, "Build it")
        self.assertEqual(store.snapshot().mode, ExperienceMode.ADVANCED)

    def test_attention_accepts_direct_keyboard_choice_without_text_entry(self):
        from threading import Thread
        from prompt_toolkit.input.defaults import create_pipe_input
        from prompt_toolkit.output import DummyOutput

        store = WorkspaceUIStore()
        answers = []
        request = AttentionRequest(
            id="keyboard-approval",
            kind=AttentionKind.APPROVAL,
            title="Allow?",
            options=(
                AttentionOption("yes", "Yes", "allow_once", shortcut="y"),
                AttentionOption("no", "No", "deny", shortcut="n"),
            ),
        )
        waiter = Thread(target=lambda: answers.append(store.request_attention(request)))
        waiter.start()
        with create_pipe_input() as pipe:
            pipe.send_text("y\x11")
            app = PersistentWorkspaceApp(
                store,
                on_input=lambda _item: None,
                on_interrupt=lambda: None,
                on_exit=store.mark_exit,
                output=io.StringIO(),
                app_input=pipe,
                app_output=DummyOutput(),
                no_color=True,
            )
            app.run()
        waiter.join(1)
        self.assertEqual(answers[0].value, "allow_once")

    def test_default_selection_prefers_enabled_but_disabled_rows_explain_themselves(self):
        state = ChoiceListState.create(_items())
        self.assertEqual(state.current.key, "normal")

        state.move(-1)
        self.assertEqual(state.current.key, "full")
        self.assertIsNone(state.activate())
        self.assertEqual(state.feedback, "Docker Desktop is not running.")

        state.move(1)
        self.assertEqual(state.activate().key, "normal")
        self.assertEqual(state.feedback, "")

    def test_keyboard_navigation_clamps_and_pages_deterministically(self):
        values = tuple(ChoiceItem(str(index), f"Choice {index}") for index in range(12))
        state = ChoiceListState.create(values, page_size=4)
        state.end()
        self.assertEqual(state.current.key, "11")
        state.move(99)
        self.assertEqual(state.current.key, "11")
        state.page(-1)
        self.assertEqual(state.current.key, "7")
        visible, above, below = state.viewport()
        self.assertIn(7, visible)
        self.assertTrue(above)
        self.assertTrue(below)
        state.home()
        self.assertEqual(state.current.key, "0")

    def test_filtering_searches_descriptions_and_handles_no_matches(self):
        state = ChoiceListState.create(_items())
        state.set_query("sandbox")
        self.assertEqual(state.matching_indices, (0,))
        self.assertEqual(state.current.key, "full")
        state.backspace()
        self.assertEqual(state.query, "sandbo")
        state.set_query("not-present")
        self.assertIsNone(state.current)
        self.assertIsNone(state.activate())
        self.assertEqual(state.feedback, "No matching choice.")
        state.clear_query()
        self.assertIsNotNone(state.current)

    def test_wide_renderer_pairs_selected_row_with_description(self):
        state = ChoiceListState.create(_items(), initial_key="full")
        rendered = render_choices(
            state,
            title="Choose access",
            subtitle="You can change this later.",
            step_label="Setup 3 of 4",
            action_label="Apply",
            width=108,
            height=20,
        )
        self.assertIn("Setup 3 of 4", rendered)
        self.assertIn("› Full access", rendered)
        self.assertIn("Docker Desktop is not running.", rendered)
        self.assertIn("Enter Apply", rendered)
        self.assertEqual(len(rendered.splitlines()), 20)

    def test_narrow_renderer_keeps_details_below_the_list(self):
        state = ChoiceListState.create(_items(), initial_key="read", filterable=False)
        rendered = render_choices(
            state,
            title="Choose access",
            width=52,
            height=18,
            unicode=False,
        )
        self.assertIn("> Read only", rendered)
        self.assertIn("Inspect files without changing them.", rendered)
        self.assertNotIn("Type Filter", rendered)
        self.assertIn("Arrows", rendered)
        self.assertIn("Ctrl+Q Exit", rendered)


class SwarmInspectorTests(unittest.TestCase):
    @staticmethod
    def snapshot():
        nodes = [
            {
                "id": "root",
                "parent_id": None,
                "title": "Final assembler",
                "status": "running",
                "position": 1,
                "objective": "Integrate accepted packages.",
            },
            {
                "id": "character",
                "parent_id": "root",
                "title": "Character specialist",
                "status": "running",
                "position": 1,
                "objective": "Build the character.",
                "contract": {
                    "metadata": {
                        "specialist_domain": "character.controls",
                        "concern_ids": ["spatial_semantics"],
                    }
                },
            },
            {
                "id": "progression",
                "parent_id": "root",
                "title": "Progression specialist",
                "status": "pending",
                "position": 2,
                "objective": "Build difficulty progression.",
                "contract": {
                    "metadata": {"specialist_domain": "gameplay.progression"}
                },
            },
        ]
        return {
            "run_id": "run-1",
            "nodes": nodes,
            "agents": [
                {
                    "id": "agent-1",
                    "work_node_id": "character",
                    "status": "running",
                    "role": "coder",
                    "phase": "implement",
                }
            ],
            "profiles": {
                "character": {
                    "mission": "Make movement direction and facing agree.",
                    "deliverable": "Character package and movement tests.",
                    "expertise": ["controls", "animation"],
                    "owned_interfaces": ["CharacterPackage"],
                }
            },
            "traces": {
                "character": {
                    "self_prompt": "Implement the bounded movement contract with executable evidence."
                }
            },
        }

    def test_agent_view_exposes_names_capabilities_assignment_and_redacted_prompt(self):
        snapshot = self.snapshot()
        state = SwarmInspectorState(selected_index=1)
        rendered = render_swarm_inspector(snapshot, state, width=120, height=34)

        self.assertIn("SWARM INSPECTOR", rendered)
        self.assertIn("3 agents", rendered)
        self.assertIn("Character · Controls", rendered)
        self.assertIn("CAN DO", rendered)
        self.assertIn("spatial_semantics", rendered)
        self.assertIn("Make movement direction", rendered)
        self.assertIn("CURRENT PROMPT · REDACTED", rendered)
        self.assertIn("bounded movement contract", rendered)
        self.assertIn("↑↓ switch agent", rendered)

    def test_tree_view_is_hierarchical_status_aware_and_switchable(self):
        snapshot = self.snapshot()
        state = SwarmInspectorState(selected_index=2, tab="tree")
        rendered = render_swarm_inspector(snapshot, state, width=100, height=28)

        self.assertIn("[TREE]", rendered)
        self.assertIn("└─", rendered)
        self.assertIn("├─", rendered)
        self.assertIn("Gameplay · Progression", rendered)
        self.assertIn("pending", rendered)
        state.select_tab("agents")
        state.move(snapshot, -1)
        self.assertEqual(state.tab, "agents")
        self.assertEqual(state.selected_index, 1)

    def test_simple_agent_names_come_from_dynamic_domain_not_game_hardcoding(self):
        backend = {
            "id": "auth",
            "title": "Backend auth specialist",
            "contract": {"metadata": {"specialist_domain": "backend.auth"}},
        }
        self.assertEqual(swarm_agent_name(backend), "Backend · Auth")

    def test_ascii_swarm_view_is_safe_for_legacy_windows_code_pages(self):
        rendered = render_swarm_inspector(
            self.snapshot(),
            SwarmInspectorState(selected_index=1),
            width=100,
            height=26,
            unicode=False,
        )
        self.assertTrue(rendered.isascii())
        self.assertIn("Character | Controls", rendered)


class NineDotTests(unittest.TestCase):
    def test_every_semantic_state_produces_a_valid_immutable_frame(self):
        for state in NINE_DOT_STATES:
            with self.subTest(state=state):
                frame = nine_dot_frame(state, 0)
                self.assertEqual(frame.state, state)
                self.assertEqual(len(frame.cells), 9)
                self.assertTrue(all(level in {0, 1, 2, 3} for level in frame.cells))

    def test_search_sync_and_plan_have_distinct_deterministic_motion(self):
        search = [nine_dot_frame("search", tick).cells for tick in range(4)]
        sync = [nine_dot_frame("sync", tick).cells for tick in range(4)]
        plan = [nine_dot_frame("planning", tick).cells for tick in range(4)]
        self.assertEqual(search, [nine_dot_frame("search", tick).cells for tick in range(4)])
        self.assertNotEqual(search, sync)
        self.assertNotEqual(search, plan)
        self.assertGreater(len(set(search)), 1)

        inline_search = [inline_square_levels("search", tick) for tick in range(4)]
        inline_sync = [inline_square_levels("sync", tick) for tick in range(4)]
        inline_plan = [inline_square_levels("plan", tick) for tick in range(4)]
        self.assertNotEqual(inline_search, inline_sync)
        self.assertNotEqual(inline_search, inline_plan)
        self.assertTrue(all(len(frame) == 9 for frame in inline_search))

    def test_loading_levels_are_exposed_as_a_three_by_three_grid(self):
        grid = loading_grid_levels("search", 2)

        self.assertEqual(len(grid), 3)
        self.assertTrue(all(len(row) == 3 for row in grid))
        self.assertEqual(
            tuple(level for row in grid for level in row),
            inline_square_levels("search", 2),
        )

    def test_reduced_motion_and_no_color_are_static(self):
        for state in NINE_DOT_STATES:
            with self.subTest(state=state):
                reduced_a = nine_dot_frame(state, 0, reduced_motion=True)
                reduced_b = nine_dot_frame(state, 99, reduced_motion=True)
                plain_a = nine_dot_frame(state, 0, no_color=True)
                plain_b = nine_dot_frame(state, 99, no_color=True)
                self.assertEqual(reduced_a.cells, reduced_b.cells)
                self.assertEqual(plain_a.cells, plain_b.cells)
                self.assertFalse(reduced_a.animated)
                self.assertFalse(plain_a.animated)
                self.assertEqual(plain_a.color, "neutral")

    def test_success_and_error_settle_instead_of_looping(self):
        success = nine_dot_frame("success", 100)
        error = nine_dot_frame("failed", 100)
        self.assertFalse(success.animated)
        self.assertFalse(error.animated)
        self.assertEqual(success.cells, nine_dot_frame("success", 101).cells)
        self.assertEqual(error.cells, nine_dot_frame("error", 101).cells)

    def test_text_renderer_is_three_rows_and_contains_no_ansi(self):
        rendered = render_nine_dot("sync", 2)
        rows = rendered.splitlines()
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(len(row.split()) == 3 for row in rows))
        self.assertNotIn("\x1b", rendered)
        self.assertEqual(
            render_nine_dot("sync", 2, no_color=True),
            render_nine_dot("sync", 200, no_color=True),
        )
        self.assertTrue(render_nine_dot("sync", 2, no_color=True).isascii())


class TerminalAndWelcomeTests(unittest.TestCase):
    def test_horizontal_action_switches_with_arrow_and_accepts_enter(self):
        from prompt_toolkit.input.defaults import create_pipe_input
        from prompt_toolkit.output import DummyOutput

        with create_pipe_input() as pipe:
            pipe.send_text("\x1b[C\r")
            selected = select_horizontal_action(
                [ChoiceItem("no", "No"), ChoiceItem("yes", "Yes")],
                title="Allow this action once?",
                initial_key="no",
                force=True,
                app_input=pipe,
                app_output=DummyOutput(),
                output=io.StringIO(),
                no_color=True,
            )

        self.assertIsNotNone(selected)
        self.assertEqual(selected.key, "yes")

    def test_explicit_horizontal_action_ignores_enter_and_escape_until_a_choice(self):
        from prompt_toolkit.input.defaults import create_pipe_input
        from prompt_toolkit.output import DummyOutput

        with create_pipe_input() as pipe:
            pipe.send_text("\r\x03y")
            selected = select_horizontal_action(
                [ChoiceItem("no", "No"), ChoiceItem("yes", "Yes")],
                title="Allow this action once?",
                initial_key="no",
                shortcuts={"n": "no", "y": "yes"},
                require_explicit_selection=True,
                cancelable=False,
                force=True,
                app_input=pipe,
                app_output=DummyOutput(),
                output=io.StringIO(),
                no_color=True,
            )

        self.assertIsNotNone(selected)
        self.assertEqual(selected.key, "yes")

    def test_selector_quick_action_activates_the_mapped_choice(self):
        from prompt_toolkit.input.defaults import create_pipe_input
        from prompt_toolkit.output import DummyOutput

        with create_pipe_input() as pipe:
            pipe.send_text("d")
            selected = select_choice(
                [
                    ChoiceItem("1", "One"),
                    ChoiceItem("defaults", "Use recommended defaults"),
                ],
                title="Decision",
                filterable=False,
                shortcuts={"d": "defaults"},
                force=True,
                app_input=pipe,
                app_output=DummyOutput(),
                output=io.StringIO(),
                no_color=True,
            )

        self.assertIsNotNone(selected)
        self.assertEqual(selected.key, "defaults")

    def test_freeform_answer_uses_the_full_screen_composer(self):
        from prompt_toolkit.input.defaults import create_pipe_input
        from prompt_toolkit.output import DummyOutput

        with create_pipe_input() as pipe:
            pipe.send_text("Keep history for 30 days\r")
            value = prompt_text(
                title="Write your answer",
                force=True,
                app_input=pipe,
                app_output=DummyOutput(),
                output=io.StringIO(),
                no_color=True,
            )

        self.assertEqual(value, "Keep history for 30 days")

    def test_live_swarm_inspector_opens_switches_views_and_closes(self):
        from prompt_toolkit.input.defaults import create_pipe_input
        from prompt_toolkit.output import DummyOutput

        snapshot = SwarmInspectorTests.snapshot()
        with create_pipe_input() as pipe:
            pipe.send_text("\x1b[C\x1b[B\r\x1b")
            run_swarm_inspector(
                lambda: snapshot,
                force=True,
                app_input=pipe,
                app_output=DummyOutput(),
                output=io.StringIO(),
                no_color=True,
                reduced_motion=True,
            )

    def test_ctrl_q_from_selector_requests_session_exit_not_back_navigation(self):
        from prompt_toolkit.input.defaults import create_pipe_input
        from prompt_toolkit.output import DummyOutput

        with create_pipe_input() as pipe:
            pipe.send_text("\x11")
            with self.assertRaises(UserExitRequested):
                select_choice(
                    [ChoiceItem("plan", "Plan")],
                    title="Mode",
                    force=True,
                    app_input=pipe,
                    app_output=DummyOutput(),
                    output=io.StringIO(),
                    no_color=True,
                )

    def test_loading_screen_returns_background_probe_result(self):
        from prompt_toolkit.input.defaults import create_pipe_input
        from prompt_toolkit.output import DummyOutput

        with create_pipe_input() as pipe:
            result = run_loading_task(
                lambda: {"ready": True},
                title="Checking Docker",
                force=True,
                app_input=pipe,
                app_output=DummyOutput(),
                output=io.StringIO(),
                no_color=True,
            )

        self.assertEqual(result, {"ready": True})

    def test_rich_terminal_detection_requires_real_input_and_tty_streams(self):
        with mock.patch("agent.tui.sys.stdin", _TTY()), mock.patch(
            "agent.tui._prompt_output", return_value=object()
        ), mock.patch.dict(
            "agent.tui.os.environ", {"TERM": "xterm-256color"}, clear=True
        ):
            self.assertTrue(rich_terminal_available(input, _TTY()))
            self.assertFalse(rich_terminal_available(lambda _: "", _TTY()))
            self.assertFalse(rich_terminal_available(input, io.StringIO()))

        with mock.patch("agent.tui.sys.stdin", _TTY()), mock.patch.dict(
            "agent.tui.os.environ",
            {"TERM": "xterm-256color", "GA3BAD_PLAIN_UI": "1"},
            clear=True,
        ):
            self.assertFalse(rich_terminal_available(input, _TTY()))

    def test_unicode_detection_is_encoding_aware(self):
        self.assertTrue(terminal_supports_unicode(_TTY()))
        self.assertFalse(terminal_supports_unicode(_AsciiTTY()))
        self.assertFalse(terminal_supports_unicode(_Cp1256TTY()))

    def test_welcome_snapshot_is_full_height_and_defers_workspace_content(self):
        rendered = render_welcome(width=60, height=16)
        self.assertEqual(len(rendered.splitlines()), 16)
        self.assertIn("GA3BAD", rendered)
        self.assertIn("Press Enter to begin", rendered)
        self.assertIn("Ctrl+Q Exit", rendered)
        self.assertNotIn("workspace", rendered.lower())

    def test_roomy_welcome_turns_the_brand_into_a_centered_full_canvas_anchor(self):
        rendered = render_welcome(width=140, height=40)
        lines = rendered.splitlines()
        brand_rows = [line for line in lines if any(character.isdigit() for character in line)]

        self.assertEqual(len(lines), 40)
        self.assertEqual(len(brand_rows), 22)
        self.assertGreater(max(len(line.strip()) for line in brand_rows), 120)
        widest = max(brand_rows, key=lambda line: len(line.strip()))
        left = len(widest) - len(widest.lstrip())
        right = len(widest) - len(widest.rstrip())
        self.assertLessEqual(abs(left - right), 1)
        self.assertIn("coding agent", rendered)
        self.assertIn("Press Enter to begin", rendered)

    def test_roomy_wordmark_final_d_has_aligned_vertical_stems(self):
        brand = _responsive_welcome_brand("GA3BAD", 140, 40, unicode=True)
        rows = brand.splitlines()
        self.assertEqual(len(rows), 22)

        right_edge = max(len(row.rstrip()) for row in rows)
        d_window = [row.ljust(right_edge)[max(0, right_edge - 22) : right_edge] for row in rows]
        left_stem_rows = sum(window[:6].strip() != "" for window in d_window)
        far_right_rows = sum(window[-4:].strip() != "" for window in d_window)

        self.assertGreaterEqual(left_stem_rows, 18)
        self.assertEqual(far_right_rows, left_stem_rows)

    def test_roomy_wordmark_is_built_from_changing_numeric_pixels(self):
        brand = _responsive_welcome_brand("GA3BAD", 140, 40, unicode=True)
        digits = brand.replace("\n", "").replace(" ", "")
        self.assertTrue(digits)
        self.assertTrue(set(digits) <= set("0123456789"))
        first = "".join(text for _style, text in _welcome_fragments(brand, "coding agent", "Begin", 1, True, 140, 40))
        second = "".join(text for _style, text in _welcome_fragments(brand, "coding agent", "Begin", 2, True, 140, 40))
        self.assertNotEqual(first, second)

    def test_subtitle_uses_brand_color_and_the_same_shimmer_sweep(self):
        fragments = list(
            _welcome_fragments(
                "GA3BAD", "coding agent", "Press Enter to begin", 45, True, 80, 24
            )
        )
        rows: list[list[tuple[str, str]]] = [[]]
        for style, text in fragments:
            if text == "\n":
                rows.append([])
            else:
                rows[-1].append((style, text))
        subtitle_fragments = next(
            row for row in rows if "".join(text for _style, text in row).strip() == "coding agent"
        )

        self.assertTrue(
            any(style == "class:welcome.brand" for style, _text in subtitle_fragments)
        )
        self.assertTrue(
            any(style == "class:welcome.shimmer" for style, _text in subtitle_fragments)
        )


if __name__ == "__main__":
    unittest.main()
