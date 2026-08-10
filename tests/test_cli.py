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
    _interactive_setup,
    _latest_recoverable_session_id,
    build_parser,
    choose_access_level,
    choose_interaction_mode,
    choose_project_protection,
    choose_workspace,
    execute_command,
    interactive_loop,
    main,
)
from agent.commands import (
    CommandKind,
    CommandParseError,
    InternalActionKind,
    UnknownCommandParseError,
    internal_action,
    parse_command,
)
from agent.config import InteractionMode, RuntimeConfig, SessionPreferences
from agent.models import GoalStatus
from agent.store import StateStore
from agent.testing import ScriptedProvider
from agent.tui import WorkspaceInput
from agent.ui import ConsoleUI, DashboardView, WorkspaceRefreshRequested
from agent.ui_state import WorkspaceUIStore
from agent.version_control import GitProtectionManager


class _TTY(io.StringIO):
    encoding = "utf-8"

    def isatty(self) -> bool:
        return True


class CLITests(unittest.TestCase):
    def test_explicit_resume_prefers_goal_owning_session_over_newer_empty_default(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            with StateStore(workspace) as store:
                store.save_workflow_session(
                    "workspace-active",
                    goal_id=None,
                    session_mode="normal",
                    plan_state="none",
                    run_state="idle",
                    state={},
                )
                store.create_goal_and_bind_workflow_session(
                    "resume this durable goal",
                    session_id="workspace-active",
                )
                # Reproduce project-136: a later command created/refreshed the
                # empty default row after the real goal-owning session.
                store.save_workflow_session(
                    "workspace-session",
                    goal_id=None,
                    session_mode="normal",
                    plan_state="none",
                    run_state="idle",
                    state={},
                )

            self.assertEqual(
                _latest_recoverable_session_id(workspace),
                "workspace-active",
            )

    def test_explicit_resume_recovers_legacy_session_with_missing_goal_projection(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            with StateStore(workspace) as store:
                store.save_workflow_session(
                    "workspace-legacy",
                    goal_id=None,
                    session_mode="normal",
                    plan_state="none",
                    run_state="idle",
                    state={},
                )
                store.create_goal_and_bind_workflow_session(
                    "resume the legacy durable goal",
                    session_id="workspace-legacy",
                )
                # Reproduce project-135: the authoritative Goal retains its
                # session_id while the denormalized session projection is null.
                store.save_workflow_session(
                    "workspace-legacy",
                    goal_id=None,
                    session_mode="normal",
                    plan_state="none",
                    run_state="planning",
                    state={},
                )
                store.save_workflow_session(
                    "workspace-session",
                    goal_id=None,
                    session_mode="normal",
                    plan_state="none",
                    run_state="idle",
                    state={},
                )

            self.assertEqual(
                _latest_recoverable_session_id(workspace),
                "workspace-legacy",
            )

    def test_new_session_flag_creates_an_empty_thread_instead_of_resuming_default(self):
        with tempfile.TemporaryDirectory() as directory:
            with StateStore(directory) as store:
                store.save_workflow_session(
                    "workspace-session",
                    goal_id=None,
                    session_mode="normal",
                    plan_state="none",
                    run_state="idle",
                    state={
                        "model_snapshot": {
                            "provider": "scripted",
                            "model": "scripted",
                            "execution_class": "local",
                        },
                    },
                )
                store.create_goal_and_bind_workflow_session(
                    "an old request",
                    session_id="workspace-session",
                )
            with mock.patch(
                "agent.cli.ModelDescriptor.create_provider",
                return_value=ScriptedProvider([]),
            ):
                code = main(
                    [
                        "--workspace", directory,
                        "--new-session",
                        "--provider", "ollama",
                        "--model", "gemma4:e4b",
                        "--command", "/help",
                        "--plain",
                        "--no-color",
                    ]
                )

            self.assertEqual(code, 0)
            with StateStore(directory) as store:
                sessions = store.list_workflow_sessions(limit=20)

            fresh = [item for item in sessions if item["id"] != "workspace-session"]
            self.assertEqual(len(fresh), 1)
            self.assertTrue(fresh[0]["id"].startswith("workspace-"))
            self.assertIsNone(fresh[0]["goal_id"])
            self.assertEqual(
                fresh[0]["state"]["model_snapshot"]["provider"],
                "ollama",
            )

    def test_reopening_project_reuses_saved_setup_without_reasking(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            store = StateStore(workspace)
            store.save_workflow_session(
                "workspace-session",
                goal_id=None,
                session_mode="normal",
                plan_state="none",
                run_state="idle",
                state={
                    "model_snapshot": {
                        "provider": "ollama",
                        "model": "gemma4:e4b",
                        "execution_class": "local",
                        "capabilities": ["tools", "structured_output"],
                    },
                    "access_level": "normal",
                    "concurrency": 1,
                    "interaction_mode": "working",
                },
            )
            store.close()
            GitProtectionManager(workspace).configure(
                auto_checkpoint=False,
                auto_push=False,
                provider="snapshot",
            )
            args = build_parser().parse_args(
                ["--workspace", str(workspace), "--provider", "ollama", "--plain", "--no-color"]
            )
            output = io.StringIO()
            console = ConsoleUI(
                stream=output,
                color=False,
                input_func=lambda prompt: (_ for _ in ()).throw(
                    AssertionError(f"setup unexpectedly prompted: {prompt}")
                ),
            )
            setup = _interactive_setup(args, console, "ollama")

            self.assertIsNotNone(setup)
            assert setup is not None
            self.assertEqual(setup[1].id.split("@", 1)[0], "ollama:gemma4:e4b")
            self.assertEqual(setup[3].value, "normal")
            self.assertEqual(setup[4].concurrency, 1)
            self.assertIn("Loaded this project's saved setup", output.getvalue())

    def test_new_interactive_session_does_not_inherit_saved_plan_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            with StateStore(workspace) as store:
                store.save_workflow_session(
                    "workspace-session",
                    goal_id=None,
                    session_mode="plan",
                    plan_state="none",
                    run_state="idle",
                    state={
                        "model_snapshot": {
                            "provider": "ollama",
                            "model": "gemma4:e4b",
                            "execution_class": "local",
                            "capabilities": ["tools"],
                        },
                        "access_level": "normal",
                        "concurrency": 1,
                        "interaction_mode": "plan",
                    },
                )
            GitProtectionManager(workspace).configure(
                auto_checkpoint=False,
                auto_push=False,
                provider="snapshot",
            )
            args = build_parser().parse_args(
                [
                    "--workspace", str(workspace), "--new-session",
                    "--provider", "ollama", "--plain", "--no-color",
                ]
            )
            console = ConsoleUI(
                stream=io.StringIO(),
                color=False,
                input_func=lambda prompt: (_ for _ in ()).throw(
                    AssertionError(f"setup unexpectedly prompted: {prompt}")
                ),
            )

            setup = _interactive_setup(args, console, "ollama")

            self.assertIsNotNone(setup)
            assert setup is not None
            self.assertIs(setup[4].mode, InteractionMode.NORMAL)

    def test_strongest_local_failover_skips_failed_model_aliases(self):
        from agent.cli import _strongest_local_model
        from agent.model_catalog import ExecutionClass

        strongest = SimpleNamespace(
            id="ollama:gemma4:e4b",
            provider="ollama",
            model="gemma4:e4b",
            execution_class=ExecutionClass.LOCAL,
            supports_tools=True,
            metadata={"parameter_size": "8B", "capability_band": "high"},
        )
        next_best = SimpleNamespace(
            id="ollama:qwen2.5-coder:7b",
            provider="ollama",
            model="qwen2.5-coder:7b",
            execution_class=ExecutionClass.LOCAL,
            supports_tools=True,
            metadata={"parameter_size": "7B", "capability_band": "medium"},
        )

        self.assertIs(
            _strongest_local_model((next_best, strongest)),
            strongest,
        )
        self.assertIs(
            _strongest_local_model((next_best, strongest), excluded={"gemma4:e4b"}),
            next_best,
        )
        self.assertIsNone(
            _strongest_local_model(
                (next_best, strongest),
                excluded={"ollama:gemma4:e4b", "qwen2.5-coder:7b"},
            )
        )

    def test_working_tui_prompt_stays_terminal_only(self):
        import time

        from agent.cli import _persistent_interactive_loop

        opened = Event()
        executed = Event()
        captured = {}

        class FakeApp:
            def __init__(self, _store, *, on_input, on_exit, **_kwargs):
                captured["store"] = _store
                self.on_input = on_input
                self.on_exit = on_exit
                self.overlay_kind = ""

            def run(self):
                self.on_input(
                    WorkspaceInput(
                        text="Build a local-first Three.js calculator",
                    )
                )
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    if executed.is_set():
                        break
                    time.sleep(0.01)
                self.on_exit()

            def stop(self):
                return None

            def restore_composer(self, *_args, **_kwargs):
                return None

            def open_details(self, *_args, **_kwargs):
                return None

            def open_swarm(self, *_args, **_kwargs):
                return None

            def update_swarm(self, *_args, **_kwargs):
                return None

        runtime = mock.Mock()
        runtime.workspace = Path("workspace")
        runtime.session_id = "local-first-session"
        runtime.model_name = "gemma4:e4b"
        runtime.execution_class = "local"
        runtime.interaction_mode = SimpleNamespace(value="working")
        runtime.active_goal.return_value = None
        runtime.dashboard.return_value = SimpleNamespace(
            status="idle",
            tasks=(),
            objective="",
            goal_id="",
            plan_revision=0,
        )
        runtime.store.list_timeline_entries.return_value = []
        runtime.store.count_queued_prompts.return_value = 0
        runtime.store.list_queued_prompts.return_value = []
        runtime.store.claim_next_prompt.return_value = None
        runtime.store.get_workflow_session.return_value = {"state": {}}
        runtime.store.get_accepted_plan.return_value = None
        runtime.local_web_server.take_execution_request.return_value = False
        runtime.version_control.diff.return_value = ""
        console = mock.Mock()
        console.stream = io.StringIO()
        console.color = False

        with mock.patch("agent.cli.PersistentWorkspaceApp", FakeApp), mock.patch(
            "agent.cli.TelemetrySampler"
        ) as telemetry, mock.patch(
            "agent.cli.question_session", return_value=None
        ), mock.patch(
            "agent.cli._current_ultra_run", return_value=None
        ), mock.patch(
            "agent.cli._open_local_web_view",
            side_effect=lambda *_args: (
                captured.__setitem__("opened_after_execute", executed.is_set()),
                opened.set(),
            ),
        ) as open_view, mock.patch(
            "agent.cli.execute_command", side_effect=lambda *_args: executed.set() or True
        ) as execute:
            telemetry.return_value.start.return_value = None
            telemetry.return_value.stop.return_value = None
            _persistent_interactive_loop(runtime, console, SessionPreferences())

        self.assertTrue(executed.is_set())
        self.assertFalse(opened.is_set())
        self.assertNotIn("opened_after_execute", captured)
        open_view.assert_not_called()
        execute.assert_called_once()

    def test_working_plan_approval_offers_terminal_actions_and_web_review(self):
        import time

        from agent.cli import _persistent_interactive_loop

        web_opened = Event()
        captured = {}

        class FakeApp:
            def __init__(self, _store, *, on_exit, **_kwargs):
                captured["store"] = _store
                self.on_exit = on_exit
                self.overlay_kind = ""

            def run(self):
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    request = captured["store"].active_attention()
                    if request is not None:
                        captured["options"] = [option.value for option in request.options]
                        captured["default"] = request.default_key
                        captured["cancel"] = request.cancel_key
                        captured["store"].resolve_attention("web")
                        break
                    time.sleep(0.01)
                web_opened.wait(3)
                self.on_exit()

            def stop(self):
                return None

            def close_overlay(self):
                return None

            def restore_composer(self, *_args, **_kwargs):
                return None

            def open_details(self, *_args, **_kwargs):
                return None

            def open_swarm(self, *_args, **_kwargs):
                return None

            def update_swarm(self, *_args, **_kwargs):
                return None

        goal = SimpleNamespace(
            id="goal-1",
            objective="Create hello.txt",
            status=GoalStatus.AWAITING_PLAN_APPROVAL,
            metadata={},
        )
        view = SimpleNamespace(
            status="awaiting_plan_approval",
            tasks=(SimpleNamespace(title="Create hello.txt"),),
            objective="Create hello.txt",
            goal_id="goal-1",
            plan_revision=2,
            plan_summary="Create and verify exactly one file.",
        )
        runtime = mock.Mock()
        runtime.workspace = Path("workspace")
        runtime.session_id = "approval-session"
        runtime.model_name = "gemma4:e4b"
        runtime.execution_class = "local"
        runtime.interaction_mode = SimpleNamespace(value="working")
        runtime.active_goal.return_value = goal
        runtime.dashboard.return_value = view
        runtime.sleep_mode_enabled.return_value = False
        runtime.sleep_mode_policy.return_value = "off"
        runtime.store.list_timeline_entries.return_value = []
        runtime.store.count_queued_prompts.return_value = 0
        runtime.store.list_queued_prompts.return_value = []
        runtime.store.claim_next_prompt.return_value = None
        runtime.store.get_workflow_session.return_value = {"state": {}}
        runtime.store.get_accepted_plan.return_value = None
        runtime.local_web_server.take_execution_request.return_value = False
        runtime.version_control.diff.return_value = ""
        console = mock.Mock()
        console.stream = io.StringIO()
        console.color = False

        with mock.patch("agent.cli.PersistentWorkspaceApp", FakeApp), mock.patch(
            "agent.cli.TelemetrySampler"
        ) as telemetry, mock.patch(
            "agent.cli.question_session", return_value=None
        ), mock.patch(
            "agent.cli._current_ultra_run", return_value=None
        ), mock.patch(
            "agent.cli._open_local_web_view", side_effect=lambda *_args: web_opened.set()
        ) as open_view:
            telemetry.return_value.start.return_value = None
            telemetry.return_value.stop.return_value = None
            _persistent_interactive_loop(runtime, console, SessionPreferences())

        self.assertEqual(captured["options"], ["start", "open", "web", "cancel"])
        self.assertEqual(captured["default"], "cancel")
        self.assertEqual(captured["cancel"], "cancel")
        self.assertTrue(web_opened.is_set())
        open_view.assert_called_once_with(runtime, console, "plan")

    def test_completed_goal_presents_handoff_and_opens_project_folder(self):
        import time

        from agent.cli import _persistent_interactive_loop

        folder_opened = Event()
        captured = {}

        class FakeApp:
            def __init__(self, _store, *, on_exit, **_kwargs):
                captured["store"] = _store
                self.on_exit = on_exit
                self.overlay_kind = ""

            def run(self):
                first_request = ""
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    request = captured["store"].active_attention()
                    if request is None:
                        time.sleep(0.01)
                        continue
                    if not first_request:
                        first_request = request.id
                        captured["options"] = [option.value for option in request.options]
                        captured["default"] = request.default_key
                        captured["store"].resolve_attention("explorer")
                    elif request.id != first_request:
                        captured["store"].resolve_attention("dismiss")
                        break
                    time.sleep(0.01)
                folder_opened.wait(3)
                self.on_exit()

            def stop(self):
                return None

            def close_overlay(self):
                return None

            def restore_composer(self, *_args, **_kwargs):
                return None

            def open_details(self, *_args, **_kwargs):
                return None

            def open_swarm(self, *_args, **_kwargs):
                return None

            def update_swarm(self, *_args, **_kwargs):
                return None

        goal = SimpleNamespace(
            id="goal-complete",
            objective="Create hello.txt",
            status=GoalStatus.COMPLETED,
            metadata={
                "goal_change_sets": (
                    {"changed_files": ("hello.txt",)},
                ),
            },
        )
        view = SimpleNamespace(
            status="completed",
            tasks=(SimpleNamespace(status="done"),),
            objective="Create hello.txt",
            goal_id="goal-complete",
            plan_revision=1,
        )
        runtime = mock.Mock()
        runtime.workspace = Path.cwd()
        runtime.session_id = "completed-session"
        runtime.model_name = "gemma4:e4b"
        runtime.execution_class = "local"
        runtime.interaction_mode = SimpleNamespace(value="working")
        runtime.active_goal.return_value = goal
        runtime.dashboard.return_value = view
        runtime.sleep_mode_enabled.return_value = False
        runtime.sleep_mode_policy.return_value = "off"
        runtime.store.list_timeline_entries.return_value = []
        runtime.store.count_queued_prompts.return_value = 0
        runtime.store.list_queued_prompts.return_value = []
        runtime.store.claim_next_prompt.return_value = None
        runtime.store.get_workflow_session.return_value = {"state": {}}
        runtime.store.get_accepted_plan.return_value = None
        runtime.store.list_evidence.return_value = []
        runtime.local_web_server.take_execution_request.return_value = False
        runtime.version_control.diff.return_value = ""
        console = mock.Mock()
        console.stream = io.StringIO()
        console.color = False

        with mock.patch("agent.cli.PersistentWorkspaceApp", FakeApp), mock.patch(
            "agent.cli.TelemetrySampler"
        ) as telemetry, mock.patch(
            "agent.cli.question_session", return_value=None
        ), mock.patch(
            "agent.cli._current_ultra_run", return_value=None
        ), mock.patch(
            "agent.cli.tools.web_preview.list_previews", return_value=()
        ), mock.patch(
            "agent.cli.os.startfile", create=True, side_effect=lambda *_args: folder_opened.set()
        ) as startfile:
            telemetry.return_value.start.return_value = None
            telemetry.return_value.stop.return_value = None
            _persistent_interactive_loop(runtime, console, SessionPreferences())

        self.assertEqual(captured["options"], ["explorer", "changes", "dismiss"])
        self.assertEqual(captured["default"], "dismiss")
        self.assertTrue(folder_opened.is_set())
        startfile.assert_called_once()

    def test_session_recovery_backfills_model_snapshot_from_capability_envelope(self):
        with tempfile.TemporaryDirectory() as directory:
            with StateStore(directory) as store:
                store.save_workflow_session(
                    "recover-me",
                    goal_id=None,
                    session_mode="normal",
                    plan_state="none",
                    run_state="idle",
                    state={
                        "interaction_mode": "working",
                        "access_level": "normal",
                        "model_capability_envelope": {
                            "provider": "ollama",
                            "model": "gpt-oss:120b-cloud",
                            "execution_class": "cloud",
                            "capability_band": "high",
                            "parameter_count_billions": 116.8,
                            "context_window_tokens": 131072,
                            "maximum_output_tokens": None,
                            "tool_calling": False,
                            "structured_output": False,
                            "thinking": False,
                            "vision": False,
                        },
                    },
                )
            output = io.StringIO()
            with mock.patch(
                "agent.model_catalog.ModelDescriptor.create_provider",
                return_value=ScriptedProvider([]),
            ), redirect_stdout(output):
                code = main(
                    [
                        "--workspace", directory,
                        "--session", "recover-me",
                        "--provider", "ollama",
                        "--command", "/help",
                        "--plain",
                        "--no-color",
                    ]
                )
            with StateStore(directory) as store:
                recovered = store.get_workflow_session("recover-me")

        self.assertEqual(code, 0)
        self.assertEqual(
            recovered["state"]["model_snapshot"]["model"],
            "gpt-oss:120b-cloud",
        )
        self.assertEqual(
            recovered["state"]["model_snapshot"]["source"],
            "session-capability-recovery",
        )

    def test_reapplying_current_mode_is_silent(self):
        from agent.cli import _set_interaction_mode

        runtime = SimpleNamespace(
            ultra_readiness_issue=lambda: "",
            transition_mode=mock.Mock(return_value="normal"),
            active_goal=lambda: None,
        )
        console = SimpleNamespace(set_mode=mock.Mock(), write=mock.Mock())
        preferences = SimpleNamespace(mode=InteractionMode.NORMAL)

        _set_interaction_mode(
            runtime,
            console,
            preferences,
            InteractionMode.NORMAL,
            detailed=False,
        )

        console.write.assert_not_called()

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
        self.assertIn("approve", values)
        self.assertNotIn("normal", values)
        self.assertNotIn("ultra", values)
        self.assertNotIn("start", values)
        self.assertEqual(sum(item.recommended for item in request.options), 1)

        ultra_plan_request = _plan_attention(
            view,
            (),
            plan_only=True,
            ultra_available=True,
            normal_available=False,
        )
        ultra_values = {item.value for item in ultra_plan_request.options}
        self.assertIn("approve", ultra_values)
        self.assertNotIn("normal", ultra_values)
        self.assertNotIn("ultra", ultra_values)

    def test_working_plan_review_offers_terminal_approval_edit_and_web(self):
        from agent.cli import _plan_attention

        view = SimpleNamespace(
            goal_id="goal-1",
            plan_revision=3,
            plan_summary="A saved recursive plan",
            tasks=(SimpleNamespace(title="Create the requested file"),),
        )

        request = _plan_attention(view, (), plan_only=False)

        self.assertEqual(
            [item.value for item in request.options],
            ["start", "open", "web", "cancel"],
        )
        self.assertEqual(request.options[2].label, "Open in website")
        self.assertEqual(request.default_key, "cancel")
        self.assertEqual(request.cancel_key, "cancel")

    def test_completed_work_offers_project_preview_and_diff_handoff(self):
        from agent.cli import _completion_attention

        view = SimpleNamespace(
            goal_id="goal-1",
            tasks=(
                SimpleNamespace(status="done"),
                SimpleNamespace(status="skipped"),
            ),
        )

        request = _completion_attention(view, preview_available=True)

        self.assertEqual(
            [item.value for item in request.options],
            ["explorer", "preview", "changes", "dismiss"],
        )
        self.assertTrue(request.options[0].recommended)
        self.assertEqual(request.default_key, "dismiss")
        self.assertEqual(request.cancel_key, "dismiss")
        self.assertIn("2/2 planned steps complete", request.message)

        without_preview = _completion_attention(view, preview_available=False)
        self.assertEqual(
            [item.value for item in without_preview.options],
            ["explorer", "changes", "dismiss"],
        )

    def test_mode_command_changes_depth_without_starting_a_second_foundation(self):
        output = io.StringIO()
        console = ConsoleUI(stream=output, color=False)
        preferences = SessionPreferences(mode=InteractionMode.NORMAL)
        runtime = mock.Mock()
        runtime.ultra_readiness_issue.return_value = ""
        runtime.active_goal.return_value = SimpleNamespace(
            status=GoalStatus.PAUSED,
            metadata={},
        )

        self.assertTrue(
            execute_command(
                runtime,
                console,
                internal_action(InternalActionKind.MODE, mode="ultra"),
                preferences,
            )
        )

        runtime.increase_execution_depth.assert_called_once_with()
        runtime.prepare_ultra_from_existing_goal.assert_not_called()
        self.assertEqual(preferences.mode, InteractionMode.NORMAL)

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
                        captured["recovery_options"] = {
                            option.value for option in request.options
                        }
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
        self.assertNotIn("local", captured["recovery_options"])
        self.assertIn("model", captured["recovery_options"])

    def test_full_auto_retries_saved_local_semantic_boundary_without_attention(self):
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
                "id": "q-local-autopilot",
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
                provider_message="local model is reloading",
                endpoint="http://localhost:11434/api/chat",
            )
        )
        pending = {
            "turn_id": "turn-local-autopilot",
            "original_input": "Build the saved project",
            "status": "awaiting_provider",
            "stage": "goal_intake",
            "last_error": "local model is reloading",
        }
        boundary_visible = {"value": False}

        class FakeApp:
            def __init__(self, store, *, on_exit, **_kwargs):
                self.store = store
                self.on_exit = on_exit
                self.overlay_kind = ""

            def run(self):
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    if any(
                        call.args
                        and getattr(call.args[0], "kind", None) is CommandKind.RESUME
                        for call in runtime.apply_command.call_args_list
                    ):
                        self.on_exit()
                        return
                    request = self.store.active_attention()
                    if request is not None and request.kind.value != "recovery":
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
        runtime.model_name = "gemma4:e4b"
        runtime.execution_class = "local"
        runtime.session_id = "local-autopilot-session"
        runtime.interaction_mode = SimpleNamespace(value="working")
        runtime.sleep_mode_policy.return_value = "full"
        runtime.active_goal.return_value = None
        runtime.dashboard.return_value = SimpleNamespace(
            status="paused", tasks=(), objective="Build the saved project", goal_id="", plan_revision=0
        )
        runtime.store.list_timeline_entries.return_value = []
        runtime.store.count_queued_prompts.return_value = 0
        runtime.store.list_queued_prompts.return_value = []
        runtime.store.claim_next_prompt.return_value = None
        runtime.store.get_workflow_session.side_effect = lambda *_args, **_kwargs: {
            "state": (
                {"pending_semantic_turn": dict(pending)}
                if boundary_visible["value"]
                else {}
            )
        }
        runtime.local_web_server.take_execution_request.return_value = False

        def save_pending(value):
            pending.clear()
            pending.update(dict(value))

        runtime._save_pending_semantic_turn.side_effect = save_pending

        def fail_provider(*_args, **_kwargs):
            boundary_visible["value"] = True
            raise failure

        console = mock.Mock()
        console.stream = io.StringIO()
        console.color = False

        with mock.patch("agent.cli.PersistentWorkspaceApp", FakeApp), mock.patch(
            "agent.cli.TelemetrySampler"
        ) as telemetry, mock.patch(
            "agent.cli.question_session", return_value=session
        ), mock.patch(
            "agent.cli.answer_question", side_effect=fail_provider
        ), mock.patch(
            "agent.cli._full_auto_retry_delay", return_value=0.0
        ), mock.patch(
            "agent.cli._show_runtime_state"
        ), mock.patch(
            "agent.cli._current_ultra_run", return_value=None
        ):
            telemetry.return_value.start.return_value = None
            telemetry.return_value.stop.return_value = None
            _persistent_interactive_loop(runtime, console, SessionPreferences())

        self.assertGreaterEqual(runtime._save_pending_semantic_turn.call_count, 1)
        self.assertEqual(pending["status"], "awaiting_provider")
        self.assertEqual(pending["full_auto_retry_attempts"], 1)
        resume_commands = [
            call.args[0]
            for call in runtime.apply_command.call_args_list
            if call.args and getattr(call.args[0], "kind", None) is CommandKind.RESUME
        ]
        self.assertGreaterEqual(len(resume_commands), 1)
        self.assertFalse(runtime.store.request_attention.called)

    def test_full_auto_switches_cloud_failure_to_local_and_resumes(self):
        import time

        from agent.cli import _persistent_interactive_loop
        from agent.local_provider import (
            ProviderDiagnostic,
            ProviderFailureKind,
            ProviderRequestError,
        )
        from agent.model_catalog import ExecutionClass

        session = SimpleNamespace(
            source="intake",
            current={
                "id": "q-cloud-failure",
                "question": "Continue with the cloud model?",
                "options": (
                    {"label": "Yes", "description": "Continue", "recommended": True},
                ),
            },
        )
        failure = ProviderRequestError(
            ProviderDiagnostic(
                True,
                ProviderFailureKind.HTTP_4XX,
                "provider_call",
                status_code=429,
                provider_message="cloud usage limit exhausted",
                endpoint="https://provider.invalid/chat",
            )
        )

        class FakeApp:
            def __init__(self, store, *, on_exit, **_kwargs):
                self.store = store
                self.on_exit = on_exit

            def run(self):
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    if runtime.continue_with_local_model.called and runtime.resume.called:
                        self.on_exit()
                        return
                    request = self.store.active_attention()
                    if request is not None:
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
        runtime.model_name = "gpt-oss:120b-cloud"
        runtime.execution_class = "cloud"
        runtime.session_id = "cloud-autopilot-session"
        runtime.interaction_mode = SimpleNamespace(value="working")
        runtime.sleep_mode_policy.return_value = "full"
        runtime.dashboard.return_value = SimpleNamespace(
            status="paused", tasks=(), objective="Build the approved project", goal_id="goal-cloud", plan_revision=1
        )
        runtime.active_goal.return_value = SimpleNamespace(
            id="goal-cloud", status=GoalStatus.PAUSED, metadata={}, active_plan_revision=1
        )
        runtime.store.list_timeline_entries.return_value = []
        runtime.store.count_queued_prompts.return_value = 0
        runtime.store.list_queued_prompts.return_value = []
        runtime.store.claim_next_prompt.return_value = None
        runtime.store.get_workflow_session.return_value = {"state": {}}
        runtime.store.get_accepted_plan.return_value = SimpleNamespace(fingerprint="accepted-plan")
        runtime.local_web_server.take_execution_request.return_value = False
        console = mock.Mock()
        console.stream = io.StringIO()
        console.color = False

        local_descriptor = SimpleNamespace(
            id="ollama:coder",
            provider="ollama",
            model="coder",
            display_name="coder (ollama)",
            source="ollama",
            execution_class=ExecutionClass.LOCAL,
            supports_tools=True,
            metadata={"parameter_size": "70B", "capability_band": "high"},
            create_provider=mock.Mock(return_value=ScriptedProvider([])),
        )
        weaker_descriptor = SimpleNamespace(
            id="ollama:tiny",
            provider="ollama",
            model="tiny",
            display_name="tiny (ollama)",
            source="ollama",
            execution_class=ExecutionClass.LOCAL,
            supports_tools=True,
            metadata={"parameter_size": "7B", "capability_band": "low"},
            create_provider=mock.Mock(return_value=ScriptedProvider([])),
        )
        catalog = SimpleNamespace(
            discover=mock.Mock(return_value=(weaker_descriptor, local_descriptor)),
            diagnostics=(),
        )

        with mock.patch("agent.cli.PersistentWorkspaceApp", FakeApp), mock.patch(
            "agent.cli.TelemetrySampler"
        ) as telemetry, mock.patch(
            "agent.cli.question_session", return_value=session
        ), mock.patch(
            "agent.cli.answer_question", side_effect=failure
        ), mock.patch(
            "agent.cli.ModelCatalog", return_value=catalog
        ), mock.patch(
            "agent.cli._show_runtime_state"
        ), mock.patch(
            "agent.cli._current_ultra_run", return_value=None
        ):
            telemetry.return_value.start.return_value = None
            telemetry.return_value.stop.return_value = None
            _persistent_interactive_loop(runtime, console, SessionPreferences())

        runtime.continue_with_local_model.assert_called_once_with(
            local_descriptor.create_provider.return_value,
            local_descriptor,
        )
        runtime.resume.assert_called_once_with()
        fallback_updates = [
            call.kwargs
            for call in runtime.store.update_goal_metadata.call_args_list
            if "provider_recovery" in call.kwargs
        ]
        self.assertTrue(
            any(item["provider_recovery"]["automatic_fallback"] for item in fallback_updates)
        )
        self.assertFalse(runtime.store.request_attention.called)

    def test_full_auto_switches_cloud_planning_failure_to_local(self):
        import time

        from agent.cli import _persistent_interactive_loop
        from agent.model_catalog import ExecutionClass

        class FakeApp:
            def __init__(self, store, *, on_exit, **_kwargs):
                self.store = store
                self.on_exit = on_exit

            def run(self):
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    if runtime.resume.called:
                        self.on_exit()
                        return
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

        goal = SimpleNamespace(
            id="goal-planning-cloud",
            status=GoalStatus.PAUSED,
            active_plan_revision=None,
            metadata={
                "provider_recovery": {
                    "state": "paused",
                    "error": "provider unavailable after retries: quota exhausted",
                },
                "waiting_question": "The provider retries were exhausted at a saved planning checkpoint.",
            },
        )
        runtime = mock.Mock()
        runtime.workspace = Path("workspace")
        runtime.model_name = "gpt-oss:120b-cloud"
        runtime.execution_class = "cloud"
        runtime.provider_name = "openai"
        runtime.reasoning_effort = "medium"
        runtime.access_level = "normal"
        runtime.session_id = "planning-fallback-session"
        runtime.interaction_mode = SimpleNamespace(value="working")
        runtime.sleep_mode_policy.return_value = "full"
        runtime.active_goal.return_value = goal
        runtime.dashboard.return_value = SimpleNamespace(
            status="paused", tasks=(), objective="Build the project", goal_id=goal.id, plan_revision=0
        )
        runtime.session_snapshot.return_value = {}
        runtime.store.list_timeline_entries.return_value = []
        runtime.store.count_queued_prompts.return_value = 0
        runtime.store.list_queued_prompts.return_value = []
        runtime.store.claim_next_prompt.return_value = None
        runtime.store.get_workflow_session.return_value = {"state": {}}
        runtime.store.get_accepted_plan.return_value = None
        runtime.local_web_server.take_execution_request.return_value = False

        local_descriptor = SimpleNamespace(
            id="ollama:coder",
            provider="ollama",
            model="coder",
            display_name="coder (ollama)",
            source="ollama",
            execution_class=ExecutionClass.LOCAL,
            supports_tools=True,
            metadata={"parameter_size": "70B", "capability_band": "high"},
            create_provider=mock.Mock(return_value=ScriptedProvider([])),
        )
        catalog = SimpleNamespace(discover=mock.Mock(return_value=(local_descriptor,)), diagnostics=())

        def replace_provider(_provider, descriptor, **_kwargs):
            runtime.execution_class = descriptor.execution_class.value

        runtime.replace_provider.side_effect = replace_provider

        console = mock.Mock()
        console.stream = io.StringIO()
        console.color = False

        with mock.patch("agent.cli.PersistentWorkspaceApp", FakeApp), mock.patch(
            "agent.cli.TelemetrySampler"
        ) as telemetry, mock.patch(
            "agent.cli.ModelCatalog", return_value=catalog
        ), mock.patch(
            "agent.cli.question_session", return_value=None
        ), mock.patch(
            "agent.cli._show_runtime_state"
        ), mock.patch(
            "agent.cli._current_ultra_run", return_value=None
        ):
            telemetry.return_value.start.return_value = None
            telemetry.return_value.stop.return_value = None
            _persistent_interactive_loop(runtime, console, SessionPreferences())

        runtime.replace_provider.assert_called_once_with(
            local_descriptor.create_provider.return_value,
            local_descriptor,
        )
        runtime.resume.assert_called_once_with()
        recovery_updates = [
            call.kwargs.get("provider_recovery", {})
            for call in runtime.store.update_goal_metadata.call_args_list
        ]
        self.assertTrue(any(item.get("automatic_fallback_attempted") for item in recovery_updates))

    def test_persistent_controller_recovers_approved_work_without_replaying_approval(self):
        import time

        from agent.cli import _persistent_interactive_loop

        started = Event()

        class FakeApp:
            def __init__(self, store, *, on_exit, **_kwargs):
                self.store = store
                self.on_exit = on_exit
                self.overlay_kind = ""

            def run(self):
                deadline = time.monotonic() + 3
                while not started.is_set() and time.monotonic() < deadline:
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
        runtime.model_name = "test-model"
        runtime.session_id = "web-approved-session"
        runtime.interaction_mode = SimpleNamespace(value="working")
        runtime.dashboard.return_value = SimpleNamespace(
            status="running",
            tasks=(),
            objective="Build the approved project",
            goal_id="goal-web",
            plan_revision=1,
        )
        goal = SimpleNamespace(
            id="goal-web",
            status=GoalStatus.RUNNING,
            metadata={},
            active_plan_revision=1,
        )
        runtime.active_goal.return_value = goal
        runtime.store.list_timeline_entries.return_value = []
        runtime.store.count_queued_prompts.return_value = 0
        runtime.store.list_queued_prompts.return_value = []
        runtime.store.get_workflow_session.return_value = {"state": {}}
        runtime.local_web_server.take_execution_request.return_value = False
        console = mock.Mock()
        console.stream = io.StringIO()
        console.color = False

        with mock.patch("agent.cli.PersistentWorkspaceApp", FakeApp), mock.patch(
            "agent.cli.TelemetrySampler"
        ) as telemetry, mock.patch(
            "agent.cli.question_session", return_value=None
        ), mock.patch(
            "agent.cli._current_ultra_run", return_value=None
        ), mock.patch(
            "agent.cli._run_auto", side_effect=lambda *_args, **_kwargs: started.set()
        ) as run_auto:
            telemetry.return_value.start.return_value = None
            telemetry.return_value.stop.return_value = None
            _persistent_interactive_loop(runtime, console, SessionPreferences())

        self.assertTrue(started.is_set())
        run_auto.assert_called_once_with(runtime, console)

    def test_general_sleep_mode_is_available_outside_ultra(self):
        output = io.StringIO()
        console = ConsoleUI(stream=output, color=False)
        runtime = mock.Mock()
        runtime.sleep_mode_enabled.return_value = True
        runtime.sleep_mode_policy.return_value = "safe"
        preferences = SessionPreferences(mode=InteractionMode.NORMAL)

        self.assertTrue(
            execute_command(
                runtime,
                console,
                internal_action(InternalActionKind.SLEEP, action="on"),
                preferences,
            )
        )
        self.assertTrue(console.sleep_enabled)
        runtime.set_sleep_mode.assert_called_once_with(True, policy="safe")
        runtime.sleep_profile.assert_not_called()
        self.assertTrue(
            execute_command(
                runtime,
                console,
                internal_action(InternalActionKind.SLEEP, action="status"),
                preferences,
            )
        )
        self.assertIn("safe recommended choices only", output.getvalue())

    def test_full_sleep_mode_is_an_explicit_cli_action(self):
        output = io.StringIO()
        console = ConsoleUI(stream=output, color=False)
        runtime = mock.Mock()
        preferences = SessionPreferences(mode=InteractionMode.NORMAL)

        self.assertTrue(
            execute_command(
                runtime,
                console,
                internal_action(InternalActionKind.SLEEP, action="full"),
                preferences,
            )
        )

        runtime.set_sleep_mode.assert_called_once_with(True, policy="full")
        self.assertTrue(console.sleep_enabled)

    def test_ultra_sleep_gate_failure_does_not_disable_safe_ui_sleep(self):
        output = io.StringIO()
        console = ConsoleUI(stream=output, color=False)
        runtime = mock.Mock()
        runtime.sleep_profile.side_effect = RuntimeError("Docker is not ready")
        preferences = SessionPreferences(mode=InteractionMode.ULTRA)

        self.assertTrue(
            execute_command(
                runtime,
                console,
                internal_action(InternalActionKind.SLEEP, action="on"),
                preferences,
            )
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

    def test_mode_picker_exposes_only_ultra_and_ultra_plan(self):
        answers = iter(("2",))
        output = io.StringIO()

        selected = choose_interaction_mode(
            rich=False,
            input_func=lambda _prompt: next(answers),
            output=output,
            ultra_disabled_reason="A usable local GPU was not detected.",
        )

        self.assertEqual(selected, InteractionMode.PLAN)
        rendered = output.getvalue().casefold()
        self.assertIn("ultra", rendered)
        self.assertIn("ultra-plan", rendered)
        self.assertNotIn("working", rendered)

    def test_help_command_is_offline_import_safe_and_creates_durable_state(self):
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
                        "/help",
                        "--no-color",
                    ]
                )
            self.assertEqual(code, 0)
            rendered = output.getvalue()
            self.assertIn("Slash commands", rendered)
            self.assertNotIn("\x1b", rendered)
            self.assertTrue((Path(directory) / ".coding-agent" / "state.db").is_file())

    def test_public_slash_surface_is_exact_and_old_commands_are_rejected(self):
        self.assertEqual(parse_command("/").kind, CommandKind.MENU)
        self.assertEqual(parse_command("/   ").kind, CommandKind.MENU)
        expected = {
            "/plan": CommandKind.PLAN,
            "/live": CommandKind.LIVE,
            "/show-diff": CommandKind.SHOW_DIFF,
            "/advanced-tracing": CommandKind.ADVANCED_TRACING,
            "/settings": CommandKind.SETTINGS,
            "/pause": CommandKind.PAUSE,
            "/resume": CommandKind.RESUME,
            "/stop": CommandKind.STOP,
            "/undo": CommandKind.UNDO,
            "/help": CommandKind.HELP,
            "/quit": CommandKind.QUIT,
        }
        self.assertEqual(
            {text: parse_command(text).kind for text in expected}, expected
        )
        self.assertEqual(parse_command("/undo").args, {"steps": 1})
        self.assertEqual(parse_command("/undo 3").args, {"steps": 3})
        for removed in (
            "/thinking", "/doctor", "/skills", "/processes", "/diff",
            "/model gemma4:e4b", "/mode ultra", "/approve 1", "/continue",
        ):
            with self.subTest(command=removed):
                with self.assertRaisesRegex(UnknownCommandParseError, "/help"):
                    parse_command(removed)
        for malformed_public in ("/settings color off", "/plan edit"):
            with self.assertRaises(CommandParseError):
                parse_command(malformed_public)

    def test_settings_is_the_single_public_configuration_surface(self):
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
                        "/settings",
                        "--no-color",
                    ]
                )

            self.assertEqual(code, 0)
            rendered = output.getvalue()
            self.assertIn("Session settings", rendered)
            self.assertIn("mode       = automatic", rendered)
            self.assertNotIn("API_KEY", rendered)

    def test_startup_mode_is_persisted_before_the_first_intake(self):
        with tempfile.TemporaryDirectory() as directory:
            output = io.StringIO()
            with redirect_stdout(output):
                code = main(
                    [
                        "--workspace",
                        directory,
                        "--provider",
                        "ollama",
                        "--mode",
                        "ultra",
                        "--command",
                        "/help",
                        "--no-color",
                    ]
                )
            with StateStore(directory) as state:
                session = state.get_workflow_session("workspace-session")

        self.assertEqual(code, 0)
        self.assertEqual(session["session_mode"], "normal")
        self.assertEqual(session["state"]["interaction_mode"], "working")
        self.assertEqual(session["state"]["minimum_strategy"], "recursive")

    def test_normal_startup_recovers_locked_workflow_without_reapplying_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            with StateStore(directory) as store:
                store.save_workflow_session(
                    "workspace-session",
                    goal_id=None,
                    session_mode="normal",
                    plan_state="none",
                    run_state="planning",
                    state={
                        "interaction_mode": "working",
                        "pending_semantic_turn": {
                            "status": "in_progress",
                            "stage": "dispatching",
                            "request": "Build the saved application",
                        },
                    },
                )

            output = io.StringIO()
            with redirect_stdout(output):
                code = main(
                    [
                        "--workspace",
                        directory,
                        "--provider",
                        "ollama",
                        "--mode",
                        "plan",
                        "--command",
                        "/help",
                        "--no-color",
                        "--plain",
                    ]
                )

            with StateStore(directory) as store:
                session = store.get_workflow_session("workspace-session")

        self.assertEqual(code, 0, output.getvalue())
        self.assertEqual(session["session_mode"], "normal")
        self.assertEqual(session["state"]["interaction_mode"], "working")
        self.assertIn("Recovered the active workflow with its saved", output.getvalue())
        self.assertNotIn("fatal: Mode is locked", output.getvalue())

    def test_startup_mode_race_recovers_if_workflow_locks_after_initial_read(self):
        from agent.runtime import RuntimeStateError

        with tempfile.TemporaryDirectory() as directory:
            output = io.StringIO()
            with mock.patch(
                "agent.runtime.AgentRuntime.workflow_mode_lock",
                side_effect=[
                    SimpleNamespace(locked=False),
                    SimpleNamespace(locked=True, stage="dispatching"),
                ],
            ), mock.patch(
                "agent.runtime.AgentRuntime.transition_mode",
                side_effect=RuntimeStateError("Mode is locked until this workflow completes or is cancelled."),
            ), redirect_stdout(output):
                code = main(
                    [
                        "--workspace", directory,
                        "--provider", "ollama",
                        "--model", "gemma4:e4b",
                        "--mode", "plan",
                        "--command", "/help",
                        "--plain",
                        "--no-color",
                    ]
                )

        self.assertEqual(code, 0, output.getvalue())
        self.assertIn("Another launcher started this workflow", output.getvalue())
        self.assertNotIn("fatal: Mode is locked", output.getvalue())

    def test_duplicate_launcher_attaches_to_existing_web_owner(self):
        from agent.session_owner import SessionOwnerInfo

        with tempfile.TemporaryDirectory() as directory:
            existing = SessionOwnerInfo(
                workspace=str(Path(directory).resolve()),
                session_id="workspace-session",
                pid=4321,
                host="test-host",
                started_at=1.0,
                heartbeat_at=2.0,
                web_port=54321,
                web_token="owner-token",
                owner_token="owner-4321",
            )
            output = io.StringIO()
            with mock.patch(
                "agent.cli.SessionOwnerLease.acquire", return_value=None
            ), mock.patch(
                "agent.cli.SessionOwnerLease.read_existing", return_value=existing
            ), mock.patch(
                "webbrowser.open", return_value=True
            ) as open_browser, redirect_stdout(output):
                code = main(
                    [
                        "--workspace", directory,
                        "--provider", "ollama",
                        "--model", "gemma4:e4b",
                        "--command", "/help",
                        "--plain", "--no-color",
                    ]
                )

        self.assertEqual(code, 0, output.getvalue())
        open_browser.assert_not_called()
        self.assertIn("already running", output.getvalue())
        self.assertIn("open http://127.0.0.1:54321", output.getvalue())
        self.assertIn("owner-token", output.getvalue())
        self.assertNotIn("fatal: Mode is locked", output.getvalue())

    def test_working_recovery_does_not_open_a_web_workspace(self):
        from agent.model_catalog import ModelDescriptor
        from agent.sandbox import AccessLevel, DockerSandbox

        with tempfile.TemporaryDirectory() as directory:
            with StateStore(directory) as store:
                store.save_workflow_session(
                    "workspace-session",
                    goal_id=None,
                    session_mode="normal",
                    plan_state="none",
                    run_state="planning",
                    state={
                        "interaction_mode": "working",
                        "pending_semantic_turn": {
                            "status": "in_progress",
                            "stage": "dispatching",
                            "request": "Build the saved application",
                        },
                    },
                )
            setup = (
                Path(directory),
                ModelDescriptor(
                    provider="ollama",
                    model="gemma4:e4b",
                    execution_class="local",
                ),
                DockerSandbox(),
                AccessLevel.NORMAL,
                SessionPreferences(mode=InteractionMode.PLAN),
            )
            output = io.StringIO()
            with mock.patch("agent.cli._interactive_setup", return_value=setup), mock.patch(
                "agent.cli._open_local_web_view"
            ) as open_view, mock.patch("agent.cli.interactive_loop"), redirect_stdout(output):
                code = main(["--provider", "ollama", "--plain", "--no-color"])

        self.assertEqual(code, 0, output.getvalue())
        open_view.assert_not_called()

    def test_doctor_is_a_typed_diagnostics_action_not_a_slash_command(self):
        with tempfile.TemporaryDirectory() as directory:
            output = io.StringIO()
            runtime = mock.Mock()
            runtime.workspace = Path(directory)
            with mock.patch("agent.cli._show_doctor") as show_doctor:
                self.assertTrue(
                    execute_command(
                        runtime,
                        ConsoleUI(stream=output, color=False),
                        internal_action(InternalActionKind.DOCTOR, live=False, record=True),
                        SessionPreferences(),
                    )
                )
            show_doctor.assert_called_once()
        with self.assertRaises(UnknownCommandParseError):
            parse_command("/doctor --record")

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
                internal_action(
                    InternalActionKind.SETTINGS_UPDATE,
                    key="work_quantum",
                    value="7",
                ),
                preferences,
            )
        )
        self.assertEqual(runtime.config.work_quantum_steps, 7)
        runtime.replace_config.assert_called_once()

        self.assertTrue(
            execute_command(
                runtime,
                console,
                internal_action(
                    InternalActionKind.SETTINGS_UPDATE,
                    key="work_quantum",
                    value=None,
                ),
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
                internal_action(
                    InternalActionKind.MODEL,
                    model="evil\x1b[31mred",
                    effort=None,
                ),
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
                    internal_action(InternalActionKind.MODE, mode="normal"),
                    preferences,
                )
            )
            run_auto.assert_not_called()

            self.assertTrue(
                execute_command(
                    runtime,
                    console,
                    internal_action(InternalActionKind.APPROVE, revision=1),
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
                    internal_action(InternalActionKind.APPROVE, revision=1),
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

    def test_unprefixed_exit_is_ordinary_composer_text(self):
        self.assertEqual(parse_command("exit").kind, CommandKind.TEXT)
        self.assertEqual(parse_command("quit").kind, CommandKind.TEXT)

    def test_common_setup_errors_return_friendly_exit_code(self):
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(["--workspace", "definitely-missing-workspace", "--command", "/help"])
        self.assertEqual(code, 2)
        self.assertIn("fatal:", output.getvalue())

        with tempfile.TemporaryDirectory() as directory:
            output = io.StringIO()
            with redirect_stdout(output):
                code = main([
                    "--workspace", directory, "--create-workspace", "--command", "/help"
                ])
            self.assertEqual(code, 0, output.getvalue())
            self.assertNotIn("fatal:", output.getvalue())

        with tempfile.TemporaryDirectory() as directory:
            created = Path(directory) / "fresh-project"
            output = io.StringIO()
            with redirect_stdout(output):
                code = main([
                    "--workspace", str(created), "--create-workspace", "--command", "/help"
                ])
            self.assertEqual(code, 0, output.getvalue())
            self.assertTrue(created.is_dir())
            self.assertNotIn("fatal:", output.getvalue())

    def test_invalid_provider_environment_is_reported_even_with_model_override(self):
        with tempfile.TemporaryDirectory() as directory:
            output = io.StringIO()
            with mock.patch.dict("os.environ", {"LLM_PROVIDER": " invalid provider "}, clear=False):
                with redirect_stdout(output):
                    code = main([
                        "--workspace", directory, "--model", "anything", "--command", "/help"
                    ])
            self.assertEqual(code, 2)
            self.assertIn("Unknown LLM_PROVIDER", output.getvalue())

    def test_noninteractive_approval_fails_closed(self):
        output = io.StringIO()
        console = ConsoleUI(stream=output, color=False, input_func=lambda _prompt: (_ for _ in ()).throw(EOFError()))
        with self.assertRaisesRegex(RuntimeError, "approval interface is unavailable"):
            console.confirm_action("write_file", {"path": "x"}, "high")
        self.assertIn("Approval is still required", output.getvalue())

    def test_persistent_simple_policy_allows_project_edits_without_interrupting(self):
        console = ConsoleUI(stream=io.StringIO(), color=False)
        store = WorkspaceUIStore()
        console.bind_workspace_store(store)
        self.assertTrue(console.confirm_action("write_file", {"path": "app.py"}, "risky"))
        self.assertIsNone(store.active_attention())

    def test_sleep_auto_approves_only_safe_unattended_project_actions(self):
        output = io.StringIO()
        console = ConsoleUI(
            stream=output,
            color=False,
            input_func=lambda _prompt: (_ for _ in ()).throw(
                AssertionError("safe Sleep actions must not ask the user")
            ),
        )
        console.set_sleep_mode(True)

        self.assertTrue(
            console.confirm_action(
                "preview_html",
                {"path": "index.html", "open_browser": True},
                "high",
            )
        )
        self.assertTrue(
            console.confirm_action(
                "run_command",
                {"command": "python -m pytest -q"},
                "high",
            )
        )
        self.assertIn("Sleep auto-approved", output.getvalue())

        console.input_func = lambda _prompt: (_ for _ in ()).throw(EOFError())
        with self.assertRaisesRegex(RuntimeError, "approval interface is unavailable"):
            console.confirm_action(
                "run_command",
                {"command": "npm install untrusted-package"},
                "critical",
            )

    def test_persistent_sleep_auto_approves_preview_without_attention_panel(self):
        console = ConsoleUI(stream=io.StringIO(), color=False)
        store = WorkspaceUIStore()
        console.bind_workspace_store(store)
        console.set_sleep_mode(True)

        self.assertTrue(
            console.confirm_action(
                "preview_html",
                {"path": "index.html", "open_browser": True},
                "high",
            )
        )
        self.assertIsNone(store.active_attention())
        self.assertTrue(
            any("sleep.auto_approval" in item for item in store.snapshot().advanced_log)
        )

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
        self.assertIn("allow_session", [item.key for item in request.options])
        session_option = next(item for item in request.options if item.key == "allow_session")
        self.assertEqual(session_option.label, "Always allow this session")
        self.assertIn("until this session ends", session_option.description)
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
                    "/help",
                    "--no-color",
                    "--plain",
                ]
            )

        self.assertEqual(code, 130)
        checkpoint.assert_called_once_with()
        show_state.assert_not_called()


if __name__ == "__main__":
    unittest.main()
