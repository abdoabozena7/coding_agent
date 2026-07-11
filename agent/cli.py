"""Import-safe CLI and interactive ASCII interface for the persistent runtime."""

from __future__ import annotations

import argparse
import os
import sys
import unicodedata
from pathlib import Path
from typing import Callable, Iterable, TextIO

from colorama import just_fix_windows_console
from dotenv import load_dotenv

from . import __version__
from .commands import CommandKind, CommandParseError, UserCommand, parse_command
from .config import (
    InteractionMode,
    RuntimeConfig,
    SessionPreferences,
    normalize_runtime_setting_name,
    runtime_config_values,
    runtime_setting_names,
    update_runtime_config,
)
from .events import EventBus
from .models import DomainError, GoalStatus
from .providers import get_provider
from .runtime import AgentRuntime, RuntimeErrorBase, SliceResult
from .store import StateCorruptionError, StateStore, StateStoreError
from .ui import ConsoleUI, HELP_TEXT, render_plan, render_slash_menu


APP_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROJECTS_ROOT = APP_ROOT / "projects"


def _next_project_name(root: Path) -> str:
    index = 1
    while (root / f"project-{index:03d}").exists():
        index += 1
    return f"project-{index:03d}"


def _resolve_workspace(path: str | os.PathLike[str], *, create: bool = False) -> Path:
    candidate = Path(path).expanduser()
    try:
        if create:
            candidate.mkdir(parents=True, exist_ok=False)
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"workspace does not exist or is invalid: {candidate}") from exc
    if not resolved.is_dir():
        raise ValueError(f"workspace is not a directory: {resolved}")
    return resolved


def choose_workspace(
    projects_root: str | os.PathLike[str] = DEFAULT_PROJECTS_ROOT,
    *,
    input_func: Callable[[str], str] = input,
    output: TextIO = sys.stdout,
) -> Path:
    """Choose any existing directory, or create a contained numbered workspace."""
    root = Path(projects_root).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    root = root.resolve(strict=True)

    contained: list[Path] = []
    for candidate in root.iterdir():
        if not candidate.is_dir():
            continue
        try:
            resolved = candidate.resolve(strict=True)
            if resolved.parent == root:
                contained.append(resolved)
        except (OSError, RuntimeError):
            continue
    contained.sort(key=lambda item: item.name.casefold())

    print("Workspaces", file=output)
    for index, workspace in enumerate(contained, start=1):
        print(f"  {index:>2}. {workspace.name}", file=output)
    print("Enter a number, a project name, or an existing directory path.", file=output)
    print("Press Enter to create the next project-NNN workspace.", file=output)

    while True:
        choice = input_func("workspace> ").strip()
        if not choice:
            created = root / _next_project_name(root)
            return _resolve_workspace(created, create=True)
        if choice.isdigit():
            index = int(choice)
            if 1 <= index <= len(contained):
                return contained[index - 1]
            print("That workspace number is not listed.", file=output)
            continue

        direct_child = root / choice
        if direct_child.is_dir():
            try:
                resolved = direct_child.resolve(strict=True)
                if resolved.parent == root:
                    return resolved
            except (OSError, RuntimeError):
                pass
        try:
            return _resolve_workspace(choice)
        except ValueError as exc:
            print(str(exc), file=output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="coding-agent",
        description="Persistent plan-first coding agent with adaptive workers and an ASCII control surface.",
    )
    parser.add_argument("--workspace", help="Existing project directory. Interactive selection is used when omitted.")
    parser.add_argument("--create-workspace", action="store_true", help="Create the path passed to --workspace.")
    parser.add_argument("--projects-root", default=os.getenv("AGENT_PROJECTS_DIR", str(DEFAULT_PROJECTS_ROOT)))
    parser.add_argument("--provider", choices=("openai", "gemini", "ollama"), help="Override LLM_PROVIDER.")
    parser.add_argument("--model", help="Override the selected provider's model for this run.")
    parser.add_argument(
        "--mode",
        choices=(InteractionMode.PLAN.value, InteractionMode.GOAL.value),
        help="Interaction mode: plan waits after approval; goal continues automatically.",
    )
    parser.add_argument(
        "--command",
        action="append",
        default=[],
        metavar="TEXT",
        help="Run a UI command (repeatable), for example --command '/status'.",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Retry/self-prompt without limit until completion or real user input.",
    )
    parser.add_argument("--interactive", action="store_true", help="Enter the REPL after --command actions.")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI colors.")
    parser.add_argument("--debug", action="store_true", help="Show tracebacks for harness errors.")
    parser.add_argument("--version", action="version", version=f"coding-agent {__version__}")
    return parser


def _validated_model_name(value: str) -> str:
    model = str(value).strip()
    if (
        not model
        or len(model) > 200
        or any(unicodedata.category(character).startswith("C") for character in model)
    ):
        raise ValueError(
            "model must be a non-empty name without control characters (maximum 200 characters)"
        )
    return model


def _configure_provider_environment(provider: str | None, model: str | None) -> str:
    if provider:
        os.environ["LLM_PROVIDER"] = provider
    selected = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    if selected not in {"openai", "gemini", "ollama"}:
        raise ValueError(
            f"Unknown LLM_PROVIDER {selected!r}; choose openai, gemini, or ollama"
        )
    if model is not None:
        model = _validated_model_name(model)
        variable = {"openai": "OPENAI_MODEL", "gemini": "GEMINI_MODEL", "ollama": "OLLAMA_MODEL"}[selected]
        os.environ[variable] = model
    return selected


def _show_history(runtime: AgentRuntime, console: ConsoleUI) -> None:
    goal = runtime.active_goal() or runtime.store.get_latest_goal()
    if goal is None:
        console.write("No goal history yet.")
        return
    events = runtime.store.list_recent_events(goal.id, limit=50)
    console.write(f"History for {goal.id} ({goal.status.value})")
    for event in events[-30:]:
        detail = event.payload.get("reason") or event.payload.get("summary") or event.payload.get("to") or event.entity_id or ""
        console.write(f"  {event.sequence:>4}  {event.event_type:<28} {str(detail)[:120]}")
    delegations = runtime.store.list_delegations(goal.id)
    if delegations:
        console.write("Adaptive workers")
        for worker in delegations[-20:]:
            console.write(
                f"  {worker.id[-10:]:<10} {worker.status.value:<12} {worker.task_id:<8} "
                f"{worker.role.name}: {(worker.result_summary or worker.brief)[:100]}"
            )
    uncertain = runtime.store.list_actions(goal.id, status="uncertain")
    if uncertain:
        console.write("Uncertain actions (inspect, then /resolve ACTION_ID applied|not-run NOTE)")
        for action in uncertain:
            console.write(
                f"  {action['id']}  {action['tool_name']} task={action['task_id'] or '-'} risk={action['risk']}"
            )
    uncertain_workers = [
        worker for worker in delegations if worker.status.value == "uncertain"
    ]
    if uncertain_workers:
        console.write("Uncertain workers (inspect, then /resolve DELEGATION_ID applied|not-run NOTE)")
        for worker in uncertain_workers:
            console.write(
                f"  {worker.id}  task={worker.task_id} role={worker.role.name}"
            )


def _show_runtime_state(runtime: AgentRuntime, console: ConsoleUI) -> None:
    view = runtime.dashboard()
    console.show_dashboard(view)
    if view.status == GoalStatus.AWAITING_PLAN_APPROVAL.value:
        console.write(render_plan(view))


def _set_interaction_mode(
    runtime: AgentRuntime,
    console: ConsoleUI,
    preferences: SessionPreferences,
    mode: str | InteractionMode,
) -> None:
    selected = InteractionMode.parse(mode)
    preferences.mode = selected
    console.set_mode(selected)
    if selected == InteractionMode.PLAN:
        console.write(
            "PLAN mode active: create/review the plan, approve it, then use /run or /auto explicitly."
        )
        return
    goal = runtime.active_goal()
    suffix = (
        " The current goal is already approved; use /auto to continue it now."
        if goal is not None and goal.status == GoalStatus.RUNNING
        else ""
    )
    console.write(
        "GOAL mode active: plan approval remains mandatory; after your next /approve, "
        f"the agent continues automatically.{suffix}"
    )


def _show_settings(
    runtime: AgentRuntime,
    console: ConsoleUI,
    preferences: SessionPreferences,
    key: str | None = None,
) -> None:
    runtime_values = runtime_config_values(runtime.config)
    safe_values: dict[str, object] = {
        "mode": preferences.mode.value,
        "color": console.color_mode,
        "provider": runtime.provider_name,
        "model": runtime.model_name,
        "workspace": str(runtime.workspace),
        **runtime_values,
    }
    if key:
        normalized = normalize_runtime_setting_name(key)
        if normalized not in safe_values:
            available = ", ".join(
                ("mode", "color", "provider", "model", "workspace", *runtime_setting_names())
            )
            raise ValueError(f"unknown setting {key!r}; available settings: {available}")
        console.write(f"{normalized} = {safe_values[normalized]}")
        return

    console.write("Session settings (API keys and secrets are never shown)")
    console.write(f"  mode       = {preferences.mode.value}")
    console.write(f"  color      = {console.color_mode}")
    console.write(f"  provider   = {runtime.provider_name}")
    console.write(f"  model      = {runtime.model_name}")
    console.write(f"  workspace  = {runtime.workspace}")
    console.write("Runtime limits (session only)")
    for name, value in runtime_values.items():
        console.write(f"  {name:<27} = {value}")
    console.write("Change with /settings NAME VALUE; use /model NAME for the model.")


def _execute_settings(
    runtime: AgentRuntime,
    console: ConsoleUI,
    preferences: SessionPreferences,
    command: UserCommand,
) -> None:
    key = command.args.get("key")
    value = command.args.get("value")
    if key is None:
        _show_settings(runtime, console, preferences)
        return
    if key == "mode":
        if value is None:
            _show_settings(runtime, console, preferences, "mode")
        else:
            _set_interaction_mode(runtime, console, preferences, value)
        return
    if key == "color":
        if value is None:
            _show_settings(runtime, console, preferences, "color")
        else:
            console.set_color(value)
            console.write(f"color = {console.color_mode}")
        return
    if key == "reset":
        if value is not None:
            raise ValueError("Usage: /settings reset")
        runtime.replace_config(RuntimeConfig.from_env())
        console.write("Runtime limits reset from environment/defaults; mode and color were kept.")
        return
    if key in {"provider", "workspace", "model"}:
        if value is not None:
            hint = "Use /model NAME." if key == "model" else "Restart with the matching CLI option."
            raise ValueError(f"{key} is read-only here. {hint}")
        _show_settings(runtime, console, preferences, key)
        return
    if value is None:
        _show_settings(runtime, console, preferences, key)
        return
    updated = update_runtime_config(runtime.config, key, value)
    runtime.replace_config(updated)
    canonical = normalize_runtime_setting_name(key)
    console.write(f"{canonical} = {getattr(updated, canonical)} (session only)")


def _execute_model(runtime: AgentRuntime, console: ConsoleUI, value: str | None) -> None:
    if value is None:
        console.write(f"model = {runtime.model_name}")
        return
    model = _validated_model_name(value)
    if not hasattr(runtime.provider, "model"):
        raise ValueError("the active provider does not support session model switching")
    runtime.provider.model = model
    variable = {
        "openai": "OPENAI_MODEL",
        "gemini": "GEMINI_MODEL",
        "ollama": "OLLAMA_MODEL",
    }.get(runtime.provider_name)
    if variable:
        os.environ[variable] = model
    console.write(f"model = {runtime.model_name} (session only)")


def _run_auto(runtime: AgentRuntime, console: ConsoleUI) -> None:
    console.write(
        "Unbounded goal mode is active. Attempts self-reprompt until completion or real user input; "
        "press Ctrl-C to checkpoint."
    )
    while True:
        goal = runtime.active_goal()
        if goal is None:
            return
        retryable = bool(goal.metadata.get("auto_retryable"))
        if goal.status == GoalStatus.PAUSED and retryable:
            runtime.wait_for_scheduled_retry()
            try:
                runtime.resume()
            except RuntimeErrorBase:
                current = runtime.active_goal()
                if current is None or not (
                    current.status == GoalStatus.PAUSED
                    and current.metadata.get("auto_retryable")
                ):
                    raise
                console.write(
                    "Planning attempt failed transiently; the durable unbounded retry loop will try again."
                )
            _show_runtime_state(runtime, console)
            continue
        if goal.status != GoalStatus.RUNNING:
            return
        runtime.wait_for_scheduled_retry()
        result = runtime.run_slice()
        _show_runtime_state(runtime, console)
        if result.completed or result.needs_user or result.status != GoalStatus.RUNNING.value:
            return


def execute_command(
    runtime: AgentRuntime,
    console: ConsoleUI,
    command: UserCommand,
    preferences: SessionPreferences,
) -> bool:
    """Execute one parsed command. Return False when the session should exit."""
    if command.kind == CommandKind.QUIT:
        return False
    if command.kind == CommandKind.MENU:
        console.write(render_slash_menu())
        return True
    if command.kind == CommandKind.HELP:
        console.write(HELP_TEXT.rstrip())
        return True
    if command.kind == CommandKind.MODE:
        selected = command.args.get("mode")
        if selected is None:
            console.write(f"mode = {preferences.mode.value}; choose /mode plan or /mode goal")
        else:
            _set_interaction_mode(runtime, console, preferences, selected)
        return True
    if command.kind == CommandKind.SETTINGS:
        _execute_settings(runtime, console, preferences, command)
        return True
    if command.kind == CommandKind.MODEL:
        _execute_model(runtime, console, command.args.get("model"))
        return True
    if command.kind == CommandKind.PLAN:
        console.write(render_plan(runtime.dashboard()))
        return True
    if command.kind == CommandKind.STATUS:
        console.show_dashboard(runtime.dashboard())
        return True
    if command.kind == CommandKind.HISTORY:
        _show_history(runtime, console)
        return True
    if command.kind == CommandKind.AUTO:
        _run_auto(runtime, console)
        return True

    result = runtime.apply_command(command)
    if isinstance(result, SliceResult):
        console.write(result.message)
    _show_runtime_state(runtime, console)
    goal_mode_triggers = {CommandKind.APPROVE, CommandKind.RESUME, CommandKind.TEXT}
    nonempty_guidance = command.kind != CommandKind.TEXT or bool(command.args.get("text", "").strip())
    if (
        command.kind in goal_mode_triggers
        and nonempty_guidance
        and preferences.mode == InteractionMode.GOAL
    ):
        goal = runtime.active_goal()
        if goal is not None and goal.status == GoalStatus.RUNNING:
            reason = {
                CommandKind.APPROVE: "plan approved",
                CommandKind.RESUME: "goal resumed",
                CommandKind.TEXT: "guidance received",
            }[command.kind]
            console.write(f"GOAL mode: {reason}; continuing automatically.")
            _run_auto(runtime, console)
    return True


def interactive_loop(
    runtime: AgentRuntime,
    console: ConsoleUI,
    preferences: SessionPreferences,
) -> None:
    _show_runtime_state(runtime, console)
    while True:
        try:
            line = console.prompt()
        except EOFError:
            console.write("\nInput closed. Durable goal state is saved.")
            return
        except KeyboardInterrupt:
            console.write("\nNo action was running. Type /quit to exit or / for controls.")
            continue
        try:
            command = parse_command(line)
            if not execute_command(runtime, console, command, preferences):
                return
        except KeyboardInterrupt:
            runtime.checkpoint_interrupt()
            _show_runtime_state(runtime, console)
        except (CommandParseError, RuntimeErrorBase, StateStoreError, DomainError, ValueError) as exc:
            console.write(f"error: {exc}")


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    just_fix_windows_console()
    console = ConsoleUI(color=False if args.no_color else None)
    preferences = SessionPreferences()
    store: StateStore | None = None
    try:
        load_dotenv(APP_ROOT / ".env", override=False)
        selected_provider = _configure_provider_environment(args.provider, args.model)
        preferences = SessionPreferences.from_env(args.mode)
        console.set_mode(preferences.mode)
        if args.interactive or not args.command:
            console.show_brand()
        if args.workspace:
            workspace_path = Path(args.workspace).expanduser()
            workspace = _resolve_workspace(
                workspace_path,
                create=args.create_workspace,
            )
        else:
            if args.create_workspace:
                raise ValueError("--create-workspace requires --workspace")
            if not sys.stdin.isatty():
                raise ValueError(
                    "--workspace is required when no interactive terminal is available"
                )
            workspace = choose_workspace(args.projects_root)

        provider = get_provider(selected_provider)
        _validated_model_name(str(getattr(provider, "model", "")))
        bus = EventBus()
        bus.subscribe(console.on_event)
        store = StateStore(workspace)
        runtime = AgentRuntime(
            provider,
            store,
            workspace,
            events=bus,
            approval=console.confirm_action,
        )
        console.write(
            f"GA3BAD coding agent | provider={runtime.provider_name} model={runtime.model_name} | "
            "state=.coding-agent/state.db"
        )

        for raw in args.command:
            try:
                if not execute_command(runtime, console, parse_command(raw), preferences):
                    return 0
            except KeyboardInterrupt:
                runtime.checkpoint_interrupt()
                _show_runtime_state(runtime, console)
                return 130
        if args.auto:
            try:
                _run_auto(runtime, console)
            except KeyboardInterrupt:
                runtime.checkpoint_interrupt()
                _show_runtime_state(runtime, console)
                return 130
        if args.interactive or not args.command:
            interactive_loop(runtime, console, preferences)
        return 0
    except (
        StateCorruptionError,
        StateStoreError,
        RuntimeErrorBase,
        DomainError,
        OSError,
        ValueError,
    ) as exc:
        if args.debug:
            raise
        console.write(f"fatal: {exc}")
        return 2
    finally:
        if store is not None:
            store.close()


if __name__ == "__main__":
    raise SystemExit(main())
