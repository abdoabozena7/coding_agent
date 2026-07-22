"""Import-safe CLI and interactive ASCII interface for the persistent runtime."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import textwrap
import time
import unicodedata
from collections import Counter
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread
from typing import Any, Callable, Iterable, Mapping, TextIO

from colorama import just_fix_windows_console

from . import tools
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
from .diagnostics import (
    audit_agent_readiness,
    benchmark_agent_readiness,
    probe_ollama_orchestration_delta_live,
    record_agent_readiness_report,
)
from .evaluation import learn_from_benchmark_trend, record_benchmark_trend
from .events import EventBus
from .model_catalog import ExecutionClass, ModelCatalog, ModelDescriptor
from .models import DomainError, GoalStatus
from .providers import get_provider
from .runtime import AgentRuntime, RuntimeErrorBase, SliceResult
from .rock_coding_agent_intro import play_intro
from .sandbox import AccessLevel, DockerSandbox, PermissionAdapter, SandboxError
from .store import StateCorruptionError, StateStore, StateStoreError
from .tui import (
    ChoiceItem,
    PersistentWorkspaceApp,
    UserExitRequested,
    WorkspaceInput,
    prompt_text,
    rich_terminal_available,
    run_loading_task,
    run_swarm_inspector,
    select_choice,
    select_horizontal_action,
)
from .ui_state import (
    ActivityStage,
    AttentionKind,
    AttentionOption,
    AttentionRequest,
    ExperienceMode,
    WorkspaceUIStore,
    answer_question,
    answer_recommended_remaining,
    is_recommended_defaults_utterance,
    question_attention,
    question_session,
)
from .action_policy import plan_review_reasons
from .ui import (
    ApprovalPromptRequested,
    COMMAND_GROUPS,
    SLASH_COMMANDS,
    ConsoleUI,
    HELP_TEXT,
    WorkspaceRefreshRequested,
    render_agent_detail,
    render_agents,
    render_memory,
    render_plan,
    render_slash_menu,
    render_trace,
    render_tree,
    contextual_commands,
)
from .ultra_models import AgentRunStatus, BrainSection
from .version_control import GitProtectionManager, GitProtectionStatus


APP_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROJECTS_ROOT = APP_ROOT / "projects"


class PickerBack(Exception):
    """Internal navigation signal used by the staged interactive setup."""


_PALETTE_COMMANDS_NEEDING_TEXT = {
    "/goal", "/reject", "/replan", "/answer", "/add", "/edit", "/remove",
    "/done", "/todo", "/block", "/skip", "/resolve", "/cancel", "/stop-process",
    "/agent",
}


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
    rich: bool | None = None,
    initial: str | os.PathLike[str] | None = None,
    step_label: str = "Setup 1 of 5",
    no_color: bool = False,
    reduced_motion: bool = False,
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

    use_rich = (
        rich_terminal_available(input_func=input_func, output=output)
        if rich is None
        else bool(rich)
    )
    if use_rich:
        next_name = _next_project_name(root)
        recent: Path | None = None
        if initial is not None:
            try:
                candidate = Path(initial).expanduser().resolve(strict=True)
                if candidate in contained:
                    recent = candidate
            except (OSError, RuntimeError):
                pass
        if recent is None and contained:
            try:
                recent = max(contained, key=lambda item: item.stat().st_mtime)
            except OSError:
                recent = contained[-1]
        choices = [
            ChoiceItem(
                key=str(workspace),
                label=workspace.name,
                description=(
                    f"{workspace}\nExisting workspace. Enter opens it without changing files."
                ),
                meta="Recent" if workspace == recent else "Existing",
                value=workspace,
            )
            for workspace in contained
        ]
        choices.extend(
            (
                ChoiceItem(
                    key=InteractionMode.PLAN.value,
                    label="Plan",
                    description="Create and revise the approval-bound plan without executing tools.",
                    meta="Current" if selected_mode is InteractionMode.PLAN else "Read-only",
                    value=InteractionMode.PLAN,
                ),
                ChoiceItem(
                    key="__create__",
                    label=f"Create {next_name}",
                    description=(
                        f"Create {root / next_name}\nA clean numbered workspace; creation happens only after you choose this row."
                    ),
                    meta="New",
                    value="__create__",
                ),
                ChoiceItem(
                    key="__path__",
                    label="Open another folder...",
                    description="Enter an existing directory path outside the numbered project list.",
                    meta="Custom path",
                    value="__path__",
                ),
            )
        )
        selected = select_choice(
            choices,
            title="Choose a workspace",
            subtitle="Open an existing project or explicitly create a new one.",
            initial_key=str(recent) if recent is not None else "__create__",
            filterable=True,
            step_label=step_label,
            action_label="Open",
            no_color=no_color,
            reduced_motion=reduced_motion,
            input_func=input_func,
            output=output,
        )
        if selected is None:
            raise PickerBack()
        if selected.value == "__create__":
            return _resolve_workspace(root / next_name, create=True)
        if selected.value == "__path__":
            try:
                raw_path = input_func("folder path (leave blank to go back)> ").strip()
            except (EOFError, KeyboardInterrupt) as exc:
                raise PickerBack() from exc
            if not raw_path:
                raise PickerBack()
            return _resolve_workspace(raw_path)
        return Path(selected.value)

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


def choose_model(
    catalog: ModelCatalog,
    *,
    input_func: Callable[[str], str] = input,
    output: TextIO = sys.stdout,
    rich: bool | None = None,
    initial: str | None = None,
    step_label: str = "Setup 3 of 5",
    no_color: bool = False,
    reduced_motion: bool = False,
) -> ModelDescriptor:
    """Require one explicit tool-capable model selection, Ollama first."""

    use_rich = (
        rich_terminal_available(input_func=input_func, output=output)
        if rich is None
        else bool(rich)
    )
    discovered = (
        run_loading_task(
            catalog.discover,
            title="Finding available models",
            detail="Checking Ollama and configured cloud providers",
            state="search",
            input_func=input_func,
            output=output,
            no_color=no_color,
            reduced_motion=reduced_motion,
        )
        if use_rich
        else catalog.discover()
    )
    if discovered is None:
        raise PickerBack()
    models = tuple(discovered)
    if not models:
        for diagnostic in catalog.diagnostics:
            print(f"  {diagnostic.source}: {diagnostic.message}", file=output)
        raise ValueError(
            "no tool-capable model is available; start Ollama or configure OpenAI/Gemini"
        )
    if use_rich:
        recommended = next(
            (item for item in models if item.execution_class is ExecutionClass.CLOUD),
            models[0],
        )
        initial_key = next(
            (
                item.id
                for item in models
                if initial in {item.id, item.model, item.display_name}
            ),
            recommended.id,
        )
        choices = []
        for descriptor in models:
            speed = (
                "PARALLEL"
                if descriptor.execution_class is ExecutionClass.CLOUD
                else "SEQUENTIAL"
            )
            location = descriptor.execution_class.value.upper()
            provider_note = (
                (
                    "Credentials configured; connectivity and model access are verified on first use."
                    if descriptor.source == "environment"
                    else "Network inference; independent agents can run in parallel."
                )
                if descriptor.execution_class is ExecutionClass.CLOUD
                else "Runs locally; one agent works at a time."
            )
            choices.append(
                ChoiceItem(
                    key=descriptor.id,
                    label=descriptor.display_name,
                    description=(
                        f"{provider_note}\nProvider: {descriptor.provider} · Tool calling available."
                    ),
                    meta=(
                        f"{location} · {speed} · Recommended"
                        if descriptor.id == recommended.id
                        else f"{location} · {speed}"
                    ),
                    value=descriptor,
                )
            )
        selected = select_choice(
            choices,
            title="Choose a model",
            subtitle="Ollama models are probed now; configured cloud models are verified on first use.",
            initial_key=initial_key,
            filterable=True,
            step_label=step_label,
            action_label="Use model",
            no_color=no_color,
            reduced_motion=reduced_motion,
            input_func=input_func,
            output=output,
        )
        if selected is None:
            raise PickerBack()
        return selected.value
    print("Models", file=output)
    for index, descriptor in enumerate(models, start=1):
        speed = "parallel" if descriptor.execution_class is ExecutionClass.CLOUD else "sequential"
        print(
            f"  {index:>2}. {descriptor.display_name}  "
            f"[{descriptor.execution_class.value} · {speed}]",
            file=output,
        )
    while True:
        choice = input_func("model> ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(models):
            return models[int(choice) - 1]
        if choice:
            exact = [
                item
                for item in models
                if choice in {item.id, item.model, item.display_name}
            ]
            if len(exact) == 1:
                return exact[0]
        print("Choose one listed model by number or exact name.", file=output)


def choose_access_level(
    *,
    input_func: Callable[[str], str] = input,
    output: TextIO = sys.stdout,
    sandbox: DockerSandbox | None = None,
    rich: bool | None = None,
    initial: str | AccessLevel = AccessLevel.NORMAL,
    step_label: str = "Setup 4 of 5",
    no_color: bool = False,
    reduced_motion: bool = False,
) -> AccessLevel:
    use_rich = (
        rich_terminal_available(input_func=input_func, output=output)
        if rich is None
        else bool(rich)
    )
    sandbox = sandbox or DockerSandbox()
    if use_rich:
        status = run_loading_task(
            sandbox.status,
            title="Checking Full access",
            detail="Validating Docker and the GA3BAD sandbox",
            state="sync",
            input_func=input_func,
            output=output,
            no_color=no_color,
            reduced_motion=reduced_motion,
        )
        if status is None:
            raise PickerBack()
        full_reason = status.reason or "Run /setup once before enabling Full access."
        choices = (
            ChoiceItem(
                key=AccessLevel.NORMAL.value,
                label="Normal",
                description=(
                    "Ask before risky actions. Works without Docker and is recommended for most projects."
                ),
                meta="Recommended",
                value=AccessLevel.NORMAL,
            ),
            ChoiceItem(
                key=AccessLevel.FULL.value,
                label="Full",
                description=(
                    "Fewer workspace confirmations, isolated inside the configured Docker sandbox."
                    if status.ready
                    else f"Unavailable: {full_reason}"
                ),
                meta="Docker ready" if status.ready else "Unavailable",
                value=AccessLevel.FULL,
                disabled=not status.ready,
                disabled_reason=full_reason,
            ),
        )
        initial_level = AccessLevel.parse(initial)
        selected = select_choice(
            choices,
            title="Choose access",
            subtitle="You can change this later with F4 or /permissions.",
            initial_key=(
                initial_level.value
                if initial_level is AccessLevel.NORMAL or status.ready
                else AccessLevel.NORMAL.value
            ),
            filterable=False,
            step_label=step_label,
            action_label="Use access",
            no_color=no_color,
            reduced_motion=reduced_motion,
            input_func=input_func,
            output=output,
        )
        if selected is None:
            raise PickerBack()
        return selected.value
    status = sandbox.status()
    full_reason = status.reason or "Run /setup once before enabling Full access."
    print("Permissions", file=output)
    print("  1. normal  approvals stay enabled", file=output)
    print(
        "  2. full    "
        + (
            "no workspace confirmations; isolated Docker sandbox is ready"
            if status.ready
            else f"unavailable: {full_reason}"
        ),
        file=output,
    )
    while True:
        choice = input_func("permissions [normal]> ").strip().lower()
        if choice in {"2", "full"}:
            if not status.ready:
                print(f"Full access is unavailable: {full_reason}", file=output)
                continue
            return AccessLevel.FULL
        if choice in {"", "1", "normal"}:
            return AccessLevel.NORMAL
        print("Choose normal or full.", file=output)


def choose_project_protection(
    workspace: str | os.PathLike[str],
    *,
    input_func: Callable[[str], str] = input,
    output: TextIO = sys.stdout,
    rich: bool | None = None,
    step_label: str = "Setup 2 of 5",
    no_color: bool = False,
    reduced_motion: bool = False,
) -> GitProtectionStatus:
    """Choose an honest recovery tier before model execution can modify files."""

    manager = GitProtectionManager(workspace)
    use_rich = (
        rich_terminal_available(input_func=input_func, output=output)
        if rich is None
        else bool(rich)
    )
    while True:
        status = (
            run_loading_task(
                manager.inspect,
                title="Checking project protection",
                detail="Inspecting local Git history and GitHub backup",
                state="sync",
                input_func=input_func,
                output=output,
                no_color=no_color,
                reduced_motion=reduced_motion,
            )
            if use_rich
            else manager.inspect()
        )
        if status is None:
            raise PickerBack()

        github_unavailable_reason = ""
        if not status.gh_available:
            github_unavailable_reason = (
                "GitHub CLI is not installed. Install `gh`, run `gh auth login`, then Refresh."
            )
        elif not status.gh_authenticated:
            github_unavailable_reason = (
                "GitHub CLI is not signed in. Run `gh auth login`, then Refresh."
            )

        if status.github_connected:
            choices = (
                ChoiceItem(
                    key="continue_github",
                    label="Continue with GitHub protection",
                    description=(
                        "Accepted results become local checkpoints and are backed up to "
                        f"{status.remote_url}."
                    ),
                    meta="Recommended",
                    value="continue_github",
                ),
                ChoiceItem(
                    key="refresh",
                    label="Refresh connection",
                    description="Check Git and GitHub again without changing the project.",
                    meta="Recheck",
                    value="refresh",
                ),
                ChoiceItem(
                    key="local",
                    label="Keep checkpoints local",
                    description="Keep multi-step undo, but do not push new checkpoints to GitHub.",
                    meta="No remote sync",
                    value="local",
                ),
            )
            initial_key = "continue_github"
        else:
            connect_description = (
                "Create a private GitHub repository, connect origin, and push the protected baseline."
                if not github_unavailable_reason
                else f"Unavailable: {github_unavailable_reason}"
            )
            if status.dedicated_repository:
                local_label = "Continue with local Git history"
                local_description = (
                    f"Keep {status.commit_count} existing checkpoint(s) locally. "
                    "Multi-step undo works; there is no off-device backup."
                )
            else:
                local_label = "Enable local Git history"
                local_description = (
                    "Create a dedicated repository and protected baseline now. "
                    "This enables multi-step undo without publishing anything."
                )
            can_connect = not bool(github_unavailable_reason)
            choices = (
                ChoiceItem(
                    key="github",
                    label="Create private GitHub backup",
                    description=connect_description,
                    meta="Recommended" if can_connect else "Unavailable",
                    value="github",
                    disabled=not can_connect,
                    disabled_reason=github_unavailable_reason,
                ),
                ChoiceItem(
                    key="local",
                    label=local_label,
                    description=local_description,
                    meta="Recommended" if not can_connect else "Local only",
                    value="local",
                    disabled=not status.git_available,
                    disabled_reason="Git is not installed." if not status.git_available else "",
                ),
                ChoiceItem(
                    key="refresh",
                    label="Refresh connection",
                    description=(
                        "Use this after installing/signing in to GitHub CLI or connecting origin yourself."
                    ),
                    meta="Recheck",
                    value="refresh",
                ),
                ChoiceItem(
                    key="snapshot",
                    label="Continue without Git history",
                    description=(
                        "No version-based undo is available. Ultra can still roll back its current rejected "
                        "attempt, but older accepted versions cannot be selected later."
                    ),
                    meta="Limited undo",
                    value="snapshot",
                ),
            )
            initial_key = "github" if can_connect else "local" if status.git_available else "snapshot"

        if use_rich:
            selected = select_choice(
                choices,
                title="Protect this project before starting",
                subtitle=(
                    f"{status.detail} GitHub adds remote backup; local Git provides the version history."
                ),
                initial_key=initial_key,
                filterable=False,
                step_label=step_label,
                action_label="Continue",
                no_color=no_color,
                reduced_motion=reduced_motion,
                input_func=input_func,
                output=output,
            )
            if selected is None:
                raise PickerBack()
            action = str(selected.value)
        else:
            print("Project protection", file=output)
            print(f"  {status.detail}", file=output)
            available = [item for item in choices if not item.disabled]
            for index, item in enumerate(available, start=1):
                recommended = " [Recommended]" if item.key == initial_key else ""
                print(f"  {index}. {item.label}{recommended}", file=output)
                print(f"     {item.description}", file=output)
            while True:
                raw = input_func("protection [1]> ").strip()
                if not raw:
                    action = initial_key
                    break
                if raw.isdigit() and 1 <= int(raw) <= len(available):
                    action = str(available[int(raw) - 1].value)
                    break
                print("Choose one available protection option.", file=output)

        if action == "refresh":
            continue
        if action == "github":
            return manager.connect_github_private()
        if action == "local":
            protected = manager.ensure_local_history()
            # An already-connected repository can intentionally disable auto-push.
            if status.github_connected:
                manager.configure(auto_checkpoint=True, auto_push=False, provider="local_git")
                return manager.inspect()
            return protected
        if action == "continue_github":
            manager.configure(auto_checkpoint=True, auto_push=True, provider="github")
            return manager.inspect()
        if action == "snapshot":
            return manager.use_snapshot_only()


def choose_interaction_mode(
    *,
    input_func: Callable[[str], str] = input,
    output: TextIO = sys.stdout,
    rich: bool | None = None,
    initial: str | InteractionMode = InteractionMode.NORMAL,
    step_label: str = "Setup 5 of 5",
    no_color: bool = False,
    reduced_motion: bool = False,
    ultra_disabled_reason: str = "",
) -> InteractionMode:
    use_rich = (
        rich_terminal_available(input_func=input_func, output=output)
        if rich is None
        else bool(rich)
    )
    if use_rich:
        selected_mode = InteractionMode.parse(initial)
        selected = select_choice(
            (
                ChoiceItem(
                    key=InteractionMode.NORMAL.value,
                    label="Normal",
                    description=(
                        "Intent intake, one durable goal, planning, review, and automatic execution. "
                        "Complex requests are promoted to Ultra automatically."
                    ),
                    meta="Current" if selected_mode is InteractionMode.NORMAL else "Basic",
                    value=InteractionMode.NORMAL,
                ),
                ChoiceItem(
                    key=InteractionMode.ULTRA.value,
                    label="Ultra",
                    description=(
                        "Use Project Brain, nested agents, and review/test/fix/integration loops. "
                        "Best for large projects; uses more time and tokens."
                    ),
                    meta="Current" if selected_mode is InteractionMode.ULTRA else "Deep workflow",
                    value=InteractionMode.ULTRA,
                    disabled=bool(ultra_disabled_reason),
                    disabled_reason=ultra_disabled_reason,
                ),
            ),
            title="Choose how GA3BAD should work",
            subtitle="You can switch workflow mode at a safe checkpoint with /mode.",
            initial_key=(
                InteractionMode.NORMAL.value
                if selected_mode is InteractionMode.ULTRA and ultra_disabled_reason
                else selected_mode.value
            ),
            filterable=False,
            step_label=step_label,
            action_label="Use mode",
            no_color=no_color,
            reduced_motion=reduced_motion,
            input_func=input_func,
            output=output,
        )
        if selected is None:
            raise PickerBack()
        return selected.value
    print("Mode", file=output)
    print("  1. plan    planning and revision only; approval switches to Normal", file=output)
    print("  2. normal  intent intake, durable goal, plan, review, and automatic execution", file=output)
    print(
        "  3. ultra   "
        + (
            f"unavailable: {ultra_disabled_reason}"
            if ultra_disabled_reason
            else "recursive specialists, component packages, and deeper quality gates"
        ),
        file=output,
    )
    while True:
        choice = input_func("mode> ").strip().lower()
        aliases = {"1": "plan", "2": "normal", "3": "ultra"}
        choice = aliases.get(choice, choice)
        if choice == "ultra" and ultra_disabled_reason:
            print(f"Ultra is unavailable: {ultra_disabled_reason}", file=output)
            continue
        if choice in {"normal", "ultra", "chat", "plan", "goal"}:
            return InteractionMode.parse(choice)
        print("Choose plan, normal, or ultra.", file=output)


def _descriptor_for_explicit_model(
    provider: str,
    model: str,
    *,
    catalog: ModelCatalog | None = None,
) -> ModelDescriptor:
    model = _validated_model_name(model)
    if catalog is not None and provider == "ollama":
        for item in catalog.discover():
            if item.provider == provider and item.model == model:
                return item
        omitted = [
            item.message
            for item in catalog.diagnostics
            if item.source == f"ollama:{model}" and "omit" in item.message.casefold()
        ]
        if omitted:
            raise ValueError(
                f"Ollama model {model!r} is not selectable because it does not advertise tool calling"
            )
    cloud = provider in {"openai", "gemini"} or model.casefold().endswith(
        (":cloud", "-cloud")
    )
    host = (catalog.ollama_host if catalog is not None else os.getenv("OLLAMA_HOST")) if provider == "ollama" else None
    if provider == "ollama" and host:
        lowered = host.casefold()
        if not any(marker in lowered for marker in ("localhost", "127.0.0.1", "[::1]")):
            cloud = True
    return ModelDescriptor(
        provider=provider,
        model=model,
        host=host,
        execution_class=ExecutionClass.CLOUD if cloud else ExecutionClass.LOCAL,
        capabilities=("tools",),
        source="explicit",
    )


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
        type=lambda value: InteractionMode.parse(value).value,
        choices=(InteractionMode.PLAN.value, InteractionMode.NORMAL.value, InteractionMode.ULTRA.value),
        help="Run mode: planning-only plan, durable normal, or recursive-specialist ultra.",
    )
    parser.add_argument(
        "--permissions",
        choices=(AccessLevel.NORMAL.value, AccessLevel.BOUNDED.value, AccessLevel.FULL.value, AccessLevel.HOST.value),
        help="Workspace permission profile. Full uses Docker; host is direct and task-scoped in the web app.",
    )
    parser.add_argument(
        "--setup-sandbox",
        action="store_true",
        help="Build/validate the one-time versioned Docker sandbox, then continue.",
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
        help="Persist through no-progress attempts; pause on repeated provider failures or real user input.",
    )
    parser.add_argument("--interactive", action="store_true", help="Enter the REPL after --command actions.")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI colors.")
    parser.add_argument(
        "--plain",
        action="store_true",
        help="Use the line-oriented UI instead of full-screen pickers and motion.",
    )
    parser.add_argument(
        "--reduced-motion",
        action="store_true",
        help="Use slower, simpler terminal animation without shimmer.",
    )
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


def _show_versions(runtime: AgentRuntime, console: ConsoleUI) -> None:
    status = runtime.version_control.inspect()
    checkpoints = runtime.version_history(30)
    console.write(f"Project protection: {status.detail}")
    if status.remote_url:
        console.write(f"GitHub: {status.remote_url}")
    if not checkpoints:
        console.write("No protected checkpoints yet.")
        return
    console.write("Protected checkpoints (newest first)")
    for index, item in enumerate(checkpoints, start=1):
        summary = runtime.version_control.change_summary(item.commit)
        console.write(
            f"  {index:>2}. {item.commit[:8]}  {item.kind:<9}  "
            f"{item.created_at[:19]}  {item.subject}\n      {summary}"
        )


def _show_runtime_state(
    runtime: AgentRuntime,
    console: ConsoleUI,
    *,
    force: bool = False,
) -> None:
    view = runtime.dashboard()
    active_agents = 0
    try:
        ultra_run = runtime.active_ultra_run()
        if ultra_run is not None and isinstance(getattr(ultra_run, "id", None), str):
            active_agents = len(
                runtime.store.list_agent_runs(
                    ultra_run.id,
                    status=AgentRunStatus.RUNNING,
                )
            )
    except (AttributeError, StateStoreError, TypeError):
        active_agents = 0
    access = getattr(runtime, "access_level", "normal")
    execution = getattr(runtime, "execution_class", "local")
    console.set_runtime_identity(
        access_level=access if isinstance(access, str) else "normal",
        execution_class=execution if isinstance(execution, str) else "local",
        active_agents=active_agents,
        model=runtime.model_name,
        reasoning_effort=runtime.reasoning_effort,
        workspace=str(runtime.workspace),
    )
    console.show_status(view, force=force)
    if view.status == GoalStatus.AWAITING_PLAN_APPROVAL.value and (
        force or not console.live_activity_enabled
    ):
        console.write(
            f"Plan r{view.plan_revision} is ready · {len(view.tasks)} task(s) · "
            "use /plan to expand, /approve to continue, or /reject with feedback."
        )


def _set_interaction_mode(
    runtime: AgentRuntime,
    console: ConsoleUI,
    preferences: SessionPreferences,
    mode: str | InteractionMode,
    *,
    detailed: bool = True,
) -> None:
    selected = InteractionMode.parse(mode)
    if selected == InteractionMode.ULTRA:
        issue = runtime.ultra_readiness_issue()
        if issue:
            raise ValueError(f"Ultra is unavailable: {issue}")
    runtime.set_workflow_mode(selected.value)
    runtime.transition_mode("normal" if selected is InteractionMode.PLAN else selected.value)
    preferences.mode = selected
    console.set_mode(selected)
    if not detailed:
        console.write(f"Mode switched to {selected.value.upper()}.")
        return
    if selected == InteractionMode.ULTRA:
        console.write(
            "ULTRA mode active: GoalSpec → architecture → one master approval → "
            "nested module waves → independent review/test/fix/integration → final evidence."
        )
        return
    if selected == InteractionMode.PLAN:
        console.write(
            "PLAN mode active: build and revise an approval-bound plan without executing. "
            "Approving the latest revision switches to NORMAL and begins implementation."
        )
        return
    goal = runtime.active_goal()
    suffix = (
        " The current goal is already approved; use /auto to continue it now."
        if goal is not None and goal.status == GoalStatus.RUNNING
        else ""
    )
    console.write(
        "NORMAL mode active: every request passes through Intent Architect, then uses a durable "
        f"goal, plan, review, and automatic execution. Complex work auto-promotes to ULTRA.{suffix}"
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
        "reasoning_effort": runtime.reasoning_effort,
        "permissions": runtime.access_level,
        "execution_class": runtime.execution_class,
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
    console.write(f"  reasoning  = {runtime.reasoning_effort}")
    console.write(f"  access     = {runtime.access_level}")
    console.write(f"  execution  = {runtime.execution_class}")
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


def _choose_reasoning_effort(console: ConsoleUI, initial: str = "medium") -> str:
    options = (
        ("low", "Fast", "Quick edits and straightforward questions"),
        ("medium", "Balanced", "Default balance of speed and reasoning"),
        ("high", "Deep", "Complex implementation and debugging"),
        ("xhigh", "Maximum", "Long-horizon, difficult agentic work"),
    )
    rich = not console.plain and rich_terminal_available(console.input_func, console.stream)
    if rich:
        selected = select_choice(
            [ChoiceItem(key, label, description, meta="Current" if key == initial else "") for key, label, description in options],
            title="Choose reasoning effort",
            subtitle="More reasoning can improve difficult work but may take longer.",
            initial_key=initial,
            input_func=console.input_func,
            output=console.stream,
            no_color=not console.color,
            reduced_motion=console.reduced_motion,
        )
        if selected is None:
            raise PickerBack()
        return str(selected.resolved_value)
    answer = console.input_func(f"reasoning [low|medium|high|xhigh] [{initial}]> ").strip().lower()
    return answer or initial


def _execute_model(runtime: AgentRuntime, console: ConsoleUI, value: str | None, effort: str | None = None) -> None:
    catalog = ModelCatalog()
    descriptor = None
    if value is None:
        if effort is not None:
            selected = runtime.set_reasoning_effort(effort)
            console.write(f"reasoning effort = {selected} (session only)")
            return
        try:
            with console.full_screen_modal():
                descriptor = choose_model(
                    catalog,
                    input_func=console.input_func,
                    output=console.stream,
                    rich=(
                        not bool(getattr(console, "plain", False))
                        and rich_terminal_available(
                            input_func=console.input_func,
                            output=console.stream,
                        )
                    ),
                    initial=runtime.model_name,
                    step_label="Session · Model",
                    no_color=not console.color,
                    reduced_motion=console.reduced_motion,
                )
                effort = _choose_reasoning_effort(console, runtime.reasoning_effort)
        except PickerBack:
            return
    else:
        model = _validated_model_name(value)
        discovered = catalog.discover()
        matches = [
            item
            for item in discovered
            if model in {item.id, item.model, item.display_name}
        ]
        if len(matches) == 1:
            descriptor = matches[0]
        else:
            if runtime.provider_name not in {"openai", "gemini", "ollama"}:
                raise ValueError("the active provider does not support session model switching")
            descriptor = _descriptor_for_explicit_model(
                runtime.provider_name,
                model,
                catalog=catalog,
            )
    provider = descriptor.create_provider()
    setattr(provider, "reasoning_effort", effort or runtime.reasoning_effort)
    runtime.replace_provider(provider, descriptor)
    variable = {
        "openai": "OPENAI_MODEL",
        "gemini": "GEMINI_MODEL",
        "ollama": "OLLAMA_MODEL",
    }.get(descriptor.provider)
    if variable:
        os.environ[variable] = descriptor.model
    os.environ["LLM_PROVIDER"] = descriptor.provider
    console.set_runtime_identity(
        access_level=runtime.access_level,
        execution_class=runtime.execution_class,
        model=runtime.model_name,
        reasoning_effort=runtime.reasoning_effort,
        workspace=str(runtime.workspace),
    )
    console.write(
        f"model = {descriptor.provider}/{descriptor.model} · "
        f"{descriptor.execution_class.value} آ· reasoning {runtime.reasoning_effort} (session only)"
    )


def _run_auto(runtime: AgentRuntime, console: ConsoleUI) -> None:
    console.write(
        "Durable goal mode is active. No-progress attempts self-reprompt until completion or real user input; "
        "repeated provider failures pause for repair; "
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
                    "Planning failed transiently; the durable retry policy will try again within its provider-failure limit."
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


def _current_ultra_run(runtime: AgentRuntime) -> object | None:
    try:
        return runtime.active_ultra_run()
    except (AttributeError, StateStoreError):
        return None


def _show_questions(runtime: AgentRuntime, console: ConsoleUI) -> None:
    session = question_session(runtime)
    if session is None or session.current is None:
        console.write("Decisions\n\nNo decisions are waiting for you.")
        return
    item = session.current
    question_id = str(item.get("id", "?"))
    position = session.completed + 1
    lines = [
        f"Decision {position}/{session.total}  {item.get('header', '')}",
        "",
        str(item.get("question", "")),
        "",
    ]
    for index, option in enumerate(item.get("options", ()), 1):
        if not isinstance(option, Mapping):
            continue
        recommended = "  Recommended" if option.get("recommended") or index == 1 else ""
        lines.append(f"  {index}  {option.get('label', '')}{recommended}")
        description = str(option.get("description", "")).strip()
        if description:
            lines.append(f"     {description}")
        lines.append("")
    lines.extend(
        (
            "  4  Write your own answer",
            "",
            "Reply normally, press D in the decision screen to use defaults,",
            f"or use /answer {question_id} 1.",
        )
    )
    console.write("\n".join(lines))


def _answer_pending_question_with_picker(
    runtime: AgentRuntime,
    console: ConsoleUI,
) -> bool:
    """Answer one normalized question with arrows/Enter."""

    session = question_session(runtime)
    if session is None or session.current is None:
        return False
    question = session.current
    options = [
        ChoiceItem(
            key=str(index),
            label=str(option.get("label") or f"Option {index}"),
            description=str(option.get("description") or ""),
            meta="Recommended" if option.get("recommended") or index == 1 else "",
            value=str(option.get("label") or ""),
        )
        for index, option in enumerate(question.get("options", ()), 1)
        if isinstance(option, Mapping)
    ]
    custom_key = str(len(options) + 1)
    options.append(
        ChoiceItem(
            key=custom_key,
            label="Write your answer",
            description="Enter a custom decision in your own words.",
            value=None,
        )
    )
    options.append(
        ChoiceItem(
            key="__defaults__",
            label="Use recommended defaults",
            description=(
                "Accept the recommended option for this decision and every "
                "remaining decision in this interview."
            ),
            meta=f"All {len(session.pending)} remaining",
            value="__defaults__",
        )
    )
    selected = select_choice(
        options,
        title=str(question.get("question") or "Choose an answer"),
        subtitle=str(question.get("reason") or ""),
        step_label=(
            f"Decision {session.completed + 1}/{session.total}  "
            f"{question.get('header', 'Planning')}"
        ),
        action_label="Answer",
        initial_key="1",
        filterable=False,
        page_size=5,
        input_func=console.input_func,
        output=console.stream,
        no_color=not console.color,
        shortcuts={"d": "__defaults__"},
    )
    if selected is None:
        return False
    if selected.key == "__defaults__":
        answer_recommended_remaining(runtime)
        console.write("Recommended answers saved.")
        return True
    if selected.value is None:
        answer = prompt_text(
            title="Write your answer",
            subtitle=str(question.get("question") or ""),
            step_label=f"Decision {session.completed + 1}/{session.total}",
            input_func=console.input_func,
            output=console.stream,
            no_color=not console.color,
        )
        if not answer:
            return False
    else:
        answer = str(selected.resolved_value)
    result = answer_question(runtime, session, str(question.get("id")), answer)
    if isinstance(result, SliceResult):
        console.write(result.message)
    return True


def _review_pending_plan_with_picker(
    runtime: AgentRuntime,
    console: ConsoleUI,
) -> UserCommand | None:
    """Show the plan and its fixed approval actions on one focused screen."""

    goal = runtime.active_goal()
    if goal is None or goal.status != GoalStatus.AWAITING_PLAN_APPROVAL:
        return None
    view = runtime.dashboard()
    width = max(44, min(104, shutil.get_terminal_size((112, 30)).columns - 4))

    def wrapped(value: Any, *, lines: int) -> list[str]:
        normalized = " ".join(str(value or "").split())
        values = textwrap.wrap(normalized, width=width) or ["-"]
        if len(values) > lines:
            values = values[:lines]
            values[-1] = textwrap.shorten(values[-1], width=max(8, width - 3), placeholder="...")
        return values

    body_lines = ["GOAL", *wrapped(view.objective, lines=2), "", "PLAN SUMMARY"]
    body_lines.extend(wrapped(view.plan_summary or view.objective, lines=3))
    body_lines.extend(("", f"TASKS  {len(view.tasks)}"))
    for task in view.tasks[:6]:
        title = textwrap.shorten(
            " ".join(str(task.title).split()),
            width=max(24, width - len(str(task.id)) - 7),
            placeholder="...",
        )
        body_lines.append(f"  {task.id}  {title}")
    if len(view.tasks) > 6:
        body_lines.append(f"  + {len(view.tasks) - 6} more task(s) in Full plan")
    body_lines.extend(("", "EXPECTED CHANGES"))
    if view.expected_changes:
        for item in view.expected_changes[:4]:
            path = str(item.get("path") or "Resolved during execution")
            body_lines.append(f"  {path}")
        if len(view.expected_changes) > 4:
            body_lines.append(f"  + {len(view.expected_changes) - 4} more change(s) in Full plan")
    else:
        body_lines.append("  No file list supplied")

    options = (
        ChoiceItem(
            key="approve",
            label="Approve",
            description="Approve this exact revision and start work.",
            meta="Recommended",
            value="approve",
        ),
        ChoiceItem(
            key="revise",
            label="Request changes",
            description="Describe what the plan got wrong and rebuild this approval-bound revision.",
            value="revise",
        ),
        ChoiceItem(
            key="view",
            label="Full plan",
            description="Print every acceptance criterion, verification step, and file change.",
            value="view",
        ),
        ChoiceItem(
            key="back",
            label="Back",
            description="Keep the plan pending and return to the composer.",
            value="back",
        ),
    )
    selected = select_horizontal_action(
        options,
        title=f"Plan r{view.plan_revision} is ready",
        body="\n".join(body_lines),
        subtitle="The plan stays visible; switch actions with Left/Right and press Enter.",
        step_label="Plan review",
        initial_key="approve",
        input_func=console.input_func,
        output=console.stream,
        no_color=not console.color,
        shortcuts={"a": "approve", "r": "revise", "v": "view", "b": "back"},
    )
    if selected is None or selected.key == "back":
        return None
    if selected.key == "approve":
        return parse_command(f"/approve {view.plan_revision}")
    if selected.key == "view":
        console.write(render_plan(view))
        return None
    feedback = prompt_text(
        title="What should change?",
        subtitle="Be specific about scope, files, behavior, or verification.",
        step_label=f"Revise plan r{view.plan_revision}",
        input_func=console.input_func,
        output=console.stream,
        no_color=not console.color,
    )
    return parse_command(f"/replan {feedback}") if feedback else None


def _swarm_inspector_snapshot(runtime: AgentRuntime, run: Any) -> Mapping[str, Any]:
    """Read one consistent-enough observer frame from durable ULTRA state."""

    nodes = list(runtime.store.list_work_nodes(run.id))
    agents = list(runtime.store.list_agent_runs(run.id))
    profiles = {
        str(item.get("work_node_id")): item
        for item in runtime.store.list_specialist_profiles(run.id)
    }
    traces: dict[str, Any] = {}
    for trace in runtime.store.list_prompt_traces(run.id, limit=1_000):
        node_id = str(getattr(trace, "work_node_id", "") or "")
        if node_id and node_id not in traces:
            traces[node_id] = trace
    return {
        "run_id": run.id,
        "status": getattr(getattr(run, "status", ""), "value", getattr(run, "status", "")),
        "nodes": nodes,
        "agents": agents,
        "profiles": profiles,
        "traces": traces,
    }


def _open_swarm_inspector(
    runtime: AgentRuntime,
    console: ConsoleUI,
    run: Any,
    *,
    initial_tab: str,
) -> bool:
    if not (
        not console.workspace_active
        and rich_terminal_available(input_func=console.input_func, output=console.stream)
        and not console.plain
        and console.stream is sys.stdout
    ):
        return False
    with console.full_screen_modal(coalesce_events=True):
        run_swarm_inspector(
            lambda: _swarm_inspector_snapshot(runtime, run),
            initial_tab=initial_tab,
            input_func=console.input_func,
            output=console.stream,
            no_color=not console.color,
            reduced_motion=console.reduced_motion,
        )
    return True


def _show_tree(runtime: AgentRuntime, console: ConsoleUI, target: str | None) -> None:
    run = _current_ultra_run(runtime)
    if run is None:
        console.write("Project tree\n  (no ULTRA run yet)")
        return
    if target is None and _open_swarm_inspector(
        runtime, console, run, initial_tab="tree"
    ):
        return
    nodes = runtime.store.list_work_nodes(run.id, parent_id=target, recursive=True)
    if target:
        try:
            root = runtime.store.get_work_node(target)
        except StateStoreError:
            console.write(f"Project tree\n  (node {target!r} was not found)")
            return
        values: list[object] = [
            {
                "id": root.id,
                "parent_id": None,
                "title": root.title,
                "status": root.status.value,
                "kind": root.kind.value,
                "position": root.position,
            }
        ]
        values.extend(
            {
                "id": item.id,
                "parent_id": item.parent_id,
                "title": item.title,
                "status": item.status.value,
                "kind": item.kind.value,
                "position": item.position,
            }
            for item in nodes
            if item.id != root.id
        )
        console.write(render_tree(values))
        return
    console.write(render_tree(nodes))


def _show_agents(
    runtime: AgentRuntime,
    console: ConsoleUI,
    *,
    include_finished: bool,
) -> None:
    run = _current_ultra_run(runtime)
    if run is None:
        console.write("Agents\n  (no ULTRA run yet)")
        return
    if not include_finished and _open_swarm_inspector(
        runtime, console, run, initial_tab="agents"
    ):
        return
    nodes = runtime.store.list_work_nodes(run.id)
    node_titles = {node.id: node.title for node in nodes}
    console.write(
        render_agents(
            runtime.store.list_agent_runs(run.id),
            include_finished=include_finished,
            node_titles=node_titles,
            nodes=nodes,
            run_id=run.id,
        )
    )


def _show_agent(runtime: AgentRuntime, console: ConsoleUI, target: str | None) -> None:
    run = _current_ultra_run(runtime)
    if run is None:
        console.write("Specialist view | READ ONLY\n  (no ULTRA run yet)")
        return
    nodes = list(runtime.store.list_work_nodes(run.id))
    agents = list(runtime.store.list_agent_runs(run.id))
    if not target:
        _show_agents(runtime, console, include_finished=False)
        return

    normalized = target.strip().casefold()
    selected_node = None
    selected_agent = None
    display_index = None
    if normalized.isdigit():
        index = int(normalized)
        if 1 <= index <= len(nodes):
            display_index = index
            selected_node = nodes[index - 1]
    if selected_node is None:
        selected_agent = next(
            (
                item
                for item in reversed(agents)
                if item.id.casefold() == normalized
                or item.id.casefold().startswith(normalized)
            ),
            None,
        )
        node_target = selected_agent.work_node_id if selected_agent is not None else normalized
        matches = [
            (index, item)
            for index, item in enumerate(nodes, 1)
            if item.id.casefold() == str(node_target).casefold()
            or item.id.casefold().startswith(str(node_target).casefold())
            or item.title.casefold() == normalized
        ]
        if len(matches) == 1:
            display_index, selected_node = matches[0]
    if selected_node is None and selected_agent is None:
        console.write(
            f"Specialist view | READ ONLY\n  No unique agent or specialist matches {target!r}.\n"
            "  Use /agents or /agents --all to copy a number or id."
        )
        return
    if selected_node is not None and selected_agent is None:
        selected_agent = next(
            (
                item
                for item in reversed(agents)
                if item.work_node_id == selected_node.id
            ),
            None,
        )

    node_id = selected_node.id if selected_node is not None else selected_agent.work_node_id
    profiles = {
        str(item.get("work_node_id")): item
        for item in runtime.store.list_specialist_profiles(run.id)
    }
    trace = None
    if selected_agent is not None:
        traces = runtime.store.list_prompt_traces(
            run.id,
            agent_run_id=selected_agent.id,
            limit=1,
        )
        trace = traces[0] if traces else None
    if trace is None and node_id:
        traces = runtime.store.list_prompt_traces(
            run.id,
            work_node_id=node_id,
            limit=1,
        )
        trace = traces[0] if traces else None
    ancestry = (
        runtime.store.work_node_ancestors(selected_node.id)
        if selected_node is not None
        else ()
    )
    console.write(
        render_agent_detail(
            node=selected_node,
            agent_run=selected_agent,
            profile=profiles.get(str(node_id)),
            trace=trace,
            ancestry=ancestry,
            display_index=display_index,
        )
    )


def _show_memory(runtime: AgentRuntime, console: ConsoleUI, target: str | None) -> None:
    run = _current_ultra_run(runtime)
    if run is None:
        console.write("Project Brain\n  (no ULTRA run yet)")
        return
    section = None
    query = ""
    if target:
        normalized = target.strip().lower()
        if normalized in {"artifact", "artifacts", "artifact_index"}:
            entries = [
                {
                    "section": "artifact_index",
                    "title": item.path or item.uri,
                    "content": (
                        f"{item.kind} · sha256 {item.content_hash or 'not recorded'}"
                    ),
                }
                for item in runtime.store.list_artifacts(run.id)
            ]
            console.write(render_memory(entries, title="Project Brain · Artifact Index"))
            return
        try:
            section = BrainSection(normalized)
        except ValueError:
            query = target
    entries = (
        runtime.store.search_brain(run.id, query, section=section)
        if query
        else runtime.store.list_brain_entries(run.id, section=section)
    )
    console.write(render_memory(entries))


def _show_trace(runtime: AgentRuntime, console: ConsoleUI, target: str | None) -> None:
    run = _current_ultra_run(runtime)
    trace = None
    if target and target.lower() not in {"latest"}:
        try:
            trace = runtime.store.get_prompt_trace(target)
        except StateStoreError:
            traces = runtime.store.list_prompt_traces(target, limit=1)
            trace = traces[0] if traces else None
    elif run is not None:
        trace = runtime.store.latest_prompt_trace(run.id)
    console.write(render_trace(trace))


def _show_thinking(console: ConsoleUI) -> None:
    blocks = console.thought_blocks()
    if not blocks:
        console.write("Thinking\n  (no model thoughts captured in this session)")
        return
    if (
        not console.workspace_active
        and rich_terminal_available(input_func=console.input_func, output=console.stream)
        and not console.plain
        and console.stream is sys.stdout
    ):
        items = tuple(
            ChoiceItem(
                key=str(block["id"]),
                label=f"Thought {block['id']}",
                description=str(block.get("text") or "(empty)"),
                meta=(
                    f"{str(block.get('actor', 'agent')).replace('_', ' ')} · "
                    f"step {block.get('step', '?')} · {block.get('duration_seconds', 0)}s"
                ),
                value=block["id"],
            )
            for block in blocks
        )
        with console.full_screen_modal():
            select_choice(
                items,
                title="Thinking history",
                subtitle="Raw provider thoughts are redacted, session-only, and collapsed by default.",
                initial_key=items[-1].key,
                filterable=True,
                step_label="Inspect · Thinking",
                action_label="Close",
                no_color=not console.color,
                reduced_motion=console.reduced_motion,
                input_func=console.input_func,
                output=console.stream,
            )
        return
    lines = ["Thinking · session only"]
    for block in blocks:
        lines.extend(
            (
                f"  Thought {block['id']} · {block.get('actor', 'agent')} · "
                f"step {block.get('step', '?')} · {block.get('duration_seconds', 0)}s",
                textwrap.indent(str(block.get("text") or "(empty)"), "    "),
            )
        )
    console.write("\n".join(lines))


def _show_insights(runtime: AgentRuntime, console: ConsoleUI, target: str | None) -> None:
    run = _current_ultra_run(runtime)
    if run is None:
        console.write("Insights\n  (no ULTRA run yet)")
        return
    entries = []
    for section in (BrainSection.DECISION, BrainSection.LESSON, BrainSection.KNOWLEDGE):
        entries.extend(
            runtime.store.list_brain_entries(
                run.id,
                section=section,
                work_node_id=target,
                limit=100,
            )
        )
    console.write(render_memory(entries, title="Insights"))


def _show_metrics(runtime: AgentRuntime, console: ConsoleUI) -> None:
    run = _current_ultra_run(runtime)
    if run is None:
        console.write(
            f"Metrics\n  provider {runtime.provider_name}/{runtime.model_name}\n"
            "  (no ULTRA run yet)"
        )
        return
    nodes = runtime.store.list_work_nodes(run.id)
    agents = runtime.store.list_agent_runs(run.id)
    node_counts = Counter(item.status.value for item in nodes)
    agent_counts = Counter(item.status.value for item in agents)
    input_tokens = sum(int(item.usage.get("input_tokens", 0) or 0) for item in agents)
    output_tokens = sum(int(item.usage.get("output_tokens", 0) or 0) for item in agents)
    fixes = sum(item.attempts for item in nodes)
    lines = [
        f"Metrics · run {run.id}",
        f"  execution   {run.execution_class.value} · configured concurrency {run.concurrency}",
        f"  nodes       {len(nodes)} · " + ", ".join(f"{key}={value}" for key, value in sorted(node_counts.items())),
        f"  agents      {len(agents)} · " + ", ".join(f"{key}={value}" for key, value in sorted(agent_counts.items())),
        f"  fix attempts {fixes}",
        f"  tokens      in={input_tokens} out={output_tokens}",
        f"  traces      {len(runtime.store.list_prompt_traces(run.id, limit=10_000))}",
        f"  artifacts   {len(runtime.store.list_artifacts(run.id, limit=10_000))}",
    ]
    console.write("\n".join(lines))


def _show_doctor(
    runtime: AgentRuntime,
    console: ConsoleUI,
    *,
    live: bool = False,
    record: bool = False,
) -> None:
    require_gpu = bool(getattr(getattr(runtime, "config", None), "require_local_gpu", False))
    report = audit_agent_readiness(require_gpu=require_gpu)
    behavioral = benchmark_agent_readiness(require_gpu=require_gpu)
    recorded: list[dict[str, Any]] = []
    trends: list[dict[str, Any]] = []
    learned: list[Mapping[str, Any]] = []
    provider_label = str(getattr(runtime, "provider_name", "") or "provider")
    model_label = str(getattr(runtime, "model_name", "") or "agent")
    lines = [
        f"Agent readiness · {'PASS' if report.passed else 'FAIL'}",
        f"  required local GPU: {'yes' if report.require_gpu else 'no'}",
        f"  GPU probe: {'available' if report.gpu.gpu_available else 'unavailable'} via {report.gpu.source}",
    ]
    if record:
        recorded.append(
            record_agent_readiness_report(
                runtime.store,
                report,
                scenario_name="structural",
                provider=provider_label,
                model=model_label,
            )
        )
        trends.append(
            record_benchmark_trend(
                runtime.store,
                suite_name="agent-readiness",
                scenario_name="structural",
                provider=provider_label,
                model=model_label,
            )
        )
        learned.append(learn_from_benchmark_trend(runtime.store, trends[-1]))
        recorded.append(
            record_agent_readiness_report(
                runtime.store,
                behavioral,
                scenario_name="behavioral",
                provider=provider_label,
                model=model_label,
            )
        )
        trends.append(
            record_benchmark_trend(
                runtime.store,
                suite_name="agent-readiness",
                scenario_name="behavioral",
                provider=provider_label,
                model=model_label,
            )
        )
        learned.append(learn_from_benchmark_trend(runtime.store, trends[-1]))
    if report.gpu.devices:
        for device in report.gpu.devices:
            name = str(device.get("name") or "GPU")
            driver = str(device.get("driver") or "").strip()
            memory = str(device.get("memory") or "").strip()
            detail = " · ".join(item for item in (driver, memory) if item)
            lines.append(f"    - {name}{(' · ' + detail) if detail else ''}")
    elif report.gpu.message:
        lines.append(f"    {report.gpu.message}")
    for check in report.checks:
        mark = "OK" if check.passed else "FAIL"
        detail = ", ".join(check.evidence if check.passed else check.missing)
        lines.append(f"  [{mark}] {check.name}{(' · ' + detail) if detail else ''}")
        if not check.passed and check.message:
            lines.append(f"       {check.message}")
    lines.append(f"Behavioral probes · {'PASS' if behavioral.passed else 'FAIL'}")
    for check in behavioral.checks:
        mark = "OK" if check.passed else "FAIL"
        detail = ", ".join(check.evidence if check.passed else check.missing)
        lines.append(f"  [{mark}] {check.name}{(' · ' + detail) if detail else ''}")
        if not check.passed and check.message:
            lines.append(f"       {check.message}")
    if live:
        provider_name = str(getattr(runtime, "provider_name", "") or "").casefold()
        model_name = str(getattr(runtime, "model_name", "") or "").strip()
        if provider_name != "ollama":
            lines.append("Live model probe · SKIPPED · current provider is not Ollama")
        else:
            host = None
            descriptor = getattr(runtime, "model_descriptor", None)
            if descriptor is not None:
                host = getattr(descriptor, "host", None)
            live_report = probe_ollama_orchestration_delta_live(
                model_name,
                host=host,
                require_gpu=require_gpu,
            )
            if record:
                recorded.append(
                    record_agent_readiness_report(
                        runtime.store,
                        live_report,
                        scenario_name="live-orchestration-delta",
                        provider=provider_name or "ollama",
                        model=model_name or "ollama",
                    )
                )
                trends.append(
                    record_benchmark_trend(
                        runtime.store,
                        suite_name="agent-readiness",
                        scenario_name="live-orchestration-delta",
                        provider=provider_name or "ollama",
                        model=model_name or "ollama",
                    )
                )
                learned.append(learn_from_benchmark_trend(runtime.store, trends[-1]))
            lines.append(f"Live orchestration delta · {'PASS' if live_report.passed else 'FAIL'}")
            for check in live_report.checks:
                mark = "OK" if check.passed else "FAIL"
                detail = ", ".join(check.evidence if check.passed else check.missing)
                lines.append(f"  [{mark}] {check.name}{(' · ' + detail) if detail else ''}")
                if not check.passed and check.message:
                    lines.append(f"       {check.message}")
    if recorded:
        details = ", ".join(f"{item['scenario_name']}={item['id']}" for item in recorded)
        lines.append(f"Recorded benchmark runs · {details}")
    if trends:
        trend_details = []
        for item in trends:
            inputs = item.get("inputs") if isinstance(item.get("inputs"), dict) else {}
            trend = inputs.get("trend") if isinstance(inputs.get("trend"), dict) else {}
            source = str(inputs.get("source_scenario_name") or item.get("scenario_name") or "scenario")
            verdict = str(trend.get("verdict") or ("regressed" if item.get("result") == "failed" else "stable"))
            trend_details.append(f"{source}={verdict}:{item['id']}")
        lines.append("Benchmark trends · " + ", ".join(trend_details))
    learned_items = [item for item in learned if item.get("recorded")]
    if learned_items:
        details = ", ".join(f"{item['verdict']}={item['brain_entry_id']}" for item in learned_items)
        lines.append("Benchmark learning · " + details)
    console.write("\n".join(lines))


def _execute_permissions(
    runtime: AgentRuntime,
    console: ConsoleUI,
    level: str | None,
) -> None:
    if level is None:
        if rich_terminal_available(
            input_func=console.input_func,
            output=console.stream,
        ) and not bool(getattr(console, "plain", False)):
            sandbox = (
                runtime.permission_adapter.sandbox
                if runtime.permission_adapter is not None
                else DockerSandbox()
            )
            try:
                with console.full_screen_modal():
                    level = choose_access_level(
                        input_func=console.input_func,
                        output=console.stream,
                        sandbox=sandbox,
                        rich=True,
                        initial=runtime.access_level,
                        step_label="Session · Access",
                        no_color=not console.color,
                        reduced_motion=console.reduced_motion,
                    ).value
            except PickerBack:
                return
        else:
            console.write(f"permissions = {runtime.access_level}")
            return
    sandbox = (
        runtime.permission_adapter.sandbox
        if runtime.permission_adapter is not None
        else DockerSandbox()
    )
    adapter = PermissionAdapter(level, sandbox)
    runtime.replace_permission_adapter(adapter)
    console.set_runtime_identity(
        access_level=runtime.access_level,
        execution_class=runtime.execution_class,
    )
    if adapter.selection.reason:
        console.write(adapter.selection.reason)
    console.write(f"permissions = {adapter.access_level.value} (this session)")


def _setup_sandbox(runtime: AgentRuntime, console: ConsoleUI) -> None:
    sandbox = (
        runtime.permission_adapter.sandbox
        if runtime.permission_adapter is not None
        else DockerSandbox()
    )
    config = sandbox.setup()
    console.write(f"Full sandbox ready · {config.image} · user {config.container_user}")


def _open_command_palette_inner(console: ConsoleUI, status: str) -> str | None:
    """Open a small contextual root, then one command group at a time."""

    if not rich_terminal_available(
        input_func=console.input_func,
        output=console.stream,
    ) or bool(getattr(console, "plain", False)) or console.stream is not sys.stdout:
        return None

    descriptions = dict(SLASH_COMMANDS)
    group_map = {label: (description, commands) for label, description, commands in COMMAND_GROUPS}
    roots = [
        ChoiceItem(
            key="__suggested__",
            label="Suggested now",
            description="Actions that match the current goal checkpoint.",
            meta=str(status).replace("_", " ").title(),
            value="__suggested__",
        )
    ]
    roots.extend(
        ChoiceItem(
            key=label,
            label=label,
            description=description,
            meta=f"{len(commands)} commands",
            value=label,
        )
        for label, description, commands in COMMAND_GROUPS
    )

    while True:
        selected_group = select_choice(
            roots,
            title="Commands",
            subtitle="Choose a category. Type part of a command at the prompt for direct search.",
            initial_key="__suggested__",
            filterable=False,
            step_label="Command palette",
            action_label="Open",
            no_color=not console.color,
            reduced_motion=console.reduced_motion,
            input_func=console.input_func,
            output=console.stream,
        )
        if selected_group is None:
            return None
        if selected_group.value == "__suggested__":
            command_pairs = contextual_commands(status)
            group_title = "Suggested now"
        else:
            group_title = str(selected_group.value)
            command_pairs = tuple(
                (command, descriptions[command])
                for command in group_map[group_title][1]
            )
        selected_command = select_choice(
            tuple(
                ChoiceItem(
                    key=command,
                    label=command,
                    description=description,
                    meta="Enter to open",
                    value=command,
                )
                for command, description in command_pairs
            ),
            title=group_title,
            subtitle="Esc returns to command categories.",
            filterable=True,
            step_label="Command palette",
            action_label="Choose",
            no_color=not console.color,
            reduced_motion=console.reduced_motion,
            input_func=console.input_func,
            output=console.stream,
        )
        if selected_command is not None:
            return str(selected_command.value)


def _open_command_palette(console: ConsoleUI, status: str) -> str | None:
    with console.full_screen_modal():
        return _open_command_palette_inner(console, status)


def execute_command(
    runtime: AgentRuntime,
    console: ConsoleUI,
    command: UserCommand,
    preferences: SessionPreferences,
    *,
    structured_attention: bool = False,
) -> bool:
    """Execute one parsed command. Return False when the session should exit."""
    if command.kind == CommandKind.QUIT:
        return False
    if command.kind == CommandKind.MENU:
        selected_command = _open_command_palette(
            console,
            getattr(runtime.dashboard(), "status", "idle"),
        )
        if selected_command is None:
            if not rich_terminal_available(
                input_func=console.input_func,
                output=console.stream,
            ) or bool(getattr(console, "plain", False)):
                console.write(render_slash_menu())
            return True
        if selected_command in _PALETTE_COMMANDS_NEEDING_TEXT:
            console.prefill_prompt(selected_command + " ")
            return True
        return execute_command(
            runtime,
            console,
            parse_command(selected_command),
            preferences,
            structured_attention=structured_attention,
        )
    if command.kind == CommandKind.HELP:
        console.write(HELP_TEXT.rstrip())
        return True
    if command.kind == CommandKind.MODE:
        selected = command.args.get("mode")
        if selected is None:
            if rich_terminal_available(
                input_func=console.input_func,
                output=console.stream,
            ) and not bool(getattr(console, "plain", False)):
                try:
                    with console.full_screen_modal():
                        selected = choose_interaction_mode(
                            input_func=console.input_func,
                            output=console.stream,
                            rich=True,
                            initial=preferences.mode,
                            step_label="Session · Mode",
                            no_color=not console.color,
                            reduced_motion=console.reduced_motion,
                            ultra_disabled_reason=runtime.ultra_readiness_issue() or "",
                        )
                except PickerBack:
                    return True
                _set_interaction_mode(
                    runtime,
                    console,
                    preferences,
                    selected,
                    detailed=False,
                )
            else:
                console.write(
                    f"mode = {preferences.mode.value}; choose /mode normal or /mode ultra"
                )
        else:
            _set_interaction_mode(runtime, console, preferences, selected)
        return True
    if command.kind == CommandKind.SETTINGS:
        _execute_settings(runtime, console, preferences, command)
        return True
    if command.kind == CommandKind.MODEL:
        _execute_model(runtime, console, command.args.get("model"), command.args.get("effort"))
        return True
    if command.kind == CommandKind.PERMISSIONS:
        _execute_permissions(runtime, console, command.args.get("level"))
        return True
    if command.kind == CommandKind.IDE:
        console.write(
            "IDE context bridge is not connected in this local agent yet. "
            f"Workspace context is active: {runtime.workspace}"
        )
        return True
    if command.kind == CommandKind.KEYMAP:
        console.write(
            "Keymap: F2 Simple/Advanced display, F3 model, F4 permissions, "
            "Ctrl+K actions, Ctrl+C safe checkpoint while running, Ctrl+Q exit "
            "at a checkpoint. Type /vim to toggle Vim composer keys."
        )
        return True
    if command.kind == CommandKind.VIM:
        requested = command.args.get("state")
        enabled = console.set_vim_mode(
            None if requested is None else requested == "on"
        )
        console.write(f"Vim composer mode {'on' if enabled else 'off'}.")
        return True
    if command.kind == CommandKind.SANDBOX_ADD_READ_DIR:
        path = Path(str(command.args.get("path") or "")).expanduser()
        if not path.is_absolute():
            console.write("error: /sandbox-add-read-dir needs an absolute path.")
            return True
        if not path.is_dir():
            console.write(f"error: read directory does not exist: {path}")
            return True
        console.write(
            "Sandbox read-dir UI command accepted. This agent's current Docker "
            "sandbox still mounts the active workspace only, so use --workspace "
            f"or /settings workspace when you need execution inside {path}."
        )
        return True
    if command.kind == CommandKind.EXPERIMENTAL:
        console.write(
            "Experimental features are controlled through /settings in this agent. "
            f"Requested state: {command.args.get('state')}."
        )
        return True
    if command.kind == CommandKind.SKILLS:
        rows = tools.capability_report()
        lines = ["Available local agent capabilities"]
        for item in rows:
            status = "ready" if item["available"] else "unavailable"
            detail = f" · {item['detail']}" if item.get("detail") else ""
            lines.append(
                f"  {item['name']:<24} {item['category']:<9} {status:<11} "
                f"risk={item['risk']} approval={item['approval']}{detail}"
            )
        console.write("\n".join(lines))
        return True
    if command.kind == CommandKind.DOCTOR:
        _show_doctor(
            runtime,
            console,
            live=bool(command.args.get("live")),
            record=bool(command.args.get("record")),
        )
        return True
    if command.kind == CommandKind.PROCESSES:
        with tools.workspace_context(runtime.workspace):
            processes = tools.process_manager.list_processes()
            previews = tools.web_preview.list_previews()
        if not processes and not previews:
            console.write("No managed processes or previews are active.")
        else:
            lines = ["Managed resources"]
            lines.extend(f"  {item['process_id']} · {item['status']} · {item['command']}" for item in processes)
            lines.extend(f"  {item['preview_id']} · preview · {item['url']}" for item in previews)
            console.write("\n".join(lines))
        return True
    if command.kind == CommandKind.STOP_PROCESS:
        resource_id = str(command.args["resource_id"])
        with tools.workspace_context(runtime.workspace):
            result = (
                tools.run_tool("stop_preview", {"preview_id": resource_id})
                if resource_id.startswith("preview-")
                else tools.run_tool("stop_process", {"process_id": resource_id})
            )
        console.write(result)
        return True
    if command.kind == CommandKind.SETUP:
        _setup_sandbox(runtime, console)
        return True
    if command.kind == CommandKind.SLEEP:
        status = runtime.sleep_profile(command.args["action"], preferences.mode)
        console.write(f"Sleep profile {status['profile']} · state {status['state']}")
        return True
    if command.kind == CommandKind.TREE:
        _show_tree(runtime, console, command.args.get("target"))
        return True
    if command.kind == CommandKind.AGENTS:
        _show_agents(runtime, console, include_finished=bool(command.args.get("all")))
        return True
    if command.kind == CommandKind.AGENT:
        _show_agent(runtime, console, command.args.get("target"))
        return True
    if command.kind == CommandKind.MEMORY:
        _show_memory(runtime, console, command.args.get("target"))
        return True
    if command.kind == CommandKind.TRACE:
        _show_trace(runtime, console, command.args.get("target"))
        return True
    if command.kind == CommandKind.THINKING:
        _show_thinking(console)
        return True
    if command.kind == CommandKind.INSIGHTS:
        _show_insights(runtime, console, command.args.get("target"))
        return True
    if command.kind == CommandKind.QUESTIONS:
        _show_questions(runtime, console)
        return True
    if command.kind == CommandKind.METRICS:
        _show_metrics(runtime, console)
        return True
    if command.kind == CommandKind.PLAN:
        console.write(render_plan(runtime.dashboard()))
        return True
    if command.kind == CommandKind.STATUS:
        _show_runtime_state(runtime, console, force=True)
        return True
    if command.kind == CommandKind.HISTORY:
        _show_history(runtime, console)
        return True
    if command.kind == CommandKind.DIFF:
        console.write(runtime.version_control.diff(command.args.get("target")))
        return True
    if command.kind == CommandKind.VERSIONS:
        _show_versions(runtime, console)
        return True
    if command.kind == CommandKind.UNDO:
        reverted = runtime.undo_versions(int(command.args.get("steps") or 1))
        console.write(
            f"Undo complete: reverted {len(reverted)} accepted checkpoint(s). "
            "The undo itself is preserved in Git history."
        )
        return True
    if command.kind == CommandKind.AUTO:
        if preferences.mode == InteractionMode.ULTRA:
            console.write("ULTRA execution already runs in the background after master approval.")
            _show_runtime_state(runtime, console)
            return True
        _run_auto(runtime, console)
        return True

    if command.kind in {
        CommandKind.GOAL,
        CommandKind.TEXT,
    }:
        text = command.args.get("objective", command.args.get("text", "")).strip()
        if not text:
            return True
        active = runtime.active_goal()
        active_metadata = getattr(active, "metadata", {}) if active is not None else {}
        session = question_session(runtime) if command.kind == CommandKind.TEXT else None
        if session is not None and session.current is not None:
            if is_recommended_defaults_utterance(text):
                answers = answer_recommended_remaining(runtime)
                result = answers[-1] if answers else None
                console.write("Recommended answers saved.")
            else:
                result = answer_question(
                    runtime,
                    session,
                    str(session.current.get("id", "")),
                    text,
                )
        elif active is not None and active_metadata.get("ultra_run_id"):
            pending = [
                item
                for item in runtime.ultra_questions()
                if not active_metadata.get("plan_answers", {}).get(
                    str(item.get("id"))
                )
            ]
            result = (
                runtime.answer_ultra_question(str(pending[0].get("id")), text)
                if len(pending) == 1
                else runtime.add_ultra_guidance(text)
            )
        else:
            result = runtime.apply_command(command)
    elif preferences.mode == InteractionMode.ULTRA and command.kind == CommandKind.RUN:
        console.write("ULTRA module waves run in the background; use /agents, /tree, or /pause.")
        result = None
    else:
        result = runtime.apply_command(command)
    if isinstance(result, SliceResult) and not (
        structured_attention and result.needs_user
    ):
        console.write(result.message)
    if (
        result is not None
        and runtime.active_goal() is not None
        and runtime.active_goal().status == GoalStatus.AWAITING_PLAN_APPROVAL
        and not structured_attention
    ):
        # Rich terminals open the focused plan-review surface from the main
        # loop. Plain/redirected sessions retain the complete textual plan.
        view = runtime.dashboard()
        if rich_terminal_available(console.input_func, console.stream) and not console.plain:
            console.write(
                f"Plan r{view.plan_revision} is ready for review. "
                "Choose Approve, Read, Revise, or Edit in the review screen."
            )
        else:
            console.write(render_plan(view))
    try:
        actual_mode = InteractionMode.parse(
            runtime.store.get_workflow_session(runtime.session_id)["session_mode"]
        )
        keep_plan_policy = (
            preferences.mode is InteractionMode.PLAN
            and runtime.active_goal() is not None
            and runtime.active_goal().status == GoalStatus.AWAITING_PLAN_APPROVAL
        )
        if actual_mode is not preferences.mode and not keep_plan_policy:
            preferences.mode = actual_mode
            console.set_mode(actual_mode)
    except (StateStoreError, ValueError, TypeError, KeyError, AttributeError):
        pass
    pending_intake = runtime.intake_questions()
    if isinstance(pending_intake, (tuple, list)) and pending_intake:
        if not structured_attention:
            _show_questions(runtime, console)
    if not structured_attention:
        _show_runtime_state(runtime, console)
    if command.kind == CommandKind.APPROVE and preferences.mode is InteractionMode.PLAN:
        preferences.mode = InteractionMode.NORMAL
        console.set_mode(InteractionMode.NORMAL)
        runtime.set_workflow_mode("normal")
        runtime.transition_mode("normal")
        console.write("Plan approved; switching to NORMAL implementation.")
    goal_mode_triggers = {CommandKind.APPROVE, CommandKind.RESUME, CommandKind.TEXT}
    nonempty_guidance = command.kind != CommandKind.TEXT or bool(command.args.get("text", "").strip())
    if (
        command.kind in goal_mode_triggers
        and nonempty_guidance
        and preferences.mode == InteractionMode.NORMAL
    ):
        goal = runtime.active_goal()
        if goal is not None and goal.status == GoalStatus.RUNNING:
            reason = {
                CommandKind.APPROVE: "plan approved",
                CommandKind.RESUME: "goal resumed",
                CommandKind.TEXT: "guidance received",
            }[command.kind]
            console.write(f"NORMAL mode: {reason}; continuing automatically.")
            _run_auto(runtime, console)
    return True


def _plan_attention(view: Any, reasons: tuple[str, ...]) -> AttentionRequest:
    task_lines = [
        f"{index}. {str(getattr(task, 'title', '') or 'Project step')}"
        for index, task in enumerate(tuple(getattr(view, "tasks", ()) or ())[:4], 1)
    ]
    summary = str(getattr(view, "plan_summary", "") or "I prepared a focused plan.")
    message = "\n".join([summary, *task_lines][:6])
    return AttentionRequest(
        id=f"plan:{getattr(view, 'goal_id', 'goal')}:r{getattr(view, 'plan_revision', 0)}:{time.monotonic_ns()}",
        kind=AttentionKind.PLAN_REVIEW,
        title="Review this plan",
        message=message,
        options=(
            AttentionOption(
                "start", "Approve and start", "start",
                description="Approve exactly this revision and begin execution.",
                shortcut="s", primary=True,
            ),
            AttentionOption(
                "change", "Request changes", "change",
                description="Describe what the plan should change before any work starts.",
                shortcut="c",
            ),
            AttentionOption(
                "cancel", "Keep paused", "cancel",
                description="Leave this revision unapproved.", shortcut="n",
            ),
        ),
        details="\n".join(reasons) or "Every implementation plan requires explicit approval.",
    )


def _action_attention(store: WorkspaceUIStore) -> AttentionRequest:
    snapshot = store.snapshot()
    running = snapshot.running
    ar = snapshot.locale == "ar"
    def copy(en: str, arabic: str) -> str:
        return arabic if ar else en
    values = [
        AttentionOption("new", copy("New task", "مهمة جديدة"), "new", description=copy("Start a separate goal.", "ابدأ هدفًا منفصلًا."), shortcut="n", primary=not running),
        AttentionOption("stop", copy("Stop safely", "إيقاف آمن"), "stop", description=copy("Save a checkpoint after the current action.", "احفظ نقطة استعادة بعد الإجراء الحالي."), shortcut="s"),
        AttentionOption("status", copy("Task status", "حالة المهمة"), "status", description=copy("Show the durable goal and plan state.", "اعرض حالة الهدف والخطة المحفوظة."), shortcut="t"),
        AttentionOption("changes", copy("Review changes", "مراجعة التغييرات"), "changes", description=copy("Show the current Git diff.", "اعرض تغييرات Git الحالية."), shortcut="r"),
        AttentionOption("result", copy("Managed previews", "المعاينات المُدارة"), "result", description=copy("List running previews and processes.", "اعرض المعاينات والعمليات الجارية."), shortcut="o"),
        AttentionOption("permissions", copy("Permissions", "الصلاحيات"), "permissions", description=copy("Choose normal or ready Docker access.", "اختر الوصول العادي أو Docker الجاهز."), shortcut="p"),
        AttentionOption("advanced", copy("Advanced details", "تفاصيل متقدمة"), "advanced", description=copy("Show technical activity in the workspace.", "اعرض النشاط التقني في مساحة العمل."), shortcut="a"),
    ]
    return AttentionRequest(
        id=f"actions:{time.monotonic_ns()}",
        kind=AttentionKind.QUESTION,
        title=copy("What would you like to do?", "ماذا تريد أن تفعل؟"),
        options=tuple(values),
    )


def _persistent_interactive_loop(
    runtime: AgentRuntime,
    console: ConsoleUI,
    preferences: SessionPreferences,
) -> None:
    """Own keyboard and rendering in one prompt_toolkit application for the session."""

    store = WorkspaceUIStore()
    store.update_identity(
        workspace=str(runtime.workspace),
        model=str(runtime.model_name),
        status=str(runtime.dashboard().status),
    )
    console.bind_workspace_store(store)
    inbox: Queue[WorkspaceInput] = Queue()
    stop = Event()
    work_running = Event()

    def submit(item: WorkspaceInput) -> None:
        inbox.put(item)

    def interrupt() -> None:
        try:
            runtime.checkpoint_interrupt()
            if work_running.is_set():
                store.set_activity(
                    ActivityStage.PAUSED,
                    "Checkpoint saved; waiting for the current action to return",
                    running=True,
                )
            else:
                store.set_activity(ActivityStage.PAUSED, "Checkpoint saved", running=False)
        except Exception as exc:
            store.append_log(f"interrupt: {exc}")

    def exit_session() -> bool:
        if work_running.is_set():
            interrupt()
            store.append_transcript(
                "assistant",
                "Exit will be available after the current action reaches its saved checkpoint.",
            )
            return False
        stop.set()
        store.mark_exit()
        return True

    app = PersistentWorkspaceApp(
        store,
        on_input=submit,
        on_interrupt=interrupt,
        on_exit=exit_session,
        output=console.stream,
        no_color=not console.color,
    )

    def controller() -> None:
        active_work: Thread | None = None
        work_done = Event()
        work_errors: list[BaseException] = []
        work_results: list[bool] = []
        last_command: UserCommand | None = None
        shown_questions: set[str] = set()
        reviewed_plans: set[str] = set()
        completed_goals: set[str] = set()
        queued: list[str] = []
        pending_new_task: list[str] = []
        slow_prompted = False
        slow_request_id: str | None = None

        def work(command: UserCommand) -> None:
            try:
                work_results.append(execute_command(runtime, console, command, preferences))
            except BaseException as exc:
                work_errors.append(exc)
            finally:
                work_running.clear()
                work_done.set()

        def start(command: UserCommand) -> None:
            nonlocal active_work, last_command, slow_prompted
            last_command = command
            slow_prompted = False
            work_done.clear()
            active_work = Thread(
                target=work,
                args=(command,),
                name="ga3bad-persistent-work",
                daemon=False,
            )
            console.set_background_working(True)
            work_running.set()
            store.set_activity(ActivityStage.UNDERSTANDING, "Understanding your request", running=True)
            active_work.start()

        def ask_custom(title: str, message: str) -> str:
            resolution = store.request_attention(
                AttentionRequest(
                    id=f"custom:{time.monotonic_ns()}",
                    kind=AttentionKind.QUESTION,
                    title=title,
                    message=message,
                    options=(AttentionOption("cancel", "Cancel", "", shortcut="c"),),
                    allow_custom=True,
                )
            )
            return resolution.text.strip() if resolution.key == "custom" else ""

        def handle_permissions() -> None:
            sandbox = (
                runtime.permission_adapter.sandbox
                if runtime.permission_adapter is not None
                else DockerSandbox()
            )
            status = sandbox.status()
            options = [
                AttentionOption(
                    "normal", "Normal", "normal",
                    description="Ask before risky actions on this computer.",
                    shortcut="n", primary=runtime.access_level == "normal",
                )
            ]
            if status.ready:
                options.append(
                    AttentionOption(
                        "full", "Full Docker", "full",
                        description="Run build and test actions in the ready Docker sandbox.",
                        shortcut="f", primary=runtime.access_level == "full",
                    )
                )
            options.append(AttentionOption("cancel", "Cancel", "", shortcut="c"))
            choice = store.request_attention(
                AttentionRequest(
                    id=f"permissions:{time.monotonic_ns()}",
                    kind=AttentionKind.APPROVAL,
                    title="Choose permissions",
                    message=(
                        "Full Docker is unavailable until /setup completes successfully."
                        if not status.ready
                        else "Choose where project commands may run."
                    ),
                    options=tuple(options),
                )
            )
            if choice.value:
                start(parse_command(f"/permissions {choice.value}"))

        def handle_model() -> None:
            goal = runtime.active_goal()
            if goal is not None and goal.status in {
                GoalStatus.RUNNING, GoalStatus.VERIFYING, GoalStatus.REVIEWING, GoalStatus.RECOVERING,
            }:
                store.append_transcript(
                    "assistant",
                    "Pause at a safe checkpoint before changing models.",
                )
                return
            store.set_activity(ActivityStage.UNDERSTANDING, "Checking available models", running=True)
            catalog = ModelCatalog()
            models = catalog.discover()
            if not models:
                detail = "; ".join(f"{item.source}: {item.message}" for item in catalog.diagnostics)
                store.set_activity(ActivityStage.PROBLEM, "No model is currently available", running=False)
                store.append_transcript(
                    "assistant",
                    "No tool-capable model is available." + (f" {detail}" if detail else ""),
                )
                return
            by_key = {f"model-{index}": descriptor for index, descriptor in enumerate(models, 1)}
            options = tuple(
                AttentionOption(
                    key,
                    descriptor.display_name,
                    key,
                    description=(
                        f"{descriptor.provider} · {descriptor.execution_class.value} · "
                        + ("configured; connection is verified on first use" if descriptor.source == "environment" else "tool calling verified")
                    ),
                    shortcut=str(index) if index < 10 else "",
                    primary=descriptor.model == runtime.model_name,
                )
                for index, (key, descriptor) in enumerate(by_key.items(), 1)
            )
            resolution = store.request_attention(
                AttentionRequest(
                    id=f"model:{time.monotonic_ns()}",
                    kind=AttentionKind.QUESTION,
                    title="Choose a model",
                    message="Cloud credentials are configuration evidence; connectivity is checked on first use.",
                    options=options,
                )
            )
            descriptor = by_key.get(resolution.value)
            if descriptor is None:
                store.set_activity(ActivityStage.PAUSED, "Model selection cancelled", running=False)
                return
            provider = descriptor.create_provider()
            setattr(provider, "reasoning_effort", runtime.reasoning_effort)
            runtime.replace_provider(provider, descriptor)
            console.set_runtime_identity(
                access_level=runtime.access_level,
                execution_class=runtime.execution_class,
                model=runtime.model_name,
                reasoning_effort=runtime.reasoning_effort,
                workspace=str(runtime.workspace),
            )
            store.set_activity(ActivityStage.IDLE, "Ready", running=False)
            store.append_transcript(
                "assistant",
                f"Model changed to {descriptor.provider}/{descriptor.model} ({descriptor.execution_class.value}).",
            )

        def handle_workflow_mode() -> None:
            issue = runtime.ultra_readiness_issue()
            options = [
                AttentionOption(
                    "plan", "Plan", "plan",
                    description="Create and revise a plan without executing tools.",
                    shortcut="p", primary=preferences.mode is InteractionMode.PLAN,
                ),
                AttentionOption(
                    "normal", "Normal", "normal",
                    description="One durable goal with plan, review, and automatic execution.",
                    shortcut="n", primary=preferences.mode is InteractionMode.NORMAL,
                )
            ]
            if not issue:
                options.append(
                    AttentionOption(
                        "ultra", "Ultra", "ultra",
                        description="Recursive specialists and deeper integration gates.",
                        shortcut="u", primary=preferences.mode is InteractionMode.ULTRA,
                    )
                )
            options.append(AttentionOption("cancel", "Cancel", "", shortcut="c"))
            resolution = store.request_attention(
                AttentionRequest(
                    id=f"mode:{time.monotonic_ns()}",
                    kind=AttentionKind.QUESTION,
                    title="Choose workflow mode",
                    message=(f"Ultra is unavailable: {issue}" if issue else "This changes orchestration, not the Simple/Advanced display."),
                    options=tuple(options),
                )
            )
            if resolution.value:
                _set_interaction_mode(
                    runtime, console, preferences, resolution.value, detailed=False
                )

        def handle_actions() -> None:
            resolution = store.request_attention(_action_attention(store))
            action = resolution.value
            if action == "advanced":
                store.set_mode(ExperienceMode.ADVANCED)
            elif action == "stop":
                interrupt()
            elif action == "changes":
                start(parse_command("/diff"))
            elif action == "result":
                start(parse_command("/processes"))
            elif action == "status":
                start(parse_command("/status"))
            elif action == "permissions":
                handle_permissions()
            elif action == "new":
                objective = ask_custom("Start a new task", "Describe what you want to build or change.")
                if objective:
                    current_goal = runtime.active_goal()
                    if current_goal is not None and current_goal.status not in {
                        GoalStatus.COMPLETED, GoalStatus.CANCELLED,
                    }:
                        runtime.cancel("CANCEL")
                    store.observe_user_text(objective)
                    store.append_transcript("user", objective)
                    start(parse_command(objective))

        try:
            _show_runtime_state(runtime, console, force=True)
            while not stop.is_set():
                if active_work is not None and work_done.is_set():
                    active_work.join()
                    active_work = None
                    console.set_background_working(False)
                    work_done.clear()
                    if slow_request_id is not None:
                        active_attention = store.active_attention()
                        if active_attention is not None and active_attention.id == slow_request_id:
                            store.resolve_attention("keep")
                        store.take_attention_result(slow_request_id)
                        slow_request_id = None
                    if pending_new_task:
                        objective = pending_new_task.pop(0)
                        pending_new_task.clear()
                        queued.clear()
                        store.set_queued_count(0)
                        current_goal = runtime.active_goal()
                        if current_goal is not None and current_goal.status not in {
                            GoalStatus.COMPLETED, GoalStatus.CANCELLED,
                        }:
                            runtime.cancel("CANCEL")
                        work_errors.clear()
                        work_results.clear()
                        store.append_transcript(
                            "assistant",
                            "The previous task reached a checkpoint. Starting your new task now.",
                        )
                        start(parse_command(objective))
                        continue
                    for guidance in tuple(queued):
                        try:
                            runtime.add_guidance(guidance)
                        except Exception as exc:
                            store.append_log(f"queued guidance: {exc}")
                    queued.clear()
                    store.set_queued_count(0)
                    if work_errors:
                        exc = work_errors.pop(0)
                        store.append_log(f"error: {exc}")
                        store.set_activity(ActivityStage.PROBLEM, "I could not finish that step", running=False)
                        resolution = store.request_attention(
                            AttentionRequest(
                                id=f"recovery:{time.monotonic_ns()}",
                                kind=AttentionKind.RECOVERY,
                                title="That step did not finish",
                                message="You can retry, change your request, or inspect the technical details.",
                                options=(
                                    AttentionOption("retry", "Retry", "retry", shortcut="r", primary=True),
                                    AttentionOption("change", "Change", "change", shortcut="c"),
                                    AttentionOption("details", "Details", "details", shortcut="d"),
                                ),
                            )
                        )
                        if resolution.value == "retry" and last_command is not None:
                            start(last_command)
                        elif resolution.value == "change":
                            feedback = ask_custom(
                                "Change the request",
                                "Describe what should change before I try again.",
                            )
                            if feedback:
                                goal = runtime.active_goal()
                                if goal is not None and goal.active_plan_revision is not None:
                                    start(parse_command(f"/replan {feedback}"))
                                elif last_command is not None:
                                    runtime.add_guidance(feedback)
                                    start(last_command)
                        elif resolution.value == "details":
                            store.set_mode(ExperienceMode.ADVANCED)
                        continue
                    if work_results and work_results.pop(0) is False:
                        stop.set()
                        store.mark_exit()
                        break

                if slow_request_id is not None:
                    resolution = store.take_attention_result(slow_request_id)
                    if resolution is not None:
                        slow_request_id = None
                        if resolution.value == "stop":
                            interrupt()

                if active_work is not None and not slow_prompted:
                    snapshot = store.snapshot()
                    last_signal = snapshot.activity.last_signal_at
                    quiet = (
                        0.0
                        if last_signal is None
                        else time.monotonic() - last_signal
                    )
                    if quiet >= 60 and snapshot.attention is None:
                        slow_prompted = True
                        slow_request_id = f"slow:{time.monotonic_ns()}"
                        store.present_attention(
                            AttentionRequest(
                                id=slow_request_id,
                                kind=AttentionKind.RECOVERY,
                                title="This is taking longer than usual",
                                message="Work is still active. Nothing has been rejected or lost.",
                                options=(
                                    AttentionOption("keep", "Keep waiting", "keep", shortcut="k", primary=True),
                                    AttentionOption("stop", "Stop safely", "stop", shortcut="s"),
                                ),
                            )
                        )

                if active_work is None:
                    session = question_session(runtime)
                    if session is not None and session.current is not None:
                        question = session.current
                        key = f"{session.source}:{question.get('id')}"
                        if key not in shown_questions:
                            shown_questions.add(key)
                            store.set_activity(ActivityStage.PAUSED, "One choice is needed", running=False)
                            resolution = store.request_attention(question_attention(session))
                            answer = resolution.text if resolution.key == "custom" else resolution.value
                            answer_question(runtime, session, str(question.get("id", "")), answer or "1")
                            _show_runtime_state(runtime, console, force=True)
                            continue
                        continue

                    goal = runtime.active_goal()
                    goal_id = str(getattr(goal, "id", ""))
                    if goal is not None and goal.status == GoalStatus.AWAITING_PLAN_APPROVAL:
                        view = runtime.dashboard()
                        plan_key = f"{goal_id}:r{view.plan_revision}"
                        if plan_key not in reviewed_plans:
                            reviewed_plans.add(plan_key)
                            reasons = plan_review_reasons(view)
                            store.set_activity(ActivityStage.PAUSED, "Review the plan", running=False)
                            resolution = store.request_attention(_plan_attention(view, reasons))
                            if resolution.value == "start":
                                start(parse_command(f"/approve {view.plan_revision}"))
                            elif resolution.value == "change":
                                feedback = ask_custom("Change the plan", "What should I change?")
                                if feedback:
                                    start(parse_command(f"/replan {feedback}"))
                            else:
                                store.append_transcript("assistant", "The plan is paused. Nothing has started.")
                            continue

                    if goal is not None and str(goal.status.value) == "completed" and goal_id not in completed_goals:
                        completed_goals.add(goal_id)
                        view = runtime.dashboard()
                        completed = sum(task.status in {"done", "skipped"} for task in view.tasks)
                        actual_files = tuple(dict.fromkeys(
                            str(path)
                            for change_set in tuple(getattr(goal, "metadata", {}).get("goal_change_sets", ()) or ())
                            if isinstance(change_set, Mapping)
                            for path in tuple(change_set.get("changed_files", ()) or ())
                            if str(path).strip()
                        ))
                        verified = tuple(
                            item.summary
                            for item in runtime.store.list_evidence(goal_id)
                            if getattr(item, "verified", False)
                        )
                        files_receipt = (
                            " Changed files: " + ", ".join(actual_files[:8]) + ("…" if len(actual_files) > 8 else "") + "."
                            if actual_files
                            else " No workspace file changes were recorded."
                        )
                        verification_receipt = (
                            " Verification: " + " | ".join(" ".join(item.split())[:180] for item in verified[-3:]) + "."
                            if verified
                            else " Verification: no verified evidence receipt was recorded."
                        )
                        store.set_activity(ActivityStage.DONE, "Done", completed=completed, total=len(view.tasks), running=False)
                        store.append_transcript(
                            "assistant",
                            (
                                f"Done. {completed}/{len(view.tasks)} planned steps completed. "
                                + files_receipt
                                + verification_receipt
                                + " Press Ctrl+K for status, changes, and managed previews."
                            ),
                        )

                try:
                    item = inbox.get(timeout=0.1)
                except Empty:
                    continue
                text = item.text.strip()
                if not text:
                    if item.kind == "actions":
                        if active_work is None:
                            handle_actions()
                    elif item.kind == "model":
                        if active_work is None:
                            handle_model()
                    elif item.kind == "permissions":
                        if active_work is None:
                            handle_permissions()
                    continue
                if text == "/" or item.kind == "actions":
                    if active_work is None:
                        handle_actions()
                    else:
                        store.append_transcript("assistant", "I’m still working. Press Ctrl+C to stop safely, or send guidance here.")
                    continue
                store.observe_user_text(text)
                store.append_transcript("user", text)
                if active_work is not None:
                    try:
                        active_command = parse_command(text)
                    except CommandParseError as exc:
                        store.append_transcript("assistant", f"I couldn’t understand that: {exc}")
                        continue
                    observer_kinds = {
                        CommandKind.STATUS, CommandKind.PLAN, CommandKind.DIFF,
                        CommandKind.PROCESSES, CommandKind.THINKING, CommandKind.AGENTS,
                        CommandKind.AGENT, CommandKind.TREE, CommandKind.METRICS,
                        CommandKind.MEMORY, CommandKind.TRACE, CommandKind.INSIGHTS,
                        CommandKind.HISTORY, CommandKind.VERSIONS, CommandKind.HELP,
                        CommandKind.KEYMAP,
                    }
                    if active_command.kind in observer_kinds:
                        execute_command(runtime, console, active_command, preferences)
                        continue
                    if active_command.kind is CommandKind.PAUSE:
                        interrupt()
                        continue
                    if active_command.kind is CommandKind.QUIT:
                        exit_session()
                        continue
                    if active_command.kind is not CommandKind.TEXT:
                        store.append_transcript(
                            "assistant",
                            "That command needs the current action to reach a checkpoint first. Use Ctrl+C to request one.",
                        )
                        continue
                    goal = runtime.active_goal()
                    resolution = store.request_attention(
                        AttentionRequest(
                            id=f"active-intent:{time.monotonic_ns()}",
                            kind=AttentionKind.QUESTION,
                            title="How should this message be used?",
                            message=f"Current task: {str(getattr(goal, 'objective', 'active task'))[:180]}",
                            options=(
                                AttentionOption(
                                    "update", "Update current task", "update",
                                    description="Save this as guidance at the next safe point.",
                                    shortcut="u", primary=True,
                                ),
                                AttentionOption(
                                    "new", "Start a new task", "new",
                                    description="Checkpoint and cancel the current goal, then plan this separately.",
                                    shortcut="n",
                                ),
                                AttentionOption("cancel", "Cancel", "cancel", shortcut="c"),
                            ),
                        )
                    )
                    if resolution.value == "new":
                        pending_new_task.append(text)
                        interrupt()
                        store.append_transcript(
                            "assistant",
                            "New task queued. I’ll switch after the current action reaches its checkpoint.",
                        )
                    elif resolution.value == "update":
                        try:
                            runtime.add_guidance(text)
                            store.append_transcript("assistant", "Guidance saved for the current task.")
                        except Exception:
                            queued.append(text)
                            store.set_queued_count(len(queued))
                    continue
                try:
                    command = parse_command(text)
                except CommandParseError as exc:
                    store.append_transcript("assistant", f"I couldn’t understand that: {exc}")
                    continue
                if command.kind is CommandKind.MODEL and not command.args.get("model") and not command.args.get("effort"):
                    handle_model()
                    continue
                if command.kind is CommandKind.PERMISSIONS and not command.args.get("level"):
                    handle_permissions()
                    continue
                if command.kind is CommandKind.MODE and not command.args.get("mode"):
                    handle_workflow_mode()
                    continue
                goal = runtime.active_goal()
                if (
                    command.kind is CommandKind.TEXT
                    and goal is not None
                    and goal.status not in {GoalStatus.COMPLETED, GoalStatus.CANCELLED}
                ):
                    resolution = store.request_attention(
                        AttentionRequest(
                            id=f"intent:{time.monotonic_ns()}",
                            kind=AttentionKind.QUESTION,
                            title="How should this affect the current task?",
                            message=f"Current task: {str(getattr(goal, 'objective', 'active task'))[:180]}",
                            options=(
                                AttentionOption(
                                    "update", "Update current task", "update",
                                    description="Treat this message as guidance for the active goal.",
                                    shortcut="u", primary=True,
                                ),
                                AttentionOption(
                                    "replace", "Start a new task", "replace",
                                    description="Cancel the current goal and plan this request separately.",
                                    shortcut="n",
                                ),
                                AttentionOption("cancel", "Cancel", "cancel", shortcut="c"),
                            ),
                        )
                    )
                    if resolution.value == "cancel":
                        continue
                    if resolution.value == "replace":
                        runtime.cancel("CANCEL")
                start(command)
        finally:
            stop.set()
            store.mark_exit()
            app.stop()

    controller_thread = Thread(
        target=controller,
        name="ga3bad-workspace-controller",
        daemon=True,
    )
    controller_thread.start()
    try:
        app.run()
    finally:
        stop.set()
        store.mark_exit()
        controller_thread.join(timeout=5)
        console.bind_workspace_store(None)


def _legacy_interactive_loop(
    runtime: AgentRuntime,
    console: ConsoleUI,
    preferences: SessionPreferences,
) -> None:
    _show_runtime_state(runtime, console, force=True)
    active_work: Thread | None = None
    work_done = Event()
    work_errors: list[BaseException] = []
    deferred_picker_question: str | None = None
    deferred_plan_review: str | None = None
    queued_guidance: list[str] = []

    def work(command: UserCommand) -> None:
        try:
            execute_command(runtime, console, command, preferences)
        except BaseException as exc:
            work_errors.append(exc)
        finally:
            work_done.set()
            console.wake_prompt()

    background_kinds = {
        CommandKind.TEXT,
        CommandKind.GOAL,
        CommandKind.RUN,
        CommandKind.AUTO,
        CommandKind.APPROVE,
        CommandKind.RESUME,
    }
    active_observer_kinds = {
        CommandKind.STATUS,
        CommandKind.THINKING,
        CommandKind.AGENTS,
        CommandKind.AGENT,
        CommandKind.TREE,
        CommandKind.PLAN,
        CommandKind.QUESTIONS,
        CommandKind.METRICS,
        CommandKind.MEMORY,
        CommandKind.TRACE,
        CommandKind.INSIGHTS,
        CommandKind.HISTORY,
        CommandKind.DIFF,
        CommandKind.VERSIONS,
        CommandKind.PROCESSES,
        CommandKind.HELP,
        CommandKind.KEYMAP,
        CommandKind.PAUSE,
    }
    while True:
        if console.has_pending_approval() is True:
            console.resolve_pending_approval()
            continue
        if active_work is not None and work_done.is_set():
            active_work.join()
            active_work = None
            console.set_background_working(False)
            work_done.clear()
            if work_errors:
                exc = work_errors.pop(0)
                if isinstance(exc, KeyboardInterrupt):
                    runtime.checkpoint_interrupt()
                else:
                    console.write(f"error: {exc}")
            if queued_guidance:
                saved = 0
                for guidance in tuple(queued_guidance):
                    try:
                        runtime.add_guidance(guidance)
                        saved += 1
                    except (RuntimeErrorBase, StateStoreError, ValueError) as exc:
                        console.write(f"Queued guidance could not be saved: {exc}")
                queued_guidance.clear()
                if saved:
                    console.write(
                        f"Saved {saved} queued guidance note(s) at the safe checkpoint."
                    )
        if active_work is None:
            session = question_session(runtime)
            pending_id = (
                f"{session.source}:{session.current.get('id')}"
                if session is not None and session.current is not None
                else None
            )
            if pending_id and pending_id != deferred_picker_question:
                if _answer_pending_question_with_picker(runtime, console):
                    deferred_picker_question = None
                    _show_runtime_state(runtime, console)
                    continue
                deferred_picker_question = pending_id
                _show_questions(runtime, console)
            elif pending_id is None:
                deferred_picker_question = None
            if session is None:
                goal = runtime.active_goal()
                plan_key = (
                    f"{goal.id}:r{runtime.dashboard().plan_revision}"
                    if goal is not None
                    and goal.status == GoalStatus.AWAITING_PLAN_APPROVAL
                    else None
                )
                if plan_key and plan_key != deferred_plan_review:
                    deferred_plan_review = plan_key
                    review_command = _review_pending_plan_with_picker(runtime, console)
                    if review_command is not None:
                        work_done.clear()
                        active_work = Thread(
                            target=work,
                            args=(review_command,),
                            name="ga3bad-background-work",
                            daemon=False,
                        )
                        console.set_background_working(True)
                        active_work.start()
                        continue
                elif plan_key is None:
                    deferred_plan_review = None
        try:
            line = console.prompt()
        except (ApprovalPromptRequested, WorkspaceRefreshRequested):
            continue
        except EOFError:
            console.write("\nInput closed. Durable goal state is saved.")
            return
        except KeyboardInterrupt:
            runtime.checkpoint_interrupt()
            continue
        try:
            command = parse_command(line)
            if (
                active_work is not None
                and command.kind == CommandKind.TEXT
                and str(command.args.get("text", "")).strip()
            ):
                guidance = str(command.args["text"]).strip()
                queued_guidance.append(guidance)
                console.write(
                    "Guidance queued for the next safe checkpoint. "
                    "Use /pause if it must apply before more work runs."
                )
                continue
            if active_work is not None and command.kind not in active_observer_kinds:
                console.write(
                    "This action waits for a safe checkpoint. Use /status to inspect work "
                    "or /pause to stop cooperatively."
                )
                continue
            if active_work is not None and command.kind == CommandKind.QUIT:
                console.write("Active work needs a checkpoint first; use /pause, then /quit.")
                continue
            if (
                active_work is None
                and command.kind in background_kinds
                and console.live_activity_enabled
            ):
                work_done.clear()
                active_work = Thread(
                    target=work,
                    args=(command,),
                    name="ga3bad-background-work",
                    daemon=False,
                )
                console.set_background_working(True)
                active_work.start()
                continue
            if not execute_command(runtime, console, command, preferences):
                return
        except KeyboardInterrupt:
            runtime.checkpoint_interrupt()
        except UserExitRequested:
            return
        except (
            CommandParseError,
            RuntimeErrorBase,
            StateStoreError,
            SandboxError,
            RuntimeError,
            DomainError,
            ValueError,
        ) as exc:
            console.write(f"error: {exc}")


def interactive_loop(
    runtime: AgentRuntime,
    console: ConsoleUI,
    preferences: SessionPreferences,
) -> None:
    use_persistent = (
        not console.plain
        and console.live_activity_enabled
        and os.getenv("GA3BAD_LEGACY_UI", "").strip().lower() not in {"1", "true", "yes", "on"}
    )
    if not use_persistent:
        _legacy_interactive_loop(runtime, console, preferences)
        return
    try:
        _persistent_interactive_loop(runtime, console, preferences)
    except RuntimeError as exc:
        console.bind_workspace_store(None)
        console.write(f"The simplified workspace could not start ({exc}); using the compatible terminal view.")
        _legacy_interactive_loop(runtime, console, preferences)


def _interactive_setup(
    args: argparse.Namespace,
    console: ConsoleUI,
    selected_provider: str,
) -> tuple[Path, ModelDescriptor, DockerSandbox, AccessLevel, SessionPreferences] | None:
    """Run the visible setup as a reversible, one-decision-per-screen flow."""

    plain_env = os.getenv("GA3BAD_PLAIN_UI", "").strip().lower() in {
        "1", "true", "yes", "on",
    }
    rich = not (bool(getattr(console, "plain", False)) or plain_env) and rich_terminal_available(
        input_func=console.input_func,
        output=console.stream,
    )
    if rich:
        play_intro()
    else:
        console.show_brand()

    catalog = ModelCatalog()
    sandbox = DockerSandbox()
    if args.setup_sandbox:
        config = sandbox.setup()
        console.write(f"Full sandbox ready · {config.image} · user {config.container_user}")

    workspace: Path | None = None
    descriptor: ModelDescriptor | None = None
    requested_access: AccessLevel | None = None
    selected_mode: InteractionMode | None = None

    if args.workspace:
        workspace = _resolve_workspace(
            Path(args.workspace).expanduser(),
            create=args.create_workspace,
        )
    elif args.create_workspace:
        raise ValueError("--create-workspace requires --workspace")

    if args.model is not None:
        environment_provider = get_provider(selected_provider)
        selected_model = args.model or str(getattr(environment_provider, "model", ""))
        descriptor = _descriptor_for_explicit_model(
            selected_provider,
            selected_model,
            catalog=catalog if selected_provider == "ollama" else None,
        )
    if args.permissions:
        requested_access = AccessLevel.parse(args.permissions)
    if args.mode:
        selected_mode = InteractionMode.parse(args.mode)

    steps = []
    if workspace is None:
        steps.append(("workspace", "1. Choose a workspace"))
    steps.append(("protection", "2. Protect this project"))
    if descriptor is None:
        steps.append(("model", "3. Choose a model"))
    if requested_access is None:
        steps.append(("permissions", "4. Set permissions"))
    steps.append(("mode", "5. Choose workflow mode"))

    total = len(steps)
    index = 0
    while index < total:
        stage, step_label = steps[index]
        try:
            if stage == "workspace":
                workspace = choose_workspace(
                    args.projects_root,
                    input_func=console.input_func,
                    output=console.stream,
                    rich=rich,
                    initial=workspace,
                    step_label=step_label,
                    no_color=not console.color,
                    reduced_motion=console.reduced_motion,
                )
            elif stage == "protection":
                assert workspace is not None
                choose_project_protection(
                    workspace,
                    input_func=console.input_func,
                    output=console.stream,
                    rich=rich,
                    step_label=step_label,
                    no_color=not console.color,
                    reduced_motion=console.reduced_motion,
                )
            elif stage == "model":
                descriptor = choose_model(
                    catalog,
                    input_func=console.input_func,
                    output=console.stream,
                    rich=rich,
                    initial=descriptor.model if descriptor is not None else None,
                    step_label=step_label,
                    no_color=not console.color,
                    reduced_motion=console.reduced_motion,
                )
            elif stage == "permissions":
                requested_access = choose_access_level(
                    input_func=console.input_func,
                    output=console.stream,
                    sandbox=sandbox,
                    rich=rich,
                    initial=requested_access or AccessLevel.NORMAL,
                    step_label=step_label,
                    no_color=not console.color,
                    reduced_motion=console.reduced_motion,
                )
            elif stage == "mode":
                selected_mode = choose_interaction_mode(
                    input_func=console.input_func,
                    output=console.stream,
                    rich=rich,
                    initial=selected_mode or InteractionMode.NORMAL,
                    step_label=step_label,
                    no_color=not console.color,
                    reduced_motion=console.reduced_motion,
                    ultra_disabled_reason="",
                )
        except UserExitRequested:
            return None
        except PickerBack:
            if index > 0:
                index -= 1
                continue
            if rich:
                play_intro()
            continue
        index += 1

    assert workspace is not None
    assert descriptor is not None
    assert requested_access is not None
    return (
        workspace,
        descriptor,
        sandbox,
        requested_access,
        SessionPreferences(mode=selected_mode or InteractionMode.NORMAL),
    )


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    just_fix_windows_console()
    console = ConsoleUI(
        color=False if args.no_color else None,
        plain=bool(args.plain),
        reduced_motion=bool(args.reduced_motion),
    )
    preferences = SessionPreferences()
    store: StateStore | None = None
    runtime: AgentRuntime | None = None
    try:
        load_dotenv(APP_ROOT / ".env", override=False)
        console.plain = console.plain or os.getenv("GA3BAD_PLAIN_UI", "").strip().lower() in {
            "1", "true", "yes", "on",
        }
        console.reduced_motion = console.reduced_motion or os.getenv(
            "GA3BAD_REDUCED_MOTION", ""
        ).strip().lower() in {"1", "true", "yes", "on"}
        selected_provider = _configure_provider_environment(args.provider, args.model)
        interactive_launch = bool(args.interactive or not args.command)
        if interactive_launch:
            setup = _interactive_setup(args, console, selected_provider)
            if setup is None:
                return 0
            workspace, descriptor, sandbox, requested_access, preferences = setup
        else:
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
                workspace = choose_workspace(
                    args.projects_root,
                    rich=False if args.plain else None,
                )

            catalog = ModelCatalog()
            if args.model is None:
                model_required = args.auto
                for raw in args.command:
                    try:
                        kind = parse_command(raw).kind
                    except CommandParseError:
                        kind = CommandKind.TEXT
                    if kind in {
                        CommandKind.TEXT,
                        CommandKind.GOAL,
                        CommandKind.APPROVE,
                        CommandKind.REJECT,
                        CommandKind.REPLAN,
                        CommandKind.RUN,
                        CommandKind.AUTO,
                        CommandKind.RESUME,
                    }:
                        model_required = True
                if model_required:
                    raise ValueError(
                        "--model is required for non-interactive model work; offline inspection commands are exempt"
                    )
            environment_provider = get_provider(selected_provider)
            selected_model = args.model or str(getattr(environment_provider, "model", ""))
            descriptor = _descriptor_for_explicit_model(
                selected_provider,
                selected_model,
                catalog=catalog if selected_provider == "ollama" and args.model is not None else None,
            )
            sandbox = DockerSandbox()
            if args.setup_sandbox:
                sandbox.setup()
                console.write("Full sandbox setup is ready.")
            requested_access = (
                AccessLevel.parse(args.permissions)
                if args.permissions
                else AccessLevel.NORMAL
            )
            preferences = SessionPreferences.from_env(args.mode)

        os.environ["LLM_PROVIDER"] = descriptor.provider
        provider = descriptor.create_provider()
        _validated_model_name(str(getattr(provider, "model", "")))
        permission_adapter = PermissionAdapter(requested_access, sandbox)
        if permission_adapter.selection.reason:
            console.write(permission_adapter.selection.reason)

        console.set_mode(preferences.mode)
        console.set_runtime_identity(
            access_level=permission_adapter.access_level.value,
            execution_class=descriptor.execution_class.value,
            model=descriptor.model,
            workspace=str(workspace),
        )
        bus = EventBus()
        bus.subscribe(console.on_event)
        store = StateStore(workspace)
        runtime = AgentRuntime(
            provider,
            store,
            workspace,
            events=bus,
            approval=console.confirm_action_decision,
            model_descriptor=descriptor,
            permission_adapter=permission_adapter,
            auto_promote_ultra=False,
            workflow_mode=preferences.mode.value,
            direct_normal_execution=True,
        )
        if not interactive_launch:
            console.write(
                f"Ready · {runtime.provider_name}/{runtime.model_name} · "
                f"{runtime.execution_class} · {runtime.access_level} · state .coding-agent/state.db"
            )

        for raw in args.command:
            try:
                if not execute_command(runtime, console, parse_command(raw), preferences):
                    return 0
            except KeyboardInterrupt:
                runtime.checkpoint_interrupt()
                return 130
            except UserExitRequested:
                return 0
        if args.auto:
            try:
                if preferences.mode == InteractionMode.ULTRA:
                    if runtime.ultra_session is not None and runtime.ultra_session.running:
                        console.write(
                            "ULTRA convergence is active. Quality-only specialist revisions "
                            "will continue inside the approved scope until product acceptance "
                            "or a real external/scope blocker."
                        )
                        runtime.converge_ultra()
                        _show_runtime_state(runtime, console)
                else:
                    _run_auto(runtime, console)
            except KeyboardInterrupt:
                runtime.checkpoint_interrupt()
                return 130
        if args.interactive or not args.command:
            interactive_loop(runtime, console, preferences)
        return 0
    except (
        StateCorruptionError,
        StateStoreError,
        RuntimeErrorBase,
        RuntimeError,
        DomainError,
        OSError,
        ValueError,
        SandboxError,
    ) as exc:
        if args.debug:
            raise
        console.write(f"fatal: {exc}")
        return 2
    finally:
        if runtime is not None:
            runtime.close()
        if store is not None:
            store.close()
        console.close()


if __name__ == "__main__":
    raise SystemExit(main())
