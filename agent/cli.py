"""Import-safe CLI and interactive ASCII interface for the persistent runtime."""

from __future__ import annotations

import argparse
import os
import sys
import unicodedata
from collections import Counter
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
from .model_catalog import ExecutionClass, ModelCatalog, ModelDescriptor
from .models import DomainError, GoalStatus
from .providers import get_provider
from .runtime import AgentRuntime, RuntimeErrorBase, SliceResult
from .sandbox import AccessLevel, DockerSandbox, PermissionAdapter, SandboxError
from .store import StateCorruptionError, StateStore, StateStoreError
from .ui import (
    ConsoleUI,
    HELP_TEXT,
    render_agents,
    render_memory,
    render_plan,
    render_slash_menu,
    render_trace,
    render_tree,
)
from .ultra_models import AgentRunStatus, BrainSection


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


def choose_model(
    catalog: ModelCatalog,
    *,
    input_func: Callable[[str], str] = input,
    output: TextIO = sys.stdout,
) -> ModelDescriptor:
    """Require one explicit tool-capable model selection, Ollama first."""

    models = catalog.discover()
    print("Models", file=output)
    if not models:
        for diagnostic in catalog.diagnostics:
            print(f"  {diagnostic.source}: {diagnostic.message}", file=output)
        raise ValueError(
            "no tool-capable model is available; start Ollama or configure OpenAI/Gemini"
        )
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
) -> AccessLevel:
    print("Permissions", file=output)
    print("  1. normal  approvals stay enabled", file=output)
    print("  2. full    no workspace confirmations; Docker sandbox required", file=output)
    choice = input_func("permissions [normal]> ").strip().lower()
    if choice in {"2", "full"}:
        return AccessLevel.FULL
    if choice in {"", "1", "normal"}:
        return AccessLevel.NORMAL
    print("Unknown choice; using normal.", file=output)
    return AccessLevel.NORMAL


def choose_interaction_mode(
    *,
    input_func: Callable[[str], str] = input,
    output: TextIO = sys.stdout,
) -> InteractionMode:
    print("Mode", file=output)
    print("  1. plan   approve, then run manually", file=output)
    print("  2. goal   approve, then continue automatically", file=output)
    print("  3. ultra  Project Brain, nested nodes, review/test/fix/integration", file=output)
    while True:
        choice = input_func("mode> ").strip().lower()
        aliases = {"1": "plan", "2": "goal", "3": "ultra"}
        choice = aliases.get(choice, choice)
        if choice in {"plan", "goal", "ultra"}:
            return InteractionMode.parse(choice)
        print("Choose plan, goal, or ultra.", file=output)


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
        choices=(InteractionMode.PLAN.value, InteractionMode.GOAL.value, InteractionMode.ULTRA.value),
        help="Interaction mode: plan, goal, or full Project-Brain ULTRA execution.",
    )
    parser.add_argument(
        "--permissions",
        choices=(AccessLevel.NORMAL.value, AccessLevel.FULL.value),
        help="Workspace permission profile. Full is accepted only in the ready Docker sandbox.",
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
    )
    console.show_status(view)
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
    if selected == InteractionMode.ULTRA:
        console.write(
            "ULTRA mode active: GoalSpec → architecture → one master approval → "
            "nested module waves → independent review/test/fix/integration → final evidence."
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


def _execute_model(runtime: AgentRuntime, console: ConsoleUI, value: str | None) -> None:
    catalog = ModelCatalog()
    if value is None:
        descriptor = choose_model(
            catalog,
            input_func=console.input_func,
            output=console.stream,
        )
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
    runtime.replace_provider(descriptor.create_provider(), descriptor)
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
    )
    console.write(
        f"model = {descriptor.provider}/{descriptor.model} · "
        f"{descriptor.execution_class.value} (session only)"
    )


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


def _current_ultra_run(runtime: AgentRuntime) -> object | None:
    try:
        return runtime.active_ultra_run()
    except (AttributeError, StateStoreError):
        return None


def _show_questions(runtime: AgentRuntime, console: ConsoleUI) -> None:
    goal = runtime.active_goal()
    questions = (
        runtime.ultra_questions()
        if goal is not None and goal.metadata.get("ultra_run_id")
        else runtime.plan_questions()
    )
    if not questions:
        console.write("Questions\n  (none pending)")
        return
    answers = dict(goal.metadata.get("plan_answers", {})) if goal else {}
    lines = [f"Questions · {len(questions)}"]
    for item in questions:
        question_id = str(item.get("id", "?"))
        answer = answers.get(question_id)
        mark = "[x]" if answer else "[ ]"
        lines.append(f"  {mark} {question_id} · {item.get('header', '')}")
        lines.append(f"      {item.get('question', '')}")
        for option in item.get("options", ()):
            if not isinstance(option, dict):
                continue
            recommended = " (recommended)" if option.get("recommended") else ""
            lines.append(
                f"      - {option.get('label', '')}{recommended}: {option.get('description', '')}"
            )
        if answer:
            lines.append(f"      answer: {answer}")
    lines.append("  Use /answer QUESTION_ID VALUE")
    console.write("\n".join(lines))


def _show_tree(runtime: AgentRuntime, console: ConsoleUI, target: str | None) -> None:
    run = _current_ultra_run(runtime)
    if run is None:
        console.write("Project tree\n  (no ULTRA run yet)")
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
    node_titles = {
        node.id: node.title for node in runtime.store.list_work_nodes(run.id)
    }
    console.write(
        render_agents(
            runtime.store.list_agent_runs(run.id),
            include_finished=include_finished,
            node_titles=node_titles,
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


def _execute_permissions(
    runtime: AgentRuntime,
    console: ConsoleUI,
    level: str | None,
) -> None:
    if level is None:
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
            console.write(
                f"mode = {preferences.mode.value}; choose /mode plan, /mode goal, or /mode ultra"
            )
        else:
            _set_interaction_mode(runtime, console, preferences, selected)
        return True
    if command.kind == CommandKind.SETTINGS:
        _execute_settings(runtime, console, preferences, command)
        return True
    if command.kind == CommandKind.MODEL:
        _execute_model(runtime, console, command.args.get("model"))
        return True
    if command.kind == CommandKind.PERMISSIONS:
        _execute_permissions(runtime, console, command.args.get("level"))
        return True
    if command.kind == CommandKind.SETUP:
        _setup_sandbox(runtime, console)
        return True
    if command.kind == CommandKind.TREE:
        _show_tree(runtime, console, command.args.get("target"))
        return True
    if command.kind == CommandKind.AGENTS:
        _show_agents(runtime, console, include_finished=bool(command.args.get("all")))
        return True
    if command.kind == CommandKind.MEMORY:
        _show_memory(runtime, console, command.args.get("target"))
        return True
    if command.kind == CommandKind.TRACE:
        _show_trace(runtime, console, command.args.get("target"))
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
        _show_runtime_state(runtime, console)
        return True
    if command.kind == CommandKind.HISTORY:
        _show_history(runtime, console)
        return True
    if command.kind == CommandKind.AUTO:
        if preferences.mode == InteractionMode.ULTRA:
            console.write("ULTRA execution already runs in the background after master approval.")
            _show_runtime_state(runtime, console)
            return True
        _run_auto(runtime, console)
        return True

    if preferences.mode == InteractionMode.ULTRA and command.kind in {
        CommandKind.GOAL,
        CommandKind.TEXT,
    }:
        text = command.args.get("objective", command.args.get("text", "")).strip()
        if not text:
            return True
        if runtime.active_goal() is None:
            result = runtime.start_ultra(text)
        else:
            pending = [
                item
                for item in runtime.ultra_questions()
                if not runtime.active_goal().metadata.get("plan_answers", {}).get(
                    str(item.get("id"))
                )
            ]
            result = (
                runtime.answer_ultra_question(str(pending[0].get("id")), text)
                if len(pending) == 1
                else runtime.add_ultra_guidance(text)
            )
    elif preferences.mode == InteractionMode.ULTRA and command.kind == CommandKind.RUN:
        console.write("ULTRA module waves run in the background; use /agents, /tree, or /pause.")
        result = None
    else:
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
            runtime.checkpoint_interrupt()
            console.write(
                "\nCheckpoint saved; new agents will not launch until /resume. "
                "Type /quit to exit or / for controls."
            )
            _show_runtime_state(runtime, console)
            continue
        try:
            command = parse_command(line)
            if not execute_command(runtime, console, command, preferences):
                return
        except KeyboardInterrupt:
            runtime.checkpoint_interrupt()
            _show_runtime_state(runtime, console)
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


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    just_fix_windows_console()
    console = ConsoleUI(color=False if args.no_color else None)
    preferences = SessionPreferences()
    store: StateStore | None = None
    runtime: AgentRuntime | None = None
    try:
        load_dotenv(APP_ROOT / ".env", override=False)
        selected_provider = _configure_provider_environment(args.provider, args.model)
        interactive_launch = bool(args.interactive or not args.command)
        if interactive_launch:
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

        catalog = ModelCatalog()
        if interactive_launch and args.model is None:
            descriptor = choose_model(catalog, output=console.stream)
        else:
            if not interactive_launch and args.model is None:
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
        os.environ["LLM_PROVIDER"] = descriptor.provider
        provider = descriptor.create_provider()
        _validated_model_name(str(getattr(provider, "model", "")))

        sandbox = DockerSandbox()
        if args.setup_sandbox:
            sandbox.setup()
            console.write("Full sandbox setup is ready.")
        requested_access = (
            AccessLevel.parse(args.permissions)
            if args.permissions
            else choose_access_level(output=console.stream)
            if interactive_launch
            else AccessLevel.NORMAL
        )
        permission_adapter = PermissionAdapter(requested_access, sandbox)
        if permission_adapter.selection.reason:
            console.write(permission_adapter.selection.reason)

        if args.mode:
            preferences = SessionPreferences.from_env(args.mode)
        elif interactive_launch:
            preferences = SessionPreferences(mode=choose_interaction_mode(output=console.stream))
        else:
            preferences = SessionPreferences.from_env()
        console.set_mode(preferences.mode)
        console.set_runtime_identity(
            access_level=permission_adapter.access_level.value,
            execution_class=descriptor.execution_class.value,
        )
        bus = EventBus()
        bus.subscribe(console.on_event)
        store = StateStore(workspace)
        runtime = AgentRuntime(
            provider,
            store,
            workspace,
            events=bus,
            approval=console.confirm_action,
            model_descriptor=descriptor,
            permission_adapter=permission_adapter,
        )
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
                _show_runtime_state(runtime, console)
                return 130
        if args.auto and preferences.mode != InteractionMode.ULTRA:
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


if __name__ == "__main__":
    raise SystemExit(main())
