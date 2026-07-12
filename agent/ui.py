"""ASCII terminal UI for persistent goals, plans, workers, and live events."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import textwrap
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Iterable, Mapping, TextIO

try:
    from .config import InteractionMode, runtime_setting_names
    from .events import UIEvent
except ImportError:  # direct ``python agent/main.py`` compatibility
    from config import InteractionMode, runtime_setting_names  # type: ignore
    from events import UIEvent  # type: ignore

try:  # Optional at import time; basic input and the bare-/ menu remain available.
    from prompt_toolkit import ANSI, PromptSession
    from prompt_toolkit.completion import Completer, Completion
    from prompt_toolkit.patch_stdout import patch_stdout
    from prompt_toolkit.shortcuts import CompleteStyle
except ImportError:  # pragma: no cover - exercised by minimal installations
    ANSI = PromptSession = Completer = Completion = CompleteStyle = patch_stdout = None  # type: ignore


BRAND_ART = (
    "  ____    _    _____ ____    _    ____ ",
    " / ___|  / \\  |___ /| __ )  / \\  |  _ \\",
    "| |  _  / _ \\   |_ \\|  _ \\ / _ \\ | | | |",
    "| |_| |/ ___ \\ ___) | |_) / ___ \\| |_| |",
    " \\____/_/   \\_\\____/|____/_/   \\_\\____/ ",
)
BRAND_WIDTH = max(len(line) for line in BRAND_ART)
BRAND_SUBTITLE = "coding agent"

SLASH_COMMANDS: tuple[tuple[str, str], ...] = (
    ("/mode", "switch PLAN / GOAL / ULTRA mode"),
    ("/model", "choose a provider model"),
    ("/permissions", "switch NORMAL / FULL access"),
    ("/tree", "show the hierarchical project tree"),
    ("/agents", "show active and recent agents"),
    ("/memory", "inspect the Project Brain"),
    ("/trace", "inspect redacted prompts and run trace"),
    ("/insights", "show durable findings and decisions"),
    ("/questions", "show pending plan questions"),
    ("/answer", "answer a durable plan question"),
    ("/metrics", "show quality, usage, and timing metrics"),
    ("/settings", "inspect or change session settings"),
    ("/goal", "start a durable goal"),
    ("/plan", "show the complete plan and checklist"),
    ("/approve", "approve the displayed plan revision"),
    ("/reject", "reject the draft plan with feedback"),
    ("/replan", "request a revised plan with feedback"),
    ("/add", "add a checklist task"),
    ("/edit", "edit a checklist task"),
    ("/remove", "remove a checklist task"),
    ("/done", "mark a task done with evidence"),
    ("/todo", "reopen a checklist task"),
    ("/block", "mark a checklist task blocked"),
    ("/skip", "skip a checklist task"),
    ("/run", "run one bounded work slice"),
    ("/auto", "continue the approved goal until a checkpoint"),
    ("/status", "show the dashboard"),
    ("/history", "show durable activity"),
    ("/pause", "checkpoint active work"),
    ("/resume", "continue paused work"),
    ("/resolve", "reconcile uncertain crash-window work"),
    ("/cancel", "explicitly abandon the active goal"),
    ("/setup", "validate the one-time sandbox setup"),
    ("/help", "show every command"),
    ("/quit", "save state and exit"),
)


if Completer is not None:
    class SlashCommandCompleter(Completer):  # type: ignore[misc,valid-type]
        """Completion palette shown as soon as an interactive prompt sees '/'."""

        @staticmethod
        def _values(current: str, values: tuple[str, ...]):
            for value in values:
                if value.startswith(current.lower()):
                    yield Completion(value, start_position=-len(current))

        def get_completions(self, document, complete_event):
            text = document.text_before_cursor
            lowered = text.lower()
            if not lowered.startswith("/"):
                return
            if not any(character.isspace() for character in lowered):
                for command, description in SLASH_COMMANDS:
                    if command.startswith(lowered):
                        yield Completion(
                            command,
                            start_position=-len(text),
                            display_meta=description,
                        )
                return
            tokens = lowered.split()
            if not tokens:
                return
            trailing_space = bool(lowered and lowered[-1].isspace())
            current = "" if trailing_space else tokens[-1]
            if tokens[0] == "/mode" and len(tokens) <= 2:
                yield from self._values(current, ("plan", "goal", "ultra"))
            elif tokens[0] in {"/permissions", "/permission", "/access"} and len(tokens) <= 2:
                yield from self._values(current, ("normal", "full"))
            elif tokens[0] == "/settings" and len(tokens) >= 2 and tokens[1] == "mode" and len(tokens) <= 3:
                yield from self._values(current, ("plan", "goal", "ultra"))
            elif tokens[0] == "/settings" and len(tokens) >= 2 and tokens[1] == "color" and len(tokens) <= 3:
                yield from self._values(current, ("auto", "on", "off"))
            elif tokens[0] == "/settings" and (
                len(tokens) == 1 or (len(tokens) == 2 and not trailing_space)
            ):
                choices = (
                    "mode",
                    "color",
                    "provider",
                    "model",
                    "workspace",
                    "concurrency",
                    "ultra_depth",
                    "ultra_nodes",
                    "fix_attempts",
                    "reset",
                    *runtime_setting_names(),
                )
                yield from self._values(current, choices)
else:
    SlashCommandCompleter = None  # type: ignore


@dataclass
class TaskView:
    id: str
    title: str
    status: str = "pending"
    role: str = ""
    acceptance_criteria: list[str] = field(default_factory=list)
    verification: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)
    risk: str = "medium"


@dataclass
class WorkerView:
    id: str
    task_id: str
    role: str
    status: str = "running"


@dataclass
class DashboardView:
    objective: str = "No active goal. Enter a goal to begin."
    status: str = "idle"
    plan_revision: int = 0
    approved_revision: int | None = None
    plan_summary: str = ""
    plan_fingerprint: str = ""
    plan_applicability: list[dict[str, Any]] = field(default_factory=list)
    execution_strategy: str = ""
    expected_changes: list[dict[str, Any]] = field(default_factory=list)
    goal_attempt: int = 0
    retry_reason: str = ""
    tasks: list[TaskView] = field(default_factory=list)
    workers: list[WorkerView] = field(default_factory=list)
    provider: str = "-"
    model: str = "-"
    interaction_mode: str = InteractionMode.PLAN.value
    workspace: str = "."
    waiting_question: str = ""
    activity: list[str] = field(default_factory=list)


def _ansi(code: str, enabled: bool) -> str:
    return code if enabled else ""


def _isatty(stream: Any) -> bool:
    """Best-effort TTY detection for stdio wrappers such as legacy colorama."""

    current = stream
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        method = getattr(current, "isatty", None)
        if callable(method):
            try:
                return bool(method())
            except (AttributeError, OSError, ValueError):
                return False
        current = getattr(current, "wrapped", None) or getattr(current, "stream", None)
    return False


def _fit(text: Any, width: int) -> str:
    value = " ".join(str(text or "").split())
    if width <= 0:
        return ""
    if len(value) <= width:
        return value.ljust(width)
    if width <= 3:
        return value[:width]
    return (value[: width - 3] + "...").ljust(width)


def _wrap(text: Any, width: int, max_lines: int = 3) -> list[str]:
    value = " ".join(str(text or "").split())
    lines = textwrap.wrap(value, width=max(1, width), break_long_words=True) or [""]
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = _fit(lines[-1], width).rstrip()
        if width > 3:
            lines[-1] = lines[-1][: width - 3] + "..."
    return lines


def _task_mark(status: str) -> str:
    return {
        "pending": "[ ]",
        "in_progress": "[>]",
        "done": "[x]",
        "blocked": "[!]",
        "skipped": "[-]",
        "uncertain": "[?]",
    }.get(str(status).lower(), "[?]")


def render_brand() -> str:
    """Return the color-free startup identity; coloring belongs to ConsoleUI."""

    return "\n".join((*BRAND_ART, BRAND_SUBTITLE.center(BRAND_WIDTH)))


def render_slash_menu() -> str:
    width = max(len(command) for command, _description in SLASH_COMMANDS)
    lines = ["Slash commands (type a command, or use arrows/Tab in the live menu)"]
    lines.extend(
        f"  {command:<{width}}  {description}"
        for command, description in SLASH_COMMANDS
    )
    lines.extend(
        (
            "",
            "Modes",
            "  /mode plan   plan, review, and approve before manual execution",
            "  /mode goal   plan and approve first, then continue automatically",
            "  /mode ultra  deep project brain, hierarchical nodes, review/test/fix loops",
            "",
            "Legacy :commands remain supported.",
        )
    )
    return "\n".join(lines)


def render_status(
    view: DashboardView,
    *,
    access_level: str = "normal",
    execution_class: str = "local",
    active_agents: int = 0,
) -> str:
    """Render a sparse Codex-like status summary without dashboard boxes."""

    completed = sum(task.status in {"done", "skipped"} for task in view.tasks)
    lines = [
        f"GA3BAD CODING AGENT · MODE {view.interaction_mode.upper()} · STATUS {view.status.upper()} · {view.provider}/{view.model}",
        f"workspace  {view.workspace}",
        f"access     {access_level.upper()} · {execution_class.upper()} · agents {active_agents}",
        f"goal       {view.objective}",
        f"progress   {completed}/{len(view.tasks)} · status {view.status.upper()} · plan r{view.plan_revision}",
    ]
    if view.waiting_question:
        lines.append(f"input      {view.waiting_question}")
    return "\n".join(lines)


def render_tree(nodes: Iterable[Any], *, root_id: str | None = None) -> str:
    """Render work nodes from dataclasses or mappings as a compact hierarchy."""

    values = list(nodes)
    if not values:
        return "Project tree\n  (no ULTRA work nodes yet)"

    def field(item: Any, name: str, default: Any = "") -> Any:
        if isinstance(item, Mapping):
            return item.get(name, default)
        value = getattr(item, name, default)
        return getattr(value, "value", value)

    by_parent: dict[str | None, list[Any]] = {}
    for item in values:
        by_parent.setdefault(field(item, "parent_id", None), []).append(item)
    for children in by_parent.values():
        children.sort(
            key=lambda item: (
                int(field(item, "position", field(item, "priority", 0))),
                str(field(item, "id")),
            )
        )
    marks = {
        "completed": "[x]",
        "done": "[x]",
        "running": "[>]",
        "in_progress": "[>]",
        "failed": "[!]",
        "blocked": "[!]",
        "conflict": "[!]",
        "uncertain": "[?]",
        "cancelled": "[-]",
    }
    lines = ["Project tree"]

    def visit(parent: str | None, prefix: str = "") -> None:
        children = by_parent.get(parent, [])
        for index, item in enumerate(children):
            last = index == len(children) - 1
            node_id = str(field(item, "id"))
            status = str(field(item, "status", "pending")).lower()
            kind = str(field(item, "kind", field(item, "node_type", "task")))
            title = str(field(item, "title", node_id))
            branch = "`-" if last else "|-"
            lines.append(f"{prefix}{branch} {marks.get(status, '[ ]')} {node_id} · {kind} · {title}")
            visit(node_id, prefix + ("   " if last else "|  "))

    visit(root_id)
    return "\n".join(lines)


def render_agents(
    runs: Iterable[Any],
    *,
    include_finished: bool = False,
    node_titles: Mapping[str, str] | None = None,
) -> str:
    values = list(runs)
    if not include_finished:
        def status_of(item: Any) -> str:
            raw = item.get("status", "") if isinstance(item, Mapping) else getattr(item, "status", "")
            return str(getattr(raw, "value", raw)).lower()

        values = [
            item
            for item in values
            if status_of(item)
            in {"queued", "pending", "running", "rate_limited", "paused", "uncertain"}
        ]
    if not values:
        return "Agents\n  (none active)"
    lines = [f"Agents · {len(values)}"]
    for item in values:
        get = item.get if isinstance(item, Mapping) else lambda key, default="": getattr(item, key, default)
        status = getattr(get("status", ""), "value", get("status", ""))
        role = get("role", get("phase", "worker"))
        node = get("work_node_id", get("node_id", get("task_id", "-")))
        phase = get("phase", "-")
        model = get("model", "-")
        node_label = (node_titles or {}).get(str(node), str(node))
        lines.append(f"  [{role}] {status} · {phase} · {node_label} · {model}")
    return "\n".join(lines)


def render_memory(entries: Iterable[Any], *, title: str = "Project Brain") -> str:
    values = list(entries)
    if not values:
        return f"{title}\n  (empty)"
    lines = [title]
    for item in values:
        get = item.get if isinstance(item, Mapping) else lambda key, default="": getattr(item, key, default)
        category_value = get("section", get("category", "memory"))
        category = getattr(category_value, "value", category_value)
        key = get("title", get("key", get("id", "-")))
        content = get("content", get("summary", ""))
        lines.append(f"  [{category}] {key}")
        if content and str(content).strip() != str(key).strip():
            one_line = " ".join(str(content).split())
            lines.append(f"      {_fit(one_line, 116).rstrip()}")
    return "\n".join(lines)


def render_trace(trace: Any | None) -> str:
    if trace is None:
        return "Trace\n  (no prompt trace recorded)"
    get = trace.get if isinstance(trace, Mapping) else lambda key, default="": getattr(trace, key, default)
    lines = [
        f"Trace {get('id', '-')}",
        f"  role      {get('role', '-')}",
        f"  node      {get('work_node_id', get('node_id', '-'))}",
        f"  redacted  {get('redacted', True)}",
        f"  truncated {get('truncated', False)}",
    ]
    system_prompt = get("system_prompt", get("prompt", get("request_text", "")))
    context_package = get("context_package", {})
    self_prompt = get("self_prompt", "")
    summary = get("reasoning_summary", get("summary", get("response_summary", "")))
    if system_prompt:
        lines.extend(("", "System prompt", str(system_prompt)))
    if context_package:
        lines.extend(
            (
                "",
                "Context package",
                json.dumps(context_package, ensure_ascii=False, indent=2, default=str),
            )
        )
    if self_prompt:
        lines.extend(("", "Self prompt", str(self_prompt)))
    if summary:
        lines.extend(("", "Reasoning summary (no hidden chain-of-thought)", str(summary)))
    return "\n".join(lines)


def render_dashboard(view: DashboardView, width: int | None = None) -> str:
    """Render a responsive, strictly ASCII dashboard suitable for snapshots."""
    terminal_width = shutil.get_terminal_size((112, 30)).columns
    width = max(4, min(width or terminal_width, 140))
    inner = width - 2
    title = " GA3BAD CODING AGENT "[:inner]
    top = "+" + title + "-" * max(0, inner - len(title)) + "+"
    rule = "+" + "-" * inner + "+"

    approved = f"r{view.approved_revision}" if view.approved_revision is not None else "pending"
    completed = sum(task.status in {"done", "skipped"} for task in view.tasks)
    total = len(view.tasks)
    bar_width = min(24, max(8, width // 6))
    filled = round(bar_width * completed / total) if total else 0
    progress = "[" + "#" * filled + "-" * (bar_width - filled) + f"] {completed}/{total}"
    meta = (
        f" MODE {str(view.interaction_mode).upper()} | STATUS {str(view.status).upper()} "
        f"| PLAN r{view.plan_revision} / {approved} "
        f"| {progress} | {view.provider}/{view.model} "
    )

    lines = [top, "|" + _fit(meta, inner) + "|"]
    if view.goal_attempt:
        attempt = f" ATTEMPT {view.goal_attempt} | unbounded goal retries active"
        if view.retry_reason:
            attempt += f" | {view.retry_reason}"
        lines.append("|" + _fit(attempt, inner) + "|")
    lines.append("|" + _fit(f" GOAL  {view.objective}", inner) + "|")
    lines.append("|" + _fit(f" ROOT  {view.workspace}", inner) + "|")
    if view.waiting_question:
        lines.append("|" + _fit(f" INPUT NEEDED  {view.waiting_question}", inner) + "|")
    lines.append(rule)

    if width >= 96:
        left = int(inner * 0.62)
        right = inner - left - 1
        lines.append("|" + _fit(" CHECKLIST", left) + "|" + _fit(" ACTIVE WORKERS / DYNAMIC ROLES", right) + "|")
        rows = max(1, min(max(len(view.tasks), len(view.workers)), 14))
        for index in range(rows):
            if index < len(view.tasks):
                task = view.tasks[index]
                task_text = f" {_task_mark(task.status)} {task.id} {task.title}"
            else:
                task_text = ""
            if index < len(view.workers):
                worker = view.workers[index]
                worker_text = f" {worker.id} {worker.status.upper()} {worker.task_id} - {worker.role}"
            else:
                worker_text = ""
            lines.append("|" + _fit(task_text, left) + "|" + _fit(worker_text, right) + "|")
        if len(view.tasks) > rows or len(view.workers) > rows:
            task_more = f" ... +{len(view.tasks) - rows} more; use /plan" if len(view.tasks) > rows else ""
            worker_more = f" ... +{len(view.workers) - rows} more" if len(view.workers) > rows else ""
            lines.append("|" + _fit(task_more, left) + "|" + _fit(worker_more, right) + "|")
    else:
        lines.append("|" + _fit(" CHECKLIST", inner) + "|")
        for task in view.tasks[:12]:
            lines.append("|" + _fit(f" {_task_mark(task.status)} {task.id} {task.title}", inner) + "|")
        if len(view.tasks) > 12:
            lines.append("|" + _fit(f" ... +{len(view.tasks) - 12} more; use /plan", inner) + "|")
        if view.workers:
            lines.append("|" + _fit(" WORKERS / DYNAMIC ROLES", inner) + "|")
            for worker in view.workers[:5]:
                lines.append("|" + _fit(f" {worker.id} {worker.status.upper()} {worker.task_id} - {worker.role}", inner) + "|")

    lines.append(rule)
    lines.append("|" + _fit(" RECENT ACTIVITY", inner) + "|")
    for item in (view.activity[-4:] or ["Ready."]):
        lines.append("|" + _fit(f" {item}", inner) + "|")
    lines.append(rule)
    commands = " /  /mode  /approve  /run [steps]  /plan  /settings  /status  /help  /quit"
    lines.append("|" + _fit(commands, inner) + "|")
    lines.append("+" + "-" * inner + "+")
    return "\n".join(lines)


def render_plan(view: DashboardView, width: int | None = None) -> str:
    """Render the complete checklist (the dashboard intentionally shows a viewport)."""
    terminal_width = shutil.get_terminal_size((112, 30)).columns
    width = max(4, min(width or terminal_width, 140))
    inner = width - 2
    title = " COMPLETE CHECKLIST "[:inner]
    lines = [
        "+" + title + "-" * max(0, inner - len(title)) + "+",
        "|" + _fit(
            f" STATUS {view.status.upper()} | PLAN r{view.plan_revision} / "
            f"{('r' + str(view.approved_revision)) if view.approved_revision is not None else 'pending'} "
            f"| fingerprint={view.plan_fingerprint[:12] or '-'}",
            inner,
        ) + "|",
        "|" + _fit(f" GOAL  {view.objective}", inner) + "|",
    ]
    if view.plan_summary:
        for index, summary_line in enumerate(_wrap(view.plan_summary, max(10, inner - 10), max_lines=20)):
            label = " PLAN  " if index == 0 else "       "
            lines.append("|" + _fit(label + summary_line, inner) + "|")
    if view.plan_applicability:
        lines.append("|" + _fit(" APPLICABILITY EVIDENCE", inner) + "|")
        for item in view.plan_applicability:
            supports = ",".join(str(value) for value in item.get("supports_tasks", ()))
            value = f"[{supports or '-'}] {item.get('fact', '')} (source: {item.get('source', '-')})"
            for evidence_line in _wrap(value, max(10, inner - 4), max_lines=20):
                lines.append("|" + _fit("   " + evidence_line, inner) + "|")
    if view.execution_strategy:
        lines.append("|" + _fit(" EXECUTION STRATEGY", inner) + "|")
        for strategy_line in _wrap(view.execution_strategy, max(10, inner - 4), max_lines=40):
            lines.append("|" + _fit("   " + strategy_line, inner) + "|")
    if view.expected_changes:
        lines.append("|" + _fit(" EXPECTED WORKSPACE CHANGES", inner) + "|")
        for item in view.expected_changes:
            supports = ",".join(str(value) for value in item.get("supports_tasks", ()))
            value = f"[{supports or '-'}] {item.get('path', '-')}: {item.get('intent', '')}"
            for change_line in _wrap(value, max(10, inner - 4), max_lines=20):
                lines.append("|" + _fit("   " + change_line, inner) + "|")
    lines.append("+" + "-" * inner + "+")
    if not view.tasks:
        lines.append("|" + _fit(" (no checklist items)", inner) + "|")
    for task in view.tasks:
        prefix = f" {_task_mark(task.status)} {task.id} "
        wrapped = _wrap(task.title, max(10, inner - len(prefix)), max_lines=20)
        lines.append("|" + _fit(prefix + wrapped[0], inner) + "|")
        for continuation in wrapped[1:]:
            lines.append("|" + _fit(" " * len(prefix) + continuation, inner) + "|")
        meta = f"     risk={task.risk} depends_on={','.join(task.depends_on) if task.depends_on else '-'}"
        lines.append("|" + _fit(meta, inner) + "|")
        for criterion in task.acceptance_criteria:
            for index, criterion_line in enumerate(_wrap(criterion, max(10, inner - 12), max_lines=20)):
                label = "       accept: " if index == 0 else "               "
                lines.append("|" + _fit(label + criterion_line, inner) + "|")
        for check in task.verification:
            for index, check_line in enumerate(_wrap(check, max(10, inner - 12), max_lines=20)):
                label = "       verify: " if index == 0 else "               "
                lines.append("|" + _fit(label + check_line, inner) + "|")
    lines.append("+" + "-" * inner + "+")
    return "\n".join(lines)


HELP_TEXT = """\
Slash palette and modes
  /                           open the slash-command palette
  /mode                      show the current interaction mode
  /mode plan                 plan/approve, then wait for manual /run
  /mode goal                 plan/approve, then continue automatically
  /mode ultra                Project Brain + hierarchical agents + full quality gates
  /settings                  show safe session settings (never secrets)
  /settings NAME VALUE       change mode, color, or a runtime limit this session
  /model                     reopen the tool-capable model picker at a safe checkpoint
  /permissions normal|full   choose approvals or Docker-isolated Full access
  /setup                     build/validate the one-time Full-access sandbox

Goal and plan
  /goal TEXT                 start a durable goal (plain text also works when idle)
  /approve [REV]             approve exactly the displayed plan revision
  /reject FEEDBACK           reject and regenerate the draft
  /replan FEEDBACK           ask for a revised plan
  /plan                      show the dashboard/checklist
  /questions                 show non-discoverable decisions awaiting input
  /answer ID VALUE           save a durable answer bound into the plan fingerprint

Editable checklist
  /add TEXT :: CRITERIA      add a user-approved task
  /edit ID [FIELD] VALUE     edit whole task or title/description/accept/verify/depends/risk
  /remove TASK_ID            remove a task
  /done TASK_ID NOTE         mark done with user evidence
  /todo|/block|/skip ID NOTE change task status

Execution
  /run [STEPS]               run a bounded work slice; goal remains durable
  /auto                      retry/self-prompt without limit until completion or real user input
  /pause / /resume           cooperatively stop or continue
  /history / /status         inspect durable execution state
  /tree [NODE]               inspect the ULTRA module/submodule/task hierarchy
  /agents [--all]            inspect active or recent role-isolated agents
  /memory [SECTION]          inspect/search Project Brain entries
  /trace [latest|RUN_ID]     show redacted prompts, context, and reasoning summaries
  /insights [NODE]           show findings, decisions, and lessons
  /metrics                   show token, agent, node, fix, and concurrency metrics
  /resolve ENTITY_ID applied|not-run NOTE
                              reconcile a crash-window action/worker after inspection
  /cancel CANCEL             explicitly abandon an unfinished goal
  /quit                      exit safely; unfinished state resumes next launch

All legacy :commands remain supported.
"""


class ConsoleUI:
    """Event renderer with serialized output for model streaming and workers."""

    def __init__(
        self,
        stream: TextIO | None = None,
        color: bool | None = None,
        input_func: Any = input,
        interaction_mode: str | InteractionMode = InteractionMode.PLAN,
    ) -> None:
        self.stream = stream or sys.stdout
        self.input_func = input_func
        self._lock = RLock()
        self._stream_kind: str | None = None
        self._prompt_session: Any = None
        self.activity: list[str] = []
        self.interaction_mode = InteractionMode.parse(interaction_mode)
        self.access_level = "normal"
        self.execution_class = "local"
        self.active_agents = 0
        self.color_mode = "auto" if color is None else ("on" if color else "off")
        self.color = False
        self.set_color(self.color_mode)

    def set_color(self, mode: str) -> None:
        normalized = str(mode).strip().lower()
        aliases = {"true": "on", "yes": "on", "1": "on", "false": "off", "no": "off", "0": "off"}
        normalized = aliases.get(normalized, normalized)
        if normalized not in {"auto", "on", "off"}:
            raise ValueError("color must be auto, on, or off")
        self.color_mode = normalized
        self.color = _isatty(self.stream) if normalized == "auto" else normalized == "on"
        self.bold = _ansi("\033[1m", self.color)
        self.dim = _ansi("\033[2m", self.color)
        self.cyan = _ansi("\033[36m", self.color)
        self.green = _ansi("\033[32m", self.color)
        self.yellow = _ansi("\033[33m", self.color)
        self.gold = _ansi("\033[38;5;220m", self.color)
        self.red = _ansi("\033[31m", self.color)
        self.magenta = _ansi("\033[35m", self.color)
        self.reset = _ansi("\033[0m", self.color)

    def set_mode(self, mode: str | InteractionMode) -> None:
        self.interaction_mode = InteractionMode.parse(mode)

    def set_runtime_identity(
        self,
        *,
        access_level: str = "normal",
        execution_class: str = "local",
        active_agents: int = 0,
    ) -> None:
        self.access_level = str(access_level).lower()
        self.execution_class = str(execution_class).lower()
        self.active_agents = max(0, int(active_agents))

    def write(self, text: str = "", *, end: str = "\n", flush: bool = True) -> None:
        with self._lock:
            print(text, end=end, file=self.stream, flush=flush)

    def show_dashboard(self, view: DashboardView) -> None:
        view.activity = view.activity or self.activity[-4:]
        view.interaction_mode = self.interaction_mode.value
        self.write(render_dashboard(view))

    def show_status(self, view: DashboardView) -> None:
        view.activity = view.activity or self.activity[-4:]
        view.interaction_mode = self.interaction_mode.value
        self.write(
            render_status(
                view,
                access_level=self.access_level,
                execution_class=self.execution_class,
                active_agents=self.active_agents,
            )
        )

    def show_brand(self) -> None:
        for line in BRAND_ART:
            self.write(f"{self.bold}{self.green}{line}{self.reset}")
        self.write(f"{self.dim}{self.green}{BRAND_SUBTITLE.center(BRAND_WIDTH)}{self.reset}")
        self.write()

    def on_event(self, event: UIEvent) -> None:
        kind, message, data = event.kind, event.message, event.data
        with self._lock:
            if kind in {"model_text", "model_thought"}:
                label = "Response" if kind == "model_text" else "Thinking…"
                color = self.gold if self.interaction_mode is InteractionMode.ULTRA else self.cyan
                if self._stream_kind != kind:
                    if self._stream_kind:
                        print(self.reset, file=self.stream)
                    print(f"{self.bold}{color}{label}{self.reset}", file=self.stream)
                    self._stream_kind = kind
                # Provider thought streams may contain private scratch work.
                # The UI shows activity, while durable traces keep only the
                # model-authored reasoning summary and explicit insights.
                if kind == "model_text":
                    print(message, end="", file=self.stream, flush=True)
                return
            if self._stream_kind:
                print(self.reset, file=self.stream)
                self._stream_kind = None

            accent = self.gold if self.interaction_mode is InteractionMode.ULTRA else self.cyan
            if kind == "step":
                actor = data.get("actor", "coordinator")
                print(
                    f"{self.bold}{accent}◇ {actor}{self.reset} "
                    f"{self.dim}step {data.get('step', '?')}{self.reset}",
                    file=self.stream,
                )
            elif kind == "tool_call":
                args = data.get("args", {})
                actor = data.get("actor", "agent")
                detail = args.get("path") or args.get("command") or ""
                detail = _fit(detail, 90).rstrip()
                suffix = f" {self.dim}{detail}{self.reset}" if detail else ""
                print(f"{self.bold}{accent}[{actor}]{self.reset} {message}{suffix}", file=self.stream)
            elif kind == "tool_result":
                one_line = " ".join(message.split())
                failed = one_line.startswith(("Error:", "Permission denied"))
                color = self.red if failed else self.green
                mark = "failed" if failed else "updated"
                print(f"{self.bold}{color}{mark}{self.reset} {_fit(one_line, 120).rstrip()}", file=self.stream)
            elif kind == "usage":
                print(
                    f"{self.dim}tokens in={data.get('input_tokens', 0)} "
                    f"cached={data.get('cached_tokens', 0)} out={data.get('output_tokens', 0)}{self.reset}",
                    file=self.stream,
                )
            elif kind in {"error", "warning"}:
                color = self.red if kind == "error" else self.yellow
                print(f"{self.bold}{color}{kind}{self.reset} {message}", file=self.stream)
            elif kind == "ultra.agent_started":
                self.active_agents += 1
                role = data.get("role", "agent")
                node = data.get("node_id")
                node_text = f"{node} · " if node else ""
                print(f"{self.bold}{self.gold}[{node_text}{role}]{self.reset} {data.get('phase', message)}", file=self.stream)
            elif kind == "ultra.agent":
                self.active_agents = max(0, self.active_agents - 1)
                role = data.get("role", "agent")
                print(f"{self.bold}{self.gold}[{role}]{self.reset} {message}", file=self.stream)
            elif kind.startswith("ultra."):
                color = self.green if kind in {"ultra.completed"} else self.gold
                print(f"{self.bold}{color}{message or kind}{self.reset}", file=self.stream)
            elif kind in {"phase", "checkpoint", "delegation", "plan", "recovery", "questions"}:
                print(f"{self.bold}{accent}{message or kind}{self.reset}", file=self.stream)
            elif message:
                print(message, file=self.stream)
            self.stream.flush()
        if message and kind not in {"model_text", "model_thought", "tool_result"}:
            self.activity.append(f"{kind}: {' '.join(message.split())[:180]}")
            del self.activity[:-50]

    def confirm_action(self, name: str, args: dict, risk: str = "risky") -> bool:
        canonical = json.dumps(args, ensure_ascii=False, sort_keys=True, default=str)
        digest = hashlib.sha256(canonical.encode("utf-8", "replace")).hexdigest()[:12]
        preview: dict[str, Any] = {}
        for key, value in args.items():
            if isinstance(value, str) and len(value) > 1_600:
                preview[key] = (
                    value[:1_100]
                    + f"\n... [{len(value) - 1_500} chars omitted; sha256="
                    + hashlib.sha256(value.encode("utf-8", "replace")).hexdigest()[:12]
                    + "] ...\n"
                    + value[-400:]
                )
            else:
                preview[key] = value
        self.write(f"{self.yellow}APPROVAL [{risk}] {name} action={digest}{self.reset}")
        pretty = json.dumps(preview, ensure_ascii=False, indent=2, default=str)
        for line in pretty.splitlines():
            self.write(f"  {line}")
        try:
            answer = self.input_func(
                f"{self.bold}Allow this action once? [y/N]{self.reset} "
            ).strip().lower()
        except EOFError:
            self.write("Approval denied: no interactive confirmation was received.")
            return False
        except KeyboardInterrupt:
            self.write("\nApproval interrupted; checkpointing active work.")
            raise
        return answer in {"y", "yes"}

    def prompt(self) -> str:
        accent = self.gold if self.interaction_mode is InteractionMode.ULTRA else self.green
        label = (
            f"{self.bold}{accent}GA3BAD{self.reset} "
            f"{self.dim}[{self.interaction_mode.value.upper()}]{self.reset}> "
        )
        rich_prompt = (
            PromptSession is not None
            and SlashCommandCompleter is not None
            and self.input_func is input
            and self.stream is sys.stdout
            and _isatty(sys.stdin)
            and _isatty(self.stream)
        )
        if rich_prompt:
            if self._prompt_session is None:
                self._prompt_session = PromptSession(
                    completer=SlashCommandCompleter(),
                    complete_while_typing=True,
                )
            if patch_stdout is not None:
                with patch_stdout(raw=True):
                    return self._prompt_session.prompt(
                        ANSI(label),
                        complete_style=CompleteStyle.MULTI_COLUMN,
                        reserve_space_for_menu=10,
                    )
            return self._prompt_session.prompt(ANSI(label))
        return self.input_func(label)


# ---------------------------------------------------------------------------
# Compatibility wrappers for the original educational loop.
_default_console = ConsoleUI()


def banner(provider: str, model: str | None = None) -> None:
    _default_console.show_brand()
    _default_console.write(f"GA3BAD coding agent - provider={provider} model={model or '-'}")


def user_prompt() -> str:
    return (
        f"{_default_console.bold}{_default_console.green}GA3BAD{_default_console.reset} "
        f"{_default_console.dim}[PLAN]{_default_console.reset}> "
    )


def step_header(n: int) -> None:
    _default_console.on_event(UIEvent("step", data={"step": n, "actor": "coordinator"}))


class Streamer:
    def on_text(self, fragment: str) -> None:
        _default_console.on_event(UIEvent("model_text", fragment))

    def on_thought(self, fragment: str) -> None:
        _default_console.on_event(UIEvent("model_thought", fragment))

    def close(self) -> None:
        if _default_console._stream_kind:
            _default_console.write(_default_console.reset)
            _default_console._stream_kind = None


def tool_call(name: str, args: dict) -> None:
    _default_console.on_event(UIEvent("tool_call", name, {"args": args}))


def tool_result(result: Any, limit: int = 100) -> None:
    _default_console.on_event(UIEvent("tool_result", str(result)[: max(limit, 1_000)]))


def usage(value: Any) -> None:
    _default_console.on_event(UIEvent("usage", data={
        "input_tokens": getattr(value, "input_tokens", 0),
        "cached_tokens": getattr(value, "cached_tokens", 0),
        "output_tokens": getattr(value, "output_tokens", 0),
    }))


def confirm_prompt(name: str) -> str:
    return input(f"run {name}? [y/N] ").strip().lower()


def turn_end() -> None:
    _default_console.write()
