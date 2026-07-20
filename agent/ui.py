"""ASCII terminal UI for persistent goals, plans, workers, and live events."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
import textwrap
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Event, RLock, Thread
from typing import Any, Iterable, Mapping, TextIO

try:
    from .config import InteractionMode, runtime_setting_names
    from .events import UIEvent
    from .safety import redact_text
    from .tui import inline_square_levels, terminal_supports_unicode
except ImportError:  # direct ``python agent/main.py`` compatibility
    from config import InteractionMode, runtime_setting_names  # type: ignore
    from events import UIEvent  # type: ignore
    from safety import redact_text  # type: ignore
    from tui import inline_square_levels, terminal_supports_unicode  # type: ignore

try:  # Optional at import time; basic input and the bare-/ menu remain available.
    from prompt_toolkit import ANSI, PromptSession
    from prompt_toolkit.completion import Completer, Completion
    from prompt_toolkit.keys import Keys
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.patch_stdout import patch_stdout
    from prompt_toolkit.shortcuts import CompleteStyle
    from prompt_toolkit.styles import Style
except ImportError:  # pragma: no cover - exercised by minimal installations
    ANSI = PromptSession = Completer = Completion = CompleteStyle = KeyBindings = Keys = patch_stdout = Style = None  # type: ignore


BRAND_ART = (
    "  ____    _    _____ ____    _    ____ ",
    " / ___|  / \\  |___ /| __ )  / \\  |  _ \\",
    "| |  _  / _ \\   |_ \\|  _ \\ / _ \\ | | | |",
    "| |_| |/ ___ \\ ___) | |_) / ___ \\| |_| |",
    " \\____/_/   \\_\\____/|____/_/   \\_\\____/ ",
)
BRAND_WIDTH = max(len(line) for line in BRAND_ART)
BRAND_SUBTITLE = "coding agent"
LONG_PROMPT_RECEIPT_CHARS = 2_000

CODEX_SLASH_COMMANDS: tuple[tuple[str, str], ...] = (
    ("/model", "choose what model and reasoning effort to use"),
    ("/ide", "include current selection, open files, and other context from your IDE"),
    ("/permissions", "choose what Codex is allowed to do"),
    ("/keymap", "remap TUI shortcuts"),
    ("/vim", "toggle Vim mode for the composer"),
    ("/sandbox-add-read-dir", "let sandbox read a directory: /sandbox-add-read-dir <absolute_path>"),
    ("/experimental", "toggle experimental features"),
    ("/approve", "approve one retry of a recent auto-review denial"),
)

SLASH_COMMANDS: tuple[tuple[str, str], ...] = (
    ("/mode", "switch NORMAL / ULTRA mode"),
    ("/model", "choose a model and reasoning effort"),
    ("/permissions", "switch NORMAL / FULL access"),
    ("/tree", "show the hierarchical project tree"),
    ("/agents", "open the read-only live specialist observer"),
    ("/agent", "view one specialist's assignment and redacted prompt"),
    ("/memory", "inspect the Project Brain"),
    ("/trace", "inspect redacted prompts and run trace"),
    ("/thinking", "expand redacted thoughts captured in this session"),
    ("/insights", "show durable findings and decisions"),
    ("/questions", "show pending intake or plan questions"),
    ("/answer", "answer with 1/2/3 or free-form text"),
    ("/metrics", "show quality, usage, and timing metrics"),
    ("/doctor", "audit weak-model agent readiness; add --live/--record for probes/history"),
    ("/settings", "inspect or change session settings"),
    ("/skills", "show real local tools, availability, risk, and approval policy"),
    ("/processes", "list active managed processes and HTML previews"),
    ("/stop-process", "stop a managed process or preview by ID"),
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
    ("/sleep", "control the Ultra Sleep profile"),
    ("/help", "show every command"),
    ("/quit", "save state and exit"),
)

ALL_SLASH_COMMANDS: tuple[tuple[str, str], ...] = tuple(
    dict((*CODEX_SLASH_COMMANDS, *SLASH_COMMANDS)).items()
)

# The parser remains the execution authority; this registry only controls how
# commands are progressively disclosed in the interactive palette.
COMMAND_GROUPS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "Workflow",
        "Start, approve, run, pause, or resume the current goal",
        (
            "/goal", "/plan", "/approve", "/reject", "/replan", "/run",
            "/auto", "/pause", "/resume", "/resolve", "/cancel",
        ),
    ),
    (
        "Tasks",
        "Edit the accepted checklist and record task outcomes",
        ("/add", "/edit", "/remove", "/done", "/todo", "/block", "/skip"),
    ),
    (
        "Inspect",
        "Open status, activity, agents, memory, traces, and metrics",
        (
            "/status", "/history", "/tree", "/agents", "/agent", "/memory", "/trace", "/thinking",
            "/insights", "/questions", "/answer", "/metrics", "/doctor", "/skills", "/processes",
        ),
    ),
    (
        "Session",
        "Change mode, model, access, and terminal settings",
        ("/mode", "/model", "/permissions", "/settings", "/setup", "/sleep", "/stop-process"),
    ),
    (
        "Help & exit",
        "Show guidance or checkpoint and leave the session",
        ("/help", "/quit"),
    ),
)

_COMMAND_DESCRIPTIONS = dict(SLASH_COMMANDS)
_CONTEXT_COMMANDS: dict[str, tuple[str, ...]] = {
    "idle": ("/goal", "/mode", "/model", "/settings"),
    "new": ("/goal", "/mode", "/model", "/settings"),
    "discovering": ("/status", "/pause", "/model", "/thinking"),
    "revising": ("/status", "/pause", "/plan", "/thinking"),
    "awaiting_plan_approval": ("/approve", "/reject", "/plan", "/questions"),
    "running": ("/status", "/agents", "/pause", "/thinking", "/tree"),
    "paused": ("/resume", "/status", "/resolve", "/trace"),
    "recovering": ("/resolve", "/status", "/trace", "/help"),
    "reviewing": ("/status", "/agents", "/trace", "/pause"),
    "verifying": ("/status", "/agents", "/trace", "/pause"),
}


def contextual_commands(status: str) -> tuple[tuple[str, str], ...]:
    """Return the small command set worth showing at this checkpoint."""

    names = _CONTEXT_COMMANDS.get(str(status).strip().lower(), ("/status", "/help", "/quit"))
    return tuple((name, _COMMAND_DESCRIPTIONS[name]) for name in names)


def prompt_receipt(value: str, threshold: int = LONG_PROMPT_RECEIPT_CHARS) -> str:
    """Return a compact visual receipt without changing the submitted prompt."""

    text = str(value)
    if len(text) < threshold:
        return text
    return f"[Pasted Content {len(text)} chars]"


def _memory_snapshot() -> tuple[float, float]:
    """Return process working-set MiB and total system RAM load."""
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [("length", wintypes.DWORD), ("load", wintypes.DWORD),
                    ("total_phys", ctypes.c_ulonglong), ("avail_phys", ctypes.c_ulonglong),
                    ("total_page", ctypes.c_ulonglong), ("avail_page", ctypes.c_ulonglong),
                    ("total_virtual", ctypes.c_ulonglong), ("avail_virtual", ctypes.c_ulonglong),
                    ("avail_extended", ctypes.c_ulonglong)]

            class ProcessMemory(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD),
                    ("peak_working", ctypes.c_size_t), ("working", ctypes.c_size_t),
                    ("peak_paged", ctypes.c_size_t), ("paged", ctypes.c_size_t),
                    ("peak_nonpaged", ctypes.c_size_t), ("nonpaged", ctypes.c_size_t),
                    ("pagefile", ctypes.c_size_t), ("peak_pagefile", ctypes.c_size_t)]

            system, process = MemoryStatus(), ProcessMemory()
            system.length, process.cb = ctypes.sizeof(system), ctypes.sizeof(process)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(system))
            get_process_memory = getattr(
                ctypes.windll.kernel32,
                "K32GetProcessMemoryInfo",
                ctypes.windll.psapi.GetProcessMemoryInfo,
            )
            ctypes.windll.kernel32.GetCurrentProcess.restype = ctypes.c_void_p
            get_process_memory.argtypes = [ctypes.c_void_p, ctypes.POINTER(ProcessMemory), wintypes.DWORD]
            get_process_memory.restype = wintypes.BOOL
            if not get_process_memory(
                ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(process), process.cb
            ):
                raise OSError("GetProcessMemoryInfo failed")
            return process.working / (1024 * 1024), float(system.load)
        except (AttributeError, OSError, ValueError):
            pass
    return 0.0, 0.0


if Completer is not None:
    class SlashCommandCompleter(Completer):  # type: ignore[misc,valid-type]
        """Completion palette shown as soon as an interactive prompt sees '/'."""

        def __init__(self, status_provider: Any = None) -> None:
            self.status_provider = status_provider

        @staticmethod
        def _values(current: str, values: tuple[str | tuple[str, str], ...]):
            for item in values:
                value, description = item if isinstance(item, tuple) else (item, "")
                if value.startswith(current.lower()):
                    yield Completion(
                        value,
                        start_position=-len(current),
                        display_meta=description,
                    )

        def get_completions(self, document, complete_event):
            text = document.text_before_cursor
            lowered = text.lower()
            if not lowered.startswith("/"):
                return
            if not any(character.isspace() for character in lowered):
                commands = ALL_SLASH_COMMANDS
                for command, description in commands:
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
                yield from self._values(
                    current,
                    (
                        ("normal", "shared intake, durable goal, planning, review, and automatic execution"),
                        ("ultra", "recursive specialists with component and quality gates"),
                    ),
                )
            elif tokens[0] in {"/permissions", "/permission", "/access"} and len(tokens) <= 2:
                yield from self._values(
                    current,
                    (
                        ("normal", "ask before risky actions"),
                        ("full", "Docker-isolated access without workspace prompts"),
                    ),
                )
            elif tokens[0] == "/settings" and len(tokens) >= 2 and tokens[1] == "mode" and len(tokens) <= 3:
                yield from self._values(current, ("normal", "ultra"))
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
    # Standalone view construction retains the legacy PLAN rendering default;
    # live sessions explicitly inject the chat-first preference.
    interaction_mode: str = InteractionMode.NORMAL.value
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
    """Render the plain-terminal fallback of the Codex-like slash palette."""

    width = max(len(command) for command, _description in CODEX_SLASH_COMMANDS)
    lines = ["Commands"]
    lines.extend(
        f"{command:<{width}}  {description}"
        for command, description in CODEX_SLASH_COMMANDS
    )
    lines.extend(
        (
            "",
            "More GA3BAD commands remain available directly, including:",
            "  /mode normal shared intake, durable goal, planning, review, and automatic execution",
            "  /mode ultra  recursive specialists, component packages, and deeper quality gates",
            "  /settings    inspect or change session settings",
            "  /trace       inspect redacted prompts and run trace",
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
    width = max(32, min(shutil.get_terminal_size((112, 30)).columns, 140))

    def row(label: str, value: Any) -> str:
        prefix = f"{label:<10}"
        return prefix + _fit(value, max(1, width - len(prefix))).rstrip()

    objective = str(view.objective or "").strip()
    if objective.casefold().startswith("prompt:"):
        objective = objective.partition(":")[2].lstrip()

    lines = [
        _fit(
            f"GA3BAD CODING AGENT · MODE {view.interaction_mode.upper()} · STATUS {view.status.upper()} · {view.provider}/{view.model}",
            width,
        ).rstrip(),
        row("workspace", view.workspace),
        row("access", f"{access_level.upper()} · {execution_class.upper()} · agents {active_agents}"),
        row("goal", objective),
        row("progress", f"{completed}/{len(view.tasks)} · plan r{view.plan_revision}"),
    ]
    if view.waiting_question:
        lines.append(row("input", view.waiting_question))
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


def _render_agents_legacy(
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


def render_agents(
    runs: Iterable[Any],
    *,
    include_finished: bool = False,
    node_titles: Mapping[str, str] | None = None,
    nodes: Iterable[Any] = (),
    run_id: str | None = None,
) -> str:
    """Render a lightweight, stable, read-only view of the live swarm."""

    all_runs = list(runs)
    node_values = list(nodes)

    def field(item: Any, name: str, default: Any = "") -> Any:
        if isinstance(item, Mapping):
            value = item.get(name, default)
        else:
            value = getattr(item, name, default)
        return getattr(value, "value", value)

    def status_of(item: Any) -> str:
        return str(field(item, "status", "")).lower()

    if not node_values:
        return _render_agents_legacy(
            all_runs,
            include_finished=include_finished,
            node_titles=node_titles,
        )

    node_index = {
        str(field(item, "id")): index for index, item in enumerate(node_values, 1)
    }
    counts: dict[str, int] = {}
    for item in node_values:
        state = status_of(item) or "pending"
        counts[state] = counts.get(state, 0) + 1
    count_text = " · ".join(
        f"{key} {counts[key]}"
        for key in (
            "running",
            "in_progress",
            "revision_required",
            "failed",
            "blocked",
            "completed",
            "pending",
        )
        if counts.get(key)
    )
    count_text = " | ".join(count_text.split(" · "))
    lines = [
        "Swarm observer | READ ONLY",
        f"  run {run_id or '-'} | specialists {len(node_values)}"
        + (f" | {count_text}" if count_text else ""),
    ]

    active_runs = (
        list(all_runs)
        if include_finished
        else [
            item
            for item in all_runs
            if status_of(item)
            in {"queued", "pending", "running", "rate_limited", "paused", "uncertain"}
        ]
    )
    lines.append("Working now" if active_runs else "Working now\n  (none active)")
    for item in active_runs:
        status = field(item, "status", "")
        role = field(item, "role", field(item, "phase", "worker"))
        node = field(
            item,
            "work_node_id",
            field(item, "node_id", field(item, "task_id", "-")),
        )
        phase = field(item, "phase", "-")
        model = field(item, "model", "-")
        node_label = (node_titles or {}).get(str(node), str(node))
        index_label = (
            f"{node_index[str(node)]:02d}"
            if str(node) in node_index
            else str(field(item, "id", "-"))[:8]
        )
        lines.append(
            f"  [{index_label}] {node_label} | {role}/{phase} | {status} | {model}"
        )

    if include_finished:
        latest_by_node: dict[str, Any] = {}
        for item in all_runs:
            node_id = str(field(item, "work_node_id", ""))
            if node_id:
                latest_by_node[node_id] = item
        lines.append("Specialist topology")
        for item in node_values:
            node_id = str(field(item, "id"))
            latest = latest_by_node.get(node_id)
            state = status_of(latest) if latest is not None else status_of(item)
            depth = max(0, int(field(item, "depth", 1)) - 1)
            title = str(field(item, "title", node_id))
            lines.append(
                f"  {'  ' * depth}[{node_index[node_id]:02d}] "
                f"{title} | {state or 'pending'}"
            )
    lines.extend(
        (
            "",
            "View only: /agent NUMBER|NODE_ID|AGENT_ID",
            "All specialists: /agents --all",
        )
    )
    return "\n".join(lines)


def render_agent_detail(
    *,
    node: Any | None,
    agent_run: Any | None,
    profile: Mapping[str, Any] | None = None,
    trace: Any | None = None,
    ancestry: Iterable[Any] = (),
    display_index: int | None = None,
) -> str:
    """Render one specialist assignment without exposing hidden reasoning."""

    def field(item: Any, name: str, default: Any = "") -> Any:
        if item is None:
            return default
        if isinstance(item, Mapping):
            value = item.get(name, default)
        else:
            value = getattr(item, name, default)
        return getattr(value, "value", value)

    node_id = str(field(node, "id", field(agent_run, "work_node_id", "-")))
    title = str(field(node, "title", field(agent_run, "role", "Agent")))
    path = " > ".join(
        [str(field(item, "title", field(item, "id", ""))) for item in ancestry]
        + [title]
    )
    status = field(agent_run, "status", field(node, "status", "pending"))
    phase = field(agent_run, "phase", field(node, "phase", "-"))
    role = field(agent_run, "role", field(node, "assigned_role", "specialist"))
    lines = [
        "Specialist view | READ ONLY",
        f"  specialist  {display_index if display_index is not None else '-'} | {title}",
        f"  node        {node_id}",
        f"  agent run   {field(agent_run, 'id', '(not started)')}",
        f"  state       {status} | {role}/{phase}",
        f"  path        {path}",
    ]
    mission = str((profile or {}).get("mission") or field(node, "objective", "")).strip()
    deliverable = str((profile or {}).get("deliverable") or "").strip()
    interfaces = list((profile or {}).get("owned_interfaces") or ())
    dependencies = list((profile or {}).get("dependencies") or field(node, "depends_on", ()))
    contract = field(node, "contract", None)
    if contract is not None:
        interfaces = interfaces or list(field(contract, "owned_interfaces", ()))
        dependencies = dependencies or list(field(contract, "depends_on", ()))
    if mission:
        lines.extend(("", "Mission", mission))
    if deliverable:
        lines.extend(("", "Deliverable", deliverable))
    lines.extend(
        (
            "",
            "Owned interfaces",
            "\n".join(f"  - {value}" for value in interfaces)
            if interfaces
            else "  (none recorded)",
            "",
            "Dependencies",
            "\n".join(f"  - {value}" for value in dependencies)
            if dependencies
            else "  (none)",
        )
    )
    if trace is not None:
        lines.extend(("", "Current redacted prompt", render_trace(trace)))
    else:
        lines.extend(
            (
                "",
                "Current redacted prompt",
                "  (model call not started, or this run predates live prompt snapshots)",
                "  The mission and typed contract above remain the authoritative assignment.",
            )
        )
    summary = str(field(agent_run, "error", "") or "").strip()
    result = field(agent_run, "result", None)
    if not summary and result is not None:
        summary = str(field(result, "summary", "")).strip()
    if summary:
        lines.extend(("", "Latest activity", summary))
    lines.append("\nNo controls are available in this view.")
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
    execution_started = view.approved_revision is not None or view.status.lower() in {
        "running", "paused", "reviewing", "verifying", "completed",
    }
    title = (" EXECUTION PLAN " if execution_started else " COMPLETE CHECKLIST ")[:inner]
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
    if view.plan_applicability and not execution_started:
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
    if view.expected_changes and not execution_started:
        lines.append("|" + _fit(" EXPECTED WORKSPACE CHANGES", inner) + "|")
        for item in view.expected_changes:
            supports = ",".join(str(value) for value in item.get("supports_tasks", ()))
            value = f"[{supports or '-'}] {item.get('path', '-')}: {item.get('intent', '')}"
            for change_line in _wrap(value, max(10, inner - 4), max_lines=20):
                lines.append("|" + _fit("   " + change_line, inner) + "|")
    lines.append("+" + "-" * inner + "+")
    if not view.tasks:
        lines.append("|" + _fit(" (no checklist items)", inner) + "|")

    def render_tasks(tasks: list[TaskView]) -> None:
        for task in tasks:
            prefix = f" {_task_mark(task.status)} {task.id} "
            wrapped = _wrap(task.title, max(10, inner - len(prefix)), max_lines=20)
            lines.append("|" + _fit(prefix + wrapped[0], inner) + "|")
            for continuation in wrapped[1:]:
                lines.append("|" + _fit(" " * len(prefix) + continuation, inner) + "|")
            if execution_started:
                dependencies = ",".join(task.depends_on) if task.depends_on else "ready"
                meta = f"       {task.status.replace('_', ' ')} | risk {task.risk} | dependencies {dependencies}"
            else:
                dependencies = ",".join(task.depends_on) if task.depends_on else "-"
                meta = f"     risk={task.risk} depends_on={dependencies}"
            lines.append("|" + _fit(meta, inner) + "|")
            for criterion in task.acceptance_criteria:
                for index, criterion_line in enumerate(_wrap(criterion, max(10, inner - 12), max_lines=20)):
                    label = "       accept: " if index == 0 else "               "
                    lines.append("|" + _fit(label + criterion_line, inner) + "|")
            for check in task.verification:
                for index, check_line in enumerate(_wrap(check, max(10, inner - 12), max_lines=20)):
                    label = "       verify: " if index == 0 else "               "
                    lines.append("|" + _fit(label + check_line, inner) + "|")

    if execution_started and view.tasks:
        groups = (
            ("IN USE", [task for task in view.tasks if task.status in {"pending", "in_progress", "blocked", "uncertain"}]),
            ("COMPLETED", [task for task in view.tasks if task.status == "done"]),
            ("NOT USED", [task for task in view.tasks if task.status == "skipped"]),
        )
        for label, tasks in groups:
            if not tasks:
                continue
            lines.append("|" + _fit(f" {label} ({len(tasks)})", inner) + "|")
            render_tasks(tasks)
    else:
        render_tasks(view.tasks)
    lines.append("+" + "-" * inner + "+")
    return "\n".join(lines)


HELP_TEXT = """\
Slash palette and modes
  /                           open the slash-command palette
  /mode                      show the current interaction mode
  /mode normal               intent intake + durable goal + planning/review/automatic execution
  /mode ultra                recursive specialists + Project Brain + full quality gates
  /settings                  show safe session settings (never secrets)
  /settings NAME VALUE       change mode, color, or a runtime limit this session
  /model                     reopen the tool-capable model picker at a safe checkpoint
  /permissions normal|full   choose approvals or Docker-isolated Full access
  /setup                     build/validate the one-time Full-access sandbox
  /skills                    show actual local tools and live capability status
  /processes                 list managed processes and HTML previews
  /stop-process ID           stop a managed process or preview

Goal and plan
  /goal TEXT                 start a durable goal (plain text also works when idle)
  /approve [REV]             approve exactly the displayed plan revision
  /reject FEEDBACK           reject and regenerate the draft
  /replan FEEDBACK           ask for a revised plan
  /plan                      show the dashboard/checklist
  /questions                 show non-discoverable intake/planning decisions
  /answer ID 1|2|3|TEXT      select a suggestion or save a free-form answer

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
  /agents [--all|AGENT]      read-only live swarm list/topology
  /agent NUMBER|ID           inspect one specialist and its redacted prompt
  /memory [SECTION]          inspect/search Project Brain entries
  /trace [latest|RUN_ID]     show redacted prompts, context, and reasoning summaries
  /thinking                  expand redacted thoughts captured this session
  /insights [NODE]           show findings, decisions, and lessons
  /metrics                   show token, agent, node, fix, and concurrency metrics
  /doctor [--live] [--record]
                              audit weak-model architecture; --live probes Ollama, --record saves benchmark history
  /resolve ENTITY_ID applied|not-run NOTE
                              reconcile a crash-window action/worker after inspection
  /cancel CANCEL             explicitly abandon an unfinished goal
  /quit                      exit safely; unfinished state resumes next launch

All legacy :commands remain supported.
"""


_TOOL_ACTIVITY: dict[str, tuple[str, str, str]] = {
    "list_files": ("search", "Listing workspace", "Inspected workspace"),
    "read_file": ("search", "Reading file", "Read file"),
    "grep": ("search", "Searching workspace", "Searched workspace"),
    "write_file": ("run", "Writing file", "Wrote file"),
    "edit_file": ("run", "Editing file", "Edited file"),
    "run_bash": ("run", "Running command", "Ran command"),
    "run_command": ("run", "Running command", "Ran command"),
    "apply_patch": ("run", "Applying patch", "Applied patch"),
    "materialize_artifact": ("run", "Writing generated artifact", "Wrote generated artifact"),
    "install_dependencies": ("sync", "Installing project dependencies", "Installed dependencies"),
    "start_process": ("run", "Starting managed process", "Started process"),
    "poll_process": ("sync", "Checking managed process", "Checked process"),
    "read_process_output": ("search", "Reading process output", "Read process output"),
    "stop_process": ("run", "Stopping managed process", "Stopped process"),
    "open_path": ("run", "Opening workspace file", "Opened file"),
    "preview_html": ("run", "Starting and verifying HTML preview", "Preview ready"),
    "inspect_preview": ("search", "Inspecting HTML preview", "Inspected preview"),
    "stop_preview": ("run", "Stopping HTML preview", "Stopped preview"),
    "propose_plan": ("plan", "Preparing plan", "Prepared plan"),
    "request_plan_input": ("plan", "Preparing question", "Prepared question"),
    "submit_plan_review": ("review", "Reviewing plan", "Reviewed plan"),
    "submit_review": ("review", "Reviewing completion", "Reviewed completion"),
    "delegate_task": ("sync", "Starting focused agent", "Started focused agent"),
    "update_task": ("sync", "Updating checklist", "Updated checklist"),
    "finish_goal": ("review", "Checking completion", "Checked completion"),
}


def _activity_state_for_actor(actor: str) -> str:
    normalized = str(actor).casefold()
    if "review" in normalized or "critic" in normalized:
        return "review"
    if any(token in normalized for token in ("coordinator", "worker", "implement")):
        return "run"
    if "planner" in normalized:
        return "plan"
    return "sync"


def _actual_activity_label(value: str) -> str:
    """Use the latest real reasoning statement as the visible activity label."""

    redacted = redact_text(str(value), 40_000)
    lines = [" ".join(line.split()) for line in redacted.splitlines()]
    lines = [
        line
        for line in lines
        if line and not line.lstrip().startswith(("{", "[", "\""))
    ]
    if not lines:
        return ""
    sentences = [
        item.strip()
        for item in re.split(r"(?<=[.!?])\s+", lines[-1])
        if item.strip()
    ]
    return sentences[-1] if sentences else lines[-1]


def _is_user_facing_model_text(actor: str) -> bool:
    """Return whether streamed model text is an answer rather than harness protocol.

    Chat text may contain Markdown, source code, or a generated artifact.  It must
    never be appended to the provider-thought buffer merely because the event has
    an actor (all runtime model events do).
    """

    return str(actor).strip().casefold() in {"chat", "assistant", "user_facing"}


def _visible_thought_lines(value: str, limit: int = 8) -> tuple[str, ...]:
    """Extract concise prose steps for the transcript, excluding code/protocol."""

    result: list[str] = []
    fenced = False
    for raw in redact_text(str(value), 40_000).splitlines():
        stripped = raw.strip()
        if stripped.startswith("```"):
            fenced = not fenced
            continue
        if fenced or not stripped or stripped.startswith(("{", "}", "[", "]", '"')):
            continue
        if re.match(r"^(?:def |class |import |from |const |let |var |function |<[/!A-Za-z])", stripped):
            continue
        line = " ".join(stripped.lstrip("-*0123456789. ").split())
        if line and line not in result:
            result.append(line[:240])
        if len(result) >= limit:
            break
    return tuple(result)


class _LiveActivity:
    """Maintain one animated single-line activity region without scrollback spam."""

    def __init__(self, owner: "ConsoleUI") -> None:
        self.owner = owner
        self._stop = Event()
        self._thread: Thread | None = None
        self._key_thread: Thread | None = None
        self._visible = False
        self._state = "idle"
        self._label = ""
        self._started = 0.0
        self._last_signal = 0.0
        self._ultra_live = False
        self._ultra_run_id = ""
        self._ultra_phase = ""
        self._current_work = ""
        self._last_completed = ""
        self._next_work = ""
        self._total_nodes = 0
        self._completed_nodes = 0
        self._active_nodes = 0
        self._pending_nodes = 0
        self._completed_node_ids: set[str] = set()
        self._node_started_at: dict[str, float] = {}
        self._node_durations: list[float] = []
        self._rendered_lines = 0
        self._state_lock = RLock()

    def start(self, state: str, label: str) -> None:
        if not self.owner.live_activity_enabled:
            return
        with self._state_lock:
            if self._ultra_live and self._thread is not None and self._thread.is_alive():
                self._state = "ultra"
                self._label = " ".join(str(label).split())
                self._current_work = self._label
                self._last_signal = time.monotonic()
                return
        self.stop()
        with self._state_lock:
            self._state = state
            self._label = " ".join(str(label).split())
            self._started = time.monotonic()
            self._last_signal = self._started
            self._ultra_live = False
        self._stop = Event()
        self._thread = Thread(target=self._run, name="ga3bad-ui-motion", daemon=True)
        self._thread.start()
        if self._supports_esc_interrupt():
            self._key_thread = Thread(
                target=self._watch_escape,
                name="ga3bad-ui-escape",
                daemon=True,
            )
            self._key_thread.start()

    @property
    def ultra_live(self) -> bool:
        with self._state_lock:
            return self._ultra_live

    def update_label(self, text: str) -> None:
        with self._state_lock:
            self._label = " ".join(str(text).split())
            self._last_signal = time.monotonic()

    def update_ultra(self, kind: str, message: str, data: Mapping[str, Any]) -> None:
        """Update the persistent ULTRA loader from observable runtime facts."""

        if not self.owner.live_activity_enabled:
            return
        now = time.monotonic()
        start_thread = False
        with self._state_lock:
            run_id = str(data.get("run_id") or self._ultra_run_id or "")
            new_run = bool(run_id and run_id != self._ultra_run_id)
            if not self._ultra_live or new_run:
                self._ultra_live = True
                self._ultra_run_id = run_id
                if new_run or not self._started:
                    self._started = now
                    self._completed_node_ids.clear()
                    self._node_started_at.clear()
                    self._node_durations.clear()
                start_thread = self._thread is None or not self._thread.is_alive()
            self._last_signal = now
            self._state = "ultra"
            phase = str(data.get("ultra_phase") or data.get("phase") or "").strip()
            if kind == "ultra.phase" and phase:
                self._ultra_phase = phase
            elif not self._ultra_phase:
                self._ultra_phase = phase or "working"

            for key, attribute in (
                ("total_nodes", "_total_nodes"),
                ("completed_nodes", "_completed_nodes"),
                ("active_nodes", "_active_nodes"),
                ("pending_nodes", "_pending_nodes"),
            ):
                if key in data:
                    try:
                        setattr(self, attribute, max(0, int(data.get(key) or 0)))
                    except (TypeError, ValueError):
                        pass

            node_id = str(data.get("node_id") or data.get("current_node_id") or "").strip()
            node_title = str(data.get("current_node_title") or node_id).strip()
            status = str(data.get("status") or "").casefold()
            role = str(data.get("role") or "").strip()
            work_phase = str(data.get("current_node_phase") or data.get("phase") or "").strip()
            if kind == "ultra.node" and status == "running" and node_id:
                self._node_started_at.setdefault(node_id, now)
            if kind == "ultra.node" and status == "completed" and node_id:
                if node_id not in self._completed_node_ids:
                    self._completed_node_ids.add(node_id)
                    started = self._node_started_at.pop(node_id, None)
                    if started is not None:
                        self._node_durations.append(max(0.1, now - started))
                        del self._node_durations[:-12]
                self._last_completed = node_title or node_id
            elif kind == "ultra.agent":
                self._last_completed = " ".join(str(message).split())[:120]

            if kind == "ultra.agent_started":
                detail = " / ".join(item for item in (role, work_phase) if item)
                self._current_work = node_title or "Preparing specialist"
                if detail:
                    self._current_work += f" · {detail}"
            elif kind == "ultra.node" and status == "running":
                self._current_work = node_title or "Running specialist"
            elif kind == "ultra.phase":
                self._current_work = " ".join(str(message).split())
            elif kind == "ultra.graph_ready" and not self._current_work:
                self._current_work = "Scheduling the first ready specialist"
            next_titles = data.get("next_node_titles")
            if isinstance(next_titles, (list, tuple)) and next_titles:
                self._next_work = ", ".join(str(item) for item in next_titles[:2])
            self._label = self._current_work or " ".join(str(message).split())

        if start_thread:
            self._stop = Event()
            self._thread = Thread(target=self._run, name="ga3bad-ultra-live", daemon=True)
            self._thread.start()
            if self._supports_esc_interrupt():
                self._key_thread = Thread(
                    target=self._watch_escape,
                    name="ga3bad-ui-escape",
                    daemon=True,
                )
                self._key_thread.start()

    def _run(self) -> None:
        tick = 0
        interval = 0.65 if self.owner.reduced_motion else 0.12
        while not self._stop.is_set():
            self._draw(tick)
            tick += 1
            if self._stop.wait(interval):
                break

    def _draw(self, tick: int) -> None:
        with self._state_lock:
            label_value = self._label
            started = self._started
            last_signal = self._last_signal
            ultra_live = self._ultra_live
            ultra_phase = self._ultra_phase
            current_work = self._current_work
            last_completed = self._last_completed
            next_work = self._next_work
            total_nodes = self._total_nodes
            completed_nodes = self._completed_nodes
            active_nodes = self._active_nodes
            pending_nodes = self._pending_nodes
            durations = tuple(self._node_durations)
        elapsed = max(0, int(time.monotonic() - started))
        if ultra_live:
            self._draw_ultra(
                tick,
                elapsed=elapsed,
                signal_age=max(0, int(time.monotonic() - last_signal)),
                phase=ultra_phase,
                current=current_work or label_value,
                last_completed=last_completed,
                next_work=next_work,
                total=total_nodes,
                completed=completed_nodes,
                active=active_nodes,
                pending=pending_nodes,
                durations=durations,
            )
            return
        columns = max(24, shutil.get_terminal_size((112, 30)).columns)
        label = _fit("Working", max(8, min(96, columns - 28))).rstrip()
        bullet = self.owner._icon("\u2022", "o")
        separator = "\u2022" if self.owner._can_encode("\u2022") else "*"
        interrupt_key = "esc" if self._supports_esc_interrupt() else "Ctrl+C"
        interrupt = f"({elapsed}s {separator} {interrupt_key} to interrupt)"
        if label_value:
            # The detailed activity still drives /thinking and completed event
            # lines; the live row itself mirrors Codex's compact chrome.
            interrupt = _fit(interrupt, max(8, columns - len(label) - 5)).rstrip()
        with self.owner._lock:
            if self._visible:
                print("\033[1A", end="", file=self.owner.stream)
            else:
                print("\033[?25l", end="", file=self.owner.stream)
                self._visible = True
            print(
                f"\r\033[2K{self.owner.dim}{bullet}{self.owner.reset}  "
                f"{self.owner.bold}{label}{self.owner.reset} "
                f"{self.owner.dim}{interrupt}{self.owner.reset}",
                file=self.owner.stream,
            )
            self.owner.stream.flush()

    @staticmethod
    def _duration(value: float) -> str:
        seconds = max(0, int(value))
        if seconds < 60:
            return f"{seconds}s"
        minutes, seconds = divmod(seconds, 60)
        if minutes < 60:
            return f"{minutes}m {seconds:02d}s"
        hours, minutes = divmod(minutes, 60)
        return f"{hours}h {minutes:02d}m"

    def _draw_ultra(
        self,
        tick: int,
        *,
        elapsed: int,
        signal_age: int,
        phase: str,
        current: str,
        last_completed: str,
        next_work: str,
        total: int,
        completed: int,
        active: int,
        pending: int,
        durations: tuple[float, ...],
    ) -> None:
        columns = max(36, shutil.get_terminal_size((112, 30)).columns)
        content_width = max(16, columns - 4)
        frames = ("◐", "◓", "◑", "◒") if self.owner._can_encode("◐") else ("|", "/", "-", "\\")
        pulse = frames[0 if self.owner.reduced_motion else tick % len(frames)]
        percent = min(100, round(100 * completed / total)) if total else 0
        bar_width = max(8, min(28, columns - 54))
        filled = round(bar_width * completed / total) if total else 0
        bar = "[" + "#" * filled + "-" * (bar_width - filled) + "]"
        if total and completed and durations:
            average = sum(durations) / len(durations)
            remaining = max(0, total - completed)
            estimate = average * remaining
            eta = f"ETA ~{self._duration(estimate * 0.8)}–{self._duration(estimate * 1.25)}"
        elif total and completed:
            estimate = elapsed * max(0, total - completed) / max(1, completed)
            eta = f"ETA ~{self._duration(estimate * 0.75)}–{self._duration(estimate * 1.35)}"
        else:
            eta = "ETA learning from first completed specialist"

        if signal_age <= 15:
            signal = "live now"
        elif signal_age <= 120:
            signal = f"model call open · quiet {self._duration(signal_age)}"
        else:
            signal = f"long model call · no event {self._duration(signal_age)}"
        interrupt_key = "Esc" if self._supports_esc_interrupt() else "Ctrl+C"
        phase_label = (phase or "working").replace("_", " ").upper()
        progress_label = (
            f"{bar} {completed}/{total} · {percent}% · active {active} · queued {pending} · {eta}"
            if total
            else f"mapping specialist graph · {eta}"
        )
        rows = [
            f"{pulse} ULTRA {phase_label} · elapsed {self._duration(elapsed)} · {signal}",
            f"  {progress_label}",
            f"  now   {_fit(current or 'Preparing the next step', max(8, content_width - 6)).rstrip()}",
            f"  done  {_fit((last_completed or 'No specialist completed yet') + ' · next ' + (next_work or 'scheduler deciding'), max(8, content_width - 6)).rstrip()}",
            f"  {interrupt_key} checkpoints safely · /agents details · /status snapshot",
        ]
        rows = [_fit(row, columns).rstrip() for row in rows]
        with self.owner._lock:
            if self._visible and self._rendered_lines:
                print(f"\033[{self._rendered_lines}A", end="", file=self.owner.stream)
            else:
                print("\033[?25l", end="", file=self.owner.stream)
                self._visible = True
            for row in rows:
                print(f"\r\033[2K{row}", file=self.owner.stream)
            self._rendered_lines = len(rows)
            self.owner.stream.flush()

    def _supports_esc_interrupt(self) -> bool:
        return os.name == "nt" and self.owner.input_func is input and _isatty(sys.stdin)

    def _watch_escape(self) -> None:
        try:
            import _thread
            import msvcrt
        except ImportError:  # pragma: no cover - Windows-only enhancement.
            return
        while not self._stop.wait(0.05):
            try:
                if self.owner._composer_active:
                    continue
                if msvcrt.kbhit() and msvcrt.getwch() == "\x1b":
                    _thread.interrupt_main()
                    return
            except OSError:
                return

    def stop(self) -> None:
        thread = self._thread
        self._stop.set()
        if thread is not None and thread.is_alive():
            thread.join(timeout=0.25)
        key_thread = self._key_thread
        if key_thread is not None and key_thread.is_alive():
            key_thread.join(timeout=0.25)
        self._thread = None
        self._key_thread = None
        with self.owner._lock:
            if not self._visible:
                return
            lines = max(1, self._rendered_lines)
            print(f"\033[{lines}A", end="", file=self.owner.stream)
            for _ in range(lines):
                print("\r\033[2K", file=self.owner.stream)
            print(f"\033[{lines}A\r\033[?25h", end="", file=self.owner.stream)
            self.owner.stream.flush()
            self._visible = False
            self._rendered_lines = 0
            self._ultra_live = False


class ConsoleUI:
    """Event renderer with serialized output for model streaming and workers."""

    def __init__(
        self,
        stream: TextIO | None = None,
        color: bool | None = None,
        input_func: Any = input,
        interaction_mode: str | InteractionMode = InteractionMode.NORMAL,
        plain: bool = False,
        reduced_motion: bool = False,
    ) -> None:
        self.stream = stream or sys.stdout
        self.input_func = input_func
        self._lock = RLock()
        self._event_lock = RLock()
        self._approval_lock = RLock()
        self._stream_kind: str | None = None
        self._prompt_session: Any = None
        self._composer_active = False
        self._background_working = False
        self._pasted_content: dict[str, str] = {}
        self._prompt_bindings: Any = None
        self._prompt_default = ""
        self._current_status = "idle"
        self.plain = bool(plain)
        self.reduced_motion = bool(reduced_motion)
        self._live_activity = _LiveActivity(self)
        self._pending_tools: dict[tuple[str, str], dict[str, Any]] = {}
        self._plan_format_retries = 0
        self._plan_recovered_retries = 0
        self._technical_activity: list[str] = []
        self._active_thought: dict[str, Any] | None = None
        self._thought_blocks: list[dict[str, Any]] = []
        self._thought_sequence = 0
        self._coalesced_activity: dict[str, dict[str, Any]] = {}
        self._last_activity_key: str | None = None
        self._last_event_kind = ""
        self._last_status_signature: tuple[Any, ...] | None = None
        self._full_screen_depth = 0
        self._buffered_events: list[UIEvent] = []
        self.activity: list[str] = []
        self.interaction_mode = InteractionMode.parse(interaction_mode)
        self.access_level = "normal"
        self.execution_class = "local"
        self.active_agents = 0
        self.active_model = "model"
        self.reasoning_effort = "medium"
        self.workspace_label = ""
        self.context_tokens = 0
        self.vim_mode = False
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
        self.color = (
            _isatty(self.stream) and "NO_COLOR" not in os.environ
            if normalized == "auto"
            else normalized == "on"
        )
        self.bold = _ansi("\033[1m", self.color)
        self.dim = _ansi("\033[2m", self.color)
        self.cyan = _ansi("\033[36m", self.color)
        self.green = _ansi("\033[32m", self.color)
        self.yellow = _ansi("\033[33m", self.color)
        self.gold = _ansi("\033[38;5;220m", self.color)
        self.response = _ansi("\033[38;5;255m", self.color)
        self.thought_done = _ansi("\033[38;5;242m", self.color)
        self.red = _ansi("\033[31m", self.color)
        self.magenta = _ansi("\033[35m", self.color)
        self.reset = _ansi("\033[0m", self.color)

    @property
    def live_activity_enabled(self) -> bool:
        return not self.plain and _isatty(self.stream)

    def set_mode(self, mode: str | InteractionMode) -> None:
        self.interaction_mode = InteractionMode.parse(mode)

    def prefill_prompt(self, text: str) -> None:
        """Seed the next composer turn after a palette command needs arguments."""

        self._prompt_default = str(text)

    def set_background_working(self, active: bool) -> None:
        """Keep the composer available without tearing down live worker chrome."""
        self._background_working = bool(active)

    def set_vim_mode(self, enabled: bool | None = None) -> bool:
        self.vim_mode = (not self.vim_mode) if enabled is None else bool(enabled)
        self._prompt_session = None
        return self.vim_mode

    def set_runtime_identity(
        self,
        *,
        access_level: str = "normal",
        execution_class: str = "local",
        active_agents: int = 0,
        model: str | None = None,
        reasoning_effort: str | None = None,
        workspace: str | None = None,
    ) -> None:
        self.access_level = str(access_level).lower()
        self.execution_class = str(execution_class).lower()
        self.active_agents = max(0, int(active_agents))
        if model is not None:
            self.active_model = str(model)
        if reasoning_effort is not None:
            self.reasoning_effort = str(reasoning_effort)
        if workspace is not None:
            self.workspace_label = str(workspace)

    def thought_blocks(self) -> tuple[dict[str, Any], ...]:
        """Return redacted, session-only thought blocks for the inspector."""

        values = [dict(item) for item in self._thought_blocks]
        if self._active_thought is not None:
            active = dict(self._active_thought)
            active["text"] = redact_text(str(active.get("text", "")), 40_000)
            active["active"] = True
            values.append(active)
        return tuple(values)

    def _begin_thought(self, actor: str, step: Any) -> None:
        self._finish_thought()
        self._thought_sequence += 1
        self._active_thought = {
            "id": self._thought_sequence,
            "actor": str(actor),
            "step": step,
            "text": "",
            "started": time.monotonic(),
            "active": True,
        }
        actor_label = str(actor).replace("_", " ").title()
        self._live_activity.start(
            _activity_state_for_actor(actor),
            f"{actor_label} · step {step}",
        )

    def _append_thought(self, fragment: str) -> None:
        if self._active_thought is None:
            self._begin_thought("agent", "?")
        assert self._active_thought is not None
        combined = str(self._active_thought.get("text", "")) + str(fragment)
        self._active_thought["text"] = combined[-40_000:]
        actual_label = _actual_activity_label(combined)
        if actual_label:
            self._live_activity.update_label(actual_label)

    def _finish_thought(self) -> None:
        block = self._active_thought
        if block is None:
            return
        self._active_thought = None
        ultra_live = self._live_activity.ultra_live
        if not ultra_live:
            self._live_activity.stop()
        text = redact_text(str(block.get("text", "")), 40_000).strip()
        if not text:
            return
        duration = max(1, int(time.monotonic() - float(block.get("started", time.monotonic()))))
        finished = {
            **block,
            "text": text,
            "duration_seconds": duration,
            "active": False,
        }
        self._thought_blocks.append(finished)
        del self._thought_blocks[:-50]
        visible_lines = _visible_thought_lines(text)
        summary = visible_lines[-1] if visible_lines else _actual_activity_label(text)
        if not summary:
            summary = (
                f"{str(finished.get('actor', 'agent')).replace('_', ' ').title()} "
                f"step {finished.get('step', '?')}"
            )
        if ultra_live:
            self._live_activity.update_label(summary)
            return
        for line in visible_lines:
            self._live_line(">", line, self.thought_done)
        self._live_line(
            self._icon("◇", "[~]"),
            f"{summary} · {duration}s · collapsed",
            self.thought_done,
        )

    def write(self, text: str = "", *, end: str = "\n", flush: bool = True) -> None:
        self._finish_thought()
        self._live_activity.stop()
        self._last_activity_key = None
        with self._lock:
            print(text, end=end, file=self.stream, flush=flush)

    def close(self) -> None:
        self._finish_thought()
        self._live_activity.stop()

    @contextmanager
    def full_screen_modal(self):
        """Buffer worker events while an alternate-screen picker owns stdout."""

        self._live_activity.stop()
        with self._event_lock:
            self._full_screen_depth += 1
        try:
            yield
        finally:
            with self._event_lock:
                self._full_screen_depth = max(0, self._full_screen_depth - 1)
                if self._full_screen_depth == 0:
                    buffered = self._buffered_events
                    self._buffered_events = []
                    # Keep the event lock while replaying so a newly arriving
                    # worker event cannot overtake older buffered activity.
                    for event in buffered:
                        self._render_event(event)

    def show_dashboard(self, view: DashboardView) -> None:
        self._current_status = str(view.status or "idle")
        view.activity = view.activity or self.activity[-4:]
        view.interaction_mode = self.interaction_mode.value
        self.write(render_dashboard(view))

    def show_status(self, view: DashboardView, *, force: bool = False) -> None:
        self._current_status = str(view.status or "idle")
        view.activity = view.activity or self.activity[-4:]
        view.interaction_mode = self.interaction_mode.value
        completed = sum(task.status in {"done", "skipped"} for task in view.tasks)
        signature = (
            view.status,
            view.plan_revision,
            completed,
            len(view.tasks),
            view.waiting_question,
            self.interaction_mode.value,
            self.access_level,
            self.execution_class,
            self.active_agents,
        )
        if self.live_activity_enabled and not force and self._last_status_signature is not None:
            if signature == self._last_status_signature:
                return
            self._last_status_signature = signature
            if self._last_event_kind in {"plan", "error", "warning", "checkpoint", "questions"}:
                return
            summary = (
                f"{str(view.status).replace('_', ' ').title()} · "
                f"{completed}/{len(view.tasks)} tasks · plan r{view.plan_revision}"
            )
            self._live_line(self._icon("◇", "[>]"), summary, self.cyan)
            return
        self._last_status_signature = signature
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

    def _icon(self, unicode_value: str, ascii_value: str) -> str:
        return unicode_value if self._can_encode(unicode_value) else ascii_value

    def _can_encode(self, value: str) -> bool:
        encoding = getattr(self.stream, "encoding", None) or "utf-8"
        try:
            value.encode(encoding)
        except (LookupError, UnicodeEncodeError):
            return False
        return True

    def _prompt_mark(self) -> str:
        return self._icon("\u203a", ">")

    def _prompt_style(self):
        if Style is None or not self.color:
            return None
        return Style.from_dict(
            {
                "prompt": "#ffffff bold",
                "placeholder": "#666666",
                "bottom-toolbar": "bg:#0b0b0b #707070 noreverse",
                "footer.model": "bg:#0b0b0b #e8d58f bold noreverse",
                "footer.effort": "bg:#0b0b0b #b8b8b8 noreverse",
                "footer.separator": "bg:#0b0b0b #555555 noreverse",
                "footer.path": "bg:#0b0b0b #8fd39a noreverse",
                "footer.memory": "bg:#0b0b0b #7fb4d8 noreverse",
                "completion-menu.completion": "bg:#000000 #d8d8d8",
                "completion-menu.completion.current": "bg:#000000 #4db6ff bold",
                "completion-menu.meta.completion": "bg:#000000 #6f6f6f",
                "completion-menu.meta.completion.current": "bg:#000000 #4db6ff bold",
            }
        )

    def _live_line(
        self,
        icon: str,
        message: str,
        color: str = "",
        *,
        dedupe_key: str | None = None,
        show_count: bool = True,
    ) -> None:
        clean = " ".join(str(message).split())
        columns = max(24, shutil.get_terminal_size((112, 30)).columns)
        message_width = max(8, min(118, columns - len(icon) - 1))
        key = dedupe_key or ""
        if key and key in self._coalesced_activity:
            record = self._coalesced_activity[key]
            if self._last_activity_key != key:
                # The same operation in a later phase (for example a
                # verification read) is meaningful and should be visible.
                self._coalesced_activity.pop(key, None)
            else:
                record["count"] = int(record.get("count", 1)) + 1
            if self._last_activity_key == key and show_count and self.live_activity_enabled:
                count = record["count"]
                counted = _fit(f"{clean} · x{count}", message_width).rstrip()
                with self._lock:
                    print("\033[1A\r\033[2K", end="", file=self.stream)
                    print(
                        f"{self.bold}{color}{icon}{self.reset} "
                        f"{counted}",
                        file=self.stream,
                        flush=True,
                    )
            if self._last_activity_key == key:
                return
        with self._lock:
            print(
                f"{self.bold}{color}{icon}{self.reset} {_fit(clean, message_width).rstrip()}",
                file=self.stream,
                flush=True,
            )
        self.activity.append(clean[:180])
        del self.activity[:-50]
        if key:
            self._coalesced_activity[key] = {
                "count": 1,
                "message": clean,
                "icon": icon,
            }
            while len(self._coalesced_activity) > 200:
                oldest = next(iter(self._coalesced_activity))
                self._coalesced_activity.pop(oldest, None)
            self._last_activity_key = key
        else:
            self._last_activity_key = None

    def _finish_live_stream(self) -> None:
        if self._stream_kind:
            with self._lock:
                print(self.reset, file=self.stream)
                self.stream.flush()
            self._stream_kind = None

    @staticmethod
    def _tool_detail(name: str, args: Mapping[str, Any]) -> str:
        if name in {"list_files", "read_file", "grep", "write_file", "edit_file"}:
            path = str(args.get("path") or ".")
            return f" · {path}"
        if name == "run_bash":
            command = " ".join(str(args.get("command") or "").split())
            return f" · {_fit(command, 72).rstrip()}" if command else ""
        return ""

    @staticmethod
    def _pending_key(data: Mapping[str, Any]) -> tuple[str, str]:
        return (str(data.get("actor", "agent")), str(data.get("node_id") or ""))

    def _take_pending_tool(
        self,
        data: Mapping[str, Any],
        tool_name: str,
    ) -> dict[str, Any]:
        key = self._pending_key(data)
        pending = self._pending_tools.pop(key, None)
        if pending is not None:
            return pending
        # Older event producers did not attach node_id to results. Fall back
        # only to an actor/tool match, preserving concurrent node isolation.
        actor = key[0]
        for candidate_key, candidate in reversed(tuple(self._pending_tools.items())):
            if candidate_key[0] == actor and candidate.get("name") == tool_name:
                self._pending_tools.pop(candidate_key, None)
                return candidate
        return {}

    def _on_live_event(self, event: UIEvent) -> None:
        """Render meaningful operations while keeping raw protocol detail folded."""

        kind, message, data = event.kind, event.message, event.data
        if kind == "model_thought":
            self._append_thought(message)
            return
        if kind == "step":
            self._finish_live_stream()
            actor = str(data.get("actor", "agent"))
            self._begin_thought(actor, data.get("step", "?"))
            return

        if (
            kind == "model_text"
            and data.get("actor")
            and not _is_user_facing_model_text(str(data.get("actor")))
        ):
            # Planner/coordinator/reviewer prose is protocol work, not a user
            # response.  Some providers emit JSON or a narrated tool choice in
            # content before the actual control call; keep it inside the same
            # live/collapsible thought block instead of polluting scrollback.
            prefix = ""
            if self._active_thought is not None:
                prior = str(self._active_thought.get("text", ""))
                if prior and not prior[-1:].isspace() and message and not message[:1].isspace():
                    prefix = "\n"
            self._append_thought(prefix + message)
            return

        # Usage, text, and tool/control events all mark the end of the current
        # provider reasoning block.  Fold it before rendering the next step.
        self._finish_thought()
        if kind == "model_text":
            self._live_activity.stop()
            self._last_activity_key = None
            if self._stream_kind != kind:
                self._finish_live_stream()
                with self._lock:
                    print(f"{self.bold}{self.response}Response{self.reset}", file=self.stream)
                self._stream_kind = kind
            with self._lock:
                print(f"{self.response}{message}", end="", file=self.stream, flush=True)
            return

        self._finish_live_stream()
        if kind.startswith("ultra."):
            if kind == "ultra.agent_started":
                self.active_agents += 1
            elif kind == "ultra.agent":
                self.active_agents = max(0, self.active_agents - 1)
            terminal_kinds = {
                "ultra.foundation_ready",
                "ultra.completed",
                "ultra.cancelled",
                "ultra.revision_required",
                "ultra.paused",
            }
            if kind in terminal_kinds:
                self._live_activity.stop()
                color = self.green if kind == "ultra.completed" else self.gold
                self._live_line(
                    self._icon("✓", "[x]")
                    if kind == "ultra.completed"
                    else self._icon("◇", "[>]"),
                    message or kind,
                    color,
                    dedupe_key=f"{kind}:{' '.join(str(message).split())[:240]}",
                )
            else:
                self._live_activity.update_ultra(kind, message or kind, data)
            return

        if kind == "tool_call":
            args = data.get("args", {})
            args = args if isinstance(args, Mapping) else {}
            actor = str(data.get("actor", "agent"))
            state, active, _done = _TOOL_ACTIVITY.get(
                message,
                ("run", str(message).replace("_", " ").title(), str(message)),
            )
            self._pending_tools[self._pending_key(data)] = {
                "name": message,
                "args": dict(args),
                "actor": actor,
                "done": _done,
            }
            self._live_activity.start(state, active + self._tool_detail(message, args))
            return

        if kind == "tool_result":
            if not self._live_activity.ultra_live:
                self._live_activity.stop()
            one_line = " ".join(str(message).split())
            tool_name = str(data.get("tool") or "operation")
            pending = self._take_pending_tool(data, tool_name)
            tool_name = str(data.get("tool") or pending.get("name") or "operation")
            args = pending.get("args", {})
            if one_line.startswith("Error: invalid plan proposal"):
                self._plan_format_retries += 1
                self._technical_activity.append(one_line[:2_000])
                del self._technical_activity[:-20]
                self._live_activity.start(
                    "warning",
                    f"Correcting the plan format · attempt {self._plan_format_retries}/4",
                )
                return

            failed = one_line.startswith(("Error:", "Permission denied"))
            if failed and self._live_activity.ultra_live:
                self._live_activity.stop()
            if not failed and tool_name in {
                "propose_plan",
                "submit_plan_review",
                "request_plan_input",
            }:
                if tool_name == "propose_plan" and self._plan_format_retries:
                    self._plan_recovered_retries = max(
                        self._plan_recovered_retries,
                        self._plan_format_retries,
                    )
                    self._plan_format_retries = 0
                return
            if failed:
                icon = self._icon("×", "[X]")
                summary = one_line
                self._technical_activity.append(one_line[:2_000])
                color = self.red
                dedupe_key = f"error:{tool_name}:{one_line[:240]}"
            else:
                icon = self._icon("✓", "[x]")
                summary = str(pending.get("done") or _TOOL_ACTIVITY.get(tool_name, ("", "", tool_name))[2])
                summary += self._tool_detail(tool_name, args if isinstance(args, Mapping) else {})
                if tool_name == "list_files" and "(no files under" in one_line:
                    summary = "Inspected workspace · no project files yet"
                color = self.green
                detail = self._tool_detail(
                    tool_name,
                    args if isinstance(args, Mapping) else {},
                )
                dedupe_key = f"success:{tool_name}:{detail}:{summary}"
            if self._live_activity.ultra_live and not failed:
                self._live_activity.update_label(summary)
                return
            self._live_line(
                icon,
                summary,
                color,
                dedupe_key=dedupe_key,
                show_count=tool_name not in {"list_files", "read_file", "grep"},
            )
            return

        if kind == "usage":
            self.context_tokens = max(
                self.context_tokens,
                int(data.get("input_tokens", 0) or 0),
            )
            return

        if kind in {"error", "warning"}:
            self._live_activity.stop()
            if data.get("planning_terminal") or (
                kind == "error"
                and (data.get("attempts") or "plan could not" in str(message).casefold())
            ):
                self._plan_format_retries = 0
                self._plan_recovered_retries = 0
            technical = data.get("technical_detail")
            if technical:
                self._technical_activity.append(str(technical)[:2_000])
            icon = self._icon("×", "[X]") if kind == "error" else self._icon("△", "[!]")
            self._live_line(
                icon,
                message,
                self.red if kind == "error" else self.yellow,
                dedupe_key=f"{kind}:{' '.join(str(message).split())[:240]}",
            )
            return

        if kind == "ultra.agent_started":
            self.active_agents += 1
            role = str(data.get("role", "agent"))
            phase = str(data.get("phase", message or "working"))
            self._live_activity.start("sync", f"{role} · {phase}")
            return

        if kind == "ultra.agent":
            self.active_agents = max(0, self.active_agents - 1)
            self._live_activity.stop()
            self._live_line(self._icon("✓", "[x]"), message, self.green)
            return

        if kind in {"phase", "delegation", "recovery"}:
            state = "sync" if kind in {"delegation", "recovery"} else "plan"
            self._live_activity.start(state, message or kind.replace("_", " ").title())
            return

        if kind == "plan":
            self._live_activity.stop()
            summary = message
            if self._plan_recovered_retries:
                summary = (
                    f"{message} Corrected {self._plan_recovered_retries} internal "
                    "format retries before review."
                )
                self._plan_recovered_retries = 0
            self._live_line(
                self._icon("✓", "[x]"),
                summary,
                self.green,
                dedupe_key=f"plan:{' '.join(str(message).split())}",
            )
            return

        if kind in {"checkpoint", "questions"} or kind.startswith("ultra."):
            self._live_activity.stop()
            if kind in {"checkpoint", "questions"}:
                self._plan_format_retries = 0
                self._plan_recovered_retries = 0
            icon = self._icon("◇", "[>]")
            color = self.gold if kind.startswith("ultra.") else self.cyan
            self._live_line(icon, message or kind, color)
            return

        if message:
            self._live_activity.stop()
            self._live_line(self._icon("◇", "[>]"), message, self.cyan)

    def on_event(self, event: UIEvent) -> None:
        with self._event_lock:
            if self._full_screen_depth:
                self._buffered_events.append(event)
                return
            self._render_event(event)

    def _render_event(self, event: UIEvent) -> None:
        self._last_event_kind = event.kind
        if self.live_activity_enabled or event.kind in {
            "model_text",
            "model_thought",
            "step",
            "tool_call",
            "tool_result",
            "usage",
        }:
            self._on_live_event(event)
            return
        kind, message, data = event.kind, event.message, event.data
        with self._lock:
            if kind in {"model_text", "model_thought"}:
                label = "Response" if kind == "model_text" else "Thinking…"
                color = self.gold if kind == "model_thought" else self.response
                if self._stream_kind != kind:
                    if self._stream_kind:
                        print(self.reset, file=self.stream)
                    print(f"{self.bold}{color}{label}{self.reset}", file=self.stream)
                    self._stream_kind = kind
                # Provider thought streams may contain private scratch work.
                # The UI shows activity, while durable traces keep only the
                # model-authored reasoning summary and explicit insights.
                if kind == "model_text":
                    print(f"{self.response}{message}", end="", file=self.stream, flush=True)
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
        with self._approval_lock:
            return self._confirm_action(name, args, risk)

    def _confirm_action(self, name: str, args: dict, risk: str = "risky") -> bool:
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
        allowed = answer in {"y", "yes"}
        return allowed

    def prompt(self) -> str:
        if not self._background_working:
            self._finish_thought()
            self._live_activity.stop()
        self._last_activity_key = None
        accent = self.gold if self.interaction_mode is InteractionMode.ULTRA else self.green
        label = (
            f"{self.bold}{accent}GA3BAD{self.reset} "
            f"{self.dim}[{self.interaction_mode.value.upper()}]{self.reset}> "
        )
        rich_prompt = (
            not self.plain
            and
            PromptSession is not None
            and SlashCommandCompleter is not None
            and self.input_func is input
            and self.stream is sys.stdout
            and _isatty(sys.stdin)
            and _isatty(self.stream)
        )
        if rich_prompt:
            if self._prompt_session is None:
                bindings = KeyBindings()

                @bindings.add("f2")
                def _open_mode(event) -> None:
                    event.app.exit(result="/mode")

                @bindings.add("f3")
                def _open_model(event) -> None:
                    event.app.exit(result="/model")

                @bindings.add("f4")
                def _open_permissions(event) -> None:
                    event.app.exit(result="/permissions")

                @bindings.add("c-k")
                def _open_commands(event) -> None:
                    event.app.exit(result="/")

                @bindings.add("c-q")
                def _safe_exit(event) -> None:
                    event.app.exit(result="/quit")

                @bindings.add(Keys.BracketedPaste)
                def _collapse_long_paste(event) -> None:
                    pasted = str(event.data or "")
                    if len(pasted) < LONG_PROMPT_RECEIPT_CHARS:
                        event.current_buffer.insert_text(pasted)
                        return
                    token = f"[Pasted Content {len(pasted)} chars]"
                    suffix = 2
                    unique = token
                    while unique in self._pasted_content:
                        unique = token[:-1] + f" #{suffix}]"
                        suffix += 1
                    self._pasted_content[unique] = pasted
                    event.current_buffer.insert_text(unique)

                self._prompt_bindings = bindings
                self._prompt_session = PromptSession(
                    completer=SlashCommandCompleter(lambda: self._current_status),
                    complete_while_typing=True,
                    key_bindings=bindings,
                    vi_mode=self.vim_mode,
                    style=self._prompt_style(),
                    # prompt_toolkit exposes this as a PromptSession setting,
                    # not a per-call PromptSession.prompt() argument.
                    erase_when_done=True,
                )
            def footer() -> str:
                columns = shutil.get_terminal_size((112, 30)).columns
                if columns < 48:
                    return f"{self.active_model} {self.reasoning_effort}"
                if columns < 78:
                    return f"{self.active_model} {self.reasoning_effort} \u00b7 / commands"
                location = f" \u00b7 {self.workspace_label}" if self.workspace_label else ""
                return f"{self.active_model} {self.reasoning_effort}{location}"
            def styled_footer():
                columns = shutil.get_terminal_size((112, 30)).columns
                process_mib, ram_percent = _memory_snapshot()
                context = f"ctx {self.context_tokens / 1000:.1f}k" if self.context_tokens else "ctx 0"
                fragments = [
                    ("class:footer.model", self.active_model),
                    ("class:footer.separator", "  "),
                    ("class:footer.effort", self.reasoning_effort),
                ]
                if columns >= 70:
                    memory = f"{context}  proc {process_mib:.0f}MB  RAM {ram_percent:.0f}%"
                    fragments.extend((("class:footer.separator", "  \u00b7  "), ("class:footer.memory", memory)))
                if columns >= 110 and self.workspace_label:
                    fragments.extend((("class:footer.separator", "  \u00b7  "), ("class:footer.path", self.workspace_label)))
                elif columns >= 48:
                    fragments.append(("class:footer.separator", "  \u00b7  / commands"))
                return fragments
            default = self._prompt_default
            self._prompt_default = ""
            rich_label = [("class:prompt", self._prompt_mark() + " ")]
            placeholder = [("class:placeholder", "Use /skills to list available skills")]
            self._pasted_content = {}
            self._composer_active = True
            try:
                if patch_stdout is not None:
                    with patch_stdout(raw=True):
                        value = self._prompt_session.prompt(
                            rich_label,
                            default=default,
                            bottom_toolbar=styled_footer if self.color else footer,
                            complete_style=CompleteStyle.COLUMN,
                            placeholder=placeholder,
                            reserve_space_for_menu=12,
                        )
                else:
                    value = self._prompt_session.prompt(
                        rich_label,
                        default=default,
                        bottom_toolbar=styled_footer if self.color else footer,
                        complete_style=CompleteStyle.COLUMN,
                        placeholder=placeholder,
                    )
            finally:
                self._composer_active = False
            for token, pasted in self._pasted_content.items():
                value = value.replace(token, pasted)
            receipt = prompt_receipt(value)
            with self._lock:
                receipt_color = self.cyan if receipt != value else ""
                print(
                    f"{self.bold}{accent}{self._prompt_mark()}{self.reset} "
                    f"{receipt_color}{receipt}{self.reset}",
                    file=self.stream,
                    flush=True,
                )
            return value
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
