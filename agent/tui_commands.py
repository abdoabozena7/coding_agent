"""Shared command metadata for every terminal presentation.

The parser remains the execution authority.  This module only describes how
commands are discovered and whether a persistent workspace may satisfy them
from an immutable/durable snapshot while another operation is running.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class CommandSpec:
    name: str
    description: str
    category: str
    aliases: tuple[str, ...] = ()
    arguments: str = ""
    live_safe: bool = False
    checkpoint_required: bool = False

    @property
    def search_text(self) -> str:
        return " ".join(
            (self.name, *self.aliases, self.description, self.category, self.arguments)
        ).casefold()


@dataclass(frozen=True, slots=True)
class CommandAvailability:
    visible: bool = True
    enabled: bool = True
    reason: str = ""


COMMAND_SPECS: tuple[CommandSpec, ...] = (
    CommandSpec("/status", "show the latest project snapshot", "Inspect", live_safe=True),
    CommandSpec("/agents", "inspect live specialists", "Inspect", arguments="[--all|AGENT]", live_safe=True),
    CommandSpec("/tree", "inspect the specialist hierarchy", "Inspect", arguments="[NODE]", live_safe=True),
    CommandSpec("/agent", "open one specialist", "Inspect", arguments="[AGENT]", live_safe=True),
    CommandSpec("/thinking", "show or hide redacted reasoning summaries", "Inspect", aliases=("/reasoning",), arguments="[show|hide|status]", live_safe=True),
    CommandSpec("/details", "open the latest collapsed item or diagnostic", "Inspect", arguments="[ID]", live_safe=True),
    CommandSpec("/plan", "enter Plan mode or review/edit the durable plan", "Inspect", arguments="[edit]", live_safe=True),
    CommandSpec("/chat", "open the durable workspace conversation", "Inspect", live_safe=True),
    CommandSpec("/diff", "show current changes or one checkpoint diff", "Inspect", arguments="[CHECKPOINT]", live_safe=True),
    CommandSpec("/trace", "inspect redacted prompts and run trace", "Inspect", arguments="[TARGET]", live_safe=True),
    CommandSpec("/history", "show durable activity", "Inspect", live_safe=True),
    CommandSpec("/versions", "show accepted project checkpoints", "Inspect", live_safe=True),
    CommandSpec("/memory", "inspect the Project Brain", "Inspect", arguments="[QUERY]", live_safe=True),
    CommandSpec("/insights", "show durable findings and decisions", "Inspect", arguments="[NODE]", live_safe=True),
    CommandSpec("/metrics", "show quality, usage, and timing metrics", "Inspect", live_safe=True),
    CommandSpec("/questions", "show pending intake or plan questions", "Inspect", live_safe=True),
    CommandSpec("/processes", "list managed processes and previews", "Inspect", live_safe=True),
    CommandSpec("/help", "show command help", "Help", arguments="[TOPIC]", live_safe=True),
    CommandSpec("/keymap", "show predictable keyboard controls", "Help", live_safe=True),
    CommandSpec("/pause", "request a cooperative checkpoint", "Workflow", live_safe=True),
    CommandSpec("/resume", "continue a valid paused state", "Workflow", aliases=("/continue",), checkpoint_required=True),
    CommandSpec("/resolve", "reconcile uncertain crash-window work", "Workflow", arguments="ACTION_ID applied|not-run NOTE", checkpoint_required=True),
    CommandSpec("/goal", "start a durable goal", "Workflow", arguments="OBJECTIVE", checkpoint_required=True),
    CommandSpec("/approve", "approve the displayed plan revision", "Workflow", arguments="[REVISION]", checkpoint_required=True),
    CommandSpec("/reject", "reject the draft plan with feedback", "Workflow", arguments="FEEDBACK", checkpoint_required=True),
    CommandSpec("/replan", "request a revised plan", "Workflow", arguments="FEEDBACK", checkpoint_required=True),
    CommandSpec("/run", "run one bounded work slice", "Workflow", arguments="[STEPS]", checkpoint_required=True),
    CommandSpec("/auto", "continue until the next checkpoint", "Workflow", checkpoint_required=True),
    CommandSpec("/cancel", "explicitly abandon the active goal", "Workflow", checkpoint_required=True),
    CommandSpec("/mode", "switch PLAN / NORMAL / ULTRA orchestration", "Session", arguments="plan|normal|ultra", checkpoint_required=True),
    CommandSpec("/model", "choose model and reasoning effort", "Session", arguments="[MODEL] [EFFORT]", checkpoint_required=True),
    CommandSpec("/permissions", "choose NORMAL / FULL access", "Session", aliases=("/permission", "/access"), arguments="[normal|full]", checkpoint_required=True),
    CommandSpec("/sleep", "control safe unattended choices", "Session", arguments="[on|off|status]", live_safe=True),
    CommandSpec("/settings", "inspect or change session settings", "Session", arguments="[KEY VALUE]", checkpoint_required=True),
    CommandSpec("/doctor", "audit local model readiness", "Session", arguments="[--live] [--record]", checkpoint_required=True),
    CommandSpec("/skills", "show local tools and approval policy", "Session", live_safe=True),
    CommandSpec("/setup", "validate one-time sandbox setup", "Session", checkpoint_required=True),
    CommandSpec("/stop-process", "stop a managed process", "Session", arguments="ID", checkpoint_required=True),
    CommandSpec("/explorer", "open the selected project folder", "Session", aliases=("/open-folder",), live_safe=True),
    CommandSpec("/undo", "revert accepted checkpoints", "Session", arguments="[STEPS]", checkpoint_required=True),
    CommandSpec("/answer", "answer the current question", "Workflow", arguments="[QUESTION_ID] VALUE", checkpoint_required=True),
    CommandSpec("/add", "add a checklist task", "Tasks", arguments="TEXT", checkpoint_required=True),
    CommandSpec("/edit", "edit a checklist task", "Tasks", arguments="TASK_ID VALUE", checkpoint_required=True),
    CommandSpec("/remove", "remove a checklist task", "Tasks", arguments="TASK_ID", checkpoint_required=True),
    CommandSpec("/done", "mark a task done with evidence", "Tasks", arguments="TASK_ID [NOTE]", checkpoint_required=True),
    CommandSpec("/todo", "reopen a checklist task", "Tasks", arguments="TASK_ID [NOTE]", checkpoint_required=True),
    CommandSpec("/block", "mark a checklist task blocked", "Tasks", arguments="TASK_ID [NOTE]", checkpoint_required=True),
    CommandSpec("/skip", "skip a checklist task", "Tasks", arguments="TASK_ID [NOTE]", checkpoint_required=True),
    CommandSpec("/quit", "checkpoint and leave the session", "Help", aliases=("/q",)),
)


ALL_SLASH_COMMANDS: tuple[tuple[str, str], ...] = tuple(
    (spec.name, spec.description) for spec in COMMAND_SPECS
)

# Compatibility exports retained for the plain slash menu and embedders.
_CODEX_NAMES = {"/model", "/permissions", "/keymap", "/approve"}
CODEX_SLASH_COMMANDS: tuple[tuple[str, str], ...] = tuple(
    item for item in ALL_SLASH_COMMANDS if item[0] in _CODEX_NAMES
)
SLASH_COMMANDS: tuple[tuple[str, str], ...] = tuple(
    item for item in ALL_SLASH_COMMANDS if item[0] not in _CODEX_NAMES
)

COMMAND_GROUPS: tuple[tuple[str, str, tuple[str, ...]], ...] = tuple(
    (
        category,
        {
            "Workflow": "Start, approve, pause, or resume the current goal",
            "Tasks": "Edit the accepted checklist and record outcomes",
            "Inspect": "Open compact read-only project details",
            "Session": "Change runtime and terminal settings",
            "Help": "Show guidance or leave safely",
        }[category],
        tuple(spec.name for spec in COMMAND_SPECS if spec.category == category),
    )
    for category in ("Workflow", "Tasks", "Inspect", "Session", "Help")
)

_DESCRIPTIONS = dict(ALL_SLASH_COMMANDS)
_CONTEXT_COMMANDS: dict[str, tuple[str, ...]] = {
    "idle": ("/goal", "/mode", "/model", "/settings"),
    "new": ("/goal", "/mode", "/model", "/settings"),
    "discovering": ("/status", "/pause", "/agents", "/thinking"),
    "revising": ("/status", "/pause", "/plan", "/thinking"),
    "awaiting_plan_approval": ("/approve", "/reject", "/plan", "/questions"),
    "running": ("/status", "/agents", "/tree", "/pause", "/thinking"),
    "paused": ("/resume", "/status", "/resolve", "/trace"),
    "recovering": ("/status", "/trace", "/details", "/help"),
    "reviewing": ("/status", "/agents", "/trace", "/pause"),
    "verifying": ("/status", "/agents", "/trace", "/pause"),
}


def contextual_commands(status: str) -> tuple[tuple[str, str], ...]:
    names = _CONTEXT_COMMANDS.get(
        str(status).strip().lower(), ("/status", "/help", "/quit")
    )
    return tuple((name, _DESCRIPTIONS[name]) for name in names if name in _DESCRIPTIONS)


def command_availability(spec: CommandSpec, snapshot: Any | None) -> CommandAvailability:
    if snapshot is None:
        return CommandAvailability()
    status = str(getattr(snapshot, "status", "")).casefold()
    running = bool(getattr(snapshot, "running", False))
    if spec.name == "/resume" and status != "paused":
        return CommandAvailability(False, False, "Available only while paused")
    if spec.name == "/pause" and not running:
        return CommandAvailability(False, False, "Available while work is running")
    if spec.name == "/undo" and not bool(getattr(snapshot, "undo_available", False)):
        return CommandAvailability(False, False, "Available after a completed checkpoint")
    return CommandAvailability()


def matching_commands(
    query: str,
    *,
    limit: int = 9,
    snapshot: Any | None = None,
) -> tuple[CommandSpec, ...]:
    needle = str(query).strip().casefold()
    if needle and not needle.startswith("/"):
        needle = "/" + needle
    matches = tuple(
        spec
        for spec in COMMAND_SPECS
        if command_availability(spec, snapshot).visible
        and (not needle or spec.name.startswith(needle) or needle in spec.search_text)
    )
    return matches[: max(1, int(limit))]
