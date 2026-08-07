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
    CommandSpec("/activity", "open the verified live workflow timeline", "Inspect", live_safe=True),
    CommandSpec("/agents", "open Agent Tree in the local web workspace", "Inspect", live_safe=True),
    CommandSpec("/execution", "open the live execution tree in the local web workspace", "Inspect", aliases=("/tree",), live_safe=True),
    CommandSpec("/thinking", "show or hide redacted reasoning summaries", "Inspect", aliases=("/reasoning",), arguments="[show|hide|status]", live_safe=True),
    CommandSpec("/details", "open the latest collapsed item or diagnostic", "Inspect", arguments="[ID]", live_safe=True),
    CommandSpec("/queue", "inspect durable prompts waiting behind current work", "Inspect", live_safe=True),
    CommandSpec("/project-brain", "inspect project-scoped memory and current context", "Inspect", aliases=("/memory",), arguments="[QUERY]", live_safe=True),
    CommandSpec(
        "/sessions",
        "inspect sessions or resume one at its checkpoint",
        "Inspect",
        arguments="[SESSION_ID]",
        live_safe=True,
    ),
    CommandSpec("/plan", "open Plan Workspace; create or review a plan", "Inspect", live_safe=True),
    CommandSpec("/review", "open Change Review in the local web workspace", "Inspect", live_safe=True),
    CommandSpec("/chat", "open the durable workspace conversation", "Inspect", live_safe=True),
    CommandSpec("/trace", "inspect redacted prompts and run trace", "Inspect", arguments="[TARGET]", live_safe=True),
    CommandSpec("/history", "show durable activity", "Inspect", live_safe=True),
    CommandSpec("/diff", "open the latest recorded diff in Change Review", "Inspect", live_safe=True),
    CommandSpec("/versions", "show accepted project checkpoints", "Inspect", live_safe=True),
    CommandSpec("/memory", "inspect the Project Brain", "Inspect", arguments="[QUERY]", live_safe=True),
    CommandSpec("/insights", "show durable findings and decisions", "Inspect", arguments="[NODE]", live_safe=True),
    CommandSpec("/metrics", "show quality, usage, and timing metrics", "Inspect", live_safe=True),
    CommandSpec("/questions", "show pending intake or plan questions", "Inspect", live_safe=True),
    CommandSpec("/processes", "list managed processes and previews", "Inspect", live_safe=True),
    CommandSpec("/help", "show command help", "Help", arguments="[TOPIC]", live_safe=True),
    CommandSpec("/keymap", "show predictable keyboard controls", "Help", live_safe=True),
    CommandSpec("/pause", "request a cooperative checkpoint", "Workflow", live_safe=True),
    CommandSpec("/stop", "stop the active provider/tool and save a resumable checkpoint", "Workflow", arguments="[ollama]", checkpoint_required=True),
    CommandSpec("/enqueue", "queue an immutable prompt with its selected mode", "Workflow", arguments="plan|normal|ultra TEXT", live_safe=True),
    CommandSpec("/guide", "add guidance to the currently active goal", "Workflow", arguments="TEXT", live_safe=True),
    CommandSpec("/resume", "continue a valid paused state", "Workflow", aliases=("/continue",), checkpoint_required=True),
    CommandSpec("/resolve", "reconcile uncertain crash-window work", "Workflow", arguments="ACTION_ID applied|not-run NOTE", checkpoint_required=True),
    CommandSpec("/goal", "start a durable goal", "Workflow", arguments="OBJECTIVE", checkpoint_required=True),
    CommandSpec("/run", "run one bounded work slice", "Workflow", arguments="[STEPS]", checkpoint_required=True),
    CommandSpec("/auto", "continue until the next checkpoint", "Workflow", checkpoint_required=True),
    CommandSpec("/cancel", "explicitly abandon the active goal", "Workflow", checkpoint_required=True),
    CommandSpec("/mode", "legacy alias for choosing Working or Plan", "Session", arguments="[normal|plan]", checkpoint_required=True),
    CommandSpec("/model", "choose model and reasoning effort", "Session", arguments="[MODEL] [EFFORT]", checkpoint_required=True),
    CommandSpec("/permissions", "choose NORMAL / FULL access", "Session", aliases=("/permission", "/access"), arguments="[normal|full]", checkpoint_required=True),
    CommandSpec("/sleep", "control unattended approvals", "Session", arguments="[on|safe|full|off|status]", live_safe=True),
    # Argument-level entries keep the most important unattended controls
    # discoverable from the palette instead of requiring the user to remember
    # the second token while work is active.
    CommandSpec("/sleep on", "turn on safe unattended approvals", "Session", live_safe=True),
    CommandSpec("/sleep safe", "turn on safe unattended approvals", "Session", live_safe=True),
    CommandSpec("/sleep full", "approve reviewed plans and every tool action in this workspace", "Session", live_safe=True),
    CommandSpec("/sleep off", "turn off unattended approvals", "Session", live_safe=True),
    CommandSpec("/sleep status", "show the current Sleep Mode policy", "Session", live_safe=True),
    CommandSpec("/settings", "inspect settings or add a provider API key", "Session", arguments="[api-key PROVIDER|KEY VALUE]", live_safe=True),
    CommandSpec("/settings reconfigure", "reopen the saved project profile and change it at a safe checkpoint", "Session", live_safe=True),
    CommandSpec("/api-key", "securely add or replace a provider API key", "Session", aliases=("/apikey", "/add-api-key"), arguments="[openai|gemini|ollama]", live_safe=True),
    CommandSpec("/doctor", "audit local model readiness", "Session", arguments="[--live] [--record]", checkpoint_required=True),
    CommandSpec("/skills", "show local tools and approval policy", "Session", live_safe=True),
    CommandSpec("/setup", "validate one-time sandbox setup", "Session", checkpoint_required=True),
    CommandSpec("/setup project", "reconfigure the saved project profile", "Session", checkpoint_required=True),
    CommandSpec("/stop-process", "stop a managed process", "Session", arguments="ID", checkpoint_required=True),
    CommandSpec("/explorer", "open the selected project folder", "Session", aliases=("/open-folder",), live_safe=True),
    CommandSpec("/open-web", "open the primary Workspace control surface", "Inspect", live_safe=True),
    CommandSpec("/undo", "revert accepted checkpoints", "Session", arguments="[STEPS]", checkpoint_required=True),
    CommandSpec("/answer", "answer the current question", "Workflow", arguments="[QUESTION_ID] VALUE", checkpoint_required=True),
    CommandSpec("/done", "mark a task done with evidence", "Tasks", arguments="TASK_ID [NOTE]", checkpoint_required=True),
    CommandSpec("/todo", "reopen a checklist task", "Tasks", arguments="TASK_ID [NOTE]", checkpoint_required=True),
    CommandSpec("/block", "mark a checklist task blocked", "Tasks", arguments="TASK_ID [NOTE]", checkpoint_required=True),
    CommandSpec("/skip", "skip a checklist task", "Tasks", arguments="TASK_ID [NOTE]", checkpoint_required=True),
    CommandSpec("/quit", "checkpoint and leave the session", "Help", aliases=("/q",)),
)


_HIDDEN_COMPAT_NAMES = {"/mode", "/goal"}
ALL_SLASH_COMMANDS: tuple[tuple[str, str], ...] = tuple(
    (spec.name, spec.description)
    for spec in COMMAND_SPECS
    if spec.name not in _HIDDEN_COMPAT_NAMES
)

# Compatibility exports retained for the plain slash menu and embedders.
_CODEX_NAMES = {"/model", "/permissions", "/keymap"}
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
_LIVE_SESSION_COMMANDS = (
    "/activity", "/status", "/sleep on", "/sleep safe", "/sleep full", "/sleep off", "/sleep status",
    "/stop", "/queue", "/execution", "/review", "/history", "/diff", "/open-web", "/pause",
)
_CONTEXT_COMMANDS: dict[str, tuple[str, ...]] = {
    "idle": ("/plan", "/model", "/settings", "/help", "/history"),
    "new": ("/plan", "/model", "/settings", "/help", "/history"),
    "discovering": _LIVE_SESSION_COMMANDS,
    "routing": _LIVE_SESSION_COMMANDS,
    "planning": _LIVE_SESSION_COMMANDS,
    "revising": _LIVE_SESSION_COMMANDS,
    "awaiting_plan_approval": ("/plan", "/questions", "/sleep on", "/sleep full", "/sleep off", "/sleep status", "/status"),
    "waiting_for_approval": ("/activity", "/status", "/sleep on", "/sleep full", "/sleep off", "/sleep status", "/queue"),
    "waiting_for_process": _LIVE_SESSION_COMMANDS,
    "working": _LIVE_SESSION_COMMANDS,
    "running": _LIVE_SESSION_COMMANDS,
    "retrying": _LIVE_SESSION_COMMANDS,
    "recovering": _LIVE_SESSION_COMMANDS,
    "paused": ("/resume", "/status", "/resolve", "/trace", "/activity", "/sleep on", "/sleep full", "/sleep off", "/sleep status"),
    "reviewing": _LIVE_SESSION_COMMANDS,
    "verifying": _LIVE_SESSION_COMMANDS,
}


def contextual_commands(status: str) -> tuple[tuple[str, str], ...]:
    names = _CONTEXT_COMMANDS.get(
        str(status).strip().lower(), ("/status", "/help", "/quit")
    )
    return tuple((name, _DESCRIPTIONS[name]) for name in names if name in _DESCRIPTIONS)


def command_availability(spec: CommandSpec, snapshot: Any | None) -> CommandAvailability:
    if spec.name in {"/mode", "/goal"}:
        # Runtime/parser support remains for automation and old sessions, but
        # the interactive product no longer asks users to classify requests.
        return CommandAvailability(False, False, "Selected automatically after the prompt")
    if snapshot is None:
        return CommandAvailability()
    status = str(getattr(snapshot, "status", "")).casefold()
    running = bool(getattr(snapshot, "running", False))
    if spec.name == "/resume" and status != "paused":
        return CommandAvailability(False, False, "Available only while paused")
    if spec.name == "/pause" and not running:
        return CommandAvailability(False, False, "Available while work is running")
    if spec.name == "/stop" and not running and status not in {"routing", "planning", "working", "retrying", "paused"}:
        return CommandAvailability(False, False, "Available while a provider, tool, or process is active")
    if spec.name == "/undo" and not bool(getattr(snapshot, "undo_available", False)):
        return CommandAvailability(False, False, "Available after a completed checkpoint")
    if spec.name == "/mode" and status not in {"idle", "completed", "cancelled"}:
        return CommandAvailability(
            True,
            False,
            "Mode is locked until this workflow completes or is cancelled",
        )
    return CommandAvailability()


def matching_commands(
    query: str,
    *,
    limit: int | None = None,
    snapshot: Any | None = None,
) -> tuple[CommandSpec, ...]:
    needle = str(query).strip().casefold()
    if needle and not needle.startswith("/"):
        needle = "/" + needle
    if needle == "/":
        needle = ""
    matches = tuple(
        spec
        for spec in COMMAND_SPECS
        if command_availability(spec, snapshot).visible
        and (not needle or spec.name.startswith(needle) or needle in spec.search_text)
    )
    if not needle and snapshot is not None:
        status = str(getattr(snapshot, "status", "idle")).casefold()
        preferred = dict.fromkeys(_CONTEXT_COMMANDS.get(status, _CONTEXT_COMMANDS["idle"]))
        matches = tuple(sorted(
            matches,
            key=lambda spec: (
                0 if spec.name in preferred else 1,
                tuple(preferred).index(spec.name) if spec.name in preferred else 0,
                spec.category,
                spec.name,
            ),
        ))
    if limit is None:
        return matches
    return matches[: max(1, int(limit))]
