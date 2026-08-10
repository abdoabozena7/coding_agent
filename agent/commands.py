"""The deliberately small public slash-command language.

Only user-facing navigation and checkpoint controls belong here.  Attention
cards, settings rows, and recovery controls dispatch typed actions directly;
they are not compatibility slash commands.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CommandKind(str, Enum):
    TEXT = "text"
    MENU = "menu"
    PLAN = "plan"
    LIVE = "live"
    SHOW_DIFF = "show_diff"
    ADVANCED_TRACING = "advanced_tracing"
    SETTINGS = "settings"
    PAUSE = "pause"
    RESUME = "resume"
    STOP = "stop"
    UNDO = "undo"
    HELP = "help"
    QUIT = "quit"


class InternalActionKind(str, Enum):
    """Typed TUI actions which must never be parsed from user slash text.

    This is the migration boundary for existing attention cards and settings
    controls.  Keeping it separate makes the public command surface auditable.
    """

    MODE = "mode"
    SETTINGS_UPDATE = "settings_update"
    MODEL = "model"
    PERMISSIONS = "permissions"
    KEYMAP = "keymap"
    DOCTOR = "doctor"
    SKILLS = "skills"
    PROCESSES = "processes"
    STOP_PROCESS = "stop_process"
    TREE = "tree"
    AGENTS = "agents"
    AGENT = "agent"
    MEMORY = "memory"
    TRACE = "trace"
    THINKING = "thinking"
    DETAILS = "details"
    ACTIVITY = "activity"
    QUEUE = "queue"
    ENQUEUE = "enqueue"
    GUIDE = "guide"
    PROJECT_BRAIN = "project_brain"
    EFFECTIVE_PLAN = "effective_plan"
    ULTRA_DETAILS = "ultra_details"
    SESSIONS = "sessions"
    INSIGHTS = "insights"
    QUESTIONS = "questions"
    ANSWER = "answer"
    METRICS = "metrics"
    SETUP = "setup"
    SLEEP = "sleep"
    GOAL = "goal"
    APPROVE = "approve"
    REJECT = "reject"
    REPLAN = "replan"
    REVIEW = "review"
    CHAT = "chat"
    EXPLORER = "explorer"
    OPEN_WEB = "open_web"
    ADD = "add"
    EDIT = "edit"
    REMOVE = "remove"
    TASK_STATUS = "task_status"
    RUN = "run"
    AUTO = "auto"
    STATUS = "status"
    HISTORY = "history"
    DIFF = "diff"
    VERSIONS = "versions"
    RESOLVE = "resolve"
    CANCEL = "cancel"


@dataclass(frozen=True)
class UserCommand:
    kind: CommandKind | InternalActionKind
    args: dict[str, Any] = field(default_factory=dict)
    raw: str = ""


class CommandParseError(ValueError):
    pass


class UnknownCommandParseError(CommandParseError):
    """A slash prefix which is not part of the small public surface."""


def internal_action(
    kind: InternalActionKind,
    /,
    **args: Any,
) -> UserCommand:
    """Build a typed internal action without routing through slash parsing."""

    return UserCommand(kind=kind, args=dict(args), raw="")


def parse_command(line: str) -> UserCommand:
    """Parse exactly the supported public command language."""

    raw = line.rstrip("\r\n")
    stripped = raw.strip()
    if not stripped:
        return UserCommand(CommandKind.TEXT, {"text": ""}, raw)
    if not stripped.startswith("/"):
        return UserCommand(CommandKind.TEXT, {"text": stripped}, raw)

    body = stripped[1:].strip()
    if not body:
        return UserCommand(CommandKind.MENU, raw=raw)
    parts = body.split(maxsplit=1)
    name = parts[0].casefold()
    rest = parts[1].strip() if len(parts) == 2 else ""

    no_argument_commands = {
        "plan": CommandKind.PLAN,
        "live": CommandKind.LIVE,
        "show-diff": CommandKind.SHOW_DIFF,
        "advanced-tracing": CommandKind.ADVANCED_TRACING,
        "settings": CommandKind.SETTINGS,
        "pause": CommandKind.PAUSE,
        "resume": CommandKind.RESUME,
        "stop": CommandKind.STOP,
        "help": CommandKind.HELP,
        "quit": CommandKind.QUIT,
    }
    if name in no_argument_commands:
        if rest:
            raise CommandParseError(f"/{name} does not take arguments.")
        args = {"key": None, "value": None} if name == "settings" else {}
        return UserCommand(no_argument_commands[name], args, raw)

    if name == "undo":
        steps = 1
        if rest:
            try:
                steps = int(rest)
            except ValueError as exc:
                raise CommandParseError("Usage: /undo [STEPS]") from exc
            if steps < 1:
                raise CommandParseError("Undo steps must be a positive integer.")
        return UserCommand(CommandKind.UNDO, {"steps": steps}, raw)

    raise UnknownCommandParseError(
        "Unknown slash command. Use /help to see the supported commands."
    )
