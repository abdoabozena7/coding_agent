"""Parsing for the interactive agent command language.

Commands deliberately remain plain text so the same controls work in a basic
terminal, over SSH, and in richer frontends. Both ``:`` and ``/`` prefixes are
accepted; unprefixed text is treated as a goal or guidance by the runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CommandKind(str, Enum):
    TEXT = "text"
    MENU = "menu"
    MODE = "mode"
    SETTINGS = "settings"
    MODEL = "model"
    PERMISSIONS = "permissions"
    IDE = "ide"
    KEYMAP = "keymap"
    VIM = "vim"
    SANDBOX_ADD_READ_DIR = "sandbox_add_read_dir"
    EXPERIMENTAL = "experimental"
    DOCTOR = "doctor"
    SKILLS = "skills"
    PROCESSES = "processes"
    STOP_PROCESS = "stop_process"
    TREE = "tree"
    AGENTS = "agents"
    MEMORY = "memory"
    TRACE = "trace"
    THINKING = "thinking"
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
    PLAN = "plan"
    ADD = "add"
    EDIT = "edit"
    REMOVE = "remove"
    TASK_STATUS = "task_status"
    RUN = "run"
    AUTO = "auto"
    PAUSE = "pause"
    RESUME = "resume"
    STATUS = "status"
    HISTORY = "history"
    RESOLVE = "resolve"
    CANCEL = "cancel"
    HELP = "help"
    QUIT = "quit"


@dataclass(frozen=True)
class UserCommand:
    kind: CommandKind
    args: dict[str, Any] = field(default_factory=dict)
    raw: str = ""


class CommandParseError(ValueError):
    pass


def _required(text: str, usage: str) -> str:
    value = text.strip()
    if not value:
        raise CommandParseError(f"Missing value. Usage: {usage}")
    return value


def _task_and_text(rest: str, usage: str) -> tuple[str, str]:
    parts = rest.strip().split(maxsplit=1)
    if len(parts) != 2:
        raise CommandParseError(f"Usage: {usage}")
    return parts[0].upper(), parts[1].strip()


def _text_and_criteria(rest: str) -> tuple[str, str]:
    # A lightweight delimiter avoids fragile shell-style quoting for long prose.
    text, separator, criteria = rest.partition("::")
    return _required(text, ":add TEXT [:: ACCEPTANCE CRITERIA]"), criteria.strip()


def parse_command(line: str) -> UserCommand:
    raw = line.rstrip("\r\n")
    stripped = raw.strip()
    if not stripped:
        return UserCommand(CommandKind.TEXT, {"text": ""}, raw)
    if stripped[0] not in {":", "/"}:
        if stripped.lower() in {"exit", "quit"}:
            return UserCommand(CommandKind.QUIT, raw=raw)
        return UserCommand(CommandKind.TEXT, {"text": stripped}, raw)

    prefix = stripped[0]
    body = stripped[1:].strip()
    if not body:
        return UserCommand(CommandKind.MENU, raw=raw)
    command_parts = body.split(maxsplit=1)
    name = command_parts[0].lower()
    rest = command_parts[1].strip() if len(command_parts) == 2 else ""

    aliases = {
        "exit": "quit",
        "q": "quit",
        "ls": "plan",
        "list": "plan",
        "continue": "run",
        "go": "run",
        "yes": "approve",
    }
    name = aliases.get(name, name)

    def usage(command: str, suffix: str = "") -> str:
        return f"{prefix}{command}{(' ' + suffix) if suffix else ''}"

    if name == "mode":
        if not rest:
            return UserCommand(CommandKind.MODE, {"mode": None}, raw)
        mode = rest.lower()
        mode_aliases = {
            "manual": "normal",
            "default": "normal",
            "auto": "normal",
            "agent": "normal",
            "chat": "normal",
            "plan": "normal",
            "goal": "normal",
            "deep": "ultra",
            "max": "ultra",
        }
        mode = mode_aliases.get(mode, mode)
        if mode not in {"normal", "ultra"}:
            raise CommandParseError(f"Usage: {usage('mode', 'normal|ultra')}")
        return UserCommand(CommandKind.MODE, {"mode": mode}, raw)
    if name == "settings":
        if not rest:
            return UserCommand(CommandKind.SETTINGS, {"key": None, "value": None}, raw)
        setting_parts = rest.split(maxsplit=1)
        key = setting_parts[0]
        value = setting_parts[1] if len(setting_parts) == 2 else ""
        key = key.lower().replace("-", "_")
        return UserCommand(
            CommandKind.SETTINGS,
            {"key": key, "value": value.strip() or None},
            raw,
        )
    if name == "model":
        parts = rest.split()
        efforts = {"low", "medium", "high", "xhigh"}
        effort = parts[-1].lower() if parts and parts[-1].lower() in efforts else None
        if effort:
            parts.pop()
        return UserCommand(CommandKind.MODEL, {"model": " ".join(parts) or None, "effort": effort}, raw)
    if name in {"permissions", "permission", "access"}:
        level = rest.lower() or None
        if level not in {None, "normal", "full"}:
            raise CommandParseError(f"Usage: {usage('permissions', 'normal|full')}")
        return UserCommand(CommandKind.PERMISSIONS, {"level": level}, raw)
    if name == "ide":
        if rest:
            raise CommandParseError(f"{prefix}ide does not take arguments.")
        return UserCommand(CommandKind.IDE, raw=raw)
    if name == "keymap":
        if rest:
            raise CommandParseError(f"{prefix}keymap does not take arguments.")
        return UserCommand(CommandKind.KEYMAP, raw=raw)
    if name == "vim":
        if rest and rest.lower() not in {"on", "off"}:
            raise CommandParseError(f"Usage: {usage('vim', '[on|off]')}")
        return UserCommand(CommandKind.VIM, {"state": rest.lower() or None}, raw=raw)
    if name == "sandbox-add-read-dir":
        return UserCommand(CommandKind.SANDBOX_ADD_READ_DIR, {"path": _required(rest, usage("sandbox-add-read-dir", "ABSOLUTE_PATH"))}, raw=raw)
    if name == "experimental":
        if rest and rest.lower() not in {"on", "off", "status"}:
            raise CommandParseError(f"Usage: {usage('experimental', '[on|off|status]')}")
        return UserCommand(CommandKind.EXPERIMENTAL, {"state": rest.lower() or "status"}, raw=raw)
    if name in {"doctor", "readiness"}:
        args: dict[str, Any] = {}
        for token in rest.lower().split():
            if token in {"--live", "live"}:
                args["live"] = True
            elif token in {"--record", "record"}:
                args["record"] = True
            else:
                raise CommandParseError(f"Usage: {usage(name, '[--live] [--record]')}")
        return UserCommand(CommandKind.DOCTOR, args, raw=raw)
    if name == "skills":
        if rest:
            raise CommandParseError(f"{prefix}skills does not take arguments.")
        return UserCommand(CommandKind.SKILLS, raw=raw)
    if name == "processes":
        if rest:
            raise CommandParseError(f"{prefix}processes does not take arguments.")
        return UserCommand(CommandKind.PROCESSES, raw=raw)
    if name == "stop-process":
        return UserCommand(
            CommandKind.STOP_PROCESS,
            {"resource_id": _required(rest, usage("stop-process", "ID"))},
            raw=raw,
        )
    if name in {"tree", "memory", "trace", "insights"}:
        kind = {
            "tree": CommandKind.TREE,
            "memory": CommandKind.MEMORY,
            "trace": CommandKind.TRACE,
            "insights": CommandKind.INSIGHTS,
        }[name]
        return UserCommand(kind, {"target": rest or None}, raw)
    if name == "thinking":
        if rest:
            raise CommandParseError(f"{prefix}thinking does not take arguments.")
        return UserCommand(CommandKind.THINKING, raw=raw)
    if name == "agents":
        if rest not in {"", "--all", "all"}:
            raise CommandParseError(f"Usage: {usage('agents', '[--all]')}")
        return UserCommand(CommandKind.AGENTS, {"all": bool(rest)}, raw)
    if name in {"questions", "metrics", "setup"}:
        if rest:
            raise CommandParseError(f"{prefix}{name} does not take arguments.")
        return UserCommand(CommandKind(name), raw=raw)
    if name == "sleep":
        action = rest.lower() or "status"
        if action not in {"on", "off", "status"}:
            raise CommandParseError(f"Usage: {usage('sleep', 'on|off|status')}")
        return UserCommand(CommandKind.SLEEP, {"action": action}, raw)
    if name == "answer":
        parts = rest.split(maxsplit=1)
        if len(parts) != 2:
            raise CommandParseError(f"Usage: {usage('answer', 'QUESTION_ID VALUE')}")
        return UserCommand(CommandKind.ANSWER, {"question_id": parts[0], "value": parts[1].strip()}, raw)

    if name == "goal":
        return UserCommand(CommandKind.GOAL, {"objective": _required(rest, usage("goal", "OBJECTIVE"))}, raw)
    if name == "approve":
        revision = None
        if rest:
            try:
                revision = int(rest)
            except ValueError as exc:
                raise CommandParseError("Plan revision must be an integer.") from exc
        return UserCommand(CommandKind.APPROVE, {"revision": revision}, raw)
    if name in {"reject", "replan"}:
        feedback = _required(rest, usage(name, "FEEDBACK"))
        kind = CommandKind.REJECT if name == "reject" else CommandKind.REPLAN
        return UserCommand(kind, {"feedback": feedback}, raw)
    if name == "help":
        return UserCommand(CommandKind.HELP, {"topic": rest.lower() or None}, raw)
    if name in {"plan", "status", "history", "auto", "pause", "resume", "quit"}:
        if rest:
            raise CommandParseError(f"{prefix}{name} does not take arguments.")
        return UserCommand(CommandKind(name), raw=raw)
    if name == "cancel":
        return UserCommand(CommandKind.CANCEL, {"confirmation": rest}, raw)
    if name == "resolve":
        parts = rest.split(maxsplit=2)
        if len(parts) < 3:
            raise CommandParseError(
                f"Usage: {usage('resolve', 'ACTION_OR_DELEGATION_ID applied|not-run INSPECTION NOTE')}"
            )
        resolution = parts[1].lower().replace("_", "-")
        if resolution not in {"applied", "not-run"}:
            raise CommandParseError("Resolution must be 'applied' or 'not-run'.")
        return UserCommand(
            CommandKind.RESOLVE,
            {"action_id": parts[0], "resolution": resolution, "note": parts[2].strip()},
            raw,
        )
    if name == "run":
        steps = None
        if rest:
            try:
                steps = int(rest)
            except ValueError as exc:
                raise CommandParseError("Run steps must be a positive integer.") from exc
            if steps < 1:
                raise CommandParseError("Run steps must be a positive integer.")
        return UserCommand(CommandKind.RUN, {"steps": steps}, raw)
    if name == "add":
        try:
            text, criteria = _text_and_criteria(rest)
        except CommandParseError as exc:
            if prefix == "/":
                raise CommandParseError(str(exc).replace(":add", "/add")) from exc
            raise
        return UserCommand(CommandKind.ADD, {"text": text, "acceptance_criteria": criteria}, raw)
    if name == "edit":
        parts = rest.split(maxsplit=2)
        fields = {"title", "description", "accept", "verify", "depends", "risk"}
        if len(parts) >= 3 and parts[1].lower() in fields:
            task_id, field_name, value = parts[0].upper(), parts[1].lower(), parts[2].strip()
        else:
            task_id, value = _task_and_text(rest, usage("edit", "TASK_ID [FIELD] VALUE"))
            field_name = "task"
        return UserCommand(
            CommandKind.EDIT,
            {"task_id": task_id, "field": field_name, "value": value},
            raw,
        )
    if name in {"remove", "rm"}:
        return UserCommand(
            CommandKind.REMOVE,
            {"task_id": _required(rest, usage("remove", "TASK_ID")).upper()},
            raw,
        )
    if name in {"done", "todo", "block", "skip"}:
        parts = rest.split(maxsplit=1)
        if not parts:
            raise CommandParseError(f"Usage: {usage(name, 'TASK_ID [NOTE]')}")
        status = {"done": "done", "todo": "pending", "block": "blocked", "skip": "skipped"}[name]
        return UserCommand(
            CommandKind.TASK_STATUS,
            {"task_id": parts[0].upper(), "status": status, "note": parts[1] if len(parts) > 1 else ""},
            raw,
        )
    if name == "todo":  # defensive; handled above, retained for readability
        raise CommandParseError(f"Usage: {usage('todo', 'TASK_ID [NOTE]')}")

    raise CommandParseError(
        f"Unknown command '{prefix}{name}'. Type / for commands or /help for details."
    )
