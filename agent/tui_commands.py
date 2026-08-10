"""Metadata for the deliberately small public terminal surface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class CommandSpec:
    name: str
    description: str
    category: str
    arguments: str = ""
    live_safe: bool = False
    checkpoint_required: bool = False

    @property
    def aliases(self) -> tuple[str, ...]:
        return ()

    @property
    def search_text(self) -> str:
        return " ".join(
            (self.name, self.description, self.category, self.arguments)
        ).casefold()


@dataclass(frozen=True, slots=True)
class CommandAvailability:
    visible: bool = True
    enabled: bool = True
    reason: str = ""


COMMAND_SPECS: tuple[CommandSpec, ...] = (
    CommandSpec("/plan", "open the explicit Ultra Plan workspace", "Workspace", live_safe=True),
    CommandSpec("/live", "open the simple read-only Live workspace", "Workspace", live_safe=True),
    CommandSpec(
        "/show-diff",
        "open the simple live workflow diff",
        "Workspace",
        live_safe=True,
    ),
    CommandSpec(
        "/advanced-tracing",
        "open the standalone developer trace",
        "Workspace",
        live_safe=True,
    ),
    CommandSpec("/settings", "open runtime, provider, project, terminal, and diagnostic settings", "Session", live_safe=True),
    CommandSpec("/pause", "pause active work at a safe boundary", "Control", checkpoint_required=True),
    CommandSpec("/resume", "resume the saved checkpoint", "Control", checkpoint_required=True),
    CommandSpec("/stop", "stop now and keep the saved stage resumable", "Control", checkpoint_required=True),
    CommandSpec("/undo", "revert accepted checkpoints", "Control", arguments="[STEPS]", checkpoint_required=True),
    CommandSpec("/help", "show the public commands and current key bindings", "Session", live_safe=True),
    CommandSpec("/quit", "checkpoint and leave the session", "Session"),
)

ALL_SLASH_COMMANDS: tuple[tuple[str, str], ...] = tuple(
    (spec.name, spec.description) for spec in COMMAND_SPECS
)
CODEX_SLASH_COMMANDS: tuple[tuple[str, str], ...] = ()
SLASH_COMMANDS = ALL_SLASH_COMMANDS

_GROUP_COPY = {
    "Workspace": "Open the plan, live view, workflow diff, or developer trace",
    "Control": "Pause, resume, stop, or undo checkpointed work",
    "Session": "Configure, get help, or leave safely",
}
COMMAND_GROUPS: tuple[tuple[str, str, tuple[str, ...]], ...] = tuple(
    (
        category,
        _GROUP_COPY[category],
        tuple(spec.name for spec in COMMAND_SPECS if spec.category == category),
    )
    for category in ("Workspace", "Control", "Session")
)

_DESCRIPTIONS = dict(ALL_SLASH_COMMANDS)
_CONTEXT_COMMANDS: dict[str, tuple[str, ...]] = {
    "idle": ("/plan", "/live", "/show-diff", "/advanced-tracing", "/settings", "/help", "/quit"),
    "paused": ("/resume", "/stop", "/show-diff", "/advanced-tracing", "/live", "/settings", "/help", "/quit"),
    "completed": ("/show-diff", "/live", "/advanced-tracing", "/undo", "/settings", "/help", "/quit"),
}
_ACTIVE = ("/pause", "/stop", "/live", "/show-diff", "/advanced-tracing", "/settings", "/help", "/quit")


def contextual_commands(status: str) -> tuple[tuple[str, str], ...]:
    normalized = str(status).strip().casefold()
    names = _CONTEXT_COMMANDS.get(
        normalized,
        _ACTIVE if normalized not in {"idle", "ready", "new"} else _CONTEXT_COMMANDS["idle"],
    )
    return tuple((name, _DESCRIPTIONS[name]) for name in names)


def command_availability(
    spec: CommandSpec,
    snapshot: Any | None,
) -> CommandAvailability:
    if snapshot is None:
        return CommandAvailability()
    status = str(getattr(snapshot, "status", "")).casefold()
    running = bool(getattr(snapshot, "running", False))
    if spec.name == "/resume" and status != "paused":
        return CommandAvailability(False, False, "Available only while paused")
    if spec.name == "/pause" and not running:
        return CommandAvailability(False, False, "Available while work is running")
    if spec.name == "/stop" and not running and status not in {
        "routing", "planning", "working", "retrying", "recovering", "paused",
    }:
        return CommandAvailability(False, False, "Available while work is active or paused")
    if spec.name == "/undo" and not bool(getattr(snapshot, "undo_available", False)):
        return CommandAvailability(False, False, "Available after a completed checkpoint")
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
        preferred = tuple(name for name, _ in contextual_commands(getattr(snapshot, "status", "idle")))
        rank = {name: index for index, name in enumerate(preferred)}
        matches = tuple(sorted(matches, key=lambda spec: (rank.get(spec.name, 999), spec.name)))
    if limit is None:
        return matches
    return matches[: max(1, int(limit))]
