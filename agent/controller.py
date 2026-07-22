"""UI-neutral command controller shared by interactive frontends.

The command language remains the compatibility boundary.  Terminal and web
clients submit the same :class:`UserCommand` values, while presentation adapters
decide how output, activity, and attention requests are rendered.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .cli import execute_command
from .commands import CommandKind, UserCommand, parse_command
from .config import SessionPreferences
from .runtime import AgentRuntime
from .ui import ConsoleUI
from .ui_state import WorkspaceUIStore


@dataclass(frozen=True, slots=True)
class CommandResult:
    accepted: bool
    command: str
    status: str
    transcript_start: int
    transcript_end: int


class AgentController:
    """Dispatch commands without owning a terminal or an HTTP transport."""

    def __init__(
        self,
        runtime: AgentRuntime,
        console: ConsoleUI,
        presentation: WorkspaceUIStore,
        preferences: SessionPreferences,
    ) -> None:
        self.runtime = runtime
        self.console = console
        self.presentation = presentation
        self.preferences = preferences

    def execute(self, value: str | UserCommand) -> CommandResult:
        command = value if isinstance(value, UserCommand) else parse_command(str(value))
        before = self.presentation.snapshot()
        start = before.transcript[-1].id if before.transcript else 0

        # Browser settings render model choices themselves.  Never let a bare
        # `/model` command fall through to a stdin picker in a headless worker.
        if command.kind is CommandKind.MODEL and not command.args.get("model") and not command.args.get("effort"):
            self.console.write(
                f"model = {self.runtime.provider_name}/{self.runtime.model_name}; "
                "choose a model from Settings or use /model NAME"
            )
            keep_running = True
        else:
            keep_running = execute_command(
                self.runtime,
                self.console,
                command,
                self.preferences,
                structured_attention=True,
            )

        after = self.presentation.snapshot()
        end = after.transcript[-1].id if after.transcript else start
        return CommandResult(
            accepted=bool(keep_running),
            command=command.raw or str(value),
            status=str(getattr(self.runtime.dashboard(), "status", "idle")),
            transcript_start=start,
            transcript_end=end,
        )

    def resolve_attention(self, request_id: str, key: str, text: str = "") -> bool:
        request = self.presentation.active_attention()
        if request is None or request.id != str(request_id):
            return False
        option = next((item for item in request.options if item.key == str(key)), None)
        if option is None:
            if not request.allow_custom or str(key) != "custom":
                return False
            return self.presentation.resolve_attention("custom", text=str(text))
        return self.presentation.resolve_attention(option.key, text=str(text))


__all__ = ["AgentController", "CommandResult"]
