"""Platform-shell execution with a caller-selected bounded timeout."""

from __future__ import annotations

from . import run_bash


REQUIRES_APPROVAL = True

SCHEMA = {
    "type": "function",
    "function": {
        "name": "run_command",
        "description": (
            "Run a foreground command in the active workspace using the platform shell. "
            "Use start_process for a server or application that must stay running."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": run_bash.MAX_COMMAND_CHARS,
                },
                "timeout_seconds": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 3600,
                    "default": 120,
                },
                "cwd": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 4096,
                    "default": ".",
                },
            },
            "required": ["command"],
            "additionalProperties": False,
        },
    },
}


def run(command: str, timeout_seconds: int = 120, cwd: str = ".") -> str:
    return run_bash.run_with_timeout(command, timeout_seconds, cwd)
