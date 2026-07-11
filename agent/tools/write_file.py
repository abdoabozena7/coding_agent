"""
write_file — create a new file or overwrite an existing one (WRITES TO DISK).

This is the agent's first tool with side effects. Until now everything was
read-only; from here the agent can change your project. (Phase 7 will add a
permission gate in front of risky tools like this one.)
"""

from ._security import (
    MAX_PATH_CHARS,
    MAX_WRITE_BYTES,
    atomic_write_bytes,
    display_path,
    encoded_text,
    resolve_workspace_path,
)

REQUIRES_APPROVAL = True  # writes to disk — ask the human first

SCHEMA = {
    "type": "function",
    "function": {
        "name": "write_file",
        "description": (
            "Create a new file, or overwrite an existing file, with the given "
            "content. Parent directories are created as needed. This replaces the "
            "ENTIRE file — to make a small change to an existing file, prefer "
            "edit_file."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": MAX_PATH_CHARS,
                    "description": "Path to the file within the active workspace.",
                },
                "content": {
                    "type": "string",
                    "maxLength": MAX_WRITE_BYTES,
                    "description": "Full UTF-8 contents to write to the file.",
                },
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
    },
}


def run(path: str, content: str) -> str:
    resolved = resolve_workspace_path(path)
    data = encoded_text(content, limit=MAX_WRITE_BYTES)
    atomic_write_bytes(resolved, data, overwrite=True)
    return f"Wrote {len(content)} characters to {display_path(resolved)}"
