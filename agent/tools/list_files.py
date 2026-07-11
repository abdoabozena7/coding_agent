"""Bounded, workspace-confined recursive directory listing."""

from __future__ import annotations

from ._security import (
    MAX_LIST_ENTRIES,
    MAX_PATH_CHARS,
    MAX_TOOL_OUTPUT_CHARS,
    MAX_TRAVERSAL_DEPTH,
    MAX_TRAVERSAL_ENTRIES,
    RESERVED_DIRECTORY,
    display_path,
    reject_sensitive_path,
    reject_sensitive_spelling,
    resolve_workspace_path,
)
from ._traversal import BoundedWalker


IGNORE = {
    ".git", ".venv", "venv", "__pycache__", "node_modules", ".idea", ".vscode",
    RESERVED_DIRECTORY,
}

REQUIRES_APPROVAL = False

SCHEMA = {
    "type": "function",
    "function": {
        "name": "list_files",
        "description": (
            "List files in a directory, recursively. Returns relative paths, "
            "one per line (directories end with '/'). Use this to explore the "
            "project structure before reading specific files."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": MAX_PATH_CHARS,
                    "description": (
                        "Directory to list, relative to the active workspace. "
                        "Defaults to '.'."
                    ),
                },
            },
            "required": [],
            "additionalProperties": False,
        },
    },
}


def _render(entries: list[str], truncated: bool) -> str:
    lines: list[str] = []
    used = 0
    for entry in sorted(entries):
        added = len(entry) + (1 if lines else 0)
        if used + added > MAX_TOOL_OUTPUT_CHARS:
            truncated = True
            break
        lines.append(entry)
        used += added
    result = "\n".join(lines)
    if truncated:
        marker = f"... (truncated at {len(lines)} entries)"
        while lines:
            result = "\n".join(lines)
            if len(result) + 1 + len(marker) <= MAX_TOOL_OUTPUT_CHARS:
                break
            lines.pop()
            marker = f"... (truncated at {len(lines)} entries)"
        result = "\n".join(lines)
        result = f"{result}\n{marker}" if result else marker
        if len(result) > MAX_TOOL_OUTPUT_CHARS:
            result = result[:MAX_TOOL_OUTPUT_CHARS]
    return result


def run(path: str = ".") -> str:
    reject_sensitive_spelling(path)
    base = resolve_workspace_path(path, allow_workspace=True, must_exist=True)
    reject_sensitive_path(base)
    if not base.is_dir():
        return f"Error: '{display_path(base)}' is not a directory"

    entries: list[str] = []
    truncated = False
    walker = BoundedWalker(
        base,
        ignore=IGNORE,
        max_entries=MAX_TRAVERSAL_ENTRIES,
        max_depth=MAX_TRAVERSAL_DEPTH,
    )
    for entry in walker:
        if len(entries) >= MAX_LIST_ENTRIES:
            truncated = True
            break
        relative = entry.path.relative_to(base).as_posix()
        entries.append(relative + "/" if entry.is_directory else relative)
    truncated = truncated or walker.truncated

    if not entries:
        return f"(no files under '{display_path(base)}')"
    return _render(entries, truncated)
