"""
list_files — let the agent see what's in the project (read-only).

Walks a directory recursively and returns the relative paths it finds, so the
agent can orient itself in an unfamiliar codebase before reading anything.
"""

import os

# Directories we never want to walk into — noise that would bury the real files
# (and waste tokens). Add to this set as needed.
IGNORE = {".git", ".venv", "venv", "__pycache__", "node_modules", ".idea", ".vscode"}

REQUIRES_APPROVAL = False  # read-only — safe to run automatically

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
                    "description": "Directory to list, relative to the current directory. Defaults to '.'.",
                },
            },
            "required": [],
        },
    },
}


def run(path: str = ".") -> str:
    if not os.path.isdir(path):
        return f"Error: '{path}' is not a directory"

    entries = []
    for root, dirs, files in os.walk(path):
        # Prune ignored directories in place so os.walk doesn't descend into them.
        dirs[:] = [d for d in dirs if d not in IGNORE]
        for d in sorted(dirs):
            rel = os.path.relpath(os.path.join(root, d), path)
            entries.append(rel + "/")
        for f in sorted(files):
            rel = os.path.relpath(os.path.join(root, f), path)
            entries.append(rel)

    if not entries:
        return f"(no files under '{path}')"
    return "\n".join(sorted(entries))
