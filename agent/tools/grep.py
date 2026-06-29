"""
grep — let the agent search file contents (read-only).

Scans text files under a path for a regex pattern and returns matching
"file:line: text" hits, so the agent can locate where something is defined or
used without reading every file.
"""

import os
import re

IGNORE = {".git", ".venv", "venv", "__pycache__", "node_modules", ".idea", ".vscode"}

# Don't try to grep binaries; cap matches so a broad pattern can't flood context.
MAX_MATCHES = 200

REQUIRES_APPROVAL = False  # read-only — safe to run automatically

SCHEMA = {
    "type": "function",
    "function": {
        "name": "grep",
        "description": (
            "Search file contents for a regular-expression pattern, recursively. "
            "Returns matching lines as 'path:line: text'. Use this to find where "
            "something is defined or used."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regular expression to search for.",
                },
                "path": {
                    "type": "string",
                    "description": "File or directory to search, relative to the current directory. Defaults to '.'.",
                },
            },
            "required": ["pattern"],
        },
    },
}


def _search_file(file_path: str, regex: "re.Pattern", hits: list) -> None:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                if regex.search(line):
                    hits.append(f"{file_path}:{lineno}: {line.rstrip()}")
                    if len(hits) >= MAX_MATCHES:
                        return
    except (UnicodeDecodeError, OSError):
        # Skip binaries and unreadable files rather than failing the whole search.
        return


def run(pattern: str, path: str = ".") -> str:
    try:
        regex = re.compile(pattern)
    except re.error as e:
        return f"Error: invalid regex: {e}"

    hits: list = []

    if os.path.isfile(path):
        _search_file(path, regex, hits)
    elif os.path.isdir(path):
        for root, dirs, files in os.walk(path):
            dirs[:] = [d for d in dirs if d not in IGNORE]
            for f in sorted(files):
                _search_file(os.path.join(root, f), regex, hits)
                if len(hits) >= MAX_MATCHES:
                    break
            if len(hits) >= MAX_MATCHES:
                break
    else:
        return f"Error: '{path}' not found"

    if not hits:
        return f"(no matches for /{pattern}/ under '{path}')"

    result = "\n".join(hits)
    if len(hits) >= MAX_MATCHES:
        result += f"\n... (truncated at {MAX_MATCHES} matches)"
    return result
