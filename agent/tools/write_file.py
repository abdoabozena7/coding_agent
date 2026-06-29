"""
write_file — create a new file or overwrite an existing one (WRITES TO DISK).

This is the agent's first tool with side effects. Until now everything was
read-only; from here the agent can change your project. (Phase 7 will add a
permission gate in front of risky tools like this one.)
"""

import os

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
                "path": {"type": "string", "description": "Path to the file to write."},
                "content": {"type": "string", "description": "Full contents to write to the file."},
            },
            "required": ["path", "content"],
        },
    },
}


def run(path: str, content: str) -> str:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)  # create intermediate dirs if needed
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Wrote {len(content)} characters to {path}"
