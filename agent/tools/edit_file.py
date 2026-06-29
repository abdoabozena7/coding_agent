"""
edit_file — make a targeted change by replacing an exact string (WRITES TO DISK).

This is the workhorse of code editing. Instead of rewriting a whole file, the
agent supplies the exact text to find (old_str) and what to replace it with
(new_str). Result: small, reviewable changes instead of giant overwrites.

To avoid editing the wrong place, old_str must match EXACTLY ONCE. If it matches
zero or many times we return an error, and the agent makes its match more
specific and tries again — this is the "errors are tool results" idea at work.

Special case: if the file does not exist and old_str is empty, the file is
created with new_str as its contents (a handy way to create files).
"""

import os

REQUIRES_APPROVAL = True  # writes to disk — ask the human first

SCHEMA = {
    "type": "function",
    "function": {
        "name": "edit_file",
        "description": (
            "Edit a file by replacing an exact substring. 'old_str' must appear "
            "exactly once in the file and is replaced with 'new_str'. If the file "
            "does not exist and 'old_str' is empty, the file is created with "
            "'new_str' as its contents. Prefer this over write_file for changes "
            "to existing files."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to edit."},
                "old_str": {
                    "type": "string",
                    "description": "Exact text to find; must be unique in the file. Empty string to create a new file.",
                },
                "new_str": {"type": "string", "description": "Text to replace old_str with."},
            },
            "required": ["path", "old_str", "new_str"],
        },
    },
}


def run(path: str, old_str: str, new_str: str) -> str:
    # --- Create-a-new-file case: empty old_str means "make this file". ---
    if old_str == "":
        if os.path.exists(path):
            return f"Error: {path} already exists; pass a non-empty old_str to edit it."
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_str)
        return f"Created {path} ({len(new_str)} characters)"

    # --- Edit-an-existing-file case. ---
    if not os.path.isfile(path):
        return f"Error: {path} does not exist"

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    count = content.count(old_str)
    if count == 0:
        return "Error: old_str not found. It must match the file exactly, including whitespace."
    if count > 1:
        return f"Error: old_str appears {count} times; make it more specific so it matches exactly once."

    with open(path, "w", encoding="utf-8") as f:
        f.write(content.replace(old_str, new_str))
    return f"Edited {path} (1 replacement)"
