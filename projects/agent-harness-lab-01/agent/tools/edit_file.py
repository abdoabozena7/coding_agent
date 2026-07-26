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

from ._security import (
    MAX_PATH_CHARS,
    MAX_WRITE_BYTES,
    atomic_write_bytes,
    display_path,
    encoded_text,
    file_fingerprint,
    read_text_limited,
    resolve_workspace_path,
)

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
                "path": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": MAX_PATH_CHARS,
                    "description": "Path to the file within the active workspace.",
                },
                "old_str": {
                    "type": "string",
                    "maxLength": MAX_WRITE_BYTES,
                    "description": "Exact text to find; must be unique in the file. Empty string to create a new file.",
                },
                "new_str": {
                    "type": "string",
                    "maxLength": MAX_WRITE_BYTES,
                    "description": "Text to replace old_str with.",
                },
            },
            "required": ["path", "old_str", "new_str"],
            "additionalProperties": False,
        },
    },
}


def run(path: str, old_str: str, new_str: str) -> str:
    resolved = resolve_workspace_path(path)

    # --- Create-a-new-file case: empty old_str means "make this file". ---
    if old_str == "":
        if resolved.exists():
            return (
                f"Error: {display_path(resolved)} already exists; "
                "pass a non-empty old_str to edit it."
            )
        data = encoded_text(new_str, limit=MAX_WRITE_BYTES)
        atomic_write_bytes(resolved, data, overwrite=False)
        return f"Created {display_path(resolved)} ({len(new_str)} characters)"

    # --- Edit-an-existing-file case. ---
    if not resolved.exists():
        return f"Error: {display_path(resolved)} does not exist"

    content, initial_info = read_text_limited(resolved)

    count = content.count(old_str)
    if count == 0:
        return "Error: old_str not found. It must match the file exactly, including whitespace."
    if count > 1:
        return f"Error: old_str appears {count} times; make it more specific so it matches exactly once."

    replacement = content.replace(old_str, new_str)
    data = encoded_text(replacement, limit=MAX_WRITE_BYTES)
    atomic_write_bytes(
        resolved,
        data,
        overwrite=True,
        expected=file_fingerprint(initial_info),
    )
    return f"Edited {display_path(resolved)} (1 replacement)"
