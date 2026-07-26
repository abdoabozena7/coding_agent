"""
read_file — the agent's first tool (read-only, no side effects).

A tool is two things bundled together:
  • SCHEMA — what the MODEL sees, so it knows the tool exists and how to call it.
  • run()  — what actually executes on our machine when the model asks for it.
"""

import os
from typing import Optional # ADDED: Necessary import for type hinting

from ._security import (
    MAX_PATH_CHARS,
    bounded_output,
    read_text_limited,
    reject_sensitive_path,
    reject_sensitive_spelling,
    resolve_workspace_path,
    sensitive_content_reason,
)

# Common image file extensions — reading these as text will always fail.
_IMAGE_EXTENSIONS = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".svg",
    ".ico", ".tiff", ".tif", ".avif", ".heic", ".heif",
})

# What the model sees. This is OpenAI's "function tool" format: the model reads
# the description + parameters to decide WHEN to call it and WITH WHAT arguments.
REQUIRES_APPROVAL = False  # read-only — safe to run automatically

SCHEMA = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read and return the full contents of a text file at the given path.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": MAX_PATH_CHARS,
                    "description": "Path to the file, relative to the active workspace.",
                },
                "start_line": {
                    "type": "integer",
                    "description": "The starting line number (1-based) for the read operation. Optional.",
                },
                "end_line": {
                    "type": "integer",
                    "description": "The ending line number (1-based, inclusive) for the read operation. Optional.",
                }
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    },
}

def _is_image(path: str) -> bool:
    """Check whether the path has a known image-file extension."""
    _name, ext = os.path.splitext(path)
    return ext.casefold() in _IMAGE_EXTENSIONS


def run(path: str, start_line: Optional[int] = None, end_line: Optional[int] = None) -> str:
    """Read the file and return its contents as a string."""
    if (start_line is not None and start_line < 1) or \
       (end_line is not None and end_line < 1):
        raise ValueError("Line numbers must be 1 or greater.")

    # Determine slice boundaries for 0-based indexing [start:end]
    if start_line is None and end_line is None:
        start, end = None, None
    elif start_line is not None and end_line is None: # Start only (L:EOF)
        # Inclusive range [start], 0-based index: [start - 1 : EOF]
        start = start_line - 1
        end = None
    elif start_line is None and end_line is not None: # End only (0:L)
        # Inclusive range [end], slice up to index 'end' (exclusive).
        start = None
        end = end_line
    else: # Both provided ([start]:[end])
        if start_line > end_line:
            raise ValueError("End line number cannot be less than start line number.")
        # Inclusive range [L_start, L_end] maps to slice indices [L_start - 1 : L_end]
        start = start_line - 1
        end = end_line

    reject_sensitive_spelling(path)
    resolved = resolve_workspace_path(path, must_exist=True)
    reject_sensitive_path(resolved)
    if _is_image(path):
        return (
            f'Cannot read "{path}" (this model does not support image input). '
            "Inform the user."
        )
    content, _ = read_text_limited(resolved)
    if sensitive_content_reason(content) is not None:
        return "Error: file content is protected by the sensitive-data policy"

    # --- Range Reading Logic ---
    all_lines = content.splitlines(keepends=True)
    
    selected_lines = []
    if start is None and end is None: 
        selected_lines = all_lines
    elif start is not None and end is None: # Start only (Slice from index 'start' to the end)
        selected_lines = all_lines[start:]
    elif start is None and end is not None: # End only (Slice up to, but NOT including, index 'end')
        selected_lines = all_lines[:end]
    else: # Both specified (Slice from index 'start' up to, but NOT including, index 'end')
        selected_lines = all_lines[start:end]

    combined_content = "".join(selected_lines)

    # bouded_output handles None/empty strings correctly.
    result, _ = bounded_output(combined_content)
    return result