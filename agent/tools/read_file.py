"""
read_file — the agent's first tool (read-only, no side effects).

A tool is two things bundled together:
  • SCHEMA — what the MODEL sees, so it knows the tool exists and how to call it.
  • run()  — what actually executes on our machine when the model asks for it.
"""

import os

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


def run(path: str) -> str:
    """Read the file and return its contents as a string."""
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
    result, _ = bounded_output(content)
    return result
