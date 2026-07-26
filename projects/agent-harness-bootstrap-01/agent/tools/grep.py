"""Bounded regular-expression search confined to the active workspace."""

from __future__ import annotations

import re
from pathlib import Path

from ._security import (
    MAX_GREP_FILES,
    MAX_GREP_LINE_CHARS,
    MAX_GREP_PATTERN_CHARS,
    MAX_PATH_CHARS,
    MAX_TOOL_OUTPUT_CHARS,
    MAX_TRAVERSAL_DEPTH,
    MAX_TRAVERSAL_ENTRIES,
    RESERVED_DIRECTORY,
    ToolSecurityError,
    display_path,
    read_text_limited,
    reject_sensitive_path,
    reject_sensitive_spelling,
    resolve_workspace_path,
    sensitive_content_reason,
)
from ._traversal import BoundedWalker


IGNORE = {
    ".git", ".venv", "venv", "__pycache__", "node_modules", ".idea", ".vscode",
    RESERVED_DIRECTORY,
}
MAX_MATCHES = 200

REQUIRES_APPROVAL = False

SCHEMA = {
    "type": "function",
    "function": {
        "name": "grep",
        "description": (
            "Search UTF-8 file contents for a regular-expression pattern, "
            "recursively. Returns matching lines as 'path:line: text'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "maxLength": MAX_GREP_PATTERN_CHARS,
                    "description": "Regular expression to search for.",
                },
                "path": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": MAX_PATH_CHARS,
                    "description": (
                        "File or directory to search, relative to the active "
                        "workspace. Defaults to '.'."
                    ),
                },
            },
            "required": ["pattern"],
            "additionalProperties": False,
        },
    },
}


def _search_file(
    file_path: Path,
    regex: re.Pattern[str],
    hits: list[str],
) -> tuple[bool, bool]:
    """Search one file; return (searched, line_was_truncated)."""

    try:
        safe_path = resolve_workspace_path(str(file_path), must_exist=True)
        reject_sensitive_path(safe_path)
    except ValueError:
        return False, False
    try:
        content, _ = read_text_limited(safe_path)
    except ToolSecurityError:
        # Binary, oversized, special, and unreadable files are intentionally
        # skipped rather than aborting an otherwise useful repository search.
        return False, False
    if sensitive_content_reason(content) is not None:
        return False, False

    truncated_line = False
    label = display_path(safe_path)
    for lineno, line in enumerate(content.splitlines(), start=1):
        searchable = line
        if len(searchable) > MAX_GREP_LINE_CHARS:
            searchable = searchable[:MAX_GREP_LINE_CHARS]
            truncated_line = True
        if regex.search(searchable):
            rendered = searchable
            if len(line) > MAX_GREP_LINE_CHARS:
                rendered += "..."
            hits.append(f"{label}:{lineno}: {rendered}")
            # Keep one sentinel match so an exact-at-the-limit result is not
            # incorrectly labelled as truncated.
            if len(hits) > MAX_MATCHES:
                break
    return True, truncated_line


def _unsafe_regex_reason(pattern: str) -> str | None:
    """Reject constructs with well-known catastrophic backtracking shapes."""

    try:
        from re import _constants, _parser  # type: ignore[attr-defined]
    except ImportError:
        # Conservative fallback for Python versions without the internal parser.
        if re.search(r"\\[1-9]", pattern):
            return "backreferences are not allowed"
        if re.search(r"\([^)]*[+*][^)]*\)[+*{]", pattern):
            return "nested repetition is not allowed"
        return None

    repeat_ops = {
        _constants.MAX_REPEAT,
        _constants.MIN_REPEAT,
        getattr(_constants, "POSSESSIVE_REPEAT", object()),
    }
    group_ref_ops = {
        _constants.GROUPREF,
        _constants.GROUPREF_EXISTS,
        getattr(_constants, "GROUPREF_IGNORE", object()),
        getattr(_constants, "GROUPREF_LOC_IGNORE", object()),
        getattr(_constants, "GROUPREF_UNI_IGNORE", object()),
    }

    def inspect(nodes: object, *, inside_repeat: bool = False) -> str | None:
        for operation, argument in nodes:  # type: ignore[union-attr]
            if operation in group_ref_ops:
                return "backreferences are not allowed"
            if operation in repeat_ops:
                if inside_repeat:
                    return "nested repetition is not allowed"
                child = argument[-1]
                reason = inspect(child, inside_repeat=True)
                if reason is not None:
                    return reason
            elif operation is _constants.SUBPATTERN:
                reason = inspect(argument[-1], inside_repeat=inside_repeat)
                if reason is not None:
                    return reason
            elif operation is _constants.BRANCH:
                if inside_repeat:
                    return "alternation inside repetition is not allowed"
                for branch in argument[1]:
                    reason = inspect(branch, inside_repeat=inside_repeat)
                    if reason is not None:
                        return reason
            elif operation in {_constants.ASSERT, _constants.ASSERT_NOT}:
                reason = inspect(argument[1], inside_repeat=inside_repeat)
                if reason is not None:
                    return reason
        return None

    try:
        return inspect(_parser.parse(pattern, 0))
    except (re.error, ValueError, TypeError):
        return "pattern could not be safely analysed"


def _render(hits: list[str], *, truncated: bool) -> str:
    lines: list[str] = []
    used = 0
    for hit in hits[:MAX_MATCHES]:
        added = len(hit) + (1 if lines else 0)
        if used + added > MAX_TOOL_OUTPUT_CHARS:
            truncated = True
            break
        lines.append(hit)
        used += added
    result = "\n".join(lines)
    if truncated:
        marker = f"... (truncated at {len(lines)} displayed matches)"
        while lines:
            result = "\n".join(lines)
            if len(result) + 1 + len(marker) <= MAX_TOOL_OUTPUT_CHARS:
                break
            lines.pop()
            marker = f"... (truncated at {len(lines)} displayed matches)"
        result = "\n".join(lines)
        result = f"{result}\n{marker}" if result else marker
        if len(result) > MAX_TOOL_OUTPUT_CHARS:
            result = result[:MAX_TOOL_OUTPUT_CHARS]
    return result


def run(pattern: str, path: str = ".") -> str:
    if len(pattern) > MAX_GREP_PATTERN_CHARS:
        return f"Error: regex exceeds the {MAX_GREP_PATTERN_CHARS}-character limit"
    try:
        regex = re.compile(pattern)
    except re.error as exc:
        return f"Error: invalid regex: {exc}"
    unsafe_reason = _unsafe_regex_reason(pattern)
    if unsafe_reason is not None:
        return f"Error: regex rejected by safety policy: {unsafe_reason}"

    reject_sensitive_spelling(path)
    base = resolve_workspace_path(path, allow_workspace=True, must_exist=True)
    reject_sensitive_path(base)
    hits: list[str] = []
    searched_files = 0
    traversal_truncated = False
    line_truncated = False

    if base.is_file():
        searched, shortened = _search_file(base, regex, hits)
        searched_files += int(searched)
        line_truncated = shortened
    elif base.is_dir():
        walker = BoundedWalker(
            base,
            ignore=IGNORE,
            max_entries=MAX_TRAVERSAL_ENTRIES,
            max_depth=MAX_TRAVERSAL_DEPTH,
        )
        for entry in walker:
            if entry.is_directory:
                continue
            if searched_files >= MAX_GREP_FILES:
                traversal_truncated = True
                break
            searched, shortened = _search_file(entry.path, regex, hits)
            searched_files += int(searched)
            line_truncated = line_truncated or shortened
            if len(hits) > MAX_MATCHES:
                traversal_truncated = True
                break
        traversal_truncated = traversal_truncated or walker.truncated
    else:
        return f"Error: '{display_path(base)}' is not a file or directory"

    if not hits:
        suffix = " (search was bounded)" if traversal_truncated or line_truncated else ""
        return f"(no matches for /{pattern}/ under '{display_path(base)}'){suffix}"
    return _render(
        hits,
        truncated=traversal_truncated or line_truncated or len(hits) > MAX_MATCHES,
    )
