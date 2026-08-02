"""Apply a standard unified diff without invoking an ambient shell utility."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from ._security import (
    MAX_PATH_CHARS,
    MAX_WRITE_BYTES,
    atomic_write_bytes,
    display_path,
    encoded_text,
    reject_sensitive_path,
    resolve_workspace_path,
)


# Applying an in-scope patch is a normal workspace mutation. The harness still
# enforces path, scope, footprint, journaling, and uncertain-mutation gates.
REQUIRES_APPROVAL = False
MAX_PATCH_CHARS = MAX_WRITE_BYTES

SCHEMA = {
    "type": "function",
    "function": {
        "name": "apply_patch",
        "description": (
            "Apply a text patch atomically inside the workspace. Accepts a standard "
            "unified diff with ---/+++ headers, or the common *** Begin Patch / "
            "*** Add File / Update File / Delete File format. Update hunks must "
            "contain an exact, unambiguous preimage. Binary patches and paths "
            "outside the workspace are rejected."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "patch": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": MAX_PATCH_CHARS,
                },
                "base_path": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": MAX_PATH_CHARS,
                    "default": ".",
                },
            },
            "required": ["patch"],
            "additionalProperties": False,
        },
    },
}


@dataclass(frozen=True)
class _FilePatch:
    old_path: str | None
    new_path: str | None
    hunks: tuple[tuple[int, int, int, int, tuple[str, ...]], ...]


_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def _find_unique_preimage(
    source: list[str],
    expected: list[str],
    *,
    start: int,
) -> int:
    """Return the one exact authored preimage location at or after ``start``."""

    if not expected:
        raise ValueError(
            "Update File insertion-only hunks are ambiguous; include unchanged "
            "context or use a standard unified diff with line numbers"
        )
    matches = [
        index
        for index in range(max(0, start), len(source) - len(expected) + 1)
        if source[index : index + len(expected)] == expected
    ]
    if not matches:
        raise ValueError("Update File preimage does not match the current file")
    if len(matches) != 1:
        raise ValueError(
            "Update File preimage is ambiguous; add more unchanged context"
        )
    return matches[0]


def _normalize_patch_syntax(text: str, base_path: str) -> str:
    """Convert a common ``*** Begin Patch`` envelope to a unified diff.

    This is transport normalization only. Add/delete content is present in the
    authored patch or current file, and update locations are derived only from
    an exact unique preimage. No path, source text, or requested effect is
    invented by the harness.
    """

    stripped = text.strip()
    if not stripped.startswith("*** Begin Patch"):
        return text
    lines = stripped.splitlines()
    if not lines or lines[0].strip() != "*** Begin Patch":
        raise ValueError("invalid *** Begin Patch envelope")
    output: list[str] = []
    index = 1
    saw_file = False
    ended = False
    while index < len(lines):
        line = lines[index]
        if line.strip() == "*** End Patch":
            index += 1
            ended = True
            while index < len(lines) and lines[index].strip() == "*** End Patch":
                index += 1
            if index < len(lines) and lines[index].strip() == "*** Begin Patch":
                index += 1
                ended = False
                continue
            break
        match = re.fullmatch(
            r"\*\*\* (Add|Update|Delete) File:\s*(.+?)\s*",
            line,
        )
        if match is None:
            raise ValueError(
                "invalid file section in *** Begin Patch envelope"
            )
        operation = match.group(1)
        path = match.group(2).strip()
        if not path:
            raise ValueError(f"{operation} File requires a path")
        index += 1
        if operation == "Add":
            body: list[str] = []
            while index < len(lines) and not lines[index].startswith("*** "):
                if not lines[index].startswith("+"):
                    raise ValueError("every Add File content line must start with '+'")
                body.append(lines[index])
                index += 1
            output.extend(
                [
                    "--- /dev/null",
                    f"+++ b/{path}",
                    f"@@ -0,0 +1,{len(body)} @@",
                    *body,
                ]
            )
        elif operation == "Delete":
            if index < len(lines) and not lines[index].startswith("*** "):
                raise ValueError("Delete File sections cannot contain patch body lines")
            resolved = resolve_workspace_path(_clean_header_path(path, base_path), must_exist=True)
            reject_sensitive_path(resolved)
            source = resolved.read_text(encoding="utf-8").replace("\r\n", "\n").splitlines()
            output.extend(
                [
                    f"--- a/{path}",
                    "+++ /dev/null",
                    f"@@ -1,{len(source)} +0,0 @@" if source else "@@ -0,0 +0,0 @@",
                    *(f"-{value}" for value in source),
                ]
            )
        else:
            resolved = resolve_workspace_path(_clean_header_path(path, base_path), must_exist=True)
            reject_sensitive_path(resolved)
            source = resolved.read_text(encoding="utf-8").replace("\r\n", "\n").splitlines()
            hunks: list[str] = []
            search_from = 0
            cumulative_delta = 0
            saw_hunk = False
            while index < len(lines) and not lines[index].startswith("*** "):
                if not lines[index].startswith("@@"):
                    raise ValueError("Update File body must begin with an @@ hunk")
                saw_hunk = True
                index += 1
                body: list[str] = []
                while (
                    index < len(lines)
                    and not lines[index].startswith("@@")
                    and not lines[index].startswith("*** ")
                ):
                    if not lines[index].startswith((" ", "+", "-")):
                        raise ValueError(
                            "Update File hunk lines must start with space, '+', or '-'"
                        )
                    body.append(lines[index])
                    index += 1
                expected = [value[1:] for value in body if value.startswith((" ", "-"))]
                start = _find_unique_preimage(source, expected, start=search_from)
                old_count = len(expected)
                new_count = sum(value.startswith((" ", "+")) for value in body)
                old_start = start + 1
                new_start = old_start + cumulative_delta
                hunks.extend(
                    [
                        f"@@ -{old_start},{old_count} +{new_start},{new_count} @@",
                        *body,
                    ]
                )
                search_from = start + old_count
                cumulative_delta += new_count - old_count
            if not saw_hunk:
                raise ValueError("Update File section contains no hunks")
            output.extend([f"--- a/{path}", f"+++ b/{path}", *hunks])
        saw_file = True
    while index < len(lines) and lines[index].strip() == "*** End Patch":
        index += 1
    if index != len(lines) or not ended or not saw_file:
        raise ValueError("invalid or empty *** Begin Patch envelope")
    return "\n".join(output) + "\n"


def patch_paths(text: str, base_path: str = ".") -> tuple[str, ...]:
    """Extract authored patch targets for scope and mutation checks."""

    paths: list[str] = []
    base = "" if base_path in {"", "."} else base_path.rstrip("/\\") + "/"
    for match in re.finditer(
        r"(?m)^\*\*\* (?:Add|Update|Delete) File:\s*(.+?)\s*$",
        str(text or ""),
    ):
        paths.append((base + match.group(1).strip()).replace("\\", "/"))
    for match in re.finditer(
        r"(?m)^(?:---|\+\+\+)\s+(?:[ab]/)?([^\t\r\n]+)",
        str(text or ""),
    ):
        value = match.group(1).strip()
        if value != "/dev/null":
            paths.append((base + value).replace("\\", "/").removeprefix("./"))
    return tuple(dict.fromkeys(path for path in paths if path))


def _clean_header_path(value: str, base_path: str) -> str | None:
    raw = value.split("\t", 1)[0].strip()
    if raw == "/dev/null":
        return None
    if raw.startswith(("a/", "b/")):
        raw = raw[2:]
    base = "" if base_path in {"", "."} else base_path.rstrip("/\\") + "/"
    return base + raw


def _parse(text: str, base_path: str) -> tuple[_FilePatch, ...]:
    if "GIT binary patch" in text or "Binary files" in text:
        raise ValueError("binary patches are not supported")
    lines = text.splitlines(keepends=True)
    result: list[_FilePatch] = []
    index = 0
    while index < len(lines):
        if not lines[index].startswith("--- "):
            index += 1
            continue
        old_path = _clean_header_path(lines[index][4:], base_path)
        index += 1
        if index >= len(lines) or not lines[index].startswith("+++ "):
            raise ValueError("every --- header must be followed by +++")
        new_path = _clean_header_path(lines[index][4:], base_path)
        index += 1
        hunks = []
        while index < len(lines) and not lines[index].startswith("--- "):
            if not lines[index].startswith("@@ "):
                index += 1
                continue
            match = _HUNK.match(lines[index].rstrip("\r\n"))
            if match is None:
                raise ValueError(f"invalid hunk header: {lines[index].strip()}")
            old_start, old_count, new_start, new_count = (
                int(match.group(1)), int(match.group(2) or 1),
                int(match.group(3)), int(match.group(4) or 1),
            )
            index += 1
            body: list[str] = []
            while index < len(lines) and not lines[index].startswith(("@@ ", "--- ")):
                line = lines[index]
                if line.startswith("\\ No newline at end of file"):
                    index += 1
                    continue
                if not line.startswith((" ", "+", "-")):
                    raise ValueError(f"invalid unified-diff line: {line[:80].rstrip()}")
                body.append(line)
                index += 1
            hunks.append((old_start, old_count, new_start, new_count, tuple(body)))
        if not hunks and old_path is not None and new_path is not None:
            raise ValueError("file patch contains no hunks")
        result.append(_FilePatch(old_path, new_path, tuple(hunks)))
    if not result:
        raise ValueError("patch contains no unified file headers")
    return tuple(result)


def _apply_hunks(original: str, patch: _FilePatch) -> str:
    source = original.splitlines(keepends=True)
    output: list[str] = []
    cursor = 0
    for old_start, old_count, _new_start, new_count, body in patch.hunks:
        target = max(0, old_start - 1)
        if target < cursor or target > len(source):
            raise ValueError("hunk position is outside the source file")
        output.extend(source[cursor:target])
        cursor = target
        consumed = produced = 0
        for line in body:
            marker, value = line[0], line[1:]
            if marker in {" ", "-"}:
                if cursor >= len(source) or source[cursor] != value:
                    raise ValueError("patch preimage does not match the current file")
                cursor += 1
                consumed += 1
            if marker in {" ", "+"}:
                output.append(value)
                produced += 1
        if consumed != old_count or produced != new_count:
            raise ValueError("hunk line counts do not match its header")
    output.extend(source[cursor:])
    return "".join(output)


def run(patch: str, base_path: str = ".") -> str:
    if not isinstance(patch, str) or not patch.strip():
        return "Error: patch must be a non-empty string"
    try:
        patch = _normalize_patch_syntax(patch, base_path)
        parsed = _parse(patch, base_path)
        originals: dict[Path, bytes | None] = {}
        replacements: dict[Path, bytes | None] = {}
        for item in parsed:
            source_path = resolve_workspace_path(item.old_path) if item.old_path else None
            target_path = resolve_workspace_path(item.new_path) if item.new_path else None
            if source_path is not None:
                reject_sensitive_path(source_path)
            if target_path is not None:
                reject_sensitive_path(target_path)
            if source_path is None and target_path is None:
                raise ValueError("patch cannot read and write /dev/null")
            if source_path is not None and target_path is not None and source_path != target_path:
                raise ValueError("rename patches are not supported")
            if source_path is not None:
                raw = source_path.read_bytes()
                originals.setdefault(source_path, raw)
                original = raw.decode("utf-8")
            else:
                original = ""
            newline = "\r\n" if "\r\n" in original else "\n"
            updated = _apply_hunks(original.replace("\r\n", "\n"), item)
            if newline == "\r\n":
                updated = updated.replace("\n", "\r\n")
            if target_path is None:
                assert source_path is not None
                replacements[source_path] = None
            else:
                if target_path.exists() and target_path not in originals:
                    originals[target_path] = target_path.read_bytes()
                elif target_path not in originals:
                    originals[target_path] = None
                replacements[target_path] = encoded_text(updated, limit=MAX_WRITE_BYTES)

        completed: list[Path] = []
        try:
            for path, data in replacements.items():
                if data is None:
                    path.unlink()
                else:
                    atomic_write_bytes(path, data, overwrite=True)
                completed.append(path)
        except Exception:
            for path in reversed(completed):
                previous = originals.get(path)
                if previous is None:
                    try:
                        path.unlink()
                    except FileNotFoundError:
                        pass
                else:
                    atomic_write_bytes(path, previous, overwrite=True)
            raise
        changed = ", ".join(display_path(path) for path in replacements)
        return f"Applied patch to {len(replacements)} file(s): {changed}"
    except (OSError, UnicodeError, ValueError) as exc:
        return f"Error: patch could not be applied: {exc}"
