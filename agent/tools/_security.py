"""Shared security primitives for filesystem-backed tools.

The process working directory is deliberately *not* a security boundary.  A
workspace must be configured explicitly, and every path is resolved through
that workspace before it is used.  Keeping these checks in one module makes it
much harder for a newly added tool to accidentally implement a weaker policy.
"""

from __future__ import annotations

import contextvars
import os
import stat
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


RESERVED_DIRECTORY = ".coding-agent"
MAX_PATH_CHARS = 4_096
MAX_FILE_BYTES = 1_048_576
MAX_WRITE_BYTES = 1_048_576
MAX_LIST_ENTRIES = 2_000
MAX_TRAVERSAL_ENTRIES = 10_000
MAX_TRAVERSAL_DEPTH = 64
MAX_GREP_FILES = 2_000
MAX_GREP_PATTERN_CHARS = 1_000
MAX_GREP_LINE_CHARS = 2_000
MAX_TOOL_OUTPUT_CHARS = 50_000

_SENSITIVE_DIRECTORIES = {
    ".aws", ".azure", ".docker", ".git", ".gcloud", ".gnupg", ".kube", ".ssh",
    ".secrets", "credentials", "secrets",
}
_SENSITIVE_EXACT_FILES = {
    ".env", ".envrc", ".git-credentials", ".htpasswd", ".netrc", ".npmrc", ".pypirc",
    ".terraformrc", "credentials", "credentials.json", "kubeconfig",
    "id_dsa", "id_ecdsa", "id_ed25519", "id_rsa", "secrets", "secrets.json",
    "service-account.json",
}
_SENSITIVE_SUFFIXES = {
    ".jks", ".key", ".keystore", ".p12", ".pem", ".pfx", ".pkcs12",
    ".tfstate", ".tfvars",
}
_SENSITIVE_DATA_SUFFIXES = {".conf", ".ini", ".json", ".toml", ".yaml", ".yml"}
_SENSITIVE_STEMS = {
    "client_secret", "credential", "credentials", "password", "passwords",
    "private_key", "secret", "secrets", "service_account", "token", "tokens",
}
_PRIVATE_KEY_MARKERS = (
    "-----BEGIN PRIVATE KEY-----",
    "-----BEGIN RSA PRIVATE KEY-----",
    "-----BEGIN EC PRIVATE KEY-----",
    "-----BEGIN DSA PRIVATE KEY-----",
    "-----BEGIN OPENSSH PRIVATE KEY-----",
    "-----BEGIN PGP PRIVATE KEY BLOCK-----",
)


class ToolSecurityError(ValueError):
    """A safe, user-facing error raised when a tool request is rejected."""


def safe_os_error(exc: OSError) -> str:
    """Describe an OS failure without echoing host paths or secret filenames."""

    code = exc.errno if exc.errno is not None else getattr(exc, "winerror", None)
    detail = exc.strerror or "operating-system error"
    return f"[{code}] {detail}" if code is not None else detail


@dataclass(frozen=True)
class ToolContext:
    """Immutable execution context shared by all tools in one logical task."""

    workspace: Path


_CONTEXT: contextvars.ContextVar[ToolContext | None] = contextvars.ContextVar(
    "coding_agent_tool_context", default=None
)


def _canonical_workspace(path: str | os.PathLike[str]) -> Path:
    if not isinstance(path, (str, os.PathLike)):
        raise ToolSecurityError("workspace must be a path")
    try:
        workspace = Path(path).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        detail = safe_os_error(exc) if isinstance(exc, OSError) else "resolution failed"
        raise ToolSecurityError(f"invalid workspace: {detail}") from exc
    if not workspace.is_dir():
        raise ToolSecurityError(f"workspace is not a directory: {workspace}")
    return workspace


def configure_workspace(path: str | os.PathLike[str]) -> ToolContext:
    """Set the workspace for the current execution context.

    This is intentionally explicit and never changes ``os.getcwd()``.  The
    returned value can be logged or retained by callers, while tools obtain it
    through :func:`get_tool_context`.
    """

    context = ToolContext(workspace=_canonical_workspace(path))
    _CONTEXT.set(context)
    return context


@contextmanager
def workspace_context(path: str | os.PathLike[str]) -> Iterator[ToolContext]:
    """Temporarily select a workspace, restoring the prior one on exit."""

    context = ToolContext(workspace=_canonical_workspace(path))
    token = _CONTEXT.set(context)
    try:
        yield context
    finally:
        _CONTEXT.reset(token)


def get_tool_context() -> ToolContext:
    context = _CONTEXT.get()
    if context is None:
        raise ToolSecurityError(
            "workspace is not configured; call tools.configure_workspace(path) first"
        )
    # A workspace can be deleted or replaced after configuration.  Resolve it
    # again so a stale context cannot silently become a symlink elsewhere.
    current = _canonical_workspace(context.workspace)
    if current != context.workspace:
        raise ToolSecurityError("configured workspace changed; configure it again")
    return context


def get_workspace() -> Path:
    return get_tool_context().workspace


def _normalised_component(component: str) -> str:
    # Win32 treats trailing dots/spaces as aliases, so `.coding-agent.` must be
    # protected exactly like `.coding-agent`.  Case is also insignificant.
    if os.name == "nt":
        component = component.rstrip(" .")
    return component.casefold()


def _validate_relative_parts(parts: tuple[str, ...]) -> None:
    windows_devices = {
        "con", "prn", "aux", "nul", "clock$",
        *(f"com{i}" for i in range(1, 10)),
        *(f"lpt{i}" for i in range(1, 10)),
    }
    for part in parts:
        normalised = _normalised_component(part)
        if normalised == RESERVED_DIRECTORY.casefold():
            raise ToolSecurityError(
                f"'{RESERVED_DIRECTORY}' is reserved for agent state"
            )
        if os.name == "nt":
            if ":" in part:
                raise ToolSecurityError("Windows alternate data streams are not allowed")
            device_name = normalised.split(".", 1)[0]
            if device_name in windows_devices:
                raise ToolSecurityError(f"Windows device path '{part}' is not allowed")


def _reject_reserved_spelling(parts: tuple[str, ...]) -> None:
    # Validate the spelling as well as the canonical target.  Otherwise a
    # symlink *inside* the reserved directory that points back into the normal
    # workspace could be used to enter agent state without leaving containment.
    if any(_normalised_component(part) == RESERVED_DIRECTORY.casefold() for part in parts):
        raise ToolSecurityError(f"'{RESERVED_DIRECTORY}' is reserved for agent state")


def _sensitive_parts_reason(
    parts: tuple[str, ...], *, final_is_directory: bool = False
) -> str | None:
    normalised = tuple(_normalised_component(part) for part in parts if part not in {"", ".", ".."})
    if not normalised:
        return None
    if any(
        part in _SENSITIVE_DIRECTORIES or part == ".env.example"
        for part in normalised[:-1]
    ):
        return "protected credential directory"

    name = normalised[-1]
    # Checked after directory protection: a template inside a secrets directory
    # is still part of that protected tree.
    if name == ".env.example":
        return "protected credential directory" if final_is_directory else None
    if name in _SENSITIVE_DIRECTORIES:
        return "protected credential directory"
    if name in _SENSITIVE_EXACT_FILES or name.startswith(".env."):
        return "environment or credential file"
    if name.endswith(".tfstate.backup"):
        return "state file may contain plaintext secrets"

    suffix = Path(name).suffix.casefold()
    stem = Path(name).stem.casefold().replace("-", "_")
    if suffix in _SENSITIVE_SUFFIXES:
        return "private key, credential, or plaintext state file"
    if suffix in _SENSITIVE_DATA_SUFFIXES:
        words = {word for word in stem.replace(".", "_").split("_") if word}
        if stem in _SENSITIVE_STEMS or words.intersection(_SENSITIVE_STEMS):
            return "credential or secret data file"
    return None


def sensitive_spelling_reason(path: str) -> str | None:
    """Classify a supplied path without checking whether it exists."""

    if not isinstance(path, str):
        return None
    try:
        return _sensitive_parts_reason(Path(path).parts)
    except (OSError, ValueError):
        return None


def sensitive_path_reason(path: Path) -> str | None:
    """Classify a canonical workspace path, including its resolved target."""

    try:
        parts = path.relative_to(get_workspace()).parts
    except ValueError:
        parts = path.parts
    try:
        final_is_directory = path.is_dir()
    except OSError:
        final_is_directory = False
    return _sensitive_parts_reason(parts, final_is_directory=final_is_directory)


def reject_sensitive_spelling(path: str) -> None:
    if sensitive_spelling_reason(path) is not None:
        raise ToolSecurityError("access to sensitive paths is denied by tool policy")


def reject_sensitive_path(path: Path) -> None:
    if sensitive_path_reason(path) is not None:
        raise ToolSecurityError("access to sensitive paths is denied by tool policy")


def is_sensitive_path(path: Path) -> bool:
    return sensitive_path_reason(path) is not None


def sensitive_content_reason(content: str) -> str | None:
    for marker in _PRIVATE_KEY_MARKERS:
        start = content.find(marker)
        if start < 0:
            continue
        end_marker = marker.replace("BEGIN", "END", 1)
        end = content.find(end_marker, start + len(marker))
        if end >= 0 and end - (start + len(marker)) >= 40:
            return "private key material"
    return None


def resolve_workspace_path(
    path: str,
    *,
    allow_workspace: bool = False,
    must_exist: bool | None = None,
) -> Path:
    """Return a canonical path proven to be inside the active workspace.

    ``Path.resolve`` follows every existing symlink (including a symlink in a
    parent directory), so containment is checked against the actual target and
    not merely the spelling supplied by the model.
    """

    if not isinstance(path, str):
        raise ToolSecurityError("path must be a string")
    if not path or not path.strip():
        raise ToolSecurityError("path must not be empty")
    if "\x00" in path:
        raise ToolSecurityError("path must not contain NUL bytes")
    if len(path) > MAX_PATH_CHARS:
        raise ToolSecurityError(f"path exceeds the {MAX_PATH_CHARS}-character limit")

    workspace = get_workspace()
    supplied = Path(path)
    _reject_reserved_spelling(supplied.parts)
    candidate = supplied if supplied.is_absolute() else workspace / supplied
    try:
        resolved = candidate.resolve(strict=False)
        relative = resolved.relative_to(workspace)
    except ValueError as exc:
        raise ToolSecurityError("path escapes the active workspace") from exc
    except (OSError, RuntimeError) as exc:
        detail = safe_os_error(exc) if isinstance(exc, OSError) else "resolution failed"
        raise ToolSecurityError(f"invalid path: {detail}") from exc

    if not allow_workspace and resolved == workspace:
        raise ToolSecurityError("a file path is required, not the workspace root")
    _validate_relative_parts(relative.parts)

    if must_exist is True and not resolved.exists():
        raise ToolSecurityError(f"path does not exist: {display_path(resolved)}")
    if must_exist is False and resolved.exists():
        raise ToolSecurityError(f"path already exists: {display_path(resolved)}")
    return resolved


def display_path(path: Path) -> str:
    """Render a canonical path without leaking paths above the workspace."""

    try:
        relative = path.relative_to(get_workspace())
    except ValueError:
        return "<outside workspace>"
    rendered = relative.as_posix()
    return rendered if rendered else "."


def ensure_regular_file(path: Path) -> os.stat_result:
    try:
        info = path.stat()
    except OSError as exc:
        raise ToolSecurityError(
            f"cannot inspect {display_path(path)}: {safe_os_error(exc)}"
        ) from exc
    if not stat.S_ISREG(info.st_mode):
        raise ToolSecurityError(f"not a regular file: {display_path(path)}")
    return info


def read_text_limited(path: Path, *, limit: int = MAX_FILE_BYTES) -> tuple[str, os.stat_result]:
    """Read UTF-8 text with a hard byte cap and return its initial stat data."""

    flags = (
        os.O_RDONLY
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    descriptor = -1
    try:
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            info = os.fstat(handle.fileno())
            if not stat.S_ISREG(info.st_mode):
                raise ToolSecurityError(f"not a regular file: {display_path(path)}")
            if info.st_size > limit:
                raise ToolSecurityError(
                    f"file exceeds the {limit}-byte read limit: {display_path(path)}"
                )
            data = handle.read(limit + 1)
            final_info = os.fstat(handle.fileno())
    except OSError as exc:
        raise ToolSecurityError(
            f"cannot read {display_path(path)}: {safe_os_error(exc)}"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if len(data) > limit:
        raise ToolSecurityError(
            f"file exceeds the {limit}-byte read limit: {display_path(path)}"
        )
    if file_fingerprint(info) != file_fingerprint(final_info):
        raise ToolSecurityError("file changed while it was being read; retry")

    # Validate the name and identity again after the read.  O_NOFOLLOW protects
    # the final component where available; this post-check also covers Windows
    # and detects a path swapped during the operation before any bytes escape.
    current_path = resolve_workspace_path(str(path), must_exist=True)
    if current_path != path:
        raise ToolSecurityError("file target changed while it was being read")
    current_info = ensure_regular_file(current_path)
    if file_fingerprint(current_info) != file_fingerprint(final_info):
        raise ToolSecurityError("file changed while it was being read; retry")
    try:
        return data.decode("utf-8"), final_info
    except UnicodeDecodeError as exc:
        raise ToolSecurityError(f"file is not valid UTF-8 text: {display_path(path)}") from exc


def encoded_text(content: str, *, limit: int = MAX_WRITE_BYTES) -> bytes:
    if not isinstance(content, str):
        raise ToolSecurityError("content must be a string")
    try:
        data = content.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ToolSecurityError("content is not valid UTF-8 text") from exc
    if len(data) > limit:
        raise ToolSecurityError(f"content exceeds the {limit}-byte write limit")
    return data


def file_fingerprint(info: os.stat_result) -> tuple[int, int, int, int]:
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)


def atomic_write_bytes(
    path: Path,
    data: bytes,
    *,
    overwrite: bool,
    expected: tuple[int, int, int, int] | None = None,
) -> None:
    """Commit complete bytes in one filesystem operation.

    A temporary file is written, flushed, and fsynced in the destination
    directory.  Existing targets are replaced atomically.  New targets use an
    atomic hard-link installation so an observer can never see a partial file.
    """

    workspace = get_workspace()
    parent = path.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ToolSecurityError(
            f"cannot create parent directory: {safe_os_error(exc)}"
        ) from exc

    # Re-resolve after mkdir.  This catches a parent that was a symlink, was
    # swapped during creation, or canonicalised somewhere unexpected.
    canonical_parent = resolve_workspace_path(
        str(parent), allow_workspace=True, must_exist=True
    )
    if not canonical_parent.is_dir() or canonical_parent != parent:
        raise ToolSecurityError("destination parent changed during validation")
    if workspace != get_workspace():
        raise ToolSecurityError("workspace changed during the write")

    existing_mode: int | None = None
    if path.exists():
        existing = ensure_regular_file(path)
        existing_mode = stat.S_IMODE(existing.st_mode)
        if not overwrite:
            raise ToolSecurityError(f"path already exists: {display_path(path)}")

    fd = -1
    temporary: str | None = None
    try:
        fd, temporary = tempfile.mkstemp(prefix=".agent-write-", dir=str(parent))
        with os.fdopen(fd, "wb") as handle:
            fd = -1
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if existing_mode is not None:
            os.chmod(temporary, existing_mode)

        # Re-check the parent and (for targeted edits) the exact file version
        # immediately before committing.
        if Path(parent).resolve(strict=True) != parent:
            raise ToolSecurityError("destination parent changed during the write")
        if expected is not None:
            try:
                current = path.stat()
            except OSError as exc:
                raise ToolSecurityError("file changed during the edit") from exc
            if file_fingerprint(current) != expected:
                raise ToolSecurityError("file changed during the edit; read it and retry")

        if overwrite:
            os.replace(temporary, path)
        else:
            # link() fails if another writer created the destination first.
            os.link(temporary, path)
            os.unlink(temporary)
        temporary = None
    except ToolSecurityError:
        raise
    except FileExistsError as exc:
        raise ToolSecurityError(f"path already exists: {display_path(path)}") from exc
    except OSError as exc:
        raise ToolSecurityError(f"atomic write failed: {safe_os_error(exc)}") from exc
    finally:
        if fd >= 0:
            os.close(fd)
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def bounded_output(text: str, *, limit: int = MAX_TOOL_OUTPUT_CHARS) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    marker = f"\n... (truncated at {limit} characters)"
    if len(marker) >= limit:
        return marker[:limit], True
    kept = max(0, limit - len(marker))
    return text[:kept] + marker, True
