"""Write a content-addressed Chat artifact to a workspace file."""

from __future__ import annotations

import hashlib
from pathlib import Path
from threading import RLock
from typing import Callable, Mapping, Any

from ._security import MAX_PATH_CHARS, MAX_WRITE_BYTES, atomic_write_bytes, encoded_text, get_workspace, resolve_workspace_path


REQUIRES_APPROVAL = True
_LOCK = RLock()
_PROVIDERS: dict[str, Callable[[str], Mapping[str, Any]]] = {}

SCHEMA = {"type":"function","function":{"name":"materialize_artifact","description":"Write an exact content-addressed Chat code artifact to a workspace path without regenerating its contents.","parameters":{"type":"object","properties":{"artifact_id":{"type":"string","minLength":1,"maxLength":128},"path":{"type":"string","minLength":1,"maxLength":MAX_PATH_CHARS},"expected_sha256":{"type":"string","pattern":"^[0-9a-fA-F]{64}$"}},"required":["artifact_id","path"],"additionalProperties":False}}}


def register_provider(workspace: str | Path, provider: Callable[[str], Mapping[str, Any]]) -> None:
    with _LOCK:
        _PROVIDERS[str(Path(workspace).resolve())] = provider


def unregister_provider(workspace: str | Path) -> None:
    with _LOCK:
        _PROVIDERS.pop(str(Path(workspace).resolve()), None)


def run(artifact_id: str, path: str, expected_sha256: str | None = None) -> str:
    with _LOCK:
        provider = _PROVIDERS.get(str(get_workspace()))
    if provider is None:
        return "Error: Chat artifact storage is unavailable in this runtime"
    try:
        artifact = provider(artifact_id)
        content = artifact.get("content")
        if not isinstance(content, str):
            return f"Error: artifact {artifact_id!r} has no text content"
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        stored = str(artifact.get("content_hash") or "")
        if stored and digest != stored:
            return "Error: stored Chat artifact failed its content hash"
        if expected_sha256 and digest != expected_sha256.casefold():
            return "Error: artifact hash does not match expected_sha256"
        target = resolve_workspace_path(path)
        data = encoded_text(content, limit=MAX_WRITE_BYTES)
        atomic_write_bytes(target, data, overwrite=True)
        return f"Materialized artifact {artifact_id} ({digest}) to {path}"
    except (KeyError, OSError, ValueError) as exc:
        return f"Error: artifact could not be materialized: {exc}"
