"""Small platform-local registry for recently opened external workspaces."""

from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Any


def registry_path() -> Path:
    if os.name == "nt" and os.getenv("LOCALAPPDATA"):
        root = Path(os.environ["LOCALAPPDATA"]) / "GA3BAD"
    elif os.getenv("XDG_STATE_HOME"):
        root = Path(os.environ["XDG_STATE_HOME"]) / "ga3bad"
    elif __import__("sys").platform == "darwin":
        root = Path.home() / "Library" / "Application Support" / "GA3BAD"
    else:
        root = Path.home() / ".local" / "state" / "ga3bad"
    return root / "workspaces.json"


def list_recent_workspaces(*, limit: int = 12) -> tuple[dict[str, Any], ...]:
    path = registry_path()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return ()
    values = raw.get("workspaces", ()) if isinstance(raw, dict) else ()
    result: list[dict[str, Any]] = []
    for item in values if isinstance(values, list) else ():
        if not isinstance(item, dict) or not str(item.get("path") or "").strip():
            continue
        candidate = Path(str(item["path"])).expanduser()
        result.append(
            {
                "path": str(candidate),
                "name": str(item.get("name") or candidate.name or candidate),
                "last_opened": float(item.get("last_opened") or 0),
                "available": candidate.is_dir(),
            }
        )
    result.sort(key=lambda value: value["last_opened"], reverse=True)
    return tuple(result[: max(1, int(limit))])


def record_workspace(workspace: str | os.PathLike[str]) -> None:
    resolved = Path(workspace).expanduser().resolve(strict=True)
    if not resolved.is_dir():
        return
    path = registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = list(list_recent_workspaces(limit=50))
    normalized = os.path.normcase(str(resolved))
    values = [
        item for item in existing
        if os.path.normcase(str(Path(item["path"]))) != normalized
    ]
    values.insert(
        0,
        {
            "path": str(resolved),
            "name": resolved.name or str(resolved),
            "last_opened": time.time(),
        },
    )
    payload = json.dumps({"version": 1, "workspaces": values[:20]}, indent=2, ensure_ascii=False)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


__all__ = ["list_recent_workspaces", "record_workspace", "registry_path"]
