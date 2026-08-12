"""Publish a structured, platform-neutral final result to the Output page."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from threading import RLock
from typing import Any, Callable

from ._security import display_path, get_workspace, resolve_workspace_path


REQUIRES_APPROVAL = False
SCHEMA = {
    "type": "function",
    "function": {
        "name": "publish_output",
        "description": (
            "Structure the final task handoff for the generic Output page. Include "
            "the final message, independently copyable text sections when useful, "
            "and any existing workspace files or screenshots the user should see."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "maxLength": 240},
                "message": {"type": "string", "minLength": 1, "maxLength": 50000},
                "copy_sections": {
                    "type": "array",
                    "maxItems": 20,
                    "items": {
                        "type": "object",
                        "properties": {
                            "label": {"type": "string", "minLength": 1, "maxLength": 120},
                            "text": {"type": "string", "minLength": 1, "maxLength": 50000},
                        },
                        "required": ["label", "text"],
                        "additionalProperties": False,
                    },
                },
                "assets": {
                    "type": "array",
                    "maxItems": 40,
                    "items": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "minLength": 1, "maxLength": 4096},
                            "label": {"type": "string", "maxLength": 160},
                            "kind": {"type": "string", "enum": ["image", "file"]},
                        },
                        "required": ["path"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["message"],
            "additionalProperties": False,
        },
    },
}
RESULT_CONTRACT = {
    "format": "json",
    "fields": ["status", "output_id", "title", "copy_sections", "assets"],
}

Publisher = Callable[[dict[str, Any]], dict[str, Any]]
_LOCK = RLock()
_PUBLISHERS: dict[str, Publisher] = {}


def register_provider(workspace: str | Path, provider: Publisher) -> None:
    with _LOCK:
        _PUBLISHERS[str(Path(workspace).resolve())] = provider


def unregister_provider(workspace: str | Path) -> None:
    with _LOCK:
        _PUBLISHERS.pop(str(Path(workspace).resolve()), None)


def _asset(raw: dict[str, Any], index: int) -> dict[str, Any]:
    target = resolve_workspace_path(str(raw.get("path") or ""), must_exist=True)
    if not target.is_file():
        raise ValueError("output assets must be regular files")
    suffix = target.suffix.casefold()
    kind = str(raw.get("kind") or "").strip().casefold()
    if not kind:
        kind = "image" if suffix in {".png", ".jpg", ".jpeg", ".webp", ".gif"} else "file"
    if kind == "image" and suffix not in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
        raise ValueError("image output assets must use PNG, JPEG, WebP, or GIF")
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    if kind == "image":
        # A screenshot may be published only after the current visual
        # evaluator selected its exact bytes. This prevents a weak model from
        # attaching an arbitrary workspace image or an older screenshot to a
        # completed Output envelope.
        from .inspect_images import has_current_evidence

        if not has_current_evidence(target, digest):
            raise ValueError(
                "image output assets must have current visual evidence and be selected by the evaluator"
            )
    relative = display_path(target)
    return {
        "id": hashlib.sha256(f"{relative}:{index}".encode("utf-8")).hexdigest()[:16],
        "path": relative,
        "label": str(raw.get("label") or target.stem).strip()[:160],
        "kind": kind,
        "sha256": digest,
        "byte_size": target.stat().st_size,
    }


def run(
    message: str,
    title: str = "",
    copy_sections: list | None = None,
    assets: list | None = None,
) -> str:
    workspace = get_workspace()
    with _LOCK:
        publisher = _PUBLISHERS.get(str(workspace))
    if publisher is None:
        return "Error: Output publishing is not configured for this workspace"
    prepared = {
        "version": 1,
        "title": str(title or "Task output").strip()[:240],
        "message": str(message).strip(),
        "copy_sections": [
            {
                "id": hashlib.sha256(f"{index}:{item['label']}:{item['text']}".encode("utf-8")).hexdigest()[:16],
                "label": str(item["label"]).strip()[:120],
                "text": str(item["text"]).strip(),
            }
            for index, item in enumerate(copy_sections or (), start=1)
            if isinstance(item, dict) and str(item.get("label") or "").strip() and str(item.get("text") or "").strip()
        ],
        "assets": [
            _asset(dict(item), index)
            for index, item in enumerate(assets or (), start=1)
            if isinstance(item, dict)
        ],
    }
    result = publisher(prepared)
    return json.dumps(result, ensure_ascii=False)
