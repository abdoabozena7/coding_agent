"""Send workspace images to the runtime's configured vision-capable model."""

from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Mapping

from ._security import MAX_PATH_CHARS, display_path, ensure_regular_file, get_workspace, reject_sensitive_path, resolve_workspace_path


REQUIRES_APPROVAL = False
MAX_IMAGE_BYTES = 20 * 1024 * 1024
_LOCK = RLock()
_EVALUATORS: dict[str, Callable[[list[dict[str, str]], str, str], Mapping[str, Any]]] = {}
_EVIDENCE: dict[str, dict[str, str]] = {}

SCHEMA = {
    "type": "function",
    "function": {
        "name": "inspect_images",
        "description": (
            "Evaluate and rank workspace images using the configured vision-capable model. "
            "Use this after capture and before selecting images or writing image-dependent copy."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "paths": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 8,
                    "items": {"type": "string", "minLength": 1, "maxLength": MAX_PATH_CHARS},
                },
                "purpose": {"type": "string", "minLength": 1, "maxLength": 2_000},
                "criteria": {"type": "string", "minLength": 1, "maxLength": 5_000},
            },
            "required": ["paths", "purpose", "criteria"],
            "additionalProperties": False,
        },
    },
}

RESULT_CONTRACT = {
    "format": "json",
    "fields": ["status", "model", "evaluations", "ranking", "selected", "copy_facts"],
    "passed_evidence": (
        "status=evaluated proves the configured model accepted the current image bytes; "
        "each evaluation is bound to path and sha256"
    ),
}


def register_evaluator(
    workspace: str | Path,
    evaluator: Callable[[list[dict[str, str]], str, str], Mapping[str, Any]],
) -> None:
    with _LOCK:
        _EVALUATORS[str(Path(workspace).resolve())] = evaluator


def unregister_evaluator(workspace: str | Path) -> None:
    with _LOCK:
        key = str(Path(workspace).resolve())
        _EVALUATORS.pop(key, None)
        _EVIDENCE.pop(key, None)


def has_current_evidence(path: str | Path, digest: str) -> bool:
    with _LOCK:
        return _EVIDENCE.get(str(get_workspace()), {}).get(display_path(Path(path))) == digest


def _prepare(path: str) -> dict[str, str]:
    source = resolve_workspace_path(path, must_exist=True)
    reject_sensitive_path(source)
    info = ensure_regular_file(source)
    suffix = source.suffix.casefold()
    if suffix not in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
        raise ValueError("inspect_images accepts PNG, JPG, JPEG, WebP, or GIF files")
    if info.st_size > MAX_IMAGE_BYTES:
        raise ValueError(f"image exceeds the {MAX_IMAGE_BYTES}-byte limit")
    data = source.read_bytes()
    if len(data) != info.st_size:
        raise ValueError("image changed while it was being read; retry")
    mime_type = mimetypes.guess_type(source.name)[0] or "application/octet-stream"
    return {
        "path": display_path(source),
        "mime_type": mime_type,
        "sha256": hashlib.sha256(data).hexdigest(),
        "data": base64.b64encode(data).decode("ascii"),
    }


def run(paths: list[str], purpose: str, criteria: str) -> str:
    with _LOCK:
        evaluator = _EVALUATORS.get(str(get_workspace()))
    if evaluator is None:
        return "Error: vision evaluation is unavailable in this runtime"
    try:
        prepared = [_prepare(path) for path in paths]
        result = dict(evaluator(prepared, str(purpose).strip(), str(criteria).strip()))
        if result.get("status") != "evaluated":
            detail = str(result.get("reason") or "the configured model did not produce visual evidence")
            return f"Error: vision evaluation unavailable: {detail}"
        by_path = {item["path"]: item["sha256"] for item in prepared}
        evaluations = result.get("evaluations")
        if not isinstance(evaluations, list) or not evaluations:
            return "Error: vision evaluator returned no per-image evaluations"
        normalized_evaluations: list[dict[str, Any]] = []
        for item in evaluations:
            if not isinstance(item, Mapping) or str(item.get("path") or "") not in by_path:
                return "Error: vision evaluator returned an unknown image path"
            normalized = dict(item)
            normalized["sha256"] = by_path[str(item["path"])]
            normalized_evaluations.append(normalized)
        selected = result.get("selected")
        if not isinstance(selected, list) or any(str(path) not in by_path for path in selected):
            return "Error: vision evaluator returned an invalid selected-image list"
        result["evaluations"] = normalized_evaluations
        with _LOCK:
            receipts = _EVIDENCE.setdefault(str(get_workspace()), {})
            for path in by_path:
                receipts.pop(path, None)
            receipts.update({str(path): by_path[str(path)] for path in selected})
        result["image_hashes"] = by_path
        return json.dumps(result, ensure_ascii=False)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        return f"Error: images could not be evaluated: {exc}"
