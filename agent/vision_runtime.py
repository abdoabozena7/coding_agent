"""Capability-probed visual evaluator selection for ordinary Action turns.

The coding model remains the task coordinator.  A screenshot request may use a
separate small visual evaluator only after that evaluator proves it can read
pixels.  Full/session-wide authority permits installing the configured local
Ollama fallback; Normal access never downloads a model implicitly.
"""

from __future__ import annotations

import json
import os
import urllib.request
from collections.abc import Callable, Iterable
from typing import Any

from .model_catalog import DEFAULT_OLLAMA_HOST, ModelCatalog, ModelDescriptor


DEFAULT_VISION_FALLBACK_MODEL = "qwen3-vl:4b"


def fallback_model_name(environ: dict[str, str] | None = None) -> str:
    values = os.environ if environ is None else environ
    return str(
        values.get("AGENT_VISION_MODEL") or DEFAULT_VISION_FALLBACK_MODEL
    ).strip()


def installed_vision_models(
    *,
    ollama_host: str = DEFAULT_OLLAMA_HOST,
    exclude: Iterable[str] = (),
) -> tuple[ModelDescriptor, ...]:
    rejected = {str(item).strip().casefold() for item in exclude}
    values = [
        descriptor
        for descriptor in ModelCatalog(ollama_host=ollama_host).discover()
        if "vision" in descriptor.capabilities
        and descriptor.id.casefold() not in rejected
        and descriptor.model.casefold() not in rejected
    ]
    return tuple(values)


def pull_ollama_vision_model(
    model: str,
    *,
    host: str = DEFAULT_OLLAMA_HOST,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
    timeout: float = 900.0,
) -> ModelDescriptor:
    """Pull one configured model through Ollama's progress-bearing HTTP API."""

    selected = str(model).strip()
    if not selected:
        raise ValueError("a visual fallback model name is required")
    endpoint = str(host or DEFAULT_OLLAMA_HOST).rstrip("/")
    request = urllib.request.Request(
        endpoint + "/api/pull",
        data=json.dumps({"model": selected, "stream": True}).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/x-ndjson"},
        method="POST",
    )
    last: dict[str, Any] = {}
    with urllib.request.urlopen(request, timeout=max(30.0, float(timeout))) as response:
        for raw in response:
            if not raw.strip():
                continue
            payload = json.loads(raw.decode("utf-8"))
            if not isinstance(payload, dict):
                continue
            last = payload
            if on_progress is not None:
                on_progress(dict(payload))
            if payload.get("error"):
                raise RuntimeError(str(payload["error"]))
    if str(last.get("status") or "").casefold() != "success":
        raise RuntimeError(
            "Ollama did not confirm visual model installation"
            + (f": {last.get('status')}" if last else "")
        )
    matches = [
        descriptor
        for descriptor in ModelCatalog(ollama_host=endpoint).discover()
        if descriptor.model.casefold() == selected.casefold()
        and "vision" in descriptor.capabilities
    ]
    if not matches:
        raise RuntimeError(
            f"installed model {selected!r} is not advertised as vision-capable"
        )
    return matches[0]
