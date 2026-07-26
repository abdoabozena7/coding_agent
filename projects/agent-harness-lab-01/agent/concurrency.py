"""Conservative model-aware worker concurrency selection."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Mapping

from .model_catalog import ExecutionClass, ModelDescriptor


@dataclass(frozen=True, slots=True)
class ConcurrencyCapacity:
    safe_max: int
    recommended: int
    reason: str


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _physical_cores() -> int | None:
    try:
        import psutil  # type: ignore

        return _positive_int(psutil.cpu_count(logical=False))
    except (ImportError, OSError, RuntimeError):
        return None


def _total_ram_bytes() -> int | None:
    try:
        import psutil  # type: ignore

        return _positive_int(psutil.virtual_memory().total)
    except (ImportError, OSError, RuntimeError):
        return None


def _memory_bytes(value: Any) -> int | None:
    direct = _positive_int(value)
    if direct is not None:
        return direct
    text = str(value or "").strip().casefold().replace(",", "")
    match = re.search(r"(\d+(?:\.\d+)?)\s*(gib|gb|mib|mb)", text)
    if not match:
        return None
    amount = float(match.group(1))
    unit = match.group(2)
    multiplier = 1024**3 if unit in {"gib", "gb"} else 1024**2
    return int(amount * multiplier)


def probe_concurrency(
    descriptor: ModelDescriptor,
    *,
    provider_cap: int | None = None,
    environ: Mapping[str, str] | None = None,
) -> ConcurrencyCapacity:
    """Return a safe cap; incomplete local telemetry intentionally means one."""

    env = os.environ if environ is None else environ
    metadata = dict(descriptor.metadata)
    if descriptor.execution_class is ExecutionClass.CLOUD:
        configured = _positive_int(
            provider_cap
            or metadata.get("provider_concurrency_cap")
            or env.get("AGENT_PROVIDER_CONCURRENCY_CAP")
            or 8
        ) or 1
        safe = max(1, min(8, configured))
        return ConcurrencyCapacity(
            safe,
            min(2, safe),
            f"cloud provider/config cap permits up to {safe}",
        )

    declared = _positive_int(
        metadata.get("parallelism")
        or metadata.get("max_parallel")
        or env.get("OLLAMA_NUM_PARALLEL")
    )
    model_bytes = _positive_int(metadata.get("size_bytes"))
    cores = _physical_cores()
    ram_bytes = _total_ram_bytes()
    if None in {declared, model_bytes, cores, ram_bytes}:
        return ConcurrencyCapacity(
            1,
            1,
            "local parallelism, physical-core, RAM, or model-size telemetry is incomplete",
        )

    # Reserve 30% of RAM and an extra 25% of model size per worker for context.
    assert declared is not None and model_bytes is not None
    assert cores is not None and ram_bytes is not None
    per_worker = max(1, int(model_bytes * 1.25))
    ram_cap = int((ram_bytes * 0.70) // per_worker)
    safe = min(8, declared, max(1, cores // 2), max(1, ram_cap))

    gpu_memory = _memory_bytes(
        metadata.get("vram_bytes")
        or metadata.get("gpu_memory")
        or env.get("AGENT_GPU_VRAM_BYTES")
    )
    if metadata.get("gpu_required") and gpu_memory is None:
        return ConcurrencyCapacity(1, 1, "local GPU memory telemetry is incomplete")
    if gpu_memory is not None:
        safe = min(safe, max(1, int((gpu_memory * 0.70) // per_worker)))
    safe = max(1, safe)
    return ConcurrencyCapacity(
        safe,
        min(2, safe),
        f"local model fits {safe} isolated context worker(s) after the 30% reserve",
    )
