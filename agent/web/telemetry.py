"""Honest local-work progress estimates and bounded host resource telemetry."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import ctypes
import json
import math
import os
import statistics
import subprocess
from typing import Any, Iterable, Mapping


DEFAULT_INFERENCE_PROFILE: dict[str, Any] = {
    "device": "auto",
    "context_window": 16_384,
    "max_output_tokens": 4_096,
    "gpu_layers": -1,
    "cpu_threads": max(1, min(64, (os.cpu_count() or 4))),
    "temperature": 0.2,
    "top_p": 0.9,
    "top_k": 40,
    "performance": "balanced",
    "estimated_minutes_per_step": 30,
    "planning_steps": 16,
    "work_quantum_steps": 24,
    "review_steps": 12,
    "max_provider_retries": 3,
    "ultra_cloud_concurrency": 4,
    "ultra_max_depth": 8,
}


def _parse_time(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _duration(job: Mapping[str, Any]) -> float | None:
    started = _parse_time(job.get("started_at"))
    ended = _parse_time(job.get("completed_at"))
    if started is None or ended is None or ended <= started:
        return None
    return (ended - started).total_seconds()


def progress_estimate(
    dashboard: Mapping[str, Any] | None,
    presentation: Mapping[str, Any] | None,
    job: Mapping[str, Any] | None,
    history: Iterable[Mapping[str, Any]],
    *,
    workflow_mode: str,
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a range, never a fabricated single-point promise.

    Early estimates use the user's calibrated local step duration. Once work
    completes milestones, elapsed milestone velocity replaces that baseline.
    Historical completed turns are a secondary signal for pre-plan work.
    """

    board = dict(dashboard) if isinstance(dashboard, Mapping) else {}
    view = dict(presentation) if isinstance(presentation, Mapping) else {}
    tasks = list(board.get("tasks") or [])
    completed_states = {"done", "completed", "skipped"}
    active_states = {"running", "in_progress"}
    completed = sum(str(task.get("status", "")).casefold() in completed_states for task in tasks)
    total = len(tasks)
    current_task = next(
        (str(task.get("title") or "Current step") for task in tasks if str(task.get("status", "")).casefold() in active_states),
        "",
    )
    status = str(board.get("status") or view.get("status") or "idle")
    if status == "completed":
        percent = 100
    elif total:
        # Reserve a small visible portion for the active step without claiming
        # it is complete. This keeps progress monotonic at milestone boundaries.
        active_fraction = 0.35 if current_task else 0.0
        percent = min(99, round(100 * (completed + active_fraction) / total))
    else:
        stage = str((view.get("activity") or {}).get("stage") or "idle")
        percent = {
            "understanding": 4,
            "planning": 8,
            "waiting": 10,
            "executing": 18,
            "checking": 88,
            "done": 100,
        }.get(stage, 0)

    started = _parse_time((job or {}).get("started_at") or (job or {}).get("created_at"))
    now = datetime.now(timezone.utc)
    elapsed = max(0.0, (now - started).total_seconds()) if started else 0.0
    calibrated_minutes = max(1, min(720, int(profile.get("estimated_minutes_per_step", 30) or 30)))
    mode_factor = {"plan": 0.45, "normal": 1.0, "ultra": 1.8}.get(str(workflow_mode), 1.0)
    confidence = "low"
    basis = "calibrated local step baseline"
    if total and completed > 0 and elapsed > 0:
        projected_total = elapsed * total / completed
        confidence = "medium" if completed < max(2, math.ceil(total / 2)) else "high"
        basis = "observed milestone velocity"
    else:
        completed_durations = [value for value in (_duration(item) for item in history) if value]
        if not total and completed_durations:
            projected_total = statistics.median(completed_durations[-12:])
            confidence = "medium" if len(completed_durations) >= 3 else "low"
            basis = "recent completed local turns"
        else:
            projected_total = calibrated_minutes * 60 * max(1, total or 1) * mode_factor

    remaining = max(0.0, projected_total - elapsed)
    low_factor, high_factor = (0.85, 1.2) if confidence == "high" else (0.7, 1.45) if confidence == "medium" else (0.55, 1.8)
    low = int(remaining * low_factor)
    high = int(remaining * high_factor)
    running = str((job or {}).get("status") or "") == "running"
    finish = now + timedelta(seconds=(low + high) / 2) if running and status != "completed" else None
    milestones = [
        {
            "id": str(task.get("id") or index),
            "title": str(task.get("title") or f"Step {index + 1}"),
            "status": str(task.get("status") or "pending"),
        }
        for index, task in enumerate(tasks)
    ]
    return {
        "percent": percent,
        "completed_steps": completed,
        "total_steps": total,
        "current_step": current_task or str((view.get("activity") or {}).get("summary") or status.replace("_", " ")),
        "elapsed_seconds": int(elapsed),
        "remaining_seconds_low": low,
        "remaining_seconds_high": high,
        "estimated_finish_at": finish.isoformat(timespec="seconds") if finish else None,
        "confidence": confidence,
        "basis": basis,
        "paused_for_attention": bool(view.get("attention")) or status in {
            "awaiting_plan_approval", "waiting_for_input", "paused"
        },
        "milestones": milestones,
    }


def _memory_snapshot() -> dict[str, Any]:
    try:
        import psutil  # type: ignore[import-not-found]

        memory = psutil.virtual_memory()
        return {
            "used_bytes": int(memory.used),
            "total_bytes": int(memory.total),
            "percent": round(float(memory.percent), 1),
        }
    except (ImportError, OSError):
        pass
    if os.name == "nt":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                ("total_phys", ctypes.c_ulonglong), ("avail_phys", ctypes.c_ulonglong),
                ("total_page", ctypes.c_ulonglong), ("avail_page", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong), ("avail_virtual", ctypes.c_ulonglong),
                ("avail_extended", ctypes.c_ulonglong),
            ]

        state = MemoryStatus()
        state.length = ctypes.sizeof(state)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state)):
            used = int(state.total_phys - state.avail_phys)
            return {"used_bytes": used, "total_bytes": int(state.total_phys), "percent": round(100 * used / max(1, state.total_phys), 1)}
    return {"used_bytes": None, "total_bytes": None, "percent": None}


def _gpu_snapshot() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return []
    if completed.returncode != 0:
        return []
    result: list[dict[str, Any]] = []
    for index, line in enumerate(completed.stdout.splitlines()):
        values = [item.strip() for item in line.split(",")]
        if len(values) != 5:
            continue
        try:
            used_mib, total_mib = float(values[2]), float(values[3])
            result.append({
                "index": index,
                "name": values[0],
                "utilization_percent": float(values[1]),
                "used_bytes": int(used_mib * 1024 * 1024),
                "total_bytes": int(total_mib * 1024 * 1024),
                "percent": round(100 * used_mib / max(1.0, total_mib), 1),
                "temperature_c": float(values[4]),
            })
        except ValueError:
            continue
    return result


def resource_snapshot(*, context_used: int | None, context_limit: int | None) -> dict[str, Any]:
    used = max(0, int(context_used or 0))
    limit = max(0, int(context_limit or 0))
    cpu_percent: float | None = None
    try:
        import psutil  # type: ignore[import-not-found]

        cpu_percent = round(float(psutil.cpu_percent(interval=None)), 1)
    except (ImportError, OSError):
        cpu_percent = None
    return {
        "ram": _memory_snapshot(),
        "gpus": _gpu_snapshot(),
        "cpu": {"logical_cores": os.cpu_count(), "utilization_percent": cpu_percent},
        "context": {
            "used_tokens": used if limit else None,
            "limit_tokens": limit or None,
            "remaining_tokens": max(0, limit - used) if limit else None,
            "percent": round(100 * used / limit, 1) if limit else None,
            "source": "latest provider usage" if limit else "provider did not expose a context window",
        },
        "sampled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def latest_context_usage(activity: Iterable[Mapping[str, Any]]) -> int | None:
    for item in reversed(list(activity)):
        if str(item.get("kind")) != "usage":
            continue
        try:
            details = json.loads(str(item.get("details") or "{}"))
            return int(details.get("input_tokens") or 0) + int(details.get("output_tokens") or 0)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
    return None


__all__ = [
    "DEFAULT_INFERENCE_PROFILE",
    "latest_context_usage",
    "progress_estimate",
    "resource_snapshot",
]
