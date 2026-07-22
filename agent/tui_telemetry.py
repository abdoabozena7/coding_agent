"""Non-blocking resource telemetry for the persistent terminal workspace."""

from __future__ import annotations

import csv
import io
import shutil
import subprocess
import time
from dataclasses import dataclass
from threading import Event, Thread
from typing import Callable

try:
    import psutil
except ImportError:  # pragma: no cover - requirements install it; fallback stays safe.
    psutil = None  # type: ignore[assignment]


@dataclass(frozen=True, slots=True)
class TelemetrySample:
    cpu_percent: float | None = None
    process_memory_mib: float | None = None
    memory_percent: float | None = None
    memory_used_gib: float | None = None
    memory_total_gib: float | None = None
    gpu_percent: float | None = None
    gpu_memory_used_mib: float | None = None
    gpu_memory_total_mib: float | None = None
    gpu_label: str = ""
    gpu_available: bool = False
    sampled_at: float | None = None

    def as_update(self) -> dict[str, object]:
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        }


def _float(value: object) -> float | None:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def probe_gpu(timeout: float = 1.0) -> dict[str, object]:
    """Return measurable GPU load without making availability claims from guesses."""

    nvidia = shutil.which("nvidia-smi")
    if nvidia:
        try:
            completed = subprocess.run(
                [
                    nvidia,
                    "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=max(0.1, float(timeout)),
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            completed = None
        if completed is not None and completed.returncode == 0:
            rows = list(csv.reader(io.StringIO(completed.stdout)))
            if rows and len(rows[0]) >= 4:
                name, load, used, total = (item.strip() for item in rows[0][:4])
                return {
                    "gpu_available": True,
                    "gpu_label": name,
                    "gpu_percent": _float(load),
                    "gpu_memory_used_mib": _float(used),
                    "gpu_memory_total_mib": _float(total),
                }

    rocm = shutil.which("rocm-smi")
    if rocm:
        try:
            completed = subprocess.run(
                [rocm, "--showuse", "--showmemuse", "--json"],
                capture_output=True,
                text=True,
                timeout=max(0.1, float(timeout)),
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            completed = None
        if completed is not None and completed.returncode == 0:
            # ROCm JSON differs by release. Extract the first numeric values
            # conservatively and leave memory units unknown rather than lying.
            import json

            try:
                parsed = json.loads(completed.stdout)
                device = next(iter(parsed.values())) if isinstance(parsed, dict) else {}
            except (json.JSONDecodeError, StopIteration):
                device = {}
            if isinstance(device, dict):
                load = next(
                    (_float(value) for key, value in device.items() if "use" in key.casefold()),
                    None,
                )
                return {
                    "gpu_available": True,
                    "gpu_label": "AMD GPU",
                    "gpu_percent": load,
                }
    return {"gpu_available": False}


def sample_system(gpu: dict[str, object] | None = None) -> TelemetrySample:
    values: dict[str, object] = dict(gpu or {})
    if psutil is not None:
        try:
            memory = psutil.virtual_memory()
            process = psutil.Process()
            values.update(
                cpu_percent=float(psutil.cpu_percent(interval=None)),
                process_memory_mib=process.memory_info().rss / (1024 * 1024),
                memory_percent=float(memory.percent),
                memory_used_gib=(memory.total - memory.available) / (1024**3),
                memory_total_gib=memory.total / (1024**3),
            )
        except (OSError, RuntimeError, AttributeError):
            pass
    values["sampled_at"] = time.monotonic()
    allowed = TelemetrySample.__dataclass_fields__
    return TelemetrySample(**{key: value for key, value in values.items() if key in allowed})


class TelemetrySampler:
    """Sample resources on a daemon thread and publish immutable snapshots."""

    def __init__(
        self,
        callback: Callable[[TelemetrySample], None],
        *,
        interval: float = 1.0,
        gpu_interval: float = 5.0,
        gpu_timeout: float = 1.0,
    ) -> None:
        self.callback = callback
        self.interval = max(0.1, float(interval))
        self.gpu_interval = max(self.interval, float(gpu_interval))
        self.gpu_timeout = max(0.1, float(gpu_timeout))
        self._stop = Event()
        self._thread: Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = Thread(target=self._run, name="ga3bad-tui-telemetry", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(0.0, float(timeout)))
        self._thread = None

    def _run(self) -> None:
        gpu: dict[str, object] = {"gpu_available": False}
        next_gpu = 0.0
        while not self._stop.is_set():
            now = time.monotonic()
            if now >= next_gpu:
                gpu = probe_gpu(self.gpu_timeout)
                next_gpu = now + self.gpu_interval
            try:
                self.callback(sample_system(gpu))
            except Exception:
                # Telemetry is optional and must never take down the workspace.
                pass
            self._stop.wait(self.interval)


__all__ = ["TelemetrySample", "TelemetrySampler", "probe_gpu", "sample_system"]
