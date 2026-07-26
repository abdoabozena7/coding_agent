"""Local hardware probes used to avoid misleading weak-model benchmarks.

The probe is deliberately conservative: if the user marks local inference as
GPU-required, unknown hardware is treated as unavailable rather than silently
falling back to CPU.
"""

from __future__ import annotations

import os
import json
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class HardwareProbeResult:
    gpu_available: bool
    source: str
    devices: tuple[Mapping[str, str], ...] = field(default_factory=tuple)
    message: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "gpu_available": self.gpu_available,
            "source": self.source,
            "devices": [dict(item) for item in self.devices],
            "message": self.message,
        }


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().casefold() in {"1", "true", "yes", "on", "required", "gpu"}


def _first_available_command(*names: str) -> str | None:
    for name in names:
        found = shutil.which(name)
        if found:
            return found
    return None


def _text(value: object) -> str:
    return str(value or "").strip()


_REJECTED_WINDOWS_ADAPTER_MARKERS = (
    "microsoft basic display",
    "basic render",
    "remote display",
    "virtual",
    "vmware",
    "parallels",
    "hyper-v",
    "citrix",
    "mirage",
    "vnc",
)

_KNOWN_GPU_FAMILY_MARKERS = (
    "nvidia",
    "geforce",
    "quadro",
    "rtx",
    "gtx",
    "amd",
    "advanced micro devices",
    "radeon",
    "intel arc",
    "intel(r) arc",
    "arc(tm)",
)


def _is_usable_windows_adapter(adapter: Mapping[str, Any]) -> bool:
    name = _text(adapter.get("Name"))
    vendor = _text(adapter.get("AdapterCompatibility"))
    status = _text(adapter.get("Status")).casefold()
    combined = f"{name} {vendor}".casefold()
    if not name:
        return False
    if status and status not in {"ok", "unknown"}:
        return False
    if any(marker in combined for marker in _REJECTED_WINDOWS_ADAPTER_MARKERS):
        return False
    return any(marker in combined for marker in _KNOWN_GPU_FAMILY_MARKERS)


def _windows_adapter_to_device(adapter: Mapping[str, Any]) -> dict[str, str]:
    return {
        "name": _text(adapter.get("Name")),
        "vendor": _text(adapter.get("AdapterCompatibility")),
        "driver": _text(adapter.get("DriverVersion")),
        "memory": _text(adapter.get("AdapterRAM")),
        "status": _text(adapter.get("Status")),
    }


def _probe_windows_display_adapters() -> HardwareProbeResult:
    powershell = _first_available_command("powershell", "powershell.exe", "pwsh", "pwsh.exe")
    if not powershell:
        return HardwareProbeResult(False, "win32-video-controller", message="PowerShell is unavailable")
    try:
        completed = subprocess.run(
            [
                powershell,
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                (
                    "Get-CimInstance Win32_VideoController | "
                    "Select-Object Name,AdapterCompatibility,DriverVersion,AdapterRAM,Status | "
                    "ConvertTo-Json -Compress"
                ),
            ],
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return HardwareProbeResult(False, "win32-video-controller", message=str(exc))
    if completed.returncode != 0:
        return HardwareProbeResult(
            False,
            "win32-video-controller",
            message=(completed.stderr or completed.stdout).strip(),
        )
    try:
        parsed = json.loads(completed.stdout or "[]")
    except json.JSONDecodeError as exc:
        return HardwareProbeResult(False, "win32-video-controller", message=f"Invalid adapter JSON: {exc}")
    adapters = parsed if isinstance(parsed, list) else [parsed]
    usable_devices = [
        _windows_adapter_to_device(adapter)
        for adapter in adapters
        if isinstance(adapter, Mapping) and _is_usable_windows_adapter(adapter)
    ]
    if usable_devices:
        return HardwareProbeResult(
            True,
            "win32-video-controller",
            tuple(usable_devices),
            "Windows display GPU detected",
        )
    inspected = [
        _windows_adapter_to_device(adapter)
        for adapter in adapters
        if isinstance(adapter, Mapping)
    ]
    return HardwareProbeResult(
        False,
        "win32-video-controller",
        tuple(inspected),
        "No usable GPU adapter detected; only unsupported/basic/virtual adapters were found",
    )


def probe_local_gpu(environ: Mapping[str, str] | None = None) -> HardwareProbeResult:
    env = dict(os.environ if environ is None else environ)
    forced = env.get("AGENT_GPU_AVAILABLE")
    if forced is not None:
        available = _truthy(forced)
        name = env.get("AGENT_GPU_NAME", "configured-gpu" if available else "")
        return HardwareProbeResult(
            gpu_available=available,
            source="env",
            devices=({"name": name, "driver": env.get("AGENT_GPU_DRIVER", "")},) if available else (),
            message="GPU availability supplied by AGENT_GPU_AVAILABLE",
        )

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            completed = subprocess.run(
                [
                    nvidia_smi,
                    "--query-gpu=name,driver_version,memory.total",
                    "--format=csv,noheader",
                ],
                text=True,
                capture_output=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return HardwareProbeResult(False, "nvidia-smi", message=str(exc))
        if completed.returncode == 0:
            devices = []
            for line in completed.stdout.splitlines():
                parts = [part.strip() for part in line.split(",")]
                if parts and parts[0]:
                    devices.append(
                        {
                            "name": parts[0],
                            "driver": parts[1] if len(parts) > 1 else "",
                            "memory": parts[2] if len(parts) > 2 else "",
                        }
                    )
            if devices:
                return HardwareProbeResult(True, "nvidia-smi", tuple(devices), "NVIDIA GPU detected")
        return HardwareProbeResult(False, "nvidia-smi", message=(completed.stderr or completed.stdout).strip())

    windows_probe = _probe_windows_display_adapters()
    if windows_probe.gpu_available or windows_probe.devices:
        return windows_probe

    return HardwareProbeResult(False, "probe", message="No supported local GPU probe found")
