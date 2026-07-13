"""Install declared dependencies into project-local environments."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from ._security import get_workspace, resolve_workspace_path, safe_os_error
from .run_bash import _scrubbed_environment


REQUIRES_APPROVAL = True
MAX_OUTPUT = 20_000

SCHEMA = {
    "type": "function",
    "function": {
        "name": "install_dependencies",
        "description": (
            "Install dependencies declared by the project using a project-local environment. "
            "Auto-detects uv, Poetry, pip requirements, npm, pnpm, or Yarn; never installs a global package."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "directory": {"type": "string", "default": "."},
                "manager": {
                    "type": "string",
                    "enum": ["auto", "uv", "poetry", "pip", "npm", "pnpm", "yarn"],
                    "default": "auto",
                },
                "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 3600, "default": 1200},
            },
            "additionalProperties": False,
        },
    },
}


def _select(directory: Path, requested: str) -> tuple[str, list[str]]:
    python_venv = directory / ".venv"
    venv_python = python_venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    uv = shutil.which("uv")
    poetry = shutil.which("poetry")
    pnpm = shutil.which("pnpm")
    yarn = shutil.which("yarn")
    npm = shutil.which("npm")
    candidates: list[tuple[str, bool, list[str]]] = [
        ("uv", (directory / "uv.lock").exists() and uv is not None, [uv or "uv", "sync"]),
        ("poetry", (directory / "poetry.lock").exists() and poetry is not None, [poetry or "poetry", "install"]),
        ("pip", (directory / "requirements.txt").exists() or (directory / "pyproject.toml").exists(), []),
        ("pnpm", (directory / "pnpm-lock.yaml").exists() and pnpm is not None, [pnpm or "pnpm", "install"]),
        ("yarn", (directory / "yarn.lock").exists() and yarn is not None, [yarn or "yarn", "install"]),
        ("npm", (directory / "package.json").exists() and npm is not None, [npm or "npm", "install"]),
    ]
    if requested != "auto":
        selected = next((item for item in candidates if item[0] == requested), None)
        if selected is None or not selected[1]:
            raise ValueError(f"{requested} is unavailable or has no matching manifest")
    else:
        selected = next((item for item in candidates if item[1]), None)
        if selected is None:
            raise ValueError("no supported dependency manifest and installed manager were found")
    name, _available, command = selected
    if name == "pip":
        if not venv_python.exists():
            created = subprocess.run(
                [sys.executable, "-m", "venv", str(python_venv)],
                cwd=str(directory), env=_scrubbed_environment(),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                timeout=300, check=False,
            )
            if created.returncode != 0:
                detail = created.stdout.decode("utf-8", errors="replace")[-2_000:]
                raise ValueError("project virtual environment could not be created: " + detail)
        command = (
            [str(venv_python), "-m", "pip", "install", "-r", "requirements.txt"]
            if (directory / "requirements.txt").exists()
            else [str(venv_python), "-m", "pip", "install", "-e", "."]
        )
    return name, command


def run(directory: str = ".", manager: str = "auto", timeout_seconds: int = 1200) -> str:
    try:
        root = get_workspace() if (directory or ".").strip() in {"", "."} else resolve_workspace_path(directory, must_exist=True)
        if not root.is_dir():
            raise ValueError("dependency directory must be a directory")
        selected, command = _select(root, manager)
        completed = subprocess.run(
            command,
            cwd=str(root),
            env=_scrubbed_environment(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=max(1, min(int(timeout_seconds), 3600)),
            check=False,
        )
        output = completed.stdout.decode("utf-8", errors="replace")
        if len(output) > MAX_OUTPUT:
            output = output[-MAX_OUTPUT:]
        payload = {
            "manager": selected,
            "command": command,
            "exit_code": completed.returncode,
            "output": output,
        }
        if completed.returncode != 0:
            return "Error: dependency installation failed: " + json.dumps(payload, ensure_ascii=False)
        return json.dumps({"status": "installed", **payload}, ensure_ascii=False)
    except subprocess.TimeoutExpired:
        return f"Error: dependency installation timed out after {timeout_seconds} seconds"
    except (OSError, ValueError) as exc:
        return f"Error: dependencies could not be installed: {safe_os_error(exc) if isinstance(exc, OSError) else exc}"
