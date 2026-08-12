"""Install declared dependencies into project-local environments."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping

from ._security import get_workspace, resolve_workspace_path, safe_os_error
from .run_bash import _scrubbed_environment


REQUIRES_APPROVAL = True
MAX_OUTPUT = 20_000
_CONSERVATIVE_PYTHON_MINORS = ((3, 12), (3, 11), (3, 10))
_DISCOVERY_EXCLUDES = {
    ".git", ".coding-agent", ".venv", "node_modules", "dist", "build",
    "coverage", "vendor", "output", "target", "__pycache__",
}
_MANIFESTS_BY_MANAGER = {
    "uv": ("uv.lock",),
    "poetry": ("poetry.lock",),
    "pip": ("requirements.txt", "pyproject.toml"),
    "npm": ("package.json",),
    "pnpm": ("pnpm-lock.yaml", "package.json"),
    "yarn": ("yarn.lock", "package.json"),
}

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


def derived_mutation_paths(args: Mapping[str, Any]) -> tuple[str, ...]:
    """Declare only deterministic, project-local dependency side effects.

    Dependency trees and virtual environments are excluded from workspace
    evidence snapshots. Lockfiles are durable source artifacts, so they must be
    journaled and reviewed instead of being silently ignored.
    """

    raw_directory = str(args.get("directory") or ".").strip().replace("\\", "/")
    directory = "" if raw_directory in {"", "."} else raw_directory.strip("/")
    manager = str(args.get("manager") or "auto").strip().casefold()
    if not directory:
        try:
            candidates = dependency_directories(get_workspace(), manager)
        except (OSError, ValueError):
            candidates = ()
        if len(candidates) == 1 and candidates[0] != ".":
            directory = candidates[0]
    filenames: list[str] = []
    if manager in {"auto", "npm"}:
        filenames.append("package-lock.json")
    if manager == "pnpm":
        filenames.append("pnpm-lock.yaml")
    if manager == "yarn":
        filenames.append("yarn.lock")
    if manager == "uv":
        filenames.append("uv.lock")
    if manager == "poetry":
        filenames.append("poetry.lock")
    return tuple(f"{directory}/{name}" if directory else name for name in filenames)


def _python_runtime(directory: Path) -> list[str]:
    """Choose a broadly compatible local Python for legacy declared pins.

    Many inspected projects predate the agent host's Python.  Python 3.13, for
    example, cannot install several common 3.11-era binary pins.  Prefer the
    current runtime through 3.12; on newer hosts, use the newest installed
    conservative runtime without downloading or globally installing Python.
    """

    if sys.version_info[:2] <= (3, 12):
        return [sys.executable]
    if os.name == "nt" and shutil.which("py"):
        try:
            listed = subprocess.run(
                ["py", "-0p"],
                cwd=str(directory),
                env=_scrubbed_environment(),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=15,
                check=False,
            ).stdout.decode("utf-8", errors="replace")
        except (OSError, subprocess.SubprocessError):
            listed = ""
        for major, minor in _CONSERVATIVE_PYTHON_MINORS:
            if f"-{major}.{minor}" in listed or f"{major}.{minor}" in listed:
                return ["py", f"-{major}.{minor}"]
    for major, minor in _CONSERVATIVE_PYTHON_MINORS:
        executable = shutil.which(f"python{major}.{minor}")
        if executable:
            return [executable]
    return [sys.executable]


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
        manifests = _MANIFESTS_BY_MANAGER.get(requested, ())
        has_manifest = any((directory / name).is_file() for name in manifests)
        executable_available = (
            True if requested == "pip"
            else shutil.which(requested) is not None
        )
        if not has_manifest:
            expected = " or ".join(manifests) or "a supported manifest"
            raise ValueError(
                f"{requested} manifest not found in {directory}; expected {expected}. "
                "Pass directory for the component that owns the manifest."
            )
        if not executable_available:
            raise ValueError(
                f"{requested} executable is not available on PATH; the manifest exists at {directory}"
            )
        if selected is None or not selected[1]:
            raise ValueError(f"{requested} could not be selected for {directory}")
    else:
        selected = next((item for item in candidates if item[1]), None)
        if selected is None:
            declared = [
                manager_name
                for manager_name, manifests in _MANIFESTS_BY_MANAGER.items()
                if any((directory / name).is_file() for name in manifests)
            ]
            if declared:
                raise ValueError(
                    "declared dependency manager is unavailable on PATH: "
                    + ", ".join(dict.fromkeys(declared))
                )
            raise ValueError(
                f"no supported dependency manifest exists in {directory}; "
                "pass directory for a nested project component"
            )
    name, _available, command = selected
    if name == "pip":
        if not venv_python.exists():
            runtime = _python_runtime(directory)
            created = subprocess.run(
                [*runtime, "-m", "venv", str(python_venv)],
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


def dependency_directories(
    workspace: str | Path,
    manager: str = "auto",
    *,
    max_depth: int = 3,
) -> tuple[str, ...]:
    """Return bounded component directories that contain matching manifests.

    This is intentionally discovery-only.  A monorepo with more than one
    component must be selected explicitly rather than installing every package
    tree because an unqualified install can be both expensive and mutating.
    """

    root = Path(workspace).resolve(strict=False)
    requested = str(manager or "auto").strip().casefold()
    wanted = (
        tuple({name for values in _MANIFESTS_BY_MANAGER.values() for name in values})
        if requested == "auto"
        else _MANIFESTS_BY_MANAGER.get(requested, ())
    )
    if not wanted or not root.is_dir():
        return ()
    found: set[str] = set()
    for current, dirs, files in os.walk(root):
        current_path = Path(current)
        try:
            relative = current_path.relative_to(root)
        except ValueError:
            continue
        depth = len(relative.parts)
        dirs[:] = [
            name for name in dirs
            if name.casefold() not in _DISCOVERY_EXCLUDES and depth < max_depth
        ]
        if depth > max_depth or not set(files).intersection(wanted):
            continue
        found.add("." if not relative.parts else relative.as_posix())
    return tuple(sorted(found, key=lambda value: (value.count("/"), value.casefold())))


def dependency_applicability_issue(
    workspace: str | Path,
    directory: str = ".",
    manager: str = "auto",
) -> str:
    """Explain an ambiguous component before asking for installation approval."""

    raw = str(directory or ".").strip().replace("\\", "/")
    if raw not in {"", "."}:
        return ""
    candidates = dependency_directories(workspace, manager)
    if "." in candidates or len(candidates) <= 1:
        return ""
    quoted = ", ".join(repr(item) for item in candidates[:12])
    return (
        "dependency manifest is not at the workspace root and multiple components were found: "
        f"{quoted}. Call install_dependencies again with directory set to the component to run."
    )


def _declared_dependencies_ready(
    directory: Path,
    manager: str,
    command: list[str],
) -> tuple[bool, str]:
    """Avoid reinstalling a healthy existing JavaScript dependency tree."""

    if manager == "pip":
        venv_python = directory / ".venv" / (
            "Scripts/python.exe" if os.name == "nt" else "bin/python"
        )
        if not venv_python.is_file():
            return False, ""
        verification = [str(venv_python), "-m", "pip", "install", "--dry-run"]
        if (directory / "requirements.txt").is_file():
            verification.extend(("-r", "requirements.txt"))
        else:
            verification.extend(("-e", "."))
        try:
            completed = subprocess.run(
                verification,
                cwd=str(directory),
                env=_scrubbed_environment(),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=120,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return False, ""
        output = completed.stdout.decode("utf-8", errors="replace")[-4_000:]
        pending = "would install" in output.casefold()
        return completed.returncode == 0 and not pending, output
    if manager not in {"npm", "pnpm", "yarn"} or not (directory / "node_modules").is_dir():
        return False, ""
    executable = command[0]
    verification = (
        [executable, "ls", "--depth=0", "--silent"]
        if manager == "npm"
        else [executable, "list", "--depth", "0"]
    )
    try:
        completed = subprocess.run(
            verification,
            cwd=str(directory),
            env=_scrubbed_environment(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False, ""
    output = completed.stdout.decode("utf-8", errors="replace")[-2_000:]
    return completed.returncode == 0, output


def run(directory: str = ".", manager: str = "auto", timeout_seconds: int = 1200) -> str:
    try:
        root = get_workspace() if (directory or ".").strip() in {"", "."} else resolve_workspace_path(directory, must_exist=True)
        if not root.is_dir():
            raise ValueError("dependency directory must be a directory")
        candidates = dependency_directories(root, manager)
        if not any((root / name).is_file() for name in (
            tuple({item for values in _MANIFESTS_BY_MANAGER.values() for item in values})
            if manager == "auto" else _MANIFESTS_BY_MANAGER.get(manager, ())
        )):
            if len(candidates) == 1:
                root = (root / candidates[0]).resolve(strict=True)
            elif len(candidates) > 1:
                raise ValueError(
                    "multiple dependency components were found: "
                    + ", ".join(candidates)
                    + "; pass directory explicitly so only the intended component is changed"
                )
        selected, command = _select(root, manager)
        ready, verification_output = _declared_dependencies_ready(root, selected, command)
        if ready:
            return json.dumps(
                {
                    "status": "already_satisfied",
                    "manager": selected,
                    "directory": str(root),
                    "command": [],
                    "exit_code": 0,
                    "output": verification_output,
                    "reason": (
                        "Existing .venv satisfies the declared Python dependencies; no install was run."
                        if selected == "pip"
                        else "Existing node_modules passed the package manager integrity check; no install was run."
                    ),
                },
                ensure_ascii=False,
            )
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
            "directory": str(root),
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
