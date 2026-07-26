"""Open a workspace file with its platform-associated application."""

from __future__ import annotations

import os
import subprocess
import sys

from ._security import resolve_workspace_path, safe_os_error

REQUIRES_APPROVAL = True
SCHEMA = {"type":"function","function":{"name":"open_path","description":"Open an existing non-sensitive file inside the workspace using the OS default application.","parameters":{"type":"object","properties":{"path":{"type":"string","minLength":1,"maxLength":4096}},"required":["path"],"additionalProperties":False}}}

def run(path: str) -> str:
    try:
        target = resolve_workspace_path(path, must_exist=True)
        if target.is_dir():
            return "Error: open_path accepts files, not directories"
        if os.name == "nt":
            os.startfile(str(target))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(target)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            subprocess.Popen(["xdg-open", str(target)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return f"Opened {path} with the platform default application"
    except OSError as exc:
        return f"Error: file could not be opened: {safe_os_error(exc)}"
