"""Access-aware local terminal sessions for the web workspace."""

from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
from threading import RLock
from typing import Any, Callable

from ..sandbox import DockerSandbox
from .registry import WebRegistry


_READ_ONLY_PREFIXES = (
    "dir", "ls", "pwd", "cd", "get-childitem", "get-content", "type ",
    "rg ", "git status", "git diff", "git log", "git show", "git branch",
    "python -m pytest", "pytest", "npm test", "npm run test", "npm run typecheck",
)
_NEVER_ALLOWED = re.compile(
    r"(?i)(?:format\s+[a-z]:|diskpart|shutdown(?:\.exe)?|restart-computer|stop-computer|"
    r"remove-item\s+(?:-[^ ]+\s+)*(?:[a-z]:\\|/)|(?:rd|rmdir|del)\s+/s\s+(?:[a-z]:\\|\\)|"
    r"rm\s+-rf\s+(?:/|~|\$home))"
)
_MANAGED_SHELL_SYNTAX = re.compile(r"(?:&&|\|\||[|;&><`]|\$\()")
_CHANGE_DIRECTORY = re.compile(r"(?i)^(?:cd|set-location)\s+(.+)$")


class TerminalPolicyError(RuntimeError):
    pass


class TerminalManager:
    """Durable command history with a deliberately narrow default boundary.

    Docker Full uses the existing hardened sandbox. Host Full may execute an
    arbitrary workspace command, but immutable destructive-target guards still
    apply. Default and Bounded accept inspection and verification commands only.
    """

    def __init__(
        self,
        registry: WebRegistry,
        effective_access: Callable[[str], str],
    ) -> None:
        self.registry = registry
        self.effective_access = effective_access
        self._lock = RLock()
        self._running: dict[str, subprocess.Popen[str]] = {}

    def open(self, thread_id: str, workspace: Path) -> dict[str, Any]:
        access = self.effective_access(thread_id)
        mode = "docker" if access == "full" else "host" if access == "host" else "managed"
        return self.registry.create_terminal_session(thread_id, mode, str(workspace.resolve(strict=True)))

    def execute(self, session_id: str, command: str) -> dict[str, Any]:
        session = self.registry.get_terminal_session(session_id)
        if session is None or session.get("status") != "open":
            raise KeyError("terminal session not found")
        clean = str(command).strip()
        if not clean or "\x00" in clean:
            raise ValueError("terminal command is empty or malformed")
        if _NEVER_ALLOWED.search(clean):
            raise TerminalPolicyError("This command targets a protected system or root path.")
        workspace = Path(str(session["cwd"])).resolve(strict=True)
        access = self.effective_access(str(session["thread_id"]))
        directory_change = _CHANGE_DIRECTORY.fullmatch(clean)
        if directory_change:
            thread = self.registry.get_thread(str(session["thread_id"]))
            project = self.registry.get_project(str((thread or {}).get("project_id") or ""))
            if project is None:
                raise TerminalPolicyError("The terminal project no longer exists.")
            project_root = Path(str(project["path"])).resolve(strict=True)
            raw_target = directory_change.group(1).strip().strip('"').strip("'")
            candidate = Path(raw_target)
            target = (candidate if candidate.is_absolute() else workspace / candidate).resolve(strict=True)
            if not target.is_dir() or (target != project_root and project_root not in target.parents):
                raise TerminalPolicyError("Terminal folders must stay inside the selected workspace.")
            history = [*list(session.get("history") or []), clean][-200:]
            block = f"> {clean}\n{target}\n[exit 0]\n"
            updated = self.registry.update_terminal_session(
                session_id, cwd=str(target), history=history,
                scrollback=str(session.get("scrollback") or "") + block,
            )
            return {**updated, "output": str(target), "returncode": 0}
        if access not in {"full", "host"} and not clean.casefold().startswith(_READ_ONLY_PREFIXES):
            raise TerminalPolicyError(
                "Default and Bounded terminals allow inspection and verification commands only. "
                "Use the agent approval flow or explicitly enable Docker Full/Host Full for mutations."
            )
        if access not in {"full", "host"} and _MANAGED_SHELL_SYNTAX.search(clean):
            raise TerminalPolicyError(
                "Command chaining, redirects, substitutions, and pipelines are disabled in Default and Bounded terminals."
            )
        if access == "full":
            result = DockerSandbox().run(clean, workspace)
            output = result.render()
            returncode = result.returncode
        else:
            completed = subprocess.run(
                clean,
                cwd=workspace,
                shell=True,
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                env=dict(os.environ),
            )
            output = (completed.stdout or "") + (completed.stderr or "")
            returncode = int(completed.returncode)
        history = [*list(session.get("history") or []), clean][-200:]
        block = f"> {clean}\n{output or '(no output)'}\n[exit {returncode}]\n"
        updated = self.registry.update_terminal_session(
            session_id,
            history=history,
            scrollback=str(session.get("scrollback") or "") + block,
        )
        return {**updated, "output": output[-200_000:], "returncode": returncode}

    def close(self, session_id: str) -> dict[str, Any]:
        return self.registry.update_terminal_session(session_id, status="closed")
