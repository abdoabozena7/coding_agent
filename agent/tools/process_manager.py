"""Session-owned long-running process lifecycle with bounded logs."""

from __future__ import annotations

from dataclasses import dataclass, field
import atexit
import json
import os
from pathlib import Path
import socket
import subprocess
from threading import RLock, Thread
import time
from urllib.request import urlopen
import uuid

from ._security import get_workspace, resolve_workspace_path, safe_os_error
from .run_bash import MAX_COMMAND_CHARS, _scrubbed_environment, _terminate


MAX_LOG_BYTES = 1_000_000


@dataclass
class ManagedProcess:
    id: str
    command: str
    cwd: Path
    process: subprocess.Popen[bytes]
    log_path: Path
    log_handle: object
    reader: Thread | None = None
    started_at: float = field(default_factory=time.time)

    def snapshot(self) -> dict[str, object]:
        code = self.process.poll()
        return {
            "process_id": self.id,
            "pid": self.process.pid,
            "status": "running" if code is None else "exited",
            "exit_code": code,
            "command": self.command,
            "cwd": str(self.cwd),
            "log_path": str(self.log_path),
        }


_LOCK = RLock()
_PROCESSES: dict[tuple[str, str], ManagedProcess] = {}


def _drain_output(stream: object, handle: object) -> None:
    written = 0
    try:
        while True:
            reader = getattr(stream, "read1", None)
            chunk = reader(8_192) if callable(reader) else stream.read(8_192)  # type: ignore[attr-defined]
            if not chunk:
                return
            remaining = MAX_LOG_BYTES - written
            if remaining > 0:
                handle.write(chunk[:remaining])  # type: ignore[attr-defined]
                written += min(len(chunk), remaining)
    except (OSError, ValueError):
        return
    finally:
        try:
            stream.close()  # type: ignore[attr-defined]
        except (OSError, ValueError):
            pass


def _key(process_id: str) -> tuple[str, str]:
    return str(get_workspace()), process_id


def _cwd(value: str) -> Path:
    path = get_workspace() if (value or ".").strip() in {"", "."} else resolve_workspace_path(value, must_exist=True)
    if not path.is_dir():
        raise ValueError("process cwd must be a directory")
    return path


def _ready(item: ManagedProcess, readiness_type: str, readiness_value: str) -> bool:
    if item.process.poll() is not None:
        return False
    if readiness_type == "none":
        return True
    if readiness_type == "port":
        port = int(readiness_value)
        with socket.socket() as sock:
            sock.settimeout(0.25)
            return sock.connect_ex(("127.0.0.1", port)) == 0
    if readiness_type == "url":
        if not readiness_value.startswith(("http://127.0.0.1:", "http://localhost:")):
            raise ValueError("readiness URL must use loopback HTTP")
        try:
            with urlopen(readiness_value, timeout=0.5) as response:
                return 200 <= int(response.status) < 500
        except OSError:
            return False
    if readiness_type == "log":
        try:
            return readiness_value in item.log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return False
    raise ValueError("unknown readiness_type")


def start(
    command: str,
    cwd: str = ".",
    readiness_type: str = "none",
    readiness_value: str = "",
    timeout_seconds: int = 30,
) -> str:
    if not isinstance(command, str) or not command.strip():
        return "Error: command must be a non-empty string"
    if len(command) > MAX_COMMAND_CHARS or "\x00" in command:
        return "Error: command is invalid or too long"
    try:
        working = _cwd(cwd)
        state = get_workspace() / ".coding-agent" / "processes"
        state.mkdir(parents=True, exist_ok=True)
        process_id = "process-" + uuid.uuid4().hex[:16]
        log_path = state / f"{process_id}.log"
        handle = log_path.open("ab", buffering=0)
        options: dict[str, object] = {}
        if os.name == "nt":
            options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            options["start_new_session"] = True
        process = subprocess.Popen(
            command,
            shell=True,
            cwd=str(working),
            env=_scrubbed_environment(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            **options,
        )
        assert process.stdout is not None
        reader = Thread(target=_drain_output, args=(process.stdout, handle), name=f"{process_id}-log", daemon=True)
        item = ManagedProcess(process_id, command, working, process, log_path, handle, reader)
        reader.start()
        with _LOCK:
            _PROCESSES[_key(process_id)] = item
        deadline = time.monotonic() + max(0, min(int(timeout_seconds), 300))
        ready = False
        while time.monotonic() <= deadline:
            ready = _ready(item, readiness_type, readiness_value)
            if ready or process.poll() is not None:
                break
            time.sleep(0.1)
        payload = item.snapshot()
        payload["ready"] = ready
        if not ready:
            payload["output_tail"] = output(process_id, 80, raw=True)
            stop(process_id)
            return "Error: managed process did not become ready: " + json.dumps(payload, ensure_ascii=False)
        return json.dumps(payload, ensure_ascii=False)
    except (OSError, ValueError) as exc:
        return f"Error: process could not start: {safe_os_error(exc) if isinstance(exc, OSError) else exc}"


def poll(process_id: str) -> str:
    with _LOCK:
        item = _PROCESSES.get(_key(process_id))
    if item is None:
        return f"Error: unknown managed process {process_id!r}"
    return json.dumps(item.snapshot(), ensure_ascii=False)


def output(process_id: str, lines: int = 100, *, raw: bool = False) -> str:
    with _LOCK:
        item = _PROCESSES.get(_key(process_id))
    if item is None:
        return f"Error: unknown managed process {process_id!r}"
    try:
        data = item.log_path.read_bytes()
        if len(data) > MAX_LOG_BYTES:
            data = data[-MAX_LOG_BYTES:]
        text = data.decode("utf-8", errors="replace")
        result = "\n".join(text.splitlines()[-max(1, min(int(lines), 2_000)):])
        return result if raw else json.dumps({"process_id": process_id, "output": result}, ensure_ascii=False)
    except OSError as exc:
        return f"Error: process log could not be read: {safe_os_error(exc)}"


def stop(process_id: str) -> str:
    with _LOCK:
        item = _PROCESSES.pop(_key(process_id), None)
    if item is None:
        return f"Error: unknown managed process {process_id!r}"
    if item.process.poll() is None:
        _terminate(item.process)
        try:
            item.process.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            pass
    if item.reader is not None:
        item.reader.join(timeout=2)
    try:
        item.log_handle.close()
    except OSError:
        pass
    payload = item.snapshot()
    payload["stopped"] = True
    return json.dumps(payload, ensure_ascii=False)


def list_processes() -> tuple[dict[str, object], ...]:
    root = str(get_workspace())
    with _LOCK:
        return tuple(item.snapshot() for (workspace, _), item in _PROCESSES.items() if workspace == root)


def shutdown_workspace(workspace: str | Path) -> None:
    root = str(Path(workspace).resolve())
    with _LOCK:
        ids = [process_id for (owner, process_id) in _PROCESSES if owner == root]
    for process_id in ids:
        try:
            key = (root, process_id)
            with _LOCK:
                item = _PROCESSES.pop(key, None)
            if item is not None:
                if item.process.poll() is None:
                    _terminate(item.process)
                if item.reader is not None:
                    item.reader.join(timeout=2)
                item.log_handle.close()
        except (OSError, ValueError):
            continue


def _shutdown_all() -> None:
    with _LOCK:
        workspaces = {owner for owner, _ in _PROCESSES}
    for workspace in workspaces:
        shutdown_workspace(workspace)


atexit.register(_shutdown_all)
