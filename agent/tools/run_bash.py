"""Approval-gated shell execution with bounded output and a scrubbed context."""

from __future__ import annotations

import os
import signal
import subprocess
import threading
from dataclasses import dataclass, field
from typing import BinaryIO

from ._security import get_workspace, safe_os_error


TIMEOUT_SECONDS = 120
MAX_OUTPUT_CHARS = 10_000
MAX_CAPTURE_BYTES = 10_000
MAX_COMMAND_CHARS = 32_768

REQUIRES_APPROVAL = True

# Shell children receive only operational values needed to locate executables,
# temporary storage, and the user's home.  API keys, auth tokens, cookies,
# dotenv configuration, Python injection variables, and application secrets are
# absent by construction rather than relying on an incomplete deny-list.
_ENV_ALLOWLIST = {
    "path", "pathext", "systemroot", "windir", "comspec",
    "temp", "tmp", "tmpdir",
    "home", "userprofile", "homedrive", "homepath",
    "lang", "language", "lc_all", "lc_ctype", "term", "colorterm",
    "os", "processor_architecture", "number_of_processors",
}


SCHEMA = {
    "type": "function",
    "function": {
        "name": "run_bash",
        "description": (
            "Run an approval-gated platform shell command in the active workspace "
            "(cmd.exe on Windows, /bin/sh on POSIX; despite the legacy tool name) and "
            "return its stdout, stderr, and exit code. Secret environment "
            "variables are not inherited."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": MAX_COMMAND_CHARS,
                    "description": "The shell command to run.",
                },
            },
            "required": ["command"],
            "additionalProperties": False,
        },
    },
}


def _is_safe(command: str) -> bool:
    """Shell text is never safe enough to bypass the human permission gate.

    Even commands marketed as read-only can execute code (test discovery and
    Python), dump secrets (``env``), invoke pager hooks (git), or read arbitrary
    file paths.  Dedicated file tools provide the genuinely safe read surface.
    """

    return False


def requires_approval(args: dict) -> bool:
    return True


def _scrubbed_environment(source: dict[str, str] | None = None) -> dict[str, str]:
    source = os.environ if source is None else source
    return {key: value for key, value in source.items() if key.casefold() in _ENV_ALLOWLIST}


@dataclass
class _Capture:
    data: bytearray = field(default_factory=bytearray)
    truncated: bool = False


def _drain(stream: BinaryIO, capture: _Capture) -> None:
    try:
        while True:
            chunk = stream.read(8_192)
            if not chunk:
                return
            remaining = MAX_CAPTURE_BYTES - len(capture.data)
            if remaining > 0:
                capture.data.extend(chunk[:remaining])
            if len(chunk) > remaining:
                capture.truncated = True
    except (OSError, ValueError):
        return
    finally:
        try:
            stream.close()
        except OSError:
            pass


def _terminate(process: subprocess.Popen[bytes]) -> None:
    try:
        if os.name != "nt":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            # CREATE_NEW_PROCESS_GROUP alone does not make Popen.kill terminate
            # grandchildren. taskkill /T closes the full approved command tree.
            terminated = subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=_scrubbed_environment(),
                timeout=5,
                check=False,
            )
            if terminated.returncode != 0:
                process.kill()
    except (OSError, ProcessLookupError, subprocess.TimeoutExpired):
        try:
            process.kill()
        except OSError:
            pass


def _decode(capture: _Capture) -> str:
    return bytes(capture.data).decode("utf-8", errors="replace")


def _truncate(text: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    marker = f"\n... (truncated at {MAX_OUTPUT_CHARS} characters)"
    if len(marker) >= MAX_OUTPUT_CHARS:
        return marker[:MAX_OUTPUT_CHARS]
    kept = max(0, MAX_OUTPUT_CHARS - len(marker))
    return text[:kept] + marker


def _format_result(returncode: int, stdout: _Capture, stderr: _Capture) -> str:
    parts = [f"exit code: {returncode}"]

    for label, capture in (("stdout", stdout), ("stderr", stderr)):
        text = _decode(capture)
        if not text and not capture.truncated:
            continue
        section = f"{label}:\n{text}"
        if capture.truncated:
            section += "\n... (output truncated)"
        parts.append(section)

    if len(parts) == 1:
        parts.append("(no output)")
    return _truncate("\n".join(parts))


def run(command: str) -> str:
    if not isinstance(command, str):
        return "Error: command must be a string"
    if not command.strip():
        return "Error: command must not be empty"
    if "\x00" in command:
        return "Error: command must not contain NUL bytes"
    if len(command) > MAX_COMMAND_CHARS:
        return f"Error: command exceeds the {MAX_COMMAND_CHARS}-character limit"

    workspace = get_workspace()
    process_options: dict[str, object] = {}
    if os.name == "nt":
        process_options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        process_options["start_new_session"] = True

    try:
        process = subprocess.Popen(
            command,
            shell=True,
            cwd=str(workspace),
            env=_scrubbed_environment(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            **process_options,
        )
    except OSError as exc:
        return f"Error: command could not start: {safe_os_error(exc)}"

    assert process.stdout is not None
    assert process.stderr is not None
    stdout = _Capture()
    stderr = _Capture()
    readers = [
        threading.Thread(target=_drain, args=(process.stdout, stdout), daemon=True),
        threading.Thread(target=_drain, args=(process.stderr, stderr), daemon=True),
    ]
    for reader in readers:
        reader.start()

    try:
        returncode = process.wait(timeout=TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        _terminate(process)
        try:
            process.wait(timeout=5)
        except (subprocess.TimeoutExpired, OSError):
            pass
        for reader in readers:
            reader.join(timeout=2)
        return f"Error: command timed out after {TIMEOUT_SECONDS} seconds"
    except KeyboardInterrupt:
        _terminate(process)
        try:
            process.wait(timeout=5)
        except (subprocess.TimeoutExpired, OSError):
            pass
        for reader in readers:
            reader.join(timeout=2)
        raise

    for reader in readers:
        reader.join(timeout=2)
    return _format_result(returncode, stdout, stderr)
