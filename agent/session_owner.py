"""Process ownership for one durable workspace session.

The CLI is intentionally restartable, but two launchers must never control the
same session at the same time.  SQLite protects individual writes; this small
OS-backed lease protects the larger invariant that only one runtime owns the
session's provider, terminal loop, and loopback Web server.

The lock is held by an open file descriptor, so the operating system releases
it when a process exits unexpectedly.  A sidecar JSON file is only a discover-
able description of the owner (pid, heartbeat, Web port, and handshake token);
it is replaced atomically while the lock is held and is never used as the
authority for takeover.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class SessionOwnerInfo:
    """Public, non-secret owner metadata plus the local Web handshake token."""

    workspace: str
    session_id: str
    pid: int
    host: str
    started_at: float
    heartbeat_at: float
    web_port: int | None = None
    web_token: str | None = None
    owner_token: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "workspace": self.workspace,
            "session_id": self.session_id,
            "pid": self.pid,
            "host": self.host,
            "started_at": self.started_at,
            "heartbeat_at": self.heartbeat_at,
            "web_port": self.web_port,
            "web_token": self.web_token,
            "owner_token": self.owner_token,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SessionOwnerInfo | None":
        try:
            workspace = str(value.get("workspace") or "").strip()
            session_id = str(value.get("session_id") or "").strip()
            pid = int(value.get("pid"))
            started_at = float(value.get("started_at"))
            heartbeat_at = float(value.get("heartbeat_at"))
        except (TypeError, ValueError):
            return None
        if not workspace or not session_id or pid <= 0:
            return None
        port_raw = value.get("web_port")
        try:
            web_port = int(port_raw) if port_raw is not None else None
        except (TypeError, ValueError):
            web_port = None
        if web_port is not None and not 1 <= web_port <= 65535:
            web_port = None
        token_raw = value.get("web_token")
        owner_raw = value.get("owner_token")
        return cls(
            workspace=workspace,
            session_id=session_id,
            pid=pid,
            host=str(value.get("host") or ""),
            started_at=started_at,
            heartbeat_at=heartbeat_at,
            web_port=web_port,
            web_token=str(token_raw) if token_raw else None,
            owner_token=str(owner_raw or ""),
        )


class SessionOwnerLease:
    """Hold an OS-level lease and publish a heartbeat for one session.

    ``acquire`` returns ``None`` when another process owns the lock.  A caller
    must keep the returned object alive for the duration of the runtime and
    call :meth:`release` during normal shutdown.  Unexpected process exit is
    safe: the open descriptor is closed by the OS and the next launcher can
    acquire the lock without guessing from a stale timestamp.
    """

    heartbeat_interval = 2.0

    def __init__(
        self,
        *,
        workspace: Path,
        session_id: str,
        lock_path: Path,
        metadata_path: Path,
        handle: Any,
        info: SessionOwnerInfo,
    ) -> None:
        self.workspace = workspace
        self.session_id = session_id
        self.lock_path = lock_path
        self.metadata_path = metadata_path
        self._handle = handle
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._released = False
        self._info = info
        self._thread = threading.Thread(
            target=self._heartbeat_loop,
            name=f"ga3bad-owner-{os.getpid()}",
            daemon=True,
        )
        self._thread.start()

    @staticmethod
    def _paths(workspace: Path, session_id: str) -> tuple[Path, Path]:
        state_dir = workspace / ".coding-agent"
        state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            state_dir.resolve(strict=True).relative_to(workspace)
        except (OSError, ValueError) as exc:
            raise RuntimeError("session owner state directory escapes the workspace") from exc
        digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:20]
        stem = f"session-{digest}.owner"
        return state_dir / f"{stem}.lock", state_dir / f"{stem}.json"

    @staticmethod
    def _try_lock(handle: Any) -> bool:
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                # msvcrt.locking requires a byte to exist at the current
                # position.  The lock file's content is not authoritative.
                handle.seek(0, os.SEEK_END)
                if handle.tell() == 0:
                    handle.write(b"\0")
                    handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError):
            return False
        return True

    @staticmethod
    def _unlock(handle: Any) -> None:
        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except (OSError, ValueError):
            pass

    @staticmethod
    def read_info(metadata_path: str | os.PathLike[str]) -> SessionOwnerInfo | None:
        path = Path(metadata_path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError):
            return None
        return SessionOwnerInfo.from_mapping(payload) if isinstance(payload, Mapping) else None

    @classmethod
    def read_existing(
        cls,
        workspace: str | os.PathLike[str],
        session_id: str,
    ) -> SessionOwnerInfo | None:
        root = Path(workspace).resolve(strict=True)
        _lock_path, metadata_path = cls._paths(root, str(session_id))
        return cls.read_info(metadata_path)

    @classmethod
    def acquire(
        cls,
        workspace: str | os.PathLike[str],
        session_id: str,
    ) -> "SessionOwnerLease | None":
        root = Path(workspace).resolve(strict=True)
        session = str(session_id).strip()
        if not session:
            raise ValueError("session_id must not be empty")
        lock_path, metadata_path = cls._paths(root, session)
        try:
            handle = lock_path.open("a+b")
        except OSError:
            return None
        if not cls._try_lock(handle):
            handle.close()
            return None
        now = time.time()
        info = SessionOwnerInfo(
            workspace=str(root),
            session_id=session,
            pid=os.getpid(),
            host=socket.gethostname(),
            started_at=now,
            heartbeat_at=now,
            owner_token=f"{os.getpid()}-{time.monotonic_ns()}",
        )
        lease = cls(
            workspace=root,
            session_id=session,
            lock_path=lock_path,
            metadata_path=metadata_path,
            handle=handle,
            info=info,
        )
        lease._write_info(info)
        return lease

    @property
    def info(self) -> SessionOwnerInfo:
        with self._lock:
            return self._info

    @property
    def released(self) -> bool:
        return self._released

    def _write_info(self, info: SessionOwnerInfo) -> None:
        temporary: Path | None = None
        try:
            self.metadata_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            fd, name = tempfile.mkstemp(
                prefix=f".{self.metadata_path.name}.",
                suffix=".tmp",
                dir=str(self.metadata_path.parent),
            )
            temporary = Path(name)
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
                json.dump(info.to_dict(), stream, ensure_ascii=False, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.metadata_path)
            temporary = None
        except (OSError, UnicodeError, TypeError, ValueError):
            # The descriptor remains authoritative.  A heartbeat failure must
            # never make a live runtime silently relinquish its session.
            return
        finally:
            if temporary is not None:
                try:
                    temporary.unlink()
                except OSError:
                    pass

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(self.heartbeat_interval):
            with self._lock:
                if self._released:
                    return
                self._info = SessionOwnerInfo(
                    **{
                        **self._info.to_dict(),
                        "heartbeat_at": time.time(),
                    }
                )
                self._write_info(self._info)

    def set_web_endpoint(self, port: int, token: str) -> None:
        """Publish the authenticated loopback Web endpoint for attachers."""

        try:
            normalized_port = int(port)
        except (TypeError, ValueError) as exc:
            raise ValueError("web port must be an integer") from exc
        if not 1 <= normalized_port <= 65535:
            raise ValueError("web port must be between 1 and 65535")
        token_value = str(token).strip()
        if not token_value:
            raise ValueError("web token must not be empty")
        with self._lock:
            if self._released:
                raise RuntimeError("session owner lease is released")
            self._info = SessionOwnerInfo(
                **{
                    **self._info.to_dict(),
                    "web_port": normalized_port,
                    "web_token": token_value,
                    "heartbeat_at": time.time(),
                }
            )
            self._write_info(self._info)

    def release(self) -> None:
        with self._lock:
            if self._released:
                return
            self._released = True
            self._stop.set()
            self._write_info(
                SessionOwnerInfo(
                    **{
                        **self._info.to_dict(),
                        "heartbeat_at": time.time(),
                    }
                )
            )
        if self._thread.is_alive():
            self._thread.join(timeout=max(0.2, self.heartbeat_interval + 0.2))
        with self._lock:
            current = self.read_info(self.metadata_path)
            if current is not None and current.owner_token == self._info.owner_token:
                try:
                    self.metadata_path.unlink()
                except OSError:
                    pass
            self._unlock(self._handle)
            try:
                self._handle.close()
            except (OSError, ValueError):
                pass

    def __enter__(self) -> "SessionOwnerLease":
        return self

    def __exit__(self, *_: object) -> None:
        self.release()


__all__ = ["SessionOwnerInfo", "SessionOwnerLease"]
