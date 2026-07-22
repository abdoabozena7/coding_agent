"""Application-level project, thread, transcript, and job registry."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
from threading import RLock
from typing import Any, Iterator, Mapping
from uuid import uuid4

from .telemetry import DEFAULT_INFERENCE_PROFILE


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex}"


def default_registry_path() -> Path:
    override = os.getenv("GA3BAD_WEB_STATE_DIR", "").strip()
    if override:
        root = Path(override).expanduser()
    elif os.name == "nt" and os.getenv("LOCALAPPDATA"):
        root = Path(os.environ["LOCALAPPDATA"]) / "GA3BAD"
    else:
        root = Path(os.getenv("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "ga3bad"
    root.mkdir(parents=True, exist_ok=True)
    return root / "web-state.db"


class WebRegistry:
    """Small global registry; agent truth remains in each workspace database."""

    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        self.path = Path(path).resolve(strict=False) if path else default_registry_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._connection = sqlite3.connect(
            self.path,
            timeout=30,
            isolation_level=None,
            check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row
        with self._lock:
            self._connection.execute("PRAGMA foreign_keys=ON")
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=FULL")
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY,
                    path TEXT NOT NULL,
                    path_key TEXT NOT NULL UNIQUE,
                    name TEXT NOT NULL,
                    pinned INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_opened_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS threads (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    session_id TEXT NOT NULL,
                    goal_id TEXT,
                    title TEXT NOT NULL,
                    status TEXT NOT NULL,
                    pinned INTEGER NOT NULL DEFAULT 0,
                    archived INTEGER NOT NULL DEFAULT 0,
                    workflow_mode TEXT NOT NULL DEFAULT 'normal',
                    access_policy TEXT NOT NULL DEFAULT 'default',
                    model_overrides_json TEXT NOT NULL DEFAULT '{}',
                    pending_model_overrides_json TEXT NOT NULL DEFAULT '{}',
                    view_mode TEXT NOT NULL DEFAULT 'transcript',
                    plan_series_id TEXT NOT NULL DEFAULT '',
                    state_revision INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(project_id, session_id)
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    thread_id TEXT NOT NULL REFERENCES threads(id) ON DELETE CASCADE,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    technical INTEGER NOT NULL DEFAULT 0,
                    turn_id TEXT NOT NULL DEFAULT '',
                    direction TEXT NOT NULL DEFAULT 'auto',
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    thread_id TEXT NOT NULL REFERENCES threads(id) ON DELETE CASCADE,
                    kind TEXT NOT NULL,
                    input_text TEXT NOT NULL,
                    client_request_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    result_text TEXT NOT NULL DEFAULT '',
                    error TEXT NOT NULL DEFAULT '',
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    delivery TEXT NOT NULL DEFAULT 'queue',
                    blocked_reason TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    UNIQUE(thread_id, client_request_id)
                );
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS turn_activity (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    thread_id TEXT NOT NULL REFERENCES threads(id) ON DELETE CASCADE,
                    turn_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    details TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS thread_drafts (
                    thread_id TEXT PRIMARY KEY REFERENCES threads(id) ON DELETE CASCADE,
                    text TEXT NOT NULL DEFAULT '',
                    revision INTEGER NOT NULL DEFAULT 1,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS terminal_sessions (
                    id TEXT PRIMARY KEY,
                    thread_id TEXT NOT NULL REFERENCES threads(id) ON DELETE CASCADE,
                    mode TEXT NOT NULL,
                    status TEXT NOT NULL,
                    cwd TEXT NOT NULL,
                    history_json TEXT NOT NULL DEFAULT '[]',
                    scrollback TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_threads_project_updated
                    ON threads(project_id, archived, pinned, updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_messages_thread
                    ON messages(thread_id, id);
                CREATE INDEX IF NOT EXISTS idx_jobs_status_created
                    ON jobs(status, created_at);
                CREATE INDEX IF NOT EXISTS idx_turn_activity
                    ON turn_activity(thread_id, turn_id, id);
                """
            )
            self._ensure_column("threads", "workflow_mode", "TEXT NOT NULL DEFAULT 'normal'")
            self._ensure_column("threads", "access_policy", "TEXT NOT NULL DEFAULT 'default'")
            self._ensure_column("threads", "model_overrides_json", "TEXT NOT NULL DEFAULT '{}'")
            self._ensure_column("threads", "pending_model_overrides_json", "TEXT NOT NULL DEFAULT '{}'")
            self._ensure_column("threads", "view_mode", "TEXT NOT NULL DEFAULT 'transcript'")
            self._ensure_column("threads", "plan_series_id", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column("threads", "state_revision", "INTEGER NOT NULL DEFAULT 1")
            self._ensure_column("messages", "turn_id", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column("messages", "direction", "TEXT NOT NULL DEFAULT 'auto'")
            self._ensure_column("jobs", "delivery", "TEXT NOT NULL DEFAULT 'queue'")
            self._ensure_column("jobs", "blocked_reason", "TEXT NOT NULL DEFAULT ''")
        for key, value in {
            "theme": "dark",
            "locale": "auto",
            "experience": "simple",
            "mode": "normal",
            "access": "normal",
            "reduced_motion": False,
            "model_profile": {"main": "", "router": "", "verifier": "", "embedding": ""},
            "inference_profile": dict(DEFAULT_INFERENCE_PROFILE),
        }.items():
            self.set_setting(key, value, only_if_missing=True)

    def _ensure_column(self, table: str, name: str, definition: str) -> None:
        columns = {
            str(row["name"])
            for row in self._connection.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if name not in columns:
            self._connection.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                yield self._connection
            except Exception:
                self._connection.execute("ROLLBACK")
                raise
            else:
                self._connection.execute("COMMIT")

    @staticmethod
    def _row(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        value = dict(row)
        for key in ("pinned", "archived", "cancel_requested", "technical"):
            if key in value:
                value[key] = bool(value[key])
        if "model_overrides_json" in value:
            try:
                value["model_overrides"] = json.loads(value.pop("model_overrides_json") or "{}")
            except (TypeError, ValueError):
                value["model_overrides"] = {}
        if "pending_model_overrides_json" in value:
            try:
                value["pending_model_overrides"] = json.loads(value.pop("pending_model_overrides_json") or "{}")
            except (TypeError, ValueError):
                value["pending_model_overrides"] = {}
        return value

    def add_project(self, raw_path: str) -> tuple[dict[str, Any], bool]:
        if not str(raw_path).strip():
            raise ValueError("project path is required")
        try:
            path = Path(raw_path).expanduser().resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ValueError("project path does not exist or cannot be resolved") from exc
        if not path.is_dir():
            raise ValueError("project path must be a directory")
        canonical = str(path)
        path_key = canonical.casefold() if os.name == "nt" else canonical
        now = _now()
        created = False
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM projects WHERE path_key=?", (path_key,)
            ).fetchone()
            if row is None:
                project_id = _id("project")
                connection.execute(
                    "INSERT INTO projects(id,path,path_key,name,pinned,created_at,updated_at,last_opened_at) "
                    "VALUES(?,?,?,?,0,?,?,?)",
                    (project_id, canonical, path_key, path.name or canonical, now, now, now),
                )
                created = True
            else:
                project_id = str(row["id"])
                connection.execute(
                    "UPDATE projects SET last_opened_at=?,updated_at=? WHERE id=?",
                    (now, now, project_id),
                )
        self._import_legacy_thread(project_id, path)
        project = self.get_project(project_id)
        assert project is not None
        return project, created

    def _import_legacy_thread(self, project_id: str, workspace: Path) -> None:
        database = workspace / ".coding-agent" / "state.db"
        if not database.is_file():
            return
        try:
            source = sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True)
            source.row_factory = sqlite3.Row
            session = source.execute(
                "SELECT * FROM workflow_sessions WHERE id='workspace-session'"
            ).fetchone()
            if session is None:
                source.close()
                return
            goal = None
            if session["goal_id"]:
                goal = source.execute(
                    "SELECT id,objective,status,updated_at FROM goals WHERE id=?",
                    (session["goal_id"],),
                ).fetchone()
            source.close()
        except sqlite3.DatabaseError:
            return
        title = "Legacy workspace session"
        goal_id = None
        status = "idle"
        if goal is not None:
            goal_id = str(goal["id"])
            title = " ".join(str(goal["objective"]).split())[:72] or title
            status = str(goal["status"])
        now = _now()
        with self.transaction() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO threads(id,project_id,session_id,goal_id,title,status,pinned,archived,created_at,updated_at) "
                "VALUES(?,?,?,?,?,?,0,0,?,?)",
                (_id("thread"), project_id, "workspace-session", goal_id, title, status, now, now),
            )

    def list_projects(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM projects ORDER BY pinned DESC,last_opened_at DESC,name"
            ).fetchall()
        return [self._row(row) or {} for row in rows]

    def get_project(self, project_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM projects WHERE id=?", (str(project_id),)
            ).fetchone()
        return self._row(row)

    def create_thread(self, project_id: str, title: str = "New task") -> dict[str, Any]:
        if self.get_project(project_id) is None:
            raise KeyError("project not found")
        now = _now()
        thread_id = _id("thread")
        session_id = f"web-{thread_id}"
        clean_title = " ".join(str(title or "New task").split())[:120] or "New task"
        default_mode = str(self.settings().get("mode", "normal"))
        if default_mode not in {"plan", "normal", "ultra"}:
            default_mode = "normal"
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO threads(id,project_id,session_id,goal_id,title,status,pinned,archived,workflow_mode,created_at,updated_at) "
                "VALUES(?,?,?,?,?,'idle',0,0,?,?,?)",
                (thread_id, project_id, session_id, None, clean_title, default_mode, now, now),
            )
        result = self.get_thread(thread_id)
        assert result is not None
        return result

    def list_threads(self, project_id: str, *, include_archived: bool = True) -> list[dict[str, Any]]:
        clause = "" if include_archived else " AND archived=0"
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM threads WHERE project_id=?" + clause
                + " ORDER BY archived ASC,pinned DESC,updated_at DESC",
                (str(project_id),),
            ).fetchall()
        return [self._row(row) or {} for row in rows]

    def get_thread(self, thread_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM threads WHERE id=?", (str(thread_id),)
            ).fetchone()
        return self._row(row)

    def update_thread(self, thread_id: str, **changes: Any) -> dict[str, Any]:
        allowed = {
            "title", "status", "pinned", "archived", "goal_id",
            "workflow_mode", "access_policy", "model_overrides",
            "pending_model_overrides", "view_mode", "plan_series_id",
        }
        values = {key: value for key, value in changes.items() if key in allowed and value is not None}
        if "title" in values:
            values["title"] = " ".join(str(values["title"]).split())[:120] or "New task"
        for key in ("pinned", "archived"):
            if key in values:
                values[key] = int(bool(values[key]))
        if "workflow_mode" in values and values["workflow_mode"] not in {"plan", "normal", "ultra"}:
            raise ValueError("workflow mode must be plan, normal, or ultra")
        if "access_policy" in values and values["access_policy"] not in {"default", "bounded", "full"}:
            raise ValueError("persisted access policy must be default, bounded, or full")
        if "model_overrides" in values:
            overrides = dict(values.pop("model_overrides") or {})
            values["model_overrides_json"] = json.dumps(overrides, ensure_ascii=False, separators=(",", ":"))
        if "pending_model_overrides" in values:
            overrides = dict(values.pop("pending_model_overrides") or {})
            values["pending_model_overrides_json"] = json.dumps(overrides, ensure_ascii=False, separators=(",", ":"))
        if "view_mode" in values and values["view_mode"] not in {"transcript", "visualize"}:
            raise ValueError("view mode must be transcript or visualize")
        if not values:
            result = self.get_thread(thread_id)
            if result is None:
                raise KeyError("thread not found")
            return result
        values["updated_at"] = _now()
        stateful = {"workflow_mode", "access_policy", "model_overrides_json", "pending_model_overrides_json", "view_mode"}.intersection(values)
        revision_assignment = ",state_revision=state_revision+1" if stateful else ""
        assignments = ",".join(f"{key}=?" for key in values)
        with self.transaction() as connection:
            cursor = connection.execute(
                f"UPDATE threads SET {assignments}{revision_assignment} WHERE id=?",
                (*values.values(), str(thread_id)),
            )
            if cursor.rowcount != 1:
                raise KeyError("thread not found")
        result = self.get_thread(thread_id)
        assert result is not None
        return result

    def append_message(
        self,
        thread_id: str,
        role: str,
        content: str,
        *,
        technical: bool = False,
        turn_id: str = "",
        direction: str = "auto",
    ) -> int:
        clean = str(content).strip()
        if not clean:
            return 0
        with self.transaction() as connection:
            cursor = connection.execute(
                "INSERT INTO messages(thread_id,role,content,technical,turn_id,direction,created_at) VALUES(?,?,?,?,?,?,?)",
                (
                    str(thread_id), str(role), clean, int(bool(technical)),
                    str(turn_id), direction if direction in {"auto", "rtl"} else "auto", _now(),
                ),
            )
            connection.execute(
                "UPDATE threads SET updated_at=? WHERE id=?", (_now(), str(thread_id))
            )
        return int(cursor.lastrowid)

    def update_message_direction(self, thread_id: str, message_id: int, direction: str) -> dict[str, Any]:
        if direction not in {"auto", "rtl"}:
            raise ValueError("message direction must be auto or rtl")
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM messages WHERE id=? AND thread_id=?",
                (int(message_id), str(thread_id)),
            ).fetchone()
            if row is None:
                raise KeyError("message not found")
            if str(row["role"]) == "user":
                raise ValueError("direction can be changed only for assistant output")
            connection.execute(
                "UPDATE messages SET direction=? WHERE id=?", (direction, int(message_id))
            )
            updated = connection.execute("SELECT * FROM messages WHERE id=?", (int(message_id),)).fetchone()
        return self._row(updated) or {}

    def list_messages(self, thread_id: str, *, limit: int = 500) -> list[dict[str, Any]]:
        with self._lock:
            rows = list(reversed(self._connection.execute(
                "SELECT * FROM messages WHERE thread_id=? ORDER BY id DESC LIMIT ?",
                (str(thread_id), max(1, min(int(limit), 2_000))),
            ).fetchall()))
        return [self._row(row) or {} for row in rows]

    def append_activity(
        self,
        thread_id: str,
        turn_id: str,
        kind: str,
        summary: str,
        *,
        details: str = "",
    ) -> int:
        clean = " ".join(str(summary).split())[:1_000]
        if not clean:
            return 0
        with self.transaction() as connection:
            cursor = connection.execute(
                "INSERT INTO turn_activity(thread_id,turn_id,kind,summary,details,created_at) VALUES(?,?,?,?,?,?)",
                (str(thread_id), str(turn_id), str(kind)[:80], clean, str(details)[:20_000], _now()),
            )
        return int(cursor.lastrowid)

    def list_activity(self, thread_id: str, *, limit: int = 2_000) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM turn_activity WHERE thread_id=? ORDER BY id DESC LIMIT ?",
                (str(thread_id), max(1, min(int(limit), 5_000))),
            ).fetchall()
        return [self._row(row) or {} for row in reversed(rows)]

    def conversation_turns(self, thread_id: str) -> list[dict[str, Any]]:
        messages = self.list_messages(thread_id)
        activities = self.list_activity(thread_id)
        turns: dict[str, dict[str, Any]] = {}
        current_legacy = ""
        for message in messages:
            turn_id = str(message.get("turn_id") or "")
            if not turn_id and message.get("role") == "user":
                current_legacy = f"legacy-{message['id']}"
            turn_id = turn_id or current_legacy or f"legacy-{message['id']}"
            turn = turns.setdefault(turn_id, {"id": turn_id, "messages": [], "activity": []})
            turn["messages"].append(message)
        for item in activities:
            turn_id = str(item.get("turn_id") or "")
            if not turn_id:
                continue
            turn = turns.setdefault(turn_id, {"id": turn_id, "messages": [], "activity": []})
            turn["activity"].append(item)
        jobs = {str(job["id"]): job for job in self.list_jobs() if job["thread_id"] == thread_id}
        for turn_id, turn in turns.items():
            turn["job"] = jobs.get(turn_id)
        return list(turns.values())

    def create_job(
        self,
        project_id: str,
        thread_id: str,
        *,
        kind: str,
        text: str,
        client_request_id: str,
        delivery: str = "queue",
    ) -> tuple[dict[str, Any], bool]:
        with self._lock:
            existing = self._connection.execute(
                "SELECT * FROM jobs WHERE thread_id=? AND client_request_id=?",
                (str(thread_id), str(client_request_id)),
            ).fetchone()
        if existing is not None:
            return self._row(existing) or {}, False
        job_id = _id("job")
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO jobs(id,project_id,thread_id,kind,input_text,client_request_id,status,delivery,created_at) "
                "VALUES(?,?,?,?,?,?,'queued',?,?)",
                (job_id, project_id, thread_id, kind, str(text), str(client_request_id), str(delivery), _now()),
            )
            connection.execute(
                "UPDATE threads SET status='queued',updated_at=? WHERE id=?",
                (_now(), thread_id),
            )
        result = self.get_job(job_id)
        assert result is not None
        return result, True

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute("SELECT * FROM jobs WHERE id=?", (str(job_id),)).fetchone()
        return self._row(row)

    def get_job_by_request(self, thread_id: str, client_request_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM jobs WHERE thread_id=? AND client_request_id=?",
                (str(thread_id), str(client_request_id)),
            ).fetchone()
        return self._row(row)

    def list_jobs(self, *, statuses: tuple[str, ...] = ()) -> list[dict[str, Any]]:
        query = "SELECT * FROM jobs"
        params: tuple[Any, ...] = ()
        if statuses:
            query += " WHERE status IN (" + ",".join("?" for _ in statuses) + ")"
            params = tuple(statuses)
        query += " ORDER BY created_at"
        with self._lock:
            rows = self._connection.execute(query, params).fetchall()
        return [self._row(row) or {} for row in rows]

    def thread_queue(self, thread_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM jobs WHERE thread_id=? AND status='queued' ORDER BY created_at,id",
                (str(thread_id),),
            ).fetchall()
        return [self._row(row) or {} for row in rows]

    def cancel_queued_job(self, thread_id: str, job_id: str) -> dict[str, Any]:
        now = _now()
        with self.transaction() as connection:
            cursor = connection.execute(
                "UPDATE jobs SET status='cancelled',cancel_requested=1,completed_at=? "
                "WHERE id=? AND thread_id=? AND status='queued'",
                (now, str(job_id), str(thread_id)),
            )
            if cursor.rowcount != 1:
                raise KeyError("queued job not found")
        result = self.get_job(job_id)
        assert result is not None
        return result

    def update_job(self, job_id: str, status: str, **changes: Any) -> dict[str, Any]:
        values: dict[str, Any] = {"status": str(status)}
        for key in ("result_text", "error", "cancel_requested", "started_at", "completed_at", "blocked_reason"):
            if key in changes:
                values[key] = int(bool(changes[key])) if key == "cancel_requested" else changes[key]
        assignments = ",".join(f"{key}=?" for key in values)
        with self.transaction() as connection:
            cursor = connection.execute(
                f"UPDATE jobs SET {assignments} WHERE id=?",
                (*values.values(), str(job_id)),
            )
            if cursor.rowcount != 1:
                raise KeyError("job not found")
        result = self.get_job(job_id)
        assert result is not None
        return result

    def recover_jobs(self) -> None:
        now = _now()
        with self.transaction() as connection:
            connection.execute(
                "UPDATE jobs SET status='recovery_required',error=?,completed_at=? "
                "WHERE status IN ('running','pause_requested')",
                ("The web process stopped; inspect recovery state before resuming.", now),
            )
            connection.execute(
                "UPDATE jobs SET status='queued' WHERE status='queued'"
            )
            connection.execute(
                "UPDATE threads SET status='recovery_required',updated_at=? WHERE id IN "
                "(SELECT thread_id FROM jobs WHERE status='recovery_required')",
                (now,),
            )

    def get_draft(self, thread_id: str) -> dict[str, Any]:
        if self.get_thread(thread_id) is None:
            raise KeyError("thread not found")
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM thread_drafts WHERE thread_id=?", (str(thread_id),)
            ).fetchone()
        return self._row(row) or {"thread_id": str(thread_id), "text": "", "revision": 0, "updated_at": None}

    def save_draft(self, thread_id: str, text: str, expected_revision: int | None = None) -> dict[str, Any]:
        if self.get_thread(thread_id) is None:
            raise KeyError("thread not found")
        now = _now()
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT revision FROM thread_drafts WHERE thread_id=?", (str(thread_id),)
            ).fetchone()
            current = int(row["revision"]) if row else 0
            if expected_revision is not None and int(expected_revision) != current:
                raise ValueError("stale_draft")
            connection.execute(
                "INSERT INTO thread_drafts(thread_id,text,revision,updated_at) VALUES(?,?,?,?) "
                "ON CONFLICT(thread_id) DO UPDATE SET text=excluded.text,revision=excluded.revision,updated_at=excluded.updated_at",
                (str(thread_id), str(text)[:20_000], current + 1, now),
            )
        return self.get_draft(thread_id)

    def create_terminal_session(self, thread_id: str, mode: str, cwd: str) -> dict[str, Any]:
        if self.get_thread(thread_id) is None:
            raise KeyError("thread not found")
        session_id, now = _id("terminal"), _now()
        with self.transaction() as connection:
            connection.execute(
                "UPDATE terminal_sessions SET status='closed',updated_at=? WHERE thread_id=? AND status='open'",
                (now, str(thread_id)),
            )
            connection.execute(
                "INSERT INTO terminal_sessions(id,thread_id,mode,status,cwd,created_at,updated_at) VALUES(?,?,?,'open',?,?,?)",
                (session_id, str(thread_id), str(mode), str(cwd), now, now),
            )
        return self.get_terminal_session(session_id) or {}

    def get_terminal_session(self, session_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM terminal_sessions WHERE id=?", (str(session_id),)
            ).fetchone()
        value = self._row(row)
        if value is not None:
            value["history"] = json.loads(value.pop("history_json") or "[]")
        return value

    def update_terminal_session(self, session_id: str, *, status: str | None = None, cwd: str | None = None, history: list[str] | None = None, scrollback: str | None = None) -> dict[str, Any]:
        values: dict[str, Any] = {"updated_at": _now()}
        if status is not None:
            values["status"] = str(status)
        if cwd is not None:
            values["cwd"] = str(cwd)
        if history is not None:
            values["history_json"] = json.dumps([str(item)[:4_000] for item in history[-200:]], ensure_ascii=False)
        if scrollback is not None:
            values["scrollback"] = str(scrollback)[-200_000:]
        assignments = ",".join(f"{key}=?" for key in values)
        with self.transaction() as connection:
            cursor = connection.execute(
                f"UPDATE terminal_sessions SET {assignments} WHERE id=?",
                (*values.values(), str(session_id)),
            )
            if cursor.rowcount != 1:
                raise KeyError("terminal session not found")
        return self.get_terminal_session(session_id) or {}

    def set_setting(self, key: str, value: Any, *, only_if_missing: bool = False) -> None:
        encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        with self.transaction() as connection:
            if only_if_missing:
                connection.execute(
                    "INSERT OR IGNORE INTO settings(key,value_json,updated_at) VALUES(?,?,?)",
                    (str(key), encoded, _now()),
                )
            else:
                connection.execute(
                    "INSERT INTO settings(key,value_json,updated_at) VALUES(?,?,?) "
                    "ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json,updated_at=excluded.updated_at",
                    (str(key), encoded, _now()),
                )

    def settings(self) -> dict[str, Any]:
        with self._lock:
            rows = self._connection.execute("SELECT key,value_json FROM settings").fetchall()
        return {str(row["key"]): json.loads(row["value_json"]) for row in rows}

    def close(self) -> None:
        with self._lock:
            self._connection.close()


__all__ = ["WebRegistry", "default_registry_path"]
