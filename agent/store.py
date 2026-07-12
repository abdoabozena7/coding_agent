"""Transactional SQLite state store for persistent, crash-recoverable goals."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import zlib
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from threading import RLock
from typing import Any, Iterable, Iterator, Mapping

from .models import (
    Delegation,
    DelegationStatus,
    Evidence,
    Goal,
    GoalStatus,
    IN_FLIGHT_TASK_STATUSES,
    Plan,
    PlanApproval,
    PlanStatus,
    RecoveryReport,
    RoleProfile,
    RuntimeEvent,
    Task,
    TaskStatus,
    TERMINAL_GOAL_STATUSES,
    ensure_delegation_transition,
    ensure_goal_transition,
    ensure_plan_transition,
    ensure_task_transition,
    new_id,
    utc_now,
    validate_task_dag,
)
from .ultra_models import (
    AccessLevel,
    AgentRun,
    AgentRunStatus,
    ArchitectureSpecV1,
    Artifact,
    BrainEntry,
    BrainSection,
    ContractScopeError,
    ExecutionClass,
    GoalSpecV1,
    InsightV1,
    LeaseStatus,
    PromptTraceV1,
    ResourceLease,
    ResultPackageV1,
    TaskContractV1,
    UltraPhase,
    UltraRecoveryReport,
    UltraRun,
    UltraRunStatus,
    WorkNode,
    WorkNodeKind,
    WorkNodeStatus,
    assert_child_contract,
    normalize_contract_path,
)


SCHEMA_VERSION = 3

DEFAULT_TRACE_MAX_BYTES = 256_000
MAX_TRACE_MAX_BYTES = 2_000_000
MAX_PROMPT_TRACES_PER_RUN = 2_000


class StateStoreError(RuntimeError):
    pass


class StateCorruptionError(StateStoreError):
    pass


class ActiveGoalError(StateStoreError):
    pass


class NotFoundError(StateStoreError):
    pass


class StalePlanError(StateStoreError):
    pass


class CompletionGateError(StateStoreError):
    pass


class LeaseConflictError(StateStoreError):
    pass


class ConcurrentBrainUpdateError(StateStoreError):
    pass


class MasterApprovalError(StateStoreError):
    pass


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise StateCorruptionError("invalid JSON in agent state database") from exc


_TRACE_SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"\bAIza[A-Za-z0-9_-]{20,}\b"),
    re.compile(
        r"(?i)\b(authorization\s*[:=]\s*(?:bearer\s+)?)([^\s,;\"']+)"
    ),
    re.compile(
        r"(?i)\b((?:api[_-]?key|access[_-]?token|password|secret)\s*[:=]\s*[\"']?)([^\s,;\"']+)"
    ),
)


def _redact_trace_text(value: Any) -> tuple[str, bool]:
    text = str(value)
    changed = False
    for index, pattern in enumerate(_TRACE_SECRET_PATTERNS):
        if index < 2:
            text, count = pattern.subn("[REDACTED]", text)
        else:
            text, count = pattern.subn(lambda match: match.group(1) + "[REDACTED]", text)
        changed = changed or bool(count)
    return text, changed


def _redact_trace_value(value: Any) -> tuple[Any, bool]:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        changed = False
        for key, item in value.items():
            key_text = str(key)
            if re.search(r"(?i)(?:api[_-]?key|token|password|secret|authorization)", key_text):
                result[key_text] = "[REDACTED]"
                changed = True
            else:
                result[key_text], item_changed = _redact_trace_value(item)
                changed = changed or item_changed
        return result, changed
    if isinstance(value, (list, tuple)):
        result_list: list[Any] = []
        changed = False
        for item in value:
            safe, item_changed = _redact_trace_value(item)
            result_list.append(safe)
            changed = changed or item_changed
        return result_list, changed
    if isinstance(value, str):
        return _redact_trace_text(value)
    return value, False


def _truncate_utf8(value: str, maximum: int) -> tuple[str, bool]:
    encoded = value.encode("utf-8")
    if len(encoded) <= maximum:
        return value, False
    suffix = "\n...[trace truncated by harness]"
    budget = max(0, maximum - len(suffix.encode("utf-8")))
    shortened = encoded[:budget].decode("utf-8", errors="ignore") + suffix
    return shortened, True


def _paths_overlap(first: str, second: str) -> bool:
    left = normalize_contract_path(first).casefold()
    right = normalize_contract_path(second).casefold()
    return (
        left == "."
        or right == "."
        or left == right
        or left.startswith(right + "/")
        or right.startswith(left + "/")
    )


def _plan_fingerprint(
    summary: str,
    tasks: Iterable[Task],
    applicability_evidence: Iterable[Mapping[str, Any]],
    execution_strategy: str,
    expected_changes: Iterable[Mapping[str, Any]],
) -> str:
    payload = {
        "summary": summary,
        "applicability_evidence": list(applicability_evidence),
        "execution_strategy": execution_strategy,
        "expected_changes": list(expected_changes),
        "tasks": [
            {
                "id": task.id,
                "title": task.title,
                "description": task.description,
                "parent_id": task.parent_id,
                "depends_on": list(task.depends_on),
                "acceptance_criteria": list(task.acceptance_criteria),
                "verification": list(task.verification),
                "role": task.role.to_dict(),
                "mode": task.mode,
                "risk": task.risk,
                "priority": task.priority,
                "origin": task.origin,
            }
            for task in tasks
        ],
    }
    return hashlib.sha256(_json(payload).encode("utf-8")).hexdigest()


def _validate_plan_basis(
    tasks: Iterable[Task],
    applicability: Iterable[Mapping[str, Any]],
    strategy: str,
    changes: Iterable[Mapping[str, Any]],
) -> None:
    task_ids = {task.id for task in tasks}
    evidence_coverage: set[str] = set()
    for item in applicability:
        fact = str(item.get("fact", "")).strip()
        source = str(item.get("source", "")).strip()
        supports = {str(value).strip().upper() for value in item.get("supports_tasks", ())}
        if not fact or not source or not supports:
            raise ValueError("applicability evidence requires fact, source, and supported tasks")
        if supports - task_ids:
            raise ValueError("applicability evidence references a task outside this plan")
        evidence_coverage.update(supports)
    if evidence_coverage != task_ids:
        raise ValueError("applicability evidence must cover every plan task")
    if len(strategy) > 8_000:
        raise ValueError("execution strategy exceeds 8,000 characters")
    change_coverage: set[str] = set()
    for item in changes:
        path = str(item.get("path", "")).strip()
        intent = str(item.get("intent", "")).strip()
        supports = {str(value).strip().upper() for value in item.get("supports_tasks", ())}
        if not path or not intent or not supports:
            raise ValueError("expected changes require path, intent, and supported tasks")
        if supports - task_ids:
            raise ValueError("expected changes reference a task outside this plan")
        change_coverage.update(supports)
    if not change_coverage:
        raise ValueError("a coding plan must bind at least one workspace change to a task")


class StateStore:
    """One coordinator-owned SQLite journal, safe across restarts and threads."""

    def __init__(self, workspace: str | os.PathLike[str], path: str | os.PathLike[str] | None = None) -> None:
        self.workspace = Path(workspace).resolve(strict=True)
        if not self.workspace.is_dir():
            raise StateStoreError("workspace must be a directory")
        if path is None:
            state_dir = self.workspace / ".coding-agent"
            state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
            try:
                state_dir.resolve(strict=True).relative_to(self.workspace)
            except (ValueError, OSError) as exc:
                raise StateStoreError("agent state directory escapes the workspace") from exc
            self.path = state_dir / "state.db"
            self._install_git_exclude()
        else:
            self.path = Path(path).resolve(strict=False)
        self._lock = RLock()
        try:
            self._connection = sqlite3.connect(
                self.path,
                timeout=30,
                isolation_level=None,
                check_same_thread=False,
            )
        except sqlite3.DatabaseError as exc:
            raise StateCorruptionError(f"cannot open state database: {exc}") from exc
        self._connection.row_factory = sqlite3.Row
        try:
            self._initialize()
        except Exception:
            self._connection.close()
            raise

    def _install_git_exclude(self) -> None:
        """Keep local state out of git status without editing tracked files."""
        git_dir = self.workspace / ".git"
        try:
            if not git_dir.is_dir():
                return  # Includes linked worktrees where .git is a file.
            git_dir.resolve(strict=True).relative_to(self.workspace)
            info_dir = git_dir / "info"
            info_dir.mkdir(parents=True, exist_ok=True)
            info_dir.resolve(strict=True).relative_to(self.workspace)
            exclude = info_dir / "exclude"
            if exclude.exists():
                exclude.resolve(strict=True).relative_to(self.workspace)
                original = exclude.read_text(encoding="utf-8")
            else:
                original = ""
            pattern = "/.coding-agent/"
            if pattern in {line.strip() for line in original.splitlines()}:
                return
            separator = "" if not original or original.endswith(("\n", "\r")) else "\n"
            updated = original + separator + pattern + "\n"
            temporary = exclude.with_name(f".exclude.agent-{os.getpid()}-{new_id('tmp')}")
            try:
                with temporary.open("w", encoding="utf-8", newline="\n") as handle:
                    handle.write(updated)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, exclude)
            finally:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass
        except (OSError, UnicodeError, ValueError):
            # Git integration is a UX convenience, never a reason to lose state.
            return

    def _initialize(self) -> None:
        try:
            with self._lock:
                self._connection.execute("PRAGMA foreign_keys=ON")
                self._connection.execute("PRAGMA busy_timeout=30000")
                self._connection.execute("PRAGMA journal_mode=WAL")
                self._connection.execute("PRAGMA synchronous=FULL")
                existing = self._connection.execute("PRAGMA user_version").fetchone()[0]
                if existing > SCHEMA_VERSION:
                    raise StateStoreError(
                        f"state schema {existing} is newer than supported schema {SCHEMA_VERSION}"
                    )
                if existing < 1:
                    self._migrate_v1()
                    existing = 1
                if existing < 2:
                    self._migrate_v2()
                    existing = 2
                if existing < 3:
                    self._migrate_v3()
                self._fts5_available = self._ensure_brain_fts()
                check = self._connection.execute("PRAGMA quick_check").fetchone()[0]
                if check != "ok":
                    raise StateCorruptionError(f"state database integrity check failed: {check}")
        except sqlite3.DatabaseError as exc:
            raise StateCorruptionError(f"state database initialization failed: {exc}") from exc

    def _migrate_v1(self) -> None:
        schema = """
        CREATE TABLE IF NOT EXISTS goals (
            id TEXT PRIMARY KEY,
            objective TEXT NOT NULL,
            success_criteria_json TEXT NOT NULL,
            constraints_json TEXT NOT NULL,
            status TEXT NOT NULL,
            active_plan_revision INTEGER,
            metadata_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS plans (
            id TEXT PRIMARY KEY,
            goal_id TEXT NOT NULL REFERENCES goals(id) ON DELETE CASCADE,
            revision INTEGER NOT NULL,
            summary TEXT NOT NULL,
            status TEXT NOT NULL,
            proposed_by TEXT NOT NULL,
            fingerprint TEXT NOT NULL,
            accepted_by TEXT,
            accepted_at TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(goal_id, revision)
        );
        CREATE TABLE IF NOT EXISTS tasks (
            plan_id TEXT NOT NULL REFERENCES plans(id) ON DELETE CASCADE,
            task_id TEXT NOT NULL,
            goal_id TEXT NOT NULL REFERENCES goals(id) ON DELETE CASCADE,
            plan_revision INTEGER NOT NULL,
            parent_id TEXT,
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            status TEXT NOT NULL,
            depends_on_json TEXT NOT NULL,
            acceptance_json TEXT NOT NULL,
            verification_json TEXT NOT NULL,
            role_json TEXT NOT NULL,
            mode TEXT NOT NULL,
            risk TEXT NOT NULL,
            priority INTEGER NOT NULL,
            attempts INTEGER NOT NULL,
            origin TEXT NOT NULL,
            metadata_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY(plan_id, task_id)
        );
        CREATE TABLE IF NOT EXISTS evidence (
            id TEXT PRIMARY KEY,
            goal_id TEXT NOT NULL REFERENCES goals(id) ON DELETE CASCADE,
            plan_revision INTEGER,
            task_id TEXT,
            kind TEXT NOT NULL,
            summary TEXT NOT NULL,
            artifact_uri TEXT,
            data_json TEXT NOT NULL,
            created_by TEXT NOT NULL,
            verified INTEGER NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS delegations (
            id TEXT PRIMARY KEY,
            goal_id TEXT NOT NULL REFERENCES goals(id) ON DELETE CASCADE,
            task_id TEXT NOT NULL,
            plan_revision INTEGER NOT NULL,
            parent_id TEXT,
            worker_id TEXT,
            brief TEXT NOT NULL,
            role_json TEXT NOT NULL,
            status TEXT NOT NULL,
            attempt INTEGER NOT NULL,
            result_summary TEXT,
            error TEXT,
            metadata_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS approvals (
            id TEXT PRIMARY KEY,
            goal_id TEXT NOT NULL REFERENCES goals(id) ON DELETE CASCADE,
            plan_id TEXT NOT NULL REFERENCES plans(id) ON DELETE CASCADE,
            revision INTEGER NOT NULL,
            fingerprint TEXT NOT NULL,
            approved_by TEXT NOT NULL,
            approved_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS actions (
            id TEXT PRIMARY KEY,
            goal_id TEXT NOT NULL REFERENCES goals(id) ON DELETE CASCADE,
            task_id TEXT,
            tool_name TEXT NOT NULL,
            args_hash TEXT NOT NULL,
            args_json TEXT NOT NULL,
            risk TEXT NOT NULL,
            mutating INTEGER NOT NULL,
            status TEXT NOT NULL,
            result_summary TEXT,
            started_at TEXT NOT NULL,
            completed_at TEXT
        );
        CREATE TABLE IF NOT EXISTS events (
            sequence INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id TEXT NOT NULL UNIQUE,
            event_type TEXT NOT NULL,
            goal_id TEXT,
            entity_type TEXT,
            entity_id TEXT,
            payload_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_goals_status ON goals(status, updated_at);
        CREATE INDEX IF NOT EXISTS idx_plans_goal_revision ON plans(goal_id, revision);
        CREATE INDEX IF NOT EXISTS idx_tasks_goal_revision ON tasks(goal_id, plan_revision);
        CREATE INDEX IF NOT EXISTS idx_evidence_goal ON evidence(goal_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_delegations_goal ON delegations(goal_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_actions_status ON actions(status, started_at);
        CREATE INDEX IF NOT EXISTS idx_events_goal_sequence ON events(goal_id, sequence);
        PRAGMA user_version=1;
        """
        self._connection.executescript("BEGIN IMMEDIATE;" + schema + "COMMIT;")

    def _migrate_v2(self) -> None:
        schema = """
        BEGIN IMMEDIATE;
        ALTER TABLE plans ADD COLUMN applicability_json TEXT NOT NULL DEFAULT '[]';
        ALTER TABLE plans ADD COLUMN execution_strategy TEXT NOT NULL DEFAULT '';
        ALTER TABLE plans ADD COLUMN expected_changes_json TEXT NOT NULL DEFAULT '[]';
        PRAGMA user_version=2;
        COMMIT;
        """
        self._connection.executescript(schema)

    def _migrate_v3(self) -> None:
        """Install ULTRA state alongside the stable v1/v2 goal journal."""
        schema = """
        BEGIN IMMEDIATE;
        CREATE TABLE IF NOT EXISTS ultra_runs (
            id TEXT PRIMARY KEY,
            goal_id TEXT NOT NULL REFERENCES goals(id) ON DELETE CASCADE,
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            execution_class TEXT NOT NULL,
            access_level TEXT NOT NULL,
            concurrency INTEGER NOT NULL,
            phase TEXT NOT NULL,
            status TEXT NOT NULL,
            goal_spec_json TEXT,
            architecture_spec_json TEXT,
            plan_revision INTEGER,
            master_plan_fingerprint TEXT NOT NULL DEFAULT '',
            master_approved INTEGER NOT NULL DEFAULT 0,
            master_approved_by TEXT,
            master_approved_at TEXT,
            config_json TEXT NOT NULL,
            error TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS work_nodes (
            id TEXT PRIMARY KEY,
            ultra_run_id TEXT NOT NULL REFERENCES ultra_runs(id) ON DELETE CASCADE,
            parent_id TEXT REFERENCES work_nodes(id) ON DELETE CASCADE,
            master_task_id TEXT,
            kind TEXT NOT NULL,
            title TEXT NOT NULL,
            objective TEXT NOT NULL,
            status TEXT NOT NULL,
            depth INTEGER NOT NULL,
            position INTEGER NOT NULL,
            depends_on_json TEXT NOT NULL,
            contract_json TEXT NOT NULL,
            assigned_role TEXT NOT NULL,
            attempts INTEGER NOT NULL,
            max_attempts INTEGER NOT NULL,
            result_json TEXT,
            error TEXT,
            checkpoint TEXT NOT NULL DEFAULT '',
            metadata_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS agent_runs (
            id TEXT PRIMARY KEY,
            ultra_run_id TEXT NOT NULL REFERENCES ultra_runs(id) ON DELETE CASCADE,
            work_node_id TEXT REFERENCES work_nodes(id) ON DELETE SET NULL,
            role TEXT NOT NULL,
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            phase TEXT NOT NULL,
            status TEXT NOT NULL,
            attempt INTEGER NOT NULL,
            usage_json TEXT NOT NULL,
            result_json TEXT,
            error TEXT,
            prompt_trace_id TEXT,
            side_effects INTEGER NOT NULL DEFAULT 0,
            started_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            finished_at TEXT
        );
        CREATE TABLE IF NOT EXISTS brain_entries (
            id TEXT PRIMARY KEY,
            ultra_run_id TEXT NOT NULL REFERENCES ultra_runs(id) ON DELETE CASCADE,
            goal_id TEXT NOT NULL REFERENCES goals(id) ON DELETE CASCADE,
            work_node_id TEXT REFERENCES work_nodes(id) ON DELETE SET NULL,
            agent_run_id TEXT REFERENCES agent_runs(id) ON DELETE SET NULL,
            section TEXT NOT NULL,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            data_json TEXT NOT NULL,
            role TEXT,
            version INTEGER NOT NULL,
            supersedes_id TEXT REFERENCES brain_entries(id) ON DELETE SET NULL,
            expires_at TEXT,
            metadata_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(ultra_run_id, section, title, role, version)
        );
        CREATE TABLE IF NOT EXISTS artifacts (
            id TEXT PRIMARY KEY,
            ultra_run_id TEXT NOT NULL REFERENCES ultra_runs(id) ON DELETE CASCADE,
            work_node_id TEXT REFERENCES work_nodes(id) ON DELETE SET NULL,
            agent_run_id TEXT REFERENCES agent_runs(id) ON DELETE SET NULL,
            kind TEXT NOT NULL,
            uri TEXT NOT NULL,
            path TEXT,
            content_hash TEXT,
            pre_write_hash TEXT,
            evidence_json TEXT NOT NULL,
            metadata_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS prompt_traces (
            id TEXT PRIMARY KEY,
            ultra_run_id TEXT NOT NULL REFERENCES ultra_runs(id) ON DELETE CASCADE,
            work_node_id TEXT REFERENCES work_nodes(id) ON DELETE SET NULL,
            agent_run_id TEXT REFERENCES agent_runs(id) ON DELETE SET NULL,
            role TEXT NOT NULL,
            system_prompt_blob BLOB NOT NULL,
            context_blob BLOB NOT NULL,
            self_prompt_blob BLOB NOT NULL,
            compression TEXT NOT NULL,
            raw_size INTEGER NOT NULL,
            stored_size INTEGER NOT NULL,
            redacted INTEGER NOT NULL,
            truncated INTEGER NOT NULL,
            reasoning_summary TEXT NOT NULL,
            insights_json TEXT NOT NULL,
            omitted_json TEXT NOT NULL,
            metadata_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS memory_access (
            id TEXT PRIMARY KEY,
            ultra_run_id TEXT NOT NULL REFERENCES ultra_runs(id) ON DELETE CASCADE,
            work_node_id TEXT REFERENCES work_nodes(id) ON DELETE SET NULL,
            agent_run_id TEXT REFERENCES agent_runs(id) ON DELETE SET NULL,
            brain_entry_id TEXT REFERENCES brain_entries(id) ON DELETE SET NULL,
            direction TEXT NOT NULL,
            query TEXT NOT NULL,
            score REAL,
            metadata_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS resource_leases (
            id TEXT PRIMARY KEY,
            ultra_run_id TEXT NOT NULL REFERENCES ultra_runs(id) ON DELETE CASCADE,
            work_node_id TEXT NOT NULL REFERENCES work_nodes(id) ON DELETE CASCADE,
            agent_run_id TEXT REFERENCES agent_runs(id) ON DELETE SET NULL,
            normalized_path TEXT NOT NULL,
            pre_write_hash TEXT,
            status TEXT NOT NULL,
            acquired_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            released_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_ultra_runs_goal ON ultra_runs(goal_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_ultra_runs_status ON ultra_runs(status, updated_at);
        CREATE INDEX IF NOT EXISTS idx_work_nodes_run_parent ON work_nodes(ultra_run_id, parent_id, position);
        CREATE INDEX IF NOT EXISTS idx_work_nodes_run_status ON work_nodes(ultra_run_id, status, position);
        CREATE INDEX IF NOT EXISTS idx_agent_runs_run_status ON agent_runs(ultra_run_id, status, updated_at);
        CREATE INDEX IF NOT EXISTS idx_brain_entries_lookup ON brain_entries(ultra_run_id, section, role, updated_at);
        CREATE INDEX IF NOT EXISTS idx_artifacts_node ON artifacts(ultra_run_id, work_node_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_traces_run ON prompt_traces(ultra_run_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_memory_access_run ON memory_access(ultra_run_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_leases_active ON resource_leases(ultra_run_id, status, normalized_path, expires_at);
        PRAGMA user_version=3;
        COMMIT;
        """
        self._connection.executescript(schema)

    def _ensure_brain_fts(self) -> bool:
        """Enable FTS5 when SQLite provides it; LIKE search remains portable."""
        try:
            self._connection.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS brain_entries_fts USING fts5("
                "entry_id UNINDEXED,title,content,section,role)"
            )
            count = self._connection.execute("SELECT COUNT(*) FROM brain_entries_fts").fetchone()[0]
            if not count:
                self._connection.execute(
                    "INSERT INTO brain_entries_fts(entry_id,title,content,section,role) "
                    "SELECT id,title,content,section,COALESCE(role,'') FROM brain_entries"
                )
            return True
        except sqlite3.OperationalError:
            return False

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                yield self._connection
                self._connection.execute("COMMIT")
            except Exception:
                try:
                    self._connection.execute("ROLLBACK")
                except sqlite3.DatabaseError:
                    pass
                raise

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> "StateStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _event(
        self,
        connection: sqlite3.Connection,
        event_type: str,
        *,
        goal_id: str | None = None,
        entity_type: str | None = None,
        entity_id: str | None = None,
        payload: Mapping[str, Any] | None = None,
    ) -> RuntimeEvent:
        event = RuntimeEvent(
            event_type=event_type,
            goal_id=goal_id,
            entity_type=entity_type,
            entity_id=entity_id,
            payload=dict(payload or {}),
        )
        cursor = connection.execute(
            "INSERT INTO events(event_id,event_type,goal_id,entity_type,entity_id,payload_json,created_at) VALUES(?,?,?,?,?,?,?)",
            (event.id, event.event_type, goal_id, entity_type, entity_id, _json(event.payload), _iso(event.created_at)),
        )
        return replace(event, sequence=cursor.lastrowid)

    def append_event(
        self,
        event_type: str,
        *,
        goal_id: str | None = None,
        entity_type: str | None = None,
        entity_id: str | None = None,
        payload: Mapping[str, Any] | None = None,
    ) -> RuntimeEvent:
        with self.transaction() as connection:
            return self._event(
                connection,
                event_type,
                goal_id=goal_id,
                entity_type=entity_type,
                entity_id=entity_id,
                payload=payload,
            )

    def create_goal(
        self,
        objective: str,
        *,
        success_criteria: Iterable[str] = (),
        constraints: Iterable[str] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> Goal:
        objective = str(objective).strip()
        if not objective:
            raise ValueError("goal objective must not be empty")
        if "\x00" in objective:
            raise ValueError("goal objective must not contain NUL bytes")
        if len(objective) > 20_000:
            raise ValueError("goal objective exceeds 20,000 characters")
        with self.transaction() as connection:
            terminal = tuple(status.value for status in TERMINAL_GOAL_STATUSES)
            placeholders = ",".join("?" for _ in terminal)
            active = connection.execute(
                f"SELECT id FROM goals WHERE status NOT IN ({placeholders}) LIMIT 1", terminal
            ).fetchone()
            if active:
                raise ActiveGoalError(f"unfinished goal already exists: {active['id']}")
            goal = Goal(
                id=new_id("goal"),
                objective=objective,
                success_criteria=tuple(str(item) for item in success_criteria),
                constraints=tuple(str(item) for item in constraints),
                metadata=dict(metadata or {}),
            )
            connection.execute(
                "INSERT INTO goals VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    goal.id,
                    goal.objective,
                    _json(goal.success_criteria),
                    _json(goal.constraints),
                    goal.status.value,
                    goal.active_plan_revision,
                    _json(goal.metadata),
                    _iso(goal.created_at),
                    _iso(goal.updated_at),
                ),
            )
            self._event(connection, "goal.created", goal_id=goal.id, entity_type="goal", entity_id=goal.id, payload={"objective": goal.objective})
            return goal

    @staticmethod
    def _goal_from_row(row: sqlite3.Row) -> Goal:
        return Goal(
            id=row["id"],
            objective=row["objective"],
            success_criteria=tuple(_loads(row["success_criteria_json"], [])),
            constraints=tuple(_loads(row["constraints_json"], [])),
            status=GoalStatus(row["status"]),
            active_plan_revision=row["active_plan_revision"],
            metadata=_loads(row["metadata_json"], {}),
            created_at=_dt(row["created_at"]),
            updated_at=_dt(row["updated_at"]),
        )

    def get_goal(self, goal_id: str) -> Goal:
        with self._lock:
            row = self._connection.execute("SELECT * FROM goals WHERE id=?", (goal_id,)).fetchone()
        if row is None:
            raise NotFoundError(f"goal not found: {goal_id}")
        return self._goal_from_row(row)

    def load_active_goal(self) -> Goal | None:
        terminal = tuple(status.value for status in TERMINAL_GOAL_STATUSES)
        placeholders = ",".join("?" for _ in terminal)
        with self._lock:
            rows = self._connection.execute(
                f"SELECT * FROM goals WHERE status NOT IN ({placeholders}) ORDER BY created_at DESC", terminal
            ).fetchall()
        if len(rows) > 1:
            raise StateCorruptionError("multiple unfinished goals exist")
        return self._goal_from_row(rows[0]) if rows else None

    def get_latest_goal(self) -> Goal | None:
        """Return the most recently created goal, including terminal goals."""
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM goals ORDER BY created_at DESC,id DESC LIMIT 1"
            ).fetchone()
        return self._goal_from_row(row) if row else None

    def transition_goal(
        self,
        goal_id: str,
        status: GoalStatus | str,
        *,
        reason: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> Goal:
        target = GoalStatus(status)
        with self.transaction() as connection:
            row = connection.execute("SELECT * FROM goals WHERE id=?", (goal_id,)).fetchone()
            if row is None:
                raise NotFoundError(f"goal not found: {goal_id}")
            current = self._goal_from_row(row)
            ensure_goal_transition(current.status, target)
            merged = dict(current.metadata)
            merged.update(metadata or {})
            now = utc_now()
            connection.execute(
                "UPDATE goals SET status=?,metadata_json=?,updated_at=? WHERE id=?",
                (target.value, _json(merged), _iso(now), goal_id),
            )
            self._event(
                connection,
                "goal.status_changed",
                goal_id=goal_id,
                entity_type="goal",
                entity_id=goal_id,
                payload={"from": current.status.value, "to": target.value, "reason": reason},
            )
            return replace(current, status=target, metadata=merged, updated_at=now)

    def update_goal_metadata(self, goal_id: str, **metadata: Any) -> Goal:
        current = self.get_goal(goal_id)
        merged = dict(current.metadata)
        merged.update(metadata)
        with self.transaction() as connection:
            now = utc_now()
            connection.execute(
                "UPDATE goals SET metadata_json=?,updated_at=? WHERE id=?",
                (_json(merged), _iso(now), goal_id),
            )
            self._event(connection, "goal.metadata_updated", goal_id=goal_id, entity_type="goal", entity_id=goal_id, payload={"keys": sorted(metadata)})
        return replace(current, metadata=merged, updated_at=now)

    @staticmethod
    def coerce_task(value: Task | Mapping[str, Any], goal_id: str, revision: int, origin: str) -> Task:
        if isinstance(value, Task):
            task_id = value.id.strip().upper()
            if not re.fullmatch(r"[A-Z0-9][A-Z0-9_.-]{0,23}", task_id):
                raise ValueError(f"invalid task id: {value.id!r}")
            return replace(
                value,
                id=task_id,
                goal_id=goal_id,
                plan_revision=revision,
                parent_id=value.parent_id.strip().upper() if value.parent_id else None,
                depends_on=tuple(item.strip().upper() for item in value.depends_on),
            )
        role_value = value.get("role")
        role = RoleProfile.from_dict(role_value if isinstance(role_value, Mapping) else None)
        task_id = str(value.get("id") or new_id("task")[:24]).strip().upper()
        if not re.fullmatch(r"[A-Z0-9][A-Z0-9_.-]{0,23}", task_id):
            raise ValueError(f"invalid task id: {task_id!r}")
        return Task(
            id=task_id,
            title=str(value.get("title") or "Untitled task"),
            description=str(value.get("description") or value.get("title") or ""),
            goal_id=goal_id,
            plan_revision=revision,
            parent_id=str(value["parent_id"]).strip().upper() if value.get("parent_id") else None,
            status=TaskStatus(value.get("status", TaskStatus.PENDING.value)),
            depends_on=tuple(str(item).strip().upper() for item in value.get("depends_on", ())),
            acceptance_criteria=tuple(str(item) for item in value.get("acceptance_criteria", ())),
            verification=tuple(str(item) for item in value.get("verification", ())),
            role=role,
            mode=str(value.get("mode", "auto")),
            risk=str(value.get("risk", "medium")).lower(),
            priority=int(value.get("priority", 0)),
            attempts=int(value.get("attempts", 0)),
            origin=str(value.get("origin", origin)),
            metadata=dict(value.get("metadata") or {}),
        )

    def create_plan(
        self,
        goal_id: str,
        summary: str,
        tasks: Iterable[Task | Mapping[str, Any]],
        *,
        applicability_evidence: Iterable[Mapping[str, Any]],
        execution_strategy: str,
        expected_changes: Iterable[Mapping[str, Any]],
        proposed_by: str = "agent",
        submit: bool = True,
    ) -> Plan:
        summary = str(summary).strip()
        if not summary:
            raise ValueError("a plan requires a non-empty summary")
        if len(summary) > 20_000:
            raise ValueError("plan summary exceeds 20,000 characters")
        applicability = tuple(dict(item) for item in applicability_evidence)
        strategy = str(execution_strategy).strip()
        changes = tuple(dict(item) for item in expected_changes)
        if not applicability:
            raise ValueError("a plan requires inspected applicability evidence")
        if not strategy:
            raise ValueError("a plan requires an executable strategy")
        if not changes:
            raise ValueError("a coding plan requires at least one expected workspace change")
        with self.transaction() as connection:
            if connection.execute("SELECT 1 FROM goals WHERE id=?", (goal_id,)).fetchone() is None:
                raise NotFoundError(f"goal not found: {goal_id}")
            revision = connection.execute(
                "SELECT COALESCE(MAX(revision),0)+1 FROM plans WHERE goal_id=?", (goal_id,)
            ).fetchone()[0]
            normalized = tuple(self.coerce_task(item, goal_id, revision, proposed_by) for item in tasks)
            if not normalized:
                raise ValueError("a plan requires at least one task")
            validate_task_dag(normalized)
            _validate_plan_basis(normalized, applicability, strategy, changes)
            now = utc_now()
            plan = Plan(
                id=new_id("plan"),
                goal_id=goal_id,
                revision=revision,
                summary=summary,
                status=PlanStatus.PENDING_APPROVAL if submit else PlanStatus.DRAFT,
                tasks=normalized,
                applicability_evidence=applicability,
                execution_strategy=strategy,
                expected_changes=changes,
                proposed_by=proposed_by,
                fingerprint=_plan_fingerprint(summary, normalized, applicability, strategy, changes),
                created_at=now,
                updated_at=now,
            )
            connection.execute(
                "UPDATE plans SET status=?,updated_at=? WHERE goal_id=? AND status IN (?,?)",
                (PlanStatus.SUPERSEDED.value, _iso(now), goal_id, PlanStatus.DRAFT.value, PlanStatus.PENDING_APPROVAL.value),
            )
            connection.execute(
                "INSERT INTO plans(id,goal_id,revision,summary,status,proposed_by,fingerprint,accepted_by,accepted_at,created_at,updated_at,applicability_json,execution_strategy,expected_changes_json) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    plan.id, goal_id, revision, plan.summary, plan.status.value,
                    proposed_by, plan.fingerprint, None, None, _iso(now), _iso(now),
                    _json(applicability), strategy, _json(changes),
                ),
            )
            for task in normalized:
                connection.execute(
                    "INSERT INTO tasks VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        plan.id, task.id, goal_id, revision, task.parent_id, task.title,
                        task.description, task.status.value, _json(task.depends_on),
                        _json(task.acceptance_criteria), _json(task.verification),
                        _json(task.role.to_dict()), task.mode, task.risk, task.priority,
                        task.attempts, task.origin, _json(task.metadata), _iso(task.created_at),
                        _iso(task.updated_at),
                    ),
                )
            self._event(
                connection,
                "plan.submitted" if submit else "plan.created",
                goal_id=goal_id,
                entity_type="plan",
                entity_id=plan.id,
                payload={"revision": revision, "fingerprint": plan.fingerprint, "proposed_by": proposed_by},
            )
            return plan

    revise_plan = create_plan

    @staticmethod
    def _task_from_row(row: sqlite3.Row) -> Task:
        return Task(
            id=row["task_id"], title=row["title"], description=row["description"],
            goal_id=row["goal_id"], plan_revision=row["plan_revision"], parent_id=row["parent_id"],
            status=TaskStatus(row["status"]), depends_on=tuple(_loads(row["depends_on_json"], [])),
            acceptance_criteria=tuple(_loads(row["acceptance_json"], [])),
            verification=tuple(_loads(row["verification_json"], [])),
            role=RoleProfile.from_dict(_loads(row["role_json"], {})), mode=row["mode"], risk=row["risk"],
            priority=row["priority"], attempts=row["attempts"], origin=row["origin"],
            metadata=_loads(row["metadata_json"], {}), created_at=_dt(row["created_at"]), updated_at=_dt(row["updated_at"]),
        )

    def _plan_from_row(self, row: sqlite3.Row) -> Plan:
        with self._lock:
            task_rows = self._connection.execute(
                "SELECT * FROM tasks WHERE plan_id=? ORDER BY priority DESC,rowid", (row["id"],)
            ).fetchall()
        return Plan(
            id=row["id"], goal_id=row["goal_id"], revision=row["revision"], summary=row["summary"],
            status=PlanStatus(row["status"]), tasks=tuple(self._task_from_row(item) for item in task_rows),
            applicability_evidence=tuple(_loads(row["applicability_json"], [])),
            execution_strategy=row["execution_strategy"],
            expected_changes=tuple(_loads(row["expected_changes_json"], [])),
            proposed_by=row["proposed_by"], fingerprint=row["fingerprint"], accepted_by=row["accepted_by"],
            accepted_at=_dt(row["accepted_at"]), created_at=_dt(row["created_at"]), updated_at=_dt(row["updated_at"]),
        )

    def get_plan(self, goal_id: str, revision: int) -> Plan:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM plans WHERE goal_id=? AND revision=?", (goal_id, revision)
            ).fetchone()
        if row is None:
            raise NotFoundError(f"plan r{revision} not found")
        return self._plan_from_row(row)

    def get_latest_plan(self, goal_id: str) -> Plan | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM plans WHERE goal_id=? ORDER BY revision DESC LIMIT 1", (goal_id,)
            ).fetchone()
        return self._plan_from_row(row) if row else None

    def get_accepted_plan(self, goal_id: str) -> Plan | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM plans WHERE goal_id=? AND status=? ORDER BY revision DESC LIMIT 1",
                (goal_id, PlanStatus.ACCEPTED.value),
            ).fetchone()
        return self._plan_from_row(row) if row else None

    def list_tasks(self, goal_id: str, revision: int | None = None) -> tuple[Task, ...]:
        plan = self.get_plan(goal_id, revision) if revision is not None else self.get_latest_plan(goal_id)
        return plan.tasks if plan else ()

    def approve_plan(
        self,
        goal_id: str,
        revision: int,
        *,
        approved_by: str = "user",
        expected_fingerprint: str | None = None,
    ) -> tuple[Plan, PlanApproval]:
        with self.transaction() as connection:
            latest = connection.execute("SELECT MAX(revision) FROM plans WHERE goal_id=?", (goal_id,)).fetchone()[0]
            if latest != revision:
                raise StalePlanError(f"cannot approve stale plan r{revision}; latest is r{latest}")
            row = connection.execute(
                "SELECT * FROM plans WHERE goal_id=? AND revision=?", (goal_id, revision)
            ).fetchone()
            if row is None:
                raise NotFoundError(f"plan r{revision} not found")
            if (
                not _loads(row["applicability_json"], [])
                or not str(row["execution_strategy"]).strip()
                or not _loads(row["expected_changes_json"], [])
            ):
                raise StalePlanError(
                    "plan lacks fingerprinted applicability evidence; use /replan before approval"
                )
            current_status = PlanStatus(row["status"])
            ensure_plan_transition(current_status, PlanStatus.ACCEPTED)
            if expected_fingerprint and expected_fingerprint != row["fingerprint"]:
                raise StalePlanError("plan content changed after it was displayed")
            goal_row = connection.execute("SELECT * FROM goals WHERE id=?", (goal_id,)).fetchone()
            goal = self._goal_from_row(goal_row)
            ensure_goal_transition(goal.status, GoalStatus.RUNNING)
            now = utc_now()
            connection.execute(
                "UPDATE plans SET status=?,updated_at=? WHERE goal_id=? AND status=?",
                (PlanStatus.SUPERSEDED.value, _iso(now), goal_id, PlanStatus.ACCEPTED.value),
            )
            connection.execute(
                "UPDATE plans SET status=?,accepted_by=?,accepted_at=?,updated_at=? WHERE id=?",
                (PlanStatus.ACCEPTED.value, approved_by, _iso(now), _iso(now), row["id"]),
            )
            connection.execute(
                "UPDATE goals SET status=?,active_plan_revision=?,updated_at=? WHERE id=?",
                (GoalStatus.RUNNING.value, revision, _iso(now), goal_id),
            )
            approval = PlanApproval(new_id("approval"), goal_id, row["id"], revision, row["fingerprint"], approved_by, now)
            connection.execute(
                "INSERT INTO approvals VALUES(?,?,?,?,?,?,?)",
                (approval.id, approval.goal_id, approval.plan_id, approval.revision, approval.fingerprint, approval.approved_by, _iso(approval.approved_at)),
            )
            self._event(connection, "plan.approved", goal_id=goal_id, entity_type="plan", entity_id=row["id"], payload={"revision": revision, "fingerprint": row["fingerprint"], "approved_by": approved_by})
        return self.get_plan(goal_id, revision), approval

    def reject_plan(self, goal_id: str, revision: int, feedback: str, *, rejected_by: str = "user") -> Plan:
        with self.transaction() as connection:
            row = connection.execute("SELECT * FROM plans WHERE goal_id=? AND revision=?", (goal_id, revision)).fetchone()
            if row is None:
                raise NotFoundError(f"plan r{revision} not found")
            ensure_plan_transition(PlanStatus(row["status"]), PlanStatus.REJECTED)
            goal = self._goal_from_row(connection.execute("SELECT * FROM goals WHERE id=?", (goal_id,)).fetchone())
            ensure_goal_transition(goal.status, GoalStatus.REVISING)
            now = utc_now()
            connection.execute("UPDATE plans SET status=?,updated_at=? WHERE id=?", (PlanStatus.REJECTED.value, _iso(now), row["id"]))
            connection.execute("UPDATE goals SET status=?,updated_at=? WHERE id=?", (GoalStatus.REVISING.value, _iso(now), goal_id))
            self._event(connection, "plan.rejected", goal_id=goal_id, entity_type="plan", entity_id=row["id"], payload={"revision": revision, "feedback": feedback, "rejected_by": rejected_by})
        return self.get_plan(goal_id, revision)

    def transition_task(
        self,
        goal_id: str,
        revision: int,
        task_id: str,
        status: TaskStatus | str,
        *,
        note: str = "",
        evidence: Iterable[str] = (),
        actor: str = "agent",
    ) -> Task:
        target = TaskStatus(status)
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT t.* FROM tasks t JOIN plans p ON p.id=t.plan_id WHERE t.goal_id=? AND t.plan_revision=? AND t.task_id=?",
                (goal_id, revision, task_id),
            ).fetchone()
            if row is None:
                raise NotFoundError(f"task not found: {task_id}")
            task = self._task_from_row(row)
            ensure_task_transition(task.status, target)
            if target in {TaskStatus.READY, TaskStatus.IN_PROGRESS, TaskStatus.VERIFYING, TaskStatus.COMPLETED}:
                for dependency in task.depends_on:
                    dep = connection.execute(
                        "SELECT status FROM tasks WHERE plan_id=? AND task_id=?", (row["plan_id"], dependency)
                    ).fetchone()
                    if dep is None or TaskStatus(dep["status"]) not in {TaskStatus.COMPLETED, TaskStatus.OBSOLETE}:
                        raise CompletionGateError(f"task {task_id} has unfinished dependency {dependency}")
            now = utc_now()
            for summary in evidence:
                item = Evidence(
                    goal_id=goal_id, plan_revision=revision, task_id=task_id,
                    kind="task", summary=str(summary), created_by=actor,
                )
                connection.execute(
                    "INSERT INTO evidence VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    (item.id, item.goal_id, item.plan_revision, item.task_id, item.kind, item.summary, item.artifact_uri, _json(item.data), item.created_by, int(item.verified), _iso(item.created_at)),
                )
            if target == TaskStatus.COMPLETED:
                count = connection.execute(
                    "SELECT COUNT(*) FROM evidence WHERE goal_id=? AND plan_revision=? AND task_id=?",
                    (goal_id, revision, task_id),
                ).fetchone()[0]
                if not count:
                    raise CompletionGateError("completed tasks require recorded evidence")
            if target == TaskStatus.BLOCKED and not note.strip():
                raise CompletionGateError("blocked tasks require a concrete blocker note")
            attempts = task.attempts + (1 if target == TaskStatus.IN_PROGRESS and task.status != target else 0)
            metadata = dict(task.metadata)
            if note:
                metadata["last_note"] = note
            connection.execute(
                "UPDATE tasks SET status=?,attempts=?,metadata_json=?,updated_at=? WHERE plan_id=? AND task_id=?",
                (target.value, attempts, _json(metadata), _iso(now), row["plan_id"], task_id),
            )
            self._event(connection, "task.status_changed", goal_id=goal_id, entity_type="task", entity_id=task_id, payload={"revision": revision, "from": task.status.value, "to": target.value, "note": note, "actor": actor})
            return replace(task, status=target, attempts=attempts, metadata=metadata, updated_at=now)

    def add_evidence(self, evidence: Evidence | None = None, **kwargs: Any) -> Evidence:
        item = evidence or Evidence(**kwargs)
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO evidence VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (item.id, item.goal_id, item.plan_revision, item.task_id, item.kind, item.summary, item.artifact_uri, _json(item.data), item.created_by, int(item.verified), _iso(item.created_at)),
            )
            self._event(connection, "evidence.added", goal_id=item.goal_id, entity_type="evidence", entity_id=item.id, payload={"task_id": item.task_id, "kind": item.kind, "summary": item.summary})
        return item

    @staticmethod
    def _evidence_from_row(row: sqlite3.Row) -> Evidence:
        return Evidence(
            id=row["id"], goal_id=row["goal_id"], plan_revision=row["plan_revision"], task_id=row["task_id"],
            kind=row["kind"], summary=row["summary"], artifact_uri=row["artifact_uri"], data=_loads(row["data_json"], {}),
            created_by=row["created_by"], verified=bool(row["verified"]), created_at=_dt(row["created_at"]),
        )

    def list_evidence(self, goal_id: str, *, task_id: str | None = None, kind: str | None = None) -> tuple[Evidence, ...]:
        sql, params = "SELECT * FROM evidence WHERE goal_id=?", [goal_id]
        if task_id is not None:
            sql += " AND task_id=?"; params.append(task_id)
        if kind is not None:
            sql += " AND kind=?"; params.append(kind)
        sql += " ORDER BY created_at,id"
        with self._lock:
            rows = self._connection.execute(sql, tuple(params)).fetchall()
        return tuple(self._evidence_from_row(row) for row in rows)

    def create_delegation(self, delegation: Delegation | None = None, **kwargs: Any) -> Delegation:
        item = delegation or Delegation(**kwargs)
        with self.transaction() as connection:
            task_exists = connection.execute(
                "SELECT 1 FROM tasks t JOIN plans p ON p.id=t.plan_id "
                "WHERE t.goal_id=? AND t.plan_revision=? AND t.task_id=? AND p.status=?",
                (
                    item.goal_id,
                    item.plan_revision,
                    item.task_id,
                    PlanStatus.ACCEPTED.value,
                ),
            ).fetchone()
            if task_exists is None:
                raise NotFoundError(
                    f"delegation task is not in accepted plan r{item.plan_revision}: {item.task_id}"
                )
            if item.parent_id is not None:
                parent = connection.execute(
                    "SELECT 1 FROM delegations WHERE id=? AND goal_id=?",
                    (item.parent_id, item.goal_id),
                ).fetchone()
                if parent is None:
                    raise NotFoundError(
                        f"parent delegation not found in this goal: {item.parent_id}"
                    )
            connection.execute(
                "INSERT INTO delegations VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    item.id, item.goal_id, item.task_id, item.plan_revision, item.parent_id, item.worker_id,
                    item.brief, _json(item.role.to_dict()), item.status.value, item.attempt, item.result_summary,
                    item.error, _json(item.metadata), _iso(item.created_at), _iso(item.updated_at),
                ),
            )
            self._event(connection, "delegation.created", goal_id=item.goal_id, entity_type="delegation", entity_id=item.id, payload={"task_id": item.task_id, "role": item.role.to_dict()})
        return item

    @staticmethod
    def _delegation_from_row(row: sqlite3.Row) -> Delegation:
        return Delegation(
            id=row["id"], goal_id=row["goal_id"], task_id=row["task_id"], plan_revision=row["plan_revision"],
            parent_id=row["parent_id"], worker_id=row["worker_id"], brief=row["brief"],
            role=RoleProfile.from_dict(_loads(row["role_json"], {})), status=DelegationStatus(row["status"]),
            attempt=row["attempt"], result_summary=row["result_summary"], error=row["error"],
            metadata=_loads(row["metadata_json"], {}), created_at=_dt(row["created_at"]), updated_at=_dt(row["updated_at"]),
        )

    def transition_delegation(
        self,
        delegation_id: str,
        status: DelegationStatus | str,
        *,
        result_summary: str | None = None,
        error: str | None = None,
    ) -> Delegation:
        target = DelegationStatus(status)
        with self.transaction() as connection:
            row = connection.execute("SELECT * FROM delegations WHERE id=?", (delegation_id,)).fetchone()
            if row is None:
                raise NotFoundError(f"delegation not found: {delegation_id}")
            item = self._delegation_from_row(row)
            ensure_delegation_transition(item.status, target)
            now = utc_now()
            attempt = item.attempt + (1 if target == DelegationStatus.IN_PROGRESS else 0)
            connection.execute(
                "UPDATE delegations SET status=?,attempt=?,result_summary=?,error=?,updated_at=? WHERE id=?",
                (target.value, attempt, result_summary, error, _iso(now), delegation_id),
            )
            self._event(connection, "delegation.status_changed", goal_id=item.goal_id, entity_type="delegation", entity_id=item.id, payload={"from": item.status.value, "to": target.value})
            return replace(item, status=target, attempt=attempt, result_summary=result_summary, error=error, updated_at=now)

    def list_delegations(self, goal_id: str) -> tuple[Delegation, ...]:
        with self._lock:
            rows = self._connection.execute("SELECT * FROM delegations WHERE goal_id=? ORDER BY created_at,id", (goal_id,)).fetchall()
        return tuple(self._delegation_from_row(row) for row in rows)

    def begin_action(
        self,
        goal_id: str,
        tool_name: str,
        args: Mapping[str, Any],
        *,
        task_id: str | None = None,
        risk: str = "unknown",
        mutating: bool = False,
    ) -> str:
        action_id, now = new_id("action"), utc_now()
        encoded = _json(dict(args))
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO actions VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (action_id, goal_id, task_id, tool_name, hashlib.sha256(encoded.encode()).hexdigest(), encoded, risk, int(mutating), "running", None, _iso(now), None),
            )
            self._event(connection, "action.started", goal_id=goal_id, entity_type="action", entity_id=action_id, payload={"tool": tool_name, "task_id": task_id, "risk": risk, "mutating": mutating})
        return action_id

    def complete_action(self, action_id: str, result_summary: str, *, status: str = "completed") -> None:
        if status not in {"completed", "denied", "failed"}:
            raise ValueError("invalid action terminal status")
        with self.transaction() as connection:
            row = connection.execute("SELECT * FROM actions WHERE id=?", (action_id,)).fetchone()
            if row is None:
                raise NotFoundError(f"action not found: {action_id}")
            if row["status"] != "running":
                raise StateStoreError(f"action is already {row['status']}")
            now = utc_now()
            connection.execute("UPDATE actions SET status=?,result_summary=?,completed_at=? WHERE id=?", (status, result_summary, _iso(now), action_id))
            self._event(connection, f"action.{status}", goal_id=row["goal_id"], entity_type="action", entity_id=action_id, payload={"tool": row["tool_name"], "result": result_summary})

    def list_actions(self, goal_id: str, *, status: str | None = None) -> tuple[dict[str, Any], ...]:
        sql, params = "SELECT * FROM actions WHERE goal_id=?", [goal_id]
        if status:
            sql += " AND status=?"; params.append(status)
        sql += " ORDER BY started_at,id"
        with self._lock:
            rows = self._connection.execute(sql, tuple(params)).fetchall()
        return tuple(dict(row) for row in rows)

    def count_recent_identical_actions(
        self,
        goal_id: str,
        tool_name: str,
        args: Mapping[str, Any],
        *,
        scan_limit: int = 20,
    ) -> int:
        """Count consecutive identical journaled actions, newest first."""
        encoded = _json(dict(args))
        expected_hash = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        with self._lock:
            rows = self._connection.execute(
                "SELECT tool_name,args_hash FROM actions WHERE goal_id=? "
                "ORDER BY started_at DESC,id DESC LIMIT ?",
                (goal_id, max(1, min(scan_limit, 1_000))),
            ).fetchall()
        count = 0
        for row in rows:
            if row["tool_name"] != tool_name or row["args_hash"] != expected_hash:
                break
            count += 1
        return count

    def list_events(self, goal_id: str | None = None, *, after_sequence: int = 0, limit: int = 1_000) -> tuple[RuntimeEvent, ...]:
        sql, params = "SELECT * FROM events WHERE sequence>?", [after_sequence]
        if goal_id:
            sql += " AND goal_id=?"; params.append(goal_id)
        sql += " ORDER BY sequence LIMIT ?"; params.append(max(1, min(limit, 10_000)))
        with self._lock:
            rows = self._connection.execute(sql, tuple(params)).fetchall()
        return tuple(
            RuntimeEvent(
                id=row["event_id"], sequence=row["sequence"], event_type=row["event_type"], goal_id=row["goal_id"],
                entity_type=row["entity_type"], entity_id=row["entity_id"], payload=_loads(row["payload_json"], {}), created_at=_dt(row["created_at"]),
            )
            for row in rows
        )

    def list_recent_events(self, goal_id: str | None = None, *, limit: int = 100) -> tuple[RuntimeEvent, ...]:
        """Return the newest events in chronological display order."""
        sql, params = "SELECT * FROM events", []
        if goal_id:
            sql += " WHERE goal_id=?"; params.append(goal_id)
        sql += " ORDER BY sequence DESC LIMIT ?"; params.append(max(1, min(limit, 10_000)))
        with self._lock:
            rows = list(reversed(self._connection.execute(sql, tuple(params)).fetchall()))
        return tuple(
            RuntimeEvent(
                id=row["event_id"], sequence=row["sequence"], event_type=row["event_type"], goal_id=row["goal_id"],
                entity_type=row["entity_type"], entity_id=row["entity_id"], payload=_loads(row["payload_json"], {}), created_at=_dt(row["created_at"]),
            )
            for row in rows
        )

    def resolve_action(self, action_id: str, resolution: str, note: str, *, actor: str = "user") -> dict[str, Any]:
        """Reconcile an uncertain crash-window action after explicit inspection."""
        resolution = resolution.lower().replace("-", "_")
        if resolution not in {"applied", "not_run"}:
            raise ValueError("resolution must be 'applied' or 'not-run'")
        if not note.strip():
            raise ValueError("action resolution requires an inspection note")
        with self.transaction() as connection:
            row = connection.execute("SELECT * FROM actions WHERE id=?", (action_id,)).fetchone()
            if row is None:
                raise NotFoundError(f"action not found: {action_id}")
            if row["status"] != "uncertain":
                raise StateStoreError(f"action is {row['status']}, not uncertain")
            status = f"resolved_{resolution}"
            now = utc_now()
            summary = f"{resolution}: {note.strip()}"
            connection.execute(
                "UPDATE actions SET status=?,result_summary=?,completed_at=? WHERE id=?",
                (status, summary, _iso(now), action_id),
            )
            if row["task_id"]:
                task_rows = connection.execute(
                    "SELECT t.plan_id,t.task_id,t.status,t.metadata_json FROM tasks t JOIN plans p ON p.id=t.plan_id "
                    "WHERE t.goal_id=? AND t.task_id=? AND t.status=? AND p.status IN (?,?)",
                    (
                        row["goal_id"], row["task_id"], TaskStatus.UNCERTAIN.value,
                        PlanStatus.ACCEPTED.value, PlanStatus.PENDING_APPROVAL.value,
                    ),
                ).fetchall()
                for task_row in task_rows:
                    task_metadata = _loads(task_row["metadata_json"], {})
                    task_metadata["last_note"] = f"uncertain action reconciled by {actor}: {note.strip()}"
                    connection.execute(
                        "UPDATE tasks SET status=?,metadata_json=?,updated_at=? "
                        "WHERE plan_id=? AND task_id=?",
                        (
                            TaskStatus.IN_PROGRESS.value,
                            _json(task_metadata),
                            _iso(now), task_row["plan_id"], task_row["task_id"],
                        ),
                    )
                    self._event(
                        connection, "task.status_changed", goal_id=row["goal_id"], entity_type="task",
                        entity_id=task_row["task_id"], payload={"from": "uncertain", "to": "in_progress", "actor": actor, "note": note.strip()},
                    )
            self._event(
                connection, "action.resolved", goal_id=row["goal_id"], entity_type="action", entity_id=action_id,
                payload={"resolution": resolution, "note": note.strip(), "actor": actor},
            )
        result = dict(row)
        result.update({"status": status, "result_summary": summary, "completed_at": _iso(now)})
        return result

    def resolve_delegation(
        self,
        delegation_id: str,
        resolution: str,
        note: str,
        *,
        actor: str = "user",
    ) -> Delegation:
        """Reconcile an interrupted worker after inspecting its real effects."""
        resolution = resolution.lower().replace("-", "_")
        if resolution not in {"applied", "not_run"}:
            raise ValueError("resolution must be 'applied' or 'not-run'")
        if not note.strip():
            raise ValueError("delegation resolution requires an inspection note")
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM delegations WHERE id=?", (delegation_id,)
            ).fetchone()
            if row is None:
                raise NotFoundError(f"delegation not found: {delegation_id}")
            item = self._delegation_from_row(row)
            if item.status != DelegationStatus.UNCERTAIN:
                raise StateStoreError(
                    f"delegation is {item.status.value}, not uncertain"
                )
            target = (
                DelegationStatus.COMPLETED
                if resolution == "applied"
                else DelegationStatus.FAILED
            )
            ensure_delegation_transition(item.status, target)
            now = utc_now()
            connection.execute(
                "UPDATE delegations SET status=?,result_summary=?,error=?,updated_at=? WHERE id=?",
                (
                    target.value,
                    note.strip() if target == DelegationStatus.COMPLETED else None,
                    note.strip() if target == DelegationStatus.FAILED else None,
                    _iso(now),
                    delegation_id,
                ),
            )
            task_rows = connection.execute(
                "SELECT t.plan_id,t.task_id,t.metadata_json FROM tasks t "
                "JOIN plans p ON p.id=t.plan_id WHERE t.goal_id=? AND t.task_id=? "
                "AND t.status=? AND p.status IN (?,?)",
                (
                    item.goal_id,
                    item.task_id,
                    TaskStatus.UNCERTAIN.value,
                    PlanStatus.ACCEPTED.value,
                    PlanStatus.PENDING_APPROVAL.value,
                ),
            ).fetchall()
            for task_row in task_rows:
                metadata = _loads(task_row["metadata_json"], {})
                metadata["last_note"] = (
                    f"uncertain delegation reconciled by {actor}: {note.strip()}"
                )
                connection.execute(
                    "UPDATE tasks SET status=?,metadata_json=?,updated_at=? "
                    "WHERE plan_id=? AND task_id=?",
                    (
                        TaskStatus.IN_PROGRESS.value,
                        _json(metadata),
                        _iso(now),
                        task_row["plan_id"],
                        task_row["task_id"],
                    ),
                )
                self._event(
                    connection,
                    "task.status_changed",
                    goal_id=item.goal_id,
                    entity_type="task",
                    entity_id=task_row["task_id"],
                    payload={
                        "from": "uncertain",
                        "to": "in_progress",
                        "actor": actor,
                        "note": note.strip(),
                    },
                )
            self._event(
                connection,
                "delegation.resolved",
                goal_id=item.goal_id,
                entity_type="delegation",
                entity_id=delegation_id,
                payload={
                    "resolution": resolution,
                    "note": note.strip(),
                    "actor": actor,
                },
            )
        return replace(
            item,
            status=target,
            result_summary=note.strip() if target == DelegationStatus.COMPLETED else None,
            error=note.strip() if target == DelegationStatus.FAILED else None,
            updated_at=now,
        )

    def recover_inflight(self) -> RecoveryReport:
        """Mark crash-window work uncertain; never replay a side effect automatically."""
        with self.transaction() as connection:
            task_rows = connection.execute(
                "SELECT DISTINCT t.plan_id,t.task_id,t.goal_id FROM tasks t "
                "JOIN plans p ON p.id=t.plan_id "
                "WHERE t.status IN (?,?) AND p.status IN (?,?) AND ("
                "EXISTS(SELECT 1 FROM actions a WHERE a.goal_id=t.goal_id "
                "       AND a.task_id=t.task_id AND a.status='running') OR "
                "EXISTS(SELECT 1 FROM delegations d WHERE d.goal_id=t.goal_id "
                "       AND d.task_id=t.task_id AND d.plan_revision=t.plan_revision "
                "       AND d.status=?))",
                (
                    *(status.value for status in IN_FLIGHT_TASK_STATUSES),
                    PlanStatus.ACCEPTED.value,
                    PlanStatus.PENDING_APPROVAL.value,
                    DelegationStatus.IN_PROGRESS.value,
                ),
            ).fetchall()
            delegation_rows = connection.execute(
                "SELECT id,goal_id FROM delegations WHERE status=?",
                (DelegationStatus.IN_PROGRESS.value,),
            ).fetchall()
            action_rows = connection.execute("SELECT id,goal_id FROM actions WHERE status='running'").fetchall()
            affected_goals = {row["goal_id"] for row in task_rows} | {row["goal_id"] for row in delegation_rows} | {row["goal_id"] for row in action_rows}
            now = utc_now()
            for row in task_rows:
                connection.execute("UPDATE tasks SET status=?,updated_at=? WHERE plan_id=? AND task_id=?", (TaskStatus.UNCERTAIN.value, _iso(now), row["plan_id"], row["task_id"]))
            for row in delegation_rows:
                connection.execute("UPDATE delegations SET status=?,updated_at=? WHERE id=?", (DelegationStatus.UNCERTAIN.value, _iso(now), row["id"]))
            for row in action_rows:
                connection.execute("UPDATE actions SET status='uncertain',completed_at=? WHERE id=?", (_iso(now), row["id"]))
            recovered_goals: list[str] = []
            for goal_id in sorted(affected_goals):
                goal_row = connection.execute("SELECT * FROM goals WHERE id=?", (goal_id,)).fetchone()
                if goal_row is None:
                    continue
                status = GoalStatus(goal_row["status"])
                try:
                    ensure_goal_transition(status, GoalStatus.RECOVERING)
                except Exception:
                    continue
                connection.execute("UPDATE goals SET status=?,updated_at=? WHERE id=?", (GoalStatus.RECOVERING.value, _iso(now), goal_id))
                recovered_goals.append(goal_id)
            if task_rows or delegation_rows or action_rows:
                payload = {"tasks": [row["task_id"] for row in task_rows], "delegations": [row["id"] for row in delegation_rows], "actions": [row["id"] for row in action_rows], "goals": recovered_goals}
                if affected_goals:
                    for goal_id in sorted(affected_goals):
                        self._event(connection, "recovery.performed", goal_id=goal_id, payload=payload)
                else:
                    self._event(connection, "recovery.performed", payload=payload)
            return RecoveryReport(
                task_ids=tuple(row["task_id"] for row in task_rows),
                delegation_ids=tuple(row["id"] for row in delegation_rows),
                action_ids=tuple(row["id"] for row in action_rows),
                goal_ids=tuple(recovered_goals),
            )

    # ------------------------------------------------------------------
    # ULTRA v3 state.  These APIs are additive: legacy goals/plans remain the
    # approval authority while dynamic decomposition lives in work_nodes.

    def create_ultra_run(
        self,
        run: UltraRun | None = None,
        **kwargs: Any,
    ) -> UltraRun:
        item = run or UltraRun(**kwargs)
        with self.transaction() as connection:
            if connection.execute("SELECT 1 FROM goals WHERE id=?", (item.goal_id,)).fetchone() is None:
                raise NotFoundError(f"goal not found: {item.goal_id}")
            active = connection.execute(
                "SELECT id FROM ultra_runs WHERE goal_id=? AND status NOT IN (?,?,?) LIMIT 1",
                (
                    item.goal_id,
                    UltraRunStatus.COMPLETED.value,
                    UltraRunStatus.CANCELLED.value,
                    UltraRunStatus.BLOCKED.value,
                ),
            ).fetchone()
            if active:
                raise ActiveGoalError(f"unfinished ULTRA run already exists: {active['id']}")
            connection.execute(
                "INSERT INTO ultra_runs("
                "id,goal_id,provider,model,execution_class,access_level,concurrency,phase,status,"
                "goal_spec_json,architecture_spec_json,plan_revision,master_plan_fingerprint,"
                "master_approved,config_json,error,created_at,updated_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    item.id,
                    item.goal_id,
                    item.provider,
                    item.model,
                    item.execution_class.value,
                    item.access_level.value,
                    item.concurrency,
                    item.phase.value,
                    item.status.value,
                    _json(item.goal_spec.to_dict()) if item.goal_spec else None,
                    _json(item.architecture_spec.to_dict()) if item.architecture_spec else None,
                    item.plan_revision,
                    item.master_plan_fingerprint,
                    int(item.master_approved),
                    _json(item.config),
                    item.error,
                    _iso(item.created_at),
                    _iso(item.updated_at),
                ),
            )
            self._event(
                connection,
                "ultra.run_created",
                goal_id=item.goal_id,
                entity_type="ultra_run",
                entity_id=item.id,
                payload={
                    "provider": item.provider,
                    "model": item.model,
                    "execution_class": item.execution_class.value,
                    "concurrency": item.concurrency,
                },
            )
        return item

    @staticmethod
    def _ultra_run_from_row(row: sqlite3.Row) -> UltraRun:
        goal_spec = _loads(row["goal_spec_json"], None)
        architecture = _loads(row["architecture_spec_json"], None)
        return UltraRun(
            id=row["id"],
            goal_id=row["goal_id"],
            provider=row["provider"],
            model=row["model"],
            execution_class=ExecutionClass(row["execution_class"]),
            access_level=AccessLevel(row["access_level"]),
            concurrency=row["concurrency"],
            phase=UltraPhase(row["phase"]),
            status=UltraRunStatus(row["status"]),
            goal_spec=GoalSpecV1.from_dict(goal_spec) if goal_spec else None,
            architecture_spec=ArchitectureSpecV1.from_dict(architecture) if architecture else None,
            plan_revision=row["plan_revision"],
            master_plan_fingerprint=row["master_plan_fingerprint"],
            master_approved=bool(row["master_approved"]),
            config=_loads(row["config_json"], {}),
            error=row["error"],
            created_at=_dt(row["created_at"]),
            updated_at=_dt(row["updated_at"]),
        )

    def get_ultra_run(self, run_id: str) -> UltraRun:
        with self._lock:
            row = self._connection.execute("SELECT * FROM ultra_runs WHERE id=?", (run_id,)).fetchone()
        if row is None:
            raise NotFoundError(f"ULTRA run not found: {run_id}")
        return self._ultra_run_from_row(row)

    def get_active_ultra_run(self, goal_id: str | None = None) -> UltraRun | None:
        sql = "SELECT * FROM ultra_runs WHERE status NOT IN (?,?,?)"
        params: list[Any] = [
            UltraRunStatus.COMPLETED.value,
            UltraRunStatus.CANCELLED.value,
            UltraRunStatus.BLOCKED.value,
        ]
        if goal_id:
            sql += " AND goal_id=?"
            params.append(goal_id)
        sql += " ORDER BY updated_at DESC LIMIT 1"
        with self._lock:
            row = self._connection.execute(sql, tuple(params)).fetchone()
        return self._ultra_run_from_row(row) if row else None

    def list_ultra_runs(self, goal_id: str | None = None) -> tuple[UltraRun, ...]:
        sql, params = "SELECT * FROM ultra_runs", []
        if goal_id:
            sql += " WHERE goal_id=?"
            params.append(goal_id)
        sql += " ORDER BY created_at,id"
        with self._lock:
            rows = self._connection.execute(sql, tuple(params)).fetchall()
        return tuple(self._ultra_run_from_row(row) for row in rows)

    def update_ultra_run(
        self,
        run_id: str,
        *,
        provider: str | None = None,
        model: str | None = None,
        execution_class: ExecutionClass | str | None = None,
        access_level: AccessLevel | str | None = None,
        concurrency: int | None = None,
        phase: UltraPhase | str | None = None,
        status: UltraRunStatus | str | None = None,
        goal_spec: GoalSpecV1 | Mapping[str, Any] | None = None,
        architecture_spec: ArchitectureSpecV1 | Mapping[str, Any] | None = None,
        config: Mapping[str, Any] | None = None,
        error: str | None = None,
    ) -> UltraRun:
        current = self.get_ultra_run(run_id)
        next_provider = str(provider).strip() if provider is not None else current.provider
        next_model = str(model).strip() if model is not None else current.model
        next_execution = (
            ExecutionClass(execution_class)
            if execution_class is not None
            else current.execution_class
        )
        next_access = AccessLevel(access_level) if access_level is not None else current.access_level
        next_concurrency = int(concurrency) if concurrency is not None else current.concurrency
        if not next_provider or not next_model or not 1 <= next_concurrency <= 8:
            raise ValueError("invalid ULTRA provider/model/concurrency update")
        if next_execution is ExecutionClass.LOCAL and next_concurrency != 1:
            raise ValueError("local ULTRA execution must remain sequential")
        next_phase = UltraPhase(phase) if phase is not None else current.phase
        next_status = UltraRunStatus(status) if status is not None else current.status
        next_goal_spec = (
            GoalSpecV1.from_dict(goal_spec) if isinstance(goal_spec, Mapping) else goal_spec
        ) if goal_spec is not None else current.goal_spec
        next_architecture = (
            ArchitectureSpecV1.from_dict(architecture_spec)
            if isinstance(architecture_spec, Mapping)
            else architecture_spec
        ) if architecture_spec is not None else current.architecture_spec
        next_config = dict(current.config)
        if config is not None:
            next_config.update(config)
        now = utc_now()
        with self.transaction() as connection:
            cursor = connection.execute(
                "UPDATE ultra_runs SET provider=?,model=?,execution_class=?,access_level=?,concurrency=?,"
                "phase=?,status=?,goal_spec_json=?,architecture_spec_json=?,config_json=?,error=?,updated_at=? "
                "WHERE id=? AND updated_at=?",
                (
                    next_provider,
                    next_model,
                    next_execution.value,
                    next_access.value,
                    next_concurrency,
                    next_phase.value,
                    next_status.value,
                    _json(next_goal_spec.to_dict()) if next_goal_spec else None,
                    _json(next_architecture.to_dict()) if next_architecture else None,
                    _json(next_config),
                    error,
                    _iso(now),
                    run_id,
                    _iso(current.updated_at),
                ),
            )
            if not cursor.rowcount:
                raise StateStoreError("ULTRA run changed concurrently")
            self._event(
                connection,
                "ultra.run_updated",
                goal_id=current.goal_id,
                entity_type="ultra_run",
                entity_id=run_id,
                payload={"phase": next_phase.value, "status": next_status.value},
            )
        return replace(
            current,
            provider=next_provider,
            model=next_model,
            execution_class=next_execution,
            access_level=next_access,
            concurrency=next_concurrency,
            phase=next_phase,
            status=next_status,
            goal_spec=next_goal_spec,
            architecture_spec=next_architecture,
            config=next_config,
            error=error,
            updated_at=now,
        )

    def approve_ultra_master(
        self,
        run_id: str,
        plan_revision: int,
        fingerprint: str,
        *,
        approved_by: str = "user",
    ) -> UltraRun:
        """Bind ULTRA execution to the already accepted legacy master plan."""
        run = self.get_ultra_run(run_id)
        with self.transaction() as connection:
            plan = connection.execute(
                "SELECT * FROM plans WHERE goal_id=? AND revision=?",
                (run.goal_id, plan_revision),
            ).fetchone()
            if plan is None:
                raise MasterApprovalError("master plan revision does not exist")
            if plan["status"] != PlanStatus.ACCEPTED.value:
                raise MasterApprovalError("master plan must be accepted first")
            if not fingerprint or plan["fingerprint"] != fingerprint:
                raise MasterApprovalError("master plan fingerprint does not match")
            now = utc_now()
            connection.execute(
                "UPDATE ultra_runs SET plan_revision=?,master_plan_fingerprint=?,master_approved=1,"
                "master_approved_by=?,master_approved_at=?,phase=?,status=?,updated_at=? WHERE id=?",
                (
                    plan_revision,
                    fingerprint,
                    approved_by,
                    _iso(now),
                    UltraPhase.MODULE_WAVES.value,
                    UltraRunStatus.RUNNING.value,
                    _iso(now),
                    run_id,
                ),
            )
            self._event(
                connection,
                "ultra.master_approved",
                goal_id=run.goal_id,
                entity_type="ultra_run",
                entity_id=run_id,
                payload={"revision": plan_revision, "fingerprint": fingerprint},
            )
        return replace(
            run,
            plan_revision=plan_revision,
            master_plan_fingerprint=fingerprint,
            master_approved=True,
            phase=UltraPhase.MODULE_WAVES,
            status=UltraRunStatus.RUNNING,
            updated_at=now,
        )

    @staticmethod
    def _work_node_from_row(row: sqlite3.Row) -> WorkNode:
        result = _loads(row["result_json"], None)
        return WorkNode(
            id=row["id"],
            ultra_run_id=row["ultra_run_id"],
            parent_id=row["parent_id"],
            master_task_id=row["master_task_id"],
            kind=WorkNodeKind(row["kind"]),
            title=row["title"],
            objective=row["objective"],
            status=WorkNodeStatus(row["status"]),
            depth=row["depth"],
            position=row["position"],
            depends_on=tuple(_loads(row["depends_on_json"], [])),
            contract=TaskContractV1.from_dict(_loads(row["contract_json"], {})),
            assigned_role=row["assigned_role"],
            attempts=row["attempts"],
            max_attempts=row["max_attempts"],
            result=ResultPackageV1.from_dict(result) if result else None,
            error=row["error"],
            checkpoint=row["checkpoint"],
            metadata=_loads(row["metadata_json"], {}),
            created_at=_dt(row["created_at"]),
            updated_at=_dt(row["updated_at"]),
        )

    def create_work_node(self, node: WorkNode | None = None, **kwargs: Any) -> WorkNode:
        item = node or WorkNode(**kwargs)
        run = self.get_ultra_run(item.ultra_run_id)
        config = dict(run.config)
        max_depth = max(1, min(int(config.get("max_depth", 5)), 12))
        max_nodes = max(1, min(int(config.get("max_nodes", 500)), 5_000))
        with self.transaction() as connection:
            count = connection.execute(
                "SELECT COUNT(*) FROM work_nodes WHERE ultra_run_id=?", (item.ultra_run_id,)
            ).fetchone()[0]
            if count >= max_nodes:
                raise StateStoreError(f"ULTRA work-node cap reached ({max_nodes})")
            parent: WorkNode | None = None
            depth = 0
            master_task_id = item.master_task_id
            if item.parent_id:
                row = connection.execute("SELECT * FROM work_nodes WHERE id=?", (item.parent_id,)).fetchone()
                if row is None:
                    raise NotFoundError(f"parent work node not found: {item.parent_id}")
                parent = self._work_node_from_row(row)
                if parent.ultra_run_id != item.ultra_run_id:
                    raise StateStoreError("parent work node belongs to another ULTRA run")
                depth = parent.depth + 1
                if depth > max_depth:
                    raise StateStoreError(f"ULTRA decomposition depth exceeds {max_depth}")
                assert_child_contract(parent.contract, item.contract)
                if item.master_task_id and item.master_task_id != parent.master_task_id:
                    raise ContractScopeError(("child changes its approved master-module binding",))
                master_task_id = parent.master_task_id
            elif item.kind == WorkNodeKind.MODULE:
                if not run.master_approved or not item.master_task_id:
                    raise MasterApprovalError("a master module requires approved plan-task binding")
                task = connection.execute(
                    "SELECT 1 FROM tasks t JOIN plans p ON p.id=t.plan_id "
                    "WHERE t.goal_id=? AND t.plan_revision=? AND t.task_id=? AND p.status=?",
                    (
                        run.goal_id,
                        run.plan_revision,
                        item.master_task_id,
                        PlanStatus.ACCEPTED.value,
                    ),
                ).fetchone()
                if task is None:
                    raise MasterApprovalError("master module is not present in the accepted plan")
            for dependency in item.depends_on:
                row = connection.execute(
                    "SELECT ultra_run_id FROM work_nodes WHERE id=?", (dependency,)
                ).fetchone()
                if row is None or row["ultra_run_id"] != item.ultra_run_id:
                    raise StateStoreError(f"work-node dependency is missing or foreign: {dependency}")
                if dependency == item.id:
                    raise StateStoreError("a work node cannot depend on itself")
            stored = replace(item, depth=depth, master_task_id=master_task_id)
            connection.execute(
                "INSERT INTO work_nodes("
                "id,ultra_run_id,parent_id,master_task_id,kind,title,objective,status,depth,position,"
                "depends_on_json,contract_json,assigned_role,attempts,max_attempts,result_json,error,"
                "checkpoint,metadata_json,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    stored.id,
                    stored.ultra_run_id,
                    stored.parent_id,
                    stored.master_task_id,
                    stored.kind.value,
                    stored.title,
                    stored.objective,
                    stored.status.value,
                    stored.depth,
                    stored.position,
                    _json(stored.depends_on),
                    _json(stored.contract.to_dict()),
                    stored.assigned_role,
                    stored.attempts,
                    stored.max_attempts,
                    _json(stored.result.to_dict()) if stored.result else None,
                    stored.error,
                    stored.checkpoint,
                    _json(stored.metadata),
                    _iso(stored.created_at),
                    _iso(stored.updated_at),
                ),
            )
            self._event(
                connection,
                "ultra.node_created",
                goal_id=run.goal_id,
                entity_type="work_node",
                entity_id=stored.id,
                payload={"kind": stored.kind.value, "parent_id": stored.parent_id, "depth": stored.depth},
            )
        return stored

    def get_work_node(self, node_id: str) -> WorkNode:
        with self._lock:
            row = self._connection.execute("SELECT * FROM work_nodes WHERE id=?", (node_id,)).fetchone()
        if row is None:
            raise NotFoundError(f"work node not found: {node_id}")
        return self._work_node_from_row(row)

    def list_work_nodes(
        self,
        run_id: str,
        *,
        parent_id: str | None = None,
        status: WorkNodeStatus | str | None = None,
        recursive: bool = True,
    ) -> tuple[WorkNode, ...]:
        sql, params = "SELECT * FROM work_nodes WHERE ultra_run_id=?", [run_id]
        if not recursive:
            if parent_id is None:
                sql += " AND parent_id IS NULL"
            else:
                sql += " AND parent_id=?"
                params.append(parent_id)
        elif parent_id is not None:
            # A recursive CTE keeps hierarchy reads deterministic and bounded.
            sql = (
                "WITH RECURSIVE subtree(id) AS (SELECT id FROM work_nodes WHERE id=? AND ultra_run_id=? "
                "UNION ALL SELECT w.id FROM work_nodes w JOIN subtree s ON w.parent_id=s.id) "
                "SELECT * FROM work_nodes WHERE id IN (SELECT id FROM subtree)"
            )
            params = [parent_id, run_id]
        if status is not None:
            sql += " AND status=?"
            params.append(WorkNodeStatus(status).value)
        sql += " ORDER BY depth,position,created_at,id"
        with self._lock:
            rows = self._connection.execute(sql, tuple(params)).fetchall()
        return tuple(self._work_node_from_row(row) for row in rows)

    def work_node_ancestors(self, node_id: str) -> tuple[WorkNode, ...]:
        node = self.get_work_node(node_id)
        result: list[WorkNode] = []
        current = node
        seen = {node.id}
        while current.parent_id:
            current = self.get_work_node(current.parent_id)
            if current.id in seen or current.ultra_run_id != node.ultra_run_id:
                raise StateCorruptionError("invalid work-node parent hierarchy")
            seen.add(current.id)
            result.append(current)
        result.reverse()
        return tuple(result)

    def transition_work_node(
        self,
        node_id: str,
        status: WorkNodeStatus | str,
        *,
        result: ResultPackageV1 | Mapping[str, Any] | None = None,
        error: str | None = None,
        checkpoint: str | None = None,
        increment_attempt: bool = False,
    ) -> WorkNode:
        node = self.get_work_node(node_id)
        target = WorkNodeStatus(status)
        package = ResultPackageV1.from_dict(result) if isinstance(result, Mapping) else result
        if target == WorkNodeStatus.IN_PROGRESS and node.depends_on:
            with self._lock:
                rows = self._connection.execute(
                    "SELECT id,status FROM work_nodes WHERE id IN (%s)" % ",".join("?" for _ in node.depends_on),
                    tuple(node.depends_on),
                ).fetchall()
            states = {row["id"]: row["status"] for row in rows}
            incomplete = [dep for dep in node.depends_on if states.get(dep) != WorkNodeStatus.COMPLETED.value]
            if incomplete:
                raise CompletionGateError(f"work-node dependencies are incomplete: {incomplete!r}")
        if target == WorkNodeStatus.COMPLETED and package is None and node.result is None:
            raise CompletionGateError("completing a work node requires a result package")
        attempts = node.attempts + int(increment_attempt)
        if attempts > node.max_attempts and target in {WorkNodeStatus.IN_PROGRESS, WorkNodeStatus.FIXING}:
            target = WorkNodeStatus.REVISION_REQUIRED
            error = error or "bounded fix attempts exhausted"
        now = utc_now()
        final_result = package or node.result
        with self.transaction() as connection:
            cursor = connection.execute(
                "UPDATE work_nodes SET status=?,attempts=?,result_json=?,error=?,checkpoint=?,updated_at=? "
                "WHERE id=? AND updated_at=?",
                (
                    target.value,
                    attempts,
                    _json(final_result.to_dict()) if final_result else None,
                    error,
                    checkpoint if checkpoint is not None else node.checkpoint,
                    _iso(now),
                    node_id,
                    _iso(node.updated_at),
                ),
            )
            if not cursor.rowcount:
                raise StateStoreError("work node changed concurrently")
            run = self._ultra_run_from_row(
                connection.execute("SELECT * FROM ultra_runs WHERE id=?", (node.ultra_run_id,)).fetchone()
            )
            self._event(
                connection,
                "ultra.node_status_changed",
                goal_id=run.goal_id,
                entity_type="work_node",
                entity_id=node_id,
                payload={"from": node.status.value, "to": target.value, "attempts": attempts},
            )
        return replace(
            node,
            status=target,
            attempts=attempts,
            result=final_result,
            error=error,
            checkpoint=checkpoint if checkpoint is not None else node.checkpoint,
            updated_at=now,
        )

    def sync_master_modules(self, run_id: str) -> tuple[WorkNode, ...]:
        """Materialize approved plan tasks without changing their fingerprint."""
        run = self.get_ultra_run(run_id)
        if not run.master_approved or run.plan_revision is None:
            raise MasterApprovalError("ULTRA master plan is not approved")
        plan = self.get_plan(run.goal_id, run.plan_revision)
        existing = {node.master_task_id: node for node in self.list_work_nodes(run_id) if node.is_master_module}
        modules: list[WorkNode] = []
        architecture_interfaces = run.architecture_spec.interfaces if run.architecture_spec else {}
        for position, task in enumerate(plan.tasks):
            if task.id in existing:
                modules.append(existing[task.id])
                continue
            write_paths = tuple(
                str(change.get("path"))
                for change in plan.expected_changes
                if task.id in {str(value).strip().upper() for value in change.get("supports_tasks", ())}
            )
            contract = TaskContractV1(
                objective=task.description or task.title,
                success_criteria=task.acceptance_criteria,
                write_paths=write_paths or (".",),
                forbidden_changes=tuple(self.get_goal(run.goal_id).constraints),
                interfaces=architecture_interfaces,
                metadata={"plan_revision": plan.revision, "task_id": task.id},
            )
            modules.append(
                self.create_work_node(
                    WorkNode(
                        ultra_run_id=run_id,
                        title=task.title,
                        objective=task.description or task.title,
                        contract=contract,
                        kind=WorkNodeKind.MODULE,
                        master_task_id=task.id,
                        position=position,
                        depends_on=tuple(
                            existing[dep].id for dep in task.depends_on if dep in existing
                        ),
                        assigned_role=task.role.name,
                    )
                )
            )
            existing[task.id] = modules[-1]
        return tuple(modules)

    @staticmethod
    def _agent_run_from_row(row: sqlite3.Row) -> AgentRun:
        result = _loads(row["result_json"], None)
        return AgentRun(
            id=row["id"],
            ultra_run_id=row["ultra_run_id"],
            work_node_id=row["work_node_id"],
            role=row["role"],
            provider=row["provider"],
            model=row["model"],
            phase=row["phase"],
            status=AgentRunStatus(row["status"]),
            attempt=row["attempt"],
            usage=_loads(row["usage_json"], {}),
            result=ResultPackageV1.from_dict(result) if result else None,
            error=row["error"],
            prompt_trace_id=row["prompt_trace_id"],
            side_effects=bool(row["side_effects"]),
            started_at=_dt(row["started_at"]),
            updated_at=_dt(row["updated_at"]),
            finished_at=_dt(row["finished_at"]),
        )

    def create_agent_run(self, agent_run: AgentRun | None = None, **kwargs: Any) -> AgentRun:
        item = agent_run or AgentRun(**kwargs)
        with self.transaction() as connection:
            run = connection.execute("SELECT goal_id FROM ultra_runs WHERE id=?", (item.ultra_run_id,)).fetchone()
            if run is None:
                raise NotFoundError(f"ULTRA run not found: {item.ultra_run_id}")
            if item.work_node_id:
                node = connection.execute(
                    "SELECT ultra_run_id FROM work_nodes WHERE id=?", (item.work_node_id,)
                ).fetchone()
                if node is None or node["ultra_run_id"] != item.ultra_run_id:
                    raise StateStoreError("agent work node is missing or belongs to another run")
            connection.execute(
                "INSERT INTO agent_runs("
                "id,ultra_run_id,work_node_id,role,provider,model,phase,status,attempt,usage_json,"
                "result_json,error,prompt_trace_id,side_effects,started_at,updated_at,finished_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    item.id,
                    item.ultra_run_id,
                    item.work_node_id,
                    item.role,
                    item.provider,
                    item.model,
                    item.phase,
                    item.status.value,
                    item.attempt,
                    _json(item.usage),
                    _json(item.result.to_dict()) if item.result else None,
                    item.error,
                    item.prompt_trace_id,
                    int(item.side_effects),
                    _iso(item.started_at),
                    _iso(item.updated_at),
                    _iso(item.finished_at),
                ),
            )
            self._event(
                connection,
                "ultra.agent_created",
                goal_id=run["goal_id"],
                entity_type="agent_run",
                entity_id=item.id,
                payload={"role": item.role, "node_id": item.work_node_id, "phase": item.phase},
            )
        return item

    def get_agent_run(self, agent_run_id: str) -> AgentRun:
        with self._lock:
            row = self._connection.execute("SELECT * FROM agent_runs WHERE id=?", (agent_run_id,)).fetchone()
        if row is None:
            raise NotFoundError(f"agent run not found: {agent_run_id}")
        return self._agent_run_from_row(row)

    def list_agent_runs(
        self,
        run_id: str,
        *,
        work_node_id: str | None = None,
        status: AgentRunStatus | str | None = None,
    ) -> tuple[AgentRun, ...]:
        sql, params = "SELECT * FROM agent_runs WHERE ultra_run_id=?", [run_id]
        if work_node_id:
            sql += " AND work_node_id=?"
            params.append(work_node_id)
        if status is not None:
            sql += " AND status=?"
            params.append(AgentRunStatus(status).value)
        sql += " ORDER BY started_at,id"
        with self._lock:
            rows = self._connection.execute(sql, tuple(params)).fetchall()
        return tuple(self._agent_run_from_row(row) for row in rows)

    def update_agent_run(
        self,
        agent_run_id: str,
        status: AgentRunStatus | str,
        *,
        usage: Mapping[str, Any] | None = None,
        result: ResultPackageV1 | Mapping[str, Any] | None = None,
        error: str | None = None,
        prompt_trace_id: str | None = None,
        side_effects: bool | None = None,
    ) -> AgentRun:
        item = self.get_agent_run(agent_run_id)
        target = AgentRunStatus(status)
        package = ResultPackageV1.from_dict(result) if isinstance(result, Mapping) else result
        terminal = target in {
            AgentRunStatus.COMPLETED,
            AgentRunStatus.FAILED,
            AgentRunStatus.CANCELLED,
            AgentRunStatus.UNCERTAIN,
        }
        now = utc_now()
        next_usage = dict(item.usage)
        if usage:
            next_usage.update(usage)
        with self.transaction() as connection:
            cursor = connection.execute(
                "UPDATE agent_runs SET status=?,usage_json=?,result_json=?,error=?,prompt_trace_id=?,"
                "side_effects=?,updated_at=?,finished_at=? WHERE id=? AND updated_at=?",
                (
                    target.value,
                    _json(next_usage),
                    _json(package.to_dict()) if package else (
                        _json(item.result.to_dict()) if item.result else None
                    ),
                    error,
                    prompt_trace_id if prompt_trace_id is not None else item.prompt_trace_id,
                    int(side_effects if side_effects is not None else item.side_effects),
                    _iso(now),
                    _iso(now) if terminal else None,
                    agent_run_id,
                    _iso(item.updated_at),
                ),
            )
            if not cursor.rowcount:
                raise StateStoreError("agent run changed concurrently")
            goal_id = connection.execute(
                "SELECT goal_id FROM ultra_runs WHERE id=?", (item.ultra_run_id,)
            ).fetchone()[0]
            self._event(
                connection,
                "ultra.agent_status_changed",
                goal_id=goal_id,
                entity_type="agent_run",
                entity_id=agent_run_id,
                payload={"from": item.status.value, "to": target.value, "role": item.role},
            )
        return replace(
            item,
            status=target,
            usage=next_usage,
            result=package or item.result,
            error=error,
            prompt_trace_id=prompt_trace_id if prompt_trace_id is not None else item.prompt_trace_id,
            side_effects=side_effects if side_effects is not None else item.side_effects,
            updated_at=now,
            finished_at=now if terminal else None,
        )

    @staticmethod
    def _brain_entry_from_row(row: sqlite3.Row) -> BrainEntry:
        return BrainEntry(
            id=row["id"],
            ultra_run_id=row["ultra_run_id"],
            goal_id=row["goal_id"],
            work_node_id=row["work_node_id"],
            agent_run_id=row["agent_run_id"],
            section=BrainSection(row["section"]),
            title=row["title"],
            content=row["content"],
            data=_loads(row["data_json"], {}),
            role=row["role"],
            version=row["version"],
            supersedes_id=row["supersedes_id"],
            expires_at=_dt(row["expires_at"]),
            metadata=_loads(row["metadata_json"], {}),
            created_at=_dt(row["created_at"]),
            updated_at=_dt(row["updated_at"]),
        )

    def put_brain_entry(
        self,
        entry: BrainEntry | None = None,
        *,
        expected_version: int | None = None,
        **kwargs: Any,
    ) -> BrainEntry:
        proposed = entry or BrainEntry(**kwargs)
        with self.transaction() as connection:
            run = connection.execute(
                "SELECT goal_id FROM ultra_runs WHERE id=?", (proposed.ultra_run_id,)
            ).fetchone()
            if run is None or run["goal_id"] != proposed.goal_id:
                raise StateStoreError("brain entry goal does not match its ULTRA run")
            if proposed.work_node_id:
                node = connection.execute(
                    "SELECT ultra_run_id FROM work_nodes WHERE id=?", (proposed.work_node_id,)
                ).fetchone()
                if node is None or node["ultra_run_id"] != proposed.ultra_run_id:
                    raise StateStoreError("brain entry work node is missing or foreign")
            latest = connection.execute(
                "SELECT * FROM brain_entries WHERE ultra_run_id=? AND section=? AND title=? "
                "AND COALESCE(role,'')=COALESCE(?, '') ORDER BY version DESC LIMIT 1",
                (
                    proposed.ultra_run_id,
                    proposed.section.value,
                    proposed.title,
                    proposed.role,
                ),
            ).fetchone()
            current_version = latest["version"] if latest else 0
            if expected_version is not None and expected_version != current_version:
                raise ConcurrentBrainUpdateError(
                    f"brain entry expected version {expected_version}, found {current_version}"
                )
            now = utc_now()
            stored = replace(
                proposed,
                version=current_version + 1,
                supersedes_id=latest["id"] if latest else None,
                created_at=now,
                updated_at=now,
            )
            connection.execute(
                "INSERT INTO brain_entries("
                "id,ultra_run_id,goal_id,work_node_id,agent_run_id,section,title,content,data_json,role,"
                "version,supersedes_id,expires_at,metadata_json,created_at,updated_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    stored.id,
                    stored.ultra_run_id,
                    stored.goal_id,
                    stored.work_node_id,
                    stored.agent_run_id,
                    stored.section.value,
                    stored.title,
                    stored.content,
                    _json(stored.data),
                    stored.role,
                    stored.version,
                    stored.supersedes_id,
                    _iso(stored.expires_at),
                    _json(stored.metadata),
                    _iso(stored.created_at),
                    _iso(stored.updated_at),
                ),
            )
            if self._fts5_available:
                connection.execute(
                    "INSERT INTO brain_entries_fts(entry_id,title,content,section,role) VALUES(?,?,?,?,?)",
                    (stored.id, stored.title, stored.content, stored.section.value, stored.role or ""),
                )
            self._event(
                connection,
                "ultra.brain_updated",
                goal_id=stored.goal_id,
                entity_type="brain_entry",
                entity_id=stored.id,
                payload={"section": stored.section.value, "title": stored.title, "version": stored.version},
            )
        return stored

    def get_brain_entry(self, entry_id: str) -> BrainEntry:
        with self._lock:
            row = self._connection.execute("SELECT * FROM brain_entries WHERE id=?", (entry_id,)).fetchone()
        if row is None:
            raise NotFoundError(f"brain entry not found: {entry_id}")
        return self._brain_entry_from_row(row)

    def list_brain_entries(
        self,
        run_id: str,
        *,
        section: BrainSection | str | None = None,
        role: str | None = None,
        work_node_id: str | None = None,
        latest_only: bool = True,
        include_expired: bool = False,
        limit: int = 1_000,
    ) -> tuple[BrainEntry, ...]:
        filters, params = ["b.ultra_run_id=?"], [run_id]
        if section is not None:
            filters.append("b.section=?")
            params.append(BrainSection(section).value)
        if role is not None:
            filters.append("b.role=?")
            params.append(role)
        if work_node_id is not None:
            filters.append("b.work_node_id=?")
            params.append(work_node_id)
        if not include_expired:
            filters.append("(b.expires_at IS NULL OR b.expires_at>?)")
            params.append(_iso(utc_now()))
        if latest_only:
            filters.append(
                "NOT EXISTS(SELECT 1 FROM brain_entries newer WHERE newer.supersedes_id=b.id)"
            )
        sql = "SELECT b.* FROM brain_entries b WHERE " + " AND ".join(filters)
        sql += " ORDER BY b.updated_at DESC,b.id DESC LIMIT ?"
        params.append(max(1, min(limit, 10_000)))
        with self._lock:
            rows = self._connection.execute(sql, tuple(params)).fetchall()
        return tuple(self._brain_entry_from_row(row) for row in rows)

    def search_brain(
        self,
        run_id: str,
        query: str,
        *,
        section: BrainSection | str | None = None,
        role: str | None = None,
        limit: int = 20,
    ) -> tuple[BrainEntry, ...]:
        query = str(query).strip()
        if not query:
            return self.list_brain_entries(run_id, section=section, role=role, limit=limit)
        rows: list[sqlite3.Row] = []
        if self._fts5_available:
            filters, params = ["b.ultra_run_id=?", "brain_entries_fts MATCH ?"], [run_id, query]
            if section is not None:
                filters.append("b.section=?")
                params.append(BrainSection(section).value)
            if role is not None:
                filters.append("b.role=?")
                params.append(role)
            sql = (
                "SELECT b.* FROM brain_entries_fts f JOIN brain_entries b ON b.id=f.entry_id WHERE "
                + " AND ".join(filters)
                + " AND (b.expires_at IS NULL OR b.expires_at>?) "
                "ORDER BY bm25(brain_entries_fts),b.updated_at DESC LIMIT ?"
            )
            params.extend((_iso(utc_now()), max(1, min(limit, 100))))
            try:
                with self._lock:
                    rows = self._connection.execute(sql, tuple(params)).fetchall()
            except sqlite3.OperationalError:
                rows = []
        if not rows:
            filters, params = ["ultra_run_id=?", "(title LIKE ? ESCAPE '\\' OR content LIKE ? ESCAPE '\\')"], [run_id]
            escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            params.extend((f"%{escaped}%", f"%{escaped}%"))
            if section is not None:
                filters.append("section=?")
                params.append(BrainSection(section).value)
            if role is not None:
                filters.append("role=?")
                params.append(role)
            filters.append("(expires_at IS NULL OR expires_at>?)")
            params.append(_iso(utc_now()))
            sql = "SELECT * FROM brain_entries WHERE " + " AND ".join(filters)
            sql += " ORDER BY updated_at DESC LIMIT ?"
            params.append(max(1, min(limit, 100)))
            with self._lock:
                rows = self._connection.execute(sql, tuple(params)).fetchall()
        # Search returns the latest logical version of a matching entry.
        seen: set[tuple[str, str, str]] = set()
        result: list[BrainEntry] = []
        for row in rows:
            item = self._brain_entry_from_row(row)
            key = (item.section.value, item.title, item.role or "")
            if key not in seen:
                seen.add(key)
                result.append(item)
        return tuple(result[:limit])

    def record_memory_access(
        self,
        run_id: str,
        *,
        direction: str,
        query: str = "",
        work_node_id: str | None = None,
        agent_run_id: str | None = None,
        brain_entry_id: str | None = None,
        score: float | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        if direction not in {"read", "write"}:
            raise ValueError("memory direction must be 'read' or 'write'")
        access_id = new_id("memory")
        with self.transaction() as connection:
            if connection.execute("SELECT 1 FROM ultra_runs WHERE id=?", (run_id,)).fetchone() is None:
                raise NotFoundError(f"ULTRA run not found: {run_id}")
            connection.execute(
                "INSERT INTO memory_access("
                "id,ultra_run_id,work_node_id,agent_run_id,brain_entry_id,direction,query,score,metadata_json,created_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    access_id,
                    run_id,
                    work_node_id,
                    agent_run_id,
                    brain_entry_id,
                    direction,
                    str(query)[:4_000],
                    score,
                    _json(metadata or {}),
                    _iso(utc_now()),
                ),
            )
        return access_id

    def list_memory_access(self, run_id: str, *, limit: int = 100) -> tuple[dict[str, Any], ...]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM memory_access WHERE ultra_run_id=? ORDER BY created_at DESC,id DESC LIMIT ?",
                (run_id, max(1, min(limit, 10_000))),
            ).fetchall()
        return tuple(
            {
                **dict(row),
                "metadata": _loads(row["metadata_json"], {}),
            }
            for row in rows
        )

    @staticmethod
    def _artifact_from_row(row: sqlite3.Row) -> Artifact:
        return Artifact(
            id=row["id"],
            ultra_run_id=row["ultra_run_id"],
            work_node_id=row["work_node_id"],
            agent_run_id=row["agent_run_id"],
            kind=row["kind"],
            uri=row["uri"],
            path=row["path"],
            content_hash=row["content_hash"],
            pre_write_hash=row["pre_write_hash"],
            evidence=_loads(row["evidence_json"], {}),
            metadata=_loads(row["metadata_json"], {}),
            created_at=_dt(row["created_at"]),
        )

    def add_artifact(self, artifact: Artifact | None = None, **kwargs: Any) -> Artifact:
        item = artifact or Artifact(**kwargs)
        with self.transaction() as connection:
            run = connection.execute(
                "SELECT goal_id FROM ultra_runs WHERE id=?", (item.ultra_run_id,)
            ).fetchone()
            if run is None:
                raise NotFoundError(f"ULTRA run not found: {item.ultra_run_id}")
            if item.work_node_id:
                node = connection.execute(
                    "SELECT ultra_run_id FROM work_nodes WHERE id=?", (item.work_node_id,)
                ).fetchone()
                if node is None or node["ultra_run_id"] != item.ultra_run_id:
                    raise StateStoreError("artifact work node is missing or foreign")
            connection.execute(
                "INSERT INTO artifacts("
                "id,ultra_run_id,work_node_id,agent_run_id,kind,uri,path,content_hash,pre_write_hash,"
                "evidence_json,metadata_json,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    item.id,
                    item.ultra_run_id,
                    item.work_node_id,
                    item.agent_run_id,
                    item.kind,
                    item.uri,
                    item.path,
                    item.content_hash,
                    item.pre_write_hash,
                    _json(item.evidence),
                    _json(item.metadata),
                    _iso(item.created_at),
                ),
            )
            self._event(
                connection,
                "ultra.artifact_added",
                goal_id=run["goal_id"],
                entity_type="artifact",
                entity_id=item.id,
                payload={"kind": item.kind, "uri": item.uri, "node_id": item.work_node_id},
            )
        return item

    def list_artifacts(
        self,
        run_id: str,
        *,
        work_node_id: str | None = None,
        kind: str | None = None,
        limit: int = 1_000,
    ) -> tuple[Artifact, ...]:
        sql, params = "SELECT * FROM artifacts WHERE ultra_run_id=?", [run_id]
        if work_node_id:
            sql += " AND work_node_id=?"
            params.append(work_node_id)
        if kind:
            sql += " AND kind=?"
            params.append(kind)
        sql += " ORDER BY created_at,id LIMIT ?"
        params.append(max(1, min(limit, 10_000)))
        with self._lock:
            rows = self._connection.execute(sql, tuple(params)).fetchall()
        return tuple(self._artifact_from_row(row) for row in rows)

    def add_prompt_trace(
        self,
        trace: PromptTraceV1 | None = None,
        *,
        max_bytes: int = DEFAULT_TRACE_MAX_BYTES,
        **kwargs: Any,
    ) -> PromptTraceV1:
        """Persist a compressed, redacted trace without hidden reasoning."""
        proposed = trace or PromptTraceV1(**kwargs)
        max_bytes = max(4_096, min(int(max_bytes), MAX_TRACE_MAX_BYTES))
        safe_system, changed_system = _redact_trace_text(proposed.system_prompt)
        safe_context_value, changed_context = _redact_trace_value(proposed.context_package)
        safe_self, changed_self = _redact_trace_text(proposed.self_prompt)
        safe_reasoning, changed_reasoning = _redact_trace_text(proposed.reasoning_summary)
        safe_metadata, changed_metadata = _redact_trace_value(proposed.metadata)
        safe_insights: list[InsightV1] = []
        changed_insights = False
        for insight in proposed.insights:
            value, changed = _redact_trace_value(insight.to_dict())
            safe_insights.append(InsightV1.from_dict(value))
            changed_insights = changed_insights or changed
        context_text = _json(safe_context_value)
        original_size = sum(
            len(value.encode("utf-8")) for value in (safe_system, context_text, safe_self)
        )
        # Context gets half the budget because it normally carries the useful
        # retrieval evidence; the two prompts split the remaining half.
        safe_system, cut_system = _truncate_utf8(safe_system, max_bytes // 4)
        context_text, cut_context = _truncate_utf8(context_text, max_bytes // 2)
        safe_self, cut_self = _truncate_utf8(safe_self, max_bytes - 3 * (max_bytes // 4))
        try:
            safe_context = json.loads(context_text)
        except json.JSONDecodeError:
            safe_context = {"truncated_serialized_context": context_text}
        blobs = tuple(zlib.compress(value.encode("utf-8"), level=9) for value in (safe_system, context_text, safe_self))
        redacted = (
            proposed.redacted
            or changed_system
            or changed_context
            or changed_self
            or changed_reasoning
            or changed_metadata
            or changed_insights
        )
        truncated = proposed.truncated or cut_system or cut_context or cut_self
        stored = replace(
            proposed,
            system_prompt=safe_system,
            context_package=safe_context,
            self_prompt=safe_self,
            reasoning_summary=safe_reasoning[:20_000],
            insights=tuple(safe_insights),
            redacted=redacted,
            truncated=truncated,
            metadata=dict(safe_metadata),
        )
        with self.transaction() as connection:
            run = connection.execute(
                "SELECT goal_id FROM ultra_runs WHERE id=?", (stored.ultra_run_id,)
            ).fetchone()
            if run is None:
                raise NotFoundError(f"ULTRA run not found: {stored.ultra_run_id}")
            connection.execute(
                "INSERT INTO prompt_traces("
                "id,ultra_run_id,work_node_id,agent_run_id,role,system_prompt_blob,context_blob,"
                "self_prompt_blob,compression,raw_size,stored_size,redacted,truncated,reasoning_summary,"
                "insights_json,omitted_json,metadata_json,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    stored.id,
                    stored.ultra_run_id,
                    stored.work_node_id,
                    stored.agent_run_id,
                    stored.role,
                    blobs[0],
                    blobs[1],
                    blobs[2],
                    "zlib",
                    original_size,
                    sum(len(blob) for blob in blobs),
                    int(stored.redacted),
                    int(stored.truncated),
                    stored.reasoning_summary,
                    _json([item.to_dict() for item in stored.insights]),
                    _json(stored.omitted_sections),
                    _json(stored.metadata),
                    _iso(stored.created_at),
                ),
            )
            connection.execute(
                "DELETE FROM prompt_traces WHERE ultra_run_id=? AND id NOT IN ("
                "SELECT id FROM prompt_traces WHERE ultra_run_id=? "
                "ORDER BY created_at DESC,id DESC LIMIT ?)",
                (
                    stored.ultra_run_id,
                    stored.ultra_run_id,
                    MAX_PROMPT_TRACES_PER_RUN,
                ),
            )
            self._event(
                connection,
                "ultra.trace_added",
                goal_id=run["goal_id"],
                entity_type="prompt_trace",
                entity_id=stored.id,
                payload={"role": stored.role, "redacted": redacted, "truncated": truncated},
            )
        return stored

    @staticmethod
    def _prompt_trace_from_row(row: sqlite3.Row) -> PromptTraceV1:
        try:
            if row["compression"] != "zlib":
                raise StateCorruptionError("unsupported prompt-trace compression")
            system_prompt = zlib.decompress(row["system_prompt_blob"]).decode("utf-8")
            context_text = zlib.decompress(row["context_blob"]).decode("utf-8")
            self_prompt = zlib.decompress(row["self_prompt_blob"]).decode("utf-8")
            try:
                context = json.loads(context_text)
            except json.JSONDecodeError:
                context = {"truncated_serialized_context": context_text}
        except (ValueError, TypeError, zlib.error, UnicodeDecodeError) as exc:
            raise StateCorruptionError("invalid compressed prompt trace") from exc
        return PromptTraceV1(
            id=row["id"],
            ultra_run_id=row["ultra_run_id"],
            work_node_id=row["work_node_id"],
            agent_run_id=row["agent_run_id"],
            role=row["role"],
            system_prompt=system_prompt,
            context_package=context,
            self_prompt=self_prompt,
            reasoning_summary=row["reasoning_summary"],
            insights=tuple(InsightV1.from_dict(item) for item in _loads(row["insights_json"], [])),
            omitted_sections=tuple(_loads(row["omitted_json"], [])),
            redacted=bool(row["redacted"]),
            truncated=bool(row["truncated"]),
            metadata=_loads(row["metadata_json"], {}),
            created_at=_dt(row["created_at"]),
        )

    def get_prompt_trace(self, trace_id: str) -> PromptTraceV1:
        with self._lock:
            row = self._connection.execute("SELECT * FROM prompt_traces WHERE id=?", (trace_id,)).fetchone()
        if row is None:
            raise NotFoundError(f"prompt trace not found: {trace_id}")
        return self._prompt_trace_from_row(row)

    def list_prompt_traces(
        self,
        run_id: str,
        *,
        work_node_id: str | None = None,
        agent_run_id: str | None = None,
        limit: int = 100,
    ) -> tuple[PromptTraceV1, ...]:
        sql, params = "SELECT * FROM prompt_traces WHERE ultra_run_id=?", [run_id]
        if work_node_id:
            sql += " AND work_node_id=?"
            params.append(work_node_id)
        if agent_run_id:
            sql += " AND agent_run_id=?"
            params.append(agent_run_id)
        sql += " ORDER BY created_at DESC,id DESC LIMIT ?"
        params.append(max(1, min(limit, 1_000)))
        with self._lock:
            rows = self._connection.execute(sql, tuple(params)).fetchall()
        return tuple(self._prompt_trace_from_row(row) for row in rows)

    def latest_prompt_trace(self, run_id: str) -> PromptTraceV1 | None:
        traces = self.list_prompt_traces(run_id, limit=1)
        return traces[0] if traces else None

    @staticmethod
    def _resource_lease_from_row(row: sqlite3.Row) -> ResourceLease:
        return ResourceLease(
            id=row["id"],
            ultra_run_id=row["ultra_run_id"],
            work_node_id=row["work_node_id"],
            agent_run_id=row["agent_run_id"],
            normalized_path=row["normalized_path"],
            pre_write_hash=row["pre_write_hash"],
            status=LeaseStatus(row["status"]),
            acquired_at=_dt(row["acquired_at"]),
            updated_at=_dt(row["updated_at"]),
            expires_at=_dt(row["expires_at"]),
            released_at=_dt(row["released_at"]),
        )

    def acquire_resource_lease(
        self,
        run_id: str,
        work_node_id: str,
        path: str,
        *,
        agent_run_id: str | None = None,
        pre_write_hash: str | None = None,
        ttl_seconds: int = 300,
    ) -> ResourceLease:
        normalized = normalize_contract_path(path)
        ttl_seconds = max(5, min(int(ttl_seconds), 3_600))
        now = utc_now()
        expires = now + timedelta(seconds=ttl_seconds)
        with self.transaction() as connection:
            node = connection.execute(
                "SELECT ultra_run_id FROM work_nodes WHERE id=?", (work_node_id,)
            ).fetchone()
            if node is None or node["ultra_run_id"] != run_id:
                raise StateStoreError("lease work node is missing or belongs to another run")
            connection.execute(
                "UPDATE resource_leases SET status=?,updated_at=?,released_at=? "
                "WHERE status=? AND expires_at<=?",
                (
                    LeaseStatus.EXPIRED.value,
                    _iso(now),
                    _iso(now),
                    LeaseStatus.ACTIVE.value,
                    _iso(now),
                ),
            )
            active = connection.execute(
                "SELECT * FROM resource_leases WHERE status=? ORDER BY acquired_at,id",
                (LeaseStatus.ACTIVE.value,),
            ).fetchall()
            for row in active:
                current = self._resource_lease_from_row(row)
                if not _paths_overlap(current.normalized_path, normalized):
                    continue
                same_owner = (
                    current.ultra_run_id == run_id
                    and current.work_node_id == work_node_id
                    and current.agent_run_id == agent_run_id
                    and current.normalized_path.casefold() == normalized.casefold()
                )
                if same_owner:
                    connection.execute(
                        "UPDATE resource_leases SET expires_at=?,updated_at=?,pre_write_hash=? WHERE id=?",
                        (_iso(expires), _iso(now), pre_write_hash, current.id),
                    )
                    return replace(
                        current,
                        expires_at=expires,
                        updated_at=now,
                        pre_write_hash=pre_write_hash,
                    )
                raise LeaseConflictError(
                    f"path scope {normalized!r} overlaps active lease {current.normalized_path!r}"
                )
            lease = ResourceLease(
                ultra_run_id=run_id,
                work_node_id=work_node_id,
                agent_run_id=agent_run_id,
                normalized_path=normalized,
                pre_write_hash=pre_write_hash,
                expires_at=expires,
                acquired_at=now,
                updated_at=now,
            )
            connection.execute(
                "INSERT INTO resource_leases("
                "id,ultra_run_id,work_node_id,agent_run_id,normalized_path,pre_write_hash,status,"
                "acquired_at,updated_at,expires_at,released_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (
                    lease.id,
                    lease.ultra_run_id,
                    lease.work_node_id,
                    lease.agent_run_id,
                    lease.normalized_path,
                    lease.pre_write_hash,
                    lease.status.value,
                    _iso(lease.acquired_at),
                    _iso(lease.updated_at),
                    _iso(lease.expires_at),
                    None,
                ),
            )
        return lease

    def renew_resource_lease(self, lease_id: str, *, ttl_seconds: int = 300) -> ResourceLease:
        ttl_seconds = max(5, min(int(ttl_seconds), 3_600))
        now = utc_now()
        expires = now + timedelta(seconds=ttl_seconds)
        with self.transaction() as connection:
            row = connection.execute("SELECT * FROM resource_leases WHERE id=?", (lease_id,)).fetchone()
            if row is None:
                raise NotFoundError(f"resource lease not found: {lease_id}")
            item = self._resource_lease_from_row(row)
            if item.status != LeaseStatus.ACTIVE or item.expires_at <= now:
                raise LeaseConflictError("resource lease is no longer active")
            connection.execute(
                "UPDATE resource_leases SET expires_at=?,updated_at=? WHERE id=?",
                (_iso(expires), _iso(now), lease_id),
            )
        return replace(item, expires_at=expires, updated_at=now)

    def release_resource_lease(
        self,
        lease_id: str,
        *,
        status: LeaseStatus | str = LeaseStatus.RELEASED,
    ) -> ResourceLease:
        target = LeaseStatus(status)
        if target not in {LeaseStatus.RELEASED, LeaseStatus.EXPIRED, LeaseStatus.UNCERTAIN}:
            raise ValueError("lease release status must be released, expired, or uncertain")
        now = utc_now()
        with self.transaction() as connection:
            row = connection.execute("SELECT * FROM resource_leases WHERE id=?", (lease_id,)).fetchone()
            if row is None:
                raise NotFoundError(f"resource lease not found: {lease_id}")
            item = self._resource_lease_from_row(row)
            if item.status != LeaseStatus.ACTIVE:
                if item.status == LeaseStatus.UNCERTAIN and target == LeaseStatus.RELEASED:
                    connection.execute(
                        "UPDATE resource_leases SET status=?,updated_at=?,released_at=? WHERE id=?",
                        (target.value, _iso(now), _iso(now), lease_id),
                    )
                    return replace(
                        item,
                        status=target,
                        updated_at=now,
                        released_at=now,
                    )
                return item
            connection.execute(
                "UPDATE resource_leases SET status=?,updated_at=?,released_at=? WHERE id=?",
                (target.value, _iso(now), _iso(now), lease_id),
            )
        return replace(item, status=target, updated_at=now, released_at=now)

    def assert_lease_hash(self, lease_id: str, current_hash: str | None) -> ResourceLease:
        """Turn a stale pre-write snapshot into an explicit integration conflict."""
        with self.transaction() as connection:
            row = connection.execute("SELECT * FROM resource_leases WHERE id=?", (lease_id,)).fetchone()
            if row is None:
                raise NotFoundError(f"resource lease not found: {lease_id}")
            item = self._resource_lease_from_row(row)
            if item.status != LeaseStatus.ACTIVE:
                raise LeaseConflictError("resource lease is not active")
            if item.pre_write_hash == current_hash:
                return item
            now = utc_now()
            connection.execute(
                "UPDATE resource_leases SET status=?,updated_at=?,released_at=? WHERE id=?",
                (LeaseStatus.UNCERTAIN.value, _iso(now), _iso(now), lease_id),
            )
            connection.execute(
                "UPDATE work_nodes SET status=?,error=?,updated_at=? WHERE id=?",
                (
                    WorkNodeStatus.UNCERTAIN.value,
                    "pre-write hash changed; integration conflict requires inspection",
                    _iso(now),
                    item.work_node_id,
                ),
            )
            run = connection.execute(
                "SELECT goal_id FROM ultra_runs WHERE id=?", (item.ultra_run_id,)
            ).fetchone()
            self._event(
                connection,
                "ultra.lease_conflict",
                goal_id=run["goal_id"] if run else None,
                entity_type="resource_lease",
                entity_id=lease_id,
                payload={"path": item.normalized_path, "expected_hash": item.pre_write_hash},
            )
        raise LeaseConflictError(
            f"stale pre-write hash for {item.normalized_path!r}; work node marked uncertain"
        )

    def list_resource_leases(
        self,
        run_id: str | None = None,
        *,
        status: LeaseStatus | str | None = None,
    ) -> tuple[ResourceLease, ...]:
        sql, params = "SELECT * FROM resource_leases", []
        filters: list[str] = []
        if run_id:
            filters.append("ultra_run_id=?")
            params.append(run_id)
        if status is not None:
            filters.append("status=?")
            params.append(LeaseStatus(status).value)
        if filters:
            sql += " WHERE " + " AND ".join(filters)
        sql += " ORDER BY acquired_at,id"
        with self._lock:
            rows = self._connection.execute(sql, tuple(params)).fetchall()
        return tuple(self._resource_lease_from_row(row) for row in rows)

    def reap_expired_leases(self) -> tuple[str, ...]:
        now = utc_now()
        with self.transaction() as connection:
            rows = connection.execute(
                "SELECT id FROM resource_leases WHERE status=? AND expires_at<=?",
                (LeaseStatus.ACTIVE.value, _iso(now)),
            ).fetchall()
            if rows:
                connection.execute(
                    "UPDATE resource_leases SET status=?,updated_at=?,released_at=? "
                    "WHERE status=? AND expires_at<=?",
                    (
                        LeaseStatus.EXPIRED.value,
                        _iso(now),
                        _iso(now),
                        LeaseStatus.ACTIVE.value,
                        _iso(now),
                    ),
                )
        return tuple(row["id"] for row in rows)

    def recover_ultra_inflight(self) -> UltraRecoveryReport:
        """Checkpoint interrupted ULTRA workers without replaying side effects."""
        now = utc_now()
        in_flight_nodes = (
            WorkNodeStatus.IN_PROGRESS.value,
            WorkNodeStatus.REVIEWING.value,
            WorkNodeStatus.TESTING.value,
            WorkNodeStatus.FIXING.value,
            WorkNodeStatus.INTEGRATING.value,
        )
        with self.transaction() as connection:
            agent_rows = connection.execute(
                "SELECT id,ultra_run_id,work_node_id FROM agent_runs WHERE status IN (?,?)",
                (AgentRunStatus.RUNNING.value, AgentRunStatus.RATE_LIMITED.value),
            ).fetchall()
            placeholders = ",".join("?" for _ in in_flight_nodes)
            node_rows = connection.execute(
                f"SELECT id,ultra_run_id FROM work_nodes WHERE status IN ({placeholders})",
                in_flight_nodes,
            ).fetchall()
            lease_rows = connection.execute(
                "SELECT id FROM resource_leases WHERE status=? AND expires_at<=?",
                (LeaseStatus.ACTIVE.value, _iso(now)),
            ).fetchall()
            run_ids = sorted(
                {row["ultra_run_id"] for row in agent_rows}
                | {row["ultra_run_id"] for row in node_rows}
            )
            if agent_rows:
                connection.execute(
                    "UPDATE agent_runs SET status=?,error=COALESCE(error,?),updated_at=?,finished_at=? "
                    "WHERE status IN (?,?)",
                    (
                        AgentRunStatus.UNCERTAIN.value,
                        "interrupted before a durable completion checkpoint",
                        _iso(now),
                        _iso(now),
                        AgentRunStatus.RUNNING.value,
                        AgentRunStatus.RATE_LIMITED.value,
                    ),
                )
            if node_rows:
                connection.execute(
                    f"UPDATE work_nodes SET status=?,error=COALESCE(error,?),updated_at=? "
                    f"WHERE status IN ({placeholders})",
                    (
                        WorkNodeStatus.UNCERTAIN.value,
                        "interrupted work requires inspection before resume",
                        _iso(now),
                        *in_flight_nodes,
                    ),
                )
            if lease_rows:
                connection.execute(
                    "UPDATE resource_leases SET status=?,updated_at=?,released_at=? "
                    "WHERE status=? AND expires_at<=?",
                    (
                        LeaseStatus.EXPIRED.value,
                        _iso(now),
                        _iso(now),
                        LeaseStatus.ACTIVE.value,
                        _iso(now),
                    ),
                )
            for run_id in run_ids:
                connection.execute(
                    "UPDATE ultra_runs SET status=?,updated_at=? WHERE id=? AND status NOT IN (?,?)",
                    (
                        UltraRunStatus.RECOVERING.value,
                        _iso(now),
                        run_id,
                        UltraRunStatus.COMPLETED.value,
                        UltraRunStatus.CANCELLED.value,
                    ),
                )
                goal = connection.execute(
                    "SELECT goal_id FROM ultra_runs WHERE id=?", (run_id,)
                ).fetchone()
                self._event(
                    connection,
                    "ultra.recovery_performed",
                    goal_id=goal["goal_id"] if goal else None,
                    entity_type="ultra_run",
                    entity_id=run_id,
                    payload={
                        "agents": [row["id"] for row in agent_rows if row["ultra_run_id"] == run_id],
                        "nodes": [row["id"] for row in node_rows if row["ultra_run_id"] == run_id],
                        "expired_leases": [row["id"] for row in lease_rows],
                    },
                )
        return UltraRecoveryReport(
            ultra_run_ids=tuple(run_ids),
            work_node_ids=tuple(row["id"] for row in node_rows),
            agent_run_ids=tuple(row["id"] for row in agent_rows),
            lease_ids=tuple(row["id"] for row in lease_rows),
        )
