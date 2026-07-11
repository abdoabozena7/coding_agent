"""Transactional SQLite state store for persistent, crash-recoverable goals."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime
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


SCHEMA_VERSION = 2


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
