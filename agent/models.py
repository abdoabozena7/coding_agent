"""Durable domain types for the coding-agent control plane.

The language model may propose work, roles, and state changes, but the harness
owns these values and validates every transition.  The types in this module are
deliberately independent from provider SDKs and from the terminal UI so they can
be safely persisted, tested, and restored after a crash.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import re
from typing import Any, Mapping
from uuid import uuid4


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


def new_id(prefix: str) -> str:
    """Create a readable, globally unique domain identifier."""
    return f"{prefix}_{uuid4().hex}"


class DomainError(ValueError):
    """Base class for invalid domain operations."""


class InvalidTransitionError(DomainError):
    """Raised when an entity attempts a forbidden state transition."""


class TaskGraphError(DomainError):
    """Raised when a plan's task dependency graph is not a valid DAG."""


class GoalStatus(str, Enum):
    NEW = "new"
    DISCOVERING = "discovering"
    AWAITING_PLAN_APPROVAL = "awaiting_plan_approval"
    RUNNING = "running"
    REVISING = "revising"
    VERIFYING = "verifying"
    REVIEWING = "reviewing"
    PAUSED = "paused"
    RECOVERING = "recovering"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class PlanStatus(str, Enum):
    DRAFT = "draft"
    PENDING_APPROVAL = "pending_approval"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"


class TaskStatus(str, Enum):
    PENDING = "pending"
    READY = "ready"
    IN_PROGRESS = "in_progress"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"
    OBSOLETE = "obsolete"
    UNCERTAIN = "uncertain"


class DelegationStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNCERTAIN = "uncertain"


class RuntimeEventType(str, Enum):
    GOAL_CREATED = "goal.created"
    GOAL_STATUS_CHANGED = "goal.status_changed"
    PLAN_CREATED = "plan.created"
    PLAN_REVISED = "plan.revised"
    PLAN_SUBMITTED = "plan.submitted"
    PLAN_APPROVED = "plan.approved"
    PLAN_REJECTED = "plan.rejected"
    TASK_STATUS_CHANGED = "task.status_changed"
    EVIDENCE_ADDED = "evidence.added"
    DELEGATION_CREATED = "delegation.created"
    DELEGATION_STATUS_CHANGED = "delegation.status_changed"
    RECOVERY_PERFORMED = "recovery.performed"


_GOAL_TRANSITIONS: Mapping[GoalStatus, frozenset[GoalStatus]] = {
    GoalStatus.NEW: frozenset(
        {
            GoalStatus.DISCOVERING,
            GoalStatus.AWAITING_PLAN_APPROVAL,
            GoalStatus.PAUSED,
            GoalStatus.CANCELLED,
        }
    ),
    GoalStatus.DISCOVERING: frozenset(
        {
            GoalStatus.AWAITING_PLAN_APPROVAL,
            GoalStatus.PAUSED,
            GoalStatus.RECOVERING,
            GoalStatus.CANCELLED,
        }
    ),
    GoalStatus.AWAITING_PLAN_APPROVAL: frozenset(
        {
            GoalStatus.RUNNING,
            GoalStatus.REVISING,
            GoalStatus.PAUSED,
            GoalStatus.CANCELLED,
        }
    ),
    GoalStatus.RUNNING: frozenset(
        {
            GoalStatus.REVISING,
            GoalStatus.VERIFYING,
            GoalStatus.PAUSED,
            GoalStatus.RECOVERING,
            GoalStatus.BLOCKED,
            GoalStatus.CANCELLED,
        }
    ),
    GoalStatus.REVISING: frozenset(
        {
            GoalStatus.AWAITING_PLAN_APPROVAL,
            GoalStatus.RUNNING,
            GoalStatus.PAUSED,
            GoalStatus.RECOVERING,
            GoalStatus.CANCELLED,
        }
    ),
    GoalStatus.VERIFYING: frozenset(
        {
            GoalStatus.REVIEWING,
            GoalStatus.RUNNING,
            GoalStatus.REVISING,
            GoalStatus.PAUSED,
            GoalStatus.RECOVERING,
            GoalStatus.BLOCKED,
            GoalStatus.CANCELLED,
        }
    ),
    GoalStatus.REVIEWING: frozenset(
        {
            GoalStatus.COMPLETED,
            GoalStatus.RUNNING,
            GoalStatus.REVISING,
            GoalStatus.PAUSED,
            GoalStatus.RECOVERING,
            GoalStatus.BLOCKED,
            GoalStatus.CANCELLED,
        }
    ),
    GoalStatus.PAUSED: frozenset(
        {
            GoalStatus.DISCOVERING,
            GoalStatus.AWAITING_PLAN_APPROVAL,
            GoalStatus.RUNNING,
            GoalStatus.REVISING,
            GoalStatus.VERIFYING,
            GoalStatus.REVIEWING,
            GoalStatus.RECOVERING,
            GoalStatus.BLOCKED,
            GoalStatus.CANCELLED,
        }
    ),
    GoalStatus.RECOVERING: frozenset(
        {
            GoalStatus.RUNNING,
            GoalStatus.REVISING,
            GoalStatus.VERIFYING,
            GoalStatus.REVIEWING,
            GoalStatus.PAUSED,
            GoalStatus.BLOCKED,
            GoalStatus.CANCELLED,
        }
    ),
    GoalStatus.BLOCKED: frozenset(
        {
            GoalStatus.RUNNING,
            GoalStatus.REVISING,
            GoalStatus.PAUSED,
            GoalStatus.RECOVERING,
            GoalStatus.CANCELLED,
        }
    ),
    GoalStatus.COMPLETED: frozenset(),
    GoalStatus.CANCELLED: frozenset(),
}

_PLAN_TRANSITIONS: Mapping[PlanStatus, frozenset[PlanStatus]] = {
    PlanStatus.DRAFT: frozenset(
        {PlanStatus.PENDING_APPROVAL, PlanStatus.REJECTED, PlanStatus.SUPERSEDED}
    ),
    PlanStatus.PENDING_APPROVAL: frozenset(
        {PlanStatus.ACCEPTED, PlanStatus.REJECTED, PlanStatus.SUPERSEDED}
    ),
    PlanStatus.ACCEPTED: frozenset({PlanStatus.SUPERSEDED}),
    PlanStatus.REJECTED: frozenset({PlanStatus.SUPERSEDED}),
    PlanStatus.SUPERSEDED: frozenset(),
}

_TASK_TRANSITIONS: Mapping[TaskStatus, frozenset[TaskStatus]] = {
    TaskStatus.PENDING: frozenset(
        {TaskStatus.READY, TaskStatus.IN_PROGRESS, TaskStatus.COMPLETED, TaskStatus.BLOCKED, TaskStatus.CANCELLED, TaskStatus.OBSOLETE}
    ),
    TaskStatus.READY: frozenset(
        {TaskStatus.PENDING, TaskStatus.IN_PROGRESS, TaskStatus.BLOCKED, TaskStatus.CANCELLED, TaskStatus.OBSOLETE}
    ),
    TaskStatus.IN_PROGRESS: frozenset(
        {
            TaskStatus.PENDING,
            TaskStatus.READY,
            TaskStatus.VERIFYING,
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.BLOCKED,
            TaskStatus.CANCELLED,
            TaskStatus.OBSOLETE,
            TaskStatus.UNCERTAIN,
        }
    ),
    TaskStatus.VERIFYING: frozenset(
        {
            TaskStatus.PENDING,
            TaskStatus.READY,
            TaskStatus.IN_PROGRESS,
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.BLOCKED,
            TaskStatus.CANCELLED,
            TaskStatus.OBSOLETE,
            TaskStatus.UNCERTAIN,
        }
    ),
    TaskStatus.COMPLETED: frozenset({TaskStatus.PENDING, TaskStatus.READY, TaskStatus.OBSOLETE}),
    TaskStatus.FAILED: frozenset(
        {TaskStatus.PENDING, TaskStatus.READY, TaskStatus.IN_PROGRESS, TaskStatus.BLOCKED, TaskStatus.CANCELLED, TaskStatus.OBSOLETE}
    ),
    TaskStatus.BLOCKED: frozenset(
        {
            TaskStatus.PENDING,
            TaskStatus.READY,
            TaskStatus.IN_PROGRESS,
            TaskStatus.CANCELLED,
            TaskStatus.OBSOLETE,
        }
    ),
    TaskStatus.CANCELLED: frozenset({TaskStatus.PENDING, TaskStatus.READY, TaskStatus.OBSOLETE}),
    TaskStatus.OBSOLETE: frozenset({TaskStatus.PENDING, TaskStatus.READY}),
    TaskStatus.UNCERTAIN: frozenset(
        {
            TaskStatus.PENDING,
            TaskStatus.READY,
            TaskStatus.IN_PROGRESS,
            TaskStatus.VERIFYING,
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.BLOCKED,
            TaskStatus.CANCELLED,
            TaskStatus.OBSOLETE,
        }
    ),
}

_DELEGATION_TRANSITIONS: Mapping[DelegationStatus, frozenset[DelegationStatus]] = {
    DelegationStatus.PENDING: frozenset(
        {DelegationStatus.IN_PROGRESS, DelegationStatus.CANCELLED}
    ),
    DelegationStatus.IN_PROGRESS: frozenset(
        {
            DelegationStatus.COMPLETED,
            DelegationStatus.FAILED,
            DelegationStatus.CANCELLED,
            DelegationStatus.UNCERTAIN,
        }
    ),
    DelegationStatus.COMPLETED: frozenset(),
    DelegationStatus.FAILED: frozenset(
        {DelegationStatus.PENDING, DelegationStatus.IN_PROGRESS, DelegationStatus.CANCELLED}
    ),
    DelegationStatus.CANCELLED: frozenset(),
    DelegationStatus.UNCERTAIN: frozenset(
        {
            DelegationStatus.PENDING,
            DelegationStatus.IN_PROGRESS,
            DelegationStatus.COMPLETED,
            DelegationStatus.FAILED,
            DelegationStatus.CANCELLED,
        }
    ),
}


def _ensure_transition(current: Enum, target: Enum, graph: Mapping[Enum, frozenset[Enum]], entity: str) -> None:
    if current == target:
        return
    if target not in graph[current]:
        raise InvalidTransitionError(
            f"invalid {entity} transition: {current.value!r} -> {target.value!r}"
        )


def ensure_goal_transition(current: GoalStatus, target: GoalStatus) -> None:
    _ensure_transition(current, target, _GOAL_TRANSITIONS, "goal")


def ensure_plan_transition(current: PlanStatus, target: PlanStatus) -> None:
    _ensure_transition(current, target, _PLAN_TRANSITIONS, "plan")


def ensure_task_transition(current: TaskStatus, target: TaskStatus) -> None:
    _ensure_transition(current, target, _TASK_TRANSITIONS, "task")


def ensure_delegation_transition(current: DelegationStatus, target: DelegationStatus) -> None:
    _ensure_transition(current, target, _DELEGATION_TRANSITIONS, "delegation")


@dataclass(frozen=True, slots=True)
class RoleProfile:
    """A task-specific role synthesized at runtime, never selected from a fixed list."""

    name: str = "focused worker"
    mission: str = "Complete the assigned task and provide verifiable evidence."
    expertise: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    deliverables: tuple[str, ...] = ()
    tool_policy: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.mission.strip():
            raise DomainError("a role requires a non-empty name and mission")
        if len(self.name) > 200 or len(self.mission) > 4_000:
            raise DomainError("role name/mission exceeds the durable state limit")
        object.__setattr__(self, "expertise", tuple(self.expertise))
        object.__setattr__(self, "constraints", tuple(self.constraints))
        object.__setattr__(self, "deliverables", tuple(self.deliverables))
        object.__setattr__(self, "tool_policy", dict(self.tool_policy))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "mission": self.mission,
            "expertise": list(self.expertise),
            "constraints": list(self.constraints),
            "deliverables": list(self.deliverables),
            "tool_policy": dict(self.tool_policy),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | None) -> "RoleProfile":
        value = value or {}
        return cls(
            name=str(value.get("name", "focused worker")),
            mission=str(
                value.get(
                    "mission", "Complete the assigned task and provide verifiable evidence."
                )
            ),
            expertise=tuple(value.get("expertise") or ()),
            constraints=tuple(value.get("constraints") or ()),
            deliverables=tuple(value.get("deliverables") or ()),
            tool_policy=dict(value.get("tool_policy") or {}),
            metadata=dict(value.get("metadata") or {}),
        )


@dataclass(frozen=True, slots=True)
class Goal:
    id: str
    objective: str
    success_criteria: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    status: GoalStatus = GoalStatus.NEW
    active_plan_revision: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if not self.id or not self.objective.strip():
            raise DomainError("a goal requires a non-empty id and objective")
        if len(self.objective) > 20_000:
            raise DomainError("goal objective exceeds 20,000 characters")
        object.__setattr__(self, "status", GoalStatus(self.status))
        object.__setattr__(self, "success_criteria", tuple(self.success_criteria))
        object.__setattr__(self, "constraints", tuple(self.constraints))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class Task:
    id: str
    title: str
    description: str = ""
    goal_id: str = ""
    plan_revision: int = 0
    parent_id: str | None = None
    status: TaskStatus = TaskStatus.PENDING
    depends_on: tuple[str, ...] = ()
    acceptance_criteria: tuple[str, ...] = ()
    verification: tuple[str, ...] = ()
    role: RoleProfile = field(default_factory=RoleProfile)
    mode: str = "auto"
    risk: str = "medium"
    priority: int = 0
    attempts: int = 0
    origin: str = "agent"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if not self.id or not self.title.strip():
            raise DomainError("a task requires a non-empty id and title")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,23}", self.id):
            raise DomainError(f"invalid task id: {self.id!r}")
        if len(self.title) > 180 or len(self.description) > 4_000:
            raise DomainError(f"task {self.id!r} title/description exceeds limits")
        object.__setattr__(self, "status", TaskStatus(self.status))
        object.__setattr__(self, "depends_on", tuple(self.depends_on))
        object.__setattr__(self, "acceptance_criteria", tuple(self.acceptance_criteria))
        object.__setattr__(self, "verification", tuple(self.verification))
        if not self.acceptance_criteria:
            raise DomainError(f"task {self.id!r} requires observable acceptance criteria")
        if not self.verification:
            raise DomainError(f"task {self.id!r} requires a verification method")
        if len(self.acceptance_criteria) > 20 or any(
            not item.strip() or len(item) > 1_000 for item in self.acceptance_criteria
        ):
            raise DomainError(f"task {self.id!r} has invalid acceptance criteria")
        if len(self.verification) > 20 or any(
            not item.strip() or len(item) > 1_000 for item in self.verification
        ):
            raise DomainError(f"task {self.id!r} has invalid verification steps")
        if len(self.depends_on) > 80:
            raise DomainError(f"task {self.id!r} has too many dependencies")
        if self.risk not in {"low", "medium", "high", "critical"}:
            raise DomainError(f"task {self.id!r} has invalid risk {self.risk!r}")
        if self.attempts < 0:
            raise DomainError(f"task {self.id!r} attempts cannot be negative")
        if isinstance(self.role, Mapping):
            object.__setattr__(self, "role", RoleProfile.from_dict(self.role))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class Plan:
    id: str
    goal_id: str
    revision: int
    summary: str
    status: PlanStatus = PlanStatus.DRAFT
    tasks: tuple[Task, ...] = ()
    applicability_evidence: tuple[Mapping[str, Any], ...] = ()
    execution_strategy: str = ""
    expected_changes: tuple[Mapping[str, Any], ...] = ()
    proposed_by: str = "agent"
    fingerprint: str = ""
    accepted_by: str | None = None
    accepted_at: datetime | None = None
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if not self.id or not self.goal_id or self.revision < 1:
            raise DomainError("a plan requires ids and a positive revision")
        if not self.summary.strip() or len(self.summary) > 20_000:
            raise DomainError("a plan requires a bounded non-empty summary")
        object.__setattr__(self, "status", PlanStatus(self.status))
        object.__setattr__(self, "tasks", tuple(self.tasks))
        object.__setattr__(
            self,
            "applicability_evidence",
            tuple(dict(item) for item in self.applicability_evidence),
        )
        object.__setattr__(self, "expected_changes", tuple(dict(item) for item in self.expected_changes))
        if not self.tasks or len(self.tasks) > 80:
            raise DomainError("a plan requires between 1 and 80 tasks")
        if len(self.applicability_evidence) > 40 or len(self.expected_changes) > 80:
            raise DomainError("plan applicability data exceeds durable limits")
        if len(self.execution_strategy) > 8_000:
            raise DomainError("plan execution strategy exceeds durable limits")
        validate_task_dag(self.tasks)


@dataclass(frozen=True, slots=True)
class Evidence:
    goal_id: str
    summary: str
    id: str = field(default_factory=lambda: new_id("evidence"))
    plan_revision: int | None = None
    task_id: str | None = None
    kind: str = "note"
    artifact_uri: str | None = None
    data: Mapping[str, Any] = field(default_factory=dict)
    created_by: str = "agent"
    verified: bool = False
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if not self.id or not self.goal_id or not self.summary.strip():
            raise DomainError("evidence requires an id, goal id, and summary")
        object.__setattr__(self, "data", dict(self.data))


@dataclass(frozen=True, slots=True)
class Delegation:
    goal_id: str
    task_id: str
    brief: str
    id: str = field(default_factory=lambda: new_id("delegation"))
    plan_revision: int = 0
    parent_id: str | None = None
    worker_id: str | None = None
    role: RoleProfile = field(default_factory=RoleProfile)
    status: DelegationStatus = DelegationStatus.PENDING
    attempt: int = 0
    result_summary: str | None = None
    error: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if not self.id or not self.goal_id or not self.task_id or not self.brief.strip():
            raise DomainError("a delegation requires ids and a non-empty brief")
        object.__setattr__(self, "status", DelegationStatus(self.status))
        if isinstance(self.role, Mapping):
            object.__setattr__(self, "role", RoleProfile.from_dict(self.role))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class RuntimeEvent:
    event_type: RuntimeEventType | str
    id: str = field(default_factory=lambda: new_id("event"))
    sequence: int | None = None
    goal_id: str | None = None
    entity_type: str | None = None
    entity_id: str | None = None
    payload: Mapping[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        event_type = self.event_type.value if isinstance(self.event_type, RuntimeEventType) else str(self.event_type)
        if not event_type:
            raise DomainError("an event requires a non-empty type")
        object.__setattr__(self, "event_type", event_type)
        object.__setattr__(self, "payload", dict(self.payload))


@dataclass(frozen=True, slots=True)
class PlanApproval:
    id: str
    goal_id: str
    plan_id: str
    revision: int
    fingerprint: str
    approved_by: str
    approved_at: datetime


@dataclass(frozen=True, slots=True)
class RecoveryReport:
    task_ids: tuple[str, ...] = ()
    delegation_ids: tuple[str, ...] = ()
    action_ids: tuple[str, ...] = ()
    goal_ids: tuple[str, ...] = ()

    @property
    def changed(self) -> bool:
        return bool(self.task_ids or self.delegation_ids or self.action_ids or self.goal_ids)


TERMINAL_GOAL_STATUSES = frozenset({GoalStatus.COMPLETED, GoalStatus.CANCELLED})
IN_FLIGHT_TASK_STATUSES = frozenset({TaskStatus.IN_PROGRESS, TaskStatus.VERIFYING})
IN_FLIGHT_DELEGATION_STATUSES = frozenset({DelegationStatus.IN_PROGRESS})


def validate_task_dag(tasks: tuple[Task, ...] | list[Task]) -> tuple[str, ...]:
    """Validate dependencies and return a deterministic topological order.

    Dependency identifiers are scoped to one plan revision.  Parent links form
    a separate hierarchy: they are checked for missing nodes and cycles, but do
    not imply execution ordering.  Keeping the two graphs separate avoids
    inventing dependencies merely because a checklist item groups subtasks.
    """

    task_by_id: dict[str, Task] = {}
    for task in tasks:
        if task.id in task_by_id:
            raise TaskGraphError(f"duplicate task id in plan: {task.id!r}")
        task_by_id[task.id] = task

    task_ids = set(task_by_id)
    for task in task_by_id.values():
        if len(set(task.depends_on)) != len(task.depends_on):
            raise TaskGraphError(f"task {task.id!r} has duplicate dependencies")
        missing = set(task.depends_on) - task_ids
        if missing:
            raise TaskGraphError(
                f"task {task.id!r} depends on unknown tasks: {sorted(missing)!r}"
            )
        if task.id in task.depends_on:
            raise TaskGraphError(f"task {task.id!r} cannot depend on itself")
        if task.parent_id is not None and task.parent_id not in task_ids:
            raise TaskGraphError(
                f"task {task.id!r} has unknown parent {task.parent_id!r}"
            )
        if task.parent_id == task.id:
            raise TaskGraphError(f"task {task.id!r} cannot parent itself")

    # Kahn's algorithm gives both cycle detection and a useful stable order.
    indegree = {task_id: 0 for task_id in task_ids}
    dependants: dict[str, list[str]] = {task_id: [] for task_id in task_ids}
    for task in task_by_id.values():
        indegree[task.id] = len(task.depends_on)
        for dependency in task.depends_on:
            dependants[dependency].append(task.id)

    ready = sorted(task_id for task_id, degree in indegree.items() if degree == 0)
    ordered: list[str] = []
    while ready:
        task_id = ready.pop(0)
        ordered.append(task_id)
        for dependant in sorted(dependants[task_id]):
            indegree[dependant] -= 1
            if indegree[dependant] == 0:
                ready.append(dependant)
                ready.sort()
    if len(ordered) != len(task_ids):
        cyclic = sorted(task_id for task_id, degree in indegree.items() if degree)
        raise TaskGraphError(f"task dependency cycle detected: {cyclic!r}")

    # Parent pointers are a forest and need their own cycle check.
    for starting_id in sorted(task_ids):
        seen: set[str] = set()
        current_id: str | None = starting_id
        while current_id is not None:
            if current_id in seen:
                raise TaskGraphError(
                    f"task parent cycle detected from {starting_id!r}"
                )
            seen.add(current_id)
            current_id = task_by_id[current_id].parent_id

    return tuple(ordered)
