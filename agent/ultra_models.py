"""Provider-independent domain types for ULTRA orchestration.

The ULTRA engine deliberately persists contracts and results as versioned data
objects.  Provider responses are converted into these types before they reach
the scheduler, which keeps scope checks and crash recovery under harness
control rather than model control.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import hashlib
import json
from pathlib import PurePosixPath
from typing import Any, Mapping

from .model_catalog import ExecutionClass
from .models import DomainError, new_id, utc_now
from .sandbox import AccessLevel


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    )


def _bounded_text(value: Any, name: str, *, maximum: int = 20_000) -> str:
    result = str(value).strip()
    if not result:
        raise DomainError(f"{name} must not be empty")
    if "\x00" in result or len(result) > maximum:
        raise DomainError(f"{name} exceeds its durable-state limit")
    return result


def _string_tuple(values: Any) -> tuple[str, ...]:
    if values is None:
        return ()
    return tuple(str(value).strip() for value in values if str(value).strip())


class UltraPhase(str, Enum):
    GOAL_INTERVIEW = "goal_interview"
    GOAL_SPEC = "goal_spec"
    ARCHITECTURE = "architecture"
    MASTER_PLAN = "master_plan"
    AWAITING_APPROVAL = "awaiting_approval"
    MODULE_WAVES = "module_waves"
    INTEGRATION = "integration"
    GLOBAL_REVIEW = "global_review"
    EVIDENCE_GATE = "evidence_gate"
    COMPLETED = "completed"


class UltraRunStatus(str, Enum):
    DRAFT = "draft"
    AWAITING_APPROVAL = "awaiting_approval"
    RUNNING = "running"
    PAUSED = "paused"
    REVISION_REQUIRED = "revision_required"
    RECOVERING = "recovering"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    UNCERTAIN = "uncertain"


class WorkNodeKind(str, Enum):
    MILESTONE = "milestone"
    MODULE = "module"
    SUBMODULE = "submodule"
    TASK = "task"


class WorkNodeStatus(str, Enum):
    PENDING = "pending"
    READY = "ready"
    IN_PROGRESS = "in_progress"
    REVIEWING = "reviewing"
    TESTING = "testing"
    FIXING = "fixing"
    INTEGRATING = "integrating"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    CONFLICT = "conflict"
    CANCELLED = "cancelled"
    UNCERTAIN = "uncertain"
    REVISION_REQUIRED = "revision_required"


class AgentRunStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RATE_LIMITED = "rate_limited"
    UNCERTAIN = "uncertain"


class BrainSection(str, Enum):
    NORTH_STAR = "north_star"
    ARCHITECTURE = "architecture"
    DECISION = "decision"
    CONSTRAINT = "constraint"
    TASK_GRAPH = "task_graph"
    ARTIFACT_INDEX = "artifact_index"
    KNOWLEDGE = "knowledge"
    LESSON = "lesson"
    ROLE_MEMORY = "role_memory"


class LeaseStatus(str, Enum):
    ACTIVE = "active"
    RELEASED = "released"
    EXPIRED = "expired"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True, slots=True)
class GoalSpecV1:
    objective: str
    scope: tuple[str, ...] = ()
    success_criteria: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    non_goals: tuple[str, ...] = ()
    answered_questions: Mapping[str, Any] = field(default_factory=dict)
    version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "objective", _bounded_text(self.objective, "goal objective"))
        for name in ("scope", "success_criteria", "constraints", "non_goals"):
            object.__setattr__(self, name, _string_tuple(getattr(self, name)))
        object.__setattr__(self, "answered_questions", dict(self.answered_questions))
        if self.version != 1:
            raise DomainError("GoalSpecV1 only accepts version 1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "objective": self.objective,
            "scope": list(self.scope),
            "success_criteria": list(self.success_criteria),
            "constraints": list(self.constraints),
            "non_goals": list(self.non_goals),
            "answered_questions": dict(self.answered_questions),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GoalSpecV1":
        return cls(
            objective=value.get("objective", ""),
            scope=value.get("scope", ()),
            success_criteria=value.get("success_criteria", ()),
            constraints=value.get("constraints", ()),
            non_goals=value.get("non_goals", ()),
            answered_questions=value.get("answered_questions", {}),
            version=int(value.get("version", 1)),
        )

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(_canonical_json(self.to_dict()).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ArchitectureSpecV1:
    summary: str
    components: tuple[Mapping[str, Any], ...] = ()
    interfaces: Mapping[str, Any] = field(default_factory=dict)
    decisions: tuple[Mapping[str, Any], ...] = ()
    constraints: tuple[str, ...] = ()
    version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "summary", _bounded_text(self.summary, "architecture summary"))
        object.__setattr__(self, "components", tuple(dict(item) for item in self.components))
        object.__setattr__(self, "interfaces", dict(self.interfaces))
        object.__setattr__(self, "decisions", tuple(dict(item) for item in self.decisions))
        object.__setattr__(self, "constraints", _string_tuple(self.constraints))
        if self.version != 1:
            raise DomainError("ArchitectureSpecV1 only accepts version 1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "summary": self.summary,
            "components": [dict(item) for item in self.components],
            "interfaces": dict(self.interfaces),
            "decisions": [dict(item) for item in self.decisions],
            "constraints": list(self.constraints),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ArchitectureSpecV1":
        return cls(
            summary=value.get("summary", ""),
            components=value.get("components", ()),
            interfaces=value.get("interfaces", {}),
            decisions=value.get("decisions", ()),
            constraints=value.get("constraints", ()),
            version=int(value.get("version", 1)),
        )


def normalize_contract_path(value: str) -> str:
    """Return a portable workspace-relative path suitable for scope checks."""
    text = str(value).strip().replace("\\", "/")
    if not text or text == ".":
        return "."
    if text.startswith("/") or re_drive_prefix(text):
        raise DomainError(f"contract path must be workspace-relative: {value!r}")
    parts = PurePosixPath(text).parts
    if any(part in {"", ".."} for part in parts):
        raise DomainError(f"contract path escapes the workspace: {value!r}")
    normalized = "/".join(part for part in parts if part != ".").rstrip("/")
    return normalized or "."


def re_drive_prefix(value: str) -> bool:
    return len(value) >= 2 and value[0].isalpha() and value[1] == ":"


@dataclass(frozen=True, slots=True)
class TaskContractV1:
    objective: str
    success_criteria: tuple[str, ...] = ()
    write_paths: tuple[str, ...] = ()
    read_paths: tuple[str, ...] = (".",)
    forbidden_changes: tuple[str, ...] = ()
    interfaces: Mapping[str, Any] = field(default_factory=dict)
    external_dependencies: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "objective", _bounded_text(self.objective, "task objective", maximum=8_000))
        object.__setattr__(self, "success_criteria", _string_tuple(self.success_criteria))
        object.__setattr__(
            self, "write_paths", tuple(dict.fromkeys(normalize_contract_path(p) for p in self.write_paths))
        )
        object.__setattr__(
            self, "read_paths", tuple(dict.fromkeys(normalize_contract_path(p) for p in self.read_paths))
        )
        object.__setattr__(self, "forbidden_changes", _string_tuple(self.forbidden_changes))
        object.__setattr__(self, "interfaces", dict(self.interfaces))
        object.__setattr__(self, "external_dependencies", _string_tuple(self.external_dependencies))
        object.__setattr__(self, "metadata", dict(self.metadata))
        if self.version != 1:
            raise DomainError("TaskContractV1 only accepts version 1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "objective": self.objective,
            "success_criteria": list(self.success_criteria),
            "write_paths": list(self.write_paths),
            "read_paths": list(self.read_paths),
            "forbidden_changes": list(self.forbidden_changes),
            "interfaces": dict(self.interfaces),
            "external_dependencies": list(self.external_dependencies),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TaskContractV1":
        return cls(
            objective=value.get("objective", ""),
            success_criteria=value.get("success_criteria", ()),
            write_paths=value.get("write_paths", ()),
            read_paths=value.get("read_paths", (".",)),
            forbidden_changes=value.get("forbidden_changes", ()),
            interfaces=value.get("interfaces", {}),
            external_dependencies=value.get("external_dependencies", ()),
            metadata=value.get("metadata", {}),
            version=int(value.get("version", 1)),
        )


class ContractScopeError(DomainError):
    """A dynamic child exceeds its approved parent and needs plan revision."""

    requires_plan_revision = True

    def __init__(self, reasons: tuple[str, ...]) -> None:
        self.reasons = reasons
        super().__init__("contract exceeds approved scope: " + "; ".join(reasons))


def _within(child: str, parent: str) -> bool:
    child_key = normalize_contract_path(child).casefold()
    parent_key = normalize_contract_path(parent).casefold()
    return parent_key == "." or child_key == parent_key or child_key.startswith(parent_key + "/")


def contract_scope_violations(parent: TaskContractV1, child: TaskContractV1) -> tuple[str, ...]:
    """Explain any child changes that require a new master-plan approval."""
    reasons: list[str] = []
    for path in child.write_paths:
        if not any(_within(path, allowed) for allowed in parent.write_paths):
            reasons.append(f"write path {path!r} is outside the parent contract")
    parent_forbidden = {item.casefold() for item in parent.forbidden_changes}
    child_forbidden = {item.casefold() for item in child.forbidden_changes}
    if missing := parent_forbidden - child_forbidden:
        reasons.append(f"child drops forbidden changes: {sorted(missing)!r}")
    for name, definition in child.interfaces.items():
        if name not in parent.interfaces or _canonical_json(parent.interfaces[name]) != _canonical_json(definition):
            reasons.append(f"interface {name!r} is new or changed")
    parent_dependencies = {item.casefold() for item in parent.external_dependencies}
    added_dependencies = {
        item for item in child.external_dependencies if item.casefold() not in parent_dependencies
    }
    if added_dependencies:
        reasons.append(f"external dependencies are new: {sorted(added_dependencies)!r}")
    return tuple(reasons)


def assert_child_contract(parent: TaskContractV1, child: TaskContractV1) -> None:
    if reasons := contract_scope_violations(parent, child):
        raise ContractScopeError(reasons)


@dataclass(frozen=True, slots=True)
class InsightV1:
    summary: str
    category: str = "general"
    details: str = ""
    severity: str = "info"
    evidence: tuple[str, ...] = ()
    id: str = field(default_factory=lambda: new_id("insight"))
    created_at: datetime = field(default_factory=utc_now)
    version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "summary", _bounded_text(self.summary, "insight summary", maximum=4_000))
        object.__setattr__(self, "category", _bounded_text(self.category, "insight category", maximum=100))
        object.__setattr__(self, "details", str(self.details)[:20_000])
        object.__setattr__(self, "evidence", _string_tuple(self.evidence))
        if self.severity not in {"info", "warning", "error", "critical"}:
            raise DomainError("invalid insight severity")

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1, "id": self.id, "summary": self.summary,
            "category": self.category, "details": self.details, "severity": self.severity,
            "evidence": list(self.evidence), "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "InsightV1":
        created = value.get("created_at")
        return cls(
            id=str(value.get("id") or new_id("insight")), summary=value.get("summary", ""),
            category=value.get("category", "general"), details=value.get("details", ""),
            severity=value.get("severity", "info"), evidence=value.get("evidence", ()),
            created_at=datetime.fromisoformat(created) if created else utc_now(),
            version=int(value.get("version", 1)),
        )


@dataclass(frozen=True, slots=True)
class ResultPackageV1:
    summary: str
    changed_files: tuple[str, ...] = ()
    tests: tuple[Mapping[str, Any], ...] = ()
    artifacts: tuple[str, ...] = ()
    insights: tuple[InsightV1, ...] = ()
    issues: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "summary", _bounded_text(self.summary, "result summary", maximum=20_000))
        object.__setattr__(self, "changed_files", _string_tuple(self.changed_files))
        object.__setattr__(self, "tests", tuple(dict(item) for item in self.tests))
        object.__setattr__(self, "artifacts", _string_tuple(self.artifacts))
        object.__setattr__(self, "insights", tuple(
            item if isinstance(item, InsightV1) else InsightV1.from_dict(item) for item in self.insights
        ))
        object.__setattr__(self, "issues", _string_tuple(self.issues))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1, "summary": self.summary, "changed_files": list(self.changed_files),
            "tests": [dict(item) for item in self.tests], "artifacts": list(self.artifacts),
            "insights": [item.to_dict() for item in self.insights], "issues": list(self.issues),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ResultPackageV1":
        return cls(
            summary=value.get("summary", ""), changed_files=value.get("changed_files", ()),
            tests=value.get("tests", ()), artifacts=value.get("artifacts", ()),
            insights=tuple(InsightV1.from_dict(item) for item in value.get("insights", ())),
            issues=value.get("issues", ()), metadata=value.get("metadata", {}),
            version=int(value.get("version", 1)),
        )


@dataclass(frozen=True, slots=True)
class PromptTraceV1:
    ultra_run_id: str
    role: str
    system_prompt: str
    context_package: Mapping[str, Any]
    self_prompt: str
    reasoning_summary: str = ""
    insights: tuple[InsightV1, ...] = ()
    omitted_sections: tuple[str, ...] = ()
    work_node_id: str | None = None
    agent_run_id: str | None = None
    redacted: bool = False
    truncated: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: new_id("trace"))
    created_at: datetime = field(default_factory=utc_now)
    version: int = 1

    def __post_init__(self) -> None:
        if not self.ultra_run_id or not self.role.strip():
            raise DomainError("a prompt trace requires run id and role")
        object.__setattr__(self, "system_prompt", str(self.system_prompt))
        object.__setattr__(self, "context_package", dict(self.context_package))
        object.__setattr__(self, "self_prompt", str(self.self_prompt))
        object.__setattr__(self, "reasoning_summary", str(self.reasoning_summary))
        object.__setattr__(self, "insights", tuple(
            item if isinstance(item, InsightV1) else InsightV1.from_dict(item) for item in self.insights
        ))
        object.__setattr__(self, "omitted_sections", _string_tuple(self.omitted_sections))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class UltraRun:
    goal_id: str
    provider: str
    model: str
    execution_class: ExecutionClass = ExecutionClass.LOCAL
    access_level: AccessLevel = AccessLevel.NORMAL
    concurrency: int = 1
    phase: UltraPhase = UltraPhase.GOAL_INTERVIEW
    status: UltraRunStatus = UltraRunStatus.DRAFT
    goal_spec: GoalSpecV1 | None = None
    architecture_spec: ArchitectureSpecV1 | None = None
    plan_revision: int | None = None
    master_plan_fingerprint: str = ""
    master_approved: bool = False
    config: Mapping[str, Any] = field(default_factory=dict)
    error: str | None = None
    id: str = field(default_factory=lambda: new_id("ultra"))
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if not self.goal_id or not str(self.provider).strip() or not str(self.model).strip():
            raise DomainError("an ULTRA run requires goal, provider, and model")
        object.__setattr__(self, "execution_class", ExecutionClass(self.execution_class))
        object.__setattr__(self, "access_level", AccessLevel(self.access_level))
        object.__setattr__(self, "phase", UltraPhase(self.phase))
        object.__setattr__(self, "status", UltraRunStatus(self.status))
        if not 1 <= int(self.concurrency) <= 8:
            raise DomainError("ULTRA concurrency must be between 1 and 8")
        if self.execution_class == ExecutionClass.LOCAL and self.concurrency != 1:
            raise DomainError("local ULTRA execution must be sequential")
        object.__setattr__(self, "config", dict(self.config))


@dataclass(frozen=True, slots=True)
class WorkNode:
    ultra_run_id: str
    title: str
    objective: str
    contract: TaskContractV1
    kind: WorkNodeKind = WorkNodeKind.TASK
    status: WorkNodeStatus = WorkNodeStatus.PENDING
    parent_id: str | None = None
    master_task_id: str | None = None
    depth: int = 0
    position: int = 0
    depends_on: tuple[str, ...] = ()
    assigned_role: str = "coder"
    attempts: int = 0
    max_attempts: int = 3
    result: ResultPackageV1 | None = None
    error: str | None = None
    checkpoint: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: new_id("node"))
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if not self.ultra_run_id or not self.id:
            raise DomainError("a work node requires ids")
        object.__setattr__(self, "title", _bounded_text(self.title, "node title", maximum=300))
        object.__setattr__(self, "objective", _bounded_text(self.objective, "node objective", maximum=8_000))
        if isinstance(self.contract, Mapping):
            object.__setattr__(self, "contract", TaskContractV1.from_dict(self.contract))
        object.__setattr__(self, "kind", WorkNodeKind(self.kind))
        object.__setattr__(self, "status", WorkNodeStatus(self.status))
        object.__setattr__(self, "depends_on", _string_tuple(self.depends_on))
        object.__setattr__(self, "metadata", dict(self.metadata))
        if self.depth < 0 or self.attempts < 0 or self.max_attempts < 1:
            raise DomainError("invalid work-node counters")

    @property
    def is_master_module(self) -> bool:
        return self.kind == WorkNodeKind.MODULE and self.parent_id is None and bool(self.master_task_id)


@dataclass(frozen=True, slots=True)
class AgentRun:
    ultra_run_id: str
    role: str
    provider: str
    model: str
    phase: str
    work_node_id: str | None = None
    status: AgentRunStatus = AgentRunStatus.QUEUED
    attempt: int = 1
    usage: Mapping[str, Any] = field(default_factory=dict)
    result: ResultPackageV1 | None = None
    error: str | None = None
    prompt_trace_id: str | None = None
    side_effects: bool = False
    id: str = field(default_factory=lambda: new_id("agent_run"))
    started_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    finished_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.ultra_run_id or not self.role.strip() or not self.phase.strip():
            raise DomainError("an agent run requires run id, role, and phase")
        object.__setattr__(self, "status", AgentRunStatus(self.status))
        object.__setattr__(self, "usage", dict(self.usage))
        if isinstance(self.result, Mapping):
            object.__setattr__(self, "result", ResultPackageV1.from_dict(self.result))


@dataclass(frozen=True, slots=True)
class BrainEntry:
    ultra_run_id: str
    goal_id: str
    section: BrainSection
    title: str
    content: str
    data: Mapping[str, Any] = field(default_factory=dict)
    work_node_id: str | None = None
    agent_run_id: str | None = None
    role: str | None = None
    version: int = 1
    supersedes_id: str | None = None
    expires_at: datetime | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: new_id("brain"))
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        object.__setattr__(self, "section", BrainSection(self.section))
        object.__setattr__(self, "title", _bounded_text(self.title, "brain entry title", maximum=500))
        object.__setattr__(self, "content", _bounded_text(self.content, "brain entry content", maximum=100_000))
        object.__setattr__(self, "data", dict(self.data))
        object.__setattr__(self, "metadata", dict(self.metadata))
        if self.version < 1:
            raise DomainError("brain entry version must be positive")


@dataclass(frozen=True, slots=True)
class Artifact:
    ultra_run_id: str
    kind: str
    uri: str
    work_node_id: str | None = None
    agent_run_id: str | None = None
    path: str | None = None
    content_hash: str | None = None
    pre_write_hash: str | None = None
    evidence: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: new_id("artifact"))
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if not self.ultra_run_id or not self.kind.strip() or not self.uri.strip():
            raise DomainError("an artifact requires run id, kind, and URI")
        object.__setattr__(self, "evidence", dict(self.evidence))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class ResourceLease:
    ultra_run_id: str
    work_node_id: str
    normalized_path: str
    expires_at: datetime
    agent_run_id: str | None = None
    pre_write_hash: str | None = None
    status: LeaseStatus = LeaseStatus.ACTIVE
    id: str = field(default_factory=lambda: new_id("lease"))
    acquired_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    released_at: datetime | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "normalized_path", normalize_contract_path(self.normalized_path))
        object.__setattr__(self, "status", LeaseStatus(self.status))


@dataclass(frozen=True, slots=True)
class UltraRecoveryReport:
    ultra_run_ids: tuple[str, ...] = ()
    work_node_ids: tuple[str, ...] = ()
    agent_run_ids: tuple[str, ...] = ()
    lease_ids: tuple[str, ...] = ()

    @property
    def changed(self) -> bool:
        return bool(self.ultra_run_ids or self.work_node_ids or self.agent_run_ids or self.lease_ids)


@dataclass(frozen=True, slots=True)
class ContextPackageV1:
    ultra_run_id: str
    work_node_id: str
    role: str
    sections: Mapping[str, Any]
    omitted_sections: tuple[str, ...] = ()
    size_chars: int = 0
    version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "sections", dict(self.sections))
        object.__setattr__(self, "omitted_sections", _string_tuple(self.omitted_sections))

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1, "ultra_run_id": self.ultra_run_id,
            "work_node_id": self.work_node_id, "role": self.role,
            "sections": dict(self.sections), "omitted_sections": list(self.omitted_sections),
            "size_chars": self.size_chars,
        }
