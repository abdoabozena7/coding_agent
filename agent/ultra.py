"""Provider-neutral ULTRA orchestration foundation.

ULTRA turns a compact goal into an approved project contract, expands that
contract only inside approved boundaries, and applies the same role-separated
quality loop to every work node.  Persistence, terminal rendering, and model
selection are injected adapters so this module can be integrated without
coupling it to the existing plan/goal runtime.

The trace model intentionally stores prompts, focused context, model-authored
summaries, and structured insights only.  It never requests or persists hidden
chain-of-thought.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from collections import defaultdict
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

from .intake import answer_from_value, normalize_question
from .local_provider import repair_structured_json_object
from uuid import uuid4

from .events import EventBus
from .safety import redact_data, redact_text
from .scheduler import (
    AdaptiveConcurrency,
    BackgroundRunController,
    CancellationRequested,
    CooperativeControl,
    DeterministicWaveScheduler,
    ExecutionClass,
    RateLimitError,
    ResourceLeaseManager,
    ScheduleReport,
    ScheduleStatus,
)


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _strings(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, Iterable) or isinstance(value, Mapping):
        return ()
    return tuple(str(item).strip() for item in value if str(item).strip())


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _with_quality_milestone(
    milestones: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    values = [dict(item) for item in milestones if isinstance(item, Mapping)]
    label = "Quality Refactor and Global Gate"
    if not any(
        str(item.get("title") or item.get("name") or item.get("id") or "").strip().casefold()
        == label.casefold()
        for item in values
    ):
        values.append({"title": label, "kind": "quality_gate"})
    return tuple(values)


class UltraPhase(str, Enum):
    NEW = "new"
    GOAL_SPEC = "goal_spec"
    AWAITING_QUESTIONS = "awaiting_questions"
    ARCHITECTURE = "architecture"
    MASTER_PLAN = "master_plan"
    AWAITING_APPROVAL = "awaiting_approval"
    EXPANDING = "expanding"
    MODULE_WAVES = "module_waves"
    INTEGRATION = "integration"
    GLOBAL_REVIEW = "global_review"
    FINAL_EVIDENCE = "final_evidence"
    REVISION_REQUIRED = "revision_required"
    PAUSED = "paused"
    CANCELLED = "cancelled"
    FAILED = "failed"
    COMPLETED = "completed"


_TERMINAL_ULTRA_PHASES = {
    UltraPhase.CANCELLED,
    UltraPhase.FAILED,
    UltraPhase.COMPLETED,
}


def ensure_ultra_phase_transition(current: UltraPhase, target: UltraPhase) -> None:
    """Reject phase jumps that would bypass approval or quality gates."""

    if current is target:
        return
    common_interrupts = {
        UltraPhase.PAUSED,
        UltraPhase.CANCELLED,
        UltraPhase.FAILED,
        UltraPhase.REVISION_REQUIRED,
    }
    allowed: dict[UltraPhase, set[UltraPhase]] = {
        UltraPhase.NEW: {UltraPhase.GOAL_SPEC},
        UltraPhase.GOAL_SPEC: {UltraPhase.AWAITING_QUESTIONS, UltraPhase.ARCHITECTURE},
        UltraPhase.AWAITING_QUESTIONS: {UltraPhase.ARCHITECTURE},
        UltraPhase.ARCHITECTURE: {UltraPhase.MASTER_PLAN},
        UltraPhase.MASTER_PLAN: {UltraPhase.AWAITING_APPROVAL},
        UltraPhase.AWAITING_APPROVAL: {UltraPhase.EXPANDING},
        UltraPhase.EXPANDING: {UltraPhase.MODULE_WAVES},
        UltraPhase.MODULE_WAVES: {UltraPhase.INTEGRATION},
        UltraPhase.INTEGRATION: {UltraPhase.GLOBAL_REVIEW},
        UltraPhase.GLOBAL_REVIEW: {UltraPhase.FINAL_EVIDENCE},
        UltraPhase.FINAL_EVIDENCE: {UltraPhase.COMPLETED},
        UltraPhase.PAUSED: set(UltraPhase) - _TERMINAL_ULTRA_PHASES,
        UltraPhase.REVISION_REQUIRED: {UltraPhase.AWAITING_APPROVAL},
        UltraPhase.CANCELLED: set(),
        UltraPhase.FAILED: set(),
        UltraPhase.COMPLETED: set(),
    }
    if target in common_interrupts and current not in _TERMINAL_ULTRA_PHASES:
        return
    if target not in allowed.get(current, set()):
        raise UltraError(
            f"invalid ULTRA phase transition: {current.value} -> {target.value}"
        )


class InnerPhase(str, Enum):
    CONTEXT = "context"
    MINI_PLAN = "mini_plan"
    DECOMPOSE = "decompose"
    RESEARCH = "research"
    IMPLEMENT = "implement"
    REVIEW = "review"
    TEST = "test"
    FIX = "fix"
    REPLAN = "replan"
    INTEGRATE = "integrate"
    MEMORY_WRITEBACK = "memory_writeback"
    GLOBAL_INTEGRATION = "global_integration"
    GLOBAL_REVIEW = "global_review"
    FINAL_EVIDENCE = "final_evidence"


class AgentRole(str, Enum):
    GOAL_UNDERSTANDING = "goal_understanding"
    ARCHITECT = "architect"
    PLANNER = "planner"
    DECOMPOSER = "decomposer"
    RESEARCHER = "researcher"
    CODER = "coder"
    REVIEWER = "reviewer"
    TESTER = "tester"
    INTEGRATOR = "integrator"
    GOAL_CHECKER = "goal_checker"
    MEMORY = "memory"
    CLEAN_CODE_REVIEWER = "clean_code_reviewer"
    SECURITY_REVIEWER = "security_reviewer"
    TEST_QUALITY_REVIEWER = "test_quality_reviewer"
    QUALITY_TRIAGER = "quality_triager"


class NodeKind(str, Enum):
    MILESTONE = "milestone"
    MODULE = "module"
    SUBMODULE = "submodule"
    TASK = "task"


class NodeStatus(str, Enum):
    PENDING = "pending"
    PLANNING = "planning"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    CONFLICT = "conflict"
    CANCELLED = "cancelled"
    UNCERTAIN = "uncertain"
    REVISION_REQUIRED = "revision_required"


class BrainSection(str, Enum):
    NORTH_STAR = "north_star"
    ARCHITECTURE = "architecture"
    DECISION = "decision"
    CONSTRAINT = "constraint"
    TASK_GRAPH = "task_graph"
    ARTIFACT = "artifact"
    KNOWLEDGE = "knowledge"
    LESSON = "lesson"
    ROLE_MEMORY = "role_memory"


class UltraEventKind(str, Enum):
    PHASE = "ultra.phase"
    FOUNDATION_READY = "ultra.foundation_ready"
    APPROVED = "ultra.approved"
    NODE = "ultra.node"
    AGENT = "ultra.agent"
    INSIGHT = "ultra.insight"
    FIX = "ultra.fix"
    COMPLETED = "ultra.completed"
    CANCELLED = "ultra.cancelled"
    REVISION_REQUIRED = "ultra.revision_required"


class UltraError(RuntimeError):
    pass


class ApprovalRequiredError(UltraError):
    pass


class ApprovalMismatchError(UltraError):
    pass


class AgentProtocolError(UltraError):
    pass


class ScopeRevisionRequired(UltraError):
    pass


class NodePipelineFailed(UltraError):
    def __init__(self, result: "ResultPackageV1") -> None:
        super().__init__(result.summary or f"node {result.node_id} failed its quality gate")
        self.result = result


@dataclass(frozen=True, slots=True)
class GoalSpecV1:
    objective: str
    success_criteria: tuple[str, ...]
    constraints: tuple[str, ...] = ()
    in_scope: tuple[str, ...] = ()
    out_of_scope: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()
    questions: tuple[Mapping[str, Any], ...] = ()
    version: int = 1

    def __post_init__(self) -> None:
        if not self.objective.strip():
            raise AgentProtocolError("GoalSpecV1 requires a non-empty objective")
        if not self.success_criteria:
            raise AgentProtocolError("GoalSpecV1 requires observable success criteria")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "GoalSpecV1":
        data = _mapping(value.get("goal_spec", value))
        return cls(
            objective=str(data.get("objective", "")).strip(),
            success_criteria=_strings(data.get("success_criteria")),
            constraints=_strings(data.get("constraints")),
            in_scope=_strings(data.get("in_scope", data.get("scope"))),
            out_of_scope=_strings(data.get("out_of_scope", data.get("non_goals"))),
            assumptions=_strings(data.get("assumptions")),
            questions=tuple(
                dict(item) for item in data.get("questions", ()) if isinstance(item, Mapping)
            ),
        )


@dataclass(frozen=True, slots=True)
class ArchitectureSpecV1:
    summary: str
    components: tuple[Mapping[str, Any], ...]
    interfaces: tuple[Mapping[str, Any], ...] = ()
    decisions: tuple[Mapping[str, Any], ...] = ()
    dependencies: tuple[str, ...] = ()
    invariants: tuple[str, ...] = ()
    version: int = 1

    def __post_init__(self) -> None:
        if not self.summary.strip() or not self.components:
            raise AgentProtocolError("ArchitectureSpecV1 requires a summary and components")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ArchitectureSpecV1":
        data = _mapping(value.get("architecture", value))
        return cls(
            summary=str(data.get("summary", "")).strip(),
            components=tuple(
                dict(item) for item in data.get("components", ()) if isinstance(item, Mapping)
            ),
            interfaces=tuple(
                dict(item) for item in data.get("interfaces", ()) if isinstance(item, Mapping)
            ),
            decisions=tuple(
                dict(item) for item in data.get("decisions", ()) if isinstance(item, Mapping)
            ),
            dependencies=_strings(data.get("dependencies")),
            invariants=_strings(data.get("invariants")),
        )


@dataclass(frozen=True, slots=True)
class TaskContractV1:
    id: str
    title: str
    objective: str
    acceptance_criteria: tuple[str, ...]
    verification: tuple[str, ...]
    depends_on: tuple[str, ...] = ()
    write_paths: tuple[str, ...] = ()
    forbidden_changes: tuple[str, ...] = ()
    owned_interfaces: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    version: int = 1

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", self.id):
            raise AgentProtocolError(f"invalid ULTRA node id: {self.id!r}")
        if not self.title.strip() or not self.objective.strip():
            raise AgentProtocolError(f"node {self.id!r} requires a title and objective")
        if not self.acceptance_criteria or not self.verification:
            raise AgentProtocolError(
                f"node {self.id!r} requires acceptance criteria and verification"
            )
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, fallback_id: str = "") -> "TaskContractV1":
        title = str(value.get("title", value.get("name", ""))).strip()
        objective = str(value.get("objective", value.get("description", ""))).strip()
        acceptance = _strings(value.get("acceptance_criteria"))
        verification = _strings(value.get("verification"))
        if objective and not acceptance:
            acceptance = (f"Module outcome is implemented: {objective}",)
        if objective and not verification:
            verification = (f"Execute or inspect the module outcome against its objective: {objective}",)
        return cls(
            id=str(value.get("id", fallback_id)).strip(),
            title=title,
            objective=objective,
            acceptance_criteria=acceptance,
            verification=verification,
            depends_on=_strings(value.get("depends_on")),
            write_paths=_strings(value.get("write_paths", value.get("paths"))),
            forbidden_changes=_strings(value.get("forbidden_changes")),
            owned_interfaces=_strings(value.get("owned_interfaces", value.get("interfaces"))),
            metadata=_mapping(value.get("metadata")),
        )


@dataclass(frozen=True, slots=True)
class WorkNode:
    contract: TaskContractV1
    parent_id: str | None = None
    depth: int = 1
    kind: NodeKind = NodeKind.MODULE
    order: int = 0
    status: NodeStatus = NodeStatus.PENDING
    phase: InnerPhase | None = None
    children: tuple[str, ...] = ()
    pre_write_hashes: Mapping[str, str | None] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return self.contract.id

    @property
    def depends_on(self) -> tuple[str, ...]:
        return self.contract.depends_on

    @property
    def write_paths(self) -> tuple[str, ...]:
        return self.contract.write_paths


@dataclass(frozen=True, slots=True)
class SpecialistProfileV1:
    id: str
    node_id: str
    mission: str
    expertise: tuple[str, ...]
    context: Mapping[str, Any]
    owned_interfaces: tuple[str, ...]
    deliverable: str
    quality_rubric: Mapping[str, Any]
    dependencies: tuple[str, ...] = ()
    parent_profile_id: str | None = None
    version: int = 1


@dataclass(frozen=True, slots=True)
class ConcernRequirementV1:
    """An observable quality concern that must have an owner and verifier.

    Concerns are derived by the harness from the task family and repository
    evidence.  They are deliberately not dependent on the user knowing which
    engineering risks to mention in the prompt.
    """

    id: str
    title: str
    objective: str
    acceptance_criteria: tuple[str, ...]
    verification: tuple[str, ...]
    owner_hint: str
    critical: bool = True
    source: str = "harness_inferred"
    version: int = 1

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-z][a-z0-9_]{1,63}", self.id):
            raise AgentProtocolError(f"invalid concern id: {self.id!r}")
        if not self.title.strip() or not self.objective.strip():
            raise AgentProtocolError(f"concern {self.id!r} requires a title and objective")
        if not self.acceptance_criteria or not self.verification:
            raise AgentProtocolError(
                f"concern {self.id!r} requires acceptance criteria and verification"
            )


@dataclass(frozen=True, slots=True)
class ConcernCoverageMatrixV1:
    """Typed, inspectable proof that important task risks are owned."""

    task_family: str
    concerns: tuple[ConcernRequirementV1, ...]
    version: int = 1

    def __post_init__(self) -> None:
        ids = [item.id for item in self.concerns]
        if not self.task_family.strip() or not self.concerns:
            raise AgentProtocolError("ConcernCoverageMatrixV1 requires a family and concerns")
        if len(ids) != len(set(ids)):
            raise AgentProtocolError("ConcernCoverageMatrixV1 concern ids must be unique")

    @property
    def critical_ids(self) -> tuple[str, ...]:
        return tuple(item.id for item in self.concerns if item.critical)

    def missing_critical_owners(
        self,
        children: Iterable[Mapping[str, Any]],
    ) -> tuple[str, ...]:
        owned = {
            str(concern)
            for child in children
            if isinstance(child, Mapping)
            for concern in _strings(_mapping(child.get("metadata")).get("concern_ids"))
        }
        return tuple(item for item in self.critical_ids if item not in owned)


@dataclass(frozen=True, slots=True)
class LeafReadinessV1:
    node_id: str
    ready: bool
    score: float
    reasons: tuple[str, ...]
    recommended_children: int = 0
    version: int = 1


@dataclass(frozen=True, slots=True)
class ComponentPackageV1:
    node_id: str
    implementation: Mapping[str, Any]
    interface: Mapping[str, Any]
    tests: tuple[Mapping[str, Any], ...]
    preview: Mapping[str, Any]
    dependencies: tuple[str, ...]
    evidence: tuple[Mapping[str, Any], ...]
    quality: Mapping[str, Any]
    status: str = "published"
    version: int = 1


@dataclass(frozen=True, slots=True)
class NodeQualityTargetV1:
    node_id: str
    minimum_overall_score: float = 0.95
    minimum_critical_score: float = 0.90
    critical_dimensions: tuple[str, ...] = ("functional", "integration")
    plateau_window: int = 3
    plateau_delta: float = 0.02
    version: int = 1


@dataclass(frozen=True, slots=True)
class MasterPlanV1:
    summary: str
    modules: tuple[TaskContractV1, ...]
    milestones: tuple[Mapping[str, Any], ...] = ()
    execution_strategy: str = ""
    fingerprint: str = ""
    revision: int = 1
    version: int = 1

    def __post_init__(self) -> None:
        if not self.summary.strip() or not self.modules:
            raise AgentProtocolError("MasterPlanV1 requires a summary and modules")
        ids = [module.id for module in self.modules]
        if len(ids) != len(set(ids)):
            raise AgentProtocolError("MasterPlanV1 module ids must be unique")
        known = set(ids)
        for module in self.modules:
            missing = set(module.depends_on) - known
            if missing:
                raise AgentProtocolError(
                    f"module {module.id!r} has unknown dependencies: {sorted(missing)}"
                )
        if not self.fingerprint:
            payload = {
                "summary": self.summary,
                "modules": [asdict(module) for module in self.modules],
                "milestones": list(self.milestones),
                "execution_strategy": self.execution_strategy,
                "revision": self.revision,
            }
            object.__setattr__(self, "fingerprint", _fingerprint(payload))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "MasterPlanV1":
        data = _mapping(value.get("master_plan", value))
        raw_modules = [item for item in data.get("modules", ()) if isinstance(item, Mapping)]
        namespace_matches = [
            re.fullmatch(r"(r[a-f0-9]{12})\..+", str(item.get("id", "")).strip())
            for item in raw_modules
        ]
        namespaces = {match.group(1) for match in namespace_matches if match is not None}
        namespace = (
            next(iter(namespaces))
            if raw_modules and len(namespaces) == 1 and all(match is not None for match in namespace_matches)
            else ""
        )

        def canonical_id(index: int) -> str:
            local = f"M{index:03d}"
            return f"{namespace}.{local}" if namespace else local

        legacy_ids: dict[str, str] = {}
        for index, item in enumerate(raw_modules, start=1):
            generated = canonical_id(index)
            raw_id = str(item.get("id", "")).strip()
            if raw_id:
                legacy_ids[raw_id.casefold()] = generated
            legacy_ids[f"m{index}".casefold()] = generated
            legacy_ids[f"m{index:02d}".casefold()] = generated
            legacy_ids[f"m{index:03d}".casefold()] = generated
            legacy_ids[generated.casefold()] = generated

        normalized_modules: list[dict[str, Any]] = []
        for index, item in enumerate(raw_modules, start=1):
            normalized = dict(item)
            normalized["id"] = canonical_id(index)
            title = str(item.get("title", item.get("name", ""))).strip()
            objective = str(item.get("objective", item.get("description", ""))).strip()
            acceptance = _strings(item.get("acceptance_criteria"))
            verification = _strings(item.get("verification"))
            if not title and (objective or acceptance or verification):
                title = f"Module M{index:03d}"
            if not objective:
                if acceptance:
                    objective = acceptance[0]
                elif verification:
                    objective = verification[0]
            normalized["title"] = title
            normalized["objective"] = objective
            raw_dependencies = item.get("depends_on", ())
            if raw_dependencies is None:
                raw_dependencies = ()
            if not isinstance(raw_dependencies, (list, tuple)):
                raw_dependencies = (raw_dependencies,)
            dependencies: list[str] = []
            for dependency in raw_dependencies:
                text = str(dependency).strip()
                resolved = legacy_ids.get(text.casefold())
                if resolved is None:
                    # Weak models often copy the human label into a dependency,
                    # e.g. ``M001: Renderer Core``. The stable ID prefix is the
                    # dependency; descriptive suffixes must not break the DAG.
                    match = re.match(r"^[Mm]0*(\d+)(?:\b|\s*[:\-])", text)
                    resolved = canonical_id(int(match.group(1))) if match else text
                if resolved and resolved not in dependencies:
                    dependencies.append(resolved)
            normalized["depends_on"] = dependencies
            normalized_modules.append(normalized)
        modules = tuple(
            TaskContractV1.from_mapping(item, fallback_id=canonical_id(index))
            for index, item in enumerate(normalized_modules, start=1)
        )
        return cls(
            summary=str(data.get("summary", "")).strip(),
            modules=modules,
            milestones=tuple(
                dict(item) for item in data.get("milestones", ()) if isinstance(item, Mapping)
            ),
            execution_strategy=str(data.get("execution_strategy", "")).strip(),
            revision=max(1, int(data.get("revision", 1))),
        )


@dataclass(frozen=True, slots=True)
class InsightV1:
    summary: str
    node_id: str | None = None
    source: str = "agent"
    severity: str = "info"
    details: Mapping[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: _id("insight"))
    created_at: datetime = field(default_factory=_now)
    version: int = 1


@dataclass(frozen=True, slots=True)
class PromptTraceV1:
    run_id: str
    role: AgentRole
    phase: str
    system_prompt: str
    context_package: Mapping[str, Any]
    self_prompt: str
    reasoning_summary: str = ""
    node_id: str | None = None
    agent_run_id: str | None = None
    omitted_context: tuple[str, ...] = ()
    id: str = field(default_factory=lambda: _id("trace"))
    created_at: datetime = field(default_factory=_now)
    version: int = 1


@dataclass(frozen=True, slots=True)
class ResultPackageV1:
    node_id: str
    success: bool
    summary: str
    status: str = "completed"
    artifacts: tuple[Mapping[str, Any], ...] = ()
    evidence: tuple[Mapping[str, Any], ...] = ()
    test_results: tuple[Mapping[str, Any], ...] = ()
    findings: tuple[str, ...] = ()
    insights: tuple[InsightV1, ...] = ()
    component_package: Mapping[str, Any] = field(default_factory=dict)
    fix_attempts: int = 0
    version: int = 1


@dataclass(frozen=True, slots=True)
class QualityGateResultV1:
    responses: tuple[AgentResponse, ...]
    consensus: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class BrainEntryV1:
    section: BrainSection
    key: str
    value: Mapping[str, Any]
    run_id: str
    node_id: str | None = None
    role: AgentRole | None = None
    version: int = 1
    expires_at: datetime | None = None
    created_at: datetime = field(default_factory=_now)


@dataclass(frozen=True, slots=True)
class AgentRunV1:
    run_id: str
    role: AgentRole
    phase: str
    status: str
    provider: str = ""
    model: str = ""
    node_id: str | None = None
    usage: Mapping[str, int] = field(default_factory=dict)
    summary: str = ""
    error: str = ""
    prompt_trace_id: str | None = None
    id: str = field(default_factory=lambda: _id("agent"))
    created_at: datetime = field(default_factory=_now)


@dataclass(frozen=True, slots=True)
class UltraRunV1:
    prompt: str
    execution_class: ExecutionClass
    phase: UltraPhase = UltraPhase.NEW
    concurrency: int = 1
    master_fingerprint: str = ""
    approved: bool = False
    model_snapshot: Mapping[str, Any] = field(default_factory=dict)
    config_snapshot: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: _id("ultra"))
    created_at: datetime = field(default_factory=_now)
    updated_at: datetime = field(default_factory=_now)


@dataclass(frozen=True, slots=True)
class AgentRequest:
    run_id: str
    role: AgentRole
    phase: str
    system_prompt: str
    context: Mapping[str, Any]
    task: Mapping[str, Any]
    node_id: str | None = None
    agent_run_id: str | None = None


@dataclass(frozen=True, slots=True)
class AgentResponse:
    payload: Mapping[str, Any]
    summary: str = ""
    insights: tuple[InsightV1, ...] = ()
    reasoning_summary: str = ""
    usage: Mapping[str, int] = field(default_factory=dict)
    provider: str = ""
    model: str = ""

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        node_id: str | None = None,
        provider: str = "",
        model: str = "",
        usage: Mapping[str, int] | None = None,
    ) -> "AgentResponse":
        raw_insights = value.get("insights", ())
        insights: list[InsightV1] = []
        if isinstance(raw_insights, Sequence) and not isinstance(raw_insights, (str, bytes)):
            for item in raw_insights:
                if isinstance(item, str):
                    insights.append(InsightV1(item, node_id=node_id))
                elif isinstance(item, Mapping) and str(item.get("summary", "")).strip():
                    insights.append(
                        InsightV1(
                            summary=str(item["summary"]),
                            node_id=str(item.get("node_id") or node_id) if (item.get("node_id") or node_id) else None,
                            source=str(item.get("source", "agent")),
                            severity=str(item.get("severity", "info")),
                            details=_mapping(item.get("details")),
                        )
                    )
        payload = dict(_mapping(value.get("payload", value)))
        # The response contract places inspectable reasoning artifacts beside
        # ``summary`` and ``payload``. Preserve those typed envelope fields
        # when normalizing the phase payload; otherwise every valid reviewer
        # vote is downgraded after the model did exactly what was requested.
        for key in (
            "reasoning_artifact",
            "hypothesis",
            "architecture_candidate",
            "decision_record",
            "critic_verdict",
            "quality_finding",
        ):
            if key in value and key not in payload:
                payload[key] = value[key]
        return cls(
            payload=payload,
            summary=str(value.get("summary", "")).strip(),
            insights=tuple(insights),
            reasoning_summary=str(value.get("reasoning_summary", "")).strip(),
            usage=dict(usage or {}),
            provider=provider,
            model=model,
        )


class UltraAgent(Protocol):
    def execute(self, request: AgentRequest) -> AgentResponse: ...


class UltraAgentFactory(Protocol):
    def create(
        self,
        role: AgentRole,
        *,
        run_id: str,
        node_id: str | None = None,
    ) -> UltraAgent: ...


class UltraStateAdapter(Protocol):
    def save_ultra_run(self, run: UltraRunV1) -> None: ...
    def save_work_node(self, run_id: str, node: WorkNode) -> None: ...
    def save_agent_run(self, agent_run: AgentRunV1) -> None: ...
    def save_prompt_trace(self, trace: PromptTraceV1) -> None: ...
    def save_result_package(self, run_id: str, result: ResultPackageV1) -> None: ...
    def append_brain_entry(self, entry: BrainEntryV1) -> None: ...
    def list_brain_entries(self, run_id: str) -> tuple[BrainEntryV1, ...]: ...
    def get_result_package(self, run_id: str, node_id: str) -> ResultPackageV1 | None: ...
    def save_specialist_profile(self, run_id: str, profile: SpecialistProfileV1) -> None: ...
    def save_component_package(self, run_id: str, package: ComponentPackageV1) -> None: ...
    def save_node_quality_target(self, run_id: str, target: NodeQualityTargetV1) -> None: ...


class InMemoryUltraState:
    """Thread-safe reference adapter used by tests and incremental integration."""

    def __init__(self) -> None:
        self.runs: dict[str, UltraRunV1] = {}
        self.nodes: dict[str, dict[str, WorkNode]] = defaultdict(dict)
        self.agent_runs: list[AgentRunV1] = []
        self.traces: list[PromptTraceV1] = []
        self.results: dict[str, dict[str, ResultPackageV1]] = defaultdict(dict)
        self.brain: list[BrainEntryV1] = []
        self.specialists: dict[str, dict[str, SpecialistProfileV1]] = defaultdict(dict)
        self.component_packages: dict[str, dict[str, ComponentPackageV1]] = defaultdict(dict)
        self.quality_targets: dict[str, dict[str, NodeQualityTargetV1]] = defaultdict(dict)
        self._lock = threading.RLock()

    def save_ultra_run(self, run: UltraRunV1) -> None:
        with self._lock:
            self.runs[run.id] = run

    def save_work_node(self, run_id: str, node: WorkNode) -> None:
        with self._lock:
            self.nodes[run_id][node.id] = node

    def save_agent_run(self, agent_run: AgentRunV1) -> None:
        with self._lock:
            for index, current in enumerate(self.agent_runs):
                if current.id == agent_run.id:
                    self.agent_runs[index] = agent_run
                    break
            else:
                self.agent_runs.append(agent_run)

    def save_prompt_trace(self, trace: PromptTraceV1) -> None:
        with self._lock:
            for index, current in enumerate(self.traces):
                if current.id == trace.id:
                    self.traces[index] = trace
                    break
            else:
                self.traces.append(trace)

    def save_result_package(self, run_id: str, result: ResultPackageV1) -> None:
        with self._lock:
            self.results[run_id][result.node_id] = result

    def append_brain_entry(self, entry: BrainEntryV1) -> None:
        with self._lock:
            self.brain.append(entry)

    def list_brain_entries(self, run_id: str) -> tuple[BrainEntryV1, ...]:
        now = _now()
        with self._lock:
            return tuple(
                entry
                for entry in self.brain
                if entry.run_id == run_id
                and (entry.expires_at is None or entry.expires_at > now)
            )

    def get_result_package(self, run_id: str, node_id: str) -> ResultPackageV1 | None:
        with self._lock:
            return self.results.get(run_id, {}).get(node_id)

    def save_specialist_profile(self, run_id: str, profile: SpecialistProfileV1) -> None:
        with self._lock:
            self.specialists[run_id][profile.node_id] = profile

    def save_component_package(self, run_id: str, package: ComponentPackageV1) -> None:
        with self._lock:
            self.component_packages[run_id][package.node_id] = package

    def save_node_quality_target(self, run_id: str, target: NodeQualityTargetV1) -> None:
        with self._lock:
            self.quality_targets[run_id][target.node_id] = target


class JournaledUltraState(InMemoryUltraState):
    """Compatibility bridge that mirrors v3 records into current StateStore events.

    The forthcoming schema-v3 store can implement :class:`UltraStateAdapter`
    directly.  Until then this adapter gives crash/debug journals without
    reaching into StateStore internals.
    """

    def __init__(self, state_store: Any) -> None:
        super().__init__()
        self.state_store = state_store

    def _journal(self, kind: str, entity_id: str, payload: Mapping[str, Any]) -> None:
        self.state_store.append_event(
            kind,
            entity_type="ultra",
            entity_id=entity_id,
            payload=redact_data(dict(payload)),
        )

    def save_ultra_run(self, run: UltraRunV1) -> None:
        super().save_ultra_run(run)
        self._journal("ultra.run.saved", run.id, asdict(run))

    def save_work_node(self, run_id: str, node: WorkNode) -> None:
        super().save_work_node(run_id, node)
        self._journal("ultra.node.saved", node.id, {"run_id": run_id, **asdict(node)})

    def save_agent_run(self, agent_run: AgentRunV1) -> None:
        super().save_agent_run(agent_run)
        self._journal("ultra.agent.saved", agent_run.id, asdict(agent_run))

    def save_prompt_trace(self, trace: PromptTraceV1) -> None:
        super().save_prompt_trace(trace)
        self._journal("ultra.trace.saved", trace.id, asdict(trace))

    def save_result_package(self, run_id: str, result: ResultPackageV1) -> None:
        super().save_result_package(run_id, result)
        self._journal("ultra.result.saved", result.node_id, {"run_id": run_id, **asdict(result)})

    def append_brain_entry(self, entry: BrainEntryV1) -> None:
        super().append_brain_entry(entry)
        self._journal("ultra.brain.saved", entry.key, asdict(entry))


@dataclass(frozen=True, slots=True)
class ContextRequest:
    run: UltraRunV1
    node: WorkNode
    role: AgentRole
    goal: GoalSpecV1
    architecture: ArchitectureSpecV1
    plan: MasterPlanV1
    nodes: Mapping[str, WorkNode]
    brain: tuple[BrainEntryV1, ...]
    dependency_results: Mapping[str, ResultPackageV1]


class ContextBuilder(Protocol):
    def build(self, request: ContextRequest) -> Mapping[str, Any]: ...


class FocusedContextBuilder:
    """Build ordered, bounded context and report any budget omissions."""

    def __init__(self, max_chars: int = 24_000) -> None:
        self.max_chars = max(2_000, int(max_chars))

    def build(self, request: ContextRequest) -> Mapping[str, Any]:
        ancestors: list[Mapping[str, Any]] = []
        current = request.node
        while current.parent_id and current.parent_id in request.nodes:
            current = request.nodes[current.parent_id]
            ancestors.append(asdict(current.contract))
        decisions = [asdict(entry) for entry in request.brain if entry.section is BrainSection.DECISION]
        role_memory = [
            asdict(entry)
            for entry in request.brain
            if entry.section is BrainSection.ROLE_MEMORY and entry.role is request.role
        ]
        dependency_results = {
            node_id: asdict(result) for node_id, result in request.dependency_results.items()
        }
        package: dict[str, Any] = {
            "node": asdict(request.node.contract),
            "ancestors": ancestors,
            "north_star": asdict(request.goal),
            "architecture": {
                "summary": request.architecture.summary,
                "interfaces": list(request.architecture.interfaces),
                "invariants": list(request.architecture.invariants),
            },
            "decisions": decisions,
            "dependency_artifacts": dependency_results,
            "role_memory": role_memory,
            "_omitted": [],
        }
        # Drop least essential categories in a deterministic order.  The node,
        # ancestors, north star, and architecture contract are never omitted.
        for key in ("role_memory", "dependency_artifacts", "decisions"):
            if len(_json(package)) <= self.max_chars:
                break
            package[key] = [] if key != "dependency_artifacts" else {}
            package["_omitted"].append(key)
        if len(_json(package)) > self.max_chars:
            package["architecture"] = {
                "summary": request.architecture.summary,
                "interfaces": list(request.node.contract.owned_interfaces),
            }
            package["_omitted"].append("architecture_details")
        return package


def _extract_json(text: str) -> Mapping[str, Any]:
    value, _actions = repair_structured_json_object(text)
    if value is None:
        raise AgentProtocolError("ULTRA agents must return one JSON object")
    if not isinstance(value, Mapping):
        raise AgentProtocolError("ULTRA agent response must be a JSON object")
    return value


class ProviderAgentAdapter:
    """Adapt the existing neutral Provider API to one role-specific ULTRA call."""

    def __init__(self, provider: Any, *, provider_name: str = "", model: str = "") -> None:
        self.provider = provider
        self.provider_name = provider_name or type(provider).__name__
        self.model = model or str(getattr(provider, "model", ""))

    def execute(self, request: AgentRequest) -> AgentResponse:
        user_payload = {
            "task": request.task,
            "focused_context": request.context,
            "response_contract": {
                "payload": "phase-specific structured object",
                "summary": "brief factual result summary",
                "reasoning_summary": "brief conclusion and evidence; no hidden chain-of-thought",
                "insights": "optional list of structured insights",
            },
        }
        try:
            turn = self.provider.call(
                [{"role": "user", "content": _json(user_payload)}],
                [],
                request.system_prompt,
            )
        except Exception as exc:
            status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
            if status == 429 or "rate limit" in str(exc).casefold():
                raise RateLimitError(str(exc)) from exc
            raise
        data = _extract_json(str(getattr(turn, "text", "") or ""))
        usage_obj = getattr(turn, "usage", None)
        usage = {
            key: int(getattr(usage_obj, key, 0) or 0)
            for key in ("input_tokens", "cached_tokens", "output_tokens")
        }
        return AgentResponse.from_mapping(
            data,
            node_id=request.node_id,
            provider=self.provider_name,
            model=self.model,
            usage=usage,
        )


class ProviderFactoryAdapter:
    """Create an independent provider instance for every role invocation."""

    def __init__(
        self,
        provider_factory: Callable[[], Any],
        *,
        provider_name: str = "",
        model: str = "",
    ) -> None:
        self.provider_factory = provider_factory
        self.provider_name = provider_name
        self.model = model

    def create(
        self,
        role: AgentRole,
        *,
        run_id: str,
        node_id: str | None = None,
    ) -> ProviderAgentAdapter:
        del role, run_id, node_id
        return ProviderAgentAdapter(
            self.provider_factory(),
            provider_name=self.provider_name,
            model=self.model,
        )


@dataclass(frozen=True, slots=True)
class UltraConfig:
    min_top_modules: int = 4
    max_top_modules: int = 12
    max_depth: int = 8
    max_nodes: int = 1_000
    max_fix_attempts: int = 3
    cloud_concurrency: int = 4
    max_concurrency: int = 8
    provider_retries: int = 3
    role_memory_ttl_hours: int = 24
    context_chars: int = 24_000
    prompt_trace_chars: int = 256_000

    def __post_init__(self) -> None:
        if not 1 <= self.min_top_modules <= self.max_top_modules <= 80:
            raise ValueError("invalid top-level module bounds")
        if not 1 <= self.max_depth <= 10 or not 1 <= self.max_nodes <= 5_000:
            raise ValueError("invalid ULTRA expansion bounds")
        if not 1 <= self.max_fix_attempts <= 20:
            raise ValueError("max_fix_attempts must be between 1 and 20")
        if not 4_096 <= self.prompt_trace_chars <= 2_000_000:
            raise ValueError("prompt_trace_chars must be between 4096 and 2000000")


@dataclass(frozen=True, slots=True)
class UltraRunResult:
    run: UltraRunV1
    plan: MasterPlanV1
    node_results: tuple[ResultPackageV1, ...]
    schedule: ScheduleReport[ResultPackageV1] | None = None
    global_result: ResultPackageV1 | None = None

    @property
    def successful(self) -> bool:
        return self.run.phase is UltraPhase.COMPLETED


_SYSTEM_PROMPT = """You are the {role} in GA3BAD ULTRA mode.
Perform only phase {phase}. Respect the approved task contract, write scopes,
forbidden changes, interfaces, and success criteria in focused_context.
Return exactly one JSON object matching response_contract. Give only a concise,
factual reasoning_summary with decisions and evidence. Never reveal or request
hidden chain-of-thought. Do not invent tool results or completion evidence.
"""


class UltraOrchestrator:
    """Goal-to-evidence ULTRA state machine with role-isolated agent calls."""

    def __init__(
        self,
        agent_factory: UltraAgentFactory,
        *,
        execution_class: ExecutionClass | str = ExecutionClass.LOCAL,
        state: UltraStateAdapter | None = None,
        events: EventBus | None = None,
        config: UltraConfig | None = None,
        context_builder: ContextBuilder | None = None,
        control: CooperativeControl | None = None,
        leases: ResourceLeaseManager | None = None,
        rate_limit_backoff: Callable[[int], None] | None = None,
        model_snapshot: Mapping[str, Any] | None = None,
    ) -> None:
        self.agent_factory = agent_factory
        self.execution_class = ExecutionClass(execution_class)
        self.state = state or InMemoryUltraState()
        self.events = events or EventBus()
        self.config = config or UltraConfig()
        self.context_builder = context_builder or FocusedContextBuilder(self.config.context_chars)
        self.control = control or CooperativeControl()
        self.adaptive = AdaptiveConcurrency(
            self.execution_class,
            cloud_default=self.config.cloud_concurrency,
            maximum=self.config.max_concurrency,
        )
        self.scheduler = DeterministicWaveScheduler(
            self.execution_class,
            cloud_default=self.config.cloud_concurrency,
            maximum=self.config.max_concurrency,
            rate_limit_retries=self.config.provider_retries,
            rate_limit_backoff=rate_limit_backoff,
            leases=leases,
            control=self.control,
            adaptive=self.adaptive,
            on_event=self._scheduler_event,
        )
        self.background = BackgroundRunController[UltraRunResult](self.control)
        self.model_snapshot = dict(model_snapshot or {})
        self.run_state: UltraRunV1 | None = None
        self.goal_spec: GoalSpecV1 | None = None
        self.architecture: ArchitectureSpecV1 | None = None
        self.master_plan: MasterPlanV1 | None = None
        self.nodes: dict[str, WorkNode] = {}
        self._prepared: dict[str, tuple[Mapping[str, Any], Mapping[str, Any]]] = {}
        self._research_required: dict[str, bool] = {}
        self._results: dict[str, ResultPackageV1] = {}
        self._insights: list[InsightV1] = []
        self._order = 0
        self._lock = threading.RLock()
        self._phase_before_pause: UltraPhase | None = None

    @property
    def phase(self) -> UltraPhase:
        return self.run_state.phase if self.run_state else UltraPhase.NEW

    def _scheduler_event(self, kind: str, message: str, data: Mapping[str, Any]) -> None:
        self.events.publish(kind, message, **dict(data))

    def _save_run(self, **changes: Any) -> UltraRunV1:
        if self.run_state is None:
            raise UltraError("ULTRA run has not been prepared")
        self.run_state = replace(
            self.run_state,
            updated_at=_now(),
            concurrency=self.adaptive.current,
            **changes,
        )
        self.state.save_ultra_run(self.run_state)
        return self.run_state

    def _set_phase(self, phase: UltraPhase, message: str = "") -> None:
        ensure_ultra_phase_transition(self.phase, phase)
        self._save_run(phase=phase)
        self.events.publish(
            UltraEventKind.PHASE.value,
            message or phase.value.replace("_", " ").title(),
            run_id=self.run_state.id,
            phase=phase.value,
        )

    def _new_context(self, node: WorkNode, role: AgentRole) -> Mapping[str, Any]:
        assert self.run_state and self.goal_spec and self.architecture and self.master_plan
        dependencies = {
            dep: result for dep in node.depends_on if (result := self._results.get(dep)) is not None
        }
        return self.context_builder.build(
            ContextRequest(
                run=self.run_state,
                node=node,
                role=role,
                goal=self.goal_spec,
                architecture=self.architecture,
                plan=self.master_plan,
                nodes=dict(self.nodes),
                brain=self.state.list_brain_entries(self.run_state.id),
                dependency_results=dependencies,
            )
        )

    def _invoke(
        self,
        role: AgentRole,
        phase: InnerPhase | str,
        *,
        task: Mapping[str, Any],
        context: Mapping[str, Any],
        node_id: str | None = None,
    ) -> AgentResponse:
        assert self.run_state
        self.control.checkpoint()
        phase_value = phase.value if isinstance(phase, InnerPhase) else str(phase)
        system = _SYSTEM_PROMPT.format(role=role.value, phase=phase_value).strip()
        safe_context = redact_data(dict(context))
        safe_task = redact_data(dict(task))
        if phase_value == "master_plan":
            # Engine node identifiers are persisted in a shared project DB.
            # Give each run its own protocol namespace so a later replan can
            # safely reuse human-friendly model ids such as M001.
            safe_task["protocol_node_namespace"] = f"r{self.run_state.id[-12:]}"
        request = AgentRequest(
            run_id=self.run_state.id,
            role=role,
            phase=phase_value,
            system_prompt=system,
            context=safe_context,
            task=safe_task,
            node_id=node_id,
        )
        last_error: Exception | None = None
        typed_repair_used = False
        for attempt in range(1, self.config.provider_retries + 3):
            self.control.checkpoint()
            agent_id = _id("agent")
            omitted = safe_context.get("_omitted", ())
            trace = PromptTraceV1(
                run_id=self.run_state.id,
                role=role,
                phase=phase_value,
                system_prompt=redact_text(system)[:8_000],
                context_package=safe_context,
                self_prompt=redact_text(_json(safe_task))[:16_000],
                node_id=node_id,
                agent_run_id=agent_id,
                omitted_context=_strings(omitted),
            )
            stage_next_action = getattr(self.state, "stage_next_action", None)
            if callable(stage_next_action):
                stage_next_action(
                    agent_id,
                    role=role.value,
                    phase=phase_value,
                    node_id=node_id,
                    task=safe_task,
                    context=safe_context,
                )
            self.state.save_agent_run(
                AgentRunV1(
                    id=agent_id,
                    run_id=self.run_state.id,
                    role=role,
                    phase=phase_value,
                    status="running",
                    provider=str(self.model_snapshot.get("provider", "")),
                    model=str(self.model_snapshot.get("model", "")),
                    node_id=node_id,
                    prompt_trace_id=trace.id,
                )
            )
            # Save the exact redacted assignment before the local provider
            # call, so the read-only observer remains useful during a long
            # generation. Hidden chain-of-thought is never captured.
            self.state.save_prompt_trace(trace)
            response: AgentResponse | None = None
            try:
                agent = self.agent_factory.create(
                    role, run_id=self.run_state.id, node_id=node_id
                )
                response = agent.execute(replace(request, agent_run_id=agent_id))
                normalized_payload, normalization_actions = self._normalize_typed_payload(
                    phase_value, response.payload, safe_task
                )
                if normalization_actions:
                    response = replace(response, payload=normalized_payload)
                    self.events.publish(
                        "ultra.typed_normalized",
                        "; ".join(normalization_actions),
                        run_id=self.run_state.id,
                        node_id=node_id,
                        role=role.value,
                        phase=phase_value,
                        actions=list(normalization_actions),
                    )
                # Typed returns are validated before the lifecycle can become
                # COMPLETED.  This prevents an empty GoalSpec (and equivalent
                # malformed review/worker payloads) from being accepted and
                # failing later in the engine.
                self._validate_typed_response(phase_value, response.payload)
                self.adaptive.on_success()
                agent_run = AgentRunV1(
                    id=agent_id,
                    run_id=self.run_state.id,
                    role=role,
                    phase=phase_value,
                    status="completed",
                    provider=response.provider,
                    model=response.model,
                    node_id=node_id,
                    usage=dict(response.usage),
                    summary=redact_text(response.summary)[:4_000],
                    prompt_trace_id=trace.id,
                )
                self.state.save_agent_run(agent_run)
                trace = replace(
                    trace,
                    reasoning_summary=redact_text(response.reasoning_summary)[:4_000],
                )
                self.state.save_prompt_trace(trace)
                for insight in response.insights:
                    self._insights.append(insight)
                    self.state.append_brain_entry(
                        BrainEntryV1(
                            BrainSection.KNOWLEDGE,
                            insight.id,
                            {
                                "summary": insight.summary,
                                "severity": insight.severity,
                                "source": insight.source,
                                "details": dict(insight.details),
                            },
                            self.run_state.id,
                            node_id=insight.node_id,
                            role=role,
                        )
                    )
                    self.events.publish(
                        UltraEventKind.INSIGHT.value,
                        insight.summary,
                        run_id=self.run_state.id,
                        node_id=insight.node_id,
                        severity=insight.severity,
                    )
                self.events.publish(
                    UltraEventKind.AGENT.value,
                    response.summary or f"{role.value} finished {phase_value}",
                    run_id=self.run_state.id,
                    agent_run_id=agent_run.id,
                    role=role.value,
                    phase=phase_value,
                    node_id=node_id,
                )
                return response
            except AgentProtocolError as exc:
                self.state.save_agent_run(
                    AgentRunV1(
                        id=agent_id,
                        run_id=self.run_state.id,
                        role=role,
                        phase=phase_value,
                        status="failed",
                        provider=str(self.model_snapshot.get("provider", "")),
                        model=str(self.model_snapshot.get("model", "")),
                        node_id=node_id,
                        error=redact_text(f"typed return validation: {exc}")[:4_000],
                    )
                )
                self.events.publish(
                    "ultra.typed_return_rejected",
                    f"{role.value} returned invalid {phase_value}: {redact_text(exc, 500)}",
                    run_id=self.run_state.id,
                    agent_run_id=agent_id,
                    role=role.value,
                    phase=phase_value,
                    repair_attempt=1 if not typed_repair_used else 2,
                )
                unrecoverable_transport_token = "unused token" in str(exc).casefold()
                if typed_repair_used or unrecoverable_transport_token:
                    raise AgentProtocolError(
                        f"ULTRA foundation/phase {phase_value} failed after one targeted typed-return repair: {exc}"
                    ) from exc
                typed_repair_used = True
                repair_task = {
                    **safe_task,
                    "typed_return_repair": {
                        "contract": phase_value,
                        "errors": [str(exc)],
                        "previous_payload": (
                            redact_data(dict(response.payload))
                            if response is not None
                            else {}
                        ),
                        "instruction": "Return one complete corrected payload. Repair only the listed contract defects.",
                    },
                }
                request = replace(request, task=repair_task)
                continue
            except RateLimitError as exc:
                last_error = exc
                concurrency = self.adaptive.on_rate_limit()
                self.state.save_agent_run(
                    AgentRunV1(
                        id=agent_id,
                        run_id=self.run_state.id,
                        role=role,
                        phase=phase_value,
                        status="rate_limited",
                        provider=str(self.model_snapshot.get("provider", "")),
                        model=str(self.model_snapshot.get("model", "")),
                        node_id=node_id,
                        error=redact_text(str(exc))[:4_000],
                    )
                )
                self.events.publish(
                    UltraEventKind.AGENT.value,
                    f"{role.value} rate limited during {phase_value}",
                    run_id=self.run_state.id,
                    agent_run_id=agent_id,
                    role=role.value,
                    phase=phase_value,
                    node_id=node_id,
                    status="rate_limited",
                )
                self.events.publish(
                    "ultra.rate_limited",
                    f"Rate limited; concurrency reduced to {concurrency}",
                    run_id=self.run_state.id,
                    role=role.value,
                    node_id=node_id,
                    attempt=attempt,
                    concurrency=concurrency,
                )
                if attempt > self.config.provider_retries:
                    self.state.save_agent_run(
                        AgentRunV1(
                            id=agent_id,
                            run_id=self.run_state.id,
                            role=role,
                            phase=phase_value,
                            status="failed",
                            provider=str(self.model_snapshot.get("provider", "")),
                            model=str(self.model_snapshot.get("model", "")),
                            node_id=node_id,
                            error=redact_text(str(exc))[:4_000],
                        )
                    )
                    break
            except CancellationRequested:
                self.state.save_agent_run(
                    AgentRunV1(
                        id=agent_id,
                        run_id=self.run_state.id,
                        role=role,
                        phase=phase_value,
                        status="cancelled",
                        provider=str(self.model_snapshot.get("provider", "")),
                        model=str(self.model_snapshot.get("model", "")),
                        node_id=node_id,
                        error="cancelled at a safe checkpoint",
                    )
                )
                self.events.publish(
                    UltraEventKind.AGENT.value,
                    f"{role.value} cancelled during {phase_value}",
                    run_id=self.run_state.id,
                    agent_run_id=agent_id,
                    role=role.value,
                    phase=phase_value,
                    node_id=node_id,
                    status="cancelled",
                )
                raise
            except Exception as exc:
                self.state.save_agent_run(
                    AgentRunV1(
                        id=agent_id,
                        run_id=self.run_state.id,
                        role=role,
                        phase=phase_value,
                        status="failed",
                        node_id=node_id,
                        error=redact_text(f"{type(exc).__name__}: {exc}")[:4_000],
                    )
                )
                self.events.publish(
                    UltraEventKind.AGENT.value,
                    f"{role.value} failed during {phase_value}",
                    run_id=self.run_state.id,
                    agent_run_id=agent_id,
                    role=role.value,
                    phase=phase_value,
                    node_id=node_id,
                    status="failed",
                )
                raise
        assert last_error is not None
        raise last_error

    def _foundation_project_lessons(self, query: str, phase: str) -> tuple[Mapping[str, Any], ...]:
        if not self.run_state:
            return ()
        getter = getattr(self.state, "foundation_project_lessons", None)
        if not callable(getter):
            return ()
        try:
            raw = getter(self.run_state.id, query, phase=phase, limit=4)
        except TypeError:
            raw = getter(self.run_state.id, query, phase=phase)
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            return ()
        compact: list[Mapping[str, Any]] = []
        seen: set[str] = set()
        for item in raw:
            if not isinstance(item, Mapping):
                continue
            content = str(item.get("content", "")).strip()
            title = str(item.get("title", "")).strip()
            key = re.sub(r"\s+", " ", f"{title} {content}").casefold()[:500]
            if not key or key in seen:
                continue
            seen.add(key)
            compact.append(
                {
                    "id": item.get("id"),
                    "section": item.get("section"),
                    "phase": phase,
                    "title": title[:200],
                    "content": content[:1_200],
                    "confidence": item.get("confidence"),
                    "effective_confidence": item.get("effective_confidence"),
                    "reuse_count": item.get("reuse_count", 0),
                    "evidence_refs": list(item.get("evidence_refs", ()) or ())[:4],
                }
            )
            if len(compact) >= 4:
                break
        return tuple(compact)

    @staticmethod
    def _normalize_typed_payload(
        phase: str,
        payload: Mapping[str, Any],
        task: Mapping[str, Any],
    ) -> tuple[dict[str, Any], tuple[str, ...]]:
        normalized = dict(payload)
        actions: list[str] = []
        nested = normalized.get(phase)
        if isinstance(nested, Mapping):
            normalized = dict(nested)
            actions.append(f"{phase} envelope unwrapped")
        if phase == "goal_spec" and not str(normalized.get("objective", "")).strip():
            authoritative_prompt = str(task.get("prompt", "")).strip()
            if authoritative_prompt:
                normalized["objective"] = authoritative_prompt
                actions.append("goal_spec.objective restored from authoritative user prompt")
        if phase == "goal_spec":
            authoritative_prompt = str(task.get("prompt", "")).strip()
            prompt_key = authoritative_prompt.casefold()
            constraints = list(_strings(normalized.get("constraints")))
            in_scope = list(_strings(normalized.get("in_scope", normalized.get("scope"))))
            success_criteria = list(_strings(normalized.get("success_criteria")))
            if (
                "index.html" in prompt_key
                and any(term in prompt_key for term in ("single-file", "single file", "self-contained"))
            ):
                constraint = (
                    "All implementation must remain in one workspace-relative index.html; "
                    "no other local JS, CSS, asset, or build files are allowed."
                )
                if constraint not in constraints:
                    constraints.append(constraint)
                if "Single workspace-relative index.html artifact" not in in_scope:
                    in_scope.append("Single workspace-relative index.html artifact")
                actions.append("goal_spec preserved explicit single-file index.html boundary")
            if "jsdelivr" in prompt_key and "three" in prompt_key:
                constraint = (
                    "Three.js from the explicitly requested jsDelivr CDN is the only allowed "
                    "external runtime dependency."
                )
                if constraint not in constraints:
                    constraints.append(constraint)
                actions.append("goal_spec preserved explicit Three.js jsDelivr dependency boundary")
            if any(term in prompt_key for term in ("browser verification", "screenshot", "visual quality")):
                criterion = (
                    "The running artifact passes real browser verification and screenshot-based "
                    "visual quality review after any refinement."
                )
                if criterion not in success_criteria:
                    success_criteria.append(criterion)
                actions.append("goal_spec preserved explicit browser visual-quality gate")
            if constraints:
                normalized["constraints"] = constraints
            if in_scope:
                normalized["in_scope"] = in_scope
            if success_criteria:
                normalized["success_criteria"] = success_criteria
        if phase == "master_plan":
            goal_spec = _mapping(task.get("goal_spec"))
            architecture = _mapping(task.get("architecture"))
            modules = [dict(item) for item in normalized.get("modules", ()) if isinstance(item, Mapping)]
            restored_from_architecture = not modules
            if not modules:
                components = [
                    dict(item)
                    for item in architecture.get("components", ())
                    if isinstance(item, Mapping)
                ]
                for index, component in enumerate(components, start=1):
                    name = str(component.get("name") or component.get("title") or f"Module {index}").strip()
                    responsibility = str(
                        component.get("responsibility")
                        or component.get("objective")
                        or f"Implement {name}"
                    ).strip()
                    modules.append(
                        {
                            "id": f"M{index:03d}",
                            "title": name,
                            "objective": responsibility,
                            "acceptance_criteria": [f"{name} satisfies its architecture responsibility: {responsibility}"],
                            "verification": [f"Execute or inspect {name} against its architecture contract"],
                            "depends_on": [] if index == 1 else [f"M{index - 1:03d}"],
                            "write_paths": ["index.html"],
                        }
                    )
            if restored_from_architecture and modules:
                normalized["modules"] = modules
                normalized["summary"] = str(
                    normalized.get("summary") or architecture.get("summary") or goal_spec.get("objective")
                ).strip()
                normalized["execution_strategy"] = str(
                    normalized.get("execution_strategy")
                    or "Implement architecture components sequentially, then run the final quality gate."
                ).strip()
                actions.append("master_plan.modules restored from accepted architecture components")

            # Requirements may be distributed across objective, constraints,
            # scope, and success criteria. Inspect the whole accepted GoalSpec
            # so a terse weak-model objective cannot erase an explicit QA gate.
            objective_text = json.dumps(goal_spec, ensure_ascii=False, default=str).casefold()
            requires_single_index = (
                "index.html" in objective_text
                and any(
                    term in objective_text
                    for term in (
                        "single-file",
                        "single file",
                        "self-contained",
                        "single index.html",
                        "one workspace-relative index.html",
                        "all code must reside in a single",
                    )
                )
            )
            if requires_single_index:
                rewrote_paths = False
                only_three_cdn = "jsdelivr" in objective_text and "three" in objective_text
                for item in modules:
                    current_paths = tuple(str(path) for path in item.get("write_paths", ()) or ())
                    if current_paths != ("index.html",):
                        item["write_paths"] = ["index.html"]
                        rewrote_paths = True
                    acceptance = list(_strings(item.get("acceptance_criteria")))
                    single_file_gate = (
                        "All implementation remains inside index.html; no local JS, CSS, asset, or build files are created."
                    )
                    if single_file_gate not in acceptance:
                        acceptance.append(single_file_gate)
                    if only_three_cdn:
                        dependency_gate = (
                            "The explicitly allowed Three.js jsDelivr CDN is the only external runtime dependency; no other libraries or assets are introduced."
                        )
                        if dependency_gate not in acceptance:
                            acceptance.append(dependency_gate)
                    item["acceptance_criteria"] = acceptance
                if only_three_cdn and re.search(r"\b(?:hammer|babylon|phaser|jquery|react|vue)\.?(?:js)?\b", str(normalized.get("summary", "")), re.I):
                    normalized["summary"] = str(goal_spec.get("objective") or "Implement the approved single-file Three.js game").strip()
                    actions.append("master_plan summary removed an unapproved external runtime dependency")
                if rewrote_paths:
                    actions.append("master_plan write ownership constrained to explicit single index.html artifact")
                normalized["modules"] = modules
            if not re.search(r"\b0\.\d+\s*,\s*0\.\d+", objective_text):
                for item in modules:
                    acceptance = list(_strings(item.get("acceptance_criteria")))
                    normalized_acceptance: list[str] = []
                    replaced_prng_fixture = False
                    for criterion in acceptance:
                        key = criterion.casefold()
                        if (
                            any(term in key for term in ("prng", "nextfloat", "random sequence"))
                            and re.search(r"\[\s*0\.\d+", criterion)
                        ):
                            replacement = (
                                "The same explicit PRNG seed produces the same sequence across reloads; "
                                "no Math.random source is used."
                            )
                            if replacement not in normalized_acceptance:
                                normalized_acceptance.append(replacement)
                            replaced_prng_fixture = True
                        else:
                            normalized_acceptance.append(criterion)
                    if replaced_prng_fixture:
                        item["acceptance_criteria"] = normalized_acceptance
                        actions.append(
                            "master_plan removed an unrequested brittle PRNG numeric fixture"
                        )
                normalized["modules"] = modules
            requires_browser_qa = any(
                term in objective_text
                for term in ("browser", "screenshot", "visual refinement", "visual quality")
            )
            dedicated_qa_terms = ("browser qa", "visual qa", "quality gate", "validation gate")
            has_dedicated_browser_qa = any(
                any(term in str(item.get("title", "")).casefold() for term in dedicated_qa_terms)
                for item in modules
            )
            if has_dedicated_browser_qa:
                qa_modules = [
                    item
                    for item in modules
                    if any(term in str(item.get("title", "")).casefold() for term in dedicated_qa_terms)
                ]
                implementation_modules = [item for item in modules if item not in qa_modules]
                qa_ids = {str(item.get("id", "")) for item in qa_modules}
                for item in implementation_modules:
                    item["depends_on"] = [
                        dependency
                        for dependency in _strings(item.get("depends_on"))
                        if dependency not in qa_ids
                    ]
                if implementation_modules:
                    last_implementation_id = str(implementation_modules[-1].get("id", ""))
                    for item in qa_modules:
                        item["depends_on"] = [last_implementation_id]
                    modules = [*implementation_modules, *qa_modules]
                    normalized["modules"] = modules
                    actions.append("master_plan Browser QA gate ordered after implementation")
            # A screenshot mentioned inside an implementation module is not an
            # independent quality gate: weak planners commonly use it as a
            # checkbox and omit interaction coverage, error evidence, and the
            # mandatory refine-and-retest loop.
            if requires_browser_qa and not has_dedicated_browser_qa:
                bounds = _mapping(task.get("module_bounds"))
                try:
                    maximum_modules = max(0, int(bounds.get("maximum", 0) or 0))
                except (TypeError, ValueError):
                    maximum_modules = 0
                if maximum_modules >= 2 and len(modules) >= maximum_modules:
                    merge_at = maximum_modules - 2
                    head = modules[:merge_at]
                    tail = modules[merge_at:]
                    tail_ids = {str(item.get("id", "")) for item in tail}
                    merged_id = str(tail[0].get("id") or f"M{merge_at + 1:03d}")
                    merged_dependencies: list[str] = []
                    for item in tail:
                        for dependency in _strings(item.get("depends_on")):
                            if dependency not in tail_ids and dependency not in merged_dependencies:
                                merged_dependencies.append(dependency)
                    merged = {
                        "id": merged_id,
                        "title": "Combined gameplay implementation",
                        "objective": " Integrate these approved responsibilities: " + " | ".join(
                            str(item.get("objective") or item.get("title") or "implementation")
                            for item in tail
                        ),
                        "acceptance_criteria": list(
                            dict.fromkeys(
                                criterion
                                for item in tail
                                for criterion in _strings(item.get("acceptance_criteria"))
                            )
                        )[:20],
                        "verification": list(
                            dict.fromkeys(
                                step
                                for item in tail
                                for step in _strings(item.get("verification"))
                            )
                        )[:20],
                        "depends_on": merged_dependencies,
                        "write_paths": list(
                            dict.fromkeys(
                                path
                                for item in tail
                                for path in _strings(item.get("write_paths"))
                            )
                        ),
                        "forbidden_changes": list(
                            dict.fromkeys(
                                value
                                for item in tail
                                for value in _strings(item.get("forbidden_changes"))
                            )
                        ),
                        "owned_interfaces": list(
                            dict.fromkeys(
                                value
                                for item in tail
                                for value in _strings(item.get("owned_interfaces"))
                            )
                        ),
                        "metadata": {"coalesced_module_ids": sorted(tail_ids)},
                    }
                    modules = [*head, merged]
                    actions.append(
                        f"master_plan coalesced {len(tail)} implementation modules to reserve Browser QA capacity"
                    )
                qa_id = f"M{len(modules) + 1:03d}"
                modules.append(
                    {
                        "id": qa_id,
                        "title": "Browser QA and Visual Refinement Gate",
                        "objective": "Run the real game in a browser, prove gameplay and runtime stability, score visual quality, and refine any below-target candidate.",
                        "acceptance_criteria": [
                            "A real 1280x720 screenshot is captured from the running game, never a placeholder.",
                            "Console and page errors are zero and unexpected network failures are absent or handled by a visible graceful fallback.",
                            "Primary input, core hazards/collisions, scoring or progress, game-over, and restart are exercised exactly as required by the GoalSpec.",
                            "Visual hierarchy, depth, contrast, density, legibility, and responsiveness meet the approved quality target.",
                            "Any below-target visual dimension triggers code refinement and fresh browser verification.",
                        ],
                        "verification": [
                            "Preview index.html at 1280x720 and capture the actual browser screenshot.",
                            "Inspect console, page, and network evidence and exercise the GoalSpec's complete gameplay lifecycle.",
                            "Repeat screenshot and runtime verification after every visual refinement mutation.",
                        ],
                        "depends_on": [str(modules[-1].get("id"))] if modules else [],
                        "write_paths": ["index.html"],
                    }
                )
                normalized["modules"] = modules
                actions.append("master_plan Browser QA gate added from explicit goal requirements")
            namespace = str(task.get("protocol_node_namespace", "")).strip()
            if namespace and modules:
                aliases: dict[str, str] = {}
                for index, module in enumerate(modules, start=1):
                    raw_id = str(module.get("id", f"M{index:03d}")).strip() or f"M{index:03d}"
                    aliases[raw_id] = (
                        raw_id if raw_id.startswith(f"{namespace}.") else f"{namespace}.{raw_id}"
                    )
                for index, module in enumerate(modules, start=1):
                    raw_id = str(module.get("id", f"M{index:03d}")).strip() or f"M{index:03d}"
                    module["id"] = aliases[raw_id]
                    dependencies = list(_strings(module.get("depends_on")))
                    module["depends_on"] = [aliases.get(item, item) for item in dependencies]
                normalized["modules"] = modules
                actions.append(f"master_plan node ids isolated in run namespace {namespace}")
        if phase == InnerPhase.DECOMPOSE.value:
            parent_contract = _mapping(task.get("contract"))
            parent_id = str(parent_contract.get("id", "")).strip()
            parent_title = str(parent_contract.get("title", parent_id or "Parent module")).strip()
            raw_children = normalized.get("children", ())
            if isinstance(raw_children, Sequence) and not isinstance(raw_children, (str, bytes)):
                child_id_aliases: dict[str, str] = {}
                for index, child in enumerate(raw_children, start=1):
                    if not isinstance(child, Mapping):
                        continue
                    raw_id = str(
                        child.get("id", f"{parent_id}.{index}" if parent_id else f"child.{index}")
                    ).strip()
                    namespaced = (
                        raw_id
                        if parent_id and raw_id.startswith(f"{parent_id}.")
                        else f"{parent_id}.{index}" if parent_id else f"child.{index}"
                    )
                    child_id_aliases.setdefault(raw_id, namespaced)
                repaired_children: list[Any] = []
                for index, child in enumerate(raw_children, start=1):
                    if not isinstance(child, Mapping):
                        repaired_children.append(child)
                        continue
                    item = dict(child)
                    raw_child_id = str(
                        item.get("id", f"{parent_id}.{index}" if parent_id else f"child.{index}")
                    ).strip()
                    child_id = child_id_aliases.get(raw_child_id, raw_child_id)
                    if child_id != raw_child_id:
                        item["id"] = child_id
                        actions.append(
                            f"decompose child id {raw_child_id} namespaced as {child_id}"
                        )
                    dependencies = list(_strings(item.get("depends_on")))
                    remapped_dependencies = [
                        child_id_aliases.get(dependency, dependency)
                        for dependency in dependencies
                    ]
                    if remapped_dependencies != dependencies:
                        item["depends_on"] = remapped_dependencies
                        actions.append(f"decompose child {child_id} dependencies namespaced")
                    title = str(item.get("title", item.get("name", ""))).strip()
                    objective = str(item.get("objective", item.get("description", ""))).strip()
                    acceptance = list(_strings(item.get("acceptance_criteria")))
                    verification = list(_strings(item.get("verification")))
                    hint = str(
                        item.get("finding")
                        or item.get("reason")
                        or item.get("summary")
                        or item.get("focus")
                        or ""
                    ).strip()
                    if not title:
                        suffix = hint or child_id
                        title = f"{parent_title} refinement {index}" if "refinement" in child_id.casefold() else f"{parent_title} subtask {index}"
                        if hint and hint.casefold() not in title.casefold():
                            title = f"{title}: {hint[:80]}"
                        item["title"] = title
                        actions.append(f"decompose child {child_id} title restored")
                    if not objective:
                        objective = hint or (acceptance[0] if acceptance else "") or (verification[0] if verification else "")
                        if not objective:
                            objective = f"Advance {parent_title} within the approved contract."
                        item["objective"] = objective
                        actions.append(f"decompose child {child_id} objective restored")
                    if objective and not acceptance:
                        item["acceptance_criteria"] = [f"Child outcome is implemented: {objective}"]
                        actions.append(f"decompose child {child_id} acceptance restored")
                    if objective and not verification:
                        item["verification"] = [f"Execute or inspect the child outcome against its objective: {objective}"]
                        actions.append(f"decompose child {child_id} verification restored")
                    repaired_children.append(item)
                normalized["children"] = repaired_children
        if phase in {"review", "test", "global_review"} and not isinstance(
            normalized.get("passed"), bool
        ):
            metadata = (
                dict(task.get("contract", {}).get("metadata", {}))
                if isinstance(task.get("contract"), Mapping)
                and isinstance(task.get("contract", {}).get("metadata"), Mapping)
                else {}
            )
            component_review = bool(metadata.get("component_package_only"))
            if component_review and isinstance(normalized.get("success"), bool):
                normalized["passed"] = bool(normalized["success"])
                actions.append(
                    f"{phase}.passed derived from explicit component success verdict"
                )
            else:
                findings = (
                    *_strings(normalized.get("issues")),
                    *_strings(normalized.get("findings")),
                )
                evidence = normalized.get("evidence")
                test_results = normalized.get("test_results")
                typed_tests = tuple(
                    item
                    for item in (
                        test_results
                        if isinstance(test_results, Sequence)
                        and not isinstance(test_results, (str, bytes))
                        else ()
                    )
                    if isinstance(item, Mapping)
                )
                tests_observably_passed = bool(typed_tests) and all(
                    bool(item.get("passed")) for item in typed_tests
                )
                evidence_backed_clear = bool(evidence) or tests_observably_passed
                if component_review and not findings and evidence_backed_clear:
                    # A clean-context reviewer occasionally omits the redundant
                    # boolean while still returning typed evidence and no
                    # finding. Preserve that observable verdict; never infer a
                    # pass from prose or an empty payload.
                    normalized["passed"] = True
                    actions.append(
                        f"{phase}.passed derived from finding-free typed component evidence"
                    )
                else:
                    # Missing verdicts are never interpreted as success without
                    # typed component evidence. False routes the result through
                    # the bounded remediation loop instead of crashing.
                    normalized["passed"] = False
                    actions.append(
                        f"{phase}.passed defaulted to false for safe remediation"
                    )
        if phase in {"integrate", "global_integration", "final_evidence"} and not isinstance(
            normalized.get("success", normalized.get("passed")), bool
        ):
            normalized["success"] = False
            actions.append(f"{phase}.success defaulted to false for safe remediation")
        return normalized, tuple(actions)

    @staticmethod
    def _validate_typed_response(phase: str, payload: Mapping[str, Any]) -> None:
        """Phase-specific semantic validation shared by every agent lifecycle."""

        if phase == "goal_spec":
            GoalSpecV1.from_mapping(payload)
            return
        if phase == "architecture":
            ArchitectureSpecV1.from_mapping(payload)
            return
        if phase == "master_plan":
            MasterPlanV1.from_mapping(payload)
            return
        if phase in {"integrate", "global_integration", "final_evidence"}:
            if not isinstance(payload.get("success", payload.get("passed")), bool):
                raise AgentProtocolError(f"{phase}.success (or passed) must be boolean")
        if phase in {
            "review",
            "test",
            "global_review",
        }:
            if not isinstance(payload.get("passed"), bool):
                raise AgentProtocolError(f"{phase}.passed must be boolean")
        for key in ("evidence", "findings", "issues"):
            if key in payload and not isinstance(payload[key], (list, tuple)):
                raise AgentProtocolError(f"{phase}.{key} must be an array")

    @staticmethod
    def _validated_questions(raw: Iterable[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
        questions: list[dict[str, Any]] = []
        seen: set[str] = set()
        for index, item in enumerate(raw, start=1):
            question_id = str(item.get("id") or f"Q{index}").strip()
            text = str(item.get("question", "")).strip()
            options = tuple(
                dict(option)
                for option in item.get("options", ())
                if isinstance(option, Mapping)
            )
            combined = " ".join(
                (text, str(item.get("header", "")), str(item.get("reason", "")))
            ).casefold()
            verification_terms = (
                "verify", "verification", "read-back", "read back", "metadata",
                "test method", "check method", "successful write",
            )
            consequential_terms = (
                "platform", "target user", "product behavior", "compatibility",
                "deployment", "migration", "destructive", "irreversible",
                "public api", "interface contract", "scope boundary",
            )
            implementation_terms = (
                "enemy behavior", "enemy ai", "threat vector", "combat balance",
                "asset complexity", "input priorit", "state machine depth",
                "implementation complexity", "visual effect", "pacing",
                "particle", "shader", "shaders", "muzzle flash", "explosion",
                "environmental interactivity", "animated rails", "decorative",
                "traversable", "collision geometry", "player pathing",
                "scope creep", "physics integration",
            )
            if (
                any(term in combined for term in (*verification_terms, *implementation_terms))
                and not any(term in combined for term in consequential_terms)
            ):
                # Verification mechanics are harness policy, not product
                # decisions. Reversible implementation trade-offs are owned by
                # the architecture pass. Prefer the strongest safe local check
                # or a concrete bounded implementation without pausing.
                continue
            if not any(term in combined for term in consequential_terms):
                # Bounded, reversible product/implementation preferences are
                # not blockers.  The architecture pass owns the recommended
                # safe default so autonomous runs do not stall on optional
                # audio, touch style, polish, or feature-depth questions.
                continue
            if not question_id or question_id in seen or not text:
                raise AgentProtocolError("ULTRA questions require unique ids and non-empty text")
            if not options:
                # Open-ended questions without bounded choices are almost
                # always weak-model uncertainty, not a real user decision.
                # The architecture and implementation passes should pick a
                # safe default and continue autonomously.
                continue
            if len(options) == 1:
                continue
            seen.add(question_id)
            try:
                normalized = normalize_question(
                    {
                        "id": question_id,
                        "header": str(item.get("header", question_id)).strip()[:40],
                        "question": text[:1_000],
                        "options": options,
                        "allow_freeform": True,
                        "reason": str(
                            item.get("reason", "Required to finalize the master plan.")
                        ).strip()[:1_000],
                    },
                    index=index,
                )
            except ValueError as exc:
                raise AgentProtocolError(str(exc)) from exc
            questions.append(normalized.to_dict())
        if len(questions) > 3:
            raise AgentProtocolError("ULTRA may ask at most three questions in one foundation round")
        return tuple(questions)

    @staticmethod
    def _question_reopens_explicit_prompt_constraint(
        question: Mapping[str, Any], prompt: str
    ) -> bool:
        combined = " ".join(
            (
                str(question.get("header", "")),
                str(question.get("question", "")),
                str(question.get("reason", "")),
            )
        ).casefold()
        objective = str(prompt).casefold()
        explicit_markers = (
            "placeholder",
            "single-file",
            "single file",
            "cdn",
            "visual quality",
            "visually impressive",
            "production-quality",
            "production quality",
        )
        if any(marker in combined and marker in objective for marker in explicit_markers):
            return True
        if "aspect ratio" in combined and re.search(r"\b\d{3,5}\s*[x×]\s*\d{3,5}\b", objective):
            return True
        return False

    def _fallback_goal_spec(self, prompt: str) -> GoalSpecV1:
        """Recover the durable envelope without pretending implementation exists."""

        success = [
            "The requested behavior is implemented as a runnable artifact.",
            "Automated functional and integration checks pass with no runtime errors.",
            "Unrelated project behavior and files remain unchanged.",
            "Independent review finds no unresolved critical quality finding.",
        ]
        if self._requires_visual_artifact(prompt):
            success.extend(
                (
                    "Component and final screenshots show intentional, non-placeholder visual design.",
                    "The interactive artifact is usable through its documented controls and restart flow.",
                )
            )
        return GoalSpecV1(
            objective=prompt[:4_000],
            success_criteria=tuple(success),
            constraints=(
                "Use bounded recursive specialists with isolated contracts and evidence.",
                "Only the FinalAssembler owns final output paths.",
                "Persist checkpoints, findings, packages, and lessons outside model context.",
            ),
            in_scope=(
                "intent expansion",
                "implementation",
                "component verification",
                "integration",
                "independent quality evaluation",
            ),
            out_of_scope=(
                "unrequested changes to unrelated project behavior",
                "accepting prose or placeholder artifacts as completion",
            ),
            assumptions=(
                "Reversible technical choices may be selected by the architecture harness.",
            ),
            questions=(),
        )

    def _fallback_architecture(
        self,
        prompt: str,
        candidate_index: int,
    ) -> ArchitectureSpecV1:
        """Recover protocol shape without asking the builder to solve the whole task.

        The local model still implements every component.  This fallback only
        supplies the inspectable component/interface envelope that a malformed
        architecture response failed to express.
        """

        assert self.goal_spec
        if self._requires_visual_artifact(prompt):
            components = (
                ("World", "Road, environment, lighting, depth, and scene composition."),
                ("Vehicles", "Chassis, wheels, cabin, glass, lights, materials, and variants."),
                ("Character", "Readable body, animation states, controls, and feedback."),
                ("Gameplay", "Traffic, collision, scoring, progression, and restart state."),
                ("Presentation", "Camera, HUD, audio, effects, responsiveness, and accessibility."),
                ("QA", "Functional, visual, performance, browser, and regression evidence."),
            )
            return ArchitectureSpecV1(
                summary=(
                    "Artifact-first recursive specialist architecture."
                    if candidate_index % 2
                    else "State-contract recursive specialist architecture."
                ),
                components=tuple(
                    {
                        "name": name,
                        "responsibility": responsibility,
                        "artifact_boundary": f"{name}Package",
                        "independent_preview": True,
                    }
                    for name, responsibility in components
                ),
                interfaces=tuple(
                    {
                        "name": f"{name}Package",
                        "producer": name,
                        "consumer": "FinalAssembler",
                        "requires": ["implementation", "preview", "tests", "evidence"],
                    }
                    for name, _responsibility in components
                ),
                decisions=(
                    {
                        "decision": "FinalAssembler owns final output paths.",
                        "reason": "Specialists remain isolated and their exact packages are consumed.",
                    },
                    {
                        "decision": "Every visual leaf has an independently runnable preview.",
                        "reason": "Weak-model output is evaluated before integration.",
                    },
                ),
                dependencies=(
                    "Gameplay consumes World/Vehicle/Character contracts.",
                    "Presentation observes Gameplay state.",
                    "QA evaluates component and integrated artifacts.",
                ),
                invariants=(
                    "No specialist writes the final artifact.",
                    "No placeholder or documentation proxy can satisfy a visual package.",
                    "Integration preserves accepted package hashes and interfaces.",
                ),
            )
        scoped = tuple(self.goal_spec.in_scope) or ("implementation", "verification")
        return ArchitectureSpecV1(
            summary="Contract-first recursive delivery architecture.",
            components=tuple(
                {
                    "name": f"Scope{index}",
                    "responsibility": item,
                    "artifact_boundary": f"Scope{index}Package",
                }
                for index, item in enumerate(scoped[:8], start=1)
            ),
            interfaces=(
                {
                    "name": "IntegratedDelivery",
                    "producer": "specialist packages",
                    "consumer": "FinalAssembler",
                },
            ),
            decisions=(
                {
                    "decision": "Split by independently testable responsibility.",
                    "reason": "Keep each local-model context bounded.",
                },
            ),
            invariants=("Every component has observable acceptance evidence.",),
        )

    def _fallback_master_plan(self, prompt: str) -> MasterPlanV1:
        assert self.goal_spec
        paths = self._final_output_paths(prompt)
        return MasterPlanV1(
            summary="Harness-recovered goal plan with recursive specialist execution.",
            modules=(
                TaskContractV1(
                    id=f"r{self.run_state.id[-12:]}.M001",
                    title="Goal delivery and final assembly",
                    objective=self.goal_spec.objective,
                    acceptance_criteria=self.goal_spec.success_criteria,
                    verification=tuple(
                        f"Verify observable criterion: {criterion}"
                        for criterion in self.goal_spec.success_criteria
                    ),
                    write_paths=paths,
                    owned_interfaces=("IntegratedDelivery",),
                    metadata={"force_recursive_specialists": True},
                ),
            ),
            milestones=(
                {"name": "materialized specialists", "evidence": "component packages"},
                {"name": "integration", "evidence": "runtime and tests"},
                {"name": "acceptance", "evidence": "final quality gate"},
            ),
            execution_strategy=(
                "Build bounded specialists, verify each package, assemble exact accepted "
                "artifacts, then run adversarial and final evidence gates."
            ),
        )

    @staticmethod
    def _compact_goal_for_model(goal: GoalSpecV1) -> Mapping[str, Any]:
        """Project the durable north star into a small foundation packet."""

        return {
            "objective": goal.objective[:1_200],
            "success_criteria": list(goal.success_criteria[:8]),
            "constraints": list(goal.constraints[:8]),
            "in_scope": list(goal.in_scope[:8]),
            "out_of_scope": list(goal.out_of_scope[:6]),
            "assumptions": list(goal.assumptions[:6]),
        }

    @staticmethod
    def _compact_architecture_for_model(
        architecture: ArchitectureSpecV1,
    ) -> Mapping[str, Any]:
        return {
            "summary": architecture.summary[:1_200],
            "components": [
                {
                    key: item.get(key)
                    for key in ("name", "responsibility", "artifact_boundary")
                    if item.get(key) not in (None, "", (), [], {})
                }
                for item in architecture.components[:12]
            ],
            "interfaces": [
                {
                    key: item.get(key)
                    for key in ("name", "producer", "consumer", "contract")
                    if item.get(key) not in (None, "", (), [], {})
                }
                for item in architecture.interfaces[:12]
            ],
            "invariants": list(architecture.invariants[:10]),
        }

    def _finish_foundation(self, prompt: str) -> MasterPlanV1:
        """Continue Architecture -> Master Plan after goal decisions are complete."""

        assert self.run_state and self.goal_spec
        self._set_phase(UltraPhase.ARCHITECTURE, "Designing architecture")
        lesson_context = self._foundation_project_lessons(prompt, "architecture")
        candidate_count = 3 if any(
            marker in prompt.casefold()
            for marker in ("migration", "security", "production", "destructive", "ترحيل", "أمان")
        ) else 2
        candidate_specs: list[ArchitectureSpecV1] = []
        candidate_payloads: list[Mapping[str, Any]] = []
        for candidate_index in range(1, candidate_count + 1):
            try:
                architecture_response = self._invoke(
                    AgentRole.ARCHITECT,
                    "architecture",
                    task={
                        "goal_spec": self._compact_goal_for_model(self.goal_spec),
                        "candidate_index": candidate_index,
                        "candidate_count": candidate_count,
                        "instruction": (
                            "Produce an architecture materially different from the other candidates; "
                            "optimize quality, specialist isolation, integration contracts, and verification."
                        ),
                    },
                    context={
                        "prompt": prompt,
                        "cross_run_project_lessons": lesson_context,
                    },
                )
                candidate = ArchitectureSpecV1.from_mapping(architecture_response.payload)
            except AgentProtocolError as exc:
                candidate = self._fallback_architecture(prompt, candidate_index)
                self.events.publish(
                    "ultra.architecture_protocol_recovered",
                    (
                        f"Architecture candidate {candidate_index} used the typed harness "
                        "fallback after malformed local-model output."
                    ),
                    run_id=self.run_state.id,
                    candidate_index=candidate_index,
                    error=str(exc),
                )
            candidate_specs.append(candidate)
            candidate_payloads.append(asdict(candidate))
        compact_candidates = [
            self._compact_architecture_for_model(candidate)
            for candidate in candidate_specs
        ]
        try:
            critic = self._invoke(
                AgentRole.REVIEWER,
                "architecture_critique",
                task={
                    "goal_spec": self._compact_goal_for_model(self.goal_spec),
                    "candidates": compact_candidates,
                    "instruction": "Identify omissions, integration risks, weak-model failure modes, and unverifiable claims for every candidate.",
                },
                context={"prompt": prompt, "clean_context": True},
            )
            critic_payload = dict(critic.payload)
        except AgentProtocolError as exc:
            critic_payload = {
                "verdict": "harness_recovery",
                "risks": [
                    "Validate every specialist interface before integration.",
                    "Reject packages without runnable evidence.",
                ],
                "protocol_error": str(exc),
            }
        try:
            judge = self._invoke(
                AgentRole.GOAL_CHECKER,
                "architecture_judge",
                task={
                    "goal_spec": self._compact_goal_for_model(self.goal_spec),
                    "candidates": compact_candidates,
                    "critic_verdict": critic_payload,
                    "instruction": "Select the strongest candidate or return a synthesized architecture and cite observable reasons.",
                },
                context={"prompt": prompt, "clean_context": True},
            )
            judge_payload = dict(judge.payload)
        except AgentProtocolError as exc:
            judge_payload = {
                "selected_index": 1,
                "verdict": "harness_recovery",
                "protocol_error": str(exc),
            }
        try:
            selected_index = int(judge_payload.get("selected_index", 1) or 1)
        except (TypeError, ValueError):
            selected_index = 1
        selected_index = min(max(1, selected_index), len(candidate_specs))
        synthesized = judge_payload.get("architecture")
        self.architecture = (
            ArchitectureSpecV1.from_mapping({"architecture": synthesized})
            if isinstance(synthesized, Mapping)
            else candidate_specs[selected_index - 1]
        )
        self.state.append_brain_entry(
            BrainEntryV1(
                BrainSection.DECISION,
                "architecture_debate",
                {
                    "candidate_count": candidate_count,
                    "candidates": candidate_payloads,
                    "critic": critic_payload,
                    "judge": judge_payload,
                    "selected_index": selected_index,
                    "selected_summary": self.architecture.summary,
                },
                self.run_state.id,
            )
        )
        self._set_phase(UltraPhase.MASTER_PLAN, "Building master plan")
        try:
            plan_response = self._invoke(
                AgentRole.PLANNER,
                "master_plan",
                task={
                    "goal_spec": self._compact_goal_for_model(self.goal_spec),
                    "architecture": self._compact_architecture_for_model(
                        self.architecture
                    ),
                    "module_bounds": {
                        "preferred_min": self.config.min_top_modules,
                        "maximum": self.config.max_top_modules,
                    },
                },
                context={
                    "prompt": prompt,
                    "cross_run_project_lessons": self._foundation_project_lessons(
                        f"{prompt} {self.architecture.summary}",
                        "master_plan",
                    ),
                },
            )
            proposed_raw = MasterPlanV1.from_mapping(plan_response.payload)
        except AgentProtocolError as exc:
            proposed_raw = self._fallback_master_plan(prompt)
            self.events.publish(
                "ultra.master_plan_protocol_recovered",
                "Master plan used the typed harness fallback after malformed local-model output.",
                run_id=self.run_state.id,
                error=str(exc),
            )
        proposed = self._enforce_master_artifact_contract(prompt, proposed_raw)
        proposed = self._enforce_concern_coverage_contract(prompt, proposed)
        quality_checklist = (
            "\n\nULTRA Quality Checklist (approval-bound): clean-code review; security review; "
            "tests and test-quality review; remediation Change Sets receive fresh reviews; "
            "integration and global evidence require zero open Critical, High, or Medium findings."
        )
        proposed = MasterPlanV1(
            summary=proposed.summary,
            modules=proposed.modules,
            milestones=_with_quality_milestone(proposed.milestones),
            execution_strategy=proposed.execution_strategy.rstrip() + quality_checklist,
            revision=proposed.revision,
        )
        answers = _mapping(self.run_state.metadata.get("question_answers"))
        if answers:
            strategy = proposed.execution_strategy.rstrip()
            strategy += "\n\nApproval-bound goal decisions: " + _json(answers)
            proposed = MasterPlanV1(
                summary=proposed.summary,
                modules=proposed.modules,
                milestones=proposed.milestones,
                execution_strategy=strategy,
                revision=proposed.revision,
            )
        self.master_plan = proposed
        if len(self.master_plan.modules) > self.config.max_top_modules:
            raise AgentProtocolError(
                f"master plan exceeds {self.config.max_top_modules} top-level modules"
            )
        self.nodes.clear()
        self._prepared.clear()
        self._research_required.clear()
        self._results.clear()
        self._order = 0
        for contract in self.master_plan.modules:
            self._order += 1
            node = WorkNode(
                contract=contract,
                order=self._order,
                pre_write_hashes=_mapping(contract.metadata.get("pre_write_hashes")),
            )
            self.nodes[node.id] = node
            self.state.save_work_node(self.run_state.id, node)
        self.state.append_brain_entry(
            BrainEntryV1(
                BrainSection.NORTH_STAR,
                "goal",
                asdict(self.goal_spec),
                self.run_state.id,
            )
        )
        self.state.append_brain_entry(
            BrainEntryV1(
                BrainSection.ARCHITECTURE,
                "approved_candidate",
                asdict(self.architecture),
                self.run_state.id,
            )
        )
        self._save_run(master_fingerprint=self.master_plan.fingerprint)
        self._set_phase(UltraPhase.AWAITING_APPROVAL, "Master plan awaits approval")
        self.events.publish(
            UltraEventKind.FOUNDATION_READY.value,
            f"Architecture ready · {len(self.master_plan.modules)} modules",
            run_id=self.run_state.id,
            modules=len(self.master_plan.modules),
            fingerprint=self.master_plan.fingerprint,
        )
        return self.master_plan

    def prepare(self, prompt: str) -> MasterPlanV1 | None:
        """Build GoalSpec -> Architecture -> MasterPlan, then await approval."""

        prompt = str(prompt).strip()
        if not prompt:
            raise ValueError("ULTRA prompt must not be empty")
        if self.run_state and self.phase not in {
            UltraPhase.CANCELLED,
            UltraPhase.COMPLETED,
            UltraPhase.FAILED,
        }:
            raise UltraError("an ULTRA run is already active")
        concurrency = 1 if self.execution_class is ExecutionClass.LOCAL else self.adaptive.current
        self.run_state = UltraRunV1(
            prompt=prompt,
            execution_class=self.execution_class,
            concurrency=concurrency,
            phase=UltraPhase.GOAL_SPEC,
            model_snapshot=self.model_snapshot,
            config_snapshot=asdict(self.config),
        )
        self.state.save_ultra_run(self.run_state)
        try:
            try:
                goal_response = self._invoke(
                    AgentRole.GOAL_UNDERSTANDING,
                    "goal_spec",
                    task={"prompt": prompt},
                    context={
                        "instruction": (
                            "Inspect the repository first. Derive GoalSpecV1 and ask at most three "
                            "questions only for high-impact decisions that cannot be discovered."
                        ),
                        "cross_run_project_lessons": self._foundation_project_lessons(prompt, "goal_spec"),
                    },
                )
                self.goal_spec = self._enforce_goal_artifact_contract(
                    prompt,
                    GoalSpecV1.from_mapping(goal_response.payload),
                )
            except AgentProtocolError as exc:
                self.goal_spec = self._enforce_goal_artifact_contract(
                    prompt,
                    self._fallback_goal_spec(prompt),
                )
                self.events.publish(
                    "ultra.goal_spec_protocol_recovered",
                    "GoalSpec used the typed harness fallback after malformed local-model output.",
                    run_id=self.run_state.id,
                    error=str(exc),
                )
            raw_questions = tuple(self.goal_spec.questions)
            questions = self._validated_questions(raw_questions)
            questions = tuple(
                item
                for item in questions
                if not self._question_reopens_explicit_prompt_constraint(item, prompt)
            )
            if len(questions) < len(raw_questions):
                self.events.publish(
                    "ultra.questions_autoresolved",
                    f"Resolved {len(raw_questions) - len(questions)} verification-policy question(s) by harness policy",
                    run_id=self.run_state.id,
                    removed=len(raw_questions) - len(questions),
                )
            if questions:
                self.goal_spec = replace(self.goal_spec, questions=questions)
                self._save_run(
                    metadata={
                        **self.run_state.metadata,
                        "pending_questions": list(questions),
                    }
                )
                self._set_phase(UltraPhase.AWAITING_QUESTIONS, "Goal decisions need user input")
                self.events.publish(
                    "ultra.questions",
                    f"{len(questions)} goal decision(s) await an answer",
                    run_id=self.run_state.id,
                    questions=list(questions),
                )
                return None
            # Do not retain model-proposed questions that harness policy has
            # explicitly auto-resolved; callers use GoalSpec.questions as the
            # authoritative live checkpoint state.
            self.goal_spec = replace(self.goal_spec, questions=())
            return self._finish_foundation(prompt)
        except CancellationRequested:
            self._set_phase(UltraPhase.CANCELLED, "ULTRA foundation cancelled")
            raise
        except Exception:
            self._set_phase(UltraPhase.FAILED, "ULTRA foundation failed")
            raise

    def answer_questions(self, answers: Mapping[str, Any]) -> MasterPlanV1:
        """Bind answers into the master fingerprint and continue foundation work."""

        if (
            self.phase is not UltraPhase.AWAITING_QUESTIONS
            or not self.goal_spec
            or not self.run_state
        ):
            raise UltraError("this ULTRA run is not waiting for goal questions")
        questions = {str(item["id"]): item for item in self.goal_spec.questions}
        normalized: dict[str, str] = {}
        for key, value in answers.items():
            question_id = str(key)
            if question_id not in questions:
                normalized[question_id] = str(value).strip()
                continue
            question = normalize_question(questions[question_id])
            normalized[question_id] = answer_from_value(question, str(value))[0]
        missing = set(questions) - {key for key, value in normalized.items() if value}
        unknown = set(normalized) - set(questions)
        if unknown:
            raise AgentProtocolError(f"unknown ULTRA question ids: {sorted(unknown)}")
        if missing:
            raise ApprovalRequiredError(f"answer every pending question first: {sorted(missing)}")
        for question_id, value in normalized.items():
            item = questions[question_id]
            labels = {
                str(option.get("label", "")).strip()
                for option in item.get("options", ())
            }
            if labels and value not in labels and not item.get("allow_freeform", True):
                raise AgentProtocolError(
                    f"answer for {question_id} must be one of: {', '.join(sorted(labels))}"
                )
        assumptions = tuple(
            dict.fromkeys(
                (
                    *self.goal_spec.assumptions,
                    *(f"{key}: {value}" for key, value in normalized.items()),
                )
            )
        )
        self.goal_spec = replace(self.goal_spec, questions=(), assumptions=assumptions)
        self._save_run(
            metadata={
                **self.run_state.metadata,
                "pending_questions": [],
                "question_answers": normalized,
            }
        )
        return self._finish_foundation(self.run_state.prompt)

    def approve(self, expected_fingerprint: str | None = None) -> MasterPlanV1:
        if self.phase is not UltraPhase.AWAITING_APPROVAL or not self.master_plan:
            raise ApprovalRequiredError("there is no pending ULTRA master plan")
        if expected_fingerprint and expected_fingerprint != self.master_plan.fingerprint:
            raise ApprovalMismatchError("master plan fingerprint changed; review the latest revision")
        self._save_run(approved=True, master_fingerprint=self.master_plan.fingerprint)
        self.events.publish(
            UltraEventKind.APPROVED.value,
            "Master plan approved",
            run_id=self.run_state.id,
            fingerprint=self.master_plan.fingerprint,
        )
        return self.master_plan

    @staticmethod
    def _requires_single_html_artifact(prompt: str) -> bool:
        text = str(prompt).casefold()
        return bool(re.search(r"\bsingle[- ]file\b.{0,80}\bhtml\b", text)) or any(
            marker in text
            for marker in (
                "single self-contained html",
                "single html",
                "single-file html",
                "one html file",
                "one file",
                "ملف واحد",
            )
        )

    @staticmethod
    def _task_family(value: str) -> str:
        text = str(value).casefold()
        if any(
            marker in text
            for marker in (
                "three.js", "threejs", "webgl", "game", "gameplay", "crossy",
                "لعبة", "جيم", "ثري.js",
            )
        ):
            return "interactive_game"
        if any(
            marker in text
            for marker in (
                "machine learning", "ml project", "training pipeline", "churn",
                "classification model", "regression model", "تعلم آلي",
            )
        ):
            return "ml"
        if any(
            marker in text
            for marker in (
                "backend", "back-end", "rest api", "graphql", "appointment",
                "booking api", "microservice", "خلفية", "واجهة برمجة",
            )
        ):
            return "backend"
        if any(
            marker in text
            for marker in (
                "frontend", "front-end", "dashboard", "web app", "landing page",
                "react", "vue", "واجهة", "لوحة تحكم",
            )
        ):
            return "frontend"
        return "general"

    @classmethod
    def _requires_game_artifact(cls, prompt: str) -> bool:
        return cls._task_family(prompt) == "interactive_game"

    @classmethod
    def _concern_coverage_matrix(cls, value: str) -> ConcernCoverageMatrixV1:
        """Derive concern ownership needs from semantics, never prompt length."""

        family = cls._task_family(value)

        def concern(
            id: str,
            title: str,
            objective: str,
            acceptance: str,
            verification: str,
            owner: str,
            *,
            critical: bool = True,
        ) -> ConcernRequirementV1:
            return ConcernRequirementV1(
                id=id,
                title=title,
                objective=objective,
                acceptance_criteria=(acceptance,),
                verification=(verification,),
                owner_hint=owner,
                critical=critical,
            )

        common = (
            concern(
                "functional_correctness", "Functional correctness",
                "Preserve explicit invariants and complete the requested behavior.",
                "Core happy paths and consequential edge cases match the approved brief.",
                "Run deterministic behavior checks that assert outputs and state transitions.",
                "implementation",
            ),
            concern(
                "integration_contracts", "Integration contracts",
                "Keep component interfaces, data shapes, and lifecycle ordering coherent.",
                "All consumed interfaces are compatible and no accepted component is reinvented.",
                "Run contract and end-to-end integration checks against materialized artifacts.",
                "integration",
            ),
            concern(
                "regression_safety", "Regression safety",
                "Protect unrelated behavior and repository boundaries while changing the target.",
                "Existing relevant behavior remains green and changes stay inside approved scope.",
                "Run focused regression tests and inspect the final diff and dependency surface.",
                "verification",
            ),
        )
        families: dict[str, tuple[ConcernRequirementV1, ...]] = {
            "interactive_game": (
                concern(
                    "spatial_semantics", "Spatial and input semantics",
                    "Keep facing, movement intent, camera axes, lane direction, and animation coherent.",
                    "Every moving actor faces and animates consistently with its actual movement direction.",
                    "Exercise every input direction and assert pose, heading, displacement, and camera response.",
                    "character_controls",
                ),
                concern(
                    "gameplay_state", "Gameplay state machine",
                    "Define deterministic start, play, collision, win, game-over, and restart transitions.",
                    "No state can become stuck, skip required feedback, or retain stale round data.",
                    "Run a full playable session covering collision, success, restart, and repeated rounds.",
                    "gameplay_state",
                ),
                concern(
                    "progression_pacing", "Progression and pacing",
                    "Create a fair, measurable difficulty curve and meaningful replay progression.",
                    "Success advances difficulty through bounded speed, density, timing, or layout changes.",
                    "Compare deterministic early and later rounds and assert increasing challenge without impossible states.",
                    "gameplay_progression",
                ),
                concern(
                    "world_scale", "World scale and continuity",
                    "Size and extend the traversable world to support the intended session and progression.",
                    "The world does not visibly end or exhaust meaningful traversal during normal play.",
                    "Play beyond the initial viewport and inspect segment generation, cleanup, and collision continuity.",
                    "world",
                ),
                concern(
                    "runtime_performance", "Runtime performance",
                    "Bound rendering, update, allocation, and entity lifecycle costs.",
                    "Frame pacing remains stable during a representative late-game load.",
                    "Measure frame/update time, entity counts, and memory across an extended session.",
                    "qa_performance",
                ),
                concern(
                    "interactive_accessibility", "Interactive accessibility",
                    "Provide readable feedback, keyboard/touch reachability, and reduced-motion-safe behavior.",
                    "Core play and status feedback remain usable across supported input and viewport modes.",
                    "Test keyboard, touch-sized controls, responsive layout, focus, and reduced motion.",
                    "presentation",
                    critical=False,
                ),
            ),
            "backend": (
                concern(
                    "security_boundaries", "Security boundaries",
                    "Enforce authentication, authorization, validation, secret handling, and abuse controls.",
                    "No trust boundary accepts unauthorized or unvalidated operations.",
                    "Run authorization, injection, malformed-input, rate-limit, and secret-leak tests.",
                    "auth",
                ),
                concern(
                    "data_integrity", "Data integrity",
                    "Preserve transactional invariants, migrations, and durable consistency.",
                    "Concurrent and failed operations cannot corrupt or partially commit domain state.",
                    "Run rollback, migration, constraint, and persistence recovery tests.",
                    "persistence",
                ),
                concern(
                    "concurrency_idempotency", "Concurrency and idempotency",
                    "Handle races, retries, duplicate delivery, and ordering explicitly.",
                    "Repeated or concurrent requests have deterministic safe outcomes.",
                    "Run race, duplicate-request, retry, and transaction-boundary tests.",
                    "persistence",
                ),
                concern(
                    "failure_recovery", "Failure and recovery",
                    "Expose stable errors and recover safely from dependency failures and partial work.",
                    "Failure paths are bounded, observable, and do not leak inconsistent state.",
                    "Inject dependency, timeout, malformed-input, and restart failures.",
                    "api",
                ),
                concern(
                    "operability", "Performance and operability",
                    "Meet latency/resource budgets with actionable logs, metrics, and health signals.",
                    "Representative load stays within budget and operational failures are diagnosable.",
                    "Run bounded load checks and inspect logs, metrics, health, and trace correlation.",
                    "operations",
                ),
            ),
            "frontend": (
                concern(
                    "ui_state_integrity", "UI state integrity",
                    "Handle loading, empty, error, stale, optimistic, and recovered states coherently.",
                    "Every async path has a usable state and cannot display contradictory data.",
                    "Exercise delayed, empty, failed, stale, and recovered data fixtures.",
                    "data",
                ),
                concern(
                    "frontend_accessibility", "Frontend accessibility",
                    "Provide semantic, keyboard, focus, contrast, and motion-safe interaction.",
                    "Core journeys pass keyboard and automated accessibility checks.",
                    "Run keyboard journeys, focus assertions, semantics audit, contrast, and reduced-motion checks.",
                    "accessibility",
                ),
                concern(
                    "frontend_security", "Frontend security",
                    "Prevent unsafe rendering, token leakage, untrusted navigation, and insecure client assumptions.",
                    "Untrusted content cannot execute and sensitive state is not exposed.",
                    "Run XSS fixtures, URL validation, storage inspection, and authorization-boundary checks.",
                    "quality",
                ),
                concern(
                    "frontend_performance", "Frontend performance",
                    "Bound initial load, rendering churn, asset cost, and long-task behavior.",
                    "Representative pages remain responsive within explicit performance budgets.",
                    "Measure bundle/load behavior, long tasks, rendering, and representative interactions.",
                    "quality",
                ),
                concern(
                    "visual_responsiveness", "Visual and responsive quality",
                    "Maintain hierarchy, readability, and interaction across supported viewports.",
                    "Critical layouts have no clipping, overlap, unreadable states, or visual regressions.",
                    "Capture and compare deterministic desktop/mobile screenshots and interaction states.",
                    "visual_qa",
                ),
            ),
            "ml": (
                concern(
                    "data_leakage", "Data leakage prevention",
                    "Keep target, future, identity, and split leakage out of features and evaluation.",
                    "Training and evaluation separation is provable at data and transformation boundaries.",
                    "Run schema, temporal split, duplicate/entity overlap, and target-proxy checks.",
                    "data",
                ),
                concern(
                    "ml_reproducibility", "ML reproducibility",
                    "Version inputs/configuration and make training outcomes reproducible within tolerance.",
                    "A clean rerun records lineage and reproduces metrics within declared tolerance.",
                    "Repeat training from pinned inputs and compare artifacts, metrics, and lineage.",
                    "training",
                ),
                concern(
                    "evaluation_validity", "Evaluation validity",
                    "Use relevant baselines, metrics, slices, calibration, and robustness evidence.",
                    "Model acceptance is supported by baseline and slice results, not a single aggregate metric.",
                    "Run baseline, holdout, slice, calibration, robustness, and threshold checks.",
                    "evaluation",
                ),
                concern(
                    "ml_serving_reliability", "Serving reliability",
                    "Keep training-serving schemas aligned with bounded latency and drift observability.",
                    "Serving rejects invalid inputs and exposes latency, version, and drift signals.",
                    "Run schema parity, invalid-input, latency, versioning, and drift-monitor fixtures.",
                    "serving",
                ),
            ),
            "general": (
                concern(
                    "failure_recovery", "Failure and recovery",
                    "Handle consequential failures without corrupting state or hiding partial work.",
                    "Failure paths are explicit, recoverable where appropriate, and observable.",
                    "Exercise boundary failures, interruption, retry, and restart behavior.",
                    "integration",
                ),
                concern(
                    "risk_security_performance", "Security and performance risks",
                    "Discover and test relevant trust-boundary and resource risks for this task.",
                    "No material security or performance risk remains unassessed.",
                    "Run a task-specific threat/resource review and its highest-risk executable checks.",
                    "quality_risk",
                ),
            ),
        }
        return ConcernCoverageMatrixV1(task_family=family, concerns=(*common, *families[family]))

    @staticmethod
    def _requires_visual_artifact(prompt: str) -> bool:
        text = str(prompt).casefold()
        return any(
            marker in text
            for marker in (
                "three.js", "threejs", "webgl", "3d", "game", "visual",
                "dashboard", "landing page", "لعبة", "واجهة", "تصميم",
            )
        )

    @classmethod
    def _final_output_paths(cls, prompt: str) -> tuple[str, ...]:
        text = str(prompt).casefold()
        if cls._requires_single_html_artifact(prompt):
            return ("index.html",)
        if any(
            marker in text
            for marker in ("multi-file", "multi file", "multi-file project", "ملفات متعددة")
        ):
            return ("index.html", "src")
        return ("index.html", "components")

    @classmethod
    def _enforce_goal_artifact_contract(
        cls,
        prompt: str,
        goal: GoalSpecV1,
    ) -> GoalSpecV1:
        if not cls._requires_visual_artifact(prompt):
            return goal
        final_paths = cls._final_output_paths(prompt)
        if not cls._requires_game_artifact(prompt):
            return replace(
                goal,
                constraints=tuple(
                    dict.fromkeys(
                        (
                            *goal.constraints,
                            f"The final deliverable uses approved packaging paths: {', '.join(final_paths)}.",
                            "Visual acceptance requires real browser evidence at supported viewports.",
                        )
                    )
                ),
                in_scope=tuple(dict.fromkeys((*goal.in_scope, *final_paths, "browser runtime"))),
                success_criteria=tuple(
                    dict.fromkeys(
                        (
                            *goal.success_criteria,
                            "The browser runtime has zero console, page, or network errors.",
                            "Responsive, accessibility, async-state, security, and performance checks pass.",
                        )
                    )
                ),
            )
        return replace(
            goal,
            constraints=tuple(
                dict.fromkeys(
                    (
                        *goal.constraints,
                        f"The final deliverable uses approved packaging paths: {', '.join(final_paths)}.",
                        "Only FinalAssembler may write final output paths; specialists publish materialized component packages.",
                    )
                )
            ),
            in_scope=tuple(dict.fromkeys((*goal.in_scope, *final_paths, "playable browser runtime"))),
            success_criteria=tuple(
                dict.fromkeys(
                    (
                        *goal.success_criteria,
                        "index.html runs with zero console, page, network, or WebGL errors",
                        "keyboard controls, gameplay loop, collision, scoring, and restart are playable",
                        "overall quality >= 0.95 and critical visual dimensions >= 0.90",
                    )
                )
            ),
        )

    @classmethod
    def _enforce_concern_coverage_contract(
        cls,
        prompt: str,
        proposed: MasterPlanV1,
    ) -> MasterPlanV1:
        semantic_source = " ".join(
            (
                prompt,
                proposed.summary,
                *(f"{module.title} {module.objective}" for module in proposed.modules),
            )
        )
        matrix = cls._concern_coverage_matrix(semantic_source)
        modules = list(proposed.modules)
        assignments: dict[int, list[ConcernRequirementV1]] = defaultdict(list)
        owner_terms: Mapping[str, tuple[str, ...]] = {
            "implementation": ("implementation", "domain", "model", "component", "gameplay"),
            "integration": ("integration", "test", "evaluation", "qa", "component"),
            "verification": ("test", "evaluation", "qa", "verification", "visual"),
            "world": ("world", "road", "environment"),
            "character_controls": ("character", "control", "input", "movement"),
            "gameplay_state": ("gameplay", "state", "logic", "progression"),
            "gameplay_progression": ("progression", "difficulty", "traffic", "gameplay"),
            "qa_performance": ("performance", "qa", "test"),
            "presentation": ("presentation", "accessibility", "hud", "ui"),
            "auth": ("auth", "security"),
            "persistence": ("persistence", "database", "storage", "repository"),
            "api": ("api", "http", "interface"),
            "operations": ("operation", "observability", "performance", "deploy"),
            "data": ("data", "feature"),
            "training": ("training", "pipeline"),
            "evaluation": ("evaluation", "metric", "test"),
            "serving": ("serving", "inference", "deploy", "api"),
            "accessibility": ("accessibility", "a11y"),
            "quality": ("quality", "security", "performance"),
            "visual_qa": ("visual", "qa", "test"),
            "quality_risk": ("quality", "risk", "security", "performance"),
        }
        searchable = [
            " ".join((module.title, module.objective)).casefold()
            for module in modules
        ]
        for position, item in enumerate(matrix.concerns):
            terms = owner_terms.get(item.owner_hint, (item.owner_hint,))
            owner_index = next(
                (
                    index
                    for index, text in enumerate(searchable)
                    if any(term in text for term in terms)
                ),
                # Verification/integration concerns naturally belong to the
                # last approved module; other unmatched concerns are spread
                # deterministically instead of duplicating an entire swarm.
                len(modules) - 1
                if item.owner_hint in {"verification", "integration", "qa_performance"}
                else position % len(modules),
            )
            assignments[owner_index].append(item)

        enriched: list[TaskContractV1] = []
        for index, module in enumerate(modules):
            owned = assignments.get(index, [])
            enriched.append(
                replace(
                    module,
                    acceptance_criteria=tuple(
                        dict.fromkeys(
                            (
                                *module.acceptance_criteria,
                                *(criterion for item in owned for criterion in item.acceptance_criteria),
                            )
                        )
                    ),
                    verification=tuple(
                        dict.fromkeys(
                            (
                                *module.verification,
                                *(check for item in owned for check in item.verification),
                            )
                        )
                    ),
                    metadata={
                        **dict(module.metadata),
                        "task_family": matrix.task_family,
                        "concern_source": semantic_source[:4_000],
                        "concern_coverage_matrix": asdict(matrix),
                        "critical_concern_ids": list(matrix.critical_ids),
                        "concern_ids": [item.id for item in owned],
                        "concern_contracts": [asdict(item) for item in owned],
                        "require_complete_concern_ownership": True,
                        "cross_domain_template_root": len(modules) == 1,
                    },
                )
            )
        return replace(
            proposed,
            modules=tuple(enriched),
            execution_strategy=(
                proposed.execution_strategy.rstrip()
                + "\nHarness invariant: every inferred critical concern must have a named "
                "specialist owner and executable verification evidence before acceptance."
            ),
        )

    @classmethod
    def _enforce_master_artifact_contract(
        cls,
        prompt: str,
        proposed: MasterPlanV1,
    ) -> MasterPlanV1:
        if not cls._requires_game_artifact(prompt):
            return proposed
        inherited = proposed.modules[0]
        final_paths = cls._final_output_paths(prompt)
        final_module = TaskContractV1(
            id=inherited.id,
            title="FinalAssembler for the Three.js vehicle game",
            objective=(
                "Build a polished playable 3D vehicle game and compose the World, Vehicles, Character, "
                "Gameplay, Presentation, and QA component packages into one self-contained index.html."
            ),
            acceptance_criteria=(
                "index.html contains a visibly modeled road/world, detailed vehicles, and readable character",
                "Keyboard input drives a complete collision, scoring/progression, game-over, and restart loop",
                "Camera, HUD, lighting, effects, responsiveness, and accessibility form one cohesive presentation",
                "Browser, WebGL, functional, visual, and performance evidence passes with no runtime errors",
            ),
            verification=(
                "Run index.html in Playwright and inspect console, page, network, WebGL, and input state",
                "Capture staged and final screenshots and score the visual quality rubric",
                "Verify overall score >= 0.95 and every critical visual score >= 0.90",
            ),
            depends_on=(),
            write_paths=final_paths,
            forbidden_changes=inherited.forbidden_changes,
            owned_interfaces=tuple(
                dict.fromkeys(
                    (
                        *inherited.owned_interfaces,
                        "WorldPackage",
                        "VehiclePackage",
                        "CharacterPackage",
                        "GameplayPackage",
                        "PresentationPackage",
                        "QAPackage",
                    )
                )
            ),
            metadata={
                **dict(inherited.metadata),
                "force_recursive_specialists": True,
                "final_output_paths": list(final_paths),
                "materialized_components_required": True,
                "packaging": (
                    "single_html"
                    if len(final_paths) == 1
                    else ("multi_file" if "src" in final_paths else "modular_best_final")
                ),
                "source_module_count": len(proposed.modules),
            },
        )
        return MasterPlanV1(
            summary=proposed.summary + " Harness-enforced materialized recursive assembly.",
            modules=(final_module,),
            milestones=proposed.milestones,
            execution_strategy=(
                proposed.execution_strategy.rstrip()
                + "\nHarness invariant: specialists publish isolated materialized packages; FinalAssembler alone writes final outputs."
            ),
            revision=proposed.revision,
        )

    @staticmethod
    def _contains_scope(parent: str, child: str) -> bool:
        p = parent.replace("\\", "/").rstrip("/") or "."
        c = child.replace("\\", "/").rstrip("/") or "."
        if p in {".", "*", "**", "**/*"}:
            return True
        if any(char in p for char in "*?["):
            import fnmatch

            return fnmatch.fnmatchcase(c, p)
        return c == p or c.startswith(p + "/")

    @staticmethod
    def _leaf_readiness(node: WorkNode) -> LeafReadinessV1:
        text = " ".join(
            (
                node.contract.title,
                node.contract.objective,
                *node.contract.acceptance_criteria,
            )
        ).casefold()
        component_terms = (
            "world", "road", "environment", "vehicle", "car", "wheel", "chassis",
            "character", "gameplay", "collision", "scoring", "camera", "hud", "audio",
            "lighting", "quality", "accessibility", "العربية", "الطريق", "الشخصية", "اللوجيك",
        )
        matched = tuple(term for term in component_terms if term in text)
        shared_artifact = len(node.write_paths) == 1 and node.write_paths[0].casefold().endswith((".html", ".htm"))
        score = 1.0
        reasons: list[str] = []
        visual_shared = any(marker in text for marker in ("three.js", "threejs", "webgl", "3d", "game", "لعبة"))
        if shared_artifact and (len(matched) >= 2 or visual_shared):
            score -= 0.55
            reasons.append("single final artifact still contains multiple independently evaluable components")
        if len(node.contract.acceptance_criteria) > 6:
            score -= 0.20
            reasons.append("broad acceptance surface")
        if len(node.contract.objective) > 1_000:
            score -= 0.15
            reasons.append("large objective")
        ready = score >= 0.65 or bool(node.contract.metadata.get("component_leaf"))
        return LeafReadinessV1(
            node_id=node.id,
            ready=ready,
            score=max(0.0, min(1.0, score)),
            reasons=tuple(reasons) or ("bounded specialist contract",),
            recommended_children=0 if ready else min(8, max(3, len(matched))),
        )

    @classmethod
    def _concerns_for_owner(
        cls,
        matrix: ConcernCoverageMatrixV1,
        *owners: str,
    ) -> tuple[ConcernRequirementV1, ...]:
        wanted = {str(item).casefold() for item in owners if str(item).strip()}
        return tuple(
            item
            for item in matrix.concerns
            if item.owner_hint.casefold() in wanted
        )

    @staticmethod
    def _with_concern_contract(
        child: Mapping[str, Any],
        concerns: Sequence[ConcernRequirementV1],
        *,
        family: str,
    ) -> dict[str, Any]:
        value = dict(child)
        metadata = _mapping(value.get("metadata"))
        existing_ids = _strings(metadata.get("concern_ids"))
        metadata.update(
            {
                "task_family": family,
                "concern_ids": list(dict.fromkeys((*existing_ids, *(item.id for item in concerns)))),
                "concern_contracts": [asdict(item) for item in concerns],
            }
        )
        value["metadata"] = metadata
        acceptance = list(_strings(value.get("acceptance_criteria")))
        verification = list(_strings(value.get("verification")))
        for item in concerns:
            acceptance.extend(item.acceptance_criteria)
            verification.extend(item.verification)
        value["acceptance_criteria"] = list(dict.fromkeys(acceptance))
        value["verification"] = list(dict.fromkeys(verification))
        return value

    @staticmethod
    def _inherited_concerns_for_child(
        parent: WorkNode,
        child_domain: str,
    ) -> tuple[ConcernRequirementV1, ...]:
        targets: Mapping[str, tuple[str, ...]] = {
            "functional_correctness": ("qa.functional",),
            "integration_contracts": ("qa.functional",),
            "regression_safety": ("qa.functional",),
            "spatial_semantics": ("character.controls.movement",),
            "gameplay_state": ("gameplay.progression.state",),
            "progression_pacing": (
                "gameplay.traffic.difficulty",
                "gameplay.progression.rewards",
            ),
            "world_scale": ("world.road.geometry",),
            "runtime_performance": ("qa.performance",),
            "interactive_accessibility": ("presentation.accessibility",),
        }
        selected: list[ConcernRequirementV1] = []
        for raw in parent.contract.metadata.get("concern_contracts", ()):
            if not isinstance(raw, Mapping):
                continue
            concern_id = str(raw.get("id", ""))
            destinations = targets.get(concern_id, ())
            if destinations and not any(
                target == child_domain or target.startswith(child_domain + ".")
                for target in destinations
            ):
                continue
            if not destinations:
                # Unmapped concerns stay owned by the parent integrator rather
                # than being copied into every leaf.
                continue
            selected.append(
                ConcernRequirementV1(
                    id=concern_id,
                    title=str(raw.get("title", concern_id)),
                    objective=str(raw.get("objective", concern_id)),
                    acceptance_criteria=_strings(raw.get("acceptance_criteria")),
                    verification=_strings(raw.get("verification")),
                    owner_hint=str(raw.get("owner_hint", "implementation")),
                    critical=bool(raw.get("critical", True)),
                    source=str(raw.get("source", "harness_inferred")),
                )
            )
        return tuple(selected)

    @staticmethod
    def _specialist_profile(node: WorkNode) -> SpecialistProfileV1:
        default_interface = (
            f"create{re.sub(r'[^A-Za-z0-9]+', ' ', node.contract.title).title().replace(' ', '')}"
            "(context)"
        )
        owned_interfaces = node.contract.owned_interfaces or (default_interface,)
        expertise = tuple(
            dict.fromkeys(
                (
                    node.contract.title,
                    *owned_interfaces,
                    *tuple(str(item) for item in node.contract.metadata.get("expertise", ())),
                )
            )
        )
        return SpecialistProfileV1(
            id=f"specialist:{node.id}",
            node_id=node.id,
            parent_profile_id=f"specialist:{node.parent_id}" if node.parent_id else None,
            mission=node.contract.objective,
            expertise=expertise,
            context={
                "acceptance_criteria": list(node.contract.acceptance_criteria),
                "verification": list(node.contract.verification),
                "component_package_only": bool(node.contract.metadata.get("component_package_only")),
                "task_family": str(node.contract.metadata.get("task_family", "general")),
                "owned_concerns": list(node.contract.metadata.get("concern_contracts", ())),
                "instruction": (
                    "Treat every owned concern as a required engineering contract. "
                    "Produce executable evidence for it; do not replace it with a prose claim."
                ),
            },
            owned_interfaces=owned_interfaces,
            deliverable=(
                "A typed component package for the parent assembler"
                if node.contract.metadata.get("component_package_only")
                else "An integrated implementation inside the approved write scope"
            ),
            quality_rubric={
                "minimum_overall_score": 0.95,
                "minimum_critical_score": 0.90,
                "dimensions": list(
                    dict.fromkeys(
                        (
                            "functional",
                            "integration",
                            "maintainability",
                            *tuple(
                                str(item)
                                for item in node.contract.metadata.get("concern_ids", ())
                            ),
                        )
                    )
                ),
            },
            dependencies=node.depends_on,
        )

    @staticmethod
    def _interface_contract(node: WorkNode) -> Mapping[str, Any]:
        profile = UltraOrchestrator._specialist_profile(node)
        return {
            "schema_name": "InterfaceContractV1",
            "node_id": node.id,
            "exports": list(profile.owned_interfaces),
            "imports": list(node.depends_on),
            "invariants": [
                criterion
                for criterion in node.contract.acceptance_criteria[:4]
            ],
            "integration_points": [
                "Parent consumes the exact materialized file hashes and declared exports.",
                "Specialist output never writes a final output path.",
            ],
            "version": 1,
        }

    @staticmethod
    def _deterministic_shared_artifact_children(parent: WorkNode) -> tuple[Mapping[str, Any], ...]:
        readiness = UltraOrchestrator._leaf_readiness(parent)
        forced = bool(parent.contract.metadata.get("force_recursive_specialists"))
        source = str(parent.contract.metadata.get("concern_source") or " ".join(
            (parent.contract.title, parent.contract.objective)
        ))
        family = str(parent.contract.metadata.get("task_family") or UltraOrchestrator._task_family(source))
        if family != "interactive_game":
            return ()
        if not forced and (
            readiness.ready
            or not any(
                path.casefold().endswith((".html", ".htm"))
                for path in parent.write_paths
            )
        ):
            return ()
        domains = (
            ("world", "WorldPackage", "World, road, environment, lighting, spatial composition, and scene depth"),
            ("vehicles", "VehiclePackage", "Vehicle modeling including chassis, wheels, cabin, glass, lights, and materials"),
            ("character", "CharacterPackage", "Character modeling, pose, animation, control feedback, and visual readability"),
            ("gameplay", "GameplayPackage", "Gameplay state, traffic, collisions, scoring, progression, and input logic"),
            ("presentation", "PresentationPackage", "Camera, HUD, audio hooks, effects, responsiveness, and accessibility"),
            ("qa", "QAPackage", "Functional, visual, performance, browser, and accessibility acceptance evidence"),
        )
        children: list[Mapping[str, Any]] = []
        matrix = UltraOrchestrator._concern_coverage_matrix(source)
        concern_owners = {
            "world": ("world",),
            "vehicles": ("implementation",),
            "character": ("character_controls",),
            "gameplay": ("implementation", "gameplay_state", "gameplay_progression"),
            "presentation": ("presentation",),
            "qa": ("integration", "verification", "qa_performance"),
        }
        previous: str | None = None
        for suffix, owned_interface, objective in domains:
            child_id = f"{parent.id}.{suffix}"
            children.append(
                UltraOrchestrator._with_concern_contract(
                    {
                    "id": child_id,
                    "title": suffix.replace("_", " ").title() + " specialist",
                    "objective": objective,
                    "acceptance_criteria": [
                        f"The {suffix} component is detailed, coherent, and ready for parent integration.",
                        "The package exposes explicit integration guidance and observable quality evidence.",
                    ],
                    "verification": [
                        "Review the isolated component contract and its integration fixture.",
                        "Check the package against the parent acceptance criteria it supports.",
                    ],
                    "depends_on": [previous] if previous and suffix in {"gameplay", "presentation", "qa"} else [],
                    "write_paths": [],
                    "owned_interfaces": (
                        [owned_interface]
                        if owned_interface in parent.contract.owned_interfaces
                        else []
                    ),
                    "metadata": {
                        "component_package_only": True,
                        "materialized_components_required": True,
                        "final_output_paths": list(parent.write_paths),
                        "specialist_domain": suffix,
                    },
                    },
                    UltraOrchestrator._concerns_for_owner(
                        matrix, *concern_owners.get(suffix, ())
                    ),
                    family=matrix.task_family,
                )
            )
            previous = child_id
        missing = matrix.missing_critical_owners(children)
        if missing:
            raise AgentProtocolError(
                "interactive game specialist tree leaves critical concerns without owners: "
                + ", ".join(missing)
            )
        return tuple(children)

    @staticmethod
    def _deterministic_specialist_children(parent: WorkNode) -> tuple[Mapping[str, Any], ...]:
        """Recursively split broad visual/game domains into local-model leaves."""

        metadata = parent.contract.metadata
        domain = str(metadata.get("specialist_domain") or "").casefold()
        if not domain:
            return ()
        parts: Mapping[str, tuple[tuple[str, str], ...]] = {
            "world": (
                ("road", "Road geometry, lane language, collision surface, and reusable segment contract"),
                ("environment", "Environment props, depth layers, boundaries, spawn anchors, and composition"),
                ("lighting", "Lighting, atmosphere, shadows, time treatment, and material readability"),
            ),
            "vehicles": (
                ("chassis", "Vehicle body silhouette, chassis proportions, bumpers, and structural details"),
                ("wheels", "Wheel geometry, suspension placement, steering pose, and grounded contact"),
                ("cabin", "Cabin, glass, interior hints, mirrors, and driver readability"),
                ("materials", "Paint, glass, tire, light materials, emissive cues, and color variants"),
            ),
            "character": (
                ("body", "Character body model, proportions, silhouette, and readable pose"),
                ("animation", "Animation states, transitions, timing, and motion feedback"),
                ("controls", "Character controls, interaction boundaries, and game-loop contract"),
            ),
            "gameplay": (
                ("traffic", "Traffic spawning, lane behavior, difficulty pacing, and deterministic updates"),
                ("collision", "Collision rules, recovery, invulnerability windows, and observable feedback"),
                ("progression", "Scoring, progression, restart flow, goals, and state transitions"),
            ),
            "presentation": (
                ("camera", "Camera composition, follow behavior, shake, framing, and responsive viewport"),
                ("hud", "HUD hierarchy, readability, controls help, score, state, and responsive layout"),
                ("effects", "Audio hooks, particles, impacts, speed feedback, and polish effects"),
                ("accessibility", "Keyboard/touch affordances, reduced motion, contrast, and status feedback"),
            ),
            "qa": (
                ("functional", "Playable-flow, input, collision, scoring, restart, console, and WebGL checks"),
                ("visual", "Modeling, composition, lighting, readability, feedback, and polish rubric"),
                ("performance", "Frame stability, resize behavior, asset/runtime constraints, and accessibility checks"),
            ),
            "world.road": (
                ("geometry", "Reusable lane and verge geometry with stable dimensions and segment APIs"),
                ("markings", "Lane markings, crossings, shoulders, palette, and distance readability"),
                ("collision", "Collision surfaces, bounds, spawn-safe zones, and debug visualization"),
            ),
            "world.environment": (
                ("terrain", "Terrain layers, boundaries, depth planes, and world extents"),
                ("props", "Reusable stylized props, placement anchors, density, and variation"),
                ("composition", "Background layering, palette rhythm, landmark spacing, and readability"),
            ),
            "world.environment.terrain": (
                ("banks", "Road-side bank geometry, contracted corridor clearance, grounding, and extents"),
                ("verges", "Layered verge contours, drainage and edge transitions outside the road corridor"),
                ("ground_cover", "Deterministic grass, soil, pebble, and low ground-cover variation outside the playable corridor"),
            ),
            "world.environment.props": (
                ("trees", "Reusable broadleaf and pine prop models with coherent low-poly silhouettes"),
                ("rocks", "Reusable faceted boulder and rock-cluster models with grounded variation"),
                ("shrubs", "Reusable layered shrub, flower, and low vegetation clusters"),
                ("roadside", "Reusable signpost, fence, bollard, and roadside-detail models"),
            ),
            "world.environment.composition": (
                ("hills", "Distant rolling hill silhouettes and atmospheric depth separation"),
                ("tree_line", "Irregular distant tree-line rhythm with controlled gaps and scale variation"),
                ("landmarks", "Asymmetric farm landmarks outside the playable corridor for orientation and visual interest"),
            ),
            "world.lighting": (
                ("rig", "Key, fill, ambient, and hemisphere lighting contracts"),
                ("atmosphere", "Fog, sky, exposure, color treatment, and depth separation"),
                ("shadows", "Shadow quality, contact cues, performance bounds, and material readability"),
            ),
            "world.lighting.rig": (
                ("lights", "Bounded production key, hemisphere fill, rim, and shadow-camera light set"),
                ("fixture", "Compact neutral multi-material forms and receiver used to prove the light-set response"),
            ),
            "world.lighting.atmosphere": (
                ("settings", "Reusable color-background, fog, tone-mapping, and exposure application contract"),
                ("fixture", "Visible near/mid/far low-poly forms used to prove atmospheric depth separation"),
            ),
            "world.lighting.shadows": (
                ("settings", "Reusable renderer and harness-key shadow configuration with bounded bias and map size"),
                ("fixture", "Horizontal receiver and grounded casting forms used to prove contact-shadow readability"),
            ),
            "vehicles.chassis": (
                ("shell", "Distinctive body shell silhouette, hood, roof, doors, and coherent proportions"),
                ("structure", "Bumpers, grille, fenders, underbody, wheel arches, and structural detail"),
                ("variants", "Reusable vehicle-size and style variants without losing a coherent design language"),
            ),
            "vehicles.chassis.shell": (
                ("volumes", "Primary tub, hood, cabin, and rear-deck volumes with a fixed X-width/Z-length envelope"),
                ("panels", "Fenders, doors, fascia, trim, and four wheel-mount cues aligned to the primary volumes"),
            ),
            "vehicles.wheels": (
                ("tire", "Rounded tire geometry, tread/readability, believable width, and material response"),
                ("rim", "Rim, spokes, hub, axle alignment, and visible rotational detail"),
                ("contact", "Wheel placement, wheel-arch clearance, ground contact, steering pose, and shadow cues"),
            ),
            "vehicles.cabin": (
                ("frame", "Cabin volume, roof pillars, windshield rake, side and rear window framing"),
                ("glass", "Transparent glass treatment, reflections, tint, thickness cues, and readability"),
                ("interior", "Seats, dashboard, steering-wheel hints, mirrors, and driver-facing detail"),
            ),
            "vehicles.materials": (
                ("paint", "Layered body paint, palette variants, highlights, roughness, and visual hierarchy"),
                ("lights", "Headlights, tail lights, indicators, emissive response, and lamp housings"),
                ("surface", "Coherent tire, metal, chrome, glass, trim, and underside materials"),
            ),
            "character.body": (
                ("silhouette", "Immediately readable stylized crossing character silhouette and proportions"),
                ("anatomy", "Head, torso, wings or arms, legs, feet, facial cues, and coherent articulation"),
                ("surface", "Character materials, color blocking, small details, and lighting readability"),
            ),
            "character.animation": (
                ("rig", "Animation-ready pivots, articulated parts, pose invariants, and state hooks"),
                ("locomotion", "Idle, hop, run, landing, hit, and celebration motion with clear timing"),
                ("secondary", "Secondary overlap, squash/stretch, anticipation, and non-synchronized personality"),
            ),
            "character.controls": (
                ("input", "Keyboard, pointer, and touch input mapping with buffered intent and focus safety"),
                ("movement", "Grid/world movement, lane crossing, bounds, orientation, and deterministic timing"),
                ("feedback", "Input acknowledgement, blocked movement, hit/recovery, and state-machine integration"),
            ),
            "gameplay.traffic": (
                ("spawning", "Deterministic lane definitions, safe spawn rules, pooling, and density control"),
                ("behavior", "Vehicle speed, direction, variety, spacing, despawn, and readable near misses"),
                ("difficulty", "Progressive traffic pacing, fairness, recovery windows, and reproducible seeds"),
            ),
            "gameplay.collision": (
                ("bounds", "Consistent player/vehicle collision volumes and lane/world coordinate contracts"),
                ("resolution", "Hit detection, knockback or fail state, invulnerability, and restart safety"),
                ("feedback", "Visible and audible impact feedback with deterministic test hooks"),
            ),
            "gameplay.progression": (
                ("score", "Forward progress scoring, best score, milestones, and anti-farming rules"),
                ("state", "Start, playing, paused, game-over, restart, and persistence transitions"),
                ("rewards", "Difficulty curve, goals, celebratory feedback, and replay motivation"),
            ),
            "presentation.camera": (
                ("framing", "Readable isometric/perspective framing, subject prominence, and world depth"),
                ("follow", "Smooth follow, look-ahead, bounds, responsive resize, and stable orientation"),
                ("impact", "Restrained shake, zoom, transitions, and reduced-motion alternatives"),
            ),
            "presentation.hud": (
                ("hierarchy", "Score, best, state, and controls with restrained hierarchy and no card clutter"),
                ("responsive", "Desktop/mobile placement, safe areas, touch targets, and typography scaling"),
                ("states", "Start, pause, game-over, restart, help, and accessibility status messaging"),
            ),
            "presentation.effects": (
                ("particles", "Landing dust, motion trails, collisions, pickups, and celebration particles"),
                ("audio", "Event-driven music/SFX hooks, volume controls, and graceful no-audio fallback"),
                ("polish", "Transitions, subtle environmental motion, varied timing, and performance bounds"),
            ),
        }
        definitions = parts.get(domain, ())
        if metadata.get("component_leaf") and not definitions:
            return ()
        root_domain = domain.split(".", 1)[0]
        domain_interface = {
            "world": "WorldPackage",
            "vehicles": "VehiclePackage",
            "character": "CharacterPackage",
            "gameplay": "GameplayPackage",
            "presentation": "PresentationPackage",
            "qa": "QAPackage",
        }.get(root_domain, f"{root_domain.title()}Package")
        children: list[Mapping[str, Any]] = []
        family = str(metadata.get("task_family", "interactive_game"))
        for suffix, objective in definitions:
            child_domain = f"{domain}.{suffix}"
            child = {
                "id": f"{parent.id}.{suffix}",
                "title": f"{suffix.replace('_', ' ').title()} specialist",
                "objective": objective,
                "acceptance_criteria": [
                    f"The {suffix.replace('_', ' ')} package is concrete and directly integrable.",
                    "Implementation, interface, tests, preview fixture, and evidence are explicit.",
                ],
                "verification": [
                    "Evaluate the actual component candidate against its bounded contract.",
                    "Reject unsupported claims and cite observable package evidence.",
                ],
                "depends_on": [],
                "write_paths": [],
                "owned_interfaces": (
                    [domain_interface]
                    if domain_interface in parent.contract.owned_interfaces
                    else []
                ),
                "metadata": {
                    "component_package_only": True,
                    "materialized_components_required": True,
                    "component_leaf": True,
                    "specialist_domain": child_domain,
                    "visual_required": child_domain not in {
                        "world.lighting.atmosphere.settings",
                        "world.lighting.shadows.settings",
                    },
                    "final_output_paths": list(metadata.get("final_output_paths", ())),
                },
            }
            children.append(
                UltraOrchestrator._with_concern_contract(
                    child,
                    UltraOrchestrator._inherited_concerns_for_child(parent, child_domain),
                    family=family,
                )
            )
        return tuple(children)

    @staticmethod
    def _deterministic_cross_domain_children(
        parent: WorkNode,
    ) -> tuple[Mapping[str, Any], ...]:
        """Give common large coding domains concrete, evaluable ownership."""

        if parent.contract.metadata.get("component_package_only"):
            return ()
        if parent.contract.metadata.get("cross_domain_template_root") is False:
            # The approved master plan already split the work. Its concern
            # contracts were distributed across those modules, so expanding
            # every module into the same family template would be brute force.
            return ()
        text = " ".join(
            (
                parent.contract.title,
                parent.contract.objective,
                *parent.contract.acceptance_criteria,
            )
        ).casefold()
        source = str(parent.contract.metadata.get("concern_source") or text)
        templates: tuple[tuple[str, str, tuple[str, ...]], ...] = ()
        family = str(parent.contract.metadata.get("task_family") or UltraOrchestrator._task_family(source))
        if family == "ml":
            templates = (
                ("data", "Data contracts, validation, leakage prevention, features, and reproducible splits", ("data",)),
                ("model", "Model architecture, baselines, calibration, and inference contract", ("implementation",)),
                ("training", "Training orchestration, configuration, reproducibility, and checkpoints", ("training",)),
                ("evaluation", "Metrics, slice analysis, robustness, regression, and acceptance evidence", ("evaluation", "verification", "integration")),
                ("serving", "Serving interface, schema validation, observability, and deployment fixture", ("serving",)),
            )
        elif family == "backend":
            templates = (
                ("domain", "Domain entities, invariants, policies, and service contracts", ("implementation",)),
                ("api", "HTTP/API surface, schemas, validation, stable errors, and compatibility", ("api",)),
                ("persistence", "Persistence model, transactions, concurrency, idempotency, migrations, and recovery", ("persistence",)),
                ("auth", "Authentication, authorization, validation, abuse cases, and security boundaries", ("auth",)),
                ("operations", "Latency/resource budgets, health, logs, metrics, traces, and deployment failure behavior", ("operations",)),
                ("tests", "Contract, integration, concurrency, failure-path, security, load, and regression tests", ("verification", "integration")),
            )
        elif family == "frontend":
            templates = (
                ("layout", "Responsive information architecture, hierarchy, navigation, and layout", ()),
                ("components", "Reusable UI components, states, interaction, and design-system contracts", ("implementation", "integration")),
                ("data", "Data fetching, cache/state, loading, error, stale, and empty-state behavior", ("data",)),
                ("accessibility", "Keyboard, focus, semantics, contrast, motion, and assistive feedback", ("accessibility",)),
                ("quality", "Client security boundaries, rendering/load performance, and resource budgets", ("quality",)),
                ("visual_qa", "Screenshot fixtures, visual rubric, responsive regression, and polish evidence", ("visual_qa", "verification")),
            )
        elif family == "general":
            # For an unknown task family the model-authored decomposition owns
            # the domain split. The harness still injects generic concerns into
            # approved contracts, but does not blindly manufacture four extra
            # agents for every small task.
            return ()
        if not templates or family == "interactive_game":
            return ()
        matrix = UltraOrchestrator._concern_coverage_matrix(source)
        children = tuple(
            UltraOrchestrator._with_concern_contract(
                {
                "id": f"{parent.id}.{suffix}",
                "title": f"{family.upper()} {suffix.replace('_', ' ').title()} specialist",
                "objective": objective,
                "acceptance_criteria": [
                    f"The {suffix.replace('_', ' ')} deliverable is concrete and independently testable.",
                    "The package exposes real files, explicit interfaces, tests, and a runnable review fixture.",
                ],
                "verification": [
                    "Run the bounded component checks and preview/report fixture.",
                    "Reject descriptions without materialized implementation evidence.",
                ],
                "depends_on": [],
                "write_paths": [],
                "owned_interfaces": [],
                "metadata": {
                    "component_package_only": True,
                    "materialized_components_required": True,
                    "component_leaf": True,
                    "specialist_domain": f"{family}.{suffix}",
                    "final_output_paths": list(parent.write_paths),
                },
                },
                UltraOrchestrator._concerns_for_owner(matrix, *owners),
                family=family,
            )
            for suffix, objective, owners in templates
        )
        missing = matrix.missing_critical_owners(children)
        if missing:
            raise AgentProtocolError(
                f"{family} specialist tree leaves critical concerns without owners: {', '.join(missing)}"
            )
        return children

    def _validated_children(
        self,
        parent: WorkNode,
        raw_children: Any,
    ) -> tuple[WorkNode, ...]:
        if not isinstance(raw_children, Sequence) or isinstance(raw_children, (str, bytes)):
            return ()
        if parent.depth >= self.config.max_depth and raw_children:
            raise ScopeRevisionRequired(
                f"node {parent.id} expansion exceeds max depth {self.config.max_depth}"
            )
        candidates = [item for item in raw_children if isinstance(item, Mapping)]
        child_ids = {
            str(item.get("id", f"{parent.id}.{index}")).strip()
            for index, item in enumerate(candidates, start=1)
        }
        if len(child_ids) != len(candidates):
            raise ScopeRevisionRequired(f"node {parent.id} produced duplicate child ids")
        known_dependencies = child_ids | set(parent.depends_on)
        children: list[WorkNode] = []
        for index, item in enumerate(candidates, start=1):
            fallback = f"{parent.id}.{index}"
            child = TaskContractV1.from_mapping(item, fallback_id=fallback)
            if child.id in self.nodes:
                raise ScopeRevisionRequired(f"dynamic node id {child.id!r} already exists")
            if set(child.depends_on) - known_dependencies:
                raise ScopeRevisionRequired(
                    f"child {child.id} depends outside its approved module boundary"
                )
            if child.write_paths and not parent.write_paths:
                raise ScopeRevisionRequired(
                    f"child {child.id} introduces write paths outside its parent contract"
                )
            for path in child.write_paths:
                if not any(self._contains_scope(scope, path) for scope in parent.write_paths):
                    raise ScopeRevisionRequired(
                        f"child {child.id} write path {path!r} exceeds parent scope"
                    )
            added_interfaces = set(child.owned_interfaces) - set(parent.contract.owned_interfaces)
            if added_interfaces:
                raise ScopeRevisionRequired(
                    f"child {child.id} introduces interfaces outside its parent contract: "
                    f"{sorted(added_interfaces)}"
                )
            metadata = dict(child.metadata)
            if metadata.get("scope_change") or metadata.get("new_external_dependencies"):
                raise ScopeRevisionRequired(
                    f"child {child.id} requests an approval-bound scope/interface change"
                )
            parent_external = set(_strings(parent.contract.metadata.get("external_dependencies")))
            child_external = set(_strings(metadata.get("external_dependencies")))
            if child_external - parent_external:
                raise ScopeRevisionRequired(
                    f"child {child.id} introduces external dependencies: "
                    f"{sorted(child_external - parent_external)}"
                )
            inherited_dependencies = tuple(dict.fromkeys((*parent.depends_on, *child.depends_on)))
            inherited_forbidden = tuple(
                dict.fromkeys((*parent.contract.forbidden_changes, *child.forbidden_changes))
            )
            component_only = bool(metadata.get("component_package_only")) or (
                len(parent.write_paths) == 1
                and parent.write_paths[0].casefold().endswith((".html", ".htm"))
                and not child.write_paths
            )
            if component_only:
                metadata["component_package_only"] = True
                metadata.setdefault("final_output_paths", list(parent.write_paths))
            child = replace(
                child,
                depends_on=inherited_dependencies,
                write_paths=() if component_only else (child.write_paths or parent.write_paths),
                forbidden_changes=inherited_forbidden,
                metadata=metadata,
            )
            self._order += 1
            children.append(
                WorkNode(
                    contract=child,
                    parent_id=parent.id,
                    depth=parent.depth + 1,
                    kind=(NodeKind.TASK if parent.depth + 1 >= self.config.max_depth else NodeKind.SUBMODULE),
                    order=self._order,
                    pre_write_hashes=_mapping(item.get("pre_write_hashes")),
                )
            )
        return tuple(children)

    def _plan_and_expand(self, node_id: str) -> None:
        assert self.run_state
        self.control.checkpoint()
        node = self.nodes[node_id]
        profile = self._specialist_profile(node)
        target = NodeQualityTargetV1(node_id=node.id)
        profile_saver = getattr(self.state, "save_specialist_profile", None)
        if callable(profile_saver):
            profile_saver(self.run_state.id, profile)
        interface_saver = getattr(self.state, "save_interface_contract", None)
        if callable(interface_saver):
            interface_saver(
                self.run_state.id,
                node.id,
                self._interface_contract(node),
            )
        target_saver = getattr(self.state, "save_node_quality_target", None)
        if callable(target_saver):
            target_saver(self.run_state.id, target)
        node = replace(node, status=NodeStatus.PLANNING, phase=InnerPhase.CONTEXT)
        self.nodes[node_id] = node
        self.state.save_work_node(self.run_state.id, node)
        context = self._new_context(node, AgentRole.PLANNER)
        contract_is_self_planning = bool(
            node.contract.metadata.get("component_package_only")
            and node.contract.objective
            and node.contract.acceptance_criteria
            and node.contract.verification
        )
        try:
            if contract_is_self_planning:
                plan_response = self._deterministic_mini_plan(node)
                self.events.publish(
                    "ultra.mini_plan_derived",
                    f"[{node.id}] derived mini-plan from the approved component contract",
                    run_id=self.run_state.id,
                    node_id=node.id,
                    steps=list(plan_response.payload.get("steps", ())),
                )
            else:
                plan_response = self._invoke(
                    AgentRole.PLANNER,
                    InnerPhase.MINI_PLAN,
                    task={"contract": asdict(node.contract)},
                    context=context,
                    node_id=node.id,
                )
        except (AgentProtocolError, RuntimeError) as exc:
            # The contract and specialist profile already contain all facts
            # required for a bounded node plan. A small local model sometimes
            # mistakes node ids for repository paths or exhausts JSON repair.
            # That must not invalidate a previously approved, durable tree.
            plan_response = self._deterministic_mini_plan(node)
            self.events.publish(
                "ultra.mini_plan_repaired",
                f"[{node.id}] replaced invalid local mini-plan with a contract-derived plan",
                run_id=self.run_state.id,
                node_id=node.id,
                error=str(exc),
                steps=list(plan_response.payload.get("steps", ())),
            )
        readiness = self._leaf_readiness(node)
        deterministic_children = self._deterministic_shared_artifact_children(node)
        specialist_children = self._deterministic_specialist_children(node)
        cross_domain_children = self._deterministic_cross_domain_children(node)
        decompose_payload: Mapping[str, Any] = {}
        if specialist_children:
            raw_children = specialist_children
        elif node.contract.metadata.get("component_leaf"):
            raw_children = ()
        elif deterministic_children:
            # The final artifact contract requires independently evaluable
            # domains. A weak model's generic "subtask 1" decomposition is not
            # a substitute for named specialist ownership.
            raw_children = deterministic_children
        elif cross_domain_children:
            raw_children = cross_domain_children
        elif node.contract.metadata.get("component_package_only"):
            raw_children = ()
        else:
            decompose_response = self._invoke(
                AgentRole.DECOMPOSER,
                InnerPhase.DECOMPOSE,
                task={
                    "contract": asdict(node.contract),
                    "mini_plan": dict(plan_response.payload),
                    "leaf_readiness": asdict(readiness),
                    "remaining_node_budget": self.config.max_nodes - len(self.nodes),
                },
                context=context,
                node_id=node.id,
            )
            decompose_payload = dict(decompose_response.payload)
            raw_children = decompose_payload.get("children", ())
        children = self._validated_children(node, raw_children)
        if len(self.nodes) + len(children) > self.config.max_nodes:
            raise ScopeRevisionRequired(
                f"dynamic expansion exceeds max node count {self.config.max_nodes}"
            )
        self._prepared[node.id] = (context, dict(plan_response.payload))
        self._research_required[node.id] = bool(
            plan_response.payload.get("research_required")
            or decompose_payload.get("research_required")
            or node.contract.metadata.get("research_required")
        )
        if children:
            topology_recorder = getattr(self.state, "record_specialist_topology", None)
            if callable(topology_recorder):
                topology_recorder(
                    self.run_state.id,
                    node,
                    children,
                    readiness,
                )
            for child in children:
                self.nodes[child.id] = child
                self.state.save_work_node(self.run_state.id, child)
            updated_contract = replace(
                node.contract,
                depends_on=tuple(dict.fromkeys((*node.depends_on, *(child.id for child in children)))),
            )
            node = replace(
                node,
                contract=updated_contract,
                children=tuple(child.id for child in children),
            )
            self.nodes[node.id] = node
            self.state.save_work_node(self.run_state.id, node)
            for child in children:
                self._plan_and_expand(child.id)
        node = replace(self.nodes[node_id], status=NodeStatus.READY, phase=InnerPhase.DECOMPOSE)
        self.nodes[node_id] = node
        self.state.save_work_node(self.run_state.id, node)

    @staticmethod
    def _deterministic_mini_plan(node: WorkNode) -> AgentResponse:
        """Build the smallest inspectable plan already implied by a node contract."""

        contract = node.contract
        steps = [
            f"Implement the bounded objective: {contract.objective}",
            *(
                f"Verify acceptance criterion: {criterion}"
                for criterion in contract.acceptance_criteria
            ),
            *(f"Collect evidence by: {check}" for check in contract.verification),
        ]
        if contract.metadata.get("component_package_only"):
            steps.extend(
                (
                    "Publish real implementation, interface, tests, and a runnable preview "
                    "through the typed component artifact contract.",
                    "Revise only this component until runtime and independent visual gates pass.",
                )
            )
        return AgentResponse.from_mapping(
            {
                "payload": {
                    "steps": list(dict.fromkeys(step for step in steps if step.strip())),
                    "research_required": bool(contract.metadata.get("research_required")),
                    "source": "deterministic_contract_fallback",
                },
                "summary": "Contract-derived mini-plan",
                "reasoning_summary": (
                    "The approved task contract already provides the objective, acceptance "
                    "criteria, verification, ownership, and artifact boundary."
                ),
            },
            node_id=node.id,
            provider="harness",
            model="deterministic",
        )

    def _ensure_expanded(self, node_id: str) -> None:
        """Resume-aware expansion that never recreates durable child nodes."""

        node = self.nodes[node_id]
        if node.status is NodeStatus.COMPLETED:
            return
        if node.children:
            self._prepared.setdefault(node.id, ({}, {}))
            for child_id in node.children:
                if child_id in self.nodes:
                    self._ensure_expanded(child_id)
            if node.status in {NodeStatus.PENDING, NodeStatus.PLANNING}:
                node = replace(node, status=NodeStatus.READY, phase=InnerPhase.DECOMPOSE)
                self.nodes[node.id] = node
                assert self.run_state
                self.state.save_work_node(self.run_state.id, node)
            return
        if node.status is NodeStatus.READY and node.phase is InnerPhase.DECOMPOSE:
            self._prepared.setdefault(node.id, ({}, {}))
            return
        self._plan_and_expand(node_id)

    @staticmethod
    def _passed(response: AgentResponse) -> bool:
        return bool(response.payload.get("passed", response.payload.get("success", True)))

    @staticmethod
    def _findings(*responses: AgentResponse) -> tuple[str, ...]:
        values: list[str] = []
        for response in responses:
            values.extend(_strings(response.payload.get("issues")))
            values.extend(_strings(response.payload.get("findings")))
        return tuple(dict.fromkeys(values))

    @staticmethod
    def _records(response: AgentResponse, key: str) -> tuple[Mapping[str, Any], ...]:
        raw = response.payload.get(key, ())
        if isinstance(raw, Mapping):
            raw = [raw]
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            return ()
        return tuple(dict(item) for item in raw if isinstance(item, Mapping))

    def _quality_vote_records(
        self,
        node: WorkNode,
        responses: tuple[AgentResponse, ...],
    ) -> tuple[Mapping[str, Any], ...]:
        labels = [
            "clean_code",
            "security",
            "runtime_tests",
            "test_quality",
            "triage",
        ]
        if len(responses) > len(labels):
            labels.extend(
                f"harness_gate_{index}"
                for index in range(1, len(responses) - len(labels) + 1)
            )
        records: list[Mapping[str, Any]] = []
        for label, response in zip(labels, responses):
            payload = response.payload
            reasoning_evaluation = _mapping(payload.get("harness_reasoning_evaluation"))
            explicit_consensus_vote = str(
                payload.get("consensus_vote") or ""
            ).strip().casefold()
            declared_verdict = str(
                explicit_consensus_vote or payload.get("verdict") or ""
            ).strip().casefold()
            passed = self._passed(response)
            # ``passed`` is the normalized typed quality verdict. Small local
            # models sometimes emit a contradictory free-form ``verdict``
            # field (for example passed=true plus verdict=reject). Letting that
            # redundant field control consensus can turn six passing gates into
            # a rejection with no finding. Consensus therefore votes from the
            # typed gate result and retains the raw declaration only as audit
            # evidence.
            raw_verdict = "accept" if passed else "reject"
            # ``consensus_vote`` is the formal typed swarm protocol and must
            # remain authoritative even when a provider also emits a
            # contradictory generic ``passed`` field. Free-form ``verdict``
            # remains audit evidence only because weak models use it loosely.
            if explicit_consensus_vote in {"reject", "rejected", "deny", "no"}:
                raw_verdict = "reject"
            elif explicit_consensus_vote in {"accept", "accepted", "approve", "yes"}:
                raw_verdict = "accept" if passed else "reject"
            if reasoning_evaluation and not bool(reasoning_evaluation.get("passed", True)):
                raw_verdict = "reject"
            try:
                confidence = float(payload.get("confidence", payload.get("quality_confidence", 1.0)))
            except (TypeError, ValueError):
                confidence = 1.0
            if reasoning_evaluation and not bool(reasoning_evaluation.get("passed", True)):
                confidence = max(confidence, 1.0)
            records.append(
                {
                    "voter_agent_id": f"{node.id}:{label}",
                    "role": label,
                    "verdict": raw_verdict,
                    "confidence": max(0.0, min(1.0, confidence)),
                    "passed": passed,
                    "summary": response.summary,
                    "rationale": response.reasoning_summary or response.summary,
                    "evidence": {
                        "declared_verdict": declared_verdict,
                        "issues": list(_strings(payload.get("issues"))),
                        "findings": list(_strings(payload.get("findings"))),
                        "test_results": list(self._records(response, "test_results")),
                        "harness_reasoning_evaluation": reasoning_evaluation,
                    },
                }
            )
        return tuple(records)

    def _record_quality_consensus(
        self,
        node: WorkNode,
        responses: tuple[AgentResponse, ...],
    ) -> Mapping[str, Any]:
        recorder = getattr(self.state, "record_quality_consensus", None)
        if not callable(recorder):
            return {}
        return dict(
            recorder(
                node.id,
                self._quality_vote_records(node, responses),
            )
            or {}
        )

    @staticmethod
    def _consensus_accepted(consensus: Mapping[str, Any]) -> bool:
        if not consensus:
            return True
        return str(consensus.get("status", "")).casefold() == "accepted"

    def _quality_gate_passed(self, result: QualityGateResultV1) -> bool:
        return all(self._passed(item) for item in result.responses) and self._consensus_accepted(result.consensus)

    @staticmethod
    def _review_candidate(response: AgentResponse) -> Mapping[str, Any]:
        return {
            "summary": response.summary,
            "payload": dict(response.payload),
            "artifacts": list(UltraOrchestrator._records(response, "artifacts")),
            "evidence": list(UltraOrchestrator._records(response, "evidence")),
            "test_results": list(UltraOrchestrator._records(response, "test_results")),
        }

    @staticmethod
    def _merge_candidate_response(
        base: AgentResponse,
        revision: AgentResponse,
    ) -> AgentResponse:
        def merge(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
            combined = dict(left)
            for key, value in right.items():
                previous = combined.get(key)
                combined[key] = (
                    merge(previous, value)
                    if isinstance(previous, Mapping) and isinstance(value, Mapping)
                    else value
                )
            return combined

        return AgentResponse(
            payload=merge(base.payload, revision.payload),
            summary=revision.summary or base.summary,
            insights=tuple((*base.insights, *revision.insights)),
            reasoning_summary=revision.reasoning_summary or base.reasoning_summary,
            usage={
                key: int(base.usage.get(key, 0)) + int(revision.usage.get(key, 0))
                for key in set(base.usage) | set(revision.usage)
            },
            provider=revision.provider or base.provider,
            model=revision.model or base.model,
        )

    @staticmethod
    def _refine_contract_from_replan(
        contract: TaskContractV1,
        replan: Mapping[str, Any],
    ) -> TaskContractV1:
        requirements: list[str] = []

        def collect(value: Any, key: str = "") -> None:
            if isinstance(value, Mapping):
                for child_key, child in value.items():
                    name = str(child_key)
                    collect(
                        child,
                        name if name in {"verification_plan", "findings", "claim"} else key,
                    )
                return
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                for child in value:
                    collect(child, key)
                return
            text = str(value or "").strip()
            if text and key in {"verification_plan", "findings", "claim"}:
                requirements.append(text[:600])

        collect(replan)
        requirements = list(dict.fromkeys(requirements))[:8]
        if not requirements:
            return contract
        signatures: list[str] = []
        for requirement in requirements:
            signatures.extend(
                match.strip()
                for match in re.findall(
                    r"`([^`(]{2,160}\([^`]*\)(?:\s*:\s*[A-Za-z][\w<>\[\]]*)?)`",
                    requirement,
                )
                if match.strip()
            )
        metadata = dict(contract.metadata)
        metadata["replan_refinement_requirements"] = requirements
        return replace(
            contract,
            acceptance_criteria=tuple(
                dict.fromkeys(
                    (
                        *contract.acceptance_criteria,
                        *(f"Replan refinement requirement: {item}" for item in requirements),
                    )
                )
            ),
            verification=tuple(
                dict.fromkeys((*contract.verification, *requirements))
            ),
            owned_interfaces=tuple(
                dict.fromkeys((*contract.owned_interfaces, *signatures))
            ),
            metadata=metadata,
        )

    def _materialize_component_gate(
        self,
        node: WorkNode,
        candidate_response: AgentResponse,
        *,
        revision: int,
    ) -> tuple[Mapping[str, Any], tuple[AgentResponse, ...]]:
        if not node.contract.metadata.get("component_package_only"):
            return {}, ()
        materializer = getattr(self.state, "materialize_component_candidate", None)
        if not callable(materializer):
            # Lightweight engine adapters may omit filesystem integration.
            # The production adapter always implements this gate.
            return {}, ()
        assert self.run_state
        try:
            result = dict(
                materializer(
                    self.run_state.id,
                    node,
                    candidate_response,
                    revision=revision,
                    child_packages={
                        child_id: dict(self._results[child_id].component_package)
                        for child_id in node.children
                        if child_id in self._results
                    },
                )
                or {}
            )
            passed = bool(result.get("passed"))
            findings = tuple(_strings(result.get("findings")))
            status = str(result.get("status") or ("accepted" if passed else "rejected"))
            return result, (
                AgentResponse(
                    payload={
                        "passed": passed,
                        "success": passed,
                        "findings": list(findings),
                        "status": status,
                        "materialized_component_gate": result,
                        "confidence": 1.0,
                    },
                    summary=(
                        f"Materialized component gate {status} for {node.id}"
                    ),
                    reasoning_summary=(
                        "Harness-owned file hashes, preview runtime, and independent "
                        "visual evidence determine this verdict."
                    ),
                    provider="harness",
                    model="materialized-component-v2",
                ),
            )
        except Exception as exc:
            finding = f"materialized component gate failed: {exc}"
            return {"passed": False, "status": "rejected", "findings": [finding]}, (
                AgentResponse(
                    payload={
                        "passed": False,
                        "success": False,
                        "status": "rejected",
                        "findings": [finding],
                        "confidence": 1.0,
                    },
                    summary=f"Materialized component rejected for {node.id}",
                    reasoning_summary="The specialist did not produce a valid runnable package.",
                    provider="harness",
                    model="materialized-component-v2",
                ),
            )

    def _package_consumption_gate(
        self,
        node: WorkNode,
    ) -> tuple[AgentResponse, ...]:
        verifier = getattr(self.state, "verify_package_consumption", None)
        if not callable(verifier) or not node.children or not node.write_paths:
            return ()
        assert self.run_state
        try:
            result = dict(
                verifier(
                    self.run_state.id,
                    node,
                    tuple(
                        dict(self._results[child_id].component_package)
                        for child_id in node.children
                        if child_id in self._results
                    ),
                )
                or {}
            )
            passed = bool(result.get("passed"))
            findings = list(_strings(result.get("findings")))
        except Exception as exc:
            passed = False
            findings = [f"package consumption verification failed: {exc}"]
            result = {"passed": False, "findings": findings}
        return (
            AgentResponse(
                payload={
                    "passed": passed,
                    "success": passed,
                    "findings": findings,
                    "package_consumption_gate": result,
                    "confidence": 1.0,
                },
                summary=(
                    f"Package consumption {'verified' if passed else 'rejected'} for {node.id}"
                ),
                reasoning_summary=(
                    "The harness compared staged component hashes/content with final assembler outputs."
                ),
                provider="harness",
                model="package-consumption-v1",
            ),
        )

    def _quality(
        self,
        node: WorkNode,
        candidate_response: AgentResponse,
        *,
        authoritative_responses: tuple[AgentResponse, ...] = (),
    ) -> QualityGateResultV1:
        candidate = self._review_candidate(candidate_response)
        evaluation_policy = {
            "evaluate_candidate_not_contract_alone": True,
            "cite_candidate_evidence_for_every_blocking_finding": True,
            "do_not_invent_requirements_outside_contract": True,
            "component_package_only": bool(node.contract.metadata.get("component_package_only")),
            "component_packages_must_not_write_final_output": True,
            "component_files_are_embedded_in_candidate": True,
            "component_reads_use_package_relative_paths": True,
            "review_is_read_only": True,
        }
        clean_code = self._invoke(
            AgentRole.CLEAN_CODE_REVIEWER,
            InnerPhase.REVIEW,
            task={"contract": asdict(node.contract), "candidate": candidate, "evaluation_policy": evaluation_policy, "fresh_review": True, "category": "clean_code", "read_only": True},
            context=self._new_context(node, AgentRole.CLEAN_CODE_REVIEWER),
            node_id=node.id,
        )
        security = self._invoke(
            AgentRole.SECURITY_REVIEWER,
            InnerPhase.REVIEW,
            task={"contract": asdict(node.contract), "candidate": candidate, "evaluation_policy": evaluation_policy, "fresh_review": True, "category": "security", "read_only": True},
            context=self._new_context(node, AgentRole.SECURITY_REVIEWER),
            node_id=node.id,
        )
        tests = self._invoke(
            AgentRole.TESTER,
            InnerPhase.TEST,
            task={"contract": asdict(node.contract), "candidate": candidate, "evaluation_policy": evaluation_policy, "fresh_test_context": True},
            context=self._new_context(node, AgentRole.TESTER),
            node_id=node.id,
        )
        test_quality = self._invoke(
            AgentRole.TEST_QUALITY_REVIEWER,
            InnerPhase.REVIEW,
            task={"contract": asdict(node.contract), "candidate": candidate, "evaluation_policy": evaluation_policy, "fresh_review": True, "category": "test_quality", "read_only": True},
            context=self._new_context(node, AgentRole.TEST_QUALITY_REVIEWER),
            node_id=node.id,
        )
        review_pairs = tuple(
            zip(
                ("clean_code", "security", "runtime_tests", "test_quality"),
                (clean_code, security, tests, test_quality),
            )
        )
        if evaluation_policy["component_package_only"]:
            # Component reviewers already operate in separate clean contexts.
            # Normalization is deliberately deterministic: asking the same
            # weak builder to reinterpret typed votes can invert a rejection
            # or invent a finding, while adding no new evidence.
            normalized_findings = self._findings(
                clean_code,
                security,
                tests,
                test_quality,
            )
            reviewers_passed = all(
                self._passed(response) for _, response in review_pairs
            )
            triage = AgentResponse(
                payload={
                    "passed": reviewers_passed,
                    "success": reviewers_passed,
                    "issues": list(normalized_findings),
                    "findings": list(normalized_findings),
                    "evidence": [
                        {
                            "role": role,
                            "passed": self._passed(response),
                            "summary": response.summary,
                        }
                        for role, response in review_pairs
                    ],
                    "confidence": 1.0,
                },
                summary=(
                    "Typed component review verdicts accepted."
                    if reviewers_passed
                    else "Typed component review verdicts require revision."
                ),
                reasoning_summary=(
                    "The harness normalized and deduplicated independent typed "
                    "verdicts without asking the builder model to reinterpret them."
                ),
                provider="harness",
                model="deterministic-quality-triage-v1",
            )
        else:
            triage = self._invoke(
                AgentRole.QUALITY_TRIAGER,
                InnerPhase.REVIEW,
                task={
                    "contract": asdict(node.contract),
                    "candidate": candidate,
                    "evaluation_policy": evaluation_policy,
                    "read_only": True,
                    "normalize_and_deduplicate": True,
                    "review_summaries": [
                        clean_code.summary,
                        security.summary,
                        tests.summary,
                        test_quality.summary,
                    ],
                    "review_verdicts": [
                        {
                            "role": role,
                            "passed": self._passed(response),
                            "payload": dict(response.payload),
                        }
                        for role, response in review_pairs
                    ],
                },
                context=self._new_context(node, AgentRole.QUALITY_TRIAGER),
                node_id=node.id,
            )
        recorder = getattr(self.state, "record_quality_review", None)
        if callable(recorder):
            recorder(node.id, "clean_code", self._passed(clean_code))
            recorder(node.id, "security", self._passed(security))
            recorder(node.id, "test_quality", self._passed(test_quality))
        finding_recorder = getattr(self.state, "record_quality_findings", None)
        if callable(finding_recorder):
            finding_recorder(node.id, "clean_code", self._records(clean_code, "findings"))
            finding_recorder(node.id, "security", self._records(security, "findings"))
            finding_recorder(node.id, "test_quality", self._records(test_quality, "findings"))
        responses = (
            clean_code,
            security,
            tests,
            test_quality,
            triage,
            *authoritative_responses,
        )
        consensus = self._record_quality_consensus(node, responses)
        return QualityGateResultV1(responses=responses, consensus=consensus)

    def _execute_node(self, scheduled_node: WorkNode) -> ResultPackageV1:
        assert self.run_state
        node = self.nodes[scheduled_node.id]
        prior_result = self._results.get(node.id)
        prior_component = (
            dict(prior_result.component_package)
            if prior_result is not None and prior_result.component_package
            else {}
        )
        prior_replan = prior_component.get("replan")
        if isinstance(prior_replan, Mapping) and prior_replan:
            refined_contract = self._refine_contract_from_replan(node.contract, prior_replan)
            if refined_contract != node.contract:
                node = replace(node, contract=refined_contract)
                self.nodes[node.id] = node
                self.state.save_work_node(self.run_state.id, node)
        node = replace(node, status=NodeStatus.RUNNING, phase=InnerPhase.RESEARCH)
        self.nodes[node.id] = node
        self.state.save_work_node(self.run_state.id, node)
        self.events.publish(
            UltraEventKind.NODE.value,
            f"[{node.contract.title}] started",
            run_id=self.run_state.id,
            node_id=node.id,
            status="running",
        )
        responses: list[AgentResponse] = []
        if self._research_required.get(node.id, False):
            responses.append(
                self._invoke(
                    AgentRole.RESEARCHER,
                    InnerPhase.RESEARCH,
                    task={"contract": asdict(node.contract)},
                    context=self._new_context(node, AgentRole.RESEARCHER),
                    node_id=node.id,
                )
            )
        is_parent_assembler = bool(node.children)
        is_final_assembler = is_parent_assembler and bool(node.write_paths)
        node = replace(
            node,
            phase=InnerPhase.INTEGRATE if is_parent_assembler else InnerPhase.IMPLEMENT,
        )
        self.nodes[node.id] = node
        self.state.save_work_node(self.run_state.id, node)
        restorer = getattr(self.state, "restore_passed_component_candidate", None)
        revision_finding_reader = getattr(
            self.state,
            "component_revision_findings",
            None,
        )
        external_revision_findings = (
            tuple(revision_finding_reader(self.run_state.id, node))
            if callable(revision_finding_reader)
            and node.contract.metadata.get("component_package_only")
            and not is_parent_assembler
            else ()
        )
        implementation = (
            restorer(self.run_state.id, node)
            if callable(restorer)
            and node.contract.metadata.get("component_package_only")
            and not is_parent_assembler
            and not external_revision_findings
            else None
        )
        if implementation is not None:
            self.events.publish(
                "ultra.component_checkpoint_restored",
                f"[{node.id}] restored the last runtime-passing component package",
                run_id=self.run_state.id,
                node_id=node.id,
                phase=InnerPhase.IMPLEMENT.value,
            )
        else:
            implementation = self._invoke(
                AgentRole.INTEGRATOR if is_parent_assembler else AgentRole.CODER,
                InnerPhase.INTEGRATE if is_parent_assembler else InnerPhase.IMPLEMENT,
                task={
                    "contract": asdict(node.contract),
                    "mini_plan": dict(self._prepared.get(node.id, ({}, {}))[1]),
                    "final_assembler": is_final_assembler,
                    "component_assembler": is_parent_assembler and not is_final_assembler,
                    "child_component_packages": {
                        child_id: dict(self._results[child_id].component_package)
                        for child_id in node.children
                        if child_id in self._results
                    },
                    "prior_best_candidate": dict(prior_component.get("candidate", {})),
                    "prior_replan_guidance": dict(prior_component.get("replan", {})),
                    "prior_findings": list(
                        dict.fromkeys(
                            (
                                *(prior_result.findings if prior_result else ()),
                                *external_revision_findings,
                            )
                        )
                    ),
                    "findings": list(external_revision_findings),
                },
                context=self._new_context(
                    node, AgentRole.INTEGRATOR if is_parent_assembler else AgentRole.CODER
                ),
                node_id=node.id,
            )
        responses.append(implementation)
        candidate_response = implementation
        materialized_component, materialized_gate = self._materialize_component_gate(
            node,
            candidate_response,
            revision=1,
        )
        if materialized_component:
            candidate_response = replace(
                candidate_response,
                payload={
                    **dict(candidate_response.payload),
                    "materialized_component_package": materialized_component.get("package", {}),
                    "materialized_preview": materialized_component.get("preview", {}),
                    "visual_evaluations": materialized_component.get("visual_evaluations", ()),
                },
            )
        authoritative_gate = (
            *materialized_gate,
            *self._package_consumption_gate(node),
        )
        quality_gate = (
            QualityGateResultV1(responses=authoritative_gate)
            if authoritative_gate
            and not all(self._passed(item) for item in authoritative_gate)
            else self._quality(
                node,
                candidate_response,
                authoritative_responses=authoritative_gate,
            )
        )
        responses.extend(quality_gate.responses)
        fixes = 0
        while not self._quality_gate_passed(quality_gate) and fixes < self.config.max_fix_attempts:
            fixes += 1
            node = replace(node, phase=InnerPhase.FIX)
            self.nodes[node.id] = node
            self.state.save_work_node(self.run_state.id, node)
            findings = self._findings(*quality_gate.responses)
            if quality_gate.consensus and not self._consensus_accepted(quality_gate.consensus):
                findings = tuple(
                    dict.fromkeys(
                        (
                            *findings,
                            f"quality consensus {quality_gate.consensus.get('status')} for {node.id}",
                        )
                    )
                )
            self.events.publish(
                UltraEventKind.FIX.value,
                f"Fix loop {fixes}/{self.config.max_fix_attempts}",
                run_id=self.run_state.id,
                node_id=node.id,
                attempt=fixes,
                findings=findings,
            )
            fix_role = AgentRole.INTEGRATOR if is_parent_assembler else AgentRole.CODER
            fix = self._invoke(
                fix_role,
                InnerPhase.FIX,
                task={
                    "contract": asdict(node.contract),
                    "findings": findings,
                    "attempt": fixes,
                    "change_approach": fixes >= 3,
                    "optimization_variable": (
                        "finding_specific_implementation"
                        if fixes == 1
                        else "specialist_prompt_and_contract"
                        if fixes == 2
                        else "specialist_topology_or_interface_boundary"
                        if fixes == 3
                        else "fresh_challenger_rebuild"
                    ),
                    "champion_challenger": {
                        "preserve_previous_best": True,
                        "do_not_patch_a_rejected_visual_placeholder": fixes >= 3,
                        "candidate_must_win_pairwise": fixes > 1,
                    },
                    "final_assembler": is_final_assembler,
                    "component_assembler": is_parent_assembler and not is_final_assembler,
                    "child_component_packages": {
                        child_id: dict(self._results[child_id].component_package)
                        for child_id in node.children
                        if child_id in self._results
                    },
                },
                context=self._new_context(node, fix_role),
                node_id=node.id,
            )
            responses.append(fix)
            candidate_response = self._merge_candidate_response(candidate_response, fix)
            materialized_component, materialized_gate = self._materialize_component_gate(
                node,
                candidate_response,
                revision=fixes + 1,
            )
            if materialized_component:
                candidate_response = replace(
                    candidate_response,
                    payload={
                        **dict(candidate_response.payload),
                        "materialized_component_package": materialized_component.get("package", {}),
                        "materialized_preview": materialized_component.get("preview", {}),
                        "visual_evaluations": materialized_component.get("visual_evaluations", ()),
                    },
                )
            authoritative_gate = (
                *materialized_gate,
                *self._package_consumption_gate(node),
            )
            quality_gate = (
                QualityGateResultV1(responses=authoritative_gate)
                if authoritative_gate
                and not all(self._passed(item) for item in authoritative_gate)
                else self._quality(
                    node,
                    candidate_response,
                    authoritative_responses=authoritative_gate,
                )
            )
            responses.extend(quality_gate.responses)

        if not self._quality_gate_passed(quality_gate):
            replan = self._invoke(
                AgentRole.PLANNER,
                InnerPhase.REPLAN,
                task={
                    "contract": asdict(node.contract),
                    "findings": self._findings(*quality_gate.responses),
                    "consensus": dict(quality_gate.consensus),
                    "attempts": fixes,
                },
                context=self._new_context(node, AgentRole.PLANNER),
                node_id=node.id,
            )
            responses.append(replan)
            result = ResultPackageV1(
                node_id=node.id,
                success=False,
                status="revision_required",
                summary=replan.summary or "Quality gate exhausted; replan required",
                findings=tuple(
                    dict.fromkeys(
                        (
                            *self._findings(*quality_gate.responses),
                            *(
                                (f"quality consensus {quality_gate.consensus.get('status')} for {node.id}",)
                                if quality_gate.consensus and not self._consensus_accepted(quality_gate.consensus)
                                else ()
                            ),
                        )
                    )
                ),
                insights=tuple(insight for response in responses for insight in response.insights),
                fix_attempts=fixes,
                component_package={
                    "candidate": dict(self._review_candidate(candidate_response)),
                    "replan": dict(replan.payload),
                    "status": "best_candidate_below_target",
                },
            )
            self._results[node.id] = result
            self.state.save_result_package(self.run_state.id, result)
            node = replace(node, status=NodeStatus.REVISION_REQUIRED, phase=InnerPhase.REPLAN)
            self.nodes[node.id] = node
            self.state.save_work_node(self.run_state.id, node)
            raise NodePipelineFailed(result)

        if is_parent_assembler or materialized_component:
            # Parent fixes are cumulative integration revisions. Publish the
            # exact candidate that passed the final quality gate, never the
            # stale first integration response.
            integration = candidate_response
        else:
            node = replace(node, phase=InnerPhase.INTEGRATE)
            self.nodes[node.id] = node
            self.state.save_work_node(self.run_state.id, node)
            integration = self._invoke(
                AgentRole.INTEGRATOR,
                InnerPhase.INTEGRATE,
                task={
                    "contract": asdict(node.contract),
                    "publish_component_package": True,
                },
                context=self._new_context(node, AgentRole.INTEGRATOR),
                node_id=node.id,
            )
            responses.append(integration)
        node = replace(node, phase=InnerPhase.MEMORY_WRITEBACK)
        self.nodes[node.id] = node
        self.state.save_work_node(self.run_state.id, node)
        memory = self._invoke(
            AgentRole.MEMORY,
            InnerPhase.MEMORY_WRITEBACK,
            task={
                "contract": asdict(node.contract),
                "result_summaries": [response.summary for response in responses if response.summary],
            },
            context=self._new_context(node, AgentRole.MEMORY),
            node_id=node.id,
        )
        responses.append(memory)
        artifacts = tuple(
            item for response in responses for item in self._records(response, "artifacts")
        )
        evidence = tuple(
            item for response in responses for item in self._records(response, "evidence")
        )
        test_results = tuple(
            item for response in responses for item in self._records(response, "test_results")
        )
        raw_component = integration.payload.get("component_package")
        implementation_payload = (
            dict(raw_component.get("implementation", {}))
            if isinstance(raw_component, Mapping)
            else {
                "summary": integration.summary or implementation.summary,
                "artifacts": list(artifacts),
                "component_only": bool(node.contract.metadata.get("component_package_only")),
            }
        )
        passed_votes = sum(self._passed(item) for item in quality_gate.responses)
        quality_score = passed_votes / max(1, len(quality_gate.responses))
        component_package = ComponentPackageV1(
            node_id=node.id,
            implementation=implementation_payload,
            interface={
                "owned_interfaces": list(node.contract.owned_interfaces),
                "integration_guidance": (
                    raw_component.get("interface", {})
                    if isinstance(raw_component, Mapping)
                    else node.contract.metadata.get("integration_guidance", {})
                ),
            },
            tests=test_results,
            preview=(
                dict(raw_component.get("preview", {}))
                if isinstance(raw_component, Mapping)
                else {}
            ),
            dependencies=node.depends_on,
            evidence=evidence,
            quality={
                "overall_score": quality_score,
                "consensus": dict(quality_gate.consensus),
                "critical_minimum": 0.90,
                "target": 0.95,
            },
            status="published" if self._passed(integration) else "rejected",
        )
        result = ResultPackageV1(
            node_id=node.id,
            success=self._passed(integration),
            status="completed" if self._passed(integration) else "failed",
            summary=integration.summary or implementation.summary or f"{node.id} completed",
            artifacts=artifacts,
            evidence=evidence,
            test_results=test_results,
            findings=self._findings(*responses),
            insights=tuple(insight for response in responses for insight in response.insights),
            component_package=(
                dict(materialized_component.get("package", {}))
                if materialized_component.get("package")
                else asdict(component_package)
            ),
            fix_attempts=fixes,
        )
        self._results[node.id] = result
        self.state.save_result_package(self.run_state.id, result)
        package_saver = getattr(self.state, "save_component_package", None)
        if callable(package_saver) and not materialized_component.get("package"):
            package_saver(self.run_state.id, component_package)
        if not result.success:
            node = replace(node, status=NodeStatus.FAILED, phase=InnerPhase.INTEGRATE)
            self.nodes[node.id] = node
            self.state.save_work_node(self.run_state.id, node)
            raise NodePipelineFailed(result)
        node = replace(node, status=NodeStatus.COMPLETED, phase=InnerPhase.MEMORY_WRITEBACK)
        self.nodes[node.id] = node
        self.state.save_work_node(self.run_state.id, node)
        self.state.append_brain_entry(
            BrainEntryV1(
                BrainSection.ROLE_MEMORY,
                f"{node.id}:result",
                {
                    "summary": result.summary,
                    "artifacts": list(result.artifacts),
                    "findings": list(result.findings),
                },
                self.run_state.id,
                node_id=node.id,
                role=AgentRole.CODER,
                expires_at=_now() + timedelta(hours=self.config.role_memory_ttl_hours),
            )
        )
        self.events.publish(
            UltraEventKind.NODE.value,
            f"[{node.contract.title}] completed",
            run_id=self.run_state.id,
            node_id=node.id,
            status="completed",
        )
        return result

    def _global_gate(self) -> ResultPackageV1:
        assert self.run_state and self.goal_spec and self.architecture and self.master_plan
        node_summaries = [
            {
                "node_id": result.node_id,
                "summary": result.summary,
                "artifacts": list(result.artifacts),
                "evidence": list(result.evidence),
            }
            for result in self._results.values()
        ]
        self._set_phase(UltraPhase.INTEGRATION, "Integrating all modules")
        integration = self._invoke(
            AgentRole.INTEGRATOR,
            InnerPhase.GLOBAL_INTEGRATION,
            task={"modules": node_summaries},
            context={
                "goal_spec": asdict(self.goal_spec),
                "architecture": asdict(self.architecture),
            },
        )
        self._set_phase(UltraPhase.GLOBAL_REVIEW, "Running global review")
        review = self._invoke(
            AgentRole.REVIEWER,
            InnerPhase.GLOBAL_REVIEW,
            task={"integration": dict(integration.payload), "modules": node_summaries},
            context={"master_plan": asdict(self.master_plan)},
        )
        self._set_phase(UltraPhase.FINAL_EVIDENCE, "Checking final evidence")
        evidence = self._invoke(
            AgentRole.GOAL_CHECKER,
            InnerPhase.FINAL_EVIDENCE,
            task={
                "integration": dict(integration.payload),
                "review": dict(review.payload),
                "node_results": node_summaries,
            },
            context={"goal_spec": asdict(self.goal_spec)},
        )
        success = self._passed(integration) and self._passed(review) and self._passed(evidence)
        return ResultPackageV1(
            node_id="__global__",
            success=success,
            status="completed" if success else "revision_required",
            summary=evidence.summary or review.summary or integration.summary,
            artifacts=self._records(integration, "artifacts"),
            evidence=self._records(evidence, "evidence"),
            test_results=self._records(evidence, "test_results"),
            findings=self._findings(integration, review, evidence),
            insights=tuple(
                insight
                for response in (integration, review, evidence)
                for insight in response.insights
            ),
        )

    def _record_global_evaluation_gate(self, global_result: ResultPackageV1) -> Mapping[str, Any]:
        recorder = getattr(self.state, "record_global_evaluation_gate", None)
        if not callable(recorder):
            return {}
        return dict(
            recorder(
                global_result,
                tuple(self._results[node_id] for node_id in sorted(self._results)),
            )
            or {}
        )

    def run(self) -> UltraRunResult:
        """Expand the approved plan, execute waves, and apply global gates."""

        if not self.run_state or not self.master_plan or not self.run_state.approved:
            raise ApprovalRequiredError("approve the current ULTRA master plan before execution")
        if self.phase is not UltraPhase.AWAITING_APPROVAL:
            raise UltraError(f"cannot execute ULTRA from phase {self.phase.value}")
        schedule: ScheduleReport[ResultPackageV1] | None = None
        try:
            self._set_phase(UltraPhase.EXPANDING, "Expanding approved modules")
            for module in self.master_plan.modules:
                self._ensure_expanded(module.id)
            self.state.append_brain_entry(
                BrainEntryV1(
                    BrainSection.TASK_GRAPH,
                    "expanded_graph",
                    {"nodes": [asdict(node) for node in self.nodes.values()]},
                    self.run_state.id,
                )
            )
            self._set_phase(UltraPhase.MODULE_WAVES, "Executing module waves")
            completed_before = {
                node.id for node in self.nodes.values() if node.status is NodeStatus.COMPLETED
            }
            uncertain = [
                node.id for node in self.nodes.values() if node.status is NodeStatus.UNCERTAIN
            ]
            if uncertain:
                raise UltraError(
                    "uncertain work must be reconciled before resume: "
                    + ", ".join(sorted(uncertain))
                )
            schedulable = tuple(
                node
                for node in self.nodes.values()
                if node.status
                not in {NodeStatus.COMPLETED, NodeStatus.CANCELLED, NodeStatus.UNCERTAIN}
            )
            schedule = self.scheduler.run(
                schedulable,
                self._execute_node,
                initially_completed=completed_before,
            )
            for outcome in schedule.outcomes:
                if outcome.status is ScheduleStatus.FAILED:
                    node = self.nodes[outcome.item_id]
                    self.nodes[node.id] = replace(
                        node,
                        status=NodeStatus.FAILED,
                    )
                    self.state.save_work_node(self.run_state.id, self.nodes[node.id])
                elif outcome.status is ScheduleStatus.BLOCKED:
                    node = self.nodes[outcome.item_id]
                    self.nodes[node.id] = replace(node, status=NodeStatus.BLOCKED)
                    self.state.save_work_node(self.run_state.id, self.nodes[node.id])
                elif outcome.status is ScheduleStatus.CONFLICT:
                    node = self.nodes[outcome.item_id]
                    self.nodes[node.id] = replace(node, status=NodeStatus.CONFLICT)
                    self.state.save_work_node(self.run_state.id, self.nodes[node.id])
                elif outcome.status is ScheduleStatus.CANCELLED:
                    node = self.nodes[outcome.item_id]
                    uncertain_phases = {
                        InnerPhase.IMPLEMENT,
                        InnerPhase.FIX,
                        InnerPhase.INTEGRATE,
                        InnerPhase.MEMORY_WRITEBACK,
                    }
                    status = (
                        NodeStatus.UNCERTAIN
                        if node.phase in uncertain_phases
                        else NodeStatus.CANCELLED
                    )
                    self.nodes[node.id] = replace(node, status=status)
                    self.state.save_work_node(self.run_state.id, self.nodes[node.id])
            if not schedule.successful:
                if self.control.cancelled or any(
                    outcome.status is ScheduleStatus.CANCELLED for outcome in schedule.outcomes
                ):
                    self._set_phase(UltraPhase.CANCELLED, "ULTRA execution cancelled")
                elif any(result.status == "revision_required" for result in self._results.values()):
                    self._set_phase(UltraPhase.REVISION_REQUIRED, "Quality gate requires a revised plan")
                else:
                    self._set_phase(UltraPhase.FAILED, "One or more module waves failed")
                return UltraRunResult(
                    self.run_state,
                    self.master_plan,
                    tuple(self._results[node_id] for node_id in sorted(self._results)),
                    schedule,
                )
            global_result = self._global_gate()
            evaluation_gate = self._record_global_evaluation_gate(global_result)
            if evaluation_gate and not bool(evaluation_gate.get("passed", True)):
                global_result = replace(
                    global_result,
                    success=False,
                    status="revision_required",
                    summary=(
                        str(evaluation_gate.get("blocker") or "").strip()
                        or global_result.summary
                        or "Automatic evaluation gate requires revision"
                    ),
                    findings=tuple(
                        dict.fromkeys(
                            (
                                *global_result.findings,
                                str(evaluation_gate.get("blocker") or "automatic evaluation gate failed"),
                            )
                        )
                    ),
                    evidence=(
                        *global_result.evidence,
                        {
                            "kind": "automatic_evaluation_gate",
                            "metrics": dict(evaluation_gate.get("metrics", {})),
                            "scores": dict(evaluation_gate.get("scores", {})),
                            "benchmark_id": evaluation_gate.get("benchmark_id"),
                        },
                    ),
                )
            self.state.save_result_package(self.run_state.id, global_result)
            if not global_result.success:
                self._set_phase(UltraPhase.REVISION_REQUIRED, "Final evidence gate requires revision")
            else:
                self._set_phase(UltraPhase.COMPLETED, "ULTRA goal completed")
                self.events.publish(
                    UltraEventKind.COMPLETED.value,
                    "All modules and final evidence gates completed",
                    run_id=self.run_state.id,
                    nodes=len(self.nodes),
                )
            return UltraRunResult(
                self.run_state,
                self.master_plan,
                tuple(self._results[node_id] for node_id in sorted(self._results)),
                schedule,
                global_result,
            )
        except ScopeRevisionRequired as exc:
            self._set_phase(UltraPhase.REVISION_REQUIRED, str(exc))
            self.events.publish(
                UltraEventKind.REVISION_REQUIRED.value,
                str(exc),
                run_id=self.run_state.id,
            )
            return UltraRunResult(
                self.run_state,
                self.master_plan,
                tuple(self._results.values()),
                schedule,
            )
        except CancellationRequested:
            self._set_phase(UltraPhase.CANCELLED, "ULTRA execution cancelled")
            self.events.publish(
                UltraEventKind.CANCELLED.value,
                "ULTRA execution cancelled at a safe checkpoint",
                run_id=self.run_state.id,
            )
            return UltraRunResult(
                self.run_state,
                self.master_plan,
                tuple(self._results.values()),
                schedule,
            )

    def start_background(self):
        return self.background.start(self.run)

    def pause(self) -> None:
        if self.run_state and self.phase not in {UltraPhase.COMPLETED, UltraPhase.CANCELLED}:
            if self.phase is not UltraPhase.PAUSED:
                self._phase_before_pause = self.phase
                self._set_phase(UltraPhase.PAUSED, "ULTRA paused")
            self.control.pause()
            self.events.publish("ultra.paused", "ULTRA will pause at the next safe checkpoint")

    def resume(self) -> None:
        if self.phase is UltraPhase.PAUSED and self._phase_before_pause is not None:
            target = self._phase_before_pause
            self._phase_before_pause = None
            self._set_phase(target, "ULTRA execution resumed")
        self.control.resume()
        self.events.publish("ultra.resumed", "ULTRA execution resumed")

    def cancel(self) -> None:
        self.control.cancel()
        self.events.publish("ultra.cancelling", "ULTRA will cancel at the next safe checkpoint")


__all__ = [
    "AgentProtocolError",
    "AgentRequest",
    "AgentResponse",
    "AgentRole",
    "AgentRunV1",
    "ApprovalMismatchError",
    "ApprovalRequiredError",
    "ArchitectureSpecV1",
    "BrainEntryV1",
    "BrainSection",
    "ContextBuilder",
    "ContextRequest",
    "ComponentPackageV1",
    "ExecutionClass",
    "FocusedContextBuilder",
    "GoalSpecV1",
    "InMemoryUltraState",
    "InnerPhase",
    "InsightV1",
    "JournaledUltraState",
    "MasterPlanV1",
    "LeafReadinessV1",
    "NodeKind",
    "NodeStatus",
    "NodeQualityTargetV1",
    "PromptTraceV1",
    "ProviderAgentAdapter",
    "ProviderFactoryAdapter",
    "QualityGateResultV1",
    "ResultPackageV1",
    "ScopeRevisionRequired",
    "SpecialistProfileV1",
    "TaskContractV1",
    "UltraAgent",
    "UltraAgentFactory",
    "UltraConfig",
    "UltraError",
    "UltraEventKind",
    "UltraOrchestrator",
    "UltraPhase",
    "UltraRunResult",
    "UltraRunV1",
    "UltraStateAdapter",
    "WorkNode",
]
