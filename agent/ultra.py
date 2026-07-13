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
        legacy_ids: dict[str, str] = {}
        for index, item in enumerate(raw_modules, start=1):
            generated = f"M{index:03d}"
            raw_id = str(item.get("id", "")).strip()
            if raw_id:
                legacy_ids[raw_id.casefold()] = generated
            legacy_ids[f"m{index}".casefold()] = generated
            legacy_ids[f"m{index:02d}".casefold()] = generated
            legacy_ids[generated.casefold()] = generated

        normalized_modules: list[dict[str, Any]] = []
        for index, item in enumerate(raw_modules, start=1):
            normalized = dict(item)
            normalized["id"] = f"M{index:03d}"
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
                    resolved = f"M{int(match.group(1)):03d}" if match else text
                if resolved and resolved not in dependencies:
                    dependencies.append(resolved)
            normalized["depends_on"] = dependencies
            normalized_modules.append(normalized)
        modules = tuple(
            TaskContractV1.from_mapping(item, fallback_id=f"M{index:03d}")
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
        payload = value.get("payload", value)
        return cls(
            payload=_mapping(payload),
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


class InMemoryUltraState:
    """Thread-safe reference adapter used by tests and incremental integration."""

    def __init__(self) -> None:
        self.runs: dict[str, UltraRunV1] = {}
        self.nodes: dict[str, dict[str, WorkNode]] = defaultdict(dict)
        self.agent_runs: list[AgentRunV1] = []
        self.traces: list[PromptTraceV1] = []
        self.results: dict[str, dict[str, ResultPackageV1]] = defaultdict(dict)
        self.brain: list[BrainEntryV1] = []
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
    candidate = str(text or "").strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.IGNORECASE)
        candidate = re.sub(r"\s*```$", "", candidate)
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise AgentProtocolError("ULTRA agents must return one JSON object") from exc
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
    max_depth: int = 5
    max_nodes: int = 500
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
                )
            )
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
                )
                self.state.save_agent_run(agent_run)
                omitted = safe_context.get("_omitted", ())
                trace = PromptTraceV1(
                    run_id=self.run_state.id,
                    role=role,
                    phase=phase_value,
                    system_prompt=redact_text(system)[:8_000],
                    context_package=safe_context,
                    self_prompt=redact_text(_json(safe_task))[:16_000],
                    reasoning_summary=redact_text(response.reasoning_summary)[:4_000],
                    node_id=node_id,
                    agent_run_id=agent_id,
                    omitted_context=_strings(omitted),
                )
                self.state.save_prompt_trace(trace)
                agent_run = replace(agent_run, prompt_trace_id=trace.id)
                self.state.save_agent_run(agent_run)
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
                if typed_repair_used:
                    raise AgentProtocolError(
                        f"ULTRA foundation/phase {phase_value} failed after one targeted typed-return repair: {exc}"
                    ) from exc
                typed_repair_used = True
                repair_task = {
                    **safe_task,
                    "typed_return_repair": {
                        "contract": phase_value,
                        "errors": [str(exc)],
                        "previous_payload": redact_data(dict(response.payload)),
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
        raw = getter(self.run_state.id, query, phase=phase)
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            return ()
        return tuple(dict(item) for item in raw if isinstance(item, Mapping))

    @staticmethod
    def _normalize_typed_payload(
        phase: str,
        payload: Mapping[str, Any],
        task: Mapping[str, Any],
    ) -> tuple[dict[str, Any], tuple[str, ...]]:
        normalized = dict(payload)
        actions: list[str] = []
        if phase == "goal_spec" and not str(normalized.get("objective", "")).strip():
            authoritative_prompt = str(task.get("prompt", "")).strip()
            if authoritative_prompt:
                normalized["objective"] = authoritative_prompt
                actions.append("goal_spec.objective restored from authoritative user prompt")
        if phase == "master_plan":
            goal_spec = _mapping(task.get("goal_spec"))
            architecture = _mapping(task.get("architecture"))
            modules = [dict(item) for item in normalized.get("modules", ()) if isinstance(item, Mapping)]
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
                if modules:
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
            requires_browser_qa = any(
                term in objective_text
                for term in ("browser", "screenshot", "visual refinement", "visual quality")
            )
            dedicated_qa_terms = ("browser qa", "visual qa", "quality gate", "validation gate")
            has_dedicated_browser_qa = any(
                any(term in str(item.get("title", "")).casefold() for term in dedicated_qa_terms)
                for item in modules
            )
            # A screenshot mentioned inside an implementation module is not an
            # independent quality gate: weak planners commonly use it as a
            # checkbox and omit interaction coverage, error evidence, and the
            # mandatory refine-and-retest loop.
            if requires_browser_qa and not has_dedicated_browser_qa:
                qa_id = f"M{len(modules) + 1:03d}"
                modules.append(
                    {
                        "id": qa_id,
                        "title": "Browser QA and Visual Refinement Gate",
                        "objective": "Run the real game in a browser, prove gameplay and runtime stability, score visual quality, and refine any below-target candidate.",
                        "acceptance_criteria": [
                            "A real 1280x720 screenshot is captured from the running game, never a placeholder.",
                            "Console and page errors are zero and unexpected network failures are absent or handled by a visible graceful fallback.",
                            "Movement, shooting, pause, pickups, wave progression, game-over, and restart are exercised.",
                            "Visual hierarchy, depth, contrast, density, legibility, and responsiveness meet the approved quality target.",
                            "Any below-target visual dimension triggers code refinement and fresh browser verification.",
                        ],
                        "verification": [
                            "Preview index.html at 1280x720 and capture the actual browser screenshot.",
                            "Inspect console, page, and network evidence and exercise the complete gameplay lifecycle.",
                            "Repeat screenshot and runtime verification after every visual refinement mutation.",
                        ],
                        "depends_on": [str(modules[-1].get("id"))] if modules else [],
                        "write_paths": ["index.html"],
                    }
                )
                normalized["modules"] = modules
                actions.append("master_plan Browser QA gate added from explicit goal requirements")
        if phase == InnerPhase.DECOMPOSE.value:
            parent_contract = _mapping(task.get("contract"))
            parent_id = str(parent_contract.get("id", "")).strip()
            parent_title = str(parent_contract.get("title", parent_id or "Parent module")).strip()
            raw_children = normalized.get("children", ())
            if isinstance(raw_children, Sequence) and not isinstance(raw_children, (str, bytes)):
                repaired_children: list[Any] = []
                for index, child in enumerate(raw_children, start=1):
                    if not isinstance(child, Mapping):
                        repaired_children.append(child)
                        continue
                    item = dict(child)
                    child_id = str(item.get("id", f"{parent_id}.{index}" if parent_id else f"child.{index}")).strip()
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
            if not question_id or question_id in seen or not text:
                raise AgentProtocolError("ULTRA questions require unique ids and non-empty text")
            if not options:
                # Open-ended questions without bounded choices are almost
                # always weak-model uncertainty, not a real user decision.
                # The architecture and implementation passes should pick a
                # safe default and continue autonomously.
                continue
            if len(options) == 1:
                # A single offered option is not a user decision. Treat it as
                # the model's own bounded default and continue foundation.
                continue
            if options and not 2 <= len(options) <= 3:
                raise AgentProtocolError("ULTRA question options must contain two or three choices")
            seen.add(question_id)
            questions.append(
                {
                    "id": question_id,
                    "header": str(item.get("header", question_id)).strip()[:40],
                    "question": text[:1_000],
                    "options": options,
                    "allow_freeform": bool(item.get("allow_freeform", True)),
                    "reason": str(
                        item.get("reason", "Required to finalize the master plan.")
                    ).strip()[:1_000],
                }
            )
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

    def _finish_foundation(self, prompt: str) -> MasterPlanV1:
        """Continue Architecture -> Master Plan after goal decisions are complete."""

        assert self.run_state and self.goal_spec
        self._set_phase(UltraPhase.ARCHITECTURE, "Designing architecture")
        architecture_response = self._invoke(
            AgentRole.ARCHITECT,
            "architecture",
            task={"goal_spec": asdict(self.goal_spec)},
            context={
                "prompt": prompt,
                "cross_run_project_lessons": self._foundation_project_lessons(prompt, "architecture"),
            },
        )
        self.architecture = ArchitectureSpecV1.from_mapping(architecture_response.payload)
        self._set_phase(UltraPhase.MASTER_PLAN, "Building master plan")
        plan_response = self._invoke(
            AgentRole.PLANNER,
            "master_plan",
            task={
                "goal_spec": asdict(self.goal_spec),
                "architecture": asdict(self.architecture),
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
        proposed = MasterPlanV1.from_mapping(plan_response.payload)
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
            self.goal_spec = GoalSpecV1.from_mapping(goal_response.payload)
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
        normalized = {str(key): str(value).strip() for key, value in answers.items()}
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
    def _contains_scope(parent: str, child: str) -> bool:
        p = parent.replace("\\", "/").rstrip("/") or "."
        c = child.replace("\\", "/").rstrip("/") or "."
        if p in {".", "*", "**", "**/*"}:
            return True
        if any(char in p for char in "*?["):
            import fnmatch

            return fnmatch.fnmatchcase(c, p)
        return c == p or c.startswith(p + "/")

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
            child = replace(
                child,
                depends_on=inherited_dependencies,
                forbidden_changes=inherited_forbidden,
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
        node = replace(node, status=NodeStatus.PLANNING, phase=InnerPhase.CONTEXT)
        self.nodes[node_id] = node
        self.state.save_work_node(self.run_state.id, node)
        context = self._new_context(node, AgentRole.PLANNER)
        plan_response = self._invoke(
            AgentRole.PLANNER,
            InnerPhase.MINI_PLAN,
            task={"contract": asdict(node.contract)},
            context=context,
            node_id=node.id,
        )
        decompose_response = self._invoke(
            AgentRole.DECOMPOSER,
            InnerPhase.DECOMPOSE,
            task={
                "contract": asdict(node.contract),
                "mini_plan": dict(plan_response.payload),
                "remaining_node_budget": self.config.max_nodes - len(self.nodes),
            },
            context=context,
            node_id=node.id,
        )
        children = self._validated_children(node, decompose_response.payload.get("children", ()))
        if len(self.nodes) + len(children) > self.config.max_nodes:
            raise ScopeRevisionRequired(
                f"dynamic expansion exceeds max node count {self.config.max_nodes}"
            )
        self._prepared[node.id] = (context, dict(plan_response.payload))
        self._research_required[node.id] = bool(
            plan_response.payload.get("research_required")
            or decompose_response.payload.get("research_required")
            or node.contract.metadata.get("research_required")
        )
        if children:
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
        labels = (
            "clean_code",
            "security",
            "runtime_tests",
            "test_quality",
            "triage",
        )
        records: list[Mapping[str, Any]] = []
        for label, response in zip(labels, responses):
            payload = response.payload
            reasoning_evaluation = _mapping(payload.get("harness_reasoning_evaluation"))
            raw_verdict = str(payload.get("consensus_vote") or payload.get("verdict") or "").strip().casefold()
            if raw_verdict not in {"accept", "reject", "abstain"}:
                raw_verdict = "accept" if self._passed(response) else "reject"
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
                    "passed": self._passed(response),
                    "summary": response.summary,
                    "rationale": response.reasoning_summary or response.summary,
                    "evidence": {
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

    def _quality(self, node: WorkNode) -> QualityGateResultV1:
        clean_code = self._invoke(
            AgentRole.CLEAN_CODE_REVIEWER,
            InnerPhase.REVIEW,
            task={"contract": asdict(node.contract), "fresh_review": True, "category": "clean_code", "read_only": True},
            context=self._new_context(node, AgentRole.CLEAN_CODE_REVIEWER),
            node_id=node.id,
        )
        security = self._invoke(
            AgentRole.SECURITY_REVIEWER,
            InnerPhase.REVIEW,
            task={"contract": asdict(node.contract), "fresh_review": True, "category": "security", "read_only": True},
            context=self._new_context(node, AgentRole.SECURITY_REVIEWER),
            node_id=node.id,
        )
        tests = self._invoke(
            AgentRole.TESTER,
            InnerPhase.TEST,
            task={"contract": asdict(node.contract), "fresh_test_context": True},
            context=self._new_context(node, AgentRole.TESTER),
            node_id=node.id,
        )
        test_quality = self._invoke(
            AgentRole.TEST_QUALITY_REVIEWER,
            InnerPhase.REVIEW,
            task={"contract": asdict(node.contract), "fresh_review": True, "category": "test_quality", "read_only": True},
            context=self._new_context(node, AgentRole.TEST_QUALITY_REVIEWER),
            node_id=node.id,
        )
        triage = self._invoke(
            AgentRole.QUALITY_TRIAGER,
            InnerPhase.REVIEW,
            task={
                "contract": asdict(node.contract),
                "read_only": True,
                "normalize_and_deduplicate": True,
                "review_summaries": [clean_code.summary, security.summary, tests.summary, test_quality.summary],
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
        responses = (clean_code, security, tests, test_quality, triage)
        consensus = self._record_quality_consensus(node, responses)
        return QualityGateResultV1(responses=responses, consensus=consensus)

    def _execute_node(self, scheduled_node: WorkNode) -> ResultPackageV1:
        assert self.run_state
        node = self.nodes[scheduled_node.id]
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
        node = replace(node, phase=InnerPhase.IMPLEMENT)
        self.nodes[node.id] = node
        self.state.save_work_node(self.run_state.id, node)
        implementation = self._invoke(
            AgentRole.CODER,
            InnerPhase.IMPLEMENT,
            task={
                "contract": asdict(node.contract),
                "mini_plan": dict(self._prepared.get(node.id, ({}, {}))[1]),
            },
            context=self._new_context(node, AgentRole.CODER),
            node_id=node.id,
        )
        responses.append(implementation)
        quality_gate = self._quality(node)
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
            fix = self._invoke(
                AgentRole.CODER,
                InnerPhase.FIX,
                task={
                    "contract": asdict(node.contract),
                    "findings": findings,
                    "attempt": fixes,
                    "change_approach": fixes == self.config.max_fix_attempts,
                },
                context=self._new_context(node, AgentRole.CODER),
                node_id=node.id,
            )
            responses.append(fix)
            quality_gate = self._quality(node)
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
            )
            self._results[node.id] = result
            self.state.save_result_package(self.run_state.id, result)
            node = replace(node, status=NodeStatus.REVISION_REQUIRED, phase=InnerPhase.REPLAN)
            self.nodes[node.id] = node
            self.state.save_work_node(self.run_state.id, node)
            raise NodePipelineFailed(result)

        node = replace(node, phase=InnerPhase.INTEGRATE)
        self.nodes[node.id] = node
        self.state.save_work_node(self.run_state.id, node)
        integration = self._invoke(
            AgentRole.INTEGRATOR,
            InnerPhase.INTEGRATE,
            task={"contract": asdict(node.contract)},
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
        result = ResultPackageV1(
            node_id=node.id,
            success=self._passed(integration),
            status="completed" if self._passed(integration) else "failed",
            summary=integration.summary or implementation.summary or f"{node.id} completed",
            artifacts=tuple(
                item for response in responses for item in self._records(response, "artifacts")
            ),
            evidence=tuple(
                item for response in responses for item in self._records(response, "evidence")
            ),
            test_results=tuple(
                item for response in responses for item in self._records(response, "test_results")
            ),
            findings=self._findings(*responses),
            insights=tuple(insight for response in responses for insight in response.insights),
            fix_attempts=fixes,
        )
        self._results[node.id] = result
        self.state.save_result_package(self.run_state.id, result)
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
                if outcome.status is ScheduleStatus.BLOCKED:
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
    "ExecutionClass",
    "FocusedContextBuilder",
    "GoalSpecV1",
    "InMemoryUltraState",
    "InnerPhase",
    "InsightV1",
    "JournaledUltraState",
    "MasterPlanV1",
    "NodeKind",
    "NodeStatus",
    "PromptTraceV1",
    "ProviderAgentAdapter",
    "ProviderFactoryAdapter",
    "QualityGateResultV1",
    "ResultPackageV1",
    "ScopeRevisionRequired",
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
