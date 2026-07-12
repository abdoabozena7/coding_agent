"""Runnable integration between ULTRA orchestration and the v3 state store.

The provider-neutral engine is intentionally independent from the legacy
goal/plan runtime.  This module supplies the concrete adapters needed by the
CLI: real workspace tools, Docker-only Full shell access, durable v3 records,
legacy master-plan approval, file hashes, and resource leases.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import re
import threading
from concurrent.futures import Future
from dataclasses import asdict, replace
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Mapping, Sequence

from . import tools
from .events import EventBus
from .model_catalog import ExecutionClass, ModelDescriptor
from .models import DomainError, GoalStatus, Plan, RoleProfile, TaskStatus
from .providers.base import AssistantTurn, ToolCall
from .project_brain import ProjectBrain
from .safety import redact_data, redact_text
from .sandbox import AccessLevel, PermissionAdapter
from .scheduler import ResourceLease as RuntimeLease
from .scheduler import AdaptiveConcurrency, RateLimitError, ResourceLeaseManager, StaleWriteError
from .store import NotFoundError, StateStore, StateStoreError
from .ultra import (
    AgentRequest,
    AgentResponse,
    AgentRole,
    ArchitectureSpecV1 as EngineArchitectureSpec,
    BrainEntryV1,
    BrainSection as EngineBrainSection,
    GoalSpecV1 as EngineGoalSpec,
    ContextRequest,
    FocusedContextBuilder,
    InMemoryUltraState,
    InnerPhase,
    MasterPlanV1,
    NodeKind,
    NodeStatus,
    PromptTraceV1 as EnginePromptTrace,
    ResultPackageV1 as EngineResult,
    UltraConfig,
    UltraOrchestrator,
    UltraPhase as EnginePhase,
    UltraRunResult,
    UltraRunV1,
    WorkNode as EngineWorkNode,
    _extract_json,
)
from .ultra_models import (
    AgentRun,
    AgentRunStatus,
    ArchitectureSpecV1,
    Artifact,
    BrainEntry,
    BrainSection,
    GoalSpecV1,
    InsightV1,
    PromptTraceV1,
    ResultPackageV1,
    TaskContractV1,
    UltraPhase,
    UltraRun,
    UltraRunStatus,
    WorkNode,
    WorkNodeKind,
    WorkNodeStatus,
)


_READ_TOOLS = frozenset({"read_file", "list_files", "grep"})
_WRITE_TOOLS = frozenset({"write_file", "edit_file", "run_bash"})
_TOOL_RISK = {
    "read_file": "low",
    "list_files": "low",
    "grep": "low",
    "write_file": "high",
    "edit_file": "high",
    "run_bash": "critical",
}


def _schema_name(schema: Mapping[str, Any]) -> str:
    return str(schema.get("function", {}).get("name", ""))


def _schemas(names: Iterable[str]) -> list[dict[str, Any]]:
    wanted = set(names)
    return [schema for schema in tools.TOOL_SCHEMAS if _schema_name(schema) in wanted]


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _hash_file(workspace: Path, value: str) -> str | None:
    parts = PurePosixPath(_normalized_path(value)).parts
    candidate = workspace.joinpath(*parts).resolve(strict=False)
    try:
        candidate.relative_to(workspace)
    except ValueError:
        return None
    if not candidate.is_file():
        return None
    digest = hashlib.sha256()
    try:
        with candidate.open("rb") as handle:
            for chunk in iter(lambda: handle.read(64 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def _normalized_path(value: str) -> str:
    text = str(value or ".").strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return str(PurePosixPath(text or ".")).rstrip("/") or "."


def _within_scope(path: str, scopes: Iterable[str]) -> bool:
    candidate = _normalized_path(path).casefold()
    for raw in scopes:
        scope = _normalized_path(raw).casefold()
        if scope in {".", "*", "**", "**/*"}:
            return True
        if any(character in scope for character in "*?["):
            if fnmatch.fnmatchcase(candidate, scope):
                return True
            continue
        if candidate == scope or candidate.startswith(scope + "/"):
            return True
    return False


def _snapshot_write_scopes(
    workspace: Path,
    scopes: Iterable[str],
    *,
    max_files: int = 50_000,
) -> dict[str, str | None]:
    """Capture file hashes protected by a node's write lease."""

    root = workspace.resolve()
    snapshot: dict[str, str | None] = {}
    seen: set[Path] = set()

    def remember(candidate: Path) -> None:
        resolved = candidate.resolve(strict=False)
        try:
            relative = resolved.relative_to(root)
        except ValueError:
            return
        if not resolved.is_file() or ".coding-agent" in relative.parts:
            return
        if resolved in seen:
            return
        if len(seen) >= max_files:
            raise RuntimeError(
                "write-scope snapshot exceeded 50000 files; narrow the module write_paths"
            )
        seen.add(resolved)
        key = _normalized_path(relative.as_posix())
        snapshot[key] = _hash_file(root, key)

    for raw_scope in scopes:
        scope = _normalized_path(raw_scope)
        if scope in {".", "*", "**", "**/*"}:
            for candidate in root.rglob("*"):
                remember(candidate)
            continue
        if any(character in scope for character in "*?["):
            for candidate in root.glob(scope):
                if candidate.is_dir():
                    for child in candidate.rglob("*"):
                        remember(child)
                else:
                    remember(candidate)
            continue
        candidate = root.joinpath(*PurePosixPath(scope).parts)
        if candidate.is_dir():
            for child in candidate.rglob("*"):
                remember(child)
        elif candidate.is_file():
            remember(candidate)
        else:
            snapshot.setdefault(scope, None)
    return snapshot


_PHASE_CONTRACTS: dict[str, Mapping[str, Any]] = {
    "goal_spec": {
        "payload": {
            "objective": "string",
            "success_criteria": ["observable criterion"],
            "constraints": ["constraint"],
            "in_scope": ["scope item"],
            "out_of_scope": ["non-goal"],
            "assumptions": ["assumption"],
            "questions": [
                {
                    "id": "stable id",
                    "header": "short label",
                    "question": "non-discoverable decision only",
                    "options": [
                        {"label": "choice", "description": "impact", "recommended": True}
                    ],
                    "allow_freeform": True,
                    "reason": "why this cannot be discovered",
                }
            ],
        }
    },
    "architecture": {
        "payload": {
            "summary": "architecture summary",
            "components": [{"name": "component", "responsibility": "..."}],
            "interfaces": [{"name": "interface", "contract": "..."}],
            "decisions": [{"decision": "...", "reason": "...", "alternatives": []}],
            "dependencies": [],
            "invariants": [],
        }
    },
    "master_plan": {
        "payload": {
            "summary": "complete master plan",
            "execution_strategy": "dependency waves and integration strategy",
            "milestones": [],
            "modules": [
                {
                    "id": "M001",
                    "title": "module title",
                    "objective": "bounded objective",
                    "acceptance_criteria": ["observable criterion"],
                    "verification": ["command or inspection"],
                    "depends_on": [],
                    "write_paths": ["workspace/relative/path"],
                    "forbidden_changes": [],
                    "owned_interfaces": [],
                    "metadata": {"external_dependencies": []},
                }
            ],
        }
    },
    InnerPhase.MINI_PLAN.value: {
        "payload": {"steps": ["step"], "research_required": False}
    },
    InnerPhase.DECOMPOSE.value: {
        "payload": {
            "children": [],
            "research_required": False,
        }
    },
    InnerPhase.REVIEW.value: {
        "payload": {"passed": True, "issues": [], "findings": [], "evidence": []}
    },
    InnerPhase.TEST.value: {
        "payload": {"passed": True, "issues": [], "test_results": [], "evidence": []}
    },
    InnerPhase.IMPLEMENT.value: {
        "payload": {"success": True, "artifacts": [], "evidence": [], "findings": []}
    },
    InnerPhase.FIX.value: {
        "payload": {"success": True, "artifacts": [], "evidence": [], "findings": []}
    },
    InnerPhase.INTEGRATE.value: {
        "payload": {"passed": True, "artifacts": [], "evidence": [], "findings": []}
    },
    InnerPhase.GLOBAL_INTEGRATION.value: {
        "payload": {"passed": True, "artifacts": [], "evidence": [], "findings": []}
    },
    InnerPhase.GLOBAL_REVIEW.value: {
        "payload": {"passed": True, "issues": [], "findings": [], "evidence": []}
    },
    InnerPhase.FINAL_EVIDENCE.value: {
        "payload": {"passed": True, "evidence": [], "test_results": [], "findings": []}
    },
}


class WorkspaceUltraAgent:
    """One isolated role conversation with real, policy-controlled tools."""

    def __init__(
        self,
        provider: Any,
        *,
        role: AgentRole,
        provider_name: str,
        model: str,
        executor: Callable[[ToolCall, AgentRequest], str],
        events: EventBus,
        max_steps: int = 16,
    ) -> None:
        self.provider = provider
        self.role = role
        self.provider_name = provider_name
        self.model = model
        self.executor = executor
        self.events = events
        self.max_steps = max(2, int(max_steps))

    def _allowed_tools(self) -> frozenset[str]:
        if self.role in {AgentRole.CODER, AgentRole.INTEGRATOR}:
            return _READ_TOOLS | _WRITE_TOOLS
        if self.role in {AgentRole.TESTER, AgentRole.RESEARCHER}:
            return _READ_TOOLS | {"run_bash"}
        return _READ_TOOLS

    def execute(self, request: AgentRequest) -> AgentResponse:
        contract = _PHASE_CONTRACTS.get(
            request.phase,
            {"payload": {"success": True, "findings": [], "evidence": []}},
        )
        user_payload = {
            "task": request.task,
            "focused_context": request.context,
            "response_contract": {
                **contract,
                "summary": "brief factual result summary",
                "reasoning_summary": (
                    "brief conclusion, decisions, and evidence only; never hidden chain-of-thought"
                ),
                "insights": [
                    {
                        "summary": "durable insight",
                        "severity": "info|warning|error",
                        "details": {},
                    }
                ],
            },
        }
        conversation: list[dict[str, Any]] = [
            {"role": "user", "content": _json(user_payload)}
        ]
        schemas = _schemas(self._allowed_tools())
        totals = {"input_tokens": 0, "cached_tokens": 0, "output_tokens": 0}
        inspection_observed = False
        self.events.publish(
            "ultra.agent_started",
            f"[{self.role.value}] {request.phase}",
            run_id=request.run_id,
            node_id=request.node_id,
            role=self.role.value,
            phase=request.phase,
        )
        last_error: Exception | None = None
        for step in range(1, self.max_steps + 1):
            try:
                turn = self.provider.call(conversation, schemas, request.system_prompt)
            except Exception as exc:
                status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
                if status == 429 or "rate limit" in str(exc).casefold():
                    raise RateLimitError(str(exc)) from exc
                raise
            if not isinstance(turn, AssistantTurn):
                raise TypeError("ULTRA provider returned an invalid turn")
            if turn.usage:
                for key in totals:
                    totals[key] += int(getattr(turn.usage, key, 0) or 0)
            conversation.append(turn.to_message())
            if turn.tool_calls:
                for call in turn.tool_calls:
                    if call.name in _READ_TOOLS:
                        inspection_observed = True
                    result = self.executor(call, request)
                    conversation.append(
                        {
                            "role": "tool",
                            "id": call.id,
                            "name": call.name,
                            "content": result,
                        }
                    )
                continue
            try:
                data = _extract_json(str(turn.text or ""))
                response = AgentResponse.from_mapping(
                    data,
                    node_id=request.node_id,
                    provider=self.provider_name,
                    model=self.model,
                    usage=totals,
                )
                if request.phase == "goal_spec" and not inspection_observed:
                    last_error = RuntimeError(
                        "GoalSpecV1 requires repository inspection before questions or planning"
                    )
                    conversation.append(
                        {
                            "role": "user",
                            "content": (
                                "Inspect the workspace with an available read tool before "
                                "returning GoalSpecV1. Do not ask for facts the repository can answer."
                            ),
                        }
                    )
                    continue
                return response
            except Exception as exc:
                last_error = exc
                conversation.append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous response was not the single JSON object required by "
                            "response_contract. Return the corrected JSON object now."
                        ),
                    }
                )
        raise RuntimeError(
            f"{self.role.value} did not produce a valid structured result after "
            f"{self.max_steps} steps: {last_error or 'tool loop exhausted'}"
        )


class WorkspaceUltraAgentFactory:
    def __init__(
        self,
        descriptor: ModelDescriptor,
        executor: Callable[[ToolCall, AgentRequest], str],
        events: EventBus,
        *,
        max_steps: int,
    ) -> None:
        self.descriptor = descriptor
        self.executor = executor
        self.events = events
        self.max_steps = max_steps

    def create(
        self,
        role: AgentRole,
        *,
        run_id: str,
        node_id: str | None = None,
    ) -> WorkspaceUltraAgent:
        del run_id, node_id
        return WorkspaceUltraAgent(
            self.descriptor.create_provider(),
            role=role,
            provider_name=self.descriptor.provider,
            model=self.descriptor.model,
            executor=self.executor,
            events=self.events,
            max_steps=self.max_steps,
        )


class DurableContextBuilder:
    """Prefer SQLite/FTS retrieval and fall back during not-yet-flushed expansion."""

    def __init__(self, store: StateStore, run_id: Callable[[], str | None], max_chars: int) -> None:
        self.store = store
        self.run_id = run_id
        self.max_chars = max_chars
        self.fallback = FocusedContextBuilder(max_chars)

    def build(self, request: ContextRequest) -> Mapping[str, Any]:
        run_id = self.run_id()
        if not run_id:
            return self.fallback.build(request)
        try:
            package = ProjectBrain(self.store, run_id).build_context(
                request.node.id,
                request.role.value,
                query=request.node.contract.objective,
                budget_chars=self.max_chars,
            )
        except (StateStoreError, DomainError):
            return self.fallback.build(request)
        sections = dict(package.sections)
        sections.setdefault("north_star", asdict(request.goal))
        sections.setdefault(
            "architecture_contract",
            {
                "summary": request.architecture.summary,
                "interfaces": list(request.architecture.interfaces),
                "invariants": list(request.architecture.invariants),
            },
        )
        sections["_omitted"] = list(package.omitted_sections)
        return sections


def _store_phase(phase: EnginePhase) -> UltraPhase:
    return {
        EnginePhase.NEW: UltraPhase.GOAL_INTERVIEW,
        EnginePhase.GOAL_SPEC: UltraPhase.GOAL_SPEC,
        EnginePhase.AWAITING_QUESTIONS: UltraPhase.GOAL_SPEC,
        EnginePhase.ARCHITECTURE: UltraPhase.ARCHITECTURE,
        EnginePhase.MASTER_PLAN: UltraPhase.MASTER_PLAN,
        EnginePhase.AWAITING_APPROVAL: UltraPhase.AWAITING_APPROVAL,
        EnginePhase.EXPANDING: UltraPhase.MODULE_WAVES,
        EnginePhase.MODULE_WAVES: UltraPhase.MODULE_WAVES,
        EnginePhase.INTEGRATION: UltraPhase.INTEGRATION,
        EnginePhase.GLOBAL_REVIEW: UltraPhase.GLOBAL_REVIEW,
        EnginePhase.FINAL_EVIDENCE: UltraPhase.EVIDENCE_GATE,
        EnginePhase.COMPLETED: UltraPhase.COMPLETED,
    }.get(phase, UltraPhase.MODULE_WAVES)


def _store_run_status(phase: EnginePhase) -> UltraRunStatus:
    if phase is EnginePhase.AWAITING_APPROVAL:
        return UltraRunStatus.AWAITING_APPROVAL
    if phase is EnginePhase.AWAITING_QUESTIONS:
        return UltraRunStatus.PAUSED
    if phase is EnginePhase.PAUSED:
        return UltraRunStatus.PAUSED
    if phase is EnginePhase.REVISION_REQUIRED:
        return UltraRunStatus.REVISION_REQUIRED
    if phase is EnginePhase.CANCELLED:
        return UltraRunStatus.CANCELLED
    if phase is EnginePhase.FAILED:
        return UltraRunStatus.BLOCKED
    if phase is EnginePhase.COMPLETED:
        return UltraRunStatus.COMPLETED
    if phase in {
        EnginePhase.EXPANDING,
        EnginePhase.MODULE_WAVES,
        EnginePhase.INTEGRATION,
        EnginePhase.GLOBAL_REVIEW,
        EnginePhase.FINAL_EVIDENCE,
    }:
        return UltraRunStatus.RUNNING
    return UltraRunStatus.DRAFT


def _store_node_status(status: NodeStatus) -> WorkNodeStatus:
    return {
        NodeStatus.PENDING: WorkNodeStatus.PENDING,
        NodeStatus.PLANNING: WorkNodeStatus.IN_PROGRESS,
        NodeStatus.READY: WorkNodeStatus.READY,
        NodeStatus.RUNNING: WorkNodeStatus.IN_PROGRESS,
        NodeStatus.COMPLETED: WorkNodeStatus.COMPLETED,
        NodeStatus.FAILED: WorkNodeStatus.FAILED,
        NodeStatus.BLOCKED: WorkNodeStatus.BLOCKED,
        NodeStatus.CONFLICT: WorkNodeStatus.CONFLICT,
        NodeStatus.CANCELLED: WorkNodeStatus.CANCELLED,
        NodeStatus.UNCERTAIN: WorkNodeStatus.UNCERTAIN,
        NodeStatus.REVISION_REQUIRED: WorkNodeStatus.REVISION_REQUIRED,
    }[status]


def _store_kind(kind: NodeKind) -> WorkNodeKind:
    return WorkNodeKind(kind.value)


def _safe_task_id(value: str, index: int, used: set[str]) -> str:
    base = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value).upper()).strip("-._")
    base = (base or f"M{index:03d}")[:24]
    candidate = base
    suffix = 2
    while candidate in used:
        tail = f"-{suffix}"
        candidate = base[: 24 - len(tail)] + tail
        suffix += 1
    used.add(candidate)
    return candidate


class StateStoreUltraAdapter(InMemoryUltraState):
    """Mirror live engine state into the durable v3 schema."""

    def __init__(
        self,
        store: StateStore,
        goal_id: str,
        descriptor: ModelDescriptor,
        access_level: AccessLevel,
        config: UltraConfig,
    ) -> None:
        super().__init__()
        self.store = store
        self.goal_id = goal_id
        self.descriptor = descriptor
        self.access_level = access_level
        self.config = config
        self.run_id: str | None = None
        self.plan: Plan | None = None
        self.approved = False
        self.task_ids: dict[str, str] = {}
        self._pending_nodes: dict[str, EngineWorkNode] = {}
        self._result_cache: dict[str, EngineResult] = {}
        self._persisted_nodes: set[str] = set()
        self._persisted_agents: set[str] = set()
        self._persisted_traces: set[str] = set()
        self._pending_artifacts: list[Artifact] = []
        self._brain_results: set[str] = set()
        self._lease_ids: dict[str, list[str]] = {}
        self._lease_scopes: dict[str, tuple[str, ...]] = {}
        self._lease_hashes: dict[str, dict[str, str | None]] = {}
        self._adapter_lock = threading.RLock()

    def _run_config(self, run: UltraRunV1) -> dict[str, Any]:
        return {
            **dict(run.config_snapshot),
            "model_snapshot": dict(run.model_snapshot),
            "prompt": run.prompt,
            "engine_metadata": dict(run.metadata),
        }

    def save_ultra_run(self, run: UltraRunV1) -> None:
        super().save_ultra_run(run)
        with self._adapter_lock:
            self.run_id = run.id
            try:
                self.store.get_ultra_run(run.id)
            except NotFoundError:
                self.store.create_ultra_run(
                    UltraRun(
                        id=run.id,
                        goal_id=self.goal_id,
                        provider=self.descriptor.provider,
                        model=self.descriptor.model,
                        execution_class=self.descriptor.execution_class,
                        access_level=self.access_level,
                        concurrency=run.concurrency,
                        phase=_store_phase(run.phase),
                        status=_store_run_status(run.phase),
                        config=self._run_config(run),
                    )
                )
                self.store.update_goal_metadata(
                    self.goal_id,
                    ultra_run_id=run.id,
                    interaction_mode="ultra",
                )
                return
            self.store.update_ultra_run(
                run.id,
                provider=self.descriptor.provider,
                model=self.descriptor.model,
                execution_class=self.descriptor.execution_class,
                access_level=self.access_level,
                concurrency=(
                    1
                    if self.descriptor.execution_class is ExecutionClass.LOCAL
                    else max(1, min(8, run.concurrency))
                ),
                phase=_store_phase(run.phase),
                status=_store_run_status(run.phase),
                config=self._run_config(run),
                error=("ULTRA execution failed" if run.phase is EnginePhase.FAILED else None),
            )

    @staticmethod
    def _goal_spec(value: EngineGoalSpec) -> GoalSpecV1:
        return GoalSpecV1(
            objective=value.objective,
            scope=value.in_scope,
            success_criteria=value.success_criteria,
            constraints=value.constraints,
            non_goals=value.out_of_scope,
        )

    @staticmethod
    def _architecture(value: EngineArchitectureSpec) -> ArchitectureSpecV1:
        interfaces: dict[str, Any] = {}
        for index, item in enumerate(value.interfaces, start=1):
            name = str(item.get("name") or item.get("id") or f"interface-{index}")
            interfaces[name] = dict(item)
        return ArchitectureSpecV1(
            summary=value.summary,
            components=value.components,
            interfaces=interfaces,
            decisions=value.decisions,
            constraints=value.invariants,
        )

    def checkpoint_questions(self, goal_spec: EngineGoalSpec) -> None:
        assert self.run_id
        self.store.update_ultra_run(
            self.run_id,
            phase=UltraPhase.GOAL_SPEC,
            status=UltraRunStatus.PAUSED,
            goal_spec=self._goal_spec(goal_spec),
            config={"pending_questions": list(goal_spec.questions)},
        )
        goal = self.store.get_goal(self.goal_id)
        self.store.update_goal_metadata(
            self.goal_id,
            ultra_run_id=self.run_id,
            plan_questions=list(goal_spec.questions),
            plan_answers={},
            waiting_question=(
                str(goal_spec.questions[0].get("question", "")) if goal_spec.questions else ""
            ),
            resume_status=GoalStatus.DISCOVERING.value,
            auto_retryable=False,
        )
        if goal.status != GoalStatus.PAUSED:
            self.store.transition_goal(
                self.goal_id,
                GoalStatus.PAUSED,
                reason="ULTRA goal decisions require user input",
            )

    def _plan_payload(self, master: MasterPlanV1) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        used: set[str] = set()
        self.task_ids = {
            module.id: _safe_task_id(module.id, index, used)
            for index, module in enumerate(master.modules, start=1)
        }
        tasks: list[dict[str, Any]] = []
        changes: list[dict[str, Any]] = []
        for index, module in enumerate(master.modules, start=1):
            task_id = self.task_ids[module.id]
            dependencies = [
                self.task_ids[item]
                for item in module.depends_on
                if item in self.task_ids
            ]
            tasks.append(
                {
                    "id": task_id,
                    "title": module.title[:180],
                    "description": module.objective[:4_000],
                    "acceptance_criteria": list(module.acceptance_criteria),
                    "verification": list(module.verification),
                    "depends_on": dependencies,
                    "risk": str(module.metadata.get("risk", "high")).lower()
                    if str(module.metadata.get("risk", "high")).lower()
                    in {"low", "medium", "high", "critical"}
                    else "high",
                    "role": RoleProfile(
                        name="ULTRA module orchestrator",
                        mission=module.objective,
                        expertise=("planning", "implementation", "review", "testing", "integration"),
                        constraints=module.forbidden_changes,
                        deliverables=module.acceptance_criteria,
                        tool_policy={"write_paths": list(module.write_paths)},
                    ).to_dict(),
                    "priority": max(0, len(master.modules) - index),
                    "metadata": {"ultra_node_id": module.id},
                }
            )
            paths = module.write_paths or (".",)
            for path in paths:
                changes.append(
                    {
                        "path": path,
                        "intent": module.objective[:1_000],
                        "supports_tasks": [task_id],
                    }
                )
        return tasks, changes

    def bind_foundation(
        self,
        goal_spec: EngineGoalSpec,
        architecture: EngineArchitectureSpec,
        master: MasterPlanV1,
    ) -> Plan:
        assert self.run_id
        tasks, changes = self._plan_payload(master)
        all_ids = [item["id"] for item in tasks]
        strategy = master.execution_strategy or (
            "Execute dependency-safe module waves. Every node uses isolated planning, "
            "implementation, independent review, tests, bounded fixes, integration, and memory write-back."
        )
        self.plan = self.store.create_plan(
            self.goal_id,
            master.summary,
            tasks,
            applicability_evidence=[
                {
                    "fact": "ULTRA GoalSpec and Architecture were produced after repository inspection.",
                    "source": f"ultra:{self.run_id}:foundation",
                    "supports_tasks": all_ids,
                }
            ],
            execution_strategy=strategy,
            expected_changes=changes,
            proposed_by="ultra-planner",
            submit=True,
        )
        self.store.update_ultra_run(
            self.run_id,
            phase=UltraPhase.AWAITING_APPROVAL,
            status=UltraRunStatus.AWAITING_APPROVAL,
            goal_spec=self._goal_spec(goal_spec),
            architecture_spec=self._architecture(architecture),
            config={
                "master_plan_fingerprint": master.fingerprint,
                "module_count": len(master.modules),
            },
        )
        brain = ProjectBrain(self.store, self.run_id)
        canonical_goal = self._goal_spec(goal_spec)
        canonical_architecture = self._architecture(architecture)
        brain.set_north_star(canonical_goal)
        brain.set_architecture(canonical_architecture)
        for index, decision in enumerate(architecture.decisions, start=1):
            choice = str(
                decision.get("decision")
                or decision.get("choice")
                or decision.get("summary")
                or _json(decision)
            )
            brain.record_decision(
                str(decision.get("title") or decision.get("name") or f"Architecture decision {index}"),
                choice,
                reason=str(decision.get("reason", "")),
                alternatives=tuple(str(item) for item in decision.get("alternatives", ())),
                status=str(decision.get("status", "accepted")),
            )
        for index, constraint in enumerate(goal_spec.constraints, start=1):
            brain.write(
                BrainSection.CONSTRAINT,
                f"Constraint {index}",
                constraint,
                data={"constraint": constraint, "source": "GoalSpecV1"},
            )
        goal = self.store.get_goal(self.goal_id)
        self.store.update_goal_metadata(
            self.goal_id,
            ultra_run_id=self.run_id,
            plan_questions=[],
            waiting_question="",
            auto_retryable=False,
        )
        if goal.status != GoalStatus.AWAITING_PLAN_APPROVAL:
            self.store.transition_goal(
                self.goal_id,
                GoalStatus.AWAITING_PLAN_APPROVAL,
                reason="ULTRA master plan awaits one user approval",
            )
        return self.plan

    def approve_master(self, master: MasterPlanV1) -> Plan:
        if not self.run_id or not self.plan:
            raise StateStoreError("ULTRA foundation is not bound to a durable master plan")
        accepted, _ = self.store.approve_plan(
            self.goal_id,
            self.plan.revision,
            approved_by="user",
            expected_fingerprint=self.plan.fingerprint,
        )
        self.store.approve_ultra_master(
            self.run_id,
            accepted.revision,
            accepted.fingerprint,
            approved_by="user",
        )
        self.plan = accepted
        self.approved = True
        self._flush_nodes()
        return accepted

    @staticmethod
    def _contract(node: EngineWorkNode, *, root: bool) -> TaskContractV1:
        interfaces = {name: {"owned": True} for name in node.contract.owned_interfaces}
        return TaskContractV1(
            objective=node.contract.objective,
            success_criteria=node.contract.acceptance_criteria,
            write_paths=node.contract.write_paths or ((".",) if root else ()),
            read_paths=(".",),
            forbidden_changes=node.contract.forbidden_changes,
            interfaces=interfaces,
            external_dependencies=tuple(
                str(item)
                for item in node.contract.metadata.get("external_dependencies", ())
            ),
            metadata={
                **dict(node.contract.metadata),
                "verification": list(node.contract.verification),
                "owned_interfaces": list(node.contract.owned_interfaces),
            },
        )

    def _stored_node(self, node: EngineWorkNode) -> WorkNode:
        root = node.parent_id is None and node.kind is NodeKind.MODULE
        result = self._result_cache.get(node.id)
        return WorkNode(
            id=node.id,
            ultra_run_id=self.run_id or "",
            title=node.contract.title,
            objective=node.contract.objective,
            contract=self._contract(node, root=root),
            kind=_store_kind(node.kind),
            status=_store_node_status(node.status),
            parent_id=node.parent_id,
            master_task_id=self.task_ids.get(node.id) if root else None,
            position=node.order,
            depends_on=node.depends_on,
            assigned_role=(node.phase.value if node.phase else "coder"),
            attempts=result.fix_attempts if result else 0,
            max_attempts=self.config.max_fix_attempts,
            result=self._result(result) if result else None,
            error=(result.summary if result and not result.success else None),
            checkpoint=(node.phase.value if node.phase else ""),
            metadata={"children": list(node.children)},
        )

    def _flush_nodes(self) -> None:
        if not self.approved:
            return
        while True:
            progressed = False
            for node_id, node in list(self._pending_nodes.items()):
                if node.parent_id and node.parent_id not in self._persisted_nodes:
                    continue
                if any(dep not in self._persisted_nodes for dep in node.depends_on):
                    continue
                stored = self._stored_node(node)
                self.store.create_work_node(stored)
                self._persisted_nodes.add(node_id)
                self._pending_nodes.pop(node_id, None)
                progressed = True
            if not progressed:
                break
        if not self._pending_nodes:
            self._flush_artifacts()

    def save_work_node(self, run_id: str, node: EngineWorkNode) -> None:
        super().save_work_node(run_id, node)
        with self._adapter_lock:
            self._pending_nodes[node.id] = node
            if node.id not in self._persisted_nodes:
                self._flush_nodes()
                return
            current = self.store.get_work_node(node.id)
            target = _store_node_status(node.status)
            result = self._result_cache.get(node.id)
            if current.status != target or result is not None:
                self.store.transition_work_node(
                    node.id,
                    target,
                    result=self._result(result) if result else current.result,
                    error=(result.summary if result and not result.success else current.error),
                    checkpoint=node.phase.value if node.phase else current.checkpoint,
                )
            self._pending_nodes.pop(node.id, None)
            self._sync_master_task(node, result)

    @staticmethod
    def _insight(value: Any) -> InsightV1:
        kwargs: dict[str, Any] = {
            "summary": str(getattr(value, "summary", "Insight")),
            "category": str(getattr(value, "source", "agent")),
            "details": _json(getattr(value, "details", {})),
            "severity": str(getattr(value, "severity", "info"))
            if str(getattr(value, "severity", "info")) in {"info", "warning", "error", "critical"}
            else "info",
        }
        if getattr(value, "id", None):
            kwargs["id"] = value.id
        return InsightV1(**kwargs)

    @classmethod
    def _result(cls, value: EngineResult) -> ResultPackageV1:
        changed: list[str] = []
        artifact_uris: list[str] = []
        for item in value.artifacts:
            path = str(item.get("path", "")).strip()
            uri = str(item.get("uri", path)).strip()
            if path:
                changed.append(path)
            if uri:
                artifact_uris.append(uri)
        return ResultPackageV1(
            summary=value.summary or f"{value.node_id} result",
            changed_files=tuple(dict.fromkeys(changed)),
            tests=value.test_results,
            artifacts=tuple(dict.fromkeys(artifact_uris)),
            insights=tuple(cls._insight(item) for item in value.insights),
            issues=value.findings,
            metadata={
                "success": value.success,
                "status": value.status,
                "fix_attempts": value.fix_attempts,
                "evidence": list(value.evidence),
            },
        )

    def save_result_package(self, run_id: str, result: EngineResult) -> None:
        super().save_result_package(run_id, result)
        with self._adapter_lock:
            self._result_cache[result.node_id] = result
            if result.node_id in self._persisted_nodes:
                current = self.store.get_work_node(result.node_id)
                target = (
                    WorkNodeStatus.COMPLETED
                    if result.success
                    else WorkNodeStatus.REVISION_REQUIRED
                    if result.status == "revision_required"
                    else WorkNodeStatus.FAILED
                )
                self.store.transition_work_node(
                    result.node_id,
                    target,
                    result=self._result(result),
                    error=None if result.success else result.summary,
                )
                node = self.nodes.get(run_id, {}).get(result.node_id)
                if node:
                    self._sync_master_task(node, result)
                if result.node_id not in self._brain_results:
                    ProjectBrain(self.store, run_id).write_back_result(
                        result.node_id,
                        self._result(result),
                    )
                    self._brain_results.add(result.node_id)
            for item in result.artifacts:
                uri = str(item.get("uri") or item.get("path") or "").strip()
                if not uri:
                    continue
                self._queue_artifact(
                    Artifact(
                        ultra_run_id=run_id,
                        work_node_id=(result.node_id if result.node_id != "__global__" else None),
                        kind=str(item.get("kind", "result")),
                        uri=uri,
                        path=str(item.get("path") or "") or None,
                        content_hash=str(item.get("hash") or "") or None,
                        evidence={"result": result.summary},
                    )
                )

    def _master_node(self, node: EngineWorkNode) -> str | None:
        current = node
        seen: set[str] = set()
        while current.parent_id and current.parent_id not in seen:
            seen.add(current.id)
            parent = self.nodes.get(self.run_id or "", {}).get(current.parent_id)
            if parent is None:
                break
            current = parent
        return current.id if current.id in self.task_ids else None

    def master_task_for_node(self, node_id: str | None) -> str | None:
        if not node_id or not self.run_id:
            return None
        node = self.nodes.get(self.run_id, {}).get(node_id)
        root = self._master_node(node) if node else None
        return self.task_ids.get(root or "")

    def _sync_master_task(self, node: EngineWorkNode, result: EngineResult | None) -> None:
        if not self.plan:
            return
        task_id = self.master_task_for_node(node.id)
        if not task_id or self.task_ids.get(node.id) != task_id:
            return
        task = next((item for item in self.store.get_plan(self.goal_id, self.plan.revision).tasks if item.id == task_id), None)
        if task is None:
            return
        if node.status is NodeStatus.RUNNING and task.status in {TaskStatus.PENDING, TaskStatus.READY}:
            self.store.transition_task(
                self.goal_id,
                self.plan.revision,
                task_id,
                TaskStatus.IN_PROGRESS,
                actor="ultra-scheduler",
            )
        elif result and result.success and task.status != TaskStatus.COMPLETED:
            self.store.transition_task(
                self.goal_id,
                self.plan.revision,
                task_id,
                TaskStatus.COMPLETED,
                note=result.summary,
                evidence=(result.summary,),
                actor="ultra-evidence-gate",
            )
        elif result and not result.success and task.status not in {TaskStatus.FAILED, TaskStatus.BLOCKED}:
            self.store.transition_task(
                self.goal_id,
                self.plan.revision,
                task_id,
                TaskStatus.BLOCKED,
                note=result.summary or "ULTRA quality gate failed",
                actor="ultra-quality-gate",
            )

    def save_agent_run(self, item: Any) -> None:
        super().save_agent_run(item)
        with self._adapter_lock:
            status = {
                "running": AgentRunStatus.RUNNING,
                "completed": AgentRunStatus.COMPLETED,
                "failed": AgentRunStatus.FAILED,
                "cancelled": AgentRunStatus.CANCELLED,
                "rate_limited": AgentRunStatus.RATE_LIMITED,
                "uncertain": AgentRunStatus.UNCERTAIN,
            }.get(item.status, AgentRunStatus.FAILED)
            if item.id in self._persisted_agents:
                self.store.update_agent_run(
                    item.id,
                    status,
                    usage=item.usage,
                    error=item.error or None,
                    prompt_trace_id=item.prompt_trace_id,
                    side_effects=item.role in {AgentRole.CODER, AgentRole.INTEGRATOR},
                )
                return
            self.store.create_agent_run(
                AgentRun(
                    id=item.id,
                    ultra_run_id=item.run_id,
                    work_node_id=item.node_id,
                    role=item.role.value,
                    provider=item.provider or self.descriptor.provider,
                    model=item.model or self.descriptor.model,
                    phase=item.phase,
                    status=status,
                    usage=item.usage,
                    error=item.error or None,
                    prompt_trace_id=item.prompt_trace_id,
                    side_effects=item.role in {AgentRole.CODER, AgentRole.INTEGRATOR},
                )
            )
            self._persisted_agents.add(item.id)

    def save_prompt_trace(self, trace: EnginePromptTrace) -> None:
        super().save_prompt_trace(trace)
        with self._adapter_lock:
            if trace.id in self._persisted_traces:
                return
            self.store.add_prompt_trace(
                PromptTraceV1(
                    id=trace.id,
                    ultra_run_id=trace.run_id,
                    work_node_id=trace.node_id,
                    agent_run_id=trace.agent_run_id,
                    role=trace.role.value,
                    system_prompt=trace.system_prompt,
                    context_package=trace.context_package,
                    self_prompt=trace.self_prompt,
                    reasoning_summary=trace.reasoning_summary,
                    omitted_sections=trace.omitted_context,
                    redacted=True,
                    metadata={
                        "phase": trace.phase,
                        "chain_of_thought": "not stored",
                    },
                ),
                max_bytes=self.config.prompt_trace_chars,
            )
            self._persisted_traces.add(trace.id)

    @staticmethod
    def _brain_section(section: EngineBrainSection) -> BrainSection:
        return {
            EngineBrainSection.NORTH_STAR: BrainSection.NORTH_STAR,
            EngineBrainSection.ARCHITECTURE: BrainSection.ARCHITECTURE,
            EngineBrainSection.DECISION: BrainSection.DECISION,
            EngineBrainSection.CONSTRAINT: BrainSection.CONSTRAINT,
            EngineBrainSection.TASK_GRAPH: BrainSection.TASK_GRAPH,
            EngineBrainSection.ARTIFACT: BrainSection.ARTIFACT_INDEX,
            EngineBrainSection.KNOWLEDGE: BrainSection.KNOWLEDGE,
            EngineBrainSection.LESSON: BrainSection.LESSON,
            EngineBrainSection.ROLE_MEMORY: BrainSection.ROLE_MEMORY,
        }.get(section, BrainSection.KNOWLEDGE)

    def append_brain_entry(self, entry: BrainEntryV1) -> None:
        super().append_brain_entry(entry)
        role = entry.role.value if entry.role else None
        stored = self.store.put_brain_entry(
            BrainEntry(
                ultra_run_id=entry.run_id,
                goal_id=self.goal_id,
                work_node_id=entry.node_id,
                section=self._brain_section(entry.section),
                title=entry.key,
                content=_json(entry.value),
                data=entry.value,
                role=role,
                expires_at=entry.expires_at,
                metadata={"engine_section": entry.section.value},
            )
        )
        self.store.record_memory_access(
            entry.run_id,
            direction="write",
            work_node_id=entry.node_id,
            brain_entry_id=stored.id,
            query=entry.key,
            metadata={"section": stored.section.value, "role": role or ""},
        )

    def list_brain_entries(self, run_id: str) -> tuple[BrainEntryV1, ...]:
        live = super().list_brain_entries(run_id)
        result: list[BrainEntryV1] = []
        for item in self.store.list_brain_entries(run_id, latest_only=True):
            engine_name = str(item.metadata.get("engine_section", item.section.value))
            try:
                section = EngineBrainSection(engine_name)
            except ValueError:
                section = EngineBrainSection.KNOWLEDGE
            try:
                role = AgentRole(item.role) if item.role else None
            except ValueError:
                role = None
            result.append(
                BrainEntryV1(
                    section=section,
                    key=item.title,
                    value=dict(item.data),
                    run_id=run_id,
                    node_id=item.work_node_id,
                    role=role,
                    version=item.version,
                    expires_at=item.expires_at,
                    created_at=item.created_at,
                )
            )
        merged: dict[tuple[str, str, str], BrainEntryV1] = {
            (item.section.value, item.key, item.role.value if item.role else ""): item
            for item in result
        }
        for item in live:
            merged[(item.section.value, item.key, item.role.value if item.role else "")] = item
        return tuple(merged.values())

    def _queue_artifact(self, artifact: Artifact) -> None:
        self._pending_artifacts.append(artifact)
        self._flush_artifacts()

    def _flush_artifacts(self) -> None:
        remaining: list[Artifact] = []
        for artifact in self._pending_artifacts:
            if artifact.work_node_id and artifact.work_node_id not in self._persisted_nodes:
                remaining.append(artifact)
                continue
            try:
                stored = self.store.add_artifact(artifact)
            except StateStoreError:
                remaining.append(artifact)
                continue
            try:
                self.store.put_brain_entry(
                    BrainEntry(
                        ultra_run_id=stored.ultra_run_id,
                        goal_id=self.goal_id,
                        work_node_id=stored.work_node_id,
                        agent_run_id=stored.agent_run_id,
                        section=BrainSection.ARTIFACT_INDEX,
                        title=stored.path or stored.uri,
                        content=f"{stored.kind} artifact: {stored.uri}",
                        data={
                            "artifact_id": stored.id,
                            "kind": stored.kind,
                            "uri": stored.uri,
                            "path": stored.path,
                            "content_hash": stored.content_hash,
                            "pre_write_hash": stored.pre_write_hash,
                            "evidence": dict(stored.evidence),
                        },
                        metadata={"source": "artifact_index"},
                    )
                )
            except StateStoreError:
                # The artifacts table remains authoritative if the searchable
                # Project Brain mirror cannot be refreshed.
                pass
        self._pending_artifacts = remaining

    def record_file_artifact(
        self,
        node_id: str | None,
        path: str,
        pre_hash: str | None,
        post_hash: str | None,
        tool_name: str,
    ) -> None:
        if not self.run_id:
            return
        self._queue_artifact(
            Artifact(
                ultra_run_id=self.run_id,
                work_node_id=node_id,
                kind="file",
                uri=f"workspace:{_normalized_path(path)}",
                path=_normalized_path(path),
                content_hash=post_hash,
                pre_write_hash=pre_hash,
                evidence={"tool": tool_name},
            )
        )

    def lease_manager(self, workspace: Path) -> ResourceLeaseManager:
        def acquired(lease: RuntimeLease) -> None:
            if not self.run_id:
                return
            scopes = tuple(_normalized_path(path) for path in lease.paths)
            hashes = _snapshot_write_scopes(workspace, scopes)
            created: list[str] = []
            try:
                for path in lease.paths:
                    row = self.store.acquire_resource_lease(
                        self.run_id,
                        lease.owner,
                        path,
                        pre_write_hash=_hash_file(workspace, path),
                    )
                    created.append(row.id)
            except Exception:
                for lease_id in created:
                    self.store.release_resource_lease(lease_id)
                raise
            with self._adapter_lock:
                self._lease_ids[lease.owner] = created
                self._lease_scopes[lease.owner] = scopes
                self._lease_hashes[lease.owner] = hashes

        def released(lease: RuntimeLease) -> None:
            with self._adapter_lock:
                lease_ids = self._lease_ids.pop(lease.owner, [])
                self._lease_scopes.pop(lease.owner, None)
                self._lease_hashes.pop(lease.owner, None)
            for lease_id in lease_ids:
                self.store.release_resource_lease(lease_id)

        return ResourceLeaseManager(
            lambda path: _hash_file(workspace, path),
            on_acquire=acquired,
            on_release=released,
        )

    def lease_hash(self, owner: str, path: str) -> tuple[bool, str | None]:
        normalized = _normalized_path(path)
        with self._adapter_lock:
            scopes = self._lease_scopes.get(owner, ())
            if not scopes or not _within_scope(normalized, scopes):
                return False, None
            return True, self._lease_hashes.get(owner, {}).get(normalized)

    def advance_lease_hash(self, owner: str, path: str, value: str | None) -> None:
        normalized = _normalized_path(path)
        with self._adapter_lock:
            scopes = self._lease_scopes.get(owner, ())
            if scopes and _within_scope(normalized, scopes):
                self._lease_hashes.setdefault(owner, {})[normalized] = value


class UltraSession:
    """Interactive ULTRA profile owned by one :class:`AgentRuntime`."""

    def __init__(
        self,
        *,
        store: StateStore,
        workspace: Path,
        descriptor: ModelDescriptor,
        permission_adapter: PermissionAdapter,
        approval: Callable[[str, dict[str, Any], str], bool],
        events: EventBus,
        config: UltraConfig,
        agent_steps: int,
    ) -> None:
        self.store = store
        self.workspace = workspace
        self.descriptor = descriptor
        self.permission_adapter = permission_adapter
        self.approval = approval
        self.events = events
        self.config = config
        self.agent_steps = agent_steps
        self.goal_id: str | None = None
        self.adapter: StateStoreUltraAdapter | None = None
        self.orchestrator: UltraOrchestrator | None = None
        self.future: Future[UltraRunResult] | None = None
        self.answers: dict[str, str] = {}

    @property
    def running(self) -> bool:
        return bool(self.future and not self.future.done())

    @property
    def safe_for_reconfiguration(self) -> bool:
        if not self.running:
            return True
        if not self.orchestrator or not self.orchestrator.control.paused or not self.run_id:
            return False
        return not any(
            item.status is AgentRunStatus.RUNNING
            for item in self.store.list_agent_runs(self.run_id)
        )

    @property
    def run_id(self) -> str | None:
        return self.adapter.run_id if self.adapter else None

    def _node(self, node_id: str | None) -> EngineWorkNode | None:
        if not node_id or not self.orchestrator:
            return None
        return self.orchestrator.nodes.get(node_id)

    def _execute_tool(self, call: ToolCall, request: AgentRequest) -> str:
        allowed = WorkspaceUltraAgent(
            None,
            role=request.role,
            provider_name=self.descriptor.provider,
            model=self.descriptor.model,
            executor=lambda _call, _request: "",
            events=self.events,
        )._allowed_tools()
        if call.name not in allowed:
            return f"Error: role {request.role.value} cannot use {call.name}"
        args = call.args if isinstance(call.args, dict) else {}
        node = self._node(request.node_id)
        if call.name in {"write_file", "edit_file"}:
            path = str(args.get("path", ""))
            scopes = node.write_paths if node else ()
            if not scopes or not _within_scope(path, scopes):
                return (
                    f"Error: path {path!r} is outside this node's approved write scope; "
                    "a master-plan revision is required"
                )
            normalized = _normalized_path(path)
            expected_known = False
            expected: str | None = None
            if node:
                for raw_path, value in node.pre_write_hashes.items():
                    if _normalized_path(raw_path) == normalized:
                        expected_known, expected = True, value
                        break
            if not expected_known and self.adapter and request.node_id:
                expected_known, expected = self.adapter.lease_hash(request.node_id, normalized)
            current = _hash_file(self.workspace, path)
            if expected_known and current != expected:
                raise StaleWriteError(
                    f"pre-write hash changed for {path!r}: expected {expected!r}, got {current!r}"
                )
        risk = _TOOL_RISK.get(call.name, "unknown")
        normal_requirement = tools.requires_approval(call.name, args)
        needs_approval = self.permission_adapter.requires_approval(normal_requirement)
        self.events.publish(
            "tool_call",
            call.name,
            args=redact_data(args),
            actor=request.role.value,
            node_id=request.node_id,
        )
        task_id = self.adapter.master_task_for_node(request.node_id) if self.adapter else None
        action_id = self.store.begin_action(
            self.goal_id or "",
            call.name,
            {
                "arguments": redact_data(args),
                "ultra_run_id": self.run_id,
                "node_id": request.node_id,
                "role": request.role.value,
                "phase": request.phase,
            },
            task_id=task_id,
            risk=risk,
            mutating=call.name in _WRITE_TOOLS,
        )
        if needs_approval and not self.approval(call.name, dict(args), risk):
            result = "Permission denied by the user. Do not repeat the same action."
            self.store.complete_action(action_id, result, status="denied")
            self.events.publish("tool_result", result, tool=call.name, actor=request.role.value)
            return result
        path = str(args.get("path", "")) if call.name in {"write_file", "edit_file"} else ""
        pre_hash = _hash_file(self.workspace, path) if path else None
        try:
            with tools.workspace_context(self.workspace):
                if call.name == "run_bash":
                    assert self.orchestrator
                    with self.orchestrator.scheduler.leases.mutating_shell(request.node_id or request.role.value):
                        result = self.permission_adapter.run_shell(
                            str(args.get("command", "")),
                            self.workspace,
                            normal_runner=lambda command: tools.run_tool(
                                "run_bash", {"command": command}
                            ),
                        )
                else:
                    result = tools.run_tool(call.name, args)
            result = redact_text(result, 50_000)
            status = "failed" if result.startswith("Error:") else "completed"
            self.store.complete_action(action_id, redact_text(result, 2_000), status=status)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            result = f"Error: ULTRA tool harness failure: {type(exc).__name__}: {redact_text(exc, 500)}"
            self.store.complete_action(action_id, result, status="failed")
        if path and not result.startswith("Error:") and self.adapter:
            post_hash = _hash_file(self.workspace, path)
            self.adapter.record_file_artifact(
                request.node_id,
                path,
                pre_hash,
                post_hash,
                call.name,
            )
            if request.node_id:
                self.adapter.advance_lease_hash(request.node_id, path, post_hash)
        self.events.publish(
            "tool_result",
            result,
            tool=call.name,
            actor=request.role.value,
            node_id=request.node_id,
        )
        return result

    def start(self, objective: str) -> MasterPlanV1 | None:
        if self.orchestrator and self.orchestrator.phase not in {
            EnginePhase.COMPLETED,
            EnginePhase.CANCELLED,
            EnginePhase.FAILED,
        }:
            raise RuntimeError("an ULTRA run is already active")
        goal = self.store.create_goal(redact_text(objective, 20_000))
        self.store.transition_goal(goal.id, GoalStatus.DISCOVERING, reason="ULTRA foundation started")
        return self._prepare_existing_goal(goal.id, objective)

    def restart_foundation(self, goal_id: str, objective: str) -> MasterPlanV1 | None:
        """Create a fresh ULTRA run/revision while preserving the durable goal."""

        return self._prepare_existing_goal(goal_id, objective)

    def _prepare_existing_goal(self, goal_id: str, objective: str) -> MasterPlanV1 | None:
        self.goal_id = goal_id
        self.answers = {}
        self.adapter = StateStoreUltraAdapter(
            self.store,
            goal_id,
            self.descriptor,
            self.permission_adapter.access_level,
            self.config,
        )
        factory = WorkspaceUltraAgentFactory(
            self.descriptor,
            self._execute_tool,
            self.events,
            max_steps=self.agent_steps,
        )
        self.orchestrator = UltraOrchestrator(
            factory,
            execution_class=self.descriptor.execution_class,
            state=self.adapter,
            events=self.events,
            config=self.config,
            context_builder=DurableContextBuilder(
                self.store,
                lambda: self.adapter.run_id if self.adapter else None,
                self.config.context_chars,
            ),
            leases=self.adapter.lease_manager(self.workspace),
            model_snapshot=self.descriptor.to_dict(),
        )
        plan = self.orchestrator.prepare(objective)
        if plan is None:
            assert self.orchestrator.goal_spec
            self.adapter.checkpoint_questions(self.orchestrator.goal_spec)
            return None
        assert self.orchestrator.goal_spec and self.orchestrator.architecture
        self.adapter.bind_foundation(
            self.orchestrator.goal_spec,
            self.orchestrator.architecture,
            plan,
        )
        return plan

    @staticmethod
    def _engine_node_status(status: WorkNodeStatus) -> NodeStatus:
        return {
            WorkNodeStatus.PENDING: NodeStatus.PENDING,
            WorkNodeStatus.READY: NodeStatus.READY,
            WorkNodeStatus.IN_PROGRESS: NodeStatus.RUNNING,
            WorkNodeStatus.REVIEWING: NodeStatus.RUNNING,
            WorkNodeStatus.TESTING: NodeStatus.RUNNING,
            WorkNodeStatus.FIXING: NodeStatus.RUNNING,
            WorkNodeStatus.INTEGRATING: NodeStatus.RUNNING,
            WorkNodeStatus.COMPLETED: NodeStatus.COMPLETED,
            WorkNodeStatus.FAILED: NodeStatus.FAILED,
            WorkNodeStatus.BLOCKED: NodeStatus.BLOCKED,
            WorkNodeStatus.CANCELLED: NodeStatus.CANCELLED,
            WorkNodeStatus.UNCERTAIN: NodeStatus.UNCERTAIN,
            WorkNodeStatus.REVISION_REQUIRED: NodeStatus.REVISION_REQUIRED,
        }[status]

    @staticmethod
    def _engine_result(node_id: str, value: ResultPackageV1) -> EngineResult:
        artifacts = tuple(
            {"path": path, "uri": f"workspace:{path}", "kind": "file"}
            for path in value.changed_files
        ) + tuple({"uri": uri, "kind": "result"} for uri in value.artifacts)
        return EngineResult(
            node_id=node_id,
            success=bool(value.metadata.get("success", True)),
            status=str(value.metadata.get("status", "completed")),
            summary=value.summary,
            artifacts=artifacts,
            evidence=tuple(value.metadata.get("evidence", ())),
            test_results=value.tests,
            findings=value.issues,
            fix_attempts=int(value.metadata.get("fix_attempts", 0) or 0),
        )

    def restore(self, run_id: str) -> Future[UltraRunResult]:
        """Rebuild the scheduler from durable evidence without replaying uncertainty."""

        run = self.store.get_ultra_run(run_id)
        nodes = self.store.list_work_nodes(run_id)
        uncertain_nodes = [item.id for item in nodes if item.status is WorkNodeStatus.UNCERTAIN]
        uncertain_agents = [
            item.id
            for item in self.store.list_agent_runs(run_id)
            if item.status is AgentRunStatus.UNCERTAIN
        ]
        uncertain_actions = [
            item["id"] for item in self.store.list_actions(run.goal_id, status="uncertain")
        ]
        if uncertain_nodes or uncertain_agents or uncertain_actions:
            values = [*uncertain_nodes, *uncertain_agents, *uncertain_actions]
            raise RuntimeError(
                "reconcile uncertain ULTRA work before resume: " + ", ".join(values[:12])
            )
        if not run.master_approved or run.plan_revision is None:
            raise RuntimeError("the interrupted ULTRA run has no approved master plan")
        plan = self.store.get_plan(run.goal_id, run.plan_revision)
        if run.goal_spec is None or run.architecture_spec is None:
            raise RuntimeError("the interrupted ULTRA foundation is incomplete; use /replan")

        self.goal_id = run.goal_id
        self.adapter = StateStoreUltraAdapter(
            self.store,
            run.goal_id,
            self.descriptor,
            self.permission_adapter.access_level,
            self.config,
        )
        self.adapter.run_id = run.id
        self.adapter.plan = plan
        self.adapter.approved = True
        self.adapter.task_ids = {
            str(task.metadata.get("ultra_node_id", task.id)): task.id for task in plan.tasks
        }
        self.adapter._persisted_nodes = {item.id for item in nodes}

        factory = WorkspaceUltraAgentFactory(
            self.descriptor,
            self._execute_tool,
            self.events,
            max_steps=self.agent_steps,
        )
        self.orchestrator = UltraOrchestrator(
            factory,
            execution_class=self.descriptor.execution_class,
            state=self.adapter,
            events=self.events,
            config=self.config,
            context_builder=DurableContextBuilder(
                self.store,
                lambda: self.adapter.run_id if self.adapter else None,
                self.config.context_chars,
            ),
            leases=self.adapter.lease_manager(self.workspace),
            model_snapshot=self.descriptor.to_dict(),
        )
        goal_spec = EngineGoalSpec(
            objective=run.goal_spec.objective,
            success_criteria=run.goal_spec.success_criteria
            or ("Complete every approved module and final evidence gate.",),
            constraints=run.goal_spec.constraints,
            in_scope=run.goal_spec.scope,
            out_of_scope=run.goal_spec.non_goals,
            assumptions=tuple(
                f"{key}: {value}"
                for key, value in run.goal_spec.answered_questions.items()
            ),
        )
        interface_values = tuple(
            {"name": name, **(dict(value) if isinstance(value, Mapping) else {"contract": value})}
            for name, value in run.architecture_spec.interfaces.items()
        )
        architecture = EngineArchitectureSpec(
            summary=run.architecture_spec.summary,
            components=run.architecture_spec.components or ({"name": "restored-project"},),
            interfaces=interface_values,
            decisions=run.architecture_spec.decisions,
            invariants=run.architecture_spec.constraints,
        )
        stored_by_id = {item.id: item for item in nodes}
        engine_nodes: dict[str, EngineWorkNode] = {}
        module_contracts = []
        for item in nodes:
            verification = tuple(item.contract.metadata.get("verification", ()))
            if not verification and item.master_task_id:
                legacy = next(
                    (task for task in plan.tasks if task.id == item.master_task_id),
                    None,
                )
                verification = legacy.verification if legacy else ("Inspect the durable evidence.",)
            contract = self._engine_contract(item, verification)
            children = tuple(str(value) for value in item.metadata.get("children", ()))
            dependencies = tuple(dict.fromkeys((*item.depends_on, *children)))
            try:
                phase = InnerPhase(item.checkpoint) if item.checkpoint else None
            except ValueError:
                phase = None
            engine = EngineWorkNode(
                contract=contract,
                parent_id=item.parent_id,
                depth=item.depth or 1,
                kind=NodeKind(item.kind.value),
                order=item.position,
                status=self._engine_node_status(item.status),
                phase=phase,
                children=children,
                pre_write_hashes={},
            )
            if dependencies != contract.depends_on:
                engine = EngineWorkNode(
                    contract=type(contract)(
                        id=contract.id,
                        title=contract.title,
                        objective=contract.objective,
                        acceptance_criteria=contract.acceptance_criteria,
                        verification=contract.verification,
                        depends_on=dependencies,
                        write_paths=contract.write_paths,
                        forbidden_changes=contract.forbidden_changes,
                        owned_interfaces=contract.owned_interfaces,
                        metadata=contract.metadata,
                    ),
                    parent_id=engine.parent_id,
                    depth=engine.depth,
                    kind=engine.kind,
                    order=engine.order,
                    status=engine.status,
                    phase=engine.phase,
                    children=engine.children,
                    pre_write_hashes=engine.pre_write_hashes,
                )
            engine_nodes[engine.id] = engine
            if item.parent_id is None and item.kind is WorkNodeKind.MODULE:
                module_contracts.append(contract)
            if item.result:
                converted = self._engine_result(item.id, item.result)
                self.adapter._result_cache[item.id] = converted
                self.adapter.results[run_id][item.id] = converted

        if not module_contracts:
            raise RuntimeError("the approved ULTRA run has no durable master modules")
        master = MasterPlanV1(
            summary=plan.summary,
            modules=tuple(module_contracts),
            execution_strategy=plan.execution_strategy,
            revision=plan.revision,
            fingerprint=run.master_plan_fingerprint
            or str(run.config.get("master_plan_fingerprint", "")),
        )
        prompt = str(run.config.get("prompt") or self.store.get_goal(run.goal_id).objective)
        engine_run = UltraRunV1(
            id=run.id,
            prompt=prompt,
            execution_class=self.descriptor.execution_class,
            phase=EnginePhase.AWAITING_APPROVAL,
            concurrency=run.concurrency,
            master_fingerprint=master.fingerprint,
            approved=True,
            model_snapshot=self.descriptor.to_dict(),
            config_snapshot=dict(run.config),
            metadata={"restored": True},
            created_at=run.created_at,
            updated_at=run.updated_at,
        )
        self.orchestrator.run_state = engine_run
        self.orchestrator.goal_spec = goal_spec
        self.orchestrator.architecture = architecture
        self.orchestrator.master_plan = master
        self.orchestrator.nodes = engine_nodes
        self.orchestrator._results = dict(self.adapter.results[run_id])
        self.orchestrator._order = max((item.order for item in engine_nodes.values()), default=0)
        self.adapter.runs[run_id] = engine_run
        self.adapter.nodes[run_id] = dict(engine_nodes)
        self.store.update_ultra_run(
            run_id,
            phase=UltraPhase.MODULE_WAVES,
            status=UltraRunStatus.RUNNING,
            config={"restored_from_evidence": True},
        )
        self.future = self.orchestrator.background.start(self._run_and_finalize)
        return self.future

    @staticmethod
    def _engine_contract(item: WorkNode, verification: Sequence[str]) -> Any:
        from .ultra import TaskContractV1 as EngineTaskContract

        interface_names = tuple(item.contract.interfaces)
        return EngineTaskContract(
            id=item.id,
            title=item.title,
            objective=item.objective,
            acceptance_criteria=item.contract.success_criteria
            or ("Complete this durable node contract.",),
            verification=tuple(verification) or ("Inspect the durable evidence.",),
            depends_on=item.depends_on,
            write_paths=item.contract.write_paths,
            forbidden_changes=item.contract.forbidden_changes,
            owned_interfaces=interface_names,
            metadata={
                **dict(item.contract.metadata),
                "external_dependencies": list(item.contract.external_dependencies),
            },
        )

    def questions(self) -> tuple[Mapping[str, Any], ...]:
        if self.orchestrator and self.orchestrator.goal_spec:
            return tuple(self.orchestrator.goal_spec.questions)
        if self.run_id:
            return tuple(self.store.get_ultra_run(self.run_id).config.get("pending_questions", ()))
        return ()

    def add_guidance(self, text: str) -> None:
        if not self.adapter or not self.run_id:
            raise RuntimeError("there is no live ULTRA run for guidance")
        key = "user-guidance-" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
        self.adapter.append_brain_entry(
            BrainEntryV1(
                EngineBrainSection.KNOWLEDGE,
                key,
                {"summary": text, "source": "user", "priority": "high"},
                self.run_id,
            )
        )
        self.events.publish("ultra.guidance", "User guidance added to Project Brain")

    def switch_model(self, descriptor: ModelDescriptor) -> None:
        if not self.safe_for_reconfiguration:
            raise RuntimeError(
                "pause ULTRA and wait for active agents to reach a safe checkpoint before switching models"
            )
        self.descriptor = descriptor
        if self.adapter:
            self.adapter.descriptor = descriptor
        if not self.orchestrator:
            return
        factory = self.orchestrator.agent_factory
        if isinstance(factory, WorkspaceUltraAgentFactory):
            factory.descriptor = descriptor
        adaptive = AdaptiveConcurrency(
            descriptor.execution_class,
            cloud_default=self.config.cloud_concurrency,
            maximum=self.config.max_concurrency,
        )
        self.orchestrator.execution_class = descriptor.execution_class
        self.orchestrator.adaptive = adaptive
        self.orchestrator.scheduler.execution_class = descriptor.execution_class
        self.orchestrator.scheduler.adaptive = adaptive
        if self.orchestrator.run_state:
            self.orchestrator.run_state = replace(
                self.orchestrator.run_state,
                execution_class=descriptor.execution_class,
                concurrency=adaptive.current,
                model_snapshot=descriptor.to_dict(),
            )
            self.orchestrator.state.save_ultra_run(self.orchestrator.run_state)
        self.events.publish(
            "ultra.model_changed",
            f"ULTRA model changed to {descriptor.provider}/{descriptor.model}",
            execution_class=descriptor.execution_class.value,
            concurrency=adaptive.current,
        )

    def switch_permissions(self, adapter: PermissionAdapter) -> None:
        if not self.safe_for_reconfiguration:
            raise RuntimeError(
                "pause ULTRA and wait for active agents to reach a safe checkpoint before changing permissions"
            )
        self.permission_adapter = adapter
        if self.adapter:
            self.adapter.access_level = adapter.access_level
        if self.orchestrator and self.orchestrator.run_state:
            self.orchestrator.state.save_ultra_run(self.orchestrator.run_state)
        self.events.publish(
            "ultra.permissions_changed",
            f"ULTRA permissions changed to {adapter.access_level.value}",
            access_level=adapter.access_level.value,
        )

    def answer(self, question_id: str, value: str) -> MasterPlanV1 | None:
        if not self.orchestrator or not self.adapter or not self.goal_id:
            raise RuntimeError("there is no live ULTRA question round")
        pending = {str(item.get("id")): item for item in self.questions()}
        if question_id not in pending:
            raise ValueError(f"unknown ULTRA question id: {question_id}")
        self.answers[question_id] = str(value).strip()
        self.store.update_goal_metadata(self.goal_id, plan_answers=dict(self.answers))
        unanswered = set(pending) - set(self.answers)
        if unanswered:
            next_id = sorted(unanswered)[0]
            self.store.update_goal_metadata(
                self.goal_id,
                waiting_question=str(pending[next_id].get("question", "")),
            )
            return None
        goal = self.store.get_goal(self.goal_id)
        if goal.status is GoalStatus.PAUSED:
            self.store.transition_goal(
                self.goal_id,
                GoalStatus.DISCOVERING,
                reason="ULTRA goal questions answered",
            )
        plan = self.orchestrator.answer_questions(self.answers)
        assert self.orchestrator.goal_spec and self.orchestrator.architecture
        self.adapter.bind_foundation(
            self.orchestrator.goal_spec,
            self.orchestrator.architecture,
            plan,
        )
        return plan

    def approve(self, revision: int | None = None) -> Plan:
        if not self.orchestrator or not self.adapter or not self.orchestrator.master_plan:
            raise RuntimeError("there is no ULTRA master plan to approve")
        if revision is not None and self.adapter.plan and revision != self.adapter.plan.revision:
            raise ValueError(f"ULTRA is awaiting plan revision {self.adapter.plan.revision}")
        self.orchestrator.approve(self.orchestrator.master_plan.fingerprint)
        accepted = self.adapter.approve_master(self.orchestrator.master_plan)
        self.future = self.orchestrator.background.start(self._run_and_finalize)
        return accepted

    def _run_and_finalize(self) -> UltraRunResult:
        assert self.orchestrator
        try:
            result = self.orchestrator.run()
        except Exception as exc:
            self._record_engine_failure(exc)
            raise
        self._finalize_result(result)
        return result

    def _record_engine_failure(self, exc: Exception) -> None:
        if not self.goal_id:
            return
        try:
            goal = self.store.get_goal(self.goal_id)
            if goal.status not in {GoalStatus.BLOCKED, GoalStatus.CANCELLED}:
                self.store.transition_goal(
                    self.goal_id,
                    GoalStatus.BLOCKED,
                    reason=f"ULTRA engine failed: {redact_text(exc, 500)}",
                )
        except Exception:
            pass
        self.events.publish("error", f"ULTRA execution failed: {redact_text(exc, 500)}")

    def _finalize_result(self, result: UltraRunResult) -> None:
        if not self.goal_id:
            return
        try:
            goal = self.store.get_goal(self.goal_id)
            if result.successful:
                if goal.status is GoalStatus.RUNNING:
                    self.store.transition_goal(self.goal_id, GoalStatus.VERIFYING, reason="ULTRA module waves completed")
                goal = self.store.get_goal(self.goal_id)
                if goal.status is GoalStatus.VERIFYING:
                    self.store.transition_goal(self.goal_id, GoalStatus.REVIEWING, reason="ULTRA global review passed")
                goal = self.store.get_goal(self.goal_id)
                if goal.status is GoalStatus.REVIEWING:
                    self.store.transition_goal(self.goal_id, GoalStatus.COMPLETED, reason="ULTRA final evidence gate passed")
            elif result.run.phase is EnginePhase.CANCELLED:
                if goal.status is not GoalStatus.CANCELLED:
                    self.store.transition_goal(self.goal_id, GoalStatus.CANCELLED, reason="ULTRA cancelled")
            elif result.run.phase is EnginePhase.REVISION_REQUIRED:
                if goal.status is GoalStatus.RUNNING:
                    self.store.transition_goal(self.goal_id, GoalStatus.REVISING, reason="ULTRA requires master-plan revision")
                self.store.update_goal_metadata(
                    self.goal_id,
                    waiting_question="A quality or scope gate requires a revised master plan.",
                    auto_retryable=False,
                )
            elif goal.status is GoalStatus.RUNNING:
                self.store.transition_goal(self.goal_id, GoalStatus.BLOCKED, reason="ULTRA module wave failed")
        except Exception as exc:
            self.events.publish("error", f"ULTRA completion persistence failed: {redact_text(exc, 500)}")

    def pause(self) -> None:
        if not self.orchestrator:
            raise RuntimeError("there is no live ULTRA run")
        self.orchestrator.pause()

    def resume(self) -> None:
        if not self.orchestrator:
            raise RuntimeError("there is no live ULTRA run")
        self.orchestrator.resume()

    def cancel(self) -> None:
        if not self.orchestrator:
            raise RuntimeError("there is no live ULTRA run")
        if self.running:
            self.orchestrator.cancel()
        elif self.orchestrator.phase not in {
            EnginePhase.COMPLETED,
            EnginePhase.CANCELLED,
            EnginePhase.FAILED,
        }:
            self.orchestrator._set_phase(EnginePhase.CANCELLED, "ULTRA cancelled")

    def close(self) -> None:
        if self.orchestrator:
            self.orchestrator.background.close()


__all__ = [
    "StateStoreUltraAdapter",
    "UltraSession",
    "WorkspaceUltraAgent",
    "WorkspaceUltraAgentFactory",
]
