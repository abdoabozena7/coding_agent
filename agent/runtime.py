"""Deterministic orchestration above the probabilistic model/tool loop.

The model proposes plans, actions, dynamic roles, and completion.  This runtime
owns plan approval, state transitions, evidence, retries, recovery, delegation
limits, and the final completion gate.
"""

from __future__ import annotations

import copy
import difflib
import hashlib
import inspect
import json
import importlib.util
import os
import platform
import re
import time
import shutil
import shlex
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from queue import Empty, Queue
from threading import Event, RLock, Thread
from typing import Any, Callable, Iterable, Mapping, Sequence

from . import context, tools
from .commands import CommandKind, UserCommand
from .chat_runtime import (
    RequestedEffectV2,
    RouteDecisionV1,
    RouteKind,
    SEMANTIC_GOAL_INTAKE_SCHEMA,
    SEMANTIC_ROUTE_SCHEMA,
    SemanticContractError,
    SemanticGoalIntakeV3,
    SemanticTurnDecisionV2,
    corrective_prompt,
)
from .capability import (
    ExecutionStrategyV1,
    InteractionModeV2,
    LocalAdaptationPolicy,
    ModelCapabilityEnvelopeV1,
    StrategyDecisionV1,
    TaskDemandV1,
    select_execution_strategy,
)
from .config import RuntimeConfig
from .control import (
    CONTROL_NAMES,
    COORDINATOR_SCHEMAS,
    DELEGATE_TASK,
    PLANNER_SCHEMAS,
    PLAN_REVIEWER_SCHEMAS,
    REVIEWER_SCHEMAS,
    WORKER_SCHEMAS,
    ControlValidationError,
    validate_control_call,
)
from .events import EventBus
from .hardware import probe_local_gpu
from .intake import (
    ClarificationQuestionV1,
    IntakeStatus,
    IntentArchitect,
    RunMode,
    answer_from_value,
    normalize_question,
    normalize_questions,
)
from .learning import GlobalLessonStore, LearnedLessonV1
from .models import (
    CompletionDisposition,
    Delegation,
    DelegationStatus,
    DomainError,
    Evidence,
    Goal,
    GoalStatus,
    Plan,
    PlanStatus,
    RoleProfile,
    Task,
    TaskStatus,
    validate_task_dag,
)
from .prompts import (
    CHAT_SYSTEM_PROMPT,
    COORDINATOR_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    PLAN_REVIEWER_SYSTEM_PROMPT,
    REVIEWER_SYSTEM_PROMPT,
    SEMANTIC_GOAL_INTAKE_SYSTEM_PROMPT,
    SEMANTIC_ROUTER_SYSTEM_PROMPT,
    state_envelope,
    subagent_system_prompt,
)
from .providers.base import AssistantTurn, ProviderActivityV1, ProviderCallPolicyV1, ToolCall
from .plan_document import parse_plan_document, render_plan_document
from .safety import ProgressWatchdog, redact_data, redact_text
from .store import (
    NotFoundError,
    StateStore,
    StateStoreError,
    WorkflowSessionConflictError,
)
from .ui import DashboardView, TaskView, WorkerView
from .workflow import (
    PlanDraftError,
    RetryKind,
    RetryLedger,
    first_ready_task,
    is_unambiguous_plan_approval,
    normalize_plan_draft,
    validate_normalized_plan,
    WorkflowBoundaryKind,
    WorkflowStageCheckpointV1,
    fingerprint as workflow_fingerprint,
    SessionMode,
    PlanState,
    RunState,
)
from .sleep_profile import SleepController
from .action_policy import ApprovalRequirement, classify_action
from .quality import ChangeSetStatus
from .run_context import GoalContractV1, is_goal_escalation_approval
from .semantic import (
    canonicalize_requirement_anchors,
    RequestedEffect,
    ResourceClaimV1,
    SemanticGoalV2,
    StrategyAttemptV1,
    VerificationContractV1,
)
from .verifiers import discover_verifier_plugins
from .weak_model import WeakModelPolicy
from .repository_index import OllamaEmbeddingProvider, RepositoryIndex
from .diagnostics import ErrorSignature, FailureDomain, normalize_error_message
from .version_control import GitProtectionManager, VersionControlError
from .local_provider import (
    extract_first_json_object,
    normalize_action_proposal,
    normalize_generated_tool_payload,
    ProviderFailureKind,
    ProviderRequestError,
    repair_structured_json_object,
)
from .ultra_models import normalize_contract_path

try:
    from .model_catalog import ExecutionClass, ModelDescriptor
    from .sandbox import AccessLevel, PermissionAdapter
except ImportError:  # pragma: no cover - direct-script compatibility
    ExecutionClass = ModelDescriptor = AccessLevel = PermissionAdapter = Any  # type: ignore


ApprovalCallback = Callable[[str, dict[str, Any], str], Any]

READ_ONLY_TOOLS = tools.names(categories={"read"})
MUTATING_TOOLS = tools.names(mutating=True)
TOOL_RISK = tools.risk_map()


class RuntimeErrorBase(RuntimeError):
    pass


class RuntimeStateError(RuntimeErrorBase):
    pass


class HardStopError(RuntimeStateError):
    """The user explicitly stopped the active provider/tool boundary."""


class ProviderUnavailableError(RuntimeErrorBase):
    pass


def _provider_retry_after_seconds(exc: BaseException) -> int | None:
    value = getattr(exc, "retry_after", None)
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", {}) if response is not None else {}
    if value is None and isinstance(headers, Mapping):
        value = headers.get("retry-after") or headers.get("Retry-After")
    try:
        parsed = int(float(value))
    except (TypeError, ValueError):
        match = re.search(r"retry[- ]after[^0-9]*(\d+)", str(exc), re.IGNORECASE)
        parsed = int(match.group(1)) if match else 0
    return max(1, min(parsed, 86_400)) if parsed > 0 else None


def _provider_is_temporarily_overloaded(exc: BaseException) -> bool:
    """Return whether retrying immediately would only repeat provider backpressure.

    Ollama's cloud bridge commonly holds the request open for about a minute
    before returning HTTP 503. Replaying that response through the ordinary
    transport retry loop turns one recoverable boundary into several minutes
    of apparent inactivity. This classifier is transport policy, not semantic
    fallback: the exact turn stays durable and no route is guessed.
    """

    diagnostic = getattr(exc, "diagnostic", None)
    status_code = getattr(diagnostic, "status_code", None)
    provider_message = str(getattr(diagnostic, "provider_message", "") or "")
    text = f"{provider_message} {exc}".casefold()
    overloaded = "overloaded" in text or "service unavailable" in text
    return bool(overloaded and (status_code in {None, 503} or "503" in text))


@dataclass(frozen=True)
class SliceResult:
    status: str
    message: str
    steps: int = 0
    completed: bool = False
    needs_user: bool = False
    disposition: str | None = None
    limitations: tuple[str, ...] = ()
    # Optional boundary metadata.  These fields intentionally come last so
    # persisted/third-party callers using the original positional contract
    # remain valid.
    phase: str = ""
    reason: str = ""
    waiting_on: str = ""
    last_tool: str = ""
    workspace_mutated: bool = False
    resume_action: str = ""
    heartbeat_at: float | None = None


@dataclass(frozen=True)
class WorkflowRuntimeSnapshotV1:
    """One truthful, presentation-neutral view of the durable workflow.

    This is deliberately derived from the store on every call.  It is not a
    second state machine: a goal status wins over stale UI/runtime events,
    followed by an active question/approval, then the latest event.
    """

    session_mode: str = "ready"
    route: str = "pending"
    execution_strategy: str = "pending"
    model: str = "-"
    provider: str = "-"
    capability_band: str = "minimal"
    phase: str = "ready"
    current_task: str = ""
    last_tool: str = ""
    waiting_on: str = ""
    reason: str = ""
    last_event_id: str = ""
    heartbeat_at: float | None = None
    heartbeat_age: float | None = None
    workspace_mutated: bool = False
    resume_action: str = ""
    objective: str = ""
    current_task_id: str = ""
    active_actor: str = ""
    active_step: int = 0
    activity_sequence: int = 0
    liveness: str = "ready"
    active_operation: str = ""
    provider_request_state: str = "idle"
    received_bytes: int = 0
    received_chunks: int = 0
    received_tokens: int = 0
    last_signal_at: float | None = None
    timeline_preview: tuple[dict[str, Any], ...] = ()
    stream_state: str = "idle"
    stream_kind: str = "none"
    safe_stream_preview: str = ""
    first_byte_at: float | None = None
    task_items: tuple[dict[str, Any], ...] = ()
    next_task: str = ""
    # Durable identity for reconciling Web and terminal projections.  These
    # values describe the saved session/content and the current provider
    # attempt; they must not be inferred from a UI poll timestamp.
    session_revision: int = 0
    content_revision: int = 0
    attempt_id: str = ""
    attempt_state: str = "idle"
    attempt_model: str = ""
    retry_at: float | None = None
    failure_kind: str = ""
    local_adaptation_policy: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_mode": self.session_mode,
            "route": self.route,
            "execution_strategy": self.execution_strategy,
            "model": self.model,
            "provider": self.provider,
            "capability_band": self.capability_band,
            "phase": self.phase,
            "current_task": self.current_task,
            "last_tool": self.last_tool,
            "waiting_on": self.waiting_on,
            "reason": self.reason,
            "last_event_id": self.last_event_id,
            "heartbeat_at": self.heartbeat_at,
            "heartbeat_age": self.heartbeat_age,
            "workspace_mutated": self.workspace_mutated,
            "resume_action": self.resume_action,
            "objective": self.objective,
            "current_task_id": self.current_task_id,
            "active_actor": self.active_actor,
            "active_step": self.active_step,
            "activity_sequence": self.activity_sequence,
            "liveness": self.liveness,
            "active_operation": self.active_operation,
            "provider_request_state": self.provider_request_state,
            "received_bytes": self.received_bytes,
            "received_chunks": self.received_chunks,
            "received_tokens": self.received_tokens,
            "last_signal_at": self.last_signal_at,
            "timeline_preview": [dict(item) for item in self.timeline_preview],
            "stream_state": self.stream_state,
            "stream_kind": self.stream_kind,
            "safe_stream_preview": self.safe_stream_preview,
            "first_byte_at": self.first_byte_at,
            "task_items": [dict(item) for item in self.task_items],
            "next_task": self.next_task,
            "session_revision": self.session_revision,
            "content_revision": self.content_revision,
            "attempt_id": self.attempt_id,
            "attempt_state": self.attempt_state,
            "attempt_model": self.attempt_model,
            "retry_at": self.retry_at,
            "failure_kind": self.failure_kind,
            "local_adaptation_policy": dict(self.local_adaptation_policy),
        }


@dataclass(frozen=True)
class WorkflowModeLock:
    locked: bool
    reason: str = ""
    stage: str = "idle"


def _tool_name(schema: Mapping[str, Any]) -> str:
    return str(schema.get("function", {}).get("name", ""))


def _external_schema_map() -> dict[str, dict[str, Any]]:
    return {_tool_name(schema): schema for schema in tools.TOOL_SCHEMAS}


def _schemas(names: Iterable[str]) -> list[dict[str, Any]]:
    wanted = set(names)
    return [schema for schema in tools.TOOL_SCHEMAS if _tool_name(schema) in wanted]


def _task_dict(task: Task) -> dict[str, Any]:
    return {
        "id": task.id,
        "title": task.title,
        "description": task.description,
        "parent_id": task.parent_id,
        "status": task.status.value,
        "depends_on": list(task.depends_on),
        "acceptance_criteria": list(task.acceptance_criteria),
        "verification": list(task.verification),
        "role": task.role.to_dict(),
        "mode": task.mode,
        "risk": task.risk,
        "priority": task.priority,
        "attempts": task.attempts,
        "origin": task.origin,
        "metadata": dict(task.metadata),
    }


def _display_task_status(status: TaskStatus) -> str:
    return {
        TaskStatus.COMPLETED: "done",
        TaskStatus.OBSOLETE: "skipped",
        TaskStatus.IN_PROGRESS: "in_progress",
        TaskStatus.VERIFYING: "in_progress",
        TaskStatus.BLOCKED: "blocked",
        TaskStatus.FAILED: "blocked",
        TaskStatus.UNCERTAIN: "uncertain",
        TaskStatus.CANCELLED: "skipped",
    }.get(status, "pending")


class AgentRuntime:
    """Persistent coordinator with injectable provider, approvals, and clock."""

    def __init__(
        self,
        provider: Any,
        store: StateStore,
        workspace: str | Path,
        *,
        events: EventBus | None = None,
        approval: ApprovalCallback | None = None,
        config: RuntimeConfig | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        model_descriptor: ModelDescriptor | None = None,
        permission_adapter: PermissionAdapter | None = None,
        session_id: str = "workspace-session",
    ) -> None:
        self.provider = provider
        self.store = store
        self.workspace = Path(workspace).resolve(strict=True)
        self.version_control = GitProtectionManager(self.workspace)
        self.events = events or EventBus()
        self.approval = approval or (lambda _name, _args, _risk: False)
        # Optional bridge used by the loopback web workspace.  The terminal
        # remains the owner of the live approval request; the bridge can only
        # resolve a matching fingerprint through that owner.
        self._external_tool_approval_resolver: Callable[[str, str], bool] | None = None
        self.config = config or RuntimeConfig.from_env()
        self.sleeper = sleeper
        self.model_descriptor = model_descriptor
        self.permission_adapter = permission_adapter
        self.ultra_session: Any | None = None
        self._closed = False
        self._lock = RLock()
        # Provider callbacks run on a dedicated worker while workflow methods
        # may intentionally hold ``_lock`` across a governed model turn. Live
        # counters must never contend with that state lock or both threads can
        # wait on each other. Presentation telemetry therefore has its own
        # narrow lock and never mutates durable workflow state.
        self._live_activity_lock = RLock()
        self._work_conversation: list[dict[str, Any]] = []
        self._watchdog = ProgressWatchdog(self.config.repeated_action_limit)
        self._delegations_this_slice = 0
        self._provider_input_tokens = 0
        self._provider_output_tokens = 0
        self._live_provider_activity: dict[str, Any] = {
            "state": "idle",
            "actor": "",
            "operation": "",
            "received_bytes": 0,
            "received_chunks": 0,
            "received_tokens": 0,
            "started_at": None,
            "last_signal_at": None,
        }
        try:
            provider_parameters = inspect.signature(self.provider.call).parameters.values()
            self._provider_accepts_activity = any(
                item.name == "on_activity" or item.kind is inspect.Parameter.VAR_KEYWORD
                for item in provider_parameters
            )
        except (TypeError, ValueError):
            self._provider_accepts_activity = False
        try:
            provider_parameters = inspect.signature(self.provider.call).parameters.values()
            self._provider_accepts_policy = any(
                item.name == "policy" or item.kind is inspect.Parameter.VAR_KEYWORD
                for item in provider_parameters
            )
        except (TypeError, ValueError):
            self._provider_accepts_policy = False
        self._stop_event = Event()
        self._active_provider_abandon: Event | None = None
        self._worker_id = f"worker-{uuid.uuid4().hex[:16]}"
        self._foreign_execution_owner_live = False
        self.retry_ledger = RetryLedger()
        self._chat_conversation: list[dict[str, Any]] = []
        self.session_id = str(session_id).strip() or "workspace-session"
        self.local_web_server: Any | None = None
        self.sleep_controller = SleepController()
        self.weak_model_policy = WeakModelPolicy()
        self.intent_architect = IntentArchitect()
        model_name = str(getattr(provider, "model", "")).casefold()
        self._global_memory_enabled = not model_name.startswith(("offline", "fake", "test"))
        self.global_lessons = GlobalLessonStore()
        self._used_global_lesson_ids: set[str] = set()
        self.repository_index = RepositoryIndex(
            self.workspace,
            embedding_provider=OllamaEmbeddingProvider.from_environment(),
            cache_path=self.workspace / ".coding-agent" / "repository-index-v1.json",
        )
        self.repository_index_warmup_error = ""
        if self.config.repository_index_warmup_files > 0:
            try:
                self.repository_index.update_all(
                    max_files=self.config.repository_index_warmup_files,
                )
                self.store.sync_repository_index(self.workspace, self.repository_index)
            except (OSError, UnicodeError, ValueError) as exc:
                self.repository_index_warmup_error = f"{type(exc).__name__}: {exc}"
        try:
            self.store.get_workflow_session(self.session_id)
        except NotFoundError:
            self.store.save_workflow_session(
                self.session_id,
                goal_id=None,
                session_mode=SessionMode.NORMAL.value,
                plan_state=PlanState.NONE.value,
                run_state=RunState.IDLE.value,
                state=self._session_runtime_snapshot(),
            )
        else:
            # A second process may open the same session for a read-only
            # command while the real worker is blocked in a model/tool call.
            # It must not overwrite the worker identity or run crash recovery.
            self._foreign_execution_owner_live = bool(
                self._live_foreign_execution_lease()
            )
            if not self._foreign_execution_owner_live:
                self._persist_runtime_snapshot()
        self._chat_conversation = [dict(item) for item in self.store.list_chat_messages(self.session_id)]
        tools.register_artifact_provider(self.workspace, self.store.get_chat_artifact)
        tools.configure_workspace(self.workspace)

        active_policy_goal = self.store.load_active_goal(self.session_id)
        if active_policy_goal is not None:
            persisted_policy = active_policy_goal.metadata.get("weak_model_policy")
            if isinstance(persisted_policy, Mapping):
                self.weak_model_policy = WeakModelPolicy.from_dict(persisted_policy)

        recovery = (
            None
            if self._foreign_execution_owner_live
            else self.store.recover_inflight()
        )
        if recovery is not None and recovery.changed:
            self.events.publish(
                "recovery",
                "Interrupted actions were marked uncertain and were not replayed.",
                tasks=list(recovery.task_ids),
                delegations=list(recovery.delegation_ids),
                actions=list(recovery.action_ids),
            )
            goal = self.store.load_active_goal(self.session_id)
            if goal and goal.status == GoalStatus.RECOVERING:
                uncertain_sets = list(goal.metadata.get("goal_change_sets", ()))
                for action_id in recovery.action_ids:
                    uncertain_sets.append({
                        "id": f"goal-changeset-uncertain-{action_id}",
                        "version": 1,
                        "responsible_agent": "interrupted-runtime",
                        "parent_task": None,
                        "changed_files": [],
                        "pre_hashes": {},
                        "post_hashes": {},
                        "diff": "",
                        "tool_action_ids": [action_id],
                        "review_status": "uncertain",
                        "integration_status": "uncertain",
                        "mutation_sequence": goal.metadata.get("mutation_sequence", 0),
                    })
                if uncertain_sets:
                    self.store.update_goal_metadata(
                        goal.id,
                        goal_change_sets=uncertain_sets,
                        convergence_state="reverifying",
                        latest_evaluation_stale=True,
                    )
                resume_status = (
                    GoalStatus.RUNNING.value
                    if goal.active_plan_revision is not None
                    else GoalStatus.DISCOVERING.value
                )
                self.store.update_goal_metadata(
                    goal.id,
                    waiting_question=(
                        "The previous run stopped during work. Inspect uncertain tasks/actions, "
                        "then use /resume when their real state is known."
                    ),
                    resume_status=resume_status,
                )
                self.store.transition_goal(goal.id, GoalStatus.PAUSED, reason="crash recovery requires user inspection")

        ultra_recovery = (
            None
            if self._foreign_execution_owner_live
            else self.store.recover_ultra_inflight()
        )
        if ultra_recovery is not None and ultra_recovery.changed:
            self.events.publish(
                "recovery",
                "Interrupted ULTRA agents and write leases were marked uncertain; nothing was replayed.",
                runs=list(ultra_recovery.ultra_run_ids),
                nodes=list(ultra_recovery.work_node_ids),
                agents=list(ultra_recovery.agent_run_ids),
                leases=list(ultra_recovery.lease_ids),
            )
            for run_id in ultra_recovery.ultra_run_ids:
                try:
                    recovered_run = self.store.get_ultra_run(run_id)
                    recovered_goal = self.store.get_goal(recovered_run.goal_id)
                    self.store.update_goal_metadata(
                        recovered_goal.id,
                        ultra_run_id=run_id,
                        resume_status=GoalStatus.RUNNING.value,
                        waiting_question=(
                            "ULTRA stopped between evidence gates. Inspect and reconcile every "
                            "UNCERTAIN node/action, then use /resume. Nothing is replayed automatically."
                        ),
                        auto_retryable=False,
                    )
                    if recovered_goal.status == GoalStatus.RUNNING:
                        self.store.transition_goal(
                            recovered_goal.id,
                            GoalStatus.PAUSED,
                            reason="ULTRA crash recovery requires uncertain-work inspection",
                        )
                except (NotFoundError, StateStoreError, DomainError):
                    continue

        auto_reconciled = (
            ()
            if self._foreign_execution_owner_live
            else self._auto_reconcile_read_only_ultra_uncertainty()
        )
        if auto_reconciled:
            recovered_goal = self.store.load_active_goal(self.session_id)
            if recovered_goal is not None:
                self.store.update_goal_metadata(
                    recovered_goal.id,
                    waiting_question=(
                        "Interrupted read-only/component-package work was safely reset to its "
                        "durable checkpoint; use /resume to continue."
                    ),
                    resume_status=GoalStatus.RUNNING.value,
                )
            self.events.publish(
                "recovery",
                "Read-only ULTRA uncertainty was reconciled automatically; no workspace write was replayed.",
                entities=list(auto_reconciled),
            )

        # Planning/review phases are model-call transients. A process can stop
        # there without an action row, so normalize them to an explicit paused
        # checkpoint instead of leaving a non-runnable goal stranded.
        goal = self.store.load_active_goal(self.session_id)
        if not self._foreign_execution_owner_live and goal and goal.status in {
            GoalStatus.DISCOVERING,
            GoalStatus.REVISING,
            GoalStatus.VERIFYING,
            GoalStatus.REVIEWING,
        }:
            resume_status = (
                goal.status.value
                if goal.status in {GoalStatus.DISCOVERING, GoalStatus.REVISING}
                else GoalStatus.RUNNING.value
            )
            self.store.update_goal_metadata(
                goal.id,
                resume_status=resume_status,
                waiting_question=(
                    "The previous process stopped during planning/review. Use /resume to restart "
                    "that phase from durable goal and evidence state."
                ),
            )
            self.store.transition_goal(goal.id, GoalStatus.PAUSED, reason="transient phase interrupted")

        # Schema-v1 plans did not bind approval to workspace applicability.
        # Never silently grandfather an active implementation plan into the
        # stronger v2 contract; require a fresh inspected revision and approval.
        goal = self.store.load_active_goal(self.session_id)
        if goal and goal.status == GoalStatus.RUNNING:
            accepted = self.store.get_accepted_plan(goal.id)
            ultra_run_id = str(goal.metadata.get("ultra_run_id") or "").strip()
            has_bound_ultra_foundation = bool(
                accepted
                and ultra_run_id
                and any(
                    str(item.get("source") or "").strip()
                    == f"ultra:{ultra_run_id}:foundation"
                    for item in accepted.applicability_evidence
                )
            )
            if accepted and (
                not accepted.applicability_evidence
                or not accepted.execution_strategy.strip()
                or (
                    not goal.metadata.get("semantic_goal")
                    and not has_bound_ultra_foundation
                )
            ):
                self.store.update_goal_metadata(
                    goal.id,
                    resume_status=GoalStatus.REVISING.value,
                    waiting_question=(
                        "The accepted legacy plan predates applicability evidence. Use /resume or /replan "
                        "to inspect the workspace and create a newly approved executable plan."
                    ),
                    auto_retryable=False,
                )
                self.store.transition_goal(
                    goal.id,
                    GoalStatus.PAUSED,
                    reason="legacy accepted plan requires evidence-bound revision",
                )
        goal = self.store.load_active_goal(self.session_id)
        if goal and goal.status is GoalStatus.AWAITING_PLAN_APPROVAL:
            pending = self.store.get_latest_plan(goal.id)
            if (
                pending is not None
                and pending.status is PlanStatus.PENDING_APPROVAL
                and not goal.metadata.get("accepted_semantic_fingerprint")
            ):
                self.store.update_goal_metadata(
                    goal.id,
                    legacy_semantic_enrichment_required=True,
                    waiting_question=(
                        "This pending plan predates staged semantic acceptance. "
                        "Approval will first create an inspected semantic-enriched "
                        "revision for fresh review."
                    ),
                )

    @property
    def provider_name(self) -> str:
        return self.provider.__class__.__name__.removesuffix("Provider").lower() or "provider"

    @property
    def model_name(self) -> str:
        return str(getattr(self.provider, "model", "unknown"))

    @property
    def reasoning_effort(self) -> str:
        return str(getattr(self.provider, "reasoning_effort", "medium"))

    def model_capability_envelope(self) -> ModelCapabilityEnvelopeV1:
        """Return the metadata-only capability snapshot used for new work."""

        return self._capability_envelope_for(self.provider, self.model_descriptor)

    def local_adaptation_policy(
        self,
        envelope: ModelCapabilityEnvelopeV1 | None = None,
    ) -> dict[str, Any]:
        """Return packet adaptation without changing the user's workflow mode."""

        selected = envelope or self.model_capability_envelope()
        return LocalAdaptationPolicy.from_envelope(selected).to_dict()

    def _workflow_concurrency_limit(self) -> int:
        """Return the model-aware worker limit for the active workflow.

        Session/UI preferences are never allowed to widen a capability decision.
        Before routing, the conservative metadata envelope is authoritative;
        after routing, the persisted strategy decision is authoritative.
        """

        goal = self.active_goal()
        if goal is not None:
            decision = goal.metadata.get("strategy_decision")
            if isinstance(decision, Mapping):
                try:
                    return max(1, min(8, int(decision.get("max_concurrency", 1))))
                except (TypeError, ValueError):
                    pass
        return max(1, min(8, int(self.model_capability_envelope().max_concurrency)))

    def _capability_envelope_for(
        self,
        provider_instance: Any,
        descriptor: ModelDescriptor | None,
    ) -> ModelCapabilityEnvelopeV1:
        """Build a conservative envelope without mutating the selected model."""

        profile = getattr(provider_instance, "capability_profile", None)
        provider_capabilities = getattr(provider_instance, "capabilities", None)
        if descriptor is not None:
            capabilities = descriptor.capabilities
            metadata = descriptor.metadata
            execution_class = descriptor.execution_class.value
            concurrency = int(
                self.config.ultra_local_concurrency
                if descriptor.execution_class is ExecutionClass.LOCAL
                else self.config.ultra_cloud_concurrency
            )
            provider = descriptor.provider
            model = descriptor.model
        else:
            capabilities = tuple(
                name
                for name, enabled in {
                    "tools": bool(getattr(provider_capabilities, "supports_tools", False)),
                    "structured_output": bool(getattr(provider_capabilities, "supports_json_schema", False)),
                    "thinking": bool(getattr(provider_capabilities, "supports_thinking", False)),
                    "vision": bool(getattr(provider_capabilities, "supports_vision", False)),
                }.items()
                if enabled
            )
            metadata = {}
            provider_name = (
                provider_instance.__class__.__name__.removesuffix("Provider").lower()
                or "provider"
            )
            model_name = str(getattr(provider_instance, "model", "unknown"))
            execution_class = (
                "cloud"
                if provider_name in {"openai", "gemini"}
                or model_name.casefold().endswith((":cloud", "-cloud"))
                else "local"
            )
            concurrency = int(
                self.config.ultra_local_concurrency
                if execution_class == "local"
                else self.config.ultra_cloud_concurrency
            )
            provider = provider_name
            model = model_name
        return ModelCapabilityEnvelopeV1.from_metadata(
            provider=provider,
            model=model,
            execution_class=execution_class,
            capabilities=capabilities,
            metadata=metadata,
            provider_profile=profile,
            default_concurrency=concurrency,
        )

    def _session_runtime_snapshot(self) -> dict[str, Any]:
        descriptor = (
            self.model_descriptor.to_dict()
            if self.model_descriptor is not None
            else {
                "provider": self.provider_name,
                "model": self.model_name,
                "execution_class": self.execution_class,
            }
        )
        return {
            "model_snapshot": descriptor,
            "model_capability_envelope": self.model_capability_envelope().to_dict(),
            "local_adaptation_policy": self.local_adaptation_policy(),
            "reasoning_effort": self.reasoning_effort,
            "access_level": self.access_level,
            "concurrency": self._workflow_concurrency_limit(),
            "checkpoint": "safe" if not (
                self.ultra_session is not None and self.ultra_session.running
            ) else "running",
        }

    def _ensure_workflow_session(self) -> dict[str, Any]:
        """Recover a missing presentation row without losing the durable goal.

        The SQLite journal is the authority, but a freshly opened Web/terminal
        pair can observe the tiny window between session initialization and its
        first committed snapshot.  A missing row must not turn the controller
        into an internal-error recovery card. Recreate only the secret-free
        session envelope and bind it to the latest goal, if one exists; goal,
        plan, task, and evidence rows remain untouched.
        """

        try:
            return self.store.get_workflow_session(self.session_id)
        except NotFoundError:
            goal = self.store.get_latest_goal(self.session_id)
            if goal is None:
                goal_id = None
                session_mode = SessionMode.NORMAL.value
                plan_state = PlanState.NONE.value
                run_state = RunState.IDLE.value
            else:
                goal_id = goal.id
                policy = goal.metadata.get("execution_policy")
                policy_mode = (
                    str(policy.get("mode") or "")
                    if isinstance(policy, Mapping)
                    else ""
                )
                try:
                    session_mode = (
                        SessionMode.parse(policy_mode).value
                        if policy_mode
                        else SessionMode.NORMAL.value
                    )
                except ValueError:
                    session_mode = SessionMode.NORMAL.value
                latest_plan = self.store.get_latest_plan(goal.id)
                if goal.active_plan_revision is not None:
                    plan_state = PlanState.APPROVED.value
                elif latest_plan is not None and latest_plan.status is PlanStatus.PENDING_APPROVAL:
                    plan_state = PlanState.AWAITING_APPROVAL.value
                else:
                    plan_state = PlanState.INSPECTING.value
                run_state = {
                    GoalStatus.COMPLETED: RunState.COMPLETED.value,
                    GoalStatus.CANCELLED: RunState.CANCELLED.value,
                    GoalStatus.PAUSED: RunState.BLOCKED.value,
                    GoalStatus.BLOCKED: RunState.BLOCKED.value,
                    GoalStatus.RUNNING: RunState.EXECUTING.value,
                    GoalStatus.VERIFYING: RunState.VERIFYING.value,
                    GoalStatus.REVIEWING: RunState.REVIEWING.value,
                }.get(goal.status, RunState.PLANNING.value)
            self.store.save_workflow_session(
                self.session_id,
                goal_id=goal_id,
                session_mode=session_mode,
                plan_state=plan_state,
                run_state=run_state,
                state=self._session_runtime_snapshot(),
            )
            self.store.append_event(
                "workflow.session_recreated",
                goal_id=goal_id,
                entity_type="session",
                entity_id=self.session_id,
                payload={"reason": "missing session envelope observed during UI/runtime read"},
            )
            return self.store.get_workflow_session(self.session_id)

    def _persist_runtime_snapshot(self, **extra: Any) -> None:
        """Merge secret-free runtime identity into the durable session row."""

        def reduce_session(current: dict[str, Any]) -> Mapping[str, Any]:
            state = {
                **dict(current.get("state", {})),
                **self._session_runtime_snapshot(),
                **extra,
            }
            return {
                "state": state,
                "goal_id": current.get("goal_id"),
                "session_mode": str(current.get("session_mode") or SessionMode.NORMAL.value),
                "plan_state": str(current.get("plan_state") or PlanState.NONE.value),
                "run_state": str(current.get("run_state") or RunState.IDLE.value),
                "ultra_profile": str(current.get("ultra_profile") or "standard"),
                "sleep_state": str(current.get("sleep_state") or "off"),
            }

        self.store.mutate_workflow_session(self.session_id, reduce_session)

    def session_snapshot(self) -> Mapping[str, Any]:
        return dict(self._ensure_workflow_session().get("state", {}))

    def workflow_runtime_snapshot(self) -> WorkflowRuntimeSnapshotV1:
        """Derive the single status snapshot consumed by all frontends."""

        session = self._ensure_workflow_session()
        state = dict(session.get("state") or {})
        try:
            session_revision = max(0, int(session.get("revision") or 0))
        except (TypeError, ValueError):
            session_revision = 0
        try:
            content_revision = max(0, int(state.get("content_revision") or 0))
        except (TypeError, ValueError):
            content_revision = 0
        goal = self.active_goal()
        pending = state.get("pending_semantic_turn")
        pending = pending if isinstance(pending, Mapping) else None
        # Route is accepted semantic data, never inferred from presentation.
        route = str(state.get("route") or "pending").casefold()
        strategy_raw = state.get("execution_strategy")
        if not strategy_raw and isinstance(state.get("strategy_decision"), Mapping):
            strategy_raw = state["strategy_decision"].get("strategy")
        strategy = str(strategy_raw or "").casefold()
        for source in (goal.metadata if goal is not None else {}, pending or {}):
            if route == "pending":
                decision = source.get("route_decision") or source.get("decision")
                if isinstance(decision, Mapping):
                    route = str(decision.get("route") or "pending").casefold()
            if not strategy:
                raw_strategy = source.get("execution_strategy")
                if not raw_strategy and isinstance(source.get("strategy_decision"), Mapping):
                    raw_strategy = source["strategy_decision"].get("strategy")
                if raw_strategy:
                    strategy = str(raw_strategy).casefold()
        # An active durable Goal is itself the authoritative route.  Older
        # sessions can retain ``route=pending`` when the route turn was
        # completed just before the goal row was created; keeping that stale
        # value makes the TUI claim that a running project is still routing.
        if goal is not None and route in {"", "pending", "unknown"}:
            route = "goal"
        if goal is not None and strategy in {"", "pending", "unknown"}:
            legacy_policy = goal.metadata.get("execution_policy")
            if isinstance(legacy_policy, Mapping):
                strategy = str(
                    legacy_policy.get("strategy") or legacy_policy.get("mode") or ""
                ).casefold()
        if strategy in {"normal", "working"}:
            strategy = "staged"
        if strategy in {"ultra", "deep"}:
            strategy = "recursive"
        if not strategy:
            strategy = "pending"

        envelope = state.get("model_capability_envelope")
        if not isinstance(envelope, Mapping):
            try:
                envelope = self.model_capability_envelope().to_dict()
            except Exception:
                envelope = {}
        provider = str(envelope.get("provider") or self.provider_name)
        model = str(envelope.get("model") or self.model_name)
        capability_band = str(envelope.get("capability_band") or "minimal")
        local_policy_raw = state.get("local_adaptation_policy")
        if not isinstance(local_policy_raw, Mapping) and goal is not None:
            local_policy_raw = goal.metadata.get("local_adaptation_policy")
        if not isinstance(local_policy_raw, Mapping) and pending is not None:
            local_policy_raw = pending.get("local_adaptation_policy")
        try:
            local_policy = (
                dict(local_policy_raw)
                if isinstance(local_policy_raw, Mapping)
                else self.local_adaptation_policy(
                    ModelCapabilityEnvelopeV1.from_mapping(envelope)
                )
            )
        except (TypeError, ValueError):
            local_policy = {}

        # Durable goal status has priority over stale approval/activity events.
        phase = "ready"
        reason = "Ready for a request."
        waiting_on = ""
        resume_action = ""
        current_task = ""
        current_task_id = ""
        objective = str(
            goal.objective
            if goal is not None
            else (pending or {}).get("original_input") or ""
        )
        raw_execution_lease = state.get("execution_lease")
        execution_lease = (
            dict(raw_execution_lease)
            if isinstance(raw_execution_lease, Mapping)
            else {}
        )
        try:
            lease_active = (
                str(execution_lease.get("lease_state") or "").casefold()
                == "active"
                and float(execution_lease.get("expires_at") or 0.0) > time.time()
            )
        except (TypeError, ValueError):
            lease_active = False
        active_actor = ""
        active_step = 0
        last_tool = ""
        task_items: list[dict[str, Any]] = []
        next_task = ""
        mutated = bool((goal.metadata if goal is not None else {}).get("workspace_mutated"))
        if goal is not None:
            phase_by_status = {
                GoalStatus.NEW: "routing",
                GoalStatus.DISCOVERING: "planning",
                GoalStatus.AWAITING_PLAN_APPROVAL: "awaiting_approval",
                GoalStatus.RUNNING: "working",
                GoalStatus.REVISING: "planning",
                GoalStatus.VERIFYING: "reviewing",
                GoalStatus.REVIEWING: "reviewing",
                GoalStatus.PAUSED: "paused",
                GoalStatus.RECOVERING: "retrying",
                GoalStatus.BLOCKED: "paused",
                GoalStatus.COMPLETED: "completed",
                GoalStatus.CANCELLED: "completed",
            }
            phase = phase_by_status.get(goal.status, str(goal.status.value))
            # A live execution lease is stronger than a stale blocked/paused
            # projection.  Never tell the user that work is blocked while a
            # worker still owns the current mutation boundary.
            if lease_active and goal.status in {
                GoalStatus.PAUSED,
                GoalStatus.BLOCKED,
            }:
                phase = "working"
            is_boundary = goal.status in {
                GoalStatus.PAUSED,
                GoalStatus.RECOVERING,
                GoalStatus.BLOCKED,
            }
            reason = (
                str(
                    goal.metadata.get("waiting_question")
                    or goal.metadata.get("retry_reason")
                    or ""
                )
                if is_boundary
                else ""
            )
            if lease_active and goal.status in {
                GoalStatus.PAUSED,
                GoalStatus.BLOCKED,
            }:
                reason = "A live worker owns the current execution checkpoint."
                waiting_on = "model"
                resume_action = ""
            waiting_on = (
                str(goal.metadata.get("waiting_on") or "")
                if goal.status in {GoalStatus.RUNNING, GoalStatus.PAUSED, GoalStatus.RECOVERING}
                else ""
            )
            resume_action = (
                str(goal.metadata.get("resume_action") or "Retry")
                if is_boundary
                else ""
            )
            mutated = mutated or bool(goal.metadata.get("mutation_sequence", 0))
            plan = self.store.get_latest_plan(goal.id)
            if plan is not None:
                try:
                    content_revision = max(content_revision, int(plan.revision))
                except (TypeError, ValueError):
                    pass
            if goal.status is GoalStatus.AWAITING_PLAN_APPROVAL:
                waiting_on = "user"
                resume_action = "Review plan"
                reason = (
                    f"Plan r{plan.revision} is ready for review."
                    if plan is not None
                    else "The plan is ready for review."
                )
            if plan is not None:
                active = [task for task in plan.tasks if task.status.value in {"in_progress", "verifying"}]
                if active:
                    current_task_id = active[0].id
                    current_task = f"{active[0].id} · {active[0].title}"
            if plan is not None:
                for task in plan.tasks:
                    status = str(task.status.value if hasattr(task.status, "value") else task.status)
                    task_items.append({
                        "id": str(task.id),
                        "title": str(task.title),
                        "status": status,
                        "dependencies": list(getattr(task, "depends_on", ()) or ()),
                        "evidence": " ".join(str(getattr(task, "evidence", "") or "").split()),
                    })
                    if not next_task and status in {"pending", "ready", "queued"}:
                        next_task = str(task.title)
            actions = self.store.list_actions(goal.id, status="running")
            if actions:
                last_tool = str(actions[-1].get("tool_name") or "")
                if not waiting_on:
                    waiting_on = "tool"
        elif pending is not None and str(pending.get("status") or "").casefold() != "completed":
            phase = "retrying" if str(pending.get("status")) == "awaiting_provider" else "routing"
            reason = str(pending.get("last_error") or "Routing the saved request.")
            waiting_on = "provider" if phase == "retrying" else "model"
            resume_action = "Retry" if phase == "retrying" else ""
            route = route if route != "" else "pending"
        else:
            # No goal and no pending turn means the session has not started.
            phase = "ready"
            route = route if route not in {"", "unknown"} else "pending"
            strategy = strategy if strategy != "" else "pending"

        latest = None
        try:
            events = (
                self.store.list_recent_events(goal.id, limit=40)
                if goal is not None
                else ()
            )
            if events:
                latest = events[-1]
                payload = dict(latest.payload)
                # State-transition events do not all repeat actor/step context.
                # Preserve the newest explicit runtime identity instead of
                # clearing the activity strip whenever a bookkeeping event wins.
                contextual_payload = next(
                    (
                        dict(event.payload)
                        for event in reversed(events)
                        if event.payload.get("actor") or event.payload.get("active_actor")
                    ),
                    {},
                )
                active_actor = str(
                    contextual_payload.get("active_actor")
                    or contextual_payload.get("actor")
                    or ""
                )
                try:
                    active_step = max(
                        0,
                        int(
                            contextual_payload.get("active_step")
                            or contextual_payload.get("step")
                            or 0
                        ),
                    )
                except (TypeError, ValueError):
                    active_step = 0
                running_actions = {
                    str(item.get("id") or ""): item
                    for item in self.store.list_actions(goal.id, status="running")
                } if goal is not None else {}
                event_action_active = bool(
                    latest.entity_id and str(latest.entity_id) in running_actions
                )
                if goal is not None and goal.status is GoalStatus.RUNNING:
                    if latest.event_type == "approval.requested":
                        phase = "waiting_for_approval"
                        waiting_on = "user"
                        reason = str(
                            payload.get("reason")
                            or f"Approval is required for {payload.get('tool') or latest.entity_id or 'the next action'}."
                        )
                    elif latest.event_type == "process.waiting":
                        if event_action_active:
                            phase = "waiting_for_process"
                            waiting_on = "process"
                    elif latest.event_type in {"tool.started", "execution.started"}:
                        if latest.event_type == "execution.started" or event_action_active:
                            phase = "starting" if latest.event_type == "execution.started" else "working"
                    elif latest.event_type == "approval.received":
                        phase = "starting"
                        waiting_on = "tool"
                    elif latest.event_type in {"tool.completed", "tool.failed"}:
                        # A completed action is ordinary work.  A failed action
                        # may still be recoverable by the coordinator, but it
                        # must never be projected as normal ``Working``: that
                        # made the TUI look healthy while the user was staring
                        # at a tool error.  ``retrying`` is an active repair
                        # state; a later durable boundary promotes it to
                        # ``paused`` with an explicit recovery action.
                        phase = "working"
                        waiting_on = ""
                        if latest.event_type == "tool.failed":
                            phase = "retrying"
                            waiting_on = "coordinator"
                            reason = str(
                                payload.get("result")
                                or "The last tool failed; the coordinator is repairing it."
                            )
                            resume_action = "Retry"
                last_tool = last_tool or str(payload.get("tool") or payload.get("last_tool") or "")
                if not waiting_on:
                    waiting_on = str(payload.get("waiting_on") or "")
                if not reason:
                    reason = str(payload.get("reason") or payload.get("summary") or "")
                heartbeat_at = payload.get("heartbeat_at")
            else:
                heartbeat_at = None
        except Exception:
            heartbeat_at = None
        if heartbeat_at is None:
            raw_heartbeat = state.get("heartbeat_at") or (goal.metadata.get("heartbeat_at") if goal is not None else None)
            try:
                heartbeat_at = float(raw_heartbeat) if raw_heartbeat is not None else None
            except (TypeError, ValueError):
                heartbeat_at = None
        with self._live_activity_lock:
            provider_activity = dict(self._live_provider_activity)
        provider_signal = provider_activity.get("last_signal_at")
        if provider_signal is not None:
            try:
                provider_signal = float(provider_signal)
            except (TypeError, ValueError):
                provider_signal = None
        if provider_signal and (heartbeat_at is None or provider_signal > heartbeat_at):
            heartbeat_at = provider_signal
        age = max(0.0, time.time() - heartbeat_at) if heartbeat_at else None
        provider_state = str(provider_activity.get("state") or "idle")
        # A live goal can remain RUNNING while the durable retry policy waits
        # for its next attempt.  Project that boundary explicitly instead of
        # leaving the header at a misleading generic Working/Planning state.
        retry_scheduled = bool(
            goal is not None
            and str(goal.metadata.get("auto_retryable") or "").casefold() in {"true", "1"}
            and int(goal.metadata.get("retry_after_ms", 0) or 0) > 0
        )
        provider_boundary = provider_state in {"failed", "network_unavailable"}
        # A stale transport marker must not override an active worker lease.
        # The worker heartbeat/lease is the durable evidence that the current
        # attempt is still progressing; the next provider event can replace
        # the telemetry marker without manufacturing a network outage.
        if lease_active and provider_state == "network_unavailable":
            provider_boundary = False
        local_runner = self.execution_class == "local"
        if (
            (retry_scheduled or provider_boundary)
            and goal is not None
            and goal.status in {GoalStatus.RUNNING, GoalStatus.DISCOVERING, GoalStatus.REVISING}
        ):
            phase = "retrying"
            waiting_on = (
                "model"
                if provider_state == "network_unavailable" and local_runner
                else "network"
                if provider_state == "network_unavailable"
                else "provider"
            )
            default_boundary_reason = (
                "Local model runner unavailable; the saved stage is unchanged."
                if local_runner and provider_state == "network_unavailable"
                else "Internet/provider unavailable; the saved stage is unchanged."
                if provider_state == "network_unavailable"
                else "The previous provider attempt failed; preparing a bounded retry."
            )
            stored_reason = str(
                goal.metadata.get("retry_reason")
                or goal.metadata.get("waiting_question")
                or ""
            ).strip()
            # Older sessions persisted the cloud-facing wording even when the
            # selected model was local.  Do not let that stale metadata split
            # the terminal/Web explanation after a restart.
            stale_network_wording = any(
                marker in stored_reason.casefold()
                for marker in (
                    "internet/provider unavailable",
                    "provider/network unavailable",
                    "internet or provider is unavailable",
                )
            )
            reason = (
                default_boundary_reason
                if local_runner and provider_state == "network_unavailable" and stale_network_wording
                else stored_reason or default_boundary_reason
            )
            resume_action = "Retry"
        # A reviewable change set is a durable user boundary even when the
        # underlying Ultra run and goal rows are still RUNNING.  Project it
        # from the same store as the Web review badge so the terminal, header,
        # and Execution view cannot disagree about whether work is active.
        review_pending = False
        if goal is not None:
            try:
                review_run = self.store.get_active_ultra_run(goal.id)
                if review_run is not None:
                    reviewable_states = {
                        ChangeSetStatus.OPEN,
                        ChangeSetStatus.CLOSED,
                        ChangeSetStatus.REVIEWING,
                        ChangeSetStatus.BLOCKED,
                    }
                    review_pending = any(
                        item.status in reviewable_states
                        for item in self.store.list_change_sets(review_run.id)
                    )
            except Exception:
                review_pending = False
        if review_pending and phase not in {"paused", "retrying", "waiting_for_approval"}:
            phase = "reviewing"
            waiting_on = "user"
            reason = "Recorded changes are ready for review."
            resume_action = "Review changes"
        # The pending approval marker is the durable authority.  The original
        # ``approval.requested`` event can fall outside the bounded event
        # window after a long run or process restart; projecting only that
        # event would turn a waiting action into stale ``Working``/``stalled``
        # status and hide the button the user needs.
        pending_tool = (
            goal.metadata.get("pending_tool_approval")
            if goal is not None
            else None
        )
        if isinstance(pending_tool, Mapping):
            pending_decision = str(pending_tool.get("decision") or "").casefold()
            if not pending_decision:
                pending_name = str(pending_tool.get("tool") or "the next action")
                phase = "waiting_for_approval"
                waiting_on = "user"
                last_tool = last_tool or pending_name
                reason = str(
                    goal.metadata.get("waiting_question")
                    or f"Approval is required before {pending_name} can run."
                )
                resume_action = "Retry"
        if phase == "completed":
            liveness = "completed"
        elif provider_state == "network_unavailable":
            liveness = "network_unavailable"
        elif phase in {
            "paused",
            "retrying",
            "waiting_for_approval",
            "awaiting_approval",
            "reviewing",
        } and waiting_on == "user":
            liveness = "waiting" if waiting_on == "user" else "paused"
        elif provider_state == "receiving":
            liveness = "receiving"
        elif provider_state in {"request_created", "request_sent", "connection_opened", "provider_connected"}:
            liveness = "request_sent"
        elif provider_state == "server_processing":
            liveness = "processing_response"
        elif provider_state == "completed" and phase in {"routing", "planning", "working", "reviewing"}:
            liveness = "processing_response"
        elif phase in {"routing", "planning", "starting", "working", "reviewing", "waiting_for_process"}:
            liveness = "client_active"
        else:
            liveness = "ready"
        stale_after = max(15.0, float(self.config.activity_heartbeat_seconds) * 4.0)
        if liveness in {"client_active", "request_sent", "receiving", "processing_response"} and age is not None and age > stale_after:
            liveness = "stalled"
        actor_label = (active_actor or str(provider_activity.get("actor") or "")).replace("-", " ").replace("_", " ").strip().title()
        active_operation = str(provider_activity.get("operation") or "")
        if not active_operation:
            active_operation = current_task or reason or (
                f"{actor_label} is active" if actor_label else phase.replace("_", " ").title()
            )
        timeline = tuple(
            item.to_dict()
            for item in self.events.list_live_events(
                after_sequence=max(0, self.events.latest_sequence - 24),
                limit=24,
            )[-8:]
        )
        stream_kind = str(provider_activity.get("stream_kind") or "none")
        stream_state = str(provider_activity.get("stream_state") or "idle")
        safe_stream_preview = str(provider_activity.get("safe_stream_preview") or "")[-4_000:]
        first_byte_at = provider_activity.get("first_byte_at")
        try:
            first_byte_at = float(first_byte_at) if first_byte_at is not None else None
        except (TypeError, ValueError):
            first_byte_at = None
        attempt_source = pending or state.get("last_semantic_turn") or {}
        if not isinstance(attempt_source, Mapping):
            attempt_source = {}
        attempt_id = str(attempt_source.get("attempt_id") or "")
        attempt_model = str(attempt_source.get("attempt_model") or model)
        attempt_state = str(attempt_source.get("attempt_state") or "")
        if not attempt_state:
            if str(attempt_source.get("status") or "").casefold() == "completed":
                attempt_state = "completed"
            elif provider_state in {"failed", "network_unavailable"}:
                attempt_state = "failed"
            elif pending is not None or provider_state not in {"", "idle"}:
                attempt_state = "running"
            else:
                attempt_state = "idle"
        retry_at = attempt_source.get("retry_at") or attempt_source.get("retry_not_before")
        try:
            retry_at = float(retry_at) if retry_at is not None else None
        except (TypeError, ValueError):
            retry_at = None
        failure_kind = str(attempt_source.get("failure_kind") or "")
        if not failure_kind:
            if provider_state == "network_unavailable":
                failure_kind = "transport"
            elif provider_state == "failed":
                failure_kind = "provider"
        return WorkflowRuntimeSnapshotV1(
            session_mode=("plan" if self.interaction_mode is InteractionModeV2.PLAN else "working")
            if (goal is not None or pending is not None)
            else "ready",
            route=route,
            execution_strategy=strategy,
            model=model,
            provider=provider,
            capability_band=capability_band,
            phase=phase,
            current_task=current_task,
            last_tool=last_tool,
            waiting_on=waiting_on,
            reason=reason,
            last_event_id=str(latest.id if latest is not None else ""),
            heartbeat_at=heartbeat_at,
            heartbeat_age=age,
            workspace_mutated=mutated,
            resume_action=resume_action,
            objective=objective,
            current_task_id=current_task_id,
            active_actor=active_actor,
            active_step=active_step,
            activity_sequence=self.events.latest_sequence,
            liveness=liveness,
            active_operation=active_operation,
            provider_request_state=provider_state,
            received_bytes=max(0, int(provider_activity.get("received_bytes") or 0)),
            received_chunks=max(0, int(provider_activity.get("received_chunks") or 0)),
            received_tokens=max(0, int(provider_activity.get("received_tokens") or 0)),
            last_signal_at=provider_signal or heartbeat_at,
            timeline_preview=timeline,
            stream_state=stream_state,
            stream_kind=stream_kind,
            safe_stream_preview=safe_stream_preview,
            first_byte_at=first_byte_at,
            task_items=tuple(task_items),
            next_task=next_task,
            session_revision=session_revision,
            content_revision=content_revision,
            attempt_id=attempt_id,
            attempt_state=attempt_state,
            attempt_model=attempt_model,
            retry_at=retry_at,
            failure_kind=failure_kind,
            local_adaptation_policy=local_policy,
        )

    # Short alias for API consumers that use the “runtime snapshot” wording.
    def runtime_snapshot(self) -> WorkflowRuntimeSnapshotV1:
        return self.workflow_runtime_snapshot()

    def _decorate_slice_result(self, result: SliceResult) -> SliceResult:
        """Attach durable runtime context without changing the old result API."""

        try:
            snapshot = self.workflow_runtime_snapshot()
        except Exception:
            return result
        return replace(
            result,
            phase=result.phase or snapshot.phase,
            reason=result.reason or snapshot.reason,
            waiting_on=result.waiting_on or snapshot.waiting_on,
            last_tool=result.last_tool or snapshot.last_tool,
            workspace_mutated=result.workspace_mutated or snapshot.workspace_mutated,
            resume_action=result.resume_action or snapshot.resume_action,
            heartbeat_at=result.heartbeat_at or snapshot.heartbeat_at,
        )

    def _update_execution_lease(self, *, stage: str, action_id: str = "", state: str = "active") -> None:
        """Persist a cooperative worker lease in session JSON (no migration)."""

        now = time.time()
        fallback_goal = self.active_goal()
        fallback_goal_id = fallback_goal.id if fallback_goal is not None else None
        lease_holder: dict[str, Any] = {}

        def reduce_session(current: dict[str, Any]) -> Mapping[str, Any]:
            current_state = dict(current.get("state") or {})
            lease = dict(current_state.get("execution_lease") or {})
            lease.update({
                "lease_kind": "workflow",
                "goal_id": current.get("goal_id") or fallback_goal_id,
                "worker_id": self._worker_id,
                "process_id": os.getpid(),
                "host": platform.node(),
                "stage": str(stage),
                "action_id": str(action_id),
                "heartbeat_at": now,
                "lease_state": str(state),
                "expires_at": now + max(30.0, float(self.config.activity_heartbeat_seconds) * 8.0),
            })
            current_state["execution_lease"] = lease
            current_state["heartbeat_at"] = now
            lease_holder.update(lease)
            return {
                "state": current_state,
                "goal_id": current.get("goal_id"),
                "session_mode": str(current.get("session_mode") or SessionMode.NORMAL.value),
                "plan_state": str(current.get("plan_state") or PlanState.NONE.value),
                "run_state": str(current.get("run_state") or RunState.IDLE.value),
                "ultra_profile": str(current.get("ultra_profile") or "standard"),
                "sleep_state": str(current.get("sleep_state") or "off"),
            }

        self.store.mutate_workflow_session(self.session_id, reduce_session)
        self.store.append_event(
            "workflow.heartbeat" if state == "active" else "workflow.state",
            goal_id=lease_holder.get("goal_id"),
            entity_type="worker",
            entity_id=self._worker_id,
            payload={"stage": stage, "action_id": action_id, "heartbeat_at": now, "lease_state": state},
        )

    def _live_foreign_execution_lease(self) -> Mapping[str, Any] | None:
        """Return a live mutation owner without changing its durable state."""

        session = self.store.get_workflow_session(self.session_id)
        existing = dict((session.get("state") or {}).get("execution_lease") or {})
        now = time.time()
        owner = str(existing.get("worker_id") or "")
        expires = float(existing.get("expires_at") or 0.0)
        stage = str(existing.get("stage") or "")
        recorded_pid = int(existing.get("process_id") or 0)
        recorded_host = str(existing.get("host") or "")
        same_host = not recorded_host or recorded_host == platform.node()
        process_alive = (
            self._process_is_alive(recorded_pid)
            if same_host and recorded_pid > 0
            else False
        )
        active_owner = bool(
            owner
            and owner != self._worker_id
            and str(existing.get("lease_state")) == "active"
            and (
                # On the same host, process liveness is stronger evidence than
                # a heartbeat deadline. Model/tool calls can legitimately run
                # longer than the lease TTL and must not admit a second worker.
                (recorded_pid > 0 and same_host and process_alive)
                or expires > now
            )
        )
        if not active_owner:
            return None
        # Provider heartbeats created before execution-lease separation do not
        # own mutations. Likewise, a recorded process that no longer exists
        # must not block restart until the wall-clock expiry.
        legacy_provider_heartbeat = (
            stage.startswith("provider:")
            and not str(existing.get("action_id") or "")
            and str(existing.get("lease_kind") or "") != "workflow"
        )
        known_dead_process = recorded_pid > 0 and same_host and not process_alive
        if legacy_provider_heartbeat or known_dead_process:
            return None
        return existing

    def _claim_execution_lease(self, goal: Goal) -> bool:
        if self._live_foreign_execution_lease() is not None:
            return False
        self._update_execution_lease(stage="starting", state="active")
        self.store.append_event("execution.started", goal_id=goal.id, payload={"worker_id": self._worker_id, "stage": "starting"})
        return True

    def _claim_workflow_lease(self, stage: str) -> bool:
        """Own routing/planning/model work before any transient state exists."""

        if self._live_foreign_execution_lease() is not None:
            return False
        self._update_execution_lease(stage=stage, state="active")
        return True

    @staticmethod
    def _process_is_alive(process_id: int) -> bool:
        """Read process liveness without signalling or mutating that process."""

        pid = int(process_id or 0)
        if pid <= 0:
            return False
        if pid == os.getpid():
            return True
        if os.name == "nt":
            try:
                import ctypes
                from ctypes import wintypes

                query_limited_information = 0x1000
                still_active = 259
                handle = ctypes.windll.kernel32.OpenProcess(
                    query_limited_information,
                    False,
                    pid,
                )
                if not handle:
                    return False
                try:
                    exit_code = wintypes.DWORD()
                    if not ctypes.windll.kernel32.GetExitCodeProcess(
                        handle, ctypes.byref(exit_code)
                    ):
                        return False
                    return int(exit_code.value) == still_active
                finally:
                    ctypes.windll.kernel32.CloseHandle(handle)
            except (AttributeError, OSError, ValueError):
                return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return False
        return True

    def _release_execution_lease(self, *, stage: str, state: str = "boundary") -> None:
        try:
            self._update_execution_lease(stage=stage, state=state)
        except Exception:
            return

    def _start_execution_heartbeat(self, stage: str) -> tuple[Event, Thread]:
        """Refresh a worker lease while one blocking model/tool loop is active."""

        stopped = Event()
        interval = max(
            1.0,
            min(10.0, float(self.config.activity_heartbeat_seconds)),
        )

        def heartbeat() -> None:
            while not stopped.wait(interval):
                try:
                    self._update_execution_lease(stage=stage, state="active")
                except Exception:
                    # The foreground boundary remains authoritative. A missed
                    # heartbeat is observable, but must not crash useful work.
                    continue

        thread = Thread(
            target=heartbeat,
            name=f"workflow-heartbeat-{self._worker_id[-8:]}",
            daemon=True,
        )
        thread.start()
        return stopped, thread

    @staticmethod
    def _stop_execution_heartbeat(stopped: Event, thread: Thread) -> None:
        stopped.set()
        thread.join(timeout=2.0)

    @property
    def interaction_mode(self) -> InteractionModeV2:
        session = self.store.get_workflow_session(self.session_id)
        state = dict(session.get("state") or {})
        explicit = state.get("interaction_mode")
        if explicit:
            return InteractionModeV2.parse(explicit)
        return (
            InteractionModeV2.PLAN
            if SessionMode.parse(str(session["session_mode"])) is SessionMode.PLAN
            else InteractionModeV2.WORKING
        )

    def set_reasoning_effort(self, effort: str) -> str:
        from .config import ReasoningEffort

        selected = ReasoningEffort.parse(effort).value
        goal = self.active_goal()
        if goal is not None and goal.status in {
            GoalStatus.RUNNING, GoalStatus.VERIFYING, GoalStatus.REVIEWING, GoalStatus.RECOVERING,
        }:
            raise RuntimeStateError("reasoning effort can change only at a safe checkpoint")
        setattr(self.provider, "reasoning_effort", selected)
        self._persist_runtime_snapshot()
        return selected

    @property
    def execution_class(self) -> str:
        if self.model_descriptor is not None:
            return self.model_descriptor.execution_class.value
        model = self.model_name.casefold()
        return "cloud" if self.provider_name in {"openai", "gemini"} or model.endswith((":cloud", "-cloud")) else "local"

    @property
    def access_level(self) -> str:
        if self.permission_adapter is not None:
            return self.permission_adapter.access_level.value
        return "normal"

    def replace_provider(
        self,
        provider: Any,
        descriptor: ModelDescriptor,
        *,
        _allow_compensated_local_continuation: bool = False,
    ) -> None:
        """Switch models only at a user-visible safe checkpoint."""

        goal = self.active_goal()
        if (
            self.ultra_session is not None
            and self.ultra_session.running
            and not self.ultra_session.safe_for_reconfiguration
        ):
            raise RuntimeStateError(
                "pause ULTRA and wait for active agents to reach a safe checkpoint before switching models"
            )
        if goal is not None and goal.status in {
            GoalStatus.RUNNING,
            GoalStatus.VERIFYING,
            GoalStatus.REVIEWING,
            GoalStatus.RECOVERING,
        }:
            raise RuntimeStateError("model switching is allowed only at a safe checkpoint")
        incoming_envelope = self._capability_envelope_for(provider, descriptor)
        previous_envelope = self.model_capability_envelope()
        if goal is not None and bool(goal.metadata.get("strategy_locked")):
            approved_raw = (
                goal.metadata.get("approved_model_capability_envelope")
                or goal.metadata.get("model_capability_envelope")
            )
            approved_envelope = (
                ModelCapabilityEnvelopeV1.from_mapping(approved_raw)
                if isinstance(approved_raw, Mapping)
                else previous_envelope
            )
            weaker_than_approved = (
                incoming_envelope.level < approved_envelope.level
                or (
                    approved_envelope.context_window_tokens is not None
                    and (
                        incoming_envelope.context_window_tokens is None
                        or incoming_envelope.context_window_tokens
                        < approved_envelope.context_window_tokens
                    )
                )
            )
            if weaker_than_approved and not _allow_compensated_local_continuation:
                raise RuntimeStateError(
                    "After approval, provider recovery requires a model with an equal or "
                    "stronger documented capability envelope."
                )
            if (
                weaker_than_approved
                and _allow_compensated_local_continuation
                and descriptor.execution_class is not ExecutionClass.LOCAL
            ):
                raise RuntimeStateError(
                    "Capability compensation is available only for an explicit local-model continuation."
                )
        self.provider = provider
        self.model_descriptor = descriptor
        # A model switch is a durable attempt boundary.  Update the pending
        # semantic turn in the same transaction path as all other workflow
        # state so a stale provider/capability envelope cannot survive into
        # the next retry or Web snapshot.
        try:
            session_state = self.store.get_workflow_session(self.session_id).get("state", {})
            pending_raw = session_state.get("pending_semantic_turn")
            if isinstance(pending_raw, Mapping) and str(
                pending_raw.get("status") or ""
            ).casefold() != "completed":
                pending = dict(pending_raw)
                pending.update(
                    {
                        "model_capability_envelope": incoming_envelope.to_dict(),
                        "capability_fingerprint": incoming_envelope.fingerprint,
                        "attempt_model": descriptor.model,
                        "retry_not_before": None,
                        "last_error": "",
                        "failure_kind": "",
                        "attempt_state": "running",
                    }
                )
                self._save_pending_semantic_turn(pending)
        except Exception:
            # Provider replacement must remain usable even for legacy sessions
            # whose persisted pending envelope is malformed; the new runtime
            # snapshot is still persisted below.
            pass
        if self.ultra_session is not None:
            self.ultra_session.switch_model(descriptor)
            phase = getattr(
                getattr(self.ultra_session, "orchestrator", None),
                "phase",
                None,
            )
            if getattr(phase, "value", "") in {
                "failed",
                "revision_required",
                "cancelled",
                "completed",
            }:
                self.ultra_session.close()
                self.ultra_session = None
        if goal is not None and not bool(goal.metadata.get("strategy_locked")):
            current = self.store.get_goal(goal.id)
            raw_demand = current.metadata.get("task_demand")
            if isinstance(raw_demand, Mapping):
                demand = TaskDemandV1.from_mapping(raw_demand)
                existing = ExecutionStrategyV1.parse(
                    current.metadata.get("execution_strategy")
                    or dict(current.metadata.get("execution_policy") or {}).get("mode")
                )
                decision = select_execution_strategy(
                    incoming_envelope,
                    demand,
                    minimum=existing,
                    allow_capability_escalation=(
                        existing is ExecutionStrategyV1.RECURSIVE
                    ),
                )
                policy = dict(current.metadata.get("execution_policy") or {})
                policy.update({
                    "mode": (
                        RunMode.ULTRA.value
                        if decision.strategy is ExecutionStrategyV1.RECURSIVE
                        else RunMode.NORMAL.value
                    ),
                    "strategy": decision.strategy.value,
                    "concurrency": decision.max_concurrency,
                })
                self.store.update_goal_metadata(
                    goal.id,
                    model_capability_envelope=incoming_envelope.to_dict(),
                    capability_fingerprint=incoming_envelope.fingerprint,
                    execution_strategy=decision.strategy.value,
                    strategy_decision=decision.to_dict(),
                    strategy_fingerprint=decision.fingerprint,
                    execution_policy=policy,
                )
                self.store.append_event(
                    "execution_strategy.reassessed",
                    goal_id=goal.id,
                    payload={
                        "previous_capability_fingerprint": previous_envelope.fingerprint,
                        "capability_fingerprint": incoming_envelope.fingerprint,
                        "strategy": decision.strategy.value,
                        "strategy_fingerprint": decision.fingerprint,
                    },
                )
        self._persist_runtime_snapshot()

    def continue_with_local_model(
        self,
        provider: Any,
        descriptor: ModelDescriptor,
    ) -> Mapping[str, Any]:
        """Continue at a local model with smaller packets and unchanged quality gates.

        This is the only post-approval path that may accept a weaker documented
        envelope. It never edits the accepted plan or lowers its quality target;
        instead it narrows the remaining execution packets to the incoming
        model's cohesive-component limit and keeps fresh evidence plus an
        independent final evaluation mandatory.
        """

        if descriptor.execution_class is not ExecutionClass.LOCAL:
            raise RuntimeStateError("continue with local model requires a local model")
        if not bool(getattr(descriptor, "supports_tools", False)):
            raise RuntimeStateError("the selected local model must support tools")

        goal = self.active_goal()
        incoming = self._capability_envelope_for(provider, descriptor)
        previous = self.model_capability_envelope()
        accepted_plan = self.store.get_accepted_plan(goal.id) if goal is not None else None
        plan = accepted_plan or (self.store.get_latest_plan(goal.id) if goal is not None else None)
        accepted_plan_fingerprint = accepted_plan.fingerprint if accepted_plan is not None else ""
        terminal = {
            TaskStatus.COMPLETED,
            TaskStatus.CANCELLED,
            TaskStatus.OBSOLETE,
        }
        remaining = tuple(task for task in (plan.tasks if plan is not None else ()) if task.status not in terminal)
        quality_target = dict(goal.metadata.get("quality_target") or {}) if goal is not None else {}
        quality_fingerprint = hashlib.sha256(
            json.dumps(quality_target, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        cohesive_limit = max(1, int(incoming.max_cohesive_components))
        abstraction_level = "atomic" if cohesive_limit == 1 else "narrow" if cohesive_limit <= 2 else "bounded"
        policy = {
            "version": 1,
            "mode": "local_capability_compensation",
            "model_id": descriptor.id,
            "model": descriptor.model,
            "provider": descriptor.provider,
            "previous_capability_fingerprint": previous.fingerprint,
            "local_capability_fingerprint": incoming.fingerprint,
            "abstraction": {
                "level": abstraction_level,
                "scope": "remaining_tasks_only",
                "max_cohesive_components_per_packet": cohesive_limit,
                "one_primary_decision_per_turn": True,
                "split_before_mutation_when_over_limit": True,
                "narrow_context": True,
            },
            "quality_floor": {
                "accepted_plan_fingerprint": accepted_plan_fingerprint,
                "current_plan_fingerprint": plan.fingerprint if plan is not None else "",
                "quality_target_fingerprint": quality_fingerprint,
                "quality_target_unchanged": True,
                "mandatory_executable_evidence": True,
                "fresh_evaluation_after_mutation": True,
                "independent_final_evaluation": True,
                "completion_gates_unchanged": True,
            },
            "remaining_task_packets": [
                {
                    "task_id": task.id,
                    "objective": task.title,
                    "status": task.status.value,
                    "acceptance_criteria": list(task.acceptance_criteria),
                    "verification": list(task.verification),
                    "max_cohesive_components": cohesive_limit,
                }
                for task in remaining
            ],
        }

        self.replace_provider(
            provider,
            descriptor,
            _allow_compensated_local_continuation=True,
        )
        if goal is not None:
            current = self.store.get_goal(goal.id)
            execution_policy = dict(current.metadata.get("execution_policy") or {})
            execution_policy.update(
                {
                    "continuation": "local_capability_compensation",
                    "abstraction_level": abstraction_level,
                    "max_cohesive_components_per_packet": cohesive_limit,
                    "decomposition": "bounded_remaining_task_packets",
                    "quality_gates": "unchanged",
                }
            )
            self.store.update_goal_metadata(
                goal.id,
                model_capability_envelope=incoming.to_dict(),
                capability_fingerprint=incoming.fingerprint,
                execution_policy=execution_policy,
                local_continuation_policy=policy,
                # A provider-specific backoff belongs to the failed cloud
                # endpoint.  Once the user or Full Auto has selected a local
                # provider, carrying that old retry window into ``resume``
                # would strand an otherwise valid continuation for minutes.
                retry_after_ms=0,
                retry_not_before=None,
                auto_retryable=False,
            )
            self.store.append_event(
                "model.local_continuation_configured",
                goal_id=goal.id,
                entity_type="model",
                entity_id=descriptor.id,
                payload={
                    "model": descriptor.model,
                    "provider": descriptor.provider,
                    "remaining_tasks": len(remaining),
                    "abstraction_level": abstraction_level,
                    "max_cohesive_components_per_packet": cohesive_limit,
                    "accepted_plan_fingerprint": accepted_plan_fingerprint,
                    "quality_target_fingerprint": quality_fingerprint,
                    "quality_gates_unchanged": True,
                },
            )
            self._work_conversation.append(
                {
                    "role": "user",
                    "content": (
                        "Local-model continuation is active. Preserve the accepted plan, scope, "
                        "acceptance criteria, verification, and quality target exactly. Work only "
                        f"on one {abstraction_level} packet at a time with at most {cohesive_limit} "
                        "cohesive components; split a remaining task before mutation when it exceeds "
                        "that limit. Completion still requires fresh executable evidence and an "
                        "independent final evaluation."
                    ),
                }
            )
            self._persist_runtime_snapshot(local_continuation_policy=policy)
        else:
            session_state = self.store.get_workflow_session(self.session_id).get(
                "state", {}
            )
            pending_raw = session_state.get("pending_semantic_turn")
            if isinstance(pending_raw, Mapping) and str(
                pending_raw.get("status") or ""
            ).casefold() != "completed":
                pending = dict(pending_raw)
                pending.update(
                    {
                        "model_capability_envelope": incoming.to_dict(),
                        "capability_fingerprint": incoming.fingerprint,
                        "local_continuation_policy": policy,
                        "retry_not_before": None,
                        "last_error": "",
                        "failure_kind": "",
                        "attempt_state": "running",
                    }
                )
                self._save_pending_semantic_turn(pending)
                self.store.append_event(
                    "model.local_continuation_configured",
                    entity_type="semantic_turn",
                    entity_id=str(pending.get("turn_id") or descriptor.id),
                    payload={
                        "model": descriptor.model,
                        "provider": descriptor.provider,
                        "remaining_tasks": 0,
                        "abstraction_level": abstraction_level,
                        "max_cohesive_components_per_packet": cohesive_limit,
                        "quality_gates_unchanged": True,
                        "pre_goal": True,
                    },
                )
                self._persist_runtime_snapshot(local_continuation_policy=policy)
        return {
            "model_id": descriptor.id,
            "provider": descriptor.provider,
            "model": descriptor.model,
            "remaining_tasks": len(remaining),
            "abstraction_level": abstraction_level,
            "max_cohesive_components_per_packet": cohesive_limit,
            "quality_gates_unchanged": True,
            "accepted_plan_fingerprint": accepted_plan_fingerprint,
            "quality_target_fingerprint": quality_fingerprint,
        }

    def replace_permission_adapter(self, adapter: PermissionAdapter) -> None:
        if (
            self.ultra_session is not None
            and self.ultra_session.running
            and not self.ultra_session.safe_for_reconfiguration
        ):
            raise RuntimeStateError(
                "pause ULTRA and wait for active agents to reach a safe checkpoint before changing permissions"
            )
        self.permission_adapter = adapter
        if self.ultra_session is not None:
            self.ultra_session.switch_permissions(adapter)
        self._persist_runtime_snapshot()

    def replace_config(self, config: RuntimeConfig) -> None:
        """Apply validated slice limits at an interactive command checkpoint."""

        if not isinstance(config, RuntimeConfig):
            raise TypeError("config must be a RuntimeConfig")
        with self._lock:
            self.config = config
            # Preserve the in-memory action history so changing an unrelated
            # display/runtime setting cannot clear the no-progress guardrail.
            self._watchdog.repeat_limit = max(1, self.config.repeated_action_limit)
        self._persist_runtime_snapshot()

    def _require_ultra_setup(self) -> tuple[ModelDescriptor, PermissionAdapter]:
        if self.model_descriptor is None:
            provider = self.provider_name
            if provider not in {"openai", "gemini", "ollama"}:
                raise RuntimeStateError(
                    "ULTRA requires a selected tool-capable model descriptor; reopen /model"
                )
            model = self.model_name
            cloud = provider in {"openai", "gemini"} or model.casefold().endswith(
                (":cloud", "-cloud")
            )
            self.model_descriptor = ModelDescriptor(
                provider=provider,
                model=model,
                execution_class=ExecutionClass.CLOUD if cloud else ExecutionClass.LOCAL,
                host=getattr(self.provider, "host", None),
                capabilities=("tools",),
                source="runtime",
            )
        if (
            self.config.require_local_gpu
            and self.model_descriptor.execution_class is ExecutionClass.LOCAL
        ):
            probe = probe_local_gpu()
            if not probe.gpu_available:
                raise RuntimeStateError(
                    "Local ULTRA is configured as GPU-required, but no usable GPU was detected. "
                    f"Probe source={probe.source}; {probe.message or 'no GPU evidence'}. "
                    "Set AGENT_REQUIRE_LOCAL_GPU=0 only if CPU fallback is intentional."
                )
            self.model_descriptor = replace(
                self.model_descriptor,
                metadata={
                    **dict(self.model_descriptor.metadata),
                    "gpu_required": True,
                    "hardware_probe": probe.to_dict(),
                },
            )
        if self.permission_adapter is None:
            raise RuntimeStateError(
                "ULTRA permissions are not initialized; restart interactively or use /permissions"
            )
        return self.model_descriptor, self.permission_adapter

    def ultra_readiness_issue(self) -> str | None:
        """Ultra is a depth policy on the unified engine, not a separate runtime."""

        return None

    def _make_ultra_session(self) -> Any:
        descriptor, permission_adapter = self._require_ultra_setup()
        from .ultra import UltraConfig
        from .ultra_session import UltraSession

        concurrency_limit = self._workflow_concurrency_limit()
        return UltraSession(
            store=self.store,
            workspace=self.workspace,
            descriptor=descriptor,
            permission_adapter=permission_adapter,
            approval=self._approval_allowed,
            events=self.events,
            config=UltraConfig(
                min_top_modules=self.config.ultra_top_modules_min,
                max_top_modules=self.config.ultra_top_modules_max,
                max_depth=self.config.ultra_max_depth,
                max_nodes=self.config.ultra_max_nodes,
                max_fix_attempts=self.config.ultra_fix_attempts,
                cloud_concurrency=concurrency_limit,
                local_concurrency=concurrency_limit,
                max_concurrency=concurrency_limit,
                provider_retries=self.config.max_provider_retries,
                role_memory_ttl_hours=self.config.role_memory_ttl_hours,
                context_chars=self._provider_conversation_budget(),
                prompt_trace_chars=self.config.prompt_trace_chars,
            ),
            agent_steps=self.config.subagent_steps,
            reasoning_effort=self.reasoning_effort,
            version_control=self.version_control,
        )

    def start_ultra(
        self,
        objective: str,
        *,
        requested_effects: Sequence[str] = (),
    ) -> Any:
        """Start the Ultra foundation and checkpoint at questions/approval."""

        if self.active_goal() is not None:
            raise RuntimeStateError("finish or cancel the active goal before starting ULTRA")
        if not self._claim_workflow_lease("planning:ultra-foundation"):
            raise RuntimeStateError(
                "another live process owns this workflow; Ultra planning was not replayed"
            )
        # Lightweight/test providers without an explicit model and permission
        # descriptor use the shared repository-grounded planner with Ultra
        # depth policy. Fully configured CLI runs use the durable specialist
        # scheduler below.
        if self.model_descriptor is None or self.permission_adapter is None:
            return self.start_goal(
                redact_text(objective, 20_000),
                execution_mode=RunMode.ULTRA,
                entry_surface="ultra",
            )
        self.ultra_session = self._make_ultra_session()
        return self.ultra_session.start(
            redact_text(objective, 20_000),
            requested_effects=requested_effects,
        )

    def retry_ultra_foundation(self) -> Any:
        """Retry a failed pre-approval Ultra foundation without duplication."""

        goal = self.active_goal()
        if goal is None:
            raise RuntimeStateError("there is no unfinished Ultra foundation to retry")
        if goal.active_plan_revision is not None:
            raise RuntimeStateError("the Ultra foundation already has a durable plan")
        if goal.status not in {GoalStatus.DISCOVERING, GoalStatus.PAUSED}:
            raise RuntimeStateError(
                f"cannot retry the Ultra foundation while goal is {goal.status.value}"
            )
        if self.ultra_session is not None and self.ultra_session.running:
            raise RuntimeStateError("pause ULTRA before retrying its foundation")
        if goal.status is GoalStatus.PAUSED:
            self.store.transition_goal(
                goal.id,
                GoalStatus.DISCOVERING,
                reason="retrying unified Ultra planning",
            )
        if self.model_descriptor is None or self.permission_adapter is None:
            self.events.publish(
                "planning.retry",
                "Retrying the saved goal with Ultra reasoning depth.",
                goal_id=goal.id,
            )
            return self.generate_plan(
                "Retry the repository-grounded plan using Ultra reasoning depth."
            )
        self.ultra_session = self._make_ultra_session()
        self.events.publish(
            "ultra.foundation_retry",
            "Retrying the saved ULTRA foundation with a clean model request.",
            goal_id=goal.id,
        )
        return self.ultra_session.restart_foundation(goal.id, goal.objective)

    def prepare_ultra_from_existing_goal(self) -> Any:
        """Compatibility alias for a pre-approval, one-way depth increase."""

        return self.increase_execution_depth()

    def increase_execution_depth(self) -> Any:
        """Increase staged execution to recursive before the plan is approved."""

        goal = self.active_goal()
        if goal is None:
            lock = self.workflow_mode_lock()
            if lock.locked:
                raise RuntimeStateError(lock.reason)
            session = self.store.get_workflow_session(self.session_id)
            self.store.mutate_workflow_session(
                self.session_id,
                lambda current_state: {
                    "state": {
                        **dict(current_state.get("state") or {}),
                        "minimum_strategy": ExecutionStrategyV1.RECURSIVE.value,
                        "interaction_mode": InteractionModeV2.WORKING.value,
                    },
                    "session_mode": SessionMode.NORMAL.value,
                },
                expected_revision=int(session.get("revision") or 0),
            )
            return None
        if bool(goal.metadata.get("strategy_locked")) or goal.active_plan_revision is not None:
            raise RuntimeStateError(
                "Execution depth is locked after plan approval and cannot be changed."
            )
        if goal.status not in {
            GoalStatus.DISCOVERING,
            GoalStatus.AWAITING_PLAN_APPROVAL,
            GoalStatus.PAUSED,
            GoalStatus.REVISING,
        }:
            raise RuntimeStateError(
                f"execution depth can increase only before approval; goal is {goal.status.value}"
            )
        current = self.store.get_goal(goal.id)
        raw_demand = current.metadata.get("task_demand")
        demand = (
            TaskDemandV1.from_mapping(raw_demand)
            if isinstance(raw_demand, Mapping)
            else TaskDemandV1.from_legacy(
                component_count=max(1, len(self.latest_plan().tasks) if self.latest_plan() else 1),
                parallelism_required=False,
                reasons=("pre-approval depth increase",),
            )
        )
        capability_raw = current.metadata.get("model_capability_envelope")
        capability = (
            ModelCapabilityEnvelopeV1.from_mapping(capability_raw)
            if isinstance(capability_raw, Mapping)
            else self.model_capability_envelope()
        )
        decision = select_execution_strategy(
            capability,
            demand,
            minimum=ExecutionStrategyV1.RECURSIVE,
        )
        if str(current.metadata.get("execution_strategy")) == ExecutionStrategyV1.RECURSIVE.value:
            return self.latest_plan()
        policy = dict(current.metadata.get("execution_policy") or {})
        policy.update(
            {
                "mode": RunMode.ULTRA.value,
                "strategy": ExecutionStrategyV1.RECURSIVE.value,
                "decomposition": "deep_when_independent",
                "concurrency": decision.max_concurrency,
            }
        )
        current = self.store.update_goal_metadata(
            goal.id,
            execution_policy=policy,
            execution_strategy=ExecutionStrategyV1.RECURSIVE.value,
            strategy_decision=decision.to_dict(),
            strategy_fingerprint=decision.fingerprint,
            strategy_locked=False,
        )
        session = self.store.get_workflow_session(self.session_id)
        self.store.mutate_workflow_session(
            self.session_id,
            lambda current_state: {
                "state": {
                    **dict(current_state.get("state") or {}),
                    "interaction_mode": str(
                        current.metadata.get("interaction_mode")
                        or InteractionModeV2.WORKING.value
                    ),
                    "strategy_decision": decision.to_dict(),
                    "strategy_fingerprint": decision.fingerprint,
                    "execution_strategy": decision.strategy.value,
                },
                "goal_id": current_state.get("goal_id") or goal.id,
                "session_mode": SessionMode.ULTRA.value,
            },
            expected_revision=int(session.get("revision") or 0),
        )
        self.events.publish(
            "execution_strategy.increased",
            "Execution depth increased to recursive before approval.",
            goal_id=goal.id,
            strategy_fingerprint=decision.fingerprint,
        )
        if current.metadata.get("ultra_run_id"):
            return self.store.get_latest_plan(current.id)
        if self.model_descriptor is not None and self.permission_adapter is not None:
            if current.status is GoalStatus.AWAITING_PLAN_APPROVAL:
                latest = self.latest_plan()
                if latest is not None and latest.status is PlanStatus.PENDING_APPROVAL:
                    self.store.reject_plan(
                        current.id,
                        latest.revision,
                        "Superseded by pre-approval recursive execution depth",
                        rejected_by="capability-policy",
                    )
                self.store.transition_goal(
                    current.id,
                    GoalStatus.REVISING,
                    reason="building recursive plan revision",
                )
            self.ultra_session = self._make_ultra_session()
            return self.ultra_session.restart_foundation(current.id, current.objective)
        if current.status is GoalStatus.AWAITING_PLAN_APPROVAL:
            latest = self.latest_plan()
            if latest is not None and latest.status is PlanStatus.PENDING_APPROVAL:
                self.store.reject_plan(
                    current.id,
                    latest.revision,
                    "Superseded by pre-approval recursive execution depth",
                    rejected_by="capability-policy",
                )
            self.store.transition_goal(
                current.id,
                GoalStatus.REVISING,
                reason="building recursive plan revision",
            )
        return self.generate_plan(
            "Increase decomposition depth for the same accepted semantics. "
            "Create a fresh recursive plan revision without expanding scope."
        )

    def intake_questions(self) -> tuple[Mapping[str, Any], ...]:
        pending = self.store.get_pending_intake(self.session_id)
        if pending is None:
            return ()
        return tuple(
            dict(item)
            for item in pending.get("questions", ())
            if not str(item.get("answer") or "").strip()
        )

    def _intake_repository_facts(self, query: str) -> tuple[str, ...]:
        """Return a small provenance-bearing slice before asking the user."""

        facts: list[str] = []
        if self._global_memory_enabled:
            for lesson in self.global_lessons.search(query, limit=4):
                self._used_global_lesson_ids.add(lesson.id)
                facts.append(
                    "Cross-run learned lesson: "
                    f"{lesson.title} — {lesson.content} (confidence={lesson.confidence:.2f})"
                )
        try:
            context_slice = self.repository_index.context_slice(
                query,
                max_entries=8,
                budget_chars=6_000,
            )
        except (OSError, UnicodeError, ValueError, TypeError):
            return tuple(facts)
        for entry in context_slice.entries:
            facts.append(
                "Discovered repository context: "
                f"{entry.path} -> {entry.kind} {entry.name} "
                f"(confidence={entry.confidence:.2f}, provenance={entry.provenance}, "
                f"hash={entry.file_hash[:12]})"
            )
        if context_slice.omitted_entries:
            facts.append(
                f"Repository retrieval omitted {context_slice.omitted_entries} lower-ranked entries."
            )
        return tuple(facts)

    def _record_global_learning(
        self,
        goal: Goal,
        *,
        succeeded: bool,
        evidence_ref: str,
        blocker: str = "",
    ) -> None:
        if not self._global_memory_enabled:
            return
        for lesson_id in tuple(self._used_global_lesson_ids):
            self.global_lessons.record_outcome(lesson_id, succeeded=succeeded)
        if not succeeded:
            return
        semantic_fingerprint = str(
            goal.metadata.get("semantic_goal_fingerprint") or goal.id
        )
        repository_signature = hashlib.sha256(
            json.dumps(
                goal.metadata.get("discovered_verifier_plugins", ()),
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        mode = str(
            dict(goal.metadata.get("execution_policy", {})).get("mode") or "normal"
        )
        tags = (
            f"repository:{repository_signature[:16]}",
            f"task:{semantic_fingerprint[:16]}",
            f"mode:{mode}",
        )
        content = (
            "This repository/task signature succeeded with fresh criterion-bound "
            "verification and independent review. Reuse only the verified strategy "
            "mechanics; re-inspect before applying it to different resources."
        )
        self.global_lessons.put(
            LearnedLessonV1(
                title=f"Verified strategy {repository_signature[:8]}/{semantic_fingerprint[:8]}",
                content=content,
                applicability_tags=tags,
                evidence_refs=(evidence_ref,),
                scope="project",
                successes=1 if succeeded else 0,
                failures=0 if succeeded else 1,
            )
        )

    def _route_intake(
        self,
        intake: Mapping[str, Any],
        brief: Any,
        *,
        entry_surface: str = "goal",
    ) -> Any:
        session = self.store.get_workflow_session(self.session_id)
        session_state = dict(session.get("state") or {})
        pending = session_state.get("pending_semantic_turn")
        pending = dict(pending) if isinstance(pending, Mapping) else {}
        capability_raw = pending.get("model_capability_envelope")
        capability_envelope = (
            ModelCapabilityEnvelopeV1.from_mapping(capability_raw)
            if isinstance(capability_raw, Mapping)
            else self.model_capability_envelope()
        )
        demand_raw = pending.get("task_demand")
        demand = (
            TaskDemandV1.from_mapping(demand_raw)
            if isinstance(demand_raw, Mapping)
            else TaskDemandV1.from_legacy(
                component_count=max(1, int(dict(intake.get("complexity") or {}).get("component_count", 1))),
                parallelism_required=(
                    str(dict(intake.get("complexity") or {}).get("coordination", "")).casefold()
                    == "parallel"
                ),
                reasons=tuple(dict(intake.get("complexity") or {}).get("reasons", ())),
            )
        )
        strategy_raw = pending.get("strategy_decision")
        explicit_ultra = RunMode.parse(brief.requested_mode) is RunMode.ULTRA
        strategy = (
            StrategyDecisionV1.from_mapping(strategy_raw)
            if explicit_ultra and isinstance(strategy_raw, Mapping)
            else select_execution_strategy(
                capability_envelope,
                demand,
                allow_capability_escalation=explicit_ultra,
            )
        )
        interaction_mode = (
            InteractionModeV2.PLAN
            if RunMode.parse(brief.requested_mode) is RunMode.PLAN
            else InteractionModeV2.WORKING
        )
        # ``workflow_mode`` is the user's durable choice.  The capability
        # strategy may narrow packets, but it must never rewrite Normal into
        # Ultra as a side effect of a weak local model.  Recursive execution
        # is therefore reachable here only through an explicit Ultra request.
        requested_mode = RunMode.parse(brief.requested_mode)
        # Plan is an interaction surface, not a stronger execution policy.
        # Keep the durable execution mode Normal while recording the separate
        # plan interaction mode above; otherwise a planning-only UI choice
        # leaks into the goal policy and makes a later run look like a mode
        # escalation.
        routed = (
            RunMode.ULTRA
            if requested_mode is RunMode.ULTRA
            else RunMode.NORMAL
        )
        self.store.complete_intake_session(
            str(intake["id"]),
            brief=brief.to_dict(),
            routed_mode=routed.value,
            route_reason=brief.route_reason,
        )
        # This is the harness binding the first execution strategy, not a
        # user-visible mode switch.  It occurs before a Goal exists and keeps
        # the pending semantic turn intact.
        binding_session = self.store.get_workflow_session(self.session_id)
        self.store.mutate_workflow_session(
            self.session_id,
            lambda current_state: {
                "state": {
                    **dict(current_state.get("state") or {}),
                    "interaction_mode": interaction_mode.value,
                    "strategy_decision": strategy.to_dict(),
                    "execution_strategy": strategy.strategy.value,
                },
                "session_mode": routed.value,
            },
            expected_revision=int(binding_session.get("revision") or 0),
        )
        self.events.publish(
            "intake.routed",
            f"Capability policy selected {strategy.strategy.value} execution: {brief.route_reason}",
            intake_id=intake["id"],
            mode=routed.value,
            interaction_mode=interaction_mode.value,
            strategy=strategy.to_dict(),
            capability_envelope=capability_envelope.to_dict(),
            complexity=dict(intake.get("complexity", {})),
            execution_brief=brief.to_dict(),
        )
        # The execution brief is audit metadata only. The exact request remains
        # the semantic source of truth for every mode.
        route_decision = pending.get("route_decision") or pending.get("decision")
        route_effects = (
            dict(route_decision.get("requested_effects") or {})
            if isinstance(route_decision, Mapping)
            else {}
        )
        requested_effects = tuple(
            str(name)
            for name, enabled in route_effects.items()
            if bool(enabled)
        )
        result = (
            self.start_ultra(
                brief.original_input,
                requested_effects=requested_effects,
            )
            if routed is RunMode.ULTRA
            else self.start_goal(
                brief.original_input,
                planning_only=False,
                execution_mode=routed,
                entry_surface=entry_surface,
            )
        )
        goal = self.active_goal()
        if goal is not None:
            self.store.update_goal_metadata(
                goal.id,
                interaction_mode=interaction_mode.value,
                model_capability_envelope=capability_envelope.to_dict(),
                capability_fingerprint=capability_envelope.fingerprint,
                task_demand=demand.to_dict(),
                task_demand_fingerprint=demand.fingerprint,
                strategy_decision=strategy.to_dict(),
                strategy_fingerprint=strategy.fingerprint,
                execution_strategy=strategy.strategy.value,
                strategy_locked=False,
            )
            current_session = self.store.get_workflow_session(self.session_id)
            self.store.mutate_workflow_session(
                self.session_id,
                lambda current_state: {
                    "state": {
                        **dict(current_state.get("state") or {}),
                        "interaction_mode": interaction_mode.value,
                        "model_capability_envelope": capability_envelope.to_dict(),
                        "strategy_decision": strategy.to_dict(),
                        "execution_strategy": strategy.strategy.value,
                    },
                    "goal_id": current_state.get("goal_id") or goal.id,
                    "session_mode": routed.value,
                },
                expected_revision=int(current_session.get("revision") or 0),
            )
        return result

    def submit_intent(
        self,
        text: str,
        *,
        requested_mode: str | RunMode = RunMode.NORMAL,
        entry_surface: str = "goal",
        semantic_decision: SemanticTurnDecisionV2 | None = None,
        semantic_turn_id: str = "",
    ) -> Any:
        """Run every new objective through the shared, durable intake gate."""

        value = str(text)
        if not value.strip():
            return None
        if self.active_goal() is None:
            self._stop_event.clear()
        pending = self.store.get_pending_intake(self.session_id)
        if pending is not None:
            unanswered = [
                item for item in pending.get("questions", ())
                if not str(item.get("answer") or "").strip()
            ]
            if not unanswered:
                raise RuntimeStateError("intake is ready but has not been routed")
            return self.answer_intake_question(str(unanswered[0]["id"]), value)
        if self.active_goal() is not None:
            return self.add_guidance(value)
        if semantic_decision is None:
            semantic_turn, semantic_decision = self._semantic_preflight(
                value,
                forced_route=RouteKind.GOAL,
                requested_mode=requested_mode,
            )
            semantic_turn_id = str(semantic_turn["turn_id"])
        if semantic_decision.route is not RouteKind.GOAL or semantic_decision.goal_intake is None:
            raise RuntimeStateError("Goal dispatch requires an accepted model-authored goal intake")
        decision = self.intent_architect.validate(
            semantic_decision.goal_intake,
            original_input=value,
            requested_mode=requested_mode,
            repository_facts=self._intake_repository_facts(value),
        )
        intake = self.store.create_intake_session(
            self.session_id,
            original_input=value,
            brief=decision.brief.to_dict(),
            complexity=decision.complexity.to_dict(),
            requested_mode=decision.brief.requested_mode.value,
            routed_mode=decision.brief.routed_mode.value,
            route_reason=decision.brief.route_reason,
            status=decision.status.value,
            questions=(item.to_dict() for item in decision.questions),
        )
        self.store.save_prompt_completeness(
            str(intake["id"]),
            decision.completeness.to_dict(),
        )
        self.events.publish(
            "intake.analyzed",
            (
                f"Intent Architect needs {len(decision.questions)} decision(s)."
                if decision.questions
                else "Intent Architect prepared the model-aware execution brief."
            ),
            intake_id=intake["id"],
            mode=decision.brief.routed_mode.value,
            complexity=decision.complexity.to_dict(),
            questions=[item.to_dict() for item in decision.questions],
        )
        if decision.questions:
            question_session = self.store.get_workflow_session(self.session_id)
            self.store.mutate_workflow_session(
                self.session_id,
                lambda current_state: {
                    "state": {
                        **dict(current_state.get("state") or {}),
                        "intake_id": intake["id"],
                        "intake_status": IntakeStatus.AWAITING_ANSWERS.value,
                    },
                    "session_mode": decision.brief.routed_mode.value,
                    "plan_state": PlanState.INSPECTING.value,
                    "run_state": RunState.PLANNING.value,
                },
                expected_revision=int(question_session.get("revision") or 0),
            )
            result = SliceResult(
                "awaiting_answers",
                f"Intent Architect needs {len(decision.questions)} decision(s) before planning.",
                needs_user=True,
            )
            if semantic_turn_id:
                self._complete_semantic_turn(semantic_turn_id, result_status=result.status)
            return result
        result = self._route_intake(
            intake,
            decision.brief,
            entry_surface=entry_surface,
        )
        if semantic_turn_id:
            self._complete_semantic_turn(
                semantic_turn_id,
                result_status=str(getattr(getattr(result, "status", "planning"), "value", getattr(result, "status", "planning"))),
            )
        return result

    def route_input(self, text: str) -> tuple[RouteDecisionV1, Any]:
        """Route idle plain text using one durable model-authored preflight."""

        mode = self.store.get_workflow_session(self.session_id)["session_mode"]
        semantic_turn, semantic = self._semantic_preflight(text, requested_mode=mode)
        decision = RouteDecisionV1(
            semantic.route,
            semantic.interpretation,
            False,
            semantic,
        )
        self.store.append_event(
            "input.routed",
            payload={
                "route": decision.kind.value,
                "reason": decision.reason,
                "explicit": False,
                "semantic_version": 2,
                "contract_fingerprint": semantic.fingerprint,
            },
        )
        semantic_turn["status"] = "dispatching"
        self._save_pending_semantic_turn(semantic_turn)
        if semantic.route is RouteKind.CHAT and not semantic.needs_workspace_tools:
            compact_response, _created = self._artifactize_chat_text(semantic.direct_response)
            assistant = {"role": "assistant", "content": compact_response}
            self._chat_conversation.append(assistant)
            self.store.append_chat_message(
                self.session_id,
                assistant,
                event_key=f"semantic:{semantic_turn['turn_id']}:assistant",
                run_id=str(semantic_turn["turn_id"]),
            )
            result = SliceResult("chat", semantic.direct_response)
        elif semantic.route in {RouteKind.CHAT, RouteKind.ACTION}:
            result = self.chat(
                text,
                _route_checked=True,
                semantic_decision=semantic,
                semantic_turn_id=str(semantic_turn["turn_id"]),
            )
        else:
            result = self.submit_intent(
                text,
                requested_mode=mode,
                entry_surface="chat",
                semantic_decision=semantic,
                semantic_turn_id=str(semantic_turn["turn_id"]),
            )
        self._complete_semantic_turn(str(semantic_turn["turn_id"]), result_status=str(getattr(result, "status", "routed")))
        return decision, result

    def answer_intake_question(self, question_id: str, value: str) -> Any:
        pending = self.store.get_pending_intake(self.session_id)
        if pending is None:
            raise RuntimeStateError("there is no active intake question")
        raw_questions = {str(item["id"]): item for item in pending.get("questions", ())}
        if question_id not in raw_questions:
            raise RuntimeStateError(f"unknown intake question id: {question_id}")
        question: ClarificationQuestionV1 = normalize_question(raw_questions[question_id])
        answer, source = answer_from_value(question, redact_text(value, 2_000))
        updated = self.store.answer_intake_question(
            str(pending["id"]), question_id, answer, source=source
        )
        unanswered = [
            item for item in updated.get("questions", ())
            if not str(item.get("answer") or "").strip()
        ]
        self.events.publish(
            "intake.question_answered",
            f"Saved {question_id}; {len(unanswered)} decision(s) remain.",
            intake_id=pending["id"],
            question_id=question_id,
            answer_source=source,
        )
        if question_id == "execution_mode" and answer.casefold().startswith("edit request"):
            self.store.cancel_intake_session(
                str(pending["id"]),
                reason="user chose to edit the request before planning",
            )
            cancelled_session = self.store.get_workflow_session(self.session_id)
            self.store.mutate_workflow_session(
                self.session_id,
                lambda current_state: {
                    "state": {
                        **dict(current_state.get("state") or {}),
                        "intake_id": pending["id"],
                        "intake_status": IntakeStatus.CANCELLED.value,
                    },
                    "session_mode": str(updated["requested_mode"]),
                    "plan_state": PlanState.INSPECTING.value,
                    "run_state": RunState.IDLE.value,
                    "goal_id": None,
                },
                expected_revision=int(cancelled_session.get("revision") or 0),
            )
            self.events.publish(
                "intake.edit_requested",
                "Request editing is active; update the objective in the composer when ready.",
                intake_id=pending["id"],
            )
            return SliceResult(
                "intake_edit_requested",
                "Planning is paused. Edit the request in the composer and send it when ready.",
                needs_user=True,
            )
        if unanswered:
            return SliceResult(
                "awaiting_answers",
                f"Saved {question_id}; {len(unanswered)} decision(s) remain.",
                needs_user=True,
            )
        answers = {
            str(item["id"]): str(item.get("answer") or "")
            for item in updated.get("questions", ())
        }
        semantic_turn, semantic = self._semantic_preflight(
            str(updated["original_input"]),
            forced_route=RouteKind.GOAL,
            requested_mode=str(updated["requested_mode"]),
            answers=answers,
        )
        assert semantic.goal_intake is not None
        decision = self.intent_architect.validate(
            semantic.goal_intake,
            original_input=str(updated["original_input"]),
            requested_mode=str(updated["requested_mode"]),
            answers=answers,
            repository_facts=self._intake_repository_facts(str(updated["original_input"])),
        )
        self.store.save_prompt_completeness(
            str(updated["id"]),
            decision.completeness.to_dict(),
        )
        result = self._route_intake(updated, decision.brief)
        self._complete_semantic_turn(str(semantic_turn["turn_id"]), result_status="intake_routed")
        return result

    def active_ultra_run(self) -> Any | None:
        goal = self.active_goal() or self.store.get_latest_goal(self.session_id)
        run_id = str(goal.metadata.get("ultra_run_id", "")) if goal else ""
        if run_id:
            try:
                return self.store.get_ultra_run(run_id)
            except NotFoundError:
                pass
        active = self.store.get_active_ultra_run(goal.id if goal else None)
        if active is not None:
            return active
        runs = self.store.list_ultra_runs(goal.id if goal else None)
        return runs[-1] if runs else None

    def ultra_questions(self) -> tuple[Mapping[str, Any], ...]:
        if self.ultra_session is not None:
            return self.ultra_session.questions()
        goal = self.active_goal()
        return tuple(goal.metadata.get("plan_questions", ())) if goal else ()

    def _ensure_ultra_session(self, *, start_background: bool = True) -> Any:
        """Lazily rebuild the current Ultra engine from durable state."""

        if self.ultra_session is not None:
            return self.ultra_session
        run = self.active_ultra_run()
        if run is None:
            raise RuntimeStateError("there is no durable ULTRA run to restore")
        self.restore_ultra(run.id, start_background=start_background)
        assert self.ultra_session is not None
        return self.ultra_session

    def answer_ultra_question(self, question_id: str, value: str) -> Any:
        session = self._ensure_ultra_session()
        return session.answer(question_id, value)

    def add_ultra_guidance(self, text: str) -> Evidence:
        goal = self.active_goal()
        if goal is None or not goal.metadata.get("ultra_run_id"):
            raise RuntimeStateError("there is no active ULTRA run")
        safe = redact_text(text, 4_000)
        item = self.store.add_evidence(
            goal_id=goal.id,
            plan_revision=goal.active_plan_revision,
            kind="guidance",
            summary=safe,
            created_by="user",
        )
        self._ensure_ultra_session(start_background=False).add_guidance(safe)
        return item

    def approve_ultra(
        self,
        revision: int | None = None,
        *,
        approved_by: str = "user",
    ) -> Plan:
        accepted = self._ensure_ultra_session().approve(
            revision,
            approved_by=approved_by,
        )
        goal = self.active_goal()
        if goal is not None:
            self.store.update_goal_metadata(
                goal.id,
                waiting_question="",
                retry_reason="",
                waiting_on="",
                resume_action="",
                legacy_semantic_enrichment_required=False,
            )
        return accepted

    def wait_for_ultra(self) -> Any:
        return self._ensure_ultra_session().wait()

    @staticmethod
    def _plan_change_paths(plan: Plan | None) -> set[str]:
        if plan is None:
            return set()
        paths: set[str] = set()
        for item in plan.expected_changes:
            raw = str(item.get("path") or "").strip()
            if not raw or raw.startswith("<"):
                continue
            try:
                paths.add(normalize_contract_path(raw))
            except DomainError:
                # Persisted legacy plans may contain a malformed path. Keep it
                # visible to the scope comparison so it cannot be approved by
                # accidentally dropping the unsafe entry.
                paths.add(raw.replace("\\", "/"))
        return paths

    @classmethod
    def _repair_revision_is_in_scope(
        cls,
        approved: Plan,
        proposed: Plan,
        repair_tasks: Iterable[Mapping[str, Any]],
    ) -> bool:
        approved_paths = cls._plan_change_paths(approved)
        proposed_paths = cls._plan_change_paths(proposed)
        if not proposed_paths.issubset(approved_paths):
            return False
        def plan_text(plan: Plan, tasks: Iterable[Mapping[str, Any]]) -> str:
            return json.dumps(
                {
                    "summary": plan.summary,
                    "execution_strategy": plan.execution_strategy,
                    "expected_changes": list(plan.expected_changes),
                    "tasks": list(tasks),
                },
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            ).casefold()

        def sensitive_capabilities(value: str) -> set[str]:
            markers = {
                "install": ("install",),
                "dependency": ("dependenc",),
                "network": ("network",),
                "external_service": ("external service",),
                "credential": ("credential", "secret"),
                "permission": ("permission",),
                "deploy": ("deploy", "publish"),
                "payment": ("payment",),
            }
            return {
                category
                for category, aliases in markers.items()
                if any(alias in value for alias in aliases)
            }

        approved_tasks = [
            {
                "title": task.title,
                "description": task.description,
                "acceptance_criteria": list(task.acceptance_criteria),
                "verification": list(task.verification),
            }
            for task in approved.tasks
        ]
        proposed_text = plan_text(
            proposed,
            list(repair_tasks),
        )
        approved_text = plan_text(
            approved,
            approved_tasks,
        )
        return sensitive_capabilities(proposed_text).issubset(
            sensitive_capabilities(approved_text)
        )

    def _ultra_quality_feedback(self, result: Any) -> str:
        findings: list[str] = []
        for package in (
            *tuple(
                getattr(result, "node_results", None)
                or getattr(result, "results", ())
                or ()
            ),
            *(
                (getattr(result, "global_result"),)
                if getattr(result, "global_result", None) is not None
                else ()
            ),
        ):
            findings.extend(
                str(item).strip()
                for item in getattr(package, "findings", ())
                if str(item).strip()
            )
        run_id = str(getattr(getattr(result, "run", None), "id", "") or "")
        if run_id:
            for item in self.store.list_quality_findings(run_id):
                if item.status.value == "resolved":
                    continue
                owner = item.repair_node_id or "unassigned"
                findings.append(
                    f"[{item.severity.value}] {item.category.value} finding for "
                    f"{owner}: {item.remediation}"
                )
        compact = tuple(dict.fromkeys(findings))[:24]
        return (
            "AUTONOMOUS QUALITY REVISION. Preserve the approved product scope and final "
            "output paths. Change the specialist topology, narrow weak component contracts, "
            "or replace the failed integration strategy; do not repeat the same approach. "
            "Confirmed blockers:\n- "
            + "\n- ".join(compact or ("the previous candidate failed its durable quality gate",))
        )

    def converge_ultra(self) -> Any:
        """Keep quality-only Ultra revisions alive after the one user approval.

        A revision is auto-approved only when its declared write paths remain
        within the previously approved scope.  Scope expansion still stops at
        the ordinary approval boundary.
        """

        self._ensure_ultra_session()
        approved_scope = self._plan_change_paths(
            self.store.get_accepted_plan(self.active_goal().id)
            if self.active_goal()
            else None
        )
        while True:
            result = self.wait_for_ultra()
            goal = self.active_goal() or self.store.get_latest_goal(self.session_id)
            if goal is None or result is None:
                return result
            outcome = None
            try:
                outcome = self.store.get_goal_outcome_contract(goal.id)
            except NotFoundError:
                pass
            if goal.status is GoalStatus.COMPLETED or (
                outcome and outcome.get("state") == "accepted"
            ):
                return result
            phase = str(getattr(getattr(result, "run", None), "phase", "")).casefold()
            if "revision_required" not in phase:
                return result
            if outcome and not bool(dict(outcome.get("contract") or {}).get("auto_converge", True)):
                return result
            feedback = self._ultra_quality_feedback(result)
            self.events.publish(
                "ultra.strategy_revision",
                "Quality remained below target; rebuilding the weak specialist boundary.",
                findings=feedback,
            )
            accepted_plan = self.store.get_accepted_plan(goal.id)
            prior_in_scope_revisions = max(
                int(goal.metadata.get("in_scope_quality_revision_attempts", 0) or 0),
                max(0, int(getattr(accepted_plan, "revision", 1) or 1) - 1),
            )
            revision_attempt = prior_in_scope_revisions + 1
            allow_scope_expansion = revision_attempt >= 3
            self.store.update_goal_metadata(
                goal.id,
                in_scope_quality_revision_attempts=revision_attempt,
                evidence_bound_scope_expansion=allow_scope_expansion,
            )
            proposed_master = self.replan_ultra(
                feedback,
                allow_scope_expansion=allow_scope_expansion,
            )
            if isinstance(proposed_master, SliceResult):
                return proposed_master
            while proposed_master is None:
                questions = self.ultra_questions()
                if not questions:
                    return result
                question = questions[0]
                options = tuple(
                    item for item in question.get("options", ()) if isinstance(item, Mapping)
                )
                if not options:
                    return result
                recommended = options[0]
                answer = str(
                    recommended.get("value")
                    or recommended.get("label")
                    or recommended.get("description")
                    or ""
                ).strip()
                if not answer:
                    return result
                proposed_master = self.answer_ultra_question(
                    str(question.get("id")), answer
                )
            # The recursive engine returns its typed MasterPlanV1, while scope
            # and approval are intentionally bound to the persisted legacy
            # Plan projection created by bind_foundation().  Never pass the
            # engine contract to helpers that require expected_changes.
            proposed_plan = self.latest_plan()
            if proposed_plan is None or proposed_plan.status is not PlanStatus.PENDING_APPROVAL:
                raise RuntimeStateError(
                    "ULTRA revision produced no pending approval-bound Plan projection"
                )
            proposed_scope = self._plan_change_paths(proposed_plan)
            if approved_scope and not proposed_scope.issubset(approved_scope):
                self.events.publish(
                    "ultra.scope_expansion_blocked",
                    "Autonomous quality revision requested paths outside the approved scope.",
                    approved_scope=sorted(approved_scope),
                    proposed_scope=sorted(proposed_scope),
                )
                return result
            if not approved_scope:
                approved_scope = set(proposed_scope)
            self.approve_ultra(
                proposed_plan.revision,
                approved_by="risk-adaptive-policy",
            )

    def restore_ultra(
        self,
        run_id: str,
        *,
        start_background: bool = True,
    ) -> Any:
        self.ultra_session = self._make_ultra_session()
        return self.ultra_session.restore(
            run_id,
            start_background=start_background,
        )

    def replan_ultra(
        self,
        feedback: str,
        *,
        allow_scope_expansion: bool | None = None,
    ) -> Any:
        from .ultra_models import UltraRunStatus

        goal = self.active_goal()
        run = self.active_ultra_run()
        if goal is None or run is None or not goal.metadata.get("ultra_run_id"):
            raise RuntimeStateError("there is no active ULTRA master plan to revise")
        if allow_scope_expansion is None:
            allow_scope_expansion = bool(
                goal.metadata.get("evidence_bound_scope_expansion")
            )
        source_run_id = str(
            goal.metadata.get("accepted_foundation_source_run_id") or ""
        )
        if not source_run_id:
            if run.goal_spec is not None and run.architecture_spec is not None:
                source_run_id = run.id
            else:
                candidates = [
                    item
                    for item in self.store.list_ultra_runs(goal.id)
                    if item.goal_spec is not None
                    and item.architecture_spec is not None
                ]
                if candidates:
                    source_run_id = candidates[0].id
        if not source_run_id:
            raise RuntimeStateError(
                "the durable goal has no accepted Ultra semantic foundation"
            )
        self.store.update_goal_metadata(
            goal.id,
            accepted_foundation_source_run_id=source_run_id,
        )
        if self.ultra_session is None:
            # Replanning consumes the durable accepted foundation, not the
            # interrupted run's pending-plan boundary. Restoring first makes a
            # rejected or scope-contracted revision impossible to repair
            # because restore correctly requires a still-pending plan. A fresh
            # session can read the source GoalSpec/Architecture directly below.
            self.ultra_session = self._make_ultra_session()
        if self.ultra_session is not None and self.ultra_session.running:
            raise RuntimeStateError("pause ULTRA at a safe checkpoint before requesting a replan")
        safe_feedback = redact_text(feedback, 4_000)
        latest = self.store.get_latest_plan(goal.id)
        if latest and latest.status == PlanStatus.PENDING_APPROVAL:
            self.store.reject_plan(
                goal.id,
                latest.revision,
                safe_feedback,
                rejected_by="user",
            )
        else:
            current = self.store.get_goal(goal.id)
            if current.status == GoalStatus.PAUSED:
                self.store.transition_goal(
                    goal.id,
                    GoalStatus.REVISING,
                    reason="ULTRA master-plan revision requested",
                )
            elif current.status in {GoalStatus.RUNNING, GoalStatus.BLOCKED}:
                self.store.transition_goal(
                    goal.id,
                    GoalStatus.REVISING,
                    reason=(
                        "ULTRA master-plan revision requested after a blocked quality gate"
                        if current.status is GoalStatus.BLOCKED
                        else "ULTRA master-plan revision requested"
                    ),
                )
            elif current.status != GoalStatus.REVISING:
                raise RuntimeStateError(
                    f"cannot revise ULTRA while goal is {current.status.value}"
                )
        self.store.update_ultra_run(
            run.id,
            status=UltraRunStatus.BLOCKED,
            error=f"superseded by master-plan revision: {safe_feedback}",
        )
        self.ultra_session = self._make_ultra_session()
        try:
            return self.ultra_session.restart_plan_from_accepted_foundation(
                goal.id,
                source_run_id,
                safe_feedback,
                allow_scope_expansion=allow_scope_expansion,
            )
        except Exception as exc:
            from .ultra import AgentProtocolError

            if not isinstance(exc, AgentProtocolError):
                raise
            current = self.store.get_goal(goal.id)
            if current.status is GoalStatus.REVISING:
                self.store.transition_goal(
                    goal.id,
                    GoalStatus.PAUSED,
                    reason="ULTRA repair plan contract did not converge",
                )
            message = redact_text(str(exc), 2_000)
            checkpoint = WorkflowStageCheckpointV1(
                stage="ultra_repair_plan",
                substage="applicability",
                category=WorkflowBoundaryKind.CONTRACT_INCOMPATIBILITY.value,
                message=message,
                attempts=3,
                resumable=True,
            )
            self.store.update_goal_metadata(
                goal.id,
                boundary_kind=WorkflowBoundaryKind.CONTRACT_INCOMPATIBILITY.value,
                waiting_question=(
                    "The in-scope repair plan did not satisfy its executable contract "
                    "within the bounded repair budget. Retry the saved repair stage or "
                    "change to an equal-or-stronger model."
                ),
                waiting_on="provider_contract",
                resume_action="ultra_replan",
                resume_status=GoalStatus.REVISING.value,
                retry_reason=message,
                replan_feedback=safe_feedback,
                workflow_stage_checkpoint=checkpoint.to_dict(),
            )
            self.store.append_event(
                "planning.checkpoint",
                goal_id=goal.id,
                payload=checkpoint.to_dict(),
            )
            return SliceResult(
                "paused",
                "The saved Ultra repair plan needs another targeted contract attempt. "
                "No new workspace mutation or approval occurred.",
                needs_user=True,
                phase="planning",
                reason=message,
                waiting_on="provider_contract",
                workspace_mutated=False,
                resume_action="ultra_replan",
            )

    def close(self) -> None:
        """Checkpoint background ULTRA work before the SQLite connection closes."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        if self.ultra_session is not None:
            self.ultra_session.close()
        try:
            session = self.store.get_workflow_session(self.session_id)
            lease = dict((session.get("state") or {}).get("execution_lease") or {})
            if str(lease.get("worker_id") or "") == self._worker_id:
                self._release_execution_lease(stage="runtime-closed", state="released")
        except Exception:
            pass
        if self.local_web_server is not None:
            self.local_web_server.stop()
            self.local_web_server = None
        # A read-only companion process does not own the active worker's
        # previews or child processes. Closing it must be observational only.
        if not self._foreign_execution_owner_live:
            tools.shutdown_workspace_resources(self.workspace)
            for resource in self.store.list_managed_resources(self.session_id):
                if resource.get("status") in {"running", "ready"}:
                    metadata = dict(resource.get("metadata", {}))
                    metadata["closed_by_runtime"] = True
                    self.store.save_managed_resource(
                        str(resource["id"]), self.session_id,
                        kind=str(resource["kind"]), status="stopped", metadata=metadata,
                    )

    def active_goal(self) -> Goal | None:
        return self.store.load_active_goal(self.session_id)

    def set_external_tool_approval_resolver(
        self,
        resolver: Callable[[str, str], bool] | None,
    ) -> None:
        """Attach the owning UI's resolver for loopback web approvals."""

        self._external_tool_approval_resolver = resolver

    def _approval_session_groups(self) -> set[str]:
        """Return permission groups explicitly allowed for this session."""

        session = self.store.get_workflow_session(self.session_id)
        raw = dict(session.get("state") or {}).get("approval_session_groups") or []
        if not isinstance(raw, (list, tuple, set, frozenset)):
            return set()
        return {str(item).strip() for item in raw if str(item).strip()}

    def _remember_approval_session_group(self, group: str) -> None:
        normalized = str(group or "").strip()
        if not normalized:
            return
        groups = self._approval_session_groups()
        if normalized in groups:
            return
        groups.add(normalized)
        self._persist_runtime_snapshot(approval_session_groups=sorted(groups))

    def resolve_tool_approval(self, action_fingerprint: str, decision: str) -> bool:
        """Resolve one visible tool approval without weakening the policy.

        The exact durable fingerprint must match the currently pending action.
        A live terminal resolver is preferred; the durable marker keeps the
        decision recoverable if that in-memory attention request was recreated.
        """

        value = str(decision or "").strip().casefold()
        if value not in {
            "allow", "approve", "allow_once", "allow_session", "deny", "reject"
        }:
            return False
        goal = self.active_goal()
        pending = dict(goal.metadata.get("pending_tool_approval") or {}) if goal is not None else {}
        if goal is None or str(pending.get("action_fingerprint") or "") != str(action_fingerprint or ""):
            return False
        session_group = str(pending.get("policy_group") or "").strip()
        if value == "allow_session" and not session_group:
            arguments = pending.get("arguments")
            if isinstance(arguments, Mapping):
                session_group = classify_action(
                    str(pending.get("tool") or "action"),
                    dict(arguments),
                    workspace=str(self.workspace),
                    sandboxed=self.access_level == "full",
                ).group

        def persist_durable_decision() -> None:
            pending["decision"] = (
                "allow_session"
                if value == "allow_session"
                else "allow_once" if value in {"allow", "approve", "allow_once"} else "deny"
            )
            self.store.update_goal_metadata(goal.id, pending_tool_approval=pending)
            if value == "allow_session":
                self._remember_approval_session_group(session_group)

        resolver = self._external_tool_approval_resolver
        if resolver is None:
            persist_durable_decision()
            return True
        try:
            # Record the exact decision before waking the live UI owner. The
            # worker can resume as soon as that resolver returns, so writing
            # afterwards creates a short window where Web still renders the
            # already-accepted approval as actionable.
            persist_durable_decision()
            if bool(resolver(str(action_fingerprint or ""), str(decision or ""))):
                return True
            # The owning UI may have been recreated after restart and have no
            # in-memory request. The durable marker above is the recovery path.
            return True
        except Exception as exc:
            pending.pop("decision", None)
            self.store.update_goal_metadata(goal.id, pending_tool_approval=pending)
            self.store.append_event(
                "approval.resolution_failed",
                goal_id=(self.active_goal().id if self.active_goal() is not None else None),
                entity_type="tool",
                payload={"reason": str(exc)[:500]},
            )
            return False

    def _approval_allowed(self, name: str, args: dict[str, Any], risk: str) -> bool:
        """Normalize legacy booleans and explicit interactive decisions."""

        goal = self.active_goal()
        approval_task = ""
        approval_task_id = ""
        if goal is not None:
            approval_plan = self.store.get_latest_plan(goal.id)
            if approval_plan is not None:
                active = next(
                    (
                        task
                        for task in approval_plan.tasks
                        if task.status in {TaskStatus.IN_PROGRESS, TaskStatus.VERIFYING}
                    ),
                    None,
                )
                if active is not None:
                    approval_task_id = active.id
                    approval_task = f"{active.id} · {active.title}"
        policy = classify_action(
            name,
            args,
            workspace=str(self.workspace),
            sandboxed=self.access_level == "full",
        )
        if policy.group in self._approval_session_groups():
            self.store.append_event(
                "approval.session_reused",
                goal_id=goal.id if goal is not None else None,
                entity_type="tool",
                entity_id=name,
                payload={"tool": name, "risk": risk, "policy_group": policy.group},
            )
            self.events.publish(
                "approval.session_reused",
                f"Session permission reused for {str(name).replace('_', ' ')}.",
                tool=name,
                risk=risk,
                policy_group=policy.group,
                waiting_on="tool",
                phase="starting",
            )
            return True
        approval_fingerprint = hashlib.sha256(
            json.dumps({"tool": name, "args": redact_data(args), "risk": risk}, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        approval_event = self.store.append_event(
            "approval.requested",
            goal_id=goal.id if goal is not None else None,
            entity_type="tool",
            entity_id=name,
            payload={"tool": name, "risk": risk, "action_fingerprint": approval_fingerprint},
        )
        self.events.publish(
            "approval.requested",
            f"Waiting for approval: {name}",
            tool=name,
            risk=risk,
            action_fingerprint=approval_fingerprint,
            waiting_on="user",
            phase="waiting_for_approval",
            objective=(goal.objective if goal is not None else ""),
            current_task=approval_task,
            current_task_id=approval_task_id,
            active_actor="coordinator",
        )
        # Persist the exact boundary before entering the UI callback.  If the
        # terminal is refreshed, or the browser is opened in another tab, the
        # pending action can still be reconstructed without replaying the
        # command.  This metadata is cleared only after the matching approval
        # is received (or retained as a paused checkpoint when denied).
        if goal is not None:
            self.store.update_goal_metadata(
                goal.id,
                pending_tool_approval={
                    "tool": name,
                    "arguments": redact_data(dict(args)),
                    "risk": risk,
                    "action_id": approval_fingerprint,
                    "action_fingerprint": approval_fingerprint,
                    "requested_sequence": int(approval_event.sequence or 0),
                    "policy_group": policy.group,
                    "policy_reason": policy.reason,
                    "policy_scope": policy.scope,
                },
                waiting_on="approval",
                resume_action="Retry",
            )
        session = self.store.get_workflow_session(self.session_id)
        sleep_enabled = str(session.get("sleep_state") or "off") == "on"
        sleep_policy = str(dict(session.get("state") or {}).get("sleep_policy") or "safe").casefold()
        sleep_full = sleep_enabled and sleep_policy == "full"
        sleep_safe = (
            policy.requirement is ApprovalRequirement.AUTO
            or (
                policy.requirement is ApprovalRequirement.SESSION
                and policy.group in {"project_checks", "project_preview"}
            )
            or (str(name) == "stop_preview" and policy.group == "host_action")
        )
        decision: Any
        decision_from_user = False
        if sleep_enabled and (sleep_safe or sleep_full):
            decision = (
                "allow_session"
                if policy.requirement is ApprovalRequirement.SESSION and not sleep_full
                else "allow_once"
            )
            self.store.append_event(
                "sleep.full_auto_approval" if sleep_full else "sleep.auto_approval",
                goal_id=goal.id if goal is not None else None,
                entity_type="tool",
                entity_id=name,
                payload={
                    "tool": name,
                    "risk": risk,
                    "action_fingerprint": approval_fingerprint,
                    "policy_group": policy.group,
                    "reason": policy.reason,
                    "sleep_policy": sleep_policy,
                },
            )
            self.events.publish(
                "sleep.full_auto_approval" if sleep_full else "sleep.auto_approval",
                f"{'Full Auto' if sleep_full else 'Sleep'} auto-approved {str(name).replace('_', ' ')}.",
                tool=name,
                risk=risk,
                action_fingerprint=approval_fingerprint,
            )
        else:
            pending = goal.metadata.get("pending_tool_approval") if goal is not None else None
            pending = dict(pending) if isinstance(pending, Mapping) else {}
            pending_fingerprint = str(pending.get("action_fingerprint") or "")
            pending_decision = str(pending.get("decision") or "").casefold()
            if pending_fingerprint == approval_fingerprint and pending_decision in {
                "allow", "allow_once", "allow_session", "deny", "reject"
            }:
                # A browser or recovered terminal may have answered while no
                # in-memory callback was alive. Consume that one-shot decision
                # here; the normal approval.received/checkpoint events below
                # remain authoritative and are emitted exactly once.
                decision = pending_decision
                decision_from_user = True
            else:
                decision = self.approval(name, args, risk)
                decision_from_user = True
        resolved_value = ""
        if isinstance(decision, bool):
            allowed = decision
        else:
            resolved_value = str(getattr(decision, "value", decision)).strip().casefold()
            if resolved_value == "ui_error":
                raise RuntimeStateError(
                    "The approval interface closed without a decision; the action is still waiting."
                )
            allowed = resolved_value in {"allow_once", "allow_session", "allow", "yes"}
        if allowed and decision_from_user and resolved_value == "allow_session":
            self._remember_approval_session_group(policy.group)
        if allowed:
            # ``goal`` was read before the approval callback and therefore
            # may not contain the marker persisted immediately above. Reload
            # the durable row before clearing it; otherwise an approved
            # command can leave a stale approval banner that is shown again
            # as actionable (and may be replayed on the next checkpoint).
            latest_goal = self.active_goal()
            if latest_goal is not None:
                pending = latest_goal.metadata.get("pending_tool_approval")
                if (
                    isinstance(pending, Mapping)
                    and str(pending.get("action_fingerprint") or "")
                    == approval_fingerprint
                ):
                    self.store.update_goal_metadata(
                        latest_goal.id,
                        pending_tool_approval={},
                        waiting_question="",
                        waiting_on="tool",
                        resume_action="",
                    )
            self.store.append_event(
                "approval.received",
                goal_id=goal.id if goal is not None else None,
                entity_type="tool",
                entity_id=name,
                payload={"tool": name, "risk": risk, "action_fingerprint": approval_fingerprint, "decision": "allow"},
            )
            self.events.publish(
                "approval.received",
                "Approved — resuming the saved action.",
                tool=name,
                action_fingerprint=approval_fingerprint,
                waiting_on="tool",
                phase="starting",
                objective=(goal.objective if goal is not None else ""),
                current_task=approval_task,
                current_task_id=approval_task_id,
                active_actor="coordinator",
            )
        return allowed

    def set_sleep_mode(self, enabled: bool, *, policy: str = "safe") -> Mapping[str, Any]:
        """Persist unattended mode independently of the current UI process.

        ``safe`` keeps the existing reversible/project-check policy. ``full``
        is an explicit session opt-in that approves every tool boundary inside
        the already selected workspace and records every decision durably.
        """

        normalized_policy = str(policy or "safe").strip().casefold()
        if normalized_policy not in {"safe", "full"}:
            raise ValueError("Sleep policy must be safe or full.")

        session = self.store.get_workflow_session(self.session_id)
        self.store.mutate_workflow_session(
            self.session_id,
            lambda current_state: {
                "state": {
                    **dict(current_state.get("state") or {}),
                    "sleep_enabled": bool(enabled),
                    "sleep_policy": normalized_policy if enabled else "off",
                },
                "sleep_state": "on" if enabled else "off",
            },
            expected_revision=int(session.get("revision") or 0),
        )
        self.events.publish(
            "sleep.mode_changed",
            f"Sleep Mode {'enabled' if enabled else 'disabled'}.",
            enabled=bool(enabled),
            policy=normalized_policy if enabled else "off",
            sleep_state="on" if enabled else "off",
            source="runtime",
        )
        return {
            "enabled": bool(enabled),
            "sleep_state": "on" if enabled else "off",
            "policy": normalized_policy if enabled else "off",
            "safe_actions_only": normalized_policy != "full" or not enabled,
        }

    def sleep_mode_enabled(self) -> bool:
        session = self.store.get_workflow_session(self.session_id)
        return str(session.get("sleep_state") or "off") == "on"

    def sleep_mode_policy(self) -> str:
        session = self.store.get_workflow_session(self.session_id)
        if str(session.get("sleep_state") or "off") != "on":
            return "off"
        policy = str(dict(session.get("state") or {}).get("sleep_policy") or "safe").casefold()
        return policy if policy in {"safe", "full"} else "safe"

    def auto_resolve_pending_sleep_approval(self) -> bool:
        """Re-evaluate the currently visible approval after Sleep is enabled.

        Safe Sleep remains policy-bound; explicit Full Auto may resolve any
        current tool boundary. The exact durable fingerprint is always used so
        enabling either mode cannot approve a stale or different action.
        """

        if not self.sleep_mode_enabled():
            return False
        goal = self.active_goal()
        pending = dict(goal.metadata.get("pending_tool_approval") or {}) if goal is not None else {}
        fingerprint = str(pending.get("action_fingerprint") or "")
        tool = str(pending.get("tool") or "")
        arguments = pending.get("arguments")
        if not fingerprint or not tool or not isinstance(arguments, Mapping):
            return False
        policy = classify_action(
            tool,
            dict(arguments),
            workspace=str(self.workspace),
            sandboxed=self.access_level == "full",
        )
        safe = self.sleep_mode_policy() == "full" or (
            policy.requirement is ApprovalRequirement.AUTO
            or (
                policy.requirement is ApprovalRequirement.SESSION
                and policy.group in {"project_checks", "project_preview"}
            )
            or (tool == "stop_preview" and policy.group == "host_action")
        )
        if not safe:
            return False
        return bool(self.resolve_tool_approval(fingerprint, "allow_once"))

    @staticmethod
    def _full_auto_question_choice(
        question: Mapping[str, Any],
    ) -> tuple[str, str] | None:
        """Choose one bounded answer for an unattended interview checkpoint.

        Full Auto is an explicit user instruction to keep the saved workflow
        moving.  It must still make a reproducible choice rather than asking a
        hidden prompt or inventing prose: prefer one model-authored
        recommendation, then an explicit default, then the first canonical
        option.  Malformed/free-form-only questions remain a truthful boundary
        because there is no safe deterministic answer to record.
        """

        raw_options = question.get("options")
        if isinstance(raw_options, (list, tuple)):
            raw_recommended = tuple(
                item
                for item in raw_options
                if isinstance(item, Mapping)
                and (
                    bool(item.get("recommended"))
                    or str(item.get("label") or "")
                    .strip()
                    .casefold()
                    .endswith("(recommended)")
                )
            )
            if len(raw_recommended) > 1:
                return None
        try:
            normalized = normalize_question(question)
        except (TypeError, ValueError):
            return None
        options = tuple(normalized.options)
        recommended = tuple(item for item in options if item.recommended)
        if len(recommended) == 1:
            return recommended[0].value, "recommended"

        defaults = (
            question.get("default"),
            question.get("default_value"),
            question.get("defaultValue"),
        )
        for raw_default in defaults:
            candidate = str(raw_default or "").strip()
            if not candidate:
                continue
            for item in options:
                if candidate.casefold() in {
                    item.value.casefold(),
                    item.label.casefold(),
                }:
                    return item.value, "default"

        if options:
            return options[0].value, "first_option"
        return None

    def _auto_resolve_full_auto_question(self) -> bool:
        """Advance one deterministic intake/plan/ULTRA question if present."""

        source = ""
        goal_id = ""
        question: Mapping[str, Any] | None = None

        intake = self.intake_questions()
        if intake:
            source = "intake"
            question = intake[0]
        else:
            goal = self.active_goal()
            if goal is None:
                return False
            goal_id = str(goal.id)
            raw_questions = (
                self.ultra_questions()
                if goal.metadata.get("ultra_run_id")
                else self.plan_questions()
            )
            for item in raw_questions:
                if not isinstance(item, Mapping):
                    continue
                if str(item.get("answer") or "").strip():
                    continue
                source = "ultra" if goal.metadata.get("ultra_run_id") else "plan"
                question = item
                break

        if not source or not isinstance(question, Mapping):
            return False
        question_id = str(question.get("id") or "").strip()
        choice = self._full_auto_question_choice(question)
        if not question_id or choice is None:
            self.store.append_event(
                "sleep.full_auto_question_blocked",
                goal_id=goal_id or None,
                entity_type="question",
                entity_id=question_id or "unknown",
                payload={
                    "source": source,
                    "reason": "no bounded deterministic answer",
                },
            )
            return False

        answer, selection = choice
        if source == "intake":
            self.answer_intake_question(question_id, answer)
        elif source == "ultra":
            self.answer_ultra_question(question_id, answer)
        else:
            self.answer_plan_question(question_id, answer)
        self.store.append_event(
            "sleep.full_auto_question_answered",
            goal_id=goal_id or None,
            entity_type="question",
            entity_id=question_id,
            payload={
                "source": source,
                "selection": selection,
                "answer": redact_text(answer, 500),
            },
        )
        self.events.publish(
            "sleep.full_auto_question_answered",
            f"Full Auto selected the {selection} answer for {question_id}; continuing.",
            goal_id=goal_id or None,
            source=source,
            question_id=question_id,
            selection=selection,
        )
        return True

    def auto_resolve_full_auto_boundary(self) -> tuple[str, ...]:
        """Resolve unattended boundaries that are safe under explicit Full Auto.

        Tool approvals remain the primary boundary.  Full Auto also advances
        bounded intake/plan/ULTRA questions using a recorded deterministic
        choice, then accepts the exact critic-approved plan revision.  These
        unattended decisions are recorded separately from user approval so the
        history remains explicit; conflicting or malformed questions are kept
        as a visible recovery boundary.
        """

        if self.sleep_mode_policy() != "full":
            return ()
        resolved: list[str] = []
        if self._auto_resolve_full_auto_question():
            # Answer one question per pass.  The answer can call the provider
            # and create a new plan, so the controller must refresh its durable
            # snapshot before selecting the next boundary.
            resolved.append("question")
            return tuple(resolved)
        if self.auto_resolve_pending_sleep_approval():
            resolved.append("tool")
            # A browser can enable Full Auto after the original worker has
            # already exited at a durable approval checkpoint (for example
            # after a process restart).  ``resolve_tool_approval`` records the
            # decision, but deliberately does not transition the goal itself;
            # the next worker must consume that exact decision.  Resume only
            # while the marker is still present, so a live worker that was
            # woken by the external resolver is never started twice.
            current_goal = self.active_goal()
            pending = (
                dict(current_goal.metadata.get("pending_tool_approval") or {})
                if current_goal is not None
                else {}
            )
            pending_decision = str(pending.get("decision") or "").casefold()
            if (
                current_goal is not None
                and current_goal.status is GoalStatus.PAUSED
                and pending_decision in {"allow", "allow_once", "allow_session"}
            ):
                try:
                    self.resume()
                except (RuntimeErrorBase, StateStoreError, ValueError) as exc:
                    self.store.append_event(
                        "sleep.full_auto_tool_resume_blocked",
                        goal_id=current_goal.id,
                        entity_type="tool",
                        entity_id=str(pending.get("tool") or "action"),
                        payload={"reason": redact_text(str(exc), 800)},
                    )
                    self.events.publish(
                        "sleep.full_auto_tool_resume_blocked",
                        "Full Auto accepted the saved tool boundary, but execution remains paused for recovery.",
                        goal_id=current_goal.id,
                        reason=redact_text(str(exc), 800),
                    )
                    resolved.remove("tool")
        goal = self.active_goal()
        plan = self.latest_plan()
        if (
            goal is not None
            and goal.status is GoalStatus.AWAITING_PLAN_APPROVAL
            and plan is not None
            and plan.status is PlanStatus.PENDING_APPROVAL
            and not bool(goal.metadata.get("legacy_semantic_enrichment_required"))
        ):
            try:
                approved = self.approve_plan(
                    plan.revision,
                    approved_by="sleep-full-auto",
                )
            except (RuntimeErrorBase, StateStoreError, ValueError) as exc:
                self.store.append_event(
                    "sleep.full_auto_plan_blocked",
                    goal_id=goal.id,
                    entity_type="plan",
                    entity_id=plan.id,
                    payload={
                        "revision": plan.revision,
                        "reason": redact_text(str(exc), 800),
                    },
                )
                self.events.publish(
                    "sleep.full_auto_plan_blocked",
                    "Full Auto could not approve the saved plan; the plan remains waiting for review.",
                    goal_id=goal.id,
                    revision=plan.revision,
                    reason=redact_text(str(exc), 800),
                )
                return tuple(resolved)
            self.store.append_event(
                "sleep.full_auto_plan_approval",
                goal_id=goal.id,
                entity_type="plan",
                entity_id=approved.id,
                payload={
                    "revision": approved.revision,
                    "fingerprint": approved.fingerprint,
                    "approved_by": "sleep-full-auto",
                    "quality_gates_unchanged": True,
                },
            )
            self.events.publish(
                "sleep.full_auto_plan_approval",
                f"Full Auto approved critic-reviewed plan r{approved.revision}; execution can resume.",
                goal_id=goal.id,
                revision=approved.revision,
                approved_by="sleep-full-auto",
            )
            resolved.append("plan")
        return tuple(resolved)

    def _checkpoint_tool_approval_boundary(
        self,
        goal: Goal,
        *,
        tool: str,
        args: Mapping[str, Any],
        risk: str,
        action_id: str,
        reason: str,
    ) -> None:
        fingerprint = hashlib.sha256(
            json.dumps(
                {"tool": tool, "args": redact_data(dict(args)), "risk": risk},
                sort_keys=True,
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        self.store.update_goal_metadata(
            goal.id,
            pending_tool_approval={
                "tool": tool,
                "arguments": redact_data(dict(args)),
                "risk": risk,
                "action_id": action_id,
                "action_fingerprint": fingerprint,
            },
            waiting_question=(
                f"Approval is required before {tool} can run. No workspace mutation "
                "from this action occurred. Retry from the saved checkpoint to approve it."
            ),
            waiting_on="approval",
            resume_action="Retry",
            auto_retryable=False,
        )
        if goal.status is GoalStatus.RUNNING:
            self.store.transition_goal(
                goal.id,
                GoalStatus.PAUSED,
                reason=reason,
            )
        self.store.append_event(
            "execution.boundary",
            goal_id=goal.id,
            entity_type="action",
            entity_id=action_id,
            payload={
                "phase": "waiting_for_approval",
                "reason": reason,
                "waiting_on": "user",
                "resume_action": "Retry",
                "tool": tool,
                "risk": risk,
                "action_fingerprint": fingerprint,
                "workspace_mutated": False,
            },
        )

    def version_history(self, limit: int = 20) -> tuple[Any, ...]:
        return self.version_control.history(limit)

    def undo_versions(self, steps: int = 1) -> tuple[str, ...]:
        if self.active_goal() is not None:
            raise RuntimeStateError(
                "Undo is disabled while a goal is active; finish or cancel it first."
            )
        ultra_future = getattr(self.ultra_session, "future", None)
        if ultra_future is not None and not ultra_future.done():
            raise RuntimeStateError("Undo is disabled while Ultra is still working.")
        selected = list(self.version_control.undo_candidates(max(steps, 1)))[:steps]
        if len(selected) < steps:
            raise VersionControlError(
                f"Only {len(selected)} accepted checkpoint(s) can be undone."
            )
        approved = self._approval_allowed(
            "undo_accepted_checkpoints",
            {
                "steps": steps,
                "commits": [item.commit for item in selected],
                "subjects": [item.subject for item in selected],
            },
            "high",
        )
        if not approved:
            raise RuntimeStateError("Undo cancelled; the workspace was not changed.")
        reverted = self.version_control.undo(steps)
        self.store.append_event(
            "version_control.undo",
            payload={"steps": steps, "reverted_commits": list(reverted)},
        )
        return reverted

    def _checkpoint_accepted_goal(self, goal: Goal, *, source: str) -> str | None:
        commit = self.version_control.create_checkpoint(
            f"{goal.objective[:120]} ({source})",
            kind="accepted",
        )
        if commit:
            self.store.append_event(
                "version_control.checkpoint",
                goal_id=goal.id,
                payload={"commit": commit, "source": source},
            )
        return commit

    def latest_plan(self) -> Plan | None:
        goal = self.active_goal()
        return self.store.get_latest_plan(goal.id) if goal else None

    def mode_transition_issue(self, mode: str) -> str:
        """Return a user-facing policy reason without changing durable state."""
        target = SessionMode.parse(mode)
        session = self.store.get_workflow_session(self.session_id)
        current = SessionMode.parse(str(session["session_mode"]))
        if target is current:
            return ""
        goal = self.active_goal()
        if (
            target is SessionMode.ULTRA
            and goal is not None
            and not bool(goal.metadata.get("strategy_locked"))
            and goal.active_plan_revision is None
            and goal.status in {
                GoalStatus.DISCOVERING,
                GoalStatus.AWAITING_PLAN_APPROVAL,
                GoalStatus.PAUSED,
                GoalStatus.REVISING,
            }
        ):
            return ""
        lock = self.workflow_mode_lock()
        return lock.reason if lock.locked else ""

    def workflow_mode_lock(self) -> WorkflowModeLock:
        """Return the authoritative mode lock derived from durable workflow state."""

        session = self.store.get_workflow_session(self.session_id)
        state = dict(session.get("state", {}))
        pending = state.get("pending_semantic_turn")
        if isinstance(pending, Mapping) and str(pending.get("status")) != "completed":
            stage = str(pending.get("stage") or pending.get("status") or "routing")
            return WorkflowModeLock(
                True,
                "Mode is locked until this workflow completes or is cancelled. "
                f"Current stage: {stage.replace('_', ' ')}.",
                stage,
            )
        goal = self.active_goal()
        if goal is not None and goal.status not in {GoalStatus.COMPLETED, GoalStatus.CANCELLED}:
            return WorkflowModeLock(
                True,
                "Mode is locked until this workflow completes or is cancelled. "
                f"Current stage: {goal.status.value.replace('_', ' ')}.",
                goal.status.value,
            )
        return WorkflowModeLock(False)

    def transition_mode(self, mode: str) -> str:
        """Persist Working/Plan input; legacy Ultra is a one-way depth request."""
        target = SessionMode.parse(mode)
        session = self.store.get_workflow_session(self.session_id)
        previous = SessionMode.parse(str(session["session_mode"]))
        if target is previous:
            if target is SessionMode.NORMAL and self.active_goal() is None:
                state = dict(session.get("state", {}))
                pending = state.get("pending_semantic_turn")
                if not (
                    isinstance(pending, Mapping)
                    and str(pending.get("status")) != "completed"
                ) and state.pop("minimum_strategy", None) is not None:
                    self.store.mutate_workflow_session(
                        self.session_id,
                        lambda current_state: {
                            "state": {
                                key: value
                                for key, value in dict(
                                    current_state.get("state") or {}
                                ).items()
                                if key != "minimum_strategy"
                            }
                        },
                        expected_revision=int(session.get("revision") or 0),
                    )
            return previous.value
        if target is SessionMode.ULTRA:
            self.increase_execution_depth()
            return SessionMode.NORMAL.value if self.active_goal() is None else SessionMode.ULTRA.value
        state = dict(session.get("state", {}))
        goal = self.active_goal()
        issue = self.mode_transition_issue(target.value)
        if issue:
            raise RuntimeStateError(issue)
        state["interaction_mode"] = (
            InteractionModeV2.PLAN.value
            if target is SessionMode.PLAN
            else InteractionModeV2.WORKING.value
        )
        if target is SessionMode.NORMAL:
            state.pop("minimum_strategy", None)
        if goal is not None:
            policy = dict(goal.metadata.get("execution_policy") or {})
            policy.update(
                {
                    "mode": target.value,
                    "decomposition": (
                        "deep_when_independent"
                        if target is SessionMode.ULTRA
                        else "adaptive"
                    ),
                    "concurrency": (
                        max(
                            1,
                            int(
                                self.config.ultra_local_concurrency
                                if self.execution_class == "local"
                                else self.config.ultra_cloud_concurrency
                            ),
                        )
                        if target is SessionMode.ULTRA
                        else 1
                    ),
                }
            )
            self.store.update_goal_metadata(goal.id, execution_policy=policy)
            state.update({
                "run_id": goal.metadata.get("run_id", goal.id),
                "goal_contract_fingerprint": goal.metadata.get("goal_contract_fingerprint"),
                "mutation_sequence": goal.metadata.get("mutation_sequence", 0),
                "convergence_state": goal.metadata.get("convergence_state", "not_evaluated"),
            })
        self.store.mutate_workflow_session(
            self.session_id,
            lambda current_state: {
                "state": state,
                "goal_id": current_state.get("goal_id") or (goal.id if goal else None),
                "session_mode": target.value,
            },
            expected_revision=int(session.get("revision") or 0),
        )
        if goal is not None and target is not previous:
            self.store.append_event(
                "mode.transition", goal_id=goal.id,
                payload={
                    "from": previous.value, "to": target.value,
                    "run_id": goal.metadata.get("run_id", goal.id),
                    "reason": "execution policy changed; durable run context preserved",
                },
            )
        return target.value

    def _emit_usage(self, turn: AssistantTurn) -> None:
        if not turn.usage:
            return
        self._provider_input_tokens += turn.usage.input_tokens
        self._provider_output_tokens += turn.usage.output_tokens
        self.events.publish(
            "usage",
            input_tokens=turn.usage.input_tokens,
            cached_tokens=turn.usage.cached_tokens,
            output_tokens=turn.usage.output_tokens,
        )

    def _provider_conversation_budget(
        self,
        system: str = "",
        schemas: Sequence[Mapping[str, Any]] = (),
    ) -> int:
        """Reserve provider space for system, tools, durable state, and output."""

        configured = max(4_000, int(self.config.conversation_chars))
        try:
            context_tokens = int(getattr(self.provider, "context_size", None))
        except (TypeError, ValueError):
            return configured
        if self.model_descriptor is not None and self.execution_class == "local":
            try:
                stored_policy: Mapping[str, Any] | None = None
                active_goal = self.active_goal()
                if active_goal is not None and isinstance(
                    active_goal.metadata.get("local_adaptation_policy"), Mapping
                ):
                    stored_policy = active_goal.metadata.get(
                        "local_adaptation_policy"
                    )
                context_tokens = min(
                    context_tokens,
                    int(
                        dict(stored_policy or self.local_adaptation_policy()).get(
                            "context_budget_tokens", context_tokens
                        )
                    ),
                )
            except (TypeError, ValueError):
                pass
        total_chars = max(8_000, context_tokens * 3)
        fixed_chars = len(system) + len(
            json.dumps(list(schemas), ensure_ascii=False, default=str)
        )
        output_tokens = int(
            getattr(self.provider, "max_output_tokens", 2_048) or 2_048
        )
        reserved = fixed_chars + max(4_000, output_tokens * 3)
        return max(4_000, min(configured, total_chars - reserved))

    def _provider_call_policy(self, actor: str) -> ProviderCallPolicyV1:
        """Return stage budgets without treating local inference like a network call.

        Local Ollama generations can spend most of their deadline in queueing and
        GPU decoding, especially for the first structured response.  Cloud calls
        keep the historical tighter bounds; local calls get bounded, stage-aware
        headroom so a slow-but-live runner is not mislabeled unavailable.
        """
        name = str(actor or "").casefold()
        local = self.execution_class == "local"
        if name == "semantic-router":
            tokens, deadline = 1_536, 600.0 if local else 120.0
        elif name == "semantic-goal-intake":
            tokens, deadline = 2_048, 600.0 if local else 180.0
        elif name in {"semantic-goal", "semantic-interpretation"}:
            tokens, deadline = 6_144, 600.0 if local else 240.0
        elif name in {"planner", "plan-reviewer"}:
            tokens, deadline = (
                (8_192 if name == "planner" else 4_096),
                600.0 if local else 360.0,
            )
        else:
            # Execution is intentionally bounded at fifteen minutes for a
            # local atomic request.  Planning has its own six-minute budget
            # above; a larger catch-all deadline used to make a hung local
            # mutation look alive indefinitely and diverged from the runtime
            # acceptance contract.
            tokens, deadline = None, 900.0
        effort = str(getattr(self.provider, "reasoning_effort", "") or "") or None
        return ProviderCallPolicyV1(
            stage=name,
            max_output_tokens=tokens,
            reasoning_effort=effort,
            stage_deadline_seconds=deadline,
        )

    def _structured_repair_limit(self) -> int:
        """Return the number of *repair* turns allowed for a contract.

        A real local runner should get one focused correction and then a
        smaller packet/model on the next checkpoint.  Replaying a large
        schema four times is both expensive and a common cause of the
        ``architecture_judge`` loop observed in the live workflow.  Test and
        non-Ollama adapters retain the historical two repairs so compatibility
        providers are not needlessly made less tolerant.
        """

        provider = self.provider_name.casefold()
        if self.execution_class == "local" and provider in {
            "ollama",
            "lmstudio",
            "llamacpp",
            "local",
        }:
            return 1
        return 2

    @staticmethod
    def _is_transport_boundary(exc: BaseException) -> bool:
        diagnostic = getattr(exc, "diagnostic", None)
        kind = getattr(diagnostic, "kind", None)
        kind_value = str(getattr(kind, "value", kind) or "")
        if kind in {
            ProviderFailureKind.DNS_OR_SOCKET,
            ProviderFailureKind.CONNECTION_REFUSED,
            ProviderFailureKind.TIMEOUT,
        } or kind_value in {
            item.value for item in (
                ProviderFailureKind.DNS_OR_SOCKET,
                ProviderFailureKind.CONNECTION_REFUSED,
                ProviderFailureKind.TIMEOUT,
            )
        }:
            return True
        text = str(getattr(diagnostic, "provider_message", "") or exc).casefold()
        return any(marker in text for marker in ("offline", "network unavailable", "connection refused", "dns failure", "no route to host", "timed out"))

    def _call_provider(
        self,
        conversation: list[dict[str, Any]],
        schemas: Sequence[dict[str, Any]],
        system: str,
        *,
        actor: str,
        step: int,
        stream_text: bool = True,
        normalization_context: Mapping[str, Any] | None = None,
    ) -> AssistantTurn:
        self.events.publish("step", actor=actor, step=step)
        current_goal = self.active_goal()
        if current_goal is not None:
            contract_data = current_goal.metadata.get("goal_contract")
            if isinstance(contract_data, Mapping):
                contract = GoalContractV1.from_dict(contract_data)
                projection = contract.projection(actor=actor)
                contract_envelope = state_envelope(projection, "GOAL_CONTRACT_PROJECTION", max_chars=12_000)
                conversation = [dict(message) for message in conversation]
                if conversation and conversation[0].get("role") == "user":
                    conversation[0]["content"] = contract_envelope + "\n\n" + str(conversation[0].get("content", ""))
                else:
                    conversation.insert(0, {"role": "user", "content": contract_envelope})
                self.store.append_event(
                    "goal_contract.projected",
                    goal_id=current_goal.id,
                    payload={
                        "actor": actor,
                        "fingerprint": contract.fingerprint,
                        "policy_version": self.weak_model_policy.version,
                        "rules": self.weak_model_policy.applied_rules("provider_call"),
                    },
                )
        checkpoint = (
            state_envelope(
                {
                    "goal_id": current_goal.id,
                    "status": current_goal.status.value,
                    "semantic_goal": current_goal.metadata.get("semantic_goal", {}),
                    "active_plan_revision": current_goal.active_plan_revision,
                },
                "PROVIDER_CONTEXT_CHECKPOINT",
                max_chars=12_000,
            )
            if current_goal is not None
            else "No durable goal exists yet; preserve the latest exact user turn."
        )
        provider_budget = self._provider_conversation_budget(system, schemas)
        conversation = context.suspend_and_revive(
            conversation,
            checkpoint,
            getattr(self.provider, "summarize", None),
            max_chars=provider_budget,
            on_suspend=lambda count: self.events.publish(
                "checkpoint",
                f"Provider-aware context rotation suspended {count} transient messages.",
                continues=True,
            ),
        )
        ensure_capabilities = getattr(self.provider, "_ensure_capabilities", None)
        if callable(ensure_capabilities):
            ensure_capabilities()
        capability_profile = getattr(self.provider, "capability_profile", None)
        native_tools = bool(getattr(capability_profile, "tool_call_support", True))
        if current_goal is not None and capability_profile is not None:
            self.store.append_event(
                "provider.capability_selected", goal_id=current_goal.id,
                payload={
                    "actor": actor,
                    "provider": getattr(capability_profile, "provider", self.provider_name),
                    "model": getattr(capability_profile, "model_name", self.model_name),
                    "protocol": getattr(capability_profile, "api_protocol", "unknown"),
                    "endpoint": getattr(capability_profile, "endpoint", ""),
                    "tools": native_tools,
                    "structured_output": bool(getattr(capability_profile, "structured_output_support", False)),
                    "vision": bool(getattr(capability_profile, "vision_support", False)),
                    "health": getattr(capability_profile, "health_status", "unknown"),
                },
            )
        if not native_tools and schemas:
            names = [_tool_name(schema) for schema in schemas if _tool_name(schema)]
            compact_contracts = [
                {
                    "name": _tool_name(schema),
                    "parameters": dict(schema.get("function", {}).get("parameters", {})),
                }
                for schema in schemas
                if _tool_name(schema)
            ]
            system = (
                system
                + "\n\nNATIVE TOOLS ARE UNAVAILABLE. Make exactly one bounded action proposal as "
                + '{"name":"AVAILABLE_NAME","args":{...}} with no lifecycle IDs. '
                + f"Available names: {', '.join(names)}. The harness validates and executes it.\n"
                + "Use this exact compact action contract:\n"
                + redact_text(
                    json.dumps(
                        compact_contracts,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    40_000,
                )
            )
            if current_goal is not None:
                self.store.append_event(
                    "provider.request_adapter_selected", goal_id=current_goal.id,
                    payload={"actor": actor, "adapter": "constrained_json_action", "native_tools": False},
                )
        last_error: Exception | None = None
        allowed_tools = {
            _tool_name(schema) for schema in schemas if _tool_name(schema)
        }
        provider_conversation = [dict(message) for message in conversation]
        contract_repairs = 0
        contract_repair_limit = self._structured_repair_limit()
        policy = self._provider_call_policy(actor)
        transport_recovery_attempted = False
        transport_recovery_succeeded = False
        for attempt in range(self.config.max_provider_retries + 1):
            try:
                while True:
                    turn = self._provider_call_with_watchdog(
                        provider_conversation,
                        schemas,
                        system,
                        actor=actor,
                        step=step,
                        stream_text=stream_text,
                        policy=policy,
                    )
                    if not isinstance(turn, AssistantTurn):
                        raise TypeError(
                            f"provider returned {type(turn).__name__}, expected AssistantTurn"
                        )
                    receipts: list[dict[str, Any]] = []
                    for call in turn.tool_calls:
                        # V2 exposed one combined semantic return. During the
                        # V3 split, accept that already-persisted transport name
                        # only when it maps unambiguously to the sole advertised
                        # semantic stage. This is an alias conversion, not an
                        # expansion of the per-turn tool allowlist.
                        if call.name == "submit_semantic_turn":
                            if allowed_tools == {"submit_semantic_route"}:
                                call.name = "submit_semantic_route"
                            elif allowed_tools == {"submit_goal_intake"}:
                                nested = (
                                    call.args.get("goal_intake")
                                    if isinstance(call.args, Mapping)
                                    else None
                                )
                                if isinstance(nested, Mapping):
                                    call.args = dict(nested)
                                call.name = "submit_goal_intake"
                        raw_args = call.args if isinstance(call.args, dict) else {}
                        call.args, receipt = normalize_generated_tool_payload(
                            call.name, raw_args, context=normalization_context
                        )
                        if receipt.actions:
                            receipts.append(receipt.to_dict())
                    if not turn.tool_calls and schemas and turn.text:
                        # Some weak models advertise native tool calling but emit
                        # the requested call as JSON assistant text. Normalize only
                        # an allow-listed proposal; every other name is rejected
                        # below before dispatch.
                        candidate = extract_first_json_object(turn.text)
                        proposal = (
                            normalize_action_proposal(candidate)
                            if candidate is not None
                            else None
                        )
                        if proposal is None:
                            repaired_candidate, repair_actions = repair_structured_json_object(
                                turn.text
                            )
                            repaired_proposal = (
                                normalize_action_proposal(repaired_candidate)
                                if repaired_candidate is not None
                                else None
                            )
                            if repaired_proposal is not None:
                                candidate = repaired_candidate
                                proposal = repaired_proposal
                                if repair_actions and current_goal is not None:
                                    self.store.append_event(
                                        "provider.response_repaired",
                                        goal_id=current_goal.id,
                                        payload={
                                            "actor": actor,
                                            "actions": list(repair_actions),
                                        },
                                    )
                        if proposal is not None:
                            name, args = proposal
                            args, receipt = normalize_generated_tool_payload(
                                name, args, context=normalization_context
                            )
                            generated_id = (
                                f"harness-{actor.replace(':', '-')}-{step}-{attempt}"
                            )
                            turn.tool_calls.append(
                                ToolCall(id=generated_id, name=name, args=args)
                            )
                            if receipt.actions:
                                receipts.append(receipt.to_dict())
                            if current_goal is not None and name in allowed_tools:
                                self.store.append_event(
                                    "tool_action.proposal_normalized",
                                    goal_id=current_goal.id,
                                    payload={
                                        "actor": actor,
                                        "tool": name,
                                        "generated_id": generated_id,
                                        "advertised_native_tools": native_tools,
                                    },
                                )
                    invalid_names = tuple(
                        dict.fromkeys(
                            call.name
                            for call in turn.tool_calls
                            if call.name not in allowed_tools
                        )
                    )
                    if receipts:
                        for receipt in receipts:
                            self.store.append_event(
                                "tool_payload.normalized",
                                goal_id=(current_goal.id if current_goal is not None else None),
                                entity_type=(
                                    "goal" if current_goal is not None else "session"
                                ),
                                entity_id=(
                                    current_goal.id if current_goal is not None else self.session_id
                                ),
                                payload={"actor": actor, **receipt},
                            )
                    if not invalid_names:
                        break
                    contract_repairs += 1
                    if current_goal is not None:
                        self.store.append_event(
                            "tool_contract.rejected",
                            goal_id=current_goal.id,
                            payload={
                                "actor": actor,
                                "attempt": contract_repairs,
                                "received": list(invalid_names),
                                "allowed": sorted(allowed_tools),
                                "stage": actor,
                            },
                        )
                    if contract_repairs > contract_repair_limit:
                        # Never hand an unadvertised call to a downstream
                        # dispatcher. The caller receives an empty turn and can
                        # reach its own precise, durable stage boundary.
                        turn.tool_calls.clear()
                        turn.text = ""
                        break
                    provider_conversation.append(turn.to_message())
                    provider_conversation.append(
                        {
                            "role": "user",
                            "content": (
                                "TOOL CONTRACT ERROR: the previous action is not "
                                "available in this stage. Make exactly one call from: "
                                + (", ".join(sorted(allowed_tools)) or "none; answer with prose")
                                + ". Do not repeat the rejected action."
                            ),
                        }
                    )
                if current_goal is not None:
                    recorded_calls = redact_text(
                        json.dumps(
                            [
                                {
                                    "name": call.name,
                                    "args": redact_data(call.args),
                                }
                                for call in turn.tool_calls
                            ],
                            ensure_ascii=False,
                            sort_keys=True,
                            default=str,
                        ),
                        8_000,
                    )
                    self.store.append_event(
                        "provider.turn_recorded",
                        goal_id=current_goal.id,
                        payload={
                            "actor": actor,
                            "step": step,
                            "tool_names": [call.name for call in turn.tool_calls],
                            "tool_calls_redacted": recorded_calls,
                            "text_excerpt": redact_text(turn.text or "", 1_000),
                        },
                    )
                self._emit_usage(turn)
                return turn
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as exc:
                if isinstance(exc, HardStopError):
                    raise
                last_error = exc
                message = redact_text(exc, 500)
                provider_overloaded = _provider_is_temporarily_overloaded(exc)
                retry_record = self.retry_ledger.record(
                    RetryKind.PROVIDER_TRANSPORT,
                    stage=actor,
                    reason=message,
                    input_value={"step": step, "conversation_messages": len(conversation)},
                    output_value={"error_type": type(exc).__name__},
                    next_action=(
                        "retry_later"
                        if provider_overloaded
                        else
                        "retry_same_provider"
                        if not isinstance(exc, (AssertionError, TypeError, ValueError))
                        and attempt < self.config.max_provider_retries
                        else "stop"
                    ),
                )
                current_goal = self.active_goal()
                if current_goal is not None:
                    self.store.append_event(
                        "workflow.retry",
                        goal_id=current_goal.id,
                        payload={
                            "kind": retry_record.kind.value,
                            "stage": retry_record.stage,
                            "reason": retry_record.reason,
                            "attempt": retry_record.attempt,
                            "input_fingerprint": retry_record.input_fingerprint,
                            "output_fingerprint": retry_record.output_fingerprint,
                            "progress": False,
                            "next_action": retry_record.next_action,
                        },
                    )
                fatal_provider_state = any(
                    marker in message.casefold()
                    for marker in ("429", "quota", "authentication", "unauthorized", "forbidden")
                )
                transport_boundary = self._is_transport_boundary(exc)
                if transport_recovery_succeeded:
                    # The one automatic reconnect was consumed; a second
                    # failure is a manual retry boundary.
                    transport_recovery_succeeded = False
                if transport_boundary and not transport_recovery_attempted:
                    diagnose = getattr(self.provider, "diagnose_connectivity", None)
                    if callable(diagnose):
                        transport_recovery_attempted = True
                        self.events.publish(
                            "provider.activity",
                            "Provider connection lost · checking for recovery",
                            actor=actor, phase=actor, state="waiting",
                            provider_state="network_unavailable", waiting_on="network",
                            heartbeat_at=time.time(),
                        )
                        self.sleeper(5.0)
                        try:
                            diagnosis = diagnose()
                            reachable = bool(
                                getattr(diagnosis, "reachable", None)
                                if not isinstance(diagnosis, Mapping)
                                else diagnosis.get("reachable")
                            )
                        except Exception:
                            reachable = False
                        if reachable:
                            self.events.publish(
                                "provider.activity",
                                "Provider connection restored · resuming the same saved stage",
                                actor=actor, phase=actor, state="active",
                                provider_state="provider_connected", waiting_on="model",
                                heartbeat_at=time.time(),
                            )
                            transport_recovery_succeeded = True
                            continue
                if (
                    isinstance(exc, (AssertionError, TypeError, ValueError))
                    or fatal_provider_state
                    or provider_overloaded
                    or (transport_boundary and not transport_recovery_succeeded)
                    or attempt >= self.config.max_provider_retries
                ):
                    break
                delay = self.config.retry_base_ms * (2**attempt) / 1_000
                self.events.publish(
                    "warning",
                    f"{actor} provider call failed ({type(exc).__name__}); retry {attempt + 1}/{self.config.max_provider_retries}",
                    delay=delay,
                )
                if delay:
                    self.sleeper(delay)
        assert last_error is not None
        boundary_prefix = (
            "Local model runner unavailable (provider unavailable); saved stage unchanged: "
            if self.execution_class == "local"
            else "Internet/provider unavailable; saved stage unchanged: "
        )
        unavailable = ProviderUnavailableError(
            (
                boundary_prefix
                if self._is_transport_boundary(last_error)
                else "provider is temporarily overloaded; the saved request was not replayed: "
                if _provider_is_temporarily_overloaded(last_error)
                else "provider unavailable after retries: "
            )
            + f"{type(last_error).__name__}: {redact_text(last_error, 500)}"
        )
        retry_after = _provider_retry_after_seconds(last_error)
        if retry_after is None and _provider_is_temporarily_overloaded(last_error):
            retry_after = 30
        if retry_after is not None:
            setattr(unavailable, "retry_after_seconds", retry_after)
        raise unavailable from last_error

    def _provider_call_with_watchdog(
        self,
        conversation: Sequence[Mapping[str, Any]],
        schemas: Sequence[Mapping[str, Any]],
        system: str,
        *,
        actor: str,
        step: int = 0,
        stream_text: bool,
        policy: ProviderCallPolicyV1 | None = None,
    ) -> AssistantTurn:
        """Bound a provider call and report liveness without replaying tools."""

        if not self._claim_workflow_lease(f"provider:{actor}"):
            raise RuntimeStateError(
                "another live process owns this workflow; the provider call was not replayed"
            )
        results: Queue[tuple[str, Any]] = Queue(maxsize=1)
        abandoned = Event()
        self._active_provider_abandon = abandoned
        started_at = time.monotonic()
        signal_at = [started_at]
        actor_name = str(actor).casefold()
        provider_phase = (
            "routing"
            if actor_name == "semantic-router"
            else "planning"
            if actor_name in {"semantic-goal-intake", "planner", "plan-reviewer"}
            else "working"
        )
        structured_call = bool(schemas)
        request_operation = (
            f"Calling {self.model_name} for {actor}"
            + (" · structured response is atomic" if structured_call else "")
        )
        provider_goal = self.active_goal()
        self.events.publish(
            "workflow.state",
            f"Request created for {actor}",
            actor=actor,
            active_actor=actor,
            active_step=step,
            phase=provider_phase,
            waiting_on="model",
            reason="The provider request was created",
            objective=(provider_goal.objective if provider_goal is not None else ""),
            model=self.model_name,
            provider=self.provider_name,
        )

        with self._live_activity_lock:
            self._live_provider_activity = {
                "state": "client_active",
                "actor": actor,
                "operation": request_operation,
                "received_bytes": 0,
                "received_chunks": 0,
                "received_tokens": 0,
                "stream_kind": "chat" if not structured_call else "structured",
                "stream_state": "active",
                "safe_stream_preview": "",
                "first_byte_at": None,
                "started_at": time.time(),
                "last_signal_at": time.time(),
            }
        self.events.publish(
            "provider.activity",
            f"Request created for {actor}",
            source_kind="MODEL",
            actor=actor,
            phase=provider_phase,
            state="started",
            provider_state="request_created",
            operation=request_operation,
            waiting_on="provider",
            elapsed_seconds=0,
            received_bytes=0,
            received_chunks=0,
            received_tokens=0,
            heartbeat_at=time.time(),
        )
        self.events.publish(
            "provider.activity",
            f"Opening {self.model_name} request for {actor}",
            source_kind="MODEL",
            actor=actor,
            phase=provider_phase,
            state="started",
            provider_state="request_created",
            operation=f"Calling {self.model_name} for {actor}",
            waiting_on="model",
            elapsed_seconds=0,
            received_bytes=0,
            received_chunks=0,
            received_tokens=0,
            heartbeat_at=time.time(),
        )

        def provider_activity(activity: ProviderActivityV1) -> None:
            if abandoned.is_set():
                return
            signal_at[0] = time.monotonic()
            now_wall = time.time()
            with self._live_activity_lock:
                current = dict(self._live_provider_activity)
                first_chunk = (
                    activity.state == "receiving"
                    and int(current.get("received_chunks") or 0) == 0
                )
                if activity.state == "receiving":
                    current["received_bytes"] = int(current.get("received_bytes") or 0) + max(0, int(activity.received_bytes or 0))
                    current["received_chunks"] = int(current.get("received_chunks") or 0) + max(0, int(activity.received_chunks or 0))
                    current["received_tokens"] = int(current.get("received_tokens") or 0) + max(0, int(activity.received_tokens or 0))
                elif activity.received_tokens:
                    current["received_tokens"] = max(
                        int(current.get("received_tokens") or 0),
                        int(activity.received_tokens),
                    )
                current["state"] = str(activity.state or current.get("state") or "client_active")
                current["last_signal_at"] = now_wall
                if activity.state == "receiving" and current.get("first_byte_at") is None:
                    current["first_byte_at"] = now_wall
                current["stream_state"] = (
                    "receiving" if activity.state == "receiving"
                    else "validating" if activity.state == "completed"
                    else "waiting" if activity.state in {"failed", "network_unavailable"}
                    else "active"
                )
                current["operation"] = (
                    f"Receiving {self.model_name} response for {actor}"
                    if current["state"] == "receiving"
                    else f"Processing {self.model_name} response for {actor}"
                    if current["state"] == "completed"
                    else request_operation
                )
                self._live_provider_activity = current
            state_label = str(activity.state or "active").replace("_", " ")
            message = (
                "First response bytes received"
                if first_chunk
                else "Model response bytes received"
                if activity.state == "receiving"
                else "Model response received; validating it"
                if activity.state == "completed"
                else f"Provider {state_label}"
            )
            self.events.publish(
                "provider.activity",
                message,
                source_kind="MODEL",
                actor=actor,
                phase=provider_phase,
                state=("receiving" if activity.state == "receiving" else "completed" if activity.state == "completed" else "active"),
                provider_state=activity.state,
                operation=current["operation"],
                detail=activity.detail,
                waiting_on="model" if activity.state != "completed" else "harness",
                elapsed_seconds=max(0, int(time.monotonic() - started_at)),
                received_bytes=current["received_bytes"],
                received_chunks=current["received_chunks"],
                received_tokens=current["received_tokens"],
                heartbeat_at=now_wall,
            )

        def signal(kind: str, fragment: Any) -> None:
            signal_at[0] = time.monotonic()
            if abandoned.is_set():
                return
            if kind == "model_text" and not structured_call:
                with self._live_activity_lock:
                    current = dict(self._live_provider_activity)
                    current["safe_stream_preview"] = (
                        str(current.get("safe_stream_preview") or "") + str(fragment)
                    )[-4_000:]
                    current["stream_kind"] = "chat"
                    current["stream_state"] = "receiving"
                    current.setdefault("first_byte_at", time.time())
                    self._live_provider_activity = current
            if not self._provider_accepts_activity and kind in {"model_text", "model_thought"}:
                provider_activity(ProviderActivityV1(
                    state="receiving",
                    received_bytes=len(str(fragment).encode("utf-8", errors="replace")),
                    received_chunks=1,
                ))
            self.events.publish(kind, str(fragment), actor=actor)

        def invoke() -> None:
            try:
                call_kwargs: dict[str, Any] = {
                    "on_text": (
                        (lambda fragment: signal("model_text", fragment))
                        if stream_text else None
                    ),
                    "on_thought": lambda fragment: signal("model_thought", fragment),
                }
                if self._provider_accepts_activity:
                    call_kwargs["on_activity"] = provider_activity
                if self._provider_accepts_policy and policy is not None:
                    call_kwargs["policy"] = policy
                value = self.provider.call(
                    list(conversation),
                    list(schemas),
                    system,
                    **call_kwargs,
                )
            except BaseException as exc:
                diagnostic = getattr(exc, "diagnostic", None)
                transport_state = (
                    "network_unavailable"
                    if self._is_transport_boundary(exc)
                    else "failed"
                )
                with self._live_activity_lock:
                    self._live_provider_activity["state"] = transport_state
                    self._live_provider_activity["stream_state"] = "waiting"
                    self._live_provider_activity["last_signal_at"] = time.time()
                if transport_state == "network_unavailable":
                    local_runner = self.execution_class == "local"
                    self.events.publish(
                        "provider.activity",
                        (
                            "Local model runner unavailable · saved stage unchanged"
                            if local_runner
                            else "Internet/provider unavailable · saved stage unchanged"
                        ),
                        source_kind="MODEL", actor=actor, phase=provider_phase,
                        state="waiting", provider_state=transport_state,
                        operation=(
                            "Waiting for local model runner"
                            if local_runner
                            else f"Waiting for {self.provider_name} connectivity"
                        ),
                        waiting_on="model" if local_runner else "network",
                        detail=str(getattr(diagnostic, "provider_message", "") or exc),
                        heartbeat_at=time.time(), elapsed_seconds=max(0, int(time.monotonic() - started_at)),
                    )
                self.events.publish(
                    "provider.activity",
                    f"Provider request failed: {type(exc).__name__}",
                    source_kind="MODEL",
                    actor=actor,
                    phase=provider_phase,
                    state=("waiting" if transport_state == "network_unavailable" else "failed"),
                    provider_state=transport_state,
                    operation=f"{self.model_name} request failed",
                    waiting_on="retry",
                    elapsed_seconds=max(0, int(time.monotonic() - started_at)),
                    heartbeat_at=time.time(),
                )
                results.put(("error", exc))
            else:
                signal_at[0] = time.monotonic()
                if not self._provider_accepts_activity:
                    provider_activity(ProviderActivityV1(state="completed"))
                results.put(("result", value))

        worker = Thread(
            target=invoke,
            name=f"ga3bad-provider-{actor[:24]}",
            daemon=True,
        )
        worker.start()
        heartbeat = max(1, int(self.config.activity_heartbeat_seconds))
        next_heartbeat = started_at + heartbeat
        cloud = self.execution_class == "cloud"
        slow_warning_sent = False
        local_poll_at = started_at + 5.0
        connectivity_probe_at = started_at + 5.0
        diagnostic_in_flight = False
        diagnostic_results: Queue[dict[str, Any]] = Queue()
        while True:
            try:
                kind, value = results.get(timeout=min(1.0, float(heartbeat)))
            except Empty:
                now = time.monotonic()
                total = now - started_at
                quiet = now - signal_at[0]
                if self._stop_event.is_set():
                    abandoned.set()
                    cancel = getattr(self.provider, "cancel_active_request", None) or getattr(self.provider, "cancel", None)
                    if callable(cancel):
                        try:
                            cancel()
                        except Exception:
                            pass
                    self.events.publish(
                        "provider.activity", "Request stopped by user", actor=actor,
                        phase=provider_phase, state="stopped", provider_state="stopped",
                        operation=f"Stopped {self.model_name} for {actor}", waiting_on="user",
                        heartbeat_at=time.time(), elapsed_seconds=int(total),
                    )
                    raise HardStopError("provider request stopped by user; saved stage is resumable")
                if not diagnostic_in_flight and now >= local_poll_at:
                    diagnostic = getattr(self.provider, "diagnose_activity", None) if self.execution_class == "local" else getattr(self.provider, "diagnose_connectivity", None)
                    if callable(diagnostic):
                        diagnostic_in_flight = True
                        def diagnose() -> None:
                            try:
                                value = diagnostic()
                                if isinstance(value, Mapping):
                                    diagnostic_results.put(dict(value))
                                else:
                                    diagnostic_results.put({"state": str(value)})
                            except Exception as exc:
                                diagnostic_results.put({"state": "unavailable", "detail": f"{type(exc).__name__}: {exc}"})
                        Thread(target=diagnose, name=f"ga3bad-provider-diagnose-{actor[:16]}", daemon=True).start()
                    local_poll_at = now + 5.0
                try:
                    diagnosis = diagnostic_results.get_nowait()
                except Empty:
                    diagnosis = None
                if diagnosis:
                    diagnostic_in_flight = False
                    dstate = str(diagnosis.get("state") or "")
                    if self.execution_class == "local" and dstate == "generating":
                        self.events.publish(
                            "provider.activity",
                            "Ollama is actively generating on GPU · structured response is atomic",
                            source_kind="MODEL", actor=actor, phase=provider_phase,
                            state="active", provider_state="server_processing",
                            operation=f"Local model generating for {actor}", waiting_on="model",
                            received_bytes=0, received_chunks=0, heartbeat_at=time.time(),
                        )
                    elif self.execution_class == "cloud" and not bool(diagnosis.get("reachable", True)):
                        self.events.publish(
                            "provider.activity",
                            "Internet/provider unavailable · saved stage unchanged",
                            source_kind="MODEL", actor=actor, phase=provider_phase,
                            state="waiting", provider_state="network_unavailable",
                            operation=f"Waiting for {self.provider_name} connectivity", waiting_on="network",
                            detail=str(diagnosis.get("detail") or diagnosis.get("state") or ""),
                            heartbeat_at=time.time(),
                        )
                if now >= next_heartbeat:
                    with self._live_activity_lock:
                        live_activity = dict(self._live_provider_activity)
                    self.events.publish(
                        "heartbeat",
                        f"{actor} provider request is open",
                        actor=actor,
                        elapsed_seconds=int(total),
                        quiet_seconds=int(quiet),
                        phase=provider_phase,
                        waiting_on="model",
                        heartbeat_at=time.time(),
                        active_actor=actor,
                        active_step=step,
                        source_kind="MODEL",
                        state="active",
                        provider_state=str(live_activity.get("state") or "client_active"),
                        operation=str(live_activity.get("operation") or f"Calling {self.model_name}"),
                        received_bytes=int(live_activity.get("received_bytes") or 0),
                        received_chunks=int(live_activity.get("received_chunks") or 0),
                        received_tokens=int(live_activity.get("received_tokens") or 0),
                    )
                    try:
                        session = self.store.get_workflow_session(self.session_id)
                        lease = dict(
                            (session.get("state") or {}).get("execution_lease") or {}
                        )
                        if (
                            str(lease.get("worker_id") or "") == self._worker_id
                            and str(lease.get("lease_state") or "") == "active"
                        ):
                            self._update_execution_lease(
                                stage=f"provider:{actor}", state="active"
                            )
                    except Exception:
                        pass
                    next_heartbeat = now + heartbeat
                if quiet >= 60 and not slow_warning_sent:
                    slow_warning_sent = True
                    self.events.publish(
                        "warning",
                        f"Still calling {self.model_name} for {actor}; no response bytes for "
                        f"{int(quiet)} seconds. The request is saved and no workspace action "
                        "has been replayed.",
                        actor=actor,
                        non_blocking=True,
                        phase=provider_phase,
                        waiting_on="model",
                    )
                deadline = float(policy.stage_deadline_seconds) if policy and policy.stage_deadline_seconds else float(self.config.provider_call_timeout_seconds)
                timed_out = total >= deadline
                silent = cloud and quiet >= self.config.cloud_idle_timeout_seconds
                if timed_out or silent:
                    abandoned.set()
                    cancel = getattr(self.provider, "cancel_active_request", None) or getattr(self.provider, "cancel", None)
                    if callable(cancel):
                        try:
                            cancel()
                        except Exception:
                            pass
                    reason = (
                        f"cloud stream was silent for {int(quiet)} seconds"
                        if silent
                        else f"provider call exceeded {int(deadline)} seconds"
                    )
                    raise TimeoutError(reason)
                continue
            if kind == "error":
                raise value
            if abandoned.is_set() or self._stop_event.is_set():
                raise HardStopError("provider response arrived after the request was stopped")
            self._active_provider_abandon = None
            return value

    def start_goal(
        self,
        objective: str,
        *,
        planning_only: bool = False,
        execution_mode: str | RunMode = RunMode.NORMAL,
        entry_surface: str = "goal",
    ) -> Plan | None:
        with self._lock:
            self._stop_event.clear()
            # Durable objective state preserves the exact user request. Trace
            # and tool records are redacted separately at their boundaries.
            safe_objective = str(objective)
            if not safe_objective.strip():
                raise ValueError("goal objective must not be empty")
            selected_mode = RunMode.parse(execution_mode)
            prior_session = self.store.get_workflow_session(self.session_id)
            prior_state = dict(prior_session.get("state", {}))
            pending_turn = prior_state.get("pending_semantic_turn")
            pending_turn = dict(pending_turn) if isinstance(pending_turn, Mapping) else {}
            capability_snapshot = dict(
                pending_turn.get("model_capability_envelope")
                or self.model_capability_envelope().to_dict()
            )
            demand_snapshot = dict(pending_turn.get("task_demand") or {})
            strategy_snapshot = dict(pending_turn.get("strategy_decision") or {})
            route_snapshot = dict(
                pending_turn.get("route_decision")
                or pending_turn.get("decision")
                or {}
            )
            interaction_mode = str(
                pending_turn.get("interaction_mode")
                or prior_state.get("interaction_mode")
                or InteractionModeV2.WORKING.value
            )
            initial_strategy = str(strategy_snapshot.get("strategy") or "").casefold()
            if selected_mode is not RunMode.ULTRA and initial_strategy in {"recursive", "ultra", "deep"}:
                # A stale/legacy pending envelope must not silently turn a
                # normal request into Ultra.  Explicit Ultra remains the only
                # user-authorized capability escalation path.
                initial_strategy = ExecutionStrategyV1.STAGED.value
            if not initial_strategy:
                initial_strategy = (
                    ExecutionStrategyV1.RECURSIVE.value
                    if selected_mode is RunMode.ULTRA
                    else ExecutionStrategyV1.STAGED.value
                )
            if strategy_snapshot:
                strategy_snapshot["strategy"] = initial_strategy
            try:
                adaptation_envelope = (
                    ModelCapabilityEnvelopeV1.from_mapping(capability_snapshot)
                    if capability_snapshot
                    else self.model_capability_envelope()
                )
                adaptation_policy = self.local_adaptation_policy(
                    adaptation_envelope
                )
            except (TypeError, ValueError):
                adaptation_envelope = self.model_capability_envelope()
                adaptation_policy = self.local_adaptation_policy()
            continuation = prior_state.get("below_target_continuation")
            continuing_chat_candidate = bool(
                isinstance(continuation, Mapping) and is_goal_escalation_approval(safe_objective)
            )
            if continuing_chat_candidate:
                safe_objective = str(continuation.get("objective") or prior_state.get("original_objective") or safe_objective)
            run_id = (
                str(prior_state.get("run_id"))
                if continuing_chat_candidate and prior_state.get("run_id")
                else f"run-{hashlib.sha256((safe_objective + str(time.time_ns())).encode()).hexdigest()[:20]}"
            )
            preserved_artifacts = tuple(continuation.get("artifacts", ())) if continuing_chat_candidate else ()
            contract = GoalContractV1(
                run_id=run_id,
                original_objective=safe_objective,
                interpreted_objective=safe_objective,
                forbidden_shortcuts=("prose-only completion", "model-declared completion without fresh evidence", "automatic acceptance of the first syntactically valid result"),
                completion_conditions=("all accepted tasks complete", "required executable evidence is fresh", "independent evaluation passes", "quality target is converged"),
                artifact_expectations=preserved_artifacts,
            )
            semantic_goal = SemanticGoalV2.pending(safe_objective)
            verifier_plugins = discover_verifier_plugins(self.workspace)
            goal_metadata = {
                    "run_id": run_id,
                    "weak_model_policy": self.weak_model_policy.to_dict(),
                    "goal_contract": contract.to_dict(),
                    "goal_contract_fingerprint": contract.fingerprint,
                    "semantic_goal": semantic_goal.to_dict(),
                    "semantic_goal_fingerprint": semantic_goal.fingerprint,
                    "execution_policy": {
                        "mode": selected_mode.value,
                        "strategy": initial_strategy,
                        "reasoning_depth": (
                            "direct" if entry_surface == "chat" else
                            "deep" if selected_mode is RunMode.ULTRA else
                            "plan_only" if planning_only else "adaptive"
                        ),
                        "parallelism": (
                            "independent_tasks_only"
                            if selected_mode is RunMode.ULTRA
                            else "conservative"
                        ),
                        "entry_surface": entry_surface,
                    },
                    "interaction_mode": interaction_mode,
                    "route": str(route_snapshot.get("route") or "goal"),
                    "model_capability_envelope": capability_snapshot,
                    "local_adaptation_policy": adaptation_policy,
                    "capability_fingerprint": str(
                        pending_turn.get("capability_fingerprint")
                        or adaptation_envelope.fingerprint
                    ),
                    "task_demand": demand_snapshot,
                    "task_demand_fingerprint": str(
                        strategy_snapshot.get("demand_fingerprint") or ""
                    ),
                    "strategy_decision": strategy_snapshot,
                    "strategy_fingerprint": str(
                        hashlib.sha256(
                            json.dumps(strategy_snapshot, sort_keys=True).encode()
                        ).hexdigest()
                        if strategy_snapshot
                        else ""
                    ),
                    "strategy_locked": False,
                    "discovered_verifier_plugins": [
                        item.to_dict() for item in verifier_plugins
                    ],
                    "convergence_state": "not_evaluated",
                    "mutation_sequence": 0,
                    "continued_from_chat": continuing_chat_candidate,
                    "chat_candidate": dict(continuation) if continuing_chat_candidate else {},
                }
            bound_state = {
                **prior_state,
                "route": str(route_snapshot.get("route") or "goal"),
                "execution_strategy": initial_strategy,
                "strategy_decision": strategy_snapshot,
                "interaction_mode": interaction_mode,
            }
            goal, _bound_session = self.store.create_goal_and_bind_workflow_session(
                safe_objective,
                session_id=self.session_id,
                metadata=goal_metadata,
                session_mode=(
                    SessionMode.PLAN.value if planning_only else selected_mode.value
                ),
                plan_state=PlanState.INSPECTING.value,
                run_state=RunState.PLANNING.value,
                sleep_state=str(prior_session.get("sleep_state") or "off"),
                state=bound_state,
                expected_revision=int(prior_session.get("revision") or 0),
                complete_semantic_turn_id=(
                    str(pending_turn.get("turn_id") or "") or None
                ),
                initial_status=GoalStatus.DISCOVERING,
            )
            self.store.append_event(
                "goal_contract.created", goal_id=goal.id,
                payload={"run_id": run_id, "fingerprint": contract.fingerprint, "policy_version": self.weak_model_policy.version},
            )
            self.store.append_event(
                "workflow.state",
                goal_id=goal.id,
                payload={
                    "route": "goal",
                    "execution_strategy": str(strategy_snapshot.get("strategy") or ("recursive" if selected_mode is RunMode.ULTRA else "staged")),
                    "phase": "planning",
                    "model": self.model_name,
                    "provider": self.provider_name,
                },
            )
            self._work_conversation.clear()
            self.events.publish("phase", "Discovering the workspace and drafting a plan.", goal_id=goal.id)
            try:
                return self.generate_plan()
            except ProviderUnavailableError as exc:
                self.store.append_event("planning.checkpoint", goal_id=goal.id, payload={"error": redact_text(exc, 500)})
                retry_after = getattr(exc, "retry_after_seconds", None)
                overloaded = "temporarily overloaded" in str(exc).casefold()
                self.store.update_goal_metadata(
                    goal.id,
                    waiting_question=(
                        (
                            "The selected provider is temporarily overloaded at a saved planning checkpoint. "
                            "Retry after the shown backoff or change model; no work was replayed and no local fallback was applied."
                        )
                        if overloaded
                        else
                        "The provider retries were exhausted at a saved planning checkpoint. "
                        "Choose retry, wait, or change model; no local fallback was applied."
                    ),
                    resume_status=GoalStatus.DISCOVERING.value,
                    provider_recovery={
                        "state": "paused",
                        "error": redact_text(exc, 500),
                        "automatic_fallback": False,
                    },
                    retry_not_before=(
                        time.time() + int(retry_after)
                        if retry_after is not None
                        else None
                    ),
                )
                self.store.transition_goal(
                    goal.id,
                    GoalStatus.PAUSED,
                    reason="provider unavailable at planning checkpoint",
                )
                self.events.publish("error", str(exc))
                return None

    def _planner_tools(self) -> list[dict[str, Any]]:
        return [*_schemas(READ_ONLY_TOOLS), *PLANNER_SCHEMAS]

    @staticmethod
    def _plan_requires_critic(candidate: Mapping[str, Any]) -> bool:
        # Every accepted semantic/plan pair receives an independent critic.
        # Compatibility affects transport parsing only, never review coverage.
        return True

    def _save_pending_semantic_turn(self, turn: Mapping[str, Any]) -> None:
        pending_value = dict(turn)
        pending_status = str(turn.get("status") or "").casefold()
        # Give every saved provider attempt a stable identity.  The identity
        # changes when the selected model/capability envelope changes, while
        # repeated polling/recovery writes for the same attempt remain the
        # same attempt.  Web and terminal can therefore reconcile a failover
        # without displaying an old model envelope as if it were current.
        turn_id = str(pending_value.get("turn_id") or "")
        capability_fingerprint = str(
            pending_value.get("capability_fingerprint") or ""
        )
        # The runtime's selected provider is authoritative after a model
        # switch/failover; never retain the previous model label in the saved
        # attempt envelope.
        attempt_model = self.model_name
        attempt_key = "\0".join(
            (turn_id, self.provider_name, attempt_model, capability_fingerprint)
        )
        if turn_id:
            pending_value["attempt_id"] = "attempt-" + hashlib.sha256(
                attempt_key.encode("utf-8", errors="replace")
            ).hexdigest()[:24]
        pending_value["attempt_model"] = attempt_model
        pending_value["attempt_state"] = (
            "completed"
            if pending_status == "completed"
            else "waiting"
            if pending_status in {"awaiting_provider", "blocked", "paused"}
            else "failed"
            if pending_status in {"failed", "error"}
            else "running"
        )
        pending_value["retry_at"] = pending_value.get("retry_at") or pending_value.get(
            "retry_not_before"
        )
        if pending_value.get("last_error") and not pending_value.get("failure_kind"):
            error_text = str(pending_value.get("last_error") or "").casefold()
            pending_value["failure_kind"] = (
                "transport"
                if any(marker in error_text for marker in ("network", "offline", "timeout", "connection"))
                else "quota"
                if any(marker in error_text for marker in ("quota", "usage limit", "limit exhausted"))
                else "rate_limit"
                if any(marker in error_text for marker in ("rate limit", "too many requests", "throttl"))
                else "contract"
                if any(marker in error_text for marker in ("schema", "structured", "validation", "invalid"))
                else "provider"
            )
        try:
            raw_envelope = pending_value.get("model_capability_envelope")
            envelope = (
                ModelCapabilityEnvelopeV1.from_mapping(raw_envelope)
                if isinstance(raw_envelope, Mapping)
                else self.model_capability_envelope()
            )
            pending_value["local_adaptation_policy"] = self.local_adaptation_policy(
                envelope
            )
        except (TypeError, ValueError):
            # A malformed legacy envelope must not make the durable semantic
            # checkpoint disappear; the current provider remains authoritative.
            pending_value["local_adaptation_policy"] = self.local_adaptation_policy()

        def reduce_session(current: dict[str, Any]) -> Mapping[str, Any]:
            state = dict(current.get("state") or {})
            state["pending_semantic_turn"] = pending_value
            decision = pending_value.get("route_decision") or pending_value.get("decision")
            # A provider boundary during Goal Intake must not erase the route
            # that the preceding semantic stage already accepted. Only the
            # route stage itself is genuinely pending.
            if pending_status == "routing" or (
                pending_status == "awaiting_provider" and not isinstance(decision, Mapping)
            ):
                state["route"] = "pending"
            elif isinstance(decision, Mapping):
                state["route"] = str(decision.get("route") or state.get("route") or "pending")
                strategy = pending_value.get("strategy_decision")
                if isinstance(strategy, Mapping) and strategy.get("strategy"):
                    state["execution_strategy"] = str(strategy["strategy"])
            run_state = str(current.get("run_state") or RunState.IDLE.value)
            if not current.get("goal_id"):
                if pending_status == "awaiting_provider":
                    run_state = RunState.BLOCKED.value
                elif pending_status and pending_status != "completed":
                    run_state = RunState.PLANNING.value
            return {
                "state": state,
                "goal_id": current.get("goal_id"),
                "session_mode": str(current.get("session_mode") or SessionMode.NORMAL.value),
                "plan_state": str(current.get("plan_state") or PlanState.NONE.value),
                "run_state": run_state,
                "ultra_profile": str(current.get("ultra_profile") or "standard"),
                "sleep_state": str(current.get("sleep_state") or "off"),
            }

        self.store.mutate_workflow_session(self.session_id, reduce_session)

    def _complete_semantic_turn(self, turn_id: str, *, result_status: str) -> None:
        session = self.store.get_workflow_session(self.session_id)
        expected_revision = int(session.get("revision") or 0)

        def reduce_session(current: dict[str, Any]) -> Mapping[str, Any]:
            state = dict(current.get("state", {}))
            pending = dict(state.get("pending_semantic_turn", {}))
            if str(pending.get("turn_id")) != str(turn_id):
                raise WorkflowSessionConflictError(
                    "semantic turn was replaced before completion"
                )
            pending.update({"status": "completed", "result_status": str(result_status)})
            pending["attempt_state"] = "completed"
            decision = pending.get("route_decision") or pending.get("decision")
            if isinstance(decision, Mapping):
                state["route"] = str(decision.get("route") or state.get("route") or "pending")
                strategy = pending.get("strategy_decision")
                if isinstance(strategy, Mapping) and strategy.get("strategy"):
                    state["execution_strategy"] = str(strategy["strategy"])
            state["last_semantic_turn"] = pending
            state.pop("pending_semantic_turn", None)
            return {
                "state": state,
                "goal_id": current.get("goal_id"),
                "session_mode": str(current.get("session_mode") or SessionMode.NORMAL.value),
                "plan_state": str(current.get("plan_state") or PlanState.NONE.value),
                "run_state": str(current.get("run_state") or RunState.IDLE.value),
                "ultra_profile": str(current.get("ultra_profile") or "standard"),
                "sleep_state": str(current.get("sleep_state") or "off"),
            }

        try:
            self.store.mutate_workflow_session(
                self.session_id,
                reduce_session,
                expected_revision=expected_revision,
            )
        except WorkflowSessionConflictError:
            # A newer worker already completed/replaced the turn.  Its durable
            # state is authoritative; never overwrite it with this stale read.
            return

    def _record_semantic_action(
        self,
        turn_id: str,
        action_id: str,
        *,
        category: str,
        mutating: bool,
        status: str,
        output: str = "",
        changed_paths: Sequence[str] = (),
    ) -> None:
        if not turn_id:
            return
        session = self.store.get_workflow_session(self.session_id)
        expected_revision = int(session.get("revision") or 0)
        record = {
            "action_id": action_id,
            "category": str(category),
            "mutating": bool(mutating),
            "status": str(status),
            "output": redact_text(output, 2_000),
            "changed_paths": list(changed_paths),
        }

        def reduce_session(current: dict[str, Any]) -> Mapping[str, Any]:
            state = dict(current.get("state", {}))
            pending = dict(state.get("pending_semantic_turn", {}))
            if str(pending.get("turn_id")) != str(turn_id):
                raise WorkflowSessionConflictError(
                    "semantic action belongs to a newer turn"
                )
            records = [
                dict(item)
                for item in pending.get("action_records", ())
                if isinstance(item, Mapping)
            ]
            replaced = False
            for index, item in enumerate(records):
                if str(item.get("action_id")) == action_id:
                    records[index] = record
                    replaced = True
                    break
            if not replaced:
                records.append(record)
            pending["action_records"] = records
            pending["status"] = "dispatching"
            state["pending_semantic_turn"] = pending
            return {
                "state": state,
                "goal_id": current.get("goal_id"),
                "session_mode": str(current.get("session_mode") or SessionMode.NORMAL.value),
                "plan_state": str(current.get("plan_state") or PlanState.NONE.value),
                "run_state": str(current.get("run_state") or RunState.IDLE.value),
                "ultra_profile": str(current.get("ultra_profile") or "standard"),
                "sleep_state": str(current.get("sleep_state") or "off"),
            }

        try:
            self.store.mutate_workflow_session(
                self.session_id,
                reduce_session,
                expected_revision=expected_revision,
            )
        except WorkflowSessionConflictError:
            return

    def _semantic_artifact_manifest(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            {
                key: item.get(key)
                for key in ("id", "language", "suggested_name", "content_hash", "byte_size")
            }
            for item in self.store.list_chat_artifacts(self.session_id)[-10:]
        )

    def _semantic_preflight(
        self,
        text: str | None = None,
        *,
        forced_route: RouteKind | None = None,
        requested_mode: str | RunMode | None = None,
        answers: Mapping[str, str] | None = None,
        resume_pending: bool = False,
    ) -> tuple[dict[str, Any], SemanticTurnDecisionV2]:
        """Persist, obtain, and validate one model-owned semantic decision."""

        session = self.store.get_workflow_session(self.session_id)
        state = dict(session.get("state", {}))
        existing = state.get("pending_semantic_turn")
        if not resume_pending and isinstance(existing, Mapping):
            pending_status = str(existing.get("status") or "")
            incoming = str(text or "")
            if pending_status != "completed" and incoming == str(existing.get("original_input") or ""):
                resume_pending = True
            elif pending_status != "completed":
                raise RuntimeStateError(
                    "Another request is already saved in semantic routing. Resume it with /resume, "
                    "queue this request, or cancel the current workflow first."
                )
        if resume_pending:
            if not isinstance(existing, Mapping):
                raise RuntimeStateError("there is no pending semantic turn to resume")
            pending = dict(existing)
            original = str(pending.get("original_input") or "")
            forced_value = str(pending.get("forced_route") or "")
            forced_route = RouteKind(forced_value) if forced_value else None
            requested_mode = str(pending.get("requested_mode") or session["session_mode"])
            answers = dict(pending.get("answers", {}))
        else:
            original = str(text or "")
            if not original.strip():
                raise ValueError("semantic input must not be empty")
            bound_mode = RunMode.parse(requested_mode or session["session_mode"])
            turn_id = "turn-" + hashlib.sha256(
                f"{self.session_id}\0{time.time_ns()}\0{original}".encode("utf-8")
            ).hexdigest()[:24]
            capability_envelope = self.model_capability_envelope()
            pending = {
                "turn_id": turn_id,
                "original_input": original,
                "request_fingerprint": hashlib.sha256(original.encode("utf-8")).hexdigest(),
                "requested_mode": bound_mode.value,
                "interaction_mode": (
                    InteractionModeV2.PLAN.value
                    if bound_mode is RunMode.PLAN
                    else InteractionModeV2.WORKING.value
                ),
                "minimum_strategy": (
                    ExecutionStrategyV1.RECURSIVE.value
                    if (
                        bound_mode is RunMode.ULTRA
                        or str(state.get("minimum_strategy") or "").casefold()
                        in {"recursive", "ultra"}
                    )
                    else ExecutionStrategyV1.STAGED.value
                ),
                "forced_route": forced_route.value if forced_route else "",
                "answers": dict(answers or {}),
                "status": "routing",
                "stage": "route",
                "schema_attempts": 0,
                "semantic_attempts": 0,
                "route_schema_attempts": 0,
                "route_semantic_attempts": 0,
                "intake_schema_attempts": 0,
                "intake_semantic_attempts": 0,
                "decision": None,
                "model_capability_envelope": capability_envelope.to_dict(),
                "capability_fingerprint": capability_envelope.fingerprint,
                "created_at_ns": time.time_ns(),
            }
            self._save_pending_semantic_turn(pending)
            # The exact request is durable before the first provider call.
            self.store.append_chat_message(
                self.session_id,
                {"role": "user", "content": original},
                event_key=f"semantic:{turn_id}:user",
                run_id=turn_id,
            )
            self._chat_conversation.append({"role": "user", "content": original})
            self.store.append_event(
                "semantic_turn.routing",
                entity_type="semantic_turn",
                entity_id=turn_id,
                payload={
                    "request_fingerprint": pending["request_fingerprint"],
                    "requested_mode": pending["requested_mode"],
                    "forced_route": pending["forced_route"],
                    "provider": self.provider_name,
                    "model": self.model_name,
                    "capability_fingerprint": capability_envelope.fingerprint,
                },
            )

        accepted = pending.get("route_decision") or pending.get("decision")
        if isinstance(accepted, Mapping) and (
            str(pending.get("status")) in {"routed", "dispatching"}
            or str(pending.get("stage")) == "goal_intake"
        ):
            decision = SemanticTurnDecisionV2.from_mapping(
                accepted,
                original_input=original,
                forced_route=forced_route,
                parse_goal_intake=False,
            )
            capability_raw = pending.get("model_capability_envelope")
            capability_envelope = (
                ModelCapabilityEnvelopeV1.from_mapping(capability_raw)
                if isinstance(capability_raw, Mapping)
                else self.model_capability_envelope()
            )
            strategy = select_execution_strategy(
                capability_envelope,
                decision.task_demand,
                minimum=(
                    ExecutionStrategyV1.RECURSIVE
                    if str(pending.get("minimum_strategy") or pending.get("requested_mode")).casefold()
                    in {"recursive", "ultra"}
                    else ExecutionStrategyV1.STAGED
                ),
                allow_capability_escalation=(
                    str(pending.get("minimum_strategy") or pending.get("requested_mode")).casefold()
                    in {"recursive", "ultra"}
                ),
            )
            if strategy.strategy is ExecutionStrategyV1.RECURSIVE:
                pending["minimum_strategy"] = ExecutionStrategyV1.RECURSIVE.value
            if decision.route is RouteKind.ACTION and strategy.strategy is ExecutionStrategyV1.RECURSIVE:
                decision = decision.promote_action_to_goal()
                pending.update({
                    "status": "routed",
                    "stage": "goal_intake",
                    "route_decision": decision.to_dict(),
                    "decision": decision.to_dict(),
                    "strategy_decision": strategy.to_dict(),
                })
                self._save_pending_semantic_turn(pending)
            if decision.route is RouteKind.GOAL:
                legacy_intake = accepted.get("goal_intake")
                if isinstance(legacy_intake, Mapping) and not pending.get("legacy_goal_intake"):
                    pending["legacy_goal_intake"] = dict(legacy_intake)
                return pending, self._semantic_goal_intake_preflight(
                    pending,
                    decision,
                    answers=answers,
                )
            return pending, decision

        manifest = self._semantic_artifact_manifest()
        recent = [
            {"role": item.get("role"), "content": item.get("content")}
            for item in self._chat_conversation[-12:]
            if item.get("role") in {"user", "assistant"}
        ]
        capability_value = pending.get("model_capability_envelope")
        capability_envelope = (
            ModelCapabilityEnvelopeV1.from_mapping(capability_value)
            if isinstance(capability_value, Mapping)
            else self.model_capability_envelope()
        )
        envelope = {
            "exact_latest_user_input": original,
            "recent_conversation": recent,
            "workflow_mode": str(pending.get("requested_mode")),
            "artifact_manifest": manifest,
            "repository_manifest": self._intake_repository_facts(original),
            "answered_intake_decisions": dict(answers or {}),
            "forced_route": forced_route.value if forced_route else None,
            "MODEL_CAPABILITY_ENVELOPE": capability_envelope.to_dict(),
        }
        conversation: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": state_envelope(
                    envelope,
                    "SEMANTIC_TURN_INPUT",
                    max_chars=max(18_000, len(original) * 2 + 12_000),
                ),
            }
        ]
        schema_repairs = int(pending.get("schema_attempts", 0))
        semantic_repairs = int(pending.get("semantic_attempts", 0))
        repair_limit = self._structured_repair_limit()
        step = schema_repairs + semantic_repairs + 1
        while True:
            try:
                turn = self._call_provider(
                    conversation,
                    [SEMANTIC_ROUTE_SCHEMA],
                    SEMANTIC_ROUTER_SYSTEM_PROMPT,
                    actor="semantic-router",
                    step=step,
                    stream_text=False,
                    normalization_context={"exact_latest_user_input": original},
                )
            except ProviderUnavailableError as exc:
                pending.update({
                    "status": "awaiting_provider",
                    "stage": "route",
                    "last_error": redact_text(exc, 1_000),
                    "last_validation_error": {
                        "stage": "route",
                        "category": "provider",
                        "message": redact_text(exc, 1_000),
                    },
                })
                self._save_pending_semantic_turn(pending)
                self.store.append_event(
                    "semantic_turn.provider_boundary",
                    entity_type="semantic_turn",
                    entity_id=str(pending["turn_id"]),
                    payload={"error": redact_text(exc, 500), "resumable": True},
                )
                self._release_execution_lease(
                    stage="provider:semantic-router",
                    state="boundary",
                )
                raise
            conversation.append(turn.to_message())
            structural_error = ""
            if len(turn.tool_calls) != 1:
                structural_error = "submit_semantic_route must be called exactly once"
            elif turn.tool_calls[0].name not in {"submit_semantic_route", "submit_semantic_turn"}:
                structural_error = "the only allowed call is submit_semantic_route"
            elif not isinstance(turn.tool_calls[0].args, Mapping):
                structural_error = "submit_semantic_turn arguments must be an object"
            if structural_error:
                schema_repairs += 1
                pending.update({
                    "schema_attempts": schema_repairs,
                    "route_schema_attempts": schema_repairs,
                    "status": "routing",
                    "stage": "route",
                    "last_error": structural_error,
                    "last_validation_error": {
                        "stage": "route", "category": "schema",
                        "message": structural_error, "attempts": schema_repairs,
                    },
                })
                self._save_pending_semantic_turn(pending)
                if schema_repairs > repair_limit:
                    break
                conversation.append({"role": "user", "content": "SCHEMA VALIDATION ERROR: " + structural_error + ". Repair only the function-call shape."})
                step += 1
                continue
            try:
                decision = SemanticTurnDecisionV2.from_mapping(
                    turn.tool_calls[0].args,
                    original_input=original,
                    forced_route=forced_route,
                    parse_goal_intake=False,
                )
                strategy = select_execution_strategy(
                    capability_envelope,
                    decision.task_demand,
                    minimum=(
                        ExecutionStrategyV1.RECURSIVE
                        if str(pending.get("minimum_strategy") or "").casefold()
                        in {"recursive", "ultra"}
                        else ExecutionStrategyV1.STAGED
                    ),
                    allow_capability_escalation=(
                        str(pending.get("minimum_strategy") or "").casefold()
                        in {"recursive", "ultra"}
                    ),
                )
                if strategy.strategy is ExecutionStrategyV1.RECURSIVE:
                    pending["minimum_strategy"] = ExecutionStrategyV1.RECURSIVE.value
                if decision.route is RouteKind.ACTION and strategy.strategy is ExecutionStrategyV1.RECURSIVE:
                    decision = decision.promote_action_to_goal()
                if (
                    RunMode.parse(str(pending.get("requested_mode"))) is RunMode.PLAN
                    and decision.route is RouteKind.ACTION
                    and any(effect is not RequestedEffectV2.READ for effect in decision.requested_effects)
                ):
                    raise ValueError(
                        "Plan mode forbids changing Action execution; return a Goal route for planning only"
                    )
            except (TypeError, ValueError) as exc:
                semantic_repairs += 1
                message = str(exc)
                detail = exc.to_dict() if isinstance(exc, SemanticContractError) else {"message": message}
                pending.update({
                    "semantic_attempts": semantic_repairs,
                    "route_semantic_attempts": semantic_repairs,
                    "status": "routing",
                    "stage": "route",
                    "last_error": message,
                    "last_validation_error": {
                        "stage": "route", "category": "semantic",
                        "attempts": semantic_repairs, **detail,
                    },
                })
                self._save_pending_semantic_turn(pending)
                if semantic_repairs > repair_limit:
                    break
                conversation.append({"role": "user", "content": "SEMANTIC CONSISTENCY ERROR: " + message + ". Preserve the exact request and repair only this inconsistency."})
                step += 1
                continue
            raw_intake = turn.tool_calls[0].args.get("goal_intake")
            pending.update({
                "status": "routed",
                "stage": "goal_intake" if decision.route is RouteKind.GOAL else "dispatching",
                "route_decision": decision.to_dict(),
                "route_tool_name": turn.tool_calls[0].name,
                "decision": decision.to_dict(),
                "schema_attempts": schema_repairs, "semantic_attempts": semantic_repairs,
                "route_schema_attempts": schema_repairs,
                "route_semantic_attempts": semantic_repairs,
                "contract_fingerprint": decision.fingerprint, "last_error": "",
                "task_demand": decision.task_demand.to_dict(),
                "strategy_decision": strategy.to_dict(),
                "last_validation_error": {},
            })
            if isinstance(raw_intake, Mapping):
                pending["legacy_goal_intake"] = dict(raw_intake)
            self._save_pending_semantic_turn(pending)
            self.store.append_event(
                "semantic_turn.routed",
                entity_type="semantic_turn",
                entity_id=str(pending["turn_id"]),
                payload={
                    "route": decision.route.value,
                    "rationale": decision.interpretation,
                    "contract_fingerprint": decision.fingerprint,
                    "provider": self.provider_name,
                    "model": self.model_name,
                    "schema_attempts": schema_repairs,
                    "semantic_attempts": semantic_repairs,
                    "task_demand_fingerprint": decision.task_demand.fingerprint,
                    "strategy": strategy.strategy.value,
                    "strategy_fingerprint": strategy.fingerprint,
                    "capability_fingerprint": capability_envelope.fingerprint,
                },
            )
            self.store.append_event(
                "workflow.state",
                entity_type="semantic_turn",
                entity_id=str(pending["turn_id"]),
                payload={
                    "route": decision.route.value,
                    "execution_strategy": strategy.strategy.value,
                    "phase": "routing" if decision.route is RouteKind.GOAL else "working",
                    "model": self.model_name,
                    "provider": self.provider_name,
                },
            )
            if decision.route is RouteKind.GOAL:
                return pending, self._semantic_goal_intake_preflight(
                    pending,
                    decision,
                    answers=answers,
                )
            return pending, decision

        pending.update({
            "status": "awaiting_provider",
            "stage": "route",
            "last_error": str(pending.get("last_error") or "semantic decision validation exhausted"),
        })
        self._save_pending_semantic_turn(pending)
        self.store.append_event(
            "semantic_turn.validation_boundary",
            entity_type="semantic_turn",
            entity_id=str(pending["turn_id"]),
            payload={
                "stage": "route_schema" if schema_repairs > repair_limit else "route_semantic",
                "schema_attempts": schema_repairs,
                "semantic_attempts": semantic_repairs,
                "error": pending["last_error"],
                "resumable": True,
            },
        )
        raise ProviderUnavailableError(
            "semantic routing is saved but could not be validated: " + str(pending["last_error"])
        )

    def _semantic_goal_intake_preflight(
        self,
        pending: dict[str, Any],
        decision: SemanticTurnDecisionV2,
        *,
        answers: Mapping[str, str] | None,
    ) -> SemanticTurnDecisionV2:
        """Obtain Goal intake without replaying or reconsidering the accepted route."""

        original = str(pending.get("original_input") or "")
        requested_mode = str(pending.get("requested_mode") or "normal")
        repository_manifest = self._intake_repository_facts(original)
        capability_raw = pending.get("model_capability_envelope")
        capability_envelope = (
            ModelCapabilityEnvelopeV1.from_mapping(capability_raw)
            if isinstance(capability_raw, Mapping)
            else self.model_capability_envelope()
        )
        minimum_strategy = (
            ExecutionStrategyV1.RECURSIVE
            if (
                str(pending.get("minimum_strategy") or requested_mode).casefold()
                in {"recursive", "ultra"}
                or str(dict(pending.get("strategy_decision") or {}).get("strategy"))
                == ExecutionStrategyV1.RECURSIVE.value
            )
            else ExecutionStrategyV1.STAGED
        )

        accepted = pending.get("goal_intake")
        if isinstance(accepted, Mapping):
            intake = SemanticGoalIntakeV3.from_mapping(accepted)
            self.intent_architect.validate(
                intake,
                original_input=original,
                requested_mode=requested_mode,
                answers=answers,
                repository_facts=tuple(repository_manifest),
            )
            combined = decision.with_goal_intake(intake)
            strategy = select_execution_strategy(
                capability_envelope,
                intake.task_demand,
                minimum=minimum_strategy,
                allow_capability_escalation=(
                    minimum_strategy is ExecutionStrategyV1.RECURSIVE
                ),
            )
            pending.update({
                "decision": combined.to_dict(),
                "task_demand": intake.task_demand.to_dict(),
                "strategy_decision": strategy.to_dict(),
            })
            self._save_pending_semantic_turn(pending)
            return combined

        envelope = {
            "exact_latest_user_input": original,
            "accepted_route": decision.to_dict(),
            "workflow_mode": requested_mode,
            "repository_manifest": repository_manifest,
            "answered_intake_decisions": dict(answers or {}),
            "MODEL_CAPABILITY_ENVELOPE": capability_envelope.to_dict(),
        }
        conversation: list[dict[str, Any]] = [{
            "role": "user",
            "content": state_envelope(
                envelope,
                "SEMANTIC_GOAL_INTAKE_INPUT",
                max_chars=max(18_000, len(original) * 2 + 12_000),
            ),
        }]
        schema_repairs = int(pending.get("intake_schema_attempts", 0))
        semantic_repairs = int(pending.get("intake_semantic_attempts", 0))
        repair_limit = self._structured_repair_limit()
        step = schema_repairs + semantic_repairs + 1
        candidate: Mapping[str, Any] | None = None
        legacy = pending.get("legacy_goal_intake")
        if isinstance(legacy, Mapping) and schema_repairs == 0 and semantic_repairs == 0:
            candidate = dict(legacy)

        while True:
            if candidate is None:
                try:
                    turn = self._call_provider(
                        conversation,
                        [SEMANTIC_GOAL_INTAKE_SCHEMA],
                        SEMANTIC_GOAL_INTAKE_SYSTEM_PROMPT,
                        actor="semantic-goal-intake",
                        step=step,
                        stream_text=False,
                    )
                except ProviderUnavailableError as exc:
                    pending.update({
                        "status": "awaiting_provider",
                        "stage": "goal_intake",
                        "last_error": redact_text(exc, 1_000),
                        "last_validation_error": {
                            "stage": "goal_intake",
                            "category": "provider",
                            "message": redact_text(exc, 1_000),
                        },
                    })
                    self._save_pending_semantic_turn(pending)
                    self.store.append_event(
                        "semantic_turn.provider_boundary",
                        entity_type="semantic_turn",
                        entity_id=str(pending["turn_id"]),
                        payload={
                            "stage": "goal_intake",
                            "error": redact_text(exc, 500),
                            "resumable": True,
                        },
                    )
                    self._release_execution_lease(
                        stage="provider:semantic-goal-intake",
                        state="boundary",
                    )
                    raise
                conversation.append(turn.to_message())
                structural_error = ""
                if len(turn.tool_calls) != 1:
                    structural_error = "submit_goal_intake must be called exactly once"
                elif turn.tool_calls[0].name not in {"submit_goal_intake", "submit_semantic_turn"}:
                    structural_error = "the only allowed call is submit_goal_intake"
                elif not isinstance(turn.tool_calls[0].args, Mapping):
                    structural_error = "submit_goal_intake arguments must be an object"
                if structural_error:
                    schema_repairs += 1
                    pending.update({
                        "intake_schema_attempts": schema_repairs,
                        "status": "routing",
                        "stage": "goal_intake",
                        "last_error": structural_error,
                        "last_validation_error": {
                            "stage": "goal_intake", "category": "schema",
                            "message": structural_error, "attempts": schema_repairs,
                        },
                    })
                    self._save_pending_semantic_turn(pending)
                    if schema_repairs > repair_limit:
                        break
                    conversation.append({
                        "role": "user",
                        "content": "SCHEMA VALIDATION ERROR: " + structural_error
                        + ". Repair only the function-call shape.",
                    })
                    step += 1
                    continue
                raw_candidate = dict(turn.tool_calls[0].args)
                nested = raw_candidate.get("goal_intake")
                if isinstance(nested, Mapping):
                    raw_candidate = dict(nested)
                semantic_nested = raw_candidate.get("semantic_turn")
                if isinstance(semantic_nested, Mapping) and semantic_nested.get("objective"):
                    raw_candidate = dict(semantic_nested)
                candidate = raw_candidate

            try:
                intake = SemanticGoalIntakeV3.from_mapping(candidate)
                if (
                    str(pending.get("route_tool_name")) == "submit_semantic_route"
                    and decision.uncertainty == "clear"
                    and intake.questions
                    and not isinstance(legacy, Mapping)
                ):
                    raise ValueError(
                        "the accepted route is clear; goal_intake.questions must be empty. "
                        "Defer non-consequential choices to planning."
                    )
                self.intent_architect.validate(
                    intake,
                    original_input=original,
                    requested_mode=requested_mode,
                    answers=answers,
                    repository_facts=tuple(repository_manifest),
                )
            except (TypeError, ValueError) as exc:
                semantic_repairs += 1
                message = str(exc)
                detail = exc.to_dict() if isinstance(exc, SemanticContractError) else {"message": message}
                pending.update({
                    "intake_semantic_attempts": semantic_repairs,
                    "status": "routing",
                    "stage": "goal_intake",
                    "last_error": message,
                    "last_validation_error": {
                        "stage": "goal_intake", "category": "semantic",
                        "attempts": semantic_repairs, **detail,
                    },
                    "last_rejected_goal_intake": redact_data(candidate),
                })
                self._save_pending_semantic_turn(pending)
                if semantic_repairs > repair_limit:
                    break
                conversation.append({
                    "role": "user",
                    "content": (
                        "GOAL INTAKE VALIDATION ERROR: " + message
                        + ". The Goal route is already accepted. Repair only this intake field; "
                        "do not reclassify the request."
                    ),
                })
                candidate = None
                step += 1
                continue

            combined = decision.with_goal_intake(intake)
            strategy = select_execution_strategy(
                capability_envelope,
                intake.task_demand,
                minimum=minimum_strategy,
                allow_capability_escalation=(
                    minimum_strategy is ExecutionStrategyV1.RECURSIVE
                ),
            )
            pending.update({
                "status": "routed",
                "stage": "dispatching",
                "goal_intake": intake.to_dict(),
                "decision": combined.to_dict(),
                "intake_schema_attempts": schema_repairs,
                "intake_semantic_attempts": semantic_repairs,
                "contract_fingerprint": combined.fingerprint,
                "task_demand": intake.task_demand.to_dict(),
                "strategy_decision": strategy.to_dict(),
                "last_error": "",
                "last_validation_error": {},
            })
            self._save_pending_semantic_turn(pending)
            self.store.append_event(
                "semantic_turn.intake_accepted",
                entity_type="semantic_turn",
                entity_id=str(pending["turn_id"]),
                payload={
                    "route": "goal",
                    "contract_fingerprint": combined.fingerprint,
                    "provider": self.provider_name,
                    "model": self.model_name,
                    "schema_attempts": schema_repairs,
                    "semantic_attempts": semantic_repairs,
                    "task_demand_fingerprint": intake.task_demand.fingerprint,
                    "strategy": strategy.strategy.value,
                    "strategy_fingerprint": strategy.fingerprint,
                    "capability_fingerprint": capability_envelope.fingerprint,
                },
            )
            return combined

        pending.update({
            "status": "awaiting_provider",
            "stage": "goal_intake",
            "last_error": str(pending.get("last_error") or "Goal intake validation exhausted"),
        })
        self._save_pending_semantic_turn(pending)
        self.store.append_event(
            "semantic_turn.validation_boundary",
            entity_type="semantic_turn",
            entity_id=str(pending["turn_id"]),
            payload={
                "stage": "intake_schema" if schema_repairs > repair_limit else "intake_semantic",
                "schema_attempts": schema_repairs,
                "semantic_attempts": semantic_repairs,
                "error": pending["last_error"],
                "resumable": True,
            },
        )
        raise ProviderUnavailableError(
            "Goal intake is saved but could not be validated: " + str(pending["last_error"])
        )

    @staticmethod
    def _route_effects_as_goal_effects(
        requested_effects: Mapping[str, Any] | Sequence[Any] | None,
    ) -> tuple[RequestedEffect, ...]:
        """Translate an accepted route capability contract without inferring intent."""

        if isinstance(requested_effects, Mapping):
            values = [key for key, enabled in requested_effects.items() if enabled]
        elif isinstance(requested_effects, Sequence) and not isinstance(
            requested_effects, (str, bytes)
        ):
            values = list(requested_effects)
        else:
            values = []
        translated: list[RequestedEffect] = []
        for value in values:
            route_effect = RequestedEffectV2.parse(value)
            mapped = {
                RequestedEffectV2.READ: RequestedEffect.READ_WORKSPACE,
                RequestedEffectV2.WRITE: RequestedEffect.MUTATE_WORKSPACE,
                RequestedEffectV2.RUN: RequestedEffect.EXECUTE_CODE,
                RequestedEffectV2.PREVIEW: RequestedEffect.EXECUTE_CODE,
                RequestedEffectV2.INSTALL: RequestedEffect.INSTALL_DEPENDENCIES,
                RequestedEffectV2.EXTERNAL: RequestedEffect.EXTERNAL_SIDE_EFFECT,
            }[route_effect]
            translated.append(mapped)
        return tuple(dict.fromkeys(translated))

    def _accepted_route_effects(self, original_request: str) -> tuple[RequestedEffect, ...]:
        session = self.store.get_workflow_session(self.session_id)
        state = dict(session.get("state", {}))
        candidates = (
            state.get("pending_semantic_turn"),
            state.get("last_semantic_turn"),
        )
        for raw_turn in candidates:
            if not isinstance(raw_turn, Mapping):
                continue
            if str(raw_turn.get("original_input") or "") != str(original_request):
                continue
            decision = raw_turn.get("route_decision") or raw_turn.get("decision")
            if not isinstance(decision, Mapping):
                continue
            return self._route_effects_as_goal_effects(
                decision.get("requested_effects")
            )
        return ()

    @staticmethod
    def _validate_semantic_stage(
        goal: Goal,
        value: Mapping[str, Any],
        *,
        successful_inspection_ids: frozenset[str],
        accepted_requested_effects: Sequence[RequestedEffect] = (),
    ) -> SemanticGoalV2:
        semantic = SemanticGoalV2.from_mapping(value, original_request=goal.objective)
        if semantic.status != "interpreted":
            raise ValueError("semantic_goal.status must be interpreted")
        if semantic.unresolved_decisions:
            raise ValueError(
                "semantic interpretation still has unresolved decisions; "
                "call request_plan_input"
            )
        cited = {
            ref[len("inspection:") :]
            for ref in semantic.repository_evidence_refs
            if ref.startswith("inspection:")
        }
        if not cited and len(successful_inspection_ids) == 1:
            only_inspection = next(iter(successful_inspection_ids))
            semantic = SemanticGoalV2(
                original_request=semantic.original_request,
                interpreted_outcome=semantic.interpreted_outcome,
                requested_effects=semantic.requested_effects,
                required_outcomes=semantic.required_outcomes,
                constraints=semantic.constraints,
                exclusions=semantic.exclusions,
                acceptance_criteria=semantic.acceptance_criteria,
                requirement_anchors=semantic.requirement_anchors,
                unresolved_decisions=semantic.unresolved_decisions,
                repository_evidence_refs=(f"inspection:{only_inspection}",),
                status=semantic.status,
            )
            cited = {only_inspection}
        if not cited or not cited.issubset(successful_inspection_ids):
            raise ValueError(
                "semantic interpretation must cite successful inspection references"
            )
        # Reading the workspace is execution provenance already proved by the
        # cited inspection, not product semantics authored by the model. Record
        # that observed effect mechanically so a capable plan is not rejected
        # merely because the model omitted the redundant enum value. Never infer
        # mutation, execution, network, or any other requested effect.
        if RequestedEffect.READ_WORKSPACE not in semantic.requested_effects:
            semantic = SemanticGoalV2(
                original_request=semantic.original_request,
                interpreted_outcome=semantic.interpreted_outcome,
                requested_effects=(
                    RequestedEffect.READ_WORKSPACE,
                    *semantic.requested_effects,
                ),
                required_outcomes=semantic.required_outcomes,
                constraints=semantic.constraints,
                exclusions=semantic.exclusions,
                acceptance_criteria=semantic.acceptance_criteria,
                requirement_anchors=semantic.requirement_anchors,
                unresolved_decisions=semantic.unresolved_decisions,
                repository_evidence_refs=semantic.repository_evidence_refs,
                status=semantic.status,
            )
        missing_accepted_effects = tuple(
            effect
            for effect in accepted_requested_effects
            if effect not in semantic.requested_effects
        )
        if missing_accepted_effects:
            semantic = SemanticGoalV2(
                original_request=semantic.original_request,
                interpreted_outcome=semantic.interpreted_outcome,
                requested_effects=(
                    *semantic.requested_effects,
                    *missing_accepted_effects,
                ),
                required_outcomes=semantic.required_outcomes,
                constraints=semantic.constraints,
                exclusions=semantic.exclusions,
                acceptance_criteria=semantic.acceptance_criteria,
                requirement_anchors=semantic.requirement_anchors,
                unresolved_decisions=semantic.unresolved_decisions,
                repository_evidence_refs=semantic.repository_evidence_refs,
                status=semantic.status,
            )
        return semantic

    def _accepted_semantic_metadata(
        self,
        goal: Goal,
        proposed: Mapping[str, Any],
        inspection_records: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Build contracts only from critic-accepted model output and evidence."""

        tasks = tuple(dict(item) for item in proposed.get("tasks", ()))
        changes = tuple(dict(item) for item in proposed.get("expected_changes", ()))
        interpreted = SemanticGoalV2.from_mapping(
            dict(proposed.get("semantic_goal") or {}),
            original_request=goal.objective,
        )
        criteria = interpreted.acceptance_criteria
        if not criteria:
            # Some local tool-calling models place the complete acceptance
            # contract on each task while omitting the redundant semantic-level
            # projection.  Preserve the quality gate by reusing only the
            # model-authored task criteria; do not invent a new requirement or
            # infer one from the objective.  The normalization event makes the
            # repair visible and auditable in both Web and terminal surfaces.
            task_criteria = tuple(
                dict.fromkeys(
                    str(value).strip()
                    for task in tasks
                    for value in task.get("acceptance_criteria", ())
                    if str(value).strip()
                )
            )
            if task_criteria:
                criteria = task_criteria
                self.store.append_event(
                    "planning.semantic_criteria_reconciled",
                    goal_id=goal.id,
                    payload={
                        "source": "task_acceptance_criteria",
                        "count": len(task_criteria),
                    },
                )
        refs = interpreted.repository_evidence_refs
        requested_effects = interpreted.requested_effects
        if changes and RequestedEffect.MUTATE_WORKSPACE not in requested_effects:
            # A weak planner can correctly describe the file mutation in its
            # accepted plan while omitting the redundant semantic effect. The
            # plan's top-level expected-change contract is authoritative
            # evidence that mutation is required; reconcile that transport
            # omission mechanically instead of pausing after the critic has
            # already approved the executable work.
            requested_effects = (
                *requested_effects,
                RequestedEffect.MUTATE_WORKSPACE,
            )
            self.store.append_event(
                "planning.semantic_effects_reconciled",
                goal_id=goal.id,
                payload={
                    "source": "accepted_plan_expected_changes",
                    "effect": RequestedEffect.MUTATE_WORKSPACE.value,
                    "count": len(changes),
                },
            )
        semantic = SemanticGoalV2(
            original_request=interpreted.original_request,
            interpreted_outcome=interpreted.interpreted_outcome,
            requested_effects=requested_effects,
            required_outcomes=interpreted.required_outcomes,
            constraints=interpreted.constraints,
            exclusions=interpreted.exclusions,
            acceptance_criteria=criteria,
            requirement_anchors=interpreted.requirement_anchors,
            unresolved_decisions=interpreted.unresolved_decisions,
            repository_evidence_refs=interpreted.repository_evidence_refs,
            status="critic_accepted",
        )
        claims = tuple(
            ResourceClaimV1(
                purpose=str(change.get("intent") or "").strip()
                or f"Apply accepted task contract to {change.get('path')}",
                kind="file",
                supports_tasks=tuple(change.get("supports_tasks") or ()),
                inspection_refs=tuple(change.get("evidence_refs") or refs),
                selector=str(change.get("path") or ""),
                resolved_paths=(str(change.get("path") or ""),),
                state="resolved",
            )
            for change in changes
        )
        plugins = discover_verifier_plugins(self.workspace)
        verification_items: list[VerificationContractV1] = []
        for task in tasks:
            methods = tuple(
                str(item) for item in task.get("verification", ()) if str(item).strip()
            )
            for index, criterion in enumerate(
                task.get("acceptance_criteria", ())
            ):
                method = methods[min(index, len(methods) - 1)]
                normalized_method = method.casefold()
                matched_plugin = next(
                    (
                        plugin
                        for plugin in plugins
                        if plugin.name.casefold() in normalized_method
                        or " ".join(plugin.command).casefold()
                        in normalized_method
                    ),
                    None,
                )
                authority = (
                    f"{matched_plugin.authority} ({matched_plugin.evidence_path})"
                    if matched_plugin is not None
                    else "accepted task contract with fresh read-back evidence"
                )
                verification_items.append(
                    VerificationContractV1(
                        criterion=str(criterion),
                        method=method,
                        scope=str(task.get("id") or ""),
                        expected_result=str(criterion),
                        authority=authority,
                    )
                )
        verification = tuple(verification_items)
        return {
            "semantic_goal": semantic.to_dict(),
            "semantic_goal_fingerprint": semantic.fingerprint,
            "accepted_semantic_fingerprint": interpreted.fingerprint,
            "resource_claims": [item.to_dict() for item in claims],
            "verification_contracts": [item.to_dict() for item in verification],
        }

    @staticmethod
    def _validate_semantic_candidate(
        goal: Goal,
        proposed: Mapping[str, Any],
        *,
        successful_inspection_ids: frozenset[str],
    ) -> SemanticGoalV2:
        original_proposed = proposed
        proposed = dict(proposed or {})
        semantic_raw, aliases = canonicalize_requirement_anchors(
            dict(proposed.get("semantic_goal") or {})
        )
        proposed["semantic_goal"] = semantic_raw
        rewritten_tasks: list[dict[str, Any]] = []
        for task in proposed.get("tasks", ()):
            copied = dict(task)
            refs: list[str] = []
            for raw_ref in copied.get("requirement_refs", ()):
                ref = str(raw_ref or "").strip().upper()
                mapped = aliases.get(ref, (ref,))
                for canonical in mapped:
                    if canonical not in refs:
                        refs.append(canonical)
            copied["requirement_refs"] = refs
            rewritten_tasks.append(copied)
        proposed["tasks"] = rewritten_tasks
        if isinstance(original_proposed, dict):
            original_proposed.clear()
            original_proposed.update(proposed)
        semantic = SemanticGoalV2.from_mapping(
            dict(proposed.get("semantic_goal") or {}),
            original_request=goal.objective,
        )
        # ``requirement_refs`` is a traceability aid, not new product
        # semantics. Older/weak structured providers often omit the optional
        # field on every task even though their task text names the requested
        # technology or outcome. In that transport-only case, bind existing
        # model-authored anchors to the strongest matching task once; keep
        # partial/contradictory mappings strict so real coverage gaps remain
        # actionable repair boundaries.
        tasks = list(proposed.get("tasks") or ())
        if semantic.requirement_anchors and tasks and not any(
            task.get("requirement_refs") for task in tasks if isinstance(task, Mapping)
        ):
            stop_words = {
                "the", "and", "with", "from", "that", "this", "into", "only",
                "user", "request", "finished", "result", "needs", "must", "should",
                "for", "not", "using", "through", "inside", "every", "actual",
            }

            def terms(value: Any) -> set[str]:
                return {
                    token
                    for token in re.findall(r"[a-z0-9]+", str(value or "").casefold())
                    if len(token) > 2 and token not in stop_words
                }

            task_texts = [
                " ".join(
                    str(task.get(key) or "")
                    for key in ("title", "description", "acceptance_criteria", "verification")
                ).casefold()
                if isinstance(task, Mapping)
                else ""
                for task in tasks
            ]
            inferred: dict[int, list[str]] = {index: [] for index in range(len(tasks))}
            for anchor in semantic.requirement_anchors:
                anchor_terms = terms(
                    " ".join(
                        (
                            anchor.verbatim_span,
                            anchor.interpreted_requirement,
                            *anchor.observable_implications,
                        )
                    )
                )
                anchor_span = anchor.verbatim_span.casefold().strip()
                best_index = -1
                best_score = 0
                for index, text_value in enumerate(task_texts):
                    score = len(anchor_terms & terms(text_value))
                    if anchor_span and anchor_span in text_value:
                        score += 100
                    if score > best_score:
                        best_index, best_score = index, score
                if best_index >= 0 and best_score > 0:
                    inferred[best_index].append(anchor.id)
            if any(inferred.values()):
                for index, refs in inferred.items():
                    if refs and isinstance(tasks[index], dict):
                        tasks[index]["requirement_refs"] = refs
                proposed["tasks"] = tasks
                if isinstance(original_proposed, dict):
                    original_proposed.clear()
                    original_proposed.update(proposed)
        if semantic.status != "interpreted":
            raise ValueError("semantic_goal.status must be interpreted before critic review")
        if semantic.unresolved_decisions:
            raise ValueError(
                "semantic goal contains unresolved decisions; call request_plan_input "
                "instead of proposing executable work"
            )
        cited_inspections = {
            value[len("inspection:") :]
            for value in semantic.repository_evidence_refs
            if value.startswith("inspection:")
        }
        if not cited_inspections or not cited_inspections.issubset(
            successful_inspection_ids
        ):
            raise ValueError(
                "semantic goal must cite successful repository inspection references"
            )
        anchor_ids = {item.id for item in semantic.requirement_anchors}
        covered_anchor_ids: set[str] = set()
        for index, task in enumerate(proposed.get("tasks", ())):
            refs = {
                str(item).strip().upper()
                for item in task.get("requirement_refs", ())
                if str(item).strip()
            }
            unknown = refs - anchor_ids
            if unknown:
                raise ValueError(
                    f"task {index + 1} references unknown requirement anchors: "
                    + ", ".join(sorted(unknown))
                )
            covered_anchor_ids.update(refs)
        uncovered = anchor_ids - covered_anchor_ids
        if uncovered:
            raise ValueError(
                "plan tasks do not cover requirement anchors: "
                + ", ".join(sorted(uncovered))
            )
        # Semantic equivalence between an accepted criterion and a task is
        # deliberately model-owned. Keyword/token overlap rejected valid
        # paraphrases and made punctuation-heavy criteria brittle. The fresh
        # plan critic receives this fingerprint-bound semantic object and the
        # complete task contracts; deterministic checks below retain safety,
        # provenance, effects, paths, and executable structure.
        effects = set(semantic.requested_effects)
        changes = tuple(proposed.get("expected_changes", ()))
        # A plan can carry an explicit mutation contract even when the
        # semantic proposal omitted the redundant effect enum.  The accepted
        # metadata pass performs the durable, auditable reconciliation; use
        # the same mechanical projection while validating this plan so the
        # weak-model omission does not consume the single semantic repair.
        if changes and RequestedEffect.MUTATE_WORKSPACE not in effects:
            effects.add(RequestedEffect.MUTATE_WORKSPACE)
        if RequestedEffect.READ_WORKSPACE not in effects:
            raise ValueError(
                "repository-grounded semantics require requested_effects=read_workspace"
            )
        if changes and RequestedEffect.MUTATE_WORKSPACE not in effects:
            raise ValueError(
                "expected changes require requested_effects=mutate_workspace"
            )
        if not changes and RequestedEffect.MUTATE_WORKSPACE in effects:
            raise ValueError(
                "mutation semantics require evidence-backed expected change paths"
            )
        return semantic

    @staticmethod
    def _risk_adaptive_plan_approval(
        proposed: Mapping[str, Any],
        semantic: SemanticGoalV2,
        *,
        planning_only: bool,
    ) -> tuple[bool, str]:
        """Require one approval for every initial executable project plan."""

        del proposed, semantic
        return (
            True,
            "Plan mode never executes"
            if planning_only
            else "one explicit approval is required before autonomous execution",
        )

    def _review_plan_candidate(
        self,
        goal: Goal,
        candidate: dict[str, Any],
        inspection_records: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Any]:
        conversation: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": state_envelope(
                    {
                        "objective": goal.objective,
                        "proposed_plan": candidate,
                        "successful_workspace_inspections": list(inspection_records.values()),
                        "runtime_capabilities": tools.capability_report(),
                        "runtime_environment": self._runtime_environment_payload(),
                    },
                    "PLAN_REVIEW_INPUT",
                    max_chars=220_000,
                ),
            }
        ]
        for step in range(1, self.config.review_steps + 1):
            turn = self._call_provider(
                conversation,
                PLAN_REVIEWER_SCHEMAS,
                PLAN_REVIEWER_SYSTEM_PROMPT,
                actor="plan-critic",
                step=step,
            )
            conversation.append(turn.to_message())
            for call in turn.tool_calls:
                if call.name != "submit_plan_review":
                    result = f"Error: plan critic may only call submit_plan_review, not {call.name}"
                else:
                    try:
                        result_args = validate_control_call(call.name, call.args)
                        issues = [
                            dict(item)
                            for item in result_args.get("issues", ())
                            if isinstance(item, Mapping)
                        ]
                        if result_args.get("verdict") == "pass" and issues:
                            self.store.append_event(
                                "plan.critic_advisories",
                                goal_id=goal.id,
                                payload={
                                    "items": [
                                        str(item.get("detail") or "")
                                        for item in issues
                                        if str(item.get("detail") or "").strip()
                                    ],
                                    "structured_items": issues,
                                    "summary": result_args.get("summary", ""),
                                },
                            )
                            result_args = {
                                **result_args,
                                "issues": [],
                            }
                        elif result_args.get("verdict") == "revise":
                            blocking = [
                                item for item in issues if bool(item.get("blocking"))
                            ]
                            unclassified = [
                                item
                                for item in issues
                                if not bool(item.get("classified", False))
                            ]
                            if unclassified or not blocking:
                                raise ControlValidationError(
                                    "verdict=revise requires at least one explicitly "
                                    "classified blocking issue object; optional improvements "
                                    "must use verdict=pass with advisory issues"
                                )
                        conversation.append({"role": "tool", "id": call.id, "name": call.name, "content": "Review accepted by harness."})
                        return result_args
                    except ControlValidationError as exc:
                        result = f"Error: invalid plan review: {exc}"
                conversation.append({"role": "tool", "id": call.id, "name": call.name, "content": result})
            if not turn.tool_calls:
                conversation.append(
                    {
                        "role": "user",
                        "content": "A prose review is not a verdict. Call submit_plan_review now.",
                    }
                )
        return {
            "verdict": "contract_error",
            "summary": "The plan critic did not produce a valid structured verdict.",
            "issues": [],
            "contract_error": True,
        }

    def _validate_plan_applicability(
        self,
        proposed: Mapping[str, Any],
        tasks: Iterable[Task],
        *,
        successful_inspection_ids: frozenset[str],
        original_request: str,
    ) -> None:
        if not successful_inspection_ids:
            raise ValueError(
                "the planner must successfully inspect the workspace with a read-only tool before proposing"
            )
        semantic_scope = proposed.get("semantic_goal", {})
        semantic_scope_text = " ".join(
            str(value)
            for field in ("constraints", "exclusions")
            for value in (
                semantic_scope.get(field, ())
                if isinstance(semantic_scope, Mapping)
                else ()
            )
        ).casefold()
        exact_file_scope = any(
            marker in semantic_scope_text
            for marker in (
                "only create",
                "only modify",
                "do not create or modify any files outside",
                "no other files",
            )
        )
        explicit_request_paths = {
            match.group(0).replace("\\", "/").casefold()
            for match in re.finditer(
                r"[\w.-]+(?:[/\\][\w.-]+)*\."
                r"(?:py|html?|js|ts|tsx|jsx|css|json|md|txt|ya?ml|toml)",
                original_request,
                re.IGNORECASE,
            )
        }
        task_ids = {task.id for task in tasks}
        evidence_coverage: set[str] = set()
        for item in proposed["applicability_evidence"]:
            source = str(item.get("source") or "").strip()
            if source.startswith("inspection:"):
                source_id = source[len("inspection:") :]
            elif source.startswith("tool:"):  # legacy persisted/test plans
                source_id = source[len("tool:") :]
            else:
                source_id = ""
            if source_id not in successful_inspection_ids:
                raise ValueError(
                    f"applicability source {source!r} does not match a successful earlier "
                    "inspection; cite the stable inspection:I001-style reference"
                )
            supports = {str(task_id).upper() for task_id in item["supports_tasks"]}
            unknown = supports - task_ids
            if unknown:
                raise ValueError(
                    f"applicability evidence references unknown tasks: {', '.join(sorted(unknown))}"
                )
            evidence_coverage.update(supports)
        missing = task_ids - evidence_coverage
        if missing:
            raise ValueError(
                f"every task needs factual applicability evidence; missing: {', '.join(sorted(missing))}"
            )
        change_coverage: set[str] = set()
        for item in proposed["expected_changes"]:
            raw_path = str(item["path"]).strip().replace("\\", "/")
            path = raw_path.casefold()
            if (
                exact_file_scope
                and explicit_request_paths
                and path not in explicit_request_paths
            ):
                raise ValueError(
                    f"expected change {raw_path!r} violates the accepted exact "
                    "file scope; allowed paths are "
                    + ", ".join(sorted(explicit_request_paths))
                )
            if (
                not raw_path
                or raw_path.startswith("/")
                or re.match(r"^[a-zA-Z]:/", raw_path)
                or ".." in Path(raw_path).parts
                or raw_path in {".", "./"}
                or any(marker in path for marker in ("<", ">", "tbd", "unknown", "determine later"))
            ):
                raise ValueError("expected workspace changes must name real paths, not placeholders")
            if raw_path.endswith("/") or (
                (self.workspace / raw_path).resolve(strict=False).is_dir()
            ):
                raise ValueError(
                    "expected workspace changes must identify exact leased files, "
                    "not broad directories"
                )
            basis = str(item.get("basis") or "").strip()
            evidence_refs = {
                str(value).strip()
                for value in item.get("evidence_refs", ())
                if str(value).strip()
            }
            candidate = (self.workspace / raw_path).resolve(strict=False)
            exists = candidate.is_file()
            if exists and basis not in {
                "existing_inspected_path",
                "repository_convention",
            }:
                raise ValueError(
                    "an existing target must cite inspected-path or repository-convention basis"
                )
            if not exists and basis not in {
                "repository_convention",
                "model_selected_new_layout",
                "explicit_user_requirement",
            }:
                raise ValueError(
                    "a new target must cite repository_convention, "
                    "model_selected_new_layout, or explicit_user_requirement"
                )
            normalized_request = str(original_request).replace("\\", "/").casefold()
            request_path = raw_path.casefold().removeprefix("./")
            if basis == "explicit_user_requirement":
                if "user:request" not in evidence_refs:
                    raise ValueError(
                        "explicit-user path basis requires evidence_refs=user:request"
                    )
                if request_path not in normalized_request:
                    raise ValueError(
                        "explicit-user path basis requires the exact workspace-relative "
                        "path to appear in the original request"
                    )
            elif basis == "model_selected_new_layout":
                if exists:
                    raise ValueError(
                        "model-selected-new-layout basis is valid only for a new target"
                    )
                if request_path in normalized_request:
                    raise ValueError(
                        "a path written verbatim by the user must use "
                        "explicit_user_requirement basis"
                    )
                cited = {
                    value[len("inspection:") :]
                    for value in evidence_refs
                    if value.startswith("inspection:")
                }
                if not cited or not cited.issubset(successful_inspection_ids):
                    raise ValueError(
                        "model-selected new layout must cite the successful "
                        "new/empty-workspace inspection"
                    )
            else:
                cited = {
                    value[len("inspection:") :]
                    for value in evidence_refs
                    if value.startswith("inspection:")
                }
                if not cited or not cited.issubset(successful_inspection_ids):
                    raise ValueError(
                        "path basis must cite successful repository inspection evidence"
                    )
            supports = {str(task_id).upper() for task_id in item["supports_tasks"]}
            unknown = supports - task_ids
            if unknown:
                raise ValueError(
                    f"expected changes reference unknown tasks: {', '.join(sorted(unknown))}"
                )
            change_coverage.update(supports)
        # An evidence-backed analysis/test-only request may legitimately have
        # no mutations.  Empty change coverage is therefore meaningful rather
        # than a reason for the harness to invent an output file.

    @staticmethod
    def _bind_plan_inspection_sources(
        proposed: dict[str, Any],
        inspection_records: Mapping[str, Mapping[str, Any]],
        *,
        original_request: str = "",
        normalization_actions: list[str] | None = None,
    ) -> dict[str, Any]:
        """Canonicalize provider-specific/placeholder citations to harness refs."""

        records = dict(inspection_records)
        aliases: dict[str, str] = {}
        for reference, record in records.items():
            canonical = f"inspection:{reference}"
            aliases[canonical.casefold()] = canonical
            aliases[f"tool:{reference}".casefold()] = canonical
            call_id = str(record.get("call_id") or "").strip()
            if call_id:
                aliases[f"tool:{call_id}".casefold()] = canonical
        only_reference = next(iter(records), None) if len(records) == 1 else None
        repository_reference = next(
            (
                reference
                for reference, record in records.items()
                if str(record.get("tool") or "").strip() == "list_files"
                and str(
                    (record.get("arguments") or {}).get("path", ".")
                    if isinstance(record.get("arguments"), Mapping)
                    else "."
                ).strip().replace("\\", "/") in {"", ".", "./"}
            ),
            None,
        ) or next(
            (
                reference
                for reference, record in records.items()
                if str(record.get("tool") or "").strip() == "list_files"
            ),
            only_reference,
        )
        placeholder = re.compile(
            r"^(?:tool|inspection):(?:call(?:_id|_\d+)?|\d+)$",
            re.IGNORECASE,
        )
        bound = dict(proposed)
        evidence = [dict(item) for item in proposed.get("applicability_evidence", ())]
        task_ids = [
            str(item.get("id"))
            for item in proposed.get("tasks", ())
            if isinstance(item, Mapping) and str(item.get("id") or "").strip()
        ]
        if not evidence and repository_reference is not None and task_ids:
            record = records[repository_reference]
            evidence = [
                {
                    "fact": (
                        "The workspace was inspected before this plan was proposed. "
                        + str(record.get("result") or "")[:600]
                    ),
                    "source": f"inspection:{repository_reference}",
                    "supports_tasks": task_ids,
                }
            ]
        for item in evidence:
            source = str(item.get("source") or "").strip()
            canonical = aliases.get(source.casefold())
            if canonical is None and repository_reference is not None:
                record = records[repository_reference]
                tool_alias = f"tool:{record.get('tool', '')}".casefold()
                if (
                    not source
                    or placeholder.fullmatch(source)
                    or source.casefold() == tool_alias
                    or source.casefold() == "user:request"
                    or source.casefold().startswith("repo_convention:")
                ):
                    canonical = f"inspection:{repository_reference}"
                    if source.casefold() == "user:request" or (
                        source.casefold().startswith("repo_convention:")
                    ):
                        item["fact"] = (
                            "The workspace was inspected before this plan was proposed. "
                            + str(record.get("result") or "")[:600]
                        )
            if canonical is not None:
                item["source"] = canonical
        bound["applicability_evidence"] = evidence
        changes = [dict(item) for item in proposed.get("expected_changes", ())]
        normalized_request = original_request.replace("\\", "/").casefold()
        path_pattern = re.compile(
            r"(?<![\w./-])([A-Za-z0-9_.-]+\."
            r"(?:html?|py|js|ts|tsx|jsx|css|json|md|txt|ya?ml|toml))\b",
            re.IGNORECASE,
        )
        placeholder_paths = {
            "explicit_user_requirement",
            "explicit user requirement",
            "explicit_user",
            "user:request",
            "user_request",
            "model_selected_new_layout",
            "model selected new layout",
            "generated",
            "new_file",
            "new file",
        }
        task_by_id = {
            str(task.get("id") or ""): task
            for task in proposed.get("tasks", ())
            if isinstance(task, Mapping) and str(task.get("id") or "").strip()
        }

        def candidate_paths(item: Mapping[str, Any]) -> list[str]:
            supports = [
                str(value).strip()
                for value in item.get("supports_tasks", ())
                if str(value).strip()
            ]
            scoped = [task_by_id[value] for value in supports if value in task_by_id]
            if not scoped:
                scoped = [
                    task
                    for task in proposed.get("tasks", ())
                    if isinstance(task, Mapping)
                ]
            sources = [original_request]
            for task in scoped:
                sources.extend(
                    str(task.get(key) or "")
                    for key in (
                        "title",
                        "description",
                        "acceptance_criteria",
                        "verification",
                    )
                )
            paths: list[str] = []
            for source in sources:
                for match in path_pattern.findall(source):
                    normalized = match.replace("\\", "/")
                    if normalized.casefold() not in {
                        value.casefold() for value in paths
                    }:
                        paths.append(normalized)
            return paths

        for item in changes:
            raw_path = str(item.get("path") or "").replace("\\", "/")
            if raw_path.casefold().strip() in placeholder_paths:
                candidates = candidate_paths(item)
                if len(candidates) == 1:
                    item["path"] = candidates[0]
                    raw_path = candidates[0]
                    if normalization_actions is not None:
                        normalization_actions.append(
                            f"/expected_changes path placeholder reconciled to {raw_path}"
                        )
            inspected_path_refs: list[str] = []
            normalized_path = raw_path.casefold().removeprefix("./")
            for reference, record in records.items():
                arguments = record.get("arguments", {})
                argument_path = (
                    str(arguments.get("path") or "")
                    if isinstance(arguments, Mapping)
                    else ""
                ).replace("\\", "/").casefold().removeprefix("./")
                result_paths = {
                    line.strip().replace("\\", "/").casefold().removeprefix("./")
                    for line in str(record.get("result") or "").splitlines()
                    if line.strip()
                }
                if normalized_path and (
                    argument_path == normalized_path or normalized_path in result_paths
                ):
                    inspected_path_refs.append(f"inspection:{reference}")
            if inspected_path_refs:
                # A repair/revision plan may target files created by the active
                # accepted plan. Their current existence and the fresh inspection
                # are execution provenance, so rebind a stale provider basis
                # without changing the path, intent, or scope.
                item["basis"] = "existing_inspected_path"
                item["evidence_refs"] = inspected_path_refs
            request_path = raw_path.casefold().removeprefix("./")
            if (
                item.get("basis") == "explicit_user_requirement"
                and request_path
                and (
                    raw_path.casefold() in normalized_request
                    or request_path in normalized_request
                )
            ):
                # `user:request` is the canonical harness citation for a path
                # that the model has already classified as an explicit user
                # requirement.  Adding the citation does not choose a path or
                # infer product semantics; it binds the model-authored basis to
                # the exact, persisted source text.
                refs = [
                    str(value)
                    for value in item.get("evidence_refs", ())
                    if str(value).strip()
                ]
                if "user:request" not in refs:
                    refs.append("user:request")
                item["evidence_refs"] = refs
        if repository_reference is not None:
            for item in changes:
                # Basis is model-owned. The harness may bind an omitted citation
                # to a successful root inspection, but it never changes one
                # semantic basis into another.
                if item.get("basis") in {
                    "repository_convention",
                    "model_selected_new_layout",
                }:
                    item["evidence_refs"] = [
                        f"inspection:{repository_reference}"
                    ]
        bound["expected_changes"] = changes
        return bound

    def _pause_planning(
        self,
        goal: Goal,
        question: str,
        reason: str,
        *,
        provider_failure: bool = False,
    ) -> None:
        """Checkpoint a bounded/failed planning pass as an explicit user-visible pause."""
        current = self.store.get_goal(goal.id)
        if current.status not in {GoalStatus.DISCOVERING, GoalStatus.REVISING}:
            return
        attempt = int(current.metadata.get("goal_attempt", 0)) + 1
        consecutive = int(current.metadata.get("consecutive_retries", 0)) + 1
        retry_ms = self._goal_retry_delay_ms(consecutive)
        retryable = provider_failure and consecutive < self.config.provider_failure_limit
        waiting = (
            question
            if not provider_failure or retryable
            else (
                "Planning stopped after repeated provider failures. Check the selected "
                "model, credentials, network, or local service, then use /model or /resume."
            )
        )
        adaptation_updates: dict[str, Any] = {}
        if self.execution_class == "local" and not provider_failure:
            # A failed structured planning pass gets one smaller retry packet;
            # this changes only packet/context policy, never workflow mode or
            # the quality gates.
            raw_policy = current.metadata.get("local_adaptation_policy")
            policy = (
                dict(raw_policy)
                if isinstance(raw_policy, Mapping)
                else self.local_adaptation_policy()
            )
            try:
                context_budget = int(policy.get("context_budget_tokens") or 16_000)
            except (TypeError, ValueError):
                context_budget = 16_000
            policy.update(
                {
                    "packet_size": 1,
                    "context_budget_tokens": max(8_000, min(context_budget, 16_000)),
                    "abstraction_level": "atomic",
                    "quality_gates_unchanged": True,
                }
            )
            adaptation_updates["local_adaptation_policy"] = policy
        self.store.update_goal_metadata(
            goal.id,
            waiting_question=waiting,
            resume_status=current.status.value,
            goal_attempt=attempt,
            consecutive_retries=consecutive,
            retry_reason=reason,
            retry_after_ms=retry_ms if retryable else 0,
            auto_retryable=retryable,
            **adaptation_updates,
        )
        self.store.transition_goal(goal.id, GoalStatus.PAUSED, reason=reason)

    def _pause_for_plan_questions(
        self,
        goal: Goal,
        questions: Sequence[Mapping[str, Any]],
    ) -> None:
        """Persist a non-retryable planning interview checkpoint."""

        current = self.store.get_goal(goal.id)
        if current.status not in {GoalStatus.DISCOVERING, GoalStatus.REVISING}:
            raise RuntimeStateError("planning questions can only pause an active planning phase")
        values = [
            redact_data(normalize_question(item, index=index).to_dict())
            for index, item in enumerate(questions, 1)
        ]
        first = str(values[0].get("question", ""))
        self.store.update_goal_metadata(
            goal.id,
            plan_questions=values,
            plan_answers={},
            waiting_question=first,
            resume_status=current.status.value,
            retry_reason="",
            retry_after_ms=0,
            auto_retryable=False,
        )
        self.store.append_event(
            "plan.questions_requested",
            goal_id=goal.id,
            payload={"questions": values},
        )
        self.store.transition_goal(
            goal.id,
            GoalStatus.PAUSED,
            reason="planner needs non-discoverable user decisions",
        )
        self.events.publish(
            "questions",
            f"Planning needs {len(values)} decision(s). Use /questions and /answer ID VALUE.",
            questions=values,
        )

    def plan_questions(self) -> tuple[dict[str, Any], ...]:
        goal = self.active_goal()
        if goal is None:
            return ()
        answers = dict(goal.metadata.get("plan_answers", {}))
        return tuple(
            {**dict(item), "answer": answers.get(str(item.get("id")))}
            for item in goal.metadata.get("plan_questions", ())
            if isinstance(item, Mapping)
        )

    def answer_plan_question(self, question_id: str, value: str) -> Plan | None:
        goal = self.active_goal()
        if goal is None:
            raise RuntimeStateError("there is no active planning interview")
        questions = {
            str(item.get("id")): dict(item)
            for item in goal.metadata.get("plan_questions", ())
            if isinstance(item, Mapping)
        }
        question_id = str(question_id).strip()
        answer = redact_text(value, 2_000).strip()
        if question_id not in questions:
            raise RuntimeStateError(f"unknown planning question id: {question_id}")
        if not answer:
            raise ValueError("question answers must not be empty")
        item = questions[question_id]
        normalized_question = normalize_question(item)
        answer, _answer_source = answer_from_value(normalized_question, answer)
        values = {
            str(option.get("value", "")).strip()
            for option in item.get("options", ())
            if isinstance(option, Mapping)
        }
        if values and answer not in values and not bool(item.get("allow_freeform", True)):
            raise ValueError(
                f"answer must be one of: {', '.join(sorted(values))}"
            )
        answers = dict(goal.metadata.get("plan_answers", {}))
        answers[question_id] = answer
        unanswered = [key for key in questions if not str(answers.get(key, "")).strip()]
        waiting = str(questions[unanswered[0]].get("question", "")) if unanswered else ""
        self.store.update_goal_metadata(
            goal.id,
            plan_answers=answers,
            waiting_question=waiting,
        )
        self.store.append_event(
            "plan.question_answered",
            goal_id=goal.id,
            entity_type="question",
            entity_id=question_id,
            payload={"answer": answer},
        )
        if unanswered:
            self.events.publish(
                "questions",
                f"Saved {question_id}; {len(unanswered)} planning decision(s) remain.",
            )
            return None
        if goal.status != GoalStatus.PAUSED:
            raise RuntimeStateError("all answers are saved, but planning is not paused")
        desired = GoalStatus(goal.metadata.get("resume_status", GoalStatus.DISCOVERING.value))
        if desired not in {GoalStatus.DISCOVERING, GoalStatus.REVISING}:
            desired = GoalStatus.DISCOVERING
        self.store.transition_goal(
            goal.id,
            desired,
            reason="planning questions answered",
        )
        self.events.publish("phase", "Planning decisions saved; rebuilding the approval-bound plan.")
        return self.generate_plan("Use the durable user answers when finalizing this plan.")

    def _goal_retry_delay_ms(self, consecutive: int) -> int:
        exponent = min(max(0, consecutive - 1), 12)
        return min(
            self.config.goal_retry_max_ms,
            self.config.goal_retry_base_ms * (2**exponent),
        )

    def _schedule_goal_retry(self, goal: Goal, reason: str) -> Goal:
        attempt = int(goal.metadata.get("goal_attempt", 0)) + 1
        consecutive = int(goal.metadata.get("consecutive_retries", 0)) + 1
        retry_ms = self._goal_retry_delay_ms(consecutive)
        safe_reason = redact_text(reason, 1_000)
        updated = self.store.update_goal_metadata(
            goal.id,
            goal_attempt=attempt,
            consecutive_retries=consecutive,
            retry_reason=safe_reason,
            retry_after_ms=retry_ms,
            auto_retryable=True,
        )
        self.store.append_event(
            "goal.retry_scheduled",
            goal_id=goal.id,
            payload={"attempt": attempt, "delay_ms": retry_ms, "reason": safe_reason},
        )
        self._work_conversation.append(
            {
                "role": "user",
                "content": (
                    f"SELF-RETRY ATTEMPT {attempt}. The prior attempt did not advance the durable goal: "
                    f"{safe_reason}. Reassess the failed hypothesis and recent evidence. Choose a materially "
                    "different evidence-producing action, delegate a narrower role, or propose a revised "
                    "executable plan. Do not repeat the same action or answer only in prose."
                ),
            }
        )
        return updated

    def _schedule_provider_retry(self, goal: Goal, reason: str) -> tuple[Goal, bool]:
        """Retry transient provider failures, then stop at an actionable boundary."""

        updated = self._schedule_goal_retry(goal, reason)
        consecutive = int(updated.metadata.get("consecutive_retries", 0))
        if consecutive < self.config.provider_failure_limit:
            return updated, True
        waiting = (
            "Provider access failed repeatedly. Check the selected model, credentials, "
            "network, or local service, then use /model or /resume."
        )
        updated = self.store.update_goal_metadata(
            updated.id,
            retry_after_ms=0,
            auto_retryable=False,
            waiting_question=waiting,
            resume_status=GoalStatus.RUNNING.value,
        )
        if updated.status is GoalStatus.RUNNING:
            updated = self.store.transition_goal(
                updated.id,
                GoalStatus.PAUSED,
                reason="repeated provider failures require user action",
            )
        self.events.publish(
            "checkpoint",
            waiting,
            paused=True,
            retry_exhausted=True,
            attempts=consecutive,
        )
        return updated, False

    def wait_for_scheduled_retry(self) -> int:
        """Apply one bounded backoff delay for a retryable durable attempt."""
        goal = self.active_goal()
        if goal is None:
            return 0
        delay_ms = max(0, int(goal.metadata.get("retry_after_ms", 0)))
        if delay_ms:
            self.events.publish(
                "retry_wait",
                f"Retry {goal.metadata.get('goal_attempt', 0)} in {delay_ms / 1000:.1f}s",
                delay_ms=delay_ms,
                attempt=goal.metadata.get("goal_attempt", 0),
            )
            remaining = delay_ms / 1_000
            while remaining > 0:
                current = self.store.get_goal(goal.id)
                if (
                    not current.metadata.get("auto_retryable")
                    or int(current.metadata.get("retry_after_ms", 0) or 0) <= 0
                ):
                    break
                interval = min(0.25, remaining)
                self.sleeper(interval)
                remaining -= interval
            self.store.update_goal_metadata(goal.id, retry_after_ms=0)
        return delay_ms

    def generate_plan(
        self,
        feedback: str = "",
        *,
        auto_approve_in_scope_repair: bool = False,
    ) -> Plan | None:
        try:
            return self._generate_plan(
                feedback,
                auto_approve_in_scope_repair=auto_approve_in_scope_repair,
            )
        except ProviderUnavailableError as exc:
            goal = self.active_goal()
            if goal is not None:
                overloaded = "temporarily overloaded" in str(exc).casefold()
                self._pause_planning(
                    goal,
                    (
                        "The selected provider is temporarily overloaded. The planning checkpoint is saved; retry after the backoff or change model."
                        if overloaded
                        else "Planning provider retries were exhausted. Fix connectivity/rate limits, add guidance if useful, then use /resume."
                    ),
                    (
                        "planning provider temporarily overloaded"
                        if overloaded
                        else "planning provider unavailable after bounded retries"
                    ),
                    provider_failure=True,
                )
            # A planning call may be entered directly by the Web workspace
            # (before the terminal controller has a worker thread to finalize
            # it).  The provider watchdog claims the cooperative workflow
            # lease, so every provider boundary must release that lease at the
            # saved checkpoint.  Leaving it active makes Full Auto's recovery
            # look enabled while its local-model switch is rejected as a
            # second live worker.
            self._release_execution_lease(
                stage="planning-boundary",
                state="boundary",
            )
            raise

    def _generate_plan(
        self,
        feedback: str = "",
        *,
        auto_approve_in_scope_repair: bool = False,
    ) -> Plan | None:
        goal = self.active_goal()
        if goal is None:
            raise RuntimeStateError("no active goal")
        if goal.status not in {GoalStatus.DISCOVERING, GoalStatus.REVISING, GoalStatus.AWAITING_PLAN_APPROVAL}:
            raise RuntimeStateError(f"cannot generate a plan while goal is {goal.status.value}")
        if goal.status == GoalStatus.AWAITING_PLAN_APPROVAL:
            current = self.latest_plan()
            if current:
                self.store.reject_plan(goal.id, current.revision, feedback or "regenerate requested", rejected_by="user")
                goal = self.active_goal()

        previous_plan = self.store.get_latest_plan(goal.id)
        accepted_route_effects = self._accepted_route_effects(goal.objective)
        planning_questions = tuple(goal.metadata.get("plan_questions", ()))
        planning_answers = dict(goal.metadata.get("plan_answers", {}))
        semantic_intake_complete = bool(
            str(goal.metadata.get("task_demand_fingerprint") or "").strip()
        )
        planning_policy = goal.metadata.get("local_adaptation_policy")
        if not isinstance(planning_policy, Mapping):
            try:
                planning_policy = self.local_adaptation_policy(
                    ModelCapabilityEnvelopeV1.from_mapping(
                        goal.metadata.get("model_capability_envelope") or {}
                    )
                )
            except (TypeError, ValueError):
                planning_policy = self.local_adaptation_policy()
        conversation: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": state_envelope(
                    {
                        "objective": goal.objective,
                        "workspace": str(self.workspace),
                        "runtime_environment": self._runtime_environment_payload(),
                        "runtime_capabilities": tools.capability_report(),
                        "execution_policy": goal.metadata.get(
                            "execution_policy", {}
                        ),
                        "local_adaptation_policy": dict(planning_policy),
                        "discovered_verifier_plugins": goal.metadata.get(
                            "discovered_verifier_plugins", ()
                        ),
                        "user_feedback": feedback,
                        "planning_questions": list(planning_questions),
                        "planning_answers": planning_answers,
                        "semantic_intake_complete": semantic_intake_complete,
                        "open_consequential_decisions": list(
                            dict(goal.metadata.get("semantic_goal") or {}).get(
                                "unresolved_decisions", ()
                            )
                        ),
                        "previous_plan": None
                        if previous_plan is None
                        else {
                            "revision": previous_plan.revision,
                            "status": previous_plan.status.value,
                            "summary": previous_plan.summary,
                            "applicability_evidence": list(previous_plan.applicability_evidence),
                            "execution_strategy": previous_plan.execution_strategy,
                            "expected_changes": list(previous_plan.expected_changes),
                            "tasks": [_task_dict(task) for task in previous_plan.tasks],
                        },
                    },
                    "PLANNING_INPUT",
                    max_chars=220_000,
                ),
            }
        ]
        revisions = 0
        invalid_plan_calls = 0
        question_repairs = 0
        semantic_repairs = 0
        semantic_mapping_repairs = 0
        dag_repairs = 0
        applicability_repairs = 0
        critic_repairs = 0
        repair_limit = self._structured_repair_limit()
        critic_recovery_attempts = 0
        rejected_stage_fingerprints: dict[str, str] = {}
        unproductive_turns_after_inspection = 0
        last_plan_format_error = ""
        exhausted_stage = ""
        plan_format_exhausted = False
        accepted_plan_for_revision = self.store.get_accepted_plan(goal.id)
        semantic_locked_to_approval = accepted_plan_for_revision is not None
        if semantic_locked_to_approval and isinstance(
            goal.metadata.get("semantic_goal"),
            Mapping,
        ):
            stored_staged_semantic = dict(goal.metadata["semantic_goal"])
            stored_staged_semantic["status"] = "interpreted"
        else:
            stored_staged_semantic = goal.metadata.get("planning_semantic_goal")
        staged_semantic: dict[str, Any] | None = (
            dict(stored_staged_semantic)
            if isinstance(stored_staged_semantic, Mapping)
            else None
        )
        if (
            staged_semantic is not None
            and not semantic_locked_to_approval
            and accepted_route_effects
        ):
            existing_effects = tuple(staged_semantic.get("requested_effects") or ())
            merged_effects = tuple(
                dict.fromkeys(
                    (
                        *(RequestedEffect.parse(item) for item in existing_effects),
                        *accepted_route_effects,
                    )
                )
            )
            if tuple(RequestedEffect.parse(item) for item in existing_effects) != merged_effects:
                staged_semantic["requested_effects"] = [
                    item.value for item in merged_effects
                ]
                enriched = SemanticGoalV2.from_mapping(
                    staged_semantic,
                    original_request=goal.objective,
                )
                staged_semantic = enriched.to_dict()
                self.store.update_goal_metadata(
                    goal.id,
                    planning_semantic_goal=staged_semantic,
                    planning_semantic_fingerprint=enriched.fingerprint,
                    planning_semantic_status="accepted",
                )
                self.store.append_event(
                    "planning.semantic_enriched_from_route",
                    goal_id=goal.id,
                    payload={
                        "fingerprint": enriched.fingerprint,
                        "accepted_route_effects": [
                            item.value for item in accepted_route_effects
                        ],
                    },
                )
        semantic_stage_attempted = staged_semantic is not None
        plan_stage_prompted = False
        successful_inspection_ids: set[str] = set()
        inspection_records: dict[str, dict[str, Any]] = {}
        inspection_cache: dict[str, str] = {}
        if self.model_descriptor is not None and self.execution_class == "local":
            # Smaller local models frequently jump straight to the semantic
            # proposal even though the planner contract requires an inspection
            # first.  Seed one deterministic, read-only workspace fact so the
            # contract remains evidence-gated without asking the user to retry
            # a request that has not mutated anything.  Cloud models keep the
            # model-authored inspection path and its stricter provenance.
            preflight_call = ToolCall(
                id="preflight-" + uuid.uuid4().hex[:12],
                name="list_files",
                args={"path": "."},
            )
            self.events.publish(
                "tool_call",
                preflight_call.name,
                args=redact_data(preflight_call.args),
                actor="planner",
                phase="planning",
            )
            preflight_result = self._execute_workspace_tool(
                goal,
                preflight_call,
                task_id=None,
                actor="planner",
            )
            self.events.publish(
                "tool_result",
                preflight_result,
                tool=preflight_call.name,
                actor="planner",
                phase="planning",
            )
            if (
                not preflight_result.startswith("Error:")
                and not preflight_result.startswith("Permission denied")
            ):
                reference = "I001"
                inspection_cache['list_files:{"path": "."}'] = reference
                inspection_records[reference] = {
                    "reference": f"inspection:{reference}",
                    "call_id": preflight_call.id,
                    "tool": preflight_call.name,
                    "arguments": redact_data(preflight_call.args),
                    "result": redact_text(preflight_result, 4_000),
                    "source": "harness_preflight",
                }
                successful_inspection_ids.add(reference)
                self.store.append_event(
                    "planning.inspection_recorded",
                    goal_id=goal.id,
                    payload={
                        "reference": f"inspection:{reference}",
                        "call_id": preflight_call.id,
                        "tool": preflight_call.name,
                        "arguments": redact_data(preflight_call.args),
                        "source": "harness_preflight",
                    },
                )
        for step in range(1, self.config.planning_steps + 1):
            inspections_before_turn = frozenset(successful_inspection_ids)
            planner_tools = self._planner_tools()
            if semantic_intake_complete and not planning_questions:
                # The semantic gateway already had its dedicated opportunity
                # to ask consequential questions. Do not let planning reopen
                # ordinary implementation preferences and block a clear Goal.
                planner_tools = [
                    schema
                    for schema in planner_tools
                    if _tool_name(schema) != "request_plan_input"
                ]
            if (
                semantic_locked_to_approval
                and staged_semantic is not None
                and not inspection_records
            ):
                planner_tools = [
                    schema
                    for schema in planner_tools
                    if _tool_name(schema) in READ_ONLY_TOOLS
                ]
            if staged_semantic is not None and inspection_records:
                planner_tools = []
                for schema in self._planner_tools():
                    if _tool_name(schema) != "propose_plan":
                        continue
                    staged_schema = copy.deepcopy(schema)
                    parameters = staged_schema["function"]["parameters"]
                    parameters["properties"].pop("semantic_goal", None)
                    parameters["properties"]["applicability_evidence"] = {
                        "type": "array",
                        "description": (
                            "Repository facts with fact, source, and "
                            "supports_tasks fields."
                        ),
                        "items": {
                            "type": "object",
                            "additionalProperties": True,
                        },
                    }
                    parameters["properties"]["expected_changes"] = {
                        "type": "array",
                        "description": (
                            "Files or directories with path, intent, basis, "
                            "evidence_refs, and supports_tasks fields."
                        ),
                        "items": {
                            "type": "object",
                            "additionalProperties": True,
                        },
                    }
                    parameters["properties"]["tasks"] = {
                        "type": "array",
                        "minItems": 1,
                        "description": (
                            "Task objects using the exact keys title, "
                            "description, acceptance_criteria, verification, "
                            "depends_on, and risk."
                        ),
                        "items": {
                            "type": "object",
                            "additionalProperties": True,
                        },
                    }
                    planner_tools.append(staged_schema)
                if not plan_stage_prompted:
                    accepted_semantic = SemanticGoalV2.from_mapping(
                        staged_semantic,
                        original_request=goal.objective,
                    )
                    conversation = [{
                        "role": "user",
                        "content": state_envelope(
                            {
                                "objective": goal.objective,
                                "accepted_semantic_goal": staged_semantic,
                                "accepted_semantic_fingerprint": accepted_semantic.fingerprint,
                                "local_adaptation_policy": dict(planning_policy),
                                "successful_inspections": list(inspection_records.values()),
                                "planning_answers": planning_answers,
                                "previous_plan": None if previous_plan is None else {
                                    "revision": previous_plan.revision,
                                    "summary": previous_plan.summary,
                                    "tasks": [_task_dict(task) for task in previous_plan.tasks],
                                },
                                "required_next_action": (
                                    "Call propose_plan exactly once. The semantic stage is accepted; "
                                    "do not repeat it and do not answer with prose. "
                                    "Preserve the accepted objective: when requested_effects includes "
                                    "mutate_workspace, return at least one top-level expected_changes "
                                    "entry with a concrete workspace-relative path and a task whose "
                                    "acceptance and verification cover that mutation; do not submit a "
                                    "read-only-only plan or an empty expected_changes array."
                                ),
                            },
                            "PLAN_GENERATION_STAGE",
                            max_chars=120_000,
                        ),
                    }]
                    plan_stage_prompted = True
            turn = self._call_provider(
                conversation,
                planner_tools,
                PLANNER_SYSTEM_PROMPT,
                actor="planner",
                step=step,
            )
            if turn.tool_calls or str(turn.text or "").strip():
                conversation.append(turn.to_message())
            proposed: dict[str, Any] | None = None
            requested_questions: list[dict[str, Any]] | None = None
            for call in turn.tool_calls:
                self.events.publish("tool_call", call.name, args=redact_data(call.args), actor="planner")
                conversation_result: str | None = None
                if call.name == "propose_semantic_goal":
                    semantic_stage_attempted = True
                    try:
                        submitted_effects = {
                            str(item)
                            for item in (call.args.get("requested_effects") or ())
                        }
                        submitted_refs = {
                            str(item)
                            for item in (call.args.get("repository_evidence_refs") or ())
                            if str(item).strip()
                        }
                        validated_semantic = self._validate_semantic_stage(
                            goal,
                            call.args,
                            successful_inspection_ids=inspections_before_turn,
                            accepted_requested_effects=accepted_route_effects,
                        )
                        staged_semantic = validated_semantic.to_dict()
                        result = (
                            "Semantic interpretation accepted. Now call propose_plan "
                            f"with semantic_fingerprint={validated_semantic.fingerprint}."
                        )
                        self.store.update_goal_metadata(
                            goal.id,
                            planning_semantic_goal=staged_semantic,
                            planning_semantic_fingerprint=(
                                validated_semantic.fingerprint
                            ),
                            planning_semantic_status="accepted",
                        )
                        self.store.append_event(
                            "planning.semantic_accepted",
                            goal_id=goal.id,
                            payload={
                                "fingerprint": validated_semantic.fingerprint,
                                "observed_effects_added": (
                                    sorted(
                                        effect.value
                                        for effect in validated_semantic.requested_effects
                                        if effect.value not in submitted_effects
                                    )
                                ),
                                "observed_evidence_refs_added": sorted(
                                    set(validated_semantic.repository_evidence_refs)
                                    - submitted_refs
                                ),
                            },
                        )
                    except (ValueError, DomainError) as exc:
                        rejected_fingerprint = workflow_fingerprint(call.args)
                        duplicate_rejection = (
                            rejected_stage_fingerprints.get("semantic_interpretation")
                            == rejected_fingerprint
                        )
                        rejected_stage_fingerprints["semantic_interpretation"] = (
                            rejected_fingerprint
                        )
                        if not duplicate_rejection:
                            semantic_repairs += 1
                        last_plan_format_error = redact_text(exc, 1_000)
                        result = (
                            "Error: semantic interpretation rejected: "
                            f"{last_plan_format_error}. Repair only the semantic contract."
                        )
                        if not duplicate_rejection:
                            self.retry_ledger.record(
                                RetryKind.PLAN_SEMANTIC_REPAIR,
                                stage="semantic_interpretation",
                                reason=last_plan_format_error,
                                input_value=call.args,
                                next_action=(
                                    "targeted_semantic_repair"
                                    if semantic_repairs <= repair_limit
                                    else "stop"
                                ),
                            )
                        if duplicate_rejection:
                            result = (
                                "Error: the same invalid semantic payload was returned unchanged. "
                                "The saved stage is checkpointing without replaying routing or inspection."
                            )
                            exhausted_stage = "semantic_interpretation"
                            plan_format_exhausted = True
                        elif semantic_repairs > repair_limit:
                            exhausted_stage = "semantic_interpretation"
                            plan_format_exhausted = True
                elif call.name == "propose_plan":
                    try:
                        proposed, normalization_actions = normalize_plan_draft(call.args)
                        validate_normalized_plan(proposed)
                        if staged_semantic is not None:
                            accepted_fingerprint = SemanticGoalV2.from_mapping(
                                staged_semantic,
                                original_request=goal.objective,
                            ).fingerprint
                            requested_fingerprint = str(
                                proposed.get("semantic_fingerprint") or ""
                            ).strip()
                            if not requested_fingerprint:
                                if isinstance(
                                    proposed.get("semantic_goal"),
                                    Mapping,
                                ):
                                    # Backward compatibility for the original
                                    # combined proposal transport.  The accepted
                                    # semantic object still wins below, so a
                                    # legacy planner cannot drift approved scope.
                                    requested_fingerprint = accepted_fingerprint
                                    normalization_actions = (
                                        *normalization_actions,
                                        "legacy combined semantic replaced by "
                                        "the accepted semantic fingerprint",
                                    )
                                else:
                                    raise ValueError(
                                        "staged propose_plan requires the accepted "
                                        "semantic_fingerprint"
                                    )
                            if requested_fingerprint != accepted_fingerprint:
                                # The semantic fingerprint is a harness-owned
                                # binding, not product meaning authored by the
                                # model.  A planner can echo a stale, truncated,
                                # or otherwise malformed fingerprint after a
                                # repair turn even though its tasks are meant for
                                # the accepted semantic contract.  Rebind the
                                # proposal to the durable accepted fingerprint
                                # before validating tasks; never let transport
                                # identity alone create a workflow boundary.
                                normalization_actions = (
                                    *normalization_actions,
                                    "replaced model semantic_fingerprint with the "
                                    "harness-accepted semantic fingerprint",
                                )
                                proposed["semantic_fingerprint"] = accepted_fingerprint
                            proposed["semantic_goal"] = dict(staged_semantic)
                            proposed["_semantic_source"] = "staged"
                        elif proposed.get("semantic_goal"):
                            proposed["_semantic_source"] = "combined_proposal"
                        else:
                            # A plan cannot define product meaning on behalf of a
                            # missing semantic stage. Ask for that stage directly;
                            # this is a contract transition, not a plan-format
                            # failure, and therefore consumes no plan retry budget.
                            semantic_stage_attempted = True
                            proposed = None
                            result = (
                                "Error: no semantic contract is accepted yet. "
                                "Call propose_semantic_goal with a targeted repair; "
                                "this does not consume the plan-format budget."
                            )
                            conversation.append(
                                {
                                    "role": "tool",
                                    "id": call.id,
                                    "name": call.name,
                                    "content": result,
                                }
                            )
                            self.events.publish(
                                "tool_result",
                                result,
                                tool=call.name,
                                actor="planner",
                                phase="planning",
                            )
                            continue
                        for task in proposed["tasks"]:
                            task.pop("_unresolved_dependencies", None)
                        result = "Plan proposal normalized and captured for harness validation."
                        if normalization_actions:
                            self.store.append_event(
                                "planning.normalized",
                                goal_id=goal.id,
                                payload={"actions": list(normalization_actions)},
                            )
                    except (ControlValidationError, PlanDraftError, ValueError) as exc:
                        proposed = None
                        rejected_fingerprint = workflow_fingerprint(call.args)
                        duplicate_rejection = (
                            rejected_stage_fingerprints.get("plan_format")
                            == rejected_fingerprint
                        )
                        rejected_stage_fingerprints["plan_format"] = rejected_fingerprint
                        if not duplicate_rejection:
                            invalid_plan_calls += 1
                        last_plan_format_error = redact_text(exc, 1_000)
                        gross_format_failure = isinstance(exc, PlanDraftError) and all(
                            any(token in issue.path for token in ("/title", "/description", "/acceptance_criteria", "/verification"))
                            for issue in exc.issues
                        )
                        if not duplicate_rejection:
                            retry_record = self.retry_ledger.record(
                                RetryKind.PLAN_FORMAT_REPAIR,
                                stage=getattr(exc, "stage", "plan_normalization"),
                                reason=last_plan_format_error,
                                input_value=call.args,
                                next_action=(
                                    "targeted_repair"
                                    if invalid_plan_calls <= repair_limit
                                    else "stop"
                                ),
                            )
                            self.store.append_event(
                                "workflow.retry",
                                goal_id=goal.id,
                                entity_type="plan",
                                payload={
                                    "kind": retry_record.kind.value,
                                    "stage": retry_record.stage,
                                    "reason": retry_record.reason,
                                    "attempt": retry_record.attempt,
                                    "input_fingerprint": retry_record.input_fingerprint,
                                    "output_fingerprint": retry_record.output_fingerprint,
                                    "progress": retry_record.progress,
                                    "next_action": retry_record.next_action,
                                },
                            )
                        result = (
                            "Error: plan validation failed at exact field(s); submit one targeted "
                            f"repair using the same simplified contract: {last_plan_format_error}"
                        )
                        if duplicate_rejection:
                            result = (
                                "Error: the same invalid plan payload was returned unchanged. "
                                "Planning is checkpointing at plan_format without consuming "
                                "another independent repair attempt."
                            )
                            exhausted_stage = "plan_format"
                            plan_format_exhausted = True
                        elif invalid_plan_calls > repair_limit:
                            exhausted_stage = "plan_format"
                            plan_format_exhausted = True
                elif call.name == "request_plan_input":
                    try:
                        normalized = normalize_questions(
                            tuple(
                                item
                                for item in call.args.get("questions", ())
                                if isinstance(item, Mapping)
                            )
                        )
                        request = validate_control_call(
                            call.name,
                            {"questions": [item.to_dict() for item in normalized]},
                        )
                        if not inspections_before_turn:
                            raise ControlValidationError(
                                "inspect the workspace successfully before asking the user"
                            )
                        ids = [str(item["id"]) for item in request["questions"]]
                        if len(ids) != len(set(ids)):
                            raise ControlValidationError("question ids must be unique")
                        requested_questions = [dict(item) for item in request["questions"]]
                        result = "Question round captured; planning will checkpoint for the user."
                    except (ControlValidationError, ValueError) as exc:
                        rejected_fingerprint = workflow_fingerprint(call.args)
                        duplicate_rejection = (
                            rejected_stage_fingerprints.get("plan_questions")
                            == rejected_fingerprint
                        )
                        rejected_stage_fingerprints["plan_questions"] = rejected_fingerprint
                        if not duplicate_rejection:
                            question_repairs += 1
                        last_plan_format_error = redact_text(exc, 1_000)
                        if (
                            self.model_descriptor is not None
                            and self.execution_class == "local"
                            and "reason" in last_plan_format_error.casefold()
                        ):
                            # A weak local model can emit an optional planning
                            # question with a one-word rationale.  The question
                            # is not a user requirement and cannot justify a
                            # workflow boundary; discard only this malformed
                            # transport proposal and let the planner continue
                            # toward the already-inspected plan.
                            requested_questions = None
                            result = (
                                "Optional planning question omitted because its reason "
                                "was incomplete. Continue with one propose_plan call."
                            )
                            question_repairs = 0
                            rejected_stage_fingerprints.pop("plan_questions", None)
                            conversation.append(
                                {
                                    "role": "tool",
                                    "id": call.id,
                                    "name": call.name,
                                    "content": result,
                                }
                            )
                            self.events.publish(
                                "tool_result",
                                result,
                                tool=call.name,
                                actor="planner",
                                phase="planning",
                            )
                            continue
                        if not duplicate_rejection:
                            retry_record = self.retry_ledger.record(
                                RetryKind.PLAN_QUESTION_REPAIR,
                                stage="plan_questions",
                                reason=last_plan_format_error,
                                input_value=call.args,
                                next_action=(
                                    "targeted_repair" if question_repairs < 2 else "stop"
                                ),
                            )
                            self.store.append_event(
                                "workflow.retry",
                                goal_id=goal.id,
                                entity_type="plan",
                                payload={
                                    "kind": retry_record.kind.value,
                                    "stage": retry_record.stage,
                                    "reason": retry_record.reason,
                                    "attempt": retry_record.attempt,
                                    "input_fingerprint": retry_record.input_fingerprint,
                                    "output_fingerprint": retry_record.output_fingerprint,
                                    "progress": retry_record.progress,
                                    "next_action": retry_record.next_action,
                                },
                            )
                        result = (
                            "Error: invalid plan question request. Repair only the question "
                            f"shape or continue planning without a question: {last_plan_format_error}"
                        )
                        if duplicate_rejection:
                            result = (
                                "Error: the same invalid question payload was returned unchanged. "
                                "Planning is checkpointing at plan_questions without consuming "
                                "another question-shape repair."
                            )
                            exhausted_stage = "plan_questions"
                            plan_format_exhausted = True
                        elif question_repairs >= 2:
                            exhausted_stage = "plan_questions"
                            plan_format_exhausted = True
                elif call.name in READ_ONLY_TOOLS:
                    normalized_args = dict(call.args)
                    if call.name == "list_files" and not str(normalized_args.get("path") or "").strip():
                        normalized_args["path"] = "."
                    inspection_key = f"{call.name}:{json.dumps(normalized_args, ensure_ascii=False, sort_keys=True, default=str)}"
                    existing_reference = inspection_cache.get(inspection_key)
                    if existing_reference is not None:
                        record = inspection_records[existing_reference]
                        result = str(record["result"])
                        reference = existing_reference
                    else:
                        call.args = normalized_args
                        result = self._execute_workspace_tool(goal, call, task_id=None, actor="planner")
                        reference = ""
                    if not result.startswith("Error:") and not result.startswith("Permission denied"):
                        if not reference:
                            reference = f"I{len(inspection_records) + 1:03d}"
                            inspection_cache[inspection_key] = reference
                            inspection_records[reference] = {
                                "reference": f"inspection:{reference}",
                                "call_id": call.id,
                                "tool": call.name,
                                "arguments": redact_data(normalized_args),
                                "result": redact_text(result, 4_000),
                            }
                            self.store.append_event(
                                "planning.inspection_recorded",
                                goal_id=goal.id,
                                payload={
                                    "reference": f"inspection:{reference}",
                                    "call_id": call.id,
                                    "tool": call.name,
                                    "arguments": redact_data(normalized_args),
                                    "result": redact_text(result, 4_000),
                                },
                            )
                        successful_inspection_ids.add(reference)
                        if goal.metadata.get("strategy_reinspection_required"):
                            self.store.update_goal_metadata(
                                goal.id,
                                strategy_reinspection_required=False,
                            )
                        conversation_result = (
                            f"Stable inspection reference: inspection:{reference}. "
                            "Use this exact source in applicability_evidence.\n"
                            f"{result}"
                        )
                else:
                    result = f"Error: planning is read-only; tool '{call.name}' is unavailable before approval."
                conversation.append(
                    {
                        "role": "tool",
                        "id": call.id,
                        "name": call.name,
                        "content": conversation_result or result,
                    }
                )
                self.events.publish(
                    "tool_result",
                    result,
                    tool=call.name,
                    actor="planner",
                    phase="planning",
                )
                if plan_format_exhausted:
                    # A single model turn may contain several proposals.  The
                    # retry budget is global to this planning run, so stop at
                    # the third rejected proposal instead of processing an
                    # arbitrary remainder from the same response.
                    break

            if plan_format_exhausted:
                break

            if requested_questions is not None and proposed is None:
                self._pause_for_plan_questions(goal, requested_questions)
                return None

            if proposed is None and requested_questions is None and inspection_records:
                unproductive_turns_after_inspection += 1
                if staged_semantic is not None:
                    fingerprint = SemanticGoalV2.from_mapping(
                        staged_semantic,
                        original_request=goal.objective,
                    ).fingerprint
                    conversation.append(
                        {
                            "role": "user",
                            "content": (
                                "The semantic stage is already accepted. Do not "
                                "repeat it and do not answer with prose. Make exactly "
                                "one propose_plan tool call now, using "
                                f"semantic_fingerprint={fingerprint}. Include tasks, "
                                "dependencies, expected_changes, execution_strategy, "
                                "and applicability_evidence. Every task object MUST "
                                "use these exact JSON keys: title, description, "
                                "acceptance_criteria, verification, depends_on, risk."
                            ),
                        }
                    )

            if proposed is not None:
                if planning_answers:
                    proposed = dict(proposed)
                    proposed["execution_strategy"] = (
                        str(proposed["execution_strategy"]).rstrip()
                        + "\n\nApproval-bound user planning decisions: "
                        + json.dumps(
                            planning_answers,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                    )
                normalization_actions = list(normalization_actions)
                proposed = self._bind_plan_inspection_sources(
                    proposed,
                    inspection_records,
                    original_request=goal.objective,
                    normalization_actions=normalization_actions,
                )
                normalization_actions = tuple(dict.fromkeys(normalization_actions))
                try:
                    if len(json.dumps(proposed, ensure_ascii=False, default=str)) > 120_000:
                        raise ValueError(
                            "aggregate plan exceeds 120,000 characters; use concise tasks and evolve later revisions"
                        )
                    next_revision = (self.store.get_latest_plan(goal.id).revision + 1) if self.store.get_latest_plan(goal.id) else 1
                    preview = tuple(
                        self.store.coerce_task(item, goal.id, next_revision, "agent")
                        for item in proposed["tasks"]
                    )
                    validate_task_dag(preview)
                except (ValueError, DomainError) as exc:
                    rejected_fingerprint = workflow_fingerprint(proposed)
                    duplicate_rejection = (
                        rejected_stage_fingerprints.get("task_dag")
                        == rejected_fingerprint
                    )
                    rejected_stage_fingerprints["task_dag"] = rejected_fingerprint
                    if not duplicate_rejection:
                        dag_repairs += 1
                    last_plan_format_error = redact_text(exc, 1_000)
                    conversation.append(
                        {
                            "role": "user",
                            "content": (
                                "Harness DAG validation rejected the plan: "
                                f"{last_plan_format_error}. Repair only task IDs "
                                "and dependencies, then resubmit the complete plan."
                            ),
                        }
                    )
                    proposed = None
                    if duplicate_rejection or dag_repairs > repair_limit:
                        exhausted_stage = "task_dag"
                        break
                    continue
                try:
                    self._validate_semantic_candidate(
                        goal,
                        proposed,
                        successful_inspection_ids=inspections_before_turn,
                    )
                except (ValueError, DomainError) as exc:
                    rejected_fingerprint = workflow_fingerprint(proposed)
                    duplicate_rejection = (
                        rejected_stage_fingerprints.get("semantic_mapping")
                        == rejected_fingerprint
                    )
                    rejected_stage_fingerprints["semantic_mapping"] = rejected_fingerprint
                    if not duplicate_rejection:
                        semantic_mapping_repairs += 1
                    last_plan_format_error = redact_text(exc, 1_000)
                    conversation.append(
                        {
                            "role": "user",
                            "content": (
                                "Harness semantic mapping rejected the plan: "
                                f"{last_plan_format_error}. Repair only the semantic "
                                "contract or its criterion mapping. The accepted semantic includes "
                                "mutate_workspace, so the next complete propose_plan MUST include "
                                "at least one top-level expected_changes object with a concrete "
                                "workspace-relative path, and at least one task that creates or "
                                "updates that path with matching acceptance and verification; do "
                                "not resubmit a read-only-only task or empty expected_changes."
                            ),
                        }
                    )
                    proposed = None
                    if duplicate_rejection or semantic_mapping_repairs > repair_limit:
                        exhausted_stage = "semantic_mapping"
                        break
                    continue
                try:
                    self._validate_plan_applicability(
                        proposed,
                        preview,
                        successful_inspection_ids=inspections_before_turn,
                        original_request=goal.objective,
                    )
                except (ValueError, DomainError) as exc:
                    rejected_fingerprint = workflow_fingerprint(proposed)
                    duplicate_rejection = (
                        rejected_stage_fingerprints.get("applicability")
                        == rejected_fingerprint
                    )
                    rejected_stage_fingerprints["applicability"] = rejected_fingerprint
                    if not duplicate_rejection:
                        applicability_repairs += 1
                    last_plan_format_error = redact_text(exc, 1_000)
                    conversation.append(
                        {
                            "role": "user",
                            "content": (
                                "Harness plan validation rejected the proposal: "
                                f"{last_plan_format_error}. Repair every listed ID, dependency, "
                                "evidence citation, and criterion in one complete proposal. "
                                "For a new path selected after inspecting a new/empty workspace, "
                                "use basis=model_selected_new_layout with that inspection ref; "
                                "use explicit_user_requirement only when the exact path is "
                                "verbatim in the original request."
                            ),
                        }
                    )
                    proposed = None
                    if duplicate_rejection or applicability_repairs > repair_limit:
                        exhausted_stage = "applicability"
                        break
                    continue
                critique = (
                    self._review_plan_candidate(goal, proposed, inspection_records)
                    if self._plan_requires_critic(proposed)
                    else {
                        "verdict": "pass",
                        "summary": (
                            "Simple low-risk legacy plan passed deterministic "
                            "schema, evidence, DAG, and semantic projection checks."
                        ),
                        "issues": [],
                    }
                )
                if critique.get("contract_error"):
                    last_plan_format_error = redact_text(
                        str(critique.get("summary") or "plan critic contract failed"),
                        1_000,
                    )
                    exhausted_stage = "critic_contract"
                    break
                if critique["verdict"] == "pass" and not critique["issues"]:
                    semantic_metadata = self._accepted_semantic_metadata(
                        goal, proposed, inspection_records
                    )
                    accepted_semantic = SemanticGoalV2.from_mapping(
                        semantic_metadata["semantic_goal"],
                        original_request=goal.objective,
                    )
                    current_session = self.store.get_workflow_session(
                        self.session_id
                    )
                    manual_approval, approval_reason = (
                        self._risk_adaptive_plan_approval(
                            proposed,
                            accepted_semantic,
                            planning_only=(
                                SessionMode.parse(
                                    current_session["session_mode"]
                                )
                                is SessionMode.PLAN
                            ),
                        )
                    )
                    plan = self.store.create_plan(
                        goal.id,
                        proposed["summary"],
                        proposed["tasks"],
                        applicability_evidence=proposed["applicability_evidence"],
                        execution_strategy=proposed["execution_strategy"],
                        expected_changes=proposed["expected_changes"],
                        proposed_by="agent",
                        submit=True,
                    )
                    if (
                        accepted_plan_for_revision is not None
                        and auto_approve_in_scope_repair
                    ):
                        in_scope_repair = self._repair_revision_is_in_scope(
                            accepted_plan_for_revision,
                            plan,
                            proposed["tasks"],
                        )
                        manual_approval = not in_scope_repair
                        approval_reason = (
                            "harness-approved in-scope repair revision"
                            if in_scope_repair
                            else (
                                "repair revision expands paths or introduces "
                                "a dependency, external effect, or sensitive action"
                            )
                        )
                    current_goal = self.store.get_goal(goal.id)
                    if current_goal.status != GoalStatus.AWAITING_PLAN_APPROVAL:
                        self.store.transition_goal(
                            goal.id,
                            GoalStatus.AWAITING_PLAN_APPROVAL,
                            reason="plan passed independent critique and awaits user approval",
                        )
                    self.store.append_event(
                        "plan.critic_passed",
                        goal_id=goal.id,
                        entity_type="plan",
                        entity_id=plan.id,
                        payload={"summary": critique["summary"]},
                    )
                    self.events.publish(
                        "plan",
                        (
                            f"Plan r{plan.revision} is ready. Review, edit, and "
                            "apply it in Plan Studio."
                            if manual_approval
                            else (
                                f"In-scope repair plan r{plan.revision} passed "
                                "review and will continue automatically."
                            )
                        ),
                    )
                    self.store.update_goal_metadata(
                        goal.id,
                        **semantic_metadata,
                        plan_approval_policy={
                            "requires_user": manual_approval,
                            "reason": approval_reason,
                            "risk_adaptive": True,
                        },
                        consecutive_retries=0,
                        retry_reason="",
                        retry_after_ms=0,
                        auto_retryable=False,
                        plan_questions=[],
                        waiting_question="",
                        strategy_attempts=[],
                        failed_attempts=[],
                        workflow_stage_checkpoint={},
                        boundary_kind="",
                    )
                    def reduce_plan_session(current: dict[str, Any]) -> Mapping[str, Any]:
                        plan_session_state = {
                            **dict(current.get("state") or {}),
                            "plan_revision": plan.revision,
                            "plan_fingerprint": plan.fingerprint,
                        }
                        return {
                            # Preserve a newer binding made by another worker;
                            # only fill the id if the row is still unbound.
                            "goal_id": current.get("goal_id") or goal.id,
                            "session_mode": str(
                                current.get("session_mode") or SessionMode.NORMAL.value
                            ),
                            "plan_state": PlanState.AWAITING_APPROVAL.value,
                            "run_state": RunState.PLANNING.value,
                            "state": plan_session_state,
                            "ultra_profile": str(current.get("ultra_profile") or "standard"),
                            "sleep_state": str(current.get("sleep_state") or "off"),
                        }

                    self.store.mutate_workflow_session(
                        self.session_id,
                        reduce_plan_session,
                    )
                    if not manual_approval:
                        return self.approve_plan(
                            plan.revision,
                            approved_by="risk-adaptive-policy",
                        )
                    return plan
                revisions += 1
                critic_repairs += 1
                last_plan_format_error = redact_text(
                    str(critique.get("summary") or "independent critic rejected the plan"),
                    1_000,
                )
                conversation.append(
                    {
                        "role": "user",
                        "content": state_envelope(
                            {
                                "critic_verdict": critique,
                                "instruction": "Repair every issue and call propose_plan with the complete revised plan.",
                            },
                            "PLAN_CRITIQUE",
                        ),
                    }
                )
                # The initial candidate is not itself a repair. After three
                # rejected candidates, give the planner one fresh-context
                # quality-recovery pass before exposing a user boundary.  A
                # long conversation containing several malformed candidates
                # often makes a capable provider repeat the same omission;
                # the recovery packet preserves the exact objective, accepted
                # semantic contract, latest candidate, and critic findings
                # without replaying inspection or mutating the workspace.
                if critic_repairs >= repair_limit + 1:
                    if critic_recovery_attempts < 1:
                        critic_recovery_attempts += 1
                        self.store.append_event(
                            "planning.critic_recovery",
                            goal_id=goal.id,
                            payload={
                                "attempt": critic_recovery_attempts,
                                "rejected_candidates": critic_repairs,
                                "summary": last_plan_format_error,
                            },
                        )
                        accepted_semantic = (
                            SemanticGoalV2.from_mapping(
                                staged_semantic,
                                original_request=goal.objective,
                            )
                            if isinstance(staged_semantic, Mapping)
                            else None
                        )
                        conversation = [{
                            "role": "user",
                            "content": state_envelope(
                                {
                                    "objective": goal.objective,
                                    "accepted_semantic_goal": staged_semantic,
                                    "accepted_semantic_fingerprint": (
                                        accepted_semantic.fingerprint
                                        if accepted_semantic is not None
                                        else ""
                                    ),
                                    "successful_inspections": list(inspection_records.values()),
                                    "last_rejected_plan": proposed,
                                    "critic_findings": critique,
                                    "required_next_action": (
                                        "Produce exactly one complete propose_plan tool call. "
                                        "Address every blocking critic finding explicitly, "
                                        "preserve all accepted requirements, use only the "
                                        "listed runtime capabilities, and do not repeat the "
                                        "rejected plan unchanged."
                                    ),
                                },
                                "PLAN_QUALITY_RECOVERY",
                                max_chars=140_000,
                            ),
                        }]
                        # A fresh packet is intentionally not treated as a new
                        # semantic or inspection stage. Keep the accepted
                        # fingerprint and durable evidence unchanged.
                        continue
                    if critic_repairs >= (repair_limit + 1) * 2:
                        exhausted_stage = "independent_critic"
                        break
            elif not turn.tool_calls:
                conversation.append(
                    {
                        "role": "user",
                        "content": "Planning is not complete in prose. Inspect if needed, then call propose_plan with a validated plan.",
                    }
                )

        boundary_kind = (
            WorkflowBoundaryKind.SEMANTIC_CONFLICT
            if exhausted_stage in {"semantic_interpretation", "semantic_mapping"}
            else WorkflowBoundaryKind.QUALITY_BLOCKER
            if exhausted_stage == "independent_critic"
            else WorkflowBoundaryKind.CONTRACT_INCOMPATIBILITY
            if exhausted_stage
            else WorkflowBoundaryKind.NO_PROGRESS
        )
        checkpoint_reason = (
            f"{boundary_kind.value}: {exhausted_stage or 'planning'} did not converge"
        )
        stage_attempts = {
            "plan_questions": question_repairs,
            "plan_format": invalid_plan_calls,
            "semantic_interpretation": semantic_repairs,
            "semantic_mapping": semantic_mapping_repairs,
            "task_dag": dag_repairs,
            "applicability": applicability_repairs,
            "independent_critic": critic_repairs,
        }.get(exhausted_stage, 0)
        stage_checkpoint = WorkflowStageCheckpointV1(
            stage=exhausted_stage or "planning",
            substage=(
                "question_transport"
                if exhausted_stage == "plan_questions"
                else exhausted_stage or "planner"
            ),
            category=boundary_kind.value,
            message=last_plan_format_error,
            attempts=stage_attempts,
            rejected_fingerprint=(
                rejected_stage_fingerprints.get(exhausted_stage)
                or (
                    self.retry_ledger.records[-1].input_fingerprint
                    if self.retry_ledger.records
                    else ""
                )
            ),
            semantic_fingerprint=str(
                goal.metadata.get("planning_semantic_fingerprint") or ""
            ),
            inspection_refs=tuple(
                f"inspection:{reference}" for reference in inspection_records
            ),
        )
        self.store.update_goal_metadata(
            goal.id,
            workflow_stage_checkpoint=stage_checkpoint.to_dict(),
            boundary_kind=boundary_kind.value,
        )
        self.store.append_event(
            "planning.checkpoint",
            goal_id=goal.id,
            payload={
                "reason": checkpoint_reason,
                "format_attempts": invalid_plan_calls,
                "question_attempts": question_repairs,
                "semantic_attempts": semantic_repairs,
                "semantic_mapping_attempts": semantic_mapping_repairs,
                "dag_attempts": dag_repairs,
                "applicability_attempts": applicability_repairs,
                "critic_attempts": critic_repairs,
                "failed_stage": exhausted_stage,
                "technical_detail": last_plan_format_error,
                "checkpoint_type": boundary_kind.value,
                "stage_checkpoint": stage_checkpoint.to_dict(),
                "resumable": True,
            },
        )
        if exhausted_stage:
            boundary_detail = (
                "The model response could not be reconciled with this stage's "
                "typed contract. This is not evidence that the selected model is weak."
                if boundary_kind is WorkflowBoundaryKind.CONTRACT_INCOMPATIBILITY
                else "The saved stage did not converge within its independent repair budget."
            )
            self.events.publish(
                "error",
                (
                    "Plan could not be prepared within the independent repair budget. "
                    f"Failed stage: {exhausted_stage}. {boundary_detail} "
                    f"Detail: {last_plan_format_error}. No workspace changes were made. "
                    "Retry the saved stage or use /replan."
                ),
                technical_detail=last_plan_format_error,
                attempts=max(invalid_plan_calls, question_repairs),
                planning_terminal=True,
            )
        else:
            self.events.publish(
                "warning",
                "Planning stopped before a valid plan was produced. Inspect the planning checkpoint for the exact stage and use /replan with guidance.",
                planning_terminal=True,
            )
        if exhausted_stage == "semantic_interpretation":
            pause_question = (
                f"Planning paused at {exhausted_stage}: the model's semantic "
                "tool output did not satisfy the required contract after the "
                f"repair budget. Detail: {last_plan_format_error}. "
                "No workspace changes were made. Retry the saved stage or use /replan; "
                "changing the model is optional."
            )
        elif exhausted_stage:
            pause_question = (
                f"Planning paused at {exhausted_stage}: {last_plan_format_error}. "
                "No workspace changes were made. Retry the saved stage or use /replan."
            )
        else:
            pause_question = (
                "The planner did not produce a critic-approved structured plan in its "
                "bounded pass. Add guidance, then use /resume or /replan."
            )
        self._pause_planning(goal, pause_question, checkpoint_reason)
        return None

    def _lock_strategy_after_approval(self, goal: Goal, plan: Plan) -> None:
        current = self.store.get_goal(goal.id)
        approved_capability = self.model_capability_envelope()
        raw = current.metadata.get("strategy_decision")
        if isinstance(raw, Mapping):
            decision = StrategyDecisionV1.from_mapping(raw).lock()
        else:
            legacy = dict(current.metadata.get("execution_policy") or {}).get("mode")
            strategy = ExecutionStrategyV1.parse(legacy)
            capability = self.model_capability_envelope()
            demand_raw = current.metadata.get("task_demand")
            demand = (
                TaskDemandV1.from_mapping(demand_raw)
                if isinstance(demand_raw, Mapping)
                else TaskDemandV1.from_legacy(
                    component_count=max(1, len(plan.tasks)),
                    parallelism_required=strategy is ExecutionStrategyV1.RECURSIVE,
                    reasons=("legacy accepted plan",),
                )
            )
            decision = select_execution_strategy(
                capability,
                demand,
                minimum=strategy,
                allow_capability_escalation=(
                    strategy is ExecutionStrategyV1.RECURSIVE
                ),
            ).lock()
        self.store.update_goal_metadata(
            goal.id,
            interaction_mode=InteractionModeV2.WORKING.value,
            execution_strategy=decision.strategy.value,
            strategy_decision=decision.to_dict(),
            strategy_fingerprint=decision.fingerprint,
            strategy_locked=True,
            approved_plan_fingerprint=plan.fingerprint,
            approved_capability_fingerprint=decision.capability_fingerprint,
            approved_model_capability_envelope=approved_capability.to_dict(),
            approved_scope_paths=sorted(self._plan_change_paths(plan)),
            evidence_bound_scope_expansion=False,
            in_scope_quality_revision_attempts=0,
            scope_contraction_attempts=0,
        )
        session = self.store.get_workflow_session(self.session_id)
        state = dict(session.get("state") or {})
        state.update({
            "interaction_mode": InteractionModeV2.WORKING.value,
            "strategy_decision": decision.to_dict(),
            "strategy_fingerprint": decision.fingerprint,
            "strategy_locked": True,
            "plan_revision": plan.revision,
            "plan_fingerprint": plan.fingerprint,
        })
        self.store.mutate_workflow_session(
            self.session_id,
            lambda current_state: {
                "state": state,
                "goal_id": current_state.get("goal_id") or goal.id,
                "session_mode": (
                    SessionMode.ULTRA.value
                    if decision.strategy is ExecutionStrategyV1.RECURSIVE
                    else SessionMode.NORMAL.value
                ),
                "plan_state": PlanState.APPROVED.value,
                "run_state": RunState.EXECUTING.value,
            },
            expected_revision=int(session.get("revision") or 0),
        )
        self.store.append_event(
            "execution_strategy.locked",
            goal_id=goal.id,
            entity_type="plan",
            entity_id=plan.id,
            payload={
                "strategy": decision.strategy.value,
                "strategy_fingerprint": decision.fingerprint,
                "capability_fingerprint": decision.capability_fingerprint,
                "plan_fingerprint": plan.fingerprint,
            },
        )

    def approve_plan(self, revision: int | None = None, *, approved_by: str = "user") -> Plan:
        goal = self.active_goal()
        plan = self.latest_plan()
        if goal is None or plan is None:
            raise RuntimeStateError("there is no plan to approve")
        session_mode = SessionMode.parse(
            self.store.get_workflow_session(self.session_id)["session_mode"]
        )
        if goal.metadata.get("ultra_run_id"):
            accepted_ultra = self.approve_ultra(
                revision,
                approved_by=approved_by,
            )
            self._lock_strategy_after_approval(goal, accepted_ultra)
            return accepted_ultra
        if goal.metadata.get("legacy_semantic_enrichment_required"):
            self.store.update_goal_metadata(
                goal.id,
                legacy_semantic_enrichment_required=False,
            )
            enriched = self.generate_plan(
                "Enrich this legacy pending plan with a repository-grounded "
                "semantic contract and produce a fresh reviewed revision. "
                "Preserve its original approved scope and exact user request."
            )
            if enriched is None:
                raise RuntimeStateError(
                    "legacy semantic enrichment checkpointed before a fresh "
                    "reviewable revision was produced"
                )
            return enriched
        requested = plan.revision if revision is None else revision
        accepted, _approval = self.store.approve_plan(
            goal.id,
            requested,
            approved_by=approved_by,
            expected_fingerprint=plan.fingerprint if requested == plan.revision else None,
        )
        self._lock_strategy_after_approval(goal, accepted)
        self.store.update_goal_metadata(
            goal.id,
            waiting_question="",
            retry_reason="",
            waiting_on="",
            resume_action="",
        )
        current = self.store.get_goal(goal.id)
        contract_data = current.metadata.get("goal_contract")
        semantic_refs = tuple(
            str(value)
            for value in dict(current.metadata.get("semantic_goal", {})).get(
                "repository_evidence_refs", ()
            )
            if str(value).strip()
        )
        applicability_refs_by_task: dict[str, list[str]] = {}
        for basis in accepted.applicability_evidence:
            if not isinstance(basis, Mapping):
                continue
            source = str(basis.get("source") or "").strip()
            if not source:
                continue
            for supported_task in basis.get("supports_tasks", ()):
                task_key = str(supported_task).strip().upper()
                if task_key:
                    applicability_refs_by_task.setdefault(task_key, []).append(source)
        resource_claims = []
        for change in accepted.expected_changes:
            if not isinstance(change, Mapping):
                continue
            path = str(change.get("path") or "").strip()
            supports = tuple(
                str(value).strip().upper()
                for value in change.get("supports_tasks", ())
                if str(value).strip()
            )
            refs = tuple(
                str(value).strip()
                for value in change.get("evidence_refs", ())
                if str(value).strip()
            )
            if not refs:
                refs = tuple(
                    dict.fromkeys(
                        (
                            *semantic_refs,
                            *(
                                ref
                                for task_key in supports
                                for ref in applicability_refs_by_task.get(task_key, ())
                            ),
                        )
                    )
                )
            if not path or path.startswith("<") or not supports or not refs:
                continue
            resource_claims.append(
                ResourceClaimV1(
                    purpose=str(change.get("intent") or f"Apply {path}"),
                    kind="file",
                    supports_tasks=supports,
                    inspection_refs=refs,
                    selector=path,
                    resolved_paths=(path,),
                    state="resolved",
                ).to_dict()
            )
        # Resource ownership is approval-bound plan state, not an optional
        # semantic-enrichment feature. Compatibility plans therefore receive
        # the same task-scoped claims as newly generated plans.
        self.store.update_goal_metadata(goal.id, resource_claims=resource_claims)
        if isinstance(contract_data, Mapping):
            contract = GoalContractV1.from_dict(contract_data)
            criteria = tuple(dict.fromkeys(
                criterion for task in accepted.tasks for criterion in task.acceptance_criteria
            ))
            verification = tuple(dict.fromkeys(
                check for task in accepted.tasks for check in task.verification
            ))
            artifact_paths = tuple(dict.fromkeys(
                str(change.get("path") or change.get("artifact") or "").strip()
                for change in accepted.expected_changes if isinstance(change, Mapping)
            ))
            artifact_paths = tuple(dict.fromkeys((*contract.artifact_expectations, *(path for path in artifact_paths if path))))
            declared_risks = {
                str(task.risk or "medium").casefold() for task in accepted.tasks
            }
            standard_dimensions = [
                ("requirement-completeness", "Every explicit requested outcome is implemented", True, 1.0),
                ("functional-correctness", "Required behavior works under authoritative execution", True, 1.0),
            ]
            if declared_risks - {"low"} or len(accepted.tasks) > 1:
                standard_dimensions.extend(
                    [
                        ("runtime-stability", "No relevant runtime or provider error remains", True, 1.0),
                        ("integration-correctness", "The change is integrated without breaking observed contracts", True, 1.0),
                        ("regression-safety", "Impacted focused and configured regression checks pass", True, 1.0),
                    ]
                )
            if declared_risks & {"high", "critical"}:
                standard_dimensions.extend(
                    [
                        ("security-safety", "Affected trust boundaries and sensitive operations are verified", True, 1.0),
                        ("performance-safety", "Affected resource and performance constraints are measured", False, 0.85),
                    ]
                )
            standard_dimensions.append(
                ("maintainability", "The implementation is coherent, bounded, and avoids unnecessary complexity", False, 0.85)
            )
            artifact_dimensions = []
            if any(path.casefold().endswith((".html", ".htm")) for path in artifact_paths):
                artifact_dimensions = [
                    ("visual-quality", "Visual composition, hierarchy, detail, and polish meet the requested quality", False, 0.85),
                    ("interaction-quality", "Interactive and animated behavior is understandable, stable, and appropriately varied", False, 0.85),
                ]
            generated_dimensions = [
                {
                    "id": dimension_id,
                    "description": description,
                    "hard_gate": hard_gate,
                    "minimum_score": minimum,
                    "required_evidence": list(verification),
                    "evaluation_method": (
                        "vision_and_runtime"
                        if dimension_id in {
                            item[0] for item in artifact_dimensions
                        } and not hard_gate
                        else "deterministic_then_independent_review"
                    ),
                    "confidence": "medium",
                    "latest_artifact_hash": None,
                    "latest_mutation_sequence": None,
                }
                for dimension_id, description, hard_gate, minimum in (
                    *standard_dimensions,
                    *artifact_dimensions,
                )
            ]
            updated_contract = GoalContractV1(
                **{
                    **contract.to_dict(),
                    "required_outcomes": tuple(task.title for task in accepted.tasks),
                    "acceptance_criteria": criteria,
                    "required_verification": verification,
                    "artifact_expectations": artifact_paths,
                    "completion_conditions": (*contract.completion_conditions, *criteria),
                }
            )
            target = {
                "version": 1,
                "id": f"quality-{goal.id}",
                "objective": updated_contract.interpreted_objective,
                "artifact_ids": artifact_paths or ("workspace",),
                "minimum_overall_score": 0.95,
                "hard_gates": list(verification),
                "dimensions": [
                    {
                        "id": f"criterion-{index:03d}",
                        "description": criterion,
                        "hard_gate": True,
                        "minimum_score": 1.0,
                        "required_evidence": list(verification),
                    }
                    for index, criterion in enumerate(criteria, 1)
                ] + generated_dimensions,
            }
            updated_contract = GoalContractV1(**{**updated_contract.to_dict(), "quality_target_id": target["id"]})
            self.store.update_goal_metadata(
                goal.id,
                goal_contract=updated_contract.to_dict(),
                goal_contract_fingerprint=updated_contract.fingerprint,
                quality_target=target,
                convergence_state="not_evaluated",
                resource_claims=resource_claims,
            )
            self.store.append_event(
                "quality_target.created", goal_id=goal.id,
                payload={"target_id": target["id"], "dimensions": len(target["dimensions"]), "hard_gates": len(target["hard_gates"])},
            )
        self._work_conversation = [
            {
                "role": "user",
                "content": f"The user approved plan r{accepted.revision}. Begin the first ready task and keep the checklist current.",
            }
        ]
        self.events.publish(
            "workflow.state",
            f"Approval recorded for revision r{accepted.revision}; execution is ready to start.",
            **self.workflow_runtime_snapshot().to_dict(),
        )
        return accepted

    def reject_plan(self, feedback: str) -> Plan | None:
        feedback = redact_text(feedback, 4_000)
        goal, plan = self.active_goal(), self.latest_plan()
        if goal is None:
            raise RuntimeStateError("there is no active goal")
        if goal.status == GoalStatus.PAUSED and goal.metadata.get("resume_status") in {
            GoalStatus.DISCOVERING.value,
            GoalStatus.REVISING.value,
        }:
            desired = GoalStatus(goal.metadata["resume_status"])
            self.store.update_goal_metadata(goal.id, waiting_question="")
            goal = self.store.transition_goal(
                goal.id,
                desired,
                reason="user restarted paused planning with new guidance",
            )
        if plan is None:
            if goal.status not in {GoalStatus.DISCOVERING, GoalStatus.REVISING}:
                raise RuntimeStateError("there is no plan to reject")
            return self.generate_plan(feedback)
        if goal.status == GoalStatus.RUNNING:
            self.store.transition_goal(goal.id, GoalStatus.REVISING, reason=f"user requested replan: {feedback}")
        elif plan.status == PlanStatus.PENDING_APPROVAL:
            self.store.reject_plan(goal.id, plan.revision, feedback, rejected_by="user")
        elif goal.status != GoalStatus.REVISING:
            raise RuntimeStateError(f"cannot replan while goal is {goal.status.value}")
        return self.generate_plan(feedback)

    def _next_task_id(self, tasks: Iterable[Task | Mapping[str, Any]]) -> str:
        used = {str(task.id if isinstance(task, Task) else task.get("id", "")).upper() for task in tasks}
        index = 1
        while f"T{index:03d}" in used:
            index += 1
        return f"T{index:03d}"

    def _revision_context(self) -> tuple[Goal, Plan]:
        goal, plan = self.active_goal(), self.latest_plan()
        if goal is None or plan is None:
            raise RuntimeStateError("an existing plan is required")
        if goal.status not in {
            GoalStatus.AWAITING_PLAN_APPROVAL,
            GoalStatus.RUNNING,
            GoalStatus.PAUSED,
            GoalStatus.REVISING,
            GoalStatus.VERIFYING,
            GoalStatus.REVIEWING,
        }:
            raise RuntimeStateError(f"cannot revise a plan while goal is {goal.status.value}")
        return goal, plan

    def revise_plan(
        self,
        *,
        reason: str,
        add: Iterable[Mapping[str, Any]] = (),
        edit: tuple[str, str] | tuple[str, str, str] | None = None,
        remove: str | None = None,
        proposed_by: str = "user",
        inherit_approved_scope: bool = False,
    ) -> Plan:
        reason = redact_text(reason, 4_000)
        goal, old_plan = self._revision_context()
        task_values = [_task_dict(task) for task in old_plan.tasks]
        reset_ids: set[str] = set()
        for value in task_values:
            if value["status"] in {
                TaskStatus.IN_PROGRESS.value,
                TaskStatus.VERIFYING.value,
                TaskStatus.UNCERTAIN.value,
            }:
                value["status"] = TaskStatus.PENDING.value
                reset_ids.add(value["id"].upper())
        if edit:
            if len(edit) == 2:
                task_id, value = edit
                field_name = "task"
            else:
                task_id, field_name, value = edit
            found = False
            for value in task_values:
                if value["id"].upper() == task_id.upper():
                    edited_value = redact_text(edit[-1], 4_000).strip()
                    if not edited_value:
                        raise RuntimeStateError("task edit value must not be empty")
                    if field_name == "title":
                        value["title"] = edited_value
                    elif field_name == "description":
                        value["description"] = edited_value
                    elif field_name == "accept":
                        value["acceptance_criteria"] = [
                            item.strip() for item in edited_value.split("||") if item.strip()
                        ]
                    elif field_name == "verify":
                        value["verification"] = [
                            item.strip() for item in edited_value.split("||") if item.strip()
                        ]
                    elif field_name == "depends":
                        value["depends_on"] = [
                            item.strip().upper()
                            for item in edited_value.split(",")
                            if item.strip() and item.strip() != "-"
                        ]
                    elif field_name == "risk":
                        if edited_value.lower() not in {"low", "medium", "high", "critical"}:
                            raise RuntimeStateError("risk must be low, medium, high, or critical")
                        value["risk"] = edited_value.lower()
                    elif field_name == "task":
                        value["title"] = edited_value
                        value["description"] = edited_value
                        value["acceptance_criteria"] = [f"{edited_value} is implemented and directly evidenced."]
                        value["verification"] = [f"Run or inspect the most direct verification for: {edited_value}"]
                        value["role"] = RoleProfile().to_dict()
                    else:
                        raise RuntimeStateError(f"unknown editable task field: {field_name}")
                    value["status"] = TaskStatus.PENDING.value
                    reset_ids.add(value["id"].upper())
                    found = True
            if not found:
                raise NotFoundError(f"task not found: {task_id}")
        if remove:
            before = len(task_values)
            task_values = [value for value in task_values if value["id"].upper() != remove.upper()]
            if len(task_values) == before:
                raise NotFoundError(f"task not found: {remove}")
            if not task_values:
                raise RuntimeStateError("a plan must retain at least one task")
            for value in task_values:
                depended_on_removed = any(
                    item.upper() == remove.upper() for item in value["depends_on"]
                )
                value["depends_on"] = [item for item in value["depends_on"] if item.upper() != remove.upper()]
                if str(value.get("parent_id") or "").upper() == remove.upper():
                    value["parent_id"] = None
                    depended_on_removed = True
                if depended_on_removed:
                    value["status"] = TaskStatus.PENDING.value
                    reset_ids.add(value["id"].upper())
        for raw in add:
            value = dict(raw)
            if not value.get("id") or any(item["id"].upper() == str(value["id"]).upper() for item in task_values):
                value["id"] = self._next_task_id(task_values)
            value.setdefault("status", TaskStatus.PENDING.value)
            value.setdefault("origin", proposed_by)
            value.setdefault("depends_on", [])
            value.setdefault("risk", "medium")
            value.setdefault("role", RoleProfile().to_dict())
            value.setdefault("mode", "auto")
            # Plans are displayed in descending priority order. A newly added
            # user request belongs at the end of the existing execution order
            # unless the user explicitly reorders it.
            value.setdefault(
                "priority",
                min(
                    (int(item.get("priority", 0)) for item in task_values),
                    default=0,
                )
                - 1,
            )
            value.setdefault("attempts", 0)
            value.setdefault("metadata", {})
            task_values.append(value)

        # Editing a prerequisite invalidates completed dependants as well.
        changed = True
        while changed:
            changed = False
            for value in task_values:
                if any(dep.upper() in reset_ids for dep in value["depends_on"]) and value["id"].upper() not in reset_ids:
                    value["status"] = TaskStatus.PENDING.value
                    reset_ids.add(value["id"].upper())
                    changed = True

        # Validate the full replacement before changing the live goal phase. If
        # a user mistypes an ID or an agent proposes a cyclic DAG, the accepted
        # plan remains runnable instead of getting stranded in REVISING.
        preview_revision = old_plan.revision + 1
        preview_tasks = tuple(
            self.store.coerce_task(value, goal.id, preview_revision, proposed_by)
            for value in task_values
        )
        validate_task_dag(preview_tasks)
        retained_ids = {task.id for task in preview_tasks}
        applicability = []
        for item in old_plan.applicability_evidence:
            copied = dict(item)
            copied["supports_tasks"] = [
                str(task_id).upper()
                for task_id in copied.get("supports_tasks", ())
                if str(task_id).upper() in retained_ids
            ]
            if copied["supports_tasks"]:
                applicability.append(copied)
        expected_changes = []
        for item in old_plan.expected_changes:
            copied = dict(item)
            copied["supports_tasks"] = [
                str(task_id).upper()
                for task_id in copied.get("supports_tasks", ())
                if str(task_id).upper() in retained_ids
            ]
            if copied["supports_tasks"]:
                expected_changes.append(copied)
        covered = {
            str(task_id).upper()
            for item in applicability
            for task_id in item["supports_tasks"]
        }
        change_covered = {
            str(task_id).upper()
            for item in expected_changes
            for task_id in item["supports_tasks"]
        }
        for task in preview_tasks:
            if task.id not in covered:
                applicability.append(
                    {
                        "fact": f"{task.title} was added to plan r{preview_revision}: {task.description}",
                        "source": f"{proposed_by} plan revision",
                        "supports_tasks": [task.id],
                    }
                )
            if task.id not in change_covered:
                if inherit_approved_scope and expected_changes:
                    for item in expected_changes:
                        item["supports_tasks"] = list(
                            dict.fromkeys(
                                (*item.get("supports_tasks", ()), task.id)
                            )
                        )
                elif not inherit_approved_scope:
                    expected_changes.append(
                        {
                            "path": f"<resolved during {task.id}>",
                            "intent": task.description,
                            "supports_tasks": [task.id],
                        }
                    )

        original_status = goal.status
        if original_status != GoalStatus.REVISING:
            self.store.transition_goal(goal.id, GoalStatus.REVISING, reason=reason)
        try:
            plan = self.store.create_plan(
                goal.id,
                f"{old_plan.summary}\nRevision reason: {reason}",
                task_values,
                applicability_evidence=applicability,
                execution_strategy=(
                    f"{old_plan.execution_strategy}\nRevision strategy: {reason}"
                ),
                expected_changes=expected_changes,
                proposed_by=proposed_by,
                submit=True,
            )
        except Exception:
            if original_status != GoalStatus.REVISING:
                fallback = (
                    original_status
                    if original_status in {GoalStatus.AWAITING_PLAN_APPROVAL, GoalStatus.RUNNING, GoalStatus.PAUSED}
                    else GoalStatus.RUNNING
                )
                self.store.transition_goal(goal.id, fallback, reason="invalid plan revision rolled back")
            raise
        for value in task_values:
            if value["status"] == TaskStatus.COMPLETED.value and value["id"].upper() not in reset_ids:
                prior = self.store.list_evidence(goal.id, task_id=value["id"])
                for item in prior[-10:]:
                    self.store.add_evidence(
                        goal_id=goal.id,
                        plan_revision=plan.revision,
                        task_id=value["id"],
                        kind=item.kind,
                        summary=f"Carried from r{item.plan_revision}: {item.summary}",
                        data=item.data,
                        created_by="harness",
                        verified=item.verified,
                    )
        self.store.transition_goal(
            goal.id,
            GoalStatus.AWAITING_PLAN_APPROVAL,
            reason=f"plan revision r{plan.revision} requires user approval",
        )
        self.events.publish("plan", f"Plan r{plan.revision} is pending approval: {reason}")
        return plan

    def plan_document(self) -> str:
        plan = self.latest_plan()
        if plan is None:
            raise RuntimeStateError("there is no plan to review")
        saved = self.store.get_plan_document(plan.id)
        return str(saved["content"]) if saved is not None else render_plan_document(plan)

    def replace_plan_document(
        self,
        document: str,
        *,
        reason: str = "manual full-plan edit",
        edited_by: str = "user",
    ) -> Plan:
        parsed = parse_plan_document(document)
        goal, old_plan = self._revision_context()
        prior_by_id = {task.id: task for task in old_plan.tasks}
        task_values: list[dict[str, Any]] = []
        for index, value in enumerate(parsed.tasks):
            prior = prior_by_id.get(str(value["id"]).upper())
            task_values.append(
                {
                    **dict(value),
                    "status": TaskStatus.PENDING.value,
                    "role": (prior.role.to_dict() if prior is not None else RoleProfile().to_dict()),
                    "mode": prior.mode if prior is not None else "auto",
                    "priority": prior.priority if prior is not None else index,
                    "origin": "user-editor",
                    "metadata": dict(prior.metadata) if prior is not None else {},
                }
            )
        original_status = goal.status
        if original_status is not GoalStatus.REVISING:
            self.store.transition_goal(goal.id, GoalStatus.REVISING, reason=reason)
        try:
            plan = self.store.create_plan(
                goal.id,
                parsed.summary,
                task_values,
                applicability_evidence=parsed.applicability_evidence,
                execution_strategy=parsed.execution_strategy,
                expected_changes=parsed.expected_changes,
                proposed_by="user-editor",
                submit=True,
                source_document=document,
                source_format_version=1,
                edited_by=edited_by,
            )
        except Exception:
            if original_status is not GoalStatus.REVISING:
                self.store.transition_goal(
                    goal.id,
                    original_status,
                    reason="invalid manual plan edit rolled back",
                )
            raise
        self.store.transition_goal(
            goal.id,
            GoalStatus.AWAITING_PLAN_APPROVAL,
            reason=f"manual plan document saved as r{plan.revision}",
        )
        session = self.store.get_workflow_session(self.session_id)
        self.store.mutate_workflow_session(
            self.session_id,
            lambda current_state: {
                "state": {
                    **dict(current_state.get("state") or {}),
                    "plan_revision": plan.revision,
                    "plan_fingerprint": plan.fingerprint,
                },
                "goal_id": current_state.get("goal_id") or goal.id,
                "plan_state": PlanState.AWAITING_APPROVAL.value,
                "run_state": RunState.PLANNING.value,
            },
            expected_revision=int(session.get("revision") or 0),
        )
        self.events.publish(
            "plan",
            f"Manual plan edit saved as revision r{plan.revision}.",
            revision=plan.revision,
            source="user-editor",
        )
        return plan

    def add_user_task(self, text: str, acceptance_criteria: str = "") -> Plan:
        text = redact_text(text, 2_000)
        acceptance_criteria = redact_text(acceptance_criteria, 2_000)
        current = self.latest_plan()
        if current is None:
            raise RuntimeStateError("create a plan before adding checklist items")
        item = {
            "id": self._next_task_id(current.tasks),
            "title": text[:180],
            "description": text,
            "acceptance_criteria": [acceptance_criteria or f"{text} is implemented and evidenced."],
            "verification": [f"Run or inspect the most direct verification for: {text}"],
            "depends_on": [],
            "risk": "medium",
            "origin": "user",
        }
        return self.revise_plan(reason="user added a checklist item", add=[item], proposed_by="user")

    def update_task_from_user(self, task_id: str, status: str, note: str = "") -> Task:
        goal, plan = self.active_goal(), self.latest_plan()
        if goal is None or plan is None or plan.status != PlanStatus.ACCEPTED:
            raise RuntimeStateError("task status can change only on an accepted plan")
        mapping = {
            "done": TaskStatus.COMPLETED,
            "pending": TaskStatus.PENDING,
            "blocked": TaskStatus.BLOCKED,
            "skipped": TaskStatus.OBSOLETE,
        }
        target = mapping[status]
        note = redact_text(note, 2_000)
        if target in {TaskStatus.COMPLETED, TaskStatus.BLOCKED, TaskStatus.OBSOLETE} and not note.strip():
            raise RuntimeStateError(f"{status} requires an evidence/reason note")
        evidence = [f"User evidence: {note}"] if target == TaskStatus.COMPLETED else []
        if target == TaskStatus.PENDING:
            self._reset_dependants(goal, plan, task_id, actor="user")
        return self.store.transition_task(
            goal.id,
            plan.revision,
            task_id,
            target,
            note=note,
            evidence=evidence,
            actor="user",
        )

    def add_guidance(self, text: str) -> Evidence:
        goal = self.active_goal()
        if goal is None:
            raise RuntimeStateError("no active goal")
        item = self.store.add_evidence(
            goal_id=goal.id,
            plan_revision=goal.active_plan_revision,
            kind="guidance",
            summary=redact_text(text, 4_000),
            created_by="user",
        )
        latest = self.store.get_goal(goal.id)
        feedback = item.summary
        lowered = feedback.casefold()
        if (
            latest.status is GoalStatus.PAUSED
            and latest.metadata.get("convergence_state") == "user_review_required"
            and lowered.strip().rstrip(".! ") in {"yes", "y", "accept", "accepted", "approve", "approved", "looks good", "ship it"}
        ):
            plan = self.store.get_accepted_plan(goal.id)
            blocker = self._completion_precheck(latest, plan) if plan is not None else "no accepted plan"
            if blocker:
                self.store.append_event("completion.rejected", goal_id=goal.id, payload={"reason": blocker, "source": "user_visual_acceptance"})
                raise RuntimeStateError(f"visual acceptance cannot bypass completion blocker: {blocker}")
            evaluation = dict(latest.metadata.get("latest_evaluation", {}))
            evaluation["user_visual_acceptance_evidence_id"] = item.id
            evaluation["confidence"] = "user_accepted_subjective"
            evaluation["accepted_artifact_hashes"] = self._current_artifact_hashes(
                latest.metadata.get("quality_target", {}).get("artifact_ids", ())
            )
            self.store.update_goal_metadata(
                goal.id,
                convergence_state="converged",
                latest_evaluation=evaluation,
                waiting_question="",
            )
            self._checkpoint_accepted_goal(
                self.store.get_goal(goal.id),
                source="user_visual_acceptance",
            )
            self.store.transition_goal(goal.id, GoalStatus.REVIEWING, reason="user accepted only the unresolved subjective visual dimension")
            self.store.transition_goal(goal.id, GoalStatus.COMPLETED, reason="correctness gates and independent review passed; user accepted subjective visual quality")
            self._record_global_learning(
                self.store.get_goal(goal.id),
                succeeded=True,
                evidence_ref=f"goal:{goal.id}:user-visual-acceptance:{item.id}",
            )
            self.store.append_event(
                "quality_convergence.decided", goal_id=goal.id,
                payload={"state": "converged", "source": "explicit_user_visual_acceptance", "evidence_id": item.id},
            )
            return item
        operational_guidance = self._guidance_is_operational(feedback)
        if operational_guidance:
            actions = []
            for action in latest.metadata.get("refinement_actions", ()):
                if not isinstance(action, Mapping):
                    continue
                value = dict(action)
                if (
                    value.get("status") == "pending"
                    and self._guidance_is_operational(
                        str(value.get("feedback") or "")
                    )
                ):
                    value["status"] = "resolved"
                    value["resolution"] = (
                        "Operational recovery guidance does not change the "
                        "accepted product quality target."
                    )
                    value["resolution_evidence_id"] = item.id
                actions.append(value)
            self.store.update_goal_metadata(
                goal.id,
                refinement_actions=actions,
                convergence_state=(
                    "reverifying"
                    if latest.metadata.get("quality_target")
                    else latest.metadata.get("convergence_state", "not_evaluated")
                ),
            )
            self.store.append_event(
                "guidance.operational",
                goal_id=goal.id,
                payload={"evidence_id": item.id},
            )
            self._work_conversation.append(
                {"role": "user", "content": f"User guidance: {item.summary}"}
            )
            if goal.status == GoalStatus.PAUSED and goal.metadata.get(
                "waiting_question"
            ):
                self.store.update_goal_metadata(goal.id, user_answer=item.summary)
                if not self._unresolved_recovery_entities(goal.id):
                    self.resume()
            return item
        # Feedback is attached to the accepted semantic contract without
        # guessing a product dimension from keywords.
        dimensions = ["requirement_completeness"]

        contract_data = latest.metadata.get("goal_contract")
        components: list[dict[str, Any]] = []
        if isinstance(contract_data, Mapping):
            contract = GoalContractV1.from_dict(contract_data)
            artifact_paths = contract.artifact_expectations
            for path in artifact_paths:
                candidate = (self.workspace / path).resolve(strict=False)
                if candidate.is_file() and candidate.is_relative_to(self.workspace):
                    self.repository_index.update(candidate.relative_to(self.workspace).as_posix())
            retrieval_query = feedback
            context_slice = self.repository_index.context_slice(
                retrieval_query,
                max_entries=30,
                budget_chars=30_000,
            )
            components = [
                {"path": entry.path, "kind": entry.kind, "name": entry.name, "start": entry.start, "end": entry.end, "file_hash": entry.file_hash}
                for entry in context_slice.entries
            ]
            context_summary = {
                "query": retrieval_query,
                "size_chars": context_slice.size_chars,
                "omitted_entries": context_slice.omitted_entries,
                "callers": {key: list(value) for key, value in context_slice.callers.items()},
                "callees": {key: list(value) for key, value in context_slice.callees.items()},
                "dependencies": {key: list(value) for key, value in context_slice.dependencies.items()},
            }
            updated_contract = GoalContractV1(**{
                **contract.to_dict(),
                "user_feedback": (*contract.user_feedback, feedback),
                "file_symbol_scope": tuple(dict.fromkeys(
                    (*contract.file_symbol_scope, *(f"{entry['path']}:{entry['kind']}:{entry['name']}" for entry in components))
                )),
            })
            target = dict(latest.metadata.get("quality_target", {}))
            target["explicit_feedback"] = [*target.get("explicit_feedback", []), feedback]
            actions = list(latest.metadata.get("refinement_actions", ()))
            action = {
                "id": f"refinement-{len(actions) + 1:03d}",
                "feedback": feedback,
                "affected_dimensions": dimensions,
                "affected_components": components,
                "repository_context_slice": context_summary,
                "objective": f"Improve {', '.join(dimensions)} while preserving previously verified functionality.",
                "status": "pending",
            }
            actions.append(action)
            self.store.update_goal_metadata(
                goal.id,
                goal_contract=updated_contract.to_dict(),
                goal_contract_fingerprint=updated_contract.fingerprint,
                quality_target=target,
                refinement_actions=actions,
                convergence_state="refining",
            )
            self.store.append_event(
                "refinement_action.created", goal_id=goal.id,
                payload={"action_id": action["id"], "dimensions": dimensions, "components": len(components), "feedback_evidence_id": item.id},
            )
        self._work_conversation.append({"role": "user", "content": f"User guidance: {item.summary}"})
        if goal.status == GoalStatus.PAUSED and goal.metadata.get("waiting_question"):
            self.store.update_goal_metadata(goal.id, user_answer=item.summary)
            if not self._unresolved_recovery_entities(goal.id):
                self.resume()
            else:
                self.events.publish(
                    "warning",
                    "Guidance was saved, but crash-window work is still uncertain; reconcile it with /resolve before resuming.",
                )
        return item

    @staticmethod
    def _guidance_is_operational(text: str) -> bool:
        lowered = str(text).casefold()
        return (
            any(
                marker in lowered
                for marker in (
                    "tool permissions are now",
                    "permissions are now available",
                    "authoritative command",
                    "do not mutate outside",
                    "finish_goal",
                    "complete t00",
                )
            )
            or (
                any(marker in lowered for marker in ("retry", "resume", "continue"))
                and any(
                    marker in lowered
                    for marker in ("pytest", "verification", "existing evidence")
                )
                and any(
                    marker in lowered
                    for marker in ("do not", "already", "currently", "now")
                )
            )
        )

    def _unresolved_recovery_entities(self, goal_id: str) -> tuple[str, ...]:
        action_ids = [
            str(item["id"])
            for item in self.store.list_actions(goal_id, status="uncertain")
        ]
        delegation_ids = [
            item.id
            for item in self.store.list_delegations(goal_id)
            if item.status == DelegationStatus.UNCERTAIN
        ]
        plan = self.store.get_latest_plan(goal_id)
        task_ids = [
            task.id
            for task in (() if plan is None else plan.tasks)
            if task.status == TaskStatus.UNCERTAIN
        ]
        ultra_ids: list[str] = []
        goal = self.store.get_goal(goal_id)
        run_id = str(goal.metadata.get("ultra_run_id", ""))
        if run_id:
            try:
                ultra_ids.extend(
                    item.id
                    for item in self.store.list_work_nodes(run_id)
                    if item.status.value == "uncertain"
                )
                ultra_ids.extend(
                    item.id
                    for item in self.store.list_agent_runs(run_id)
                    if item.status.value == "uncertain"
                )
                ultra_ids.extend(
                    item.id
                    for item in self.store.list_resource_leases(run_id)
                    if item.status.value == "uncertain"
                )
            except StateStoreError:
                pass
        return tuple(
            dict.fromkeys([*action_ids, *delegation_ids, *task_ids, *ultra_ids])
        )

    def _auto_reconcile_declared_tool_side_effects(self, goal: Goal) -> tuple[str, ...]:
        """Reconcile legacy uncertainty caused solely by a declared tool footprint.

        The action is never replayed. A prior authoritative success event and a
        strict subset proof against the same tool's fixed derived paths are both
        required. Unknown or source-file changes remain uncertain.
        """

        records = [
            dict(item)
            for item in goal.metadata.get("uncertain_actions", ())
            if isinstance(item, Mapping)
        ]
        if not records:
            return ()
        actions = {str(item["id"]): item for item in self.store.list_actions(goal.id)}
        events = self.store.list_recent_events(goal.id, limit=2_000)
        reconciled: list[str] = []
        remaining: list[dict[str, Any]] = []
        for record in records:
            action_id = str(record.get("action_id") or "")
            action = actions.get(action_id)
            if action is None or str(action.get("status")) != "uncertain":
                remaining.append(record)
                continue
            tool_name = str(action.get("tool_name") or "")
            if tool_name != "install_dependencies":
                remaining.append(record)
                continue
            try:
                journal_args = json.loads(str(action.get("args_json") or "{}"))
            except (TypeError, ValueError, json.JSONDecodeError):
                remaining.append(record)
                continue
            arguments = journal_args.get("arguments", {})
            if not isinstance(arguments, Mapping):
                remaining.append(record)
                continue
            footprint = tools.mutation_footprint(tool_name, arguments, ())
            changed = {
                str(path).strip().replace("\\", "/")
                for path in record.get("paths", ())
                if str(path).strip()
            }
            if not changed or not changed.issubset(set(footprint.derived_paths)):
                remaining.append(record)
                continue
            completion = next(
                (
                    event
                    for event in reversed(events)
                    if event.entity_id == action_id
                    and event.event_type == "action.completed"
                ),
                None,
            )
            if completion is None:
                remaining.append(record)
                continue
            result_text = str(completion.payload.get("result") or "")
            try:
                result_value = json.loads(result_text)
            except (TypeError, ValueError, json.JSONDecodeError):
                result_value = {}
            try:
                successful = (
                    isinstance(result_value, Mapping)
                    and str(result_value.get("status") or "") == "installed"
                    and int(result_value.get("exit_code", -1)) == 0
                )
            except (TypeError, ValueError):
                successful = False
            if not successful:
                # Historical summaries may be safely truncated after the
                # structured fields but before the closing JSON brace. Both
                # markers are emitted by the harness-owned installer before
                # arbitrary command output.
                successful = bool(
                    re.search(r'"status"\s*:\s*"installed"', result_text)
                    and re.search(r'"exit_code"\s*:\s*0(?:\D|$)', result_text)
                )
            if not successful:
                remaining.append(record)
                continue
            note = (
                "Harness reconciled a successful dependency install whose only "
                "previously-unleased output is now a declared lockfile footprint: "
                + ", ".join(sorted(changed))
            )
            self.store.resolve_action(
                action_id,
                "applied",
                note,
                actor="harness-footprint-v1",
            )
            self.store.append_event(
                "execution.reconciled",
                goal_id=goal.id,
                entity_type="action",
                entity_id=action_id,
                payload={
                    "reason": "declared mutation footprint",
                    "paths": sorted(changed),
                    "footprint_fingerprint": footprint.fingerprint,
                    "mutation_replayed": False,
                },
            )
            reconciled.append(action_id)
        if reconciled:
            self.store.update_goal_metadata(
                goal.id,
                uncertain_actions=remaining,
                waiting_question=(
                    str(goal.metadata.get("waiting_question") or "")
                    if remaining
                    else ""
                ),
            )
        return tuple(reconciled)

    def _auto_reconcile_read_only_ultra_uncertainty(self) -> tuple[str, ...]:
        """Reset provably side-effect-free crash windows without user ceremony."""

        from .ultra_models import AgentRunStatus, WorkNodeStatus

        goal = self.store.load_active_goal(self.session_id)
        if goal is None:
            return ()
        run_id = str(goal.metadata.get("ultra_run_id", ""))
        if not run_id:
            return ()
        reconciled: list[str] = []
        uncertain_agents = [
            item
            for item in self.store.list_agent_runs(run_id)
            if item.status is AgentRunStatus.UNCERTAIN
        ]
        uncertain_mutation_scopes: set[tuple[str, str]] = set()
        for action in self.store.list_actions(goal.id):
            if not bool(action.get("mutating")):
                continue
            if str(action.get("status") or "").casefold() not in {
                "running",
                "uncertain",
            }:
                # A completed mutation has a durable tool receipt and is
                # recovered through its ChangeSet/artifact hashes. It must not
                # make a later interrupted read/model call look unsafe.
                continue
            arguments = action.get("args")
            if not isinstance(arguments, Mapping):
                arguments = action.get("arguments")
            if not isinstance(arguments, Mapping):
                try:
                    decoded = json.loads(str(action.get("args_json") or "{}"))
                except (TypeError, ValueError, json.JSONDecodeError):
                    decoded = {}
                arguments = decoded if isinstance(decoded, Mapping) else {}
            if not isinstance(arguments, Mapping):
                continue
            node_id = str(arguments.get("node_id") or "")
            if node_id:
                uncertain_mutation_scopes.add(
                    (node_id, str(arguments.get("phase") or ""))
                )
        unsafe_nodes = {
            str(item.work_node_id)
            for item in uncertain_agents
            if item.side_effects
            and item.work_node_id
            and (
                str(item.work_node_id),
                str(item.phase),
            )
            in uncertain_mutation_scopes
        }
        for agent in uncertain_agents:
            if (
                agent.side_effects
                and (
                    str(agent.work_node_id or ""),
                    str(agent.phase),
                )
                in uncertain_mutation_scopes
            ):
                continue
            self.store.update_agent_run(
                agent.id,
                AgentRunStatus.CANCELLED,
                error="interrupted read-only model call; safe to recompute from durable input",
            )
            reconciled.append(agent.id)
        for node in self.store.list_work_nodes(run_id):
            if node.status is not WorkNodeStatus.UNCERTAIN:
                continue
            if node.id in unsafe_nodes:
                continue
            self.store.transition_work_node(
                node.id,
                WorkNodeStatus.PENDING,
                error=None,
                checkpoint="auto_reconciled_read_only",
            )
            reconciled.append(node.id)
        return tuple(reconciled)

    def pause(self, reason: str = "paused by user") -> Goal:
        goal = self.active_goal()
        if goal is None:
            raise RuntimeStateError("no active goal")
        if goal.status == GoalStatus.PAUSED:
            return goal
        if self.ultra_session is not None and self.ultra_session.running:
            self.ultra_session.pause()
        self.store.update_goal_metadata(
            goal.id,
            resume_status=goal.status.value,
            auto_retryable=False,
            retry_after_ms=0,
        )
        result = self.store.transition_goal(goal.id, GoalStatus.PAUSED, reason=reason)
        self.events.publish("phase", "Goal paused safely; state is durable.")
        return result

    def request_pause(self, reason: str = "pause requested by user") -> Goal:
        """Persist a cooperative pause request without misreporting completion."""

        goal = self.active_goal()
        if goal is None:
            raise RuntimeStateError("no active goal")
        if goal.status is GoalStatus.PAUSED:
            return goal
        if self.ultra_session is not None and self.ultra_session.running:
            self.ultra_session.pause()
        self.store.update_goal_metadata(
            goal.id,
            pause_requested=True,
            pause_requested_at=time.time(),
            pause_reason=str(reason),
            resume_status=goal.status.value,
            auto_retryable=False,
            retry_after_ms=0,
        )
        self.events.publish(
            "phase",
            "Pause requested; finishing the current operation before the saved checkpoint.",
            pause_requested=True,
        )
        return self.store.get_goal(goal.id)

    def stop_now(self, *, shutdown_ollama: bool = False) -> Any:
        """Hard-stop the active boundary while preserving a resumable stage."""
        if shutdown_ollama and hasattr(self.provider, "shutdown_local_server") and not bool(getattr(self.provider, "is_loopback_host", getattr(self.provider, "is_local_runner", False))):
            raise RuntimeStateError("/stop ollama is allowed only for the local loopback Ollama runner")
        self._stop_event.set()
        active = self._active_provider_abandon
        if active is not None:
            active.set()
        # Cancellation/unload can involve a socket or process wait. Keep the
        # command responsive and let the durable pause be visible immediately.
        def release_provider() -> None:
            cancel = getattr(self.provider, "cancel_active_request", None) or getattr(self.provider, "cancel", None)
            if callable(cancel):
                try:
                    cancel()
                except Exception:
                    pass
            unload = getattr(self.provider, "unload_model", None)
            if callable(unload):
                try:
                    unload()
                except Exception:
                    pass
            if shutdown_ollama:
                shutdown = getattr(self.provider, "shutdown_local_server", None)
                if callable(shutdown):
                    try:
                        shutdown()
                    except Exception as exc:
                        self.events.publish("warning", f"Ollama shutdown was not completed: {redact_text(exc, 300)}")
        Thread(target=release_provider, name="ga3bad-hard-stop", daemon=True).start()
        try:
            tools.shutdown_workspace_resources(self.workspace)
        except Exception:
            pass
        try:
            recovery = self.store.recover_inflight()
        except Exception:
            recovery = None
        goal = self.active_goal()
        if goal is not None:
            self.store.update_goal_metadata(
                goal.id,
                hard_stop=True,
                hard_stop_at=time.time(),
                shutdown_ollama=bool(shutdown_ollama),
                resume_action="resume_same_stage",
                auto_retryable=False,
                uncertain_on_stop=(
                    {
                        "tasks": list(getattr(recovery, "task_ids", ())),
                        "actions": list(getattr(recovery, "action_ids", ())),
                        "delegations": list(getattr(recovery, "delegation_ids", ())),
                    }
                    if recovery is not None else {}
                ),
            )
            try:
                if goal.status is not GoalStatus.PAUSED:
                    goal = self.store.transition_goal(goal.id, GoalStatus.PAUSED, reason="stopped by user; saved stage is resumable")
            except Exception:
                goal = self.store.get_goal(goal.id)
            self.events.publish(
                "execution.boundary",
                "Stopped by user. The saved stage is resumable and no late response will be applied.",
                goal_id=goal.id, reason="hard_stop", shutdown_ollama=bool(shutdown_ollama),
            )
            return goal
        session = self.store.get_workflow_session(self.session_id)
        def reduce_stop(current_state: dict[str, Any]) -> Mapping[str, Any]:
            state = dict(current_state.get("state") or {})
            pending = state.get("pending_semantic_turn")
            if isinstance(pending, Mapping):
                saved = dict(pending)
                saved.update(
                    {
                        "status": "paused",
                        "stage": saved.get("stage") or "routing",
                        "hard_stop": True,
                        "shutdown_ollama": bool(shutdown_ollama),
                        "resume_action": "resume_same_stage",
                    }
                )
                state["pending_semantic_turn"] = saved
            state["hard_stop"] = {
                "at": time.time(),
                "shutdown_ollama": bool(shutdown_ollama),
            }
            return {
                "state": state,
                "run_state": "paused",
            }

        self.store.mutate_workflow_session(
            self.session_id,
            reduce_stop,
            expected_revision=int(session.get("revision") or 0),
        )
        self.events.publish("execution.boundary", "Stopped by user. The saved request is resumable.", reason="hard_stop", shutdown_ollama=bool(shutdown_ollama))
        return SliceResult("paused", "Stopped now. The saved stage is saved and can be resumed without replaying mutations.", needs_user=False, phase="paused", reason="hard_stop", resume_action="resume_same_stage")

    def complete_requested_pause(self) -> Goal | None:
        """Commit PAUSED only once active work has actually drained."""

        goal = self.active_goal()
        if goal is None or not bool(goal.metadata.get("pause_requested")):
            return goal
        if (
            self.ultra_session is not None
            and self.ultra_session.running
            and not self.ultra_session.safe_for_reconfiguration
        ):
            return goal
        self.store.update_goal_metadata(
            goal.id,
            pause_requested=False,
            pause_completed_at=time.time(),
        )
        result = self.pause(str(goal.metadata.get("pause_reason") or "paused by user"))
        self.events.publish(
            "checkpoint",
            "Paused at a saved checkpoint. Use /resume or /continue when ready.",
            paused=True,
            status="paused",
        )
        return result

    def _resume_pending_semantic_turn(self) -> Any:
        self._stop_event.clear()
        session = self.store.get_workflow_session(self.session_id)
        pending = dict(session.get("state", {}).get("pending_semantic_turn", {}))
        if not pending:
            raise RuntimeStateError("there is no pending semantic turn to resume")
        action_ids = {
            str(item.get("action_id"))
            for item in pending.get("action_records", ())
            if isinstance(item, Mapping) and str(item.get("action_id") or "")
        }
        uncertain = [
            item for item in self.store.list_session_actions(self.session_id)
            if str(item.get("id")) in action_ids and str(item.get("status")) == "uncertain"
        ]
        if uncertain:
            return SliceResult(
                "uncertain",
                "The prior process stopped during a mutating Action. Its real workspace state must be reconciled before continuing; the action was not replayed.",
                needs_user=True,
            )
        semantic_turn, semantic = self._semantic_preflight(resume_pending=True)
        semantic_turn["status"] = "dispatching"
        self._save_pending_semantic_turn(semantic_turn)
        original = str(semantic_turn["original_input"])
        if semantic.route is RouteKind.CHAT and not semantic.needs_workspace_tools:
            compact_response, _created = self._artifactize_chat_text(semantic.direct_response)
            assistant = {"role": "assistant", "content": compact_response}
            self._chat_conversation.append(assistant)
            self.store.append_chat_message(
                self.session_id,
                assistant,
                event_key=f"semantic:{semantic_turn['turn_id']}:assistant",
                run_id=str(semantic_turn["turn_id"]),
            )
            result: Any = SliceResult("chat", semantic.direct_response)
        elif semantic.route in {RouteKind.CHAT, RouteKind.ACTION}:
            result = self.chat(
                original,
                _route_checked=True,
                semantic_decision=semantic,
                semantic_turn_id=str(semantic_turn["turn_id"]),
            )
        else:
            result = self.submit_intent(
                original,
                requested_mode=str(semantic_turn.get("requested_mode") or session["session_mode"]),
                semantic_decision=semantic,
                semantic_turn_id=str(semantic_turn["turn_id"]),
            )
        self._complete_semantic_turn(str(semantic_turn["turn_id"]), result_status=str(getattr(result, "status", "routed")))
        return result

    def resume(self) -> Any:
        self._stop_event.clear()
        goal = self.active_goal()
        while goal is not None and goal.status is GoalStatus.AWAITING_PLAN_APPROVAL:
            pending_plan = self.latest_plan()
            accepted_plan = self.store.get_accepted_plan(goal.id)
            if (
                pending_plan is not None
                and pending_plan.status is PlanStatus.PENDING_APPROVAL
                and accepted_plan is not None
                and accepted_plan.revision < pending_plan.revision
                and self._repair_revision_is_in_scope(
                    accepted_plan,
                    pending_plan,
                    [_task_dict(task) for task in pending_plan.tasks],
                )
            ):
                self.store.update_goal_metadata(
                    goal.id,
                    scope_contraction_attempts=0,
                    plan_approval_policy={
                        "requires_user": False,
                        "reason": "harness-approved in-scope repair revision after recovery",
                        "risk_adaptive": True,
                    },
                )
                return self.approve_plan(
                    pending_plan.revision,
                    approved_by="risk-adaptive-policy",
                )
            if (
                pending_plan is None
                or pending_plan.status is not PlanStatus.PENDING_APPROVAL
                or accepted_plan is None
                or accepted_plan.revision >= pending_plan.revision
            ):
                break
            if goal.metadata.get("evidence_bound_scope_expansion"):
                # The in-scope strategy was already exhausted against
                # authoritative evidence. Keep the expanded revision pending
                # for explicit user approval instead of silently contracting it
                # back into the proven-insufficient scope.
                break
            contraction_attempts = int(
                goal.metadata.get("scope_contraction_attempts", 0) or 0
            )
            if contraction_attempts >= 2:
                break
            approved_paths = sorted(self._plan_change_paths(accepted_plan))
            self.store.reject_plan(
                goal.id,
                pending_plan.revision,
                "Harness rejected an avoidable repair scope expansion; preserve the "
                "accepted paths and capabilities.",
                rejected_by="scope-policy",
            )
            self.store.update_goal_metadata(
                goal.id,
                scope_contraction_attempts=contraction_attempts + 1,
            )
            contraction_feedback = (
                "Repair the active plan without adding paths, dependencies, network "
                "effects, permissions, or external side effects. Reuse only these "
                f"approved paths: {approved_paths!r}. Keep already-mutated artifacts "
                "and fresh evidence; produce a materially different executable strategy."
            )
            if goal.metadata.get("ultra_run_id"):
                # A recursive quality revision already owns an accepted semantic
                # foundation and repository inspection. Re-entering the Normal
                # planner here used to replay semantic interpretation, consume an
                # unrelated budget, and occasionally pause because the repair
                # response did not repeat old inspection references. Rebuild the
                # Ultra plan from its durable accepted foundation instead.
                result = self.replan_ultra(contraction_feedback)
                goal = self.active_goal()
                if goal is None or goal.status is not GoalStatus.AWAITING_PLAN_APPROVAL:
                    return result
                continue
            result = self.generate_plan(
                contraction_feedback,
                auto_approve_in_scope_repair=True,
            )
            goal = self.active_goal()
            if goal is None or goal.status is not GoalStatus.AWAITING_PLAN_APPROVAL:
                return result
        if goal is None:
            pending = self.store.get_workflow_session(self.session_id).get("state", {}).get(
                "pending_semantic_turn"
            )
            if isinstance(pending, Mapping):
                return self._resume_pending_semantic_turn()
        if goal is not None and goal.status in {
            GoalStatus.DISCOVERING,
            GoalStatus.REVISING,
        }:
            # Planning provider calls do not perform workspace mutations, so
            # there is no uncertain side effect to reconcile after a process
            # dies between turns. Rebuild the bounded planning conversation
            # from the exact goal, accepted semantic metadata, and fresh
            # repository inspection instead of leaving the durable goal in an
            # unresumable non-paused state.
            self.store.append_event(
                "planning.resumed",
                goal_id=goal.id,
                payload={
                    "status": goal.status.value,
                    "reason": "resume requested for an interrupted planning stage",
                },
            )
            return self.generate_plan(
                "Resume the interrupted planning stage from persisted semantic "
                "state. Preserve the exact original request and accepted scope."
            )
        if (
            goal is not None
            and goal.status is GoalStatus.PAUSED
            and str(goal.metadata.get("resume_action") or "") == "ultra_replan"
        ):
            feedback = str(
                goal.metadata.get("replan_feedback")
                or "Retry the saved in-scope repair plan from the accepted foundation."
            )
            self.store.transition_goal(
                goal.id,
                GoalStatus.REVISING,
                reason="retrying saved Ultra repair-plan checkpoint",
            )
            self.store.update_goal_metadata(
                goal.id,
                waiting_question="",
                waiting_on="",
                resume_action="",
            )
            return self.replan_ultra(feedback)
        ultra_run = self.active_ultra_run() if goal is not None else None
        resumable_failed_ultra = bool(
            goal is not None
            and goal.status == GoalStatus.BLOCKED
            and goal.metadata.get("ultra_run_id")
            and ultra_run is not None
            and ultra_run.status.value in {
                "running",
                "recovering",
                "blocked",
                # The engine may finish before the control-plane completion
                # gate rejects an unreviewed checkpoint. That mismatch is a
                # durable repair boundary, not a dead terminal session.
                "completed",
            }
            and not str(goal.metadata.get("waiting_question") or "").strip()
        )
        if goal is None or (
            goal.status != GoalStatus.PAUSED and not resumable_failed_ultra
        ):
            raise RuntimeStateError("goal is not paused")
        retry_not_before = goal.metadata.get("retry_not_before")
        if (
            retry_not_before is not None
            and float(retry_not_before) > time.time()
        ):
            remaining = max(1, int(float(retry_not_before) - time.time()))
            raise RuntimeStateError(
                f"saved provider retry is waiting for {remaining} more second(s)"
            )
        reconciled_side_effects = self._auto_reconcile_declared_tool_side_effects(goal)
        if reconciled_side_effects:
            goal = self.store.get_goal(goal.id)
            self.events.publish(
                "recovery",
                "Reconciled declared dependency lockfile side effects without replaying the install.",
                actions=list(reconciled_side_effects),
                mutation_replayed=False,
            )
        unresolved = self._unresolved_recovery_entities(goal.id)
        if unresolved:
            preview = ", ".join(unresolved[:5])
            suffix = " ..." if len(unresolved) > 5 else ""
            raise RuntimeStateError(
                "cannot resume while crash-window work is uncertain; inspect it and use "
                f"/resolve first ({preview}{suffix})"
            )
        desired = (
            GoalStatus.RUNNING
            if resumable_failed_ultra
            else GoalStatus(goal.metadata.get("resume_status", GoalStatus.RUNNING.value))
        )
        if desired in {
            GoalStatus.NEW,
            GoalStatus.PAUSED,
            GoalStatus.RECOVERING,
            GoalStatus.VERIFYING,
            GoalStatus.REVIEWING,
            GoalStatus.BLOCKED,
        }:
            desired = GoalStatus.RUNNING if goal.active_plan_revision else GoalStatus.DISCOVERING
        self.store.update_goal_metadata(goal.id, waiting_question="")
        self.store.update_goal_metadata(goal.id, no_progress_slices=0)
        self.store.update_goal_metadata(
            goal.id,
            retry_after_ms=0,
            retry_not_before=None,
            auto_retryable=False,
            provider_recovery={"state": "retrying", "automatic_fallback": False},
        )
        result = self.store.transition_goal(
            goal.id,
            desired,
            reason=(
                "resumed after a recoverable ULTRA engine failure"
                if resumable_failed_ultra
                else "resumed by user"
            ),
        )
        self.events.publish("phase", f"Goal resumed in {desired.value}.")
        ultra_run_id = str(goal.metadata.get("ultra_run_id", ""))
        if ultra_run_id and self.ultra_session is None:
            try:
                self.restore_ultra(ultra_run_id)
            except Exception:
                current = self.store.get_goal(goal.id)
                if current.status == GoalStatus.RUNNING:
                    self.store.transition_goal(
                        goal.id,
                        GoalStatus.PAUSED,
                        reason="ULTRA restore could not safely start",
                    )
                raise
            self.events.publish(
                "phase",
                "ULTRA scheduler rebuilt from the last durable evidence gate.",
            )
            return result
        if self.ultra_session is not None and self.ultra_session.running:
            self.ultra_session.resume()
            return result
        if desired in {GoalStatus.DISCOVERING, GoalStatus.REVISING}:
            self.generate_plan(
                "Resume the interrupted planning pass from durable goal state.",
                auto_approve_in_scope_repair=(
                    desired is GoalStatus.REVISING
                    and self.store.get_accepted_plan(goal.id) is not None
                ),
            )
            return self.active_goal() or self.store.get_goal(goal.id)
        return result

    def cancel(self, confirmation: str) -> Goal | SliceResult:
        if confirmation.strip().upper() != "CANCEL":
            raise RuntimeStateError("cancelling an unfinished goal requires ':cancel CANCEL'")
        goal = self.active_goal()
        if goal is None:
            session = self.store.get_workflow_session(self.session_id)
            state = dict(session.get("state", {}))
            pending = state.get("pending_semantic_turn")
            if not isinstance(pending, Mapping):
                raise RuntimeStateError("no active goal")
            cancelled = dict(pending)
            cancelled.update({"status": "cancelled", "result_status": "cancelled"})
            state["last_semantic_turn"] = cancelled
            state.pop("pending_semantic_turn", None)
            self.store.mutate_workflow_session(
                self.session_id,
                lambda current_state: {
                    "state": state,
                    "run_state": RunState.IDLE.value,
                },
                expected_revision=int(session.get("revision") or 0),
            )
            self.store.append_event(
                "semantic_turn.cancelled",
                entity_type="semantic_turn",
                entity_id=str(cancelled.get("turn_id") or ""),
                payload={"reason": "explicitly cancelled by user"},
            )
            return SliceResult("cancelled", "The saved request was cancelled.", completed=True)
        if self.ultra_session is not None:
            self.ultra_session.cancel()
        result = self.store.cancel_goal_and_session(
            goal.id,
            self.session_id,
            reason="explicitly cancelled by user",
        )
        self.events.publish("phase", "Goal cancelled by explicit user request.")
        return result

    def resolve_action(self, action_id: str, resolution: str, note: str) -> Any:
        goal = self.active_goal()
        if goal is None:
            raise RuntimeStateError("no unfinished goal has actions to resolve")
        safe_note = redact_text(note, 2_000)
        try:
            result: Any = self.store.resolve_action(
                action_id, resolution, safe_note, actor="user"
            )
            entity = "action"
        except NotFoundError:
            try:
                result = self.store.resolve_delegation(
                    action_id, resolution, safe_note, actor="user"
                )
                entity = "delegation"
            except NotFoundError:
                try:
                    from .ultra_models import ResultPackageV1, WorkNodeStatus

                    node = self.store.get_work_node(action_id)
                    if node.status is not WorkNodeStatus.UNCERTAIN:
                        raise RuntimeStateError(
                            f"ULTRA node {action_id} is not uncertain"
                        )
                    if resolution == "applied":
                        result = self.store.transition_work_node(
                            action_id,
                            WorkNodeStatus.COMPLETED,
                            result=ResultPackageV1(
                                summary=safe_note,
                                metadata={
                                    "success": True,
                                    "reconciled_by": "user",
                                    "resolution": resolution,
                                },
                            ),
                            checkpoint="reconciled",
                        )
                    else:
                        result = self.store.transition_work_node(
                            action_id,
                            WorkNodeStatus.PENDING,
                            error=None,
                            checkpoint="reconciled_not_run",
                        )
                    entity = "ULTRA node"
                except NotFoundError:
                    from .ultra_models import AgentRunStatus, ResultPackageV1

                    try:
                        agent = self.store.get_agent_run(action_id)
                    except NotFoundError:
                        lease = next(
                            (
                                item
                                for item in self.store.list_resource_leases()
                                if item.id == action_id
                            ),
                            None,
                        )
                        if lease is None:
                            raise NotFoundError(
                                f"recovery entity not found: {action_id}"
                            )
                        result = self.store.release_resource_lease(action_id)
                        entity = "ULTRA lease"
                    else:
                        if agent.status is not AgentRunStatus.UNCERTAIN:
                            raise RuntimeStateError(
                                f"ULTRA agent {action_id} is not uncertain"
                            )
                        result = self.store.update_agent_run(
                            action_id,
                            AgentRunStatus.COMPLETED
                            if resolution == "applied"
                            else AgentRunStatus.CANCELLED,
                            result=(
                                ResultPackageV1(
                                    summary=safe_note,
                                    metadata={
                                        "success": resolution == "applied",
                                        "reconciled_by": "user",
                                    },
                                )
                                if resolution == "applied"
                                else None
                            ),
                            error=None if resolution == "applied" else safe_note,
                        )
                        entity = "ULTRA agent"
        self.events.publish(
            "recovery",
            f"Resolved uncertain {entity} {action_id} as {resolution}: {safe_note}",
        )
        return result

    def checkpoint_interrupt(self) -> Goal | None:
        """Convert an asynchronous Ctrl-C window into explicit recoverable state."""
        recovery = self.store.recover_inflight()
        goal = self.active_goal()
        if goal is None:
            return None
        if goal.status == GoalStatus.RECOVERING:
            resume_status = (
                GoalStatus.RUNNING.value
                if goal.active_plan_revision is not None
                else GoalStatus.DISCOVERING.value
            )
            self.store.update_goal_metadata(
                goal.id,
                resume_status=resume_status,
                waiting_question=(
                    "Work was interrupted during an action. Inspect uncertain state before /resume; "
                    "the harness did not replay it."
                ),
            )
            goal = self.store.transition_goal(goal.id, GoalStatus.PAUSED, reason="user interrupted uncertain work")
        elif goal.status != GoalStatus.PAUSED:
            goal = self.pause("user interrupted the current work slice")
        self.events.publish(
            "checkpoint",
            "Interrupt checkpoint saved. No unfinished side effect was replayed. Use /resume to continue.",
            paused=True,
            uncertain_tasks=list(recovery.task_ids),
            uncertain_actions=list(recovery.action_ids),
        )
        return goal

    def _runtime_environment_payload(self) -> dict[str, str]:
        shell = os.environ.get("COMSPEC") if os.name == "nt" else "/bin/sh"
        return {
            "platform": platform.system(),
            "os_name": os.name,
            "shell": shell or ("cmd.exe" if os.name == "nt" else "/bin/sh"),
            "workspace": str(self.workspace),
            "note": (
                "run_bash is a legacy name and invokes cmd.exe; do not use POSIX "
                "heredoc syntax. Use python -c or an accepted in-scope verifier."
                if os.name == "nt"
                else "run_bash is a legacy name and invokes the platform shell shown here."
            ),
        }

    def _state_payload(self, goal: Goal, plan: Plan | None = None) -> dict[str, Any]:
        plan = plan or self.store.get_latest_plan(goal.id)
        evidence = self.store.list_evidence(goal.id)
        delegations = self.store.list_delegations(goal.id)
        actions = self.store.list_actions(goal.id)
        task_summaries = [] if plan is None else [
            {
                "id": task.id,
                "title": task.title,
                "status": task.status.value,
                "depends_on": list(task.depends_on),
                "risk": task.risk,
                "attempts": task.attempts,
                "requirement_refs": list(task.metadata.get("requirement_refs") or ()),
            }
            for task in plan.tasks
        ]
        focus_tasks: list[dict[str, Any]] = []
        if plan is not None:
            focus = [
                task for task in plan.tasks
                if task.status in {TaskStatus.IN_PROGRESS, TaskStatus.VERIFYING, TaskStatus.BLOCKED, TaskStatus.UNCERTAIN}
            ]
            focus.extend(
                task for task in plan.tasks
                if task.status in {TaskStatus.PENDING, TaskStatus.READY} and task not in focus
            )
            focus_tasks = [_task_dict(task) for task in focus[:8]]
        raw_claims = goal.metadata.get("resource_claims") or ()
        approved_resource_claims = [
            {
                "path": str(item.get("selector") or path).replace("\\", "/"),
                "resolved_paths": [
                    str(path).replace("\\", "/")
                    for path in item.get("resolved_paths", ())
                    if str(path).strip()
                ],
                "supports_tasks": [
                    str(task_id).strip().upper()
                    for task_id in item.get("supports_tasks", ())
                    if str(task_id).strip()
                ],
                "state": str(item.get("state") or "resolved"),
            }
            for item in raw_claims
            if isinstance(item, Mapping)
            for path in (str(item.get("selector") or "").strip(),)
            if path
        ]
        return {
            "goal": {
                "id": goal.id,
                "objective": goal.objective,
                "status": goal.status.value,
                "success_criteria": list(goal.success_criteria),
                "constraints": list(goal.constraints),
                "active_plan_revision": goal.active_plan_revision,
            },
            "runtime_environment": self._runtime_environment_payload(),
            "accepted_semantic_goal": dict(goal.metadata.get("semantic_goal") or {}),
            "local_continuation_policy": dict(
                goal.metadata.get("local_continuation_policy") or {}
            ),
            "plan": None
            if plan is None
            else {
                "revision": plan.revision,
                "status": plan.status.value,
                "summary": plan.summary,
                "fingerprint": plan.fingerprint,
                "applicability_evidence": list(plan.applicability_evidence),
                "execution_strategy": plan.execution_strategy,
                "expected_changes": list(plan.expected_changes),
                "tasks": task_summaries,
                "focus_task_details": focus_tasks,
            },
            "approved_resource_claims": approved_resource_claims,
            "mutation_guidance": (
                "For a mutating tool, use only a path listed in approved_resource_claims "
                "for the harness-selected task. An empty workspace is expected; create "
                "the accepted paths rather than inventing a different filename."
            ),
            "durable_memory_and_evidence": [
                {
                    "task_id": item.task_id,
                    "plan_revision": item.plan_revision,
                    "kind": item.kind,
                    "summary": item.summary[:500],
                    "verified": item.verified,
                }
                for item in evidence[-60:]
            ],
            "delegations": [
                {
                    "id": item.id,
                    "task_id": item.task_id,
                    "status": item.status.value,
                    "role": item.role.to_dict(),
                    "result": (item.result_summary or "")[:500],
                }
                for item in delegations[-30:]
            ],
            "recent_actions": [
                {
                    "id": item["id"],
                    "task_id": item["task_id"],
                    "tool": item["tool_name"],
                    "status": item["status"],
                    "risk": item["risk"],
                    "result": str(item["result_summary"] or "")[:500],
                }
                for item in actions[-20:]
            ],
            "limits": {
                "work_slice_steps": self.config.work_quantum_steps,
                "max_delegation_depth": self.config.max_delegation_depth,
                "note": "These bound one slice, not the durable goal's lifetime.",
            },
        }

    def _durable_progress_snapshot(self, goal_id: str) -> str:
        """Fingerprint only durable, outcome-relevant progress.

        Successful reads and repeated inspections are deliberately excluded.
        """

        def stable_verification_result(tool_name: object, raw: object) -> object:
            """Remove per-invocation handles from equivalent verification evidence."""

            name = str(tool_name or "")
            text = str(raw or "")
            if name not in {"preview_html", "inspect_preview"}:
                return text
            try:
                payload = json.loads(text)
            except (TypeError, json.JSONDecodeError):
                return text
            if not isinstance(payload, Mapping):
                return text
            return {
                "status": payload.get("status"),
                "http_status": payload.get("http_status"),
                "verification": payload.get("verification"),
                "console_errors": payload.get("console_errors", ()),
                "page_errors": payload.get("page_errors", ()),
                "network_errors": payload.get("network_errors", ()),
                "screenshot_captured": bool(payload.get("screenshot_path")),
            }

        goal = self.store.get_goal(goal_id)
        plan = self.store.get_latest_plan(goal_id)
        evidence = self.store.list_evidence(goal_id)
        actions = self.store.list_actions(goal_id)
        verification_tools = {
            "run_bash",
            "run_command",
            "preview_html",
            "inspect_preview",
            "poll_process",
            "read_process_output",
        }
        target = goal.metadata.get("quality_target", {})
        artifact_ids = (
            tuple(target.get("artifact_ids", ()))
            if isinstance(target, Mapping)
            else ()
        )
        payload = {
            "goal_status": goal.status.value,
            "active_plan_revision": goal.active_plan_revision,
            "accepted_plan": (
                plan.fingerprint
                if plan is not None and plan.status is PlanStatus.ACCEPTED
                else None
            ),
            "tasks": (
                []
                if plan is None
                else [
                    {
                        "id": task.id,
                        "status": task.status.value,
                        "attempts": task.attempts,
                    }
                    for task in plan.tasks
                ]
            ),
            "mutation_sequence": int(
                goal.metadata.get("mutation_sequence", 0) or 0
            ),
            "artifact_hashes": (
                self._current_artifact_hashes(artifact_ids)
                if artifact_ids
                else {}
            ),
            "verified_evidence": sorted(
                {
                    json.dumps(
                        {
                            "task_id": item.task_id,
                            "kind": item.kind,
                            "summary": item.summary,
                            "tool": item.data.get("tool"),
                            "path": item.data.get("path"),
                            "file_hash": item.data.get("file_hash"),
                            "result": stable_verification_result(
                                item.data.get("tool"),
                                item.data.get("result"),
                            ),
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                        default=str,
                    )
                    for item in evidence
                    if item.verified
                }
            ),
            "verification_actions": sorted(
                {
                    (
                        str(item.get("tool_name") or ""),
                        json.dumps(
                            stable_verification_result(
                                item.get("tool_name"),
                                item.get("result_summary"),
                            ),
                            ensure_ascii=False,
                            sort_keys=True,
                            default=str,
                        ),
                    )
                    for item in actions
                    if item.get("status") == "completed"
                    and item.get("tool_name") in verification_tools
                }
            ),
            "resolved_findings": [
                str(item.get("id") or "")
                for item in goal.metadata.get("refinement_actions", ())
                if isinstance(item, Mapping)
                and item.get("status") == "resolved"
            ],
            "reviewed_change_sets": [
                str(item.get("id") or "")
                for item in goal.metadata.get("goal_change_sets", ())
                if isinstance(item, Mapping)
                and item.get("review_status") == "passed"
                and item.get("integration_status") == "integrated"
            ],
        }
        return hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                default=str,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    def _current_task_id(self, plan: Plan) -> str | None:
        for task in plan.tasks:
            if task.status in {TaskStatus.IN_PROGRESS, TaskStatus.VERIFYING}:
                return task.id
        return None

    @staticmethod
    def _is_harness_resource_claim_task(task: Task) -> bool:
        """Identify a legacy checklist item for harness-owned resource leases.

        Resource claims are derived from accepted expected-change paths and are
        never user work. Older planners occasionally materialized a separate
        "resource claim" task; treating it as obsolete keeps those accepted
        sessions resumable without changing their product scope.
        """

        text = " ".join((task.title, task.description)).casefold()
        return any(
            marker in text
            for marker in ("resource claim", "accepted claim", "claim for")
        )

    def _repair_legacy_resource_claims(self, goal: Goal, plan: Plan) -> None:
        """Rebind persisted claims from legacy bookkeeping tasks to real work."""

        raw_claims = goal.metadata.get("resource_claims") or ()
        if not raw_claims:
            return
        changed = False
        claims: list[dict[str, Any]] = []
        for raw in raw_claims:
            if not isinstance(raw, Mapping):
                continue
            claim = dict(raw)
            path = str(claim.get("selector") or "").replace("\\", "/").casefold()
            matches: list[str] = []
            if path:
                for task in plan.tasks:
                    if self._is_harness_resource_claim_task(task):
                        continue
                    text = "\n".join(
                        (
                            task.title,
                            task.description,
                            *task.acceptance_criteria,
                            *task.verification,
                        )
                    ).replace("\\", "/").casefold()
                    if path in text:
                        matches.append(task.id)
            if matches and tuple(str(item).upper() for item in claim.get("supports_tasks", ())) != tuple(matches):
                claim["supports_tasks"] = matches
                changed = True
            claims.append(claim)
        if changed:
            self.store.update_goal_metadata(
                goal.id,
                resource_claims=claims,
                resource_claims_repaired=True,
            )

    def _activate_ready_task(self, goal: Goal, plan: Plan) -> tuple[Plan, Task | None]:
        """Bind the slice to one dependency-ready task without model cooperation.

        Weak models frequently start using workspace tools before emitting the
        bookkeeping-only ``update_task(in_progress)`` call.  That used to leave
        otherwise authoritative tool evidence unscoped (``task_id=None``), so a
        later completion attempt could never satisfy the evidence gate.  Task
        selection is a deterministic scheduler decision and belongs here.
        """

        # Compatibility repair for plans produced before resource claims were
        # made harness-owned. This transition is deterministic bookkeeping; it
        # does not mark any product work complete or create evidence.
        for task in plan.tasks:
            if (
                task.status in {TaskStatus.PENDING, TaskStatus.READY}
                and self._is_harness_resource_claim_task(task)
            ):
                self.store.transition_task(
                    goal.id,
                    plan.revision,
                    task.id,
                    TaskStatus.OBSOLETE,
                    note="resource claims are harness-owned and are not execution tasks",
                    actor="harness",
                )
        plan = self.store.get_latest_plan(goal.id)
        self._repair_legacy_resource_claims(goal, plan)
        active_id = self._current_task_id(plan)
        if active_id is not None:
            return plan, next(item for item in plan.tasks if item.id == active_id)
        selected = first_ready_task(plan.tasks)
        if selected is None:
            return plan, None
        activated = self.store.transition_task(
            goal.id,
            plan.revision,
            selected.id,
            TaskStatus.IN_PROGRESS,
            note="selected automatically by the dependency-ready harness scheduler",
            actor="harness",
        )
        refreshed = self.store.get_latest_plan(goal.id)
        self.store.append_event(
            "execution.task_selected",
            goal_id=goal.id,
            entity_type="task",
            entity_id=activated.id,
            payload={"reason": "all dependencies are complete; first ready task in plan order", "activated": True},
        )
        return refreshed, next(item for item in refreshed.tasks if item.id == activated.id)

    def _execute_workspace_tool(
        self,
        goal: Goal,
        call: ToolCall,
        *,
        task_id: str | None,
        actor: str,
    ) -> str:
        if call.name not in _external_schema_map():
            return f"Error: unknown workspace tool '{call.name}'"
        goal = self.store.get_goal(goal.id)
        active_task_label = str(task_id or "")
        active_plan = self.store.get_latest_plan(goal.id)
        if active_plan is not None and task_id:
            active_task = next((item for item in active_plan.tasks if item.id == task_id), None)
            if active_task is not None:
                active_task_label = f"{active_task.id} · {active_task.title}"
        args = call.args if isinstance(call.args, dict) else {}
        if call.name == "list_files" and not str(args.get("path") or "").strip():
            # Omitted, empty, and explicit-root paths mean the same operation.
            # Canonicalizing here lets retry detection see them as identical.
            args = {**args, "path": "."}
        path_normalizations: list[str] = []
        tool_spec = tools.get_spec(call.name)
        for field_name in (() if tool_spec is None else tool_spec.path_fields):
            authored_path = str(args.get(field_name) or "").strip()
            if not authored_path or not Path(authored_path).is_absolute():
                continue
            resolved_path = Path(authored_path).resolve(strict=False)
            if not resolved_path.is_relative_to(self.workspace):
                return (
                    f"Error: {call.name} {field_name} is outside the configured "
                    "workspace; use a workspace-relative path."
                )
            relative_path = resolved_path.relative_to(self.workspace).as_posix()
            args = {**args, field_name: relative_path or "."}
            path_normalizations.append(
                f"/{field_name} absolute in-workspace path normalized to "
                f"{relative_path or '.'}"
            )
        if path_normalizations:
            self.store.append_event(
                "tool_payload.normalized",
                goal_id=goal.id,
                payload={
                    "tool": call.name,
                    "actor": actor,
                    "actions": path_normalizations,
                },
            )
        if call.name == "apply_patch":
            authored_base = str(args.get("base_path") or ".").strip() or "."
            base_candidate = Path(authored_base)
            if base_candidate.is_absolute():
                resolved_base = base_candidate.resolve(strict=False)
                if not resolved_base.is_relative_to(self.workspace):
                    return (
                        "Error: apply_patch base_path is outside the configured "
                        "workspace; use a workspace-relative base_path."
                    )
                relative_base = resolved_base.relative_to(self.workspace).as_posix()
                args = {**args, "base_path": relative_base or "."}
                self.store.append_event(
                    "tool_payload.normalized",
                    goal_id=goal.id,
                    payload={
                        "tool": "apply_patch",
                        "actor": actor,
                        "actions": [
                            "/base_path absolute in-workspace path normalized to "
                            f"{relative_base or '.'}"
                        ],
                    },
                )
        try:
            args = dict(tools.validate_tool_arguments(tool_spec.schema, args))
        except (TypeError, ValueError) as exc:
            message = f"Error: invalid arguments: {redact_text(exc, 1_000)}"
            self.store.append_event(
                "tool_contract.rejected",
                goal_id=goal.id,
                payload={
                    "actor": actor,
                    "received": [call.name],
                    "stage": "execution_arguments",
                    "error": redact_text(exc, 1_000),
                    "approval_requested": False,
                    "workspace_mutated": False,
                },
            )
            return message
        applicability_error = tools.applicability_issue(
            call.name, args, self.workspace
        )
        if applicability_error:
            self.store.append_event(
                "tool_contract.rejected",
                goal_id=goal.id,
                payload={
                    "actor": actor,
                    "received": [call.name],
                    "stage": "execution_applicability",
                    "error": applicability_error,
                    "approval_requested": False,
                    "workspace_mutated": False,
                },
            )
            return f"Error: {applicability_error}"
        scoped_name = f"{actor}:{call.name}"
        journal_args = {
            "_harness_actor": actor,
            # A denied call can be invalid for the current task's resource
            # lease yet become valid when the scheduler advances to the task
            # that owns that path.  Scope retry/no-progress identity to the
            # durable task so an earlier denial cannot poison later work.
            "_harness_task_id": task_id,
            "_harness_plan_revision": goal.active_plan_revision,
            "_harness_mutation_sequence": int(
                goal.metadata.get("mutation_sequence", 0) or 0
            ),
            "arguments": redact_data(args),
        }
        approach_fingerprint = hashlib.sha256(
            json.dumps({"tool": call.name, "args": redact_data(args)}, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        persisted_failures = tuple(goal.metadata.get("failed_attempts", ()))
        recent_non_improving = [
            item
            for item in goal.metadata.get("strategy_attempts", ())
            if isinstance(item, Mapping)
            and item.get("outcome") in {"unchanged", "regressed"}
        ][-3:]
        if (
            call.name not in READ_ONLY_TOOLS
            and
            len(recent_non_improving) == 3
            and len(
                {
                    str(item.get("fingerprint") or item.get("strategy_fingerprint"))
                    for item in recent_non_improving
                }
            )
            == 3
        ):
            self.store.update_goal_metadata(
                goal.id,
                strategy_reinspection_required=True,
                convergence_state="replanning",
            )
            if goal.status is GoalStatus.RUNNING:
                self.store.transition_goal(
                    goal.id,
                    GoalStatus.REVISING,
                    reason=(
                        "three materially different strategies produced no "
                        "improvement; fresh inspection and planning are required"
                    ),
                )
            self.store.append_event(
                "strategy.replan_required",
                goal_id=goal.id,
                payload={
                    "reason": "three materially different strategies did not improve evidence",
                    "task_id": task_id,
                },
            )
            return (
                "Error: three non-improving strategies require fresh repository "
                "inspection and a revised plan before further execution."
            )
        equivalent_count = sum(
            1 for attempt in persisted_failures
            if isinstance(attempt, Mapping) and attempt.get("approach_fingerprint") == approach_fingerprint
        )
        if equivalent_count >= self.weak_model_policy.max_equivalent_failed_approaches:
            self.store.append_event(
                "approach.change_forced", goal_id=goal.id,
                payload={
                    "approach_fingerprint": approach_fingerprint,
                    "equivalent_failures": equivalent_count,
                    "reason": "maximum equivalent failed approaches reached; reinspection or a materially different mechanism is required",
                    "rules": self.weak_model_policy.applied_rules("retry"),
                },
            )
            return "Error: equivalent failed approach limit reached; re-inspect, split the task, or use a materially different implementation strategy."
        if actor == "planner":
            journal_args["_harness_goal_attempt"] = int(goal.metadata.get("goal_attempt", 0))
        if equivalent_count == 0 and self.store.count_recent_identical_actions(
            goal.id,
            call.name,
            journal_args,
            scan_limit=self.config.repeated_action_limit + 2,
        ) >= self.config.repeated_action_limit:
            return (
                "Error: persistent no-progress circuit breaker: this actor repeated the identical "
                "action across checkpoints; inspect prior results and choose a different approach."
            )
        decision = self._watchdog.check(scoped_name, args)
        if decision.stalled and equivalent_count == 0:
            return f"Error: {decision.reason}"
        risk = TOOL_RISK.get(call.name, "unknown")
        normal_requirement = tools.requires_approval(call.name, args)
        needs_approval = (
            self.permission_adapter.requires_approval(normal_requirement)
            if self.permission_adapter is not None
            else normal_requirement
        )
        if call.name == "open_path" or (call.name == "preview_html" and bool(args.get("open_browser", True))):
            # Full only relaxes sandboxed workspace actions. Host GUI launch
            # still needs direct user intent/approval.
            needs_approval = True
        action_id: str | None = None
        approval_error = ""
        approval_granted = True
        if needs_approval:
            try:
                approval_granted = self._approval_allowed(
                    call.name,
                    copy.deepcopy(args),
                    risk,
                )
            except Exception as exc:
                approval_granted = False
                approval_error = str(exc)
        if needs_approval and not approval_granted:
            action_id = self.store.begin_action(
                goal.id,
                call.name,
                journal_args,
                task_id=task_id,
                risk=risk,
                mutating=call.name in MUTATING_TOOLS,
            )
            result = (
                "Approval could not be collected; the action was not executed."
                if approval_error
                else "Permission denied by the user. The action was not executed."
            )
            self.store.complete_action(
                action_id,
                result,
                status="failed" if approval_error else "denied",
            )
            self._checkpoint_tool_approval_boundary(
                goal,
                tool=call.name,
                args=args,
                risk=risk,
                action_id=action_id,
                reason=(
                    f"Approval for {call.name} could not be collected."
                    if approval_error
                    else f"Approval for {call.name} was not granted."
                ),
            )
            self._watchdog.record(scoped_name, args, result)
            return result

        action_id = self.store.begin_action(
            goal.id,
            call.name,
            journal_args,
            task_id=task_id,
            risk=risk,
            mutating=call.name in MUTATING_TOOLS,
        )
        self.store.update_goal_metadata(
            goal.id,
            last_tool=call.name,
            waiting_on=("user" if needs_approval else "tool"),
            workspace_mutated=bool(goal.metadata.get("workspace_mutated", False)),
            heartbeat_at=time.time(),
        )
        self.store.append_event(
            "tool.started",
            goal_id=goal.id,
            entity_type="action",
            entity_id=action_id,
            payload={"tool": call.name, "task_id": task_id, "action_id": action_id, "waiting_on": "tool", "heartbeat_at": time.time()},
        )
        self.events.publish(
            "execution.started",
            f"{call.name} started",
            goal_id=goal.id,
            objective=goal.objective,
            current_task=active_task_label,
            current_task_id=str(task_id or ""),
            active_actor=actor,
            tool=call.name,
            action_id=action_id,
            task_id=task_id,
            phase="working",
            waiting_on="tool",
            heartbeat_at=time.time(),
        )
        if call.name in {"run_bash", "run_command", "preview_html", "open_path"}:
            self.store.append_event(
                "process.waiting",
                goal_id=goal.id,
                entity_type="action",
                entity_id=action_id,
                payload={"tool": call.name, "action_id": action_id, "command": str(args.get("command") or ""), "heartbeat_at": time.time()},
            )
            self.events.publish(
                "process.waiting",
                f"Waiting for {call.name} to finish",
                goal_id=goal.id,
                objective=goal.objective,
                current_task=active_task_label,
                current_task_id=str(task_id or ""),
                active_actor=actor,
                tool=call.name,
                action_id=action_id,
                phase="waiting_for_process",
                waiting_on="process",
                heartbeat_at=time.time(),
            )
        leased_paths: tuple[str, ...] = ()
        effective_mutation_paths: tuple[str, ...] = ()
        mutation_journal_ids: tuple[str, ...] = ()
        direct_path = (
            str(args.get("path") or "").strip().replace("\\", "/")
            if call.name in {"write_file", "edit_file", "materialize_artifact"}
            else ""
        )
        requested_paths = {direct_path} if direct_path else set()
        if call.name == "apply_patch":
            requested_paths.update(
                tools.apply_patch.patch_paths(
                    str(args.get("patch") or ""),
                    str(args.get("base_path") or ".").strip(),
                )
            )
        if call.name in MUTATING_TOOLS:
            leased_paths = self.store.lease_resource_claims(goal.id, task_id)
            footprint = tools.mutation_footprint(call.name, args, leased_paths)
            invalid_derived_paths = []
            for relative in footprint.derived_paths:
                candidate = (self.workspace / relative).resolve(strict=False)
                if (
                    not candidate.is_relative_to(self.workspace)
                    or not self.store._journal_path_allowed(relative)
                ):
                    invalid_derived_paths.append(relative)
            if invalid_derived_paths:
                result = (
                    "Error: the tool declared an unsafe mutation side effect: "
                    + ", ".join(sorted(invalid_derived_paths))
                )
                self.store.complete_action(action_id, result, status="denied")
                self.store.release_resource_claims(goal.id, task_id)
                return result
            effective_mutation_paths = footprint.effective_paths
            if footprint.derived_paths:
                self.store.append_event(
                    "mutation.footprint_derived",
                    goal_id=goal.id,
                    entity_type="action",
                    entity_id=action_id,
                    payload={
                        "tool": call.name,
                        "accepted_paths": list(footprint.accepted_paths),
                        "derived_paths": list(footprint.derived_paths),
                        "footprint_fingerprint": footprint.fingerprint,
                    },
                )
            mutation_journal_ids = self.store.prepare_mutation_journal(
                goal.id,
                task_id,
                effective_mutation_paths,
            )
            if requested_paths - set(effective_mutation_paths):
                accepted_for_task = tuple(
                    sorted(set(effective_mutation_paths))
                )
                result = (
                    "Error: mutation target(s) are not covered by accepted, "
                    "repository-evidenced resource claims: "
                    + ", ".join(
                        sorted(requested_paths - set(effective_mutation_paths))
                    )
                    + (
                        "; accepted paths for this task: "
                        + ", ".join(accepted_for_task)
                        if accepted_for_task
                        else "; no accepted paths are leased for this task; "
                        "use the accepted plan paths or propose a scoped plan change"
                    )
                )
                self.store.complete_action(action_id, result, status="denied")
                self.store.finish_mutation_journal(
                    mutation_journal_ids, applied=False
                )
                self.store.release_resource_claims(goal.id, task_id)
                return result
        pre_path = str(args.get("path", "")).strip() if call.name in {"write_file", "edit_file", "materialize_artifact"} else ""
        pre_bytes: bytes | None = None
        pre_hash: str | None = None
        mutation_before = (
            self._chat_workspace_hashes(self.workspace)
            if call.name in MUTATING_TOOLS else {}
        )
        mutation_observed = False
        if pre_path:
            pre_candidate = (self.workspace / pre_path).resolve(strict=False)
            if pre_candidate.is_file() and pre_candidate.is_relative_to(self.workspace):
                pre_bytes = pre_candidate.read_bytes()
                pre_hash = hashlib.sha256(pre_bytes).hexdigest()
        try:
            with tools.workspace_context(
                self.workspace,
                session_id=self.session_id,
                goal_id=goal.id,
                task_id=task_id or "unassigned",
            ):
                if call.name in {"run_bash", "run_command"} and self.permission_adapter is not None:
                    shell_command = str(args.get("command", ""))
                    if (
                        call.name == "run_command"
                        and str(args.get("cwd", ".")).strip() not in {"", "."}
                        and self.permission_adapter.access_level.value == "full"
                    ):
                        shell_command = f"cd -- {shlex.quote(str(args['cwd']))} && {shell_command}"
                    raw_result = self.permission_adapter.run_shell(
                        shell_command,
                        self.workspace,
                        normal_runner=lambda command: tools.run_tool(
                            call.name, {**args, "command": command}
                        ),
                    )
                else:
                    raw_result = tools.run_tool(call.name, args)
            result = redact_text(raw_result, 50_000)
            if call.name in {"run_bash", "run_command"}:
                shell_exit = re.search(r"(?im)^exit code:\s*(-?\d+)", result)
                if shell_exit and int(shell_exit.group(1)) != 0:
                    result = "Error: shell command failed; " + result
            terminal = "failed" if result.startswith("Error:") else "completed"
            self.store.complete_action(action_id, redact_text(result, 2_000), status=terminal)
            self.store.update_goal_metadata(
                goal.id,
                last_tool=call.name,
                waiting_on="" if terminal == "completed" else "user",
                workspace_mutated=bool(mutation_observed or goal.metadata.get("workspace_mutated", False)),
                heartbeat_at=time.time(),
            )
            self.store.append_event(
                "tool.completed" if terminal == "completed" else "tool.failed",
                goal_id=goal.id,
                entity_type="action",
                entity_id=action_id,
                payload={"tool": call.name, "action_id": action_id, "status": terminal, "heartbeat_at": time.time(), "result": redact_text(result, 1_000)},
            )
            mutation_after = (
                self._chat_workspace_hashes(self.workspace)
                if terminal == "completed" and call.name in MUTATING_TOOLS else mutation_before
            )
            actual_changed_files = [
                path for path in sorted(set(mutation_before) | set(mutation_after))
                if mutation_before.get(path) != mutation_after.get(path)
            ]
            mutation_observed = bool(actual_changed_files)
            unleased_changes = [
                path
                for path in actual_changed_files
                if path not in set(effective_mutation_paths)
            ]
            if (
                terminal == "completed"
                and call.name in MUTATING_TOOLS
                and unleased_changes
            ):
                result = (
                    "Error: mutation escaped accepted resource leases; user "
                    "inspection is required for: "
                    + ", ".join(unleased_changes)
                )
                terminal = "failed"
                self.store.mark_action_uncertain(action_id, result)
                latest = self.store.get_goal(goal.id)
                uncertain = list(latest.metadata.get("uncertain_actions", ()))
                uncertain.append(
                    {
                        "action_id": action_id,
                        "paths": unleased_changes,
                        "reason": "mutation escaped accepted resource leases",
                    }
                )
                self.store.update_goal_metadata(
                    goal.id,
                    uncertain_actions=uncertain[-50:],
                    waiting_question=(
                        "A mutating action changed paths outside its accepted "
                        "leases. Inspect and resolve the uncertain action before resuming."
                    ),
                    resume_status=GoalStatus.RUNNING.value,
                )
                current_goal = self.store.get_goal(goal.id)
                if current_goal.status is GoalStatus.RUNNING:
                    self.store.transition_goal(
                        goal.id,
                        GoalStatus.PAUSED,
                        reason="mutation escaped accepted resource leases",
                    )
            if terminal == "completed" and call.name in MUTATING_TOOLS and actual_changed_files:
                refreshed = self.store.get_goal(goal.id)
                sequence = int(refreshed.metadata.get("mutation_sequence", 0)) + 1
                self.store.update_goal_metadata(
                    goal.id,
                    mutation_sequence=sequence,
                    convergence_state="reverifying",
                    latest_evaluation_stale=True,
                )
                self.store.append_event(
                    "quality_evaluation.invalidated", goal_id=goal.id,
                    payload={
                        "mutation_sequence": sequence,
                        "action_id": action_id,
                        "rules": self.weak_model_policy.applied_rules("mutation"),
                    },
                )
                post_hash: str | None = None
                diff = ""
                if pre_path:
                    post_candidate = (self.workspace / pre_path).resolve(strict=False)
                    if post_candidate.is_file() and post_candidate.is_relative_to(self.workspace):
                        post_bytes = post_candidate.read_bytes()
                        post_hash = hashlib.sha256(post_bytes).hexdigest()
                        before_text = (pre_bytes or b"").decode("utf-8", errors="replace").splitlines()
                        after_text = post_bytes.decode("utf-8", errors="replace").splitlines()
                        diff = "\n".join(difflib.unified_diff(before_text, after_text, fromfile=f"a/{pre_path}", tofile=f"b/{pre_path}", lineterm=""))
                latest_for_change = self.store.get_goal(goal.id)
                changes = list(latest_for_change.metadata.get("goal_change_sets", ()))
                active_refinements = [
                    action.get("id") for action in latest_for_change.metadata.get("refinement_actions", ())
                    if isinstance(action, Mapping) and action.get("status") == "pending"
                ]
                change_set = {
                    "id": f"goal-changeset-{len(changes) + 1:04d}",
                    "version": 1,
                    "responsible_agent": actor,
                    "parent_task": task_id,
                    "refinement_actions": active_refinements,
                    "quality_target_id": latest_for_change.metadata.get("quality_target", {}).get("id") if isinstance(latest_for_change.metadata.get("quality_target"), Mapping) else None,
                    "affected_quality_dimensions": sorted({
                        dimension
                        for action in latest_for_change.metadata.get("refinement_actions", ())
                        if isinstance(action, Mapping) and action.get("id") in active_refinements
                        for dimension in action.get("affected_dimensions", ())
                    }),
                    "changed_files": actual_changed_files,
                    "pre_hashes": {path: mutation_before.get(path) for path in actual_changed_files},
                    "post_hashes": {path: mutation_after.get(path) for path in actual_changed_files},
                    "diff": redact_text(diff, 30_000),
                    "tool_action_ids": [action_id],
                    "verification_evidence_ids": [],
                    "review_status": "pending",
                    "integration_status": "pending",
                    "mutation_sequence": sequence,
                }
                changes.append(change_set)
                successful_strategy = StrategyAttemptV1(
                    task_id=task_id or "unassigned",
                    hypothesis="The accepted task contract predicts this mutation advances its criteria.",
                    approach=json.dumps(
                        {"tool": call.name, "arguments": redact_data(args)},
                        ensure_ascii=False,
                        sort_keys=True,
                        default=str,
                    ),
                    evidence_refs=(f"action:{action_id}", change_set["id"]),
                    outcome="improved",
                )
                prior_strategies = list(
                    latest_for_change.metadata.get("strategy_attempts", ())
                )
                prior_strategies.append(successful_strategy.to_dict())
                self.store.update_goal_metadata(
                    goal.id,
                    goal_change_sets=changes,
                    strategy_attempts=prior_strategies[-100:],
                )
                self.store.append_event(
                    "change_set.created", goal_id=goal.id,
                    entity_type="change_set", entity_id=change_set["id"],
                    payload={"task_id": task_id, "files": change_set["changed_files"], "mutation_sequence": sequence},
                )
                path_value = str(args.get("path", "")).strip()
                if path_value:
                    candidate = (self.workspace / path_value).resolve(strict=False)
                    if candidate.is_file() and candidate.is_relative_to(self.workspace):
                        indexed = self.repository_index.update(candidate.relative_to(self.workspace).as_posix())
                        snapshot = {
                            candidate.relative_to(self.workspace).as_posix(): [
                                {
                                    "kind": item.kind, "name": item.name, "start": item.start,
                                    "end": item.end, "file_hash": item.file_hash,
                                }
                                for item in indexed
                            ]
                        }
                        latest = self.store.get_goal(goal.id)
                        previous = dict(latest.metadata.get("artifact_index", {}))
                        previous.update(snapshot)
                        self.store.update_goal_metadata(goal.id, artifact_index=previous)
                        self.store.append_event(
                            "artifact_index.updated", goal_id=goal.id,
                            payload={"path": path_value, "entries": len(indexed), "file_hash": indexed[0].file_hash},
                        )
            if terminal == "completed" and actor != "planner":
                evidence_data: dict[str, Any] = {
                    "action_id": action_id,
                    "tool": call.name,
                    "arguments": redact_data(args),
                    "result": redact_text(result, 4_000),
                }
                path_value = str(args.get("path", "")).strip()
                if path_value:
                    candidate = (self.workspace / path_value).resolve(strict=False)
                    if candidate.is_file() and candidate.is_relative_to(self.workspace):
                        evidence_data.update(
                            {
                                "path": candidate.relative_to(self.workspace).as_posix(),
                                "file_hash": hashlib.sha256(candidate.read_bytes()).hexdigest(),
                                "file_exists": True,
                            }
                        )
                self.store.add_evidence(
                    goal_id=goal.id,
                    plan_revision=goal.active_plan_revision,
                    task_id=task_id,
                    kind="tool_result",
                    summary=f"{call.name} completed with authoritative harness evidence",
                    data=evidence_data,
                    created_by="harness",
                    verified=True,
                )
        except (KeyboardInterrupt, SystemExit):
            # Deliberately leave the action running; restart recovery will mark
            # the crash-window side effect uncertain instead of replaying it.
            raise
        except Exception as exc:
            result = f"Error: tool harness failure: {type(exc).__name__}: {redact_text(exc, 500)}"
            self.store.complete_action(action_id, result, status="failed")
        if call.name in MUTATING_TOOLS:
            self.store.finish_mutation_journal(
                mutation_journal_ids,
                applied=mutation_observed or not result.startswith("Error:"),
            )
            self.store.release_resource_claims(goal.id, task_id)
        if result.startswith("Error:"):
            domain = (
                FailureDomain.PERMISSION if "permission" in result.casefold()
                else FailureDomain.SYNTAX if "syntax" in result.casefold()
                else FailureDomain.TEST if "assert" in result.casefold() or "test" in result.casefold()
                else FailureDomain.RUNTIME
            )
            mentioned_paths = tuple(dict.fromkeys(
                match.replace("\\", "/")
                for match in re.findall(r"(?i)([A-Za-z0-9_./\\-]+\.(?:py|js|ts|tsx|jsx|html|css|json|toml|yaml|yml))(?::\d+)?", result)
            ))
            explicit_path = str(args.get("path", "")).strip()
            signature_paths = tuple(dict.fromkeys(filter(None, (explicit_path, *mentioned_paths))))
            file_hashes: dict[str, str] = {}
            for relative in signature_paths:
                candidate = (self.workspace / relative).resolve(strict=False)
                if candidate.is_file() and candidate.is_relative_to(self.workspace):
                    file_hashes[relative] = hashlib.sha256(candidate.read_bytes()).hexdigest()
            exit_match = re.search(r"(?i)(?:exit(?:\s+code)?|returned)\s*[:=]?\s*(-?\d+)", result)
            signature = ErrorSignature(
                domain=domain,
                operation=call.name,
                command=str(args.get("command", "")),
                exit_code=int(exit_match.group(1)) if exit_match else None,
                normalized_message=normalize_error_message(result),
                paths=signature_paths,
                file_hashes=file_hashes,
            )
            latest = self.store.get_goal(goal.id)
            failures = list(latest.metadata.get("failed_hypotheses", ()))
            failures.append({"signature": signature.fingerprint, "operation": call.name, "message": signature.normalized_message})
            attempts = list(latest.metadata.get("failed_attempts", ()))
            strategy_attempt = StrategyAttemptV1(
                task_id=task_id or "unassigned",
                hypothesis=signature.normalized_message,
                approach=json.dumps(
                    {"tool": call.name, "arguments": redact_data(args)},
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                ),
                evidence_refs=(f"action:{action_id}",),
                outcome="unchanged",
                next_strategy=(
                    "re-inspect and select a materially different mechanism"
                    if equivalent_count >= 1
                    else "use failure evidence to revise the hypothesis"
                ),
            )
            attempts.append(
                {
                    **strategy_attempt.to_dict(),
                    "signature": signature.fingerprint,
                    "strategy_fingerprint": strategy_attempt.fingerprint,
                    "approach_fingerprint": approach_fingerprint,
                    "operation": call.name,
                }
            )
            metadata_update: dict[str, Any] = {
                "failed_hypotheses": failures[-20:],
                "failed_attempts": attempts[-50:],
                "strategy_attempts": [
                    *list(latest.metadata.get("strategy_attempts", ())),
                    strategy_attempt.to_dict(),
                ][-100:],
            }
            contract_data = latest.metadata.get("goal_contract")
            if isinstance(contract_data, Mapping):
                contract = GoalContractV1.from_dict(contract_data)
                for relative in signature_paths:
                    candidate = (self.workspace / relative).resolve(strict=False)
                    if candidate.is_file() and candidate.is_relative_to(self.workspace):
                        self.repository_index.update(candidate.relative_to(self.workspace).as_posix())
                context_slice = self.repository_index.context_slice(
                    signature.normalized_message,
                    max_entries=20,
                    budget_chars=20_000,
                )
                related = context_slice.entries
                updated_contract = GoalContractV1(**{
                    **contract.to_dict(),
                    "failed_hypotheses": (*contract.failed_hypotheses, signature.normalized_message)[-20:],
                    "file_symbol_scope": tuple(dict.fromkeys((
                        *contract.file_symbol_scope,
                        *(f"{entry.path}:{entry.kind}:{entry.name}" for entry in related),
                        *signature_paths,
                    ))),
                    "task_boundaries": (
                        f"Diagnose {signature.domain.value} failure {signature.fingerprint[:12]}",
                        "Change only components implicated by authoritative failure evidence",
                        "Rerun the narrow failing check before broader regression verification",
                    ),
                })
                metadata_update.update(
                    goal_contract=updated_contract.to_dict(),
                    goal_contract_fingerprint=updated_contract.fingerprint,
                    error_context_slice={
                        "query": signature.normalized_message,
                        "size_chars": context_slice.size_chars,
                        "omitted_entries": context_slice.omitted_entries,
                        "callers": {key: list(value) for key, value in context_slice.callers.items()},
                        "callees": {key: list(value) for key, value in context_slice.callees.items()},
                        "dependencies": {key: list(value) for key, value in context_slice.dependencies.items()},
                    },
                )
            self.store.update_goal_metadata(goal.id, **metadata_update)
            self.store.append_event(
                "error_signature.created", goal_id=goal.id,
                payload={"fingerprint": signature.fingerprint, "domain": domain.value, "operation": call.name},
            )
        self._watchdog.record(scoped_name, args, result)
        return result

    def _reset_dependants(self, goal: Goal, plan: Plan, task_id: str, *, actor: str) -> None:
        task_id = task_id.upper()
        by_id = {task.id: task for task in plan.tasks}
        if task_id not in by_id:
            raise NotFoundError(f"task not found: {task_id}")
        invalidated = {task_id}
        changed = True
        while changed:
            changed = False
            for item in plan.tasks:
                if item.id not in invalidated and any(dep in invalidated for dep in item.depends_on):
                    invalidated.add(item.id)
                    changed = True
        for item in reversed(plan.tasks):
            if item.id in invalidated and item.id != task_id:
                self.store.transition_task(
                    goal.id,
                    plan.revision,
                    item.id,
                    TaskStatus.PENDING,
                    note=f"invalidated because prerequisite {task_id} was reopened by {actor}",
                    actor=actor,
                )

    def _control_update_task(self, goal: Goal, plan: Plan, args: dict[str, Any]) -> str:
        mapping = {
            "pending": TaskStatus.PENDING,
            "in_progress": TaskStatus.IN_PROGRESS,
            "done": TaskStatus.COMPLETED,
            "blocked": TaskStatus.BLOCKED,
        }
        target = mapping[args["status"]]
        if target == TaskStatus.COMPLETED and not args["evidence"]:
            return "Error: done requires concrete evidence; verify the work first."
        if target == TaskStatus.COMPLETED:
            task_evidence = [
                item
                for item in self.store.list_evidence(goal.id, task_id=args["task_id"])
                if item.plan_revision == plan.revision
                and (item.verified or item.created_by == "user")
            ]
            if not task_evidence:
                self._inherit_dependency_verification_evidence(
                    goal, plan, args["task_id"]
                )
                task_evidence = [
                    item
                    for item in self.store.list_evidence(
                        goal.id, task_id=args["task_id"]
                    )
                    if item.plan_revision == plan.revision
                    and (item.verified or item.created_by == "user")
                ]
            if not task_evidence:
                return (
                    "Error: done requires authoritative evidence bound to this task. "
                    "Keep it in_progress and run its required workspace verification first."
                )
        if target == TaskStatus.BLOCKED and not args["note"].strip():
            return "Error: blocked requires a concrete blocker note."
        try:
            if target == TaskStatus.PENDING:
                self._reset_dependants(goal, plan, args["task_id"], actor="coordinator")
            task = self.store.transition_task(
                goal.id,
                plan.revision,
                args["task_id"],
                target,
                note=args["note"],
                evidence=[redact_text(item, 2_000) for item in args["evidence"]],
                actor="coordinator",
            )
        except (StateStoreError, ValueError) as exc:
            return f"Error: checklist update rejected: {redact_text(exc, 1_000)}"
        return f"Checklist {task.id} -> {task.status.value}. Durable state updated."

    def _inherit_dependency_verification_evidence(
        self,
        goal: Goal,
        plan: Plan,
        task_id: str,
    ) -> None:
        """Bind fresh prerequisite command evidence to a verification-only task.

        Models sometimes run the final test command while finishing the last
        implementation task. Requiring an identical rerun solely because the
        scheduler advanced one task later wastes work and can trap weaker
        models in bookkeeping loops. Reuse is deliberately narrow: the target
        must have no expected mutation, the evidence must come from a direct
        dependency, the exact command must occur in the target contract, and
        no accepted workspace mutation may follow the command.
        """

        normalized_id = str(task_id).upper()
        task = next((item for item in plan.tasks if item.id == normalized_id), None)
        if task is None or not task.depends_on:
            return
        if any(
            normalized_id
            in {str(value).upper() for value in change.get("supports_tasks", ())}
            for change in plan.expected_changes
            if isinstance(change, Mapping)
        ):
            return
        contract = " ".join(
            (
                task.title,
                task.description,
                *task.acceptance_criteria,
                *task.verification,
            )
        ).casefold()
        actions = list(self.store.list_actions(goal.id))
        action_index = {
            str(item.get("id") or ""): index
            for index, item in enumerate(actions)
        }
        mutation_action_ids = {
            str(action_id)
            for change_set in goal.metadata.get("goal_change_sets", ())
            if isinstance(change_set, Mapping)
            for action_id in change_set.get("tool_action_ids", ())
        }
        last_mutation_index = max(
            (action_index.get(action_id, -1) for action_id in mutation_action_ids),
            default=-1,
        )
        by_action = {
            str(item.get("id") or ""): item for item in actions
        }
        candidates = []
        for dependency_id in task.depends_on:
            candidates.extend(
                item
                for item in self.store.list_evidence(
                    goal.id, task_id=dependency_id
                )
                if item.plan_revision == plan.revision and item.verified
            )
        for evidence in reversed(candidates):
            data = dict(evidence.data)
            if data.get("tool") not in {"run_command", "run_bash"}:
                continue
            command = str(dict(data.get("arguments") or {}).get("command") or "").strip()
            normalized_command = " ".join(command.casefold().split())
            if not normalized_command or normalized_command not in " ".join(contract.split()):
                continue
            action_id = str(data.get("action_id") or "")
            action = by_action.get(action_id)
            if (
                action is None
                or action.get("status") != "completed"
                or action_index.get(action_id, -1) < last_mutation_index
            ):
                continue
            result = str(data.get("result") or action.get("result_summary") or "")
            if result.startswith("Error:") or not re.search(
                r"(?im)^exit code:\s*0\b", result
            ):
                continue
            inherited = self.store.add_evidence(
                goal_id=goal.id,
                plan_revision=plan.revision,
                task_id=normalized_id,
                kind="inherited_verification",
                summary=(
                    f"Fresh successful verification from dependency "
                    f"{evidence.task_id}: {command}"
                ),
                data={
                    "source_evidence_id": evidence.id,
                    "source_task_id": evidence.task_id,
                    "action_id": action_id,
                    "command": command,
                    "fresh_after_mutation_index": last_mutation_index,
                },
                created_by="harness",
                verified=True,
            )
            self.store.append_event(
                "verification_evidence.inherited",
                goal_id=goal.id,
                entity_type="evidence",
                entity_id=inherited.id,
                payload={
                    "task_id": normalized_id,
                    "source_task_id": evidence.task_id,
                    "source_evidence_id": evidence.id,
                    "action_id": action_id,
                },
            )
            return

    def _record_memory(self, goal: Goal, plan: Plan, args: dict[str, Any]) -> str:
        item = self.store.add_evidence(
            goal_id=goal.id,
            plan_revision=plan.revision,
            kind="memory",
            summary=redact_text(args["fact"], 2_000),
            data={"source": redact_text(args["source"], 1_000)},
            created_by="coordinator",
        )
        return f"Durable memory recorded ({item.id})."

    def _inspect_task(self, goal: Goal, plan: Plan, args: dict[str, Any]) -> str:
        task_id = args["task_id"].upper()
        task = next((item for item in plan.tasks if item.id == task_id), None)
        if task is None:
            return f"Error: task not found in accepted plan r{plan.revision}: {task_id}"
        evidence = [
            item
            for item in self.store.list_evidence(goal.id, task_id=task_id)
            if item.plan_revision == plan.revision
        ]
        offset = args["evidence_offset"]
        limit = args["evidence_limit"]
        page = evidence[offset : offset + limit]
        return json.dumps(
            {
                "task": _task_dict(task),
                "evidence_total": len(evidence),
                "evidence_offset": offset,
                "evidence_returned": len(page),
                "has_more": offset + len(page) < len(evidence),
                "evidence": [
                    {
                        "id": item.id,
                        "kind": item.kind,
                        "summary": redact_text(item.summary, 2_000),
                        "verified": item.verified,
                        "created_by": item.created_by,
                        "observed_result": (
                            str(item.data.get("result"))[:2_000]
                            if isinstance(item.data, Mapping)
                            and item.data.get("result") is not None
                            else ""
                        ),
                        "observed_command": (
                            str((item.data.get("arguments") or {}).get("command"))[:2_000]
                            if isinstance(item.data, Mapping)
                            and isinstance(item.data.get("arguments"), Mapping)
                            and item.data.get("arguments", {}).get("command") is not None
                            else ""
                        ),
                    }
                    for item in page
                ],
            },
            ensure_ascii=False,
        )

    def _request_user(self, goal: Goal, args: dict[str, Any]) -> str:
        self.store.update_goal_metadata(
            goal.id,
            waiting_question=redact_text(args["question"], 2_000),
            waiting_reason=redact_text(args["reason"], 2_000),
            resume_status=GoalStatus.RUNNING.value,
        )
        self.store.transition_goal(goal.id, GoalStatus.PAUSED, reason="coordinator requested user input")
        return "Goal paused durably for user input."

    def _coerce_role(self, role_text: str, task: str, allowed_tools: list[str]) -> RoleProfile:
        compact = " ".join(role_text.split())
        name = compact.split(".", 1)[0][:120] or "task-specific worker"
        return RoleProfile(
            name=name,
            mission=compact,
            expertise=(),
            constraints=("Stay within the delegated assignment.",),
            deliverables=(task,),
            tool_policy={"allowed_tools": allowed_tools},
        )

    def _delegate(
        self,
        goal: Goal,
        plan: Plan,
        args: dict[str, Any],
        *,
        parent_id: str | None = None,
        depth: int = 1,
    ) -> dict[str, Any]:
        if depth > self.config.max_delegation_depth:
            return {"outcome": "blocked", "summary": "delegation depth limit reached", "evidence": []}
        if self._delegations_this_slice >= self.config.max_delegations_per_slice:
            return {"outcome": "blocked", "summary": "per-slice delegation limit reached", "evidence": []}
        allowed = []
        external = set(_external_schema_map())
        for name in args["allowed_tools"]:
            if name in external or name == "delegate_task":
                if name not in allowed:
                    allowed.append(name)
        if not allowed:
            return {"outcome": "blocked", "summary": "no valid worker tools were requested", "evidence": []}
        current_task = args["task_id"].upper()
        task_by_id = {item.id: item for item in plan.tasks}
        assigned_task = task_by_id.get(current_task)
        if assigned_task is None:
            return {"outcome": "blocked", "summary": f"unknown accepted-plan task {current_task}", "evidence": []}
        if assigned_task.status in {TaskStatus.COMPLETED, TaskStatus.OBSOLETE, TaskStatus.CANCELLED}:
            return {"outcome": "blocked", "summary": f"task {current_task} is already {assigned_task.status.value}", "evidence": []}
        unfinished_dependencies = [
            dependency
            for dependency in assigned_task.depends_on
            if task_by_id[dependency].status not in {TaskStatus.COMPLETED, TaskStatus.OBSOLETE}
        ]
        if unfinished_dependencies:
            return {
                "outcome": "blocked",
                "summary": f"task {current_task} has unfinished dependencies: {', '.join(unfinished_dependencies)}",
                "evidence": [],
            }
        role = self._coerce_role(args["role"], args["task"], allowed)
        delegation = self.store.create_delegation(
            Delegation(
                goal_id=goal.id,
                task_id=current_task,
                plan_revision=plan.revision,
                parent_id=parent_id,
                brief=args["task"],
                role=role,
                metadata={"success_criteria": args["success_criteria"], "depth": depth},
            )
        )
        self.store.transition_delegation(delegation.id, DelegationStatus.IN_PROGRESS)
        self._delegations_this_slice += 1
        self.events.publish("delegation", f"{delegation.id}: {role.name}", task_id=current_task, depth=depth)

        worker_schemas = [*_schemas(name for name in allowed if name in external), *WORKER_SCHEMAS]
        if "delegate_task" in allowed and depth < self.config.max_delegation_depth:
            worker_schemas.append(DELEGATE_TASK)
        conversation: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": state_envelope(
                    {
                        "root_objective": goal.objective,
                        "accepted_plan_revision": plan.revision,
                        "task_id": current_task,
                        "assignment": args["task"],
                        "success_criteria": args["success_criteria"],
                        "context": args["context"],
                        "allowed_tools": allowed,
                        "runtime_environment": self._runtime_environment_payload(),
                    },
                    "WORKER_BRIEF",
                ),
            }
        ]
        report: dict[str, Any] | None = None
        try:
            for step in range(1, self.config.subagent_steps + 1):
                turn = self._call_provider(
                    conversation,
                    worker_schemas,
                    subagent_system_prompt(role.mission, depth, self.config.max_delegation_depth),
                    actor=f"worker:{delegation.id[-8:]}",
                    step=step,
                )
                conversation.append(turn.to_message())
                for call in turn.tool_calls:
                    self.events.publish("tool_call", call.name, args=redact_data(call.args), actor=delegation.id)
                    if call.name == "return_work":
                        try:
                            report = validate_control_call(call.name, call.args)
                            result = "Structured worker report accepted."
                        except ControlValidationError as exc:
                            result = f"Error: invalid worker report: {exc}"
                    elif call.name == "delegate_task" and "delegate_task" in allowed:
                        try:
                            child_args = validate_control_call(call.name, call.args)
                            child = self._delegate(goal, plan, child_args, parent_id=delegation.id, depth=depth + 1)
                            result = json.dumps(child, ensure_ascii=False)
                        except (ControlValidationError, Exception) as exc:
                            result = f"Error: child delegation failed: {redact_text(exc, 1_000)}"
                    elif call.name in allowed and call.name in external:
                        result = self._execute_workspace_tool(goal, call, task_id=current_task, actor=delegation.id)
                    else:
                        result = f"Error: tool '{call.name}' is outside this worker's policy."
                    conversation.append({"role": "tool", "id": call.id, "name": call.name, "content": result})
                    self.events.publish("tool_result", result, tool=call.name, actor=delegation.id)
                if report is not None:
                    break
                if not turn.tool_calls:
                    conversation.append(
                        {"role": "user", "content": "Prose is not a worker result. Verify the assignment and call return_work."}
                    )
        except Exception as exc:
            error = f"{type(exc).__name__}: {redact_text(exc, 1_000)}"
            self.store.transition_delegation(delegation.id, DelegationStatus.FAILED, error=error)
            return {"outcome": "blocked", "summary": error, "evidence": []}

        if report is None:
            report = {
                "outcome": "partial",
                "summary": "Worker reached its bounded slice without a valid return_work report.",
                "evidence": [],
                "changed_paths": [],
                "remaining_risks": ["Worker result is incomplete; coordinator must inspect current state."],
                "proposed_subtasks": [],
            }
        status = DelegationStatus.COMPLETED if report["outcome"] == "success" else DelegationStatus.FAILED
        self.store.transition_delegation(
            delegation.id,
            status,
            result_summary=redact_text(report["summary"], 4_000),
            error=None if status == DelegationStatus.COMPLETED else redact_text(report["summary"], 1_000),
        )
        for item in report.get("evidence", []):
            self.store.add_evidence(
                goal_id=goal.id,
                plan_revision=plan.revision,
                task_id=current_task,
                kind="delegation",
                summary=redact_text(item, 2_000),
                data={"delegation_id": delegation.id, "role": role.name},
                created_by=delegation.id,
            )
        return report

    def _completion_precheck(self, goal: Goal, plan: Plan) -> str | None:
        quality_state = str(goal.metadata.get("convergence_state", ""))
        if goal.metadata.get("quality_target") and quality_state in {"below_target", "refining"}:
            return f"quality target is {quality_state}; concrete refinement and fresh verification are required"
        if goal.metadata.get("latest_evaluation_stale") and quality_state == "converged":
            return "the claimed converged evaluation predates the latest mutation"
        if plan.status != PlanStatus.ACCEPTED or goal.active_plan_revision != plan.revision:
            return "the latest plan revision is not accepted"
        incomplete = [
            task.id
            for task in plan.tasks
            if task.status not in {TaskStatus.COMPLETED, TaskStatus.OBSOLETE}
        ]
        if incomplete:
            return f"unfinished checklist items: {', '.join(incomplete)}"
        for task in plan.tasks:
            if task.status == TaskStatus.COMPLETED:
                task_evidence = [
                    item for item in self.store.list_evidence(goal.id, task_id=task.id)
                    if item.plan_revision == plan.revision
                ]
                if not task_evidence:
                    return f"completed task {task.id} has no evidence"
                if not any(item.verified or item.created_by == "user" for item in task_evidence):
                    return f"completed task {task.id} has only unverified model-authored prose; authoritative harness or user evidence is required"
        uncertain = self.store.list_actions(goal.id, status="uncertain")
        if uncertain:
            return f"{len(uncertain)} action(s) have uncertain crash-window state"
        uncertain_workers = [
            item
            for item in self.store.list_delegations(goal.id)
            if item.status == DelegationStatus.UNCERTAIN
        ]
        if uncertain_workers:
            return f"{len(uncertain_workers)} delegation(s) have uncertain crash-window state"
        target = goal.metadata.get("quality_target", {})
        artifact_ids = (
            tuple(target.get("artifact_ids", ()))
            if isinstance(target, Mapping)
            else ()
        )
        _checks, visual_blocker = self._visual_operational_checks(
            goal,
            artifact_ids,
        )
        if visual_blocker:
            return visual_blocker
        return None

    def _visual_operational_checks(
        self,
        goal: Goal,
        artifact_ids: Iterable[str],
    ) -> tuple[tuple[str, ...], str | None]:
        html_artifacts = tuple(
            str(path)
            for path in artifact_ids
            if str(path).casefold().endswith((".html", ".htm"))
        )
        if not html_artifacts:
            return (), None
        hashes = self._current_artifact_hashes(html_artifacts)
        missing = [
            path
            for path in html_artifacts
            if hashes.get(path) in {None, "MISSING"}
        ]
        if missing:
            return (), (
                "visual artifact gate is missing expected HTML artifact(s): "
                + ", ".join(missing)
            )
        previews: list[dict[str, Any]] = []
        for action in self.store.list_actions(goal.id):
            if (
                action.get("tool_name") != "preview_html"
                or action.get("status") != "completed"
            ):
                continue
            try:
                payload = json.loads(str(action.get("result_summary") or ""))
                journal = json.loads(str(action.get("args_json") or "{}"))
            except (TypeError, json.JSONDecodeError):
                continue
            if isinstance(payload, Mapping):
                previews.append(
                    {
                        **dict(payload),
                        "_harness_mutation_sequence": int(
                            journal.get("_harness_mutation_sequence", -1) or 0
                        ),
                    }
                )
        current_sequence = int(goal.metadata.get("mutation_sequence", 0) or 0)
        passing = next(
            (
                item
                for item in reversed(previews)
                if int(item.get("http_status", 0) or 0) == 200
                and item.get("verification") == "passed"
                and not item.get("console_errors")
                and not item.get("page_errors")
                and str(item.get("screenshot_path") or "").strip()
                and Path(str(item.get("screenshot_path"))).exists()
                and int(item.get("_harness_mutation_sequence", -1))
                == current_sequence
            ),
            None,
        )
        if passing is None:
            return (
                ("artifact_hashes",),
                "HTML completion requires a successful real-browser preview "
                "with HTTP 200, a screenshot, and zero console/page errors",
            )
        return (
            (
                "artifact_hashes",
                "browser_runtime",
                "http_200",
                "console_errors_zero",
                "page_errors_zero",
                "screenshot_captured",
            ),
            None,
        )

    def _current_artifact_hashes(self, artifact_ids: Iterable[str]) -> dict[str, str]:
        hashes: dict[str, str] = {}
        for artifact_id in artifact_ids:
            relative = str(artifact_id).strip() or "workspace"
            candidate = self.workspace if relative == "workspace" else (self.workspace / relative).resolve(strict=False)
            if not candidate.exists() or not candidate.is_relative_to(self.workspace):
                hashes[relative] = "MISSING"
                continue
            if candidate.is_file():
                hashes[relative] = hashlib.sha256(candidate.read_bytes()).hexdigest()
                continue
            members = []
            for path in sorted(candidate.rglob("*")):
                if not path.is_file() or any(part in {".git", ".coding-agent", "__pycache__", ".pytest_cache"} for part in path.parts):
                    continue
                member = path.relative_to(self.workspace).as_posix()
                members.append((member, hashlib.sha256(path.read_bytes()).hexdigest()))
            hashes[relative] = hashlib.sha256(json.dumps(members, separators=(",", ":")).encode("utf-8")).hexdigest()
        return hashes

    def _review_completion(
        self,
        goal: Goal,
        plan: Plan,
        claim: dict[str, Any],
    ) -> dict[str, Any] | None:
        goal_level_evidence = [
            item
            for item in self.store.list_evidence(goal.id)
            if item.task_id is None and item.plan_revision in {None, plan.revision}
        ]
        conversation: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": state_envelope(
                    {
                        "goal": {
                            "id": goal.id,
                            "objective": goal.objective,
                            "success_criteria": list(goal.success_criteria),
                            "constraints": list(goal.constraints),
                        },
                        "accepted_plan": {
                            "revision": plan.revision,
                            "fingerprint": plan.fingerprint,
                            "summary": plan.summary,
                            "applicability_evidence": list(plan.applicability_evidence),
                            "execution_strategy": plan.execution_strategy,
                            "expected_changes": list(plan.expected_changes),
                            "task_count": len(plan.tasks),
                        },
                        "completion_claim": claim,
                        "goal_level_evidence": [
                            {
                                "kind": item.kind,
                                "summary": item.summary[:1_000],
                                "verified": item.verified,
                                "observed_result": (
                                    str(item.data.get("result"))[:2_000]
                                    if isinstance(item.data, Mapping)
                                    and item.data.get("result") is not None
                                    else ""
                                ),
                                "observed_command": (
                                    str((item.data.get("arguments") or {}).get("command"))[:2_000]
                                    if isinstance(item.data, Mapping)
                                    and isinstance(item.data.get("arguments"), Mapping)
                                    and item.data.get("arguments", {}).get("command") is not None
                                    else ""
                                ),
                            }
                            for item in goal_level_evidence[-20:]
                        ],
                        "inspection": (
                            "Every task follows in complete review chunks. Use inspect_task for paginated "
                            "evidence beyond each task's recent sample."
                        ),
                    },
                    "FINAL_REVIEW_INPUT",
                    max_chars=40_000,
                ),
            }
        ]
        for index, task in enumerate(plan.tasks):
            task_evidence = [
                item
                for item in self.store.list_evidence(goal.id, task_id=task.id)
                if item.plan_revision == plan.revision
            ]
            conversation.append(
                {
                    "role": "user",
                    "content": state_envelope(
                        {
                            "chunk": index + 1,
                            "of": len(plan.tasks),
                            "task": _task_dict(task),
                            "evidence_total": len(task_evidence),
                            "recent_evidence": [
                                {
                                    "id": item.id,
                                    "kind": item.kind,
                                    "summary": item.summary[:600],
                                    "verified": item.verified,
                                    "created_by": item.created_by,
                                    "observed_result": (
                                        str(item.data.get("result"))[:2_000]
                                        if isinstance(item.data, Mapping)
                                        and item.data.get("result") is not None
                                        else ""
                                    ),
                                    "observed_command": (
                                        str((item.data.get("arguments") or {}).get("command"))[:2_000]
                                        if isinstance(item.data, Mapping)
                                        and isinstance(item.data.get("arguments"), Mapping)
                                        and item.data.get("arguments", {}).get("command") is not None
                                        else ""
                                    ),
                                }
                                for item in task_evidence[-3:]
                            ],
                        },
                        "FINAL_REVIEW_TASK",
                        max_chars=30_000,
                    ),
                }
            )
        inspection_schemas = [*_schemas(READ_ONLY_TOOLS), *REVIEWER_SCHEMAS]
        verdict_schemas = [
            schema
            for schema in REVIEWER_SCHEMAS
            if schema.get("function", {}).get("name") == "submit_review"
        ]
        review_steps = max(1, int(self.config.review_steps))
        for step in range(1, review_steps + 1):
            final_verdict_turn = step == review_steps
            if final_verdict_turn:
                conversation.append(
                    {
                        "role": "user",
                        "content": (
                            "FINAL REVIEW VERDICT TURN. Inspection is closed. "
                            "Using the complete accepted-plan chunks, durable "
                            "evidence, and inspection results already in this "
                            "conversation, call submit_review exactly once now. "
                            "Return fail with actionable issues when evidence is "
                            "insufficient; do not request another inspection."
                        ),
                    }
                )
            turn = self._call_provider(
                conversation,
                verdict_schemas if final_verdict_turn else inspection_schemas,
                REVIEWER_SYSTEM_PROMPT,
                actor="independent-reviewer",
                step=step,
            )
            conversation.append(turn.to_message())
            for call in turn.tool_calls:
                if call.name == "submit_review":
                    try:
                        verdict = validate_control_call(call.name, call.args)
                        if verdict["verdict"] == "pass" and verdict["issues"]:
                            result = "Error: a passing verdict cannot include unresolved issues."
                        elif verdict["verdict"] == "pass" and set(
                            item.upper() for item in verdict["checked_task_ids"]
                        ) != {task.id for task in plan.tasks}:
                            result = (
                                "Error: pass must explicitly cover every accepted task in checked_task_ids."
                            )
                        else:
                            conversation.append({"role": "tool", "id": call.id, "name": call.name, "content": "Review verdict accepted."})
                            return verdict
                    except ControlValidationError as exc:
                        result = f"Error: invalid review verdict: {exc}"
                elif call.name == "inspect_task":
                    try:
                        inspect_args = validate_control_call(call.name, call.args)
                        result = self._inspect_task(goal, plan, inspect_args)
                    except ControlValidationError as exc:
                        result = f"Error: invalid task inspection: {exc}"
                elif call.name in READ_ONLY_TOOLS:
                    result = self._execute_workspace_tool(goal, call, task_id=None, actor="reviewer")
                else:
                    result = f"Error: final review cannot use '{call.name}'."
                conversation.append({"role": "tool", "id": call.id, "name": call.name, "content": result})
            if not turn.tool_calls:
                conversation.append(
                    {"role": "user", "content": "A prose opinion is not a completion verdict. Inspect evidence and call submit_review."}
                )
            elif not final_verdict_turn:
                conversation.append(
                    {
                        "role": "user",
                        "content": (
                            f"Inspection result recorded. {review_steps - step} "
                            "review turn(s) remain. Submit a verdict as soon as "
                            "the evidence is sufficient; the final turn permits "
                            "submit_review only."
                        ),
                    }
                )
        return None

    def _finish_goal(self, goal: Goal, plan: Plan, args: dict[str, Any]) -> str:
        blocked = self._completion_precheck(goal, plan)
        if blocked:
            self.store.append_event(
                "completion.rejected", goal_id=goal.id,
                payload={
                    "reason": blocked,
                    "convergence_state": goal.metadata.get("convergence_state"),
                    "mutation_sequence": goal.metadata.get("mutation_sequence", 0),
                    "rules": self.weak_model_policy.applied_rules("completion"),
                },
            )
            return f"Error: completion gate rejected: {blocked}. Continue the goal."
        self.store.transition_goal(goal.id, GoalStatus.VERIFYING, reason="completion requested; deterministic gate passed")
        self.store.transition_goal(goal.id, GoalStatus.REVIEWING, reason="fresh-context independent review started")
        current = self.store.get_goal(goal.id)
        try:
            verdict = self._review_completion(current, plan, args)
        except ProviderUnavailableError as exc:
            self.store.transition_goal(goal.id, GoalStatus.RUNNING, reason="review provider unavailable")
            return f"Error: independent review could not run: {redact_text(exc, 1_000)}. Goal remains active."
        if verdict is None:
            self.store.transition_goal(goal.id, GoalStatus.RUNNING, reason="review reached slice limit without verdict")
            return "Error: independent review produced no valid verdict. Goal remains active."
        self.store.add_evidence(
            goal_id=goal.id,
            plan_revision=plan.revision,
            kind="final_review",
            summary=redact_text(verdict["summary"], 4_000),
            data={"verdict": verdict["verdict"], "issues": redact_data(verdict["issues"])},
            created_by="independent-reviewer",
            verified=verdict["verdict"] == "pass",
        )
        if verdict["verdict"] == "pass":
            fresh_goal = self.store.get_goal(goal.id)
            mutation_sequence = int(fresh_goal.metadata.get("mutation_sequence", 0))
            evidence = self.store.list_evidence(goal.id)
            hashes = {
                str(item.data.get("path")): str(item.data.get("file_hash"))
                for item in evidence
                if item.data.get("path") and item.data.get("file_hash")
            }
            target = fresh_goal.metadata.get("quality_target", {})
            target_artifacts = tuple(target.get("artifact_ids", ())) if isinstance(target, Mapping) else ()
            current_hashes = self._current_artifact_hashes(target_artifacts)
            hashes.update(current_hashes)
            visual_target = any(str(path).casefold().endswith((".html", ".htm")) for path in target_artifacts)
            provider_profile = getattr(self.provider, "capability_profile", None)
            vision_available = bool(
                getattr(provider_profile, "vision_support", False)
                or getattr(getattr(self.provider, "capabilities", None), "supports_vision", False)
            )
            evaluator_profile = {
                "version": 1,
                "test_runner_available": bool(shutil.which("pytest") or (self.workspace / ".venv" / "Scripts" / "pytest.exe").exists()),
                "static_analyzer_available": bool(shutil.which("ruff") or shutil.which("mypy")),
                "browser_available": importlib.util.find_spec("playwright") is not None,
                "screenshot_available": importlib.util.find_spec("playwright") is not None,
                "vision_evaluator_available": vision_available,
                "accessibility_checker_available": bool(shutil.which("axe")),
                "performance_profiler_available": bool(shutil.which("py-spy")),
                "user_review_required": False,
                "subjective_visual_quality_verified": (
                    not visual_target or vision_available
                ),
                "routing_order": [
                    "deterministic_verification", "static_analysis", "runtime_integration",
                    "artifact_structure", "independent_review",
                    (
                        "vision_evaluation"
                        if vision_available
                        else "subjective_quality_unverified"
                    ),
                ],
            }
            disposition = (
                CompletionDisposition.COMPLETED_WITH_LIMITATIONS
                if visual_target and not vision_available
                else CompletionDisposition.VERIFIED
            )
            limitations = (
                (
                    "Subjective visual quality was not machine-verified because "
                    "no vision evaluator was available.",
                )
                if disposition
                is CompletionDisposition.COMPLETED_WITH_LIMITATIONS
                else ()
            )
            operational_checks, operational_blocker = (
                self._visual_operational_checks(fresh_goal, target_artifacts)
            )
            if operational_blocker:
                self.store.transition_goal(
                    goal.id,
                    GoalStatus.RUNNING,
                    reason="visual operational completion gate needs repair",
                )
                return f"Error: {operational_blocker}. Goal remains active."
            convergence_state = (
                "converged_with_limitations"
                if disposition
                is CompletionDisposition.COMPLETED_WITH_LIMITATIONS
                else "converged"
            )
            authoritative_ids = [item.id for item in evidence if item.verified or item.created_by == "user"]
            dimensions = [dict(item) for item in target.get("dimensions", ())] if isinstance(target, Mapping) else []
            dimension_scores = []
            for dimension in dimensions:
                requires_vision = dimension.get("evaluation_method") == "vision_and_runtime"
                proven = not requires_vision or vision_available
                score = 1.0 if proven else 0.0
                dimension_scores.append({
                    "dimension_id": dimension.get("id"),
                    "score": score,
                    "passed": score >= float(dimension.get("minimum_score", 0.8)),
                    "evidence_ids": authoritative_ids,
                    "confidence": "high" if proven else "low",
                    "finding": None if proven else "subjective visual quality is not provable with available evaluators",
                })
                dimension["latest_artifact_hash"] = hashlib.sha256(
                    json.dumps(hashes, sort_keys=True).encode("utf-8")
                ).hexdigest() if hashes else None
                dimension["latest_mutation_sequence"] = mutation_sequence
            overall_score = (
                sum(item["score"] for item in dimension_scores) / len(dimension_scores)
                if dimension_scores else 0.0
            )
            target = {**dict(target), "dimensions": dimensions}
            previous_evaluation = fresh_goal.metadata.get("latest_evaluation")
            evaluation_record = {
                "version": 1,
                "target_id": target.get("id"),
                "rubric_version": target.get("version", 1),
                "mutation_sequence": mutation_sequence,
                "artifact_hashes": hashes,
                "change_set_ids": [item.get("id") for item in fresh_goal.metadata.get("goal_change_sets", ()) if isinstance(item, Mapping)],
                "evaluators": ["authoritative_tool_evidence", "independent-reviewer"] + (["vision"] if vision_available else []),
                "evaluator_capability_profile": evaluator_profile,
                "evidence_ids": authoritative_ids,
                "hard_gate_results": {str(gate): True for gate in target.get("hard_gates", ())},
                "scores": dimension_scores,
                "overall_score": overall_score,
                "confidence": (
                    "limited"
                    if disposition
                    is CompletionDisposition.COMPLETED_WITH_LIMITATIONS
                    else "high"
                ),
                "previous_overall_score": previous_evaluation.get("overall_score") if isinstance(previous_evaluation, Mapping) else None,
                "evaluated_at_unix": time.time(),
                "contract_fingerprint": fresh_goal.metadata.get("goal_contract_fingerprint"),
            }
            reviewed_change_sets = []
            for change_set in fresh_goal.metadata.get("goal_change_sets", ()):
                if not isinstance(change_set, Mapping):
                    continue
                reviewed_change_sets.append({
                    **dict(change_set),
                    "verification_evidence_ids": [item.id for item in evidence if item.verified],
                    "review_status": "passed",
                    "integration_status": "integrated",
                })
            self.store.update_goal_metadata(
                goal.id,
                convergence_state=convergence_state,
                latest_evaluation_stale=False,
                latest_evaluation=evaluation_record,
                quality_target=target,
                evaluator_capability_profile=evaluator_profile,
                goal_change_sets=reviewed_change_sets,
                completion_disposition=disposition.value,
                completion_limitations=list(limitations),
                passed_operational_checks=list(operational_checks),
            )
            self.store.append_event(
                "quality_convergence.decided", goal_id=goal.id,
                payload={"state": convergence_state, "mutation_sequence": mutation_sequence, "artifact_hashes": hashes},
            )
            self._checkpoint_accepted_goal(
                self.store.get_goal(goal.id),
                source="independent_final_review",
            )
            self.store.transition_goal(
                goal.id,
                GoalStatus.COMPLETED,
                reason=(
                    "all operational gates and independent review passed; "
                    "subjective visual evaluation is recorded as a limitation"
                    if disposition
                    is CompletionDisposition.COMPLETED_WITH_LIMITATIONS
                    else "all checklist evidence passed independent final review"
                ),
                metadata={
                    "completion_summary": redact_text(args["summary"], 4_000),
                    "completion_disposition": disposition.value,
                    "completion_limitations": list(limitations),
                    "passed_operational_checks": list(operational_checks),
                    "evaluator_capability_profile": evaluator_profile,
                },
            )
            self._record_global_learning(
                self.store.get_goal(goal.id),
                succeeded=True,
                evidence_ref=f"goal:{goal.id}:evaluation:{evaluation_record.get('evaluated_at_unix')}",
            )
            completion_session = self.store.get_workflow_session(self.session_id)
            self.store.mutate_workflow_session(
                self.session_id,
                lambda current_state: {
                    "state": {
                        **dict(current_state.get("state") or {}),
                        "plan_revision": plan.revision,
                        "completion": "evidence_gate_passed",
                        "completion_disposition": disposition.value,
                        "completion_limitations": list(limitations),
                    },
                    "goal_id": current_state.get("goal_id") or goal.id,
                    "session_mode": SessionMode.GOAL.value,
                    "plan_state": PlanState.APPROVED.value,
                    "run_state": RunState.COMPLETED.value,
                },
                expected_revision=int(completion_session.get("revision") or 0),
            )
            self.events.publish("phase", "Goal completed after evidence gate and independent review.")
            return (
                "Goal completed with limitations. All operational gates and "
                "independent review passed; subjective visual quality was not "
                "claimed as verified."
                if disposition
                is CompletionDisposition.COMPLETED_WITH_LIMITATIONS
                else "Goal completed. The harness accepted the independent review."
            )

        self._record_global_learning(
            self.store.get_goal(goal.id),
            succeeded=False,
            evidence_ref=f"goal:{goal.id}:review-failed",
            blocker=str(verdict.get("summary") or "independent review failed"),
        )
        repair_tasks = []
        existing = list(plan.tasks)
        for issue in verdict["issues"]:
            repair_tasks.append(
                {
                    "id": self._next_task_id([*existing, *repair_tasks]),
                    "title": issue["title"],
                    "description": issue["details"],
                    "acceptance_criteria": issue["acceptance_criteria"],
                    "verification": [f"Independently verify repair: {criterion}" for criterion in issue["acceptance_criteria"]],
                    "depends_on": [],
                    "risk": issue["severity"],
                    "origin": "reviewer",
                }
            )
        if not repair_tasks:
            repair_tasks.append(
                {
                    "id": self._next_task_id(existing),
                    "title": "Resolve failed independent review",
                    "description": verdict["summary"],
                    "acceptance_criteria": ["A fresh independent review returns pass with direct evidence."],
                    "verification": ["Repeat the completion audit after addressing the review summary."],
                    "depends_on": [],
                    "risk": "high",
                    "origin": "reviewer",
                }
            )
        new_plan = self.revise_plan(
            reason=f"independent review failed: {verdict['summary']}",
            add=repair_tasks,
            proposed_by="reviewer",
            inherit_approved_scope=True,
        )
        refreshed = self.store.get_goal(goal.id)
        actions = list(refreshed.metadata.get("refinement_actions", ()))
        for repair in repair_tasks:
            actions.append({
                "id": f"refinement-{len(actions) + 1:03d}",
                "feedback": verdict["summary"],
                "affected_dimensions": ["functional_correctness", "regression_safety"],
                "affected_components": [],
                "objective": repair["description"],
                "acceptance_criteria": repair["acceptance_criteria"],
                "verification": repair["verification"],
                "status": "pending",
                "source": "independent-reviewer",
            })
        self.store.update_goal_metadata(
            goal.id,
            refinement_actions=actions,
            convergence_state="refining",
        )
        self.store.append_event(
            "refinement_cycle.started", goal_id=goal.id,
            payload={"source_evaluation": "independent-reviewer", "repair_tasks": [item["id"] for item in repair_tasks], "plan_revision": new_plan.revision},
        )
        if self._repair_revision_is_in_scope(plan, new_plan, repair_tasks):
            accepted_repair = self.approve_plan(
                new_plan.revision,
                approved_by="harness-quality-convergence",
            )
            self.store.update_goal_metadata(
                goal.id,
                refinement_actions=actions,
                convergence_state="refining",
            )
            return (
                f"Independent review found {len(repair_tasks)} deficiency task(s). "
                f"Harness-approved in-scope repair plan r{accepted_repair.revision}; "
                "Goal refinement continues autonomously."
            )
        self.store.update_goal_metadata(
            goal.id,
            refinement_actions=actions,
            convergence_state="scope_expansion_pending",
            waiting_question=(
                "Independent review proposed repair work that adds a dependency, "
                "external effect, sensitive permission, or path outside the approved "
                "scope. Review and approve the new plan revision."
            ),
        )
        return (
            f"Independent review found {len(repair_tasks)} deficiency task(s), "
            f"but plan r{new_plan.revision} expands the approved scope and awaits "
            "explicit approval."
        )

    def _handle_control_call(self, goal: Goal, plan: Plan, call: ToolCall) -> str:
        try:
            args = validate_control_call(call.name, call.args)
        except ControlValidationError as exc:
            return f"Error: invalid {call.name} request: {exc}"
        try:
            if call.name == "update_task":
                return self._control_update_task(goal, plan, args)
            if call.name == "record_memory":
                return self._record_memory(goal, plan, args)
            if call.name == "inspect_task":
                return self._inspect_task(goal, plan, args)
            if call.name == "request_user":
                return self._request_user(goal, args)
            if call.name == "delegate_task":
                return json.dumps(self._delegate(goal, plan, args), ensure_ascii=False)
            if call.name == "propose_plan_change":
                new_plan = self.revise_plan(
                    reason=args["reason"],
                    add=args["tasks"],
                    proposed_by="coordinator",
                )
                return f"Plan r{new_plan.revision} proposed. Execution paused for user approval."
            if call.name == "finish_goal":
                return self._finish_goal(goal, plan, args)
            return f"Error: control tool '{call.name}' is unavailable in coordinator mode."
        except (
            DomainError,
            RuntimeErrorBase,
            StateStoreError,
            ValueError,
            KeyError,
            TypeError,
        ) as exc:
            return f"Error: {call.name} transition rejected: {redact_text(exc, 1_500)}"

    def run_slice(self, steps: int | None = None) -> SliceResult:
        """Run one bounded slice under a lease that ends at the checkpoint.

        ``continue_until_boundary`` owns a longer-lived lease and therefore
        keeps it across slices.  Direct ``/run`` and test/API callers do not;
        without this wrapper their last provider heartbeat remained active
        until TTL expiry and made an immediate restart look like a duplicate
        worker even though the slice had already returned.
        """

        goal = self.active_goal()
        claimed_here = False
        if goal is not None and goal.status is GoalStatus.RUNNING:
            session = self.store.get_workflow_session(self.session_id)
            lease = dict((session.get("state") or {}).get("execution_lease") or {})
            owned_here = (
                str(lease.get("worker_id") or "") == self._worker_id
                and str(lease.get("lease_state") or "") == "active"
            )
            if owned_here:
                # A planning/provider call may already have established this
                # runtime's lease.  A directly invoked bounded slice still
                # owns the responsibility to release it when the checkpoint
                # returns.
                claimed_here = True
            else:
                if not self._claim_execution_lease(goal):
                    return self._decorate_slice_result(
                        SliceResult(
                            "paused",
                            "Another worker owns the active execution lease; this slice was not replayed.",
                            needs_user=True,
                            phase="waiting_for_process",
                            reason="A live worker lease is still active.",
                            waiting_on="worker",
                            resume_action="Inspect",
                        )
                    )
                claimed_here = True
        try:
            return self._run_slice_impl(steps)
        finally:
            if claimed_here:
                self._release_execution_lease(
                    stage="bounded-slice-checkpoint",
                    state="boundary",
                )

    def _run_slice_impl(self, steps: int | None = None) -> SliceResult:
        with self._lock:
            goal = self.active_goal()
            if goal is None:
                raise RuntimeStateError("no active goal")
            if goal.status != GoalStatus.RUNNING:
                return self._decorate_slice_result(SliceResult(
                    goal.status.value,
                    f"Goal is {goal.status.value}; it cannot execute until the required user action occurs.",
                    completed=goal.status is GoalStatus.COMPLETED,
                    needs_user=goal.status in {GoalStatus.AWAITING_PLAN_APPROVAL, GoalStatus.PAUSED, GoalStatus.BLOCKED},
                    disposition=goal.metadata.get("completion_disposition"),
                    limitations=tuple(
                        goal.metadata.get("completion_limitations", ())
                    ),
                    phase=("awaiting_approval" if goal.status is GoalStatus.AWAITING_PLAN_APPROVAL else "paused"),
                    reason=str(goal.metadata.get("waiting_question") or goal.metadata.get("retry_reason") or ""),
                    waiting_on=str(goal.metadata.get("waiting_on") or "user"),
                    resume_action="Retry" if goal.status is GoalStatus.PAUSED else "Approve",
                ))
            plan = self.store.get_accepted_plan(goal.id)
            if plan is None or plan.revision != goal.active_plan_revision:
                raise RuntimeStateError("running goal has no matching accepted plan")
            budget = steps or self.config.work_quantum_steps
            if budget < 1:
                raise ValueError("slice steps must be positive")
            self._delegations_this_slice = 0
            no_action = 0
            progress_before = self._durable_progress_snapshot(goal.id)
            if not self._work_conversation:
                self._work_conversation = [
                    {"role": "user", "content": f"Resume durable goal {goal.id} at accepted plan r{plan.revision}."}
                ]
            schemas = [*tools.TOOL_SCHEMAS, *COORDINATOR_SCHEMAS]

            completed_steps = 0
            for step in range(1, budget + 1):
                completed_steps = step
                goal = self.store.get_goal(goal.id)
                plan = self.store.get_latest_plan(goal.id)
                if goal.status != GoalStatus.RUNNING or plan is None or plan.status != PlanStatus.ACCEPTED:
                    break
                plan, selected = self._activate_ready_task(goal, plan)
                selected_label = (
                    f"{selected.id} · {selected.title}"
                    if selected is not None
                    else "Checking completion and evidence"
                )
                self.events.publish(
                    "workflow.state",
                    f"Coordinator is deciding the next action for {selected_label}",
                    goal_id=goal.id,
                    objective=goal.objective,
                    current_task=selected_label,
                    current_task_id=(selected.id if selected is not None else ""),
                    actor="coordinator",
                    active_step=step,
                    phase="working",
                    waiting_on="model",
                    reason="Waiting for the model to choose the next evidence-producing action",
                )
                state_payload = {
                    **self._state_payload(goal, plan),
                    "harness_selected_task": None if selected is None else _task_dict(selected),
                    "selection_rule": "Work only on the harness-selected first dependency-ready task.",
                }
                local_checkpoint_message = ""
                if (
                    self.execution_class == "local"
                    and plan is not None
                    and plan.tasks
                    and all(task.status is TaskStatus.COMPLETED for task in plan.tasks)
                ):
                    local_checkpoint_message = (
                        "All accepted checklist tasks are complete and have authoritative evidence. "
                        "Do not delegate, invent a new task, or repeat verification. Call finish_goal "
                        "now with a concise summary and the concrete evidence already recorded."
                    )
                if (
                    selected is not None
                    and self.execution_class == "local"
                    and selected.status in {
                        TaskStatus.IN_PROGRESS,
                        TaskStatus.VERIFYING,
                    }
                ):
                    task_evidence = self.store.list_evidence(
                        goal.id,
                        task_id=selected.id,
                    )
                    verified_evidence = [
                        item.summary[:500]
                        for item in task_evidence
                        if item.plan_revision == plan.revision
                        and (item.verified or item.created_by == "user")
                    ]
                    if verified_evidence:
                        local_checkpoint_message = (
                            f"Task {selected.id} already has authoritative evidence from the latest "
                            "workspace action. Do not repeat the same read or command. Compare that "
                            "evidence with every acceptance criterion now; if they are satisfied, "
                            "call update_task with status=done, a factual note, and the evidence "
                            f"for {selected.id}. If a criterion is still missing, perform only the "
                            "different required check."
                        )
                        state_payload["local_checkpoint_instruction"] = local_checkpoint_message
                        state_payload["local_task_evidence"] = verified_evidence[-6:]
                if selected is not None and self.execution_class == "local":
                    # Keep a weak local coordinator from trying to preview or
                    # inspect an artifact before the selected mutation task
                    # has created it.  The accepted expected-change contract
                    # is the source of truth for this ordering guard; no
                    # content is invented here.
                    missing_paths: list[str] = []
                    for change in plan.expected_changes:
                        if not isinstance(change, Mapping):
                            continue
                        supports = {
                            str(item).strip().upper()
                            for item in (change.get("supports_tasks") or ())
                        }
                        if supports and selected.id.upper() not in supports:
                            continue
                        raw_path = str(change.get("path") or "").strip()
                        if not raw_path:
                            continue
                        try:
                            relative = normalize_contract_path(raw_path)
                            candidate_path = (self.workspace / relative).resolve(strict=False)
                            if not candidate_path.is_relative_to(self.workspace):
                                continue
                        except (DomainError, OSError, ValueError):
                            continue
                        if not candidate_path.exists():
                            missing_paths.append(relative)
                    if missing_paths:
                        local_checkpoint_message = (
                            f"Task {selected.id} has an accepted mutation contract for "
                            f"{', '.join(dict.fromkeys(missing_paths))}, but the artifact is not present yet. "
                            "Do not call preview_html or inspect_preview first. Use the mutation tool "
                            "(write_file/materialize_artifact) to create the exact artifact described by "
                            "the selected task, then run its verification."
                        )
                        state_payload["local_checkpoint_instruction"] = local_checkpoint_message
                request_conversation = [
                    *self._work_conversation,
                    {
                        "role": "user",
                        "content": state_envelope(state_payload),
                    },
                ]
                if local_checkpoint_message:
                    request_conversation.append(
                        {
                            "role": "user",
                            "content": local_checkpoint_message,
                        }
                    )
                try:
                    turn = self._call_provider(
                        request_conversation,
                        schemas,
                        COORDINATOR_SYSTEM_PROMPT,
                        actor="coordinator",
                        step=step,
                    )
                except ProviderUnavailableError as exc:
                    self.store.append_event("execution.checkpoint", goal_id=goal.id, payload={"error": redact_text(exc, 1_000)})
                    current, retrying = self._schedule_provider_retry(
                        self.store.get_goal(goal.id),
                        f"provider unavailable after bounded transport retries: {redact_text(exc, 500)}",
                    )
                    self.events.publish("error", str(exc))
                    return self._decorate_slice_result(SliceResult(
                        current.status.value,
                        (
                            f"Provider unavailable; retry {current.metadata.get('goal_attempt')} scheduled."
                            if retrying
                            else str(current.metadata.get("waiting_question") or "Provider access needs attention.")
                        ),
                        completed_steps,
                        needs_user=not retrying,
                        phase="retrying" if retrying else "paused",
                        reason=str(current.metadata.get("retry_reason") or current.metadata.get("waiting_question") or str(exc)),
                        waiting_on="provider",
                        resume_action="Retry" if not retrying else "",
                    ))
                self._work_conversation.append(turn.to_message())

                if not turn.tool_calls:
                    no_action += 1
                    self._work_conversation.append(
                        {
                            "role": "user",
                            "content": (
                                "Prose does not finish this persistent goal. Re-read the harness state, choose the next "
                                "evidence-producing action, update the checklist, or call request_user only for a true blocker."
                            ),
                        }
                    )
                    if no_action >= self.config.no_action_limit:
                        self.store.append_event("execution.no_progress", goal_id=goal.id, payload={"reason": "repeated prose-only turns"})
                        self.events.publish("warning", "Model made no structured progress; slice checkpointed without abandoning the goal.")
                        break
                    continue

                no_action = 0
                for call in turn.tool_calls:
                    self.events.publish("tool_call", call.name, args=redact_data(call.args), actor="coordinator")
                    current_goal = self.store.get_goal(goal.id)
                    current_plan = self.store.get_latest_plan(goal.id)
                    if current_goal.status != GoalStatus.RUNNING:
                        result = f"Error: goal changed to {current_goal.status.value}; no further actions run this turn."
                    elif call.name in CONTROL_NAMES:
                        result = self._handle_control_call(current_goal, current_plan, call)
                    else:
                        result = self._execute_workspace_tool(
                            current_goal,
                            call,
                            task_id=self._current_task_id(current_plan),
                            actor="coordinator",
                        )
                    self._work_conversation.append(
                        {"role": "tool", "id": call.id, "name": call.name, "content": result}
                    )
                    self.events.publish("tool_result", result, tool=call.name, actor="coordinator")
                durable_checkpoint = state_envelope(self._state_payload(
                    self.store.get_goal(goal.id),
                    self.store.get_latest_plan(goal.id),
                ))
                self._work_conversation = context.suspend_and_revive(
                    self._work_conversation,
                    durable_checkpoint,
                    getattr(self.provider, "summarize", None),
                    max_chars=self._provider_conversation_budget(
                        COORDINATOR_SYSTEM_PROMPT,
                        schemas,
                    ),
                    on_suspend=lambda count: self.events.publish(
                        "checkpoint",
                        f"Suspended {count} transient messages and revived a fresh model context from durable goal memory.",
                        continues=True,
                    ),
                )
                if self.store.get_goal(goal.id).status != GoalStatus.RUNNING:
                    break

            current = self.store.get_goal(goal.id)
            if (
                current.status is GoalStatus.REVISING
                and current.metadata.get("strategy_reinspection_required")
            ):
                self.store.update_goal_metadata(
                    current.id,
                    strategy_reinspection_required=False,
                )
                try:
                    self.generate_plan(
                        "Three materially different strategies failed to improve "
                        "the evidence. Re-inspect the repository and produce a "
                        "materially different plan.",
                        auto_approve_in_scope_repair=True,
                    )
                except ProviderUnavailableError:
                    # generate_plan persists the resumable planning pause.
                    pass
                current = self.store.get_goal(goal.id)
            if current.status == GoalStatus.RUNNING:
                made_progress = (
                    self._durable_progress_snapshot(goal.id) != progress_before
                )
                stalled = 0 if made_progress else int(current.metadata.get("no_progress_slices", 0)) + 1
                current = self.store.update_goal_metadata(
                    goal.id,
                    no_progress_slices=stalled,
                )
                if made_progress:
                    current = self.store.update_goal_metadata(
                        goal.id,
                        consecutive_retries=0,
                        retry_reason="",
                        retry_after_ms=0,
                        auto_retryable=False,
                    )
                else:
                    current = self._schedule_goal_retry(
                        current,
                        (
                            f"no durable progress in work slice ({stalled} consecutive slice(s)); "
                            "the next attempt must change hypothesis or decomposition"
                        ),
                    )
                    if stalled % self.config.stalled_slice_limit == 0:
                        self._work_conversation.append(
                            {
                                "role": "user",
                                "content": (
                                    "RETRY ESCALATION: repeated local attempts have not progressed. Stop refining the "
                                    "same approach. Reinspect the accepted evidence, split the task into a narrower "
                                    "dynamic worker assignment, or propose a materially different plan revision."
                                ),
                            }
                        )
            completed = current.status == GoalStatus.COMPLETED
            needs_user = current.status in {GoalStatus.AWAITING_PLAN_APPROVAL, GoalStatus.PAUSED, GoalStatus.BLOCKED}
            message = (
                "Goal completed."
                if completed
                else f"Work slice checkpointed at {completed_steps} step(s); durable goal status is {current.status.value}."
            )
            self.store.append_event(
                "execution.checkpoint",
                goal_id=goal.id,
                payload={"steps": completed_steps, "status": current.status.value},
            )
            self.events.publish(
                "checkpoint",
                message,
                status=current.status.value,
                paused=current.status is GoalStatus.PAUSED,
                continues=current.status is GoalStatus.RUNNING,
            )
            return self._decorate_slice_result(SliceResult(
                current.status.value,
                message,
                completed_steps,
                completed,
                needs_user,
                current.metadata.get("completion_disposition"),
                tuple(current.metadata.get("completion_limitations", ())),
                phase=("completed" if completed else "paused" if needs_user else "working"),
                reason=str(current.metadata.get("waiting_question") or current.metadata.get("retry_reason") or ""),
                waiting_on=str(current.metadata.get("waiting_on") or ("user" if needs_user else "")),
                last_tool=str(current.metadata.get("last_tool") or ""),
                workspace_mutated=bool(current.metadata.get("mutation_sequence", 0)),
                resume_action=("Retry" if needs_user else ""),
            ))

    def continue_until_boundary(
        self,
        *,
        on_checkpoint: Callable[[SliceResult], None] | None = None,
    ) -> Any:
        """Continue Normal or Ultra work until completion or a real boundary."""

        goal = self.active_goal()
        if goal is None:
            return self._decorate_slice_result(SliceResult("idle", "There is no active goal.", phase="ready"))
        if goal.status is GoalStatus.RUNNING and not self._claim_execution_lease(goal):
            result = SliceResult(
                "paused",
                "Another worker owns the active execution lease; this worker will not replay actions.",
                needs_user=True,
                phase="waiting_for_process",
                reason="A live worker lease is still active.",
                waiting_on="worker",
                resume_action="Inspect",
            )
            self.store.append_event(
                "execution.boundary",
                goal_id=goal.id,
                payload={"status": "lease_conflict", "worker_id": self._worker_id},
            )
            return self._decorate_slice_result(result)
        if goal.metadata.get("ultra_run_id"):
            session = self._ensure_ultra_session()
            if session.running:
                heartbeat_stop, heartbeat_thread = self._start_execution_heartbeat(
                    "ultra-working"
                )
                try:
                    result = self.converge_ultra()
                finally:
                    self._stop_execution_heartbeat(
                        heartbeat_stop,
                        heartbeat_thread,
                    )
                if isinstance(result, SliceResult):
                    result = self._decorate_slice_result(result)
                self._release_execution_lease(stage="ultra-boundary", state="boundary")
                return result
            current = self.active_goal() or goal
            result = SliceResult(
                current.status.value,
                f"ULTRA checkpointed at {current.status.value}.",
                completed=current.status is GoalStatus.COMPLETED,
                needs_user=current.status in {
                    GoalStatus.AWAITING_PLAN_APPROVAL,
                    GoalStatus.PAUSED,
                    GoalStatus.BLOCKED,
                },
            )
            self._release_execution_lease(stage="ultra-boundary", state="boundary")
            return self._decorate_slice_result(result)

        last = SliceResult(goal.status.value, f"Goal is {goal.status.value}.")
        while True:
            goal = self.active_goal()
            if goal is None:
                self._release_execution_lease(stage="no-goal", state="boundary")
                return last
            if goal.status is GoalStatus.PAUSED and bool(
                goal.metadata.get("auto_retryable")
            ):
                self.wait_for_scheduled_retry()
                try:
                    self.resume()
                except RuntimeErrorBase:
                    current = self.active_goal()
                    if current is None or not (
                        current.status is GoalStatus.PAUSED
                        and current.metadata.get("auto_retryable")
                    ):
                        raise
                continue
            if goal.status is not GoalStatus.RUNNING:
                self._release_execution_lease(stage=f"{goal.status.value}-boundary", state="boundary")
                return self._decorate_slice_result(last)
            self._update_execution_lease(stage="working", state="active")
            self.wait_for_scheduled_retry()
            heartbeat_stop, heartbeat_thread = self._start_execution_heartbeat(
                "normal-working"
            )
            try:
                # The continuous controller already owns the execution lease;
                # call the implementation directly so the bounded-call wrapper
                # does not release it between autonomous slices.
                last = self._decorate_slice_result(self._run_slice_impl())
            finally:
                self._stop_execution_heartbeat(
                    heartbeat_stop,
                    heartbeat_thread,
                )
            if on_checkpoint is not None:
                on_checkpoint(last)
            if (
                last.completed
                or last.needs_user
                or last.status != GoalStatus.RUNNING.value
            ):
                self.store.append_event(
                    "execution.boundary",
                    goal_id=goal.id,
                    payload={
                        "status": last.status,
                        "phase": last.phase,
                        "reason": last.reason or last.message,
                        "waiting_on": last.waiting_on,
                        "resume_action": last.resume_action,
                        "workspace_mutated": last.workspace_mutated,
                    },
                )
                self._release_execution_lease(stage=last.phase or last.status, state="boundary")
                return last

    def dashboard(self) -> DashboardView:
        goal = self.active_goal() or self.store.get_latest_goal(self.session_id)
        if goal is None:
            session = self.store.get_workflow_session(self.session_id)
            pending = session.get("state", {}).get("pending_semantic_turn")
            if isinstance(pending, Mapping):
                pending_status = str(pending.get("status") or "routing")
                stage = str(pending.get("stage") or "route").replace("_", " ")
                status = "needs_attention" if pending_status == "awaiting_provider" else "routing"
                runtime = self.workflow_runtime_snapshot()
                return DashboardView(
                    objective=str(pending.get("original_input") or "Saved request"),
                    status=status,
                    retry_reason=str(pending.get("last_error") or ""),
                    provider=self.provider_name,
                    model=self.model_name,
                    workspace=str(self.workspace),
                    activity=[f"Semantic {stage}: {pending_status}"],
                    route=runtime.route,
                    execution_strategy_name=runtime.execution_strategy,
                    runtime_phase=runtime.phase,
                    waiting_on=runtime.waiting_on,
                    last_tool=runtime.last_tool,
                    resume_action=runtime.resume_action,
                    heartbeat_at=runtime.heartbeat_at,
                    workspace_mutated=runtime.workspace_mutated,
                )
            runtime = self.workflow_runtime_snapshot()
            return DashboardView(
                provider=self.provider_name,
                model=self.model_name,
                workspace=str(self.workspace),
                route=runtime.route,
                execution_strategy_name=runtime.execution_strategy,
                runtime_phase=runtime.phase,
                waiting_on=runtime.waiting_on,
                last_tool=runtime.last_tool,
                resume_action=runtime.resume_action,
                heartbeat_at=runtime.heartbeat_at,
                workspace_mutated=runtime.workspace_mutated,
            )
        plan = self.store.get_latest_plan(goal.id)
        tasks = [] if plan is None else [
            TaskView(
                task.id,
                task.title,
                _display_task_status(task.status),
                task.role.name,
                list(task.acceptance_criteria),
                list(task.verification),
                list(task.depends_on),
                task.risk,
            )
            for task in plan.tasks
        ]
        delegations = self.store.list_delegations(goal.id)
        workers = [
            WorkerView(item.id[-10:], item.task_id, item.role.name, item.status.value)
            for item in delegations
            if item.status in {DelegationStatus.PENDING, DelegationStatus.IN_PROGRESS}
        ]
        events = self.store.list_recent_events(goal.id, limit=100)
        activity = [
            f"{event.event_type}: {str(event.payload.get('reason') or event.payload.get('summary') or event.entity_id or '')[:120]}"
            for event in events[-4:]
        ]
        runtime = self.workflow_runtime_snapshot()
        visible_question = (
            str(goal.metadata.get("waiting_question", ""))
            if runtime.phase in {
                "paused",
                "retrying",
                "waiting_for_approval",
            }
            else ""
        )
        return DashboardView(
            objective=goal.objective,
            status=goal.status.value,
            plan_revision=plan.revision if plan else 0,
            approved_revision=goal.active_plan_revision,
            plan_summary=plan.summary if plan else "",
            plan_fingerprint=plan.fingerprint if plan else "",
            plan_applicability=[] if plan is None else [dict(item) for item in plan.applicability_evidence],
            execution_strategy="" if plan is None else plan.execution_strategy,
            expected_changes=[] if plan is None else [dict(item) for item in plan.expected_changes],
            goal_attempt=int(goal.metadata.get("goal_attempt", 0)),
            retry_reason=str(goal.metadata.get("retry_reason", "")),
            tasks=tasks,
            workers=workers,
            provider=self.provider_name,
            model=self.model_name,
            workspace=str(self.workspace),
            waiting_question=visible_question,
            activity=activity,
            route=runtime.route,
            execution_strategy_name=runtime.execution_strategy,
            runtime_phase=runtime.phase,
            waiting_on=runtime.waiting_on,
            last_tool=runtime.last_tool,
            resume_action=runtime.resume_action,
            heartbeat_at=runtime.heartbeat_at,
            workspace_mutated=runtime.workspace_mutated,
        )

    def apply_command(self, command: UserCommand) -> Any:
        kind, args = command.kind, command.args
        if kind == CommandKind.ANSWER:
            question_id = str(args.get("question_id") or "").strip()
            if self.store.get_pending_intake(self.session_id) is not None:
                pending = self.intake_questions()
                if not pending:
                    raise RuntimeStateError("there is no active intake question")
                return self.answer_intake_question(
                    question_id or str(pending[0].get("id", "")),
                    args["value"],
                )
            goal = self.active_goal()
            if goal and goal.metadata.get("ultra_run_id"):
                questions = self.ultra_questions()
                answers = dict(goal.metadata.get("plan_answers", {}))
                pending = [
                    item for item in questions
                    if not str(answers.get(str(item.get("id")), "")).strip()
                ]
                if not question_id and not pending:
                    raise RuntimeStateError("there is no active ULTRA question")
                return self.answer_ultra_question(
                    question_id or str(pending[0].get("id", "")),
                    args["value"],
                )
            pending = self.plan_questions()
            if not question_id:
                pending = [item for item in pending if not str(item.get("answer") or "").strip()]
                if not pending:
                    raise RuntimeStateError("there is no active planning question")
                question_id = str(pending[0].get("id", ""))
            return self.answer_plan_question(question_id, args["value"])
        if kind == CommandKind.GOAL:
            mode = self.store.get_workflow_session(self.session_id)["session_mode"]
            return self.submit_intent(args["objective"], requested_mode=mode)
        if kind == CommandKind.APPROVE:
            return self.approve_plan(args["revision"])
        if kind in {CommandKind.REJECT, CommandKind.REPLAN}:
            goal = self.active_goal()
            if goal and goal.metadata.get("ultra_run_id"):
                return self.replan_ultra(args["feedback"])
            return self.reject_plan(args["feedback"])
        if kind == CommandKind.ADD:
            return self.add_user_task(args["text"], args["acceptance_criteria"])
        if kind == CommandKind.EDIT:
            return self.revise_plan(
                reason=f"user edited checklist field {args['field']}",
                edit=(args["task_id"], args["field"], args["value"]),
            )
        if kind == CommandKind.REMOVE:
            return self.revise_plan(reason="user removed a checklist item", remove=args["task_id"])
        if kind == CommandKind.TASK_STATUS:
            return self.update_task_from_user(args["task_id"], args["status"], args["note"])
        if kind == CommandKind.RUN:
            return self.run_slice(args["steps"])
        if kind == CommandKind.PAUSE:
            return self.pause()
        if kind == CommandKind.RESUME:
            return self.resume()
        if kind == CommandKind.CANCEL:
            return self.cancel(args["confirmation"])
        if kind == CommandKind.RESOLVE:
            return self.resolve_action(args["action_id"], args["resolution"], args["note"])
        if kind == CommandKind.TEXT:
            text = args["text"]
            if not text:
                return None
            if self.store.get_pending_intake(self.session_id) is not None:
                pending = self.intake_questions()
                if not pending:
                    raise RuntimeStateError("intake is ready but has not been routed")
                return self.answer_intake_question(str(pending[0]["id"]), text)
            goal, plan = self.active_goal(), self.latest_plan()
            if goal is not None:
                answers = dict(goal.metadata.get("plan_answers", {}))
                questions = (
                    self.ultra_questions()
                    if goal.metadata.get("ultra_run_id")
                    else self.plan_questions()
                )
                pending_questions = [
                    item
                    for item in questions
                    if not str(item.get("answer") or "").strip()
                    and not str(answers.get(str(item.get("id")), "")).strip()
                ]
                if pending_questions:
                    question_id = str(pending_questions[0].get("id", ""))
                    return (
                        self.answer_ultra_question(question_id, text)
                        if goal.metadata.get("ultra_run_id")
                        else self.answer_plan_question(question_id, text)
                    )
            if (
                goal is not None
                and plan is not None
                and goal.status is GoalStatus.AWAITING_PLAN_APPROVAL
                and plan.status is PlanStatus.PENDING_APPROVAL
                and is_unambiguous_plan_approval(text)
            ):
                self.store.append_event(
                    "plan.natural_language_approval",
                    goal_id=goal.id,
                    entity_type="plan",
                    entity_id=plan.id,
                    payload={"utterance": redact_text(text, 200)},
                )
                return self.approve_plan(plan.revision)
            if self.active_goal() is None:
                _decision, result = self.route_input(text)
                return result
            return self.add_guidance(text)
        return None

    @staticmethod
    def _chat_path_hash(workspace: Path, relative: str) -> str | None:
        if not relative or relative == ".":
            return None
        try:
            candidate = (workspace / relative).resolve(strict=True)
            candidate.relative_to(workspace)
            if candidate.is_file():
                return hashlib.sha256(candidate.read_bytes()).hexdigest()
        except (OSError, RuntimeError, ValueError):
            return None
        return None

    @staticmethod
    def _chat_workspace_hashes(workspace: Path) -> dict[str, str]:
        ignored = {".coding-agent", ".git", ".venv", "node_modules", "__pycache__", ".pytest_cache"}
        result: dict[str, str] = {}
        for candidate in workspace.rglob("*"):
            try:
                relative = candidate.relative_to(workspace)
                if any(part in ignored for part in relative.parts) or not candidate.is_file():
                    continue
                if candidate.stat().st_size > 10_000_000:
                    continue
                result[relative.as_posix()] = hashlib.sha256(candidate.read_bytes()).hexdigest()
                if len(result) >= 5_000:
                    break
            except OSError:
                continue
        return result

    def _artifactize_chat_text(self, text: str) -> tuple[str, tuple[dict[str, Any], ...]]:
        """Persist large generated code and replace provider history with stable handles."""

        original = str(text or "")
        artifacts: list[dict[str, Any]] = []
        pattern = re.compile(r"```([\w.+-]*)\s*\n([\s\S]*?)```", re.MULTILINE)

        def replace_block(match: re.Match[str]) -> str:
            language = (match.group(1) or "text").casefold()
            content = match.group(2)
            if len(content) < 2_048 and "<!doctype html" not in content.casefold():
                return match.group(0)
            names = {
                "html": "index.html",
                "javascript": "generated.js",
                "js": "generated.js",
                "python": "generated.py",
                "py": "generated.py",
                "css": "styles.css",
            }
            artifact = self.store.add_chat_artifact(
                self.session_id,
                content,
                language=language,
                suggested_name=names.get(language, "generated.txt"),
            )
            artifacts.append(artifact)
            return (
                f'<CHAT_ARTIFACT id="{artifact["id"]}" language="{language}" '
                f'suggested_name="{artifact["suggested_name"]}" '
                f'sha256="{artifact["content_hash"]}" bytes="{artifact["byte_size"]}" />'
            )

        compact = pattern.sub(replace_block, original)
        if (
            not artifacts
            and len(original) >= 2_048
            and "<!doctype html" in original.casefold()
        ):
            start = original.casefold().find("<!doctype html")
            content = original[start:]
            artifact = self.store.add_chat_artifact(
                self.session_id,
                content,
                language="html",
                suggested_name="index.html",
            )
            artifacts.append(artifact)
            compact = (
                f'<CHAT_ARTIFACT id="{artifact["id"]}" language="html" '
                f'suggested_name="index.html" sha256="{artifact["content_hash"]}" '
                f'bytes="{artifact["byte_size"]}" />'
            )
        return compact, tuple(artifacts)

    def _execute_chat_tool(
        self,
        call: ToolCall,
        decision: SemanticTurnDecisionV2,
        semantic_turn_id: str = "",
    ) -> tuple[tools.ToolExecutionResult, tuple[str, ...]]:
        spec = tools.get_spec(call.name)
        args = call.args if isinstance(call.args, dict) else {}
        if spec is None:
            return tools.ToolExecutionResult(False, f"Error: unknown chat tool {call.name!r}"), ()
        if not decision.permits_tool_category(spec.category):
            return tools.ToolExecutionResult(
                False,
                f"Error: tool category {spec.category!r} is outside the accepted semantic effect contract",
                error_code="effect_contract",
            ), ()
        risk = spec.risk
        normal_requirement = tools.requires_approval(call.name, args)
        needs_approval = (
            self.permission_adapter.requires_approval(normal_requirement)
            if self.permission_adapter is not None else normal_requirement
        )
        if call.name == "open_path" or (call.name == "preview_html" and bool(args.get("open_browser", True))):
            needs_approval = True
        action_id = self.store.begin_session_action(
            self.session_id, call.name, redact_data(args), risk=risk,
            mutating=spec.mutates_workspace,
        )
        self._record_semantic_action(
            semantic_turn_id,
            action_id,
            category=spec.category,
            mutating=spec.mutates_workspace,
            status="running",
        )
        self.events.publish("tool_call", call.name, args=redact_data(args), actor="chat", id=call.id)
        if needs_approval and not self._approval_allowed(call.name, dict(args), risk):
            result = tools.ToolExecutionResult(False, "Permission denied by the user.", error_code="permission")
            self.store.complete_session_action(action_id, result.output, status="denied")
            self._record_semantic_action(
                semantic_turn_id,
                action_id,
                category=spec.category,
                mutating=spec.mutates_workspace,
                status="denied",
                output=result.output,
            )
            return result, ()

        candidate_paths = [
            str(args.get(field, "")).strip()
            for field in spec.path_fields
            if str(args.get(field, "")).strip() not in {"", "."}
        ]
        if call.name == "apply_patch":
            candidate_paths.extend(
                tools.apply_patch.patch_paths(
                    str(args.get("patch", "")),
                    str(args.get("base_path") or ".").strip(),
                )
            )
        before = {path: self._chat_path_hash(self.workspace, path) for path in candidate_paths}
        workspace_before = (
            self._chat_workspace_hashes(self.workspace)
            if call.name in {"run_bash", "run_command", "install_dependencies"}
            else {}
        )
        try:
            with tools.workspace_context(self.workspace):
                if call.name in {"run_bash", "run_command"} and self.permission_adapter is not None:
                    command = str(args.get("command", ""))
                    if (
                        call.name == "run_command"
                        and str(args.get("cwd", ".")).strip() not in {"", "."}
                        and self.permission_adapter.access_level.value == "full"
                    ):
                        command = f"cd -- {shlex.quote(str(args['cwd']))} && {command}"
                    detailed = self.permission_adapter.run_shell(
                        command,
                        self.workspace,
                        normal_runner=lambda value: tools.run_tool(
                            call.name,
                            {**args, "command": value},
                        ),
                    )
                    result = tools.ToolExecutionResult.from_output(detailed)
                elif (
                    call.name in {"start_process", "install_dependencies"}
                    and self.permission_adapter is not None
                    and self.permission_adapter.access_level.value == "full"
                ):
                    result = tools.ToolExecutionResult(
                        False,
                        f"Error: {call.name} cannot run as a persistent host action in Full Docker mode; use run_command inside Docker or switch to Normal.",
                        error_code="sandbox",
                    )
                else:
                    result = tools.run_tool_detailed(call.name, args)
            if call.name in {"run_bash", "run_command"} and result.ok:
                shell_exit = re.search(r"(?im)^exit code:\s*(-?\d+)", result.output)
                if shell_exit and int(shell_exit.group(1)) != 0:
                    result = tools.ToolExecutionResult(
                        False,
                        "Error: shell command failed; " + result.output,
                        error_code="nonzero_exit",
                    )
            if call.name == "preview_html" and result.ok:
                try:
                    preview_result = json.loads(result.output)
                    incomplete = (
                        (bool(args.get("open_browser", True)) and not preview_result.get("browser_opened"))
                        or (
                            bool(args.get("verify", True))
                            and preview_result.get("verification") in {None, "unavailable", "not_requested"}
                        )
                    )
                    if incomplete:
                        result = tools.ToolExecutionResult(
                            False,
                            "Error: HTML preview started but the requested browser open/verification capability was unavailable: "
                            + result.output,
                            error_code="browser_unavailable",
                        )
                except (TypeError, json.JSONDecodeError):
                    result = tools.ToolExecutionResult(False, "Error: preview_html returned malformed evidence", error_code="invalid_result")
            if call.name == "start_process" and result.ok:
                try:
                    process_result = json.loads(result.output)
                    if not process_result.get("ready") or process_result.get("status") != "running":
                        result = tools.ToolExecutionResult(
                            False,
                            "Error: managed process did not reach its requested ready state: " + result.output,
                            error_code="process_not_ready",
                        )
                except (TypeError, json.JSONDecodeError):
                    result = tools.ToolExecutionResult(False, "Error: start_process returned malformed evidence", error_code="invalid_result")
            result = tools.ToolExecutionResult(
                result.ok,
                redact_text(result.output, 50_000),
                data=result.data,
                changed_paths=result.changed_paths,
                error_code=result.error_code,
            )
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            result = tools.ToolExecutionResult(
                False,
                f"Error: Chat tool harness failure: {type(exc).__name__}: {redact_text(exc, 500)}",
                error_code="harness",
            )
        after = {path: self._chat_path_hash(self.workspace, path) for path in candidate_paths}
        changed_list = [path for path in candidate_paths if result.ok and before.get(path) != after.get(path)]
        if workspace_before and result.ok:
            workspace_after = self._chat_workspace_hashes(self.workspace)
            changed_list.extend(
                path for path in sorted(set(workspace_before) | set(workspace_after))
                if workspace_before.get(path) != workspace_after.get(path)
            )
        changed = tuple(dict.fromkeys(changed_list))
        self.store.complete_session_action(
            action_id,
            redact_text(result.output, 2_000),
            status="completed" if result.ok else "failed",
            changed_paths=changed,
        )
        self._record_semantic_action(
            semantic_turn_id,
            action_id,
            category=spec.category,
            mutating=spec.mutates_workspace,
            status="completed" if result.ok else "failed",
            output=result.output,
            changed_paths=changed,
        )
        if result.ok and call.name in {
            "start_process", "poll_process", "stop_process",
            "preview_html", "inspect_preview", "stop_preview",
        }:
            try:
                resource = json.loads(result.output)
                resource_id = str(resource.get("process_id") or resource.get("preview_id") or "")
                if resource_id:
                    self.store.save_managed_resource(
                        resource_id,
                        self.session_id,
                        kind="preview" if resource_id.startswith("preview-") else "process",
                        status=str(resource.get("status") or ("stopped" if resource.get("stopped") else "running")),
                        metadata=resource,
                    )
            except (TypeError, json.JSONDecodeError):
                pass
        return result, changed

    @staticmethod
    def _chat_evidence(outputs: list[tuple[str, str]]) -> str:
        evidence: list[str] = []
        for name, output in outputs[-5:]:
            if name == "preview_html":
                try:
                    payload = json.loads(output)
                    evidence.append(
                        f"preview {payload.get('url')} · HTTP {payload.get('http_status')} · "
                        f"verification {payload.get('verification')} · browser_opened={payload.get('browser_opened')}"
                    )
                    continue
                except (TypeError, json.JSONDecodeError):
                    pass
            evidence.append(f"{name}: {' '.join(output.split())[:240]}")
        return "\n".join(f"- {item}" for item in evidence)

    def chat(
        self,
        text: str,
        *,
        steps: int = 12,
        _route_checked: bool = False,
        semantic_decision: SemanticTurnDecisionV2 | None = None,
        semantic_turn_id: str = "",
    ) -> Any:
        """Run bounded Chat with durable artifacts and action postcondition gates."""

        prompt = str(text)
        if not prompt.strip():
            return SliceResult("idle", "", 0)
        if self.active_goal() is None:
            self._stop_event.clear()
        if not _route_checked:
            _route, result = self.route_input(prompt)
            return result
        if semantic_decision is None:
            raise RuntimeStateError("bounded Chat/Action requires an accepted semantic effect contract")
        if semantic_decision.route not in {RouteKind.CHAT, RouteKind.ACTION}:
            raise RuntimeStateError("Goal decisions cannot execute in the bounded Chat loop")
        if not semantic_turn_id:
            user_message = {"role": "user", "content": prompt}
            self._chat_conversation.append(user_message)
            self.store.append_chat_message(self.session_id, user_message)
        session = self.store.get_workflow_session(self.session_id)
        session_state = dict(session.get("state", {}))
        session_state.setdefault(
            "run_id",
            f"run-{hashlib.sha256((prompt + str(time.time_ns())).encode()).hexdigest()[:20]}",
        )
        session_state.setdefault("original_objective", prompt)
        session_state.setdefault("user_messages", [])
        session_state["user_messages"] = [
            *session_state["user_messages"],
            prompt,
        ][-50:]

        known_artifacts = self.store.list_chat_artifacts(self.session_id)

        capability_rows = tools.capability_report()
        capabilities = json.dumps(capability_rows, ensure_ascii=False)
        artifact_rows = [
            {
                key: item.get(key)
                for key in (
                    "id",
                    "language",
                    "suggested_name",
                    "content_hash",
                    "byte_size",
                )
            }
            for item in known_artifacts[-10:]
        ]
        system = CHAT_SYSTEM_PROMPT + "\n\nRUNTIME CAPABILITIES:\n" + capabilities
        system += (
            "\n\nACCEPTED SEMANTIC EFFECT CONTRACT:\n"
            + json.dumps(semantic_decision.to_dict(), ensure_ascii=False, sort_keys=True)
            + "\nUse no tool category outside this contract. The final response must be model-authored and evidence-based."
        )
        if artifact_rows:
            system += (
                "\n\nDURABLE CHAT ARTIFACTS:\n"
                + json.dumps(artifact_rows, ensure_ascii=False)
            )

        executed = 0
        changed_paths: list[str] = []
        successful_categories: list[str] = []
        successful_outputs: list[tuple[str, str]] = []
        if semantic_turn_id:
            pending_state = dict(
                self.store.get_workflow_session(self.session_id).get("state", {}).get(
                    "pending_semantic_turn", {}
                )
            )
            for record in pending_state.get("action_records", ()):
                if isinstance(record, Mapping) and record.get("status") == "completed":
                    successful_categories.append(str(record.get("category") or ""))
                    successful_outputs.append((
                        str(record.get("category") or "action"),
                        str(record.get("output") or "completed"),
                    ))
        failure_outputs: list[str] = []
        no_action_attempts = 0
        final_text = ""

        for step in range(1, max(1, steps) + 1):
            allowed_names = tools.names(categories=semantic_decision.allowed_categories)
            allowed_schemas = [schema for schema in tools.TOOL_SCHEMAS if _tool_name(schema) in allowed_names]
            try:
                turn = self._call_provider(
                    self._chat_conversation,
                    allowed_schemas,
                    system,
                    actor="chat",
                    step=step,
                    stream_text=False,
                )
            except ProviderUnavailableError:
                if not successful_outputs:
                    raise
                evidence = self._chat_evidence(successful_outputs)
                final_text = (
                    "The requested tool actions completed, but the provider could not compose the final response.\n\n"
                    "Action receipt (tool evidence only):\n" + evidence
                )
                break
            message = turn.to_message()
            display_text = turn.text or ""
            if turn.text:
                compact_text, created = self._artifactize_chat_text(turn.text)
                message["content"] = compact_text
                if created:
                    session_state["chat_artifact_ids"] = list(
                        dict.fromkeys(
                            [
                                *session_state.get("chat_artifact_ids", []),
                                *(item["id"] for item in created),
                            ]
                        )
                    )
            self._chat_conversation.append(message)
            self.store.append_chat_message(self.session_id, message)

            if turn.tool_calls:
                for call in turn.tool_calls:
                    result, changed = self._execute_chat_tool(
                        call, semantic_decision, semantic_turn_id
                    )
                    executed += 1
                    tool_message = {
                        "role": "tool",
                        "id": call.id,
                        "name": call.name,
                        "content": result.output,
                    }
                    self._chat_conversation.append(tool_message)
                    self.store.append_chat_message(self.session_id, tool_message)
                    self.events.publish(
                        "tool_result",
                        result.output,
                        tool=call.name,
                        actor="chat",
                        id=call.id,
                    )
                    if result.ok:
                        spec = tools.get_spec(call.name)
                        if spec is not None:
                            successful_categories.append(spec.category)
                        successful_outputs.append((call.name, result.output))
                        changed_paths.extend(changed)
                    else:
                        failure_outputs.append(result.output)
                continue

            missing = semantic_decision.missing_effects(successful_categories)
            if missing:
                no_action_attempts += 1
                if no_action_attempts >= self.config.no_action_limit:
                    detail = (
                        failure_outputs[-1]
                        if failure_outputs
                        else "the model repeatedly returned prose without using the available tools"
                    )
                    final_text = f"Action could not be completed: {detail}"
                    break
                correction_message = {
                    "role": "user",
                    "content": corrective_prompt(missing, capabilities),
                }
                self._chat_conversation.append(correction_message)
                self.store.append_chat_message(self.session_id, correction_message)
                continue

            if display_text:
                final_text = display_text
                break
            no_action_attempts += 1
            if no_action_attempts >= self.config.no_action_limit:
                if successful_outputs:
                    final_text = (
                        "The requested tool actions completed, but the provider did not compose a final response.\n\n"
                        "Action receipt (tool evidence only):\n" + self._chat_evidence(successful_outputs)
                    )
                    break
                raise ProviderUnavailableError("provider returned no natural Chat response")
            self._chat_conversation.append({
                "role": "user",
                "content": "FINAL HANDOFF REQUIRED: Write a concise natural response grounded only in the tool evidence. Do not call another tool unless a requested effect is still unsatisfied.",
            })

        if not final_text:
            raise ProviderUnavailableError("bounded turn ended without a model-authored response or action receipt")
        if successful_outputs and semantic_decision.route is RouteKind.ACTION and "Action receipt" not in final_text:
            final_text = (
                final_text.rstrip()
                + "\n\nEvidence:\n"
                + self._chat_evidence(successful_outputs)
            )

        artifacts = list(dict.fromkeys(changed_paths))
        # Tool execution and semantic action receipts can advance the session
        # revision while this bounded turn is running.  Re-read the durable
        # envelope before clearing the transient chat state; using the initial
        # revision here made a successful small-file action fail at handoff
        # with a false WorkflowSessionConflictError.
        for _attempt in range(2):
            latest = self.store.get_workflow_session(self.session_id)
            latest_state = dict(latest.get("state") or {})
            # Only carry the chat-owned fields forward.  Preserve newer
            # semantic/action records written by the tool boundary.
            for key in ("run_id", "original_objective", "user_messages", "chat_artifact_ids"):
                if key in session_state:
                    latest_state[key] = session_state[key]
            try:
                self.store.mutate_workflow_session(
                    self.session_id,
                    lambda current_state, state=latest_state: {
                        "state": state,
                        "goal_id": None,
                        "plan_state": PlanState.NONE.value,
                        "run_state": RunState.IDLE.value,
                    },
                    expected_revision=int(latest.get("revision") or 0),
                )
                break
            except WorkflowSessionConflictError:
                if _attempt == 1:
                    raise
        return SliceResult("chat", final_text, executed)

    def sleep_profile(self, action: str, mode: Any) -> Mapping[str, Any]:
        """Control session-scoped Sleep without weakening Ultra/Docker gates."""

        from .config import InteractionMode
        from .sandbox import AccessLevel

        normalized = str(action).strip().lower()
        if normalized == "status":
            return self.sleep_controller.status()
        if normalized == "off":
            self.sleep_controller.disable()
            return self.sleep_controller.status()
        selected = InteractionMode.parse(mode)
        access = self.permission_adapter.access_level if self.permission_adapter else AccessLevel.NORMAL
        goal = self.active_goal() or self.store.get_latest_goal(self.session_id)
        run_id = str(goal.metadata.get("ultra_run_id", "")) if goal else ""
        uncertain = bool(
            run_id
            and any(
                item.status is ChangeSetStatus.UNCERTAIN
                for item in self.store.list_change_sets(run_id)
            )
        )
        self.sleep_controller.enable(
            mode=selected,
            access_level=access,
            docker_ready=access is AccessLevel.FULL,
            safe_checkpoint=bool(self.ultra_session and self.ultra_session.safe_for_reconfiguration),
            active_uncertain_mutation=uncertain,
        )
        return self.sleep_controller.status()
