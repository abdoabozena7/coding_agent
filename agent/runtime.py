"""Deterministic orchestration above the probabilistic model/tool loop.

The model proposes plans, actions, dynamic roles, and completion.  This runtime
owns plan approval, state transitions, evidence, retries, recovery, delegation
limits, and the final completion gate.
"""

from __future__ import annotations

import copy
import difflib
import hashlib
import json
import importlib.util
import os
import platform
import re
import time
import shutil
import shlex
from dataclasses import dataclass, replace
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Iterable, Mapping, Sequence

from . import context, tools
from .commands import CommandKind, UserCommand
from .chat_runtime import ChatIntentV1, corrective_prompt
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
    state_envelope,
    subagent_system_prompt,
)
from .providers.base import AssistantTurn, ToolCall
from .safety import ProgressWatchdog, redact_data, redact_text
from .store import NotFoundError, StateStore, StateStoreError
from .ui import DashboardView, TaskView, WorkerView
from .workflow import (
    PlanDraftError,
    RetryKind,
    RetryLedger,
    first_ready_task,
    is_unambiguous_plan_approval,
    normalize_plan_draft,
    validate_normalized_plan,
    SessionMode,
    PlanState,
    RunState,
)
from .sleep_profile import SleepController
from .quality import ChangeSetStatus
from .run_context import GoalContractV1, is_goal_escalation_approval
from .weak_model import WeakModelPolicy
from .repository_index import OllamaEmbeddingProvider, RepositoryIndex
from .diagnostics import ErrorSignature, FailureDomain, normalize_error_message
from .local_provider import (
    extract_first_json_object,
    normalize_action_proposal,
    normalize_generated_tool_args,
)

try:
    from .model_catalog import ExecutionClass, ModelDescriptor
    from .sandbox import AccessLevel, PermissionAdapter
except ImportError:  # pragma: no cover - direct-script compatibility
    ExecutionClass = ModelDescriptor = AccessLevel = PermissionAdapter = Any  # type: ignore


ApprovalCallback = Callable[[str, dict[str, Any], str], bool]

READ_ONLY_TOOLS = tools.names(categories={"read"})
MUTATING_TOOLS = tools.names(mutating=True)
TOOL_RISK = tools.risk_map()


class RuntimeErrorBase(RuntimeError):
    pass


class RuntimeStateError(RuntimeErrorBase):
    pass


class ProviderUnavailableError(RuntimeErrorBase):
    pass


@dataclass(frozen=True)
class SliceResult:
    status: str
    message: str
    steps: int = 0
    completed: bool = False
    needs_user: bool = False


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
    ) -> None:
        self.provider = provider
        self.store = store
        self.workspace = Path(workspace).resolve(strict=True)
        self.events = events or EventBus()
        self.approval = approval or (lambda _name, _args, _risk: False)
        self.config = config or RuntimeConfig.from_env()
        self.sleeper = sleeper
        self.model_descriptor = model_descriptor
        self.permission_adapter = permission_adapter
        self.ultra_session: Any | None = None
        self._closed = False
        self._lock = RLock()
        self._work_conversation: list[dict[str, Any]] = []
        self._watchdog = ProgressWatchdog(self.config.repeated_action_limit)
        self._delegations_this_slice = 0
        self._provider_input_tokens = 0
        self._provider_output_tokens = 0
        self.retry_ledger = RetryLedger()
        self._chat_conversation: list[dict[str, Any]] = []
        self.session_id = "workspace-session"
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
            )
        self._chat_conversation = [dict(item) for item in self.store.list_chat_messages(self.session_id)]
        tools.register_artifact_provider(self.workspace, self.store.get_chat_artifact)
        tools.configure_workspace(self.workspace)

        active_policy_goal = self.store.load_active_goal()
        if active_policy_goal is not None:
            persisted_policy = active_policy_goal.metadata.get("weak_model_policy")
            if isinstance(persisted_policy, Mapping):
                self.weak_model_policy = WeakModelPolicy.from_dict(persisted_policy)

        recovery = self.store.recover_inflight()
        if recovery.changed:
            self.events.publish(
                "recovery",
                "Interrupted actions were marked uncertain and were not replayed.",
                tasks=list(recovery.task_ids),
                delegations=list(recovery.delegation_ids),
                actions=list(recovery.action_ids),
            )
            goal = self.store.load_active_goal()
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

        ultra_recovery = self.store.recover_ultra_inflight()
        if ultra_recovery.changed:
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

        auto_reconciled = self._auto_reconcile_read_only_ultra_uncertainty()
        if auto_reconciled:
            recovered_goal = self.store.load_active_goal()
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
        goal = self.store.load_active_goal()
        if goal and goal.status in {
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
        goal = self.store.load_active_goal()
        if goal and goal.status == GoalStatus.RUNNING:
            accepted = self.store.get_accepted_plan(goal.id)
            if accepted and (
                not accepted.applicability_evidence
                or not accepted.execution_strategy.strip()
                or not accepted.expected_changes
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

    @property
    def provider_name(self) -> str:
        return self.provider.__class__.__name__.removesuffix("Provider").lower() or "provider"

    @property
    def model_name(self) -> str:
        return str(getattr(self.provider, "model", "unknown"))

    @property
    def reasoning_effort(self) -> str:
        return str(getattr(self.provider, "reasoning_effort", "medium"))

    def set_reasoning_effort(self, effort: str) -> str:
        from .config import ReasoningEffort

        selected = ReasoningEffort.parse(effort).value
        goal = self.active_goal()
        if goal is not None and goal.status in {
            GoalStatus.RUNNING, GoalStatus.VERIFYING, GoalStatus.REVIEWING, GoalStatus.RECOVERING,
        }:
            raise RuntimeStateError("reasoning effort can change only at a safe checkpoint")
        setattr(self.provider, "reasoning_effort", selected)
        return selected

    @property
    def execution_class(self) -> str:
        if self.model_descriptor is not None:
            return self.model_descriptor.execution_class.value
        return "cloud" if self.provider_name in {"openai", "gemini"} else "local"

    @property
    def access_level(self) -> str:
        if self.permission_adapter is not None:
            return self.permission_adapter.access_level.value
        return "normal"

    def replace_provider(
        self,
        provider: Any,
        descriptor: ModelDescriptor,
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
        self.provider = provider
        self.model_descriptor = descriptor
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

    def replace_config(self, config: RuntimeConfig) -> None:
        """Apply validated slice limits at an interactive command checkpoint."""

        if not isinstance(config, RuntimeConfig):
            raise TypeError("config must be a RuntimeConfig")
        with self._lock:
            self.config = config
            # Preserve the in-memory action history so changing an unrelated
            # display/runtime setting cannot clear the no-progress guardrail.
            self._watchdog.repeat_limit = max(1, self.config.repeated_action_limit)

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
        """Return the exact reason Ultra cannot start, for disabled UI choices."""

        try:
            self._require_ultra_setup()
        except RuntimeStateError as exc:
            return str(exc)
        return None

    def _make_ultra_session(self) -> Any:
        descriptor, permission_adapter = self._require_ultra_setup()
        from .ultra import UltraConfig
        from .ultra_session import UltraSession

        return UltraSession(
            store=self.store,
            workspace=self.workspace,
            descriptor=descriptor,
            permission_adapter=permission_adapter,
            approval=self.approval,
            events=self.events,
            config=UltraConfig(
                min_top_modules=self.config.ultra_top_modules_min,
                max_top_modules=self.config.ultra_top_modules_max,
                max_depth=self.config.ultra_max_depth,
                max_nodes=self.config.ultra_max_nodes,
                max_fix_attempts=self.config.ultra_fix_attempts,
                cloud_concurrency=self.config.ultra_cloud_concurrency,
                max_concurrency=8,
                provider_retries=self.config.max_provider_retries,
                role_memory_ttl_hours=self.config.role_memory_ttl_hours,
                context_chars=min(
                    self.config.conversation_chars,
                    120_000 if descriptor.execution_class is ExecutionClass.CLOUD else self.weak_model_policy.max_context_characters,
                ),
                prompt_trace_chars=self.config.prompt_trace_chars,
            ),
            agent_steps=self.config.subagent_steps,
            reasoning_effort=self.reasoning_effort,
        )

    def start_ultra(self, objective: str) -> Any:
        """Start the sequential foundation and checkpoint at questions/approval."""

        if self.active_goal() is not None:
            raise RuntimeStateError("finish or cancel the active goal before starting ULTRA")
        self.ultra_session = self._make_ultra_session()
        return self.ultra_session.start(redact_text(objective, 20_000))

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
        visual = any(
            str(path).casefold().endswith((".html", ".htm"))
            for path in dict(goal.metadata.get("quality_target", {})).get("artifact_ids", ())
        )
        tags = ("normal", "visual", "browser") if visual else ("normal", "implementation", "verification")
        content = (
            "Require deterministic evidence and independent review before completion; "
            + (
                "interactive HTML also requires browser/runtime and visual evidence."
                if visual
                else "preserve the durable goal, plan, and checklist across retries."
            )
        )
        if blocker:
            content += f" Last blocker pattern: {redact_text(blocker, 500)}"
        self.global_lessons.put(
            LearnedLessonV1(
                title="Normal evidence-gated execution",
                content=content,
                applicability_tags=tags,
                evidence_refs=(evidence_ref,),
                successes=1 if succeeded else 0,
                failures=0 if succeeded else 1,
            )
        )

    def _route_intake(self, intake: Mapping[str, Any], brief: Any) -> Any:
        routed = RunMode.parse(brief.routed_mode)
        self.store.complete_intake_session(
            str(intake["id"]),
            brief=brief.to_dict(),
            routed_mode=routed.value,
            route_reason=brief.route_reason,
        )
        self.transition_mode(routed.value)
        self.events.publish(
            "intake.routed",
            f"Intent Architect routed the task to {routed.value.upper()}: {brief.route_reason}",
            intake_id=intake["id"],
            mode=routed.value,
            complexity=dict(intake.get("complexity", {})),
            execution_brief=brief.to_dict(),
        )
        canonical = brief.canonical_prompt()
        return self.start_ultra(canonical) if routed is RunMode.ULTRA else self.start_goal(canonical)

    def submit_intent(
        self,
        text: str,
        *,
        requested_mode: str | RunMode = RunMode.NORMAL,
    ) -> Any:
        """Run every new objective through the shared, durable intake gate."""

        value = redact_text(text, 20_000).strip()
        if not value:
            return None
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
        decision = self.intent_architect.analyze(
            value,
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
                else f"Intent Architect prepared a {decision.brief.routed_mode.value.upper()} execution brief."
            ),
            intake_id=intake["id"],
            mode=decision.brief.routed_mode.value,
            complexity=decision.complexity.to_dict(),
            questions=[item.to_dict() for item in decision.questions],
        )
        if decision.questions:
            self.store.save_workflow_session(
                self.session_id,
                goal_id=None,
                session_mode=decision.brief.routed_mode.value,
                plan_state=PlanState.INSPECTING.value,
                run_state=RunState.PLANNING.value,
                state={
                    "intake_id": intake["id"],
                    "intake_status": IntakeStatus.AWAITING_ANSWERS.value,
                },
            )
            return SliceResult(
                "awaiting_answers",
                f"Intent Architect needs {len(decision.questions)} decision(s) before planning.",
                needs_user=True,
            )
        return self._route_intake(intake, decision.brief)

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
        decision = self.intent_architect.analyze(
            str(updated["original_input"]),
            requested_mode=str(updated["requested_mode"]),
            answers=answers,
            repository_facts=self._intake_repository_facts(str(updated["original_input"])),
        )
        self.store.save_prompt_completeness(
            str(updated["id"]),
            decision.completeness.to_dict(),
        )
        return self._route_intake(updated, decision.brief)

    def active_ultra_run(self) -> Any | None:
        goal = self.active_goal() or self.store.get_latest_goal()
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

    def answer_ultra_question(self, question_id: str, value: str) -> Any:
        if self.ultra_session is None:
            raise RuntimeStateError(
                "this ULTRA question round belongs to a previous process; use /replan to rebuild the foundation"
            )
        return self.ultra_session.answer(question_id, value)

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
        if self.ultra_session is not None:
            self.ultra_session.add_guidance(safe)
        return item

    def approve_ultra(self, revision: int | None = None) -> Plan:
        if self.ultra_session is None:
            raise RuntimeStateError(
                "the ULTRA engine is not live in this process; use /replan to restore from durable evidence"
            )
        return self.ultra_session.approve(revision)

    def wait_for_ultra(self) -> Any:
        if self.ultra_session is None:
            raise RuntimeStateError("the ULTRA engine is not live in this process")
        return self.ultra_session.wait()

    @staticmethod
    def _plan_change_paths(plan: Plan | None) -> set[str]:
        if plan is None:
            return set()
        return {
            str(item.get("path") or "").strip().replace("\\", "/")
            for item in plan.expected_changes
            if str(item.get("path") or "").strip()
        }

    def _ultra_quality_feedback(self, result: Any) -> str:
        findings: list[str] = []
        for package in (
            *tuple(getattr(result, "results", ()) or ()),
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

        if self.ultra_session is None:
            raise RuntimeStateError("the ULTRA engine is not live in this process")
        approved_scope = self._plan_change_paths(
            self.store.get_accepted_plan(self.active_goal().id)
            if self.active_goal()
            else None
        )
        while True:
            result = self.wait_for_ultra()
            goal = self.active_goal() or self.store.get_latest_goal()
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
            proposed = self.replan_ultra(feedback)
            while proposed is None:
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
                proposed = self.answer_ultra_question(str(question.get("id")), answer)
            proposed_scope = self._plan_change_paths(proposed)
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
            self.approve_ultra(proposed.revision)

    def restore_ultra(self, run_id: str) -> Any:
        self.ultra_session = self._make_ultra_session()
        return self.ultra_session.restore(run_id)

    def replan_ultra(self, feedback: str) -> Any:
        from .ultra_models import UltraRunStatus

        goal = self.active_goal()
        run = self.active_ultra_run()
        if goal is None or run is None or not goal.metadata.get("ultra_run_id"):
            raise RuntimeStateError("there is no active ULTRA master plan to revise")
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
        objective = (
            f"{goal.objective}\n\nMASTER PLAN REVISION REQUEST:\n{safe_feedback}\n"
            "Preserve verified evidence and explicitly identify any changed scope, interface, or dependency."
        )
        self.ultra_session = self._make_ultra_session()
        return self.ultra_session.restart_foundation(goal.id, objective)

    def close(self) -> None:
        """Checkpoint background ULTRA work before the SQLite connection closes."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        if self.ultra_session is not None:
            self.ultra_session.close()
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
        return self.store.load_active_goal()

    def latest_plan(self) -> Plan | None:
        goal = self.active_goal()
        return self.store.get_latest_plan(goal.id) if goal else None

    def transition_mode(self, mode: str) -> str:
        """Persist a policy change without replacing the active run or its memory."""
        target = SessionMode.parse(mode)
        session = self.store.get_workflow_session(self.session_id)
        previous = SessionMode.parse(str(session["session_mode"]))
        state = dict(session.get("state", {}))
        goal = self.active_goal()
        if goal is not None:
            state.update({
                "run_id": goal.metadata.get("run_id", goal.id),
                "goal_contract_fingerprint": goal.metadata.get("goal_contract_fingerprint"),
                "mutation_sequence": goal.metadata.get("mutation_sequence", 0),
                "convergence_state": goal.metadata.get("convergence_state", "not_evaluated"),
            })
        self.store.save_workflow_session(
            self.session_id,
            goal_id=goal.id if goal else session.get("goal_id"),
            session_mode=target.value,
            plan_state=str(session["plan_state"]),
            run_state=str(session["run_state"]),
            ultra_profile=str(session.get("ultra_profile", "standard")),
            sleep_state=str(session.get("sleep_state", "off")),
            state=state,
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

    def _call_provider(
        self,
        conversation: list[dict[str, Any]],
        schemas: Sequence[dict[str, Any]],
        system: str,
        *,
        actor: str,
        step: int,
        stream_text: bool = True,
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
            system = (
                system
                + "\n\nNATIVE TOOLS ARE UNAVAILABLE. Make exactly one bounded action proposal as "
                + '{"name":"AVAILABLE_NAME","args":{...}} with no lifecycle IDs. '
                + f"Available names: {', '.join(names)}. The harness validates and executes it."
            )
            if current_goal is not None:
                self.store.append_event(
                    "provider.request_adapter_selected", goal_id=current_goal.id,
                    payload={"actor": actor, "adapter": "constrained_json_action", "native_tools": False},
                )
        last_error: Exception | None = None
        for attempt in range(self.config.max_provider_retries + 1):
            try:
                turn = self.provider.call(
                    conversation,
                    list(schemas),
                    system,
                    on_text=(
                        (lambda fragment: self.events.publish("model_text", str(fragment), actor=actor))
                        if stream_text else None
                    ),
                    on_thought=lambda fragment: self.events.publish("model_thought", str(fragment), actor=actor),
                )
                if not isinstance(turn, AssistantTurn):
                    raise TypeError(f"provider returned {type(turn).__name__}, expected AssistantTurn")
                for call in turn.tool_calls:
                    if not isinstance(call.args, dict):
                        call.args = {}
                    else:
                        call.args = normalize_generated_tool_args(call.name, call.args)
                if not turn.tool_calls and schemas and turn.text:
                    # Some weak models advertise native tool calling but emit
                    # the requested call as a JSON object in assistant text.
                    # Treat this as a recoverable transport-shape mismatch and
                    # normalize exactly one allow-listed proposal.  The normal
                    # schema, permission, and action journal gates still own
                    # execution.
                    candidate = extract_first_json_object(turn.text)
                    proposal = normalize_action_proposal(candidate) if candidate is not None else None
                    if proposal is not None:
                        name, args = proposal
                        args = normalize_generated_tool_args(name, args)
                        allowed = {_tool_name(schema) for schema in schemas}
                        if name in allowed:
                            generated_id = f"harness-{actor.replace(':', '-')}-{step}-{attempt}"
                            turn.tool_calls.append(ToolCall(id=generated_id, name=name, args=args))
                            if current_goal is not None:
                                self.store.append_event(
                                    "tool_action.proposal_normalized", goal_id=current_goal.id,
                                    payload={
                                        "actor": actor,
                                        "tool": name,
                                        "generated_id": generated_id,
                                        "advertised_native_tools": native_tools,
                                    },
                                )
                self._emit_usage(turn)
                return turn
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as exc:
                last_error = exc
                message = redact_text(exc, 500)
                retry_record = self.retry_ledger.record(
                    RetryKind.PROVIDER_TRANSPORT,
                    stage=actor,
                    reason=message,
                    input_value={"step": step, "conversation_messages": len(conversation)},
                    output_value={"error_type": type(exc).__name__},
                    next_action=(
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
                if isinstance(exc, (AssertionError, TypeError, ValueError)) or attempt >= self.config.max_provider_retries:
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
        raise ProviderUnavailableError(
            f"provider unavailable after retries: {type(last_error).__name__}: {redact_text(last_error, 500)}"
        ) from last_error

    def start_goal(self, objective: str) -> Plan | None:
        with self._lock:
            safe_objective = redact_text(objective, 20_000)
            prior_session = self.store.get_workflow_session(self.session_id)
            prior_state = dict(prior_session.get("state", {}))
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
            goal = self.store.create_goal(
                safe_objective,
                metadata={
                    "run_id": run_id,
                    "weak_model_policy": self.weak_model_policy.to_dict(),
                    "goal_contract": contract.to_dict(),
                    "goal_contract_fingerprint": contract.fingerprint,
                    "convergence_state": "not_evaluated",
                    "mutation_sequence": 0,
                    "continued_from_chat": continuing_chat_candidate,
                    "chat_candidate": dict(continuation) if continuing_chat_candidate else {},
                },
            )
            self.store.append_event(
                "goal_contract.created", goal_id=goal.id,
                payload={"run_id": run_id, "fingerprint": contract.fingerprint, "policy_version": self.weak_model_policy.version},
            )
            self.store.save_workflow_session(
                self.session_id,
                goal_id=goal.id,
                session_mode=SessionMode.PLAN.value,
                plan_state=PlanState.INSPECTING.value,
                run_state=RunState.PLANNING.value,
            )
            self.store.transition_goal(goal.id, GoalStatus.DISCOVERING, reason="planning started")
            self._work_conversation.clear()
            self.events.publish("phase", "Discovering the workspace and drafting a plan.", goal_id=goal.id)
            try:
                return self.generate_plan()
            except ProviderUnavailableError as exc:
                self.store.append_event("planning.checkpoint", goal_id=goal.id, payload={"error": redact_text(exc, 500)})
                self.events.publish("error", str(exc))
                return None

    def _planner_tools(self) -> list[dict[str, Any]]:
        return [*_schemas(READ_ONLY_TOOLS), *PLANNER_SCHEMAS]

    @staticmethod
    def _plan_requires_critic(candidate: Mapping[str, Any]) -> bool:
        tasks = tuple(candidate.get("tasks", ()))
        return len(tasks) > 3 or any(
            str(item.get("risk", "medium")).lower() in {"high", "critical"}
            for item in tasks
            if isinstance(item, Mapping)
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
            "verdict": "revise",
            "summary": "The plan critic did not produce a valid structured verdict.",
            "issues": ["Re-evaluate coverage and submit a valid plan; keep every criterion observable."],
        }

    @staticmethod
    def _validate_plan_applicability(
        proposed: Mapping[str, Any],
        tasks: Iterable[Task],
        *,
        successful_inspection_ids: frozenset[str],
    ) -> None:
        if not successful_inspection_ids:
            raise ValueError(
                "the planner must successfully inspect the workspace with a read-only tool before proposing"
            )
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
            path = str(item["path"]).strip().casefold()
            if any(marker in path for marker in ("<", ">", "tbd", "unknown", "determine later")):
                raise ValueError("expected workspace changes must name real paths, not placeholders")
            supports = {str(task_id).upper() for task_id in item["supports_tasks"]}
            unknown = supports - task_ids
            if unknown:
                raise ValueError(
                    f"expected changes reference unknown tasks: {', '.join(sorted(unknown))}"
                )
            change_coverage.update(supports)
        if not change_coverage:
            raise ValueError("the plan must identify a concrete workspace change")

    @staticmethod
    def _bind_plan_inspection_sources(
        proposed: dict[str, Any],
        inspection_records: Mapping[str, Mapping[str, Any]],
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
        placeholder = re.compile(
            r"^(?:tool|inspection):(?:call(?:_id|_\d+)?|\d+)$",
            re.IGNORECASE,
        )
        bound = dict(proposed)
        evidence = [dict(item) for item in proposed.get("applicability_evidence", ())]
        for item in evidence:
            source = str(item.get("source") or "").strip()
            canonical = aliases.get(source.casefold())
            if canonical is None and only_reference is not None:
                record = records[only_reference]
                tool_alias = f"tool:{record.get('tool', '')}".casefold()
                if not source or placeholder.fullmatch(source) or source.casefold() == tool_alias:
                    canonical = f"inspection:{only_reference}"
            if canonical is not None:
                item["source"] = canonical
        bound["applicability_evidence"] = evidence
        return bound

    def _harness_fallback_plan(
        self,
        goal: Goal,
        inspection_records: Mapping[str, Mapping[str, Any]],
    ) -> Plan | None:
        """Create a narrow approval-bound plan when a weak planner stalls.

        This fallback is deliberately limited to short objectives naming one
        concrete artifact. It never executes or approves the plan.
        """

        objective = str(goal.objective).strip()
        if not inspection_records or len(objective) > 1_200:
            return None
        paths = re.findall(
            r"(?<![\w./-])([A-Za-z0-9_.-]+\.(?:html?|py|js|ts|tsx|jsx|css|json|md|txt|ya?ml|toml))\b",
            objective,
            flags=re.IGNORECASE,
        )
        unique_paths = tuple(dict.fromkeys(path.replace("\\", "/") for path in paths))
        if len(unique_paths) != 1:
            return None
        target = unique_paths[0]
        is_html = target.casefold().endswith((".html", ".htm"))
        reference, record = next(iter(inspection_records.items()))
        raw = {
            "summary": f"Implement and verify {target} from the user's explicit objective.",
            "applicability_evidence": [
                {
                    "fact": (
                        "The workspace was inspected and the named artifact can be implemented in place. "
                        + str(record.get("result") or "")[:400]
                    ),
                    "source": f"inspection:{reference}",
                    "supports_tasks": ["T001"],
                }
            ],
            "execution_strategy": (
                f"Create or update {target}, read it back, then "
                + ("run browser preview verification and inspect runtime errors." if is_html else "run the narrowest applicable verification.")
            ),
            "expected_changes": [
                {"path": target, "intent": objective, "supports_tasks": ["T001"]}
            ],
            "tasks": [
                {
                    "title": f"Implement and verify {target}",
                    "description": objective,
                    "acceptance_criteria": [
                        f"{target} exists and implements every behavior explicitly requested in the objective.",
                        "The saved artifact is re-read and contains no placeholder-only implementation.",
                    ],
                    "verification": [
                        f"Read {target} after writing and compare it with the objective.",
                        (
                            f"Open {target} with the browser preview and confirm HTTP success with no console, page, or network errors."
                            if is_html
                            else "Run the narrowest executable or static check applicable to the artifact."
                        ),
                    ],
                    "depends_on": [],
                    "risk": "low",
                }
            ],
        }
        proposed, actions = normalize_plan_draft(raw)
        validate_normalized_plan(proposed)
        for task in proposed["tasks"]:
            task.pop("_unresolved_dependencies", None)
        plan = self.store.create_plan(
            goal.id,
            proposed["summary"],
            proposed["tasks"],
            applicability_evidence=proposed["applicability_evidence"],
            execution_strategy=proposed["execution_strategy"],
            expected_changes=proposed["expected_changes"],
            proposed_by="harness-weak-model-fallback",
            submit=True,
        )
        current_goal = self.store.get_goal(goal.id)
        if current_goal.status != GoalStatus.AWAITING_PLAN_APPROVAL:
            self.store.transition_goal(
                goal.id,
                GoalStatus.AWAITING_PLAN_APPROVAL,
                reason="narrow harness fallback plan awaits user approval",
            )
        self.store.append_event(
            "planning.harness_fallback",
            goal_id=goal.id,
            entity_type="plan",
            entity_id=plan.id,
            payload={"target": target, "normalization_actions": list(actions)},
        )
        self.events.publish(
            "plan",
            f"Plan r{plan.revision} was recovered from the explicit single-artifact objective. Review it, then use /approve {plan.revision}.",
        )
        self.store.update_goal_metadata(
            goal.id,
            consecutive_retries=0,
            retry_reason="",
            retry_after_ms=0,
            auto_retryable=False,
            plan_questions=[],
            waiting_question="",
        )
        self.store.save_workflow_session(
            self.session_id,
            goal_id=goal.id,
            session_mode=SessionMode.PLAN.value,
            plan_state=PlanState.AWAITING_APPROVAL.value,
            run_state=RunState.PLANNING.value,
            state={"plan_revision": plan.revision, "plan_fingerprint": plan.fingerprint, "fallback": True},
        )
        return plan

    def _pause_planning(self, goal: Goal, question: str, reason: str) -> None:
        """Checkpoint a bounded/failed planning pass as an explicit user-visible pause."""
        current = self.store.get_goal(goal.id)
        if current.status not in {GoalStatus.DISCOVERING, GoalStatus.REVISING}:
            return
        attempt = int(current.metadata.get("goal_attempt", 0)) + 1
        consecutive = int(current.metadata.get("consecutive_retries", 0)) + 1
        retry_ms = self._goal_retry_delay_ms(consecutive)
        self.store.update_goal_metadata(
            goal.id,
            waiting_question=question,
            resume_status=current.status.value,
            goal_attempt=attempt,
            consecutive_retries=consecutive,
            retry_reason=reason,
            retry_after_ms=retry_ms,
            auto_retryable=True,
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
        labels = {
            str(option.get("label", "")).strip()
            for option in item.get("options", ())
            if isinstance(option, Mapping)
        }
        if labels and answer not in labels and not bool(item.get("allow_freeform", True)):
            raise ValueError(
                f"answer must be one of: {', '.join(sorted(labels))}"
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

    def wait_for_scheduled_retry(self) -> int:
        """Apply one bounded backoff delay; retry count itself remains unbounded."""
        goal = self.active_goal()
        if goal is None:
            return 0
        delay_ms = max(0, int(goal.metadata.get("retry_after_ms", 0)))
        if delay_ms:
            self.events.publish(
                "checkpoint",
                f"Goal retry {goal.metadata.get('goal_attempt', 0)} waits {delay_ms / 1000:.1f}s; Ctrl-C checkpoints safely.",
            )
            self.sleeper(delay_ms / 1_000)
            self.store.update_goal_metadata(goal.id, retry_after_ms=0)
        return delay_ms

    def generate_plan(self, feedback: str = "") -> Plan | None:
        try:
            return self._generate_plan(feedback)
        except ProviderUnavailableError as exc:
            goal = self.active_goal()
            if goal is not None:
                self._pause_planning(
                    goal,
                    "Planning provider retries were exhausted. Fix connectivity/rate limits, add guidance if useful, then use /resume.",
                    "planning provider unavailable after bounded retries",
                )
            raise

    def _generate_plan(self, feedback: str = "") -> Plan | None:
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
        planning_questions = tuple(goal.metadata.get("plan_questions", ()))
        planning_answers = dict(goal.metadata.get("plan_answers", {}))
        conversation: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": state_envelope(
                    {
                        "objective": goal.objective,
                        "workspace": str(self.workspace),
                        "user_feedback": feedback,
                        "planning_questions": list(planning_questions),
                        "planning_answers": planning_answers,
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
        unproductive_turns_after_inspection = 0
        last_plan_format_error = ""
        plan_format_exhausted = False
        successful_inspection_ids: set[str] = set()
        inspection_records: dict[str, dict[str, Any]] = {}
        inspection_cache: dict[str, str] = {}
        for step in range(1, self.config.planning_steps + 1):
            inspections_before_turn = frozenset(successful_inspection_ids)
            turn = self._call_provider(
                conversation,
                self._planner_tools(),
                PLANNER_SYSTEM_PROMPT,
                actor="planner",
                step=step,
            )
            conversation.append(turn.to_message())
            proposed: dict[str, Any] | None = None
            requested_questions: list[dict[str, Any]] | None = None
            for call in turn.tool_calls:
                self.events.publish("tool_call", call.name, args=redact_data(call.args), actor="planner")
                conversation_result: str | None = None
                if call.name == "propose_plan":
                    try:
                        proposed, normalization_actions = normalize_plan_draft(call.args)
                        validate_normalized_plan(proposed)
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
                        invalid_plan_calls += 1
                        last_plan_format_error = redact_text(exc, 1_000)
                        gross_format_failure = isinstance(exc, PlanDraftError) and all(
                            any(token in issue.path for token in ("/title", "/description", "/acceptance_criteria", "/verification"))
                            for issue in exc.issues
                        )
                        retry_record = self.retry_ledger.record(
                            RetryKind.PLAN_SEMANTIC_REPAIR if isinstance(exc, PlanDraftError) else RetryKind.PLAN_FORMAT_REPAIR,
                            stage=getattr(exc, "stage", "plan_normalization"),
                            reason=last_plan_format_error,
                            input_value=call.args,
                            next_action="targeted_repair" if invalid_plan_calls == 1 else "stop",
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
                        if invalid_plan_calls >= (4 if gross_format_failure else 2):
                            plan_format_exhausted = True
                elif call.name == "request_plan_input":
                    try:
                        # Weak/local models often omit the third choice or mark
                        # several recommendations. Canonicalize first, then
                        # validate the strict persisted/UI contract.
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
                    except ControlValidationError as exc:
                        result = f"Error: invalid plan question request: {exc}"
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
                self.events.publish("tool_result", result, tool=call.name, actor="planner")
                if plan_format_exhausted:
                    # A single model turn may contain several proposals.  The
                    # retry budget is global to this planning run, so stop at
                    # the fourth rejected proposal instead of processing an
                    # arbitrary remainder from the same response.
                    break

            if plan_format_exhausted:
                break

            if requested_questions is not None and proposed is None:
                self._pause_for_plan_questions(goal, requested_questions)
                return None

            if proposed is None and requested_questions is None and inspection_records:
                unproductive_turns_after_inspection += 1
                if unproductive_turns_after_inspection >= 2:
                    fallback_plan = self._harness_fallback_plan(goal, inspection_records)
                    if fallback_plan is not None:
                        return fallback_plan

            if proposed is not None:
                # Inspection provenance and persistence-only cross references
                # are harness data, not fields the model must manufacture.
                task_ids = [str(item["id"]) for item in proposed["tasks"]]
                if not proposed.get("applicability_evidence"):
                    proposed["applicability_evidence"] = [
                        {
                            "fact": (
                                "Repository exists; no project files were found; the project will be created from scratch."
                                if not str(record.get("result", "")).strip()
                                or "No files" in str(record.get("result", ""))
                                else f"Workspace inspection completed with {record.get('tool', 'read-only tool')}."
                            ),
                            "source": str(record["reference"]),
                            "supports_tasks": task_ids,
                        }
                        for record in inspection_records.values()
                    ]
                if not proposed.get("expected_changes"):
                    proposed["expected_changes"] = [
                        {
                            "path": (
                                re.search(r"[A-Za-z0-9_./-]+\.[A-Za-z0-9_-]+", str(item["description"])).group(0)
                                if re.search(r"[A-Za-z0-9_./-]+\.[A-Za-z0-9_-]+", str(item["description"]))
                                else "."
                            ),
                            "intent": str(item["description"]),
                            "supports_tasks": [str(item["id"])],
                        }
                        for item in proposed["tasks"]
                    ]
                for change in proposed.get("expected_changes", ()):
                    if str(change.get("path", "")).startswith("<"):
                        match = re.search(
                            r"[A-Za-z0-9_./-]+\.[A-Za-z0-9_-]+",
                            str(change.get("intent", "")),
                        )
                        change["path"] = match.group(0) if match else "."
                if not str(proposed.get("execution_strategy", "")).strip():
                    proposed["execution_strategy"] = (
                        "Execute ready tasks in dependency order and run each task's required verification."
                    )
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
                proposed = self._bind_plan_inspection_sources(
                    proposed,
                    inspection_records,
                )
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
                    self._validate_plan_applicability(
                        proposed,
                        preview,
                        successful_inspection_ids=inspections_before_turn,
                    )
                except (ValueError, DomainError) as exc:
                    invalid_plan_calls += 1
                    last_plan_format_error = redact_text(exc, 1_000)
                    conversation.append(
                        {
                            "role": "user",
                            "content": (
                                "Harness plan validation rejected the proposal: "
                                f"{last_plan_format_error}. Repair every listed ID, dependency, "
                                "evidence citation, and criterion in one complete proposal."
                            ),
                        }
                    )
                    proposed = None
                    if invalid_plan_calls >= 4:
                        plan_format_exhausted = True
                        break
                    continue
                critique = (
                    self._review_plan_candidate(goal, proposed, inspection_records)
                    if self._plan_requires_critic(proposed)
                    else {
                        "verdict": "pass",
                        "summary": "Deterministic validation passed; independent criticism is not required for this low-complexity plan.",
                        "issues": [],
                    }
                )
                if critique["verdict"] == "pass" and not critique["issues"]:
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
                        f"Plan r{plan.revision} is ready. Review it, edit if needed, then use /approve {plan.revision}.",
                    )
                    self.store.update_goal_metadata(
                        goal.id,
                        consecutive_retries=0,
                        retry_reason="",
                        retry_after_ms=0,
                        auto_retryable=False,
                        plan_questions=[],
                        waiting_question="",
                    )
                    self.store.save_workflow_session(
                        self.session_id,
                        goal_id=goal.id,
                        session_mode=SessionMode.PLAN.value,
                        plan_state=PlanState.AWAITING_APPROVAL.value,
                        run_state=RunState.PLANNING.value,
                        state={"plan_revision": plan.revision, "plan_fingerprint": plan.fingerprint},
                    )
                    return plan
                revisions += 1
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
                if revisions >= 1:
                    break
            elif not turn.tool_calls:
                conversation.append(
                    {
                        "role": "user",
                        "content": "Planning is not complete in prose. Inspect if needed, then call propose_plan with a validated plan.",
                    }
                )

        fallback_plan = self._harness_fallback_plan(goal, inspection_records)
        if fallback_plan is not None:
            return fallback_plan

        checkpoint_reason = (
            (
                "planner repeatedly returned an invalid structured plan; exact field validation is recorded"
                if invalid_plan_calls >= 4
                else "plan validation and its single targeted repair failed"
            )
            if plan_format_exhausted
            else "planner did not produce a critic-approved structured plan"
        )
        self.store.append_event(
            "planning.checkpoint",
            goal_id=goal.id,
            payload={
                "reason": checkpoint_reason,
                "format_attempts": invalid_plan_calls,
                "technical_detail": last_plan_format_error,
            },
        )
        if plan_format_exhausted:
            self.events.publish(
                "error",
                (
                    "Plan could not be prepared after one targeted repair. "
                    f"Stage/field detail: {last_plan_format_error}. Edit the request or use /replan."
                ),
                technical_detail=last_plan_format_error,
                attempts=invalid_plan_calls,
                planning_terminal=True,
            )
        else:
            self.events.publish(
                "warning",
                "Planning stopped before a valid plan was produced. Inspect the planning checkpoint for the exact stage and use /replan with guidance.",
                planning_terminal=True,
            )
        self._pause_planning(
            goal,
            (
                f"Plan validation failed after one targeted repair: {last_plan_format_error}. Add guidance, then use /replan."
                if plan_format_exhausted
                else "The planner did not produce a critic-approved structured plan in its bounded pass. Add guidance, then use /resume or /replan."
            ),
            checkpoint_reason,
        )
        return None

    def approve_plan(self, revision: int | None = None, *, approved_by: str = "user") -> Plan:
        goal = self.active_goal()
        plan = self.latest_plan()
        if goal is None or plan is None:
            raise RuntimeStateError("there is no plan to approve")
        if goal.metadata.get("ultra_run_id"):
            return self.approve_ultra(revision)
        requested = plan.revision if revision is None else revision
        accepted, _approval = self.store.approve_plan(
            goal.id,
            requested,
            approved_by=approved_by,
            expected_fingerprint=plan.fingerprint if requested == plan.revision else None,
        )
        current = self.store.get_goal(goal.id)
        contract_data = current.metadata.get("goal_contract")
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
            standard_dimensions = [
                ("requirement-completeness", "Every explicit requested outcome is implemented", True, 1.0),
                ("functional-correctness", "Required behavior works under authoritative execution", True, 1.0),
                ("runtime-stability", "No relevant runtime, provider, browser, or console error remains", True, 1.0),
                ("integration-correctness", "The change is integrated without breaking existing contracts", True, 1.0),
                ("regression-safety", "Impacted focused checks and appropriate broader regression checks pass", True, 1.0),
                ("maintainability", "The implementation is coherent, bounded, and avoids unnecessary complexity", False, 0.85),
            ]
            objective_lower = contract.interpreted_objective.casefold()
            castle_dimensions = []
            artifact_dimensions = []
            if any(path.casefold().endswith((".html", ".htm")) for path in artifact_paths):
                artifact_dimensions = [
                    ("visual-quality", "Visual composition, hierarchy, detail, and polish meet the requested quality", False, 0.85),
                    ("interaction-quality", "Interactive and animated behavior is understandable, stable, and appropriately varied", False, 0.85),
                ]
            if "castle" in objective_lower and "siege" in objective_lower:
                castle_dimensions = [
                    ("castle-recognizable", "Castle is visually recognizable and more detailed than placeholder rectangles", False, 0.85),
                    ("main-gate-visible", "Main gate is visible and not hidden by overlap", True, 1.0),
                    ("ram-soldiers", "Soldiers visibly attempt to breach the gate with a battering ram", False, 0.85),
                    ("ram-motion", "Battering ram has repeated understandable motion", False, 0.85),
                    ("siege-tower", "Siege tower is clearly represented and participates in the scene", False, 0.85),
                    ("moving-arrows", "Archers visibly release moving arrows", False, 0.85),
                    ("catapult-projectiles", "Catapults visibly launch moving projectiles", False, 0.85),
                    ("projectile-distinction", "Projectiles are distinguishable from static decoration", False, 0.85),
                    ("animation-variety", "Actors do not all use identical synchronized animation", False, 0.85),
                    ("scene-depth", "Scene has meaningful layering, depth, and a non-empty composition", False, 0.85),
                    ("self-contained", "The requested single HTML artifact is self-contained with no unexpected network requests", True, 1.0),
                    ("extended-stability", "Animation remains stable during an extended run without JavaScript errors", True, 1.0),
                    ("responsive-usability", "Wide and narrow viewports remain usable without harmful overflow", True, 1.0),
                ]
            generated_dimensions = [
                {
                    "id": dimension_id,
                    "description": description,
                    "hard_gate": hard_gate,
                    "minimum_score": minimum,
                    "required_evidence": list(verification),
                    "evaluation_method": (
                        "vision_and_runtime" if dimension_id in {item[0] for item in (*artifact_dimensions, *castle_dimensions)} and not hard_gate
                        else "deterministic_then_independent_review"
                    ),
                    "confidence": "medium",
                    "latest_artifact_hash": None,
                    "latest_mutation_sequence": None,
                }
                for dimension_id, description, hard_gate, minimum in (*standard_dimensions, *artifact_dimensions, *castle_dimensions)
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
        self.events.publish("phase", f"Plan r{accepted.revision} approved by {approved_by}; execution is active.")
        self.store.save_workflow_session(
            self.session_id,
            goal_id=goal.id,
            session_mode=SessionMode.GOAL.value,
            plan_state=PlanState.APPROVED.value,
            run_state=RunState.EXECUTING.value,
            state={"plan_revision": accepted.revision, "plan_fingerprint": accepted.fingerprint},
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
        dimensions = []
        for dimension, words in {
            "visual_quality": ("graphic", "visual", "detail", "empty", "color", "castle", "catapult"),
            "interaction_quality": ("animation", "motion", "basic", "slow", "fast"),
            "functional_correctness": ("error", "broken", "still happening", "doesn't", "does not"),
        }.items():
            if any(word in lowered for word in words):
                dimensions.append(dimension)
        if not dimensions:
            dimensions.append("requirement_completeness")

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
            if "visual_quality" in dimensions:
                retrieval_query += " style css color lighting scene castle component"
            if "interaction_quality" in dimensions:
                retrieval_query += " keyframe animation event movement timing"
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

    def _auto_reconcile_read_only_ultra_uncertainty(self) -> tuple[str, ...]:
        """Reset provably side-effect-free crash windows without user ceremony."""

        from .ultra_models import AgentRunStatus, WorkNodeStatus

        goal = self.store.load_active_goal()
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
        mutating_action_nodes: set[str] = set()
        for action in self.store.list_actions(goal.id):
            if not bool(action.get("mutating")):
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
                mutating_action_nodes.add(node_id)
        unsafe_nodes = {
            str(item.work_node_id)
            for item in uncertain_agents
            if item.side_effects
            and item.work_node_id
            and str(item.work_node_id) in mutating_action_nodes
        }
        for agent in uncertain_agents:
            if agent.side_effects and str(agent.work_node_id or "") in mutating_action_nodes:
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
            component_only = bool(node.contract.metadata.get("component_package_only"))
            if not component_only or node.contract.write_paths or node.id in unsafe_nodes:
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
        self.store.update_goal_metadata(goal.id, resume_status=goal.status.value)
        result = self.store.transition_goal(goal.id, GoalStatus.PAUSED, reason=reason)
        self.events.publish("phase", "Goal paused safely; state is durable.")
        return result

    def resume(self) -> Goal:
        goal = self.active_goal()
        ultra_run = self.active_ultra_run() if goal is not None else None
        resumable_failed_ultra = bool(
            goal is not None
            and goal.status == GoalStatus.BLOCKED
            and goal.metadata.get("ultra_run_id")
            and ultra_run is not None
            and ultra_run.status.value in {"running", "recovering"}
        )
        if goal is None or (
            goal.status != GoalStatus.PAUSED and not resumable_failed_ultra
        ):
            raise RuntimeStateError("goal is not paused")
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
        self.store.update_goal_metadata(goal.id, retry_after_ms=0, auto_retryable=False)
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
            self.generate_plan("Resume the interrupted planning pass from durable goal state.")
            return self.active_goal() or self.store.get_goal(goal.id)
        return result

    def cancel(self, confirmation: str) -> Goal:
        if confirmation.strip().upper() != "CANCEL":
            raise RuntimeStateError("cancelling an unfinished goal requires ':cancel CANCEL'")
        goal = self.active_goal()
        if goal is None:
            raise RuntimeStateError("no active goal")
        if self.ultra_session is not None:
            self.ultra_session.cancel()
        result = self.store.transition_goal(goal.id, GoalStatus.CANCELLED, reason="explicitly cancelled by user")
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
            uncertain_tasks=list(recovery.task_ids),
            uncertain_actions=list(recovery.action_ids),
        )
        return goal

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
        return {
            "goal": {
                "id": goal.id,
                "objective": goal.objective,
                "status": goal.status.value,
                "success_criteria": list(goal.success_criteria),
                "constraints": list(goal.constraints),
                "active_plan_revision": goal.active_plan_revision,
            },
            "runtime_environment": {
                "platform": platform.system(),
                "os_name": os.name,
                "shell": os.environ.get("COMSPEC") if os.name == "nt" else "/bin/sh",
                "workspace": str(self.workspace),
                "note": "run_bash is a legacy name; it invokes the platform shell shown here.",
            },
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

    def _current_task_id(self, plan: Plan) -> str | None:
        for task in plan.tasks:
            if task.status in {TaskStatus.IN_PROGRESS, TaskStatus.VERIFYING}:
                return task.id
        return None

    def _activate_ready_task(self, goal: Goal, plan: Plan) -> tuple[Plan, Task | None]:
        """Bind the slice to one dependency-ready task without model cooperation.

        Weak models frequently start using workspace tools before emitting the
        bookkeeping-only ``update_task(in_progress)`` call.  That used to leave
        otherwise authoritative tool evidence unscoped (``task_id=None``), so a
        later completion attempt could never satisfy the evidence gate.  Task
        selection is a deterministic scheduler decision and belongs here.
        """

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
        args = call.args if isinstance(call.args, dict) else {}
        if call.name == "list_files" and not str(args.get("path") or "").strip():
            # Omitted, empty, and explicit-root paths mean the same operation.
            # Canonicalizing here lets retry detection see them as identical.
            args = {**args, "path": "."}
        scoped_name = f"{actor}:{call.name}"
        journal_args = {
            "_harness_actor": actor,
            "_harness_plan_revision": goal.active_plan_revision,
            "arguments": redact_data(args),
        }
        approach_fingerprint = hashlib.sha256(
            json.dumps({"tool": call.name, "args": redact_data(args)}, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        persisted_failures = tuple(goal.metadata.get("failed_attempts", ()))
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
        if needs_approval and not self.approval(call.name, copy.deepcopy(args), risk):
            action_id = self.store.begin_action(
                goal.id,
                call.name,
                journal_args,
                task_id=task_id,
                risk=risk,
                mutating=call.name in MUTATING_TOOLS,
            )
            result = "Permission denied by the user. Do not repeat the same action."
            self.store.complete_action(action_id, result, status="denied")
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
        pre_path = str(args.get("path", "")).strip() if call.name in {"write_file", "edit_file", "materialize_artifact"} else ""
        pre_bytes: bytes | None = None
        pre_hash: str | None = None
        mutation_before = (
            self._chat_workspace_hashes(self.workspace)
            if call.name in MUTATING_TOOLS else {}
        )
        if pre_path:
            pre_candidate = (self.workspace / pre_path).resolve(strict=False)
            if pre_candidate.is_file() and pre_candidate.is_relative_to(self.workspace):
                pre_bytes = pre_candidate.read_bytes()
                pre_hash = hashlib.sha256(pre_bytes).hexdigest()
        try:
            with tools.workspace_context(self.workspace):
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
            mutation_after = (
                self._chat_workspace_hashes(self.workspace)
                if terminal == "completed" and call.name in MUTATING_TOOLS else mutation_before
            )
            actual_changed_files = [
                path for path in sorted(set(mutation_before) | set(mutation_after))
                if mutation_before.get(path) != mutation_after.get(path)
            ]
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
                self.store.update_goal_metadata(goal.id, goal_change_sets=changes)
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
            attempts.append({
                "signature": signature.fingerprint,
                "approach_fingerprint": approach_fingerprint,
                "operation": call.name,
            })
            metadata_update: dict[str, Any] = {
                "failed_hypotheses": failures[-20:], "failed_attempts": attempts[-50:]
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

    def _maybe_complete_harness_fallback_task(self, goal: Goal, plan: Plan) -> bool:
        """Close a narrow fallback task from authoritative artifact evidence.

        Weak models often keep re-reading a proven file instead of emitting the
        checklist control call.  For the single-artifact fallback only, the
        harness can evaluate the exact deterministic gates it authored.
        """

        if plan.proposed_by != "harness-weak-model-fallback" or len(plan.tasks) != 1:
            return False
        task = plan.tasks[0]
        if task.status not in {TaskStatus.READY, TaskStatus.IN_PROGRESS, TaskStatus.VERIFYING}:
            return False
        if not plan.expected_changes:
            return False
        relative = str(plan.expected_changes[0].get("path") or "").strip()
        candidate = (self.workspace / relative).resolve(strict=False)
        if not relative or not candidate.is_file() or not candidate.is_relative_to(self.workspace):
            return False
        evidence = [
            item
            for item in self.store.list_evidence(goal.id, task_id=task.id)
            if item.plan_revision == plan.revision and item.verified
        ]
        tools_seen = {str(item.data.get("tool") or "") for item in evidence}
        if "read_file" not in tools_seen:
            return False
        objective = goal.objective.casefold()
        if relative.casefold().endswith((".html", ".htm")):
            if not tools_seen.intersection({"preview_html", "inspect_preview"}):
                return False
            preview_results = [
                str(item.data.get("result") or "")
                for item in evidence
                if str(item.data.get("tool") or "") in {"preview_html", "inspect_preview"}
            ]
            if not any(
                ('"verification": "passed"' in result or '"status": "passed"' in result)
                and '"console_errors": []' in result
                and '"page_errors": []' in result
                and '"network_errors": []' in result
                for result in preview_results
            ):
                return False
            html = candidate.read_text(encoding="utf-8", errors="replace")
            lowered = html.casefold()
            if "<!doctype html" not in lowered or "<html" not in lowered:
                return False
            if "accessible" in objective and not (re.search(r"<html\b[^>]*\blang=", lowered) and "<button" in lowered):
                return False
            if "visible" in objective and "counter" in objective:
                if not (
                    re.search(r"id=[\"']counter[\"'][^>]*>\s*\d+", lowered)
                    and "<button" in lowered
                    and re.search(r"(?:addEventListener|onclick|increment)", html, re.IGNORECASE)
                ):
                    return False
        self.store.transition_task(
            goal.id,
            plan.revision,
            task.id,
            TaskStatus.COMPLETED,
            note="harness-authored fallback verification gates passed",
            evidence=(
                f"Harness verified {relative}: saved artifact read-back and required runtime/static gates passed.",
            ),
            actor="harness-fallback-verifier",
        )
        self.store.append_event(
            "task.fallback_verified",
            goal_id=goal.id,
            entity_type="task",
            entity_id=task.id,
            payload={"path": relative, "tools": sorted(tools_seen)},
        )
        return True

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
        return None

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
                            {"kind": item.kind, "summary": item.summary[:1_000], "verified": item.verified}
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
                                }
                                for item in task_evidence[-3:]
                            ],
                        },
                        "FINAL_REVIEW_TASK",
                        max_chars=30_000,
                    ),
                }
            )
        schemas = [*_schemas(READ_ONLY_TOOLS), *REVIEWER_SCHEMAS]
        for step in range(1, self.config.review_steps + 1):
            turn = self._call_provider(
                conversation,
                schemas,
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
                "user_review_required": visual_target and not vision_available,
                "routing_order": [
                    "deterministic_verification", "static_analysis", "runtime_integration",
                    "artifact_structure", "independent_review",
                    "vision_evaluation" if vision_available else "user_review",
                ],
            }
            convergence_state = "user_review_required" if visual_target and not vision_available else "converged"
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
                "confidence": "low" if convergence_state == "user_review_required" else "high",
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
            )
            self.store.append_event(
                "quality_convergence.decided", goal_id=goal.id,
                payload={"state": convergence_state, "mutation_sequence": mutation_sequence, "artifact_hashes": hashes},
            )
            if convergence_state == "user_review_required":
                self.store.transition_goal(
                    goal.id, GoalStatus.PAUSED,
                    reason="deterministic and independent checks passed, but subjective visual quality lacks a trustworthy evaluator",
                    metadata={
                        "waiting_question": "Review the latest visual artifact and explicitly accept it or provide refinement feedback.",
                        "resume_status": GoalStatus.RUNNING.value,
                    },
                )
                return "Candidate verified structurally, but subjective visual quality requires user review; it was not released as a verified final result."
            self.store.transition_goal(
                goal.id,
                GoalStatus.COMPLETED,
                reason="all checklist evidence passed independent final review",
                metadata={"completion_summary": redact_text(args["summary"], 4_000)},
            )
            self._record_global_learning(
                self.store.get_goal(goal.id),
                succeeded=True,
                evidence_ref=f"goal:{goal.id}:evaluation:{evaluation_record.get('evaluated_at_unix')}",
            )
            self.store.save_workflow_session(
                self.session_id,
                goal_id=goal.id,
                session_mode=SessionMode.GOAL.value,
                plan_state=PlanState.APPROVED.value,
                run_state=RunState.COMPLETED.value,
                state={"plan_revision": plan.revision, "completion": "evidence_gate_passed"},
            )
            self.events.publish("phase", "Goal completed after evidence gate and independent review.")
            return "Goal completed. The harness accepted the independent review."

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
        accepted_repair = self.approve_plan(new_plan.revision, approved_by="harness-quality-convergence")
        self.store.update_goal_metadata(goal.id, refinement_actions=actions, convergence_state="refining")
        return (
            f"Independent review found {len(repair_tasks)} deficiency task(s). "
            f"Harness-approved in-scope repair plan r{accepted_repair.revision}; Goal refinement continues autonomously."
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
        except (DomainError, RuntimeErrorBase, StateStoreError, ValueError) as exc:
            return f"Error: {call.name} transition rejected: {redact_text(exc, 1_500)}"

    def run_slice(self, steps: int | None = None) -> SliceResult:
        with self._lock:
            goal = self.active_goal()
            if goal is None:
                raise RuntimeStateError("no active goal")
            if goal.status != GoalStatus.RUNNING:
                return SliceResult(
                    goal.status.value,
                    f"Goal is {goal.status.value}; it cannot execute until the required user action occurs.",
                    needs_user=goal.status in {GoalStatus.AWAITING_PLAN_APPROVAL, GoalStatus.PAUSED},
                )
            plan = self.store.get_accepted_plan(goal.id)
            if plan is None or plan.revision != goal.active_plan_revision:
                raise RuntimeStateError("running goal has no matching accepted plan")
            budget = steps or self.config.work_quantum_steps
            if budget < 1:
                raise ValueError("slice steps must be positive")
            self._delegations_this_slice = 0
            no_action = 0
            made_progress = False
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
                request_conversation = [
                    *self._work_conversation,
                    {
                        "role": "user",
                        "content": state_envelope(
                            {
                                **self._state_payload(goal, plan),
                                "harness_selected_task": None if selected is None else _task_dict(selected),
                                "selection_rule": "Work only on the harness-selected first dependency-ready task.",
                            }
                        ),
                    },
                ]
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
                    current = self._schedule_goal_retry(
                        self.store.get_goal(goal.id),
                        f"provider unavailable after bounded transport retries: {redact_text(exc, 500)}",
                    )
                    self.events.publish("error", str(exc))
                    return SliceResult(
                        GoalStatus.RUNNING.value,
                        f"Provider unavailable; durable goal retry {current.metadata.get('goal_attempt')} scheduled.",
                        completed_steps,
                        needs_user=False,
                    )
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
                    if not result.startswith("Error:") and not result.startswith("Permission denied"):
                        made_progress = True

                refreshed_goal = self.store.get_goal(goal.id)
                refreshed_plan = self.store.get_latest_plan(goal.id)
                if self._maybe_complete_harness_fallback_task(refreshed_goal, refreshed_plan):
                    made_progress = True
                    self._work_conversation.append(
                        {
                            "role": "user",
                            "content": (
                                "HARNESS FALLBACK VERIFICATION PASSED: the selected checklist task is now completed "
                                "from authoritative read-back and runtime evidence. Request finish_goal with a concise "
                                "evidence summary; do not repeat file reads."
                            ),
                        }
                    )

                durable_checkpoint = state_envelope(self._state_payload(
                    self.store.get_goal(goal.id),
                    self.store.get_latest_plan(goal.id),
                ))
                self._work_conversation = context.suspend_and_revive(
                    self._work_conversation,
                    durable_checkpoint,
                    self.provider.summarize,
                    max_chars=self.config.conversation_chars,
                    on_suspend=lambda count: self.events.publish(
                        "checkpoint",
                        f"Suspended {count} transient messages and revived a fresh model context from durable goal memory.",
                    ),
                )
                if self.store.get_goal(goal.id).status != GoalStatus.RUNNING:
                    break

            current = self.store.get_goal(goal.id)
            if current.status == GoalStatus.RUNNING:
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
            needs_user = current.status in {GoalStatus.AWAITING_PLAN_APPROVAL, GoalStatus.PAUSED}
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
            self.events.publish("checkpoint", message)
            return SliceResult(current.status.value, message, completed_steps, completed, needs_user)

    def dashboard(self) -> DashboardView:
        goal = self.active_goal() or self.store.get_latest_goal()
        if goal is None:
            return DashboardView(
                provider=self.provider_name,
                model=self.model_name,
                workspace=str(self.workspace),
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
            waiting_question=str(goal.metadata.get("waiting_question", "")),
            activity=activity,
        )

    def apply_command(self, command: UserCommand) -> Any:
        kind, args = command.kind, command.args
        if kind == CommandKind.ANSWER:
            if self.store.get_pending_intake(self.session_id) is not None:
                return self.answer_intake_question(args["question_id"], args["value"])
            goal = self.active_goal()
            if goal and goal.metadata.get("ultra_run_id"):
                return self.answer_ultra_question(args["question_id"], args["value"])
            return self.answer_plan_question(args["question_id"], args["value"])
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
                mode = self.store.get_workflow_session(self.session_id)["session_mode"]
                return self.submit_intent(text, requested_mode=mode)
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

        def replace(match: re.Match[str]) -> str:
            language = (match.group(1) or "text").casefold()
            content = match.group(2)
            if len(content) < 2_048 and "<!doctype html" not in content.casefold():
                return match.group(0)
            names = {
                "html": "index.html", "javascript": "generated.js", "js": "generated.js",
                "python": "generated.py", "py": "generated.py", "css": "styles.css",
            }
            artifact = self.store.add_chat_artifact(
                self.session_id, content, language=language,
                suggested_name=names.get(language, "generated.txt"),
            )
            artifacts.append(artifact)
            return (
                f"<CHAT_ARTIFACT id=\"{artifact['id']}\" language=\"{language}\" "
                f"suggested_name=\"{artifact['suggested_name']}\" sha256=\"{artifact['content_hash']}\" "
                f"bytes=\"{artifact['byte_size']}\" />"
            )

        compact = pattern.sub(replace, original)
        if not artifacts and len(original) >= 2_048 and "<!doctype html" in original.casefold():
            start = original.casefold().find("<!doctype html")
            content = original[start:]
            artifact = self.store.add_chat_artifact(
                self.session_id, content, language="html", suggested_name="index.html",
            )
            artifacts.append(artifact)
            compact = (
                f"<CHAT_ARTIFACT id=\"{artifact['id']}\" language=\"html\" suggested_name=\"index.html\" "
                f"sha256=\"{artifact['content_hash']}\" bytes=\"{artifact['byte_size']}\" />"
            )
        return compact, tuple(artifacts)

    def _execute_chat_tool(
        self,
        call: ToolCall,
        intent: ChatIntentV1,
    ) -> tuple[tools.ToolExecutionResult, tuple[str, ...]]:
        spec = tools.get_spec(call.name)
        args = call.args if isinstance(call.args, dict) else {}
        if spec is None:
            return tools.ToolExecutionResult(False, f"Error: unknown chat tool {call.name!r}"), ()
        risk = spec.risk
        normal_requirement = tools.requires_approval(call.name, args)
        needs_approval = (
            self.permission_adapter.requires_approval(normal_requirement)
            if self.permission_adapter is not None else normal_requirement
        )
        if call.name == "open_path" or (call.name == "preview_html" and bool(args.get("open_browser", True))):
            needs_approval = True
        if intent.authorizes(call.name):
            needs_approval = False
        action_id = self.store.begin_session_action(
            self.session_id, call.name, redact_data(args), risk=risk,
            mutating=spec.mutates_workspace,
        )
        self.events.publish("tool_call", call.name, args=redact_data(args), actor="chat", id=call.id)
        if needs_approval and not self.approval(call.name, dict(args), risk):
            result = tools.ToolExecutionResult(False, "Permission denied by the user.", error_code="permission")
            self.store.complete_session_action(action_id, result.output, status="denied")
            return result, ()

        candidate_paths = [
            str(args.get(field, "")).strip()
            for field in spec.path_fields
            if str(args.get(field, "")).strip() not in {"", "."}
        ]
        if call.name == "apply_patch":
            for match in re.finditer(r"(?m)^\+\+\+\s+(?:b/)?([^\t\r\n]+)", str(args.get("patch", ""))):
                if match.group(1) != "/dev/null":
                    candidate_paths.append(match.group(1).strip())
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

    def chat(self, text: str, *, steps: int = 12) -> SliceResult:
        """Run ordinary Chat with durable artifacts and action postcondition gates."""

        prompt = str(text).strip()
        if not prompt:
            return SliceResult("idle", "", 0)
        user_message = {"role": "user", "content": prompt}
        self._chat_conversation.append(user_message)
        self.store.append_chat_message(self.session_id, user_message)
        session = self.store.get_workflow_session(self.session_id)
        session_state = dict(session.get("state", {}))
        session_state.setdefault("run_id", f"run-{hashlib.sha256((prompt + str(time.time_ns())).encode()).hexdigest()[:20]}")
        session_state.setdefault("original_objective", prompt)
        session_state.setdefault("user_messages", [])
        session_state["user_messages"] = [*session_state["user_messages"], prompt][-50:]

        intent = ChatIntentV1.parse(prompt)
        known_artifacts = self.store.list_chat_artifacts(self.session_id)
        latest_html = next((item for item in reversed(known_artifacts) if item.get("language") == "html"), None)
        if intent.requires_run and latest_html is not None:
            suggested = str(latest_html.get("suggested_name") or "index.html")
            if not (self.workspace / suggested).exists():
                intent = ChatIntentV1(prompt, requires_write=True, requires_run=True, requires_install=intent.requires_install)

        capability_rows = tools.capability_report()
        capabilities = json.dumps(capability_rows, ensure_ascii=False)
        artifact_rows = [
            {key: item.get(key) for key in ("id", "language", "suggested_name", "content_hash", "byte_size")}
            for item in known_artifacts[-10:]
        ]
        system = CHAT_SYSTEM_PROMPT + "\n\nRUNTIME CAPABILITIES:\n" + capabilities
        if artifact_rows:
            system += "\n\nDURABLE CHAT ARTIFACTS:\n" + json.dumps(artifact_rows, ensure_ascii=False)

        executed = 0
        changed_paths: list[str] = []
        successful_tools: list[str] = []
        successful_outputs: list[tuple[str, str]] = []
        failure_outputs: list[str] = []
        no_action_attempts = 0
        final_text = ""

        for step in range(1, max(1, steps) + 1):
            turn = self._call_provider(
                self._chat_conversation,
                list(tools.TOOL_SCHEMAS),
                system,
                actor="chat",
                step=step,
                stream_text=False,
            )
            message = turn.to_message()
            display_text = turn.text or ""
            if turn.text:
                compact_text, created = self._artifactize_chat_text(turn.text)
                message["content"] = compact_text
                if created:
                    session_state["chat_artifact_ids"] = list(dict.fromkeys([
                        *session_state.get("chat_artifact_ids", []), *(item["id"] for item in created)
                    ]))
            self._chat_conversation.append(message)
            self.store.append_chat_message(self.session_id, message)

            if turn.tool_calls:
                for call in turn.tool_calls:
                    result, changed = self._execute_chat_tool(call, intent)
                    executed += 1
                    tool_message = {"role": "tool", "id": call.id, "name": call.name, "content": result.output}
                    self._chat_conversation.append(tool_message)
                    self.store.append_chat_message(self.session_id, tool_message)
                    self.events.publish("tool_result", result.output, tool=call.name, actor="chat", id=call.id)
                    if result.ok:
                        successful_tools.append(call.name)
                        successful_outputs.append((call.name, result.output))
                        changed_paths.extend(changed)
                    else:
                        failure_outputs.append(result.output)
                continue

            missing = intent.missing(successful_tools)
            if missing:
                if intent.requires_run and "?" in display_text and latest_html is None:
                    candidates = [
                        path for path in self.workspace.rglob("*.htm*")
                        if ".coding-agent" not in path.parts
                    ]
                    if len(candidates) != 1:
                        # A real target choice is allowed; capability denial is not.
                        final_text = display_text
                        break
                no_action_attempts += 1
                if no_action_attempts >= self.config.no_action_limit:
                    detail = failure_outputs[-1] if failure_outputs else "the model repeatedly returned prose without using the available tools"
                    final_text = f"Action could not be completed: {detail}"
                    break
                correction = corrective_prompt(intent, missing, capabilities)
                correction_message = {"role": "user", "content": correction}
                self._chat_conversation.append(correction_message)
                self.store.append_chat_message(self.session_id, correction_message)
                continue

            final_text = display_text or "Completed the requested action."
            break

        if not final_text:
            final_text = f"Chat turn checkpointed after {executed} tool action(s); execution can continue in the same session."
        if successful_outputs and intent.actionable:
            final_text = final_text.rstrip() + "\n\nEvidence:\n" + self._chat_evidence(successful_outputs)

        artifacts = list(dict.fromkeys(changed_paths))
        if artifacts:
            session_state["below_target_continuation"] = {
                "status": "below_target",
                "objective": session_state["original_objective"],
                "target": "evidence-backed convergence",
                "weaknesses": ["candidate has not passed independent evaluation"],
                "expected_refinement_scope": artifacts,
                "artifacts": artifacts,
                "recommended_action": "continue the same run in Goal mode",
                "memory_preserved": True,
            }
            final_text += (
                "\n\nQuality status: BELOW_TARGET (not independently evaluated). "
                "The artifact and action evidence are preserved for Goal mode."
            )
        self.store.save_workflow_session(
            self.session_id, goal_id=None, session_mode=SessionMode.CHAT.value,
            plan_state=PlanState.NONE.value, run_state=RunState.IDLE.value, state=session_state,
        )
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
        goal = self.active_goal() or self.store.get_latest_goal()
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
