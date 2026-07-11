"""Deterministic orchestration above the probabilistic model/tool loop.

The model proposes plans, actions, dynamic roles, and completion.  This runtime
owns plan approval, state transitions, evidence, retries, recovery, delegation
limits, and the final completion gate.
"""

from __future__ import annotations

import copy
import json
import os
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Iterable, Mapping, Sequence

from . import context, tools
from .commands import CommandKind, UserCommand
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


ApprovalCallback = Callable[[str, dict[str, Any], str], bool]

READ_ONLY_TOOLS = frozenset({"read_file", "list_files", "grep"})
MUTATING_TOOLS = frozenset({"write_file", "edit_file", "run_bash"})
TOOL_RISK = {"read_file": "low", "list_files": "low", "grep": "low", "write_file": "high", "edit_file": "high", "run_bash": "critical"}


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
    ) -> None:
        self.provider = provider
        self.store = store
        self.workspace = Path(workspace).resolve(strict=True)
        self.events = events or EventBus()
        self.approval = approval or (lambda _name, _args, _risk: False)
        self.config = config or RuntimeConfig.from_env()
        self.sleeper = sleeper
        self._lock = RLock()
        self._work_conversation: list[dict[str, Any]] = []
        self._watchdog = ProgressWatchdog(self.config.repeated_action_limit)
        self._delegations_this_slice = 0
        self._provider_input_tokens = 0
        self._provider_output_tokens = 0
        tools.configure_workspace(self.workspace)

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

    def replace_config(self, config: RuntimeConfig) -> None:
        """Apply validated slice limits at an interactive command checkpoint."""

        if not isinstance(config, RuntimeConfig):
            raise TypeError("config must be a RuntimeConfig")
        with self._lock:
            self.config = config
            # Preserve the in-memory action history so changing an unrelated
            # display/runtime setting cannot clear the no-progress guardrail.
            self._watchdog.repeat_limit = max(1, self.config.repeated_action_limit)

    def active_goal(self) -> Goal | None:
        return self.store.load_active_goal()

    def latest_plan(self) -> Plan | None:
        goal = self.active_goal()
        return self.store.get_latest_plan(goal.id) if goal else None

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
    ) -> AssistantTurn:
        self.events.publish("step", actor=actor, step=step)
        last_error: Exception | None = None
        for attempt in range(self.config.max_provider_retries + 1):
            try:
                turn = self.provider.call(
                    conversation,
                    list(schemas),
                    system,
                    on_text=lambda fragment: self.events.publish("model_text", str(fragment), actor=actor),
                    on_thought=lambda fragment: self.events.publish("model_thought", str(fragment), actor=actor),
                )
                if not isinstance(turn, AssistantTurn):
                    raise TypeError(f"provider returned {type(turn).__name__}, expected AssistantTurn")
                for call in turn.tool_calls:
                    if not isinstance(call.args, dict):
                        call.args = {}
                self._emit_usage(turn)
                return turn
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as exc:
                last_error = exc
                message = redact_text(exc, 500)
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
            goal = self.store.create_goal(redact_text(objective, 20_000))
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
            source = str(item["source"]).strip()
            if not source.startswith("tool:") or source[5:] not in successful_inspection_ids:
                raise ValueError(
                    "each applicability fact must cite a successful earlier inspection as tool:CALL_ID"
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
        conversation: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": state_envelope(
                    {
                        "objective": goal.objective,
                        "workspace": str(self.workspace),
                        "user_feedback": feedback,
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
        successful_inspection_ids: set[str] = set()
        inspection_records: dict[str, dict[str, Any]] = {}
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
            for call in turn.tool_calls:
                self.events.publish("tool_call", call.name, args=redact_data(call.args), actor="planner")
                if call.name == "propose_plan":
                    try:
                        proposed = validate_control_call(call.name, call.args)
                        result = "Plan proposal captured for independent critique."
                    except ControlValidationError as exc:
                        result = f"Error: invalid plan proposal: {exc}"
                elif call.name in READ_ONLY_TOOLS:
                    result = self._execute_workspace_tool(goal, call, task_id=None, actor="planner")
                    if not result.startswith("Error:") and not result.startswith("Permission denied"):
                        successful_inspection_ids.add(call.id)
                        inspection_records[call.id] = {
                            "call_id": call.id,
                            "tool": call.name,
                            "arguments": redact_data(call.args),
                            "result": redact_text(result, 4_000),
                        }
                else:
                    result = f"Error: planning is read-only; tool '{call.name}' is unavailable before approval."
                conversation.append({"role": "tool", "id": call.id, "name": call.name, "content": result})
                self.events.publish("tool_result", result, tool=call.name, actor="planner")

            if proposed is not None:
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
                    conversation.append(
                        {
                            "role": "user",
                            "content": (
                                "Harness plan validation rejected the proposal: "
                                f"{redact_text(exc, 1_000)}. Repair the IDs/dependencies/criteria and call propose_plan again."
                            ),
                        }
                    )
                    proposed = None
                    continue
                critique = self._review_plan_candidate(goal, proposed, inspection_records)
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
                if revisions >= 3:
                    break
            elif not turn.tool_calls:
                conversation.append(
                    {
                        "role": "user",
                        "content": "Planning is not complete in prose. Inspect if needed, then call propose_plan with a validated plan.",
                    }
                )

        self.store.append_event(
            "planning.checkpoint",
            goal_id=goal.id,
            payload={"reason": "planner did not produce a critic-approved structured plan"},
        )
        self.events.publish(
            "warning",
            "The planner reached its bounded slice without a valid plan. The goal remains durable; use /replan with guidance.",
        )
        self._pause_planning(
            goal,
            "The planner did not produce a critic-approved structured plan in its bounded pass. Add guidance, then use /resume or /replan.",
            "bounded planning pass produced no valid plan",
        )
        return None

    def approve_plan(self, revision: int | None = None) -> Plan:
        goal = self.active_goal()
        plan = self.latest_plan()
        if goal is None or plan is None:
            raise RuntimeStateError("there is no plan to approve")
        requested = plan.revision if revision is None else revision
        accepted, _approval = self.store.approve_plan(
            goal.id,
            requested,
            approved_by="user",
            expected_fingerprint=plan.fingerprint if requested == plan.revision else None,
        )
        self._work_conversation = [
            {
                "role": "user",
                "content": f"The user approved plan r{accepted.revision}. Begin the first ready task and keep the checklist current.",
            }
        ]
        self.events.publish("phase", f"Plan r{accepted.revision} approved; execution is active.")
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
        return tuple(dict.fromkeys([*action_ids, *delegation_ids, *task_ids]))

    def pause(self, reason: str = "paused by user") -> Goal:
        goal = self.active_goal()
        if goal is None:
            raise RuntimeStateError("no active goal")
        if goal.status == GoalStatus.PAUSED:
            return goal
        self.store.update_goal_metadata(goal.id, resume_status=goal.status.value)
        result = self.store.transition_goal(goal.id, GoalStatus.PAUSED, reason=reason)
        self.events.publish("phase", "Goal paused safely; state is durable.")
        return result

    def resume(self) -> Goal:
        goal = self.active_goal()
        if goal is None or goal.status != GoalStatus.PAUSED:
            raise RuntimeStateError("goal is not paused")
        unresolved = self._unresolved_recovery_entities(goal.id)
        if unresolved:
            preview = ", ".join(unresolved[:5])
            suffix = " ..." if len(unresolved) > 5 else ""
            raise RuntimeStateError(
                "cannot resume while crash-window work is uncertain; inspect it and use "
                f"/resolve first ({preview}{suffix})"
            )
        desired = GoalStatus(goal.metadata.get("resume_status", GoalStatus.RUNNING.value))
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
        result = self.store.transition_goal(goal.id, desired, reason="resumed by user")
        self.events.publish("phase", f"Goal resumed in {desired.value}.")
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
            result = self.store.resolve_delegation(
                action_id, resolution, safe_note, actor="user"
            )
            entity = "delegation"
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
            "Interrupt checkpoint saved. No unfinished side effect was replayed.",
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
        args = call.args if isinstance(call.args, dict) else {}
        scoped_name = f"{actor}:{call.name}"
        journal_args = {
            "_harness_actor": actor,
            "_harness_plan_revision": goal.active_plan_revision,
            "arguments": redact_data(args),
        }
        if actor == "planner":
            journal_args["_harness_goal_attempt"] = int(goal.metadata.get("goal_attempt", 0))
        if self.store.count_recent_identical_actions(
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
        if decision.stalled:
            return f"Error: {decision.reason}"
        risk = TOOL_RISK.get(call.name, "unknown")
        needs_approval = tools.requires_approval(call.name, args)
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
        try:
            raw_result = tools.run_tool(call.name, args)
            result = redact_text(raw_result, 50_000)
            terminal = "failed" if result.startswith("Error:") else "completed"
            self.store.complete_action(action_id, redact_text(result, 2_000), status=terminal)
        except (KeyboardInterrupt, SystemExit):
            # Deliberately leave the action running; restart recovery will mark
            # the crash-window side effect uncertain instead of replaying it.
            raise
        except Exception as exc:
            result = f"Error: tool harness failure: {type(exc).__name__}: {redact_text(exc, 500)}"
            self.store.complete_action(action_id, result, status="failed")
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
            if task.status == TaskStatus.COMPLETED and not any(
                item.plan_revision == plan.revision
                for item in self.store.list_evidence(goal.id, task_id=task.id)
            ):
                return f"completed task {task.id} has no evidence"
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
            self.store.transition_goal(
                goal.id,
                GoalStatus.COMPLETED,
                reason="all checklist evidence passed independent final review",
                metadata={"completion_summary": redact_text(args["summary"], 4_000)},
            )
            self.events.publish("phase", "Goal completed after evidence gate and independent review.")
            return "Goal completed. The harness accepted the independent review."

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
        self.revise_plan(
            reason=f"independent review failed: {verdict['summary']}",
            add=repair_tasks,
            proposed_by="reviewer",
        )
        return "Independent review failed. Repair tasks were added as a new revision pending user approval."

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
                request_conversation = [
                    *self._work_conversation,
                    {"role": "user", "content": state_envelope(self._state_payload(goal, plan))},
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

                self._work_conversation = context.maybe_compact(
                    self._work_conversation,
                    self.provider.summarize,
                    max_chars=self.config.conversation_chars,
                    on_compact=lambda count: self.events.publish("checkpoint", f"Compacted {count} older messages; durable state was retained."),
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
        if kind == CommandKind.GOAL:
            return self.start_goal(args["objective"])
        if kind == CommandKind.APPROVE:
            return self.approve_plan(args["revision"])
        if kind in {CommandKind.REJECT, CommandKind.REPLAN}:
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
            return self.start_goal(text) if self.active_goal() is None else self.add_guidance(text)
        return None
