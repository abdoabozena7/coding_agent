"""Harness-owned workflow state, planning normalization, and retry accounting.

This module contains no provider code.  Models may supply draft content, but the
functions here own identifiers, dependencies, lifecycle transitions, and repair
budgets.  Keeping these rules provider-neutral makes Plan/Goal/Ultra behavior
repeatable and independently testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import re
from typing import Any, Iterable, Mapping
from datetime import datetime

from .models import utc_now

from .models import InvalidTransitionError


class SessionMode(str, Enum):
    NORMAL = "normal"
    ULTRA = "ultra"
    CHAT = "normal"
    PLAN = "normal"
    GOAL = "normal"

    @classmethod
    def parse(cls, value: str | "SessionMode") -> "SessionMode":
        if isinstance(value, cls):
            return value
        normalized = str(getattr(value, "value", value)).strip().casefold()
        normalized = {
            "chat": "normal", "plan": "normal", "goal": "normal",
            "manual": "normal", "default": "normal", "auto": "normal",
            "agent": "normal", "deep": "ultra", "max": "ultra",
        }.get(normalized, normalized)
        try:
            return cls(normalized)
        except ValueError as exc:
            raise ValueError("session mode must be normal or ultra") from exc


class PlanState(str, Enum):
    NONE = "none"
    INSPECTING = "inspecting"
    DRAFTING = "drafting"
    NORMALIZING = "normalizing"
    VALIDATING = "validating"
    REVIEWING = "reviewing"
    AWAITING_APPROVAL = "awaiting_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    FAILED = "failed"


class RunState(str, Enum):
    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    REVIEWING = "reviewing"
    VERIFYING = "verifying"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentState(str, Enum):
    CREATED = "created"
    READY = "ready"
    RUNNING = "running"
    WAITING_FOR_TOOL = "waiting_for_tool"
    WAITING_FOR_DEPENDENCY = "waiting_for_dependency"
    REVIEWING = "reviewing"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class UltraProfile(str, Enum):
    STANDARD = "standard"
    SLEEP = "sleep"


class SleepState(str, Enum):
    OFF = "off"
    ARMED = "armed"
    RUNNING = "running"
    BLOCKED = "blocked"
    COMPLETED = "completed"


_PLAN_TRANSITIONS: dict[PlanState, frozenset[PlanState]] = {
    PlanState.NONE: frozenset({PlanState.INSPECTING}),
    PlanState.INSPECTING: frozenset({PlanState.DRAFTING, PlanState.FAILED}),
    PlanState.DRAFTING: frozenset({PlanState.NORMALIZING, PlanState.FAILED}),
    PlanState.NORMALIZING: frozenset({PlanState.VALIDATING, PlanState.FAILED}),
    PlanState.VALIDATING: frozenset({PlanState.REVIEWING, PlanState.AWAITING_APPROVAL, PlanState.FAILED}),
    PlanState.REVIEWING: frozenset({PlanState.NORMALIZING, PlanState.VALIDATING, PlanState.FAILED}),
    PlanState.AWAITING_APPROVAL: frozenset({PlanState.APPROVED, PlanState.REJECTED}),
    PlanState.APPROVED: frozenset(),
    PlanState.REJECTED: frozenset(),
    PlanState.FAILED: frozenset(),
}


def ensure_plan_state_transition(current: PlanState, target: PlanState) -> None:
    if current == target:
        return
    if target not in _PLAN_TRANSITIONS[current]:
        raise InvalidTransitionError(
            f"invalid plan state transition: {current.value!r} -> {target.value!r}"
        )


class RetryKind(str, Enum):
    PROVIDER_TRANSPORT = "provider_transport"
    TYPED_PARSE_REPAIR = "typed_parse_repair"
    PLAN_FORMAT_REPAIR = "plan_format_repair"
    PLAN_SEMANTIC_REPAIR = "plan_semantic_repair"
    CRITIC_REVISION = "critic_revision"
    WORKER_RETURN_REPAIR = "worker_return_repair"
    REVIEW_VERDICT_REPAIR = "review_verdict_repair"
    EXECUTION_NO_PROGRESS = "execution_no_progress"
    VERIFICATION_RETRY = "verification_retry"
    SLEEP_APPROACH_ATTEMPT = "sleep_approach_attempt"


@dataclass(frozen=True, slots=True)
class RetryRecord:
    kind: RetryKind
    stage: str
    reason: str
    attempt: int
    input_fingerprint: str
    output_fingerprint: str
    progress: bool
    next_action: str


@dataclass
class RetryLedger:
    """Separate retry counters with a structured audit trail."""

    counts: dict[RetryKind, int] = field(default_factory=dict)
    records: list[RetryRecord] = field(default_factory=list)

    def record(
        self,
        kind: RetryKind,
        *,
        stage: str,
        reason: str,
        input_value: Any = None,
        output_value: Any = None,
        progress: bool = False,
        next_action: str = "stop",
    ) -> RetryRecord:
        attempt = self.counts.get(kind, 0) + 1
        self.counts[kind] = attempt
        item = RetryRecord(
            kind=kind,
            stage=stage,
            reason=str(reason),
            attempt=attempt,
            input_fingerprint=fingerprint(input_value),
            output_fingerprint=fingerprint(output_value),
            progress=bool(progress),
            next_action=str(next_action),
        )
        self.records.append(item)
        return item


def fingerprint(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class PlanValidationIssue:
    path: str
    message: str

    def __str__(self) -> str:
        return f"{self.path}: {self.message}"


class PlanDraftError(ValueError):
    def __init__(self, issues: Iterable[PlanValidationIssue], stage: str = "semantic_validation") -> None:
        self.issues = tuple(issues)
        self.stage = stage
        super().__init__("; ".join(str(item) for item in self.issues))


def _text(value: Any) -> str:
    return str(value or "").strip()


def _unique_text(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, (list, tuple)):
        values = [values]
    result: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = _text(raw)
        key = value.casefold()
        if value and key not in seen:
            seen.add(key)
            result.append(value)
    return result


_EARLIER_TASK = re.compile(r"^(?:task\s*)?(\d+)$", re.IGNORECASE)
_STABLE_TASK = re.compile(r"^T(\d{1,3})$", re.IGNORECASE)


def _dependency_number(value: Any, legacy_ids: Mapping[str, int]) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    text = _text(value)
    if not text:
        return None
    match = _EARLIER_TASK.fullmatch(text) or _STABLE_TASK.fullmatch(text)
    if match:
        return int(match.group(1))
    return legacy_ids.get(text.casefold())


def normalize_plan_draft(raw: Mapping[str, Any]) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Mechanically normalize a simplified or legacy proposal.

    Stable task IDs and all cross references are generated here.  A missing
    description is repaired only when title, observable criteria, and
    verification already make the task semantically complete; otherwise
    missing objectives, criteria, or verification remain validation errors for
    one targeted repair.
    """

    actions: list[str] = []
    tasks_raw = raw.get("tasks", ())
    if not isinstance(tasks_raw, (list, tuple)):
        tasks_raw = [tasks_raw]
        actions.append("/tasks converted to an array")
    legacy_ids: dict[str, int] = {}
    for index, item in enumerate(tasks_raw, 1):
        if isinstance(item, Mapping) and _text(item.get("id")):
            legacy_ids[_text(item.get("id")).casefold()] = index

    tasks: list[dict[str, Any]] = []
    for index, raw_task in enumerate(tasks_raw, 1):
        item = dict(raw_task) if isinstance(raw_task, Mapping) else {}
        task_id = f"T{index:03d}"
        if _text(item.get("id")) != task_id:
            actions.append(f"/tasks/{index - 1}/id generated as {task_id}")
        verification = _unique_text(item.get("verification"))
        if isinstance(item.get("verification"), str):
            actions.append(f"/tasks/{index - 1}/verification converted to an array")
        acceptance = _unique_text(item.get("acceptance_criteria", item.get("acceptance")))
        expected = _unique_text(item.get("expected_changes", item.get("changes")))
        dependencies_raw = item.get("depends_on", item.get("dependencies", ()))
        if dependencies_raw is None:
            dependencies_raw = []
        if not isinstance(dependencies_raw, (list, tuple)):
            dependencies_raw = [dependencies_raw]
        dependencies: list[str] = []
        unresolved: list[str] = []
        for raw_dependency in dependencies_raw:
            number = _dependency_number(raw_dependency, legacy_ids)
            if number is None:
                unresolved.append(_text(raw_dependency))
                continue
            dependencies.append(f"T{number:03d}")
        dependencies = list(dict.fromkeys(dependencies))
        risk = _text(item.get("risk", "medium")).lower()
        title = _text(item.get("title"))
        description = _text(item.get("description", item.get("objective")))
        if not description and title and acceptance and verification:
            description = (
                f"Complete {title} and satisfy its observable acceptance criteria "
                "using the specified verification."
            )
            actions.append(f"/tasks/{index - 1}/description derived from complete task contract")
        tasks.append(
            {
                "id": task_id,
                "title": title,
                "description": description,
                "expected_changes": expected,
                "acceptance_criteria": acceptance,
                "verification": verification,
                "depends_on": dependencies,
                "risk": risk or "medium",
                "_unresolved_dependencies": unresolved,
            }
        )

    all_ids = [item["id"] for item in tasks]
    applicability: list[dict[str, Any]] = []
    for evidence in raw.get("applicability_evidence", ()) or ():
        if not isinstance(evidence, Mapping):
            continue
        fact = _text(evidence.get("fact"))
        source = _text(evidence.get("source"))
        if fact:
            applicability.append({"fact": fact, "source": source, "supports_tasks": all_ids})
    expected_changes: list[dict[str, Any]] = []
    for change in raw.get("expected_changes", ()) or ():
        if isinstance(change, Mapping):
            path, intent = _text(change.get("path")), _text(change.get("intent"))
        else:
            path, intent = "", _text(change)
        if path or intent:
            expected_changes.append({"path": path or "<resolved during execution>", "intent": intent, "supports_tasks": all_ids})
    if not expected_changes:
        for task in tasks:
            for change in task.pop("expected_changes"):
                expected_changes.append({"path": "<resolved during execution>", "intent": change, "supports_tasks": [task["id"]]})
    else:
        for task in tasks:
            task.pop("expected_changes", None)

    normalized = {
        "summary": _text(raw.get("summary", raw.get("objective"))),
        "applicability_evidence": applicability,
        "execution_strategy": _text(raw.get("execution_strategy", raw.get("strategy"))),
        "expected_changes": expected_changes,
        "tasks": tasks,
    }
    return normalized, tuple(dict.fromkeys(actions))


def validate_normalized_plan(value: Mapping[str, Any]) -> None:
    issues: list[PlanValidationIssue] = []
    if not _text(value.get("summary")):
        issues.append(PlanValidationIssue("/summary", "a non-empty objective summary is required"))
    tasks = value.get("tasks", ())
    if not isinstance(tasks, (list, tuple)) or not tasks:
        issues.append(PlanValidationIssue("/tasks", "at least one task is required"))
        raise PlanDraftError(issues)
    ids = {str(item.get("id")) for item in tasks if isinstance(item, Mapping)}
    for index, item in enumerate(tasks):
        path = f"/tasks/{index}"
        if not isinstance(item, Mapping):
            issues.append(PlanValidationIssue(path, "task must be an object"))
            continue
        if not _text(item.get("title")):
            issues.append(PlanValidationIssue(path + "/title", "non-empty title is required"))
        if not _text(item.get("description")):
            issues.append(PlanValidationIssue(path + "/description", "non-empty objective is required"))
        if not item.get("acceptance_criteria"):
            issues.append(PlanValidationIssue(path + "/acceptance_criteria", "at least one observable criterion is required"))
        if not item.get("verification"):
            issues.append(PlanValidationIssue(path + "/verification", "at least one verification requirement is required"))
        unresolved = item.get("_unresolved_dependencies", ())
        if unresolved:
            issues.append(PlanValidationIssue(path + "/depends_on", f"ambiguous dependency reference(s): {list(unresolved)!r}"))
        for dependency in item.get("depends_on", ()):
            if dependency not in ids:
                issues.append(PlanValidationIssue(path + "/depends_on", f"dependency {dependency!r} does not exist"))
            elif dependency >= str(item.get("id")):
                issues.append(PlanValidationIssue(path + "/depends_on", f"dependency {dependency!r} must refer to an earlier task"))
        if _text(item.get("risk")) not in {"low", "medium", "high", "critical"}:
            issues.append(PlanValidationIssue(path + "/risk", "must be low, medium, high, or critical"))
    if issues:
        raise PlanDraftError(issues)


_APPROVAL = re.compile(
    r"^(?:yes[,.!]?\s*)?(?:do it|go ahead|accept(?: it| the plan)?|approve(?: it| the plan)?|proceed|looks good|ship it)[.!\s]*$",
    re.IGNORECASE,
)


def is_unambiguous_plan_approval(text: str, *, pending_plans: int = 1) -> bool:
    return pending_plans == 1 and bool(_APPROVAL.fullmatch(str(text).strip()))


def first_ready_task(tasks: Iterable[Any]) -> Any | None:
    """Select the first dependency-ready task without a coordinator model call."""

    values = list(tasks)
    completed = {
        str(getattr(item, "id", item.get("id") if isinstance(item, Mapping) else ""))
        for item in values
        if str(getattr(getattr(item, "status", None), "value", getattr(item, "status", ""))) in {"completed", "done"}
    }
    for item in values:
        status = str(getattr(getattr(item, "status", None), "value", getattr(item, "status", "")))
        if status not in {"pending", "ready"}:
            continue
        dependencies = tuple(getattr(item, "depends_on", item.get("depends_on", ()) if isinstance(item, Mapping) else ()))
        if all(str(value) in completed for value in dependencies):
            return item
    return None


@dataclass(frozen=True, slots=True)
class WorkerContractV1:
    objective: str
    task_id: str
    task: Mapping[str, Any]
    parent_contract: Mapping[str, Any] = field(default_factory=dict)
    allowed_paths: tuple[str, ...] = ()
    expected_files: tuple[str, ...] = ()
    acceptance_criteria: tuple[str, ...] = ()
    permitted_tools: tuple[str, ...] = ()
    required_verification: tuple[str, ...] = ()
    existing_evidence: tuple[Mapping[str, Any], ...] = ()
    exclusions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.objective.strip() or not self.task_id.strip():
            raise ValueError("worker contract requires objective and task id")
        object.__setattr__(self, "task", dict(self.task))
        object.__setattr__(self, "parent_contract", dict(self.parent_contract))
        for name in ("allowed_paths", "expected_files", "acceptance_criteria", "permitted_tools", "required_verification", "exclusions"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        object.__setattr__(self, "existing_evidence", tuple(dict(item) for item in self.existing_evidence))


@dataclass(frozen=True, slots=True)
class AgentRegistryEntryV1:
    runtime_id: str
    display_index: int
    role: str
    state: AgentState
    provider: str
    model: str
    ultra_run_id: str | None = None
    assigned_id: str | None = None
    parent_runtime_id: str | None = None
    message_stream: tuple[Mapping[str, Any], ...] = ()
    prompt_trace_refs: tuple[str, ...] = ()
    tool_call_refs: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    blocker: str | None = None
    failure_reason: str | None = None
    usage: Mapping[str, Any] = field(default_factory=dict)
    started_at: datetime | None = None
    ended_at: datetime | None = None
    updated_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if not self.runtime_id or self.display_index < 1 or not self.role:
            raise ValueError("agent registry entry requires runtime id, positive display index, and role")
        object.__setattr__(self, "message_stream", tuple(dict(item) for item in self.message_stream))
        object.__setattr__(self, "usage", dict(self.usage))
