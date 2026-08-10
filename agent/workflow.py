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
import os
import re
from typing import Any, Iterable, Mapping
from datetime import datetime

from .models import utc_now

from .models import InvalidTransitionError


class SessionMode(str, Enum):
    NORMAL = "normal"
    PLAN = "plan"
    ULTRA = "ultra"
    CHAT = "normal"
    GOAL = "normal"

    @classmethod
    def parse(cls, value: str | "SessionMode") -> "SessionMode":
        if isinstance(value, cls):
            return value
        normalized = str(getattr(value, "value", value)).strip().casefold()
        normalized = {
            "chat": "normal", "goal": "normal",
            "manual": "normal", "default": "normal", "auto": "normal",
            "agent": "normal", "working": "normal", "work": "normal",
            "ultra-plan": "plan", "ultra_plan": "plan", "ultraplan": "plan",
            "deep": "ultra", "max": "ultra",
        }.get(normalized, normalized)
        try:
            return cls(normalized)
        except ValueError as exc:
            raise ValueError("session mode must be plan, normal, or ultra") from exc


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
    PLAN_QUESTION_REPAIR = "plan_question_repair"
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
    stage_counts: dict[tuple[RetryKind, str], int] = field(default_factory=dict)
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
        self.counts[kind] = self.counts.get(kind, 0) + 1
        stage_key = (kind, str(stage))
        attempt = self.stage_counts.get(stage_key, 0) + 1
        self.stage_counts[stage_key] = attempt
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


@dataclass(frozen=True, slots=True)
class WorkflowStageCheckpointV1:
    """Durable, JSON-storable boundary for an independently repairable stage."""

    stage: str
    substage: str = ""
    category: str = ""
    field_path: str = ""
    message: str = ""
    received: Any = None
    expected: tuple[str, ...] = ()
    attempts: int = 0
    rejected_fingerprint: str = ""
    accepted_route_fingerprint: str = ""
    semantic_fingerprint: str = ""
    inspection_refs: tuple[str, ...] = ()
    resumable: bool = True
    version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "substage": self.substage,
            "category": self.category,
            "field_path": self.field_path,
            "message": self.message,
            "received": self.received,
            "expected": list(self.expected),
            "attempts": self.attempts,
            "rejected_fingerprint": self.rejected_fingerprint,
            "accepted_route_fingerprint": self.accepted_route_fingerprint,
            "semantic_fingerprint": self.semantic_fingerprint,
            "inspection_refs": list(self.inspection_refs),
            "resumable": self.resumable,
            "version": self.version,
        }


class WorkflowBoundaryKind(str, Enum):
    CONTRACT_INCOMPATIBILITY = "contract_incompatibility"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    SEMANTIC_CONFLICT = "semantic_conflict"
    QUALITY_BLOCKER = "quality_blocker"
    PERMISSION_REQUIRED = "permission_required"
    EXECUTION_UNCERTAIN = "execution_uncertain"
    NO_PROGRESS = "no_progress"


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


def _windows_verification_step(value: str) -> str:
    """Translate an unambiguous POSIX file-content check to Windows cmd."""

    text = _text(value)
    lowered = text.casefold()
    if "cat " not in lowered or "grep" not in lowered:
        return text
    path_match = re.search(r"\bcat\s+['\"]?([./\\\w.-]+)", text, re.IGNORECASE)
    expected_match = re.search(
        r"grep\s+(?:-q\s+)?['\"]?\^?([^'\"$]+)\$?['\"]?",
        text,
        re.IGNORECASE,
    )
    if path_match is None or expected_match is None:
        return text
    path = path_match.group(1).replace("\\", "/").removeprefix("./")
    expected = expected_match.group(1).strip()
    if not path or not expected:
        return text
    escaped_path = path.replace("'", "''")
    escaped_expected = expected.replace("'", "''")
    return (
        "Run python -c \"from pathlib import Path; assert "
        f"Path('{escaped_path}').read_text(encoding='utf-8') == '{escaped_expected}'\" "
        "and require exit code 0."
    )


_EARLIER_TASK = re.compile(r"^(?:task\s*)?(\d+)$", re.IGNORECASE)
_STABLE_TASK = re.compile(r"^T(\d{1,3})$", re.IGNORECASE)


def _dependency_number(value: Any, legacy_ids: Mapping[str, int]) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, Mapping):
        keys = [str(key).strip() for key in value if str(key).strip()]
        if len(keys) != 1:
            return None
        value = keys[0]
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
    verification array may be projected only from the task's already-authored
    observable acceptance criteria; this supplies transport shape without
    weakening the independent critic or final fresh-evidence gate.
    """

    actions: list[str] = []
    tasks_raw = raw.get("tasks", ())
    if not isinstance(tasks_raw, (list, tuple)):
        tasks_raw = [tasks_raw]
        actions.append("/tasks converted to an array")
    legacy_ids: dict[str, int] = {}
    title_positions: dict[str, list[int]] = {}
    for index, item in enumerate(tasks_raw, 1):
        if isinstance(item, Mapping) and _text(item.get("id")):
            legacy_ids[_text(item.get("id")).casefold()] = index
        if isinstance(item, Mapping) and _text(item.get("title")):
            title_positions.setdefault(
                _text(item.get("title")).casefold(), []
            ).append(index)
    for title, positions in title_positions.items():
        if len(positions) == 1:
            legacy_ids[title] = positions[0]

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
        if not verification and acceptance:
            verification_source = " ".join(
                (
                    _text(item.get("title")),
                    _text(item.get("description", item.get("objective"))),
                    " ".join(acceptance),
                    _text(raw.get("execution_strategy")),
                )
            ).casefold()
            if "python -m pytest" in verification_source:
                verification = [
                    "Run python -m pytest and require exit code 0."
                ]
            elif re.search(r"\bpytest\b", verification_source):
                verification = ["Run pytest and require exit code 0."]
            else:
                verification = [
                    "Collect fresh executable, inspection, or managed-preview "
                    f"evidence for this accepted criterion: {criterion}"
                    for criterion in acceptance
                ]
            if verification:
                actions.append(
                    f"/tasks/{index - 1}/verification projected from an "
                    "explicit command or observable acceptance criterion in "
                    "its task contract"
                )
        if os.name == "nt" and verification:
            platform_verification = [_windows_verification_step(item) for item in verification]
            if platform_verification != verification:
                verification = platform_verification
                actions.append(
                    f"/tasks/{index - 1}/verification normalized for Windows cmd"
                )
        expected = _unique_text(item.get("expected_changes", item.get("changes")))
        requirement_refs = [value.upper() for value in _unique_text(item.get("requirement_refs"))]
        dependencies_raw = item.get("depends_on", item.get("dependencies", ()))
        if dependencies_raw is None:
            dependencies_raw = []
        if not isinstance(dependencies_raw, (list, tuple)):
            dependencies_raw = [dependencies_raw]
        dependencies: list[str] = []
        unresolved: list[str] = []
        for raw_dependency in dependencies_raw:
            if raw_dependency is None or not _text(raw_dependency):
                actions.append(
                    f"/tasks/{index - 1}/depends_on dropped an empty dependency"
                )
                continue
            if isinstance(raw_dependency, Mapping) and not raw_dependency:
                actions.append(
                    f"/tasks/{index - 1}/depends_on dropped an empty dependency object"
                )
                continue
            number = _dependency_number(raw_dependency, legacy_ids)
            if number is None:
                unresolved.append(_text(raw_dependency))
                continue
            if isinstance(raw_dependency, Mapping):
                actions.append(
                    f"/tasks/{index - 1}/depends_on used the singleton dependency object's key"
                )
            dependency_key = _text(raw_dependency).casefold()
            if isinstance(raw_dependency, Mapping):
                dependency_key = next(iter(raw_dependency)).strip().casefold()
            if (
                dependency_key in title_positions
                and len(title_positions[dependency_key]) == 1
            ):
                actions.append(
                    f"/tasks/{index - 1}/depends_on resolved exact task title "
                    f"{raw_dependency!r} to T{number:03d}"
                )
            dependencies.append(f"T{number:03d}")
        dependencies = list(dict.fromkeys(dependencies))
        risk = _text(item.get("risk", "medium")).lower()
        risk_prefix = re.match(r"^(low|medium|high|critical)\b", risk)
        if risk_prefix and risk != risk_prefix.group(1):
            risk = risk_prefix.group(1)
            actions.append(
                f"/tasks/{index - 1}/risk normalized to {risk}"
            )
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
                "requirement_refs": requirement_refs,
                "acceptance_criteria": acceptance,
                "verification": verification,
                "depends_on": dependencies,
                "risk": risk or "medium",
                "_unresolved_dependencies": unresolved,
            }
        )

    all_ids = [item["id"] for item in tasks]

    def normalized_supports(raw_supports: Any) -> list[str]:
        values = raw_supports
        if values is None:
            values = []
        if not isinstance(values, (list, tuple)):
            values = [values]
        result: list[str] = []
        for raw_support in values:
            number = _dependency_number(raw_support, legacy_ids)
            if number is not None and 1 <= number <= len(all_ids):
                result.append(f"T{number:03d}")
        if not result and len(all_ids) == 1:
            result = ["T001"]
        return list(dict.fromkeys(result))

    applicability: list[dict[str, Any]] = []
    for evidence in raw.get("applicability_evidence", ()) or ():
        if not isinstance(evidence, Mapping):
            continue
        fact = _text(evidence.get("fact"))
        source = _text(evidence.get("source"))
        if fact:
            applicability.append(
                {
                    "fact": fact,
                    "source": source,
                    "supports_tasks": normalized_supports(
                        evidence.get("supports_tasks")
                    ),
                }
            )
    expected_changes: list[dict[str, Any]] = []
    raw_expected_changes = list(raw.get("expected_changes", ()) or ())
    # A few local tool-call models understand the per-task shape more readily
    # than the canonical plan-level mutation contract.  Lift those authored
    # objects mechanically when the plan-level array is omitted; this keeps
    # the path, intent, basis, and evidence model-owned while giving the
    # deterministic semantic/applicability gates the contract they validate.
    if not raw_expected_changes:
        for task_index, raw_task in enumerate(tasks_raw, 1):
            if not isinstance(raw_task, Mapping):
                continue
            local_changes = raw_task.get("expected_changes", raw_task.get("changes", ()))
            if isinstance(local_changes, Mapping):
                local_changes = (local_changes,)
            if not isinstance(local_changes, (list, tuple)):
                continue
            for local_change in local_changes:
                if not isinstance(local_change, Mapping):
                    continue
                lifted = dict(local_change)
                supports = lifted.get("supports_tasks")
                if supports in (None, "", (), []):
                    lifted["supports_tasks"] = [f"T{task_index:03d}"]
                raw_expected_changes.append(lifted)
                actions.append(
                    f"/tasks/{task_index - 1}/expected_changes lifted to the "
                    "plan-level mutation contract"
                )
    for change in raw_expected_changes:
        if isinstance(change, Mapping):
            path, intent = _text(change.get("path")), _text(change.get("intent"))
            raw_basis = _text(change.get("basis"))
            basis_aliases = {
                "existing inspected path": "existing_inspected_path",
                "existing_inspected_path": "existing_inspected_path",
                "repository convention": "repository_convention",
                "repository_convention": "repository_convention",
                "model selected new layout": "model_selected_new_layout",
                "model-selected new layout": "model_selected_new_layout",
                "model_selected_new_layout": "model_selected_new_layout",
                "new layout": "model_selected_new_layout",
                "new file layout": "model_selected_new_layout",
                "generated": "model_selected_new_layout",
                "generated path": "model_selected_new_layout",
                "new file": "model_selected_new_layout",
                "explicit user requirement": "explicit_user_requirement",
                "explicit user": "explicit_user_requirement",
                "explicit_user_requirement": "explicit_user_requirement",
                "user_request": "explicit_user_requirement",
                "user request": "explicit_user_requirement",
                "user:request": "explicit_user_requirement",
            }
            normalized_basis_text = raw_basis.casefold().replace("-", " ")
            basis = basis_aliases.get(normalized_basis_text, raw_basis)
            if "repository convention" in normalized_basis_text:
                basis = "repository_convention"
            elif (
                "explicit user requirement" in normalized_basis_text
                or "user requirement" in normalized_basis_text
                or "user objective" in normalized_basis_text
                or "project requirement" in normalized_basis_text
                or (
                    "objective" in normalized_basis_text
                    and "require" in normalized_basis_text
                )
            ):
                basis = "explicit_user_requirement"
            if raw_basis and basis != raw_basis:
                actions.append(
                    f"/expected_changes/{len(expected_changes)}/basis "
                    f"normalized to {basis}"
                )
            evidence_refs = _unique_text(change.get("evidence_refs"))
        else:
            path, intent, basis, evidence_refs = "", _text(change), "", []
        if path in {".", "./"}:
            actions.append(
                f"/expected_changes/{len(expected_changes)} broad execution "
                "context removed from the file mutation contract"
            )
            continue
        if path:
            supports_tasks = normalized_supports(
                change.get("supports_tasks")
            )
            if not intent and len(supports_tasks) == 1:
                task_number = int(supports_tasks[0][1:]) - 1
                if 0 <= task_number < len(tasks):
                    intent = (
                        _text(tasks[task_number].get("description"))
                        or _text(tasks[task_number].get("title"))
                    )
                    if intent:
                        actions.append(
                            f"/expected_changes/{len(expected_changes)}/intent "
                            "projected from its supported task"
                        )
            expected_changes.append(
                {
                    "path": path,
                    "intent": intent,
                    "basis": basis,
                    "evidence_refs": evidence_refs,
                    "supports_tasks": supports_tasks,
                }
            )
    # ``supports_tasks`` is a redundant cross-reference, not product
    # semantics. Weak tool-calling models may omit it or attach a path to an
    # operational "resource claim" task instead of the task that actually
    # creates/uses the file. Reconcile the cross-reference from exact path
    # mentions in the authored task contracts; never invent a path or change
    # task meaning.
    for change_index, change in enumerate(expected_changes):
        path = _text(change.get("path")).replace("\\", "/").casefold()
        if not path:
            continue
        matching_tasks: list[str] = []
        for task in tasks:
            task_label = " ".join(
                (_text(task.get("title")), _text(task.get("description")))
            ).casefold()
            contract_text = "\n".join(
                (
                    _text(task.get("title")),
                    _text(task.get("description")),
                    *(_unique_text(task.get("acceptance_criteria"))),
                    *(_unique_text(task.get("verification"))),
                )
            ).replace("\\", "/").casefold()
            if path in contract_text:
                # Resource leases are harness-owned control state, not a
                # product task. If a provider creates a separate "resource
                # claim" checklist item, keep the file claim attached to the
                # authored implementation/verification task instead.
                if any(
                    marker in task_label
                    for marker in ("resource claim", "accepted claim", "claim for")
                ):
                    continue
                matching_tasks.append(str(task["id"]))
        if not matching_tasks and len(tasks) == 1:
            matching_tasks = [str(tasks[0]["id"])]
        if matching_tasks:
            current = list(dict.fromkeys(str(item).upper() for item in change.get("supports_tasks", ())))
            if current != list(dict.fromkeys(matching_tasks)):
                change["supports_tasks"] = list(dict.fromkeys(matching_tasks))
                actions.append(
                    f"/expected_changes/{change_index}/supports_tasks reconciled from exact task path mentions"
                )

    # One repository fact commonly applies to the whole proposed plan (for
    # example, an empty-workspace inspection). Its task links are redundant
    # transport metadata: if a weak model attaches the sole root fact only to
    # the scaffold task, every later task would otherwise fail applicability
    # despite sharing the same inspected workspace. Broaden only the links;
    # never alter or invent the fact itself.
    if len(applicability) == 1 and all_ids:
        current_supports = list(applicability[0].get("supports_tasks") or ())
        if current_supports != list(all_ids):
            applicability[0]["supports_tasks"] = list(all_ids)
            actions.append(
                "/applicability_evidence/0/supports_tasks bound to all tasks "
                "from the sole plan fact"
            )
    if not expected_changes:
        path_pattern = re.compile(
            r"(?<![\w./-])([A-Za-z0-9_.-]+\."
            r"(?:html?|py|js|ts|tsx|jsx|css|json|md|txt|ya?ml|toml))\b",
            re.IGNORECASE,
        )
        for task in tasks:
            for intent in task.get("expected_changes", ()):
                match = path_pattern.search(str(intent))
                if not match:
                    continue
                expected_changes.append(
                    {
                        "path": match.group(1).replace("\\", "/"),
                        "intent": str(intent).strip(),
                        "basis": "model_selected_new_layout",
                        "evidence_refs": [],
                        "supports_tasks": [task["id"]],
                    }
                )
    for task in tasks:
        # Task-local prose is not a path claim.  Keep it out of the mutation
        # contract unless the model supplied an exact top-level path.
        task.pop("expected_changes", None)

    summary = _text(raw.get("summary", raw.get("objective")))
    if not summary:
        # Small local models often provide the complete execution strategy but
        # omit the redundant top-level summary. Reuse that authored strategy
        # as the plan heading instead of spending the bounded repair budget on
        # a cosmetic field that carries no new product meaning.
        summary = _text(raw.get("execution_strategy", raw.get("strategy")))
        if summary:
            actions.append("/summary derived from execution_strategy")
    if not summary and tasks:
        first_task = tasks[0]
        summary = _text(first_task.get("description", first_task.get("title")))
        if summary:
            actions.append("/summary derived from the first task objective")

    execution_strategy = _text(raw.get("execution_strategy", raw.get("strategy")))
    if not execution_strategy and tasks:
        # The strategy is an executable persistence field, not optional model
        # decoration.  A local model can omit it while still supplying a
        # complete task contract; derive the smallest semantics-preserving
        # orchestration rule from those authored tasks so plan persistence
        # does not fail after an otherwise valid review.
        execution_strategy = (
            "Execute the accepted tasks in dependency order, apply each bounded change, "
            "then run every listed verification step and record authoritative evidence."
        )
        actions.append("/execution_strategy derived from the accepted task contract")

    normalized = {
        "semantic_goal": (
            dict(raw.get("semantic_goal"))
            if isinstance(raw.get("semantic_goal"), Mapping)
            else {}
        ),
        "semantic_fingerprint": _text(raw.get("semantic_fingerprint")),
        "summary": summary,
        "applicability_evidence": applicability,
        "execution_strategy": execution_strategy,
        "expected_changes": expected_changes,
        "tasks": tasks,
    }
    return normalized, tuple(dict.fromkeys(actions))


def validate_normalized_plan(value: Mapping[str, Any]) -> None:
    issues: list[PlanValidationIssue] = []
    if value.get("semantic_goal") and not isinstance(value.get("semantic_goal"), Mapping):
        issues.append(
            PlanValidationIssue(
                "/semantic_goal",
                "must be an object when supplied",
            )
        )
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
        for criterion_index, criterion in enumerate(item.get("acceptance_criteria", ())):
            if not str(criterion).strip() or len(str(criterion)) > 1_000:
                issues.append(
                    PlanValidationIssue(
                        path + f"/acceptance_criteria/{criterion_index}",
                        "must contain 1 to 1,000 characters",
                    )
                )
        if not item.get("verification"):
            issues.append(PlanValidationIssue(path + "/verification", "at least one verification requirement is required"))
        for verification_index, verification in enumerate(item.get("verification", ())):
            if not str(verification).strip() or len(str(verification)) > 1_000:
                issues.append(
                    PlanValidationIssue(
                        path + f"/verification/{verification_index}",
                        "must contain 1 to 1,000 characters; name the check instead "
                        "of embedding an implementation script",
                    )
                )
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
    for index, item in enumerate(value.get("expected_changes", ()) or ()):
        path = f"/expected_changes/{index}"
        if not isinstance(item, Mapping):
            issues.append(
                PlanValidationIssue(path, "expected change must be an object")
            )
            continue
        if not _text(item.get("path")):
            issues.append(
                PlanValidationIssue(path + "/path", "non-empty path is required")
            )
        if not _text(item.get("intent")):
            issues.append(
                PlanValidationIssue(
                    path + "/intent",
                    "non-empty intent or one supported task is required",
                )
            )
        if not item.get("supports_tasks"):
            issues.append(
                PlanValidationIssue(
                    path + "/supports_tasks",
                    "at least one supported task is required",
                )
            )
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
