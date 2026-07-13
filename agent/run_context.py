"""Shared durable run memory used across Chat, Plan, Goal, and Ultra."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
import hashlib
import json
from typing import Any, Mapping

from .convergence import QualityConvergenceState, QualityTargetV1
from .models import new_id, utc_now
from .workflow import SessionMode
from .weak_model import WeakModelPolicy


@dataclass(frozen=True, slots=True)
class GoalContractV1:
    run_id: str
    original_objective: str
    interpreted_objective: str
    required_outcomes: tuple[str, ...] = ()
    out_of_scope: tuple[str, ...] = ()
    acceptance_criteria: tuple[str, ...] = ()
    forbidden_shortcuts: tuple[str, ...] = ()
    artifact_expectations: tuple[str, ...] = ()
    required_verification: tuple[str, ...] = ()
    quality_target_id: str | None = None
    current_task: str | None = None
    task_boundaries: tuple[str, ...] = ()
    file_symbol_scope: tuple[str, ...] = ()
    user_feedback: tuple[str, ...] = ()
    failed_hypotheses: tuple[str, ...] = ()
    blockers: tuple[str, ...] = ()
    completion_conditions: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GoalContractV1":
        return cls(**{key: value[key] for key in cls.__dataclass_fields__ if key in value})

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def projection(self, *, actor: str, task_id: str | None = None) -> dict[str, Any]:
        """Compact harness-built contract sent before every participating call."""
        return {
            "contract_version": 1,
            "contract_fingerprint": self.fingerprint,
            "run_id": self.run_id,
            "role": actor,
            "objective": self.interpreted_objective,
            "required_outcomes": list(self.required_outcomes),
            "acceptance_criteria": list(self.acceptance_criteria),
            "forbidden_shortcuts": list(self.forbidden_shortcuts),
            "required_verification": list(self.required_verification),
            "current_task": task_id or self.current_task,
            "task_boundaries": list(self.task_boundaries),
            "file_symbol_scope": list(self.file_symbol_scope),
            "relevant_feedback": list(self.user_feedback[-5:]),
            "failed_hypotheses": list(self.failed_hypotheses[-3:]),
            "active_blockers": list(self.blockers),
            "completion_conditions": list(self.completion_conditions),
        }


@dataclass(frozen=True, slots=True)
class ModeTransitionV1:
    previous: SessionMode
    current: SessionMode
    reason: str
    at: datetime = field(default_factory=utc_now)


@dataclass(frozen=True, slots=True)
class RunContextV1:
    workspace_id: str
    workspace_fingerprint: str
    original_objective: str
    current_objective: str
    mode: SessionMode = SessionMode.CHAT
    run_id: str = field(default_factory=lambda: new_id("run"))
    version: int = 1
    weak_model_policy: WeakModelPolicy = field(default_factory=WeakModelPolicy)
    goal_contract: GoalContractV1 | None = None
    user_messages: tuple[str, ...] = ()
    accepted_guidance: tuple[str, ...] = ()
    mode_history: tuple[ModeTransitionV1, ...] = ()
    plan: Mapping[str, Any] = field(default_factory=dict)
    approval_state: str = "none"
    task_dag: Mapping[str, Any] = field(default_factory=dict)
    active_node: str | None = None
    artifact_ids: tuple[str, ...] = ()
    index_snapshot: Mapping[str, Any] = field(default_factory=dict)
    agent_ids: tuple[str, ...] = ()
    change_set_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    quality_target: QualityTargetV1 | None = None
    evaluation_ids: tuple[str, ...] = ()
    refinement_cycle_ids: tuple[str, ...] = ()
    finding_ids: tuple[str, ...] = ()
    capability_profile: Mapping[str, Any] = field(default_factory=dict)
    blockers: tuple[str, ...] = ()
    resume_checkpoint: Mapping[str, Any] = field(default_factory=dict)
    convergence_state: QualityConvergenceState = QualityConvergenceState.NOT_EVALUATED
    mutation_sequence: int = 0
    updated_at: datetime = field(default_factory=utc_now)

    def transition(self, target: SessionMode, reason: str = "user requested mode change") -> "RunContextV1":
        if target is self.mode:
            return self
        return replace(self, mode=target, mode_history=(*self.mode_history, ModeTransitionV1(self.mode, target, reason)), updated_at=utc_now())


def is_goal_escalation_approval(text: str) -> bool:
    return str(text).strip().casefold().rstrip(".! ") in {
        "", "yes", "y", "continue", "improve it", "use goal", "go ahead", "proceed"
    }
