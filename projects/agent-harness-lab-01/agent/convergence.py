"""Harness-owned quality convergence contracts and progress watchdog."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import hashlib
import json
from typing import Any, Mapping

from .models import DomainError, new_id, utc_now


class QualityConvergenceState(str, Enum):
    NOT_EVALUATED = "not_evaluated"
    EVALUATING = "evaluating"
    BELOW_TARGET = "below_target"
    REFINING = "refining"
    REVERIFYING = "reverifying"
    CONVERGED = "converged"
    BLOCKED = "blocked"
    USER_REVIEW_REQUIRED = "user_review_required"


class EvaluationConfidence(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True, slots=True)
class EvaluatorCapabilityProfile:
    evaluator_id: str
    deterministic: bool = False
    visual: bool = False
    runtime: bool = False
    independent: bool = True
    capabilities: tuple[str, ...] = ()
    test_runner_available: bool = False
    static_analyzer_available: bool = False
    browser_available: bool = False
    screenshot_available: bool = False
    vision_evaluator_available: bool = False
    accessibility_checker_available: bool = False
    performance_profiler_available: bool = False
    user_review_required: bool = False

    def routing_order(self) -> tuple[str, ...]:
        routes = []
        if self.deterministic or self.test_runner_available:
            routes.append("deterministic_verification")
        if self.static_analyzer_available:
            routes.append("static_analysis")
        if self.runtime:
            routes.append("runtime_integration")
        routes.append("artifact_structure")
        if self.independent:
            routes.append("independent_review")
        if self.visual or self.vision_evaluator_available:
            routes.append("vision_evaluation")
        elif self.user_review_required:
            routes.append("user_review")
        return tuple(routes)


@dataclass(frozen=True, slots=True)
class QualityDimensionV1:
    id: str
    description: str
    weight: float = 1.0
    hard_gate: bool = False
    # Individual rubrics may opt into the stricter Ultra critical threshold.
    # Keeping the primitive neutral preserves explicitly-authored lower targets.
    minimum_score: float = 0.8
    required_evidence: tuple[str, ...] = ()
    evaluation_method: str = "evidence_review"
    confidence: EvaluationConfidence = EvaluationConfidence.MEDIUM
    remediation_guidance: str = ""
    version: int = 1

    def __post_init__(self) -> None:
        if self.version != 1 or not self.id or not self.description:
            raise DomainError("quality dimension requires version 1, id, and description")
        if self.weight < 0 or not 0 <= self.minimum_score <= 1:
            raise DomainError("quality dimension weight/threshold is invalid")


@dataclass(frozen=True, slots=True)
class QualityRubricV1:
    dimensions: tuple[QualityDimensionV1, ...]
    id: str = field(default_factory=lambda: new_id("rubric"))
    version: int = 1

    def __post_init__(self) -> None:
        if self.version != 1 or not self.dimensions:
            raise DomainError("quality rubric requires at least one dimension")
        if len({item.id for item in self.dimensions}) != len(self.dimensions):
            raise DomainError("quality dimension IDs must be unique")


@dataclass(frozen=True, slots=True)
class QualityTargetV1:
    objective: str
    artifact_ids: tuple[str, ...]
    rubric: QualityRubricV1
    hard_gates: tuple[str, ...] = ()
    # Product modes set 0.95 explicitly; the reusable contract remains
    # backwards-compatible for callers that author their own target.
    minimum_overall_score: float = 0.9
    independent_evaluations: int = 1
    stable_successful_evaluations: int = 1
    plateau_window: int = 3
    plateau_delta: float = 0.02
    allowed_automatic_refinements: tuple[str, ...] = ()
    user_preferences: Mapping[str, Any] = field(default_factory=dict)
    explicit_feedback: tuple[str, ...] = ()
    verification_environment: Mapping[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: new_id("quality_target"))
    version: int = 1

    def __post_init__(self) -> None:
        if self.version != 1 or not self.objective or not self.artifact_ids:
            raise DomainError("quality target requires objective and artifacts")
        if not 0 <= self.minimum_overall_score <= 1:
            raise DomainError("overall threshold must be between 0 and 1")
        if self.independent_evaluations < 1 or self.stable_successful_evaluations < 1:
            raise DomainError("evaluation counts must be positive")
        if self.plateau_window < 2 or not 0 <= self.plateau_delta <= 1:
            raise DomainError("plateau window/delta is invalid")
        object.__setattr__(self, "user_preferences", dict(self.user_preferences))
        object.__setattr__(self, "verification_environment", dict(self.verification_environment))


@dataclass(frozen=True, slots=True)
class QualityScoreV1:
    dimension_id: str
    score: float
    passed: bool
    evidence_ids: tuple[str, ...] = ()
    findings: tuple[str, ...] = ()
    confidence: EvaluationConfidence = EvaluationConfidence.MEDIUM
    version: int = 1

    def __post_init__(self) -> None:
        if self.version != 1 or not 0 <= self.score <= 1:
            raise DomainError("quality score must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class QualityEvaluationV1:
    target_id: str
    artifact_hashes: Mapping[str, str]
    scores: tuple[QualityScoreV1, ...]
    hard_gate_results: Mapping[str, bool]
    evaluator: EvaluatorCapabilityProfile
    overall_score: float
    id: str = field(default_factory=lambda: new_id("evaluation"))
    version: int = 1
    mutation_sequence: int = 0
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if self.version != 1 or not self.target_id or not 0 <= self.overall_score <= 1:
            raise DomainError("invalid quality evaluation")
        object.__setattr__(self, "artifact_hashes", dict(self.artifact_hashes))
        object.__setattr__(self, "hard_gate_results", dict(self.hard_gate_results))


@dataclass(frozen=True, slots=True)
class RefinementActionV1:
    finding: str
    dimension_id: str
    objective: str
    affected_components: tuple[str, ...]
    acceptance_criteria: tuple[str, ...]
    verification: tuple[str, ...]
    id: str = field(default_factory=lambda: new_id("refinement_action"))
    version: int = 1


@dataclass(frozen=True, slots=True)
class RefinementCycleV1:
    run_id: str
    attempt: int
    approach_fingerprint: str
    before_evaluation_id: str | None
    actions: tuple[RefinementActionV1, ...]
    changed_files: tuple[str, ...] = ()
    after_evaluation_id: str | None = None
    progress_reasons: tuple[str, ...] = ()
    error_signatures: tuple[str, ...] = ()
    id: str = field(default_factory=lambda: new_id("refinement_cycle"))
    version: int = 1
    created_at: datetime = field(default_factory=utc_now)


def evaluation_passes(target: QualityTargetV1, evaluation: QualityEvaluationV1) -> bool:
    dimensions = {item.id: item for item in target.rubric.dimensions}
    scores = {item.dimension_id: item for item in evaluation.scores}
    return (
        evaluation.target_id == target.id
        and evaluation.overall_score >= target.minimum_overall_score
        and all(evaluation.hard_gate_results.get(gate) is True for gate in target.hard_gates)
        and all(scores.get(key) is not None and scores[key].score >= value.minimum_score for key, value in dimensions.items())
    )


class ConvergenceWatchdog:
    """Detect evidence-backed progress and prevent equivalent repair loops."""

    def __init__(self, equivalent_attempt_limit: int = 3) -> None:
        self.equivalent_attempt_limit = equivalent_attempt_limit
        self._failures: dict[str, int] = {}

    @staticmethod
    def fingerprint(value: Any) -> str:
        raw = json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
        return hashlib.sha256(raw.encode()).hexdigest()

    def record(self, approach_fingerprint: str, *, progress: bool) -> str:
        if progress:
            self._failures[approach_fingerprint] = 0
            return "continue"
        count = self._failures.get(approach_fingerprint, 0) + 1
        self._failures[approach_fingerprint] = count
        return "replan" if count >= self.equivalent_attempt_limit else "retry"


class QualityConvergenceEngine:
    """Deterministic lifecycle owner; models provide findings, never completion."""

    def __init__(self, target: QualityTargetV1):
        self.target = target
        self.state = QualityConvergenceState.NOT_EVALUATED
        self.evaluations: list[QualityEvaluationV1] = []
        self.cycles: list[RefinementCycleV1] = []
        self.latest_mutation_sequence = 0
        self.watchdog = ConvergenceWatchdog()

    def mutated(self) -> None:
        self.latest_mutation_sequence += 1
        self.state = QualityConvergenceState.REVERIFYING

    def accept_evaluation(self, evaluation: QualityEvaluationV1) -> QualityConvergenceState:
        self.state = QualityConvergenceState.EVALUATING
        if evaluation.mutation_sequence != self.latest_mutation_sequence:
            raise DomainError("stale evaluation does not correspond to the latest mutation")
        if set(evaluation.artifact_hashes) != set(self.target.artifact_ids):
            raise DomainError("evaluation does not cover every target artifact")
        self.evaluations.append(evaluation)
        if not evaluation_passes(self.target, evaluation):
            recent = self.evaluations[-self.target.plateau_window :]
            plateau = (
                len(recent) >= self.target.plateau_window
                and max(item.overall_score for item in recent)
                - min(item.overall_score for item in recent)
                < self.target.plateau_delta
            )
            self.state = (
                QualityConvergenceState.BLOCKED
                if plateau
                else QualityConvergenceState.BELOW_TARGET
            )
            return self.state
        fresh = [item for item in reversed(self.evaluations) if item.mutation_sequence == self.latest_mutation_sequence]
        required = max(self.target.independent_evaluations, self.target.stable_successful_evaluations)
        recent = fresh[:required]
        independent = len({item.evaluator.evaluator_id for item in recent if item.evaluator.independent})
        if len(recent) < required or not all(evaluation_passes(self.target, item) for item in recent):
            self.state = QualityConvergenceState.EVALUATING
        elif self.target.independent_evaluations > 1 and independent < self.target.independent_evaluations:
            self.state = QualityConvergenceState.EVALUATING
        else:
            self.state = QualityConvergenceState.CONVERGED
        return self.state

    def may_release_final(self, *, unresolved_blocking_findings: int = 0, uncertain_operations: int = 0) -> bool:
        return self.state is QualityConvergenceState.CONVERGED and not unresolved_blocking_findings and not uncertain_operations
