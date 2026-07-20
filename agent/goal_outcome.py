"""Durable product-outcome contracts for autonomous Ultra convergence.

The orchestration engine may finish a plan or a wave.  It may not declare the
user-visible product accepted.  This module owns that higher-level distinction
and deliberately evaluates only durable, independently attributable evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import json
from typing import Any, Iterable, Mapping

from .models import DomainError, new_id, utc_now


class GoalOutcomeState(str, Enum):
    RUNNING = "running"
    RECOVERING = "recovering"
    QUALITY_BLOCKED = "quality_blocked"
    EXTERNAL_BLOCKED = "external_blocked"
    ACCEPTED = "accepted"


class ExperimentOutcome(str, Enum):
    CHAMPION = "champion"
    REJECTED = "rejected"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True, slots=True)
class GoalOutcomeContractV1:
    """Acceptance authority above individual Ultra runs and work nodes."""

    goal_id: str
    objective: str
    required_evidence: tuple[str, ...] = (
        "final_artifact",
        "runtime",
        "screenshots",
        "independent_visual",
        "pairwise_baseline",
        "codex_visual_review",
    )
    minimum_overall_score: float = 0.95
    minimum_critical_score: float = 0.90
    required_clean_visual_acceptances: int = 2
    require_zero_critical_findings: bool = True
    require_candidate_preferred: bool = True
    auto_converge: bool = True
    max_strategy_repetitions: int = 2
    id: str = field(default_factory=lambda: new_id("outcome_contract"))
    version: int = 1

    def __post_init__(self) -> None:
        if self.version != 1 or not self.goal_id or not self.objective.strip():
            raise DomainError("GoalOutcomeContractV1 requires version 1, goal, and objective")
        if not self.required_evidence:
            raise DomainError("outcome contract requires durable evidence kinds")
        if not 0 <= self.minimum_overall_score <= 1:
            raise DomainError("outcome overall threshold must be between zero and one")
        if not 0 <= self.minimum_critical_score <= 1:
            raise DomainError("outcome critical threshold must be between zero and one")
        if self.required_clean_visual_acceptances < 2:
            raise DomainError("Ultra requires at least two clean visual acceptances")
        if self.max_strategy_repetitions < 1:
            raise DomainError("strategy repetition limit must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "GoalOutcomeContractV1",
            "version": 1,
            "id": self.id,
            "goal_id": self.goal_id,
            "objective": self.objective,
            "required_evidence": list(self.required_evidence),
            "minimum_overall_score": self.minimum_overall_score,
            "minimum_critical_score": self.minimum_critical_score,
            "required_clean_visual_acceptances": self.required_clean_visual_acceptances,
            "require_zero_critical_findings": self.require_zero_critical_findings,
            "require_candidate_preferred": self.require_candidate_preferred,
            "auto_converge": self.auto_converge,
            "max_strategy_repetitions": self.max_strategy_repetitions,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GoalOutcomeContractV1":
        return cls(
            id=str(value.get("id") or new_id("outcome_contract")),
            goal_id=str(value.get("goal_id") or ""),
            objective=str(value.get("objective") or ""),
            required_evidence=tuple(
                str(item) for item in value.get("required_evidence", ()) if str(item)
            ),
            minimum_overall_score=float(value.get("minimum_overall_score", 0.95)),
            minimum_critical_score=float(value.get("minimum_critical_score", 0.90)),
            required_clean_visual_acceptances=int(
                value.get("required_clean_visual_acceptances", 2)
            ),
            require_zero_critical_findings=bool(
                value.get("require_zero_critical_findings", True)
            ),
            require_candidate_preferred=bool(
                value.get("require_candidate_preferred", True)
            ),
            auto_converge=bool(value.get("auto_converge", True)),
            max_strategy_repetitions=int(value.get("max_strategy_repetitions", 2)),
            version=int(value.get("version", 1)),
        )


@dataclass(frozen=True, slots=True)
class FinalAcceptanceEvidenceV1:
    ultra_run_id: str
    kind: str
    authority: str
    passed: bool
    score: float = 0.0
    critical_findings: int = 0
    artifact_hash: str = ""
    details: Mapping[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: new_id("acceptance_evidence"))
    created_at: datetime = field(default_factory=utc_now)
    version: int = 1

    def __post_init__(self) -> None:
        if self.version != 1 or not self.ultra_run_id or not self.kind or not self.authority:
            raise DomainError("acceptance evidence requires run, kind, and authority")
        if not 0 <= self.score <= 1 or self.critical_findings < 0:
            raise DomainError("acceptance evidence score/findings are invalid")
        object.__setattr__(self, "details", dict(self.details))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "FinalAcceptanceEvidenceV1",
            "version": 1,
            "id": self.id,
            "ultra_run_id": self.ultra_run_id,
            "kind": self.kind,
            "authority": self.authority,
            "passed": self.passed,
            "score": self.score,
            "critical_findings": self.critical_findings,
            "artifact_hash": self.artifact_hash,
            "details": dict(self.details),
            "created_at": self.created_at.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class OptimizationExperimentV1:
    ultra_run_id: str
    node_id: str
    variable: str
    baseline: Mapping[str, Any]
    candidate: Mapping[str, Any]
    hypothesis: str
    before_score: float
    after_score: float
    outcome: ExperimentOutcome
    evidence: tuple[str, ...] = ()
    id: str = field(default_factory=lambda: new_id("optimization"))
    created_at: datetime = field(default_factory=utc_now)
    version: int = 1

    def __post_init__(self) -> None:
        if self.version != 1 or not self.ultra_run_id or not self.node_id:
            raise DomainError("optimization experiment requires run and node")
        if not self.variable or not self.hypothesis:
            raise DomainError("optimization experiment requires variable and hypothesis")
        if not 0 <= self.before_score <= 1 or not 0 <= self.after_score <= 1:
            raise DomainError("optimization scores must be between zero and one")
        object.__setattr__(self, "baseline", dict(self.baseline))
        object.__setattr__(self, "candidate", dict(self.candidate))

    @property
    def delta(self) -> float:
        return self.after_score - self.before_score

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "OptimizationExperimentV1",
            "version": 1,
            "id": self.id,
            "ultra_run_id": self.ultra_run_id,
            "node_id": self.node_id,
            "variable": self.variable,
            "baseline": dict(self.baseline),
            "candidate": dict(self.candidate),
            "hypothesis": self.hypothesis,
            "before_score": self.before_score,
            "after_score": self.after_score,
            "delta": self.delta,
            "outcome": self.outcome.value,
            "evidence": list(self.evidence),
            "created_at": self.created_at.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class FinalAcceptanceDecisionV1:
    contract_id: str
    accepted: bool
    missing: tuple[str, ...]
    blockers: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    overall_score: float
    critical_minimum: float
    id: str = field(default_factory=lambda: new_id("acceptance_decision"))
    created_at: datetime = field(default_factory=utc_now)
    version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "FinalAcceptanceDecisionV1",
            "version": 1,
            "id": self.id,
            "contract_id": self.contract_id,
            "accepted": self.accepted,
            "missing": list(self.missing),
            "blockers": list(self.blockers),
            "evidence_ids": list(self.evidence_ids),
            "overall_score": self.overall_score,
            "critical_minimum": self.critical_minimum,
            "created_at": self.created_at.isoformat(),
        }


class FinalAcceptanceGate:
    """Fail-closed product acceptance derived from durable evidence."""

    def __init__(self, contract: GoalOutcomeContractV1):
        self.contract = contract

    def evaluate(
        self,
        evidence: Iterable[FinalAcceptanceEvidenceV1 | Mapping[str, Any]],
    ) -> FinalAcceptanceDecisionV1:
        values = tuple(
            item
            if isinstance(item, FinalAcceptanceEvidenceV1)
            else FinalAcceptanceEvidenceV1(
                id=str(item.get("id") or new_id("acceptance_evidence")),
                ultra_run_id=str(item.get("ultra_run_id") or ""),
                kind=str(item.get("kind") or ""),
                authority=str(item.get("authority") or ""),
                passed=bool(item.get("passed")),
                score=float(item.get("score", 0.0)),
                critical_findings=int(item.get("critical_findings", 0)),
                artifact_hash=str(item.get("artifact_hash") or ""),
                details=dict(item.get("details") or {}),
                created_at=(
                    datetime.fromisoformat(str(item["created_at"]))
                    if item.get("created_at")
                    else utc_now()
                ),
            )
            for item in evidence
        )
        latest: dict[tuple[str, str], FinalAcceptanceEvidenceV1] = {}
        for item in values:
            latest[(item.kind, item.authority)] = item
        effective = tuple(latest.values())
        by_kind: dict[str, list[FinalAcceptanceEvidenceV1]] = {}
        for item in effective:
            by_kind.setdefault(item.kind, []).append(item)
        missing = tuple(
            kind
            for kind in self.contract.required_evidence
            if not any(item.passed for item in by_kind.get(kind, ()))
        )
        blockers: list[str] = []
        passed_values = [item for item in effective if item.passed]
        scored = [item.score for item in passed_values if item.score > 0]
        overall = min(scored) if scored else 0.0
        critical_minimum = min(scored) if scored else 0.0
        if missing:
            blockers.append("missing required evidence: " + ", ".join(missing))
        if self.contract.require_zero_critical_findings and any(
            item.critical_findings for item in effective
        ):
            blockers.append("one or more critical findings remain")
        visual_required = "independent_visual" in self.contract.required_evidence
        visual = [
            item
            for item in by_kind.get("independent_visual", ())
            if item.passed and item.authority not in {"builder", "self"}
        ]
        if (
            visual_required
            and len({item.authority for item in visual})
            < self.contract.required_clean_visual_acceptances
        ):
            blockers.append("fewer than two independent clean visual authorities accepted")
        if scored and overall < self.contract.minimum_overall_score:
            blockers.append(
                f"minimum accepted evidence score {overall:.3f} is below "
                f"{self.contract.minimum_overall_score:.3f}"
            )
        critical_scores = [
            item.score
            for item in effective
            if item.passed and bool(item.details.get("critical", False))
        ]
        if critical_scores:
            critical_minimum = min(critical_scores)
            if critical_minimum < self.contract.minimum_critical_score:
                blockers.append(
                    f"critical evidence score {critical_minimum:.3f} is below "
                    f"{self.contract.minimum_critical_score:.3f}"
                )
        if (
            self.contract.require_candidate_preferred
            and "pairwise_baseline" in self.contract.required_evidence
        ):
            pairwise = by_kind.get("pairwise_baseline", ())
            if not any(
                item.passed and bool(item.details.get("candidate_preferred", False))
                for item in pairwise
            ):
                blockers.append("Ultra candidate was not proven preferable to baseline")
        return FinalAcceptanceDecisionV1(
            contract_id=self.contract.id,
            accepted=not missing and not blockers,
            missing=missing,
            blockers=tuple(dict.fromkeys(blockers)),
            evidence_ids=tuple(item.id for item in effective),
            overall_score=overall,
            critical_minimum=critical_minimum,
        )


def experiment_fingerprint(value: Mapping[str, Any]) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def heartbeat_is_stale(
    heartbeat_at: datetime | None,
    *,
    now: datetime | None = None,
    stale_after_seconds: int = 600,
) -> bool:
    if heartbeat_at is None:
        return True
    return (now or utc_now()) - heartbeat_at > timedelta(seconds=stale_after_seconds)


__all__ = [
    "ExperimentOutcome",
    "FinalAcceptanceDecisionV1",
    "FinalAcceptanceEvidenceV1",
    "FinalAcceptanceGate",
    "GoalOutcomeContractV1",
    "GoalOutcomeState",
    "OptimizationExperimentV1",
    "experiment_fingerprint",
    "heartbeat_is_stale",
]
