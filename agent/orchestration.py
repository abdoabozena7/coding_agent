"""Evidence-driven worker contracts and deterministic adaptive routing.

The selected model may author code, findings, and candidate solutions.  This
module deliberately keeps the decisions that control worker count, evidence
freshness, contribution credit, and early stopping inside the harness.  It is
provider-neutral and shared by Normal and Ultra workflows.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
import hashlib
import json
from typing import Any, Iterable, Mapping, Sequence

from .models import DomainError, new_id, utc_now


class WorkerRole(str, Enum):
    PREDICTOR = "predictor"
    BUILDER = "builder"
    FALSIFIER = "falsifier"
    CHALLENGER = "challenger"
    SELECTOR = "selector"
    REPAIRER = "repairer"
    REVIEWER = "reviewer"


class WorkerVisibility(str, Enum):
    CONTRACT_ONLY = "contract_only"
    ARTIFACT_WITHOUT_RATIONALE = "artifact_without_rationale"
    ANONYMOUS_CANDIDATES = "anonymous_candidates"
    VERIFIED_FINDINGS_ONLY = "verified_findings_only"
    FULL_SCOPED_CONTEXT = "full_scoped_context"


class MutationPolicy(str, Enum):
    READ_ONLY = "read_only"
    STAGING_ONLY = "staging_only"
    SINGLE_WRITER = "single_writer"


class EvidenceVerdict(str, Enum):
    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"
    NEEDS_EVIDENCE = "needs_evidence"


class EvidenceAuthority(str, Enum):
    HARNESS = "harness"
    TEST = "test"
    BROWSER = "browser"
    USER = "user"
    MODEL = "model"


class WorkerImpactOutcome(str, Enum):
    USEFUL = "useful"
    NEUTRAL = "neutral"
    HARMFUL = "harmful"


class OrchestrationArm(str, Enum):
    SINGLE_AGENT = "single_agent"
    COMPUTE_MATCHED = "compute_matched"
    CURRENT = "current_orchestration"
    ADAPTIVE = "adaptive_evidence"
    PRODUCTION = "production"


class RiskTier(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    AMBIGUOUS = "ambiguous"


class SpecialistReviewRole(str, Enum):
    """The fixed post-build review surface shared by Normal and Ultra."""

    SECURITY = "security"
    CLEAN_CODE = "clean_code"
    TESTING = "testing"
    ARCHITECTURE = "architecture"
    FIDELITY = "fidelity"
    REGRESSION = "regression"


FIXED_SPECIALIST_REVIEW_ORDER: tuple[SpecialistReviewRole, ...] = (
    SpecialistReviewRole.SECURITY,
    SpecialistReviewRole.CLEAN_CODE,
    SpecialistReviewRole.TESTING,
    SpecialistReviewRole.ARCHITECTURE,
    SpecialistReviewRole.FIDELITY,
    SpecialistReviewRole.REGRESSION,
)


_READ_ONLY_ROLES = {
    WorkerRole.PREDICTOR,
    WorkerRole.FALSIFIER,
    WorkerRole.REVIEWER,
    WorkerRole.SELECTOR,
}


def _bounded_text(value: Any, maximum: int, *, field_name: str) -> str:
    text = " ".join(str(value or "").split())
    if not text:
        raise DomainError(f"{field_name} must not be empty")
    if len(text) > maximum:
        raise DomainError(f"{field_name} exceeds {maximum} characters")
    return text


def _bounded_ratio(value: float, *, field_name: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise DomainError(f"{field_name} must be between zero and one")
    return parsed


def approach_fingerprint(
    *,
    role: WorkerRole | str,
    objective: str,
    falsification_targets: Iterable[str] = (),
    context_refs: Iterable[str] = (),
) -> str:
    """Return the identity used to reject correlated duplicate approaches."""

    normalized = {
        "role": WorkerRole(role).value,
        "objective": " ".join(str(objective).casefold().split()),
        "falsification_targets": sorted(
            {" ".join(str(item).casefold().split()) for item in falsification_targets if str(item).strip()}
        ),
        "context_refs": sorted({str(item).strip() for item in context_refs if str(item).strip()}),
    }
    return hashlib.sha256(
        json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class AdaptiveOrchestrationPolicyV1:
    """Versioned limits for weak-model worker routing.

    ``max_model_call_multiplier`` includes the builder call.  The default
    therefore permits one builder and at most three focused worker calls.
    """

    max_model_call_multiplier: int = 4
    medium_extra_calls: int = 1
    high_extra_calls: int = 3
    minimum_learning_samples: int = 20
    suppress_below_useful_rate: float = 0.15
    reprobe_fraction: float = 0.10
    early_stop_on_no_novel_evidence: bool = True
    shadow_mode: bool = True
    version: int = 1

    def __post_init__(self) -> None:
        if self.version != 1:
            raise DomainError("unsupported adaptive orchestration policy version")
        if not 1 <= self.max_model_call_multiplier <= 4:
            raise DomainError("worker policy call multiplier must be between one and four")
        if not 0 <= self.medium_extra_calls < self.max_model_call_multiplier:
            raise DomainError("invalid medium-risk worker budget")
        if not 0 <= self.high_extra_calls < self.max_model_call_multiplier:
            raise DomainError("invalid high-risk worker budget")
        if self.minimum_learning_samples < 1:
            raise DomainError("minimum learning samples must be positive")
        _bounded_ratio(self.suppress_below_useful_rate, field_name="suppress_below_useful_rate")
        _bounded_ratio(self.reprobe_fraction, field_name="reprobe_fraction")

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(
            json.dumps(asdict(self), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {"schema": "AdaptiveOrchestrationPolicyV1", **asdict(self), "fingerprint": self.fingerprint}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | None) -> "AdaptiveOrchestrationPolicyV1":
        raw = dict(value or {})
        known = cls.__dataclass_fields__
        return cls(**{key: raw[key] for key in known if key in raw})


@dataclass(frozen=True, slots=True)
class TaskRiskSignalsV1:
    """Harness-observed signals; no free-form model confidence is accepted."""

    declared_risk: str = "medium"
    changed_file_count: int = 0
    touches_interfaces: bool = False
    security_sensitive: bool = False
    tests_changed: bool = False
    prior_failed_approaches: int = 0
    missing_evidence_count: int = 0
    ambiguous_design: bool = False
    deterministic_gate_available: bool = True
    subjective_acceptance: bool = False
    version: int = 1

    def __post_init__(self) -> None:
        if self.version != 1:
            raise DomainError("unsupported task risk signal version")
        if self.declared_risk not in {"low", "medium", "high", "critical"}:
            raise DomainError("declared risk must be low, medium, high, or critical")
        if min(self.changed_file_count, self.prior_failed_approaches, self.missing_evidence_count) < 0:
            raise DomainError("task risk counters cannot be negative")

    @property
    def tier(self) -> RiskTier:
        if self.ambiguous_design:
            return RiskTier.AMBIGUOUS
        if (
            self.declared_risk in {"high", "critical"}
            or self.security_sensitive
            or self.touches_interfaces
            or self.changed_file_count >= 4
            or self.prior_failed_approaches >= 1
            or self.missing_evidence_count >= 2
            or self.subjective_acceptance
        ):
            return RiskTier.HIGH
        if (
            self.declared_risk == "medium"
            or self.changed_file_count >= 2
            or self.tests_changed
            or self.missing_evidence_count == 1
            or not self.deterministic_gate_available
        ):
            return RiskTier.MEDIUM
        return RiskTier.LOW


@dataclass(frozen=True, slots=True)
class WorkerMissionV2:
    role: WorkerRole
    objective: str
    success_criteria: tuple[str, ...]
    visibility: WorkerVisibility
    mutation_policy: MutationPolicy
    allowed_tools: tuple[str, ...] = ()
    falsification_targets: tuple[str, ...] = ()
    context_refs: tuple[str, ...] = ()
    max_model_calls: int = 1
    seed: int | None = None
    id: str = field(default_factory=lambda: new_id("mission"))
    version: int = 2

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", WorkerRole(self.role))
        object.__setattr__(self, "visibility", WorkerVisibility(self.visibility))
        object.__setattr__(self, "mutation_policy", MutationPolicy(self.mutation_policy))
        object.__setattr__(self, "objective", _bounded_text(self.objective, 4_000, field_name="mission objective"))
        object.__setattr__(self, "success_criteria", tuple(dict.fromkeys(str(item).strip() for item in self.success_criteria if str(item).strip())))
        object.__setattr__(self, "allowed_tools", tuple(dict.fromkeys(str(item).strip() for item in self.allowed_tools if str(item).strip())))
        object.__setattr__(self, "falsification_targets", tuple(dict.fromkeys(str(item).strip() for item in self.falsification_targets if str(item).strip()))[:3])
        object.__setattr__(self, "context_refs", tuple(dict.fromkeys(str(item).strip() for item in self.context_refs if str(item).strip())))
        if self.version != 2 or not self.id:
            raise DomainError("worker mission requires v2 and an id")
        if not self.success_criteria:
            raise DomainError("worker mission requires success criteria")
        if not 1 <= self.max_model_calls <= 4:
            raise DomainError("worker mission model-call budget must be between one and four")
        if self.role in _READ_ONLY_ROLES and self.mutation_policy is not MutationPolicy.READ_ONLY:
            raise DomainError(f"{self.role.value} must be read-only")
        if self.role is WorkerRole.CHALLENGER and self.mutation_policy is not MutationPolicy.STAGING_ONLY:
            raise DomainError("challenger output must remain in staging")
        if self.role is WorkerRole.REPAIRER and self.visibility is not WorkerVisibility.VERIFIED_FINDINGS_ONLY:
            raise DomainError("repairer may receive only verified findings")

    @property
    def approach_fingerprint(self) -> str:
        return approach_fingerprint(
            role=self.role,
            objective=self.objective,
            falsification_targets=self.falsification_targets,
            context_refs=self.context_refs,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "WorkerMissionV2",
            "version": self.version,
            "id": self.id,
            "role": self.role.value,
            "objective": self.objective,
            "success_criteria": list(self.success_criteria),
            "visibility": self.visibility.value,
            "mutation_policy": self.mutation_policy.value,
            "allowed_tools": list(self.allowed_tools),
            "falsification_targets": list(self.falsification_targets),
            "context_refs": list(self.context_refs),
            "max_model_calls": self.max_model_calls,
            "seed": self.seed,
            "approach_fingerprint": self.approach_fingerprint,
        }


@dataclass(frozen=True, slots=True)
class EvidenceClaimV1:
    criterion_id: str
    claim: str
    artifact_hash: str
    evidence_refs: tuple[str, ...]
    falsification_check: str
    verdict: EvidenceVerdict = EvidenceVerdict.PENDING
    authority: EvidenceAuthority = EvidenceAuthority.MODEL
    producer_id: str = ""
    verifier_id: str = ""
    id: str = field(default_factory=lambda: new_id("claim"))
    created_at: datetime = field(default_factory=utc_now)
    version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "criterion_id", _bounded_text(self.criterion_id, 200, field_name="criterion id"))
        object.__setattr__(self, "claim", _bounded_text(self.claim, 2_000, field_name="evidence claim"))
        object.__setattr__(self, "falsification_check", _bounded_text(self.falsification_check, 2_000, field_name="falsification check"))
        object.__setattr__(self, "verdict", EvidenceVerdict(self.verdict))
        object.__setattr__(self, "authority", EvidenceAuthority(self.authority))
        object.__setattr__(self, "evidence_refs", tuple(dict.fromkeys(str(item).strip() for item in self.evidence_refs if str(item).strip())))
        if self.version != 1 or not self.id:
            raise DomainError("evidence claim requires v1 and an id")
        if self.verdict in {EvidenceVerdict.PASSED, EvidenceVerdict.FAILED}:
            if not self.artifact_hash or not self.evidence_refs or not self.verifier_id:
                raise DomainError("decisive evidence claims require artifact hash, evidence, and verifier")
        if self.authority is EvidenceAuthority.MODEL and self.verdict is EvidenceVerdict.PASSED:
            if not self.producer_id or self.producer_id == self.verifier_id:
                raise DomainError("a model-authored pass must be independently verified")

    @property
    def authoritative(self) -> bool:
        if self.verdict not in {EvidenceVerdict.PASSED, EvidenceVerdict.FAILED}:
            return False
        if self.authority in {
            EvidenceAuthority.HARNESS,
            EvidenceAuthority.TEST,
            EvidenceAuthority.BROWSER,
            EvidenceAuthority.USER,
        }:
            return bool(self.artifact_hash and self.evidence_refs and self.verifier_id)
        return bool(self.producer_id and self.verifier_id and self.producer_id != self.verifier_id)

    @property
    def hard_veto(self) -> bool:
        return self.authoritative and self.verdict is EvidenceVerdict.FAILED and self.authority is not EvidenceAuthority.MODEL

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "EvidenceClaimV1",
            "version": self.version,
            "id": self.id,
            "criterion_id": self.criterion_id,
            "claim": self.claim,
            "artifact_hash": self.artifact_hash,
            "evidence_refs": list(self.evidence_refs),
            "falsification_check": self.falsification_check,
            "verdict": self.verdict.value,
            "authority": self.authority.value,
            "producer_id": self.producer_id,
            "verifier_id": self.verifier_id,
            "authoritative": self.authoritative,
            "hard_veto": self.hard_veto,
            "created_at": self.created_at.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class WorkerImpactV1:
    experiment_id: str
    worker_id: str
    role: WorkerRole
    task_class: str
    model_fingerprint: str
    outcome: WorkerImpactOutcome
    approach_fingerprint: str
    evidence_novelty: float = 0.0
    verified_findings: int = 0
    false_findings: int = 0
    accepted_fixes: int = 0
    score_delta: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    claims: tuple[EvidenceClaimV1, ...] = ()
    evidence: tuple[str, ...] = ()
    reason: str = ""
    delegation_id: str | None = None
    agent_run_id: str | None = None
    id: str = field(default_factory=lambda: new_id("worker_impact"))
    created_at: datetime = field(default_factory=utc_now)
    version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", WorkerRole(self.role))
        object.__setattr__(self, "outcome", WorkerImpactOutcome(self.outcome))
        object.__setattr__(self, "claims", tuple(self.claims))
        object.__setattr__(self, "evidence", tuple(dict.fromkeys(str(item).strip() for item in self.evidence if str(item).strip())))
        if self.version != 1 or not self.id or not self.experiment_id or not self.worker_id:
            raise DomainError("worker impact requires ids and version 1")
        _bounded_ratio(self.evidence_novelty, field_name="evidence novelty")
        if min(
            self.verified_findings,
            self.false_findings,
            self.accepted_fixes,
            self.input_tokens,
            self.output_tokens,
            self.latency_ms,
        ) < 0:
            raise DomainError("worker impact counters cannot be negative")
        if not -1.0 <= self.score_delta <= 1.0:
            raise DomainError("worker score delta must be between -1 and 1")
        if self.outcome is WorkerImpactOutcome.USEFUL and not (
            self.verified_findings or self.accepted_fixes or self.evidence_novelty > 0.0 or self.score_delta > 0.0
        ):
            raise DomainError("useful workers require a verified novel contribution")

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "WorkerImpactV1",
            "version": self.version,
            "id": self.id,
            "experiment_id": self.experiment_id,
            "worker_id": self.worker_id,
            "delegation_id": self.delegation_id,
            "agent_run_id": self.agent_run_id,
            "role": self.role.value,
            "task_class": self.task_class,
            "model_fingerprint": self.model_fingerprint,
            "outcome": self.outcome.value,
            "approach_fingerprint": self.approach_fingerprint,
            "evidence_novelty": self.evidence_novelty,
            "verified_findings": self.verified_findings,
            "false_findings": self.false_findings,
            "accepted_fixes": self.accepted_fixes,
            "score_delta": self.score_delta,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "latency_ms": self.latency_ms,
            "claims": [item.to_dict() for item in self.claims],
            "evidence": list(self.evidence),
            "reason": self.reason,
            "created_at": self.created_at.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class OrchestrationExperimentV1:
    goal_id: str
    unit_id: str
    arm: OrchestrationArm
    model_fingerprint: str
    policy_fingerprint: str
    task_class: str
    baseline_score: float = 0.0
    candidate_score: float = 0.0
    success: bool = False
    false_completion: bool = False
    model_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    gpu_ms: int = 0
    causal: bool = False
    matched_benchmark: bool = False
    metrics: Mapping[str, Any] = field(default_factory=dict)
    evidence: tuple[str, ...] = ()
    ultra_run_id: str | None = None
    id: str = field(default_factory=lambda: new_id("orchestration"))
    created_at: datetime = field(default_factory=utc_now)
    version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "arm", OrchestrationArm(self.arm))
        object.__setattr__(self, "metrics", dict(self.metrics))
        object.__setattr__(self, "evidence", tuple(dict.fromkeys(str(item).strip() for item in self.evidence if str(item).strip())))
        if self.version != 1 or not self.goal_id or not self.unit_id or not self.model_fingerprint:
            raise DomainError("orchestration experiment requires goal, unit, and model fingerprint")
        _bounded_ratio(self.baseline_score, field_name="baseline score")
        _bounded_ratio(self.candidate_score, field_name="candidate score")
        if min(self.model_calls, self.input_tokens, self.output_tokens, self.latency_ms, self.gpu_ms) < 0:
            raise DomainError("orchestration experiment counters cannot be negative")
        if self.causal and not self.matched_benchmark:
            raise DomainError("causal uplift is valid only for a matched benchmark")

    @property
    def score_delta(self) -> float:
        return self.candidate_score - self.baseline_score

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "OrchestrationExperimentV1",
            "version": self.version,
            "id": self.id,
            "goal_id": self.goal_id,
            "ultra_run_id": self.ultra_run_id,
            "unit_id": self.unit_id,
            "arm": self.arm.value,
            "model_fingerprint": self.model_fingerprint,
            "policy_fingerprint": self.policy_fingerprint,
            "task_class": self.task_class,
            "baseline_score": self.baseline_score,
            "candidate_score": self.candidate_score,
            "score_delta": self.score_delta,
            "success": self.success,
            "false_completion": self.false_completion,
            "model_calls": self.model_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_ms": self.latency_ms,
            "gpu_ms": self.gpu_ms,
            "causal": self.causal,
            "matched_benchmark": self.matched_benchmark,
            "metrics": dict(self.metrics),
            "evidence": list(self.evidence),
            "created_at": self.created_at.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class WorkerRouteV1:
    tier: RiskTier
    roles: tuple[WorkerRole, ...]
    max_model_calls: int
    reason: str
    early_stop_required: bool = True
    version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "WorkerRouteV1",
            "version": self.version,
            "tier": self.tier.value,
            "roles": [item.value for item in self.roles],
            "max_model_calls": self.max_model_calls,
            "reason": self.reason,
            "early_stop_required": self.early_stop_required,
        }


@dataclass(frozen=True, slots=True)
class SpecialistReviewResultV1:
    """One read-only specialist verdict bound to an exact artifact revision."""

    role: SpecialistReviewRole
    artifact_hash: str
    mutation_sequence: int
    passed: bool | None
    summary: str
    issues: tuple[Mapping[str, Any], ...] = ()
    evidence_refs: tuple[str, ...] = ()
    reviewer_id: str = ""
    version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", SpecialistReviewRole(self.role))
        object.__setattr__(self, "summary", _bounded_text(self.summary, 4_000, field_name="specialist review summary"))
        object.__setattr__(self, "issues", tuple(dict(item) for item in self.issues if isinstance(item, Mapping)))
        object.__setattr__(self, "evidence_refs", tuple(dict.fromkeys(str(item).strip() for item in self.evidence_refs if str(item).strip())))
        if self.version != 1 or not self.artifact_hash or self.mutation_sequence < 0:
            raise DomainError("specialist review requires a current artifact hash and mutation sequence")
        if self.passed not in {True, False, None}:
            raise DomainError("specialist review passed must be true, false, or unknown")
        if self.passed is False and not self.issues:
            raise DomainError("a failed specialist review requires an actionable issue")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "SpecialistReviewResultV1",
            "version": self.version,
            "role": self.role.value,
            "artifact_hash": self.artifact_hash,
            "mutation_sequence": self.mutation_sequence,
            "passed": self.passed,
            "summary": self.summary,
            "issues": [dict(item) for item in self.issues],
            "evidence_refs": list(self.evidence_refs),
            "reviewer_id": self.reviewer_id,
        }


@dataclass(frozen=True, slots=True)
class FixedSpecialistReviewGateV1:
    """Deterministic evidence gate after all six fixed specialist reviews."""

    artifact_hash: str
    mutation_sequence: int
    reviews: tuple[SpecialistReviewResultV1, ...]
    version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "reviews", tuple(self.reviews))
        roles = [item.role for item in self.reviews]
        if self.version != 1 or not self.artifact_hash or self.mutation_sequence < 0:
            raise DomainError("fixed specialist gate requires a current artifact revision")
        if len(roles) != len(set(roles)):
            raise DomainError("fixed specialist gate cannot contain duplicate roles")

    @property
    def missing_roles(self) -> tuple[SpecialistReviewRole, ...]:
        present = {item.role for item in self.reviews}
        return tuple(role for role in FIXED_SPECIALIST_REVIEW_ORDER if role not in present)

    @property
    def stale_roles(self) -> tuple[SpecialistReviewRole, ...]:
        return tuple(
            item.role
            for item in self.reviews
            if item.artifact_hash != self.artifact_hash
            or item.mutation_sequence != self.mutation_sequence
        )

    @property
    def failed_roles(self) -> tuple[SpecialistReviewRole, ...]:
        return tuple(item.role for item in self.reviews if item.passed is False)

    @property
    def verdict(self) -> EvidenceVerdict:
        if self.missing_roles or self.stale_roles or any(item.passed is None for item in self.reviews):
            return EvidenceVerdict.NEEDS_EVIDENCE
        if self.failed_roles:
            return EvidenceVerdict.FAILED
        return EvidenceVerdict.PASSED

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "FixedSpecialistReviewGateV1",
            "version": self.version,
            "artifact_hash": self.artifact_hash,
            "mutation_sequence": self.mutation_sequence,
            "required_roles": [item.value for item in FIXED_SPECIALIST_REVIEW_ORDER],
            "reviews": [item.to_dict() for item in self.reviews],
            "missing_roles": [item.value for item in self.missing_roles],
            "stale_roles": [item.value for item in self.stale_roles],
            "failed_roles": [item.value for item in self.failed_roles],
            "verdict": self.verdict.value,
        }


class AdaptiveWorkerRouter:
    def __init__(self, policy: AdaptiveOrchestrationPolicyV1 | None = None) -> None:
        self.policy = policy or AdaptiveOrchestrationPolicyV1()

    def route(
        self,
        signals: TaskRiskSignalsV1,
        *,
        suppressed_roles: Iterable[WorkerRole | str] = (),
    ) -> WorkerRouteV1:
        tier = signals.tier
        if tier is RiskTier.LOW:
            roles: tuple[WorkerRole, ...] = ()
            reason = "deterministic verification is sufficient for the observed low-risk change"
        elif tier is RiskTier.MEDIUM:
            roles = (WorkerRole.FALSIFIER,)
            reason = "one blind falsifier targets the observed evidence or verification gap"
        elif tier is RiskTier.AMBIGUOUS:
            roles = (WorkerRole.CHALLENGER, WorkerRole.SELECTOR, WorkerRole.REPAIRER)
            reason = "independent staged candidates and anonymous evidence selection address design ambiguity"
        else:
            roles = (WorkerRole.PREDICTOR, WorkerRole.FALSIFIER, WorkerRole.REPAIRER)
            reason = "pre-mortem, falsification, and verified-finding repair cover the observed high-risk signals"

        suppressed = {WorkerRole(item) for item in suppressed_roles}
        roles = tuple(item for item in roles if item not in suppressed)
        extra_budget = (
            0
            if tier is RiskTier.LOW
            else self.policy.medium_extra_calls
            if tier is RiskTier.MEDIUM
            else self.policy.high_extra_calls
        )
        max_calls = min(self.policy.max_model_call_multiplier, 1 + extra_budget, 1 + len(roles))
        return WorkerRouteV1(
            tier=tier,
            roles=roles[: max(0, max_calls - 1)],
            max_model_calls=max_calls,
            reason=reason,
            early_stop_required=self.policy.early_stop_on_no_novel_evidence,
        )


def evidence_decision(claims: Sequence[EvidenceClaimV1]) -> EvidenceVerdict:
    """Aggregate claims without allowing a majority to override hard evidence."""

    items = tuple(claims)
    if any(item.hard_veto for item in items):
        return EvidenceVerdict.FAILED
    authoritative = tuple(item for item in items if item.authoritative)
    if not authoritative:
        return EvidenceVerdict.NEEDS_EVIDENCE
    criterion_ids = {item.criterion_id for item in items}
    passed_ids = {
        item.criterion_id
        for item in authoritative
        if item.verdict is EvidenceVerdict.PASSED
    }
    failed_ids = {
        item.criterion_id
        for item in authoritative
        if item.verdict is EvidenceVerdict.FAILED
    }
    if failed_ids:
        return EvidenceVerdict.FAILED
    return EvidenceVerdict.PASSED if criterion_ids and criterion_ids <= passed_ids else EvidenceVerdict.NEEDS_EVIDENCE


def evidence_novelty(existing_refs: Iterable[str], candidate_refs: Iterable[str]) -> float:
    existing = {str(item).strip() for item in existing_refs if str(item).strip()}
    candidate = {str(item).strip() for item in candidate_refs if str(item).strip()}
    if not candidate:
        return 0.0
    return round(len(candidate - existing) / len(candidate), 6)


def classify_worker_impact(
    *,
    verified_findings: int,
    false_findings: int,
    accepted_fixes: int,
    novelty: float,
    score_delta: float,
) -> WorkerImpactOutcome:
    if false_findings > verified_findings and score_delta <= 0:
        return WorkerImpactOutcome.HARMFUL
    if verified_findings > 0 or accepted_fixes > 0 or novelty > 0 or score_delta > 0:
        return WorkerImpactOutcome.USEFUL
    return WorkerImpactOutcome.NEUTRAL


def should_reprobe_suppressed_role(
    *,
    unit_id: str,
    role: WorkerRole | str,
    model_fingerprint: str,
    fraction: float = 0.10,
) -> bool:
    """Deterministic 10% exploration without random or restart-dependent state."""

    bounded = _bounded_ratio(fraction, field_name="reprobe fraction")
    if bounded <= 0:
        return False
    digest = hashlib.sha256(
        f"{unit_id}\0{WorkerRole(role).value}\0{model_fingerprint}".encode("utf-8")
    ).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64 - 1)
    return value < bounded


__all__ = [
    "AdaptiveOrchestrationPolicyV1",
    "AdaptiveWorkerRouter",
    "EvidenceAuthority",
    "EvidenceClaimV1",
    "EvidenceVerdict",
    "FIXED_SPECIALIST_REVIEW_ORDER",
    "FixedSpecialistReviewGateV1",
    "MutationPolicy",
    "OrchestrationArm",
    "OrchestrationExperimentV1",
    "RiskTier",
    "SpecialistReviewResultV1",
    "SpecialistReviewRole",
    "TaskRiskSignalsV1",
    "WorkerImpactOutcome",
    "WorkerImpactV1",
    "WorkerMissionV2",
    "WorkerRole",
    "WorkerRouteV1",
    "WorkerVisibility",
    "approach_fingerprint",
    "classify_worker_impact",
    "evidence_decision",
    "evidence_novelty",
    "should_reprobe_suppressed_role",
]
