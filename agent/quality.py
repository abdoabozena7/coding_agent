"""Versioned Change Set and quality lifecycle contracts for Ultra/Sleep."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import Enum
import hashlib
import json
from typing import Any, Mapping

from .models import DomainError, utc_now, new_id


class QualityCategory(str, Enum):
    CLEAN_CODE = "clean_code"
    SECURITY = "security"
    TEST_QUALITY = "test_quality"
    ARCHITECTURE = "architecture"
    API = "api"
    BACKWARD_COMPATIBILITY = "backward_compatibility"
    PERFORMANCE = "performance"
    VISUAL = "visual"


class FindingSeverity(str, Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def blocks_completion(self) -> bool:
        return self in {self.MEDIUM, self.HIGH, self.CRITICAL}


class FindingStatus(str, Enum):
    OPEN = "open"
    ACCEPTED = "accepted"
    REMEDIATING = "remediating"
    RESOLVED = "resolved"
    REPORTED = "reported"
    REQUIRES_PLAN_REVISION = "requires_plan_revision"


class QualityCycleKind(str, Enum):
    BASELINE = "baseline"
    DELTA = "delta"
    PROJECT_SWEEP = "project_sweep"


class ChangeSetStatus(str, Enum):
    OPEN = "open"
    CLOSED = "closed"
    REVIEWING = "reviewing"
    APPROVED = "approved"
    BLOCKED = "blocked"
    INTEGRATED = "integrated"
    UNCERTAIN = "uncertain"


PRINCIPLES: tuple[str, ...] = (
    "correctness", "simplicity", "naming", "cohesion_srp", "coupling",
    "duplication", "abstraction_quality", "error_handling", "resource_handling",
    "types_contracts", "input_validation", "authentication_authorization",
    "secrets", "permissions", "unsafe_apis", "dependency_security",
    "testability", "determinism", "coverage", "architecture_consistency",
    "api_consistency", "backward_compatibility",
    "visual_quality",
)


def _hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class QualityPolicyV1:
    version: int = 1
    principles: tuple[str, ...] = PRINCIPLES
    blocking_severities: tuple[str, ...] = ("critical", "high", "medium")
    required_reviews: tuple[str, ...] = ("clean_code", "security", "test_quality")
    performance_requires_measurement: bool = True

    def __post_init__(self) -> None:
        if self.version != 1:
            raise DomainError("QualityPolicyV1 only accepts version 1")
        unknown = set(self.principles) - set(PRINCIPLES)
        if unknown:
            raise DomainError(f"unknown quality principles: {sorted(unknown)}")

    def to_dict(self) -> dict[str, Any]:
        return {"version": self.version, "principles": list(self.principles), "blocking_severities": list(self.blocking_severities), "required_reviews": list(self.required_reviews), "performance_requires_measurement": self.performance_requires_measurement}


@dataclass(frozen=True, slots=True)
class ChangeSetV1:
    ultra_run_id: str
    responsible_agent_id: str
    parent_id: str
    id: str = field(default_factory=lambda: new_id("changeset"))
    version: int = 1
    status: ChangeSetStatus = ChangeSetStatus.OPEN
    changed_files: tuple[str, ...] = ()
    pre_hashes: Mapping[str, str | None] = field(default_factory=dict)
    post_hashes: Mapping[str, str | None] = field(default_factory=dict)
    diff: str = ""
    mutation_commands: tuple[str, ...] = ()
    shell_created_files: tuple[str, ...] = ()
    verification_evidence_ids: tuple[str, ...] = ()
    review_status: Mapping[str, str] = field(default_factory=dict)
    integration_status: str = "pending"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if self.version != 1 or not self.ultra_run_id or not self.responsible_agent_id or not self.parent_id:
            raise DomainError("ChangeSetV1 requires version 1, run, agent, and parent")
        object.__setattr__(self, "changed_files", tuple(dict.fromkeys(self.changed_files)))
        object.__setattr__(self, "pre_hashes", dict(self.pre_hashes))
        object.__setattr__(self, "post_hashes", dict(self.post_hashes))
        object.__setattr__(self, "review_status", dict(self.review_status))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def review_complete(self) -> bool:
        return all(self.review_status.get(role) == "passed" for role in ("clean_code", "security", "test_quality"))

    def integrate(self) -> "ChangeSetV1":
        if self.status is not ChangeSetStatus.APPROVED or not self.review_complete:
            raise DomainError("an unreviewed Change Set cannot integrate")
        return replace(self, status=ChangeSetStatus.INTEGRATED, integration_status="integrated", updated_at=utc_now())


@dataclass(frozen=True, slots=True)
class QualityFindingV1:
    ultra_run_id: str
    principle_id: str
    category: QualityCategory
    severity: FindingSeverity
    path: str
    location: str
    file_hash: str
    evidence: Mapping[str, Any]
    remediation: str
    acceptance_criteria: tuple[str, ...]
    verification: tuple[str, ...]
    id: str = field(default_factory=lambda: new_id("finding"))
    version: int = 1
    status: FindingStatus = FindingStatus.OPEN
    repair_node_id: str | None = None
    fingerprint: str = ""
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if self.version != 1 or self.principle_id not in PRINCIPLES:
            raise DomainError("QualityFindingV1 has an invalid version or principle")
        if self.category is QualityCategory.PERFORMANCE and not self.evidence.get("measurement"):
            raise DomainError("performance findings require measurable profiling evidence")
        if not self.path or not self.file_hash or not self.remediation or not self.acceptance_criteria or not self.verification:
            raise DomainError("quality finding is missing required evidence or remediation fields")
        expected = _hash({"category": self.category.value, "principle": self.principle_id, "path": self.path, "location": self.location, "file_hash": self.file_hash, "evidence": self.evidence})
        object.__setattr__(self, "fingerprint", self.fingerprint or expected)
        object.__setattr__(self, "evidence", dict(self.evidence))


@dataclass(frozen=True, slots=True)
class QualityCycleV1:
    ultra_run_id: str
    kind: QualityCycleKind
    attempt: int
    approach_fingerprint: str
    inputs: Mapping[str, Any]
    outputs: Mapping[str, Any]
    metrics: Mapping[str, Any]
    result: str
    id: str = field(default_factory=lambda: new_id("quality_cycle"))
    version: int = 1
    blocker: str | None = None
    started_at: datetime = field(default_factory=utc_now)
    ended_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.version != 1 or self.attempt < 1 or not self.approach_fingerprint:
            raise DomainError("QualityCycleV1 requires version 1, an attempt, and approach fingerprint")


def completion_blockers(change_sets: list[ChangeSetV1], findings: list[QualityFindingV1]) -> tuple[str, ...]:
    blockers: list[str] = []
    for item in change_sets:
        if item.status is not ChangeSetStatus.INTEGRATED:
            blockers.append(f"Change Set {item.id} is {item.status.value}")
    for item in findings:
        if item.severity.blocks_completion and item.status is not FindingStatus.RESOLVED:
            blockers.append(f"{item.severity.value} finding {item.id} remains {item.status.value}")
    return tuple(blockers)


def finding_requires_plan_revision(
    *,
    public_behavior: bool = False,
    public_interface: bool = False,
    dependency_change: bool = False,
    path_scope_change: bool = False,
    architecture_change: bool = False,
) -> bool:
    return any((public_behavior, public_interface, dependency_change, path_scope_change, architecture_change))


def may_auto_fix_low(finding: QualityFindingV1, *, deterministic: bool, safe: bool, within_approved_scope: bool) -> bool:
    return (
        finding.severity is FindingSeverity.LOW
        and deterministic
        and safe
        and within_approved_scope
        and finding.status in {FindingStatus.OPEN, FindingStatus.ACCEPTED}
    )
