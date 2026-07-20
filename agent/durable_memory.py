"""Durable weak-model memory and resumable next-action packets.

The model is an executor, not the state store.  These records keep the complete
workflow outside the context window and project only the smallest actionable
slice into each specialist call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import hashlib
import json
from typing import Any, Mapping

from .models import DomainError, new_id, utc_now


def _fingerprint(value: Mapping[str, Any]) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class NextActionStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    RECOVERING = "recovering"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class NextActionPacketV1:
    ultra_run_id: str
    role: str
    phase: str
    objective: str
    work_node_id: str | None = None
    contract: Mapping[str, Any] = field(default_factory=dict)
    checkpoint: Mapping[str, Any] = field(default_factory=dict)
    dependency_evidence: tuple[Mapping[str, Any], ...] = ()
    relevant_memory: tuple[Mapping[str, Any], ...] = ()
    required_outputs: tuple[str, ...] = ()
    omitted_sections: tuple[str, ...] = ()
    context_budget_chars: int = 16_000
    sequence: int = 0
    version: int = 1

    def __post_init__(self) -> None:
        if self.version != 1 or not self.ultra_run_id or not self.role or not self.phase:
            raise DomainError("NextActionPacketV1 requires run, role, and phase")
        if not self.objective.strip():
            raise DomainError("next action requires a bounded objective")
        if not 2_000 <= self.context_budget_chars <= 120_000:
            raise DomainError("next-action context budget is outside safe bounds")
        object.__setattr__(self, "contract", dict(self.contract))
        object.__setattr__(self, "checkpoint", dict(self.checkpoint))
        object.__setattr__(
            self,
            "dependency_evidence",
            tuple(dict(item) for item in self.dependency_evidence),
        )
        object.__setattr__(
            self,
            "relevant_memory",
            tuple(dict(item) for item in self.relevant_memory),
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": "NextActionPacketV1",
            "version": 1,
            "ultra_run_id": self.ultra_run_id,
            "work_node_id": self.work_node_id,
            "role": self.role,
            "phase": self.phase,
            "objective": self.objective,
            "contract": dict(self.contract),
            "checkpoint": dict(self.checkpoint),
            "dependency_evidence": [dict(item) for item in self.dependency_evidence],
            "relevant_memory": [dict(item) for item in self.relevant_memory],
            "required_outputs": list(self.required_outputs),
            "omitted_sections": list(self.omitted_sections),
            "context_budget_chars": self.context_budget_chars,
            "sequence": self.sequence,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "fingerprint": self.fingerprint}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "NextActionPacketV1":
        return cls(
            ultra_run_id=str(value.get("ultra_run_id") or ""),
            work_node_id=str(value.get("work_node_id") or "") or None,
            role=str(value.get("role") or ""),
            phase=str(value.get("phase") or ""),
            objective=str(value.get("objective") or ""),
            contract=dict(value.get("contract") or {}),
            checkpoint=dict(value.get("checkpoint") or {}),
            dependency_evidence=tuple(value.get("dependency_evidence") or ()),
            relevant_memory=tuple(value.get("relevant_memory") or ()),
            required_outputs=tuple(str(item) for item in value.get("required_outputs", ())),
            omitted_sections=tuple(str(item) for item in value.get("omitted_sections", ())),
            context_budget_chars=int(value.get("context_budget_chars", 16_000)),
            sequence=int(value.get("sequence", 0)),
            version=int(value.get("version", 1)),
        )

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self._payload())


@dataclass(frozen=True, slots=True)
class AgentMemorySnapshotV1:
    ultra_run_id: str
    role: str
    objective: str
    work_node_id: str | None = None
    checkpoint: str = ""
    completed_actions: tuple[str, ...] = ()
    open_findings: tuple[Mapping[str, Any], ...] = ()
    decision_refs: tuple[str, ...] = ()
    artifact_refs: tuple[str, ...] = ()
    dependency_refs: tuple[str, ...] = ()
    last_result: Mapping[str, Any] = field(default_factory=dict)
    next_action_id: str | None = None
    id: str = field(default_factory=lambda: new_id("agent_memory"))
    revision: int = 1
    created_at: datetime = field(default_factory=utc_now)
    version: int = 1

    def __post_init__(self) -> None:
        if self.version != 1 or not self.ultra_run_id or not self.role:
            raise DomainError("AgentMemorySnapshotV1 requires run and role")
        if self.revision < 1 or not self.objective.strip():
            raise DomainError("agent memory requires an objective and positive revision")
        object.__setattr__(
            self, "open_findings", tuple(dict(item) for item in self.open_findings)
        )
        object.__setattr__(self, "last_result", dict(self.last_result))

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": "AgentMemorySnapshotV1",
            "version": 1,
            "id": self.id,
            "ultra_run_id": self.ultra_run_id,
            "work_node_id": self.work_node_id,
            "role": self.role,
            "objective": self.objective,
            "checkpoint": self.checkpoint,
            "completed_actions": list(self.completed_actions),
            "open_findings": [dict(item) for item in self.open_findings],
            "decision_refs": list(self.decision_refs),
            "artifact_refs": list(self.artifact_refs),
            "dependency_refs": list(self.dependency_refs),
            "last_result": dict(self.last_result),
            "next_action_id": self.next_action_id,
            "revision": self.revision,
            "created_at": self.created_at.isoformat(),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "fingerprint": self.fingerprint}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AgentMemorySnapshotV1":
        created_at = value.get("created_at")
        return cls(
            id=str(value.get("id") or new_id("agent_memory")),
            ultra_run_id=str(value.get("ultra_run_id") or ""),
            work_node_id=str(value.get("work_node_id") or "") or None,
            role=str(value.get("role") or ""),
            objective=str(value.get("objective") or ""),
            checkpoint=str(value.get("checkpoint") or ""),
            completed_actions=tuple(str(item) for item in value.get("completed_actions", ())),
            open_findings=tuple(value.get("open_findings") or ()),
            decision_refs=tuple(str(item) for item in value.get("decision_refs", ())),
            artifact_refs=tuple(str(item) for item in value.get("artifact_refs", ())),
            dependency_refs=tuple(str(item) for item in value.get("dependency_refs", ())),
            last_result=dict(value.get("last_result") or {}),
            next_action_id=str(value.get("next_action_id") or "") or None,
            revision=int(value.get("revision", 1)),
            created_at=(
                datetime.fromisoformat(str(created_at)) if created_at else utc_now()
            ),
            version=int(value.get("version", 1)),
        )

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self._payload())


__all__ = [
    "AgentMemorySnapshotV1",
    "NextActionPacketV1",
    "NextActionStatus",
]
