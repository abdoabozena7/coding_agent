"""Formal swarm communication and consensus contracts.

These objects are intentionally small and deterministic.  They give agents a
stable protocol for messages, votes, and consensus without exposing hidden
reasoning or relying on ad hoc JSON blobs inside individual agent records.
"""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Mapping

from .models import new_id, utc_now


class SwarmMessageType(str, Enum):
    INFORM = "inform"
    REQUEST = "request"
    PROPOSAL = "proposal"
    VOTE = "vote"
    DECISION = "decision"
    BLOCKER = "blocker"
    CONTRACT_QUERY = "contract_query"
    CONTRACT_RESPONSE = "contract_response"
    PACKAGE_PUBLISHED = "package_published"
    QUALITY_FINDING = "quality_finding"
    REVISION_REQUEST = "revision_request"
    INTEGRATION_READY = "integration_ready"
    CONSENSUS_RESULT = "consensus_result"


class ConsensusStatus(str, Enum):
    OPEN = "open"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    TIED = "tied"


def _b64_encode(value: str | bytes) -> str:
    data = value if isinstance(value, bytes) else value.encode("utf-8")
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _b64_decode(value: str) -> bytes:
    text = str(value or "")
    padding = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode((text + padding).encode("ascii"))


def _parse_kv_frame(payload: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for part in str(payload).split():
        if "=" not in part:
            raise ValueError("invalid swarm DSL token")
        key, value = part.split("=", 1)
        if not key:
            raise ValueError("invalid swarm DSL key")
        result[key] = value
    return result


@dataclass(frozen=True, slots=True)
class SwarmMessageV1:
    ultra_run_id: str
    sender_agent_id: str
    recipient_agent_id: str
    message_type: SwarmMessageType | str
    topic: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    correlation_id: str = ""
    parent_message_id: str | None = None
    fencing_token: int = 0
    deadline: str | None = None
    evidence: tuple[Mapping[str, Any], ...] = ()
    schema_name: str = "SwarmMessageV1"
    id: str = field(default_factory=lambda: new_id("swarm_msg"))
    protocol_version: int = 1
    created_at: Any = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if not self.ultra_run_id or not self.sender_agent_id or not self.recipient_agent_id:
            raise ValueError("swarm message requires run, sender, and recipient")
        object.__setattr__(self, "message_type", SwarmMessageType(self.message_type))
        object.__setattr__(self, "topic", str(self.topic).strip()[:500])
        if not self.topic:
            raise ValueError("swarm message requires a topic")
        object.__setattr__(self, "payload", dict(self.payload))
        object.__setattr__(self, "confidence", max(0.0, min(1.0, float(self.confidence))))
        object.__setattr__(self, "fencing_token", max(0, int(self.fencing_token)))
        object.__setattr__(self, "evidence", tuple(dict(item) for item in self.evidence))
        if not str(self.schema_name).strip():
            raise ValueError("swarm message requires a schema name")

    def to_dict(self) -> dict[str, Any]:
        created = self.created_at.isoformat() if hasattr(self.created_at, "isoformat") else str(self.created_at)
        return {
            "id": self.id,
            "protocol_version": self.protocol_version,
            "ultra_run_id": self.ultra_run_id,
            "sender_agent_id": self.sender_agent_id,
            "recipient_agent_id": self.recipient_agent_id,
            "message_type": self.message_type.value,
            "topic": self.topic,
            "payload": dict(self.payload),
            "confidence": self.confidence,
            "correlation_id": self.correlation_id,
            "parent_message_id": self.parent_message_id,
            "fencing_token": self.fencing_token,
            "deadline": self.deadline,
            "evidence": [dict(item) for item in self.evidence],
            "schema_name": self.schema_name,
            "created_at": created,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SwarmMessageV1":
        created = value.get("created_at")
        if isinstance(created, str):
            try:
                created = datetime.fromisoformat(created)
            except ValueError:
                pass
        return cls(
            id=str(value.get("id") or new_id("swarm_msg")),
            protocol_version=int(value.get("protocol_version", 1)),
            ultra_run_id=str(value.get("ultra_run_id") or ""),
            sender_agent_id=str(value.get("sender_agent_id") or ""),
            recipient_agent_id=str(value.get("recipient_agent_id") or ""),
            message_type=value.get("message_type") or value.get("type") or SwarmMessageType.INFORM,
            topic=str(value.get("topic") or ""),
            payload=value.get("payload") if isinstance(value.get("payload"), Mapping) else {},
            confidence=float(value.get("confidence", 1.0) or 0.0),
            correlation_id=str(value.get("correlation_id") or ""),
            parent_message_id=str(value["parent_message_id"]) if value.get("parent_message_id") else None,
            fencing_token=int(value.get("fencing_token", 0) or 0),
            deadline=str(value["deadline"]) if value.get("deadline") else None,
            evidence=tuple(
                dict(item) for item in value.get("evidence", ()) if isinstance(item, Mapping)
            ),
            schema_name=str(value.get("schema_name") or "SwarmMessageV1"),
            created_at=created or utc_now(),
        )

    def encode_frame(self) -> str:
        """Encode one newline-free SWARM/1 frame for logs, sockets, or queues."""

        return "SWARM/1 " + json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @classmethod
    def decode_frame(cls, frame: str) -> "SwarmMessageV1":
        prefix = "SWARM/1 "
        if not str(frame).startswith(prefix):
            raise ValueError("invalid swarm frame prefix")
        data = json.loads(str(frame)[len(prefix):])
        if not isinstance(data, Mapping):
            raise ValueError("invalid swarm frame payload")
        return cls.from_mapping(data)

    def encode_dsl_frame(self) -> str:
        """Encode a compact key-value DSL frame for line transports.

        Free-form fields are base64url encoded so the frame remains a single
        whitespace-delimited line and can pass through logs, pipes, or sockets
        without ad-hoc escaping.
        """

        payload = json.dumps(dict(self.payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        parts = {
            "id": _b64_encode(self.id),
            "run": _b64_encode(self.ultra_run_id),
            "from": _b64_encode(self.sender_agent_id),
            "to": _b64_encode(self.recipient_agent_id),
            "type": self.message_type.value,
            "topic": _b64_encode(self.topic),
            "confidence": f"{self.confidence:.6f}",
            "correlation": _b64_encode(self.correlation_id),
            "parent": _b64_encode(self.parent_message_id or ""),
            "fencing": str(self.fencing_token),
            "deadline": _b64_encode(self.deadline or ""),
            "evidence": _b64_encode(json.dumps(list(self.evidence), ensure_ascii=False, sort_keys=True, separators=(",", ":"))),
            "schema": _b64_encode(self.schema_name),
            "created": _b64_encode(self.to_dict()["created_at"]),
            "payload": _b64_encode(payload),
        }
        return "SWARMDSL/1 " + " ".join(f"{key}={value}" for key, value in parts.items())

    @classmethod
    def decode_dsl_frame(cls, frame: str) -> "SwarmMessageV1":
        prefix = "SWARMDSL/1 "
        if not str(frame).startswith(prefix):
            raise ValueError("invalid swarm DSL frame prefix")
        values = _parse_kv_frame(str(frame)[len(prefix):])
        required = {"id", "run", "from", "to", "type", "topic", "confidence", "payload"}
        missing = sorted(required - set(values))
        if missing:
            raise ValueError(f"missing swarm DSL fields: {', '.join(missing)}")
        payload = json.loads(_b64_decode(values["payload"]).decode("utf-8") or "{}")
        if not isinstance(payload, Mapping):
            raise ValueError("invalid swarm DSL payload")
        return cls(
            id=_b64_decode(values["id"]).decode("utf-8"),
            ultra_run_id=_b64_decode(values["run"]).decode("utf-8"),
            sender_agent_id=_b64_decode(values["from"]).decode("utf-8"),
            recipient_agent_id=_b64_decode(values["to"]).decode("utf-8"),
            message_type=values["type"],
            topic=_b64_decode(values["topic"]).decode("utf-8"),
            payload=payload,
            confidence=float(values["confidence"]),
            correlation_id=_b64_decode(values.get("correlation", "")).decode("utf-8"),
            parent_message_id=(
                _b64_decode(values.get("parent", "")).decode("utf-8") or None
            ),
            fencing_token=int(values.get("fencing", "0") or 0),
            deadline=_b64_decode(values.get("deadline", "")).decode("utf-8") or None,
            evidence=tuple(
                dict(item)
                for item in json.loads(_b64_decode(values.get("evidence", "W10")).decode("utf-8") or "[]")
                if isinstance(item, Mapping)
            ),
            schema_name=_b64_decode(values.get("schema", _b64_encode("SwarmMessageV1"))).decode("utf-8") or "SwarmMessageV1",
            created_at=(
                datetime.fromisoformat(_b64_decode(values["created"]).decode("utf-8"))
                if values.get("created")
                else utc_now()
            ),
        )

    def encode_binary_frame(self) -> bytes:
        """Encode a length-prefixed binary-safe frame with a checksum."""

        payload = json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()
        header = f"SWARMBIN/1 {len(payload)} {digest}\n".encode("ascii")
        return header + payload

    @classmethod
    def decode_binary_frame(cls, frame: bytes) -> "SwarmMessageV1":
        if not isinstance(frame, (bytes, bytearray)):
            raise ValueError("binary swarm frame must be bytes")
        header, separator, payload = bytes(frame).partition(b"\n")
        if not separator:
            raise ValueError("invalid binary swarm frame header")
        parts = header.decode("ascii").split()
        if len(parts) != 3 or parts[0] != "SWARMBIN/1":
            raise ValueError("invalid binary swarm frame prefix")
        expected_length = int(parts[1])
        expected_digest = parts[2]
        if len(payload) != expected_length:
            raise ValueError("binary swarm frame length mismatch")
        if hashlib.sha256(payload).hexdigest() != expected_digest:
            raise ValueError("binary swarm frame checksum mismatch")
        data = json.loads(payload.decode("utf-8"))
        if not isinstance(data, Mapping):
            raise ValueError("invalid binary swarm frame payload")
        return cls.from_mapping(data)

    @classmethod
    def decode_any_frame(cls, frame: str | bytes) -> "SwarmMessageV1":
        if isinstance(frame, (bytes, bytearray)):
            return cls.decode_binary_frame(bytes(frame))
        text = str(frame)
        if text.startswith("SWARMDSL/1 "):
            return cls.decode_dsl_frame(text)
        return cls.decode_frame(text)


@dataclass(frozen=True, slots=True)
class ConsensusVoteV1:
    round_id: str
    voter_agent_id: str
    verdict: str
    confidence: float = 1.0
    rationale: str = ""
    evidence: Mapping[str, Any] = field(default_factory=dict)
    created_at: Any = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if not self.round_id or not self.voter_agent_id:
            raise ValueError("consensus vote requires round and voter")
        verdict = str(self.verdict).strip().casefold()
        if verdict not in {"accept", "reject", "abstain"}:
            raise ValueError("consensus vote verdict must be accept, reject, or abstain")
        object.__setattr__(self, "verdict", verdict)
        object.__setattr__(self, "confidence", max(0.0, min(1.0, float(self.confidence))))
        object.__setattr__(self, "rationale", str(self.rationale)[:2_000])
        object.__setattr__(self, "evidence", dict(self.evidence))


@dataclass(frozen=True, slots=True)
class ConsensusRoundV1:
    ultra_run_id: str
    topic: str
    leader_agent_id: str
    quorum: int
    status: ConsensusStatus | str = ConsensusStatus.OPEN
    decision: Mapping[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: new_id("consensus"))
    created_at: Any = field(default_factory=utc_now)
    updated_at: Any = field(default_factory=utc_now)
    closed_at: Any | None = None

    def __post_init__(self) -> None:
        if not self.ultra_run_id or not self.topic.strip() or not self.leader_agent_id:
            raise ValueError("consensus round requires run, topic, and leader")
        if self.quorum < 1:
            raise ValueError("consensus quorum must be positive")
        object.__setattr__(self, "status", ConsensusStatus(self.status))
        object.__setattr__(self, "topic", str(self.topic).strip()[:500])
        object.__setattr__(self, "decision", dict(self.decision))
