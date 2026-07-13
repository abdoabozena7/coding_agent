"""Higher-level swarm proposal, voting, and decision workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .swarm_bus import SwarmBus
from .swarm_protocol import ConsensusStatus, ConsensusVoteV1, SwarmMessageType, SwarmMessageV1


@dataclass(frozen=True, slots=True)
class SwarmDecisionWorkflowV1:
    proposal_message_id: str
    consensus_round_id: str
    leader_agent_id: str
    voter_agent_ids: tuple[str, ...]
    request_message_ids: tuple[str, ...]


class SwarmCoordinator:
    """Coordinate the formal proposal -> votes -> decision lifecycle."""

    def __init__(self, store: Any, bus: SwarmBus | None = None) -> None:
        self.store = store
        self.bus = bus or SwarmBus(store)

    def propose(
        self,
        *,
        ultra_run_id: str,
        proposer_agent_id: str,
        topic: str,
        proposal: Mapping[str, Any],
        voters: Iterable[str],
        quorum: int | None = None,
        leader_agent_id: str | None = None,
        confidence: float = 1.0,
    ) -> SwarmDecisionWorkflowV1:
        voter_ids = tuple(dict.fromkeys(str(item) for item in voters if str(item).strip()))
        if not voter_ids:
            raise ValueError("swarm proposal requires at least one voter")
        proposal_message = self.bus.publish(
            SwarmMessageV1(
                ultra_run_id=ultra_run_id,
                sender_agent_id=proposer_agent_id,
                recipient_agent_id=leader_agent_id or "swarm",
                message_type=SwarmMessageType.PROPOSAL,
                topic=topic,
                payload=dict(proposal),
                confidence=confidence,
                correlation_id=topic,
            )
        )
        round_item = self.store.open_consensus_round(
            ultra_run_id=ultra_run_id,
            topic=topic,
            leader_agent_id=leader_agent_id,
            quorum=quorum or len(voter_ids),
            candidates=(leader_agent_id,) if leader_agent_id else voter_ids,
        )
        request_ids: list[str] = []
        for voter_id in voter_ids:
            request = self.bus.publish(
                SwarmMessageV1(
                    ultra_run_id=ultra_run_id,
                    sender_agent_id=round_item["leader_agent_id"],
                    recipient_agent_id=voter_id,
                    message_type=SwarmMessageType.REQUEST,
                    topic=f"consensus-vote:{round_item['id']}",
                    payload={
                        "round_id": round_item["id"],
                        "proposal_message_id": proposal_message["id"],
                        "proposal": dict(proposal),
                        "topic": topic,
                        "quorum": round_item["quorum"],
                    },
                    confidence=1.0,
                    correlation_id=round_item["id"],
                    parent_message_id=proposal_message["id"],
                )
            )
            request_ids.append(str(request["id"]))
        return SwarmDecisionWorkflowV1(
            proposal_message_id=str(proposal_message["id"]),
            consensus_round_id=str(round_item["id"]),
            leader_agent_id=str(round_item["leader_agent_id"]),
            voter_agent_ids=voter_ids,
            request_message_ids=tuple(request_ids),
        )

    def submit_vote(
        self,
        *,
        round_id: str,
        voter_agent_id: str,
        verdict: str,
        confidence: float = 1.0,
        rationale: str = "",
        evidence: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        updated = self.store.record_consensus_vote(
            ConsensusVoteV1(
                round_id=round_id,
                voter_agent_id=voter_agent_id,
                verdict=verdict,
                confidence=confidence,
                rationale=rationale,
                evidence=dict(evidence or {}),
            )
        )
        if ConsensusStatus(updated["status"]) is not ConsensusStatus.OPEN:
            self._publish_decision(updated)
        return updated

    def _publish_decision(self, round_item: Mapping[str, Any]) -> Mapping[str, Any]:
        return self.bus.publish(
            SwarmMessageV1(
                ultra_run_id=str(round_item["ultra_run_id"]),
                sender_agent_id=str(round_item["leader_agent_id"]),
                recipient_agent_id="swarm",
                message_type=SwarmMessageType.DECISION,
                topic=f"consensus-decision:{round_item['id']}",
                payload={
                    "round_id": round_item["id"],
                    "status": round_item["status"],
                    "decision": round_item.get("decision", {}),
                    "votes": round_item.get("votes", ()),
                    "topic": round_item.get("topic", ""),
                },
                confidence=1.0 if round_item["status"] == ConsensusStatus.ACCEPTED.value else 0.0,
                correlation_id=str(round_item["id"]),
            )
        )

