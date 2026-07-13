"""Harness-owned reasoning scaffolds for weak local models.

The harness never asks for or stores hidden chain-of-thought.  This module
provides a small, auditable frame that tells agents which external reasoning
artifacts must be summarized: assumptions, alternatives, verification, risks,
and short debate-style objections.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class ReasoningScaffoldV1:
    role: str
    phase: str
    mode: str
    required_summary_fields: tuple[str, ...]
    debate_prompts: tuple[str, ...]
    verification_bias: str
    version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "role": self.role,
            "phase": self.phase,
            "mode": self.mode,
            "required_summary_fields": list(self.required_summary_fields),
            "debate_prompts": list(self.debate_prompts),
            "verification_bias": self.verification_bias,
            "privacy_rule": "Do not reveal hidden chain-of-thought; return concise decisions and evidence only.",
        }


@dataclass(frozen=True, slots=True)
class ReasoningDebateProtocolV1:
    role: str
    phase: str
    required: bool
    required_fields: tuple[str, ...]
    critic_questions: tuple[str, ...]
    minimum_evidence_items: int = 1
    graph_required: bool = True
    minimum_reasoning_nodes: int = 2
    minimum_reasoning_edges: int = 1
    version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "role": self.role,
            "phase": self.phase,
            "required": self.required,
            "required_fields": list(self.required_fields),
            "critic_questions": list(self.critic_questions),
            "minimum_evidence_items": self.minimum_evidence_items,
            "external_reasoning_graph": {
                "required": self.graph_required,
                "output_key": "reasoning_graph",
                "minimum_nodes": self.minimum_reasoning_nodes,
                "minimum_edges": self.minimum_reasoning_edges,
                "node_contract": {
                    "id": "stable short id",
                    "type": "assumption|option|decision|risk|verification",
                    "summary": "short external conclusion, not hidden chain-of-thought",
                    "status": "chosen|rejected|open|verified",
                    "evidence_refs": ["tool/test/hash/browser evidence ids when available"],
                },
                "edge_contract": {
                    "from": "source node id",
                    "to": "target node id",
                    "relation": "supports|depends_on|contradicts|verifies|rejects",
                },
            },
            "output_key": "reasoning_artifact",
            "privacy_rule": (
                "Return short external conclusions only. Do not expose hidden chain-of-thought, "
                "private scratchpad, token-by-token reasoning, or internal deliberation."
            ),
        }


@dataclass(frozen=True, slots=True)
class ReasoningArtifactEvaluationV1:
    passed: bool
    score: float
    missing_fields: tuple[str, ...] = ()
    weak_fields: tuple[str, ...] = ()
    evidence_items: int = 0
    notes: tuple[str, ...] = field(default_factory=tuple)
    version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "passed": self.passed,
            "score": self.score,
            "missing_fields": list(self.missing_fields),
            "weak_fields": list(self.weak_fields),
            "evidence_items": self.evidence_items,
            "notes": list(self.notes),
        }


def reasoning_scaffold_for(role: str, phase: str, task: Mapping[str, Any]) -> ReasoningScaffoldV1:
    contract = task.get("contract", {}) if isinstance(task, Mapping) else {}
    objective = str(contract.get("objective") or task.get("prompt") or "").strip()
    base_fields = (
        "assumptions_checked",
        "candidate_approaches_considered",
        "chosen_approach_and_reason",
        "verification_plan",
        "remaining_risks",
    )
    debate_prompts = (
        "What would make this output fail the explicit contract?",
        "What is the simplest alternative that satisfies the same contract?",
        "What evidence would disprove the current success claim?",
    )
    if role in {"tester", "reviewer", "goal_checker", "quality_triager"}:
        debate_prompts = (
            "What concrete evidence supports passing this gate?",
            "What failure would a weak model be likely to miss?",
            "Which claim is not executable, observable, or hash-bound yet?",
        )
    if role in {"coder", "integrator"}:
        debate_prompts = (
            "Which existing file or interface can be reused before inventing new structure?",
            "What minimal mutation proves progress without broad collateral change?",
            "What verification should run immediately after this mutation?",
        )
    return ReasoningScaffoldV1(
        role=role,
        phase=phase,
        mode="external_structured_summary",
        required_summary_fields=base_fields + (("objective",) if objective else ()),
        debate_prompts=debate_prompts,
        verification_bias="prefer executable/browser/static evidence over prose confidence",
    )


def reasoning_debate_protocol_for(role: str, phase: str, task: Mapping[str, Any]) -> ReasoningDebateProtocolV1:
    del task
    critical_roles = {
        "architect",
        "planner",
        "coder",
        "integrator",
        "reviewer",
        "tester",
        "goal_checker",
        "clean_code_reviewer",
        "security_reviewer",
        "test_quality_reviewer",
        "quality_triager",
    }
    critical_phases = {
        "architecture",
        "master_plan",
        "implement",
        "fix",
        "review",
        "test",
        "integrate",
        "global_integration",
        "global_review",
        "final_evidence",
    }
    required = role in critical_roles or phase in critical_phases
    critic_questions = (
        "Which success claim is weakest or least evidenced?",
        "What alternative approach was rejected and why?",
        "What concrete check could falsify the current answer?",
    )
    if role in {"tester", "reviewer", "goal_checker", "quality_triager", "clean_code_reviewer", "security_reviewer", "test_quality_reviewer"}:
        critic_questions = (
            "Which pass/fail claim is directly observable?",
            "Which issue would a small local model most likely overlook?",
            "What evidence would force a reject vote?",
        )
    if role in {"coder", "integrator"}:
        critic_questions = (
            "What existing contract/interface constrains this change?",
            "What simpler implementation path was rejected?",
            "What immediate verification proves the mutation worked?",
        )
    return ReasoningDebateProtocolV1(
        role=role,
        phase=phase,
        required=required,
        required_fields=(
            "claim",
            "supporting_evidence",
            "counterarguments",
            "rejected_alternatives",
            "verification_plan",
        ),
        critic_questions=critic_questions,
        minimum_evidence_items=1,
        graph_required=required,
        minimum_reasoning_nodes=2,
        minimum_reasoning_edges=1,
    )


def _items(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        return (dict(value),)
    if isinstance(value, (str, bytes)):
        return (value,) if str(value).strip() else ()
    if isinstance(value, (list, tuple)):
        return tuple(item for item in value if str(item).strip())
    return (value,)


def _reasoning_graph_findings(
    artifact: Mapping[str, Any],
    protocol: ReasoningDebateProtocolV1,
) -> tuple[bool, tuple[str, ...], tuple[str, ...]]:
    if not protocol.graph_required:
        return True, (), ()
    graph = artifact.get("reasoning_graph")
    if not isinstance(graph, Mapping):
        return False, ("reasoning_graph",), ("reasoning_graph missing or non-object",)
    nodes = graph.get("nodes")
    edges = graph.get("edges")
    node_items = tuple(item for item in _items(nodes) if isinstance(item, Mapping))
    edge_items = tuple(item for item in _items(edges) if isinstance(item, Mapping))
    weak: list[str] = []
    notes: list[str] = []
    if len(node_items) < protocol.minimum_reasoning_nodes:
        weak.append("reasoning_graph.nodes")
        notes.append("reasoning graph has too few external decision/alternative nodes")
    if len(edge_items) < protocol.minimum_reasoning_edges:
        weak.append("reasoning_graph.edges")
        notes.append("reasoning graph has too few support/contradiction/verification edges")
    statuses = {
        str(item.get("status") or "").casefold()
        for item in node_items
        if str(item.get("status") or "").strip()
    }
    if "chosen" not in statuses and "verified" not in statuses:
        weak.append("reasoning_graph.chosen_node")
        notes.append("reasoning graph lacks a chosen or verified node")
    if "rejected" not in statuses and len(node_items) >= 2:
        weak.append("reasoning_graph.rejected_alternative")
        notes.append("reasoning graph lacks a rejected alternative node")
    node_ids = {
        str(item.get("id") or "").strip()
        for item in node_items
        if str(item.get("id") or "").strip()
    }
    for edge in edge_items:
        source = str(edge.get("from") or "").strip()
        target = str(edge.get("to") or "").strip()
        relation = str(edge.get("relation") or "").strip().casefold()
        if source not in node_ids or target not in node_ids:
            weak.append("reasoning_graph.edge_refs")
            notes.append("reasoning graph edge references an unknown node")
            break
        if relation not in {"supports", "depends_on", "contradicts", "verifies", "rejects"}:
            weak.append("reasoning_graph.edge_relation")
            notes.append("reasoning graph edge uses an unsupported relation")
            break
    return not weak, tuple(dict.fromkeys(weak)), tuple(dict.fromkeys(notes))


def evaluate_reasoning_artifact(
    artifact: Any,
    protocol: ReasoningDebateProtocolV1,
) -> ReasoningArtifactEvaluationV1:
    if not protocol.required:
        return ReasoningArtifactEvaluationV1(passed=True, score=1.0, notes=("not required",))
    if not isinstance(artifact, Mapping):
        return ReasoningArtifactEvaluationV1(
            passed=False,
            score=0.0,
            missing_fields=protocol.required_fields,
            notes=("reasoning_artifact missing or non-object",),
        )
    missing: list[str] = []
    weak: list[str] = []
    score = 0.0
    for field_name in protocol.required_fields:
        if field_name not in artifact:
            missing.append(field_name)
            continue
        values = _items(artifact.get(field_name))
        if not values:
            weak.append(field_name)
            continue
        score += 1.0 / len(protocol.required_fields)
    evidence_items = len(_items(artifact.get("supporting_evidence")))
    if evidence_items < protocol.minimum_evidence_items:
        weak.append("supporting_evidence")
    graph_passed, graph_weak, graph_notes = _reasoning_graph_findings(artifact, protocol)
    if not graph_passed:
        weak.extend(graph_weak)
    passed = not missing and not weak and score >= 0.8
    notes = graph_notes
    if passed:
        notes = ("external debate/graph artifact satisfied",)
    return ReasoningArtifactEvaluationV1(
        passed=passed,
        score=round(score, 4),
        missing_fields=tuple(missing),
        weak_fields=tuple(dict.fromkeys(weak)),
        evidence_items=evidence_items,
        notes=notes,
    )
