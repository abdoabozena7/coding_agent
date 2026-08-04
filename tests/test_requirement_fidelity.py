from types import SimpleNamespace
from pathlib import Path
import tempfile

import pytest

from agent.control import SEMANTIC_GOAL_SCHEMA, TASK_SCHEMA
from agent.prompts import (
    COORDINATOR_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    PLAN_REVIEWER_SYSTEM_PROMPT,
    REVIEWER_SYSTEM_PROMPT,
)
from agent.runtime import AgentRuntime
from agent.semantic import SemanticContractError, SemanticGoalV2
from agent.store import StateStore
from agent.testing import ScriptedProvider
from agent.workflow import normalize_plan_draft


REQUEST = "Create a Three.js 3D calculator with impressive design and run it"


def _semantic() -> dict:
    return {
        "original_request": REQUEST,
        "interpreted_outcome": "Create a visibly three-dimensional interactive calculator.",
        "requested_effects": ["read_workspace", "mutate_workspace", "execute_code"],
        "required_outcomes": ["A runnable 3D calculator is rendered with Three.js."],
        "constraints": [],
        "exclusions": [],
        "acceptance_criteria": [
            "The rendered calculator visibly uses perspective, depth, lit geometry, and interactive controls."
        ],
        "requirement_anchors": [
            {
                "id": "R001",
                "verbatim_span": "Three.js 3D calculator",
                "interpreted_requirement": (
                    "Three.js is the visible 3D interaction medium, not only an installed dependency."
                ),
                "observable_implications": [
                    "The runtime renders calculator surfaces as lit 3D geometry through a perspective camera.",
                    "Input interaction changes the calculator state inside the rendered experience.",
                ],
                "kind": "technology_and_medium",
            },
            {
                "id": "R002",
                "verbatim_span": "impressive design",
                "interpreted_requirement": "The finished experience needs a deliberate visual direction.",
                "observable_implications": [
                    "Rendered evidence shows a coherent palette, lighting, depth, and motion treatment."
                ],
                "kind": "experiential_quality",
            },
        ],
        "unresolved_decisions": [],
        "repository_evidence_refs": ["inspection:I001"],
    }


def _candidate(refs: list[str]) -> dict:
    return {
        "semantic_goal": _semantic(),
        "expected_changes": [{"path": "main.js"}],
        "tasks": [
            {
                "title": "Build and verify the 3D calculator",
                "description": "Implement the accepted visible experience.",
                "acceptance_criteria": ["The anchored 3D and visual outcomes are observable."],
                "verification": ["Run browser verification and inspect rendered evidence."],
                "requirement_refs": refs,
            }
        ],
    }


def test_requirement_anchor_span_must_be_exact_user_text() -> None:
    value = _semantic()
    value["requirement_anchors"][0]["verbatim_span"] = "Babylon.js"
    with pytest.raises(SemanticContractError, match="verbatim substrings"):
        SemanticGoalV2.from_mapping(value, original_request=REQUEST)


def test_plan_must_cover_every_model_authored_requirement_anchor() -> None:
    goal = SimpleNamespace(objective=REQUEST)
    with pytest.raises(ValueError, match="do not cover requirement anchors: R002"):
        AgentRuntime._validate_semantic_candidate(
            goal,
            _candidate(["R001"]),
            successful_inspection_ids=frozenset({"I001"}),
        )


def test_plan_rejects_unknown_requirement_anchor_reference() -> None:
    goal = SimpleNamespace(objective=REQUEST)
    with pytest.raises(ValueError, match="unknown requirement anchors: R999"):
        AgentRuntime._validate_semantic_candidate(
            goal,
            _candidate(["R001", "R002", "R999"]),
            successful_inspection_ids=frozenset({"I001"}),
        )


def test_valid_anchor_coverage_survives_plan_normalization() -> None:
    candidate = _candidate(["R001", "R002"])
    candidate.update(
        {
            "summary": "Build the anchored experience.",
            "applicability_evidence": [{"fact": "The workspace was inspected."}],
            "execution_strategy": "Implement, run, and visually verify the anchored result.",
        }
    )
    normalized, _actions = normalize_plan_draft(candidate)
    assert normalized["tasks"][0]["requirement_refs"] == ["R001", "R002"]
    semantic = AgentRuntime._validate_semantic_candidate(
        SimpleNamespace(objective=REQUEST),
        normalized,
        successful_inspection_ids=frozenset({"I001"}),
    )
    assert [item.id for item in semantic.requirement_anchors] == ["R001", "R002"]


def test_omitted_refs_are_bound_when_all_task_text_covers_the_anchors() -> None:
    candidate = _candidate([])
    candidate.update(
        {
            "summary": "Build the anchored experience.",
            "applicability_evidence": [{"fact": "The workspace was inspected."}],
            "execution_strategy": "Implement, run, and visually verify the anchored result.",
        }
    )
    semantic = AgentRuntime._validate_semantic_candidate(
        SimpleNamespace(objective=REQUEST),
        candidate,
        successful_inspection_ids=frozenset({"I001"}),
    )
    assert [item.id for item in semantic.requirement_anchors] == ["R001", "R002"]
    assert candidate["tasks"][0]["requirement_refs"] == ["R001", "R002"]


def test_new_provider_contract_requires_anchors_and_task_traceability() -> None:
    assert "requirement_anchors" in SEMANTIC_GOAL_SCHEMA["required"]
    assert "requirement_refs" in TASK_SCHEMA["properties"]


def test_all_model_stages_reject_superficial_technology_compliance() -> None:
    combined = " ".join(
        (
            PLANNER_SYSTEM_PROMPT,
            PLAN_REVIEWER_SYSTEM_PROMPT,
            COORDINATOR_SYSTEM_PROMPT,
            REVIEWER_SYSTEM_PROMPT,
        )
    ).casefold()
    assert "distinctive capability" in combined
    assert "installation/import" in combined


def test_execution_state_exposes_accepted_anchors_and_task_refs() -> None:
    with tempfile.TemporaryDirectory() as directory:
        workspace = Path(directory)
        store = StateStore(workspace)
        runtime = AgentRuntime(ScriptedProvider([]), store, workspace)
        try:
            goal = store.create_goal(REQUEST, session_id=runtime.session_id)
            store.update_goal_metadata(goal.id, semantic_goal=_semantic())
            goal = store.get_goal(goal.id)
            plan = store.create_plan(
                goal.id,
                "Build the anchored 3D experience.",
                [
                    {
                        "id": "T001",
                        "title": "Build the 3D calculator",
                        "description": "Implement and verify the accepted experience.",
                        "acceptance_criteria": ["The 3D and visual anchors are observable."],
                        "verification": ["Run browser and rendered-evidence checks."],
                        "requirement_refs": ["R001", "R002"],
                    }
                ],
                applicability_evidence=[
                    {
                        "source": "inspection:I001",
                        "fact": "The workspace was inspected.",
                        "supports_tasks": ["T001"],
                    }
                ],
                execution_strategy="Implement, run, and inspect rendered evidence.",
                expected_changes=[
                    {
                        "path": "main.js",
                        "intent": "Implement the 3D experience.",
                        "supports_tasks": ["T001"],
                    }
                ],
            )

            payload = runtime._state_payload(goal, plan)
            assert payload["accepted_semantic_goal"]["requirement_anchors"][0]["id"] == "R001"
            assert payload["plan"]["tasks"][0]["requirement_refs"] == ["R001", "R002"]
        finally:
            runtime.close()
            store.close()


def test_plan_workspace_exposes_anchor_meaning_before_approval() -> None:
    script = (Path(__file__).parents[1] / "agent" / "web_views" / "static" / "app.js").read_text(
        encoding="utf-8"
    )
    assert "What your words mean in the finished result" in script
    assert "requirement_anchors" in script
