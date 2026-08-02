from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile

import pytest

from agent.capability import (
    CapabilityBand,
    ExecutionStrategyV1,
    InteractionModeV2,
    ModelCapabilityEnvelopeV1,
    TaskDemandV1,
    select_execution_strategy,
)
from agent.chat_runtime import (
    RouteKind,
    SemanticContractError,
    SemanticTurnDecisionV2,
)
from agent.config import RuntimeConfig
from agent.model_catalog import ExecutionClass, ModelDescriptor
from agent.models import GoalStatus
from agent.runtime import AgentRuntime, RuntimeStateError
from agent.sandbox import DockerSandbox, PermissionAdapter
from agent.store import StateStore
from agent.testing import ScriptedProvider


def envelope(
    size: str | None,
    *,
    context: int | None = 32_768,
    execution_class: str = "local",
    **metadata,
) -> ModelCapabilityEnvelopeV1:
    values = dict(metadata)
    if size is not None:
        values["parameter_size"] = size
    if context is not None:
        values["context_window_tokens"] = context
    return ModelCapabilityEnvelopeV1.from_metadata(
        provider="test",
        model="opaque-id",
        execution_class=execution_class,
        capabilities=("tools", "structured_output"),
        metadata=values,
    )


def demand(level: int, *, components: int = 1, parallel: bool = False) -> TaskDemandV1:
    return TaskDemandV1.from_mapping({
        "reasoning": level,
        "implementation": level,
        "context_breadth": level,
        "coordination": level,
        "verification": level,
        "visual_runtime": level,
        "component_count": components,
        "independently_parallelizable": parallel,
        "rationale": ["Scripted relative task demand"],
    })


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        ("1B", CapabilityBand.MINIMAL),
        ("7B", CapabilityBand.LIMITED),
        ("32B", CapabilityBand.STANDARD),
        ("70B", CapabilityBand.HIGH),
    ],
)
def test_parameter_metadata_selects_documented_capability_band(size, expected) -> None:
    value = envelope(size)
    assert value.capability_band is expected
    assert value.parameter_count_billions == float(size[:-1])
    assert value.metadata_complete is True
    assert value.sources["band"] == "parameter_band"


def test_unknown_size_or_context_is_conservatively_minimal() -> None:
    assert envelope(None).capability_band is CapabilityBand.MINIMAL
    assert envelope("70B", context=None).capability_band is CapabilityBand.MINIMAL
    assert envelope(None).metadata_complete is False


def test_raw_parameter_count_is_normalized_to_billions() -> None:
    value = envelope("116829156672")
    assert value.parameter_count_billions == pytest.approx(116.829156672)
    assert value.capability_band is CapabilityBand.HIGH


def test_single_model_authored_rationale_string_repairs_to_one_item_array() -> None:
    value = TaskDemandV1.from_mapping({
        "reasoning": 2,
        "implementation": 2,
        "context_breadth": 2,
        "coordination": 1,
        "verification": 2,
        "visual_runtime": 1,
        "component_count": 1,
        "independently_parallelizable": False,
        "rationale": "A single model-authored reason",
    })
    assert value.rationale == ("A single model-authored reason",)


def test_task_demand_normalizes_descriptive_levels_and_boolean_strings() -> None:
    value = TaskDemandV1.from_mapping({
        "reasoning": "low",
        "implementation": "moderate",
        "context_breadth": "high",
        "coordination": "very_high",
        "verification": "3",
        "visual_runtime": "not applicable",
        "component_count": "2",
        "independently_parallelizable": "yes",
        "rationale": ["Model-authored demand retained"],
    })

    assert value.reasoning == 1
    assert value.implementation == 2
    assert value.context_breadth == 3
    assert value.coordination == 4
    assert value.verification == 3
    assert value.visual_runtime == 1
    assert value.independently_parallelizable is True


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("level 3 (high)", 3),
        ("medium-high", 3),
        ({"level": 4}, 4),
        ({"rating": "moderate"}, 2),
    ],
)
def test_task_demand_normalizes_common_structured_level_shapes(raw, expected) -> None:
    value = demand(1).to_dict()
    value["reasoning"] = raw
    assert TaskDemandV1.from_mapping(value).reasoning == expected


def test_missing_task_demand_level_is_conservatively_maximum() -> None:
    value = demand(1).to_dict()
    value.pop("reasoning")
    assert TaskDemandV1.from_mapping(value).reasoning == 4


def test_explicit_trusted_band_wins_and_moe_uses_active_or_smallest_confirmed_size() -> None:
    explicit = envelope("7B", capability_band="high")
    conservative_moe = envelope("8x7B")
    active_moe = envelope("8x22B", active_parameter_size="13B")
    assert explicit.capability_band is CapabilityBand.HIGH
    assert explicit.sources["band"] == "explicit_metadata"
    assert conservative_moe.parameter_count_billions == 7
    assert active_moe.parameter_count_billions == 13
    assert active_moe.capability_band is CapabilityBand.LIMITED


def test_local_or_cloud_label_does_not_change_strength() -> None:
    local = envelope("7B", execution_class="local")
    cloud = envelope("7B", execution_class="cloud")
    assert local.capability_band is cloud.capability_band is CapabilityBand.LIMITED
    assert local.parameter_count_billions == cloud.parameter_count_billions


def test_same_task_selects_strategy_relative_to_selected_model() -> None:
    task = demand(3, components=3)
    weak = select_execution_strategy(envelope("7B"), task)
    strong = select_execution_strategy(envelope("70B"), task)
    assert weak.strategy is ExecutionStrategyV1.RECURSIVE
    assert weak.max_concurrency == 1
    assert strong.strategy is ExecutionStrategyV1.STAGED


def test_component_limits_and_documented_concurrency_are_enforced() -> None:
    weak = envelope("7B", max_concurrency=8)
    decision = select_execution_strategy(weak, demand(2, components=3, parallel=True))
    assert decision.strategy is ExecutionStrategyV1.RECURSIVE
    assert decision.max_concurrency == 3
    undocumented = select_execution_strategy(
        envelope("7B", execution_class="cloud"),
        demand(3, components=3, parallel=True),
    )
    assert undocumented.max_concurrency == 1


def test_runtime_strategy_limit_cannot_be_widened_by_session_concurrency() -> None:
    with tempfile.TemporaryDirectory() as directory:
        workspace = Path(directory)
        store = StateStore(workspace)
        local_descriptor = ModelDescriptor(
            provider="ollama",
            model="documented-local",
            execution_class=ExecutionClass.LOCAL,
            capabilities=("tools", "structured_output"),
            metadata={
                "parameter_size": "7B",
                "context_window_tokens": 32_768,
                "max_concurrency": 8,
            },
        )
        runtime = AgentRuntime(
            ScriptedProvider([], model="documented-local"),
            store,
            workspace,
            model_descriptor=local_descriptor,
            permission_adapter=PermissionAdapter("normal", DockerSandbox()),
            config=replace(
                RuntimeConfig(),
                ultra_local_concurrency=8,
                repository_index_warmup_files=0,
            ),
        )
        try:
            goal = store.create_goal(
                "Execute one local module at a time",
                session_id=runtime.session_id,
            )
            store.transition_goal(goal.id, GoalStatus.DISCOVERING)
            decision = select_execution_strategy(
                runtime.model_capability_envelope(),
                demand(3, components=3, parallel=False),
            )
            assert decision.max_concurrency == 1
            store.update_goal_metadata(
                goal.id,
                strategy_decision=decision.to_dict(),
                execution_strategy=decision.strategy.value,
            )

            assert runtime._workflow_concurrency_limit() == 1
            session = runtime._make_ultra_session()
            try:
                assert session.config.local_concurrency == 1
                assert session.config.cloud_concurrency == 1
                assert session.config.max_concurrency == 1
            finally:
                session.close()
        finally:
            runtime.close()
            store.close()


def test_recursive_strategy_is_a_one_way_minimum_before_approval() -> None:
    decision = select_execution_strategy(
        envelope("70B"),
        demand(1),
        minimum=ExecutionStrategyV1.RECURSIVE,
    )
    assert decision.strategy is ExecutionStrategyV1.RECURSIVE
    assert decision.lock().locked is True


def test_chat_route_is_never_promoted_because_model_is_weak() -> None:
    original = "Explain how a calculator app works"
    decision = SemanticTurnDecisionV2.from_mapping(
        {
            "route": "chat",
            "outcome_kind": "explanation",
            "interpretation": "Explain the concept without changing the workspace.",
            "requested_effects": {
                "read": False, "write": False, "run": False, "install": False,
                "preview": False, "external_side_effect": False,
            },
            "authority_spans": {
                "read": [], "write": [], "run": [], "install": [],
                "preview": [], "external_side_effect": [],
            },
            "needs_workspace_tools": False,
            "direct_response": "A calculator maps inputs to arithmetic operations.",
            "uncertainty": "clear",
            "clarification_question": "",
            "task_demand": demand(1).to_dict(),
        },
        original_input=original,
    )
    assert decision.route is RouteKind.CHAT
    assert decision.promote_action_to_goal() is decision


def test_chat_route_rejects_an_internally_contradictory_implementation_demand() -> None:
    original = "Create a runnable calculator"
    payload = {
        "route": "chat",
        "outcome_kind": "conversation",
        "interpretation": "The user requests a complete runnable calculator.",
        "requested_effects": [],
        "authority_spans": {},
        "needs_workspace_tools": False,
        "direct_response": "Would you like me to build it?",
        "uncertainty": "clear",
        "clarification_question": "",
        "task_demand": demand(4, components=4).to_dict(),
    }

    with pytest.raises(SemanticContractError, match="implementation.*must be 1 for Chat"):
        SemanticTurnDecisionV2.from_mapping(payload, original_input=original)


def test_public_interaction_mode_has_only_working_and_plan() -> None:
    assert {item.value for item in InteractionModeV2} == {"working", "plan"}
    assert InteractionModeV2.parse("normal") is InteractionModeV2.WORKING
    assert InteractionModeV2.parse("ultra") is InteractionModeV2.WORKING


def descriptor(size: str, *, context: int = 32_768) -> ModelDescriptor:
    return ModelDescriptor(
        provider="ollama",
        model=f"opaque-{size}",
        execution_class=ExecutionClass.LOCAL,
        metadata={
            "parameter_size": size,
            "context_window_tokens": context,
        },
    )


def test_model_change_before_approval_reassesses_but_never_downgrades_depth() -> None:
    with tempfile.TemporaryDirectory() as directory:
        workspace = Path(directory)
        store = StateStore(workspace)
        runtime = AgentRuntime(
            ScriptedProvider([], model="weak"),
            store,
            workspace,
            model_descriptor=descriptor("7B"),
            config=replace(RuntimeConfig(), repository_index_warmup_files=0),
        )
        try:
            goal = store.create_goal("Preserve recursive depth", session_id=runtime.session_id)
            store.transition_goal(goal.id, GoalStatus.DISCOVERING)
            store.transition_goal(goal.id, GoalStatus.AWAITING_PLAN_APPROVAL)
            task = demand(3, components=3)
            original = select_execution_strategy(envelope("7B"), task)
            store.update_goal_metadata(
                goal.id,
                task_demand=task.to_dict(),
                execution_strategy=original.strategy.value,
                strategy_decision=original.to_dict(),
                strategy_locked=False,
                execution_policy={"mode": "ultra", "strategy": "recursive"},
            )
            runtime.replace_provider(
                ScriptedProvider([], model="strong"),
                descriptor("70B"),
            )
            current = store.get_goal(goal.id)
            assert current.metadata["execution_strategy"] == "recursive"
            assert current.metadata["strategy_decision"]["strategy"] == "recursive"
        finally:
            runtime.close()
            store.close()


def test_locked_workflow_rejects_weaker_provider_recovery_model() -> None:
    with tempfile.TemporaryDirectory() as directory:
        workspace = Path(directory)
        store = StateStore(workspace)
        strong_descriptor = descriptor("32B", context=65_536)
        runtime = AgentRuntime(
            ScriptedProvider([], model="approved"),
            store,
            workspace,
            model_descriptor=strong_descriptor,
            config=replace(RuntimeConfig(), repository_index_warmup_files=0),
        )
        try:
            goal = store.create_goal("Locked strategy", session_id=runtime.session_id)
            store.transition_goal(goal.id, GoalStatus.DISCOVERING)
            store.transition_goal(goal.id, GoalStatus.AWAITING_PLAN_APPROVAL)
            store.update_goal_metadata(
                goal.id,
                strategy_locked=True,
                model_capability_envelope=runtime.model_capability_envelope().to_dict(),
                execution_strategy="staged",
            )
            with pytest.raises(RuntimeStateError, match="equal or stronger"):
                runtime.replace_provider(
                    ScriptedProvider([], model="weaker"),
                    descriptor("7B", context=32_768),
                )
            assert runtime.model_descriptor == strong_descriptor
        finally:
            runtime.close()
            store.close()
