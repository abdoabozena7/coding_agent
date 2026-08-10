from __future__ import annotations

from pathlib import Path
import tempfile
from types import SimpleNamespace

import pytest

from agent.commands import CommandKind, parse_command
from agent.control import validate_control_call
from agent.local_provider import (
    ProviderFailureKind,
    ProviderDiagnostic,
    ProviderRequestError,
    normalize_generated_tool_payload,
)
from agent.model_catalog import ExecutionClass, ModelDescriptor
from agent.providers.ollama_provider import OllamaProvider
from agent.runtime import AgentRuntime, ProviderUnavailableError, RuntimeStateError
from agent.semantic import SemanticGoalV2
from agent.store import StateStore
from agent.testing import ScriptedProvider


def test_stop_commands_are_explicit_and_unambiguous() -> None:
    assert parse_command("/stop").kind is CommandKind.STOP
    assert parse_command("/stop").args == {}
    with pytest.raises(ValueError):
        parse_command("/stop ollama")
    with pytest.raises(ValueError):
        parse_command("/stop everything")


def test_harness_canonicalizes_bad_and_duplicate_anchor_ids() -> None:
    request = "Create a Three.js 3D calculator"
    semantic = SemanticGoalV2.from_mapping(
        {
            "original_request": request,
            "interpreted_outcome": "A runnable 3D calculator",
            "requirement_anchors": [
                {"id": "not a valid id", "verbatim_span": "Three.js", "interpreted_requirement": "3D medium", "observable_implications": ["lit geometry"], "kind": "technology"},
                {"id": "not a valid id", "verbatim_span": "calculator", "interpreted_requirement": "calculator behavior", "observable_implications": ["buttons calculate"], "kind": "behavior"},
            ],
        },
        original_request=request,
    )
    assert [item.id for item in semantic.requirement_anchors] == ["R001", "R002"]


def test_anchor_meaning_can_be_recovered_from_model_authored_implications() -> None:
    request = "Create a Three.js calculator"
    semantic = SemanticGoalV2.from_mapping(
        {
            "original_request": request,
            "interpreted_outcome": "A runnable calculator",
            "requirement_anchors": [
                {
                    "verbatim_span": "Three.js",
                    "observable_implications": ["the finished runtime visibly renders a 3D scene"],
                    "kind": "technology",
                }
            ],
        },
        original_request=request,
    )
    assert semantic.requirement_anchors[0].interpreted_requirement.startswith("the finished runtime")


def test_model_lifecycle_status_is_harness_owned() -> None:
    """A legacy ``pending`` echo must not stop an otherwise valid proposal."""

    request = "Create a calculator"
    semantic = SemanticGoalV2.from_mapping(
        {
            "original_request": request,
            "interpreted_outcome": "A runnable calculator",
            "status": "pending",
        },
        original_request=request,
    )
    assert semantic.status == "interpreted"


def test_empty_requested_effect_sentinel_is_ignored() -> None:
    request = "Create a calculator"
    semantic = SemanticGoalV2.from_mapping(
        {
            "original_request": request,
            "interpreted_outcome": "A runnable calculator",
            "requested_effects": ["none", "mutate_workspace", "none"],
        },
        original_request=request,
    )
    assert [item.value for item in semantic.requested_effects] == ["mutate_workspace"]


def test_plan_change_transport_repair_drops_leases_and_maps_task_aliases() -> None:
    normalized, receipt = normalize_generated_tool_payload(
        "propose_plan_change",
        {
            "reason": "The failed mutation needs a narrower repair.",
            "tasks": [
                {
                    "name": "Repair the accepted file",
                    "summary": "Use one accepted path and verify the resulting artifact.",
                    "acceptance": "The artifact is present and verified.",
                    "verification_steps": "Run the accepted check.",
                    "resource_claims": [{"path": "outside-scope.js"}],
                }
            ],
        },
    )
    task = normalized["tasks"][0]
    assert task["title"] == "Repair the accepted file"
    assert task["description"].startswith("Use one accepted path")
    assert task["acceptance_criteria"] == ["The artifact is present and verified."]
    assert task["verification"] == ["Run the accepted check."]
    assert "resource_claims" not in task
    assert any("resource_claims" in action for action in receipt.actions)


def test_control_validator_accepts_the_same_plan_change_repair_directly() -> None:
    value = validate_control_call(
        "propose_plan_change",
        {
            "reason": "Repair the accepted artifact.",
            "tasks": [
                {
                    "name": "Repair artifact",
                    "summary": "Restore the accepted file and verify it.",
                    "acceptance": "The file is correct.",
                    "verification_steps": "Run the accepted check.",
                    "resource_claims": [{"path": "index.html"}],
                }
            ],
        },
    )
    assert value["tasks"][0]["title"] == "Repair artifact"
    assert "resource_claims" not in value["tasks"][0]


def test_legacy_resource_claim_checklist_items_are_harness_bookkeeping() -> None:
    assert AgentRuntime._is_harness_resource_claim_task(
        SimpleNamespace(
            title="Resource claim for index.html",
            description="Obtain an accepted claim before creating the file.",
        )
    )
    assert not AgentRuntime._is_harness_resource_claim_task(
        SimpleNamespace(
            title="Create the index entry point",
            description="Build the requested page.",
        )
    )


def test_transport_error_is_a_single_boundary_not_a_retry_storm() -> None:
    class FailingProvider(ScriptedProvider):
        def __init__(self):
            super().__init__([], model="cloud-test")
            self.calls = 0

        def call(self, *args, **kwargs):
            self.calls += 1
            raise ProviderRequestError(
                ProviderDiagnostic(
                    reachable=False,
                    kind=ProviderFailureKind.DNS_OR_SOCKET,
                    operation="chat",
                    provider_message="DNS failure",
                )
            )

    with tempfile.TemporaryDirectory() as directory:
        workspace = Path(directory)
        store = StateStore(workspace)
        provider = FailingProvider()
        runtime = AgentRuntime(
            provider,
            store,
            workspace,
            model_descriptor=ModelDescriptor("openai", "cloud-test", ExecutionClass.CLOUD),
        )
        try:
            with pytest.raises(ProviderUnavailableError, match="Internet/provider unavailable"):
                runtime._call_provider([], [], "system", actor="semantic-router", step=1)
            assert provider.calls == 1
        finally:
            runtime.close()
            store.close()


def test_local_transport_error_names_the_model_runner_not_the_internet() -> None:
    class FailingProvider(ScriptedProvider):
        def __init__(self):
            super().__init__([], model="local-test")

        def call(self, *args, **kwargs):
            raise ProviderRequestError(
                ProviderDiagnostic(
                    reachable=False,
                    kind=ProviderFailureKind.DNS_OR_SOCKET,
                    operation="chat",
                    provider_message="Ollama is not reachable",
                )
            )

    with tempfile.TemporaryDirectory() as directory:
        workspace = Path(directory)
        store = StateStore(workspace)
        runtime = AgentRuntime(
            FailingProvider(),
            store,
            workspace,
            model_descriptor=ModelDescriptor("ollama", "local-test", ExecutionClass.LOCAL),
        )
        try:
            with pytest.raises(ProviderUnavailableError, match="Local model runner unavailable"):
                runtime._call_provider([], [], "system", actor="semantic-router", step=1)
        finally:
            runtime.close()
            store.close()


def test_stop_ollama_rejects_remote_endpoint() -> None:
    provider = OllamaProvider(model="gemma4:e4b", host="https://example.invalid")
    with tempfile.TemporaryDirectory() as directory:
        workspace = Path(directory)
        store = StateStore(workspace)
        runtime = AgentRuntime(provider, store, workspace)
        try:
            with pytest.raises(RuntimeStateError, match="loopback"):
                runtime.stop_now(shutdown_ollama=True)
        finally:
            runtime.close()
            store.close()
