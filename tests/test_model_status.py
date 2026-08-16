"""Regression coverage for truthful model lifecycle and failure diagnosis."""

from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from agent.local_provider import (
    ModelCapabilityProfile,
    ProviderDiagnostic,
    ProviderFailureKind,
    ProviderRequestError,
)
from agent.model_catalog import ExecutionClass, ModelDescriptor
from agent.model_status import (
    ModelFailureCategory,
    classify_model_failure,
    model_status_for,
    preflight_model_selection,
)
from agent.cli import _descriptor_for_explicit_model
from agent.providers import AssistantTurn
from agent.runtime import AgentRuntime
from agent.store import StateStore
from agent.testing import ScriptedProvider
from agent.ultra import AgentProtocolError, ArchitectureSpecV1, UltraOrchestrator


class ModelFailureDiagnosisTests(unittest.TestCase):
    def test_cloud_quota_is_an_availability_boundary(self):
        cause = ProviderRequestError(
            ProviderDiagnostic(
                True,
                ProviderFailureKind.HTTP_4XX,
                "POST",
                status_code=429,
                provider_message="session usage limit reached",
            )
        )
        wrapper = RuntimeError("provider unavailable after retries")
        wrapper.__cause__ = cause

        diagnosis = classify_model_failure(wrapper)

        self.assertEqual(diagnosis.category, ModelFailureCategory.QUOTA_EXCEEDED)
        self.assertTrue(diagnosis.provider_boundary)
        self.assertEqual(diagnosis.title("cloud"), "Provider usage limit reached")

    def test_pending_semantic_schema_error_is_not_a_provider_failure(self):
        pending = {
            "stage": "route",
            "last_validation_error": {
                "stage": "route",
                "category": "schema",
                "message": "submit_semantic_route must be called exactly once",
            },
        }

        diagnosis = classify_model_failure(RuntimeError("saved turn failed"), pending)

        self.assertEqual(
            diagnosis.category, ModelFailureCategory.TOOL_CONTRACT_FAILED
        )
        self.assertFalse(diagnosis.provider_boundary)

    def test_architecture_error_wins_over_stale_pending_route_metadata(self):
        pending = {
            "stage": "route",
            "last_validation_error": {
                "stage": "route",
                "category": "provider",
                "message": "old provider boundary",
            },
        }

        diagnosis = classify_model_failure(
            RuntimeError(
                "ArchitectureSpecV1 is missing required fields: summary, components"
            ),
            pending,
        )

        self.assertEqual(
            diagnosis.category, ModelFailureCategory.TYPED_RETURN_FAILED
        )
        self.assertEqual(diagnosis.stage, "architecture")
        self.assertFalse(diagnosis.provider_boundary)
        self.assertNotIn("local model failed", diagnosis.pause_reason("cloud"))


class ModelLifecycleTests(unittest.TestCase):
    def test_selected_probed_model_is_not_claimed_responsive(self):
        descriptor = ModelDescriptor(
            "ollama",
            "gpt-oss:120b-cloud",
            ExecutionClass.CLOUD,
            source="ollama",
            metadata={"ollama_version": "0.20.0"},
        )
        provider = SimpleNamespace(
            model=descriptor.model,
            capability_profile=ModelCapabilityProfile(
                model_name=descriptor.model,
                health_status="reachable",
            ),
        )

        status = model_status_for(provider, descriptor)

        self.assertEqual(status["selection_status"], "selected")
        self.assertEqual(status["inventory_status"], "discovered")
        self.assertEqual(status["probe_status"], "passed")
        self.assertEqual(status["response_status"], "not_run")
        self.assertEqual(status["contract_status"], "not_run")

    def test_ollama_preflight_uses_exact_adapter_handshake(self):
        descriptor = ModelDescriptor(
            "ollama", "gpt-oss:120b-cloud", ExecutionClass.CLOUD
        )
        provider = SimpleNamespace(_ensure_capabilities=mock.Mock())

        preflight_model_selection(provider, descriptor)

        provider._ensure_capabilities.assert_called_once_with()

    def test_validated_ultra_event_updates_shared_model_lifecycle(self):
        provider = ScriptedProvider([], model="gpt-oss:120b-cloud")
        descriptor = ModelDescriptor(
            "ollama",
            provider.model,
            ExecutionClass.CLOUD,
            source="ollama",
            metadata={"ollama_version": "0.20.0"},
        )
        with tempfile.TemporaryDirectory() as directory:
            with StateStore(directory) as store:
                runtime = AgentRuntime(
                    provider,
                    store,
                    Path(directory),
                    model_descriptor=descriptor,
                    session_id="ultra-lifecycle",
                )
                runtime.events.publish(
                    "ultra.agent",
                    "planner returned a valid plan",
                    phase="master_plan",
                )
                status = runtime.model_status_snapshot()
                runtime.close()

        self.assertEqual(status["response_status"], "passed")
        self.assertEqual(status["contract_status"], "verified")
        self.assertEqual(status["contract_stage"], "master_plan")

    def test_explicit_plain_name_is_rejected_with_cloud_alias_suggestion(self):
        cloud = ModelDescriptor(
            "ollama",
            "gpt-oss:120b-cloud",
            ExecutionClass.CLOUD,
            source="ollama",
        )
        catalog = mock.Mock()
        catalog.ollama_host = "http://localhost:11434"
        catalog.discover.return_value = (cloud,)
        catalog.diagnostics = ()

        with self.assertRaisesRegex(ValueError, "gpt-oss:120b-cloud"):
            _descriptor_for_explicit_model(
                "ollama", "gpt-oss:120b", catalog=catalog
            )

    def test_required_native_tool_falls_back_to_constrained_json(self):
        provider = ScriptedProvider(
            [
                AssistantTurn(),
                '{"name":"submit_semantic_route","args":{"route":"action"}}',
            ],
            model="gpt-oss:120b-cloud",
        )
        descriptor = ModelDescriptor(
            "ollama",
            provider.model,
            ExecutionClass.CLOUD,
            source="ollama",
            metadata={"ollama_version": "0.20.0"},
        )
        schema = {
            "type": "function",
            "function": {
                "name": "submit_semantic_route",
                "description": "Return the route",
                "parameters": {
                    "type": "object",
                    "properties": {"route": {"type": "string"}},
                    "required": ["route"],
                },
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            with StateStore(directory) as store:
                runtime = AgentRuntime(
                    provider,
                    store,
                    Path(directory),
                    model_descriptor=descriptor,
                )
                turn = runtime._call_provider(
                    [{"role": "user", "content": "route this"}],
                    [schema],
                    "Return the route",
                    actor="semantic-router",
                    step=1,
                    stream_text=False,
                    require_tool_call=True,
                )

        self.assertEqual([call.name for call in turn.tool_calls], ["submit_semantic_route"])
        self.assertTrue(provider.calls[0].tools)
        self.assertFalse(provider.calls[1].tools)
        self.assertIn("NATIVE TOOL TRANSPORT IS DISABLED", provider.calls[1].system)

    def test_architecture_aliases_are_normalized_without_inventing_content(self):
        payload, actions = UltraOrchestrator._normalize_typed_payload(
            "architecture",
            {
                "architecture_spec": {
                    "overview": "A bounded service architecture",
                    "modules": [{"name": "API", "responsibility": "Serve requests"}],
                }
            },
            {},
        )

        architecture = ArchitectureSpecV1.from_mapping(payload)

        self.assertEqual(architecture.summary, "A bounded service architecture")
        self.assertEqual(architecture.components[0]["name"], "API")
        self.assertIn("architecture.overview normalized to summary", actions)

    def test_architecture_validation_lists_exact_missing_fields(self):
        with self.assertRaisesRegex(
            AgentProtocolError,
            r"summary, components \(a non-empty array of objects\)",
        ):
            ArchitectureSpecV1.from_mapping({})


if __name__ == "__main__":
    unittest.main()
