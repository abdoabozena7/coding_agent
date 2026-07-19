from __future__ import annotations

import sqlite3
import tempfile
import unittest
import json
import os
from pathlib import Path
from unittest import mock

from agent.component_artifacts import (
    ComponentArtifactError,
    ComponentArtifactStore,
)
from agent.events import NullEventBus
from agent.intake import IntentArchitect, PromptSlotStatus
from agent.providers.base import AssistantTurn, ToolCall
from agent.store import StateStore
from agent.tui import ChoiceItem, ChoiceListState
from agent.ultra import AgentRequest, AgentRole, TaskContractV1, UltraOrchestrator, WorkNode
from agent.ultra_session import WorkspaceUltraAgent, _PUBLISH_COMPONENT_TOOL
from agent.visual_judge import (
    VisualJudgeUnavailable,
    create_visual_judge,
    screenshot_anomalies,
)


def component_payload(label: str = "vehicle") -> dict:
    return {
        "implementation": {
            "files": [
                {
                    "path": f"{label}.js",
                    "role": "implementation",
                    "content": (
                        f"export function create{label.title()}(scene) "
                        "{ return { scene, ready: true }; }\n"
                    ),
                },
                {
                    "path": "preview.html",
                    "role": "preview",
                    "content": (
                        "<!doctype html><title>Component preview</title>"
                        f"<main data-component='{label}'>{label}</main>"
                    ),
                },
            ]
        },
        "interface": {
            "exports": [f"create{label.title()}(scene)"],
            "imports": [],
            "integration_points": ["scene graph"],
        },
        "tests": [
            {
                "path": f"{label}.test.js",
                "content": "if (!true) throw new Error('failed');\n",
            }
        ],
        "preview": {"entrypoint": "preview.html"},
    }


class PromptCompletenessV9Tests(unittest.TestCase):
    def test_short_complete_prompt_is_not_questioned_for_length(self) -> None:
        decision = IntentArchitect().analyze(
            "Fix parser.py on the existing platform and run tests."
        )
        self.assertEqual(decision.questions, ())
        self.assertTrue(decision.completeness.complete)

    def test_long_visual_prompt_still_asks_for_missing_packaging(self) -> None:
        prompt = (
            "Build a polished stylized desktop browser Three.js game with responsive "
            "controls, collision, scoring, accessibility, performance checks, and a "
            "complete acceptance suite. Preserve compatibility and test every system. "
        ) * 4
        decision = IntentArchitect().analyze(prompt)
        self.assertIn("packaging", [item.id for item in decision.questions])
        self.assertIs(
            decision.completeness.slot("packaging").status,
            PromptSlotStatus.MISSING_CONSEQUENTIAL,
        )

    def test_simple_threejs_request_asks_platform_packaging_and_style(self) -> None:
        decision = IntentArchitect().analyze(
            "اعمل لي لعبة زي Crossy Road بـThree.js",
            repository_facts=(
                "Cross-run learned lesson: interactive HTML requires browser/runtime evidence.",
            ),
        )
        self.assertEqual(
            [item.id for item in decision.questions],
            ["platform", "packaging", "visual_direction"],
        )


class MaterializedPackageV9Tests(unittest.TestCase):
    def test_leaf_materializes_real_files_manifest_preview_and_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifacts = ComponentArtifactStore(directory)
            package = artifacts.materialize(
                run_id="run",
                node_id="vehicle.wheels",
                component=component_payload("wheels"),
            )
            root = Path(package.root)
            self.assertTrue((root / "wheels.js").is_file())
            self.assertTrue((root / "preview.html").is_file())
            self.assertTrue((root / "component-package.json").is_file())
            self.assertEqual(package.schema_name, "MaterializedComponentPackageV2")
            self.assertTrue(package.content_hash)
            self.assertTrue(package.interface.exports)

    def test_descriptive_package_without_files_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifacts = ComponentArtifactStore(directory)
            with self.assertRaises(ComponentArtifactError):
                artifacts.materialize(
                    run_id="run",
                    node_id="vehicle",
                    component={
                        "implementation": {"summary": "a detailed car"},
                        "interface": {"exports": ["createVehicle(scene)"]},
                        "preview": {"entrypoint": "preview.html"},
                    },
                )

    def test_component_files_can_be_staged_incrementally_before_publish(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifacts = ComponentArtifactStore(directory)
            receipt = artifacts.stage_draft_file(
                run_id="run",
                node_id="environment",
                path="environment.js",
                content="export const environment = true;\n",
                role="implementation",
            )
            self.assertEqual(receipt["role"], "implementation")
            self.assertEqual(
                artifacts.draft_files(run_id="run", node_id="environment")[0]["path"],
                "environment.js",
            )

    def test_assembler_must_consume_every_implementation_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            artifacts = ComponentArtifactStore(workspace)
            package = artifacts.materialize(
                run_id="run",
                node_id="vehicle",
                component=component_payload("vehicle"),
            )
            output = workspace / "index.html"
            output.write_text("<html>unrelated replacement</html>", encoding="utf-8")
            rejected = artifacts.verify_consumption(
                assembler_node_id="final",
                packages=(package,),
                target_paths=(output,),
            )
            self.assertFalse(rejected[0].passed)
            source = (Path(package.root) / "vehicle.js").read_text(encoding="utf-8")
            output.write_text(f"<script type='module'>{source}</script>", encoding="utf-8")
            accepted = artifacts.verify_consumption(
                assembler_node_id="final",
                packages=(package,),
                target_paths=(output,),
            )
            self.assertTrue(accepted[0].passed)

    def test_workforce_templates_cover_ml_backend_and_frontend(self) -> None:
        cases = (
            ("ML churn project", {"data", "model", "training", "evaluation", "serving"}),
            ("Backend appointment booking API", {"domain", "api", "persistence", "auth", "tests"}),
            ("Frontend analytics dashboard", {"layout", "components", "data", "accessibility", "visual_qa"}),
        )
        for title, expected in cases:
            with self.subTest(title=title):
                node = WorkNode(
                    TaskContractV1(
                        id="ROOT",
                        title=title,
                        objective=f"Build a production-ready {title}",
                        acceptance_criteria=("The project works end to end.",),
                        verification=("Run its complete test suite.",),
                        write_paths=("src",),
                    )
                )
                children = UltraOrchestrator._deterministic_cross_domain_children(node)
                self.assertEqual(
                    {str(item["id"]).split(".")[-1] for item in children},
                    expected,
                )
                self.assertTrue(
                    all(item["metadata"]["materialized_components_required"] for item in children)
                )

    def test_oversized_world_leaves_have_recursive_specialists(self) -> None:
        road = WorkNode(
            TaskContractV1(
                id="world.road",
                title="Road specialist",
                objective="Build the complete road system.",
                acceptance_criteria=("Road is independently reviewable.",),
                verification=("Run its preview.",),
                metadata={
                    "component_package_only": True,
                    "component_leaf": True,
                    "specialist_domain": "world.road",
                    "owned_interfaces": ["WorldPackage"],
                },
            )
        )
        children = UltraOrchestrator._deterministic_specialist_children(road)
        self.assertEqual(
            {item["metadata"]["specialist_domain"] for item in children},
            {"world.road.geometry", "world.road.markings", "world.road.collision"},
        )
        self.assertTrue(all(item["metadata"]["component_leaf"] for item in children))

    def test_component_specialist_cannot_finish_without_publish_receipt(self) -> None:
        class Provider:
            reasoning_effort = "medium"

            def __init__(self) -> None:
                self.calls = 0
                self.tool_names: list[set[str]] = []
                self.user_payloads: list[dict] = []
                self.conversations: list[list[dict]] = []

            def call(self, conversation, tools, system):
                del system
                self.calls += 1
                self.user_payloads.append(json.loads(conversation[0]["content"]))
                self.conversations.append(json.loads(json.dumps(conversation)))
                self.tool_names.append(
                    {
                        str(item.get("function", {}).get("name", ""))
                        for item in tools
                    }
                )
                if self.calls == 1:
                    return AssistantTurn(
                        text=json.dumps(
                            {
                                "payload": {"success": True},
                                "summary": "Descriptive claim only",
                            }
                        )
                    )
                if self.calls in {2, 3}:
                    return AssistantTurn(
                        tool_calls=[
                            ToolCall(
                                f"publish-{self.calls - 1}",
                                "publish_component",
                                component_payload("road"),
                            )
                        ]
                    )
                return AssistantTurn(
                    text=json.dumps(
                        {
                            "payload": {"success": True},
                            "summary": "Published",
                        }
                    )
                )

        provider = Provider()

        def executor(call, request):
            del request
            if call.name == "publish_component":
                if not getattr(executor, "attempted", False):
                    executor.attempted = True
                    return json.dumps(
                        {"passed": False, "findings": ["preview entrypoint is missing"]}
                    )
                return json.dumps({"passed": True, "package_id": "pkg"})
            return "unused"

        agent = WorkspaceUltraAgent(
            provider,
            role=AgentRole.CODER,
            provider_name="ollama",
            model="fixture",
            executor=executor,
            events=NullEventBus(),
        )
        response = agent.execute(
            AgentRequest(
                run_id="run",
                role=AgentRole.CODER,
                phase="implement",
                system_prompt="Implement the component.",
                context={},
                task={
                    "contract": {
                        "metadata": {"component_package_only": True},
                        "write_paths": [],
                    }
                },
                node_id="road",
            )
        )
        self.assertEqual(response.summary, "Published")
        self.assertEqual(provider.calls, 4)
        self.assertEqual(provider.reasoning_effort, "off")
        self.assertEqual(provider.max_output_tokens, 2_048)
        self.assertIn("component_task", provider.user_payloads[0])
        self.assertNotIn("harness_reasoning_scaffold", provider.user_payloads[0])
        self.assertLess(len(json.dumps(provider.user_payloads[0])), 5_000)
        replayed_publish = next(
            call
            for message in provider.conversations[2]
            for call in message.get("tool_calls", ())
            if call.get("name") == "publish_component"
        )
        self.assertEqual(replayed_publish["args"], {"rejected_candidate_omitted": True})
        self.assertTrue(all("publish_component" in names for names in provider.tool_names))
        self.assertTrue(all("stage_component_file" in names for names in provider.tool_names))
        self.assertTrue(
            all(
                names.isdisjoint({"write_file", "edit_file", "run_command", "run_bash"})
                for names in provider.tool_names
            )
        )

    def test_contract_derived_mini_plan_preserves_artifact_quality_gates(self) -> None:
        node = WorkNode(
            TaskContractV1(
                id="road",
                title="Road specialist",
                objective="Build reusable lane geometry and collision surfaces.",
                acceptance_criteria=("The road preview is concrete and directly integrable.",),
                verification=("Run the preview and inspect collision evidence.",),
                metadata={"component_package_only": True},
            )
        )
        response = UltraOrchestrator._deterministic_mini_plan(node)
        self.assertEqual(response.payload["source"], "deterministic_contract_fallback")
        joined = " ".join(response.payload["steps"])
        self.assertIn("publish", joined.casefold())
        self.assertIn("independent visual", joined.casefold())

    def test_publish_manifest_requires_preview_entrypoint(self) -> None:
        preview_schema = _PUBLISH_COMPONENT_TOOL["function"]["parameters"]["properties"][
            "preview"
        ]
        self.assertEqual(preview_schema["required"], ["entrypoint"])


class PersistenceAndPickerV9Tests(unittest.TestCase):
    def test_schema_v9_contains_truthful_quality_tables(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(directory)
            store.close()
            connection = sqlite3.connect(Path(directory) / ".coding-agent" / "state.db")
            try:
                self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 9)
                tables = {
                    row[0]
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    )
                }
                self.assertTrue(
                    {
                        "prompt_completeness",
                        "component_files",
                        "interface_contracts",
                        "visual_evaluations",
                        "pairwise_visual_comparisons",
                        "package_consumption_evidence",
                    }.issubset(tables)
                )
            finally:
                connection.close()

    def test_four_row_picker_defaults_to_recommended(self) -> None:
        state = ChoiceListState.create(
            (
                ChoiceItem("1", "Recommended", meta="Recommended"),
                ChoiceItem("2", "Second"),
                ChoiceItem("3", "Third"),
                ChoiceItem("4", "Write your answer"),
            ),
            initial_key="1",
            filterable=False,
        )
        self.assertEqual(state.current.key, "1")
        state.move(3)
        self.assertEqual(state.activate().key, "4")


class TruthfulVisualAuthorityV9Tests(unittest.TestCase):
    def test_builder_cannot_judge_its_own_visual_output(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"AGENT_VISION_PROVIDER": "ollama", "AGENT_VISION_MODEL": "gemma4:e4b"},
            clear=False,
        ):
            judge = create_visual_judge(
                builder_provider="ollama",
                builder_model="gemma4:e4b",
            )
        with self.assertRaises(VisualJudgeUnavailable):
            judge.evaluate()

    def test_near_empty_screenshot_is_anomaly_only_rejection(self) -> None:
        from PIL import Image

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "blank.png"
            Image.new("RGB", (320, 180), "white").save(path)
            self.assertTrue(screenshot_anomalies(path))


if __name__ == "__main__":
    unittest.main()
