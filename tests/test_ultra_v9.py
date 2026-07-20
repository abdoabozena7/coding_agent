from __future__ import annotations

import sqlite3
import tempfile
import unittest
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from agent.component_artifacts import (
    ComponentArtifactError,
    ComponentArtifactStore,
)
from agent.events import NullEventBus
from agent.intake import IntentArchitect, PromptSlotStatus
from agent.local_provider import normalize_generated_tool_args
from agent.providers.base import AssistantTurn, ToolCall
from agent.store import StateStore
from agent.tui import ChoiceItem, ChoiceListState
from agent.ultra import (
    AgentRequest,
    AgentRole,
    MasterPlanV1,
    TaskContractV1,
    UltraOrchestrator,
    WorkNode,
)
from agent.ultra_session import (
    StateStoreUltraAdapter,
    UltraSession,
    WorkspaceUltraAgent,
    _PUBLISH_COMPONENT_TOOL,
    _specialist_quality_blueprint,
)
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


class SpecialistQualityBlueprintTests(unittest.TestCase):
    def test_terrain_contract_reserves_road_and_forbids_random_lighting(self) -> None:
        blueprint = "\n".join(
            _specialist_quality_blueprint("world.environment.terrain")
        )

        self.assertIn("central corridor X=-6.2..6.2 completely empty", blueprint)
        self.assertIn("no Math.random", blueprint)
        self.assertIn("never create/replace Scene, Renderer, DOM, camera, lights", blueprint)
        self.assertIn("12-18 readable low-poly props", blueprint)
        self.assertIn("complete syntactically valid file is mandatory under 95 lines", blueprint)

    def test_environment_specialists_own_distinct_non_overlapping_slices(self) -> None:
        props = "\n".join(
            _specialist_quality_blueprint("world.environment.props")
        )
        composition = "\n".join(
            _specialist_quality_blueprint("world.environment.composition")
        )

        self.assertIn("Own only a reusable stylized prop kit", props)
        self.assertIn("broadleaf tree, pine tree", props)
        self.assertIn("never create/replace Scene, Renderer, DOM, camera, lights", props)
        self.assertIn("Own only distant scenic composition", composition)
        self.assertIn("central corridor X=-7..7", composition)
        self.assertIn("Never use identical gray blocks", composition)

    def test_lighting_rig_has_bounded_production_contract(self) -> None:
        rig = "\n".join(_specialist_quality_blueprint("world.lighting.rig"))

        self.assertIn("total intensity stays at or below 2.4", rig)
        self.assertIn("small charcoal ground receiver", rig)
        self.assertIn("Never build a game grid", rig)
        self.assertIn("window.LightingRigAPI", rig)

    def test_chassis_shell_contract_is_small_but_visually_specific(self) -> None:
        shell = "\n".join(
            _specialist_quality_blueprint("vehicles.chassis.shell")
        )

        self.assertIn("facing +Z", shell)
        self.assertIn("width 2.8, length 5.2", shell)
        self.assertIn("8-14 connected meshes", shell)
        self.assertIn("window.ChassisShellAPI", shell)
        self.assertIn("tuple-array loops", shell)
        self.assertIn("55 physical lines", shell)
        self.assertIn("abs(x)<=1.38", shell)
        self.assertIn("hood/front at positive Z", shell)

    def test_chassis_shell_split_profiles_have_absolute_bounded_contracts(self) -> None:
        volumes = "\n".join(
            _specialist_quality_blueprint("vehicles.chassis.shell.volumes")
        )
        panels = "\n".join(
            _specialist_quality_blueprint("vehicles.chassis.shell.panels")
        )

        self.assertIn("[2.8,0.35,5.0]", volumes)
        self.assertIn("window.ChassisVolumesAPI", volumes)
        self.assertIn("every center x must be zero", volumes)
        self.assertIn("window.ChassisPanelsAPI", panels)
        self.assertIn("X=+/-1.38 and Z=+/-1.55", panels)
        self.assertIn("neutral reference body", panels)

    def test_lighting_rig_children_split_lights_from_visual_fixture(self) -> None:
        lights = "\n".join(
            _specialist_quality_blueprint("world.lighting.rig.lights")
        )
        fixture = "\n".join(
            _specialist_quality_blueprint("world.lighting.rig.fixture")
        )

        self.assertIn("Own only one named production light root", lights)
        self.assertIn("total intensity 1.75", lights)
        self.assertIn("create no meshes", lights)
        self.assertIn("Own only a compact neutral material/shadow proof fixture", fixture)
        self.assertIn("create no lights", fixture)
        self.assertIn("six spaced forms", fixture)

    def test_atmosphere_contract_uses_scene_settings_not_sky_mesh(self) -> None:
        atmosphere = "\n".join(
            _specialist_quality_blueprint("world.lighting.atmosphere")
        )

        self.assertIn("scene.background to a THREE.Color", atmosphere)
        self.assertIn("Never model the sky as a PlaneGeometry", atmosphere)
        self.assertIn("window.AtmosphereAPI", atmosphere)

        settings = "\n".join(
            _specialist_quality_blueprint("world.lighting.atmosphere.settings")
        )
        fixture = "\n".join(
            _specialist_quality_blueprint("world.lighting.atmosphere.fixture")
        )
        self.assertIn("create no meshes", settings)
        self.assertIn("exposure 0.9", settings)
        self.assertIn("six varied faceted forms", fixture)
        self.assertIn("avoid identical cubes", fixture)

    def test_non_visual_settings_leaf_can_disable_visual_judging_explicitly(self) -> None:
        node = WorkNode(
            TaskContractV1(
                id="world.lighting.atmosphere.settings",
                title="Atmosphere settings",
                objective="Publish reusable fog and exposure settings.",
                acceptance_criteria=("Settings are deterministic.",),
                verification=("Run contract test.",),
                metadata={
                    "specialist_domain": "world.lighting.atmosphere.settings",
                    "visual_required": False,
                },
            )
        )

        self.assertFalse(StateStoreUltraAdapter._requires_visual_component(node))


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
    def test_component_review_missing_boolean_uses_typed_evidence_not_prose(self) -> None:
        task = {
            "contract": {
                "metadata": {"component_package_only": True},
            }
        }
        normalized, actions = UltraOrchestrator._normalize_typed_payload(
            "review",
            {
                "findings": [],
                "issues": [],
                "evidence": [{"path": "preview/scene.js", "sha256": "abc"}],
            },
            task,
        )
        self.assertTrue(normalized["passed"])
        self.assertTrue(any("typed component evidence" in item for item in actions))

        empty, _ = UltraOrchestrator._normalize_typed_payload(
            "review",
            {"findings": [], "issues": [], "evidence": []},
            task,
        )
        self.assertFalse(empty["passed"])

    def test_weak_model_nested_component_file_arguments_are_normalized(self) -> None:
        normalized = normalize_generated_tool_args(
            "stage_component_file",
            {
                "file": {
                    "file_path": "",
                    "source": "<!doctype html><html><canvas></canvas></html>",
                }
            },
        )
        self.assertEqual(normalized["path"], "preview/index.html")
        self.assertEqual(normalized["role"], "preview")
        self.assertIn("<canvas>", normalized["content"])

    def test_weak_model_publish_aliases_are_normalized(self) -> None:
        normalized = normalize_generated_tool_args(
            "publish_component",
            {
                "exports": ["createRoad(scene)"],
                "preview_entrypoint": "preview/index.html",
            },
        )
        self.assertEqual(normalized["interface"]["exports"], ["createRoad(scene)"])
        self.assertEqual(
            normalized["preview"]["entrypoint"],
            "preview/index.html",
        )

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

    def test_component_preview_does_not_fail_on_implicit_favicon_request(self) -> None:
        capability = __import__(
            "agent.tools.web_preview",
            fromlist=["browser_capability"],
        ).browser_capability()
        if not capability["available"] or not capability["playwright"]:
            self.skipTest("local Playwright browser is unavailable")
        with tempfile.TemporaryDirectory() as directory:
            artifacts = ComponentArtifactStore(directory)
            package = artifacts.materialize(
                run_id="run",
                node_id="component",
                component=component_payload("component"),
            )
            result = artifacts.verify_preview(package, settle_ms=50)
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["network_errors"], [])

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

    def test_javascript_renamed_to_html_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifacts = ComponentArtifactStore(directory)
            payload = component_payload("road")
            payload["implementation"]["files"][1]["content"] = (
                "import { Road } from '../src/index.js';\n"
                "const road = new Road();\n"
            )
            with self.assertRaisesRegex(
                ComponentArtifactError,
                "must contain a real HTML document",
            ):
                artifacts.materialize(
                    run_id="run",
                    node_id="world.road.geometry",
                    component=payload,
                )

    def test_typescript_inside_html_preview_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifacts = ComponentArtifactStore(directory)
            payload = component_payload("road")
            payload["implementation"]["files"][1]["content"] = (
                "<!doctype html><html><body><canvas id='road'></canvas>"
                "<script>const canvas = document.querySelector('canvas') "
                "as HTMLCanvasElement;</script></body></html>"
            )
            with self.assertRaisesRegex(
                ComponentArtifactError,
                "contains TypeScript syntax",
            ):
                artifacts.materialize(
                    run_id="run",
                    node_id="world.road.geometry",
                    component=payload,
                )

    def test_visual_specialist_contract_rejects_documentation_proxy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            preview = Path(directory) / "preview.html"
            preview.write_text(
                "<!doctype html><h1>Road Geometry</h1>"
                "<p>Generated Cross-Section (Visualization Proxy)</p>"
                "<pre>API Demonstration</pre>",
                encoding="utf-8",
            )
            node = WorkNode(
                TaskContractV1(
                    id="world.road.geometry",
                    title="Road geometry",
                    objective="Build actual road geometry.",
                    acceptance_criteria=("The road is visibly modeled.",),
                    verification=("Inspect the isolated preview.",),
                    metadata={
                        "specialist_domain": "world.road.geometry",
                        "component_package_only": True,
                    },
                )
            )
            package = type(
                "Package",
                (),
                {"root": directory, "preview_entrypoint": "preview.html"},
            )()
            adapter = object.__new__(StateStoreUltraAdapter)
            with self.assertRaisesRegex(
                ComponentArtifactError,
                "not a materialized 3D specialist artifact",
            ):
                adapter._assert_domain_preview(node, package)

    def test_visual_specialist_accepts_threejs_created_renderer_canvas(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            preview = Path(directory) / "preview.html"
            preview.write_text(
                "<!doctype html><script type='module'>"
                "import * as THREE from "
                "'https://unpkg.com/three/build/three.module.js';"
                "const renderer = new THREE.WebGLRenderer({antialias:true});"
                "document.body.appendChild(renderer.domElement);"
                "</script>",
                encoding="utf-8",
            )
            node = WorkNode(
                TaskContractV1(
                    id="vehicles.wheels.rim",
                    title="Wheel rim",
                    objective="Model the wheel rim.",
                    acceptance_criteria=("The rim is visibly modeled.",),
                    verification=("Inspect its Three.js preview.",),
                    metadata={"specialist_domain": "vehicles.wheels.rim"},
                )
            )
            package = type(
                "Package",
                (),
                {"root": directory, "preview_entrypoint": "preview.html"},
            )()
            adapter = object.__new__(StateStoreUltraAdapter)
            adapter._assert_domain_preview(node, package)

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

    def test_threejs_leaf_cannot_replace_harness_scene_renderer_or_dom(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "preview").mkdir()
            (root / "preview" / "index.html").write_text(
                "<canvas></canvas><script src='three.min.js'></script>"
                "<script>new THREE.Scene()</script>",
                encoding="utf-8",
            )
            (root / "preview" / "scene.js").write_text(
                "window.buildPreview=()=>{"
                "const scene=new THREE.Scene();"
                "const renderer=new THREE.WebGLRenderer();"
                "document.body.appendChild(renderer.domElement);"
                "requestAnimationFrame(()=>{});"
                "};",
                encoding="utf-8",
            )
            node = WorkNode(
                TaskContractV1(
                    id="world.road.geometry",
                    title="Road geometry",
                    objective="Build the isolated road geometry.",
                    acceptance_criteria=("Preview uses harness objects.",),
                    verification=("Run preview.",),
                    metadata={"specialist_domain": "world.road.geometry"},
                )
            )
            package = SimpleNamespace(
                root=root,
                preview_entrypoint="preview/index.html",
                files=(SimpleNamespace(path="preview/scene.js"),),
            )
            adapter = object.__new__(StateStoreUltraAdapter)
            with self.assertRaisesRegex(
                ComponentArtifactError,
                "specialist-created Scene",
            ):
                adapter._assert_domain_preview(node, package)

    def test_typed_chassis_root_is_valid_entrypoint_without_build_preview(self) -> None:
        cases = (
            ("vehicles.chassis.shell.volumes", "window.ChassisVolumesAPI={root,forward:'+Z'};"),
            ("vehicles.chassis.shell.panels", "window.ChassisPanelsAPI={root,forward:'+Z',wheelMounts:[{},{},{},{}]};"),
            ("vehicles.chassis.shell", "window.ChassisShellAPI={root,forward:'+Z',wheelMounts:[{},{},{},{}]};"),
        )
        for domain, export_source in cases:
            with self.subTest(domain=domain), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "preview").mkdir()
                (root / "preview" / "index.html").write_text(
                    "<canvas></canvas><script src='three.min.js'></script>",
                    encoding="utf-8",
                )
                (root / "preview" / "scene.js").write_text(
                    "const root=new THREE.Group();" + export_source,
                    encoding="utf-8",
                )
                node = WorkNode(
                    TaskContractV1(
                        id=domain,
                        title="Typed chassis component",
                        objective="Build a typed chassis root.",
                        acceptance_criteria=("Typed root is materialized.",),
                        verification=("Run preview.",),
                        metadata={"specialist_domain": domain},
                    )
                )
                package = SimpleNamespace(
                    root=root,
                    preview_entrypoint="preview/index.html",
                    files=(SimpleNamespace(path="preview/scene.js"),),
                )

                adapter = object.__new__(StateStoreUltraAdapter)
                adapter._assert_domain_preview(node, package)

    def test_terrain_leaf_rejects_random_layout_before_browser_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "preview").mkdir()
            (root / "preview" / "index.html").write_text(
                "<canvas></canvas><script src='three.min.js'></script>",
                encoding="utf-8",
            )
            (root / "preview" / "scene.js").write_text(
                (
                    "window.buildPreview=({THREE,scene})=>{const root=new THREE.Group();"
                    "root.position.x=Math.random();scene.add(root);"
                    "window.TerrainAPI={root,corridorHalfWidth:6.2,"
                    "extents:{minZ:-26,maxZ:26}};};"
                ),
                encoding="utf-8",
            )
            node = WorkNode(
                TaskContractV1(
                    id="world.environment.terrain",
                    title="Terrain",
                    objective="Build deterministic road-side terrain.",
                    acceptance_criteria=("Keep the road corridor clear.",),
                    verification=("Run preview.",),
                    metadata={"specialist_domain": "world.environment.terrain"},
                )
            )
            package = SimpleNamespace(
                root=root,
                preview_entrypoint="preview/index.html",
                files=(SimpleNamespace(path="preview/scene.js"),),
            )
            adapter = object.__new__(StateStoreUltraAdapter)

            with self.assertRaisesRegex(ComponentArtifactError, "deterministic authored transforms"):
                adapter._assert_domain_preview(node, package)

    def test_component_reviewer_reads_package_relative_paths_from_isolated_staging(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            artifacts = ComponentArtifactStore(workspace)
            package = artifacts.materialize(
                run_id="run",
                node_id="api.persistence",
                component=component_payload("repository"),
            )
            node = WorkNode(
                TaskContractV1(
                    id="api.persistence",
                    title="Persistence specialist",
                    objective="Implement the repository boundary.",
                    acceptance_criteria=("Repository is independently testable.",),
                    verification=("Review its isolated package.",),
                    metadata={"component_package_only": True},
                )
            )
            session = object.__new__(UltraSession)
            session.workspace = workspace
            session.adapter = SimpleNamespace(
                run_id="run",
                component_artifacts=artifacts,
            )
            request = AgentRequest(
                run_id="run",
                role=AgentRole.CLEAN_CODE_REVIEWER,
                phase="review",
                system_prompt="Review exact package bytes.",
                context={},
                task={
                    "contract": {"metadata": {"component_package_only": True}},
                    "candidate": {
                        "payload": {
                            "materialized_component_package": package.to_dict(
                                include_content=True
                            )
                        }
                    },
                },
                node_id=node.id,
            )
            resolved = session._component_read_path(
                request,
                node,
                "api.persistence/repository.js",
            )
            self.assertIsNotNone(resolved)
            self.assertTrue(str(resolved).endswith("repository.js"))
            self.assertTrue((workspace / str(resolved)).is_file())

    def test_repository_preservation_rejects_untracked_or_out_of_scope_changes(self) -> None:
        unexpected, escaped = StateStoreUltraAdapter._repository_preservation_delta(
            baseline_hashes={
                "src/domain.py": "old-domain",
                "src/unrelated.py": "old-unrelated",
            },
            current_hashes={
                "src/domain.py": "new-domain",
                "src/unrelated.py": "corrupted",
                "secrets.txt": "new",
            },
            tracked_paths=("src/domain.py", "secrets.txt"),
            approved_scopes=("src/domain.py",),
        )
        self.assertEqual(unexpected, ("src/unrelated.py",))
        self.assertEqual(escaped, ("secrets.txt",))

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
            ("Backend appointment booking API", {"domain", "api", "persistence", "auth", "operations", "tests"}),
            ("Frontend analytics dashboard", {"layout", "components", "data", "accessibility", "quality", "visual_qa"}),
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

    def test_minimal_game_prompt_infers_logic_and_spatial_concerns(self) -> None:
        matrix = UltraOrchestrator._concern_coverage_matrix(
            "اعمل لي لعبة 3D بـThree.js"
        )
        self.assertEqual(matrix.task_family, "interactive_game")
        self.assertTrue(
            {
                "spatial_semantics",
                "gameplay_state",
                "progression_pacing",
                "world_scale",
                "runtime_performance",
            }.issubset({item.id for item in matrix.concerns})
        )

        node = WorkNode(
            TaskContractV1(
                id="ROOT",
                title="Final game assembler",
                objective="اعمل لي لعبة 3D بـThree.js",
                acceptance_criteria=("The game is playable.",),
                verification=("Play a complete session.",),
                write_paths=("index.html",),
                metadata={"force_recursive_specialists": True},
            )
        )
        children = UltraOrchestrator._deterministic_shared_artifact_children(node)
        owned = {
            concern
            for child in children
            for concern in child["metadata"]["concern_ids"]
        }
        self.assertFalse(matrix.missing_critical_owners(children))
        self.assertIn("spatial_semantics", owned)
        self.assertIn("progression_pacing", owned)

        character = next(child for child in children if child["id"].endswith(".character"))
        character_node = WorkNode(TaskContractV1.from_mapping(character))
        character_children = UltraOrchestrator._deterministic_specialist_children(character_node)
        controls = next(
            child for child in character_children if child["id"].endswith(".controls")
        )
        movement_parent = WorkNode(TaskContractV1.from_mapping(controls))
        movement = next(
            child
            for child in UltraOrchestrator._deterministic_specialist_children(movement_parent)
            if child["id"].endswith(".movement")
        )
        self.assertIn("spatial_semantics", movement["metadata"]["concern_ids"])
        self.assertTrue(
            any("faces and animates" in item for item in movement["acceptance_criteria"])
        )

    def test_concern_coverage_is_domain_specific_not_game_hard_coded(self) -> None:
        cases = {
            "Backend appointment booking API": {
                "security_boundaries",
                "data_integrity",
                "concurrency_idempotency",
                "operability",
            },
            "Frontend analytics dashboard": {
                "ui_state_integrity",
                "frontend_accessibility",
                "frontend_security",
                "frontend_performance",
            },
            "ML churn project": {
                "data_leakage",
                "ml_reproducibility",
                "evaluation_validity",
                "ml_serving_reliability",
            },
        }
        for prompt, required in cases.items():
            with self.subTest(prompt=prompt):
                matrix = UltraOrchestrator._concern_coverage_matrix(prompt)
                self.assertTrue(required.issubset({item.id for item in matrix.concerns}))
                node = WorkNode(
                    TaskContractV1(
                        id="ROOT",
                        title=prompt,
                        objective=f"Build {prompt}",
                        acceptance_criteria=("Works end to end.",),
                        verification=("Run the complete checks.",),
                        write_paths=("src",),
                    )
                )
                children = UltraOrchestrator._deterministic_cross_domain_children(node)
                self.assertFalse(matrix.missing_critical_owners(children))

    def test_frontend_visual_prompt_is_not_rewritten_as_vehicle_game(self) -> None:
        self.assertFalse(
            UltraOrchestrator._requires_game_artifact("Frontend analytics dashboard")
        )
        self.assertTrue(
            UltraOrchestrator._requires_visual_artifact("Frontend analytics dashboard")
        )

    def test_existing_master_modules_receive_concerns_without_duplicate_swarms(self) -> None:
        modules = tuple(
            TaskContractV1(
                id=f"M{index}",
                title=title,
                objective=f"Implement {title}",
                acceptance_criteria=("Works.",),
                verification=("Test it.",),
                write_paths=("src",),
            )
            for index, title in enumerate(
                ("Domain", "API", "Persistence", "Authentication", "Operations", "Tests"),
                start=1,
            )
        )
        plan = MasterPlanV1(summary="Backend plan", modules=modules)
        enriched = UltraOrchestrator._enforce_concern_coverage_contract(
            "Backend appointment booking API",
            plan,
        )
        owned = {
            concern
            for module in enriched.modules
            for concern in module.metadata["concern_ids"]
        }
        matrix = UltraOrchestrator._concern_coverage_matrix(
            "Backend appointment booking API"
        )
        self.assertTrue(set(matrix.critical_ids).issubset(owned))
        self.assertTrue(
            all(module.metadata["cross_domain_template_root"] is False for module in enriched.modules)
        )
        self.assertTrue(
            all(
                not UltraOrchestrator._deterministic_cross_domain_children(WorkNode(module))
                for module in enriched.modules
            )
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

    def test_repeated_chassis_shell_plateau_splits_into_two_bounded_leaves(self) -> None:
        shell = WorkNode(
            TaskContractV1(
                id="vehicles.chassis.shell",
                title="Shell specialist",
                objective="Build a complete body shell.",
                acceptance_criteria=("Shell is independently reviewable.",),
                verification=("Run its preview.",),
                metadata={
                    "component_package_only": True,
                    "component_leaf": True,
                    "specialist_domain": "vehicles.chassis.shell",
                    "owned_interfaces": ["VehiclePackage"],
                },
            )
        )

        children = UltraOrchestrator._deterministic_specialist_children(shell)

        self.assertEqual(
            {item["metadata"]["specialist_domain"] for item in children},
            {
                "vehicles.chassis.shell.volumes",
                "vehicles.chassis.shell.panels",
            },
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
                try:
                    payload = json.loads(conversation[0]["content"])
                except json.JSONDecodeError:
                    payload = {"raw": conversation[0]["content"]}
                self.user_payloads.append(payload)
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
                if self.calls == 2:
                    return AssistantTurn(
                        tool_calls=[
                            ToolCall(
                                "stage-preview",
                                "stage_component_file",
                                {
                                    "path": "road.js",
                                    "content": "window.buildPreview=()=>{};",
                                    "role": "preview",
                                },
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
                executor.publish_calls = getattr(executor, "publish_calls", 0) + 1
                return json.dumps(
                    {
                        "passed": True,
                        "status": "provisional_codex_review",
                        "package_id": "pkg",
                        "preview": {
                            "status": "passed",
                            "screenshot_path": "preview.png",
                        },
                    }
                )
            if call.name == "stage_component_file":
                executor.stage_paths = [
                    *getattr(executor, "stage_paths", []),
                    call.args["path"],
                ]
                return json.dumps({"passed": True, "path": call.args["path"]})
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
        self.assertIn("runtime verified", response.summary)
        self.assertTrue(response.payload["success"])
        self.assertEqual(provider.calls, 2)
        self.assertEqual(executor.publish_calls, 1)
        self.assertEqual(executor.stage_paths[0], "preview/scene.js")
        self.assertEqual(provider.reasoning_effort, "off")
        self.assertEqual(provider.max_output_tokens, 2_048)
        self.assertEqual(provider.temperature, 0.25)
        self.assertIn("component_task", provider.user_payloads[0])
        self.assertNotIn("harness_reasoning_scaffold", provider.user_payloads[0])
        self.assertLess(len(json.dumps(provider.user_payloads[0])), 5_000)
        self.assertNotIn("publish_component", provider.tool_names[0])
        self.assertNotIn("publish_component", provider.tool_names[1])
        self.assertTrue(all("stage_component_file" in names for names in provider.tool_names))
        self.assertTrue(
            all(
                names.isdisjoint({"write_file", "edit_file", "run_command", "run_bash"})
                for names in provider.tool_names
            )
        )

    def test_rejected_component_ends_weak_model_turn_after_one_candidate(self) -> None:
        class Provider:
            reasoning_effort = "medium"

            def __init__(self) -> None:
                self.calls = 0

            def call(self, conversation, tools, system):
                del conversation, tools, system
                self.calls += 1
                return AssistantTurn(
                    tool_calls=[
                        ToolCall(
                            "stage-preview",
                            "stage_component_file",
                            {
                                "path": "preview/scene.js",
                                "content": "window.buildPreview=()=>{};",
                                "role": "preview",
                            },
                        )
                    ]
                )

        provider = Provider()

        def executor(call, request):
            del request
            if call.name == "stage_component_file":
                return json.dumps({"passed": True, "path": call.args["path"]})
            if call.name == "publish_component":
                return json.dumps(
                    {
                        "passed": False,
                        "status": "rejected",
                        "findings": ["typed runtime assertion failed"],
                    }
                )
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

        self.assertEqual(provider.calls, 1)
        self.assertFalse(response.payload["success"])
        self.assertEqual(response.payload["findings"], ["typed runtime assertion failed"])
        self.assertIn("fresh challenger", response.summary)

    def test_parent_component_assembles_exact_children_without_model_prompt(self) -> None:
        class Provider:
            reasoning_effort = "medium"

            def call(self, conversation, tools, system):
                del conversation, tools, system
                self.fail("parent assembler must not call the local model")

        calls: list[ToolCall] = []

        def executor(call, request):
            del request
            calls.append(call)
            if call.name == "stage_component_file":
                return json.dumps({"status": "staged", "path": call.args["path"]})
            if call.name == "publish_component":
                return json.dumps(
                    {
                        "passed": True,
                        "status": "provisional_codex_review",
                        "stored_package_id": "parent-pkg",
                        "preview": {"screenshot_path": "parent.png"},
                    }
                )
            return "unused"

        provider = Provider()
        provider.fail = self.fail
        agent = WorkspaceUltraAgent(
            provider,
            role=AgentRole.INTEGRATOR,
            provider_name="ollama",
            model="fixture",
            executor=executor,
            events=NullEventBus(),
        )
        children = {
            "vehicles.chassis.shell.volumes": {
                "id": "pkg-volumes",
                "content_hash": "hash-volumes",
                "file_contents": {
                    "preview/scene.js": (
                        "const root=new THREE.Group();"
                        "window.ChassisVolumesAPI={root,forward:'+Z'};"
                    )
                },
            },
            "vehicles.chassis.shell.panels": {
                "id": "pkg-panels",
                "content_hash": "hash-panels",
                "file_contents": {
                    "preview/scene.js": (
                        "const root=new THREE.Group();"
                        "window.ChassisPanelsAPI={root,forward:'+Z',wheelMounts:[{},{},{},{}]};"
                    )
                },
            },
        }
        response = agent.execute(
            AgentRequest(
                run_id="run",
                role=AgentRole.INTEGRATOR,
                phase="integrate",
                system_prompt="Integrate exact child packages.",
                context={},
                task={
                    "contract": {
                        "owned_interfaces": ["VehiclePackage"],
                        "metadata": {
                            "component_package_only": True,
                            "specialist_domain": "vehicles.chassis.shell",
                        },
                    },
                    "component_assembler": True,
                    "child_component_packages": children,
                },
                node_id="vehicles.chassis.shell",
            )
        )

        self.assertTrue(response.payload["success"])
        self.assertEqual(response.model, "exact-child-assembler-v1")
        scene_call = next(
            call
            for call in calls
            if call.name == "stage_component_file"
            and call.args["path"] == "preview/scene.js"
        )
        self.assertIn("hash-volumes", scene_call.args["content"])
        self.assertIn("hash-panels", scene_call.args["content"])
        self.assertIn("__componentConsumption", scene_call.args["content"])
        self.assertIn('const typedApi=childWindow["ChassisVolumesAPI"]', scene_call.args["content"])
        self.assertIn('const typedApi=childWindow["ChassisPanelsAPI"]', scene_call.args["content"])
        self.assertIn("window.ChassisShellAPI", scene_call.args["content"])
        publish_call = next(call for call in calls if call.name == "publish_component")
        self.assertIn("ChassisShellAPI", publish_call.args["interface"]["exports"])
        self.assertEqual(
            [item["package_id"] for item in response.payload["package_consumption"]],
            ["pkg-panels", "pkg-volumes"],
        )

    def test_chassis_shell_assembler_rejects_a_missing_typed_child(self) -> None:
        class Provider:
            reasoning_effort = "medium"

            def call(self, conversation, tools, system):
                del conversation, tools, system
                self.fail("invalid child set must fail before a model call")

        provider = Provider()
        provider.fail = self.fail

        def executor(call, request):
            del call, request
            self.fail("invalid child set must fail before staging")

        agent = WorkspaceUltraAgent(
            provider,
            role=AgentRole.INTEGRATOR,
            provider_name="ollama",
            model="fixture",
            executor=executor,
            events=NullEventBus(),
        )
        response = agent.execute(
            AgentRequest(
                run_id="run",
                role=AgentRole.INTEGRATOR,
                phase="integrate",
                system_prompt="Integrate exact child packages.",
                context={},
                task={
                    "contract": {
                        "metadata": {
                            "component_package_only": True,
                            "specialist_domain": "vehicles.chassis.shell",
                        }
                    },
                    "component_assembler": True,
                    "child_component_packages": {
                        "vehicles.chassis.shell.volumes": {
                            "id": "pkg-volumes",
                            "content_hash": "hash-volumes",
                            "file_contents": {
                                "preview/scene.js": (
                                    "const root=new THREE.Group();"
                                    "window.ChassisVolumesAPI={root,forward:'+Z'};"
                                )
                            },
                        }
                    },
                },
                node_id="vehicles.chassis.shell",
            )
        )

        self.assertFalse(response.payload["success"])
        self.assertIn("exactly its volumes and panels", response.payload["findings"][0])

    def test_component_review_uses_bounded_durable_projection_without_tools(self) -> None:
        class Provider:
            reasoning_effort = "medium"

            def __init__(self) -> None:
                self.tools = None
                self.payload = None

            def call(self, conversation, tools, system):
                self.tools = list(tools)
                self.payload = json.loads(conversation[0]["content"])
                self.asserted_system = system
                return AssistantTurn(
                    text=json.dumps(
                        {
                            "payload": {
                                "passed": True,
                                "issues": [],
                                "findings": [],
                                "evidence": [
                                    {
                                        "path": "preview/scene.js",
                                        "observation": "bounded source reviewed",
                                    }
                                ],
                            },
                            "summary": "No observable clean-code blocker.",
                            "reasoning_summary": "Hash-backed source projection reviewed.",
                        }
                    )
                )

        provider = Provider()
        agent = WorkspaceUltraAgent(
            provider,
            role=AgentRole.CLEAN_CODE_REVIEWER,
            provider_name="ollama",
            model="fixture",
            executor=lambda call, request: self.fail(
                f"component reviewer unexpectedly called {call.name}"
            ),
            events=NullEventBus(),
        )
        response = agent.execute(
            AgentRequest(
                run_id="run",
                role=AgentRole.CLEAN_CODE_REVIEWER,
                phase="review",
                system_prompt="Review component.",
                context={"oversized": "x" * 50_000},
                task={
                    "contract": {
                        "objective": "Review the isolated road geometry.",
                        "acceptance_criteria": ["No placeholder implementation."],
                        "verification": ["Runtime preview passed."],
                        "metadata": {"component_package_only": True},
                    },
                    "candidate": {
                        "payload": {
                            "component_publication": {
                                "package_id": "pkg-1",
                                "status": "provisional_codex_review",
                                "screenshot_path": "evidence/preview.png",
                            },
                            "materialized_preview": {
                                "status": "passed",
                                "console_errors": [],
                                "page_errors": [],
                            },
                            "materialized_component_package": {
                                "id": "pkg-1",
                                "content_hash": "abc",
                                "root": "staging/component",
                                "preview_entrypoint": "preview/index.html",
                                "file_contents": {
                                    "preview/scene.js": "export const scene = 1;\n"
                                    + "const detail = true;\n" * 1_000,
                                    "test/scene.test.js": "if (!true) throw Error();\n",
                                },
                            },
                        }
                    },
                },
                node_id="world.road.geometry",
            )
        )
        self.assertTrue(response.payload["passed"])
        self.assertEqual(provider.tools, [])
        projection = provider.payload["durable_memory_projection"]
        self.assertLessEqual(sum(len(item["excerpt"]) for item in projection), 6_100)
        self.assertEqual(projection[0]["path"], "preview/scene.js")
        self.assertNotIn("oversized", provider.payload)
        self.assertIn("durable memory", provider.asserted_system)
        self.assertEqual(provider.max_output_tokens, 640)
        self.assertEqual(
            response.payload["harness_reasoning_synthesized"],
            "observable_component_review_evidence",
        )
        self.assertTrue(response.payload["harness_reasoning_evaluation"]["passed"])
        self.assertIn(
            "component-package-sha256:abc",
            response.payload["reasoning_artifact"]["supporting_evidence"],
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

    def test_component_preview_preflight_rejects_missing_local_script(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifacts = ComponentArtifactStore(directory)
            payload = component_payload("road")
            payload["implementation"]["files"][1]["content"] = (
                "<!doctype html><script type='module' src='./missing.js'></script>"
            )
            with self.assertRaisesRegex(
                ComponentArtifactError,
                "missing local preview reference",
            ):
                artifacts.materialize(
                    run_id="run",
                    node_id="road",
                    component=payload,
                )

    def test_component_preview_preflight_rejects_unmapped_bare_import(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifacts = ComponentArtifactStore(directory)
            payload = component_payload("road")
            payload["implementation"]["files"][1]["content"] = (
                "<!doctype html><script type='module'>"
                "import * as THREE from 'three';"
                "</script>"
            )
            with self.assertRaisesRegex(ComponentArtifactError, "bare browser import"):
                artifacts.materialize(
                    run_id="run",
                    node_id="road",
                    component=payload,
                )

    def test_harness_generated_threejs_preview_calls_specialist_scene_contract(self) -> None:
        source = (
            "import * as THREE from 'three';\n"
            "window.buildPreview=({THREE,scene})=>{"
            "scene.add(new THREE.Mesh(new THREE.BoxGeometry(3,.2,12),"
            "new THREE.MeshStandardMaterial({color:0x333333})));};"
        )
        html = StateStoreUltraAdapter._generated_threejs_preview(source)
        self.assertNotIn("import * as THREE", html)
        self.assertIn("window.buildPreview=({THREE,scene})", html)
        self.assertIn("window.buildPreview(previewContext)", html)
        self.assertIn("window.buildPreview(THREE,scene,camera,renderer)", html)
        self.assertIn("window.buildPreview.length >= 2", html)
        self.assertIn("Object.assign(Object.create(scene)", html)
        self.assertIn("requestAnimationFrame(frame)", html)
        self.assertIn("camera.lookAt(0,0,0)", html)

    def test_harness_preview_normalizes_esm_exports_and_scene_only_signature(self) -> None:
        source = (
            "export function buildPreview(scene) {"
            "scene.add(new THREE.Mesh(new THREE.BoxGeometry(1,1,1),"
            "new THREE.MeshStandardMaterial()));"
            "}\nexport { buildPreview };\n"
        )
        html = StateStoreUltraAdapter._generated_threejs_preview(source)
        self.assertNotIn("export function", html)
        self.assertNotIn("export { buildPreview }", html)
        self.assertIn('typeof window.buildPreview !== "function"', html)
        self.assertIn("window.buildPreview = buildPreview", html)

    def test_markings_preview_supplies_non_owned_reference_road(self) -> None:
        html = StateStoreUltraAdapter._generated_threejs_preview(
            "window.buildPreview=({THREE,scene})=>{};",
            node_id="root.world.road.markings",
        )

        self.assertIn("__harness_reference_road_not_component_output", html)
        self.assertIn("referenceAsphalt", html)
        self.assertLess(
            html.index("__harness_reference_road_not_component_output"),
            html.index("window.buildPreview(previewContext)"),
        )

    def test_collision_preview_runs_typed_api_assertions(self) -> None:
        html = StateStoreUltraAdapter._generated_threejs_preview(
            (
                "window.buildPreview=({THREE,scene})=>{"
                "window.RoadCollisionAPI={"
                "overlapsAABB:(a,b)=>a.maxX>=b.minX&&a.minX<=b.maxX"
                "&&a.maxZ>=b.minZ&&a.minZ<=b.maxZ,"
                "isInsideRoad:(p,hw,hd)=>p.x-hw>=-5.2&&p.x+hw<=5.2"
                "&&p.z-hd>=-24&&p.z+hd<=24};};"
            ),
            node_id="root.world.road.collision",
        )

        self.assertIn("__harness_reference_road_not_component_output", html)
        self.assertIn("collision overlap positive assertion failed", html)
        self.assertIn("collision touching-edge assertion failed", html)
        self.assertIn("collision outside-road assertion failed", html)
        self.assertIn("window.__componentSelfTest", html)

    def test_terrain_preview_enforces_density_corridor_and_owned_root(self) -> None:
        html = StateStoreUltraAdapter._generated_threejs_preview(
            (
                "window.buildPreview=({THREE,scene})=>{"
                "const root=new THREE.Group();scene.add(root);"
                "window.TerrainAPI={root,corridorHalfWidth:6.2,"
                "extents:{minZ:-26,maxZ:26}};};"
            ),
            node_id="root.world.environment.terrain",
        )

        self.assertIn("__harness_reference_road_not_component_output", html)
        self.assertIn("terrain package must publish TerrainAPI.root", html)
        self.assertIn("terrain root must contain two banks", html)
        self.assertIn("terrain root must contain two banks and at least twelve authored prop objects", html)
        self.assertIn("terrain meshes intrude into the reserved road corridor", html)
        self.assertIn("bounds.min.x < 6.19 && bounds.max.x > -6.19", html)
        self.assertIn('checks:["api","extents","mesh-density"', html)

    def test_lighting_rig_preview_removes_default_lights_and_checks_intensity(self) -> None:
        html = StateStoreUltraAdapter._generated_threejs_preview(
            (
                "window.buildPreview=({THREE,scene})=>{const root=new THREE.Group();"
                "scene.add(root);window.LightingRigAPI={root,totalIntensity:0};};"
            ),
            node_id="root.world.lighting.rig",
        )

        self.assertIn("if (item && item.isLight) scene.remove(item)", html)
        self.assertIn("lighting rig combined intensity exceeds 2.4", html)
        self.assertIn("lighting rig requires one shadow-casting directional key", html)
        self.assertIn('checks:["owned-rig","bounded-intensity"', html)

    def test_atmosphere_preview_checks_api_fog_background_and_mesh_bounds(self) -> None:
        html = StateStoreUltraAdapter._generated_threejs_preview(
            (
                "window.buildPreview=({THREE,scene})=>{const root=new THREE.Group();"
                "scene.add(root);scene.background=new THREE.Color(0x88aacc);"
                "scene.fog=new THREE.FogExp2(0x88aacc,.015);"
                "window.AtmosphereAPI={root,apply:()=>{}};};"
            ),
            node_id="root.world.lighting.atmosphere",
        )

        self.assertIn("five to eight visible depth forms", html)
        self.assertIn("color background without sky geometry", html)
        self.assertIn("occluding oversized mesh", html)
        self.assertIn('checks:["api","color-background","fog"', html)
        self.assertIn("camera.position.set(8,5.5,10)", html)

    def test_shadow_preview_uses_close_framing_and_typed_api_gate(self) -> None:
        html = StateStoreUltraAdapter._generated_threejs_preview(
            (
                "window.buildPreview=({THREE,scene})=>{const root=new THREE.Group();"
                "scene.add(root);window.ShadowQualityAPI={root,"
                "configureRenderer:()=>{},configureLight:()=>{}};};"
            ),
            node_id="root.world.lighting.shadows",
        )

        self.assertIn("ShadowQualityAPI root and configure functions", html)
        self.assertIn("four to seven visible meshes", html)
        self.assertIn("exactly one thin horizontal receiver", html)
        self.assertIn("one receiving ground and at least three casting forms", html)
        self.assertIn("grounded low-bias key", html)
        self.assertIn("item.isHemisphereLight) item.intensity=.28", html)
        self.assertIn("camera.position.set(7,5.5,9)", html)
        self.assertIn('checks:["api","fixture-density","close-framing"]', html)

    def test_chassis_shell_preview_checks_proportions_materials_and_mounts(self) -> None:
        html = StateStoreUltraAdapter._generated_threejs_preview(
            (
                "window.buildPreview=({THREE,scene})=>{const root=new THREE.Group();"
                "scene.add(root);window.ChassisShellAPI={root,forward:'+Z',"
                "wheelMounts:[{},{},{},{}]};};"
            ),
            node_id="root.vehicles.chassis.shell",
        )

        self.assertIn("+Z forward and four wheel mounts", html)
        self.assertIn("distinct paint, cladding, and trim colors", html)
        self.assertIn("contracted vehicle envelope", html)
        self.assertIn("measured x=${shellSize.x.toFixed(2)}", html)
        self.assertIn("X is lateral width", html)
        self.assertIn("camera.position.set(4.7,3.1,6.4)", html)
        self.assertIn("__harness_vehicle_floor_not_component_output", html)

    def test_chassis_shell_api_root_materializes_without_duplicate_builder(self) -> None:
        html = StateStoreUltraAdapter._generated_threejs_preview(
            (
                "const root=new THREE.Group();"
                "window.ChassisShellAPI={root,forward:'+Z',"
                "wheelMounts:[{},{},{},{}]};"
            ),
            node_id="root.vehicles.chassis.shell",
        )

        self.assertIn(
            "target.add(window.ChassisShellAPI&&window.ChassisShellAPI.root)",
            html,
        )
        self.assertNotIn('ChassisShellAPI.root")', html)

    def test_chassis_volume_preview_uses_bounded_primary_volume_gate(self) -> None:
        html = StateStoreUltraAdapter._generated_threejs_preview(
            (
                "const root=new THREE.Group();"
                "window.ChassisVolumesAPI={root,forward:'+Z'};"
            ),
            node_id="root.vehicles.chassis.shell.volumes",
        )

        self.assertIn("ChassisVolumesAPI.root with +Z forward", html)
        self.assertIn("four to seven meshes", html)
        self.assertIn("chassis volume envelope failed", html)
        self.assertIn("target.add(window.ChassisVolumesAPI&&window.ChassisVolumesAPI.root)", html)

    def test_chassis_panel_preview_supplies_reference_and_mount_grid_gate(self) -> None:
        html = StateStoreUltraAdapter._generated_threejs_preview(
            (
                "const root=new THREE.Group();"
                "window.ChassisPanelsAPI={root,forward:'+Z',wheelMounts:[{},{},{},{}]};"
            ),
            node_id="root.vehicles.chassis.shell.panels",
        )

        self.assertIn("__harness_body_reference_not_component_output", html)
        self.assertIn("every X=+/-1.38 and Z=+/-1.55 pair", html)
        self.assertIn("nine to sixteen meshes", html)
        self.assertIn("target.add(window.ChassisPanelsAPI&&window.ChassisPanelsAPI.root)", html)

    def test_panel_contract_adapter_preserves_model_source_and_adds_typed_envelope(self) -> None:
        node = WorkNode(
            contract=TaskContractV1(
                id="root.vehicles.chassis.shell.panels",
                title="Panels",
                objective="Build panel details",
                acceptance_criteria=("Runnable panels",),
                verification=("Preview",),
                metadata={
                    "component_package_only": True,
                    "specialist_domain": "vehicles.chassis.shell.panels",
                },
            )
        )
        source = (
            "window.buildPreview=({THREE,scene})=>{"
            "const root=new THREE.Group();scene.add(root);};"
        )

        adapted = StateStoreUltraAdapter._adapt_typed_component_source(node, source)

        self.assertIn(source, adapted)
        self.assertIn("window.ChassisPanelsAPI", adapted)
        self.assertIn("if(size.x>3.2)", adapted)
        self.assertIn("forward:'+Z'", adapted)

    def test_vehicle_details_adapter_repairs_canonical_wheel_centers_and_axis(self) -> None:
        node = WorkNode(
            contract=TaskContractV1(
                id="root.vehicles.vehicle_details_cut",
                title="Vehicle details",
                objective="Build wheels and details",
                acceptance_criteria=("Runnable details",),
                verification=("Preview",),
                metadata={
                    "component_package_only": True,
                    "specialist_domain": "vehicles.vehicle_details_cut",
                },
            )
        )
        source = "window.buildPreview=({scene})=>{const root={};scene.add(root);};"

        adapted = StateStoreUltraAdapter._adapt_typed_component_source(node, source)

        self.assertIn(source, adapted)
        self.assertIn("wheel.position.set(center.x,center.y,center.z)", adapted)
        self.assertIn("part.rotation.set(0,Math.PI/2,0)", adapted)
        self.assertIn("forward:'+Z'", adapted)


class PersistenceAndPickerV9Tests(unittest.TestCase):
    def test_schema_v9_contains_truthful_quality_tables(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(directory)
            store.close()
            connection = sqlite3.connect(Path(directory) / ".coding-agent" / "state.db")
            try:
                self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 11)
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
