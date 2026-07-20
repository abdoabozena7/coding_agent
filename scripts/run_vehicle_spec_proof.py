"""Ask four Gemma specialists for bounded specs, then compile one vehicle."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.component_dsl import VehicleDesignSpecV1, compile_vehicle_design_spec
from agent.events import EventBus
from agent.local_provider import extract_first_json_object
from agent.model_catalog import ExecutionClass, ModelDescriptor
from agent.sandbox import AccessLevel, DockerSandbox, PermissionAdapter
from agent.store import StateStore
from agent.ultra import NodeKind, NodeStatus, TaskContractV1, UltraConfig, WorkNode
from agent.ultra_session import UltraSession


PART_PROMPTS = {
    "body": (
        "Return JSON only with keys paint, paint_secondary, trim, cabin_taper, "
        "hood_slope, stance, style_name. Colors are #RRGGBB. Design a polished "
        "sunlit stylized rally hatch for a cheerful lane-crossing game."
    ),
    "wheels": (
        "Return JSON only with keys radius (0.40-0.52), width (0.24-0.34), "
        "spokes (5-8), rim (#RRGGBB). Choose grounded sporty wheels that do not "
        "dominate a compact 2.8-wide vehicle."
    ),
    "glass": (
        "Return JSON only with keys tint (#RRGGBB) and opacity (0.42-0.68). "
        "Choose readable thin automotive glass for a warm stylized scene."
    ),
    "fascia": (
        "Return JSON only with keys trim, headlight, taillight as #RRGGBB. "
        "Choose a coherent charcoal grille, warm headlights, and crisp red tails."
    ),
}


def _part_spec(descriptor: ModelDescriptor, part: str) -> dict:
    provider = descriptor.create_provider()
    provider.reasoning_effort = "off"
    provider.force_json = True
    provider.max_output_tokens = 192
    provider.temperature = 0.35
    turn = provider.call(
        [{"role": "user", "content": PART_PROMPTS[part]}],
        [],
        (
            f"You are the isolated {part} design specialist. Make only bounded "
            "aesthetic decisions; the geometry compiler owns topology."
        ),
    )
    parsed = extract_first_json_object(turn.text)
    return dict(parsed or {})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model", default="gemma4:e4b")
    parser.add_argument("--parent", required=True)
    parser.add_argument("--reuse-spec-from")
    args = parser.parse_args()

    os.environ.setdefault("LLM_PROVIDER", "ollama")
    os.environ.setdefault("OLLAMA_MODEL", args.model)
    os.environ.setdefault("OLLAMA_NUM_GPU", "999")
    os.environ.setdefault("OLLAMA_CONTEXT_SIZE", "4096")
    os.environ.setdefault("AGENT_REQUIRE_LOCAL_GPU", "1")
    descriptor = ModelDescriptor(
        provider="ollama",
        model=args.model,
        execution_class=ExecutionClass.LOCAL,
        host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        capabilities=("tools",),
        source="vehicle-spec-proof",
        metadata={"gpu_required": True},
    )
    if args.reuse_spec_from:
        cached = json.loads(Path(args.reuse_spec_from).read_text(encoding="utf-8"))
        parts = {
            str(key): dict(value)
            for key, value in dict(cached.get("parts") or {}).items()
            if isinstance(value, dict)
        }
    else:
        parts = {part: _part_spec(descriptor, part) for part in PART_PROMPTS}
    spec = VehicleDesignSpecV1.from_parts(parts)
    source = compile_vehicle_design_spec(spec)

    workspace = Path(args.workspace).resolve(strict=True)
    store = StateStore(workspace)
    session = UltraSession(
        store=store,
        workspace=workspace,
        descriptor=descriptor,
        permission_adapter=PermissionAdapter(AccessLevel.NORMAL, DockerSandbox()),
        approval=lambda _tool, _args, _risk: True,
        events=EventBus(),
        config=UltraConfig(max_fix_attempts=2, context_chars=12_000),
        agent_steps=8,
    )
    try:
        session.restore(args.run_id, start_background=False)
        assert session.orchestrator is not None and session.adapter is not None
        orchestrator = session.orchestrator
        node_id = f"{args.parent}.compiled_vehicle_cut"
        if node_id not in orchestrator.nodes:
            node = WorkNode(
                contract=TaskContractV1(
                    id=node_id,
                    title="Compiled vehicle specialist synthesis",
                    objective="Compile the accepted body/wheel/glass/fascia specialist specs into one cohesive vehicle.",
                    acceptance_criteria=(
                        "The component is a cohesive detailed vehicle, not detached primitives.",
                        "The preview has grounded wheels, thin glazing, readable fascia, and a strong silhouette.",
                        "The exact specialist specs and compiled source are durable and independently reviewable.",
                    ),
                    verification=(
                        "Run the typed compiled-vehicle geometry, material, bounds, and screenshot gates.",
                    ),
                    owned_interfaces=("VehiclePackage",),
                    metadata={
                        "component_package_only": True,
                        "component_leaf": True,
                        "materialized_components_required": True,
                        "specialist_domain": "vehicles.compiled_vehicle_cut",
                        "visual_required": True,
                        "execution_cut": True,
                        "spec_specialists": list(PART_PROMPTS),
                    },
                ),
                parent_id=args.parent,
                depth=2,
                kind=NodeKind.TASK,
                order=max((item.order for item in orchestrator.nodes.values()), default=0) + 1,
                status=NodeStatus.READY,
            )
            orchestrator.nodes[node_id] = node
            orchestrator.state.save_work_node(args.run_id, node)
        node = orchestrator.nodes[node_id]
        session.adapter.stage_component_file_tool(
            args.run_id,
            node,
            path="preview/scene.js",
            content=source,
            role="preview",
        )
        session.adapter.stage_component_file_tool(
            args.run_id,
            node,
            path="spec/vehicle-design.json",
            content=json.dumps(
                {"parts": parts, "normalized": spec.to_dict()},
                ensure_ascii=False,
                indent=2,
            ),
            role="asset",
        )
        session.adapter.stage_component_file_tool(
            args.run_id,
            node,
            path="test/vehicle-spec-contract.test.js",
            content=(
                "export function verify(api){if(!api||api.forward!=='+Z'||"
                "api.wheelCenters.length!==4)throw new Error('compiled vehicle contract');return true;}\n"
            ),
            role="test",
        )
        result = session.adapter.publish_component_tool(
            args.run_id,
            node,
            {
                "interface": {"exports": ["CompiledVehicleAPI"]},
                "preview": {"entrypoint": "preview/index.html"},
                "quality": {"strategy": "typed_specialist_spec_compiler_v1"},
                "evidence": [
                    {"kind": "specialist_specs", "specialists": list(parts)},
                    {"kind": "deterministic_compiler", "schema": "VehicleDesignSpecV1"},
                ],
            },
        )
        print(json.dumps({
            "node_id": node_id,
            "passed": result.get("passed"),
            "status": result.get("status"),
            "parts": parts,
            "normalized_spec": spec.to_dict(),
            "preview": result.get("preview"),
            "findings": result.get("findings", []),
            "package": dict(result.get("package") or {}).get("root"),
        }, ensure_ascii=False), flush=True)
        return 0 if result.get("passed") else 2
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
