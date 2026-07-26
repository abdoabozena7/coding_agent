"""Ask isolated Gemma specialists for bounded character specs, then compile it."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.component_dsl import CharacterDesignSpecV1, compile_character_design_spec
from agent.events import EventBus
from agent.local_provider import extract_first_json_object
from agent.model_catalog import ExecutionClass, ModelDescriptor
from agent.sandbox import AccessLevel, DockerSandbox, PermissionAdapter
from agent.store import StateStore
from agent.ultra import NodeKind, NodeStatus, TaskContractV1, UltraConfig, WorkNode
from agent.ultra_session import UltraSession


PART_PROMPTS = {
    "form": (
        "Return JSON only with body_scale (0.88-1.14), head_scale (0.88-1.18), "
        "wing_scale (0.82-1.18), style_name. Design a charming readable chicken "
        "hero for a polished sunlit lane-crossing game; avoid a block placeholder."
    ),
    "palette": (
        "Return JSON only with body, wing, accent, beak, leg, eye as #RRGGBB. "
        "Choose a warm farm palette with crisp eyes and a red comb, readable on "
        "green grass and dark asphalt."
    ),
    "motion": (
        "Return JSON only with hop_height (0.48-0.88), hop_duration (0.26-0.46), "
        "flap_angle (0.48-0.92), squash (0.10-0.23). Choose responsive playful "
        "hop motion that clearly telegraphs takeoff, air, and landing."
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
        f"You are the isolated character {part} specialist. Make bounded decisions only; the compiler owns topology and animation plumbing.",
    )
    return dict(extract_first_json_object(turn.text) or {})


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
        provider="ollama", model=args.model, execution_class=ExecutionClass.LOCAL,
        host=os.getenv("OLLAMA_HOST", "http://localhost:11434"), capabilities=("tools",),
        source="character-spec-proof", metadata={"gpu_required": True},
    )
    if args.reuse_spec_from:
        cached = json.loads(Path(args.reuse_spec_from).read_text(encoding="utf-8"))
        parts = {str(k): dict(v) for k, v in dict(cached.get("parts") or {}).items() if isinstance(v, dict)}
    else:
        parts = {part: _part_spec(descriptor, part) for part in PART_PROMPTS}
    spec = CharacterDesignSpecV1.from_parts(parts)
    source = compile_character_design_spec(spec)

    workspace = Path(args.workspace).resolve(strict=True)
    store = StateStore(workspace)
    session = UltraSession(
        store=store, workspace=workspace, descriptor=descriptor,
        permission_adapter=PermissionAdapter(AccessLevel.NORMAL, DockerSandbox()),
        approval=lambda _tool, _args, _risk: True, events=EventBus(),
        config=UltraConfig(max_fix_attempts=2, context_chars=12_000), agent_steps=8,
    )
    try:
        session.restore(args.run_id, start_background=False)
        assert session.orchestrator is not None and session.adapter is not None
        orchestrator = session.orchestrator
        node_id = f"{args.parent}.compiled_character_cut"
        if node_id not in orchestrator.nodes:
            node = WorkNode(
                contract=TaskContractV1(
                    id=node_id, title="Compiled animated character specialist synthesis",
                    objective="Compile form, palette, and motion specialist specs into one expressive chicken hero.",
                    acceptance_criteria=(
                        "The character has a readable chicken silhouette and modeled facial/body details.",
                        "The character exposes deterministic idle, hop, and hit animation states.",
                        "The exact specialist specs and compiled source remain durable and independently reviewable.",
                    ),
                    verification=("Run typed character geometry, palette, animation API, bounds, and screenshot gates.",),
                    # Animation is an exported capability of the already-owned
                    # CharacterPackage, not a new cross-subsystem interface.
                    owned_interfaces=("CharacterPackage",),
                    metadata={
                        "component_package_only": True, "component_leaf": True,
                        "materialized_components_required": True,
                        "specialist_domain": "character.compiled_character_cut",
                        "visual_required": True, "execution_cut": True,
                        "spec_specialists": list(PART_PROMPTS),
                    },
                ), parent_id=args.parent, depth=2, kind=NodeKind.TASK,
                order=max((item.order for item in orchestrator.nodes.values()), default=0) + 1,
                status=NodeStatus.READY,
            )
            orchestrator.nodes[node_id] = node
            orchestrator.state.save_work_node(args.run_id, node)
        node = orchestrator.nodes[node_id]
        session.adapter.stage_component_file_tool(args.run_id, node, path="preview/scene.js", content=source, role="preview")
        session.adapter.stage_component_file_tool(
            args.run_id, node, path="spec/character-design.json",
            content=json.dumps({"parts": parts, "normalized": spec.to_dict()}, ensure_ascii=False, indent=2), role="asset",
        )
        session.adapter.stage_component_file_tool(
            args.run_id, node, path="test/character-contract.test.js",
            content="export function verify(api){if(!api||api.forward!=='+Z'||!api.states.includes('hop')||typeof api.create!=='function')throw new Error('compiled character contract');return true;}\n",
            role="test",
        )
        result = session.adapter.publish_component_tool(
            args.run_id, node,
            {"interface": {"exports": ["CompiledCharacterAPI"]},
             "preview": {"entrypoint": "preview/index.html"},
             "quality": {"strategy": "typed_specialist_spec_compiler_v1"},
             "evidence": [{"kind": "specialist_specs", "specialists": list(parts)},
                          {"kind": "deterministic_compiler", "schema": "CharacterDesignSpecV1"}]},
        )
        print(json.dumps({
            "node_id": node_id, "passed": result.get("passed"), "status": result.get("status"),
            "parts": parts, "normalized_spec": spec.to_dict(), "preview": result.get("preview"),
            "findings": result.get("findings", []), "package": dict(result.get("package") or {}).get("root"),
        }, ensure_ascii=False), flush=True)
        return 0 if result.get("passed") else 2
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
