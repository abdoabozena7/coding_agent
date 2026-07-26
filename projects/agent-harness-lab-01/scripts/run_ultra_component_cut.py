"""Execute a tiny, measurable Ultra component cut without waking the full DAG.

The builder model only implements leaf artifacts.  The harness materializes and
runtime-checks them, then composes their exact hashes deterministically.  Model
self-review is deliberately absent: Codex (or a configured vision adapter)
reviews the resulting screenshots before a second attempt is authorized.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from agent.events import EventBus
from agent.model_catalog import ExecutionClass, ModelDescriptor
from agent.sandbox import AccessLevel, DockerSandbox, PermissionAdapter
from agent.store import StateStore
from agent.ultra import (
    AgentResponse,
    AgentRole,
    InnerPhase,
    NodeKind,
    NodeStatus,
    ResultPackageV1,
    TaskContractV1,
    UltraConfig,
    WorkNode,
)
from agent.ultra_session import UltraSession


def _save_candidate(orchestrator, node, materialized, response) -> ResultPackageV1:
    passed = bool(materialized.get("passed"))
    package = dict(materialized.get("package") or {})
    findings = tuple(str(item) for item in materialized.get("findings", ()) if str(item))
    result = ResultPackageV1(
        node_id=node.id,
        success=passed,
        status=("provisional_codex_review" if passed else "revision_required"),
        summary=response.summary or f"Materialized component candidate for {node.id}",
        evidence=tuple(
            item for item in response.payload.get("evidence", ()) if isinstance(item, dict)
        ),
        findings=findings,
        component_package=package,
    )
    orchestrator._results[node.id] = result
    orchestrator.state.save_result_package(orchestrator.run_state.id, result)
    updated = replace(
        node,
        status=NodeStatus.COMPLETED if passed else NodeStatus.REVISION_REQUIRED,
        phase=InnerPhase.INTEGRATE if passed else InnerPhase.FIX,
    )
    orchestrator.nodes[node.id] = updated
    orchestrator.state.save_work_node(orchestrator.run_state.id, updated)
    return result


def _next_revision(store: StateStore, run_id: str, node_id: str) -> int:
    return 1 + max(
        (
            int(item.get("version") or 0)
            for item in store.list_component_packages(run_id, work_node_id=node_id)
        ),
        default=0,
    )


def _build_leaf(session: UltraSession, node_id: str) -> ResultPackageV1:
    orchestrator = session.orchestrator
    assert orchestrator is not None and orchestrator.run_state is not None
    node = orchestrator.nodes[node_id]
    if node.children:
        raise RuntimeError(f"execution-cut leaf {node_id} unexpectedly has children")
    if not node.contract.metadata.get("component_package_only"):
        raise RuntimeError(f"execution-cut leaf {node_id} is not an isolated component")
    revision = _next_revision(session.store, orchestrator.run_state.id, node.id)
    prior_findings = ()
    reader = getattr(orchestrator.state, "component_revision_findings", None)
    if callable(reader):
        prior_findings = tuple(reader(orchestrator.run_state.id, node))
    prior_result = orchestrator._results.get(node.id)
    if prior_result is not None:
        prior_findings = tuple(
            dict.fromkeys((*prior_findings, *prior_result.findings))
        )
    required_api = {
        "vehicles.chassis.shell.volumes": (
            "window.ChassisVolumesAPI={root,dimensions:{width:2.8,height:1.5,length:5.2},forward:'+Z'};"
        ),
        "vehicles.chassis.shell.panels": (
            "window.ChassisPanelsAPI={root,wheelMounts:[{x:1.38,y:0.55,z:1.55},{x:-1.38,y:0.55,z:1.55},{x:1.38,y:0.55,z:-1.55},{x:-1.38,y:0.55,z:-1.55}],forward:'+Z'};"
        ),
        "vehicles.vehicle_details_cut": (
            "Use <=70 lines and tuple loops only; close every brace. Build 20-36 meshes, add one root to scene, and publish window.VehicleDetailsAPI with root and forward:'+Z'."
        ),
        "vehicles.vehicle_wheels_cut": (
            "Use <=65 lines and close every brace. One createWheel helper: tire, rim, hub, then 5 spokes in one loop. One loop places exactly four groups at canonical centers. No narrative comments or per-spoke variables."
        ),
        "vehicles.vehicle_glass_cut": (
            "Use <=65 lines and close every brace. Use one addPane helper plus tuple arrays for six thin transparent panes, pillars, roof trim, and two mirrors. No narrative comments."
        ),
        "vehicles.vehicle_fascia_cut": (
            "Use <=65 lines and close every brace. Use one addBox helper and tuple arrays for paired lamps, bumpers, grille bars, intake, side trim, and badges. No narrative comments."
        ),
    }.get(str(node.contract.metadata.get("specialist_domain") or ""), "")
    response = orchestrator._invoke(
        AgentRole.CODER,
        InnerPhase.FIX if revision > 1 else InnerPhase.IMPLEMENT,
        task={
            "contract": asdict(node.contract),
            "attempt": revision,
            "findings": list(prior_findings),
            "execution_cut": True,
            "independent_review_owner": "Codex",
            "hard_blocker_before_publish": (
                f"The complete scene.js must contain this exact typed export after root is built: {required_api}"
                if required_api
                else ""
            ),
        },
        context=orchestrator._new_context(node, AgentRole.CODER),
        node_id=node.id,
    )
    materialized, _gate = orchestrator._materialize_component_gate(
        node,
        response,
        revision=revision,
    )
    return _save_candidate(orchestrator, node, materialized, response)


def _adapt_existing_leaf(
    session: UltraSession,
    node_id: str,
    *,
    source_version: int | None = None,
) -> ResultPackageV1:
    """Repackage the latest staged bytes through a new harness adapter only."""

    orchestrator = session.orchestrator
    assert (
        orchestrator is not None
        and orchestrator.run_state is not None
        and session.adapter is not None
    )
    node = orchestrator.nodes[node_id]
    if source_version is not None:
        if session.adapter.component_artifacts is None:
            raise RuntimeError("component artifact store is unavailable")
        source_root = session.adapter.component_artifacts.package_root(
            orchestrator.run_state.id,
            node.id,
            source_version,
        )
        staged_sources = [("preview/scene.js", "preview")]
        staged_sources.extend(
            (
                path.relative_to(source_root).as_posix(),
                "test",
            )
            for path in sorted((source_root / "test").glob("*.js"))
        )
        for relative, role in staged_sources:
            path = source_root / relative
            if not path.is_file():
                raise RuntimeError(f"missing champion file: {path}")
            session.adapter.stage_component_file_tool(
                orchestrator.run_state.id,
                node,
                path=relative,
                content=path.read_text(encoding="utf-8"),
                role=role,
            )
    result = dict(
        session.adapter.publish_component_tool(
            orchestrator.run_state.id,
            node,
            {
                "interface": {
                    "exports": list(node.contract.owned_interfaces),
                    "integration_points": [
                        "Harness adapts a valid buildPreview root to the typed parent contract."
                    ],
                },
                "preview": {"entrypoint": "preview/index.html"},
                "quality": {"strategy": "typed_buildPreview_adapter"},
            },
        )
    )
    response = AgentResponse(
        payload={"evidence": [{"kind": "existing_bytes_adapter"}]},
        summary=f"Repackaged existing Gemma bytes through the typed adapter for {node.id}",
        provider="harness",
        model="typed-buildPreview-adapter-v1",
    )
    return _save_candidate(orchestrator, node, result, response)


def _assemble_parent(session: UltraSession, parent_id: str) -> ResultPackageV1:
    orchestrator = session.orchestrator
    assert orchestrator is not None and orchestrator.run_state is not None
    node = orchestrator.nodes[parent_id]
    missing = [child_id for child_id in node.children if child_id not in orchestrator._results]
    if missing:
        raise RuntimeError(f"parent {parent_id} is missing cut results: {missing}")
    failed = [
        child_id
        for child_id in node.children
        if not orchestrator._results[child_id].success
    ]
    if failed:
        raise RuntimeError(f"parent {parent_id} has rejected children: {failed}")
    revision = _next_revision(session.store, orchestrator.run_state.id, node.id)
    response = orchestrator._invoke(
        AgentRole.INTEGRATOR,
        InnerPhase.INTEGRATE,
        task={
            "contract": asdict(node.contract),
            "component_assembler": True,
            "final_assembler": False,
            "execution_cut": True,
            "child_component_packages": {
                child_id: dict(orchestrator._results[child_id].component_package)
                for child_id in node.children
            },
        },
        context=orchestrator._new_context(node, AgentRole.INTEGRATOR),
        node_id=node.id,
    )
    materialized, _gate = orchestrator._materialize_component_gate(
        node,
        response,
        revision=revision,
    )
    return _save_candidate(orchestrator, node, materialized, response)


def _ensure_vehicle_proof_nodes(
    session: UltraSession,
    shell_id: str,
) -> tuple[str, str]:
    orchestrator = session.orchestrator
    assert orchestrator is not None and orchestrator.run_state is not None
    if shell_id not in orchestrator.nodes:
        raise RuntimeError(f"unknown accepted shell node: {shell_id}")
    vehicles_parent = shell_id.rsplit(".chassis.shell", 1)[0]
    details_id = f"{vehicles_parent}.vehicle_details_cut"
    preview_id = f"{vehicles_parent}.vehicle_preview_cut"
    if details_id not in orchestrator.nodes:
        details = WorkNode(
            contract=TaskContractV1(
                id=details_id,
                title="Wheels, glass, lights, and trim specialist",
                objective=(
                    "Build only the complete aligned detail kit for an accepted stylized "
                    "2.8-wide, 5.2-long vehicle body: four finished wheels, cabin glass, "
                    "head/tail lamps, bumpers, grille, mirrors, and trim."
                ),
                acceptance_criteria=(
                    "Four grounded tire/rim/hub assemblies align to X=+/-1.48 and Z=+/-1.55.",
                    "Cabin glass, lamps, bumpers, grille, mirrors, and trim are separate readable meshes.",
                    "The runnable preview contains 20-36 meshes, five colors, and no replacement chassis.",
                ),
                verification=(
                    "Run the isolated Three.js preview and verify density, bounds, materials, and alignment.",
                    "Codex reviews the screenshot independently; Gemma does not judge its own work.",
                ),
                owned_interfaces=("VehiclePackage",),
                metadata={
                    "component_package_only": True,
                    "component_leaf": True,
                    "materialized_components_required": True,
                    "specialist_domain": "vehicles.vehicle_details_cut",
                    "visual_required": True,
                    "execution_cut": True,
                },
            ),
            parent_id=vehicles_parent,
            depth=3,
            kind=NodeKind.TASK,
            order=max((item.order for item in orchestrator.nodes.values()), default=0) + 1,
            status=NodeStatus.READY,
        )
        orchestrator.nodes[details_id] = details
        orchestrator.state.save_work_node(orchestrator.run_state.id, details)
    if preview_id not in orchestrator.nodes:
        preview = WorkNode(
            contract=TaskContractV1(
                id=preview_id,
                title="Accepted vehicle preview assembler",
                objective="Compose the exact accepted body and detail packages without rewriting either.",
                acceptance_criteria=(
                    "The preview consumes both exact package hashes.",
                    "The integrated vehicle has body, wheels, glass, lights, and trim in one readable silhouette.",
                ),
                verification=(
                    "Run the integrated preview and verify package consumption plus visual density.",
                ),
                depends_on=(shell_id, details_id),
                owned_interfaces=("VehiclePackage",),
                metadata={
                    "component_package_only": True,
                    "materialized_components_required": True,
                    "specialist_domain": "vehicles.vehicle_preview_cut",
                    "visual_required": True,
                    "execution_cut": True,
                },
            ),
            parent_id=vehicles_parent,
            depth=2,
            kind=NodeKind.TASK,
            order=max((item.order for item in orchestrator.nodes.values()), default=0) + 1,
            status=NodeStatus.READY,
            children=(shell_id, details_id),
        )
        orchestrator.nodes[preview_id] = preview
        orchestrator.state.save_work_node(orchestrator.run_state.id, preview)
    return details_id, preview_id


def _ensure_vehicle_proof_v2_nodes(
    session: UltraSession,
    shell_id: str,
) -> tuple[tuple[str, ...], str]:
    orchestrator = session.orchestrator
    assert orchestrator is not None and orchestrator.run_state is not None
    if shell_id not in orchestrator.nodes:
        raise RuntimeError(f"unknown accepted shell node: {shell_id}")
    vehicles_parent = shell_id.rsplit(".chassis.shell", 1)[0]
    definitions = (
        (
            "vehicle_wheels_cut",
            "Wheels specialist",
            "Build only four complete aligned wheel assemblies with tire, inset rim, hub, and visible spoke/tread detail.",
            "vehicles.vehicle_wheels_cut",
        ),
        (
            "vehicle_glass_cut",
            "Glass and mirrors specialist",
            "Build only thin transparent windshield, rear and side glazing, slim pillars/trim, and two side mirrors.",
            "vehicles.vehicle_glass_cut",
        ),
        (
            "vehicle_fascia_cut",
            "Fascia, lights, and trim specialist",
            "Build only paired headlights/tail lights, bumpers, grille, lower intake, side trim, and small badges.",
            "vehicles.vehicle_fascia_cut",
        ),
    )
    leaf_ids: list[str] = []
    next_order = max((item.order for item in orchestrator.nodes.values()), default=0)
    for suffix, title, objective, domain in definitions:
        node_id = f"{vehicles_parent}.{suffix}"
        leaf_ids.append(node_id)
        if node_id in orchestrator.nodes:
            continue
        next_order += 1
        node = WorkNode(
            contract=TaskContractV1(
                id=node_id,
                title=title,
                objective=objective,
                acceptance_criteria=(
                    "The owned vehicle slice is concrete, aligned, and independently previewable.",
                    "The package contains no replacement chassis or unrelated scene ownership.",
                    "Codex independently reviews the screenshot after deterministic runtime gates pass.",
                ),
                verification=(
                    "Run the bounded Three.js preview and specialist-specific geometry/material checks.",
                ),
                owned_interfaces=("VehiclePackage",),
                metadata={
                    "component_package_only": True,
                    "component_leaf": True,
                    "materialized_components_required": True,
                    "specialist_domain": domain,
                    "visual_required": True,
                    "execution_cut": True,
                },
            ),
            parent_id=vehicles_parent,
            depth=3,
            kind=NodeKind.TASK,
            order=next_order,
            status=NodeStatus.READY,
        )
        orchestrator.nodes[node_id] = node
        orchestrator.state.save_work_node(orchestrator.run_state.id, node)
    preview_id = f"{vehicles_parent}.vehicle_preview_v2_cut"
    if preview_id not in orchestrator.nodes:
        next_order += 1
        preview = WorkNode(
            contract=TaskContractV1(
                id=preview_id,
                title="Split-specialist vehicle preview assembler",
                objective="Compose exact accepted body, wheels, glazing, and fascia packages without rewriting them.",
                acceptance_criteria=(
                    "All four accepted package hashes are consumed.",
                    "The integrated vehicle has a coherent silhouette, grounded wheels, thin glazing, readable lamps, and restrained trim.",
                ),
                verification=(
                    "Run the integrated preview, package-consumption evidence, and Codex screenshot review.",
                ),
                depends_on=(shell_id, *leaf_ids),
                owned_interfaces=("VehiclePackage",),
                metadata={
                    "component_package_only": True,
                    "materialized_components_required": True,
                    "specialist_domain": "vehicles.vehicle_preview_v2_cut",
                    "visual_required": True,
                    "execution_cut": True,
                },
            ),
            parent_id=vehicles_parent,
            depth=2,
            kind=NodeKind.TASK,
            order=next_order,
            status=NodeStatus.READY,
            children=(shell_id, *leaf_ids),
        )
        orchestrator.nodes[preview_id] = preview
        orchestrator.state.save_work_node(orchestrator.run_state.id, preview)
    return tuple(leaf_ids), preview_id


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model", default="gemma4:e4b")
    parser.add_argument("--parent")
    parser.add_argument("--leaf", action="append", default=[])
    parser.add_argument("--adapt-leaf", action="append", default=[])
    parser.add_argument("--adapt-version", type=int)
    parser.add_argument("--vehicle-proof-shell")
    parser.add_argument("--vehicle-proof-v2-shell")
    args = parser.parse_args()

    os.environ.setdefault("LLM_PROVIDER", "ollama")
    os.environ.setdefault("OLLAMA_MODEL", args.model)
    os.environ.setdefault("OLLAMA_NUM_GPU", "999")
    os.environ.setdefault("OLLAMA_CONTEXT_SIZE", "4096")
    os.environ.setdefault("AGENT_REQUIRE_LOCAL_GPU", "1")

    workspace = Path(args.workspace).resolve(strict=True)
    descriptor = ModelDescriptor(
        provider="ollama",
        model=args.model,
        execution_class=ExecutionClass.LOCAL,
        host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        capabilities=("tools",),
        source="ultra-component-cut",
        metadata={"gpu_required": True},
    )
    store = StateStore(workspace)
    bus = EventBus()
    bus.subscribe(
        lambda event: print(
            f"[{event.kind}] {event.message}",
            flush=True,
        )
    )
    session = UltraSession(
        store=store,
        workspace=workspace,
        descriptor=descriptor,
        permission_adapter=PermissionAdapter(AccessLevel.NORMAL, DockerSandbox()),
        approval=lambda _tool, _args, _risk: True,
        events=bus,
        config=UltraConfig(max_fix_attempts=2, context_chars=12_000),
        agent_steps=8,
        reasoning_effort="medium",
    )
    try:
        session.restore(args.run_id, start_background=False)
        if args.vehicle_proof_shell:
            details_id, preview_id = _ensure_vehicle_proof_nodes(
                session,
                args.vehicle_proof_shell,
            )
            details = _build_leaf(session, details_id)
            print(json.dumps({
                "node_id": details_id,
                "success": details.success,
                "status": details.status,
                "findings": list(details.findings),
            }, ensure_ascii=False), flush=True)
            if not details.success:
                return 2
            preview = _assemble_parent(session, preview_id)
            package = dict(preview.component_package)
            print(json.dumps({
                "node_id": preview_id,
                "success": preview.success,
                "status": preview.status,
                "root": package.get("root"),
                "findings": list(preview.findings),
            }, ensure_ascii=False), flush=True)
            return 0 if preview.success else 3
        if args.vehicle_proof_v2_shell:
            leaf_ids, preview_id = _ensure_vehicle_proof_v2_nodes(
                session,
                args.vehicle_proof_v2_shell,
            )
            for leaf_id in leaf_ids:
                existing = session.orchestrator._results.get(leaf_id)
                result = (
                    existing
                    if existing is not None and existing.success
                    else _build_leaf(session, leaf_id)
                )
                print(json.dumps({
                    "node_id": leaf_id,
                    "success": result.success,
                    "status": (
                        "reused_accepted_component"
                        if existing is not None and existing.success
                        else result.status
                    ),
                    "findings": list(result.findings),
                }, ensure_ascii=False), flush=True)
                if not result.success:
                    return 2
            preview = _assemble_parent(session, preview_id)
            package = dict(preview.component_package)
            print(json.dumps({
                "node_id": preview_id,
                "success": preview.success,
                "status": preview.status,
                "root": package.get("root"),
                "findings": list(preview.findings),
            }, ensure_ascii=False), flush=True)
            return 0 if preview.success else 3
        adapted = set(args.adapt_leaf)
        for leaf_id in args.leaf:
            if leaf_id in adapted:
                result = _adapt_existing_leaf(
                    session,
                    leaf_id,
                    source_version=args.adapt_version,
                )
            else:
                result = _build_leaf(session, leaf_id)
            preview = dict(result.component_package.get("preview") or {})
            print(
                json.dumps(
                    {
                        "node_id": leaf_id,
                        "success": result.success,
                        "status": result.status,
                        "screenshot": preview.get("screenshot"),
                        "findings": list(result.findings),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if not result.success:
                return 2
        if not args.parent:
            return 0
        parent = _assemble_parent(session, args.parent)
        package = dict(parent.component_package)
        print(
            json.dumps(
                {
                    "node_id": args.parent,
                    "success": parent.success,
                    "status": parent.status,
                    "package_id": package.get("id"),
                    "root": package.get("root"),
                    "preview": package.get("preview"),
                    "findings": list(parent.findings),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return 0 if parent.success else 3
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
