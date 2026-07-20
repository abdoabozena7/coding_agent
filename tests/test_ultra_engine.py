from __future__ import annotations

import threading
import time
import unittest
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from agent.events import EventBus
from agent.scheduler import (
    AdaptiveConcurrency,
    DeterministicWaveScheduler,
    ExecutionClass,
    RateLimitError,
    ResourceLeaseManager,
    ScheduleStatus,
)
from agent.reasoning import evaluate_reasoning_artifact, repair_reasoning_artifact_graph, reasoning_debate_protocol_for
from agent.ultra import (
    AgentRequest,
    AgentResponse,
    AgentRole,
    InMemoryUltraState,
    InnerPhase,
    TaskContractV1,
    UltraConfig,
    UltraOrchestrator,
    UltraPhase,
)


@dataclass(frozen=True)
class Item:
    id: str
    depends_on: tuple[str, ...] = ()
    write_paths: tuple[str, ...] = ()
    pre_write_hashes: Mapping[str, str | None] = field(default_factory=dict)
    order: int = 0


class FakeAgent:
    def __init__(self, handler: Callable[[AgentRequest], AgentResponse]):
        self.handler = handler

    def execute(self, request: AgentRequest) -> AgentResponse:
        return self.handler(request)


class FakeFactory:
    def __init__(self, handler: Callable[[AgentRequest], AgentResponse]):
        self.handler = handler
        self.created: list[tuple[AgentRole, str | None]] = []
        self.requests: list[AgentRequest] = []
        self._lock = threading.Lock()

    def create(self, role: AgentRole, *, run_id: str, node_id: str | None = None):
        del run_id
        with self._lock:
            self.created.append((role, node_id))

        def execute(request: AgentRequest) -> AgentResponse:
            with self._lock:
                self.requests.append(request)
            return self.handler(request)

        return FakeAgent(execute)


class ProjectMemoryState(InMemoryUltraState):
    def __init__(self) -> None:
        super().__init__()
        self.lookups: list[tuple[str, str]] = []

    def foundation_project_lessons(self, run_id: str, query: str, *, phase: str):
        self.lookups.append((phase, query))
        return (
            {
                "id": f"lesson-{phase}",
                "title": "Avoid shallow visual acceptance",
                "content": "Require browser/runtime evidence before planning completion.",
                "confidence": 0.91,
                "reuse_count": 3,
                "evidence_refs": ["bench:visual"],
            },
        )


def module(module_id: str, path: str, depends_on=()):
    return {
        "id": module_id,
        "title": f"Module {module_id}",
        "objective": f"Implement {module_id}",
        "acceptance_criteria": [f"{module_id} works"],
        "verification": [f"test {module_id}"],
        "depends_on": list(depends_on),
        "write_paths": [path],
        "forbidden_changes": ["do not change public scope"],
    }


def standard_handler(
    request: AgentRequest,
    *,
    modules: list[dict[str, Any]] | None = None,
) -> AgentResponse:
    modules = modules or [
        module("M1", "src/one"),
        module("M2", "src/two"),
        module("M3", "src/three"),
        module("M4", "src/four"),
    ]
    if request.phase == "goal_spec":
        payload = {
            "objective": "Build a complete system",
            "success_criteria": ["all modules pass their tests"],
            "constraints": ["preserve compatibility"],
        }
    elif request.phase == "architecture":
        payload = {
            "summary": "A modular architecture",
            "components": [{"name": item["id"]} for item in modules],
            "interfaces": [{"name": "stable-api"}],
        }
    elif request.phase == "master_plan":
        payload = {
            "summary": "Implement all modules and verify them",
            "execution_strategy": "dependency-safe waves",
            "modules": modules,
        }
    elif request.phase == InnerPhase.MINI_PLAN.value:
        payload = {"steps": ["implement", "verify"], "research_required": False}
    elif request.phase == InnerPhase.DECOMPOSE.value:
        payload = {"children": []}
    elif request.phase in {
        InnerPhase.REVIEW.value,
        InnerPhase.TEST.value,
        InnerPhase.INTEGRATE.value,
        InnerPhase.GLOBAL_INTEGRATION.value,
        InnerPhase.GLOBAL_REVIEW.value,
        InnerPhase.FINAL_EVIDENCE.value,
    }:
        payload = {
            "passed": True,
            "evidence": [{"kind": request.phase, "verified": True}],
            "test_results": [{"name": request.phase, "passed": True}],
        }
    elif request.phase == InnerPhase.IMPLEMENT.value:
        payload = {"artifacts": [{"path": f"{request.node_id}.py"}]}
    else:
        payload = {}
    return AgentResponse(
        payload=payload,
        summary=f"{request.phase}:{request.node_id or 'global'}",
        reasoning_summary="Decision based on the supplied contract and evidence.",
        provider="fake",
        model="scripted",
    )


def prepared_engine(
    execution_class=ExecutionClass.LOCAL,
    *,
    handler: Callable[[AgentRequest], AgentResponse] | None = None,
    modules: list[dict[str, Any]] | None = None,
    state: InMemoryUltraState | None = None,
):
    selected = handler or (lambda request: standard_handler(request, modules=modules))
    factory = FakeFactory(selected)
    engine = UltraOrchestrator(
        factory,
        execution_class=execution_class,
        state=state,
        config=UltraConfig(
            min_top_modules=1,
            max_top_modules=12,
            cloud_concurrency=4,
            provider_retries=2,
        ),
    )
    plan = engine.prepare("build the whole product")
    engine.approve(plan.fingerprint)
    return engine, factory, plan


class SchedulerTests(unittest.TestCase):
    def test_malformed_architecture_candidate_recovers_to_typed_harness_topology(self):
        def handler(request: AgentRequest) -> AgentResponse:
            response = standard_handler(request)
            if request.phase == "architecture":
                return AgentResponse(
                    payload={"summary": "", "components": []},
                    summary="malformed architecture envelope",
                    provider="fake",
                    model="scripted",
                )
            return response

        factory = FakeFactory(handler)
        engine = UltraOrchestrator(
            factory,
            execution_class=ExecutionClass.LOCAL,
            config=UltraConfig(
                min_top_modules=1,
                max_top_modules=12,
                provider_retries=1,
            ),
        )
        plan = engine.prepare("Build a polished Three.js browser game")
        self.assertIsNotNone(plan)
        self.assertIn("recursive", engine.architecture.summary.casefold())
        self.assertEqual(
            {item["name"] for item in engine.architecture.components},
            {"World", "Vehicles", "Character", "Gameplay", "Presentation", "QA"},
        )

    def test_cross_run_lessons_are_injected_into_foundation_planning(self):
        state = ProjectMemoryState()
        engine, factory, _plan = prepared_engine(state=state)
        del engine
        by_phase = {request.phase: request for request in factory.requests}

        for phase in ("goal_spec", "architecture", "master_plan"):
            lessons = by_phase[phase].context["cross_run_project_lessons"]
            self.assertEqual(lessons[0]["title"], "Avoid shallow visual acceptance")
            self.assertEqual(lessons[0]["confidence"], 0.91)
        self.assertEqual([phase for phase, _query in state.lookups], ["goal_spec", "architecture", "master_plan"])

    def test_waves_are_dependency_safe_and_write_disjoint(self):
        items = [
            Item("A", write_paths=("src/shared",), order=1),
            Item("B", write_paths=("src/shared/file.py",), order=2),
            Item("C", write_paths=("src/other",), order=3),
            Item("D", depends_on=("A",), write_paths=("tests",), order=4),
        ]
        scheduler = DeterministicWaveScheduler(ExecutionClass.CLOUD, cloud_default=4)
        report = scheduler.run(items, lambda item: item.id)

        self.assertEqual(report.waves[0], ("A", "C"))
        self.assertNotIn("B", report.waves[0])
        self.assertGreaterEqual(len(report.waves), 2)
        self.assertTrue(report.successful)

    def test_prewrite_hash_conflict_blocks_before_worker(self):
        called: list[str] = []
        leases = ResourceLeaseManager(lambda _path: "new-hash")
        scheduler = DeterministicWaveScheduler(ExecutionClass.LOCAL, leases=leases)
        report = scheduler.run(
            [
                Item(
                    "A",
                    write_paths=("src/a.py",),
                    pre_write_hashes={"src/a.py": "old-hash"},
                )
            ],
            lambda item: called.append(item.id),
        )

        self.assertEqual(called, [])
        self.assertEqual(report.outcomes[0].status, ScheduleStatus.CONFLICT)
        self.assertIn("pre-write hash changed", report.outcomes[0].error)

    def test_rate_limit_retries_and_adapts_cloud_concurrency(self):
        attempts = Counter()
        adaptive = AdaptiveConcurrency(ExecutionClass.CLOUD, cloud_default=4, recover_after=99)
        scheduler = DeterministicWaveScheduler(
            ExecutionClass.CLOUD,
            adaptive=adaptive,
            rate_limit_retries=2,
        )

        def worker(item):
            attempts[item.id] += 1
            if item.id == "A" and attempts[item.id] == 1:
                raise RateLimitError("429")
            return item.id

        report = scheduler.run(
            [Item("A", write_paths=("a",)), Item("B", write_paths=("b",))],
            worker,
        )

        self.assertTrue(report.successful)
        self.assertEqual(attempts["A"], 2)
        self.assertEqual(adaptive.current, 3)


class UltraEngineTests(unittest.TestCase):
    def test_live_progress_events_include_real_graph_counts_and_current_assignment(self):
        events = EventBus()
        captured = []
        events.subscribe(captured.append)
        modules = [module("M1", "src/one")]
        factory = FakeFactory(lambda request: standard_handler(request, modules=modules))
        engine = UltraOrchestrator(
            factory,
            execution_class=ExecutionClass.LOCAL,
            events=events,
            config=UltraConfig(min_top_modules=1, max_top_modules=4),
        )
        plan = engine.prepare("build the whole product")
        engine.approve(plan.fingerprint)
        result = engine.run()

        self.assertTrue(result.successful)
        graph = next(item for item in captured if item.kind == "ultra.graph_ready")
        self.assertGreaterEqual(graph.data["total_nodes"], 1)
        started = next(
            item
            for item in captured
            if item.kind == "ultra.agent_started" and item.data.get("node_id")
        )
        self.assertEqual(started.data["total_nodes"], graph.data["total_nodes"])
        self.assertTrue(started.data["current_node_title"])
        completed = [
            item
            for item in captured
            if item.kind == "ultra.node" and item.data.get("status") == "completed"
        ]
        self.assertEqual(completed[-1].data["completed_nodes"], graph.data["total_nodes"])

    def test_quality_consensus_uses_typed_passed_over_contradictory_declaration(self):
        engine, _factory, _plan = prepared_engine()
        node = next(iter(engine.nodes.values()))
        records = engine._quality_vote_records(
            node,
            (
                AgentResponse(
                    payload={"passed": True, "verdict": "reject"},
                    summary="Typed review passed.",
                ),
                AgentResponse(
                    payload={"passed": False, "verdict": "accept"},
                    summary="Typed review failed.",
                ),
            ),
        )

        self.assertEqual(records[0]["verdict"], "accept")
        self.assertEqual(records[0]["evidence"]["declared_verdict"], "reject")
        self.assertEqual(records[1]["verdict"], "reject")
        self.assertEqual(records[1]["evidence"]["declared_verdict"], "accept")

    def test_agent_response_preserves_typed_envelope_reasoning_artifact(self):
        artifact = {
            "claim": "candidate passes",
            "supporting_evidence": ["review:candidate"],
            "counterarguments": ["runtime still pending"],
            "rejected_alternatives": ["contract-only review"],
            "verification_plan": ["run the candidate"],
        }
        response = AgentResponse.from_mapping(
            {
                "summary": "reviewed",
                "payload": {"passed": True},
                "reasoning_artifact": artifact,
            }
        )

        self.assertTrue(response.payload["passed"])
        self.assertEqual(response.payload["reasoning_artifact"], artifact)

    def test_small_model_reasoning_graph_ids_are_repaired_without_inventing_evidence(self):
        artifact = {
            "claim": "candidate passes",
            "supporting_evidence": ["candidate:api"],
            "counterarguments": ["browser pending"],
            "rejected_alternatives": ["contract-only review"],
            "verification_plan": ["run candidate"],
            "reasoning_graph": {
                "nodes": [
                    {"id": "choice", "summary": "candidate passes"},
                    {"id": "alt", "summary": "contract-only review"},
                ],
                "edges": [{"from": "missing", "to": "alt", "relation": "supports"}],
            },
        }
        repaired, actions = repair_reasoning_artifact_graph(artifact)
        protocol = reasoning_debate_protocol_for("reviewer", "review", {})

        self.assertTrue(actions)
        self.assertEqual(repaired["supporting_evidence"], artifact["supporting_evidence"])
        self.assertTrue(evaluate_reasoning_artifact(repaired, protocol).passed)

    def test_fix_candidate_is_composed_with_prior_implementation(self):
        base = AgentResponse(
            payload={
                "implementation": {"code": "class Environment {}", "api": {"setup": True}},
                "evidence": [{"id": "initial"}],
            },
            summary="initial",
        )
        fix = AgentResponse(
            payload={
                "implementation": {"api": {"update": True}},
                "evidence": [{"id": "fix"}],
            },
            summary="fixed",
        )

        combined = UltraOrchestrator._merge_candidate_response(base, fix)

        self.assertEqual(combined.payload["implementation"]["code"], "class Environment {}")
        self.assertEqual(
            combined.payload["implementation"]["api"],
            {"setup": True, "update": True},
        )
        self.assertEqual(combined.payload["evidence"], [{"id": "fix"}])

    def test_single_html_artifact_uses_recursive_specialists_and_parent_packages(self):
        state = InMemoryUltraState()
        html_module = module("GAME", "index.html")
        html_module.update(
            {
                "title": "Three.js vehicle game",
                "objective": "Build a polished 3D vehicle, road, character, gameplay, and presentation",
            }
        )
        engine, factory, _plan = prepared_engine(
            state=state,
            modules=[html_module],
        )

        result = engine.run()

        self.assertTrue(result.successful)
        root = next(node for node in engine.nodes.values() if node.parent_id is None)
        children = [node for node in engine.nodes.values() if node.parent_id]
        direct_children = [node for node in children if node.parent_id == root.id]
        recursive_leaves = [node for node in children if node.parent_id != root.id]
        self.assertGreaterEqual(len(children), 6)
        self.assertEqual(len(direct_children), 6)
        self.assertGreaterEqual(len(recursive_leaves), 18)
        self.assertTrue(all(node.contract.metadata.get("component_leaf") for node in recursive_leaves))
        self.assertTrue(all(not node.contract.write_paths for node in children))
        self.assertTrue(
            all(node.contract.metadata.get("component_package_only") for node in children)
        )
        run_id = engine.run_state.id
        self.assertGreaterEqual(len(state.specialists[run_id]), len(children) + 1)
        self.assertTrue(all(child.id in state.component_packages[run_id] for child in children))
        assembler_requests = [
            request
            for request in factory.requests
            if request.phase == InnerPhase.INTEGRATE.value
            and request.task.get("final_assembler")
        ]
        self.assertEqual(len(assembler_requests), 1)
        self.assertEqual(
            set(assembler_requests[0].task["child_component_packages"]),
            {child.id for child in direct_children},
        )
        component_assemblers = [
            request
            for request in factory.requests
            if request.phase == InnerPhase.INTEGRATE.value
            and request.task.get("component_assembler")
        ]
        self.assertGreaterEqual(len(component_assemblers), len(direct_children))
        self.assertTrue(all(not request.task.get("final_assembler") for request in component_assemblers))
        review_requests = [
            request
            for request in factory.requests
            if request.phase in {InnerPhase.REVIEW.value, InnerPhase.TEST.value}
            and request.task.get("fresh_review", request.task.get("fresh_test_context", False))
        ]
        self.assertTrue(review_requests)
        self.assertTrue(all("candidate" in request.task for request in review_requests))

    def test_foundation_is_sequential_fingerprint_bound_and_traced_without_cot(self):
        state = InMemoryUltraState()
        engine, factory, plan = prepared_engine(state=state)

        self.assertEqual(engine.phase, UltraPhase.AWAITING_APPROVAL)
        self.assertTrue(engine.run_state.approved)
        self.assertEqual(
            [request.phase for request in factory.requests[:6]],
            [
                "goal_spec",
                "architecture",
                "architecture",
                "architecture_critique",
                "architecture_judge",
                "master_plan",
            ],
        )
        self.assertEqual(len(factory.created), 6)
        self.assertEqual(plan.fingerprint, engine.run_state.master_fingerprint)
        self.assertTrue(all("chain-of-thought" not in trace.reasoning_summary for trace in state.traces))
        self.assertTrue(all(not hasattr(trace, "chain_of_thought") for trace in state.traces))

    def test_local_and_cloud_execute_identical_pipeline_with_different_parallelism(self):
        local, local_factory, _ = prepared_engine(ExecutionClass.LOCAL)
        cloud, cloud_factory, _ = prepared_engine(ExecutionClass.CLOUD)

        local_result = local.run()
        cloud_result = cloud.run()

        self.assertTrue(local_result.successful)
        self.assertTrue(cloud_result.successful)
        self.assertEqual(local_result.schedule.peak_concurrency, 1)
        self.assertEqual(cloud_result.schedule.peak_concurrency, 4)
        local_pipeline = Counter(request.phase for request in local_factory.requests)
        cloud_pipeline = Counter(request.phase for request in cloud_factory.requests)
        self.assertEqual(local_pipeline, cloud_pipeline)
        self.assertEqual(
            [result.node_id.split(".", 1)[-1] for result in local_result.node_results],
            [result.node_id.split(".", 1)[-1] for result in cloud_result.node_results],
        )
        self.assertNotEqual(
            [result.node_id for result in local_result.node_results],
            [result.node_id for result in cloud_result.node_results],
        )

    def test_dynamic_children_inherit_contract_and_out_of_scope_child_requires_revision(self):
        parent_module = [module("M1", "src/feature")]

        def contained(request: AgentRequest) -> AgentResponse:
            response = standard_handler(request, modules=parent_module)
            if request.phase == InnerPhase.DECOMPOSE.value and request.node_id.endswith(".M001"):
                return AgentResponse(
                    payload={
                        "children": [
                            module("M1.child", "src/feature/child.py"),
                        ]
                    },
                    summary="contained child",
                )
            return response

        engine, _, _ = prepared_engine(handler=contained, modules=parent_module)
        result = engine.run()
        self.assertTrue(result.successful)
        parent_id = next(node_id for node_id in engine.nodes if node_id.endswith(".M001"))
        child_id = engine.nodes[parent_id].children[0]
        child = engine.nodes[child_id]
        self.assertEqual(child.parent_id, parent_id)
        self.assertIn("do not change public scope", child.contract.forbidden_changes)
        self.assertIn(child_id, engine.nodes[parent_id].depends_on)

        def escaped(request: AgentRequest) -> AgentResponse:
            response = standard_handler(request, modules=parent_module)
            if request.phase == InnerPhase.DECOMPOSE.value and request.node_id.endswith(".M001"):
                return AgentResponse(
                    payload={"children": [module("M1.escape", "outside/file.py")]},
                    summary="bad child",
                )
            return response

        escaped_engine, _, _ = prepared_engine(handler=escaped, modules=parent_module)
        escaped_result = escaped_engine.run()
        self.assertFalse(escaped_result.successful)
        self.assertEqual(escaped_engine.phase, UltraPhase.REVISION_REQUIRED)

    def test_quality_loop_is_bounded_then_requests_replan(self):
        modules = [module("M1", "src/one")]
        phases = Counter()

        def failing_quality(request: AgentRequest) -> AgentResponse:
            phases[request.phase] += 1
            response = standard_handler(request, modules=modules)
            if request.phase in {InnerPhase.REVIEW.value, InnerPhase.TEST.value}:
                return AgentResponse(
                    payload={"passed": False, "issues": ["still broken"]},
                    summary="quality failed",
                )
            if request.phase == InnerPhase.REPLAN.value:
                return AgentResponse(payload={"revision": "change approach"}, summary="replan now")
            return response

        engine, _, _ = prepared_engine(handler=failing_quality, modules=modules)
        result = engine.run()

        self.assertFalse(result.successful)
        self.assertEqual(engine.phase, UltraPhase.REVISION_REQUIRED)
        self.assertEqual(phases[InnerPhase.FIX.value], 3)
        self.assertEqual(phases[InnerPhase.REPLAN.value], 1)
        self.assertEqual(result.node_results[0].fix_attempts, 3)
        self.assertEqual(
            result.node_results[0].component_package["status"],
            "best_candidate_below_target",
        )
        self.assertEqual(
            result.node_results[0].component_package["replan"]["revision"],
            "change approach",
        )

    def test_replan_refines_contract_without_expanding_scope(self):
        original = TaskContractV1.from_mapping(module("M1", "src/one"))
        refined = UltraOrchestrator._refine_contract_from_replan(
            original,
            {
                "reasoning_artifact": {
                    "claim": "The integration contract must expose `masterUpdate(deltaTime): void`.",
                    "findings": [
                        "The current package has no stable parent-callable update entrypoint."
                    ],
                    "verification_plan": [
                        "Call `masterUpdate(deltaTime): void` twice and verify deterministic state."
                    ],
                }
            },
        )

        self.assertIn("masterUpdate(deltaTime): void", refined.owned_interfaces)
        self.assertIn(
            "Call `masterUpdate(deltaTime): void` twice and verify deterministic state.",
            refined.verification,
        )
        self.assertTrue(
            any("integration contract" in item for item in refined.acceptance_criteria)
        )
        self.assertEqual(refined.write_paths, original.write_paths)
        self.assertEqual(refined.depends_on, original.depends_on)
        self.assertEqual(refined.forbidden_changes, original.forbidden_changes)
        self.assertTrue(refined.metadata["replan_refinement_requirements"])

    def test_background_cancel_stops_at_safe_checkpoint(self):
        modules = [module("M1", "src/one")]
        entered = threading.Event()
        release = threading.Event()

        def blocking(request: AgentRequest) -> AgentResponse:
            if request.phase == InnerPhase.IMPLEMENT.value:
                entered.set()
                release.wait(timeout=2)
            return standard_handler(request, modules=modules)

        engine, _, _ = prepared_engine(handler=blocking, modules=modules)
        engine.start_background()
        self.assertTrue(entered.wait(timeout=2))
        engine.cancel()
        release.set()
        result = engine.background.result(timeout=3)

        self.assertFalse(result.successful)
        self.assertEqual(engine.phase, UltraPhase.CANCELLED)
        self.assertTrue(engine.control.cancelled)


if __name__ == "__main__":
    unittest.main()
