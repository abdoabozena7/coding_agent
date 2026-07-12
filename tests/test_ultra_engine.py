from __future__ import annotations

import threading
import time
import unittest
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from agent.scheduler import (
    AdaptiveConcurrency,
    DeterministicWaveScheduler,
    ExecutionClass,
    RateLimitError,
    ResourceLeaseManager,
    ScheduleStatus,
)
from agent.ultra import (
    AgentRequest,
    AgentResponse,
    AgentRole,
    InMemoryUltraState,
    InnerPhase,
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
    def test_foundation_is_sequential_fingerprint_bound_and_traced_without_cot(self):
        state = InMemoryUltraState()
        engine, factory, plan = prepared_engine(state=state)

        self.assertEqual(engine.phase, UltraPhase.AWAITING_APPROVAL)
        self.assertTrue(engine.run_state.approved)
        self.assertEqual(
            [request.phase for request in factory.requests[:3]],
            ["goal_spec", "architecture", "master_plan"],
        )
        self.assertEqual(len(factory.created), 3)
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
            [result.node_id for result in local_result.node_results],
            [result.node_id for result in cloud_result.node_results],
        )

    def test_dynamic_children_inherit_contract_and_out_of_scope_child_requires_revision(self):
        parent_module = [module("M1", "src/feature")]

        def contained(request: AgentRequest) -> AgentResponse:
            response = standard_handler(request, modules=parent_module)
            if request.phase == InnerPhase.DECOMPOSE.value and request.node_id == "M1":
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
        child = engine.nodes["M1.child"]
        self.assertEqual(child.parent_id, "M1")
        self.assertIn("do not change public scope", child.contract.forbidden_changes)
        self.assertIn("M1.child", engine.nodes["M1"].depends_on)

        def escaped(request: AgentRequest) -> AgentResponse:
            response = standard_handler(request, modules=parent_module)
            if request.phase == InnerPhase.DECOMPOSE.value and request.node_id == "M1":
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
