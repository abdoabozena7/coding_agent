from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from pathlib import Path
import sqlite3
import tempfile
import unittest
import zlib

from agent.models import GoalStatus, RoleProfile, Task
from agent.evaluation import analyze_benchmark_trend, learn_from_benchmark_trend, record_benchmark_trend
from agent.model_catalog import ExecutionClass as CatalogExecutionClass, ModelDescriptor
from agent.project_brain import ProjectBrain
from agent.sandbox import AccessLevel as SandboxAccessLevel
from agent.store import (
    ConcurrentBrainUpdateError,
    LeaseConflictError,
    StateStore,
)
from agent.swarm_bus import SwarmBus
from agent.swarm_coordinator import SwarmCoordinator
from agent.swarm_protocol import ConsensusStatus, ConsensusVoteV1, SwarmMessageType, SwarmMessageV1
from agent.ultra_models import (
    AgentRun,
    AgentRunStatus,
    ArchitectureSpecV1,
    Artifact,
    BrainSection,
    ContractScopeError,
    ExecutionClass,
    GoalSpecV1,
    InsightV1,
    LeaseStatus,
    PromptTraceV1,
    ResultPackageV1,
    TaskContractV1,
    UltraRun,
    UltraRunStatus,
    WorkNode,
    WorkNodeKind,
    WorkNodeStatus,
    contract_scope_violations,
)
from agent.models import utc_now
from agent.sandbox import AccessLevel as PermissionAccessLevel
from agent.ultra import ResultPackageV1 as EngineResult, UltraConfig
from agent.ultra_session import StateStoreUltraAdapter
from agent.workflow import AgentRegistryEntryV1, AgentState


def task(task_id: str, *, depends_on: tuple[str, ...] = ()) -> Task:
    return Task(
        id=task_id,
        title=f"Module {task_id}",
        description=f"Implement the {task_id} module",
        depends_on=depends_on,
        acceptance_criteria=(f"{task_id} behavior works",),
        verification=(f"test {task_id}",),
        role=RoleProfile(name="module coder", mission=f"Implement {task_id}"),
    )


def plan_basis(*task_ids: str) -> dict[str, object]:
    return {
        "applicability_evidence": tuple(
            {
                "fact": f"Workspace supports {task_id}",
                "source": "repo inspection",
                "supports_tasks": [task_id],
            }
            for task_id in task_ids
        ),
        "execution_strategy": "Implement modules in dependency order.",
        "expected_changes": tuple(
            {
                "path": f"src/{task_id.lower()}/",
                "intent": f"Implement {task_id}",
                "supports_tasks": [task_id],
            }
            for task_id in task_ids
        ),
    }


class UltraStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp.name)
        self.store = StateStore(self.workspace)
        self.goal = self.store.create_goal(
            "Build a complete game",
            constraints=("Do not change the public save format",),
        )
        self.store.transition_goal(self.goal.id, GoalStatus.AWAITING_PLAN_APPROVAL)
        self.plan = self.store.create_plan(
            self.goal.id,
            "Build core and physics modules",
            [task("CORE"), task("PHYSICS", depends_on=("CORE",))],
            **plan_basis("CORE", "PHYSICS"),
        )
        self.plan, _ = self.store.approve_plan(
            self.goal.id,
            self.plan.revision,
            expected_fingerprint=self.plan.fingerprint,
        )
        self.goal_spec = GoalSpecV1(
            objective="Build a complete game",
            scope=("game runtime",),
            success_criteria=("all game tests pass",),
            answered_questions={"platform": "desktop"},
        )
        self.architecture = ArchitectureSpecV1(
            summary="Modular game architecture",
            components=({"name": "core"}, {"name": "physics"}),
            interfaces={"GameLoop": {"tick": "float -> None"}},
        )
        self.run = self.store.create_ultra_run(
            UltraRun(
                goal_id=self.goal.id,
                provider="ollama",
                model="qwen3-coder",
                execution_class=ExecutionClass.LOCAL,
                concurrency=1,
                goal_spec=self.goal_spec,
                architecture_spec=self.architecture,
                config={"max_depth": 5, "max_nodes": 500, "model": {"temperature": 0.1}},
            )
        )
        self.run = self.store.approve_ultra_master(
            self.run.id, self.plan.revision, self.plan.fingerprint
        )

    def tearDown(self) -> None:
        self.store.close()
        self.temp.cleanup()

    def test_schema_v4_and_specs_survive_reopen(self) -> None:
        from agent.ultra_models import AccessLevel

        self.assertIs(ExecutionClass, CatalogExecutionClass)
        self.assertIs(AccessLevel, SandboxAccessLevel)
        connection = sqlite3.connect(self.store.path)
        try:
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 7)
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type IN ('table','view')"
                )
            }
        finally:
            connection.close()
        self.assertTrue(
            {
                "ultra_runs",
                "work_nodes",
                "agent_runs",
                "brain_entries",
                "artifacts",
                "prompt_traces",
                "memory_access",
                "resource_leases",
                "swarm_messages",
                "consensus_rounds",
                "consensus_votes",
            }.issubset(tables)
        )
        run_id = self.run.id
        self.store.close()
        self.store = StateStore(self.workspace)
        restored = self.store.get_ultra_run(run_id)
        self.assertEqual(restored.goal_spec.fingerprint, self.goal_spec.fingerprint)
        self.assertEqual(restored.architecture_spec.interfaces, self.architecture.interfaces)
        self.assertEqual(restored.config["model"]["temperature"], 0.1)

    def test_dynamic_child_must_remain_inside_approved_module_contract(self) -> None:
        modules = self.store.sync_master_modules(self.run.id)
        core = next(item for item in modules if item.master_task_id == "CORE")
        self.assertTrue(core.is_master_module)
        child_contract = TaskContractV1(
            objective="Implement loop",
            success_criteria=("loop works",),
            write_paths=("src/core/loop.py",),
            forbidden_changes=core.contract.forbidden_changes,
            interfaces={"GameLoop": self.architecture.interfaces["GameLoop"]},
        )
        child = self.store.create_work_node(
            WorkNode(
                ultra_run_id=self.run.id,
                parent_id=core.id,
                title="Game loop",
                objective="Implement loop",
                contract=child_contract,
            )
        )
        self.assertEqual(child.depth, 1)
        self.assertEqual(child.master_task_id, "CORE")

        escaped = replace(child_contract, write_paths=("src/physics/motor.py",))
        self.assertTrue(contract_scope_violations(core.contract, escaped))
        with self.assertRaises(ContractScopeError):
            self.store.create_work_node(
                WorkNode(
                    ultra_run_id=self.run.id,
                    parent_id=core.id,
                    title="Out of scope",
                    objective="Change physics from core",
                    contract=escaped,
                )
            )

    def test_work_node_completion_and_agent_result_are_durable(self) -> None:
        module = self.store.sync_master_modules(self.run.id)[0]
        child = self.store.create_work_node(
            WorkNode(
                ultra_run_id=self.run.id,
                parent_id=module.id,
                title="Implementation",
                objective="Implement module",
                contract=TaskContractV1(
                    objective="Implement module",
                    write_paths=module.contract.write_paths,
                    forbidden_changes=module.contract.forbidden_changes,
                ),
            )
        )
        child = self.store.transition_work_node(child.id, WorkNodeStatus.IN_PROGRESS)
        agent = self.store.create_agent_run(
            AgentRun(
                ultra_run_id=self.run.id,
                work_node_id=child.id,
                role="coder",
                provider="ollama",
                model="qwen3-coder",
                phase="implement",
                status=AgentRunStatus.RUNNING,
            )
        )
        package = ResultPackageV1(
            summary="Implemented module",
            changed_files=("src/core/main.py",),
            tests=({"name": "unit", "passed": True},),
            insights=(InsightV1(summary="Keep loop deterministic"),),
        )
        self.store.update_agent_run(agent.id, AgentRunStatus.COMPLETED, result=package)
        completed = self.store.transition_work_node(
            child.id, WorkNodeStatus.COMPLETED, result=package
        )
        self.assertEqual(completed.result.changed_files, ("src/core/main.py",))
        self.assertEqual(self.store.get_agent_run(agent.id).result.tests[0]["passed"], True)

    def test_project_brain_versions_searches_and_builds_focused_context(self) -> None:
        module = self.store.sync_master_modules(self.run.id)[0]
        brain = ProjectBrain(self.store, self.run.id)
        brain.set_north_star(self.goal_spec)
        brain.set_architecture(self.architecture)
        first = brain.record_decision(
            "Physics numeric type", "Use float for torque", reason="Engine convention"
        )
        second = brain.record_decision(
            "Physics numeric type",
            "Use double for torque",
            reason="Precision tests",
            expected_version=1,
        )
        self.assertEqual((first.version, second.version), (1, 2))
        with self.assertRaises(ConcurrentBrainUpdateError):
            brain.record_decision(
                "Physics numeric type", "Use decimal", expected_version=1
            )
        brain.remember_for_role(
            "coder", "Preferred pattern", "Use deterministic fixed updates", work_node_id=module.id
        )
        self.assertEqual(brain.search("double")[0].id, second.id)

        lesson = brain.record_lesson(
            "Core deterministic loop",
            "Core modules should use deterministic fixed updates after runtime failures.",
            work_node_id=module.id,
        )
        promoted = self.store.promote_brain_entry_to_project_memory(
            lesson.id,
            confidence=0.82,
            evidence_refs=("unit:test",),
        )
        knowledge = brain.record_knowledge(
            "Core runtime platform",
            "Core runtime uses deterministic clock ticks and fixed-step scheduling.",
            work_node_id=module.id,
            confidence=0.76,
            evidence_refs=("doc:runtime",),
        )
        self.assertEqual(promoted["confidence"], 0.82)
        self.assertEqual(promoted["evidence_refs"], ["unit:test"])
        self.assertTrue(
            self.store.search_project_memory(
                "deterministic runtime",
                section=BrainSection.LESSON,
            )
        )

        package = brain.build_context(module.id, "coder", budget_chars=20_000)
        self.assertIn("task", package.sections)
        self.assertIn("architecture", package.sections)
        self.assertIn("decisions", package.sections)
        self.assertIn("project_lessons", package.sections)
        self.assertIn("project_knowledge", package.sections)
        self.assertEqual(package.sections["project_lessons"][0]["title"], "Core deterministic loop")
        self.assertEqual(package.sections["project_knowledge"][0]["title"], "Core runtime platform")
        self.assertEqual(
            self.store.search_project_memory("fixed-step scheduling", section=BrainSection.KNOWLEDGE)[0]["source_brain_entry_id"],
            knowledge.id,
        )
        self.assertLessEqual(package.size_chars, 20_000)
        self.assertTrue(self.store.list_memory_access(self.run.id))

        # The deterministic LIKE fallback is a supported runtime path.
        self.store._fts5_available = False
        self.assertEqual(brain.search("double")[0].id, second.id)

    def test_foundation_project_lessons_are_reused_before_planning(self) -> None:
        module = self.store.sync_master_modules(self.run.id)[0]
        brain = ProjectBrain(self.store, self.run.id)
        lesson = brain.record_lesson(
            "Browser evidence before completion",
            "Require real browser evidence before accepting visual work.",
            work_node_id=module.id,
        )
        memory = self.store.promote_brain_entry_to_project_memory(
            lesson.id,
            confidence=0.9,
            evidence_refs=("bench:browser",),
        )
        adapter = StateStoreUltraAdapter(
            self.store,
            self.goal.id,
            ModelDescriptor("ollama", "gemma4", CatalogExecutionClass.LOCAL),
            PermissionAccessLevel.NORMAL,
            UltraConfig(),
        )
        adapter.run_id = self.run.id

        lessons = adapter.foundation_project_lessons(
            self.run.id,
            "browser visual completion",
            phase="master_plan",
        )

        self.assertEqual(lessons[0]["title"], "Browser evidence before completion")
        self.assertEqual(lessons[0]["phase"], "master_plan")
        restored = self.store.search_project_memory("browser visual completion")[0]
        self.assertEqual(restored["id"], memory["id"])
        self.assertEqual(restored["reuse_count"], 1)

    def test_foundation_project_context_reuses_cross_run_knowledge(self) -> None:
        module = self.store.sync_master_modules(self.run.id)[0]
        brain = ProjectBrain(self.store, self.run.id)
        memory_entry = brain.record_knowledge(
            "Browser runtime invariant",
            "Browser visual completion requires console-clean runtime evidence.",
            work_node_id=module.id,
            confidence=0.88,
            evidence_refs=("bench:console-clean",),
        )
        adapter = StateStoreUltraAdapter(
            self.store,
            self.goal.id,
            ModelDescriptor("ollama", "gemma4", CatalogExecutionClass.LOCAL),
            PermissionAccessLevel.NORMAL,
            UltraConfig(),
        )
        adapter.run_id = self.run.id

        context = adapter.foundation_project_lessons(
            self.run.id,
            "browser console runtime completion",
            phase="architecture",
        )

        self.assertEqual(context[0]["title"], "Browser runtime invariant")
        self.assertEqual(context[0]["section"], BrainSection.KNOWLEDGE.value)
        restored = self.store.search_project_memory("console-clean runtime", section=BrainSection.KNOWLEDGE)[0]
        self.assertEqual(restored["source_brain_entry_id"], memory_entry.id)
        self.assertEqual(restored["reuse_count"], 1)

    def test_global_evaluation_outcome_reinforces_used_project_lessons(self) -> None:
        module = self.store.sync_master_modules(self.run.id)[0]
        brain = ProjectBrain(self.store, self.run.id)
        lesson = brain.record_lesson(
            "Browser evidence before completion",
            "Require real browser evidence before accepting visual work.",
            work_node_id=module.id,
        )
        memory = self.store.promote_brain_entry_to_project_memory(
            lesson.id,
            confidence=0.72,
            evidence_refs=("bench:seed",),
        )
        adapter = StateStoreUltraAdapter(
            self.store,
            self.goal.id,
            ModelDescriptor("ollama", "gemma4", CatalogExecutionClass.LOCAL),
            PermissionAccessLevel.NORMAL,
            UltraConfig(),
        )
        adapter.run_id = self.run.id

        adapter.foundation_project_lessons(
            self.run.id,
            "browser visual completion",
            phase="master_plan",
        )
        gate = adapter.record_global_evaluation_gate(
            EngineResult(
                node_id="global",
                success=True,
                summary="final evidence passed",
                evidence=({"kind": "browser", "passed": True},),
            ),
            (),
        )

        self.assertTrue(gate["passed"])
        self.assertEqual(gate["project_lesson_outcomes"][0]["id"], memory["id"])
        restored = self.store.search_project_memory("browser visual completion")[0]
        self.assertEqual(restored["metadata"]["positive_outcomes"], 1)
        self.assertGreater(restored["confidence"], memory["confidence"])
        self.assertIn(f"benchmark:{gate['benchmark_id']}", restored["evidence_refs"])

    def test_failed_global_evaluation_penalizes_used_project_lessons(self) -> None:
        module = self.store.sync_master_modules(self.run.id)[0]
        brain = ProjectBrain(self.store, self.run.id)
        lesson = brain.record_lesson(
            "Weak visual shortcut",
            "Accept visual work when an index.html exists.",
            work_node_id=module.id,
        )
        memory = self.store.promote_brain_entry_to_project_memory(
            lesson.id,
            confidence=0.78,
            evidence_refs=("bench:seed",),
        )
        adapter = StateStoreUltraAdapter(
            self.store,
            self.goal.id,
            ModelDescriptor("ollama", "gemma4", CatalogExecutionClass.LOCAL),
            PermissionAccessLevel.NORMAL,
            UltraConfig(),
        )
        adapter.run_id = self.run.id

        adapter.foundation_project_lessons(
            self.run.id,
            "weak visual shortcut index.html",
            phase="architecture",
        )
        gate = adapter.record_global_evaluation_gate(
            EngineResult(
                node_id="global",
                success=False,
                summary="final review rejected shallow output",
                evidence=({"kind": "review", "passed": False},),
                findings=("visual benchmark failed",),
            ),
            (),
        )

        self.assertFalse(gate["passed"])
        restored = self.store.search_project_memory("weak visual shortcut", min_confidence=0.0)[0]
        self.assertEqual(restored["id"], memory["id"])
        self.assertEqual(restored["metadata"]["negative_outcomes"], 1)
        self.assertLess(restored["confidence"], memory["confidence"])
        self.assertIn("global integration/review/final evidence did not pass", restored["metadata"]["last_outcome"]["reason"])

    def test_failed_global_evaluation_records_cross_run_remediation_knowledge(self) -> None:
        adapter = StateStoreUltraAdapter(
            self.store,
            self.goal.id,
            ModelDescriptor("ollama", "gemma4", CatalogExecutionClass.LOCAL),
            PermissionAccessLevel.NORMAL,
            UltraConfig(),
        )
        adapter.run_id = self.run.id

        gate = adapter.record_global_evaluation_gate(
            EngineResult(
                node_id="global",
                success=True,
                summary="model claimed completion without proof",
                evidence=(),
                test_results=(),
            ),
            (),
        )

        self.assertFalse(gate["passed"])
        self.assertIn("durable evidence", gate["blocker"])
        remediation = gate["remediation_knowledge"]
        self.assertTrue(remediation["recorded"])
        self.assertTrue(remediation["brain_entry_id"])
        self.assertTrue(any("final evidence" in step for step in remediation["remediation_steps"]))
        memories = self.store.search_project_memory(
            "global remediation durable evidence final evidence benchmark",
            section=BrainSection.KNOWLEDGE,
        )
        self.assertTrue(memories)
        self.assertEqual(memories[0]["title"], remediation["title"])
        self.assertIn(f"benchmark:{gate['benchmark_id']}", memories[0]["evidence_refs"])
        self.store.update_ultra_run(
            self.run.id,
            status=UltraRunStatus.BLOCKED,
            error="closed failed run before starting cross-run retrieval check",
        )

        next_run = self.store.create_ultra_run(
            UltraRun(
                goal_id=self.goal.id,
                provider="ollama",
                model="qwen3-coder",
                execution_class=ExecutionClass.LOCAL,
                concurrency=1,
                goal_spec=self.goal_spec,
                architecture_spec=self.architecture,
                config={"max_depth": 5, "max_nodes": 500},
            )
        )
        next_run = self.store.approve_ultra_master(
            next_run.id, self.plan.revision, self.plan.fingerprint
        )
        next_adapter = StateStoreUltraAdapter(
            self.store,
            self.goal.id,
            ModelDescriptor("ollama", "gemma4", CatalogExecutionClass.LOCAL),
            PermissionAccessLevel.NORMAL,
            UltraConfig(),
        )
        next_adapter.run_id = next_run.id

        context = next_adapter.foundation_project_lessons(
            next_run.id,
            "final evidence global benchmark remediation",
            phase="master_plan",
        )

        self.assertTrue(any(item["id"] == memories[0]["id"] for item in context))
        self.assertTrue(any("Required remediation steps" in item["content"] for item in context))

    def test_global_evaluation_gate_records_benchmark_trend_automatically(self) -> None:
        self.store.record_benchmark_result(
            suite_name="ultra-automatic-evaluation",
            scenario_name="global-completion-gate",
            provider="ollama",
            model="gemma4",
            ultra_run_id=self.run.id,
            metrics={"input_tokens": 1000, "node_successes": 2},
            scores={"global_success": 1.0, "final_evidence_score": 1.0},
            result="passed",
        )
        adapter = StateStoreUltraAdapter(
            self.store,
            self.goal.id,
            ModelDescriptor("ollama", "gemma4", CatalogExecutionClass.LOCAL),
            PermissionAccessLevel.NORMAL,
            UltraConfig(),
        )
        adapter.run_id = self.run.id

        gate = adapter.record_global_evaluation_gate(
            EngineResult(
                node_id="global",
                success=False,
                summary="regressed final gate",
                evidence=(),
                findings=("no durable evidence",),
            ),
            (),
        )

        self.assertFalse(gate["passed"])
        self.assertTrue(gate["benchmark_trend_id"])
        trends = self.store.list_benchmark_results(
            suite_name="benchmark-trend",
            scenario_name="ultra-automatic-evaluation/global-completion-gate",
        )
        self.assertEqual(trends[0]["id"], gate["benchmark_trend_id"])
        self.assertEqual(trends[0]["result"], "failed")
        self.assertEqual(trends[0]["scores"]["regression"], 1.0)
        self.assertEqual(trends[0]["inputs"]["latest_id"], gate["benchmark_id"])
        learning = gate["benchmark_trend_learning"]
        self.assertTrue(learning["recorded"])
        self.assertEqual(learning["verdict"], "regressed")
        self.assertTrue(learning["brain_entry_id"])
        memories = self.store.search_project_memory(
            "benchmark regression ultra automatic evaluation global completion gate",
            section=BrainSection.KNOWLEDGE,
        )
        self.assertTrue(memories)
        self.assertEqual(memories[0]["title"], learning["title"])
        self.assertIn(f"benchmark-trend:{gate['benchmark_trend_id']}", memories[0]["evidence_refs"])

    def test_global_evaluation_gate_blocks_quality_score_regression(self) -> None:
        self.store.record_benchmark_result(
            suite_name="ultra-automatic-evaluation",
            scenario_name="global-completion-gate",
            provider="ollama",
            model="gemma4",
            ultra_run_id=self.run.id,
            metrics={"input_tokens": 1000},
            scores={
                "global_success": 1.0,
                "final_evidence_score": 1.0,
                "node_success_ratio": 1.0,
            },
            result="passed",
        )
        adapter = StateStoreUltraAdapter(
            self.store,
            self.goal.id,
            ModelDescriptor("ollama", "gemma4", CatalogExecutionClass.LOCAL),
            PermissionAccessLevel.NORMAL,
            UltraConfig(),
        )
        adapter.run_id = self.run.id

        gate = adapter.record_global_evaluation_gate(
            EngineResult(
                node_id="global",
                success=True,
                summary="current gate has evidence but weaker score mix",
                evidence=({"kind": "final", "passed": True},),
            ),
            (),
        )

        self.assertFalse(gate["passed"])
        self.assertIn("quality regressed", gate["blocker"])
        trends = self.store.list_benchmark_results(
            suite_name="benchmark-trend",
            scenario_name="ultra-automatic-evaluation/global-completion-gate",
        )
        self.assertEqual(trends[0]["result"], "failed")
        self.assertLess(trends[0]["metrics"]["score_delta:node_success_ratio"], 0)

    def test_metric_only_benchmark_trend_does_not_become_quality_blocker(self) -> None:
        self.assertFalse(
            StateStoreUltraAdapter._trend_quality_regression(
                {
                    "result": "failed",
                    "metrics": {"metric_delta:input_tokens": 500.0},
                    "scores": {"regression": 1.0},
                }
            )
        )

    def test_project_memory_confidence_accumulates_and_tracks_outcomes(self) -> None:
        module = self.store.sync_master_modules(self.run.id)[0]
        brain = ProjectBrain(self.store, self.run.id)
        first = brain.record_lesson(
            "Runtime browser gate",
            "Require browser runtime evidence before accepting generated HTML.",
            work_node_id=module.id,
        )
        promoted = self.store.promote_brain_entry_to_project_memory(
            first.id,
            confidence=0.7,
            evidence_refs=("bench:first",),
        )
        second = brain.write(
            BrainSection.LESSON,
            "Runtime browser gate",
            "Require browser runtime and console evidence before accepting generated HTML.",
            work_node_id=module.id,
            expected_version=1,
        )
        reinforced = self.store.promote_brain_entry_to_project_memory(
            second.id,
            confidence=0.74,
            evidence_refs=("bench:second",),
        )

        self.assertEqual(reinforced["id"], promoted["id"])
        self.assertGreater(reinforced["confidence"], 0.74)
        self.assertEqual(reinforced["metadata"]["promotion_count"], 2)
        self.assertEqual(reinforced["evidence_refs"], ["bench:first", "bench:second"])

        positive = self.store.record_project_memory_outcome(
            reinforced["id"],
            succeeded=True,
            evidence_ref="bench:passed",
            reason="Browser gate prevented a blank completion.",
        )
        self.assertGreater(positive["confidence"], reinforced["confidence"])
        self.assertGreaterEqual(positive["effective_confidence"], positive["confidence"])
        self.assertEqual(positive["metadata"]["positive_outcomes"], 1)

        negative = self.store.record_project_memory_outcome(
            reinforced["id"],
            succeeded=False,
            evidence_ref="bench:regressed",
            reason="The reused lesson missed a console regression.",
            weight=2.0,
        )
        self.assertLess(negative["confidence"], positive["confidence"])
        self.assertEqual(negative["metadata"]["negative_outcomes"], 1)
        self.assertIn("bench:regressed", negative["evidence_refs"])

    def test_project_memory_search_uses_effective_confidence_not_keyword_only(self) -> None:
        module = self.store.sync_master_modules(self.run.id)[0]
        brain = ProjectBrain(self.store, self.run.id)
        stale = brain.record_lesson(
            "Visual completion gate",
            "Browser visual completion requires checking screenshots.",
            work_node_id=module.id,
        )
        stale_memory = self.store.promote_brain_entry_to_project_memory(
            stale.id,
            confidence=0.72,
            evidence_refs=("bench:old",),
        )
        reliable = brain.record_lesson(
            "Visual completion console gate",
            "Browser visual completion requires screenshots and console runtime evidence.",
            work_node_id=module.id,
        )
        reliable_memory = self.store.promote_brain_entry_to_project_memory(
            reliable.id,
            confidence=0.64,
            evidence_refs=("bench:new",),
        )

        self.store.record_project_memory_outcome(
            stale_memory["id"],
            succeeded=False,
            evidence_ref="bench:stale-failed",
            weight=2.0,
        )
        self.store.record_project_memory_outcome(
            reliable_memory["id"],
            succeeded=True,
            evidence_ref="bench:reliable-passed",
            weight=2.0,
        )
        self.store.record_project_memory_use(reliable_memory["id"])

        results = self.store.search_project_memory(
            "visual completion browser",
            section=BrainSection.LESSON,
            min_confidence=0.5,
        )

        self.assertEqual(results[0]["id"], reliable_memory["id"])
        self.assertTrue(all(item["effective_confidence"] >= 0.5 for item in results))
        self.assertNotIn(stale_memory["id"], [item["id"] for item in results])

    def test_benchmark_results_are_first_class_metrics(self) -> None:
        recorded = self.store.record_benchmark_result(
            suite_name="weak-model-html",
            scenario_name="threejs-single-file",
            provider="ollama",
            model="gemma4:e4b",
            ultra_run_id=self.run.id,
            inputs={"prompt_hash": "abc"},
            metrics={"tokens": 123, "wall_ms": 456},
            scores={"runtime": 0.0, "visual": 0.2},
            result="failed",
            artifact_refs=("workspace:index.html",),
            blocker="console errors",
        )
        self.assertEqual(recorded["metrics"]["tokens"], 123)
        self.assertEqual(recorded["scores"]["runtime"], 0.0)
        restored = self.store.list_benchmark_results(
            suite_name="weak-model-html",
            scenario_name="threejs-single-file",
        )
        self.assertEqual(restored[0]["artifact_refs"], ["workspace:index.html"])
        self.assertEqual(restored[0]["blocker"], "console errors")

    def test_benchmark_trend_detects_regression_and_records_comparison(self) -> None:
        self.store.record_benchmark_result(
            suite_name="repository-retrieval",
            scenario_name="hybrid-search",
            provider="ollama",
            model="gemma4",
            metrics={"mean_reciprocal_rank": 0.9, "input_tokens": 1000, "wall_ms": 500},
            scores={"accuracy_at_k": 1.0},
            result="passed",
        )
        latest = self.store.record_benchmark_result(
            suite_name="repository-retrieval",
            scenario_name="hybrid-search",
            provider="ollama",
            model="gemma4",
            metrics={"mean_reciprocal_rank": 0.6, "input_tokens": 1400, "wall_ms": 750},
            scores={"accuracy_at_k": 0.5},
            result="failed",
            blocker="missed symbol",
        )

        trend = analyze_benchmark_trend(
            self.store.list_benchmark_results(
                suite_name="repository-retrieval",
                scenario_name="hybrid-search",
            )
        )
        self.assertTrue(trend.regressed)
        self.assertEqual(trend.latest_id, latest["id"])
        self.assertIn("score:accuracy_at_k", trend.changed_keys)
        self.assertIn("metric:input_tokens", trend.changed_keys)

        recorded = record_benchmark_trend(
            self.store,
            suite_name="repository-retrieval",
            scenario_name="hybrid-search",
            provider="ollama",
            model="gemma4",
        )
        self.assertEqual(recorded["suite_name"], "benchmark-trend")
        self.assertEqual(recorded["scenario_name"], "repository-retrieval/hybrid-search")
        self.assertEqual(recorded["result"], "failed")
        self.assertEqual(recorded["scores"]["regression"], 1.0)
        self.assertEqual(recorded["inputs"]["latest_id"], latest["id"])
        self.assertEqual(recorded["inputs"]["trend"]["verdict"], "regressed")

        learned = learn_from_benchmark_trend(self.store, recorded)

        self.assertTrue(learned["recorded"])
        self.assertEqual(learned["verdict"], "regressed")
        self.assertTrue(learned["brain_entry_id"])
        memories = self.store.search_project_memory(
            "repository retrieval hybrid search regressed changed signals",
            section=BrainSection.KNOWLEDGE,
        )
        self.assertTrue(memories)
        self.assertEqual(memories[0]["title"], learned["title"])
        self.assertIn(f"benchmark-trend:{recorded['id']}", memories[0]["evidence_refs"])
        self.assertIn(f"benchmark:{latest['id']}", memories[0]["evidence_refs"])
        self.assertEqual(memories[0]["metadata"]["source"], "record_knowledge")

    def test_benchmark_trend_detects_lower_cost_improvement(self) -> None:
        self.store.record_benchmark_result(
            suite_name="ultra-automatic-evaluation",
            scenario_name="global-completion-gate",
            provider="ollama",
            model="gemma4",
            metrics={"input_tokens": 3000, "wall_ms": 2000},
            scores={"global_success": 1.0},
            result="passed",
        )
        self.store.record_benchmark_result(
            suite_name="ultra-automatic-evaluation",
            scenario_name="global-completion-gate",
            provider="ollama",
            model="gemma4",
            metrics={"input_tokens": 1800, "wall_ms": 1200},
            scores={"global_success": 1.0},
            result="passed",
        )

        trend = analyze_benchmark_trend(
            self.store.list_benchmark_results(
                suite_name="ultra-automatic-evaluation",
                scenario_name="global-completion-gate",
            )
        )

        self.assertTrue(trend.improved)
        self.assertFalse(trend.regressed)
        self.assertLess(trend.metric_deltas["input_tokens"], 0)
        self.assertLess(trend.metric_deltas["wall_ms"], 0)

    def test_benchmark_trend_learning_skips_insufficient_history(self) -> None:
        recorded = self.store.record_benchmark_result(
            suite_name="agent-readiness",
            scenario_name="structural",
            provider="ollama",
            model="gemma4",
            scores={"all_passed": 1.0},
            result="passed",
        )
        trend = record_benchmark_trend(
            self.store,
            suite_name="agent-readiness",
            scenario_name="structural",
            provider="ollama",
            model="gemma4",
        )

        learned = learn_from_benchmark_trend(self.store, trend)

        self.assertFalse(learned["recorded"])
        self.assertEqual(learned["reason"], "trend_not_actionable")
        self.assertEqual(learned["verdict"], "insufficient_history")
        self.assertFalse(
            self.store.search_project_memory(recorded["id"], section=BrainSection.KNOWLEDGE)
        )

    def test_swarm_messages_are_durable_filterable_and_acknowledged(self) -> None:
        message = self.store.post_swarm_message(
            SwarmMessageV1(
                ultra_run_id=self.run.id,
                sender_agent_id="planner-1",
                recipient_agent_id="critic-1",
                message_type=SwarmMessageType.PROPOSAL,
                topic="html-preview-gate",
                payload={"proposal": "run browser QA before completion"},
                confidence=0.74,
                correlation_id="corr-preview",
            )
        )

        self.assertEqual(message["message_type"], "proposal")
        self.assertEqual(message["payload"]["proposal"], "run browser QA before completion")
        inbox = self.store.list_swarm_messages(self.run.id, recipient_agent_id="critic-1")
        self.assertEqual([item["id"] for item in inbox], [message["id"]])
        self.assertFalse(self.store.list_swarm_messages(self.run.id, recipient_agent_id="planner-1"))

        consumed = self.store.mark_swarm_message_consumed(message["id"])
        self.assertIsNotNone(consumed["consumed_at"])
        self.assertFalse(self.store.list_swarm_messages(self.run.id, recipient_agent_id="critic-1"))
        self.assertEqual(
            self.store.list_swarm_messages(self.run.id, recipient_agent_id="critic-1", include_consumed=True)[0]["id"],
            message["id"],
        )

    def test_swarm_frame_codec_roundtrips_formal_messages(self) -> None:
        message = SwarmMessageV1(
            ultra_run_id=self.run.id,
            sender_agent_id="planner-1",
            recipient_agent_id="tester-1",
            message_type="request",
            topic="runtime-check",
            payload={"path": "index.html", "required": True},
            confidence=0.8,
            correlation_id="corr-runtime",
        )
        frame = message.encode_frame()
        self.assertTrue(frame.startswith("SWARM/1 "))
        restored = SwarmMessageV1.decode_frame(frame)
        self.assertEqual(restored.to_dict(), message.to_dict())

    def test_swarm_dsl_and_binary_frames_roundtrip_and_reject_tampering(self) -> None:
        message = SwarmMessageV1(
            ultra_run_id=self.run.id,
            sender_agent_id="planner-1",
            recipient_agent_id="tester-1",
            message_type="request",
            topic="runtime check: index.html",
            payload={"path": "index.html", "threshold": 0.8, "note": "no console errors"},
            confidence=0.8,
            correlation_id="corr-runtime",
        )
        dsl = message.encode_dsl_frame()
        self.assertTrue(dsl.startswith("SWARMDSL/1 "))
        self.assertEqual(SwarmMessageV1.decode_any_frame(dsl).to_dict(), message.to_dict())

        binary = message.encode_binary_frame()
        self.assertTrue(binary.startswith(b"SWARMBIN/1 "))
        self.assertEqual(SwarmMessageV1.decode_any_frame(binary).to_dict(), message.to_dict())
        tampered = binary[:-2] + (b"0" if binary[-1:] != b"0" else b"1")
        with self.assertRaises(ValueError):
            SwarmMessageV1.decode_any_frame(tampered)

    def test_swarm_bus_realtime_publish_persists_and_filters_subscribers(self) -> None:
        bus = SwarmBus(self.store)
        tester_seen: list[dict[str, object]] = []
        critic_seen: list[dict[str, object]] = []
        bus.subscribe(
            self.run.id,
            tester_seen.append,
            recipient_agent_id="tester-1",
            topic_prefix="runtime",
            message_type=SwarmMessageType.REQUEST,
        )
        unsubscribe = bus.subscribe(
            self.run.id,
            critic_seen.append,
            recipient_agent_id="critic-1",
        )

        delivered = bus.publish(
            ultra_run_id=self.run.id,
            sender_agent_id="planner-1",
            recipient_agent_id="tester-1",
            message_type=SwarmMessageType.REQUEST,
            topic="runtime-check",
            payload={"path": "index.html"},
        )

        self.assertEqual(tester_seen[0]["id"], delivered["id"])
        self.assertFalse(critic_seen)
        self.assertEqual(
            self.store.list_swarm_messages(self.run.id, recipient_agent_id="tester-1")[0]["id"],
            delivered["id"],
        )
        unsubscribe()

    def test_swarm_bus_accepts_dsl_and_binary_wire_frames(self) -> None:
        bus = SwarmBus(self.store)
        tester_seen: list[dict[str, object]] = []
        bus.subscribe(self.run.id, tester_seen.append, recipient_agent_id="tester-1")
        messages = (
            SwarmMessageV1(
                ultra_run_id=self.run.id,
                sender_agent_id="planner-1",
                recipient_agent_id="tester-1",
                message_type=SwarmMessageType.REQUEST,
                topic="runtime-check",
                payload={"path": "index.html"},
            ),
            SwarmMessageV1(
                ultra_run_id=self.run.id,
                sender_agent_id="critic-1",
                recipient_agent_id="tester-1",
                message_type=SwarmMessageType.INFORM,
                topic="quality-note",
                payload={"note": "inspect console errors"},
            ),
        )

        dsl_frame, binary_frame = SwarmBus.frames(messages, wire_format="dsl")[0], SwarmBus.frames(messages[1:], wire_format="binary")[0]
        first = bus.publish_frame(dsl_frame)
        second = bus.publish_frame(binary_frame)

        self.assertEqual([item["id"] for item in tester_seen], [first["id"], second["id"]])
        inbox = self.store.list_swarm_messages(self.run.id, recipient_agent_id="tester-1")
        self.assertEqual([item["id"] for item in inbox], [first["id"], second["id"]])

    def test_swarm_bus_drains_durable_inbox_and_acknowledges(self) -> None:
        self.store.post_swarm_message(
            ultra_run_id=self.run.id,
            sender_agent_id="planner-1",
            recipient_agent_id="tester-1",
            message_type=SwarmMessageType.REQUEST,
            topic="visual-check",
            payload={"threshold": 0.8},
        )
        bus = SwarmBus(self.store)
        seen: list[dict[str, object]] = []
        bus.subscribe(self.run.id, seen.append, recipient_agent_id="tester-1")
        drained = bus.drain(self.run.id, recipient_agent_id="tester-1")

        self.assertEqual(len(drained), 1)
        self.assertEqual(seen[0]["topic"], "visual-check")
        self.assertFalse(self.store.list_swarm_messages(self.run.id, recipient_agent_id="tester-1"))

    def test_consensus_leader_election_uses_stable_registry_order(self) -> None:
        self.store.save_agent_registry_entry(
            AgentRegistryEntryV1(
                runtime_id="critic-2",
                ultra_run_id=self.run.id,
                display_index=2,
                role="critic",
                state=AgentState.READY,
                provider="ollama",
                model="gemma4",
            )
        )
        self.store.save_agent_registry_entry(
            AgentRegistryEntryV1(
                runtime_id="planner-1",
                ultra_run_id=self.run.id,
                display_index=1,
                role="planner",
                state=AgentState.READY,
                provider="ollama",
                model="gemma4",
            )
        )
        self.store.save_agent_registry_entry(
            AgentRegistryEntryV1(
                runtime_id="broken-0",
                ultra_run_id=self.run.id,
                display_index=99,
                role="failed-worker",
                state=AgentState.FAILED,
                provider="ollama",
                model="gemma4",
            )
        )

        self.assertEqual(self.store.elect_consensus_leader(self.run.id), "planner-1")
        self.assertEqual(
            self.store.elect_consensus_leader(self.run.id, candidates=("critic-2",)),
            "critic-2",
        )

    def test_consensus_votes_close_round_when_quorum_is_met(self) -> None:
        round_item = self.store.open_consensus_round(
            ultra_run_id=self.run.id,
            topic="Ship only after runtime and visual gates pass",
            leader_agent_id="planner-1",
            quorum=2,
        )
        self.assertEqual(round_item["status"], ConsensusStatus.OPEN.value)

        partial = self.store.record_consensus_vote(
            ConsensusVoteV1(
                round_id=round_item["id"],
                voter_agent_id="critic-1",
                verdict="accept",
                confidence=0.6,
                rationale="Runtime gate catches blank output.",
            )
        )
        self.assertEqual(partial["status"], ConsensusStatus.OPEN.value)

        closed = self.store.record_consensus_vote(
            ConsensusVoteV1(
                round_id=round_item["id"],
                voter_agent_id="tester-1",
                verdict="reject",
                confidence=0.9,
                rationale="Visual gate still weak.",
                evidence={"screenshot_score": 0.35},
            )
        )
        self.assertEqual(closed["status"], ConsensusStatus.REJECTED.value)
        self.assertEqual(closed["decision"]["verdict"], "reject")
        self.assertGreater(closed["decision"]["reject_score"], closed["decision"]["accept_score"])
        listed = self.store.list_consensus_rounds(
            self.run.id,
            status=ConsensusStatus.REJECTED,
            topic_prefix="Ship only",
        )
        self.assertEqual([item["id"] for item in listed], [round_item["id"]])

    def test_swarm_coordinator_runs_proposal_vote_and_decision_flow(self) -> None:
        bus = SwarmBus(self.store)
        tester_seen: list[dict[str, object]] = []
        critic_seen: list[dict[str, object]] = []
        decisions: list[dict[str, object]] = []
        bus.subscribe(self.run.id, tester_seen.append, recipient_agent_id="tester-1")
        bus.subscribe(self.run.id, critic_seen.append, recipient_agent_id="critic-1")
        bus.subscribe(self.run.id, decisions.append, recipient_agent_id="swarm", message_type=SwarmMessageType.DECISION)
        coordinator = SwarmCoordinator(self.store, bus)

        workflow = coordinator.propose(
            ultra_run_id=self.run.id,
            proposer_agent_id="planner-1",
            topic="Ship only if visual and runtime gates pass",
            proposal={"gate": "visual+runtime", "required": True},
            voters=("tester-1", "critic-1"),
            quorum=2,
            leader_agent_id="planner-1",
        )

        self.assertEqual(workflow.leader_agent_id, "planner-1")
        self.assertEqual(len(workflow.request_message_ids), 2)
        self.assertEqual(tester_seen[0]["payload"]["round_id"], workflow.consensus_round_id)
        self.assertEqual(critic_seen[0]["payload"]["proposal_message_id"], workflow.proposal_message_id)

        open_round = coordinator.submit_vote(
            round_id=workflow.consensus_round_id,
            voter_agent_id="tester-1",
            verdict="accept",
            confidence=0.8,
            rationale="Runtime gate passed.",
            evidence={"runtime": "passed"},
        )
        self.assertEqual(open_round["status"], ConsensusStatus.OPEN.value)
        closed = coordinator.submit_vote(
            round_id=workflow.consensus_round_id,
            voter_agent_id="critic-1",
            verdict="accept",
            confidence=0.7,
            rationale="Visual gate passed.",
            evidence={"visual": "passed"},
        )

        self.assertEqual(closed["status"], ConsensusStatus.ACCEPTED.value)
        self.assertEqual(decisions[0]["payload"]["decision"]["verdict"], "accept")
        decision_inbox = self.store.list_swarm_messages(
            self.run.id,
            recipient_agent_id="swarm",
            topic=f"consensus-decision:{workflow.consensus_round_id}",
        )
        self.assertEqual(decision_inbox[0]["message_type"], SwarmMessageType.DECISION.value)

    def test_artifacts_preserve_pre_and_post_write_hashes(self) -> None:
        module = self.store.sync_master_modules(self.run.id)[0]
        artifact = self.store.add_artifact(
            Artifact(
                ultra_run_id=self.run.id,
                work_node_id=module.id,
                kind="code",
                uri="workspace://src/core/main.py",
                path="src/core/main.py",
                pre_write_hash="before",
                content_hash="after",
                evidence={"test": "passed"},
            )
        )
        restored = self.store.list_artifacts(self.run.id, work_node_id=module.id)[0]
        self.assertEqual(restored.id, artifact.id)
        self.assertEqual((restored.pre_write_hash, restored.content_hash), ("before", "after"))

    def test_prompt_trace_is_redacted_compressed_and_capped(self) -> None:
        module = self.store.sync_master_modules(self.run.id)[0]
        secret = "sk-supersecretvalue123456"
        trace = self.store.add_prompt_trace(
            PromptTraceV1(
                ultra_run_id=self.run.id,
                work_node_id=module.id,
                role="coder",
                system_prompt=f"Authorization: Bearer {secret}\n" + "S" * 12_000,
                context_package={"api_key": secret, "files": ["F" * 12_000]},
                self_prompt="Implement safely " + "P" * 12_000,
                reasoning_summary="Selected the smallest compatible change.",
            ),
            max_bytes=4_096,
        )
        self.assertTrue(trace.redacted)
        self.assertTrue(trace.truncated)
        self.assertNotIn(secret, trace.system_prompt)
        self.assertNotIn(secret, str(trace.context_package))
        restored = self.store.get_prompt_trace(trace.id)
        self.assertNotIn(secret, restored.system_prompt)
        connection = sqlite3.connect(self.store.path)
        try:
            blobs = connection.execute(
                "SELECT system_prompt_blob,context_blob,self_prompt_blob,stored_size FROM prompt_traces WHERE id=?",
                (trace.id,),
            ).fetchone()
        finally:
            connection.close()
        self.assertLess(blobs[3], 4_096)
        self.assertNotIn(secret.encode(), b"".join(blobs[:3]))
        self.assertIn("[REDACTED]", zlib.decompress(blobs[0]).decode())

    def test_overlapping_leases_and_stale_hashes_are_safe(self) -> None:
        modules = self.store.sync_master_modules(self.run.id)
        first, second = modules
        lease = self.store.acquire_resource_lease(
            self.run.id, first.id, "src/core", pre_write_hash="old", ttl_seconds=60
        )
        with self.assertRaises(LeaseConflictError):
            self.store.acquire_resource_lease(
                self.run.id, second.id, "src/core/main.py", pre_write_hash="old", ttl_seconds=60
            )
        with self.assertRaises(LeaseConflictError):
            self.store.assert_lease_hash(lease.id, "changed")
        self.assertEqual(self.store.get_work_node(first.id).status, WorkNodeStatus.UNCERTAIN)
        self.assertEqual(
            self.store.list_resource_leases(self.run.id)[0].status, LeaseStatus.UNCERTAIN
        )

    def test_recovery_marks_inflight_ultra_work_uncertain_without_replay(self) -> None:
        module = self.store.sync_master_modules(self.run.id)[0]
        self.store.transition_work_node(module.id, WorkNodeStatus.IN_PROGRESS)
        agent = self.store.create_agent_run(
            AgentRun(
                ultra_run_id=self.run.id,
                work_node_id=module.id,
                role="coder",
                provider="ollama",
                model="qwen3-coder",
                phase="implement",
                status=AgentRunStatus.RUNNING,
                side_effects=True,
            )
        )
        report = self.store.recover_ultra_inflight()
        self.assertIn(module.id, report.work_node_ids)
        self.assertIn(agent.id, report.agent_run_ids)
        self.assertEqual(self.store.get_work_node(module.id).status, WorkNodeStatus.UNCERTAIN)
        self.assertEqual(self.store.get_agent_run(agent.id).status, AgentRunStatus.UNCERTAIN)
        self.assertEqual(self.store.get_ultra_run(self.run.id).status, UltraRunStatus.RECOVERING)

    def test_expired_lease_reaping_is_durable(self) -> None:
        module = self.store.sync_master_modules(self.run.id)[0]
        lease = self.store.acquire_resource_lease(self.run.id, module.id, "src/core")
        with self.store.transaction() as connection:
            connection.execute(
                "UPDATE resource_leases SET expires_at=? WHERE id=?",
                ((utc_now() - timedelta(seconds=1)).isoformat(), lease.id),
            )
        self.assertEqual(self.store.reap_expired_leases(), (lease.id,))
        self.assertEqual(
            self.store.list_resource_leases(self.run.id)[0].status, LeaseStatus.EXPIRED
        )


if __name__ == "__main__":
    unittest.main()
