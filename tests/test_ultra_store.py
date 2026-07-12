from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from pathlib import Path
import sqlite3
import tempfile
import unittest
import zlib

from agent.models import GoalStatus, RoleProfile, Task
from agent.model_catalog import ExecutionClass as CatalogExecutionClass
from agent.project_brain import ProjectBrain
from agent.sandbox import AccessLevel as SandboxAccessLevel
from agent.store import (
    ConcurrentBrainUpdateError,
    LeaseConflictError,
    StateStore,
)
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

    def test_schema_v3_and_specs_survive_reopen(self) -> None:
        from agent.ultra_models import AccessLevel

        self.assertIs(ExecutionClass, CatalogExecutionClass)
        self.assertIs(AccessLevel, SandboxAccessLevel)
        connection = sqlite3.connect(self.store.path)
        try:
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 3)
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

        package = brain.build_context(module.id, "coder", budget_chars=20_000)
        self.assertIn("task", package.sections)
        self.assertIn("architecture", package.sections)
        self.assertIn("decisions", package.sections)
        self.assertLessEqual(package.size_chars, 20_000)
        self.assertTrue(self.store.list_memory_access(self.run.id))

        # The deterministic LIKE fallback is a supported runtime path.
        self.store._fts5_available = False
        self.assertEqual(brain.search("double")[0].id, second.id)

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
