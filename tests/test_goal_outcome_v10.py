from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from agent.goal_outcome import (
    ExperimentOutcome,
    FinalAcceptanceEvidenceV1,
    FinalAcceptanceGate,
    GoalOutcomeContractV1,
    GoalOutcomeState,
    OptimizationExperimentV1,
)
from agent.models import GoalStatus
from agent.quality import FindingSeverity, QualityCategory, QualityFindingV1
from agent.store import StateStore
from agent.ultra_models import (
    AccessLevel,
    ExecutionClass,
    UltraPhase,
    UltraRun,
    UltraRunStatus,
)
from agent.ultra import TaskContractV1, UltraOrchestrator, WorkNode


class GoalOutcomeContractTests(unittest.TestCase):
    def test_final_gate_fails_closed_until_every_product_authority_exists(self) -> None:
        contract = GoalOutcomeContractV1(goal_id="goal", objective="Build a game")
        run = "run"
        partial = (
            FinalAcceptanceEvidenceV1(run, "final_artifact", "harness", True, 1.0),
            FinalAcceptanceEvidenceV1(run, "runtime", "playwright", True, 1.0),
        )
        rejected = FinalAcceptanceGate(contract).evaluate(partial)
        self.assertFalse(rejected.accepted)
        self.assertIn("codex_visual_review", rejected.missing)

        complete = (
            *partial,
            FinalAcceptanceEvidenceV1(run, "screenshots", "playwright", True, 1.0),
            FinalAcceptanceEvidenceV1(
                run,
                "independent_visual",
                "vision:a",
                True,
                0.96,
                details={"critical": True},
            ),
            FinalAcceptanceEvidenceV1(
                run,
                "independent_visual",
                "vision:b",
                True,
                0.97,
                details={"critical": True},
            ),
            FinalAcceptanceEvidenceV1(
                run,
                "pairwise_baseline",
                "blind-judge",
                True,
                0.98,
                details={"candidate_preferred": True, "critical": True},
            ),
            FinalAcceptanceEvidenceV1(
                run,
                "codex_visual_review",
                "codex-supervisor",
                True,
                0.96,
                details={"critical": True},
            ),
        )
        accepted = FinalAcceptanceGate(contract).evaluate(complete)
        self.assertTrue(accepted.accepted)
        self.assertEqual(accepted.blockers, ())

    def test_one_critical_finding_rejects_a_high_average(self) -> None:
        contract = GoalOutcomeContractV1(goal_id="goal", objective="Build a game")
        evidence = tuple(
            FinalAcceptanceEvidenceV1(
                "run",
                kind,
                f"authority:{index}",
                True,
                0.99,
                critical_findings=1 if kind == "codex_visual_review" else 0,
                details={
                    "candidate_preferred": kind == "pairwise_baseline",
                    "critical": True,
                },
            )
            for index, kind in enumerate(contract.required_evidence)
        ) + (
            FinalAcceptanceEvidenceV1(
                "run",
                "independent_visual",
                "vision:second",
                True,
                0.99,
                details={"critical": True},
            ),
        )
        decision = FinalAcceptanceGate(contract).evaluate(evidence)
        self.assertFalse(decision.accepted)
        self.assertIn("one or more critical findings remain", decision.blockers)

    def test_new_supervisory_verdict_supersedes_older_same_authority(self) -> None:
        contract = GoalOutcomeContractV1(
            goal_id="goal",
            objective="Build",
            required_evidence=("codex_visual_review",),
            require_candidate_preferred=False,
        )
        old = FinalAcceptanceEvidenceV1(
            "run",
            "codex_visual_review",
            "codex-supervisor",
            False,
            0.70,
            critical_findings=2,
        )
        new = FinalAcceptanceEvidenceV1(
            "run",
            "codex_visual_review",
            "codex-supervisor",
            True,
            0.97,
            critical_findings=0,
            details={"critical": True},
        )
        self.assertTrue(FinalAcceptanceGate(contract).evaluate((old, new)).accepted)

    def test_visual_quality_finding_is_a_blocking_typed_finding(self) -> None:
        finding = QualityFindingV1(
            ultra_run_id="run",
            principle_id="visual_quality",
            category=QualityCategory.VISUAL,
            severity=FindingSeverity.HIGH,
            path="preview.html",
            location="component preview",
            file_hash="a" * 64,
            evidence={"screenshot": "preview.png", "score": 0.62},
            remediation="Replace placeholder geometry with a modeled component.",
            acceptance_criteria=("Critical visual score is at least 0.90.",),
            verification=("Inspect a fresh deterministic screenshot.",),
            repair_node_id="vehicles.wheels.rim",
        )
        self.assertTrue(finding.severity.blocks_completion)
        self.assertEqual(finding.category, QualityCategory.VISUAL)


class GoalOutcomePersistenceTests(unittest.TestCase):
    def test_schema_v10_persists_contract_heartbeat_experiments_and_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(directory)
            goal = store.create_goal("Build a polished browser game")
            store.transition_goal(goal.id, GoalStatus.DISCOVERING)
            run = UltraRun(
                id="ultra_test",
                goal_id=goal.id,
                provider="ollama",
                model="gemma4:e4b",
                execution_class=ExecutionClass.LOCAL,
                access_level=AccessLevel.NORMAL,
                concurrency=1,
                phase=UltraPhase.GOAL_SPEC,
                status=UltraRunStatus.DRAFT,
            )
            store.create_ultra_run(run)
            contract = GoalOutcomeContractV1(
                goal_id=goal.id,
                objective=goal.objective,
            )
            saved = store.save_goal_outcome_contract(
                contract,
                ultra_run_id=run.id,
            )
            self.assertEqual(saved["state"], GoalOutcomeState.RUNNING.value)
            self.assertIsNotNone(saved["heartbeat_at"])

            experiment = OptimizationExperimentV1(
                ultra_run_id=run.id,
                node_id="vehicle.wheels",
                variable="specialist_topology",
                baseline={"children": 1},
                candidate={"children": 3},
                hypothesis="Narrow wheel specialists improve modeled detail.",
                before_score=0.70,
                after_score=0.91,
                outcome=ExperimentOutcome.CHAMPION,
                evidence=("screenshot:a",),
            )
            store.record_optimization_experiment(experiment)
            self.assertEqual(
                store.list_optimization_experiments(run.id)[0]["variable"],
                "specialist_topology",
            )
            store.record_final_acceptance_evidence(
                FinalAcceptanceEvidenceV1(
                    run.id,
                    "final_artifact",
                    "harness",
                    True,
                    1.0,
                )
            )
            self.assertEqual(
                store.list_final_acceptance_evidence(run.id)[0]["kind"],
                "final_artifact",
            )
            store.close()

            connection = sqlite3.connect(Path(directory) / ".coding-agent" / "state.db")
            try:
                self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 12)
                tables = {
                    row[0]
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    )
                }
                self.assertTrue(
                    {
                        "goal_outcome_contracts",
                        "optimization_experiments",
                        "final_acceptance_evidence",
                    }.issubset(tables)
                )
            finally:
                connection.close()


class RecursiveSpecialistTopologyTests(unittest.TestCase):
    @staticmethod
    def _node(identifier: str, domain: str) -> WorkNode:
        return WorkNode(
            TaskContractV1(
                id=identifier,
                title=f"{domain} specialist",
                objective=f"Build only {domain} at independently reviewable quality.",
                acceptance_criteria=("The bounded component has a runnable preview.",),
                verification=("Inspect the component without the rest of the game.",),
                owned_interfaces=("VehiclePackage",),
                metadata={
                    "component_package_only": True,
                    "component_leaf": True,
                    "specialist_domain": domain,
                },
            )
        )

    def test_vehicle_and_wheel_are_recursively_split_into_narrow_agents(self) -> None:
        vehicle_children = UltraOrchestrator._deterministic_specialist_children(
            self._node("vehicle", "vehicles")
        )
        self.assertEqual(
            {item["metadata"]["specialist_domain"] for item in vehicle_children},
            {
                "vehicles.chassis",
                "vehicles.wheels",
                "vehicles.cabin",
                "vehicles.materials",
            },
        )
        wheel_children = UltraOrchestrator._deterministic_specialist_children(
            self._node("vehicle.wheels", "vehicles.wheels")
        )
        self.assertEqual(
            {item["metadata"]["specialist_domain"] for item in wheel_children},
            {
                "vehicles.wheels.tire",
                "vehicles.wheels.rim",
                "vehicles.wheels.contact",
            },
        )
        self.assertTrue(
            all(item["metadata"]["component_package_only"] for item in wheel_children)
        )


if __name__ == "__main__":
    unittest.main()
