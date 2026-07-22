from __future__ import annotations

from pathlib import Path
import sqlite3
import tempfile
import unittest

from agent.durable_memory import (
    AgentMemorySnapshotV1,
    NextActionPacketV1,
    NextActionStatus,
)
from agent.goal_outcome import GoalOutcomeContractV1, GoalOutcomeState
from agent.project_brain import ProjectBrain
from agent.store import StateStore
from agent.ultra_models import (
    ExecutionClass,
    TaskContractV1,
    UltraRun,
    WorkNode,
)


class DurableMemoryV11Tests(unittest.TestCase):
    def _store_and_run(self, directory: str) -> tuple[StateStore, UltraRun]:
        store = StateStore(directory)
        goal = store.create_goal("Build a safe multi-file application")
        run = store.create_ultra_run(
            UltraRun(
                goal_id=goal.id,
                provider="ollama",
                model="gemma4:e4b",
                execution_class=ExecutionClass.LOCAL,
            )
        )
        return store, run

    def test_running_next_action_recovers_with_packet_and_memory_after_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store, run = self._store_and_run(directory)
            packet = NextActionPacketV1(
                ultra_run_id=run.id,
                role="coder",
                phase="implement",
                objective="Implement only the appointment persistence adapter",
                contract={"write_paths": ["src/persistence/"]},
                required_outputs=("adapter tests pass",),
                context_budget_chars=16_000,
            )
            store.stage_scheduled_agent_action(
                "action-1",
                packet,
                agent_run_id="agent-1",
            )
            stored_memory = store.save_agent_memory_snapshot(
                AgentMemorySnapshotV1(
                    ultra_run_id=run.id,
                    role="coder",
                    objective=packet.objective,
                    checkpoint="implementation staged",
                    next_action_id="action-1",
                )
            )
            self.assertEqual(stored_memory.revision, 1)
            store.close()

            reopened = StateStore(directory)
            try:
                action = reopened.get_scheduled_agent_action("action-1")
                self.assertEqual(action["status"], NextActionStatus.RECOVERING.value)
                self.assertEqual(action["packet"]["objective"], packet.objective)
                restored = reopened.latest_agent_memory_snapshot(
                    run.id,
                    work_node_id=None,
                    role="coder",
                )
                self.assertIsNotNone(restored)
                self.assertEqual(restored.checkpoint, "implementation staged")
                connection = sqlite3.connect(
                    Path(directory) / ".coding-agent" / "state.db"
                )
                try:
                    self.assertEqual(
                        connection.execute("PRAGMA user_version").fetchone()[0],
                        12,
                    )
                finally:
                    connection.close()
            finally:
                reopened.close()

    def test_context_projects_one_bounded_action_not_full_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store, run = self._store_and_run(directory)
            node = store.create_work_node(
                WorkNode(
                    ultra_run_id=run.id,
                    title="Appointment API persistence",
                    objective="Implement transaction-safe appointment persistence",
                    contract=TaskContractV1(
                        objective="Implement transaction-safe appointment persistence",
                        success_criteria=("concurrent booking is safe",),
                        write_paths=("src/persistence/",),
                        interfaces={"AppointmentStore": {"reserve": "slot -> result"}},
                    ),
                    assigned_role="coder",
                    checkpoint="implement",
                )
            )
            brain = ProjectBrain(store, run.id)
            for index in range(20):
                brain.remember_for_role(
                    "coder",
                    f"Long historical note {index}",
                    ("historical detail " * 180) + str(index),
                )
            store.save_agent_memory_snapshot(
                AgentMemorySnapshotV1(
                    ultra_run_id=run.id,
                    work_node_id=node.id,
                    role="coder",
                    objective=node.objective,
                    checkpoint="contracts loaded",
                )
            )
            package = brain.build_context(
                node.id,
                "coder",
                budget_chars=16_000,
            )
            self.assertLessEqual(package.size_chars, 16_000)
            self.assertIn("next_action_packet", package.sections)
            packet = package.sections["next_action_packet"]
            self.assertEqual(packet["schema"], "NextActionPacketV1")
            self.assertEqual(packet["objective"], node.objective)
            self.assertIn("role_memory", package.omitted_sections)
            store.close()

    def test_active_goal_heartbeat_does_not_recover_live_action_on_reopen(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store, run = self._store_and_run(directory)
            store.save_goal_outcome_contract(
                GoalOutcomeContractV1(
                    goal_id=run.goal_id,
                    objective="Build a safe multi-file application",
                ),
                ultra_run_id=run.id,
            )
            packet = NextActionPacketV1(
                ultra_run_id=run.id,
                role="coder",
                phase="implement",
                objective="Continue the live GPU-only implementation",
            )
            store.stage_scheduled_agent_action("live-action", packet)
            store.close()

            reopened = StateStore(directory)
            try:
                action = reopened.get_scheduled_agent_action("live-action")
                self.assertEqual(action["status"], NextActionStatus.RUNNING.value)
                outcome = reopened.get_goal_outcome_contract(run.goal_id)
                self.assertEqual(outcome["state"], GoalOutcomeState.RUNNING.value)
            finally:
                reopened.close()


if __name__ == "__main__":
    unittest.main()
