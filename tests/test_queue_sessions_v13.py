from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.commands import CommandKind, parse_command
from agent.concurrency import probe_concurrency
from agent.model_catalog import ExecutionClass, ModelDescriptor
from agent.models import QueuedPromptStatus
from agent.store import StateStore, StateStoreError


class QueueAndSessionV13Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temporary.name)
        self.store = StateStore(self.workspace)
        for session_id in ("session-a", "session-b"):
            self.store.save_workflow_session(
                session_id,
                goal_id=None,
                session_mode="normal",
                plan_state="none",
                run_state="idle",
            )

    def tearDown(self) -> None:
        self.store.close()
        self.temporary.cleanup()

    def test_unfinished_goals_are_isolated_per_session(self) -> None:
        first = self.store.create_goal("first", session_id="session-a")
        second = self.store.create_goal("second", session_id="session-b")
        self.assertEqual(self.store.load_active_goal("session-a").id, first.id)
        self.assertEqual(self.store.load_active_goal("session-b").id, second.id)
        with self.assertRaises(StateStoreError):
            self.store.create_goal("duplicate", session_id="session-a")

    def test_queue_is_fifo_mode_preserving_and_blocked_head_stops_claiming(self) -> None:
        one = self.store.enqueue_prompt("session-a", "plan this", "plan")
        two = self.store.enqueue_prompt("session-a", "implement this", "ultra")
        claimed = self.store.claim_next_prompt("session-a")
        self.assertEqual(claimed.id, one.id)
        self.assertEqual(claimed.mode, "plan")
        self.store.finish_queued_prompt(
            claimed.id,
            status=QueuedPromptStatus.BLOCKED,
            error="approval needed",
        )
        self.assertIsNone(self.store.claim_next_prompt("session-a"))
        items = self.store.list_queued_prompts("session-a")
        self.assertEqual([item.id for item in items], [one.id, two.id])
        self.assertEqual(items[1].mode, "ultra")

    def test_queue_limit_counts_blocked_items(self) -> None:
        first = self.store.enqueue_prompt("session-a", "item 0", "normal")
        claimed = self.store.claim_next_prompt("session-a")
        self.assertEqual(claimed.id, first.id)
        self.store.finish_queued_prompt(first.id, status="blocked", error="blocked")
        for index in range(1, 10):
            self.store.enqueue_prompt("session-a", f"item {index}", "normal")
        self.assertEqual(self.store.count_queued_prompts("session-a"), 10)
        with self.assertRaisesRegex(StateStoreError, "queue is full"):
            self.store.enqueue_prompt("session-a", "overflow", "normal")

    def test_running_unbound_prompt_returns_to_pending_after_restart(self) -> None:
        item = self.store.enqueue_prompt("session-a", "recover me", "normal")
        self.store.claim_next_prompt("session-a")
        self.store.close()
        self.store = StateStore(self.workspace)
        restored = self.store.list_queued_prompts("session-a")
        self.assertEqual(restored[0].id, item.id)
        self.assertEqual(restored[0].status, QueuedPromptStatus.PENDING)

    def test_removed_queue_views_are_not_public_slash_commands(self) -> None:
        from agent.commands import UnknownCommandParseError

        for command in (
            "/queue", "/effective-plan", "/ultra-details",
            "/project-brain architecture", "/sessions session-b", "/enqueue task",
        ):
            with self.subTest(command=command):
                with self.assertRaises(UnknownCommandParseError):
                    parse_command(command)


class ConcurrencyProbeTests(unittest.TestCase):
    def test_cloud_capacity_honors_provider_cap_and_recommends_two(self) -> None:
        descriptor = ModelDescriptor("openai", "gpt-test", ExecutionClass.CLOUD)
        capacity = probe_concurrency(descriptor, provider_cap=6)
        self.assertEqual(capacity.safe_max, 6)
        self.assertEqual(capacity.recommended, 2)

    def test_missing_local_telemetry_fails_closed_to_one(self) -> None:
        descriptor = ModelDescriptor("ollama", "local-test", ExecutionClass.LOCAL)
        self.assertEqual(probe_concurrency(descriptor, environ={}).safe_max, 1)

    def test_complete_local_telemetry_reserves_context_and_hardware(self) -> None:
        descriptor = ModelDescriptor(
            "ollama",
            "local-test",
            ExecutionClass.LOCAL,
            metadata={"parallelism": 6, "size_bytes": 2 * 1024**3},
        )
        with (
            patch("agent.concurrency._physical_cores", return_value=12),
            patch("agent.concurrency._total_ram_bytes", return_value=32 * 1024**3),
        ):
            capacity = probe_concurrency(descriptor, environ={})
        self.assertGreater(capacity.safe_max, 1)
        self.assertEqual(capacity.recommended, 2)


if __name__ == "__main__":
    unittest.main()
