from __future__ import annotations

import unittest

from agent.events import EventBus
from agent.ui_state import WorkspaceUIStore


class LiveActivityProjectionTests(unittest.TestCase):
    def test_event_sequences_are_monotonic_and_reasoning_is_never_exposed(self) -> None:
        events = EventBus()
        first = events.publish(
            "model_thought",
            '{"secret_partial_tool_args":"do not render"}',
            actor="planner",
        )
        second = events.publish(
            "provider.activity",
            "First response bytes received",
            source_kind="MODEL",
            actor="planner",
            phase="planning",
            state="receiving",
            provider_state="receiving",
            received_bytes=12,
            received_chunks=1,
        )

        self.assertEqual((first.sequence, second.sequence), (1, 2))
        projected = events.list_live_events()
        self.assertEqual([item.sequence for item in projected], [1, 2])
        self.assertEqual(projected[0].message, "Model reasoning activity received")
        self.assertNotIn("secret_partial_tool_args", str(projected[0].to_dict()))
        self.assertEqual(projected[1].received_bytes, 12)

    def test_ui_distinguishes_request_open_from_actual_model_output(self) -> None:
        store = WorkspaceUIStore()
        store.handle_event(
            "provider.activity",
            "Provider request sent",
            {
                "sequence": 1,
                "source": "MODEL",
                "phase": "planning",
                "provider_state": "request_sent",
                "operation": "Preparing the project plan",
                "received_bytes": 0,
                "received_chunks": 0,
            },
        )
        waiting = store.snapshot()
        self.assertEqual(waiting.liveness, "request_sent")
        self.assertEqual(waiting.received_bytes, 0)
        self.assertEqual(waiting.received_chunks, 0)

        store.handle_event(
            "provider.activity",
            "First response bytes received",
            {
                "sequence": 2,
                "source": "MODEL",
                "phase": "planning",
                "provider_state": "receiving",
                "operation": "Receiving the project plan",
                "received_bytes": 48,
                "received_chunks": 1,
            },
        )
        receiving = store.snapshot()
        self.assertEqual(receiving.liveness, "receiving")
        self.assertEqual(receiving.received_bytes, 48)
        self.assertEqual(receiving.received_chunks, 1)
        self.assertEqual(receiving.live_timeline[-1].source, "MODEL")

        store.handle_event(
            "provider.activity",
            "Provider request failed",
            {
                "sequence": 3,
                "source": "MODEL",
                "phase": "planning",
                "provider_state": "failed",
                "state": "failed",
                "operation": "Provider request failed",
            },
        )
        failed = store.snapshot()
        self.assertEqual(failed.liveness, "stalled")

    def test_tool_failure_is_failed_activity_not_completed_activity(self) -> None:
        events = EventBus()
        event = events.publish(
            "tool_result",
            "Error: plan validation failed",
            tool="propose_plan",
            actor="planner",
            phase="planning",
        )
        projected = events.list_live_events()
        self.assertEqual(projected[-1].state, "failed")

        store = WorkspaceUIStore()
        store.handle_event(
            event.kind,
            event.message,
            {**event.data, "sequence": event.sequence},
        )
        self.assertEqual(store.snapshot().live_timeline[-1].state, "failed")


if __name__ == "__main__":
    unittest.main()
