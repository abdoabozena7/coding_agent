from __future__ import annotations

import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path
from threading import Event, Lock

from agent.config import RuntimeConfig
from agent.model_catalog import ExecutionClass, ModelDescriptor
from agent.local_provider import (
    ProviderDiagnostic,
    ProviderFailureKind,
    ProviderRequestError,
)
from agent.runtime import AgentRuntime, ProviderUnavailableError
from agent.providers import AssistantTurn, Usage
from agent.store import StateStore
from agent.testing import ScriptedProvider, ScriptedTurn


class _SilentProvider:
    name = "openai"
    model = "watchdog-test"
    reasoning_effort = "high"

    def __init__(self, *, failure: str = "", retry_after: int | None = None) -> None:
        self.calls = 0
        self.failure = failure
        self.retry_after = retry_after

    def call(self, *_args, **_kwargs):
        self.calls += 1
        if self.failure:
            error = RuntimeError(self.failure)
            if self.retry_after is not None:
                error.retry_after = self.retry_after
            raise error
        time.sleep(2)
        raise AssertionError("silent provider should have timed out")


class _LateThenFreshProvider:
    name = "openai"
    model = "watchdog-test"
    reasoning_effort = "high"

    def __init__(self) -> None:
        self.calls = 0
        self.release_late = Event()
        self._lock = Lock()

    def call(self, *_args, **_kwargs):
        with self._lock:
            self.calls += 1
            call_number = self.calls
        if call_number == 1:
            self.release_late.wait(timeout=2)
            return AssistantTurn(text="stale")
        return AssistantTurn(text="fresh")

    def cancel_active_request(self) -> None:
        # Simulate a transport whose underlying request cannot be interrupted.
        # The watchdog's abandoned-result guard must still isolate the late turn.
        return None


class ProviderWatchdogTests(unittest.TestCase):
    def _runtime(self, provider, **updates) -> tuple[AgentRuntime, StateStore]:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        workspace = Path(temporary.name)
        store = StateStore(workspace)
        self.addCleanup(store.close)
        config_values = {
            "repository_index_warmup_files": 0,
            "max_provider_retries": 0,
            **updates,
        }
        config = replace(RuntimeConfig(), **config_values)
        runtime = AgentRuntime(
            provider,
            store,
            workspace,
            config=config,
            model_descriptor=ModelDescriptor(
                "openai",
                "watchdog-test",
                ExecutionClass.CLOUD,
            ),
        )
        self.addCleanup(runtime.close)
        return runtime, store

    def test_silent_cloud_call_is_cancelled_by_watchdog(self) -> None:
        runtime, _store = self._runtime(
            _SilentProvider(),
            cloud_idle_timeout_seconds=0,
            provider_call_timeout_seconds=10,
            activity_heartbeat_seconds=1,
        )
        started = time.monotonic()
        with self.assertRaisesRegex(TimeoutError, "silent"):
            runtime._provider_call_with_watchdog(
                [],
                [],
                "system",
                actor="test",
                stream_text=True,
            )
        self.assertLess(time.monotonic() - started, 1.5)

    def test_quota_failure_does_not_retry_or_fallback(self) -> None:
        provider = _SilentProvider(failure="429 quota exhausted", retry_after=17)
        runtime, _store = self._runtime(provider)
        with self.assertRaises(ProviderUnavailableError) as raised:
            runtime._call_provider([], [], "system", actor="test", step=1)
        self.assertEqual(provider.calls, 1)
        self.assertEqual(runtime.execution_class, "cloud")
        self.assertEqual(raised.exception.retry_after_seconds, 17)

    def test_overloaded_503_becomes_one_resumable_boundary_without_replay(self) -> None:
        provider = _SilentProvider()

        def overloaded(*_args, **_kwargs):
            provider.calls += 1
            raise ProviderRequestError(
                ProviderDiagnostic(
                    reachable=True,
                    kind=ProviderFailureKind.HTTP_5XX,
                    operation="chat",
                    status_code=503,
                    provider_message="model is temporarily overloaded, please retry shortly",
                )
            )

        provider.call = overloaded  # type: ignore[method-assign]
        runtime, _store = self._runtime(provider, max_provider_retries=3)
        with self.assertRaisesRegex(
            ProviderUnavailableError,
            "temporarily overloaded",
        ) as raised:
            runtime._call_provider([], [], "system", actor="semantic-router", step=1)
        self.assertEqual(provider.calls, 1)
        self.assertEqual(raised.exception.retry_after_seconds, 30)

    def test_stream_activity_records_real_chunks_without_counting_heartbeats(self) -> None:
        provider = ScriptedProvider(
            [
                ScriptedTurn(
                    AssistantTurn(text="hello", usage=Usage(output_tokens=2)),
                    text_chunks=["he", "llo"],
                )
            ],
            model="watchdog-test",
        )
        runtime, _store = self._runtime(provider)

        turn = runtime._provider_call_with_watchdog(
            [],
            [],
            "system",
            actor="chat",
            stream_text=True,
        )

        self.assertEqual(turn.text, "hello")
        snapshot = runtime.workflow_runtime_snapshot()
        self.assertEqual(snapshot.received_bytes, 5)
        self.assertEqual(snapshot.received_chunks, 2)
        self.assertEqual(snapshot.received_tokens, 2)
        provider_events = [
            item for item in runtime.events.list_live_events()
            if item.operation and item.source == "MODEL"
        ]
        self.assertTrue(any(item.state == "receiving" for item in provider_events))
        self.assertEqual(max(item.received_chunks for item in provider_events), 2)

    def test_local_stage_deadlines_allow_slow_first_structured_response(self) -> None:
        runtime, _store = self._runtime(_SilentProvider())
        cloud = runtime._provider_call_policy("planner")
        runtime.model_descriptor = ModelDescriptor(
            "ollama", "qwen2.5-coder:7b", ExecutionClass.LOCAL
        )
        local = runtime._provider_call_policy("planner")
        self.assertEqual(cloud.stage_deadline_seconds, 360.0)
        self.assertEqual(local.stage_deadline_seconds, 600.0)
        self.assertGreater(local.stage_deadline_seconds, cloud.stage_deadline_seconds)

    def test_late_abandoned_response_cannot_replace_newer_response(self) -> None:
        provider = _LateThenFreshProvider()
        runtime, _store = self._runtime(
            provider,
            cloud_idle_timeout_seconds=0,
            provider_call_timeout_seconds=10,
            activity_heartbeat_seconds=1,
        )

        with self.assertRaisesRegex(TimeoutError, "silent"):
            runtime._provider_call_with_watchdog(
                [], [], "system", actor="planner", stream_text=True,
                logical_request_id="logical-old", physical_attempt=1,
            )
        fresh = runtime._provider_call_with_watchdog(
            [], [], "system", actor="planner", stream_text=True,
            logical_request_id="logical-new", physical_attempt=1,
        )
        provider.release_late.set()
        time.sleep(0.05)

        self.assertEqual(fresh.text, "fresh")
        self.assertEqual(provider.calls, 2)


if __name__ == "__main__":
    unittest.main()
