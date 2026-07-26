from __future__ import annotations

import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path

from agent.config import RuntimeConfig
from agent.model_catalog import ExecutionClass, ModelDescriptor
from agent.runtime import AgentRuntime, ProviderUnavailableError
from agent.store import StateStore


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


class ProviderWatchdogTests(unittest.TestCase):
    def _runtime(self, provider: _SilentProvider, **updates) -> tuple[AgentRuntime, StateStore]:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        workspace = Path(temporary.name)
        store = StateStore(workspace)
        self.addCleanup(store.close)
        config = replace(
            RuntimeConfig(),
            repository_index_warmup_files=0,
            max_provider_retries=0,
            **updates,
        )
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


if __name__ == "__main__":
    unittest.main()
