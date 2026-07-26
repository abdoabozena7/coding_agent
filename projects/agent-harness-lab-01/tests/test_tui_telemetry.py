from __future__ import annotations

import subprocess
import time
import unittest
from unittest import mock

from agent.tui_telemetry import TelemetrySampler, probe_gpu, sample_system


class TelemetryTests(unittest.TestCase):
    def test_system_sample_is_an_immutable_numeric_snapshot(self):
        sample = sample_system({"gpu_available": False})
        self.assertIsNotNone(sample.sampled_at)
        self.assertFalse(sample.gpu_available)
        if sample.memory_percent is not None:
            self.assertGreaterEqual(sample.memory_percent, 0)

    def test_gpu_probe_timeout_is_bounded_and_reports_unavailable(self):
        with mock.patch("agent.tui_telemetry.shutil.which") as which, mock.patch(
            "agent.tui_telemetry.subprocess.run",
            side_effect=subprocess.TimeoutExpired("nvidia-smi", 0.01),
        ):
            which.side_effect = lambda name: name if name == "nvidia-smi" else None
            started = time.monotonic()
            result = probe_gpu(timeout=0.01)
        self.assertLess(time.monotonic() - started, 0.5)
        self.assertFalse(result["gpu_available"])

    def test_sampler_publishes_without_blocking_the_caller_and_stops_cleanly(self):
        samples = []
        with mock.patch(
            "agent.tui_telemetry.probe_gpu", return_value={"gpu_available": False}
        ):
            sampler = TelemetrySampler(samples.append, interval=0.02, gpu_interval=0.04)
            started = time.monotonic()
            sampler.start()
            self.assertLess(time.monotonic() - started, 0.1)
            deadline = time.monotonic() + 1.0
            while not samples and time.monotonic() < deadline:
                time.sleep(0.01)
            sampler.stop()
        self.assertTrue(samples)
        self.assertIsNone(sampler._thread)


if __name__ == "__main__":
    unittest.main()
