"""Offline, injected-process tests for the Full Docker sandbox boundary."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agent.sandbox import (
    AccessLevel,
    DockerSandbox,
    DockerUnavailableError,
    PermissionAdapter,
    ProcessOutput,
    SANDBOX_IMAGE_VERSION,
    SANDBOX_LABEL,
    SandboxConfigError,
    SandboxError,
    SandboxLimits,
    sandbox_config_path,
    select_access_level,
)


class FakeDocker:
    def __init__(self):
        self.available = True
        self.image_exists = False
        self.image_user = "10001:10001"
        self.image_version = SANDBOX_IMAGE_VERSION
        self.build_output = ProcessOutput(0, b"built")
        self.run_output = ProcessOutput(0, b"ok\n", b"")
        self.calls = []

    def __call__(
        self,
        argv,
        *,
        input_data,
        timeout,
        max_output_bytes,
        env,
    ):
        argv = tuple(argv)
        self.calls.append(
            {
                "argv": argv,
                "input_data": input_data,
                "timeout": timeout,
                "max_output_bytes": max_output_bytes,
                "env": dict(env),
            }
        )
        operation = argv[1] if len(argv) > 1 else ""
        if operation == "version":
            if self.available:
                return ProcessOutput(0, b'"27.1.0"\n')
            return ProcessOutput(1, b"", b"daemon unavailable")
        if operation == "build":
            if self.build_output.returncode == 0 and not self.build_output.timed_out:
                self.image_exists = True
            return self.build_output
        if operation == "image" and argv[2] == "inspect":
            if not self.image_exists:
                return ProcessOutput(1, b"", b"No such image")
            value = {
                "User": self.image_user,
                "Labels": {SANDBOX_LABEL: self.image_version},
            }
            return ProcessOutput(0, json.dumps(value).encode("utf-8"))
        if operation == "run":
            return self.run_output
        if operation == "rm":
            return ProcessOutput(0, b"removed")
        raise AssertionError(f"unexpected Docker invocation: {argv!r}")

    def operations(self):
        return [call["argv"][1] for call in self.calls]


class AccessLevelTests(unittest.TestCase):
    def test_parse_and_defaults(self):
        self.assertEqual(AccessLevel.parse("safe"), AccessLevel.NORMAL)
        self.assertEqual(AccessLevel.parse("sandbox"), AccessLevel.FULL)
        with self.assertRaisesRegex(ValueError, "normal.*full"):
            AccessLevel.parse("host-yolo")

    def test_platform_config_paths_do_not_live_in_workspace(self):
        self.assertEqual(
            sandbox_config_path({"APPDATA": "C:/Users/me/AppData/Roaming"}),
            Path("C:/Users/me/AppData/Roaming/GA3BAD/sandbox.json"),
        )
        self.assertEqual(
            sandbox_config_path({"XDG_CONFIG_HOME": "/custom/config"}),
            Path("/custom/config/ga3bad/sandbox.json"),
        )
        self.assertEqual(
            sandbox_config_path({}, home="/home/me"),
            Path("/home/me/.config/ga3bad/sandbox.json"),
        )


class DockerSandboxTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        self.config_path = self.root / "platform" / "sandbox.json"
        self.docker = FakeDocker()
        self.environ = {
            "PATH": "safe-path",
            "SYSTEMROOT": "C:/Windows",
            "HOME": "C:/Users/private",
            "USERPROFILE": "C:/Users/private",
            "DOCKER_CONFIG": "C:/Users/private/.docker",
            "OPENAI_API_KEY": "top-secret",
            "PYTHONPATH": "host-injection",
        }
        self.sandbox = DockerSandbox(
            config_path=self.config_path,
            process_runner=self.docker,
            environ=self.environ,
            limits=SandboxLimits(timeout_seconds=30, max_output_bytes=2_048),
        )

    def test_normal_is_a_zero_probe_behavior_adapter(self):
        adapter = PermissionAdapter(AccessLevel.NORMAL, self.sandbox)
        called = []

        result = adapter.run_shell(
            "echo normal",
            self.workspace,
            normal_runner=lambda command: called.append(command) or "host-normal",
        )

        self.assertEqual(result, "host-normal")
        self.assertEqual(called, ["echo normal"])
        self.assertTrue(adapter.requires_approval(True))
        self.assertEqual(self.docker.calls, [])

    def test_bounded_requires_approval_for_every_mutation_or_process_launch(self):
        adapter = PermissionAdapter(AccessLevel.BOUNDED, self.sandbox)

        self.assertFalse(adapter.requires_approval(False, bounded_operation=False))
        self.assertTrue(adapter.requires_approval(False, bounded_operation=True))
        self.assertTrue(adapter.requires_approval(True, bounded_operation=True))
        self.assertEqual(self.docker.calls, [])

    def test_host_full_bypasses_repeated_approval_but_keeps_destructive_guards(self):
        adapter = PermissionAdapter(AccessLevel.HOST, self.sandbox)
        calls = []
        result = adapter.run_shell(
            "echo safe", self.workspace,
            normal_runner=lambda command: calls.append(command) or "host-safe",
        )

        self.assertEqual(result, "host-safe")
        self.assertFalse(adapter.requires_approval(True, bounded_operation=True))
        with self.assertRaisesRegex(SandboxError, "blocked"):
            adapter.run_shell(
                "Remove-Item -Recurse C:\\Windows", self.workspace,
                normal_runner=lambda _command: "must not run",
            )
        with self.assertRaisesRegex(SandboxError, "sensitive"):
            adapter.run_shell(
                "Get-Content C:\\Users\\me\\.ssh\\id_rsa", self.workspace,
                normal_runner=lambda _command: "must not run",
            )
        self.assertEqual(calls, ["echo safe"])

    def test_full_unavailable_downgrades_and_never_builds_or_starts_docker(self):
        self.docker.available = False

        selection = select_access_level(AccessLevel.FULL, self.sandbox)

        self.assertEqual(selection.requested, AccessLevel.FULL)
        self.assertEqual(selection.effective, AccessLevel.NORMAL)
        self.assertTrue(selection.downgraded)
        self.assertIn("using Normal", selection.reason)
        self.assertEqual(self.docker.operations(), ["version"])
        all_arguments = [argument for call in self.docker.calls for argument in call["argv"]]
        self.assertNotIn("build", all_arguments)
        self.assertNotIn("start", all_arguments)
        self.assertNotIn("install", all_arguments)

    def test_setup_is_explicit_versioned_non_root_and_one_time(self):
        config = self.sandbox.setup()
        same = self.sandbox.setup()

        self.assertEqual(config, same)
        self.assertEqual(self.docker.operations().count("build"), 1)
        build = next(call for call in self.docker.calls if call["argv"][1] == "build")
        dockerfile = build["input_data"].decode("utf-8")
        self.assertIn(f"LABEL {SANDBOX_LABEL}=\"{SANDBOX_IMAGE_VERSION}\"", dockerfile)
        self.assertIn("USER 10001:10001", dockerfile)
        self.assertNotIn("top-secret", dockerfile)

        persisted = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.assertEqual(persisted["image_version"], SANDBOX_IMAGE_VERSION)
        self.assertEqual(persisted["container_user"], "10001:10001")
        persisted_text = repr(persisted)
        self.assertNotIn("top-secret", persisted_text)
        self.assertNotIn("OPENAI", persisted_text)

        for call in self.docker.calls:
            self.assertEqual(call["env"], {"PATH": "safe-path", "SYSTEMROOT": "C:/Windows"})

    def test_setup_refuses_missing_daemon_without_build(self):
        self.docker.available = False
        with self.assertRaisesRegex(DockerUnavailableError, "daemon"):
            self.sandbox.setup()
        self.assertNotIn("build", self.docker.operations())
        self.assertFalse(self.config_path.exists())

    def test_setup_rejects_root_or_wrong_version_images(self):
        self.docker.image_user = "root"
        with self.assertRaisesRegex(SandboxError, "non-root"):
            self.sandbox.setup()
        self.assertFalse(self.config_path.exists())

        self.docker.image_user = "10001:10001"
        self.docker.image_version = "old"
        with self.assertRaisesRegex(SandboxError, "version label"):
            self.sandbox.setup(force=True)
        self.assertFalse(self.config_path.exists())

    def test_full_run_has_only_workspace_bind_no_host_secrets_and_bounds(self):
        self.sandbox.setup()
        result = self.sandbox.run("python3 -m unittest", self.workspace)

        self.assertTrue(result.ok)
        self.assertEqual(result.stdout, "ok\n")
        run_call = [call for call in self.docker.calls if call["argv"][1] == "run"][-1]
        argv = run_call["argv"]
        self.assertIn("--read-only", argv)
        self.assertEqual(argv[argv.index("--network") + 1], "bridge")
        self.assertEqual(argv[argv.index("--user") + 1], "10001:10001")
        self.assertEqual(argv[argv.index("--workdir") + 1], "/workspace")
        self.assertEqual(argv[argv.index("--pids-limit") + 1], "256")
        self.assertEqual(argv[argv.index("--memory") + 1], "2g")
        self.assertEqual(argv[argv.index("--cpus") + 1], "2.0")
        self.assertEqual(argv.count("--mount"), 1)
        mount = argv[argv.index("--mount") + 1]
        self.assertIn(str(self.workspace.resolve()), mount)
        self.assertEqual(mount.count("target="), 1)
        flattened = " ".join(argv).casefold()
        self.assertNotIn("docker.sock", flattened)
        self.assertNotIn("c:/users/private", flattened)
        self.assertNotIn("top-secret", flattened)
        self.assertNotIn("--env", argv)
        self.assertNotIn("--env-file", argv)
        self.assertEqual(run_call["max_output_bytes"], 2_048)
        self.assertEqual(run_call["timeout"], 30)

    def test_timeout_is_not_retried_and_forces_container_cleanup(self):
        self.sandbox.setup()
        self.docker.run_output = ProcessOutput(
            137,
            b"partial",
            b"",
            timed_out=True,
        )

        result = self.sandbox.run("long task", self.workspace)

        self.assertTrue(result.timed_out)
        self.assertEqual(self.docker.operations().count("run"), 1)
        self.assertEqual(self.docker.operations()[-1], "rm")
        cleanup = self.docker.calls[-1]["argv"]
        self.assertIn("--force", cleanup)

    def test_output_truncation_is_reported(self):
        self.sandbox.setup()
        self.docker.run_output = ProcessOutput(
            0,
            b"bounded",
            b"",
            stdout_truncated=True,
        )
        rendered = self.sandbox.run("verbose", self.workspace).render()
        self.assertIn("output truncated", rendered)

    def test_ready_full_bypasses_approval_and_routes_through_container(self):
        self.sandbox.setup()
        adapter = PermissionAdapter(AccessLevel.FULL, self.sandbox)
        self.assertEqual(adapter.access_level, AccessLevel.FULL)
        self.assertFalse(adapter.requires_approval(True))
        result = adapter.run_shell(
            "echo full",
            self.workspace,
            normal_runner=lambda _command: self.fail("normal runner must not be used"),
        )
        self.assertIn("exit code: 0", result)
        self.assertEqual(self.docker.operations().count("run"), 1)

    def test_corrupt_or_secret_bearing_config_fails_closed(self):
        self.config_path.parent.mkdir(parents=True)
        self.config_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "image_version": SANDBOX_IMAGE_VERSION,
                    "image": "ga3bad/coding-agent-sandbox:3",
                    "runtime": "docker",
                    "container_user": "10001:10001",
                    "configured_at": "now",
                    "api_key": "not-allowed",
                }
            ),
            encoding="utf-8",
        )

        status = self.sandbox.status()

        self.assertFalse(status.ready)
        self.assertIn("unsupported fields", status.reason)
        self.assertEqual(
            select_access_level("full", self.sandbox).effective,
            AccessLevel.NORMAL,
        )

    def test_config_parser_rejects_missing_or_root_user(self):
        self.config_path.parent.mkdir(parents=True)
        self.config_path.write_text("{}", encoding="utf-8")
        with self.assertRaises(SandboxConfigError):
            self.sandbox.load_config()


if __name__ == "__main__":
    unittest.main()
