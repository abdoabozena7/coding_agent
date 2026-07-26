"""Security regression tests for the coding agent's tool boundary.

These tests intentionally use only the standard library so they can run in a
fresh checkout with ``python -m unittest``.
"""

from __future__ import annotations

import contextvars
import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agent import tools
from agent.tools import _security, grep, list_files, run_bash


class ToolSecurityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.workspace = self.root / "workspace"
        self.outside = self.root / "outside"
        self.workspace.mkdir()
        self.outside.mkdir()
        self.context = tools.workspace_context(self.workspace)
        self.context.__enter__()

    def tearDown(self) -> None:
        self.context.__exit__(None, None, None)
        self.temporary.cleanup()

    def call(self, name: str, **args: object) -> str:
        return tools.run_tool(name, args)

    def test_workspace_is_explicit_and_never_changes_process_cwd(self) -> None:
        original_cwd = Path.cwd()
        result = self.call("write_file", path="nested/example.txt", content="hello")

        self.assertIn("Wrote 5 characters", result)
        self.assertEqual((self.workspace / "nested/example.txt").read_text(), "hello")
        self.assertEqual(Path.cwd(), original_cwd)
        self.assertEqual(self.call("read_file", path="nested/example.txt"), "hello")

        unconfigured = contextvars.Context().run(
            tools.run_tool, "read_file", {"path": "nested/example.txt"}
        )
        self.assertIn("workspace is not configured", unconfigured)

    def test_workspace_context_is_nestable(self) -> None:
        second = self.root / "second"
        second.mkdir()
        with tools.workspace_context(second):
            self.call("write_file", path="second.txt", content="two")
        self.call("write_file", path="first.txt", content="one")

        self.assertEqual((second / "second.txt").read_text(), "two")
        self.assertEqual((self.workspace / "first.txt").read_text(), "one")

    def test_empty_optional_list_path_uses_the_documented_workspace_default(self) -> None:
        (self.workspace / "example.txt").write_text("safe", encoding="utf-8")

        result = self.call("list_files", path="")

        self.assertIn("example.txt", result)
        self.assertNotIn("invalid arguments", result)

    def test_relative_and_absolute_escape_attempts_are_rejected(self) -> None:
        secret = self.outside / "secret.txt"
        secret.write_text("outside-secret", encoding="utf-8")

        for path in ("../outside/secret.txt", str(secret)):
            with self.subTest(path=path):
                self.assertIn("escapes the active workspace", self.call("read_file", path=path))
                self.assertIn(
                    "escapes the active workspace",
                    self.call("write_file", path=path, content="changed"),
                )
                self.assertIn(
                    "escapes the active workspace",
                    self.call("grep", path=path, pattern="secret"),
                )

        self.assertEqual(secret.read_text(encoding="utf-8"), "outside-secret")

    def test_symlink_escapes_are_rejected_and_not_traversed(self) -> None:
        secret = self.outside / "secret.txt"
        secret.write_text("symlink-secret", encoding="utf-8")
        file_link = self.workspace / "file-link.txt"
        directory_link = self.workspace / "directory-link"
        try:
            file_link.symlink_to(secret)
            directory_link.symlink_to(self.outside, target_is_directory=True)
        except (OSError, NotImplementedError) as exc:
            self.skipTest(f"symlinks are unavailable: {exc}")

        self.assertIn("escapes the active workspace", self.call("read_file", path="file-link.txt"))
        self.assertIn(
            "escapes the active workspace",
            self.call("write_file", path="directory-link/new.txt", content="bad"),
        )
        listing = self.call("list_files")
        self.assertNotIn("file-link.txt", listing)
        self.assertNotIn("directory-link", listing)
        self.assertNotIn("symlink-secret", self.call("grep", pattern="symlink-secret"))
        self.assertFalse((self.outside / "new.txt").exists())

    def test_safe_internal_symlink_does_not_weaken_containment(self) -> None:
        target = self.workspace / "target.txt"
        target.write_text("inside", encoding="utf-8")
        link = self.workspace / "inside-link.txt"
        try:
            link.symlink_to(target)
        except (OSError, NotImplementedError) as exc:
            self.skipTest(f"symlinks are unavailable: {exc}")
        self.assertEqual(self.call("read_file", path="inside-link.txt"), "inside")

    def test_reserved_agent_state_is_inaccessible_and_hidden(self) -> None:
        state = self.workspace / ".coding-agent"
        state.mkdir()
        (state / "state.txt").write_text("private-state", encoding="utf-8")

        for name, args in (
            ("read_file", {"path": ".coding-agent/state.txt"}),
            ("write_file", {"path": ".coding-agent/new.txt", "content": "bad"}),
            (
                "edit_file",
                {"path": ".coding-agent/state.txt", "old_str": "private", "new_str": "public"},
            ),
            ("list_files", {"path": ".coding-agent"}),
            ("grep", {"path": ".coding-agent", "pattern": "private"}),
        ):
            with self.subTest(tool=name):
                self.assertIn("reserved for agent state", tools.run_tool(name, args))

        self.assertNotIn(".coding-agent", self.call("list_files"))
        self.assertNotIn(
            ".coding-agent/state.txt:", self.call("grep", pattern="private-state")
        )

    def test_sensitive_files_are_denied_but_env_example_is_allowed(self) -> None:
        (self.workspace / ".env").write_text("API_TOKEN=real-secret\n", encoding="utf-8")
        (self.workspace / ".env.local").write_text(
            "API_TOKEN=other-secret\n", encoding="utf-8"
        )
        (self.workspace / ".env.example").write_bytes(b"API_TOKEN=replace-me\n")
        credential_dir = self.workspace / ".aws"
        credential_dir.mkdir()
        (credential_dir / "credentials").write_text(
            "aws_secret_access_key=cloud-secret\n", encoding="utf-8"
        )

        for path in (".env", ".env.local", ".aws/credentials"):
            with self.subTest(path=path):
                result = self.call("read_file", path=path)
                self.assertIn("sensitive paths is denied", result)
                self.assertNotIn("real-secret", result)
                self.assertNotIn("other-secret", result)
                self.assertNotIn("cloud-secret", result)

        self.assertEqual(
            self.call("read_file", path=".env.example"), "API_TOKEN=replace-me\n"
        )
        listing = self.call("list_files")
        self.assertIn(".env.example", listing)
        self.assertNotIn(".env\n", listing + "\n")
        self.assertNotIn(".env.local", listing)
        self.assertNotIn(".aws", listing)

        search = self.call("grep", pattern="secret|replace-me")
        self.assertIn(".env.example:1: API_TOKEN=replace-me", search)
        self.assertNotIn("real-secret", search)
        self.assertNotIn("other-secret", search)
        self.assertNotIn("cloud-secret", search)
        self.assertIn(
            "sensitive paths is denied",
            self.call("grep", path=".env", pattern="API_TOKEN"),
        )

    def test_private_key_content_is_not_returned_under_an_innocent_name(self) -> None:
        key = (
            "-----BEGIN PRIVATE KEY-----\n"
            + ("A" * 80)
            + "\n-----END PRIVATE KEY-----\n"
        )
        (self.workspace / "innocent.txt").write_text(key, encoding="utf-8")

        result = self.call("read_file", path="innocent.txt")
        self.assertIn("protected by the sensitive-data policy", result)
        self.assertNotIn("A" * 20, result)
        self.assertNotIn("innocent.txt:", self.call("grep", pattern="BEGIN PRIVATE"))

    def test_arguments_are_strictly_validated_before_dispatch(self) -> None:
        cases = (
            ("read_file", {}, "missing required"),
            ("read_file", {"path": 7}, "must be string"),
            ("read_file", {"path": "x", "surprise": True}, "unknown argument"),
            ("list_files", [], "must be object"),
            ("run_bash", {"command": False}, "must be string"),
        )
        for name, args, expected in cases:
            with self.subTest(tool=name, args=args):
                result = tools.run_tool(name, args)  # type: ignore[arg-type]
                self.assertIn("Error: invalid arguments", result)
                self.assertIn(expected, result)

        self.assertTrue(tools.requires_approval("read_file", {"path": 7}))
        self.assertFalse(tools.requires_approval("read_file", {"path": "missing.txt"}))
        for schema in tools.TOOL_SCHEMAS:
            self.assertIs(
                schema["function"]["parameters"].get("additionalProperties"), False
            )

    def test_failed_atomic_write_preserves_the_original(self) -> None:
        target = self.workspace / "atomic.txt"
        target.write_text("original", encoding="utf-8")

        with mock.patch("agent.tools._security.os.replace", side_effect=OSError("injected")):
            result = self.call("write_file", path="atomic.txt", content="replacement")

        self.assertIn("atomic write failed", result)
        self.assertEqual(target.read_text(encoding="utf-8"), "original")
        self.assertEqual(list(self.workspace.glob(".agent-write-*")), [])

    def test_failed_atomic_edit_preserves_the_original(self) -> None:
        target = self.workspace / "edit.txt"
        target.write_text("before", encoding="utf-8")

        with mock.patch("agent.tools._security.os.replace", side_effect=OSError("injected")):
            result = self.call(
                "edit_file", path="edit.txt", old_str="before", new_str="after"
            )

        self.assertIn("atomic write failed", result)
        self.assertEqual(target.read_text(encoding="utf-8"), "before")
        self.assertEqual(list(self.workspace.glob(".agent-write-*")), [])

    def test_file_and_read_output_limits_are_enforced(self) -> None:
        oversized_write = "x" * (_security.MAX_WRITE_BYTES + 1)
        result = self.call("write_file", path="too-big.txt", content=oversized_write)
        self.assertIn("limit", result)
        self.assertFalse((self.workspace / "too-big.txt").exists())

        large_read = self.workspace / "large-read.txt"
        large_read.write_text(
            "y" * (_security.MAX_TOOL_OUTPUT_CHARS + 100), encoding="utf-8"
        )
        result = self.call("read_file", path="large-read.txt")
        self.assertIn("truncated at", result)
        self.assertLessEqual(len(result), _security.MAX_TOOL_OUTPUT_CHARS)

        too_large_to_read = self.workspace / "oversized.txt"
        too_large_to_read.write_bytes(b"z" * (_security.MAX_FILE_BYTES + 1))
        self.assertIn("read limit", self.call("read_file", path="oversized.txt"))
        grep_result = self.call("grep", path="oversized.txt", pattern="z")
        self.assertIn("no matches", grep_result)
        self.assertNotIn("not defined", grep_result)

    def test_listing_and_grep_have_deterministic_caps(self) -> None:
        for index in range(4):
            (self.workspace / f"file-{index}.txt").write_text(
                f"needle {index}\n", encoding="utf-8"
            )

        with mock.patch.object(list_files, "MAX_LIST_ENTRIES", 2):
            listing = self.call("list_files")
        self.assertIn("truncated at 2 entries", listing)

        with mock.patch.object(grep, "MAX_MATCHES", 2):
            matches = self.call("grep", pattern="needle")
        self.assertEqual(matches.count("needle"), 2)
        self.assertIn("truncated at 2 displayed matches", matches)

    def test_grep_rejects_catastrophic_backtracking_patterns(self) -> None:
        (self.workspace / "input.txt").write_text("a" * 2_000 + "!", encoding="utf-8")
        result = self.call("grep", pattern="(a+)+$", path="input.txt")
        self.assertIn("regex rejected by safety policy", result)

    def test_all_shell_commands_remain_approval_gated(self) -> None:
        commands = (
            "pytest",
            "python -m pytest",
            "python script.py",
            "env",
            "cat .env",
            "grep token .env",
            "git status",
            "pwd",
            "echo harmless",
        )
        for command in commands:
            with self.subTest(command=command):
                self.assertTrue(tools.requires_approval("run_bash", {"command": command}))
                self.assertFalse(run_bash._is_safe(command))

    def test_shell_uses_workspace_cwd_and_does_not_inherit_secrets(self) -> None:
        script = (
            "import os; "
            "print(os.getcwd()); "
            "print(os.getenv('OPENAI_API_KEY', '<missing>')); "
            "print(os.getenv('PYTHONPATH', '<missing>'))"
        )
        command = subprocess.list2cmdline([sys.executable, "-c", script])
        original_cwd = Path.cwd()
        with mock.patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "top-secret", "PYTHONPATH": "host-injection"},
        ):
            result = run_bash.run(command)

        self.assertIn("exit code: 0", result)
        self.assertIn(str(self.workspace), result)
        self.assertNotIn("top-secret", result)
        self.assertNotIn("host-injection", result)
        self.assertGreaterEqual(result.count("<missing>"), 2)
        self.assertEqual(Path.cwd(), original_cwd)

    def test_shell_capture_is_bounded_while_process_runs(self) -> None:
        amount = run_bash.MAX_CAPTURE_BYTES * 3
        command = subprocess.list2cmdline(
            [sys.executable, "-c", f"import sys; sys.stdout.write('x' * {amount})"]
        )
        result = run_bash.run(command)

        self.assertIn("exit code: 0", result)
        self.assertIn("truncated at", result)
        self.assertLessEqual(len(result), run_bash.MAX_OUTPUT_CHARS)

    def test_keyboard_interrupt_terminates_command_tree_before_propagating(self) -> None:
        process = mock.MagicMock()
        process.pid = 12345
        process.stdout = io.BytesIO(b"")
        process.stderr = io.BytesIO(b"")
        process.wait.side_effect = [KeyboardInterrupt(), 0]
        with mock.patch.object(run_bash.subprocess, "Popen", return_value=process):
            with mock.patch.object(run_bash, "_terminate") as terminate:
                with self.assertRaises(KeyboardInterrupt):
                    run_bash.run("long-running-command")
        terminate.assert_called_once_with(process)

    def test_secret_environment_scrubber_is_allowlist_based(self) -> None:
        scrubbed = run_bash._scrubbed_environment(
            {
                "PATH": "safe-path",
                "OPENAI_API_KEY": "secret",
                "DATABASE_URL": "secret-db",
                "PYTHONPATH": "injection",
                "BASH_ENV": "startup-hook",
            }
        )
        self.assertEqual(scrubbed, {"PATH": "safe-path"})


if __name__ == "__main__":
    unittest.main()
