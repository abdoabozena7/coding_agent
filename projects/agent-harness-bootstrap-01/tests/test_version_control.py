from __future__ import annotations

import tempfile
import unittest
import shutil
import subprocess
from pathlib import Path

from agent.version_control import GitProtectionManager, VersionControlError


class GitProtectionManagerTests(unittest.TestCase):
    def test_ancestor_repository_does_not_count_as_project_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            GitProtectionManager(parent).ensure_local_history()
            child = parent / "nested-project"
            child.mkdir()

            status = GitProtectionManager(child).inspect()

            self.assertFalse(status.dedicated_repository)
            self.assertEqual(status.tier, "snapshot")
            self.assertIn("inside another repository", status.detail)

    def test_local_history_excludes_environment_secrets_and_creates_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "app.py").write_text("print('safe')\n", encoding="utf-8")
            (workspace / ".env").write_text(
                "OPENAI_API_KEY=not-a-real-but-long-secret-value\n",
                encoding="utf-8",
            )
            manager = GitProtectionManager(workspace)

            status = manager.ensure_local_history()

            self.assertTrue(status.dedicated_repository)
            self.assertEqual(status.commit_count, 1)
            tracked = manager._git("ls-files", check=True).stdout.splitlines()
            self.assertIn("app.py", tracked)
            self.assertNotIn(".env", tracked)
            self.assertTrue(manager.load_config().auto_checkpoint)
            self.assertFalse(manager.load_config().auto_push)

    def test_secret_scan_refuses_nonstandard_secret_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "config.txt").write_text(
                "token=ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ123456\n",
                encoding="utf-8",
            )
            manager = GitProtectionManager(workspace)

            with self.assertRaisesRegex(VersionControlError, "possible secret"):
                manager.ensure_local_history()

    def test_accepted_checkpoints_can_be_undone_one_at_a_time(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            target = workspace / "value.txt"
            target.write_text("baseline", encoding="utf-8")
            manager = GitProtectionManager(workspace)
            manager.ensure_local_history()
            target.write_text("one", encoding="utf-8")
            first = manager.create_checkpoint("first accepted")
            target.write_text("two", encoding="utf-8")
            second = manager.create_checkpoint("second accepted")

            self.assertTrue(first)
            self.assertTrue(second)
            self.assertEqual(target.read_text(encoding="utf-8"), "two")
            self.assertIn("second accepted", manager.diff("1"))
            self.assertIn("1 file changed", manager.change_summary(str(second)))

            manager.undo(1)
            self.assertEqual(target.read_text(encoding="utf-8"), "one")
            self.assertEqual(len(manager.undo_candidates()), 1)

            manager.undo(1)
            self.assertEqual(target.read_text(encoding="utf-8"), "baseline")
            self.assertEqual(manager.undo_candidates(), ())

    def test_undo_refuses_dirty_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            target = workspace / "value.txt"
            target.write_text("baseline", encoding="utf-8")
            manager = GitProtectionManager(workspace)
            manager.ensure_local_history()
            target.write_text("accepted", encoding="utf-8")
            manager.create_checkpoint("accepted")
            target.write_text("uncommitted", encoding="utf-8")

            with self.assertRaisesRegex(VersionControlError, "uncommitted changes"):
                manager.undo(1)

    def test_current_diff_includes_untracked_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            manager = GitProtectionManager(workspace)
            manager.ensure_local_history()
            (workspace / "new.py").write_text("print('visible')\n", encoding="utf-8")

            preview = manager.diff()

            self.assertIn("UNTRACKED FILES", preview)
            self.assertIn("new.py", preview)
            self.assertIn("+print('visible')", preview)

    def test_github_connection_creates_private_repo_and_enables_push(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "app.py").write_text("print('safe')\n", encoding="utf-8")
            manager = GitProtectionManager(workspace)
            manager.ensure_local_history()
            original_run = manager._run
            calls: list[tuple[str, ...]] = []

            def fake_run(*args: str, **kwargs):
                if args and args[0] == "gh":
                    calls.append(tuple(args))
                    if args[1:3] == ("auth", "status"):
                        return subprocess.CompletedProcess(args, 0, "", "")
                    if args[1:3] == ("repo", "create"):
                        manager._git(
                            "remote", "add", "origin",
                            "https://github.com/example/private-project.git",
                            check=True,
                        )
                        return subprocess.CompletedProcess(args, 0, "", "")
                return original_run(*args, **kwargs)

            real_git = shutil.which("git")
            manager._which = lambda name: real_git if name == "git" else "gh"
            manager._run = fake_run

            status = manager.connect_github_private(repository_name="private-project")

            self.assertTrue(status.github_connected)
            create = next(call for call in calls if call[1:3] == ("repo", "create"))
            self.assertIn("--private", create)
            self.assertIn("--push", create)
            self.assertTrue(manager.load_config().auto_push)


if __name__ == "__main__":
    unittest.main()
