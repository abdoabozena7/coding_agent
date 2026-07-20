"""Project protection through local Git checkpoints and optional GitHub backup.

The agent treats accepted outcomes as immutable recovery points.  GitHub is an
optional remote copy; local Git itself is what makes multi-step undo possible.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from .safety import redact_text


class VersionControlError(RuntimeError):
    """Raised when a requested protection operation cannot be completed safely."""


@dataclass(frozen=True)
class GitProtectionStatus:
    workspace: str
    git_available: bool
    dedicated_repository: bool
    github_connected: bool
    remote_url: str = ""
    branch: str = ""
    commit_count: int = 0
    dirty: bool = False
    gh_available: bool = False
    gh_authenticated: bool = False
    detail: str = ""

    @property
    def tier(self) -> str:
        if self.github_connected:
            return "github"
        if self.dedicated_repository:
            return "local_git"
        return "snapshot"

    @property
    def supports_multi_step_undo(self) -> bool:
        return self.dedicated_repository and self.commit_count > 0


@dataclass(frozen=True)
class VersionControlConfig:
    auto_checkpoint: bool = False
    auto_push: bool = False
    provider: str = "snapshot"


@dataclass(frozen=True)
class GitCheckpoint:
    commit: str
    subject: str
    created_at: str
    kind: str


class GitProtectionManager:
    """Own a *workspace-local* Git repository and accepted checkpoints.

    An ancestor repository deliberately does not count.  This prevents a user
    selecting a nested project from accidentally versioning or reverting its
    parent workspace.
    """

    CHECKPOINT_TRAILER = "GA3BAD-Checkpoint"
    CONFIG_PATH = Path(".coding-agent") / "version-control.json"
    _GITHUB_REMOTE_RE = re.compile(
        r"(?:github\.com[/:])(?P<owner>[^/\s:]+)/(?P<repo>[^/\s]+?)(?:\.git)?$",
        re.IGNORECASE,
    )
    _HIGH_CONFIDENCE_SECRET_RE = re.compile(
        r"(?:\bsk-[A-Za-z0-9_-]{16,}\b|"
        r"\bAIza[0-9A-Za-z_-]{20,}\b|"
        r"\bgh[opusr]_[A-Za-z0-9]{20,}\b|"
        r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b|"
        r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b|"
        r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY(?: BLOCK)?-----)",
        re.IGNORECASE,
    )
    _SAFE_EXCLUDES = (
        ".coding-agent/",
        ".env",
        ".env.*",
        "*.pem",
        "*.key",
        "*credentials*",
        "*secret*",
        "node_modules/",
        ".venv/",
        "venv/",
        "__pycache__/",
        "run-artifacts/",
        "output/playwright/",
        ".playwright-cli/",
    )

    def __init__(
        self,
        workspace: str | Path,
        *,
        runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
        which: Callable[[str], str | None] | None = None,
    ) -> None:
        self.workspace = Path(workspace).resolve(strict=True)
        self._runner = runner or subprocess.run
        self._which = which or shutil.which

    @property
    def config_path(self) -> Path:
        return self.workspace / self.CONFIG_PATH

    def _run(
        self,
        *args: str,
        check: bool = False,
        timeout: int = 30,
    ) -> subprocess.CompletedProcess[str]:
        try:
            result = self._runner(
                list(args),
                cwd=str(self.workspace),
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                timeout=timeout,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise VersionControlError(f"Could not run {' '.join(args[:2])}: {exc}") from exc
        if check and result.returncode != 0:
            message = (result.stderr or result.stdout or "command failed").strip()
            raise VersionControlError(redact_text(message, 1_000))
        return result

    def _git(self, *args: str, check: bool = False) -> subprocess.CompletedProcess[str]:
        return self._run("git", *args, check=check)

    def _repo_root(self) -> Path | None:
        if not self._which("git"):
            return None
        result = self._git("rev-parse", "--show-toplevel")
        if result.returncode != 0:
            return None
        try:
            return Path(result.stdout.strip()).resolve(strict=True)
        except (OSError, RuntimeError):
            return None

    def _is_dedicated_repository(self) -> bool:
        return self._repo_root() == self.workspace

    def _gh_authenticated(self) -> bool:
        if not self._which("gh"):
            return False
        return self._run("gh", "auth", "status", timeout=15).returncode == 0

    def inspect(self) -> GitProtectionStatus:
        git_available = bool(self._which("git"))
        dedicated = git_available and self._is_dedicated_repository()
        branch = ""
        remote_url = ""
        commit_count = 0
        dirty = False
        if dedicated:
            branch_result = self._git("branch", "--show-current")
            branch = branch_result.stdout.strip() if branch_result.returncode == 0 else ""
            remote_result = self._git("remote", "get-url", "origin")
            remote_url = remote_result.stdout.strip() if remote_result.returncode == 0 else ""
            count_result = self._git("rev-list", "--count", "HEAD")
            if count_result.returncode == 0 and count_result.stdout.strip().isdigit():
                commit_count = int(count_result.stdout.strip())
            status_result = self._git("status", "--porcelain")
            dirty = status_result.returncode == 0 and bool(status_result.stdout.strip())
        gh_available = bool(self._which("gh"))
        gh_authenticated = gh_available and self._gh_authenticated()
        github_connected = bool(
            dedicated and remote_url and self._GITHUB_REMOTE_RE.search(remote_url)
        )
        detail = (
            "Git is not installed. Only the current-run recovery snapshot is available."
            if not git_available
            else "This folder is inside another repository, but has no dedicated project history."
            if self._repo_root() is not None and not dedicated
            else "No dedicated Git history exists for this project."
            if not dedicated
            else "Protected locally and backed up on GitHub."
            if github_connected
            else "Protected by local Git history; no GitHub backup is connected."
        )
        return GitProtectionStatus(
            workspace=str(self.workspace),
            git_available=git_available,
            dedicated_repository=bool(dedicated),
            github_connected=github_connected,
            remote_url=remote_url,
            branch=branch,
            commit_count=commit_count,
            dirty=dirty,
            gh_available=gh_available,
            gh_authenticated=gh_authenticated,
            detail=detail,
        )

    def load_config(self) -> VersionControlConfig:
        try:
            payload = json.loads(self.config_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return VersionControlConfig()
        if not isinstance(payload, Mapping):
            return VersionControlConfig()
        return VersionControlConfig(
            auto_checkpoint=bool(payload.get("auto_checkpoint")),
            auto_push=bool(payload.get("auto_push")),
            provider=str(payload.get("provider") or "snapshot"),
        )

    def _save_config(self, config: VersionControlConfig) -> None:
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(
            json.dumps(asdict(config), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def configure(self, *, auto_checkpoint: bool, auto_push: bool, provider: str) -> None:
        self._save_config(VersionControlConfig(auto_checkpoint, auto_push, provider))

    def _write_local_excludes(self) -> None:
        resolved = self._git("rev-parse", "--git-path", "info/exclude", check=True).stdout.strip()
        exclude_path = Path(resolved)
        if not exclude_path.is_absolute():
            exclude_path = self.workspace / exclude_path
        exclude_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            existing = exclude_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            existing = ""
        lines = existing.splitlines()
        additions = [item for item in self._SAFE_EXCLUDES if item not in lines]
        if additions:
            separator = "" if not existing or existing.endswith("\n") else "\n"
            exclude_path.write_text(
                existing + separator + "\n".join(additions) + "\n",
                encoding="utf-8",
            )

    def _scan_staged_for_secrets(self) -> None:
        listed = self._git("diff", "--cached", "--name-only", "-z", check=True).stdout
        unsafe: list[str] = []
        for raw_name in listed.split("\0"):
            if not raw_name:
                continue
            path = self.workspace / raw_name
            try:
                if not path.is_file() or path.stat().st_size > 50 * 1024 * 1024:
                    if path.is_file():
                        unsafe.append(f"{raw_name} (larger than 50 MiB)")
                    continue
            except OSError:
                continue
            patch = self._git(
                "diff", "--cached", "--unified=0", "--", raw_name, check=True,
            ).stdout
            added_lines = "\n".join(
                line[1:]
                for line in patch.splitlines()
                if line.startswith("+") and not line.startswith("+++")
            )
            if self._HIGH_CONFIDENCE_SECRET_RE.search(added_lines):
                unsafe.append(f"{raw_name} (possible secret)")
        if unsafe:
            self._git("reset", check=False)
            raise VersionControlError(
                "Refusing to checkpoint files that may expose secrets: "
                + ", ".join(unsafe[:10])
            )

    def _commit(self, message: str, *, allow_empty: bool = False) -> str:
        args = [
            "-c", "user.name=GA3BAD Agent",
            "-c", "user.email=ga3bad-agent@local.invalid",
            "commit",
            "--no-gpg-sign",
        ]
        if allow_empty:
            args.append("--allow-empty")
        args.extend(("-m", message))
        self._git(*args, check=True)
        return self._git("rev-parse", "HEAD", check=True).stdout.strip()

    def ensure_local_history(self) -> GitProtectionStatus:
        status = self.inspect()
        if not status.git_available:
            raise VersionControlError("Git is not installed; install Git to enable multi-step undo.")
        if not status.dedicated_repository:
            self._git("init", "-b", "main", check=True)
        self._write_local_excludes()
        status = self.inspect()
        if status.commit_count == 0:
            self._git("add", "--all", check=True)
            self._scan_staged_for_secrets()
            self._commit(
                "chore: protect project baseline\n\n"
                f"{self.CHECKPOINT_TRAILER}: baseline",
                allow_empty=True,
            )
        self._save_config(
            VersionControlConfig(
                auto_checkpoint=True,
                auto_push=False,
                provider="local_git",
            )
        )
        return self.inspect()

    @staticmethod
    def _repo_slug(value: str) -> str:
        slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-.")
        return slug[:100] or "ga3bad-project"

    def connect_github_private(self, *, repository_name: str | None = None) -> GitProtectionStatus:
        status = self.inspect()
        if not status.gh_available:
            raise VersionControlError(
                "GitHub CLI is unavailable. Install `gh`, run `gh auth login`, then choose Refresh."
            )
        if not status.gh_authenticated:
            raise VersionControlError(
                "GitHub CLI is not authenticated. Run `gh auth login`, then choose Refresh."
            )
        if not status.dedicated_repository:
            status = self.ensure_local_history()
        if status.github_connected:
            self._save_config(VersionControlConfig(True, True, "github"))
            return status
        if status.remote_url:
            raise VersionControlError(
                "This project already has a non-GitHub `origin`. Connect GitHub manually "
                "with a different remote name, then choose Refresh."
            )
        name = self._repo_slug(repository_name or self.workspace.name)
        self._run(
            "gh", "repo", "create", name,
            "--private",
            "--source", str(self.workspace),
            "--remote", "origin",
            "--push",
            check=True,
            timeout=120,
        )
        status = self.inspect()
        if not status.github_connected:
            raise VersionControlError("GitHub reported success, but no GitHub origin was detected.")
        self._save_config(VersionControlConfig(True, True, "github"))
        return status

    def use_snapshot_only(self) -> GitProtectionStatus:
        self._save_config(VersionControlConfig(False, False, "snapshot"))
        return self.inspect()

    def create_checkpoint(self, label: str, *, kind: str = "accepted") -> str | None:
        config = self.load_config()
        if not config.auto_checkpoint:
            return None
        status = self.inspect()
        if not status.dedicated_repository:
            raise VersionControlError("Accepted checkpoint required, but local Git history is unavailable.")
        self._write_local_excludes()
        self._git("add", "--all", check=True)
        self._scan_staged_for_secrets()
        staged = self._git("diff", "--cached", "--quiet")
        if staged.returncode == 0:
            commit = self._git("rev-parse", "HEAD", check=True).stdout.strip()
            if config.auto_push:
                self._git("push", "-u", "origin", "HEAD", check=True)
            return commit
        safe_label = redact_text(str(label), 180).replace("\n", " ").strip()
        commit = self._commit(
            f"checkpoint: {safe_label or 'accepted result'}\n\n"
            f"{self.CHECKPOINT_TRAILER}: {kind}"
        )
        if config.auto_push:
            pushed = self._git("push", "-u", "origin", "HEAD", check=False)
            if pushed.returncode != 0:
                raise VersionControlError(
                    "The local checkpoint was created, but GitHub backup failed: "
                    + redact_text(pushed.stderr or pushed.stdout, 800)
                )
        return commit

    def history(self, limit: int = 20) -> tuple[GitCheckpoint, ...]:
        status = self.inspect()
        if not status.dedicated_repository or status.commit_count == 0:
            return ()
        record_sep = "\x1e"
        field_sep = "\x1f"
        result = self._git(
            "log",
            f"--max-count={max(1, min(int(limit), 100))}",
            f"--format=%H{field_sep}%s{field_sep}%aI{field_sep}%B{record_sep}",
            check=True,
        )
        checkpoints: list[GitCheckpoint] = []
        for record in result.stdout.split(record_sep):
            fields = record.strip().split(field_sep, 3)
            if len(fields) != 4:
                continue
            commit, subject, created_at, body = fields
            match = re.search(
                rf"^{re.escape(self.CHECKPOINT_TRAILER)}:\s*(\S+)",
                body,
                re.MULTILINE,
            )
            if match:
                checkpoints.append(GitCheckpoint(commit, subject, created_at, match.group(1)))
        return tuple(checkpoints)

    def _reverted_checkpoint_ids(self) -> set[str]:
        result = self._git("log", "--format=%B%x1e", check=True)
        return {
            match.group(1).casefold()
            for match in re.finditer(
                r"This reverts commit ([0-9a-f]{40})\.",
                result.stdout,
                re.IGNORECASE,
            )
        }

    def undo_candidates(self, limit: int = 100) -> tuple[GitCheckpoint, ...]:
        if not self.inspect().dedicated_repository:
            return ()
        reverted = self._reverted_checkpoint_ids()
        return tuple(
            item
            for item in self.history(limit)
            if item.kind == "accepted" and item.commit.casefold() not in reverted
        )

    def change_summary(self, commit: str) -> str:
        result = self._git(
            "show", "--no-renames", "--format=", "--shortstat", str(commit),
            check=True,
        )
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        return lines[-1] if lines else "no file changes"

    def diff(self, target: str | None = None, *, limit: int = 60_000) -> str:
        status = self.inspect()
        if not status.dedicated_repository:
            raise VersionControlError("Diff view needs a dedicated local Git repository.")
        selected = str(target or "").strip()
        if not selected:
            unstaged = self._git("diff", "--stat", "--patch", "--no-ext-diff", check=True).stdout
            staged = self._git(
                "diff", "--cached", "--stat", "--patch", "--no-ext-diff", check=True,
            ).stdout
            value = ""
            if staged.strip():
                value += "STAGED CHANGES\n" + staged
            if unstaged.strip():
                value += ("\n" if value else "") + "WORKING CHANGES\n" + unstaged
            untracked = self._git(
                "ls-files", "--others", "--exclude-standard", "-z", check=True,
            ).stdout
            new_file_diffs: list[str] = []
            for raw_name in untracked.split("\0"):
                if not raw_name:
                    continue
                path = self.workspace / raw_name
                try:
                    size = path.stat().st_size
                    if size > 1_000_000:
                        new_file_diffs.append(
                            f"diff --git a/{raw_name} b/{raw_name}\n"
                            f"new untracked file ({size} bytes; content preview omitted)"
                        )
                        continue
                    content = path.read_text(encoding="utf-8")
                except UnicodeError:
                    new_file_diffs.append(
                        f"diff --git a/{raw_name} b/{raw_name}\nnew untracked binary file"
                    )
                    continue
                except OSError:
                    continue
                added = "\n".join("+" + line for line in content.splitlines())
                new_file_diffs.append(
                    f"diff --git a/{raw_name} b/{raw_name}\n"
                    "new file mode (untracked)\n"
                    f"--- /dev/null\n+++ b/{raw_name}\n{added}"
                )
                if sum(len(item) for item in new_file_diffs) >= limit:
                    break
            if new_file_diffs:
                value += ("\n" if value else "") + "UNTRACKED FILES\n" + "\n".join(new_file_diffs)
            if not value:
                value = "Working tree is clean; there are no unaccepted changes."
        else:
            commit = selected
            if selected.isdigit():
                index = int(selected)
                checkpoints = self.history(max(index, 1))
                if index < 1 or index > len(checkpoints):
                    raise VersionControlError(
                        f"Checkpoint {index} is not available; run /versions to list valid numbers."
                    )
                commit = checkpoints[index - 1].commit
            result = self._git(
                "show", "--stat", "--patch", "--no-ext-diff", "--format=fuller", commit,
            )
            if result.returncode != 0:
                raise VersionControlError(
                    "Unknown checkpoint or commit: " + redact_text(result.stderr or selected, 500)
                )
            value = result.stdout
        redacted = redact_text(value)
        if len(redacted) > limit:
            return redacted[:limit] + f"\n... diff truncated by {len(redacted) - limit} characters"
        return redacted

    def undo(self, steps: int = 1) -> tuple[str, ...]:
        steps = int(steps)
        if steps < 1:
            raise VersionControlError("Undo steps must be a positive integer.")
        status = self.inspect()
        if not status.dedicated_repository:
            raise VersionControlError("Multi-step undo needs local Git history.")
        if status.dirty:
            raise VersionControlError(
                "Undo is blocked because the workspace has uncommitted changes. "
                "Finish, checkpoint, or discard that work first."
            )
        candidates = list(self.undo_candidates(100))
        if len(candidates) < steps:
            raise VersionControlError(
                f"Only {len(candidates)} accepted checkpoint(s) can be undone."
            )
        reverted: list[str] = []
        for item in candidates[:steps]:
            result = self._git("revert", "--no-edit", item.commit)
            if result.returncode != 0:
                self._git("revert", "--abort", check=False)
                raise VersionControlError(
                    "Undo conflicted and was aborted without accepting a partial result: "
                    + redact_text(result.stderr or result.stdout, 800)
                )
            reverted.append(item.commit)
        config = self.load_config()
        if config.auto_push:
            self._git("push", "origin", "HEAD", check=True)
        return tuple(reverted)

    def describe(self) -> Mapping[str, Any]:
        status = self.inspect()
        return {**asdict(status), "tier": status.tier, "config": asdict(self.load_config())}
