"""Fail-closed access policy and a bounded Docker command sandbox.

``normal`` access is an adapter over the agent's existing approval-gated tool
behavior.  ``full`` access is granted only after an explicit one-time Docker
setup has produced the exact versioned, non-root image described here.  This
module never installs or starts Docker and never falls back to an unrestricted
host shell.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, BinaryIO


SANDBOX_SCHEMA_VERSION = 1
SANDBOX_IMAGE_VERSION = "3"
SANDBOX_IMAGE = f"ga3bad/coding-agent-sandbox:{SANDBOX_IMAGE_VERSION}"
SANDBOX_LABEL = "io.ga3bad.sandbox.version"
SANDBOX_USER = "10001:10001"
DOCKER_RUNTIME = "docker"
MAX_COMMAND_CHARS = 32_768


class AccessLevel(str, Enum):
    """Workspace mutation policy selected for this process."""

    NORMAL = "normal"
    FULL = "full"

    @classmethod
    def parse(cls, value: str | "AccessLevel") -> "AccessLevel":
        if isinstance(value, cls):
            return value
        normalized = str(value or "normal").strip().lower()
        aliases = {"default": "normal", "safe": "normal", "sandbox": "full"}
        try:
            return cls(aliases.get(normalized, normalized))
        except ValueError as exc:
            raise ValueError("access level must be 'normal' or 'full'") from exc


class SandboxError(RuntimeError):
    """Base error for safe, user-displayable sandbox failures."""


class DockerUnavailableError(SandboxError):
    pass


class SandboxNotReadyError(SandboxError):
    pass


class SandboxConfigError(SandboxError):
    pass


@dataclass(frozen=True, slots=True)
class SandboxLimits:
    timeout_seconds: float = 120.0
    max_output_bytes: int = 10_000
    memory: str = "2g"
    cpus: float = 2.0
    pids: int = 256
    tmpfs_size: str = "256m"

    def __post_init__(self) -> None:
        if not 1 <= self.timeout_seconds <= 3_600:
            raise ValueError("sandbox timeout must be between 1 and 3600 seconds")
        if not 1_024 <= self.max_output_bytes <= 10_000_000:
            raise ValueError("sandbox output bound must be between 1024 and 10000000 bytes")
        if not 0.1 <= self.cpus <= 64:
            raise ValueError("sandbox CPU limit must be between 0.1 and 64")
        if not 16 <= self.pids <= 4_096:
            raise ValueError("sandbox PID limit must be between 16 and 4096")
        for label, value in (("memory", self.memory), ("tmpfs_size", self.tmpfs_size)):
            if not value or any(character not in "0123456789.kmgtKMGT" for character in value):
                raise ValueError(f"invalid sandbox {label} value")


@dataclass(frozen=True, slots=True)
class SandboxConfig:
    schema_version: int = SANDBOX_SCHEMA_VERSION
    image_version: str = SANDBOX_IMAGE_VERSION
    image: str = SANDBOX_IMAGE
    runtime: str = DOCKER_RUNTIME
    container_user: str = SANDBOX_USER
    configured_at: str = ""

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SandboxConfig":
        allowed = {
            "schema_version",
            "image_version",
            "image",
            "runtime",
            "container_user",
            "configured_at",
        }
        unknown = set(value) - allowed
        if unknown:
            raise SandboxConfigError(
                f"sandbox config has unsupported fields: {', '.join(sorted(map(str, unknown)))}"
            )
        try:
            config = cls(
                schema_version=int(value["schema_version"]),
                image_version=str(value["image_version"]),
                image=str(value["image"]),
                runtime=str(value["runtime"]),
                container_user=str(value["container_user"]),
                configured_at=str(value.get("configured_at") or ""),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise SandboxConfigError("sandbox config is incomplete or malformed") from exc
        config.validate_current()
        return config

    def validate_current(self) -> None:
        if self.schema_version != SANDBOX_SCHEMA_VERSION:
            raise SandboxConfigError("sandbox config schema is out of date")
        if self.image_version != SANDBOX_IMAGE_VERSION or self.image != SANDBOX_IMAGE:
            raise SandboxConfigError("sandbox image version is out of date")
        if self.runtime != DOCKER_RUNTIME:
            raise SandboxConfigError("only the Docker sandbox runtime is supported")
        if _is_root_user(self.container_user):
            raise SandboxConfigError("sandbox config must select a non-root container user")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "image_version": self.image_version,
            "image": self.image,
            "runtime": self.runtime,
            "container_user": self.container_user,
            "configured_at": self.configured_at,
        }


@dataclass(frozen=True, slots=True)
class ProcessOutput:
    returncode: int
    stdout: bytes = b""
    stderr: bytes = b""
    timed_out: bool = False
    stdout_truncated: bool = False
    stderr_truncated: bool = False


@dataclass(frozen=True, slots=True)
class SandboxStatus:
    ready: bool
    docker_available: bool
    configured: bool
    reason: str = ""
    image: str = SANDBOX_IMAGE
    docker_version: str | None = None


@dataclass(frozen=True, slots=True)
class SandboxCommandResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    output_truncated: bool = False

    @property
    def ok(self) -> bool:
        return not self.timed_out and self.returncode == 0

    def render(self) -> str:
        if self.timed_out:
            return "Error: sandbox command timed out"
        parts = [f"exit code: {self.returncode}"]
        if self.stdout:
            parts.append(f"stdout:\n{self.stdout}")
        if self.stderr:
            parts.append(f"stderr:\n{self.stderr}")
        if len(parts) == 1:
            parts.append("(no output)")
        if self.output_truncated:
            parts.append("... (output truncated)")
        return "\n".join(parts)


@dataclass(frozen=True, slots=True)
class AccessSelection:
    requested: AccessLevel
    effective: AccessLevel
    reason: str = ""

    @property
    def downgraded(self) -> bool:
        return self.requested is not self.effective


ProcessRunner = Callable[..., ProcessOutput]


def sandbox_config_path(
    environ: Mapping[str, str] | None = None,
    *,
    home: str | os.PathLike[str] | None = None,
) -> Path:
    """Return the per-user, non-workspace platform configuration path."""

    values = os.environ if environ is None else environ
    if values.get("APPDATA"):
        return Path(values["APPDATA"]).expanduser() / "GA3BAD" / "sandbox.json"
    if values.get("XDG_CONFIG_HOME"):
        return Path(values["XDG_CONFIG_HOME"]).expanduser() / "ga3bad" / "sandbox.json"
    root = Path(home).expanduser() if home is not None else Path.home()
    return root / ".config" / "ga3bad" / "sandbox.json"


class DockerSandbox:
    """Validate, explicitly set up, and run the versioned Full sandbox."""

    def __init__(
        self,
        *,
        config_path: str | os.PathLike[str] | None = None,
        process_runner: ProcessRunner | None = None,
        environ: Mapping[str, str] | None = None,
        docker_executable: str = "docker",
        limits: SandboxLimits | None = None,
    ) -> None:
        self.environ = dict(os.environ if environ is None else environ)
        self.config_path = Path(config_path) if config_path else sandbox_config_path(self.environ)
        self.process_runner = process_runner or _bounded_process
        self.docker_executable = str(docker_executable).strip() or "docker"
        self.limits = limits or SandboxLimits()

    def load_config(self) -> SandboxConfig | None:
        if not self.config_path.exists():
            return None
        try:
            value = json.loads(self.config_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SandboxConfigError("sandbox config could not be read") from exc
        if not isinstance(value, Mapping):
            raise SandboxConfigError("sandbox config must contain a JSON object")
        return SandboxConfig.from_dict(value)

    def status(self) -> SandboxStatus:
        try:
            config = self.load_config()
        except SandboxConfigError as exc:
            return SandboxStatus(False, False, True, str(exc))

        available, version, reason = self._docker_version()
        if not available:
            return SandboxStatus(
                False,
                False,
                config is not None,
                reason,
                docker_version=version,
            )
        if config is None:
            return SandboxStatus(
                False,
                True,
                False,
                "Full access needs one-time sandbox setup",
                docker_version=version,
            )
        valid, reason = self._validate_image(config)
        return SandboxStatus(
            valid,
            True,
            True,
            "" if valid else reason,
            image=config.image,
            docker_version=version,
        )

    def setup(self, *, force: bool = False) -> SandboxConfig:
        """Explicitly build/validate the image and persist a secret-free marker.

        This is the setup-wizard action.  It never installs Docker and never
        attempts to start its daemon.  Repeated calls are no-ops while the
        configured image remains valid, unless ``force`` is requested.
        """

        if not force:
            current = self.status()
            if current.ready:
                config = self.load_config()
                assert config is not None
                return config

        available, _version, reason = self._docker_version()
        if not available:
            raise DockerUnavailableError(
                reason or "Docker is unavailable; install/start it before Full setup"
            )

        built = self._run_process(
            [
                self.docker_executable,
                "build",
                "--tag",
                SANDBOX_IMAGE,
                "--label",
                f"{SANDBOX_LABEL}={SANDBOX_IMAGE_VERSION}",
                "-",
            ],
            input_data=_DOCKERFILE.encode("utf-8"),
            timeout=max(self.limits.timeout_seconds, 600.0),
            max_output_bytes=max(self.limits.max_output_bytes, 100_000),
        )
        if built.timed_out:
            raise SandboxError("Docker sandbox image build timed out")
        if built.returncode != 0:
            raise SandboxError("Docker sandbox image build failed: " + _process_error(built))

        candidate = SandboxConfig(
            configured_at=datetime.now(timezone.utc).isoformat(timespec="seconds")
        )
        valid, reason = self._validate_image(candidate)
        if not valid:
            raise SandboxError("built sandbox image failed validation: " + reason)
        self._save_config(candidate)
        return candidate

    def run(
        self,
        command: str,
        workspace: str | os.PathLike[str],
    ) -> SandboxCommandResult:
        """Execute one shell command in the ready Full sandbox."""

        if not isinstance(command, str) or not command.strip():
            raise ValueError("sandbox command must be a non-empty string")
        if "\x00" in command:
            raise ValueError("sandbox command must not contain NUL bytes")
        if len(command) > MAX_COMMAND_CHARS:
            raise ValueError(f"sandbox command exceeds {MAX_COMMAND_CHARS} characters")
        root = Path(workspace).expanduser().resolve(strict=True)
        if not root.is_dir():
            raise ValueError("sandbox workspace must be a directory")
        if "," in str(root):
            raise ValueError("Docker bind-mount paths containing commas are unsupported")

        status = self.status()
        if not status.ready:
            raise SandboxNotReadyError(
                status.reason or "Full sandbox is not ready; using a host shell is forbidden"
            )
        config = self.load_config()
        assert config is not None

        container_name = "ga3bad-" + uuid.uuid4().hex
        mount = f"type=bind,source={root},target=/workspace"
        argv = [
            self.docker_executable,
            "run",
            "--rm",
            "--name",
            container_name,
            "--network",
            "bridge",
            "--read-only",
            "--security-opt",
            "no-new-privileges",
            "--cap-drop",
            "ALL",
            "--pids-limit",
            str(self.limits.pids),
            "--memory",
            self.limits.memory,
            "--cpus",
            str(self.limits.cpus),
            "--tmpfs",
            f"/tmp:rw,nosuid,nodev,size={self.limits.tmpfs_size}",
            "--mount",
            mount,
            "--workdir",
            "/workspace",
            "--user",
            config.container_user,
            config.image,
            "/bin/bash",
            "-lc",
            command,
        ]
        output = self._run_process(
            argv,
            timeout=self.limits.timeout_seconds,
            max_output_bytes=self.limits.max_output_bytes,
        )
        if output.timed_out:
            # Killing the attached CLI is not enough on every Docker backend;
            # explicitly remove the uniquely named container as a bounded
            # cleanup action.  Never retry the uncertain workload.
            self._run_process(
                [self.docker_executable, "rm", "--force", container_name],
                timeout=10.0,
                max_output_bytes=2_048,
            )
        return SandboxCommandResult(
            returncode=output.returncode,
            stdout=_decode(output.stdout),
            stderr=_decode(output.stderr),
            timed_out=output.timed_out,
            output_truncated=output.stdout_truncated or output.stderr_truncated,
        )

    def _docker_version(self) -> tuple[bool, str | None, str]:
        try:
            output = self._run_process(
                [
                    self.docker_executable,
                    "version",
                    "--format",
                    "{{json .Server.Version}}",
                ],
                timeout=10.0,
                max_output_bytes=8_192,
            )
        except OSError as exc:
            return False, None, f"Docker is not installed or executable: {exc}"
        if output.timed_out:
            return False, None, "Docker availability check timed out"
        if output.returncode != 0:
            return (
                False,
                None,
                "Docker is unavailable or its daemon is not running: " + _process_error(output),
            )
        raw = _decode(output.stdout).strip()
        try:
            parsed = json.loads(raw)
            version = str(parsed) if parsed is not None else raw
        except json.JSONDecodeError:
            version = raw
        return True, version or "unknown", ""

    def _validate_image(self, config: SandboxConfig) -> tuple[bool, str]:
        try:
            config.validate_current()
        except SandboxConfigError as exc:
            return False, str(exc)
        output = self._run_process(
            [
                self.docker_executable,
                "image",
                "inspect",
                config.image,
                "--format",
                "{{json .Config}}",
            ],
            timeout=15.0,
            max_output_bytes=32_768,
        )
        if output.timed_out:
            return False, "sandbox image validation timed out"
        if output.returncode != 0:
            return False, "versioned sandbox image is missing"
        try:
            image_config = json.loads(_decode(output.stdout))
        except json.JSONDecodeError:
            return False, "Docker returned malformed sandbox image metadata"
        if not isinstance(image_config, Mapping):
            return False, "Docker returned incomplete sandbox image metadata"
        user = str(image_config.get("User") or "").strip()
        if _is_root_user(user) or user != config.container_user:
            return False, "sandbox image is not pinned to the expected non-root user"
        labels = image_config.get("Labels")
        if not isinstance(labels, Mapping) or str(labels.get(SANDBOX_LABEL)) != config.image_version:
            return False, "sandbox image version label does not match this agent"
        return True, ""

    def _save_config(self, config: SandboxConfig) -> None:
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.config_path.with_name(
            f".{self.config_path.name}.{os.getpid()}.tmp"
        )
        try:
            temporary.write_text(
                json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            try:
                temporary.chmod(0o600)
            except OSError:
                pass
            os.replace(temporary, self.config_path)
        except OSError as exc:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise SandboxConfigError("sandbox config could not be saved") from exc

    def _run_process(
        self,
        argv: list[str],
        *,
        input_data: bytes | None = None,
        timeout: float,
        max_output_bytes: int,
    ) -> ProcessOutput:
        return self.process_runner(
            tuple(argv),
            input_data=input_data,
            timeout=timeout,
            max_output_bytes=max_output_bytes,
            env=_scrubbed_client_environment(self.environ),
        )


def select_access_level(
    requested: str | AccessLevel,
    sandbox: DockerSandbox,
) -> AccessSelection:
    """Resolve Full fail-closed, explicitly downgrading to Normal if unready."""

    requested_level = AccessLevel.parse(requested)
    if requested_level is AccessLevel.NORMAL:
        return AccessSelection(requested_level, AccessLevel.NORMAL)
    status = sandbox.status()
    if status.ready:
        return AccessSelection(requested_level, AccessLevel.FULL)
    reason = status.reason or "Full sandbox is not ready"
    return AccessSelection(
        requested_level,
        AccessLevel.NORMAL,
        f"Full access unavailable; using Normal: {reason}",
    )


class PermissionAdapter:
    """Keep Normal behavior intact while routing ready Full shell work safely."""

    def __init__(self, requested: str | AccessLevel, sandbox: DockerSandbox) -> None:
        self.sandbox = sandbox
        self.selection = select_access_level(requested, sandbox)

    @property
    def access_level(self) -> AccessLevel:
        return self.selection.effective

    def requires_approval(self, normal_requirement: bool = True) -> bool:
        return bool(normal_requirement) if self.access_level is AccessLevel.NORMAL else False

    def run_shell(
        self,
        command: str,
        workspace: str | os.PathLike[str],
        *,
        normal_runner: Callable[[str], str],
    ) -> str:
        if self.access_level is AccessLevel.NORMAL:
            return normal_runner(command)
        return self.sandbox.run(command, workspace).render()


@dataclass
class _Capture:
    limit: int
    data: bytearray = field(default_factory=bytearray)
    truncated: bool = False


def _drain(stream: BinaryIO, capture: _Capture) -> None:
    try:
        while True:
            chunk = stream.read(8_192)
            if not chunk:
                return
            remaining = capture.limit - len(capture.data)
            if remaining > 0:
                capture.data.extend(chunk[:remaining])
            if len(chunk) > max(remaining, 0):
                capture.truncated = True
    except (OSError, ValueError):
        return
    finally:
        try:
            stream.close()
        except OSError:
            pass


def _bounded_process(
    argv: tuple[str, ...],
    *,
    input_data: bytes | None,
    timeout: float,
    max_output_bytes: int,
    env: Mapping[str, str],
) -> ProcessOutput:
    """Run a no-shell subprocess while keeping each output stream bounded."""

    process = subprocess.Popen(
        list(argv),
        stdin=subprocess.PIPE if input_data is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False,
        env=dict(env),
        start_new_session=os.name != "nt",
        creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0),
    )
    assert process.stdout is not None
    assert process.stderr is not None
    stdout = _Capture(max_output_bytes)
    stderr = _Capture(max_output_bytes)
    readers = [
        threading.Thread(target=_drain, args=(process.stdout, stdout), daemon=True),
        threading.Thread(target=_drain, args=(process.stderr, stderr), daemon=True),
    ]
    for reader in readers:
        reader.start()

    if input_data is not None and process.stdin is not None:
        try:
            process.stdin.write(input_data)
            process.stdin.close()
        except (BrokenPipeError, OSError):
            pass

    timed_out = False
    try:
        returncode = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        _terminate(process, env)
        try:
            returncode = process.wait(timeout=5)
        except (subprocess.TimeoutExpired, OSError):
            returncode = -1
    except KeyboardInterrupt:
        _terminate(process, env)
        raise
    finally:
        for reader in readers:
            reader.join(timeout=2)

    return ProcessOutput(
        returncode=returncode,
        stdout=bytes(stdout.data),
        stderr=bytes(stderr.data),
        timed_out=timed_out,
        stdout_truncated=stdout.truncated,
        stderr_truncated=stderr.truncated,
    )


def _terminate(process: subprocess.Popen[bytes], env: Mapping[str, str]) -> None:
    try:
        if os.name == "nt":
            result = subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=dict(env),
                timeout=5,
                check=False,
            )
            if result.returncode != 0:
                process.kill()
        else:
            os.killpg(process.pid, signal.SIGKILL)
    except (OSError, ProcessLookupError, subprocess.TimeoutExpired):
        try:
            process.kill()
        except OSError:
            pass


def _scrubbed_client_environment(source: Mapping[str, str]) -> dict[str, str]:
    # Docker CLI needs executable/system paths only.  HOME/USERPROFILE,
    # DOCKER_CONFIG, credentials, API keys, and language injection variables
    # are absent by construction and none of these values are passed with
    # ``docker run --env``.
    allowed = {
        "path",
        "pathext",
        "systemroot",
        "windir",
        "comspec",
        "temp",
        "tmp",
        "tmpdir",
        "lang",
        "lc_all",
        "term",
    }
    return {key: value for key, value in source.items() if key.casefold() in allowed}


def _is_root_user(value: str) -> bool:
    user = str(value or "").strip().casefold().split(":", 1)[0]
    return user in {"", "0", "root"}


def _decode(value: bytes) -> str:
    return value.decode("utf-8", errors="replace")


def _process_error(output: ProcessOutput) -> str:
    message = _decode(output.stderr or output.stdout).strip()
    return message[:2_000] or f"exit code {output.returncode}"


_DOCKERFILE = f"""\
FROM ubuntu:24.04
LABEL {SANDBOX_LABEL}=\"{SANDBOX_IMAGE_VERSION}\"
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \\
    bash ca-certificates curl git build-essential python3 python3-pip nodejs npm \\
    && rm -rf /var/lib/apt/lists/*
RUN groupadd --gid 10001 agent && useradd --uid 10001 --gid 10001 --create-home agent
ENV HOME=/tmp
USER {SANDBOX_USER}
WORKDIR /workspace
CMD [\"/bin/bash\"]
"""


__all__ = [
    "AccessLevel",
    "AccessSelection",
    "DockerSandbox",
    "DockerUnavailableError",
    "PermissionAdapter",
    "ProcessOutput",
    "SandboxCommandResult",
    "SandboxConfig",
    "SandboxConfigError",
    "SandboxError",
    "SandboxLimits",
    "SandboxNotReadyError",
    "SandboxStatus",
    "sandbox_config_path",
    "select_access_level",
]
