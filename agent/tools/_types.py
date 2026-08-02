"""Typed metadata and results for every model-facing workspace tool."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Callable, Iterable, Mapping


ToolRunner = Callable[..., Any]
ApprovalDecider = Callable[[Mapping[str, Any]], bool]
MutationPathResolver = Callable[[Mapping[str, Any]], Iterable[str]]


def _normalise_relative_path(value: object) -> str:
    text = str(value or "").strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return "" if text in {"", "."} else text.rstrip("/")


@dataclass(frozen=True, slots=True)
class MutationFootprintV1:
    """The bounded files a mutating tool may change as a documented side effect.

    ``accepted_paths`` come from the reviewed plan. ``derived_paths`` are not
    model-authored permissions: they are fixed, tool-owned outputs such as an
    npm lockfile.  Keeping the two sets separate makes the expansion auditable
    and prevents a command from turning one accepted file into a directory-wide
    write lease.
    """

    tool_name: str
    accepted_paths: tuple[str, ...] = ()
    derived_paths: tuple[str, ...] = ()
    version: int = 1

    @classmethod
    def build(
        cls,
        tool_name: str,
        *,
        accepted_paths: Iterable[str] = (),
        derived_paths: Iterable[str] = (),
    ) -> "MutationFootprintV1":
        accepted_values = [
            path
            for item in accepted_paths
            if (path := _normalise_relative_path(item))
        ]
        derived_values = [
            path
            for item in derived_paths
            if (path := _normalise_relative_path(item))
        ]
        accepted = tuple(dict.fromkeys(accepted_values))
        derived = tuple(
            path for path in dict.fromkeys(derived_values) if path not in accepted
        )
        return cls(str(tool_name), accepted, derived)

    @property
    def effective_paths(self) -> tuple[str, ...]:
        return (*self.accepted_paths, *self.derived_paths)

    @property
    def fingerprint(self) -> str:
        payload = {
            "version": self.version,
            "tool_name": self.tool_name,
            "accepted_paths": self.accepted_paths,
            "derived_paths": self.derived_paths,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class ToolExecutionResult:
    """Harness-owned interpretation of one tool invocation."""

    ok: bool
    output: str
    data: Mapping[str, Any] = field(default_factory=dict)
    changed_paths: tuple[str, ...] = ()
    error_code: str | None = None

    @classmethod
    def from_output(
        cls,
        output: str,
        *,
        changed_paths: tuple[str, ...] = (),
    ) -> "ToolExecutionResult":
        text = str(output)
        failed = text.startswith(("Error:", "Permission denied"))
        return cls(
            not failed,
            text,
            changed_paths=() if failed else changed_paths,
            error_code="tool_error" if failed else None,
        )


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """Single source of truth for schema, policy, and execution behavior."""

    schema: Mapping[str, Any]
    runner: ToolRunner
    risk: str
    category: str
    mutates_workspace: bool = False
    requires_approval: bool | ApprovalDecider = True
    path_fields: tuple[str, ...] = ()
    lifecycle: str = "one_shot"
    capability: str | None = None
    derived_mutation_paths: MutationPathResolver | None = None
    result_contract: Mapping[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return str(self.schema.get("function", {}).get("name", ""))

    def approval_required(self, args: Mapping[str, Any]) -> bool:
        if callable(self.requires_approval):
            return bool(self.requires_approval(args))
        return bool(self.requires_approval)


__all__ = [
    "ApprovalDecider", "MutationFootprintV1", "MutationPathResolver",
    "ToolExecutionResult", "ToolRunner", "ToolSpec",
]
