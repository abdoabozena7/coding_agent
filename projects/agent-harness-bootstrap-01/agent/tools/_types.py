"""Typed metadata and results for every model-facing workspace tool."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping


ToolRunner = Callable[..., Any]
ApprovalDecider = Callable[[Mapping[str, Any]], bool]


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

    @property
    def name(self) -> str:
        return str(self.schema.get("function", {}).get("name", ""))

    def approval_required(self, args: Mapping[str, Any]) -> bool:
        if callable(self.requires_approval):
            return bool(self.requires_approval(args))
        return bool(self.requires_approval)


__all__ = ["ApprovalDecider", "ToolExecutionResult", "ToolRunner", "ToolSpec"]
