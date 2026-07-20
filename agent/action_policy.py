"""Human-centered action, question, and plan attention policies."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


class ApprovalRequirement(str, Enum):
    AUTO = "auto"
    ONCE = "once"
    SESSION = "session"


@dataclass(frozen=True, slots=True)
class ActionPolicyDecision:
    requirement: ApprovalRequirement
    group: str
    reason: str
    scope: str = "project"


_READ_TOOLS = frozenset(
    {"read_file", "list_files", "grep", "poll_process", "read_process_output", "inspect_preview"}
)
_WRITE_TOOLS = frozenset({"write_file", "edit_file", "apply_patch", "materialize_artifact"})
_SAFE_COMMAND_RE = re.compile(
    r"^\s*(?:"
    r"git\s+(?:status|diff|log|show)\b|"
    r"(?:python(?:\.exe)?\s+-m\s+)?pytest\b|"
    r"python(?:\.exe)?\s+-m\s+(?:unittest|compileall)\b|"
    r"npm\s+(?:test|run\s+(?:test|build|lint|check))\b|"
    r"(?:pnpm|yarn)\s+(?:test|build|lint|check)\b|"
    r"cargo\s+(?:test|check|clippy|build)\b|"
    r"go\s+test\b|dotnet\s+(?:test|build)\b"
    r")",
    re.IGNORECASE,
)
_DANGEROUS_COMMAND_RE = re.compile(
    r"(?:\brm\b|\brmdir\b|\bdel\b|remove-item|format\b|diskpart\b|"
    r"git\s+(?:push|reset|clean|checkout)\b|curl\b|wget\b|invoke-webrequest|"
    r"pip\s+install\b|npm\s+install\b|pnpm\s+(?:add|install)\b|yarn\s+add\b)",
    re.IGNORECASE,
)


def classify_action(
    name: str,
    args: Mapping[str, Any],
    *,
    workspace: str | Path,
    sandboxed: bool,
) -> ActionPolicyDecision:
    """Return the smallest user-attention boundary that safely fits an action."""

    tool = str(name)
    if tool in _READ_TOOLS:
        return ActionPolicyDecision(ApprovalRequirement.AUTO, "read", "Read-only project inspection")
    if tool in _WRITE_TOOLS:
        return ActionPolicyDecision(
            ApprovalRequirement.AUTO,
            "workspace_write",
            "Edits stay inside the selected project and remain reviewable",
        )
    if tool in {"run_bash", "run_command"}:
        command = str(args.get("command") or "")
        if _DANGEROUS_COMMAND_RE.search(command):
            return ActionPolicyDecision(
                ApprovalRequirement.ONCE,
                "dangerous_command",
                "This command can delete data, install software, use the network, or rewrite history",
            )
        if _SAFE_COMMAND_RE.search(command):
            return ActionPolicyDecision(
                ApprovalRequirement.AUTO if sandboxed else ApprovalRequirement.SESSION,
                "project_checks",
                "Build and test commands are isolated" if sandboxed else "Allow project checks on this computer for this session",
            )
        return ActionPolicyDecision(
            ApprovalRequirement.ONCE,
            "shell_command",
            "This command runs directly on the computer",
        )
    if tool == "install_dependencies":
        return ActionPolicyDecision(
            ApprovalRequirement.ONCE,
            "dependencies",
            "Installing dependencies can contact the network and change the environment",
        )
    if tool in {"start_process", "preview_html"}:
        if tool == "preview_html" and not bool(args.get("open_browser", True)):
            return ActionPolicyDecision(ApprovalRequirement.AUTO, "preview", "Local preview inspection")
        return ActionPolicyDecision(
            ApprovalRequirement.SESSION,
            "project_preview",
            "Allow local project previews for this session",
        )
    if tool in {"stop_process", "stop_preview", "open_path"}:
        return ActionPolicyDecision(
            ApprovalRequirement.ONCE,
            "host_action",
            "This action changes or opens something outside the coding transcript",
        )
    return ActionPolicyDecision(
        ApprovalRequirement.ONCE,
        "unknown",
        "This action is not covered by the safe project policy",
    )


_PLAN_GATE_RE = re.compile(
    r"(?:delete|remove|migration|migrate|schema|dependency|install|network|deploy|"
    r"publish|upload|secret|credential|auth|security|permission|outside|external|"
    r"payment|billing|database|production)",
    re.IGNORECASE,
)


def plan_review_reasons(view: Any) -> tuple[str, ...]:
    """Return deterministic reasons why a plan must not auto-start."""

    reasons: list[str] = []
    text_parts = [str(getattr(view, "plan_summary", "")), str(getattr(view, "objective", ""))]
    for change in getattr(view, "expected_changes", ()) or ():
        if isinstance(change, Mapping):
            text_parts.extend((str(change.get("path", "")), str(change.get("intent", ""))))
    if _PLAN_GATE_RE.search(" ".join(text_parts)):
        reasons.append("The plan includes a high-impact or external change")
    if any(str(getattr(task, "risk", "")).lower() == "critical" for task in getattr(view, "tasks", ())):
        reasons.append("The plan contains a critical-risk task")
    return tuple(dict.fromkeys(reasons))


_CONSEQUENTIAL_QUESTION_RE = re.compile(
    r"(?:appearance|design|layout|user|workflow|behavior|scope|persist|history|"
    r"account|authentication|security|privacy|payment|finance|delete|platform|mobile|desktop|web)",
    re.IGNORECASE,
)


def should_surface_question(question: Mapping[str, Any]) -> bool:
    text = " ".join(
        str(question.get(key) or "") for key in ("header", "question", "reason")
    )
    return bool(_CONSEQUENTIAL_QUESTION_RE.search(text))


__all__ = [
    "ActionPolicyDecision",
    "ApprovalRequirement",
    "classify_action",
    "plan_review_reasons",
    "should_surface_question",
]
