"""Validated HTTP payloads for the local artifact workspaces."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RetryPolicyPayload(BaseModel):
    max_retries: int = Field(default=2, ge=0, le=20)
    backoff_seconds: float = Field(default=0, ge=0, le=3600)


class TaskPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: str = Field(min_length=1, max_length=24, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,23}$")
    title: str = Field(min_length=1, max_length=180)
    description: str = Field(default="", max_length=4000)
    parent_id: str | None = None
    dependencies: list[str] = Field(default_factory=list, max_length=80)
    agent_role: str = Field(default="coder", min_length=1, max_length=120)
    inputs: list[str] = Field(default_factory=list, max_length=80)
    outputs: list[str] = Field(default_factory=list, max_length=80)
    expected_files: list[str] = Field(default_factory=list, max_length=80)
    acceptance_criteria: list[str] = Field(default_factory=list, min_length=1, max_length=20)
    tests: list[str] = Field(default_factory=list, min_length=1, max_length=20)
    risk_level: Literal["low", "medium", "high", "critical"] = "medium"
    required_tools: list[str] = Field(default_factory=list, max_length=40)
    memory_dependencies: list[str] = Field(default_factory=list, max_length=40)
    retry_policy: RetryPolicyPayload = Field(default_factory=RetryPolicyPayload)
    approval_gate: bool = False
    constraints: list[str] = Field(default_factory=list, max_length=40)
    parallel: bool = False
    comments: list[str] = Field(default_factory=list, max_length=40)

    @field_validator(
        "dependencies",
        "inputs",
        "outputs",
        "expected_files",
        "acceptance_criteria",
        "tests",
        "required_tools",
        "memory_dependencies",
        "constraints",
        "comments",
    )
    @classmethod
    def bounded_items(cls, value: list[str]) -> list[str]:
        cleaned = [str(item).strip() for item in value if str(item).strip()]
        if any(len(item) > 1000 or "\x00" in item for item in cleaned):
            raise ValueError("list values must be non-empty, NUL-free, and at most 1,000 characters")
        return list(dict.fromkeys(cleaned))


class PlanPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    base_revision: int = Field(ge=1)
    summary: str = Field(min_length=1, max_length=20_000)
    tasks: list[TaskPayload] = Field(min_length=1, max_length=80)
    global_constraints: list[str] = Field(default_factory=list, max_length=80)
    protected_paths: list[str] = Field(default_factory=list, max_length=80)
    change_note: str = Field(default="Edited in Plan Studio", max_length=4000)

    @field_validator("global_constraints", "protected_paths")
    @classmethod
    def bounded_plan_items(cls, value: list[str]) -> list[str]:
        cleaned = [str(item).strip() for item in value if str(item).strip()]
        if any(len(item) > 1000 or "\x00" in item for item in cleaned):
            raise ValueError("plan values must be NUL-free and at most 1,000 characters")
        return list(dict.fromkeys(cleaned))


class PlanApprovalPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    revision: int = Field(ge=1)


class QueuePromptPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    text: str = Field(min_length=1, max_length=20_000)
    mode: Literal["plan", "normal", "ultra"] | None = None


class QueueReorderPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    ordered_ids: list[str] = Field(min_length=1, max_length=10)

    @field_validator("ordered_ids")
    @classmethod
    def unique_ids(cls, value: list[str]) -> list[str]:
        cleaned = [str(item).strip() for item in value if str(item).strip()]
        if len(cleaned) != len(value) or len(set(cleaned)) != len(cleaned):
            raise ValueError("ordered queue ids must be non-empty and unique")
        return cleaned


class ReviewDecisionPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    target_type: Literal["file", "hunk"]
    file_path: str = Field(min_length=1, max_length=1000)
    hunk_id: str | None = Field(default=None, max_length=200)
    decision: Literal["accepted", "rejected", "changes_requested"]
    reason: str = Field(default="", max_length=4000)


class ReviewCommentPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    file_path: str = Field(min_length=1, max_length=1000)
    hunk_id: str | None = Field(default=None, max_length=200)
    line: int | None = Field(default=None, ge=1)
    body: str = Field(min_length=1, max_length=4000)


class ReviewSubmissionPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    checkpoint_id: str = Field(min_length=1, max_length=200)
    decisions: list[ReviewDecisionPayload] = Field(min_length=1, max_length=2000)
    comments: list[ReviewCommentPayload] = Field(default_factory=list, max_length=2000)
    summary: str = Field(default="", max_length=4000)


class ExplanationRequestPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    agent_id: str = Field(min_length=1, max_length=200)
    question: str = Field(default="Explain your current work and any blockers.", min_length=1, max_length=2000)


def model_errors(exc: Exception) -> dict[str, Any]:
    errors = getattr(exc, "errors", None)
    return {"error": str(exc), "details": errors() if callable(errors) else []}
