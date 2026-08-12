"""Validated HTTP payloads for the local artifact workspaces."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RetryPolicyPayload(BaseModel):
    max_retries: int = Field(default=2, ge=0, le=20)
    backoff_seconds: float = Field(default=0, ge=0, le=3600)


class TraceRevealPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    trace_id: str = Field(min_length=1, max_length=200)
    goal_id: str | None = Field(default=None, max_length=200)
    run_id: str | None = Field(default=None, max_length=200)


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
    requirement_refs: list[str] = Field(default_factory=list, max_length=40)
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
        "requirement_refs",
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
    plan_fingerprint: str = Field(default="", max_length=128)
    team_fingerprint: str = Field(default="", max_length=128)


class PlanDocumentPayload(BaseModel):
    """The calm editor's single-document plan handoff."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    base_revision: int = Field(ge=1)
    document: str = Field(min_length=1, max_length=40_000)


class ToolApprovalPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    action_fingerprint: str = Field(min_length=16, max_length=128)
    decision: Literal["allow", "approve", "allow_once", "allow_session", "deny", "reject"]


class PlanRequestPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    request: str = Field(min_length=1, max_length=20_000)


class WorkspaceContextPayload(BaseModel):
    """Stable navigation/attention contract for the unified local workspace."""

    model_config = ConfigDict(extra="forbid")

    session_id: str
    session_short: str
    requested_view: Literal["plan", "review", "agents", "execution", "history", "thread"]
    required_view: Literal["plan", "review", "agents", "execution", "history", "thread"] | None = None
    current_view: Literal["plan", "review", "agents", "execution", "history", "thread"]
    checkpoint_id: str | None = None
    goal: dict[str, Any] | None = None
    mode: Literal["execution", "ultra-plan"]
    attention: dict[str, Any]
    navigation: dict[str, dict[str, Any]]
    capabilities: dict[str, bool]
    queue: dict[str, Any]
    updated_at: str
    # Additive runtime truth used by the TUI and browser workspace.  Defaults
    # keep older clients and saved sessions compatible.
    runtime: dict[str, Any] = Field(default_factory=dict)
    route: str = "pending"
    execution_strategy: str = "pending"
    phase: str = "ready"
    waiting_on: str = ""
    resume_action: str = ""
    tool_approval: dict[str, Any] | None = None
    control_surface: Literal["web", "terminal_fallback"] = "web"
    required_action: dict[str, Any] | None = None
    pending_question: dict[str, Any] | None = None
    workflow_identity: dict[str, Any] = Field(default_factory=dict)
    history_cursor: int = 0
    local_continuation: dict[str, Any] | None = None
    provider_recovery: dict[str, Any] | None = None
    # Additive left-rail projection.  Keeping it in the workspace snapshot
    # avoids a second polling race while preserving the standalone index API.
    project_sessions: dict[str, Any] = Field(default_factory=dict)
    current_plan: dict[str, Any] = Field(default_factory=dict)
    todo: dict[str, Any] = Field(default_factory=dict)


class WorkspaceActionRequest(BaseModel):
    """One idempotent action submitted by the primary workspace surface."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    action: Literal[
        "approve_plan", "allow_tool", "allow_tool_session", "deny_tool", "retry", "resume",
        "answer", "pause", "stop", "switch_model",
        "continue_local_model", "reconfigure_protection",
        "reconfigure_mode", "reconfigure_concurrency",
    ]
    target_id: str | None = Field(default=None, max_length=200)
    action_fingerprint: str = Field(default="", max_length=256)
    expected_sequence: int | None = Field(default=None, ge=0)
    source: Literal["web", "terminal_fallback"] = "web"
    value: str | None = Field(default=None, max_length=20_000)


class WorkspaceActionReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    accepted: bool
    action: str
    source: str
    duplicate: bool = False
    next_view: str | None = None
    next_phase: str | None = None
    event_sequence: int | None = None
    message: str


class ThreadItemPayload(BaseModel):
    """Stable append-only item returned by the unified thread projection."""

    model_config = ConfigDict(extra="allow")

    item_id: str
    type: Literal[
        "user_message", "assistant_message", "workflow_status", "plan",
        "tool_run", "approval", "change_set", "review", "recovery", "completion",
    ]
    kind: str
    sequence: int = Field(ge=0)
    content_revision: int = Field(ge=0)
    created_at: str
    payload: dict[str, Any] = Field(default_factory=dict)


class ThreadSnapshotPayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    session_id: str
    items: list[ThreadItemPayload] = Field(default_factory=list)
    after_sequence: int = Field(default=0, ge=0)
    next_sequence: int = Field(default=0, ge=0)
    has_more: bool = False
    activity_sequence: int = Field(default=0, ge=0)
    content_revision: int = Field(default=0, ge=0)
    connection: str = "connected"


class HistoryEventPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sequence: int
    event_id: str
    event_type: str
    phase: str = ""
    actor: str = "harness"
    summary: str
    why: str = ""
    evidence: list[Any] = Field(default_factory=list)
    workspace_mutated: bool = False
    retry_count: int = 0
    next_state: str = ""
    goal_id: str | None = None
    entity_id: str | None = None
    created_at: str


class HistorySnapshotPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    goal_id: str | None = None
    items: list[HistoryEventPayload] = Field(default_factory=list)
    next_cursor: int | None = None
    has_more: bool = False
    filters: dict[str, Any] = Field(default_factory=dict)
    connection: str = "connected"
    goals: list[dict[str, Any]] = Field(default_factory=list)


class QueuePromptPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    text: str = Field(min_length=1, max_length=20_000)
    mode: Literal["execution", "ultra-plan", "working", "plan", "normal", "ultra"] | None = None


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
