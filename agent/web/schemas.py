"""Versioned HTTP and WebSocket contracts for the local web client."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProjectCreateV1(StrictModel):
    path: str = Field(min_length=1, max_length=4096)


class ThreadCreateV1(StrictModel):
    title: str = Field(default="New task", max_length=120)


class ThreadPatchV1(StrictModel):
    title: str | None = Field(default=None, max_length=120)
    pinned: bool | None = None
    archived: bool | None = None


class WorkspaceViewPatchV1(StrictModel):
    view_mode: Literal["transcript", "visualize"]
    expected_revision: int | None = Field(default=None, ge=1)


class DraftPatchV1(StrictModel):
    text: str = Field(default="", max_length=20_000)
    expected_revision: int | None = Field(default=None, ge=0)


class TerminalCommandV1(StrictModel):
    command: str = Field(min_length=1, max_length=8_000)


class WorkflowModePatchV1(StrictModel):
    mode: Literal["plan", "normal", "ultra"]
    expected_revision: int | None = Field(default=None, ge=1)


class MessageDirectionPatchV1(StrictModel):
    direction: Literal["auto", "rtl"]


class PlanDecisionV1(StrictModel):
    action: Literal["implement", "keep_planning"]
    revision: int = Field(ge=1)
    fingerprint: str = Field(min_length=8, max_length=256)
    feedback: str = Field(default="", max_length=20_000)
    client_request_id: str = Field(min_length=8, max_length=200)


class AccessChangeV1(StrictModel):
    policy: Literal["default", "bounded", "full", "host"]
    confirmation_token: str = Field(default="", max_length=256)
    expected_revision: int | None = Field(default=None, ge=1)


class ModelRolePatchV1(StrictModel):
    descriptor_id: str = Field(default="", max_length=500)


class ModelValidationV1(StrictModel):
    descriptor_id: str = Field(min_length=1, max_length=500)


class InferenceProfileV1(StrictModel):
    device: Literal["auto", "cpu", "gpu"] = "auto"
    context_window: int = Field(default=16_384, ge=2_048, le=131_072)
    max_output_tokens: int = Field(default=4_096, ge=128, le=65_536)
    gpu_layers: int = Field(default=-1, ge=-1, le=999)
    cpu_threads: int = Field(default=4, ge=1, le=256)
    temperature: float = Field(default=0.2, ge=0, le=2)
    top_p: float = Field(default=0.9, ge=0, le=1)
    top_k: int = Field(default=40, ge=0, le=1_000)
    performance: Literal["eco", "balanced", "performance"] = "balanced"
    estimated_minutes_per_step: int = Field(default=30, ge=1, le=720)
    planning_steps: int = Field(default=16, ge=2, le=100)
    work_quantum_steps: int = Field(default=24, ge=1, le=500)
    review_steps: int = Field(default=12, ge=2, le=100)
    max_provider_retries: int = Field(default=3, ge=0, le=10)
    ultra_cloud_concurrency: int = Field(default=4, ge=1, le=8)
    ultra_max_depth: int = Field(default=8, ge=1, le=12)


class ThreadInputV1(StrictModel):
    kind: Literal["message", "command"] = "message"
    delivery: Literal["queue", "guidance"] = "queue"
    text: str = Field(min_length=1, max_length=20_000)
    client_request_id: str = Field(min_length=8, max_length=200)


class AttentionResolutionV1(StrictModel):
    option_key: str = Field(min_length=1, max_length=120)
    text: str = Field(default="", max_length=20_000)


class SettingsPatchV1(StrictModel):
    theme: Literal["dark", "light", "system"] | None = None
    locale: Literal["auto", "en", "ar"] | None = None
    experience: Literal["simple", "advanced"] | None = None
    mode: Literal["plan", "normal", "ultra"] | None = None
    access: Literal["normal", "bounded", "full"] | None = None
    reduced_motion: bool | None = None


class WebEventV1(StrictModel):
    version: Literal[1] = 1
    sequence: int
    project_id: str | None = None
    thread_id: str | None = None
    type: str
    payload: dict[str, Any] = Field(default_factory=dict)
    emitted_at: str = Field(default_factory=utc_iso)


class ErrorV1(StrictModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "AttentionResolutionV1",
    "AccessChangeV1",
    "DraftPatchV1",
    "ErrorV1",
    "InferenceProfileV1",
    "MessageDirectionPatchV1",
    "ModelRolePatchV1",
    "ModelValidationV1",
    "PlanDecisionV1",
    "ProjectCreateV1",
    "SettingsPatchV1",
    "ThreadCreateV1",
    "ThreadInputV1",
    "ThreadPatchV1",
    "TerminalCommandV1",
    "WorkspaceViewPatchV1",
    "WorkflowModePatchV1",
    "WebEventV1",
]
