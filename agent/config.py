"""Validated runtime configuration for bounded slices of an unbounded goal."""

from __future__ import annotations

import os
from dataclasses import dataclass, fields, replace
from enum import Enum
from typing import Any


class InteractionMode(str, Enum):
    """Session input policy; separate from durable goal/task lifecycle state."""

    PLAN = "plan"
    GOAL = "goal"

    @classmethod
    def parse(cls, value: str | "InteractionMode") -> "InteractionMode":
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().lower()
        aliases = {
            "manual": cls.PLAN.value,
            "default": cls.PLAN.value,
            "auto": cls.GOAL.value,
            "agent": cls.GOAL.value,
        }
        normalized = aliases.get(normalized, normalized)
        try:
            return cls(normalized)
        except ValueError as exc:
            raise ValueError("mode must be 'plan' or 'goal'") from exc


@dataclass
class SessionPreferences:
    """Mutable UI preferences that intentionally last only for this process."""

    mode: InteractionMode = InteractionMode.PLAN

    @classmethod
    def from_env(cls, mode: str | None = None) -> "SessionPreferences":
        return cls(mode=InteractionMode.parse(mode or os.getenv("AGENT_MODE", "plan")))


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


@dataclass(frozen=True)
class RuntimeConfig:
    """Safety limits are per work slice, never a deadline for the durable goal."""

    planning_steps: int = 16
    work_quantum_steps: int = 24
    review_steps: int = 12
    subagent_steps: int = 16
    max_delegation_depth: int = 4
    max_delegations_per_slice: int = 12
    max_provider_retries: int = 3
    repeated_action_limit: int = 2
    no_action_limit: int = 3
    stalled_slice_limit: int = 3
    conversation_chars: int = 120_000
    retry_base_ms: int = 250
    goal_retry_base_ms: int = 1_000
    goal_retry_max_ms: int = 30_000

    @classmethod
    def from_env(cls) -> "RuntimeConfig":
        config = cls(
            planning_steps=_env_int("AGENT_PLANNING_STEPS", 16, 2, 100),
            work_quantum_steps=_env_int("AGENT_WORK_QUANTUM", 24, 1, 500),
            review_steps=_env_int("AGENT_REVIEW_STEPS", 12, 2, 100),
            subagent_steps=_env_int("AGENT_SUBAGENT_STEPS", 16, 1, 200),
            max_delegation_depth=_env_int("AGENT_MAX_DELEGATION_DEPTH", 4, 0, 12),
            max_delegations_per_slice=_env_int("AGENT_MAX_DELEGATIONS_PER_SLICE", 12, 0, 100),
            max_provider_retries=_env_int("AGENT_PROVIDER_RETRIES", 3, 0, 10),
            repeated_action_limit=_env_int("AGENT_REPEAT_LIMIT", 2, 1, 10),
            no_action_limit=_env_int("AGENT_NO_ACTION_LIMIT", 3, 1, 20),
            stalled_slice_limit=_env_int("AGENT_STALLED_SLICE_LIMIT", 3, 1, 20),
            conversation_chars=_env_int("AGENT_CONTEXT_CHARS", 120_000, 20_000, 2_000_000),
            retry_base_ms=_env_int("AGENT_RETRY_BASE_MS", 250, 0, 30_000),
            goal_retry_base_ms=_env_int("AGENT_GOAL_RETRY_BASE_MS", 1_000, 0, 60_000),
            goal_retry_max_ms=_env_int("AGENT_GOAL_RETRY_MAX_MS", 30_000, 0, 60_000),
        )
        if config.goal_retry_base_ms > config.goal_retry_max_ms:
            raise ValueError("AGENT_GOAL_RETRY_BASE_MS cannot exceed AGENT_GOAL_RETRY_MAX_MS")
        return config


RUNTIME_SETTING_BOUNDS: dict[str, tuple[int, int]] = {
    "planning_steps": (2, 100),
    "work_quantum_steps": (1, 500),
    "review_steps": (2, 100),
    "subagent_steps": (1, 200),
    "max_delegation_depth": (0, 12),
    "max_delegations_per_slice": (0, 100),
    "max_provider_retries": (0, 10),
    "repeated_action_limit": (1, 10),
    "no_action_limit": (1, 20),
    "stalled_slice_limit": (1, 20),
    "conversation_chars": (20_000, 2_000_000),
    "retry_base_ms": (0, 30_000),
    "goal_retry_base_ms": (0, 60_000),
    "goal_retry_max_ms": (0, 60_000),
}

RUNTIME_SETTING_ALIASES: dict[str, str] = {
    "work_quantum": "work_quantum_steps",
    "planning": "planning_steps",
    "review": "review_steps",
    "subagent": "subagent_steps",
    "delegation_depth": "max_delegation_depth",
    "delegations": "max_delegations_per_slice",
    "provider_retries": "max_provider_retries",
    "repeat_limit": "repeated_action_limit",
    "context_chars": "conversation_chars",
}


def runtime_setting_names() -> tuple[str, ...]:
    return tuple(field.name for field in fields(RuntimeConfig))


def normalize_runtime_setting_name(key: str) -> str:
    normalized = key.strip().lower().replace("-", "_")
    return RUNTIME_SETTING_ALIASES.get(normalized, normalized)


def runtime_config_values(config: RuntimeConfig) -> dict[str, Any]:
    return {field.name: getattr(config, field.name) for field in fields(config)}


def update_runtime_config(config: RuntimeConfig, key: str, raw_value: str) -> RuntimeConfig:
    """Return a validated session config with one integer setting replaced."""

    normalized = normalize_runtime_setting_name(key)
    if normalized not in RUNTIME_SETTING_BOUNDS:
        available = ", ".join(runtime_setting_names())
        raise ValueError(f"unknown setting {key!r}; available runtime settings: {available}")
    try:
        value = int(raw_value.strip())
    except (AttributeError, ValueError) as exc:
        raise ValueError(f"{normalized} must be an integer") from exc
    minimum, maximum = RUNTIME_SETTING_BOUNDS[normalized]
    if not minimum <= value <= maximum:
        raise ValueError(f"{normalized} must be between {minimum} and {maximum}")
    updated = replace(config, **{normalized: value})
    if updated.goal_retry_base_ms > updated.goal_retry_max_ms:
        raise ValueError("goal_retry_base_ms cannot exceed goal_retry_max_ms")
    return updated
