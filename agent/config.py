"""Validated runtime configuration for bounded slices of an unbounded goal."""

from __future__ import annotations

import os
from dataclasses import dataclass, fields, replace
from enum import Enum
from typing import Any

from .intake import RunMode


class InteractionMode(str, Enum):
    """Public run policy; intake/planning are internal lifecycle phases."""

    NORMAL = "normal"
    ULTRA = "ultra"
    CHAT = "normal"
    PLAN = "normal"
    GOAL = "normal"

    @classmethod
    def parse(cls, value: str | "InteractionMode") -> "InteractionMode":
        if isinstance(value, cls):
            return value
        normalized = RunMode.parse(str(getattr(value, "value", value))).value
        try:
            return cls(normalized)
        except ValueError as exc:
            raise ValueError("mode must be 'normal' or 'ultra'") from exc


class ReasoningEffort(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"

    @classmethod
    def parse(cls, value: str | "ReasoningEffort") -> "ReasoningEffort":
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().lower().replace("-", "")
        aliases = {"minimal": "low", "default": "medium", "max": "xhigh", "extra_high": "xhigh"}
        normalized = aliases.get(normalized, normalized)
        try:
            return cls(normalized)
        except ValueError as exc:
            raise ValueError("reasoning effort must be low, medium, high, or xhigh") from exc


@dataclass
class SessionPreferences:
    """Mutable UI preferences that intentionally last only for this process."""

    mode: InteractionMode = InteractionMode.NORMAL

    @classmethod
    def from_env(cls, mode: str | None = None) -> "SessionPreferences":
        return cls(mode=InteractionMode.parse(mode or os.getenv("AGENT_MODE", "normal")))


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


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    normalized = raw.strip().casefold()
    if normalized in {"1", "true", "yes", "on", "required", "gpu"}:
        return True
    if normalized in {"0", "false", "no", "off", "optional", "cpu"}:
        return False
    raise ValueError(f"{name} must be a boolean")


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
    ultra_cloud_concurrency: int = 4
    ultra_max_depth: int = 8
    ultra_max_nodes: int = 1_000
    ultra_top_modules_min: int = 4
    ultra_top_modules_max: int = 12
    ultra_fix_attempts: int = 3
    prompt_trace_chars: int = 256_000
    role_memory_ttl_hours: int = 168
    require_local_gpu: bool = False
    repository_index_warmup_files: int = 200

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
            ultra_cloud_concurrency=_env_int("AGENT_ULTRA_CLOUD_CONCURRENCY", 4, 1, 8),
            ultra_max_depth=_env_int("AGENT_ULTRA_MAX_DEPTH", 8, 1, 12),
            ultra_max_nodes=_env_int("AGENT_ULTRA_MAX_NODES", 1_000, 10, 5_000),
            ultra_top_modules_min=_env_int("AGENT_ULTRA_MODULES_MIN", 4, 1, 32),
            ultra_top_modules_max=_env_int("AGENT_ULTRA_MODULES_MAX", 12, 1, 80),
            ultra_fix_attempts=_env_int("AGENT_ULTRA_FIX_ATTEMPTS", 3, 1, 20),
            prompt_trace_chars=_env_int("AGENT_PROMPT_TRACE_CHARS", 256_000, 16_000, 2_000_000),
            role_memory_ttl_hours=_env_int("AGENT_ROLE_MEMORY_TTL_HOURS", 168, 1, 8_760),
            require_local_gpu=_env_bool("AGENT_REQUIRE_LOCAL_GPU", False),
            repository_index_warmup_files=_env_int("AGENT_REPOSITORY_INDEX_WARMUP_FILES", 200, 0, 100_000),
        )
        if config.goal_retry_base_ms > config.goal_retry_max_ms:
            raise ValueError("AGENT_GOAL_RETRY_BASE_MS cannot exceed AGENT_GOAL_RETRY_MAX_MS")
        if config.ultra_top_modules_min > config.ultra_top_modules_max:
            raise ValueError("AGENT_ULTRA_MODULES_MIN cannot exceed AGENT_ULTRA_MODULES_MAX")
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
    "ultra_cloud_concurrency": (1, 8),
    "ultra_max_depth": (1, 12),
    "ultra_max_nodes": (10, 5_000),
    "ultra_top_modules_min": (1, 32),
    "ultra_top_modules_max": (1, 80),
    "ultra_fix_attempts": (1, 20),
    "prompt_trace_chars": (16_000, 2_000_000),
    "role_memory_ttl_hours": (1, 8_760),
    "repository_index_warmup_files": (0, 100_000),
}

RUNTIME_BOOL_SETTINGS: set[str] = {
    "require_local_gpu",
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
    "concurrency": "ultra_cloud_concurrency",
    "ultra_concurrency": "ultra_cloud_concurrency",
    "ultra_depth": "ultra_max_depth",
    "ultra_nodes": "ultra_max_nodes",
    "fix_attempts": "ultra_fix_attempts",
    "gpu": "require_local_gpu",
    "require_gpu": "require_local_gpu",
    "local_gpu": "require_local_gpu",
    "repo_index_warmup": "repository_index_warmup_files",
    "repository_index_warmup": "repository_index_warmup_files",
    "index_warmup": "repository_index_warmup_files",
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
    if normalized in RUNTIME_BOOL_SETTINGS:
        return replace(config, **{normalized: _parse_bool_value(normalized, raw_value)})
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
    if updated.ultra_top_modules_min > updated.ultra_top_modules_max:
        raise ValueError("ultra_top_modules_min cannot exceed ultra_top_modules_max")
    return updated


def _parse_bool_value(name: str, raw_value: str) -> bool:
    normalized = str(raw_value).strip().casefold()
    if normalized in {"1", "true", "yes", "on", "required", "gpu"}:
        return True
    if normalized in {"0", "false", "no", "off", "optional", "cpu"}:
        return False
    raise ValueError(f"{name} must be a boolean")
