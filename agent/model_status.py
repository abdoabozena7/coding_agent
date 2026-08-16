"""Truthful model lifecycle and recovery diagnostics.

Model identity, provider availability, and structured-contract compliance are
different facts.  This module keeps those facts separate so every frontend can
describe the same failure without inferring it from presentation state.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from .local_provider import ProviderFailureKind, ProviderRequestError
from .safety import redact_text


class ModelFailureCategory(str, Enum):
    RUNNER_UNREACHABLE = "runner_unreachable"
    MODEL_NOT_INSTALLED = "model_not_installed"
    MODEL_LOAD_FAILED = "model_load_failed"
    AUTHENTICATION_FAILED = "authentication_failed"
    QUOTA_EXCEEDED = "quota_exceeded"
    RATE_LIMITED = "rate_limited"
    TRANSPORT_FAILED = "transport_failed"
    EMPTY_RESPONSE = "empty_response"
    TOOL_CONTRACT_FAILED = "tool_contract_failed"
    TYPED_RETURN_FAILED = "typed_return_failed"
    CAPABILITY_MISMATCH = "capability_mismatch"
    INVALID_REQUEST = "invalid_request"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    INTERNAL_RUNTIME_ERROR = "internal_runtime_error"


_AVAILABILITY_FAILURES = frozenset(
    {
        ModelFailureCategory.RUNNER_UNREACHABLE,
        ModelFailureCategory.MODEL_NOT_INSTALLED,
        ModelFailureCategory.MODEL_LOAD_FAILED,
        ModelFailureCategory.AUTHENTICATION_FAILED,
        ModelFailureCategory.QUOTA_EXCEEDED,
        ModelFailureCategory.RATE_LIMITED,
        ModelFailureCategory.TRANSPORT_FAILED,
        ModelFailureCategory.PROVIDER_UNAVAILABLE,
    }
)


@dataclass(frozen=True, slots=True)
class RecoveryDiagnosis:
    category: ModelFailureCategory
    message: str
    stage: str = ""

    @property
    def provider_boundary(self) -> bool:
        """Whether availability recovery/fallback is appropriate."""

        return self.category in _AVAILABILITY_FAILURES

    def title(self, execution_class: str) -> str:
        location = "Local model" if str(execution_class).casefold() == "local" else "Cloud model"
        labels = {
            ModelFailureCategory.RUNNER_UNREACHABLE: f"{location} runner is unreachable",
            ModelFailureCategory.MODEL_NOT_INSTALLED: "Selected model is not installed",
            ModelFailureCategory.MODEL_LOAD_FAILED: f"{location} could not load",
            ModelFailureCategory.AUTHENTICATION_FAILED: "Provider authentication failed",
            ModelFailureCategory.QUOTA_EXCEEDED: "Provider usage limit reached",
            ModelFailureCategory.RATE_LIMITED: "Provider rate limit reached",
            ModelFailureCategory.TRANSPORT_FAILED: "Provider connection failed",
            ModelFailureCategory.EMPTY_RESPONSE: "Model returned an empty response",
            ModelFailureCategory.TOOL_CONTRACT_FAILED: "Model tool contract needs attention",
            ModelFailureCategory.TYPED_RETURN_FAILED: "Model response contract needs attention",
            ModelFailureCategory.CAPABILITY_MISMATCH: "Model capability is incompatible",
            ModelFailureCategory.INVALID_REQUEST: "Provider rejected the request",
            ModelFailureCategory.PROVIDER_UNAVAILABLE: f"{location} request failed",
            ModelFailureCategory.INTERNAL_RUNTIME_ERROR: "Workflow step did not finish",
        }
        return labels[self.category]

    def pause_reason(self, execution_class: str) -> str:
        stage = f" at the {self.stage.replace('_', ' ')} stage" if self.stage else ""
        return f"{self.title(execution_class).casefold()}{stage}: {self.message}"


def _provider_request_failure(error: BaseException) -> ProviderRequestError | None:
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, ProviderRequestError):
            return current
        cause = getattr(current, "__cause__", None)
        current = cause if isinstance(cause, BaseException) else getattr(
            current, "__context__", None
        )
    return None


def _category_from_provider(error: ProviderRequestError) -> ModelFailureCategory:
    diagnostic = error.diagnostic
    kind = diagnostic.kind
    message = str(diagnostic.provider_message or error).casefold()
    if kind in {ProviderFailureKind.DNS_OR_SOCKET, ProviderFailureKind.CONNECTION_REFUSED}:
        return ModelFailureCategory.RUNNER_UNREACHABLE
    if kind is ProviderFailureKind.TIMEOUT:
        return ModelFailureCategory.TRANSPORT_FAILED
    if kind is ProviderFailureKind.MODEL_NOT_INSTALLED:
        return ModelFailureCategory.MODEL_NOT_INSTALLED
    if kind is ProviderFailureKind.MODEL_LOAD_FAILED:
        return ModelFailureCategory.MODEL_LOAD_FAILED
    if kind is ProviderFailureKind.INVALID_TYPED_OUTPUT:
        return ModelFailureCategory.TYPED_RETURN_FAILED
    if kind is ProviderFailureKind.MALFORMED_STREAM:
        return ModelFailureCategory.EMPTY_RESPONSE if "empty" in message else ModelFailureCategory.TRANSPORT_FAILED
    if kind is ProviderFailureKind.EMPTY_RESPONSE:
        return ModelFailureCategory.EMPTY_RESPONSE
    if kind in {
        ProviderFailureKind.UNSUPPORTED_TOOLS,
        ProviderFailureKind.UNSUPPORTED_STRUCTURED_OUTPUT,
        ProviderFailureKind.UNSUPPORTED_PARAMETER,
    }:
        return ModelFailureCategory.CAPABILITY_MISMATCH
    if kind in {ProviderFailureKind.INVALID_PAYLOAD, ProviderFailureKind.CONTEXT_LIMIT}:
        return ModelFailureCategory.INVALID_REQUEST
    status = int(diagnostic.status_code or 0)
    if status in {401, 403} or any(token in message for token in ("unauthorized", "forbidden", "authentication")):
        return ModelFailureCategory.AUTHENTICATION_FAILED
    if status == 429:
        if any(token in message for token in ("quota", "usage limit", "session usage limit", "exhausted")):
            return ModelFailureCategory.QUOTA_EXCEEDED
        return ModelFailureCategory.RATE_LIMITED
    if kind is ProviderFailureKind.HTTP_5XX:
        return ModelFailureCategory.PROVIDER_UNAVAILABLE
    if kind is ProviderFailureKind.ENDPOINT_NOT_FOUND:
        return ModelFailureCategory.INVALID_REQUEST
    return ModelFailureCategory.PROVIDER_UNAVAILABLE


def classify_model_failure(
    error: BaseException,
    pending_semantic: Mapping[str, Any] | None = None,
) -> RecoveryDiagnosis:
    """Classify the root failure; pending state supplies stage, never causality."""

    pending = dict(pending_semantic or {})
    validation = pending.get("last_validation_error")
    validation = dict(validation) if isinstance(validation, Mapping) else {}
    stage = str(validation.get("stage") or pending.get("stage") or "").strip()
    error_message = redact_text(str(error), 1_000).strip()
    error_text = error_message.casefold()

    typed_provider = _provider_request_failure(error)
    if typed_provider is not None:
        message = redact_text(
            typed_provider.diagnostic.provider_message or str(typed_provider), 1_000
        ).strip()
        return RecoveryDiagnosis(_category_from_provider(typed_provider), message, stage)

    # Strong contract signatures from the current exception win over possibly
    # stale pending semantic metadata left behind after routing completed.
    if any(
        token in error_text
        for token in (
            "architecturespecv1",
            "typed return",
            "typed-return",
            "requires a summary and components",
            "invalid typed",
        )
    ):
        contract_stage = "architecture" if "architecture" in error_text else stage
        return RecoveryDiagnosis(
            ModelFailureCategory.TYPED_RETURN_FAILED,
            error_message,
            contract_stage or "typed_return",
        )
    if any(
        token in error_text
        for token in (
            "must be called exactly once",
            "only allowed call is",
            "tool contract",
            "submit_semantic_route",
            "submit_goal_intake",
        )
    ):
        return RecoveryDiagnosis(ModelFailureCategory.TOOL_CONTRACT_FAILED, error_message, stage)
    if any(token in error_text for token in ("empty response", "no usable", "unused token")):
        return RecoveryDiagnosis(ModelFailureCategory.EMPTY_RESPONSE, error_message, stage)

    category = str(validation.get("category") or "").casefold()
    pending_message = redact_text(
        str(validation.get("message") or pending.get("last_error") or error_message), 1_000
    ).strip()
    if category in {"schema", "tool", "tool_contract"}:
        return RecoveryDiagnosis(ModelFailureCategory.TOOL_CONTRACT_FAILED, pending_message, stage)
    if category in {"semantic", "contract", "typed_return", "validation"}:
        return RecoveryDiagnosis(ModelFailureCategory.TYPED_RETURN_FAILED, pending_message, stage)
    if category == "provider" or isinstance(error, RuntimeError) and "provider unavailable" in error_text:
        return RecoveryDiagnosis(ModelFailureCategory.PROVIDER_UNAVAILABLE, pending_message, stage)
    return RecoveryDiagnosis(ModelFailureCategory.INTERNAL_RUNTIME_ERROR, error_message, stage)


def model_status_for(
    provider: Any,
    descriptor: Any | None,
    prior: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a secret-free lifecycle snapshot for the selected model."""

    provider_name = str(
        getattr(descriptor, "provider", "")
        or provider.__class__.__name__.removesuffix("Provider").casefold()
        or "provider"
    )
    model = str(getattr(descriptor, "model", "") or getattr(provider, "model", "unknown"))
    execution = str(
        getattr(getattr(descriptor, "execution_class", ""), "value", "")
        or getattr(descriptor, "execution_class", "")
        or ("cloud" if model.casefold().endswith((":cloud", "-cloud")) else "local")
    )
    descriptor_id = str(getattr(descriptor, "id", "") or f"{provider_name}:{model}")
    source = str(getattr(descriptor, "source", "") or "configured")
    metadata = getattr(descriptor, "metadata", {})
    metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
    profile = getattr(provider, "capability_profile", None)
    health = str(getattr(profile, "health_status", "") or "").casefold()
    same_model = str((prior or {}).get("descriptor_id") or "") == descriptor_id
    carried = dict(prior or {}) if same_model else {}
    discovered = provider_name != "ollama" or source == "ollama" or bool(metadata.get("ollama_version"))
    probed = health == "reachable" or (provider_name == "ollama" and bool(metadata.get("ollama_version")))
    return {
        "descriptor_id": descriptor_id,
        "provider": provider_name,
        "model": model,
        "execution_class": execution,
        "selection_status": "selected",
        "inventory_status": "discovered" if discovered else carried.get("inventory_status", "unknown"),
        "probe_status": "passed" if probed else carried.get("probe_status", "not_run"),
        "response_status": carried.get("response_status", "not_run"),
        "contract_status": carried.get("contract_status", "not_run"),
        "contract_stage": carried.get("contract_stage", ""),
        "failure_kind": carried.get("failure_kind", ""),
        "last_error": carried.get("last_error", ""),
    }


def preflight_model_selection(provider: Any, descriptor: Any) -> None:
    """Verify exact Ollama inventory/capabilities before durable replacement."""

    if str(getattr(descriptor, "provider", "")).casefold() != "ollama":
        return
    ensure = getattr(provider, "_ensure_capabilities", None)
    if not callable(ensure):
        # Injected/offline test adapters intentionally omit network probes.
        # The production Ollama adapter always exposes this boundary.
        return
    ensure()


__all__ = [
    "ModelFailureCategory",
    "RecoveryDiagnosis",
    "classify_model_failure",
    "model_status_for",
    "preflight_model_selection",
]
