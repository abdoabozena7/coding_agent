"""Capability-aware, local-first execution policy.

Semantic demand is authored by the selected model.  The harness only compares
that structured demand with a secret-free metadata envelope and chooses a
durable execution strategy.  No prompt keywords or model-name guesses belong
in this module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
import re
from typing import Any, Mapping, Sequence


class InteractionModeV2(str, Enum):
    WORKING = "working"
    PLAN = "plan"

    @classmethod
    def parse(cls, value: Any) -> "InteractionModeV2":
        normalized = str(getattr(value, "value", value) or "working").strip().casefold()
        normalized = {
            "normal": "working",
            "ultra": "working",
            "chat": "working",
            "goal": "working",
            "default": "working",
        }.get(normalized, normalized)
        try:
            return cls(normalized)
        except ValueError as exc:
            raise ValueError("interaction mode must be 'working' or 'plan'") from exc


class ExecutionStrategyV1(str, Enum):
    STAGED = "staged"
    RECURSIVE = "recursive"

    @classmethod
    def parse(cls, value: Any) -> "ExecutionStrategyV1":
        normalized = str(getattr(value, "value", value) or "staged").strip().casefold()
        normalized = {
            "normal": "staged",
            "adaptive": "staged",
            "direct": "staged",
            "ultra": "recursive",
            "deep": "recursive",
        }.get(normalized, normalized)
        try:
            return cls(normalized)
        except ValueError as exc:
            raise ValueError("execution strategy must be 'staged' or 'recursive'") from exc


class CapabilityBand(str, Enum):
    MINIMAL = "minimal"
    LIMITED = "limited"
    STANDARD = "standard"
    HIGH = "high"

    @property
    def level(self) -> int:
        return {
            CapabilityBand.MINIMAL: 1,
            CapabilityBand.LIMITED: 2,
            CapabilityBand.STANDARD: 3,
            CapabilityBand.HIGH: 4,
        }[self]

    @property
    def max_cohesive_components(self) -> int:
        return (1, 2, 4, 8)[self.level - 1]


@dataclass(frozen=True, slots=True)
class LocalAdaptationPolicy:
    """Model-packet adaptation independent from the selected workflow mode.

    The user chooses Normal/Plan/Ultra; this policy only controls how much
    context and how many cohesive components one model request may carry.
    Weak local models therefore receive narrower packets without silently
    changing the workflow into a recursive/Ultra run.
    """

    execution_class: str
    packet_size: int
    context_budget_tokens: int
    max_repairs: int
    abstraction_level: str
    quality_gates_unchanged: bool = True
    version: int = 1

    @classmethod
    def from_envelope(cls, envelope: "ModelCapabilityEnvelopeV1") -> "LocalAdaptationPolicy":
        local = str(envelope.execution_class or "local").casefold() == "local"
        packet_size = max(1, min(int(envelope.max_cohesive_components or 1), 4))
        if local:
            # Keep the first local packet intentionally narrow.  The harness
            # can widen later after a successful checkpoint, never by changing
            # workflow_mode or execution_strategy.
            packet_size = min(packet_size, 2)
        context = int(envelope.context_window_tokens or (24_000 if local else 64_000))
        context = max(8_000, min(context, 64_000 if not local else 32_000))
        repairs = 1 if local else 2
        level = (
            "atomic" if packet_size == 1
            else "narrow" if packet_size == 2
            else "bounded"
        )
        return cls(
            execution_class="local" if local else "cloud",
            packet_size=packet_size,
            context_budget_tokens=context,
            max_repairs=repairs,
            abstraction_level=level,
        )

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(
            json.dumps(
                asdict(self), ensure_ascii=False, sort_keys=True
            ).encode()
        ).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["fingerprint"] = self.fingerprint
        return value


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        parsed = float(value)
        if parsed <= 0:
            return None
        return parsed / 1e9 if parsed >= 1e6 else parsed
    text = str(value).strip().casefold().replace(",", "")
    if not text:
        return None
    # For MoE labels such as 8x7B, the smallest confirmed parameter figure is
    # the conservative effective value unless active_parameter_size is given.
    values = []
    for raw, suffix in re.findall(r"(\d+(?:\.\d+)?)\s*([kmbt]?)", text):
        parsed = float(raw)
        parsed *= {"": 1.0, "k": 1e-6, "m": 1e-3, "b": 1.0, "t": 1e3}[suffix]
        if parsed > 0:
            values.append(parsed)
    if not values:
        return None
    result = min(values)
    # Ollama can expose a raw parameter count such as 116829156672 instead of
    # a human-readable 116.8B value. Normalize both representations to billions.
    return result / 1e9 if result >= 1e6 else result


def _integer(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None


def _band_for_parameters(parameter_count_billions: float | None) -> CapabilityBand:
    if parameter_count_billions is None or parameter_count_billions < 4:
        return CapabilityBand.MINIMAL
    if parameter_count_billions < 15:
        return CapabilityBand.LIMITED
    if parameter_count_billions < 70:
        return CapabilityBand.STANDARD
    return CapabilityBand.HIGH


@dataclass(frozen=True, slots=True)
class ModelCapabilityEnvelopeV1:
    provider: str
    model: str
    model_fingerprint: str
    execution_class: str
    parameter_count_billions: float | None
    context_window_tokens: int | None
    maximum_output_tokens: int | None
    tool_calling: bool
    structured_output: bool
    thinking: bool
    vision: bool
    max_concurrency: int
    capability_band: CapabilityBand
    metadata_complete: bool
    metadata_completeness: float
    sources: Mapping[str, str]
    version: int = 1

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ModelCapabilityEnvelopeV1":
        try:
            band = CapabilityBand(str(value.get("capability_band") or "minimal"))
        except ValueError as exc:
            raise ValueError("capability_band is invalid") from exc
        return cls(
            provider=str(value.get("provider") or "unknown"),
            model=str(value.get("model") or "unknown"),
            model_fingerprint=str(value.get("model_fingerprint") or ""),
            execution_class=str(value.get("execution_class") or "local"),
            parameter_count_billions=(
                float(value["parameter_count_billions"])
                if value.get("parameter_count_billions") is not None
                else None
            ),
            context_window_tokens=_integer(value.get("context_window_tokens")),
            maximum_output_tokens=_integer(value.get("maximum_output_tokens")),
            tool_calling=bool(value.get("tool_calling", False)),
            structured_output=bool(value.get("structured_output", False)),
            thinking=bool(value.get("thinking", False)),
            vision=bool(value.get("vision", False)),
            max_concurrency=max(1, int(value.get("max_concurrency", 1))),
            capability_band=band,
            metadata_complete=bool(value.get("metadata_complete", False)),
            metadata_completeness=max(
                0.0,
                min(
                    1.0,
                    float(
                        value.get(
                            "metadata_completeness",
                            1.0 if value.get("metadata_complete") else 0.0,
                        )
                    ),
                ),
            ),
            sources=dict(value.get("sources") or {}),
            version=int(value.get("version", 1)),
        )

    @classmethod
    def from_metadata(
        cls,
        *,
        provider: str,
        model: str,
        execution_class: str,
        capabilities: Sequence[str] = (),
        metadata: Mapping[str, Any] | None = None,
        provider_profile: Any = None,
        default_concurrency: int = 1,
    ) -> "ModelCapabilityEnvelopeV1":
        values = dict(metadata or {})
        normalized_capabilities = {str(item).strip().casefold() for item in capabilities}
        active_parameters = values.get("active_parameter_size")
        parameters = _number(
            active_parameters
            if active_parameters is not None
            else values.get("parameter_size", values.get("parameter_count"))
        )
        context = _integer(
            values.get("context_window_tokens", values.get("context_length", values.get("context_size")))
        )
        maximum_output = _integer(
            values.get("maximum_output_tokens", values.get("max_output_tokens"))
        )
        if provider_profile is not None:
            context = context or _integer(getattr(provider_profile, "context_size", None))
            maximum_output = maximum_output or _integer(
                getattr(provider_profile, "maximum_output_size", None)
            )
        documented_concurrency = _integer(
            values.get("max_concurrency", values.get("provider_concurrency"))
        )
        if documented_concurrency is None and provider_profile is not None:
            documented_concurrency = _integer(
                getattr(provider_profile, "max_concurrency", None)
            )

        explicit_band_raw = str(values.get("capability_band") or "").strip().casefold()
        try:
            explicit_band = CapabilityBand(explicit_band_raw) if explicit_band_raw else None
        except ValueError:
            explicit_band = None
        # Unknown size *or* unknown context is deliberately treated as the
        # weakest trusted envelope.  This is conservative by product policy.
        complete = parameters is not None and context is not None
        band = (explicit_band or _band_for_parameters(parameters)) if complete else CapabilityBand.MINIMAL
        completeness_parts = (
            parameters is not None,
            context is not None,
            maximum_output is not None,
            bool(normalized_capabilities) or provider_profile is not None,
            documented_concurrency is not None,
        )
        completeness = sum(1 for present in completeness_parts if present) / len(completeness_parts)

        def profile_bool(name: str, aliases: set[str]) -> bool:
            profile_value = getattr(provider_profile, name, None) if provider_profile is not None else None
            descriptor_value = bool(normalized_capabilities & aliases)
            if profile_value is True:
                return True
            if profile_value is False:
                # A newly-created provider profile frequently starts with
                # conservative False defaults before its handshake has run.
                # Do not let that erase documented catalog capabilities. A
                # completed healthy probe may still authoritatively disable a
                # capability the endpoint rejected.
                health = str(getattr(provider_profile, "health_status", "") or "").casefold()
                if health in {"healthy", "ready", "available", "supported"}:
                    return False
            return descriptor_value

        digest = str(
            values.get("digest")
            or values.get("model_fingerprint")
            or getattr(provider_profile, "model_fingerprint", "")
            or ""
        )
        if not digest:
            digest = hashlib.sha256(f"{provider}\0{model}\0{parameters}\0{context}".encode()).hexdigest()
        sources = {
            "parameters": "active_parameter_size" if active_parameters is not None else ("metadata" if parameters is not None else "unknown"),
            "context": "metadata_or_provider" if context is not None else "unknown",
            "maximum_output": "metadata_or_provider" if maximum_output is not None else "unknown",
            "band": "explicit_metadata" if explicit_band is not None and complete else ("parameter_band" if complete else "conservative_unknown"),
            "capabilities": "provider_profile" if provider_profile is not None else "descriptor",
            "concurrency": "metadata_or_provider" if documented_concurrency is not None else "conservative_single_worker",
        }
        return cls(
            provider=str(provider).strip().casefold() or "unknown",
            model=str(model).strip() or "unknown",
            model_fingerprint=digest,
            execution_class=str(execution_class).strip().casefold() or "local",
            parameter_count_billions=parameters,
            context_window_tokens=context,
            maximum_output_tokens=maximum_output,
            tool_calling=profile_bool("tool_call_support", {"tools", "tool_calling", "tool-calling"}),
            structured_output=profile_bool("structured_output_support", {"structured", "structured_output", "json_schema"}),
            thinking=profile_bool("thinking_support", {"thinking", "reasoning"}),
            vision=profile_bool("vision_support", {"vision"}),
            max_concurrency=max(1, int(documented_concurrency or 1)),
            capability_band=band,
            metadata_complete=complete,
            metadata_completeness=completeness,
            sources=sources,
        )

    @property
    def level(self) -> int:
        return self.capability_band.level

    @property
    def max_cohesive_components(self) -> int:
        return self.capability_band.max_cohesive_components

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(
            json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["capability_band"] = self.capability_band.value
        value["sources"] = dict(self.sources)
        value["fingerprint"] = hashlib.sha256(
            json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()
        return value


_DEMAND_FIELDS = (
    "reasoning",
    "implementation",
    "context_breadth",
    "coordination",
    "verification",
    "visual_runtime",
)


_DEMAND_LEVEL_ALIASES = {
    "none": 1,
    "not applicable": 1,
    "n/a": 1,
    "minimal": 1,
    "very low": 1,
    "low": 1,
    "simple": 1,
    "limited": 2,
    "medium": 2,
    "moderate": 2,
    "standard": 2,
    "high": 3,
    "complex": 3,
    "very high": 4,
    "extreme": 4,
    "maximum": 4,
}


def _demand_level(raw: Any, *, field: str) -> int:
    """Normalize weak-model transport variants without authoring demand."""

    original = raw
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        # Missing semantic capability data must never make work look easier or
        # stop an otherwise valid route. Treat it as maximum demand so the
        # deterministic strategy selector can only become more conservative.
        return 4
    if isinstance(raw, bool):
        raise ValueError(
            f"task_demand.{field} must be an integer from 1 to 4; received={original!r}"
        )
    if isinstance(raw, Mapping):
        for key in ("level", "value", "score", "rating"):
            if key in raw:
                raw = raw[key]
                break
    if isinstance(raw, str):
        normalized = " ".join(raw.strip().lower().replace("_", " ").replace("-", " ").split())
        if normalized in _DEMAND_LEVEL_ALIASES:
            raw = _DEMAND_LEVEL_ALIASES[normalized]
        else:
            numeric = re.search(r"(?<!\d)([1-4])(?!\d)", normalized)
            if numeric:
                raw = int(numeric.group(1))
            elif "very high" in normalized or "extreme" in normalized or "maximum" in normalized:
                raw = 4
            elif "high" in normalized or "complex" in normalized:
                raw = 3
            elif "medium" in normalized or "moderate" in normalized or "standard" in normalized:
                raw = 2
            elif "low" in normalized or "minimal" in normalized or "simple" in normalized:
                raw = 1
            else:
                raw = normalized
    try:
        level = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"task_demand.{field} must be an integer from 1 to 4; received={original!r}"
        ) from exc
    if level not in {1, 2, 3, 4}:
        raise ValueError(
            f"task_demand.{field} must be an integer from 1 to 4; received={original!r}"
        )
    return level


def _transport_bool(raw: Any, *, field: str) -> bool:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        normalized = raw.strip().lower()
        if normalized in {"true", "yes", "1"}:
            return True
        if normalized in {"false", "no", "0"}:
            return False
    raise ValueError(f"task_demand.{field} must be a boolean")


@dataclass(frozen=True, slots=True)
class TaskDemandV1:
    reasoning: int
    implementation: int
    context_breadth: int
    coordination: int
    verification: int
    visual_runtime: int
    component_count: int
    independently_parallelizable: bool
    rationale: tuple[str, ...]
    version: int = 1

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TaskDemandV1":
        levels: dict[str, int] = {}
        for field in _DEMAND_FIELDS:
            levels[field] = _demand_level(value.get(field), field=field)
        raw_count = value.get("component_count", 1)
        if isinstance(raw_count, bool):
            raise ValueError("task_demand.component_count must be a positive integer")
        try:
            count = int(raw_count)
        except (TypeError, ValueError) as exc:
            raise ValueError("task_demand.component_count must be a positive integer") from exc
        if count < 1:
            raise ValueError("task_demand.component_count must be a positive integer")
        parallel = _transport_bool(
            value.get("independently_parallelizable", False),
            field="independently_parallelizable",
        )
        raw_rationale = value.get("rationale", ())
        if isinstance(raw_rationale, str):
            # Weak structured-output models often collapse a one-item string
            # array to its scalar value. This repairs transport shape only; the
            # model-authored meaning is preserved verbatim.
            raw_rationale = (raw_rationale,)
        if not isinstance(raw_rationale, Sequence) or isinstance(raw_rationale, bytes):
            raise ValueError("task_demand.rationale must be an array of strings")
        rationale = tuple(str(item).strip() for item in raw_rationale if str(item).strip())
        if not rationale:
            raise ValueError("task_demand.rationale requires at least one model-authored reason")
        return cls(**levels, component_count=count, independently_parallelizable=parallel, rationale=rationale)

    @classmethod
    def from_legacy(
        cls,
        *,
        component_count: int,
        parallelism_required: bool,
        reasons: Sequence[str],
    ) -> "TaskDemandV1":
        # Compatibility projection from already model-authored V3 facts.  It
        # never inspects the user's words or invents product semantics.
        component_level = 1 if component_count <= 1 else 2 if component_count <= 2 else 3 if component_count <= 4 else 4
        return cls(
            reasoning=component_level,
            implementation=component_level,
            context_breadth=component_level,
            coordination=max(component_level, 3 if parallelism_required else 1),
            verification=max(1, component_level),
            visual_runtime=1,
            component_count=max(1, component_count),
            independently_parallelizable=bool(parallelism_required),
            rationale=tuple(str(item) for item in reasons if str(item).strip()) or ("legacy model-authored intake",),
        )

    @property
    def maximum_level(self) -> int:
        return max(getattr(self, field) for field in _DEMAND_FIELDS)

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(
            json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["rationale"] = list(self.rationale)
        return value


@dataclass(frozen=True, slots=True)
class StrategyDecisionV1:
    strategy: ExecutionStrategyV1
    reasons: tuple[str, ...]
    capability_fingerprint: str
    demand_fingerprint: str
    max_concurrency: int
    locked: bool = False
    version: int = 1

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "StrategyDecisionV1":
        return cls(
            strategy=ExecutionStrategyV1.parse(value.get("strategy")),
            reasons=tuple(str(item) for item in value.get("reasons", ()) if str(item).strip()),
            capability_fingerprint=str(value.get("capability_fingerprint") or ""),
            demand_fingerprint=str(value.get("demand_fingerprint") or ""),
            max_concurrency=max(1, int(value.get("max_concurrency", 1))),
            locked=bool(value.get("locked", False)),
            version=int(value.get("version", 1)),
        )

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(
            json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "strategy": self.strategy.value,
            "reasons": list(self.reasons),
            "capability_fingerprint": self.capability_fingerprint,
            "demand_fingerprint": self.demand_fingerprint,
            "max_concurrency": self.max_concurrency,
            "locked": self.locked,
        }

    def lock(self) -> "StrategyDecisionV1":
        return StrategyDecisionV1(
            self.strategy,
            self.reasons,
            self.capability_fingerprint,
            self.demand_fingerprint,
            self.max_concurrency,
            True,
        )


def select_execution_strategy(
    envelope: ModelCapabilityEnvelopeV1,
    demand: TaskDemandV1,
    *,
    minimum: ExecutionStrategyV1 = ExecutionStrategyV1.STAGED,
    allow_capability_escalation: bool = True,
) -> StrategyDecisionV1:
    reasons: list[str] = []
    requires_recursive = False
    if allow_capability_escalation and demand.maximum_level > envelope.level:
        requires_recursive = True
        reasons.append(
            f"task demand {demand.maximum_level} exceeds model capability band {envelope.level}"
        )
    if allow_capability_escalation and demand.component_count > envelope.max_cohesive_components:
        requires_recursive = True
        reasons.append(
            f"{demand.component_count} components exceed cohesive limit {envelope.max_cohesive_components}"
        )
    elif not allow_capability_escalation and (
        demand.maximum_level > envelope.level
        or demand.component_count > envelope.max_cohesive_components
    ):
        reasons.append(
            "task demand exceeds this model's cohesive envelope; keep the selected "
            "strategy and narrow packets before mutation"
        )
    if not envelope.metadata_complete:
        reasons.append("model metadata is incomplete, so the conservative minimal envelope applies")
    if minimum is ExecutionStrategyV1.RECURSIVE:
        requires_recursive = True
        reasons.append("execution depth was explicitly increased before approval")
    recursive = requires_recursive
    strategy = ExecutionStrategyV1.RECURSIVE if recursive else ExecutionStrategyV1.STAGED
    concurrency = (
        min(envelope.max_concurrency, demand.component_count)
        if strategy is ExecutionStrategyV1.RECURSIVE and demand.independently_parallelizable
        else 1
    )
    return StrategyDecisionV1(
        strategy,
        tuple(reasons or ("task demand fits the selected model capability envelope",)),
        envelope.fingerprint,
        demand.fingerprint,
        max(1, concurrency),
    )


__all__ = [
    "CapabilityBand",
    "ExecutionStrategyV1",
    "InteractionModeV2",
    "LocalAdaptationPolicy",
    "ModelCapabilityEnvelopeV1",
    "StrategyDecisionV1",
    "TaskDemandV1",
    "select_execution_strategy",
]
