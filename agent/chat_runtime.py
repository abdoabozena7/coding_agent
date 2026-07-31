"""Model-authored turn routing contracts and deterministic validation.

The model owns semantic interpretation.  This module deliberately contains no
keyword lists, prompt-length thresholds, or product-name rules.  The harness
only validates the structured decision and turns its requested effects into a
bounded tool contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from typing import Any, Iterable, Mapping, Sequence


class RouteKind(str, Enum):
    CHAT = "chat"
    ACTION = "action"
    GOAL = "goal"


class RequestedEffectV2(str, Enum):
    READ = "read"
    WRITE = "write"
    RUN = "run"
    INSTALL = "install"
    PREVIEW = "preview"
    EXTERNAL = "external_side_effect"

    @classmethod
    def parse(cls, value: Any) -> "RequestedEffectV2":
        normalized = str(getattr(value, "value", value)).strip().casefold()
        aliases = {
            "read_workspace": "read",
            "write_workspace": "write",
            "run_command": "run",
            "install_dependency": "install",
            "install_dependencies": "install",
            "preview_or_open": "preview",
            "open": "preview",
        }
        try:
            return cls(aliases.get(normalized, normalized))
        except ValueError as exc:
            raise ValueError(f"unknown requested effect: {value!r}") from exc


_MUTATING_EFFECTS = frozenset(
    {RequestedEffectV2.WRITE, RequestedEffectV2.RUN, RequestedEffectV2.INSTALL,
     RequestedEffectV2.PREVIEW, RequestedEffectV2.EXTERNAL}
)


@dataclass(frozen=True, slots=True)
class SemanticIntakeV2:
    objective: str
    deliverables: tuple[str, ...]
    constraints: tuple[str, ...]
    exclusions: tuple[str, ...]
    acceptance_expectations: tuple[str, ...]
    assumptions: tuple[str, ...]
    risks: tuple[str, ...]
    breadth: str
    coordination: str
    uncertainty: str
    complexity_reasons: tuple[str, ...]
    recommended_mode: str
    recommendation_reason: str
    questions: tuple[Mapping[str, Any], ...] = ()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SemanticIntakeV2":
        def required(name: str) -> str:
            result = str(value.get(name) or "").strip()
            if not result:
                raise ValueError(f"goal_intake.{name} is required")
            return result

        def strings(name: str) -> tuple[str, ...]:
            raw = value.get(name, ())
            if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
                raise ValueError(f"goal_intake.{name} must be an array of strings")
            return tuple(str(item).strip() for item in raw if str(item).strip())

        questions = value.get("questions", ())
        if not isinstance(questions, Sequence) or isinstance(questions, (str, bytes)):
            raise ValueError("goal_intake.questions must be an array")
        if len(questions) > 3 or any(not isinstance(item, Mapping) for item in questions):
            raise ValueError("goal_intake.questions must contain at most three objects")
        recommended = required("recommended_mode").casefold()
        if recommended not in {"normal", "ultra"}:
            raise ValueError("goal_intake.recommended_mode must be normal or ultra")
        breadth = required("breadth").casefold()
        if breadth not in {"bounded", "cohesive", "multi_component"}:
            raise ValueError("goal_intake.breadth is invalid")
        coordination = required("coordination").casefold()
        if coordination not in {"single", "sequential", "parallel", "recursive"}:
            raise ValueError("goal_intake.coordination is invalid")
        return cls(
            objective=required("objective"),
            deliverables=strings("deliverables"),
            constraints=strings("constraints"),
            exclusions=strings("exclusions"),
            acceptance_expectations=strings("acceptance_expectations"),
            assumptions=strings("assumptions"),
            risks=strings("risks"),
            breadth=breadth,
            coordination=coordination,
            uncertainty=required("uncertainty"),
            complexity_reasons=strings("complexity_reasons"),
            recommended_mode=recommended,
            recommendation_reason=required("recommendation_reason"),
            questions=tuple(dict(item) for item in questions),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "deliverables": list(self.deliverables),
            "constraints": list(self.constraints),
            "exclusions": list(self.exclusions),
            "acceptance_expectations": list(self.acceptance_expectations),
            "assumptions": list(self.assumptions),
            "risks": list(self.risks),
            "breadth": self.breadth,
            "coordination": self.coordination,
            "uncertainty": self.uncertainty,
            "complexity_reasons": list(self.complexity_reasons),
            "recommended_mode": self.recommended_mode,
            "recommendation_reason": self.recommendation_reason,
            "questions": [dict(item) for item in self.questions],
        }


@dataclass(frozen=True, slots=True)
class SemanticTurnDecisionV2:
    route: RouteKind
    interpretation: str
    requested_effects: tuple[RequestedEffectV2, ...]
    authority_spans: Mapping[str, tuple[str, ...]]
    needs_workspace_tools: bool
    direct_response: str
    uncertainty: str
    clarification_question: str
    goal_intake: SemanticIntakeV2 | None = None
    version: int = 2

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        original_input: str,
        forced_route: RouteKind | None = None,
    ) -> "SemanticTurnDecisionV2":
        # Some providers preserve all authored fields but wrap a subset under
        # the function's semantic_turn label. Flatten that transport shape;
        # this copies model output verbatim and never supplies semantics.
        source = dict(value)
        nested = source.get("semantic_turn")
        if isinstance(nested, Mapping):
            for field in (
                "interpretation", "requested_effects", "uncertainty",
                "clarification_question", "goal_intake",
            ):
                if field not in source and field in nested:
                    source[field] = nested[field]
            if (
                str(source.get("route") or "").strip().casefold() == "goal"
                and "goal_intake" not in source
                and str(nested.get("objective") or "").strip()
            ):
                source["goal_intake"] = dict(nested)
        value = source
        try:
            route = RouteKind(str(value.get("route") or "").strip().casefold())
        except ValueError as exc:
            raise ValueError("route must be chat, action, or goal") from exc
        if forced_route is not None and route is not forced_route:
            raise ValueError(f"route must be {forced_route.value} for this explicit command")
        interpretation = str(value.get("interpretation") or "").strip()
        if not interpretation:
            raise ValueError("interpretation is required")
        raw_effects = value.get("requested_effects", ())
        if isinstance(raw_effects, Mapping):
            selected_effects: list[str] = []
            for name, selected in raw_effects.items():
                if isinstance(selected, bool):
                    enabled = selected
                elif isinstance(selected, Sequence) and not isinstance(selected, (str, bytes)):
                    enabled = bool(selected)
                else:
                    raise ValueError(
                        "requested_effects object values must be booleans or span arrays"
                    )
                if enabled:
                    selected_effects.append(str(name))
            raw_effects = selected_effects
        if not isinstance(raw_effects, Sequence) or isinstance(raw_effects, (str, bytes)):
            raise ValueError("requested_effects must be an array or canonical boolean object")
        effects = tuple(dict.fromkeys(RequestedEffectV2.parse(item) for item in raw_effects))
        raw_spans = value.get("authority_spans", {})
        if not isinstance(raw_spans, Mapping):
            raise ValueError("authority_spans must be an object")
        spans: dict[str, tuple[str, ...]] = {}
        for effect in effects:
            raw = raw_spans.get(effect.value, ())
            if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
                raise ValueError(f"authority_spans.{effect.value} must be an array")
            normalized = tuple(str(item) for item in raw if str(item))
            if effect in _MUTATING_EFFECTS and not normalized:
                raise ValueError(f"authority_spans.{effect.value} is required")
            for span in normalized:
                if span not in original_input:
                    raise ValueError(
                        f"authority span for {effect.value} is not verbatim user input: {span!r}"
                    )
            spans[effect.value] = normalized
        needs_tools = bool(value.get("needs_workspace_tools", False))
        direct_response = str(value.get("direct_response") or "").strip()
        uncertainty = str(value.get("uncertainty") or "clear").strip().casefold()
        clarification = str(value.get("clarification_question") or "").strip()
        if uncertainty not in {"clear", "clarification_needed"}:
            raise ValueError("uncertainty must be clear or clarification_needed")
        if uncertainty == "clarification_needed" and not clarification:
            raise ValueError("clarification_question is required when uncertainty is unresolved")
        intake_raw = value.get("goal_intake")
        intake = SemanticIntakeV2.from_mapping(intake_raw) if isinstance(intake_raw, Mapping) else None

        if route is RouteKind.CHAT:
            if any(effect is not RequestedEffectV2.READ for effect in effects):
                raise ValueError("chat may request only the read effect")
            if effects and not needs_tools:
                raise ValueError("chat read effects require needs_workspace_tools=true")
            if not needs_tools and not direct_response:
                raise ValueError("direct chat requires a natural direct_response")
            if intake is not None:
                raise ValueError("chat must not include goal_intake")
        elif route is RouteKind.ACTION:
            if not effects:
                raise ValueError("action requires at least one requested effect")
            if not needs_tools:
                raise ValueError("action requires workspace tools")
            if intake is not None:
                raise ValueError("action must not include goal_intake")
        else:
            if intake is None:
                raise ValueError("goal requires goal_intake")
            if direct_response:
                raise ValueError("goal must not claim a direct response")

        return cls(
            route=route,
            interpretation=interpretation,
            requested_effects=effects,
            authority_spans=spans,
            needs_workspace_tools=needs_tools,
            direct_response=direct_response,
            uncertainty=uncertainty,
            clarification_question=clarification,
            goal_intake=intake,
        )

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(
            json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()

    @property
    def actionable(self) -> bool:
        return self.route is RouteKind.ACTION

    @property
    def allowed_categories(self) -> frozenset[str]:
        categories = {"read"}
        mapping = {
            RequestedEffectV2.WRITE: {"write"},
            RequestedEffectV2.RUN: {"command", "process"},
            RequestedEffectV2.INSTALL: {"install"},
            RequestedEffectV2.PREVIEW: {"preview", "open", "process"},
        }
        for effect in self.requested_effects:
            categories.update(mapping.get(effect, set()))
        return frozenset(categories)

    def permits_tool_category(self, category: str) -> bool:
        return str(category) in self.allowed_categories

    def missing_effects(self, successful_categories: Iterable[str]) -> tuple[str, ...]:
        categories = set(successful_categories)
        expected = {
            RequestedEffectV2.READ: {"read"},
            RequestedEffectV2.WRITE: {"write"},
            RequestedEffectV2.RUN: {"command", "process"},
            RequestedEffectV2.INSTALL: {"install"},
            RequestedEffectV2.PREVIEW: {"preview", "open"},
            RequestedEffectV2.EXTERNAL: {"external"},
        }
        return tuple(
            effect.value for effect in self.requested_effects
            if not categories.intersection(expected[effect])
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "route": self.route.value,
            "interpretation": self.interpretation,
            "requested_effects": {
                item.value: item in self.requested_effects for item in RequestedEffectV2
            },
            "authority_spans": {key: list(value) for key, value in self.authority_spans.items()},
            "needs_workspace_tools": self.needs_workspace_tools,
            "direct_response": self.direct_response,
            "uncertainty": self.uncertainty,
            "clarification_question": self.clarification_question,
            "goal_intake": self.goal_intake.to_dict() if self.goal_intake else None,
        }


@dataclass(frozen=True, slots=True)
class RouteDecisionV1:
    """Compatibility view over the accepted V2 semantic decision."""

    kind: RouteKind
    reason: str
    explicit: bool = False
    semantic: SemanticTurnDecisionV2 | None = None


def _tool_schema(name: str, description: str, parameters: Mapping[str, Any]) -> dict[str, Any]:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": dict(parameters)}}


_QUESTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["id", "header", "question", "reason", "options"],
    "properties": {
        "id": {"type": "string"},
        "header": {"type": "string"},
        "question": {"type": "string"},
        "reason": {"type": "string"},
        "options": {
            "type": "array", "minItems": 3, "maxItems": 3,
            "items": {"type": "object", "required": ["value", "label", "description", "recommended"],
                "properties": {
                    "value": {"type": "string"}, "label": {"type": "string"},
                    "description": {"type": "string"}, "recommended": {"type": "boolean"},
                }},
        },
    },
}


SEMANTIC_TURN_SCHEMA = _tool_schema(
    "submit_semantic_turn",
    "Classify the actual requested outcome and effects. This is semantic interpretation, not execution.",
    {
        "type": "object",
        "required": ["route", "interpretation", "requested_effects", "authority_spans", "needs_workspace_tools", "direct_response", "uncertainty", "clarification_question"],
        "properties": {
            "route": {"type": "string", "enum": ["chat", "action", "goal"]},
            "interpretation": {"type": "string"},
            "requested_effects": {
                "type": "object",
                "description": "Set each canonical effect to true only when the user requested it.",
                "properties": {
                    item.value: {"type": "boolean"} for item in RequestedEffectV2
                },
                "required": [item.value for item in RequestedEffectV2],
                "additionalProperties": False,
            },
            "authority_spans": {
                "type": "object",
                "description": (
                    "Verbatim substrings from exact_latest_user_input, grouped by the canonical "
                    "effect name. Supply every property; use an empty array when that effect was "
                    "not requested. Changing effects require at least one exact substring."
                ),
                "properties": {
                    item.value: {"type": "array", "items": {"type": "string"}}
                    for item in RequestedEffectV2
                },
                "required": [item.value for item in RequestedEffectV2],
                "additionalProperties": False,
            },
            "needs_workspace_tools": {"type": "boolean"},
            "direct_response": {"type": "string"},
            "uncertainty": {"type": "string", "enum": ["clear", "clarification_needed"]},
            "clarification_question": {"type": "string"},
            "goal_intake": {
                "type": "object",
                "required": ["objective", "deliverables", "constraints", "exclusions", "acceptance_expectations", "assumptions", "risks", "breadth", "coordination", "uncertainty", "complexity_reasons", "recommended_mode", "recommendation_reason", "questions"],
                "properties": {
                    "objective": {"type": "string"},
                    "deliverables": {"type": "array", "items": {"type": "string"}},
                    "constraints": {"type": "array", "items": {"type": "string"}},
                    "exclusions": {"type": "array", "items": {"type": "string"}},
                    "acceptance_expectations": {"type": "array", "items": {"type": "string"}},
                    "assumptions": {"type": "array", "items": {"type": "string"}},
                    "risks": {"type": "array", "items": {"type": "string"}},
                    "breadth": {"type": "string", "enum": ["bounded", "cohesive", "multi_component"]},
                    "coordination": {"type": "string", "enum": ["single", "sequential", "parallel", "recursive"]},
                    "uncertainty": {"type": "string"},
                    "complexity_reasons": {"type": "array", "items": {"type": "string"}},
                    "recommended_mode": {"type": "string", "enum": ["normal", "ultra"]},
                    "recommendation_reason": {"type": "string"},
                    "questions": {"type": "array", "maxItems": 3, "items": _QUESTION_SCHEMA},
                },
            },
        },
    },
)


def corrective_prompt(missing: tuple[str, ...], capabilities: str) -> str:
    return (
        "HARNESS EFFECT CONTRACT: The accepted semantic decision requested effects that still "
        f"lack tool evidence: {', '.join(missing)}. Use only an available tool category in the "
        "accepted contract. Do not replace execution with a prose claim. If execution is "
        "impossible, report the concrete tool or permission error.\nCapabilities: " + capabilities
    )


__all__ = [
    "RequestedEffectV2", "RouteDecisionV1", "RouteKind", "SEMANTIC_TURN_SCHEMA",
    "SemanticIntakeV2", "SemanticTurnDecisionV2", "corrective_prompt",
]
