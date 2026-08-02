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

from .capability import TaskDemandV1


def _transport_bool(value: Any, *, field: str) -> bool:
    """Accept boolean spellings emitted by weak structured-output models."""

    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "yes", "1"}:
            return True
        if normalized in {"false", "no", "0"}:
            return False
    raise ValueError(f"{field} must be a boolean")


def _uncertainty_value(value: Any) -> str:
    normalized = " ".join(
        str(value or "clear").strip().casefold().replace("_", " ").replace("-", " ").split()
    )
    aliases = {
        "none": "clear",
        "no": "clear",
        "low": "clear",
        "certain": "clear",
        "resolved": "clear",
        "unambiguous": "clear",
        "clarification needed": "clarification_needed",
        "needs clarification": "clarification_needed",
        "unclear": "clarification_needed",
        "ambiguous": "clarification_needed",
        "high": "clarification_needed",
    }
    return aliases.get(normalized, normalized.replace(" ", "_"))


class SemanticContractError(ValueError):
    """Structured provider-contract failure suitable for durable recovery UI."""

    def __init__(
        self,
        path: str,
        message: str,
        *,
        received: Any = None,
        allowed: Sequence[Any] = (),
    ) -> None:
        self.path = str(path)
        self.received = received
        self.allowed = tuple(allowed)
        detail = f"{self.path}: {message}"
        if received is not None:
            detail += f"; received={received!r}"
        if self.allowed:
            detail += "; allowed=" + ", ".join(repr(item) for item in self.allowed)
        super().__init__(detail)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "message": str(self),
            "received": self.received,
            "allowed": list(self.allowed),
        }


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
            "read_file": "read",
            "list_files": "read",
            "grep": "read",
            "write_workspace": "write",
            "write_file": "write",
            "edit_file": "write",
            "apply_patch": "write",
            "materialize_artifact": "write",
            "run_command": "run",
            "execute_shell": "run",
            "run_shell": "run",
            "shell": "run",
            "run_bash": "run",
            "start_process": "run",
            "install_dependency": "install",
            "install_dependencies": "install",
            "preview_or_open": "preview",
            "open": "preview",
            "preview_html": "preview",
            "inspect_preview": "preview",
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
class DecisionNeedV1:
    """Model-authored proof that a question needs user authority now."""

    impact: str
    affected_scope: tuple[str, ...]
    affected_effects: tuple[str, ...]
    reversible: bool
    requires_user_authority: bool
    reason: str
    evidence_refs: tuple[str, ...] = ()
    version: int = 1

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "DecisionNeedV1":
        def text(name: str) -> str:
            result = str(value.get(name) or "").strip()
            if not result:
                raise ValueError(f"decision_need.{name} is required")
            return result

        def strings(name: str) -> tuple[str, ...]:
            raw = value.get(name, ())
            if isinstance(raw, str):
                raw = (raw,)
            if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
                raise ValueError(f"decision_need.{name} must be an array of strings")
            return tuple(str(item).strip() for item in raw if str(item).strip())

        reversible = _transport_bool(value.get("reversible"), field="decision_need.reversible")
        authority = _transport_bool(
            value.get("requires_user_authority"),
            field="decision_need.requires_user_authority",
        )
        scope = strings("affected_scope")
        effects = strings("affected_effects")
        if not scope and not effects:
            raise ValueError(
                "decision_need must identify affected_scope or affected_effects"
            )
        return cls(
            impact=text("impact"),
            affected_scope=scope,
            affected_effects=effects,
            reversible=reversible,
            requires_user_authority=authority,
            reason=text("reason"),
            evidence_refs=strings("evidence_refs"),
        )

    @property
    def blocks_work(self) -> bool:
        return self.requires_user_authority

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "impact": self.impact,
            "affected_scope": list(self.affected_scope),
            "affected_effects": list(self.affected_effects),
            "reversible": self.reversible,
            "requires_user_authority": self.requires_user_authority,
            "reason": self.reason,
            "evidence_refs": list(self.evidence_refs),
        }


@dataclass(frozen=True, slots=True)
class SemanticGoalIntakeV3:
    objective: str
    deliverables: tuple[str, ...]
    constraints: tuple[str, ...]
    exclusions: tuple[str, ...]
    acceptance_expectations: tuple[str, ...]
    assumptions: tuple[str, ...]
    risks: tuple[str, ...]
    component_count: int
    parallelism_required: bool
    coordination_summary: str
    uncertainty: str
    complexity_reasons: tuple[str, ...]
    task_demand: TaskDemandV1
    questions: tuple[Mapping[str, Any], ...] = ()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SemanticGoalIntakeV3":
        def required(name: str) -> str:
            result = str(value.get(name) or "").strip()
            if not result:
                raise ValueError(f"goal_intake.{name} is required")
            return result

        def strings(name: str) -> tuple[str, ...]:
            raw = value.get(name, ())
            if isinstance(raw, str):
                raw = (raw,)
            if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
                raise ValueError(f"goal_intake.{name} must be an array of strings")
            return tuple(str(item).strip() for item in raw if str(item).strip())

        questions = value.get("questions", ())
        if isinstance(questions, Mapping):
            questions = (questions,) if questions else ()
        if not isinstance(questions, Sequence) or isinstance(questions, (str, bytes)):
            raise ValueError("goal_intake.questions must be an array")
        if len(questions) > 3 or any(not isinstance(item, Mapping) for item in questions):
            raise ValueError("goal_intake.questions must contain at most three objects")
        # Tool-capable weak models occasionally emit `{}` as an array
        # placeholder despite the schema.  It contains no user-facing semantic
        # question, so omit it rather than inventing visible fallback copy.
        usable_questions = tuple(
            dict(item)
            for item in questions
            if str(item.get("question") or "").strip()
        )
        # V3 asks the model for direct facts rather than brittle semantic enum
        # labels.  Old accepted V2 payloads remain readable without migration.
        raw_count = value.get("component_count")
        if raw_count is None:
            legacy_breadth = str(value.get("breadth") or "").strip().casefold()
            raw_count = 2 if legacy_breadth == "multi_component" else 1
        if isinstance(raw_count, bool):
            raise SemanticContractError(
                "goal_intake.component_count", "must be a positive integer", received=raw_count
            )
        try:
            component_count = int(raw_count)
        except (TypeError, ValueError) as exc:
            raise SemanticContractError(
                "goal_intake.component_count", "must be a positive integer", received=raw_count
            ) from exc
        if component_count < 1:
            raise SemanticContractError(
                "goal_intake.component_count", "must be at least 1", received=raw_count
            )

        raw_parallel = value.get("parallelism_required")
        if raw_parallel is None:
            legacy_coordination = str(value.get("coordination") or "").strip().casefold()
            raw_parallel = legacy_coordination in {"parallel", "recursive"}
        try:
            raw_parallel = _transport_bool(
                raw_parallel, field="goal_intake.parallelism_required"
            )
        except ValueError as exc:
            raise SemanticContractError(
                "goal_intake.parallelism_required", "must be a boolean", received=raw_parallel
            ) from exc
        coordination_summary = str(value.get("coordination_summary") or "").strip()
        if not coordination_summary:
            coordination_summary = str(value.get("coordination") or "single coordinated workflow").strip()
        complexity_reasons = strings("complexity_reasons")
        raw_demand = value.get("task_demand")
        if isinstance(raw_demand, Mapping):
            # Some structured-output models place the shared demand facts in
            # the surrounding intake object while emitting only the six level
            # fields inside task_demand. Rebind those exact model-authored
            # values; do not invent a rationale or infer it from the request.
            demand_value = dict(raw_demand)
            if "component_count" not in demand_value:
                demand_value["component_count"] = component_count
            if "independently_parallelizable" not in demand_value:
                demand_value["independently_parallelizable"] = raw_parallel
            if not demand_value.get("rationale") and complexity_reasons:
                demand_value["rationale"] = list(complexity_reasons)
            raw_demand = demand_value
        task_demand = (
            TaskDemandV1.from_mapping(raw_demand)
            if isinstance(raw_demand, Mapping)
            else TaskDemandV1.from_legacy(
                component_count=component_count,
                parallelism_required=raw_parallel,
                reasons=complexity_reasons
                or (str(value.get("recommendation_reason") or "legacy model-authored intake"),),
            )
        )
        return cls(
            objective=required("objective"),
            deliverables=strings("deliverables"),
            constraints=strings("constraints"),
            exclusions=strings("exclusions"),
            acceptance_expectations=strings("acceptance_expectations"),
            assumptions=strings("assumptions"),
            risks=strings("risks"),
            component_count=component_count,
            parallelism_required=raw_parallel,
            coordination_summary=coordination_summary,
            uncertainty=required("uncertainty"),
            complexity_reasons=complexity_reasons or task_demand.rationale,
            task_demand=task_demand,
            questions=usable_questions,
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
            "component_count": self.component_count,
            "parallelism_required": self.parallelism_required,
            "coordination_summary": self.coordination_summary,
            "uncertainty": self.uncertainty,
            "complexity_reasons": list(self.complexity_reasons),
            "task_demand": self.task_demand.to_dict(),
            "questions": [dict(item) for item in self.questions],
        }

    @property
    def recommended_mode(self) -> str:
        """Read-only compatibility projection for pre-V4 integrations."""

        return "ultra" if self.task_demand.maximum_level >= 3 else "normal"

    @property
    def recommendation_reason(self) -> str:
        return "; ".join(self.task_demand.rationale)

    @property
    def breadth(self) -> str:
        """Compatibility view for the persisted V1 complexity assessment."""

        return "multi_component" if self.component_count > 1 else "cohesive"

    @property
    def coordination(self) -> str:
        """Compatibility view derived from the model-authored V3 facts."""

        return "parallel" if self.parallelism_required else "sequential"


# Existing databases and integrations import the V2 name.  The wire parser
# above accepts both old and new payloads, so no data migration is required.
SemanticIntakeV2 = SemanticGoalIntakeV3


@dataclass(frozen=True, slots=True)
class SemanticRouteDecisionV3:
    route: RouteKind
    outcome_kind: str
    interpretation: str
    requested_effects: tuple[RequestedEffectV2, ...]
    authority_spans: Mapping[str, tuple[str, ...]]
    needs_workspace_tools: bool
    direct_response: str
    uncertainty: str
    clarification_question: str
    task_demand: TaskDemandV1
    goal_intake: SemanticGoalIntakeV3 | None = None
    version: int = 4

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        original_input: str,
        forced_route: RouteKind | None = None,
        parse_goal_intake: bool = True,
    ) -> "SemanticRouteDecisionV3":
        # Some providers preserve all authored fields but wrap a subset under
        # the function's semantic_turn label. Flatten that transport shape;
        # this copies model output verbatim and never supplies semantics.
        source = dict(value)
        nested = source.get("semantic_turn")
        if isinstance(nested, Mapping):
            for field in (
                "outcome_kind", "interpretation", "requested_effects", "uncertainty",
                "clarification_question", "task_demand", "goal_intake",
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
        route_raw = str(value.get("route") or "").strip()
        try:
            route = RouteKind(route_raw.casefold())
        except ValueError as exc:
            raise SemanticContractError(
                "route",
                "must select an allowed semantic route",
                received=route_raw,
                allowed=("chat", "action", "goal"),
            ) from exc
        if forced_route is not None and route is not forced_route:
            raise ValueError(f"route must be {forced_route.value} for this explicit command")
        outcome_kind = str(value.get("outcome_kind") or "").strip().casefold()
        if not outcome_kind:
            outcome_kind = {
                RouteKind.CHAT: "conversation",
                RouteKind.ACTION: "workspace_operation",
                RouteKind.GOAL: "durable_project",
            }[route]
        allowed_outcomes = (
            "conversation", "explanation", "workspace_operation",
            "runnable_product", "durable_project",
        )
        if outcome_kind not in allowed_outcomes:
            raise SemanticContractError(
                "outcome_kind", "must select an allowed requested outcome",
                received=outcome_kind, allowed=allowed_outcomes,
            )
        allowed_routes = {
            "conversation": {RouteKind.CHAT},
            "explanation": {RouteKind.CHAT},
            "workspace_operation": {RouteKind.ACTION, RouteKind.GOAL},
            "runnable_product": {RouteKind.GOAL},
            "durable_project": {RouteKind.GOAL},
        }
        if route not in allowed_routes[outcome_kind]:
            raise SemanticContractError(
                "route",
                f"is inconsistent with outcome_kind={outcome_kind!r}",
                received=route.value,
                allowed=tuple(item.value for item in allowed_routes[outcome_kind]),
            )
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
        if isinstance(raw_effects, str):
            raw_effects = (raw_effects,)
        if not isinstance(raw_effects, Sequence) or isinstance(raw_effects, (str, bytes)):
            raise ValueError("requested_effects must be an array or canonical boolean object")
        effects = tuple(dict.fromkeys(RequestedEffectV2.parse(item) for item in raw_effects))
        raw_spans = value.get("authority_spans", {})
        if not isinstance(raw_spans, Mapping):
            raise ValueError("authority_spans must be an object")
        spans: dict[str, tuple[str, ...]] = {}
        authorized_effects: list[RequestedEffectV2] = []
        for effect in effects:
            raw = raw_spans.get(effect.value, ())
            if isinstance(raw, str):
                raw = (raw,)
            if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
                raise ValueError(f"authority_spans.{effect.value} must be an array")
            normalized = tuple(str(item) for item in raw if str(item))
            if effect is RequestedEffectV2.EXTERNAL and not normalized:
                # External authority can never be inferred or mechanically
                # widened from a local build request. Contract the unsupported
                # effect instead of blocking otherwise authorized local work.
                continue
            if effect in _MUTATING_EFFECTS and not normalized:
                raise ValueError(f"authority_spans.{effect.value} is required")
            for span in normalized:
                if span not in original_input:
                    raise ValueError(
                        f"authority span for {effect.value} is not verbatim user input: {span!r}"
                    )
            spans[effect.value] = normalized
            authorized_effects.append(effect)
        effects = tuple(authorized_effects)
        needs_tools = _transport_bool(
            value.get("needs_workspace_tools", False),
            field="needs_workspace_tools",
        )
        direct_response = str(value.get("direct_response") or "").strip()
        uncertainty = _uncertainty_value(value.get("uncertainty"))
        clarification = str(value.get("clarification_question") or "").strip()
        if uncertainty not in {"clear", "clarification_needed"}:
            raise ValueError("uncertainty must be clear or clarification_needed")
        if uncertainty == "clarification_needed" and not clarification:
            raise ValueError("clarification_question is required when uncertainty is unresolved")
        intake_raw = value.get("goal_intake")
        intake = (
            SemanticGoalIntakeV3.from_mapping(intake_raw)
            if parse_goal_intake and isinstance(intake_raw, Mapping)
            else None
        )
        raw_demand = value.get("task_demand")
        task_demand = (
            TaskDemandV1.from_mapping(raw_demand)
            if isinstance(raw_demand, Mapping)
            else TaskDemandV1.from_legacy(
                component_count=intake.component_count if intake is not None else 1,
                parallelism_required=(intake.parallelism_required if intake is not None else False),
                reasons=(
                    intake.complexity_reasons
                    if intake is not None
                    else ("legacy model-authored semantic route",)
                ),
            )
        )

        if route is RouteKind.CHAT:
            if task_demand.implementation > 1:
                raise SemanticContractError(
                    "task_demand.implementation",
                    (
                        "must be 1 for Chat because Chat does not implement a "
                        "workspace outcome; choose Action or Goal when the same "
                        "decision says implementation work is requested"
                    ),
                    received=task_demand.implementation,
                    allowed=(1,),
                )
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
            if direct_response:
                raise ValueError("goal must not claim a direct response")

        return cls(
            route=route,
            outcome_kind=outcome_kind,
            interpretation=interpretation,
            requested_effects=effects,
            authority_spans=spans,
            needs_workspace_tools=needs_tools,
            direct_response=direct_response,
            uncertainty=uncertainty,
            clarification_question=clarification,
            task_demand=task_demand,
            goal_intake=intake,
        )

    def with_goal_intake(self, intake: SemanticGoalIntakeV3) -> "SemanticRouteDecisionV3":
        if self.route is not RouteKind.GOAL:
            raise ValueError("goal intake can be attached only to a Goal route")
        return SemanticRouteDecisionV3(
            route=self.route,
            outcome_kind=self.outcome_kind,
            interpretation=self.interpretation,
            requested_effects=self.requested_effects,
            authority_spans=self.authority_spans,
            needs_workspace_tools=self.needs_workspace_tools,
            direct_response=self.direct_response,
            uncertainty=self.uncertainty,
            clarification_question=self.clarification_question,
            task_demand=self.task_demand,
            goal_intake=intake,
        )

    def promote_action_to_goal(self) -> "SemanticRouteDecisionV3":
        """Safety-only promotion; semantic Chat and Goal decisions never change."""

        if self.route is not RouteKind.ACTION:
            return self
        return SemanticRouteDecisionV3(
            route=RouteKind.GOAL,
            outcome_kind="durable_project",
            interpretation=self.interpretation,
            requested_effects=self.requested_effects,
            authority_spans=self.authority_spans,
            needs_workspace_tools=self.needs_workspace_tools,
            direct_response="",
            uncertainty=self.uncertainty,
            clarification_question=self.clarification_question,
            task_demand=self.task_demand,
            goal_intake=None,
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
            "outcome_kind": self.outcome_kind,
            "interpretation": self.interpretation,
            "requested_effects": {
                item.value: item in self.requested_effects for item in RequestedEffectV2
            },
            "authority_spans": {key: list(value) for key, value in self.authority_spans.items()},
            "needs_workspace_tools": self.needs_workspace_tools,
            "direct_response": self.direct_response,
            "uncertainty": self.uncertainty,
            "clarification_question": self.clarification_question,
            "task_demand": self.task_demand.to_dict(),
            "goal_intake": self.goal_intake.to_dict() if self.goal_intake else None,
        }


SemanticTurnDecisionV2 = SemanticRouteDecisionV3


@dataclass(frozen=True, slots=True)
class RouteDecisionV1:
    """Compatibility view over the accepted model-authored semantic decision."""

    kind: RouteKind
    reason: str
    explicit: bool = False
    semantic: SemanticRouteDecisionV3 | None = None


def _tool_schema(name: str, description: str, parameters: Mapping[str, Any]) -> dict[str, Any]:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": dict(parameters)}}


_QUESTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["question", "options", "decision_need"],
    "properties": {
        "id": {"type": "string"},
        "header": {"type": "string"},
        "question": {"type": "string"},
        "reason": {"type": "string"},
        "options": {
            "type": "array", "minItems": 2, "maxItems": 3,
            "items": {},
        },
        "allow_freeform": {"type": "boolean"},
        "allow_free_form": {"type": "boolean"},
        "decision_need": {
            "type": "object",
            "required": [
                "impact", "affected_scope", "affected_effects", "reversible",
                "requires_user_authority", "reason", "evidence_refs",
            ],
            "properties": {
                "impact": {"type": "string"},
                "affected_scope": {"type": "array", "items": {"type": "string"}},
                "affected_effects": {"type": "array", "items": {"type": "string"}},
                "reversible": {"type": "boolean"},
                "requires_user_authority": {"type": "boolean"},
                "reason": {"type": "string"},
                "evidence_refs": {"type": "array", "items": {"type": "string"}},
            },
            "additionalProperties": False,
        },
    },
    "additionalProperties": True,
}


_ROUTE_PROPERTIES: dict[str, Any] = {
    "route": {"type": "string", "enum": ["chat", "action", "goal"]},
    "outcome_kind": {
        "type": "string",
        "enum": [
            "conversation", "explanation", "workspace_operation",
            "runnable_product", "durable_project",
        ],
    },
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
            "Verbatim substrings from exact_latest_user_input, grouped by canonical effect. "
            "Supply every property and use an empty array for effects not requested."
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
    "task_demand": {
        "type": "object",
        "description": "Model-authored demand relative to the supplied capability envelope.",
        "required": [
            "reasoning", "implementation", "context_breadth", "coordination",
            "verification", "visual_runtime", "component_count",
            "independently_parallelizable", "rationale",
        ],
        "properties": {
            "reasoning": {"type": "integer", "minimum": 1, "maximum": 4},
            "implementation": {"type": "integer", "minimum": 1, "maximum": 4},
            "context_breadth": {"type": "integer", "minimum": 1, "maximum": 4},
            "coordination": {"type": "integer", "minimum": 1, "maximum": 4},
            "verification": {"type": "integer", "minimum": 1, "maximum": 4},
            "visual_runtime": {"type": "integer", "minimum": 1, "maximum": 4},
            "component_count": {"type": "integer", "minimum": 1},
            "independently_parallelizable": {"type": "boolean"},
            "rationale": {"type": "array", "minItems": 1, "items": {"type": "string"}},
        },
        "additionalProperties": False,
    },
}


SEMANTIC_ROUTE_SCHEMA = _tool_schema(
    "submit_semantic_route",
    "Classify the actual requested outcome and effects. This is semantic interpretation, not execution.",
    {
        "type": "object",
        "required": ["route", "outcome_kind", "interpretation", "requested_effects", "authority_spans", "needs_workspace_tools", "direct_response", "uncertainty", "clarification_question", "task_demand"],
        "properties": _ROUTE_PROPERTIES,
        "additionalProperties": False,
    },
)


SEMANTIC_GOAL_INTAKE_SCHEMA = _tool_schema(
    "submit_goal_intake",
    "Describe the accepted Goal without reclassifying it or executing work.",
    {
        "type": "object",
        "required": [
            "objective", "deliverables", "constraints", "exclusions",
            "acceptance_expectations", "assumptions", "risks", "component_count",
            "parallelism_required", "coordination_summary", "uncertainty",
            "complexity_reasons", "task_demand", "questions",
        ],
        "properties": {
            "objective": {"type": "string"},
            "deliverables": {"type": "array", "items": {"type": "string"}},
            "constraints": {"type": "array", "items": {"type": "string"}},
            "exclusions": {"type": "array", "items": {"type": "string"}},
            "acceptance_expectations": {"type": "array", "items": {"type": "string"}},
            "assumptions": {"type": "array", "items": {"type": "string"}},
            "risks": {"type": "array", "items": {"type": "string"}},
            "component_count": {"type": "integer", "minimum": 1},
            "parallelism_required": {"type": "boolean"},
            "coordination_summary": {"type": "string"},
            "uncertainty": {"type": "string"},
            "complexity_reasons": {"type": "array", "items": {"type": "string"}},
            "task_demand": {
                "type": "object",
                "required": [
                    "reasoning", "implementation", "context_breadth", "coordination",
                    "verification", "visual_runtime", "component_count",
                    "independently_parallelizable", "rationale",
                ],
                "properties": {
                    "reasoning": {"type": "integer", "minimum": 1, "maximum": 4},
                    "implementation": {"type": "integer", "minimum": 1, "maximum": 4},
                    "context_breadth": {"type": "integer", "minimum": 1, "maximum": 4},
                    "coordination": {"type": "integer", "minimum": 1, "maximum": 4},
                    "verification": {"type": "integer", "minimum": 1, "maximum": 4},
                    "visual_runtime": {"type": "integer", "minimum": 1, "maximum": 4},
                    "component_count": {"type": "integer", "minimum": 1},
                    "independently_parallelizable": {"type": "boolean"},
                    "rationale": {"type": "array", "minItems": 1, "items": {"type": "string"}},
                },
                "additionalProperties": False,
            },
            "questions": {"type": "array", "maxItems": 3, "items": _QUESTION_SCHEMA},
        },
        "additionalProperties": False,
    },
)


# Compatibility export: callers that used the old schema now receive the
# route-only V3 contract.  Old combined provider output is still parsed.
SEMANTIC_TURN_SCHEMA = SEMANTIC_ROUTE_SCHEMA


def corrective_prompt(missing: tuple[str, ...], capabilities: str) -> str:
    return (
        "HARNESS EFFECT CONTRACT: The accepted semantic decision requested effects that still "
        f"lack tool evidence: {', '.join(missing)}. Use only an available tool category in the "
        "accepted contract. Do not replace execution with a prose claim. If execution is "
        "impossible, report the concrete tool or permission error.\nCapabilities: " + capabilities
    )


__all__ = [
    "DecisionNeedV1", "RequestedEffectV2", "RouteDecisionV1", "RouteKind", "SEMANTIC_TURN_SCHEMA",
    "SEMANTIC_ROUTE_SCHEMA", "SEMANTIC_GOAL_INTAKE_SCHEMA", "SemanticContractError",
    "SemanticGoalIntakeV3", "SemanticIntakeV2", "SemanticRouteDecisionV3",
    "SemanticTurnDecisionV2", "corrective_prompt",
]
