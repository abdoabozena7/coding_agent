"""Validation for model-authored Goal intake.

Semantic interpretation belongs to the configured model.  The deterministic
code here preserves the exact request, validates structure, and applies mode
policy without deriving complexity or ambiguity from words in the request.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from enum import Enum
import hashlib
from typing import Any, Mapping, Sequence

from .chat_runtime import DecisionNeedV1, SemanticIntakeV2


class RunMode(str, Enum):
    NORMAL = "normal"
    PLAN = "plan"
    ULTRA = "ultra"

    @classmethod
    def parse(cls, value: str | "RunMode") -> "RunMode":
        if isinstance(value, cls):
            return value
        normalized = str(getattr(value, "value", value)).strip().casefold()
        normalized = {
            "chat": "normal", "goal": "normal", "manual": "normal",
            "default": "normal", "auto": "normal", "agent": "normal",
            "working": "normal", "work": "normal",
            "ultra-plan": "plan", "ultra_plan": "plan", "ultraplan": "plan",
            "deep": "ultra", "max": "ultra",
        }.get(normalized, normalized)
        try:
            return cls(normalized)
        except ValueError as exc:
            raise ValueError("mode must be 'plan', 'normal', or 'ultra'") from exc


class IntakeStatus(str, Enum):
    ANALYZING = "analyzing"
    AWAITING_ANSWERS = "awaiting_answers"
    READY = "ready"
    ROUTED = "routed"
    CANCELLED = "cancelled"


class PromptSlotStatus(str, Enum):
    EXPLICIT = "explicit"
    DISCOVERED = "discovered"
    SAFELY_INFERRED = "safely_inferred"
    MISSING_CONSEQUENTIAL = "missing_consequential"


@dataclass(frozen=True, slots=True)
class PromptDecisionSlotV1:
    name: str
    status: PromptSlotStatus
    value: str = ""
    provenance: str = ""

    @property
    def complete(self) -> bool:
        return self.status is not PromptSlotStatus.MISSING_CONSEQUENTIAL

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["status"] = self.status.value
        return value


@dataclass(frozen=True, slots=True)
class PromptCompletenessV1:
    slots: tuple[PromptDecisionSlotV1, ...]
    version: int = 2

    @property
    def complete(self) -> bool:
        return all(slot.complete for slot in self.slots)

    @property
    def missing_consequential(self) -> tuple[str, ...]:
        return tuple(slot.name for slot in self.slots if not slot.complete)

    def slot(self, name: str) -> PromptDecisionSlotV1:
        for slot in self.slots:
            if slot.name == name:
                return slot
        raise KeyError(name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "slots": [slot.to_dict() for slot in self.slots],
            "complete": self.complete,
            "missing_consequential": list(self.missing_consequential),
            "version": self.version,
        }


@dataclass(frozen=True, slots=True)
class QuestionOptionV1:
    value: str
    label: str
    description: str
    recommended: bool = False

    def __post_init__(self) -> None:
        if not self.value.strip() or not self.label.strip():
            raise ValueError("question options require value and label")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ClarificationQuestionV1:
    id: str
    header: str
    question: str
    options: tuple[QuestionOptionV1, ...]
    reason: str
    allow_freeform: bool = True
    decision_need: DecisionNeedV1 | None = None

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.question.strip():
            raise ValueError("clarification questions require id and question")
        if len(self.options) not in {2, 3}:
            raise ValueError("clarification questions require two or three suggested answers")
        if len({item.value for item in self.options}) != len(self.options):
            raise ValueError("question option values must be unique")
        recommended = [index for index, item in enumerate(self.options) if item.recommended]
        if len(recommended) > 1 or (recommended and recommended != [0]):
            raise ValueError("a sole recommended option must be presented first")
        if not self.allow_freeform:
            raise ValueError("clarification questions must allow a free-form answer")

    def to_dict(self) -> dict[str, Any]:
        value = {
            "id": self.id, "header": self.header, "question": self.question,
            "options": [item.to_dict() for item in self.options],
            "allow_freeform": True, "reason": self.reason,
        }
        if self.decision_need is not None:
            value["decision_need"] = self.decision_need.to_dict()
        return value


@dataclass(frozen=True, slots=True)
class TaskComplexityAssessmentV1:
    """Compatibility projection of the model's qualitative assessment."""

    score: float
    hard_triggers: tuple[str, ...] = ()
    component_count: int = 1
    reasons: tuple[str, ...] = ()
    breadth: str = "cohesive"
    coordination: str = "single"

    @property
    def ultra_required(self) -> bool:
        return bool(self.hard_triggers)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ExecutionBriefV1:
    original_input: str
    objective: str
    deliverables: tuple[str, ...]
    constraints: tuple[str, ...]
    success_criteria: tuple[str, ...]
    assumptions: tuple[str, ...]
    risks: tuple[str, ...]
    requested_mode: RunMode
    routed_mode: RunMode
    route_reason: str
    answers: Mapping[str, str] = field(default_factory=dict)
    exclusions: tuple[str, ...] = ()
    version: int = 2

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["requested_mode"] = self.requested_mode.value
        value["routed_mode"] = self.routed_mode.value
        value["answers"] = dict(self.answers)
        return value

    def canonical_prompt(self) -> str:
        return self.original_input


@dataclass(frozen=True, slots=True)
class IntakeDecisionV1:
    brief: ExecutionBriefV1
    complexity: TaskComplexityAssessmentV1
    completeness: PromptCompletenessV1
    questions: tuple[ClarificationQuestionV1, ...] = ()

    @property
    def status(self) -> IntakeStatus:
        return IntakeStatus.AWAITING_ANSWERS if self.questions else IntakeStatus.READY


def normalize_question(value: Mapping[str, Any], *, index: int = 1) -> ClarificationQuestionV1:
    """Canonicalize harmless question transport variants before validation.

    The configured model still owns every user-visible label.  Stable ids and
    option values are transport identifiers only, so the harness may derive
    them without inventing a product choice.  This deliberately accepts the
    compact string-option shape emitted by JSON-only local/cloud adapters.
    """

    question_text = str(value.get("question") or "").strip()[:1000]
    supplied_id = str(value.get("id") or "").strip()[:64]
    stable_id = supplied_id or (
        "decision_" + hashlib.sha256(question_text.encode("utf-8")).hexdigest()[:12]
        if question_text
        else ""
    )
    raw_options = value.get("options")
    if not isinstance(raw_options, Sequence) or isinstance(raw_options, (str, bytes)):
        raise ValueError(f"question {index} options must be an array")
    normalized_options: list[QuestionOptionV1] = []
    for option_index, item in enumerate(raw_options, start=1):
        if isinstance(item, Mapping):
            label = str(item.get("label") or item.get("value") or "").strip()
            description = str(item.get("description") or "").strip()
            recommended = bool(item.get("recommended", False))
            supplied_value = str(item.get("value") or "").strip()
        elif isinstance(item, str):
            label = item.strip()
            description = ""
            recommended = False
            supplied_value = ""
        else:
            raise ValueError(
                f"question {index} option {option_index} must be text or an object"
            )
        marker = "(recommended)"
        if label.casefold().endswith(marker):
            recommended = True
            label = label[: -len(marker)].rstrip(" -\u2013\u2014")
        if not label:
            raise ValueError(
                f"question {index} option {option_index} requires visible text"
            )
        opaque_value = supplied_value or (
            f"option_{option_index}_"
            + hashlib.sha256(
                f"{question_text}\0{option_index}\0{label}".encode("utf-8")
            ).hexdigest()[:10]
        )
        normalized_options.append(
            QuestionOptionV1(
                value=opaque_value[:100],
                label=label[:80],
                description=description[:500],
                recommended=recommended,
            )
        )
    options = tuple(normalized_options)
    recommended = [index for index, item in enumerate(options) if item.recommended]
    if len(recommended) == 1 and recommended[0] != 0:
        selected = options[recommended[0]]
        options = (selected,) + tuple(
            item for index, item in enumerate(options) if index != recommended[0]
        )
    elif len(recommended) > 1:
        # Conflicting recommendations must never become an automatic choice.
        options = tuple(replace(item, recommended=False) for item in options)
    raw_need = value.get("decision_need")
    decision_need = (
        DecisionNeedV1.from_mapping(raw_need)
        if isinstance(raw_need, Mapping)
        else None
    )
    return ClarificationQuestionV1(
        id=stable_id,
        header=str(value.get("header") or "").strip()[:40],
        question=question_text,
        options=options,
        reason=str(value.get("reason") or "").strip()[:1000],
        allow_freeform=bool(
            value.get(
                "allow_freeform",
                value.get("allow_free_form", value.get("allowFreeform", True)),
            )
        ),
        decision_need=decision_need,
    )


def normalize_questions(values: Sequence[Mapping[str, Any]]) -> tuple[ClarificationQuestionV1, ...]:
    if len(values) > 3:
        raise ValueError("intake supports at most three consequential questions")
    result = tuple(normalize_question(item, index=index) for index, item in enumerate(values, 1))
    if len({item.id for item in result}) != len(result):
        raise ValueError("intake question ids must be unique")
    return result


class IntentArchitect:
    """Validate a model-authored SemanticIntakeV2 and deterministic mode policy."""

    def validate(
        self,
        proposal: SemanticIntakeV2 | Mapping[str, Any],
        *,
        original_input: str,
        requested_mode: str | RunMode = RunMode.NORMAL,
        answers: Mapping[str, str] | None = None,
        repository_facts: Sequence[str] = (),
    ) -> IntakeDecisionV1:
        original = str(original_input)
        if not original.strip():
            raise ValueError("intent input must not be empty")
        semantic = proposal if isinstance(proposal, SemanticIntakeV2) else SemanticIntakeV2.from_mapping(proposal)
        requested = RunMode.parse(requested_mode)
        resolved_answers = {
            str(key): str(value).strip() for key, value in dict(answers or {}).items()
            if str(value).strip()
        }
        # Workflow mode is bound when the turn starts.  Intake may recommend a
        # future mode, but it cannot silently switch or pause the active turn.
        # Legacy execution_mode questions are ignored rather than creating an
        # unusable choice whose answer cannot be applied safely.
        questions = tuple(
            item
            for item in normalize_questions(semantic.questions)
            if item.id != "execution_mode"
            and (item.decision_need is None or item.decision_need.blocks_work)
        )
        routed = requested

        answered_ids = set(resolved_answers)
        active_questions = tuple(item for item in questions if item.id not in answered_ids)
        slots = tuple(
            PromptDecisionSlotV1(
                item.id,
                PromptSlotStatus.MISSING_CONSEQUENTIAL,
                "",
                "model_authored_consequential_question",
            )
            for item in active_questions
        ) or (
            PromptDecisionSlotV1(
                "semantic_intake", PromptSlotStatus.EXPLICIT,
                semantic.objective, "model_semantic_preflight",
            ),
        )
        completeness = PromptCompletenessV1(slots)
        maximum_demand = semantic.task_demand.maximum_level
        complexity = TaskComplexityAssessmentV1(
            score=maximum_demand / 4.0,
            hard_triggers=("high_model_authored_task_demand",) if maximum_demand >= 3 else (),
            component_count=semantic.component_count,
            reasons=semantic.task_demand.rationale,
            breadth=semantic.breadth,
            coordination=semantic.coordination,
        )
        route_reason = (
            "explicit Plan request" if requested is RunMode.PLAN else
            "legacy explicit recursive request" if requested is RunMode.ULTRA else
            "; ".join(semantic.task_demand.rationale)
        )
        brief = ExecutionBriefV1(
            original_input=original,
            objective=semantic.objective,
            deliverables=semantic.deliverables,
            constraints=semantic.constraints,
            success_criteria=semantic.acceptance_expectations,
            assumptions=tuple(dict.fromkeys((*semantic.assumptions, *(
                str(item).strip() for item in repository_facts if str(item).strip()
            )))),
            risks=semantic.risks,
            requested_mode=requested,
            routed_mode=routed,
            route_reason=route_reason,
            answers=resolved_answers,
            exclusions=semantic.exclusions,
        )
        return IntakeDecisionV1(brief, complexity, completeness, active_questions)

    def analyze(self, *args: Any, **kwargs: Any) -> IntakeDecisionV1:
        raise RuntimeError(
            "IntentArchitect no longer infers semantics from prompt text; use validate() "
            "with a model-authored SemanticIntakeV2"
        )


def answer_from_value(question: ClarificationQuestionV1, value: str) -> tuple[str, str]:
    raw = str(value).strip()
    if not raw:
        raise ValueError("question answers must not be empty")
    if raw in {"1", "2", "3"}:
        return question.options[int(raw) - 1].value, "suggested"
    for option in question.options:
        if raw.casefold() in {option.label.casefold(), option.value.casefold()}:
            return option.value, "suggested"
    if raw == "4":
        raise ValueError("choice 4 requires free-form text, for example: 4 your answer")
    if raw.startswith("4 "):
        raw = raw[2:].strip()
    return raw, "freeform"


__all__ = [
    "ClarificationQuestionV1", "ExecutionBriefV1", "IntakeDecisionV1",
    "IntakeStatus", "IntentArchitect", "PromptCompletenessV1",
    "PromptDecisionSlotV1", "PromptSlotStatus", "QuestionOptionV1", "RunMode",
    "TaskComplexityAssessmentV1", "answer_from_value", "normalize_question",
    "normalize_questions",
]
