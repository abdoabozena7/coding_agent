"""Validation for model-authored Goal intake.

Semantic interpretation belongs to the configured model.  The deterministic
code here preserves the exact request, validates structure, and applies mode
policy without deriving complexity or ambiguity from words in the request.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

from .chat_runtime import SemanticIntakeV2


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
        if not self.value.strip() or not self.label.strip() or not self.description.strip():
            raise ValueError("question options require value, label, and description")

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

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.header.strip() or not self.question.strip() or not self.reason.strip():
            raise ValueError("clarification questions require id, header, question, and reason")
        if len(self.options) != 3:
            raise ValueError("clarification questions require exactly three suggested answers")
        if len({item.value for item in self.options}) != 3:
            raise ValueError("question option values must be unique")
        if [index for index, item in enumerate(self.options) if item.recommended] != [0]:
            raise ValueError("the first option must be the only recommended answer")
        if not self.allow_freeform:
            raise ValueError("clarification questions must allow a free-form fourth answer")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "header": self.header, "question": self.question,
            "options": [item.to_dict() for item in self.options],
            "allow_freeform": True, "reason": self.reason,
        }


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
    """Strictly validate model-authored content; never invent visible text."""

    raw_options = value.get("options")
    if not isinstance(raw_options, Sequence) or isinstance(raw_options, (str, bytes)):
        raise ValueError(f"question {index} options must be an array")
    options = tuple(
        QuestionOptionV1(
            value=str(item.get("value") or "").strip()[:100],
            label=str(item.get("label") or "").strip()[:80],
            description=str(item.get("description") or "").strip()[:500],
            recommended=bool(item.get("recommended", False)),
        )
        for item in raw_options
        if isinstance(item, Mapping)
    )
    return ClarificationQuestionV1(
        id=str(value.get("id") or "").strip()[:64],
        header=str(value.get("header") or "").strip()[:40],
        question=str(value.get("question") or "").strip()[:1000],
        options=options,
        reason=str(value.get("reason") or "").strip()[:1000],
        allow_freeform=bool(value.get("allow_freeform", True)),
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
        questions = normalize_questions(semantic.questions)
        mode_answer = resolved_answers.get("execution_mode", "").casefold()
        if mode_answer == "edit_request":
            routed = RunMode.NORMAL
        elif requested is RunMode.PLAN:
            routed = RunMode.PLAN
        elif requested is RunMode.ULTRA:
            routed = RunMode.ULTRA
        elif mode_answer == "ultra":
            routed = RunMode.ULTRA
        else:
            routed = RunMode.NORMAL

        if semantic.recommended_mode == "ultra" and requested is RunMode.NORMAL and not mode_answer:
            mode_questions = [item for item in questions if item.id == "execution_mode"]
            if len(mode_questions) != 1:
                raise ValueError("an Ultra recommendation requires one execution_mode question")
            if tuple(item.value for item in mode_questions[0].options) != ("ultra", "normal", "edit_request"):
                raise ValueError("execution_mode option values must be ultra, normal, edit_request")

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
        complexity = TaskComplexityAssessmentV1(
            score=0.75 if semantic.recommended_mode == "ultra" else 0.35,
            hard_triggers=("model_recommends_ultra",) if semantic.recommended_mode == "ultra" else (),
            component_count=2 if semantic.breadth == "multi_component" else 1,
            reasons=semantic.complexity_reasons or (semantic.recommendation_reason,),
            breadth=semantic.breadth,
            coordination=semantic.coordination,
        )
        route_reason = (
            "explicit Plan request" if requested is RunMode.PLAN else
            "explicit Ultra request" if requested is RunMode.ULTRA else
            semantic.recommendation_reason
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
