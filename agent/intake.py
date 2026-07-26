"""Domain-neutral request preservation and execution-policy selection.

Intake keeps the user's text verbatim. Repository-grounded model planning and
an independent critic own all semantic interpretation after inspection.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence


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
            "chat": "normal",
            "goal": "normal",
            "manual": "normal",
            "default": "normal",
            "auto": "normal",
            "agent": "normal",
            "deep": "ultra",
            "max": "ultra",
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
    version: int = 1

    @property
    def complete(self) -> bool:
        return all(slot.complete for slot in self.slots)

    @property
    def missing_consequential(self) -> tuple[str, ...]:
        return tuple(
            slot.name
            for slot in self.slots
            if slot.status is PromptSlotStatus.MISSING_CONSEQUENTIAL
        )

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
    label: str
    description: str
    recommended: bool = False

    def __post_init__(self) -> None:
        if not self.label.strip() or not self.description.strip():
            raise ValueError("question options require a label and description")

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
        if not self.id.strip() or not self.header.strip() or not self.question.strip():
            raise ValueError("clarification questions require id, header, and question")
        if len(self.options) != 3:
            raise ValueError("clarification questions require exactly three suggested answers")
        recommended = [
            index for index, option in enumerate(self.options) if option.recommended
        ]
        if recommended != [0]:
            raise ValueError("the first option must be the only recommended answer")
        if not self.allow_freeform:
            raise ValueError("clarification questions must allow a free-form fourth answer")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "header": self.header,
            "question": self.question,
            "options": [item.to_dict() for item in self.options],
            "allow_freeform": True,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class TaskComplexityAssessmentV1:
    score: float
    hard_triggers: tuple[str, ...] = ()
    component_count: int = 1
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "score", max(0.0, min(1.0, float(self.score))))
        object.__setattr__(self, "component_count", max(1, int(self.component_count)))

    @property
    def ultra_required(self) -> bool:
        return self.score >= 0.65 or bool(self.hard_triggers)

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
    version: int = 1

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["requested_mode"] = self.requested_mode.value
        value["routed_mode"] = self.routed_mode.value
        value["answers"] = dict(self.answers)
        return value

    def canonical_prompt(self) -> str:
        """Compatibility method that cannot add semantics to the request."""

        return self.original_input


@dataclass(frozen=True, slots=True)
class IntakeDecisionV1:
    brief: ExecutionBriefV1
    complexity: TaskComplexityAssessmentV1
    completeness: PromptCompletenessV1
    questions: tuple[ClarificationQuestionV1, ...] = ()

    @property
    def status(self) -> IntakeStatus:
        return (
            IntakeStatus.AWAITING_ANSWERS
            if self.questions
            else IntakeStatus.READY
        )


def normalize_question(
    value: Mapping[str, Any], *, index: int = 1
) -> ClarificationQuestionV1:
    """Normalize model-authored questions without inventing task semantics."""

    raw_options = [
        dict(item)
        for item in value.get("options", ())
        if isinstance(item, Mapping)
    ][:3]
    options: list[QuestionOptionV1] = []
    for position, item in enumerate(raw_options):
        options.append(
            QuestionOptionV1(
                label=str(item.get("label") or f"Option {position + 1}").strip()[:80],
                description=str(
                    item.get("description")
                    or "Use this user-selected planning decision."
                ).strip()[:500],
                recommended=position == 0,
            )
        )
    while len(options) < 3:
        position = len(options)
        options.append(
            QuestionOptionV1(
                label=f"Option {position + 1}",
                description="Use this user-selected planning decision.",
                recommended=position == 0,
            )
        )
    options = [
        QuestionOptionV1(
            item.label,
            item.description,
            recommended=position == 0,
        )
        for position, item in enumerate(options)
    ]
    return ClarificationQuestionV1(
        id=str(value.get("id") or f"Q{index}").strip()[:64],
        header=str(value.get("header") or "Decision").strip()[:40],
        question=str(
            value.get("question") or "Which direction should planning use?"
        ).strip()[:1000],
        options=tuple(options),
        reason=str(
            value.get("reason")
            or "The inspected plan requires a consequential user decision."
        ).strip()[:1000],
        allow_freeform=True,
    )


def normalize_questions(
    values: Sequence[Mapping[str, Any]],
) -> tuple[ClarificationQuestionV1, ...]:
    return tuple(
        normalize_question(item, index=index)
        for index, item in enumerate(values[:3], 1)
    )


class IntentArchitect:
    """Preserve request text and select only the explicit execution policy."""

    def assess_complexity(self, prompt: str) -> TaskComplexityAssessmentV1:
        if not str(prompt).strip():
            raise ValueError("intent input must not be empty")
        return TaskComplexityAssessmentV1(
            score=0.0,
            hard_triggers=(),
            component_count=1,
            reasons=("repository-grounded assessment pending",),
        )

    def evaluate_completeness(
        self,
        prompt: str,
        *,
        answers: Mapping[str, str] | None = None,
        repository_facts: Sequence[str] = (),
    ) -> PromptCompletenessV1:
        del answers, repository_facts
        original = str(prompt).strip()
        return PromptCompletenessV1(
            (
                PromptDecisionSlotV1(
                    "goal_output",
                    (
                        PromptSlotStatus.EXPLICIT
                        if original
                        else PromptSlotStatus.MISSING_CONSEQUENTIAL
                    ),
                    original,
                    "verbatim_user_request",
                ),
            )
        )

    def analyze(
        self,
        prompt: str,
        *,
        requested_mode: str | RunMode = RunMode.NORMAL,
        answers: Mapping[str, str] | None = None,
        repository_facts: Sequence[str] = (),
    ) -> IntakeDecisionV1:
        original = str(prompt).strip()
        if not original:
            raise ValueError("intent input must not be empty")
        requested = RunMode.parse(requested_mode)
        resolved_answers = {
            str(key): str(value)
            for key, value in dict(answers or {}).items()
            if str(value).strip()
        }
        brief = ExecutionBriefV1(
            original_input=original,
            objective=original,
            deliverables=(),
            constraints=(),
            success_criteria=(),
            assumptions=tuple(
                str(item).strip()
                for item in repository_facts
                if str(item).strip()
            ),
            risks=(),
            requested_mode=requested,
            routed_mode=requested,
            route_reason=(
                "explicit user mode; execution depth is policy, not task semantics"
            ),
            answers=resolved_answers,
        )
        return IntakeDecisionV1(
            brief=brief,
            complexity=self.assess_complexity(original),
            completeness=self.evaluate_completeness(original),
            questions=(),
        )


def answer_from_value(
    question: ClarificationQuestionV1, value: str
) -> tuple[str, str]:
    raw = str(value).strip()
    if not raw:
        raise ValueError("question answers must not be empty")
    if raw in {"1", "2", "3"}:
        return question.options[int(raw) - 1].label, "suggested"
    for option in question.options:
        if raw.casefold() == option.label.casefold():
            return option.label, "suggested"
    if raw == "4":
        raise ValueError(
            "choice 4 requires free-form text, for example: 4 your answer"
        )
    if raw.startswith("4 "):
        raw = raw[2:].strip()
    return raw, "freeform"


__all__ = [
    "ClarificationQuestionV1",
    "ExecutionBriefV1",
    "IntakeDecisionV1",
    "IntakeStatus",
    "IntentArchitect",
    "PromptCompletenessV1",
    "PromptDecisionSlotV1",
    "PromptSlotStatus",
    "QuestionOptionV1",
    "RunMode",
    "TaskComplexityAssessmentV1",
    "answer_from_value",
    "normalize_question",
    "normalize_questions",
]
