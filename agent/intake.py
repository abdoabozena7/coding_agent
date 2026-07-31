"""Domain-neutral request preservation and execution-policy selection.

Intake keeps the user's text verbatim. Repository-grounded model planning and
an independent critic own all semantic interpretation after inspection.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import re
from typing import Any, Iterable, Mapping, Sequence


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


_VISUAL_TERMS = (
    "three" + ".js", "threejs", "webgl", "3d", "game", "لعبة", "visual",
    "animation", "interactive", "واجهة", "dashboard", "landing page", "تصميم",
)
_MIGRATION_TERMS = (
    "migration", "migrate", "database schema", "security", "auth", "permission",
    "production", "deploy", "ترحيل", "أمان", "صلاحيات", "نشر",
)
_MULTI_COMPONENT_TERMS = (
    "frontend", "backend", "api", "database", "worker", "agent", "agents",
    "swarm", "character", "vehicle", "road", "logic", "server", "client",
    "pipeline", "واجهة", "خادم", "قاعدة بيانات", "شخصية", "عربية", "طريق", "منطق",
)
_VAGUE_ONLY = re.compile(
    r"^(?:make|build|create|fix|improve|do|اعمل|سوي|سوّي|ظبط|اضبط|حسن|صلح)"
    r"(?:\s+(?:it|this|ده|دي|الحاجة))?$",
    re.IGNORECASE,
)
_REFINEMENT_TERMS = (
    "more advanced", "improve", "improved", "enhance", "upgrade", "polish",
    "make it better", "make this better", "طور", "تطوير", "حسن", "تحسين",
    "خليه أحسن",
)
_NEGATED_VISUAL_BUILD = re.compile(
    r"\b(?:do not|don't|never)\s+(?:build|create|make)\s+(?:a\s+)?"
    r"(?:game|dashboard|visual|animation)\b|"
    r"(?:لا|مات|ما\s*ت)\s*(?:تبني|تعمل|تنشئ)\s*(?:لعبة|واجهة)",
    re.IGNORECASE,
)


def _contains(text: str, terms: Iterable[str]) -> tuple[str, ...]:
    lowered = text.casefold()
    return tuple(term for term in terms if term.casefold() in lowered)


def _is_refinement_request(text: str) -> bool:
    return bool(_contains(str(text), _REFINEMENT_TERMS))


def _option(
    label: str, description: str, recommended: bool = False
) -> QuestionOptionV1:
    return QuestionOptionV1(
        label=label, description=description, recommended=recommended
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
    """Preserve request text while selecting safe clarification and execution policy."""

    def assess_complexity(self, prompt: str) -> TaskComplexityAssessmentV1:
        text = str(prompt).strip()
        if not text:
            raise ValueError("intent input must not be empty")
        meta_or_negated = bool(
            _NEGATED_VISUAL_BUILD.search(text)
            or re.search(r"\b(?:classifier|classification|naming)\b", text, re.I)
        )
        semantic_text = "" if meta_or_negated else text
        visual = _contains(semantic_text, _VISUAL_TERMS)
        migrations = _contains(text, _MIGRATION_TERMS)
        components = _contains(text, _MULTI_COMPONENT_TERMS)
        component_count = max(1, len(set(components)))
        hard: list[str] = []
        reasons: list[str] = []
        score = 0.12
        if len(text) >= 600:
            score += 0.18
            reasons.append("long multi-requirement prompt")
        if visual:
            score += 0.35
            reasons.append("visual or interactive quality")
            hard.append("visual_interactive_showcase")
        if migrations:
            score += 0.35
            reasons.append("high-risk migration/security/deployment")
            hard.append("high_risk_change")
        if component_count >= 3:
            score += 0.30
            reasons.append(f"{component_count} independently identifiable components")
            hard.append("multi_component_system")
        elif component_count == 2:
            score += 0.18
            reasons.append("multiple interacting subsystems")
        if any(
            term in text.casefold()
            for term in ("recursive", "multi-agent", "multi agent", "specialist", "debate")
        ):
            score += 0.25
            hard.append("recursive_specialization_benefit")
            reasons.append("recursive specialist execution requested")
        return TaskComplexityAssessmentV1(
            score=min(1.0, score),
            hard_triggers=tuple(dict.fromkeys(hard)),
            component_count=component_count,
            reasons=tuple(reasons) or ("cohesive bounded task",),
        )

    @staticmethod
    def _mentions(text: str, values: Iterable[str]) -> bool:
        lowered = text.casefold()
        return any(value.casefold() in lowered for value in values)

    def evaluate_completeness(
        self,
        prompt: str,
        *,
        answers: Mapping[str, str] | None = None,
        repository_facts: Sequence[str] = (),
    ) -> PromptCompletenessV1:
        text = str(prompt).strip()
        lowered = text.casefold()
        answers = {
            str(key).casefold(): str(value).strip()
            for key, value in dict(answers or {}).items()
            if str(value).strip()
        }
        facts = tuple(
            str(item).strip()
            for item in repository_facts
            if str(item).strip()
            and not str(item).strip().casefold().startswith(
                ("cross-run learned lesson:", "learned lesson:", "lesson:")
            )
        )
        facts_text = "\n".join(facts).casefold()
        refinement = _is_refinement_request(text) and bool(facts)
        meta_or_negated = bool(
            _NEGATED_VISUAL_BUILD.search(text)
            or re.search(r"\b(?:classifier|classification|naming)\b", lowered)
        )
        visual = bool(_contains(text, _VISUAL_TERMS)) and not meta_or_negated
        explicit_artifact = re.search(
            r"\b[\w.-]+\.(?:html?|py|js|ts|tsx|jsx|css|json|md|ya?ml|toml)\b",
            text,
            re.I,
        )
        slots: list[PromptDecisionSlotV1] = []

        def add(name: str, status: PromptSlotStatus, value: str, provenance: str) -> None:
            slots.append(PromptDecisionSlotV1(name, status, value, provenance))

        if text and not _VAGUE_ONLY.fullmatch(text):
            add("goal_output", PromptSlotStatus.EXPLICIT, text, "user_prompt")
        else:
            add(
                "goal_output", PromptSlotStatus.MISSING_CONSEQUENTIAL, "",
                "user_prompt_does_not_identify_an_outcome",
            )

        platform_terms = (
            "browser", "web", "desktop", "mobile", "android", "ios", "cli",
            "terminal", "windows", "linux", "macos", "متصفح", "موبايل", "ديسكتوب",
        )
        if "platform" in answers:
            add("platform_audience", PromptSlotStatus.EXPLICIT, answers["platform"], "intake_answer")
        elif self._mentions(lowered, platform_terms):
            add("platform_audience", PromptSlotStatus.EXPLICIT, "declared in prompt", "user_prompt")
        elif self._mentions(facts_text, platform_terms):
            add("platform_audience", PromptSlotStatus.DISCOVERED, "repository platform", "repository_facts")
        elif refinement:
            add("platform_audience", PromptSlotStatus.DISCOVERED, "preserve existing project platform", "repository_refinement_baseline")
        elif visual:
            add("platform_audience", PromptSlotStatus.MISSING_CONSEQUENTIAL, "", "visual input and layout depend on platform")
        else:
            add("platform_audience", PromptSlotStatus.SAFELY_INFERRED, "existing project platform", "repository-local default")

        packaging_terms = (
            "single html", "single-file", "single file", "one file",
            "self-contained", "multi-file", "multiple files", "modular",
            "package", "ملف واحد", "ملفات متعددة", "موديول",
        )
        if "packaging" in answers:
            add("packaging", PromptSlotStatus.EXPLICIT, answers["packaging"], "intake_answer")
        elif explicit_artifact and not explicit_artifact.group(0).casefold().endswith("three" + ".js"):
            add("packaging", PromptSlotStatus.EXPLICIT, explicit_artifact.group(0), "user_prompt")
        elif self._mentions(lowered, packaging_terms):
            add("packaging", PromptSlotStatus.EXPLICIT, "declared in prompt", "user_prompt")
        elif refinement:
            add("packaging", PromptSlotStatus.DISCOVERED, "preserve existing project packaging", "repository_refinement_baseline")
        elif visual:
            add("packaging", PromptSlotStatus.MISSING_CONSEQUENTIAL, "", "delivery shape affects integration")
        else:
            add("packaging", PromptSlotStatus.SAFELY_INFERRED, "follow repository conventions", "repository-local default")

        visual_terms = (
            "stylized", "realistic", "neon", "minimal", "material", "lighting",
            "pixel", "low-poly", "art direction", "ستايل", "واقعي", "كرتوني",
        )
        if "visual_direction" in answers:
            add("visual_direction", PromptSlotStatus.EXPLICIT, answers["visual_direction"], "intake_answer")
        elif refinement:
            add("visual_direction", PromptSlotStatus.DISCOVERED, "preserve existing visual direction", "repository_refinement_baseline")
        elif not visual:
            add("visual_direction", PromptSlotStatus.SAFELY_INFERRED, "not applicable", "non_visual_task")
        elif self._mentions(lowered, visual_terms):
            add("visual_direction", PromptSlotStatus.EXPLICIT, "declared in prompt", "user_prompt")
        else:
            add("visual_direction", PromptSlotStatus.MISSING_CONSEQUENTIAL, "", "visual quality requires an art direction")

        add(
            "constraints_acceptance", PromptSlotStatus.SAFELY_INFERRED,
            "functional, runtime, review, and regression gates", "harness_quality_floor",
        )
        add(
            "deployment_irreversible", PromptSlotStatus.SAFELY_INFERRED,
            "local artifact only; no deployment", "no irreversible side effect requested",
        )
        return PromptCompletenessV1(tuple(slots))

    def _questions(self, prompt: str) -> tuple[ClarificationQuestionV1, ...]:
        visual = bool(_contains(prompt, _VISUAL_TERMS)) and not bool(
            _NEGATED_VISUAL_BUILD.search(prompt)
        )
        if visual:
            values = (
                {
                    "id": "platform", "header": "Platform",
                    "question": "Where should the finished experience work best?",
                    "options": (
                        {"label": "Desktop browser", "description": "Optimize controls and performance for desktop browsers."},
                        {"label": "Mobile browser", "description": "Prioritize touch controls and smaller screens."},
                        {"label": "Desktop and mobile", "description": "Build responsive controls for both."},
                    ),
                    "reason": "Platform changes input, layout, and performance decisions.",
                },
                {
                    "id": "packaging", "header": "Packaging",
                    "question": "How should the finished experience be packaged?",
                    "options": (
                        {"label": "Modular staging, best final", "description": "Build components and assemble the strongest final package."},
                        {"label": "Single self-contained HTML", "description": "Deliver one portable HTML file."},
                        {"label": "Multi-file project", "description": "Deliver maintainable source modules and an entrypoint."},
                    ),
                    "reason": "Packaging changes integration and deployment.",
                },
                {
                    "id": "visual_direction", "header": "Visual style",
                    "question": "Which visual direction should guide implementation?",
                    "options": (
                        {"label": "Polished stylized", "description": "Use cohesive shapes, lighting, motion, and detail."},
                        {"label": "Realistic", "description": "Favor plausible proportions, materials, and lighting."},
                        {"label": "Arcade neon", "description": "Favor saturated color, speed effects, and dramatic feedback."},
                    ),
                    "reason": "A concrete art direction makes review objective.",
                },
            )
        else:
            values = (
                {
                    "id": "outcome", "header": "Outcome",
                    "question": "What kind of result should the agent produce?",
                    "options": (
                        {"label": "Complete implementation", "description": "Implement, test, review, and deliver the result."},
                        {"label": "Fix existing work", "description": "Inspect and repair the relevant existing artifact."},
                        {"label": "Analysis only", "description": "Investigate and report without changing files."},
                    ),
                    "reason": "The requested outcome is not explicit enough.",
                },
                {
                    "id": "priority", "header": "Priority",
                    "question": "Which priority should control tradeoffs?",
                    "options": (
                        {"label": "Highest quality", "description": "Use deeper verification and revision."},
                        {"label": "Balanced", "description": "Balance quality, scope, and time."},
                        {"label": "Fastest useful result", "description": "Prefer a narrow useful result."},
                    ),
                    "reason": "Priority determines review depth.",
                },
                {
                    "id": "scope", "header": "Scope",
                    "question": "How broadly may the agent change the project?",
                    "options": (
                        {"label": "Relevant files", "description": "Change every file needed for an integrated result."},
                        {"label": "Small focused change", "description": "Keep mutations narrowly bounded."},
                        {"label": "Broader refactor", "description": "Allow structural cleanup when justified."},
                    ),
                    "reason": "Scope affects ownership and risk.",
                },
            )
        return normalize_questions(values)

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
        complexity = self.assess_complexity(original)
        resolved_answers = {
            str(key): str(value)
            for key, value in dict(answers or {}).items()
            if str(value).strip()
        }
        mode_answer = resolved_answers.get("execution_mode", "").casefold()
        routed = (
            RunMode.PLAN if requested is RunMode.PLAN else
            RunMode.ULTRA if requested is RunMode.ULTRA or mode_answer.startswith("ultra") else
            RunMode.NORMAL
        )
        completeness = self.evaluate_completeness(
            original, answers=resolved_answers, repository_facts=repository_facts
        )
        meta_or_negated = bool(
            _NEGATED_VISUAL_BUILD.search(original)
            or re.search(r"\b(?:classifier|classification|naming)\b", original, re.I)
        )
        question_by_slot = {
            "platform_audience": "platform",
            "packaging": "packaging",
            "visual_direction": "visual_direction",
            "goal_output": "outcome",
        }
        selected = {
            question_by_slot[name]
            for name in completeness.missing_consequential
            if name in question_by_slot
        }
        if "goal_output" in completeness.missing_consequential:
            selected.update({"outcome", "priority", "scope"})
        questions = tuple(item for item in self._questions(original) if item.id in selected)[:3]
        if requested is RunMode.NORMAL and complexity.ultra_required and not mode_answer:
            escalation = ClarificationQuestionV1(
                id="execution_mode",
                header="Large project",
                question="This goal can benefit from Ultra specialists. How should it run?",
                options=(
                    QuestionOptionV1("Ultra mode", "Use recursive specialists and deeper integration checks.", recommended=True),
                    QuestionOptionV1("Normal mode", "Keep one durable lower-cost workflow."),
                    QuestionOptionV1("Edit request", "Pause so the request can be narrowed."),
                ),
                reason="Complexity crossed the Ultra recommendation threshold.",
            )
            questions = (escalation, *questions)[:3]
        if requested is RunMode.ULTRA and meta_or_negated:
            questions = ()
        refinement = _is_refinement_request(original) and bool(repository_facts)
        constraints = ["Preserve unrelated user work", "Use the real workspace and available tools"]
        success = [
            "Every explicit requirement is covered",
            "Critical functional checks pass",
            "Independent review finds no unresolved blocking issue",
        ]
        if refinement:
            constraints.extend(
                (
                    "Treat the current working project as the accepted baseline; improve it instead of rebuilding it",
                    "Preserve working behavior, public interfaces, packaging, and unrelated files",
                )
            )
            success.extend(
                (
                    "The candidate wins an evidence-backed comparison against the pre-change baseline",
                    "Previously working checks do not regress",
                )
            )
        brief = ExecutionBriefV1(
            original_input=original,
            objective=original,
            deliverables=(),
            constraints=tuple(constraints),
            success_criteria=tuple(success),
            assumptions=tuple(
                str(item).strip()
                for item in repository_facts
                if str(item).strip()
            ),
            risks=(),
            requested_mode=requested,
            routed_mode=routed,
            route_reason=(
                "explicit Ultra request"
                if requested is RunMode.ULTRA
                else "; ".join(complexity.reasons)
            ),
            answers=resolved_answers,
        )
        return IntakeDecisionV1(
            brief=brief,
            complexity=complexity,
            completeness=completeness,
            questions=questions,
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
