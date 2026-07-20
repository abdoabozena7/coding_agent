"""Harness-owned intent intake, clarification, and Normal/Ultra routing.

The model may enrich an execution brief later, but the public interaction
contract and routing floor are deterministic.  This keeps short or ambiguous
requests from falling through to an unstructured chat turn.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import re
from typing import Any, Iterable, Mapping, Sequence


class RunMode(str, Enum):
    NORMAL = "normal"
    ULTRA = "ultra"

    @classmethod
    def parse(cls, value: str | "RunMode") -> "RunMode":
        if isinstance(value, cls):
            return value
        normalized = str(getattr(value, "value", value)).strip().casefold()
        normalized = {
            "chat": "normal",
            "plan": "normal",
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
            raise ValueError("mode must be 'normal' or 'ultra'") from exc


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
        recommended = [index for index, option in enumerate(self.options) if option.recommended]
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
        sections = [
            self.objective.strip(),
            "\nCANONICAL EXECUTION BRIEF:",
            "Deliverables:\n- " + "\n- ".join(self.deliverables),
            "Constraints:\n- " + "\n- ".join(self.constraints),
            "Success criteria:\n- " + "\n- ".join(self.success_criteria),
            "Assumptions:\n- " + "\n- ".join(self.assumptions),
            "Risks:\n- " + "\n- ".join(self.risks),
            f"Harness route: {self.routed_mode.value} ({self.route_reason})",
        ]
        if self.answers:
            sections.append(
                "User decisions:\n- "
                + "\n- ".join(f"{key}: {value}" for key, value in sorted(self.answers.items()))
            )
        return "\n".join(section for section in sections if section.strip())


@dataclass(frozen=True, slots=True)
class IntakeDecisionV1:
    brief: ExecutionBriefV1
    complexity: TaskComplexityAssessmentV1
    completeness: PromptCompletenessV1
    questions: tuple[ClarificationQuestionV1, ...] = ()

    @property
    def status(self) -> IntakeStatus:
        return IntakeStatus.AWAITING_ANSWERS if self.questions else IntakeStatus.READY


_VISUAL_TERMS = (
    "three.js", "threejs", "webgl", "3d", "game", "لعبة", "visual", "animation",
    "interactive", "واجهة", "dashboard", "landing page", "تصميم",
)
_MIGRATION_TERMS = (
    "migration", "migrate", "database schema", "security", "auth", "permission",
    "production", "deploy", "ترحيل", "أمان", "صلاحيات", "نشر",
)
_MULTI_COMPONENT_TERMS = (
    "frontend", "backend", "api", "database", "worker", "agent", "agents", "swarm",
    "character", "vehicle", "road", "logic", "server", "client", "pipeline",
    "واجهة", "خادم", "قاعدة بيانات", "شخصية", "عربية", "طريق", "منطق",
)
_VAGUE_ONLY = re.compile(
    r"^(?:make|build|create|fix|improve|do|اعمل|سوي|سوّي|ظبط|اضبط|حسن|صلح)(?:\s+(?:it|this|ده|دي|الحاجة))?$",
    re.IGNORECASE,
)


def _contains(text: str, terms: Iterable[str]) -> tuple[str, ...]:
    lowered = text.casefold()
    return tuple(term for term in terms if term.casefold() in lowered)


def _option(label: str, description: str, recommended: bool = False) -> QuestionOptionV1:
    return QuestionOptionV1(label=label, description=description, recommended=recommended)


def normalize_question(value: Mapping[str, Any], *, index: int = 1) -> ClarificationQuestionV1:
    """Repair weak-model question shapes into the strict four-way UX contract."""

    raw_options = [dict(item) for item in value.get("options", ()) if isinstance(item, Mapping)]
    options: list[QuestionOptionV1] = []
    for position, item in enumerate(raw_options[:3]):
        label = str(item.get("label") or f"Option {position + 1}").strip()[:80]
        description = str(item.get("description") or "Use this direction for the execution brief.").strip()[:500]
        options.append(_option(label, description, recommended=position == 0))
    fallbacks = (
        _option("Best quality", "Let the agent choose the strongest quality-first direction."),
        _option("Balanced scope", "Keep the scope complete while controlling unnecessary complexity."),
        _option("Focused result", "Prioritize the smallest polished result that proves the goal."),
    )
    used = {item.label.casefold() for item in options}
    for candidate in fallbacks:
        if len(options) >= 3:
            break
        if candidate.label.casefold() not in used:
            options.append(candidate)
            used.add(candidate.label.casefold())
    options = [
        QuestionOptionV1(item.label, item.description, recommended=position == 0)
        for position, item in enumerate(options[:3])
    ]
    return ClarificationQuestionV1(
        id=str(value.get("id") or f"Q{index}").strip()[:64],
        header=str(value.get("header") or "Decision").strip()[:40],
        question=str(value.get("question") or "Which direction should the agent use?").strip()[:1000],
        options=tuple(options),
        allow_freeform=True,
        reason=str(value.get("reason") or "Required to finalize the execution brief.").strip()[:1000],
    )


def normalize_questions(values: Sequence[Mapping[str, Any]]) -> tuple[ClarificationQuestionV1, ...]:
    return tuple(normalize_question(item, index=index) for index, item in enumerate(values[:3], 1))


class IntentArchitect:
    """Create a durable brief and choose the minimum quality-preserving mode."""

    def assess_complexity(self, prompt: str) -> TaskComplexityAssessmentV1:
        text = str(prompt).strip()
        visual = _contains(text, _VISUAL_TERMS)
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
        if any(term in text.casefold() for term in ("recursive", "multi-agent", "multi agent", "specialist", "debate")):
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
        """Classify consequential decisions semantically, never by prompt length."""

        text = str(prompt).strip()
        lowered = text.casefold()
        answers = {
            str(key).casefold(): str(value).strip()
            for key, value in dict(answers or {}).items()
            if str(value).strip()
        }
        discoverable_facts = tuple(
            str(item).strip()
            for item in repository_facts
            if str(item).strip()
            and not str(item).strip().casefold().startswith(
                ("cross-run learned lesson:", "learned lesson:", "lesson:")
            )
        )
        facts_text = "\n".join(discoverable_facts).casefold()
        visual = bool(_contains(text, _VISUAL_TERMS))
        explicit_artifact = re.search(
            r"\b[\w.-]+\.(?:html?|py|js|ts|tsx|jsx|css|json|md|ya?ml|toml)\b",
            text,
            re.I,
        )
        slots: list[PromptDecisionSlotV1] = []

        def add(
            name: str,
            status: PromptSlotStatus,
            value: str,
            provenance: str,
        ) -> None:
            slots.append(PromptDecisionSlotV1(name, status, value, provenance))

        if text and not _VAGUE_ONLY.fullmatch(text):
            add("goal_output", PromptSlotStatus.EXPLICIT, text, "user_prompt")
        else:
            add(
                "goal_output",
                PromptSlotStatus.MISSING_CONSEQUENTIAL,
                "",
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
        elif visual:
            add(
                "platform_audience",
                PromptSlotStatus.MISSING_CONSEQUENTIAL,
                "",
                "input_controls_and_layout_depend_on_platform",
            )
        else:
            add(
                "platform_audience",
                PromptSlotStatus.SAFELY_INFERRED,
                "existing project platform",
                "reversible repository-local default",
            )

        packaging_terms = (
            "single html", "single-file", "single file", "one file", "self-contained",
            "multi-file", "multiple files", "modular", "package", "ملف واحد",
            "ملفات متعددة", "موديول",
        )
        if "packaging" in answers:
            add("packaging", PromptSlotStatus.EXPLICIT, answers["packaging"], "intake_answer")
        elif (
            explicit_artifact
            and not str(explicit_artifact.group(0)).casefold().endswith("three.js")
        ):
            add("packaging", PromptSlotStatus.EXPLICIT, explicit_artifact.group(0), "user_prompt")
        elif self._mentions(lowered, packaging_terms):
            add("packaging", PromptSlotStatus.EXPLICIT, "declared in prompt", "user_prompt")
        elif visual:
            add(
                "packaging",
                PromptSlotStatus.MISSING_CONSEQUENTIAL,
                "",
                "final_delivery_shape_affects_assembly_and_deployment",
            )
        else:
            add(
                "packaging",
                PromptSlotStatus.SAFELY_INFERRED,
                "follow repository conventions",
                "reversible repository-local default",
            )

        visual_terms = (
            "stylized", "realistic", "neon", "minimal", "material", "lighting",
            "pixel", "low-poly", "art direction", "ستايل", "واقعي", "كرتوني",
        )
        if "visual_direction" in answers:
            add(
                "visual_direction",
                PromptSlotStatus.EXPLICIT,
                answers["visual_direction"],
                "intake_answer",
            )
        elif not visual:
            add(
                "visual_direction",
                PromptSlotStatus.SAFELY_INFERRED,
                "not applicable",
                "non_visual_task",
            )
        elif self._mentions(lowered, visual_terms):
            add("visual_direction", PromptSlotStatus.EXPLICIT, "declared in prompt", "user_prompt")
        else:
            add(
                "visual_direction",
                PromptSlotStatus.MISSING_CONSEQUENTIAL,
                "",
                "visual_quality_requires_a_reviewable_art_direction",
            )

        constraint_terms = (
            "must", "should", "accept", "test", "performance", "compatible",
            "without", "no ", "لازم", "اختبار", "أداء", "متوافق", "بدون",
        )
        if self._mentions(lowered, constraint_terms):
            add("constraints_acceptance", PromptSlotStatus.EXPLICIT, "declared in prompt", "user_prompt")
        else:
            add(
                "constraints_acceptance",
                PromptSlotStatus.SAFELY_INFERRED,
                "functional, runtime, review, and regression gates",
                "harness_quality_floor",
            )

        deployment_terms = ("deploy", "production", "publish", "hosting", "نشر", "إنتاج")
        if self._mentions(lowered, deployment_terms):
            add("deployment_irreversible", PromptSlotStatus.EXPLICIT, "declared in prompt", "user_prompt")
        else:
            add(
                "deployment_irreversible",
                PromptSlotStatus.SAFELY_INFERRED,
                "local artifact only; no deployment",
                "no irreversible side effect requested",
            )
        return PromptCompletenessV1(tuple(slots))

    @staticmethod
    def _needs_questions(completeness: PromptCompletenessV1) -> bool:
        return not completeness.complete

    def _questions(self, prompt: str) -> tuple[ClarificationQuestionV1, ...]:
        visual = bool(_contains(prompt, _VISUAL_TERMS))
        if visual:
            values = (
                {
                    "id": "platform",
                    "header": "Platform",
                    "question": "Where should the finished experience work best?",
                    "options": (
                        {"label": "Desktop browser", "description": "Optimize controls, framing, and performance for desktop browsers."},
                        {"label": "Mobile browser", "description": "Prioritize touch controls and smaller screens."},
                        {"label": "Desktop and mobile", "description": "Build responsive controls and layouts for both."},
                    ),
                    "reason": "Platform changes input, layout, and performance decisions.",
                },
                {
                    "id": "packaging",
                    "header": "Packaging",
                    "question": "How should the finished experience be packaged?",
                    "options": (
                        {"label": "Modular staging, best final", "description": "Build isolated components and let the assembler choose the strongest final packaging."},
                        {"label": "Single self-contained HTML", "description": "Deliver one portable HTML file with runtime code and assets embedded."},
                        {"label": "Multi-file project", "description": "Deliver maintainable source modules and a browser entrypoint."},
                    ),
                    "reason": "Packaging changes component contracts, integration, and deployment.",
                },
                {
                    "id": "visual_direction",
                    "header": "Visual style",
                    "question": "Which visual direction should guide the specialists?",
                    "options": (
                        {"label": "Polished stylized", "description": "Use cohesive shapes, lighting, motion, and readable detail."},
                        {"label": "Realistic", "description": "Favor physically plausible proportions, materials, and lighting."},
                        {"label": "Arcade neon", "description": "Favor saturated color, speed effects, and dramatic feedback."},
                    ),
                    "reason": "A concrete art direction makes component reviews objective.",
                },
            )
        else:
            values = (
                {
                    "id": "outcome",
                    "header": "Outcome",
                    "question": "What kind of result should the agent produce?",
                    "options": (
                        {"label": "Complete implementation", "description": "Implement, test, review, and deliver the finished result."},
                        {"label": "Fix existing work", "description": "Inspect the workspace and repair the most relevant existing artifact."},
                        {"label": "Analysis only", "description": "Investigate and report without changing files."},
                    ),
                    "reason": "The requested outcome is not explicit enough to authorize execution.",
                },
                {
                    "id": "priority",
                    "header": "Priority",
                    "question": "Which priority should control tradeoffs?",
                    "options": (
                        {"label": "Highest quality", "description": "Use deeper verification and revision even if execution takes longer."},
                        {"label": "Balanced", "description": "Balance quality, scope, and execution time."},
                        {"label": "Fastest useful result", "description": "Prefer a narrow result with essential checks."},
                    ),
                    "reason": "Priority determines review depth and stopping criteria.",
                },
                {
                    "id": "scope",
                    "header": "Scope",
                    "question": "How broadly may the agent change the project?",
                    "options": (
                        {"label": "Relevant files", "description": "Change every file needed for a complete, integrated result."},
                        {"label": "Small focused change", "description": "Keep mutations narrowly bounded to the immediate issue."},
                        {"label": "Broader refactor", "description": "Allow structural cleanup when it improves the result."},
                    ),
                    "reason": "Scope affects ownership, risk, and the execution plan.",
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
        routed = RunMode.ULTRA if requested is RunMode.ULTRA or complexity.ultra_required else RunMode.NORMAL
        route_reason = (
            "explicit Ultra request"
            if requested is RunMode.ULTRA
            else "; ".join(complexity.reasons)
        )
        resolved_answers = {
            str(key): str(value)
            for key, value in dict(answers or {}).items()
            if str(value).strip()
        }
        completeness = self.evaluate_completeness(
            original,
            answers=resolved_answers,
            repository_facts=repository_facts,
        )
        question_by_slot = {
            "platform_audience": "platform",
            "packaging": "packaging",
            "visual_direction": "visual_direction",
            "goal_output": "outcome",
        }
        selected_question_ids = {
            question_by_slot[name]
            for name in completeness.missing_consequential
            if name in question_by_slot
        }
        if "goal_output" in completeness.missing_consequential:
            selected_question_ids.update({"outcome", "priority", "scope"})
        questions = tuple(
            item
            for item in self._questions(original)
            if item.id in selected_question_ids
        )[:3]
        visual_experience = bool(_contains(original, _VISUAL_TERMS))
        packaging = resolved_answers.get("packaging", "")
        single_html = (
            "single" in packaging.casefold()
            or "ملف واحد" in packaging
            or bool(
                re.search(
                    r"\b(?:single[- ]file|one file|self-contained)\b",
                    original,
                    re.I,
                )
            )
        )
        deliverable = (
            "A self-contained, playable index.html with integrated Three.js/runtime code and no split source output"
            if visual_experience and single_html
            else (
                "Materialized specialist components plus the strongest integrated browser packaging"
                if visual_experience
                else (
                    "A complete, integrated implementation with executable verification"
                    if "analysis only" not in " ".join(resolved_answers.values()).casefold()
                    else "An evidence-backed analysis without workspace mutation"
                )
            )
        )
        constraints = ["Preserve unrelated user work", "Use the real workspace and available tools"]
        if visual_experience:
            constraints.extend(
                (
                    "FinalAssembler is the only writer of final output paths",
                    "Specialists publish materialized component packages and cannot replace the final artifact",
                )
            )
        success_criteria = [
            "Every explicit requirement is covered",
            "Critical functional checks pass",
            "Independent review finds no unresolved blocking issue",
        ]
        if visual_experience:
            success_criteria.extend(
                (
                    "The game is playable with zero browser console or WebGL runtime errors",
                    "Overall quality is at least 0.95 and every critical visual category is at least 0.90",
                )
            )
        brief = ExecutionBriefV1(
            original_input=original,
            objective=original,
            deliverables=(deliverable,),
            constraints=tuple(constraints),
            success_criteria=tuple(success_criteria),
            assumptions=(
                "Reversible technical choices use the strongest safe default",
                "Quality is preferred over execution speed",
                *tuple(str(item).strip() for item in repository_facts if str(item).strip()),
            ),
            risks=("Ambiguous requirements are resolved before mutation",),
            requested_mode=requested,
            routed_mode=routed,
            route_reason=route_reason,
            answers=resolved_answers,
        )
        return IntakeDecisionV1(
            brief=brief,
            complexity=complexity,
            completeness=completeness,
            questions=questions,
        )


def answer_from_value(question: ClarificationQuestionV1, value: str) -> tuple[str, str]:
    """Resolve numeric/label/free-form input and return (answer, source)."""

    raw = str(value).strip()
    if not raw:
        raise ValueError("question answers must not be empty")
    if raw in {"1", "2", "3"}:
        return question.options[int(raw) - 1].label, "suggested"
    for option in question.options:
        if raw.casefold() == option.label.casefold():
            return option.label, "suggested"
    if raw == "4":
        raise ValueError("choice 4 requires free-form text, for example: 4 your answer")
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
