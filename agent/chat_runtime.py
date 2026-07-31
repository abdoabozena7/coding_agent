"""Deterministic Chat action intent and completion requirements."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from typing import Iterable


_WRITE = re.compile(
    r"\b(save|write|create|edit|fix|patch|put|store|materiali[sz]e)\b|"
    r"(?:احفظ|اكتب|أنشئ|انشئ|عد[ّ]?ل|اصلح|أصلح|ضعه|حطه)", re.IGNORECASE,
)
_RUN = re.compile(
    r"\b(run|execute|launch|open|preview|serve|start|verify|test|check)\b|"
    r"(?:شغ[ّ]?ل|نف[ّ]?ذ|افتح|اعرض|ابدأ)", re.IGNORECASE,
)
_INSTALL = re.compile(
    r"\b(install|dependencies|dependency|packages?)\b|"
    r"(?:ثب[ّ]?ت|نز[ّ]?ل|مكتبات|اعتماديات)", re.IGNORECASE,
)
_QUESTION = re.compile(r"^\s*(how|why|what|when|where|who|explain|tell me|هل|لماذا|ليه|ما |ماذا|اشرح)", re.IGNORECASE)
_PROJECT = re.compile(
    r"\b(project|application|app|system|service|website|game|backend|frontend|"
    r"refactor|migration|end[- ]to[- ]end|multiple files?|full workflow|"
    r"polished|production[- ]ready)\b|"
    r"(?:مشروع|تطبيق|موقع|لعبة|نظام|واجهة|خادم|كامل|متكامل)",
    re.IGNORECASE,
)
_MULTISTEP = re.compile(
    r"\b(and then|then|after that|implement.*test|build.*verify|fix.*tests?)\b|"
    r"(?:وبعدين|ثم|وبعدها|نفذ.*اختبر|اعمل.*اختبر)",
    re.IGNORECASE,
)
_REPOSITORY_REFINEMENT = re.compile(
    r"\b(fix|repair|refactor|debug|resolve|improve|update)\b|"
    r"(?:أصلح|اصلح|صحح|عد[ّ]?ل|حس[ّ]?ن|حل)",
    re.IGNORECASE,
)
_FILE = re.compile(
    r"(?<![\w./-])[A-Za-z0-9_.-]+\."
    r"(?:html?|py|js|ts|tsx|jsx|css|json|md|txt|ya?ml|toml)\b",
    re.IGNORECASE,
)


class RouteKind(str, Enum):
    CHAT = "chat"
    ACTION = "action"
    GOAL = "goal"


@dataclass(frozen=True, slots=True)
class RouteDecisionV1:
    kind: RouteKind
    reason: str
    explicit: bool = False


@dataclass(frozen=True, slots=True)
class ChatIntentV1:
    text: str
    requires_write: bool = False
    requires_run: bool = False
    requires_install: bool = False

    @classmethod
    def parse(cls, text: str) -> "ChatIntentV1":
        value = str(text).strip()
        # An explanatory question mentioning "run" should remain ordinary Chat.
        explanatory = bool(_QUESTION.search(value)) and not re.search(r"\b(do it|run it|go ahead)\b|(?:اعمله|شغله|نفذه)", value, re.I)
        return cls(
            value,
            requires_write=bool(_WRITE.search(value)) and not explanatory,
            requires_run=bool(_RUN.search(value)) and not explanatory,
            requires_install=bool(_INSTALL.search(value)) and not explanatory,
        )

    @property
    def actionable(self) -> bool:
        return self.requires_write or self.requires_run or self.requires_install

    @property
    def required_categories(self) -> tuple[str, ...]:
        result = []
        if self.requires_write:
            result.append("write")
        if self.requires_install:
            result.append("install")
        if self.requires_run:
            result.append("run")
        return tuple(result)

    def authorizes(self, tool_name: str) -> bool:
        if self.requires_write and tool_name in {
            "write_file", "edit_file", "apply_patch", "materialize_artifact",
        }:
            return True
        if self.requires_install and tool_name == "install_dependencies":
            return True
        if self.requires_run and tool_name in {
            "preview_html", "inspect_preview", "stop_preview", "open_path", "start_process",
            "poll_process", "read_process_output", "stop_process",
        }:
            return True
        return False

    def missing(self, successful_tools: Iterable[str]) -> tuple[str, ...]:
        tools = set(successful_tools)
        missing = []
        if self.requires_write and not tools.intersection({
            "write_file", "edit_file", "apply_patch", "materialize_artifact",
        }):
            missing.append("write")
        if self.requires_install and not tools.intersection({"install_dependencies", "run_command", "run_bash"}):
            missing.append("install")
        if self.requires_run and not tools.intersection({
            "preview_html", "open_path", "start_process", "run_command", "run_bash",
        }):
            missing.append("run")
        return tuple(missing)


def route_input(text: str, *, explicit_goal: bool = False) -> RouteDecisionV1:
    """Separate conversation, bounded actions, and durable project work."""

    value = str(text).strip()
    if explicit_goal:
        return RouteDecisionV1(RouteKind.GOAL, "explicit goal command", True)
    intent = ChatIntentV1.parse(value)
    if not intent.actionable:
        if _PROJECT.search(value) or _MULTISTEP.search(value) or len(value) > 320:
            return RouteDecisionV1(
                RouteKind.GOAL, "project-scale or multi-step outcome"
            )
        return RouteDecisionV1(
            RouteKind.CHAT, "explanatory or conversational input"
        )
    files = tuple(dict.fromkeys(_FILE.findall(value)))
    if _REPOSITORY_REFINEMENT.search(value) and not files:
        return RouteDecisionV1(
            RouteKind.GOAL,
            "repository refinement needs inspection and durable verification",
        )
    if (
        len(value) <= 260
        and len(files) <= 1
        and not _PROJECT.search(value)
        and not _MULTISTEP.search(value)
    ):
        return RouteDecisionV1(
            RouteKind.ACTION, "single bounded explicit action"
        )
    return RouteDecisionV1(
        RouteKind.GOAL, "action requires a durable project workflow"
    )


def corrective_prompt(intent: ChatIntentV1, missing: tuple[str, ...], capabilities: str) -> str:
    return (
        "HARNESS ACTION REQUIREMENT: The user requested an executable action, but the prior "
        f"turn supplied no evidence for: {', '.join(missing)}. The available tools are real. "
        "Call the relevant tool now. For generated code use materialize_artifact; for HTML use "
        "preview_html. Do not tell the user to copy, save, install, or run it manually. A blocker "
        "is valid only after a concrete tool error, unavailable capability, or permission denial.\n"
        f"Capabilities: {capabilities}"
    )


__all__ = [
    "ChatIntentV1",
    "RouteDecisionV1",
    "RouteKind",
    "corrective_prompt",
    "route_input",
]
