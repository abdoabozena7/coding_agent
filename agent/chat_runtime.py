"""Deterministic Chat action intent and completion requirements."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable


_WRITE = re.compile(
    r"\b(save|write|create|edit|fix|patch|put|store|materiali[sz]e)\b|"
    r"(?:احفظ|اكتب|أنشئ|انشئ|عد[ّ]?ل|اصلح|أصلح|ضعه|حطه)", re.IGNORECASE,
)
_RUN = re.compile(
    r"\b(run|execute|launch|open|preview|serve|start)\b|"
    r"(?:شغ[ّ]?ل|نف[ّ]?ذ|افتح|اعرض|ابدأ)", re.IGNORECASE,
)
_INSTALL = re.compile(
    r"\b(install|dependencies|dependency|packages?)\b|"
    r"(?:ثب[ّ]?ت|نز[ّ]?ل|مكتبات|اعتماديات)", re.IGNORECASE,
)
_QUESTION = re.compile(r"^\s*(how|why|what|when|where|who|explain|tell me|هل|لماذا|ليه|ما |ماذا|اشرح)", re.IGNORECASE)


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


def corrective_prompt(intent: ChatIntentV1, missing: tuple[str, ...], capabilities: str) -> str:
    return (
        "HARNESS ACTION REQUIREMENT: The user requested an executable action, but the prior "
        f"turn supplied no evidence for: {', '.join(missing)}. The available tools are real. "
        "Call the relevant tool now. For generated code use materialize_artifact; for HTML use "
        "preview_html. Do not tell the user to copy, save, install, or run it manually. A blocker "
        "is valid only after a concrete tool error, unavailable capability, or permission denial.\n"
        f"Capabilities: {capabilities}"
    )


__all__ = ["ChatIntentV1", "corrective_prompt"]
