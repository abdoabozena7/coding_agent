"""Evidence-backed output contracts for bounded workspace actions.

The contract is deliberately platform-neutral.  It tracks what the user asked
to receive (for example browser screenshots and independently copyable text),
then requires those artifacts to appear on the generic Output surface before a
weak model may claim the action is complete.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any, Iterable


_SCREENSHOT_RE = re.compile(
    r"(?:screen\s*shots?|screenshots?|سكرين\s*شوت(?:ات)?|لقط(?:ة|ات)\s*(?:شاشة)?)",
    re.IGNORECASE,
)
_COPY_RE = re.compile(
    r"(?:\bpost\b|\bcaption\b|\bcopy[ -]?ready\b|\bwrite\b.{0,36}\bcopy\b|"
    r"بوست|منشور|كابشن|جاهز\s*للنسخ)",
    re.IGNORECASE | re.DOTALL,
)
_NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "واحد": 1,
    "واحدة": 1,
    "اتنين": 2,
    "اثنين": 2,
    "اثنتين": 2,
    "تلات": 3,
    "ثلاث": 3,
    "ثلاثة": 3,
    "اربع": 4,
    "أربع": 4,
    "اربعة": 4,
    "أربعة": 4,
    "خمس": 5,
    "خمسة": 5,
}


def _requested_screenshot_count(text: str) -> int:
    source = str(text or "")
    match = _SCREENSHOT_RE.search(source)
    if match is None:
        return 0
    nearby = source[max(0, match.start() - 64) : min(len(source), match.end() + 64)]
    digit = re.search(r"(?<!\d)([1-8])(?!\d)", nearby)
    if digit is not None:
        return int(digit.group(1))
    for word in re.findall(r"[\w\u0600-\u06ff]+", nearby.casefold()):
        if word in _NUMBER_WORDS:
            return _NUMBER_WORDS[word]
    return 1


def _json_object(output: str) -> dict[str, Any]:
    try:
        value = json.loads(str(output))
    except (TypeError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, dict) else {}


def _normalised_paths(values: Iterable[Any]) -> list[str]:
    result: list[str] = []
    for value in values:
        raw = str(value or "").strip()
        if not raw:
            continue
        try:
            rendered = str(Path(raw).resolve()) if Path(raw).is_file() else raw.replace("\\", "/")
        except OSError:
            continue
        if rendered not in result:
            result.append(rendered)
    return result


def _same_path(left: str, right: str) -> bool:
    """Compare absolute and workspace-relative renderings of one artifact."""

    first = str(left or "").strip().replace("\\", "/").rstrip("/")
    second = str(right or "").strip().replace("\\", "/").rstrip("/")
    if not first or not second:
        return False
    first_folded = first.casefold()
    second_folded = second.casefold()
    return (
        first_folded == second_folded
        or first_folded.endswith("/" + second_folded)
        or second_folded.endswith("/" + first_folded)
    )


@dataclass(slots=True)
class ActionOutcomeContractV1:
    """Track exact deliverables independently from coarse semantic effects."""

    screenshot_count: int = 0
    require_copy: bool = False
    require_browser_open: bool = False
    require_visual_inspection: bool = False
    require_output: bool = True
    captured_images: list[str] = field(default_factory=list)
    inspected_images: list[str] = field(default_factory=list)
    selected_images: list[str] = field(default_factory=list)
    output_images: list[str] = field(default_factory=list)
    copy_sections: list[dict[str, str]] = field(default_factory=list)
    browser_opened: bool = False
    output_ready: bool = False
    output_id: str = ""
    output_title: str = ""
    captured_hashes: list[str] = field(default_factory=list)
    captured_perceptual_hashes: list[str] = field(default_factory=list)

    @staticmethod
    def _near_duplicate(candidate: str, existing: Iterable[str]) -> bool:
        if not candidate:
            return False
        try:
            raw = int(candidate, 16)
        except ValueError:
            return False
        for value in existing:
            if len(value) != len(candidate):
                continue
            try:
                distance = (raw ^ int(value, 16)).bit_count()
            except ValueError:
                continue
            if distance <= max(8, int(len(candidate) * 4 * 0.04)):
                return True
        return False

    @classmethod
    def from_request(
        cls,
        text: str,
        *,
        requested_effects: Iterable[Any] = (),
    ) -> "ActionOutcomeContractV1":
        count = _requested_screenshot_count(text)
        effects = {
            str(getattr(item, "value", item)).strip().casefold()
            for item in requested_effects
        }
        return cls(
            screenshot_count=count,
            require_copy=bool(_COPY_RE.search(str(text or ""))),
            require_browser_open="preview" in effects or count > 0,
            require_visual_inspection=count > 0,
            require_output=bool(effects) or count > 0 or bool(_COPY_RE.search(str(text or ""))),
        )

    @property
    def active(self) -> bool:
        return any(
            (
                self.screenshot_count,
                self.require_copy,
                self.require_browser_open,
                self.require_visual_inspection,
                self.require_output,
            )
        )

    def observe(self, tool_name: str, output: str) -> None:
        payload = _json_object(output)
        name = str(tool_name)
        if name in {"browser_open", "browser_inspect", "browser_act"} and payload:
            self.browser_opened = self.browser_opened or bool(payload.get("browser_opened"))
        elif name == "browser_screenshot" and payload:
            self.browser_opened = self.browser_opened or bool(payload.get("browser_opened"))
            digest = str(payload.get("sha256") or "").strip().casefold()
            if digest and digest in self.captured_hashes:
                return
            perceptual = str(payload.get("perceptual_hash") or "").strip().casefold()
            if self._near_duplicate(perceptual, self.captured_perceptual_hashes):
                return
            for path in _normalised_paths((payload.get("screenshot_path"), payload.get("workspace_path"))):
                if path not in self.captured_images:
                    self.captured_images.append(path)
                    if digest:
                        self.captured_hashes.append(digest)
                    if perceptual:
                        self.captured_perceptual_hashes.append(perceptual)
                    break
        elif name in {"preview_html", "preview_url", "inspect_preview"} and payload:
            # Compatibility evidence from sessions started before the general
            # browser controller was introduced.
            self.browser_opened = self.browser_opened or bool(payload.get("browser_opened"))
            images: list[Any] = [payload.get("screenshot_path")]
            for result in payload.get("interaction_results") or ():
                if isinstance(result, dict) and bool(result.get("passed")):
                    images.append(result.get("screenshot_path"))
            for path in _normalised_paths(images):
                if path not in self.captured_images:
                    self.captured_images.append(path)
        elif name == "inspect_images" and payload.get("status") == "evaluated":
            evaluated = [
                item.get("path")
                for item in payload.get("evaluations") or ()
                if isinstance(item, dict)
            ]
            for path in _normalised_paths(evaluated):
                if path not in self.inspected_images:
                    self.inspected_images.append(path)
            self.selected_images = _normalised_paths(payload.get("selected") or ())
        elif name == "publish_output" and payload.get("status") == "ready":
            self.output_ready = True
            self.output_id = str(payload.get("output_id") or "")
            self.output_title = str(payload.get("title") or "")
            self.copy_sections = [
                {"label": str(item.get("label") or "Copy ready"), "text": str(item.get("text") or "")}
                for item in payload.get("copy_sections") or ()
                if isinstance(item, dict) and str(item.get("text") or "").strip()
            ]
            images = [
                item.get("path")
                for item in payload.get("assets") or ()
                if isinstance(item, dict) and str(item.get("kind") or "") == "image"
            ]
            self.output_images = _normalised_paths(images)

    def restore_durable_screenshot(self, output: str) -> None:
        """Restore image bytes without claiming the old browser is still open."""

        payload = _json_object(output)
        if not payload:
            return
        digest = str(payload.get("sha256") or "").strip().casefold()
        if digest and digest in self.captured_hashes:
            return
        perceptual = str(payload.get("perceptual_hash") or "").strip().casefold()
        if self._near_duplicate(perceptual, self.captured_perceptual_hashes):
            return
        for path in _normalised_paths(
            (payload.get("screenshot_path"), payload.get("workspace_path"))
        ):
            if path not in self.captured_images:
                self.captured_images.append(path)
                if digest:
                    self.captured_hashes.append(digest)
                if perceptual:
                    self.captured_perceptual_hashes.append(perceptual)
                break

    def missing(self, *, include_output: bool = True) -> tuple[str, ...]:
        missing: list[str] = []
        if self.require_browser_open and not self.browser_opened:
            missing.append("open the project in a Playwright-controlled browser")
        if self.screenshot_count and len(self.captured_images) < self.screenshot_count:
            missing.append(
                f"capture {self.screenshot_count} distinct browser screenshots "
                f"({len(self.captured_images)} verified so far)"
            )
        if self.require_visual_inspection and len(self.inspected_images) < self.screenshot_count:
            missing.append(
                f"inspect the current bytes of all {self.screenshot_count} screenshots with vision"
            )
        if self.require_visual_inspection and not self.selected_images:
            missing.append("capture an image that the visual evaluator selects as acceptable")
        if include_output and self.require_output:
            if not self.output_ready:
                missing.append("publish the final result to the Output page")
            missing_selected = [
                path
                for path in self.selected_images
                if not any(_same_path(path, output) for output in self.output_images)
            ]
            unselected = [
                path
                for path in self.output_images
                if not any(_same_path(path, selected) for selected in self.selected_images)
            ]
            if missing_selected:
                missing.append(
                    f"attach all {len(self.selected_images)} vision-selected screenshots to Output"
                )
            if self.require_visual_inspection and unselected:
                missing.append("remove Output images that were not selected by visual evaluation")
            if self.require_copy and not self.copy_sections:
                missing.append("add the requested text as an independently copyable Output section")
        return tuple(dict.fromkeys(missing))

    def corrective_prompt(self, capabilities: str) -> str:
        missing = self.missing()
        suggestions: list[str] = []
        if any("Playwright" in item for item in missing):
            suggestions.append("Use browser_open with the running URL or workspace HTML path.")
        if any("capture" in item for item in missing):
            suggestions.append("Use browser_inspect/browser_act for distinct states, then browser_screenshot once per state.")
        if any("inspect" in item for item in missing):
            suggestions.append("Call inspect_images with every current screenshot path before selecting or describing them.")
        if any("Output" in item or "copyable" in item for item in missing):
            suggestions.append("Call publish_output with the final message, copy_sections, and the requested image assets.")
        return (
            "HARNESS OUTPUT GATE: The action is not complete. Missing: "
            + "; ".join(missing)
            + "\n"
            + " ".join(suggestions)
            + "\nAvailable capabilities: "
            + capabilities
        )

    def handoff_receipt(self) -> str:
        rows = [f"Output: {self.output_title or 'Task output'} ({self.output_id})"]
        if self.output_images:
            rows.append(f"Images attached: {len(self.output_images)}")
        if self.copy_sections:
            rows.append(f"Copy-ready sections: {len(self.copy_sections)}")
        return "\n".join(f"- {row}" for row in rows)
