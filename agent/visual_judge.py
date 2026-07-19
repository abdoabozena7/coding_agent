"""Independent, screenshot-grounded visual evaluation for Ultra.

Pixel statistics are useful only for detecting broken/blank renders. Acceptance
comes from a vision-capable model in clean contexts plus a blind comparison.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import base64
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any, Mapping, Protocol, Sequence
from urllib import request


class VisualJudgeUnavailable(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class VisualFindingV1:
    severity: str
    category: str
    message: str
    evidence: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class VisualJudgeVerdictV1:
    evaluator: str
    model: str
    accepted: bool
    scores: Mapping[str, float]
    findings: tuple[VisualFindingV1, ...]
    summary: str
    confidence: float
    screenshot_hash: str
    context_fingerprint: str
    status: str = "evaluated"
    version: int = 1

    @property
    def critical_findings(self) -> tuple[VisualFindingV1, ...]:
        return tuple(item for item in self.findings if item.severity == "critical")

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "scores": dict(self.scores),
            "findings": [item.to_dict() for item in self.findings],
            "critical_findings": len(self.critical_findings),
        }


@dataclass(frozen=True, slots=True)
class PairwiseVisualComparisonV1:
    evaluator: str
    model: str
    preferred: str
    confidence: float
    rationale: str
    candidate_hash: str
    baseline_hash: str
    context_fingerprint: str
    version: int = 1

    @property
    def candidate_preferred(self) -> bool:
        return self.preferred == "candidate"

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "candidate_preferred": self.candidate_preferred}


class VisionJudge(Protocol):
    evaluator: str
    model: str

    def evaluate(
        self,
        *,
        brief: str,
        rubric: Mapping[str, Any],
        screenshot: str | Path,
        runtime_evidence: Mapping[str, Any],
        clean_context_nonce: str,
    ) -> VisualJudgeVerdictV1: ...

    def compare(
        self,
        *,
        brief: str,
        rubric: Mapping[str, Any],
        candidate: str | Path,
        baseline: str | Path,
        clean_context_nonce: str,
    ) -> PairwiseVisualComparisonV1: ...


def _hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _clamp(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def screenshot_anomalies(path: str | Path) -> tuple[str, ...]:
    """Detect only obvious empty/broken renders; never award visual quality."""

    try:
        from PIL import Image

        image = Image.open(path).convert("RGB")
        image.thumbnail((192, 108))
        quantized = image.quantize(colors=16)
        counts = quantized.getcolors(maxcolors=16) or []
        total = max(1, image.width * image.height)
        dominant = max((count for count, _color in counts), default=total)
        dominant_fraction = dominant / total
    except Exception:
        return ()
    findings: list[str] = []
    if dominant_fraction >= 0.90:
        findings.append(
            f"screenshot is visually near-empty: one color occupies {dominant_fraction:.1%}"
        )
    return tuple(findings)


def _json_object(raw: str) -> Mapping[str, Any]:
    text = str(raw).strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.casefold().startswith("json"):
            text = text[4:].lstrip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise VisualJudgeUnavailable("vision judge returned non-JSON output") from exc
        try:
            value = json.loads(text[start : end + 1])
        except json.JSONDecodeError as nested:
            raise VisualJudgeUnavailable("vision judge returned malformed JSON") from nested
    if not isinstance(value, Mapping):
        raise VisualJudgeUnavailable("vision judge response must be an object")
    return value


class OllamaVisionJudge:
    evaluator = "ollama_vision"

    def __init__(
        self,
        model: str,
        *,
        host: str = "http://127.0.0.1:11434",
        timeout_seconds: int = 180,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._assert_vision()

    def _post(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = request.Request(
            f"{self.host}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                decoded = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            raise VisualJudgeUnavailable(f"Ollama vision request failed: {exc}") from exc
        if not isinstance(decoded, Mapping):
            raise VisualJudgeUnavailable("Ollama vision response was not an object")
        return decoded

    def _assert_vision(self) -> None:
        body = json.dumps({"name": self.model}).encode("utf-8")
        req = request.Request(
            f"{self.host}/api/show",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=20) as response:
                shown = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            raise VisualJudgeUnavailable(f"cannot inspect Ollama model: {exc}") from exc
        capabilities = {
            str(item).casefold() for item in shown.get("capabilities", ())
        } if isinstance(shown, Mapping) else set()
        if "vision" not in capabilities:
            raise VisualJudgeUnavailable(
                f"Ollama model {self.model!r} does not advertise vision capability"
            )

    def _chat(self, prompt: str, images: Sequence[str | Path]) -> Mapping[str, Any]:
        encoded = [
            base64.b64encode(Path(path).read_bytes()).decode("ascii")
            for path in images
        ]
        response = self._post(
            {
                "model": self.model,
                "stream": False,
                "think": False,
                "format": "json",
                "messages": [{"role": "user", "content": prompt, "images": encoded}],
                "options": {"temperature": 0.1},
            }
        )
        message = response.get("message", {})
        if not isinstance(message, Mapping):
            raise VisualJudgeUnavailable("Ollama vision response omitted message")
        return _json_object(str(message.get("content") or ""))

    def evaluate(
        self,
        *,
        brief: str,
        rubric: Mapping[str, Any],
        screenshot: str | Path,
        runtime_evidence: Mapping[str, Any],
        clean_context_nonce: str,
    ) -> VisualJudgeVerdictV1:
        context = {
            "brief": brief,
            "rubric": dict(rubric),
            "runtime_evidence": dict(runtime_evidence),
            "nonce": clean_context_nonce,
        }
        context_fingerprint = hashlib.sha256(
            json.dumps(context, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        value = self._chat(
            "You are an independent visual QA judge. Inspect the supplied screenshot against "
            "the brief and component-specific rubric. Do not reward mere colorfulness, contrast, "
            "or edge density. Return strict JSON with accepted:boolean, confidence:0..1, summary, "
            "scores:{rubric_dimension:0..1}, findings:[{severity:critical|major|minor,"
            "category,message,evidence}]. A critical finding or any critical score below 0.85 "
            "must make accepted false.\nCONTEXT:\n"
            + json.dumps(context, ensure_ascii=False, sort_keys=True),
            (screenshot,),
        )
        findings = tuple(
            VisualFindingV1(
                severity=str(item.get("severity") or "major").casefold(),
                category=str(item.get("category") or "visual"),
                message=str(item.get("message") or "Unspecified visual issue"),
                evidence=str(item.get("evidence") or ""),
            )
            for item in value.get("findings", ())
            if isinstance(item, Mapping)
        )
        scores = {
            str(key): _clamp(score)
            for key, score in dict(value.get("scores") or {}).items()
        }
        required_dimensions = tuple(
            str(item)
            for item in rubric.get("dimensions", ())
            if str(item).strip()
        )
        missing_dimensions = tuple(
            item for item in required_dimensions if item not in scores
        )
        threshold = _clamp(rubric.get("critical_minimum", 0.85))
        below_threshold = tuple(
            item for item in required_dimensions if scores.get(item, 0.0) < threshold
        )
        if missing_dimensions:
            findings += (
                VisualFindingV1(
                    "critical",
                    "rubric_coverage",
                    "Judge omitted required rubric dimensions.",
                    ", ".join(missing_dimensions),
                ),
            )
        if below_threshold:
            findings += (
                VisualFindingV1(
                    "critical",
                    "quality_threshold",
                    f"Critical visual dimensions are below {threshold:.2f}.",
                    ", ".join(below_threshold),
                ),
            )
        accepted = (
            bool(value.get("accepted"))
            and bool(scores)
            and not any(item.severity == "critical" for item in findings)
        )
        return VisualJudgeVerdictV1(
            evaluator=self.evaluator,
            model=self.model,
            accepted=accepted,
            scores=scores,
            findings=findings,
            summary=str(value.get("summary") or ""),
            confidence=_clamp(value.get("confidence")),
            screenshot_hash=_hash(screenshot),
            context_fingerprint=context_fingerprint,
        )

    def compare(
        self,
        *,
        brief: str,
        rubric: Mapping[str, Any],
        candidate: str | Path,
        baseline: str | Path,
        clean_context_nonce: str,
    ) -> PairwiseVisualComparisonV1:
        items = [("candidate", Path(candidate)), ("baseline", Path(baseline))]
        random.Random(clean_context_nonce).shuffle(items)
        labels = {f"image_{index + 1}": name for index, (name, _path) in enumerate(items)}
        context = {
            "brief": brief,
            "rubric": dict(rubric),
            "nonce": clean_context_nonce,
        }
        fingerprint = hashlib.sha256(
            json.dumps(context, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        value = self._chat(
            "Blindly compare the two screenshots against this brief and rubric. Return strict "
            "JSON {preferred:'image_1'|'image_2'|'tie',confidence:0..1,rationale:string}. "
            "Judge modeling, composition, readability, polish, and task fit—not raw saturation.\n"
            + json.dumps(context, ensure_ascii=False, sort_keys=True),
            tuple(path for _name, path in items),
        )
        raw_preferred = str(value.get("preferred") or "tie").casefold()
        preferred = labels.get(raw_preferred, "tie")
        return PairwiseVisualComparisonV1(
            evaluator=self.evaluator,
            model=self.model,
            preferred=preferred,
            confidence=_clamp(value.get("confidence")),
            rationale=str(value.get("rationale") or ""),
            candidate_hash=_hash(candidate),
            baseline_hash=_hash(baseline),
            context_fingerprint=fingerprint,
        )


class OpenAIVisionJudge(OllamaVisionJudge):
    evaluator = "openai_vision"

    def __init__(self, model: str) -> None:
        if not os.getenv("OPENAI_API_KEY"):
            raise VisualJudgeUnavailable("OPENAI_API_KEY is not configured")
        self.model = model
        self.timeout_seconds = 180
        try:
            from openai import OpenAI

            self._client = OpenAI()
        except Exception as exc:
            raise VisualJudgeUnavailable(f"OpenAI vision adapter unavailable: {exc}") from exc

    def _chat(self, prompt: str, images: Sequence[str | Path]) -> Mapping[str, Any]:
        content: list[Mapping[str, Any]] = [{"type": "text", "text": prompt}]
        for image in images:
            media = "image/png" if Path(image).suffix.casefold() == ".png" else "image/jpeg"
            encoded = base64.b64encode(Path(image).read_bytes()).decode("ascii")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{media};base64,{encoded}"},
                }
            )
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": content}],
                response_format={"type": "json_object"},
                temperature=0.1,
            )
            text = response.choices[0].message.content or ""
        except Exception as exc:
            raise VisualJudgeUnavailable(f"OpenAI vision request failed: {exc}") from exc
        return _json_object(text)


class GeminiVisionJudge(OllamaVisionJudge):
    evaluator = "gemini_vision"

    def __init__(self, model: str) -> None:
        if not os.getenv("GEMINI_API_KEY"):
            raise VisualJudgeUnavailable("GEMINI_API_KEY is not configured")
        self.model = model
        self.timeout_seconds = 180
        try:
            from google import genai

            self._client = genai.Client()
        except Exception as exc:
            raise VisualJudgeUnavailable(f"Gemini vision adapter unavailable: {exc}") from exc

    def _chat(self, prompt: str, images: Sequence[str | Path]) -> Mapping[str, Any]:
        try:
            from google.genai import types

            contents: list[Any] = [prompt]
            for image in images:
                media = "image/png" if Path(image).suffix.casefold() == ".png" else "image/jpeg"
                contents.append(
                    types.Part.from_bytes(
                        data=Path(image).read_bytes(),
                        mime_type=media,
                    )
                )
            response = self._client.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.1,
                ),
            )
            text = str(response.text or "")
        except Exception as exc:
            raise VisualJudgeUnavailable(f"Gemini vision request failed: {exc}") from exc
        return _json_object(text)


class UnavailableVisionJudge:
    evaluator = "unavailable"
    model = ""

    def __init__(self, reason: str) -> None:
        self.reason = reason

    def evaluate(self, **_kwargs: Any) -> VisualJudgeVerdictV1:
        raise VisualJudgeUnavailable(self.reason)

    def compare(self, **_kwargs: Any) -> PairwiseVisualComparisonV1:
        raise VisualJudgeUnavailable(self.reason)


def create_visual_judge(
    *,
    builder_provider: str,
    builder_model: str,
    ollama_host: str = "http://127.0.0.1:11434",
) -> VisionJudge:
    provider = os.getenv("AGENT_VISION_PROVIDER", "").strip().casefold()
    model = os.getenv("AGENT_VISION_MODEL", "").strip()
    if not model and builder_model.casefold().startswith(("offline", "fake", "test")):
        return UnavailableVisionJudge("test/offline builder has no vision evaluator")
    if provider in {"", "ollama"} and (
        model or builder_provider.casefold() == "ollama"
    ):
        selected_model = model or builder_model
        if (
            builder_provider.casefold() == "ollama"
            and selected_model.casefold() == builder_model.casefold()
        ):
            return UnavailableVisionJudge(
                "independent visual judging requires a model different from the builder"
            )
        try:
            return OllamaVisionJudge(
                selected_model,
                host=os.getenv("AGENT_VISION_OLLAMA_HOST", ollama_host),
            )
        except VisualJudgeUnavailable as exc:
            return UnavailableVisionJudge(str(exc))
    if provider == "openai":
        try:
            return OpenAIVisionJudge(model or os.getenv("OPENAI_VISION_MODEL", "gpt-4.1"))
        except VisualJudgeUnavailable as exc:
            return UnavailableVisionJudge(str(exc))
    if provider == "gemini":
        try:
            return GeminiVisionJudge(
                model or os.getenv("GEMINI_VISION_MODEL", "gemini-2.5-flash")
            )
        except VisualJudgeUnavailable as exc:
            return UnavailableVisionJudge(str(exc))
    return UnavailableVisionJudge(
        f"no configured independent vision adapter for provider {provider or 'unknown'}"
    )


def require_two_clean_acceptances(
    judge: VisionJudge,
    *,
    brief: str,
    rubric: Mapping[str, Any],
    screenshot: str | Path,
    runtime_evidence: Mapping[str, Any],
    nonce_prefix: str,
) -> tuple[VisualJudgeVerdictV1, VisualJudgeVerdictV1]:
    verdicts = tuple(
        judge.evaluate(
            brief=brief,
            rubric=rubric,
            screenshot=screenshot,
            runtime_evidence=runtime_evidence,
            clean_context_nonce=f"{nonce_prefix}:{index}",
        )
        for index in (1, 2)
    )
    assert len(verdicts) == 2
    return verdicts  # type: ignore[return-value]


__all__ = [
    "OllamaVisionJudge",
    "OpenAIVisionJudge",
    "GeminiVisionJudge",
    "PairwiseVisualComparisonV1",
    "UnavailableVisionJudge",
    "VisionJudge",
    "VisualFindingV1",
    "VisualJudgeUnavailable",
    "VisualJudgeVerdictV1",
    "create_visual_judge",
    "require_two_clean_acceptances",
    "screenshot_anomalies",
]
