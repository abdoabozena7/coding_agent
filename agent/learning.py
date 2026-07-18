"""Sanitized cross-project lessons with evidence-updated confidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
from threading import RLock
from typing import Any, Iterable, Mapping

from .safety import redact_text


@dataclass(frozen=True, slots=True)
class LearnedLessonV1:
    title: str
    content: str
    applicability_tags: tuple[str, ...]
    evidence_refs: tuple[str, ...] = ()
    scope: str = "global"
    successes: int = 1
    failures: int = 0
    confidence: float = 2.0 / 3.0
    version: int = 1
    superseded_by: str | None = None
    id: str = ""

    def __post_init__(self) -> None:
        if self.scope not in {"project", "global"}:
            raise ValueError("lesson scope must be project or global")
        if not self.title.strip() or not self.content.strip():
            raise ValueError("lesson title/content must be non-empty")
        lesson_id = self.id or "lesson-" + hashlib.sha256(
            (self.title.casefold() + "\n" + " ".join(sorted(self.applicability_tags))).encode("utf-8")
        ).hexdigest()[:20]
        object.__setattr__(self, "id", lesson_id)
        object.__setattr__(self, "confidence", max(0.0, min(1.0, float(self.confidence))))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _tokens(value: str) -> set[str]:
    return {
        item.casefold()
        for item in re.findall(r"[\w.+#-]+", str(value), flags=re.UNICODE)
        if len(item) > 1
    }


def sanitize_global_lesson(value: str, *, limit: int = 2_000) -> str:
    text = redact_text(str(value), limit * 2)
    text = re.sub(r"```[\s\S]*?```", "[code omitted]", text)
    text = re.sub(r"(?:[A-Za-z]:\\|/)(?:[^\s'\"]+[\\/])+[^\s'\"]+", "[project path]", text)
    return " ".join(text.split())[:limit]


class GlobalLessonStore:
    def __init__(self, path: str | Path | None = None) -> None:
        configured = path or os.getenv("AGENT_GLOBAL_MEMORY_PATH")
        self.path = Path(configured).expanduser() if configured else Path.home() / ".coding-agent" / "global-lessons.json"
        self._lock = RLock()

    def _load(self) -> dict[str, dict[str, Any]]:
        if not self.path.is_file():
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return {}
        return {
            str(item["id"]): dict(item)
            for item in payload.get("lessons", ())
            if isinstance(item, Mapping) and item.get("id")
        } if isinstance(payload, Mapping) else {}

    def _save(self, values: Mapping[str, Mapping[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps({"version": 1, "lessons": list(values.values())}, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(self.path)

    def put(self, lesson: LearnedLessonV1) -> LearnedLessonV1:
        sanitized = LearnedLessonV1(
            **{
                **lesson.to_dict(),
                "title": sanitize_global_lesson(lesson.title, limit=200),
                "content": sanitize_global_lesson(lesson.content),
                "evidence_refs": tuple(
                    sanitize_global_lesson(item, limit=200) for item in lesson.evidence_refs
                ),
            }
        )
        with self._lock:
            values = self._load()
            prior = values.get(sanitized.id)
            if prior:
                successes = int(prior.get("successes", 0)) + sanitized.successes
                failures = int(prior.get("failures", 0)) + sanitized.failures
                confidence = (successes + 1.0) / (successes + failures + 2.0)
                sanitized = LearnedLessonV1(
                    **{
                        **sanitized.to_dict(),
                        "successes": successes,
                        "failures": failures,
                        "confidence": confidence,
                    }
                )
            values[sanitized.id] = sanitized.to_dict()
            self._save(values)
        return sanitized

    def search(self, query: str, *, limit: int = 8) -> tuple[LearnedLessonV1, ...]:
        wanted = _tokens(query)
        with self._lock:
            values = self._load().values()
        scored: list[tuple[float, LearnedLessonV1]] = []
        for value in values:
            try:
                lesson = LearnedLessonV1(**dict(value))
            except (TypeError, ValueError):
                continue
            if lesson.superseded_by:
                continue
            available = _tokens(" ".join((lesson.title, lesson.content, *lesson.applicability_tags)))
            similarity = len(wanted & available) / max(1, len(wanted | available))
            score = 0.7 * similarity + 0.3 * lesson.confidence
            if score > 0.05:
                scored.append((score, lesson))
        return tuple(item for _score, item in sorted(scored, key=lambda pair: (-pair[0], pair[1].id))[:limit])

    def record_outcome(self, lesson_id: str, *, succeeded: bool) -> LearnedLessonV1 | None:
        with self._lock:
            values = self._load()
            raw = values.get(lesson_id)
            if raw is None:
                return None
            successes = int(raw.get("successes", 0)) + int(bool(succeeded))
            failures = int(raw.get("failures", 0)) + int(not succeeded)
            raw["successes"] = successes
            raw["failures"] = failures
            raw["confidence"] = (successes + 1.0) / (successes + failures + 2.0)
            values[lesson_id] = raw
            self._save(values)
            return LearnedLessonV1(**raw)


__all__ = ["GlobalLessonStore", "LearnedLessonV1", "sanitize_global_lesson"]
