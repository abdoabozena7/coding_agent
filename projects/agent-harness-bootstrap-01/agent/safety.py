"""Secret redaction and no-progress detection for durable event/tool records."""

from __future__ import annotations

import hashlib
import json
import re
from collections import deque
from dataclasses import dataclass
from typing import Any


_SECRET_PATTERNS = (
    re.compile(r"(?i)\b((?:openai|gemini|google|github|gitlab|slack|aws)[_-]?(?:api[_-]?)?(?:key|token|secret))\s*[:=]\s*([^\s,;]+)"),
    re.compile(
        r"(?i)\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret|"
        r"password|passwd)\s*[:=]\s*['\"]?[^\s,;'\"]{8,}['\"]?"
    ),
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{20,}\b"),
    re.compile(r"\bgh[opusr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
    re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{12,}"),
    re.compile(
        r"-----BEGIN ((?:[A-Z0-9]+ )*PRIVATE KEY(?: BLOCK)?)-----"
        r"[\s\S]{20,20000}?-----END \1-----"
    ),
)


def redact_text(value: Any, limit: int | None = None) -> str:
    text = str(value)
    for index, pattern in enumerate(_SECRET_PATTERNS):
        if index == 0:
            text = pattern.sub(lambda match: f"{match.group(1)}=[REDACTED]", text)
        else:
            text = pattern.sub("[REDACTED]", text)
    if limit is not None and len(text) > limit:
        text = text[:limit] + f"\n... [truncated {len(text) - limit} chars]"
    return text


def redact_data(value: Any) -> Any:
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if re.search(r"(?i)(key|token|secret|password|authorization)", str(key)):
                result[key] = "[REDACTED]"
            else:
                result[key] = redact_data(item)
        return result
    if isinstance(value, list):
        return [redact_data(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_data(item) for item in value)
    if isinstance(value, str):
        return redact_text(value)
    return value


def action_fingerprint(name: str, args: dict[str, Any]) -> str:
    canonical = json.dumps({"name": name, "args": args}, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(canonical.encode("utf-8", "replace")).hexdigest()


@dataclass(frozen=True)
class ProgressDecision:
    stalled: bool
    reason: str = ""


class ProgressWatchdog:
    """Reject repeated identical actions/results while allowing long useful runs."""

    def __init__(self, repeat_limit: int = 2, history_size: int = 100) -> None:
        self.repeat_limit = max(1, repeat_limit)
        self._history: deque[tuple[str, str]] = deque(maxlen=max(4, history_size))

    def check(self, name: str, args: dict[str, Any], previous_result: str | None = None) -> ProgressDecision:
        fingerprint = action_fingerprint(name, args)
        result_hash = hashlib.sha256((previous_result or "").encode("utf-8", "replace")).hexdigest()
        identical = 0
        for old_action, old_result in reversed(self._history):
            if old_action != fingerprint or (previous_result is not None and old_result != result_hash):
                break
            identical += 1
        if identical >= self.repeat_limit:
            return ProgressDecision(
                True,
                "no-progress circuit breaker: this identical action already repeated; inspect the failure and choose a different approach",
            )
        return ProgressDecision(False)

    def record(self, name: str, args: dict[str, Any], result: str) -> None:
        self._history.append(
            (
                action_fingerprint(name, args),
                hashlib.sha256(result.encode("utf-8", "replace")).hexdigest(),
            )
        )
