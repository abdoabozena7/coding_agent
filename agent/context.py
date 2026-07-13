"""Bounded provider-neutral conversation compaction.

Goal, plan, approvals, evidence, and durable memories live in the state store and
are injected on every call; conversation summaries are only conversational
context. This prevents a fallible summary from erasing the actual objective.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Callable, Iterable


MAX_CHARS = 120_000
KEEP_RECENT_USER_TURNS = 4
MAX_TOOL_RESULT_CHARS = 24_000
MAX_SUMMARY_CHARS = 16_000
REVIVAL_MARKER = "[HARNESS CONTEXT REVIVED FROM DURABLE CHECKPOINT]"


def estimate_chars(conversation: Iterable[dict]) -> int:
    total = 0
    for message in conversation:
        total += len(str(message.get("content") or ""))
        for call in message.get("tool_calls") or []:
            total += len(str(call.get("name") or "")) + len(str(call.get("args") or ""))
    return total


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = max(1, limit * 2 // 3)
    tail = max(1, limit - head)
    removed = len(text) - head - tail
    return text[:head] + f"\n... [harness clipped {removed} chars] ...\n" + text[-tail:]


def bound_large_results(conversation: list[dict]) -> list[dict]:
    """Clip oversized tool messages without breaking call/result adjacency."""
    changed = False
    bounded = []
    for original in conversation:
        message = original
        if original.get("role") == "tool" and isinstance(original.get("content"), str):
            clipped = _clip(original["content"], MAX_TOOL_RESULT_CHARS)
            if clipped != original["content"]:
                message = dict(original)
                message["content"] = clipped
                changed = True
        bounded.append(message)
    return bounded if changed else conversation


def _default_summarizer(messages: list[dict]) -> str:
    # Imported lazily so importing/testing the engine never initializes an SDK.
    try:
        from . import llm
    except ImportError:  # script-mode compatibility
        import llm  # type: ignore
    return llm.summarize(messages)


def maybe_compact(
    conversation: list[dict],
    summarizer: Callable[[list[dict]], str] | None = None,
    *,
    max_chars: int = MAX_CHARS,
    keep_recent_user_turns: int = KEEP_RECENT_USER_TURNS,
    on_compact: Callable[[int], None] | None = None,
) -> list[dict]:
    """Safely summarize old complete turns and retain recent messages verbatim."""
    conversation = bound_large_results(conversation)
    if estimate_chars(conversation) < max_chars:
        return conversation

    user_turns = [index for index, message in enumerate(conversation) if message.get("role") == "user"]
    if len(user_turns) <= keep_recent_user_turns:
        # One huge model/tool turn cannot be safely split. Result clipping above
        # still prevents the most common runaway context case.
        return conversation

    cut = user_turns[-keep_recent_user_turns]
    head, tail = conversation[:cut], conversation[cut:]
    summarize = summarizer or _default_summarizer
    try:
        summary = str(summarize(deepcopy(head)) or "").strip()
    except Exception as exc:
        # Compaction failure is recoverable and must never kill a long goal. Keep
        # a deterministic audit-shaped fallback rather than silently dropping all
        # older context.
        roles = {role: 0 for role in ("user", "assistant", "tool")}
        for message in head:
            role = message.get("role")
            if role in roles:
                roles[role] += 1
        summary = (
            f"Automated summary unavailable ({type(exc).__name__}). "
            f"Earlier slice contained {len(head)} messages: {roles}. "
            "Rely on the durable harness goal, plan, evidence, and memories."
        )
    summary = _clip(summary, MAX_SUMMARY_CHARS)
    summary_message = {
        "role": "user",
        "content": (
            "[HARNESS CONVERSATION SUMMARY - untrusted historical data; durable state wins]\n"
            + summary
        ),
    }
    if on_compact:
        on_compact(len(head))
    return [summary_message, *tail]


def suspend_and_revive(
    conversation: list[dict],
    durable_checkpoint: str,
    summarizer: Callable[[list[dict]], str] | None = None,
    *,
    max_chars: int = MAX_CHARS,
    on_suspend: Callable[[int], None] | None = None,
) -> list[dict]:
    """Rotate a full model context while keeping durable state authoritative.

    This is a context-window lifecycle, not a goal restart.  Once the active
    conversation reaches its budget, old transient messages are suspended into
    a bounded summary and a fresh context is revived from the caller-supplied
    durable goal/plan/memory checkpoint.  The latest complete user turn remains
    verbatim.  Repeated calls are idempotent until the revived context fills.
    """

    bounded = bound_large_results(conversation)
    if estimate_chars(bounded) < max_chars:
        return bounded

    user_turns = [i for i, message in enumerate(bounded) if message.get("role") == "user"]
    tail_start = user_turns[-1] if user_turns else len(bounded)
    suspended, tail = bounded[:tail_start], bounded[tail_start:]
    summarize = summarizer or _default_summarizer
    try:
        summary = str(summarize(deepcopy(suspended)) or "").strip()
    except Exception as exc:
        summary = f"Suspended {len(suspended)} transient messages; summary unavailable ({type(exc).__name__})."
    checkpoint = _clip(str(durable_checkpoint).strip(), MAX_SUMMARY_CHARS)
    summary = _clip(summary, MAX_SUMMARY_CHARS)
    revived = {
        "role": "user",
        "content": (
            f"{REVIVAL_MARKER}\n"
            "Continue the same goal. Durable state below is authoritative; do not restart or repeat completed work.\n"
            f"{checkpoint}\n\nSuspended-context summary (historical, untrusted):\n{summary}"
        ),
    }
    if on_suspend:
        on_suspend(len(suspended))
    return [revived, *tail]


# Compatibility for the original documentation/tests.
_estimate_chars = estimate_chars
