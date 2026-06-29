"""
context.py — keep the conversation from growing without bound.

The model is stateless, so we resend the whole history every turn (Phase 1).
Prompt caching makes the repeated PREFIX cheap, but it does NOT stop the history
from eventually overflowing the context window. Compaction is the fix: once the
history gets large, summarize the OLDER turns into a short note and keep only
the recent turns verbatim.

The one subtlety: we must cut the history at a SAFE boundary. Every assistant
`tool_call` must keep its matching `role:"tool"` result, or the next API call is
rejected. We only ever cut at the start of a real user turn — which guarantees
each turn's (assistant tool_calls + tool results) stays together on one side of
the cut.
"""

import llm

# Rough budget. Production code counts tokens exactly (e.g. tiktoken); a
# characters/4 heuristic is plenty for a learning project. ~24k chars ≈ ~6k tokens.
MAX_CHARS = 24_000
KEEP_RECENT_USER_TURNS = 2  # how many recent user turns to keep verbatim


def _estimate_chars(conversation) -> int:
    """Rough size of the conversation, in characters."""
    total = 0
    for m in conversation:
        total += len(str(m.get("content") or ""))
        for tc in m.get("tool_calls") or []:
            total += len(tc["name"]) + len(str(tc["args"]))
    return total


def maybe_compact(conversation):
    """Summarize the older part of the history if it has grown too large.

    Returns a new conversation list (or the same one if no compaction was
    needed). Safe to call after every turn.
    """
    if _estimate_chars(conversation) < MAX_CHARS:
        return conversation  # still small — nothing to do

    # Real user turns are role:"user" (tool results are role:"tool", assistant
    # replies are role:"assistant"). Cutting at a user turn keeps every
    # tool_call paired with its result.
    user_turns = [i for i, m in enumerate(conversation) if m.get("role") == "user"]
    if len(user_turns) <= KEEP_RECENT_USER_TURNS:
        return conversation  # not enough turns to safely compact yet

    cut = user_turns[-KEEP_RECENT_USER_TURNS]   # index where the kept tail begins
    head, tail = conversation[:cut], conversation[cut:]

    # Replace the entire head with one summary message. Because the whole head is
    # replaced, no tool_call inside it can be left dangling.
    summary = llm.summarize(head)
    summary_msg = {
        "role": "user",
        "content": "[Summary of earlier conversation]\n" + summary,
    }
    print(f"  [compacted {len(head)} older messages into a summary]")
    return [summary_msg] + tail
