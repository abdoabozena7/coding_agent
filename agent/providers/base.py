"""
Neutral, provider-agnostic types + the Provider interface.

The rest of the agent (main.py, tools, context) speaks ONLY these types and the
neutral conversation format below. Each provider adapter translates between them
and its own SDK — so adding a provider never touches the agent loop. It's the
same idea as the tools/ registry, applied to LLM backends.

Neutral conversation format (the list `main.py` keeps):
  user:      {"role": "user", "content": str}
  assistant: {"role": "assistant", "content": str|None, "tool_calls": [ToolCall-dict]?}
  tool:      {"role": "tool", "id": str, "name": str, "content": str}
where a ToolCall-dict is {"id": str, "name": str, "args": dict}  (args already parsed).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol


@dataclass
class ToolCall:
    """One tool the model wants to run. `args` is already a parsed dict —
    providers that hand us a JSON string parse it inside their adapter."""
    id: str
    name: str
    args: dict


@dataclass
class Usage:
    """Token counts, normalized across providers."""
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0


@dataclass
class AssistantTurn:
    """What every provider's call() returns — the neutral shape of one model reply."""
    text: Optional[str] = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Optional[Usage] = None

    def to_message(self) -> dict:
        """Convert to a neutral conversation message dict for the history."""
        msg = {"role": "assistant", "content": self.text}
        if self.tool_calls:
            msg["tool_calls"] = [
                {"id": c.id, "name": c.name, "args": c.args} for c in self.tool_calls
            ]
        return msg


class Provider(Protocol):
    """The contract every backend implements. main.py only ever sees this."""

    def call(
        self,
        conversation,
        tools,
        system,
        on_text: Optional[Callable[[str], None]] = None,
        on_thought: Optional[Callable[[str], None]] = None,
    ) -> AssistantTurn: ...

    def summarize(self, messages) -> str: ...


def render_for_summary(messages) -> str:
    """Flatten neutral conversation messages into plain text (used for compaction)."""
    lines = []
    for m in messages:
        role = m.get("role")
        if role == "tool":
            lines.append(f"[tool result] {m.get('content')}")
        elif role == "assistant":
            if m.get("content"):
                lines.append(f"assistant: {m['content']}")
            for tc in m.get("tool_calls") or []:
                lines.append(f"assistant called {tc['name']}({tc['args']})")
        else:  # user
            lines.append(f"user: {m.get('content')}")
    return "\n".join(lines)


# Shared system prompt for the summarizer (compaction), used by every adapter.
SUMMARY_INSTRUCTION = (
    "Summarize this slice of a coding-assistant conversation into a concise, "
    "factual note. Preserve decisions made, files read or edited, commands run, "
    "and key results. Omit pleasantries."
)
