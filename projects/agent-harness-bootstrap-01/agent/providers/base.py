"""Provider-neutral request and response types.

The agent loop stores a deliberately small conversation format::

    {"role": "user", "content": "..."}
    {"role": "assistant", "content": "...", "tool_calls": [...]}
    {"role": "tool", "id": "...", "name": "...", "content": "..."}

``native`` is an optional escape hatch for state that a provider must receive
again on the next request (for example a Gemini thought signature).  The agent
does not interpret it, but :meth:`AssistantTurn.to_message` keeps it in history
so the originating adapter can replay it losslessly.
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol


@dataclass(frozen=True)
class ProviderCapabilities:
    """Features an adapter can provide to a harness.

    The short field names are convenient for display code.  The
    ``supports_*`` properties make capability checks read naturally and keep
    callers from inferring support from a provider's class name.
    """

    streaming: bool = False
    tool_calling: bool = False
    thinking: bool = False
    tool_call_ids: bool = False
    native_replay: bool = False

    @property
    def supports_streaming(self) -> bool:
        return self.streaming

    @property
    def supports_tools(self) -> bool:
        return self.tool_calling

    @property
    def supports_tool_calling(self) -> bool:
        return self.tool_calling

    @property
    def supports_thinking(self) -> bool:
        return self.thinking

    @property
    def supports_tool_call_ids(self) -> bool:
        return self.tool_call_ids

    @property
    def preserves_native_replay(self) -> bool:
        return self.native_replay

    def as_dict(self) -> dict[str, bool]:
        return {
            "streaming": self.streaming,
            "tool_calling": self.tool_calling,
            "thinking": self.thinking,
            "tool_call_ids": self.tool_call_ids,
            "native_replay": self.native_replay,
        }


@dataclass
class ToolCall:
    """A normalized tool request.

    ``args`` is always a dictionary, even when a backend emitted malformed
    JSON.  ``native`` may contain adapter-owned replay data and is ignored by
    the tool runner.
    """

    id: str
    name: str
    args: dict[str, Any]
    native: dict[str, Any] = field(default_factory=dict)


@dataclass
class Usage:
    """Token counts normalized across providers."""

    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0


@dataclass
class AssistantTurn:
    """The provider-neutral shape of one model reply."""

    text: Optional[str] = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Optional[Usage] = None
    native: dict[str, Any] = field(default_factory=dict)

    def to_message(self) -> dict[str, Any]:
        """Convert the turn to the durable neutral history representation."""

        msg: dict[str, Any] = {"role": "assistant", "content": self.text}
        if self.tool_calls:
            calls = []
            for call in self.tool_calls:
                item: dict[str, Any] = {
                    "id": call.id,
                    "name": call.name,
                    "args": copy.deepcopy(call.args),
                }
                if call.native:
                    item["native"] = copy.deepcopy(call.native)
                calls.append(item)
            msg["tool_calls"] = calls
        if self.native:
            msg["native"] = copy.deepcopy(self.native)
        return msg


class Provider(Protocol):
    """The contract implemented by every model backend."""

    capabilities: ProviderCapabilities

    def call(
        self,
        conversation,
        tools,
        system,
        on_text: Optional[Callable[[str], None]] = None,
        on_thought: Optional[Callable[[str], None]] = None,
    ) -> AssistantTurn: ...

    def summarize(self, messages) -> str: ...


def coerce_tool_args(value: Any) -> dict[str, Any]:
    """Best-effort conversion of provider tool arguments to a dictionary.

    Model streams can end early and leave a partial JSON string.  Treating an
    invalid or non-object value as an empty argument object lets the normal
    tool error path tell the model what went wrong instead of crashing the
    harness.
    """

    if value is None or value == "":
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, (bytes, bytearray)):
        try:
            value = bytes(value).decode("utf-8")
        except UnicodeDecodeError:
            return {}
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, TypeError, ValueError):
            return {}
        return dict(parsed) if isinstance(parsed, Mapping) else {}
    return {}


def tool_call_id(provider: str, value: Any, index: int, name: Any = None) -> str:
    """Return a usable provider ID, synthesizing one when it is absent."""

    if value is not None:
        candidate = str(value).strip()
        if candidate:
            return candidate
    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "-", str(name or "tool")).strip("-")
    return f"{provider}-call-{index}-{safe_name or 'tool'}"


def unique_tool_call_id(
    provider: str,
    value: Any,
    index: int,
    name: Any,
    used: set[str],
) -> str:
    """Normalize an ID and make it unique inside one assistant turn."""

    base = tool_call_id(provider, value, index, name)
    candidate = base
    suffix = 2
    while candidate in used:
        candidate = f"{base}-{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


def native_data(message: Mapping[str, Any], provider: str) -> dict[str, Any]:
    """Read provider-owned replay data while tolerating legacy nesting."""

    native = message.get("native")
    if not isinstance(native, Mapping):
        return {}
    if native.get("provider") == provider:
        return dict(native)
    nested = native.get(provider)
    return dict(nested) if isinstance(nested, Mapping) else {}


def render_for_summary(messages) -> str:
    """Flatten neutral conversation messages into plain text for compaction."""

    lines = []
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        role = message.get("role")
        if role == "tool":
            lines.append(f"[tool result] {message.get('content')}")
        elif role == "assistant":
            if message.get("content"):
                lines.append(f"assistant: {message['content']}")
            for call in message.get("tool_calls") or []:
                if isinstance(call, Mapping):
                    lines.append(
                        f"assistant called {call.get('name', 'unknown_tool')}"
                        f"({coerce_tool_args(call.get('args'))})"
                    )
        else:
            lines.append(f"user: {message.get('content')}")
    return "\n".join(lines)


SUMMARY_INSTRUCTION = (
    "Summarize this slice of a coding-assistant conversation into a concise, "
    "factual note. Preserve decisions made, files read or edited, commands run, "
    "and key results. Omit pleasantries."
)
