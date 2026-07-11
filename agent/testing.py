"""Offline provider utilities for tests and deterministic agent demos.

``ScriptedProvider`` implements the same contract as a real model adapter but
never opens a socket.  Tests can queue complete turns (or callables that inspect
the recorded request), exercise streaming callbacks, and assert that the whole
script was consumed.
"""

from __future__ import annotations

import copy
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .providers import AssistantTurn, ProviderCapabilities, ToolCall, Usage
from .providers.base import coerce_tool_args, native_data, render_for_summary, tool_call_id


@dataclass(frozen=True)
class ProviderRequest:
    conversation: list[Any]
    tools: list[Any]
    system: str


@dataclass
class ScriptedTurn:
    """A turn plus the exact fragments delivered to streaming callbacks."""

    turn: AssistantTurn
    text_chunks: list[str] = field(default_factory=list)
    thought_chunks: list[str] = field(default_factory=list)


ScriptItem = (
    AssistantTurn
    | ScriptedTurn
    | str
    | Mapping[str, Any]
    | BaseException
    | Callable[[ProviderRequest], Any]
)


class ScriptedProvider:
    """Deterministic, network-free provider implementing the production API."""

    capabilities = ProviderCapabilities(
        streaming=True,
        tool_calling=True,
        thinking=True,
        tool_call_ids=True,
        native_replay=True,
    )

    def __init__(
        self,
        turns: Iterable[ScriptItem] = (),
        *,
        summaries: Iterable[str | BaseException | Callable[[list[Any]], str]] = (),
        model: str = "scripted",
    ):
        self.model = model
        self._turns = deque(turns)
        self._summaries = deque(summaries)
        self.calls: list[ProviderRequest] = []
        self.summary_calls: list[list[Any]] = []

    @property
    def remaining(self) -> int:
        return len(self._turns)

    @staticmethod
    def _coerce_usage(value: Any) -> Usage | None:
        if value is None or isinstance(value, Usage):
            return copy.deepcopy(value)
        if isinstance(value, Mapping):
            return Usage(
                input_tokens=value.get("input_tokens", 0) or 0,
                cached_tokens=value.get("cached_tokens", 0) or 0,
                output_tokens=value.get("output_tokens", 0) or 0,
            )
        return None

    @staticmethod
    def _coerce_turn(value: Any) -> ScriptedTurn:
        if isinstance(value, ScriptedTurn):
            return copy.deepcopy(value)
        if isinstance(value, AssistantTurn):
            return ScriptedTurn(copy.deepcopy(value))
        if isinstance(value, str):
            return ScriptedTurn(AssistantTurn(text=value))
        if not isinstance(value, Mapping):
            raise TypeError(f"Unsupported scripted turn: {type(value).__name__}")

        raw_calls = value.get("tool_calls") or []
        calls = []
        if isinstance(raw_calls, (list, tuple)):
            for index, call in enumerate(raw_calls):
                if isinstance(call, ToolCall):
                    calls.append(copy.deepcopy(call))
                    continue
                if not isinstance(call, Mapping):
                    continue
                name = str(call.get("name") or "unknown_tool")
                native = call.get("native")
                calls.append(
                    ToolCall(
                        id=tool_call_id("scripted", call.get("id"), index, name),
                        name=name,
                        args=coerce_tool_args(call.get("args")),
                        native=(
                            copy.deepcopy(dict(native))
                            if isinstance(native, Mapping)
                            else {}
                        ),
                    )
                )
        native = value.get("native")
        turn = AssistantTurn(
            text=value.get("text", value.get("content")),
            tool_calls=calls,
            usage=ScriptedProvider._coerce_usage(value.get("usage")),
            native=(copy.deepcopy(dict(native)) if isinstance(native, Mapping) else {}),
        )
        text_chunks = value.get("text_chunks") or []
        thought_chunks = value.get("thought_chunks") or []
        return ScriptedTurn(
            turn=turn,
            text_chunks=[str(part) for part in text_chunks],
            thought_chunks=[str(part) for part in thought_chunks],
        )

    @staticmethod
    def _native_thoughts(turn: AssistantTurn) -> list[str]:
        ollama = native_data(turn.to_message(), "ollama")
        if ollama.get("thinking"):
            return [str(ollama["thinking"])]
        gemini = native_data(turn.to_message(), "gemini")
        parts = gemini.get("parts")
        if isinstance(parts, list):
            return [
                str(part.get("text"))
                for part in parts
                if isinstance(part, Mapping)
                and part.get("type") == "thought"
                and part.get("text")
            ]
        return []

    def call(self, conversation, tools, system, on_text=None, on_thought=None) -> AssistantTurn:
        request = ProviderRequest(
            conversation=copy.deepcopy(list(conversation or [])),
            tools=copy.deepcopy(list(tools or [])),
            system=str(system or ""),
        )
        self.calls.append(request)
        if not self._turns:
            raise AssertionError("ScriptedProvider has no turn left to return")

        item = self._turns.popleft()
        if isinstance(item, BaseException):
            raise item
        if callable(item):
            item = item(request)
            if isinstance(item, BaseException):
                raise item
        scripted = self._coerce_turn(item)

        thoughts = scripted.thought_chunks or self._native_thoughts(scripted.turn)
        if on_thought:
            for fragment in thoughts:
                on_thought(fragment)
        text_chunks = scripted.text_chunks
        if not text_chunks and scripted.turn.text:
            text_chunks = [scripted.turn.text]
        if on_text:
            for fragment in text_chunks:
                on_text(fragment)
        return copy.deepcopy(scripted.turn)

    def summarize(self, messages) -> str:
        snapshot = copy.deepcopy(list(messages or []))
        self.summary_calls.append(snapshot)
        if not self._summaries:
            return render_for_summary(snapshot)
        item = self._summaries.popleft()
        if isinstance(item, BaseException):
            raise item
        if callable(item):
            return str(item(snapshot))
        return str(item)

    def assert_exhausted(self) -> None:
        if self._turns or self._summaries:
            raise AssertionError(
                "ScriptedProvider still has "
                f"{len(self._turns)} turn(s) and {len(self._summaries)} summary item(s)"
            )


__all__ = ["ProviderRequest", "ScriptedProvider", "ScriptedTurn"]
