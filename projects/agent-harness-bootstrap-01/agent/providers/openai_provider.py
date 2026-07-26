"""OpenAI Chat Completions adapter.

All streaming reassembly and wire-format quirks stay in this module.  In
particular, incomplete tool-call streams are normalized instead of being
allowed to crash the agent while decoding JSON.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any

from openai import OpenAI

from .base import (
    AssistantTurn,
    ProviderCapabilities,
    SUMMARY_INSTRUCTION,
    ToolCall,
    Usage,
    coerce_tool_args,
    render_for_summary,
    tool_call_id,
    unique_tool_call_id,
)

DEFAULT_MODEL = "gpt-5.5"


def _value(obj: Any, name: str, default=None):
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


class OpenAIProvider:
    capabilities = ProviderCapabilities(
        streaming=True,
        tool_calling=True,
        thinking=False,
        tool_call_ids=True,
        native_replay=False,
    )

    def __init__(self, model: str | None = None, reasoning_effort: str = "medium"):
        self.model = model or os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
        self.reasoning_effort = str(reasoning_effort)
        self._client = None

    def _client_(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI()
        return self._client

    def _to_messages(self, conversation, system):
        """Translate neutral history and repair absent call/result IDs."""

        messages = [{"role": "system", "content": str(system or "")}]
        pending: list[dict[str, str]] = []

        for message_index, message in enumerate(conversation or []):
            if not isinstance(message, Mapping):
                continue
            role = message.get("role")
            if role == "user":
                messages.append(
                    {"role": "user", "content": str(message.get("content") or "")}
                )
            elif role == "assistant":
                out: dict[str, Any] = {
                    "role": "assistant",
                    "content": message.get("content"),
                }
                calls = []
                raw_calls = message.get("tool_calls") or []
                if isinstance(raw_calls, (list, tuple)):
                    for call_index, call in enumerate(raw_calls):
                        if not isinstance(call, Mapping):
                            continue
                        name = str(call.get("name") or "unknown_tool")
                        call_id = tool_call_id(
                            "openai",
                            call.get("id"),
                            message_index * 1000 + call_index,
                            name,
                        )
                        calls.append(
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": json.dumps(
                                        coerce_tool_args(call.get("args")),
                                        ensure_ascii=False,
                                        default=str,
                                    ),
                                },
                            }
                        )
                        pending.append({"id": call_id, "name": name})
                if calls:
                    out["tool_calls"] = calls
                messages.append(out)
            elif role == "tool":
                supplied_id = message.get("id")
                selected = None
                if supplied_id is not None and str(supplied_id).strip():
                    call_id = str(supplied_id).strip()
                    selected = next(
                        (item for item in pending if item["id"] == call_id), None
                    )
                else:
                    name = str(message.get("name") or "")
                    selected = next(
                        (item for item in pending if name and item["name"] == name),
                        pending[0] if pending else None,
                    )
                    call_id = (
                        selected["id"]
                        if selected
                        else tool_call_id("openai", None, message_index, name)
                    )
                if selected in pending:
                    pending.remove(selected)
                content = message.get("content")
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": content if isinstance(content, str) else str(content or ""),
                    }
                )
        return messages

    def call(self, conversation, tools, system, on_text=None, on_thought=None) -> AssistantTurn:
        # Chat Completions currently exposes no reasoning-summary stream, so
        # on_thought is intentionally unused.
        request: dict[str, Any] = {
            "model": self.model,
            "messages": self._to_messages(conversation, system),
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        request["reasoning_effort"] = self.reasoning_effort
        if tools:
            request["tools"] = tools
        stream = self._client_().chat.completions.create(**request)

        usage = None
        content_parts: list[str] = []
        # A tuple key keeps real indices ordered before anonymous fallback
        # positions and avoids comparing ints with strings while sorting.
        slots: dict[tuple[int, int], dict[str, Any]] = {}

        for chunk in stream:
            chunk_usage = _value(chunk, "usage")
            if chunk_usage is not None:
                usage = chunk_usage
            choices = _value(chunk, "choices") or []
            if not choices:
                continue
            delta = _value(choices[0], "delta")
            if delta is None:
                continue

            content = _value(delta, "content")
            if content:
                fragment = content if isinstance(content, str) else str(content)
                content_parts.append(fragment)
                if on_text:
                    on_text(fragment)

            streamed_calls = _value(delta, "tool_calls") or []
            for position, streamed_call in enumerate(streamed_calls):
                raw_index = _value(streamed_call, "index")
                key = (
                    (0, raw_index)
                    if isinstance(raw_index, int) and raw_index >= 0
                    else (1, position)
                )
                slot = slots.setdefault(
                    key,
                    {"id": "", "name": "", "arg_fragments": [], "args": None},
                )
                call_id_fragment = _value(streamed_call, "id")
                if call_id_fragment:
                    fragment = str(call_id_fragment)
                    if not slot["id"]:
                        slot["id"] = fragment
                    elif fragment != slot["id"]:
                        slot["id"] += fragment

                function = _value(streamed_call, "function")
                if function is None:
                    continue
                name_fragment = _value(function, "name")
                if name_fragment:
                    fragment = str(name_fragment)
                    if not slot["name"]:
                        slot["name"] = fragment
                    elif fragment != slot["name"]:
                        slot["name"] += fragment
                arguments = _value(function, "arguments")
                if isinstance(arguments, Mapping):
                    slot["args"] = dict(arguments)
                elif arguments is not None:
                    slot["arg_fragments"].append(str(arguments))

        tool_calls = []
        seen_ids: set[str] = set()
        for output_index, (_, slot) in enumerate(sorted(slots.items())):
            name = slot["name"] or "unknown_tool"
            raw_args = (
                slot["args"]
                if slot["args"] is not None
                else "".join(slot["arg_fragments"])
            )
            call_id = unique_tool_call_id(
                "openai", slot["id"], output_index, name, seen_ids
            )
            tool_calls.append(
                ToolCall(
                    id=call_id,
                    name=name,
                    args=coerce_tool_args(raw_args),
                )
            )

        normalized_usage = None
        if usage is not None:
            details = _value(usage, "prompt_tokens_details")
            normalized_usage = Usage(
                input_tokens=_value(usage, "prompt_tokens", 0) or 0,
                cached_tokens=_value(details, "cached_tokens", 0) or 0,
                output_tokens=_value(usage, "completion_tokens", 0) or 0,
            )

        return AssistantTurn(
            text="".join(content_parts) or None,
            tool_calls=tool_calls,
            usage=normalized_usage,
        )

    def summarize(self, messages) -> str:
        response = self._client_().chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SUMMARY_INSTRUCTION},
                {"role": "user", "content": render_for_summary(messages)},
            ],
        )
        choices = _value(response, "choices") or []
        if not choices:
            return ""
        message = _value(choices[0], "message")
        return _value(message, "content", "") or ""
