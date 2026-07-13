"""Gemini adapter built on ``google-genai``.

Gemini function calls carry IDs in the installed SDK and may carry opaque
thought signatures.  Both have to be replayed with the model turn, while
function responses must be sent as ``role='user'`` content.  This adapter keeps
that provider-owned state under the neutral message's ``native`` key.
"""

from __future__ import annotations

import base64
import os
from collections.abc import Mapping
from typing import Any

from google import genai
from google.genai import types

from .base import (
    AssistantTurn,
    ProviderCapabilities,
    SUMMARY_INSTRUCTION,
    ToolCall,
    Usage,
    coerce_tool_args,
    native_data,
    render_for_summary,
    tool_call_id,
    unique_tool_call_id,
)

DEFAULT_MODEL = "gemini-2.5-flash"


def _value(obj: Any, name: str, default=None):
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _encode_signature(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.encode("utf-8")
    if isinstance(value, bytearray):
        value = bytes(value)
    if not isinstance(value, bytes):
        return None
    return base64.b64encode(value).decode("ascii")


def _decode_signature(value: Any) -> bytes | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value
    if not isinstance(value, str):
        return None
    try:
        return base64.b64decode(value.encode("ascii"), validate=True)
    except (ValueError, UnicodeEncodeError):
        # Tolerate hand-written/legacy metadata that stored the raw string.
        return value.encode("utf-8")


def _function_declaration(function: Mapping[str, Any]) -> "types.FunctionDeclaration":
    params = function.get("parameters")
    try:
        return types.FunctionDeclaration(
            name=str(function["name"]),
            description=function.get("description"),
            parameters_json_schema=params,
        )
    except Exception:
        # Older google-genai releases used ``parameters``.  Keeping this
        # fallback costs nothing and makes the adapter portable.
        return types.FunctionDeclaration(
            name=str(function["name"]),
            description=function.get("description"),
            parameters=params,
        )


class GeminiProvider:
    capabilities = ProviderCapabilities(
        streaming=True,
        tool_calling=True,
        thinking=True,
        tool_call_ids=True,
        native_replay=True,
    )

    def __init__(self, model: str | None = None, reasoning_effort: str = "medium"):
        self.model = model or os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
        self.reasoning_effort = str(reasoning_effort)
        self._client = None

    def _client_(self):
        if self._client is None:
            self._client = genai.Client()
        return self._client

    @staticmethod
    def _part_from_native(item: Mapping[str, Any]) -> "types.Part | None":
        kind = item.get("type")
        signature = _decode_signature(item.get("thought_signature"))
        if kind == "function_call":
            name = str(item.get("name") or "unknown_tool")
            call_id = item.get("id")
            return types.Part(
                function_call=types.FunctionCall(
                    id=str(call_id) if call_id else None,
                    name=name,
                    args=coerce_tool_args(item.get("args")),
                ),
                thought=bool(item.get("thought")) or None,
                thought_signature=signature,
            )
        if kind in {"text", "thought"}:
            text = item.get("text")
            if text is None:
                return None
            return types.Part(
                text=str(text),
                thought=(kind == "thought") or bool(item.get("thought")),
                thought_signature=signature,
            )
        return None

    def _assistant_parts(self, message: Mapping[str, Any], message_index: int):
        replay = native_data(message, "gemini")
        raw_parts = replay.get("parts")
        if isinstance(raw_parts, list):
            native_parts = [
                part
                for item in raw_parts
                if isinstance(item, Mapping)
                for part in [self._part_from_native(item)]
                if part is not None
            ]
            if native_parts:
                return native_parts

        parts = []
        if message.get("content"):
            parts.append(types.Part(text=str(message["content"])))
        raw_calls = message.get("tool_calls") or []
        if isinstance(raw_calls, (list, tuple)):
            for call_index, call in enumerate(raw_calls):
                if not isinstance(call, Mapping):
                    continue
                name = str(call.get("name") or "unknown_tool")
                call_id = tool_call_id(
                    "gemini",
                    call.get("id"),
                    message_index * 1000 + call_index,
                    name,
                )
                call_native = native_data(call, "gemini")
                parts.append(
                    types.Part(
                        function_call=types.FunctionCall(
                            id=call_id,
                            name=name,
                            args=coerce_tool_args(call.get("args")),
                        ),
                        thought_signature=_decode_signature(
                            call_native.get("thought_signature")
                        ),
                    )
                )
        return parts

    def _to_contents(self, conversation):
        contents = []
        pending_tool_parts = []

        def flush_tools():
            if pending_tool_parts:
                # google-genai's own automatic function-calling implementation
                # uses role="user" and groups a batch of responses in one turn.
                contents.append(types.Content(role="user", parts=list(pending_tool_parts)))
                pending_tool_parts.clear()

        for message_index, message in enumerate(conversation or []):
            if not isinstance(message, Mapping):
                continue
            role = message.get("role")
            if role == "tool":
                name = str(message.get("name") or "unknown_tool")
                response_id = message.get("id")
                content = message.get("content")
                response = (
                    dict(content)
                    if isinstance(content, Mapping)
                    else {"result": content if content is not None else ""}
                )
                pending_tool_parts.append(
                    types.Part(
                        function_response=types.FunctionResponse(
                            id=str(response_id) if response_id else None,
                            name=name,
                            response=response,
                        )
                    )
                )
                continue

            flush_tools()
            if role == "user":
                contents.append(
                    types.Content(
                        role="user",
                        parts=[types.Part(text=str(message.get("content") or ""))],
                    )
                )
            elif role == "assistant":
                parts = self._assistant_parts(message, message_index)
                if parts:
                    contents.append(types.Content(role="model", parts=parts))

        flush_tools()
        return contents

    def _to_tools(self, tools):
        declarations = []
        for tool in tools or []:
            if not isinstance(tool, Mapping):
                continue
            function = tool.get("function")
            if not isinstance(function, Mapping) or not function.get("name"):
                continue
            declarations.append(_function_declaration(function))
        return [types.Tool(function_declarations=declarations)] if declarations else []

    def call(self, conversation, tools, system, on_text=None, on_thought=None) -> AssistantTurn:
        gemini_tools = self._to_tools(tools)
        config = types.GenerateContentConfig(
            system_instruction=str(system or ""),
            tools=gemini_tools or None,
            thinking_config=types.ThinkingConfig(
                include_thoughts=True,
                thinking_budget={"low": 1024, "medium": 4096, "high": 8192, "xhigh": 16384}.get(
                    str(getattr(self, "reasoning_effort", "medium")), 4096
                ),
            ),
        )
        stream = self._client_().models.generate_content_stream(
            model=self.model,
            contents=self._to_contents(conversation),
            config=config,
        )

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        native_parts: list[dict[str, Any]] = []
        usage = None
        seen_ids: set[str] = set()

        for chunk in stream:
            chunk_usage = _value(chunk, "usage_metadata")
            if chunk_usage is not None:
                usage = chunk_usage
            candidates = _value(chunk, "candidates") or []
            if not candidates:
                continue
            content = _value(candidates[0], "content")
            parts = _value(content, "parts") or []
            for part in parts:
                function_call = _value(part, "function_call")
                signature = _encode_signature(_value(part, "thought_signature"))
                if function_call is not None:
                    name = str(_value(function_call, "name") or "unknown_tool")
                    output_index = len(tool_calls)
                    call_id = unique_tool_call_id(
                        "gemini",
                        _value(function_call, "id"),
                        output_index,
                        name,
                        seen_ids,
                    )
                    args = coerce_tool_args(_value(function_call, "args"))
                    call_native: dict[str, Any] = {"provider": "gemini"}
                    native_part: dict[str, Any] = {
                        "type": "function_call",
                        "id": call_id,
                        "name": name,
                        "args": args,
                    }
                    if signature:
                        call_native["thought_signature"] = signature
                        native_part["thought_signature"] = signature
                    if _value(part, "thought", False):
                        native_part["thought"] = True
                    tool_calls.append(
                        ToolCall(
                            id=call_id,
                            name=name,
                            args=args,
                            native=call_native if len(call_native) > 1 else {},
                        )
                    )
                    native_parts.append(native_part)
                    continue

                text = _value(part, "text")
                if text is None:
                    continue
                fragment = text if isinstance(text, str) else str(text)
                is_thought = bool(_value(part, "thought", False))
                native_part = {
                    "type": "thought" if is_thought else "text",
                    "text": fragment,
                }
                if signature:
                    native_part["thought_signature"] = signature
                native_parts.append(native_part)
                if is_thought:
                    if on_thought and fragment:
                        on_thought(fragment)
                else:
                    text_parts.append(fragment)
                    if on_text and fragment:
                        on_text(fragment)

        normalized_usage = None
        if usage is not None:
            normalized_usage = Usage(
                input_tokens=_value(usage, "prompt_token_count", 0) or 0,
                cached_tokens=_value(usage, "cached_content_token_count", 0) or 0,
                output_tokens=_value(usage, "candidates_token_count", 0) or 0,
            )

        native = (
            {"provider": "gemini", "parts": native_parts} if native_parts else {}
        )
        return AssistantTurn(
            text="".join(text_parts) or None,
            tool_calls=tool_calls,
            usage=normalized_usage,
            native=native,
        )

    def summarize(self, messages) -> str:
        response = self._client_().models.generate_content(
            model=self.model,
            contents=render_for_summary(messages),
            config=types.GenerateContentConfig(
                system_instruction=SUMMARY_INSTRUCTION
            ),
        )
        return _value(response, "text", "") or ""
