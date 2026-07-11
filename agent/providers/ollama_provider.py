"""Ollama chat adapter, including local and Ollama cloud models."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Mapping
from typing import Any

from .base import (
    AssistantTurn,
    ProviderCapabilities,
    SUMMARY_INSTRUCTION,
    ToolCall,
    Usage,
    coerce_tool_args,
    native_data,
    render_for_summary,
    unique_tool_call_id,
)

MODEL_NAME = "gpt-oss:120b-cloud"
DEFAULT_HOST = "http://localhost:11434"


class OllamaProvider:
    capabilities = ProviderCapabilities(
        streaming=True,
        tool_calling=True,
        thinking=True,
        # Ollama does not guarantee native call IDs, although this adapter
        # synthesizes stable neutral IDs for the harness.
        tool_call_ids=False,
        native_replay=True,
    )

    def __init__(self, model: str | None = None, host: str | None = None):
        self.model = model or os.getenv("OLLAMA_MODEL") or MODEL_NAME
        self.host = (host or os.getenv("OLLAMA_HOST", DEFAULT_HOST)).rstrip("/")

    def _post_json(self, path: str, payload: dict):
        data = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        request = urllib.request.Request(
            f"{self.host}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            return urllib.request.urlopen(request, timeout=300)
        except urllib.error.URLError as error:
            raise RuntimeError(f"Could not reach Ollama at {self.host}: {error}") from error

    def _to_messages(self, conversation, system):
        messages = [{"role": "system", "content": str(system or "")}]
        for message in conversation or []:
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
                    "content": str(message.get("content") or ""),
                }
                replay = native_data(message, "ollama")
                thinking = replay.get("thinking") or message.get("thinking")
                if thinking:
                    out["thinking"] = str(thinking)

                calls = []
                raw_calls = message.get("tool_calls") or []
                if isinstance(raw_calls, (list, tuple)):
                    for call in raw_calls:
                        if not isinstance(call, Mapping):
                            continue
                        calls.append(
                            {
                                "function": {
                                    "name": str(call.get("name") or "unknown_tool"),
                                    "arguments": coerce_tool_args(call.get("args")),
                                }
                            }
                        )
                if calls:
                    out["tool_calls"] = calls
                messages.append(out)
            elif role == "tool":
                content = message.get("content")
                messages.append(
                    {
                        "role": "tool",
                        "content": content if isinstance(content, str) else str(content or ""),
                        # Ollama associates a result with the requested
                        # function by tool_name, not OpenAI's tool_call_id.
                        "tool_name": str(message.get("name") or "unknown_tool"),
                    }
                )
        return messages

    def call(self, conversation, tools, system, on_text=None, on_thought=None) -> AssistantTurn:
        payload = {
            "model": self.model,
            "messages": self._to_messages(conversation, system),
            "tools": tools or [],
            "stream": True,
            "think": True,
        }

        text_parts: list[str] = []
        thought_parts: list[str] = []
        raw_tool_calls: list[Mapping[str, Any]] = []
        usage = None

        with self._post_json("/api/chat", payload) as response:
            for raw_line in response:
                if not raw_line or not raw_line.strip():
                    continue
                try:
                    chunk = json.loads(raw_line)
                except (json.JSONDecodeError, UnicodeDecodeError, TypeError):
                    # A truncated NDJSON line should not take down a long run;
                    # later chunks may still contain a complete answer/call.
                    continue
                if not isinstance(chunk, Mapping):
                    continue
                if chunk.get("error"):
                    raise RuntimeError(f"Ollama error: {chunk['error']}")
                message = chunk.get("message")
                if not isinstance(message, Mapping):
                    message = {}

                thought = message.get("thinking") or chunk.get("thinking")
                if thought:
                    fragment = thought if isinstance(thought, str) else str(thought)
                    thought_parts.append(fragment)
                    if on_thought:
                        on_thought(fragment)

                content = message.get("content")
                if content:
                    fragment = content if isinstance(content, str) else str(content)
                    text_parts.append(fragment)
                    if on_text:
                        on_text(fragment)

                calls = message.get("tool_calls")
                if isinstance(calls, (list, tuple)):
                    raw_tool_calls.extend(
                        call for call in calls if isinstance(call, Mapping)
                    )

                if chunk.get("done"):
                    usage = Usage(
                        input_tokens=chunk.get("prompt_eval_count", 0) or 0,
                        output_tokens=chunk.get("eval_count", 0) or 0,
                    )

        tool_calls = []
        seen_ids: set[str] = set()
        for output_index, raw_call in enumerate(raw_tool_calls):
            function = raw_call.get("function")
            if not isinstance(function, Mapping):
                function = {}
            name = str(
                function.get("name")
                or raw_call.get("tool_name")
                or raw_call.get("name")
                or "unknown_tool"
            )
            args = coerce_tool_args(
                function.get("arguments", raw_call.get("arguments"))
            )
            call_id = unique_tool_call_id(
                "ollama",
                raw_call.get("id") or function.get("id"),
                output_index,
                name,
                seen_ids,
            )
            tool_calls.append(ToolCall(id=call_id, name=name, args=args))

        native = (
            {"provider": "ollama", "thinking": "".join(thought_parts)}
            if thought_parts
            else {}
        )
        return AssistantTurn(
            text="".join(text_parts) or None,
            tool_calls=tool_calls,
            usage=usage,
            native=native,
        )

    def summarize(self, messages) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SUMMARY_INSTRUCTION},
                {"role": "user", "content": render_for_summary(messages)},
            ],
            "stream": False,
        }
        with self._post_json("/api/chat", payload) as response:
            try:
                data = json.loads(response.read().decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError, AttributeError):
                return ""
        message = data.get("message") if isinstance(data, Mapping) else None
        if not isinstance(message, Mapping):
            return ""
        return str(message.get("content") or "").strip()
