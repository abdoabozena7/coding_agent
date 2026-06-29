"""
Ollama adapter. Talks to a local Ollama server, including Ollama cloud models.

Change MODEL_NAME below to switch models. You can still override it temporarily
with the OLLAMA_MODEL environment variable.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from .base import AssistantTurn, ToolCall, Usage, render_for_summary, SUMMARY_INSTRUCTION

# Change only this name to switch Ollama models, for example:
#   "gpt-oss:120b-cloud"
#   "qwen2.5-coder:7b"
#   "deepseek-coder:6.7b"
#   "gemma4:e4b"
MODEL_NAME = "gpt-oss:120b-cloud"
DEFAULT_HOST = "http://localhost:11434"


class OllamaProvider:
    def __init__(self, model: str | None = None, host: str | None = None):
        self.model = model or os.getenv("OLLAMA_MODEL") or MODEL_NAME
        self.host = (host or os.getenv("OLLAMA_HOST", DEFAULT_HOST)).rstrip("/")

    def _post_json(self, path: str, payload: dict):
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            return urllib.request.urlopen(req, timeout=300)
        except urllib.error.URLError as e:
            raise RuntimeError(f"Could not reach Ollama at {self.host}: {e}") from e

    def _to_messages(self, conversation, system):
        messages = [{"role": "system", "content": system}]
        for m in conversation:
            role = m["role"]
            if role == "user":
                messages.append({"role": "user", "content": m["content"]})
            elif role == "assistant":
                msg = {"role": "assistant", "content": m.get("content") or ""}
                if m.get("tool_calls"):
                    msg["tool_calls"] = [
                        {
                            "function": {
                                "name": tc["name"],
                                "arguments": tc.get("args") or {},
                            }
                        }
                        for tc in m["tool_calls"]
                    ]
                messages.append(msg)
            elif role == "tool":
                messages.append({"role": "tool", "content": m["content"]})
        return messages

    def call(self, conversation, tools, system, on_text=None, on_thought=None) -> AssistantTurn:
        payload = {
            "model": self.model,
            "messages": self._to_messages(conversation, system),
            "tools": tools,
            "stream": True,
            "think": True,
        }

        text_parts = []
        raw_tool_calls = []
        usage = None

        with self._post_json("/api/chat", payload) as resp:
            for raw_line in resp:
                if not raw_line.strip():
                    continue
                chunk = json.loads(raw_line)
                message = chunk.get("message") or {}

                thought = message.get("thinking") or chunk.get("thinking")
                if thought and on_thought:
                    on_thought(thought)

                content = message.get("content")
                if content:
                    text_parts.append(content)
                    if on_text:
                        on_text(content)

                if message.get("tool_calls"):
                    raw_tool_calls.extend(message["tool_calls"])

                if chunk.get("done"):
                    usage = Usage(
                        input_tokens=chunk.get("prompt_eval_count", 0) or 0,
                        output_tokens=chunk.get("eval_count", 0) or 0,
                    )

        tool_calls = []
        for idx, tc in enumerate(raw_tool_calls):
            fn = tc.get("function") or {}
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                args = json.loads(args or "{}")
            name = fn.get("name")
            if name:
                tool_calls.append(ToolCall(id=f"ollama-{idx}-{name}", name=name, args=args))

        return AssistantTurn(text="".join(text_parts) or None, tool_calls=tool_calls, usage=usage)

    def summarize(self, messages) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SUMMARY_INSTRUCTION},
                {"role": "user", "content": render_for_summary(messages)},
            ],
            "stream": False,
        }
        with self._post_json("/api/chat", payload) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return ((data.get("message") or {}).get("content") or "").strip()
