"""
OpenAI adapter (Chat Completions). Translates neutral <-> OpenAI shapes.

This is the from-scratch agent's original LLM code, moved behind the Provider
interface. All the OpenAI-specific quirks (JSON-string tool arguments, the
role:"tool" + tool_call_id pairing, streaming reassembly) live here now, so
main.py never sees them.
"""

import json
import os

from openai import OpenAI

from .base import AssistantTurn, ToolCall, Usage, render_for_summary, SUMMARY_INSTRUCTION

DEFAULT_MODEL = "gpt-5.5"


class OpenAIProvider:
    def __init__(self, model: str | None = None):
        self.model = model or os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
        self._client = None  # lazy: built on first use, after .env is loaded

    def _client_(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI()  # reads OPENAI_API_KEY
        return self._client

    # --- neutral conversation -> OpenAI messages ---
    def _to_messages(self, conversation, system):
        msgs = [{"role": "system", "content": system}]
        for m in conversation:
            role = m["role"]
            if role == "user":
                msgs.append({"role": "user", "content": m["content"]})
            elif role == "assistant":
                om = {"role": "assistant", "content": m.get("content")}
                if m.get("tool_calls"):
                    om["tool_calls"] = [
                        {
                            "id": tc["id"],
                            "type": "function",
                            # OpenAI wants the arguments as a JSON STRING:
                            "function": {"name": tc["name"], "arguments": json.dumps(tc["args"])},
                        }
                        for tc in m["tool_calls"]
                    ]
                msgs.append(om)
            elif role == "tool":
                # OpenAI links a tool result by tool_call_id.
                msgs.append({"role": "tool", "tool_call_id": m["id"], "content": m["content"]})
        return msgs

    def call(self, conversation, tools, system, on_text=None, on_thought=None) -> AssistantTurn:
        # (Chat Completions exposes no reasoning summary, so on_thought is unused here.)
        stream = self._client_().chat.completions.create(
            model=self.model,
            messages=self._to_messages(conversation, system),
            tools=tools,
            stream=True,
            stream_options={"include_usage": True},
        )

        usage = None
        content_parts = []
        slots = {}  # tool-call index -> {"id", "name", "args"}

        for chunk in stream:
            if chunk.usage:
                usage = chunk.usage
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                content_parts.append(delta.content)
                if on_text:
                    on_text(delta.content)
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    slot = slots.setdefault(tc.index, {"id": None, "name": None, "args": ""})
                    if tc.id:
                        slot["id"] = tc.id
                    if tc.function and tc.function.name:
                        slot["name"] = tc.function.name
                    if tc.function and tc.function.arguments:
                        slot["args"] += tc.function.arguments

        tool_calls = [
            ToolCall(id=s["id"], name=s["name"], args=json.loads(s["args"] or "{}"))
            for _, s in sorted(slots.items())
        ]

        u = None
        if usage:
            details = getattr(usage, "prompt_tokens_details", None)
            u = Usage(
                input_tokens=usage.prompt_tokens or 0,
                cached_tokens=(getattr(details, "cached_tokens", 0) or 0),
                output_tokens=usage.completion_tokens or 0,
            )

        return AssistantTurn(text="".join(content_parts) or None, tool_calls=tool_calls, usage=u)

    def summarize(self, messages) -> str:
        resp = self._client_().chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SUMMARY_INSTRUCTION},
                {"role": "user", "content": render_for_summary(messages)},
            ],
        )
        return resp.choices[0].message.content or ""
