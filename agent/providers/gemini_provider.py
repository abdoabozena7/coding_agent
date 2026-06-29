"""
Gemini adapter (google-genai). Translates neutral <-> Gemini shapes.

Needs a Gemini key in the env (GEMINI_API_KEY or GOOGLE_API_KEY). The Gemini
wire format differs from OpenAI's in several ways, all absorbed here:
  • turns are `contents` of `parts`; the assistant role is "model"
  • tool calls are function_call parts; their args are already a dict
  • tool results are function_response parts in a role:"tool" content (matched
    by NAME, not an id)
  • the system prompt is a separate `system_instruction`, not a message

Bonus: Gemini returns its reasoning as "thought" parts inline (no separate API,
unlike OpenAI). We surface those via on_thought when include_thoughts is on.
"""

import os

from google import genai
from google.genai import types

from .base import AssistantTurn, ToolCall, Usage, render_for_summary, SUMMARY_INSTRUCTION

DEFAULT_MODEL = "gemini-2.5-flash"


def _function_declaration(fn: dict) -> "types.FunctionDeclaration":
    """Build a Gemini FunctionDeclaration from our OpenAI-shaped tool schema
    inner `function` dict ({name, description, parameters})."""
    params = fn.get("parameters")
    try:
        # Newer google-genai accepts standard JSON Schema directly.
        return types.FunctionDeclaration(
            name=fn["name"], description=fn.get("description"), parameters_json_schema=params
        )
    except Exception:
        # Fallback: hand it as `parameters` (the SDK coerces the dict to a Schema).
        return types.FunctionDeclaration(
            name=fn["name"], description=fn.get("description"), parameters=params
        )


class GeminiProvider:
    def __init__(self, model: str | None = None):
        self.model = model or os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
        self._client = None

    def _client_(self):
        if self._client is None:
            self._client = genai.Client()  # reads GEMINI_API_KEY / GOOGLE_API_KEY
        return self._client

    # --- neutral conversation -> Gemini contents ---
    def _to_contents(self, conversation):
        contents = []
        for m in conversation:
            role = m["role"]
            if role == "user":
                contents.append(types.Content(role="user", parts=[types.Part(text=m["content"])]))
            elif role == "assistant":
                parts = []
                if m.get("content"):
                    parts.append(types.Part(text=m["content"]))
                for tc in m.get("tool_calls") or []:
                    parts.append(types.Part(function_call=types.FunctionCall(name=tc["name"], args=tc["args"])))
                contents.append(types.Content(role="model", parts=parts))
            elif role == "tool":
                part = types.Part.from_function_response(name=m["name"], response={"result": m["content"]})
                contents.append(types.Content(role="tool", parts=[part]))
        return contents

    def _to_tools(self, tools):
        # `tools` are our OpenAI-shaped schemas: {"type":"function","function":{...}}.
        decls = [_function_declaration(t["function"]) for t in tools]
        return [types.Tool(function_declarations=decls)]

    def call(self, conversation, tools, system, on_text=None, on_thought=None) -> AssistantTurn:
        config = types.GenerateContentConfig(
            system_instruction=system,
            tools=self._to_tools(tools),
            thinking_config=types.ThinkingConfig(include_thoughts=True),
        )

        stream = self._client_().models.generate_content_stream(
            model=self.model,
            contents=self._to_contents(conversation),
            config=config,
        )

        text_parts = []
        tool_calls = []
        usage = None

        for chunk in stream:
            if getattr(chunk, "usage_metadata", None):
                usage = chunk.usage_metadata
            if not getattr(chunk, "candidates", None):
                continue  # some thinking chunks arrive with no candidates
            cand = chunk.candidates[0]
            if not cand.content or not cand.content.parts:
                continue
            for part in cand.content.parts:
                fc = getattr(part, "function_call", None)
                if fc:
                    # Gemini calls have no id; use the name (results are matched by name).
                    tool_calls.append(ToolCall(id=fc.name, name=fc.name, args=dict(fc.args or {})))
                    continue
                text = getattr(part, "text", None)
                if not text:
                    continue
                if getattr(part, "thought", False):
                    if on_thought:
                        on_thought(text)  # reasoning summary
                else:
                    text_parts.append(text)
                    if on_text:
                        on_text(text)

        u = None
        if usage:
            u = Usage(
                input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
                cached_tokens=getattr(usage, "cached_content_token_count", 0) or 0,
                output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
            )

        return AssistantTurn(text="".join(text_parts) or None, tool_calls=tool_calls, usage=u)

    def summarize(self, messages) -> str:
        resp = self._client_().models.generate_content(
            model=self.model,
            contents=render_for_summary(messages),
            config=types.GenerateContentConfig(system_instruction=SUMMARY_INSTRUCTION),
        )
        return resp.text or ""
