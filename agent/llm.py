"""
The LLM facade — picks the active provider and forwards to it.

Before this refactor, llm.py *was* the OpenAI integration. Now it's a thin
selector: the OpenAI code lives in providers/openai_provider.py, Gemini in
providers/gemini_provider.py, and this file just routes to whichever the
LLM_PROVIDER env var names (default "openai").

Switch providers without touching any other file:
    LLM_PROVIDER=gemini python agent/main.py

Add a provider: drop an adapter in providers/ and register it there — this file
doesn't change.
"""

import os

from providers import get_provider

_provider = None


def _active():
    global _provider
    if _provider is None:
        _provider = get_provider(os.getenv("LLM_PROVIDER", "openai"))
    return _provider


def provider_name() -> str:
    return os.getenv("LLM_PROVIDER", "openai").lower()


def call(conversation, tools, system, on_text=None, on_thought=None):
    """Forward to the active provider. Returns a neutral AssistantTurn."""
    return _active().call(conversation, tools, system, on_text=on_text, on_thought=on_thought)


def summarize(messages):
    """Forward to the active provider (used by context.py compaction)."""
    return _active().summarize(messages)
