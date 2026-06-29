"""
The provider registry.

Adding a new LLM backend = add an adapter module here with a class that has
call() + summarize() (see base.Provider), then register it in get_provider().
Nothing in main.py / tools / context changes — same payoff as the tool registry.

We import each adapter LAZILY (only when selected) so you don't need a
provider's SDK installed unless you actually use it.
"""

from .base import AssistantTurn, ToolCall, Usage, Provider  # re-exported for convenience

_PROVIDERS = ("openai", "gemini", "ollama")


def get_provider(name: str):
    name = (name or "openai").lower()
    if name == "openai":
        from .openai_provider import OpenAIProvider
        return OpenAIProvider()
    if name == "gemini":
        from .gemini_provider import GeminiProvider
        return GeminiProvider()
    if name == "ollama":
        from .ollama_provider import OllamaProvider
        return OllamaProvider()
    raise ValueError(f"Unknown LLM_PROVIDER '{name}'. Options: {', '.join(_PROVIDERS)}")
