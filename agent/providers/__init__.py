"""
The provider registry.

Adding a new LLM backend = add an adapter module here with a class that has
call() + summarize() (see base.Provider), then register it in get_provider().
Nothing in main.py / tools / context changes — same payoff as the tool registry.

We import each adapter LAZILY (only when selected) so you don't need a
provider's SDK installed unless you actually use it.
"""

from .base import (  # re-exported for convenience
    AssistantTurn,
    Provider,
    ProviderCapabilities,
    ToolCall,
    Usage,
)

__all__ = [
    "AssistantTurn",
    "Provider",
    "ProviderCapabilities",
    "ToolCall",
    "Usage",
    "get_provider",
    "get_provider_capabilities",
]

_PROVIDERS = ("openai", "gemini", "ollama")


def get_provider(name: str):
    name = (name or "openai").strip().lower()
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


def get_provider_capabilities(name: str) -> ProviderCapabilities:
    """Return feature metadata without creating a network connection."""

    return get_provider(name).capabilities
