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
    "create_provider",
]

_PROVIDERS = ("openai", "gemini", "ollama")


def get_provider(
    name: str,
    *,
    model: str | None = None,
    host: str | None = None,
    require_gpu: bool | None = None,
):
    """Create an independent adapter instance.

    The optional arguments preserve the original environment-driven behavior
    while allowing a model picker to pin the exact model (and Ollama host) in a
    durable descriptor.  This function never caches adapters, which is
    important when parallel agents each need isolated SDK/client state.
    """

    name = (name or "openai").strip().lower()
    if name == "openai":
        from .openai_provider import OpenAIProvider
        return OpenAIProvider(model=model)
    if name == "gemini":
        from .gemini_provider import GeminiProvider
        return GeminiProvider(model=model)
    if name == "ollama":
        from .ollama_provider import OllamaProvider
        return OllamaProvider(model=model, host=host, require_gpu=require_gpu)
    raise ValueError(f"Unknown LLM_PROVIDER '{name}'. Options: {', '.join(_PROVIDERS)}")


def create_provider(descriptor):
    """Create a fresh adapter from a ``ModelDescriptor``-like object."""

    try:
        name = descriptor.provider
        model = descriptor.model
    except AttributeError as exc:
        raise TypeError("descriptor must expose provider and model attributes") from exc
    execution = str(getattr(getattr(descriptor, "execution_class", ""), "value", getattr(descriptor, "execution_class", "")))
    return get_provider(
        name,
        model=model,
        host=getattr(descriptor, "host", None),
        require_gpu=False if execution == "cloud" else None,
    )


def get_provider_capabilities(name: str) -> ProviderCapabilities:
    """Return feature metadata without creating a network connection."""

    return get_provider(name).capabilities
