"""Live model discovery without coupling the UI to provider SDKs.

The catalog intentionally uses Ollama's small HTTP metadata endpoints rather
than instantiating an :class:`~agent.providers.ollama_provider.OllamaProvider`.
That keeps startup probes fast, bounded, and straightforward to exercise in
offline tests.  Provider instances are created later, one per agent run, from
the immutable :class:`ModelDescriptor` selected by the user.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any


DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_OPENAI_MODEL = "gpt-5.5"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_PROBE_TIMEOUT = 3.0


class ExecutionClass(str, Enum):
    """Where inference is performed for scheduling purposes."""

    LOCAL = "local"
    CLOUD = "cloud"

    @classmethod
    def parse(cls, value: str | "ExecutionClass") -> "ExecutionClass":
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError as exc:
            raise ValueError("execution class must be 'local' or 'cloud'") from exc

    @property
    def default_concurrency(self) -> int:
        return 1 if self is ExecutionClass.LOCAL else 4


@dataclass(frozen=True, slots=True)
class ModelDescriptor:
    """A secret-free, durable description of one selectable model."""

    provider: str
    model: str
    execution_class: ExecutionClass | str
    host: str | None = None
    capabilities: tuple[str, ...] = ("tools",)
    label: str | None = None
    source: str = "configured"
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False, hash=False)

    def __post_init__(self) -> None:
        provider = str(self.provider).strip().lower()
        model = str(self.model).strip()
        if provider not in {"ollama", "openai", "gemini"}:
            raise ValueError(f"unsupported model provider: {provider!r}")
        if not model:
            raise ValueError("a model descriptor requires a model name")
        host = str(self.host).strip().rstrip("/") if self.host else None
        if provider == "ollama":
            host = _normalize_host(host or DEFAULT_OLLAMA_HOST)
        capabilities = tuple(
            dict.fromkeys(
                str(capability).strip().lower()
                for capability in self.capabilities
                if str(capability).strip()
            )
        )
        object.__setattr__(self, "provider", provider)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "execution_class", ExecutionClass.parse(self.execution_class))
        object.__setattr__(self, "host", host)
        object.__setattr__(self, "capabilities", capabilities)
        object.__setattr__(self, "label", str(self.label).strip() if self.label else None)
        object.__setattr__(self, "source", str(self.source).strip() or "configured")
        # Discovery metadata must never expose an API key or a mutable object.
        object.__setattr__(self, "metadata", MappingProxyType(_secret_free_metadata(self.metadata)))

    @property
    def id(self) -> str:
        suffix = f"@{self.host}" if self.provider == "ollama" and self.host else ""
        return f"{self.provider}:{self.model}{suffix}"

    @property
    def key(self) -> str:
        """Compatibility alias used by pickers and persisted selections."""

        return self.id

    @property
    def display_name(self) -> str:
        return self.label or f"{self.model} ({self.provider})"

    @property
    def supports_tools(self) -> bool:
        return any(
            capability in {"tools", "tool_calling", "tool-calling"}
            for capability in self.capabilities
        )

    @property
    def default_concurrency(self) -> int:
        return self.execution_class.default_concurrency

    def create_provider(self):
        """Create a fresh provider adapter configured for exactly this model."""

        from .providers import create_provider

        return create_provider(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "provider": self.provider,
            "model": self.model,
            "execution_class": self.execution_class.value,
            "host": self.host,
            "capabilities": list(self.capabilities),
            "label": self.label,
            "source": self.source,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class CatalogDiagnostic:
    source: str
    message: str


HttpJSON = Callable[..., Mapping[str, Any]]


class ModelCatalog:
    """Discover tool-capable Ollama models plus configured cloud fallbacks.

    ``http_json`` is an injectable callable with this shape::

        request(method, url, payload=None, headers={}, timeout=3.0) -> mapping

    Discovery is best-effort: an unreachable Ollama daemon never hides a
    configured OpenAI or Gemini model.  Details are exposed in
    :attr:`diagnostics` for the UI rather than raised as startup failures.
    """

    def __init__(
        self,
        *,
        environ: Mapping[str, str] | None = None,
        ollama_host: str | None = None,
        http_json: HttpJSON | None = None,
        timeout: float = DEFAULT_PROBE_TIMEOUT,
    ) -> None:
        self._environ = dict(os.environ if environ is None else environ)
        self.ollama_host = _normalize_host(
            ollama_host or self._environ.get("OLLAMA_HOST") or DEFAULT_OLLAMA_HOST
        )
        self._http_json = http_json or _urllib_json
        self.timeout = max(0.1, float(timeout))
        self._diagnostics: list[CatalogDiagnostic] = []

    @property
    def diagnostics(self) -> tuple[CatalogDiagnostic, ...]:
        return tuple(self._diagnostics)

    def discover(self) -> tuple[ModelDescriptor, ...]:
        """Return a deterministic Ollama-first selectable catalog."""

        self._diagnostics.clear()
        ollama = self.discover_ollama()
        configured = self.configured_cloud_models()
        return tuple(ollama + configured)

    def discover_ollama(self) -> list[ModelDescriptor]:
        headers = self._ollama_headers()
        try:
            version_data = self._request("GET", "/api/version", headers=headers)
            version = str(version_data.get("version") or "unknown")
            tags = self._request("GET", "/api/tags", headers=headers)
        except Exception as exc:
            self._diagnose("ollama", exc)
            return []

        raw_models = tags.get("models")
        if not isinstance(raw_models, (list, tuple)):
            self._diagnostics.append(
                CatalogDiagnostic("ollama", "Ollama /api/tags returned no model list")
            )
            return []

        descriptors: list[ModelDescriptor] = []
        seen: set[str] = set()
        for raw_model in raw_models:
            if not isinstance(raw_model, Mapping):
                continue
            model = str(raw_model.get("name") or raw_model.get("model") or "").strip()
            if not model or model in seen:
                continue
            seen.add(model)
            try:
                shown = self._request(
                    "POST",
                    "/api/show",
                    payload={"model": model, "verbose": False},
                    headers=headers,
                )
            except Exception as exc:
                self._diagnose(f"ollama:{model}", exc)
                continue

            capabilities = _capabilities(shown)
            if not _supports_tools(capabilities):
                self._diagnostics.append(
                    CatalogDiagnostic(
                        f"ollama:{model}",
                        "model omitted because /api/show does not advertise tools",
                    )
                )
                continue

            execution_class = (
                ExecutionClass.CLOUD
                if _is_ollama_cloud(self.ollama_host, model, raw_model, shown)
                else ExecutionClass.LOCAL
            )
            descriptors.append(
                ModelDescriptor(
                    provider="ollama",
                    model=model,
                    host=self.ollama_host,
                    execution_class=execution_class,
                    capabilities=capabilities,
                    source="ollama",
                    metadata={
                        "ollama_version": version,
                        "digest": raw_model.get("digest"),
                        "parameter_size": _nested_value(raw_model, "details", "parameter_size"),
                    },
                )
            )

        descriptors.sort(
            key=lambda item: (
                item.execution_class is ExecutionClass.CLOUD,
                item.model.casefold(),
            )
        )
        return descriptors

    def configured_cloud_models(self) -> list[ModelDescriptor]:
        """Describe SDK providers only when their credentials are configured."""

        descriptors: list[ModelDescriptor] = []
        if _configured(self._environ.get("OPENAI_API_KEY")):
            descriptors.append(
                ModelDescriptor(
                    provider="openai",
                    model=self._environ.get("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL,
                    execution_class=ExecutionClass.CLOUD,
                    capabilities=("completion", "tools", "streaming"),
                    label="OpenAI · " + (self._environ.get("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL),
                    source="environment",
                )
            )
        gemini_key = self._environ.get("GEMINI_API_KEY") or self._environ.get("GOOGLE_API_KEY")
        if _configured(gemini_key):
            descriptors.append(
                ModelDescriptor(
                    provider="gemini",
                    model=self._environ.get("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL,
                    execution_class=ExecutionClass.CLOUD,
                    capabilities=("completion", "tools", "streaming", "thinking"),
                    label="Gemini · " + (self._environ.get("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL),
                    source="environment",
                )
            )
        return descriptors

    def by_id(self, descriptor_id: str) -> ModelDescriptor | None:
        return next((item for item in self.discover() if item.id == descriptor_id), None)

    def _request(
        self,
        method: str,
        path: str,
        *,
        payload: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Mapping[str, Any]:
        result = self._http_json(
            method,
            f"{self.ollama_host}{path}",
            payload=payload,
            headers=dict(headers or {}),
            timeout=self.timeout,
        )
        if not isinstance(result, Mapping):
            raise ValueError(f"{path} returned a non-object JSON response")
        return result

    def _ollama_headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        # Direct ollama.com access needs a bearer key.  It is sent to the HTTP
        # seam only and is never copied into descriptors or diagnostics.
        api_key = self._environ.get("OLLAMA_API_KEY")
        if _host_is_ollama_cloud(self.ollama_host) and _configured(api_key):
            headers["Authorization"] = f"Bearer {api_key.strip()}"
        return headers

    def _diagnose(self, source: str, error: Exception) -> None:
        # Exception messages from urllib contain endpoints, not request
        # headers.  Still strip any configured keys defensively before display.
        message = str(error).strip() or error.__class__.__name__
        for key_name in ("OLLAMA_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
            secret = self._environ.get(key_name)
            if _configured(secret):
                message = message.replace(secret, "[REDACTED]")
        self._diagnostics.append(CatalogDiagnostic(source, message))


def _normalize_host(host: str) -> str:
    value = str(host).strip().rstrip("/")
    if not value:
        return DEFAULT_OLLAMA_HOST
    if "://" not in value:
        value = "http://" + value
    parsed = urllib.parse.urlsplit(value)
    if not parsed.hostname:
        raise ValueError(f"invalid Ollama host: {host!r}")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("Ollama host URLs must not contain credentials")
    return value


def _host_is_ollama_cloud(host: str) -> bool:
    hostname = (urllib.parse.urlsplit(host).hostname or "").casefold().rstrip(".")
    return hostname == "ollama.com" or hostname.endswith(".ollama.com")


def _cloud_name(model: str) -> bool:
    value = model.casefold()
    return value.endswith(":cloud") or value.endswith("-cloud")


def _remote_metadata(value: Mapping[str, Any]) -> bool:
    """Recognize current and forward-compatible Ollama remote markers."""

    for key, item in value.items():
        normalized = str(key).casefold().replace("-", "_")
        if normalized in {"cloud", "hosted", "is_cloud", "is_remote"}:
            if item is True or str(item).casefold() in {"true", "cloud", "remote", "hosted"}:
                return True
        if normalized in {"remote_host", "remote_model", "remote_url"} and item:
            return True
        if normalized in {"source", "location", "execution"}:
            if str(item).casefold() in {"cloud", "remote", "hosted", "ollama.com"}:
                return True
        if normalized == "remote":
            if item is True or (isinstance(item, Mapping) and bool(item)):
                return True
        if isinstance(item, Mapping) and normalized in {"details", "metadata", "model"}:
            if _remote_metadata(item):
                return True
    return False


def _is_ollama_cloud(
    host: str,
    model: str,
    tag_metadata: Mapping[str, Any],
    show_metadata: Mapping[str, Any],
) -> bool:
    return (
        _host_is_ollama_cloud(host)
        or _cloud_name(model)
        or _remote_metadata(tag_metadata)
        or _remote_metadata(show_metadata)
    )


def _capabilities(shown: Mapping[str, Any]) -> tuple[str, ...]:
    raw = shown.get("capabilities")
    if not isinstance(raw, (list, tuple, set, frozenset)):
        return ()
    return tuple(
        dict.fromkeys(
            str(value).strip().casefold()
            for value in raw
            if str(value).strip()
        )
    )


def _supports_tools(capabilities: tuple[str, ...]) -> bool:
    return any(value in {"tools", "tool_calling", "tool-calling"} for value in capabilities)


def _nested_value(value: Mapping[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _configured(value: str | None) -> bool:
    return bool(value and str(value).strip())


def _secret_free_metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in dict(value).items():
        normalized = str(key).casefold()
        if any(marker in normalized for marker in ("key", "token", "secret", "password", "authorization")):
            continue
        if isinstance(item, Mapping):
            result[str(key)] = _secret_free_metadata(item)
        elif isinstance(item, (str, int, float, bool)) or item is None:
            result[str(key)] = item
    return result


def _urllib_json(
    method: str,
    url: str,
    *,
    payload: Mapping[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
    timeout: float = DEFAULT_PROBE_TIMEOUT,
) -> Mapping[str, Any]:
    body = None
    request_headers = dict(headers or {})
    if payload is not None:
        body = json.dumps(dict(payload), ensure_ascii=False).encode("utf-8")
        request_headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        url,
        data=body,
        headers=request_headers,
        method=method.upper(),
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"could not reach {urllib.parse.urlsplit(url).netloc}: {exc}") from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise RuntimeError("model endpoint returned invalid JSON") from exc
    if not isinstance(data, Mapping):
        raise RuntimeError("model endpoint returned non-object JSON")
    return data


__all__ = [
    "CatalogDiagnostic",
    "ExecutionClass",
    "ModelCatalog",
    "ModelDescriptor",
]
