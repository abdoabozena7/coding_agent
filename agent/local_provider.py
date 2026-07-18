"""Capability negotiation and structured diagnostics for local providers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import re
from typing import Any, Mapping
import json
import urllib.error
import urllib.request

from .models import utc_now


def extract_first_json_object(text: str) -> Mapping[str, Any] | None:
    """Extract one balanced JSON object without trusting surrounding prose."""
    source = str(text or "")
    decoder = json.JSONDecoder()
    for index, character in enumerate(source):
        if character != "{":
            continue
        try:
            value, _end = decoder.raw_decode(source[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping):
            return value
    return None


def repair_structured_json_object(text: str) -> tuple[Mapping[str, Any] | None, tuple[str, ...]]:
    """Apply bounded transport repairs to a weak-model JSON envelope.

    Semantic fields are never invented here; typed validation remains the
    caller's job. The repairs cover only malformed envelope keys, trailing
    commas, and illegal control characters observed from local providers.
    """

    source = str(text or "").strip()
    if source.startswith("```"):
        source = re.sub(r"^```(?:json)?\s*", "", source, flags=re.IGNORECASE)
        source = re.sub(r"\s*```$", "", source)
    direct = extract_first_json_object(source)
    if direct is not None:
        return direct, ()
    actions: list[str] = []
    repaired = source
    key_pattern = re.compile(
        r'"(payload|summary|reasoning_summary|insights|tool_calls|artifacts|evidence|findings|issues|test_results)\\+"\s*:'
    )
    normalized = key_pattern.sub(lambda match: f'"{match.group(1)}":', repaired)
    if normalized != repaired:
        repaired = normalized
        actions.append("removed stray backslash from a known response-envelope key")
    normalized = re.sub(r",\s*([}\]])", r"\1", repaired)
    if normalized != repaired:
        repaired = normalized
        actions.append("removed trailing JSON comma")
    normalized = "".join(char for char in repaired if char in "\r\n\t" or ord(char) >= 32)
    if normalized != repaired:
        repaired = normalized
        actions.append("removed invalid JSON control characters")
    return extract_first_json_object(repaired), tuple(actions)


def normalize_action_proposal(value: Mapping[str, Any]) -> tuple[str, dict[str, Any]] | None:
    name = str(value.get("name") or value.get("tool") or value.get("action") or "").strip()
    args = value.get("args", value.get("arguments", {}))
    if not name or not isinstance(args, Mapping):
        return None
    return name, dict(args)


def normalize_generated_tool_args(name: str, args: Mapping[str, Any]) -> dict[str, Any]:
    """Repair layout escapes from weak-model tool transports without touching code strings.

    A few models emit a native ``write_file`` argument whose document separators
    are the two literal characters ``\\n``.  The JSON layer has already been
    decoded at that point, so writing the value verbatim produces a one-line,
    invalid artifact.  Only source-like full documents with almost no real line
    breaks are eligible.  Escapes inside quoted source strings remain escapes.
    """

    normalized = dict(args)
    field = "content" if str(name) == "write_file" else "new_str" if str(name) == "edit_file" else ""
    if not field or not isinstance(normalized.get(field), str):
        return normalized
    source = str(normalized[field])
    lowered = source.lstrip().casefold()
    source_like = lowered.startswith(("<!doctype", "<html", "<?xml")) or any(
        token in source for token in ("\\nimport ", "\\ndef ", "\\nclass ", "\\nfunction ")
    )
    if not source_like or source.count("\n") >= 2 or source.count(r"\n") < 4:
        return normalized

    output: list[str] = []
    quote: str | None = None
    escaped = False
    index = 0
    while index < len(source):
        char = source[index]
        if quote is not None:
            output.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            index += 1
            continue
        if char in {"'", '"', "`"}:
            quote = char
            output.append(char)
            index += 1
            continue
        if char == "\\" and index + 1 < len(source):
            marker = source[index + 1]
            if marker == "n":
                output.append("\n")
                index += 2
                continue
            if marker == "r":
                if index + 3 < len(source) and source[index + 2 : index + 4] == r"\n":
                    output.append("\n")
                    index += 4
                else:
                    output.append("\r")
                    index += 2
                continue
            if marker == "t":
                output.append("\t")
                index += 2
                continue
        output.append(char)
        index += 1
    normalized[field] = "".join(output)
    return normalized


class ProviderFailureKind(str, Enum):
    DNS_OR_SOCKET = "dns_or_socket_failure"
    CONNECTION_REFUSED = "connection_refused"
    TIMEOUT = "request_timeout"
    HTTP_4XX = "http_4xx"
    HTTP_5XX = "http_5xx"
    ENDPOINT_NOT_FOUND = "endpoint_not_found"
    MODEL_NOT_INSTALLED = "model_not_installed"
    MODEL_LOAD_FAILED = "model_load_failed"
    INVALID_PAYLOAD = "invalid_request_payload"
    UNSUPPORTED_PARAMETER = "unsupported_parameter"
    UNSUPPORTED_TOOLS = "unsupported_tool_calling"
    UNSUPPORTED_STRUCTURED_OUTPUT = "unsupported_structured_output"
    CONTEXT_LIMIT = "context_limit_exceeded"
    MALFORMED_STREAM = "malformed_streamed_response"
    INVALID_TYPED_OUTPUT = "invalid_typed_output"


_SECRET = re.compile(r"(?i)(api[_-]?key|token|password|authorization)(\s*[:=]\s*)([^\s,}\]]+)")


def redact_provider_message(value: str) -> str:
    return _SECRET.sub(lambda m: m.group(1) + m.group(2) + "[REDACTED]", str(value))


@dataclass(frozen=True, slots=True)
class ProviderDiagnostic:
    reachable: bool
    kind: ProviderFailureKind
    operation: str
    status_code: int | None = None
    provider_message: str = ""
    endpoint: str = ""
    incompatible_field: str | None = None


class ProviderRequestError(RuntimeError):
    def __init__(self, diagnostic: ProviderDiagnostic):
        self.diagnostic = diagnostic
        status = f" HTTP {diagnostic.status_code}" if diagnostic.status_code else ""
        super().__init__(f"Ollama request rejected{status}: {diagnostic.provider_message}".strip())


@dataclass(frozen=True, slots=True)
class ModelCapabilityProfile:
    model_name: str
    provider: str = "ollama"
    base_url: str = "http://localhost:11434"
    api_protocol: str = "native_chat"
    endpoint: str = "/api/chat"
    chat_support: bool = True
    completion_support: bool = False
    tool_call_support: bool = False
    structured_output_support: bool = False
    vision_support: bool = False
    embedding_support: bool = False
    streaming_support: bool = True
    thinking_support: bool = False
    context_size: int | None = None
    maximum_output_size: int | None = None
    known_unsupported_parameters: tuple[str, ...] = ()
    health_status: str = "unknown"
    last_successful_probe: datetime | None = None
    probe_evidence: Mapping[str, Any] = field(default_factory=dict)
    model_fingerprint: str = ""


class OllamaRequestCompiler:
    def compile(self, profile: ModelCapabilityProfile, *, messages: list[dict[str, Any]], tools=(), stream=True, structured=False, options: Mapping[str, Any] | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": profile.model_name, "messages": messages, "stream": bool(stream)}
        unsupported = set(profile.known_unsupported_parameters)
        if tools and profile.tool_call_support and "tools" not in unsupported:
            payload["tools"] = list(tools)
        if structured and profile.structured_output_support and "format" not in unsupported:
            payload["format"] = "json"
        for key, value in (options or {}).items():
            if key not in unsupported:
                payload[key] = value
        return payload


class OllamaHandshake:
    """Probe only safe metadata and minimal generation endpoints."""

    def __init__(self, base_url: str = "http://localhost:11434", timeout: float = 10):
        self.base_url, self.timeout = base_url.rstrip("/"), timeout

    def _json(self, path: str, payload: Mapping[str, Any] | None = None) -> tuple[int, Any]:
        data = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(self.base_url + path, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return int(response.status), json.loads(response.read().decode())
        except urllib.error.HTTPError as error:
            body = error.read().decode(errors="replace")
            raise ProviderRequestError(ProviderDiagnostic(True, ProviderFailureKind.HTTP_4XX if error.code < 500 else ProviderFailureKind.HTTP_5XX, "probe", error.code, redact_provider_message(body), self.base_url + path)) from error
        except urllib.error.URLError as error:
            reason = error.reason
            if isinstance(reason, (TimeoutError, __import__("socket").timeout)):
                kind = ProviderFailureKind.TIMEOUT
            elif isinstance(reason, ConnectionRefusedError) or "refused" in str(reason).casefold():
                kind = ProviderFailureKind.CONNECTION_REFUSED
            else:
                kind = ProviderFailureKind.DNS_OR_SOCKET
            raise ProviderRequestError(ProviderDiagnostic(False, kind, "probe", provider_message=redact_provider_message(str(reason)), endpoint=self.base_url + path)) from error

    def probe(self, model: str) -> ModelCapabilityProfile:
        version_status, version = self._json("/api/version")
        tags_status, tags = self._json("/api/tags")
        models = {str(item.get("name") or item.get("model")): item for item in tags.get("models", [])}
        if model not in models:
            raise ProviderRequestError(ProviderDiagnostic(True, ProviderFailureKind.MODEL_NOT_INSTALLED, "model_lookup", 404, f"model {model!r} is not installed", self.base_url + "/api/tags"))
        metadata = models[model]
        capabilities = set(metadata.get("capabilities") or ())
        details = metadata.get("details") or {}
        return ModelCapabilityProfile(
            model_name=model, base_url=self.base_url, endpoint="/api/chat", api_protocol="native_chat", chat_support="completion" in capabilities,
            completion_support="completion" in capabilities, tool_call_support="tools" in capabilities,
            thinking_support="thinking" in capabilities,
            vision_support="vision" in capabilities, embedding_support="embedding" in capabilities,
            context_size=details.get("context_length"), health_status="reachable",
            last_successful_probe=utc_now(), model_fingerprint=str(metadata.get("digest") or ""),
            probe_evidence={"base_url": self.base_url, "version_status": version_status,
                            "tags_status": tags_status, "ollama_version": version.get("version"),
                            "capabilities": sorted(capabilities)},
        )
