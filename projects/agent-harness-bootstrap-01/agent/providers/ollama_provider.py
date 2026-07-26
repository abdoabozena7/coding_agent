"""Ollama chat adapter, including local and Ollama cloud models."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
import socket
from dataclasses import replace
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
from ..local_provider import (
    ModelCapabilityProfile,
    OllamaRequestCompiler,
    OllamaHandshake,
    ProviderDiagnostic,
    ProviderFailureKind,
    ProviderRequestError,
    redact_provider_message,
)

MODEL_NAME = "gpt-oss:120b-cloud"
DEFAULT_HOST = "http://localhost:11434"


class OllamaProvider:
    _capability_cache: dict[tuple[str, str], ModelCapabilityProfile] = {}
    capabilities = ProviderCapabilities(
        streaming=True,
        tool_calling=True,
        thinking=True,
        # Ollama does not guarantee native call IDs, although this adapter
        # synthesizes stable neutral IDs for the harness.
        tool_call_ids=False,
        native_replay=True,
    )

    def __init__(self, model: str | None = None, host: str | None = None, capability_profile: ModelCapabilityProfile | None = None, reasoning_effort: str = "medium", context_size: int | None = None, num_gpu: int | None = None, max_output_tokens: int | None = None, force_json: bool = False, temperature: float | None = None, request_timeout: float | None = None, require_gpu: bool | None = None):
        self.model = model or os.getenv("OLLAMA_MODEL") or MODEL_NAME
        self.host = (host or os.getenv("OLLAMA_HOST", DEFAULT_HOST)).rstrip("/")
        self.capability_profile = capability_profile or ModelCapabilityProfile(
            model_name=self.model, base_url=self.host, endpoint="/api/chat",
            tool_call_support=False, structured_output_support=False,
            health_status="unprobed",
        )
        self.request_compiler = OllamaRequestCompiler()
        self.reasoning_effort = str(reasoning_effort)
        raw_context = context_size if context_size is not None else os.getenv("OLLAMA_CONTEXT_SIZE", "16384")
        try:
            parsed_context = int(raw_context)
        except (TypeError, ValueError):
            parsed_context = 16_384
        self.context_size = min(131_072, max(2_048, parsed_context))
        raw_num_gpu = num_gpu if num_gpu is not None else os.getenv("OLLAMA_NUM_GPU")
        try:
            self.num_gpu = None if raw_num_gpu in {None, ""} else max(0, int(raw_num_gpu))
        except (TypeError, ValueError):
            self.num_gpu = None
        hostname = (urllib.parse.urlsplit(self.host).hostname or "").casefold()
        remote_model = self.model.casefold().endswith((":cloud", "-cloud"))
        remote_host = hostname == "ollama.com" or hostname.endswith(".ollama.com")
        configured_gpu_requirement = str(
            os.getenv("AGENT_REQUIRE_LOCAL_GPU", "")
        ).strip().casefold() in {"1", "true", "yes", "on"}
        # AGENT_REQUIRE_LOCAL_GPU protects local inference only. Hosted Ollama
        # models have no local runner entry in /api/ps, so applying the guard to
        # them creates a deterministic failure that can never recover.
        self.require_gpu = bool(
            (configured_gpu_requirement if require_gpu is None else require_gpu)
            and not remote_model
            and not remote_host
        )
        if self.require_gpu and self.num_gpu is None:
            # Ask Ollama to offload every layer. The post-call residency check
            # below fails closed if the runner silently falls back to CPU.
            self.num_gpu = 999
        self.max_output_tokens = (
            None if max_output_tokens is None else min(65_536, max(128, int(max_output_tokens)))
        )
        raw_timeout = (
            request_timeout
            if request_timeout is not None
            else os.getenv("OLLAMA_REQUEST_TIMEOUT")
        )
        self._request_timeout_explicit = raw_timeout not in {None, ""}
        if raw_timeout in {None, ""}:
            # Local generation time grows with the requested output. A fixed
            # five-minute transport deadline can discard a valid GPU result
            # just before Ollama returns it, especially on small local models.
            predicted = 300.0
            if self.max_output_tokens is not None:
                predicted = 120.0 + (self.max_output_tokens * 0.20)
            self.request_timeout = max(300.0, min(900.0, predicted))
        else:
            try:
                self.request_timeout = max(30.0, min(1_800.0, float(raw_timeout)))
            except (TypeError, ValueError):
                self.request_timeout = 300.0
        raw_temperature = (
            temperature
            if temperature is not None
            else os.getenv("OLLAMA_TEMPERATURE")
        )
        try:
            self.temperature = (
                None
                if raw_temperature in {None, ""}
                else max(0.0, min(2.0, float(raw_temperature)))
            )
        except (TypeError, ValueError):
            self.temperature = None
        self.force_json = bool(force_json)

    def _ensure_capabilities(self) -> None:
        if self.model == "offline" or self.capability_profile.health_status != "unprobed":
            return
        key = (self.host, self.model)
        cached = self._capability_cache.get(key)
        if cached is None:
            cached = OllamaHandshake(self.host).probe(self.model)
            self._capability_cache[key] = cached
        self.capability_profile = cached

    def _post_json(self, path: str, payload: dict):
        data = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        request = urllib.request.Request(
            f"{self.host}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            timeout = self.request_timeout
            if not self._request_timeout_explicit and self.max_output_tokens is not None:
                timeout = max(
                    300.0,
                    min(900.0, 120.0 + (self.max_output_tokens * 0.20)),
                )
            return urllib.request.urlopen(request, timeout=timeout)
        except urllib.error.HTTPError as error:
            try:
                body = error.read().decode("utf-8", errors="replace")
            except Exception:
                body = str(error.reason or "")
            safe_body = redact_provider_message(body)
            status = int(error.code)
            lowered = safe_body.casefold()
            kind = ProviderFailureKind.HTTP_4XX if 400 <= status < 500 else ProviderFailureKind.HTTP_5XX
            if status == 404:
                kind = ProviderFailureKind.MODEL_NOT_INSTALLED if "model" in lowered else ProviderFailureKind.ENDPOINT_NOT_FOUND
            elif ("load model" in lowered or "loading model" in lowered or "unable to load" in lowered):
                kind = ProviderFailureKind.MODEL_LOAD_FAILED
            elif "context" in lowered and ("limit" in lowered or "length" in lowered):
                kind = ProviderFailureKind.CONTEXT_LIMIT
            elif "tool" in lowered and ("support" in lowered or "invalid" in lowered):
                kind = ProviderFailureKind.UNSUPPORTED_TOOLS
            elif "format" in lowered or "response_format" in lowered:
                kind = ProviderFailureKind.UNSUPPORTED_STRUCTURED_OUTPUT
            elif "unknown field" in lowered or "unsupported parameter" in lowered:
                kind = ProviderFailureKind.UNSUPPORTED_PARAMETER
            elif "does not support" in lowered or "not supported" in lowered:
                kind = ProviderFailureKind.UNSUPPORTED_PARAMETER
            elif "invalid" in lowered and ("json" in lowered or "payload" in lowered or "request" in lowered):
                kind = ProviderFailureKind.INVALID_PAYLOAD
            field = None
            match = __import__("re").search(r"(?:field|parameter)\s+['\"]?([\w.-]+)", safe_body, __import__("re").I)
            if match:
                field = match.group(1)
            elif "thinking" in lowered or " think" in lowered:
                field = "think"
            elif kind is ProviderFailureKind.UNSUPPORTED_TOOLS:
                field = "tools"
            elif kind is ProviderFailureKind.UNSUPPORTED_STRUCTURED_OUTPUT:
                field = "format"
            raise ProviderRequestError(ProviderDiagnostic(
                reachable=True, kind=kind, operation="POST", status_code=status,
                provider_message=safe_body, endpoint=f"{self.host}{path}", incompatible_field=field,
            )) from error
        except (TimeoutError, socket.timeout) as error:
            self._cancel_active_generation()
            raise ProviderRequestError(ProviderDiagnostic(False, ProviderFailureKind.TIMEOUT, "POST", provider_message=str(error), endpoint=f"{self.host}{path}")) from error
        except urllib.error.URLError as error:
            reason = error.reason
            kind = ProviderFailureKind.CONNECTION_REFUSED if isinstance(reason, ConnectionRefusedError) else ProviderFailureKind.DNS_OR_SOCKET
            raise ProviderRequestError(ProviderDiagnostic(False, kind, "POST", provider_message=redact_provider_message(str(reason)), endpoint=f"{self.host}{path}")) from error

    def _cancel_active_generation(self) -> None:
        """Cancel work Ollama keeps running after a client-side timeout.

        Closing the timed-out HTTP socket does not reliably stop local
        inference. Without an explicit unload, every retry queues behind the
        abandoned request while the GPU remains at 100 percent.
        """

        data = json.dumps(
            {"model": self.model, "keep_alive": 0, "stream": False}
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.host}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                response.read()
        except Exception:
            # Preserve the original, correctly classified timeout. The next
            # GPU-residency check still fails closed if cancellation failed.
            pass

    def reset_model_cache(self) -> None:
        """Unload a degraded local runner so the next GPU call gets a clean KV cache."""

        response = self._post_json(
            "/api/generate",
            {
                "model": self.model,
                "keep_alive": 0,
                "stream": False,
            },
        )
        with response:
            response.read()

    def _stream_error(self, message: Any) -> ProviderRequestError:
        """Normalize an Ollama NDJSON error into the provider error contract."""

        safe = redact_provider_message(str(message or "Ollama stream failed"))
        lowered = safe.casefold()
        if any(token in lowered for token in ("cuda error", "illegal memory access", "runner process")):
            kind = ProviderFailureKind.MODEL_LOAD_FAILED
        elif any(
            token in lowered
            for token in (
                "unexpected empty grammar stack",
                "grammar stack",
                "accepting piece",
                "structured output",
            )
        ):
            kind = ProviderFailureKind.INVALID_TYPED_OUTPUT
        else:
            kind = ProviderFailureKind.MALFORMED_STREAM
        return ProviderRequestError(
            ProviderDiagnostic(
                reachable=True,
                kind=kind,
                operation="parse_stream",
                provider_message=safe,
                endpoint=f"{self.host}/api/chat",
            )
        )

    @staticmethod
    def _runner_recovery_allowed(error: ProviderRequestError) -> bool:
        message = str(error.diagnostic.provider_message or "").casefold()
        return error.diagnostic.kind in {
            ProviderFailureKind.MODEL_LOAD_FAILED,
            ProviderFailureKind.INVALID_TYPED_OUTPUT,
        } or any(
            token in message
            for token in (
                "cuda error",
                "illegal memory access",
                "unexpected empty grammar stack",
                "accepting piece",
            )
        )

    def _assert_gpu_residency(self) -> None:
        """Fail closed unless the active model is fully resident on the GPU."""

        if not self.require_gpu:
            return
        request = urllib.request.Request(
            f"{self.host}/api/ps",
            headers={"Accept": "application/json"},
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                data = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            raise RuntimeError(
                "GPU-only Ollama execution could not verify /api/ps residency"
            ) from exc
        models = data.get("models", ()) if isinstance(data, Mapping) else ()
        normalized_target = self.model.casefold()
        active = next(
            (
                item
                for item in models
                if isinstance(item, Mapping)
                and str(item.get("name") or item.get("model") or "").casefold()
                == normalized_target
            ),
            None,
        )
        if not isinstance(active, Mapping):
            raise RuntimeError(
                f"GPU-only Ollama execution cannot find active model {self.model!r}"
            )
        try:
            total_size = int(active.get("size") or 0)
            vram_size = int(active.get("size_vram") or 0)
        except (TypeError, ValueError):
            total_size = 0
            vram_size = 0
        if total_size <= 0 or vram_size < int(total_size * 0.98):
            ratio = (vram_size / total_size) if total_size > 0 else 0.0
            raise RuntimeError(
                "GPU-only Ollama execution rejected partial/CPU offload: "
                f"model={self.model}, vram={vram_size}, size={total_size}, "
                f"gpu_fraction={ratio:.3f}"
            )

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
        self._ensure_capabilities()
        tool_specs = tuple(tools or ())
        payload = self.request_compiler.compile(
            self.capability_profile,
            messages=self._to_messages(conversation, system),
            tools=tool_specs,
            # Ollama emits native tool arguments as one completed assistant
            # message. Several thinking models lose that final call when the
            # endpoint is asked for NDJSON token streaming.
            stream=not bool(tool_specs),
            options=(
                {
                    "think": (
                        False
                        if self.reasoning_effort.casefold() in {"off", "none", "false"}
                        else "high" if self.reasoning_effort == "xhigh" else self.reasoning_effort
                    )
                }
                if self.capability_profile.thinking_support else {}
            ),
        )
        if self.force_json and "format" not in self.capability_profile.known_unsupported_parameters:
            payload["format"] = "json"
        # Model metadata may advertise a 128K+ default even when the harness
        # sends a deliberately narrow weak-model context.  Explicitly bound the
        # KV cache so local inference does not waste memory or fail at startup.
        payload["options"] = {"num_ctx": self.context_size}
        if self.num_gpu is not None:
            payload["options"]["num_gpu"] = self.num_gpu
        if self.max_output_tokens is not None:
            payload["options"]["num_predict"] = self.max_output_tokens
        if self.temperature is not None:
            payload["options"]["temperature"] = self.temperature

        text_parts: list[str] = []
        thought_parts: list[str] = []
        raw_tool_calls: list[Mapping[str, Any]] = []
        usage = None
        malformed_chunks = 0
        valid_chunks = 0

        runner_replayed = False
        try:
            response = self._post_json("/api/chat", payload)
        except ProviderRequestError as error:
            if self._runner_recovery_allowed(error):
                # Keep the execution class GPU-only: unload the corrupted
                # runner/KV cache, then replay the exact governed request once
                # so Ollama reloads it with num_gpu unchanged.
                self.reset_model_cache()
                response = self._post_json("/api/chat", payload)
                runner_replayed = True
            else:
                field = error.diagnostic.incompatible_field
                adaptable = field in {"think", "tools", "format"} and field in payload
                if not adaptable:
                    raise
                unsupported = tuple(dict.fromkeys((*self.capability_profile.known_unsupported_parameters, field)))
                self.capability_profile = replace(
                    self.capability_profile,
                    tool_call_support=False if field == "tools" else self.capability_profile.tool_call_support,
                    structured_output_support=False if field == "format" else self.capability_profile.structured_output_support,
                    thinking_support=False if field == "think" else self.capability_profile.thinking_support,
                    known_unsupported_parameters=unsupported,
                )
                adapted_payload = dict(payload)
                adapted_payload.pop(field, None)
                response = self._post_json("/api/chat", adapted_payload)

        def consume(stream: Any) -> None:
            nonlocal malformed_chunks, valid_chunks, usage
            with stream:
                for raw_line in stream:
                    if not raw_line or not raw_line.strip():
                        continue
                    try:
                        chunk = json.loads(raw_line)
                    except (json.JSONDecodeError, UnicodeDecodeError, TypeError):
                        # A truncated NDJSON line should not take down a long run;
                        # later chunks may still contain a complete answer/call.
                        malformed_chunks += 1
                        continue
                    if not isinstance(chunk, Mapping):
                        malformed_chunks += 1
                        continue
                    valid_chunks += 1
                    if chunk.get("error"):
                        raise self._stream_error(chunk["error"])
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

        try:
            consume(response)
        except ProviderRequestError as error:
            # Replaying after user-visible fragments would duplicate streamed
            # content because the neutral callback contract has no rollback.
            # Fail cleanly in that rare case; the workspace finalizer discards
            # the uncommitted attempt and offers an explicit retry.
            if (
                runner_replayed
                or not self._runner_recovery_allowed(error)
                or text_parts
                or thought_parts
                or raw_tool_calls
            ):
                raise
            self.reset_model_cache()
            runner_replayed = True
            malformed_chunks = 0
            valid_chunks = 0
            consume(self._post_json("/api/chat", payload))

        if malformed_chunks and not valid_chunks:
            raise ProviderRequestError(ProviderDiagnostic(
                reachable=True, kind=ProviderFailureKind.MALFORMED_STREAM,
                operation="parse_stream", provider_message="Ollama returned no valid NDJSON chunks",
                endpoint=f"{self.host}/api/chat",
            ))
        self._assert_gpu_residency()

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
