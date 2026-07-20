"""Offline contract and regression tests for every provider adapter."""

from __future__ import annotations

import io
import json
import socket
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from agent.providers import AssistantTurn, ProviderCapabilities, ToolCall, Usage
from agent.providers.base import coerce_tool_args
from agent.providers.gemini_provider import GeminiProvider
from agent.providers.ollama_provider import OllamaProvider
from agent.local_provider import ProviderFailureKind, ProviderRequestError
from agent.providers.openai_provider import OpenAIProvider
from agent.testing import ScriptedProvider, ScriptedTurn


def ns(**values):
    return SimpleNamespace(**values)


class NeutralTypesTests(unittest.TestCase):
    def test_native_replay_metadata_survives_without_aliasing(self):
        turn = AssistantTurn(
            text="done",
            tool_calls=[
                ToolCall(
                    id="call-1",
                    name="read_file",
                    args={"path": "a.py"},
                    native={"provider": "example", "opaque": [1]},
                )
            ],
            native={"provider": "example", "opaque": {"signature": "abc"}},
        )

        message = turn.to_message()
        message["native"]["opaque"]["signature"] = "changed"
        message["tool_calls"][0]["native"]["opaque"].append(2)

        self.assertEqual(turn.native["opaque"]["signature"], "abc")
        self.assertEqual(turn.tool_calls[0].native["opaque"], [1])

    def test_malformed_and_non_object_tool_arguments_become_empty_dicts(self):
        self.assertEqual(coerce_tool_args('{"path": "x"}'), {"path": "x"})
        self.assertEqual(coerce_tool_args('{"path":'), {})
        self.assertEqual(coerce_tool_args("[1, 2]"), {})
        self.assertEqual(coerce_tool_args(object()), {})

    def test_capability_aliases_are_explicit(self):
        capabilities = ProviderCapabilities(tool_calling=True, thinking=True)
        self.assertTrue(capabilities.supports_tools)
        self.assertTrue(capabilities.supports_tool_calling)
        self.assertTrue(capabilities.supports_thinking)
        self.assertFalse(capabilities.supports_streaming)


class OpenAIProviderTests(unittest.TestCase):
    def test_stream_repairs_missing_id_and_malformed_arguments(self):
        chunks = [
            ns(
                usage=None,
                choices=[
                    ns(
                        delta=ns(
                            content="working",
                            tool_calls=[
                                ns(
                                    index=0,
                                    id=None,
                                    function=ns(
                                        name="read_file",
                                        arguments='{"path":',
                                    ),
                                )
                            ],
                        )
                    )
                ],
            ),
            ns(
                usage=ns(
                    prompt_tokens=10,
                    completion_tokens=3,
                    prompt_tokens_details=ns(cached_tokens=4),
                ),
                choices=[],
            ),
        ]
        provider = OpenAIProvider(model="offline")
        create = Mock(return_value=chunks)
        provider._client = ns(chat=ns(completions=ns(create=create)))
        streamed = []

        turn = provider.call([], [], "system", on_text=streamed.append)

        self.assertEqual(streamed, ["working"])
        self.assertEqual(turn.text, "working")
        self.assertEqual(len(turn.tool_calls), 1)
        self.assertEqual(turn.tool_calls[0].id, "openai-call-0-read_file")
        self.assertEqual(turn.tool_calls[0].args, {})
        self.assertEqual(turn.usage, Usage(10, 4, 3))
        self.assertNotIn("tools", create.call_args.kwargs)

    def test_history_pairs_missing_assistant_and_result_ids(self):
        provider = OpenAIProvider(model="offline")
        messages = provider._to_messages(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"name": "read_file", "args": {"path": "x"}}],
                },
                {"role": "tool", "name": "read_file", "content": "ok"},
            ],
            "system",
        )
        assistant_id = messages[1]["tool_calls"][0]["id"]
        self.assertTrue(assistant_id)
        self.assertEqual(messages[2]["tool_call_id"], assistant_id)

    def test_stream_tolerates_mapping_chunks_and_non_object_args(self):
        stream = [
            {
                "usage": None,
                "choices": [
                    {
                        "delta": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "index": None,
                                    "id": "",
                                    "function": {
                                        "name": "run",
                                        "arguments": "[1]",
                                    },
                                }
                            ],
                        }
                    }
                ],
            }
        ]
        provider = OpenAIProvider(model="offline")
        provider._client = ns(
            chat=ns(completions=ns(create=Mock(return_value=stream)))
        )
        turn = provider.call([], [], "")
        self.assertEqual(turn.tool_calls[0].args, {})
        self.assertTrue(turn.tool_calls[0].id)


class GeminiProviderTests(unittest.TestCase):
    def _provider_with_stream(self, parts):
        chunk = ns(
            usage_metadata=ns(
                prompt_token_count=8,
                cached_content_token_count=2,
                candidates_token_count=5,
            ),
            candidates=[ns(content=ns(parts=parts))],
        )
        provider = GeminiProvider(model="offline")
        provider._client = ns(
            models=ns(
                generate_content_stream=Mock(return_value=[chunk]),
                generate_content=Mock(return_value=ns(text="summary")),
            )
        )
        return provider

    def test_ids_signatures_and_function_response_history_round_trip(self):
        parts = [
            ns(
                function_call=None,
                text="considering",
                thought=True,
                thought_signature=b"thought-signature",
            ),
            ns(
                function_call=ns(id="gem-call-7", name="read_file", args={"path": "x"}),
                text=None,
                thought=False,
                thought_signature=b"call-signature",
            ),
            ns(
                function_call=None,
                text="visible",
                thought=False,
                thought_signature=None,
            ),
        ]
        provider = self._provider_with_stream(parts)
        thoughts, text = [], []

        turn = provider.call([], [], "system", text.append, thoughts.append)

        self.assertEqual(thoughts, ["considering"])
        self.assertEqual(text, ["visible"])
        self.assertEqual(turn.tool_calls[0].id, "gem-call-7")
        self.assertEqual(turn.usage, Usage(8, 2, 5))
        self.assertIn("native", turn.to_message())

        contents = provider._to_contents(
            [
                turn.to_message(),
                {
                    "role": "tool",
                    "id": "gem-call-7",
                    "name": "read_file",
                    "content": "ok",
                },
            ]
        )
        model_parts = contents[0].parts
        self.assertEqual(model_parts[0].thought_signature, b"thought-signature")
        self.assertEqual(model_parts[1].function_call.id, "gem-call-7")
        self.assertEqual(model_parts[1].thought_signature, b"call-signature")
        self.assertEqual(contents[1].role, "user")
        response = contents[1].parts[0].function_response
        self.assertEqual(response.id, "gem-call-7")
        self.assertEqual(response.name, "read_file")
        self.assertEqual(response.response, {"result": "ok"})

    def test_missing_ids_are_unique_and_malformed_args_do_not_crash(self):
        provider = self._provider_with_stream(
            [
                ns(
                    function_call=ns(id=None, name="same", args='{"bad":'),
                    text=None,
                    thought=False,
                    thought_signature=None,
                ),
                ns(
                    function_call=ns(id=None, name="same", args=[1]),
                    text=None,
                    thought=False,
                    thought_signature=None,
                ),
            ]
        )
        turn = provider.call([], [], "")
        self.assertEqual([call.args for call in turn.tool_calls], [{}, {}])
        self.assertEqual(len({call.id for call in turn.tool_calls}), 2)

    def test_adjacent_function_results_are_grouped_as_one_user_content(self):
        provider = GeminiProvider(model="offline")
        contents = provider._to_contents(
            [
                {"role": "tool", "id": "1", "name": "a", "content": "A"},
                {"role": "tool", "id": "2", "name": "b", "content": "B"},
            ]
        )
        self.assertEqual(len(contents), 1)
        self.assertEqual(contents[0].role, "user")
        self.assertEqual(len(contents[0].parts), 2)


class OllamaProviderTests(unittest.TestCase):
    def test_request_timeout_scales_with_local_output_budget(self):
        provider = OllamaProvider(model="offline")
        self.assertAlmostEqual(provider.request_timeout, 300.0)
        provider.max_output_tokens = 2048

        with patch(
            "agent.providers.ollama_provider.urllib.request.urlopen",
            return_value=io.BytesIO(b"{}"),
        ) as opened:
            provider._post_json("/api/chat", {})
        self.assertAlmostEqual(opened.call_args.kwargs["timeout"], 529.6)

    def test_timeout_cancels_abandoned_ollama_generation_before_retry(self):
        provider = OllamaProvider(model="offline", request_timeout=30)
        with patch(
            "agent.providers.ollama_provider.urllib.request.urlopen",
            side_effect=[socket.timeout("timed out"), io.BytesIO(b"{}")],
        ) as opened:
            with self.assertRaises(ProviderRequestError):
                provider._post_json("/api/chat", {"messages": []})

        self.assertEqual(opened.call_count, 2)
        cleanup_request = opened.call_args_list[1].args[0]
        self.assertTrue(cleanup_request.full_url.endswith("/api/generate"))
        cleanup_payload = json.loads(cleanup_request.data.decode("utf-8"))
        self.assertEqual(cleanup_payload["keep_alive"], 0)
        self.assertFalse(cleanup_payload["stream"])

    def test_gpu_required_mode_rejects_partial_cpu_offload(self):
        provider = OllamaProvider(model="offline", num_gpu=999)
        provider.require_gpu = True
        payload = {
            "models": [
                {
                    "name": "offline",
                    "size": 4_000,
                    "size_vram": 2_000,
                }
            ]
        }
        with patch(
            "agent.providers.ollama_provider.urllib.request.urlopen",
            return_value=io.BytesIO(json.dumps(payload).encode()),
        ):
            with self.assertRaisesRegex(RuntimeError, "partial/CPU offload"):
                provider._assert_gpu_residency()

    def test_gpu_required_mode_accepts_full_residency(self):
        provider = OllamaProvider(model="offline", num_gpu=999)
        provider.require_gpu = True
        payload = {
            "models": [
                {
                    "name": "offline",
                    "size": 4_000,
                    "size_vram": 4_000,
                }
            ]
        }
        with patch(
            "agent.providers.ollama_provider.urllib.request.urlopen",
            return_value=io.BytesIO(json.dumps(payload).encode()),
        ):
            provider._assert_gpu_residency()

    def test_local_context_is_bounded_in_every_chat_request(self):
        provider = OllamaProvider(model="offline", context_size=16_384, num_gpu=0, max_output_tokens=2048, force_json=True, temperature=0.25)
        provider.capability_profile = __import__("agent.local_provider", fromlist=["ModelCapabilityProfile"]).ModelCapabilityProfile(
            "offline", tool_call_support=True, thinking_support=True, health_status="reachable"
        )
        provider._post_json = Mock(return_value=io.BytesIO(b'{"message":{"content":"ok"},"done":true}\n'))

        provider.call([], [], "system")

        payload = provider._post_json.call_args.args[1]
        self.assertEqual(payload["options"]["num_ctx"], 16_384)
        self.assertEqual(payload["options"]["num_gpu"], 0)
        self.assertEqual(payload["options"]["num_predict"], 2048)
        self.assertEqual(payload["options"]["temperature"], 0.25)
        self.assertEqual(payload["think"], "medium")
        self.assertEqual(payload["format"], "json")

    def test_internal_off_reasoning_maps_to_ollama_think_false(self):
        provider = OllamaProvider(model="offline", reasoning_effort="off")
        provider.capability_profile = __import__("agent.local_provider", fromlist=["ModelCapabilityProfile"]).ModelCapabilityProfile(
            "offline", thinking_support=True, health_status="reachable"
        )
        provider._post_json = Mock(return_value=io.BytesIO(b'{"message":{"content":"ok"},"done":true}\n'))

        provider.call([], [], "system")

        self.assertIs(provider._post_json.call_args.args[1]["think"], False)

    def test_tool_requests_are_atomic_instead_of_token_streamed(self):
        provider = OllamaProvider(model="offline")
        provider.capability_profile = __import__(
            "agent.local_provider", fromlist=["ModelCapabilityProfile"]
        ).ModelCapabilityProfile(
            "offline", tool_call_support=True, health_status="reachable"
        )
        provider._post_json = Mock(
            return_value=io.BytesIO(
                b'{"message":{"tool_calls":[{"function":{"name":"x","arguments":{}}}]},"done":true}\n'
            )
        )

        turn = provider.call(
            [],
            [{"type": "function", "function": {"name": "x"}}],
            "system",
        )

        payload = provider._post_json.call_args.args[1]
        self.assertIs(payload["stream"], False)
        self.assertEqual(turn.tool_calls[0].name, "x")

    def test_unsupported_tools_rejection_is_adapted_once_without_faking_connectivity_failure(self):
        provider = OllamaProvider(model="offline")
        provider.capability_profile = __import__("agent.local_provider", fromlist=["ModelCapabilityProfile"]).ModelCapabilityProfile(
            "offline", tool_call_support=True, health_status="reachable"
        )
        rejection = ProviderRequestError(__import__("agent.local_provider", fromlist=["ProviderDiagnostic"]).ProviderDiagnostic(
            True, ProviderFailureKind.UNSUPPORTED_TOOLS, "POST", 400,
            "tools are not supported", "/api/chat", "tools",
        ))
        valid = io.BytesIO(b'{"message":{"content":"fallback"},"done":true}\n')
        provider._post_json = Mock(side_effect=[rejection, valid])

        turn = provider.call([], [{"type":"function","function":{"name":"x"}}], "system")

        self.assertEqual(turn.text, "fallback")
        self.assertEqual(provider._post_json.call_count, 2)
        self.assertIn("tools", provider.capability_profile.known_unsupported_parameters)
        self.assertNotIn("tools", provider._post_json.call_args_list[1].args[1])

    def test_cuda_runner_failure_unloads_then_retries_with_gpu_options_intact(self):
        provider = OllamaProvider(
            model="offline",
            context_size=4096,
            num_gpu=999,
        )
        provider.capability_profile = __import__(
            "agent.local_provider", fromlist=["ModelCapabilityProfile"]
        ).ModelCapabilityProfile("offline", health_status="reachable")
        cuda_failure = ProviderRequestError(
            __import__(
                "agent.local_provider", fromlist=["ProviderDiagnostic"]
            ).ProviderDiagnostic(
                True,
                ProviderFailureKind.HTTP_5XX,
                "POST",
                500,
                "CUDA error: illegal memory access",
                "/api/chat",
            )
        )
        unloaded = io.BytesIO(b'{"done":true}\n')
        valid = io.BytesIO(
            b'{"message":{"content":"recovered"},"done":true}\n'
        )
        provider._post_json = Mock(
            side_effect=[cuda_failure, unloaded, valid]
        )

        turn = provider.call([], [], "system")

        self.assertEqual(turn.text, "recovered")
        self.assertEqual(provider._post_json.call_count, 3)
        unload_payload = provider._post_json.call_args_list[1].args[1]
        retry_payload = provider._post_json.call_args_list[2].args[1]
        self.assertEqual(unload_payload["keep_alive"], 0)
        self.assertEqual(retry_payload["options"]["num_gpu"], 999)
        self.assertEqual(retry_payload["options"]["num_ctx"], 4096)

    def test_completely_malformed_stream_is_classified_not_silently_accepted(self):
        provider = OllamaProvider(model="offline")
        provider._post_json = Mock(return_value=io.BytesIO(b"not-json\n{broken\n"))
        with self.assertRaises(ProviderRequestError) as raised:
            provider.call([], [], "system")
        self.assertEqual(raised.exception.diagnostic.kind, ProviderFailureKind.MALFORMED_STREAM)

    def test_thinking_tool_name_and_bad_ndjson_are_robust(self):
        lines = [
            b"not-json\n",
            json.dumps(
                {
                    "message": {
                        "thinking": "inspect",
                        "content": "answer",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "read_file",
                                    "arguments": '{"path":',
                                }
                            }
                        ],
                    },
                    "done": True,
                    "prompt_eval_count": 11,
                    "eval_count": 6,
                }
            ).encode()
            + b"\n",
        ]
        provider = OllamaProvider(model="offline")
        provider._post_json = Mock(return_value=io.BytesIO(b"".join(lines)))
        thoughts, text = [], []

        turn = provider.call([], [], "system", text.append, thoughts.append)

        self.assertEqual(thoughts, ["inspect"])
        self.assertEqual(text, ["answer"])
        self.assertEqual(turn.tool_calls[0].args, {})
        self.assertTrue(turn.tool_calls[0].id)
        self.assertEqual(turn.usage, Usage(11, 0, 6))

        history = provider._to_messages(
            [
                turn.to_message(),
                {
                    "role": "tool",
                    "id": turn.tool_calls[0].id,
                    "name": "read_file",
                    "content": "ok",
                },
            ],
            "system",
        )
        self.assertEqual(history[1]["thinking"], "inspect")
        self.assertEqual(history[2]["tool_name"], "read_file")
        self.assertNotIn("tool_call_id", history[2])

    def test_native_ids_are_kept_and_summary_is_offline_mocked(self):
        provider = OllamaProvider(model="offline")
        payload = (
            json.dumps(
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "ollama-native",
                                "function": {
                                    "name": "x",
                                    "arguments": {"n": 1},
                                },
                            }
                        ]
                    },
                    "done": True,
                }
            ).encode()
            + b"\n"
        )
        provider._post_json = Mock(return_value=io.BytesIO(payload))
        self.assertEqual(provider.call([], [], "").tool_calls[0].id, "ollama-native")

        provider._post_json = Mock(
            return_value=io.BytesIO(b'{"message":{"content":" compact "}}')
        )
        self.assertEqual(provider.summarize([]), "compact")


class ScriptedProviderTests(unittest.TestCase):
    def test_records_requests_streams_fragments_and_never_needs_network(self):
        provider = ScriptedProvider(
            [
                ScriptedTurn(
                    AssistantTurn(text="whole"),
                    text_chunks=["wh", "ole"],
                    thought_chunks=["plan"],
                ),
                {
                    "tool_calls": [
                        {"name": "read_file", "args": '{"path":"x"}'}
                    ]
                },
            ],
            summaries=["short"],
        )
        text, thoughts = [], []
        conversation = [{"role": "user", "content": "go"}]

        turn = provider.call(conversation, [], "system", text.append, thoughts.append)
        conversation[0]["content"] = "mutated"

        self.assertEqual(turn.text, "whole")
        self.assertEqual(text, ["wh", "ole"])
        self.assertEqual(thoughts, ["plan"])
        self.assertEqual(provider.calls[0].conversation[0]["content"], "go")
        second = provider.call([], [], "")
        self.assertEqual(second.tool_calls[0].args, {"path": "x"})
        self.assertTrue(second.tool_calls[0].id)
        self.assertEqual(provider.summarize([]), "short")
        provider.assert_exhausted()

    def test_callable_script_can_assert_request_and_exhaustion_is_clear(self):
        def respond(request):
            self.assertEqual(request.system, "rules")
            return "ok"

        provider = ScriptedProvider([respond])
        self.assertEqual(provider.call([], [], "rules").text, "ok")
        with self.assertRaisesRegex(AssertionError, "no turn left"):
            provider.call([], [], "rules")


if __name__ == "__main__":
    unittest.main()
