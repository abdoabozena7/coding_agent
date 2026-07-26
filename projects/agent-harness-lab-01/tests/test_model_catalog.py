"""Offline model catalog and descriptor factory tests."""

from __future__ import annotations

import unittest
from unittest import mock
from urllib.parse import urlsplit

from agent.model_catalog import ExecutionClass, ModelCatalog, ModelDescriptor
from agent.providers import create_provider
from agent.providers.ollama_provider import OllamaProvider


class FakeOllamaHTTP:
    def __init__(self, models=(), shows=None, *, version="0.12.4", fail=None):
        self.models = list(models)
        self.shows = dict(shows or {})
        self.version = version
        self.fail = fail
        self.calls = []

    def __call__(self, method, url, *, payload, headers, timeout):
        path = urlsplit(url).path
        self.calls.append(
            {
                "method": method,
                "url": url,
                "path": path,
                "payload": payload,
                "headers": dict(headers),
                "timeout": timeout,
            }
        )
        if self.fail == path:
            raise RuntimeError(f"offline failure at {path}")
        if path == "/api/version":
            return {"version": self.version}
        if path == "/api/tags":
            return {"models": self.models}
        if path == "/api/show":
            model = payload["model"]
            response = self.shows.get(model, {"capabilities": ["completion", "tools"]})
            if isinstance(response, Exception):
                raise response
            return response
        raise AssertionError(f"unexpected request: {method} {url}")


class ModelDescriptorTests(unittest.TestCase):
    def test_execution_class_and_descriptor_normalization(self):
        descriptor = ModelDescriptor(
            provider=" OLLAMA ",
            model=" qwen3 ",
            execution_class="LOCAL",
            host="localhost:11434/",
            capabilities=("Completion", "TOOLS", "TOOLS"),
            metadata={"digest": "abc", "api_key": "must-not-survive"},
        )

        self.assertEqual(descriptor.provider, "ollama")
        self.assertEqual(descriptor.model, "qwen3")
        self.assertEqual(descriptor.execution_class, ExecutionClass.LOCAL)
        self.assertEqual(descriptor.default_concurrency, 1)
        self.assertEqual(descriptor.capabilities, ("completion", "tools"))
        self.assertTrue(descriptor.supports_tools)
        self.assertNotIn("api_key", descriptor.metadata)
        self.assertIn("ollama:qwen3@http://localhost:11434", descriptor.id)

    def test_provider_creation_is_independent_and_pinned(self):
        descriptor = ModelDescriptor(
            provider="ollama",
            model="coder:latest",
            host="http://127.0.0.1:11435",
            execution_class=ExecutionClass.LOCAL,
        )

        first = descriptor.create_provider()
        second = create_provider(descriptor)

        self.assertIsInstance(first, OllamaProvider)
        self.assertIsInstance(second, OllamaProvider)
        self.assertIsNot(first, second)
        self.assertEqual(first.model, "coder:latest")
        self.assertEqual(first.host, "http://127.0.0.1:11435")

    def test_local_gpu_requirement_never_applies_to_ollama_cloud_models(self):
        cloud = ModelDescriptor(
            provider="ollama",
            model="gpt-oss:120b-cloud",
            host="http://localhost:11434",
            execution_class=ExecutionClass.CLOUD,
        )
        local = ModelDescriptor(
            provider="ollama",
            model="coder:latest",
            host="http://localhost:11434",
            execution_class=ExecutionClass.LOCAL,
        )
        with mock.patch.dict("os.environ", {"AGENT_REQUIRE_LOCAL_GPU": "1"}, clear=False):
            self.assertFalse(cloud.create_provider().require_gpu)
            self.assertTrue(local.create_provider().require_gpu)

    def test_invalid_values_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "execution class"):
            ExecutionClass.parse("edge")
        with self.assertRaisesRegex(ValueError, "unsupported"):
            ModelDescriptor("other", "x", ExecutionClass.CLOUD)
        with self.assertRaisesRegex(TypeError, "descriptor"):
            create_provider(object())
        with self.assertRaisesRegex(ValueError, "must not contain credentials"):
            ModelDescriptor(
                "ollama",
                "x",
                ExecutionClass.CLOUD,
                host="https://user:password@ollama.com",
            )


class ModelCatalogTests(unittest.TestCase):
    def test_probes_version_tags_and_show_and_filters_non_tool_models(self):
        http = FakeOllamaHTTP(
            models=[
                {"name": "vision", "digest": "v"},
                {
                    "name": "coder",
                    "digest": "c",
                    "details": {"parameter_size": "8B"},
                },
                {"name": "coder"},  # duplicate tags do not duplicate /show
            ],
            shows={
                "vision": {"capabilities": ["completion", "vision"]},
                "coder": {"capabilities": ["completion", "tools", "thinking"]},
            },
        )
        catalog = ModelCatalog(environ={}, http_json=http)

        discovered = catalog.discover()

        self.assertEqual([item.model for item in discovered], ["coder"])
        self.assertEqual(discovered[0].execution_class, ExecutionClass.LOCAL)
        self.assertEqual(discovered[0].metadata["ollama_version"], "0.12.4")
        self.assertEqual(discovered[0].metadata["parameter_size"], "8B")
        self.assertEqual(
            [(call["method"], call["path"]) for call in http.calls],
            [
                ("GET", "/api/version"),
                ("GET", "/api/tags"),
                ("POST", "/api/show"),
                ("POST", "/api/show"),
            ],
        )
        self.assertEqual(http.calls[-1]["payload"], {"model": "coder", "verbose": False})
        self.assertTrue(any("does not advertise tools" in item.message for item in catalog.diagnostics))

    def test_cloud_classification_uses_name_remote_metadata_and_host(self):
        http = FakeOllamaHTTP(
            models=[
                {"name": "a:cloud"},
                {"name": "b:70b-cloud"},
                {"name": "c", "remote_model": "c:remote"},
                {"name": "d"},
            ]
        )
        local_catalog = ModelCatalog(environ={}, http_json=http)
        classes = {item.model: item.execution_class for item in local_catalog.discover()}
        self.assertEqual(classes["a:cloud"], ExecutionClass.CLOUD)
        self.assertEqual(classes["b:70b-cloud"], ExecutionClass.CLOUD)
        self.assertEqual(classes["c"], ExecutionClass.CLOUD)
        self.assertEqual(classes["d"], ExecutionClass.LOCAL)

        hosted_http = FakeOllamaHTTP(models=[{"name": "plain"}])
        hosted = ModelCatalog(
            environ={"OLLAMA_API_KEY": "ollama-secret"},
            ollama_host="https://api.ollama.com/",
            http_json=hosted_http,
        ).discover()
        self.assertEqual(hosted[0].execution_class, ExecutionClass.CLOUD)
        self.assertEqual(hosted[0].default_concurrency, 4)
        self.assertEqual(
            hosted_http.calls[0]["headers"]["Authorization"],
            "Bearer ollama-secret",
        )
        self.assertNotIn("ollama-secret", repr(hosted[0].to_dict()))

    def test_remote_marker_from_show_is_honored(self):
        http = FakeOllamaHTTP(
            models=[{"name": "plain"}],
            shows={
                "plain": {
                    "capabilities": ["tools"],
                    "metadata": {"remote": {"host": "ollama.com"}},
                }
            },
        )
        item = ModelCatalog(environ={}, http_json=http).discover()[0]
        self.assertEqual(item.execution_class, ExecutionClass.CLOUD)

    def test_configured_fallbacks_follow_ollama_and_do_not_leak_keys(self):
        http = FakeOllamaHTTP(models=[{"name": "local"}])
        environ = {
            "OPENAI_API_KEY": "openai-secret",
            "OPENAI_MODEL": "gpt-custom",
            "GEMINI_API_KEY": "gemini-secret",
            "GEMINI_MODEL": "gemini-custom",
        }
        catalog = ModelCatalog(environ=environ, http_json=http)

        models = catalog.discover()

        self.assertEqual([item.provider for item in models], ["ollama", "openai", "gemini"])
        self.assertEqual([item.model for item in models], ["local", "gpt-custom", "gemini-custom"])
        self.assertTrue(all(item.supports_tools for item in models))
        self.assertTrue(all(item.execution_class is ExecutionClass.CLOUD for item in models[1:]))
        serialized = repr([item.to_dict() for item in models])
        self.assertNotIn("openai-secret", serialized)
        self.assertNotIn("gemini-secret", serialized)

    def test_unavailable_or_bad_ollama_is_nonfatal_and_records_diagnostic(self):
        failed = FakeOllamaHTTP(fail="/api/version")
        catalog = ModelCatalog(
            environ={"OPENAI_API_KEY": "configured"},
            http_json=failed,
        )

        models = catalog.discover()

        self.assertEqual([item.provider for item in models], ["openai"])
        self.assertIn("offline failure", catalog.diagnostics[0].message)

        malformed = FakeOllamaHTTP()
        malformed.models = None
        self.assertEqual(ModelCatalog(environ={}, http_json=malformed).discover(), ())

    def test_one_failed_show_does_not_hide_other_models(self):
        http = FakeOllamaHTTP(
            models=[{"name": "bad"}, {"name": "good"}],
            shows={"bad": RuntimeError("not found")},
        )
        catalog = ModelCatalog(environ={}, http_json=http)
        self.assertEqual([item.model for item in catalog.discover()], ["good"])
        self.assertEqual(catalog.diagnostics[0].source, "ollama:bad")


if __name__ == "__main__":
    unittest.main()
