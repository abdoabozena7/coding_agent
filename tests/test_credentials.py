from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agent.credentials import api_key_status, save_provider_api_key


class ProviderCredentialTests(unittest.TestCase):
    def test_save_is_atomic_replaces_one_key_and_never_returns_secret(self):
        with tempfile.TemporaryDirectory() as directory:
            env_path = Path(directory) / ".env"
            env_path.write_text(
                "LLM_PROVIDER=ollama\nOPENAI_API_KEY=old-value\n",
                encoding="utf-8",
            )
            environ: dict[str, str] = {}

            variable = save_provider_api_key(
                env_path,
                "openai",
                "new-secret-value",
                environ=environ,
            )

            saved = env_path.read_text(encoding="utf-8")
            self.assertEqual(variable, "OPENAI_API_KEY")
            self.assertEqual(environ["OPENAI_API_KEY"], "new-secret-value")
            self.assertEqual(saved.count("OPENAI_API_KEY="), 1)
            self.assertIn("OPENAI_API_KEY=new-secret-value", saved)
            self.assertNotIn("new-secret-value", variable)
            self.assertFalse(any(env_path.parent.glob(".*.tmp")))

    def test_status_reports_presence_without_exposing_values(self):
        status = api_key_status(
            {
                "OPENAI_API_KEY": "secret-openai",
                "GEMINI_API_KEY": "",
                "OLLAMA_API_KEY": "secret-ollama",
            }
        )
        self.assertEqual(
            status,
            {"openai": True, "gemini": False, "ollama": True},
        )
        self.assertNotIn("secret", repr(status))

    def test_rejects_multiline_secret_and_unknown_provider(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "single line"):
                save_provider_api_key(
                    Path(directory) / ".env",
                    "openai",
                    "first\nsecond",
                    environ={},
                )
            with self.assertRaisesRegex(ValueError, "provider must be"):
                save_provider_api_key(
                    Path(directory) / ".env",
                    "unknown",
                    "value",
                    environ={},
                )
