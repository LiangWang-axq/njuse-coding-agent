from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from coding_agent.config import load_settings


class ConfigTests(unittest.TestCase):
    def test_loads_dotenv_from_nearest_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            child = root / "project" / "sub"
            child.mkdir(parents=True)
            (root / ".env").write_text(
                "AGENT_API_KEY=sk-parent\nAGENT_MODEL=deepseek-chat\n",
                encoding="utf-8",
            )
            with mock.patch.dict(os.environ, {}, clear=False):
                for key in [k for k in os.environ if k.startswith("AGENT_") or k == "OPENAI_API_KEY"]:
                    os.environ.pop(key, None)
                settings = load_settings(child)
            self.assertEqual(settings.api_key, "sk-parent")
            self.assertEqual(settings.model, "deepseek-chat")

    def test_default_max_steps_is_24(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict(os.environ, {}, clear=False):
                for key in [k for k in os.environ if k.startswith("AGENT_") or k == "OPENAI_API_KEY"]:
                    os.environ.pop(key, None)
                settings = load_settings(Path(directory))
            self.assertEqual(settings.max_steps, 24)

    def test_max_steps_env_override(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict(os.environ, {"AGENT_MAX_STEPS": "30"}, clear=False):
                settings = load_settings(Path(directory))
            self.assertEqual(settings.max_steps, 30)


if __name__ == "__main__":
    unittest.main()
