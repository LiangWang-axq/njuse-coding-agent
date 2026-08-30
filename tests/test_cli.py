from __future__ import annotations

import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

from coding_agent import cli
from coding_agent.config import Settings
from coding_agent.session import Session, sessions_dir


class SessionCliTests(unittest.TestCase):
    def _session(self, root: Path, session_id: str = "session-one") -> Session:
        session = Session(sessions_dir(root) / f"{session_id}.jsonl", session_id=session_id)
        session.add("user", "历史任务")
        return session

    def test_list_sessions_does_not_require_api_key(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._session(root)
            output = StringIO()
            with mock.patch.object(cli.Path, "cwd", return_value=root), mock.patch.object(
                cli, "load_settings", side_effect=AssertionError("不应加载配置")
            ), mock.patch("sys.argv", ["coding-agent", "--list-sessions"]), redirect_stdout(output):
                code = cli.main()
        self.assertEqual(code, 0)
        self.assertIn("session-one", output.getvalue())

    def test_delete_session_with_yes_does_not_require_api_key(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = self._session(root)
            with mock.patch.object(cli.Path, "cwd", return_value=root), mock.patch.object(
                cli, "load_settings", side_effect=AssertionError("不应加载配置")
            ), mock.patch("sys.argv", ["coding-agent", "--delete-session", "session", "--yes"]):
                code = cli.main()
            self.assertFalse(session.path.exists())
        self.assertEqual(code, 0)

    def test_delete_session_cancel_returns_one_and_keeps_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = self._session(root)
            output = StringIO()
            with mock.patch.object(cli.Path, "cwd", return_value=root), mock.patch(
                "sys.argv", ["coding-agent", "--delete-session", "1"]
            ), mock.patch("builtins.input", return_value="n"), redirect_stdout(output):
                code = cli.main()
            self.assertTrue(session.path.exists())
        self.assertEqual(code, 1)
        self.assertIn("已取消", output.getvalue())

    def test_resume_session_selects_history_for_one_shot_task(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._session(root)
            seen = {}

            def stream(agent, task):
                seen["session_id"] = agent.session.session_id
                seen["task"] = task
                return 0

            settings = Settings("https://example.test/v1", "sk-test", "test-model")
            with mock.patch.object(cli.Path, "cwd", return_value=root), mock.patch.object(
                cli, "load_settings", return_value=settings
            ), mock.patch.object(cli, "OpenAICompatibleProvider", return_value=object()), mock.patch.object(
                cli, "_stream_answer", side_effect=stream
            ), mock.patch(
                "sys.argv", ["coding-agent", "--resume-session", "session", "继续任务"]
            ):
                code = cli.main()
        self.assertEqual(code, 0)
        self.assertEqual(seen, {"session_id": "session-one", "task": "继续任务"})

    def test_original_resume_still_loads_latest_session(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = self._session(root, "latest-session")
            loaded = cli._load_resume_session(root, True)
        self.assertEqual(loaded.session_id, expected.session_id)

    def test_management_arguments_reject_task_combination(self):
        error = StringIO()
        with mock.patch("sys.argv", ["coding-agent", "--list-sessions", "任务"]), redirect_stderr(error):
            with self.assertRaises(SystemExit) as raised:
                cli.main()
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("不能与任务", error.getvalue())


if __name__ == "__main__":
    unittest.main()
