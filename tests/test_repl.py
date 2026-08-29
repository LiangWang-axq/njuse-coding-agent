from __future__ import annotations

import os
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from coding_agent import ui
from coding_agent.config import Settings
from coding_agent.protocol import ParsedResponse
from coding_agent.repl import run_repl


class StubAgent:
    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.reset_calls = 0
        self.session = SimpleNamespace(
            session_id="test-session",
            path=Path(workspace) / ".coding_agent" / "sessions" / "test.jsonl",
        )

    def reset(self):
        self.reset_calls += 1


def make_settings() -> Settings:
    return Settings(base_url="https://example.test/v1", api_key="sk-test", model="deepseek-chat")


class ColorTests(unittest.TestCase):
    class FakeStdout:
        def __init__(self, is_tty: bool):
            self._is_tty = is_tty

        def isatty(self) -> bool:
            return self._is_tty

    def test_colors_only_when_tty_and_no_no_color(self):
        with mock.patch("coding_agent.ui.sys.stdout", self.FakeStdout(True)):
            self.assertTrue(ui.colors_enabled())
            self.assertIn("\x1b[", ui.bold("x"))
        with mock.patch("coding_agent.ui.sys.stdout", self.FakeStdout(False)):
            self.assertFalse(ui.colors_enabled())
            self.assertEqual(ui.bold("x"), "x")
        with mock.patch("coding_agent.ui.sys.stdout", self.FakeStdout(True)), mock.patch.dict(
            os.environ, {"NO_COLOR": "1"}
        ):
            self.assertFalse(ui.colors_enabled())


class ToolCallFormatTests(unittest.TestCase):
    def test_formats_ok_and_error(self):
        ok = ui.format_tool_call(
            ParsedResponse("tool", tool="read_file", arguments={"path": "a.py"}),
            {"ok": True, "result": {}},
        )
        self.assertIn("read_file", ok)
        self.assertIn("成功", ok)
        err = ui.format_tool_call(
            ParsedResponse("tool", tool="read_file", arguments={"path": "a.py"}),
            {"ok": False, "error": "文件不存在"},
        )
        self.assertIn("错误", err)

    def test_read_file_card_shows_path_and_chars(self):
        card = ui.format_tool_call(
            ParsedResponse("tool", tool="read_file", arguments={"path": "a.py"}),
            {"ok": True, "result": {"path": "a.py", "content": "x" * 50, "truncated": False}},
        )
        self.assertIn("a.py", card)
        self.assertIn("50 字符", card)

    def test_failed_run_tests_card_shows_output_tail(self):
        card = ui.format_tool_call(
            ParsedResponse("tool", tool="run_tests", arguments={"command": "python -m unittest"}),
            {
                "ok": True,
                "result": {"passed": False, "returncode": 1, "stdout": "FAILED (failures=1)"},
            },
        )
        self.assertIn("失败", card)
        self.assertIn("FAILED", card)

    def test_format_compression_stats(self):
        rendered = ui.format_compression_stats({"L3": 2, "L2": 5, "L4": 1})
        self.assertEqual(rendered, "大结果落盘 2 次 · 旧结果压缩 5 次 · LLM 摘要 1 次")
        self.assertEqual(ui.format_compression_stats({}), "无")

    def test_run_tests_shows_pass_fail(self):
        passed = ui.format_tool_call(
            ParsedResponse("tool", tool="run_tests", arguments={"command": "python -m unittest"}),
            {"ok": True, "result": {"passed": True}},
        )
        self.assertIn("通过", passed)
        failed = ui.format_tool_call(
            ParsedResponse("tool", tool="run_tests", arguments={}),
            {"ok": True, "result": {"passed": False, "returncode": 1}},
        )
        self.assertIn("失败", failed)


class ReplTests(unittest.TestCase):
    def test_exit_command(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = StubAgent(Path(directory))
            with mock.patch("builtins.input", return_value="/exit"):
                code = run_repl(agent, make_settings())
        self.assertEqual(code, 0)

    def test_new_resets_agent(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = StubAgent(Path(directory))
            with mock.patch("builtins.input", side_effect=["/new", "/exit"]):
                code = run_repl(agent, make_settings())
        self.assertEqual(code, 0)
        self.assertEqual(agent.reset_calls, 1)

    def test_unknown_command_prints_hint(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = StubAgent(Path(directory))
            output = StringIO()
            with mock.patch("builtins.input", side_effect=["/foo", "/exit"]), redirect_stdout(output):
                code = run_repl(agent, make_settings())
        self.assertEqual(code, 0)
        self.assertIn("未知命令", output.getvalue())

    def test_eof_exits_cleanly(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = StubAgent(Path(directory))
            with mock.patch("builtins.input", side_effect=EOFError):
                code = run_repl(agent, make_settings())
        self.assertEqual(code, 0)

    def test_status_shows_session(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = StubAgent(Path(directory))
            output = StringIO()
            with mock.patch("builtins.input", side_effect=["/status", "/exit"]), redirect_stdout(output):
                code = run_repl(agent, make_settings())
        self.assertEqual(code, 0)
        self.assertIn("test-session", output.getvalue())


if __name__ == "__main__":
    unittest.main()
