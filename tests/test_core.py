from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from coding_agent.agent import CodingAgent
from coding_agent.protocol import parse_model_response
from coding_agent.tools import ToolError, WorkspaceTools


class ScriptedProvider:
    def __init__(self, responses):
        self.responses = iter(responses)

    def chat(self, messages, tools=None, tool_choice=None):
        return next(self.responses)


class ProtocolTests(unittest.TestCase):
    def test_parses_fenced_tool_json(self):
        response = parse_model_response('```json\n{"type":"tool_call","tool":"read_file","arguments":{"path":"a.py"}}\n```')
        self.assertEqual(response.kind, "tool")
        self.assertEqual(response.tool, "read_file")
        self.assertEqual(response.arguments, {"path": "a.py"})

    def test_parses_native_tool_call_arguments(self):
        response = parse_model_response({"tool_calls": [{"function": {"name": "list_files", "arguments": "{}"}}]})
        self.assertEqual(response.tool, "list_files")
        self.assertEqual(response.arguments, {})

    def test_parses_multiple_native_tool_calls(self):
        response = parse_model_response(
            {
                "tool_calls": [
                    {"function": {"name": "read_file", "arguments": '{"path": "a.py"}'}},
                    {"function": {"name": "search_code", "arguments": '{"query": "TODO"}'}},
                ]
            }
        )
        self.assertEqual(response.kind, "tool")
        self.assertEqual(len(response.calls), 2)
        self.assertEqual(response.calls[0].tool, "read_file")
        self.assertEqual(response.calls[0].arguments, {"path": "a.py"})
        self.assertEqual(response.calls[1].tool, "search_code")


class WorkspaceTests(unittest.TestCase):
    def test_rejects_escape_and_shell_chain(self):
        with tempfile.TemporaryDirectory() as directory:
            tools = WorkspaceTools(Path(directory))
            with self.assertRaises(ValueError):
                tools.read_file("../outside.txt")
            with self.assertRaises(ToolError):
                tools.run_tests("python -m unittest; whoami")

    def test_agent_runs_real_local_edit_and_test_loop(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "calculator.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
            tests = root / "tests"
            tests.mkdir()
            (tests / "test_calculator.py").write_text(
                "import unittest\nfrom calculator import add\n\nclass TestCalculator(unittest.TestCase):\n    def test_add(self):\n        self.assertEqual(add(2, 3), 5)\n\nif __name__ == '__main__':\n    unittest.main()\n",
                encoding="utf-8",
            )
            provider = ScriptedProvider([
                json.dumps({"type": "tool_call", "tool": "read_file", "arguments": {"path": "calculator.py"}}),
                json.dumps({"type": "tool_call", "tool": "replace_in_file", "arguments": {"path": "calculator.py", "old_text": "return a - b", "new_text": "return a + b"}}),
                json.dumps({"type": "tool_call", "tool": "run_tests", "arguments": {"command": "python -m unittest discover -s tests -v"}}),
                json.dumps({"type": "final", "message": "已修复 add 并通过测试"}),
            ])
            result = CodingAgent(root, provider, max_steps=6).run("修复计算器并运行测试")
            self.assertTrue(result.success)
            self.assertIn("通过测试", result.message)
            self.assertIn("return a + b", (root / "calculator.py").read_text(encoding="utf-8"))

    def test_agent_executes_multiple_native_tool_calls_in_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.py").write_text("x = 1\n", encoding="utf-8")
            provider = ScriptedProvider(
                [
                    {
                        "tool_calls": [
                            {"function": {"name": "read_file", "arguments": '{"path": "a.py"}'}},
                            {"function": {"name": "list_files", "arguments": '{"path": "."}'}},
                        ]
                    },
                    json.dumps({"type": "final", "message": "检查完毕"}),
                ]
            )
            result = CodingAgent(root, provider, max_steps=4).run("检查项目")
            self.assertTrue(result.success)
            self.assertIn("检查完毕", result.message)

    def test_agent_on_step_callback_reports_tool_executions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            provider = ScriptedProvider(
                [
                    json.dumps({"type": "tool_call", "tool": "list_files", "arguments": {"path": "."}}),
                    json.dumps({"type": "final", "message": "完成"}),
                ]
            )
            seen: list[tuple[int, str]] = []
            result = CodingAgent(root, provider, max_steps=4).run(
                "查看文件",
                on_step=lambda step, response, outcome: seen.append((step, response.tool or "")),
            )
            self.assertTrue(result.success)
            self.assertEqual(seen, [(1, "list_files")])


if __name__ == "__main__":
    unittest.main()
