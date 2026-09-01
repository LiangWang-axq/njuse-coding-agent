from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from coding_agent.agent import CodingAgent
from coding_agent.protocol import parse_model_response
from coding_agent.session import Session, latest_session_path, sessions_dir
from coding_agent.tools import ToolError, WorkspaceTools


class ScriptedProvider:
    def __init__(self, responses):
        self.responses = iter(responses)

    def chat(self, messages, tools=None, tool_choice=None):
        return next(self.responses)


class StreamingScriptedProvider:
    def __init__(self, event_lists):
        self.event_lists = iter(event_lists)

    def chat_stream(self, messages, tools=None, tool_choice=None):
        for event in next(self.event_lists):
            yield event


class RecordingProvider:
    def __init__(self, response):
        self.response = response
        self.calls: list[list[dict[str, str]]] = []
        self.last_usage: dict[str, int] | None = None

    def chat(self, messages, tools=None, tool_choice=None):
        self.calls.append(messages)
        self.last_usage = {
            "prompt_tokens": 100 + 10 * len(self.calls),
            "completion_tokens": 20,
            "total_tokens": 120 + 10 * len(self.calls),
        }
        return self.response


class SummaryAwareProvider:
    """主调用走脚本响应；摘要调用（tools=None）返回结构化摘要。"""

    def __init__(self, responses):
        self.responses = iter(responses)
        self.last_usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}

    def chat(self, messages, tools=None, tool_choice=None):
        if tools is None and messages and messages[0]["content"].startswith("你是一个上下文摘要助手"):
            return "<analysis>分析</analysis>\n<summary>## 目标\n继续任务</summary>"
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

    def test_agent_executes_apply_diff_action(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "calc.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
            diff = "--- a/calc.py\n+++ b/calc.py\n@@ -1,2 +1,2 @@\n def add(a, b):\n-    return a - b\n+    return a + b\n"
            provider = ScriptedProvider(
                [
                    json.dumps({"type": "tool_call", "tool": "apply_diff", "arguments": {"path": "calc.py", "diff": diff}}),
                    json.dumps({"type": "final", "message": "已通过 diff 修复"}),
                ]
            )
            result = CodingAgent(root, provider, max_steps=4).run("修复 add")
            self.assertTrue(result.success)
            self.assertIn("return a + b", (root / "calc.py").read_text(encoding="utf-8"))

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

    def test_agent_stops_repeated_read_cycle_before_max_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            response_cycle = [
                json.dumps({"type": "tool_call", "tool": "read_file", "arguments": {"path": "orders.py"}}),
                json.dumps({"type": "tool_call", "tool": "read_file", "arguments": {"path": "tests/test_orders.py"}}),
            ]
            provider = ScriptedProvider(response_cycle * 5)

            result = CodingAgent(root, provider, max_steps=12).run("检查并修复项目")

            self.assertFalse(result.success)
            self.assertLess(result.steps, 12)
            self.assertIn("重复执行无进展工具调用", result.message)

    def test_agent_gives_one_loop_recovery_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "notes.txt").write_text("old\n", encoding="utf-8")
            diff = "--- a/notes.txt\n+++ b/notes.txt\n@@ -1,1 +1,1 @@\n-old\n+new\n"
            repeated_read = json.dumps(
                {"type": "tool_call", "tool": "read_file", "arguments": {"path": "notes.txt"}}
            )
            provider = ScriptedProvider(
                [
                    repeated_read,
                    repeated_read,
                    repeated_read,
                    json.dumps({"type": "tool_call", "tool": "apply_diff", "arguments": {"path": "notes.txt", "diff": diff}}),
                    json.dumps({"type": "final", "message": "完成"}),
                ]
            )

            result = CodingAgent(root, provider, max_steps=8).run("修改并检查文件")

            self.assertTrue(result.success)
            self.assertEqual(result.message, "完成")
            self.assertIn("不要再次读取相同文件", " ".join(m["content"] for m in result.history))

    def test_successful_edit_resets_repeated_action_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "notes.txt").write_text("old\n", encoding="utf-8")
            diff = "--- a/notes.txt\n+++ b/notes.txt\n@@ -1,1 +1,1 @@\n-old\n+new\n"
            provider = ScriptedProvider(
                [
                    json.dumps({"type": "tool_call", "tool": "read_file", "arguments": {"path": "notes.txt"}}),
                    json.dumps({"type": "tool_call", "tool": "read_file", "arguments": {"path": "notes.txt"}}),
                    json.dumps({"type": "tool_call", "tool": "apply_diff", "arguments": {"path": "notes.txt", "diff": diff}}),
                    json.dumps({"type": "tool_call", "tool": "read_file", "arguments": {"path": "notes.txt"}}),
                    json.dumps({"type": "tool_call", "tool": "read_file", "arguments": {"path": "notes.txt"}}),
                    json.dumps({"type": "final", "message": "完成"}),
                ]
            )

            result = CodingAgent(root, provider, max_steps=8).run("修改并检查文件")

            self.assertTrue(result.success)
            self.assertEqual(result.message, "完成")
            self.assertEqual((root / "notes.txt").read_text(encoding="utf-8"), "new\n")

    def test_agent_reports_compression_events_and_stats(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = Session(root / ".coding_agent" / "sessions" / "seed.jsonl")
            for _ in range(6):
                session.add("user", "任务 " + "x" * 500)
                session.add("assistant", "回复 " + "y" * 500)
            provider = SummaryAwareProvider([json.dumps({"type": "final", "message": "完成"})])
            events: list[list[str]] = []
            agent = CodingAgent(root, provider, max_steps=4, context_chars=1024, session=session)
            result = agent.run("新任务", on_context=events.append)
            self.assertTrue(result.success)
            self.assertTrue(events)
            self.assertTrue(any("L4" in event for event_list in events for event in event_list))
            self.assertEqual(agent.compression_stats.get("L4"), 1)

    def test_agent_aggregates_usage_across_turn(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            provider = ScriptedProvider(
                [
                    json.dumps({"type": "tool_call", "tool": "list_files", "arguments": {"path": "."}}),
                    json.dumps({"type": "final", "message": "完成"}),
                ]
            )
            provider.last_usage = {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}
            agent = CodingAgent(root, provider, max_steps=4)
            result = agent.run("查看文件")
            self.assertTrue(result.success)
            self.assertEqual(result.usage, {"prompt_tokens": 200, "completion_tokens": 40, "total_tokens": 240})
            self.assertEqual(result.budget_tokens, 4000)

    def test_agent_reuses_history_across_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            provider = ScriptedProvider(
                [
                    json.dumps({"type": "final", "message": "第一轮完成"}),
                    json.dumps({"type": "final", "message": "第二轮完成"}),
                ]
            )
            agent = CodingAgent(root, provider, max_steps=4)
            first = agent.run("任务一")
            second = agent.run("任务二")
            self.assertTrue(first.success)
            self.assertTrue(second.success)
            self.assertEqual(second.message, "第二轮完成")
            history = second.history
            self.assertEqual(sum(1 for m in history if m["role"] == "system"), 1)
            contents = " ".join(m["content"] for m in history)
            self.assertIn("任务一", contents)
            self.assertIn("任务二", contents)

    def test_agent_persists_messages_to_session_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            provider = ScriptedProvider(
                [
                    json.dumps({"type": "tool_call", "tool": "list_files", "arguments": {"path": "."}}),
                    json.dumps({"type": "final", "message": "完成"}),
                ]
            )
            agent = CodingAgent(root, provider, max_steps=4)
            result = agent.run("查看文件")
            self.assertTrue(result.success)
            path = latest_session_path(root)
            self.assertIsNotNone(path)
            self.assertTrue(path.is_file())
            session = Session.load(path)
            contents = " ".join(m["content"] for m in session.messages)
            self.assertIn("查看文件", contents)
            self.assertIn("list_files", contents)
            self.assertIn("完成", contents)

    def test_system_prompt_contains_convergence_rules(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            provider = ScriptedProvider([json.dumps({"type": "final", "message": "完成"})])
            agent = CodingAgent(root, provider, max_steps=4)
            system = agent._init_context().messages[0]["content"]
            self.assertIn("收敛规则", system)
            self.assertIn("立即返回 final", system)
            self.assertIn("compileall", system)
            self.assertIn("python -c", system)
            self.assertIn('{"type":"tool_call"', system)  # 提示词中的转义引号渲染正确

    def test_agent_resumes_from_session_history(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = ScriptedProvider([json.dumps({"type": "final", "message": "第一轮完成"})])
            agent = CodingAgent(root, first, max_steps=4)
            agent.run("任务一")
            path = latest_session_path(root)
            self.assertIsNotNone(path)
            session = Session.load(path)

            second = RecordingProvider(json.dumps({"type": "final", "message": "第二轮完成"}))
            resumed = CodingAgent(root, second, max_steps=4, session=session)
            result = resumed.run("任务二")
            self.assertTrue(result.success)
            self.assertEqual(result.message, "第二轮完成")
            self.assertTrue(second.calls)
            sent = second.calls[0]
            roles_and_content = [(m["role"], m["content"]) for m in sent]
            self.assertIn(("user", "任务一"), roles_and_content)  # 历史已恢复
            self.assertIn(("user", "任务二"), roles_and_content)

    def test_switch_session_resets_session_runtime_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            agent = CodingAgent(root, ScriptedProvider([]), max_steps=4)
            agent.compression_stats = {"L2": 3}
            agent.last_usage = {"total_tokens": 100}
            agent._last_context_stats = {"L2": 3}
            agent._context = object()
            target = Session(root / ".coding_agent" / "sessions" / "target.jsonl")

            agent.switch_session(target)

            self.assertIs(agent.session, target)
            self.assertIsNone(agent._context)
            self.assertEqual(agent.compression_stats, {})
            self.assertIsNone(agent.last_usage)
            self.assertIsNone(agent._last_context_stats)

    def test_reset_starts_new_session_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            provider = ScriptedProvider(
                [
                    json.dumps({"type": "final", "message": "第一轮"}),
                    json.dumps({"type": "final", "message": "第二轮"}),
                ]
            )
            agent = CodingAgent(root, provider, max_steps=4)
            agent.run("任务一")
            first_id = agent.session.session_id
            first_path = agent.session.path
            self.assertTrue(first_path.is_file())
            agent.reset()
            agent.run("任务二")
            self.assertNotEqual(agent.session.session_id, first_id)
            self.assertNotEqual(agent.session.path, first_path)
            self.assertTrue(first_path.is_file())  # 旧会话保留
            self.assertEqual(len(list(sessions_dir(root).glob("*.jsonl"))), 2)

    def test_reset_clears_history(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            provider = ScriptedProvider(
                [
                    json.dumps({"type": "final", "message": "第一轮"}),
                    json.dumps({"type": "final", "message": "第二轮"}),
                    json.dumps({"type": "final", "message": "第三轮"}),
                ]
            )
            agent = CodingAgent(root, provider, max_steps=4)
            agent.run("任务一")
            agent.run("任务二")
            agent.reset()
            third = agent.run("任务三")
            self.assertEqual(third.message, "第三轮")
            contents = " ".join(m["content"] for m in third.history)
            self.assertNotIn("任务一", contents)
            self.assertNotIn("任务二", contents)
            self.assertIn("任务三", contents)

    def test_on_text_streams_deltas_and_executes_tool_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.txt").write_text("hello", encoding="utf-8")
            provider = StreamingScriptedProvider(
                [
                    [
                        {
                            "type": "tool_calls",
                            "tool_calls": [
                                {"function": {"name": "read_file", "arguments": {"path": "a.txt"}}}
                            ],
                        }
                    ],
                    [{"type": "text", "delta": "检查"}, {"type": "text", "delta": "完毕"}],
                ]
            )
            agent = CodingAgent(root, provider, max_steps=4)
            deltas: list[str] = []
            result = agent.run("检查文件", on_text=deltas.append)
            self.assertTrue(result.success)
            self.assertEqual(result.message, "检查完毕")
            self.assertEqual(deltas, ["检查", "完毕"])

    def test_streamed_json_content_is_suppressed_and_parsed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            provider = StreamingScriptedProvider(
                [
                    [{"type": "text", "delta": '{"type":"final","mess'}, {"type": "text", "delta": 'age":"完成"}'}],
                ]
            )
            agent = CodingAgent(root, provider, max_steps=4)
            deltas: list[str] = []
            result = agent.run("任务", on_text=deltas.append)
            self.assertTrue(result.success)
            self.assertEqual(result.message, "完成")
            self.assertEqual(deltas, [])
            self.assertFalse(result.streamed_text)

    def test_streamed_plain_text_final_reaches_callback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            provider = StreamingScriptedProvider([[{"type": "text", "delta": "好的"}]])
            agent = CodingAgent(root, provider, max_steps=4)
            deltas: list[str] = []
            result = agent.run("任务", on_text=deltas.append)
            self.assertTrue(result.success)
            self.assertEqual(result.message, "好的")
            self.assertEqual(deltas, ["好的"])
            self.assertTrue(result.streamed_text)


if __name__ == "__main__":
    unittest.main()
