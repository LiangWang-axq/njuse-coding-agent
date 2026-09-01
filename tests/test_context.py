from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from coding_agent.context import ContextManager, compression_events, extract_summary


def tool_result(tool: str = "read_file", result: dict | None = None, ok: bool = True) -> str:
    payload = {"ok": ok, "tool": tool, "result": result or {"path": "a.py", "content": "x" * 1200}}
    return "工具结果 " + json.dumps(payload, ensure_ascii=False)


class ContextManagerTests(unittest.TestCase):
    def test_keeps_system_prompt_and_appends(self):
        context = ContextManager("系统提示", max_chars=10_000)
        context.append("user", "你好")
        self.assertEqual(context.messages[0], {"role": "system", "content": "系统提示"})
        self.assertEqual(context.messages[-1], {"role": "user", "content": "你好"})
        self.assertFalse(context.compressed)

    def test_prepare_returns_non_destructive_view(self):
        context = ContextManager("系统提示", max_chars=10_000)
        for index in range(10):
            context.append("user", f"消息{index}")
            context.append("assistant", "回复")
        view = context.prepare()
        self.assertEqual(len(context.messages), 21)
        self.assertEqual(len(view), 21)  # 预算内不压缩
        view.append({"role": "user", "content": "污染"})
        self.assertEqual(len(context.messages), 21)
        self.assertNotIn("污染", " ".join(m["content"] for m in context.messages))

    def test_prepare_compacts_old_results_when_over_budget(self):
        result_msg = tool_result(result={"path": "a.py", "content": "y" * 4000})
        small_result = tool_result(tool="run_tests", result={"passed": True, "stdout": "ok"})
        final_msg = '{"type":"final","message":"完成"}'
        context = ContextManager("system", max_chars=1024)
        context.append("user", "任务")
        context.append("assistant", '{"type":"tool_call","tool":"read_file","arguments":{"path":"a.py"}}')
        context.append("user", result_msg)
        context.append("assistant", '{"type":"tool_call","tool":"run_tests","arguments":{"command":"python -m unittest"}}')
        context.append("user", small_result)
        context.append("assistant", final_msg)
        self.assertFalse(context.compressed)
        view = context.prepare()
        self.assertTrue(context.compressed)
        self.assertEqual(view[0]["role"], "system")
        self.assertIn("read_file", view[3]["content"])  # 旧结果压缩成一行摘要
        self.assertNotIn("y" * 4000, view[3]["content"])
        self.assertEqual(view[-1]["content"], final_msg)
        self.assertEqual(len(context.messages), 7)  # 完整历史保留
        self.assertEqual(context.last_compression, {"L2": 1})
        self.assertIsInstance(context.last_compression["L2"], int)

    def test_compacts_agent_style_tool_result_with_suffix(self):
        """agent 回填的工具结果带 '请继续完成任务' 后缀，也要能压缩。"""
        context = ContextManager("system", max_chars=1024)
        context.append("user", "任务")
        context.append("assistant", '{"type":"tool_call","tool":"read_file","arguments":{"path":"a.py"}}')
        context.append("user", tool_result() + "\n请继续完成任务；完成后返回 final。")
        context.append("assistant", '{"type":"tool_call","tool":"run_tests","arguments":{"command":"python -m unittest"}}')
        context.append(
            "user",
            "工具结果 "
            + json.dumps({"ok": True, "tool": "run_tests", "result": {"passed": True}})
            + "\n请继续完成任务；完成后返回 final。",
        )
        view = context.prepare()
        self.assertTrue(context.compressed)
        self.assertIn("read_file", view[3]["content"])
        self.assertNotIn("请继续完成任务", view[3]["content"])
        self.assertEqual(context.last_compression, {"L2": 1})
        self.assertIsInstance(context.last_compression["L2"], int)

    def test_prepare_keeps_recent_tool_result_untouched(self):
        result_msg = tool_result()
        context = ContextManager("system", max_chars=2048)
        context.append("user", "任务")
        context.append("assistant", '{"type":"tool_call","tool":"read_file","arguments":{"path":"a.py"}}')
        context.append("user", result_msg)
        view = context.prepare()
        self.assertEqual(view[-1]["content"], result_msg)  # 最近一条不压缩

    def test_prepare_keeps_recent_tool_interaction_pairs(self):
        context = ContextManager("system", max_chars=3000)
        context.append("user", "任务")
        for path in ("orders.py", "tests/test_orders.py", "result.txt"):
            context.append(
                "assistant",
                json.dumps({"type": "tool_call", "tool": "read_file", "arguments": {"path": path}}),
            )
            context.append("user", tool_result(result={"path": path, "content": path * 100}))

        view = context.prepare()
        visible = " ".join(message["content"] for message in view)
        self.assertIn("orders.py", visible)
        self.assertIn("tests/test_orders.py", visible)
        self.assertIn("result.txt", visible)

    def test_rejects_small_budget(self):
        with self.assertRaises(ValueError):
            ContextManager("system", max_chars=100)

    def test_prepare_truncates_oversized_single_message(self):
        context = ContextManager("system", max_chars=1024)
        context.append("user", "x" * 3000)
        view = context.prepare()
        total = sum(len(message["content"]) for message in view)
        self.assertLessEqual(total, 1100)
        self.assertIn("已截断", view[-1]["content"])
        self.assertEqual(len(context.messages[-1]["content"]), 3000)  # 真实历史不截断

    def test_l3_persists_large_tool_result(self):
        with tempfile.TemporaryDirectory() as directory:
            context = ContextManager(
                "system",
                max_chars=10_000,
                results_dir=Path(directory) / "results",
            )
            big = tool_result(result={"path": "a.py", "content": "z" * 20_000})
            context.append("user", "任务")
            context.append("assistant", '{"type":"tool_call","tool":"read_file","arguments":{"path":"a.py"}}')
            context.append("user", big)
            view = context.prepare()
            self.assertIn("已落盘", view[-1]["content"])
            self.assertIn("预览", view[-1]["content"])
            self.assertNotIn("z" * 20_000, view[-1]["content"])
            files = list((Path(directory) / "results").glob("*.txt"))
            self.assertEqual(len(files), 1)
            self.assertIn("z" * 20_000, files[0].read_text(encoding="utf-8"))

    def test_l3_counts_new_persists_only_once(self):
        with tempfile.TemporaryDirectory() as directory:
            context = ContextManager(
                "system",
                max_chars=10_000,
                results_dir=Path(directory) / "results",
            )
            big = tool_result(result={"path": "a.py", "content": "z" * 20_000})
            context.append("user", big)
            first = context.prepare()
            self.assertEqual(context.last_compression, {"L3": 1})
            self.assertIn("已落盘", first[-1]["content"])
            second = context.prepare()
            self.assertIsNone(context.last_compression)  # 已落盘过，不再重复计数
            self.assertIn("已落盘", second[-1]["content"])
            files = list((Path(directory) / "results").glob("*.txt"))
            self.assertEqual(len(files), 1)

    def test_usage_anchoring_calibrates_budget(self):
        context = ContextManager("system", max_chars=4000)
        context.append("user", "任务")
        view = context.prepare()
        context.record_usage({"prompt_tokens": view[0]["content"].count("s") * 2 + 100})
        self.assertIsNotNone(context._ratio)

    def test_l4_summarizer_used_when_still_over_budget(self):
        calls = []

        def summarizer(messages):
            calls.append(messages)
            return "<analysis>旧对话分析</analysis>\n<summary>## 目标\n修复 bug\n## 下一步\n跑测试</summary>"

        context = ContextManager("system", max_chars=1024, summarizer=summarizer)
        for index in range(10):
            context.append("user", f"任务{index} " + "x" * 100)
            context.append("assistant", f"回复{index} " + "y" * 100)
        view = context.prepare()
        self.assertTrue(context.compressed)
        self.assertEqual(len(calls), 1)
        self.assertIn("已压缩历史", view[1]["content"])
        self.assertIn("修复 bug", view[1]["content"])
        self.assertEqual(len(context.messages), 21)  # 真实历史不动
        self.assertEqual(context.last_compression, {"L4": 1})

    def test_l4_skipped_without_summarizer(self):
        context = ContextManager("system", max_chars=1024)
        for index in range(10):
            context.append("user", f"任务{index}")
            context.append("assistant", f"回复{index}")
        view = context.prepare()
        self.assertFalse(context.compressed)
        self.assertGreater(len(view), 1)


class ExtractSummaryTests(unittest.TestCase):
    def test_extracts_summary_tag_only(self):
        content = "<analysis>不应该出现</analysis>\n<summary>正文</summary>"
        self.assertEqual(extract_summary(content), "正文")

    def test_falls_back_without_summary_tag(self):
        content = "<analysis>分析</analysis>\n直接输出"
        self.assertEqual(extract_summary(content), "直接输出")

    def test_compression_events_ordered(self):
        events = compression_events({"L2": 2, "L4": 1, "L3": 3})
        self.assertEqual(
            events,
            ["L3 落盘 3 条大结果", "L2 压缩 2 条旧工具结果", "L4 生成结构化摘要"],
        )


if __name__ == "__main__":
    unittest.main()
