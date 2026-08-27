from __future__ import annotations

import json
import unittest

from coding_agent.context import ContextManager


class ContextManagerTests(unittest.TestCase):
    def test_keeps_system_prompt_and_appends(self):
        context = ContextManager("系统提示", max_chars=10_000)
        context.append("user", "你好")
        self.assertEqual(context.messages[0], {"role": "system", "content": "系统提示"})
        self.assertEqual(context.messages[-1], {"role": "user", "content": "你好"})
        self.assertFalse(context.compressed)

    def test_trims_oldest_messages_and_inserts_summary(self):
        result_msg = "工具结果 " + json.dumps(
            {"ok": True, "tool": "read_file", "result": {"path": "a.py", "content": "x" * 1200}},
            ensure_ascii=False,
        )
        final_msg = '{"type":"final","message":"完成"}'
        context = ContextManager("system", max_chars=len(result_msg) + 10)
        context.append("user", "任务")
        context.append("assistant", '{"type":"tool_call","tool":"read_file","arguments":{"path":"a.py"}}')
        context.append("user", result_msg)
        context.append("assistant", final_msg)
        self.assertTrue(context.compressed)
        messages = context.messages
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("已压缩历史", messages[1]["content"])
        self.assertIn("read_file", messages[1]["content"])
        self.assertEqual(messages[-1]["content"], final_msg)

    def test_rejects_small_budget(self):
        with self.assertRaises(ValueError):
            ContextManager("system", max_chars=100)

    def test_truncates_oversized_single_message(self):
        context = ContextManager("system", max_chars=1024)
        context.append("user", "x" * 3000)
        total = sum(len(message["content"]) for message in context.messages)
        self.assertLessEqual(total, 1100)
        self.assertIn("已截断", context.messages[-1]["content"])


if __name__ == "__main__":
    unittest.main()
