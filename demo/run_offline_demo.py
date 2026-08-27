from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from coding_agent.agent import CodingAgent


class DemoProvider:
    """Deterministic responses for a repeatable no-network demonstration."""

    def __init__(self):
        self.responses = iter([
            {"type": "tool_call", "tool": "list_files", "arguments": {"path": "."}},
            {"type": "tool_call", "tool": "search_code", "arguments": {"query": "remove", "path": "."}},
            {"type": "tool_call", "tool": "read_file", "arguments": {"path": "todo.py"}},
            {"type": "tool_call", "tool": "write_file", "arguments": {"path": "tests/test_todo.py", "content": "import unittest\nfrom todo import TodoList\n\nclass TestTodo(unittest.TestCase):\n    def test_remove_by_id(self):\n        todos = TodoList()\n        todos.add('one')\n        todos.add('two')\n        self.assertEqual(todos.remove(2)['title'], 'two')\n        self.assertEqual([item['id'] for item in todos.items], [1])\n\nif __name__ == '__main__':\n    unittest.main()\n"}},
            {"type": "tool_call", "tool": "run_tests", "arguments": {"command": "python -m unittest discover -s tests -v"}},
            {"type": "tool_call", "tool": "replace_in_file", "arguments": {"path": "todo.py", "old_text": "return self.items.pop(todo_id)", "new_text": "\n        for index, item in enumerate(self.items):\n            if item[\"id\"] == todo_id:\n                return self.items.pop(index)\n        raise KeyError(todo_id)"}},
            {"type": "tool_call", "tool": "run_tests", "arguments": {"command": "python -m unittest discover -s tests -v"}},
            {"type": "final", "message": "已补充回归测试，先复现失败，再修复 Todo 编号删除逻辑，最终测试通过。"},
        ])

    def chat(self, messages, tools=None, tool_choice=None):
        return next(self.responses)


def main() -> int:
    demo_root = Path(__file__).resolve().parent / ".runtime"
    if demo_root.exists():
        shutil.rmtree(demo_root)
    demo_root.mkdir(parents=True)
    (demo_root / "todo.py").write_text(
        "class TodoList:\n    def __init__(self):\n        self.items = []\n\n    def add(self, title):\n        item = {\"id\": len(self.items) + 1, \"title\": title, \"done\": False}\n        self.items.append(item)\n        return item\n\n    def remove(self, todo_id):\n        return self.items.pop(todo_id)\n",
        encoding="utf-8",
    )
    (demo_root / "tests").mkdir()
    result = CodingAgent(demo_root, DemoProvider(), max_steps=8).run("修复 Todo 按编号删除的 bug 并运行测试")
    print(json.dumps({"success": result.success, "steps": result.steps, "message": result.message}, ensure_ascii=False, indent=2))
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
