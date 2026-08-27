from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .protocol import ParsedResponse, ProtocolError, parse_model_response
from .tools import ToolError, ToolRegistry, tool_schemas


class ChatProvider(Protocol):
    def chat(self, messages: list[dict[str, str]]) -> str | dict[str, Any]: ...


class AgentError(RuntimeError):
    pass


@dataclass
class AgentResult:
    success: bool
    message: str
    steps: int
    history: list[dict[str, str]]


class CodingAgent:
    def __init__(self, workspace: Path, provider: ChatProvider, max_steps: int = 12):
        self.workspace = workspace.resolve()
        self.provider = provider
        self.tools = ToolRegistry(self.workspace)
        self.max_steps = max(1, max_steps)

    def run(self, task: str) -> AgentResult:
        if not task.strip():
            raise AgentError("任务不能为空")
        history: list[dict[str, str]] = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": task.strip()},
        ]
        parse_failures = 0
        for step in range(1, self.max_steps + 1):
            try:
                raw = self.provider.chat(history)
            except Exception as exc:
                raise AgentError(f"第 {step} 步模型调用失败: {exc}") from exc
            assistant_content = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)
            history.append({"role": "assistant", "content": assistant_content})
            try:
                response = parse_model_response(raw)
            except (ProtocolError, json.JSONDecodeError) as exc:
                parse_failures += 1
                if parse_failures >= 2:
                    raise AgentError(f"模型输出连续解析失败: {exc}") from exc
                history.append({"role": "user", "content": f"输出格式错误：{exc}。请只返回合法 JSON 动作。"})
                continue
            parse_failures = 0
            if response.kind == "final":
                return AgentResult(True, response.message, step, history)
            result = self._execute(response)
            history.append(
                {
                    "role": "user",
                    "content": "工具结果 " + json.dumps(result, ensure_ascii=False) + "\n请继续完成任务；完成后返回 final。",
                }
            )
        return AgentResult(False, f"达到最大步骤数 {self.max_steps}，任务尚未确认完成。", self.max_steps, history)

    def _execute(self, response: ParsedResponse) -> dict[str, Any]:
        assert response.tool is not None and response.arguments is not None
        try:
            return {"ok": True, "tool": response.tool, "result": self.tools.execute(response.tool, response.arguments)}
        except (ToolError, ValueError) as exc:
            return {"ok": False, "tool": response.tool, "error": str(exc)}
        except Exception as exc:
            return {"ok": False, "tool": response.tool, "error": f"工具执行异常: {exc}"}

    def _system_prompt(self) -> str:
        schema = json.dumps(tool_schemas(), ensure_ascii=False, indent=2)
        return f"""你是一个命令行 Coding Agent。你只能通过下面列出的本地工具工作，所有 path 都必须是相对工作区根目录的路径。先检查相关文件，再修改代码，最后运行测试确认结果。不要臆测文件内容；工具报错时修正参数或方案。\n\n每次只能返回一个 JSON 对象，不要添加 Markdown 或解释文字：\n工具调用：{{\"type\":\"tool_call\",\"tool\":\"工具名\",\"arguments\":{{...}}}}\n任务完成：{{\"type\":\"final\",\"message\":\"简要说明修改和测试结果\"}}\n\n可用工具：\n{schema}\n"""
