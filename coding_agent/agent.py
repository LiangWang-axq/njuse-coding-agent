from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from .context import ContextManager
from .protocol import ParsedResponse, ProtocolError, parse_model_response
from .tools import ToolError, ToolRegistry, openai_tools, tool_schemas


class ChatProvider(Protocol):
    def chat(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
    ) -> str | dict[str, Any]: ...

    def chat_stream(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
    ) -> Any: ...


class AgentError(RuntimeError):
    pass


@dataclass
class AgentResult:
    success: bool
    message: str
    steps: int
    history: list[dict[str, str]]
    streamed_text: bool = False


class CodingAgent:
    def __init__(
        self,
        workspace: Path,
        provider: ChatProvider,
        max_steps: int = 12,
        context_chars: int = 16_000,
    ):
        self.workspace = workspace.resolve()
        self.provider = provider
        self.tools = ToolRegistry(self.workspace)
        self.max_steps = max(1, max_steps)
        self.context_chars = max(1024, int(context_chars))
        self._context: ContextManager | None = None

    def reset(self) -> None:
        """Clear the conversation history; the next run() starts a fresh context."""
        self._context = None

    def run(
        self,
        task: str,
        on_step: Callable[[int, ParsedResponse, dict[str, Any]], None] | None = None,
        on_text: Callable[[str], None] | None = None,
    ) -> AgentResult:
        if not task.strip():
            raise AgentError("任务不能为空")
        if self._context is None:
            self._context = ContextManager(self._system_prompt(), self.context_chars)
        context = self._context
        context.append("user", task.strip())
        parse_failures = 0
        final_turn_streamed = False
        for step in range(1, self.max_steps + 1):
            try:
                if on_text is not None and hasattr(self.provider, "chat_stream"):
                    raw, turn_streamed = self._chat_stream(context.messages, on_text)
                else:
                    raw = self.provider.chat(context.messages, tools=openai_tools(), tool_choice="auto")
                    turn_streamed = False
            except Exception as exc:
                raise AgentError(f"第 {step} 步模型调用失败: {exc}") from exc
            assistant_content = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)
            context.append("assistant", assistant_content)
            try:
                response = parse_model_response(raw)
            except (ProtocolError, json.JSONDecodeError) as exc:
                parse_failures += 1
                if parse_failures >= 2:
                    raise AgentError(f"模型输出连续解析失败: {exc}") from exc
                context.append("user", f"输出格式错误：{exc}。请只返回合法 JSON 动作。")
                continue
            parse_failures = 0
            if response.kind == "final":
                final_turn_streamed = turn_streamed
                return AgentResult(True, response.message, step, context.messages, streamed_text=final_turn_streamed)
            calls = response.calls or (response,)
            results: list[dict[str, Any]] = []
            for call in calls:
                result = self._execute(call)
                if on_step is not None:
                    on_step(step, call, result)
                results.append(result)
            context.append(
                "user",
                "工具结果 "
                + json.dumps(results if len(results) > 1 else results[0], ensure_ascii=False)
                + "\n请继续完成任务；完成后返回 final。",
            )
        return AgentResult(False, f"达到最大步骤数 {self.max_steps}，任务尚未确认完成。", self.max_steps, context.messages)

    def _chat_stream(
        self,
        messages: list[dict[str, str]],
        on_text: Callable[[str], None],
    ) -> tuple[str | dict[str, Any], bool]:
        """Consume provider stream events; returns (raw, displayed_text).

        Text deltas that form a complete JSON action (tool call or final) are
        suppressed from the terminal and returned as a dict for the parser,
        while plain prose is forwarded to ``on_text`` as it streams.
        """
        text_parts: list[str] = []
        held: list[str] = []
        jsonish = False
        displayed = False
        tool_calls = None
        for event in self.provider.chat_stream(messages, tools=openai_tools(), tool_choice="auto"):
            kind = event.get("type")
            if kind == "text":
                delta = event.get("delta", "")
                if not delta:
                    continue
                if jsonish:
                    held.append(delta)
                    continue
                if not held and delta.lstrip().startswith("{"):
                    held.append(delta)
                    jsonish = True
                    continue
                text_parts.append(delta)
                on_text(delta)
                displayed = True
            elif kind == "tool_calls":
                tool_calls = event.get("tool_calls")
        if tool_calls:
            return {"tool_calls": tool_calls}, displayed
        if jsonish:
            candidate = "".join(held)
            try:
                payload = json.loads(candidate)
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, dict):
                return payload, displayed
            for delta in held:
                text_parts.append(delta)
                on_text(delta)
            displayed = True
        content = "".join(text_parts)
        if not content.strip():
            raise AgentError("模型响应没有文本内容")
        return content, displayed

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
        return f"""你是一个命令行 Coding Agent。你只能通过下面列出的本地工具工作，所有 path 都必须是相对工作区根目录的路径，统一使用正斜杠（/）。先检查相关文件，再修改代码，最后运行测试确认结果。不要臆测文件内容；工具报错时修正参数或方案。任务针对工作区子目录时，读写文件用相对路径，并给 run_tests 传 cwd 参数让测试在该子目录下执行。run_tests 失败时先读返回的 stdout/stderr 中的断言或错误详情，再决定下一步，不要反复猜测目录结构；优先使用 python -m unittest discover -s tests 或 python -m pytest -q。\n\n每次只能返回一个 JSON 对象，不要添加 Markdown 或解释文字：\n工具调用：{{\"type\":\"tool_call\",\"tool\":\"工具名\",\"arguments\":{{...}}}}\n任务完成：{{\"type\":\"final\",\"message\":\"简要说明修改和测试结果\"}}\n\n可用工具：\n{schema}\n"""
