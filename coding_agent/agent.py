from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from .context import (
    SUMMARY_PROMPT_TEMPLATE,
    SUMMARY_SYSTEM_PROMPT,
    ContextManager,
    compression_events,
    extract_summary,
)
from .protocol import ParsedResponse, ProtocolError, parse_model_response
from .session import Session, new_session, results_dir
from .tools import ToolError, ToolRegistry, openai_tools, tool_schemas

CONVERGENCE_RULES = (
    "\n\n收敛规则（务必遵守）：\n"
    "1. 任务要求的所有文件已创建/修改且测试全部通过后，立即返回 final 并总结结果，"
    "不要为了“再确认”重复读取已经读过的文件。\n"
    "2. 不要对同一个文件反复执行 list_files/read_file；除非工具结果报错或任务要求发生变化。\n"
    "3. 验证语法或导入用 python -m compileall <目录>，或通过 pytest/unittest 测试验证；"
    "禁止使用 python -c 等脚本展开命令（run_tests 会拒绝，且会浪费步骤）。\n"
    "4. 工具返回成功且测试通过后，把任务状态推进到 final，而不是停留在重复检查。"
)


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
    usage: dict[str, int] | None = None  # 本轮所有模型调用的 usage 汇总
    budget_tokens: int = 0  # 上下文 token 预算
    estimated_tokens: int = 0  # 最近一次发送视图的 token 估算


class CodingAgent:
    def __init__(
        self,
        workspace: Path,
        provider: ChatProvider,
        max_steps: int = 24,
        context_chars: int = 16_000,
        session: Session | None = None,
    ):
        self.workspace = workspace.resolve()
        self.provider = provider
        self.tools = ToolRegistry(self.workspace)
        self.max_steps = max(1, max_steps)
        self.context_chars = max(1024, int(context_chars))
        self.session = session or new_session(self.workspace)
        self.compression_stats: dict[str, int] = {}  # 会话累计压缩统计
        self.last_usage: dict[str, int] | None = None  # 最近一次模型调用 usage
        self._last_context_stats: dict[str, int] | None = None
        self._context: ContextManager | None = None

    def reset(self) -> None:
        """清空对话历史并开启新会话（旧会话文件保留，可用 --resume 恢复）。"""
        self.session = new_session(self.workspace)
        self._reset_session_runtime()

    def switch_session(self, session: Session) -> None:
        """切换到已加载会话，并重置仅属于原会话的运行时统计。"""
        self.session = session
        self._reset_session_runtime()

    def _reset_session_runtime(self) -> None:
        self._context = None
        self.compression_stats = {}
        self.last_usage = None
        self._last_context_stats = None

    def run(
        self,
        task: str,
        on_step: Callable[[int, ParsedResponse, dict[str, Any]], None] | None = None,
        on_text: Callable[[str], None] | None = None,
        on_context: Callable[[list[str]], None] | None = None,
    ) -> AgentResult:
        if not task.strip():
            raise AgentError("任务不能为空")
        if self._context is None:
            self._context = self._init_context()
        context = self._context
        self._append("user", task.strip())
        parse_failures = 0
        final_turn_streamed = False
        turn_usage: dict[str, int] = {}
        for step in range(1, self.max_steps + 1):
            try:
                view = context.prepare()
                self._report_compression(context, on_context)
                if on_text is not None and hasattr(self.provider, "chat_stream"):
                    raw, turn_streamed = self._chat_stream(view, on_text)
                else:
                    raw = self.provider.chat(view, tools=openai_tools(), tool_choice="auto")
                    turn_streamed = False
            except Exception as exc:
                raise AgentError(f"第 {step} 步模型调用失败: {exc}") from exc
            self._record_usage(context, turn_usage)
            assistant_content = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)
            self._append("assistant", assistant_content)
            try:
                response = parse_model_response(raw)
            except (ProtocolError, json.JSONDecodeError) as exc:
                parse_failures += 1
                if parse_failures >= 2:
                    raise AgentError(f"模型输出连续解析失败: {exc}") from exc
                self._append("user", f"输出格式错误：{exc}。请只返回合法 JSON 动作。")
                continue
            parse_failures = 0
            if response.kind == "final":
                final_turn_streamed = turn_streamed
                return AgentResult(
                    True,
                    response.message,
                    step,
                    context.messages,
                    streamed_text=final_turn_streamed,
                    usage=turn_usage or None,
                    budget_tokens=self.context_chars // 4,
                    estimated_tokens=context.last_estimated_tokens,
                )
            calls = response.calls or (response,)
            results: list[dict[str, Any]] = []
            for call in calls:
                result = self._execute(call)
                if on_step is not None:
                    on_step(step, call, result)
                results.append(result)
            self._append(
                "user",
                "工具结果 "
                + json.dumps(results if len(results) > 1 else results[0], ensure_ascii=False)
                + "\n请继续完成任务；完成后返回 final。",
            )
        return AgentResult(
            False,
            f"达到最大步骤数 {self.max_steps}，任务尚未确认完成。",
            self.max_steps,
            context.messages,
            usage=turn_usage or None,
            budget_tokens=self.context_chars // 4,
            estimated_tokens=context.last_estimated_tokens,
        )

    def _append(self, role: str, content: str) -> None:
        """写入上下文历史并同步原子落盘到会话文件。"""
        assert self._context is not None
        self._context.append(role, content)
        self.session.add(role, content)

    def _record_usage(self, context: ContextManager, turn_usage: dict[str, int]) -> None:
        """把 provider 最近一次调用的 usage 汇总进本轮，并喂给上下文做 token 校准。"""
        usage = getattr(self.provider, "last_usage", None)
        if not isinstance(usage, dict):
            return
        self.last_usage = usage
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = usage.get(key)
            if isinstance(value, int):
                turn_usage[key] = turn_usage.get(key, 0) + value
        context.record_usage(usage)

    def _report_compression(
        self, context: ContextManager, on_context: Callable[[list[str]], None] | None
    ) -> None:
        """累计压缩统计；仅在新压缩层首次出现时回调播报，避免每步刷屏。"""
        stats = context.last_compression
        if not stats:
            return
        previous = self._last_context_stats or {}
        new_layers = set(stats) - set(previous)
        self._last_context_stats = {**previous, **stats}
        for key, value in stats.items():
            self.compression_stats[key] = self.compression_stats.get(key, 0) + value
        if on_context is not None and (new_layers or not previous):
            on_context(compression_events(stats))

    def _init_context(self) -> ContextManager:
        context = ContextManager(
            self._system_prompt() + CONVERGENCE_RULES,
            self.context_chars,
            summarizer=self._summarize_history,
            results_dir=results_dir(self.workspace),
        )
        for message in self.session.messages:
            context.append(message["role"], message["content"])
        return context

    def _summarize_history(self, messages: list[dict[str, str]]) -> str | None:
        """L4 摘要器：用同一个 provider 把旧对话压成结构化摘要；失败返回 None（不压缩）。"""
        conversation = "\n".join(
            f"{message['role']}: {message['content'][:4000]}" for message in messages
        )
        prompt = SUMMARY_PROMPT_TEMPLATE.format(conversation=conversation)
        try:
            raw = self.provider.chat(
                [
                    {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                tools=None,
            )
        except Exception:
            return None
        if not isinstance(raw, str):
            return None
        return extract_summary(raw)

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
        return f"""你是一个命令行 Coding Agent。你只能通过下面列出的本地工具工作，所有 path 都必须是相对工作区根目录的路径，统一使用正斜杠（/）。先检查相关文件，再修改代码，最后运行测试确认结果。不要臆测文件内容；工具报错时修正参数或方案。任务针对工作区子目录时，读写文件用相对路径，并给 run_tests 传 cwd 参数让测试在该子目录下执行。run_tests 失败时先读返回的 stdout/stderr 中的断言或错误详情，再决定下一步，不要反复猜测目录结构；优先使用 python -m unittest discover -s tests 或 python -m pytest -q。修改已有文件时优先用 apply_diff 做精准修改（一次可含多个 hunk）；只有新建文件或需要整体重写时才用 write_file。\n\n每次只能返回一个 JSON 对象，不要添加 Markdown 或解释文字：\n工具调用：{{\"type\":\"tool_call\",\"tool\":\"工具名\",\"arguments\":{{...}}}}\n任务完成：{{\"type\":\"final\",\"message\":\"简要说明修改和测试结果\"}}\n\n可用工具：\n{schema}\n"""
