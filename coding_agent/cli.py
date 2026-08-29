from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import ui
from .agent import AgentError, CodingAgent
from .config import load_settings
from .provider import OpenAICompatibleProvider
from .session import Session, SessionError, latest_session_path


def _stream_answer(agent: CodingAgent, task: str) -> int:
    """Run one task in one-shot mode, streaming assistant text and tool lines."""
    state = {"streamed": False}

    def on_text(delta: str) -> None:
        if not state["streamed"]:
            sys.stdout.write(ui.bold("Agent > "))
            state["streamed"] = True
        sys.stdout.write(delta)
        sys.stdout.flush()

    result = agent.run(
        task,
        on_step=lambda step, response, outcome: ui.print_tool_call(response, outcome, step=step),
        on_text=on_text,
        on_context=lambda events: print(ui.dim("[上下文] " + " · ".join(events))),
    )
    if result.streamed_text:
        print()
    else:
        print(ui.bold("Agent > ") + result.message)
    if not result.success:
        print(ui.red(f"[未完成] {result.message}"))
    token = ""
    if result.usage:
        token = (
            f"；token prompt {result.usage.get('prompt_tokens', '?')} / "
            f"completion {result.usage.get('completion_tokens', '?')} / "
            f"预算 {result.budget_tokens}"
        )
    print(ui.dim(f"步骤数: {result.steps}{token}；工作区: {agent.workspace}"))
    return 0 if result.success else 2


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ui.init()
    parser = argparse.ArgumentParser(description="Framework-free command-line Coding Agent")
    parser.add_argument("task", nargs="*", help="要完成的编程任务；不填则进入交互对话模式")
    parser.add_argument("--max-steps", type=int, default=None, help="最多模型-工具循环次数")
    parser.add_argument("--resume", action="store_true", help="恢复工作区内最近一次会话的历史")
    args = parser.parse_args()
    workspace = Path.cwd().resolve()
    settings = load_settings(workspace)
    if not settings.api_key:
        print("未找到 API key，请设置 AGENT_API_KEY 或 OPENAI_API_KEY（参考 .env.example）")
        return 1
    provider = OpenAICompatibleProvider(
        settings.base_url,
        settings.api_key,
        settings.model,
        settings.timeout_seconds,
        settings.retries,
    )
    session = _load_resume_session(workspace, args.resume)
    agent = CodingAgent(
        workspace,
        provider,
        args.max_steps or settings.max_steps,
        settings.context_chars,
        session=session,
    )
    task = " ".join(args.task).strip()
    if not task:
        from .repl import run_repl

        return run_repl(agent, settings)
    try:
        return _stream_answer(agent, task)
    except AgentError as exc:
        print(ui.red(f"Agent 失败: {exc}"))
        return 1


def _load_resume_session(workspace: Path, resume: bool) -> Session | None:
    """--resume 时加载最近一次会话；找不到或损坏则降级为新会话。"""
    if not resume:
        return None
    path = latest_session_path(workspace)
    if path is None:
        print(ui.yellow("[提示] 工作区内没有历史会话，将开始新会话。"))
        return None
    try:
        session = Session.load(path)
    except SessionError as exc:
        print(ui.yellow(f"[提示] 会话文件无法恢复（{exc}），将开始新会话。"))
        return None
    print(ui.dim(f"[已恢复会话] {session.session_id}（{session.message_count} 条历史消息）"))
    return session
