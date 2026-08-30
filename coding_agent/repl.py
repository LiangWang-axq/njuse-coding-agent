"""Interactive terminal chat loop (REPL) for the coding agent."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

from . import ui
from .agent import AgentError, CodingAgent
from .config import Settings
from .session import Session, SessionError, delete_session, list_sessions, resolve_session
from .workspace import WorkspaceError

HELP_TEXT = (
    "可用命令：\n"
    "  /help     显示本帮助\n"
    "  /status   查看工作区、模型、会话与累计步骤\n"
    "  /sessions 查看历史会话列表\n"
    "  /workspace [路径] 选择或切换工作区\n"
    "  /resume <序号|ID>  切换到历史会话\n"
    "  /delete <序号|ID>  永久删除历史会话\n"
    "  /new      清空历史并开始新会话（旧会话可用 --resume 恢复）\n"
    "  /exit     退出（等价 /quit）\n"
    "\n"
    "直接输入任务描述即可开始对话；Ctrl+C 可取消当前这一轮。"
)


def run_repl(
    agent: CodingAgent,
    settings: Settings,
    *,
    workspace_switcher: Callable[[str, Path], tuple[CodingAgent, Settings]] | None = None,
) -> int:
    print(ui.bold("Coding Agent 交互模式"))
    print(f"模型: {settings.model} · 工作区: {agent.workspace}")
    print("直接输入任务开始对话，输入 /help 查看命令。")
    total_steps = 0
    while True:
        try:
            task = input(ui.cyan("你 > ")).strip()
        except EOFError:
            print()
            return 0
        except KeyboardInterrupt:
            print()
            return 0
        if not task:
            continue
        command = task.lower()
        if command in {"/exit", "/quit"}:
            print("再见。")
            return 0
        if command == "/help":
            print(HELP_TEXT)
            continue
        if command == "/new":
            agent.reset()
            total_steps = 0
            print(ui.yellow("[已清空] 新对话开始。"))
            continue
        if command == "/sessions":
            print(ui.format_session_list(list_sessions(agent.workspace), agent.session.path))
            continue
        if command == "/workspace":
            if workspace_switcher is None:
                print(ui.yellow("当前入口未配置工作区切换。"))
                continue
            try:
                path_value = input(f"工作区路径 [{agent.workspace}] > ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                continue
            if not path_value:
                print(f"当前工作区: {agent.workspace}")
                continue
            result = _try_switch_workspace(path_value, agent, workspace_switcher)
            if result is not None:
                agent, settings = result
                total_steps = 0
            continue
        if command.startswith("/workspace "):
            if workspace_switcher is None:
                print(ui.yellow("当前入口未配置工作区切换。"))
                continue
            path_value = task.split(maxsplit=1)[1].strip()
            result = _try_switch_workspace(path_value, agent, workspace_switcher)
            if result is not None:
                agent, settings = result
                total_steps = 0
            continue
        if command == "/resume":
            print("用法: /resume <序号|会话 ID|唯一前缀>")
            continue
        if command.startswith("/resume "):
            selector = task.split(maxsplit=1)[1].strip()
            if _switch_session(agent, selector):
                total_steps = 0
            continue
        if command == "/delete":
            print("用法: /delete <序号|会话 ID|唯一前缀>")
            continue
        if command.startswith("/delete "):
            selector = task.split(maxsplit=1)[1].strip()
            if _delete_session_from_repl(agent, selector):
                total_steps = 0
            continue
        if command == "/status":
            print(f"工作区: {agent.workspace}")
            print(f"模型: {settings.model}")
            print(f"API 端点: {settings.base_url}")
            print(f"会话: {agent.session.session_id}")
            print(f"会话文件: {agent.session.path}")
            budget = getattr(agent, "context_chars", None)
            if budget:
                print(f"上下文预算: {budget // 4} token")
            compression = getattr(agent, "compression_stats", None)
            if compression:
                print(f"压缩统计: {ui.format_compression_stats(compression)}")
            usage = getattr(agent, "last_usage", None)
            if usage:
                print(
                    "最近 usage: "
                    f"prompt {usage.get('prompt_tokens', '?')} · "
                    f"completion {usage.get('completion_tokens', '?')} · "
                    f"total {usage.get('total_tokens', '?')}"
                )
            print(f"累计步骤: {total_steps}")
            continue
        if command.startswith("/"):
            print(ui.yellow(f"未知命令: {task}（输入 /help 查看可用命令）"))
            continue
        result = _run_turn(agent, task)
        if result is not None:
            total_steps += result.steps
            print(ui.dim(_turn_summary(result, total_steps)))


def _turn_summary(result, total_steps: int) -> str:
    """本轮小结：步数 + token 用量/估算 + 预算。"""
    if result.usage:
        prompt = result.usage.get("prompt_tokens", "?")
        completion = result.usage.get("completion_tokens", "?")
        token = f"token prompt {prompt} / completion {completion}"
    else:
        token = f"估算 ~{result.estimated_tokens} token"
    return f"（本轮 {result.steps} 步 · 累计 {total_steps} 步 · {token} · 预算 {result.budget_tokens}）"


def _try_switch_workspace(
    path_value: str,
    agent: CodingAgent,
    workspace_switcher: Callable[[str, Path], tuple[CodingAgent, Settings]],
):
    try:
        switched = workspace_switcher(path_value, agent.workspace)
    except (WorkspaceError, OSError, SessionError) as exc:
        print(ui.red(f"无法切换工作区: {exc}"))
        return None
    except Exception as exc:
        print(ui.red(f"无法切换工作区: {exc}"))
        return None
    new_agent, new_settings = switched
    print(ui.green(f"[已切换工作区] {new_agent.workspace}"))
    print(f"模型: {new_settings.model} · 会话: {new_agent.session.session_id}")
    return new_agent, new_settings


def _run_turn(agent: CodingAgent, task: str):
    """Run one user turn; returns AgentResult, or None when the turn was aborted."""
    state = {"streamed": False}

    def on_text(delta: str) -> None:
        if not state["streamed"]:
            sys.stdout.write(ui.bold("Agent > "))
            state["streamed"] = True
        sys.stdout.write(delta)
        sys.stdout.flush()

    try:
        result = agent.run(
            task,
            on_step=lambda _step, response, outcome: ui.print_tool_call(response, outcome),
            on_text=on_text,
            on_context=lambda events: print(ui.dim("[上下文] " + " · ".join(events))),
        )
    except KeyboardInterrupt:
        print()
        print(ui.yellow("[已中断] 本轮已取消，历史保留。"))
        return None
    except AgentError as exc:
        print(ui.red(f"[错误] {exc}"))
        return None
    except Exception as exc:
        print(ui.red(f"[异常] {exc}"))
        return None
    if result.streamed_text:
        print()
    else:
        print(ui.bold("Agent > ") + result.message)
    if not result.success:
        print(ui.red(f"[未完成] {result.message}"))
    return result


def _switch_session(agent: CodingAgent, selector: str) -> bool:
    try:
        selected = resolve_session(agent.workspace, selector)
        if selected.path.resolve() == agent.session.path.resolve():
            print(ui.dim(f"[当前会话] {selected.session_id}"))
            return False
        session = Session.load(selected.path)
    except (OSError, SessionError) as exc:
        print(ui.red(f"无法恢复会话: {exc}"))
        return False
    agent.switch_session(session)
    print(ui.green(f"[已切换会话] {session.session_id}（{session.message_count} 条历史消息）"))
    return True


def _delete_session_from_repl(agent: CodingAgent, selector: str) -> bool:
    try:
        selected = resolve_session(agent.workspace, selector, require_valid=False)
    except SessionError as exc:
        print(ui.red(f"无法删除会话: {exc}"))
        return False
    try:
        confirmed = input(f"确认永久删除会话 {selected.session_id}？[y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        confirmed = ""
    if confirmed not in {"y", "yes"}:
        print(ui.yellow("[已取消] 会话未删除。"))
        return False
    is_current = selected.path.resolve() == agent.session.path.resolve()
    try:
        deleted = delete_session(agent.workspace, selector)
    except SessionError as exc:
        print(ui.red(f"无法删除会话: {exc}"))
        return False
    if is_current:
        agent.reset()
        print(ui.green(f"[已删除会话] {deleted.session_id}；已开始新会话。"))
        return True
    print(ui.green(f"[已删除会话] {deleted.session_id}"))
    return False
