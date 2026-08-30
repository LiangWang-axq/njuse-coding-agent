from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import ui
from .agent import AgentError, CodingAgent
from .config import Settings, load_settings
from .provider import OpenAICompatibleProvider
from .session import (
    Session,
    SessionError,
    delete_session,
    latest_session_path,
    list_sessions,
    resolve_session,
)
from .workspace import WorkspaceError, resolve_workspace


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
    parser.add_argument("--workspace", "-w", metavar="PATH", help="工作区目录；默认使用当前目录")
    parser.add_argument("--max-steps", type=int, default=None, help="最多模型-工具循环次数")
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument("--resume", action="store_true", help="恢复工作区内最近一次会话的历史")
    resume_group.add_argument("--resume-session", metavar="SELECTOR", help="按序号、ID 或唯一前缀恢复会话")
    parser.add_argument("--list-sessions", action="store_true", help="列出工作区内的历史会话后退出")
    parser.add_argument("--delete-session", metavar="SELECTOR", help="永久删除指定会话后退出")
    parser.add_argument("--yes", action="store_true", help="删除会话时跳过确认")
    args = parser.parse_args()
    _validate_management_args(parser, args)
    try:
        workspace = resolve_workspace(args.workspace or ".", base=Path.cwd())
    except WorkspaceError as exc:
        parser.error(str(exc))
    if args.list_sessions:
        print(ui.format_session_list(list_sessions(workspace)))
        return 0
    if args.delete_session is not None:
        return _delete_session_from_cli(workspace, args.delete_session, args.yes)
    try:
        agent, settings = _create_agent(
            workspace,
            max_steps=args.max_steps,
            resume=args.resume,
            resume_session=args.resume_session,
        )
    except SessionError as exc:
        print(ui.red(f"无法恢复会话: {exc}"))
        return 1
    except WorkspaceError as exc:
        print(ui.red(str(exc)))
        return 1
    task = " ".join(args.task).strip()
    if not task:
        from .repl import run_repl

        def switch_workspace(path_value: str, current_workspace: Path):
            target = resolve_workspace(path_value, base=current_workspace)
            return _create_agent(
                target,
                max_steps=args.max_steps,
                resume=False,
                resume_session=None,
            )

        return run_repl(agent, settings, workspace_switcher=switch_workspace)
    try:
        return _stream_answer(agent, task)
    except AgentError as exc:
        print(ui.red(f"Agent 失败: {exc}"))
        return 1


def _create_agent(
    workspace: Path,
    *,
    max_steps: int | None,
    resume: bool,
    resume_session: str | None,
) -> tuple[CodingAgent, Settings]:
    """Create an Agent whose tools, settings, and session belong to workspace."""
    settings = load_settings(workspace)
    if not settings.api_key:
        raise WorkspaceError(
            f"工作区未找到 API key: {workspace}；请设置 AGENT_API_KEY 或配置父目录 .env"
        )
    provider = OpenAICompatibleProvider(
        settings.base_url,
        settings.api_key,
        settings.model,
        settings.timeout_seconds,
        settings.retries,
    )
    session = _load_resume_session(workspace, resume, resume_session)
    agent = CodingAgent(
        workspace,
        provider,
        max_steps or settings.max_steps,
        settings.context_chars,
        session=session,
    )
    return agent, settings


def _load_resume_session(
    workspace: Path,
    resume: bool,
    selector: str | None = None,
) -> Session | None:
    """--resume 时加载最近一次会话；找不到或损坏则降级为新会话。"""
    if not resume and selector is None:
        return None
    if selector is not None:
        selected = resolve_session(workspace, selector)
        path = selected.path
    else:
        path = latest_session_path(workspace)
    if path is None:
        print(ui.yellow("[提示] 工作区内没有历史会话，将开始新会话。"))
        return None
    try:
        session = Session.load(path)
    except (OSError, SessionError) as exc:
        if selector is not None:
            raise SessionError(str(exc)) from exc
        print(ui.yellow(f"[提示] 会话文件无法恢复（{exc}），将开始新会话。"))
        return None
    print(ui.dim(f"[已恢复会话] {session.session_id}（{session.message_count} 条历史消息）"))
    return session


def _validate_management_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    has_task = bool(" ".join(args.task).strip())
    if args.list_sessions and (has_task or args.resume or args.resume_session or args.delete_session):
        parser.error("--list-sessions 不能与任务、恢复或删除参数同时使用")
    if args.delete_session and (has_task or args.resume or args.resume_session or args.list_sessions):
        parser.error("--delete-session 不能与任务、恢复或列表参数同时使用")
    if args.yes and args.delete_session is None:
        parser.error("--yes 只能与 --delete-session 同时使用")


def _delete_session_from_cli(workspace: Path, selector: str, assume_yes: bool) -> int:
    try:
        selected = resolve_session(workspace, selector, require_valid=False)
    except SessionError as exc:
        print(ui.red(f"无法删除会话: {exc}"))
        return 1
    print(
        f"将永久删除会话 {selected.session_id} · "
        f"{selected.message_count if selected.message_count is not None else '?'} 条消息 · {selected.preview}"
    )
    if not assume_yes:
        try:
            confirmed = input("确认永久删除？[y/N] ").strip().lower() in {"y", "yes"}
        except (EOFError, KeyboardInterrupt):
            print()
            confirmed = False
        if not confirmed:
            print(ui.yellow("[已取消] 会话未删除。"))
            return 1
    try:
        deleted = delete_session(workspace, selector)
    except SessionError as exc:
        print(ui.red(f"无法删除会话: {exc}"))
        return 1
    print(ui.green(f"[已删除会话] {deleted.session_id}"))
    return 0
