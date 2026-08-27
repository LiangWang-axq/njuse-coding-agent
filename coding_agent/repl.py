"""Interactive terminal chat loop (REPL) for the coding agent."""

from __future__ import annotations

import sys

from . import ui
from .agent import AgentError, CodingAgent
from .config import Settings

HELP_TEXT = (
    "可用命令：\n"
    "  /help     显示本帮助\n"
    "  /status   查看工作区、模型与累计步骤\n"
    "  /new      清空对话历史，开始新任务\n"
    "  /exit     退出（等价 /quit）\n"
    "\n"
    "直接输入任务描述即可开始对话；Ctrl+C 可取消当前这一轮。"
)


def run_repl(agent: CodingAgent, settings: Settings) -> int:
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
        if command == "/status":
            print(f"工作区: {agent.workspace}")
            print(f"模型: {settings.model}")
            print(f"API 端点: {settings.base_url}")
            print(f"累计步骤: {total_steps}")
            continue
        if command.startswith("/"):
            print(ui.yellow(f"未知命令: {task}（输入 /help 查看可用命令）"))
            continue
        steps = _run_turn(agent, task)
        if steps is not None:
            total_steps += steps
            print(ui.dim(f"（本轮 {steps} 步 · 累计 {total_steps} 步）"))


def _run_turn(agent: CodingAgent, task: str) -> int | None:
    """Run one user turn; returns step count, or None when the turn was aborted."""
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
    return result.steps
