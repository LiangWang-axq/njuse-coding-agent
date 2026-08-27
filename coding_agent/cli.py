from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .agent import AgentError, CodingAgent
from .config import load_settings
from .protocol import ParsedResponse
from .provider import OpenAICompatibleProvider


def _print_step(step: int, response: ParsedResponse, result: dict) -> None:
    arguments = json.dumps(response.arguments or {}, ensure_ascii=False)
    if len(arguments) > 120:
        arguments = arguments[:120] + "…"
    if result.get("ok"):
        status = "OK"
    else:
        status = f"错误: {str(result.get('error', ''))[:100]}"
    print(f"[步骤 {step}] {response.tool}({arguments}) -> {status}")


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Framework-free command-line Coding Agent")
    parser.add_argument("task", nargs="*", help="要完成的编程任务；不填则进入交互输入")
    parser.add_argument("--max-steps", type=int, default=None, help="最多模型-工具循环次数")
    args = parser.parse_args()
    workspace = Path.cwd().resolve()
    task = " ".join(args.task).strip() or input("任务: ").strip()
    settings = load_settings(workspace)
    provider = OpenAICompatibleProvider(settings.base_url, settings.api_key, settings.model, settings.timeout_seconds)
    agent = CodingAgent(
        workspace,
        provider,
        args.max_steps or settings.max_steps,
        settings.context_chars,
    )
    try:
        result = agent.run(task, on_step=_print_step)
    except AgentError as exc:
        print(f"Agent 失败: {exc}")
        return 1
    print(result.message)
    print(f"步骤数: {result.steps}；工作区: {workspace}")
    return 0 if result.success else 2
