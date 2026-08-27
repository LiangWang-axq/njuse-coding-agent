from __future__ import annotations

import argparse
from pathlib import Path

from .agent import AgentError, CodingAgent
from .config import load_settings
from .provider import OpenAICompatibleProvider


def main() -> int:
    parser = argparse.ArgumentParser(description="Framework-free command-line Coding Agent")
    parser.add_argument("task", nargs="*", help="要完成的编程任务；不填则进入交互输入")
    parser.add_argument("--max-steps", type=int, default=None, help="最多模型-工具循环次数")
    args = parser.parse_args()
    workspace = Path.cwd().resolve()
    task = " ".join(args.task).strip() or input("任务: ").strip()
    settings = load_settings(workspace)
    provider = OpenAICompatibleProvider(settings.base_url, settings.api_key, settings.model, settings.timeout_seconds)
    agent = CodingAgent(workspace, provider, args.max_steps or settings.max_steps)
    try:
        result = agent.run(task)
    except AgentError as exc:
        print(f"Agent 失败: {exc}")
        return 1
    print(result.message)
    print(f"步骤数: {result.steps}；工作区: {workspace}")
    return 0 if result.success else 2
