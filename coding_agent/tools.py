from __future__ import annotations

import json
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable


class WorkspaceViolation(ValueError):
    pass


class ToolError(RuntimeError):
    pass


class WorkspaceTools:
    """Local tools whose every filesystem operation is rooted in one directory."""

    MAX_READ_CHARS = 30_000
    MAX_WRITE_CHARS = 120_000

    def __init__(self, workspace: Path):
        self.root = workspace.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, value: str) -> Path:
        if not isinstance(value, str) or not value.strip():
            raise WorkspaceViolation("path 必须是非空字符串")
        if "\x00" in value:
            raise WorkspaceViolation("path 不能包含空字符")
        candidate = Path(value)
        if candidate.is_absolute() or re.match(r"^[A-Za-z]:", value) or value.startswith(("\\\\", "/")):
            raise WorkspaceViolation("只允许使用工作区内的相对路径")
        resolved = (self.root / candidate).resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise WorkspaceViolation("路径不能访问工作区之外") from exc
        if resolved.name == ".env":
            raise WorkspaceViolation("禁止通过 Agent 写入 .env")
        return resolved

    def list_files(self, path: str = ".") -> dict[str, Any]:
        base = self._path(path)
        if not base.exists():
            raise ToolError(f"目录不存在: {path}")
        if not base.is_dir():
            raise ToolError(f"不是目录: {path}")
        files = []
        for item in sorted(base.rglob("*")):
            if any(part in {".git", "__pycache__", ".pytest_cache"} for part in item.parts):
                continue
            if item.is_file():
                files.append(str(item.relative_to(self.root)))
            if len(files) >= 200:
                break
        return {"path": path, "files": files, "truncated": len(files) >= 200}

    def read_file(self, path: str) -> dict[str, Any]:
        target = self._path(path)
        if not target.is_file():
            raise ToolError(f"文件不存在: {path}")
        try:
            content = target.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ToolError("只支持 UTF-8 文本文件") from exc
        truncated = len(content) > self.MAX_READ_CHARS
        return {"path": path, "content": content[: self.MAX_READ_CHARS], "truncated": truncated}

    def search_code(self, query: str, path: str = ".") -> dict[str, Any]:
        if not isinstance(query, str) or not query:
            raise ToolError("query 必须是非空字符串")
        base = self._path(path)
        if not base.exists():
            raise ToolError(f"搜索路径不存在: {path}")
        results: list[str] = []
        candidates = [base] if base.is_file() else base.rglob("*")
        for item in candidates:
            if not item.is_file() or any(part in {".git", "__pycache__", ".pytest_cache"} for part in item.parts):
                continue
            try:
                lines = item.read_text(encoding="utf-8").splitlines()
            except (UnicodeDecodeError, OSError):
                continue
            for number, line in enumerate(lines, 1):
                if query.lower() in line.lower():
                    results.append(f"{item.relative_to(self.root)}:{number}: {line[:240]}")
                    if len(results) >= 100:
                        return {"query": query, "matches": results, "truncated": True}
        return {"query": query, "matches": results, "truncated": False}

    def delete_file(self, path: str) -> dict[str, Any]:
        target = self._path(path)
        if not target.is_file():
            raise ToolError(f"不是文件或不存在: {path}")
        target.unlink()
        return {"path": path, "deleted": True}

    def move_file(self, source: str, target: str) -> dict[str, Any]:
        src = self._path(source)
        dst = self._path(target)
        if not src.is_file():
            raise ToolError(f"源文件不存在: {source}")
        if dst.exists():
            raise ToolError(f"目标已存在，拒绝覆盖: {target}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.replace(dst)
        return {"source": source, "target": target, "moved": True}

    def git_status(self) -> dict[str, Any]:
        try:
            status = subprocess.run(
                ["git", "status", "--short"],
                cwd=self.root,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
                shell=False,
            )
            log = subprocess.run(
                ["git", "log", "--oneline", "-5"],
                cwd=self.root,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
                shell=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ToolError("git 命令超时") from exc
        errors = (status.stderr + log.stderr).strip()
        if status.returncode != 0 and log.returncode != 0:
            raise ToolError(errors or "当前目录不是 git 仓库")
        return {
            "status": status.stdout.strip() or "(clean)",
            "recent_commits": log.stdout.strip() or "(no commits yet)",
            "errors": errors or "",
        }

    def write_file(self, path: str, content: str) -> dict[str, Any]:
        target = self._path(path)
        if not isinstance(content, str):
            raise ToolError("content 必须是字符串")
        if len(content) > self.MAX_WRITE_CHARS:
            raise ToolError("文件内容超过单次写入上限")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8", newline="\n")
        return {"path": path, "bytes": target.stat().st_size, "written": True}

    def replace_in_file(self, path: str, old_text: str, new_text: str, expected_count: int = 1) -> dict[str, Any]:
        current = self.read_file(path)["content"]
        count = current.count(old_text)
        if count != expected_count:
            raise ToolError(f"期望替换 {expected_count} 次，实际找到 {count} 次")
        self.write_file(path, current.replace(old_text, new_text))
        return {
            "path": path,
            "replacements": count,
            "written": True,
        }

    def run_tests(self, command: str) -> dict[str, Any]:
        args = self._safe_test_command(command)
        executable = args[0].lower()
        if executable in {"python", "python.exe", "py", "py.exe"}:
            args[0] = sys.executable
        elif shutil.which(args[0]) is None:
            raise ToolError(f"找不到测试命令: {args[0]}")
        try:
            completed = subprocess.run(
                args,
                cwd=self.root,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=120,
                shell=False,
            )
        except subprocess.TimeoutExpired as exc:
            return {"command": command, "returncode": -1, "stdout": str(exc.stdout or ""), "stderr": "测试命令超时"}
        return {
            "command": command,
            "returncode": completed.returncode,
            "passed": completed.returncode == 0,
            "stdout": completed.stdout[-12_000:],
            "stderr": completed.stderr[-12_000:],
        }

    @staticmethod
    def _safe_test_command(command: str) -> list[str]:
        if not isinstance(command, str) or not command.strip():
            raise ToolError("command 必须是非空字符串")
        if any(mark in command for mark in [";", "&&", "||", "|", ">", "<", "`", "$", "\n", "\r"]):
            raise ToolError("测试命令不允许 shell 拼接、重定向或脚本展开")
        try:
            args = shlex.split(command, posix=False)
        except ValueError as exc:
            raise ToolError(f"无法解析测试命令: {exc}") from exc
        if not args:
            raise ToolError("command 必须是非空字符串")
        executable = args[0].lower().strip('"')
        allowed = {"python", "python.exe", "py", "py.exe", "pytest", "pytest.exe"}
        if executable not in allowed:
            raise ToolError("仅允许 python/pytest 测试命令")
        if executable.startswith("python") or executable.startswith("py"):
            if len(args) < 3 or args[1] != "-m" or args[2] not in {"unittest", "pytest", "compileall"}:
                raise ToolError("Python 仅允许 -m unittest、-m pytest 或 -m compileall")
        for arg in args[1:]:
            clean = arg.strip('"')
            if Path(clean).is_absolute() or re.match(r"^[A-Za-z]:", clean) or ".." in Path(clean).parts:
                raise ToolError("测试命令参数不能指向工作区之外")
        return args


ToolFunction = Callable[..., dict[str, Any]]


def tool_schemas() -> list[dict[str, Any]]:
    return [
        {
            "name": "list_files",
            "description": "列出工作区内的文件",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "相对目录路径，默认 ."}},
                "required": [],
            },
        },
        {
            "name": "read_file",
            "description": "读取工作区内 UTF-8 文本文件",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "相对文件路径"}},
                "required": ["path"],
            },
        },
        {
            "name": "search_code",
            "description": "在工作区文本文件中搜索字符串",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索词"},
                    "path": {"type": "string", "description": "相对目录或文件路径，默认 ."},
                },
                "required": ["query"],
            },
        },
        {
            "name": "write_file",
            "description": "在工作区内创建或完整写入文本文件",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "相对文件路径"},
                    "content": {"type": "string", "description": "完整文件内容"},
                },
                "required": ["path", "content"],
            },
        },
        {
            "name": "replace_in_file",
            "description": "按精确文本替换修改文件",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "相对文件路径"},
                    "old_text": {"type": "string", "description": "原文"},
                    "new_text": {"type": "string", "description": "新文"},
                    "expected_count": {"type": "integer", "description": "期望替换次数，默认 1"},
                },
                "required": ["path", "old_text", "new_text"],
            },
        },
        {
            "name": "run_tests",
            "description": "在工作区根目录运行受控测试命令",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "python -m unittest ... 或 python -m pytest ..."}
                },
                "required": ["command"],
            },
        },
        {
            "name": "delete_file",
            "description": "删除工作区内的单个文本文件",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "相对文件路径"}},
                "required": ["path"],
            },
        },
        {
            "name": "move_file",
            "description": "在工作区内移动或重命名文件，拒绝覆盖已存在目标",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "源相对文件路径"},
                    "target": {"type": "string", "description": "目标相对文件路径"},
                },
                "required": ["source", "target"],
            },
        },
        {
            "name": "git_status",
            "description": "只读查看当前 git 工作区状态与最近 5 条提交",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    ]


def openai_tools() -> list[dict[str, Any]]:
    """OpenAI-compatible tool definitions built from the local tool schemas."""
    return [
        {
            "type": "function",
            "function": {
                "name": schema["name"],
                "description": schema["description"],
                "parameters": schema["parameters"],
            },
        }
        for schema in tool_schemas()
    ]


class ToolRegistry:
    def __init__(self, workspace: Path):
        impl = WorkspaceTools(workspace)
        self._tools: dict[str, ToolFunction] = {
            "list_files": impl.list_files,
            "read_file": impl.read_file,
            "search_code": impl.search_code,
            "write_file": impl.write_file,
            "replace_in_file": impl.replace_in_file,
            "run_tests": impl.run_tests,
            "delete_file": impl.delete_file,
            "move_file": impl.move_file,
            "git_status": impl.git_status,
        }

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        function = self._tools.get(name)
        if function is None:
            raise ToolError(f"未知工具: {name}")
        try:
            return function(**arguments)
        except TypeError as exc:
            raise ToolError(f"工具参数错误: {exc}") from exc
