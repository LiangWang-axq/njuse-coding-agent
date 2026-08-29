"""Terminal rendering helpers: ANSI colors, TTY detection, tool-call lines."""

from __future__ import annotations

import json
import os
import sys
from typing import Any

try:
    import ctypes
except ImportError:  # pragma: no cover
    ctypes = None

RESET = "\x1b[0m"
BOLD = "\x1b[1m"
DIM = "\x1b[2m"
RED = "\x1b[31m"
GREEN = "\x1b[32m"
YELLOW = "\x1b[33m"
CYAN = "\x1b[36m"

_vt_ready = False


def init() -> None:
    """Enable VT processing on Windows consoles so ANSI colors render."""
    global _vt_ready
    if _vt_ready or os.name != "nt" or ctypes is None:
        return
    try:
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
    except Exception:
        pass
    _vt_ready = True


def colors_enabled() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    return bool(getattr(sys.stdout, "isatty", lambda: False)())


def paint(text: str, code: str) -> str:
    return f"{code}{text}{RESET}" if colors_enabled() else text


def bold(text: str) -> str:
    return paint(text, BOLD)


def dim(text: str) -> str:
    return paint(text, DIM)


def red(text: str) -> str:
    return paint(text, RED)


def green(text: str) -> str:
    return paint(text, GREEN)


def yellow(text: str) -> str:
    return paint(text, YELLOW)


def cyan(text: str) -> str:
    return paint(text, CYAN)


def _preview(value: Any, limit: int = 200) -> str:
    """截断任意值并标注总长度，避免工具结果刷屏。"""
    text = str(value)
    if len(text) <= limit:
        return text
    return text[:limit] + f"…（共 {len(text)} 字符）"


def _summarize_result(tool: str, result: dict[str, Any]) -> str:
    """按工具类型给结果做一行摘要（多行时以换行缩进承载输出尾部）。"""
    if not isinstance(result, dict):
        return green("成功") + (f" · {_preview(result)}" if result else "")
    if tool == "run_tests":
        if result.get("passed"):
            return green("通过") + f" · {result.get('command', '')}"
        tail = str((result.get("stdout") or result.get("stderr") or "")[-220:])
        return yellow(f"失败 (returncode {result.get('returncode')})") + f"\n      输出尾: {_preview(tail)}"
    if tool == "read_file":
        content = result.get("content", "")
        detail = f"{result.get('path', '?')} · {len(content)} 字符"
        if result.get("truncated"):
            detail += "（已截断）"
        return green("成功") + f" · {detail}"
    if tool == "list_files":
        files = result.get("files") or []
        detail = f"{len(files)} 个文件"
        if result.get("truncated"):
            detail += "（已达上限）"
        return green("成功") + f" · {detail}"
    if tool == "search_code":
        matches = result.get("matches") or []
        detail = f"{len(matches)} 处匹配"
        if result.get("truncated"):
            detail += "（已达上限）"
        return green("成功") + f" · {detail}"
    if tool == "write_file":
        return green("成功") + f" · {result.get('path', '?')} · {result.get('bytes', '?')} 字节"
    if tool == "replace_in_file":
        return green("成功") + f" · 替换 {result.get('replacements', 0)} 处"
    if tool == "apply_diff":
        return (
            green("成功")
            + f" · 应用 {result.get('hunks_applied', 0)} 个 hunk"
            + f" (+{result.get('added', 0)}/-{result.get('removed', 0)})"
        )
    if tool == "delete_file":
        return green("成功") + f" · 已删除 {result.get('path', '?')}"
    if tool == "move_file":
        return green("成功") + f" · {result.get('source')} → {result.get('target')}"
    if tool == "git_status":
        return green("成功") + f" · {_preview(result.get('status', ''))}"
    return green("成功") + (f" · {_preview(result)}" if result else "")


def format_tool_call(response: Any, result: dict[str, Any], step: int | None = None) -> str:
    """Render one tool execution as a colored multi-line card."""
    prefix = f"[步骤 {step}] " if step else "[工具] "
    arguments = json.dumps(response.arguments or {}, ensure_ascii=False, indent=2)
    if len(arguments) > 240:
        arguments = arguments[:240] + "\n      …（参数较长，已省略）"
    lines = [f"{prefix}{cyan(response.tool)}"]
    lines.append(f"  参数: {arguments}")
    if result.get("ok"):
        lines.append("  结果: " + _summarize_result(response.tool, result.get("result") or {}))
    else:
        lines.append("  结果: " + red(f"错误: {_preview(result.get('error', ''), 200)}"))
    return "\n".join(lines)


def format_compression_stats(stats: dict[str, int]) -> str:
    """把累计压缩统计渲染成一行（/status 用）。"""
    labels = (
        ("L3", "大结果落盘"),
        ("L1", "中间裁切"),
        ("L2", "旧结果压缩"),
        ("L4", "LLM 摘要"),
        ("truncate", "兜底截断"),
    )
    parts = [f"{label} {stats[key]} 次" for key, label in labels if stats.get(key)]
    return " · ".join(parts) if parts else "无"


def print_tool_call(response: Any, result: dict[str, Any], step: int | None = None) -> None:
    print(format_tool_call(response, result, step))
