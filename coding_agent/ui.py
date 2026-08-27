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


def format_tool_call(response: Any, result: dict[str, Any], step: int | None = None) -> str:
    """Render one tool execution as a colored, single-line summary."""
    arguments = json.dumps(response.arguments or {}, ensure_ascii=False)
    if len(arguments) > 120:
        arguments = arguments[:120] + "…"
    prefix = f"[步骤 {step}] " if step else "[工具] "
    call = f"{prefix}{cyan(response.tool)}({arguments}) -> "
    if result.get("ok"):
        if response.tool == "run_tests":
            test_result = result.get("result") or {}
            if test_result.get("passed"):
                return call + green("通过")
            return call + green(f"失败 (returncode {test_result.get('returncode')})")
        return call + green("OK")
    return call + red(f"错误: {str(result.get('error', ''))[:100]}")


def print_tool_call(response: Any, result: dict[str, Any], step: int | None = None) -> None:
    print(format_tool_call(response, result, step))
