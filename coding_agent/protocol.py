from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


class ProtocolError(ValueError):
    """Raised when a model response cannot be interpreted safely."""


@dataclass(frozen=True)
class ParsedResponse:
    kind: str
    tool: str | None = None
    arguments: dict[str, Any] | None = None
    message: str = ""
    calls: tuple["ParsedResponse", ...] = ()


def _first_json_object(text: str) -> Any | None:
    start = text.find("{")
    while start >= 0:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : index + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


def parse_model_response(raw: str | dict[str, Any]) -> ParsedResponse:
    if isinstance(raw, dict):
        payload: Any = raw
    else:
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(lines[1:-1]) if len(lines) >= 3 else text
        payload = _first_json_object(text)
        if payload is None:
            if raw.strip():
                return ParsedResponse(kind="final", message=raw.strip())
            raise ProtocolError("模型返回了空内容")

    if not isinstance(payload, dict):
        raise ProtocolError("模型输出必须是 JSON 对象")

    native_calls = payload.get("tool_calls")
    if isinstance(native_calls, list) and native_calls:
        calls = tuple(_native_call(call) for call in native_calls[:8])
        if len(calls) == 1:
            return calls[0]
        return ParsedResponse(kind="tool", calls=calls)

    kind = payload.get("type", payload.get("action"))
    if kind in {"tool", "tool_call", "call"}:
        return _tool_response(
            payload.get("tool", payload.get("name")),
            payload.get("arguments", payload.get("args", {})),
        )
    if kind in {"final", "answer", "done"}:
        message = payload.get("message", payload.get("content", ""))
        return ParsedResponse(kind="final", message=str(message))
    raise ProtocolError("模型输出缺少有效的 type/action 字段")


def _tool_response(name: Any, arguments: Any) -> ParsedResponse:
    if not isinstance(name, str) or not name:
        raise ProtocolError("工具调用缺少工具名")
    if not isinstance(arguments, dict):
        raise ProtocolError("工具参数必须是 JSON 对象")
    return ParsedResponse(kind="tool", tool=name, arguments=arguments)


def _native_call(call: Any) -> ParsedResponse:
    if not isinstance(call, dict):
        raise ProtocolError("tool_calls 元素必须是对象")
    function = call.get("function")
    if not isinstance(function, dict):
        raise ProtocolError("tool_calls 缺少 function 字段")
    name = function.get("name")
    if not isinstance(name, str) or not name:
        raise ProtocolError("工具调用缺少工具名")
    arguments = function.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise ProtocolError(f"tool_calls 参数不是合法 JSON: {exc}") from exc
    if not isinstance(arguments, dict):
        raise ProtocolError("工具参数必须是 JSON 对象")
    return ParsedResponse(kind="tool", tool=name, arguments=arguments)
