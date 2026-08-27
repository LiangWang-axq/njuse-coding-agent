from __future__ import annotations

import json


class ContextManager:
    """Keeps the message history within a character budget.

    The system prompt is always preserved. When the history grows beyond
    ``max_chars``, the oldest non-system messages are dropped and replaced by
    one compact summary message, so the model still knows what has been done.
    A single oversized message is truncated in place as a last resort.
    """

    def __init__(self, system_prompt: str, max_chars: int = 16_000):
        if max_chars < 1024:
            raise ValueError("max_chars 至少为 1024")
        self.max_chars = max_chars
        self._messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt}
        ]
        self.compressed = False

    @property
    def messages(self) -> list[dict[str, str]]:
        return [dict(message) for message in self._messages]

    def append(self, role: str, content: str) -> None:
        if not isinstance(content, str):
            raise TypeError("content 必须是字符串")
        self._messages.append({"role": role, "content": content})
        self._trim()

    def _total_chars(self) -> int:
        return sum(len(message["content"]) for message in self._messages)

    def _trim(self) -> None:
        if self._total_chars() <= self.max_chars:
            return
        dropped: list[dict[str, str]] = []
        while self._total_chars() > self.max_chars and len(self._messages) > 2:
            dropped.append(self._messages.pop(1))
        if self._total_chars() > self.max_chars:
            self._truncate_oversized_message()
        summary = self._build_summary(dropped)
        if summary:
            self._messages.insert(1, {"role": "user", "content": summary})
            self.compressed = True

    def _truncate_oversized_message(self) -> None:
        for index in range(len(self._messages) - 1, -1, -1):
            content = self._messages[index]["content"]
            if len(content) > self.max_chars:
                self._messages[index]["content"] = (
                    content[: self.max_chars] + "\n…（内容过长，已截断）"
                )
                return

    def _build_summary(self, dropped: list[dict[str, str]]) -> str:
        lines: list[str] = []
        for message in reversed(dropped):
            content = message.get("content", "")
            if message.get("role") != "user" or not content.startswith("工具结果 "):
                continue
            try:
                payload = json.loads(content[len("工具结果 ") :])
            except json.JSONDecodeError:
                continue
            tool = payload.get("tool", "?")
            ok = bool(payload.get("ok", True))
            result = payload.get("result")
            brief = result if isinstance(result, dict) else {"value": result}
            safe_brief = {key: value for key, value in brief.items() if key != "content"}
            detail = json.dumps(safe_brief, ensure_ascii=False)
            if len(detail) > 180:
                detail = detail[:180] + "…"
            lines.append(f"- {tool}: {'成功' if ok else '失败'} {detail}")
            if len(lines) >= 20:
                break
        if not lines:
            return ""
        header = "（已压缩历史：较早步骤的结果摘要如下，请基于当前工作区状态继续任务。）"
        summary = header + "\n" + "\n".join(lines)
        return summary[:1200]
