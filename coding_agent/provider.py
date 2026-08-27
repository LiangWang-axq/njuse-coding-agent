from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from typing import Any


class ProviderError(RuntimeError):
    pass


RETRYABLE_HTTP_CODES = {429, 500, 502, 503, 504}


class OpenAICompatibleProvider:
    """Minimal HTTP client; no provider-side file or code-execution tools are used."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: int = 90,
        retries: int = 2,
    ):
        self.endpoint = base_url.rstrip("/") + "/chat/completions"
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.retries = max(0, int(retries))
        self._tools_supported = True

    def _build_request(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None,
        tool_choice: str | None,
        stream: bool,
    ) -> urllib.request.Request:
        if not self.api_key:
            raise ProviderError("未找到 API key，请设置 AGENT_API_KEY 或 OPENAI_API_KEY")
        body: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": 0.1}
        if stream:
            body["stream"] = True
        if tools and self._tools_supported:
            body["tools"] = tools
            body["tool_choice"] = tool_choice or "auto"
        return urllib.request.Request(
            self.endpoint,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

    def chat(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
    ) -> str | dict[str, Any]:
        request = self._build_request(messages, tools, tool_choice, stream=False)
        try:
            payload = self._request_with_retry(request)
        except ProviderError as exc:
            if tools and self._tools_supported and "HTTP 400" in str(exc):
                # Gateway may not support native tool calling; fall back to JSON protocol.
                self._tools_supported = False
                return self.chat(messages)
            raise
        try:
            message = payload["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError("模型响应缺少 choices[0].message") from exc
        if message.get("tool_calls"):
            return {"tool_calls": message["tool_calls"]}
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ProviderError("模型响应没有文本内容")
        return content

    def chat_stream(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Stream Chat Completions and yield text deltas or accumulated tool_calls.

        Yields ``{"type": "text", "delta": ...}`` per content chunk, and finally
        ``{"type": "tool_calls", "tool_calls": [...]}`` when the model asks for
        tools (arguments are re-parsed from the streamed string fragments).
        Transient failures are retried only before the first event arrives.
        """
        request = self._build_request(messages, tools, tool_choice, stream=True)
        try:
            response = self._open_with_retry(request)
        except ProviderError as exc:
            if tools and self._tools_supported and "HTTP 400" in str(exc):
                self._tools_supported = False
                yield from self.chat_stream(messages)
                return
            raise
        pending_calls: dict[int, dict[str, Any]] = {}
        text_chars = 0
        try:
            for data in self._iter_sse_data(response):
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                for event in self._handle_chunk(chunk, pending_calls):
                    if event["type"] == "text":
                        text_chars += len(event["delta"])
                    yield event
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ProviderError(f"流式响应中断: {exc}") from exc
        finally:
            response.close()
        if pending_calls:
            yield {"type": "tool_calls", "tool_calls": self._finalize_tool_calls(pending_calls)}
            return
        if text_chars == 0:
            raise ProviderError("模型响应没有文本内容")

    @staticmethod
    def _iter_sse_data(response) -> Iterator[str]:
        """Yield the payload of each SSE ``data:`` event from a streamed response."""
        buffer: list[str] = []
        for raw_line in response:
            for line in raw_line.decode("utf-8", errors="replace").splitlines():
                if line == "":
                    if buffer:
                        yield "\n".join(buffer)
                        buffer = []
                    continue
                if line.startswith("data:"):
                    buffer.append(line[len("data:"):].strip())
        if buffer:
            yield "\n".join(buffer)

    @staticmethod
    def _handle_chunk(
        chunk: dict[str, Any],
        pending_calls: dict[int, dict[str, Any]],
    ) -> Iterator[dict[str, Any]]:
        choices = chunk.get("choices") or [{}]
        first = choices[0] if isinstance(choices[0], dict) else {}
        delta = first.get("delta", {})
        if not isinstance(delta, dict):
            delta = {}
        content = delta.get("content")
        if isinstance(content, str) and content:
            yield {"type": "text", "delta": content}
        for call in delta.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            index = int(call.get("index", 0))
            slot = pending_calls.setdefault(
                index,
                {"id": "", "function": {"name": "", "arguments": ""}},
            )
            if call.get("id"):
                slot["id"] = call["id"]
            function = call.get("function") or {}
            if function.get("name"):
                slot["function"]["name"] = function["name"]
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                slot["function"]["arguments"] += arguments

    @staticmethod
    def _finalize_tool_calls(pending_calls: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
        calls: list[dict[str, Any]] = []
        for index in sorted(pending_calls):
            slot = pending_calls[index]
            name = slot["function"]["name"]
            if not name:
                raise ProviderError("流式 tool_calls 缺少工具名")
            arguments = slot["function"]["arguments"]
            try:
                parsed = json.loads(arguments) if arguments.strip() else {}
            except json.JSONDecodeError as exc:
                raise ProviderError(f"流式 tool_calls 参数不是合法 JSON: {exc}") from exc
            if not isinstance(parsed, dict):
                raise ProviderError("工具参数必须是 JSON 对象")
            calls.append(
                {
                    "id": slot["id"],
                    "type": "function",
                    "function": {"name": name, "arguments": parsed},
                }
            )
        return calls

    def _open_with_retry(self, request: urllib.request.Request):
        """Open the request once, retrying transient failures with 2s/4s backoff."""
        last_error = "未知错误"
        for attempt in range(self.retries + 1):
            try:
                return urllib.request.urlopen(request, timeout=self.timeout_seconds)
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:500]
                if exc.code in RETRYABLE_HTTP_CODES and attempt < self.retries:
                    last_error = f"HTTP {exc.code}: {detail}"
                    time.sleep(2 ** (attempt + 1))
                    continue
                raise ProviderError(f"模型服务 HTTP {exc.code}: {detail}") from exc
            except urllib.error.URLError as exc:
                if attempt < self.retries:
                    last_error = f"网络错误: {exc}"
                    time.sleep(2 ** (attempt + 1))
                    continue
                raise ProviderError(f"模型服务请求失败: {exc}") from exc
            except TimeoutError as exc:
                if attempt < self.retries:
                    last_error = f"请求超时: {exc}"
                    time.sleep(2 ** (attempt + 1))
                    continue
                raise ProviderError(f"模型服务请求超时: {exc}") from exc
        raise ProviderError(f"模型服务重试 {self.retries} 次后仍失败: {last_error}")

    def _request_with_retry(self, request: urllib.request.Request) -> dict[str, Any]:
        """POST once, retrying transient failures with 2s/4s backoff."""
        with self._open_with_retry(request) as response:
            try:
                return json.loads(response.read().decode("utf-8"))
            except json.JSONDecodeError as exc:
                raise ProviderError(f"模型响应不是合法 JSON: {exc}") from exc
