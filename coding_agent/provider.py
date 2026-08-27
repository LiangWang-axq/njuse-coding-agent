from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
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

    def chat(self, messages: list[dict[str, str]]) -> str | dict[str, Any]:
        if not self.api_key:
            raise ProviderError("未找到 API key，请设置 AGENT_API_KEY 或 OPENAI_API_KEY")
        body = json.dumps(
            {"model": self.model, "messages": messages, "temperature": 0.1},
            ensure_ascii=False,
        ).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        payload = self._request_with_retry(request)
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

    def _request_with_retry(self, request: urllib.request.Request) -> dict[str, Any]:
        """POST once, retrying transient failures with 2s/4s backoff."""
        last_error = "未知错误"
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    return json.loads(response.read().decode("utf-8"))
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
            except json.JSONDecodeError as exc:
                raise ProviderError(f"模型响应不是合法 JSON: {exc}") from exc
        raise ProviderError(f"模型服务重试 {self.retries} 次后仍失败: {last_error}")
