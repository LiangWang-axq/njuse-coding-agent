from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


class ProviderError(RuntimeError):
    pass


class OpenAICompatibleProvider:
    """Minimal HTTP client; no provider-side file or code-execution tools are used."""

    def __init__(self, base_url: str, api_key: str, model: str, timeout_seconds: int = 90):
        self.endpoint = base_url.rstrip("/") + "/chat/completions"
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds

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
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise ProviderError(f"模型服务 HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise ProviderError(f"模型服务请求失败: {exc}") from exc

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
