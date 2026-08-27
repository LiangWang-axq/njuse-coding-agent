from __future__ import annotations

import io
import json
import unittest
from unittest import mock
from urllib.error import HTTPError

from coding_agent.provider import OpenAICompatibleProvider, ProviderError


class FakeResponse:
    def __init__(self, payload: dict):
        self._payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self) -> bytes:
        return self._payload


def http_error(code: int) -> HTTPError:
    return HTTPError("https://example.test/v1/chat/completions", code, "error", {}, io.BytesIO(b'{"error":"boom"}'))


class ProviderRetryTests(unittest.TestCase):
    def _provider(self, retries: int = 2) -> OpenAICompatibleProvider:
        return OpenAICompatibleProvider(
            "https://example.test/v1",
            "sk-test",
            "deepseek-chat",
            timeout_seconds=30,
            retries=retries,
        )

    def test_retries_then_succeeds_on_transient_errors(self):
        provider = self._provider()
        with mock.patch("coding_agent.provider.time.sleep") as sleep, mock.patch(
            "coding_agent.provider.urllib.request.urlopen",
            side_effect=[
                http_error(429),
                http_error(500),
                FakeResponse({"choices": [{"message": {"content": "ok"}}]}),
            ],
        ) as urlopen:
            result = provider.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(result, "ok")
        self.assertEqual(urlopen.call_count, 3)
        self.assertEqual(sleep.call_args_list, [mock.call(2), mock.call(4)])

    def test_no_retry_on_400(self):
        provider = self._provider()
        with mock.patch("coding_agent.provider.time.sleep") as sleep, mock.patch(
            "coding_agent.provider.urllib.request.urlopen",
            side_effect=[http_error(400)],
        ) as urlopen:
            with self.assertRaises(ProviderError) as ctx:
                provider.chat([{"role": "user", "content": "hi"}])
        self.assertIn("HTTP 400", str(ctx.exception))
        self.assertEqual(urlopen.call_count, 1)
        self.assertFalse(sleep.called)

    def test_succeeds_without_retry(self):
        provider = self._provider()
        with mock.patch(
            "coding_agent.provider.urllib.request.urlopen",
            return_value=FakeResponse({"choices": [{"message": {"content": "ok"}}]}),
        ) as urlopen:
            result = provider.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(urlopen.call_count, 1)
        self.assertEqual(result, "ok")

    def test_gives_up_after_configured_retries(self):
        provider = self._provider(retries=1)
        with mock.patch("coding_agent.provider.time.sleep") as sleep, mock.patch(
            "coding_agent.provider.urllib.request.urlopen",
            side_effect=[http_error(503), http_error(503), http_error(503)],
        ) as urlopen:
            with self.assertRaises(ProviderError):
                provider.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(urlopen.call_count, 2)
        self.assertEqual(sleep.call_count, 1)

    def test_falls_back_to_json_protocol_when_tools_rejected(self):
        provider = self._provider()
        with mock.patch("coding_agent.provider.time.sleep") as sleep, mock.patch(
            "coding_agent.provider.urllib.request.urlopen",
            side_effect=[
                http_error(400),
                FakeResponse({"choices": [{"message": {"content": "ok"}}]}),
            ],
        ) as urlopen:
            result = provider.chat(
                [{"role": "user", "content": "hi"}],
                tools=[{"type": "function", "function": {"name": "list_files"}}],
                tool_choice="auto",
            )
        self.assertEqual(result, "ok")
        self.assertEqual(urlopen.call_count, 2)
        body = json.loads(urlopen.call_args_list[1].args[0].data)
        self.assertNotIn("tools", body)
        self.assertFalse(sleep.called)


if __name__ == "__main__":
    unittest.main()
