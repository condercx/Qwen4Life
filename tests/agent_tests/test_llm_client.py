"""Tests for LLM request payload construction."""

from __future__ import annotations

import unittest

from agent.llm_client import OpenAICompatibleRemoteLLMClient
from agent.llm_config import LLMConfig


class OpenAICompatibleRemoteLLMClientTest(unittest.TestCase):
    """Verify provider-specific request options are preserved."""

    def test_disabling_thinking_sends_ollama_reasoning_flags(self) -> None:
        config = LLMConfig(enable_thinking=False)
        client = OpenAICompatibleRemoteLLMClient(config)

        payload = client._build_payload([{"role": "user", "content": "hello"}])

        self.assertIs(payload["enable_thinking"], False)
        self.assertIs(payload["think"], False)
        self.assertEqual(payload["reasoning_effort"], "none")
        self.assertEqual(payload["reasoning"], {"effort": "none"})

    def test_unspecified_thinking_does_not_send_reasoning_flags(self) -> None:
        config = LLMConfig(enable_thinking=None)
        client = OpenAICompatibleRemoteLLMClient(config)

        payload = client._build_payload([{"role": "user", "content": "hello"}])

        self.assertNotIn("enable_thinking", payload)
        self.assertNotIn("think", payload)
        self.assertNotIn("reasoning_effort", payload)
        self.assertNotIn("reasoning", payload)


if __name__ == "__main__":
    unittest.main()
