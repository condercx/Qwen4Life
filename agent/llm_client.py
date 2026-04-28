"""LLM 客户端抽象与默认 HTTP 实现。"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any

import httpx

from agent.llm_config import LLMConfig

logger = logging.getLogger(__name__)


class LLMClient(ABC):
    """统一的大模型客户端接口。"""

    @abstractmethod
    def chat_completion(self, messages: list[dict[str, str]]) -> str:
        """发送非流式请求并返回最终文本。"""

    @abstractmethod
    def chat_completion_stream(self, messages: list[dict[str, str]]) -> Iterator[dict[str, str]]:
        """发送流式请求并持续产出增量结果。"""


class OpenAICompatibleRemoteLLMClient(LLMClient):
    """基于 OpenAI 兼容接口的远程客户端。"""

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig.from_env()

    def chat_completion(self, messages: list[dict[str, str]]) -> str:
        """发送非流式请求。"""

        payload = self._build_payload(messages, stream=False)
        return self._post_chat_completion_payload(payload)

    def chat_completion_with_overrides(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        enable_thinking: bool | None = None,
        thinking_budget: int | None = None,
        force_json_output: bool | None = None,
    ) -> str:
        """发送带临时参数覆盖的非流式请求。"""

        payload = self._build_payload(messages, stream=False)
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature
        if enable_thinking is not None:
            payload["enable_thinking"] = enable_thinking
            payload["think"] = enable_thinking
        if thinking_budget is not None:
            payload["thinking_budget"] = thinking_budget
        if force_json_output:
            payload["response_format"] = {"type": "json_object"}
        elif force_json_output is False:
            payload.pop("response_format", None)
        return self._post_chat_completion_payload(payload, allow_reasoning_fallback=True)

    def _post_chat_completion_payload(
        self,
        payload: dict[str, Any],
        *,
        allow_reasoning_fallback: bool = False,
    ) -> str:
        """提交非流式请求 payload 并提取文本。"""

        start_time = time.monotonic()
        try:
            with httpx.Client(timeout=self.config.timeout_seconds) as client:
                response = client.post(
                    self.config.chat_completions_url,
                    headers=self._get_headers(),
                    json=payload,
                )
                response.raise_for_status()
                result = response.json()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"远程模型请求失败，HTTP {exc.response.status_code}: {exc.response.text}"
            ) from exc
        except httpx.HTTPError as exc:
            raise RuntimeError(f"远程模型网络请求失败：{exc}") from exc

        logger.debug("非流式请求耗时 %.2fs", time.monotonic() - start_time)
        return _extract_message_content(result, allow_reasoning_fallback=allow_reasoning_fallback)

    def chat_completion_stream(self, messages: list[dict[str, str]]) -> Iterator[dict[str, str]]:
        """发送流式请求。"""

        payload = self._build_payload(messages, stream=True)
        start_time = time.monotonic()
        first_byte_time: float | None = None
        in_think_block = False
        has_native_reasoning = False

        try:
            with httpx.Client(timeout=self.config.timeout_seconds) as client:
                with client.stream(
                    "POST",
                    self.config.chat_completions_url,
                    headers=self._get_headers(),
                    json=payload,
                ) as response:
                    response.raise_for_status()
                    for raw_line in response.iter_lines():
                        if not raw_line.startswith("data:"):
                            continue

                        if first_byte_time is None:
                            first_byte_time = time.monotonic()
                            logger.info("流式请求首字耗时 %.2fs", first_byte_time - start_time)

                        data_text = raw_line.removeprefix("data:").strip()
                        if data_text == "[DONE]":
                            break

                        chunk = _safe_load_json(data_text)
                        if chunk is None:
                            continue

                        delta = _extract_delta(chunk)
                        if delta is None:
                            continue

                        reasoning = delta.get("reasoning_content")
                        if isinstance(reasoning, str) and reasoning:
                            has_native_reasoning = True
                            yield {"type": "reasoning", "content": reasoning}

                        content = delta.get("content")
                        if not isinstance(content, str) or not content:
                            continue

                        if has_native_reasoning:
                            yield {"type": "content", "content": content}
                            continue

                        for parsed_chunk in self._parse_think_tags(content, in_think_block):
                            yield parsed_chunk
                        if "<think>" in content:
                            in_think_block = True
                        if "</think>" in content:
                            in_think_block = False
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"远程流式请求失败，HTTP {exc.response.status_code}: {exc.response.text}"
            ) from exc
        except httpx.HTTPError as exc:
            raise RuntimeError(f"远程流式网络请求失败：{exc}") from exc
        finally:
            logger.info("流式请求总耗时 %.2fs", time.monotonic() - start_time)

    def _build_payload(self, messages: list[dict[str, str]], stream: bool) -> dict[str, Any]:
        """构造统一请求体。"""

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "stream": stream,
            "n": self.config.n,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
            "presence_penalty": self.config.presence_penalty,
            "repetition_penalty": self.config.repetition_penalty,
            "max_tokens": self.config.max_tokens,
        }
        if self.config.min_p > 0.0:
            payload["min_p"] = self.config.min_p
        if self.config.force_json_output and not stream:
            payload["response_format"] = {"type": "json_object"}
        if self.config.enable_thinking is not None:
            payload["enable_thinking"] = self.config.enable_thinking
            payload["think"] = self.config.enable_thinking
        if self.config.thinking_budget is not None:
            payload["thinking_budget"] = self.config.thinking_budget
        return payload

    def _get_headers(self) -> dict[str, str]:
        """生成请求头。"""

        if not self.config.api_key:
            raise RuntimeError("缺少 `AGENT_MODEL_API_KEY`，无法请求在线模型。")
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    @staticmethod
    def _parse_think_tags(content: str, in_think_block: bool) -> Iterator[dict[str, str]]:
        """解析 `<think>` 标签，将文本拆分为 reasoning 与 content。"""

        if not in_think_block and "<think>" in content:
            before, after = content.split("<think>", 1)
            if before:
                yield {"type": "content", "content": before}
            if "</think>" in after:
                think_part, rest = after.split("</think>", 1)
                if think_part:
                    yield {"type": "reasoning", "content": think_part}
                if rest:
                    yield {"type": "content", "content": rest}
                return
            if after:
                yield {"type": "reasoning", "content": after}
            return

        if in_think_block:
            if "</think>" in content:
                think_part, rest = content.split("</think>", 1)
                if think_part:
                    yield {"type": "reasoning", "content": think_part}
                if rest:
                    yield {"type": "content", "content": rest}
                return
            yield {"type": "reasoning", "content": content}
            return

        yield {"type": "content", "content": content}


def create_default_llm_client(config: LLMConfig | None = None) -> LLMClient:
    """根据配置创建默认客户端。"""

    resolved_config = config or LLMConfig.from_env()
    if resolved_config.backend == "openai_compatible_remote":
        return OpenAICompatibleRemoteLLMClient(resolved_config)
    if resolved_config.backend == "local":
        raise NotImplementedError("本地模型后端尚未实现，请补充 LocalLLMClient。")
    raise ValueError(f"不支持的模型后端：{resolved_config.backend}")


def _safe_load_json(text: str) -> dict[str, Any] | None:
    """安全解析 JSON 字符串。"""

    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        return None
    return result if isinstance(result, dict) else None


def _extract_delta(chunk: dict[str, Any]) -> dict[str, Any] | None:
    """从流式 chunk 中提取 delta 字段。"""

    choices = chunk.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return None
    delta = first_choice.get("delta")
    return delta if isinstance(delta, dict) else None


def _extract_message_content(result: dict[str, Any], *, allow_reasoning_fallback: bool = False) -> str:
    """从非流式响应中提取文本内容。"""

    choices = result.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"模型返回格式异常，缺少 `choices`：{result}")

    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise RuntimeError(f"模型返回格式异常，缺少 `message`：{result}")

    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or ""
    fallback_reasoning = message.get("reasoning") or ""
    result_text = ""
    if isinstance(reasoning, str) and reasoning.strip():
        result_text += f"Thought: {reasoning.strip()}\n"
    if isinstance(content, str) and content.strip():
        result_text += content.strip()

    if not result_text.strip():
        if allow_reasoning_fallback and isinstance(fallback_reasoning, str) and fallback_reasoning.strip():
            return fallback_reasoning.strip()
        raise RuntimeError(f"模型返回格式异常，缺少有效内容：{result}")
    return result_text
