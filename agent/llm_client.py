"""通用 LLM 客户端接口与默认实现。"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Any

import httpx

from agent.llm_config import LLMConfig

logger = logging.getLogger(__name__)


class LLMClient(ABC):
	"""统一的大模型客户端接口。"""

	@abstractmethod
	def chat_completion(self, messages: list[dict[str, str]]) -> str:
		"""向模型发送消息并返回文本内容。"""

	@abstractmethod
	def chat_completion_stream(self, messages: list[dict[str, str]]):
		"""向模型发送消息，并通过生成器返回类型("reasoning" / "content") 和增量文本段。"""


class OpenAICompatibleRemoteLLMClient(LLMClient):
	"""基于 OpenAI 兼容接口的远程模型客户端。"""

	def __init__(self, config: LLMConfig | None = None) -> None:
		self.config = config or LLMConfig.from_env()

	def _build_payload(self, messages: list[dict[str, str]], stream: bool) -> dict[str, Any]:
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
		if self.config.thinking_budget is not None:
			payload["thinking_budget"] = self.config.thinking_budget
		return payload

	def _get_headers(self) -> dict[str, str]:
		if not self.config.api_key:
			raise RuntimeError("缺少 AGENT_MODEL_API_KEY，无法请求在线模型")
		return {
			"Authorization": f"Bearer {self.config.api_key}",
			"Content-Type": "application/json",
			"Accept": "application/json",
		}

	def chat_completion(self, messages: list[dict[str, str]]) -> str:
		"""向远程模型发送消息并返回文本内容。"""
		payload = self._build_payload(messages, stream=False)
		start_time = time.time()

		try:
			with httpx.Client(timeout=self.config.timeout_seconds) as client:
				resp = client.post(
					self.config.chat_completions_url,
					headers=self._get_headers(),
					json=payload
				)
				resp.raise_for_status()
				result = resp.json()
				latency = time.time() - start_time
				logger.debug(f"[LLM] 非流式请求耗时: {latency:.2f}s")
				return _extract_message_content(result)
		except httpx.HTTPStatusError as exc:
			raise RuntimeError(f"远程模型请求失败，HTTP {exc.response.status_code}: {exc.response.text}") from exc
		except Exception as exc:
			raise RuntimeError(f"远程模型网络请求失败: {exc}") from exc

	def chat_completion_stream(self, messages: list[dict[str, str]]):
		"""流式返回生成器。

		对 reasoning 内容的来源做互斥处理：
		- 若后端通过 ``reasoning_content`` 字段原生返回思考内容，
		  则直接使用，忽略 ``content`` 字段中的 ``<think>`` 标签。
		- 若后端不提供 ``reasoning_content``（如本地 Ollama），
		  则通过状态机从 ``content`` 字段中拦截 ``<think>...</think>`` 块。
		"""
		payload = self._build_payload(messages, stream=True)
		start_time = time.time()
		first_byte_time = None

		# <think> 标签状态机（局部变量，不污染外部对象）
		in_think_block = False
		# 是否已检测到后端原生提供 reasoning_content
		has_native_reasoning = False

		try:
			with httpx.Client(timeout=self.config.timeout_seconds) as client:
				with client.stream("POST", self.config.chat_completions_url, headers=self._get_headers(), json=payload) as resp:
					resp.raise_for_status()

					for line in resp.iter_lines():
						if not line.startswith("data:"):
							continue

						if first_byte_time is None:
							first_byte_time = time.time()
							logger.info(f"[LLM] 流式请求首字耗时 (TTFB): {first_byte_time - start_time:.2f}s")

						data_str = line.removeprefix("data:").strip()
						if data_str == "[DONE]":
							break

						try:
							chunk = json.loads(data_str)
							choices = chunk.get("choices")
							if not choices or not isinstance(choices, list) or len(choices) == 0:
								continue
							delta = choices[0].get("delta", {})

							# 优先级 1：后端原生 reasoning_content 字段（Qwen3.5 / 硅基流动）
							reasoning = delta.get("reasoning_content")
							if reasoning:
								has_native_reasoning = True
								yield {"type": "reasoning", "content": reasoning}

							# 优先级 2：content 字段
							content = delta.get("content")
							if content:
								if has_native_reasoning:
									# 后端已通过专属字段提供 reasoning，
									# content 字段直接作为正文输出，不再拦截 <think>
									yield {"type": "content", "content": content}
								else:
									# 本地模型走 <think> 标签状态机
									yield from self._parse_think_tags(content, in_think_block)
									# 更新状态机状态
									if "<think>" in content:
										in_think_block = True
									if "</think>" in content:
										in_think_block = False

						except json.JSONDecodeError:
							continue

					total_time = time.time() - start_time
					logger.info(f"[LLM] 流式生成完毕，总耗时: {total_time:.2f}s")

		except httpx.HTTPStatusError as exc:
			raise RuntimeError(f"远程流式请求失败，HTTP {exc.response.status_code}: {exc.response.text}") from exc
		except Exception as exc:
			raise RuntimeError(f"远程流式网络请求失败: {exc}") from exc

	@staticmethod
	def _parse_think_tags(content: str, in_think_block: bool):
		"""从 content 文本中拦截 <think>...</think> 标签，拆分为 reasoning / content 类型。"""

		if not in_think_block and "<think>" in content:
			# 进入思考块
			before, after = content.split("<think>", 1)
			if before:
				yield {"type": "content", "content": before}
			if "</think>" in after:
				# 同一个 chunk 内开闭了完整的 think 块
				think_part, rest = after.split("</think>", 1)
				if think_part:
					yield {"type": "reasoning", "content": think_part}
				if rest:
					yield {"type": "content", "content": rest}
			else:
				if after:
					yield {"type": "reasoning", "content": after}
		elif in_think_block:
			if "</think>" in content:
				think_part, rest = content.split("</think>", 1)
				if think_part:
					yield {"type": "reasoning", "content": think_part}
				if rest:
					yield {"type": "content", "content": rest}
			else:
				yield {"type": "reasoning", "content": content}
		else:
			yield {"type": "content", "content": content}


def create_default_llm_client(config: LLMConfig | None = None) -> LLMClient:
	"""根据配置创建默认模型客户端。"""

	resolved_config = config or LLMConfig.from_env()
	if resolved_config.backend == "openai_compatible_remote":
		return OpenAICompatibleRemoteLLMClient(resolved_config)
	if resolved_config.backend == "local":
		raise NotImplementedError("本地模型后端尚未实现，请后续新增 LocalLLMClient")
	raise ValueError(f"不支持的模型后端: {resolved_config.backend}")


def _extract_message_content(result: dict[str, Any]) -> str:
	"""从返回体中提取第一条消息文本。"""

	choices = result.get("choices")
	if not isinstance(choices, list) or not choices:
		raise RuntimeError(f"模型返回格式异常，缺少 choices: {result}")

	message = choices[0].get("message")
	if not isinstance(message, dict):
		raise RuntimeError(f"模型返回格式异常，缺少 message: {result}")

	content = message.get("content") or ""
	reasoning = message.get("reasoning_content") or ""

	# 思考模型（如 Qwen3.5）会将推导过程放在 reasoning_content，终态放在 content
	result_text = ""
	if reasoning.strip():
		result_text += f"Thought: {reasoning.strip()}\n"
	if content.strip():
		result_text += content.strip()

	if not result_text.strip():
		raise RuntimeError(f"模型返回格式异常，缺少 content: {result}")

	return result_text
