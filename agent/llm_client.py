"""通用 LLM 客户端接口与默认实现。"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any
from urllib import error, request

from agent.llm_config import LLMConfig


class LLMClient(ABC):
	"""统一的大模型客户端接口。"""

	@abstractmethod
	def chat_completion(self, messages: list[dict[str, str]]) -> str:
		"""向模型发送消息并返回文本内容。"""


class OpenAICompatibleRemoteLLMClient(LLMClient):
	"""基于 OpenAI 兼容接口的远程模型客户端。"""

	def __init__(self, config: LLMConfig | None = None) -> None:
		self.config = config or LLMConfig.from_env()

	def chat_completion(self, messages: list[dict[str, str]]) -> str:
		"""向远程模型发送消息并返回文本内容。"""

		if not self.config.api_key:
			raise RuntimeError("缺少 AGENT_MODEL_API_KEY，无法请求在线模型")

		payload: dict[str, Any] = {
			"model": self.config.model,
			"messages": messages,
			"stream": False,
			"n": self.config.n,
			"temperature": self.config.temperature,
			"top_p": self.config.top_p,
			"max_tokens": self.config.max_tokens,
		}
		if self.config.force_json_output:
			# 参考硅基流动 Chat Completions 文档，使用 response_format 约束模型输出 JSON。
			payload["response_format"] = {"type": "json_object"}
		if self.config.enable_thinking is not None:
			payload["enable_thinking"] = self.config.enable_thinking
		if self.config.thinking_budget is not None:
			payload["thinking_budget"] = self.config.thinking_budget

		body = json.dumps(payload).encode("utf-8")
		http_request = request.Request(
			url=self.config.chat_completions_url,
			data=body,
			headers={
				"Authorization": f"Bearer {self.config.api_key}",
				"Content-Type": "application/json",
				"Accept": "application/json",
			},
			method="POST",
		)

		try:
			with request.urlopen(http_request, timeout=self.config.timeout_seconds) as response:
				result = json.loads(response.read().decode("utf-8"))
				trace_id = response.headers.get("x-siliconcloud-trace-id")
		except error.HTTPError as exc:
			detail = exc.read().decode("utf-8", errors="replace")
			raise RuntimeError(f"远程模型请求失败，HTTP {exc.code}: {detail}") from exc
		except error.URLError as exc:
			raise RuntimeError(f"远程模型网络请求失败: {exc}") from exc

		if trace_id:
			result["_trace_id"] = trace_id

		return _extract_message_content(result)


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

	content = message.get("content")
	if not isinstance(content, str) or not content.strip():
		raise RuntimeError(f"模型返回格式异常，缺少 content: {result}")

	return content
