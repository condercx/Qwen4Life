"""最小智能家居 agent 控制器（ReAct 模式）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent.llm_client import LLMClient, create_default_llm_client
from agent.llm_config import LLMConfig
from agent.parser import parse_react_output
from agent.prompts import build_system_prompt, build_user_prompt
from agent.schema import AgentResult, ReactStep
from agent.tools import ToolRegistry


# ReAct 循环最大步数，防止无限循环
_MAX_REACT_STEPS = 8


@dataclass(slots=True)
class SimpleSmartHomeAgent:
	"""通过 ReAct 循环驱动模型与环境交互。

	模型在循环中思考（Thought）、调用工具（Action）、
	接收结果（Observation），最终生成自然语言回复（Answer）。
	"""

	tools: ToolRegistry = field(default_factory=ToolRegistry)
	client: LLMClient | None = None
	config: LLMConfig | None = None
	# 会话级对话历史，支持多轮上下文
	_session_histories: dict[str, list[dict[str, str]]] = field(default_factory=dict)

	def __post_init__(self) -> None:
		if self.client is None:
			self.client = create_default_llm_client(self.config or LLMConfig.from_env())

	def create_session(self, session_id: str) -> None:
		"""初始化会话。"""

		self.tools.adapter.create_session(session_id)
		self._session_histories[session_id] = []

	def handle_user_input(self, session_id: str, user_input: str) -> AgentResult:
		"""处理用户输入，运行 ReAct 循环直到产出最终回复。"""

		self._ensure_session(session_id)

		# 构建消息序列
		system_msg = {"role": "system", "content": build_system_prompt(self.tools.get_tools_prompt())}
		history = self._session_histories[session_id]
		user_msg = {"role": "user", "content": build_user_prompt(user_input)}
		history.append(user_msg)

		steps: list[ReactStep] = []
		all_messages: list[dict[str, str]] = [system_msg] + list(history)

		for step_idx in range(_MAX_REACT_STEPS):
			# 调用模型
			raw_output = self.client.chat_completion(all_messages)
			parsed = parse_react_output(raw_output)
			steps.append(parsed)

			if parsed.type == "answer":
				# 模型给出最终回复
				history.append({"role": "assistant", "content": raw_output})
				self._trim_history(session_id)
				return AgentResult(
					session_id=session_id,
					user_input=user_input,
					reply=parsed.content,
					steps=steps,
					raw_messages=list(all_messages),
				)

			if parsed.type == "action" and parsed.tool_name:
				# 执行工具
				observation = self.tools.execute(
					session_id=session_id,
					tool_name=parsed.tool_name,
					args=parsed.tool_args or {},
				)
				obs_step = ReactStep(type="observation", content=observation)
				steps.append(obs_step)

				# 将 assistant 输出和 observation 加入对话
				all_messages.append({"role": "assistant", "content": raw_output})
				all_messages.append({"role": "user", "content": f"Observation: {observation}"})
				continue

			if parsed.type == "thought":
				# 纯思考没有动作，追加并让模型继续
				all_messages.append({"role": "assistant", "content": raw_output})
				all_messages.append({"role": "user", "content": "请继续，给出 Action 或 Answer。"})
				continue

		# 超过最大步数，用已有步骤的最后内容作为回复
		fallback_reply = "抱歉，我处理这个请求时遇到了一些困难，请你再试一次或换个说法。"
		history.append({"role": "assistant", "content": fallback_reply})
		return AgentResult(
			session_id=session_id,
			user_input=user_input,
			reply=fallback_reply,
			steps=steps,
			raw_messages=list(all_messages),
		)

	def handle_user_input_stream(self, session_id: str, user_input: str):
		"""流式处理用户输入，支持中途停顿执行工具。"""
		self._ensure_session(session_id)

		system_msg = {"role": "system", "content": build_system_prompt(self.tools.get_tools_prompt())}
		history = self._session_histories[session_id]
		user_msg = {"role": "user", "content": build_user_prompt(user_input)}
		history.append(user_msg)

		all_messages: list[dict[str, str]] = [system_msg] + list(history)

		for step_idx in range(_MAX_REACT_STEPS):
			raw_output = ""
			try:
				for chunk in self.client.chat_completion_stream(all_messages):
					if chunk["type"] == "reasoning":
						yield {"type": "reasoning", "content": chunk["content"]}
					elif chunk["type"] == "content":
						yield {"type": "content", "content": chunk["content"]}
						# 只拼接正文内容，reasoning 不参与后续 Action/Answer 解析
						raw_output += chunk["content"]
			except Exception as exc:
				yield {"type": "error", "content": f"\n[请求异常]: {exc}\n"}
				break

			# 收到完整的一轮模型输出后解析动作
			parsed = parse_react_output(raw_output)

			if parsed.type == "answer":
				history.append({"role": "assistant", "content": raw_output})
				self._trim_history(session_id)
				yield {"type": "final_reply", "content": parsed.content}
				return

			if parsed.type == "action" and parsed.tool_name:
				yield {"type": "action_start", "content": f"\n[调用工具: {parsed.tool_name}({parsed.tool_args})]\n"}
				
				observation = self.tools.execute(
					session_id=session_id,
					tool_name=parsed.tool_name,
					args=parsed.tool_args or {},
				)
				
				yield {"type": "observation", "content": f"[工具返回: {observation}]\n"}
				
				all_messages.append({"role": "assistant", "content": raw_output})
				all_messages.append({"role": "user", "content": f"Observation: {observation}"})
				continue

			if parsed.type == "thought":
				all_messages.append({"role": "assistant", "content": raw_output})
				all_messages.append({"role": "user", "content": "请继续，给出 Action 或 Answer。"})
				continue

		fallback_reply = "抱歉，我处理这个请求时遇到了一些困难，请你再试一次或换个说法。"
		history.append({"role": "assistant", "content": fallback_reply})
		self._trim_history(session_id)
		yield {"type": "error", "content": fallback_reply}

	def _ensure_session(self, session_id: str) -> None:
		"""确保 session 已初始化。"""

		if session_id not in self._session_histories:
			self.create_session(session_id)

	def _trim_history(self, session_id: str, max_turns: int = 20) -> None:
		"""修剪对话历史，防止 token 无限增长。

		保留最近 max_turns 条消息。
		"""

		history = self._session_histories.get(session_id)
		if history and len(history) > max_turns:
			self._session_histories[session_id] = history[-max_turns:]
