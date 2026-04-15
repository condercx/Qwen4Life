"""最小智能家居 agent 控制器。"""

from __future__ import annotations

from dataclasses import dataclass, field

from agent.llm_client import LLMClient, create_default_llm_client
from agent.llm_config import LLMConfig
from agent.parser import parse_agent_plan
from agent.prompts import build_system_prompt, build_user_prompt
from agent.schema import AgentPlan, AgentResult
from environment import AgentEnvironmentAdapter


@dataclass(slots=True)
class SimpleSmartHomeAgent:
	"""通过在线模型生成动作，并通过 adapter 调用环境。"""

	adapter: AgentEnvironmentAdapter = field(default_factory=AgentEnvironmentAdapter)
	client: LLMClient | None = None
	config: LLMConfig | None = None
	active_sessions: set[str] = field(default_factory=set)

	def __post_init__(self) -> None:
		if self.client is None:
			self.client = create_default_llm_client(self.config or LLMConfig.from_env())

	def create_session(self, session_id: str) -> dict:
		"""显式创建会话，并记录到 agent 内部。"""

		self.active_sessions.add(session_id)
		return self.adapter.create_session(session_id)

	def handle_user_input(self, session_id: str, user_input: str) -> AgentResult:
		"""处理单轮用户输入。"""

		self._ensure_session(session_id)
		messages = [
			{"role": "system", "content": build_system_prompt()},
			{"role": "user", "content": build_user_prompt(user_input)},
		]
		raw_model_output = self.client.chat_completion(messages)
		plan = parse_agent_plan(raw_model_output)

		if plan.plan_type == "state_query":
			state = self.adapter.fetch_state(session_id)
			events = self.adapter.fetch_events(session_id)
			reply = self._build_state_reply(plan, state, events)
			return AgentResult(
				session_id=session_id,
				user_input=user_input,
				plan=plan,
				reply=reply,
				state=state,
				events=events,
				environment_response=None,
				raw_model_output=raw_model_output,
			)

		environment_response = self.adapter.send_action(
			session_id=session_id,
			intent=plan.intent,
			action=plan.action or {},
		)
		state = self.adapter.fetch_state(session_id)
		events = self.adapter.fetch_events(session_id)
		reply = self._build_action_reply(plan, environment_response, events)
		return AgentResult(
			session_id=session_id,
			user_input=user_input,
			plan=plan,
			reply=reply,
			state=state,
			events=events,
			environment_response=environment_response,
			raw_model_output=raw_model_output,
		)

	def _ensure_session(self, session_id: str) -> None:
		"""确保 session 已初始化。"""

		if session_id not in self.active_sessions:
			self.create_session(session_id)

	def _build_state_reply(self, plan: AgentPlan, state: dict, events: list[dict]) -> str:
		"""生成状态查询类回复。"""

		prefix = plan.reply_hint or "当前环境状态如下。"
		descriptions = [self._describe_device(device) for device in state["devices"].values()]
		event_summary = ""
		if events:
			event_summary = " 刚发生的事件有：" + "，".join(self._describe_event_type(event["type"]) for event in events) + "。"
		return f"{prefix} {'；'.join(descriptions)}。{event_summary}".strip()

	def _build_action_reply(self, plan: AgentPlan, environment_response: dict, events: list[dict]) -> str:
		"""生成动作执行类回复。"""

		if environment_response["success"]:
			event_types = ", ".join(event["type"] for event in events) if events else "无"
			prefix = plan.reply_hint or "动作已执行。"
			return f"{prefix} 当前产生的事件有：{event_types}。"
		error = environment_response["error"]
		return f"动作执行失败，错误码 {error['code']}，原因：{error['message']}。"

	def _describe_device(self, device: dict) -> str:
		device_type = device["device_type"]
		if device_type == "light":
			return f"灯光{'开启' if device['is_on'] else '关闭'}，亮度 {device['brightness']}"
		if device_type == "ac":
			return (
				f"空调{'开启' if device['is_on'] else '关闭'}，模式 {device['mode']}，"
				f"目标温度 {device['target_temperature']} 度"
			)
		if device_type == "washing_machine":
			status = device["status"]
			if status == "running":
				return (
					f"洗衣机正在运行，程序 {device['program']}，"
					f"剩余 {self._format_duration(device['remaining_seconds'])}"
				)
			if status == "paused":
				return f"洗衣机已暂停，剩余 {self._format_duration(device['remaining_seconds'])}"
			if status == "completed":
				return "洗衣机已完成本轮洗衣"
			if status == "cancelled":
				return "洗衣机任务已取消"
			return "洗衣机当前空闲"
		return f"设备 {device['name']} 状态未知"

	def _describe_event_type(self, event_type: str) -> str:
		mapping = {
			"session_reset": "会话已重置",
			"light_state_changed": "灯光状态更新",
			"light_brightness_changed": "灯光亮度更新",
			"ac_state_changed": "空调状态更新",
			"ac_setting_changed": "空调参数更新",
			"washing_started": "洗衣已启动",
			"washing_paused": "洗衣已暂停",
			"washing_resumed": "洗衣已继续",
			"washing_completed": "洗衣已完成",
			"washing_cancelled": "洗衣已取消",
		}
		return mapping.get(event_type, event_type)

	def _format_duration(self, seconds: int) -> str:
		if seconds < 60:
			return f"{seconds} 秒"
		minutes, remain = divmod(seconds, 60)
		if remain == 0:
			return f"{minutes} 分钟"
		return f"{minutes} 分 {remain} 秒"
