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
			options=plan.options,
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

		light = state["devices"]["living_room_light_1"]
		ac = state["devices"]["living_room_ac_1"]
		robot = state["devices"]["robot_vacuum_1"]
		prefix = plan.reply_hint or "当前环境状态如下。"
		return (
			f"{prefix} "
			f"灯光{'开启' if light['is_on'] else '关闭'}，亮度 {light['brightness']}；"
			f"空调{'开启' if ac['is_on'] else '关闭'}，模式 {ac['mode']}，目标温度 {ac['target_temperature']}；"
			f"机器人状态 {robot['status']}，位置 ({robot['position']['x']:.2f}, {robot['position']['y']:.2f})。"
		)

	def _build_action_reply(self, plan: AgentPlan, environment_response: dict, events: list[dict]) -> str:
		"""生成动作执行类回复。"""

		if environment_response["success"]:
			event_types = ", ".join(event["type"] for event in events) if events else "无"
			prefix = plan.reply_hint or "动作已执行。"
			return f"{prefix} 当前产生的事件有：{event_types}。"
		error = environment_response["error"]
		return f"动作执行失败，错误码 {error['code']}，原因：{error['message']}。"
