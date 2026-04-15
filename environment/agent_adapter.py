"""面向 agent 的环境适配层。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from environment.smart_home_env import SmartHomeEnv


@dataclass(slots=True)
class AgentEnvironmentAdapter:
	"""对 agent 暴露稳定接口，隔离环境核心与传输层细节。"""

	environment: SmartHomeEnv = field(default_factory=SmartHomeEnv)
	request_counter: int = 0

	def create_session(self, session_id: str) -> dict[str, Any]:
		"""显式初始化一个会话，建议每个 agent 会话开始时调用。"""

		return self.environment.reset(session_id)

	def send_action(
		self,
		session_id: str,
		action: dict[str, Any],
		intent: str | None = None,
		request_id: str | None = None,
	) -> dict[str, Any]:
		"""使用语义动作直接驱动环境。"""

		request = self.build_request(
			session_id=session_id,
			action=action,
			intent=intent,
			request_id=request_id,
		)
		return self.environment.step(request)

	def send_request(self, request: dict[str, Any]) -> dict[str, Any]:
		"""直接发送完整请求体，适合规划器已产出标准协议时使用。"""

		return self.environment.step(request)

	def send_request_json(self, request_json: str) -> dict[str, Any]:
		"""接受 JSON 字符串请求，方便未来对接 HTTP/MQTT/串口协议层。"""

		request = json.loads(request_json)
		return self.send_request(request)

	def fetch_state(self, session_id: str) -> dict[str, Any]:
		"""读取当前会话状态。"""

		return self.environment.get_state(session_id)

	def fetch_events(self, session_id: str) -> list[dict[str, Any]]:
		"""读取当前会话未消费事件。"""

		return self.environment.get_events(session_id)

	def build_request(
		self,
		session_id: str,
		action: dict[str, Any],
		intent: str | None = None,
		request_id: str | None = None,
	) -> dict[str, Any]:
		"""构造标准 step 请求体，作为本地调用与协议传输的统一格式。"""

		return {
			"request_id": request_id or self._next_request_id(session_id),
			"session_id": session_id,
			"intent": intent,
			"action": action,
		}

	def _next_request_id(self, session_id: str) -> str:
		"""生成递增请求编号，便于调试和日志追踪。"""

		self.request_counter += 1
		return f"{session_id}-req-{self.request_counter}"
