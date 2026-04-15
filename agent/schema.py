"""Agent 内部使用的结构化数据定义。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class AgentPlan:
	"""模型产出的执行计划。"""

	plan_type: str
	intent: str
	reply_hint: str | None = None
	action: dict[str, Any] | None = None


@dataclass(slots=True)
class AgentResult:
	"""对外返回的 agent 执行结果。"""

	session_id: str
	user_input: str
	plan: AgentPlan
	reply: str
	state: dict[str, Any]
	events: list[dict[str, Any]]
	environment_response: dict[str, Any] | None = None
	raw_model_output: str | None = None
