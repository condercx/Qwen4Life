"""Agent 内部使用的结构化数据定义（ReAct 模式）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ReactStep:
	"""ReAct 循环中的单个步骤。"""

	type: str  # "thought", "action", "observation", "answer"
	content: str  # 原始文本内容
	tool_name: str | None = None  # 仅 action 步骤
	tool_args: dict[str, Any] | None = None  # 仅 action 步骤


@dataclass(slots=True)
class AgentResult:
	"""对外返回的 agent 执行结果。"""

	session_id: str
	user_input: str
	reply: str
	steps: list[ReactStep] = field(default_factory=list)
	raw_messages: list[dict[str, Any]] = field(default_factory=list)
