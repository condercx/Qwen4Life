"""Agent 内部使用的数据结构定义。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ReactStep:
    """表示一次 ReAct 推理步骤。"""

    type: str
    content: str
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None


@dataclass(slots=True)
class AgentResult:
    """表示一次 Agent 执行结果。"""

    session_id: str
    user_input: str
    reply: str
    steps: list[ReactStep] = field(default_factory=list)
    raw_messages: list[dict[str, Any]] = field(default_factory=list)
