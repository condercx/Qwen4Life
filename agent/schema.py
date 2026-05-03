"""Agent 内部使用的数据结构定义。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ReactStep:
    """表示一次 ReAct 推理步骤。"""

    type: str
    content: str
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    raw_action_text: str | None = None
