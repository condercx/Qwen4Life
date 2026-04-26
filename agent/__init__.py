"""智能家居 Agent 模块导出。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent.controller import SimpleSmartHomeAgent
    from agent.llm_config import LLMConfig

__all__ = ["SimpleSmartHomeAgent", "LLMConfig"]


def __getattr__(name: str) -> Any:
    """仅在调用方实际访问时再加载较重的 Agent 导出对象。"""

    if name == "SimpleSmartHomeAgent":
        from agent.controller import SimpleSmartHomeAgent

        return SimpleSmartHomeAgent
    if name == "LLMConfig":
        from agent.llm_config import LLMConfig

        return LLMConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
