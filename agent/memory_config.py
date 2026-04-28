"""Markdown 长期记忆配置。"""

from __future__ import annotations

import os
from dataclasses import dataclass

from agent.llm_config import _load_default_env_files


@dataclass(slots=True)
class MemoryConfig:
    """描述 Agent markdown 长期记忆所需的本地配置。"""

    enabled: bool = False
    memory_dir: str = ".agent_memory/profile"
    max_context_items: int = 20

    @classmethod
    def from_env(cls) -> "MemoryConfig":
        """从环境变量和默认 .env 文件读取 markdown 长期记忆配置。"""

        _load_default_env_files()
        return cls(
            enabled=_parse_bool(os.getenv("AGENT_MEMORY_ENABLED", "false")),
            memory_dir=os.getenv("AGENT_MEMORY_DIR", ".agent_memory/profile"),
            max_context_items=max(1, int(os.getenv("AGENT_MEMORY_MAX_CONTEXT_ITEMS", "20"))),
        )


def _parse_bool(value: str) -> bool:
    """解析布尔环境变量。"""

    return value.strip().lower() in {"1", "true", "yes", "on"}
