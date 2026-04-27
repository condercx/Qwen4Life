"""长期记忆配置。"""

from __future__ import annotations

import os
from dataclasses import dataclass

from agent.llm_config import _load_default_env_files


@dataclass(slots=True)
class MemoryConfig:
    """描述 Agent 长期记忆所需的本地检索配置。"""

    enabled: bool = False
    embed_backend: str = "ollama"
    ollama_embed_url: str = "http://127.0.0.1:11434/api/embed"
    embed_model: str = "bge-m3"
    chroma_path: str = ".agent_memory/chroma"
    collection: str = "agent_memory"
    top_k: int = 5
    min_score: float = 0.0
    timeout_seconds: float = 30.0

    @classmethod
    def from_env(cls) -> "MemoryConfig":
        """从环境变量和默认 .env 文件读取长期记忆配置。"""

        _load_default_env_files()
        return cls(
            enabled=_parse_bool(os.getenv("AGENT_MEMORY_ENABLED", "false")),
            embed_backend=os.getenv("AGENT_MEMORY_EMBED_BACKEND", "ollama"),
            ollama_embed_url=os.getenv(
                "AGENT_MEMORY_OLLAMA_EMBED_URL",
                "http://127.0.0.1:11434/api/embed",
            ),
            embed_model=os.getenv("AGENT_MEMORY_EMBED_MODEL", "bge-m3"),
            chroma_path=os.getenv("AGENT_MEMORY_CHROMA_PATH", ".agent_memory/chroma"),
            collection=os.getenv("AGENT_MEMORY_COLLECTION", "agent_memory"),
            top_k=max(1, int(os.getenv("AGENT_MEMORY_TOP_K", "5"))),
            min_score=float(os.getenv("AGENT_MEMORY_MIN_SCORE", "0.0")),
            timeout_seconds=float(os.getenv("AGENT_MEMORY_TIMEOUT_SECONDS", "30.0")),
        )


def _parse_bool(value: str) -> bool:
    """解析布尔环境变量。"""

    return value.strip().lower() in {"1", "true", "yes", "on"}
