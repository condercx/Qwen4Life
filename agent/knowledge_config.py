"""儿童教育知识库配置。"""

from __future__ import annotations

from dataclasses import dataclass
import os

from agent.llm_config import _load_default_env_files
from agent.memory_config import _parse_bool


@dataclass(slots=True)
class KnowledgeConfig:
    """描述本地 RAG 知识库所需的配置。"""

    enabled: bool = False
    embed_backend: str = "ollama"
    ollama_embed_url: str = "http://127.0.0.1:11434/api/embed"
    embed_model: str = "bge-m3"
    chroma_path: str = ".agent_kb/chroma"
    collection: str = "children_education"
    source_path: str = "data/knowledge/grimms_fairy_tales.txt"
    source_url: str = "https://www.gutenberg.org/ebooks/2591"
    top_k: int = 5
    min_score: float = 0.0
    chunk_chars: int = 1200
    chunk_overlap: int = 160
    embed_text_chars: int = 120
    timeout_seconds: float = 30.0

    @classmethod
    def from_env(cls) -> "KnowledgeConfig":
        """从环境变量和默认 .env 文件读取知识库配置。"""

        _load_default_env_files()
        return cls(
            enabled=_parse_bool(os.getenv("AGENT_KB_ENABLED", "false")),
            embed_backend=os.getenv("AGENT_KB_EMBED_BACKEND", "ollama"),
            ollama_embed_url=os.getenv(
                "AGENT_KB_OLLAMA_EMBED_URL",
                "http://127.0.0.1:11434/api/embed",
            ),
            embed_model=os.getenv("AGENT_KB_EMBED_MODEL", "bge-m3"),
            chroma_path=os.getenv("AGENT_KB_CHROMA_PATH", ".agent_kb/chroma"),
            collection=os.getenv("AGENT_KB_COLLECTION", "children_education"),
            source_path=os.getenv("AGENT_KB_SOURCE_PATH", "data/knowledge/grimms_fairy_tales.txt"),
            source_url=os.getenv("AGENT_KB_SOURCE_URL", "https://www.gutenberg.org/ebooks/2591"),
            top_k=max(1, int(os.getenv("AGENT_KB_TOP_K", "5"))),
            min_score=float(os.getenv("AGENT_KB_MIN_SCORE", "0.0")),
            chunk_chars=max(300, int(os.getenv("AGENT_KB_CHUNK_CHARS", "1200"))),
            chunk_overlap=max(0, int(os.getenv("AGENT_KB_CHUNK_OVERLAP", "160"))),
            embed_text_chars=max(40, int(os.getenv("AGENT_KB_EMBED_TEXT_CHARS", "120"))),
            timeout_seconds=float(os.getenv("AGENT_KB_TIMEOUT_SECONDS", "30.0")),
        )
