"""儿童教育知识库门面。"""

from __future__ import annotations

from dataclasses import dataclass

from agent.knowledge_config import KnowledgeConfig
from agent.knowledge_store import ChromaKnowledgeStore, KnowledgeStore


@dataclass(slots=True)
class AgentKnowledgeBase:
    """封装知识库检索，避免工具层依赖具体存储。"""

    store: KnowledgeStore
    config: KnowledgeConfig

    def search(self, query: str) -> str:
        """返回适合作为 tool Observation 的知识库检索结果。"""

        normalized_query = query.strip()
        if not normalized_query:
            return "知识库查询失败：query 不能为空。"

        chunks = self.store.search(
            normalized_query,
            top_k=self.config.top_k,
            min_score=self.config.min_score,
        )
        if not chunks:
            return "知识库没有检索到相关内容。"

        lines = [
            "知识库检索结果如下。请把这些片段作为儿童教育陪伴的参考材料，最终回答要自然、适合孩子，不要暴露检索细节："
        ]
        for index, chunk in enumerate(chunks, start=1):
            text = _compact_text(chunk.text, max_chars=360)
            lines.append(
                f"{index}. 故事：{chunk.title}\n"
                f"来源：{chunk.source}\n"
                f"片段：{text}"
            )
        return "\n\n".join(lines)


def create_default_knowledge_base(config: KnowledgeConfig | None = None) -> AgentKnowledgeBase:
    """创建默认 ChromaDB 知识库。"""

    resolved_config = config or KnowledgeConfig.from_env()
    return AgentKnowledgeBase(
        store=ChromaKnowledgeStore(resolved_config),
        config=resolved_config,
    )


def _compact_text(text: str, *, max_chars: int) -> str:
    compact = " ".join(line.strip() for line in text.strip().splitlines() if line.strip())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 1].rstrip() + "…"
