"""Agent 长期记忆门面。"""

from __future__ import annotations

from dataclasses import dataclass

from agent.memory_config import MemoryConfig
from agent.memory_decision import VALID_MEMORY_TYPES
from agent.memory_store import ChromaMemoryStore, MemoryItem, MemoryStore

_MEMORY_TYPE_LABELS = {
    "preference": "用户偏好",
    "alias": "设备别名",
    "habit": "用户习惯",
    "home_rule": "家庭规则",
    "agreement": "历史约定",
}


@dataclass(slots=True)
class AgentMemory:
    """封装检索和保存，避免 controller 依赖具体存储。"""

    store: MemoryStore
    config: MemoryConfig

    def search_context(self, user_id: str, session_id: str, query: str) -> str:
        """返回适合注入 system prompt 的中文长期记忆文本。"""

        del session_id
        memories = self.store.search_memory(
            user_id=user_id,
            query=query,
            top_k=self.config.top_k,
            min_score=self.config.min_score,
        )
        if not memories:
            return ""

        lines = [
            "以下是与本轮请求可能相关的历史记忆，只能作为用户偏好、习惯、设备别名和历史约定参考："
        ]
        for index, memory in enumerate(memories, start=1):
            text = _clean_text(memory.text)
            if text:
                lines.append(f"{index}. {text}")
        return "\n".join(lines).strip()

    def save_memory(self, user_id: str, session_id: str, memory_text: str, memory_type: str) -> None:
        """保存 agent 已判断有长期价值的记忆文本。"""

        clean_memory_text = _clean_text(memory_text)
        normalized_memory_type = memory_type.strip()
        if not clean_memory_text or normalized_memory_type not in VALID_MEMORY_TYPES:
            return

        self.store.add_memory(
            user_id=user_id,
            session_id=session_id,
            text=clean_memory_text,
            metadata={"memory_type": normalized_memory_type},
        )

    def list_memories(self, user_id: str) -> str:
        """返回用户长期记忆列表，供工具 Observation 使用。"""

        memories = self.store.get_all_memories(user_id)
        if not memories:
            return "当前没有长期记忆。"

        lines = ["当前长期记忆："]
        for index, memory in enumerate(memories, start=1):
            lines.append(_format_memory_item(index, memory))
        return "\n".join(lines)

    def delete_memory(self, user_id: str, memory_id: str) -> str:
        """删除一条用户长期记忆。"""

        normalized_memory_id = memory_id.strip()
        if not normalized_memory_id:
            return "删除失败：memory_id 不能为空。"
        deleted = self.store.delete_memory(user_id, normalized_memory_id)
        if deleted:
            return f"已删除长期记忆：{normalized_memory_id}"
        return f"未找到可删除的长期记忆：{normalized_memory_id}"

    def clear_user_memory(self, user_id: str) -> str:
        """清空用户长期记忆。"""

        self.store.clear_user_memory(user_id)
        return "已清空当前用户的长期记忆。"


def create_default_agent_memory(config: MemoryConfig | None = None) -> AgentMemory:
    """创建默认 ChromaDB 长期记忆实现。"""

    resolved_config = config or MemoryConfig.from_env()
    if resolved_config.embed_backend != "ollama":
        raise ValueError(f"不支持的 memory embedding 后端：{resolved_config.embed_backend}")
    return AgentMemory(store=ChromaMemoryStore(resolved_config), config=resolved_config)


def _clean_text(text: str) -> str:
    """压缩空白，避免保存 raw messages 或多余格式。"""

    return "\n".join(line.strip() for line in text.strip().splitlines() if line.strip())


def _format_memory_item(index: int, memory: MemoryItem) -> str:
    """格式化单条长期记忆，保留可删除 ID。"""

    metadata = memory.metadata
    memory_type = str(metadata.get("memory_type", "unknown"))
    label = _MEMORY_TYPE_LABELS.get(memory_type, memory_type)
    created_at = str(metadata.get("created_at", ""))
    header = f"{index}. id={memory.memory_id}，类型={label}"
    if created_at:
        header += f"，创建时间={created_at}"
    text = _clean_text(memory.text)
    return f"{header}\n{text}"
