"""Markdown 长期记忆。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import re
import uuid

from agent.memory_config import MemoryConfig

VALID_MEMORY_TYPES = {"preference", "alias", "habit", "home_rule", "agreement"}

_MEMORY_TYPE_LABELS = {
    "preference": "用户偏好",
    "alias": "设备别名",
    "habit": "用户习惯",
    "home_rule": "家庭规则",
    "agreement": "历史约定",
}

_SECTION_ORDER = ["preference", "alias", "habit", "home_rule", "agreement"]


@dataclass(slots=True)
class MemoryRecord:
    """一条 markdown 长期记忆。"""

    memory_id: str
    memory_type: str
    text: str
    created_at: str
    session_id: str = ""


@dataclass(slots=True)
class AgentMemory:
    """用 markdown 文件保存可审计、可手动编辑的用户长期记忆。"""

    config: MemoryConfig

    def search_context(self, user_id: str, session_id: str, query: str) -> str:
        """返回适合注入 system prompt 的长期记忆文本。"""

        del session_id, query
        records = self._read_records(user_id)
        if not records:
            return ""

        lines = ["以下是当前用户的长期记忆，只能作为偏好、习惯、别名、家庭规则和历史约定参考："]
        for record in records[: self.config.max_context_items]:
            label = _MEMORY_TYPE_LABELS.get(record.memory_type, record.memory_type)
            lines.append(f"- [{label}] {record.text}")
        return "\n".join(lines)

    def save_memory(self, user_id: str, session_id: str, memory_text: str, memory_type: str) -> bool:
        """保存 agent 已判断有长期价值的记忆文本。"""

        clean_memory_text = _clean_text(memory_text)
        normalized_memory_type = memory_type.strip()
        if not clean_memory_text or normalized_memory_type not in VALID_MEMORY_TYPES:
            return False

        records = self._read_records(user_id)
        if any(record.memory_type == normalized_memory_type and record.text == clean_memory_text for record in records):
            return False

        records.append(
            MemoryRecord(
                memory_id=str(uuid.uuid4()),
                memory_type=normalized_memory_type,
                text=clean_memory_text,
                created_at=datetime.now(UTC).isoformat(),
                session_id=session_id,
            )
        )
        self._write_records(user_id, records)
        return True

    def list_memories(self, user_id: str) -> str:
        """返回当前用户长期记忆列表，供工具 Observation 使用。"""

        records = self._read_records(user_id)
        if not records:
            return "当前没有长期记忆。"

        lines = ["当前长期记忆："]
        for index, record in enumerate(records, start=1):
            label = _MEMORY_TYPE_LABELS.get(record.memory_type, record.memory_type)
            lines.append(
                f"{index}. id={record.memory_id}，类型={label}，创建时间={record.created_at}\n{record.text}"
            )
        return "\n".join(lines)

    def delete_memory(self, user_id: str, memory_id: str) -> str:
        """删除当前用户的一条长期记忆。"""

        normalized_memory_id = memory_id.strip()
        if not normalized_memory_id:
            return "删除失败：memory_id 不能为空。"

        records = self._read_records(user_id)
        kept = [record for record in records if record.memory_id != normalized_memory_id]
        if len(kept) == len(records):
            return f"未找到可删除的长期记忆：{normalized_memory_id}"

        self._write_records(user_id, kept)
        return f"已删除长期记忆：{normalized_memory_id}"

    def clear_user_memory(self, user_id: str) -> str:
        """清空当前用户的长期记忆。"""

        memory_file = self._memory_file(user_id)
        if memory_file.exists():
            memory_file.unlink()
        return "已清空当前用户的长期记忆。"

    def _read_records(self, user_id: str) -> list[MemoryRecord]:
        memory_file = self._memory_file(user_id)
        if not memory_file.exists():
            return []

        text = memory_file.read_text(encoding="utf-8")
        records: list[MemoryRecord] = []
        current_type = ""
        pending_meta: dict[str, str] | None = None
        for raw_line in text.splitlines():
            line = raw_line.strip()
            section_match = re.match(r"^##\s+(.+)$", line)
            if section_match:
                current_type = _type_from_label(section_match.group(1).strip())
                pending_meta = None
                continue

            meta_match = re.match(r"^<!--\s*(.+?)\s*-->$", line)
            if meta_match:
                pending_meta = _parse_meta(meta_match.group(1))
                continue

            if not line.startswith("- ") or not current_type:
                continue

            text_value = _clean_text(line[2:])
            if not text_value:
                continue
            meta = pending_meta or {}
            records.append(
                MemoryRecord(
                    memory_id=meta.get("memory_id", str(uuid.uuid4())),
                    memory_type=current_type,
                    text=text_value,
                    created_at=meta.get("created_at", ""),
                    session_id=meta.get("session_id", ""),
                )
            )
            pending_meta = None
        return records

    def _write_records(self, user_id: str, records: list[MemoryRecord]) -> None:
        memory_file = self._memory_file(user_id)
        memory_file.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            "# Agent Memory",
            "",
            "本文件由 Agent 自动维护，用于记录长期有价值的用户偏好、别名、习惯、家庭规则和历史约定。",
            "可以手动编辑；删除条目后 Agent 将不再读取对应记忆。",
            "",
        ]
        for memory_type in _SECTION_ORDER:
            label = _MEMORY_TYPE_LABELS[memory_type]
            lines.extend([f"## {label}", ""])
            for record in records:
                if record.memory_type != memory_type:
                    continue
                lines.append(_format_meta(record))
                lines.append(f"- {record.text}")
            lines.append("")
        memory_file.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    def _memory_file(self, user_id: str) -> Path:
        base_dir = Path(self.config.memory_dir)
        if base_dir.is_absolute():
            raise RuntimeError("AGENT_MEMORY_DIR 必须是相对路径，便于不同机器测试。")
        safe_user_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", user_id.strip()) or "default"
        return base_dir / f"{safe_user_id}.md"


def create_default_agent_memory(config: MemoryConfig | None = None) -> AgentMemory:
    """创建默认 markdown 长期记忆实现。"""

    return AgentMemory(config=config or MemoryConfig.from_env())


def _clean_text(text: str) -> str:
    lines = []
    for line in text.strip().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("Thought:", "Action:", "Observation:")):
            continue
        lines.append(stripped)
    return " ".join(lines)


def _format_meta(record: MemoryRecord) -> str:
    session_id = record.session_id.replace('"', "")
    return (
        f'<!-- memory_id="{record.memory_id}" '
        f'created_at="{record.created_at}" '
        f'session_id="{session_id}" -->'
    )


def _parse_meta(meta_text: str) -> dict[str, str]:
    return {
        key: value
        for key, value in re.findall(r'([A-Za-z_]+)="([^"]*)"', meta_text)
    }


def _type_from_label(label: str) -> str:
    for memory_type, type_label in _MEMORY_TYPE_LABELS.items():
        if label == type_label:
            return memory_type
    return ""
