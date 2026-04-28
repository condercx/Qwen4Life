"""测试 markdown 长期记忆。"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch

from agent.memory import AgentMemory
from agent.memory_config import MemoryConfig


class MemoryConfigTests(unittest.TestCase):
    """验证 markdown 长期记忆环境变量配置。"""

    def test_default_memory_config_is_disabled_and_relative(self) -> None:
        with patch("agent.memory_config._load_default_env_files", lambda: None), patch.dict(
            os.environ,
            {},
            clear=True,
        ):
            config = MemoryConfig.from_env()

        self.assertFalse(config.enabled)
        self.assertEqual(config.memory_dir, ".agent_memory/profile")
        self.assertEqual(config.max_context_items, 20)

    def test_memory_config_reads_env_values(self) -> None:
        with patch("agent.memory_config._load_default_env_files", lambda: None), patch.dict(
            os.environ,
            {
                "AGENT_MEMORY_ENABLED": "true",
                "AGENT_MEMORY_DIR": ".tmp_memory/profile",
                "AGENT_MEMORY_MAX_CONTEXT_ITEMS": "3",
            },
            clear=True,
        ):
            config = MemoryConfig.from_env()

        self.assertTrue(config.enabled)
        self.assertEqual(config.memory_dir, ".tmp_memory/profile")
        self.assertEqual(config.max_context_items, 3)


class AgentMemoryTests(unittest.TestCase):
    """验证 AgentMemory 的 markdown 读写行为。"""

    def setUp(self) -> None:
        self.memory_dir = Path(".tmp_agent_memory_tests")
        shutil.rmtree(self.memory_dir, ignore_errors=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.memory_dir, ignore_errors=True)

    def test_search_context_returns_empty_when_no_memory(self) -> None:
        memory = AgentMemory(config=MemoryConfig(memory_dir=str(self.memory_dir)))

        context = memory.search_context("user-a", "session-a", "hello")

        self.assertEqual(context, "")

    def test_save_memory_writes_markdown_and_loads_context(self) -> None:
        memory = AgentMemory(config=MemoryConfig(memory_dir=str(self.memory_dir)))

        memory.save_memory(
            user_id="user-a",
            session_id="session-a",
            memory_text="  用户喜欢把灯叫小太阳  ",
            memory_type="alias",
        )
        context = memory.search_context("user-a", "session-a", "小太阳")
        memory_file = self.memory_dir / "user-a.md"

        self.assertTrue(memory_file.exists())
        self.assertIn("## 设备别名", memory_file.read_text(encoding="utf-8"))
        self.assertIn("用户喜欢把灯叫小太阳", context)
        self.assertNotIn("Thought:", context)

    def test_save_memory_ignores_invalid_type_and_duplicate(self) -> None:
        memory = AgentMemory(config=MemoryConfig(memory_dir=str(self.memory_dir)))

        memory.save_memory("user-a", "session-a", "打开客厅灯", "temporary_action")
        memory.save_memory("user-a", "session-a", "用户喜欢空调默认 24 度。", "preference")
        memory.save_memory("user-a", "session-a", "用户喜欢空调默认 24 度。", "preference")

        list_result = memory.list_memories("user-a")
        self.assertNotIn("temporary_action", list_result)
        self.assertEqual(list_result.count("用户喜欢空调默认 24 度。"), 1)

    def test_agent_memory_lists_deletes_and_clears_memories(self) -> None:
        memory = AgentMemory(config=MemoryConfig(memory_dir=str(self.memory_dir)))
        memory.save_memory("user-a", "session-a", "用户把客厅灯叫做小太阳。", "alias")

        list_result = memory.list_memories("user-a")
        memory_id = list_result.split("id=", 1)[1].split("，", 1)[0]
        delete_result = memory.delete_memory("user-a", memory_id)

        self.assertIn(memory_id, list_result)
        self.assertIn("设备别名", list_result)
        self.assertIn("已删除", delete_result)
        self.assertEqual(memory.list_memories("user-a"), "当前没有长期记忆。")

        memory.save_memory("user-a", "session-a", "用户喜欢空调默认 24 度。", "preference")
        self.assertIn("用户偏好", memory.list_memories("user-a"))
        self.assertEqual(memory.clear_user_memory("user-a"), "已清空当前用户的长期记忆。")
        self.assertEqual(memory.list_memories("user-a"), "当前没有长期记忆。")

    def test_absolute_memory_dir_is_rejected(self) -> None:
        memory = AgentMemory(config=MemoryConfig(memory_dir=str(Path.cwd())))

        with self.assertRaises(RuntimeError):
            memory.save_memory("user-a", "session-a", "用户喜欢 24 度。", "preference")


if __name__ == "__main__":
    unittest.main()
