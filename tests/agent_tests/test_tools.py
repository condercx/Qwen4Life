"""使用内存环境适配器测试 Agent 工具。"""

from __future__ import annotations

import unittest

from agent.knowledge_base import AgentKnowledgeBase
from agent.knowledge_config import KnowledgeConfig
from agent.knowledge_store import InMemoryKnowledgeStore, KnowledgeChunk
from agent.memory import AgentMemory
from agent.memory_config import MemoryConfig
from agent.tools import ToolRegistry
from environment.actions import ERROR_DEVICE_OFFLINE
from environment.adapter import InMemoryEnvironmentAdapter
from environment.scenarios import build_all_offline_devices, build_evening_home_devices
from environment.smart_home_env import SmartHomeEnv


class ToolRegistryTests(unittest.TestCase):
    """验证不依赖 HTTP 服务和模型服务的工具行为。"""

    def tearDown(self) -> None:
        import shutil

        shutil.rmtree(".tmp_tool_memory", ignore_errors=True)

    def test_query_all_devices_uses_in_memory_environment(self) -> None:
        tools = ToolRegistry(
            adapter=InMemoryEnvironmentAdapter(
                env=SmartHomeEnv(device_factory=build_evening_home_devices)
            )
        )
        tools.adapter.create_session("tool-session")

        result = tools.execute("tool-session", "query_all_devices", {})

        self.assertEqual(len(result.splitlines()), 6)
        self.assertIn("60", result)
        self.assertIn("24.0", result)
        self.assertIn("35%", result)
        self.assertIn("26.0", result)
        self.assertNotIn("list_memories", tools.get_tools_prompt())

    def test_control_device_updates_environment_state(self) -> None:
        adapter = InMemoryEnvironmentAdapter(env=SmartHomeEnv())
        tools = ToolRegistry(adapter=adapter)
        adapter.create_session("tool-session")

        result = tools.execute(
            "tool-session",
            "control_device",
            {
                "device_id": "living_room_light_1",
                "command": "set_brightness",
                "params": {"brightness": 40},
            },
        )
        state = adapter.fetch_state("tool-session")

        self.assertIn("40", result)
        self.assertEqual(state["devices"]["living_room_light_1"]["brightness"], 40)

    def test_control_device_supports_new_device_types(self) -> None:
        adapter = InMemoryEnvironmentAdapter(env=SmartHomeEnv())
        tools = ToolRegistry(adapter=adapter)
        adapter.create_session("new-device-tool-session")

        curtain_result = tools.execute(
            "new-device-tool-session",
            "control_device",
            {
                "device_id": "living_room_curtain_1",
                "command": "set_position",
                "params": {"position_percent": 80},
            },
        )
        plug_result = tools.execute(
            "new-device-tool-session",
            "control_device",
            {
                "device_id": "desk_plug_1",
                "command": "turn_on",
                "params": {"power_watts": 10},
            },
        )
        state = adapter.fetch_state("new-device-tool-session")

        self.assertIn("80%", curtain_result)
        self.assertIn("10.0W", plug_result)
        self.assertEqual(state["devices"]["living_room_curtain_1"]["position_percent"], 80)
        self.assertTrue(state["devices"]["desk_plug_1"]["is_on"])

    def test_control_device_reports_environment_errors(self) -> None:
        adapter = InMemoryEnvironmentAdapter(
            env=SmartHomeEnv(device_factory=build_all_offline_devices)
        )
        tools = ToolRegistry(adapter=adapter)
        adapter.create_session("offline-tool-session")

        result = tools.execute(
            "offline-tool-session",
            "control_device",
            {"device_id": "living_room_light_1", "command": "turn_on", "params": {}},
        )

        self.assertIn(str(ERROR_DEVICE_OFFLINE), result)

    def test_memory_tools_list_delete_and_clear_memories(self) -> None:
        memory = AgentMemory(config=MemoryConfig(memory_dir=".tmp_tool_memory"))
        adapter = InMemoryEnvironmentAdapter(env=SmartHomeEnv())
        tools = ToolRegistry(adapter=adapter, memory=memory)
        adapter.create_session("memory-tool-session")

        memory.save_memory(
            user_id="memory-tool-session",
            session_id="memory-tool-session",
            memory_text="用户把客厅主灯叫做小太阳。",
            memory_type="alias",
        )
        list_result = tools.execute("memory-tool-session", "list_memories", {})
        memory_id = list_result.split("id=", 1)[1].split("，", 1)[0]
        delete_result = tools.execute(
            "memory-tool-session",
            "delete_memory",
            {"memory_id": memory_id},
        )

        self.assertIn("save_memory", tools.get_tools_prompt())
        self.assertIn("list_memories", tools.get_tools_prompt())
        self.assertIn(memory_id, list_result)
        self.assertIn("设备别名", list_result)
        self.assertIn("已删除", delete_result)
        self.assertEqual(tools.execute("memory-tool-session", "list_memories", {}), "当前没有长期记忆。")

        memory.save_memory(
            user_id="memory-tool-session",
            session_id="memory-tool-session",
            memory_text="用户喜欢空调默认 24 度。",
            memory_type="preference",
        )
        clear_result = tools.execute("memory-tool-session", "clear_user_memory", {})

        self.assertIn("已清空", clear_result)
        self.assertEqual(tools.execute("memory-tool-session", "list_memories", {}), "当前没有长期记忆。")

    def test_save_memory_tool_writes_markdown_memory(self) -> None:
        memory = AgentMemory(config=MemoryConfig(memory_dir=".tmp_tool_memory"))
        adapter = InMemoryEnvironmentAdapter(env=SmartHomeEnv())
        tools = ToolRegistry(adapter=adapter, memory=memory)

        save_result = tools.execute(
            "memory-tool-session",
            "save_memory",
            {
                "memory_type": "preference",
                "memory_text": "用户喜欢睡前故事温柔一点。",
            },
        )
        list_result = tools.execute("memory-tool-session", "list_memories", {})

        self.assertEqual(save_result, "已保存长期记忆。")
        self.assertIn("用户喜欢睡前故事温柔一点。", list_result)

    def test_knowledge_base_tool_searches_children_education_corpus(self) -> None:
        knowledge_base = AgentKnowledgeBase(
            store=InMemoryKnowledgeStore(
                [
                    KnowledgeChunk(
                        chunk_id="red",
                        title="LITTLE RED-CAP [LITTLE RED RIDING HOOD]",
                        text="A little girl meets a wolf on the way to her grandmother.",
                        source="Project Gutenberg",
                    )
                ]
            ),
            config=KnowledgeConfig(top_k=1),
        )
        tools = ToolRegistry(
            adapter=InMemoryEnvironmentAdapter(env=SmartHomeEnv()),
            knowledge_base=knowledge_base,
        )

        result = tools.execute("kb-tool-session", "search_knowledge_base", {"query": "小红帽的故事"})

        self.assertIn("search_knowledge_base", tools.get_tools_prompt())
        self.assertIn("LITTLE RED-CAP", result)


if __name__ == "__main__":
    unittest.main()
