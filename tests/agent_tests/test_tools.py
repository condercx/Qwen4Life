"""使用内存环境适配器测试 Agent 工具。"""

from __future__ import annotations

import unittest

from agent.tools import ToolRegistry
from environment.actions import ERROR_DEVICE_OFFLINE
from environment.adapter import InMemoryEnvironmentAdapter
from environment.scenarios import build_all_offline_devices, build_evening_home_devices
from environment.smart_home_env import SmartHomeEnv


class ToolRegistryTests(unittest.TestCase):
    """验证不依赖 HTTP 服务和模型服务的工具行为。"""

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


if __name__ == "__main__":
    unittest.main()
