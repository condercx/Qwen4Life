"""测试可复用的环境设备 fixture。"""

from __future__ import annotations

import unittest

from environment.scenarios import (
    build_evening_home_devices,
    build_washing_paused_devices,
    build_washing_running_devices,
    get_device_fixture_names,
)


class DeviceFixtureTests(unittest.TestCase):
    """验证 fixture 名称和状态隔离。"""

    def test_registered_fixture_names_cover_common_agent_test_states(self) -> None:
        names = set(get_device_fixture_names())

        self.assertGreaterEqual(
            names,
            {"default", "evening_home", "all_offline", "washing_running", "washing_paused"},
        )

    def test_evening_home_fixture_returns_fresh_device_instances(self) -> None:
        first = build_evening_home_devices()
        second = build_evening_home_devices()

        first["living_room_light_1"].online = False

        self.assertTrue(second["living_room_light_1"].online)
        self.assertEqual(second["living_room_light_1"].snapshot()["brightness"], 60)
        self.assertEqual(second["living_room_curtain_1"].snapshot()["position_percent"], 35)
        self.assertEqual(second["living_room_sensor_1"].snapshot()["temperature"], 26.0)

    def test_washing_running_fixture_sets_timer_state(self) -> None:
        devices = build_washing_running_devices(current_time=100.0, duration_seconds=30)
        washing_machine = devices["washing_machine_1"].snapshot()

        self.assertEqual(washing_machine["status"], "running")
        self.assertEqual(washing_machine["remaining_seconds"], 30)
        self.assertIsNotNone(washing_machine["expected_finish_at"])

    def test_washing_paused_fixture_has_no_expected_finish_time(self) -> None:
        devices = build_washing_paused_devices(current_time=200.0, remaining_seconds=45)
        washing_machine = devices["washing_machine_1"].snapshot()

        self.assertEqual(washing_machine["status"], "paused")
        self.assertEqual(washing_machine["remaining_seconds"], 45)
        self.assertIsNone(washing_machine["expected_finish_at"])


if __name__ == "__main__":
    unittest.main()
