"""测试 SmartHomeEnv 的确定性状态流转。"""

from __future__ import annotations

import unittest

from environment.actions import ERROR_DEVICE_OFFLINE
from environment.clock import FakeClock
from environment.scenarios import (
    build_all_offline_devices,
    build_evening_home_devices,
    build_washing_running_devices,
)
from environment.smart_home_env import SmartHomeEnv


def build_step_request(
    *,
    request_id: str,
    session_id: str,
    device: str,
    target: str,
    command: str,
    params: dict[str, object] | None = None,
) -> dict[str, object]:
    """构造测试用的最小环境 step 请求。"""

    return {
        "request_id": request_id,
        "session_id": session_id,
        "intent": f"{target}:{command}",
        "action": {
            "device": device,
            "target": target,
            "command": command,
            "params": params or {},
        },
    }


class SmartHomeEnvTests(unittest.TestCase):
    """覆盖确定性时间、fixture 注入和错误响应。"""

    def test_fake_clock_completes_washing_machine_without_sleep(self) -> None:
        clock = FakeClock(current_time=1_000.0)
        env = SmartHomeEnv(
            clock=clock,
            device_factory=lambda: build_washing_running_devices(
                current_time=clock.now(),
                duration_seconds=5,
            ),
        )

        env.reset("timed-session")
        clock.advance(5)
        events = env.get_events("timed-session")
        state = env.get_state("timed-session")

        self.assertIn("washing_completed", {event["type"] for event in events})
        self.assertEqual(state["devices"]["washing_machine_1"]["status"], "completed")

    def test_device_factory_creates_fresh_devices_for_each_session(self) -> None:
        env = SmartHomeEnv(device_factory=build_evening_home_devices)

        env.reset("first")
        env.step(
            build_step_request(
                request_id="req-1",
                session_id="first",
                device="light",
                target="living_room_light_1",
                command="turn_off",
            )
        )
        second_state = env.reset("second")

        self.assertFalse(env.get_state("first")["devices"]["living_room_light_1"]["is_on"])
        self.assertTrue(second_state["devices"]["living_room_light_1"]["is_on"])
        self.assertEqual(second_state["devices"]["living_room_light_1"]["brightness"], 60)

    def test_offline_fixture_returns_protocol_error_response(self) -> None:
        env = SmartHomeEnv(device_factory=build_all_offline_devices)
        env.reset("offline-session")

        response = env.step(
            build_step_request(
                request_id="req-1",
                session_id="offline-session",
                device="light",
                target="living_room_light_1",
                command="turn_on",
            )
        )

        self.assertFalse(response["success"])
        self.assertEqual(response["error"]["code"], ERROR_DEVICE_OFFLINE)


if __name__ == "__main__":
    unittest.main()
