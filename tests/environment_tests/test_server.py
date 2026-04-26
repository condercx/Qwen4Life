"""测试环境 FastAPI 服务边界。"""

from __future__ import annotations

import itertools
import unittest

from fastapi.testclient import TestClient

import environment.server as server
from environment.smart_home_env import SmartHomeEnv


class EnvironmentServerTests(unittest.TestCase):
    """验证 HTTP 层请求会正确转发到环境核心。"""

    def setUp(self) -> None:
        """为每个测试重置服务全局环境，避免会话状态互相影响。"""

        server.env = SmartHomeEnv()
        server.REQUEST_COUNTER = itertools.count(1)
        self.client = TestClient(server.app)

    def test_reset_and_state_endpoints_return_observation(self) -> None:
        reset_response = self.client.post("/session/http-session/reset")
        state_response = self.client.get("/session/http-session/state")
        reset_devices = reset_response.json()["devices"]
        state_devices = state_response.json()["state"]["devices"]

        self.assertEqual(reset_response.status_code, 200)
        self.assertEqual(state_response.status_code, 200)
        self.assertEqual(len(reset_devices), 6)
        self.assertIn("living_room_light_1", reset_devices)
        self.assertIn("living_room_ac_1", state_devices)
        self.assertIn("living_room_curtain_1", state_devices)
        self.assertIn("desk_plug_1", state_devices)

    def test_action_endpoint_accepts_standard_action_payload(self) -> None:
        self.client.post("/session/http-action/reset")

        response = self.client.post(
            "/session/http-action/action",
            json={
                "request_id": "http-req-1",
                "intent": "调亮客厅灯",
                "action": {
                    "device": "light",
                    "target": "living_room_light_1",
                    "command": "set_brightness",
                    "params": {"brightness": 45},
                },
            },
        )
        payload = response.json()

        self.assertEqual(response.status_code, 200)
        self.assertTrue(payload["success"])
        self.assertEqual(payload["observation"]["devices"]["living_room_light_1"]["brightness"], 45)

    def test_action_endpoint_accepts_name_and_args_aliases(self) -> None:
        self.client.post("/session/http-alias/reset")

        response = self.client.post(
            "/session/http-alias/action",
            json={
                "intent": "用别名调灯光",
                "action": {
                    "target": "living_room_light_1",
                    "name": "set_brightness",
                    "args": {"brightness": 55},
                },
            },
        )
        payload = response.json()

        self.assertEqual(response.status_code, 200)
        self.assertTrue(payload["success"])
        self.assertEqual(payload["observation"]["devices"]["living_room_light_1"]["brightness"], 55)

    def test_events_endpoint_drains_unread_events(self) -> None:
        self.client.post("/session/http-events/reset")
        self.client.post(
            "/session/http-events/action",
            json={
                "action": {
                    "device": "light",
                    "target": "living_room_light_1",
                    "command": "turn_on",
                    "params": {},
                },
            },
        )

        first_events = self.client.get("/session/http-events/events").json()["events"]
        second_events = self.client.get("/session/http-events/events").json()["events"]

        self.assertIn("session_reset", {event["type"] for event in first_events})
        self.assertIn("light_state_changed", {event["type"] for event in first_events})
        self.assertEqual(second_events, [])


if __name__ == "__main__":
    unittest.main()
