"""Tests for the browser demo gateway."""

from __future__ import annotations

from collections.abc import Iterator
import json
import os
import unittest
from typing import Any
from unittest.mock import patch

from fastapi.testclient import TestClient

from agent.controller import SimpleSmartHomeAgent
from agent.server import DemoGateway, create_app
from agent.tools import ToolRegistry
from environment.adapter import InMemoryEnvironmentAdapter
from environment.smart_home_env import SmartHomeEnv


class FakeStreamingLLMClient:
    """Return pre-defined streaming chunks."""

    def __init__(self, responses: list[list[dict[str, str]]]) -> None:
        self.responses = [list(response) for response in responses]

    def chat_completion_stream(self, messages: list[dict[str, str]]) -> Iterator[dict[str, str]]:
        del messages
        if not self.responses:
            raise AssertionError("FakeStreamingLLMClient 缺少预设响应。")
        yield from self.responses.pop(0)


def build_test_client(llm_client: Any | None = None) -> TestClient:
    """Create a TestClient backed by in-memory environment and fake LLM."""

    adapter = InMemoryEnvironmentAdapter(env=SmartHomeEnv())
    tools = ToolRegistry(adapter=adapter)
    with patch.dict(os.environ, {"AGENT_MEMORY_ENABLED": "false", "AGENT_KB_ENABLED": "false"}):
        agent = SimpleSmartHomeAgent(
            tools=tools,
            client=llm_client or FakeStreamingLLMClient([]),
        )
    return TestClient(create_app(DemoGateway(agent=agent)))


def parse_sse_events(text: str) -> list[dict[str, Any]]:
    """Parse the small SSE subset emitted by the demo gateway."""

    events: list[dict[str, Any]] = []
    for block in text.strip().split("\n\n"):
        lines = block.splitlines()
        event_line = next((line for line in lines if line.startswith("event:")), "")
        data_line = next((line for line in lines if line.startswith("data:")), "")
        if not event_line or not data_line:
            continue
        events.append(
            {
                "event": event_line.removeprefix("event:").strip(),
                "data": json.loads(data_line.removeprefix("data:").strip()),
            }
        )
    return events


class DemoServerTests(unittest.TestCase):
    """Verify the web demo endpoints without real model or HTTP environment."""

    def test_index_serves_dashboard_html(self) -> None:
        client = build_test_client()

        response = client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Qwen4Life 端侧 Agent 演示", response.text)
        self.assertIn("/assets/app.js", response.text)

    def test_reset_state_and_manual_action_endpoints_share_environment(self) -> None:
        client = build_test_client()

        reset_response = client.post("/api/session/web-test/reset")
        action_response = client.post(
            "/api/session/web-test/action",
            json={
                "device": "light",
                "target": "living_room_light_1",
                "command": "set_brightness",
                "params": {"brightness": 66},
            },
        )
        state_response = client.get("/api/session/web-test/state")

        self.assertEqual(reset_response.status_code, 200)
        self.assertEqual(action_response.status_code, 200)
        self.assertEqual(state_response.status_code, 200)
        self.assertEqual(action_response.json()["state"]["devices"]["living_room_light_1"]["brightness"], 66)
        self.assertEqual(state_response.json()["state"]["devices"]["living_room_light_1"]["brightness"], 66)
        self.assertIn("light_brightness_changed", {event["type"] for event in action_response.json()["events"]})

    def test_chat_stream_emits_agent_events_and_updated_state(self) -> None:
        llm_client = FakeStreamingLLMClient(
            [
                [
                    {
                        "type": "content",
                        "content": (
                            "Thought: 需要控制灯光\n"
                            'Action: control_device(device_id="living_room_light_1", '
                            'command="set_brightness", params={"brightness": 70})'
                        ),
                    }
                ],
                [{"type": "content", "content": "Thought: 操作完成\nAnswer: 已把客厅灯调到 70。"}],
            ]
        )
        client = build_test_client(llm_client)
        client.post("/api/session/web-chat/reset")

        response = client.post(
            "/api/agent/web-chat/chat/stream",
            json={"message": "把客厅灯调到 70", "verbose": True},
        )
        events = parse_sse_events(response.text)
        event_types = [event["event"] for event in events]
        state_events = [event for event in events if event["event"] == "state"]

        self.assertEqual(response.status_code, 200)
        self.assertIn("content", event_types)
        self.assertIn("action_start", event_types)
        self.assertIn("observation", event_types)
        self.assertIn("final_reply", event_types)
        self.assertEqual(events[-1]["event"], "done")
        self.assertTrue(state_events)
        self.assertEqual(
            state_events[-1]["data"]["state"]["devices"]["living_room_light_1"]["brightness"],
            70,
        )


if __name__ == "__main__":
    unittest.main()
