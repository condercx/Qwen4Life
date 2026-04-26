"""使用假模型客户端测试 Agent 控制器。"""

from __future__ import annotations

from collections.abc import Iterator
import unittest
from typing import Any

from agent.controller import FALLBACK_REPLY, SimpleSmartHomeAgent
from agent.tools import ToolRegistry
from environment.adapter import InMemoryEnvironmentAdapter
from environment.smart_home_env import SmartHomeEnv


class FakeLLMClient:
    """按顺序返回预设内容的假模型客户端。"""

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[list[dict[str, str]]] = []

    def chat_completion(self, messages: list[dict[str, str]]) -> str:
        """记录请求消息，并返回下一条预设模型输出。"""

        self.calls.append(list(messages))
        if not self.responses:
            raise AssertionError("FakeLLMClient 缺少预设响应。")
        return self.responses.pop(0)


class FakeStreamingLLMClient:
    """按顺序返回预设流式片段的假模型客户端。"""

    def __init__(self, responses: list[list[dict[str, str]]]) -> None:
        self.responses = [list(response) for response in responses]
        self.calls: list[list[dict[str, str]]] = []

    def chat_completion_stream(self, messages: list[dict[str, str]]) -> Iterator[dict[str, str]]:
        """记录请求消息，并逐片段返回下一组预设流式输出。"""

        self.calls.append(list(messages))
        if not self.responses:
            raise AssertionError("FakeStreamingLLMClient 缺少预设响应。")
        yield from self.responses.pop(0)


class FailingLLMClient:
    """始终抛出异常的假模型客户端。"""

    def chat_completion(self, messages: list[dict[str, str]]) -> str:
        """模拟模型服务调用失败。"""

        del messages
        raise RuntimeError("模型服务不可用。")


def build_agent(client: Any, env: SmartHomeEnv | None = None) -> SimpleSmartHomeAgent:
    """构造不依赖真实 HTTP 服务和模型服务的 Agent。"""

    adapter = InMemoryEnvironmentAdapter(env=env or SmartHomeEnv())
    tools = ToolRegistry(adapter=adapter)
    return SimpleSmartHomeAgent(tools=tools, client=client)


class SimpleSmartHomeAgentTests(unittest.TestCase):
    """验证 Agent 控制器的主要非流式和流式执行路径。"""

    def test_direct_answer_does_not_require_tool_action(self) -> None:
        client = FakeLLMClient(["Thought: 用户只是问候\nAnswer: 你好，我在。"])
        agent = build_agent(client)

        result = agent.handle_user_input("agent-session", "你好")

        self.assertEqual(result.reply, "你好，我在。")
        self.assertEqual([step.type for step in result.steps], ["answer"])
        self.assertEqual(len(client.calls), 1)

    def test_action_then_answer_updates_environment(self) -> None:
        env = SmartHomeEnv()
        client = FakeLLMClient(
            [
                (
                    "Thought: 需要调用工具调灯光\n"
                    'Action: control_device(device_id="living_room_light_1", '
                    'command="set_brightness", params={"brightness": 70})'
                ),
                "Thought: 工具已经完成操作\nAnswer: 已把客厅灯调到 70。",
            ]
        )
        agent = build_agent(client, env=env)

        result = agent.handle_user_input("agent-session", "把客厅灯调到 70")
        state = env.get_state("agent-session")

        self.assertEqual(result.reply, "已把客厅灯调到 70。")
        self.assertEqual([step.type for step in result.steps], ["action", "observation", "answer"])
        self.assertEqual(state["devices"]["living_room_light_1"]["brightness"], 70)
        self.assertIn("Observation:", client.calls[1][-1]["content"])

    def test_model_exception_returns_fallback_reply(self) -> None:
        agent = build_agent(FailingLLMClient())

        result = agent.handle_user_input("agent-session", "打开灯")

        self.assertEqual(result.reply, FALLBACK_REPLY)
        self.assertEqual(result.steps, [])

    def test_stream_direct_answer_yields_final_reply(self) -> None:
        client = FakeStreamingLLMClient(
            [[{"type": "content", "content": "Thought: 用户只是问候\nAnswer: 你好，我在。"}]]
        )
        agent = build_agent(client)

        events = list(agent.handle_user_input_stream("agent-session", "你好"))

        self.assertEqual(events[-1], {"type": "final_reply", "content": "你好，我在。"})
        self.assertEqual(len(client.calls), 1)

    def test_stream_action_then_answer_updates_environment(self) -> None:
        env = SmartHomeEnv()
        client = FakeStreamingLLMClient(
            [
                [
                    {
                        "type": "content",
                        "content": (
                            "Thought: 需要调用工具调灯光\n"
                            'Action: control_device(device_id="living_room_light_1", '
                            'command="set_brightness", params={"brightness": 35})'
                        ),
                    }
                ],
                [{"type": "content", "content": "Thought: 工具已经完成操作\nAnswer: 已把客厅灯调到 35。"}],
            ]
        )
        agent = build_agent(client, env=env)

        events = list(agent.handle_user_input_stream("agent-session", "把客厅灯调到 35"))
        state = env.get_state("agent-session")
        event_types = [event["type"] for event in events]

        self.assertEqual(event_types, ["content", "action_start", "observation", "content", "final_reply"])
        self.assertEqual(events[-1]["content"], "已把客厅灯调到 35。")
        self.assertEqual(state["devices"]["living_room_light_1"]["brightness"], 35)
        self.assertIn("Observation:", client.calls[1][-1]["content"])


if __name__ == "__main__":
    unittest.main()
