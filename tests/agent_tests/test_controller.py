"""使用假模型客户端测试 Agent 控制器。"""

from __future__ import annotations

from collections.abc import Iterator
import os
import unittest
from typing import Any
from unittest.mock import patch

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

    def __init__(
        self,
        responses: list[list[dict[str, str]]],
        completion_responses: list[str] | None = None,
    ) -> None:
        self.responses = [list(response) for response in responses]
        self.completion_responses = list(completion_responses or [])
        self.calls: list[list[dict[str, str]]] = []
        self.completion_calls: list[list[dict[str, str]]] = []

    def chat_completion(self, messages: list[dict[str, str]]) -> str:
        """记录非流式请求。"""

        self.completion_calls.append(list(messages))
        if not self.completion_responses:
            raise AssertionError("FakeStreamingLLMClient 缺少预设非流式响应。")
        return self.completion_responses.pop(0)

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


class FakeAgentMemory:
    """记录 controller 对长期记忆的检索和保存调用。"""

    def __init__(
        self,
        context: str = "",
        search_error: Exception | None = None,
        save_error: Exception | None = None,
    ) -> None:
        self.context = context
        self.search_error = search_error
        self.save_error = save_error
        self.search_calls: list[tuple[str, str, str]] = []
        self.save_calls: list[tuple[str, str, str, str]] = []

    def search_context(self, user_id: str, session_id: str, query: str) -> str:
        """模拟长期记忆检索。"""

        self.search_calls.append((user_id, session_id, query))
        if self.search_error is not None:
            raise self.search_error
        return self.context

    def save_memory(self, user_id: str, session_id: str, memory_text: str, memory_type: str) -> None:
        """模拟长期记忆保存。"""

        self.save_calls.append((user_id, session_id, memory_text, memory_type))
        if self.save_error is not None:
            raise self.save_error


class FakeKnowledgeBase:
    """记录 controller 对知识库工具的查询调用。"""

    def __init__(self, result: str = "知识库检索结果：小红帽遇到了狼。") -> None:
        self.result = result
        self.search_calls: list[str] = []

    def search(self, query: str) -> str:
        self.search_calls.append(query)
        return self.result


def build_agent(
    client: Any,
    env: SmartHomeEnv | None = None,
    memory: Any | None = None,
    knowledge_base: Any | None = None,
) -> SimpleSmartHomeAgent:
    """构造不依赖真实 HTTP 服务和模型服务的 Agent。"""

    adapter = InMemoryEnvironmentAdapter(env=env or SmartHomeEnv())
    tools = ToolRegistry(adapter=adapter)
    with patch.dict(os.environ, {"AGENT_MEMORY_ENABLED": "false", "AGENT_KB_ENABLED": "false"}):
        return SimpleSmartHomeAgent(
            tools=tools,
            client=client,
            memory=memory,
            knowledge_base=knowledge_base,
        )


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

    def test_knowledge_base_action_then_answer_uses_tool_observation(self) -> None:
        knowledge_base = FakeKnowledgeBase()
        client = FakeLLMClient(
            [
                'Thought: 需要查知识库\nAction: search_knowledge_base(query="小红帽的寓意")',
                "Thought: 已获得故事材料\nAnswer: 小红帽提醒我们不要轻信陌生人。",
            ]
        )
        agent = build_agent(client, knowledge_base=knowledge_base)

        result = agent.handle_user_input("agent-session", "给孩子讲小红帽的寓意")

        self.assertEqual(result.reply, "小红帽提醒我们不要轻信陌生人。")
        self.assertEqual(knowledge_base.search_calls, ["小红帽的寓意"])
        self.assertIn("Observation:", client.calls[1][-1]["content"])
        self.assertIn("search_knowledge_base", client.calls[0][0]["content"])

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

    def test_memory_disabled_keeps_prompt_without_memory_section(self) -> None:
        client = FakeLLMClient(["Thought: direct\nAnswer: OK"])
        agent = build_agent(client)

        result = agent.handle_user_input("agent-session", "hello")

        self.assertEqual(result.reply, "OK")
        self.assertNotIn("## 长期记忆", client.calls[0][0]["content"])

    def test_non_stream_injects_retrieved_memory_into_system_prompt(self) -> None:
        memory = FakeAgentMemory(context="用户把客厅主灯叫做小太阳。")
        client = FakeLLMClient(["Thought: use memory\nAnswer: 记住了"])
        agent = build_agent(client, memory=memory)

        result = agent.handle_user_input("agent-session", "打开小太阳")

        system_prompt = client.calls[0][0]["content"]
        self.assertEqual(result.reply, "记住了")
        self.assertIn("## 长期记忆", system_prompt)
        self.assertIn("用户把客厅主灯叫做小太阳。", system_prompt)
        self.assertEqual(memory.search_calls, [("agent-session", "agent-session", "打开小太阳")])
        self.assertEqual(memory.save_calls, [])

    def test_non_stream_save_memory_tool_saves_one_memory(self) -> None:
        memory = FakeAgentMemory()
        client = FakeLLMClient(
            [
                (
                    "Thought: 用户表达了长期偏好，需要保存\n"
                    'Action: save_memory(memory_type="preference", memory_text="用户喜欢空调默认 24 度。")'
                ),
                "Thought: 记忆已经保存\nAnswer: 好的，我记住了。",
            ]
        )
        agent = build_agent(client, memory=memory)

        result = agent.handle_user_input("agent-session", "请记住我喜欢 24 度")

        self.assertEqual(result.reply, "好的，我记住了。")
        self.assertEqual([step.type for step in result.steps], ["action", "observation", "answer"])
        self.assertEqual(
            memory.save_calls,
            [("agent-session", "agent-session", "用户喜欢空调默认 24 度。", "preference")],
        )
        self.assertNotIn("Thought:", memory.save_calls[0][2])

    def test_fallback_does_not_save_memory(self) -> None:
        memory = FakeAgentMemory(context="历史偏好")
        agent = build_agent(FailingLLMClient(), memory=memory)

        result = agent.handle_user_input("agent-session", "打开灯")

        self.assertEqual(result.reply, FALLBACK_REPLY)
        self.assertEqual(memory.save_calls, [])

    def test_stream_final_reply_saves_one_memory_turn(self) -> None:
        memory = FakeAgentMemory()
        client = FakeStreamingLLMClient(
            [
                [
                    {
                        "type": "content",
                        "content": (
                            "Thought: 用户表达了长期偏好，需要保存\n"
                            'Action: save_memory(memory_type="preference", memory_text="用户喜欢空调默认 24 度。")'
                        ),
                    }
                ],
                [{"type": "content", "content": "Thought: 记忆已经保存\nAnswer: 好的，我记住了。"}],
            ],
        )
        agent = build_agent(client, memory=memory)

        events = list(agent.handle_user_input_stream("agent-session", "你好"))

        self.assertEqual(events[-1], {"type": "final_reply", "content": "好的，我记住了。"})
        self.assertEqual(memory.save_calls, [("agent-session", "agent-session", "用户喜欢空调默认 24 度。", "preference")])
        self.assertEqual(client.completion_calls, [])

    def test_empty_memory_context_does_not_render_memory_section(self) -> None:
        memory = FakeAgentMemory(context="")
        client = FakeLLMClient(["Thought: direct\nAnswer: OK"])
        agent = build_agent(client, memory=memory)

        agent.handle_user_input("agent-session", "hello")

        self.assertEqual(memory.search_calls, [("agent-session", "agent-session", "hello")])
        self.assertNotIn("## 长期记忆", client.calls[0][0]["content"])

    def test_memory_search_error_does_not_break_main_flow(self) -> None:
        memory = FakeAgentMemory(search_error=RuntimeError("search failed"))
        client = FakeLLMClient(["Thought: direct\nAnswer: OK"])
        agent = build_agent(client, memory=memory)

        result = agent.handle_user_input("agent-session", "hello")

        self.assertEqual(result.reply, "OK")
        self.assertNotIn("## 长期记忆", client.calls[0][0]["content"])
        self.assertEqual(memory.save_calls, [])

    def test_memory_save_error_does_not_break_main_flow(self) -> None:
        memory = FakeAgentMemory(save_error=RuntimeError("save failed"))
        client = FakeLLMClient(
            [
                (
                    "Thought: 用户表达了长期偏好，需要保存\n"
                    'Action: save_memory(memory_type="preference", memory_text="用户喜欢空调默认 24 度。")'
                ),
                "Thought: 保存失败不影响回复\nAnswer: 好的。",
            ]
        )
        agent = build_agent(client, memory=memory)

        result = agent.handle_user_input("agent-session", "hello")

        self.assertEqual(result.reply, "好的。")
        self.assertEqual(memory.save_calls, [("agent-session", "agent-session", "用户喜欢空调默认 24 度。", "preference")])


if __name__ == "__main__":
    unittest.main()
