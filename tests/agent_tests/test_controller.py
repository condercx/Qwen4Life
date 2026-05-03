"""使用假模型客户端测试 Agent 流式控制器。"""

from __future__ import annotations

from collections.abc import Iterator
import os
import unittest
from typing import Any
from unittest.mock import patch

from agent.controller import EMPTY_OUTPUT_CONTINUE_PROMPT, FALLBACK_REPLY, SimpleSmartHomeAgent
from agent.tools import ToolRegistry
from environment.adapter import InMemoryEnvironmentAdapter
from environment.smart_home_env import SmartHomeEnv


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


class FailingStreamingLLMClient:
    """始终抛出异常的假流式模型客户端。"""

    def chat_completion_stream(self, messages: list[dict[str, str]]) -> Iterator[dict[str, str]]:
        """模拟模型服务调用失败。"""

        del messages
        raise RuntimeError("模型服务不可用。")
        yield


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

    def save_memory(self, user_id: str, session_id: str, memory_text: str, memory_type: str) -> bool:
        """模拟长期记忆保存。"""

        self.save_calls.append((user_id, session_id, memory_text, memory_type))
        if self.save_error is not None:
            raise self.save_error
        return True


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


def collect_events(agent: SimpleSmartHomeAgent, user_input: str, session_id: str = "agent-session") -> list[dict[str, str]]:
    """收集一次流式执行的全部事件。"""

    return list(agent.handle_user_input_stream(session_id, user_input))


def event_types(events: list[dict[str, str]]) -> list[str]:
    """提取事件类型。"""

    return [event["type"] for event in events]


class SimpleSmartHomeAgentTests(unittest.TestCase):
    """验证 Agent 控制器的流式执行路径。"""

    def test_direct_answer_does_not_require_tool_action(self) -> None:
        client = FakeStreamingLLMClient(
            [[{"type": "content", "content": "Thought: 用户只是问候\nAnswer: 你好，我在。"}]]
        )
        agent = build_agent(client)

        events = collect_events(agent, "你好")

        self.assertEqual(events[-1], {"type": "final_reply", "content": "你好，我在。"})
        self.assertEqual(event_types(events), ["content", "final_reply"])
        self.assertEqual(len(client.calls), 1)

    def test_action_then_answer_updates_environment(self) -> None:
        env = SmartHomeEnv()
        client = FakeStreamingLLMClient(
            [
                [
                    {
                        "type": "content",
                        "content": (
                            "Thought: 需要调用工具调灯光\n"
                            'Action: control_device(device_id="living_room_light_1", '
                            'command="set_brightness", params={"brightness": 70})'
                        ),
                    }
                ],
                [{"type": "content", "content": "Thought: 工具已经完成操作\nAnswer: 已把客厅灯调到 70。"}],
            ]
        )
        agent = build_agent(client, env=env)

        events = collect_events(agent, "把客厅灯调到 70")
        state = env.get_state("agent-session")

        self.assertEqual(events[-1], {"type": "final_reply", "content": "已把客厅灯调到 70。"})
        self.assertEqual(event_types(events), ["content", "action_start", "observation", "content", "final_reply"])
        self.assertEqual(state["devices"]["living_room_light_1"]["brightness"], 70)
        self.assertIn("Observation:", client.calls[1][-1]["content"])

    def test_model_multiple_actions_executes_only_first_action_cleanly(self) -> None:
        env = SmartHomeEnv()
        client = FakeStreamingLLMClient(
            [
                [
                    {
                        "type": "content",
                        "content": (
                            "Thought: 用户要求多个设备控制，需要逐个执行。\n"
                            'Action: control_device(device_id="living_room_light_1", command="turn_on")\n'
                            'Action: control_device(device_id="living_room_curtain_1", command="close")\n'
                            'Action: control_device(device_id="living_room_ac_1", command="set_temperature", '
                            'params={"temperature": 24})'
                        ),
                    }
                ],
                [{"type": "content", "content": "Thought: 第一项完成\nAnswer: 已先打开客厅灯。"}],
            ]
        )
        agent = build_agent(client, env=env)

        events = collect_events(agent, "打开客厅灯，把窗帘关上，空调调到 24 度")
        state = env.get_state("agent-session")

        self.assertEqual(events[-1], {"type": "final_reply", "content": "已先打开客厅灯。"})
        self.assertEqual([event["type"] for event in events].count("observation"), 1)
        self.assertTrue(state["devices"]["living_room_light_1"]["is_on"])
        self.assertEqual(state["devices"]["living_room_curtain_1"]["position_percent"], 0)
        self.assertEqual(state["devices"]["living_room_ac_1"]["target_temperature"], 26.0)
        self.assertIn('command="turn_on"', client.calls[1][-2]["content"])
        self.assertNotIn("living_room_curtain_1", client.calls[1][-2]["content"])
        self.assertNotIn("set_temperature", client.calls[1][-2]["content"])

    def test_history_stores_final_answer_without_react_thought(self) -> None:
        client = FakeStreamingLLMClient(
            [
                [{"type": "content", "content": "Thought: 这是一大段内部推理\nAnswer: 这是给用户看的回答。"}],
                [{"type": "content", "content": "Thought: 继续\nAnswer: 第二轮回答。"}],
            ]
        )
        agent = build_agent(client)

        collect_events(agent, "先问一个复杂问题", session_id="history-session")
        collect_events(agent, "再问一个问题", session_id="history-session")

        second_call_messages = client.calls[1]
        history_text = "\n".join(message["content"] for message in second_call_messages[1:])
        self.assertIn("这是给用户看的回答。", history_text)
        self.assertNotIn("这是一大段内部推理", history_text)
        self.assertNotIn("Thought:", history_text)

    def test_history_trimming_respects_character_budget(self) -> None:
        agent = build_agent(FakeStreamingLLMClient([]))
        agent._session_histories["trim-session"] = [
            {"role": "user", "content": "旧消息" * 20},
            {"role": "assistant", "content": "旧回答" * 20},
            {"role": "user", "content": "新问题"},
            {"role": "assistant", "content": "新回答"},
        ]

        agent._trim_history("trim-session", max_messages=10, max_chars=12)

        history_text = "\n".join(message["content"] for message in agent._session_histories["trim-session"])
        self.assertNotIn("旧消息", history_text)
        self.assertNotIn("旧回答", history_text)
        self.assertIn("新问题", history_text)
        self.assertIn("新回答", history_text)

    def test_history_trimming_compacts_single_long_message(self) -> None:
        agent = build_agent(FakeStreamingLLMClient([]))
        agent._session_histories["compact-session"] = [
            {"role": "assistant", "content": "很长的回答" * 500},
        ]

        agent._trim_history("compact-session", max_messages=10, max_chars=1000)

        compacted = agent._session_histories["compact-session"][0]["content"]
        self.assertLessEqual(len(compacted), 1200)
        self.assertIn("已截断", compacted)

    def test_knowledge_base_action_then_answer_uses_tool_observation(self) -> None:
        knowledge_base = FakeKnowledgeBase()
        client = FakeStreamingLLMClient(
            [
                [{"type": "content", "content": 'Thought: 需要查知识库\nAction: search_knowledge_base(query="小红帽的寓意")'}],
                [{"type": "content", "content": "Thought: 已获得故事材料\nAnswer: 小红帽提醒我们不要轻信陌生人。"}],
            ]
        )
        agent = build_agent(client, knowledge_base=knowledge_base)

        events = collect_events(agent, "给孩子讲小红帽的寓意")

        self.assertEqual(events[-1], {"type": "final_reply", "content": "小红帽提醒我们不要轻信陌生人。"})
        self.assertEqual(knowledge_base.search_calls, ["小红帽的寓意"])
        self.assertIn("Observation:", client.calls[1][-1]["content"])
        self.assertIn("search_knowledge_base", client.calls[0][0]["content"])

    def test_model_exception_returns_fallback_reply(self) -> None:
        agent = build_agent(FailingStreamingLLMClient())

        events = collect_events(agent, "打开灯")

        self.assertEqual(events[-1], {"type": "error", "content": FALLBACK_REPLY})
        self.assertEqual(event_types(events), ["error", "error"])

    def test_empty_stream_output_continues_until_valid_answer(self) -> None:
        client = FakeStreamingLLMClient(
            [
                [],
                [{"type": "content", "content": "Thought: 补全回答\nAnswer: 现在有回答了。"}],
            ]
        )
        agent = build_agent(client)

        events = collect_events(agent, "你好")

        self.assertEqual(events[-1], {"type": "final_reply", "content": "现在有回答了。"})
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(client.calls[1][-1]["content"], EMPTY_OUTPUT_CONTINUE_PROMPT)

    def test_stream_reasoning_only_output_continues_until_valid_answer(self) -> None:
        client = FakeStreamingLLMClient(
            [
                [{"type": "reasoning", "content": "还在推理，但没有 content"}],
                [{"type": "content", "content": "Thought: 补全回答\nAnswer: 现在有回答了。"}],
            ]
        )
        agent = build_agent(client)

        events = collect_events(agent, "你好")

        self.assertEqual(events[-1], {"type": "final_reply", "content": "现在有回答了。"})
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(client.calls[1][-1]["content"], EMPTY_OUTPUT_CONTINUE_PROMPT)

    def test_stream_empty_answer_output_continues_until_valid_answer(self) -> None:
        client = FakeStreamingLLMClient(
            [
                [{"type": "content", "content": "Thought: 没有最终内容\nAnswer: "}],
                [{"type": "content", "content": "Thought: 补全回答\nAnswer: 现在有回答了。"}],
            ]
        )
        agent = build_agent(client)

        events = collect_events(agent, "你好")

        self.assertEqual(events[-1], {"type": "final_reply", "content": "现在有回答了。"})
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(client.calls[1][-1]["content"], EMPTY_OUTPUT_CONTINUE_PROMPT)

    def test_memory_disabled_keeps_prompt_without_memory_section(self) -> None:
        client = FakeStreamingLLMClient(
            [[{"type": "content", "content": "Thought: direct\nAnswer: OK"}]]
        )
        agent = build_agent(client)

        events = collect_events(agent, "hello")

        self.assertEqual(events[-1], {"type": "final_reply", "content": "OK"})
        self.assertNotIn("## 长期记忆", client.calls[0][0]["content"])

    def test_injects_retrieved_memory_into_system_prompt(self) -> None:
        memory = FakeAgentMemory(context="用户把客厅主灯叫做小太阳。")
        client = FakeStreamingLLMClient(
            [[{"type": "content", "content": "Thought: use memory\nAnswer: 记住了"}]]
        )
        agent = build_agent(client, memory=memory)

        events = collect_events(agent, "打开小太阳")

        system_prompt = client.calls[0][0]["content"]
        self.assertEqual(events[-1], {"type": "final_reply", "content": "记住了"})
        self.assertIn("## 长期记忆", system_prompt)
        self.assertIn("用户把客厅主灯叫做小太阳。", system_prompt)
        self.assertEqual(memory.search_calls, [("agent-session", "agent-session", "打开小太阳")])
        self.assertEqual(memory.save_calls, [])

    def test_save_memory_tool_saves_one_memory(self) -> None:
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
            ]
        )
        agent = build_agent(client, memory=memory)

        events = collect_events(agent, "请记住我喜欢 24 度")

        self.assertEqual(events[-1], {"type": "final_reply", "content": "好的，我记住了。"})
        self.assertEqual(event_types(events), ["content", "action_start", "observation", "content", "final_reply"])
        self.assertEqual(
            memory.save_calls,
            [("agent-session", "agent-session", "用户喜欢空调默认 24 度。", "preference")],
        )
        self.assertNotIn("Thought:", memory.save_calls[0][2])

    def test_fallback_does_not_save_memory(self) -> None:
        memory = FakeAgentMemory(context="历史偏好")
        agent = build_agent(FailingStreamingLLMClient(), memory=memory)

        events = collect_events(agent, "打开灯")

        self.assertEqual(events[-1], {"type": "error", "content": FALLBACK_REPLY})
        self.assertEqual(memory.save_calls, [])

    def test_empty_memory_context_does_not_render_memory_section(self) -> None:
        memory = FakeAgentMemory(context="")
        client = FakeStreamingLLMClient(
            [[{"type": "content", "content": "Thought: direct\nAnswer: OK"}]]
        )
        agent = build_agent(client, memory=memory)

        collect_events(agent, "hello")

        self.assertEqual(memory.search_calls, [("agent-session", "agent-session", "hello")])
        self.assertNotIn("## 长期记忆", client.calls[0][0]["content"])

    def test_memory_search_error_does_not_break_main_flow(self) -> None:
        memory = FakeAgentMemory(search_error=RuntimeError("search failed"))
        client = FakeStreamingLLMClient(
            [[{"type": "content", "content": "Thought: direct\nAnswer: OK"}]]
        )
        agent = build_agent(client, memory=memory)

        events = collect_events(agent, "hello")

        self.assertEqual(events[-1], {"type": "final_reply", "content": "OK"})
        self.assertNotIn("## 长期记忆", client.calls[0][0]["content"])
        self.assertEqual(memory.save_calls, [])

    def test_memory_save_error_does_not_break_main_flow(self) -> None:
        memory = FakeAgentMemory(save_error=RuntimeError("save failed"))
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
                [{"type": "content", "content": "Thought: 保存失败不影响回复\nAnswer: 好的。"}],
            ]
        )
        agent = build_agent(client, memory=memory)

        events = collect_events(agent, "hello")

        self.assertEqual(events[-1], {"type": "final_reply", "content": "好的。"})
        self.assertEqual(memory.save_calls, [("agent-session", "agent-session", "用户喜欢空调默认 24 度。", "preference")])


if __name__ == "__main__":
    unittest.main()
