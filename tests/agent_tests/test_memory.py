"""测试长期记忆配置、客户端和内存存储。"""

from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

import httpx

from agent.memory import AgentMemory
from agent.memory_client import OllamaEmbeddingClient
from agent.memory_config import MemoryConfig
from agent.memory_decision import build_memory_decision_messages, parse_memory_decision, request_memory_decision
from agent.memory_store import expand_memory_query
from agent.memory_store import InMemoryMemoryStore


class MemoryConfigTests(unittest.TestCase):
    """验证长期记忆环境变量配置。"""

    def test_default_memory_config_is_disabled_and_relative(self) -> None:
        with patch("agent.memory_config._load_default_env_files", lambda: None), patch.dict(
            os.environ,
            {},
            clear=True,
        ):
            config = MemoryConfig.from_env()

        self.assertFalse(config.enabled)
        self.assertEqual(config.chroma_path, ".agent_memory/chroma")
        self.assertEqual(config.embed_model, "bge-m3")

    def test_memory_config_reads_env_values(self) -> None:
        with patch("agent.memory_config._load_default_env_files", lambda: None), patch.dict(
            os.environ,
            {
                "AGENT_MEMORY_ENABLED": "true",
                "AGENT_MEMORY_EMBED_BACKEND": "ollama",
                "AGENT_MEMORY_OLLAMA_EMBED_URL": "http://localhost:11434/api/embed",
                "AGENT_MEMORY_EMBED_MODEL": "custom-bge",
                "AGENT_MEMORY_CHROMA_PATH": ".tmp_memory/chroma",
                "AGENT_MEMORY_COLLECTION": "custom_memory",
                "AGENT_MEMORY_TOP_K": "3",
                "AGENT_MEMORY_MIN_SCORE": "0.25",
            },
            clear=True,
        ):
            config = MemoryConfig.from_env()

        self.assertTrue(config.enabled)
        self.assertEqual(config.embed_backend, "ollama")
        self.assertEqual(config.ollama_embed_url, "http://localhost:11434/api/embed")
        self.assertEqual(config.embed_model, "custom-bge")
        self.assertEqual(config.chroma_path, ".tmp_memory/chroma")
        self.assertEqual(config.collection, "custom_memory")
        self.assertEqual(config.top_k, 3)
        self.assertEqual(config.min_score, 0.25)


class OllamaEmbeddingClientTests(unittest.TestCase):
    """验证 Ollama embedding 客户端不触网即可解析响应。"""

    def test_embed_parses_ollama_embeddings_field(self) -> None:
        requests: list[dict[str, object]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(json.loads(request.content.decode("utf-8")))
            return httpx.Response(200, json={"embeddings": [[0.1, 0.2], [0.3, 0.4]]})

        transport = httpx.MockTransport(handler)
        config = MemoryConfig(ollama_embed_url="http://ollama.local/api/embed", embed_model="bge-m3")
        client = OllamaEmbeddingClient(config=config, transport=transport)

        embeddings = client.embed(["hello", "world"])

        self.assertEqual(embeddings, [[0.1, 0.2], [0.3, 0.4]])
        self.assertEqual(requests[0]["model"], "bge-m3")
        self.assertEqual(requests[0]["input"], ["hello", "world"])

    def test_embed_raises_runtime_error_on_bad_response(self) -> None:
        transport = httpx.MockTransport(lambda request: httpx.Response(500, text="boom"))
        client = OllamaEmbeddingClient(
            config=MemoryConfig(ollama_embed_url="http://ollama.local/api/embed"),
            transport=transport,
        )

        with self.assertRaises(RuntimeError):
            client.embed(["hello"])


class AgentMemoryTests(unittest.TestCase):
    """验证 AgentMemory 与测试用内存存储。"""

    def test_search_context_returns_empty_when_no_memory(self) -> None:
        memory = AgentMemory(store=InMemoryMemoryStore(), config=MemoryConfig(top_k=5))

        context = memory.search_context("user-a", "session-a", "hello")

        self.assertEqual(context, "")

    def test_save_memory_and_search_context_use_clean_text(self) -> None:
        store = InMemoryMemoryStore()
        memory = AgentMemory(store=store, config=MemoryConfig(top_k=5))

        memory.save_memory(
            user_id="user-a",
            session_id="session-a",
            memory_text="  用户喜欢把灯叫小太阳  ",
            memory_type="alias",
        )
        context = memory.search_context("user-a", "session-a", "小太阳")

        self.assertIn("用户喜欢把灯叫小太阳", context)
        self.assertNotIn("Thought:", context)
        self.assertEqual(store.get_all_memories("user-a")[0].metadata["user_id"], "user-a")
        self.assertEqual(store.get_all_memories("user-a")[0].metadata["session_id"], "session-a")
        self.assertEqual(store.get_all_memories("user-a")[0].metadata["memory_type"], "alias")

    def test_save_memory_ignores_invalid_type(self) -> None:
        store = InMemoryMemoryStore()
        memory = AgentMemory(store=store, config=MemoryConfig(top_k=5))

        memory.save_memory(
            user_id="user-a",
            session_id="session-a",
            memory_text="打开客厅灯",
            memory_type="temporary_action",
        )

        self.assertEqual(store.get_all_memories("user-a"), [])

    def test_parse_memory_decision_accepts_valid_json(self) -> None:
        messages = build_memory_decision_messages("我喜欢空调 24 度", "好的，已记住。")
        decision = parse_memory_decision(
            '{"should_save": true, "memory_type": "preference", "memory_text": "用户喜欢空调默认 24 度。"}'
        )

        self.assertIn("长期记忆筛选器", messages[0]["content"])
        self.assertTrue(decision.should_save)
        self.assertEqual(decision.memory_type, "preference")
        self.assertEqual(decision.memory_text, "用户喜欢空调默认 24 度。")

    def test_parse_memory_decision_uses_last_json_object(self) -> None:
        decision = parse_memory_decision(
            'thinking {"should_save": false, "memory_type": "", "memory_text": ""}\n'
            '{"should_save": true, "memory_type": "alias", "memory_text": "以后叫旺财。"}'
        )

        self.assertTrue(decision.should_save)
        self.assertEqual(decision.memory_type, "alias")
        self.assertEqual(decision.memory_text, "以后叫旺财。")

    def test_parse_memory_decision_rejects_invalid_or_no_save(self) -> None:
        self.assertFalse(parse_memory_decision("not json").should_save)
        self.assertFalse(
            parse_memory_decision(
                '{"should_save": true, "memory_type": "unknown", "memory_text": "x"}'
            ).should_save
        )

    def test_request_memory_decision_uses_non_thinking_overrides(self) -> None:
        class OverrideClient:
            def __init__(self) -> None:
                self.kwargs: dict[str, object] = {}

            def chat_completion_with_overrides(self, messages: list[dict[str, str]], **kwargs: object) -> str:
                del messages
                self.kwargs = kwargs
                return '{"should_save": true, "memory_type": "alias", "memory_text": "以后叫旺财。"}'

        client = OverrideClient()

        decision = request_memory_decision(client, "以后你叫旺财", "好的。")

        self.assertTrue(decision.should_save)
        self.assertEqual(decision.memory_type, "alias")
        self.assertFalse(client.kwargs["enable_thinking"])
        self.assertEqual(client.kwargs["thinking_budget"], 0)
        self.assertTrue(client.kwargs["force_json_output"])
        self.assertFalse(
            parse_memory_decision(
                '{"should_save": false, "memory_type": "", "memory_text": ""}'
            ).should_save
        )

    def test_in_memory_store_filters_and_clears_by_user(self) -> None:
        store = InMemoryMemoryStore()
        store.add_memory(user_id="user-a", session_id="session-a", text="A memory")
        store.add_memory(user_id="user-b", session_id="session-b", text="B memory")

        self.assertEqual([item.text for item in store.get_all_memories("user-a")], ["A memory"])

        store.clear_user_memory("user-a")

        self.assertEqual(store.get_all_memories("user-a"), [])
        self.assertEqual([item.text for item in store.get_all_memories("user-b")], ["B memory"])

    def test_in_memory_store_deletes_one_memory_by_id(self) -> None:
        store = InMemoryMemoryStore()
        first_id = store.add_memory(user_id="user-a", session_id="session-a", text="A memory")
        second_id = store.add_memory(user_id="user-a", session_id="session-a", text="B memory")

        self.assertTrue(store.delete_memory("user-a", first_id))
        self.assertFalse(store.delete_memory("user-a", first_id))
        self.assertEqual([item.memory_id for item in store.get_all_memories("user-a")], [second_id])

    def test_agent_memory_lists_deletes_and_clears_memories(self) -> None:
        store = InMemoryMemoryStore()
        memory = AgentMemory(store=store, config=MemoryConfig(top_k=5))
        memory.save_memory("user-a", "session-a", "用户把客厅灯叫做小太阳。", "alias")
        memory_id = store.get_all_memories("user-a")[0].memory_id

        list_result = memory.list_memories("user-a")
        delete_result = memory.delete_memory("user-a", memory_id)

        self.assertIn(memory_id, list_result)
        self.assertIn("设备别名", list_result)
        self.assertIn("已删除", delete_result)
        self.assertEqual(memory.list_memories("user-a"), "当前没有长期记忆。")

        memory.save_memory("user-a", "session-a", "用户喜欢空调默认 24 度。", "preference")
        self.assertIn("用户偏好", memory.list_memories("user-a"))
        self.assertEqual(memory.clear_user_memory("user-a"), "已清空当前用户的长期记忆。")
        self.assertEqual(store.get_all_memories("user-a"), [])

    def test_hybrid_search_uses_bm25_kg_and_query_expansion(self) -> None:
        store = InMemoryMemoryStore()
        memory = AgentMemory(store=store, config=MemoryConfig(top_k=2))
        memory.save_memory("user-a", "session-a", "用户把客厅主灯叫做小太阳。", "alias")
        memory.save_memory("user-a", "session-a", "用户喜欢空调默认 24 度。", "preference")

        alias_context = memory.search_context("user-a", "session-a", "打开小太阳")
        expanded = expand_memory_query("查看记忆里的空调默认")
        preference_context = memory.search_context("user-a", "session-a", "查看记忆里的空调默认")

        self.assertIn("小太阳", alias_context)
        self.assertIn("空调", preference_context)
        self.assertTrue(any("偏好" in item or "约定" in item for item in expanded))


if __name__ == "__main__":
    unittest.main()
