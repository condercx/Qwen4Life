"""测试儿童教育 RAG 知识库。"""

from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

import httpx

from agent.embedding_client import OllamaEmbeddingClient
from agent.knowledge_base import AgentKnowledgeBase
from agent.knowledge_config import KnowledgeConfig
from agent.knowledge_store import (
    InMemoryKnowledgeStore,
    KnowledgeChunk,
    build_grimms_chunks,
    expand_knowledge_query,
)


class KnowledgeConfigTests(unittest.TestCase):
    """验证知识库配置。"""

    def test_default_knowledge_config_is_disabled_and_uses_bge_m3(self) -> None:
        with patch("agent.knowledge_config._load_default_env_files", lambda: None), patch.dict(
            os.environ,
            {},
            clear=True,
        ):
            config = KnowledgeConfig.from_env()

        self.assertFalse(config.enabled)
        self.assertEqual(config.chroma_path, ".agent_kb/chroma")
        self.assertEqual(config.embed_model, "bge-m3")
        self.assertEqual(config.embed_text_chars, 120)
        self.assertEqual(config.source_path, "data/knowledge/grimms_fairy_tales.txt")

    def test_knowledge_config_reads_env_values(self) -> None:
        with patch("agent.knowledge_config._load_default_env_files", lambda: None), patch.dict(
            os.environ,
            {
                "AGENT_KB_ENABLED": "true",
                "AGENT_KB_EMBED_MODEL": "custom-bge",
                "AGENT_KB_CHROMA_PATH": ".tmp_kb/chroma",
                "AGENT_KB_COLLECTION": "custom_kb",
                "AGENT_KB_TOP_K": "3",
            },
            clear=True,
        ):
            config = KnowledgeConfig.from_env()

        self.assertTrue(config.enabled)
        self.assertEqual(config.embed_model, "custom-bge")
        self.assertEqual(config.chroma_path, ".tmp_kb/chroma")
        self.assertEqual(config.collection, "custom_kb")
        self.assertEqual(config.top_k, 3)


class OllamaEmbeddingClientTests(unittest.TestCase):
    """验证 Ollama embedding 客户端不触网即可解析响应。"""

    def test_embed_parses_ollama_embeddings_field(self) -> None:
        requests: list[dict[str, object]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(json.loads(request.content.decode("utf-8")))
            return httpx.Response(200, json={"embeddings": [[0.1, 0.2], [0.3, 0.4]]})

        transport = httpx.MockTransport(handler)
        config = KnowledgeConfig(ollama_embed_url="http://ollama.local/api/embed", embed_model="bge-m3")
        client = OllamaEmbeddingClient(config=config, transport=transport)

        embeddings = client.embed(["hello", "world"])

        self.assertEqual(embeddings, [[0.1, 0.2], [0.3, 0.4]])
        self.assertEqual(requests[0]["model"], "bge-m3")
        self.assertEqual(requests[0]["input"], ["hello", "world"])

    def test_embed_raises_runtime_error_on_bad_response(self) -> None:
        transport = httpx.MockTransport(lambda request: httpx.Response(500, text="boom"))
        client = OllamaEmbeddingClient(
            config=KnowledgeConfig(ollama_embed_url="http://ollama.local/api/embed"),
            transport=transport,
        )

        with self.assertRaises(RuntimeError):
            client.embed(["hello"])


class KnowledgeBaseTests(unittest.TestCase):
    """验证知识库 query expansion、BM25 和格式化结果。"""

    def test_query_expansion_maps_chinese_story_titles(self) -> None:
        variants = expand_knowledge_query("给孩子讲小红帽的寓意")

        self.assertTrue(any("little red" in variant for variant in variants))
        self.assertTrue(any("moral" in variant for variant in variants))

    def test_in_memory_knowledge_store_uses_bm25_and_query_expansion(self) -> None:
        store = InMemoryKnowledgeStore(
            [
                KnowledgeChunk(
                    chunk_id="red",
                    title="LITTLE RED-CAP [LITTLE RED RIDING HOOD]",
                    text="A little girl meets a wolf on the way to her grandmother.",
                    source="test",
                ),
                KnowledgeChunk(
                    chunk_id="ash",
                    title="ASHPUTTEL",
                    text="A girl loses a slipper after the ball.",
                    source="test",
                ),
            ]
        )

        results = store.search("小红帽适合讲什么寓意", top_k=1, min_score=0.0)

        self.assertEqual(results[0].chunk_id, "red")

    def test_agent_knowledge_base_formats_observation(self) -> None:
        store = InMemoryKnowledgeStore(
            [
                KnowledgeChunk(
                    chunk_id="red",
                    title="LITTLE RED-CAP [LITTLE RED RIDING HOOD]",
                    text="A little girl meets a wolf on the way to her grandmother.",
                    source="Project Gutenberg",
                )
            ]
        )
        knowledge_base = AgentKnowledgeBase(store=store, config=KnowledgeConfig(top_k=1))

        result = knowledge_base.search("小红帽")

        self.assertIn("知识库检索结果", result)
        self.assertIn("LITTLE RED-CAP", result)
        self.assertIn("Project Gutenberg", result)

    def test_build_grimms_chunks_splits_gutenberg_text(self) -> None:
        raw_text = """*** START OF THE PROJECT GUTENBERG EBOOK GRIMMS' FAIRY TALES ***
CONTENTS:
     THE GOLDEN BIRD
     HANSEL AND GRETEL

THE GOLDEN BIRD

There was once a king with a golden apple tree.

HANSEL AND GRETEL

Hansel and Gretel walked into the forest.
*** END OF THE PROJECT GUTENBERG EBOOK GRIMMS' FAIRY TALES ***"""

        chunks = build_grimms_chunks(raw_text, source="test", chunk_chars=1200, chunk_overlap=0)

        self.assertEqual([chunk.title for chunk in chunks], ["THE GOLDEN BIRD", "HANSEL AND GRETEL"])
        self.assertIn("golden apple", chunks[0].text)


if __name__ == "__main__":
    unittest.main()
