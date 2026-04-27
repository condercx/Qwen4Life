"""
轻量 Memory RAG 系统
架构：ChromaDB 向量存储 + BAAI/bge-m3 嵌入 + Qwen3.5-4B（关闭思考模式）

为何不用 Mem0 直接驱动：
  Qwen3.5-4B 是 thinking 模型，Mem0 的 JSON 事实提取依赖结构化输出，
  在 thinking 模式下会返回空内容。本方案绕过 LLM JSON 提取，
  直接用对话 chunk 做语义检索，更稳健。
"""

import os
# 清除代理环境变量（必须在 import httpx/openai 之前）
for _k in list(os.environ):
    if 'proxy' in _k.lower() and _k != 'GOPROXY':
        del os.environ[_k]

import time
import uuid
import logging
from typing import Optional
import chromadb
from chromadb.config import Settings
from openai import OpenAI, APIConnectionError, APITimeoutError
from config import API_KEY, BASE_URL, LLM_MODEL, EMBED_MODEL

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CHROMA_PATH = "./chroma_db"


class MemoryRAGSystem:
    """
    轻量 Memory RAG 系统
    - 存储：每轮对话 chunk → 向量化 → ChromaDB
    - 检索：query → 向量化 → Top-K 语义检索
    - 生成：检索到的记忆注入 system prompt → Qwen3.5-4B
    """

    def __init__(self):
        logger.info("初始化 Memory RAG 系统...")
        self.client = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=120.0, max_retries=0)
        self.chroma = chromadb.PersistentClient(
            path=CHROMA_PATH,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.chroma.get_or_create_collection(
            name="memory_rag_v1",
            metadata={"hnsw:space": "cosine"},
        )
        self.conversation_history: list[dict] = []
        logger.info("系统初始化完成 | chroma_path=%s | model=%s", CHROMA_PATH, LLM_MODEL)

    # ------------------------------------------------------------------ #
    #  嵌入                                                                #
    # ------------------------------------------------------------------ #

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """调用 SiliconFlow bge-m3 嵌入模型"""
        resp = self.client.embeddings.create(model=EMBED_MODEL, input=texts)
        return [item.embedding for item in resp.data]

    # ------------------------------------------------------------------ #
    #  LLM 调用（关闭 Qwen3 thinking 模式）                               #
    # ------------------------------------------------------------------ #

    def _llm(self, messages: list[dict], max_tokens: int = 512,
             max_retries: int = 2) -> tuple[str, dict]:
        """调用 Qwen3.5-4B，强制关闭思考模式，含手动重试"""
        for attempt in range(max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=max_tokens,
                    extra_body={"enable_thinking": False},
                )
                content = resp.choices[0].message.content or ""
                usage = {
                    "prompt_tokens": resp.usage.prompt_tokens,
                    "completion_tokens": resp.usage.completion_tokens,
                    "total_tokens": resp.usage.total_tokens,
                }
                return content, usage
            except (APIConnectionError, APITimeoutError) as e:
                if attempt < max_retries - 1:
                    wait = 5 * (attempt + 1)
                    logger.warning("LLM 连接错误 (attempt %d/%d)，%ds 后重试: %s",
                                   attempt + 1, max_retries, wait, e)
                    time.sleep(wait)
                else:
                    logger.error("LLM 调用失败，已重试 %d 次: %s", max_retries, e)
                    raise

    # ------------------------------------------------------------------ #
    #  核心接口                                                             #
    # ------------------------------------------------------------------ #

    def add_memory(self, text: str, user_id: str, metadata: Optional[dict] = None) -> dict:
        """将一段文本存入向量记忆库"""
        t0 = time.perf_counter()
        emb = self._embed([text])[0]
        doc_id = str(uuid.uuid4())
        meta = {"user_id": user_id, "text": text[:500]}
        if metadata:
            meta.update({k: str(v) for k, v in metadata.items()})
        self.collection.add(
            ids=[doc_id],
            embeddings=[emb],
            documents=[text],
            metadatas=[meta],
        )
        latency = time.perf_counter() - t0
        return {"id": doc_id, "latency_s": round(latency, 3)}

    def search_memory(self, query: str, user_id: str, top_k: int = 5) -> dict:
        """语义检索与 query 最相关的记忆"""
        t0 = time.perf_counter()
        emb = self._embed([query])[0]
        count = self.collection.count()
        actual_k = min(top_k, max(1, count))
        results = self.collection.query(
            query_embeddings=[emb],
            n_results=actual_k,
            where={"user_id": user_id},
            include=["documents", "distances", "metadatas"],
        )
        latency = time.perf_counter() - t0
        memories = []
        if results["documents"] and results["documents"][0]:
            for doc, dist, meta in zip(
                results["documents"][0],
                results["distances"][0],
                results["metadatas"][0],
            ):
                memories.append({
                    "memory": doc,
                    "score": round(1 - dist, 4),  # cosine 距离转相似度
                    "metadata": meta,
                })
        return {"memories": memories, "latency_s": round(latency, 3)}

    def get_all_memories(self, user_id: str) -> list[dict]:
        """获取用户全部记忆"""
        results = self.collection.get(where={"user_id": user_id}, include=["documents", "metadatas"])
        return [
            {"memory": doc, "metadata": meta}
            for doc, meta in zip(results["documents"], results["metadatas"])
        ]

    def chat(self, user_message: str, user_id: str, session_id: str = "default",
             store_to_memory: bool = True) -> dict:
        """
        带记忆的对话
        流程：检索相关记忆 → 构造 prompt → LLM 生成 → 存储本轮对话
        """
        t0 = time.perf_counter()

        # 1. 检索相关历史记忆
        search_result = self.search_memory(user_message, user_id, top_k=5)
        retrieved_memories = search_result["memories"]
        retrieval_latency = search_result["latency_s"]

        # 2. 构造系统提示
        if retrieved_memories:
            memory_text = "\n".join(
                f"- {m['memory'][:200]}" for m in retrieved_memories
            )
            system_prompt = (
                "你是一个拥有长期记忆的智能助手。\n"
                "以下是你关于该用户的历史记忆，请结合这些信息给出个性化回答：\n"
                f"{memory_text}"
            )
        else:
            system_prompt = "你是一个智能助手，请认真回答用户的问题。"

        # 3. 构造消息列表（保留最近 3 轮）
        messages = [{"role": "system", "content": system_prompt}]
        messages += self.conversation_history[-6:]
        messages.append({"role": "user", "content": user_message})

        # 4. LLM 生成
        llm_t0 = time.perf_counter()
        assistant_reply, token_usage = self._llm(messages)
        llm_latency = round(time.perf_counter() - llm_t0, 3)

        # 5. 更新短期对话历史
        self.conversation_history.append({"role": "user", "content": user_message})
        self.conversation_history.append({"role": "assistant", "content": assistant_reply})

        # 6. 将本轮对话写入长期记忆（评测模式下可跳过，避免上下文滚雪球）
        if store_to_memory:
            conversation_chunk = f"用户: {user_message}\n助手: {assistant_reply}"
            self.add_memory(conversation_chunk, user_id=user_id, metadata={"session": session_id})

        total_latency = round(time.perf_counter() - t0, 3)

        return {
            "reply": assistant_reply,
            "retrieved_memories": retrieved_memories,
            "latency": {
                "retrieval_s": retrieval_latency,
                "llm_s": llm_latency,
                "total_s": total_latency,
            },
            "token_usage": token_usage,
        }

    def reset_session(self):
        """清空短期对话历史（不影响长期记忆）"""
        self.conversation_history.clear()

    def clear_user_memory(self, user_id: str):
        """清除某用户的全部长期记忆"""
        results = self.collection.get(where={"user_id": user_id})
        if results["ids"]:
            self.collection.delete(ids=results["ids"])
            logger.info("清除用户 %s 的 %d 条记忆", user_id, len(results["ids"]))
