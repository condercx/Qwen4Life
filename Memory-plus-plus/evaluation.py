"""
Memory RAG 评测模块（优化版）
- 批量嵌入，减少 API 调用
- 5 轮 QA 评测 + Full-Context 基线对比 + LLM-as-Judge
"""

import json
import time
import numpy as np
from openai import OpenAI, APIConnectionError, APITimeoutError
from memory_system import MemoryRAGSystem
from config import API_KEY, BASE_URL, LLM_MODEL

# ------------------------------------------------------------------ #
#  测试数据集                                                          #
# ------------------------------------------------------------------ #

TEST_FACTS = [
    "我叫张伟，今年28岁，是一名后端工程师，主要用Python和Go开发。",
    "我最近在学习大模型相关技术，特别是RAG和Agent。",
    "我住在北京，喜欢打篮球和看科幻小说。",
    "我的目标是今年内转型做AI工程师。",
    "我更喜欢远程办公，觉得效率更高。",
    "我用VS Code编辑器，配合GitHub Copilot写代码。",
    "我对微服务架构比较熟悉，做过电商和金融方向的项目。",
]

QA_TEST_SET = [
    {
        "question": "我是做什么工作的？用什么编程语言？",
        "expected_keywords": ["后端工程师", "Python", "Go"],
        "required_memory_keywords": ["后端工程师", "Python"],
    },
    {
        "question": "我有什么兴趣爱好？",
        "expected_keywords": ["篮球", "科幻"],
        "required_memory_keywords": ["篮球", "科幻"],
    },
    {
        "question": "我的职业目标是什么？",
        "expected_keywords": ["AI工程师", "转型"],
        "required_memory_keywords": ["AI工程师"],
    },
    {
        "question": "我喜欢什么工作方式？用什么开发工具？",
        "expected_keywords": ["远程", "VS Code"],
        "required_memory_keywords": ["远程", "VS Code"],
    },
    {
        "question": "我住在哪里？叫什么名字？",
        "expected_keywords": ["北京", "张伟"],
        "required_memory_keywords": ["北京", "张伟"],
    },
]

FULL_CONTEXT = "\n".join(TEST_FACTS)


class MemoryRAGEvaluator:
    def __init__(self, system: MemoryRAGSystem):
        self.system = system
        self.client = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=120.0, max_retries=0)
        self.user_id = "eval_user_001"
        self.metrics: dict = {}

    def _llm_call(self, messages: list[dict], max_tokens: int = 256) -> tuple[str, dict]:
        """封装 LLM 调用，关闭 thinking 模式"""
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

    # ------------------------------------------------------------------ #
    #  Step 1: 构建测试记忆库（批量嵌入）                                  #
    # ------------------------------------------------------------------ #

    def setup_memory(self) -> dict:
        print("\n[1/4] 构建测试记忆库（批量嵌入）...")
        t0 = time.perf_counter()
        # 批量嵌入所有事实
        embeddings = self.system._embed(TEST_FACTS)
        embed_latency = time.perf_counter() - t0

        import uuid
        ids = [str(uuid.uuid4()) for _ in TEST_FACTS]
        metas = [{"user_id": self.user_id, "text": f[:200]} for f in TEST_FACTS]
        self.system.collection.add(
            ids=ids, embeddings=embeddings, documents=TEST_FACTS, metadatas=metas
        )
        total_latency = time.perf_counter() - t0

        stats = {
            "total_facts_stored": len(TEST_FACTS),
            "batch_embed_latency_s": round(embed_latency, 3),
            "total_setup_latency_s": round(total_latency, 3),
            "avg_fact_embed_latency_s": round(embed_latency / len(TEST_FACTS), 3),
        }
        print(f"  存入事实数:     {stats['total_facts_stored']}")
        print(f"  批量嵌入耗时:   {stats['batch_embed_latency_s']}s")
        print(f"  单条平均嵌入:   {stats['avg_fact_embed_latency_s']}s")
        return stats

    # ------------------------------------------------------------------ #
    #  Step 2: 检索评测（Precision / Recall / F1 @ K）                    #
    # ------------------------------------------------------------------ #

    def evaluate_retrieval(self, top_k: int = 5) -> dict:
        print(f"\n[2/4] 评测记忆检索 (Top-{top_k})...")
        precision_list, recall_list, f1_list, latencies = [], [], [], []

        # 批量嵌入所有查询
        questions = [qa["question"] for qa in QA_TEST_SET]
        t_embed = time.perf_counter()
        q_embeddings = self.system._embed(questions)
        embed_latency = time.perf_counter() - t_embed

        for i, qa in enumerate(QA_TEST_SET):
            t0 = time.perf_counter()
            count = self.system.collection.count()
            actual_k = min(top_k, max(1, count))
            results = self.system.collection.query(
                query_embeddings=[q_embeddings[i]],
                n_results=actual_k,
                where={"user_id": self.user_id},
                include=["documents", "distances"],
            )
            latencies.append(time.perf_counter() - t0)

            retrieved_text = " ".join(
                results["documents"][0] if results["documents"] else []
            ).lower()

            required = qa["required_memory_keywords"]
            hits = [kw for kw in required if kw.lower() in retrieved_text]

            n_retrieved = len(results["documents"][0]) if results["documents"] else 0
            precision = len(hits) / n_retrieved if n_retrieved > 0 else 0
            recall = len(hits) / len(required) if required else 0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)
            print(
                f"  Q{i+1}: {qa['question'][:28]}... | "
                f"hits={len(hits)}/{len(required)} P={precision:.2f} R={recall:.2f} F1={f1:.2f}"
            )

        stats = {
            f"precision@{top_k}": round(float(np.mean(precision_list)), 4),
            f"recall@{top_k}": round(float(np.mean(recall_list)), 4),
            f"f1@{top_k}": round(float(np.mean(f1_list)), 4),
            "batch_query_embed_latency_s": round(embed_latency, 3),
            "search_latency_p50_s": round(float(np.percentile(latencies, 50)), 4),
            "search_latency_p95_s": round(float(np.percentile(latencies, 95)), 4),
            "search_latency_mean_s": round(float(np.mean(latencies)), 4),
        }
        print(f"  → Precision@{top_k}={stats[f'precision@{top_k}']}  Recall@{top_k}={stats[f'recall@{top_k}']}  F1@{top_k}={stats[f'f1@{top_k}']}")
        return stats

    # ------------------------------------------------------------------ #
    #  Step 3: 端到端 RAG 对话评测                                         #
    # ------------------------------------------------------------------ #

    def evaluate_rag_chat(self) -> dict:
        print("\n[3/4] 端到端 RAG 对话评测...")
        keyword_hits, latencies, token_usages = [], [], []

        for i, qa in enumerate(QA_TEST_SET):
            try:
                self.system.reset_session()  # 每题重置对话历史，避免 token 累积
                result = self.system.chat(qa["question"], user_id=self.user_id,
                                          session_id="eval", store_to_memory=False)
                reply = result["reply"].lower()
                latencies.append(result["latency"]["total_s"])
                token_usages.append(result["token_usage"]["total_tokens"])

                expected = qa["expected_keywords"]
                hits = [kw for kw in expected if kw.lower() in reply]
                kw_rate = len(hits) / len(expected) if expected else 0
                keyword_hits.append(kw_rate)

                n_mem = len(result["retrieved_memories"])
                print(
                    f"  Q{i+1}: {qa['question'][:28]}... | "
                    f"kw_hits={len(hits)}/{len(expected)} mem_retrieved={n_mem} "
                    f"latency={result['latency']['total_s']}s tokens={result['token_usage']['total_tokens']}"
                )
            except Exception as e:
                print(f"  Q{i+1}: 调用失败 ({type(e).__name__}), 跳过")
                keyword_hits.append(0.0)
                latencies.append(999.0)
                token_usages.append(0)

        stats = {
            "keyword_hit_rate": round(float(np.mean(keyword_hits)), 4),
            "e2e_latency_p50_s": round(float(np.percentile(latencies, 50)), 3),
            "e2e_latency_p95_s": round(float(np.percentile(latencies, 95)), 3),
            "e2e_latency_mean_s": round(float(np.mean(latencies)), 3),
            "token_usage_mean": round(float(np.mean(token_usages)), 1),
            "token_usage_p95": round(float(np.percentile(token_usages, 95)), 1),
        }
        return stats

    # ------------------------------------------------------------------ #
    #  Step 4: Full-Context 基线 + LLM-as-Judge（合并减少调用）            #
    # ------------------------------------------------------------------ #

    def evaluate_baseline_and_judge(self) -> dict:
        print("\n[4/4] Full-Context 基线 + LLM-as-Judge 评测...")
        base_latencies, base_tokens = [], []
        judge_scores = []

        for i, qa in enumerate(QA_TEST_SET):
            try:
                # --- 基线：Full-Context ---
                t0 = time.perf_counter()
                _, base_usage = self._llm_call([
                    {"role": "system", "content": f"以下是用户的完整历史信息：\n{FULL_CONTEXT}"},
                    {"role": "user", "content": qa["question"]},
                ])
                base_latencies.append(time.perf_counter() - t0)
                base_tokens.append(base_usage["total_tokens"])

                # --- LLM-as-Judge ---
                rag_result = self.system.chat(qa["question"], user_id=self.user_id + "_judge",
                                              store_to_memory=False)
                rag_reply = rag_result["reply"]

                judge_content, _ = self._llm_call([
                    {"role": "user", "content": (
                        f"问题：{qa['question']}\n"
                        f"期望关键词：{'、'.join(qa['expected_keywords'])}\n"
                        f"回答：{rag_reply}\n\n"
                        f"包含期望信息的程度打分0-10，只输出数字："
                    )}
                ], max_tokens=10)

                try:
                    score = float(judge_content.strip().split()[0])
                    score = max(0.0, min(10.0, score))
                except (ValueError, IndexError):
                    score = 5.0
                judge_scores.append(score)

                print(
                    f"  Q{i+1}: base_latency={base_latencies[-1]:.2f}s "
                    f"base_tokens={base_tokens[-1]} judge_score={score:.1f}"
                )
            except Exception as e:
                print(f"  Q{i+1}: 调用失败 ({type(e).__name__}), 跳过")
                base_latencies.append(999.0)
                base_tokens.append(0)
                judge_scores.append(5.0)

        return {
            "baseline_latency_p50_s": round(float(np.percentile(base_latencies, 50)), 3),
            "baseline_latency_p95_s": round(float(np.percentile(base_latencies, 95)), 3),
            "baseline_token_mean": round(float(np.mean(base_tokens)), 1),
            "llm_judge_mean": round(float(np.mean(judge_scores)), 2),
            "llm_judge_min": round(float(np.min(judge_scores)), 2),
            "llm_judge_max": round(float(np.max(judge_scores)), 2),
        }

    # ------------------------------------------------------------------ #
    #  汇总                                                                #
    # ------------------------------------------------------------------ #

    def run_all(self) -> dict:
        print("=" * 60)
        print("Memory RAG 系统评测开始")
        print("=" * 60)

        all_metrics: dict = {}
        all_metrics["memory_setup"] = self.setup_memory()
        all_metrics["retrieval"] = self.evaluate_retrieval()
        all_metrics["rag_chat"] = self.evaluate_rag_chat()
        all_metrics["baseline_and_judge"] = self.evaluate_baseline_and_judge()

        # 效率对比
        rag_tokens = all_metrics["rag_chat"]["token_usage_mean"]
        base_tokens = all_metrics["baseline_and_judge"]["baseline_token_mean"]
        rag_lat = all_metrics["rag_chat"]["e2e_latency_mean_s"]
        base_lat = all_metrics["baseline_and_judge"]["baseline_latency_p50_s"]
        if base_tokens > 0 and rag_tokens > 0:
            all_metrics["efficiency"] = {
                "token_reduction_pct": round((1 - rag_tokens / base_tokens) * 100, 1),
                "latency_change_pct": round((rag_lat / base_lat - 1) * 100, 1) if base_lat > 0 else None,
            }

        self.metrics = all_metrics
        return all_metrics
