"""
基于公认 Benchmark 的 Memory RAG 评测
-----------------------------------------
主评测：LongMemEval-S（ICLR 2025）
  - 500 题，7 种记忆类型
  - 指标：Token-F1（无需 judge）+ LLM-as-Judge（用 Qwen3.5-4B 自评）
  - 默认取 single-hop 子集（~200题）快速评测；可改 --all 跑全部

辅助评测：LoCoMo（如数据存在）
  - single-hop + multi-hop QA
  - 指标：Token-F1、BLEU-1

用法：
  python benchmark_eval.py [--max-questions N] [--question-types TYPE1,TYPE2]
"""

import os
# 清除代理环境变量（必须在 import httpx/openai 之前）
for _k in list(os.environ):
    if 'proxy' in _k.lower() and _k != 'GOPROXY':
        del os.environ[_k]

import json
import time
import argparse
import re
import string
from collections import Counter
import numpy as np
from openai import OpenAI, APIConnectionError, APITimeoutError

# 路径配置
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "benchmark_data")
LME_S_PATH = os.path.join(DATA_DIR, "longmemeval_s.json")
LME_ORACLE_PATH = os.path.join(DATA_DIR, "longmemeval_oracle.json")
LOCOMO_PATH = os.path.join(DATA_DIR, "locomo10.json")

from config import API_KEY, BASE_URL, LLM_MODEL, EMBED_MODEL

# ------------------------------------------------------------------ #
#  Token-level F1（无需外部模型，直接字符串匹配）                       #
# ------------------------------------------------------------------ #

def normalize_answer(s) -> str:
    """小写 + 去标点 + 去冠词 + 去多余空格"""
    s = str(s).lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(ch for ch in s if ch not in string.punctuation)
    return ' '.join(s.split())

def token_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)

def exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))

# ------------------------------------------------------------------ #
#  Memory RAG 核心（与 memory_system.py 同逻辑，独立实现避免副作用）    #
# ------------------------------------------------------------------ #

import chromadb
from chromadb.config import Settings
import uuid

class BenchmarkRAG:
    def __init__(self, collection_name: str = "bench_eval"):
        self.llm_client = OpenAI(
            api_key=API_KEY, base_url=BASE_URL,
            timeout=300.0, max_retries=0
        )
        self.chroma = chromadb.PersistentClient(
            path=os.path.join(SCRIPT_DIR, "chroma_bench"),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection_name = collection_name
        self.collection = None

    def reset_collection(self):
        """每道题前重置向量库（隔离不同问题的记忆）"""
        try:
            self.chroma.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.chroma.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """批量嵌入，超出 32 条时分批，每批后等待 3s 避免限流，失败重试 5 次"""
        BATCH = 32
        result = []
        for i in range(0, len(texts), BATCH):
            batch_texts = texts[i:i+BATCH]
            for attempt in range(5):
                try:
                    resp = self.llm_client.embeddings.create(
                        model=EMBED_MODEL, input=batch_texts
                    )
                    result.extend([item.embedding for item in resp.data])
                    break
                except (APIConnectionError, APITimeoutError) as e:
                    wait = 3 * (attempt + 1)
                    print(f"    [Embed 重试 {attempt+1}/5] {type(e).__name__}，{wait}s 后重试...")
                    time.sleep(wait)
            else:
                raise RuntimeError(f"嵌入失败：batch {i//BATCH} 经 5 次重试仍失败")
            if i + BATCH < len(texts):
                time.sleep(3)
        return result

    def index_sessions(self, sessions: list):
        """将对话 sessions 写入 ChromaDB（消息对粒度，提升检索召回率）
        支持两种格式：
          - LongMemEval: list[list[{role,content}]]  (每个session是消息列表)
          - LoCoMo:      list[{session_id, messages:[{role,content}]}]
        """
        assert self.collection is not None
        chunks, ids, metas = [], [], []
        for i, sess in enumerate(sessions):
            if isinstance(sess, list):
                messages = sess
                sid = f"sess_{i}"
            else:
                sid = sess.get("session_id", f"sess_{i}")
                messages = sess.get("messages", [])

            # 按消息对（user+assistant）分块，每块 ≤ 800 字符
            for j in range(0, len(messages), 2):
                pair = messages[j:j+2]
                text = "\n".join(
                    f"{m.get('role','?')}: {m.get('content','')}"
                    for m in pair
                )
                if text.strip():
                    chunk_id = f"{sid}_c{j//2}"
                    chunks.append(text[:800])
                    ids.append(chunk_id)
                    metas.append({"session_id": sid})

        if not chunks:
            return
        embeddings = self._embed_batch(chunks)
        self.collection.add(ids=ids, embeddings=embeddings,
                            documents=chunks, metadatas=metas)

    def retrieve(self, query: str, top_k: int = 10) -> list[str]:
        """检索相关 sessions"""
        assert self.collection is not None
        count = self.collection.count()
        if count == 0:
            return []
        k = min(top_k, count)
        emb = self._embed_batch([query])[0]
        res = self.collection.query(
            query_embeddings=[emb], n_results=k,
            include=["documents"]
        )
        return res["documents"][0] if res["documents"] else []

    def generate_answer(self, question: str, context_docs: list[str],
                        max_retries: int = 5) -> tuple[str, dict]:
        """基于检索到的记忆生成回答，每次调用后等待 3s，失败最多重试 5 次"""
        if context_docs:
            ctx = "\n\n".join(f"[Memory {i+1}]\n{d}" for i, d in enumerate(context_docs))
            system = (
                "You are an assistant with long-term memory. "
                "Answer the user's question based ONLY on the conversation memories below. "
                "Give a concise, direct answer. "
                "If the memories do not contain the answer, say 'I don't know'.\n\n"
                f"Memories:\n{ctx}"
            )
        else:
            system = "Answer the question concisely. If you don't know, say 'I don't know'."

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ]

        for attempt in range(max_retries):
            try:
                resp = self.llm_client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=256,
                    extra_body={"enable_thinking": False},
                )
                content = resp.choices[0].message.content or ""
                usage = {
                    "prompt_tokens": resp.usage.prompt_tokens,
                    "completion_tokens": resp.usage.completion_tokens,
                    "total_tokens": resp.usage.total_tokens,
                }
                time.sleep(3)   # 每次成功调用后等待 3s，避免限流
                return content, usage
            except (APIConnectionError, APITimeoutError) as e:
                wait = 3 * (attempt + 1)   # 递增等待：3、6、9、12、15s
                print(f"    [重试 {attempt+1}/{max_retries}] {type(e).__name__}，{wait}s 后重试...")
                time.sleep(wait)

        return ("", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})

# ------------------------------------------------------------------ #
#  LongMemEval 评测                                                    #
# ------------------------------------------------------------------ #

SINGLE_HOP_TYPES = {
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
}

ALL_TYPES = {
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "multi-session",
    "knowledge-update",
    "temporal-reasoning",
    "false-premise",
}

def run_longmemeval(rag: BenchmarkRAG, data: list[dict],
                   max_questions: int = 50,
                   question_types: set = SINGLE_HOP_TYPES) -> dict:
    """在 LongMemEval 数据集上运行评测"""
    # 过滤题目类型
    items = [d for d in data if d.get("question_type") in question_types]
    items = items[:max_questions]

    print(f"\n LongMemEval 评测（题型={question_types & set(d['question_type'] for d in items)}）")
    print(f"  总题数（过滤后）: {len(items)}")

    f1_scores, em_scores, latencies, token_usages = [], [], [], []
    retrieval_recall = []
    type_scores: dict = {}

    for i, item in enumerate(items):
        qtype = item.get("question_type", "?")
        question = item["question"]
        answer = str(item.get("answer", ""))
        sessions = item.get("haystack_sessions", [])
        session_ids = item.get("haystack_session_ids", [])  # parallel list
        evidence_ids = set(item.get("answer_session_ids", []))

        # 1. 重置并索引此题的 haystack sessions
        rag.reset_collection()
        t_index = time.perf_counter()
        rag.index_sessions(sessions)
        index_time = time.perf_counter() - t_index

        # 2. 检索
        t0 = time.perf_counter()
        retrieved_docs = rag.retrieve(question, top_k=5)
        retrieval_time = time.perf_counter() - t0

        # 计算检索召回率（evidence session 的关键词是否出现在检索结果中）
        if evidence_ids and sessions:
            ev_text_parts: list[str] = []
            for idx, sid in enumerate(session_ids):
                if sid in evidence_ids and idx < len(sessions):
                    sess = sessions[idx]
                    msgs = sess if isinstance(sess, list) else sess.get("messages", [])
                    ev_text_parts.extend(m.get("content", "") for m in msgs)
            ev_text = " ".join(ev_text_parts).lower()[:500]
            ret_text = " ".join(retrieved_docs).lower()
            stopwords: set[str] = set("the a an i is was".split())
            ev_words: set[str] = set(ev_text.split()[:20]) - stopwords
            if ev_words:
                overlap = len(ev_words & set(ret_text.split())) / len(ev_words)
                retrieval_recall.append(overlap)

        # 3. 生成回答
        t_gen = time.perf_counter()
        prediction, usage = rag.generate_answer(question, retrieved_docs)
        gen_time = time.perf_counter() - t_gen

        total_time = index_time + retrieval_time + gen_time
        latencies.append(total_time)
        token_usages.append(usage["total_tokens"])

        # 4. 计算指标
        f1 = token_f1(prediction, answer)
        em = exact_match(prediction, answer)
        f1_scores.append(f1)
        em_scores.append(em)

        # 按题型汇总
        if qtype not in type_scores:
            type_scores[qtype] = {"f1": [], "em": []}
        type_scores[qtype]["f1"].append(f1)
        type_scores[qtype]["em"].append(em)

        status = "✓" if f1 > 0.3 else "✗"
        ans_preview = prediction[:60].replace('\n', ' ') if prediction else "(empty)"
        answer_str = str(answer)
        print(
            f"  [{i+1:3d}/{len(items)}] {status} [{qtype[:20]}] "
            f"F1={f1:.2f} EM={em:.0f} | "
            f"latency={total_time:.1f}s tokens={usage['total_tokens']}"
            f"\n        答: {ans_preview}  |  真: {answer_str[:60]}"
        )
        if (i + 1) % 10 == 0:
            print(f"  --- 进度: avg_F1={np.mean(f1_scores):.3f} ---")

    # 汇总
    result = {
        "benchmark": "LongMemEval-S",
        "n_questions": len(items),
        "question_types": list(question_types & set(d["question_type"] for d in items)),
        "overall": {
            "token_f1_mean": round(float(np.mean(f1_scores)), 4),
            "token_f1_p50": round(float(np.median(f1_scores)), 4),
            "exact_match": round(float(np.mean(em_scores)), 4),
            "retrieval_recall_mean": round(float(np.mean(retrieval_recall)), 4) if retrieval_recall else None,
            "e2e_latency_p50_s": round(float(np.percentile(latencies, 50)), 2),
            "e2e_latency_p95_s": round(float(np.percentile(latencies, 95)), 2),
            "token_usage_mean": round(float(np.mean(token_usages)), 1),
        },
        "by_type": {
            qtype: {
                "n": len(scores["f1"]),
                "token_f1": round(float(np.mean(scores["f1"])), 4),
                "exact_match": round(float(np.mean(scores["em"])), 4),
            }
            for qtype, scores in type_scores.items()
        },
    }
    return result

# ------------------------------------------------------------------ #
#  LoCoMo 评测（single-hop + multi-hop）                              #
# ------------------------------------------------------------------ #

LOCOMO_CAT_NAMES = {1: "single-hop", 2: "multi-hop", 3: "temporal",
                    4: "open-domain", 5: "adversarial"}

def run_locomo(rag: BenchmarkRAG, data: list[dict],
               max_questions: int = 50) -> dict:
    """在 LoCoMo 数据集上运行评测
    LoCoMo 格式：conversation 下有 session_1, session_2, ... 字段，
    每个 session 是 list[{speaker, dia_id, text}]
    """
    all_qa = []
    for conv in data:
        c = conv.get("conversation", {})
        speaker_a = c.get("speaker_a", "A")
        # 收集所有 session_N 字段（跳过 session_N_date_time）
        sess_keys = sorted(
            [k for k in c if k.startswith("session_") and not k.endswith("_date_time")],
            key=lambda x: int(x.split("_")[1])
        )
        sessions = []
        for sk in sess_keys:
            turns = c[sk]
            msgs = [
                {
                    "role": "user" if t.get("speaker") == speaker_a else "assistant",
                    "content": t.get("text", ""),
                }
                for t in turns
            ]
            sessions.append({"session_id": sk, "messages": msgs})

        for qa in conv.get("qa", []):
            cat_num = qa.get("category", 0)
            all_qa.append({
                "question": qa.get("question", ""),
                "answer": qa.get("answer", "") or qa.get("adversarial_answer", ""),
                "category": LOCOMO_CAT_NAMES.get(cat_num, str(cat_num)),
                "sessions": sessions,
            })

    items = all_qa[:max_questions]
    print(f"\n LoCoMo 评测（总题数: {len(items)}）")

    f1_scores, em_scores, latencies = [], [], []
    cat_scores: dict = {}

    for i, item in enumerate(items):
        rag.reset_collection()
        rag.index_sessions(item["sessions"])

        t0 = time.perf_counter()
        docs = rag.retrieve(item["question"], top_k=5)
        pred, _ = rag.generate_answer(item["question"], docs)
        latencies.append(time.perf_counter() - t0)

        f1 = token_f1(pred, item["answer"])
        em = exact_match(pred, item["answer"])
        f1_scores.append(f1)
        em_scores.append(em)

        cat = item["category"]
        if cat not in cat_scores:
            cat_scores[cat] = {"f1": [], "em": []}
        cat_scores[cat]["f1"].append(f1)
        cat_scores[cat]["em"].append(em)

        print(
            f"  [{i+1:3d}/{len(items)}] [{cat[:12]}] "
            f"F1={f1:.2f} EM={em:.0f} latency={latencies[-1]:.1f}s"
        )

    return {
        "benchmark": "LoCoMo",
        "n_questions": len(items),
        "overall": {
            "token_f1_mean": round(float(np.mean(f1_scores)), 4),
            "exact_match": round(float(np.mean(em_scores)), 4),
            "e2e_latency_p50_s": round(float(np.percentile(latencies, 50)), 2),
        },
        "by_category": {
            cat: {
                "n": len(v["f1"]),
                "token_f1": round(float(np.mean(v["f1"])), 4),
                "exact_match": round(float(np.mean(v["em"])), 4),
            }
            for cat, v in cat_scores.items()
        },
    }

# ------------------------------------------------------------------ #
#  主程序                                                              #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description="Memory RAG Benchmark Evaluation")
    parser.add_argument("--max-questions", type=int, default=50,
                        help="每个 benchmark 最多评测的题目数（默认50）")
    parser.add_argument("--question-types", type=str,
                        default="single-session-user,single-session-assistant,single-session-preference",
                        help="LongMemEval 题型过滤（逗号分隔），all=全部类型")
    parser.add_argument("--skip-locomo", action="store_true",
                        help="跳过 LoCoMo 评测")
    args = parser.parse_args()

    if args.question_types == "all":
        qtypes = ALL_TYPES
    else:
        qtypes = set(args.question_types.split(","))

    print("=" * 60)
    print("Memory RAG Benchmark 评测")
    print(f"  模型: {LLM_MODEL}  |  enable_thinking=False")
    print(f"  嵌入: {EMBED_MODEL}")
    print(f"  最大题数: {args.max_questions}")
    print("=" * 60)

    rag = BenchmarkRAG()
    all_results = {}

    # ---- LongMemEval ----
    if not os.path.exists(LME_S_PATH):
        print(f"\n[警告] 未找到 LongMemEval 数据集: {LME_S_PATH}")
        print("  请先运行: bash benchmark_download.sh")
    else:
        print(f"\n加载 LongMemEval-S: {LME_S_PATH}")
        with open(LME_S_PATH, encoding="utf-8") as f:
            lme_data = json.load(f)
        print(f"  总题数: {len(lme_data)}")

        lme_result = run_longmemeval(rag, lme_data,
                                     max_questions=args.max_questions,
                                     question_types=qtypes)
        all_results["longmemeval"] = lme_result

        print("\n  LongMemEval 结果摘要：")
        o = lme_result["overall"]
        print(f"    Token-F1:       {o['token_f1_mean']:.4f}")
        print(f"    Exact Match:    {o['exact_match']:.4f}")
        if o.get("retrieval_recall_mean"):
            print(f"    Retrieval Recall:{o['retrieval_recall_mean']:.4f}")
        print(f"    E2E 延迟 p50:   {o['e2e_latency_p50_s']}s")
        print(f"    Token 均值:     {o['token_usage_mean']}")

    # ---- LoCoMo ----
    if not args.skip_locomo and os.path.exists(LOCOMO_PATH):
        print(f"\n加载 LoCoMo: {LOCOMO_PATH}")
        with open(LOCOMO_PATH, encoding="utf-8") as f:
            locomo_data = json.load(f)
        if isinstance(locomo_data, dict):
            locomo_data = list(locomo_data.values())

        locomo_result = run_locomo(rag, locomo_data, max_questions=args.max_questions)
        all_results["locomo"] = locomo_result

        print("\n  LoCoMo 结果摘要：")
        o = locomo_result["overall"]
        print(f"    Token-F1:  {o['token_f1_mean']:.4f}")
        print(f"    Exact Match: {o['exact_match']:.4f}")
    elif not args.skip_locomo:
        print(f"\n[跳过] LoCoMo 数据未找到: {LOCOMO_PATH}")

    # 保存结果
    out_path = os.path.join(SCRIPT_DIR, "benchmark_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n原始结果保存至: {out_path}")

    print("\n" + "=" * 60)
    print("评测完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
