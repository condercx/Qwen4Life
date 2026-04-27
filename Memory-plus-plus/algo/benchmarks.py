"""
LongMemEval 和 LoCoMo 评测入口。
"""

import time

import numpy as np

from .retrieval import MemoryPlusPlusRAG, _TEMPORAL_TYPES
from .scoring import token_f1, exact_match

# ------------------------------------------------------------------ #
#  LongMemEval
# ------------------------------------------------------------------ #

SINGLE_HOP_TYPES = {
    "single-session-user", "single-session-assistant", "single-session-preference",
}
ALL_TYPES = {
    "single-session-user", "single-session-assistant", "single-session-preference",
    "multi-session", "knowledge-update", "temporal-reasoning", "false-premise",
}


def run_longmemeval(rag: MemoryPlusPlusRAG, data: list[dict],
                    max_questions: int = 50,
                    question_types: set = None) -> dict:
    """运行 LongMemEval-S 评测。"""
    question_types = question_types or SINGLE_HOP_TYPES
    items = [d for d in data if d.get("question_type") in question_types]
    items = items[:max_questions]

    print(f"\n Memory++ LongMemEval 评测")
    print(f"  题型: {question_types & set(d['question_type'] for d in items)}")
    print(f"  总题数: {len(items)}")

    f1_scores, em_scores, latencies, token_usages = [], [], [], []
    retrieval_recall = []
    type_scores: dict = {}

    for i, item in enumerate(items):
        qtype = item.get("question_type", "?")
        question = item["question"]
        answer = str(item.get("answer", ""))
        sessions = item.get("haystack_sessions", [])
        session_ids = item.get("haystack_session_ids", [])
        session_dates = item.get("haystack_dates", [])
        question_date = item.get("question_date", "")
        evidence_ids = set(item.get("answer_session_ids", []))

        # 1. 重置并索引
        rag.reset()
        t_index = time.perf_counter()
        rag.index_sessions(sessions, session_dates=session_dates)
        index_time = time.perf_counter() - t_index

        # 2. 检索
        t0 = time.perf_counter()
        effective_k = 20 if qtype in ("multi-session", "knowledge-update") else (15 if qtype in _TEMPORAL_TYPES else 10)
        retrieved_docs, retrieved_dates, ret_conf = rag.retrieve_with_fallback(
            question, top_k=effective_k, question_type=qtype
        )
        retrieval_time = time.perf_counter() - t0

        # 检索召回率
        if evidence_ids and sessions:
            ev_text_parts = []
            for idx, sid in enumerate(session_ids):
                if sid in evidence_ids and idx < len(sessions):
                    sess = sessions[idx]
                    msgs = sess if isinstance(sess, list) else sess.get("messages", [])
                    ev_text_parts.extend(m.get("content", "") for m in msgs)
            ev_text = " ".join(ev_text_parts).lower()[:500]
            ret_text = " ".join(retrieved_docs).lower()
            stopwords = set("the a an i is was".split())
            ev_words = set(ev_text.split()[:20]) - stopwords
            if ev_words:
                overlap = len(ev_words & set(ret_text.split())) / len(ev_words)
                retrieval_recall.append(overlap)

        # 3. 生成
        t_gen = time.perf_counter()
        prediction, usage = rag.generate_answer(
            question, retrieved_docs,
            context_dates=retrieved_dates,
            question_type=qtype,
            question_date=question_date,
            retrieval_confidence=ret_conf,
        )
        gen_time = time.perf_counter() - t_gen

        total_time = index_time + retrieval_time + gen_time
        latencies.append(total_time)
        token_usages.append(usage["total_tokens"])

        f1 = token_f1(prediction, answer)
        em = exact_match(prediction, answer)
        f1_scores.append(f1)
        em_scores.append(em)

        if qtype not in type_scores:
            type_scores[qtype] = {"f1": [], "em": []}
        type_scores[qtype]["f1"].append(f1)
        type_scores[qtype]["em"].append(em)

        status = "✓" if f1 > 0.3 else "✗"
        ans_preview = prediction[:60].replace('\n', ' ') if prediction else "(empty)"
        print(
            f"  [{i + 1:3d}/{len(items)}] {status} [{qtype[:20]}] "
            f"F1={f1:.2f} EM={em:.0f} | "
            f"latency={total_time:.1f}s tokens={usage['total_tokens']}"
            f" KG={len(rag.kg_entities)}ent"
            f"\n        ans: {ans_preview}  |  gt: {str(answer)[:60]}"
        )
        if (i + 1) % 10 == 0:
            print(f"  --- progress: avg_F1={np.mean(f1_scores):.3f} ---")

    return {
        "benchmark": "LongMemEval-S (Memory++)",
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


# ------------------------------------------------------------------ #
#  LoCoMo
# ------------------------------------------------------------------ #

LOCOMO_CAT_NAMES = {1: "single-hop", 2: "multi-hop", 3: "temporal",
                    4: "open-domain", 5: "adversarial"}


def run_locomo(rag: MemoryPlusPlusRAG, data: list[dict],
               max_questions: int = 50) -> dict:
    """运行 LoCoMo 评测。"""
    all_qa = []
    for conv in data:
        c = conv.get("conversation", {})
        speaker_a = c.get("speaker_a", "A")
        sess_keys = sorted(
            [k for k in c if k.startswith("session_") and not k.endswith("_date_time")],
            key=lambda x: int(x.split("_")[1])
        )
        sessions = []
        session_dates = []
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
            session_dates.append(c.get(f"{sk}_date_time", ""))

        for qa in conv.get("qa", []):
            cat_num = qa.get("category", 0)
            answer = str(qa.get("answer", "") or qa.get("adversarial_answer", ""))
            all_qa.append({
                "question": qa.get("question", ""),
                "answer": answer,
                "category": LOCOMO_CAT_NAMES.get(cat_num, str(cat_num)),
                "sessions": sessions,
                "session_dates": session_dates,
            })

    items = all_qa[:max_questions]
    print(f"\n Memory++ LoCoMo 评测（总题数: {len(items)}）")

    f1_scores, em_scores, latencies = [], [], []
    cat_scores: dict = {}
    last_sessions_hash = None

    for i, item in enumerate(items):
        sess = item["sessions"]
        s_hash = (len(sess), sess[0].get("session_id", "") if sess else "")
        if s_hash != last_sessions_hash:
            rag.reset()
            rag.index_sessions(sess, session_dates=item.get("session_dates"))
            last_sessions_hash = s_hash

        t0 = time.perf_counter()
        cat = item["category"]
        locomo_k = 20 if cat in ("open-domain", "adversarial") else (15 if cat in ("multi-hop", "temporal") else 10)

        if cat == "multi-hop":
            docs, dates, _ret_conf = rag.retrieve_chain(
                item["question"], top_k=locomo_k, question_type=cat
            )
        else:
            docs, dates, _ret_conf = rag.retrieve_with_fallback(
                item["question"], top_k=locomo_k, question_type=cat
            )

        q_date = None
        if cat in _TEMPORAL_TYPES and item.get("session_dates"):
            for sd in reversed(item["session_dates"]):
                if sd:
                    q_date = sd
                    break

        pred, _ = rag.generate_answer(
            item["question"], docs,
            context_dates=dates,
            question_type=cat,
            question_date=q_date,
            benchmark="locomo",
            retrieval_confidence=_ret_conf,
        )
        latencies.append(time.perf_counter() - t0)

        f1 = token_f1(pred, item["answer"])
        em = exact_match(pred, item["answer"])
        f1_scores.append(f1)
        em_scores.append(em)

        if cat not in cat_scores:
            cat_scores[cat] = {"f1": [], "em": []}
        cat_scores[cat]["f1"].append(f1)
        cat_scores[cat]["em"].append(em)

        status = "✓" if f1 > 0.3 else "✗"
        pred_preview = pred[:60].replace('\n', ' ') if pred else "(empty)"
        print(
            f"  [{i + 1:3d}/{len(items)}] {status} [{cat[:12]}] "
            f"F1={f1:.2f} EM={em:.0f} latency={latencies[-1]:.1f}s"
            f"\n        ans: {pred_preview}  |  gt: {str(item['answer'])[:60]}"
        )

    return {
        "benchmark": "LoCoMo (Memory++)",
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
