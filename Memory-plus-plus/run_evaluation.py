"""
Memory RAG 评测主入口
用法：python run_evaluation.py
"""

import json
import time
import sys

def print_banner():
    print("""
╔══════════════════════════════════════════════════════╗
║     Memory RAG 系统 — SiliconFlow Qwen/Qwen3.5-4B   ║
║     基于 Mem0 框架 + BAAI/bge-m3 嵌入模型            ║
╚══════════════════════════════════════════════════════╝
""")

def print_metrics_report(metrics: dict):
    print("\n" + "=" * 60)
    print("评测结果汇总")
    print("=" * 60)

    # Memory 写入
    ms = metrics.get("memory_setup", {})
    print(f"\n📦 Memory 写入")
    print(f"  写入轮次:     {ms.get('total_turns_stored', '-')}")
    print(f"  提取事实数:   {ms.get('total_facts_extracted', '-')}")
    print(f"  写入延迟 p50: {ms.get('write_latency_p50_s', '-')} s")
    print(f"  写入延迟 p95: {ms.get('write_latency_p95_s', '-')} s")

    # 检索
    r = metrics.get("retrieval", {})
    top_k = 5
    print(f"\n🔍 记忆检索 (Top-{top_k})")
    print(f"  Precision@{top_k}: {r.get(f'retrieval_precision@{top_k}', '-')}")
    print(f"  Recall@{top_k}:    {r.get(f'retrieval_recall@{top_k}', '-')}")
    print(f"  F1@{top_k}:        {r.get(f'retrieval_f1@{top_k}', '-')}")
    print(f"  检索延迟 p50: {r.get('retrieval_latency_p50_s', '-')} s")
    print(f"  检索延迟 p95: {r.get('retrieval_latency_p95_s', '-')} s")

    # 回答质量
    aq = metrics.get("answer_quality", {})
    print(f"\n💬 端到端回答质量")
    print(f"  关键词命中率: {aq.get('answer_keyword_hit_rate', '-')}")
    print(f"  E2E 延迟 p50: {aq.get('e2e_latency_p50_s', '-')} s")
    print(f"  E2E 延迟 p95: {aq.get('e2e_latency_p95_s', '-')} s")
    print(f"  Token 均值:   {aq.get('token_usage_mean', '-')}")
    print(f"  Token p95:    {aq.get('token_usage_p95', '-')}")

    # 基线对比
    bl = metrics.get("full_context_baseline", {})
    print(f"\n📊 Full-Context 基线对比")
    print(f"  基线延迟 p50:  {bl.get('baseline_e2e_latency_p50_s', '-')} s")
    print(f"  基线 Token 均: {bl.get('baseline_token_usage_mean', '-')}")

    # 效率提升
    eff = metrics.get("efficiency", {})
    if eff:
        print(f"\n⚡ 效率提升（Memory RAG vs Full-Context）")
        print(f"  Token 减少:    {eff.get('token_reduction_pct', '-')}%")
        print(f"  延迟加速:      {eff.get('latency_speedup_x', '-')}x")

    # LLM Judge
    lj = metrics.get("llm_judge", {})
    print(f"\n🏆 LLM-as-Judge 评分（满分10）")
    print(f"  均值:  {lj.get('llm_judge_mean_score', '-')}")
    print(f"  最低:  {lj.get('llm_judge_min_score', '-')}")
    print(f"  最高:  {lj.get('llm_judge_max_score', '-')}")

    print("\n" + "=" * 60)


def main():
    print_banner()

    # 导入放在这里，方便看清楚依赖缺失的错误
    try:
        from memory_system import MemoryRAGSystem
        from evaluation import MemoryRAGEvaluator
    except ImportError as e:
        print(f"[错误] 依赖缺失: {e}")
        print("请先运行: pip install -r requirements.txt")
        sys.exit(1)

    # 初始化系统
    system = MemoryRAGSystem()
    evaluator = MemoryRAGEvaluator(system)

    # 运行完整评测
    start = time.time()
    metrics = evaluator.run_all()
    elapsed = time.time() - start

    # 输出报告
    print_metrics_report(metrics)
    print(f"\n总评测耗时: {elapsed:.1f}s")

    # 保存原始指标 JSON
    output_path = "evaluation_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"原始指标已保存至: {output_path}")


if __name__ == "__main__":
    main()
