"""
Memory++ CLI 入口。

Usage:
    python -m algo --max-questions 500 --question-types all
    python -m algo --max-questions 100 --ablation no_kg
    python -m algo --max-questions 50 --skip-locomo
"""

import os
import json
import argparse

# 清除代理环境变量
for _k in list(os.environ):
    if 'proxy' in _k.lower() and _k != 'GOPROXY':
        del os.environ[_k]

from .config import Config
from .retrieval import MemoryPlusPlusRAG
from .benchmarks import run_longmemeval, run_locomo, ALL_TYPES


def main():
    parser = argparse.ArgumentParser(description="Memory++ Benchmark Evaluation")
    parser.add_argument("--max-questions", type=int, default=50)
    parser.add_argument("--question-types", type=str,
                        default="single-session-user,single-session-assistant,single-session-preference")
    parser.add_argument("--skip-locomo", action="store_true")
    parser.add_argument("--ablation", type=str, default="",
                        help="Comma-separated ablation flags: no_bm25,no_kg,no_date_aware,no_type_prompt,no_reranker,no_query_expansion")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Directory containing benchmark data files")
    args = parser.parse_args()

    if args.question_types == "all":
        qtypes = ALL_TYPES
    else:
        qtypes = set(args.question_types.split(","))

    # 数据路径
    data_dir = args.data_dir or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmark_data")
    lme_path = os.path.join(data_dir, "longmemeval_s.json")
    locomo_path = os.path.join(data_dir, "locomo10.json")

    print("=" * 60)
    print("Memory++ Benchmark 评测")
    cfg = Config()
    print(f"  模型: {cfg.LLM_MODEL}  |  enable_thinking=False")
    print(f"  嵌入: {cfg.EMBED_MODEL}")
    print(f"  最大题数: {args.max_questions}")
    if args.ablation:
        print(f"  消融: {args.ablation}")
    print("=" * 60)

    rag = MemoryPlusPlusRAG(config=cfg, ablation=args.ablation)
    all_results = {}

    # LongMemEval
    if os.path.exists(lme_path):
        print(f"\n加载 LongMemEval-S: {lme_path}")
        with open(lme_path, encoding="utf-8") as f:
            lme_data = json.load(f)
        print(f"  总题数: {len(lme_data)}")

        lme_result = run_longmemeval(rag, lme_data,
                                     max_questions=args.max_questions,
                                     question_types=qtypes)
        all_results["longmemeval"] = lme_result

        print("\n  LongMemEval 结果:")
        o = lme_result["overall"]
        print(f"    Token-F1:  {o['token_f1_mean']:.4f}")
        print(f"    EM:        {o['exact_match']:.4f}")
        if o.get("retrieval_recall_mean"):
            print(f"    Recall:    {o['retrieval_recall_mean']:.4f}")

    # LoCoMo
    if not args.skip_locomo and os.path.exists(locomo_path):
        print(f"\n加载 LoCoMo: {locomo_path}")
        with open(locomo_path, encoding="utf-8") as f:
            locomo_data = json.load(f)
        if isinstance(locomo_data, dict):
            locomo_data = list(locomo_data.values())

        locomo_result = run_locomo(rag, locomo_data, max_questions=args.max_questions)
        all_results["locomo"] = locomo_result

        print("\n  LoCoMo 结果:")
        o = locomo_result["overall"]
        print(f"    Token-F1:  {o['token_f1_mean']:.4f}")
        print(f"    EM:        {o['exact_match']:.4f}")

    # 保存
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ablation_suffix = f"_ablation_{args.ablation.replace(',', '_')}" if args.ablation else ""
    out_path = os.path.join(script_dir, f"benchmark_results_kg{ablation_suffix}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n结果保存至: {out_path}")

    # 基线对比
    baseline_path = os.path.join(script_dir, "benchmark_results.json")
    if os.path.exists(baseline_path):
        with open(baseline_path) as f:
            baseline = json.load(f)
        print("\n" + "=" * 60)
        print("  Memory++ vs Baseline")
        print("=" * 60)
        if "longmemeval" in baseline and "longmemeval" in all_results:
            b = baseline["longmemeval"]["overall"]
            n = all_results["longmemeval"]["overall"]
            delta_f1 = n["token_f1_mean"] - b["token_f1_mean"]
            print(f"\n  LongMemEval-S:")
            print(f"    Token-F1:  {b['token_f1_mean']:.4f} → {n['token_f1_mean']:.4f}  ({delta_f1:+.4f})")

    print("\n" + "=" * 60)
    print("评测完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
