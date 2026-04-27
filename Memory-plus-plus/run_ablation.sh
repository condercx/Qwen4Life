#!/bin/bash
# Memory++ 消融实验：量化各模块贡献
# 用法: bash run_ablation.sh [--max-questions N]
#
# 每次去掉一个模块，对比与full system的差异

set -e
MAX_Q="${1:-500}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate mem-rag-eval
# Find the actual env directory
ENV_DIR=$(conda info --envs | awk '$1 == "mem-rag-eval" {print $NF}')
ENV_PYTHON="${ENV_DIR}/bin/python"

# 消融配置列表: "标签 ablation_flag"
ABLATIONS=(
    "full:"
    "no_bm25:no_bm25"
    "no_kg:no_kg"
    "no_date:no_date_aware"
    "no_type:no_type_prompt"
    "no_recency:no_recency_label"
    "no_reranker:no_reranker"
    "no_query_exp:no_query_expansion"
    "no_bm25_kg:no_bm25,no_kg"
)

echo "=========================================="
echo "Memory++ Ablation Study"
echo "Max questions per benchmark: $MAX_Q"
echo "Configs: ${#ABLATIONS[@]}"
echo "=========================================="

for config in "${ABLATIONS[@]}"; do
    label="${config%%:*}"
    flag="${config#*:}"
    echo ""
    echo ">>> Running: $label (ablation=$flag)"
    echo "---"

    CMD="env -i HOME=\"$HOME\" PATH=\"${ENV_DIR}/bin:$CONDA_BASE/bin:/usr/bin:/bin\" \"$ENV_PYTHON\" -u benchmark_eval_kg.py --max-questions $MAX_Q --question-types all --skip-locomo"
    if [ -n "$flag" ]; then
        CMD="$CMD --ablation $flag"
    fi

    eval "$CMD" 2>&1 | tail -20
    echo "<<< Done: $label"
done

echo ""
echo "=========================================="
echo "Ablation complete. Results in benchmark_results_kg*.json"
echo "=========================================="
