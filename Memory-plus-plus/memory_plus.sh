#!/usr/bin/env bash
# =============================================================
#  Memory++ 评测脚本 — KG增强版 RAG
#  - 在基线 RAG 基础上增加知识图谱实体索引 + 混合检索
#  - 自动与基线结果对比
#
#  用法：
#    bash memory_plus.sh                         # 500题全量评测
#    bash memory_plus.sh --max-questions 5       # 快速测试
#    bash memory_plus.sh --max-questions 500 --all-types  # 全题型
# =============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="mem-rag-eval"
MAX_QUESTIONS=500
EXTRA_ARGS=""

# 解析参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        --max-questions)  MAX_QUESTIONS="$2"; shift ;;
        --max-questions=*)MAX_QUESTIONS="${1#*=}" ;;
        --all-types)      EXTRA_ARGS="$EXTRA_ARGS --question-types all" ;;
        --skip-locomo)    EXTRA_ARGS="$EXTRA_ARGS --skip-locomo" ;;
    esac
    shift
done

# 清除代理环境变量
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy NO_PROXY no_proxy 2>/dev/null || true

# 找 conda 环境
ENV_DIR=""
for try_path in "/mnt/conda/envs/$ENV_NAME" "/opt/miniconda3/envs/$ENV_NAME"; do
    [ -d "$try_path" ] && ENV_DIR="$try_path" && break
done
if [ -z "$ENV_DIR" ]; then
    CONDA_BIN="/opt/miniconda3/bin/conda"
    ENV_DIR=$("$CONDA_BIN" info --envs 2>/dev/null | awk -v env="$ENV_NAME" '$1 == env {print $NF}')
fi
ENV_PYTHON="${ENV_DIR}/bin/python"

cd "$SCRIPT_DIR"

echo "============================================================"
echo "  Memory++ — KG增强 RAG 评测"
echo "  基线: ChromaDB + bge-m3 + Qwen3-8B"
echo "  增强: + 实体知识图谱 + 混合检索 + 日期感知 + 题型特化prompt"
echo "  最大题数: $MAX_QUESTIONS"
echo "============================================================"

# 确保数据集存在
bash "$SCRIPT_DIR/benchmark_download.sh"

# 清理 KG 版向量库
rm -rf "$SCRIPT_DIR/chroma_bench_kg"
echo "  已清理 chroma_bench_kg/"

# 运行 Memory++ 评测
echo ""
echo "启动 Memory++ 评测..."
env -i HOME="$HOME" PATH="${ENV_DIR}/bin:/usr/bin:/bin" \
    PYTHONPATH="" \
    "$ENV_PYTHON" -u benchmark_eval_kg.py \
    --max-questions "$MAX_QUESTIONS" \
    $EXTRA_ARGS

echo ""
echo "============================================================"
echo "  Memory++ 评测完成！"
echo "  结果: $SCRIPT_DIR/benchmark_results_kg.json"
echo "============================================================"
