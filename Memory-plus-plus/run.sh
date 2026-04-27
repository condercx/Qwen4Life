#!/usr/bin/env bash
# =============================================================
#  Memory RAG 系统一键运行脚本
#  - 使用独立 conda 环境 mem-rag-eval（不污染 base）
#  - Qwen3.5-4B 全程关闭思考模式（enable_thinking=False）
#
#  用法：
#    bash run.sh              # 运行玩具评测（手造数据）
#    bash run.sh --benchmark  # 下载并运行公认 Benchmark（LongMemEval + LoCoMo）
#    bash run.sh --benchmark --max-questions 100  # 限制题数
# =============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="mem-rag-eval"
CONDA_BIN="/opt/miniconda3/bin/conda"
RUN_BENCHMARK=false
MAX_QUESTIONS=50
EXTRA_ARGS=""

# 解析参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        --benchmark)      RUN_BENCHMARK=true ;;
        --max-questions)  MAX_QUESTIONS="$2"; shift ;;
        --max-questions=*)MAX_QUESTIONS="${1#*=}" ;;
        --all-types)      EXTRA_ARGS="$EXTRA_ARGS --question-types all" ;;
        --skip-locomo)    EXTRA_ARGS="$EXTRA_ARGS --skip-locomo" ;;
    esac
    shift
done

# conda 环境路径
ENV_DIR=$("$CONDA_BIN" info --envs 2>/dev/null | awk -v env="$ENV_NAME" '$1 == env {print $NF}')
if [ -z "${ENV_DIR:-}" ]; then
    for try_path in "/mnt/conda/envs/$ENV_NAME" "/opt/miniconda3/envs/$ENV_NAME"; do
        [ -d "$try_path" ] && ENV_DIR="$try_path" && break
    done
fi
ENV_PYTHON="${ENV_DIR}/bin/python"
ENV_PIP="${ENV_DIR}/bin/pip"

cd "$SCRIPT_DIR"

# 清除代理环境变量（本地代理不稳定，SiliconFlow 直连更可靠）
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy NO_PROXY no_proxy 2>/dev/null || true

echo "============================================================"
echo "  Memory RAG 系统 — SiliconFlow / Qwen3-8B"
echo "  enable_thinking=False（思考模式已关闭）"
echo "  工作目录: $SCRIPT_DIR"
if $RUN_BENCHMARK; then
    echo "  模式: Benchmark 评测（LongMemEval-S + LoCoMo）"
else
    echo "  模式: 快速评测（手造测试集）"
fi
echo "============================================================"

# ---- Step 1: 检查 conda ----
echo ""
echo "[1/5] 检查 conda..."
[ -f "$CONDA_BIN" ] || { echo "  [错误] 未找到 $CONDA_BIN"; exit 1; }
echo "  conda: $("$CONDA_BIN" --version)"

# ---- Step 2: 创建独立 conda 环境（如不存在）----
echo ""
echo "[2/5] 准备独立 conda 环境 [$ENV_NAME]..."
if [ -n "${ENV_DIR:-}" ] && [ -f "$ENV_PYTHON" ]; then
    echo "  已存在: $ENV_DIR"
else
    echo "  创建新环境 python=3.11（不影响 base）..."
    "$CONDA_BIN" create -n "$ENV_NAME" python=3.11 -y -q
    ENV_DIR=$("$CONDA_BIN" info --envs 2>/dev/null | awk -v env="$ENV_NAME" '$1 == env {print $NF}')
    ENV_PYTHON="${ENV_DIR}/bin/python"
    ENV_PIP="${ENV_DIR}/bin/pip"
    echo "  环境创建完成: $ENV_DIR"
fi
echo "  Python: $("$ENV_PYTHON" --version)"

# ---- Step 3: 安装依赖 ----
echo ""
echo "[3/5] 安装依赖至 [$ENV_NAME]..."
NO_PROXY="*" no_proxy="*" "$ENV_PIP" install --quiet openai chromadb numpy
echo "  openai=$(${ENV_PYTHON} -c 'import openai; print(openai.__version__)')  chromadb=$(${ENV_PYTHON} -c 'import chromadb; print(chromadb.__version__)')  numpy=$(${ENV_PYTHON} -c 'import numpy; print(numpy.__version__)')"

# ---- Step 4: 准备数据 ----
echo ""
echo "[4/5] 准备数据..."
if $RUN_BENCHMARK; then
    # 清理 benchmark 向量库
    rm -rf "$SCRIPT_DIR/chroma_bench"
    echo "  已清理 chroma_bench/"
    # 下载 benchmark 数据集
    bash "$SCRIPT_DIR/benchmark_download.sh"
else
    # 清理普通评测向量库
    rm -rf "$SCRIPT_DIR/chroma_db"
    echo "  已清理 chroma_db/"
fi

# ---- Step 5: 运行评测 ----
echo ""
echo "[5/5] 启动评测..."
echo "  说明：Qwen3.5-4B 思考模式已关闭（enable_thinking=False）"
echo "        SiliconFlow API 每次 LLM 调用最多等待 90s"
echo ""

if $RUN_BENCHMARK; then
    "$ENV_PYTHON" benchmark_eval.py \
        --max-questions "$MAX_QUESTIONS" \
        $EXTRA_ARGS
    echo ""
    echo "============================================================"
    echo "  Benchmark 评测完成！结果保存至:"
    echo "  $SCRIPT_DIR/benchmark_results.json"
    echo "============================================================"
else
    "$ENV_PYTHON" run_evaluation.py
    echo ""
    echo "============================================================"
    echo "  快速评测完成！结果保存至:"
    echo "  $SCRIPT_DIR/evaluation_results.json"
    echo "============================================================"
fi
