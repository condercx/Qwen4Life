#!/usr/bin/env bash
# 下载 LongMemEval 和 LoCoMo-MC 数据集
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/benchmark_data"
mkdir -p "$DATA_DIR"

echo "============================================================"
echo "  下载 Benchmark 数据集"
echo "============================================================"

# ---- LongMemEval-S（500题，单/多跳记忆 QA）----
echo ""
echo "[1/2] 下载 LongMemEval-S ..."
LME_S="$DATA_DIR/longmemeval_s.json"
LME_ORACLE="$DATA_DIR/longmemeval_oracle.json"

if [ -f "$LME_S" ]; then
    echo "  已存在: $LME_S"
else
    NO_PROXY="*" no_proxy="*" wget -q --show-progress \
        -O "$LME_S" \
        "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json" \
        && echo "  下载完成: $LME_S" \
        || echo "  [警告] 下载失败，请手动下载到 $LME_S"
fi

if [ -f "$LME_ORACLE" ]; then
    echo "  已存在: $LME_ORACLE"
else
    NO_PROXY="*" no_proxy="*" wget -q --show-progress \
        -O "$LME_ORACLE" \
        "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json" \
        && echo "  下载完成: $LME_ORACLE" \
        || echo "  [警告] 下载失败，请手动下载到 $LME_ORACLE"
fi

# ---- LoCoMo-MC（多选题，无需 judge 模型）----
echo ""
echo "[2/2] 克隆 LoCoMo 仓库（取 locomo10.json）..."
LOCOMO_DIR="$DATA_DIR/locomo"
LOCOMO_FILE="$DATA_DIR/locomo10.json"

if [ -f "$LOCOMO_FILE" ]; then
    echo "  已存在: $LOCOMO_FILE"
else
    if [ ! -d "$LOCOMO_DIR" ]; then
        NO_PROXY="*" no_proxy="*" git clone --depth 1 \
            https://github.com/snap-research/locomo.git "$LOCOMO_DIR" 2>&1 | tail -3 \
            || echo "  [警告] git clone 失败，尝试 wget..."
    fi
    if [ -f "$LOCOMO_DIR/data/locomo10.json" ]; then
        cp "$LOCOMO_DIR/data/locomo10.json" "$LOCOMO_FILE"
        echo "  已复制: $LOCOMO_FILE"
    else
        echo "  [警告] LoCoMo 数据获取失败，仅使用 LongMemEval"
    fi
fi

echo ""
echo "数据集准备完成："
ls -lh "$DATA_DIR"/*.json 2>/dev/null || echo "  无 JSON 文件"
