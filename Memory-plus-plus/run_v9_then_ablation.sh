#!/bin/bash
# Run v9 full benchmark (latest main with all improvements), then ablation study
# Usage: bash run_v9_then_ablation.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "========== Phase 1: v9 Full Benchmark =========="
bash memory_plus.sh --max-questions 500 --all-types 2>&1 | tee benchmark_kg_v9_run.log || true

# Save v9 results
cp benchmark_results_kg.json benchmark_results_kg_v9.json
echo "v9 results saved to benchmark_results_kg_v9.json"

echo ""
echo "========== Phase 2: Ablation Study (100 questions each) =========="
bash run_ablation.sh 100 2>&1 | tee ablation_run.log

echo ""
echo "========== All Done =========="
echo "Results:"
echo "  v9: benchmark_results_kg_v9.json"
echo "  Ablation: benchmark_results_kg_ablation_*.json"
