#!/usr/bin/env python3
"""
Memory++ Benchmark Runner

Thin wrapper that delegates to the full benchmark_eval_kg.py implementation.
This file provides the entry point for the Memory++ package.

Usage:
    python -m MemoryPlusPlus.run_benchmark --max-questions 500 --question-types all
    python -m MemoryPlusPlus.run_benchmark --max-questions 100 --ablation no_kg
"""

import os
import sys
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)


def main():
    """Run the full Memory++ benchmark evaluation."""
    eval_script = os.path.join(PARENT_DIR, "benchmark_eval_kg.py")
    if not os.path.exists(eval_script):
        print(f"Error: benchmark_eval_kg.py not found at {eval_script}")
        sys.exit(1)

    cmd = [sys.executable, "-u", eval_script] + sys.argv[1:]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=PARENT_DIR)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
