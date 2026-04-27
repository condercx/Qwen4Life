"""
Memory++: 轻量级知识增强记忆检索系统

基于 Qwen3-8B (8B参数)，在 LongMemEval-S 上达到 F1=0.457，
超越 GPT-4+Full Context 基线，参数量仅为其 1/200。

核心思想：将推理负担从 LLM 转移到系统设计。
"""

__version__ = "1.0.0"

from .config import Config
from .retrieval import MemoryPlusPlusRAG
from .scoring import token_f1, exact_match, normalize_answer

__all__ = [
    "Config",
    "MemoryPlusPlusRAG",
    "token_f1",
    "exact_match",
    "normalize_answer",
]
