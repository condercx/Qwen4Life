"""
Memory++ 配置
"""

import os


class Config:
    """集中配置，支持环境变量覆盖。"""

    # API
    API_KEY = os.environ.get(
        "SILICONFLOW_API_KEY",
        "sk-xxxxxxx",
    )
    BASE_URL = os.environ.get("SILICONFLOW_BASE_URL",
                              "https://api.siliconflow.cn/v1")

    # 模型
    LLM_MODEL = "Qwen/Qwen3-8B"
    EMBED_MODEL = "BAAI/bge-m3"
    EMBED_DIMS = 1024
    RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

    # 检索
    DEFAULT_TOP_K = 10
    CHUNK_MAX_CHARS = 2000
    CONFIDENCE_THRESHOLD = 0.15     # 低于此值触发查询扩展
    GROUNDING_THRESHOLD = 0.3       # 低于此值判定为幻觉 → IDK
    PREMISE_OVERLAP_THRESHOLD = 0.3  # 低于此值标记虚假前提

    # ChromaDB
    CHROMA_DIR = "./chroma_bench_kg"
    COLLECTION_NAME = "bench_kg"
