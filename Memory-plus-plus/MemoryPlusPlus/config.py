"""
Memory++ Configuration
"""

# API Configuration
API_KEY = "sk-fqzbxxcldzmedxdyilolegnkvllgbmauozxdmqmslarrvjkd"
BASE_URL = "https://api.siliconflow.cn/v1"

# Model Configuration
LLM_MODEL = "Qwen/Qwen3-8B"
EMBED_MODEL = "BAAI/bge-m3"
EMBED_DIMS = 1024
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# Retrieval Configuration
DEFAULT_TOP_K = 10
CHUNK_MAX_CHARS = 2000
CONFIDENCE_THRESHOLD = 0.15  # Below this triggers query expansion
GROUNDING_THRESHOLD = 0.3    # Below this triggers IDK
PREMISE_OVERLAP_THRESHOLD = 0.3  # Below this flags false premise

# ChromaDB
CHROMA_DIR = "./chroma_bench_kg"
COLLECTION_PREFIX = "mempp"
