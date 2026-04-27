"""
配置文件：Mem0 + SiliconFlow Qwen/Qwen3.5-4B
"""

API_KEY = "sk-fqzbxxcldzmedxdyilolegnkvllgbmauozxdmqmslarrvjkd"
BASE_URL = "https://api.siliconflow.cn/v1"
LLM_MODEL = "Qwen/Qwen3-8B"           # SiliconFlow 上的对话模型（Qwen3.5-4B 宕机，临时切换）
EMBED_MODEL = "BAAI/bge-m3"           # SiliconFlow 上的嵌入模型
EMBED_DIMS = 1024                       # bge-m3 的向量维度

# Mem0 完整配置
MEM0_CONFIG = {
    "llm": {
        "provider": "openai",
        "config": {
            "model": LLM_MODEL,
            "openai_base_url": BASE_URL,
            "api_key": API_KEY,
            "temperature": 0.1,
            "max_tokens": 2000,
        },
    },
    "embedder": {
        "provider": "openai",
        "config": {
            "model": EMBED_MODEL,
            "openai_base_url": BASE_URL,
            "api_key": API_KEY,
            "embedding_dims": EMBED_DIMS,
        },
    },
    "vector_store": {
        "provider": "chroma",
        "config": {
            "collection_name": "mem0_qwen_memory",
            "path": "./chroma_db",
        },
    },
    "version": "v1.1",
}
