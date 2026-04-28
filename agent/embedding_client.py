"""Embedding 客户端。"""

from __future__ import annotations

from typing import Any, Protocol

import httpx


class EmbeddingConfig(Protocol):
    """Ollama embedding 客户端需要的最小配置。"""

    ollama_embed_url: str
    embed_model: str
    timeout_seconds: float


class EmbeddingClient(Protocol):
    """把文本批量转换为向量的最小接口。"""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """返回与输入文本一一对应的向量。"""


class OllamaEmbeddingClient:
    """通过 Ollama `/api/embed` 调用本地 bge-m3 embedding 模型。"""

    def __init__(self, config: EmbeddingConfig, transport: httpx.BaseTransport | None = None) -> None:
        self.config = config
        self.transport = transport

    def embed(self, texts: list[str]) -> list[list[float]]:
        """调用 Ollama embedding API。"""

        if not texts:
            return []

        payload = {
            "model": self.config.embed_model,
            "input": texts,
        }
        try:
            with httpx.Client(timeout=self.config.timeout_seconds, transport=self.transport) as client:
                response = client.post(self.config.ollama_embed_url, json=payload)
                response.raise_for_status()
                result = response.json()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"Ollama embedding 请求失败，HTTP {exc.response.status_code}: {exc.response.text}"
            ) from exc
        except httpx.HTTPError as exc:
            raise RuntimeError(f"Ollama embedding 网络请求失败：{exc}") from exc
        except ValueError as exc:
            raise RuntimeError("Ollama embedding 返回非 JSON 响应。") from exc

        if not isinstance(result, dict):
            raise RuntimeError(f"Ollama embedding 返回格式异常，期望 JSON object：{result}")
        embeddings = _extract_embeddings(result)
        if len(embeddings) != len(texts):
            raise RuntimeError(
                f"Ollama embedding 返回数量异常：期望 {len(texts)}，实际 {len(embeddings)}"
            )
        return embeddings


def _extract_embeddings(result: dict[str, Any]) -> list[list[float]]:
    """兼容 Ollama `/api/embed` 的 `embeddings` 字段。"""

    embeddings = result.get("embeddings")
    if embeddings is None and "embedding" in result:
        embeddings = [result["embedding"]]
    if not isinstance(embeddings, list):
        raise RuntimeError(f"Ollama embedding 返回格式异常，缺少 `embeddings` 字段：{result}")

    normalized: list[list[float]] = []
    for item in embeddings:
        if not isinstance(item, list):
            raise RuntimeError(f"Ollama embedding 向量格式异常：{result}")
        try:
            normalized.append([float(value) for value in item])
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"Ollama embedding 向量包含非数字值：{result}") from exc
    return normalized
