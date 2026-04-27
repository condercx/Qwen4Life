"""长期记忆存储实现。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
import math
from pathlib import Path
import re
from typing import Any, Protocol
import uuid

from agent.memory_client import EmbeddingClient, OllamaEmbeddingClient
from agent.memory_config import MemoryConfig


@dataclass(slots=True)
class MemoryItem:
    """一条检索到的长期记忆。"""

    memory_id: str
    text: str
    score: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


class MemoryStore(Protocol):
    """长期记忆存储的最小接口。"""

    def add_memory(
        self,
        *,
        user_id: str,
        session_id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """写入一条长期记忆并返回记忆 ID。"""

    def search_memory(
        self,
        *,
        user_id: str,
        query: str,
        top_k: int,
        min_score: float,
    ) -> list[MemoryItem]:
        """检索与 query 相关的长期记忆。"""

    def get_all_memories(self, user_id: str) -> list[MemoryItem]:
        """返回用户的全部长期记忆。"""

    def delete_memory(self, user_id: str, memory_id: str) -> bool:
        """删除用户的一条长期记忆，返回是否删除成功。"""

    def clear_user_memory(self, user_id: str) -> None:
        """清除用户的全部长期记忆。"""


class ChromaMemoryStore:
    """基于 ChromaDB 持久化存储长期记忆。"""

    def __init__(
        self,
        config: MemoryConfig | None = None,
        embedding_client: EmbeddingClient | None = None,
    ) -> None:
        self.config = config or MemoryConfig.from_env()
        self.embedding_client = embedding_client or OllamaEmbeddingClient(self.config)
        chroma_path = Path(self.config.chroma_path)
        if chroma_path.is_absolute():
            raise RuntimeError("AGENT_MEMORY_CHROMA_PATH 必须是相对路径，便于不同机器测试。")

        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError as exc:
            raise RuntimeError("缺少 chromadb 依赖，请先安装 requirements.txt。") from exc

        self.client = chromadb.PersistentClient(
            path=self.config.chroma_path,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=self.config.collection,
            metadata={"hnsw:space": "cosine"},
        )

    def add_memory(
        self,
        *,
        user_id: str,
        session_id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """写入一条向量化后的长期记忆。"""

        normalized_text = text.strip()
        if not normalized_text:
            raise ValueError("memory text 不能为空")

        doc_id = str(uuid.uuid4())
        metadata = dict(metadata or {})
        metadata["memory_id"] = doc_id
        metadata.setdefault("kg_entities", _serialize_values(_extract_entities(normalized_text)))
        metadata.setdefault("kg_relations", _serialize_values(_extract_relation_hints(normalized_text)))
        embedding = self.embedding_client.embed([normalized_text])[0]
        self.collection.add(
            ids=[doc_id],
            documents=[normalized_text],
            embeddings=[embedding],
            metadatas=[
                _build_metadata(
                    user_id=user_id,
                    session_id=session_id,
                    metadata=metadata,
                )
            ],
        )
        return doc_id

    def search_memory(
        self,
        *,
        user_id: str,
        query: str,
        top_k: int,
        min_score: float,
    ) -> list[MemoryItem]:
        """使用 cosine 向量检索用户长期记忆。"""

        normalized_query = query.strip()
        if not normalized_query:
            return []
        if self.collection.count() == 0:
            return []

        query_variants = expand_memory_query(normalized_query)
        vector_scores: dict[str, float] = {}
        vector_items: dict[str, MemoryItem] = {}
        n_results = max(1, min(max(top_k * 2, top_k), self.collection.count()))
        for query_variant in query_variants:
            query_embedding = self.embedding_client.embed([query_variant])[0]
            result = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where={"user_id": user_id},
                include=["documents", "distances", "metadatas"],
            )
            ids = result.get("ids") or [[]]
            documents = result.get("documents") or [[]]
            distances = result.get("distances") or [[]]
            metadatas = result.get("metadatas") or [[]]

            for memory_id, document, distance, metadata in zip(ids[0], documents[0], distances[0], metadatas[0]):
                score = max(0.0, 1.0 - float(distance))
                item = MemoryItem(
                    memory_id=str(memory_id),
                    text=str(document),
                    score=score,
                    metadata=dict(metadata or {}),
                )
                vector_items[item.memory_id] = item
                vector_scores[item.memory_id] = max(vector_scores.get(item.memory_id, 0.0), score)

        all_items = self.get_all_memories(user_id)
        bm25_scores = _score_bm25(query_variants, all_items)
        kg_scores = _score_kg(query_variants, all_items)
        return _merge_hybrid_scores(
            all_items=all_items,
            vector_items=vector_items,
            vector_scores=vector_scores,
            bm25_scores=bm25_scores,
            kg_scores=kg_scores,
            top_k=top_k,
            min_score=min_score,
        )

    def get_all_memories(self, user_id: str) -> list[MemoryItem]:
        """返回用户全部长期记忆。"""

        result = self.collection.get(where={"user_id": user_id}, include=["documents", "metadatas"])
        ids = result.get("ids") or []
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        return [
            MemoryItem(memory_id=str(memory_id), text=str(document), score=1.0, metadata=dict(metadata or {}))
            for memory_id, document, metadata in zip(ids, documents, metadatas)
        ]

    def delete_memory(self, user_id: str, memory_id: str) -> bool:
        """删除用户的一条长期记忆。"""

        normalized_memory_id = memory_id.strip()
        if not normalized_memory_id:
            return False
        result = self.collection.get(ids=[normalized_memory_id], include=["metadatas"])
        ids = result.get("ids") or []
        metadatas = result.get("metadatas") or []
        if not ids or not metadatas or dict(metadatas[0] or {}).get("user_id") != user_id:
            return False
        self.collection.delete(ids=[normalized_memory_id])
        return True

    def clear_user_memory(self, user_id: str) -> None:
        """删除用户全部长期记忆。"""

        result = self.collection.get(where={"user_id": user_id})
        ids = result.get("ids") or []
        if ids:
            self.collection.delete(ids=ids)


class InMemoryMemoryStore:
    """测试用内存记忆库，不依赖 ChromaDB 或 embedding 模型。"""

    def __init__(self) -> None:
        self._items: list[tuple[str, MemoryItem]] = []

    def add_memory(
        self,
        *,
        user_id: str,
        session_id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """写入一条内存记忆。"""

        normalized_text = text.strip()
        if not normalized_text:
            raise ValueError("memory text 不能为空")
        doc_id = str(uuid.uuid4())
        metadata = dict(metadata or {})
        metadata["memory_id"] = doc_id
        metadata.setdefault("kg_entities", _serialize_values(_extract_entities(normalized_text)))
        metadata.setdefault("kg_relations", _serialize_values(_extract_relation_hints(normalized_text)))
        item = MemoryItem(
            memory_id=doc_id,
            text=normalized_text,
            score=1.0,
            metadata=_build_metadata(user_id=user_id, session_id=session_id, metadata=metadata),
        )
        self._items.append((doc_id, item))
        return doc_id

    def search_memory(
        self,
        *,
        user_id: str,
        query: str,
        top_k: int,
        min_score: float,
    ) -> list[MemoryItem]:
        """按用户过滤并按写入顺序返回记忆。"""

        query_variants = expand_memory_query(query)
        all_items = self.get_all_memories(user_id)
        vector_items: dict[str, MemoryItem] = {}
        vector_scores: dict[str, float] = {}
        query_terms = set()
        for query_variant in query_variants:
            query_terms.update(_tokenize(query_variant))

        for item in all_items:
            doc_id = item.memory_id
            if item.metadata.get("user_id") != user_id:
                continue
            score = _simple_score(item.text, query_terms)
            if score > 0:
                vector_items[doc_id] = item
                vector_scores[doc_id] = score

        bm25_scores = _score_bm25(query_variants, all_items)
        kg_scores = _score_kg(query_variants, all_items)
        return _merge_hybrid_scores(
            all_items=all_items,
            vector_items=vector_items,
            vector_scores=vector_scores,
            bm25_scores=bm25_scores,
            kg_scores=kg_scores,
            top_k=top_k,
            min_score=min_score,
        )

    def get_all_memories(self, user_id: str) -> list[MemoryItem]:
        """返回指定用户的全部内存记忆。"""

        return [
            MemoryItem(memory_id=doc_id, text=item.text, score=item.score, metadata=dict(item.metadata))
            for doc_id, item in self._items
            if item.metadata.get("user_id") == user_id
        ]

    def delete_memory(self, user_id: str, memory_id: str) -> bool:
        """删除指定用户的一条内存记忆。"""

        before_count = len(self._items)
        self._items = [
            (doc_id, item)
            for doc_id, item in self._items
            if not (doc_id == memory_id and item.metadata.get("user_id") == user_id)
        ]
        return len(self._items) < before_count

    def clear_user_memory(self, user_id: str) -> None:
        """清除指定用户的内存记忆。"""

        self._items = [
            (doc_id, item)
            for doc_id, item in self._items
            if item.metadata.get("user_id") != user_id
        ]


def _build_metadata(
    *,
    user_id: str,
    session_id: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """构造 ChromaDB 可接受的扁平 metadata。"""

    result: dict[str, Any] = {
        "user_id": user_id,
        "session_id": session_id,
        "created_at": datetime.now(UTC).isoformat(),
        "memory_type": "turn",
    }
    if metadata:
        for key, value in metadata.items():
            if value is None:
                continue
            if isinstance(value, str | int | float | bool):
                result[key] = value
            else:
                result[key] = str(value)
    return result


def expand_memory_query(query: str) -> list[str]:
    """生成轻量 query expansion 变体。"""

    normalized_query = query.strip()
    if not normalized_query:
        return []

    variants = [normalized_query]
    simplified = _simplify_query(normalized_query)
    if simplified and simplified not in variants:
        variants.append(simplified)

    expanded_terms: list[str] = []
    synonym_groups = {
        "灯": ("灯光", "主灯", "light"),
        "空调": ("ac", "温度", "制冷", "制热"),
        "窗帘": ("curtain", "开合度"),
        "洗衣机": ("washing_machine", "洗衣"),
        "插座": ("smart_plug", "电源", "功率"),
        "传感器": ("温湿度", "sensor"),
        "默认": ("偏好", "约定"),
        "记忆": ("偏好", "习惯", "规则", "别名", "约定"),
    }
    for key, synonyms in synonym_groups.items():
        if key in normalized_query:
            expanded_terms.extend(synonyms)
    if expanded_terms:
        variants.append(f"{simplified or normalized_query} {' '.join(expanded_terms)}")

    entities = _extract_entities(normalized_query)
    if entities:
        variants.append(" ".join(entities))

    deduplicated: list[str] = []
    for variant in variants:
        cleaned = variant.strip()
        if cleaned and cleaned not in deduplicated:
            deduplicated.append(cleaned)
    return deduplicated[:4]


def _merge_hybrid_scores(
    *,
    all_items: list[MemoryItem],
    vector_items: dict[str, MemoryItem],
    vector_scores: dict[str, float],
    bm25_scores: dict[str, float],
    kg_scores: dict[str, float],
    top_k: int,
    min_score: float,
) -> list[MemoryItem]:
    """合并向量、BM25 和 KG 分数。"""

    all_by_id = {item.memory_id: item for item in all_items}
    candidate_ids = set(vector_scores) | set(bm25_scores) | set(kg_scores)
    scored_items: list[MemoryItem] = []
    for memory_id in candidate_ids:
        item = vector_items.get(memory_id) or all_by_id.get(memory_id)
        if item is None:
            continue
        score = (
            vector_scores.get(memory_id, 0.0) * 0.60
            + bm25_scores.get(memory_id, 0.0) * 0.25
            + kg_scores.get(memory_id, 0.0) * 0.15
        )
        if score < min_score:
            continue
        metadata = dict(item.metadata)
        metadata["retrieval_vector_score"] = round(vector_scores.get(memory_id, 0.0), 4)
        metadata["retrieval_bm25_score"] = round(bm25_scores.get(memory_id, 0.0), 4)
        metadata["retrieval_kg_score"] = round(kg_scores.get(memory_id, 0.0), 4)
        scored_items.append(
            MemoryItem(
                memory_id=item.memory_id,
                text=item.text,
                score=score,
                metadata=metadata,
            )
        )
    scored_items.sort(key=lambda item: item.score, reverse=True)
    return scored_items[:top_k]


def _score_bm25(query_variants: list[str], items: list[MemoryItem]) -> dict[str, float]:
    """对用户记忆做轻量 BM25 关键词检索。"""

    if not query_variants or not items:
        return {}

    documents = [_tokenize(item.text) for item in items]
    document_count = len(documents)
    average_length = sum(len(document) for document in documents) / max(1, document_count)
    document_frequency: dict[str, int] = {}
    for document in documents:
        for token in set(document):
            document_frequency[token] = document_frequency.get(token, 0) + 1

    query_tokens: list[str] = []
    for query_variant in query_variants:
        query_tokens.extend(_tokenize(query_variant))
    query_tokens = list(dict.fromkeys(query_tokens))
    if not query_tokens:
        return {}

    raw_scores: dict[str, float] = {}
    k1 = 1.5
    b = 0.75
    for item, document in zip(items, documents):
        if not document:
            continue
        length = len(document)
        term_counts: dict[str, int] = {}
        for token in document:
            term_counts[token] = term_counts.get(token, 0) + 1

        score = 0.0
        for token in query_tokens:
            term_frequency = term_counts.get(token, 0)
            if term_frequency == 0:
                continue
            df = document_frequency.get(token, 0)
            idf = math.log(1 + (document_count - df + 0.5) / (df + 0.5))
            denominator = term_frequency + k1 * (1 - b + b * length / max(1.0, average_length))
            score += idf * term_frequency * (k1 + 1) / denominator
        if score > 0:
            raw_scores[item.memory_id] = score

    return _normalize_scores(raw_scores)


def _score_kg(query_variants: list[str], items: list[MemoryItem]) -> dict[str, float]:
    """基于实体和关系提示做轻量 KG 匹配。"""

    if not query_variants or not items:
        return {}

    query_entities: set[str] = set()
    for query_variant in query_variants:
        query_entities.update(_extract_entities(query_variant))
    if not query_entities:
        return {}

    scores: dict[str, float] = {}
    for item in items:
        memory_entities = set(_split_serialized_values(str(item.metadata.get("kg_entities", ""))))
        memory_entities.update(_extract_entities(item.text))
        memory_relations = set(_split_serialized_values(str(item.metadata.get("kg_relations", ""))))
        overlap = query_entities & memory_entities
        relation_hits = {
            relation
            for relation in memory_relations
            if any(entity and entity in relation for entity in query_entities)
        }
        if overlap or relation_hits:
            scores[item.memory_id] = min(1.0, (len(overlap) + len(relation_hits)) / max(1, len(query_entities)))
    return scores


def _simplify_query(query: str) -> str:
    """移除常见命令词，保留用于检索的核心内容。"""

    text = query.strip().lower()
    for word in (
        "请",
        "帮我",
        "一下",
        "查询",
        "查看",
        "打开",
        "关闭",
        "调到",
        "设置",
        "控制",
        "记得",
        "记住",
        "告诉我",
        "please",
        "turn on",
        "turn off",
        "set",
        "show",
    ):
        text = text.replace(word, " ")
    return " ".join(text.split())


def _extract_entities(text: str) -> list[str]:
    """抽取用于 KG 匹配的轻量实体。"""

    normalized = text.lower()
    entities: set[str] = set()
    for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_:-]{1,}|[0-9]+(?:\.[0-9]+)?\s*(?:度|%|w|瓦)?", normalized):
        cleaned = token.strip()
        if len(cleaned) >= 2:
            entities.add(cleaned)

    smart_home_terms = (
        "客厅",
        "卧室",
        "书房",
        "阳台",
        "主灯",
        "灯光",
        "空调",
        "窗帘",
        "洗衣机",
        "插座",
        "传感器",
        "温度",
        "湿度",
        "亮度",
        "小太阳",
        "默认",
        "偏好",
        "习惯",
        "规则",
        "约定",
    )
    for term in smart_home_terms:
        if term in text:
            entities.add(term)

    for quoted in re.findall(r"[“\"']([^“”\"']{2,30})[”\"']", text):
        entities.add(quoted.strip().lower())
    for alias in re.findall(r"(?:叫做|叫作|称为|别名是)([\u4e00-\u9fffA-Za-z0-9_ -]{2,20})", text):
        entities.add(alias.strip().lower())
    return sorted(entity for entity in entities if entity)


def _extract_relation_hints(text: str) -> list[str]:
    """抽取别名、默认值和规则等关系提示。"""

    relations: set[str] = set()
    for match in re.finditer(
        r"(?:把|将)?([\u4e00-\u9fffA-Za-z0-9_ -]{2,30})(?:叫做|叫作|称为)([\u4e00-\u9fffA-Za-z0-9_ -]{2,20})",
        text,
    ):
        subject = match.group(1).strip().lower()
        alias = match.group(2).strip().lower()
        relations.add(f"alias:{subject}->{alias}")
    for match in re.finditer(r"(默认|偏好|喜欢|规则|习惯|约定)[是为:]?([\u4e00-\u9fffA-Za-z0-9_ %.-]{2,40})", text):
        relation_type = match.group(1).strip().lower()
        value = match.group(2).strip().lower()
        relations.add(f"{relation_type}:{value}")
    return sorted(relations)


def _tokenize(text: str) -> list[str]:
    """给 BM25 使用的轻量 tokenizer，兼容中英文和设备 ID。"""

    normalized = text.lower()
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_:-]{1,}|[0-9]+(?:\.[0-9]+)?", normalized)
    cjk_runs = re.findall(r"[\u4e00-\u9fff]+", text)
    for run in cjk_runs:
        if len(run) == 1:
            tokens.append(run)
            continue
        tokens.extend(run[index : index + 2] for index in range(len(run) - 1))
        if len(run) >= 3:
            tokens.extend(run[index : index + 3] for index in range(len(run) - 2))
    return [token for token in tokens if token.strip()]


def _simple_score(text: str, query_terms: set[str]) -> float:
    """给测试内存库使用的轻量语义近似分数。"""

    if not query_terms:
        return 0.0
    text_tokens = set(_tokenize(text))
    if query_terms & text_tokens:
        return 1.0
    text_lower = text.lower()
    if any(term in text_lower for term in query_terms):
        return 0.8
    return 0.0


def _serialize_values(values: list[str]) -> str:
    """把 metadata 列表序列化为 Chroma 可接受的字符串。"""

    return "|".join(value.replace("|", " ") for value in values if value)


def _split_serialized_values(text: str) -> list[str]:
    """解析 metadata 中的列表字符串。"""

    return [value.strip().lower() for value in text.split("|") if value.strip()]


def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    """把分数缩放到 0-1。"""

    if not scores:
        return {}
    max_score = max(scores.values())
    if max_score <= 0:
        return {}
    return {key: value / max_score for key, value in scores.items() if value > 0}
