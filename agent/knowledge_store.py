"""儿童教育 RAG 知识库存储和检索。"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
from pathlib import Path
import re
from typing import Any, Protocol

from agent.embedding_client import EmbeddingClient, OllamaEmbeddingClient
from agent.knowledge_config import KnowledgeConfig


@dataclass(slots=True)
class KnowledgeChunk:
    """一段可被检索的知识库文本。"""

    chunk_id: str
    title: str
    text: str
    source: str
    score: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


class KnowledgeStore(Protocol):
    """知识库检索的最小接口。"""

    def search(self, query: str, *, top_k: int, min_score: float) -> list[KnowledgeChunk]:
        """检索与 query 相关的知识片段。"""

    def add_chunks(self, chunks: list[KnowledgeChunk]) -> None:
        """写入知识片段。"""

    def clear(self) -> None:
        """清空当前知识库 collection。"""


class ChromaKnowledgeStore:
    """基于 ChromaDB 的本地 RAG 知识库。"""

    def __init__(
        self,
        config: KnowledgeConfig | None = None,
        embedding_client: EmbeddingClient | None = None,
    ) -> None:
        self.config = config or KnowledgeConfig.from_env()
        if self.config.embed_backend != "ollama":
            raise ValueError(f"不支持的知识库 embedding 后端：{self.config.embed_backend}")
        self.embedding_client = embedding_client or OllamaEmbeddingClient(self.config)
        chroma_path = Path(self.config.chroma_path)
        if chroma_path.is_absolute():
            raise RuntimeError("AGENT_KB_CHROMA_PATH 必须是相对路径，便于不同机器测试。")

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

    def add_chunks(self, chunks: list[KnowledgeChunk]) -> None:
        """批量写入知识库片段。"""

        clean_chunks = [chunk for chunk in chunks if chunk.text.strip()]
        if not clean_chunks:
            return

        batch_size = 32
        for start in range(0, len(clean_chunks), batch_size):
            batch = clean_chunks[start : start + batch_size]
            documents = [chunk.text.strip() for chunk in batch]
            embeddings = self._embed_chunks(batch)
            self.collection.upsert(
                ids=[chunk.chunk_id for chunk in batch],
                documents=documents,
                embeddings=embeddings,
                metadatas=[
                    _build_metadata(
                        title=chunk.title,
                        source=chunk.source,
                        metadata=chunk.metadata,
                    )
                    for chunk in batch
                ],
            )

    def search(self, query: str, *, top_k: int, min_score: float) -> list[KnowledgeChunk]:
        """结合向量检索、BM25 和 query expansion 检索知识库。"""

        normalized_query = query.strip()
        if not normalized_query or self.collection.count() == 0:
            return []

        query_variants = expand_knowledge_query(normalized_query)
        vector_scores: dict[str, float] = {}
        vector_items: dict[str, KnowledgeChunk] = {}
        n_results = max(1, min(max(top_k * 3, top_k), self.collection.count()))
        for query_variant in query_variants:
            query_embedding = self.embedding_client.embed([self._text_for_embedding(query_variant)])[0]
            result = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "distances", "metadatas"],
            )
            ids = result.get("ids") or [[]]
            documents = result.get("documents") or [[]]
            distances = result.get("distances") or [[]]
            metadatas = result.get("metadatas") or [[]]
            for chunk_id, document, distance, metadata in zip(ids[0], documents[0], distances[0], metadatas[0]):
                score = max(0.0, 1.0 - float(distance))
                item = _chunk_from_raw(str(chunk_id), str(document), dict(metadata or {}), score)
                vector_items[item.chunk_id] = item
                vector_scores[item.chunk_id] = max(vector_scores.get(item.chunk_id, 0.0), score)

        all_chunks = self.get_all_chunks()
        bm25_scores = _score_bm25(query_variants, all_chunks)
        return _merge_scores(
            all_chunks=all_chunks,
            vector_items=vector_items,
            vector_scores=vector_scores,
            bm25_scores=bm25_scores,
            top_k=top_k,
            min_score=min_score,
        )

    def get_all_chunks(self) -> list[KnowledgeChunk]:
        """读取 collection 中的全部知识片段，用于 BM25。"""

        result = self.collection.get(include=["documents", "metadatas"])
        ids = result.get("ids") or []
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        return [
            _chunk_from_raw(str(chunk_id), str(document), dict(metadata or {}), 1.0)
            for chunk_id, document, metadata in zip(ids, documents, metadatas)
        ]

    def clear(self) -> None:
        """清空当前 collection。"""

        result = self.collection.get()
        ids = result.get("ids") or []
        if ids:
            self.collection.delete(ids=ids)

    def _text_for_embedding(self, text: str) -> str:
        """限制送入 Ollama embedding 的长度，避免部分本地 bge-m3 对长英文片段返回 NaN。"""

        compact = " ".join(text.strip().split())
        return compact[: self.config.embed_text_chars] if compact else text

    def _embed_chunks(self, chunks: list[KnowledgeChunk]) -> list[list[float]]:
        """批量 embedding，遇到 Ollama NaN 等本地模型问题时自动降级。"""

        texts = [self._text_for_embedding(chunk.title) for chunk in chunks]
        try:
            return self.embedding_client.embed(texts)
        except RuntimeError:
            if len(chunks) > 1:
                midpoint = len(chunks) // 2
                return self._embed_chunks(chunks[:midpoint]) + self._embed_chunks(chunks[midpoint:])

            fallback_text = self._text_for_embedding(chunks[0].title) or "story"
            return self.embedding_client.embed([fallback_text])


class InMemoryKnowledgeStore:
    """测试用内存知识库，不依赖 ChromaDB 或 embedding 模型。"""

    def __init__(self, chunks: list[KnowledgeChunk] | None = None) -> None:
        self._chunks = list(chunks or [])

    def add_chunks(self, chunks: list[KnowledgeChunk]) -> None:
        """写入内存知识片段。"""

        self._chunks.extend(chunk for chunk in chunks if chunk.text.strip())

    def search(self, query: str, *, top_k: int, min_score: float) -> list[KnowledgeChunk]:
        """使用 BM25 和 query expansion 检索内存知识库。"""

        query_variants = expand_knowledge_query(query)
        bm25_scores = _score_bm25(query_variants, self._chunks)
        substring_scores = _score_substring(query_variants, self._chunks)
        scores: dict[str, float] = {}
        for chunk in self._chunks:
            score = max(bm25_scores.get(chunk.chunk_id, 0.0), substring_scores.get(chunk.chunk_id, 0.0))
            if score >= min_score and score > 0:
                scores[chunk.chunk_id] = score

        by_id = {chunk.chunk_id: chunk for chunk in self._chunks}
        results = [
            KnowledgeChunk(
                chunk_id=chunk_id,
                title=by_id[chunk_id].title,
                text=by_id[chunk_id].text,
                source=by_id[chunk_id].source,
                score=score,
                metadata=dict(by_id[chunk_id].metadata),
            )
            for chunk_id, score in scores.items()
            if chunk_id in by_id
        ]
        results.sort(key=lambda chunk: chunk.score, reverse=True)
        return results[:top_k]

    def clear(self) -> None:
        """清空内存知识库。"""

        self._chunks = []


def build_grimms_chunks(
    raw_text: str,
    *,
    source: str,
    chunk_chars: int = 1200,
    chunk_overlap: int = 160,
) -> list[KnowledgeChunk]:
    """把 Project Gutenberg 的格林童话文本切成可检索片段。"""

    body = _strip_gutenberg_boilerplate(raw_text)
    stories = _split_grimms_stories(body)
    chunks: list[KnowledgeChunk] = []
    for title, story_text in stories:
        for index, chunk_text in enumerate(_chunk_text(story_text, chunk_chars, chunk_overlap)):
            chunk_id = _stable_chunk_id(title, index, chunk_text)
            chunks.append(
                KnowledgeChunk(
                    chunk_id=chunk_id,
                    title=title,
                    text=chunk_text,
                    source=source,
                    metadata={
                        "corpus": "grimms_fairy_tales",
                        "chunk_index": index,
                    },
                )
            )
    return chunks


def expand_knowledge_query(query: str) -> list[str]:
    """生成轻量 query expansion 变体，兼容中文提问和英文原文。"""

    normalized_query = query.strip()
    if not normalized_query:
        return []

    variants = [normalized_query]
    simplified = _simplify_query(normalized_query)
    if simplified and simplified not in variants:
        variants.append(simplified)

    expansions: list[str] = []
    synonym_groups = {
        "格林": ("grimm", "fairy tales", "children story"),
        "童话": ("fairy tale", "story", "children"),
        "故事": ("story", "tale"),
        "睡前": ("bedtime", "gentle", "children"),
        "寓意": ("moral", "lesson", "meaning"),
        "教育": ("lesson", "moral", "children"),
        "小红帽": ("little red-cap", "little red riding hood", "wolf", "grandmother"),
        "灰姑娘": ("ashputtel", "cinderella", "slipper", "stepmother"),
        "白雪公主": ("snowdrop", "snow white", "queen", "dwarfs"),
        "睡美人": ("briar rose", "sleeping beauty"),
        "糖果屋": ("hansel and gretel", "witch", "forest"),
        "汉塞尔": ("hansel", "gretel"),
        "格蕾特": ("gretel", "hansel"),
        "长发公主": ("rapunzel", "tower"),
        "青蛙王子": ("frog-prince", "frog king"),
        "狼": ("wolf",),
        "公主": ("princess", "queen", "king"),
        "勇敢": ("brave", "courage", "valiant"),
    }
    query_lower = normalized_query.lower()
    for key, synonyms in synonym_groups.items():
        if key in normalized_query or key.lower() in query_lower:
            expansions.extend(synonyms)
    if expansions:
        variants.append(f"{simplified or normalized_query} {' '.join(expansions)}")

    deduplicated: list[str] = []
    for variant in variants:
        cleaned = variant.strip()
        if cleaned and cleaned not in deduplicated:
            deduplicated.append(cleaned)
    return deduplicated[:4]


def _strip_gutenberg_boilerplate(text: str) -> str:
    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK GRIMMS' FAIRY TALES ***"
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK GRIMMS' FAIRY TALES ***"
    start = text.find(start_marker)
    if start != -1:
        text = text[start + len(start_marker) :]
    end = text.find(end_marker)
    if end != -1:
        text = text[:end]
    return text.strip()


def _split_grimms_stories(text: str) -> list[tuple[str, str]]:
    lines = text.splitlines()
    content_titles: list[str] = []
    in_contents = False
    body_start_index = 0
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "CONTENTS:":
            in_contents = True
            continue
        if in_contents and stripped == "THE GOLDEN BIRD" and content_titles and line == stripped:
            body_start_index = index
            break
        if in_contents and _looks_like_title(stripped):
            content_titles.append(stripped)

    title_set = set(content_titles)
    stories: list[tuple[str, str]] = []
    current_title = ""
    current_lines: list[str] = []
    for line in lines[body_start_index:]:
        stripped = line.strip()
        if stripped in title_set:
            if current_title and current_lines:
                stories.append((current_title, _clean_story_text("\n".join(current_lines))))
            current_title = stripped
            current_lines = []
            continue
        if current_title:
            current_lines.append(line)
    if current_title and current_lines:
        stories.append((current_title, _clean_story_text("\n".join(current_lines))))
    return [(title, story) for title, story in stories if story]


def _looks_like_title(text: str) -> bool:
    if not text or len(text) > 90:
        return False
    if text.startswith(("1.", "2.")):
        return False
    return text.upper() == text and any(char.isalpha() for char in text)


def _clean_story_text(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    return "\n".join(line.rstrip() for line in text.splitlines()).strip()


def _chunk_text(text: str, chunk_chars: int, chunk_overlap: int) -> list[str]:
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", text) if paragraph.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        next_text = paragraph if not current else f"{current}\n\n{paragraph}"
        if len(next_text) <= chunk_chars:
            current = next_text
            continue
        if current:
            chunks.append(current)
        if len(paragraph) <= chunk_chars:
            current = paragraph
            continue
        for start in range(0, len(paragraph), max(1, chunk_chars - chunk_overlap)):
            part = paragraph[start : start + chunk_chars].strip()
            if part:
                chunks.append(part)
        current = ""
    if current:
        chunks.append(current)
    return chunks


def _stable_chunk_id(title: str, index: int, text: str) -> str:
    digest = hashlib.sha1(f"{title}:{index}:{text}".encode("utf-8")).hexdigest()[:16]
    return f"grimms-{digest}"


def _build_metadata(*, title: str, source: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "title": title,
        "source": source,
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


def _chunk_from_raw(chunk_id: str, document: str, metadata: dict[str, Any], score: float) -> KnowledgeChunk:
    return KnowledgeChunk(
        chunk_id=chunk_id,
        title=str(metadata.get("title", "未知故事")),
        text=document,
        source=str(metadata.get("source", "")),
        score=score,
        metadata=metadata,
    )


def _merge_scores(
    *,
    all_chunks: list[KnowledgeChunk],
    vector_items: dict[str, KnowledgeChunk],
    vector_scores: dict[str, float],
    bm25_scores: dict[str, float],
    top_k: int,
    min_score: float,
) -> list[KnowledgeChunk]:
    by_id = {chunk.chunk_id: chunk for chunk in all_chunks}
    candidate_ids = set(vector_scores) | set(bm25_scores)
    scored_chunks: list[KnowledgeChunk] = []
    for chunk_id in candidate_ids:
        chunk = vector_items.get(chunk_id) or by_id.get(chunk_id)
        if chunk is None:
            continue
        score = vector_scores.get(chunk_id, 0.0) * 0.65 + bm25_scores.get(chunk_id, 0.0) * 0.35
        if score < min_score:
            continue
        metadata = dict(chunk.metadata)
        metadata["retrieval_vector_score"] = round(vector_scores.get(chunk_id, 0.0), 4)
        metadata["retrieval_bm25_score"] = round(bm25_scores.get(chunk_id, 0.0), 4)
        scored_chunks.append(
            KnowledgeChunk(
                chunk_id=chunk.chunk_id,
                title=chunk.title,
                text=chunk.text,
                source=chunk.source,
                score=score,
                metadata=metadata,
            )
        )
    scored_chunks.sort(key=lambda chunk: chunk.score, reverse=True)
    return scored_chunks[:top_k]


def _score_bm25(query_variants: list[str], chunks: list[KnowledgeChunk]) -> dict[str, float]:
    if not query_variants or not chunks:
        return {}

    documents = [_tokenize(f"{chunk.title} {chunk.text}") for chunk in chunks]
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
    for chunk, document in zip(chunks, documents):
        if not document:
            continue
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
            denominator = term_frequency + k1 * (1 - b + b * len(document) / max(1.0, average_length))
            score += idf * term_frequency * (k1 + 1) / denominator
        if score > 0:
            raw_scores[chunk.chunk_id] = score
    return _normalize_scores(raw_scores)


def _score_substring(query_variants: list[str], chunks: list[KnowledgeChunk]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for chunk in chunks:
        haystack = f"{chunk.title} {chunk.text}".lower()
        for query_variant in query_variants:
            tokens = [token for token in _tokenize(query_variant) if len(token) >= 2]
            if any(token in haystack for token in tokens):
                scores[chunk.chunk_id] = 1.0
                break
    return scores


def _simplify_query(query: str) -> str:
    text = query.strip().lower()
    for word in (
        "请",
        "帮我",
        "讲",
        "讲讲",
        "说说",
        "一个",
        "查一下",
        "找一下",
        "告诉我",
        "适合孩子",
        "please",
        "tell me",
        "search",
        "find",
    ):
        text = text.replace(word, " ")
    return " ".join(text.split())


def _tokenize(text: str) -> list[str]:
    normalized = text.lower()
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9'-]{1,}|[0-9]+(?:\.[0-9]+)?", normalized)
    cjk_runs = re.findall(r"[\u4e00-\u9fff]+", text)
    for run in cjk_runs:
        if len(run) == 1:
            tokens.append(run)
            continue
        tokens.extend(run[index : index + 2] for index in range(len(run) - 1))
        if len(run) >= 3:
            tokens.extend(run[index : index + 3] for index in range(len(run) - 2))
    return [token for token in tokens if token.strip()]


def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    max_score = max(scores.values())
    if max_score <= 0:
        return {}
    return {key: value / max_score for key, value in scores.items() if value > 0}
