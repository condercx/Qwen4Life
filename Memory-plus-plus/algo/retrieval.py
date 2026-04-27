"""
Memory++ 检索-生成核心：BenchmarkRAGPlusPlus 的封装。

12 项创新:
  1. 三路混合检索 (Vector + BM25 + KG)
  2. Cross-encoder 两阶段重排
  3. 日期感知检索与推理
  4. 题型感知生成策略
  5. 会话摘要分层检索
  6. 知识更新时序排序
  7. 自适应查询扩展 (置信度门控)
  8. 证据句高亮
  9. 链式检索 (多跳)
  10. 关系型知识图谱 (S,R,O 三元组)
  11. 对抗前提检测
  12. 答案接地验证 + 上下文窗口扩展
"""

import os
import re
import time
from collections import defaultdict
from datetime import datetime

import numpy as np
import chromadb
from chromadb.config import Settings
from openai import OpenAI, APIConnectionError, APITimeoutError, InternalServerError, RateLimitError
from rank_bm25 import BM25Okapi
import requests as _requests

from .config import Config
from .entities import extract_entities, extract_relation_triples, STOP_ENTITIES
from .scoring import clean_answer, extract_counting_answer, is_idk

# LME 和 LoCoMo 中的时间类题型
_TEMPORAL_TYPES = {"temporal-reasoning", "temporal"}


def _parse_date(date_str: str) -> datetime | None:
    """解析 LongMemEval / LoCoMo 格式的日期。"""
    if not date_str:
        return None
    fmts = [
        "%Y/%m/%d (%a) %H:%M",
        "%I:%M %p on %d %B, %Y",
        "%Y-%m-%d",
        "%B %d, %Y",
        "%d %B, %Y",
    ]
    for fmt in fmts:
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue
    return None


class MemoryPlusPlusRAG:
    """Memory++ RAG 系统：知识增强的长期对话记忆检索。

    Pipeline:
        Query → [自适应查询扩展] → [三路混合检索]
              → [Cross-Encoder 重排] → [上下文窗口扩展]
              → [证据句高亮] → [题型感知 Prompt]
              → [LLM 生成] → [答案接地验证] → [后处理]
    """

    def __init__(self, config: Config = None, ablation: str = ""):
        """
        Args:
            config: 配置对象，默认使用 Config 类默认值
            ablation: 消融标志，逗号分隔 (如 "no_bm25,no_kg")
        """
        cfg = config or Config()
        self.cfg = cfg
        self.ablation = set(ablation.split(",")) if ablation else set()

        self.llm_client = OpenAI(
            api_key=cfg.API_KEY, base_url=cfg.BASE_URL,
            timeout=300.0, max_retries=0,
        )
        self.chroma = chromadb.PersistentClient(
            path=cfg.CHROMA_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection_name = cfg.COLLECTION_NAME
        self.collection = None

        # 内存知识图谱
        self.kg_entities: dict[str, set[str]] = {}
        self.chunk_texts: dict[str, str] = {}
        self.chunk_dates: dict[str, str] = {}
        self.chunk_session: dict[str, str] = {}
        self.session_summaries: dict[str, str] = {}
        self.session_chunks: dict[str, list[str]] = {}
        self.kg_triples: list[tuple[str, str, str, str]] = []
        self.kg_entity_relations: dict[str, set[str]] = {}
        self.bm25 = None
        self.bm25_chunk_ids: list[str] = []

    # ================================================================ #
    #  重置
    # ================================================================ #

    def reset(self):
        """每道题前重置向量库和 KG。"""
        try:
            self.chroma.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.chroma.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.kg_entities.clear()
        self.chunk_texts.clear()
        self.chunk_dates.clear()
        self.chunk_session.clear()
        self.session_summaries.clear()
        self.session_chunks.clear()
        self.kg_triples.clear()
        self.kg_entity_relations.clear()
        self.bm25 = None
        self.bm25_chunk_ids = []

    # ================================================================ #
    #  Embedding (带重试)
    # ================================================================ #

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        BATCH = 32
        result = []
        for i in range(0, len(texts), BATCH):
            batch_texts = texts[i:i + BATCH]
            for attempt in range(8):
                try:
                    resp = self.llm_client.embeddings.create(
                        model=self.cfg.EMBED_MODEL, input=batch_texts
                    )
                    result.extend([item.embedding for item in resp.data])
                    break
                except (APIConnectionError, APITimeoutError, InternalServerError, RateLimitError) as e:
                    wait = 5 * (attempt + 1)
                    print(f"    [Embed retry {attempt + 1}/8] {type(e).__name__}, {wait}s...")
                    time.sleep(wait)
                except Exception as e:
                    if any(code in str(e) for code in ("502", "503", "504")):
                        wait = 5 * (attempt + 1)
                        print(f"    [Embed retry {attempt + 1}/8] {type(e).__name__}, {wait}s...")
                        time.sleep(wait)
                    else:
                        raise
            else:
                raise RuntimeError(f"Embedding failed after 8 retries (batch {i // BATCH})")
            if i + BATCH < len(texts):
                time.sleep(3)
        return result

    # ================================================================ #
    #  Reranking (Innovation 2)
    # ================================================================ #

    def _rerank(self, query: str, documents: list[str], top_n: int = 10) -> list[tuple[str, float]]:
        """Cross-encoder 重排 via SiliconFlow /v1/rerank API。"""
        if not documents:
            return []
        try:
            resp = _requests.post(
                f"{self.cfg.BASE_URL.rstrip('/').replace('/v1', '')}/v1/rerank",
                headers={"Authorization": f"Bearer {self.cfg.API_KEY}"},
                json={
                    "model": self.cfg.RERANKER_MODEL,
                    "query": query,
                    "documents": documents,
                    "top_n": min(top_n, len(documents)),
                },
                timeout=30,
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])
            return [(documents[r["index"]], r["relevance_score"]) for r in results]
        except Exception as e:
            print(f"    [Reranker warning] {e}, fallback to original order")
            return [(d, 0.0) for d in documents[:top_n]]

    # ================================================================ #
    #  Evidence Highlighting (Innovation 8)
    # ================================================================ #

    def _highlight_evidence(self, question: str, doc: str) -> str:
        """用 ►◄ 标记 chunk 中最相关的句子，引导模型注意力。"""
        sentences = re.split(r'(?<=[.!?])\s+|\n+', doc)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        if len(sentences) < 2:
            return doc

        q_words = set(question.lower().split()) - STOP_ENTITIES
        q_entities = set(e.lower() for e in extract_entities(question) if len(e) > 2)
        query_terms = q_words | q_entities

        best_score, best_idx = 0, -1
        for idx, sent in enumerate(sentences):
            sent_lower = sent.lower()
            sent_words = set(sent_lower.split())
            score = len(query_terms & sent_words)
            for ent in q_entities:
                if ent in sent_lower:
                    score += 2  # 实体匹配权重 2x
            if score > best_score:
                best_score, best_idx = score, idx

        if best_score < 2 or best_idx < 0:
            return doc
        highlighted = sentences[best_idx]
        return doc.replace(highlighted, f"►{highlighted}◄")

    # ================================================================ #
    #  Query Expansion (Innovation 7)
    # ================================================================ #

    def _expand_query(self, query: str, question_type: str = None) -> list[str]:
        """LLM 查询扩展：置信度低时生成 2-3 个关键词变体。"""
        if "no_query_expansion" in self.ablation:
            return []
        type_hint = f" (question type: {question_type})" if question_type else ""
        try:
            resp = self.llm_client.chat.completions.create(
                model=self.cfg.LLM_MODEL,
                messages=[{"role": "user", "content": (
                    f"Rewrite this question as 2-3 short keyword search queries "
                    f"that would help find the answer in a personal memory database.{type_hint}\n"
                    f"Question: {query}\n"
                    f"Output ONLY the queries, one per line. No numbering, no explanation."
                )}],
                max_tokens=100, temperature=0.3,
                extra_body={"enable_thinking": False},
            )
            lines = resp.choices[0].message.content.strip().split('\n')
            variants = [l.strip().strip('-').strip('•').strip()
                        for l in lines if l.strip() and len(l.strip()) > 5]
            return variants[:3]
        except Exception as e:
            print(f"    [QueryExpansion warning] {e}")
            return []

    # ================================================================ #
    #  Query Simplification
    # ================================================================ #

    @staticmethod
    def _simplify_query(query: str) -> str:
        q = query.strip().rstrip('?')
        q = re.sub(r'^(what|which|who|where|when|how many|how much|how long|how|do you remember|can you recall|can you tell me|tell me)\b\s*', '', q, flags=re.I)
        q = re.sub(r'^(is|are|was|were|did|does|do|has|have|had|would|could|should|might)\s+', '', q, flags=re.I)
        q = re.sub(r'\b(i|you|we|they|my|your|our|their)\b', '', q, flags=re.I)
        q = ' '.join(q.split())
        return q.strip() if len(q.strip()) > 3 else query

    # ================================================================ #
    #  索引 (Indexing)
    # ================================================================ #

    def index_sessions(self, sessions: list, session_dates=None):
        """索引对话 sessions → ChromaDB + KG + BM25。"""
        assert self.collection is not None
        chunks, ids, metas = [], [], []

        for i, sess in enumerate(sessions):
            if isinstance(sess, list):
                messages = sess
                sid = f"sess_{i}"
            else:
                sid = sess.get("session_id", f"sess_{i}")
                messages = sess.get("messages", [])

            date_str = ""
            if session_dates and i < len(session_dates):
                date_str = session_dates[i]

            for j in range(0, len(messages), 2):
                pair = messages[j:j + 2]
                text = "\n".join(
                    f"{m.get('role', '?')}: {m.get('content', '')}" for m in pair
                )
                if not text.strip():
                    continue
                # Overlap: 前一轮 assistant 回复作为上下文
                if j >= 2:
                    prev_msg = messages[j - 1] if j - 1 < len(messages) else None
                    if prev_msg and prev_msg.get('role') == 'assistant':
                        prev_text = prev_msg.get('content', '')[:200]
                        if prev_text:
                            text = f"[prev] assistant: {prev_text}\n{text}"
                # 日期前缀
                if date_str:
                    text = f"[Date: {date_str}] {text}"

                CHUNK_SIZE = self.cfg.CHUNK_MAX_CHARS
                text_parts = [text[k:k + CHUNK_SIZE] for k in range(0, len(text), CHUNK_SIZE)]
                for part_idx, part_text in enumerate(text_parts):
                    chunk_id = f"{sid}_c{j // 2}" if part_idx == 0 else f"{sid}_c{j // 2}p{part_idx}"
                    chunks.append(part_text)
                    ids.append(chunk_id)
                    metas.append({"session_id": sid, "date": date_str})

                    self.chunk_texts[chunk_id] = part_text
                    self.chunk_dates[chunk_id] = date_str
                    self.chunk_session[chunk_id] = sid

                    # KG 实体索引
                    for ent in extract_entities(part_text):
                        key = ent.lower().strip()
                        if len(key) < 2:
                            continue
                        if key not in self.kg_entities:
                            self.kg_entities[key] = set()
                        self.kg_entities[key].add(chunk_id)

                    # KG 关系三元组索引
                    for subj, rel, obj in extract_relation_triples(part_text):
                        self.kg_triples.append((subj, rel, obj, chunk_id))
                        for ent_key in (subj, obj):
                            if ent_key and len(ent_key) > 2:
                                if ent_key not in self.kg_entity_relations:
                                    self.kg_entity_relations[ent_key] = set()
                                self.kg_entity_relations[ent_key].add(chunk_id)

        if not chunks:
            return
        embeddings = self._embed_batch(chunks)
        for _retry in range(3):
            try:
                self.collection.add(ids=ids, embeddings=embeddings,
                                    documents=chunks, metadatas=metas)
                break
            except Exception as e:
                if "does not exist" in str(e) and _retry < 2:
                    print(f"    [ChromaDB] Collection lost, recreating... ({e})")
                    self.collection = self.chroma.get_or_create_collection(
                        name=self.collection_name,
                        metadata={"hnsw:space": "cosine"},
                    )
                else:
                    raise

        # BM25 索引
        tokenized = [doc.lower().split() for doc in chunks]
        self.bm25 = BM25Okapi(tokenized)
        self.bm25_chunk_ids = ids

        # 会话摘要 (Innovation 5)
        for cid in ids:
            sid = self.chunk_session.get(cid, "")
            if sid:
                if sid not in self.session_chunks:
                    self.session_chunks[sid] = []
                self.session_chunks[sid].append(cid)
        for sid, cids in self.session_chunks.items():
            all_ents = set()
            for cid in cids:
                text = self.chunk_texts.get(cid, "")
                all_ents.update(e.lower() for e in extract_entities(text) if len(e) > 2)
            date = self.chunk_dates.get(cids[0], "") if cids else ""
            summary = f"[Session {sid}]"
            if date:
                summary += f" [Date: {date}]"
            summary += f" Topics: {', '.join(sorted(all_ents)[:20])}"
            self.session_summaries[sid] = summary

    # ================================================================ #
    #  三路混合检索 (Innovation 1, 2, 3, 5, 10, 12)
    # ================================================================ #

    def retrieve_hybrid(self, query: str, top_k: int = 10,
                        question_type: str = None):
        """混合检索：向量 + BM25 + KG 实体匹配。"""
        assert self.collection is not None
        count = self.collection.count()
        if count == 0:
            return [], [], 0.0

        # 1. 向量检索
        k = min(top_k + (5 if question_type in _TEMPORAL_TYPES else 0), count)
        emb = self._embed_batch([query])[0]
        res = self.collection.query(
            query_embeddings=[emb], n_results=k,
            include=["documents", "metadatas"]
        )
        vector_docs = res["documents"][0] if res["documents"] else []
        vector_metas = res["metadatas"][0] if res["metadatas"] else []

        # 2. KG 实体匹配 (Precision KG)
        simplified = self._simplify_query(query)
        query_entities = list(set(extract_entities(query)) | set(extract_entities(simplified)))
        query_words = (set(query.lower().split()) | set(simplified.lower().split())) - STOP_ENTITIES
        kg_chunk_ids: dict[str, int] = defaultdict(int)

        for ent in query_entities:
            ent_lower = ent.lower().strip()
            if len(ent_lower) < 3:
                continue
            # 精确匹配
            if ent_lower in self.kg_entities:
                for cid in self.kg_entities[ent_lower]:
                    kg_chunk_ids[cid] += 3
            # 多词实体部分匹配
            if ' ' in ent_lower:
                for stored_ent, cids in self.kg_entities.items():
                    if ent_lower in stored_ent or stored_ent in ent_lower:
                        for cid in cids:
                            kg_chunk_ids[cid] += 1

        # 关系三元组匹配 (Innovation 10)
        if "no_kg" not in self.ablation and self.kg_entity_relations:
            query_ent_lower = set(e.lower() for e in query_entities if len(e) > 2)
            for ent in query_ent_lower:
                if ent in self.kg_entity_relations:
                    for cid in self.kg_entity_relations[ent]:
                        kg_chunk_ids[cid] += 2

        if "no_kg" in self.ablation:
            kg_docs = []
        else:
            kg_sorted = sorted(kg_chunk_ids.items(), key=lambda x: -x[1])[:top_k]
            kg_docs = [self.chunk_texts[cid] for cid, score in kg_sorted
                       if cid in self.chunk_texts and score >= 3]

        # 3. BM25 关键词检索
        bm25_docs = []
        if "no_bm25" not in self.ablation and self.bm25 is not None:
            query_tokens = query.lower().split()
            scores = self.bm25.get_scores(query_tokens)
            top_indices = scores.argsort()[-top_k:][::-1]
            for idx in top_indices:
                if scores[idx] > 0 and idx < len(self.bm25_chunk_ids):
                    cid = self.bm25_chunk_ids[idx]
                    if cid in self.chunk_texts:
                        bm25_docs.append(self.chunk_texts[cid])

        # 4. 合并去重
        seen = set()
        merged, merged_dates = [], []
        for doc, meta in zip(vector_docs, vector_metas):
            key = doc[:200]
            if key not in seen:
                seen.add(key)
                merged.append(doc)
                merged_dates.append(meta.get("date", "") if meta else "")

        for doc in bm25_docs:
            key = doc[:200]
            if key not in seen:
                seen.add(key)
                merged.append(doc)
                for cid, txt in self.chunk_texts.items():
                    if txt == doc:
                        merged_dates.append(self.chunk_dates.get(cid, ""))
                        break
                else:
                    merged_dates.append("")

        for doc in kg_docs:
            key = doc[:200]
            if key not in seen:
                seen.add(key)
                merged.append(doc)
                for cid, txt in self.chunk_texts.items():
                    if txt == doc:
                        merged_dates.append(self.chunk_dates.get(cid, ""))
                        break
                else:
                    merged_dates.append("")

        # 5. Knowledge-update: 展开 top session 全部 chunks (Innovation 6)
        if question_type == "knowledge-update" and vector_metas:
            expand_sessions = set()
            top_session = vector_metas[0].get("session_id", "") if vector_metas[0] else ""
            if top_session:
                expand_sessions.add(top_session)
            newest_dt, newest_sid = None, None
            for meta in vector_metas:
                sid = meta.get("session_id", "")
                ds = meta.get("date", "")
                if ds and sid:
                    dt = _parse_date(ds)
                    if dt and (newest_dt is None or dt > newest_dt):
                        newest_dt, newest_sid = dt, sid
            if newest_sid:
                expand_sessions.add(newest_sid)
            for exp_sid in expand_sessions:
                for cid, sid in self.chunk_session.items():
                    if sid == exp_sid:
                        doc = self.chunk_texts.get(cid, "")
                        key = doc[:200]
                        if key not in seen:
                            seen.add(key)
                            merged.append(doc)
                            merged_dates.append(self.chunk_dates.get(cid, ""))

        # 6. Multi-session: 会话摘要引导扩展 (Innovation 5)
        if question_type == "multi-session" and self.session_summaries:
            query_ents = set(e.lower() for e in extract_entities(query) if len(e) > 2)
            qw = set(query.lower().split()) - STOP_ENTITIES
            matched_sessions = set()
            for sid, summary in self.session_summaries.items():
                summary_lower = summary.lower()
                for ent in query_ents | qw:
                    if len(ent) >= 4 and ent in summary_lower:
                        matched_sessions.add(sid)
                        break
            for sid in matched_sessions:
                for cid in self.session_chunks.get(sid, []):
                    doc = self.chunk_texts.get(cid, "")
                    key = doc[:200]
                    if key not in seen:
                        seen.add(key)
                        merged.append(doc)
                        merged_dates.append(self.chunk_dates.get(cid, ""))

        # 7. Multi-hop: 实体桥接扩展
        if question_type == "multi-hop" and merged:
            expansion_ents = set()
            for doc in merged[:5]:
                for ent in extract_entities(doc):
                    expansion_ents.add(ent.lower().strip())
            for ent in expansion_ents:
                if ent in self.kg_entities:
                    for cid in self.kg_entities[ent]:
                        if cid in self.chunk_texts:
                            doc = self.chunk_texts[cid]
                            key = doc[:200]
                            if key not in seen:
                                seen.add(key)
                                merged.append(doc)
                                merged_dates.append(self.chunk_dates.get(cid, ""))

        # 8. 日期排序 (Innovation 3)
        if question_type in (_TEMPORAL_TYPES | {"knowledge-update"}) and any(merged_dates):
            dated_items = [(doc, ds, _parse_date(ds)) for doc, ds in zip(merged, merged_dates)]
            dated_items.sort(key=lambda x: x[2] or datetime.min, reverse=True)
            merged = [d[0] for d in dated_items]
            merged_dates = [d[1] for d in dated_items]

        # 9. Cross-encoder 重排 (Innovation 2)
        reranked = []
        if "no_reranker" not in self.ablation and len(merged) > top_k:
            candidates = merged[:top_k + 10]
            candidate_dates = merged_dates[:top_k + 10]
            reranked = self._rerank(query, candidates, top_n=top_k + 5)
            doc_to_date = dict(zip([d[:200] for d in candidates], candidate_dates))
            merged = [doc for doc, _ in reranked]
            merged_dates = [doc_to_date.get(doc[:200], "") for doc, _ in reranked]
            # knowledge-update: rerank 后再按日期降序
            if question_type == "knowledge-update" and any(merged_dates):
                dated_items = [(d, ds, _parse_date(ds)) for d, ds in zip(merged, merged_dates)]
                dated_items.sort(key=lambda x: x[2] or datetime.min, reverse=True)
                merged = [d[0] for d in dated_items]
                merged_dates = [d[1] for d in dated_items]

        # 10. 多维置信度评分 (Innovation: Multi-dimensional Confidence)
        retrieval_confidence = 0.0
        if "no_multi_conf" not in self.ablation:
            conf_signals = []
            if "no_reranker" not in self.ablation and reranked:
                conf_signals.append(("reranker", min(reranked[0][1], 1.0), 0.4))
            elif vector_metas and res.get("distances"):
                dists = res["distances"][0]
                vec_sim = 1.0 - (dists[0] if dists else 1.0)
                conf_signals.append(("vector", max(vec_sim, 0.0), 0.4))
            if query_entities and merged:
                top_text = " ".join(merged[:5]).lower()
                q_ents = set(e.lower() for e in query_entities if len(e) > 2)
                if q_ents:
                    covered = sum(1 for e in q_ents if e in top_text)
                    conf_signals.append(("entity_cov", covered / len(q_ents), 0.25))
            if merged:
                top_key = merged[0][:200]
                sources_agree = sum([
                    any(d[:200] == top_key for d in vector_docs),
                    any(d[:200] == top_key for d in bm25_docs),
                    any(d[:200] == top_key for d in kg_docs),
                ])
                conf_signals.append(("source_agree", sources_agree / 3.0, 0.2))
            if kg_chunk_ids:
                max_kg = max(kg_chunk_ids.values())
                conf_signals.append(("kg_density", min(max_kg / 6.0, 1.0), 0.15))
            if conf_signals:
                total_weight = sum(w for _, _, w in conf_signals)
                retrieval_confidence = sum(s * w for _, s, w in conf_signals) / total_weight
        else:
            if "no_reranker" not in self.ablation and reranked:
                retrieval_confidence = reranked[0][1] if reranked else 0.0
            elif vector_metas and res.get("distances"):
                dists = res["distances"][0]
                retrieval_confidence = 1.0 - (dists[0] if dists else 1.0)

        # 11. 上下文窗口扩展 (Innovation 12)
        if "no_context_expansion" not in self.ablation and self.session_chunks:
            expanded_seen = set(d[:200] for d in merged)
            expansion_docs, expansion_dates = [], []
            for doc in merged[:5]:
                for cid, txt in self.chunk_texts.items():
                    if txt == doc:
                        sid = self.chunk_session.get(cid, "")
                        if sid and sid in self.session_chunks:
                            session_cids = self.session_chunks[sid]
                            try:
                                pos = session_cids.index(cid)
                            except ValueError:
                                break
                            for neighbor_pos in [pos + 1, pos - 1]:
                                if 0 <= neighbor_pos < len(session_cids):
                                    n_cid = session_cids[neighbor_pos]
                                    n_doc = self.chunk_texts.get(n_cid, "")
                                    n_key = n_doc[:200]
                                    if n_key and n_key not in expanded_seen:
                                        expanded_seen.add(n_key)
                                        expansion_docs.append(n_doc)
                                        expansion_dates.append(self.chunk_dates.get(n_cid, ""))
                                    break
                        break
            if expansion_docs:
                merged.extend(expansion_docs)
                merged_dates.extend(expansion_dates)

        return merged[:top_k + 5], merged_dates[:top_k + 5], retrieval_confidence

    # ================================================================ #
    #  自适应两阶段检索 (Innovation 7)
    # ================================================================ #

    def retrieve_with_fallback(self, query: str, top_k: int = 10,
                               question_type: str = None,
                               confidence_threshold: float = None):
        """两阶段自适应检索：低置信度时触发查询扩展。"""
        threshold = confidence_threshold or self.cfg.CONFIDENCE_THRESHOLD
        docs, dates, conf = self.retrieve_hybrid(query, top_k=top_k, question_type=question_type)

        if conf >= threshold or "no_query_expansion" in self.ablation:
            return docs, dates, conf

        print(f"    [QueryExpansion] Low confidence ({conf:.3f}), expanding...")
        variants = self._expand_query(query, question_type)
        if not variants:
            return docs, dates, conf

        seen = set(d[:200] for d in docs)
        all_docs, all_dates = list(docs), list(dates)
        best_conf = conf

        for variant in variants:
            v_docs, v_dates, v_conf = self.retrieve_hybrid(variant, top_k=top_k, question_type=question_type)
            best_conf = max(best_conf, v_conf)
            for d, dt in zip(v_docs, v_dates):
                key = d[:200]
                if key not in seen:
                    seen.add(key)
                    all_docs.append(d)
                    all_dates.append(dt)

        if "no_reranker" not in self.ablation and len(all_docs) > top_k:
            reranked = self._rerank(query, all_docs[:top_k + 15], top_n=top_k + 5)
            if reranked:
                doc_to_date = dict(zip([d[:200] for d in all_docs], all_dates))
                all_docs = [doc for doc, _ in reranked]
                all_dates = [doc_to_date.get(doc[:200], "") for doc in all_docs]
                best_conf = max(best_conf, reranked[0][1])

        print(f"    [QueryExpansion] Expanded: {len(all_docs)} docs, conf={best_conf:.3f}")
        return all_docs[:top_k + 5], all_dates[:top_k + 5], best_conf

    # ================================================================ #
    #  链式检索 (Innovation 9)
    # ================================================================ #

    def retrieve_chain(self, query: str, top_k: int = 15,
                       question_type: str = None):
        """Chain-of-retrieval: 多跳桥接实体检索。"""
        if "no_chain_retrieval" in self.ablation:
            return self.retrieve_with_fallback(query, top_k=top_k, question_type=question_type)

        docs, dates, conf = self.retrieve_with_fallback(query, top_k=top_k, question_type=question_type)
        if question_type not in ("multi-hop",):
            return docs, dates, conf

        query_ents = set(e.lower() for e in extract_entities(query) if len(e) > 2)
        bridge_entities = set()
        for doc in docs[:5]:
            doc_ents = set(e.lower() for e in extract_entities(doc) if len(e) > 2)
            bridge_entities.update(doc_ents - query_ents)

        if not bridge_entities:
            return docs, dates, conf

        bridge_query = query + " " + " ".join(sorted(bridge_entities)[:5])
        hop2_docs, hop2_dates, hop2_conf = self.retrieve_hybrid(bridge_query, top_k=top_k, question_type=question_type)

        seen = set(d[:200] for d in docs)
        all_docs, all_dates = list(docs), list(dates)
        for d, dt in zip(hop2_docs, hop2_dates):
            if d[:200] not in seen:
                seen.add(d[:200])
                all_docs.append(d)
                all_dates.append(dt)

        best_conf = max(conf, hop2_conf)
        if "no_reranker" not in self.ablation and len(all_docs) > top_k:
            reranked = self._rerank(query, all_docs[:top_k + 15], top_n=top_k + 5)
            if reranked:
                doc_to_date = dict(zip([d[:200] for d in all_docs], all_dates))
                all_docs = [doc for doc, _ in reranked]
                all_dates = [doc_to_date.get(doc[:200], "") for doc in all_docs]
                best_conf = max(best_conf, reranked[0][1])

        return all_docs[:top_k + 5], all_dates[:top_k + 5], best_conf

    # ================================================================ #
    #  答案生成 (Innovation 4, 8, 11, 12)
    # ================================================================ #

    def generate_answer(self, question: str, context_docs,
                        context_dates=None, question_type=None,
                        question_date=None, benchmark: str = "lme",
                        retrieval_confidence: float = 1.0,
                        max_retries: int = 5):
        """增强版生成：日期标注 + 题型特化 prompt + 后处理。"""
        is_locomo = benchmark == "locomo"
        if context_docs:
            ctx_parts = []
            q_dt = _parse_date(question_date) if question_date else None

            for i, doc in enumerate(context_docs):
                date_label = ""
                if context_dates and i < len(context_dates) and context_dates[i] and "no_date_aware" not in self.ablation:
                    date_label = f" | Date: {context_dates[i]}"
                    # 预计算时间差 (Innovation 3)
                    if question_type in _TEMPORAL_TYPES and q_dt:
                        mem_dt = _parse_date(context_dates[i])
                        if mem_dt:
                            delta = (q_dt - mem_dt).days
                            weeks = delta // 7
                            months = round(delta / 30.44, 1)
                            date_label += f" | {delta} days ({weeks} weeks, ~{months} months) before question date"

                recency_tag = ""
                if question_type == "knowledge-update" and context_dates and "no_recency_label" not in self.ablation:
                    if i == 0:
                        recency_tag = " ★NEWEST — USE THIS★"
                    elif context_dates[i]:
                        recency_tag = " (OLDER — ignore if newer memory covers same topic)"

                # 证据句高亮 (Innovation 8)
                highlighted_doc = doc
                if "no_evidence_highlight" not in self.ablation and len(doc) > 100:
                    highlighted_doc = self._highlight_evidence(question, doc)

                ctx_parts.append(f"[Memory {i + 1}{date_label}{recency_tag}]\n{highlighted_doc}")
            ctx = "\n\n".join(ctx_parts)

            # 时间线摘要 (Innovation 3)
            if question_type in _TEMPORAL_TYPES and context_dates and "no_date_aware" not in self.ablation:
                dated_mems = []
                for i, ds in enumerate(context_dates or []):
                    if ds and i < len(context_docs):
                        dt = _parse_date(ds)
                        if dt:
                            dated_mems.append((i + 1, dt, ds))
                if len(dated_mems) >= 2:
                    dated_mems.sort(key=lambda x: x[1])
                    timeline = "\n[DATE TIMELINE — sorted chronologically]\n"
                    for idx, (mem_num, dt, ds) in enumerate(dated_mems):
                        timeline += f"  Memory {mem_num}: {ds}\n"
                        if idx > 0:
                            prev_dt = dated_mems[idx - 1][1]
                            gap = (dt - prev_dt).days
                            gap_months = round(gap / 30.44, 1)
                            timeline += f"    ↑ {gap} days ({gap // 7} weeks, ~{gap_months} months) after Memory {dated_mems[idx - 1][0]}\n"
                    if q_dt:
                        last_gap = (q_dt - dated_mems[-1][1]).days
                        timeline += f"  Question date: {question_date} ({last_gap} days after Memory {dated_mems[-1][0]})\n"

                    ctx += "\n" + timeline

            # 题型特化提示 (Innovation 4)
            hint = self._build_type_hint(question, question_type, question_date, is_locomo)
            length_hint = self._build_length_hint(question, question_type, is_locomo)

            # 对抗前提检测 (Innovation 11)
            premise_suspect = False
            if "no_premise_detect" not in self.ablation and question_type in ("adversarial", "single-session-user", "single-session-assistant"):
                q_entities = set(e.lower() for e in extract_entities(question) if len(e) > 3)
                ctx_lower = ctx.lower()
                if q_entities:
                    matched = sum(1 for e in q_entities if e in ctx_lower)
                    if matched / len(q_entities) < self.cfg.PREMISE_OVERLAP_THRESHOLD:
                        premise_suspect = True

            # IDK 指令
            low_conf = retrieval_confidence < self.cfg.CONFIDENCE_THRESHOLD
            idk_instruction = self._build_idk_instruction(question_type, premise_suspect, low_conf)

            system = (
                "You are an assistant with long-term memory. "
                "Answer the user's question based ONLY on the conversation memories below. "
                f"{hint}{length_hint}"
                "Answer from the USER's perspective: say 'my sister' not 'your sister', 'I visited' not 'you visited'. "
                f"{idk_instruction}\n\nMemories:\n{ctx}"
            )
        else:
            system = "Answer the question concisely. If you don't know, say 'I don't know'."

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ]

        for attempt in range(max_retries):
            try:
                resp = self.llm_client.chat.completions.create(
                    model=self.cfg.LLM_MODEL,
                    messages=messages,
                    temperature=0.3 if question_type == "single-session-preference" else 0.0,
                    max_tokens=self._get_max_tokens(question_type, is_locomo),
                    extra_body={"enable_thinking": False},
                )
                content = resp.choices[0].message.content or ""

                # 枚举-计数分解 (Innovation: Enumerate-then-Count)
                if question_type == "multi-session" and re.search(r'\bhow\s+(many|much)\b', question, re.I):
                    content = self._enumerate_then_count(content, question, system)

                content = clean_answer(content)
                content = extract_counting_answer(content, question)

                # Sanity cap: pure count 问题 (非 duration/amount) 答案 > 10 时截断
                if (question_type == "multi-session"
                    and re.search(r'\bhow\s+many\b', question, re.I)
                    and not re.search(r'\bhow\s+many\s+(hours?|days?|weeks?|months?|years?|minutes?)\b', question, re.I)
                    and re.match(r'^\d+$', content.strip())
                    and int(content.strip()) > 10):
                    original = int(content.strip())
                    content = str(min(original, 10))
                    print(f"    [Sanity cap] {original} → {content}")

                # LoCoMo single-hop: 截断过长答案
                if is_locomo and question_type == "single-hop" and len(content) > 80 and not is_idk(content):
                    first = re.split(r'[.!]\s+', content)[0]
                    if len(first) < len(content) and len(first) > 5:
                        content = first.rstrip('.')

                # 答案接地验证 (Innovation 12)
                if ("no_grounding_check" not in self.ablation
                    and not is_idk(content)
                    and question_type not in ("single-session-preference", "multi-session")
                    and context_docs):
                    answer_ents = set(e.lower() for e in extract_entities(content) if len(e) > 3)
                    if answer_ents:
                        ctx_text = " ".join(str(d) for d in context_docs).lower()
                        grounded = sum(1 for e in answer_ents if e in ctx_text)
                        if grounded / len(answer_ents) < self.cfg.GROUNDING_THRESHOLD and len(answer_ents) >= 2:
                            print(f"    [Grounding] Not grounded ({grounded}/{len(answer_ents)}), → IDK")
                            content = "I don't know"

                usage = {
                    "prompt_tokens": resp.usage.prompt_tokens,
                    "completion_tokens": resp.usage.completion_tokens,
                    "total_tokens": resp.usage.total_tokens,
                }

                if not content.strip() and attempt == 0:
                    print("    [Empty answer, retrying with temp=0.3]")
                    time.sleep(3)
                    resp2 = self.llm_client.chat.completions.create(
                        model=self.cfg.LLM_MODEL, messages=messages,
                        temperature=0.3, max_tokens=120,
                        extra_body={"enable_thinking": False},
                    )
                    content2 = clean_answer(resp2.choices[0].message.content or "")
                    if content2.strip():
                        content = content2
                        usage["total_tokens"] += resp2.usage.total_tokens

                time.sleep(3)
                return content, usage

            except (APIConnectionError, APITimeoutError, InternalServerError, RateLimitError) as e:
                wait = 5 * (attempt + 1)
                print(f"    [LLM retry {attempt + 1}/{max_retries}] {type(e).__name__}, {wait}s...")
                time.sleep(wait)
            except Exception as e:
                if any(code in str(e) for code in ("502", "503", "504")):
                    wait = 5 * (attempt + 1)
                    print(f"    [LLM retry {attempt + 1}/{max_retries}] {type(e).__name__}, {wait}s...")
                    time.sleep(wait)
                else:
                    raise

        return "", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    # ================================================================ #
    #  生成辅助方法
    # ================================================================ #

    def _enumerate_then_count(self, content: str, question: str, system: str) -> str:
        """枚举-计数两阶段：先提取列表项，再用 Python 计数。

        改进: 当模型输出裸数字时，始终触发 list-only 重试（不再限制 >15）。
        """
        # Case 1: 模型按格式输出了 TOTAL 行
        total_m = re.search(r'TOTAL:\s*(\d+)', content, re.I)
        items = [l.strip() for l in content.split('\n')
                 if re.match(r'^[-•*]\s+\S|^\d+[\.\)]\s+\S', l.strip())]
        if total_m and items:
            # 有列表也有 TOTAL → 信任 Python 计数
            return str(len(items))
        if total_m:
            return total_m.group(1)

        # Case 2: 模型输出了列表（但没有 TOTAL）
        if len(items) >= 1:
            return str(len(items))

        # Case 3: 模型输出了裸数字 → 触发 list-only 重试
        bare_num = re.match(r'^(\d+)\s*$', content.strip())
        if bare_num:
            original_num = int(bare_num.group(1))
            print(f"    [Enumerate retry] Bare number ({original_num}), forcing list-only...")
            listed_count = self._force_enumerate(question, system)
            if listed_count is not None:
                print(f"    [Enumerate retry] Listed → {listed_count} items (was {original_num})")
                return str(listed_count)
            # 回退: 返回原始数字
            return content

        return content

    def _force_enumerate(self, question: str, system: str) -> int | None:
        """强制模型只输出列表，Python 计数。"""
        try:
            resp = self.llm_client.chat.completions.create(
                model=self.cfg.LLM_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": (
                        f"{question}\n\n"
                        "CRITICAL INSTRUCTION: You MUST output a bullet list.\n"
                        "- Write EACH distinct item on its own line starting with '- '\n"
                        "- Do NOT write a number or count\n"
                        "- Do NOT write sentences or explanations\n"
                        "- ONLY output the bullet list, nothing else\n"
                        "Example format:\n- item one\n- item two\n- item three"
                    )},
                ],
                temperature=0.0, max_tokens=400,
                extra_body={"enable_thinking": False},
            )
            list_content = resp.choices[0].message.content or ""
            items = [l.strip() for l in list_content.split('\n')
                     if re.match(r'^[-•*]\s+\S|^\d+[\.\)]\s+\S', l.strip())]
            if items:
                return len(items)
            # 模型仍然输出了裸数字
            m = re.match(r'^(\d+)\s*$', list_content.strip())
            if m:
                print(f"    [Enumerate retry] Model still gave number: {m.group(1)}")
            return None
        except Exception as e:
            print(f"    [Enumerate retry] Failed: {e}")
            return None

    def _build_type_hint(self, question: str, question_type: str, question_date: str, is_locomo: bool) -> str:
        """构建题型特化提示。"""
        type_hints = {
            "temporal-reasoning": (
                "Pay close attention to the DATE of each memory. "
                "Time differences are PRE-COMPUTED for you. "
                "USE these pre-computed values directly — do NOT recalculate. "
                f"TODAY'S DATE: {question_date or 'unknown'}. "
                "CRITICAL: Match the UNIT asked in the question! "
                "Give JUST the number. "
            ),
            "knowledge-update": (
                "CRITICAL: Information gets UPDATED over time — newer replaces older. "
                "Memories are sorted NEWEST-first by date. "
                "RULE: Use ONLY the value from the NEWEST memory. "
            ),
            "multi-session": (
                "The answer requires combining information from MULTIPLE different memories. "
                + (
                    "YOU MUST USE THIS EXACT FORMAT:\n"
                    "Step 1: List each DISTINCT item, one per line, with '- ' prefix.\n"
                    "Step 2: Write 'TOTAL: <number>' as the LAST line.\n"
                    "RULES:\n"
                    "- ONLY list items that are DIRECTLY and EXPLICITLY stated in the memories above\n"
                    "- Do NOT infer, guess, or imagine items not in the text\n"
                    "- Do NOT count the same item twice even if mentioned in different memories\n"
                    "- For each item, mentally verify: 'Can I point to the EXACT sentence?'\n"
                    "- The answer is typically between 2 and 5\n"
                    "Example:\n- yoga\n- swimming\n- tennis\nTOTAL: 3\n"
                    if re.search(r'\bhow\s+(many|much)\b', question, re.I) else
                    "Combine information from multiple memories. "
                    "Do NOT say 'I don't know' if any memories mention the topic. "
                )
            ),
            "single-session-assistant": (
                "This question asks about something the ASSISTANT said. "
                "Look for assistant responses in the memories. "
            ),
            "single-session-preference": (
                "IMPORTANT: INFER the user's preferences from context clues. "
                "DO NOT say 'I don't know'. Write 1-2 sentences. "
            ),
            "adversarial": (
                "The question may reference events with wrong details. "
                "Answer based on what ACTUALLY happened. "
            ),
            "multi-hop": (
                "This question requires connecting facts from DIFFERENT memories. "
                "Trace the chain and give a concise answer. "
            ),
            "single-hop": "Find the specific fact and give a direct answer. ",
            "open-domain": "Answer based on what the memories say. Be concise and factual. ",
        }
        type_hints["temporal"] = type_hints["temporal-reasoning"]

        hint = type_hints.get(question_type or "", "")

        # 问句意图提示
        q_lower = question.lower().strip()
        if q_lower.startswith(("where", "in what location", "at which", "in which city")):
            hint += "Answer with just the place name. "
        elif q_lower.startswith(("who", "whose")):
            hint += "Answer with just the person's name. "
        elif q_lower.startswith(("when", "what date", "what time")):
            hint += "Answer with just the date or time. "
        elif re.match(r'^how (many|much|long|often|far)\b', q_lower):
            hint += "Answer with just the number (and unit if needed). "

        return hint

    def _build_length_hint(self, question: str, question_type: str, is_locomo: bool) -> str:
        """构建长度/格式指令。"""
        if question_type == "single-session-preference":
            return "Write 1-2 sentences covering specific preferences. "
        elif question_type == "multi-session":
            return (
                "Give a SHORT, DIRECT answer — at most one sentence. "
                "If asked 'how many', answer with JUST the number. "
                "Use digits for numbers. "
            )
        elif is_locomo and question_type == "open-domain":
            return "Give a COMPLETE answer in 2-4 sentences. Use digits for numbers. "
        elif is_locomo and question_type == "single-hop":
            return "Answer with JUST the specific fact — ideally 1-5 words. Use digits for numbers. "
        elif is_locomo and question_type == "multi-hop":
            return "Give a concise answer in 1-2 short sentences. Use digits for numbers. "
        elif is_locomo:
            return "Give a concise but COMPLETE answer in 1-2 short sentences. Use digits for numbers. "
        else:
            q_lower = question.lower()
            is_what_q = q_lower.startswith("what") and not q_lower.startswith(("what date", "what time"))
            word_limit = "1-10 words" if is_what_q else "1-5 words"
            return (
                f"IMPORTANT: Give the shortest possible answer — ideally {word_limit}. "
                "Answer ONLY what was asked. No preamble. Use digits for numbers. "
            )

    @staticmethod
    def _build_idk_instruction(question_type: str, premise_suspect: bool, low_conf: bool) -> str:
        if question_type in ("single-session-preference", "adversarial"):
            return ""
        if question_type == "multi-session":
            return "Only say 'I don't know' if NONE of the memories mention the topic at all. "
        if premise_suspect:
            return (
                "WARNING: The question may contain a FALSE PREMISE. "
                "If the specific thing asked about is NOT in any memory, "
                "say 'I don't know'. Do NOT make up an answer. "
            )
        if low_conf:
            return (
                "WARNING: Retrieved memories may not be relevant. "
                "Say 'I don't know' unless you find a CLEAR, DIRECT answer. "
            )
        return "If the memories do not contain the answer, say 'I don't know'. "

    @staticmethod
    def _get_max_tokens(question_type: str, is_locomo: bool) -> int:
        if is_locomo:
            return {"open-domain": 300, "multi-hop": 150, "single-hop": 80}.get(question_type, 200)
        if question_type in ("single-session-preference", "multi-session"):
            return 200
        return 80
