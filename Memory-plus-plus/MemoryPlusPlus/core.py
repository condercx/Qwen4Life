"""
Memory++ Core: BenchmarkRAGPlusPlus

This is the reference implementation of the Memory++ retrieval-augmented generation system.
The full implementation lives in ../benchmark_eval_kg.py; this module documents the
key architectural decisions and algorithms.

Usage:
    from MemoryPlusPlus.core import MemoryPlusPlus

    mp = MemoryPlusPlus(api_key="...", base_url="...",)
    mp.index_sessions(sessions)  # Index conversation sessions
    docs, dates, conf = mp.retrieve(query, question_type="single-session-user")
    answer = mp.generate_answer(query, docs, dates, question_type="single-session-user")
"""

import os
import re
import json
import time
from collections import defaultdict
from datetime import datetime

import numpy as np
import chromadb
from chromadb.config import Settings
from openai import OpenAI
from rank_bm25 import BM25Okapi
import requests

from .config import (
    API_KEY, BASE_URL, LLM_MODEL, EMBED_MODEL, EMBED_DIMS,
    RERANKER_MODEL, DEFAULT_TOP_K, CHUNK_MAX_CHARS,
    CONFIDENCE_THRESHOLD, GROUNDING_THRESHOLD, PREMISE_OVERLAP_THRESHOLD,
    CHROMA_DIR, COLLECTION_PREFIX,
)
from .utils import (
    extract_entities, extract_relation_triples, parse_date,
    normalize_answer, token_f1, _is_idk, _STOP_ENTITIES,
)


# Question types that need temporal handling
_TEMPORAL_TYPES = {"temporal-reasoning", "knowledge-update"}


class MemoryPlusPlus:
    """Memory++ RAG System: Knowledge-enhanced retrieval for long-term conversations.

    Architecture:
        Query → [Adaptive Query Expansion] → [Three-way Hybrid Retrieval]
              → [Cross-Encoder Rerank] → [Context Expansion]
              → [Evidence Highlighting] → [Type-aware Prompt]
              → [LLM Generation] → [Answer Grounding] → [Post-processing]

    12 Innovations:
        1.  Three-way hybrid retrieval (Vector + BM25 + KG)
        2.  Cross-encoder two-stage reranking
        3.  Date-aware retrieval & reasoning
        4.  Type-aware generation strategy
        5.  Session summary hierarchical retrieval
        6.  Knowledge update temporal sorting
        7.  Adaptive query expansion (confidence-gated)
        8.  Evidence sentence highlighting
        9.  Chain-of-retrieval (multi-hop bridge entities)
        10. Relation-aware knowledge graph (S,R,O triples)
        11. Adversarial premise detection
        12. Answer grounding verification + context expansion
    """

    def __init__(self, api_key: str = None, base_url: str = None,
                 chroma_dir: str = None, ablation: str = ""):
        """Initialize Memory++ system.

        Args:
            api_key: SiliconFlow API key
            base_url: API base URL
            chroma_dir: ChromaDB storage directory
            ablation: Comma-separated ablation flags (e.g., "no_bm25,no_kg")
        """
        self.api_key = api_key or API_KEY
        self.base_url = base_url or BASE_URL
        self.chroma_dir = chroma_dir or CHROMA_DIR
        self.ablation = set(ablation.split(",")) if ablation else set()

        # LLM client
        self.llm_client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        # ChromaDB
        self.chroma_client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=self.chroma_dir,
            anonymized_telemetry=False,
        ))
        self.collection = None

        # Data structures
        self.kg_entities: dict[str, set[str]] = {}         # entity → {chunk_ids}
        self.kg_triples: list[tuple[str, str, str, str]] = []  # [(subj, rel, obj, chunk_id)]
        self.kg_entity_relations: dict[str, set[str]] = {} # entity → {related chunk_ids via triples}
        self.chunk_texts: dict[str, str] = {}              # chunk_id → text
        self.chunk_dates: dict[str, str] = {}              # chunk_id → date string
        self.chunk_session: dict[str, str] = {}            # chunk_id → session_id
        self.session_summaries: dict[str, str] = {}        # session_id → extractive summary
        self.session_chunks: dict[str, list[str]] = {}     # session_id → [chunk_ids]
        self.bm25 = None
        self.bm25_chunk_ids: list[str] = []

    # ================================================================ #
    #  Embedding
    # ================================================================ #

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts using bge-m3."""
        resp = self.llm_client.embeddings.create(
            model=EMBED_MODEL, input=texts
        )
        return [d.embedding for d in resp.data]

    # ================================================================ #
    #  Reranking (Innovation 2)
    # ================================================================ #

    def _rerank(self, query: str, documents: list[str], top_n: int = 10) -> list[tuple[str, float]]:
        """Cross-encoder reranking via SiliconFlow /v1/rerank API.

        Uses bge-reranker-v2-m3 for precise query-document relevance scoring.
        """
        if "no_reranker" in self.ablation:
            return [(d, 0.5) for d in documents[:top_n]]
        try:
            resp = requests.post(
                f"{self.base_url.rstrip('/').replace('/v1', '')}/v1/rerank",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": RERANKER_MODEL,
                    "query": query,
                    "documents": documents,
                    "top_n": top_n,
                    "return_documents": True,
                },
                timeout=30,
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])
            return [(r["document"]["text"], r["relevance_score"]) for r in results]
        except Exception as e:
            print(f"  [Reranker warning] {e}")
            return [(d, 0.5) for d in documents[:top_n]]

    # ================================================================ #
    #  Indexing
    # ================================================================ #

    def index_sessions(self, sessions: list[dict]):
        """Index conversation sessions into all retrieval stores.

        For each session:
        1. Chunk into message pairs (user+assistant)
        2. Extract entities → KG inverted index
        3. Extract relation triples → KG relation index
        4. Embed chunks → ChromaDB
        5. Tokenize → BM25 index
        6. Generate session summary (extractive, zero LLM calls)

        Args:
            sessions: List of session dicts with "session_id", "date", "messages"
        """
        all_texts, all_ids, all_metas = [], [], []
        bm25_corpus = []

        for sess in sessions:
            sid = sess["session_id"]
            date = sess.get("date", "")
            messages = sess.get("messages", [])
            self.session_chunks[sid] = []
            all_ents = set()

            # Chunk messages into pairs
            chunks = self._chunk_messages(messages, sid, date)

            for chunk_id, chunk_text, chunk_date in chunks:
                self.chunk_texts[chunk_id] = chunk_text
                self.chunk_dates[chunk_id] = chunk_date
                self.chunk_session[chunk_id] = sid
                self.session_chunks[sid].append(chunk_id)

                # KG entity extraction
                entities = extract_entities(chunk_text)
                for ent in entities:
                    ent_lower = ent.lower().strip()
                    if ent_lower not in _STOP_ENTITIES and len(ent_lower) > 2:
                        if ent_lower not in self.kg_entities:
                            self.kg_entities[ent_lower] = set()
                        self.kg_entities[ent_lower].add(chunk_id)
                        all_ents.add(ent_lower)

                # KG relation triple extraction (Innovation 10)
                for subj, rel, obj in extract_relation_triples(chunk_text):
                    self.kg_triples.append((subj, rel, obj, chunk_id))
                    for ent_key in (subj, obj):
                        if ent_key and len(ent_key) > 2:
                            if ent_key not in self.kg_entity_relations:
                                self.kg_entity_relations[ent_key] = set()
                            self.kg_entity_relations[ent_key].add(chunk_id)

                # Prepare for embedding
                prefix = f"[Date: {chunk_date}] " if chunk_date else ""
                all_texts.append(prefix + chunk_text)
                all_ids.append(chunk_id)
                all_metas.append({"session_id": sid, "date": chunk_date})

                # BM25 tokenization
                bm25_corpus.append(chunk_text.lower().split())
                self.bm25_chunk_ids.append(chunk_id)

            # Session summary (Innovation 5)
            summary = f"[Session {sid}]"
            if date:
                summary += f" [Date: {date}]"
            summary += f" Topics: {', '.join(sorted(all_ents)[:20])}"
            self.session_summaries[sid] = summary

        # Embed and store in ChromaDB
        if all_texts:
            self.collection = self.chroma_client.get_or_create_collection(
                name=f"{COLLECTION_PREFIX}_memory",
                metadata={"hnsw:space": "cosine"},
            )
            # Batch embed
            batch_size = 32
            for i in range(0, len(all_texts), batch_size):
                batch_texts = all_texts[i:i+batch_size]
                batch_ids = all_ids[i:i+batch_size]
                batch_metas = all_metas[i:i+batch_size]
                embeddings = self._embed_batch(batch_texts)
                self.collection.add(
                    documents=batch_texts,
                    embeddings=embeddings,
                    ids=batch_ids,
                    metadatas=batch_metas,
                )

        # Build BM25 index
        if bm25_corpus:
            self.bm25 = BM25Okapi(bm25_corpus)

    def _chunk_messages(self, messages: list[dict], session_id: str, date: str
                        ) -> list[tuple[str, str, str]]:
        """Chunk messages into pairs (user+assistant), respecting max chunk size.

        Returns: list of (chunk_id, chunk_text, chunk_date)
        """
        chunks = []
        buffer = ""
        idx = 0
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            line = f"{role}: {content}\n"
            if len(buffer) + len(line) > CHUNK_MAX_CHARS and buffer:
                chunk_id = f"{session_id}_chunk_{idx}"
                chunks.append((chunk_id, buffer.strip(), date))
                idx += 1
                buffer = ""
            buffer += line
            # End chunk at assistant message (pair boundary)
            if role == "assistant" and buffer:
                chunk_id = f"{session_id}_chunk_{idx}"
                chunks.append((chunk_id, buffer.strip(), date))
                idx += 1
                buffer = ""
        if buffer.strip():
            chunk_id = f"{session_id}_chunk_{idx}"
            chunks.append((chunk_id, buffer.strip(), date))
        return chunks

    # ================================================================ #
    #  Retrieval Pipeline
    # ================================================================ #

    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K,
                 question_type: str = None) -> tuple[list[str], list[str], float]:
        """Full retrieval pipeline with all innovations.

        Pipeline:
        1. Three-way hybrid retrieval (vector + BM25 + KG)
        2. Date-aware sorting for temporal types
        3. Cross-encoder reranking
        4. Context window expansion (neighbor chunks)
        5. Adaptive query expansion if low confidence

        Returns: (documents, dates, retrieval_confidence)
        """
        docs, dates, conf = self._retrieve_hybrid(query, top_k, question_type)

        # Adaptive query expansion (Innovation 7)
        if conf < CONFIDENCE_THRESHOLD and "no_query_expansion" not in self.ablation:
            variants = self._expand_query(query, question_type)
            if variants:
                all_docs, all_dates = list(docs), list(dates)
                seen = set(d[:200] for d in docs)
                for variant in variants:
                    v_docs, v_dates, _ = self._retrieve_hybrid(variant, top_k, question_type)
                    for d, dt in zip(v_docs, v_dates):
                        if d[:200] not in seen:
                            seen.add(d[:200])
                            all_docs.append(d)
                            all_dates.append(dt)
                # Re-rerank the expanded pool
                if len(all_docs) > top_k:
                    reranked = self._rerank(query, all_docs, top_n=top_k + 5)
                    doc_to_date = dict(zip([d[:200] for d in all_docs], all_dates))
                    docs = [d for d, _ in reranked]
                    dates = [doc_to_date.get(d[:200], "") for d, _ in reranked]
                    conf = reranked[0][1] if reranked else conf

        # Chain-of-retrieval for multi-hop (Innovation 9)
        if question_type == "multi-hop" and "no_chain_retrieval" not in self.ablation:
            docs, dates, conf = self._chain_of_retrieval(query, docs, dates, conf, top_k)

        return docs, dates, conf

    def _retrieve_hybrid(self, query: str, top_k: int, question_type: str = None
                         ) -> tuple[list[str], list[str], float]:
        """Three-way hybrid retrieval: Vector + BM25 + KG entity matching.

        Innovation 1: Three complementary retrieval signals
        Innovation 3: Date-aware sorting for temporal types
        Innovation 2: Cross-encoder reranking
        Innovation 12: Context window expansion
        """
        if self.collection is None or self.collection.count() == 0:
            return [], [], 0.0

        count = self.collection.count()

        # 1. Vector retrieval
        k = min(top_k + (5 if question_type in _TEMPORAL_TYPES else 0), count)
        emb = self._embed_batch([query])[0]
        res = self.collection.query(
            query_embeddings=[emb], n_results=k,
            include=["documents", "metadatas", "distances"]
        )
        vector_docs = res["documents"][0] if res["documents"] else []
        vector_metas = res["metadatas"][0] if res["metadatas"] else []

        # 2. KG entity matching
        query_entities = extract_entities(query)
        query_words = set(query.lower().split()) - _STOP_ENTITIES
        kg_chunk_ids: dict[str, int] = defaultdict(int)

        if "no_kg" not in self.ablation:
            for ent in query_entities:
                ent_lower = ent.lower().strip()
                if ent_lower in self.kg_entities:
                    for cid in self.kg_entities[ent_lower]:
                        kg_chunk_ids[cid] += 3  # exact match weight
                for stored_ent, cids in self.kg_entities.items():
                    if len(ent_lower) >= 3 and (ent_lower in stored_ent or stored_ent in ent_lower):
                        for cid in cids:
                            kg_chunk_ids[cid] += 1

            # Relation triple matching (Innovation 10)
            if self.kg_entity_relations:
                query_ent_lower = set(e.lower() for e in query_entities if len(e) > 2) | \
                                  set(w for w in query_words if len(w) >= 4)
                for ent in query_ent_lower:
                    if ent in self.kg_entity_relations:
                        for cid in self.kg_entity_relations[ent]:
                            kg_chunk_ids[cid] += 2

        kg_sorted = sorted(kg_chunk_ids.items(), key=lambda x: -x[1])[:top_k]
        kg_docs = [self.chunk_texts[cid] for cid, _ in kg_sorted if cid in self.chunk_texts]

        # 3. BM25 keyword retrieval
        bm25_docs = []
        if "no_bm25" not in self.ablation and self.bm25 is not None:
            scores = self.bm25.get_scores(query.lower().split())
            top_indices = scores.argsort()[-top_k:][::-1]
            for idx in top_indices:
                if scores[idx] > 0 and idx < len(self.bm25_chunk_ids):
                    cid = self.bm25_chunk_ids[idx]
                    if cid in self.chunk_texts:
                        bm25_docs.append(self.chunk_texts[cid])

        # 4. Merge and deduplicate (vector-first ordering)
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

        # 5. Date-aware sorting for temporal types (Innovation 3)
        if question_type in (_TEMPORAL_TYPES | {"knowledge-update"}) and any(merged_dates):
            dated_items = [(d, ds, parse_date(ds)) for d, ds in zip(merged, merged_dates)]
            dated_items.sort(key=lambda x: x[2] or datetime.min, reverse=True)
            merged = [d[0] for d in dated_items]
            merged_dates = [d[1] for d in dated_items]

        # 6. Cross-encoder reranking (Innovation 2)
        reranked = []
        if "no_reranker" not in self.ablation and len(merged) > top_k:
            candidates = merged[:top_k + 10]
            candidate_dates = merged_dates[:top_k + 10]
            reranked = self._rerank(query, candidates, top_n=top_k + 5)
            doc_to_date = dict(zip([d[:200] for d in candidates], candidate_dates))
            merged = [doc for doc, _ in reranked]
            merged_dates = [doc_to_date.get(doc[:200], "") for doc, _ in reranked]

        # Retrieval confidence
        retrieval_confidence = 0.0
        if reranked:
            retrieval_confidence = reranked[0][1]
        elif res.get("distances") and res["distances"][0]:
            retrieval_confidence = 1.0 - res["distances"][0][0]

        # 7. Context window expansion (Innovation 12)
        if "no_context_expansion" not in self.ablation and self.session_chunks:
            expanded_seen = set(d[:200] for d in merged)
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
                                        merged.append(n_doc)
                                        merged_dates.append(self.chunk_dates.get(n_cid, ""))
                                    break
                        break

        return merged[:top_k + 5], merged_dates[:top_k + 5], retrieval_confidence

    def _expand_query(self, query: str, question_type: str = None) -> list[str]:
        """LLM-based query expansion (Innovation 7).

        Generates 2-3 alternative keyword queries when retrieval confidence is low.
        """
        if "no_query_expansion" in self.ablation:
            return []
        type_hint = f" (question type: {question_type})" if question_type else ""
        try:
            resp = self.llm_client.chat.completions.create(
                model=LLM_MODEL,
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
            variants = [l.strip().strip('-').strip('*').strip()
                        for l in lines if l.strip() and len(l.strip()) > 5]
            return variants[:3]
        except Exception as e:
            print(f"  [QueryExpansion warning] {e}")
            return []

    def _chain_of_retrieval(self, query: str, docs: list[str], dates: list[str],
                            conf: float, top_k: int
                            ) -> tuple[list[str], list[str], float]:
        """Chain-of-Retrieval for multi-hop questions (Innovation 9).

        1. Extract bridge entities from Hop-1 results (not in query)
        2. Augmented Hop-2 retrieval with bridge entities
        3. Merge and re-rerank
        """
        query_ents = set(e.lower() for e in extract_entities(query) if len(e) > 2)
        bridge_entities = set()
        for doc in docs[:5]:
            doc_ents = set(e.lower() for e in extract_entities(doc) if len(e) > 2)
            bridge_entities.update(doc_ents - query_ents)

        if not bridge_entities:
            return docs, dates, conf

        bridge_query = query + " " + " ".join(sorted(bridge_entities)[:5])
        hop2_docs, hop2_dates, hop2_conf = self._retrieve_hybrid(bridge_query, top_k)

        # Merge
        seen = set(d[:200] for d in docs)
        for d, dt in zip(hop2_docs, hop2_dates):
            if d[:200] not in seen:
                seen.add(d[:200])
                docs.append(d)
                dates.append(dt)

        # Re-rerank merged pool
        if len(docs) > top_k:
            reranked = self._rerank(query, docs, top_n=top_k + 5)
            doc_to_date = dict(zip([d[:200] for d in docs], dates))
            docs = [d for d, _ in reranked]
            dates = [doc_to_date.get(d[:200], "") for d, _ in reranked]
            conf = reranked[0][1] if reranked else conf

        return docs, dates, conf

    # ================================================================ #
    #  Evidence Highlighting (Innovation 8)
    # ================================================================ #

    def _highlight_evidence(self, question: str, doc: str) -> str:
        """Mark the most relevant sentence in a chunk with ► ◄ markers."""
        sentences = re.split(r'(?<=[.!?])\s+|\n+', doc)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        if len(sentences) < 2:
            return doc
        q_words = set(question.lower().split()) - _STOP_ENTITIES
        q_entities = set(e.lower() for e in extract_entities(question) if len(e) > 2)
        query_terms = q_words | q_entities
        best_score, best_idx = 0, -1
        for idx, sent in enumerate(sentences):
            sent_lower = sent.lower()
            sent_words = set(sent_lower.split())
            score = len(query_terms & sent_words)
            for ent in q_entities:
                if ent in sent_lower:
                    score += 2
            if score > best_score:
                best_score, best_idx = score, idx
        if best_score < 2 or best_idx < 0:
            return doc
        highlighted = sentences[best_idx]
        return doc.replace(highlighted, f"►{highlighted}◄")

    # ================================================================ #
    #  Answer Generation
    # ================================================================ #

    def generate_answer(self, question: str, context_docs: list[str],
                        context_dates: list[str] = None,
                        question_type: str = None,
                        retrieval_confidence: float = 1.0) -> str:
        """Generate answer using type-aware prompting with all post-processing.

        Innovations used:
        - Innovation 4: Type-aware generation
        - Innovation 8: Evidence highlighting
        - Innovation 11: Adversarial premise detection
        - Innovation 12: Answer grounding verification

        Args:
            question: The user's question
            context_docs: Retrieved documents
            context_dates: Corresponding dates
            question_type: One of the 6 LME question types
            retrieval_confidence: From retrieval pipeline

        Returns:
            Generated answer string
        """
        if not context_docs:
            return "I don't know"

        # Apply evidence highlighting
        highlighted_docs = [self._highlight_evidence(question, doc) for doc in context_docs]

        # Build context string with date annotations
        ctx_parts = []
        for i, (doc, date) in enumerate(zip(highlighted_docs, context_dates or [""] * len(highlighted_docs))):
            header = f"[Memory {i+1}"
            if date:
                header += f" | Date: {date}"
            header += "]"
            ctx_parts.append(f"{header}\n{doc}")
        ctx = "\n\n".join(ctx_parts)

        # Adversarial premise detection (Innovation 11)
        premise_suspect = False
        if "no_premise_detect" not in self.ablation and question_type in ("adversarial", "single-session-user", "single-session-assistant"):
            q_entities = set(e.lower() for e in extract_entities(question) if len(e) > 3)
            ctx_lower = ctx.lower()
            if q_entities:
                matched = sum(1 for e in q_entities if e in ctx_lower)
                if matched / len(q_entities) < PREMISE_OVERLAP_THRESHOLD:
                    premise_suspect = True

        # Type-aware prompt construction (Innovation 4)
        system_prompt = self._build_system_prompt(question_type, premise_suspect,
                                                  retrieval_confidence)

        user_prompt = f"Memories:\n{ctx}\n\nQuestion: {question}"

        # Generate
        try:
            resp = self.llm_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=200, temperature=0.1,
                extra_body={"enable_thinking": False},
            )
            content = resp.choices[0].message.content.strip()
        except Exception as e:
            return f"Error: {e}"

        # Answer grounding verification (Innovation 12)
        if ("no_grounding_check" not in self.ablation
            and not _is_idk(content)
            and context_docs):
            answer_ents = set(e.lower() for e in extract_entities(content) if len(e) > 3)
            if answer_ents:
                ctx_text = " ".join(str(d) for d in context_docs).lower()
                grounded = sum(1 for e in answer_ents if e in ctx_text)
                if len(answer_ents) >= 2 and grounded / len(answer_ents) < GROUNDING_THRESHOLD:
                    content = "I don't know"

        return content

    def _build_system_prompt(self, question_type: str, premise_suspect: bool,
                             retrieval_confidence: float) -> str:
        """Build type-specific system prompt (Innovation 4)."""
        base = ("You are an assistant with long-term memory. "
                "Answer the user's question based ONLY on the conversation memories below. "
                "Answer from the USER's perspective: say 'my sister' not 'your sister'. ")

        # Type-specific instructions
        type_hints = {
            "knowledge-update": (
                "Memories sorted NEWEST-first. Use ONLY the newest value for the answer. "
                "Older memories are for context only. "
            ),
            "temporal-reasoning": (
                "USE the pre-computed time values in [brackets]. Match the UNIT asked for. "
                "Do NOT recalculate dates. "
            ),
            "multi-session": (
                "Scan EVERY memory carefully. Count ALL distinct items mentioned. "
                "List each item, then count. "
            ),
            "single-session-preference": (
                "INFER preferences from what was said. Write 1-2 sentences. "
            ),
        }
        hint = type_hints.get(question_type, "")

        # IDK instruction based on confidence and premise
        if question_type in ("single-session-preference", "adversarial"):
            idk = ""
        elif premise_suspect:
            idk = ("WARNING: The question may contain a FALSE PREMISE. "
                   "If the specific thing asked about is NOT in any memory, "
                   "say 'I don't know'. Do NOT make up an answer. ")
        elif retrieval_confidence < CONFIDENCE_THRESHOLD:
            idk = ("WARNING: Retrieved memories may not be relevant. "
                   "Say 'I don't know' unless you find a CLEAR, DIRECT answer. ")
        else:
            idk = "If the memories do not contain the answer, say 'I don't know'. "

        return base + hint + idk
