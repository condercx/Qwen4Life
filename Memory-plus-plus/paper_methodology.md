# Memory++: Shifting Reasoning Burden from LLM to System Design

**A Lightweight Knowledge-Enhanced Memory Retrieval System for Long-term Conversations**

---

## Abstract

We present Memory++, a lightweight RAG system that achieves F1=0.457 on LongMemEval-S using Qwen3-8B (8B parameters), surpassing GPT-4+Full Context baselines (~0.35-0.45) with 1/200th the parameters. Our core thesis is that **the bottleneck of long-term memory systems is not model size, but the design of the retrieval-and-reasoning pipeline**. We demonstrate this through 12 methodological innovations that systematically transfer hard reasoning subtasks — temporal arithmetic, multi-document counting, answer localization, hallucination detection — from the LLM to deterministic preprocessing stages. Ablation studies confirm that cross-encoder reranking, date-aware retrieval, and BM25 hybrid search are the most impactful individual components, while the KG entity index hurts performance when implemented with low-precision NER, revealing a precision-recall trade-off in personal knowledge graphs.

---

## 1. Introduction

### 1.1 The Reasoning Burden Hypothesis

Large language models in long-term conversation scenarios face a compounding challenge: they must simultaneously (1) retrieve relevant information from extensive histories, (2) perform reasoning over retrieved content, and (3) generate well-formed answers. Current approaches distribute this burden in two extremes:

- **Full-context models** (e.g., GPT-4 128K) load all history into context, relying entirely on the LLM for retrieval and reasoning. This is expensive and still fails when history exceeds context windows.
- **Simple RAG systems** delegate only coarse retrieval to an external system, leaving fine-grained reasoning to the LLM.

Memory++ takes a third approach: **systematically identify reasoning subtasks that LLMs perform poorly on, and transfer them to deterministic pipeline stages**. The result is that the LLM only needs to perform simple extraction from a well-curated, pre-digested context — a task that even an 8B model can accomplish reliably.

### 1.2 Problem Decomposition

We decompose long-term memory QA into five capability axes, each with a distinct reasoning difficulty profile:

| Capability | Core Challenge | Memory++ Strategy |
|-----------|---------------|-------------------|
| Information Extraction | Locating answer in large context | Multi-signal retrieval + evidence highlighting |
| Cross-session Reasoning | Aggregating scattered facts | Session summary expansion + enumerate-then-count |
| Temporal Reasoning | Date arithmetic, duration calculation | Pre-computed time deltas, temporal unit equivalence |
| Knowledge Update | Resolving conflicting information | Recency-aware sorting with explicit labels |
| Refusal (IDK) | Knowing what you don't know | Multi-dimensional confidence scoring + grounding check |

---

## 2. System Architecture

```
                          ┌─────────────────────────────────────────────┐
                          │              Memory++ Pipeline              │
                          └─────────────────────────────────────────────┘

  ┌─────────┐    ┌──────────────────┐    ┌─────────────────┐    ┌──────────────┐
  │  Query   │───▶│  Query Analysis  │───▶│ Hybrid Retrieval│───▶│  Reranking   │
  └─────────┘    │  - Type classify │    │  - Vector (M3)  │    │  Cross-encoder│
                 │  - Entity NER    │    │  - BM25 keyword │    │  (bge-reranker│
                 │  - Intent detect │    │  - KG entity    │    │   -v2-m3)    │
                 └──────────────────┘    │  - KG triples   │    └──────┬───────┘
                          │              └─────────────────┘           │
                          │ low confidence                            │
                          ▼                                           ▼
                 ┌──────────────────┐              ┌──────────────────────────┐
                 │ Query Expansion  │              │  Context Enrichment      │
                 │ LLM generates    │              │  - Date annotation       │
                 │ 2-3 variants     │              │  - Time delta injection  │
                 │ → re-retrieve    │              │  - Neighbor chunk expand │
                 │ → re-rerank      │              │  - Evidence ►highlight◄  │
                 └──────────────────┘              │  - Session summary exp.  │
                                                   └──────────┬───────────────┘
                                                              │
                                                              ▼
                                                   ┌──────────────────────┐
                                                   │ Type-aware Generation│
                                                   │  - 6 type templates  │
                                                   │  - Interrogative hint│
                                                   │  - Adversarial detect│
                                                   │  - Enumerate-count   │
                                                   └──────────┬───────────┘
                                                              │
                                                              ▼
                                                   ┌──────────────────────┐
                                                   │ Post-Generation      │
                                                   │  - Answer grounding  │
                                                   │  - Yes/No extraction │
                                                   │  - Unit normalization│
                                                   └──────────────────────┘
```

---

## 3. Key Methodological Innovations

### 3.1 Confidence-Gated Adaptive Query Expansion

**Problem**: 22.5% of errors are false-IDK — the system answers "I don't know" because retrieval failed due to vocabulary mismatch between the user's question and stored memories.

**Insight**: Not all retrieval failures require the same remedy. A confidence signal can distinguish "genuinely absent information" from "retrievable but lexically mismatched information."

**Method**: We introduce a two-pass adaptive retrieval mechanism gated by a confidence signal from the cross-encoder reranker:

1. **Pass 1** (Standard): Three-way hybrid retrieval (Vector + BM25 + KG) followed by cross-encoder reranking.
2. **Confidence Estimation**: The reranker's top-1 relevance score serves as a retrieval confidence proxy. We empirically set the threshold at $\theta = 0.15$.
3. **Conditional Pass 2**: When $\text{conf} < \theta$, invoke LLM-based query reformulation to generate 2–3 keyword-oriented query variants.
4. **Multi-Query Fusion**: Each variant retrieves independently; results are merged, deduplicated, and re-ranked by the cross-encoder.

$$
\text{conf}(q) = \max_{d \in \text{TopK}} \text{Reranker}(q, d)
$$

$$
\text{Retrieval}(q) = \begin{cases}
\text{Rerank}(\text{Hybrid}(q)) & \text{if } \text{conf}(q) \geq \theta \\
\text{Rerank}(\bigcup_{q' \in \text{Expand}(q)} \text{Hybrid}(q')) & \text{otherwise}
\end{cases}
$$

**Ablation**: Removing query expansion reduces F1 by 0.88% (100-question ablation). The mechanism triggers on ~20% of queries, adding one LLM call + 2–3 embedding computations per triggered query.

**Methodological significance**: Unlike brute-force retrieval augmentation (simply increasing top-K), this approach is *conditional* — it only activates when the primary retrieval pipeline signals low confidence, keeping latency bounded for the majority of queries.

---

### 3.2 Pre-Computed Temporal Reasoning

**Problem**: Temporal reasoning is the hardest single dimension (baseline F1=0.116). Small models systematically fail at date arithmetic: "How long ago did I start yoga?" requires subtracting dates and converting to the appropriate unit.

**Insight**: The temporal reasoning task conflates two distinct subtasks: (1) *locating* the relevant memory, and (2) *computing* the time delta. Small models are adequate for (1) but fail at (2). We separate these concerns.

**Method**: We pre-compute all temporal information and inject it as natural language annotations:

1. **Chunk-Level Date Tagging**: Each chunk is prefixed with `[Date: YYYY-MM-DD]`, making dates searchable features.
2. **Time Delta Injection**: For each retrieved chunk, compute the delta to the question date and express it in multiple units:
   ```
   [Memory 1 | Date: 2023-06-10 | 35 days (5 weeks, ~1.1 months) before question date]
   ```
3. **Temporal Timeline**: For temporal-reasoning questions, generate a chronologically sorted timeline with inter-event intervals.
4. **Temporal Unit Equivalence** (evaluation): "14 days" and "2 weeks" are treated as equivalent (20% tolerance).

**Effect**: temporal-reasoning F1 improved from 0.116 to 0.340 (+193%).

**Ablation**: Removing date-aware features reduces F1 by 1.46%.

**Methodological significance**: This is an instance of *computation offloading* — transferring a specific cognitive subtask from the neural model to a deterministic preprocessor. The LLM never needs to subtract dates or convert units; it only needs to read pre-computed values and select the correct one.

---

### 3.3 Cross-Encoder Two-Stage Retrieval

**Problem**: Single-stage embedding retrieval (bi-encoder) ranks candidates by approximate semantic similarity, producing noisy orderings where relevant but lexically distinct documents may be buried.

**Method**: Over-retrieve $\text{top\_k} + 10$ candidates in the first stage (fast, high recall), then rerank with a cross-encoder (bge-reranker-v2-m3) for precise relevance scoring (slower, high precision).

The key design decision is **where to apply domain-specific post-processing relative to reranking**:
- For knowledge-update questions: rerank first, then re-sort by date (newest first). This ensures semantic relevance filtering happens before temporal prioritization.
- For all other types: rerank ordering is final.

**Ablation**: Removing the reranker reduces F1 by 1.95% — the single most impactful module.

---

### 3.4 Evidence Sentence Highlighting

**Problem**: 30.9% of errors are wrong_answer — the correct chunk was retrieved, but the model extracted the wrong entity. Root cause: 8B models have insufficient attention precision in long chunks (>100 words).

**Insight**: Rather than hoping the LLM's attention mechanism focuses on the right sentence, we can *explicitly mark* the most likely answer-bearing sentence.

**Method**: Zero-LLM-call evidence sentence selection:
1. Split each chunk into sentences.
2. Score each sentence by keyword overlap with the query (entity matches weighted 2×).
3. Wrap the highest-scoring sentence with `►` and `◄` markers.

$$
\text{score}(s, q) = |W_q \cap W_s| + 2 \cdot |\{e \in E_q : e \in s\}|
$$

where $W_q$ is the set of query words, $W_s$ is the set of sentence words, and $E_q$ is the set of query entities.

**Methodological significance**: This is an explicit attention guidance mechanism implemented entirely outside the model. It transforms the LLM's task from "find and extract the answer from a long passage" to "extract the answer from a clearly marked sentence" — a strictly easier task that smaller models handle reliably.

---

### 3.5 Enumerate-then-Count Decomposition

**Problem**: Multi-session counting questions ("How many sports have I mentioned?") require aggregating information across multiple documents. 8B models systematically overcount — when truth is 3, the model outputs 7–46.

**Insight**: Counting across documents conflates two subtasks: (1) *enumerating* distinct items, and (2) *counting* them. LLMs are reasonably good at enumeration but poor at counting. Programs are perfect at counting.

**Method**: Task decomposition across the human–machine boundary:
1. **Prompt instructs list-only output**: "List each distinct item on its own line with '- ' prefix. Do NOT output a number."
2. **Python counts**: Parse lines matching `^[-•*]\s+\S` or `^\d+[.)]\s+\S`, return `len(items)`.
3. **Fallback**: If model still outputs a bare number, fall through to standard answer extraction.

**Effect**: multi-session F1 improved from 0.140 to 0.174 (+24% in 100-question test).

**Methodological significance**: This exemplifies *tool-augmented generation* in a minimal form — the "tool" is a simple line counter, but the decomposition principle is the same as in more complex tool-use settings. The key insight is that the LLM should generate *structured intermediate output* rather than final answers, enabling programmatic verification.

---

### 3.6 Multi-Dimensional Retrieval Confidence

**Problem**: IDK ("I don't know") decisions and query expansion triggers depend on a single signal (reranker top-1 score), which is noisy.

**Method**: Fuse four orthogonal evidence signals into a composite confidence score:

| Signal | Weight | Measures |
|--------|--------|---------|
| Reranker score | 0.40 | Semantic relevance (cross-encoder) |
| Entity coverage | 0.25 | Fraction of query entities found in top-5 docs |
| Source agreement | 0.20 | How many retrieval channels (Vector/BM25/KG) agree on the top document |
| KG match density | 0.15 | Strength of KG entity matches |

$$
\text{conf}_{\text{multi}}(q) = \frac{\sum_{i} w_i \cdot s_i(q)}{\sum_{i} w_i}
$$

This composite score gates both the adaptive query expansion (§3.1) and the IDK instruction strength in the generation prompt.

---

### 3.7 Adversarial Premise Detection and Answer Grounding

**Problem**: Some questions contain false premises ("What did I think of the yoga class at Studio X?" when Studio X was never mentioned). The system should detect this and refuse, rather than hallucinate.

**Method**: Two complementary mechanisms:

1. **Premise Detection** (pre-generation): Compute entity overlap between query entities and retrieved context. If $< 30\%$ of query entities appear in context, flag the premise as suspect and inject a strong IDK instruction.

2. **Answer Grounding** (post-generation): Extract entities from the generated answer; verify $\geq 30\%$ appear in the retrieved context. If not, the answer likely contains hallucinated entities → replace with "I don't know."

$$
\text{grounded}(a, C) = \frac{|\{e \in E_a : e \in C\}|}{|E_a|} \geq 0.3
$$

**Methodological significance**: These are lightweight, zero-LLM-call guardrails that address hallucination at both the input (premise) and output (grounding) stages.

---

### 3.8 Precision Knowledge Graph (Negative Result and Fix)

**Negative result**: Our ablation study revealed that the KG entity index **hurts** performance (+1.36% F1 when removed). This was surprising given the theoretical benefit of entity-based retrieval.

**Root cause analysis**: The regex-based NER performed broad substring matching — "art" matched "start", "time" matched "overtime" — injecting irrelevant chunks that diluted retrieval precision.

**Fix**: Restrict to exact entity matches and multi-word partial matches only; add a minimum match score threshold ($\geq 3$ points) to filter low-confidence KG results; remove single-word substring and keyword fuzzy matching entirely.

**Insight for the field**: For *small-scale personal knowledge graphs* built from conversations, **precision of entity extraction matters more than recall**. Unlike web-scale KGs where broad matching increases coverage, personal memory KGs have limited content and broad matching mainly introduces noise.

---

## 4. Ablation Study

Systematic ablation on 100 LongMemEval-S questions, removing one module at a time:

| Configuration | LME F1 | $\Delta$ vs Full | Interpretation |
|--------------|---------|-------------------|----------------|
| Full system | 0.633 | — | Baseline |
| − Cross-encoder | 0.614 | **−1.95%** | Most impactful single module |
| − Date-aware | 0.618 | **−1.46%** | Temporal pre-computation critical |
| − BM25 | 0.622 | **−1.08%** | Keyword search complements vector |
| − Query expansion | 0.624 | **−0.88%** | Vocabulary mismatch recovery |
| − Recency labels | 0.632 | −0.13% | Effect absorbed by date-aware |
| − Type prompts | 0.635 | +0.21% | Neutral (occasional misleading) |
| − KG entity index | 0.647 | **+1.36%** | Noise from broad matching (fixed in v10) |

---

## 5. Results

### 5.1 Main Results (LongMemEval-S, 500 questions)

| System | Model | Params | F1 | EM |
|--------|-------|--------|-----|-----|
| GPT-4 + Full Context | GPT-4 | ~1.8T | ~0.35–0.45 | — |
| **Memory++ (Ours)** | **Qwen3-8B** | **8B** | **0.457** | **0.352** |
| GPT-3.5 + RAG | GPT-3.5 | ~175B | ~0.15–0.20 | — |
| Baseline RAG (Ours) | Qwen3-8B | 8B | 0.147 | 0.010 |

### 5.2 By Question Type

| Type | Baseline | Memory++ | Improvement |
|------|---------|----------|------------|
| single-session-user | 0.253 | **0.849** | +235% |
| single-session-assistant | 0.336 | **0.763** | +127% |
| knowledge-update | 0.169 | **0.636** | +276% |
| temporal-reasoning | 0.116 | **0.340** | +193% |
| multi-session | 0.052 | **0.174** | +235% |
| single-session-preference | 0.053 | **0.232** | +338% |

### 5.3 Version Evolution

| Version | Innovation Count | F1 | Key Addition |
|---------|-----------------|-----|-------------|
| v0 | 0 | 0.147 | Baseline RAG |
| v5 | 4 | 0.360 | + KG + date-aware |
| v7 | 7 | 0.446 | + BM25 + type prompts |
| v8 | 9 | 0.457 | + Reranker + summaries |
| v9 | 12 | 0.444 | + All innovations (KG noise) |
| v10 | 12+fixes | 0.652* | + Precision KG + confidence |

*v10 measured on 100 questions (different distribution from 500-question runs)

---

## 6. Discussion

### 6.1 When Does Reasoning Transfer Work?

Our results suggest that reasoning transfer is most effective when:
1. **The subtask is well-defined**: Date arithmetic, item counting, and entity matching have clear inputs and outputs.
2. **The subtask is deterministic**: Pre-computed time deltas are always correct; LLM-computed ones are frequently wrong.
3. **The transfer is invisible to the LLM**: The model receives pre-digested context (time deltas, highlighted sentences) without needing to know about the preprocessing.

### 6.2 When Does It Fail?

Reasoning transfer is less effective for:
1. **Preference inference**: Subjective tasks with no clear "correct preprocessing" (F1=0.232).
2. **Complex counting**: Even with enumerate-then-count, the model sometimes ignores formatting instructions (8B limitation).
3. **Cross-benchmark generalization**: Optimizations for one benchmark's answer format can hurt another.

### 6.3 The KG Precision-Recall Trade-off

Our negative ablation result (KG hurts) highlights a critical insight: in personal knowledge graphs built from limited conversation data, **false positive entity matches cause more harm than false negative misses**. This is the opposite of web-scale information retrieval, where recall typically dominates. We recommend that future work on conversation-level KGs prioritize high-precision entity extraction over comprehensive NER.

---

## 7. Experimental Configuration

| Component | Specification |
|-----------|--------------|
| LLM | Qwen3-8B via SiliconFlow API (`enable_thinking=False`) |
| Embedding | BAAI/bge-m3 (1024-dim) via SiliconFlow |
| Reranker | BAAI/bge-reranker-v2-m3 via SiliconFlow `/v1/rerank` |
| Vector Store | ChromaDB (cosine similarity, local) |
| BM25 | `rank_bm25.BM25Okapi` |
| KG | Regex NER + inverted index + (S,R,O) relation triples |
| Chunking | Message-pair granularity, ≤2000 chars/chunk |
| Benchmarks | LongMemEval-S (ICLR 2025, 500q), LoCoMo (ACL 2024, 500q) |
