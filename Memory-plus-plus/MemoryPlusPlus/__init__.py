"""
Memory++: Lightweight Knowledge-Enhanced Memory Retrieval for Long-term Conversations

A methodology-first RAG system that shifts reasoning burden from LLM to system design.
Built on Qwen3-8B (8B params), achieves F1=0.457 on LongMemEval-S, surpassing GPT-4+FullContext.

12 Core Innovations:
  1. Three-way Hybrid Retrieval (Vector + BM25 + KG)
  2. Cross-encoder Two-stage Reranking
  3. Date-aware Retrieval & Reasoning
  4. Type-aware Generation Strategy
  5. Session Summary Hierarchical Retrieval
  6. Knowledge Update Temporal Sorting
  7. Adaptive Query Expansion
  8. Evidence Sentence Highlighting
  9. Chain-of-Retrieval (Multi-hop)
  10. Relation-aware Knowledge Graph
  11. Adversarial Premise Detection
  12. Answer Grounding Verification + Context Expansion
"""

__version__ = "0.9.0"
__author__ = "Memory++ Team"
