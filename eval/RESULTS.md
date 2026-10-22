# Retrieval results — Labour Law 14/2025 (301 articles)

Two question sets, mostly Egyptian colloquial Arabic, with hand-labelled gold articles:
- `eval/questions.jsonl` — 27 **dev** questions (used while building; numbers are optimistic)
- `eval/test_questions.jsonl` — 40 **held-out** questions, written before tuning anything on them

| Pipeline | Dev Recall@5 | **Held-out Recall@5** | Held-out MRR |
|---|---|---|---|
| BM25 (normalized + light stemming) | 0.70 | 0.50 | 0.35 |
| + colloquial→MSA expansion | 0.81 | 0.50 | 0.35 |
| multilingual-e5-small embeddings | 0.70 | 0.50 | 0.39 |
| **bge-m3** embeddings (int8 ONNX) | 0.85 | 0.72 | 0.64 |
| Hybrid (BM25 + bge-m3, RRF) → **bge-reranker-v2-m3** on top-20 | 1.00 | **0.78** | **0.70** |

Lesson: the colloquial dictionary lifted the dev set to 0.81 but did nothing on held-out questions —
it was overfitting. Only model improvements moved the held-out number.

All models run offline on CPU via ONNX Runtime (`src/onnx_embedder.py`, `src/reranker.py`).
Remaining misses are mostly questions where the key concept is phrased very differently from the
article (e.g. "بيتنمر عليا" vs. "يُحظر التحرش أو التنمر"). Next: LLM query rewriting
colloquial→legal MSA before retrieval, and a larger held-out set.
