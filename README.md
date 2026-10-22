# Arabic Legal RAG — مساعد قانون العمل المصري

Ask a question about **Egyptian Labour Law No. 14 of 2025** in Egyptian colloquial Arabic or MSA, and get the relevant articles (with citations) and a grounded answer.

> "ينفع الشركة ترفدني من غير تحقيق؟" → **مادة 141, مادة 148**

## The problem
People ask in colloquial Arabic ("ترفدني"), the law is written in formal legal Arabic ("فصل العامل"). Keyword search alone misses most questions.

## Results (40 held-out questions, never used for tuning)

| Pipeline | Recall@5 | MRR |
|---|---|---|
| BM25 (normalized Arabic + light stemming) | 0.50 | 0.35 |
| + colloquial→MSA dictionary | 0.50 | 0.35 |
| multilingual-e5-small | 0.50 | 0.39 |
| **bge-m3** dense retrieval | 0.72 | 0.64 |
| **Hybrid (BM25 + bge-m3) → bge-reranker-v2-m3** | **0.78** | **0.70** |

The colloquial dictionary raised the *dev* set from 0.70 to 0.81 but did nothing on held-out questions — it was overfitting. That is why the project reports held-out numbers only. Details: [`eval/RESULTS.md`](eval/RESULTS.md).

## Pipeline
1. `src/parse_law.py` — splits the Wikisource text into 301 articles (book / part / chapter kept).
2. `src/normalize.py` — Arabic normalization (diacritics, alef / yaa / taa marbuta, Arabic-Indic digits) + light stemming.
3. `src/retriever.py` — BM25 + dense embeddings merged with Reciprocal Rank Fusion; optional cross-encoder reranking.
4. `src/onnx_embedder.py`, `src/reranker.py` — bge-m3 and bge-reranker-v2-m3 as int8 ONNX: CPU only, no PyTorch.
5. `src/answer.py` — grounded answer with an LLM (Groq, optional `GROQ_API_KEY` in `.env`) or an extractive answer.
6. `app.py` — Gradio chat UI (RTL).

## Run
```bash
pip install -r requirements.txt
python scripts/download_data.py     # law text + models (~1.2 GB); add --small for a 120 MB model
python -m eval.run_all              # reproduce the table
python app.py                       # chat UI
```

## Next
- LLM query rewriting (colloquial → legal MSA) before retrieval
- Larger held-out set written by people other than me

Not legal advice.
