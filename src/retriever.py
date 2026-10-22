"""Hybrid retrieval: BM25 on normalized Arabic tokens + optional multilingual embeddings,
merged with Reciprocal Rank Fusion (RRF)."""
import json
import os

import numpy as np
from rank_bm25 import BM25Okapi

from .normalize import tokenize
from .colloquial import expand

EMB_MODEL = os.getenv("EMB_MODEL", "intfloat/multilingual-e5-small")


def load_articles(path="data/articles.jsonl"):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def doc_text(a):
    return " ".join(x for x in (a.get("book"), a.get("part"), a.get("chapter"), a["text"]) if x)


class HybridRetriever:
    def __init__(self, articles, use_embeddings=True, cache="data/embeddings.npy", expand=True):
        self.articles = articles
        self.expand = expand
        self.bm25 = BM25Okapi([tokenize(doc_text(a)) for a in articles])
        self.model = None
        self.emb = None
        if use_embeddings:
            try:
                from . import onnx_embedder
                if onnx_embedder.available():
                    self.model = onnx_embedder.best_available()
                    cache = f"data/embeddings_{self.model.name}.npy"
                else:
                    from sentence_transformers import SentenceTransformer
                    self.model = SentenceTransformer(EMB_MODEL)
                if os.path.exists(cache) and np.load(cache).shape[0] == len(articles):
                    self.emb = np.load(cache)
                else:
                    self.emb = self.model.encode([getattr(self.model, "doc_prefix", "passage: ") + doc_text(a) for a in articles],
                                                 normalize_embeddings=True, batch_size=16,
                                                 show_progress_bar=True)
                    np.save(cache, self.emb)
            except Exception as e:  # no internet / package missing -> BM25 only
                print(f"[retriever] embeddings disabled: {e}")
                self.model = None

    def _bm25_rank(self, q):
        return np.argsort(-self.bm25.get_scores(tokenize(expand(q) if self.expand else q)))

    def _emb_rank(self, q):
        qv = self.model.encode([getattr(self.model, "query_prefix", "query: ") + q], normalize_embeddings=True)[0]
        return np.argsort(-(self.emb @ qv))

    def search(self, query, k=5, mode="hybrid", rrf_k=60, rerank_top=30):
        if mode == "rerank":
            if getattr(self, "reranker", None) is None:
                from .reranker import Reranker
                self.reranker = Reranker()
            cands = self.search(query, k=rerank_top, mode="hybrid", rrf_k=rrf_k)
            sc = self.reranker.scores(query, [doc_text(a) for a in cands])
            order = np.argsort(-sc)[:k]
            return [{**cands[i], "score": float(sc[i])} for i in order]
        ranks = []
        if mode in ("hybrid", "bm25"):
            ranks.append(self._bm25_rank(query))
        if mode in ("hybrid", "embeddings") and self.model is not None:
            ranks.append(self._emb_rank(query))
        if not ranks:
            ranks.append(self._bm25_rank(query))
        scores = {}
        for r in ranks:
            for pos, idx in enumerate(r[:100]):
                scores[idx] = scores.get(idx, 0) + 1 / (rrf_k + pos + 1)
        top = sorted(scores, key=scores.get, reverse=True)[:k]
        return [{**self.articles[i], "score": round(scores[i], 4)} for i in top]
