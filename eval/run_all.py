"""Recall@5 / MRR for every mode on dev (questions.jsonl) and held-out test (test_questions.jsonl)."""
import json, sys
from src.retriever import HybridRetriever, load_articles

def score(r, qs, mode, k=5):
    hits, rr, miss = 0, 0.0, []
    for it in qs:
        ids = [a["id"] for a in r.search(it["q"], k=k, mode=mode)]
        rk = [ids.index(g) + 1 for g in it["gold"] if g in ids]
        hits += bool(rk); rr += 1 / min(rk) if rk else 0
        if not rk: miss.append(it["q"])
    return hits / len(qs), rr / len(qs), miss

arts = load_articles()
sets = {s: [json.loads(l) for l in open(f"eval/{s}.jsonl", encoding="utf-8")] for s in ("questions", "test_questions")}
for ex in (True,):
    r = HybridRetriever(arts, expand=ex)
    for mode in ("bm25", "embeddings", "hybrid") + (("rerank",) if __import__("src.reranker", fromlist=["x"]).available() else ()):
        line = f"expand={ex!s:<5} {mode:<10}"
        for s, qs in sets.items():
            rc, mrr, miss = score(r, qs, mode)
            line += f" | {s[:4]} R@5 {rc:.2f} MRR {mrr:.2f}"
        print(line)
