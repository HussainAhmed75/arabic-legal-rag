"""Retrieval evaluation: Recall@k and MRR for each retrieval mode.
eval/questions.jsonl lines: {"q": "...", "gold": ["مادة 47", ...]}
Usage: python -m eval.evaluate
"""
import json

from src.retriever import HybridRetriever, load_articles


def main(k=5):
    qs = [json.loads(l) for l in open("eval/questions.jsonl", encoding="utf-8") if l.strip()]
    r = HybridRetriever(load_articles())
    modes = ["bm25"] + (["embeddings", "hybrid"] if r.model is not None else [])
    print(f"{len(qs)} questions, k={k}")
    for mode in modes:
        hits, rr = 0, 0.0
        for item in qs:
            ids = [a["id"] for a in r.search(item["q"], k=k, mode=mode)]
            ranks = [ids.index(g) + 1 for g in item["gold"] if g in ids]
            hits += bool(ranks)
            rr += 1 / min(ranks) if ranks else 0
        print(f"{mode:<11} Recall@{k}: {hits/len(qs):.2f}   MRR: {rr/len(qs):.2f}")


if __name__ == "__main__":
    main()
