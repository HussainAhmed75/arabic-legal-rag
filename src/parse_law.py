"""Parse an Arabic Wikisource law page (wikitext) into one record per article.

Usage: python -m src.parse_law data/raw/labor_law_14_2025.wiki --law "قانون العمل 14 لسنة 2025" --out data/articles.jsonl
"""
import argparse
import json
import re

from .normalize import normalize

HEADING = re.compile(r"^(=+)\s*(.+?)\s*\1\s*$")
ARTICLE = re.compile(r"^المادة\s+(.+)$")
ORDINALS = {"الأولى": 1, "الثانية": 2, "الثالثة": 3, "الرابعة": 4, "الخامسة": 5, "السادسة": 6,
            "السابعة": 7, "الثامنة": 8, "التاسعة": 9, "العاشرة": 10, "الحادية عشرة": 11,
            "الثانية عشرة": 12, "الثالثة عشرة": 13, "الرابعة عشرة": 14, "الخامسة عشرة": 15}


def clean_wiki(text: str) -> str:
    text = re.sub(r"\{\{[^{}]*\}\}", "", text)                  # templates
    text = re.sub(r"<ref[^>]*>.*?</ref>|<ref[^>]*/>", "", text, flags=re.S)
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", text)  # [[link|label]]
    text = re.sub(r"'{2,}", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def parse(wikitext: str, law: str) -> list[dict]:
    records, section = [], {"book": "", "part": "", "chapter": ""}
    current = None
    for line in wikitext.splitlines():
        m = HEADING.match(line.strip())
        if m:
            title = m.group(2).strip()
            a = ARTICLE.match(title)
            if a:
                if current:
                    records.append(current)
                label = a.group(1).strip()
                num = normalize(label)
                issuance = label in ORDINALS
                current = {
                    "law": law,
                    "article": ORDINALS.get(label) if issuance else int(num) if num.isdigit() else label,
                    "issuance_law": issuance,  # articles of the issuing law, not the attached code
                    **section,
                    "text": "",
                }
                continue
            if current:
                records.append(current)
                current = None
            if title.startswith("الكتاب"):
                section = {"book": title, "part": "", "chapter": ""}
            elif title.startswith("الباب"):
                section.update(part=title, chapter="")
            elif title.startswith("الفصل"):
                section["chapter"] = title
            continue
        if current is not None:
            current["text"] += line + "\n"
    if current:
        records.append(current)
    out = []
    for r in records:
        r["text"] = clean_wiki(r["text"])
        if r["text"]:
            prefix = "مادة إصدار" if r["issuance_law"] else "مادة"
            r["id"] = f"{prefix} {r['article']}"
            out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--law", required=True)
    ap.add_argument("--out", default="data/articles.jsonl")
    ap.add_argument("--append", action="store_true")
    args = ap.parse_args()
    with open(args.src, encoding="utf-8") as f:
        recs = parse(f.read(), args.law)
    with open(args.out, "a" if args.append else "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(recs)} articles -> {args.out}")


if __name__ == "__main__":
    main()
