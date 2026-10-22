from src.normalize import normalize, tokenize
from src.parse_law import parse
from src.retriever import HybridRetriever

SAMPLE = """{{عنوان|قانون العمل المصري}}
== نص القانون ==
=== المادة الأولى ===
يعمل بأحكام هذا القانون والقانون المرافق في شأن العمل الفردي.
=== المادة الثانية عشرة ===
يُلغى القانون رقم ١٢ لسنة ٢٠٠٣ بإصدار قانون العمل.
= الكتاب الثالث: علاقات العمل =
== الباب الأول: علاقات العمل الفردية ==
=== الفصل الأول: عقد العمل الفردي ===
==== المادة ٨٦ ====
يلتزم صاحب العمل بتحرير عقد العمل كتابةً باللغة العربية من أربع نسخ.
==== المادة ٨٧ ====
يستحق العامل إجازة سنوية مدفوعة الأجر مدتها خمسة عشر يومًا<ref>مصدر</ref>.
== المراجع ==
"""


def test_normalize():
    assert normalize("المادةُ ٨٦ إجازةٌ") == "الماده 86 اجازه"
    assert "اجازه" in tokenize("الإجازة") or "جازه" in tokenize("الإجازة")


def test_parse():
    recs = parse(SAMPLE, "قانون العمل 14 لسنة 2025")
    ids = [r["id"] for r in recs]
    assert ids == ["مادة إصدار 1", "مادة إصدار 12", "مادة 86", "مادة 87"]
    assert recs[2]["chapter"] == "الفصل الأول: عقد العمل الفردي"
    assert "مصدر" not in recs[3]["text"]


def test_retrieval_bm25():
    recs = parse(SAMPLE, "قانون العمل 14 لسنة 2025")
    r = HybridRetriever(recs, use_embeddings=False)
    assert r.search("كام يوم اجازة سنوية؟", k=1)[0]["id"] == "مادة 87"
    assert r.search("عقد العمل لازم يكون مكتوب؟", k=1)[0]["id"] == "مادة 86"
