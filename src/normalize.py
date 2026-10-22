"""Arabic text normalization used for both indexing and queries."""
import re

_DIACRITICS = re.compile(r"[ؗ-ًؚ-ْٰۖ-ۭ]")
_TATWEEL = "ـ"
_ARABIC_INDIC = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_PREFIXES = ("وال", "بال", "كال", "فال", "لل", "ال")
_STOPWORDS = {
    "في", "من", "على", "الى", "إلى", "عن", "او", "أو", "و", "ف", "ثم", "هل", "ما", "ماذا",
    "هذا", "هذه", "ذلك", "التي", "الذي", "الذين", "ان", "أن", "إن", "كان", "يكون", "لا",
    "مع", "كل", "اي", "أي", "قد", "به", "بها", "له", "لها", "هو", "هي", "عند", "اذا", "إذا",
    "انا", "عايز", "ايه", "اية", "ازاي", "امتى", "مين", "ليه", "بتاع", "بتاعي", "ده", "دي",
}


def normalize(text: str) -> str:
    text = text.translate(_ARABIC_INDIC)
    text = _DIACRITICS.sub("", text).replace(_TATWEEL, "")
    text = re.sub("[إأآٱ]", "ا", text)
    text = text.replace("ى", "ي").replace("ة", "ه").replace("ؤ", "و").replace("ئ", "ي")
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def light_stem(token: str) -> str:
    for p in _PREFIXES:
        if token.startswith(p) and len(token) - len(p) >= 2:
            return token[len(p):]
    if token[:1] in ("و", "ف", "ب", "ل") and len(token) > 4:
        token = token[1:]
    return token


def tokenize(text: str) -> list[str]:
    stop = {normalize(w) for w in _STOPWORDS}
    return [light_stem(t) for t in normalize(text).split() if t not in stop and len(t) > 1]
