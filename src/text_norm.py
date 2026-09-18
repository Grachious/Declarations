"""
Нормализация текста для сопоставления описаний товаров и текстов НПА.
"""
import re
from functools import lru_cache

TOKEN_RE = re.compile(r"[a-zA-Zа-яёА-ЯЁ]+|\d+[.,]?\d*", re.UNICODE)
STOPWORDS = {
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а",
    "то", "все", "она", "так", "его", "но", "да", "ты", "к", "у", "же",
    "вы", "за", "бы", "по", "только", "ее", "мне", "было", "вот", "от",
    "меня", "еще", "нет", "о", "из", "ему", "теперь", "когда", "даже",
    "ну", "вдруг", "ли", "если", "уже", "или", "ни", "быть", "был", "него",
    "до", "вас", "нибудь", "опять", "уж", "вам", "сказал", "ведь", "там",
    "потом", "себя", "ничего", "ей", "может", "они", "тут", "где", "есть",
    "надо", "ней", "для", "мы", "тебя", "их", "чем", "была", "сам", "чтоб",
    "без", "будто", "человек", "чего", "раз", "тоже", "себе", "под",
    "будет", "ж", "тогда", "кто", "этот", "того", "потому", "этого",
    "какой", "совсем", "ним", "здесь", "этом", "один", "почти", "мой",
    "тем", "чтобы", "нее", "также", "др", "прочие", "прочих", "прочее",
    "также", "либо", "данный", "данные", "является", "являются",
}

_PYMORPHY_AVAILABLE = False
try:
    import pymorphy2
    _morph = pymorphy2.MorphAnalyzer()
    _PYMORPHY_AVAILABLE = True
except Exception:
    _morph = None

_SUFFIXES = sorted([
    "ированный", "ированная", "ированное", "ированные",
    "ического", "ическому", "ическая", "ическое", "ические", "ических",
    "ованный", "ованная", "ованное", "ованные",
    "ейший", "ейшая", "ейшее", "ейшие",
    "ами", "ями", "ыми", "ими",
    "ого", "его", "ому", "ему", "ыми", "ими",
    "ая", "яя", "ое", "ее", "ые", "ие", "ых", "их",
    "ов", "ев", "ей", "ий", "ый", "ая", "ям", "ам", "ом", "ем",
    "а", "я", "ы", "и", "у", "ю", "о", "е", "й", "ь",
], key=len, reverse=True)


@lru_cache(maxsize=200_000)
def _stem_ru(word: str) -> str:
    if len(word) <= 4:
        return word
    for suf in _SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            return word[: -len(suf)]
    return word


@lru_cache(maxsize=200_000)
def _normalize_token(tok: str) -> str:
    tok = tok.lower().replace("ё", "е")
    if tok.isdigit() or re.match(r"^\d+[.,]?\d*$", tok):
        return tok.replace(",", ".")
    if re.match(r"^[а-я]+$", tok):
        if _PYMORPHY_AVAILABLE:
            try:
                return _morph.parse(tok)[0].normal_form
            except Exception:
                return _stem_ru(tok)
        return _stem_ru(tok)
    return tok 

def tokenize(text: str):
    if not text:
        return []
    tokens = TOKEN_RE.findall(text)
    out = []
    for t in tokens:
        nt = _normalize_token(t)
        if nt in STOPWORDS:
            continue
        if len(nt) < 2 and not nt.isdigit():
            continue
        out.append(nt)
    return out


def normalize_text(text: str) -> str:
    return " ".join(tokenize(text))
