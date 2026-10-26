import re
import string
from functools import partial

_PUNCT = str.maketrans("", "", string.punctuation + "«»—–…")


def _normalize(s: str) -> str:
    s = s.lower().replace("ё", "е")
    s = re.sub(r"\[.*?\]", "", s)
    s = s.translate(_PUNCT)
    return " ".join(s.split())


def doc_to_text(doc) -> str:
    return (
        f"Вопрос: {doc['question'].strip()}\n\n"
        "Рассуждай шаг за шагом. Последняя строка твоего вывода должна "
        "иметь вид:\nОтвет: <финальный ответ>\n"
    )


def extract_answer(text: str) -> str:
    m = re.findall(r"(?im)^[^а-яa-z]*ответ\b[\s:*—–-]*(.+)$", text)
    if m:
        return m[-1]
    lines = [l for l in text.strip().splitlines() if l.strip()]
    return lines[-1] if lines else ""


def doc_to_target(doc) -> str:
    return doc["answer"].strip()


def process_results(doc, results):
    pred = _normalize(extract_answer(results[0]))
    ok = _normalize(doc["answer"]) == pred
    return {"exact_match": 1.0 if ok else 0.0}


def _filter_cat_era(dataset, cat, era):
    return dataset.filter(lambda doc: doc["cat"] == cat and doc["era"] == era)


process_art_pre1917 = partial(_filter_cat_era, cat="art", era="pre1917")
process_art_soviet = partial(_filter_cat_era, cat="art", era="soviet")
process_folk_timeless = partial(_filter_cat_era, cat="folk", era="timeless")
process_history_modern = partial(_filter_cat_era, cat="history", era="modern")
process_history_pre1917 = partial(_filter_cat_era, cat="history", era="pre1917")
process_history_soviet = partial(_filter_cat_era, cat="history", era="soviet")
process_idioms_timeless = partial(_filter_cat_era, cat="idioms", era="timeless")
process_life_modern = partial(_filter_cat_era, cat="life", era="modern")
process_life_pre1917 = partial(_filter_cat_era, cat="life", era="pre1917")
process_life_soviet = partial(_filter_cat_era, cat="life", era="soviet")
process_life_timeless = partial(_filter_cat_era, cat="life", era="timeless")
process_lit_modern = partial(_filter_cat_era, cat="lit", era="modern")
process_lit_pre1917 = partial(_filter_cat_era, cat="lit", era="pre1917")
process_lit_soviet = partial(_filter_cat_era, cat="lit", era="soviet")
process_music_modern = partial(_filter_cat_era, cat="music", era="modern")
process_music_soviet = partial(_filter_cat_era, cat="music", era="soviet")
process_screen_modern = partial(_filter_cat_era, cat="screen", era="modern")
process_screen_soviet = partial(_filter_cat_era, cat="screen", era="soviet")
process_wordplay_modern = partial(_filter_cat_era, cat="wordplay", era="modern")
process_wordplay_timeless = partial(_filter_cat_era, cat="wordplay", era="timeless")
