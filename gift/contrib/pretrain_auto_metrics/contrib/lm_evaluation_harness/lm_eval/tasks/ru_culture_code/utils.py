import re
from functools import partial

LETTERS = ["А", "Б", "В", "Г", "Д", "Е"]

_ANS = re.compile(r"[Оо]твет\**\s*:\s*\**\s*([А-Е])\b")
_LEAD = re.compile(r"^\s*\**\s*([А-Е])\s*[).]")


def _format_question(doc):
    opts = "\n".join(f"{LETTERS[i]}) {o}" for i, o in enumerate(doc["options"]))
    return (
        f"Задание: {doc['instruction']}\n\n"
        f"Вопрос: {doc['question']}\n\n{opts}\n\nРассуждение:"
    )


def doc_to_text(doc):
    return "Теперь ответь на новый вопрос.\n\n" + _format_question(doc)


def fewshot_to_text(doc):
    return _format_question(doc)


def fewshot_to_target(doc):
    return f"{doc['reasoning']}\nОтвет: {doc['answer']}"


def process_results(doc, results):
    text = str(results[0] if results else "")
    m = _ANS.search(text) or _LEAD.match(text)
    pred = m.group(1).upper() if m else ""
    gold = str(doc.get("answer", "")).strip().upper()[:1]
    return {"exact_match": 1.0 if pred and pred == gold else 0.0}


def _filter_class(dataset, label):
    return dataset.filter(lambda doc: doc["class"] == label)


process_aphorisms = partial(_filter_class, label="Aphorisms")
process_childhood = partial(_filter_class, label="basic type; childhood subtype")
process_cultural_person = partial(_filter_class, label="cultural person")
process_digital = partial(_filter_class, label="digital culture")
process_idioms = partial(_filter_class, label="idioms")
process_language_game = partial(_filter_class, label="language game")
process_memes = partial(_filter_class, label="memes")
process_phraseology = partial(_filter_class, label="phraseology")
process_pop_culture = partial(_filter_class, label="Pop_culture")
process_proverb = partial(_filter_class, label="paremia type; proverb subtype")
process_quotes = partial(_filter_class, label="quotes")
process_saying = partial(_filter_class, label="paremia type; saying subtype")
process_subculture = partial(_filter_class, label="subculture")
