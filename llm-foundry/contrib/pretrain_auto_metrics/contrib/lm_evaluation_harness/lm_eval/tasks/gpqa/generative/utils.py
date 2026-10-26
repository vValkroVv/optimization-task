import json
import random
import re
from math import comb

import datasets


GPQA_CHOICE_SEED = 0


def preprocess(text):
    if text is None:
        return " "
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    text = text.replace("  ", " ")
    return text


def _ordered_choices(doc, rng: random.Random):
    labeled_choices = [
        ("incorrect_1", preprocess(doc["Incorrect Answer 1"])),
        ("incorrect_2", preprocess(doc["Incorrect Answer 2"])),
        ("incorrect_3", preprocess(doc["Incorrect Answer 3"])),
        ("correct", preprocess(doc["Correct Answer"])),
    ]
    rng.shuffle(labeled_choices)
    return labeled_choices


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    rng = random.Random(GPQA_CHOICE_SEED)

    def _process_doc(doc):
        ordered = _ordered_choices(doc, rng)
        choices = [choice for _label, choice in ordered]
        correct_answer_index = [label for label, _choice in ordered].index("correct")

        out_doc = {
            "choice1": choices[0],
            "choice2": choices[1],
            "choice3": choices[2],
            "choice4": choices[3],
            "choices": choices,
            "answer": f"({chr(65 + correct_answer_index)})",
        }
        return out_doc

    return dataset.map(_process_doc)


def _extract_choice(response: str) -> str:
    m = re.search(r"[Tt]he answer is\s*\(?([A-D])\)?", response)
    if m:
        return m.group(1).upper()
    matches = re.findall(r"\(([A-D])\)", response)
    if matches:
        return matches[-1].upper()
    return ""


def _check_single(response: str, doc: dict) -> int:
    pred = _extract_choice(response)
    gold = doc["answer"].strip("() ").upper()
    return 1 if pred and pred == gold else 0


def score_all_responses(resps: list, docs: list) -> list:
    result = []
    for resp, doc in zip(resps, docs):
        strings = resp if resp else [""]
        scores = [_check_single(s, doc) for s in strings]
        result.append([json.dumps(scores)])
    return result


def pass_at_k(references: list, predictions: list, k=None) -> dict:
    assert k is not None
    if isinstance(k, int):
        k = [k]
    scores = json.loads(predictions[0][0])
    n, c = len(scores), sum(scores)
    out = {}
    for ki in k:
        if ki > n:
            out[f"pass@{ki}"] = float("nan")
        elif c == 0:
            out[f"pass@{ki}"] = 0.0
        elif n - c < ki:
            out[f"pass@{ki}"] = 1.0
        else:
            out[f"pass@{ki}"] = 1.0 - comb(n - c, ki) / comb(n, ki)
    return out


def mean_at_k(references: list, predictions: list, k=None) -> dict:
    scores = json.loads(predictions[0][0])
    n = len(scores)
    if k is None:
        k_list = [n]
    elif isinstance(k, int):
        k_list = [k]
    else:
        k_list = k
    return {f"mean@{ki}": sum(scores[:ki]) / ki for ki in k_list if ki <= n}
