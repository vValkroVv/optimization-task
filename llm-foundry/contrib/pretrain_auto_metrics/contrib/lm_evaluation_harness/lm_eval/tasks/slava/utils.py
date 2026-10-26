import collections
import importlib
import json
import re
import string
import subprocess
import sys

import numpy as np
from thefuzz import fuzz


REGEX = re.compile("[%s]" % re.escape(string.punctuation + "\\n"))


def doc_to_text(doc):
    inputs = doc["inputs"]

    return doc["instruction"].format(**inputs) + "\nОтвет:"


# def doc_to_choice(doc):
#     return [opt for opt in doc["inputs"]["options"].values() if opt is not None]


# def doc_to_target(doc):
#     return int(doc["outputs"]) - 1


# def process_multiple_answers(doc, results):
#     ll, _ = zip(*results)
#     ll = np.array(ll)

#     # Convert log-likelihoods to probabilities.
#     probs = np.exp(ll)

#     # Normalize probabilities.
#     probs_norm = probs / np.sum(probs)

#     labels = np.array(list(doc['outputs']), dtype=int)
#     # Compute the normalized probability mass for the correct answer.
#     pm_true = np.sum(probs_norm[labels])

#     return {"pm_acc": pm_true}


def remove_articles(text: str) -> str:
    regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
    return re.sub(regex, " ", text)


def white_space_fix(text: str) -> str:
    return " ".join(text.split())


def remove_punctuation(text: str) -> str:
    exclude = set(string.punctuation)
    return "".join(ch for ch in text if ch not in exclude)


def to_lower(text: str) -> str:
    return text.lower()


def normalize_answer(text: str) -> str:
    lowered = to_lower(text)
    without_punctuation = remove_punctuation(lowered)
    without_articles = remove_articles(without_punctuation)
    normalized = white_space_fix(without_articles)

    return normalized


def get_tokens(text):
    if not text:
        return []
    return normalize_answer(text).split()


def only_numbers(raw_text: str) -> str:
    try:
        text = re.search(".*\\n", raw_text).group(0)
    except AttributeError:
        text = raw_text

    text = REGEX.sub("", text)
    text = "".join(text.split())

    match = re.search(r"\d+", text)
    if match:
        return match.group(0)
    else:
        return "-"


def calculate_f1_score(real_answer, model_answer):
    real_answer_tokens = get_tokens(real_answer)
    model_answer_tokens = get_tokens(model_answer)

    common = collections.Counter(real_answer_tokens) & collections.Counter(
        model_answer_tokens
    )
    num_same = sum(common.values())

    if len(real_answer_tokens) == 0 or len(model_answer_tokens) == 0:
        return int(real_answer_tokens == model_answer_tokens)

    if num_same == 0:
        return 0

    precision = 1.0 * num_same / len(model_answer_tokens)
    recall = 1.0 * num_same / len(real_answer_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match(gold, pred):
    return int(pred == gold)


def compute_one_choice(answer: str, only_numbers_model_answer: str) -> float:
    return 1.0 if answer == only_numbers_model_answer else 0.0


def compute_multi_choice(answer: str, only_numbers_model_answer: str) -> float:
    if answer == only_numbers_model_answer:
        return 1.0
    if len(answer) == len(only_numbers_model_answer):
        differences = sum(a != b for a, b in zip(answer, only_numbers_model_answer))
        if differences == 1:
            return 0.5
    if abs(len(answer) - len(only_numbers_model_answer)) == 1:
        return 0.5
    return 0.0


def compute_matching(answer: str, only_numbers_model_answer: str) -> float:
    if answer == only_numbers_model_answer:
        return 1.0
    if len(answer) == len(only_numbers_model_answer):
        differences = sum(a != b for a, b in zip(answer, only_numbers_model_answer))
        if differences == 1:
            return 0.5
    return 0.0


def compute_sequence(answer: str, only_numbers_model_answer: str) -> float:
    if answer == only_numbers_model_answer:
        return 1.0
    if len(answer) == len(only_numbers_model_answer):
        differences = sum(a != b for a, b in zip(answer, only_numbers_model_answer))
        if differences == 1:
            return 0.5
    return 0.0


def get_match_function(question_type: str):
    match_functions = {
        "выбор ответа (один)": compute_one_choice,
        "выбор ответа (мультивыбор)": compute_multi_choice,
        "установление соответствия": compute_matching,
        "указание последовательности": compute_sequence,
    }

    if question_type not in match_functions:
        raise ValueError("Unknown type of question")

    return match_functions[question_type]


def calculate_pm(gold, pred, question_type):
    only_numbers_model_answer = only_numbers(pred)
    match_function = get_match_function(question_type)
    return match_function(gold, only_numbers_model_answer)


def levenshtein_ratio(gold, pred):
    return fuzz.ratio(gold, pred) / 100


def f1_score(gold, pred):
    return calculate_f1_score(gold, pred)


def is_substring(gold, pred):
    return int(pred in gold)


def process_gen(doc, results):
    pred = str(results[0]).lower().strip()
    gold = str(doc["outputs"]).lower().strip()

    em = exact_match(gold, pred)
    lr = levenshtein_ratio(gold, pred)
    f1 = f1_score(gold, pred)

    return {
        "exact_match": em,
        "levenshtein_ratio": lr,
        "f1_score": f1,
        "mean_score": (em + lr + f1) / 3,
    }


def process_numeric_tasks(doc, results):
    pred = str(results[0]).lower().strip()
    gold = str(doc["outputs"]).lower().strip()
    task_type = doc["meta"]["type"]

    if task_type == "выбор ответа (один)":
        em = exact_match(gold, pred)
        return {"exact_match": em, "mean_score": em}
    else:
        em = exact_match(gold, pred)
        pm = calculate_pm(gold, pred, task_type)
        cc = is_substring(gold, pred)
        return {
            "exact_match": em,
            "partially_match": pm,
            "contains_check": cc,
            "mean_score": (em + pm + cc) / 3,
        }


def _score_single_gen(response, doc):
    pred = str(response).lower().strip()
    gold = str(doc["outputs"]).lower().strip()
    em = exact_match(gold, pred)
    lr = levenshtein_ratio(gold, pred)
    f1 = f1_score(gold, pred)
    return (em + lr + f1) / 3


def _score_single_numeric(response, doc):
    pred = str(response).lower().strip()
    gold = str(doc["outputs"]).lower().strip()
    task_type = doc["meta"]["type"]
    if task_type == "выбор ответа (один)":
        return float(exact_match(gold, pred))
    em = exact_match(gold, pred)
    pm = calculate_pm(gold, pred, task_type)
    cc = is_substring(gold, pred)
    return (em + pm + cc) / 3


def score_all_responses_gen(resps, docs):
    result = []
    for resp, doc in zip(resps, docs):
        strings = resp if resp else [""]
        scores = [_score_single_gen(s, doc) for s in strings]
        result.append([json.dumps(scores)])
    return result


def score_all_responses_numeric(resps, docs):
    result = []
    for resp, doc in zip(resps, docs):
        strings = resp if resp else [""]
        scores = [_score_single_numeric(s, doc) for s in strings]
        result.append([json.dumps(scores)])
    return result


def mean_at_k(references, predictions, k=None):
    scores = json.loads(predictions[0][0])
    n = len(scores)
    if k is None:
        k_list = [n]
    elif isinstance(k, int):
        k_list = [k]
    else:
        k_list = k
    return {f"mean@{ki}": sum(scores[:ki]) / ki for ki in k_list if ki <= n}
