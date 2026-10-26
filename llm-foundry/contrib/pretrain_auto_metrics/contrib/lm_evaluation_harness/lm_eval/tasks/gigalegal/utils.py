import json
import re
import math

K_LIST = [1]

def _pick_yes_no(text: str) -> str:
    m = re.search(r"\b(да|нет|неизвестно)\b", (text or "").lower())
    return m.group(1) if m else ""

def _pick_abc(text: str) -> str:
    m = re.search(r"\b([ABC])\b", (text or "").upper())
    return m.group(1) if m else ""


def process_results_passk(doc, results):
    if len(results) == 1 and isinstance(results[0], list):
        results = results[0]

    gold = _pick_yes_no(doc.get("target") or doc.get("final_answers") or doc.get("answer"))
    preds = [_pick_yes_no(r) for r in (results or [])]

    n = len(preds)
    correct = sum(1 for p in preds if p == gold)

    out = {}
#     out["avg_pass@1"] = (correct / n) if n else 0.0

    for k in K_LIST:
        kk = min(k, n)
        if kk == 0 or correct == 0:
            out[f"pass@{k}"] = 0.0
        elif (n - correct) < kk:
            out[f"pass@{k}"] = 1.0
        else:
            out[f"pass@{k}"] = 1.0 - math.comb(n - correct, kk) / math.comb(n, kk)

    return out


def process_results_passk_mcq(doc, results):
    if len(results) == 1 and isinstance(results[0], list):
        results = results[0]

    gold = (doc.get("answer_key") or "").strip().upper()
    preds = [_pick_abc(r) for r in (results or [])]

    n = len(preds)
    correct = sum(1 for p in preds if p == gold)

    out = {}
#     out["avg_pass@1"] = (correct / n) if n else 0.0

    for k in K_LIST:
        kk = min(k, n)
        if kk == 0 or correct == 0:
            out[f"pass@{k}"] = 0.0
        elif (n - correct) < kk:
            out[f"pass@{k}"] = 1.0
        else:
            out[f"pass@{k}"] = 1.0 - math.comb(n - correct, kk) / math.comb(n, kk)

    return out


def _check_single_yes_no(response, doc):
    gold = _pick_yes_no(doc.get("target") or doc.get("final_answers") or doc.get("answer"))
    pred = _pick_yes_no(response)
    return 1 if gold and pred == gold else 0


def _check_single_abc(response, doc):
    gold = (doc.get("answer_key") or "").strip().upper()
    pred = _pick_abc(response)
    return 1 if gold and pred == gold else 0


def score_all_responses_yes_no(resps, docs):
    result = []
    for resp, doc in zip(resps, docs):
        strings = resp if resp else [""]
        scores = [_check_single_yes_no(s, doc) for s in strings]
        result.append([json.dumps(scores)])
    return result


def score_all_responses_abc(resps, docs):
    result = []
    for resp, doc in zip(resps, docs):
        strings = resp if resp else [""]
        scores = [_check_single_abc(s, doc) for s in strings]
        result.append([json.dumps(scores)])
    return result


def pass_at_k(references, predictions, k=None):
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
            out[f"pass@{ki}"] = 1.0 - math.comb(n - c, ki) / math.comb(n, ki)
    return out


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
