import json
import re
from math import comb


_ANS_RE = re.compile(r"#### (-?[\d,]+)")
_NUM_RE = re.compile(r"(-?[\$\d,\.]{2,})|(-?\d+)")


def _extract_answer(response: str) -> str:
    m = _ANS_RE.search(response)
    if m:
        return m.group(1).replace(",", "")
    nums = _NUM_RE.findall(response)
    if nums:
        return (nums[-1][0] or nums[-1][1]).replace(",", "").replace("$", "")
    return response.strip()


def _extract_target(doc: dict) -> str:
    m = _ANS_RE.search(doc["answer"])
    if m:
        return m.group(1).replace(",", "")
    return doc["answer"].strip()


def _check_single(response: str, doc: dict) -> int:
    pred = _extract_answer(response).strip()
    gold = _extract_target(doc).strip()
    try:
        return 1 if float(pred) == float(gold) else 0
    except ValueError:
        return 1 if pred.lower() == gold.lower() else 0


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
