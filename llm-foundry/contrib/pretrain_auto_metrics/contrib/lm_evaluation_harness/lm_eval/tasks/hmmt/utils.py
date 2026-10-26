import json
from math import comb


try:
    import antlr4
    from math_verify import parse, verify
except (ModuleNotFoundError, AssertionError) as e:
    raise type(e)(
        "`math_verify` is required for the hmmt tasks. "
        "Please install the required packages via pip install lm-eval[math] or pip install -e .[math]"
    ) from e


def _check_single(response: str, doc: dict) -> int:
    answer_key = next(k for k in doc.keys() if k.lower() == "answer")
    target = str(doc[answer_key])
    try:
        return 1 if verify(gold=parse(f"${target}$"), target=parse(response)) else 0
    except Exception:
        return 0


def score_all_responses(resps: list, docs: list) -> list:
    result = []
    for resp, doc in zip(resps, docs):
        strings = resp if resp else [""]
        scores = [_check_single(s, doc) for s in strings]
        result.append([json.dumps(scores)])
    return result


def pass_at_k(references: list, predictions: list, k=None) -> dict:
    assert k is not None, "pass_at_k requires k to be specified in metric_list"
    if isinstance(k, int):
        k = [k]

    scores = json.loads(predictions[0][0])
    n = len(scores)
    c = sum(scores)

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
