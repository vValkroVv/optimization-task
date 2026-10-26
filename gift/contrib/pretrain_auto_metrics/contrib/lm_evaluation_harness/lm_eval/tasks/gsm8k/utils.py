import json
import re
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from math import comb


_GSM8K_EXACT_MATCH_REGEXES_TO_IGNORE = [
    ",",
    r"\$",
    r"(?s).*#### ",
    r"\.$",
]


def normalize_gsm8k_number(text: object) -> str:
    if text is None:
        return "[invalid]"
    value = str(text).strip()
    if not value or value == "[invalid]":
        return "[invalid]"

    value = value.replace("$", "").replace(",", "").strip()
    while value.endswith(".") and value.count(".") > 1:
        value = value[:-1]
    try:
        number = Decimal(value)
    except InvalidOperation:
        return str(text).strip()

    if number == number.to_integral_value():
        return str(int(number))
    normalized = format(number.normalize(), "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return normalized or "0"


def normalize_numeric_responses(
    resps: list[list[str]], docs: list[dict]
) -> list[list[str]]:
    return [[normalize_gsm8k_number(resp) for resp in resp_set] for resp_set in resps]


def raw_response_for_math_verify(
    resps: list[list[str]], docs: list[dict]
) -> list[list[dict]]:
    return [[{"raw_response": resp} for resp in resp_set] for resp_set in resps]


def _gsm8k_gold_answer(doc: dict) -> str:
    if "target" in doc:
        return str(doc["target"]).strip()
    answer = str(doc.get("answer", "")).strip()
    if "####" in answer:
        return answer.split("####")[-1].strip()
    return answer


def _apply_exact_match_normalization(text: object) -> str:
    value = str(text)
    for pattern in _GSM8K_EXACT_MATCH_REGEXES_TO_IGNORE:
        value = re.sub(pattern, "", value)
    return value.lower()


def _gsm8k_exact_match(prediction: object, doc: dict) -> int:
    normalized_prediction = _apply_exact_match_normalization(prediction)
    normalized_reference = _apply_exact_match_normalization(_gsm8k_gold_answer(doc))
    return int(normalized_prediction == normalized_reference)


@lru_cache(maxsize=1)
def _load_math_verify():
    try:
        from math_verify import parse, verify
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "`math_verify` is required for GSM8K math_verify scoring. "
            "Install it with `pip install lm-eval[math]` or `pip install -e .[math]`."
        ) from exc
    return parse, verify


def _gsm8k_math_verify(doc: dict, response: object) -> int:
    parse, verify = _load_math_verify()
    return int(verify(gold=parse(_gsm8k_gold_answer(doc)), target=parse(str(response))))


def process_results(doc: dict, results: list[object]) -> dict[str, int]:
    candidate = results[0]
    if isinstance(candidate, dict) and "raw_response" in candidate:
        return {"math_verify": _gsm8k_math_verify(doc, candidate["raw_response"])}
    return {"exact_match": _gsm8k_exact_match(candidate, doc)}


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
