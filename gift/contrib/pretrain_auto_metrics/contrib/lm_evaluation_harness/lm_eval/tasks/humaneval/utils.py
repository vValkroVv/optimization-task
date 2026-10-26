import json
from math import comb

import evaluate as hf_evaluate


FILE_CONTINUATION_MARKERS = (
    "<\uff5cend\u2581of\u2581file\uff5c>",
    "<|end_of_file|>",
    "<\uff5cbegin\u2581of\u2581file\u2581name\uff5c>",
    "<|begin_of_file_name|>",
)


try:
    compute_ = hf_evaluate.load("code_eval")
    test_cases = ["assert add(2, 3)==5"]
    candidates = [["def add(a,b): return a*b"]]
    results = compute_.compute(references=test_cases, predictions=candidates, k=[1])
except Exception as e:
    raise e


def pass_at_1(references: list[str], predictions: list[list[str]]) -> float:
    return compute_.compute(
        references=references,
        predictions=predictions,
        k=[1],
    )[0]["pass@1"]


def strip_file_continuation_markers(completion: str) -> str:
    end_index = len(completion)
    for marker in FILE_CONTINUATION_MARKERS:
        marker_index = completion.find(marker)
        if marker_index != -1:
            end_index = min(end_index, marker_index)
    return completion[:end_index]


def strip_markdown_fence(completion: str) -> str:
    fence_index = completion.find("```")
    if fence_index == -1:
        return completion
    return completion[:fence_index]


def build_predictions(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    return [
        [doc["prompt"] + strip_file_continuation_markers(r) for r in resp]
        for resp, doc in zip(resps, docs)
    ]


def build_predictions_instruct(
    resps: list[list[str]], docs: list[dict]
) -> list[list[str]]:
    return [
        [
            doc["prompt"] + strip_markdown_fence(strip_file_continuation_markers(r))
            for r in resp
        ]
        for resp, doc in zip(resps, docs)
    ]


def _reference(doc: dict) -> str:
    return doc["test"] + "\ncheck(" + doc["entry_point"] + ")"


def _score(candidates: list[str], ref: str) -> list[int]:
    try:
        _, detail = compute_.compute(
            references=[ref] * len(candidates),
            predictions=[[c] for c in candidates],
            k=[1],
        )
        return [1 if detail[i][0][1]["passed"] else 0 for i in range(len(candidates))]
    except Exception:
        return [0] * len(candidates)


def score_all_responses(resps: list[list[str]], docs: list[dict]) -> list:
    result = []
    for resp, doc in zip(resps, docs):
        strings = resp if resp else [""]
        candidates = [doc["prompt"] + r for r in strings]
        result.append([json.dumps(_score(candidates, _reference(doc)))])
    return result


def score_all_responses_instruct(resps: list[list[str]], docs: list[dict]) -> list:
    result = []
    for resp, doc in zip(resps, docs):
        strings = resp if resp else [""]
        candidates = [
            doc["prompt"] + (r if r.find("```") == -1 else r[: r.find("```")])
            for r in strings
        ]
        result.append([json.dumps(_score(candidates, _reference(doc)))])
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
