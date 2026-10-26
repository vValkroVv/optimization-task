import json
import re
from math import comb
from typing import Union

import evaluate as hf_evaluate


try:
    code_eval = hf_evaluate.load("code_eval")

    # run simple test to check code execution is enabled before model generation
    test_cases = ["assert add(2, 3)==5"]
    candidates = [["def add(a,b): return a*b"]]
    results = code_eval.compute(references=test_cases, predictions=candidates, k=[1])
except Exception as e:
    raise e


def pass_at_1(
    references: Union[str, list[str]], predictions: Union[str, list[list[str]]]
) -> float:
    if isinstance(references, str):
        references = [references]
    if isinstance(predictions[0], str):
        predictions = [[p] for p in predictions]
    return code_eval.compute(
        references=references,
        predictions=predictions,
        k=[1],
    )[0]["pass@1"]


def _looks_like_python(text: str) -> bool:
    return bool(re.search(r"(?m)^\s*(def|class|import|from)\s+", text))


def extract_code_blocks(text: str) -> str:
    pattern = r"```(?:\w+)?\n?(.*?)\n?```"

    for match in re.findall(pattern, text, re.DOTALL):
        if _looks_like_python(match):
            return match.strip()

    candidate = re.split(r"\[DONE\]|\[END\]", text, maxsplit=1)[0].strip()
    candidate = re.sub(r"^```(?:\w+)?[ \t]*\n", "", candidate)
    candidate = re.sub(r"\n?```[ \t]*$", "", candidate).strip()
    if _looks_like_python(candidate):
        return candidate

    text_without_lang = re.sub(r"```python", "```", text)
    for match in re.findall(pattern, r"```" + text_without_lang, re.DOTALL):
        if _looks_like_python(match):
            return match.strip()
    return ""


def build_predictions(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    return [[extract_code_blocks(r) for r in resp] for resp in resps]


def score_all_responses(resps: list[list[str]], docs: list[dict]) -> list:
    result = []
    for resp, doc in zip(resps, docs):
        strings = resp if resp else [""]
        ref = "\n".join(doc["test_list"][:3])
        candidates = [extract_code_blocks(r) or r for r in strings]
        try:
            _, detail = code_eval.compute(
                references=[ref] * len(candidates),
                predictions=[[c] for c in candidates],
                k=[1],
            )
            scores = [1 if detail[i][0][1]["passed"] else 0 for i in range(len(candidates))]
        except Exception:
            scores = [0] * len(candidates)
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


def list_fewshot_samples():
    return [
        {
            "task_id": 2,
            "text": "Write a function to find the similar elements from the given two tuple lists.",
            "code": "def similar_elements(test_tup1, test_tup2):\r\n  res = tuple(set(test_tup1) & set(test_tup2))\r\n  return (res) ",
            "test_list": [
                "assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)",
                "assert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4)",
                "assert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14)",
            ],
            "is_fewshot": True,
        },
        {
            "task_id": 3,
            "text": "Write a python function to identify non-prime numbers.",
            "code": "import math\r\ndef is_not_prime(n):\r\n    result = False\r\n    for i in range(2,int(math.sqrt(n)) + 1):\r\n        if n % i == 0:\r\n            result = True\r\n    return result",
            "test_list": [
                "assert is_not_prime(2) == False",
                "assert is_not_prime(10) == True",
                "assert is_not_prime(35) == True",
            ],
            "is_fewshot": True,
        },
        {
            "task_id": 4,
            "text": "Write a function to find the largest integers from a given list of numbers using heap queue algorithm.",
            "code": "import heapq as hq\r\ndef heap_queue_largest(nums,n):\r\n  largest_nums = hq.nlargest(n, nums)\r\n  return largest_nums",
            "test_list": [
                "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],3)==[85, 75, 65] ",
                "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],2)==[85, 75] ",
                "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],5)==[85, 75, 65, 58, 35]",
            ],
            "is_fewshot": True,
        },
    ]
