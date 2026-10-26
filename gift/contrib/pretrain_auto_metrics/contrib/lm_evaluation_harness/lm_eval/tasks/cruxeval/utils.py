"""
CRUXEval lm_eval tasks - CoT variants.

CRUXEval-O: given code + input -> predict output
CRUXEval-I: given code + output -> predict input

Dataset: cruxeval-org/cruxeval (800 Python functions, HuggingFace Hub)
Prompts verbatim from upstream https://github.com/facebookresearch/cruxeval/blob/main/prompts.py
Exec validation via hf_evaluate code_eval (same sandbox as HumanEval).
"""

import json
import re
from math import comb
from typing import List

import evaluate as hf_evaluate

try:
    _code_eval = hf_evaluate.load("code_eval")
except Exception as e:
    raise e


def format_prompt_cot_output(doc):
    code = doc["code"]
    inp = doc["input"]
    return (
        "You are given a Python function and an assertion containing an input to the "
        "function. Complete the assertion with a literal (no unsimplified expressions, "
        "no function calls) containing the output when executing the provided code on "
        "the given input, even if the function is incorrect or incomplete. Do NOT output "
        "any extra information. Execute the program step by step before arriving at an "
        "answer, and provide the full assertion with the correct output in [ANSWER] and "
        "[/ANSWER] tags, following the examples.\n\n"
        "[PYTHON]\n"
        'def f(s):\n    s = s + s\n    return "b" + s + "a"\n'
        'assert f("hi") == ??\n'
        "[/PYTHON]\n"
        "[THOUGHT]\n"
        "Let\'s execute the code step by step:\n\n"
        "1. The function f is defined, which takes a single argument s.\n"
        '2. The function is called with the argument "hi", so within the function, s is initially "hi".\n'
        '3. Inside the function, s is concatenated with itself, so s becomes "hihi".\n'
        '4. The function then returns a new string that starts with "b", followed by the value of s (which is now "hihi"), and ends with "a".\n'
        '5. The return value of the function is therefore "bhihia".\n'
        "[/THOUGHT]\n"
        "[ANSWER]\n"
        'assert f("hi") == "bhihia"\n'
        "[/ANSWER]\n\n"
        f"[PYTHON]\n{code}\nassert f({inp}) == ??\n[/PYTHON]\n[THOUGHT]\n"
    )


def format_prompt_cot_input(doc):
    code = doc["code"]
    out = doc["output"]
    return (
        "You will be given a function f and an output in the form f(??) == output. "
        "Your task is to find any input such that executing f on the input leads to the "
        "given output. There may be multiple answers, but only output one. First, think "
        "step by step. You MUST surround the answer with [ANSWER] and [/ANSWER] tags. "
        "Express your answer as a passing assertion containing the input and the given output.\n\n"
        "[PYTHON]\n"
        "def f(x):\n    return x + 1\n"
        "assert f(??) == 17\n"
        "[/PYTHON]\n"
        "[THOUGHT]\n"
        "To find an input such that executing f on the input leads to the given output, "
        "we can work backwards from the given assertion. We know that f(??) == 17.\n\n"
        "Since the function f(x) returns x + 1, for f(??) to be equal to 17, the value of ?? should be 16.\n"
        "[/THOUGHT]\n"
        "[ANSWER]\n"
        "assert f(16) == 17\n"
        "[/ANSWER]\n\n"
        f"[PYTHON]\n{code}\nassert f(??) == {out}\n[/PYTHON]\n[THOUGHT]\n"
    )


def build_reference(doc):
    return json.dumps({"code": doc["code"], "input": doc["input"], "output": doc["output"]})


def _extract_one(resp: str) -> str:
    match = re.search(r"\[ANSWER\](.*?)\[/ANSWER\]", resp, re.DOTALL)
    if match:
        return match.group(1).strip()
    for line in resp.split("\n"):
        if line.strip().startswith("assert"):
            return line.strip()
    return resp.strip()


def extract_answer(resps, docs):
    out = []
    for resp_list, doc in zip(resps, docs):
        out.append([_extract_one(r) for r in resp_list])
    return out


def _build_test(ref: dict, predicted: str, mode: str) -> str:
    if mode == "output":
        match = re.search(r"assert\s+f\(.*?\)\s*==\s*(.*)", predicted)
        pv = match.group(1).strip() if match else predicted.strip()
        return "assert f(%s) == %s" % (ref["input"], pv)
    if mode == "input":
        match = re.search(r"assert\s+f\((.*?)\)\s*==", predicted)
        pi = match.group(1).strip() if match else predicted.strip()
        return "assert f(%s) == %s" % (pi, ref["output"])
    raise ValueError("mode must be 'input' or 'output'")


def pass_at_1(references, predictions, mode="output"):
    """Unified CRUXEval metric. mode = 'output' (CRUXEval-O) or 'input' (CRUXEval-I)."""
    ref = json.loads(references[0])
    test = _build_test(ref, predictions[0][0], mode)
    result = _code_eval.compute(references=[test], predictions=[[ref["code"]]], k=[1])
    return result[0]["pass@1"]


def _score_all_responses(resps: List[List[str]], docs: List[dict], mode: str) -> List[List[str]]:
    result = []
    for resp, doc in zip(resps, docs):
        strings = resp if resp else [""]
        ref = json.loads(build_reference(doc))
        tests = [_build_test(ref, _extract_one(r), mode) for r in strings]
        try:
            _, detail = _code_eval.compute(
                references=tests,
                predictions=[[ref["code"]] for _ in tests],
                k=[1],
            )
            scores = [
                1 if detail[i][0][1]["passed"] else 0 for i in range(len(tests))
            ]
        except Exception:
            scores = [0] * len(tests)
        result.append([json.dumps(scores)])
    return result


def score_all_responses_output(resps: List[List[str]], docs: List[dict]) -> List[List[str]]:
    return _score_all_responses(resps, docs, "output")


def score_all_responses_input(resps: List[List[str]], docs: List[dict]) -> List[List[str]]:
    return _score_all_responses(resps, docs, "input")


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
