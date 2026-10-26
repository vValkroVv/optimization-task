import base64
import json
import pickle
import re
import zlib
from math import comb
from pathlib import Path
from typing import Any, Dict, List, Union

from lm_eval.tasks.livecodebench._stdin_runs import (
    run_code_generation_sandboxed,
)


OPEN_FENCE_RE = re.compile(r"```(?:[\w.+-]+)?\s*\n?")
FULL_FENCE_RE = re.compile(r"```(?:[\w.+-]+)?\s*\n?(.*?)\n?\s*```(?:\s|$)", re.DOTALL)

BASE_DIR = Path(__file__).parent

with open(BASE_DIR / "func.json") as f:
    FEWSHOT_FUNC = json.load(f)

with open(BASE_DIR / "stdin.json") as f:
    FEWSHOT_STDIN = json.load(f)


def pass_at_1(
    references: Union[str, List[str]],
    predictions: Union[str, List[List[str]]],
) -> float:
    if isinstance(predictions, str):
        predictions = [[predictions]]
    if not predictions or not predictions[0]:
        return 0.0

    code = predictions[0][0]
    ref_obj = json.loads(references[0])
    sample = _build_lcb_sample(ref_obj, include_private=False)
    results, _ = run_code_generation_sandboxed(sample, code, timeout=6)
    return 1.0 if _results_passed(results) else 0.0


def pass_at_1_private_tests(
    references: Union[str, List[str]],
    predictions: Union[str, List[List[str]]],
) -> float:
    if isinstance(predictions, str):
        predictions = [[predictions]]
    if not predictions or not predictions[0]:
        return 0.0

    code = predictions[0][0]
    ref_obj = json.loads(references[0])
    sample = _build_lcb_sample(ref_obj, include_private=True)
    results, _ = run_code_generation_sandboxed(sample, code, timeout=6)
    return 1.0 if _results_passed(results) else 0.0


def get_base_model_question_template_answer(question: dict):
    if question.get("starter_code"):
        examples_json = FEWSHOT_FUNC
    else:
        examples_json = FEWSHOT_STDIN

    def get_example_prompt(example):
        prompt = ""
        prompt += "### Question\n"
        prompt += example["question"]
        prompt += "\n\n"
        if question.get("starter_code"):
            prompt += "### Starter Code\n"
            prompt += example["sample_code"]
            prompt += "\n\n"
        prompt += "### Answer\n\n"
        if example["answer"]:
            prompt += "```python\n"
            prompt += example["answer"]
            prompt += "```"
            prompt += "\n\n"
        return prompt

    prompt = ""
    prompt += get_example_prompt(examples_json[0])
    prompt += get_example_prompt(
        {
            "question": question.get("question_content"),
            "sample_code": question.get("starter_code"),
            "answer": "",
        }
    )
    prompt += "```python\n"
    return prompt


def decode_private_test_cases(
    private_test_cases: Union[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    if not private_test_cases:
        return []
    if isinstance(private_test_cases, list):
        return private_test_cases

    try:
        parsed = json.loads(private_test_cases)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

    payload = private_test_cases.encode("utf-8")
    payload += b"=" * (-len(payload) % 4)
    decoded = pickle.loads(zlib.decompress(base64.b64decode(payload)))
    if isinstance(decoded, str):
        decoded = json.loads(decoded)
    if not isinstance(decoded, list):
        raise ValueError("Decoded private_test_cases must be a list")
    return decoded


def _build_lcb_sample(ref_obj: Dict[str, Any], include_private: bool) -> Dict[str, str]:
    test_cases = list(ref_obj.get("public_test_cases", []))
    if include_private:
        test_cases.extend(
            decode_private_test_cases(ref_obj.get("private_test_cases", ""))
        )
    return {
        "input_output": json.dumps(
            {
                "inputs": [case["input"] for case in test_cases],
                "outputs": [case["output"] for case in test_cases],
                "fn_name": ref_obj.get("func_name") or None,
            }
        )
    }


def _results_passed(results: List[Union[bool, int]]) -> bool:
    return all(result > 0 for result in results)


def build_reference(doc: Dict[str, Any]) -> str:
    meta = json.loads(doc["metadata"])
    function_name = meta.get("func_name", "")
    starter_code = doc.get("starter_code", "")
    private_test_cases = doc.get("private_test_cases", "")

    if not starter_code:
        return json.dumps(
            {
                "mode": "stdin",
                "public_test_cases": json.loads(doc["public_test_cases"]),
                "private_test_cases": private_test_cases,
                "func_name": function_name,
            }
        )

    public_tests = json.loads(doc["public_test_cases"])
    return json.dumps(
        {
            "mode": "func",
            "public_test_cases": public_tests,
            "private_test_cases": private_test_cases,
            "func_name": function_name,
        }
    )


def _extract_code(text: str) -> str:
    text = text.strip()
    full_match = FULL_FENCE_RE.search(text)
    if full_match:
        return full_match.group(1).strip()
    open_match = OPEN_FENCE_RE.match(text)
    if open_match:
        return text[open_match.end() :].strip()
    if text.endswith("```"):
        return text[:-3].rstrip()
    return text


def strip_backticks(resps: List[List[str]], docs: List[Dict]) -> List[List[str]]:
    result: List[List[str]] = []

    for resp in resps:
        parsed_resp: List[str] = []
        for r in resp:
            parsed_resp.append(_extract_code(r))
        result.append(parsed_resp)

    return result


def score_all_code_responses(resps: List[List[str]], docs: List[Dict]) -> List[List[str]]:
    result: List[List[str]] = []
    for resp, doc in zip(resps, docs):
        strings = resp if resp else [""]
        ref_obj = json.loads(build_reference(doc))
        sample = _build_lcb_sample(ref_obj, include_private=True)
        scores = []
        for r in strings:
            code = _extract_code(r)
            try:
                results, _ = run_code_generation_sandboxed(sample, code, timeout=6)
                scores.append(1 if _results_passed(results) else 0)
            except Exception:
                scores.append(0)
        result.append([json.dumps(scores)])
    return result


def pass_at_k(references: List, predictions: List, k=None) -> dict:
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


def mean_at_k(references: List, predictions: List, k=None) -> dict:
    scores = json.loads(predictions[0][0])
    n = len(scores)
    if k is None:
        k_list = [n]
    elif isinstance(k, int):
        k_list = [k]
    else:
        k_list = k
    return {f"mean@{ki}": sum(scores[:ki]) / ki for ki in k_list if ki <= n}
