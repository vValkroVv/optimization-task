import json
import re
from functools import partial
from math import comb


choices = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

_ANS_RE = re.compile(r"answer is \(?([ABCDEFGHIJ])\)?", re.IGNORECASE)


def format_cot_example(example, including_answer=True):
    prompt = "Question:\n"
    question = example["question"]
    options = example["options"]
    prompt += question + "\n"
    prompt += "Options:\n"

    for i, opt in enumerate(options):
        if i >= len(choices):
            break
        prompt += "{}. {}\n".format(choices[i], opt)

    if including_answer:
        cot_content = example["cot_content"].replace(
            "A: Let's think step by step.", "Answer: Let's think step by step."
        )
        prompt += cot_content + "\n\n"
    else:
        prompt += "Answer: Let's think step by step."

    return prompt


doc_to_text = partial(format_cot_example, including_answer=False)
fewshot_to_text = partial(format_cot_example, including_answer=True)


def process_docs(dataset, subject):
    return dataset.filter(lambda x: x["category"] == subject)


process_biology = partial(process_docs, subject="biology")
process_business = partial(process_docs, subject="business")
process_chemistry = partial(process_docs, subject="chemistry")
process_computer_science = partial(process_docs, subject="computer science")
process_economics = partial(process_docs, subject="economics")
process_engineering = partial(process_docs, subject="engineering")
process_health = partial(process_docs, subject="health")
process_history = partial(process_docs, subject="history")
process_law = partial(process_docs, subject="law")
process_math = partial(process_docs, subject="math")
process_other = partial(process_docs, subject="other")
process_philosophy = partial(process_docs, subject="philosophy")
process_physics = partial(process_docs, subject="physics")
process_psychology = partial(process_docs, subject="psychology")


def _check_single(response: str, doc: dict) -> int:
    m = _ANS_RE.search(response)
    pred = m.group(1).upper() if m else ""
    gold = str(doc.get("answer", "")).upper()
    return 1 if pred == gold else 0


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
