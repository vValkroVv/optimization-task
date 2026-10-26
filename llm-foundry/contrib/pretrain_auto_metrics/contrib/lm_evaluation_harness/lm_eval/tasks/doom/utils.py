import json
import re
from fractions import Fraction
from math import comb

try:
    import antlr4
    from math_verify import parse, verify
except (ModuleNotFoundError, AssertionError) as e:
    raise type(e)(
        "`math_verify`"
        "Please install the required packages via pip install lm-eval[math] or pip install -e .[math]"
    ) from e



def doc_to_text_math(doc):
    return (
        "Ты — точный помощник по математике.\n"
        "Решай задачу строго, логично и последовательно.\n"
        "Обязательно:\n"
        "1) Приведи необходимые формулы и промежуточные преобразования.\n"
        "2) Выполняй вычисления по шагам.\n"
        "3) Проверяй корректность преобразований и итоговый знак/число.\n"
        "4) Дай ответ в максимально упрощённом виде.\n"
        "Пиши чётко и без лишних слов.\n"
        "В конце выведи ровно одну строку вида: \\boxed{ответ}.\n\n"
        f"Задача:\n{doc['task']}\n"
    )


def doc_to_text_physics(doc):
    return (
        "Ты — точный помощник по физике.\n"
        "Решай задачу строго, логично и последовательно.\n"
        "Обязательно:\n"
        "1) Запиши основную физическую модель или закон, который применяешь.\n"
        "2) Приведи формулы, а затем аккуратную подстановку чисел в СИ.\n"
        "3) Выполни вычисления по шагам.\n"
        "4) Проверь размерности.\n"
        "5) Дай итог в нужных единицах.\n"
        "Пиши чётко, без лишних слов.\n"
        "В конце выведи ровно одну строку вида: \\boxed{ответ}.\n\n"
        f"Задача:\n{doc['task']}\n"
    )


def doc_to_target(doc):
    return (
        f"{doc['solution']}\n"
        f"\\boxed{{{doc['answer']}}}"
    )


_ANSWER_RE = re.compile(r"(?:^|\n)\s*ответ\s*[:\-]\s*(.+)", re.IGNORECASE)
_LAST_NUM_RE = re.compile(r"(-?\d+(?:[.,]\d+)?(?:\s*/\s*-?\d+(?:[.,]\d+)?)?)")

def extract_last_boxed(text: str) -> str | None:
    if text is None:
        return None
    t = str(text)
    starts = [m for m in re.finditer(r"\\boxed\s*\{", t)]
    if not starts:
        return None

    last_content = None
    for m in starts:
        i = m.end() - 1  
        depth = 0
        content_start = None

        for j in range(i, len(t)):
            c = t[j]
            if c == "{":
                depth += 1
                if depth == 1:
                    content_start = j + 1
            elif c == "}":
                depth -= 1
                if depth == 0 and content_start is not None:
                    last_content = t[content_start:j]
                    break 

    return last_content.strip() if last_content is not None else None


def extract_answer(text: str) -> str:
    if text is None:
        return ""
    t = str(text).strip()

    boxed = extract_last_boxed(t)
    if boxed is not None and boxed != "":
        return boxed

    m = _ANSWER_RE.search(t)
    if m:
        return m.group(1).strip()

    nums = _LAST_NUM_RE.findall(t)
    if nums:
        return nums[-1].strip()

    return t


def norm(s: str) -> str:
    s = s.strip().replace("−", "-").replace(",", ".")
    s = re.sub(r"\s+", " ", s)
    return s.rstrip(".")


def to_number(s: str):
    try:
        s = norm(s)
        if re.fullmatch(r"-?\d+\s*/\s*-?\d+", s):
            a, b = re.split(r"\s*/\s*", s)
            return Fraction(int(a), int(b))
        if re.fullmatch(r"-?\d+", s):
            return int(s)
        if re.fullmatch(r"-?\d+\.\d+", s):
            return float(s)
    except (ValueError, OverflowError, ZeroDivisionError):
        return None


def process_results(doc, results):
    pred = norm(extract_answer(results[0]))
    gold = norm(str(doc["answer"]))

    if pred == gold:
        ok = True
    else:
        pn = to_number(pred)
        gn = to_number(gold)
        ok = (pn is not None and gn is not None and pn == gn)

    _mvres = verify(
        gold=parse(doc["answer"]),
        target=parse(results[0]),
    )

    res = {
        "doom_score": 1.0 if ok else 0.0,
        "math_verify": 1 if _mvres else 0
    }
    return res


def doom_score(items):
    return sum(x["doom_score"] for x in items) / max(len(items), 1)


def _check_single(response: str, doc: dict) -> int:
    pred = norm(extract_answer(response))
    gold = norm(str(doc["answer"]))

    if pred == gold:
        return 1

    pn = to_number(pred)
    gn = to_number(gold)
    if pn is not None and gn is not None and pn == gn:
        return 1

    try:
        return 1 if verify(gold=parse(doc["answer"]), target=parse(response)) else 0
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
