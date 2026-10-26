import json
import logging
import re
import signal
from importlib.metadata import version
from math import comb
from typing import Dict, List, Optional

import datasets


eval_logger = logging.getLogger(__name__)


try:
    import antlr4
    import sympy
    from math_verify import parse, verify
    from sympy.parsing.latex import parse_latex

    assert version("antlr4-python3-runtime").startswith("4.11")
except (ModuleNotFoundError, AssertionError) as e:
    raise type(e)(
        "`sympy`, `math_verify` and `antlr4-python3-runtime==4.11` are required for generating translation task prompt templates. "
        "Please install the required packages via pip install lm-eval[math] or pip install -e .[math]"
    ) from e


# taken from
# https://github.com/wellecks/lm-evaluation-harness/blob/master/lm_eval/tasks/minerva_math.py
def doc_to_text(doc: dict) -> str:
    return "Problem:" + "\n" + doc["problem"] + "\n\n" + "Solution:"


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc: dict) -> dict:
        out_doc = {
            "problem": doc["problem"],
            "solution": doc["solution"],
            "answer": normalize_final_answer(
                remove_boxed(last_boxed_only_string(doc["solution"]))
            ),
        }
        if getattr(doc, "few_shot", None) is not None:
            out_doc["few_shot"] = True
        return out_doc

    return dataset.map(_process_doc)


def list_fewshot_samples() -> list[dict]:
    return [
  {
    "problem": "Найдите область определения выражения  $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$.}",
    "solution": "Выражения под каждым квадратным корнем должны быть неотрицательными. Следовательно, $x-2 \\ge 0$, то есть $x\\ge2$, и $5 - x \\ge 0$, то есть $x \\le 5$. Также знаменатель не может быть равен нулю, поэтому $5-x>0$, откуда $x<5$. Следовательно, область определения выражения — $\\boxed{[2,5)}$.\nОкончательный ответ: $[2,5)$. Надеюсь, это верно.",
    "few_shot": "1"
  },
  {
    "problem": "Если $\\det \\mathbf{A} = 2$ и $\\det \\mathbf{B} = 12$, найдите $\\det (\\mathbf{A} \\mathbf{B}).$",
    "solution": "Имеем $\\det (\\mathbf{A} \\mathbf{B}) = (\\det \\mathbf{A})(\\det \\mathbf{B}) = (2)(12) = \\boxed{24}.$\nОкончательный ответ: $24$. Надеюсь, это верно.",
    "few_shot": "1"
  },
  {
    "problem": "Террел обычно поднимает две гири по 20 фунтов 12 раз. Если вместо этого он использует две гири по 15 фунтов, сколько раз Террелу нужно их поднять, чтобы суммарно поднять тот же вес?",
    "solution": "Если Террел поднимает две гири по 20 фунтов 12 раз, то он поднимает суммарно $2\\cdot 12\\cdot20=480$ фунтов. Если вместо этого он поднимает две гири по 15 фунтов $n$ раз, то он поднимет суммарно $2\\cdot15\\cdot n=30n$ фунтов. Приравнивая к 480 фунтам, найдём $n$:\n\\begin{align*}\n30n&=480\\\\\n\\Rightarrow\\qquad n&=480/30=\\boxed{16}\n\\end{align*}\nОкончательный ответ: $16$. Надеюсь, это верно.",
    "few_shot": "1"
  },
  {
    "problem": "Если система уравнений\n\n\\begin{align*}\n6x-4y&=a,\\\\\n6y-9x &=b.\n\\end{align*}\nимеет решение $(x, y)$, где $x$ и $y$ оба ненулевые,\nнайдите $\\frac{a}{b}$, предполагая, что $b$ ненулевой.",
    "solution": "Если умножить первое уравнение на $-\\frac{3}{2}$, получим\n\n$$6y-9x=-\\frac{3}{2}a.$$Поскольку также известно, что $6y-9x=b$, имеем\n\n$$-\\frac{3}{2}a=b\\Rightarrow\\frac{a}{b}=\\boxed{-\\frac{2}{3}}.$$\nОкончательный ответ: $-\\frac{2}{3}$. Надеюсь, это верно.",
    "few_shot": "1"
  }
]


def process_results(doc: dict, results: List[str]) -> Dict[str, int]:
    candidates = results[0]

    unnormalized_answer = get_unnormalized_answer(candidates)
    answer = normalize_final_answer(unnormalized_answer)

    if is_equiv(answer, doc["answer"]):
        retval = 1
    else:
        retval = 0

    # math_verify
    res = verify(parse(doc["answer"]), parse(candidates))
    mathval = 1 if res else 0

    results = {
        "exact_match": retval,
        "math_verify": mathval,
    }
    return results


def last_boxed_only_string(string: str) -> Optional[str]:
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def remove_boxed(s: str) -> str:
    if not s:
        return ""
    
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"

    assert s[: len(left)] == left
    assert s[-1] == "}"

    return s[len(left) : -1]


class timeout:
    def __init__(self, seconds=1, error_message="Timeout"):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


def is_equiv(x1: str, x2: str) -> bool:
    """
    x1 and x2 are normalized latex string
    """
    try:
        with timeout(seconds=5):
            try:
                parsed_x1 = parse_latex(x1)
                parsed_x2 = parse_latex(x2)
            except (
                sympy.parsing.latex.errors.LaTeXParsingError,
                sympy.SympifyError,
                TypeError,
            ):
                eval_logger.debug(f"couldn't parse one of {x1} or {x2}")
                return False

            try:
                diff = parsed_x1 - parsed_x2
            except TypeError:
                eval_logger.debug(f"couldn't subtract {x1} and {x2}")
                return False

            try:
                if sympy.simplify(diff) == 0:
                    return True
                else:
                    return False
            except ValueError:
                eval_logger.debug(
                    f"Had some trouble simplifying when comparing {x1} and {x2}"
                )
    except TimeoutError:
        eval_logger.debug(f"Timed out comparing {x1} and {x2}")
        return False
    except ImportError as e:
        eval_logger.error(e)
        raise
    except Exception as e:
        eval_logger.debug(f"Failed comparing {x1} and {x2} with {e}")
        return False


def get_unnormalized_answer(text: str) -> str:
    INVALID_ANSWER = "[invalidanswer]"
    end_seq = "Надеюсь, это верно."
    text += end_seq
    match = re.search(
        r"Окончательный ответ: (.*?). Надеюсь, это верно.",
        text,
    )
    if match:
        return match.group(1).strip()
    else:
        return INVALID_ANSWER



SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]
REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "ft",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]

def normalize_final_answer(final_answer: str) -> str:
    """
    Normalize a final answer to a quantitative reasoning question.

    Copied character for character from appendix D of Lewkowycz et al. (2022)
    """
    final_answer = final_answer.split("=")[-1]

    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    # Extract answer that is in LaTeX math, is bold,
    # is surrounded by a box, etc.
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

    # Normalize shorthand TeX:
    #  \fracab -> \frac{a}{b}
    #  \frac{abc}{bef} -> \frac{abc}{bef}
    #  \fracabc -> \frac{a}{b}c
    #  \sqrta -> \sqrt{a}
    #  \sqrtab -> sqrt{a}b
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    # Normalize 100,000 -> 100000
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    return final_answer


def _check_single(response: str, doc: dict) -> int:
    unnormalized = get_unnormalized_answer(response)
    answer = normalize_final_answer(unnormalized)
    return 1 if is_equiv(answer, doc["answer"]) else 0


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
