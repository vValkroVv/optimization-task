import math
import os
import re
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import calc_rating  # noqa: E402
from api_client import get_client  # noqa: E402


LANG_MAP = {
    "cpp": 91,
    "python": 70,
    "java": 87,
    "rust": 75,
    "go": 32,
    "kotlin": 88,
    "csharp": 79,
    "javascript": 34,
    "node.js": 55,
    "ruby": 67,
    "d": 28,
    "pascal": 51,
    "haskell": 12,
    "perl": 13,
    "ocaml": 19,
    "delphi": 3,
    "php": 6,
    "scala": 20,
    "c": 43,
}

MAX_SOURCE_CHARS = 65535

METRIC_KEYS = (
    "codeelo_avg_rating",
    "codeelo_pass_at_1",
    "codeelo_pass_at_2",
    "codeelo_pass_at_4",
    "codeelo_pass_at_8",
)


def build_dataset_from_napi(
    contest_ids: Optional[List[int]] = None,
    exclude_problems: Optional[List[str]] = None,
    **_kwargs: Any,
) -> Any:
    from datasets import Dataset, DatasetDict

    if not contest_ids:
        raise RuntimeError(
            "contest_ids is required: --metadata '{\"contest_ids\": [2112, ...]}'"
        )
    contest_ids = [int(cid) for cid in contest_ids]
    excluded = set(exclude_problems or [])

    client = get_client()
    rows: List[Dict[str, Any]] = []
    for cid in contest_ids:
        problems = client.get_problems(cid)
        for p in problems:
            index = p["index"]
            if f"{cid}{index}" in excluded:
                continue
            texts = p.get("texts") or []
            statement = next(
                (t["content"] for t in texts if t.get("language") == "en"),
                texts[0]["content"] if texts else "",
            )
            rows.append(
                {
                    "contest_id": cid,
                    "prob": f"{cid}{index}",
                    "problem_index": index,
                    "title": p.get("englishName") or p.get("russianName") or "",
                    "statement_latex": statement,
                }
            )

    return DatasetDict({"test": Dataset.from_list(rows)})


def doc_to_target_noop(doc: Dict[str, Any]) -> str:
    return ""


def passthrough_filter(
    resps: List[List[str]], docs: List[Dict[str, Any]]
) -> List[List[str]]:
    return resps


ONESHOT_STATEMENT = r"""\begin{problem}{Watermelon}{standard input}{standard output}{1 second}{64 megabytes}

One hot summer day Pete and his friend Billy decided to buy a watermelon. They
chose the biggest and the ripest one, in their opinion. After that the
watermelon was weighed, and the scales showed $w$ kilos. They rushed home,
dying of thirst, and decided to divide the berry, however they faced a hard
problem.

Pete and Billy are great fans of even numbers, that's why they want to divide
the watermelon in such a way that each of the two parts weighs an even number
of kilos. Note that the parts must be positive.

\InputFile

The first (and the only) input line contains integer number $w$
($1 \le w \le 100$) -- the weight of the watermelon bought by the boys.

\OutputFile

Print YES, if the boys can divide the watermelon into two parts, each of them
weighing an even number of kilos; and NO in the opposite case.

\Examples

\exmp{8}{YES}

\end{problem}"""

ONESHOT_SOLUTION = """#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int w;
    cin >> w;
    cout << ((w > 2 && w % 2 == 0) ? "YES" : "NO") << "\\n";
    return 0;
}"""

ONESHOT_TEMPLATE = (
    "### Question\n"
    "{example_statement}\n\n"
    "### Answer\n\n"
    "```cpp\n"
    "{example_solution}\n"
    "```\n\n"
    "### Question\n"
    "{statement}\n\n"
    "### Answer\n\n"
    "```cpp\n"
)

ONESHOT_SOLUTION_PYTHON = """w = int(input())
print("YES" if w > 2 and w % 2 == 0 else "NO")"""

ONESHOT_TEMPLATE_PYTHON = ONESHOT_TEMPLATE.replace("```cpp\n", "```python\n")


def format_oneshot_prompt(doc: Dict[str, Any]) -> str:
    return ONESHOT_TEMPLATE.format(
        example_statement=ONESHOT_STATEMENT,
        example_solution=ONESHOT_SOLUTION,
        statement=doc["statement_latex"],
    )


def format_oneshot_prompt_python(doc: Dict[str, Any]) -> str:
    return ONESHOT_TEMPLATE_PYTHON.format(
        example_statement=ONESHOT_STATEMENT,
        example_solution=ONESHOT_SOLUTION_PYTHON,
        statement=doc["statement_latex"],
    )


def strip_code_fence(
    resps: List[List[str]], docs: List[Dict[str, Any]]
) -> List[List[str]]:
    cleaned = []
    for per_doc in resps:
        out = []
        for gen in per_doc:
            text = gen
            if text.lstrip().startswith("```"):
                text = text.lstrip()[3:]
                for tag in ("cpp", "python"):
                    if text[: len(tag)].lower() == tag:
                        text = text[len(tag) :]
                        break
                text = text.lstrip("\n")
            for marker in ("\n```", "\n### Question"):
                cut = text.find(marker)
                if cut >= 0:
                    text = text[:cut]
            out.append(text)
        cleaned.append(out)
    return cleaned


INSTRUCT_TEMPLATE = (
    "You are an expert {language} programmer. You will be given a competitive "
    "programming problem and will write a correct {language} program that "
    "solves it within the time and memory limits.\n\n"
    "### Question:\n{statement}\n\n"
    "### Format: Read the input from standard input and write the answer to "
    "standard output. Enclose the complete program in a single code block "
    "as follows.\n"
    "```{tag}\n{placeholder}\n```\n\n"
    "### Answer: (use the provided format with backticks)"
)

FENCED_BLOCK_RE = re.compile(r"```[\w+#-]*[ \t]*\n(.*?)```", re.DOTALL)


def format_instruct_prompt(doc: Dict[str, Any]) -> str:
    return INSTRUCT_TEMPLATE.format(
        language="C++",
        tag="cpp",
        placeholder="// YOUR CODE HERE",
        statement=doc["statement_latex"],
    )


def format_instruct_prompt_python(doc: Dict[str, Any]) -> str:
    return INSTRUCT_TEMPLATE.format(
        language="Python",
        tag="python",
        placeholder="# YOUR CODE HERE",
        statement=doc["statement_latex"],
    )


def extract_last_code_block(
    resps: List[List[str]], docs: List[Dict[str, Any]]
) -> List[List[str]]:
    extracted = []
    for per_doc in resps:
        out = []
        for gen in per_doc:
            blocks = FENCED_BLOCK_RE.findall(gen or "")
            out.append(blocks[-1].strip("\n") if blocks else "")
        extracted.append(out)
    return extracted


def submit_oneshot_problem(
    doc: Dict[str, Any], results: List[List[str]]
) -> Dict[str, Dict]:
    if os.environ.get("CODEELO_CACHE_ONLY"):
        empty = {"contest_id": 0, "prob": "", "verdicts": []}
        return {k: empty for k in METRIC_KEYS}
    return _submit_sources(doc, list(results[0] if results else []), "cpp")


def submit_oneshot_problem_python(
    doc: Dict[str, Any], results: List[List[str]]
) -> Dict[str, Dict]:
    if os.environ.get("CODEELO_CACHE_ONLY"):
        empty = {"contest_id": 0, "prob": "", "verdicts": []}
        return {k: empty for k in METRIC_KEYS}
    return _submit_sources(doc, list(results[0] if results else []), "python")


def _submit_sources(
    doc: Dict[str, Any], sources: List[str], lang: str
) -> Dict[str, Dict]:
    client = get_client()
    verdicts: List[str] = []
    for code in sources:
        if not code.strip():
            verdicts.append("NO_CODE")
            continue
        if len(code) > MAX_SOURCE_CHARS:
            verdicts.append("SOURCE_TOO_LONG")
            continue
        verdicts.append(client.judge(doc["prob"], LANG_MAP[lang], code))
    payload = {
        "contest_id": int(doc["contest_id"]),
        "prob": doc["prob"],
        "verdicts": verdicts,
    }
    return {k: payload for k in METRIC_KEYS}


def aggregate_avg_rating(items: List[Dict]) -> float:
    ratings = _per_contest_ratings(items)
    if not ratings:
        return math.nan
    return sum(ratings.values()) / len(ratings)


def _pass_at_k(n: int, c: int, k: int) -> float:
    if c <= 0:
        return 0.0
    if n - c < k:
        return 1.0
    prod = 1.0
    for i in range(n - c + 1, n + 1):
        prod *= 1.0 - k / i
    return 1.0 - prod


def _aggregate_pass_at_k(items: List[Dict], k: int) -> float:
    scores = []
    for p in items:
        verdicts = p["verdicts"]
        n = len(verdicts)
        if n < k:
            continue
        c = sum(1 for v in verdicts if v == "OK")
        scores.append(_pass_at_k(n, c, k))
    return (sum(scores) / len(scores)) if scores else 0.0


def aggregate_pass_at_1(items: List[Dict]) -> float:
    return _aggregate_pass_at_k(items, 1)


def aggregate_pass_at_2(items: List[Dict]) -> float:
    return _aggregate_pass_at_k(items, 2)


def aggregate_pass_at_4(items: List[Dict]) -> float:
    return _aggregate_pass_at_k(items, 4)


def aggregate_pass_at_8(items: List[Dict]) -> float:
    return _aggregate_pass_at_k(items, 8)


def _per_contest_ratings(items: List[Dict]) -> Optional[Dict[int, float]]:
    verdicts_by_prob: Dict[str, List[str]] = defaultdict(list)
    for p in items:
        verdicts_by_prob[p["prob"]].extend(p["verdicts"])
    client = get_client()
    out: Dict[int, float] = {}
    for cid in sorted({p["contest_id"] for p in items}):
        try:
            summary = client.standings(cid)
        except Exception as e:
            print(
                f"[codeelo] standings for contest {cid} failed, rating dropped: {e!r}",
                file=sys.stderr,
            )
            return None
        rating, unmatched = calc_rating.contest_rating(summary, verdicts_by_prob)
        print(
            f"[codeelo] contest {cid}: rating={rating} "
            f"participants={len(summary['participants'])} unmatched={unmatched}",
            file=sys.stderr,
        )
        if rating is None:
            print(
                f"[codeelo] contest {cid} has no rated participants, rating dropped",
                file=sys.stderr,
            )
            return None
        out[cid] = rating
    return out
