"""
HCT-QA: Human-Centric Tables Question Answering
https://arxiv.org/abs/2504.20047
Dataset: qcri-ai/HCTQA
"""

import re
from typing import Any

import datasets


# ---------------------------------------------------------------------------
# Dataset filtering by dataset_type
# ---------------------------------------------------------------------------

def filter_real(dataset: datasets.Dataset) -> datasets.Dataset:
    return dataset.filter(lambda x: x["dataset_type"] == "realWorldHCTs")


def filter_synthetic(dataset: datasets.Dataset) -> datasets.Dataset:
    return dataset.filter(lambda x: x["dataset_type"] == "syntheticHCTs")


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_FORMAT_INSTRUCTION = (
    "Answer the question based solely on the provided table.\n"
    "Format: {value1 | value2} || {value3 | value4} "
    "(rows separated by \" || \", columns within a row by \" | \", "
    "single value: {value})."
)


def doc_to_text_markdown(doc: dict) -> str:
    return (
        _FORMAT_INSTRUCTION
        + "\n\nTable (Markdown):\n"
        + doc["table_as_markdown"]
        + "\n\nQuestion: "
        + doc["question"]
        + "\nAnswer:"
    )


def doc_to_text_csv(doc: dict) -> str:
    return (
        _FORMAT_INSTRUCTION
        + "\n\nTable (CSV):\n"
        + doc["table_as_csv"]
        + "\n\nQuestion: "
        + doc["question"]
        + "\nAnswer:"
    )


def doc_to_text_html(doc: dict) -> str:
    return (
        _FORMAT_INSTRUCTION
        + "\n\nTable (HTML):\n"
        + doc["table_as_html"]
        + "\n\nQuestion: "
        + doc["question"]
        + "\nAnswer:"
    )


# ---------------------------------------------------------------------------
# Answer parsing
# ---------------------------------------------------------------------------

def _parse_answer(text: str) -> frozenset:
    """
    Parse the GT/prediction answer format into a frozenset of frozensets.

    Expected format: "{value1 | value2} || {value3 | value4}"
      - Each {...} block is a result row.
      - Values inside a row are separated by " | ".

    Also handles the brace-less fallback: "value1 | value2 || value3 | value4"
    """
    text = text.strip()

    brace_matches = list(re.finditer(r"\{([^}]+)\}", text))
    if brace_matches:
        rows = set()
        for m in brace_matches:
            cells = frozenset(
                c.strip().lower() for c in m.group(1).split("|") if c.strip()
            )
            if cells:
                rows.add(cells)
        return frozenset(rows)

    # Fallback: split by "||"
    rows = set()
    for row_str in re.split(r"\s*\|\|\s*", text):
        row_str = row_str.strip()
        if row_str:
            cells = frozenset(
                c.strip().lower() for c in row_str.split("|") if c.strip()
            )
            if cells:
                rows.add(cells)
    return frozenset(rows)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def process_results(doc: dict, results: list[Any]) -> dict[str, float]:
    pred_raw = str(results[0]).strip()
    gold_raw = str(doc["answer"]).strip()

    pred = _parse_answer(pred_raw)
    gold = _parse_answer(gold_raw)

    # Exact match (set-semantic: order of rows does not matter)
    em = float(pred == gold)

    # Row-level set F1
    if not gold and not pred:
        set_f1 = 1.0
    elif not gold or not pred:
        set_f1 = 0.0
    else:
        tp = len(gold & pred)
        precision = tp / len(pred)
        recall = tp / len(gold)
        set_f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

    return {"exact_match": em, "set_f1": set_f1}
