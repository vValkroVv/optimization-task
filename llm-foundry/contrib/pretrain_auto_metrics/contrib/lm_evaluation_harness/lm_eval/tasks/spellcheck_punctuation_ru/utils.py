import json
from typing import Dict, List

from sage.evaluation import Scorer

from lm_eval.api.filter import Filter
from lm_eval.api.registry import register_filter


@register_filter("sage_scorer")
class SageScorer(Filter):
    def __init__(self):
        self.scorer = Scorer()

    def apply(self, results, docs) -> List[Dict[str, float]]:
        golds = [doc["correction"] for doc in docs]
        srcs = [doc["source"] for doc in docs]
        answers = [resp[0] for resp in results]

        scores = self.scorer.score(
            sources=srcs, corrections=golds, answers=answers, metrics=["errant"]
        )
        scores = {k: v / 100.0 for k, v in scores.items()}
        f1_score = [
            scores["SPELL_F1"],
            scores["CASE_F1"],
            scores["PUNCT_F1"],
            scores["YO_F1"],
        ]
        f1_score = sum(f1_score) / len(f1_score)
        scores["F1"] = f1_score
        return [scores] * len(results)


def process_results(doc, results):
    """
    Returns the LIBRA score consisting of the following metrics:
    SPELL_Precision;
    SPELL_Recall;
    SPELL_F1;
    CASE_Precision;
    CASE_Recall;
    CASE_F1;
    PUNCT_Precision;
    PUNCT_Recall;
    PUNCT_F1;
    YO_Precision;
    YO_Recall;
    YO_F1;
    F1
    Where CASE: erroneously used case;
    SPELL: spelling and grammar errors;
    PUNCT: punctuation errors;
    YO: unnecessary replacement of 'YO' (ё) letter
    """
    return results[0]


_scorer = None


def _get_scorer() -> Scorer:
    global _scorer
    if _scorer is None:
        _scorer = Scorer()
    return _scorer


def _score_single(source: str, gold: str, answer: str) -> float:
    scores = _get_scorer().score(
        sources=[source], corrections=[gold], answers=[answer], metrics=["errant"]
    )
    return (
        scores["SPELL_F1"]
        + scores["CASE_F1"]
        + scores["PUNCT_F1"]
        + scores["YO_F1"]
    ) / 4


def score_all_responses(resps: list, docs: list) -> list:
    result = []
    for resp, doc in zip(resps, docs):
        strings = resp if resp else [""]
        scores = [
            _score_single(doc["source"], doc["correction"], s) for s in strings
        ]
        result.append([json.dumps(scores)])
    return result


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
