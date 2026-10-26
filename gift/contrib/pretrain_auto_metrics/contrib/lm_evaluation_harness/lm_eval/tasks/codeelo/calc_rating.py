import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


NEWCOMER_RATING = 1400
RATING_BOUNDS = (0.0, 4500.0)
ORDERINGS = 1000
SEED = 1234


def failed_attempts(verdicts: List[str], rng: random.Random) -> Optional[int]:
    order = list(verdicts)
    rng.shuffle(order)
    for failed, verdict in enumerate(order):
        if verdict == "OK":
            return failed
    return None


def model_result(
    summary: Dict[str, Any], verdicts_by_prob: Dict[str, List[str]], rng: random.Random
) -> Tuple[Tuple[float, float], int]:
    icpc = summary["scoring"] == "ICPC"
    points = 0.0
    penalty = 0.0
    unmatched = 0
    for problem in summary["problems"]:
        verdicts = verdicts_by_prob.get(f"{summary['contest_id']}{problem['index']}")
        if not verdicts:
            continue
        failed = failed_attempts(verdicts, rng)
        if failed is None:
            continue
        median = problem["medians"].get(str(failed))
        if median is None:
            unmatched += 1
            continue
        if icpc:
            points += 1
            penalty += median[1]
        else:
            points += median[1]
    return (points, penalty), unmatched


def rating_for_wins(wins: float, ratings: np.ndarray) -> float:
    lo, hi = RATING_BOUNDS
    for _ in range(60):
        mid = (lo + hi) / 2
        expected = np.sum(1.0 / (1.0 + 10.0 ** ((ratings - mid) / 400.0)))
        if expected < wins:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def contest_rating(
    summary: Dict[str, Any],
    verdicts_by_prob: Dict[str, List[str]],
    orderings: int = ORDERINGS,
    seed: int = SEED,
) -> Tuple[Optional[float], int]:
    participants = np.asarray(summary["participants"], dtype=float)
    if not len(participants):
        return None, 0
    points, penalty, ratings = participants.T
    ratings = np.where(ratings == 0, NEWCOMER_RATING, ratings)
    rng = random.Random(seed)
    by_result = {}
    total = 0.0
    unmatched = 0
    for _ in range(orderings):
        result, missing = model_result(summary, verdicts_by_prob, rng)
        unmatched += missing
        if result not in by_result:
            model_points, model_penalty = result
            wins = np.sum(
                (model_points > points)
                | ((model_points == points) & (model_penalty < penalty))
            )
            by_result[result] = rating_for_wins(wins, ratings)
        total += by_result[result]
    return total / orderings, unmatched
