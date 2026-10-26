import json
import math
from collections import defaultdict
from pathlib import Path


def task_rewards(
    result_path: Path,
    attempts: int,
    expected: int,
    *,
    binary: bool = False,
    reward_key: str | None = None,
) -> dict[str, list[float]]:
    """Read trial rewards, counting missing rewards as zero after publication checks."""
    result = json.loads(result_path.read_text())
    if expected <= 0 or result.get("n_total_trials") != expected:
        raise ValueError("selected Harbor job has a different trial population")
    grouped: dict[str, list[float]] = defaultdict(list)
    for path in sorted(result_path.parent.glob("*/result.json")):
        trial = json.loads(path.read_text())
        task = trial["task_name"]
        rewards = (trial.get("verifier_result") or {}).get("rewards") or {}
        if reward_key is not None:
            rewards = {reward_key: rewards.get(reward_key)}
        values = [
            float(value)
            for value in rewards.values()
            if isinstance(value, (int, float))
        ]
        if all(value is None for value in rewards.values()):
            values = [0.0]
        if not values or not all(math.isfinite(value) for value in values):
            raise ValueError(f"ungraded or invalid trial at {path}")
        grouped[task].append(
            float(any(value == 1 for value in values)) if binary else max(values)
        )
    if sum(map(len, grouped.values())) != expected or any(
        len(values) != attempts for values in grouped.values()
    ):
        raise ValueError("incomplete Harbor task/attempt population")
    return dict(grouped)
