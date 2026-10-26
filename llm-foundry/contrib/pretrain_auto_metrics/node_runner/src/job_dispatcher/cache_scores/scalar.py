import hashlib
import json
import math
from collections.abc import Iterable, Iterator
from pathlib import Path

from ...logger import Logger
from ..submetrics import MetricValue, is_number
from .base import CacheScoresBuilder, CacheScoresSpec


class ScalarCacheScoresBuilder(CacheScoresBuilder):
    """Keep producer-validated metric samples for JD's cache."""

    def __init__(self, metrics_dir: Path, metrics: dict[str, MetricValue]):
        super().__init__(
            metrics_dir, CacheScoresSpec(merge_keys=("benchmark", "subset"))
        )
        self.source_path = self.metrics_dir / "ci_samples.jsonl"
        self.metrics = {k: v for k, v in metrics.items() if is_number(v)}
        self.records: dict[str, list[dict]] = {}

    def add_metric(
        self,
        benchmark: str,
        samples: Iterable[tuple[str, float]],
        identity: object,
        *,
        tolerance: float = 1e-8,
    ) -> None:
        if benchmark not in self.metrics:
            return
        # JD's frontend also matches parent names to child rows.
        if any(
            key.startswith((benchmark + "/", benchmark + " (")) for key in self.metrics
        ):
            Logger.warning(f"CI unavailable for {benchmark}: ambiguous JD prefix")
            return
        items = list(samples)
        ids = [uid for uid, _ in items]
        if len(items) < 2 or len(set(ids)) != len(ids) or any(not uid for uid in ids):
            Logger.warning(
                f"CI unavailable for {benchmark}: insufficient/duplicate IDs"
            )
            return
        if any(not math.isfinite(value) for _, value in items):
            Logger.warning(f"CI unavailable for {benchmark}: non-finite contribution")
            return
        reconstructed = math.fsum(value for _, value in items) / len(items)
        if not math.isclose(
            reconstructed, self.metrics[benchmark], rel_tol=0, abs_tol=tolerance
        ):
            Logger.warning(
                f"CI unavailable for {benchmark}: contributions average "
                f"{reconstructed}, published score {self.metrics[benchmark]}"
            )
            return
        population = json.dumps(
            [identity, sorted(ids)], sort_keys=True, ensure_ascii=False
        ).encode()
        namespace = hashlib.sha256(population).hexdigest()
        self.records[benchmark] = [
            {
                "schema_version": 1,
                "scoring_kind": "mean",
                "benchmark": benchmark,
                "subset": "ALL",
                "metric_name": benchmark,
                "uid": f"{namespace}:{uid}",
                "score": float(value),
            }
            for uid, value in sorted(items)
        ]

    def add_weighted_metric(
        self,
        benchmark: str,
        categories: dict[str, list[tuple[str, float]]],
        weights: dict[str, float],
        identity: object,
        *,
        tolerance: float = 1e-8,
    ) -> None:
        """Keep category boundaries so JD can resample each category separately."""
        if benchmark not in self.metrics:
            return
        if not weights or any(not categories.get(category) for category in weights):
            Logger.warning(f"CI unavailable for {benchmark}: missing category samples")
            return
        items = [
            (category, uid, value)
            for category in weights
            for uid, value in categories[category]
        ]
        ids = [f"{category}:{uid}" for category, uid, _ in items]
        if (
            len(items) < 2
            or len(set(ids)) != len(ids)
            or any(not uid for _, uid, _ in items)
        ):
            Logger.warning(
                f"CI unavailable for {benchmark}: duplicate/insufficient IDs"
            )
            return
        if any(not math.isfinite(value) for _, _, value in items):
            Logger.warning(f"CI unavailable for {benchmark}: non-finite score")
            return
        reconstructed = math.fsum(
            weights[category]
            * math.fsum(value for _, value in categories[category])
            / len(categories[category])
            for category in weights
        )
        if not math.isclose(
            reconstructed, self.metrics[benchmark], rel_tol=0, abs_tol=tolerance
        ):
            Logger.warning(
                f"CI unavailable for {benchmark}: category means give "
                f"{reconstructed}, published score {self.metrics[benchmark]}"
            )
            return
        population = json.dumps(
            [identity, weights, sorted(ids)], sort_keys=True, ensure_ascii=False
        ).encode()
        namespace = hashlib.sha256(population).hexdigest()
        self.records[benchmark] = [
            {
                # JD's v1 scalar scorer ignores v2 until it supports stratified CI.
                "schema_version": 2,
                "scoring_kind": "weighted_mean",
                "benchmark": benchmark,
                "subset": "ALL",
                "metric_name": benchmark,
                "uid": f"{namespace}:{category}:{uid}",
                "category": category,
                "weight": weights[category],
                "score": value,
            }
            for category, uid, value in sorted(items)
        ]

    def add_ratio_metric(
        self,
        benchmark: str,
        samples: Iterable[tuple[str, float, int]],
        identity: object,
    ) -> None:
        """Keep document totals so JD can divide by sampled token counts."""
        if benchmark not in self.metrics:
            return
        items = list(samples)
        ids = [uid for uid, _, _ in items]
        if (
            len(items) < 2
            or len(set(ids)) != len(ids)
            or any(
                not uid or count <= 0 or not math.isfinite(total)
                for uid, total, count in items
            )
        ):
            Logger.warning(f"CI unavailable for {benchmark}: invalid document samples")
            return
        reconstructed = math.fsum(total for _, total, _ in items) / sum(
            count for _, _, count in items
        )
        if not math.isclose(
            reconstructed, self.metrics[benchmark], rel_tol=0, abs_tol=1e-8
        ):
            Logger.warning(
                f"CI unavailable for {benchmark}: document ratio gives "
                f"{reconstructed}, published score {self.metrics[benchmark]}"
            )
            return
        population = json.dumps(
            [identity, sorted(ids)], sort_keys=True, ensure_ascii=False
        ).encode()
        namespace = hashlib.sha256(population).hexdigest()
        self.records[benchmark] = [
            {
                # JD's v1 scalar scorer ignores v2 until it supports ratio CI.
                "schema_version": 2,
                "scoring_kind": "ratio",
                "benchmark": benchmark,
                "subset": "ALL",
                "metric_name": benchmark,
                "uid": f"{namespace}:{uid}",
                "numerator": total,
                "denominator": count,
            }
            for uid, total, count in sorted(items)
        ]

    def iter_source_files(self) -> list[Path]:
        # Retain the compact evidence locally for transfer and offline inspection.
        self.source_path.parent.mkdir(parents=True, exist_ok=True)
        with self.source_path.open("w", encoding="utf-8") as stream:
            for benchmark in sorted(self.records):
                for row in self.records[benchmark]:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        return [self.source_path]

    def covered_keys(self) -> Iterable[tuple]:
        return ((key, "ALL") for key in self.metrics)

    def derived_fields(self, source_file: Path) -> dict:
        return {}

    def iter_records(self, row: dict, derived: dict) -> Iterator[dict]:
        yield row

    def existing_rows(self) -> Iterable[dict]:
        for row in super().existing_rows():
            if (row.get("schema_version"), row.get("scoring_kind")) in (
                (1, "mean"),
                (2, "weighted_mean"),
                (2, "ratio"),
            ):
                yield row
