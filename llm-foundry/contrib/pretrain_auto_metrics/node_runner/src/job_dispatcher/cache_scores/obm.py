import hashlib
import json
import math
import re
from pathlib import Path
from typing import Iterable, NamedTuple, Optional

from ...logger import Logger
from .base import (
    CacheScoresBuilder,
    CacheScoresSpec,
    decoded_name,
    encoded_path,
    iter_jsonl,
)

OBM_CACHE_SCORES_FIELDS = [
    "uid",
    "matched",
    "matched_overall",
    "negative",
    "em_score",
    "f1_score",
    "score",
    "scores",
    "n_criteria",
    "n_parse_failures",
    "label",
    "subset",
    "category",
    "subcategory",
    "domain",
    "task_type",
    "difficulty",
    "source",
]

ROLLOUT_METRIC = re.compile(r"/(?P<reducer>pass|mean)@(?P<k>\d+)$")
HEADLINE_TOLERANCE = 1e-6


def obm_cache_files(paths: Iterable[Path]) -> dict[str, Path]:
    files = {}
    for path in paths:
        name = Path(decoded_name(path))
        if name.suffix == ".jsonl":
            files[name.stem] = path
    return files


class FullKeyVariant(NamedTuple):
    key: str
    stem: str
    metric: str
    value: float
    size: int | None
    population: dict | None
    identity: dict


class ObmCacheScoresBuilder(CacheScoresBuilder):
    """Legacy rows under the cache stem, plus full-key ``metric_name: "mean"`` rows.

    A full-key row carries the per-task score of one published metric
    (``aime_2025/accuracy/pass@8`` -> ``score`` 0/1) for metrics the legacy row does not
    express: pass@k / mean@k over rollouts, flat rollouts, keyword_score and function
    calling. Rows of a metric are written only when their mean reproduces the published
    value; a failing metric loses only its own rows.
    """

    cache_subdir = "cache"

    def __init__(self, metrics_dir: Path, run_files: Optional[set[Path]] = None):
        super().__init__(
            metrics_dir,
            CacheScoresSpec(
                raw_fields=OBM_CACHE_SCORES_FIELDS, merge_keys=("benchmark",)
            ),
            run_files,
        )
        self._cache_files = obm_cache_files(self._glob(f"{self.cache_subdir}/*"))
        self._metric_name_map = self._build_metric_name_map()
        self._variants = self._build_variants()
        if self.run_files is not None:
            self._variants = [v for v in self._variants if v.stem in self._cache_files]

    def _all_metrics_path(self) -> Path:
        path = self.metrics_dir / "all_metrics.jsonl"
        return path if path.is_file() else encoded_path(path)

    def iter_source_files(self):
        return list(self._cache_files.values())

    def covered_keys(self):
        return {(stem,) for stem in self._cache_files} | {
            (variant.key,) for variant in self._variants
        }

    def derived_fields(self, source_file: Path) -> dict:
        benchmark = Path(decoded_name(source_file)).stem
        derived = {"benchmark": benchmark}
        metric_name = self._metric_name_map.get(benchmark)
        if metric_name:
            derived["metric_name"] = metric_name
        return derived

    def _merge_existing(self, writer, local_keys: set) -> int:
        return self._write_full_key_rows(writer) + super()._merge_existing(
            writer, local_keys
        )

    def _write_full_key_rows(self, writer) -> int:
        n_rows = 0
        for variant in self._variants:
            cache_path = self._cache_files.get(variant.stem)
            if cache_path is None:
                continue
            # New rows must never cost the legacy rows or the other metrics.
            try:
                records = self._full_key_rows(variant, cache_path)
            except Exception as e:
                Logger.error(
                    f"{cache_path.name}/{variant.metric}: "
                    f"no full-key cache_scores rows: {e}"
                )
                continue
            for record in records:
                writer.write(
                    (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
                )
                n_rows += 1
        return n_rows

    def _full_key_rows(self, variant: FullKeyVariant, cache_path: Path) -> list[dict]:
        label = f"{cache_path.name}/{variant.metric}"
        population = variant.population or {}
        expected = population.get("rows", {})
        if (
            population.get("schema_version") != 1
            or not expected
            or len(expected) != variant.size
        ):
            Logger.warning(f"{label}: no verified cache population; skipping CI")
            return []
        records: dict[str, dict] = {}
        for row in iter_jsonl(cache_path):
            uid = str(row["uid"])
            digest = hashlib.sha256(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                ).encode("utf-8")
            ).hexdigest()
            if expected.get(uid) != digest:
                continue
            score = self._score(variant.metric, row)
            if score is None or not math.isfinite(score):
                if records:
                    raise ValueError(f"row uid={row.get('uid')!r} is not scorable")
                return []
            record = {"benchmark": variant.key}
            if "subset" in row:
                record["subset"] = row["subset"]
            record.update(uid=uid, metric_name="mean", score=score)
            records[uid] = record
        if set(records) != set(expected):
            Logger.warning(f"{label}: incomplete current cache; skipping CI")
            return []
        mean = sum(record["score"] for record in records.values()) / len(records)
        if (
            not math.isfinite(variant.value)
            or abs(mean - variant.value) > HEADLINE_TOLERANCE
        ):
            Logger.warning(
                f"{label}: per-task mean {mean} != published {variant.value}; "
                "no full-key cache_scores rows"
            )
            return []
        namespace = hashlib.sha256(
            json.dumps(
                [
                    variant.key,
                    variant.identity,
                    sorted((uid, row.get("subset")) for uid, row in records.items()),
                ],
                sort_keys=True,
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()
        for uid, record in records.items():
            record["uid"] = f"{namespace}:{uid}"
        return list(records.values())

    @staticmethod
    def _score(metric: str, row: dict) -> Optional[float]:
        rollouts = row.get("rollouts")
        if match := ROLLOUT_METRIC.search(metric):
            count = int(match["k"])
            if not isinstance(rollouts, list) or len(rollouts) < count:
                return None
            matched = [rollout["matched"] for rollout in rollouts[:count]]
            if match["reducer"] == "pass":
                return float(any(matched))
            return sum(matched) / len(matched)
        if isinstance(rollouts, list):
            # Flat rollouts publish accuracy over all attempts; the parent row's
            # `matched` is only the first attempt.
            return sum(rollout["matched"] for rollout in rollouts) / len(rollouts)
        if metric == "keyword_score":
            return float(row["score"])
        if isinstance(row.get("matched"), dict):
            # Function calling publishes accuracy of the call name.
            return float(row["matched"].get("name") is True)
        return None

    def _build_variants(self) -> list[FullKeyVariant]:
        all_metrics_path = self._all_metrics_path()
        variants: list[FullKeyVariant] = []
        if not all_metrics_path.is_file():
            return variants
        for entry in iter_jsonl(all_metrics_path):
            name = entry.get("name")
            if not name:
                continue
            metrics = ([entry["metric"]] if entry.get("metric") else []) + [
                metric
                for metric in entry.get("metrics") or []
                if isinstance(metric, dict)
            ]
            for metric in metrics:
                metric_name, value = metric.get("name"), metric.get("value")
                if metric_name is not None and isinstance(value, (int, float)):
                    # arxiv_math reports a macro mean across subsets, not a task mean.
                    if "accuracy_ci_95" in (metric.get("additional") or {}):
                        continue
                    variants.append(
                        FullKeyVariant(
                            f"{name}/{metric_name}",
                            name,
                            metric_name,
                            value,
                            entry.get("size"),
                            entry.get("cache_population"),
                            self._comparison_identity(entry),
                        )
                    )
        return variants

    @staticmethod
    def _comparison_identity(entry: dict) -> dict:
        details = entry.get("implementation_details") or {}
        return {
            "version": entry.get("version"),
            "dataset": {
                key: value
                for key, value in (details.get("dataset") or {}).items()
                if key != "cache_fingerprint"
            },
            "benchmark": details.get("benchmark"),
            "solver": {
                key: value
                for key, value in (details.get("solver") or {}).items()
                if key != "model"
            },
            "judge": details.get("judge"),
        }

    def _build_metric_name_map(self) -> dict[str, str]:
        all_metrics_path = self._all_metrics_path()
        mapping: dict[str, str] = {}
        if not all_metrics_path.is_file():
            return mapping
        for entry in iter_jsonl(all_metrics_path):
            name = entry.get("name")
            metric_name = (entry.get("metric") or {}).get("name")
            if name and metric_name:
                mapping[name] = metric_name
        return mapping
