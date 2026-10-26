import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator, Optional

from ...logger import Logger
from .base import CacheScoresBuilder, CacheScoresSpec, iter_jsonl

SCHEMA_VERSION = 1
SCORING_KIND = "mean"
_UID_HASH_LENGTH = 12
_TIMESTAMP_RE = re.compile(r"_(\d{4}-\d{2}-\d{2}T[\d.\-]+)$")


def _timestamp(stem: str) -> Optional[str]:
    match = _TIMESTAMP_RE.search(stem)
    return match.group(1) if match else None


class LmEvalCacheScoresBuilder(CacheScoresBuilder):
    def __init__(
        self,
        metrics_dir: Path,
        evaluator,
        run_files: Optional[set[Path]] = None,
    ):
        super().__init__(
            metrics_dir,
            CacheScoresSpec(merge_keys=("benchmark", "subset")),
            run_files,
        )
        self.evaluator = evaluator
        self._sources_without_doc_hash: set[str] = set()
        self._plan, self._covered = self._build_plan()

    def iter_source_files(self) -> list[Path]:
        return self._glob("samples_*.jsonl")

    def existing_rows(self) -> Iterable[dict]:
        for row in super().existing_rows():
            if row.get("schema_version") == SCHEMA_VERSION:
                yield row

    def derived_fields(self, source_file: Path) -> dict:
        rules = self._plan.get(source_file.name)
        if rules is None:
            if _timestamp(source_file.stem) is None:
                Logger.warning(f"No run timestamp in {source_file.name}; skipping")
            else:
                Logger.error(f"{source_file.name}: no results file describes it")
            rules = []
        return {
            "source": source_file.name,
            "rules": rules,
            "seen": Counter(),
        }

    @staticmethod
    def _owning_task(task_name: str, tasks: set[str]) -> Optional[str]:
        if task_name in tasks:
            return task_name
        prefixes = [task for task in tasks if task_name.startswith(f"{task}_")]
        return max(prefixes, key=len) if prefixes else None

    def covered_keys(self) -> set[tuple]:
        return self._covered

    def iter_records(self, row: dict, derived: dict) -> Iterator[dict]:
        doc_hash = row.get("doc_hash")
        if not isinstance(doc_hash, str) or not doc_hash:
            source = derived["source"]
            if source not in self._sources_without_doc_hash:
                self._sources_without_doc_hash.add(source)
                Logger.error(f"{source}: samples without doc_hash; no scorables for it")
            return
        row_filter = row.get("filter", "none")
        uid = self._uid(doc_hash, row_filter, derived["seen"])
        for benchmark, subset, metric_name, filter_name, scorable in derived["rules"]:
            if not scorable or filter_name != row_filter:
                continue
            score = row.get(metric_name)
            if not isinstance(score, (int, float)) or not math.isfinite(score):
                continue
            yield {
                "schema_version": SCHEMA_VERSION,
                "scoring_kind": SCORING_KIND,
                "benchmark": benchmark,
                "subset": subset,
                "uid": uid,
                "metric_name": metric_name,
                "score": float(score),
            }

    @staticmethod
    def _uid(doc_hash: str, row_filter: str, seen: Counter) -> str:
        doc_hash = doc_hash[:_UID_HASH_LENGTH]
        occurrence = seen[(row_filter, doc_hash)]
        seen[(row_filter, doc_hash)] += 1
        return f"{doc_hash}#{occurrence}"

    def _build_plan(self) -> tuple[dict, set[tuple]]:
        runs = []
        sources: dict = {}
        for results_file in self._glob("*.json"):
            timestamp = _timestamp(results_file.stem)
            if timestamp is None:
                continue
            try:
                run = json.loads(results_file.read_text(encoding="utf-8"))
            except (OSError, ValueError) as e:
                Logger.error(f"Unreadable lm_eval results {results_file}: {e}")
                continue
            results = run.get("results")
            group_subtasks = run.get("group_subtasks")
            if not results or not group_subtasks:
                continue
            leaves = {
                leaf
                for group_name in group_subtasks
                for leaf in self._leaf_subtasks(group_subtasks, group_name)
            }
            run_sources, totals = self._read_samples(timestamp, leaves)
            sources.update(run_sources)
            rules = self._rules_for_run(run, totals)
            run_date = run.get("date")
            if not isinstance(run_date, (int, float)):
                run_date = results_file.stat().st_mtime
            runs.append((run_date, results_file.name, timestamp, rules))

        owners = {}
        for run_date, file_name, timestamp, rules in sorted(runs, key=lambda r: r[:2]):
            for _, rule in rules:
                owners[rule[0]] = timestamp

        rules_by_leaf: dict = {}
        for _, _, timestamp, rules in runs:
            for leaf, rule in rules:
                entry = rules_by_leaf.setdefault((timestamp, leaf), [])
                if owners[rule[0]] == timestamp and rule not in entry:
                    entry.append(rule)
        return (
            {name: rules_by_leaf.get(key, []) for name, key in sources.items()},
            {rule[:2] for rules in rules_by_leaf.values() for rule in rules},
        )

    def _read_samples(self, timestamp: str, leaves: set[str]) -> tuple[dict, dict]:
        sources: dict = {}
        totals: dict = {}
        for source_file in self._glob(f"samples_*_{timestamp}.jsonl"):
            stem = source_file.stem
            leaf = self._owning_task(
                stem[len("samples_") : -(len(timestamp) + 1)], leaves
            )
            if leaf is None:
                continue
            sources[source_file.name] = (timestamp, leaf)
            leaf_totals = totals.setdefault(leaf, {})
            for row in iter_jsonl(source_file):
                doc_hash = row.get("doc_hash")
                if not isinstance(doc_hash, str) or not doc_hash:
                    continue
                row_filter = row.get("filter", "none")
                for name, value in row.items():
                    if isinstance(value, (int, float)) and math.isfinite(value):
                        scored, count = leaf_totals.get((name, row_filter), (0.0, 0))
                        leaf_totals[(name, row_filter)] = (scored + value, count + 1)
        return sources, totals

    def _rules_for_run(self, run: dict, totals: dict) -> list[tuple[str, tuple]]:
        published = self.evaluator.aggregate_metrics(
            self.evaluator.filter_metrics(self.evaluator.aggregate_single_run(run))
        )
        group_subtasks = run["group_subtasks"]
        num_fewshots = next(iter(run["n-shot"].values())) if run.get("n-shot") else None
        candidates: dict = {}
        for group_name, subtasks in group_subtasks.items():
            task_names = [group_name]
            if group_name in self.evaluator.GROUPS_WITH_SUBTASK_METRICS:
                task_names += list(subtasks)
            for task_name in task_names:
                leaves = self._leaf_subtasks(group_subtasks, task_name)
                for submetric_name in run["results"].get(task_name, {}):
                    if submetric_name == "alias" or "_stderr" in submetric_name:
                        continue
                    benchmark = self.evaluator.metric_key(
                        group_name, task_name, num_fewshots, submetric_name
                    )
                    if self.evaluator.should_exclude_metric(benchmark):
                        continue
                    bucketed = benchmark.startswith("ruler/")
                    if bucketed:
                        if task_name == group_name and leaves != [task_name]:
                            continue
                        benchmark = self.evaluator.ruler_bucket_key(benchmark)
                    if benchmark not in published:
                        continue
                    entries = candidates.setdefault((benchmark, submetric_name), [])
                    for leaf in leaves:
                        subset = "ALL" if leaf == task_name and not bucketed else leaf
                        entries.append((leaf, subset))
        return self._rules_for_candidates(candidates, published, totals)

    def _rules_for_candidates(
        self, candidates: dict, published: dict, totals: dict
    ) -> list[tuple[str, tuple]]:
        rules = []
        for (benchmark, submetric_name), entries in candidates.items():
            metric_name, _, filter_name = submetric_name.partition(",")
            filter_name = filter_name or "none"
            scorable = self._pools_to(
                totals,
                {leaf for leaf, _ in entries},
                (metric_name, filter_name),
                published[benchmark],
            )
            if not scorable:
                Logger.warning(f"{benchmark} is not a sample mean; no scorables")
            for leaf, subset in entries:
                rules.append(
                    (leaf, (benchmark, subset, metric_name, filter_name, scorable))
                )
        return rules

    @staticmethod
    def _pools_to(
        totals: dict, leaves: set[str], key: tuple[str, str], published: float
    ) -> bool:
        scored, count = 0.0, 0
        for leaf in leaves:
            leaf_scored, leaf_count = totals.get(leaf, {}).get(key, (0.0, 0))
            scored += leaf_scored
            count += leaf_count
        if not count or not isinstance(published, (int, float)):
            return False
        return math.isclose(scored / count, published, rel_tol=1e-6)

    def _leaf_subtasks(self, group_subtasks: dict, task_name: str) -> list[str]:
        subtasks = group_subtasks.get(task_name)
        if not subtasks:
            return [task_name]
        return [
            leaf
            for subtask in subtasks
            for leaf in self._leaf_subtasks(group_subtasks, subtask)
        ]
