from __future__ import annotations

import csv
import json
import math
import os
import re
import statistics
from pathlib import Path
from typing import ClassVar, Optional
from base64 import b64decode

from ...job_dispatcher import JD, SubmetricGroup
from ...job_dispatcher.cache_scores.scalar import ScalarCacheScoresBuilder
from ...config_model import BfclEvaluatorConfig, ModelInfoConfig
from ...job_dispatcher.measurement import MeasurementUpload
from ...logger import Logger
from ...metrics.evaluators.interface import MetricsEvaluator
from ...subprocess_manager import SubprocessManager
from ...subprocess_pool_runner import SubprocessPoolRunner


class BfclEvaluator(MetricsEvaluator):
    LATENCY = True

    RESULT_SUBDIR = "bfcl_result"
    SCORE_SUBDIR = "bfcl_score"
    SBER_SEARCH_URL_ENCODED = (
        "aHR0cHM6Ly9zZWFyY2gtcGx1Z2luLnNiZXJkZXZpY2VzLnJ1L3JldHJpZXZhbA=="
    )
    SBER_SEARCH_API_KEY_ENCODED = "eWFlN1plYXNvM2V1Z2hpNWxvaERhUGVlc2hhNGFoaG8="
    SBER_RLI_GATEWAY_URL_ENCODED = (
        "aHR0cHM6Ly9neW0uc3RhdGljLnNwYWNlL3YxL3Rvb2xzL2V4ZWN1dGU="
    )
    EXCLUDED_BFCL_COLUMNS: ClassVar[frozenset[str]] = frozenset(
        x.strip().lower()
        for x in (
            "rank",
            "model",
            "model link",
            "organization",
            "license",
            "total cost ($)",
            "latency mean (s)",
            "latency standard deviation (s)",
            "latency 95th percentile (s)",
        )
    )
    _SLUG_RE: ClassVar[re.Pattern[str]] = re.compile(r"[^a-z0-9]+")
    OVERALL_COLUMN = "Overall Acc"
    DIAGNOSTIC_COLUMNS_PREFIX = "Format Sensitivity"
    _SCORE_CAT_RE: ClassVar[re.Pattern[str]] = re.compile(
        r"BFCL_v\d+_(.+?)_score\.json$"
    )
    _RESULT_CAT_RE: ClassVar[re.Pattern[str]] = re.compile(
        r"BFCL_v\d+_(.+?)_result\.json$"
    )
    _KEEP_RAW_PASS_AT_K: ClassVar[frozenset[str]] = frozenset(
        {
            "simple_python",
            "simple_java",
            "simple_javascript",
            "irrelevance",
            "live_irrelevance",
            "multi_turn_base",
            "multi_turn_miss_func",
            "multi_turn_miss_param",
            "multi_turn_long_context",
            "web_search_base",
            "web_search_no_snippet",
            "memory_kv",
            "memory_vector",
        }
    )

    _RENAME: ClassVar[dict[str, tuple[str, ...]]] = {
        "non_live_multiple_ast": ("multiple",),
        "non_live_parallel_ast": ("parallel",),
        "non_live_parallel_multiple_ast": ("parallel_multiple",),
        "live_simple_ast": ("live_simple",),
        "live_multiple_ast": ("live_multiple",),
        "live_parallel_ast": ("live_parallel",),
        "live_parallel_multiple_ast": ("live_parallel_multiple",),
        "relevance_detection": ("live_relevance",),
        "memory_recursive_summarization": ("memory_rec_sum",),
    }

    _REPORT_TO_RAW: ClassVar[dict[str, tuple[str, ...]]] = {
        "live_acc": (
            "live_simple",
            "live_multiple",
            "live_parallel",
            "live_parallel_multiple",
        ),
    }

    _REPORT_TO_RAW_MACRO: ClassVar[dict[str, tuple[str, ...]]] = {
        "non_live_simple_ast": ("simple_python", "simple_java", "simple_javascript"),
        "irrelevance_detection": ("irrelevance", "live_irrelevance"),
        "non_live_ast_acc": (
            "non_live_simple_ast",
            "multiple",
            "parallel",
            "parallel_multiple",
        ),
        "memory_acc": ("memory_kv", "memory_vector", "memory_rec_sum"),
        "web_search_acc": ("web_search_base", "web_search_no_snippet"),
        "multi_turn_acc": (
            "multi_turn_base",
            "multi_turn_miss_func",
            "multi_turn_miss_param",
            "multi_turn_long_context",
        ),
        "agentic": ("web_search_acc", "memory_acc"),
    }

    _OVERALL_WEIGHTS: ClassVar[tuple[tuple[str, float], ...]] = (
        ("non_live_ast_acc", 10.0),
        ("live_acc", 10.0),
        ("irrelevance_detection", 10.0),
        ("multi_turn_acc", 30.0),
        ("agentic", 40.0),
    )

    def setup(
        self,
        autometric_runs_config: BfclEvaluatorConfig,
        artifacts_path: Path,
        hf_checkpoint_path: Path,
        model_info: ModelInfoConfig,
    ) -> None:
        self.autometric_runs_config = autometric_runs_config
        self.artifacts_path = Path(artifacts_path)
        self.hf_checkpoint_path = Path(hf_checkpoint_path)
        self.model_info = model_info
        self.all_metrics: dict[str, float] = {}

        self._create_metrics_dir(self.artifacts_path, "bfcl")
        self.bfcl_root = self._resolve_bfcl_root(autometric_runs_config.bfcl_bench_root)
        self._result_dir = self.metrics_dir / self.RESULT_SUBDIR
        self._score_dir = self.metrics_dir / self.SCORE_SUBDIR

        Logger.info(
            f"BfclEvaluator: repo={self.bfcl_root}, base_url={autometric_runs_config.base_url}, "
            f"model={autometric_runs_config.model}, "
            f"test_category={autometric_runs_config.test_category}"
        )

    @classmethod
    def _resolve_bfcl_root(cls, bench_root: str) -> Path:
        root = Path(bench_root)
        if not root.is_absolute():
            pretrain_root = Path(__file__).resolve().parents[4]
            root = pretrain_root / root
        return root

    @classmethod
    def _metric_slug(cls, name: str) -> str:
        s = cls._SLUG_RE.sub("_", name.strip().lower()).strip("_")
        return s or "unknown"

    @classmethod
    def _parse_numeric_cell(cls, raw: str) -> Optional[float]:
        original = (raw or "").strip()
        if not original or original.upper() == "N/A":
            return None
        is_percent = original.endswith("%")
        s = original
        if s.startswith("$"):
            s = s[1:].strip()
        if s.endswith("%"):
            s = s[:-1].strip()
        s = s.replace(",", "")
        try:
            val = float(s)
        except ValueError:
            return None
        if is_percent:
            val /= 100.0
        return val

    @classmethod
    def _row_to_bfcl_metrics(cls, row: dict[str, str]) -> dict[str, float]:
        out: dict[str, float] = {}
        for col, val in row.items():
            if not col or col.strip().lower() in cls.EXCLUDED_BFCL_COLUMNS:
                continue
            num = cls._parse_numeric_cell(val)
            if num is not None:
                out[cls._metric_key(col)] = num
        return out

    @classmethod
    def _metric_key(cls, column: str) -> str:
        return f"BFCL_{cls._metric_slug(column).upper()}"

    @classmethod
    def _submetric_groups(cls, jd_metrics: dict[str, float]) -> list[SubmetricGroup]:
        overall = cls._metric_key(cls.OVERALL_COLUMN)
        diagnostics_prefix = cls._metric_key(cls.DIAGNOSTIC_COLUMNS_PREFIX)
        diagnostics = [key for key in jd_metrics if key.startswith(diagnostics_prefix)]
        members = [
            key for key in jd_metrics if key != overall and key not in diagnostics
        ]
        return [
            SubmetricGroup(
                key=overall, members=members, value=overall, diagnostics=diagnostics
            )
        ]

    def _load_metrics_from_csv(self, score_dir: Path) -> dict[str, float]:
        path = score_dir / "data_overall.csv"
        if not path.is_file():
            Logger.warning(f"BFCL score file not found: {path}")
            return {}
        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            rows = [r for r in reader if r.get("Model")]
        if not rows:
            return {}
        return self._row_to_bfcl_metrics(rows[0])

    def _build_bfcl_env(self, base_url: str) -> dict[str, str]:
        env = {k: str(v) for k, v in os.environ.items()}
        env["OPENAI_BASE_URL"] = base_url
        env["OPENAI_API_KEY"] = "empty"
        env["OPENAI_TIMEOUT"] = str(self.autometric_runs_config.timeout)
        env["OPENAI_MAX_RETRIES"] = str(self.autometric_runs_config.max_retries)
        env["SBER_SEARCH_URL"] = b64decode(self.SBER_SEARCH_URL_ENCODED).decode("utf-8")
        env["SBER_SEARCH_API_KEY"] = b64decode(self.SBER_SEARCH_API_KEY_ENCODED).decode(
            "utf-8"
        )
        env["SBER_RLI_GATEWAY_URL"] = b64decode(
            self.SBER_RLI_GATEWAY_URL_ENCODED
        ).decode("utf-8")

        pam_models_path = os.environ.get("PAM_MODELS_PATH")
        if pam_models_path:
            memory_encoder_path = (
                Path(pam_models_path) / "sentence-transformers--all-MiniLM-L6-v2"
            )
            if memory_encoder_path.is_dir():
                env["BFCL_MEMORY_ENCODER_PATH"] = str(memory_encoder_path)
        return env

    def run(self) -> None:
        cfg = self.autometric_runs_config
        if cfg.num_runs > 1:
            self._run_multi()
            return

        self._result_dir.mkdir(parents=True, exist_ok=True)
        self._score_dir.mkdir(parents=True, exist_ok=True)

        proc_env = self._build_bfcl_env(cfg.require_base_url())
        gen_cmd = [
            cfg.venv_python,
            "-m",
            "bfcl_eval.__main__",
            "generate",
            *cfg.to_generate_bash_list(self._result_dir),
        ]
        ev_cmd = [
            cfg.venv_python,
            "-m",
            "bfcl_eval.__main__",
            "evaluate",
            *cfg.to_evaluate_bash_list(self._result_dir, self._score_dir),
        ]

        Logger.info(f"BFCL generate: {' '.join(gen_cmd)}")
        sm = SubprocessManager(
            gen_cmd,
            self.log_file,
            str(self.bfcl_root),
            run_name="bfcl_generate",
            env=proc_env,
        )
        sm.start()
        sm.wait_for_completion()
        sm.log_error()
        if not sm.is_successful():
            Logger.error(f"bfcl generate failed with exit code {sm.exit_code}")
            return

        Logger.info(f"BFCL evaluate: {' '.join(ev_cmd)}")
        sm2 = SubprocessManager(
            ev_cmd,
            self.log_file,
            str(self.bfcl_root),
            run_name="bfcl_evaluate",
            env=proc_env,
        )
        sm2.start()
        sm2.wait_for_completion()
        sm2.log_error()
        if not sm2.is_successful():
            Logger.error(f"bfcl evaluate failed with exit code {sm2.exit_code}")
            return

        self.all_metrics = self._load_metrics_from_csv(self._score_dir)
        if not self.all_metrics:
            Logger.warning("No numeric metrics parsed from BFCL data_overall.csv")
            return

        Logger.info(f"BFCL metrics ({len(self.all_metrics)} keys)")
        out_json = self.metrics_dir / "bfcl_metrics.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(self.all_metrics, f, indent=2)

    def get_metrics(self) -> dict[str, float]:
        return dict(self.all_metrics)

    def after_run(
        self, metrics: dict[str, float], uploaded: Optional[MeasurementUpload] = None
    ) -> None:
        self._send_metrics_to_jd(metrics)
        tar_path = JD.tar_metrics_dir(
            self.metrics_dir, self.artifacts_path, type(self).__name__
        )
        JD.symlink_metrics_for_transfer(
            tar_path, self.hf_checkpoint_path, type(self).__name__
        )

    @staticmethod
    def _category_samples(
        score_dir: Path, result_dir: Path
    ) -> dict[str, list[tuple[str, float]]]:
        categories = {}
        for path in score_dir.rglob("BFCL_v4_*_score.json"):
            category = path.name.removeprefix("BFCL_v4_").removesuffix("_score.json")
            if category == "format_sensitivity":
                continue
            try:
                rows = [
                    json.loads(line)
                    for line in path.read_text().splitlines()
                    if line.strip()
                ]
                header, failures = rows[0], rows[1:]
                generated_path = result_dir / path.relative_to(score_dir).with_name(
                    path.name.replace("_score.json", "_result.json")
                )
                generated = [
                    json.loads(line)
                    for line in generated_path.read_text().splitlines()
                    if line.strip()
                ]
                ids = [str(row["id"]) for row in generated]
                failed = [str(row["id"]) for row in failures]
                if (
                    not ids
                    or len(ids) != len(set(ids))
                    or len(failed) != len(set(failed))
                    or not set(failed) <= set(ids)
                    or header["total_count"] != len(ids)
                    or header["correct_count"] != len(ids) - len(failed)
                ):
                    Logger.warning(
                        f"BFCL CI unavailable for {category}: inconsistent score population"
                    )
                    continue
                failed_ids = set(failed)
                categories[category] = [
                    (uid, float(uid not in failed_ids)) for uid in ids
                ]
            except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
                Logger.warning(f"BFCL CI unavailable for {category}: {error}")
        return categories

    @staticmethod
    def _metric_weights(categories: dict[str, list]) -> dict[str, dict[str, float]]:
        """BFCL v4's published linear aggregation tree (eval_runner_helper.py)."""
        leaves = {
            "Non-Live Multiple AST": "multiple",
            "Non-Live Parallel AST": "parallel",
            "Non-Live Parallel Multiple AST": "parallel_multiple",
            "Live Simple AST": "live_simple",
            "Live Multiple AST": "live_multiple",
            "Live Parallel AST": "live_parallel",
            "Live Parallel Multiple AST": "live_parallel_multiple",
            "Multi Turn Base": "multi_turn_base",
            "Multi Turn Miss Func": "multi_turn_miss_func",
            "Multi Turn Miss Param": "multi_turn_miss_param",
            "Multi Turn Long Context": "multi_turn_long_context",
            "Web Search Base": "web_search_base",
            "Web Search No Snippet": "web_search_no_snippet",
            "Memory KV": "memory_kv",
            "Memory Vector": "memory_vector",
            "Memory Recursive Summarization": "memory_rec_sum",
            "Relevance Detection": "live_relevance",
        }
        weights = {name: {category: 1.0} for name, category in leaves.items()}
        weights["Non-Live Simple AST"] = {
            category: 1 / 3
            for category in ("simple_python", "simple_java", "simple_javascript")
        }
        weights["Non-Live AST Acc"] = {
            **{category: 1 / 12 for category in weights["Non-Live Simple AST"]},
            **{
                category: 1 / 4
                for category in ("multiple", "parallel", "parallel_multiple")
            },
        }
        live = (
            "live_simple",
            "live_multiple",
            "live_parallel",
            "live_parallel_multiple",
        )
        if all(category in categories for category in live):
            total = sum(len(categories[category]) for category in live)
            weights["Live Acc"] = {
                category: len(categories[category]) / total for category in live
            }
        weights["Multi Turn Acc"] = {
            category: 1 / 4
            for category in (
                "multi_turn_base",
                "multi_turn_miss_func",
                "multi_turn_miss_param",
                "multi_turn_long_context",
            )
        }
        weights["Web Search Acc"] = {
            category: 1 / 2 for category in ("web_search_base", "web_search_no_snippet")
        }
        weights["Memory Acc"] = {
            category: 1 / 3
            for category in ("memory_kv", "memory_vector", "memory_rec_sum")
        }
        weights["Irrelevance Detection"] = {
            category: 1 / 2 for category in ("irrelevance", "live_irrelevance")
        }
        if "Live Acc" in weights:
            weights["Overall Acc"] = {
                category: coefficient * weight
                for name, coefficient in (
                    ("Non-Live AST Acc", 0.1),
                    ("Live Acc", 0.1),
                    ("Irrelevance Detection", 0.1),
                    ("Multi Turn Acc", 0.3),
                    ("Web Search Acc", 0.2),
                    ("Memory Acc", 0.2),
                )
                for category, weight in weights[name].items()
            }
        return weights

    def _cache_scores(self, metrics: dict[str, float]) -> ScalarCacheScoresBuilder:
        scores = ScalarCacheScoresBuilder(self.metrics_dir, metrics)
        cfg = self.autometric_runs_config
        identity = cfg.model_dump(
            mode="json",
            include={"temperature", "repair_json_max_iter", "web_search"},
        )
        if cfg.num_runs > 1:
            return self._multi_run_cache_scores(scores, identity)
        model_dir = cfg.model.replace("/", "_")
        categories = self._category_samples(
            self._score_dir / model_dir, self._result_dir / model_dir
        )
        self._add_category_scores(
            scores, categories, self._metric_weights(categories), identity
        )
        return scores

    def _add_category_scores(
        self,
        scores: ScalarCacheScoresBuilder,
        categories: dict[str, list[tuple[str, float]]],
        metric_weights: dict[str, dict[str, float]],
        identity: dict,
        suffix: str = "",
    ) -> None:
        for column, weights in metric_weights.items():
            if not all(category in categories for category in weights):
                continue
            key = self._metric_key(column) + suffix
            if "WEB_SEARCH" in key:
                key += f"_{self.autometric_runs_config.web_search.upper()}"
            if len(weights) == 1 and next(iter(weights.values())) == 1:
                category = next(iter(weights))
                scores.add_metric(
                    key,
                    [
                        (f"{category}:{uid}", value)
                        for uid, value in categories[category]
                    ],
                    [identity, weights],
                    tolerance=0.0000500001,
                )
            else:
                scores.add_weighted_metric(
                    key,
                    {category: categories[category] for category in weights},
                    weights,
                    identity,
                    tolerance=0.0000500001,
                )

    def _multi_run_cache_scores(
        self, scores: ScalarCacheScoresBuilder, identity: dict
    ) -> ScalarCacheScoresBuilder:
        cfg = self.autometric_runs_config
        model_dir = cfg.model.replace("/", "_")
        runs = []
        for index in range(cfg.num_runs):
            result_dir = self._result_dir / f"run_{index}" / model_dir
            categories = self._category_samples(
                self._score_dir / f"run_{index}" / model_dir, result_dir
            )
            try:
                generated = self._collect_task_ids(result_dir)
            except (OSError, ValueError, TypeError) as error:
                Logger.warning(f"BFCL multi-run CI unavailable: {error}")
                return scores
            generated.pop("format_sensitivity", None)
            population = {
                category: {uid for uid, _ in values}
                for category, values in categories.items()
            }
            if not population or population != generated:
                Logger.warning("BFCL multi-run CI unavailable: incomplete scored run")
                return scores
            runs.append(
                {category: dict(values) for category, values in categories.items()}
            )
            if index == 0:
                expected = population
            elif population != expected:
                Logger.warning("BFCL multi-run CI unavailable: task populations differ")
                return scores

        counts = {
            category: [
                (uid, sum(run[category][uid] for run in runs)) for uid in sorted(ids)
            ]
            for category, ids in expected.items()
        }
        identity = {**identity, "num_runs": cfg.num_runs}
        means = {
            category: [(uid, count / cfg.num_runs) for uid, count in values]
            for category, values in counts.items()
        }
        mean_weights = {category: {category: 1.0} for category in counts}
        mean_weights.update(self._metric_weights(counts))
        per_run_metrics = [
            self._load_metrics_from_csv(self._score_dir / f"run_{index}")
            for index in range(cfg.num_runs)
        ]
        mean_weights = {
            column: weights
            for column, weights in mean_weights.items()
            if all(self._metric_key(column) in metrics for metrics in per_run_metrics)
        }
        self._add_category_scores(scores, means, mean_weights, identity, "_MEAN")

        # Use the same selected-category rollups as the multi-run point metrics.
        weights = {category: {category: 1.0} for category in counts}
        for slug, raw_categories in self._RENAME.items():
            if raw_categories[0] in counts:
                weights[slug] = weights[raw_categories[0]]
        for slug, raw_categories in self._REPORT_TO_RAW.items():
            selected = [category for category in raw_categories if category in counts]
            total = sum(len(counts[category]) for category in selected)
            if total:
                weights[slug] = {
                    category: len(counts[category]) / total for category in selected
                }
        for slug, inputs in self._REPORT_TO_RAW_MACRO.items():
            selected = [weights[name] for name in inputs if name in weights]
            if selected:
                weights[slug] = {
                    category: weight / len(selected)
                    for group in selected
                    for category, weight in group.items()
                }
        weights["overall_acc"] = {
            category: weight * coefficient / 100
            for name, coefficient in self._OVERALL_WEIGHTS
            for category, weight in weights.get(name, {}).items()
        }
        for k in sorted(set(cfg.pass_at_k) | {cfg.num_runs}):
            values = {
                category: [
                    (uid, self._unbiased_pass_at_k(cfg.num_runs, int(count), k))
                    for uid, count in rows
                ]
                for category, rows in counts.items()
            }
            self._add_category_scores(scores, values, weights, identity, f"_PASS@{k}")
        return scores

    def _send_metrics_to_jd(self, processed_metrics: dict[str, float]) -> None:
        if not JD.can_send_metrics(
            metrics=processed_metrics,
            hf_checkpoint_path=self.hf_checkpoint_path,
            metrics_dir=self.metrics_dir,
            model_info=self.model_info,
        ):
            return

        jd_metrics = {
            key.upper() if "/" not in key else key.replace("/", "_").upper(): value
            for key, value in processed_metrics.items()
        }

        jd_metrics = {
            (
                f"{key}_{self.autometric_runs_config.web_search.upper()}"
                if "WEB_SEARCH" in key
                else key
            ): value
            for key, value in jd_metrics.items()
        }

        JD.send_evaluator_metrics_with_scores(
            hf_checkpoint_path=self.hf_checkpoint_path,
            metrics_dir=self.metrics_dir,
            scores=self._cache_scores(jd_metrics),
            shooter_name="bfcl",
            shooter_version=JD.get_shooter_version("/home/user/bfcl_version.txt"),
            model_info=self.model_info,
            submetric_groups=self._submetric_groups(jd_metrics),
        )

    # --- multi-run (pass@k / mean±std) ---

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict]:
        entries: list[dict] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    entries.append(json.loads(stripped))
        return entries

    @staticmethod
    def _unbiased_pass_at_k(n: int, c: int, k: int) -> float:
        if n - c < k:
            return 1.0
        return 1.0 - math.comb(n - c, k) / math.comb(n, k)

    @classmethod
    def _collect_task_ids(cls, result_dir: Path) -> dict[str, set[str]]:
        out: dict[str, set[str]] = {}
        for f in result_dir.rglob("*_result.json"):
            m = cls._RESULT_CAT_RE.search(f.name)
            if not m:
                continue
            cat = m.group(1)
            if cat.endswith("_prereq"):
                continue
            ids: set[str] = set()
            for entry in cls._read_jsonl(f):
                if "id" in entry:
                    ids.add(str(entry["id"]))
            out[cat] = ids
        return out

    @classmethod
    def _collect_failed_ids(cls, score_dir: Path) -> dict[str, set[str]]:
        out: dict[str, set[str]] = {}
        for f in score_dir.rglob("*_score.json"):
            m = cls._SCORE_CAT_RE.search(f.name)
            if not m:
                continue
            cat = m.group(1)
            entries = cls._read_jsonl(f)
            failed: set[str] = set()
            for entry in entries[1:]:  # skip header
                if "id" in entry:
                    failed.add(str(entry["id"]))
            out[cat] = failed
        return out

    def _aggregate_multi_run(
        self, result_dirs: list[Path], score_dirs: list[Path]
    ) -> dict[str, float]:
        cfg = self.autometric_runs_config
        n = cfg.num_runs
        metrics: dict[str, float] = {}

        # Mean/std from CSV
        per_run = [self._load_metrics_from_csv(sd) for sd in score_dirs]
        all_keys = sorted({k for m in per_run for k in m})
        for key in all_keys:
            values = [m[key] for m in per_run if key in m]
            slug = key.removeprefix("BFCL_").lower()
            metrics[f"bfcl/{slug}/mean"] = statistics.mean(values)
            metrics[f"bfcl/{slug}/std"] = statistics.pstdev(values)

        # Pass@k from per-task data
        ks = sorted(set(cfg.pass_at_k) | {n})
        cat_task_ids: dict[str, set[str]] = {}
        cat_pass_count: dict[str, dict[str, int]] = {}

        for i in range(n):
            run_ids = self._collect_task_ids(result_dirs[i])
            run_failed = self._collect_failed_ids(score_dirs[i])
            for cat, ids in run_ids.items():
                cat_task_ids.setdefault(cat, set()).update(ids)
                cat_pass_count.setdefault(cat, {})
                failed = run_failed.get(cat, set())
                for task_id in ids:
                    cat_pass_count[cat].setdefault(task_id, 0)
                    if task_id not in failed:
                        cat_pass_count[cat][task_id] += 1

        ks = [k for k in ks if k <= n]

        # Per-category (leaf) pass@k, computed for every category so it can feed
        # both the emitted raw keys and the macro rollups below.
        leaf_pass: dict[str, dict[int, float]] = {}
        for cat, task_ids in cat_task_ids.items():
            if not task_ids:
                continue
            leaf_pass[cat] = {
                k: sum(
                    self._unbiased_pass_at_k(n, cat_pass_count[cat].get(tid, 0), k)
                    for tid in task_ids
                )
                / len(task_ids)
                for k in ks
            }

        for cat in sorted(self._KEEP_RAW_PASS_AT_K):
            if cat not in leaf_pass:
                continue
            for k in ks:
                metrics[f"bfcl/{cat}/pass@{k}"] = leaf_pass[cat][k]

        # Micro (task-weighted) slugs: pool the per-task pass counts of the raw
        # categories, so a single pass@k is computed over the union of tasks.
        for slug, raw_cat in self._RENAME.items():
            if raw_cat[0] not in leaf_pass:
                continue
            for k in ks:
                metrics[f"bfcl/{slug}/pass@{k}"] = leaf_pass[raw_cat[0]][k]

        for slug, raw_cats in self._REPORT_TO_RAW.items():
            pooled_counts: list[int] = []
            for raw in raw_cats:
                pooled_counts.extend(
                    cat_pass_count[raw].get(tid, 0)
                    for tid in cat_task_ids.get(raw, set())
                )
            if not pooled_counts:
                continue
            for k in ks:
                metrics[f"bfcl/{slug}/pass@{k}"] = sum(
                    self._unbiased_pass_at_k(n, c, k) for c in pooled_counts
                ) / len(pooled_counts)

        # Macro (category-unweighted) slugs: average the per-category pass@k of the inputs.
        macro_pass: dict[str, dict[int, float]] = {}
        for slug, inputs in self._REPORT_TO_RAW_MACRO.items():
            resolved = [macro_pass.get(name) or leaf_pass.get(name) for name in inputs]
            resolved = [r for r in resolved if r is not None]
            if not resolved:
                continue
            slug_pass = {k: sum(r[k] for r in resolved) / len(resolved) for k in ks}
            macro_pass[slug] = slug_pass
            for k in ks:
                metrics[f"bfcl/{slug}/pass@{k}"] = slug_pass[k]

        if not metrics:
            return {}

        for k in ks:
            total = 0.0
            for item in self._OVERALL_WEIGHTS:
                total += metrics.get(f"bfcl/{item[0]}/pass@{k}", 0.0) * (item[1] / 100)
            metrics[f"bfcl/overall_acc/pass@{k}"] = total

        return metrics

    def _run_multi(self) -> None:
        cfg = self.autometric_runs_config
        proc_env = self._build_bfcl_env(cfg.require_base_url())

        result_dirs: list[Path] = []
        score_dirs: list[Path] = []
        gen_cmds: dict[str, list[str]] = {}

        for i in range(cfg.num_runs):
            rd = self._result_dir / f"run_{i}"
            sd = self._score_dir / f"run_{i}"
            rd.mkdir(parents=True, exist_ok=True)
            sd.mkdir(parents=True, exist_ok=True)
            result_dirs.append(rd)
            score_dirs.append(sd)
            gen_cmds[f"gen_{i}"] = [
                cfg.venv_python,
                "-m",
                "bfcl_eval.__main__",
                "generate",
                *cfg.to_generate_bash_list(rd),
            ]

        Logger.info(
            f"BFCL multi-run generate: {cfg.num_runs} runs, "
            f"max_parallel={cfg.max_parallel_runs}"
        )
        SubprocessPoolRunner(
            cmds_to_run=gen_cmds,
            max_parallel=cfg.max_parallel_runs,
            autometrics_log_file=self.log_file,
            cwd=str(self.bfcl_root),
            env=proc_env,
        ).run_all()

        eval_cmds: dict[str, list[str]] = {}
        for i in range(cfg.num_runs):
            eval_cmds[f"eval_{i}"] = [
                cfg.venv_python,
                "-m",
                "bfcl_eval.__main__",
                "evaluate",
                *cfg.to_evaluate_bash_list(result_dirs[i], score_dirs[i]),
            ]

        Logger.info(f"BFCL multi-run evaluate: {cfg.num_runs} runs")
        SubprocessPoolRunner(
            cmds_to_run=eval_cmds,
            max_parallel=cfg.max_parallel_runs,
            autometrics_log_file=self.log_file,
            cwd=str(self.bfcl_root),
            env=proc_env,
        ).run_all()

        self.all_metrics = self._aggregate_multi_run(result_dirs, score_dirs)

        if not self.all_metrics:
            Logger.warning("No metrics aggregated from BFCL multi-run")
            return

        Logger.info(f"BFCL multi-run metrics ({len(self.all_metrics)} keys)")
        out_json = self.metrics_dir / "bfcl_metrics.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(self.all_metrics, f, indent=2)
