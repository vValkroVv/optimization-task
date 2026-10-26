import base64
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

from ...job_dispatcher import JD, SubmetricGroup
from ...job_dispatcher.cache_scores import LmEvalCacheScoresBuilder
from ...job_dispatcher.submetrics import mean
from ...config_model import (
    EvalTaskConfig,
    LmEvalEvaluatorConfig,
    ModelInfoConfig,
)
from ...job_dispatcher.measurement import MeasurementUpload
from ...logger import Logger
from ...metrics.evaluators.interface import MetricsEvaluator
from ...subprocess_pool_runner import SubprocessPoolRunner

ROBOT_SWE_TOKEN_BASE64 = "eXRjdC0yYzQyLWMwNTlkMGUxYjZkODQ0ZTU3ZWVhOWVmOTQzODY5OGU4"


def decode_base64(value: str) -> str:
    return base64.b64decode(value).decode()


class LmEvalMetricsFilter:
    EXCLUDED_METRICS = {
        "ru_sci_passage_count/0-shot/4k_count.scoring": 0.72,
        "ru_sci_passage_count/0-shot/8k_count.scoring": 0.28,
        "ru_sci_passage_count/0-shot/16k_count.scoring": 0.12,
        "ru_sci_passage_count/0-shot/32k_count.scoring": 0.03,
        "ru_sci_passage_count/0-shot/64k_count.scoring": 0.01,
        "ru_sci_passage_count/0-shot/128k_count.scoring": 0.03,
        "ru_babilong_qa5/0-shot/4k_em.scoring": 0.92,
        "ru_babilong_qa5/0-shot/8k_em.scoring": 0.9,
        "ru_babilong_qa5/0-shot/16k_em.scoring": 0.84,
        "ru_babilong_qa5/0-shot/32k_em.scoring": 0.78,
        "ru_babilong_qa5/0-shot/64k_em.scoring": 0.53,
        "ru_babilong_qa5/0-shot/128k_em.scoring": 0.58,
        "ru_babilong_qa4/0-shot/4k_em.scoring": 0.49,
        "ru_babilong_qa4/0-shot/8k_em.scoring": 0.6,
        "ru_babilong_qa4/0-shot/16k_em.scoring": 0.6,
        "ru_babilong_qa4/0-shot/32k_em.scoring": 0.59,
        "ru_babilong_qa4/0-shot/64k_em.scoring": 0.48,
        "ru_babilong_qa4/0-shot/128k_em.scoring": 0.51,
        "ru_babilong_qa3/0-shot/4k_em.scoring": 0.23,
        "ru_babilong_qa3/0-shot/8k_em.scoring": 0.25,
        "ru_babilong_qa3/0-shot/16k_em.scoring": 0.16,
        "ru_babilong_qa3/0-shot/32k_em.scoring": 0.15,
        "ru_babilong_qa3/0-shot/64k_em.scoring": 0.14,
        "ru_babilong_qa3/0-shot/128k_em.scoring": 0.13,
        "ru_babilong_qa2/0-shot/4k_em.scoring": 0.31,
        "ru_babilong_qa2/0-shot/8k_em.scoring": 0.24,
        "ru_babilong_qa2/0-shot/16k_em.scoring": 0.17,
        "ru_babilong_qa2/0-shot/32k_em.scoring": 0.15,
        "ru_babilong_qa2/0-shot/64k_em.scoring": 0.1,
        "ru_babilong_qa2/0-shot/128k_em.scoring": 0.13,
        "ru_babilong_qa1/0-shot/4k_em.scoring": 0.86,
        "ru_babilong_qa1/0-shot/8k_em.scoring": 0.84,
        "ru_babilong_qa1/0-shot/16k_em.scoring": 0.79,
        "ru_babilong_qa1/0-shot/32k_em.scoring": 0.71,
        "ru_babilong_qa1/0-shot/64k_em.scoring": 0.58,
        "ru_babilong_qa1/0-shot/128k_em.scoring": 0.55,
        "mmmlu_zh_cn_stem/5-shot/acc": 0.0,
        "mmmlu_zh_cn_other/5-shot/acc": 0.0,
        "mmmlu_zh_cn_social_sciences/5-shot/acc": 0.0,
        "mmmlu_zh_cn_humanities/5-shot/acc": 0.0,
        "gsm8k/8-shot/exact_match.strict-match": 0.0,
        "mgsm_en_cot_en/8-shot/exact_match.strict-match": 0.0,
        "mgsm_en_cot_zh/8-shot/exact_match.strict-match": 0.0,
        "mgsm_en_cot_ru/8-shot/exact_match.strict-match": 0.0,
        "ruscontext/5-shot/f1": 0.0,
    }

    def should_exclude_metric(self, metric_name: str) -> bool:
        normalized_key = metric_name.lower()
        if normalized_key in self.EXCLUDED_METRICS:
            return True

        parts = normalized_key.split("/")
        if len(parts) >= 3:
            submetric_name = parts[-1]
            if submetric_name.endswith("_max") or submetric_name.endswith("_diff"):
                return True

        return False

    def filter_metrics(self, metrics: dict[str, float]) -> dict[str, float]:
        filtered = {}
        for metric_name, value in metrics.items():
            if not self.should_exclude_metric(metric_name):
                filtered[metric_name] = value

        return filtered


class LmEvalMetricsAggregationMixin(MetricsEvaluator):
    GROUPS_WITH_SUBTASK_METRICS = ("mmlu_pro", "ruler")

    def collect_metrics_from_dir(
        self, metrics_dir: Path, files: Optional[set[Path]] = None
    ) -> dict[str, float]:
        runs = []
        for file in os.listdir(metrics_dir):
            if not file.endswith(".json"):
                continue
            path = Path(metrics_dir, file)
            if files is not None and path not in files:
                continue
            metrics_run: dict[str, Any] = self._load_json_file(path)
            run_date = metrics_run.get("date")
            if not isinstance(run_date, (int, float)):
                run_date = path.stat().st_mtime
            runs.append((run_date, file, metrics_run))

        # Several files can carry the same metric names: the harness appends a new
        # timestamped file instead of overwriting, and a task renamed by its shots
        # count leaves its pre-rename file behind. Order by the run timestamp the
        # harness records inside the file, so the latest measurement wins whatever
        # the file names are and whatever copying did to the mtimes.
        converted_metrics = {}
        for _, _, metrics_run in sorted(runs, key=lambda run: run[:2]):
            converted_metrics.update(self.aggregate_single_run(metrics_run))

        return converted_metrics

    def aggregate_metrics(self, metrics: dict[str, float]) -> dict[str, float]:
        metrics = self._aggregate_ruler_metrics(metrics)
        return metrics

    @staticmethod
    def metric_key(
        group_name: str,
        task_name: str,
        num_fewshots: Optional[int],
        submetric_name: str,
    ) -> str:
        submetric_name = submetric_name.replace(",none", "").replace(",", ".")
        if task_name == group_name:
            return f"{group_name}/{num_fewshots}-shot/{submetric_name}"
        return (
            f"{group_name}/{task_name[len(group_name) + 1 :]}"
            f"/{num_fewshots}-shot/{submetric_name}"
        )

    @staticmethod
    def ruler_bucket_key(metric_name: str) -> Optional[str]:
        parts = metric_name.split("/")
        if not parts[-1].isdigit():
            return None
        return f"ruler/{parts[-2]}/{parts[-1]}"

    def _aggregate_ruler_metrics(self, metrics: dict[str, float]) -> dict[str, float]:
        final_result = {}
        buckets: dict[str, list[float]] = defaultdict(list)
        for metric_name, metric_value in metrics.items():
            if metric_name.split("/")[0] != "ruler":
                final_result[metric_name] = metric_value
                continue
            bucket = self.ruler_bucket_key(metric_name)
            if bucket is None:
                continue
            buckets[bucket].append(float(metric_value))

        for bucket, values in buckets.items():
            final_result[bucket] = sum(values) / len(values)
        return final_result

    def aggregate_single_run(self, metrics_run: dict[str, Any]) -> dict[str, float]:
        num_fewshots = (
            next(iter(metrics_run["n-shot"].values()))
            if metrics_run.get("n-shot")
            else None
        )
        results = metrics_run.get("results")
        if not results:
            return {}

        converted_metrics = {}
        for group_name, subtasks in metrics_run["group_subtasks"].items():
            task_names = []
            if group_name in self.GROUPS_WITH_SUBTASK_METRICS:
                task_names += list(subtasks)
            task_names.append(group_name)
            for task_name in task_names:
                converted_metrics.update(
                    self._get_metric_results_and_add_correct_name(
                        results, task_name, group_name, num_fewshots
                    )
                )

        return converted_metrics

    def _get_metric_results_and_add_correct_name(
        self,
        metric_resuts: dict[str, dict[str, dict[str, float]]],
        metric_name: str,
        group_name: str,
        num_fewshots: int,
    ) -> dict[str, float]:
        return_metrics = {}
        metric_resuts[metric_name].pop("alias", None)
        for submetric_name, submetric_value in metric_resuts[metric_name].items():
            if "_stderr" in submetric_name:
                continue
            key = self.metric_key(group_name, metric_name, num_fewshots, submetric_name)
            return_metrics[key] = submetric_value

        return return_metrics


class LmEvalEvaluator(LmEvalMetricsAggregationMixin, LmEvalMetricsFilter):
    SHOOTER_VERSION_FILE = "/home/user/lm_eval_version.txt"
    METRICS_DIR_NAME = "lm_eval"

    @staticmethod
    def _submetric_groups(metrics: dict[str, float]) -> list[SubmetricGroup]:
        buckets: dict[str, list[str]] = defaultdict(list)
        for key in metrics:
            task, _, suffix = key.partition("/")
            shots = suffix.partition("/")[0]
            if task in ("ifbench", "ifbench_32"):
                buckets[f"{task}/{shots}"].append(key)
            elif task in (
                "jsonschema_bench_easy",
                "jsonschema_bench_medium",
                "jsonschema_bench_hard",
                "jsonschema_bench_easy_32",
                "jsonschema_bench_medium_32",
                "jsonschema_bench_hard_32",
            ):
                benchmark = (
                    "jsonschema_bench_32"
                    if task.endswith("_32")
                    else "jsonschema_bench"
                )
                buckets[f"{benchmark}/{shots}"].append(key)

        groups = []
        for prefix, members in buckets.items():
            task = prefix.split("/")[0]
            if task == "ifbench":
                headline = f"{prefix}/prompt_level_loose_acc"
                groups.append(
                    SubmetricGroup(
                        key=headline,
                        members=[key for key in members if key != headline],
                        value=headline,
                    )
                )
            elif task == "ifbench_32":
                headline = f"{prefix}/pass@8_prompt_level_loose_acc.all_responses"
                groups.append(
                    SubmetricGroup(
                        key=f"{prefix}/mean_8",
                        members=[key for key in members if key != headline],
                        value=headline,
                    )
                )
            else:
                groups.append(
                    SubmetricGroup(key=f"{prefix}/mean", members=members, value=mean)
                )
        return groups

    def setup(
        self,
        autometric_runs_config: LmEvalEvaluatorConfig,
        artifacts_path: Path,
        hf_checkpoint_path: Path,
        model_info: ModelInfoConfig,
    ) -> None:
        self.autometric_runs_config = autometric_runs_config
        self.artifacts_path = Path(artifacts_path)
        self.hf_checkpoint_path = Path(hf_checkpoint_path)
        self.model_info = model_info
        self.cache_s3_paths: dict[str, str] = {}
        self._create_metrics_dir(artifacts_path, self.METRICS_DIR_NAME)
        self._create_measurement_files(hf_checkpoint_path, self.METRICS_DIR_NAME)
        self.cmds_to_run = self._convert_raw_config_to_bash(
            autometric_runs_config, self.metrics_dir
        )
        self._assert_setup()

    def _build_lm_eval_env(self) -> dict[str, str]:
        env = {
            **os.environ,
            "HF_ALLOW_CODE_EVAL": "1",
            "HF_DATASETS_TRUST_REMOTE_CODE": "1",
            "YT_TOKEN": decode_base64(ROBOT_SWE_TOKEN_BASE64),
        }
        # multipl_e_humaneval validates generated code through the YT TaskProxy
        # executor. MULTIPL_E_CACHE_ONLY runs rely on the URL being unset.
        if not env.get("MULTIPL_E_CACHE_ONLY"):
            env.setdefault("YT_TASKPROXY_BASE_URL", self.get_yt_url())
        return env

    def run(self) -> None:
        Logger.info("Starting LM eval runs")

        pool = SubprocessPoolRunner(
            cmds_to_run=self.cmds_to_run,
            max_parallel=self.autometric_runs_config.pool_max_parallel,
            autometrics_log_file=self.log_file,
            cwd=None,
            env=self._build_lm_eval_env(),
        )

        pool.run_all()

    def get_metrics(self) -> dict[str, float]:
        if self.metrics_dir is None:
            Logger.warning("metrics_dir is not set for LmEvalEvaluator")
            return {}

        return self._aggregated_metrics(None)

    def measurement_metrics(self, metrics: dict[str, float]) -> dict[str, float]:
        measured = self._aggregated_metrics(self.measurement_files.paths())
        return {name: metrics[name] for name in measured if name in metrics}

    def _aggregated_metrics(self, files: Optional[set[Path]]) -> dict[str, float]:
        converted_metrics = self.collect_metrics_from_dir(self.metrics_dir, files)
        filtered_metrics = self.filter_metrics(converted_metrics)
        return self.aggregate_metrics(filtered_metrics)

    def after_run(
        self, metrics: dict[str, float], uploaded: Optional[MeasurementUpload] = None
    ) -> None:
        self.cache_s3_paths = JD.upload_metrics_and_cache_scores(
            make_builder=self.make_cache_scores_builder,
            metrics_dir=self.metrics_dir,
            hf_checkpoint_path=self.hf_checkpoint_path,
            shooter_name=self.METRICS_DIR_NAME,
            version_file=self.SHOOTER_VERSION_FILE,
            uploaded=uploaded,
        )
        self._send_metrics_to_jd(metrics)
        tar_path = JD.tar_metrics_dir(
            self.metrics_dir, self.artifacts_path, type(self).__name__
        )
        JD.symlink_metrics_for_transfer(
            tar_path, self.hf_checkpoint_path, type(self).__name__
        )

    def make_cache_scores_builder(
        self, run_files: Optional[set[Path]] = None
    ) -> LmEvalCacheScoresBuilder:
        return LmEvalCacheScoresBuilder(
            Path(self.metrics_dir), evaluator=self, run_files=run_files
        )

    def _assert_setup(self) -> None:
        assert self.autometric_runs_config is not None
        assert self.cmds_to_run is not None
        assert self.metrics_dir is not None
        Logger.info(
            f"LmEvalEvaluator setup successfully and ready to make {self.cmds_to_run} runs"
        )

    def _lm_eval_runs_to_bash_list(
        self,
        lm_eval_runs_config: LmEvalEvaluatorConfig,
        metrics_dir: Path,
    ) -> dict[str, list[str]]:
        result_commands_map: dict[str, list[str]] = {}
        for task_name, task_config in (
            *lm_eval_runs_config.pretrain.items(),
            *lm_eval_runs_config.instruct.items(),
        ):
            for run_name, run_config in self._fan_out_by_fewshot(
                task_name, task_config
            ):
                if run_name in result_commands_map:
                    raise ValueError(
                        f"Duplicate lm_eval run name '{run_name}': two configured runs "
                        "would share a command key and an output path"
                    )
                result_commands_map[run_name] = [
                    "/home/user/lm_eval/bin/python",
                    "-m",
                    "lm_eval",
                    "--tasks",
                    task_name,
                    "--output_path",
                    str(metrics_dir / f"{run_name}.json"),
                    *run_config.to_bash_list(),
                ]
        return result_commands_map

    @staticmethod
    def _fan_out_by_fewshot(
        task_name: str, task_config: EvalTaskConfig
    ) -> list[tuple[str, EvalTaskConfig]]:
        """One run per requested shots count, named so that runs never collide."""
        if task_config.num_fewshot is None:
            return [(task_name, task_config)]

        num_fewshots = task_config.num_fewshot
        if not isinstance(num_fewshots, list):
            num_fewshots = [num_fewshots]
        return [
            (
                f"{task_name}_{num_fewshot}shot",
                task_config.model_copy(update={"num_fewshot": num_fewshot}),
            )
            for num_fewshot in num_fewshots
        ]

    @staticmethod
    def _is_already_calculated(run_name: str, metrics_dir: Path) -> bool:
        metrics_files = os.listdir(metrics_dir)
        for file in metrics_files:
            path = Path(file)
            if path.suffix != ".json":
                continue

            stem = path.stem
            if stem == run_name:
                return True

            timestamp_prefix = f"{run_name}_"
            if not stem.startswith(timestamp_prefix):
                continue

            timestamp = stem[len(timestamp_prefix) :]
            if re.match(r"\d{4}-\d{2}-\d{2}T", timestamp):
                return True
        return False

    def _convert_raw_config_to_bash(
        self, lm_eval_runs_config: LmEvalEvaluatorConfig, metrics_dir: Path
    ) -> dict[str, list[str]]:
        before_calculation_skip = self._lm_eval_runs_to_bash_list(
            lm_eval_runs_config, metrics_dir
        )
        if lm_eval_runs_config.restart:
            Logger.info(
                f"restart is set, recalculating all {len(before_calculation_skip)} "
                "lm_eval runs"
            )
            return before_calculation_skip

        cmds_to_run = {}
        for run_name, run_bash_list in before_calculation_skip.items():
            if self._is_already_calculated(run_name, metrics_dir):
                Logger.info(f"{run_name} already calculated, skipping run")
                continue
            cmds_to_run[run_name] = run_bash_list
        return cmds_to_run

    def _send_metrics_to_jd(self, processed_metrics: dict[str, float]):
        if not JD.can_send_metrics(
            metrics=processed_metrics,
            hf_checkpoint_path=self.hf_checkpoint_path,
            metrics_dir=self.metrics_dir,
            model_info=self.model_info,
        ):
            return

        JD.send_evaluator_metrics(
            hf_checkpoint_path=self.hf_checkpoint_path,
            metrics_dir=self.metrics_dir,
            jd_metrics=processed_metrics,
            shooter_name=self.METRICS_DIR_NAME,
            shooter_version=JD.get_shooter_version(self.SHOOTER_VERSION_FILE),
            model_info=self.model_info,
            extra_body=self.cache_s3_paths,
            submetric_groups=self._submetric_groups(processed_metrics),
        )
