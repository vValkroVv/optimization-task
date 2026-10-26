import json
import os
from pathlib import Path
from typing import Optional

import requests

from ...config_model import (
    AccLenDatasetConfig,
    AccLenEvaluatorConfig,
    ModelInfoConfig,
)
from ...job_dispatcher import JD
from ...job_dispatcher.measurement import MeasurementUpload
from ...logger import Logger
from ...metrics.evaluators.interface import EVALUATED, MetricsEvaluator, Stage
from ...subprocess_manager import SubprocessManager

ACC_LEN_VERSION_FILE = "/home/user/sglang_version.txt"
FLUSH_TIMEOUT_S = 300
REQUEST_TIMEOUT_S = 60


class AccLenEvaluator(MetricsEvaluator):
    def setup(
        self,
        autometric_runs_config: AccLenEvaluatorConfig,
        artifacts_path: Path,
        hf_checkpoint_path: Path,
        model_info: ModelInfoConfig,
    ) -> None:
        Logger.info(
            f"Acc-len config: {autometric_runs_config.model_dump_json(indent=2)}"
        )
        self.autometric_runs_config = autometric_runs_config
        self.artifacts_path = artifacts_path
        self.hf_checkpoint_path = hf_checkpoint_path
        self.model_info = model_info
        self._create_metrics_dir(artifacts_path, "acc_len")
        self.results_json_path = self.metrics_dir / "results.json"
        self.worker_urls = autometric_runs_config.worker_urls
        Logger.info(f"AccLenEvaluator setup OK, workers: {self.worker_urls}")

    def stages(self) -> list[Stage]:
        return [Stage((EVALUATED,), self.run, pins_workers=True)]

    def _build_command(
        self,
        dataset: AccLenDatasetConfig,
        output_path: Path,
    ) -> list[str]:
        return [
            self.autometric_runs_config.venv_python,
            "-m",
            "sglang.bench_serving",
            *self.autometric_runs_config.build_cli_args(
                dataset=dataset, output_path=output_path
            ),
        ]

    def _build_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["SGLANG_BENCH_REQUEST_TIMEOUT"] = str(
            self.autometric_runs_config.request_timeout
        )
        return env

    def _calculated_results(self) -> dict[str, dict[str, float]]:
        if not self.results_json_path.exists():
            return {}
        return self._load_json_file(self.results_json_path)

    def _write_results(self, results: dict[str, dict[str, float]]) -> None:
        staged_path = self.metrics_dir / "results.json.tmp"
        with open(staged_path, "w") as results_file:
            json.dump(results, results_file, indent=4)
        os.replace(staged_path, self.results_json_path)

    def run(self) -> None:
        if not self.worker_urls:
            Logger.error(
                "Acc-len needs SGLang worker urls to read acceptance length from; "
                "it only supports the sglang inference engine, skipping run"
            )
            return

        results = self._calculated_results()
        for dataset in self.autometric_runs_config.datasets:
            if dataset.name in results:
                Logger.info(f"Acc-len already calculated for {dataset.name}, skipping")
                continue
            dataset_result = self._run_dataset(dataset)
            if dataset_result is None:
                continue
            results[dataset.name] = dataset_result
            self._write_results(results)

        if not results:
            Logger.error("Acc-len produced no results; no results.json will be saved")
            return
        Logger.info("Acc-len run completed")

    def _run_dataset(self, dataset: AccLenDatasetConfig) -> Optional[dict[str, float]]:
        output_path = self.metrics_dir / f"{dataset.name}.jsonl"
        if output_path.exists():
            output_path.unlink()

        self._flush_worker_counters()

        cmd = self._build_command(dataset, output_path)
        Logger.info(f"Starting acc-len run for {dataset.name}: {cmd}")
        subprocess_manager = SubprocessManager(
            cmd,
            self.log_file,
            None,
            f"acc_len_{dataset.name}",
            env=self._build_env(),
        )
        subprocess_manager.start()
        subprocess_manager.wait_for_completion()
        subprocess_manager.log_error()

        if not subprocess_manager.is_successful():
            Logger.error(f"Acc-len run failed for {dataset.name}")
            return None

        bench_result = self._read_bench_result(output_path)
        completed = int(bench_result.get("completed") or 0)
        requested = self.autometric_runs_config.num_prompts
        if completed == 0:
            Logger.error(
                f"Acc-len: bench_serving completed no requests for {dataset.name}"
            )
            return None
        if completed < requested:
            Logger.warning(
                f"Acc-len {dataset.name}: only {completed}/{requested} requests "
                "completed, the acceptance histogram covers those only"
            )

        internal_states = self._read_worker_internal_states()
        result = self._draft_position_metrics(internal_states, dataset.name)
        if not result:
            return None
        result.update(self._serving_metrics(bench_result))

        Logger.info(f"Acc-len {dataset.name}: {result}")
        return result

    def _flush_worker_counters(self) -> None:
        for worker_url in self.worker_urls:
            try:
                response = requests.post(
                    f"{worker_url}/flush_cache",
                    params={"timeout": FLUSH_TIMEOUT_S},
                    timeout=FLUSH_TIMEOUT_S + REQUEST_TIMEOUT_S,
                )
            except requests.RequestException as error:
                Logger.warning(f"Acc-len: /flush_cache failed on {worker_url}: {error}")
                continue
            if response.status_code != 200:
                Logger.warning(
                    f"Acc-len: /flush_cache refused by {worker_url} "
                    f"({response.status_code}), counters may carry over "
                    "from the previous dataset"
                )

    def _read_worker_internal_states(self) -> list[dict]:
        internal_states: list[dict] = []
        for worker_url in self.worker_urls:
            try:
                response = requests.get(
                    f"{worker_url}/server_info", timeout=REQUEST_TIMEOUT_S
                )
                response.raise_for_status()
                worker_states = response.json().get("internal_states") or []
            except requests.RequestException as error:
                Logger.warning(f"Acc-len: /server_info failed on {worker_url}: {error}")
                continue
            internal_states.extend(worker_states)
        return internal_states

    @staticmethod
    def _read_bench_result(output_path: Path) -> dict:
        if not output_path.exists():
            return {}
        lines = [line for line in output_path.read_text().splitlines() if line.strip()]
        if not lines:
            return {}
        return json.loads(lines[-1])

    @staticmethod
    def _serving_metrics(bench_result: dict) -> dict[str, float]:
        return {
            name: float(bench_result[bench_key])
            for name, bench_key in (
                ("output_throughput", "output_throughput"),
                ("peak_output_throughput", "max_output_tokens_per_s"),
                ("median_itl_ms", "median_itl_ms"),
            )
            if bench_result.get(bench_key) is not None
        }

    @staticmethod
    def _merge_correct_drafts_histograms(
        histograms: list[list[int]], min_length: int = 0
    ) -> list[int]:
        merged = [0] * min_length
        for histogram in histograms:
            if len(histogram) > len(merged):
                merged.extend([0] * (len(histogram) - len(merged)))
            for position, count in enumerate(histogram):
                merged[position] += count
        return merged

    # Both mirror the compute_* helpers in sglang's bench_serving.
    @staticmethod
    def _accept_length_by_num_steps(histogram: list[int]) -> list[float]:
        verify_ct = sum(histogram)
        if verify_ct == 0:
            return []
        accept_lengths = []
        accept_length = 1.0
        for position in range(1, len(histogram)):
            accept_length += sum(histogram[position:]) / verify_ct
            accept_lengths.append(accept_length)
        return accept_lengths

    @staticmethod
    def _accept_rate_by_position(histogram: list[int]) -> list[float]:
        survivors = sum(histogram)
        rates = []
        for position in range(1, len(histogram)):
            if survivors == 0:
                break
            accepted = sum(histogram[position:])
            rates.append(accepted / survivors)
            survivors = accepted
        return rates

    @classmethod
    def _draft_position_metrics(
        cls, internal_states: list[dict], dataset_name: str
    ) -> dict[str, float]:
        accept_lengths = [
            float(internal_state["avg_spec_accept_length"])
            for internal_state in internal_states
            if internal_state.get("avg_spec_accept_length") is not None
        ]
        if not accept_lengths:
            Logger.error(
                f"Acc-len: no acceptance length reported for {dataset_name}; did "
                "any worker answer, and is speculative decoding enabled in "
                "autometrics.inference?"
            )
            return {}
        Logger.info(
            f"Acc-len {dataset_name} accept length per dp rank: {accept_lengths}"
        )
        metrics = {"accept_length": sum(accept_lengths) / len(accept_lengths)}

        histograms = [
            internal_state["spec_correct_drafts_histogram"]
            for internal_state in internal_states
            if internal_state.get("spec_correct_drafts_histogram")
        ]
        if not histograms:
            Logger.warning(
                f"Acc-len {dataset_name}: no acceptance histogram reported, "
                "leaving out the per-draft-position metrics; does this sglang "
                "carry sglang-pretrain!64?"
            )
            return metrics

        topk = internal_states[0].get("speculative_eagle_topk")
        if (topk or 1) != 1:
            Logger.warning(
                f"Acc-len {dataset_name}: speculative_eagle_topk={topk} drafts a "
                "tree rather than a prefix chain, so accept length per draft depth "
                "would not describe a shallower server, leaving it out"
            )
            return metrics

        merged = cls._merge_correct_drafts_histograms(
            histograms,
            min_length=(internal_states[0].get("speculative_num_steps") or 0) + 1,
        )
        Logger.info(
            f"Acc-len {dataset_name}: correct drafts histogram {merged}, "
            f"per dp rank: {histograms}"
        )

        for depth, length in enumerate(
            cls._accept_length_by_num_steps(merged), start=1
        ):
            metrics[f"accept_length_num_steps_{depth}"] = length
        for position, rate in enumerate(cls._accept_rate_by_position(merged), start=1):
            metrics[f"accept_rate_position_{position}"] = rate
        return metrics

    def get_metrics(self) -> dict[str, float]:
        if self.metrics_dir is None:
            Logger.warning("metrics_dir is not set for AccLenEvaluator")
            return {}
        if not self.results_json_path.exists():
            Logger.warning(f"No results.json found at {self.results_json_path}")
            return {}

        data = self._load_json_file(self.results_json_path)
        configured = {dataset.name for dataset in self.autometric_runs_config.datasets}
        metrics = {
            f"acc_len/{dataset_name}/{metric_name}": value
            for dataset_name, dataset_metrics in data.items()
            if dataset_name in configured
            for metric_name, value in dataset_metrics.items()
        }
        Logger.info(f"Collected {len(metrics)} acc-len metrics")
        return metrics

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
        JD.send_evaluator_metrics(
            hf_checkpoint_path=self.hf_checkpoint_path,
            metrics_dir=self.metrics_dir,
            jd_metrics=jd_metrics,
            shooter_name="acc_len",
            shooter_version=JD.get_shooter_version(ACC_LEN_VERSION_FILE),
            model_info=self.model_info,
        )
