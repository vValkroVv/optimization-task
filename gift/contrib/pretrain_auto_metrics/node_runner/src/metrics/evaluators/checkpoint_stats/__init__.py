import math
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable, Optional

from ....config_model import CheckpointStatsEvaluatorConfig, ModelInfoConfig
from ....job_dispatcher import JD
from ....job_dispatcher.measurement import MeasurementUpload
from ....logger import Logger
from ....subprocess_manager import SubprocessManager
from ..interface import MetricsEvaluator


class CheckpointStatsEvaluator(MetricsEvaluator):
    SHOOTER_NAME = "checkpoint_stats"
    SHOOTER_VERSION = "1.0.0"
    REPORT_FILENAME = "checkpoint_stats.json"

    def setup(
        self,
        autometric_runs_config: CheckpointStatsEvaluatorConfig,
        artifacts_path: Path,
        hf_checkpoint_path: Path,
        model_info: ModelInfoConfig,
    ) -> None:
        self.autometric_runs_config = autometric_runs_config
        self.artifacts_path = Path(artifacts_path)
        self.hf_checkpoint_path = Path(hf_checkpoint_path)
        self.model_info = model_info
        self._create_metrics_dir(self.artifacts_path, self.SHOOTER_NAME)
        self.report_path = self.metrics_dir / self.REPORT_FILENAME
        cfg = self.autometric_runs_config
        self.command = [
            sys.executable,
            "-u",
            str(Path(__file__).with_name("run.py").resolve()),
            "--hf-dir",
            str(self.hf_checkpoint_path),
            "--output",
            str(self.report_path),
            "--device",
            cfg.device,
            "--num-threads",
            str(cfg.num_threads),
            "--role-overrides",
            json.dumps(cfg.role_overrides),
            "--topk-sigma",
            str(cfg.topk_sigma),
            "--svd-niter",
            str(cfg.svd_niter),
            "--spectral-numel-cap",
            str(cfg.spectral_numel_cap),
            "--hist-bins",
            str(cfg.hist_bins),
            f"--hist-log-min={cfg.hist_log_min}",
            f"--hist-log-max={cfg.hist_log_max}",
        ]
        if cfg.include is not None:
            self.command.append(f"--include={cfg.include}")
        if cfg.exclude is not None:
            self.command.append(f"--exclude={cfg.exclude}")
        Logger.info(f"CheckpointStatsEvaluator setup OK: report -> {self.report_path}")

    def run(self) -> None:
        if self.report_path.exists():
            Logger.info(
                f"Checkpoint stats already calculated at {self.report_path}, "
                "skipping run"
            )
            return
        subprocess_manager = SubprocessManager(
            self.command, self.log_file, cwd=None, run_name=self.SHOOTER_NAME
        )
        subprocess_manager.start()
        subprocess_manager.wait_for_completion()
        subprocess_manager.log_error()
        if not subprocess_manager.is_successful():
            Logger.error(
                f"checkpoint_stats failed (exit={subprocess_manager.exit_code}), see {self.log_file}"
            )
            return

    def get_metrics(self) -> dict[str, float]:
        if not self.report_path.exists():
            Logger.error(
                f"Checkpoint stats report {self.report_path} is missing, "
                "no metrics to publish"
            )
            return {}
        metrics = self.headline_metrics(self._load_json_file(self.report_path))
        Logger.info(f"Checkpoint stats: {len(metrics)} headline metrics")
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

    @classmethod
    def headline_metrics(cls, report: dict) -> dict[str, float]:
        prefix = cls.SHOOTER_NAME
        records = [rec for rec in report["tensors"].values() if "base" in rec]
        metrics: dict[str, float] = {
            f"{prefix}/n_tensors": float(report["meta"]["n_tensors"]),
            f"{prefix}/n_errors": float(
                sum(
                    "error" in rec or "error" in rec.get("spectral", {})
                    for rec in report["tensors"].values()
                )
            ),
            f"{prefix}/n_unknown": float(
                sum(rec["role"] == "unknown" for rec in report["tensors"].values())
            ),
            f"{prefix}/n_tensors_with_nan": float(
                sum(1 for rec in records if rec["base"]["n_nan"])
            ),
            f"{prefix}/n_tensors_with_inf": float(
                sum(1 for rec in records if rec["base"]["n_inf"])
            ),
        }
        max_abs = cls._values(records, lambda rec: rec["base"]["max_abs"])
        if max_abs:
            metrics[f"{prefix}/max_abs"] = max(max_abs)
        fro = cls._values(records, lambda rec: rec["base"]["fro"])
        if fro:
            metrics[f"{prefix}/fro_norm_total"] = math.sqrt(sum(v**2 for v in fro))

        by_role: dict[str, list[dict]] = defaultdict(list)
        for rec in records:
            role = rec["role"]
            if rec.get("block") == "mtp":
                role = f"mtp_{role}"
            by_role[role].append(rec)
        aggregates: list[tuple[str, Callable[[dict], float | None], Callable]] = [
            ("mean_abs_mean", lambda rec: rec["base"]["mean_abs"], cls._mean),
            ("max_abs_max", lambda rec: rec["base"]["max_abs"], max),
            ("fro_mean", lambda rec: rec["base"]["fro"], cls._mean),
            ("max_over_median_max", cls._max_over_median, max),
            ("stable_rank_min", cls._stable_rank, min),
        ]
        for role, role_records in sorted(by_role.items()):
            for name, pick, reduce in aggregates:
                values = cls._values(role_records, pick)
                if values:
                    metrics[f"{prefix}/{role}/{name}"] = reduce(values)
        return metrics

    @staticmethod
    def _values(
        records: list[dict], pick: Callable[[dict], float | None]
    ) -> list[float]:
        return [value for value in map(pick, records) if value is not None]

    @staticmethod
    def _mean(values: list[float]) -> float:
        return sum(values) / len(values)

    @staticmethod
    def _max_over_median(rec: dict) -> float | None:
        rowcol = rec.get("rowcol")
        if rowcol is None:
            return None
        finite = [
            value
            for value in (
                rowcol["row"]["max_over_median"],
                rowcol["col"]["max_over_median"],
            )
            if value is not None
        ]
        return max(finite) if finite else None

    @staticmethod
    def _stable_rank(rec: dict) -> float | None:
        spectral = rec.get("spectral")
        if spectral is None or "error" in spectral:
            return None
        return spectral["stable_rank"]

    def _send_metrics_to_jd(self, processed_metrics: dict[str, float]) -> None:
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
            shooter_name=self.SHOOTER_NAME,
            shooter_version=self.SHOOTER_VERSION,
            model_info=self.model_info,
        )
