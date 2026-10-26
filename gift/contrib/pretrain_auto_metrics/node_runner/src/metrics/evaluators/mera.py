import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Optional

from ...job_dispatcher import JD
from ...config_model import (
    MeraEvaluatorConfig,
    ModelInfoConfig,
)
from ...job_dispatcher.measurement import MeasurementUpload
from ...logger import Logger
from ...metrics.evaluators.interface import MetricsEvaluator
from ...subprocess_pool_runner import SubprocessPoolRunner


class MeraEvaluator(MetricsEvaluator):
    MERA_VENV_PYTHON = "/home/user/mera_eval/bin/python"
    MERA_VERSION_FILE = "/home/user/mera_version.txt"
    MERA_SYSTEM_INSTRUCTION_FILE = "/home/user/mera_system_instruction.txt"
    MERA_TASKS_PATH = "/home/user/mera_tasks"

    MERA_SCRIPTS_DIR = "/home/user/mera_scripts"

    _METRICS_ALLOWLIST: set[str] = {
        # TEXT
        "mera/bps_gen/1-shot/exact_match.scoring",
        "mera/rudetox/1-shot/j.scoring",
        "mera/ruhhh_gen/0-shot/em.scoring",
        "mera/ruhumaneval/0-shot/pass@1.scoring",
        "mera/ruhumaneval/0-shot/pass@5.scoring",
        "mera/ruhumaneval/0-shot/pass@10.scoring",
        "mera/rummlu_gen/1-shot/em.scoring",
        "mera/simplear/2-shot/exact_match.scoring",
        # ETHICS
        "mera/ruethics_gen/0-shot/mcc_correct_justice.scoring",
        "mera/ruethics_gen/0-shot/mcc_correct_law.scoring",
        "mera/ruethics_gen/0-shot/mcc_correct_moral.scoring",
        "mera/ruethics_gen/0-shot/mcc_correct_utilitarianism.scoring",
        "mera/ruethics_gen/0-shot/mcc_correct_virtue.scoring",
        "mera/ruethics_gen/0-shot/mcc_ethical_justice.scoring",
        "mera/ruethics_gen/0-shot/mcc_ethical_law.scoring",
        "mera/ruethics_gen/0-shot/mcc_ethical_moral.scoring",
        "mera/ruethics_gen/0-shot/mcc_ethical_utilitarianism.scoring",
        "mera/ruethics_gen/0-shot/mcc_ethical_virtue.scoring",
        "mera/ruethics_gen/0-shot/mcc_good_justice.scoring",
        "mera/ruethics_gen/0-shot/mcc_good_law.scoring",
        "mera/ruethics_gen/0-shot/mcc_good_moral.scoring",
        "mera/ruethics_gen/0-shot/mcc_good_utilitarianism.scoring",
        "mera/ruethics_gen/0-shot/mcc_good_virtue.scoring",
        # HATE SPEECH
        "mera/ruhatespeech_gen/1-shot/em.scoring",
        "mera/ruhatespeech_gen/1-shot/em_lgbt.scoring",
        "mera/ruhatespeech_gen/1-shot/em_men.scoring",
        "mera/ruhatespeech_gen/1-shot/em_migrants.scoring",
        "mera/ruhatespeech_gen/1-shot/em_nationalities.scoring",
        "mera/ruhatespeech_gen/1-shot/em_other.scoring",
        "mera/ruhatespeech_gen/1-shot/em_women.scoring",
        # CODE
        "mera/codecorrectness/0-shot/acc",
        "mera/strucom/0-shot/chrf",
        "mera/ruhumaneval_code/0-shot/pass@1.scoring",
        "mera/ruhumaneval_code/0-shot/pass@5.scoring",
        "mera/ruhumaneval_code/0-shot/pass@10.scoring",
        "mera/unittests/0-shot/code_bleu",
    }

    # Metrics that lm_eval reports on a 0-100 scale but JD expects 0-1
    _METRICS_SCALE_DOWN: set[str] = {
        "mera/strucom/0-shot/chrf",
    }

    # Task name patterns per category (derived from MERA submission scripts).
    # Used to filter sample/result files so each category's logs_public.zip
    # contains only its own logs instead of all tasks' logs.
    _CATEGORY_TASKS: dict[str, set[str]] = {
        "text": {
            "bps",
            "lcs",
            "chegeka",
            "mathlogicqa",
            "multiq",
            "parus",
            "rcb",
            "rudetox",
            "ruethics",
            "ruhatespeech",
            "ruhhh",
            "rummlu",
            "mamuramu",
            "rumodar",
            "rumultiar",
            "simplear",
            "ruopenbookqa",
            "ruworldtree",
            "rwsd",
            "use",
            "rutie",
            "ruhumaneval",
            "rucodeeval",
        },
        "industrial": {"agro_bench", "aqua_bench", "med_bench"},
        "code": {
            "ruhumaneval_code",
            "rucodeeval_code",
            "codelintereval",
            "rucodereviewer",
            "javatestgen",
            "realcode",
            "realcodejava",
            "yabloco",
            "strucom",
            "unittests",
            "codecorrectness",
        },
    }
    _DATE_RE = re.compile(r"_\d{4}-\d{2}-\d{2}T")

    @classmethod
    def _normalize_metrics(cls, metrics: dict[str, float]) -> None:
        """Rescale metrics that use non-standard ranges."""
        for key in cls._METRICS_SCALE_DOWN:
            if key in metrics:
                metrics[key] = metrics[key] / 100.0

    @classmethod
    def _extract_task_name(cls, filename: str) -> Optional[str]:
        """Extract lm_eval task name from a sample filename.
        E.g. 'samples_bps_gen_2026-02-17T10-50-09.123456.jsonl' -> 'bps_gen'
        """
        rest = filename[len("samples_") :]
        m = cls._DATE_RE.search(rest)
        if m:
            return rest[: m.start()]
        return None

    @staticmethod
    def _read_file_stripped(path: str) -> Optional[str]:
        try:
            return Path(path).read_text().strip()
        except Exception:
            return None

    @staticmethod
    def _task_matches_category(task_name: str, patterns: set[str]) -> bool:
        """Check if a task name belongs to the category.
        Handles _gen suffix: 'bps_gen' matches pattern 'bps'."""
        if task_name in patterns:
            return True
        base = task_name.removesuffix("_gen")
        if base != task_name and base in patterns:
            return True
        return False

    @staticmethod
    def _is_already_calculated(task_name: str, metrics_dir: Path) -> bool:
        metrics_files = os.listdir(metrics_dir)
        for file in metrics_files:
            if task_name == file.rsplit("_", 1)[0]:
                return True
        return False

    def setup(
        self,
        autometric_runs_config: MeraEvaluatorConfig,
        artifacts_path: Path,
        hf_checkpoint_path: Path,
        model_info: ModelInfoConfig,
    ) -> None:
        self.autometric_runs_config = autometric_runs_config
        self.artifacts_path = Path(artifacts_path)
        self.hf_checkpoint_path = Path(hf_checkpoint_path)
        self.model_info = model_info
        self._create_metrics_dir(artifacts_path, "mera")
        self.cmds_to_run = self._convert_raw_config_to_bash(
            autometric_runs_config, self.metrics_dir
        )
        Logger.info(
            f"MeraEvaluator setup complete, {len(self.cmds_to_run)} runs queued"
        )

    @staticmethod
    def _build_mera_env() -> dict[str, str]:
        """Build env with contrib/lm_evaluation_harness removed from PYTHONPATH
        so that the mera_eval venv's own lm_eval is used instead."""
        env = os.environ.copy()
        pythonpath = env.get("PYTHONPATH", "")
        filtered = [
            p
            for p in pythonpath.split(":")
            if p and "contrib/lm_evaluation_harness" not in p
        ]
        env["PYTHONPATH"] = ":".join(filtered)
        env["HF_ALLOW_CODE_EVAL"] = "1"
        env["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"
        env["OPENAI_API_KEY"] = "EMPTY"
        env["LINTER_PATH"] = "/home/user/mera_eval/bin/flake8"
        return env

    def run(self) -> None:
        Logger.info("Starting MERA eval runs")

        pool = SubprocessPoolRunner(
            cmds_to_run=self.cmds_to_run,
            max_parallel=self.autometric_runs_config.pool_max_parallel,
            autometrics_log_file=self.log_file,
            cwd=None,
            env=self._build_mera_env(),
        )

        pool.run_all()

    def get_metrics(self) -> dict[str, float]:
        if self.metrics_dir is None:
            Logger.warning("metrics_dir is not set for MeraEvaluator")
            return {}

        all_metrics = self._collect_metrics_from_dir(self.metrics_dir)
        filtered = {
            k: v for k, v in all_metrics.items() if k in self._METRICS_ALLOWLIST
        }
        self._normalize_metrics(filtered)
        Logger.info(
            f"MERA metrics: {len(filtered)} kept out of {len(all_metrics)} total"
        )
        return filtered

    def after_run(
        self, metrics: dict[str, float], uploaded: Optional[MeasurementUpload] = None
    ) -> None:
        self._send_metrics_to_jd(metrics)
        self._save_submission_metadata()
        self._generate_submissions()
        tar_path = JD.tar_metrics_dir(
            self.metrics_dir, self.artifacts_path, type(self).__name__
        )
        JD.symlink_metrics_for_transfer(
            tar_path, self.hf_checkpoint_path, type(self).__name__
        )

    def _generate_submissions(self) -> None:
        submissions_dir = self.metrics_dir / "submissions"
        os.makedirs(submissions_dir, exist_ok=True)
        self._cleanup_stale_submission_archives(submissions_dir)
        scripts_cwd = str(Path(self.MERA_TASKS_PATH).parent)

        scripts = [
            ("text", "log_to_submission_text.py", ["--gen"]),
            ("industrial", "log_to_submission_industrial.py", []),
            ("code", "log_to_submission_code.py", []),
        ]

        for category, script_name, extra_args in scripts:
            dst = str(submissions_dir / category)
            if Path(dst).exists():
                shutil.rmtree(dst)
            filtered_dir = self._create_filtered_outputs_dir(category)
            outputs_arg = filtered_dir or str(self.metrics_dir)

            cmd = [
                self.MERA_VENV_PYTHON,
                str(Path(self.MERA_SCRIPTS_DIR) / script_name),
                "--outputs_dir",
                outputs_arg,
                "--dst_dir",
                dst,
                *extra_args,
            ]
            Logger.info(f"Generating MERA {category} submission: {' '.join(cmd)}")
            try:
                result = subprocess.run(
                    cmd,
                    cwd=scripts_cwd,
                    capture_output=True,
                    text=True,
                    timeout=600,
                )
                if result.returncode != 0:
                    Logger.error(
                        f"MERA {category} submission failed (exit {result.returncode}): "
                        f"{result.stderr[:500]}"
                    )
                else:
                    Logger.info(f"MERA {category} submission generated at {dst}")
            except Exception as e:
                Logger.error(f"Failed to generate MERA {category} submission: {e}")
            finally:
                if filtered_dir:
                    shutil.rmtree(filtered_dir, ignore_errors=True)

        self._pack_all_submissions(submissions_dir)

    @staticmethod
    def _cleanup_stale_submission_archives(submissions_dir: Path) -> None:
        for zip_path in submissions_dir.glob("*.zip"):
            try:
                zip_path.unlink()
            except Exception as e:
                Logger.warning(
                    f"Failed to remove stale submission archive {zip_path}: {e}"
                )

    def _pack_all_submissions(self, submissions_dir: Path) -> None:
        category_zips = sorted(
            zp
            for zp in submissions_dir.glob("*.zip")
            if zp.name != "mera_submissions_all.zip"
        )
        if not category_zips:
            Logger.warning("No submission zips found, skipping combined archive")
            return

        archive_path = submissions_dir / "mera_submissions_all.zip"
        import zipfile

        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for zp in category_zips:
                zf.write(zp, zp.name)
            meta = submissions_dir / "metadata.json"
            if meta.exists():
                zf.write(meta, meta.name)

        Logger.info(
            f"Combined MERA submission archive: {archive_path} "
            f"({archive_path.stat().st_size / 1024 / 1024:.1f} MB, "
            f"{len(category_zips)} categories)"
        )

    def _save_submission_metadata(self) -> None:
        """Collect and save environment/model metadata required for MERA
        leaderboard submission."""
        submissions_dir = self.metrics_dir / "submissions"
        submissions_dir.mkdir(parents=True, exist_ok=True)

        metadata: dict[str, Any] = {
            "mera_version": self._read_file_stripped(self.MERA_VERSION_FILE),
            "codebase_version": self._read_file_stripped(self.MERA_VERSION_FILE),
            "seed": 42,
            "batch_size": max(self._used_batch_sizes) if self._used_batch_sizes else 1,
            "architecture": self._get_model_architecture(),
            "model_size": self._parse_model_size(),
            "gpu": "8 x H100",
            "apply_chat_template": True,
            "fewshot_as_multiturn": True,
        }

        metadata["precision"] = self._get_model_precision()

        for field, python_expr in [
            ("torch_version", "import torch; print(torch.__version__)"),
            ("cuda_version", "import torch; print(torch.version.cuda)"),
            (
                "transformers_version",
                "import transformers; print(transformers.__version__)",
            ),
        ]:
            metadata[field] = self._query_mera_venv(python_expr)

        meta_path = submissions_dir / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        Logger.info(f"Submission metadata saved to {meta_path}")

    def _query_mera_venv(self, python_expr: str) -> Optional[str]:
        """Run a short Python expression in the MERA venv and return stdout."""
        try:
            result = subprocess.run(
                [self.MERA_VENV_PYTHON, "-c", python_expr],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None

    def _get_model_architecture(self) -> Optional[str]:
        if self.hf_checkpoint_path is None:
            return None
        config_path = self.hf_checkpoint_path / "config.json"
        if not config_path.exists():
            return None
        try:
            with open(config_path, encoding="utf-8") as f:
                config = json.load(f)
            architectures = config.get("architectures")
            if architectures:
                return architectures[0]
            return config.get("model_type")
        except Exception:
            return None

    def _get_model_precision(self) -> Optional[str]:
        """Read effective precision from the HF checkpoint's config.json.
        Checks quantization_config first (e.g. fp8), falls back to torch_dtype."""
        if self.hf_checkpoint_path is None:
            return None
        config_path = self.hf_checkpoint_path / "config.json"
        if not config_path.exists():
            return None
        try:
            with open(config_path, encoding="utf-8") as f:
                config = json.load(f)
            quant_cfg = config.get("quantization_config")
            if quant_cfg and quant_cfg.get("quant_method"):
                return quant_cfg["quant_method"]
            return config.get("torch_dtype") or config.get("dtype")
        except Exception:
            return None

    def _parse_model_size(self) -> Optional[str]:
        """Extract model size (e.g. '10b', '700b') from the checkpoint path."""
        if self.hf_checkpoint_path is None:
            return self.model_info.size if self.model_info else None
        match = re.search(r"(\d+b)", str(self.hf_checkpoint_path), re.IGNORECASE)
        if match:
            return match.group(1)
        return self.model_info.size if self.model_info else None

    def _create_filtered_outputs_dir(self, category: str) -> Optional[str]:
        """Create a temp dir with symlinks to only the files relevant to
        the given category. This prevents pack_submission_logs() in MERA
        scripts from bundling ALL tasks' logs into every category's zip."""
        task_patterns = self._CATEGORY_TASKS.get(category)
        if not task_patterns:
            return None

        temp_dir = tempfile.mkdtemp(prefix=f"mera_{category}_")
        linked = 0
        for fname in os.listdir(self.metrics_dir):
            fpath = self.metrics_dir / fname
            if not fpath.is_file():
                continue
            if fname.startswith("results_"):
                os.symlink(str(fpath), os.path.join(temp_dir, fname))
                linked += 1
            elif fname.startswith("samples_"):
                task_name = self._extract_task_name(fname)
                if task_name and self._task_matches_category(task_name, task_patterns):
                    os.symlink(str(fpath), os.path.join(temp_dir, fname))
                    linked += 1

        if linked == 0:
            shutil.rmtree(temp_dir, ignore_errors=True)
            return None
        return temp_dir

    def _collect_metrics_from_dir(self, metrics_dir: Path) -> dict[str, float]:
        runs_files = [
            f
            for f in os.listdir(metrics_dir)
            if f.endswith(".json") and not f.startswith("metrics_step_")
        ]
        converted_metrics = {}
        for file in runs_files:
            metrics_run: dict[str, Any] = self._load_json_file(Path(metrics_dir, file))
            converted_metrics.update(self._aggregate_single_run(metrics_run))
        return converted_metrics

    def _aggregate_single_run(self, metrics_run: dict[str, Any]) -> dict[str, float]:
        num_fewshots = (
            next(iter(metrics_run["n-shot"].values()))
            if metrics_run.get("n-shot")
            else None
        )
        results: dict[str, Any] = metrics_run.get("results")
        if not results:
            return {}

        converted_metrics = {}
        for task_name, task_results in results.items():
            task_results.pop("alias", None)
            for submetric_name, submetric_value in task_results.items():
                if "_stderr" in submetric_name:
                    continue
                if not isinstance(submetric_value, (int, float)):
                    continue
                submetric_name = submetric_name.replace(",none", "").replace(",", ".")
                metric_key = f"mera/{task_name}/{num_fewshots}-shot/{submetric_name}"
                converted_metrics[metric_key] = submetric_value

        return converted_metrics

    def _mera_runs_to_bash_list(
        self,
        mera_runs_config: MeraEvaluatorConfig,
        metrics_dir: Path,
    ) -> dict[str, list[str]]:
        default_instruction = mera_runs_config.system_instruction
        if default_instruction is None:
            default_instruction = (
                self._read_file_stripped(self.MERA_SYSTEM_INSTRUCTION_FILE) or ""
            )

        result_commands_map = {}
        used_batch_sizes: set[int] = set()
        for task_name, task_config in mera_runs_config.tasks.items():
            used_batch_sizes.add(int(task_config.batch_size))
            if task_config.system_instruction is None and default_instruction:
                task_config.system_instruction = default_instruction

            run_bash_list = [
                self.MERA_VENV_PYTHON,
                "-m",
                "lm_eval",
                "--tasks",
                task_name,
                "--output_path",
                str(metrics_dir / f"{task_name}.json"),
                *task_config.to_bash_list(),
            ]

            result_commands_map[task_name] = run_bash_list
        self._used_batch_sizes = used_batch_sizes
        return result_commands_map

    def _convert_raw_config_to_bash(
        self, mera_runs_config: MeraEvaluatorConfig, metrics_dir: Path
    ) -> dict[str, list[str]]:
        before_calculation_skip = self._mera_runs_to_bash_list(
            mera_runs_config, metrics_dir
        )
        return {
            k: v
            for k, v in before_calculation_skip.items()
            if not self._is_already_calculated(k, self.metrics_dir)
        }

    @staticmethod
    def _drop_zero_metrics(metrics: dict[str, float]) -> dict[str, float]:
        return {name: value for name, value in metrics.items() if value != 0}

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
            jd_metrics=self._drop_zero_metrics(processed_metrics),
            shooter_name="mera",
            shooter_version=JD.get_shooter_version(self.MERA_VERSION_FILE),
            model_info=self.model_info,
        )
