import hashlib
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Optional

from ...job_dispatcher import JD
from ...job_dispatcher.cache_scores.scalar import ScalarCacheScoresBuilder
from ...config_model import ForgeEvaluatorConfig, ModelInfoConfig
from ...job_dispatcher.measurement import MeasurementUpload
from ...logger import Logger
from ...metrics.evaluators.interface import (
    ParallelInferenceEvaluator,
)
from ...subprocess_manager import SubprocessManager


class ForgeEvaluator(ParallelInferenceEvaluator):
    SHOOTER_NAME = "forge"
    ARCHIVE_PATH = Path("/home/user/forge_lite.tar.zst")
    RUNTIME_ROOT = Path("/home/user/forge_lite")
    UNZIP_LOG_PATH = Path("/home/user/unzip_forge_lite.log")

    def setup(
        self,
        autometric_runs_config: ForgeEvaluatorConfig,
        artifacts_path: Path,
        hf_checkpoint_path: Path,
        model_info: ModelInfoConfig,
    ) -> None:
        self.autometric_runs_config = autometric_runs_config
        self.artifacts_path = Path(artifacts_path)
        self.hf_checkpoint_path = Path(hf_checkpoint_path)

        self.model_info = model_info
        self.all_metrics: dict[str, float] = {}
        self._failed = False
        self._create_metrics_dir(self.artifacts_path, self.SHOOTER_NAME)

        self.forge_root = self._unzip_forge_lite()
        self.recipes_dir = self.forge_root / "recipes"
        self.data_dir = self.forge_root / "data"

        self._recipes_by_judge = self._build_recipes_by_judge()

        Logger.info(
            f"ForgeEvaluator setup: forge_root={self.forge_root}, "
            f"attack_model={autometric_runs_config.attack_model_name}, "
            f"attack_api_base={autometric_runs_config.attack_api_base}, "
            f"judge_endpoints={autometric_runs_config.judge_endpoints}, "
            f"recipes_by_judge={self._recipes_by_judge}"
        )

    def _is_already_calculated(self, results_path: Path) -> bool:
        return results_path.exists()

    def run_on_model_name(
        self,
        model_name: str,
    ) -> None:
        if self._failed:
            return

        recipes = self._recipes_by_judge.get(model_name)
        if not recipes:
            Logger.info(f"No FORGE recipes for judge '{model_name}', skipping run")
            return

        self._validate_runtime()

        group_dir = self.metrics_dir / self._safe_name(model_name)
        results_path = group_dir / "results.json"

        if self._is_already_calculated(results_path):
            Logger.info(
                f"FORGE results already present for judge "
                f"'{model_name}' at {results_path}, skipping run"
            )
            self._collect_results(model_name, results_path, recipes)
            return

        group_dir.mkdir(parents=True, exist_ok=True)
        override_map = {
            recipe: self.autometric_runs_config.build_meta_override(recipe)
            for recipe in recipes
        }
        override_path = group_dir / "meta_override.json"
        override_path.write_text(
            json.dumps(override_map, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        cmd = self._build_cmd(recipes, override_path, results_path)
        Logger.info(
            f"Starting FORGE_LITE run for recipes {recipes} (judge={model_name}): {cmd}"
        )
        sm = SubprocessManager(
            cmd,
            self.log_file,
            str(self.forge_root),
            f"FORGE {model_name}",
            self._build_env(),
        )
        sm.start()
        sm.wait_for_completion()
        sm.log_error()

        if not sm.is_successful():
            self._abort(f"FORGE_LITE exited with code {sm.exit_code}")
            return

        self._collect_results(model_name, results_path, recipes)

    def _collect_results(
        self, model_name: str, results_path: Path, recipes: list[str]
    ) -> None:
        group_metrics = self._parse_results(results_path, recipes)
        if group_metrics is None:
            self._abort(
                f"FORGE_LITE run for judge '{model_name}' had "
                "technical failures (missing results / dead attack or judge "
                "endpoint); aborting benchmark"
            )
            return

        self.all_metrics.update(group_metrics)

    def _validate_runtime(self) -> None:
        runtime_python = Path(self.autometric_runs_config.runtime_python)
        if not runtime_python.is_file() or not os.access(runtime_python, os.X_OK):
            raise FileNotFoundError(
                f"FORGE runtime_python not found/executable: {runtime_python}. "
                "See ForgeEvaluatorConfig.runtime_python "
                "(default /home/user/forge_lite_eval/bin/python)."
            )
        if not self.recipes_dir.is_dir():
            raise FileNotFoundError(
                f"FORGE recipes dir not found: {self.recipes_dir} "
                f"(forge_root={self.forge_root}). See Docker archive "
                "/home/user/forge_lite.tar.zst unpacked to /home/user/forge_lite."
            )

    def _abort(self, reason: str) -> None:
        Logger.error(reason)
        self._failed = True
        self.all_metrics = {}
        if not self.autometric_runs_config.save_cache:
            for path in self.metrics_dir.iterdir():
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)

    def get_metrics(self) -> dict[str, float]:
        if self._failed:
            return {}
        return self.all_metrics

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
        shutil.rmtree(self.RUNTIME_ROOT, ignore_errors=True)

    def _build_recipes_by_judge(self) -> dict[str, list[str]]:
        cfg = self.autometric_runs_config
        available = sorted(p.stem for p in self.recipes_dir.glob("*.attack"))

        if cfg.attacks is None:
            selected = available
        else:
            missing = [a for a in cfg.attacks if a not in available]
            if missing:
                raise ValueError(
                    f"FORGE recipes not found: {missing}. Available: {available}"
                )
            selected = list(cfg.attacks)

        if not selected:
            raise ValueError(f"No FORGE recipes selected (available: {available})")

        recipes_by_judge: dict[str, list[str]] = {}
        for recipe in selected:
            recipes_by_judge.setdefault(cfg.judge_for(recipe), []).append(recipe)
        return recipes_by_judge

    def _build_cmd(
        self, task_subset: list[str], override_path: Path, results_path: Path
    ) -> list[str]:
        cfg = self.autometric_runs_config
        cmd = [
            str(cfg.runtime_python),
            "-m",
            "forge_lite",
            "run",
            str(self.recipes_dir),
            "-o",
            str(results_path),
            "-a",
            ",".join(task_subset),
            "--meta-override",
            str(override_path),
            "-w",
            str(cfg.workers),
            "-p",
            str(cfg.recipe_workers),
        ]
        if cfg.fail_on_runtime_error:
            cmd.append("--fail-on-runtime-error")
        if cfg.limit is not None:
            cmd.extend(["-n", str(cfg.limit)])
        if cfg.combinations is not None:
            cmd.extend(["-c", str(cfg.combinations)])
        return cmd

    def _build_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["FORGE_LITE_DATA_DIR"] = str(self.data_dir)
        env["FORGE_LITE_RECIPES_DIR"] = str(self.recipes_dir)
        return env

    def _parse_results(
        self, results_path: Path, recipes: list[str]
    ) -> Optional[dict[str, float]]:
        if not results_path.exists():
            Logger.warning(f"FORGE results JSON not found: {results_path}")
            return None

        data = json.loads(results_path.read_text(encoding="utf-8"))

        if "attacks" in data:  # multi-recipe output
            per_recipe = [
                (name, payload.get("metrics", {}))
                for name, payload in data["attacks"].items()
            ]
        elif "metrics" in data:  # single-recipe output
            name = data.get("metadata", {}).get("attack_name") or (
                recipes[0] if recipes else "unknown"
            )
            per_recipe = [(name, data["metrics"])]
        else:
            Logger.warning(
                f"FORGE results JSON has unexpected shape (no 'attacks'/'metrics'): "
                f"{results_path}"
            )
            return None

        out: dict[str, float] = {}
        for name, metrics in per_recipe:
            total = metrics.get("total", 0)
            evaluated = metrics.get("evaluated", 0)
            if not total or evaluated < total:
                Logger.error(
                    f"FORGE recipe '{name}': {evaluated}/{total} runs evaluated "
                    "(attack target or judge endpoint likely failed)"
                )
                return None
            out[f"forge/{name}/asr"] = float(metrics.get("asr", 0.0))
            out[f"forge/{name}/br"] = float(metrics.get("blacklist_rate", 0.0))

        return out

    def _unzip_forge_lite(self) -> Path:
        archive = self.ARCHIVE_PATH
        target_dir = self.RUNTIME_ROOT

        if not archive.exists():
            Logger.warning(
                f"FORGE archive not found: {archive}. Using {target_dir} as is."
            )
            return target_dir

        Logger.info("Unzipping FORGE_LITE")
        if target_dir.exists():
            shutil.rmtree(target_dir, ignore_errors=True)
        if self.UNZIP_LOG_PATH.exists():
            self.UNZIP_LOG_PATH.unlink()

        unzip_cmd = [
            "tar",
            "--zstd",
            "-xf",
            archive.as_posix(),
            "-C",
            target_dir.parent.absolute().as_posix(),
        ]
        sm = SubprocessManager(unzip_cmd, self.UNZIP_LOG_PATH, None, "Unzip FORGE_LITE")
        sm.start()
        sm.wait_for_completion()
        if self.UNZIP_LOG_PATH.exists():
            self.UNZIP_LOG_PATH.unlink()
        sm.raise_for_exit_code()
        Logger.info("Successfully unzipped FORGE_LITE")
        return target_dir

    @staticmethod
    def _safe_name(name: str) -> str:
        return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)

    def _send_metrics_to_jd(self, processed_metrics: dict[str, float]) -> None:
        if not JD.can_send_metrics(
            metrics=processed_metrics,
            hf_checkpoint_path=self.hf_checkpoint_path,
            metrics_dir=self.metrics_dir,
            model_info=self.model_info,
        ):
            return

        jd_metrics = {
            key.replace("/", "_").upper(): value
            for key, value in processed_metrics.items()
        }
        scores = ScalarCacheScoresBuilder(self.metrics_dir, jd_metrics)
        for judge, recipes in self._recipes_by_judge.items():
            path = self.metrics_dir / self._safe_name(judge) / "results.json"
            try:
                data = json.loads(path.read_text())
                attacks = data.get("attacks") or {data["metadata"]["attack_name"]: data}
                for recipe in recipes:
                    payload = attacks[recipe]
                    rows = payload["results"]
                    total = payload["metrics"]["total"]
                    if (
                        not total
                        or len(rows) != total
                        or payload["metrics"]["evaluated"] != total
                    ):
                        Logger.warning(
                            f"FORGE CI unavailable for {recipe}: incomplete results"
                        )
                        continue
                    prompts = defaultdict(dict)
                    stimuli = []
                    valid = True
                    for row in rows:
                        uid = hashlib.sha256(row["prompt"].encode()).hexdigest()
                        parameters = json.dumps(row["parameters"], sort_keys=True)
                        if (
                            row.get("error")
                            or row["judge_result"] not in {"safe", "unsafe"}
                            or parameters in prompts[uid]
                        ):
                            valid = False
                            break
                        prompts[uid][parameters] = float(
                            row["judge_result"] == "unsafe"
                        )
                        stimuli.append((uid, parameters, row["messages_sent"]))
                    if (
                        not valid
                        or len({tuple(sorted(values)) for values in prompts.values()})
                        != 1
                    ):
                        Logger.warning(
                            f"FORGE CI unavailable for {recipe}: incomplete prompt/parameter matrix"
                        )
                        continue
                    override = self.autometric_runs_config.build_meta_override(recipe)
                    for section in override.values():
                        for field in ("base_url", "api_key_env"):
                            section.pop(field, None)
                    override["model"].pop("model", None)
                    identity = [
                        override,
                        hashlib.sha256(
                            (self.recipes_dir / f"{recipe}.attack").read_bytes()
                        ).hexdigest(),
                        hashlib.sha256(
                            json.dumps(sorted(stimuli), sort_keys=True).encode()
                        ).hexdigest(),
                    ]
                    scores.add_metric(
                        f"FORGE_{recipe}_ASR".upper(),
                        [
                            (uid, sum(values.values()) / len(values))
                            for uid, values in prompts.items()
                        ],
                        identity,
                        tolerance=0.0000500001,
                    )
            except (OSError, ValueError, KeyError, TypeError) as error:
                Logger.warning(f"FORGE CI unavailable for judge {judge}: {error}")
        JD.send_evaluator_metrics_with_scores(
            scores=scores,
            hf_checkpoint_path=self.hf_checkpoint_path,
            metrics_dir=self.metrics_dir,
            shooter_name=self.SHOOTER_NAME,
            shooter_version=JD.get_shooter_version("/home/user/forge_lite_version.txt"),
            model_info=self.model_info,
        )
