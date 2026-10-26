import hashlib
import json
import math
import re
import sys
from itertools import zip_longest
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlsplit

from ...config_model import (
    ExternalApiInferenceConfig,
    ModelInfoConfig,
    SafetyUnifiedEvaluatorConfig,
)
from ...inference.external_api.proxy.mappers import OpenAiMapper
from ...inference.inference_manager import InferenceManager
from ...job_dispatcher import JD
from ...job_dispatcher.measurement import MeasurementUpload
from ...logger import Logger
from ...pam_proxies.models import SGLangChatCompletionRequest
from ...subprocess_manager import SubprocessManager
from .interface import ParallelInferenceEvaluator


class SafetyUnifiedEvaluator(ParallelInferenceEvaluator):
    SHOOTER_NAME = "safety_unified"
    SHOOTER_VERSION = "0.7.0"
    MODALITY = "text"
    _JD_METRIC_LEAVES = ("score", "red_fail_rate", "content_safe_rate")
    _JD_WEIGHTED = "wgt"
    _JD_REFUSAL_SAFE = "rsf"

    def setup(
        self,
        autometric_runs_config: SafetyUnifiedEvaluatorConfig,
        artifacts_path: Path,
        hf_checkpoint_path: Path,
        model_info: ModelInfoConfig,
    ) -> None:
        self.autometric_runs_config = autometric_runs_config
        self.artifacts_path = Path(artifacts_path)
        self.original_artifacts_path = Path(artifacts_path)
        self.hf_checkpoint_path = Path(hf_checkpoint_path)
        self.model_info = model_info
        self._create_metrics_dir(self.artifacts_path, "safety_unified")
        self._runner_root = Path(autometric_runs_config.runner_root)
        self._judge_tasks = autometric_runs_config.judges
        self._completed_judges: set[str] = set()
        self._metric_identities: dict[str, str] = {}
        self._selection_ready = False
        Logger.info(f"SafetyUnified judged setup OK, {len(self._judge_tasks)} judges")

    @property
    def checkpoint_model_name(self) -> str:
        _, checkpoint_model_name, _, _ = JD.parse_checkpoint_path(
            self.hf_checkpoint_path
        )
        return checkpoint_model_name

    @staticmethod
    def _judge_key(judge_model_name: str) -> str:
        return judge_model_name.replace("/", "_")

    def _judge_out_dir(self, judge_model_name: str) -> Path:
        return Path(self.metrics_dir) / self._judge_key(judge_model_name)

    def _judge_metric_prefix(self, judge_model_name: str) -> str:
        if (
            self.autometric_runs_config.selection_manifest
            and judge_model_name == "selected"
        ):
            return "safety_unified/selected/"
        return f"safety_unified/judge_{self._judge_key(judge_model_name)}/"

    def _metric_models(self) -> list[str]:
        return (
            ["selected"]
            if self.autometric_runs_config.selection_manifest
            else list(self._judge_tasks)
        )

    def _build_overrides(self, judge_model_name: str) -> dict[str, Any]:
        out_dir = self._judge_out_dir(judge_model_name)
        out_dir.mkdir(parents=True, exist_ok=True)
        return self.autometric_runs_config.build_overrides(
            judge_model_name=judge_model_name,
            out_dir=out_dir,
            default_answers_dir=Path(self.metrics_dir) / "answers",
            checkpoint_model_name=self.checkpoint_model_name,
            checkpoint_path=self.hf_checkpoint_path,
        )

    def _run_phase(self, phase: str, overrides: dict[str, Any]) -> bool:
        runner_script = self._runner_root / "run_safety_unified.py"
        if not runner_script.exists():
            Logger.error(
                f"safety_unified: runner not found in the image: {runner_script}"
            )
            return False
        cmd = [
            sys.executable,
            str(runner_script),
            "--phase",
            phase,
            "--overrides-json",
            json.dumps(overrides, default=str),
        ]
        Logger.info(f"SafetyUnified run phase={phase}")
        sm = SubprocessManager(cmd, self.log_file, str(self._runner_root))
        sm.start()
        sm.wait_for_completion()
        sm.log_error()
        return sm.is_successful()

    def _dataset_item_ids(self, judge_model_name: str) -> list[str]:
        config = self.autometric_runs_config
        dataset_root = Path(config.dataset_root)
        datasets: list[list[str]] = []
        try:
            for file_name in config.dataset_files_for(judge_model_name):
                path = dataset_root / file_name
                ids: list[str] = []
                with path.open(encoding="utf-8") as handle:
                    for lineno, line in enumerate(handle):
                        if not line.strip():
                            continue
                        item = json.loads(line)
                        if not isinstance(item, dict):
                            Logger.error(
                                f"SafetyUnified invalid dataset row at {path}:{lineno + 1}"
                            )
                            return []
                        item_id = item.get("id") or f"{path.stem}_{lineno}"
                        if (
                            not isinstance(item_id, str)
                            or Path(item_id).name != item_id
                            or item_id in {".", ".."}
                        ):
                            Logger.error(
                                f"SafetyUnified invalid dataset id at {path}:{lineno + 1}"
                            )
                            return []
                        ids.append(item_id)
                datasets.append(ids)
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            Logger.error(f"SafetyUnified cannot read dataset: {error}")
            return []
        item_ids = [
            item_id
            for batch in zip_longest(*datasets)
            for item_id in batch
            if item_id is not None
        ]
        if len(item_ids) != len(set(item_ids)):
            Logger.error("SafetyUnified duplicate dataset item ids")
            return []
        if config.limit:
            return item_ids[: int(config.limit)]
        return item_ids

    def _is_reusable_answer(self, answer_path: Path) -> bool:
        try:
            record = json.loads(answer_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeError):
            return False
        config = self.autometric_runs_config
        return (
            isinstance(record, dict)
            and isinstance(record.get("generated"), str)
            and bool(record["generated"].strip())
            and record.get("ok") is True
            and bool(record.get("mutate")) == config.mutate
            and record.get("mutation_seed", 0) == config.mutation_seed
        )

    def _require_reusable_answers(
        self, answers_dir: Path, judge_model_name: str
    ) -> bool:
        if not answers_dir.is_dir():
            Logger.error(
                "safety_unified: answers_dir is configured but does not exist: "
                f"{answers_dir}"
            )
            return False
        item_ids = self._dataset_item_ids(judge_model_name)
        if not item_ids:
            Logger.error(
                "safety_unified: the dataset configured for judge "
                f"{judge_model_name} is empty, refusing to skip the generation "
                f"phase: {self.autometric_runs_config.dataset_root}"
            )
            return False
        missing = [
            item_id
            for item_id in item_ids
            if not self._is_reusable_answer(answers_dir / f"{item_id}.json")
        ]
        if missing:
            Logger.error(
                f"safety_unified: answers_dir {answers_dir} holds no reusable answer "
                f"for {len(missing)} of {len(item_ids)} items of the dataset "
                f"configured for judge {judge_model_name}, refusing to skip the "
                f"generation phase, first missing: {missing[:5]}"
            )
            return False
        Logger.info(f"SafetyUnified reuses {len(item_ids)} answers from {answers_dir}")
        return True

    def _valid_external_judge(self, model_name: str, overrides: dict) -> bool:
        platform, _, model = model_name.partition(":")
        try:
            config = ExternalApiInferenceConfig.for_secondary_platform(
                platform, model, InferenceManager._secondary_inference
            )
            allowed = ({}, {"chat_template_kwargs": {"enable_thinking": False}})
            address = urlsplit(config.base_url)
            if (
                config.formatting != "openai"
                or config.request_customization not in allowed
                or address.scheme not in ("http", "https")
                or not address.hostname
                or address.username
                or address.password
                or address.query
                or address.fragment
            ):
                Logger.error(
                    "SafetyUnified external judge transport is incompatible with selection"
                )
                return False
            payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": "selection protocol probe"}],
                **{
                    field: overrides["judge"][field]
                    for field in ("max_tokens", "temperature", "seed")
                },
                "chat_template_kwargs": {"enable_thinking": False},
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "selection_probe",
                        "strict": True,
                        "schema": {"type": "object"},
                    },
                },
            }
            mapped = OpenAiMapper().map_request(
                SGLangChatCompletionRequest.model_validate(payload), model
            )
            mapped.update(config.request_customization)
            if (
                any(
                    mapped.get(field) != payload[field]
                    for field in (
                        "messages",
                        "max_tokens",
                        "temperature",
                        "seed",
                        "response_format",
                        "chat_template_kwargs",
                    )
                )
                or mapped.get("model") != model
            ):
                Logger.error(
                    "SafetyUnified external mapper changes the selected judge protocol"
                )
                return False
            return True
        except (AttributeError, KeyError, TypeError, ValueError):
            Logger.error(
                "SafetyUnified cannot verify the active external judge configuration"
            )
            return False

    def run_on_model_name(self, model_name: str) -> None:
        self._completed_judges.discard(model_name)
        self._metric_identities.pop(model_name, None)
        self._selection_ready = False
        self._metric_identities.pop("selected", None)
        overrides = self._build_overrides(model_name)
        try:
            roots = [self._judge_out_dir(model_name)]
            if self.autometric_runs_config.selection_manifest:
                roots.append(self._judge_out_dir("selected"))
            for root in roots:
                for name in (
                    "safety_unified_complete.json",
                    "safety_unified_metrics.json",
                ):
                    (root / name).unlink(missing_ok=True)
        except OSError as error:
            Logger.error(f"SafetyUnified cannot invalidate previous metrics: {error}")
            return
        if self.autometric_runs_config.selection_manifest and ":" in model_name:
            if not self._valid_external_judge(model_name, overrides):
                return
        configured_answers_dir = self.autometric_runs_config.answers_dir
        if configured_answers_dir is None:
            if not self._run_phase("generation", overrides):
                return
        else:
            if not self._require_reusable_answers(
                Path(configured_answers_dir), model_name
            ):
                return
        if self._run_phase("evaluation", overrides):
            self._completed_judges.add(model_name)
            if (
                self.autometric_runs_config.selection_manifest
                and self._completed_judges == set(self._judge_tasks)
            ):
                try:
                    self._judge_out_dir("selected").mkdir(parents=True, exist_ok=True)
                    aggregate_overrides = (
                        self.autometric_runs_config.build_selection_overrides(
                            metrics_dir=Path(self.metrics_dir),
                            checkpoint_model_name=self.checkpoint_model_name,
                            checkpoint_path=self.hf_checkpoint_path,
                        )
                    )
                except (OSError, ValueError) as error:
                    Logger.error(
                        f"SafetyUnified cannot prepare selected aggregation: {error}"
                    )
                    return
                self._selection_ready = self._run_phase("metrics", aggregate_overrides)
                if not self._selection_ready:
                    Logger.error(
                        "SafetyUnified selected aggregation failed, no selected metrics published"
                    )

    def _read_judge_metrics(self, judge_model_name: str) -> dict[str, float]:
        path = self._judge_out_dir(judge_model_name) / "safety_unified_metrics.json"
        if not path.exists():
            Logger.error(
                "safety_unified: the runner reported success for judge "
                f"{judge_model_name} but left no metrics at {path}"
            )
            return {}
        try:
            data = path.read_bytes()
            raw = json.loads(data)
            complete = json.loads(
                (path.parent / "safety_unified_complete.json").read_text(
                    encoding="utf-8"
                )
            )
        except (json.JSONDecodeError, OSError, UnicodeError) as error:
            Logger.error(f"safety_unified: unreadable metrics at {path}: {error}")
            return {}
        if not isinstance(raw, dict) or any(
            value is not None
            and (type(value) not in (int, float) or not math.isfinite(value))
            for value in raw.values()
        ):
            Logger.error(f"safety_unified: invalid metric values at {path}")
            return {}
        scope_model = (
            next(iter(self._judge_tasks))
            if judge_model_name == "selected"
            and self.autometric_runs_config.selection_manifest
            else judge_model_name
        )
        total = len(self._dataset_item_ids(scope_model))
        prefix = "safety_unified/overall/"
        if (
            total == 0
            or raw.get(prefix + "n_total") != total
            or raw.get(prefix + "n_judged") != total
            or raw.get(prefix + "coverage") != 1.0
            or raw.get(prefix + "n_parse_fail") != 0
            or raw.get(prefix + "n_no_answer") != 0
        ):
            Logger.error(
                f"safety_unified: incomplete metrics at {path}, expected {total} items"
            )
            return {}
        if (
            not isinstance(complete, dict)
            or any(
                type(complete.get(key)) is not int
                for key in ("metric_version", "n_expected", "n_scored")
            )
            or complete.get("runner_version") != self.SHOOTER_VERSION
            or complete.get("metric_version") != 3
            or complete.get("n_expected") != total
            or complete.get("n_scored") != total
            or complete.get("metrics_sha256") != hashlib.sha256(data).hexdigest()
            or not isinstance(complete.get("metric_identity"), str)
            or re.fullmatch(r"[a-f0-9]{64}", complete["metric_identity"]) is None
        ):
            Logger.error(
                f"safety_unified: invalid completion manifest at {path.parent}"
            )
            return {}
        if (
            judge_model_name == "selected"
            and self.autometric_runs_config.selection_manifest
        ):
            if not self._valid_selection_completion(path.parent, complete):
                return {}
        self._metric_identities[judge_model_name] = complete["metric_identity"]
        return {key: float(value) for key, value in raw.items() if value is not None}

    def _selection_source_hashes(self) -> dict | None:
        result = {}
        runner_hash = hashlib.sha256(
            (self._runner_root / "run_safety_unified.py").read_bytes()
        ).hexdigest()
        for judge in self._judge_tasks:
            root = self._judge_out_dir(judge).resolve()
            path = root / "safety_source_complete.json"
            if not path.resolve().is_relative_to(root):
                return None
            data = path.read_bytes()
            proof = json.loads(data)
            model = judge.partition(":")[2] if ":" in judge else judge
            if (
                not isinstance(proof, dict)
                or proof.get("format") != "safety-source-pass-v2"
                or proof.get("runner_sha256") != runner_hash
                or proof.get("runner_version") != self.SHOOTER_VERSION
                or proof.get("judge_key") != judge
                or proof.get("model") != model
                or set(proof["records"]) != set(self._dataset_item_ids(judge))
                or model in result
            ):
                return None
            paths = {
                "safety_unified_complete.json": proof["completion_sha256"],
                "safety_unified_metrics.json": proof["metrics_sha256"],
                **{
                    f"judgments/{key}.json": value["judgment_sha256"]
                    for key, value in proof["records"].items()
                },
            }
            for name, expected in paths.items():
                artifact_path = (root / name).resolve()
                if (
                    not artifact_path.is_relative_to(root)
                    or hashlib.sha256(artifact_path.read_bytes()).hexdigest()
                    != expected
                ):
                    return None
            result[model] = hashlib.sha256(data).hexdigest()
        return result

    def _valid_selection_completion(self, root: Path, complete: dict) -> bool:
        config = self.autometric_runs_config
        try:
            package_root = Path(config.dataset_root).resolve()
            manifest_path = (package_root / config.selection_manifest).resolve()
            if not manifest_path.is_relative_to(package_root):
                Logger.error("SafetyUnified selection manifest is outside the package")
                return False
            manifest_data = manifest_path.read_bytes()
            manifest = json.loads(manifest_data)
            assets = {"policy.txt"} | {
                source["prompt"] for source in manifest["sources"].values()
            }
            if config.domain_rules_file:
                assets.add(config.domain_rules_file)
            asset_hashes = {}
            for name in assets:
                asset_path = (package_root / name).resolve()
                if not asset_path.is_relative_to(package_root):
                    return False
                asset_hashes[name] = hashlib.sha256(asset_path.read_bytes()).hexdigest()
            source_hashes = self._selection_source_hashes()
            valid = (
                complete.get("selection_manifest_sha256")
                == hashlib.sha256(manifest_data).hexdigest()
                and complete.get("source_order") == manifest["source_order"]
                and complete.get("mode") == manifest["mode"]
                and complete.get("dataset_file_hashes")
                == {
                    name: hashlib.sha256((package_root / name).read_bytes()).hexdigest()
                    for name in config.dataset_files_for(next(iter(self._judge_tasks)))
                }
                and complete.get("answer_file_hashes")
                == {
                    key: hashlib.sha256(
                        (
                            Path(
                                config.answers_dir or Path(self.metrics_dir) / "answers"
                            )
                            / f"{key}.json"
                        ).read_bytes()
                    ).hexdigest()
                    for key in self._dataset_item_ids(next(iter(self._judge_tasks)))
                }
                and complete.get("verdicts_sha256")
                == hashlib.sha256(
                    (root / "selected_verdicts.json").read_bytes()
                ).hexdigest()
                and complete.get("source_asset_hashes") == asset_hashes
                and source_hashes is not None
                and complete.get("source_proof_hashes") == source_hashes
                and isinstance(complete.get("aggregation_cache_key"), str)
                and re.fullmatch(r"[a-f0-9]{64}", complete["aggregation_cache_key"])
                is not None
            )
            if manifest["mode"] in {"learned", "hierarchy"}:
                entry = manifest["classifier"]
                artifact_path = (manifest_path.parent / entry["path"]).resolve()
                valid = (
                    valid
                    and artifact_path.is_relative_to(manifest_path.parent)
                    and hashlib.sha256(artifact_path.read_bytes()).hexdigest()
                    == entry["sha256"]
                    and complete.get("classifier_artifact_sha256")
                    == entry["artifact_sha256"]
                )
        except (
            OSError,
            UnicodeError,
            ValueError,
            KeyError,
            TypeError,
            AttributeError,
        ) as error:
            Logger.error(f"SafetyUnified cannot verify selected artifacts: {error}")
            return False
        if not valid:
            Logger.error(
                "SafetyUnified selected completion does not match current source or classifier artifacts"
            )
        return valid

    def get_metrics(self) -> dict[str, float]:
        if self.metrics_dir is None:
            Logger.warning("metrics_dir is not set for SafetyUnifiedEvaluator")
            return {}
        if self._completed_judges != set(self._judge_tasks):
            Logger.error(
                "SafetyUnified did not complete every configured judge, no metrics published"
            )
            return {}
        if self.autometric_runs_config.selection_manifest and not self._selection_ready:
            Logger.error("SafetyUnified selected aggregation is not complete")
            return {}
        result: dict[str, float] = {}
        for judge_model_name in self._metric_models():
            prefix = self._judge_metric_prefix(judge_model_name)
            judge_metrics = self._read_judge_metrics(judge_model_name)
            if not judge_metrics:
                return {}
            for key, value in judge_metrics.items():
                result[prefix + key.split("/", 1)[-1]] = value
        Logger.info(f"Collected {len(result)} SafetyUnified metrics")
        return result

    def _jd_metric_name(self, judge_model_name: str, metric_tail: str) -> str:
        *sections, leaf = metric_tail.split("/")
        identity = self._metric_identities[judge_model_name]
        scope_model = (
            next(iter(self._judge_tasks))
            if judge_model_name == "selected"
            and self.autometric_runs_config.selection_manifest
            else judge_model_name
        )
        scope = self.autometric_runs_config.dataset_scope_for(scope_model)
        slices = [
            section for section in sections if section not in {"stub_safe", "v2", "v3"}
        ]
        name = "_".join(
            (
                "safety",
                self.MODALITY,
                scope,
                "v3",
                self._JD_REFUSAL_SAFE if "stub_safe" in sections else self._JD_WEIGHTED,
                *slices,
                leaf,
                judge_model_name,
            )
        )
        return f"{name.replace('/', '_').upper()}_{identity[:12].upper()}"

    def _send_metrics_to_jd(self, processed_metrics: dict[str, float]) -> None:
        if not JD.can_send_metrics(
            metrics=processed_metrics,
            hf_checkpoint_path=self.hf_checkpoint_path,
            metrics_dir=self.metrics_dir,
            model_info=self.model_info,
        ):
            return
        jd_metrics: dict[str, float] = {}
        for judge_model_name in self._metric_models():
            if judge_model_name not in self._metric_identities:
                Logger.error(
                    "SafetyUnified has no verified metric identity, no JD metrics published"
                )
                return
            prefix = self._judge_metric_prefix(judge_model_name)
            for key, value in processed_metrics.items():
                if not key.startswith(prefix):
                    continue
                metric_tail = key[len(prefix) :]
                if metric_tail.rsplit("/", 1)[-1] not in self._JD_METRIC_LEAVES:
                    continue
                jd_name = self._jd_metric_name(judge_model_name, metric_tail)
                if jd_name in jd_metrics:
                    Logger.error(f"SafetyUnified duplicate JD metric key: {jd_name}")
                    return
                jd_metrics[jd_name] = value
        JD.send_evaluator_metrics(
            hf_checkpoint_path=self.hf_checkpoint_path,
            metrics_dir=self.metrics_dir,
            jd_metrics=jd_metrics,
            shooter_name=self.SHOOTER_NAME,
            shooter_version=self.SHOOTER_VERSION,
            model_info=self.model_info,
        )

    def after_run(
        self, metrics: dict[str, float], uploaded: Optional[MeasurementUpload] = None
    ) -> None:
        if not self.autometric_runs_config.publish_metrics:
            return
        if not metrics or self._completed_judges != set(self._judge_tasks):
            return
        self._send_metrics_to_jd(metrics)
        tar_path = JD.tar_metrics_dir(
            self.metrics_dir, self.original_artifacts_path, type(self).__name__
        )
        JD.symlink_metrics_for_transfer(
            tar_path, self.hf_checkpoint_path, type(self).__name__
        )
