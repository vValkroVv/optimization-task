import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config_model import ModelInfoConfig
from ..logger import Logger
from .submetrics import JDMetricEntry, SubmetricGroup, is_reportable


def _log_send_failure(retry_state) -> bool:
    Logger.error(
        f"Failed to send metrics to JD after {retry_state.attempt_number} attempt(s): "
        f"{retry_state.outcome.exception()}"
    )
    return False


class JDMetrics:
    reports_to_jd: bool = True
    _deferred_bodies: Optional[list[tuple[dict, Optional[Path]]]] = None

    @classmethod
    @contextmanager
    def metrics_sent_at_end(cls, enabled: bool):
        if not enabled:
            yield
            return
        cls._deferred_bodies = []
        try:
            yield
        finally:
            deferred, cls._deferred_bodies = cls._deferred_bodies, None
            for body, hf_checkpoint_path in deferred:
                cls.send_metrics(body, hf_checkpoint_path)

    @classmethod
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=15),
        reraise=False,
        retry_error_callback=_log_send_failure,
    )
    def send_metrics(cls, body: dict, hf_checkpoint_path: Optional[Path]) -> bool:
        if not cls.reports_to_jd:
            Logger.info("Answers come from a recording; skip sending metrics to jd")
            return False
        if cls._deferred_bodies is not None:
            cls._deferred_bodies.append((body, hf_checkpoint_path))
            Logger.info("Metrics will be sent to JD at the end of the measurement")
            return True
        if hf_checkpoint_path is not None:
            checkpoint_history = cls.collect_checkpoint_jd_data(
                cls.checkpoint_artifacts_path(hf_checkpoint_path)
            )
            if checkpoint_history is not None:
                body["checkpoint_history"] = checkpoint_history

        jd_base_url = cls.get_jd_base_url()
        res = requests.post(
            f"{jd_base_url}/ml_models/add_metrics",
            json=body,
            headers={"Authorization": "YlhIrKHpvuv0pBb6CK95kFv2AADp/79EUJ+njxqtSqk="},
            verify=False,
            timeout=15,
        )
        try:
            res.raise_for_status()
        except requests.HTTPError:
            Logger.error(f"JD rejected metrics ({res.status_code}): {res.text[:2000]}")
            raise
        Logger.info("Metrics successfully sent to JD")
        return True

    @classmethod
    def save_metrics_json(cls, body: dict, step: int, metrics_dir: Path) -> None:
        filename = f"metrics_step_{step}.json"
        save_path = metrics_dir / filename

        try:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(body, f, ensure_ascii=False, indent=4, allow_nan=False)
            Logger.info(f"Saved metrics to {save_path}")
        except Exception as e:
            Logger.error(f"Error while saving JSON metrics file: {e}")

    @classmethod
    def generate_metrics_json(
        cls,
        metrics: dict[int, dict[str, JDMetricEntry]],
    ) -> dict[int, dict[str, JDMetricEntry]]:
        result = {}
        for step_number, step_metrics in metrics.items():
            step_data = {}
            for dataset_name, entry in step_metrics.items():
                if isinstance(entry, dict):
                    if not is_reportable(entry["value"]):
                        continue
                    step_data[dataset_name] = {
                        "value": entry["value"],
                        "submetrics": {
                            name: value
                            for name, value in entry["submetrics"].items()
                            if is_reportable(value)
                        },
                    }
                elif is_reportable(entry):
                    step_data[dataset_name] = entry

            result[step_number] = step_data

        return result

    @classmethod
    def get_custom_model_name_if_exist(cls, model_name: str) -> str:
        custom_model_name = os.environ.get("CUSTOM_MODEL_NAME")
        if custom_model_name:
            model_name = custom_model_name
        return model_name

    @classmethod
    def set_autocompare_id_in_body(cls, body: dict) -> None:
        autocompare_id = os.environ.get("AUTOCOMPARE_ID")
        if autocompare_id:
            body["autocompare_id"] = autocompare_id

    @classmethod
    def get_shooter_version(cls, file_path: str) -> str:
        try:
            with open(file_path, "r") as f:
                return f.read().strip()
        except Exception as e:
            Logger.error(f"Error in getting shooter_version: {e}")
            return "PAM_FILE"

    @classmethod
    def can_send_metrics(
        cls,
        *,
        metrics: dict[str, float],
        hf_checkpoint_path: Optional[Path],
        metrics_dir: Optional[Path],
        model_info: Optional[ModelInfoConfig],
    ) -> bool:
        if not metrics:
            Logger.warning("No metrics to report, skipping JD metrics send")
            return False
        if hf_checkpoint_path is None:
            Logger.warning("hf_checkpoint_path is not set, skipping JD metrics send")
            return False
        if metrics_dir is None:
            Logger.warning("metrics_dir is not set, skipping JD metrics send")
            return False
        if model_info is None:
            Logger.warning("model_info is not set, skipping JD metrics send")
            return False
        return True

    @classmethod
    def build_metrics_body(
        cls,
        *,
        hf_checkpoint_path: Path,
        jd_metrics: dict[str, JDMetricEntry],
        shooter_name: str,
        shooter_version: str,
        model_info: ModelInfoConfig,
        extra_body: Optional[dict] = None,
        submetric_groups: list[SubmetricGroup] | None = None,
    ) -> tuple[dict, int]:
        nfs_path, model_name, _, step = cls.parse_checkpoint_path(hf_checkpoint_path)
        model_name = cls.get_custom_model_name_if_exist(model_name=model_name)
        Logger.info(f"Model name for saving: {model_name}")

        nested_metrics = cls.nest_submetrics(jd_metrics, submetric_groups or [])
        body = {
            "nfs_path": nfs_path,
            "name": model_name,
            "metrics": cls.generate_metrics_json({step: nested_metrics}),
            "shooter_name": shooter_name,
            "shooter_version": shooter_version,
            "model_class": model_info.model_class,
            "size": model_info.size,
            "username": model_info.username,
            "wandb_entity": None,
            "wandb_project": None,
            "wandb_run_id": None,
        }
        if extra_body:
            body.update(extra_body)
        cls.set_autocompare_id_in_body(body=body)
        return body, step

    @classmethod
    def send_evaluator_metrics(
        cls,
        *,
        hf_checkpoint_path: Path,
        metrics_dir: Path,
        jd_metrics: dict[str, JDMetricEntry],
        shooter_name: str,
        shooter_version: str,
        model_info: ModelInfoConfig,
        extra_body: Optional[dict] = None,
        submetric_groups: list[SubmetricGroup] | None = None,
    ) -> None:
        body, step = cls.build_metrics_body(
            hf_checkpoint_path=hf_checkpoint_path,
            jd_metrics=jd_metrics,
            shooter_name=shooter_name,
            shooter_version=shooter_version,
            model_info=model_info,
            extra_body=extra_body,
            submetric_groups=submetric_groups,
        )
        cls.save_metrics_json(body, step, metrics_dir)
        cls.send_metrics(body, hf_checkpoint_path)
