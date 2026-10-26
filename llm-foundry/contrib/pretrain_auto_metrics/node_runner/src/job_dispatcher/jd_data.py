import json
import uuid
import os
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field, ValidationError

from ..logger import Logger


class CheckpointHistoryAction(StrEnum):
    autometrics = "autometrics"
    convertation = "convertation"


class CheckpointHistoryInfoSchema(BaseModel):
    id: UUID
    nfs_path: str
    created_at: datetime
    step: int
    checkpoint_type: str
    checkpoint_creation_type: str
    workspace: Optional[str] = None
    parent_id: Optional[UUID] = None


class CheckpointHistoryRecordCreateSchema(BaseModel):
    id: UUID
    action: CheckpointHistoryAction
    created_at: datetime
    data: Optional[dict[str, Any]] = None
    checkpoint_id: UUID


class CheckpointHistoryCreateSchema(BaseModel):
    checkpoint_info: CheckpointHistoryInfoSchema
    history: list[CheckpointHistoryRecordCreateSchema] = Field(default_factory=list)


class JDData:
    @classmethod
    def _utc_now(cls) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    @classmethod
    def _new_uuid(cls) -> str:
        return str(uuid.uuid4())

    @classmethod
    def _load_jd_data(cls, jd_data_path: Path) -> Optional[dict]:
        if jd_data_path.is_file():
            try:
                with open(jd_data_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                Logger.warning(f"Failed to read jd_data.json at {jd_data_path}: {e}")
        return None

    @classmethod
    def _write_jd_data(cls, jd_data: dict, jd_data_path: Path) -> None:
        jd_data_path.parent.mkdir(parents=True, exist_ok=True)
        with open(jd_data_path, "w", encoding="utf-8") as f:
            json.dump(jd_data, f, ensure_ascii=False, indent=4)

    @classmethod
    def checkpoint_artifacts_path(cls, hf_checkpoint_path: Path) -> Path:
        return Path(hf_checkpoint_path, "artifacts")

    @classmethod
    def copy_jd_data_for_quantized_checkpoint(
        cls, source_artifacts_path: Path, quantized_checkpoint_path: Path
    ) -> None:
        jd_data_path = (
            cls.checkpoint_artifacts_path(quantized_checkpoint_path) / "jd_data.json"
        )
        if jd_data_path.is_file():
            return
        source_jd_data = cls._load_jd_data(source_artifacts_path / "jd_data.json")
        if source_jd_data is None:
            return
        source_jd_data["history"] = []
        checkpoint_info = source_jd_data.setdefault("checkpoint_info", {})
        checkpoint_info.update(
            id=cls._new_uuid(),
            nfs_path=str(quantized_checkpoint_path),
            created_at=cls._utc_now(),
            checkpoint_type="hf_fp8",
            checkpoint_creation_type="convertation",
            parent_id=checkpoint_info.get("id"),
        )
        cls._write_jd_data(source_jd_data, jd_data_path)
        Logger.info(
            f"Created {jd_data_path} from {source_artifacts_path} with empty history"
        )

    @classmethod
    def collect_checkpoint_jd_data(cls, artifacts_path: Path) -> Optional[list[dict]]:
        jd_data = cls._load_jd_data(artifacts_path / "jd_data.json")
        if jd_data is None:
            return None
        try:
            checkpoint_history = CheckpointHistoryCreateSchema.model_validate(jd_data)
        except ValidationError as e:
            Logger.warning(
                f"jd_data.json does not match CheckpointHistoryCreateSchema, "
                f"skip sending checkpoint_history: {e}"
            )
            return None
        return [checkpoint_history.model_dump(mode="json")]

    @classmethod
    def _checkpoint_type_and_id(
        cls, jd_data: Optional[dict]
    ) -> Optional[tuple[str, str]]:
        checkpoint_info = (jd_data or {}).get("checkpoint_info", {})
        checkpoint_type = checkpoint_info.get("checkpoint_type")
        checkpoint_id = checkpoint_info.get("id")
        if not checkpoint_type or not checkpoint_id:
            return None
        return str(checkpoint_type), str(checkpoint_id)

    @classmethod
    def read_checkpoint_type_and_id(
        cls, artifacts_path: Path
    ) -> Optional[tuple[str, str]]:
        """JD checkpoint type and id, or ``None`` when either one is unknown."""
        checkpoint = cls._checkpoint_type_and_id(
            cls._load_jd_data(artifacts_path / "jd_data.json")
        )
        if checkpoint is None:
            Logger.warning(
                f"No checkpoint type or id in {artifacts_path / 'jd_data.json'}; "
                "artifact keys go without them"
            )
        return checkpoint

    @classmethod
    def _find_run_record(cls, history: list, run_uuid: str) -> Optional[dict]:
        for record in history:
            if (
                record.get("action") == CheckpointHistoryAction.autometrics.value
                and (record.get("data") or {}).get("run_uuid") == run_uuid
            ):
                return record
        return None

    @classmethod
    def read_measurement_keys(
        cls, artifacts_path: Path, run_uuid: str
    ) -> Optional[tuple[str, str, str]]:
        """JD checkpoint type, checkpoint id and autometrics record id of ``run_uuid``,
        or ``None`` when any of them is missing in jd_data.json."""
        jd_data = cls._load_jd_data(artifacts_path / "jd_data.json")
        checkpoint = cls._checkpoint_type_and_id(jd_data)
        record = cls._find_run_record((jd_data or {}).get("history") or [], run_uuid)
        record_id = record.get("id") if record else None
        if checkpoint is None or not record_id:
            return None
        return checkpoint[0], checkpoint[1], str(record_id)

    @classmethod
    def read_metrics_files_mtime_ns(
        cls, artifacts_path: Path, evaluator_name: str
    ) -> int:
        jd_data = cls._load_jd_data(artifacts_path / "jd_data.json")
        return max(
            (
                (record.get("data") or {})
                .get("metrics_files_mtime_ns", {})
                .get(evaluator_name, 0)
                for record in (jd_data or {}).get("history") or []
                if record.get("action") == CheckpointHistoryAction.autometrics.value
            ),
            default=0,
        )

    @classmethod
    def add_metrics_cache_per_measurement(
        cls,
        artifacts_path: Path,
        run_uuid: str,
        evaluator_name: str,
        cache_paths: dict[str, str],
    ) -> None:
        jd_data_path = artifacts_path / "jd_data.json"
        jd_data = cls._load_jd_data(jd_data_path)
        if jd_data is None:
            return

        record = cls._find_run_record(jd_data.get("history") or [], run_uuid)
        if record is None:
            Logger.warning(
                f"No autometrics record for run {run_uuid} in {jd_data_path}; "
                f"skip recording '{evaluator_name}' cache paths"
            )
            return

        data = record.setdefault("data", {})
        data["updated_at"] = cls._utc_now()
        data.setdefault("metrics_cache_per_measurement", {}).setdefault(
            evaluator_name, {}
        ).update(cache_paths)
        cls._write_jd_data(jd_data, jd_data_path)
        Logger.info(f"Recorded '{evaluator_name}' cache paths in {jd_data_path}")

    @classmethod
    def _new_checkpoint_info(
        cls, hf_checkpoint_path: Path, workspace: Optional[str]
    ) -> dict:
        config = json.loads((hf_checkpoint_path / "config.json").read_text())
        quant_method = (config.get("quantization_config") or {}).get("quant_method")
        _, _, _, step = cls.parse_checkpoint_path(hf_checkpoint_path)
        return CheckpointHistoryInfoSchema(
            id=cls._new_uuid(),
            nfs_path=str(hf_checkpoint_path),
            created_at=cls._utc_now(),
            step=step,
            checkpoint_type="hf_fp8"
            if "fp8" in str(quant_method).lower()
            else "hf_bf16",
            checkpoint_creation_type="convertation",
            workspace=workspace,
        ).model_dump(mode="json")

    @classmethod
    def ensure_autometrics_record(
        cls,
        artifacts_path: Path,
        run_uuid: str,
        checkpoint_path: Optional[Path],
        hf_checkpoint_path: Optional[Path],
        run_name: Optional[str],
        job_id: Optional[str] = None,
        pam_config_path: Optional[Path] = None,
    ) -> None:
        try:
            jd_data_path = artifacts_path / "jd_data.json"
            jd_data = cls._load_jd_data(jd_data_path)
            workspace = os.environ.get("MLS_JOB_WORKSPACE_NAME") or os.environ.get(
                "MLS_NTB_WORKSPACE_NAME"
            )
            if jd_data is None:
                if hf_checkpoint_path is None:
                    Logger.info(
                        f"jd_data.json not found at {jd_data_path} and "
                        "hf_checkpoint_path is unknown; "
                        "skip recording autometrics record"
                    )
                    return
                jd_data = {
                    "checkpoint_info": cls._new_checkpoint_info(
                        hf_checkpoint_path, workspace
                    )
                }
                Logger.info(f"jd_data.json not found at {jd_data_path}; creating it")

            history = list(jd_data.get("history") or [])
            if cls._find_run_record(history, run_uuid) is not None:
                return

            now = cls._utc_now()
            checkpoint_info = jd_data.get("checkpoint_info") or {}
            checkpoint_id = checkpoint_info.get("id")
            checkpoint_type = checkpoint_info.get("checkpoint_type")
            record_id = cls._new_uuid()
            pam_config_s3_path = (
                cls.upload_pam_config(
                    pam_config_path,
                    hf_checkpoint_path,
                    str(checkpoint_type),
                    str(checkpoint_id),
                    record_id,
                )
                if pam_config_path and hf_checkpoint_path and checkpoint_type
                else None
            )
            history.append(
                CheckpointHistoryRecordCreateSchema(
                    id=record_id,
                    action=CheckpointHistoryAction.autometrics,
                    created_at=now,
                    checkpoint_id=checkpoint_id,
                    data={
                        "run_uuid": run_uuid,
                        "updated_at": now,
                        "checkpoint_path": str(checkpoint_path)
                        if checkpoint_path
                        else None,
                        "hf_checkpoint_path": (
                            str(hf_checkpoint_path) if hf_checkpoint_path else None
                        ),
                        "run_name": run_name,
                        "job": {
                            "name": job_id,
                            "workspace": workspace,
                        },
                        "pam_config_s3_path": pam_config_s3_path,
                        "pam_version": (Path(__file__).resolve().parents[3] / "VERSION")
                        .read_text()
                        .strip(),
                        "evaluators": [],
                        "metrics": {},
                    },
                ).model_dump(mode="json")
            )
            jd_data["history"] = history
            cls._write_jd_data(jd_data, jd_data_path)

            Logger.info(
                f"Added autometrics record for run {run_uuid} to {jd_data_path}"
            )
        except Exception as e:
            Logger.error(
                f"Error while adding an autometrics record to jd_data.json: {e}"
            )

    @classmethod
    def add_evaluator_metrics(
        cls,
        artifacts_path: Path,
        run_uuid: str,
        evaluator_name: str,
        metrics: dict[str, float],
        metrics_files_mtime_ns: Optional[int] = None,
    ) -> None:
        try:
            jd_data_path = artifacts_path / "jd_data.json"
            jd_data = cls._load_jd_data(jd_data_path)
            record = cls._find_run_record(
                (jd_data or {}).get("history") or [], run_uuid
            )
            if record is None:
                Logger.warning(
                    f"No autometrics record for run {run_uuid} in {jd_data_path}; "
                    f"skip recording '{evaluator_name}' metrics"
                )
                return

            data = record.setdefault("data", {})
            data["updated_at"] = cls._utc_now()
            data.setdefault("metrics", {}).setdefault(evaluator_name, {}).update(
                metrics
            )
            data.setdefault("evaluators", [])
            if evaluator_name not in data["evaluators"]:
                data["evaluators"].append(evaluator_name)
            if metrics_files_mtime_ns is not None:
                data.setdefault("metrics_files_mtime_ns", {})[evaluator_name] = (
                    metrics_files_mtime_ns
                )
            cls._write_jd_data(jd_data, jd_data_path)

            Logger.info(
                f"Updated autometrics record in {jd_data_path} with "
                f"{len(metrics)} '{evaluator_name}' metrics"
            )
        except Exception as e:
            Logger.error(f"Error while updating jd_data.json history: {e}")
