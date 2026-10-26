import json
import os
import re
import uuid
import requests
import socket

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

from logger import logger


_SHARD_LEAF_RE = re.compile(r"(tp_rank\d+|ep_rank\d+|[-_]rank\d+|^rank\d+$)")
JOB_DISPATCHER_BASE_URL = "https://jdmetrics.sberdevices.ru/api"
JOB_DISPATCHER_SLURM_BASE_URL = "https://dj.static.space/api"
CHECKPOINT_HISTORY_ENDPOINT = "/checkpoint_history/create"


_PRIMARY_ARG_KEYS = frozenset({
    "mappers",
    "input_checkpoint_path",
    "output_checkpoint_path",
    "model_type",
    "input_tp_size",
    "output_tp_size",
})


class CheckpointType(str, Enum):
    hf_fp8 = "hf_fp8"
    hf_bf16 = "hf_bf16"
    fsdp = "fsdp"
    gigafsdp = "gigafsdp"
    dist_ckpt = "dist_ckpt"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _new_uuid() -> str:
    return str(uuid.uuid4())


def _current_workspace() -> Optional[str]:
    return os.environ.get("MLS_JOB_WORKSPACE_NAME") or os.environ.get("MLS_NTB_WORKSPACE_NAME")


def _is_internet_available(timeout: int = 3) -> bool:
    try:
        with socket.create_connection(("1.1.1.1", 53), timeout=timeout):
            return True
    except OSError:
        return False


def get_jd_base_url() -> str:
    if _is_internet_available():
        return JOB_DISPATCHER_BASE_URL
    return JOB_DISPATCHER_SLURM_BASE_URL


@dataclass(kw_only=True)
class CheckpointInfo:
    id: str = field(default_factory=_new_uuid)
    nfs_path: str
    created_at: str = field(default_factory=_utc_now)
    step: Optional[int] = None
    checkpoint_type: Optional[str] = None
    checkpoint_creation_type: Optional[str] = None
    workspace: str
    parent_id: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "CheckpointInfo":
        data = data or {}
        return cls(**{f: data[f] for f in cls.__dataclass_fields__ if f in data})


@dataclass
class HistoryRecordData:
    mappers: list
    input_checkpoint_path: str
    output_checkpoint_path: str
    input_checkpoint_id: Optional[str]
    output_checkpoint_id: str
    model_type: Optional[str]
    input_tp_size: Optional[int]
    output_tp_size: Optional[int]
    rank: int
    world_size: int
    additional_params: dict = field(default_factory=dict)

    @classmethod
    def from_args(cls, args, input_id: Optional[str], output_id: str) -> "HistoryRecordData":
        return cls(
            mappers=list(args.mappers),
            input_checkpoint_path=str(args.input_checkpoint_path),
            output_checkpoint_path=str(args.output_checkpoint_path),
            input_checkpoint_id=input_id,
            output_checkpoint_id=output_id,
            model_type=getattr(args, "model_type", None),
            input_tp_size=getattr(args, "input_tp_size", None),
            output_tp_size=getattr(args, "output_tp_size", None),
            rank=int(os.environ.get("RANK", 0)),
            world_size=int(os.environ.get("WORLD_SIZE", 0)),
            additional_params={
                k: v for k, v in vars(args).items() if k not in _PRIMARY_ARG_KEYS
            },
        )


@dataclass(kw_only=True)
class HistoryRecord:
    id: str = field(default_factory=_new_uuid)
    action: str
    created_at: str = field(default_factory=_utc_now)
    checkpoint_id: Optional[str]
    data: dict


@dataclass
class JdData:
    checkpoint_info: CheckpointInfo = field(default_factory=CheckpointInfo)
    model_info: dict = field(default_factory=dict)
    ml_space_job: dict = field(default_factory=dict)
    history: list = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "JdData":
        data = data or {}
        return cls(
            checkpoint_info=CheckpointInfo.from_dict(data.get("checkpoint_info")),
            model_info=data.get("model_info") or {},
            ml_space_job=data.get("ml_space_job") or {},
            history=list(data.get("history") or []),
        )


def send_checkpoint_history(checkpoints: list) -> None:
    jd_base_url = get_jd_base_url()
    res = requests.post(
        f"{jd_base_url}{CHECKPOINT_HISTORY_ENDPOINT}",
        json=checkpoints,
        headers={"Authorization": "YlhIrKHpvuv0pBb6CK95kFv2AADp/79EUJ+njxqtSqk="},
        verify=False,
        timeout=10,
    )
    res.raise_for_status()
    logger.info(f"Sent {len(checkpoints)} checkpoint(s) history to job dispatcher")


def _get_artifacts_root(checkpoint_path) -> Path:
    if isinstance(checkpoint_path, (list, tuple)):
        checkpoint_path = checkpoint_path[0]
    p = Path(checkpoint_path)
    if p.is_symlink():
        p = p.resolve()
    if p.suffix == ".pt" or _SHARD_LEAF_RE.search(p.name):
        return p.parent
    return p


def find_input_jd_data_path(input_checkpoint_path: str) -> Path:
    return _get_artifacts_root(input_checkpoint_path) / "artifacts" / "jd_data.json"


def load_jd_data(jd_data_path: Path) -> Optional[JdData]:
    if jd_data_path.is_file():
        with open(jd_data_path, "r", encoding="utf-8") as f:
            return JdData.from_dict(json.load(f))
    return None


def _write_jd_data(jd_data: JdData, jd_data_path: Path) -> None:
    jd_data_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(asdict(jd_data), ensure_ascii=False, indent=4)
    with open(jd_data_path, "w", encoding="utf-8") as f:
        f.write(serialized)


_EXTRA_HF_OUTPUT_MAPPERS = frozenset({
    "map_safetensors_to_mtp_safetensors",
    "map_safetensors_graft_mtp_safetensors",
})


def _is_hf_output_mapper(mapper: str) -> bool:
    return "to_hf" in mapper or mapper in _EXTRA_HF_OUTPUT_MAPPERS


def _is_fp8_output(args) -> bool:
    output_path = args.output_checkpoint_path
    if isinstance(output_path, (list, tuple)):
        output_path = output_path[0] if output_path else None

    try:
        with open(Path(output_path) / "config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
    except (OSError, TypeError, ValueError):
        return False

    quantization_config = config.get("quantization_config")
    return isinstance(quantization_config, dict) and "fp8" in str(
        quantization_config.get("quant_method", "")
    ).lower()


def infer_output_checkpoint_type(args, state_dict=None, last_mapper=None) -> CheckpointType:
    if state_dict is not None:
        if isinstance(state_dict, (list, tuple)) and (getattr(args, "output_tp_size", 1) or 1) > 1:
            state_dict = state_dict[0] if state_dict else None
        if isinstance(state_dict, (list, tuple)):
            return CheckpointType.gigafsdp
        return CheckpointType.fsdp

    mapper = last_mapper or (args.mappers[-1] if getattr(args, "mappers", None) else "")
    if not _is_hf_output_mapper(mapper):
        return CheckpointType.fsdp
    return CheckpointType.hf_fp8 if _is_fp8_output(args) else CheckpointType.hf_bf16


def update_jd_data_history(args, state_dict=None, last_mapper: Optional[str] = None) -> None:
    if getattr(args, "global_rank", 0) != 0:
        return

    try:
        input_jd_data_path = find_input_jd_data_path(args.input_checkpoint_path)
        input_jd_data = load_jd_data(input_jd_data_path)

        if input_jd_data is None:
            logger.info(
                f"jd_data.json not found at {input_jd_data_path}; "
                "skip recording autometrics record"
            )
            return

        input_id = input_jd_data.checkpoint_info.id
        output_id = _new_uuid()

        record_data = HistoryRecordData.from_args(args, input_id, output_id)
        created_at = _utc_now()

        input_record = HistoryRecord(
            action="convertation",
            created_at=created_at,
            data=asdict(record_data),
            checkpoint_id=input_id,
        )

        input_jd_data.checkpoint_info.workspace = (
            _current_workspace() or input_jd_data.checkpoint_info.workspace
        )

        input_jd_data.history.append(asdict(input_record))
        _write_jd_data(input_jd_data, input_jd_data_path)

        output_checkpoint_info = CheckpointInfo(
            id=output_id,
            nfs_path=str(_get_artifacts_root(args.output_checkpoint_path)),
            created_at=created_at,
            step=input_jd_data.checkpoint_info.step,
            checkpoint_type=infer_output_checkpoint_type(
                args, state_dict=state_dict, last_mapper=last_mapper
            ).value,
            checkpoint_creation_type="convertation",
            workspace=input_jd_data.checkpoint_info.workspace,
            parent_id=input_id,
        )

        output_record = replace(
            input_record, id=_new_uuid(), checkpoint_id=output_id
        )

        output_jd_data = JdData(
            checkpoint_info=output_checkpoint_info,
            model_info=input_jd_data.model_info,
            ml_space_job=input_jd_data.ml_space_job,
            history=[asdict(output_record)],
        )

        output_jd_data_path = _get_artifacts_root(args.output_checkpoint_path) / "artifacts" / "jd_data.json"
        _write_jd_data(output_jd_data, output_jd_data_path)

        logger.info(
            f"Successfully wrote convertation record to jd_data.json "
            f"in {input_jd_data_path} and {output_jd_data_path}"
        )

        send_checkpoint_history([asdict(input_jd_data), asdict(output_jd_data)])
    except Exception as e:
        logger.error(f"Error occured while updating jd_data.json history: {e}")
