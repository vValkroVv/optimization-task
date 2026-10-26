from __future__ import annotations

import contextlib
import getpass
import io
import logging
import os
import re
import subprocess
import time
from abc import ABC, abstractmethod
from base64 import b64decode, b64encode
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional, Protocol, Union, runtime_checkable

from omegaconf import DictConfig, ListConfig, OmegaConf

from job_submitter.asserts.base import BaseJobAssertor
from job_submitter.core.backends import CloudBackend, SlurmBackend
from job_submitter.core.job_config import JobTemplate
from job_submitter.core.job_templates.region_configs import (
    CLOUD_REGIONS,
    DEFAULT_REGION_ENVS,
    DEFAULT_SLURM_REGION_PARAMS,
    SLURM_REGIONS,
    SLURM_TEAM_QOS_REGIONS,
)
from job_submitter.core.tracking import auto_track, extract_logs_path_from_file
from job_submitter.jobs.image_registry import LLMF_BASE_IMAGES

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class JobMonitorResult:
    final_status: str
    logs: Optional[str] = None


@runtime_checkable
class SubmittedJobHandle(Protocol):
    @property
    def job_name(self) -> str: ...

    def status(self) -> str: ...

    def logs(self, tail: int = 0, verbose: bool = True) -> Any: ...


@dataclass(slots=True, kw_only=True)
class BaseJob(ABC, BaseJobAssertor):
    user_login: str
    exp_name: str
    # ⚠️ Auto-track is OFF for generic jobs.
    # Enable it only when the job writes a tracker-readable config/log path.
    enable_auto_track: bool = False
    auto_track_users: list[str] = field(default_factory=list)
    enable_auto_share: bool = False

    n_gpus: int = 8  # number of GPUs per node
    gpu_type: Optional[str] = None
    n_workers: int = 1
    queue_name: Optional[str] = None
    priority_class: Optional[str] = None  # "low" | "medium" | "high"
    base_image: Optional[str] = None
    processes_per_worker: int = 1
    max_retry: Optional[int] = None
    submission_type: Optional[str] = None  # "cloud" | "slurm"
    region: Optional[str] = None  # Cloud: SR004 | SR008; Slurm: o1 | o2 | o5
    workspace_name: Optional[str] = None  # cloud only
    extra_env: Dict[str, str] = field(default_factory=dict)
    job_logdir: Optional[str] = None  # slurm only
    slurm_logdir: Optional[str] = None  # slurm only
    script_type: str = "binary"
    slurm_submit_type: str = "sbatch"  # "sbatch" | "api"
    cpus_per_task: Optional[int] = None
    memory: Optional[str] = None
    extra_mounts: Optional[str] = None
    turn_off_first_srun: Optional[bool] = None
    internet: bool = True
    dry_run: bool = False
    nodelist: Optional[str] = None
    exclude_nodes: Optional[str] = None
    nice: Optional[int] = None
    debug: bool = False
    cpu_only: bool = field(default=False, init=False)

    slurm_jwt_token: Optional[str] = field(default=None, repr=False, init=False)
    slurmrestd_urls: list[str] = field(default_factory=list, repr=False, init=False)
    instance_type: Optional[str] = field(default=None, repr=False, init=False)
    allocation_name: Optional[str] = field(default=None, repr=False, init=False)
    _last_submitted_job: Optional[Any] = field(default=None, repr=False, init=False)

    # Attributes scanned (in order) by the generic ``_tracking_config_path`` resolver.
    _TRACKING_CONFIG_ATTRS: ClassVar[tuple[str, ...]] = (
        "path_to_config",
        "pam_config_path",
        "runtime_env_config_path",
    )
    BASE_IMAGES: ClassVar[dict[str, str]] = LLMF_BASE_IMAGES
    _REGION_TRANSLATION: ClassVar[dict[int, str]] = str.maketrans(
        {"\u043e": "o", "\u041e": "O"}
    )
    _SLURM_REGION_VALUES: ClassVar[set[str]] = {
        SLURM_REGIONS.o1,
        SLURM_REGIONS.o2,
        SLURM_REGIONS.o5,
    }
    _CLOUD_REGION_VALUES: ClassVar[set[str]] = {
        CLOUD_REGIONS.sr004,
        CLOUD_REGIONS.sr006,
        CLOUD_REGIONS.sr008,
    }
    DEFAULT_HOMES: ClassVar[dict[str, str]] = {
        "o1": "/mnt/nfs/07-share/",
        "o2": "/mnt/nfs/07-share/",
        "o5": "/mnt/nfs/07-share/",
        "SR004": "/home/jovyan/",
        "SR006": "/home/jovyan/",
        "SR008": "/home/jovyan/",
    }
    DEFAULT_TRANSFER_DIRS: ClassVar[dict[str, str]] = {
        "o1": f"{DEFAULT_HOMES['o1']}transfer2cloud/o1/",
        "o2": f"{DEFAULT_HOMES['o2']}transfer2cloud/o2/",
        "o5": f"{DEFAULT_HOMES['o5']}transfer2cloud/o5/",
    }

    @staticmethod
    def _get_instance_type(region: str, n_gpus: int) -> str:
        if region == "SR004":
            return f"a100.{n_gpus}gpu"
        elif region == "SR006":
            return f"a100.{n_gpus}gpu.{8 * n_gpus}C.{243 * n_gpus}G"
        elif region == "SR008":
            return f"a100plus.{n_gpus}gpu.80vG.{12 * n_gpus}C.{182 * n_gpus}G"
        raise ValueError(f"Region '{region}' not supported")

    @classmethod
    def _normalize_region(cls, region: Optional[str]) -> Optional[str]:
        if region is None:
            return None

        normalized = str(region).strip().translate(cls._REGION_TRANSLATION)
        if not normalized:
            return None

        slurm_region = normalized.lower()
        if slurm_region in cls._SLURM_REGION_VALUES:
            return slurm_region

        cloud_region = normalized.upper()
        if cloud_region in cls._CLOUD_REGION_VALUES:
            return cloud_region

        return normalized

    @staticmethod
    def _normalize_submission_type(
        submission_type: Optional[str],
    ) -> Optional[str]:
        if submission_type is None:
            return None
        normalized = str(submission_type).strip().lower()
        return normalized or None

    @classmethod
    def _infer_submission_type_for_region(cls, region: str) -> Optional[str]:
        if region in cls._SLURM_REGION_VALUES:
            return "slurm"
        if region in cls._CLOUD_REGION_VALUES:
            return "cloud"
        return None

    @staticmethod
    def _parse_url(raw_config: str) -> str:
        url_regex = re.compile(r"SlurmctldHost\[(\d+)\]\s*=\s*(\S+)", re.IGNORECASE)
        return [
            f"http://{match.group(2)}:6820" for match in url_regex.finditer(raw_config)
        ]

    @staticmethod
    def _parse_cluster_name(raw_config: str) -> str:
        cluster_name_regex = re.compile(r"ClusterName\s*=\s*(\S+)", re.IGNORECASE)
        found = cluster_name_regex.search(raw_config)
        if found:
            return found.group(1).replace("-", "").replace("_", "")
        raise ValueError("Cluster name not found")

    @staticmethod
    def _run_subprocess(command: str) -> str:
        result = subprocess.run(
            command,
            shell=True,
            text=True,
            capture_output=True,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        raise ValueError(f"Failed to run subprocess: {result.stderr}")

    @property
    def llm_f_root(self) -> Path:
        return Path(__file__).absolute().parent.parent.parent.parent.parent

    @classmethod
    def from_spec(cls, spec_path: Union[str, Path]) -> "BaseJob":
        raw = OmegaConf.load(spec_path)
        merged = OmegaConf.merge(OmegaConf.structured(cls), raw)
        validated = OmegaConf.to_container(merged, resolve=True)
        return cls(**{name: value for name, value in validated.items() if name in raw})

    @abstractmethod
    def _script(self) -> str: ...

    def _before_submit(self) -> None:
        """Optional pre-submission hook; subclasses override to add side effects."""

    def _tracking_config_path(self, _cfg: JobTemplate) -> Optional[str]:
        """Resolve the tracking config path from well-known config attributes.

        Acts as a safety net: any job that exposes one of ``_TRACKING_CONFIG_ATTRS``
        is auto-tracked without a bespoke override. Subclasses override only when they
        need a custom source or fallback (e.g. VERL's script path).
        """
        for attr in self._TRACKING_CONFIG_ATTRS:
            value = getattr(self, attr, None)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
        return None

    def _tracking_merged_config(self, cfg: JobTemplate) -> Optional[dict[str, Any]]:
        return None

    def _tracking_script_path(self, _cfg: JobTemplate) -> Optional[str]:
        return None

    def _resolve_tb_log_dir(self) -> Optional[str]:
        config_path = self._tracking_config_path(JobTemplate())
        if not config_path:
            return None
        return extract_logs_path_from_file(config_path)

    def _auto_track_after_submit(self, cfg: JobTemplate) -> None:
        if not self.enable_auto_track:
            if self.enable_auto_share:
                logger.warning(
                    "Auto-share is ignored for exp='%s' because enable_auto_track=False.",
                    self.exp_name,
                )
            return
        # Auto-tracking must never break a successful submission: the job is already
        # submitted by the time we get here, so swallow any unexpected failure.
        try:
            self._run_auto_track(cfg)
        except Exception:
            logger.exception(
                "Auto-tracking raised unexpectedly for exp='%s' (non-blocking; "
                "job submission already succeeded).",
                self.exp_name,
            )

    def _run_auto_track(self, cfg: JobTemplate) -> None:
        config_path = self._tracking_config_path(cfg)
        if not config_path:
            logger.warning(
                "Auto-tracking is enabled for exp='%s' but no tracking config was "
                "found (%s). Nothing was registered.",
                self.exp_name,
                ", ".join(self._TRACKING_CONFIG_ATTRS),
            )
            return

        region = cfg.region or self.region
        if not region:
            return

        submission_type = cfg.submission_type or self.submission_type
        job_id = (
            self._last_submitted_job.job_name
            if self._last_submitted_job is not None
            else None
        )
        script_path = self._tracking_script_path(cfg)
        tracked = auto_track(
            region=region,
            user_login=self.user_login,
            auto_track_users=self.auto_track_users,
            enable_auto_share=self.enable_auto_share,
            config_path=config_path,
            merged_config=self._tracking_merged_config(cfg),
            env=cfg.env_variables,
            submission_type=submission_type,
            job_id=job_id,
            script_path=str(script_path) if script_path else None,
            job_title=cfg.job_desc,
        )
        if not tracked:
            logger.warning(
                "Auto-tracking failed or skipped for exp='%s' region='%s' config='%s'. "
                "Job submission succeeded.",
                self.exp_name,
                region,
                config_path,
            )

    def _external_code_libraries(self) -> list[str]:
        return []

    def _build_env(self) -> Dict[str, str]:
        external_code_libs = self._external_code_libraries()
        env: Dict[str, str] = {
            "PYTHONPATH": ":".join(external_code_libs) if external_code_libs else "",
            "NFS_HOME": self.DEFAULT_HOMES[self.region],
            "PAM_DG_CACHE_DIR": self.DEFAULT_HOMES[self.region]
            + "models/deepgemm_caches/",
            "PAM_MODELS_PATH": self.DEFAULT_HOMES[self.region] + "models/",
            "DG_JIT_SHARED_CACHE_DIR": self.DEFAULT_HOMES[self.region]
            + "deepgemm_shared_cache/",
            "JOB_SUBMITTER_NNODES": str(self.n_workers),
        }
        if self.queue_name is not None and self.queue_name != "None":
            env["PAM_QUEUE_NAME"] = self.queue_name
        if self.priority_class is not None and self.priority_class != "None":
            env["PAM_PRIORITY_CLASS"] = self.priority_class
        env.update(
            {
                "JS_GPUS_PER_NODE": str(self.n_gpus),
                "JS_NNODES": str(self.n_workers),
            }
        )
        try:
            resolved_base_image = self._resolve_base_image()
        except Exception:
            resolved_base_image = None
        if resolved_base_image:
            env["JS_BASE_IMAGE"] = resolved_base_image

        if self.submission_type == "slurm":
            env.update(
                {
                    "SLURM_JWT_TOKEN": self.slurm_jwt_token
                    or os.getenv("SLURM_JWT_TOKEN"),
                    "SLURMRESTD_URLS": "||".join(self.slurmrestd_urls)
                    or os.getenv("SLURMRESTD_URLS"),
                    "HF_DATASETS_OFFLINE": "1",
                    "HF_HUB_OFFLINE": "1",
                    "HF_EVALUATE_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                    "HF_HUB_ETAG_TIMEOUT": "1",
                    "HF_HUB_DOWNLOAD_TIMEOUT": "5",
                    "NNODES": str(self.n_workers),
                    "JD_SYNC_DIR": str(
                        Path(self.DEFAULT_TRANSFER_DIRS[self.region]) / "jd_sync"
                    ),
                }
            )
            if os.getenv("SLURM_CLUSTER_NAME"):
                env["SLURM_CLUSTER_NAME"] = os.getenv("SLURM_CLUSTER_NAME")
        else:
            workspace_name = self._resolve_workspace_name() or ""
            env.update(
                {
                    "GWAPI_ADDR": os.getenv("GWAPI_ADDR", ""),
                    "GWAPI_V2_URL": os.getenv("GWAPI_V2_URL", ""),
                    "GWAPI_KEY": os.getenv("GWAPI_KEY", ""),
                    "CLUSTER_KEY": os.getenv("CLUSTER_KEY", ""),
                    "NAMESPACE": os.getenv("NAMESPACE", ""),
                    "NB_PREFIX": os.getenv("NB_PREFIX", ""),
                    "WORKSPACE_ID": os.getenv("WORKSPACE_ID", ""),
                    "MLS_NTB_WORKSPACE_NAME": workspace_name,
                }
            )
            if not self.internet:
                env["HF_HUB_OFFLINE"] = "1"
                env["HF_DATASETS_OFFLINE"] = "1"
                env["HF_EVALUATE_OFFLINE"] = "1"
                env["TRANSFORMERS_OFFLINE"] = "1"
                env["HF_HUB_ETAG_TIMEOUT"] = "1"
                env["HF_HUB_DOWNLOAD_TIMEOUT"] = "5"
        env.update(DEFAULT_REGION_ENVS.get(self.region, {}))
        try:
            tb_log_dir = self._resolve_tb_log_dir()
        except Exception:
            logger.exception(
                "TensorBoard log path resolution failed for exp='%s' (non-blocking).",
                self.exp_name,
            )
        else:
            if tb_log_dir:
                env["TB_LOG_DIR"] = tb_log_dir
        env.update(self.extra_env)
        if self.submission_type == "cloud":
            env["MLS_NTB_WORKSPACE_NAME"] = workspace_name
        return env

    def _resolve_instance_type(self) -> Optional[str]:
        if self.submission_type == "cloud":
            return self.instance_type or self._get_instance_type(
                self.region, self.n_gpus
            )
        return self.instance_type

    def _get_current_region(
        self,
    ) -> tuple[Optional[str], Optional[str]]:
        slurm_region = self._normalize_region(os.getenv("SLURM_CLUSTER_NAME"))
        if slurm_region:
            return slurm_region, "slurm"
        cloud_region = self._normalize_region(os.getenv("CLUSTER_KEY"))
        if cloud_region:
            return cloud_region, "cloud"
        cluster_name = self._normalize_region(self._get_slurm_cluster_name())
        return cluster_name, "slurm"

    def _resolve_job_logdir(self) -> Optional[str]:
        if self.submission_type == "slurm":
            return self.job_logdir or DEFAULT_SLURM_REGION_PARAMS.get(
                self.region, {}
            ).get("job_logdir")
        return self.job_logdir

    def _resolve_slurm_logdir(self) -> Optional[str]:
        if self.submission_type == "slurm":
            return self.slurm_logdir or DEFAULT_SLURM_REGION_PARAMS.get(
                self.region, {}
            ).get("slurm_logdir")
        return self.slurm_logdir

    def _resolve_base_image(self) -> Optional[str]:
        return self.base_image or self.BASE_IMAGES[self.region]

    def _resolve_region(self) -> tuple[Optional[str], Optional[str]]:
        explicit_region = self._normalize_region(self.region)
        explicit_submission_type = self._normalize_submission_type(self.submission_type)

        if explicit_region:
            return explicit_region, (
                explicit_submission_type
                or self._infer_submission_type_for_region(explicit_region)
                or self._get_current_region()[1]
            )

        region, detected_submission_type = self._get_current_region()
        return region, explicit_submission_type or detected_submission_type

    def _resolve_workspace_name(self) -> Optional[str]:
        if self.submission_type != "cloud":
            return self.workspace_name

        workspace_name = (
            self.workspace_name
            or self.extra_env.get("MLS_NTB_WORKSPACE_NAME")
            or os.getenv("MLS_NTB_WORKSPACE_NAME")
        )
        if not workspace_name or not workspace_name.strip():
            raise ValueError(
                "Cloud workspace is not specified. Pass workspace_name or set "
                "MLS_NTB_WORKSPACE_NAME"
            )
        return workspace_name.strip()

    def _resolve_slurm_jwt_token(self) -> Optional[str]:
        if self.submission_type == "slurm":
            self.slurm_jwt_token = self.slurm_jwt_token or os.getenv("SLURM_JWT_TOKEN")
            if self.slurm_jwt_token:
                return self.slurm_jwt_token
            result = self._run_subprocess(  # generates jwt token for 56 days
                "scontrol token lifespan=4838400",
            )
            self.slurm_jwt_token = result.split("=")[1].strip()
            return self.slurm_jwt_token

    def _resolve_slurmrestd_urls(self) -> list[str]:
        if self.submission_type == "slurm":
            self.slurmrestd_urls = self.slurmrestd_urls or (
                os.getenv("SLURMRESTD_URLS").split("||")
                if os.getenv("SLURMRESTD_URLS")
                else []
            )
            if self.slurmrestd_urls:
                return self.slurmrestd_urls
            result = self._run_subprocess("scontrol show config")
            self.slurmrestd_urls = self._parse_url(result)
            return self.slurmrestd_urls

    def _get_slurm_cluster_name(
        self,
    ) -> Optional[str]:
        result = self._run_subprocess("scontrol show config")
        return self._parse_cluster_name(result)

    def _uses_team_qos(self) -> bool:
        return self.region in SLURM_TEAM_QOS_REGIONS

    def _resolve_slurm_team(self) -> str:
        if not self.queue_name:
            self.queue_name = self._run_subprocess(
                f"sacctmgr -nP show user {getpass.getuser()} format=defaultaccount"
            )
        return self.queue_name

    def _resolve_partition(self) -> str:
        if self.priority_class == "leads":
            return "leads"
        if self._uses_team_qos():
            if self.priority_class == "low":
                return "main"
            return self._resolve_slurm_team()
        if self.priority_class == "high":
            return "main"
        return "middle"

    def _resolve_slurm_qos(self) -> Optional[str]:
        if not self._uses_team_qos():
            return self.queue_name
        if self.priority_class == "leads":
            return "leads"
        if self.priority_class == "low":
            return "normal"
        team = self._resolve_slurm_team()
        if self.priority_class == "urgent":
            return f"{team}_high"
        if self.priority_class == "high":
            return team
        if self.priority_class == "medium":
            return f"{team}_low"
        return None

    def _resolve_slurm_account(self) -> Optional[str]:
        if self.submission_type != "slurm":
            return None
        if self.priority_class == "leads":
            return "leads"
        if self._uses_team_qos():
            return self._resolve_slurm_team()
        if self.priority_class == "high":
            return self.queue_name
        return None

    def _resolve_gpu_type(self) -> Optional[str]:
        if self.gpu_type:
            return self.gpu_type
        if self.region == "o2":
            return "h100"
        return None

    def _resolve_processes_per_worker(self) -> int:
        if self.submission_type == "slurm":
            return 1
        return self.processes_per_worker

    def _is_tts_workspace(self) -> bool:
        return (self._resolve_workspace_name() or "").lower() == "tts"

    def _resolve_allocation_name(self) -> Optional[str]:
        if self.region == "SR008":
            if self.queue_name == "giga-audio":
                return "alloc-giga-audio-h100-sr008"
            if self._is_tts_workspace():
                return "alloc-sberdevices-gigatts-h100-sr008"
            return "alloc-sberdevices-giga-vision-h100-sr008"
        return None

    def _anonimize_config(self, pam_config_path: Optional[Union[str, Path]]) -> None:
        if not self.submission_type == "slurm":
            return
        if not pam_config_path:
            return
        pam_config_path = Path(pam_config_path)
        loaded_config = OmegaConf.load(pam_config_path)
        if "autometrics" not in loaded_config:
            return
        autometrics_config = loaded_config["autometrics"]
        made_some_changes = False

        obm_proxy_config = autometrics_config.get("obm_proxy", {})
        if isinstance(obm_proxy_config, (dict, DictConfig)):
            chat_template_key = None
            if "chat-template" in obm_proxy_config:
                chat_template_key = "chat-template"
            elif "chat_template" in obm_proxy_config:
                chat_template_key = "chat_template"

            if chat_template_key:
                chat_template = obm_proxy_config[chat_template_key]
                if Path(chat_template).exists():
                    with open(chat_template, "rb") as f:
                        chat_template_content = f.read()
                    new_chat_template = f"{chat_template}.base64"
                    if not self._is_encoded_in_base64(chat_template_content):
                        with open(new_chat_template, "wb") as f:
                            f.write(b64encode(chat_template_content))
                        obm_proxy_config[chat_template_key] = new_chat_template
                        print(
                            f"[WARNING] OBM proxy chat template {chat_template} is not encoded in base64, encoded it to {new_chat_template} and replaced in config"
                        )
                        made_some_changes = True
                else:
                    print(
                        f"[WARNING] OBM proxy chat template {chat_template} not found, skipping"
                    )
                    return

            chat_template_kwargs = obm_proxy_config.get(
                "chat-template-kwargs",
                obm_proxy_config.get("chat_template_kwargs", {}),
            )
            if isinstance(chat_template_kwargs, (dict, DictConfig)):
                default_devsystem = chat_template_kwargs.get("default_devsystem")
                if default_devsystem and not self._is_encoded_in_base64(
                    default_devsystem
                ):
                    encoded_default_devsystem = b64encode(
                        default_devsystem.encode()
                    ).decode()
                    chat_template_kwargs["default_devsystem"] = (
                        encoded_default_devsystem
                    )
                    print(
                        f"[WARNING] Default devsystem is not encoded in base64, encoded it to {encoded_default_devsystem} and replaced in config"
                    )
                    made_some_changes = True

        evaluators_config = autometrics_config.get("evaluators", {})
        if isinstance(evaluators_config, (dict, DictConfig)):
            obm_all_config = evaluators_config.get("obm", {}).get("all", {})
            if isinstance(obm_all_config, (dict, DictConfig)):
                forbidden_group = b64decode("U0JFUg==").decode().lower()
                obm_group_changed = False

                obm_groups = obm_all_config.get("groups")
                if isinstance(obm_groups, (list, ListConfig)):
                    replaced_groups = []
                    for group in obm_groups:
                        if isinstance(group, str):
                            replaced_groups.append(
                                "papaya" if group == forbidden_group else group
                            )
                        else:
                            replaced_groups.append(group)
                    if replaced_groups != obm_groups:
                        obm_all_config["groups"] = replaced_groups
                        obm_group_changed = True

                if obm_group_changed:
                    print("[WARNING] OBM groups replaced: `sber` -> `papaya`")
                    made_some_changes = True

        if made_some_changes:
            OmegaConf.save(loaded_config, pam_config_path)
            print(
                f"[INFO] Config {pam_config_path} replaced in-place with PAPAYED version"
            )

    def _resolve_queue_name(self) -> Optional[str]:
        if self.region == "SR008" and self._is_tts_workspace():
            self.queue_name = None
            return None
        if self.queue_name:
            return self.queue_name
        elif env_queue_name := os.getenv("PAM_QUEUE_NAME"):
            self.queue_name = env_queue_name
            return env_queue_name
        elif self.submission_type == "slurm":
            self.queue_name = (
                self._resolve_slurm_team() if self._uses_team_qos() else None
            )
        elif self.region == "SR006":
            self.queue_name = None
        elif self.region == "SR004":
            self.queue_name = None
        elif self.region == "SR008":
            self.queue_name = "shared"
        return self.queue_name

    def _resolve_priority_class(self) -> Optional[str]:
        if self.region == "SR008" and self._is_tts_workspace():
            self.priority_class = None
            return None
        env_priority_class = os.getenv("PAM_PRIORITY_CLASS")
        if self.priority_class is None and env_priority_class:
            self.priority_class = env_priority_class
            return self.priority_class
        if self.queue_name == "shared":
            self.priority_class = None
            return None
        if self.priority_class is None:
            self.priority_class = "medium"
        return self.priority_class

    def _resolve_n_gpus(self) -> int:
        return self.n_gpus

    def _resolve_job_desc(self) -> str:
        return f"{self.user_login} | {self.exp_name}"

    def _resolve_turn_off_first_srun(self) -> bool:
        if self.turn_off_first_srun is not None:
            return self.turn_off_first_srun
        return False

    def to_config(self) -> JobTemplate:
        self.region, self.submission_type = self._resolve_region()
        self.workspace_name = self._resolve_workspace_name()

        config = JobTemplate(
            submission_type=self.submission_type,
            user_login=self.user_login,
            exp_name=self.exp_name,
            nodelist=self.nodelist,
            exclude_nodes=self.exclude_nodes,
            nice=self.nice,
            job_desc=self._resolve_job_desc(),
            internet=self.internet,
            cpus_per_task=self.cpus_per_task,
            memory=self.memory,
            extra_mounts=self.extra_mounts,
            turn_off_first_srun=self._resolve_turn_off_first_srun(),
            queue_name=self._resolve_queue_name(),
            allocation_name=self._resolve_allocation_name(),
            base_image=self._resolve_base_image(),
            priority_class=self._resolve_priority_class(),
            script=self._script(),
            partition=self._resolve_partition(),
            qos=self._resolve_slurm_qos(),
            account=self._resolve_slurm_account(),
            n_workers=self.n_workers,
            processes_per_worker=self._resolve_processes_per_worker(),
            script_type=self.script_type,
            region=self.region,
            max_retry=self.max_retry,
            gpu_num=self._resolve_n_gpus(),
            cpu_only=self.cpu_only,
            gpu_type=self._resolve_gpu_type(),
            instance_type=self._resolve_instance_type(),
            slurm_jwt_token=self._resolve_slurm_jwt_token(),
            slurmrestd_urls=self._resolve_slurmrestd_urls(),
            slurm_submit_type=self.slurm_submit_type,
            job_logdir=self._resolve_job_logdir(),
            slurm_logdir=self._resolve_slurm_logdir(),
            env_variables=self._build_env(),
        )
        config.comment = config.env_variables.get("TB_LOG_DIR")
        return config

    @staticmethod
    def _extract_status_name(status: str) -> str:
        prefix = "Job status="
        if status.startswith(prefix):
            return status[len(prefix) :]
        return status

    @staticmethod
    def _capture_logs(
        job: SubmittedJobHandle,
        logs_tail: int,
        logs_verbose: bool,
        echo_to_stdout: bool = True,
    ) -> str:
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        with (
            contextlib.redirect_stdout(stdout_capture),
            contextlib.redirect_stderr(stderr_capture),
        ):
            logs_result = job.logs(tail=logs_tail, verbose=logs_verbose)
        captured_text = stdout_capture.getvalue() + stderr_capture.getvalue()
        if captured_text and echo_to_stdout:
            print(captured_text, end="")
        returned_text = (
            logs_result
            if isinstance(logs_result, str)
            else (str(logs_result) if logs_result is not None else "")
        )
        if returned_text and captured_text:
            if returned_text in captured_text:
                return captured_text
            if captured_text in returned_text:
                return returned_text
            joiner = (
                ""
                if returned_text.endswith("\n") or captured_text.startswith("\n")
                else "\n"
            )
            return returned_text + joiner + captured_text
        return returned_text or captured_text

    def _submit_config(self, cfg: JobTemplate) -> BaseJob:
        if self.dry_run:
            print(f"Dry run: {cfg}")
            self._last_submitted_job = None
            return self

        if cfg.submission_type == "cloud":
            backend = CloudBackend()
            backend.submit(cfg)
            if not isinstance(backend.last_submitted_job, SubmittedJobHandle):
                raise TypeError("Cloud backend did not return monitorable job handle")
            self._last_submitted_job = backend.last_submitted_job
            self._auto_track_after_submit(cfg)
            return self

        if cfg.submission_type == "slurm":
            backend = SlurmBackend()
            backend.submit(cfg)
            self._last_submitted_job = backend.last_submitted_job
            self._auto_track_after_submit(cfg)
            return self

        raise ValueError(f"Unknown submission_type={cfg.submission_type}")

    def monitor(
        self,
        initial_wait_seconds: int = 15,
        poll_interval_seconds: int = 30,
        logs_tail: int = 0,
        logs_verbose: bool = True,
        capture_logs: bool = True,
        active_statuses: Optional[set[str]] = None,
        logs_retry_attempts: int = 2,
        logs_retry_interval_seconds: int = 2,
        post_completion_wait_seconds: int = 10,
    ) -> JobMonitorResult:
        assert self.submission_type != "slurm", (
            "Monitoring for slurm is not supported yet"
        )

        monitor_job = self._last_submitted_job
        if monitor_job is None:
            raise ValueError("No submitted cloud job to monitor. Call submit() first")

        if not isinstance(monitor_job, SubmittedJobHandle):
            raise TypeError(
                "Submitted job does not support monitor(): missing status/logs methods"
            )

        normalized_active_statuses = active_statuses or {"Pending", "Running"}
        if initial_wait_seconds > 0:
            print("Wait before requesting logs")
            time.sleep(initial_wait_seconds)

        while True:
            try:
                current_status = monitor_job.status()
            except Exception as exc:
                current_status = f"status request failed: {exc}"
            if not current_status.startswith("Job status="):
                print(f"Job status unavailable, retrying: {current_status}")
            elif (
                self._extract_status_name(current_status)
                not in normalized_active_statuses
            ):
                break
            time.sleep(poll_interval_seconds)

        logs_text: Optional[str] = None
        if capture_logs:
            print("Waiting for logs flush after completion...")
            time.sleep(post_completion_wait_seconds)
            print("Getting job logs...")
            attempts = max(1, logs_retry_attempts)
            current_logs: str = ""
            for attempt in range(attempts):
                current_logs = self._capture_logs(
                    job=monitor_job,
                    logs_tail=logs_tail,
                    logs_verbose=logs_verbose,
                    echo_to_stdout=False,
                )
                logs_text = current_logs
                if attempt < attempts - 1:
                    time.sleep(logs_retry_interval_seconds)
            if logs_text:
                print(logs_text, end="")
        else:
            print("Getting job logs...")
            monitor_job.logs(tail=logs_tail, verbose=logs_verbose)

        try:
            final_status = monitor_job.status()
        except Exception:
            final_status = ""
        if not final_status.startswith("Job status="):
            final_status = current_status
        return JobMonitorResult(final_status=final_status, logs=logs_text)

    def submit(self) -> BaseJob:
        cfg = self.to_config()
        self.assert_job(self, cfg)
        self._before_submit()
        return self._submit_config(cfg)
