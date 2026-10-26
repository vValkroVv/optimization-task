import re
import os
import logging
import socket
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


class JDBase:

    DEFAULT_TIMEOUT = 60
    JD_TIMEOUT = 15
    DEFAULT_CREATION_TYPE = "pretrain"
    DEFAULT_SIZE = "lite"

    # Probe target used to detect outbound internet access, which decides what job-dispatcher url is used.
    # If this address is unreachable, JOB_DISPATCHER_BASE_URL is considered unavailable.
    INTERNET_CHECK_HOST = "1.1.1.1"
    INTERNET_CHECK_PORT = 53

    @staticmethod
    def _jd_headers() -> dict:
        return {"Authorization": os.environ.get("JOB_DISPATCHER_TOKEN")}

    def _is_internet_available(self, timeout: int = 3) -> bool:
        try:
            with socket.create_connection((self.INTERNET_CHECK_HOST, self.INTERNET_CHECK_PORT), timeout=timeout):
                return True
        except OSError:
            return False

    def get_jd_base_url(self) -> Optional[str]:
        if self._is_internet_available():
            return os.environ.get("JOB_DISPATCHER_BASE_URL")
        return os.environ.get("JOB_DISPATCHER_SLURM_BASE_URL")
        
    @staticmethod
    def _parse_checkpoint_path(checkpoint_path: Path) -> tuple[str, str, int, int]:
        for candidate in (checkpoint_path, *checkpoint_path.parents):
            name = candidate.name

            match = re.fullmatch(r"ep(\d+)_ba(\d+)", name)
            if match:
                epoch = int(match.group(1))
                step = int(match.group(2))
                model_dir = candidate.parent
                return str(model_dir), model_dir.name, epoch, step

            match = re.fullmatch(r"global_step_(\d+)", name)
            if match:
                epoch = 0
                step = int(match.group(1))
                model_dir = candidate.parent
                return str(model_dir), model_dir.name, epoch, step

        return str(checkpoint_path), checkpoint_path.name, 0, 0

    @staticmethod
    def _load_llm_foundry_config(artifact_folder):
        from omegaconf import OmegaConf as om

        config_path = os.path.join(artifact_folder, "llm_foundry_config.yaml")
        if not os.path.isfile(config_path):
            log.warning(f"No llm_foundry_config.yaml at {config_path}; falling back to default model info")
            return {}

        return om.load(config_path)

    def _resolve_model_config(self, artifact_folder) -> dict:
        cfg = self._load_llm_foundry_config(artifact_folder)

        jd_model_info = (cfg.get("callbacks") or {}).get("job_dispatcher_callback_web") or {}
        autometrics_model_info = (cfg.get("autometrics") or {}).get("model_info") or {}

        if not jd_model_info:
            source = "autometrics" if autometrics_model_info else "defaults"
            log.info(f"job_dispatcher_callback_web is not configured; taking model info from {source}")

        if cfg.get("dpo_mode"):
            creation_type = "dpo"
        elif cfg.get("sft_mode"):
            creation_type = "sft"
        else:
            creation_type = self.DEFAULT_CREATION_TYPE

        return {
            "creation_type": creation_type,
            "model_class": jd_model_info.get("model_class") or autometrics_model_info.get("model_class") or creation_type,
            "username": jd_model_info.get("username") or autometrics_model_info.get("username") or cfg.get("user"),
            "s3_dest": jd_model_info.get("s3_dest") or os.environ.get("JOB_DISPATCHER_S3_DEST"),
            "rclone_path": jd_model_info.get("rclone_path") or os.environ.get("JOB_DISPATCHER_RCLONE_PATH"),
            "rclone_config": jd_model_info.get("rclone_config") or os.environ.get("JOB_DISPATCHER_RCLONE_CONFIG"),
            "s3_path_marker": jd_model_info.get("s3_path_marker") or os.environ.get("JOB_DISPATCHER_S3_PATH_MARKER"),
            "size": jd_model_info.get("size") or autometrics_model_info.get("size"),
            "load_path": cfg.get("load_path"),
        }
