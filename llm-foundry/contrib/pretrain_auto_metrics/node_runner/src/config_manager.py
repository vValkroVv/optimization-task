import os
import socket
from pathlib import Path
from typing import Optional

from omegaconf import OmegaConf

from .config_model import NodeRunnerConfigModel


class ConfigManager:
    def __init__(self):
        self.config: Optional[NodeRunnerConfigModel] = None
        self.checkpoint_path = None
        self.is_cloud = False
        self.is_slurm = False
        self.job_id = None
        self.node_name = None
        self.artifacts_path = None
        self.global_nnodes = None
        self.global_dist_init_host = None
        self.global_node_rank = None
        self.gpus_per_node = None
        self.config_path = None

    @property
    def is_master(self) -> bool:
        return self.global_node_rank == 0

    def _get_environment_info(self) -> None:
        if self._is_cloud_env():
            self._init_cloud_env()
        elif self._is_slurm_env():
            self._init_slurm_env()
        elif self.config.runs_without_checkpoint:
            self._init_local_env()
        else:
            raise ValueError("Environment not supported")

    def _is_cloud_env(self) -> bool:
        return Path(self.config.autometrics.globals.dist_init_path).exists()

    def _is_slurm_env(self) -> bool:
        return all(
            os.getenv(k)
            for k in (
                "SLURM_JOB_NUM_NODES",
                "SLURM_LAUNCH_NODE_IPADDR",
                "SLURM_PROCID",
                "SLURM_JOB_ID",
                "JS_GPUS_PER_NODE",
            )
        )

    def _init_cloud_env(self) -> None:
        self.is_cloud = True
        host_name_parts = socket.gethostname().split("-")
        self.job_id, self.node_name = self._split_job_and_node(host_name_parts)
        self.global_dist_init_host = self._read_first_line(
            self.config.autometrics.globals.dist_init_path
        )
        self.global_node_rank = self._infer_cloud_rank(self.node_name, host_name_parts)
        if Path(self.config.autometrics.globals.worker_num_file_path).exists():
            self.global_nnodes = (
                int(
                    self._read_first_line(
                        self.config.autometrics.globals.worker_num_file_path
                    )
                )
                + 1
            )
        else:
            self.global_nnodes = 1
        self.gpus_per_node = int(os.getenv("MLS_JOB_N_GPU"))

    def _init_slurm_env(self) -> None:
        self.is_slurm = True
        self.job_id = os.getenv("SLURM_JOB_ID")
        self.global_node_rank = int(os.getenv("SLURM_PROCID"))
        self.node_name = (
            f"mpiworker-{self.global_node_rank - 1}"
            if self.global_node_rank != 0
            else f"mpimaster-{self.global_node_rank}"
        )
        self.global_nnodes = int(os.getenv("SLURM_JOB_NUM_NODES"))
        self.global_dist_init_host = os.getenv("SLURM_LAUNCH_NODE_IPADDR")
        self.gpus_per_node = int(os.getenv("JS_GPUS_PER_NODE"))

    def _init_local_env(self) -> None:
        self.job_id = "local"
        self.node_name = "mpimaster-0"
        self.global_nnodes = 1
        self.global_dist_init_host = "127.0.0.1"
        self.global_node_rank = 0
        self.gpus_per_node = 1

    @staticmethod
    def _read_first_line(path: str) -> str:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().splitlines()[0]

    @staticmethod
    def _split_job_and_node(parts: list[str]) -> tuple[str, str]:
        job_id = "-".join(parts[:-2])
        node_name = "-".join(parts[-2:])
        return (job_id, node_name)

    @staticmethod
    def _infer_cloud_rank(node_name: str, parts: list[str]) -> int:
        if node_name.startswith("mpiworker"):
            return int(parts[-1]) + 1
        return 0

    def _load_config(
        self, config_path: Optional[Path], checkpoint_path: Optional[Path]
    ) -> None:
        if config_path is None:
            config_path = Path(checkpoint_path, "artifacts", "pam_config.yaml")
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        self.config = NodeRunnerConfigModel.load(config_path)
        self.config_path = config_path

    def _set_paths(self, checkpoint_path: Optional[Path]) -> None:
        if self.config.is_external_api_mode:
            model_name = self.config.autometrics.inference.model
            repo_root = Path(__file__).resolve().parent.parent.parent
            self.checkpoint_path = checkpoint_path or repo_root / model_name
            self.artifacts_path = self.checkpoint_path
            os.makedirs(self.artifacts_path, exist_ok=True)
            OmegaConf.save(
                config=self.config.model_dump(
                    by_alias=True, exclude_none=True, mode="json", round_trip=True
                ),
                f=self.artifacts_path / "pam_config.yaml",
            )
        else:
            if checkpoint_path is None:
                raise ValueError(
                    "checkpoint_path is required unless mode is external_api"
                )
            if not checkpoint_path.exists():
                raise FileNotFoundError(
                    f"Checkpoint directory not found: {checkpoint_path}"
                )
            self.checkpoint_path = checkpoint_path
            self.artifacts_path = Path(self.checkpoint_path, "artifacts")
            os.makedirs(self.artifacts_path, exist_ok=True)

    def _decode_jinja_to_ram(self) -> None:
        if self.config is None:
            return
        self.config.autometrics.obm_proxy.decode_chat_template_to_ram(
            is_slurm=self.is_slurm,
        )

    def _decode_default_devsystem_kwarg(self) -> None:
        if self.config is None:
            return
        self.config.autometrics.obm_proxy.decode_default_devsystem(
            is_slurm=self.is_slurm,
        )

    def setup(
        self,
        checkpoint_path: Path | None,
        config_path: Path | None = None,
    ):
        self._load_config(config_path, checkpoint_path)
        self._set_paths(checkpoint_path)
        self._get_environment_info()
        self._decode_jinja_to_ram()
        self._decode_default_devsystem_kwarg()
