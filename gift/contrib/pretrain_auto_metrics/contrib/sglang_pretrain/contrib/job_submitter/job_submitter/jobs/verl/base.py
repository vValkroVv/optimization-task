import os
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Optional

from job_submitter.jobs.base import BaseJob
from job_submitter.jobs.image_registry import VERL_BASE_IMAGES


@dataclass(slots=True, kw_only=True)
class RayBaseJobVERL(BaseJob):
    engine: str = "sglang"
    use_sharp: bool = False

    ray_envs: ClassVar[dict[str, str]] = {
        "RAY_local_gc_interval_s": "120",
        "RAY_global_gc_min_interval_s": "120",
        "RAY_raylet_check_gc_period_milliseconds": "100",
        "RAY_high_plasma_storage_usage": "0.7",
        "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": "python",
        # "VLLM_ATTENTION_BACKEND": "XFORMERS",
        # "TORCH_NCCL_ENABLE_MONITORING": "1",
        # "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC": "30",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
        "NCCL_DEBUG": "INFO",
        "LC_ALL": "C",
        "PYTHONUNBUFFERED": "1",
        # fix: disable NCCL GDR to free up VRAM
        "NCCL_NET_GDR_LEVEL": "LOC",
        "NCCL_NET_GDR_READ": "0",
        # fix: increase NCCL timeout to avoid NCCL errors
        "NCCL_IB_TIMEOUT": "23",
        "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC": "1800",
        # sharp off
        "OMPI_MCA_coll_hcoll_enable": "0",
        "TORCHINDUCTOR_COMPILE_THREADS": "8",
    }

    BASE_IMAGES: ClassVar[dict[str, dict[str, str]]] = VERL_BASE_IMAGES

    # `setup_ray_cluster/slurm.sh` stops Ray on TERM: `ray stop --grace-period 15`
    # plus a 5s forced stop, starting once the cleanup's ~4s discovery ends. The
    # supervisor stage has to outlast that, and the stages must still fit Slurm's
    # KillWait, so the TERM stage gives up what the supervisor takes.
    SUPERVISOR_GRACE_S: ClassVar[int] = 25
    SHUTDOWN_GRACE_S: ClassVar[int] = 25  # 25 + 25 + the 5s KILL stage = KillWait 60

    @property
    def verl_root(self) -> Path:
        if verl_root := os.getenv("VERL_ROOT_PATH"):
            return Path(verl_root)
        return (
            Path(__file__).absolute().parent.parent.parent.parent.parent.parent.parent
        )

    def _external_code_libraries(self) -> list[str]:
        if self.engine == "sglang_custom":
            return [
                str(
                    self.verl_root
                    / "sber_src"
                    / "contribs"
                    / "sglang-pretrain"
                    / "python"
                ),
                str(self.verl_root),
            ]
        return [str(self.verl_root)]

    def _resolve_base_image(self) -> Optional[str]:
        assert self.engine in self.BASE_IMAGES, self._pretty_params_print(
            {"engine": list(self.BASE_IMAGES.keys())}
        )
        return self.base_image or self.BASE_IMAGES[self.engine][self.region]

    def _build_env(self) -> dict[str, str]:
        env = BaseJob._build_env(self)
        env.update(self.ray_envs)
        if "PROCESS_SUPERVISOR_GRACE_S" not in self.extra_env:
            env["PROCESS_SUPERVISOR_GRACE_S"] = str(self.SUPERVISOR_GRACE_S)
        if "PROCESS_SHUTDOWN_GRACE_S" not in self.extra_env:
            env["PROCESS_SHUTDOWN_GRACE_S"] = str(self.SHUTDOWN_GRACE_S)
        if self.use_sharp:
            env.update(
                {
                    # 'OMPI_MCA_coll_hcoll_enable': '1',
                    # 'SHARP_SMX_SOCK_ADDR_FAMILY': 'ipv4',
                    "NCCL_COLLNET_ENABLE": "1",
                    # 'HCOLL_ENABLE_SHARP': '1',
                    # 'SHARP_COLL_ENABLE_SAT': '1',
                }
            )
        return env
