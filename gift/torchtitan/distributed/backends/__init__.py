"""The sharding backends ``parallelism.fsdp_backend`` selects between."""

from torchtitan.config.configs import ParallelismConfig
from torchtitan.config.schema.protocols.model import BaseModelConfig
from torchtitan.config.schema.distributed.sharding_backend import (
    sharding_backend_issues,
)

from torchtitan.protocols.sharding_backend import EMAStore, ShardingBackend

__all__ = ["EMAStore", "ShardingBackend", "build_sharding_backend"]


def build_sharding_backend(
    parallelism: ParallelismConfig,
    model_config: BaseModelConfig,
    *,
    cpu_offload: bool = False,
) -> ShardingBackend:
    """The backend ``parallelism.fsdp_backend`` selects, having checked the rest
    of ``parallelism`` and the model against it: this is the only way a run gets
    a backend, so the guards cannot be skipped."""
    name = parallelism.fsdp_backend
    if name not in ("fsdp2", "gfsdp2"):
        raise ValueError(
            f"unknown parallelism.fsdp_backend {name!r}; supported: 'fsdp2', 'gfsdp2'."
        )
    issues = sharding_backend_issues(
        parallelism, model_config=model_config, cpu_offload=cpu_offload
    )
    if issues:
        raise ValueError(str(issues[0]))
    if name == "fsdp2":
        from torchtitan.distributed.backends.fsdp2 import FSDP2Backend

        backend: ShardingBackend = FSDP2Backend()
        backend.validate_parallelism(parallelism)
    else:
        from torchtitan.distributed.backends.gfsdp2 import GFSDP2Backend

        backend = GFSDP2Backend(match_fsdp2_init=parallelism.gfsdp_match_fsdp2_init)
    return backend
