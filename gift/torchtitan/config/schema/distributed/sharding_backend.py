"""Torch-free rules for what each ``parallelism.fsdp_backend`` refuses.

The trainer Configs run them, and ``GFSDP2Backend`` raises the first issue
when built. The runtime keeps capability checks for backends not listed here.
"""

from torchtitan.config.configs import ParallelismConfig
from torchtitan.config.configurable import type_name_of
from torchtitan.config.schema.protocols.model import BaseModelConfig
from torchtitan.config.schema.components.checkpoint import CheckpointManagerConfig
from torchtitan.config.validation import Issue

__all__ = ["model_sharding_backend_issues", "sharding_backend_issues"]

_GFSDP2_PIPELINE_SCHEDULE = "Interleaved1F1B"


def model_sharding_backend_issues(
    backend: str, model_config: BaseModelConfig
) -> list[Issue]:
    """Check a backend against the model's declared runtime capabilities."""
    if not isinstance(model_config, BaseModelConfig):
        raise ValueError("sharding backend validation requires a model config")
    if backend in model_config.supported_sharding_backends:
        return []
    model_type = (
        type_name_of(type(model_config))
        or model_config.family
        or type(model_config).__qualname__
    )
    return [
        Issue(
            path="parallelism.fsdp_backend",
            message=f"backend {backend!r} is not supported by model type {model_type!r}; "
            f"supported backends: {model_config.supported_sharding_backends}",
        )
    ]


def sharding_backend_issues(
    parallelism: ParallelismConfig,
    *,
    model_config: BaseModelConfig | None = None,
    checkpoint: CheckpointManagerConfig | None = None,
    has_ema: bool = False,
    cpu_offload: bool = False,
) -> list[Issue]:
    """Check ``parallelism.fsdp_backend`` against the model, the other
    parallelisms and the checkpoint.

    ``checkpoint=None`` skips the checkpoint rules (the runtime backend is built
    before the checkpoint manager); ``model_config=None`` skips the model rule.
    """
    issues = (
        model_sharding_backend_issues(parallelism.fsdp_backend, model_config)
        if model_config is not None
        else []
    )
    if parallelism.fsdp_backend != "gfsdp2":
        return issues

    def error(path: str, message: str) -> None:
        issues.append(Issue(path=path, message=message))

    if parallelism.enable_fsdp_symm_mem:
        error(
            "parallelism.enable_fsdp_symm_mem",
            "is not supported on fsdp_backend='gfsdp2'; use fsdp2 for FSDP symmetric memory",
        )
    if cpu_offload:
        error(
            "training.enable_cpu_offload",
            "is not supported on fsdp_backend='gfsdp2'; use fsdp2 for CPU offload",
        )

    if parallelism.tensor_parallel_degree > 1:
        error(
            "parallelism.tensor_parallel_degree",
            f"is {parallelism.tensor_parallel_degree}, and GFSDP2 does not "
            "support tensor parallelism: it shards a parameter by flat element "
            "ranges, so a rank can hold part of a row. Run tensor parallelism "
            "on fsdp_backend='fsdp2'.",
        )
    if parallelism.pipeline_parallel_grad_reduce != "off":
        error(
            "parallelism.pipeline_parallel_grad_reduce",
            "GFSDP2 does not support in-backward gradient reduce-scatter; use fsdp2 or set it to 'off'",
        )
    if parallelism.pipeline_parallel_degree > 1:
        pp = parallelism.pipeline_parallel_degree
        if parallelism.pipeline_parallel_schedule_csv:
            error(
                "parallelism.pipeline_parallel_schedule_csv",
                f"is '{parallelism.pipeline_parallel_schedule_csv}', and GFSDP2 "
                f"runs only '{_GFSDP2_PIPELINE_SCHEDULE}'. Clear the csv path, "
                "or run it on fsdp_backend='fsdp2'.",
            )
        if parallelism.pipeline_parallel_schedule != _GFSDP2_PIPELINE_SCHEDULE:
            error(
                "parallelism.pipeline_parallel_schedule",
                f"is '{parallelism.pipeline_parallel_schedule}' with "
                f"pipeline_parallel_degree {pp}, and GFSDP2 runs only "
                f"'{_GFSDP2_PIPELINE_SCHEDULE}' under a pipeline: other "
                "schedules reach ``rms_norm`` with a sharded weight or split "
                "backward across the shared gradient buffer. Set "
                f"'{_GFSDP2_PIPELINE_SCHEDULE}', or run on fsdp_backend='fsdp2'.",
            )
        if parallelism.fsdp_reshard_after_forward == "always":
            error(
                "parallelism.fsdp_reshard_after_forward",
                f"is 'always' with pipeline_parallel_degree {pp}, and GFSDP2 "
                "cannot run that combination: resharding shares two data "
                "buffers per process group, and a pipeline keeps more parameter "
                "groups in flight than that. Use 'never' or 'default'.",
            )

    if checkpoint is None:
        return issues
    for flag, enabled in (
        ("save_in_hf", checkpoint.hf_save_enabled and checkpoint.save_in_hf),
        ("last_save_in_hf", checkpoint.hf_save_enabled and checkpoint.last_save_in_hf),
        ("initial_load_in_hf", checkpoint.enable and checkpoint.initial_load_in_hf),
    ):
        if enabled:
            error(
                f"checkpoint.{flag}",
                "is not supported on fsdp_backend='gfsdp2', which checkpoints "
                "through its own adapter: the HF export reads a plain model "
                "state dict and gfsdp2 does not produce one. Keep the ordinary "
                "DCP checkpoint -- the autometrics callback submits it itself "
                "on this backend, and the PAM job converts it to HF.",
            )
    if checkpoint.async_mode != "disabled":
        error(
            "checkpoint.async_mode",
            f"{checkpoint.async_mode!r} is not supported on fsdp_backend="
            "'gfsdp2': its checkpoint state aliases live parameter buffers. "
            "Use async_mode='disabled'.",
        )
    if has_ema and checkpoint.last_save_ema_model_also:
        error(
            "checkpoint.last_save_ema_model_also",
            "is not supported on fsdp_backend='gfsdp2' with an ema callback: "
            "every checkpoint already carries the EMA weights as ema_model.* "
            "tensors, which the converters read with use_ema.",
        )
    return issues
