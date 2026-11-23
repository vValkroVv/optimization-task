"""Configure one bounded EP4 diagnostic through the native profile runner."""

from contextlib import contextmanager
from types import SimpleNamespace


UPDATE = 21
WARMUP = UPDATE - 1
MICROBATCHES = 3
RANKS = (0, 4)


def native_args():
    return [
        "--profile-update",
        str(UPDATE),
        "--profile-microbatches",
        str(MICROBATCHES),
        "--profile-ranks",
        *(str(rank) for rank in RANKS),
    ]


def profile_settings(base):
    return base._profile_config_values(
        SimpleNamespace(profile_update=UPDATE, profile_ranks=list(RANKS))
    )


def profile_cli(base):
    def format_value(value):
        return str(value).lower() if isinstance(value, bool) else str(value)

    return {
        f"--{key}": format_value(value) for key, value in profile_settings(base).items()
    }


@contextmanager
def capture_cuda_sync(rank):
    if rank not in RANKS:
        yield
        return

    from torch._C._profiler import _set_cuda_sync_enabled_val

    _set_cuda_sync_enabled_val(True)
    try:
        print(
            f"Expert GFSDP trace rank {rank}: CUDA synchronization activities enabled",
            flush=True,
        )
        yield
    finally:
        _set_cuda_sync_enabled_val(False)


def verify_shard_group(measurement):
    from torch.distributed import get_process_group_ranks
    from torchtitan.distributed.gfsdp.fully_shard import _iter_fsdp_param_groups

    trainer = measurement.trainer
    if tuple(trainer.config.profiler.profile_ranks or ()) != RANKS:
        raise RuntimeError("Resolved profiler ranks differ from the EP4 trace pair")
    lane = measurement.rank % 4
    expected = [lane, lane + 4]
    groups = [
        group
        for part in trainer.model_parts
        for group in _iter_fsdp_param_groups(part, recurse=True)
        if {param._module_info.param_name for param in group.fsdp_params}
        == {"w13_E2FD", "w2_EDF"}
    ]
    if not groups:
        raise RuntimeError("The EP4 trace found no routed-expert GFSDP groups")
    for group in groups:
        actual = get_process_group_ranks(group._all_gather_process_group)
        if actual != expected:
            raise RuntimeError(
                f"Expert shard ranks for {group._module_fqn} are {actual}, "
                f"expected {expected}"
            )
