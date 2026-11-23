"""Select the native update-11 capture and verify its expert shard groups."""

from contextlib import contextmanager
from types import SimpleNamespace


UPDATE = 11
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
    return {
        f"--{key}": str(value).lower() if isinstance(value, bool) else str(value)
        for key, value in profile_settings(base).items()
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


def verify_shard_group(measurement, ep):
    from torch.distributed import get_process_group_ranks
    from torchtitan.distributed.gfsdp.fully_shard import _iter_fsdp_param_groups

    trainer = measurement.trainer
    if tuple(trainer.config.profiler.profile_ranks or ()) != RANKS:
        raise RuntimeError(f"Resolved profiler ranks must be {RANKS}")
    if trainer.config.parallelism.expert_parallel_degree != ep:
        raise RuntimeError(f"Resolved expert parallel degree must be {ep}")

    # The eight-rank sparse mesh puts eFSDP before EP.
    expected = tuple(range(measurement.rank % ep, 8, ep))
    groups = [
        group
        for part in trainer.model_parts
        for group in _iter_fsdp_param_groups(part, recurse=True)
        if {param._module_info.param_name for param in group.fsdp_params}
        == {"w13_E2FD", "w2_EDF"}
    ]
    if not groups:
        raise RuntimeError(f"The EP{ep} trace found no routed-expert GFSDP groups")
    for group in groups:
        actual = tuple(get_process_group_ranks(group._all_gather_process_group))
        if actual != expected:
            raise RuntimeError(
                f"Expert shard ranks for {group._module_fqn} are {actual}, "
                f"expected {expected}"
            )
    return expected
