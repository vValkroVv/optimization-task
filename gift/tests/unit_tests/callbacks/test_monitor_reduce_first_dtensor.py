"""Reduce-first metrics of an FSDP2 ``DTensor`` take the same path as GFSDP's.

Before, a ``[reduce sum, stat]`` metric on a DTensor parameter was refused by
``_guard_double_reduction`` in ``partial_stats`` mode (and, in ``full_tensor``
mode, summed every copy of the gathered parameter), while the same metric on a
GFSDP ragged parameter counted each copy once. Now both gather the parameter
once and reduce it through ``ParallelReducer.reduce_counting_copies``;
the counting itself is tested on simulated ranks in
``test_monitor_sim_partial_stats``. This test pins the routing on a real
DTensor, on one gloo rank.
"""

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor
from torch.distributed.tensor.placement_types import Shard

from torchtitan.callbacks.metrics.reducers import (
    finalize_pending_reductions,
    ParallelReducer,
)
from torchtitan.callbacks.metrics.sample import (
    Sample,
)
from torchtitan.config.schema.callbacks.metrics.sample import (
    DTensorMode,
    MetricConfig,
    MetricStepConfig,
    MetricStepKind,
    ReduceOp,
    ReduceStageConfig,
    StatOpConfig,
)
from torchtitan.callbacks.metrics.stats import apply_post_stats
from torchtitan.callbacks.sources.base import MonitorSource
from tests.unit_tests.callbacks.sim_world import SimLayout, SimParallelDims


@pytest.fixture
def single_rank_gloo(tmp_path):
    if dist.is_initialized():
        dist.destroy_process_group()
    dist.init_process_group(
        "gloo", init_method=f"file://{tmp_path / 'store'}", rank=0, world_size=1
    )
    try:
        yield
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "mode", [DTensorMode.partial_stats, DTensorMode.full_tensor], ids=lambda m: m.value
)
def test_dtensor_reduce_first_gathers_once_and_counts_copies_once(
    single_rank_gloo, monkeypatch, mode
):
    mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("fsdp",))
    full = torch.randn(8, 6, generator=torch.Generator().manual_seed(0))
    value = distribute_tensor(full, mesh, [Shard(0)])

    seen_copies = []
    original = ParallelReducer.reduce_counting_copies

    def spy(self, tensor, stages, copies):
        seen_copies.append(copies)
        return original(self, tensor, stages, copies)

    monkeypatch.setattr(ParallelReducer, "reduce_counting_copies", spy)

    source = MonitorSource(MonitorSource.Config())
    source.active = True
    source.reducer = ParallelReducer(SimParallelDims(SimLayout(), rank=lambda: 0))
    metric = MetricConfig(
        name="mean_after_sum",
        dtensor_mode=mode,
        steps=[
            MetricStepConfig(
                kind=MetricStepKind.reduce,
                reduce=ReduceStageConfig(group="fsdp", op=ReduceOp.sum),
            ),
            MetricStepConfig(kind=MetricStepKind.stat, stat=StatOpConfig(op="mean")),
        ],
    )
    source._record(Sample(name="param/{metric}", value=value, metrics=[metric]))
    records = apply_post_stats(
        finalize_pending_reductions(source.drain(), source.reducer)
    )

    (record,) = records
    torch.testing.assert_close(
        record.value.double(), full.double().mean(), rtol=1e-6, atol=1e-7
    )
    # Counted against the parameter's own mesh: its ranks hold the copies.
    assert frozenset(value.device_mesh.mesh.flatten().tolist()) in seen_copies, (
        seen_copies
    )
