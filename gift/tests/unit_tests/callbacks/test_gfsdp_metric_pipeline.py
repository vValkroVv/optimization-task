"""Shard-first monitoring and shared full-tensor fallback."""

from types import SimpleNamespace

import pytest
import torch
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Shard

from torchtitan.callbacks.metrics.sample import (
    Sample,
)
from torchtitan.config.schema.callbacks.metrics.sample import (
    AxisSpec,
    DTensorMode,
    MetricConfig,
    MetricStepConfig,
    ReduceConfig,
    ReduceOp,
    ReduceStageConfig,
    StatOpConfig,
)
from torchtitan.callbacks.sources.base import MonitorSource
from torchtitan.distributed.gfsdp._tensor import _compute_local_vector_norm


class RaggedForRouting(torch.Tensor):
    """Tensor-shaped double; numerical cross-rank behavior has a separate test."""

    @staticmethod
    def __new__(cls, data):
        result = torch.Tensor._make_subclass(cls, data.clone(), False)
        result._local_tensor = data.clone()
        result._full = data.clone()
        result.placements = ()
        result.device_mesh = SimpleNamespace(mesh_dim_names=())
        return result

    def detach(self):
        return self


class CountingShardStats:
    tensor_types = (RaggedForRouting,)
    placement_types = ()

    def __init__(self):
        self.gathers = 0

    def to_local(self, value):
        return value._local_tensor

    def local_squared_norms(self, value, dim):
        return value._local_tensor.square().sum(dim=dim)

    def to_full_tensor(self, value):
        self.gathers += 1
        return value._full.clone()


class DoublingReducer:
    def __init__(self, shard_stats):
        self.shard_stats = shard_stats
        self.ragged_tensor_types = shard_stats.tensor_types
        self.sharded_tensor_types = (DTensor, *self.ragged_tensor_types)
        self.reductions = []

    def reduce_stages(self, value, stages):
        self.reductions.append(tuple(stages))
        return value * 2

    @staticmethod
    def ranks_of(mesh):
        return frozenset()

    def reduce_counting_copies(self, value, stages, copies):
        return self.reduce_stages(value, stages), copies

    def reduce_over_groups(self, value, groups, op):
        assert groups == ("fsdp",) and op == ReduceOp.sum
        return value * 2


@pytest.fixture
def routed_source():
    support = CountingShardStats()
    source = MonitorSource(MonitorSource.Config())
    source.active = True
    source.reducer = DoublingReducer(support)
    return source, support


def test_shard_first_metrics_share_one_full_tensor(routed_source):
    source, support = routed_source
    value = torch.tensor([[-3.0, 4.0], [1.0, 2.0]])
    source._record(
        Sample(
            name="weights/{metric}",
            value=RaggedForRouting(value),
            axes=AxisSpec(names=("expert", "feature")),
            metrics=[
                MetricConfig(name="sum", pipeline=[StatOpConfig(op="sum")]),
                MetricConfig(
                    name="row_norm", pipeline=[StatOpConfig(op="norm", dim=1)]
                ),
                MetricConfig(
                    name="median", pipeline=[StatOpConfig(op="quantile", q=0.5)]
                ),
                MetricConfig(
                    name="full_mean",
                    pipeline=[StatOpConfig(op="mean")],
                    dtensor_mode=DTensorMode.full_tensor,
                ),
            ],
        )
    )
    records = {record.name: record.value for record in source.drain()}
    assert support.gathers == 1
    torch.testing.assert_close(records["weights/sum"], value.sum())
    torch.testing.assert_close(
        records["weights/row_norm"], torch.linalg.vector_norm(value, dim=1)
    )
    torch.testing.assert_close(records["weights/median"], torch.quantile(value, 0.5))
    torch.testing.assert_close(records["weights/full_mean"], value.mean())


def test_reduce_first_shares_reduced_full_tensor_for_one_prefix(routed_source):
    source, support = routed_source
    value = torch.tensor([1.0, 3.0, 5.0, 7.0])
    stage = ReduceStageConfig(group="loss", op="sum")
    source._record(
        Sample(
            name="expert/{metric}",
            value=RaggedForRouting(value),
            metrics=[
                MetricConfig(
                    name=name,
                    steps=[
                        MetricStepConfig(kind="reduce", reduce=stage),
                        MetricStepConfig(kind="stat", stat=StatOpConfig(op=op)),
                    ],
                )
                for name, op in (("mean", "mean"), ("max", "max"))
            ],
        )
    )
    records = {record.name: record for record in source.drain()}
    assert support.gathers == 1
    assert source.reducer.reductions == [(stage,)]
    torch.testing.assert_close(records["expert/mean"].value, (value * 2).mean())
    torch.testing.assert_close(records["expert/max"].value, (value * 2).max())
    assert all(not record.pending_stages for record in records.values())


@pytest.mark.parametrize(
    ("p", "expected"), [(2.0, 0.0), (float("inf"), 0.0), (float("-inf"), float("inf"))]
)
def test_empty_local_shard_has_norm_reduction_identity(p, expected):
    actual = _compute_local_vector_norm(torch.empty(0), p)
    assert actual.item() == expected


@pytest.mark.parametrize(
    ("local_values", "expected"), [([0.0, 2.0, -3.0], 4.0), ([], 0.0)]
)
def test_sharded_zero_norm_counts_nonzero_elements(
    routed_source, local_values, expected
):
    source, support = routed_source
    value = RaggedForRouting(torch.tensor(local_values))
    value.placements = (Shard(0),)
    value.device_mesh = SimpleNamespace(mesh_dim_names=("fsdp",))
    source._record(
        Sample(
            name="weights/{metric}",
            value=value,
            metrics=[
                MetricConfig(name="norm0", pipeline=[StatOpConfig(op="norm", p=0)])
            ],
        )
    )
    assert support.gathers == 0
    torch.testing.assert_close(
        source.drain()[0].value, torch.tensor(expected, dtype=torch.float32)
    )


@pytest.mark.parametrize(
    ("values", "weights", "expected"),
    [
        ([7.0, 8.0], [0.0, 0.0], 7.0),
        ([float("nan"), 3.0], [0.0, 2.0], 3.0),
        ([3.0, float("nan")], [2.0, 0.0], 3.0),
    ],
)
def test_empty_captures_do_not_divide_by_zero(routed_source, values, weights, expected):
    source, _ = routed_source
    metric = MetricConfig(
        name="mean",
        reduce=ReduceConfig(stages=[ReduceStageConfig(op=ReduceOp.mean)]),
    )
    for value, weight in zip(values, weights):
        source._accumulate("activations/mean", metric, torch.tensor(value), weight, ())
    source._flush_captures()
    torch.testing.assert_close(source.drain()[0].value, torch.tensor(expected))
