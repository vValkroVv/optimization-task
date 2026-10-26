"""Optimizer lifecycle metrics for packed weights and flat ragged shards."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, Shard
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta

from torchtitan.callbacks.metrics.reducers import (
    finalize_pending_reductions,
    ParallelReducer,
)
from torchtitan.callbacks.metrics.sample import (
    AxisSpec,
    MetricConfig,
    ReduceConfig,
    Sample,
    StatOpConfig,
)
from torchtitan.callbacks.metrics.stats import apply_stat_op
from torchtitan.callbacks.sources.base import iter_logical_weights
from torchtitan.callbacks.sources.optimizer_source import (
    compute_segment_metric,
    OptimizerSource,
)
from torchtitan.distributed.backends.gfsdp2 import GFSDPShardStats
from torchtitan.distributed.fsdp import _grouped_param_placement
from torchtitan.distributed.gfsdp._tensor import RaggedShard, RaggedShardDTensor
from torchtitan.models.common.moe import GroupedExperts


@pytest.fixture
def cpu_mesh(tmp_path):
    dist.init_process_group(
        "gloo", init_method=f"file://{tmp_path / 'store'}", rank=0, world_size=1
    )
    try:
        yield init_device_mesh("cpu", (1,), mesh_dim_names=("fsdp",))
    finally:
        dist.destroy_process_group()


def _ragged(full, mesh):
    # EFSDP partitions may cut through a logical weight inside an EP partition.
    partition = full
    if mesh.ndim == 2:
        shard_rank, expert_rank = mesh.get_coordinate()
        experts_per_rank = full.shape[0] // mesh.size(1)
        partition = full.narrow(0, expert_rank * experts_per_rank, experts_per_rank)
    else:
        shard_rank = mesh.get_local_rank()
    sizes = (
        (partition.numel(),)
        if mesh.size(0) == 1
        else (partition.numel() // 2 + 1, partition.numel() // 2 - 1)
    )
    offset = sum(sizes[:shard_rank])
    placements = (RaggedShard(sizes[shard_rank], offset, sizes),)
    if mesh.ndim == 2:
        placements = (*placements, Shard(0))
    spec = DTensorSpec(
        mesh,
        placements,
        TensorMeta(full.shape, full.stride(), full.dtype),
    )
    return RaggedShardDTensor(
        partition.flatten()[offset : offset + sizes[shard_rank]].clone(),
        spec,
        requires_grad=False,
    )


def _models(kind, mesh, *, split_reference=False):
    model = GroupedExperts.Config(dim=2, hidden_dim=3, num_experts=2).build()
    reference = nn.Module()
    for name, param in list(model.named_parameters()):
        full = torch.arange(param.numel(), dtype=torch.float32).reshape(param.shape)
        full = (full - 5) / 8
        parts = model.named_weight_slices().get(name) if split_reference else None
        if parts is None:
            reference.register_parameter(name, nn.Parameter(full.clone()))
        else:
            for expert in range(full.shape[0]):
                for group, (logical_name, _) in enumerate(parts):
                    reference.register_parameter(
                        f"{logical_name}__expert_idx_{expert}",
                        nn.Parameter(full[expert, group].clone()),
                    )
        if kind == "ragged":
            value = _ragged(full, mesh)
        elif kind == "dtensor":
            value = distribute_tensor(full, mesh, [Shard(0)])
        else:
            value = full.clone()
        setattr(model, name, nn.Parameter(value))
    return model, reference


@torch.no_grad()
def _capture(model, config, mesh, *, ragged=False):
    params = list(model.parameters())
    states = {}
    for param in params:
        param.grad = param * 0.1
        states[id(param)] = {
            "step": torch.tensor(3.0),
            "exp_avg": param * 0.02,
            "exp_avg_sq": param * param * 0.001 + 0.01,
        }
    group = SimpleNamespace(
        part_index=0,
        group_index=0,
        params=params,
        optimizer_type=torch.optim.AdamW,
        hparams={"lr": 0.01, "weight_decay": 0.1, "betas": (0.9, 0.99), "eps": 1e-8},
    )
    state = SimpleNamespace(
        model_parts=[model],
        step=1,
        optimizer_groups=lambda: (group,),
        optimizer_state=lambda param: states[id(param)],
    )
    dims = SimpleNamespace(
        pp_enabled=False,
        get_optional_mesh=lambda axis: mesh[axis]
        if axis in mesh.mesh_dim_names
        else None,
    )
    source = OptimizerSource(config)
    source.reducer = ParallelReducer(dims, GFSDPShardStats() if ragged else None)
    source.active = True
    tensors = [
        tensor
        for param in params
        for tensor in (param, param.grad, *states[id(param)].values())
    ]

    def local(tensor):
        return tensor.to_local() if hasattr(tensor, "to_local") else tensor

    before = [local(tensor).clone() for tensor in tensors]
    source.fit_start(state)
    for _ in range(2):
        source.batch_start(state)
        source.before_optimizer_step(state)
        source.after_optimizer_step(state)
        records = finalize_pending_reductions(source.drain(), source.reducer)
    for tensor, original in zip(tensors, before):
        torch.testing.assert_close(local(tensor), original, rtol=0, atol=0)
    names = [record.name for record in records]
    assert len(names) == len(set(names))
    return {record.name: record.value for record in records}


def _check_metrics(kind, mesh, family, *, per_expert=False):
    config = OptimizerSource.Config(
        gradients=family in ("all", "gradients"),
        parameters=family in ("all", "parameters"),
        optimizer_states=family in ("all", "optimizer_states"),
        per_expert_metrics=per_expert,
    )
    model, reference = _models(kind, mesh, split_reference=per_expert)
    expected = _capture(reference, config, mesh)
    with (
        patch("torchtitan.callbacks.sources.base.logger.warning") as warning,
        patch.object(
            GFSDPShardStats,
            "to_full_tensor",
            side_effect=AssertionError("optimizer statistics gathered the full tensor"),
        ),
    ):
        actual = _capture(model, config, mesh, ragged=kind == "ragged")
    warning.assert_not_called()
    if kind == "ragged":
        # Name-only users (TrainerSource's LR tags) keep the logical names.
        assert {name for name, _, _ in iter_logical_weights([model])} == {
            "w1_EFD",
            "w3_EFD",
            "w2_EDF",
        }
    assert actual.keys() == expected.keys()
    assert actual
    for name in actual:
        torch.testing.assert_close(
            actual[name], expected[name], msg=lambda e: f"{name}: {e}"
        )


@pytest.mark.parametrize("kind", ["ragged", "dtensor", "dense"])
@pytest.mark.parametrize(
    "family", ["all", "gradients", "parameters", "optimizer_states"]
)
def test_optimizer_packed_weight_metrics(cpu_mesh, kind, family):
    _check_metrics(kind, cpu_mesh, family)


@pytest.mark.parametrize("kind", ["ragged", "dtensor", "dense"])
@pytest.mark.parametrize(
    "family", ["all", "gradients", "parameters", "optimizer_states"]
)
def test_optimizer_per_expert_metrics(cpu_mesh, kind, family):
    _check_metrics(kind, cpu_mesh, family, per_expert=True)


@pytest.mark.parametrize("kind", ["ragged", "dtensor", "dense"])
def test_filter_uses_stored_parameter_name(cpu_mesh, kind):
    model, reference = _models(kind, cpu_mesh)
    config = OptimizerSource.Config(include_patterns=["w13_E2FD"])
    actual = _capture(model, config, cpu_mesh, ragged=kind == "ragged")
    expected = _capture(
        reference, OptimizerSource.Config(exclude_patterns=["w2_EDF"]), cpu_mesh
    )
    assert actual.keys() == expected.keys()
    assert actual and all("w13_E2FD" in name for name in actual)
    assert all("__expert_idx_" not in name for name in actual)


@pytest.mark.parametrize("kind", ["ragged", "dtensor", "dense"])
@pytest.mark.parametrize("family", ["gradients", "parameters", "optimizer_states"])
@pytest.mark.parametrize("per_expert", [False, True])
def test_optimizer_packed_weights_reject_offline_mode(
    cpu_mesh, kind, family, per_expert
):
    model, _ = _models(kind, cpu_mesh)
    metric = MetricConfig(
        name="mean",
        pipeline=[StatOpConfig(op="mean")],
        reduce=ReduceConfig(mode="offline"),
    )
    config = OptimizerSource.Config(
        include_patterns=["w13_E2FD"],
        metrics=[metric],
        gradients=family == "gradients",
        parameters=family == "parameters",
        optimizer_states=family == "optimizer_states",
        per_expert_metrics=per_expert,
    )
    with pytest.raises(ValueError, match="only ReduceMode.online"):
        _capture(model, config, cpu_mesh, ragged=kind == "ragged")


def _distributed_metrics(rank, store_path, expert_parallel):
    world_size = 4 if expert_parallel else 2
    dist.init_process_group(
        "gloo", init_method=store_path, rank=rank, world_size=world_size
    )
    try:
        mesh = init_device_mesh(
            "cpu",
            (2, 2) if expert_parallel else (2,),
            mesh_dim_names=("efsdp", "ep") if expert_parallel else ("fsdp",),
        )
        _check_metrics("ragged", mesh, "all")
        _check_metrics("ragged", mesh, "all", per_expert=True)
        if not expert_parallel:
            _check_metrics("dtensor", mesh, "all")
            _check_metrics("dtensor", mesh, "all", per_expert=True)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("expert_parallel", [False, True])
def test_optimizer_packed_weights_on_uneven_shards(tmp_path, expert_parallel):
    mp.spawn(
        _distributed_metrics,
        args=(f"file://{tmp_path / 'store'}", expert_parallel),
        nprocs=4 if expert_parallel else 2,
    )


def _expected_segment_metric(
    full: torch.Tensor, op: StatOpConfig, *, num_groups: int = 2
) -> torch.Tensor:
    rows = full.reshape(full.shape[0] * num_groups, -1)
    op = replace(op, dim=1)
    if op.op != "actrel":
        return apply_stat_op(rows, AxisSpec(), op).float()
    groups = rows.reshape(-1, num_groups, rows.shape[1]).unbind(1)
    return (
        torch.stack([apply_stat_op(group, AxisSpec(), op) for group in groups], dim=1)
        .flatten()
        .float()
    )


@pytest.mark.parametrize("kind", ["dense", "ragged"])
@pytest.mark.parametrize(
    "op",
    [
        StatOpConfig(op="mean"),
        StatOpConfig(op="std"),
        StatOpConfig(op="norm"),
        StatOpConfig(op="abs_max"),
        StatOpConfig(op="finite_ratio"),
        StatOpConfig(op="actrel", threshold=1.0),
    ],
    ids=lambda op: op.op,
)
def test_compute_segment_metric(cpu_mesh, kind, op):
    full = torch.arange(48, dtype=torch.float32).reshape(2, 2, 4, 3) - 13
    if op.op == "finite_ratio":
        full[0, 0, 0, 0] = float("inf")
    value = _ragged(full, cpu_mesh) if kind == "ragged" else full
    reducer = ParallelReducer(
        SimpleNamespace(pp_enabled=False),
        GFSDPShardStats() if kind == "ragged" else None,
    )
    actual, stages = compute_segment_metric(
        value,
        MetricConfig(name=op.op, pipeline=[op]),
        reducer,
        num_segments=4,
        num_groups=2,
    )
    assert stages == ()
    torch.testing.assert_close(actual, _expected_segment_metric(full, op))


def _segment_metrics_on_sharded_feature_dim(rank, store_path):
    dist.init_process_group("gloo", init_method=store_path, rank=rank, world_size=2)
    try:
        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("efsdp",))
        reducer = ParallelReducer(
            SimpleNamespace(
                pp_enabled=False,
                get_optional_mesh=lambda axis: mesh[axis]
                if axis in mesh.mesh_dim_names
                else None,
            )
        )
        metrics = [
            MetricConfig(name="mean", pipeline=[StatOpConfig(op="mean")]),
            MetricConfig(name="std", pipeline=[StatOpConfig(op="std")]),
            MetricConfig(name="norm", pipeline=[StatOpConfig(op="norm")]),
            MetricConfig(name="abs_max", pipeline=[StatOpConfig(op="abs_max")]),
            MetricConfig(
                name="finite_ratio", pipeline=[StatOpConfig(op="finite_ratio")]
            ),
            MetricConfig(
                name="actrel",
                pipeline=[StatOpConfig(op="actrel", threshold=1.0)],
            ),
        ]
        full = torch.arange(24, dtype=torch.float32).reshape(1, 2, 4, 3) - 7
        placement = _grouped_param_placement(
            nn.Parameter(full), axis_len=full.shape[0], num_shards=mesh.size()
        )
        assert placement == Shard(2)
        for metric in metrics:
            metric_full = full.clone()
            if metric.name == "finite_ratio":
                metric_full[0, 0, 0, 0] = float("inf")
            value = distribute_tensor(metric_full, mesh, [placement])
            assert value.reshape(2, -1).placements == (Shard(1),)
            actual, stages = compute_segment_metric(
                value, metric, reducer, num_segments=2, num_groups=2
            )
            expected = _expected_segment_metric(metric_full, metric.pipeline[-1])
            assert stages == ()
            torch.testing.assert_close(
                actual, expected, msg=lambda error: f"{metric.name}: {error}"
            )
    finally:
        dist.destroy_process_group()


def test_segment_metrics_reduce_sharded_feature_dim(tmp_path):
    mp.spawn(
        _segment_metrics_on_sharded_feature_dim,
        args=(f"file://{tmp_path / 'store'}",),
        nprocs=2,
    )


@pytest.mark.parametrize("kind", ["ragged", "dtensor"])
def test_packed_debug_full_matches_partial_stats(cpu_mesh, kind):
    model, _ = _models(kind, cpu_mesh)
    source = OptimizerSource(
        OptimizerSource.Config(
            per_expert_metrics=True,
            metrics=[
                MetricConfig(name="mean", pipeline=[StatOpConfig(op="mean")]),
                MetricConfig(
                    name="active", pipeline=[StatOpConfig(op="actrel", threshold=1.0)]
                ),
            ],
        )
    )
    source.active = True
    source.debug_full_tensor = True
    source.reducer = ParallelReducer(
        SimpleNamespace(
            pp_enabled=False, get_optional_mesh=lambda axis: cpu_mesh[axis]
        ),
        GFSDPShardStats() if kind == "ragged" else None,
    )
    source.fit_start(SimpleNamespace(model_parts=[model]))
    source._record_parameter_metrics(param_name="w13_E2FD", param=model.w13_E2FD)
    records = {record.name: record.value for record in source.drain()}
    assert len(records) == 24  # 3 metrics * 2 weights * 2 experts * 2 modes.
    for name, value in records.items():
        if not name.startswith("debug_full/"):
            torch.testing.assert_close(value, records[f"debug_full/{name}"])


def test_optimizer_record_preserves_base_accumulation(cpu_mesh):
    source = OptimizerSource(OptimizerSource.Config())
    source.active = True
    source.reducer = ParallelReducer(SimpleNamespace(pp_enabled=False))
    metric = MetricConfig(
        name="mean",
        pipeline=[StatOpConfig(op="mean")],
        reduce=ReduceConfig(group="fsdp", op="mean"),
    )
    for values in ([1.0, 3.0], [5.0]):
        source._record(
            Sample(name="plain", value=torch.tensor(values), metrics=[metric]),
            accumulate=True,
        )
    source.batch_end(SimpleNamespace())
    records = source.drain()
    assert len(records) == 1
    torch.testing.assert_close(records[0].value, torch.tensor(3.0))
