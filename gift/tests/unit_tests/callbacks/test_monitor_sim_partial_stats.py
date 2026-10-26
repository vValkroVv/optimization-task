"""Partial (shard-local) statistics of sharded parameters, on simulated ranks.

Every rank of a ``SimWorld`` holds one shard of a parameter (FSDP2 dim-0
chunks, GFSDP flat ranges, HSDP replicas, EP x efsdp experts) and records it
through the real ``MonitorSource``; the result on every rank must match the
statistic of the full tensor computed in one process, and must be bitwise
identical across ranks.

See ``sim_world.py`` for what is simulated and what runs unmodified.
"""

import math

import pytest
import torch

from torchtitan.callbacks.metrics import stats as stats_module
from torchtitan.callbacks.metrics.sample import (
    Sample,
)
from torchtitan.config.schema.callbacks.metrics.sample import (
    AxisSpec,
    DTensorMode,
    MetricConfig,
    MetricStepConfig,
    MetricStepKind,
    ReduceStageConfig,
    ReduceOp,
    StatOpConfig,
    ThresholdCmp,
)
from tests.unit_tests.callbacks.sim_world import (
    install,
    ParamLayout,
    reference_stats,
    run_monitor,
    sharded_value,
    SimLayout,
    SimShardStats,
    SimTopology,
    SimWorld,
)

SHAPE = (64, 48)


def _op(name: str, **kwargs) -> StatOpConfig:
    return StatOpConfig(op=name, **kwargs)


# name -> stat chain. Everything here is a whole-tensor statistic: the ones
# with a partial path run shard-local + a collective, the rest (quantile,
# gini, min_median_ratio) take the deferred full-tensor path.
SIGNED_STATS = {
    "count": [_op("count")],
    "sum": [_op("sum")],
    "mean": [_op("mean")],
    "rms": [_op("rms")],
    "std": [_op("std")],
    "var": [_op("var")],
    "kurtosis": [_op("kurtosis")],
    "norm2": [_op("norm")],
    "norm1": [_op("norm", p=1.0)],
    "norm3": [_op("norm", p=3.0)],
    "norm0": [_op("norm", p=0.0)],
    "norm_inf": [_op("norm", p=math.inf)],
    "max": [_op("max")],
    "min": [_op("min")],
    "abs_max": [_op("abs_max")],
    "max_min": [_op("max_min")],
    "nonzero_count": [_op("nonzero_count")],
    "zero_ratio": [_op("zero_ratio")],
    "positive_ratio": [_op("positive_ratio")],
    "actabs_gt": [_op("actabs", threshold=2.0)],
    "actabs_lt": [_op("actabs", threshold=0.1, cmp=ThresholdCmp.lt)],
    "actrel_gt": [_op("actrel", threshold=2.5)],
    "actrel_lt": [_op("actrel", threshold=0.2, cmp=ThresholdCmp.lt)],
    "amean": [_op("abs"), _op("mean")],
    "quantile90": [_op("quantile", q=0.9)],
}
POSITIVE_STATS = {
    "entropy": [_op("entropy")],
    "raw_entropy_norm": [_op("raw_entropy_norm")],
    "max_min_ratio": [_op("max_min_ratio")],
    "max_vio": [_op("max_vio")],
    "cv": [_op("cv")],
    "min_median_ratio": [_op("min_median_ratio")],
    "gini": [_op("gini")],
}
SPECIAL_STATS = {
    "nan_count": [_op("nan_count")],
    "inf_count": [_op("inf_count")],
    "finite_ratio": [_op("finite_ratio")],
}
SAMPLE_STATS = {
    "signed": SIGNED_STATS,
    "positive": POSITIVE_STATS,
    "special": SPECIAL_STATS,
}

EXACT = {"count", "norm0", "nonzero_count", "nan_count", "inf_count"}
LOOSE = {"kurtosis", "entropy", "raw_entropy_norm", "gini"}


def _tolerance(name: str) -> dict:
    if name in EXACT:
        return dict(rtol=0.0, atol=0.0)
    if name in LOOSE:
        return dict(rtol=1e-4, atol=1e-6)
    return dict(rtol=1e-5, atol=1e-6)


def _tensors(dtype: torch.dtype) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    signed = torch.randn(SHAPE, generator=generator) * 3 + 0.5
    signed[::7, ::5] = 0.0
    positive = torch.rand(SHAPE, generator=generator) + 0.05
    special = torch.randn(SHAPE, generator=generator)
    special[3, 4] = math.nan
    special[10, 1] = math.inf
    special[20, 2] = -math.inf
    special[::9, ::4] = 0.0
    return {
        "signed": signed.to(dtype),
        "positive": positive.to(dtype),
        "special": special.to(dtype),
    }


def _case(name: str, world_size: int) -> tuple[SimLayout, ParamLayout]:
    """A parameter layout over ``world_size`` ranks."""
    if name == "fsdp2_shard0":
        return SimLayout(dp_shard=world_size), ParamLayout(("fsdp",), ("shard0",))
    if name in ("gfsdp_even", "gfsdp_uneven", "gfsdp_first"):
        policy = name.removeprefix("gfsdp_")
        return SimLayout(dp_shard=world_size), ParamLayout(
            ("fsdp",), ("ragged",), policy
        )
    if name == "ep_gfsdp":
        ep = min(32, world_size)
        return SimLayout(dp_shard=world_size, ep=ep), ParamLayout(
            ("efsdp", "ep"), ("ragged", "shard0"), "even"
        )
    raise ValueError(name)


@pytest.fixture
def make_world(monkeypatch):
    def make(world_size: int) -> SimWorld:
        return install(monkeypatch, SimWorld(world_size))

    return make


def _metrics(stats: dict[str, list[StatOpConfig]]) -> list[MetricConfig]:
    return [MetricConfig(name=name, pipeline=chain) for name, chain in stats.items()]


def _check_against_full(
    results: list[dict[str, torch.Tensor]],
    fulls: dict[str, torch.Tensor],
    stats_by_sample: dict[str, dict[str, list[StatOpConfig]]],
    *,
    prefixes: tuple[str, ...] = ("",),
) -> None:
    first = results[0]
    for sample, stats in stats_by_sample.items():
        for name, chain in stats.items():
            expected = reference_stats(fulls[sample], chain)
            for prefix in prefixes:
                tag = f"{prefix}{sample}/{name}"
                assert tag in first, f"missing tag {tag}; got {sorted(first)[:10]}..."
                torch.testing.assert_close(
                    first[tag],
                    expected,
                    equal_nan=True,
                    msg=lambda m, tag=tag: f"{tag}: {m}",
                    **_tolerance(name),
                )
    for rank, result in enumerate(results[1:], start=1):
        assert result.keys() == first.keys(), f"rank {rank} published other tags"
        for tag, value in result.items():
            assert torch.equal(value, first[tag]) or (
                value.isnan().all() and first[tag].isnan().all()
            ), f"rank {rank} disagrees with rank 0 on {tag}: {value} vs {first[tag]}"


@pytest.mark.parametrize(
    "world_size,case,dtype",
    [
        pytest.param(7, "fsdp2_shard0", torch.float32, id="7-fsdp2_shard0-fp32"),
        pytest.param(7, "gfsdp_uneven", torch.float32, id="7-gfsdp_uneven-fp32"),
        pytest.param(64, "ep_gfsdp", torch.bfloat16, id="64-ep_gfsdp-bf16"),
    ],
)
def test_partial_stats_match_full_tensor(make_world, world_size, case, dtype):
    """Every whole-tensor stat of a sharded parameter equals the full-tensor stat.

    ``debug_full_tensor`` adds the ``debug_full/`` twin of every metric, which
    goes through the deferred full-tensor path; both must match the oracle.
    """
    layout, param = _case(case, world_size)
    topology = SimTopology(layout)
    fulls = _tensors(dtype)
    metrics = {sample: _metrics(stats) for sample, stats in SAMPLE_STATS.items()}

    def samples(rank: int) -> list[Sample]:
        return [
            Sample(
                name=f"{sample}/{{metric}}",
                value=sharded_value(full, param, topology, rank, seed=index),
                metrics=metrics[sample],
            )
            for index, (sample, full) in enumerate(fulls.items())
        ]

    world = make_world(layout.world_size)
    results = run_monitor(
        world, layout, samples, shard_stats=SimShardStats(), debug_full_tensor=True
    )
    _check_against_full(results, fulls, SAMPLE_STATS, prefixes=("", "debug_full/"))


@pytest.mark.parametrize("p", [-1.0, -math.inf])
@pytest.mark.parametrize("case", ["gfsdp_uneven", "gfsdp_first"])
def test_partial_norm_negative_orders(make_world, case, p):
    """Negative norm orders, empty shards included.

    -inf is the minimum |x| (once computed as ``(sum |x|**-inf) ** -0.0``,
    i.e. 1); a finite negative order sums ``|x|**p``, which has an identity
    for an empty shard where ``vector_norm`` raises.
    """
    layout, param = _case(case, 7)
    topology = SimTopology(layout)
    full = _tensors(torch.float32)["positive"]
    stats = {"norm_neg": [_op("norm", p=p)]}

    def samples(rank: int) -> list[Sample]:
        return [
            Sample(
                name="positive/{metric}",
                value=sharded_value(full, param, topology, rank),
                metrics=_metrics(stats),
            )
        ]

    results = run_monitor(
        make_world(layout.world_size), layout, samples, shard_stats=SimShardStats()
    )
    _check_against_full(results, {"positive": full}, {"positive": stats})


def _reduce_first(
    name: str, group, op: StatOpConfig, reduce_op: ReduceOp = ReduceOp.sum
) -> MetricConfig:
    return MetricConfig(
        name=name,
        steps=[
            MetricStepConfig(
                kind=MetricStepKind.reduce,
                reduce=ReduceStageConfig(group=group, op=reduce_op),
            ),
            MetricStepConfig(kind=MetricStepKind.stat, stat=op),
        ],
    )


HSDP = SimLayout(dp_replicate=4, dp_shard=16)
HSDP_PARAM = ParamLayout(("dp_replicate", "fsdp"), ("replicate", "ragged"), "uneven")


def test_reduce_first_counts_replicas_once(make_world):
    """A reduce-first ``sum`` over the axes a parameter is replicated on is a no-op.

    Every rank of the parameter's mesh holds a copy of the same logical tensor
    once gathered; summing the copies must not multiply it by their number.
    """
    topology = SimTopology(HSDP)
    full = _tensors(torch.float32)["signed"]
    metrics = [
        _reduce_first("mean_after_hsdp_sum", ("dp_replicate", "fsdp"), _op("mean")),
        _reduce_first("norm_after_fsdp_sum", "fsdp", _op("norm")),
        _reduce_first("max_after_world_sum", "world", _op("max")),
    ]

    def samples(rank: int) -> list[Sample]:
        value = sharded_value(full, HSDP_PARAM, topology, rank, seed=1)
        return [Sample(name="param/{metric}", value=value, metrics=metrics)]

    results = run_monitor(
        make_world(HSDP.world_size), HSDP, samples, shard_stats=SimShardStats()
    )
    expected = {
        "param/mean_after_hsdp_sum": full.double().mean(),
        "param/norm_after_fsdp_sum": torch.linalg.vector_norm(full.double()),
        "param/max_after_world_sum": full.double().max(),
    }
    for rank, result in enumerate(results):
        for tag, value in expected.items():
            torch.testing.assert_close(
                result[tag], value, rtol=1e-5, atol=1e-6, msg=f"rank {rank} {tag}"
            )


def test_reduce_first_over_copies_issues_no_collective(make_world):
    """A reduce-first stage whose group holds only copies is skipped outright.

    Every rank of the group already has the answer; summing the copies and
    dividing them back cost a full-tensor all-reduce and fp32 rounding that
    grew with the world (1.5e-5 off on 2048 ranks).
    """
    topology = SimTopology(HSDP)
    full = _tensors(torch.float32)["signed"]
    metrics = [
        _reduce_first("sum_hsdp", ("dp_replicate", "fsdp"), _op("sum")),
        _reduce_first("max_world", "world", _op("max"), ReduceOp.max),
    ]

    def samples(rank: int) -> list[Sample]:
        value = sharded_value(full, HSDP_PARAM, topology, rank, seed=2)
        return [Sample(name="param/{metric}", value=value, metrics=metrics)]

    world = make_world(HSDP.world_size)
    results = run_monitor(world, HSDP, samples, shard_stats=SimShardStats())
    assert all(not log for log in world.logs), world.logs[0][:5]
    for result in results:
        torch.testing.assert_close(
            result["param/sum_hsdp"], full.double().sum(), rtol=1e-6, atol=1e-5
        )
        assert torch.equal(result["param/max_world"], full.double().max())


@pytest.mark.parametrize("reduce_op", [ReduceOp.sum, ReduceOp.mean, ReduceOp.max])
def test_reduce_first_reduces_distinct_ep_tensors_once(make_world, reduce_op):
    """EP ranks hold different logical tensors; a reduce over ep sees each once.

    The parameter lives on the efsdp mesh only and its value depends on the
    rank's ep coordinate, so efsdp ranks are copies and ep ranks are not.
    """
    layout = SimLayout(dp_shard=16, ep=4)
    param = ParamLayout(("efsdp",), ("ragged",), "even")
    topology = SimTopology(layout)
    base = _tensors(torch.float32)["signed"]
    per_ep = [base + 10.0 * ep for ep in range(layout.ep)]

    def samples(rank: int) -> list[Sample]:
        ep = topology.mesh(rank, ("ep",)).get_coordinate()[0]
        value = sharded_value(per_ep[ep], param, topology, rank)
        metric = _reduce_first(
            "norm_after_reduce", ("efsdp", "ep"), _op("norm"), reduce_op
        )
        return [Sample(name="param/{metric}", value=value, metrics=[metric])]

    results = run_monitor(
        make_world(layout.world_size), layout, samples, shard_stats=SimShardStats()
    )
    stacked = torch.stack(per_ep).double()
    combined = {
        ReduceOp.sum: stacked.sum(0),
        ReduceOp.mean: stacked.mean(0),
        ReduceOp.max: stacked.amax(0),
    }[reduce_op]
    expected = torch.linalg.vector_norm(combined)
    for rank, result in enumerate(results):
        torch.testing.assert_close(
            result["param/norm_after_reduce"],
            expected,
            rtol=1e-5,
            atol=1e-5,
            msg=f"rank {rank}",
        )


def _reduce(group, op: ReduceOp) -> MetricStepConfig:
    return MetricStepConfig(
        kind=MetricStepKind.reduce, reduce=ReduceStageConfig(group=group, op=op)
    )


def _stat(op: StatOpConfig) -> MetricStepConfig:
    return MetricStepConfig(kind=MetricStepKind.stat, stat=op)


EP_LAYOUT = SimLayout(dp_shard=16, ep=4)


@pytest.mark.parametrize(
    "case", ["copies_only", "copies_grown_by_prefix", "distinct_after_stat"]
)
def test_reduce_after_stat_counts_copies_once(make_world, case):
    """A reduce stage after the stats of a reduce-first chain still counts copies once.

    ``copies_only``: ``[mean fsdp, norm, sum fsdp]`` on one tensor sharded
    over fsdp. The trailing sum spans copies only, so the answer is the norm,
    not ``fsdp`` times it.

    ``copies_grown_by_prefix``: ``[sum (efsdp, ep), norm, sum ep]``. The
    tensor differs per ep rank; once the prefix summed over ep, every ep rank
    holds the same result, and the trailing sum over ep must not multiply it.

    ``distinct_after_stat``: ``[mean efsdp, norm, sum ep]``. The prefix
    leaves each ep rank its own tensor, so the trailing sum adds the distinct
    norms -- the copy tracking must not suppress a legitimate sum.
    """
    if case == "copies_only":
        layout = SimLayout(dp_shard=8)
        param = ParamLayout(("fsdp",), ("ragged",), "uneven")
        steps = [
            _reduce("fsdp", ReduceOp.mean),
            _stat(_op("norm")),
            _reduce("fsdp", ReduceOp.sum),
        ]
    else:
        layout = EP_LAYOUT
        param = ParamLayout(("efsdp",), ("ragged",), "even")
        if case == "copies_grown_by_prefix":
            steps = [
                _reduce(("efsdp", "ep"), ReduceOp.sum),
                _stat(_op("norm")),
                _reduce("ep", ReduceOp.sum),
            ]
        else:
            steps = [
                _reduce("efsdp", ReduceOp.mean),
                _stat(_op("norm")),
                _reduce("ep", ReduceOp.sum),
            ]
    topology = SimTopology(layout)
    base = _tensors(torch.float32)["signed"]
    per_ep = [base + 10.0 * ep for ep in range(layout.ep)]
    metric = MetricConfig(name="chain", steps=steps)

    def samples(rank: int) -> list[Sample]:
        ep = topology.mesh(rank, ("ep",)).get_coordinate()[0] if layout.ep > 1 else 0
        value = sharded_value(per_ep[ep], param, topology, rank, seed=4)
        return [Sample(name="param/{metric}", value=value, metrics=[metric])]

    results = run_monitor(
        make_world(layout.world_size), layout, samples, shard_stats=SimShardStats()
    )
    norms = [torch.linalg.vector_norm(t.double()) for t in per_ep]
    expected = {
        "copies_only": norms[0],
        "copies_grown_by_prefix": torch.linalg.vector_norm(
            torch.stack(per_ep).double().sum(0)
        ),
        "distinct_after_stat": torch.stack(norms).sum(),
    }[case]
    for rank, result in enumerate(results):
        torch.testing.assert_close(
            result["param/chain"], expected, rtol=1e-5, atol=1e-4, msg=f"rank {rank}"
        )


# The optimizer metrics of the monitored production preset
# (``configs/presets/callbacks/monitored.yaml``).
PRODUCTION_OPTIMIZER_STATS = {
    "mean": [_op("mean")],
    "std": [_op("std")],
    "abs_max": [_op("abs_max")],
    "finite_ratio": [_op("finite_ratio")],
    "l2_norm": [_op("norm")],
    "rms": [_op("rms")],
    "max": [_op("max")],
    "min": [_op("min")],
    "zero_ratio": [_op("zero_ratio")],
    "nan_count": [_op("nan_count")],
}


def test_collective_sequence_is_identical_on_every_rank(make_world):
    """Ranks with empty shards issue the same collectives as the rest.

    A rank that issues a different collective, shape or dtype at the same
    position would hang the job; ``SimWorld`` raises ``CollectiveMismatch`` on
    the spot, and the per-rank logs must match exactly.
    """
    layout = SimLayout(dp_replicate=2, dp_shard=8, ep=4)
    topology = SimTopology(layout)
    full = _tensors(torch.bfloat16)["signed"]
    params = {
        "dense": ParamLayout(
            ("dp_replicate", "fsdp"), ("replicate", "ragged"), "first"
        ),
        "experts": ParamLayout(("efsdp", "ep"), ("ragged", "shard0"), "first"),
    }
    metrics = _metrics(PRODUCTION_OPTIMIZER_STATS)

    def samples(rank: int) -> list[Sample]:
        return [
            Sample(
                name=f"{kind}/{{metric}}",
                value=sharded_value(full, param, topology, rank, seed=3),
                metrics=metrics,
            )
            for kind, param in params.items()
        ]

    world = make_world(layout.world_size)
    results = run_monitor(world, layout, samples, shard_stats=SimShardStats())
    signatures = [[entry[1:] for entry in log] for log in world.logs]
    assert all(signature == signatures[0] for signature in signatures), (
        "ranks issued different collective sequences"
    )
    _check_against_full(
        results,
        {"dense": full, "experts": full},
        {"dense": PRODUCTION_OPTIMIZER_STATS, "experts": PRODUCTION_OPTIMIZER_STATS},
    )


def test_full_tensor_fallback_warns_once_per_metric(make_world, monkeypatch, caplog):
    """A stat with no shard-local path warns once that it gathers the full tensor.

    ``norm`` over a middle axis has no partial path (the axis-norm fast path
    covers the first and last axes only), so it falls back to a full gather.
    The warning comes once per metric however many captures hit it; metrics
    with a partial path, or that ask for ``full_tensor`` explicitly, stay quiet.
    """
    monkeypatch.setattr(stats_module, "_WARNED_FULL_TENSOR_FALLBACKS", set())
    layout, param = _case("gfsdp_even", 1)
    topology = SimTopology(layout)
    full = _tensors(torch.float32)["signed"].reshape(4, 12, 64)
    axes = AxisSpec(names=("a", "b", "c"))
    middle_norm = [_op("norm", dim=1), _op("mean")]
    metrics = [
        MetricConfig(name="middle_norm", pipeline=middle_norm),
        MetricConfig(name="mean", pipeline=[_op("mean")]),
        MetricConfig(
            name="middle_norm_explicit",
            pipeline=middle_norm,
            dtensor_mode=DTensorMode.full_tensor,
        ),
    ]

    def samples(rank: int) -> list[Sample]:
        return [
            Sample(
                name=f"capture{i}/{{metric}}",
                value=sharded_value(full, param, topology, rank),
                axes=axes,
                metrics=metrics,
            )
            for i in range(3)
        ]

    with caplog.at_level("WARNING"):
        results = run_monitor(
            make_world(layout.world_size), layout, samples, shard_stats=SimShardStats()
        )
    warnings = [
        r.getMessage()
        for r in caplog.records
        if "no shard-local path" in r.getMessage()
    ]
    assert len(warnings) == 1, warnings
    assert "'middle_norm'" in warnings[0] and "dim=1" in warnings[0]
    expected = reference_stats(full, middle_norm, axes)
    for i in range(3):
        torch.testing.assert_close(
            results[0][f"capture{i}/middle_norm"], expected, rtol=1e-5, atol=1e-6
        )
