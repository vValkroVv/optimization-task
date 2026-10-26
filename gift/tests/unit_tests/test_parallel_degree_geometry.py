"""Pre-submit and runtime use one world-size/degrees calculation."""

import pytest

from torchtitan.config.configs import ParallelismConfig
from torchtitan.config.schema.distributed.parallelism import (
    resolve_data_parallel_shard_degree,
)
from torchtitan.config.schema.trainer import _world_size_issues


@pytest.mark.parametrize(
    "degrees,world_size,expected",
    [
        ({}, 1, 1),
        ({}, 8, 8),
        ({"dp_replicate": 2, "tp": 2, "pp": 2}, 16, 2),
        ({"cp": 2, "ep": 4}, 8, 4),
        ({"dp_shard": 4, "ep": 4}, 4, 4),
        ({"dp_shard": 2, "tp": 2, "ep": 4}, 4, 2),
    ],
)
def test_resolves_auto_sharding_without_multiplying_ep(degrees, world_size, expected):
    kwargs = {
        "dp_replicate": 1,
        "dp_shard": -1,
        "cp": 1,
        "tp": 1,
        "pp": 1,
        "ep": 1,
        **degrees,
    }
    assert (
        resolve_data_parallel_shard_degree(**kwargs, world_size=world_size) == expected
    )


@pytest.mark.parametrize(
    "degrees,world_size,fragment",
    [
        ({"tp": 2}, 3, "not divisible"),
        ({"dp_shard": 2, "tp": 2}, 3, "!= WORLD_SIZE 3"),
        ({"dp_shard": 0}, 1, "dp_shard"),
        ({"ep": 0}, 1, "ep"),
        ({"dp_replicate": 2, "ep": 8}, 8, "expert_parallel_degree 8 must divide"),
        ({"dp_shard": 4, "ep": 3}, 4, "expert_parallel_degree 3 must divide"),
        ({}, 0, "WORLD_SIZE"),
    ],
)
def test_rejects_invalid_world_size_or_degrees(degrees, world_size, fragment):
    kwargs = {
        "dp_replicate": 1,
        "dp_shard": -1,
        "cp": 1,
        "tp": 1,
        "pp": 1,
        "ep": 1,
        **degrees,
    }
    with pytest.raises(ValueError, match=fragment):
        resolve_data_parallel_shard_degree(**kwargs, world_size=world_size)


@pytest.mark.parametrize("dp_shard,world_size", [(-1, 3), (2, 3), (-1, 4), (2, 4)])
def test_schema_reports_the_shared_geometry_error(dp_shard, world_size):
    config = ParallelismConfig(
        data_parallel_shard_degree=dp_shard, tensor_parallel_degree=2
    )
    kwargs = dict(
        dp_replicate=1, dp_shard=dp_shard, cp=1, tp=2, pp=1, ep=1, world_size=world_size
    )
    try:
        resolve_data_parallel_shard_degree(**kwargs)
    except ValueError as error:
        issues = _world_size_issues(config, world_size)
        assert len(issues) == 1
        assert issues[0].message == str(error)
    else:
        assert _world_size_issues(config, world_size) == []


def test_explicit_ep_geometry_is_rejected_during_pre_submit():
    from torchtitan.config.pre_submit import run_pre_submit

    result = run_pre_submit(
        "configs/experiments/gift_moe_debug.yaml",
        [
            ("parallelism.data_parallel_replicate_degree", "2"),
            ("parallelism.data_parallel_shard_degree", "4"),
            ("parallelism.expert_parallel_degree", "8"),
        ],
    )
    assert not result.ok
    assert any(
        "expert_parallel_degree 8 must divide" in i.message
        for i in result.report.errors
    )


def test_auto_sharding_ep_geometry_is_checked_when_world_size_becomes_known():
    config = ParallelismConfig(
        data_parallel_replicate_degree=2, expert_parallel_degree=8
    )
    issues = _world_size_issues(config, world_size=8)
    assert len(issues) == 1
    assert "expert_parallel_degree 8 must divide" in issues[0].message
