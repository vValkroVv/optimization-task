import pytest

pytest.importorskip("streaming")

from streaming.base.world import World


def _world(num_nodes, ranks_per_node, rank=0):
    return World(
        num_nodes=num_nodes,
        ranks_per_node=ranks_per_node,
        workers_per_rank=1,
        worker=rank,
    )


@pytest.mark.parametrize(
    ("num_nodes", "ranks_per_node", "replication", "expected_grid"),
    [
        (2, 8, 2, (2, 4)),
        (2, 8, 4, (2, 2)),
        (2, 8, 8, (2, 1)),
        (1, 8, 2, (1, 4)),
        (1, 8, 8, (1, 1)),
        (4, 8, 2, (4, 4)),
    ],
)
def test_replicate_keeps_physical_nodes(
    num_nodes, ranks_per_node, replication, expected_grid
):
    replicated = _world(num_nodes, ranks_per_node).replicate(replication)
    assert (replicated.num_nodes, replicated.ranks_per_node) == expected_grid
    assert replicated.num_ranks == num_nodes * ranks_per_node // replication


@pytest.mark.parametrize(
    ("num_nodes", "ranks_per_node", "replication", "expected_grid"),
    [
        (2, 8, 16, (1, 1)),
        (3, 8, 3, (1, 8)),
    ],
)
def test_replicate_falls_back_to_upstream_collapse(
    num_nodes, ranks_per_node, replication, expected_grid
):
    replicated = _world(num_nodes, ranks_per_node).replicate(replication)
    assert (replicated.num_nodes, replicated.ranks_per_node) == expected_grid


def test_replicate_grid_matches_plain_batch_world():
    plain_batch = _world(num_nodes=2, ranks_per_node=4)
    replicated = _world(num_nodes=2, ranks_per_node=8).replicate(2)
    assert replicated.num_nodes == plain_batch.num_nodes
    assert replicated.ranks_per_node == plain_batch.ranks_per_node
    assert replicated.num_ranks == plain_batch.num_ranks


def test_replicate_rank_and_worker_mapping_unchanged():
    for rank in range(16):
        replicated = _world(2, 8, rank=rank).replicate(2)
        assert replicated.rank == rank // 2
        assert replicated.worker == rank // 2
        assert replicated.workers_per_rank == 1


def test_replicate_validation_unchanged():
    with pytest.raises(ValueError, match="must be positive"):
        _world(1, 8).replicate(0)
    with pytest.raises(ValueError, match="divisible"):
        _world(1, 8).replicate(3)
