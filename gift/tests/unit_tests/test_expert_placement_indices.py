"""Which experts a rank owns, read off the DTensor placements.

``_caculate_indices_from_placements`` decides that from the mesh, and it used to
read ``placement.dim`` before knowing the placement had one. HSDP puts a
``Replicate()`` on the dp_replicate axis, which has no ``dim``, so the HF export
of any MoE model died there -- during a checkpoint save, hours into a run.
"""

import pytest
from torch.distributed.tensor.placement_types import _StridedShard, Replicate, Shard

from torchtitan.models.utils import MoEStateDictAdapter


class _SubMesh:
    def __init__(self, size: int, rank: int) -> None:
        self._size = size
        self._rank = rank

    def size(self) -> int:
        return self._size

    def get_local_rank(self) -> int:
        return self._rank


class _Mesh:
    """Enough DeviceMesh to answer 'how big is this axis and where am I on it'."""

    def __init__(self, axes: dict[str, tuple[int, int]]) -> None:
        self.mesh_dim_names = tuple(axes)
        self._axes = axes

    def __getitem__(self, name: str) -> _SubMesh:
        return _SubMesh(*self._axes[name])


class _Adapter(MoEStateDictAdapter):
    """Concrete enough to instantiate; the conversion halves are never called."""

    def from_hf(self, *args, **kwargs):
        raise NotImplementedError

    def to_hf(self, *args, **kwargs):
        raise NotImplementedError


# An instance, not the class: the method under test calls a sibling through
# ``self``, and with the class standing in for it that call is unbound -- the
# first real argument is swallowed as ``self``. ``__new__`` skips ``__init__``,
# which wants a model config we have nothing to do with here.
_ADAPTER = object.__new__(_Adapter)


def _indices(placements, axes, *, num_experts=8):
    return MoEStateDictAdapter._caculate_indices_from_placements(
        _ADAPTER,
        dim=0,
        dim_size=num_experts,
        dtensor_placements=placements,
        device_mesh=_Mesh(axes),
    )


def test_replicate_on_the_expert_dim_keeps_every_expert_local():
    """HSDP without expert parallelism: dp_replicate is Replicate, and the
    experts are whole on every rank."""
    axes = {"dp_replicate": (2, 1), "dp_shard": (4, 0)}
    assert _indices((Replicate(), Replicate()), axes) == (0, 8)


def test_replicate_beside_a_shard_does_not_hide_the_shard():
    axes = {"dp_replicate": (2, 1), "dp_shard": (4, 2)}
    assert _indices((Replicate(), Shard(0)), axes) == (4, 6)


def test_a_shard_on_another_dim_is_not_mistaken_for_this_one():
    axes = {"dp_shard": (4, 2), "tp": (2, 1)}
    assert _indices((Shard(1), Shard(1)), axes) == (0, 8)


def test_strided_shard_still_counts_as_sharding_the_expert_dim():
    axes = {"dp_shard": (2, 1), "ep": (2, 1)}
    start, end = _indices((_StridedShard(0, split_factor=2), Shard(0)), axes)
    assert (end - start) == 2


@pytest.mark.parametrize("placement", [Replicate(), Shard(1)])
def test_nothing_shards_the_expert_dim_means_all_of_it(placement):
    axes = {"only": (4, 3)}
    assert _indices((placement,), axes) == (0, 8)
