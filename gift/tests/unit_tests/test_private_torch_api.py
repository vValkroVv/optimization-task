"""Torch internals this fork stands on, pinned where a rename would be silent.

gfsdp2 is not written against torch's public surface. It registers itself in
the ``_composable`` contract registry so torch's own FSDP and pipelining code
can find its state, builds ``DTensorSpec`` by hand, draws every initial weight
through ``_random._rng_tracker``, and appends its tensor subclass to two
``_foreach_supported_types`` lists. None of that is covered by a compatibility
promise, and a torch bump can move any of it.

**These nine, and deliberately not the other fifty.** The tree imports 58
private torch symbols. Most of them fail *loudly* -- ``ImportError`` or
``TypeError`` before the first training step, with the symbol's name in the
traceback -- and a test for those only asserts that Python can import. Writes
into ``torch._inductor.config`` / ``torch._functorch.config`` are loud too:
``torch/utils/_config_module.py`` raises ``AttributeError`` on an unknown name
rather than quietly creating one. The ragged dispatch table is loud by
construction, ending in ``raise NotImplementedError`` with the operator name.

What is covered here is the remainder: the places where the symbol stays and
the *numbers* go wrong -- a reordered dataclass we fill positionally, a list
that becomes a tuple, a seed argument that moves. Those reach a cluster node as
a silently different run, which is the expensive way to find out.

All nine are CPU-only and need no process group.
"""

import dataclasses
import inspect

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402


def _params(func) -> list[str]:
    """Parameter names of ``func``, with ``self`` dropped."""
    names = list(inspect.signature(func).parameters)
    return names[1:] if names and names[0] == "self" else names


def _ctx() -> str:
    return f"(torch {torch.__version__})"


class _Probe(torch.Tensor):
    """Stands in for ``RaggedShardDTensor``; only the call shape is under test.

    ``__torch_dispatch__`` is not decoration: ``_make_wrapper_subclass`` refuses
    a subclass that lacks one, and it is the hook through which torch reaches
    the ragged dispatch table at all.
    """

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        # Metadata (shape, stride) is carried by the wrapper itself and must not
        # reach dispatch; if it does, this says which operator did.
        raise AssertionError(f"probe dispatched on {func}")


def test_make_wrapper_subclass_takes_the_call_shape_our_subclasses_use():
    """``RaggedShardDTensor`` and the DCP wrapper are both built by this call.

    The keywords are checked by constructing, not by reading a signature: the
    function is implemented in C and its keywords do not show up in
    ``inspect.signature``.
    """
    inner = torch.zeros(6)
    wrapper = torch.Tensor._make_wrapper_subclass(
        _Probe,
        (2, 3),
        strides=(3, 1),
        dtype=inner.dtype,
        device=inner.device,
        layout=inner.layout,
        requires_grad=False,
    )
    assert type(wrapper) is _Probe, f"subclass not honoured {_ctx()}"
    assert tuple(wrapper.shape) == (2, 3)
    assert tuple(wrapper.stride()) == (3, 1)

    # The other half of the contract: without the hook the call is refused, so a
    # subclass can never silently bypass the ragged dispatch table.
    class _NoHook(torch.Tensor):
        pass

    with pytest.raises(TypeError, match="__torch_dispatch__"):
        torch.Tensor._make_wrapper_subclass(_NoHook, (1,))


def test_the_rng_tracker_still_hands_out_a_region_per_spec():
    """``_fill_random_per_rank`` is now exactly this call and nothing else.

    Every initial weight on gfsdp2 is drawn inside
    ``_rng_tracker._distribute_region(spec)``: the spec's offset is what makes
    ranks draw different numbers and makes a run reproduce itself. If the
    region moves, initialization is silently wrong rather than absent.
    """
    from torch.distributed.tensor import _random

    assert hasattr(_random, "_rng_tracker"), (
        f"torch.distributed.tensor._random lost the module-level _rng_tracker "
        f"{_ctx()}; gfsdp2 reads that global in gfsdp/_random.py and _tensor.py"
    )
    region = _random.OffsetBasedRNGTracker._distribute_region
    assert _params(region)[:1] == ["spec"], (
        f"_distribute_region no longer takes a spec first {_ctx()}: "
        f"got {_params(region)}"
    )
    assert inspect.signature(region).parameters["generator"].default is None


def test_the_rng_offset_internals_fsdp2_matching_init_mirrors_hold():
    """``_fill_random_matching_fsdp2`` redoes the tracker's offset math by hand.

    It reads and writes the device generator's Philox offset through
    ``_PhiloxState`` and computes every shard's offset from torch's own helpers,
    for coordinates other than the current rank's. A reordered argument or a
    moved byte range keeps every call working and lands different numbers, so a
    gfsdp2 run silently stops matching fsdp2.
    """
    from torch.distributed.tensor import _random
    from torch.distributed.tensor._utils import _compute_local_shape_and_global_offset
    from torch.distributed.tensor.placement_types import Shard

    seed_bytes = torch.arange(8, dtype=torch.uint8)
    offset_bytes = torch.zeros(8, dtype=torch.uint8)
    state = _random._PhiloxState(torch.cat([seed_bytes, offset_bytes]))
    state.offset = state.offset + 40
    assert state.offset.item() == 40, f"_PhiloxState lost its offset {_ctx()}"
    assert torch.equal(state.state[:8], seed_bytes), (
        f"_PhiloxState no longer keeps the seed in the first 8 bytes {_ctx()}"
    )

    assert _params(_random._calc_shard_info) == ["mesh_coordinate", "spec"]
    assert _params(_random._calc_first_shard_size) == ["spec"]
    # Row-major over tensor dims: shard (1, 2) of a (2, 3) grid is the sixth.
    assert _random._calc_shard_linear_idx([1, 2], [2, 3]) == 5, (
        f"_calc_shard_linear_idx changed its order {_ctx()}"
    )

    assert _params(_compute_local_shape_and_global_offset)[:4] == [
        "global_shape",
        "mesh_shape",
        "my_coordinate",
        "placements",
    ], f"signature moved {_ctx()}"
    # Rows 7 over 4 ranks split 2, 2, 2, 1: rank 1 holds rows 2..3.
    assert _compute_local_shape_and_global_offset((7, 5), (4,), [1], (Shard(0),)) == (
        (2, 5),
        (2, 0),
    ), f"uneven Shard(0) split changed {_ctx()}"


def test_the_composable_contract_bridge_is_intact():
    """How torch's own code finds GFSDP state -- and the silent one.

    A composable contract keys its state by its own function, so
    ``g_fully_shard`` state is invisible to code that looks under
    ``fully_shard``. ``fully_shard.py`` publishes the same state object under
    ``fully_shard.__wrapped__`` inside ``module.__dict__[STATE_KEY]`` to bridge
    that. Nothing raises if the bridge breaks: torch simply finds no state, and
    pipelining stops reducing gradients.
    """
    from torch.distributed._composable import contract
    from torch.distributed._composable.contract import STATE_KEY, _get_registry
    from torch.distributed._composable_state import (
        _get_module_state,
        _insert_module_state,
        _State,
    )
    from torch.distributed.fsdp import fully_shard

    assert isinstance(STATE_KEY, str), f"STATE_KEY is no longer a str key {_ctx()}"
    assert hasattr(fully_shard, "__wrapped__"), (
        f"torch's fully_shard is no longer contract-decorated {_ctx()}; "
        f"gfsdp2 uses fully_shard.__wrapped__ as the registry key"
    )
    hash(fully_shard.__wrapped__)  # used as a dict key

    assert "state_cls" in inspect.signature(contract).parameters
    assert inspect.isclass(_State)
    assert _params(_get_registry) == ["module"]
    assert _params(_get_module_state) == ["module"]
    assert _params(_insert_module_state) == ["module", "state"]


def test_both_foreach_supported_type_lists_stay_mutable():
    """``install()`` appends ``RaggedShardDTensor`` to both of these.

    They are what makes the optimizer and ``clip_grad_norm_`` take the foreach
    path over our subclass. Turned into a tuple or a frozenset, the append
    raises at import; replaced by a membership function, the append succeeds and
    changes nothing -- the optimizer quietly falls back to its for-loop.
    """
    from torch.optim.optimizer import _foreach_supported_types as optim_types
    from torch.utils._foreach_utils import _foreach_supported_types as util_types

    for where, types in (
        ("torch.optim.optimizer", optim_types),
        ("torch.utils._foreach_utils", util_types),
    ):
        assert isinstance(types, list), (
            f"{where}._foreach_supported_types is {type(types).__name__}, "
            f"not a list {_ctx()}; gfsdp2 appends to it in _tensor.install()"
        )
        assert torch.Tensor in types, f"{where} no longer lists torch.Tensor {_ctx()}"
        types.append(_Probe)
        try:
            assert _Probe in types
        finally:
            types.remove(_Probe)


def test_local_shape_and_offset_keeps_its_positional_order():
    """Called positionally in three places, two of which write checkpoints.

    ``_dcp.py`` turns the returned offset into the global coordinates a DCP
    shard is stored at, and ``stats.py`` rebuilds full tensors from it. A
    parameter inserted ahead of these three would shift every argument by one
    and produce plausible, wrong offsets.
    """
    from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

    assert _params(compute_local_shape_and_global_offset)[:3] == [
        "global_shape",
        "mesh",
        "placements",
    ], f"signature moved {_ctx()}: {_params(compute_local_shape_and_global_offset)}"


def test_the_objects_we_build_positionally_keep_their_field_order():
    """``DTensorSpec``, ``TensorMeta`` and ``_NormPartial`` are filled by position.

    A reordered field passes every import check and yields a spec whose mesh and
    placements have traded places, or a gradient norm taken with the wrong p.
    """
    from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
    from torch.distributed.tensor._ops._math_ops import _NormPartial

    assert [f.name for f in dataclasses.fields(DTensorSpec)][:3] == [
        "mesh",
        "placements",
        "tensor_meta",
    ], f"DTensorSpec fields reordered {_ctx()}"
    assert TensorMeta._fields[:3] == ("shape", "stride", "dtype"), (
        f"TensorMeta fields reordered {_ctx()}"
    )
    assert _NormPartial(2).norm_type == 2, (
        f"_NormPartial no longer takes the norm type first {_ctx()}"
    )


def test_strided_shard_can_still_be_built_with_a_split_factor():
    """The one already on its way out, and the one we guard inconsistently.

    ``fsdp2_equivalent_placements`` builds ``_StridedShard(0, split_factor=n)``
    for a dim-0 shard nested under another shard axis. torch carries a TODO to
    remove the placement once it unifies with ``Shard``, its ``__init__``
    already accepts an ``sf`` alias, and ``stats.py`` imports the name behind a
    try/except while ``gfsdp/_random.py`` -- the hot path, reached on every init --
    does not. This test is the alarm for that asymmetry.
    """
    from torch.distributed.tensor.placement_types import _StridedShard

    placement = _StridedShard(0, split_factor=3)
    assert placement.dim == 0
    assert placement.split_factor == 3, (
        f"_StridedShard no longer reports split_factor {_ctx()}"
    )


def test_manual_seed_still_wants_the_mesh():
    """Our own code predicts this one will change.

    ``distributed/utils.py`` carries "remove the need of passing in a mesh once
    torch.distributed.tensor._random.manual_seed doesn't require a mesh input"
    and passes ``(seed, world_mesh)`` positionally. When the argument goes, drop
    it there -- do not silence this test: seeding the wrong value costs the
    run-to-run bitwise reproducibility every numeric comparison depends on.
    """
    from torch.distributed.tensor._random import manual_seed

    assert _params(manual_seed) == ["seed", "device_mesh"], (
        f"manual_seed signature changed {_ctx()}: {_params(manual_seed)}; "
        f"see the TODO in torchtitan/distributed/utils.py"
    )
