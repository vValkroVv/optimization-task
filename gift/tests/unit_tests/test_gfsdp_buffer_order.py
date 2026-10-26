# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Who gets which buffer, and what happens when the answer is wrong.

GFSDP deals a small number of buffers among many parameter groups, and every
crash the backend produced in September came from that dealing rather than from
arithmetic:

* two units that run next to each other in forward dealt the same **data**
  buffer -- zero-3 stopped on the second one to ask for it
  (``fqn=lm_head ... in use by fqn=layers.11[fsdp]``);
* one **gradient** buffer per process group cannot hold an unreduced gradient
  while another group reduces, so the chunked loss stopped ``tok_embeddings``
  (``in use by fqn=lm_head``) and had to be worked around by reducing on
  every chunk.

The decisions behind both are plain list and dict logic over opaque keys, so
these tests stand sentinels in for ``DeviceMesh`` and ``ProcessGroup`` and need
no distributed setup -- the same shape as ``test_gfsdp_param_meshes.py``.
"""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from torchtitan.distributed.gfsdp._param_group import GFSDPParamGroup  # noqa: E402
from torchtitan.distributed.gfsdp._state import (  # noqa: E402
    _forward_ordered_states,
    _grad_buffer_ctx_idx_map,
    get_buffer_ctx2ctx_using_param_groups_map,
)

PG_A = object()
PG_B = object()


class _FakeParamGroup:
    """Only the attributes the dealing reads.

    A class rather than ``SimpleNamespace`` because both maps put groups in a
    ``set``, and ``SimpleNamespace`` defines ``__eq__`` and so is unhashable.
    Identity is the right equality here anyway: the real dealing keys on the
    group object.

    ``data_buffer_ctx`` and ``grad_buffer_ctx`` stand in for the real buffer
    contexts; the dealing treats them as opaque and so do these tests.
    """

    def __init__(self, name: str, process_group: object = PG_A) -> None:
        self.name = name
        self.mesh_info = SimpleNamespace(shard_process_group=process_group)
        self.data_buffer_ctx = f"data:{name}"
        self.grad_buffer_ctx = f"grad:{name}"
        self.requires_grad = True

    def __repr__(self) -> str:  # pragma: no cover - only for failure output
        return f"<group {self.name}>"


class _FakeState:
    """A sharded module's state, as the ordering sees it."""

    def __init__(self, *param_groups: _FakeParamGroup) -> None:
        self._fsdp_param_groups = list(param_groups)


def _param_group(name: str, process_group: object = PG_A) -> _FakeParamGroup:
    return _FakeParamGroup(name, process_group)


def _state(*param_groups: _FakeParamGroup) -> _FakeState:
    return _FakeState(*param_groups)


# --------------------------------------------------------------------------
# Forward order
# --------------------------------------------------------------------------


def test_no_declared_order_keeps_traversal_order():
    states = [_state(), _state(), _state()]

    assert _forward_ordered_states(states, None) == states


def test_declared_order_wins_over_traversal_order():
    # The shape that broke zero-3: Decoder declares lm_head before mtp and runs
    # it after, so traversal order is not the order the buffers must avoid.
    norm, lm_head, mtp = _state(), _state(), _state()
    traversal = [norm, lm_head, mtp]

    assert _forward_ordered_states(traversal, [norm, mtp, lm_head]) == [
        norm,
        mtp,
        lm_head,
    ]


def test_undeclared_states_keep_their_traversal_position_after_declared_ones():
    first, second, third = _state(), _state(), _state()

    ordered = _forward_ordered_states([first, second, third], [third])

    assert ordered == [third, first, second]


def test_a_state_named_but_not_sharded_here_is_ignored():
    # ``apply_gfsdp`` builds the order from the model; a module that was never
    # sharded is not an error, and must not appear in the dealing either.
    mine, stranger = _state(), _state()

    assert _forward_ordered_states([mine], [stranger, mine]) == [mine]


def test_the_result_never_drops_or_duplicates_a_state():
    states = [_state() for _ in range(5)]

    ordered = _forward_ordered_states(states, [states[4], states[0]])

    assert sorted(map(id, ordered)) == sorted(map(id, states))


# --------------------------------------------------------------------------
# Data buffers: adjacency is the property that matters
# --------------------------------------------------------------------------


def _data_buffer_of(param_groups: list[SimpleNamespace]) -> dict[str, object]:
    buffer_ctx2groups, _ = get_buffer_ctx2ctx_using_param_groups_map(
        param_groups,
        "data_buffer_ctx",
        2,  # what ``_init_shared_state`` passes when any group reshards
        None,
        None,
    )
    return {
        group.name: buffer_ctx
        for buffer_ctx, groups in buffer_ctx2groups.items()
        for group in groups
    }


def test_neighbours_in_the_dealt_order_never_share_a_data_buffer():
    """The invariant zero-3 depends on.

    A unit releases its data buffer in ``post_forward``, but the prefetch for
    the next one fires from ``pre_forward`` -- so with resharding on, two units
    adjacent in forward hold their buffers at the same moment.
    """
    groups = [_param_group(name) for name in "abcdef"]

    assigned = _data_buffer_of(groups)

    for left, right in zip(groups, groups[1:]):
        assert assigned[left.name] != assigned[right.name], (
            f"{left.name} and {right.name} were dealt the same data buffer"
        )


def test_two_buffers_per_process_group_is_all_that_is_dealt():
    groups = [_param_group(name) for name in "abcdef"]

    assert len(set(_data_buffer_of(groups).values())) == 2


def test_process_groups_do_not_share_buffers_with_each_other():
    dense = [_param_group(f"dense{i}", PG_A) for i in range(3)]
    experts = [_param_group(f"expert{i}", PG_B) for i in range(3)]

    assigned = _data_buffer_of(dense + experts)

    assert not {assigned[g.name] for g in dense} & {assigned[g.name] for g in experts}


# --------------------------------------------------------------------------
# Gradient buffers: one unit holds an unreduced gradient
# --------------------------------------------------------------------------


def test_without_a_request_every_group_in_a_process_group_shares_one_buffer():
    groups = [_param_group(name) for name in "abc"]

    mapping = _grad_buffer_ctx_idx_map(groups, set())

    assert len(set(mapping.values())) == 1


def test_the_group_that_asked_gets_an_index_of_its_own():
    embeddings, layer, lm_head = (_param_group(n) for n in ("emb", "layer", "lm_head"))
    groups = [embeddings, layer, lm_head]

    mapping = _grad_buffer_ctx_idx_map(groups, {lm_head})

    assert mapping[lm_head] != mapping[embeddings]
    assert mapping[embeddings] == mapping[layer]


def test_asking_first_does_not_take_the_buffer_the_others_share():
    """Order must not decide the answer.

    ``lm_head`` is the last unit in forward but the first in declaration order,
    so both orders reach this map and both have to give the same shape of
    answer.
    """
    lm_head, embeddings, layer = (_param_group(n) for n in ("lm_head", "emb", "layer"))

    mapping = _grad_buffer_ctx_idx_map([lm_head, embeddings, layer], {lm_head})

    assert mapping[lm_head] != mapping[embeddings]
    assert mapping[embeddings] == mapping[layer]


def test_each_process_group_keeps_its_own_shared_buffer():
    dense = [_param_group(f"dense{i}", PG_A) for i in range(2)]
    experts = [_param_group(f"expert{i}", PG_B) for i in range(2)]
    lm_head = _param_group("lm_head", PG_A)

    mapping = _grad_buffer_ctx_idx_map(dense + experts + [lm_head], {lm_head})

    assert len({mapping[g] for g in dense}) == 1
    assert len({mapping[g] for g in experts}) == 1
    assert mapping[dense[0]] != mapping[experts[0]]
    assert mapping[lm_head] not in {mapping[dense[0]], mapping[experts[0]]}


def test_two_requests_do_not_collapse_onto_one_buffer():
    first, second, other = (_param_group(n) for n in ("first", "second", "other"))

    mapping = _grad_buffer_ctx_idx_map([first, second, other], {first, second})

    assert len({mapping[first], mapping[second], mapping[other]}) == 3


# --------------------------------------------------------------------------
# A prefetch into a busy buffer is skipped, not fatal
# --------------------------------------------------------------------------


class _FakeTarget:
    """Enough of a ``GFSDPParamGroup`` for the prefetch decision."""

    def __init__(self, owner: object) -> None:
        self._module_fqn = "layers.3"
        self.data_buffer_ctx = SimpleNamespace(owner=owner)
        self.unshard_async_op = False
        self.unsharded = False

    @contextmanager
    def use_training_state(self, _state):
        yield

    def unshard(self, _async_op):
        self.unshard_async_op = _async_op
        self.unsharded = True


@pytest.mark.parametrize("pass_type", ["forward", "backward"])
def test_a_free_buffer_is_prefetched(pass_type):
    target = _FakeTarget(owner=None)

    GFSDPParamGroup._prefetch_unshard(target, pass_type)

    assert target.unsharded


def test_a_buffer_the_target_already_owns_is_not_an_obstacle():
    target = _FakeTarget(owner=None)
    target.data_buffer_ctx.owner = target

    GFSDPParamGroup._prefetch_unshard(target, "forward")

    assert target.unsharded


def test_a_buffer_someone_else_holds_skips_the_prefetch():
    """Taken from gigafsdp: a prefetch is an optimization, not an obligation.

    The unit is all-gathered when it is entered instead, which is what would
    happen with no explicit prefetch at all. Before this, the same situation
    raised out of ``unshard``.
    """
    target = _FakeTarget(owner=object())

    GFSDPParamGroup._prefetch_unshard(target, "backward")

    assert not target.unsharded


def test_an_unknown_pass_type_is_refused():
    with pytest.raises(ValueError, match="Unknown pass type"):
        GFSDPParamGroup._prefetch_unshard(_FakeTarget(owner=None), "sideways")


# --------------------------------------------------------------------------
# The protocol torch's pipeline schedule dispatches on
# --------------------------------------------------------------------------


def test_gfsdp_modules_are_fsdp_modules():
    """Not cosmetic: ``torch.distributed.pipelining`` branches on this.

    ``stage.py`` clears ``param.grad`` after its shape probe, disables grad sync
    on non-last microbatches and runs the FSDP teardown once per step -- all of
    it behind ``isinstance(module, FSDPModule)``. While GFSDPModule was a plain
    class every one of those was silently skipped, which is where the nan under
    ``pp>1`` came from.
    """
    from torch.distributed.fsdp import FSDPModule

    from torchtitan.distributed.gfsdp.fully_shard import GFSDPModule

    assert issubclass(GFSDPModule, FSDPModule)


def test_the_gfsdp_module_implements_the_whole_protocol():
    from torchtitan.distributed.gfsdp.fully_shard import GFSDPModule

    for name in (
        "set_is_last_backward",
        "set_reshard_after_backward",
        "set_requires_gradient_sync",
        "unshard",
        "reshard",
    ):
        assert callable(getattr(GFSDPModule, name, None)), name


def test_torch_is_new_enough_to_carry_the_grad_cleanup_we_rely_on():
    """The pin must still clear gradients after the schedule's shape probe.

    Our own defence is the loss recognizing the probe by its target, but the
    reason the probe leaves nothing behind on fsdp2 is this cleanup, and it is
    what a torch bump could take away.
    """
    import inspect

    import torch.distributed.pipelining.stage as stage

    # By source rather than by attribute: which class carries it has moved
    # between versions, and the question here is whether the pin still does it
    # at all.
    assert "_post_metadata_inference_cleanup" in inspect.getsource(stage), (
        f"torch {torch.__version__} no longer has "
        "_post_metadata_inference_cleanup; check what it does after "
        "_backward_metadata_inference now"
    )


def test_torch_still_keeps_the_flags_the_chunked_loss_borrows():
    """``_fsdp_flags`` reads these off fsdp2's state to give them back.

    A rename in the pin would make the chunked loop restore a default instead
    of what the schedule set, which costs a reduce-scatter per microbatch and
    shows up as nothing at all.
    """
    import inspect

    from torch.distributed.fsdp._fully_shard import _fsdp_param_group, _fsdp_state

    assert "is_last_backward" in inspect.getsource(_fsdp_state.FSDPStateContext)
    assert "self._fsdp_param_group" in inspect.getsource(_fsdp_state.FSDPState)
    assert "reduce_grads" in inspect.getsource(_fsdp_param_group.FSDPParamGroup)
