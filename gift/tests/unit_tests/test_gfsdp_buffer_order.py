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

from torchtitan.distributed.gfsdp._common import (  # noqa: E402
    DataParallelMeshInfo,
    TrainingState,
)
from torchtitan.distributed.gfsdp._param import ShardedState  # noqa: E402
from torchtitan.distributed.gfsdp._param_group import (  # noqa: E402
    GFSDPBufferContext,
    GFSDPParamGroup,
)
from torchtitan.distributed.gfsdp._state import (  # noqa: E402
    GFSDPState,
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


class _FakeStream:
    def __init__(self):
        self.events = []

    def record_event(self):
        event = object()
        self.events.append(event)
        return event


class _FakeBufferContext:
    def __init__(self):
        self.owner = None
        self.release_event = None
        self.yccl_handles = {torch.float32: None}

    def conflicting_owner(self, owner):
        return (
            self.owner if self.owner is not None and self.owner is not owner else None
        )

    def wait_for_release(self, _stream):
        return None


def _post_backward_group(
    *, training_state, has_grad, reduce_grads, reshard_after_backward
):
    group = object.__new__(GFSDPParamGroup)
    stream = _FakeStream()
    param = torch.nn.Parameter(torch.ones(2))
    param.grad = torch.ones_like(param) if has_grad else None
    fsdp_param = SimpleNamespace(
        unsharded_param=param,
        unsharded_accumulated_grad=None,
        _unsharded_accumulated_grad=torch.zeros_like(param),
        unsharded_grad_data=param.grad,
        sharded_param=SimpleNamespace(grad=None, requires_grad=True),
    )
    group._training_state = training_state
    group._module_fqn = "test.group"
    group._with_fqn = lambda name: name
    group.fsdp_params = [fsdp_param]
    group._all_gather_dtype_to_fsdp_params = {torch.float32: [fsdp_param]}
    group.grad_buffer_ctx = _FakeBufferContext()
    group.data_buffer_ctx = _FakeBufferContext()
    group.device_handle = SimpleNamespace(current_stream=lambda: stream)
    group.device = torch.device("cpu")
    group.orig_dtype = torch.float32
    group.param_dtype = torch.float32
    group.reduce_dtype = torch.float32
    group.gradient_divide_factor = None
    group.force_sum_reduction_for_comms = False
    group.mp_policy = SimpleNamespace(bit32_acc_for_bit16_reduce_scatter=False)
    group._reduce_scatter_input = {torch.float32: torch.zeros(4)}
    group._padded_sharded_param_grad = {torch.float32: torch.zeros(2)}
    group.mesh_info = object.__new__(DataParallelMeshInfo)
    group.mesh_info.shard_process_group = object()
    group.comm_ctx = SimpleNamespace(
        reduce_scatter_stream=stream, retain_tail_weights=False
    )
    group.reduce_grads = reduce_grads
    group.reshard_after_backward = reshard_after_backward
    group._post_reduce_event = object()  # Must not count as a new reduction.
    group._post_backward_prefetch_target = object()
    group._direct_expert_dw_lease = None
    group._owner_transport = None
    group._staged_head_dw = None
    group._fp8_cache = None
    group.data_buffer_ctx.owner = group
    return group, stream


@pytest.mark.parametrize(
    (
        "training_state",
        "has_grad",
        "reduce_grads",
        "reshard_after_backward",
        "expected_reductions",
        "expected_prefetches",
    ),
    [
        (TrainingState.PRE_BACKWARD, True, True, True, 1, 1),
        (TrainingState.IDLE, False, True, True, 0, 0),
        # A callback with stale group state and a nonempty grad still is not an
        # actual PRE_BACKWARD completion for this call.
        (TrainingState.IDLE, True, True, True, 1, 0),
        (TrainingState.PRE_BACKWARD, True, False, True, 0, 0),
        (TrainingState.PRE_BACKWARD, True, True, False, 1, 0),
    ],
)
def test_post_backward_transition_requires_real_reduced_resharded_backward(
    monkeypatch,
    training_state,
    has_grad,
    reduce_grads,
    reshard_after_backward,
    expected_reductions,
    expected_prefetches,
):
    import torchtitan.distributed.gfsdp._param_group as param_group_module

    group, _stream = _post_backward_group(
        training_state=training_state,
        has_grad=has_grad,
        reduce_grads=reduce_grads,
        reshard_after_backward=reshard_after_backward,
    )
    reductions = []
    prefetches = []
    reduce_event = object()

    def fake_reduce_scatter(*_args, **_kwargs):
        reductions.append(True)
        return reduce_event

    def fake_reshard(self):
        self.data_buffer_ctx.release_event = _stream.record_event()
        self.data_buffer_ctx.owner = None

    monkeypatch.setattr(param_group_module, "reduce_scatter", fake_reduce_scatter)
    monkeypatch.setattr(GFSDPParamGroup, "reshard", fake_reshard)
    monkeypatch.setattr(
        GFSDPParamGroup,
        "_prefetch_unshard",
        staticmethod(lambda target, pass_type: prefetches.append((target, pass_type))),
    )

    group.post_backward()

    assert len(reductions) == expected_reductions
    assert len(prefetches) == expected_prefetches
    if expected_prefetches:
        assert prefetches == [(group._post_backward_prefetch_target, "backward")]
    if not reduce_grads and has_grad:
        assert group.grad_buffer_ctx.owner is group


def test_staged_head_callbacks_keep_the_private_gradient_until_all_chunks_write(
    monkeypatch,
):
    import torchtitan.distributed.gfsdp._param_group as param_group_module
    from torchtitan.distributed.gfsdp._staged_head_dw import StagedHeadDw

    head, stream = _post_backward_group(
        training_state=TrainingState.PRE_BACKWARD,
        has_grad=False,
        reduce_grads=False,
        reshard_after_backward=False,
    )
    param = head.fsdp_params[0]
    slot = param._unsharded_accumulated_grad
    context = head.grad_buffer_ctx
    context.owner = head
    head._reduce_scatter_input[torch.float32] = slot
    staged = StagedHeadDw(head, param, context, slot, "c1", num_chunks=2)
    staged.stream = stream
    staged.begin()
    reductions = []
    reduce_event = object()

    def fake_reduce_scatter(_params, with_grad, _output, data, *_args, **kwargs):
        assert with_grad == [param]
        assert data is slot
        assert kwargs["owner_transport"] is None
        reductions.append(data.clone())
        return reduce_event

    monkeypatch.setattr(param_group_module, "reduce_scatter", fake_reduce_scatter)
    slot.fill_(1)
    staged.writes = 1
    head.post_backward()
    assert staged.completed_callbacks == 1
    assert head._staged_head_dw is staged
    assert context.owner is head
    assert context.release_event is None
    assert reductions == []

    head._training_state = TrainingState.PRE_BACKWARD
    head.reduce_grads = True
    slot.add_(2)
    staged.writes = 2
    head.post_backward()
    assert staged.completed_callbacks == 2
    assert head._staged_head_dw is None
    assert context.owner is None
    assert context.release_event is reduce_event
    assert len(reductions) == 1
    torch.testing.assert_close(reductions[0], torch.full_like(slot, 3))


@pytest.mark.parametrize(
    "policy, buffer_name",
    [
        ("reshard_after_forward", "data buffer"),
        ("reshard_after_backward", "data buffer"),
        ("reduce_grads", "grad buffer"),
    ],
)
def test_region_retention_validation_only_compares_aliased_storage(
    monkeypatch, policy, buffer_name
):
    def buffer(buffer_type):
        ctx = GFSDPBufferContext(buffer_type)
        ctx.lazy_init({None: 128}, torch.device("cpu"))
        return ctx

    def group(name, data=None, grad=None):
        return SimpleNamespace(
            name=name,
            data_buffer_ctx=data or buffer(GFSDPBufferContext.BufferType.ALL_GATHER),
            grad_buffer_ctx=grad
            or buffer(GFSDPBufferContext.BufferType.REDUCE_SCATTER),
            _post_backward_prefetch_target=None,
            reshard_after_forward=True,
            reshard_after_backward=True,
            reduce_grads=True,
        )

    data = buffer(GFSDPBufferContext.BufferType.ALL_GATHER)
    grad = buffer(GFSDPBufferContext.BufferType.REDUCE_SCATTER)
    source = group(
        "ffn",
        data.region({None: 64}, {None: 64}),
        grad.region({None: 64}, {None: 64}),
    )
    target = group("layer7", data.region({None: 32}, {None: 64}))
    norm = group("norm", grad=grad.region({None: 32}, {None: 64}))
    sibling = group(
        "disjoint",
        data.region({None: 64}, {None: 0}),
        grad.region({None: 64}, {None: 0}),
    )
    whole_slot = group("whole_slot", data, grad)
    unrelated = group("unrelated")

    root = GFSDPState()
    root._modules = (torch.nn.Module(),)
    root._fsdp_param_groups = [source, target, norm, sibling, whole_slot, unrelated]
    root._state_ctx.all_states = [root]

    overlaps = GFSDPBufferContext.overlaps
    comparisons = []

    def count_overlap(left, right):
        comparisons.append((left._storage_ctx, right._storage_ctx))
        return overlaps(left, right)

    monkeypatch.setattr(GFSDPBufferContext, "overlaps", count_overlap)
    setattr(sibling, policy, False)
    setattr(unrelated, policy, False)
    root.validate_shared_state()
    assert comparisons == []

    root._state_ctx.buffer_region_transition_groups = (source, target)
    root.validate_shared_state()
    assert comparisons == []

    setattr(whole_slot, policy, False)
    with pytest.raises(ValueError, match=f"share a {buffer_name}"):
        root.validate_shared_state()
    expected_root = data if buffer_name == "data buffer" else grad
    assert comparisons == [(expected_root, expected_root)]


@pytest.mark.parametrize("consume_gather", [True, False])
def test_nonfinal_backward_keeps_pending_transition_data_lease(consume_gather):
    root = GFSDPState()
    group, stream = _post_backward_group(
        training_state=TrainingState.IDLE,
        has_grad=False,
        reduce_grads=True,
        reshard_after_backward=True,
    )
    root._fsdp_param_groups = [group]
    root._state_ctx.all_states = [root]
    root._state_ctx.is_last_backward = False
    root._comm_ctx.post_forward_order = []

    backing = GFSDPBufferContext(GFSDPBufferContext.BufferType.ALL_GATHER)
    backing.lazy_init({None: 128}, torch.device("cpu"))
    group.data_buffer_ctx = backing.region({None: 64}, {None: 64})
    group.grad_buffer_ctx = None
    group._sharded_state = ShardedState.SHARDED
    group._protect_data_buffer_lease = True
    group._post_reduce_event = None
    group._post_forward_indices = []
    group._all_gather_input_set_dtypes = set()
    group._bf16_cache_enabled = False
    group.fsdp_params[0].to_unsharded = lambda: None
    group.fsdp_params[0].to_sharded = lambda: None

    waits = []
    gather = SimpleNamespace(wait=lambda: waits.append(group.data_buffer_ctx.owner))
    group._all_gather_result = gather
    group.data_buffer_ctx.owner = group

    root._root_post_backward_final_callback()

    assert group._all_gather_result is gather
    assert waits == []
    assert group.data_buffer_ctx.owner is group
    assert group.data_buffer_ctx.release_event is None

    peer = object.__new__(GFSDPParamGroup)
    peer.__dict__.update(group.__dict__)
    peer._all_gather_result = None
    peer.reshard()
    assert group.data_buffer_ctx.owner is group
    assert group.data_buffer_ctx.release_event is None

    target = _FakeTarget(owner=None)
    target.data_buffer_ctx = backing.region({None: 32}, {None: 64})
    GFSDPParamGroup._prefetch_unshard(target, "backward")
    assert not target.unsharded

    if consume_gather:
        group.wait_for_unshard()
        assert group.data_buffer_ctx.owner is group
        group.reshard()
    else:
        root._state_ctx.is_last_backward = True
        root._root_post_backward_final_callback()

    assert waits == [group]
    assert group._all_gather_result is None
    assert group.data_buffer_ctx.owner is None
    assert group.data_buffer_ctx.release_event is stream.events[-1]
    GFSDPParamGroup._prefetch_unshard(target, "backward")
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
