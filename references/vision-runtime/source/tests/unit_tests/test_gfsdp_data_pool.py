"""Data arenas follow simultaneous FSDP units, rather than storage classes."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from torchtitan.distributed.gfsdp._state import _data_buffer_ctx_idx_map


class Group:
    def __init__(self, size=128, *, frozen=False, pg=None, dtype=None):
        self.mesh_info = SimpleNamespace(shard_process_group=pg)
        self.orig_dtype = torch.bfloat16 if frozen else torch.float32
        self.param_dtype = dtype or torch.bfloat16
        self.padded_unsharded_param_numel = {None: size}
        self.requires_grad = not frozen
        self.reshard_after_forward = True
        self.reshard_after_backward = True


def unit(*groups, module=None):
    return SimpleNamespace(
        _fsdp_param_groups=list(groups),
        _modules=(module if module is not None else nn.Identity(),),
        _states_to_forward_prefetch=[],
        _states_to_backward_prefetch=[],
    )


def test_homogeneous_blocks_reuse_two_arenas_across_storage_and_trainability():
    groups = [Group(frozen=i < 6) for i in range(8)]
    mapping = _data_buffer_ctx_idx_map([unit(group) for group in groups])
    assert len(set(mapping.values())) == 2
    for current, following in zip(groups, groups[1:]):
        assert mapping[current] != mapping[following]


def test_current_and_prefetched_mixed_units_have_disjoint_arenas():
    states = [unit(Group(frozen=True), Group(size=8)) for _ in range(4)]
    mapping = _data_buffer_ctx_idx_map(states)
    assert len(set(mapping.values())) == 4
    for current, following in zip(states, states[1:]):
        live = current._fsdp_param_groups + following._fsdp_param_groups
        assert len({mapping[group] for group in live}) == 4


def test_slots_match_size_not_storage_class_for_asymmetric_units():
    large_frozen, small_trainable = Group(1024, frozen=True), Group(8)
    large_trainable, small_frozen = Group(1024), Group(8, frozen=True)
    middle = unit(Group(32))
    mapping = _data_buffer_ctx_idx_map(
        [
            unit(large_frozen, small_trainable),
            middle,
            unit(small_frozen, large_trainable),
        ]
    )
    assert mapping[large_frozen] == mapping[large_trainable]
    assert mapping[small_trainable] == mapping[small_frozen]
    assert mapping[large_frozen] != mapping[small_trainable]


def test_root_and_nested_units_have_independent_arenas():
    child = nn.Linear(2, 2)
    parent = nn.Sequential(child)
    root_group, child_group, other_group = Group(), Group(), Group()
    mapping = _data_buffer_ctx_idx_map(
        [
            unit(root_group, module=parent),
            unit(child_group, module=child),
            unit(other_group),
        ]
    )
    assert mapping[root_group] not in {mapping[child_group], mapping[other_group]}


@pytest.mark.parametrize("flag", ["reshard_after_forward", "reshard_after_backward"])
def test_explicitly_retained_unit_never_shares_its_arena(flag):
    groups = [Group() for _ in range(5)]
    setattr(groups[0], flag, False)
    mapping = _data_buffer_ctx_idx_map([unit(group) for group in groups])
    assert mapping[groups[0]] not in {mapping[group] for group in groups[1:]}


def test_grouped_sibling_modules_keep_dedicated_arenas_across_their_span():
    states = [unit(Group()) for _ in range(5)]
    states[0]._modules = (nn.Identity(), nn.Identity())
    mapping = _data_buffer_ctx_idx_map(states)
    own = states[0]._fsdp_param_groups[0]
    assert mapping[own] not in {
        mapping[group] for state in states[1:] for group in state._fsdp_param_groups
    }


@pytest.mark.parametrize("direction, target", [("forward", 2), ("backward", 3)])
def test_nonadjacent_explicit_prefetch_preserves_legacy_mapping(direction, target):
    states = [unit(Group()) for _ in range(5)]
    setattr(states[0], f"_states_to_{direction}_prefetch", [states[target]])
    mapping = _data_buffer_ctx_idx_map(states)
    assert mapping is None


def test_process_groups_and_actual_payload_dtypes_remain_separate():
    a, b = object(), object()
    groups = [Group(pg=a), Group(pg=b), Group(pg=a, dtype=torch.float32)]
    mapping = _data_buffer_ctx_idx_map([unit(group) for group in groups])
    assert len(set(mapping.values())) == 3


def test_private_branch_arena_is_never_reused_by_sequential_units():
    states = [unit(Group()) for _ in range(6)]
    states[0]._own_data_buffer = True
    mapping = _data_buffer_ctx_idx_map(states)
    private = mapping[states[0]._fsdp_param_groups[0]]
    assert private not in {
        mapping[group] for state in states[1:] for group in state._fsdp_param_groups
    }
    assert len({mapping[state._fsdp_param_groups[0]] for state in states[1:]}) == 2


def test_nonadjacent_private_prefetch_preserves_sequential_arena_reuse():
    states = [unit(Group()) for _ in range(6)]
    states[0]._own_data_buffer = True
    states[3]._states_to_backward_prefetch = [states[0]]
    mapping = _data_buffer_ctx_idx_map(states)
    assert mapping is not None
    assert (
        mapping[states[0]._fsdp_param_groups[0]]
        != mapping[states[2]._fsdp_param_groups[0]]
    )


def test_self_prefetch_is_a_noop_for_arena_planning():
    states = [unit(Group()) for _ in range(6)]
    for state in states:
        state._states_to_backward_prefetch = [state]
    mapping = _data_buffer_ctx_idx_map(states)
    assert len(set(mapping.values())) == 2
