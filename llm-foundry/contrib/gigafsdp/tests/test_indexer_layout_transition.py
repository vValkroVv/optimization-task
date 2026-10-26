from collections import OrderedDict
from types import SimpleNamespace

import torch

from giga_fsdp import utils
from giga_fsdp.common import MetaInfoKeys


_LAYER = "attn.flatten"
_BASE = "attn.base.weight"
_INDEXER = "attn.indexer.wq.weight"


def _meta(fqns, numels):
    return {
        MetaInfoKeys.LAYER_NAME_TO_META_INFO_IDX: {_LAYER: 0},
        MetaInfoKeys.FREEZED_LAYER_NAME_TO_META_INFO_IDX: {},
        MetaInfoKeys.PARAM_NAME_TO_LAYER_NAME: {fqn: _LAYER for fqn in fqns},
        "main_shards_fqns": [fqns],
        "main_shards_numels": [numels],
        "main_parameter_orig_shapes": [[(numel,) for numel in numels]],
        "main_parameter_padded_numels": [sum(numels)],
    }


def _run_remap(monkeypatch, source_fqns, source, target_fqns, target):
    source_meta = _meta(source_fqns, [4] + [2] * (len(source_fqns) - 1))
    target_meta = _meta(target_fqns, [4] + [2] * (len(target_fqns) - 1))
    group = object()
    monkeypatch.setattr(utils, "_flat_buffer_process_group", lambda *_: group)

    def copy_segments(source, target, mappings, *_):
        for source_offset, target_offset, length in mappings:
            target[target_offset : target_offset + length].copy_(
                source[source_offset : source_offset + length]
            )

    monkeypatch.setattr(utils, "_redistribute_flat_segments", copy_segments)
    state = OrderedDict(
        [("model.meta", source_meta), (f"model.{_LAYER}", source)]
    )
    result = utils._remap_flat_buffer(
        "model",
        SimpleNamespace(_meta_info=target_meta),
        _LAYER,
        target,
        state,
    )
    return result, target


def test_remap_drops_source_indexer_and_preserves_base_weights(monkeypatch):
    result, target = _run_remap(
        monkeypatch,
        [_BASE, _INDEXER],
        torch.tensor([1.0, 2.0, 3.0, 4.0, 9.0, 9.0]),
        [_BASE],
        torch.full((4,), -7.0),
    )

    assert result == [f"model.{_LAYER}"]
    torch.testing.assert_close(target, torch.tensor([1.0, 2.0, 3.0, 4.0]))


def test_remap_adds_target_indexer_without_overwriting_its_initialization(monkeypatch):
    result, target = _run_remap(
        monkeypatch,
        [_BASE],
        torch.tensor([1.0, 2.0, 3.0, 4.0]),
        [_BASE, _INDEXER],
        torch.full((6,), -7.0),
    )

    assert result == [f"model.{_LAYER}"]
    torch.testing.assert_close(
        target,
        torch.tensor([1.0, 2.0, 3.0, 4.0, -7.0, -7.0]),
    )


def test_remap_rejects_non_indexer_layout_changes(monkeypatch):
    result, target = _run_remap(
        monkeypatch,
        [_BASE, "attn.other.weight"],
        torch.tensor([1.0, 2.0, 3.0, 4.0, 9.0, 9.0]),
        [_BASE],
        torch.full((4,), -7.0),
    )

    assert result is None
    torch.testing.assert_close(target, torch.full((4,), -7.0))


# ScMoEBlock vs DeepseekGMMMoeBlock keep identical expert weights under different paths.
_MOE_LAYER = "moe.flatten"
_MOE_FROZEN_LAYER = "moe.flatten_freezed"
_SCMOE_FQNS = ["moe.gemm.experts.weight1", "moe.gemm.experts.weight2"]
_GMM_FQNS = ["moe.experts.weight1", "moe.experts.weight2"]


def _moe_meta(fqns, numels, frozen=False):
    prefix = "freezed" if frozen else "main"
    layer = _MOE_FROZEN_LAYER if frozen else _MOE_LAYER
    return {
        MetaInfoKeys.LAYER_NAME_TO_META_INFO_IDX: {} if frozen else {layer: 0},
        MetaInfoKeys.FREEZED_LAYER_NAME_TO_META_INFO_IDX: {layer: 0} if frozen else {},
        MetaInfoKeys.PARAM_NAME_TO_LAYER_NAME: {fqn: layer for fqn in fqns},
        f"{prefix}_shards_fqns": [fqns],
        f"{prefix}_shards_numels": [numels],
        f"{prefix}_parameter_orig_shapes": [[(numel,) for numel in numels]],
        f"{prefix}_parameter_padded_numels": [sum(numels)],
    }


def test_canonical_fqn_maps_scmoe_onto_gmm_spelling():
    assert utils.canonical_fqn(_SCMOE_FQNS[0]) == _GMM_FQNS[0]
    assert utils.canonical_fqn(_GMM_FQNS[0]) == _GMM_FQNS[0]
    assert utils.canonical_fqn("attn.gemm_proj.weight") == "attn.gemm_proj.weight"


def test_layouts_match_through_moe_impl_alias():
    source = _moe_meta(_SCMOE_FQNS, [4, 2])
    assert utils._flat_buffer_layouts_match(
        source, _MOE_LAYER, _moe_meta(_GMM_FQNS, [4, 2]), _MOE_LAYER
    )
    # Alias never hides a real layout change.
    assert not utils._flat_buffer_layouts_match(
        source, _MOE_LAYER, _moe_meta(_GMM_FQNS, [2, 4]), _MOE_LAYER
    )
    assert not utils._flat_buffer_layouts_match(
        source, _MOE_LAYER, _moe_meta(["moe.other.weight1", _GMM_FQNS[1]], [4, 2]), _MOE_LAYER
    )


def test_resolve_compatible_frozen_moe_buffer_across_impl_alias():
    source = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    state = OrderedDict(
        [("model.meta", _moe_meta(_SCMOE_FQNS, [4, 2])), (f"model.{_MOE_LAYER}", source)]
    )
    module = SimpleNamespace(_meta_info=_moe_meta(_GMM_FQNS, [4, 2], frozen=True))

    resolved = utils._resolve_compatible_buffer_name(
        "model", module, f"model.{_MOE_FROZEN_LAYER}", torch.zeros(6), state
    )

    assert resolved == f"model.{_MOE_LAYER}"


def test_remap_resolves_shared_fqns_through_alias(monkeypatch):
    result, target = _run_remap(
        monkeypatch,
        [_SCMOE_FQNS[0], _INDEXER],
        torch.tensor([1.0, 2.0, 3.0, 4.0, 9.0, 9.0]),
        [_GMM_FQNS[0]],
        torch.full((4,), -7.0),
    )

    assert result == [f"model.{_LAYER}"]
    torch.testing.assert_close(target, torch.tensor([1.0, 2.0, 3.0, 4.0]))


def test_remap_merges_frozen_base_and_indexer_buffers(monkeypatch):
    """Stage-1 -> stage-2: one trainable target buffer is rebuilt from the
    frozen-base and trainable-indexer source buffers. The combined FQN set is
    unchanged (empty diff), which must be accepted as a legal merge."""
    frozen_layer = "attn.flatten_freezed"
    source_meta = {
        MetaInfoKeys.LAYER_NAME_TO_META_INFO_IDX: {_LAYER: 0},
        MetaInfoKeys.FREEZED_LAYER_NAME_TO_META_INFO_IDX: {frozen_layer: 0},
        MetaInfoKeys.PARAM_NAME_TO_LAYER_NAME: {
            _BASE: frozen_layer,
            _INDEXER: _LAYER,
        },
        "main_shards_fqns": [[_INDEXER]],
        "main_shards_numels": [[2]],
        "main_parameter_orig_shapes": [[(2,)]],
        "main_parameter_padded_numels": [2],
        "freezed_shards_fqns": [[_BASE]],
        "freezed_shards_numels": [[4]],
        "freezed_parameter_orig_shapes": [[(4,)]],
        "freezed_parameter_padded_numels": [4],
    }
    target_meta = _meta([_BASE, _INDEXER], [4, 2])
    group = object()
    monkeypatch.setattr(utils, "_flat_buffer_process_group", lambda *_: group)

    def copy_segments(source, target, mappings, *_):
        for source_offset, target_offset, length in mappings:
            target[target_offset : target_offset + length].copy_(
                source[source_offset : source_offset + length]
            )

    monkeypatch.setattr(utils, "_redistribute_flat_segments", copy_segments)
    state = OrderedDict(
        [
            ("model.meta", source_meta),
            (f"model.{frozen_layer}", torch.tensor([1.0, 2.0, 3.0, 4.0])),
            (f"model.{_LAYER}", torch.tensor([9.0, 8.0])),
        ]
    )
    target = torch.full((6,), -7.0)

    result = utils._remap_flat_buffer(
        "model",
        SimpleNamespace(_meta_info=target_meta),
        _LAYER,
        target,
        state,
    )

    assert result == [f"model.{frozen_layer}", f"model.{_LAYER}"]
    torch.testing.assert_close(
        target, torch.tensor([1.0, 2.0, 3.0, 4.0, 9.0, 8.0])
    )
