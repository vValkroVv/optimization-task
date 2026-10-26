"""The DSA layout-transition machinery must stay invisible to an ordinary load.

`_remap_flat_buffer`, the FQN aliases and the frozen/trainable buffer rename were added
for the DSA stage-1 -> stage-2 transition, but they sit on the path every GigaFSDP
checkpoint travels. These tests pin the ordinary case -- same FQNs on both sides, no
indexer, no aliased MoE spelling -- to the behaviour it had before: the exact key
resolves, the remap path is never entered, values land bit-for-bit and the module is
not marked as having gone through a layout transition.
"""
from collections import OrderedDict
from types import SimpleNamespace

import pytest
import torch

from giga_fsdp import utils
from giga_fsdp.common import MetaInfoKeys


_LAYER = "attn.flatten"
_FQNS = ["attn.q_proj.weight", "attn.k_proj.weight", "attn.o_proj.weight"]
_NUMELS = [4, 2, 2]


def _meta(fqns, numels, layer=_LAYER):
    return {
        MetaInfoKeys.LAYER_NAME_TO_META_INFO_IDX: {layer: 0},
        MetaInfoKeys.FREEZED_LAYER_NAME_TO_META_INFO_IDX: {},
        MetaInfoKeys.PARAM_NAME_TO_LAYER_NAME: {fqn: layer for fqn in fqns},
        "main_shards_fqns": [fqns],
        "main_shards_numels": [numels],
        "main_parameter_orig_shapes": [[(numel,) for numel in numels]],
        "main_parameter_padded_numels": [sum(numels)],
    }


class _Module(torch.nn.Module):
    """The slice of the GigaFSDP surface `load_gigafsdp_model_state_dict` touches."""

    def __init__(self, meta, buffers):
        super().__init__()
        self._meta_info = meta
        self._buffers_by_name = buffers
        self.has_ema_flag = None
        self.copied_to_bit16 = 0

    def local_state_dict(self):
        return OrderedDict(self._buffers_by_name)

    def copy_master_weight_to_bit16_buffer(self):
        self.copied_to_bit16 += 1


def _plain_model(monkeypatch, *, with_ema=False):
    """A one-module model whose target buffers match the checkpoint exactly."""
    meta = _meta(_FQNS, _NUMELS)
    buffers = {_LAYER: torch.zeros(sum(_NUMELS))}
    source = OrderedDict(
        [("model.meta", meta), ("model." + _LAYER, torch.arange(sum(_NUMELS)).float())]
    )
    if with_ema:
        ema_name = _LAYER + utils.EMA_SUFFIX
        buffers[ema_name] = torch.zeros(sum(_NUMELS))
        source["model." + ema_name] = torch.arange(sum(_NUMELS)).float() * 10.0

    module = _Module(meta, buffers)
    model = torch.nn.Module()
    model.add_module("model", module)
    monkeypatch.setattr(utils, "GigaFSDP", _Module)
    return model, module, source


@pytest.mark.parametrize(
    "fqn",
    [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.7.mlp.experts.weight1",
        "model.layers.7.block_sparse_moe.gate.weight",
        "model.embed_tokens.weight",
    ],
)
def test_canonical_fqn_leaves_an_ordinary_name_alone(fqn):
    """Only the aliased MoE spelling is rewritten; everything else is identity."""
    assert utils.canonical_fqn(fqn) == fqn


def test_canonical_fqns_keeps_the_originals_when_aliasing_would_collide():
    """Two distinct parameters must never canonicalize onto one name."""
    fqns = ["m.gemm.experts.w", "m.experts.w"]
    assert utils._canonical_fqns(fqns) == fqns


def test_identical_layout_resolves_to_the_very_same_key(monkeypatch):
    """The ordinary case short-circuits on the exact name, before any rename."""
    _, module, source = _plain_model(monkeypatch)
    resolved = utils._resolve_compatible_buffer_name(
        "model", module, "model." + _LAYER, torch.zeros(sum(_NUMELS)), source
    )
    assert resolved == "model." + _LAYER


def test_plain_load_never_enters_the_remap_path(monkeypatch):
    """A checkpoint with the same FQNs must not touch the stage-transition code."""
    model, module, source = _plain_model(monkeypatch)

    def _forbidden(*args, **kwargs):
        raise AssertionError("_remap_flat_buffer must not run for an unchanged layout")

    monkeypatch.setattr(utils, "_remap_flat_buffer", _forbidden)
    missing, unexpected = utils.load_gigafsdp_model_state_dict(model, source)

    assert missing == []
    assert unexpected == []
    torch.testing.assert_close(
        module.local_state_dict()[_LAYER], torch.arange(sum(_NUMELS)).float()
    )
    assert module.copied_to_bit16 == 1


def test_plain_load_does_not_declare_a_layout_transition(monkeypatch):
    """has_ema_flag follows the checkpoint, which is what a non-transition load did."""
    model, module, source = _plain_model(monkeypatch, with_ema=True)
    missing, unexpected = utils.load_gigafsdp_model_state_dict(model, source)

    assert missing == []
    assert unexpected == []
    # A transition would have forced this to False and dropped the EMA buffer instead.
    assert module.has_ema_flag is True
    torch.testing.assert_close(
        module.local_state_dict()[_LAYER + utils.EMA_SUFFIX],
        torch.arange(sum(_NUMELS)).float() * 10.0,
    )


def test_plain_load_without_ema_in_the_checkpoint(monkeypatch):
    """No EMA in the source means no EMA flag, and still no transition."""
    model, module, source = _plain_model(monkeypatch)
    utils.load_gigafsdp_model_state_dict(model, source)
    assert module.has_ema_flag is False


def test_remap_refuses_a_layout_change_that_is_not_about_indexers(monkeypatch):
    """Guards the one relaxation: an empty diff is legal, a plain one is not."""
    source_meta = _meta(_FQNS, _NUMELS)
    target_meta = _meta(_FQNS[:2], _NUMELS[:2])
    monkeypatch.setattr(utils, "_flat_buffer_process_group", lambda *_: object())
    result = utils._remap_flat_buffer(
        "model",
        SimpleNamespace(_meta_info=target_meta),
        _LAYER,
        torch.zeros(sum(_NUMELS[:2])),
        OrderedDict(
            [("model.meta", source_meta), ("model." + _LAYER, torch.zeros(sum(_NUMELS)))]
        ),
    )
    assert result is None
