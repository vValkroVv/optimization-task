"""Kernel-derived packing and MTP heads placed from named blocks."""

import copy

import pytest

from torchtitan.config.cli import load_document
from torchtitan.config.configurable import to_plain
from torchtitan.config.decode import decode
from torchtitan.config.errors import ConfigSchemaError
from torchtitan.config.schema.models.common.attention.const import MaskType
from torchtitan.config.sections import ModelSection

# The gift_3_5 debug blocks with softmax (MLA) attention.
_SOFTMAX_BLOCKS = ("mla_moe", "mtp")


def _hybrid_data():
    document, _ = load_document("configs/experiments/gift_3_5_debug.yaml")
    return copy.deepcopy(document.data["model"])


def _set_kernel(blocks, kernel):
    for name in _SOFTMAX_BLOCKS:
        blocks[name]["attention"]["inner_attention"] = {"type": kernel}


def test_varlen_kernel_alone_enables_document_packing():
    data = _hybrid_data()
    _set_kernel(data["config"]["blocks"], "varlen")
    model = decode(ModelSection, data).config.materialize()
    assert model.document_packing is True
    # gdn (linear) takes the mask type the softmax layers share.
    assert model.layers[0].attention.mask_type is MaskType.BLOCK_CAUSAL
    assert model.layers[2].attention.mask_type is MaskType.BLOCK_CAUSAL


@pytest.mark.parametrize("kernel,legacy", [("varlen", False), ("sdpa", True)])
def test_legacy_packing_assertion_must_match_selected_kernel(kernel, legacy):
    data = _hybrid_data()
    data["config"].update(varlen=legacy)
    _set_kernel(data["config"]["blocks"], kernel)
    with pytest.raises(ConfigSchemaError, match="unknown key"):
        decode(ModelSection, data)


def test_resolved_reload_can_change_kernel_without_stale_packing_flag():
    data = _hybrid_data()
    config = decode(ModelSection, data).config
    resolved = to_plain(config)
    assert "varlen" not in resolved
    assert resolved["blocks"]["gdn_dense"]["attention"]["mask_type"] is None
    for name in _SOFTMAX_BLOCKS:
        resolved["blocks"][name]["attention"]["inner_attention"]["type"] = "varlen"
    changed = decode(type(config), resolved).materialize()
    assert changed.document_packing
    assert changed.layers[0].attention.mask_type is MaskType.BLOCK_CAUSAL


def test_independent_mtp_ffn_without_dense_main_layers():
    data = _hybrid_data()
    values = data["config"]
    values["layers"] = [{"block": "gdn_moe"}, {"block": "mla_moe"}]
    values["blocks"]["mtp"]["feed_forward"] = {"hidden_dim": 768}
    values["mtp"] = {"layers": [{"block": "mtp", "repeat": 2}]}
    config = decode(ModelSection, data).config
    model = config.materialize()
    assert all(block.feed_forward is None for block in model.layers)
    assert model.mtp.layers[0].block.feed_forward.hidden_dim == 768
    assert model.mtp.layers[0].block.moe is None
    assert model.mtp.layers[0].block is not model.mtp.layers[1].block
    assert config.blocks["mtp"].dim is None


def test_mtp_heads_are_placed_from_the_named_block():
    data = _hybrid_data()
    model = decode(ModelSection, data).config.materialize()
    assert len(model.mtp.layers) == 2
    assert model.mtp.layers[0].block.feed_forward.hidden_dim == 1024


def test_conflicting_linear_mask_is_rejected_instead_of_overwritten():
    data = _hybrid_data()
    data["config"]["blocks"]["gdn_dense"]["attention"]["mask_type"] = "block_causal"
    config = decode(ModelSection, data).config
    with pytest.raises(ValueError, match="mixes attention masking"):
        config.materialize()


def test_explicit_mtp_mask_must_match_the_main_stack():
    data = _hybrid_data()
    data["config"]["blocks"]["mtp"]["attention"]["inner_attention"] = {"type": "varlen"}
    with pytest.raises(ValueError, match="mixes attention masking"):
        decode(ModelSection, data).config.materialize()


def test_gift_moe_dense_mtp_block():
    document, _ = load_document("configs/experiments/gift_moe_debug.yaml")
    data = copy.deepcopy(document.data["model"])
    values = data["config"]
    values["blocks"]["mtp"] = copy.deepcopy(values["blocks"]["mla_dense"])
    values["blocks"]["mtp"]["feed_forward"] = {"hidden_dim": 768}
    values["mtp"] = {"layers": [{"block": "mtp"}]}
    model = decode(ModelSection, data).config.materialize()
    assert model.mtp.layers[0].block.feed_forward.hidden_dim == 768
    assert model.mtp.layers[0].block.moe is None
