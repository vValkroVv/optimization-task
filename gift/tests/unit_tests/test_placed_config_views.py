"""Runtime collection views check placed types and retain freeze wrappers."""

import pytest

from torchtitan.config import load_raw_document
from torchtitan.config.decode import decode
from torchtitan.config.sections import ModelSection
from torchtitan.config.schema.components.lora import FrozenConfig
from torchtitan.config.schema.models.common.decoder_blocks.layer_group import (
    placed_configs,
)
from torchtitan.config.schema.models.common.nn_modules import LinearConfig


def _written(experiment):
    document = load_raw_document(f"configs/experiments/{experiment}.yaml")
    return decode(ModelSection, document.data["model"]).config


def test_decoder_view_rejects_unexpanded_groups():
    config = _written("llama3_debug")
    with pytest.raises(
        ValueError, match=r"layers.0: expected placed.*LayerGroupConfig"
    ):
        config.placed_layers()
    assert not config.materialized
    tree = config.materialize()
    layers = tree.placed_layers()
    assert isinstance(layers, tuple)
    assert len(layers) == tree.n_layers
    assert all(layer is source for layer, source in zip(layers, tree.layers))


def test_decoder_view_preserves_frozen_positions_and_exact_contract():
    from torchtitan.config.schema.models.llama4.model import (
        Llama4TransformerBlockConfig,
    )

    tree = _written("llama3_debug").materialize()
    frozen = FrozenConfig(inner=tree.layers[0])
    copied = tree._copy(layers=[frozen])
    assert copied.placed_layers()[0] is frozen
    assert frozen.inner.layer_id == 0
    with pytest.raises(ValueError, match="Llama4TransformerBlockConfig"):
        copied.placed_layers(Llama4TransformerBlockConfig)


def test_mtp_view_rejects_groups_and_preserves_head_identity():
    config = _written("gift_3_5_debug")
    assert config.mtp is not None
    with pytest.raises(ValueError, match="mtp.layers.0"):
        config.mtp.placed_layers()
    tree = config.materialize()
    assert tree.mtp is not None
    assert all(
        head is source
        for head, source in zip(tree.mtp.placed_layers(), tree.mtp.layers)
    )


@pytest.mark.parametrize("collection", ["double", "single"])
def test_flux_views_keep_frozen_blocks(collection):
    tree = _written("flux_debug").materialize()
    field = f"{collection}_blocks"
    frozen = FrozenConfig(inner=getattr(tree, field)[0])
    copied = tree._copy(**{field: [frozen]})
    assert getattr(copied, f"placed_{collection}_blocks")()[0] is frozen


def test_vision_view_requires_placement_then_preserves_identity():
    config = _written("qwen3_vl_debug")
    with pytest.raises(ValueError, match="vision_encoder.layers.0"):
        config.vision_encoder.placed_layers()
    tree = config.materialize()
    vision = tree.vision_encoder
    assert all(
        layer is source for layer, source in zip(vision.placed_layers(), vision.layers)
    )


def test_checked_proxy_view_preserves_nested_wrappers_and_rejects_other_types():
    inner = LinearConfig(in_features=2, out_features=3)
    frozen = FrozenConfig(inner=FrozenConfig(inner=inner))
    assert placed_configs([frozen], LinearConfig, path="test")[0] is frozen
    with pytest.raises(ValueError, match=r"test.0: expected placed LinearConfig"):
        placed_configs([object()], LinearConfig, path="test")
