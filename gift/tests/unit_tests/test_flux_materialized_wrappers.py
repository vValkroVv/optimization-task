"""Transparent freeze wrappers preserve the completed FLUX block contract."""

import pytest

from torchtitan.config import deep_merge, load_raw_document
from torchtitan.config.build import materialize_model_config
from torchtitan.config.decode import decode
from torchtitan.config.schema.components.lora import FrozenConfig
from torchtitan.config.sections import ModelSection


def _tree():
    document = load_raw_document("configs/experiments/flux_debug.yaml")
    return decode(ModelSection, document.data["model"]).config.materialize()


@pytest.mark.parametrize("stream", ["double_blocks", "single_blocks"])
def test_frozen_stream_blocks_survive_copy_and_materialization(stream):
    tree = _tree()
    original = getattr(tree, stream)[0]
    getattr(tree, stream)[0] = FrozenConfig(inner=original)
    for copied in (tree._copy(), tree.materialize(), materialize_model_config(tree)):
        wrapped = getattr(copied, stream)[0]
        assert isinstance(wrapped, FrozenConfig)
        assert wrapped.inner.hidden_size == tree.hidden_size
        assert wrapped.inner.num_heads == tree.num_heads


@pytest.mark.parametrize("stream", ["double_blocks", "single_blocks"])
def test_frozen_stream_blocks_still_validate_owner_dimensions(stream):
    tree = _tree()
    block = getattr(tree, stream)[0]
    # Preserve a valid block locally while making its owner dimension disagree.
    getattr(tree, stream)[0] = FrozenConfig(inner=block)
    tree.hidden_size *= 2
    tree.num_heads *= 2
    with pytest.raises(ValueError, match="block dimensions disagree"):
        tree.materialize()


def _flux(overrides: dict):
    model = deep_merge(
        load_raw_document("configs/presets/models/flux/debug.yaml").data["model"],
        {"config": overrides},
    )
    return decode(ModelSection, model).config.materialize()


# A restated projection: the user wrote the field but left widths and init unset.
_RESTATED = {
    "img_in": {"bias": True},
    "blocks": {
        "double": {"img_mlp_in": {"bias": True}},
        "single": {"linear2": {"bias": True}},
    },
}


def test_a_restated_projection_keeps_its_owners_init():
    default, restated = _flux({}), _flux(_RESTATED)
    for get in (
        lambda t: t.img_in,
        lambda t: t.double_blocks[0].img_mlp_in,
        lambda t: t.single_blocks[0].linear2,
    ):
        assert get(restated) == get(default)
        assert get(restated).param_init is not None


@pytest.mark.parametrize(
    ("overrides", "where"),
    [
        ({"img_in": {"in_features": 63}}, "img_in"),
        ({"txt_in": {"out_features": 7}}, "txt_in"),
        ({"blocks": {"double": {"img_mlp_out": {"out_features": 7}}}}, "img_mlp_out"),
        # A bool has no unset: a restatement without `bias: true` drops the bias.
        ({"img_in": {}}, "img_in"),
        ({"blocks": {"double": {"img_mlp_in": {"in_features": 1536}}}}, "img_mlp_in"),
    ],
)
def test_a_projection_of_the_wrong_width_is_rejected(overrides, where):
    with pytest.raises(ValueError, match=f"{where} mismatches"):
        _flux(overrides)
