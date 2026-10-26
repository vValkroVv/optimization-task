"""Canonical model fields have one spelling; removed YAML aliases fail loudly."""

import copy

import pytest

from torchtitan.config import ConfigSchemaError, load_raw_document
from torchtitan.config.decode import decode
from torchtitan.config.sections import ModelSection


def test_canonical_preset_builds_the_same_attention():
    document = load_raw_document("configs/presets/models/deepseek_v3/debugmodel.yaml")
    compact = decode(ModelSection, document.data["model"]).config
    tree = compact.materialize()
    assert (
        tree.layers[0].attention.n_heads
        == compact.blocks["mla_dense"].attention.n_heads
    )


@pytest.mark.parametrize(
    "field,value",
    [("mla", {}), ("mtp_mixer", "ffn"), ("varlen", False), ("logit_softcap", 30.0)],
)
def test_removed_fields_are_unknown(field, value):
    document = load_raw_document(
        "configs/presets/models/gift_moe/gift_3_5/debugmodel.yaml"
    )
    data = copy.deepcopy(document.data["model"])
    data["config"][field] = value
    with pytest.raises(
        ConfigSchemaError, match=f"unknown key.*{field}|{field}.*unknown key"
    ):
        decode(ModelSection, data)
