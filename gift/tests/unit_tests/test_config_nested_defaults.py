"""gift_4 norms are plain: an explicit zero-centered choice is kept, not overwritten,
and the HF export contract rejects the mixed norm stack it makes."""

import copy

import pytest

from torchtitan.config.cli import load_document
from torchtitan.config.decode import decode
from torchtitan.config.sections import ModelSection


def _gift4_data():
    document, _ = load_document("configs/experiments/gift_4_debug.yaml")
    return copy.deepcopy(document.data["model"])


def test_gift4_preset_norms_are_plain():
    data = _gift4_data()
    before = copy.deepcopy(data)
    config = decode(ModelSection, data).config
    assert not config.blocks["mla_moe"].attention.kv_norm.zero_centered
    assert not config.blocks["gdn_dense"].attention.o_norm.zero_centered
    assert config.blocks["mla_moe"].attention.q_norm is None
    assert data == before
    materialized = config.materialize()
    assert not materialized.layers[2].attention.kv_norm.zero_centered


@pytest.mark.parametrize(
    "block,norm,message",
    [
        ("mla_moe", "kv_norm", "norm class differs"),
        ("gdn_dense", "o_norm", "GDN gating type differs"),
    ],
)
def test_explicit_zero_centered_choice_is_kept_and_rejected_by_hf_export(
    block, norm, message
):
    data = _gift4_data()
    data["config"]["blocks"][block]["attention"][norm]["zero_centered"] = True
    config = decode(ModelSection, data).config
    assert getattr(config.blocks[block].attention, norm).zero_centered
    issues = config.materialize().validate_hf_export()
    assert any(message in issue.message for issue in issues), issues


def test_gift35_norms_remain_zero_centered():
    document, _ = load_document("configs/experiments/gift_3_5_debug.yaml")
    config = decode(ModelSection, copy.deepcopy(document.data["model"])).config
    assert config.blocks["mla_moe"].attention.kv_norm.zero_centered
    assert config.blocks["gdn_dense"].attention.o_norm.zero_centered
