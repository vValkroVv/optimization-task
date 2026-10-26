"""Output projections use typed components with parent-completed dimensions."""

import copy

import pytest

from torchtitan.config.cli import load_document
from torchtitan.config.decode import decode
from torchtitan.config.errors import ConfigSchemaError
from torchtitan.config.schema.models.common.embeddings.token_debeddings import (
    UnembeddingConfig,
)
from torchtitan.config.schema.models.common.nn_modules import LinearConfig
from torchtitan.config.sections import ModelSection


def model_data(experiment="llama3_debug.yaml"):
    document, _ = load_document("configs/experiments/" + experiment)
    return copy.deepcopy(document.data["model"])


@pytest.mark.parametrize("head_type", ["linear", "unembedding"])
def test_selected_head_preserves_type_and_completes_dimensions(head_type):
    data = model_data()
    data["config"]["lm_head"] = {"type": head_type, "bias": True}
    if head_type == "unembedding":
        data["config"]["lm_head"]["logit_softcap"] = 30.0
    compact = decode(ModelSection, data).config
    original = copy.deepcopy(compact.lm_head)
    tree = compact.materialize()
    assert tree.lm_head.in_features == compact.dim
    assert tree.lm_head.out_features == compact.vocab_size
    assert tree.lm_head.bias
    assert tree.lm_head.init_role == "lm_head"
    assert isinstance(tree.lm_head, UnembeddingConfig) == (head_type == "unembedding")
    assert compact.lm_head == original


def test_default_linear_projection_decodes_without_type():
    config = decode(LinearConfig, {"in_features": 4, "out_features": 8})
    assert type(config) is LinearConfig


@pytest.mark.parametrize("field", ["in_features", "out_features"])
def test_explicit_head_width_conflict_is_rejected(field):
    data = model_data()
    data["config"]["lm_head"] = {"type": "linear", field: 17}
    with pytest.raises(ValueError, match="lm_head"):
        decode(ModelSection, data).config.materialize()


def test_unknown_head_type_is_a_user_error():
    data = model_data()
    data["config"]["lm_head"] = {"type": "unknown"}
    with pytest.raises(ConfigSchemaError, match="unknown linear type"):
        decode(ModelSection, data)


@pytest.mark.parametrize("softcap", [-1.0, float("nan"), float("inf"), -float("inf")])
def test_invalid_softcap_is_rejected_without_runtime_imports(softcap):
    data = model_data()
    data["config"]["lm_head"] = {"type": "unembedding", "logit_softcap": softcap}
    with pytest.raises(ConfigSchemaError, match="logit_softcap"):
        decode(ModelSection, data)


def test_legacy_gift_softcap_is_rejected_without_a_typed_head():
    data = model_data("gift_4_debug.yaml")
    data["config"].pop("lm_head")
    data["config"]["logit_softcap"] = 30.0
    with pytest.raises(ConfigSchemaError, match="logit_softcap: unknown key"):
        decode(ModelSection, data)


def test_legacy_and_explicit_gift_head_conflict_is_rejected():
    data = model_data("gift_4_debug.yaml")
    data["config"]["logit_softcap"] = 40.0
    with pytest.raises(ConfigSchemaError, match="logit_softcap: unknown key"):
        decode(ModelSection, data)


def test_tied_unembedding_head_preserves_weight_alias_and_bias_initialization():
    import torch

    from torchtitan.config.build import build_model_spec
    from torchtitan.models.common.embeddings.token_debeddings import Unembedding

    data = model_data()
    data["config"]["blocks"]["dense"]["attention"]["n_heads"] = 4
    data["config"].update(
        dim=32,
        layers=[{"block": "dense"}],
        vocab_size=64,
        enable_weight_tying=True,
        rope={"type": "complex", "dim": 8, "max_seq_len": 8},
        lm_head={"type": "unembedding", "logit_softcap": 30.0, "bias": True},
    )
    spec = build_model_spec(decode(ModelSection, data))
    with torch.device("meta"):
        model = spec.model.build()
    assert isinstance(model.lm_head, Unembedding)
    assert model.lm_head.logit_softcap == 30.0
    assert model.tok_embeddings.weight is model.lm_head.weight
    model.to_empty(device="cpu")
    model.init_states(buffer_device=torch.device("cpu"))
    assert model.tok_embeddings.weight is model.lm_head.weight
    assert model.lm_head.bias is not model.lm_head.weight
    assert torch.count_nonzero(model.lm_head.bias).item() == 0
