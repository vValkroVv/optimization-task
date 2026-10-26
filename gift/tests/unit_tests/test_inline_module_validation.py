"""Written inline modules obey the same contracts as named block templates."""

import copy

import pytest
import yaml

from torchtitan.config import ConfigSchemaError, load_document, run_pre_submit
from torchtitan.config.build import materialize_model_config
from torchtitan.config.decode import decode
from torchtitan.config.sections import ModelSection


@pytest.mark.parametrize(
    "field", ["param_init", "sharding_config", "attention.init_role"]
)
def test_inline_decoder_fields_are_rejected_before_runtime(field):
    path = "configs/experiments/llama3_debug.yaml"
    document, _ = load_document(path)
    block = copy.deepcopy(document.data["model"]["config"]["blocks"]["dense"])
    parent = block
    parts = field.split(".")
    for part in parts[:-1]:
        parent = parent[part]
    parent[parts[-1]] = "linear" if field.endswith("init_role") else {}
    result = run_pre_submit(path, [("model.config.layers", yaml.safe_dump([block]))])
    assert not result.ok
    assert any("layers.0." + field in issue.path for issue in result.report.errors)


@pytest.mark.parametrize("layer_id", [-1, 99])
def test_inline_layer_position_is_not_silently_overwritten(layer_id):
    path = "configs/experiments/llama3_debug.yaml"
    document, _ = load_document(path)
    block = copy.deepcopy(document.data["model"]["config"]["blocks"]["dense"])
    block["layer_id"] = layer_id
    result = run_pre_submit(path, [("model.config.layers", yaml.safe_dump([block]))])
    assert not result.ok
    assert "layer_id" in str(result.report.errors)


def test_matching_inline_layer_position_is_valid_and_children_are_independent():
    document, _ = load_document("configs/experiments/llama3_debug.yaml")
    model = copy.deepcopy(document.data["model"])
    block = model["config"]["blocks"]["dense"]
    model["config"]["layers"] = [dict(block, layer_id=0), dict(block, layer_id=1)]
    tree = materialize_model_config(decode(ModelSection, model).config)
    assert [layer.layer_id for layer in tree.layers] == [0, 1]
    assert tree.layers[0].attention is not tree.layers[1].attention
    assert tree.layers[0].feed_forward is not tree.layers[1].feed_forward


@pytest.mark.parametrize(
    "experiment, field, name",
    [
        ("flux_debug.yaml", "double_blocks", "double"),
        ("flux_debug.yaml", "single_blocks", "single"),
    ],
)
def test_inline_flux_streams_are_checked(experiment, field, name):
    path = "configs/experiments/" + experiment
    document, _ = load_document(path)
    block = copy.deepcopy(document.data["model"]["config"]["blocks"][name])
    block["param_init"] = {}
    result = run_pre_submit(path, [("model.config." + field, yaml.safe_dump([block]))])
    assert not result.ok
    assert any(field + ".0.param_init" in issue.path for issue in result.report.errors)


def test_inline_vision_blocks_are_checked_even_without_placement_groups():
    path = "configs/experiments/qwen3_vl_debug.yaml"
    document, _ = load_document(path)
    vision = document.data["model"]["config"]["vision_encoder"]
    block = copy.deepcopy(vision["blocks"]["vit"])
    block["sharding_config"] = {}
    result = run_pre_submit(
        path, [("model.config.vision_encoder.layers", yaml.safe_dump([block] * 4))]
    )
    assert not result.ok
    assert any(
        "vision_encoder.layers.0.sharding_config" in issue.path
        for issue in result.report.errors
    )


@pytest.mark.parametrize("keep_both", [False, True])
def test_residual_block_requires_exactly_one_feed_forward_path(keep_both):
    document, _ = load_document("configs/experiments/gift_moe_debug.yaml")
    model = copy.deepcopy(document.data["model"])
    block = next(
        block for block in model["config"]["blocks"].values() if "moe" in block
    )
    if keep_both:
        block["feed_forward"] = {"hidden_dim": 256}
    else:
        block.pop("moe")
    with pytest.raises(ConfigSchemaError, match="exactly one"):
        decode(ModelSection, model)
