"""Supplied children obey owner dimensions and retain initialization defaults."""

import pytest

from torchtitan.config import ConfigSchemaError, load_document, run_pre_submit
from torchtitan.config.build import prepare_document
from torchtitan.config.schema.models.common.norms.fused_gated_rmsnorm import (
    FusedRMSNormGatedConfig,
)
from torchtitan.config.schema.models.common.norms.rmsnorm import RMSNormConfig
from torchtitan.config.schema.protocols.module import Normal, TruncNormal, Zeros


def _prepared(recipe, overrides=()):
    document, _ = load_document(f"configs/experiments/{recipe}.yaml", list(overrides))
    return prepare_document(document).model_config


@pytest.mark.parametrize(
    "recipe,path,value",
    [
        ("flux_debug", "img_in", "{in_features: 1, out_features: 1536}"),
        (
            "flux_debug",
            "time_in_config",
            "{in_dim: 256, hidden_dim: 1536, "
            "in_layer: {in_features: 1, out_features: 1536}, "
            "out_layer: {in_features: 1536, out_features: 1536}}",
        ),
        (
            "flux_debug",
            "final_layer_config",
            "{hidden_size: 1536, patch_size: 1, out_channels: 64, "
            "linear: {in_features: 1, out_features: 64}, "
            "adaln_linear: {in_features: 1536, out_features: 3072}}",
        ),
        (
            "flux_debug",
            "blocks.double.img_mod",
            "{double: true, lin: {in_features: 1, out_features: 9216}}",
        ),
        (
            "flux_debug",
            "blocks.double.img_mod",
            "{double: false, lin: {in_features: 1536, out_features: 4608}}",
        ),
        (
            "qwen3_vl_debug",
            "vision_encoder.blocks.vit.attn.qkv",
            "{in_features: 1, out_features: 768}",
        ),
    ],
)
def test_pre_submit_rejects_child_disagreement(recipe, path, value):
    result = run_pre_submit(
        f"configs/experiments/{recipe}.yaml", [(f"model.config.{path}", value)]
    )
    assert not result.ok
    assert any("mismatch" in issue.message for issue in result.report.issues)


def test_restated_flux_time_embedder_keeps_default_initializers():
    original = _prepared("flux_debug").time_in_config
    restated = _prepared(
        "flux_debug",
        [
            (
                "model.config.time_in_config",
                "{in_dim: 256, hidden_dim: 1536, "
                "in_layer: {in_features: 256, out_features: 1536, bias: true}, "
                "out_layer: {in_features: 1536, out_features: 1536, bias: true}}",
            )
        ],
    ).time_in_config
    assert original == restated


def test_flux_partial_initializer_inherits_bias_default():
    model = _prepared(
        "flux_debug",
        [("model.config.img_in", "{bias: true, param_init: {weight: {type: zeros}}}")],
    )
    assert model.img_in.in_features == 64
    assert model.img_in.out_features == model.hidden_size
    assert model.img_in.param_init == {"weight": Zeros(), "bias": Zeros()}


@pytest.mark.parametrize(
    "recipe,path",
    [
        ("flux_debug", "img_in"),
        ("flux_debug", "img_in.param_init"),
        ("qwen3_vl_debug", "vision_encoder.blocks.vit.attn.qkv"),
    ],
)
def test_restated_projection_must_keep_owner_bias(recipe, path):
    # A bool has no unset: a restatement without ``bias: true`` would drop it.
    value = "{weight: {type: zeros}}" if path.endswith("param_init") else "{}"
    with pytest.raises(
        ConfigSchemaError,
        match=r"requires bias: true \(got false\)",
    ):
        _prepared(recipe, [(f"model.config.{path}", value)])


@pytest.mark.parametrize("value", [".nan", ".inf", "-.inf", "-1", "0"])
def test_pre_submit_rejects_invalid_norm_epsilon(value):
    result = run_pre_submit(
        "configs/experiments/llama3_debug.yaml", [("model.config.norm.eps", value)]
    )
    assert not result.ok
    assert any("finite and positive" in issue.message for issue in result.report.issues)


@pytest.mark.parametrize("factory", [RMSNormConfig, FusedRMSNormGatedConfig])
def test_python_norm_config_rejects_invalid_epsilon(factory):
    with pytest.raises(ValueError, match="finite and positive"):
        factory(eps=float("nan"))


@pytest.mark.parametrize(
    "value",
    [
        "{type: normal, std: -1}",
        "{type: normal, std: .inf}",
        "{type: normal, std: 0.02, mean: .nan}",
        "{type: trunc_normal, std: 0}",
        "{type: trunc_normal, std: 1, a: 2, b: 1}",
    ],
)
def test_pre_submit_rejects_invalid_initializer(value):
    result = run_pre_submit(
        "configs/experiments/llama3_debug.yaml",
        [
            (
                "model.config.blocks.dense.feed_forward.w1.param_init",
                "{weight: " + value + "}",
            )
        ],
    )
    assert not result.ok


def test_valid_degenerate_normal_and_truncated_normal_are_supported():
    assert Normal(std=0).std == 0
    assert TruncNormal(std=0.02).std == 0.02


def test_decoder_norm_explicit_width_must_agree():
    result = run_pre_submit(
        "configs/experiments/llama3_debug.yaml",
        [("model.config.norm.normalized_shape", "128")],
    )
    assert not result.ok
    assert any("norm mismatches" in issue.message for issue in result.report.issues)


def test_mtp_partial_children_are_completed_without_overwriting_explicit_dim():
    from copy import deepcopy
    from torchtitan.config.decode import decode
    from torchtitan.config.sections import ModelSection

    document, _ = load_document("configs/experiments/llama3_debug.yaml", [])
    model = deepcopy(document.data["model"])
    head = {
        "block": model["config"]["blocks"]["dense"],
        "input_proj": {"bias": True},
        "token_norm": {"eps": 1e-5},
    }
    model["config"]["mtp"] = {"layers": [head]}
    tree = decode(ModelSection, model).config.materialize()
    completed = tree.mtp.layers[0]
    assert (completed.input_proj.in_features, completed.input_proj.out_features) == (
        2 * tree.dim,
        tree.dim,
    )
    assert completed.token_norm.normalized_shape == tree.dim
    assert completed.token_norm.eps == 1e-5
    head["dim"] = tree.dim // 2
    with pytest.raises(ValueError, match="dim"):
        decode(ModelSection, model).config.materialize()


@pytest.mark.parametrize(
    "path,typo",
    [
        ("model.config.blocks.dense.attention", "dimm"),
        ("model.config.time_in_config.in_layer", "in_featurs"),
    ],
)
def test_owner_filled_keys_are_suggested(path, typo):
    from torchtitan.config.decode import decode
    from torchtitan.config.schema.models.common.attention.softmax.gqa import (
        GQAttentionConfig,
    )
    from torchtitan.config.schema.models.common.nn_modules import LinearConfig

    schema = LinearConfig if typo == "in_featurs" else GQAttentionConfig
    with pytest.raises(ConfigSchemaError) as error:
        decode(schema, {typo: 256}, path=path)
    message = str(error.value)
    expected = "in_features" if typo == "in_featurs" else "dim"
    assert f"Did you mean '{expected}'" in message
    assert f"{expected} (filled by owner)" in message


def test_flux_embedder_owns_completion_without_a_model():
    from torchtitan.config.schema.models.common.nn_modules import LinearConfig
    from torchtitan.config.schema.models.flux.model.layers import MLPEmbedderConfig

    shared = LinearConfig(bias=True)
    config = MLPEmbedderConfig(
        in_dim=16, hidden_dim=32, in_layer=shared, out_layer=shared
    )
    assert (config.in_layer.in_features, config.in_layer.out_features) == (16, 32)
    assert (config.out_layer.in_features, config.out_layer.out_features) == (32, 32)
    assert config.in_layer is not config.out_layer
    assert shared.in_features is None and shared.param_init is None


def test_flux_final_layer_owns_completion_without_a_model():
    from torchtitan.config.schema.models.flux.model.layers import LastLayerConfig

    config = LastLayerConfig(hidden_size=32, patch_size=2, out_channels=4)
    assert (config.linear.in_features, config.linear.out_features) == (32, 16)
    assert config.adaln_linear.out_features == 64


def test_partial_gdn_conv_is_completed_and_keeps_owner_bias():
    from torchtitan.config.schema.models.common.attention.linear.gated_delta import (
        GatedDeltaAttentionConfig,
    )
    from torchtitan.config.schema.models.common.nn_modules import Conv1dConfig

    kwargs = dict(
        dim=64,
        num_k_heads=2,
        num_v_heads=2,
        head_k_dim=16,
        head_v_dim=16,
        conv_size=4,
        o_norm=FusedRMSNormGatedConfig(),
    )
    qkv = GatedDeltaAttentionConfig(**kwargs).qkv
    for partial in (
        qkv._copy(conv=Conv1dConfig(bias=False)),
        qkv._copy(conv=qkv.conv._copy(kernel_size=None)),
        qkv._copy(qkv_proj=qkv.qkv_proj._copy(out_features=None)),
    ):
        done = GatedDeltaAttentionConfig(**kwargs, qkv=partial).qkv
        assert done == qkv
    for conv, message in (
        (Conv1dConfig(), r"requires bias: false \(got true\)"),
        (Conv1dConfig(bias=False, stride=2), r"requires stride: 1 \(got 2\)"),
    ):
        with pytest.raises(ValueError, match=message):
            GatedDeltaAttentionConfig(**kwargs, qkv=qkv._copy(conv=conv))
