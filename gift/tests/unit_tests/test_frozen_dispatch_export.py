"""Freeze preserves component dispatch, module names and HF projection weights."""

from types import SimpleNamespace

import pytest
import torch

from torchtitan.config import load_raw_document
from torchtitan.config.build import build_model_spec
from torchtitan.config.configs import ParallelismConfig
from torchtitan.config.decode import decode
from torchtitan.config.sections import ModelSection
from torchtitan.config.schema.components.freeze import FreezeConverterConfig
from torchtitan.config.schema.components.lora import FrozenConfig, unwrap_frozen
from torchtitan.config.schema.models.common.attention.qkv import FusedQKVLinearConfig


_GQA_PRESETS = [
    "llama3/debugmodel.yaml",
    "qwen3/debugmodel.yaml",
    "llama4/debugmodel.yaml",
    "gpt_oss/debugmodel.yaml",
    "qwen3_vl/debugmodel.yaml",
]
_MOE_PRESETS = [
    "deepseek_v3/debugmodel.yaml",
    "qwen3/debugmodel_moe.yaml",
    "llama4/debugmodel.yaml",
    "gpt_oss/debugmodel.yaml",
    "gift_moe/debugmodel.yaml",
    "gift_moe/gift_3_5/debugmodel.yaml",
]


def _spec(preset, selectors=(), *, fused=False):
    raw = load_raw_document(f"configs/presets/models/{preset}")
    section = decode(ModelSection, raw.data["model"])
    if fused:
        for block in section.config.blocks.values():
            block.attention.qkv_linear = FusedQKVLinearConfig()
    section.config.converters = (
        [FreezeConverterConfig(selectors=list(selectors))] if selectors else []
    )
    return build_model_spec(section)


def _prepare_and_build(spec):
    spec.model.owner().prepare_config(
        spec.model,
        config=SimpleNamespace(
            parallelism=ParallelismConfig(enable_sequence_parallel=False)
        ),
    )
    with torch.device("meta"):
        return spec.model.build()


@pytest.mark.parametrize("preset", _GQA_PRESETS)
@pytest.mark.parametrize(
    "selector", ["layers.0.attention.qkv_linear", "layers.*.attention", "layers.0"]
)
def test_frozen_gqa_containers_prepare_and_build_without_changing_names(
    preset, selector
):
    spec = _spec(preset, [selector], fused=True)
    model = _prepare_and_build(spec)
    with torch.device("meta"):
        baseline = _spec(preset, fused=True).model.build()
    assert model.state_dict().keys() == baseline.state_dict().keys()
    frozen = [
        p
        for name, p in model.named_parameters()
        if name.startswith(selector.replace("*", "0") + ".")
    ]
    assert frozen and all(not p.requires_grad for p in frozen)
    assert any(p.requires_grad for p in model.parameters())


@pytest.mark.parametrize("preset", _MOE_PRESETS)
def test_frozen_moe_containers_prepare_and_build(preset):
    model = _prepare_and_build(_spec(preset, ["layers.*.moe"]))
    experts = [p for name, p in model.named_parameters() if ".moe." in name]
    assert experts and all(not p.requires_grad for p in experts)
    assert any(
        p.requires_grad for name, p in model.named_parameters() if ".moe." not in name
    )


@pytest.mark.parametrize(
    "preset", _GQA_PRESETS + ["deepseek_v3/debugmodel.yaml", "gift_moe/debugmodel.yaml"]
)
def test_frozen_inner_kernels_preserve_mask_metadata(preset):
    spec = _spec(preset, ["layers.*.attention.inner_attention"])
    model = _prepare_and_build(spec)
    assert (
        model.attention_config.attention_mask_kind
        == spec.model.attention.attention_mask_kind
    )
    for layer in spec.model.placed_layers():
        kernel = layer.attention.inner_attention
        assert isinstance(kernel, FrozenConfig)
        assert (
            type(unwrap_frozen(kernel)).mask_kind
            == model.attention_config.attention_mask_kind
        )


@pytest.mark.parametrize("preset", _GQA_PRESETS)
@pytest.mark.parametrize(
    "selector", ["layers.0.attention.qkv_linear", "layers.*.attention"]
)
def test_frozen_fused_qkv_exports_every_projection_and_round_trips(preset, selector):
    spec = _spec(preset, [selector], fused=True)
    adapter = spec.state_dict_adapter(spec.model, None)
    assert adapter.fuse_qkv
    projection = spec.model.placed_layers()[0].attention.qkv_linear.wqkv
    n_in, n_out = projection.require_features()
    key = "layers.0.attention.qkv_linear.wqkv.weight"
    native = {key: torch.arange(n_out * n_in, dtype=torch.float32).reshape(n_out, n_in)}
    # Tied-head adapters recreate the output head from the embedding. These
    # unrelated weights only supply that contract for this projection test.
    shared = torch.ones(2, 6)
    native.update({"tok_embeddings.weight": shared, "lm_head.weight": shared})
    if projection.bias:
        native[key.removesuffix("weight") + "bias"] = torch.arange(
            n_out, dtype=torch.float32
        )
    hf = adapter.to_hf(native)
    for name in ("q_proj.weight", "k_proj.weight", "v_proj.weight"):
        assert any(k.endswith(name) for k in hf), (preset, name, hf.keys())
    restored = adapter.from_hf(hf.copy())
    assert restored.keys() == native.keys()
    for name, value in native.items():
        torch.testing.assert_close(restored[name], value, rtol=0, atol=0)


@pytest.mark.parametrize("preset", _GQA_PRESETS)
def test_frozen_fused_qkv_export_ignores_legacy_rope_buffers(preset):
    spec = _spec(preset, fused=True)
    adapter = spec.state_dict_adapter(spec.model, None)
    with torch.device("meta"):
        # Expert biases are unrepresentable too; not under test here.
        state = {
            k: v
            for k, v in spec.model.build().state_dict().items()
            if "expert_bias" not in k
        }
    state["layers.0.attention.rotary_emb.inv_freq"] = torch.empty(1)
    adapter.check_hf_export(state)


@pytest.mark.parametrize(
    "preset",
    [
        "deepseek_v3/debugmodel.yaml",
        "gift_moe/debugmodel.yaml",
        "gift_moe/gift_3_5/debugmodel.yaml",
    ],
)
def test_frozen_mla_and_hybrid_attention_prepare_and_build(preset):
    model = _prepare_and_build(_spec(preset, ["layers.*.attention"]))
    attention = [p for name, p in model.named_parameters() if ".attention." in name]
    assert attention and all(not p.requires_grad for p in attention)


def test_frozen_llama4_block_still_builds_its_chunked_masks(monkeypatch):
    import torchtitan.models.llama4.model as llama4

    model = _prepare_and_build(_spec("llama4/debugmodel.yaml", ["layers.0"]))
    calls = []
    monkeypatch.setattr(
        llama4, "create_attention_mask", lambda *args: calls.append(args) or args
    )
    masks = model.get_attention_masks(torch.arange(8).unsqueeze(0))
    assert masks.keys() == {"rope", "nope"}
    assert len(calls) == 2


@pytest.mark.parametrize(
    "selector",
    [
        "layers.0.attention.qkv_linear",
        "layers.0.attention",
        "layers.0.attention.inner_attention",
        "layers.0",
    ],
)
def test_cpu_forward_and_unfrozen_gradients_match_baseline(selector):
    from tests.unit_tests.config_helpers import load_model_spec

    tiny = {
        "dim": 64,
        "vocab_size": 128,
        "rope": {"dim": 16, "max_seq_len": 32},
        "blocks": {
            "dense": {
                "attention": {
                    "n_heads": 4,
                    "n_kv_heads": 2,
                    "qkv_linear": {"type": "fused"},
                },
                "feed_forward": {"hidden_dim": 128},
            }
        },
        "layers": [{"block": "dense", "repeat": 2}],
    }
    baseline = load_model_spec("llama3/debugmodel.yaml", {"config": tiny}).model.build()
    baseline.init_weights()
    frozen = load_model_spec(
        "llama3/debugmodel.yaml",
        {
            "config": {
                **tiny,
                "converters": [{"type": "freeze", "selectors": [selector]}],
            }
        },
    ).model.build()
    frozen.load_state_dict(baseline.state_dict())
    tokens = torch.tensor([[0, 1, 2, 3], [4, 3, 2, 1]])
    expected, actual = baseline(tokens), frozen(tokens)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    expected.sum().backward()
    actual.sum().backward()
    baseline_parameters = dict(baseline.named_parameters())
    for name, parameter in frozen.named_parameters():
        if parameter.requires_grad:
            torch.testing.assert_close(
                parameter.grad, baseline_parameters[name].grad, rtol=0, atol=0
            )
        else:
            assert parameter.grad is None
