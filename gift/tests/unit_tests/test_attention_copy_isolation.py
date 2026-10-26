"""Completing one placement must not specialize its shared input template."""

from dataclasses import field

import pytest

from torchtitan.config.configurable import config
from torchtitan.config.provenance import SourceMap
from torchtitan.config.schema.components.lora import FrozenConfig
from torchtitan.config.schema.models.common.attention.softmax.gate import (
    SigmoidGateConfig,
)
from torchtitan.config.schema.models.common.attention.softmax.gqa import (
    GQAttentionConfig,
)
from torchtitan.config.schema.models.common.attention.softmax.mla import MLAConfig
from torchtitan.config.schema.models.common.attention.linear.gdn import (
    GatedDeltaNetConfig,
)
from torchtitan.config.schema.models.common.attention.linear.kda import (
    KimiDeltaAttentionConfig,
)
from torchtitan.config.schema.models.common.attention.qkv import (
    FusedQKVLinearConfig,
    LinearAttnQKVConfig,
    QKVLinearConfig,
)
from torchtitan.config.schema.models.common.nn_modules import Conv1dConfig, LinearConfig


@pytest.mark.parametrize("kind", [GQAttentionConfig, MLAConfig])
@pytest.mark.parametrize("frozen", [False, True])
def test_gate_template_can_complete_at_two_independent_widths(kind, frozen):
    gate = SigmoidGateConfig()
    template = kind(n_heads=4, gate=FrozenConfig(inner=gate) if frozen else gate)
    first = template._copy(dim=64)
    second = template._copy(dim=128)
    assert template.dim is None and gate.gate_proj is None
    assert first.gate is not template.gate and second.gate is not first.gate
    assert first.gate.gate_proj.in_features == 64
    assert second.gate.gate_proj.in_features == 128
    assert isinstance(first.gate, FrozenConfig) == frozen


@pytest.mark.parametrize("kind", [GatedDeltaNetConfig, KimiDeltaAttentionConfig])
def test_linear_attention_does_not_fill_shared_qkv_in_place(kind):
    qkv = LinearAttnQKVConfig(
        head_k_dim=8,
        head_v_dim=8,
        num_k_heads=2,
        num_v_heads=2,
        qkv_proj=LinearConfig(out_features=48),
        conv=Conv1dConfig(
            in_channels=48,
            out_channels=48,
            kernel_size=4,
            groups=48,
            padding=3,
            bias=False,
        ),
    )
    template = kind(
        num_k_heads=2, num_v_heads=2, head_k_dim=8, head_v_dim=8, conv_size=4, qkv=qkv
    )
    first = template._copy(dim=64)
    second = template._copy(dim=128)
    assert qkv.qkv_proj.in_features is None
    assert first.qkv is not qkv and second.qkv is not first.qkv
    assert first.qkv.in_features == 64 and second.qkv.in_features == 128


@pytest.mark.parametrize("kind", [QKVLinearConfig, FusedQKVLinearConfig])
def test_qkv_completion_preserves_metadata_and_init_false_state(kind):
    @config
    class AnnotatedQKV(kind):
        marker: int = field(default=0, init=False)

    template = AnnotatedQKV()
    template.marker = 17
    template._source_path = "model.config.blocks.dense.attention.qkv_linear"
    template._source_fields = SourceMap({"head_dim": "model.config.rope.dim"})
    template._source_origins = SourceMap({"model.config.rope.dim": "preset.yaml"})
    template._explicit_param_init_names = frozenset({"weight"})
    placed = template.completed(
        name="qkv", dim=64, n_heads=4, n_kv_heads=2, head_dim=16, extra=None
    )
    assert placed.marker == 17
    assert placed._source_path == template._source_path
    assert placed._source_fields == template._source_fields
    assert placed._source_origins is template._source_origins
    assert placed._explicit_param_init_names == frozenset({"weight"})
    assert template.head_dim is None


def test_frozen_copy_forwards_changes_without_modifying_either_original():
    inner = LinearConfig(in_features=2, out_features=3)
    inner._source_path = "original"
    wrapper = FrozenConfig(inner=FrozenConfig(inner=inner))
    wrapper._source_path = "wrapper"
    copied = wrapper._copy(out_features=5)
    assert isinstance(copied, FrozenConfig) and isinstance(copied.inner, FrozenConfig)
    assert copied.out_features == 5 and inner.out_features == 3
    assert copied._source_path == "wrapper"
    assert copied.inner.inner._source_path == "original"
    replacement = LinearConfig(in_features=7, out_features=8)
    swapped = wrapper._copy(inner=replacement, out_features=9)
    assert swapped.in_features == 7 and swapped.out_features == 9
    assert replacement.out_features == 8
