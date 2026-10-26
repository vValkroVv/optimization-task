import copy
import math

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from llmfoundry.models.giga_mix.configuration_giga_mix import GigaMixConfig
from llmfoundry.models.layers.attention import LlamaLatentAttention
from llmfoundry.models.layers.custom_embedding import (
    DeepseekV2YarnRotaryEmbedding,
    LlamaRotaryEmbedding,
    yarn_get_mscale,
)

DEV = "cuda"
MAX_POS = 131072
ROPE_SCALING = {
    "factor": 32.0,
    "beta_fast": 32.0,
    "beta_slow": 1.0,
    "mscale": 1.0,
    "mscale_all_dim": 1.0,
    "original_max_position_embeddings": 4096,
    "rope_type": "yarn",
}


def _config():
    return GigaMixConfig(
        vocab_size=128,
        hidden_size=1536,
        num_hidden_layers=4,
        num_attention_heads=32,
        num_key_value_heads=32,
        intermediate_size=1280,
        dense_intermediate_size=1536,
        max_position_embeddings=MAX_POS,
        rope_theta=100000,
        rope_scaling=dict(ROPE_SCALING),
        attention_type="LlamaLatentAttention",
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        kv_lora_rank=512,
        q_lora_rank=1536,
        num_krope_heads=1,
        use_custom_rotary_kernel=False,  # varlen + custom rotary kernel is rejected for MLA
        gated_attention=True,
        use_mla_scaling_factor=True,
        norm_type="ZeroCenteredGatedNorm",
        layernorm_type="pre_post",
        use_cache=False,
        varlen_input=True,
        first_k_dense_replace=4,
        num_routed_experts=8,
        num_experts_per_tok=2,
    )


@pytest.mark.unit
@pytest.mark.requires_gpu
def test_custom_yarn_matches_deepseek_reference_up_to_131k():
    config = _config()
    rope_cfg = copy.deepcopy(config)
    rope_cfg.head_dim = config.qk_rope_head_dim
    custom = LlamaRotaryEmbedding(rope_cfg, device=DEV)
    ref = DeepseekV2YarnRotaryEmbedding(
        config.qk_rope_head_dim,
        max_position_embeddings=MAX_POS,
        scaling_factor=ROPE_SCALING["factor"],
        base=config.rope_theta,
        original_max_position_embeddings=ROPE_SCALING["original_max_position_embeddings"],
        beta_fast=ROPE_SCALING["beta_fast"],
        beta_slow=ROPE_SCALING["beta_slow"],
        mscale=ROPE_SCALING["mscale"],
        mscale_all_dim=ROPE_SCALING["mscale_all_dim"],
        device=DEV,
    )
    assert torch.equal(custom.inv_freq.float(), ref.inv_freq.float())

    x = torch.zeros(1, MAX_POS, 1, config.qk_rope_head_dim, device=DEV, dtype=torch.float32)
    cos, sin = custom.forward(x, seq_len=MAX_POS)
    t = torch.arange(MAX_POS, device=DEV, dtype=torch.float32)
    freqs = torch.outer(t, ref.inv_freq.float().to(DEV))
    ratio = float(
        yarn_get_mscale(ROPE_SCALING["factor"], ROPE_SCALING["mscale"])
        / yarn_get_mscale(ROPE_SCALING["factor"], ROPE_SCALING["mscale_all_dim"])
    )
    for p in (0, 4095, 4096, 65536, MAX_POS - 1):
        assert torch.equal(cos[p].float(), freqs[p].cos() * ratio), f"cos mismatch at {p}"
        assert torch.equal(sin[p].float(), freqs[p].sin() * ratio), f"sin mismatch at {p}"


@pytest.mark.unit
@pytest.mark.requires_gpu
def test_mla_softmax_scale_follows_deepseek_yarn_convention():
    config = _config()
    layer = LlamaLatentAttention(config, layer_idx=0)
    head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
    expected = head_dim ** -0.5 * yarn_get_mscale(
        ROPE_SCALING["factor"], ROPE_SCALING["mscale_all_dim"]) ** 2
    assert math.isclose(layer.softmax_scale, expected, rel_tol=1e-9)
    assert layer.rotary_emb.rope_attention_scaling == 1.0


@pytest.mark.unit
@pytest.mark.requires_gpu
def test_dense_mla_doc_shift_invariance_at_128k():
    config = _config()
    torch.manual_seed(7)
    layer = LlamaLatentAttention(config, layer_idx=0)
    for p in layer.parameters():
        if p.dim() >= 2:
            torch.nn.init.normal_(p, std=0.02)
    layer = layer.to(DEV, torch.bfloat16).train()

    doc, prefix = 32768, MAX_POS - 32768
    gen = torch.Generator(device=DEV).manual_seed(3)
    x_doc = (torch.randn(1, doc, config.hidden_size, device=DEV, dtype=torch.float32,
                         generator=gen) * 0.5).to(torch.bfloat16)
    x_prefix = (torch.randn(1, prefix, config.hidden_size, device=DEV, dtype=torch.float32,
                            generator=gen) * 0.5).to(torch.bfloat16)

    with torch.no_grad():
        out_alone, _, _ = layer(
            x_doc,
            cu_seqlens=torch.tensor([0, doc], device=DEV, dtype=torch.int32),
            max_seqlen=doc,
        )
        out_packed, _, _ = layer(
            torch.cat([x_prefix, x_doc], dim=1),
            cu_seqlens=torch.tensor([0, prefix, MAX_POS], device=DEV, dtype=torch.int32),
            max_seqlen=prefix,
        )

    assert torch.equal(out_packed[:, prefix:], out_alone), (
        "same document at the packed tail differs from standalone "
        "(doc-relative rope or cross-doc leakage broken)"
    )


def _rotate_interleaved(x, cos, sin):
    """Reference interleaved rotary: pairs (0,1), (2,3), ... rotated by (cos, sin)."""
    x0, x1 = x[..., 0::2].float(), x[..., 1::2].float()
    c, s_ = cos.float()[:, None, :], sin.float()[:, None, :]
    return torch.stack([x0 * c - x1 * s_, x0 * s_ + x1 * c], dim=-1).flatten(-2)


@pytest.mark.unit
@pytest.mark.requires_gpu
def test_rotary_kernel_runs_at_512k_and_1m():
    """Stock flash-attn rotary places sequence blocks on CUDA grid axis y
    (65535-block cap): any sequence or single varlen document >= 524288 tokens
    failed with 'Triton Error [CUDA]: invalid argument'. custom_embedding
    installs the vendored long-seq kernel; dense MLA ropes the FULL sequence
    (after the SP all-to-all), so 512k/1M contexts depend on this."""
    import flash_attn.layers.rotary as fa_layers

    import llmfoundry.models.layers.custom_embedding  # noqa: F401  (installs patch)
    from llmfoundry.models.layers.flash_rotary_longseq import apply_rotary as vendored

    assert fa_layers.apply_rotary is vendored, "long-seq rotary patch not installed"
    for S in (524288, 1048576):
        x = torch.randn(1, S, 1, 64, device=DEV, dtype=torch.bfloat16)
        cos = torch.randn(S, 32, device=DEV, dtype=torch.bfloat16)
        sin = torch.randn(S, 32, device=DEV, dtype=torch.bfloat16)
        out = fa_layers.apply_rotary_emb(x, cos, sin, interleaved=True)
        cu = torch.tensor([0, S], device=DEV, dtype=torch.int32)
        # same convention as apply_rotary_emb above, otherwise the two paths rotate
        # different element pairs and cannot be compared to one reference
        out2 = vendored(x[0], cos, sin, cu_seqlens=cu, max_seqlen=S, interleaved=True)
        torch.cuda.synchronize()
        assert torch.isfinite(out.float()).all() and torch.isfinite(out2.float()).all()
        # isfinite alone would pass on a kernel that rotated the wrong positions, which is
        # exactly what the grid-axis cap used to do past 524288. Check the values on a
        # stride across the WHOLE sequence, so sampled positions land beyond the cap too.
        idx = torch.arange(0, S, S // 512, device=DEV)
        ref = _rotate_interleaved(x[0, idx], cos[idx], sin[idx])
        for got in (out[0, idx], out2[idx]):
            torch.testing.assert_close(got.float(), ref.float(), rtol=2e-2, atol=2e-2)
        del x, cos, sin, out, out2, ref
        torch.cuda.empty_cache()


@pytest.mark.unit
@pytest.mark.requires_gpu
@pytest.mark.parametrize("interleaved", [False, True])
def test_vendored_rotary_matches_torch_reference(interleaved):
    from llmfoundry.models.layers.flash_rotary_longseq import apply_rotary as vendored

    gen = torch.Generator(device=DEV).manual_seed(2)
    S, H, D, RD = 4096, 3, 192, 64
    x = torch.randn(2, S, H, D, device=DEV, dtype=torch.float32, generator=gen)
    cos = torch.randn(S, RD // 2, device=DEV, dtype=torch.float32, generator=gen)
    sin = torch.randn(S, RD // 2, device=DEV, dtype=torch.float32, generator=gen)
    out = vendored(x, cos, sin, interleaved=interleaved)

    ref = x.clone()
    c = cos[None, :, None, :]
    s = sin[None, :, None, :]
    if interleaved:
        x0, x1 = x[..., 0:RD:2], x[..., 1:RD:2]
        ref[..., 0:RD:2] = x0 * c - x1 * s
        ref[..., 1:RD:2] = x0 * s + x1 * c
    else:
        x0, x1 = x[..., : RD // 2], x[..., RD // 2: RD]
        ref[..., : RD // 2] = x0 * c - x1 * s
        ref[..., RD // 2: RD] = x0 * s + x1 * c
    assert torch.allclose(out, ref, rtol=0, atol=1e-6)


@pytest.mark.unit
@pytest.mark.requires_gpu
def test_flash_rotary_is_patched():
    import flash_attn.layers.rotary as layers
    import flash_attn.ops.triton.rotary as ops

    from llmfoundry.models.layers.flash_rotary_longseq import apply_rotary

    assert ops.apply_rotary is apply_rotary
    assert layers.apply_rotary is apply_rotary
