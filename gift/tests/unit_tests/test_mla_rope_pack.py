import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from torchtitan.config.schema.models.common.attention.softmax.mla import MLAConfig
from torchtitan.models.common.attention.softmax.mla_rope_pack import MLARoPEPack
from torchtitan.models.common.embeddings.rope import apply_rotary_emb_complex


@pytest.mark.parametrize("heads,seqlen", ((32, 4096), (64, 8192), (5, 17)))
def test_compiled_mla_dispatch_uses_projection_shapes(monkeypatch, heads, seqlen):
    monkeypatch.setenv("GIFT_MLA_FUSED_ROPE", "1")
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    # Dispatch only needs projection metadata. Fake CUDA has no GPU RNG.
    monkeypatch.setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    calls = []
    dim, nope, rope, value, kv_rank = 48, 12, 8, 16, 24

    def fused(q, k, k_rope, cache, positions):
        calls.append((q.shape, k.shape, k_rope.shape, cache.shape, positions.shape))
        return q.new_empty(q.shape), q.new_empty(q.shape)

    class InnerAttention(torch.nn.Module):
        def forward(self, q, k, v, **kwargs):
            assert q.shape == k.shape
            assert kwargs["attention_masks"] is mask
            return v.new_empty(v.shape)

    monkeypatch.setattr(MLARoPEPack, "apply", staticmethod(fused))
    mask = object()
    with torch.no_grad(), FakeTensorMode(), torch.device("cuda"):
        model = MLAConfig(
            dim=dim,
            n_heads=heads,
            q_lora_rank=24,
            kv_lora_rank=kv_rank,
            qk_nope_head_dim=nope,
            qk_rope_head_dim=rope,
            v_head_dim=value,
            use_mla_scaling_factor=True,
            yarn_factor=4.0,
        ).build()
        model.inner_attention = InnerAttention()
        x = torch.empty(2, seqlen, dim)
        cache = torch.empty(seqlen + 1, rope // 2, dtype=torch.complex64)
        positions = torch.empty(1, seqlen, dtype=torch.int64)
        assert model(x, cache, mask, positions).shape == x.shape

    assert calls == [
        (
            (2, seqlen, heads, nope + rope),
            (2, seqlen, heads, nope),
            (2, seqlen, rope),
            (seqlen + 1, rope // 2),
            (1, seqlen),
        )
    ]


def _reference(q, k, shared_key, cache, positions):
    nope = k.shape[-1]
    q_rope, shared_key = apply_rotary_emb_complex(
        q[..., nope:], shared_key.unsqueeze(2), freqs_cis=cache, positions=positions
    )
    return (
        torch.cat((q[..., :nope], q_rope), dim=-1),
        torch.cat((k, shared_key.expand(-1, -1, q.shape[2], -1)), dim=-1),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA Triton")
@pytest.mark.parametrize(
    ("seqlen", "heads", "nope", "rope", "position_mode", "dtype"),
    (
        (4096, 32, 128, 64, "batch", torch.bfloat16),
        (8192, 64, 128, 64, "implicit", torch.bfloat16),
        (17, 5, 11, 6, "shared", torch.float16),
        (19, 7, 0, 66, "batch", torch.float32),
    ),
)
def test_compiled_mla_rope_pack_matches_reference(
    seqlen, heads, nope, rope, position_mode, dtype
):
    batch = 2
    generator = torch.Generator(device="cuda").manual_seed(2048)

    def sample(*shape):
        return torch.randn(shape, device="cuda", dtype=dtype, generator=generator)

    # Use projection-like strides and offsets, including padded batch strides.
    q = sample(batch, seqlen + 1, heads, nope + rope)[:, :seqlen]
    k = sample(batch, seqlen, heads, nope + 13)[..., :nope]
    shared_key = sample(batch, seqlen, 24 + rope)[..., -rope:]
    theta = torch.randn((seqlen + 3, rope), device="cuda", generator=generator)
    cache = torch.polar(torch.ones_like(theta), theta)[:, ::2]
    if position_mode == "implicit":
        positions = None
    elif position_mode == "shared":
        positions = torch.arange(-1, seqlen - 1, device="cuda")[None, :]
    else:
        positions = torch.arange(2 * (seqlen + 2), device="cuda", dtype=torch.int32)
        positions = (positions % (seqlen + 3)).view(batch, -1)[:, 1 : seqlen + 1]
    grad_outputs = (sample(*q.shape), sample(*q.shape))
    if position_mode == "shared":
        # Cover views whose physical storage differs from their logical values.
        cache = cache.conj()
        grad_outputs = (
            sample(*q.shape[:-1], 2 * q.shape[-1])[..., ::2],
            sample(*q.shape[:-1], 2 * q.shape[-1])[..., 1::2],
        )

    def run(fn):
        leaves = tuple(t.detach().requires_grad_() for t in (q, k, shared_key))
        output = torch.compile(fn, fullgraph=True)(*leaves, cache, positions)
        return output, torch.autograd.grad(output, leaves, grad_outputs)

    expected, expected_grads = run(_reference)

    def pack(*inputs):
        return MLARoPEPack.apply(*inputs)

    actual, actual_grads = run(pack)
    torch.testing.assert_close(actual, expected)
    for output in actual:
        assert output.is_contiguous()
    assert actual_grads[2].shape == shared_key.shape
    # The FP32 reduction order can shift the result by one output rounding unit.
    rtol = max(torch.finfo(dtype).eps, 2e-5)
    torch.testing.assert_close(actual_grads, expected_grads, atol=2e-5, rtol=rtol)
