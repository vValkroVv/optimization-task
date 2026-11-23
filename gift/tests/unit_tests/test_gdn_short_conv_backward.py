from functools import partial
from itertools import accumulate

import pytest
import torch

from torchtitan.models.common.attention.linear.fla_custom_ops import (
    qkv_conv1d_unpack_op,
)


def _check_crop_input_gradient(actual, expected, qkv, weight, grad_outputs, eps):
    """Check BF16 cancellation outliers against local FP64 convolution."""
    bf16_eps = torch.finfo(torch.bfloat16).eps
    close = torch.isclose(actual, expected, rtol=bf16_eps, atol=3e-5)
    failures = int((~close).count_nonzero())
    assert failures <= 32, (
        f"{failures} input gradients exceed the elementwise tolerance"
    )
    taps = weight.shape[-1]
    num_heads, head_dim = grad_outputs[0].shape[-2:]
    num_v_heads, head_v_dim = grad_outputs[2].shape[-2:]
    group_width = 2 * head_dim + (num_v_heads // num_heads) * head_v_dim
    length = qkv.shape[1]
    checks = []
    for batch, token, channel in (~close).nonzero().tolist():
        group, offset = divmod(channel, group_width)
        role, feature = divmod(offset, head_dim)
        assert role < 2, "Outlier is outside the normalized Q/K features"
        head_start = group * group_width + role * head_dim
        head = slice(head_start, head_start + head_dim)
        first_input = max(0, token - taps + 1)
        last_output = min(length, token + taps)
        features = qkv[batch, first_input:last_output, head].to(
            device="cpu", dtype=torch.float64
        )
        weights = (
            weight[head].reshape(head_dim, taps).to(device="cpu", dtype=torch.float64)
        )
        grad = grad_outputs[role][batch, token:last_output, group].to(
            device="cpu", dtype=torch.float64
        )
        contributions = []
        for shift, output_token in enumerate(range(token, last_output)):
            first = max(0, output_token - taps + 1)
            window = features[first - first_input : output_token - first_input + 1]
            preactivation = (window * weights[:, -len(window) :].T).sum(0)
            silu = (preactivation * preactivation.sigmoid()).to(torch.bfloat16).double()
            rstd = (silu.square().sum() + eps).rsqrt()
            norm = (silu * rstd).to(torch.bfloat16).double()
            dactivation = (grad[shift] - norm * (grad[shift] * norm).sum()) * rstd
            dactivation = dactivation[feature].to(torch.bfloat16).double()
            pre_silu = preactivation[feature].to(torch.bfloat16).double()
            sigmoid = pre_silu.sigmoid()
            dsilu = sigmoid * (1 + pre_silu * (1 - sigmoid))
            contributions.append(
                dactivation * dsilu * weights[feature, taps - 1 - shift]
            )
        contributions = torch.stack(contributions)
        oracle = float(contributions.sum())
        # A one-ULP shift in a BF16-rounded contribution scales with the
        # magnitude before the sum. Opposite-sign taps can nearly cancel.
        bound = bf16_eps * (float(contributions.abs().sum()) + abs(oracle))
        expected_error = abs(float(expected[batch, token, channel]) - oracle)
        actual_error = abs(float(actual[batch, token, channel]) - oracle)
        assert max(expected_error, actual_error) <= bound, (
            (batch, token, channel),
            expected_error,
            actual_error,
            bound,
        )
        checks.append(((batch, token, channel), expected_error, actual_error, bound))
    return checks


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires native FLA on CUDA")
@pytest.mark.parametrize(
    (
        "batch",
        "lengths",
        "heads",
        "widths",
        "taps",
        "dtype",
        "normalize",
        "strided",
        "compiled",
    ),
    (
        pytest.param(
            2,
            (4096,),
            (16, 32),
            (128, 128),
            4,
            torch.bfloat16,
            True,
            False,
            True,
            id="crop",
        ),
        pytest.param(
            2,
            (67,),
            (2, 4),
            (31, 65),
            1,
            torch.float32,
            True,
            True,
            False,
            id="single-tap",
        ),
        pytest.param(
            1,
            (0, 1, 64, 0, 65, 4),
            (2, 6),
            (63, 17),
            3,
            torch.float16,
            True,
            True,
            True,
            id="packed-fp16",
        ),
        pytest.param(
            1,
            (69, 2, 15),
            (2, 4),
            (17, 129),
            7,
            torch.bfloat16,
            True,
            False,
            False,
            id="packed-bf16",
        ),
        pytest.param(
            2,
            (73,),
            (1, 2),
            (256, 193),
            5,
            torch.bfloat16,
            False,
            True,
            False,
            id="wide-head",
        ),
        pytest.param(
            1,
            (2, 1, 140, 5),
            (1, 2),
            (11, 17),
            67,
            torch.float32,
            True,
            False,
            False,
            id="wide-taps",
        ),
    ),
)
def test_short_conv_backward_matches_native_fla(
    batch,
    lengths,
    heads,
    widths,
    taps,
    dtype,
    normalize,
    strided,
    compiled,
    record_property,
):
    from fla.ops.common.fused_conv_qkv import fused_qkv_conv1d_unpack
    from torch.utils.checkpoint import checkpoint

    from torchtitan.distributed.activation_checkpoint import _sac_context_fn

    num_heads, num_v_heads = heads
    head_dim, head_v_dim = widths
    tokens = sum(lengths)
    channels = 2 * num_heads * head_dim + num_v_heads * head_v_dim
    generator = torch.Generator(device="cuda").manual_seed(2048)

    def sample(shape, *, scale=0.5, sample_dtype=dtype):
        if strided:
            shape = (*shape[:-1], shape[-1] * 2)
        value = torch.randn(
            shape, device="cuda", dtype=sample_dtype, generator=generator
        )
        value = value * scale
        return value[..., ::2] if strided else value

    cu_seqlens = (
        torch.tensor([0, *accumulate(lengths)], device="cuda", dtype=torch.int32)
        if len(lengths) > 1
        else None
    )
    qkv = sample((batch, tokens, channels))
    # Compiled mixed precision can present FP32 weight storage to this custom op.
    weight_dtype = torch.float32 if strided else dtype
    weight_shape = (channels, taps) if strided else (channels, 1, taps)
    weight = sample(weight_shape, scale=taps**-0.5, sample_dtype=weight_dtype)
    grad_q = sample((batch, tokens, num_heads, head_dim))
    grad_k = sample((batch, tokens, num_heads, head_dim))
    grad_v = sample((batch, tokens, num_v_heads, head_v_dim))
    grad_outputs = grad_q, grad_k, grad_v
    eps = 2e-5

    baseline_inputs = qkv.detach().requires_grad_(), weight.detach().requires_grad_()
    expected = fused_qkv_conv1d_unpack(
        *baseline_inputs,
        num_heads=num_heads,
        num_v_heads=num_v_heads,
        head_dim=head_dim,
        head_v_dim=head_v_dim,
        cu_seqlens=cu_seqlens,
        normalize_qk=normalize,
        eps=eps,
    )
    expected_grads = torch.autograd.grad(expected, baseline_inputs, grad_outputs)

    def short_conv(x, w, metadata):
        return qkv_conv1d_unpack_op(
            x,
            w,
            num_heads,
            num_v_heads,
            head_dim,
            head_v_dim,
            metadata,
            normalize,
            eps,
            True,
        )

    def checkpointed(x, w, metadata):
        return checkpoint(
            short_conv,
            x,
            w,
            metadata,
            use_reentrant=False,
            context_fn=partial(_sac_context_fn, ("experts_up",), ()),
        )

    run = torch.compile(checkpointed, fullgraph=True) if compiled else short_conv
    inputs = qkv.detach().requires_grad_(), weight.detach().requires_grad_()
    actual = run(*inputs, cu_seqlens)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as profile:
        actual_grads = torch.autograd.grad(actual, inputs, grad_outputs)
    assert any(
        event.key == "torchtitan::gdn_short_conv_backward"
        for event in profile.key_averages()
    )
    for actual_output, expected_output in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_output, expected_output, rtol=0, atol=0)

    # The two Triton layouts can sum FP32 partials in a different order.
    # Allow one output-rounding step at scale plus a small absolute margin
    # for cancellation in the FP32 weight sum.
    for index, (actual_grad, expected_grad) in enumerate(
        zip(actual_grads, expected_grads, strict=True)
    ):
        assert torch.isfinite(actual_grad).all()
        if index == 0 and dtype == torch.bfloat16 and tokens == 4096:
            outliers = _check_crop_input_gradient(
                actual_grad, expected_grad, qkv, weight, grad_outputs, eps
            )
            record_property("bf16_cancellation_outliers", outliers)
            continue
        rtol = torch.finfo(actual_grad.dtype).eps
        torch.testing.assert_close(actual_grad, expected_grad, rtol=rtol, atol=3e-5)
