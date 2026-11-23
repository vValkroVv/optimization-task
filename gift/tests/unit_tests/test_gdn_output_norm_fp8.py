"""Parity with FLA gated RMSNorm and the native CudaCherry FP8 projection."""

import importlib.util

import pytest
import torch
from torch import nn

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
    pytest.skip("CudaCherry requires an SM90 CUDA GPU", allow_module_level=True)
if importlib.util.find_spec("cudacherry") is None:
    pytest.skip("CudaCherry is required", allow_module_level=True)

from torchtitan.config import ActivationCheckpointConfig  # noqa: E402
from torchtitan.distributed.activation_checkpoint import _apply_ac_to_block  # noqa: E402
from torchtitan.models.common.attention.linear.gdn import GatedDeltaNet  # noqa: E402
from torchtitan.models.common.attention.linear.fla_custom_ops import (  # noqa: E402
    fla_vjp,
    gated_rmsnorm_op,
)
from torchtitan.models.common.attention.linear.gdn_output_fp8 import (  # noqa: E402
    gdn_output_norm_fp8,
)
from torchtitan.models.common.attention.linear.gdn_output_norm_backward import (  # noqa: E402
    gdn_output_norm_backward,
)
from torchtitan.ops.float8._cudacherry import quantize_vec128  # noqa: E402
from torchtitan.ops.float8.cuda_gemm import FP8Linear  # noqa: E402
from torchtitan.ops.float8.fp8_utils import pad_tensor_if_needed  # noqa: E402


def _inputs(batch, seq_len, heads, head_dim):
    rng = torch.Generator(device="cuda").manual_seed(310)
    shape = (batch, seq_len, heads, head_dim)
    x = torch.randn(shape, device="cuda", generator=rng).bfloat16()
    gate = torch.randn(shape, device="cuda", generator=rng).bfloat16()
    weight = torch.randn(head_dim, device="cuda", generator=rng).bfloat16()
    x[0, 0].zero_()
    return x, gate, weight


def _reference(x, gate, weight, eps, activation, zero_centered, multiplier):
    normed = gated_rmsnorm_op(x, gate, weight, eps, activation, zero_centered)
    return (normed * multiplier).flatten(2)


class _Projection(nn.Module):
    def __init__(self, projection, norm_args, fused):
        super().__init__()
        self.projection = projection
        self.norm_args = norm_args
        self.fused = fused

    def forward(self, x, gate, weight):
        norm = gdn_output_norm_fp8 if self.fused else _reference
        return self.projection(norm(x, gate, weight, *self.norm_args))


def _forward_backward(module, inputs, projection, cotangent):
    inputs = tuple(t.detach().requires_grad_(True) for t in inputs)
    out = module(*inputs)
    assert out.shape == cotangent.shape
    return out.detach(), *torch.autograd.grad(
        out, (*inputs, projection.weight), cotangent
    )


def _projection(heads, head_dim):
    config = FP8Linear.Config(
        in_features=heads * head_dim, out_features=143, bias=False
    )
    return FP8Linear(config).to(device="cuda", dtype=torch.bfloat16)


def _assert_exact(actual, expected):
    for result, ref in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, ref, atol=0, rtol=0)


def _assert_payload_matches(inputs, norm_args):
    expected = quantize_vec128(
        pad_tensor_if_needed(_reference(*inputs, *norm_args)), pack=True
    )
    candidate = gdn_output_norm_fp8(*inputs, *norm_args)
    data, scale = candidate._data, candidate._scale
    assert data.stride() == expected._data.stride()
    assert scale.stride() == expected._scale.stride()
    assert torch.equal(data.view(torch.uint8), expected._data.view(torch.uint8))
    assert torch.equal(scale.view(torch.int32), expected._scale.view(torch.int32))


@pytest.mark.parametrize(
    "shape,multiplier,activation,zero_centered",
    [
        ((2, 64, 4, 128), 2.0, "sigmoid", True),
        ((2, 67, 4, 128), -0.75, "swish", False),
        ((1, 137, 4, 32), 1.25, "sigmoid", True),
        ((2, 64, 3, 48), 1.37, "silu", False),
        ((2, 71, 3, 96), 0.25, "swish", True),
        ((2, 75, 3, 65), -1.375, "sigmoid", True),
        ((1, 129, 2, 192), 1.375, "sigmoid", False),
        ((1, 127, 3, 513), 0.75, "silu", True),
    ],
    ids=[
        "donor",
        "tokens",
        "subheads",
        "features",
        "crossing",
        "odd-width",
        "wide",
        "autotuned",
    ],
)
def test_payload_and_projection(shape, multiplier, activation, zero_centered):
    batch, seq_len, heads, head_dim = shape
    inputs = _inputs(*shape)
    norm_args = (1e-6, activation, zero_centered, multiplier)
    _assert_payload_matches(inputs, norm_args)

    projection = _projection(heads, head_dim)
    cotangent = torch.randn(
        batch, seq_len, projection.out_features, device="cuda"
    ).bfloat16()
    native = _forward_backward(
        _Projection(projection, norm_args, fused=False), inputs, projection, cotangent
    )
    fused = _forward_backward(
        _Projection(projection, norm_args, fused=True), inputs, projection, cotangent
    )
    _assert_exact(fused, native)


def test_strided_activation_and_mixed_precision_gate(production_compile_flags):
    def strided(tensor):
        output = tensor.new_empty((*tensor.shape[:-1], tensor.shape[-1] * 2))[..., ::2]
        return output.copy_(tensor)

    x, gate, weight = _inputs(2, 67, 3, 96)
    inputs = strided(x), strided(gate.float() / 7), strided(weight)
    norm_args = (1e-6, "sigmoid", True, 1.375)
    _assert_payload_matches(inputs, norm_args)
    projection = _projection(*x.shape[-2:])
    cotangent = torch.randn(2, 67, projection.out_features, device="cuda").bfloat16()
    native = _forward_backward(
        _Projection(projection, norm_args, fused=False), inputs, projection, cotangent
    )
    fused = _forward_backward(
        _Projection(projection, norm_args, fused=True), inputs, projection, cotangent
    )
    _assert_exact(fused, native)
    compiled = torch.compile(
        _Projection(projection, norm_args, fused=True), fullgraph=True
    )
    _assert_exact(_forward_backward(compiled, inputs, projection, cotangent), native)


@pytest.fixture
def production_compile_flags(monkeypatch):
    monkeypatch.setattr(torch._dynamo.config, "capture_scalar_outputs", True)
    monkeypatch.setattr(
        torch._dynamo.config, "skip_fwd_side_effects_in_bwd_under_checkpoint", True
    )


@pytest.mark.parametrize("grad_dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_direct_backward_preserves_rounding_with_strided_padded_cotangent(grad_dtype):
    batch, seq_len, heads, width = 2, 67, 3, 96
    x = torch.zeros((batch, seq_len, heads, width), device="cuda", dtype=torch.bfloat16)
    x[-1, -1, -1, 0] = 1
    gate = torch.full_like(x, 0.125, dtype=torch.float32)
    weight = torch.full_like(gate[0, 0, 0], 0.125)
    # Nonzero padding makes an incorrect logical stride visible in backward.
    incoming = torch.ones((batch, 128, 768), device="cuda", dtype=grad_dtype)[..., ::2]
    incoming[:, :seq_len, : heads * width] = 1.0078125
    eps, activation, centered, multiplier = 1e-6, "sigmoid", True, 1.375

    logical = incoming[:, :seq_len, : heads * width].reshape(x.shape)
    reference = fla_vjp(
        "gated_rmsnorm",
        [x, gate, weight],
        [logical * multiplier],
        [],
        [eps, centered],
        activation,
    )
    direct = torch.compile(gdn_output_norm_backward, fullgraph=True)
    _assert_exact(
        direct(x, gate, weight, incoming, eps, activation, centered, multiplier),
        reference,
    )


def test_compiled_projection_handles_changing_padded_shapes(production_compile_flags):
    heads, head_dim = 3, 96
    norm_args = (1e-6, "sigmoid", True, 1.375)
    projection = _projection(heads, head_dim)
    baseline = _Projection(projection, norm_args, fused=False)
    checkpointed = _apply_ac_to_block(
        _Projection(projection, norm_args, fused=True),
        ActivationCheckpointConfig(mode="selective", selective_save=[]),
        mode="selective",
    )
    candidate = torch.compile(
        checkpointed, fullgraph=True, dynamic=True, options={"triton.cudagraphs": False}
    )
    for seq_len in (67, 149, 64):
        inputs = _inputs(2, seq_len, heads, head_dim)
        cotangent = torch.randn(
            2, seq_len, projection.out_features, device="cuda"
        ).bfloat16()
        _assert_exact(
            _forward_backward(candidate, inputs, projection, cotangent),
            _forward_backward(baseline, inputs, projection, cotangent),
        )


def test_compiled_gdn_handles_changing_padded_shapes(
    monkeypatch, production_compile_flags
):
    monkeypatch.setenv("GIFT_GDN_DIRECT_FLA_BACKWARD", "0")
    monkeypatch.setenv("GIFT_GDN_FUSE_SHORT_CONV_BWD", "0")
    config = GatedDeltaNet.Config(
        dim=128,
        num_k_heads=1,
        num_v_heads=3,
        head_k_dim=32,
        head_v_dim=96,
        conv_size=4,
        o_norm_output_multiplier=1.375,
        o_proj=FP8Linear.Config(in_features=288, out_features=128, bias=False),
    )

    def build(fused):
        monkeypatch.setenv("GIFT_GDN_FUSE_OUTPUT_NORM_FP8", "1" if fused else "0")
        module = config.build().to(device="cuda", dtype=torch.bfloat16)
        module.A_log = nn.Parameter(module.A_log.float())
        module.dt_bias = nn.Parameter(module.dt_bias.float())
        return module

    baseline = build(False)
    with torch.no_grad():
        baseline.A_log.fill_(1.0)
        baseline.dt_bias.zero_()
    fused = build(True)
    fused.load_state_dict(baseline.state_dict())
    assert fused._output_norm_fp8 is gdn_output_norm_fp8

    def compile_with_sac(module):
        checkpointed = _apply_ac_to_block(
            module,
            ActivationCheckpointConfig(mode="selective", selective_save=[]),
            mode="selective",
        )
        return torch.compile(
            checkpointed,
            fullgraph=True,
            dynamic=True,
            options={"triton.cudagraphs": False},
        )

    candidate = compile_with_sac(fused)
    control = compile_with_sac(baseline)
    rope = torch.empty(0, device="cuda")

    def run(module, parameters, x, cotangent):
        x = x.detach().requires_grad_(True)
        out = module(x, rope, None)
        assert out.shape == x.shape
        return out.detach(), *torch.autograd.grad(out, (x, *parameters), cotangent)

    for seq_len in (67, 149):
        x = torch.randn(2, seq_len, 128, device="cuda").bfloat16()
        cotangent = torch.randn_like(x)
        _assert_exact(
            run(candidate, tuple(fused.parameters()), x, cotangent),
            run(control, tuple(baseline.parameters()), x, cotangent),
        )

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as profile:
        run(candidate, tuple(fused.parameters()), x, cotangent)
    assert any(
        "_gdn_output_norm_quantize_blocks_kernel" in event.name
        for event in profile.events()
    )

    seen = []

    def observe_projection(module, args):
        if module is fused.o_proj:
            seen.append(args[0])

    with fused.o_proj.register_forward_pre_hook(observe_projection):
        fused(x, rope, None)
    with nn.modules.module.register_module_forward_pre_hook(observe_projection):
        fused(x, rope, None)
    assert len(seen) == 2
    assert all(type(tensor) is torch.Tensor for tensor in seen)
    assert all(tensor.shape == (*x.shape[:2], 288) for tensor in seen)

    norm_grad_shapes = []

    def observe_norm_backward(_module, grad_inputs, _grad_outputs):
        norm_grad_shapes.append(grad_inputs[0].shape)

    with fused.o_norm.register_full_backward_hook(observe_norm_backward):
        run(fused, tuple(fused.parameters()), x, cotangent)
    assert norm_grad_shapes == [(*x.shape[:2], 3, 96)]
