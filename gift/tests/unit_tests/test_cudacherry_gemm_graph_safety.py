# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU/FakeTensor coverage for the task-local dense FP8 GEMM adapter.

The CPU dispatcher kernels below are temporary test doubles. They exercise the
real adapter and public CudaCherry GEMM path without providing a CPU fallback in
the training source or claiming CUDA GEMM/capture parity. Compiled ``out=``
support is not asserted: the installed CudaCherry GEMM documents that path as
unsupported, and the selected preset uses the functional call route.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture(scope="module", autouse=True)
def _forbid_cuda_initialization():
    """Make every test in this module safe to run on a host with visible GPUs."""
    originals = {
        "is_available": torch.cuda.is_available,
        "device_count": torch.cuda.device_count,
        "init": torch.cuda.init,
        "_lazy_init": torch.cuda._lazy_init,
    }

    def forbidden(*args, **kwargs):
        raise AssertionError("this CPU/FakeTensor test must not initialize CUDA")

    torch.cuda.is_available = lambda: False
    torch.cuda.device_count = lambda: 0
    torch.cuda.init = forbidden
    torch.cuda._lazy_init = forbidden
    try:
        yield
    finally:
        for name, value in originals.items():
            setattr(torch.cuda, name, value)


@pytest.fixture(scope="module")
def _api(_forbid_cuda_initialization):
    from cudacherry.gemm import gemm as vendor_gemm
    from torchtitan.distributed import activation_checkpoint
    from torchtitan.ops.float8 import _cudacherry, _cudacherry_graph_safe
    from torchtitan.ops.float8.cuda_gemm import FP8Gemm
    from torchtitan.ops.float8.fp8_tensor import InputLayoutType

    return SimpleNamespace(
        activation_checkpoint=activation_checkpoint,
        adapter=_cudacherry_graph_safe,
        fp8_api=_cudacherry,
        fp8_gemm=FP8Gemm,
        layout=InputLayoutType,
        vendor_gemm=vendor_gemm,
    )


@pytest.fixture
def _cpu_gemm_kernels(_api):
    """Register removable CPU doubles for the two CUDA-only dispatch paths."""
    from cudacherry.gemm.gemm import _ext

    _ext()
    events = []

    def vendor_gemm_cpu(
        a_data,
        b_data,
        out,
        a_scale,
        b_scale,
        transa,
        transb,
        a_block,
        b_block,
        a_layout,
        b_layout,
        out_dtype,
        alpha,
        beta,
        use_split_accumulator,
        allow_tf32,
        workspace_size,
    ):
        events.append(
            {
                "a_scale": a_scale,
                "b_scale": b_scale,
                "a_block": a_block,
                "b_block": b_block,
                "a_layout": a_layout,
                "b_layout": b_layout,
                "transa": transa,
                "transb": transb,
                "out": out,
                "out_dtype": out_dtype,
                "alpha": alpha,
                "beta": beta,
                "use_split_accumulator": use_split_accumulator,
                "allow_tf32": allow_tf32,
                "workspace_size": workspace_size,
                "a_shape": tuple(a_data.shape),
                "b_shape": tuple(b_data.shape),
            }
        )
        if a_layout == 1 and b_layout == 1:
            # Public wgrad operands are columnwise X then dY.
            product = b_data @ a_data.transpose(-2, -1)
        else:
            # Forward and dgrad use rowwise A and the stored B weight layout.
            product = a_data @ b_data.transpose(-2, -1)
        result = alpha * product
        if out is not None:
            if beta:
                result = result + beta * out
            out.copy_(result.to(dtype=out.dtype))
            return out
        return result.to(dtype=out_dtype or torch.bfloat16)

    vendor_lib = torch.library.Library("cudacherry", "IMPL")
    titan_lib = torch.library.Library("torchtitan", "IMPL")

    def functional_gemm_cpu(
        a_data,
        a_scale,
        a_layout,
        a_block_scaling_dim,
        b_data,
        b_scale,
        b_layout,
        b_block_scaling_dim,
        out_dtype=None,
        alpha=1.0,
        accumulate=False,
        use_split_accumulator=True,
        allow_tf32=True,
        workspace_size=32 * 1024 * 1024,
    ):
        return _api.adapter._fp8_dense_gemm_impl(
            a_data,
            a_scale,
            a_layout,
            a_block_scaling_dim,
            b_data,
            b_scale,
            b_layout,
            b_block_scaling_dim,
            None,
            out_dtype,
            alpha,
            accumulate,
            use_split_accumulator,
            allow_tf32,
            workspace_size,
        )

    vendor_lib.impl("gemm", vendor_gemm_cpu, "CPU", allow_override=False)
    titan_lib.impl(
        "fp8_dense_gemm", functional_gemm_cpu, "CPU", allow_override=False
    )
    titan_lib.impl(
        "fp8_dense_gemm_mutate",
        _api.adapter._fp8_dense_gemm_mutate_impl,
        "CPU",
        allow_override=False,
    )
    try:
        yield events
    finally:
        titan_lib._destroy()
        vendor_lib._destroy()


def _operand(data, layout, block=1):
    return data, torch.ones((1, 1), dtype=torch.float32), int(layout), block


def _inputs(layout):
    if layout == "row_row":
        a = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        b = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        return _operand(a, 0, 1), _operand(b, 0, 2)
    if layout == "row_column":
        a = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        b = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        return _operand(a, 0, 1), _operand(b, 1, 2)
    a = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    b = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    return _operand(a, 1, 2), _operand(b, 1, 2)


def test_wrapper_matches_public_signature_forwards_layouts_and_defaults(
    _api, _cpu_gemm_kernels
):
    actual = inspect.signature(_api.adapter.fp8_gemm)
    expected = inspect.signature(_api.vendor_gemm)
    assert tuple(actual.parameters) == tuple(expected.parameters)
    assert {
        name: parameter.default for name, parameter in actual.parameters.items()
    } == {
        name: parameter.default for name, parameter in expected.parameters.items()
    }

    expected_transposes = {
        "row_row": (False, True),
        "row_column": (False, False),
        "column_column": (True, False),
    }
    for pair, transpose_flags in expected_transposes.items():
        a, b = _inputs(pair)
        result = _api.fp8_api.fp8_gemm(
            a,
            b,
            out_dtype=torch.float64,
            alpha=0.25,
            use_split_accumulator=False,
            allow_tf32=False,
            workspace_size=12345,
        )
        event = _cpu_gemm_kernels[-1]
        assert event["transa"] == transpose_flags[0]
        assert event["transb"] == transpose_flags[1]
        assert event["a_layout"] == a[2]
        assert event["b_layout"] == b[2]
        assert event["a_block"] == a[3]
        assert event["b_block"] == b[3]
        assert event["a_scale"] is a[1]
        assert event["b_scale"] is b[1]
        assert event["out"] is None
        assert event["out_dtype"] is torch.float64
        assert event["alpha"] == 0.25 and event["beta"] == 0.0
        assert event["use_split_accumulator"] is False
        assert event["allow_tf32"] is False
        assert event["workspace_size"] == 12345
        assert result.dtype is torch.float64
        assert result.data_ptr() not in {a[0].data_ptr(), b[0].data_ptr()}

    a, b = _inputs("row_row")
    _api.fp8_api.fp8_gemm(a, b)
    event = _cpu_gemm_kernels[-1]
    assert event["out_dtype"] is None
    assert event["alpha"] == 1.0 and event["beta"] == 0.0
    assert event["use_split_accumulator"] is True
    assert event["allow_tf32"] is True
    assert event["workspace_size"] == 32 * 1024 * 1024

    invalid_a = (a[0], a[1], 7, a[3])
    with pytest.raises(ValueError, match="Unsupported input layout pair"):
        _api.fp8_api.fp8_gemm(invalid_a, b)
    with pytest.raises(ValueError, match="accumulate=True requires"):
        _api.fp8_api.fp8_gemm(a, b, accumulate=True)


def test_mutable_route_preserves_out_identity_overwrite_and_accumulate(
    _api, _cpu_gemm_kernels
):
    a, b = _inputs("row_row")
    product = a[0] @ b[0].transpose(-2, -1)
    storage = torch.full((2, 8), 9.0, dtype=torch.float32)
    out = storage[:, ::2]
    assert out.shape == (2, 4) and out.stride() == (8, 2)

    version = out._version
    result = _api.fp8_api.fp8_gemm(
        a,
        b,
        out=out,
        out_dtype=torch.float32,
        alpha=0.5,
        accumulate=False,
    )
    assert result is out
    assert out._version > version
    assert out.stride() == (8, 2)
    torch.testing.assert_close(out, 0.5 * product)
    assert _cpu_gemm_kernels[-1]["out"] is out
    assert _cpu_gemm_kernels[-1]["beta"] == 0.0

    out.fill_(3.0)
    version = out._version
    result = _api.fp8_api.fp8_gemm(
        a,
        b,
        out=out,
        out_dtype=torch.float32,
        alpha=0.25,
        accumulate=True,
    )
    assert result is out
    assert out._version > version
    torch.testing.assert_close(out, 3.0 + 0.25 * product)
    assert _cpu_gemm_kernels[-1]["out"] is out
    assert _cpu_gemm_kernels[-1]["beta"] == 1.0


def test_fake_tensor_metadata_comes_from_vendor_meta(_api, _cpu_gemm_kernels):
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    shape_env = ShapeEnv()
    rows = shape_env.create_unbacked_symint()
    with FakeTensorMode(shape_env=shape_env):
        a_data = torch.empty(
            (rows, 128), device="cuda", dtype=torch.float8_e4m3fn
        ).as_strided((rows, 128), (256, 2))
        a_scale = torch.empty((rows, 1), device="cuda", dtype=torch.float32)
        b_data = torch.empty_strided(
            (256, 128), (1, 256), device="cuda", dtype=torch.float8_e4m3fn
        )
        b_scale = torch.empty((2, 1), device="cuda", dtype=torch.float32)
        a = (a_data, a_scale, 0, 1)
        b = (b_data, b_scale, 0, 2)

        result = _api.fp8_api.fp8_gemm(a, b, out_dtype=torch.float32)
        assert tuple(result.shape) == (rows, 256)
        assert result.dtype is torch.float32
        assert result.device.type == "cuda"
        assert result.stride() == (256, 1)

        out = torch.empty_strided(
            (rows, 256), (512, 2), device="cuda", dtype=torch.float32
        )
        result_out = _api.fp8_api.fp8_gemm(a, b, out=out, out_dtype=torch.float32)
        assert result_out is out
        assert result_out.stride() == (512, 2)

    for op in (
        torch.ops.torchtitan.fp8_dense_gemm.default,
        torch.ops.torchtitan.fp8_dense_gemm_mutate.default,
    ):
        assert torch.Tag.cudagraph_unsafe in op.tags
        assert torch.Tag.out not in op.tags

    mutate_schema = torch.ops.torchtitan.fp8_dense_gemm_mutate.default._schema
    mutable_out = next(argument for argument in mutate_schema.arguments if argument.name == "out")
    assert mutable_out.alias_info is not None
    assert mutable_out.alias_info.is_write
    assert len(mutate_schema.returns) == 0


def test_compiled_functional_call_keeps_tagged_outer_target(
    _api, _cpu_gemm_kernels
):
    a, b = _inputs("row_row")
    graphs = []

    def record_graph(graph_module, _example_inputs):
        graphs.append(graph_module)
        return graph_module.forward

    def functional_call(a_data, a_scale, b_data, b_scale):
        return _api.fp8_api.fp8_gemm(
            (a_data, a_scale, 0, 1), (b_data, b_scale, 0, 2), out_dtype=torch.float32
        )

    compiled = torch.compile(functional_call, backend=record_graph, fullgraph=True)
    result = compiled(a[0], a[1], b[0], b[1])
    torch.testing.assert_close(result, a[0] @ b[0].transpose(-2, -1))
    functional_target = torch.ops.torchtitan.fp8_dense_gemm.default
    functional_nodes = [
        node for node in graphs[-1].graph.nodes if node.target == functional_target
    ]
    assert len(functional_nodes) == 1
    assert torch.Tag.cudagraph_unsafe in functional_nodes[0].target.tags

def _install_exact_quantizer_doubles(monkeypatch, api):
    layout = api.layout

    def scale_for(data, block):
        if block == 1:
            return torch.ones(
                (data.shape[0], max(1, (data.shape[-1] + 127) // 128)),
                dtype=torch.float32,
                device=data.device,
            )
        return torch.ones(
            (
                max(1, (data.shape[-2] + 127) // 128),
                max(1, (data.shape[-1] + 127) // 128),
            ),
            dtype=torch.float32,
            device=data.device,
        )

    def quantize_vec(data, *, pack):
        payload = data.detach().clone()
        return (payload, scale_for(payload, 1), layout.ROWWISE, 1)

    def quantize_block(data, *, transpose_output=False, pack=False):
        payload = (
            data.detach().transpose(-2, -1).contiguous()
            if transpose_output
            else data.detach().clone()
        )
        selected_layout = layout.COLUMNWISE if transpose_output else layout.ROWWISE
        return (payload, scale_for(payload, 2), selected_layout, 2)

    def transpose_operand(operand, *, pack=False):
        data, scale, _old_layout, block = api.adapter.unpack_operand(operand, "operand")
        return (
            data.detach().transpose(-2, -1).contiguous(),
            scale.detach().clone(),
            layout.COLUMNWISE,
            block,
        )

    monkeypatch.setattr(api.fp8_api, "quantize_vec128", quantize_vec)
    monkeypatch.setattr(api.fp8_api, "quantize_blk128x128", quantize_block)
    monkeypatch.setattr(api.fp8_api, "scaling_aware_fp8_transpose", transpose_operand)


def test_actual_fp8_gemm_keeps_manual_gradient_ownership(_api, _cpu_gemm_kernels, monkeypatch):
    _install_exact_quantizer_doubles(monkeypatch, _api)
    x = torch.randn((128, 128), dtype=torch.float32, requires_grad=True)
    weight = torch.randn((128, 128), dtype=torch.float32, requires_grad=True)
    cotangent = torch.randn((128, 128), dtype=torch.float32)

    output = _api.fp8_gemm.apply(x, weight, torch.float32, True, tuple(x.shape))
    torch.testing.assert_close(output, x.detach() @ weight.detach().transpose(-2, -1))
    output.backward(cotangent)
    torch.testing.assert_close(x.grad, cotangent @ weight.detach())
    torch.testing.assert_close(weight.grad, cotangent.transpose(-2, -1) @ x.detach())
    assert len(_cpu_gemm_kernels) == 3
    assert [
        (event["a_layout"], event["b_layout"]) for event in _cpu_gemm_kernels
    ] == [(0, 0), (0, 1), (1, 1)]


def test_fp8_gemm_residual_and_disabled_gradient_routes(_api, _cpu_gemm_kernels, monkeypatch):
    _install_exact_quantizer_doubles(monkeypatch, _api)
    x = torch.randn((128, 128), dtype=torch.float32, requires_grad=True)
    weight = torch.randn((128, 128), dtype=torch.float32, requires_grad=True)
    residual = torch.randn((128, 128), dtype=torch.float32, requires_grad=True)
    residual_before = residual.detach().clone()
    cotangent = torch.randn_like(residual)

    output = _api.fp8_gemm.apply(
        x, weight, torch.float32, True, tuple(x.shape), residual
    )
    torch.testing.assert_close(
        output, residual_before + x.detach() @ weight.detach().transpose(-2, -1)
    )
    output.backward(cotangent)
    torch.testing.assert_close(x.grad, cotangent @ weight.detach())
    torch.testing.assert_close(
        weight.grad, cotangent.transpose(-2, -1) @ x.detach()
    )
    torch.testing.assert_close(residual.grad, cotangent)
    assert _cpu_gemm_kernels[0]["beta"] == 1.0

    _cpu_gemm_kernels.clear()
    frozen_weight = torch.randn((128, 128), dtype=torch.float32)
    x = torch.randn((128, 128), dtype=torch.float32, requires_grad=True)
    output = _api.fp8_gemm.apply(
        x, frozen_weight, torch.float32, True, tuple(x.shape)
    )
    output.sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(output) @ frozen_weight)
    assert frozen_weight.grad is None
    assert len(_cpu_gemm_kernels) == 2
    assert _cpu_gemm_kernels[-1]["a_layout"] == 0
    assert _cpu_gemm_kernels[-1]["b_layout"] == 1

    _cpu_gemm_kernels.clear()
    x = torch.randn((128, 128), dtype=torch.float32)
    weight = torch.randn((128, 128), dtype=torch.float32, requires_grad=True)
    output = _api.fp8_gemm.apply(
        x, weight, torch.float32, False, tuple(x.shape)
    )
    output.sum().backward()
    torch.testing.assert_close(
        weight.grad, torch.ones_like(output).transpose(-2, -1) @ x
    )
    assert x.grad is None
    assert len(_cpu_gemm_kernels) == 2
    assert _cpu_gemm_kernels[-1]["a_layout"] == 1
    assert _cpu_gemm_kernels[-1]["b_layout"] == 1


def test_actual_selective_checkpoint_recomputes_dense_gemm(_api, _cpu_gemm_kernels, monkeypatch):
    from torch import nn
    from torchtitan.config import ActivationCheckpointConfig

    _install_exact_quantizer_doubles(monkeypatch, _api)

    class GemmBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn((128, 128), dtype=torch.float32))

        def forward(self, x):
            return _api.fp8_gemm.apply(
                x, self.weight, torch.float32, True, tuple(x.shape)
            )

    block_module = GemmBlock()
    weight = block_module.weight
    block = _api.activation_checkpoint._apply_op_sac(
        block_module,
        ActivationCheckpointConfig(
            mode="selective",
            per_op_sac_force_recompute_mm_shapes_by_fqns=[],
            early_stop=False,
        ),
    )
    x = torch.randn((128, 128), dtype=torch.float32, requires_grad=True)
    output = block(x)
    output.sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(output) @ weight.detach())
    torch.testing.assert_close(
        weight.grad, torch.ones_like(output).transpose(-2, -1) @ x.detach()
    )

    save_ops = _api.activation_checkpoint._get_save_ops()
    assert torch.ops.cudacherry.gemm.default not in save_ops
    assert torch.ops.torchtitan.fp8_dense_gemm.default not in save_ops
    assert torch.ops.torchtitan.fp8_dense_gemm_mutate.default not in save_ops
    events = [
        (event["a_layout"], event["b_layout"], event["a_shape"], event["b_shape"])
        for event in _cpu_gemm_kernels
    ]
    assert len(events) == 4
    assert events[0] == events[1]
    assert events[2][:2] == (0, 1)
    assert events[3][:2] == (1, 1)


def test_compiled_fp8_gemm_retains_manual_backward(_api, _cpu_gemm_kernels, monkeypatch):
    _install_exact_quantizer_doubles(monkeypatch, _api)

    def loss_fn(x, weight):
        return _api.fp8_gemm.apply(
            x, weight, torch.float32, True, tuple(x.shape)
        ).square().mean()

    x = torch.randn((128, 128), dtype=torch.float32, requires_grad=True)
    weight = torch.randn((128, 128), dtype=torch.float32, requires_grad=True)
    eager_loss = loss_fn(x, weight)
    eager_grads = torch.autograd.grad(eager_loss, (x, weight))

    compiled_loss_fn = torch.compile(loss_fn, backend="aot_eager", fullgraph=True)
    compiled_loss = compiled_loss_fn(x, weight)
    compiled_grads = torch.autograd.grad(compiled_loss, (x, weight))
    torch.testing.assert_close(compiled_loss, eager_loss)
    torch.testing.assert_close(compiled_grads[0], eager_grads[0])
    torch.testing.assert_close(compiled_grads[1], eager_grads[1])

