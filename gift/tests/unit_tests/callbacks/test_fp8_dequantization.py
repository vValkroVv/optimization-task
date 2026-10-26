# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Scale contracts at monitoring boundaries, without launching CUDA kernels."""

from types import SimpleNamespace

import pytest
import torch

from torchtitan.config.schema.callbacks.metrics.sample import ReduceOp
from torchtitan.callbacks.metrics import stats
from torchtitan.callbacks.metrics.reducers import ParallelReducer
from torchtitan.callbacks.metrics.sample import (
    MetricConfig,
    ReduceConfig,
    ReduceStageConfig,
    StatOpConfig,
)
from torchtitan.callbacks.sources.modelling_source import ModellingSource
from torchtitan.models.common.moe.experts.gmm import GroupedExperts
from torchtitan.ops.float8.cuda_gemm import FP8Linear
from torchtitan.ops.float8.fp8_tensor import Float8BlockwiseQTensor, InputLayoutType

dequantization = pytest.importorskip("cudacherry.quantization.dequantization")


@pytest.fixture(params=["reference", "triton_interpreter"])
def kernel_calls(monkeypatch, request):
    """Exercise adapters against a reference and the real kernel interpreted on CPU."""
    calls = []
    interpreted = None
    if request.param == "triton_interpreter":
        interpreter = pytest.importorskip("triton.runtime.interpreter")
        interpreted = interpreter.InterpretedFunction(
            dequantization._dequantization_kernel.fn
        )

    class Kernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                data, scale, out = args[0], args[5], args[10]
                assert tuple(args[6:8]) == scale.stride()
                calls.append(scale)
                if interpreted is not None:
                    interpreted[grid](*args, **kwargs)
                    return
                rows = torch.arange(data.shape[0]) // kwargs["SCALE_ROW_BLOCK"]
                blocks = torch.arange(out.shape[1]) // kwargs["BLOCK_SIZE"]
                out.copy_(
                    data[:, : out.shape[1]].float() * scale[rows[:, None], blocks]
                )

            return launch

    monkeypatch.setattr(dequantization, "_dequantization_kernel", Kernel())
    return calls


def _wrapper(shape, scale, *, block_scaling_dim=1, scale_block_first=False):
    data = (torch.arange(torch.Size(shape).numel()).reshape(shape) % 7 + 1).to(
        torch.float8_e4m3fn
    )
    return Float8BlockwiseQTensor(
        data,
        scale,
        torch.bfloat16,
        InputLayoutType.ROWWISE,
        block_scaling_dim,
        scale_block_first=scale_block_first,
    )


def _reference(wrapper, scale):
    row_block = 1 if wrapper._block_scaling_dim == 1 else 128
    expanded = scale.repeat_interleave(row_block, dim=-2).repeat_interleave(128, dim=-1)
    slices = tuple(slice(0, size) for size in wrapper.orig_shape)
    return (wrapper._data.float() * expanded).bfloat16()[slices]


@pytest.mark.parametrize("block_first", [False, True])
def test_square_activation_scales_follow_the_producer_contract(
    kernel_calls, block_first
):
    logical_scale = torch.tensor([[1.0, 2.0], [4.0, 8.0]])
    scale = logical_scale.t().contiguous() if block_first else logical_scale
    value = _wrapper((2, 256), scale, scale_block_first=block_first)

    actual = stats.dequantize_fp8(value)

    torch.testing.assert_close(actual, _reference(value, logical_scale), rtol=0, atol=0)
    assert len(kernel_calls) == 1
    assert kernel_calls[0].data_ptr() == scale.data_ptr()
    assert kernel_calls[0].stride() == (scale.t() if block_first else scale).stride()


def test_grouped_expert_scales_stay_row_first_even_when_flattened_square(kernel_calls):
    scale = torch.arange(1.0, 17.0).reshape(2, 2, 4)
    value = _wrapper((2, 256, 512), scale, block_scaling_dim=2)

    actual = stats.dequantize_fp8(value)

    torch.testing.assert_close(actual, _reference(value, scale), rtol=0, atol=0)
    assert kernel_calls[0].shape == (4, 4)
    assert kernel_calls[0].data_ptr() == scale.data_ptr()


def test_cudacherry_blockwise_weights_do_not_transpose(kernel_calls):
    scale = torch.tensor([[1.0, 2.0], [4.0, 8.0]])
    value = _wrapper((256, 256), scale, block_scaling_dim=2)

    actual = stats.dequantize_fp8(value)

    torch.testing.assert_close(actual, _reference(value, scale), rtol=0, atol=0)
    assert kernel_calls[0].stride() == scale.stride()


def test_cudacherry_scale_padding_and_original_shape_are_preserved(kernel_calls):
    scale = torch.arange(1.0, 17.0).reshape(4, 4)
    value = _wrapper((3, 256), scale, scale_block_first=True)
    value.orig_shape = (2, 129)

    actual = stats.dequantize_fp8(value)

    expected = value._data.float() * scale.t()[:3, :2].repeat_interleave(128, dim=-1)
    torch.testing.assert_close(actual, expected[:2, :129].bfloat16(), rtol=0, atol=0)


def test_batched_activation_padding_keeps_each_batchs_row_offsets(kernel_calls):
    scale = torch.arange(1.0, 17.0).reshape(2, 4, 2)
    value = _wrapper((2, 4, 256), scale)
    value.orig_shape = (2, 3, 129)

    actual = stats.dequantize_fp8(value)

    torch.testing.assert_close(actual, _reference(value, scale), rtol=0, atol=0)


@pytest.mark.parametrize("block_first", [False, True])
def test_columnwise_operands_are_rejected(kernel_calls, block_first):
    value = _wrapper((2, 256), torch.ones(2, 2), scale_block_first=block_first)
    value._input_layout_type = InputLayoutType.COLUMNWISE

    with pytest.raises(NotImplementedError, match="ROWWISE"):
        stats.dequantize_fp8(value)
    assert not kernel_calls


def test_wrong_scale_shape_is_not_automatically_transposed(kernel_calls):
    value = _wrapper((4, 256), torch.ones(2, 4))

    with pytest.raises(ValueError, match="row-first scale"):
        stats.dequantize_fp8(value)
    assert not kernel_calls


def test_a_foreign_rowwise_layout_enum_is_accepted(kernel_calls):
    """cudacherry's ROWWISE IntEnum must compare by value, not identity."""
    import enum

    class ForeignLayout(enum.IntEnum):
        ROWWISE = 0
        COLUMNWISE = 1

    scale = torch.ones(4, 1)
    for layout in (ForeignLayout.ROWWISE, 0):
        value = _wrapper((4, 128), scale)
        value._input_layout_type = layout
        torch.testing.assert_close(
            stats.dequantize_fp8(value), _reference(value, scale), rtol=0, atol=0
        )


def _source(**overrides):
    metric = MetricConfig(
        name="mean",
        pipeline=[StatOpConfig(op="mean")],
        reduce=ReduceConfig(stages=[ReduceStageConfig(group="dp", op=ReduceOp.mean)]),
    )
    config = dict(
        capture_forward=False,
        capture_inputs=True,
        capture_backward=True,
        capture_grad_inputs=True,
        metrics=[metric],
    )
    config.update(overrides)
    source = ModellingSource(ModellingSource.Config(**config))
    source.reducer = ParallelReducer(
        parallel_dims=SimpleNamespace(get_optional_mesh=lambda name: None)
    )
    source.active = True
    return source


@pytest.mark.parametrize("dense", [False, True])
@pytest.mark.parametrize("block_first", [False, True])
@pytest.mark.parametrize("role", ["input", "grad_input", "grad_output"])
def test_hooks_use_tensor_metadata_not_the_module(
    kernel_calls, dense, block_first, role
):
    module = (
        FP8Linear.Config(in_features=256, out_features=256).build()
        if dense
        else GroupedExperts.Config(dim=256, hidden_dim=256, num_experts=2).build()
    )
    logical_scale = torch.tensor([[1.0, 2.0], [4.0, 8.0]])
    scale = logical_scale.t().contiguous() if block_first else logical_scale
    value = _wrapper((2, 256), scale, scale_block_first=block_first)
    source = _source()

    if role == "input":
        source._forward_hook("layer.proj")(module, (value,), None)
    else:
        source._backward_hook("layer.proj")(
            module,
            (value,) if role == "grad_input" else (),
            (value,) if role == "grad_output" else (),
        )
    source._flush_captures()

    (record,) = source.drain()
    assert role in record.name
    assert record.value.item() == pytest.approx(
        _reference(value, logical_scale).double().mean().item()
    )
    assert len(kernel_calls) == 1


@pytest.mark.parametrize("disabled", ["inactive", "compiling", "no_metrics"])
def test_disabled_capture_does_not_dequantize(kernel_calls, monkeypatch, disabled):
    module = FP8Linear.Config(in_features=256, out_features=256).build()
    value = _wrapper((2, 256), torch.ones(2, 2))
    source = _source(metrics=[]) if disabled == "no_metrics" else _source()
    source.active = disabled != "inactive"
    if disabled == "compiling":
        monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)

    source._forward_hook("layer.proj")(module, (value,), None)

    assert not kernel_calls
    assert not source._capture_accum


def test_fp8_constructor_requires_scale_layout():
    data = torch.zeros(2, 256, dtype=torch.float8_e4m3fn)
    scale = torch.ones(2, 2)
    with pytest.raises(TypeError, match="scale_block_first"):
        Float8BlockwiseQTensor(data, scale, torch.bfloat16)


@pytest.mark.parametrize("block_first", [False, True])
def test_fp8_scale_layout_survives_wrapper_round_trips(kernel_calls, block_first):
    logical_scale = torch.tensor([[1.0, 2.0], [4.0, 8.0]])
    scale = logical_scale.t().contiguous() if block_first else logical_scale
    value = _wrapper((2, 256), scale, scale_block_first=block_first)
    tensors, meta = value.prepare_for_saving()
    for restored in (
        value.detach(),
        value.clone(),
        value.view(value.shape),
        meta.restore(*tensors),
        type(value).__tensor_unflatten__(
            dict(zip(value.__tensor_flatten__()[0], tensors)), meta, None, None
        ),
    ):
        assert restored._scale_block_first is block_first
        torch.testing.assert_close(
            stats.dequantize_fp8(restored),
            _reference(value, logical_scale),
            rtol=0,
            atol=0,
        )
