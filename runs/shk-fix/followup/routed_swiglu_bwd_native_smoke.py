"""Check the native routed SwiGLU backward bridge on unbacked row counts."""

import inspect
import os
from types import SimpleNamespace
from typing import NamedTuple
from unittest.mock import patch

import torch
from torch._inductor.runtime.triton_heuristics import CachingAutotuner

import native_routed_swiglu_bwd_measure as bridge
from torchtitan.ops.float8.triton_kernels import fused_swiglu_quantization as swiglu


@torch.library.custom_op("shk_smoke::backward_routed_prefix", mutates_args=())
def routed_prefix(
    x: torch.Tensor, p: torch.Tensor, dy: torch.Tensor, rows: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    count = int(rows.item())
    return x[:count].clone(), p[:count].clone(), dy[:count].clone()


@routed_prefix.register_fake
def fake_routed_prefix(x, p, dy, rows):
    del rows
    count = torch.library.get_ctx().new_dynamic_size()
    torch._check(count % swiglu.FWD_ROW_BLOCK == 0)
    return (
        x.new_empty((count, x.shape[1])),
        p.new_empty((count,)),
        dy.new_empty((count, dy.shape[1])),
    )


class BackwardContext(NamedTuple):
    saved_tensors: tuple[torch.Tensor, torch.Tensor]
    use_edges: bool
    swiglu_limit: float


def make_backward(backward_method):
    def backward(x, p, dy, rows):
        x, p, dy = routed_prefix(x, p, dy, rows)
        ctx = BackwardContext((x, p), True, 0.0)
        packed, _, dprobs, *_ = backward_method(ctx, dy)
        return packed._data, packed._scale, dprobs

    return backward


class ProjectionStep:
    def forward_backward_step(self, rows):
        return self.compiled(*self.operands, rows)


def run(operands, enabled):
    function = swiglu.FusedQuantizedSwiGLU
    originals = (
        vars(function)["forward"],
        vars(function)["backward"],
        swiglu._swiglu_quantized_backward,
        ProjectionStep.forward_backward_step,
        CachingAutotuner.run,
    )

    def worker(_, adapter):
        assert vars(function)["forward"] is originals[0]
        backward_descriptor = vars(function)["backward"]
        assert (backward_descriptor is originals[1]) != enabled
        assert inspect.isfunction(function.backward)
        assert function.backward is backward_descriptor.__func__
        assert swiglu._swiglu_quantized_backward is originals[2]
        step = adapter.trainer_class()
        step.operands = operands
        # Dynamo reserves the class's backward attribute for real autograd contexts.
        # Bind the exact installed implementation for this manual VJP check.
        step.compiled = torch.compile(
            make_backward(backward_descriptor.__func__),
            fullgraph=True,
            options={"triton.cudagraphs": False},
        )
        outputs = {}
        with torch._dynamo.config.patch(error_on_recompile=True):
            for rows in (128, 256):
                outputs[rows] = step.forward_backward_step(torch.tensor(rows))
                assert ProjectionStep.forward_backward_step is originals[3]
                assert CachingAutotuner.run is originals[4]
        torch.cuda.synchronize()
        return outputs

    result = bridge.run_native(
        [], SimpleNamespace(trainer_class=ProjectionStep), worker, enabled
    )
    assert (
        vars(function)["forward"],
        vars(function)["backward"],
        swiglu._swiglu_quantized_backward,
        ProjectionStep.forward_backward_step,
        CachingAutotuner.run,
    ) == originals
    return result


def main():
    torch.manual_seed(3297)
    x = torch.randn(256, 2 * bridge.WIDTH, device="cuda", dtype=torch.bfloat16)
    p = torch.rand(256, device="cuda", dtype=torch.float32)
    dy = torch.randn(256, bridge.WIDTH, device="cuda", dtype=torch.bfloat16)
    with patch.dict(os.environ, {"RANK": "0", swiglu._BWD_MN_MAJOR_SCALE_ENV: "0"}):
        control = run((x, p, dy), False)
        torch._dynamo.reset()
        candidate = run((x, p, dy), True)
    for rows, outputs in control.items():
        for field, left, right in zip(
            ("FP8 gradient", "row-major scales", "routing gradient"),
            outputs,
            candidate[rows],
            strict=True,
        ):
            assert left.shape[0] == rows, field
            assert (left.shape, left.stride(), left.dtype) == (
                right.shape,
                right.stride(),
                right.dtype,
            ), field
            assert torch.equal(left.view(torch.uint8), right.view(torch.uint8)), field
        assert outputs[1].stride() == (2 * bridge.COLUMNS, 1)
    print("Unbacked routed SwiGLU backward smoke: FP8, scales and dP match bitwise")


if __name__ == "__main__":
    main()
