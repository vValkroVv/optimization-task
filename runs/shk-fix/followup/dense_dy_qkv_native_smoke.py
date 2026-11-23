"""Check the native FP8Linear backward bridge with captured QKV operands."""

import argparse
import os
from pathlib import Path
from types import SimpleNamespace

import torch

import native_dense_dy_qkv_measure as bridge
from dense_dy_qkv_fixture import derive_dy, load_capture


class ProjectionStep:
    def forward_backward_step(self):
        x, weight, dy = self.operands
        y = self.compiled(x, weight)
        dx, dw = torch.autograd.grad(y, (x, weight), dy)
        return y.detach(), dx, dw


def run(output, operands, enabled):
    from torchtitan.ops.float8.cuda_gemm import FP8Linear

    projection = SimpleNamespace(_quantize_weight_tail=True, _wgrad_logical_width=False)

    def forward(x, weight):
        return FP8Linear._apply_gemm(projection, x, weight, None, False, True)

    def worker(_, adapter):
        step = adapter.trainer_class()
        step.compiled = torch.compile(
            forward, fullgraph=True, dynamic=False, options={"triton.cudagraphs": False}
        )
        step.operands = operands
        outputs = step.forward_backward_step()
        torch.cuda.synchronize()
        return outputs

    adapter = SimpleNamespace(
        trainer_class=ProjectionStep,
        options={"capture_microbatches": (), "profile_update": None},
    )
    receipt_dir = output / ("B" if enabled else "A")
    receipt_dir.mkdir()
    return bridge.run_native([], adapter, worker, receipt_dir, enabled)


def main():
    from torchtitan.models.common.attention.linear.fla_custom_ops import fla_vjp

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    os.environ["RANK"] = "0"
    for name in (
        "GIFT_CUTE_NORM_FUSE_RESIDUAL",
        "GIFT_CUTE_NORM_SAC_STATS_ONLY",
        "GIFT_FP8_QUANTIZE_WEIGHT_TAIL",
    ):
        os.environ[name] = "1"

    def record_stride(*args):
        result = fla_vjp(*args)
        dy = result[0]
        assert dy.is_contiguous() and dy.storage_offset() % 8 == 0
        print(
            f"Stock FLA QKV dY shape={tuple(dy.shape)} stride={dy.stride()}", flush=True
        )
        return result

    data, _ = load_capture(args.capture)
    dy = derive_dy(data, record_stride).to("cuda")
    operands = (
        data["x"].contiguous().to("cuda").requires_grad_(),
        data["proj_weight"].contiguous().to("cuda").requires_grad_(),
        dy,
    )
    outputs_a = run(args.output, operands, False)
    torch._dynamo.reset()
    outputs_b = run(args.output, operands, True)
    for name, a, b in zip(("Y", "dX", "dW"), outputs_a, outputs_b, strict=True):
        assert (a.dtype, a.shape, a.stride()) == (b.dtype, b.shape, b.stride()), name
        assert torch.equal(
            a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
        ), name
    print("Native QKV bridge smoke: Y, dX and dW match bitwise", flush=True)


if __name__ == "__main__":
    main()
