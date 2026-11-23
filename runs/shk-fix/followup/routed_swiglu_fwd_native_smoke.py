"""Exercise the native SwiGLU bridge with unbacked routed token counts."""

import os
from types import SimpleNamespace

import torch

import native_routed_swiglu_fwd_measure as bridge
from torchtitan.ops.float8.triton_kernels import fused_swiglu_quantization as swiglu


@torch.library.custom_op("shk_smoke::routed_prefix", mutates_args=())
def routed_prefix(
    x: torch.Tensor, p: torch.Tensor, rows: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    count = int(rows.item())
    return x[:count].clone(), p[:count].clone()


@routed_prefix.register_fake
def fake_routed_prefix(x, p, rows):
    del rows
    count = torch.library.get_ctx().new_dynamic_size()
    torch._check(count % swiglu.FWD_ROW_BLOCK == 0)
    return x.new_empty((count, x.shape[1])), p.new_empty((count,))


def forward(edge, x, p, rows):
    x, p = routed_prefix(x, p, rows)
    _, output = swiglu.fused_swiglu_quantized_fn(edge, x, p)
    return output._data, output._scale


class ProjectionStep:
    def forward_backward_step(self, rows):
        return self.compiled(*self.operands, rows)


def run(operands, enabled):
    def worker(_, adapter):
        step = adapter.trainer_class()
        step.operands = operands
        step.compiled = torch.compile(
            forward, fullgraph=True, options={"triton.cudagraphs": False}
        )
        with torch._dynamo.config.patch(error_on_recompile=True):
            outputs = {
                rows: step.forward_backward_step(torch.tensor(rows))
                for rows in (128, 256)
            }
        torch.cuda.synchronize()
        return outputs

    return bridge.run_native(
        [], SimpleNamespace(trainer_class=ProjectionStep), worker, enabled
    )


def main():
    os.environ["RANK"] = "0"
    torch.manual_seed(3297)
    edge = torch.empty((), device="cuda")
    x = torch.randn(256, 3072, device="cuda", dtype=torch.bfloat16)
    p = torch.rand(256, device="cuda", dtype=torch.float32)
    a = run((edge, x, p), False)
    torch._dynamo.reset()
    b = run((edge, x, p), True)
    for rows, outputs in a.items():
        for left, right in zip(outputs, b[rows], strict=True):
            assert left.shape == right.shape and left.shape[0] == rows
            assert torch.equal(left.view(torch.uint8), right.view(torch.uint8))
    print("Unbacked routed SwiGLU smoke: both output planes match bitwise")


if __name__ == "__main__":
    main()
