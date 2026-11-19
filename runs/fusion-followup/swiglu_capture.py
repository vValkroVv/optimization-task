"""Capture a real expert SwiGLU VJP and its first grouped dgrad consumer."""

import os
from pathlib import Path

import torch


def install(output_dir: Path):
    if int(os.environ["RANK"]) != 0:
        return
    from torchtitan.ops.float8 import deep_gemm_handle as dg
    from torchtitan.ops.float8.triton_kernels.fused_swiglu_quantization import (
        FusedQuantizedSwiGLU,
    )
    from torchtitan.trainer import Trainer

    fixture = output_dir / "swiglu-rank0.pt"
    output_dir.mkdir(parents=True, exist_ok=True)
    active = False
    tensors = {}

    def snapshot(t):
        return t.detach().to("cpu", copy=True)

    @torch.library.custom_op("swiglu_fixture::incoming", mutates_args=())
    def incoming(
        dy: torch.Tensor, x: torch.Tensor, p: torch.Tensor, limit: float
    ) -> torch.Tensor:
        if active and not tensors and x.shape[1] == 3072:
            tensors.update(x=snapshot(x), p=snapshot(p), dy=snapshot(dy), limit=limit)
        return dy.clone()

    @incoming.register_fake
    def _(dy, x, p, limit):
        return torch.empty_like(dy)

    previous_bwd = FusedQuantizedSwiGLU.backward

    def capture_backward(ctx, grad_output, grad_payload=None):
        x, p = ctx.saved_tensors
        return previous_bwd(
            ctx, incoming(grad_output, x, p, ctx.swiglu_limit), grad_payload
        )

    FusedQuantizedSwiGLU.backward = staticmethod(capture_backward)

    previous_gemm = dg._m_grouped_fp8_gemm

    def capture_consumer(a_data, a_scale, b_data, b_scale, m_indices):
        output = previous_gemm(a_data, a_scale, b_data, b_scale, m_indices)
        if (
            active
            and tensors
            and "weight" not in tensors
            and a_data.shape[1] == 3072
            and b_data.shape[-1] == 3072
        ):
            tensors.update(
                grad_data=snapshot(a_data),
                grad_scale=snapshot(a_scale),
                weight=snapshot(b_data),
                weight_scale=snapshot(b_scale),
                m_indices=snapshot(m_indices),
                dgrad=snapshot(output),
            )
        return output

    dg._m_grouped_fp8_gemm = capture_consumer
    previous_step = Trainer.forward_backward_step

    def first_step(trainer, *args, **kwargs):
        nonlocal active
        Trainer.forward_backward_step = previous_step
        active = True
        try:
            result = previous_step(trainer, *args, **kwargs)
        finally:
            active = False
            dg._m_grouped_fp8_gemm = previous_gemm
        if not tensors:
            raise RuntimeError("No native expert SwiGLU captured")
        with fixture.with_suffix(".tmp").open("xb") as handle:
            torch.save(tensors, handle)
        fixture.with_suffix(".tmp").rename(fixture)
        print(f"swiglu_native_fixture={fixture} fields={sorted(tensors)}", flush=True)
        tensors.clear()
        return result

    Trainer.forward_backward_step = first_step
