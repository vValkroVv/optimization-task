"""Capture one native GDN projection and short-convolution backward."""

import hashlib
import json
import os
import subprocess
from pathlib import Path

import torch


def _snapshot(tensor):
    host = torch.empty_strided(
        tensor.shape, tensor.stride(), dtype=tensor.dtype, device="cpu"
    )
    return host.copy_(tensor.detach())


def install(output_dir):
    if int(os.environ["RANK"]) != 0:
        return

    from torchtitan.models.common.attention.linear import gdn_short_conv
    from torchtitan.ops.float8.cuda_gemm import FP8Gemm
    from torchtitan.trainer import Trainer

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fixture = output_dir / "gdn-qkv-rank0.pt"
    if fixture.exists() or fixture.with_suffix(".json").exists():
        raise FileExistsError(fixture)
    pending = {}
    active = False
    captured = False
    original_forward = FP8Gemm.forward
    original_backward = gdn_short_conv.gdn_short_conv_backward
    original_step = Trainer.forward_backward_step

    @torch.library.custom_op("gdn_fixture::projected", mutates_args=())
    def projected(
        x: torch.Tensor, weight: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        result = y.clone(memory_format=torch.contiguous_format)
        if active and not captured:
            pending[result.data_ptr()] = (
                _snapshot(x),
                _snapshot(weight),
                result.shape,
            )
        return result

    @projected.register_fake
    def _(x, weight, y):
        return torch.empty_like(y, memory_format=torch.contiguous_format)

    def forward(ctx, x, weight, *args, **kwargs):
        y = original_forward(ctx, x, weight, *args, **kwargs)
        if (
            x.dtype == y.dtype == torch.bfloat16
            and x.ndim == y.ndim == 3
            and x.shape[-1] == 4096
            and y.shape[-1] == 8192
            and weight.shape == (8192, 4096)
        ):
            return projected(x, weight, y)
        return y

    @torch.library.custom_op("gdn_fixture::short_conv_backward", mutates_args=())
    def short_conv_backward(
        qkv: torch.Tensor,
        weight: torch.Tensor,
        grad_q: torch.Tensor,
        grad_k: torch.Tensor,
        grad_v: torch.Tensor,
        cu_seqlens: torch.Tensor | None,
        num_heads: int,
        num_v_heads: int,
        normalize_qk: bool,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        nonlocal captured
        args = (
            qkv,
            weight,
            grad_q,
            grad_k,
            grad_v,
            cu_seqlens,
            num_heads,
            num_v_heads,
            normalize_qk,
            eps,
        )
        result = original_backward(*args)
        if not active or captured or qkv.shape[-1] != 8192:
            return result
        projection = pending.get(qkv.data_ptr())
        if projection is None or projection[2] != qkv.shape:
            raise RuntimeError(
                "Could not match the native short-convolution and QKV projection"
            )

        x, proj_weight, _ = projection
        torch.save(
            {
                "x": x,
                "proj_weight": proj_weight,
                "qkv": _snapshot(qkv),
                "conv_weight": _snapshot(weight),
                "grad_q": _snapshot(grad_q),
                "grad_k": _snapshot(grad_k),
                "grad_v": _snapshot(grad_v),
                "cu_seqlens": None if cu_seqlens is None else _snapshot(cu_seqlens),
                "num_heads": num_heads,
                "num_v_heads": num_v_heads,
                "normalize_qk": normalize_qk,
                "eps": eps,
            },
            fixture.with_suffix(".tmp"),
        )
        fixture.with_suffix(".tmp").rename(fixture)
        with fixture.open("rb") as reader:
            digest = hashlib.file_digest(reader, "sha256").hexdigest()
        receipt = {
            "fixture_sha256": digest,
            "rank": 0,
            "world_size": int(os.environ["WORLD_SIZE"]),
            "source_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "qkv_shape": list(qkv.shape),
            "weight_shape": list(proj_weight.shape),
        }
        fixture.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n")
        pending.clear()
        captured = True
        print(f"[GDN_CAPTURE] fixture={fixture} sha256={digest}", flush=True)
        return result

    @short_conv_backward.register_fake
    def _(
        qkv,
        weight,
        grad_q,
        grad_k,
        grad_v,
        cu_seqlens,
        num_heads,
        num_v_heads,
        normalize_qk,
        eps,
    ):
        return original_backward(
            qkv,
            weight,
            grad_q,
            grad_k,
            grad_v,
            cu_seqlens,
            num_heads,
            num_v_heads,
            normalize_qk,
            eps,
        )

    FP8Gemm.forward = staticmethod(forward)
    gdn_short_conv.gdn_short_conv_backward = short_conv_backward

    def first_step(trainer, *args, **kwargs):
        nonlocal active
        Trainer.forward_backward_step = original_step
        if trainer.gradient_accumulation_steps != 32 or trainer.step != 1:
            raise RuntimeError(
                "GDN capture requires the first native update with accumulation 32"
            )
        active = True
        try:
            result = original_step(trainer, *args, **kwargs)
        finally:
            active = False
        if not captured:
            raise RuntimeError(
                "The first native microbatch did not capture GDN backward"
            )
        return result

    Trainer.forward_backward_step = first_step
