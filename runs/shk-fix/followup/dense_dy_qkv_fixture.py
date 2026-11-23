"""Derive the dense QKV dY fixture from captured short-convolution operands."""

import argparse
import json
from pathlib import Path

import torch

from dense_dy_quant_transpose import digest


QKV_SHAPE = (2, 4096, 8192)
INPUT_SHAPE = (2, 4096, 4096)
WEIGHT_SHAPE = (8192, 4096)


def load_capture(path):
    receipt = json.loads(path.with_suffix(".json").read_text())
    if digest(path) != receipt["fixture_sha256"]:
        raise ValueError("The capture bytes do not match the source receipt")
    if (
        tuple(receipt["qkv_shape"]) != QKV_SHAPE
        or tuple(receipt["weight_shape"]) != WEIGHT_SHAPE
    ):
        raise ValueError("Expected the native 8,192-token QKV capture")
    data = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    expected = {"x": INPUT_SHAPE, "proj_weight": WEIGHT_SHAPE, "qkv": QKV_SHAPE}
    for name, shape in expected.items():
        tensor = data[name]
        if tuple(tensor.shape) != shape or tensor.dtype != torch.bfloat16:
            raise ValueError(f"Expected captured BF16 {name} of shape {shape}")
    return data, receipt


def derive_dy(data, fla_vjp):
    gq, gk, gv = (data[name] for name in ("grad_q", "grad_k", "grad_v"))
    cu_seqlens = data["cu_seqlens"]
    # Match the stock qkv_conv1d_unpack backward arguments in fla_custom_ops.
    dy, _ = fla_vjp(
        "qkv_conv1d",
        [data[name].to("cuda") for name in ("qkv", "conv_weight")],
        [gradient.to("cuda") for gradient in (gq, gk, gv)],
        [cu_seqlens.to("cuda") if cu_seqlens is not None else None],
        [
            data["num_heads"],
            data["num_v_heads"],
            gq.shape[-1],
            gv.shape[-1],
            data["normalize_qk"],
            data["eps"],
        ],
        "",
    )
    if dy.dtype != torch.bfloat16 or tuple(dy.shape) != QKV_SHAPE:
        raise RuntimeError(
            "The stock short-convolution VJP changed the QKV gradient ABI"
        )
    if not bool(torch.isfinite(dy).all()) or not bool(torch.count_nonzero(dy)):
        raise RuntimeError("The derived QKV gradient is nonfinite or entirely zero")
    return dy.contiguous().cpu()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Specify a fresh output path")
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("Expose one Hopper GPU")
    torch.cuda.set_device(0)

    from fla.ops.common import fused_conv_qkv
    from torchtitan.models.common.attention.linear import fla_custom_ops

    data, receipt = load_capture(args.capture)
    dy = derive_dy(data, fla_custom_ops.fla_vjp)
    provenance = {
        "source": "historical native rank-zero GDN capture",
        "capture_path": str(args.capture.resolve()),
        "capture_receipt": receipt,
        "dy": "stock QKV short-convolution VJP on captured operands",
        "x": "captured BF16 projection input",
        "weight": "captured BF16 projection weight",
        "torch_revision": torch.version.git_version,
        "implementation": {
            str(Path(module.__file__).resolve()): digest(module.__file__)
            for module in (fla_custom_ops, fused_conv_qkv)
        },
    }
    with args.output.open("xb") as handle:
        torch.save(
            {
                "dy": dy,
                "x": data["x"].contiguous(),
                "weight": data["proj_weight"].contiguous(),
                "provenance": provenance,
            },
            handle,
        )
    print(f"Prepared {args.output}, SHA-256 {digest(args.output)}", flush=True)


if __name__ == "__main__":
    main()
