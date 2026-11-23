"""Isolated GIFT dense FFN compute with the production FP8 and SAC paths."""

from __future__ import annotations

import importlib.util
import os
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CASE = "dense-fp8"
SHAPE = (2, 4096, 4096)
WEIGHT_SHAPES = {
    "w1.weight": (10944, 4096),
    "w2.weight": (4096, 10944),
    "w3.weight": (10944, 4096),
}
TAIL_ENV = "GIFT_FP8_QUANTIZE_WEIGHT_TAIL"
INIT_SEED = 2048
QUANTIZER = re.compile(
    r"quantize_blk128x128_kernel_(nt|t)_computed_scale"
    r"<\s*true\s*,\s*(true|false)\s*,\s*(true|false)\s*>"
)


def layer0_crop():
    """Reuse the native component's production config resolution and crop."""
    path = ROOT / "runs/ffn-row-owner/ffn_components.py"
    spec = importlib.util.spec_from_file_location("ffn_components", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = module.resolved_config("layer0")
    crop = module.meta_component(config, "layer0")
    contract = module.contract(config, crop, "layer0")
    if contract["shape"] != list(SHAPE) or contract["ffn_shapes"] != {
        name: list(shape) for name, shape in WEIGHT_SHAPES.items()
    }:
        raise ValueError(f"unexpected production FFN contract: {contract}")
    return module, config, crop


def compile_regions(config, crop):
    from torchtitan.distributed.activation_checkpoint import apply_ac, resolve_ac_mode
    from torchtitan.distributed.compile import apply_compile
    from torchtitan.distributed.ffn_overlap import enable_ffn_overlap

    if (
        resolve_ac_mode("layers.0", config.activation_checkpoint) != "selective"
        or config.activation_checkpoint.selective_save
        or config.training.mixed_precision_param != "bfloat16"
    ):
        raise ValueError("dense replay requires BF16 and selective recomputation")

    block = crop.layers["0"]
    enable_ffn_overlap(block)
    apply_ac(crop, config.activation_checkpoint, model_compile_enabled=True)
    # The measurement driver owns external graph capture. Use the GIFT switch
    # for disabling Inductor's internal graph replay at this one compile boundary.
    config.compile.cudagraph_exclude_blocks = ["layers.0"]
    apply_compile(crop, config.compile)
    return block._ffn_overlap_regions


def validate_ffn(ffn, tail):
    from torchtitan.ops.float8.cuda_gemm import FP8Linear

    if not (
        ffn.fp8_enabled and ffn.use_shared_quantizer and ffn.dense_fused_swiglu_quant
    ):
        raise ValueError("shared FP8 activation and fused SwiGLU are required")
    for linear in (ffn.w1, ffn.w2, ffn.w3):
        if not isinstance(linear, FP8Linear) or linear._quantize_weight_tail != tail:
            raise ValueError("the requested dense FP8 weight-tail policy did not bind")


def _validate_tensor(name, tensor, shape):
    import torch

    if (
        not isinstance(tensor, torch.Tensor)
        or tuple(tensor.shape) != shape
        or tensor.dtype != torch.bfloat16
        or not tensor.is_contiguous()
    ):
        raise ValueError(f"{name} must be contiguous BF16 with shape {shape}")


def validate_tensors(payload):
    if payload["schema_version"] != 1 or payload["case"] != CASE:
        raise ValueError("invalid dense component fixture")
    _validate_tensor("input", payload["input"], SHAPE)
    _validate_tensor("tangent", payload["tangent"], SHAPE)
    if set(payload["weights"]) != set(WEIGHT_SHAPES):
        raise ValueError("dense fixture must contain all three FP8Linear weights")
    for name, shape in WEIGHT_SHAPES.items():
        _validate_tensor(name, payload["weights"][name], shape)


def _tail_enabled(case, arm):
    if case != CASE or arm.upper() not in ("A", "B"):
        raise ValueError(f"unsupported dense replay case/arm: {case}/{arm}")
    return arm.upper() == "B"


def build(case, arm, fixture_path):
    import torch

    tail = _tail_enabled(case, arm)
    payload = torch.load(fixture_path, map_location="cpu", weights_only=True)
    validate_tensors(payload)

    previous = os.environ.get(TAIL_ENV)
    os.environ[TAIL_ENV] = str(int(tail))
    try:
        _, config, crop = layer0_crop()
    finally:
        if previous is None:
            os.environ.pop(TAIL_ENV, None)
        else:
            os.environ[TAIL_ENV] = previous
    _, ffn, _ = compile_regions(config, crop)
    validate_ffn(ffn, tail)
    ffn.train()
    # Load the exact BF16 compute weights outside either timed phase.
    ffn.load_state_dict(
        {name: value.to("cuda") for name, value in payload["weights"].items()},
        assign=True,
    )
    leaves = dict(ffn.named_parameters())
    x = payload["input"].to("cuda").requires_grad_()
    leaves["input"] = x
    tangent = payload["tangent"].to("cuda")
    return lambda: ffn(x), tangent, leaves


def _kernel_counts(counts):
    result = dict.fromkeys(
        (
            "pad",
            "nt",
            "nt_row_tail",
            "nt_col_tail",
            "t",
            "t_row_tail",
            "t_col_tail",
            "swiglu_forward",
            "swiglu_backward",
            "fp8_gemm",
            "wgrad_gemm",
        ),
        0,
    )
    for name, count in counts.items():
        if "constant_pad_nd" in name:
            result["pad"] += count
        if match := QUANTIZER.search(name):
            orientation, row_tail, col_tail = match.groups()
            result[orientation] += count
            result[f"{orientation}_row_tail"] += count * (row_tail == "true")
            result[f"{orientation}_col_tail"] += count * (col_tail == "true")
        if "swiglu_forward_kernel" in name:
            result["swiglu_forward"] += count
        if "swiglu_backward_kernel" in name:
            result["swiglu_backward"] += count
        if "nvjet_sm90_" in name and "_Ablk128_Bvec128_" in name:
            result["fp8_gemm"] += count
        if "nvjet_sm90_" in name and "_Avec128_Bvec128_" in name:
            result["wgrad_gemm"] += count
    return result


def verify_kernels(case, arm, forward_counts, backward_counts):
    tail = _tail_enabled(case, arm)
    forward = _kernel_counts(forward_counts)
    backward = _kernel_counts(backward_counts)
    expected_forward = {
        "pad": 0 if tail else 3,
        "nt": 3,
        "nt_row_tail": 2 if tail else 0,
        "nt_col_tail": int(tail),
        "t": 0,
        "t_row_tail": 0,
        "t_col_tail": 0,
        "swiglu_forward": 1,
        "swiglu_backward": 0,
        "fp8_gemm": 3,
        "wgrad_gemm": 0,
    }
    # The two gate/up projections feed SwiGLU rematerialization. The down
    # projection only needs its transposed weight quantizer in backward.
    expected_backward = {
        "pad": 0 if tail else 3,
        "nt": 2,
        "nt_row_tail": 2 if tail else 0,
        "nt_col_tail": 0,
        "t": 3,
        "t_row_tail": 2 if tail else 0,
        "t_col_tail": int(tail),
        "swiglu_forward": 1,
        "swiglu_backward": 1,
        "fp8_gemm": 5,
        "wgrad_gemm": 3,
    }
    if forward != expected_forward or backward != expected_backward:
        raise ValueError(
            f"dense kernel fingerprint mismatch for arm {arm}\n"
            f"forward: {forward} expected {expected_forward}\n"
            f"backward: {backward} expected {expected_backward}"
        )
    return {"weight_tail": tail, "forward": forward, "backward": backward}
