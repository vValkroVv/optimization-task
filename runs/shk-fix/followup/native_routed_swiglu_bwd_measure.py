"""Run the native GIFT worker with the opt-in routed SwiGLU backward CTA order."""

from contextlib import nullcontext
from hashlib import sha256
import importlib.util
import inspect
import json
import os
from pathlib import Path
from textwrap import dedent, indent
from unittest.mock import patch

from native_routed_swiglu_fwd_measure import BRIDGE, MAX_GRID_Y
from routed_swiglu_bwd_order_component import (
    CANDIDATE_KERNEL,
    NUM_WARPS,
    PRODUCTION_KERNEL,
    TILE,
    digest,
    make_candidate,
    source_function,
)


HERE = Path(__file__).resolve().parent
COMPONENT = HERE / "routed_swiglu_bwd_order_component.py"
RECEIPT = HERE.parent / "artifacts/raw/routed-swiglu-bwd-order-001/result.json"
RECEIPT_SHA256 = "d8ebb1b584cc0661b970eaced63616d24e196a6005b7454ea969e7eb50cc3378"
SWITCH = "GIFT_ROUTED_SWIGLU_BWD_FEATURE_FIRST"
WIDTH = 1536
COLUMNS = WIDTH // TILE[1]
ACTIVATION = "Routed SwiGLU backward feature-first executed"
LAUNCH = "Routed SwiGLU backward launch"


def validate_receipt(swiglu, torch, triton):
    raw = RECEIPT.read_bytes()
    if sha256(raw).hexdigest() != RECEIPT_SHA256:
        raise RuntimeError("The SwiGLU backward component receipt changed")
    receipt = json.loads(raw)
    if (
        not receipt["bitwise"]
        or not receipt["parity"]["row_major_scale"]
        or not receipt["non_target_launches_matched"]
        or not all(arm["activation_verified"] for arm in receipt["profiles"].values())
        or receipt["input_shape_stride_dtype"]["x"][0][1] != 2 * WIDTH
        or receipt["clamp"] != 0.0
        or tuple(receipt["tile"]) != TILE
        or (swiglu.BWD_ROW_BLOCK, swiglu.COL_BLOCK) != TILE
        or receipt["num_warps"] != NUM_WARPS
        or receipt["native_swiglu_source_sha256"] != digest(Path(swiglu.__file__))
        or receipt["probe_sha256"] != digest(COMPONENT)
        or receipt["torch"] != torch.__version__
        or receipt["triton"] != triton.__version__
        or receipt["cuda"] != torch.version.cuda
    ):
        raise RuntimeError(
            "The native worker differs from the SwiGLU backward component"
        )
    return receipt


def make_backward(swiglu, jit, receipt):
    feature_first, kernel_source, dispatch_source = make_candidate(swiglu, jit)
    for source, key in (
        (kernel_source, "candidate_kernel_sha256"),
        (dispatch_source, "candidate_dispatch_sha256"),
    ):
        if sha256(source.encode()).hexdigest() != receipt[key]:
            raise RuntimeError(
                "The native candidate differs from the measured kernel or dispatch"
            )

    source = dedent(inspect.getsource(swiglu.FusedQuantizedSwiGLU.backward))
    call = "    grad_in, grad_in_sf, grad_probs = _swiglu_quantized_backward("
    if source.count(call) != 1:
        raise ValueError("The GIFT SwiGLU backward dispatch has changed")
    selection = dedent(f"""\
        dispatch = _swiglu_quantized_backward
        if (
            ctx.use_edges
            and x_c.is_cuda
            and x_c.dtype == torch.bfloat16
            and x_c.shape[1] == {2 * WIDTH}
            and p_c.dtype == torch.float32
            and isinstance(grad_output, torch.Tensor)
            and not isinstance(grad_output, Float8BlockwiseQTensor)
            and grad_output.dtype == torch.bfloat16
            and ctx.swiglu_limit == 0.0
            and os.environ.get(_BWD_MN_MAJOR_SCALE_ENV, "") in ("", "0")
        ):
            if torch.compiler.is_compiling():
                # Blocking HybridEP returns an unbacked padded row count.
                # Check CUDA's grid-y bound without specializing to that count.
                torch._check(x_c.shape[0] <= {MAX_GRID_Y} * BWD_ROW_BLOCK)
                dispatch = feature_first_dispatch
            elif 0 < x_c.shape[0] <= {MAX_GRID_Y} * BWD_ROW_BLOCK:
                dispatch = feature_first_dispatch
        grad_in, grad_in_sf, grad_probs = dispatch(
    """).rstrip()
    source = source.replace(call, indent(selection, "    "))
    backward = source_function(
        source, vars(swiglu) | {"feature_first_dispatch": feature_first}, "backward"
    )
    if not isinstance(backward, staticmethod):
        raise TypeError("The derived SwiGLU backward must be a staticmethod")
    return backward


def verified_grid(kernel, args, enabled):
    meta = kernel.inductor_meta
    names = (getattr(kernel.fn, "__name__", ""), meta.get("kernel_name") or "")
    candidate = any(CANDIDATE_KERNEL in name for name in names)
    if candidate and not enabled:
        raise RuntimeError("The feature-first SwiGLU backward ran in the baseline arm")
    if not (candidate if enabled else any(PRODUCTION_KERNEL in name for name in names)):
        return None
    if meta.get("grid_type") != "FixedGrid" or meta.get("extra_launcher_args") != [
        "_grid_0",
        "_grid_1",
        "_grid_2",
    ]:
        raise RuntimeError("Unexpected Inductor grid ABI for routed SwiGLU backward")
    grid = tuple(args[-3:])
    x, y, z = grid
    rows, cols = (y, x) if enabled else (x, y)
    if cols == COLUMNS and z == 1:
        if rows == 0:
            return None
        if rows > 0 and (not enabled or rows <= MAX_GRID_Y):
            return grid
    if enabled:
        raise RuntimeError(f"Unexpected feature-first SwiGLU backward grid {grid}")
    return None


def run_native(train_argv, adapter, native_main, enabled):
    import torch
    import triton
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner
    from torchtitan.ops.float8.triton_kernels import fused_swiglu_quantization as swiglu

    if torch.cuda.get_device_capability() != (9, 0):
        return native_main(train_argv, adapter)
    receipt = validate_receipt(swiglu, torch, triton)
    backward_scope = nullcontext()
    if enabled:
        backward = make_backward(swiglu, triton.jit, receipt)
        backward_scope = patch.object(swiglu.FusedQuantizedSwiGLU, "backward", backward)

    rank = int(os.environ["RANK"])
    trainer_class = adapter.trainer_class
    forward_backward = trainer_class.forward_backward_step
    original_run = CachingAutotuner.run
    verified = False

    def observe(kernel, *args, **kwargs):
        nonlocal verified
        result = original_run(kernel, *args, **kwargs)
        if not verified and not kwargs.get("benchmark_run", False):
            grid = verified_grid(kernel, args, enabled)
            if grid is not None:
                # Observe one real launch, then restore the runtime method.
                verified = True
                CachingAutotuner.run = original_run
                if enabled:
                    print(f"{ACTIVATION} rank={rank} width={WIDTH}", flush=True)
                print(f"{LAUNCH} rank={rank} grid={grid}", flush=True)
        return result

    def verify_first_microbatch(trainer, *args, **kwargs):
        with patch.object(CachingAutotuner, "run", observe):
            result = forward_backward(trainer, *args, **kwargs)
        if not verified:
            raise RuntimeError(
                "No routed SwiGLU backward launch in the first native microbatch"
            )
        if trainer_class.forward_backward_step is verify_first_microbatch:
            trainer_class.forward_backward_step = forward_backward
        return result

    with (
        backward_scope,
        patch.object(trainer_class, "forward_backward_step", verify_first_microbatch),
    ):
        result = native_main(train_argv, adapter)
    if not verified:
        raise RuntimeError("The audit saw no native routed SwiGLU backward launch")
    return result


def main():
    enabled = os.environ.get(SWITCH)
    if enabled not in (None, "0", "1"):
        raise ValueError(f"{SWITCH} expects 0 or 1")
    spec = importlib.util.spec_from_file_location(
        "shk_routed_bwd_flash_measure", BRIDGE
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not load the original measurement worker at {BRIDGE}"
        )
    bridge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bridge)
    if enabled is None:
        return bridge.main()

    bootstrap = bridge._bootstrap_and_run

    def start(train_argv, adapter, native_main):
        def with_order(argv, current_adapter):
            return run_native(argv, current_adapter, native_main, enabled == "1")

        return bootstrap(train_argv, adapter, with_order)

    with patch.object(bridge, "_bootstrap_and_run", start):
        return bridge.main()


if __name__ == "__main__":
    main()
