"""Run the native GIFT worker with the opt-in routed SwiGLU CTA order."""

from hashlib import sha256
from importlib.metadata import version
import importlib.util
import inspect
import json
import linecache
import os
from pathlib import Path
from textwrap import dedent, indent
from unittest.mock import patch

from routed_swiglu_fwd_order_component import (
    CANDIDATE_KERNEL,
    PRODUCTION_KERNEL,
    digest,
    make_candidate,
)


HERE = Path(__file__).resolve().parent
BRIDGE = HERE.parents[1] / "flash-qla/measure.py"
COMPONENT = HERE / "routed_swiglu_fwd_order_component.py"
RECEIPT = HERE.parent / "artifacts/raw/routed-swiglu-fwd-order-001/component.json"
RECEIPT_SHA256 = "afe7221549ede0ea2b25711748e3dac2c8c766e3e236afa280ae27c68f917dae"
SWITCH = "GIFT_ROUTED_SWIGLU_FWD_FEATURE_FIRST"
WIDTH = 1536
MAX_GRID_Y = 65535
ACTIVATION = "Routed SwiGLU forward feature-first executed"


def validate_receipt(swiglu, torch):
    raw = RECEIPT.read_bytes()
    if sha256(raw).hexdigest() != RECEIPT_SHA256:
        raise RuntimeError("The routed SwiGLU component receipt changed")
    receipt = json.loads(raw)
    if (
        not receipt["bitwise"]
        or not all(arm["activation_verified"] for arm in receipt["profiles"].values())
        or receipt["gate_up_shape"][-1] != 2 * WIDTH
        or receipt["swiglu_limit"] != 0.0
        or receipt["production_file_sha256"] != digest(Path(swiglu.__file__))
        or receipt["runner_sha256"] != digest(COMPONENT)
        or receipt["torch_revision"] != torch.version.git_version
        or receipt["triton"] != version("triton")
        or receipt["cuda"] != torch.version.cuda
    ):
        raise RuntimeError(
            "The native worker does not match the verified SwiGLU component"
        )
    return receipt


def with_feature_first(source):
    call = "    _swiglu_quant_fwd_kernel[grid]("
    if source.count(call) != 1:
        raise ValueError("The GIFT SwiGLU forward launch has changed")
    selection = dedent(f"""\
            kernel = _swiglu_quant_fwd_kernel
            use_feature_first = (
                ctx.use_edges
                and input_tensor.is_cuda
                and input_tensor.dtype == torch.bfloat16
                and probs.dtype == torch.float32
                and hid_dim == {WIDTH}
                and swiglu_limit == 0.0
            )
            if use_feature_first:
                if torch.compiler.is_compiling():
                    # Blocking HybridEP produces an unbacked row count.
                    # Defer the grid limit to the host shape check.
                    torch._check(grid[0] <= {MAX_GRID_Y})
                else:
                    use_feature_first = 0 < grid[0] <= {MAX_GRID_Y}
            if use_feature_first:
                kernel = {CANDIDATE_KERNEL}
                grid = (grid[1], grid[0])
            kernel[grid](
        """).rstrip()
    return source.replace(call, indent(selection, "    "))


def make_forward(swiglu, candidate):
    original = swiglu.FusedQuantizedSwiGLU.forward
    source = with_feature_first(dedent(inspect.getsource(original)))
    filename = str(Path(__file__).with_suffix(".generated.py"))
    linecache.cache[filename] = (
        len(source),
        None,
        source.splitlines(keepends=True),
        filename,
    )
    namespace = vars(swiglu) | {CANDIDATE_KERNEL: candidate}
    exec(compile(source, filename, "exec", dont_inherit=True), namespace)
    return namespace["forward"]


def verified_grid(kernel, args, enabled, columns):
    meta = kernel.inductor_meta
    names = (getattr(kernel.fn, "__name__", ""), meta.get("kernel_name") or "")
    candidate = any(CANDIDATE_KERNEL in name for name in names)
    if candidate and not enabled:
        raise RuntimeError("The feature-first SwiGLU kernel ran in the baseline arm")
    if not (candidate if enabled else any(PRODUCTION_KERNEL in name for name in names)):
        return None
    # Inductor passes the three evaluated axes as arguments for a single-grid
    # Triton kernel, including grids with a symbolic dimension.
    if meta.get("grid_type") != "FixedGrid" or meta.get("extra_launcher_args") != [
        "_grid_0",
        "_grid_1",
        "_grid_2",
    ]:
        raise RuntimeError("Unexpected Inductor grid ABI for routed SwiGLU")
    grid = tuple(args[-3:])
    x, y, z = grid
    rows, cols = (y, x) if enabled else (x, y)
    if cols == columns and z == 1 and rows == 0:
        # Triton's CUDA launcher skips zero-work grids.
        return None
    if cols == columns and 0 < rows and z == 1 and (not enabled or rows <= MAX_GRID_Y):
        return grid
    if enabled:
        raise RuntimeError(f"Unexpected feature-first SwiGLU grid {grid}")
    return None


def run_native(train_argv, adapter, native_main, enabled):
    import torch
    import triton
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner
    from torchtitan.ops.float8.triton_kernels import fused_swiglu_quantization as swiglu

    if torch.cuda.get_device_capability() != (9, 0):
        return native_main(train_argv, adapter)
    receipt = validate_receipt(swiglu, torch)
    baseline = swiglu._swiglu_quant_fwd_kernel
    original_forward = staticmethod(swiglu.FusedQuantizedSwiGLU.forward)
    if enabled:
        candidate = make_candidate(baseline, triton.jit)
        if any(
            sha256(kernel.src.encode()).hexdigest() != receipt[key]
            for key, kernel in (
                ("production_kernel_sha256", baseline),
                ("candidate_kernel_sha256", candidate),
            )
        ):
            raise RuntimeError(
                "The native SwiGLU kernels differ from the verified pair"
            )
        selected_forward = make_forward(swiglu, candidate)
    else:
        selected_forward = original_forward

    rank = int(os.environ["RANK"])
    trainer_class = adapter.trainer_class
    forward_backward = trainer_class.forward_backward_step
    original_run = CachingAutotuner.run
    verified = False

    def observe(kernel, *args, **kwargs):
        nonlocal verified
        result = original_run(kernel, *args, **kwargs)
        if not verified and not kwargs.get("benchmark_run", False):
            grid = verified_grid(kernel, args, enabled, WIDTH // swiglu.COL_BLOCK)
            if grid is not None:
                # The pinned CachingAutotuner calls the launcher before it
                # returns. Restore the class method after the first launch.
                verified = True
                CachingAutotuner.run = original_run
                if enabled:
                    print(f"{ACTIVATION} rank={rank} width={WIDTH}", flush=True)
                print(f"Routed SwiGLU launch rank={rank} grid={grid}", flush=True)
        return result

    def verify_first_microbatch(trainer, *args, **kwargs):
        with patch.object(CachingAutotuner, "run", observe):
            result = forward_backward(trainer, *args, **kwargs)
        if not verified:
            raise RuntimeError(
                "No matching routed SwiGLU launch in the first native microbatch"
            )
        if trainer_class.forward_backward_step is verify_first_microbatch:
            trainer_class.forward_backward_step = forward_backward
        return result

    with (
        patch.object(swiglu.FusedQuantizedSwiGLU, "forward", selected_forward),
        patch.object(trainer_class, "forward_backward_step", verify_first_microbatch),
    ):
        result = native_main(train_argv, adapter)
    if not verified:
        raise RuntimeError(
            "The SwiGLU launch audit saw no matching native compiled launch"
        )
    return result


def main():
    enabled = os.environ.get(SWITCH)
    if enabled not in (None, "0", "1"):
        raise ValueError(f"{SWITCH} expects 0 or 1")
    spec = importlib.util.spec_from_file_location(
        "shk_routed_swiglu_flash_measure", BRIDGE
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
