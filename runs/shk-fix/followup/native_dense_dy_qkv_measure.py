"""Run the native GIFT measurement worker with the opt-in QKV dY candidate."""

import ast
from contextlib import nullcontext
import importlib.util
import json
import os
from pathlib import Path
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
BRIDGE = HERE.parents[1] / "flash-qla/measure.py"
SWITCH = "GIFT_FP8_DENSE_DY_QKV_FUSION"
ACTIVATION = "Dense QKV dY fusion verified in %d compiled site(s)"
LAUNCH_ACTIVATION = "Dense QKV dY fusion executed"
KERNEL = "dense_dy_quant_transpose_kernel"
DY_SHAPE = (2, 4096, 8192)
X_SHAPE = (2, 4096, 4096)
WEIGHT_SHAPE = (8192, 4096)


def build_candidate(torch, cuda_gemm, native, layouts, quantize_pair):
    original = cuda_gemm.FP8Gemm

    @torch._dynamo.allow_in_graph
    class DenseDyQkvGemm(original):
        # Inherit the original forward, context initialization and save set.
        @staticmethod
        def backward(ctx, grad_output):
            if (
                not ctx.needs_wgrad
                or not ctx.act_requires_grad
                or ctx.has_residual
                or ctx.dtype != torch.bfloat16
                or ctx.w_shape != WEIGHT_SHAPE
                or ctx.inp_cur_shape != X_SHAPE
                or ctx.quantize_weight_tail
                or ctx.wgrad_logical_width
                or not isinstance(grad_output, torch.Tensor)
                or cuda_gemm.is_quantized_tensor(grad_output)
                or grad_output.device.type != "cuda"
                or grad_output.dtype != torch.bfloat16
                or grad_output.shape != DY_SHAPE
                or not grad_output.is_contiguous()
                or grad_output.storage_offset() % 8 != 0
            ):
                return original.backward(ctx, grad_output)

            x_row, weight = cuda_gemm.restore_from_saved(
                ctx.tensor_objects, ctx.saved_tensors
            )
            if (
                weight.device != grad_output.device
                or weight.dtype != torch.bfloat16
                or weight.shape != WEIGHT_SHAPE
                or not weight.is_contiguous()
            ):
                return original.backward(ctx, grad_output)

            ctx.tensor_objects = None
            weight_col = native.quantize_blk128x128(
                weight, transpose_output=True, pack=False
            )
            qr, sr, qc, sc = quantize_pair(grad_output.view(-1, DY_SHAPE[-1]))
            row = (qr.view(DY_SHAPE), sr, layouts.ROWWISE, 1)
            col = (qc.view(DY_SHAPE[-1], *DY_SHAPE[:-1]), sc, layouts.COLUMNWISE, 1)
            dx = native.fp8_gemm(
                row, weight_col, out_dtype=ctx.dtype, use_split_accumulator=True
            )
            cuda_gemm.clear_tensor_data(weight_col, row)
            x_col = native.scaling_aware_fp8_transpose(x_row, pack=False)
            dw = native.fp8_gemm(
                x_col, col, out_dtype=ctx.dtype, use_split_accumulator=True
            )
            cuda_gemm.clear_tensor_data(x_col, col)
            return (dx, dw, *(None,) * 8)

    return DenseDyQkvGemm


def buffer_name(value):
    if isinstance(value, ast.Name):
        return value.id
    if isinstance(value, ast.Call) and value.args:
        if ast.unparse(value.func) in {
            "reinterpret_tensor",
            "reinterpret_tensor_wrapper",
            "aten.view.dtype",
            "torch.ops.aten.view.dtype",
        }:
            return buffer_name(value.args[0])
    return None


def compiled_sites(source):
    """Find the four launch outputs feeding the two native gradient GEMMs."""
    if KERNEL not in source:
        return []
    sites = []
    for fn in ast.walk(ast.parse(source)):
        if not isinstance(fn, ast.FunctionDef):
            continue
        calls = sorted(
            (node for node in ast.walk(fn) if isinstance(node, ast.Call)),
            key=lambda node: node.lineno,
        )
        gemms = [
            node
            for node in calls
            if ast.unparse(node.func) == "torch.ops.torchtitan.fp8_dense_gemm.default"
            and len(node.args) >= 8
        ]
        for launch in calls:
            target = launch.func
            if (
                not isinstance(target, ast.Attribute)
                or target.attr != "run"
                or not isinstance(target.value, ast.Name)
                or not target.value.id.startswith(KERNEL)
                or len(launch.args) < 5
            ):
                continue
            row, row_scale, col, col_scale = map(buffer_name, launch.args[1:5])
            if None in (row, row_scale, col, col_scale):
                continue
            dgrads = [
                gemm.lineno
                for gemm in gemms
                if gemm.lineno > launch.lineno
                and tuple(map(buffer_name, gemm.args[:2])) == (row, row_scale)
                and tuple(map(ast.unparse, gemm.args[2:4])) == ("0", "1")
                and tuple(map(ast.unparse, gemm.args[6:8])) == ("1", "2")
            ]
            wgrads = [
                gemm.lineno
                for gemm in gemms
                if gemm.lineno > launch.lineno
                and tuple(map(buffer_name, gemm.args[4:6])) == (col, col_scale)
                and tuple(map(ast.unparse, gemm.args[2:4])) == ("1", "1")
                and tuple(map(ast.unparse, gemm.args[6:8])) == ("1", "1")
            ]
            if len(dgrads) == len(wgrads) == 1 and dgrads[0] < wgrads[0]:
                sites.append(
                    {"launch": launch.lineno, "dgrad": dgrads[0], "wgrad": wgrads[0]}
                )
    return sites


def launch_grid(kernel, args):
    meta = kernel.inductor_meta
    names = (getattr(kernel.fn, "__name__", ""), meta.get("kernel_name") or "")
    if not any(KERNEL in name for name in names):
        return None
    if meta.get("grid_type") != "FixedGrid" or meta.get("extra_launcher_args") != [
        "_grid_0",
        "_grid_1",
        "_grid_2",
    ]:
        raise RuntimeError("Unexpected Inductor grid ABI for dense QKV dY")
    grid = tuple(args[-3:])
    if grid != (64, 64, 1):
        raise RuntimeError(f"Unexpected dense QKV dY grid {grid}")
    return grid


def run_native(train_argv, adapter, native_main, measurement, enabled):
    if any(
        os.environ.get(name) != "1"
        for name in (
            "GIFT_CUTE_NORM_FUSE_RESIDUAL",
            "GIFT_CUTE_NORM_SAC_STATS_ONLY",
            "GIFT_FP8_QUANTIZE_WEIGHT_TAIL",
        )
    ):
        raise ValueError("The dense QKV candidate needs the three-fix preset")

    import torch
    from torch._inductor.codecache import PyCodeCache
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner
    from torchtitan.ops.float8 import _cudacherry, cuda_gemm
    from torchtitan.ops.float8.fp8_tensor import InputLayoutType

    import dense_dy_quant_transpose as quant_probe

    if torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("The dense QKV candidate requires Hopper")
    if (
        adapter.options["capture_microbatches"]
        or adapter.options["profile_update"] is not None
    ):
        raise RuntimeError("The dense QKV launch audit requires unprofiled measurement")

    rank = int(os.environ["RANK"])
    trainer_class = adapter.trainer_class
    forward_backward = trainer_class.forward_backward_step
    original_run = CachingAutotuner.run
    grid = None

    def observe(kernel, *args, **kwargs):
        nonlocal grid
        launched = original_run(kernel, *args, **kwargs)
        if grid is None and not kwargs.get("benchmark_run", False):
            grid = launch_grid(kernel, args)
            if grid is not None:
                if not enabled:
                    raise RuntimeError(
                        "The dense QKV dY kernel ran in the baseline arm"
                    )
                CachingAutotuner.run = original_run
                print(f"{LAUNCH_ACTIVATION} rank={rank} grid={grid}", flush=True)
        return launched

    def verify_first_microbatch(trainer, *args, **kwargs):
        with patch.object(CachingAutotuner, "run", observe):
            output = forward_backward(trainer, *args, **kwargs)
        if enabled and grid is None:
            raise RuntimeError("No dense QKV dY launch in the first native microbatch")
        if trainer_class.forward_backward_step is verify_first_microbatch:
            trainer_class.forward_backward_step = forward_backward
        return output

    selection = nullcontext()
    if enabled:
        candidate = build_candidate(
            torch, cuda_gemm, _cudacherry, InputLayoutType, quant_probe.candidate
        )
        selection = patch.object(cuda_gemm, "FP8Gemm", candidate)
    with (
        selection,
        patch.object(trainer_class, "forward_backward_step", verify_first_microbatch),
    ):
        result = native_main(train_argv, adapter)
    if enabled and grid is None:
        raise RuntimeError("No native dense QKV dY launch was observed")

    # Inspect only wrappers loaded by this worker, after the measured updates.
    compiled = []
    for path in sorted({Path(module.__file__) for module in PyCodeCache.modules}):
        source = path.read_text()
        sites = compiled_sites(source)
        if sites:
            compiled.append(
                {
                    "wrapper": str(path),
                    "sha256": quant_probe.digest(path),
                    "sites": sites,
                }
            )
    count = sum(len(entry["sites"]) for entry in compiled)
    receipt = {
        "rank": rank,
        "switch": SWITCH,
        "enabled": enabled,
        "candidate_sha256": quant_probe.digest(quant_probe.__file__),
        "dy_shape": list(DY_SHAPE),
        "weight_shape": list(WEIGHT_SHAPE),
        "x_shape": list(X_SHAPE),
        "native_launch_grid": grid,
        "compiled_sites": count,
        "wrappers": compiled,
    }
    path = measurement / f"dense-dy-qkv-rank-{rank:02d}.json"
    with path.open("x") as handle:
        json.dump(receipt, handle, indent=2)
        handle.write("\n")
    if bool(count) != enabled:
        raise RuntimeError(
            f"Dense QKV candidate wrapper verification failed. See {path}"
        )
    if enabled:
        print(ACTIVATION % count, flush=True)
    return result


def main():
    enabled = os.environ.get(SWITCH)
    if enabled not in (None, "0", "1"):
        raise ValueError("Expected the binary dense QKV experiment switch")
    spec = importlib.util.spec_from_file_location(
        "shk_dense_dy_qkv_flash_measure", BRIDGE
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
        def native_with_candidate(argv, current_adapter):
            return run_native(
                argv,
                current_adapter,
                native_main,
                bridge._measurement_dir(current_adapter),
                enabled == "1",
            )

        return bootstrap(train_argv, adapter, native_with_candidate)

    with patch.object(bridge, "_bootstrap_and_run", start):
        return bridge.main()


if __name__ == "__main__":
    main()
