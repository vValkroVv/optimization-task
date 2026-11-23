"""Compile or replay plain CuTe F1 plus F2 on one Hopper GPU."""

import argparse
import json
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch

import torch

from norm_f1_component import load_variant, measure_pairs
from norm_pair_tile_reuse_component import prepare
from norm_skip import NATIVE_SHAPE, digest, mismatched_bytes, restore


def compile_offline(candidate, args):
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_stream

    def build(key, make_op, tensors):
        if key[0] != "fwd_plain_cluster":
            raise RuntimeError(f"Unexpected compile key {key}")
        return cute.compile(
            make_op(),
            *tensors,
            make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi --gpu-arch sm_90a --generate-line-info --keep-ptx --keep-cubin",
        )

    with patch.object(candidate, "compile_op", build):
        compiled = candidate.compile_norm_same_layer_cluster(1e-6, 2.0, args.clusters)
    ptx, cubin = compiled.__ptx__, compiled.__cubin__
    if not ptx or not isinstance(cubin, bytes) or not cubin:
        raise RuntimeError("Expected PTX and CUBIN from the SM90a compile")
    args.output.with_suffix(".ptx").write_text(ptx)
    args.output.with_suffix(".cubin").write_bytes(cubin)
    if compiled.__mlir__:
        args.output.with_suffix(".mlir").write_text(str(compiled.__mlir__))
    for instruction in (
        "mapa.shared::cluster.u32",
        "ld.shared::cluster.b16",
        "ld.shared::cluster.f32",
    ):
        if instruction not in ptx:
            raise RuntimeError(f"Expected {instruction} in PTX")
    for instruction in ("barrier.cluster.arrive", "barrier.cluster.wait"):
        if ptx.count(instruction) < 3:
            raise RuntimeError(f"Expected a {instruction} for each of the three phases")


@torch.no_grad()
def replay(forward, candidate, args):
    from torchtitan.models.common.norms.cute_gated_norm.op import _aligned

    if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability(0) != (9, 0):
        raise RuntimeError("Expose exactly one Hopper GPU")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    budget = forward.resolve_num_sms(device)
    fixture = torch.load(args.fixture, map_location="cpu", mmap=True, weights_only=True)
    if fixture["torch_revision"] != torch.version.git_version:
        raise RuntimeError("The fixture and replay use different PyTorch revisions")
    record = fixture["records"]["plain"]
    x = _aligned(restore(record["tensors"]["x"], record["layouts"]["x"]))
    if x.shape != NATIVE_SHAPE or x.dtype != torch.bfloat16 or budget != 116:
        raise RuntimeError(
            "Expected native contiguous BF16 (2, 4096, 4096) and budget 116"
        )
    x = x.view(-1, 4096)[: args.rows]
    dtypes = tuple(record["tensors"][name].dtype for name in ("up", "weight", "down"))
    eps, lgw = record["eps"], record["gating_weight"]
    prep, f1, f2, split = forward._fwd_plan(
        4096, 16, dtypes, eps, lgw, device, 4, budget
    )
    if split != 8 or forward.pick_f2_stages(4096, 4) != 12:
        raise RuntimeError(
            "The control must use eight F1 splits and four 12-stage F2 chunks"
        )
    w_eff, w_f, gd_bf = prepare(prep, record)
    fused = candidate.compile_norm_same_layer_cluster(eps, lgw, args.clusters)
    partials = forward.workspace(device, [(split, args.rows, 16), (split, args.rows)])
    outputs = (
        torch.empty_like(x),
        torch.empty((args.rows,), dtype=torch.float32, device=device),
        torch.empty((args.rows, 16), dtype=torch.float32, device=device),
    )

    def baseline(_):
        f1(x, w_eff, *partials)
        f2(x, w_f, gd_bf, *partials, *outputs, x)
        return outputs

    def changed(_):
        fused(x, w_eff, w_f, gd_bf, *outputs)
        return outputs

    for tensor in (*partials, *outputs):
        tensor.fill_(float("nan"))
    baseline(None)
    reference = tuple(tensor.clone() for tensor in outputs)
    if not all(tensor.isfinite().all().item() for tensor in reference):
        raise RuntimeError("Baseline did not produce complete finite outputs")
    for tensor in outputs:
        tensor.fill_(float("nan"))
    changed(None)
    mismatches = mismatched_bytes(("y", "rho", "p"), reference, outputs)
    del reference
    pairs = (
        measure_pairs({"A": baseline, "B": changed}, args)
        if args.pairs and not any(mismatches.values())
        else []
    )
    receipt = {
        "gpu": torch.cuda.get_device_name(),
        "torch_revision": torch.version.git_version,
        "cutlass_dsl": version("nvidia-cutlass-dsl"),
        "fixture_sha256": digest(args.fixture),
        "baseline_sha256": digest(Path(forward.__file__)),
        "candidate_sha256": digest(args.candidate),
        "runner_sha256": digest(Path(__file__)),
        "rows": args.rows,
        "requested_clusters": args.clusters,
        "sm_budget": budget,
        "mismatched_bytes": mismatches,
        "warmup": args.warmup,
        "samples": args.samples,
        "pairs": pairs,
    }
    with args.output.open("x") as handle:
        json.dump(receipt, handle, indent=2)
        handle.write("\n")
    if any(mismatches.values()):
        raise AssertionError(
            f"Plain norm changed outputs at {args.rows} rows: {mismatches}"
        )


def main():
    from torchtitan.models.common.norms.cute_gated_norm import forward

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate", type=Path, required=True, help="patched forward.py"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, help="omit for offline compilation")
    parser.add_argument("--rows", type=int, choices=(65, 8192), default=65)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument(
        "--pairs", type=int, default=0, help="paired timing at 8192 rows only"
    )
    args = parser.parse_args()
    if args.pairs < 0 or min(args.warmup, args.samples) < 1:
        parser.error("Invalid timing counts")
    if args.pairs and (not args.fixture or args.rows != 8192):
        parser.error("Paired timing requires the native fixture and --rows 8192")
    args.clusters = 1 if args.rows == 65 else 14
    args.output.parent.mkdir(parents=True, exist_ok=True)
    candidate = load_variant(forward, args.candidate, "forward_same_layer_cluster")
    if args.fixture:
        replay(forward, candidate, args)
    else:
        compile_offline(candidate, args)


if __name__ == "__main__":
    main()
