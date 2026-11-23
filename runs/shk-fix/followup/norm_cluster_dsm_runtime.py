"""Replay the offline DSM probe once on a single visible SM90 GPU."""

import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream

import norm_cluster_dsm_probe as probe

ROUNDS, RANKS, LANES = 2, 8, 32
SHAPE = (ROUNDS, RANKS, LANES)
PROBE_SHA256 = "0460a007413c4029b0252f8cf87269f366159f030cca3fe636aff553decf717e"


def fixture():
    src = torch.empty(SHAPE, dtype=torch.float32, device="cpu")
    for tile in range(ROUNDS):
        sign = 1 - 2 * tile
        for lane in range(LANES):
            terms = (2**24, 1, -(2**24), 3, 5, 7, 9, 2 * lane + 11)
            # Rotate which peer contributes each term and give both rounds distinct data.
            for rank in range(RANKS):
                src[tile, rank, lane] = sign * terms[(rank + lane + 3 * tile) % RANKS]

    expected = torch.zeros((ROUNDS, LANES), dtype=torch.float32, device="cpu")
    for rank in range(RANKS):
        expected.add_(src[:, rank, :])
    exact_sum = src.sum(dim=1, dtype=torch.float64).to(torch.float32)
    order_sensitive = (expected != exact_sum).sum(dim=1).tolist()
    if not all(order_sensitive):
        raise AssertionError(
            "Both rounds must distinguish sequential FP32 from exact sums"
        )
    return src, expected, order_sensitive


def main():
    probe_sha256 = hashlib.sha256(Path(probe.__file__).read_bytes()).hexdigest()
    dsl_version = version("nvidia-cutlass-dsl")
    if probe_sha256 != PROBE_SHA256 or dsl_version != "4.4.1":
        raise RuntimeError("Expected the archived probe source and CuTe DSL 4.4.1")
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability(0) != (9, 0):
        raise RuntimeError("Expose exactly one SM90 GPU")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    fake_src, fake_dst = (
        make_fake_compact_tensor(
            cutlass.Float32, SHAPE, stride_order=(2, 1, 0), assumed_align=16
        )
        for _ in range(2)
    )
    run = cute.compile(
        probe.launch,
        fake_src,
        fake_dst,
        make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi --gpu-arch sm_90a --generate-line-info",
    )

    src, expected, order_sensitive = fixture()
    poisoned = torch.full_like(src, float("nan"))
    src_dev, dst_dev = src.to(device), poisoned.to(device)
    run(src_dev, dst_dev)
    torch.cuda.synchronize(device)

    actual_bits = dst_dev.cpu().view(torch.int32)
    expected_bits = expected.view(torch.int32).unsqueeze(1)
    mismatch_by_round_rank = (actual_bits != expected_bits).sum(dim=-1).tolist()
    print(
        json.dumps(
            {
                "probe_sha256": probe_sha256,
                "cutlass_dsl": dsl_version,
                "gpu": torch.cuda.get_device_name(device),
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "shape": SHAPE,
                "order_sensitive_lanes_by_round": order_sensitive,
                "mismatched_elements_by_round_rank": mismatch_by_round_rank,
            },
            indent=2,
        ),
        flush=True,
    )
    if any(count for row in mismatch_by_round_rank for count in row):
        raise AssertionError("DSM output differs from the CPU FP32 left fold")


if __name__ == "__main__":
    main()
