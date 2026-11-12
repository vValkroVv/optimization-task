"""Bounded diagnostic; no training/dependency changes or performance claims."""
import argparse
import ctypes
import hashlib
import inspect
import json
import os
from pathlib import Path
import time

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[6]
CACHE = ROOT / "runs/cuda-graphs-hybridep-fp8-h100/artifacts/raw/cache/deepgemm_shared/cache"
KEY = "kernel.sm90_fp8_gemm_1d1d.500d411276c82c1ae1681a9672269982"
CASES = [([128] * 4, 1024), ([128, 128, 128, 0], 1024), ([32768] * 4, 64)]


def patterns(group, m, n):
    """Integer patterns over 128 active positions; no floating point oracle."""
    k = torch.arange(128)
    rows, cols = torch.arange(m), torch.arange(n)
    p, q = 5 + group, 7 + group
    a = ((k[None, :] + 3 * group) % p == rows[:, None] % p).to(torch.int32)
    b = ((2 * k[None, :] + group) % q == cols[:, None] % q).to(torch.int32)
    sa = 1 - 2 * ((rows + group) % 2)
    sb = 1 - 2 * ((cols // 3 + group) % 2)
    a *= sa[:, None]
    b *= sb[:, None]
    # Count residue intersections independently of the packed matrices.
    counts = torch.zeros((p, q), dtype=torch.int32)
    for position in range(128):
        counts[(position + 3 * group) % p, (2 * position + group) % q] += 1
    expected = counts[rows[:, None] % p, cols[None, :] % q]
    return a, b, expected * sa[:, None] * sb[None, :]


def packed(ks, m, n):
    aa, bb, oracle = [], [], []
    for group, k in enumerate(ks):
        a, b, expected = patterns(group, m, n)
        ap = torch.zeros((m, k), dtype=torch.float8_e4m3fn)
        bp = torch.zeros((n, k), dtype=torch.float8_e4m3fn)
        if k:
            # Spread nonzeros across all K blocks, including production-scale K.
            positions = torch.arange(128) * (k // 128) + (group * 17) % (k // 128)
            ap[:, positions] = a.to(torch.float8_e4m3fn)
            bp[:, positions] = b.to(torch.float8_e4m3fn)
        else:
            expected.zero_()
        aa.append(ap.flatten())
        bb.append(bp.flatten())
        oracle.append(expected)
    return torch.cat(aa), torch.cat(bb), torch.stack(oracle).to(torch.bfloat16)


def cpu_check():
    from torchtitan.ops.float8.deep_gemm_handle import _DeepGemmHandle
    import deep_gemm

    for ks in ([128] * 4, [128, 128, 128, 0], [256] * 4):
        a, b, expected = packed(ks, 13, 11)
        ao = bo = 0
        for group, k in enumerate(ks):
            av = a[ao:ao + 13 * k].reshape(13, k).to(torch.int32)
            bv = b[bo:bo + 11 * k].reshape(11, k).to(torch.int32)
            assert torch.equal(av @ bv.T, expected[group].to(torch.int32))
            ao += 13 * k
            bo += 11 * k
    assert not torch.cuda.is_initialized()
    return {"cpu_packing_oracle": "PASS", "signature": str(inspect.signature(_DeepGemmHandle.k_grouped_fp8_gemm_contiguous)), "schema": str(torch.ops.torchtitan.k_grouped_fp8_gemm_contiguous.default._schema), "binding": deep_gemm.k_grouped_fp8_gemm_nt_contiguous.__doc__, "cuda_initialized": False}


def emit(record):
    print(json.dumps(record), flush=True)
    with (HERE / "events.jsonl").open("a") as stream:
        stream.write(json.dumps(record) + "\n")


def gpu_run():
    from torchtitan.ops.float8.deep_gemm_handle import _DeepGemmHandle
    import deep_gemm

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-c5efeb4a-3298-d3d3-d740-9aeaa1cdd60c"
    assert os.environ["TORCHINDUCTOR_CUDAGRAPHS"] == "0"
    assert os.environ["TORCHINDUCTOR_MIX_ORDER_REDUCTION"] == "0"
    handle = _DeepGemmHandle(num_sms=116)
    assert deep_gemm.get_num_sms() == 116
    started = time.monotonic()
    total = 0
    retained = []
    for case, (ks, calls) in enumerate(CASES, 1):
        a, b, expected = packed(ks, 1536, 1024)
        a, b, expected = a.cuda(), b.cuda(), expected.cuda()
        blocks = sum(ks) // 128
        sa = torch.ones((blocks, 1536), dtype=torch.float32, device="cuda").T
        sb = torch.ones((blocks, 1024), dtype=torch.float32, device="cuda").T
        kc = torch.tensor(ks, dtype=torch.int32)
        kd = kc.cuda()
        outputs = [torch.empty_like(expected) for _ in range(32)]
        retained.append((a, b, sa, sb, kc, kd, expected, outputs))
        completed = 0
        while completed < calls:
            assert time.monotonic() - started < 120, "120-second fixture cap"
            chunk = 1 if completed == 0 else min(32, calls - completed)
            emit({"event": "submit", "case": case, "first_call": completed + 1, "chunk": chunk, "total_before": total, "ks": ks})
            log_start = (HERE / "stdout.log").stat().st_size
            for index in range(chunk):
                outputs[index].zero_()
                handle.k_grouped_fp8_gemm_contiguous((a, sa), (b, sb), outputs[index], ks, kc, kd)
                total += 1
            torch.cuda.synchronize()
            ctypes.CDLL(None).fflush(None)
            if completed == 0:
                log = (HERE / "stdout.log").read_text()[log_start:]
                loads = [line for line in log.splitlines() if "Loading CUBIN:" in line]
                # Subsequent cases can reuse the same in-process specialization.
                assert loads or case > 1, "No observed loaded-CUBIN identity; admission inconclusive"
                admitted = []
                for line in loads:
                    binary = Path(line.split("Loading CUBIN:", 1)[1].strip())
                    source = binary.parent / "kernel.cu"
                    assert source.read_bytes() == (CACHE / KEY / "kernel.cu").read_bytes(), f"Template mismatch: {line}"
                    admitted.append({"binary": str(binary), "sha256": hashlib.sha256(binary.read_bytes()).hexdigest(), "metadata": (binary.parent / "kernel_meta.txt").read_text()})
                emit({"event": "admission", "case": case, "loads": loads, "reused_in_process": not loads, "exact_template": KEY, "actual_modules": admitted})
            bad = torch.stack([(out != expected).any() for out in outputs[:chunk]]).cpu().tolist()
            assert not any(bad), f"Wrong result case={case} calls={[completed+i+1 for i,v in enumerate(bad) if v]}"
            completed += chunk
            emit({"event": "checked", "case": case, "completed": completed, "total": total, "every_output_checked": True})
    assert total == 2112
    emit({"event": "result", "status": "CLEAN_NONREPRODUCTION", "calls": total})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    if args.gpu:
        try:
            gpu_run()
        except BaseException as exc:
            emit({"event": "failure", "exception": repr(exc)})
            raise
    else:
        result = cpu_check()
        result["file_sha256"] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in [Path(__file__), CACHE / KEY / "kernel.cu", CACHE / KEY / "kernel_meta.txt", CACHE / KEY / "kernel.cubin"]}
        (HERE / "cpu-check.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
