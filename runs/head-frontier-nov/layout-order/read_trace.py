"""Summarize warmed Nsight captures with CUDA launch-to-kernel correlation."""

from collections import defaultdict
import json
from pathlib import Path
import sqlite3
import sys


OPERATIONS = ("head_dhidden", "head_dw_including_pack")


def read_trace(path):
    receipt = json.loads(path.with_suffix(".json").read_text())
    chunks, arm = receipt["chunks"], receipt["arm"]
    counts = defaultdict(int)
    kernel_ms = defaultdict(float)
    stream_ids = set()
    launches = []
    transfers = defaultdict(float)
    with sqlite3.connect(path) as connection:
        windows = connection.execute("""
            SELECT text, start, end FROM NVTX_EVENTS
            WHERE text in ('head_dhidden', 'head_dw_including_pack')
            ORDER BY start
        """).fetchall()
        expected = OPERATIONS[::-1] if receipt["dw_first"] else OPERATIONS
        assert tuple(window[0] for window in windows) == expected * chunks

        for op, name, start, end, stream, gx, gy, gz, bx, by, bz in connection.execute("""
            SELECT DISTINCT n.text, s.value, k.start, k.end, k.streamId,
                k.gridX, k.gridY, k.gridZ, k.blockX, k.blockY, k.blockZ
            FROM NVTX_EVENTS n
            JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.globalTid = n.globalTid
                AND r.start >= n.start AND r.end <= n.end
            JOIN CUPTI_ACTIVITY_KIND_KERNEL k ON k.correlationId = r.correlationId
            JOIN StringIds s ON s.id = k.shortName
            WHERE n.text IN ('head_dhidden', 'head_dw_including_pack')
            ORDER BY k.start
        """):
            kind = ("dhidden_gemm" if op == "head_dhidden" else "dw_gemm") if name.startswith("nvjet") else "pack"
            counts[kind] += 1
            kernel_ms[kind] += (end - start) * 1e-6
            stream_ids.add(stream)
            launches.append([kind, name, [gx, gy, gz], [bx, by, bz]])
        for name in ("MEMSET", "MEMCPY"):
            for operation, start, end in connection.execute(f"""
                SELECT DISTINCT n.text, k.start, k.end
                FROM NVTX_EVENTS n
                JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.globalTid = n.globalTid
                    AND r.start >= n.start AND r.end <= n.end
                JOIN CUPTI_ACTIVITY_KIND_{name} k ON k.correlationId = r.correlationId
                WHERE n.text IN ('head_dhidden', 'head_dw_including_pack')
            """):
                transfers[f"{operation}:{name.lower()}"] += (end - start) * 1e-6

    assert len(stream_ids) == 1
    assert counts["dhidden_gemm"] == counts["dw_gemm"] == chunks, counts
    assert counts["pack"] == chunks * ("packed" in arm), counts
    return {
        "chunks": chunks,
        "arm": arm,
        "names_and_launches": launches,
        "kernel_ms": dict(kernel_ms),
        "counts": dict(counts),
        "gemm_sum_ms": kernel_ms["dhidden_gemm"] + kernel_ms["dw_gemm"],
        "gemm_and_pack_sum_ms": sum(kernel_ms.values()),
        "gpu_memset_memcpy_ms": dict(transfers),
        "nvtx_host_range_ms": {
            op: sum(end - start for name, start, end in windows if name == op) * 1e-6
            for op in OPERATIONS
        },
    }


if __name__ == "__main__":
    directories = map(Path, sys.argv[1:])
    result = {path.name: [read_trace(db) for db in sorted(path.glob("*.sqlite"))] for path in directories}
    print(json.dumps(result, indent=2))
