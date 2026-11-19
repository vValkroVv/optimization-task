"""Measure the CE-to-head dependency chain and the expert wgrad kernels in a Kineto trace."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from component_trace import signature, within


def summarize(path):
    raw = path.read_bytes()
    rows = json.loads(raw)["traceEvents"]
    host = {
        row["args"]["External id"]: row
        for row in rows
        if row.get("cat") == "cpu_op" and "External id" in row.get("args", {})
    }
    splits = [row for row in host.values() if row["name"] == "SplitBackward0"]
    streams = defaultdict(list)
    experts = defaultdict(list)
    for row in rows:
        if row.get("cat") not in {"kernel", "gpu_memcpy", "gpu_memset"}:
            continue
        streams[row["args"]["stream"]].append(row)
        name = row["name"]
        if "requantize_grouped_kernel" in name:
            experts["CudaCherry row2col"].append(row["dur"])
        if "_rowwise_to_columnwise_deepgemm_layout_kernel" in name:
            experts["Triton row2col"].append(row["dur"])
        if "sm90_k_grouped_fp8_gemm_rounded_f32" in name:
            experts["direct wgrad"].append(row["dur"])

    def owner(row):
        return host.get(row.get("args", {}).get("External id"), {}).get("name", "")

    chunks = []
    for stream, entries in streams.items():
        entries.sort(key=lambda row: row["ts"])
        # Both arms run the existing per-head stable CE backward kernel.
        ce = [i for i, row in enumerate(entries) if row["name"] == "backward_kernel"]
        if len(ce) % 2:
            raise ValueError(f"Unpaired stable CE backward kernel on stream {stream}")
        for left, right in zip(ce[::2], ce[1::2], strict=True):
            gemms = []
            for i in range(right + 1, len(entries)):
                row = entries[i]
                if row["name"] == "backward_kernel":
                    break
                if owner(row) == "aten::mm" and row["cat"] == "kernel":
                    gemms.append(i)
                    if len(gemms) == 2:
                        break
            if len(gemms) != 2:
                raise ValueError(
                    f"Missing native LM-head backward GEMMs following {right}"
                )
            segment = entries[left : gemms[-1] + 1]
            overhead = [
                row
                for i, row in enumerate(segment, left)
                if i not in {left, right, *gemms}
            ]
            cats = [
                row
                for row in overhead
                if owner(row) == "aten::cat"
                and "CatArrayBatchedCopy" in row["name"]
                and any(
                    within(host[row["args"]["External id"]], split) for split in splits
                )
            ]
            chunks.append(
                {
                    "stream": stream,
                    "gemms": [signature(entries[i]) for i in gemms],
                    "ce": [signature(entries[i]) for i in (left, right)],
                    "ce_to_end_of_gemms_ms": (
                        segment[-1]["ts"] + segment[-1]["dur"] - segment[0]["ts"]
                    )
                    / 1000,
                    "ce_kernel_ms": sum(entries[i]["dur"] for i in (left, right))
                    / 1000,
                    "gemm_ms": sum(entries[i]["dur"] for i in gemms) / 1000,
                    "cat_ms": sum(row["dur"] for row in cats) / 1000,
                    "cat_count": len(cats),
                    "intervening_kernels": [
                        {"owner": owner(row), "name": row["name"], "us": row["dur"]}
                        for row in overhead
                    ],
                }
            )
    if len(chunks) != 6:
        raise ValueError(
            f"Expected six folded chunks in three microbatches; found {len(chunks)}"
        )
    return {
        "trace": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "chunks": chunks,
        "totals": {
            key: sum(row[key] for row in chunks)
            for key in (
                "ce_to_end_of_gemms_ms",
                "ce_kernel_ms",
                "gemm_ms",
                "cat_ms",
                "cat_count",
            )
        },
        "expert_kernels": {
            name: {"count": len(times), "sum_ms": sum(times) / 1000}
            for name, times in experts.items()
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.trace), indent=2))
