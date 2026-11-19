"""Check the compiled CE-to-LM-head chain from a saved component Kineto trace."""

import argparse
import hashlib
import json
from pathlib import Path

MICROBATCHES = 3
CHUNKS = MICROBATCHES * 2


def within(event, region):
    return (
        (event["pid"], event["tid"]) == (region["pid"], region["tid"])
        and region["ts"] <= event["ts"]
        and event["ts"] + event["dur"] <= region["ts"] + region["dur"]
    )


def signature(kernel):
    return {
        "name": kernel["name"],
        "grid": kernel["args"]["grid"],
        "block": kernel["args"]["block"],
    }


def summarize(path, variant, rows_per_head=4096):
    raw = path.read_bytes()
    events = json.loads(raw)["traceEvents"]
    cpu = [event for event in events if event.get("cat") == "cpu_op"]
    by_id = {
        event["args"]["External id"]: event
        for event in cpu
        if "External id" in event.get("args", {})
    }
    splits = [event for event in cpu if event["name"] == "SplitBackward0"]
    backwards = [event for event in cpu if event["name"] == "CompiledFunctionBackward"]

    def owner(event):
        return by_id.get(event["args"].get("External id"), {})

    device = sorted(
        (
            event
            for event in events
            if event.get("cat") in {"kernel", "gpu_memcpy", "gpu_memset"}
        ),
        key=lambda event: event["ts"],
    )
    stream_keys = {(event["pid"], event["args"]["stream"]) for event in device}
    microbatch_names = [
        event["name"]
        for event in events
        if event.get("cat") == "user_annotation"
        and event["name"].startswith("mtp_head_microbatch_")
    ]
    if len(stream_keys) != 1 or sorted(microbatch_names) != [
        f"mtp_head_microbatch_{index}" for index in range(MICROBATCHES)
    ]:
        raise ValueError(
            "Expected a single GPU stream and three component microbatches"
        )

    ce_indices = [
        i
        for i, event in enumerate(device)
        if event["cat"] == "kernel" and event["name"] == "backward_kernel"
    ]
    gemm_indices = [
        i
        for i, event in enumerate(device)
        if event["cat"] == "kernel"
        and owner(event).get("name") == "aten::mm"
        and any(within(owner(event), region) for region in backwards)
    ]
    if len(ce_indices) != 2 * CHUNKS or len(gemm_indices) != 2 * CHUNKS:
        raise ValueError(
            "Expected two stable CE kernels and two head backward GEMMs per chunk"
        )

    chunks = []
    cat_count = 0
    for chunk in range(CHUNKS):
        ce_first, ce_last = ce_indices[chunk * 2 : chunk * 2 + 2]
        gemm_first, gemm_last = gemm_indices[chunk * 2 : chunk * 2 + 2]
        if not ce_first < ce_last < gemm_first < gemm_last or (
            chunk < CHUNKS - 1 and gemm_last >= ce_indices[(chunk + 1) * 2]
        ):
            raise ValueError(f"Unexpected CE/GEMM order in chunk {chunk + 1}")

        ce = [device[ce_first], device[ce_last]]
        gemms = [device[gemm_first], device[gemm_last]]
        if any(event["args"]["grid"] != [rows_per_head, 1, 1] for event in ce):
            raise ValueError(f"Unexpected CE grid in chunk {chunk + 1}")
        if not any(
            all(within(owner(event), region) for event in gemms) for region in backwards
        ):
            raise ValueError(
                f"GEMMs did not come from the same head backward in chunk {chunk + 1}"
            )

        cats = []
        other = []
        unexpected = []
        for index in range(ce_first, gemm_last + 1):
            if index in (ce_first, ce_last, gemm_first, gemm_last):
                continue
            event = device[index]
            host = owner(event)
            name = host.get("name")
            if (
                index < gemm_first
                and name == "aten::cat"
                and "CatArrayBatchedCopy" in event["name"]
                and any(within(host, region) for region in splits)
            ):
                cats.append(event)
            elif (
                name in ("aten::div", "aten::mul")
                and event["args"].get("grid") == [1, 1, 1]
            ) or (
                name == "aten::mm"
                and event["cat"] == "gpu_memset"
                and event["args"].get("bytes") == 4
            ):
                other.append(event)
            else:
                unexpected.append(
                    {
                        "kind": event["cat"],
                        "name": event["name"],
                        "owner": name,
                        "duration_us": event["dur"],
                    }
                )
        cat_count += len(cats)
        segment = device[ce_first : gemm_last + 1]
        span = segment[-1]["ts"] + segment[-1]["dur"] - segment[0]["ts"]
        # The intervals are on one serial CUDA stream.
        gaps = sum(
            max(0.0, right["ts"] - (left["ts"] + left["dur"]))
            for left, right in zip(segment, segment[1:])
        )
        chunks.append(
            {
                "ce_kernel_signatures": [signature(event) for event in ce],
                "head_backward_gemm_signatures": [signature(event) for event in gemms],
                "ce_to_end_of_gemms_ms": span / 1000,
                "ce_kernel_ms": sum(event["dur"] for event in ce) / 1000,
                "gemm_ms": sum(event["dur"] for event in gemms) / 1000,
                "split_cat_ms": sum(event["dur"] for event in cats) / 1000,
                "split_cat_count": len(cats),
                "other_device_ms": sum(event["dur"] for event in other) / 1000,
                "device_gap_ms": gaps / 1000,
                "unexpected_intervening_device_events": unexpected,
            }
        )

    expected_cat = 1 if variant == "base" else 0
    return {
        "variant": variant,
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "microbatches": len(microbatch_names),
        "ce_backward_kernel_calls": len(ce_indices),
        "ce_rows_per_head": rows_per_head,
        "head_backward_gemms": len(gemm_indices),
        "split_backward_calls": len(splits),
        "split_cat_count": cat_count,
        "chunks": chunks,
        "totals": {
            key: sum(chunk[key] for chunk in chunks)
            for key in (
                "ce_to_end_of_gemms_ms",
                "ce_kernel_ms",
                "gemm_ms",
                "split_cat_ms",
                "other_device_ms",
                "device_gap_ms",
            )
        },
        "passed": len(splits) == CHUNKS * expected_cat
        and all(chunk["split_cat_count"] == expected_cat for chunk in chunks)
        and not any(chunk["unexpected_intervening_device_events"] for chunk in chunks),
    }


def compare(base, folded):
    kernel_keys = ("ce_kernel_signatures", "head_backward_gemm_signatures")
    signatures_match = all(
        original[key] == candidate[key]
        for original, candidate in zip(base["chunks"], folded["chunks"], strict=True)
        for key in kernel_keys
    )
    return {
        "base": base,
        "folded": folded,
        "kernel_signatures_match": signatures_match,
        "passed": base["passed"] and folded["passed"] and signatures_match,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--folded", required=True, type=Path)
    args = parser.parse_args()
    result = compare(summarize(args.base, "base"), summarize(args.folded, "folded"))
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
