#!/usr/bin/env python3
"""Validate and compare final Expert GFSDP captures from update 11."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

from analyze_ep4 import (
    API,
    CPU,
    DIRECT_WRITE,
    DIRECT_ZERO,
    GPU,
    Event,
    Launches,
    annotated_phase,
    collective_kind,
    host_ancestry,
    microbatch_at,
    normalize_ranks,
    scope_fqn,
    wait_rows,
)

RANKS = (0, 4)
WORLD_SIZE = 8
UPDATE = 11
ARMS = {"EP8-A": 8, "EP8-B": 8, "EP4-A": 4, "EP4-B": 4}
EXPERT_PARAMS = 301_989_888
EXPERT_SCALES = 18_432
EXCHANGE_CHUNK = 150_994_944
OWNER_SCOPE = "GFSDP::bf16_slot_owner_reduce"
LEGACY_DW = "torchtitan::k_grouped_fp8_gemm_contiguous"
DTYPES = {"byte": "uint8", "float": "float32", "bf16": "bfloat16"}
WIDTHS = {"uint8": 1, "bfloat16": 2, "float32": 4}
COLLECTIVE_FIELDS = (
    "Process Group Name",
    "Process Group Ranks",
    "Group size",
    "Seq",
    "dtype",
    "In msg nelems",
    "Out msg nelems",
)
COMPUTE = {
    "other_compute",
    "grouped_fp8_gemm",
    "legacy_dw",
    "direct_dw",
    "expert_weight_quant",
    "prepared_weight_transpose",
    "wgrad_row2col",
}
HOST_BLOCK = "synchronize"


def ms(ns):
    return round(ns / 1_000_000, 6)


def merged(intervals):
    result = []
    for start, end in sorted(intervals):
        if start >= end:
            continue
        if result and result[-1][1] >= start:
            previous, previous_end = result[-1]
            result[-1] = previous, max(end, previous_end)
        else:
            result.append((start, end))
    return result


def union(events):
    return merged((event.start, event.end) for event in events)


def elapsed(intervals):
    return sum(end - start for start, end in intervals)


def intersection(left, right):
    total = first = second = 0
    while first < len(left) and second < len(right):
        start, end = left[first]
        other_start, other_end = right[second]
        total += max(0, min(end, other_end) - max(start, other_start))
        if end <= other_end:
            first += 1
        else:
            second += 1
    return total


def by_key(rows, key):
    groups = defaultdict(list)
    for row in rows:
        groups[key(row)].append(row)
    return groups


def process_group_metadata(event, scopes, inventory):
    host_record = next(
        (scope for scope in scopes if scope.name == "record_param_comms"), None
    )
    args = dict(event.args)
    conflicts = []
    for field in COLLECTIVE_FIELDS:
        if host_record is None or field not in host_record.args:
            continue
        host_value = host_record.args[field]
        if field not in args:
            args[field] = host_value
        elif field == "Process Group Ranks":
            if normalize_ranks(args[field]) != normalize_ranks(host_value):
                conflicts.append(field)
        elif args[field] != host_value:
            conflicts.append(field)
    name = args.get("Process Group Name")
    members = normalize_ranks(args.get("Process Group Ranks"))
    description, inventory_members = inventory.get(str(name), (None, None))
    if (
        members is not None
        and inventory_members is not None
        and members != inventory_members
    ):
        conflicts.append("process group inventory ranks")
    members = members or inventory_members
    size = args.get("Group size", len(members) if members else None)
    if members and size != len(members):
        conflicts.append("group size")
    dtype = str(args.get("dtype", "")).lower().removeprefix("torch.")
    return {
        "pg": str(name) if name is not None else None,
        "pg_desc": description,
        "members": members,
        "size": size,
        "seq": args.get("Seq"),
        "dtype": DTYPES.get(dtype, dtype) or None,
        "input_elements": args.get("In msg nelems"),
        "output_elements": args.get("Out msg nelems"),
        "conflicts": conflicts,
    }


def expected_expert_signature(kind, meta, ep):
    params = EXPERT_PARAMS * (8 // ep)
    scales = EXPERT_SCALES * (8 // ep)
    signature = kind, meta["dtype"], meta["input_elements"], meta["output_elements"]
    return {
        ("all_gather", "bfloat16", EXPERT_PARAMS, params): "expert_bf16_ag",
        ("all_gather", "uint8", EXPERT_PARAMS, params): "expert_fp8_data_ag",
        ("all_gather", "float32", EXPERT_SCALES, scales): "expert_fp32_scales_ag",
        ("reduce_scatter", "float32", params, EXPERT_PARAMS): "expert_fp32_rs",
        (
            "peer_exchange",
            "bfloat16",
            EXCHANGE_CHUNK,
            EXCHANGE_CHUNK,
        ): "expert_bf16_owner_exchange",
    }.get(signature)


def classify(event, scopes, inventory, members, ep):
    names = {scope.name for scope in scopes}
    gf_owners = [
        (prefix, owner)
        for scope in scopes
        for prefix, owner in [scope_fqn(scope.name)]
        if owner and prefix.startswith(("GFSDP::", "FSDP::"))
    ]
    expert_owner = next(
        (owner for _, owner in gf_owners if owner.endswith("[efsdp]")), None
    )
    compact_scope = any(prefix == OWNER_SCOPE for prefix, _ in gf_owners)
    name = event.name.lower()
    kind = collective_kind(event) if event.category == "kernel" else None
    if kind == "expert_token_exchange":
        for token, family in (
            ("dispatch_kernel", "ep_dispatch"),
            ("combine_kernel", "ep_combine"),
            ("device_sync", "ep_device_wait"),
            ("update_expected", "ep_sync_setup"),
            ("::scan<", "ep_metadata"),
        ):
            if token in name:
                return family, None, expert_owner
        return "ep_other", None, expert_owner

    if kind:
        meta = process_group_metadata(event, scopes, inventory)
        if kind == "other_nccl" and (
            "sendrecv" in name or "alltoall" in name or compact_scope
        ):
            kind = "peer_exchange"
        is_expert_pg = meta["pg_desc"] == "mesh_efsdp" or (
            meta["pg_desc"] is None and meta["members"] == members and ep == 4
        )
        if expert_owner is not None or is_expert_pg:
            family = expected_expert_signature(kind, meta, ep)
            if meta["conflicts"] or (
                meta["members"] is not None and meta["members"] != members
            ):
                family = None
            if (
                kind == "peer_exchange"
                and compact_scope
                and not meta["conflicts"]
                and (
                    meta["members"] in (None, members)
                    and meta["dtype"] in (None, "bfloat16")
                )
            ):
                # The owner scope has exactly two typed exchanges and no other NCCL operations.
                family = "expert_bf16_owner_exchange"
            return family or "expert_collective_unresolved", meta, expert_owner
        if meta["size"] == WORLD_SIZE and kind in ("all_gather", "reduce_scatter"):
            return f"dense_{kind}", meta, next((owner for _, owner in gf_owners), None)
        return f"other_nccl_{kind}", meta, None

    if compact_scope and "_write_owner" in name:
        return "expert_owner_fp32_add", None, expert_owner
    if DIRECT_ZERO in names or (
        DIRECT_WRITE in names
        and (event.category == "gpu_memset" or "aten::zero_" in names)
    ):
        return "direct_dw_zero", None, expert_owner
    if "sm90_k_grouped_fp8_gemm_rounded_f32" in name:
        return "direct_dw", None, expert_owner
    if "sm90_k_grouped_fp8_gemm" in name or (
        "sm90_fp8_gemm_1d1d_impl" in name
        and names.intersection({DIRECT_WRITE, LEGACY_DW})
    ):
        return "direct_dw" if DIRECT_WRITE in names else "legacy_dw", None, expert_owner
    if "GFSDP::fp8_expert_refresh" in names:
        return "expert_fp8_cache_refresh", None, expert_owner
    if "torchtitan::fp8_blockwise_quantize" in names and "quantization" in name:
        return "expert_weight_quant", None, expert_owner
    expert_gather = any(
        prefix == "GFSDP::all_gather" and owner == expert_owner
        for prefix, owner in gf_owners
    )
    if expert_gather and event.category == "gpu_memcpy":
        # Size-one NCCL may materialize the gather through a local copy.
        meta = process_group_metadata(event, scopes, inventory)
        return "expert_gather_local_copy", meta, expert_owner
    if (
        expert_gather
        and "bfloat16" in name
        and ("copy" in name or "aten::_to_copy" in names)
    ):
        return "expert_bf16_gather_prep", None, expert_owner
    if expert_owner and any(
        prefix == "FSDP::post_backward_accumulate" for prefix, _ in gf_owners
    ):
        if "aten::_foreach_copy_" in names:
            return "expert_grad_pack", None, expert_owner
        if "aten::_foreach_add_" in names:
            return "expert_grad_accumulate_before_reduce", None, expert_owner
    if expert_owner and any(
        prefix == "GFSDP::post_backward_reduce" for prefix, _ in gf_owners
    ):
        if "cudafunctor_add<float>" in name:
            return "expert_fp32_grad_accumulate", None, expert_owner
        return "expert_post_reduce_other", None, expert_owner
    if "torchtitan::transpose_prepared_fp8_weight" in names:
        return "prepared_weight_transpose", None, expert_owner
    if "cudacherry::row2col_dq" in names:
        return "wgrad_row2col", None, expert_owner
    if "sm90_fp8_gemm_1d2d_impl" in name and "(deep_gemm::gemmtype)1" in name:
        return "grouped_fp8_gemm", None, expert_owner
    if event.category in {"gpu_memcpy", "gpu_memset"}:
        return event.category, None, expert_owner
    if "wait" in name or "spin" in name:
        return "unclassified_device_wait", None, expert_owner
    return "other_compute", None, expert_owner


def timing(family_rows, main_compute):
    events = [row[0] for row in family_rows]
    intervals = union(events)
    duration = elapsed(intervals)
    overlap = intersection(intervals, main_compute)
    samples = []
    for name, group in by_key(family_rows, lambda row: row[0].name).items():
        samples.append((len(group), name[:200]))
    result = {
        "count": len(events),
        "kernel_sum_ms": ms(sum(event.end - event.start for event in events)),
        "busy_union_ms": ms(duration),
        "overlap_main_compute_ms": ms(overlap),
        "outside_main_compute_ms": ms(duration - overlap),
        "streams": sorted({event.stream for event in events}),
        "top_names": [name for _, name in sorted(samples, reverse=True)[:2]],
    }
    metas = [row[2] for row in family_rows if row[2] is not None]
    if metas:
        sizes = [
            meta
            for meta in metas
            if (
                not meta["conflicts"]
                and meta["dtype"] in WIDTHS
                and type(meta["input_elements"]) is int
                and type(meta["output_elements"]) is int
            )
        ]
        result["logical_payload_metadata"] = {
            "with_input_and_output_bytes": len(sizes),
            "input_bytes_sum": sum(
                meta["input_elements"] * WIDTHS[meta["dtype"]] for meta in sizes
            ),
            "output_bytes_sum": sum(
                meta["output_elements"] * WIDTHS[meta["dtype"]] for meta in sizes
            ),
        }
    return result


def analyze_rank(document, rank, ep):
    info = document.get("distributedInfo", {})
    if info.get("rank") != rank or info.get("world_size") != WORLD_SIZE:
        raise ValueError(
            f"Trace metadata does not identify rank {rank} in a world of eight"
        )
    inventory = {
        str(group["pg_name"]): (
            group.get("pg_desc"),
            normalize_ranks(group.get("ranks")),
        )
        for group in info.get("pg_config", [])
    }
    members = RANKS if ep == 4 else (rank,)
    recorded_groups = [
        ranks
        for desc, ranks in inventory.values()
        if desc == "mesh_efsdp" and ranks and rank in ranks
    ]
    if recorded_groups and set(recorded_groups) != {members}:
        raise ValueError(
            f"Unexpected eFSDP process group for EP{ep} rank {rank}: {recorded_groups}"
        )
    raw = document["traceEvents"]
    sync_categories = {"cuda_sync", "cuda_event"}
    events = defaultdict(list)
    for row in raw:
        category = row.get("cat")
        if category in CPU | API | GPU | sync_categories | {"gpu_user_annotation"} and (
            row.get("ph") == "X" or (category == "cuda_event" and row.get("ph") == "i")
        ):
            events[category].append(Event.from_record(row))
    scopes = events["user_annotation"] + events["cpu_op"]
    apis = events["cuda_runtime"] + events["cuda_driver"]
    gpu = sorted(
        (event for category in GPU for event in events[category]),
        key=lambda event: event.start,
    )
    marker = f"task.profile_update.{UPDATE}.microbatch.rank{rank}"
    marks = sorted(
        (event for event in events["user_annotation"] if event.name == marker),
        key=lambda event: event.start,
    )
    projections = sum(event.name == marker for event in events["gpu_user_annotation"])
    if (
        len(marks) != 3
        or projections < 3
        or any(a.end > b.start for a, b in zip(marks, marks[1:]))
    ):
        raise ValueError(
            f"Rank {rank} does not have three complete disjoint microbatch scopes with GPU projections"
        )
    if (
        len({mark.thread for mark in marks}) != 1
        or len({event.pid for event in gpu}) != 1
    ):
        raise ValueError(
            f"Rank {rank} must have one training thread and one observed CUDA device"
        )

    launches, ancestry = Launches(raw, apis, gpu), host_ancestry(apis, scopes)
    work, linked, skipped, coverage, collective_rows = (
        defaultdict(list),
        {},
        Counter(),
        Counter(),
        [],
    )
    for event in gpu:
        api, match = launches.match(event)
        coverage[match] += 1
        mb = microbatch_at(api.start, marks) if api is not None else None
        if mb is None:
            if api is None:
                skipped[match] += 1
            continue
        linked[event] = mb
        family, meta, owner = classify(event, ancestry[api], inventory, members, ep)
        work[mb].append(
            (event, family, meta, owner, api, annotated_phase(ancestry[api]))
        )
        if meta is not None and event.category == "kernel":
            collective_rows.append(
                {"family": family, "microbatch": mb, "owner": owner, **meta}
            )

    compute_by_stream = by_key(
        [row[0] for rows in work.values() for row in rows if row[1] in COMPUTE],
        lambda event: event.stream,
    )
    if len(work) != 3 or not compute_by_stream:
        raise ValueError(
            f"Rank {rank} lacks matched device work for all capture windows"
        )
    main_stream = max(
        compute_by_stream, key=lambda stream: elapsed(union(compute_by_stream[stream]))
    )
    records = by_key(
        [
            api
            for api in apis
            if "EventRecord" in api.name and api.correlation is not None
        ],
        lambda api: api.correlation,
    )
    waits = defaultdict(list)
    for _, writer, wait in wait_rows(
        apis, events["cuda_sync"] + events["cuda_event"], records, ancestry, marks
    ):
        expert_scopes = [scope for scope in wait["scopes"] if "[efsdp]" in scope]
        tag = (
            "direct_dw_writer"
            if writer
            else scope_fqn(expert_scopes[0])[0]
            if expert_scopes
            else "other_gfsdp"
        )
        waits[wait["microbatch"]].append((tag, wait))

    host_sync = [api for api in apis if HOST_BLOCK in api.name.lower()]
    batches = [
        analyze_batch(
            mb, mark, work[mb], waits[mb], host_sync, gpu, linked, main_stream, scopes
        )
        for mb, mark in enumerate(marks, 1)
    ]
    prefix_start, prefix_end = (
        marks[0].start,
        max(row[0].end for rows in work.values() for row in rows),
    )
    return {
        "rank": rank,
        "expert_group_expected": members,
        "expert_group_observed_in_pg_inventory": bool(recorded_groups),
        "main_compute_stream": main_stream,
        "launch_coverage_full_trace": dict(coverage),
        "unlinked_device_events_full_trace": dict(skipped),
        "cuda_sync_rows": len(events["cuda_sync"]),
        "cuda_event_rows": len(events["cuda_event"]),
        "gpu_microbatch_projection_count": projections,
        "first_three_f_b_cpu_entry_to_linked_device_completion_ms": ms(
            prefix_end - prefix_start
        ),
        "microbatches": batches,
        "collectives": collective_rows,
    }


def analyze_batch(mb, mark, work, waits, host_sync, gpu, linked, main_stream, scopes):
    events = [row[0] for row in work]
    start, end = events[0].start, max(event.end for event in events)
    main_rows = [
        row for row in work if row[1] in COMPUTE and row[0].stream == main_stream
    ]
    compute = union(row[0] for row in main_rows)
    all_visible = [event for event in gpu if event.start < end and event.end > start]
    visible = merged(
        (max(start, event.start), min(end, event.end)) for event in all_visible
    )
    idle = [(a[1], b[0]) for a, b in zip(visible, visible[1:])]
    sync = merged(
        (max(mark.start, api.start), min(mark.end, api.end)) for api in host_sync
    )
    family_groups = by_key(work, lambda row: row[1])
    family_intervals = {
        family: union(row[0] for row in rows) for family, rows in family_groups.items()
    }
    owners = Counter(f"{row[1]}: {row[3]}" for row in work if row[3] is not None)
    phase_groups = Counter(
        f"{row[1]}: {row[5]}" for row in work if row[1].startswith("expert_")
    )
    wait_evidence = {}
    for tag, rows in by_key(waits, lambda row: row[0]).items():
        wait_evidence[tag] = {
            "count": len(rows),
            "matched_record_identity": sum(
                row[1]["record_api"] is not None for row in rows
            ),
            "matched_producer_timestamp": sum(
                row[1]["producer_device_timestamp_ns"] is not None for row in rows
            ),
        }
    linked_wait_examples = [
        wait for _, wait in waits if wait["record_api"] is not None
    ][:4]
    compact_calls = [
        scope_fqn(scope.name)[1]
        for scope in sorted(scopes, key=lambda scope: scope.start)
        if scope_fqn(scope.name)[0] == OWNER_SCOPE
        and mark.start <= scope.start < mark.end
    ]
    return {
        "microbatch": mb,
        "cpu_f_b_scope_ms": ms(mark.end - mark.start),
        "cpu_entry_to_linked_device_completion_ms": ms(end - mark.start),
        "linked_device_span_ms": ms(end - start),
        "all_visible_device_busy_union_ms": ms(elapsed(visible)),
        "internal_device_idle_ms": ms(elapsed(idle)),
        "largest_internal_device_gap_ms": ms(
            max((end - start for start, end in idle), default=0)
        ),
        "main_compute_union_ms": ms(elapsed(compute)),
        "device_events_from_another_microbatch_in_span": sum(
            linked.get(event) not in (None, mb) for event in all_visible
        ),
        "device_events_without_microbatch_attribution_in_span": sum(
            event not in linked for event in all_visible
        ),
        "host_launch_union_ms": ms(elapsed(union(row[4] for row in work))),
        "host_sync_api_union_ms_inside_cpu_scope": ms(elapsed(sync)),
        "host_sync_api_overlap_device_idle_ms": ms(intersection(sync, idle)),
        "families": {
            family: timing(rows, compute)
            for family, rows in sorted(family_groups.items())
        },
        "streams": {
            stream: timing(rows, compute)
            for stream, rows in sorted(by_key(work, lambda row: row[0].stream).items())
        },
        "expert_operations_by_owner": dict(owners),
        "expert_operations_by_host_phase": dict(phase_groups),
        "compact_owner_calls_in_host_order": compact_calls,
        "wait_evidence": wait_evidence,
        "matched_wait_examples": linked_wait_examples,
        "largest_main_compute_gaps": main_gaps(compute, main_rows, family_intervals),
    }


def main_gaps(compute, main_rows, family_intervals):
    """Report coincident activity and the next launch, leaving causation open."""
    candidates = [(left[1], right[0]) for left, right in zip(compute, compute[1:])]
    following = by_key(main_rows, lambda row: row[0].start)
    result = []
    for start, end in sorted(
        candidates, key=lambda span: span[1] - span[0], reverse=True
    )[:5]:
        overlap = {
            family: ms(intersection([(start, end)], intervals))
            for family, intervals in family_intervals.items()
        }
        result.append(
            {
                "start_ns": start,
                "duration_ms": ms(end - start),
                "coincident_device_family_union_ms": {
                    family: value for family, value in overlap.items() if value
                },
                "next_compute": [
                    {
                        "kernel": row[0].name[:200],
                        "launch_api": row[4].name,
                        "launch_start_ns": row[4].start,
                        "launch_end_ns": row[4].end,
                    }
                    for row in following[end]
                ],
            }
        )
    return result


def pair_ep4(left, right):
    def index(result):
        eligible = [
            row for row in result["collectives"] if row["family"].startswith("expert_")
        ]
        matched = [
            row
            for row in eligible
            if not row["conflicts"]
            and row["members"] == RANKS
            and type(row["seq"]) is int
            and row["pg"] is not None
            and row["dtype"] is not None
            and type(row["input_elements"]) is int
            and type(row["output_elements"]) is int
        ]
        return by_key(matched, lambda row: (row["pg"], row["seq"])), len(
            eligible
        ) - len(matched)

    a, a_missing = index(left)
    b, b_missing = index(right)
    paired, unresolved = Counter(), 0
    fields = ("family", "microbatch", "dtype", "input_elements", "output_elements")
    for identity in a.keys() | b.keys():
        first, second = a.get(identity, []), b.get(identity, [])
        if len(first) == len(second) == 1 and all(
            first[0][field] == second[0][field] for field in fields
        ):
            paired[first[0]["family"]] += 1
        else:
            unresolved += 1
    return {
        "pairs_by_family": dict(paired),
        "unresolved_pg_sequences": unresolved,
        "rows_without_pairable_metadata": [a_missing, b_missing],
    }


def read_attempt(attempt, ep):
    root = attempt.resolve(strict=True)
    summary = json.loads((root / "summary.json").read_text())
    capture = summary.get("profile_capture", {})
    receipts = capture.get("traces_by_rank", {})
    calls = capture.get("call_counts_by_rank", {})
    if (
        (
            summary.get("status"),
            summary.get("mode"),
            summary.get("training_steps"),
            capture.get("warmup_updates"),
            capture.get("target_update"),
            capture.get("profile_ranks"),
        )
        != ("complete", "profile_update", UPDATE, 10, UPDATE, list(RANKS))
        or set(receipts) != {str(rank) for rank in RANKS}
        or set(calls) != {str(rank) for rank in range(WORLD_SIZE)}
    ):
        raise ValueError(
            f"{root}: expected a complete 10+1 native profile on ranks {RANKS}"
        )
    row_by_rank = {}
    for rank in range(WORLD_SIZE):
        call = calls[str(rank)]
        if (
            call["microbatch_calls"],
            call["optimizer_step_calls"],
            call["train_step_calls"],
            call["native_profiler_build_calls"],
            call["native_profiler_created"],
        ) != (32, 1, 1, 1, rank in RANKS):
            raise ValueError(f"{root}: incorrect call counts for rank {rank}")
        activation = json.loads(
            (root / "measurement" / f"expert-gfsdp-rank-{rank:02d}.json").read_text()
        )
        shard_ranks = [rank % 4, rank % 4 + 4] if ep == 4 else [rank]
        if (
            activation.get("ep") != ep
            or activation.get("shard_ranks") != shard_ranks
            or len(activation.get("groups", [])) != 7
        ):
            raise ValueError(
                f"{root}: invalid expert activation or shard group on rank {rank}"
            )
        row_by_rank[rank] = activation
    traces = []
    for rank in RANKS:
        receipt = receipts[str(rank)]
        expected = Path(
            f"native/profiling/traces/iteration_{UPDATE}/rank{rank}_trace.json"
        )
        path = (root / receipt["path"]).resolve(strict=True)
        if (
            Path(receipt["path"]) != expected
            or not path.is_relative_to(root)
            or path.stat().st_size != receipt["size_bytes"]
        ):
            raise ValueError(
                f"{root}: rank {rank} trace path or size differs from its receipt"
            )
        with path.open("rb") as source:
            if hashlib.file_digest(source, "sha256").hexdigest() != receipt["sha256"]:
                raise ValueError(
                    f"{root}: rank {rank} trace hash differs from its receipt"
                )
            source.seek(0)
            traces.append(
                analyze_rank(json.load(source, parse_float=Decimal), rank, ep)
            )
        traces[-1]["trace_receipt"] = receipt
    result = {
        "run_id": summary.get("run_id"),
        "expert_parallel_degree": ep,
        "group_activation_by_rank": row_by_rank,
        "rank_traces": traces,
    }
    if ep == 4:
        result["expert_collective_pairing"] = pair_ep4(*traces)
    for trace in traces:
        trace.pop("collectives")
    return result


def compare(first, second):
    rows = []
    for a, b in zip(first["rank_traces"], second["rank_traces"], strict=True):
        if a["rank"] != b["rank"]:
            raise ValueError("Rank order mismatch")
        microbatches = []
        for c, d in zip(a["microbatches"], b["microbatches"], strict=True):
            families = {}
            for family in sorted(c["families"].keys() | d["families"].keys()):
                lhs, rhs = c["families"].get(family, {}), d["families"].get(family, {})
                families[family] = {
                    metric: round(rhs.get(metric, 0) - lhs.get(metric, 0), 6)
                    for metric in ("count", "busy_union_ms", "outside_main_compute_ms")
                }
            microbatches.append(
                {
                    "microbatch": c["microbatch"],
                    "device_span_ms_delta": round(
                        d["linked_device_span_ms"] - c["linked_device_span_ms"], 6
                    ),
                    "families_delta": families,
                }
            )
        field = "first_three_f_b_cpu_entry_to_linked_device_completion_ms"
        rows.append(
            {
                "rank": a["rank"],
                "first_three_f_b_ms_delta": round(b[field] - a[field], 6),
                "microbatches": microbatches,
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "attempts",
        nargs="+",
        metavar="ARM=PATH",
        help="EP8-A, EP8-B, EP4-A or EP4-B and the native attempt directory",
    )
    args = parser.parse_args()
    runs = {}
    for spec in args.attempts:
        arm, sep, path = spec.partition("=")
        if not sep or not path or arm not in ARMS or arm in runs:
            parser.error(
                f"expected a unique ARM=PATH for one of {', '.join(ARMS)}; got {spec!r}"
            )
        runs[arm] = read_attempt(Path(path), ARMS[arm])
    pairs = (
        ("EP8-A", "EP8-B"),
        ("EP4-A", "EP4-B"),
        ("EP8-A", "EP4-A"),
        ("EP8-B", "EP4-B"),
    )
    result = {
        "update": UPDATE,
        "notes": [
            "Rank-local times. Peer collective identities do not align CUDA clocks.",
            "Microbatch ownership follows CUDA host launch correlation; device tails may extend beyond CPU scopes.",
            "Outside-main-compute is an interval measure. It does not establish an exposed dependency or incremental improvement.",
            "Other compute is residual kernel work and may contain unclassified waits. Family and stream unions overlap.",
            "Collective payload sums use reported logical bytes and do not estimate bytes on the wire.",
            "A missing event record or device timestamp leaves that dependency or its timing unresolved.",
            "The first three microbatches cover cold-generation and reuse behavior. The remaining microbatches and optimizer device time are outside the capture.",
        ],
        "runs": runs,
        "second_minus_first": {
            f"{b} minus {a}": compare(runs[a], runs[b])
            for a, b in pairs
            if a in runs and b in runs
        },
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
