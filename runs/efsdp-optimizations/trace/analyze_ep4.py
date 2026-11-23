#!/usr/bin/env python3
"""Extract expert collectives and wait evidence from the bounded EP4 capture."""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path


RANKS = (0, 4)
EXPECTED = {
    ("all_gather", "uint8", 301_989_888, 603_979_776): "fp8_data_ag",
    ("all_gather", "float32", 18_432, 36_864): "fp32_scales_ag",
    ("reduce_scatter", "float32", 603_979_776, 301_989_888): "fp32_grad_rs",
}
CPU = {"cpu_op", "user_annotation"}
API = {"cuda_runtime", "cuda_driver"}
GPU = {"kernel", "gpu_memcpy", "gpu_memset"}
FQN_SCOPE = ("GFSDP::all_gather", "GFSDP::post_backward_reduce", "GFSDP::pre_backward")
DIRECT_WRITE = "torchtitan::write_direct_expert_dw"
DIRECT_ZERO = "torchtitan::zero_direct_expert_dw"
DIRECT_WRITERS = {DIRECT_WRITE, DIRECT_ZERO}


@dataclass(eq=False)
class Event:
    name: str
    category: str
    start: int
    end: int
    pid: int
    tid: int
    args: dict

    @classmethod
    def from_record(cls, row):
        start = ns(row["ts"])
        return cls(
            row["name"],
            row["cat"],
            start,
            start + ns(row.get("dur", 0)),
            row["pid"],
            row["tid"],
            row.get("args", {}),
        )

    @property
    def thread(self):
        return self.pid, self.tid

    @property
    def endpoint(self):
        return *self.thread, self.start

    @property
    def stream(self):
        return self.args.get("stream", self.tid) if self.category in GPU else None

    @property
    def correlation(self):
        return self.args.get("correlation")


def ns(microseconds):
    return round(microseconds * 1000)


def is_submission(device, api):
    name = api.name.lower()
    if api.start > device.start:
        return False
    if device.category == "kernel":
        return "launch" in name and ("kernel" in name or "graph" in name)
    if device.category == "gpu_memcpy":
        return "memcpy" in name
    return "memset" in name


class Launches:
    def __init__(self, rows, apis, gpu):
        self.apis = defaultdict(list)
        self.correlations = defaultdict(list)
        self.gpu = defaultdict(list)
        self.starts = defaultdict(list)
        self.ends = defaultdict(list)
        self.unambiguous_device = len({event.pid for event in gpu}) == 1
        self.unambiguous_process = len({event.pid for event in apis}) == 1
        for api in apis:
            self.apis[api.endpoint].append(api)
            if api.correlation is not None:
                self.correlations[api.correlation].append(api)
        for device in gpu:
            self.gpu[device.endpoint].append(device)
        for row in rows:
            if row.get("cat") == "ac2g" and row.get("ph") in {"s", "f"}:
                endpoint = row["pid"], row["tid"], ns(row["ts"])
                if row["ph"] == "s":
                    self.starts[row["id"]].append(endpoint)
                else:
                    self.ends[endpoint].append(row["id"])

    def match(self, device):
        flows = self.ends[device.endpoint]
        candidates = {
            api
            for flow in flows
            for endpoint in self.starts[flow]
            for api in self.apis[endpoint]
            if is_submission(device, api)
        }
        collision = any(
            peer is not device
            and (
                device.correlation is None
                or peer.correlation in (None, device.correlation)
            )
            for peer in self.gpu[device.endpoint]
        )
        if collision:
            return None, "ambiguous_device_endpoint"
        if flows:
            # A shared clock endpoint can have multiple flow IDs. Use the raw
            # correlation to reject an API for another row at that endpoint.
            exact = {api for api in candidates if api.correlation == device.correlation}
            if device.correlation is not None and exact:
                candidates = exact
            if len(candidates) == 1:
                api = candidates.pop()
                if (
                    device.correlation is None
                    or api.correlation is None
                    or api.correlation == device.correlation
                ):
                    return api, "ac2g"
            return None, "ambiguous_or_unresolved_ac2g"
        if (
            not self.unambiguous_device
            or not self.unambiguous_process
            or device.correlation is None
        ):
            return None, "missing_launch"
        candidates = [
            api
            for api in self.correlations[device.correlation]
            if is_submission(device, api)
            and device.start - api.start <= 1_000_000_000
            and (
                not device.args.get("External id")
                or not api.args.get("External id")
                or device.args["External id"] == api.args["External id"]
            )
        ]
        if len(candidates) == 1:
            return candidates[0], "validated_correlation"
        return None, "ambiguous_correlation" if candidates else "missing_launch"


def host_ancestry(apis, scopes):
    calls, ranges = defaultdict(list), defaultdict(list)
    for api in apis:
        calls[api.thread].append(api)
    for scope in scopes:
        ranges[scope.thread].append(scope)
    result = {}
    for thread, thread_calls in calls.items():
        pending = iter(sorted(ranges[thread], key=lambda event: event.start))
        current = next(pending, None)
        active = []
        for call in sorted(thread_calls, key=lambda event: event.start):
            active = [scope for scope in active if scope.end >= call.start]
            while current is not None and current.start <= call.start:
                if current.end >= call.start:
                    active.append(current)
                current = next(pending, None)
            result[call] = sorted(
                (scope for scope in active if scope.end >= call.end),
                key=lambda scope: scope.end - scope.start,
            )
    return result


def normalize_ranks(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if isinstance(value, (tuple, list)) and all(type(rank) is int for rank in value):
        return tuple(value)
    return None


def scope_fqn(name):
    prefix, sep, fqn = name.partition(" (")
    if sep and name.endswith(")"):
        return prefix, fqn[:-1]
    return prefix, None


def event_row(event):
    return {
        "name": event.name,
        "start_ns": event.start,
        "end_ns": event.end,
        "stream": event.stream,
        "thread": event.thread,
        "correlation": event.correlation,
    }


def microbatch_at(time, marks):
    return next((i for i, m in enumerate(marks, 1) if m.start <= time < m.end), None)


def collective_kind(event):
    name = event.name.lower().replace("_", "")
    if "hybridep" in name or "deepep" in name:
        return "expert_token_exchange"
    if "nccl" not in name:
        return None
    name += str(event.args.get("Collective name", "")).lower().replace("_", "")
    for token, kind in (
        ("allgather", "all_gather"),
        ("reducescatter", "reduce_scatter"),
        ("allreduce", "all_reduce"),
    ):
        if token in name:
            return kind
    return "other_nccl"


def describe_collective(event, scopes, kind, expert_pgs):
    host_record = next(
        (scope for scope in scopes if scope.name == "record_param_comms"), None
    )
    fields = (
        "Process Group Name",
        "Process Group Ranks",
        "Group size",
        "Seq",
        "dtype",
        "In msg nelems",
        "Out msg nelems",
    )
    args = dict(event.args)
    conflicts = []
    for field in fields:
        if host_record is not None and field in host_record.args:
            peer = host_record.args[field]
            if field not in args:
                args[field] = peer
            elif field == "Process Group Ranks" and normalize_ranks(
                args[field]
            ) == normalize_ranks(peer):
                continue
            elif args[field] != peer:
                conflicts.append(field)
    pg = args.get("Process Group Name")
    pg = str(pg) if type(pg) in {int, str} else None
    ranks = normalize_ranks(args.get("Process Group Ranks"))
    seq = args.get("Seq")
    dtype = str(args.get("dtype", "")).lower().removeprefix("torch.")
    dtype = {"byte": "uint8", "float": "float32"}.get(dtype, dtype)
    ins, outs = args.get("In msg nelems"), args.get("Out msg nelems")
    signature = (kind, dtype, ins, outs)
    typed = EXPECTED.get(signature) if type(ins) is int and type(outs) is int else None
    owner = next(
        (
            fqn
            for scope in scopes
            for prefix, fqn in [scope_fqn(scope.name)]
            if prefix in FQN_SCOPE and fqn is not None
        ),
        None,
    )
    issues = [f"conflicting {field}" for field in conflicts]
    expert_candidate = (
        ranks == RANKS
        or pg in expert_pgs
        or typed is not None
        or (owner is not None and owner.endswith("[efsdp]"))
    )
    if expert_candidate:
        if (
            ranks != RANKS
            or args.get("Group size") != len(RANKS)
            or pg not in expert_pgs
        ):
            issues.append("missing or unexpected expert group metadata")
        if type(seq) is not int:
            issues.append("missing collective sequence")
        if typed is None:
            issues.append("unexpected type or element counts")
    return {
        "kind": kind,
        "expert_kind": typed if expert_candidate else None,
        "expert_candidate": expert_candidate,
        "owner": owner,
        "process_group": pg,
        "ranks": ranks,
        "group_size": args.get("Group size"),
        "sequence": seq,
        "dtype": dtype or None,
        "input_elements": ins,
        "output_elements": outs,
        "metadata_issues": issues,
    }


def operation_kind(event, scopes):
    names = {scope.name for scope in scopes}
    if any(name == "GFSDP::fp8_expert_refresh" for name in names):
        return "fp8_cache_refresh"
    if "sm90_k_grouped_fp8_gemm_rounded_f32" in event.name:
        return "direct_dw"
    if any("zero_direct_expert_dw" in name for name in names):
        return "direct_dw_empty_zero"
    for token, kind in (
        ("::dgrad_m_grouped_fp8_gemm", "dgrad"),
        ("::row2col_requantization", "wgrad_operands"),
        ("::transpose_weights_fn", "dgrad_weight_transpose"),
    ):
        if any(token in name for name in names):
            return kind
    if any(scope_fqn(name)[0] == "GFSDP::post_backward_reduce" for name in names):
        return "reduce_scope_device_work"
    return None


def compiled_dw_preparation(event, scopes, pending_dgrad):
    # SAC recompute shares the grouped GEMM symbol. A candidate must follow a
    # prepared-weight transpose in the same compiled backward call and stream.
    names = {scope.name for scope in scopes}
    backward = next(
        (scope for scope in scopes if scope.name == "CompiledFunctionBackward"), None
    )
    key = backward, event.stream
    if "torchtitan::transpose_prepared_fp8_weight" in names:
        if backward is not None:
            pending_dgrad.add(key)
        return "prepared_weight_transpose"
    if "cudacherry::row2col_dq" in names:
        return "row2col"
    if (
        key in pending_dgrad
        and "sm90_fp8_gemm_1d2d_impl" in event.name
        and "(deep_gemm::GemmType)1" in event.name
    ):
        pending_dgrad.remove(key)
        return "grouped_deepgemm_dgrad_candidate"
    return None


def annotated_phase(scopes):
    for scope in scopes:
        name = scope.name
        if name.startswith(
            (
                "GFSDP::backward_prefetch",
                "GFSDP::pre_backward",
                "GFSDP::post_backward",
                "grouped_gemm::bwd",
            )
        ):
            return "backward"
        if name.startswith(
            ("GFSDP::forward_prefetch", "GFSDP::pre_forward", "GFSDP::post_forward")
        ):
            return "forward_or_recompute"
    return "unknown"


def wait_rows(apis, sync, records, ancestry, marks):
    sync_by_wait, event_times = defaultdict(list), defaultdict(list)
    for event in sync:
        if event.category == "cuda_sync" and event.name == "Stream Wait Event":
            sync_by_wait[event.correlation].append(event)
        elif event.category == "cuda_event" and event.correlation is not None:
            event_times[event.correlation].append(event.start)
    for wait in apis:
        if "StreamWaitEvent" not in wait.name:
            continue
        microbatch = microbatch_at(wait.start, marks)
        scopes = ancestry[wait]
        gfsdp = any(scope_fqn(scope.name)[0] in FQN_SCOPE for scope in scopes)
        writer = next((scope for scope in scopes if scope.name in DIRECT_WRITERS), None)
        if not microbatch or (not gfsdp and writer is None):
            continue
        raw_matches = sync_by_wait.get(wait.correlation, ()) if wait.correlation else ()
        matched = raw_matches[0] if len(raw_matches) == 1 else None
        sync_args = matched.args if matched is not None else {}
        record_id = sync_args.get("wait_on_cuda_event_record_corr_id")
        candidates = (
            records.get(record_id, ())
            if isinstance(record_id, int) and record_id > 0
            else ()
        )
        record = candidates[0] if len(candidates) == 1 else None
        times = event_times.get(record_id, ())
        tags = [scope.name for scope in scopes if scope.name.startswith("GFSDP::")]
        producer_tags = (
            [
                scope.name
                for scope in ancestry[record]
                if scope.name.startswith("GFSDP::")
            ]
            if record
            else []
        )
        yield (
            gfsdp,
            writer,
            {
                **event_row(wait),
                "microbatch": microbatch,
                "scopes": tags,
                "sync_activity_matches": len(raw_matches),
                "wait_on_stream": sync_args.get("wait_on_stream"),
                "waiting_stream": sync_args.get("stream", matched.tid)
                if matched
                else None,
                "record_correlation": record_id,
                "record_api": event_row(record) if record else None,
                "record_scopes": producer_tags,
                "producer_device_timestamp_ns": times[0] if len(times) == 1 else None,
                "dependency_evidence": "matched_cuda_record"
                if record
                else "unknown_event_identity",
            },
        )


def analyze_rank(document, rank, update):
    rows = document["traceEvents"]
    info = document.get("distributedInfo", {})
    if info.get("rank") != rank or info.get("world_size") != 8:
        raise ValueError(
            f"Trace metadata must identify rank {rank} in a world of eight"
        )
    expert_pgs = {
        str(pg["pg_name"])
        for pg in info.get("pg_config", [])
        if normalize_ranks(pg.get("ranks")) == RANKS
        and pg.get("pg_desc") == "mesh_efsdp"
    }
    events = defaultdict(list)
    for row in rows:
        category = row.get("cat")
        if category in CPU | API | GPU | {
            "cuda_sync",
            "cuda_event",
            "gpu_user_annotation",
        } and (
            row.get("ph") == "X" or (category == "cuda_event" and row.get("ph") == "i")
        ):
            events[category].append(Event.from_record(row))
    scopes = events["user_annotation"] + events["cpu_op"]
    apis = events["cuda_runtime"] + events["cuda_driver"]
    gpu = events["kernel"] + events["gpu_memcpy"] + events["gpu_memset"]
    marker = f"task.profile_update.{update}.microbatch.rank{rank}"
    marks = sorted(
        (event for event in events["user_annotation"] if event.name == marker),
        key=lambda event: event.start,
    )
    if len(marks) != 3 or any(
        left.end > right.start for left, right in zip(marks, marks[1:])
    ):
        raise ValueError(
            f"Expected exactly three disjoint CPU microbatch scopes for {marker}"
        )
    if len({m.thread for m in marks}) != 1 or len({e.pid for e in gpu}) != 1:
        raise ValueError(
            f"Trace rank {rank} must have one training thread and one traced CUDA device"
        )

    launches = Launches(rows, apis, gpu)
    ancestry = host_ancestry(apis, scopes)
    coverage, critical_unknown = Counter(), Counter()
    collectives, work, compiled_prep = [], [], []
    writer_calls = {
        scope: {
            "operator": event_row(scope),
            "microbatch": microbatch,
            "device_writes": [],
            "waits": [],
        }
        for scope in events["cpu_op"]
        if scope.name in DIRECT_WRITERS
        and (microbatch := microbatch_at(scope.start, marks)) is not None
    }
    previous_on_stream = {}
    pending_dgrad = set()
    for event in sorted(gpu, key=lambda event: event.start):
        previous = previous_on_stream.get(event.stream)
        previous_on_stream[event.stream] = event
        api, match = launches.match(event)
        coverage[match] += 1
        ancestor = ancestry[api] if api is not None else ()
        microbatch = microbatch_at(api.start, marks) if api is not None else None
        kind = collective_kind(event) if event.category == "kernel" else None
        op = operation_kind(event, ancestor)
        writer = next((scope for scope in ancestor if scope in writer_calls), None)
        is_write = writer is not None and (
            event.category == "gpu_memset"
            or (
                event.category == "kernel"
                and (writer.name == DIRECT_ZERO or op == "direct_dw")
            )
        )
        compiled = (
            compiled_dw_preparation(event, ancestor, pending_dgrad)
            if microbatch is not None
            else None
        )
        if kind is None and op is None and not is_write and compiled is None:
            continue
        row = {
            **event_row(event),
            "launch": event_row(api) if api is not None else None,
            "launch_match": match,
            "microbatch": microbatch,
            "annotated_phase": annotated_phase(ancestor),
            "launch_scopes": [
                scope.name
                for scope in ancestor
                if scope.name.startswith(("GFSDP::", "grouped_gemm::"))
            ],
            "previous_on_stream": event_row(previous) if previous else None,
        }
        if kind is not None:
            row.update(describe_collective(event, ancestor, kind, expert_pgs))
            collectives.append(row)
            if row["expert_candidate"]:
                critical_unknown.update(row["metadata_issues"])
                if microbatch is None:
                    critical_unknown[
                        "expert collective lacks microbatch launch attribution"
                    ] += 1
                if row["owner"] is None:
                    critical_unknown["expert collective lacks GFSDP owner scope"] += 1
        elif op is not None and microbatch is not None:
            work.append({**row, "kind": op})
        if is_write:
            write_kind = (
                "zero"
                if writer.name == DIRECT_ZERO or event.category == "gpu_memset"
                else "deepgemm"
            )
            writer_calls[writer]["device_writes"].append({**row, "kind": write_kind})
        if compiled is not None:
            compiled_prep.append({**row, "kind": compiled})
    prep = [
        event
        for event in scopes
        if event.name.startswith("GFSDP::pre_backward (")
        and event.name.endswith("[efsdp])")
        and microbatch_at(event.start, marks)
    ]
    observations = backward_windows(prep, work, marks)
    records = defaultdict(list)
    for api in apis:
        if "EventRecord" in api.name and api.correlation is not None:
            records[api.correlation].append(api)
    waits = []
    for gfsdp, writer, wait in wait_rows(
        apis, events["cuda_sync"] + events["cuda_event"], records, ancestry, marks
    ):
        if gfsdp:
            waits.append(wait)
        if writer in writer_calls:
            writer_calls[writer]["waits"].append(wait)
    if not expert_pgs:
        critical_unknown["trace lacks the expected expert group inventory"] += 1
    for wait in waits:
        if wait["record_api"] is None:
            critical_unknown["GFSDP stream wait lacks event-record identity"] += 1
        if wait["producer_device_timestamp_ns"] is None:
            critical_unknown["GFSDP stream wait lacks producer device timestamp"] += 1
    return {
        "rank": rank,
        "host": document.get("host_name"),
        "time_base_ns": document.get("baseTimeNanoseconds"),
        "trace_id": document.get("trace_id"),
        "cupti_version": document.get("cupti_version"),
        "driver_version": document.get("cuda_driver_version"),
        "expert_process_groups": sorted(expert_pgs),
        "complete_microbatch_scopes": [event_row(event) for event in marks],
        "microbatch_gpu_projections": sum(
            event.name == marker for event in events["gpu_user_annotation"]
        ),
        "device_launch_coverage": dict(coverage),
        "unknown_edges": dict(critical_unknown),
        "cuda_sync_activity_count": len(events["cuda_sync"]),
        "cuda_event_activity_count": len(events["cuda_event"]),
        "collectives": collectives,
        "backward_windows": observations,
        "fp8_refresh_work": [row for row in work if row["kind"] == "fp8_cache_refresh"],
        "post_reduce_device_work": [
            row for row in work if row["kind"] == "reduce_scope_device_work"
        ],
        "gfsdp_stream_waits": waits,
        "queued_behind_expert_rs": queue_observations(collectives),
        "direct_dw": direct_dw_windows(
            prep, scopes, writer_calls, compiled_prep, marks
        ),
    }


def direct_dw_windows(prep, scopes, writer_calls, compiled_prep, marks):
    """Assign exact writer calls to an expert claim-to-reduction host bracket."""
    reductions = defaultdict(list)
    for scope in scopes:
        prefix, owner = scope_fqn(scope.name)
        if (
            prefix == "GFSDP::post_backward_reduce"
            and owner
            and owner.endswith("[efsdp]")
        ):
            reductions[scope.thread, owner].append(scope)

    windows, assigned = [], set()
    ordered_claims = sorted(prep, key=lambda scope: scope.start)
    ordered_calls = sorted(writer_calls, key=lambda scope: scope.start)
    for claim, following in zip(
        ordered_claims, [*ordered_claims[1:], None], strict=True
    ):
        _, owner = scope_fqn(claim.name)
        microbatch = microbatch_at(claim.start, marks)
        limit = min(
            marks[microbatch - 1].end, following.start if following else marks[-1].end
        )
        matched = [
            scope
            for scope in reductions[claim.thread, owner]
            if claim.end <= scope.start < limit
        ]
        reduction = matched[0] if len(matched) == 1 else None
        calls = [
            op
            for op in ordered_calls
            if reduction is not None
            and op.thread == claim.thread
            and claim.end <= op.start <= op.end <= reduction.start
        ]
        assigned.update(calls)
        before_write = calls[0].start if calls else claim.end
        windows.append(
            {
                "owner": owner,
                "microbatch": microbatch,
                "pre_backward": event_row(claim),
                "post_backward_reduce": event_row(reduction) if reduction else None,
                "calls": [writer_calls[op] for op in calls],
                "pre_first_writer_device_work": [
                    row
                    for row in compiled_prep
                    if tuple(row["launch"]["thread"]) == claim.thread
                    and claim.end <= row["launch"]["start_ns"] < before_write
                ],
            }
        )
    return {
        "owner_basis": "same-thread host bracket; not a CUPTI release-event match",
        "coverage": {
            "cpu_writer_calls": len(writer_calls),
            "calls_without_matched_device_write": sum(
                not row["device_writes"] for row in writer_calls.values()
            ),
            "owner_windows_with_two_calls": sum(
                len(window["calls"]) == 2 for window in windows
            ),
            "writer_waits_without_record_identity": sum(
                wait["record_api"] is None
                for row in writer_calls.values()
                for wait in row["waits"]
            ),
        },
        "owner_windows": windows,
        "unassigned_calls": [
            writer_calls[op] for op in ordered_calls if op not in assigned
        ],
    }


def backward_windows(prep, work, marks):
    result = []
    by_start = sorted(prep, key=lambda event: event.start)
    expected_work = {
        "dgrad",
        "wgrad_operands",
        "dgrad_weight_transpose",
        "direct_dw",
        "direct_dw_empty_zero",
    }
    for current, following in zip(by_start, [*by_start[1:], None], strict=True):
        microbatch = microbatch_at(current.start, marks)
        end = min(
            following.start if following else marks[-1].end, marks[microbatch - 1].end
        )
        nearby = [
            row
            for row in work
            if row["kind"] in expected_work
            and current.start <= row["launch"]["start_ns"] < end
        ]
        first = {
            kind: next((row for row in nearby if row["kind"] == kind), None)
            for kind in sorted(expected_work)
        }
        result.append(
            {
                "microbatch": microbatch,
                "pre_backward": event_row(current),
                "first_device_operations_in_host_window": first,
                "ownership": "temporal window; operation-to-group dependency not established",
            }
        )
    return result


def queue_observations(collectives):
    expert_rs = defaultdict(list)
    for row in collectives:
        if row["expert_kind"] == "fp32_grad_rs" and not row["metadata_issues"]:
            expert_rs[row["stream"]].append(row)
    output = []
    ends = {
        stream: [row["start_ns"] for row in rows] for stream, rows in expert_rs.items()
    }
    for ag in collectives:
        if (
            ag["expert_kind"] not in {"fp8_data_ag", "fp32_scales_ag"}
            or ag["metadata_issues"]
            or ag["launch"] is None
            or ag["microbatch"] is None
        ):
            continue
        stream = ag["stream"]
        index = bisect.bisect_left(ends.get(stream, []), ag["start_ns"]) - 1
        if index < 0:
            continue
        prior = expert_rs[stream][index]
        if (
            prior["end_ns"] <= ag["launch"]["start_ns"]
            or prior["end_ns"] > ag["start_ns"]
        ):
            continue
        output.append(
            {
                "microbatch": ag["microbatch"],
                "stream": stream,
                "ag_sequence": ag["sequence"],
                "ag_launch_ns": ag["launch"]["start_ns"],
                "ag_start_ns": ag["start_ns"],
                "preceding_rs_sequence": prior["sequence"],
                "preceding_rs_end_ns": prior["end_ns"],
                "evidence": "same-stream RS extends past AG host launch; other readiness unverified",
            }
        )
    return output


def pair_collectives(left, right):
    indexed = [defaultdict(list), defaultdict(list)]
    for index, side in enumerate((left, right)):
        for row in side["collectives"]:
            if row["expert_candidate"] and not row["metadata_issues"]:
                indexed[index][(row["process_group"], row["sequence"])].append(row)
    pairs, unresolved = [], []
    for pg, seq in sorted(indexed[0].keys() | indexed[1].keys()):
        a, b = (table.get((pg, seq), []) for table in indexed)
        if len(a) != 1 or len(b) != 1:
            unresolved.append(
                {
                    "process_group": pg,
                    "sequence": seq,
                    "rank0_count": len(a),
                    "rank4_count": len(b),
                }
            )
            continue
        a, b = a[0], b[0]
        if (
            a["expert_kind"] != b["expert_kind"]
            or (
                a["microbatch"] is not None
                and b["microbatch"] is not None
                and a["microbatch"] != b["microbatch"]
            )
            or (
                a["owner"] is not None
                and b["owner"] is not None
                and a["owner"] != b["owner"]
            )
        ):
            unresolved.append(
                {
                    "process_group": pg,
                    "sequence": seq,
                    "reason": "rank metadata or microbatch disagreement",
                }
            )
            continue
        pairs.append(
            {
                "process_group": pg,
                "sequence": seq,
                "kind": a["expert_kind"],
                "by_rank": {"0": a, "4": b},
            }
        )
    return pairs, unresolved


def read_attempt(attempt):
    attempt = attempt.resolve(strict=True)
    summary = json.loads((attempt / "summary.json").read_text())
    capture = summary.get("profile_capture", {})
    if (
        summary.get("status") != "complete"
        or summary.get("mode") != "profile_update"
        or capture.get("profile_ranks") != list(RANKS)
    ):
        raise ValueError(
            "The native summary must validate a complete bounded EP4 trace for ranks 0 and 4"
        )
    update = capture["target_update"]
    results = []
    for rank in RANKS:
        receipt = capture["traces_by_rank"][str(rank)]
        path = (attempt / receipt["path"]).resolve(strict=True)
        if (
            not path.is_relative_to(attempt)
            or path.stat().st_size != receipt["size_bytes"]
        ):
            raise ValueError(
                f"Rank {rank} native trace path or size differs from its receipt"
            )
        with path.open("rb") as source:
            if hashlib.file_digest(source, "sha256").hexdigest() != receipt["sha256"]:
                raise ValueError(
                    f"Rank {rank} native trace SHA256 differs from its receipt"
                )
        with path.open() as source:
            result = analyze_rank(json.load(source, parse_float=Decimal), rank, update)
        result["native_receipt"] = receipt
        results.append(result)
    pairs, unresolved = pair_collectives(*results)
    return {
        "target_update": update,
        "rank_traces": results,
        "expert_collective_pairs": pairs,
        "native_call_counts_by_rank": capture.get("call_counts_by_rank"),
        "unresolved_pairs": unresolved,
        "clock_scope": "raw per-rank Kineto ns; no cross-GPU clock error bound applied",
        "interpretation": "Timing proximity and runtime wait duration do not establish GPU prerequisites or exposed latency.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "attempt", type=Path, help="native attempt directory containing summary.json"
    )
    args = parser.parse_args()
    print(json.dumps(read_attempt(args.attempt.resolve(strict=True)), indent=2))


if __name__ == "__main__":
    main()
