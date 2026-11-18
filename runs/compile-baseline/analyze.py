#!/usr/bin/env python3
"""Compare two raw rank-local Kineto traces without importing them into Perfetto."""

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal
import gzip
import hashlib
import json
from pathlib import Path
import re


GPU = {"kernel", "gpu_memcpy", "gpu_memset"}
API = {"cuda_runtime", "cuda_driver"}
CPU = {"cpu_op", "user_annotation"}
LAUNCH = re.compile(r"^(?:cuda|cu)(?:Launch(?:Kernel|CooperativeKernel)|GraphLaunch)")
MICROBATCH = re.compile(r"task\.profile_update\.(\d+)\.microbatch\.rank(\d+)")
AUTOGRAD = "autograd::engine::evaluate_function:"
INDUCTOR = re.compile(r"^triton_(?:poi|per|red|for|tem|spl|unk)_fused_")
GEMM = re.compile(r"gemm|sgemm|nvjet|cutlass::kernel2|aten::(?:mm|bmm|matmul|linear)\b")
READBACK_OWNER = {"aten::_local_scalar_dense", "aten::item", "aten::copy_"}
GROUPED_GEMM = re.compile(r"(?:[mk]_)?grouped(?:_fp8)?_gemm|_grouped_mm")
COPY_KERNEL = re.compile(r"direct_copy_kernel|copy_device_to_device|copy_kernel_cuda")
EP_SYNC = re.compile(r"device_sync_kernel|barrier_kernel|barrier_all")
# Match the specific generated names, CUDA symbols or traced operator scopes.
COMPONENTS = [
    ("cross_entropy", r"cross_entropy|crossentropy|stable_ce|nll_loss|nllloss"),
    ("mla_attention", r"scaled_dot_product|sdpa_sm90|cudnn::fusion::|flash::flash_"),
    ("mla_rope", r"apply_rotary_emb|rotary_embedding|rotary_emb|view_as_complex"),
    ("gdn_gate", r"gdngatefunction|gdn_gate|fused_gate_kernel"),
    ("gdn_short_conv", r"causal_conv1d|qkv_conv1d_unpack|qkv_bwd_pack"),
    ("gdn_scan", r"gdn_chunk|fla_vjp|chunkgateddeltarule|chunk_gdr|chunk_local_cumsum"),
    ("norm", r"rms_?norm|layer_?norm|cute_gated_norm|rmsnorm|fused_(?:add_)?norm"),
    ("moe_capacity", r"capacity_mask|capacity_select|conditional_capacity"),
    ("moe_router", r"router_topk|topkrouter|router_z_loss|router_(?:gemm|logits)"),
    ("moe_routing", r"routing_metadata|tokens_per_expert|_build_m_indices"),
    ("moe_grouped_gemm", GROUPED_GEMM.pattern),
    ("embedding", r"aten::embedding|_ownerlookup|embedding_backward"),
    ("output_head", r"lm_head|output_head"),
]
COMPONENTS = [(name, re.compile(pattern)) for name, pattern in COMPONENTS]


@dataclass(frozen=True, eq=False)
class Event:
    index: int
    name: str
    cat: str
    start: int
    dur: int
    pid: int
    tid: int
    args: dict

    @property
    def end(self):
        return self.start + self.dur

    @property
    def thread(self):
        return self.pid, self.tid

    @property
    def endpoint(self):
        return self.pid, self.tid, self.start

    @property
    def corr(self):
        return self.args.get("correlation")

    @property
    def external(self):
        return self.args.get("External id")


def ns(value):
    return round(value * 1000)


def ms(value):
    return round(value / 1_000_000, 6)


def index_by(events, key):
    index = defaultdict(list)
    for event in events:
        index[key(event)].append(event)
    return index


def api_kind(event):
    if LAUNCH.search(event.name):
        return "kernel_launch"
    if "Memcpy" in event.name:
        return "copy_submit"
    if "Memset" in event.name:
        return "memset_submit"
    if "Synchronize" in event.name:
        return "host_synchronize"
    if "WaitEvent" in event.name:
        return "stream_wait_event"
    if "EventQuery" in event.name:
        return "event_query"
    return "other"


def valid_launch(device, host):
    if device.corr is not None and host.corr is not None and device.corr != host.corr:
        return False
    expected = {"kernel": "kernel_launch", "gpu_memcpy": "copy_submit", "gpu_memset": "memset_submit"}
    return host.start <= device.start and (
        api_kind(host) == expected[device.cat] or "GraphLaunch" in host.name
    )


def launches(raw, device, apis, single_process_device):
    sites = index_by(apis, lambda event: event.endpoint)
    device_sites = index_by(device, lambda event: event.endpoint)
    correlations = index_by(apis, lambda event: event.corr)
    starts, stops = defaultdict(list), defaultdict(list)
    for item in raw:
        if item.get("cat") != "ac2g":
            continue
        endpoint = item["pid"], item["tid"], ns(item["ts"])
        if item["ph"] == "s":
            starts[item["id"]].append(endpoint)
        elif item["ph"] == "f":
            stops[endpoint].append(item["id"])

    result, coverage = {}, {}
    for event in device:
        flow_ends = stops[event.endpoint]
        candidates = {
            host
            for flow_id in flow_ends
            for endpoint in starts[flow_id]
            for host in sites[endpoint]
            if valid_launch(event, host)
        }
        # Timestamp collisions stay ambiguous unless the raw correlation distinguishes
        # the device records sharing that endpoint.
        colliding = any(
            peer is not event and (event.corr is None or peer.corr is None or peer.corr == event.corr)
            for peer in device_sites[event.endpoint]
        )
        if len(candidates) == 1 and not colliding:
            result[event] = candidates.pop()
            coverage[event] = "exact_ac2g"
            continue
        if len(candidates) > 1 or colliding:
            coverage[event] = "ambiguous_ac2g"
            continue
        if flow_ends:
            coverage[event] = "unresolved_ac2g"
            continue
        # An isolated trace permits a correlation fallback only with a unique
        # compatible API, a one-second submission bound, and matching metadata.
        if single_process_device and event.corr:
            candidates = [
                host for host in correlations[event.corr]
                if valid_launch(event, host) and event.start - host.start <= 1_000_000_000
                and (not event.external or not host.external or host.external == event.external)
            ]
            if len(candidates) == 1:
                result[event] = candidates[0]
                coverage[event] = "validated_correlation"
                continue
            if len(candidates) > 1:
                coverage[event] = "ambiguous_correlation"
                continue
        coverage[event] = "missing_launch"
    return result, coverage


def ancestors(apis, cpu):
    by_thread = index_by(cpu, lambda event: event.thread)
    scopes = {}
    for thread, calls in index_by(apis, lambda event: event.thread).items():
        ranges = iter(sorted(by_thread[thread], key=lambda event: event.start))
        pending = next(ranges, None)
        active = []
        for call in sorted(calls, key=lambda event: event.start):
            active = [scope for scope in active if scope.end > call.start]
            while pending and pending.start <= call.start:
                if pending.end > call.start:
                    active.append(pending)
                pending = next(ranges, None)
            scopes[call] = sorted(
                (scope for scope in active if scope.end >= call.end),
                key=lambda scope: scope.dur,
            )
    return scopes


def device_kind(event):
    if event.cat == "gpu_memcpy":
        return "copy"
    if event.cat == "gpu_memset":
        return "memset"
    name = event.name.lower()
    if EP_SYNC.search(name):
        return "device_wait"
    if "hybrid_ep::update_expected_value_kernel" in name:
        return "device_control"
    if "hybrid_ep::scan" in name:
        return "compute"
    if any(tag in name for tag in (
        "hybrid_ep::", "deep_ep::", "deepep", "ag_nvl_kernel", "rs_nvl_kernel", "nccl",
    )):
        return "communication"
    if COPY_KERNEL.search(name):
        return "copy_kernel"
    return "compute"


def stable_ce_signature(event):
    # Four 4096-row B2/S4096 CE launches per microbatch (two chunks, main and MTP).
    # stable_ce.py launches these source symbols with one block per row and four warps.
    return (event.name in {"forward_kernel", "backward_kernel"}
            and event.args.get("grid") == [4096, 1, 1]
            and event.args.get("block") == [128, 1, 1])


def component(event, names, kind):
    name = event.name.lower()
    context = " ".join(names)
    if event.cat != "kernel":
        return kind
    if stable_ce_signature(event):
        return "cross_entropy_kernel_signature"
    if "hybrid_ep::scan" in name:
        return "moe_routing_transport_metadata"
    if kind == "device_control":
        return "moe_transport_control"
    if kind == "device_wait":
        return "moe_transport_wait" if "hybrid_ep" in name or "deepep" in name else kind
    if name.startswith("nccl") and kind == "communication":
        if "allgather" in name or "reducescatter" in name:
            return "collective_allgather_reduce_scatter"
        if (event.args.get("In msg nelems") == 128 and event.args.get("Group size") == 8
                and "allreduce_sum_u64" in name):
            return "moe_capacity_collective_candidate"
        return "collective_other"
    if kind == "communication":
        if "dispatch" in name or "ag_nvl" in name:
            return "moe_dispatch_transport"
        if "combine" in name or "rs_nvl" in name:
            return "moe_combine_transport"
        return "communication_unknown"
    if re.search(r"routing_metadata|tokens_per_expert|_build_m_indices", name):
        return "moe_routing"
    if GROUPED_GEMM.search(name) or ("deep_gemm::sm90_fp8_gemm" in name and
                                   GROUPED_GEMM.search(context)):
        return "moe_grouped_gemm"
    if "hybridep::dispatch" in context or "hybridep::combine" in context:
        if re.search(r"quantiz|fp8|swiglu|transpose|row2col", name):
            return "moe_fp8_quant_layout"
        return "moe_dispatch_prep" if "hybridep::dispatch" in context else "moe_combine_prep"
    if re.search(r"swiglu_quant|rowwise_1d_quantize|swiglu_(?:forward|backward)_kernel", name):
        return "moe_fp8_quant_layout"
    if any(tag in context for tag in ("gfsdp::", "fsdp::")):
        return "fsdp_bookkeeping"
    # Shape-generic math has no trustworthy module owner in unannotated traces.
    if GEMM.search(name) and re.search(r"\bmla\b|\.attention\.(?:wq|wkv|wo)\b", context):
        return "mla_projection"
    if GEMM.search(name) and re.search(r"\bgdn\b|\.attention\.(?:qkv_proj|z_proj|out_proj)\b", context):
        return "gdn_projection"
    for label, pattern in COMPONENTS:
        if pattern.search(name):
            return label
    for scope in names:
        for label, pattern in COMPONENTS:
            if pattern.search(scope):
                return label
    if "torchtitan::fp8_dense_gemm" in context and GEMM.search(name):
        return "fp8_dense_gemm_unknown_owner"
    if re.search(r"quantiz|fp8_transpose|row2col|fp8.*scale", name + " " + context):
        return "fp8_quant_layout_unknown_owner"
    if GEMM.search(name + " " + context):
        return "matmul_unknown_owner"
    if "aten::topk" in context or "gathertopk" in name:
        return "topk_unknown_owner"
    if "_log_softmax" in name + " " + context:
        return "log_softmax_unknown_owner"
    return "other_unattributed"


def implementation(event, names):
    if event.cat != "kernel":
        return event.cat
    name = event.name.lower()
    compiled = any(
        scope.startswith(("compiledfunction", "torch-compiled region", "## call compiledfxgraph"))
        for scope in names
    )
    if INDUCTOR.match(name):
        return "inductor_fused"
    if "at::native::" in name or any(scope.startswith("aten::") for scope in names):
        return "aten_inside_compiled_scope" if compiled else "aten_outside_compiled_scope"
    return "native_or_custom_in_compiled_scope" if compiled else "native_or_custom"


def semantic_phase(event, host, names, marker, first_backward, family):
    if marker is None:
        return "outside_microbatches" if host else "unknown_unlinked"
    name = event.name.lower()
    context = " ".join(names)
    backward = any(scope.startswith(AUTOGRAD) for scope in names)
    backward = backward or "compiledfunctionbackward" in context or "::post_backward" in context
    replay = re.search(r"checkpoint.*recomput|\bsac_recomput|\brecompute_fn\b", context)
    if replay:
        return "explicit_sac_recompute"
    if family == "cross_entropy_kernel_signature":
        return "backward" if name == "backward_kernel" else "original_forward"
    if backward:
        # Native forward inside this custom VJP is distinct from model forward.
        native_forward = re.search(r"(?:^|_)(?:fwd|fprop)(?:_|\b)|normforward", name)
        native_backward = re.search(r"(?:^|_)(?:bwd|bprop)(?:_|\b)|normbackward", name)
        if "torchtitan::fla_vjp" in context and native_forward:
            return "vjp_internal_forward"
        if native_forward and not INDUCTOR.match(name):
            return "forward_kernel_under_backward"
        if native_backward or re.search(
            r"stable_ce_backward|::(?:dispatch|combine)_bwd\b|"
            r"scaled_dot_product_cudnn_attention_backward|::(?:post_backward|backward_prefetch)|"
            r"torch::autograd::accumulategrad", context,
        ):
            return "backward"
        # Eager non-reentrant checkpoint unpack can run inside an ordinary
        # autograd evaluation and has no dedicated annotation in this recipe.
        return "backward_or_recompute"
    if family == "cross_entropy" and not re.search(r"backward|\bbwd\b", name + " " + context):
        return "original_forward"
    if first_backward is not None and host.start < first_backward and host.thread == marker.thread:
        return "original_forward"
    return "unknown_phase"


def metrics(events):
    intervals = sorted((event.start, event.end) for event in events if event.dur > 0)
    return {"count": len(events), "sum_ms": ms(sum(event.dur for event in events)),
            "busy_union_ms": ms(union(intervals))}


def union(intervals):
    covered = 0
    end = None
    for start, stop in intervals:
        if end is None or start >= end:
            covered += stop - start
            end = stop
        elif stop > end:
            covered += stop - end
            end = stop
    return covered


def counts(events, key):
    return {name: metrics(group) for name, group in sorted(index_by(events, key).items())}


def containing_marker(event, markers):
    if event is not None:
        return next((mark for mark in markers if mark.pid == event.pid and mark.start <= event.start < mark.end), None)
    return None


def readbacks(device, link, scopes, apis, origin, assignment):
    syncs = index_by(
        [api for api in apis if api_kind(api) == "host_synchronize" and api.external],
        lambda api: (api.thread, api.external),
    )
    result = []
    for event in device:
        if event.cat != "gpu_memcpy" or "DtoH" not in event.name:
            continue
        host = link.get(event)
        owner = next((scope for scope in scopes.get(host, []) if scope.name in READBACK_OWNER), None)
        external_scope = next(
            (scope for scope in scopes.get(host, []) if host.external and scope.external == host.external), None,
        )
        shared_scope = owner or external_scope
        paired = []
        if shared_scope:
            paired = [
                sync for sync in syncs[host.thread, host.external]
                if shared_scope in scopes[sync] and host.end <= sync.start and event.end <= sync.end
            ]
        sync = paired[0] if len(paired) == 1 else None
        result.append({
            "microbatch": assignment.get(event),
            "device_start_ms": ms(event.start - origin),
            "bytes": event.args.get("bytes"),
            "transfer": event.name,
            "host_api": host.name if host else None,
            "host_api_ms": ms(host.dur) if host else None,
            "host_ancestors": [scope.name for scope in scopes.get(host, [])],
            "read_type": (
                "scalar" if owner and owner.name in {"aten::_local_scalar_dense", "aten::item"}
                else "tensor" if owner else "unknown"
            ),
            "matched_following_sync": sync.name if sync else None,
            "matched_following_sync_ms": ms(sync.dur) if sync else None,
        })
    return result


def gpu_window(events, all_device, kinds, origin, lo, hi):
    if not events:
        return {"owned_events": 0}
    start = min(event.start for event in events)
    end = max(event.end for event in events)
    active = [event for event in all_device if event.start < end and start < event.end]
    intervals = sorted((max(start, event.start), min(end, event.end)) for event in active)
    busy = union(intervals)
    intervals_by_kind = defaultdict(list)
    for event in active:
        intervals_by_kind[kinds[event]].append((max(start, event.start), min(end, event.end)))
    compute = intervals_by_kind["compute"]
    communication = intervals_by_kind["communication"]
    compute_busy = union(sorted(compute))
    communication_busy = union(sorted(communication))
    overlap = compute_busy + communication_busy - union(sorted(compute + communication))
    return {
        "host_scope_ms": ms(hi - lo),
        "host_scope_start_ms": ms(lo - origin),
        "host_scope_end_ms": ms(hi - origin),
        "gpu_first_ms": ms(start - origin),
        "gpu_tail_ms": ms(end - origin),
        "gpu_tail_after_host_ms": ms(max(0, end - hi)),
        "host_start_to_gpu_tail_ms": ms(end - lo),
        "host_start_to_final_boundary_ms": ms(max(hi, end) - lo),
        "gpu_span_ms": ms(end - start),
        "owned_device": metrics(events),
        "owned_kinds": counts(events, kinds.__getitem__),
        "span_all_activity_busy_union_ms": ms(busy),
        "span_idle_ms": ms(end - start - busy),
        "span_compute_busy_union_ms": ms(compute_busy),
        "span_busy_without_compute_ms": ms(busy - compute_busy),
        "span_communication_compute_overlap_ms": ms(overlap),
        "span_other_window_events": len(set(active) - set(events)),
        "owned_kernel_count": sum(event.cat == "kernel" for event in events),
        "owned_kernel_sum_ms": ms(sum(event.dur for event in events if event.cat == "kernel")),
        "owned_device_by_stream": counts(
            events, lambda event: str(event.args.get("stream", event.tid)),
        ),
    }


def analyze(path, update, rank, expected_microbatches):
    raw = path.read_bytes()
    contents = gzip.decompress(raw) if raw.startswith(b"\x1f\x8b") else raw
    trace = json.loads(contents, parse_float=Decimal)
    source = trace["traceEvents"]
    events = [
        Event(index, row["name"], row["cat"], ns(row["ts"]), ns(row["dur"]),
              row["pid"], row["tid"], row.get("args", {}))
        for index, row in enumerate(source)
        if row.get("ph") == "X" and row.get("cat") in GPU | API | CPU
    ]
    if any(event.dur < 0 for event in events):
        raise ValueError(f"{path}: negative duration")
    device = [event for event in events if event.cat in GPU]
    apis = [event for event in events if event.cat in API]
    cpu = [event for event in events if event.cat in CPU]
    devices = {event.args.get("device", event.pid) for event in device}
    if len(devices) != 1 or not apis:
        raise ValueError(f"{path}: expected one CUDA device and host CUDA API events")
    markers = sorted(
        (e for e in cpu if e.cat == "user_annotation"
         and e.name == f"task.profile_update.{update}.microbatch.rank{rank}"),
        key=lambda event: event.start,
    )
    if len(markers) != expected_microbatches or any(
        left.end > right.start for left, right in zip(markers, markers[1:])
    ) or any(not marker.dur for marker in markers):
        present = Counter(event.name for event in cpu if MICROBATCH.fullmatch(event.name))
        raise ValueError(f"{path}: expected {expected_microbatches} disjoint CPU markers; found {present}")
    train = [
        event for event in cpu if event.cat == "user_annotation"
        and event.name == f"task.profile_update.{update}.train_step.rank{rank}"
    ]
    origin = min(event.start for event in train + markers)
    processes = {event.pid for event in apis}
    link, coverage = launches(source, device, apis, len(processes) == 1)
    scopes = ancestors(apis, cpu)
    first_backward = {
        mark: min((event.start for event in cpu if event.name.startswith(AUTOGRAD)
                   and containing_marker(event, markers) is mark), default=None)
        for mark in markers
    }
    ordinal = {mark: number for number, mark in enumerate(markers, 1)}
    owners = {event: containing_marker(link.get(event), markers) for event in device}
    assignment = {event: ordinal.get(mark) for event, mark in owners.items()}
    kinds = {event: device_kind(event) for event in device}
    names = {event: [scope.name.lower() for scope in scopes.get(link.get(event), [])] for event in device}
    families = {event: component(event, names[event], kinds[event]) for event in device}
    implementations = {event: implementation(event, names[event]) for event in device}
    phases = {
        event: semantic_phase(event, link.get(event), names[event], mark, first_backward.get(mark), families[event])
        for event, mark in owners.items()
    }
    selected = [event for event in device if owners[event] is not None]
    kernels = [event for event in selected if event.cat == "kernel"]
    if not kernels:
        raise ValueError(f"{path}: no kernel connected to a microbatch launch")
    if set(assignment[event] for event in kernels) != set(ordinal.values()):
        raise ValueError(f"{path}: one or more microbatches have no linked kernels")

    def api_summary(calls):
        return {
            "by_kind": counts(calls, api_kind),
            "by_name": counts(calls, lambda call: call.name),
        }

    microbatches = []
    for mark in markers:
        owned = [event for event in selected if owners[event] is mark]
        calls = [call for call in apis if containing_marker(call, markers) is mark]
        row = gpu_window(owned, device, kinds, origin, mark.start, mark.end)
        row.update(
            index=ordinal[mark],
            off_annotation_thread_events=sum(link[event].thread != mark.thread for event in owned),
            first_autograd_start_ms=ms(first_backward[mark] - origin) if first_backward[mark] is not None else None,
            cuda_api=api_summary(calls),
        )
        microbatches.append(row)
    calls = [call for call in apis if containing_marker(call, markers)]
    launch_set = set(link.values())
    unmatched_calls = [call for call in calls if api_kind(call) in {
        "kernel_launch", "copy_submit", "memset_submit",
    } and call not in launch_set]
    uncovered = [event for event in device if event not in link]
    outside = [event for event in device if event in link and owners[event] is None]
    normalized = {
        event: re.sub(r"_\d+$", "", event.name) if implementations[event] == "inductor_fused" else event.name
        for event in kernels
    }
    receipt = {
        "trace": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "rank": rank,
        "update": update,
        "device": sorted(devices),
        "host_processes": sorted(processes),
        "units": "milliseconds",
        "raw_device_census": counts(device, lambda event: event.cat),
        "flow_phases": dict(Counter(row["ph"] for row in source if row.get("cat") == "ac2g")),
        "launch_coverage": counts(device, coverage.__getitem__),
        "unlinked_device_examples": [
            {"index": event.index, "name": event.name, "coverage": coverage[event]}
            for event in uncovered[:20]
        ],
        "linked_outside_microbatches": counts(outside, lambda event: event.cat),
        "unmatched_selected_submission_apis": counts(unmatched_calls, lambda call: call.name),
        "train_step_marker_count": len(train),
        "microbatches": microbatches,
        "integrated": gpu_window(selected, device, kinds, origin, markers[0].start, markers[-1].end),
        "cuda_api_microbatches": api_summary(calls),
        "cuda_api_full_trace": api_summary(apis),
        "dtoh_readbacks": readbacks(device, link, scopes, apis, origin, assignment),
        "kernels_by_implementation": counts(kernels, implementations.__getitem__),
        "kernels_by_semantic_phase": counts(kernels, phases.__getitem__),
        "kernels_by_component": counts(kernels, families.__getitem__),
        "kernels_by_component_implementation_phase": counts(
            kernels, lambda event: " | ".join((families[event], implementations[event], phases[event])),
        ),
        "inductor_normalized_names": counts(
            [event for event in kernels if implementations[event] == "inductor_fused"],
            normalized.__getitem__,
        ),
        "kernel_inventory": counts(kernels, lambda event: event.name),
        "kernel_inventory_component_implementation": counts(
            kernels, lambda event: " | ".join((families[event], implementations[event], event.name)),
        ),
        "aten_launch_owners": counts(
            [event for event in kernels if implementations[event].startswith("aten_")],
            lambda event: next((scope.name for scope in scopes.get(link[event], [])
                                if scope.name.startswith("aten::")), "aten_native_without_cpu_op"),
        ),
        "compiler_scopes": counts(
            [event for event in cpu if event.name.startswith(("CompiledFunction", "Torch-Compiled Region",
                                                             "## Call CompiledFxGraph", "TorchDynamo"))],
            lambda event: event.name.split(":")[0],
        ),
    }
    if len(selected) + len(outside) + len(uncovered) != len(device):
        raise AssertionError("device launch census does not reconcile")
    return receipt


def compare(compiled, eager):
    metrics = ("host_scope_ms", "gpu_span_ms", "owned_kernel_count", "owned_kernel_sum_ms",
               "span_idle_ms", "span_all_activity_busy_union_ms")
    deltas = {
        key: round(eager["integrated"][key] - compiled["integrated"][key], 6) for key in metrics
    }
    comparisons = {
        "direction": "no_compile minus compile",
        "integrated": deltas,
        "microbatches": [
            {key: round(eager_mb[key] - compiled_mb[key], 6) for key in metrics}
            for compiled_mb, eager_mb in zip(compiled["microbatches"], eager["microbatches"])
        ],
    }
    for key in ("kernels_by_component", "kernels_by_implementation", "kernels_by_semantic_phase"):
        base, alternate = compiled[key], eager[key]
        comparisons[key] = {
            name: {
                metric: round(alternate.get(name, {}).get(metric, 0) - base.get(name, {}).get(metric, 0), 6)
                for metric in ("count", "sum_ms")
            }
            for name in sorted(base.keys() | alternate.keys())
        }
    return comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("compile", type=Path)
    parser.add_argument("no_compile", type=Path, nargs="?")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--update", type=int, default=21)
    parser.add_argument("--microbatches", type=int, default=3)
    args = parser.parse_args()
    compiled = analyze(args.compile, args.update, args.rank, args.microbatches)
    report = {
        "method": [
            "Assign captured device work by the linked CUDA API start in each half-open CPU microbatch window, including launches on other threads in that process. Device tails can outlast the CPU marker.",
            "Device span idle uses the union of all captured GPU events inside the span, including those owned by other windows. Group duration sums count concurrent kernels separately; group busy unions can overlap.",
            "Component and phase attribution uses kernel names and complete same-thread CPU ancestry. Both AOT backward and eager autograd without explicit direction can contain checkpoint replay and differentiation.",
            "CE generic kernel names use the recipe's stable_ce.py symbols and 4096-row, four-warp launch signature. Dense GEMM module ownership stays unknown without a matching recorded scope. The 128-u64 capacity collective classification is a candidate based on launch metadata.",
            "Same-thread, same-external-id DtoH/sync pairs require a shared CPU scope and a sync end at or after the transfer end. Their CUDA API duration can include waiting for earlier work.",
            "A trace covers only the selected rank, update window and captured launches. GPU gaps do not identify their cause. Use independent unprofiled runs for throughput estimates.",
        ],
        "compile": compiled,
    }
    if args.no_compile is not None:
        eager = analyze(args.no_compile, args.update, args.rank, args.microbatches)
        report.update(no_compile=eager, comparison=compare(compiled, eager))
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
