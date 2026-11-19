#!/usr/bin/env python3
"""Align the compile A/B by microbatch, GFSDP boundary and CUDA stream."""

import argparse
from bisect import bisect_right
from collections import Counter
from decimal import Decimal
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import re

from analyze import API, CPU, GPU, Event, ancestors, launches, metrics, ms, ns, union


LAYER = re.compile(r"layers\.[0-7](?:\[fsdp\])?")
MARKER = "task.profile_update.21.microbatch.rank0"
SECTION_ORDER = (
    "forward:input_and_embedding",
    *(f"forward:layers.{i}" for i in range(8)),
    "forward:norm",
    "forward:mtp.layers.0",
    "loss:chunked_lm_head_and_ce",
    "backward:mtp.layers.0",
    "backward:norm",
    *(f"backward:layers.{i}" for i in reversed(range(8))),
    "backward:embedding_and_tail",
)


def named_boundary(event):
    if event.name == "autograd::engine::evaluate_function: _OwnerLookupBackward":
        return SECTION_ORDER[-1]
    match = re.fullmatch(r"GFSDP::pre_(forward|backward) \(([^)]+)\)", event.name)
    if not match:
        return None
    phase, unit = match.groups()
    if unit == "lm_head" and phase == "forward":
        return "loss:chunked_lm_head_and_ce"
    if LAYER.fullmatch(unit):
        unit = unit.removesuffix("[fsdp]")
    if unit in {"norm", "mtp.layers.0"} or re.fullmatch(r"layers\.[0-7]", unit):
        return f"{phase}:{unit}"
    return None


def signature(event):
    args = event.args
    return (
        event.cat, event.name, tuple(args.get("grid", ())), tuple(args.get("block", ())),
        args.get("bytes"), args.get("In msg nelems"), args.get("Out msg nelems"),
        args.get("dtype"), args.get("Group size"),
    )


def stream(event):
    return event.args.get("stream", event.tid)


class Trace:
    def __init__(self, path):
        raw = path.read_bytes()
        self.sha = hashlib.sha256(raw).hexdigest()
        rows = json.loads(raw, parse_float=Decimal)["traceEvents"]
        events = [
            Event(
                i, row["name"], row["cat"], ns(row["ts"]), ns(row["dur"]),
                row["pid"], row["tid"], row.get("args", {}),
            )
            for i, row in enumerate(rows)
            if row.get("ph") == "X" and row.get("cat") in GPU | API | CPU
        ]
        self.device = sorted((e for e in events if e.cat in GPU), key=lambda e: e.start)
        apis = [e for e in events if e.cat in API]
        cpu = [e for e in events if e.cat in CPU]
        self.link, self.coverage = launches(rows, self.device, apis, len({e.pid for e in apis}) == 1)
        self.scopes = ancestors(apis, cpu)
        marks = sorted(
            (e for e in cpu if e.cat == "user_annotation" and e.name == MARKER),
            key=lambda e: e.start,
        )
        if len(marks) != 3 or any(a.end > b.start for a, b in zip(marks, marks[1:])):
            raise ValueError(f"{path}: expected three disjoint {MARKER} markers")
        self.origin = marks[0].start
        self.groups = {}
        assigned = set()
        for index, mark in enumerate(marks, 1):
            bounds = {SECTION_ORDER[0]: mark.start}
            for event in cpu:
                if event.pid == mark.pid and mark.start <= event.start < mark.end:
                    label = named_boundary(event)
                    if label:
                        bounds[label] = min(event.start, bounds.get(label, event.start))
            if tuple(sorted(bounds, key=bounds.__getitem__)) != SECTION_ORDER:
                raise ValueError(f"{path}: missing or out-of-order section boundaries in MB{index}")
            starts = list(map(bounds.__getitem__, SECTION_ORDER))
            groups = [[] for _ in SECTION_ORDER]
            for event, host in self.link.items():
                if host.pid == mark.pid and mark.start <= host.start < mark.end:
                    groups[bisect_right(starts, host.start) - 1].append(event)
                    assigned.add(event)
            for label, group in zip(SECTION_ORDER, groups):
                self.groups[index, label] = sorted(group, key=lambda e: e.start)
        self.outside = [e for e in self.device if e in self.link and e not in assigned]
        self.unlinked = [e for e in self.device if e not in self.link]
        if sum(map(len, self.groups.values())) + len(self.outside) + len(self.unlinked) != len(self.device):
            raise ValueError(f"{path}: device activity accounting is incomplete")

    def row(self, event):
        link = self.link.get(event)
        scopes = self.scopes.get(link, [])
        owners = [s.name for s in scopes]
        return {
            "raw_event": event.index,
            "external_id": event.external,
            "correlation": event.corr,
            "name": event.name,
            "category": event.cat,
            "start_ms": ms(event.start - self.origin),
            "duration_ms": ms(event.dur),
            "launch_ms": ms(link.start - self.origin) if link else None,
            "stream": stream(event),
            "grid": event.args.get("grid"),
            "block": event.args.get("block"),
            "bytes": event.args.get("bytes"),
            "nccl_elements": event.args.get("In msg nelems"),
            "cpu_owner": next(
                (name for name in owners if "::" in name and name != MARKER),
                owners[0] if owners else None,
            ),
            "enclosing_cpu_scopes": [
                name for name in owners if not name.startswith("ProfilerStep#")
            ],
        }

    def section_metrics(self, group):
        main = [e for e in group if stream(e) == 7]
        if not main:
            return {"device": metrics(group), "stream7": metrics(main)}
        lo, hi = main[0].start, max(e.end for e in main)
        return {
            "device": metrics(group),
            "stream7": metrics(main),
            "stream7_span_ms": ms(hi - lo),
            "stream7_span_all_cuda_idle_ms": ms(hi - lo - union(
                (max(lo, e.start), min(hi, e.end))
                for e in self.device if e.end > lo and e.start < hi
            )),
        }


def align(eager, compiled):
    for region, eager_group in eager.groups.items():
        compiled_group = compiled.groups[region]
        streams = {}
        for sid in sorted({stream(e) for e in eager_group + compiled_group}):
            left = [e for e in eager_group if stream(e) == sid]
            right = [e for e in compiled_group if stream(e) == sid]
            pairs = SequenceMatcher(None, list(map(signature, left)), list(map(signature, right)), autojunk=False)
            streams[str(sid)] = {
                "eager": [eager.row(e) for e in left],
                "compile": [compiled.row(e) for e in right],
                "opcodes": pairs.get_opcodes(),
            }
        yield {
            "microbatch": region[0], "section": region[1],
            "eager": eager.section_metrics(eager_group),
            "compile": compiled.section_metrics(compiled_group),
            "streams": streams,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("eager", type=Path)
    parser.add_argument("compile", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    eager, compiled = Trace(args.eager), Trace(args.compile)
    receipt = {
        "traces": {
            name: {
                "sha256": trace.sha,
                "census": dict(Counter(e.cat for e in trace.device)),
                "outside": [trace.row(e) for e in trace.outside],
                "unlinked": [trace.row(e) for e in trace.unlinked],
                "launch_coverage": dict(Counter(trace.coverage.values())),
            }
            for name, trace in (("eager", eager), ("compile", compiled))
        },
        "time_origin": "start of the first CPU microbatch annotation, separately per arm",
        "opcodes": "difflib ordered signature alignment; equal means same recorded signature, not proof of tensor identity",
        "sections": list(align(eager, compiled)),
    }
    args.output.write_text(json.dumps(receipt, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
