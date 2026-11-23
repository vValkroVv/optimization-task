"""CPU fixtures for bounded trace attribution, pairing, and missing evidence."""

import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from analyze_ep4 import analyze_rank, pair_collectives, read_attempt


def fixture(rank, *, missing_group=False, with_sync=True, first_writer=None):
    rows = []
    first = rank * 10

    def event(name, category, time, duration, *, thread=7, pid=100, **args):
        rows.append(
            dict(
                name=name,
                cat=category,
                ph="X",
                ts=time,
                dur=duration,
                pid=pid,
                tid=thread,
                args=args,
            )
        )

    def kernel(name, launch, start, duration, corr, flow, stream=24, **args):
        corr += first
        event("cuLaunchKernel", "cuda_driver", launch, 1, correlation=corr)
        event(
            name,
            "kernel",
            start,
            duration,
            pid=rank,
            thread=stream,
            stream=stream,
            correlation=corr,
            **args,
        )
        if flow:
            rows.extend(
                [
                    dict(cat="ac2g", ph="s", pid=100, tid=7, ts=launch, id=corr),
                    dict(cat="ac2g", ph="f", pid=rank, tid=stream, ts=start, id=corr),
                ]
            )

    def memset(launch, start, corr):
        event("cudaMemsetAsync", "cuda_runtime", launch, 1, correlation=corr + first)
        event(
            "Memset (Device)",
            "gpu_memset",
            start,
            1,
            pid=rank,
            thread=1,
            stream=1,
            correlation=corr + first,
        )

    def metadata(sequence, dtype, ins, outs):
        data = {
            "Process Group Name": "65",
            "Process Group Ranks": "[0, 4]",
            "Group size": 2,
            "Seq": sequence,
            "dtype": dtype,
            "In msg nelems": ins,
            "Out msg nelems": outs,
        }
        if missing_group and sequence == 102:
            data.pop("Process Group Ranks")
        return data

    for start in (100, 300, 500):
        event(
            f"task.profile_update.21.microbatch.rank{rank}",
            "user_annotation",
            start,
            100,
        )
    event(
        "GFSDP::pre_backward (layers.6[efsdp])",
        "user_annotation",
        110,
        28 if first_writer else 74,
    )
    event("GFSDP::post_backward_reduce (layers.7[efsdp])", "user_annotation", 111, 19)
    event("GFSDP::all_gather (layers.6[efsdp])", "user_annotation", 138, 20)
    event("grouped_gemm::bwd::dgrad_m_grouped_fp8_gemm", "user_annotation", 164, 10)
    kernel(
        "ncclDevKernel_ReduceScatter",
        114,
        121,
        50,
        1,
        True,
        **metadata(101, "Float", 603_979_776, 301_989_888),
    )
    kernel(
        "ncclDevKernel_AllGather",
        139,
        205,
        9,
        2,
        True,
        **metadata(102, "Byte", 301_989_888, 603_979_776),
    )
    kernel(
        "ncclDevKernel_AllGather",
        145,
        217,
        5,
        3,
        False,
        **metadata(103, "Float", 18_432, 36_864),
    )
    kernel("deep_gemm_dgrad", 170, 230, 3, 4, True, stream=1)
    event("cudaEventRecordWithFlags", "cuda_runtime", 119, 1, correlation=11 + first)
    event("cudaStreamWaitEvent", "cuda_runtime", 132, 1, correlation=12 + first)
    if with_sync:
        event(
            "Stream Wait Event",
            "cuda_sync",
            133,
            1,
            pid=rank,
            thread=1,
            correlation=12 + first,
            wait_on_cuda_event_record_corr_id=11 + first,
            wait_on_stream=24,
        )
    if first_writer:
        event(
            "GFSDP::post_backward_reduce (layers.6[efsdp])", "user_annotation", 194, 5
        )
        event("CompiledFunctionBackward", "cpu_op", 159, 35)
        grouped_gemm = "deep_gemm::sm90_fp8_gemm_1d2d_impl<0, (deep_gemm::GemmType)1>"
        kernel(grouped_gemm, 160, 226, 1, 8, True, stream=1)
        event("torchtitan::transpose_prepared_fp8_weight", "cpu_op", 161, 4)
        kernel("_transpose_last_two_kernel", 162, 228, 1, 5, True, stream=1)
        kernel(grouped_gemm, 166, 234, 1, 6, True, stream=1)
        event("cudacherry::row2col_dq", "cpu_op", 171, 4)
        kernel("requantize_grouped_kernel", 172, 236, 1, 7, True, stream=1)
        event("cudaStreamWaitEvent", "cuda_runtime", 177, 1, correlation=13 + first)
        event(
            "Stream Wait Event",
            "cuda_sync",
            177,
            1,
            pid=rank,
            thread=1,
            correlation=13 + first,
            wait_on_cuda_event_record_corr_id=-1,
            wait_on_stream=-1,
        )
        for index, kind in enumerate(
            (first_writer, "zero" if first_writer == "gemm" else "gemm")
        ):
            host, device = 176 + index * 10, 240 + index * 10
            op = "zero" if kind == "zero" else "write"
            event(f"torchtitan::{op}_direct_expert_dw", "cpu_op", host, 7)
            if kind == "gemm":
                kernel(
                    "deep_gemm::sm90_k_grouped_fp8_gemm_rounded_f32_impl",
                    host + 3,
                    device,
                    3,
                    20 + index * 2,
                    True,
                    stream=1,
                )
            memset(host + 5, device + 3, 21 + index * 2)
    return {
        "traceEvents": rows,
        "host_name": "one-node",
        "baseTimeNanoseconds": 1,
        "distributedInfo": {
            "rank": rank,
            "world_size": 8,
            "pg_config": [
                {"pg_name": "65", "pg_desc": "mesh_efsdp", "ranks": [0, 4]},
            ],
        },
    }


class TraceTest(unittest.TestCase):
    def test_pairs_scoped_by_host_launch_and_seq(self):
        left, right = (analyze_rank(fixture(rank), rank, 21) for rank in (0, 4))
        pairs, unresolved = pair_collectives(left, right)
        self.assertEqual([row["sequence"] for row in pairs], [101, 102, 103])
        self.assertEqual(unresolved, [])
        self.assertEqual(
            left["device_launch_coverage"], {"ac2g": 3, "validated_correlation": 1}
        )
        self.assertEqual(left["collectives"][1]["microbatch"], 1)
        self.assertEqual(left["collectives"][1]["start_ns"], 205_000)
        self.assertEqual(
            left["queued_behind_expert_rs"][0]["preceding_rs_end_ns"], 171_000
        )
        observed = left["backward_windows"][0]["first_device_operations_in_host_window"]
        self.assertEqual(observed["dgrad"]["start_ns"], 230_000)
        wait = left["gfsdp_stream_waits"][0]
        self.assertEqual(wait["record_api"]["start_ns"], 119_000)
        self.assertEqual(wait["dependency_evidence"], "matched_cuda_record")
        self.assertIsNone(wait["producer_device_timestamp_ns"])

    def test_missing_metadata_and_wait_id_are_explicit(self):
        left = analyze_rank(fixture(0, with_sync=False), 0, 21)
        right = analyze_rank(fixture(4, missing_group=True), 4, 21)
        pairs, unresolved = pair_collectives(left, right)
        self.assertEqual([row["sequence"] for row in pairs], [101, 103])
        self.assertEqual(unresolved[0]["sequence"], 102)
        self.assertIn(
            "missing or unexpected expert group metadata", right["unknown_edges"]
        )
        self.assertEqual(
            left["gfsdp_stream_waits"][0]["dependency_evidence"],
            "unknown_event_identity",
        )

    def test_read_attempt_carries_trace_receipts_and_rejects_mutation(self):
        capture = {"profile_ranks": [0, 4], "target_update": 21, "traces_by_rank": {}}
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for rank in (0, 4):
                name = f"rank{rank}_trace.json"
                data = json.dumps(fixture(rank)).encode()
                (root / name).write_bytes(data)
                capture["traces_by_rank"][str(rank)] = dict(
                    path=name,
                    size_bytes=len(data),
                    sha256=hashlib.sha256(data).hexdigest(),
                )
            (root / "summary.json").write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "mode": "profile_update",
                        "profile_capture": capture,
                    }
                )
            )
            result = read_attempt(root)
            self.assertEqual(len(result["expert_collective_pairs"]), 3)
            self.assertEqual(
                result["rank_traces"][0]["native_receipt"],
                capture["traces_by_rank"]["0"],
            )
            path = root / "rank0_trace.json"
            content = path.read_bytes()
            path.write_bytes(b" " + content[1:])
            with self.assertRaisesRegex(ValueError, "SHA256"):
                read_attempt(root)

    def test_compiled_direct_writer_wait_gemm_and_zero_paths(self):
        for first in ("gemm", "zero", "native_empty"):
            with self.subTest(first=first):
                result = analyze_rank(fixture(0, first_writer=first), 0, 21)
                direct = result["direct_dw"]
                self.assertEqual(direct["unassigned_calls"], [])
                self.assertEqual(
                    direct["coverage"],
                    {
                        "cpu_writer_calls": 2,
                        "calls_without_matched_device_write": 0,
                        "owner_windows_with_two_calls": 1,
                        "writer_waits_without_record_identity": 1,
                    },
                )
                self.assertEqual(len(result["gfsdp_stream_waits"]), 1)
                window = direct["owner_windows"][0]
                self.assertEqual(window["owner"], "layers.6[efsdp]")
                prepared = window["pre_first_writer_device_work"]
                self.assertEqual(
                    [row["kind"] for row in prepared],
                    [
                        "prepared_weight_transpose",
                        "grouped_deepgemm_dgrad_candidate",
                        "row2col",
                    ],
                )
                writer, other = window["calls"]
                self.assertEqual(writer["waits"][0]["waiting_stream"], 1)
                self.assertIsNone(writer["waits"][0]["record_api"])
                self.assertEqual(other["waits"], [])
                self.assertEqual(
                    writer["device_writes"][0]["kind"],
                    "deepgemm" if first == "gemm" else "zero",
                )
                native = writer if first == "gemm" else other
                self.assertEqual(
                    [row["kind"] for row in native["device_writes"]],
                    ["deepgemm", "zero"],
                )


if __name__ == "__main__":
    unittest.main()
