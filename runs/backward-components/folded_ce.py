"""Replay both native MTP-loss chunks with captured production-data tensors."""

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import torch
import triton
from component_trace import MICROBATCHES, summarize as summarize_trace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "gift"))

from torchtitan.components.loss import IGNORE_INDEX, MTPLoss  # noqa: E402
from torchtitan.config import CompileConfig  # noqa: E402
from torchtitan.models.common.nn_modules import Linear  # noqa: E402

FLAG = "GIFT_STABLE_CE_FOLDED_BACKWARD"


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_fixture(path):
    fixture = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if "ce" in fixture:
        capture = fixture["metadata"]
        data = fixture["ce"]
        head_name = capture["lm_head_class"].rsplit(".", 1)[-1]
        if head_name not in ("GFSDPLinear", "Linear") or capture["lm_head_bias"]:
            raise ValueError("Expected the production bias-free Linear LM head")
    else:
        # Historical Quack input: real data with canonical reinitialized weights.
        legacy = json.loads(path.with_name("receipt.json").read_text())
        if sha256(path) != legacy["prepared_sha256"]:
            raise ValueError("Prepared fixture does not match its input receipt")
        capture = {"origin": legacy["state"], "compile_backend": "inductor"}
        data = {
            **fixture,
            "denominator": float(legacy["global_denominator"]),
            "n_chunks": 2,
            "head_weights": [1.0, 0.3],
            "ignore_index": IGNORE_INDEX,
        }
    hidden, weight, labels = (data[key] for key in ("hidden", "weight", "labels"))
    positions = data["positions"]
    if hidden.shape != (2, 2, 4096, 4096) or weight.shape != (128256, 4096):
        raise ValueError("Expected production B2/S4096/D4096/V128256 and two heads")
    if hidden.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise ValueError("Expected BF16 hidden states and head weight")
    if not (hidden.is_contiguous() and weight.is_contiguous()):
        raise ValueError("Expected contiguous hidden states and head weight")
    if labels.dtype != torch.int64 or labels.shape != hidden.shape[1:3]:
        raise ValueError("Expected original, unshifted int64 main-head labels")
    if positions is not None and positions.shape != labels.shape:
        raise ValueError("Packed positions must match the labels")
    denominator = data["denominator"]
    if isinstance(denominator, torch.Tensor) and denominator.ndim != 0:
        raise ValueError("Expected a scalar denominator")
    if not math.isfinite(float(denominator)) or float(denominator) <= 0:
        raise ValueError("Expected a positive global denominator")
    if (
        list(data["head_weights"]) != [1.0, 0.3]
        or data["n_chunks"] != 2
        or data["ignore_index"] != IGNORE_INDEX
    ):
        raise ValueError(
            "Expected 2 chunks, 1.0/0.3 loss weights, and ignore index -100"
        )
    return (
        {
            key: value.cuda() if isinstance(value, torch.Tensor) else value
            for key, value in data.items()
        },
        capture,
    )


class Replay:
    def __init__(self, fixture, backend):
        self.hidden = fixture["hidden"].detach().requires_grad_()
        self.labels = fixture["labels"]
        self.positions = fixture["positions"]
        self.denominator = fixture["denominator"]
        self.num_chunks = fixture["n_chunks"]
        self.head_weights = fixture["head_weights"]
        self.compile_config = CompileConfig(
            enable=True, components=["loss"], backend=backend
        )
        self.weight = torch.nn.Parameter(fixture["weight"])
        vocab, width = self.weight.shape
        with torch.device("meta"):
            self.head = Linear(
                Linear.Config(in_features=width, out_features=vocab, bias=False)
            )
        self.head.weight = self.weight
        self.head.compile(backend=backend, fullgraph=True)
        self.dw = torch.empty_like(self.weight, dtype=torch.float32)
        self.weight.register_post_accumulate_grad_hook(self.collect_dw)
        self.expected_labels = self.get_expected_labels()

    def collect_dw(self, param):
        # Match the native GFSDP local accumulation boundary: each BF16
        # chunk's dW contributes to the same FP32 accumulator.
        if self.chunks_seen == 0:
            self.dw.copy_(param.grad)
        else:
            self.dw.add_(param.grad)
        self.chunks_seen += 1
        param.grad = None

    def get_expected_labels(self):
        mtp_labels = torch.full_like(self.labels, IGNORE_INDEX)
        mtp_labels[:, :-1] = self.labels[:, 1:]
        if self.positions is not None:
            same_document = self.positions[:, 1:] == self.positions[:, :-1] + 1
            mtp_labels[:, :-1].masked_fill_(~same_document, IGNORE_INDEX)
        return torch.cat((self.labels, mtp_labels))

    def make_loss(self, variant):
        os.environ[FLAG] = "1" if variant == "folded" else "0"
        loss = MTPLoss(
            MTPLoss.Config(
                num_chunks=self.num_chunks, loss_weights=self.head_weights[1:]
            ),
            compile_config=self.compile_config,
        )
        loss.set_lm_head(self.head)
        return loss

    def run(self, loss):
        self.hidden.grad = None
        self.chunks_seen = 0
        loss.reset_components()
        target = loss.pack_targets(self.labels, {"positions": self.positions})
        result = loss(self.hidden, target, self.denominator)
        result.backward()
        if self.chunks_seen != self.num_chunks:
            raise RuntimeError(f"LM-head backward ran {self.chunks_seen} times")
        return result.detach()


def equal_bits(reference, actual):
    integer_type = {
        torch.bfloat16: torch.int16,
        torch.float32: torch.int32,
    }[reference.dtype]
    return (
        reference.dtype == actual.dtype
        and reference.shape == actual.shape
        and torch.equal(reference.view(integer_type), actual.view(integer_type))
    )


def record_raw_losses(loss, variant, raw_losses):
    if variant == "folded":
        fn = loss._folded_ce
        if fn is None:
            raise RuntimeError("Folded CE was not enabled")

        def folded(*args):
            raw = fn(*args)
            raw_losses.extend(value.detach() for value in raw)
            return raw

        loss._folded_ce = folded
    else:
        fn = loss.fn

        def stable(*args):
            raw = fn(*args)
            raw_losses.append(raw.detach())
            return raw

        loss.fn = stable


def check_numerics(replay):
    baseline, folded = (replay.make_loss(variant) for variant in ("base", "folded"))
    raw_base, raw_folded = [], []
    record_raw_losses(baseline, "base", raw_base)
    record_raw_losses(folded, "folded", raw_folded)

    # Use one and the same logits allocation for both CE routes, one chunk at a time.
    ce_results = []
    for hidden, labels in zip(
        replay.hidden.detach().flatten(0, 1).chunk(replay.num_chunks, dim=1),
        replay.expected_labels.chunk(replay.num_chunks, dim=1),
    ):
        labels = labels.contiguous()
        logits = replay.head(hidden.contiguous()).detach().requires_grad_()
        baseline.reset_components()
        folded.reset_components()
        base_value = baseline._chunk_loss(
            logits, labels, replay.denominator, num_heads=len(replay.head_weights)
        )
        folded_value = folded._chunk_loss(
            logits, labels, replay.denominator, num_heads=len(replay.head_weights)
        )
        (base_grad,) = torch.autograd.grad(base_value, logits)
        (folded_grad,) = torch.autograd.grad(folded_value, logits)
        baseline_pair = raw_base[-len(replay.head_weights) :]
        folded_pair = raw_folded[-len(replay.head_weights) :]
        ce_results.append(
            {
                "raw_head_sums": [float(raw) for raw in baseline_pair],
                "raw_head_sums_bitwise": len(folded_pair) == len(replay.head_weights)
                and all(
                    equal_bits(ref, got) for ref, got in zip(baseline_pair, folded_pair)
                ),
                "weighted_chunk_loss": float(base_value.detach()),
                "weighted_chunk_loss_bitwise": equal_bits(base_value, folded_value),
                "bf16_dlogits_bitwise": equal_bits(base_grad, folded_grad),
                "bf16_dlogits_elements": base_grad.numel(),
                "dlogits_stride": list(folded_grad.stride()),
                "ignore_rows": int((labels == IGNORE_INDEX).sum()),
                "ignored_dlogits_are_positive_zero": (
                    torch.count_nonzero(
                        folded_grad.view(torch.int16)[labels == IGNORE_INDEX]
                    ).item()
                    == 0
                ),
            }
        )
        del logits, base_grad, folded_grad, base_value, folded_value

    # Check the actual full MTP path, including target packing, both compiled
    # head calls and the accumulated weight gradient. Numeric-only hooks do
    # not participate in either timing or the trace pass.
    base_acceptance, folded_acceptance = [], []
    baseline.acceptance_sinks.append(base_acceptance)
    folded.acceptance_sinks.append(folded_acceptance)
    base_targets, folded_targets = [], []
    for loss, targets in ((baseline, base_targets), (folded, folded_targets)):
        chunk_fn = loss._chunk_loss

        def record_targets(
            logits, labels, denominator, *, num_heads, fn=chunk_fn, sink=targets
        ):
            sink.append(labels.detach())
            return fn(logits, labels, denominator, num_heads=num_heads)

        loss._chunk_loss = record_targets

    base_total = replay.run(baseline)
    base_dx = replay.hidden.grad
    base_dw = replay.dw.clone()
    folded_total = replay.run(folded)
    base_components, folded_components = baseline.components(), folded.components()
    targets = replay.expected_labels.chunk(replay.num_chunks, dim=1)
    target_exact = len(base_targets) == len(folded_targets) == len(targets) and all(
        torch.equal(ref, base) and torch.equal(ref, candidate)
        for ref, base, candidate in zip(targets, base_targets, folded_targets)
    )
    acceptance_exact = (
        len(base_acceptance) == len(folded_acceptance) == 1
        and base_acceptance[0].keys() == folded_acceptance[0].keys()
        and all(
            equal_bits(base_value, folded_acceptance[0][name])
            for name, base_value in base_acceptance[0].items()
        )
    )
    return {
        "chunks": ce_results,
        "full_loss": float(base_total),
        "full_loss_bitwise": equal_bits(base_total, folded_total),
        "metrics": {key: float(value) for key, value in base_components.items()},
        "metrics_bitwise": base_components.keys() == folded_components.keys()
        and all(
            equal_bits(value, folded_components[key])
            for key, value in base_components.items()
        ),
        "acceptance_counters_bitwise": acceptance_exact,
        "folded_targets_match_independent_shift_oracle": target_exact,
        "valid_cross_chunk_mtp_targets": int(
            (
                replay.expected_labels[
                    replay.labels.shape[0] :,
                    replay.labels.shape[1] // replay.num_chunks - 1,
                ]
                != IGNORE_INDEX
            ).sum()
        ),
        "bf16_dhidden_bitwise_by_chunk": [
            equal_bits(ref, actual)
            for ref, actual in zip(
                base_dx.chunk(replay.num_chunks, dim=2),
                replay.hidden.grad.chunk(replay.num_chunks, dim=2),
            )
        ],
        "accumulated_fp32_dw_bitwise": equal_bits(base_dw, replay.dw),
    }


def passed_numerics(result):
    chunk_keys = (
        "raw_head_sums_bitwise",
        "weighted_chunk_loss_bitwise",
        "bf16_dlogits_bitwise",
        "ignored_dlogits_are_positive_zero",
    )
    keys = (
        "full_loss_bitwise",
        "metrics_bitwise",
        "acceptance_counters_bitwise",
        "folded_targets_match_independent_shift_oracle",
        "accumulated_fp32_dw_bitwise",
    )
    return (
        all(result[key] for key in keys)
        and all(result["bf16_dhidden_bitwise_by_chunk"])
        and all(all(chunk[key] for key in chunk_keys) for chunk in result["chunks"])
        and result["valid_cross_chunk_mtp_targets"] > 0
    )


def benchmark(replay, loss, warmup, samples):
    for _ in range(warmup):
        replay.run(loss)
    replay.hidden.grad = None
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    allocated_start = torch.cuda.memory_allocated()
    reserved_start = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    event_times, wall_times = [], []
    for _ in range(samples):
        begin = time.perf_counter()
        start.record()
        replay.run(loss)
        end.record()
        end.synchronize()
        wall_times.append((time.perf_counter() - begin) * 1000)
        event_times.append(start.elapsed_time(end))
    return {
        "warmup_microbatches": warmup,
        "wall_ms": wall_times,
        "cuda_events_submission_inclusive_ms": event_times,
        "median_wall_ms": statistics.median(wall_times),
        "median_cuda_event_ms": statistics.median(event_times),
        "allocated_start_bytes": allocated_start,
        "reserved_start_bytes": reserved_start,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "allocator_retries": torch.cuda.memory_stats()["num_alloc_retries"],
        "scope": (
            "One process, one CUDA stream. Each sample starts before clearing"
            " gradients and ends after both chunks, dW accumulation and device"
            " synchronization. CUDA events include submission gaps."
        ),
    }


def trace(replay, loss, path, variant):
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as prof:
        for index in range(MICROBATCHES):
            with torch.profiler.record_function(f"mtp_head_microbatch_{index}"):
                replay.run(loss)
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(path))
    batch, seq_len = replay.labels.shape
    try:
        return summarize_trace(path, variant, batch * (seq_len // replay.num_chunks))
    except ValueError as error:
        return {"path": str(path), "passed": False, "error": str(error)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--mode", choices=("numeric", "benchmark"), required=True)
    parser.add_argument("--variant", choices=("base", "folded"))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.mode == "benchmark") != (args.variant is not None):
        parser.error("--variant is required only in benchmark mode")
    if args.warmup < 1 or args.samples < 1:
        parser.error("--warmup and --samples must be positive")
    if args.trace and args.mode != "benchmark":
        parser.error("--trace requires benchmark mode")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(2048)
    fixture, capture = load_fixture(args.fixture)
    backend = capture.get("compile_backend", "inductor")
    replay = Replay(fixture, backend)
    report = {
        "mode": args.mode,
        "variant": args.variant,
        "fixture": str(args.fixture),
        "fixture_sha256": sha256(args.fixture),
        "capture": capture,
        "head_class": f"{Linear.__module__}.{Linear.__name__}",
        "compile_backend": backend,
        "compile_lm_head_and_loss": True,
        "torch": str(torch.__version__),
        "torch_git": torch.version.git_version,
        "cuda": torch.version.cuda,
        "triton": triton.__version__,
        "gpu": torch.cuda.get_device_name(),
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_hashes": {
            path: sha256(ROOT / path)
            for path in (
                "gift/torchtitan/components/loss.py",
                "gift/torchtitan/experiments/four_fusion/stable_ce.py",
                "gift/torchtitan/models/common/nn_modules.py",
                "runs/backward-components/component_trace.py",
                "runs/backward-components/folded_ce.py",
            )
        },
        "hidden_shape": list(replay.hidden.shape),
        "weight_shape": list(replay.weight.shape),
        "logits_chunk_shape": [
            replay.hidden.shape[0] * replay.hidden.shape[1],
            replay.hidden.shape[2] // replay.num_chunks,
            replay.weight.shape[0],
        ],
        "dtypes": [str(replay.hidden.dtype), str(replay.weight.dtype)],
        "head_weights": replay.head_weights,
        "denominator": float(replay.denominator),
        "denominator_type": (
            str(replay.denominator.dtype)
            if isinstance(replay.denominator, torch.Tensor)
            else type(replay.denominator).__name__
        ),
        "valid_labels_by_head": [
            int((labels != IGNORE_INDEX).sum())
            for labels in replay.expected_labels.chunk(len(replay.head_weights), dim=0)
        ],
        "seed": 2048,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
    }
    if args.mode == "numeric":
        numerical = check_numerics(replay)
        report["numerical"] = numerical
        report["passed"] = passed_numerics(numerical)
    else:
        loss = replay.make_loss(args.variant)
        report["timing"] = benchmark(replay, loss, args.warmup, args.samples)
        if args.trace:
            report["trace"] = trace(
                replay, loss, args.output.with_suffix(".trace.json"), args.variant
            )
            report["passed"] = report["trace"]["passed"]
        else:
            report["passed"] = True
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"{args.mode} {args.variant or 'A/B'} passed={report['passed']} {args.output}",
        flush=True,
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
