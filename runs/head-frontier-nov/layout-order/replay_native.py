"""Run the existing compiled head replay against the measured lm10-c1 source."""

import argparse
from collections import Counter
import hashlib
import inspect
import json
import logging
import os
from pathlib import Path
import sys
import textwrap

import torch

PERSONAL = Path("/workspace-SR008.nfs2/vkropotin")
PRIOR = PERSONAL / "optimization-task/.worktrees/lm-head-mtp-fix/runs/lm-head-mtp"
PRODUCTION = PERSONAL / "lm10-c1"
WARMUP = 10
SAMPLES = 10

sys.path.insert(0, str(PRIOR))
import frontier_gather
import replay


def overlay(module, function, replacements):
    source = textwrap.dedent(inspect.getsource(function))
    for old, new in replacements:
        if source.count(old) != 1:
            raise ValueError(f"Expected a unique {function.__name__} anchor: {old}")
        source = source.replace(old, new)
    methods = {}
    exec(compile(source, __file__, "exec"), vars(module), methods)
    return methods[function.__name__]


def save_digest(tensors):
    return {
        name: {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "sha256": hashlib.sha256(
                tensor.reshape(-1).contiguous().view(torch.uint8).numpy()
            ).hexdigest(),
        }
        for name, tensor in tensors.items()
    }


def fp64_error(actual, reference):
    sums = torch.zeros(2, device="cuda", dtype=torch.float64)
    maximum = sums.new_zeros(())
    nonfinite = torch.zeros((), device="cuda", dtype=torch.long)
    for output, expected in zip(actual.split(256), reference.split(256), strict=True):
        diff = output.double() - expected
        sums += torch.stack((diff.square().sum(), expected.square().sum()))
        maximum = torch.maximum(maximum, diff.abs().max())
        nonfinite += (~output.isfinite()).sum()
    return {
        "relative_l2": float((sums[0] / sums[1]).sqrt()),
        "max_abs": float(maximum),
        "nonfinite": int(nonfinite),
    }


def run(chunks, output, *, arm="control", writer_hook=None, dw_first=False,
        alias=False, fp64=False, stages=False, warmup=WARMUP, samples=SAMPLES):
    """The optional writer hook receives (native, chunks, rows, vocab, width, stream)."""
    assert chunks in (2, 8)
    assert os.environ["PYTORCH_ALLOC_CONF"] == "expandable_segments:True"
    os.environ.update(
        GIFT_EXPERIMENT_LM_HEAD_MATRIX="new",
        GIFT_EXPERIMENT_LM_HEAD_MATRIX_CHUNKS=str(chunks),
    )
    logging.basicConfig(level=logging.INFO)
    replay.CURRENT = PRODUCTION
    source_receipt = {}
    memory, host_submit, host_wait = [], [], []
    kinds, beta = Counter(), Counter()
    diagnostic = False
    numerical_inputs = []
    diagnostic_events = []
    routed_ce = 0
    native_head = replay.make_head
    native_save = torch.save
    writer = gather = allocator = None

    def capture_memory(stats, memory_usage):
        external, total = memory_usage(stats)
        memory.append({
            "allocated_peak_bytes": stats["allocated_bytes.all.peak"],
            "reserved_peak_bytes": stats["reserved_bytes.all.peak"],
            "external_gib": external,
            "total_gib": total,
        })
        return total

    def note_times(submit, wait):
        host_submit.append(submit * 1000)
        host_wait.append(wait * 1000)

    def begin_diagnostic():
        nonlocal diagnostic
        diagnostic = True

    def measured_mm(grad, weight):
        if not diagnostic:
            return torch.mm(grad, weight)
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        result = torch.mm(grad, weight)
        end.record()
        diagnostic_events.append(("dhidden", begin, end))
        return result

    def install_head(weight):
        nonlocal gather, allocator
        from torchtitan.components import loss
        from torchtitan.distributed.gfsdp import _param_group, _staged_head_dw as staging
        from torchtitan.distributed.gfsdp import _staged_head_dw_writer as writer_module
        from torchtitan.experiments.four_fusion import stable_ce

        for module in (loss, _param_group, staging, writer_module, stable_ce):
            path = Path(module.__file__)
            source_receipt[str(path.relative_to(PRODUCTION))] = hashlib.sha256(path.read_bytes()).hexdigest()

        if dw_first or stages:
            anchor = (
                "    dx = torch.mm(grad, weight).view_as(input)\n"
                "    ctx.lease.write(grad, input.flatten(0, -2))"
            )
            dw = "    ctx.lease.write(grad, input.flatten(0, -2))"
            dx = "    dx = " + ("_benchmark_mm" if stages else "torch.mm") + "(grad, weight).view_as(input)"
            staging._benchmark_mm = measured_mm
            staging._StagedHeadLinear.backward = staticmethod(overlay(
                staging, staging._StagedHeadLinear.backward,
                [("@staticmethod\n", ""), (anchor, "\n".join((dw, dx) if dw_first else (dx, dw)))],
            ))

        original_ce = loss.MTPLoss.__init__

        def initialize_loss(self, *args, **kwargs):
            original_ce(self, *args, **kwargs)
            assert self._release_chunk_logits and self._folded_ce_compiled
            consume = self._folded_ce_consume
            assert consume is not None

            def counted_consume(*args, **kwargs):
                nonlocal routed_ce
                routed_ce += 1
                if routed_ce == chunks:
                    self._folded_ce_consume = consume
                return consume(*args, **kwargs)

            self._folded_ce_consume = counted_consume

        loss.MTPLoss.__init__ = initialize_loss
        make_writer = staging.make_head_dw_writer

        def observed_writer(mode, selected_chunks, rows, vocab, width, stream):
            nonlocal writer
            assert writer is None
            assert (mode, selected_chunks, rows, vocab, width) == ("new", chunks, 16384 // chunks, 128256, 4096)
            native = make_writer(mode, selected_chunks, rows, vocab, width, stream)
            assert native.pair == {2: "9.14", 8: "5.5"}[chunks]
            target = native if writer_hook is None else writer_hook(native, chunks, rows, vocab, width, stream)

            class ObservedWriter:
                name = target.name
                pair = target.pair
                version = target.version

                def __call__(self, grad, hidden, slot, add, stream):
                    beta[int(add)] += 1
                    kinds[str(hidden.stride())] += 1
                    if fp64 and diagnostic:
                        numerical_inputs.append((grad, hidden))
                    if stages and diagnostic:
                        begin = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        begin.record()
                        target(grad, hidden, slot, add, stream)
                        end.record()
                        diagnostic_events.append(("dw_with_pack", begin, end))
                    else:
                        target(grad, hidden, slot, add, stream)

            writer = target
            return ObservedWriter()

        staging.make_head_dw_writer = observed_writer
        head = native_head(weight)
        install_probe = overlay(frontier_gather, frontier_gather.install_head_probe, [
            ("    assert group._fp8_cache_layout is None and group._fp8_cache is None\n", ""),
            ("    assert not group._direct_expert_dw_requested\n", ""),
        ])
        gather = install_probe(head, alias)
        allocator = torch.cuda.memory._snapshot()["allocator_settings"]["expandable_segments"]
        assert allocator is True
        return head

    def capture(tensors, destination, *args, **kwargs):
        if destination != digest:
            return native_save(tensors, destination, *args, **kwargs)
        digest.write_text(json.dumps(save_digest(tensors), indent=2) + "\n")
        if fp64:
            assert len(numerical_inputs) == chunks
            reference = torch.zeros_like(tensors["dw"], dtype=torch.float64, device="cuda")
            for grad, hidden in numerical_inputs:
                reference.addmm_(grad.double().T, hidden.double())
            error_metrics = fp64_error(tensors["dw"].cuda(), reference)
            output.with_suffix(".fp64.json").write_text(json.dumps(error_metrics, indent=2) + "\n")

    replay.make_head = install_head
    torch.save = capture
    replay._benchmark_capture_memory = capture_memory
    replay._benchmark_note_times = note_times
    replay._benchmark_begin_diagnostic = begin_diagnostic
    replay.main = overlay(replay, replay.main, [
        ("expected_stage = VARIANTS[args.variant][2] and (args.writer != \"native\" or args.chunks == 2)",
         "expected_stage = True"),
        ('kwargs.get("_staged_head_dw")', 'kwargs.get("staged_head_dw")'),
        ('assert routes == {"folded": args.chunks, "ce": 0}, routes',
         'assert routes == {"folded": 0, "ce": 0}, routes'),
        ('"ce_route_calls": routes,', '"ce_route_calls": {"folded_inplace": args.chunks},'),
        ("        if args.dump is not None:\n            result = run(reset_weight=True)",
         "        if args.dump is not None:\n            _benchmark_begin_diagnostic()\n"
         "            result = run(reset_weight=True)"),
        ("            torch.cuda.synchronize()\n"
         "            elapsed.append((time.perf_counter() - t0) * 1000)\n"
         "            total.append(device_memory_usage(torch.cuda.memory_stats())[1])",
         "            t_submitted = time.perf_counter()\n"
         "            torch.cuda.synchronize()\n"
         "            t_complete = time.perf_counter()\n"
         "            elapsed.append((t_complete - t0) * 1000)\n"
         "            _benchmark_note_times(t_submitted - t0, t_complete - t_submitted)\n"
         "            total.append(_benchmark_capture_memory(torch.cuda.memory_stats(), device_memory_usage))"),
    ])
    digest = output.with_suffix(".digest.json")
    sys.argv[1:] = [
        "current", str(output), "--chunks", str(chunks), "--sum-reduction",
        "--warmup", str(warmup), "--samples", str(samples), "--dump", str(digest),
    ]
    replay.main()
    result = json.loads(output.read_text())
    del result["writer"]
    invocations = 1 + warmup + 1 + samples + 1
    assert beta == {0: invocations, 1: invocations * (chunks - 1)}, beta
    assert routed_ce == chunks
    assert gather["gather_calls"] == gather["native_all_gather_into_tensor_calls"] == invocations + 1
    if stages:
        order = ("dw_with_pack", "dhidden") if dw_first else ("dhidden", "dw_with_pack")
        assert tuple(name for name, _, _ in diagnostic_events) == order * chunks
    result.update(
        arm=arm, dw_first=dw_first, native_inplace_ce_chunks=routed_ce,
        writer={"name": writer.name, "pair": writer.pair, "version": writer.version,
                "beta_calls": dict(beta), "source_hidden_strides": dict(kinds),
                "extra": getattr(writer, "receipt", None)},
        gather=gather, expandable_segments=allocator, memory=memory,
        host_submit_ms=host_submit, host_wait_ms=host_wait,
        source_hashes=source_receipt, torch_git=torch.version.git_version,
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        diagnostic_stage_events_ms=[
            [name, begin.elapsed_time(end)] for name, begin, end in diagnostic_events
        ],
    )
    output.write_text(json.dumps(result, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("chunks", type=int, choices=(2, 8))
    parser.add_argument("arm", choices=("control", "order", "packed", "packed-order"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--alias", action="store_true")
    parser.add_argument("--fp64", action="store_true")
    parser.add_argument("--stages", action="store_true")
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--samples", type=int, default=SAMPLES)
    args = parser.parse_args()
    hook = None
    if "packed" in args.arm:
        from packed_dw import replay_hook

        hook = replay_hook(args.plan)
    run(
        args.chunks, args.output, arm=args.arm, writer_hook=hook,
        dw_first="order" in args.arm, alias=args.alias, fp64=args.fp64,
        stages=args.stages, warmup=args.warmup, samples=args.samples,
    )


if __name__ == "__main__":
    main()
