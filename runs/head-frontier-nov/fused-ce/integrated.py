"""Run the latest isolated LM-head stack with an optional fused CE stage."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

import frontier_stack
import fused
import replay


def install(launcher):
    from torchtitan.components.loss import MTPLoss
    import ce_storage

    original_call = MTPLoss.__call__
    active_scales = None
    scale_checks = []

    def loss_call(self, pred, labels, global_valid_tokens=None, **kwargs):
        nonlocal active_scales
        assert active_scales is None
        active_scales = fused.incoming_scales(global_valid_tokens, (1.0, *self.loss_weights), pred)
        try:
            return original_call(self, pred, labels, global_valid_tokens, **kwargs)
        finally:
            active_scales = None

    @torch.library.custom_op("head_ce_frontier::consume", mutates_args={"logits"})
    def consume(
        logits: torch.Tensor, labels: torch.Tensor, scales: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        loss = logits.new_empty((logits.shape[0],), dtype=torch.float32)
        launcher(logits, labels, loss, scales)
        main, mtp = loss.chunk(2)
        return main.sum(), mtp.sum()

    @consume.register_fake
    def fake(logits, labels, scales):
        return tuple(logits.new_empty((), dtype=torch.float32) for _ in range(2))

    class PrecomputedGradient(torch.autograd.Function):
        @staticmethod
        def forward(ctx, logits, labels, scales, compiled):
            result = compiled(logits, labels, scales)
            torch.autograd.graph.increment_version(logits)
            ctx.save_for_backward(logits, scales)
            return result

        @staticmethod
        def backward(ctx, *incoming):
            assert not torch.is_grad_enabled()
            logits, scales = ctx.saved_tensors
            if not scale_checks:
                scale_checks.append((incoming, scales))
            return logits, None, None, None

    def make(backend, reuse_logits):
        assert reuse_logits and backend == "inductor"
        compiled = torch.compile(consume, backend=backend, fullgraph=True)

        def storage(logits, labels, num_heads):
            assert logits.requires_grad and active_scales is not None and num_heads == 2
            assert logits.dtype == torch.bfloat16 and logits.is_contiguous()
            assert labels.is_contiguous()
            flat = logits.view(-1, logits.shape[-1])
            return PrecomputedGradient.apply(flat, labels.view(-1), active_scales, compiled)

        return storage

    MTPLoss.__call__ = loss_call
    ce_storage.make = make
    return scale_checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=("control", "fused"))
    parser.add_argument("chunks", type=int, choices=(2, 8))
    parser.add_argument("output", type=Path)
    parser.add_argument("--method", default="cuda-3")
    parser.add_argument("--alias-gather", action="store_true")
    args = parser.parse_args()
    native_path = str(replay.CURRENT / "gift")
    sys.path.insert(0, native_path)
    from torchtitan.tools import utils

    alloc_stats = []
    original_memory = utils.device_memory_usage
    native_head = replay.make_head
    gather = None

    def observe_memory(stats):
        alloc_stats.append({
            "allocated_bytes": stats["allocated_bytes.all.peak"],
            "reserved_bytes": stats["reserved_bytes.all.peak"],
        })
        return original_memory(stats)

    utils.device_memory_usage = observe_memory

    if args.alias_gather:
        import frontier_gather

        def make_head(weight):
            nonlocal gather
            head = native_head(weight)
            gather = frontier_gather.install_head_probe(head, True)
            return head

        replay.make_head = make_head

    launcher = None
    if args.arm == "fused":
        launcher = fused.Launcher(args.method)
        scale_checks = install(launcher)
        print(f"Activated one-read BF16 in-place centered CE with {args.method}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sys.argv[1:] = ["stack", str(args.chunks), str(args.output)]
    frontier_stack.main()
    receipt = json.loads(args.output.read_text())
    assert len(alloc_stats) == receipt["warmup"] == len(receipt["elapsed_ms"]) == 10
    assert torch.cuda.memory._snapshot()["allocator_settings"]["expandable_segments"]
    if launcher:
        assert launcher.calls == 23 * args.chunks
        assert len(scale_checks) == 1
        (incoming, want), = scale_checks
        assert torch.equal(torch.stack(incoming).float(), want), (incoming, want)
        receipt["candidate"] = {
            "method": args.method,
            "calls": launcher.calls,
            "observed_actual_incoming_fp32_scales": torch.stack(incoming).tolist(),
            "input_output_alias": True,
            "launcher_sha256": hashlib.sha256(Path(fused.__file__).read_bytes()).hexdigest(),
        }
    if gather is not None:
        assert gather["native_all_gather_into_tensor_calls"] == 24
        assert gather["input_data_ptr"] == gather["output_data_ptr"]
    receipt.update(
        arm=args.arm, gpu_allocator_samples=alloc_stats, expandable_segments=True,
        gather_mode="alias" if args.alias_gather else "regular", gather_probe=gather,
        source_loss_sha256=hashlib.sha256(
            Path(native_path, "torchtitan/components/loss.py").read_bytes()
        ).hexdigest(),
    )
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
