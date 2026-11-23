"""Replay two vocabulary panels through the native main/MTP GFSDP boundary."""

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import torch
import triton
import triton.language as tl


@triton.jit
def panel_stats(
    X, Y, MAX, DEN, CHOSEN, LOSSES,
    START: tl.constexpr, COLS: tl.constexpr, VOCAB: tl.constexpr,
    FINAL: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    target = tl.load(Y + row)
    if target == -100:
        if FINAL:
            tl.store(MAX + row, 0.0)
            tl.store(DEN + row, 0.0)
            tl.store(LOSSES + row, 0.0)
        return
    tl.device_assert((target >= 0) & (target < VOCAB), "CE target outside vocabulary")
    if FINAL:
        maximum = tl.load(MAX + row)
        denominator = tl.load(DEN + row)
    else:
        maximum = float("-inf")
        denominator = 0.0
    # Since the panel boundary is aligned to BLOCK, the reduction carries
    # exactly the same FP32 state between vocabulary tiles as the full CE.
    for start in range(0, COLS, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        values = tl.load(X + row * COLS + cols, cols < COLS, other=float("-inf")).to(tl.float32)
        next_maximum = tl.maximum(maximum, tl.max(values, 0))
        denominator = denominator * tl.exp(maximum - next_maximum) + tl.sum(
            tl.exp(values - next_maximum), 0
        )
        maximum = next_maximum
    local_target = target - START
    hit = (local_target >= 0) & (local_target < COLS)
    chosen = tl.load(X + row * COLS + local_target, hit, other=0.0).to(tl.float32)
    tl.store(MAX + row, maximum)
    if FINAL:
        log_denominator = tl.log(denominator)
        chosen = tl.where(hit, chosen, tl.load(CHOSEN + row))
        tl.store(DEN + row, log_denominator)
        tl.store(LOSSES + row, -((chosen - maximum) - log_denominator))
    else:
        tl.store(DEN + row, denominator)
        tl.store(CHOSEN + row, chosen)


@triton.jit
def panel_grad(
    X, Y, MAX, LOG_DEN, MAIN_SCALE, MTP_SCALE,
    START: tl.constexpr, COLS: tl.constexpr, HALF: tl.constexpr,
    BLOCK: tl.constexpr, TILES: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offset = tl.program_id(1) * TILES * BLOCK
    target = tl.load(Y + row) - START
    maximum = tl.load(MAX + row)
    log_denominator = tl.load(LOG_DEN + row)
    scale = tl.load(tl.where(row < HALF, MAIN_SCALE, MTP_SCALE)).to(tl.float32)
    for tile in range(TILES):
        cols = offset + tile * BLOCK + tl.arange(0, BLOCK)
        values = tl.load(X + row * COLS + cols, cols < COLS, other=0.0).to(tl.float32)
        grad = (
            tl.exp((values - maximum) - log_denominator)
            - tl.where(cols == target, 1.0, 0.0)
        ) * scale
        tl.store(X + row * COLS + cols, tl.where(target == -100 - START, 0.0, grad), cols < COLS)


def project(hidden, weight, workspace):
    rows, columns = hidden.shape[0], weight.shape[0]
    tile = workspace[:rows * columns].view(rows, columns)
    torch.mm(hidden, weight.T, out=tile)
    return tile


class PanelLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, labels, lease):
        assert hidden.is_contiguous() and weight.is_contiguous() and labels.is_contiguous()
        assert hidden.shape[:-1] == labels.shape and hidden.dtype == weight.dtype == torch.bfloat16
        h, targets = hidden.flatten(0, -2), labels.reshape(-1)
        rows, vocab = h.shape[0], weight.shape[0]
        split = lease.split
        workspace = hidden.new_empty(rows * max(split, vocab - split))
        maximum, log_den, chosen, losses = [
            torch.empty(rows, device=hidden.device, dtype=torch.float32) for _ in range(4)
        ]
        for start, panel in ((0, weight[:split]), (split, weight[split:])):
            tile = project(h, panel, workspace)
            panel_stats[(rows,)](
                tile, targets, maximum, log_den, chosen, losses, start, panel.shape[0],
                vocab, start != 0, 4096, num_warps=4, enable_fp_fusion=False, debug=True,
            )
        ctx.save_for_backward(hidden, weight, targets, maximum, log_den)
        ctx.workspace = workspace
        ctx.lease = lease
        return tuple(part.sum() for part in losses.chunk(2))

    @staticmethod
    def backward(ctx, main_scale, mtp_scale):
        hidden, weight, labels, maximum, log_den = ctx.saved_tensors
        workspace, ctx.workspace = ctx.workspace, None
        lease = ctx.lease
        h = hidden.flatten(0, -2)
        rows, vocab = h.shape[0], weight.shape[0]
        split = lease.split
        dx = torch.empty(h.shape, device=h.device, dtype=torch.float32)
        lease.acquire_chunk()
        # The second panel is still in the workspace: consume it before replaying the first.
        for start, stop in ((split, vocab), (0, split)):
            panel = weight[start:stop]
            columns = stop - start
            grad = (
                workspace[:rows * columns].view(rows, columns)
                if start else project(h, panel, workspace)
            )
            block = 4096
            tiles = triton.cdiv(columns, block * lease.grad_groups)
            panel_grad[(rows, triton.cdiv(columns, block * tiles))](
                grad, labels, maximum, log_den, main_scale, mtp_scale,
                start, columns, rows // 2, block, tiles, num_warps=4, enable_fp_fusion=False,
            )
            if start:
                torch.mm(grad, panel, out_dtype=torch.float32, out=dx)
            else:
                torch.addmm(dx, grad, panel, out_dtype=torch.float32, out=dx)
            lease.writer(grad, h, lease.slot[start:stop], lease.writes != 0)
        lease.writes += 1
        del grad, workspace
        return dx.to(hidden.dtype).view_as(hidden), None, None, None


def replace_method(method, namespace, anchor, replacement):
    source = textwrap.dedent(inspect.getsource(method))
    assert source.count(anchor) == 1
    updated = source.replace(anchor, replacement)
    result = {}
    exec(compile(updated, __file__, "exec"), namespace, result)
    return result[method.__name__]


def install_panel_route(split, grad_groups):
    import ce_storage
    import direct_dw
    from torchtitan.components import loss
    from torchtitan.experiments.four_fusion import stable_ce

    class PanelLease(direct_dw.DirectHeadDw):
        def __init__(self, lease):
            super().__init__(lease, lease.num_chunks, lease.writer)
            self.split = split
            self.grad_groups = grad_groups

        def linear(self, hidden, weight):
            assert weight is self.param.unsharded_param
            assert 0 < self.split < weight.shape[0] and self.split % 4096 == 0
            labels, self.labels = self.labels, None
            return PanelLoss.apply(hidden, weight, labels, self)

        def acquire_chunk(self):
            group = self.group
            index = self.writes
            assert group._staged_head_dw is self and group._training_state == direct_dw.TrainingState.PRE_BACKWARD
            assert index == self.completed_callbacks < self.num_chunks
            assert group.reduce_grads == (index == self.num_chunks - 1)
            assert self.param.unsharded_param.grad is None and self.param.unsharded_accumulated_grad is None
            stream = group.device_handle.current_stream()
            if index == 0:
                assert self.context.owner is None
                self.context.wait_for_release(stream)
                self.context.owner = group
                self.stream = stream
            else:
                assert self.context.owner is group and self.stream == stream

    native_selector = direct_dw.make_selector

    def select(writer):
        base = native_selector(writer)

        def with_panels(head, chunks):
            return PanelLease(base(head, chunks))

        return with_panels

    direct_dw.make_selector = select
    native_supports_folded = stable_ce.supports_folded_cross_entropy
    stable_ce.supports_folded_cross_entropy = lambda value, labels: (
        isinstance(value, tuple) or native_supports_folded(value, labels)
    )
    native_storage = ce_storage.make

    def folded_storage(backend, reuse_logits):
        base = native_storage(backend, reuse_logits)

        def folded(logits, labels, num_heads):
            if isinstance(logits, tuple):
                assert len(logits) == num_heads == 2
                return logits
            return base(logits, labels, num_heads)

        return folded

    ce_storage.make = folded_storage
    method = loss.ChunkedCELoss._chunked_loss_and_grad
    anchor = "            logits = (\n"
    loss.ChunkedCELoss._chunked_loss_and_grad = replace_method(
        method, {**vars(loss), "_PanelLease": PanelLease}, anchor,
        "            if isinstance(staged_dw, _PanelLease):\n"
        "                assert not shape_probe and argmax_sink is None\n"
        "                staged_dw.labels = label_chunk\n"
        + anchor,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=("control", "panel"))
    parser.add_argument("chunks", type=int, choices=(2, 8))
    parser.add_argument("output", type=Path)
    parser.add_argument("--split", type=int, default=65536)
    parser.add_argument("--grad-groups", type=int, default=1)
    parser.add_argument("--gather-alias", action="store_true")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--dump", type=Path)
    args = parser.parse_args()
    assert os.environ["PYTORCH_ALLOC_CONF"] == "expandable_segments:True"
    import frontier_gather
    import frontier_lifetime
    import frontier_stack
    import replay

    replay.CURRENT = Path(os.environ.get("PANEL_GIFT_SOURCE", replay.CURRENT))
    base_make_head = replay.make_head
    writer = gather = None
    panel = args.arm == "panel"

    def make_head(weight):
        nonlocal gather
        import direct_dw
        from frontier_dw_lib import LtWriter, PlannedWriter, scratch_for

        frontier_lifetime.install_early_release()
        head = base_make_head(weight)
        gather = frontier_gather.install_head_probe(head, args.gather_alias)
        plan_path = Path(frontier_stack.__file__).with_name("frontier-dw") / f"tune-{args.chunks}.json"
        plan_bytes = plan_path.read_bytes()
        tuning = json.loads(plan_bytes)
        plan = PlannedWriter(tuning, frontier_stack.PAIRS[args.chunks])
        if panel:
            install_panel_route(args.split, args.grad_groups)
            widths = args.split, tuning["vocab"] - args.split
            writers = {
                width: LtWriter(tuning["rows_per_chunk"], width, tuning["width"])
                for width in set(widths)
            }
            scratch = scratch_for(*plan.configs)
            plans = {
                width: [engine.prepare(config, scratch) for config in plan.configs]
                for width, engine in writers.items()
            }

        class CountedWriter:
            def __init__(self):
                nonlocal writer
                writer = self
                self.calls = 0

            def __call__(self, grad, hidden, output, add):
                if panel:
                    width = grad.shape[1]
                    writers[width].apply(plans[width][add], grad, hidden, output, add)
                else:
                    plan(grad, hidden, output, add)
                self.calls += 1

            def receipt(self):
                if panel:
                    torch.cuda.synchronize()
                    for engine in writers.values():
                        engine.close()
                plan.close()
                return {
                    "engine": "cublasLtMatmul",
                    "algorithms": [config["config"] for config in plan.configs],
                    "cublasLt_version": plan.lt.version,
                    "tuning_sha256": hashlib.sha256(plan_bytes).hexdigest(),
                    "calls": self.calls,
                }

        direct_dw.CublasWriter = CountedWriter
        return head

    replay.make_head = make_head
    memory = frontier_gather.install_memory_probe()
    sys.argv[1:] = [
        "current", str(args.output), "--chunks", str(args.chunks),
        "--writer", "cublas", "--sum-reduction", "--storage", "reuse",
        "--warmup", str(args.warmup), "--samples", str(args.samples),
    ]
    if args.dump:
        # Save full outputs outside the measured region for the FP64 comparison.
        sys.argv.extend(("--dump", str(args.dump)))
    replay.main()
    result = json.loads(args.output.read_text())
    result["dw_writer"] = writer.receipt()
    result["gather_probe"] = gather
    result["memory"] = memory
    result["panel"] = {
        "active": panel,
        "split": args.split if panel else None,
        "grad_groups": args.grad_groups if panel else None,
        "projection_calls_per_microbatch": args.chunks * (3 if panel else 1),
        "implemented_dhidden": "FP32 mm + FP32 addmm, then BF16 cast" if panel else "BF16 mm",
    }
    result["source_git"] = subprocess.check_output(
        ["git", "-C", str(replay.CURRENT / "gift"), "rev-parse", "HEAD"], text=True
    ).strip()
    runs = 1 + args.warmup + 1 + args.samples + int(bool(args.dump))
    assert writer.calls == runs * args.chunks * (2 if panel else 1)
    assert gather["gather_calls"] == gather["native_all_gather_into_tensor_calls"] == runs + 1
    assert len(memory) == args.samples
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
