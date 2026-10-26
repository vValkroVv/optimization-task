"""Integration tests for GigaFSDP: buffer lifecycle, accumulation, fp8 all-gather.

No transformers dependency — the model is a plain nn.Module stack.  The existing
tests/utils harness builds a LlamaForCausalLM and currently dies inside
transformers ('tuple' object has no attribute 'last_hidden_state') on both master
and this branch, so it cannot answer whether GigaFSDP itself regressed.

Everything here is an INVARIANT test, not a comparison against an external
reference.  That avoids guessing whether reduce-scatter sums or averages, and it
is a stronger check: the final weights must not depend on how many reusable
buffers the pool has.

  * buffer_count=1 vs 2  -> must be BIT-IDENTICAL.  Different pool size means a
    different eviction order, so this is the direct test for the buffer-lifecycle
    changes on this branch (_backward_depth, conditional start_prepare).
  * fp8 buffer_count=1 vs 2 -> the one that matters most.  ctx.weights_quantized_gathered
    holds a view into a ReusableBuffer that a later layer overwrites; correctness
    depends on the backward re-gather landing at the same addresses.  With a
    single buffer every layer evicts every other one, so if the re-gather is
    wrong this diverges.
  * grad accumulation 1 vs 2 -> exercises _backward_depth across microbatches.

Run:
    cd contrib/gigafsdp
    torchrun --standalone --nproc_per_node=2 tests/test_gigafsdp_integration.py
    torchrun --standalone --nproc_per_node=8 tests/test_gigafsdp_integration.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from giga_fsdp import GigaFSDP
from giga_fsdp.float8_utils import FP8_TILE_SIZE

TILE = FP8_TILE_SIZE

E_EXPERTS = 2
HIDDEN = 256
INTER = 128
N_BLOCKS = 4
BATCH = 8


# ---------------------------------------------------------------------------
# Model — parameter FQNs must end in `experts.weight1` / `experts.weight2`
# for is_expert_param() to select them for fp8 all-gather.
# ---------------------------------------------------------------------------

def _dequant(fp8_data: torch.Tensor, scale: torch.Tensor, dtype) -> torch.Tensor:
    """[E, N, K] fp8 + [E, N/128, K/128] scales -> dense tensor."""
    E, N, K = fp8_data.shape
    x = fp8_data.float().reshape(E, N // TILE, TILE, K // TILE, TILE)
    x = x * scale.reshape(E, N // TILE, 1, K // TILE, 1)
    return x.reshape(E, N, K).to(dtype)


class _RouteGradToCarrier(torch.autograd.Function):
    """Forward the dequantized value; send the gradient to the bf16 carrier.

    The carrier holds NaN under fp8 all-gather, so it must never contribute its
    VALUE — only receive wgrad.  Returning a contiguous gradient matters:
    MetaParam.backward (meta_param.py:29) does grad_output.view(-1).
    """

    @staticmethod
    def forward(ctx, value, carrier):
        assert value.shape == carrier.shape, f"{value.shape} != {carrier.shape}"
        return value

    @staticmethod
    def backward(ctx, grad_out):
        g = grad_out.contiguous()
        return g, g


class Experts(nn.Module):
    """Stand-in for GroupedLlamaMLP: consumes the `_<name>_quantized` handoff.

    Under fp8 all-gather the bf16 weight1/weight2 are a NaN carrier, so reading
    them would produce NaN loss immediately — which is exactly what we want if
    the handoff ever fails to arrive.
    """

    def __init__(self, n_experts: int, inter: int, hidden: int):
        super().__init__()
        self.weight1 = nn.Parameter(torch.empty(n_experts, 2 * inter, hidden))
        self.weight2 = nn.Parameter(torch.empty(n_experts, hidden, inter))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q1 = getattr(self, "_weight1_quantized", None)
        q2 = getattr(self, "_weight2_quantized", None)
        managed = bool(getattr(self, "_giga_fp8_managed_params", None))

        if managed:
            assert q1 is not None and q2 is not None, (
                "fp8-managed experts but no quantized handoff — the bf16 weights "
                "are a NaN carrier and must never be read"
            )
            # Value comes from the fp8 handoff; the gradient is routed to the bf16
            # parameter.  This mirrors _GroupedGemm, which reads
            # weights_quantized_gathered for the value and returns wgrad into
            # `weights`.  NOTE: the carrier's VALUE must never enter the
            # computation — it is NaN, and `0.0 * NaN` is NaN, not 0.
            w1 = _RouteGradToCarrier.apply(_dequant(q1[0], q1[1], x.dtype), self.weight1)
            w2 = _RouteGradToCarrier.apply(_dequant(q2[0], q2[1], x.dtype), self.weight2)
        else:
            w1, w2 = self.weight1, self.weight2

        # Deliberately F.linear per expert rather than einsum: MetaParam.backward
        # (meta_param.py:29) does grad_output.view(-1), which requires a
        # CONTIGUOUS weight gradient.  einsum's backward hands back a transposed
        # view and blows up there.  That is a pre-existing GigaFSDP constraint,
        # not something this branch introduced — see the note in the MR.
        outs = []
        for e in range(w1.shape[0]):
            h = F.linear(x, w1[e])              # [B, 2I]
            gate, up = torch.chunk(h, 2, dim=-1)
            a = F.silu(gate) * up               # [B, I]
            outs.append(F.linear(a, w2[e]))     # [B, H]
        return torch.stack(outs, dim=1).mean(dim=1)


class Block(nn.Module):
    def __init__(self, n_experts, inter, hidden):
        super().__init__()
        self.experts = Experts(n_experts, inter, hidden)

    def forward(self, x):
        return x + self.experts(x)


class TinyMoE(nn.Module):
    def __init__(self, n_blocks=N_BLOCKS, n_experts=E_EXPERTS, inter=INTER, hidden=HIDDEN):
        super().__init__()
        self.blocks = nn.ModuleList(
            Block(n_experts, inter, hidden) for _ in range(n_blocks)
        )

    def forward(self, x, target):
        for blk in self.blocks:
            x = blk(x)
        return F.mse_loss(x, target), x


def build_model(device, dtype, seed=1234):
    """Identical weights on every rank."""
    g = torch.Generator().manual_seed(seed)
    model = TinyMoE()
    for p in model.parameters():
        with torch.no_grad():
            p.copy_(torch.randn(p.shape, generator=g) * 0.02)
    return model.to(device=device, dtype=dtype)


def wrap_spec(model):
    return [(blk, f"blocks.{i}") for i, blk in enumerate(model.blocks)]


# ---------------------------------------------------------------------------
# One training run under a given configuration
# ---------------------------------------------------------------------------

def run_config(device, rank, *, buffer_count, accum, zero_stage, fp8,
               n_iter=4, seed=1234, lr=0.05):
    """Train the model and return rank-0's full state_dict (or None elsewhere)."""
    dtype = torch.bfloat16
    torch.manual_seed(seed + rank)

    model = build_model(device, dtype, seed=seed)
    modules = wrap_spec(model)

    kwargs = dict(
        zero_stage=zero_stage,
        fsdp_process_group=None,
        model_parallel_process_group=None,
        param_dtype=dtype,
        gradient_accumulation_steps=accum,
        buffer_count=buffer_count,
        modules_to_wrap_with_names=modules,
        rogue_layer_norm_modules_with_names=None,
        layer_norm_module_cls=None,
        device_id=torch.cuda.current_device(),
    )
    if fp8:
        kwargs.update(
            alignment_value=128,
            float8_all_gather_on_experts=True,
            float8_all_gather_skip_patterns=[],
        )

    fsdp = GigaFSDP(model, **kwargs)
    opt = torch.optim.SGD(fsdp.parameters(), lr=lr)

    # Same data every run, different per rank (real data parallelism).
    dg = torch.Generator(device="cpu").manual_seed(999 + rank)
    micro = [
        (torch.randn(BATCH, HIDDEN, generator=dg).to(device=device, dtype=dtype),
         torch.randn(BATCH, HIDDEN, generator=dg).to(device=device, dtype=dtype))
        for _ in range(n_iter * accum)
    ]

    losses = []
    k = 0
    for _ in range(n_iter):
        opt.zero_grad(set_to_none=False)
        for _ in range(accum):
            x, y = micro[k]
            k += 1
            loss, _out = fsdp(x, y)
            assert torch.isfinite(loss), f"non-finite loss: {loss.item()}"
            losses.append(loss.item())
            loss.backward()
        fsdp.clip_grad_norm_(1.0)
        opt.step()

    sd = fsdp.state_dict()
    out = None
    if rank == 0:
        out = {k_: v.detach().float().cpu()
               for k_, v in sd.items() if isinstance(v, torch.Tensor)}

    del fsdp, model, opt
    torch.cuda.empty_cache()
    dist.barrier()
    return out, losses


def compare(a, b, label, rank, failures, exact=True, tol=2e-2):
    if rank != 0:
        return
    if a is None or b is None:
        failures.append(f"{label}: missing state_dict")
        return
    if a.keys() != b.keys():
        failures.append(f"{label}: different parameter sets")
        return
    for name in a:
        x, y = a[name], b[name]
        if exact:
            if not torch.equal(x, y):
                d = (x - y).abs().max().item()
                failures.append(f"{label}: '{name}' differs (max |d| = {d:.3e})")
                return
        else:
            if not torch.allclose(x, y, atol=tol, rtol=tol):
                d = (x - y).abs().max().item()
                failures.append(f"{label}: '{name}' diverged (max |d| = {d:.3e})")
                return


# ---------------------------------------------------------------------------

def main():
    if not torch.cuda.is_available():
        print("CUDA required")
        return 0
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", rank % torch.cuda.device_count())
    torch.cuda.set_device(device)

    failures = []

    def note(msg):
        if rank == 0:
            print(msg, flush=True)

    # --- 1. bf16: result must not depend on the buffer pool size -----------
    note("[1] bf16: buffer_count 1 vs 2 (zero_stage=3)")
    a, _ = run_config(device, rank, buffer_count=2, accum=1, zero_stage=3, fp8=False)
    b, _ = run_config(device, rank, buffer_count=1, accum=1, zero_stage=3, fp8=False)
    compare(a, b, "bf16 buffer_count 1 vs 2", rank, failures, exact=True)

    # --- 2. bf16: determinism ---------------------------------------------
    note("[2] bf16: same config twice must be bit-identical")
    c, _ = run_config(device, rank, buffer_count=2, accum=1, zero_stage=3, fp8=False)
    compare(a, c, "bf16 determinism", rank, failures, exact=True)

    # --- 3. bf16: gradient accumulation -----------------------------------
    note("[3] bf16: accum=2 runs and stays finite (exercises _backward_depth)")
    d, losses = run_config(device, rank, buffer_count=2, accum=2, zero_stage=3, fp8=False)
    if rank == 0 and not all(map(lambda v: v == v, losses)):
        failures.append("accum=2: NaN loss")

    # --- 4. bf16: zero_stage 2 --------------------------------------------
    note("[4] bf16: zero_stage=2 buffer_count 1 vs 2")
    e2, _ = run_config(device, rank, buffer_count=2, accum=1, zero_stage=2, fp8=False)
    f2, _ = run_config(device, rank, buffer_count=1, accum=1, zero_stage=2, fp8=False)
    compare(e2, f2, "bf16 zero2 buffer_count 1 vs 2", rank, failures, exact=True)

    # --- 5. fp8: the handoff arrives and training is finite ----------------
    note("[5] fp8: all-gather on experts, loss must stay finite")
    g, g_losses = run_config(device, rank, buffer_count=2, accum=1, zero_stage=3, fp8=True)
    if rank == 0 and not all(v == v for v in g_losses):
        failures.append("fp8: NaN loss — handoff missing, bf16 NaN carrier was read")

    # --- 6. fp8: buffer reuse across layers -- THE important one ----------
    note("[6] fp8: buffer_count 1 vs 2 (buffer reuse + backward re-gather)")
    h, _ = run_config(device, rank, buffer_count=1, accum=1, zero_stage=3, fp8=True)
    compare(g, h, "fp8 buffer_count 1 vs 2", rank, failures, exact=True)

    # --- 7. fp8: determinism ----------------------------------------------
    note("[7] fp8: same config twice must be bit-identical")
    i, _ = run_config(device, rank, buffer_count=2, accum=1, zero_stage=3, fp8=True)
    compare(g, i, "fp8 determinism", rank, failures, exact=True)

    # --- 8. fp8: grad accumulation ----------------------------------------
    note("[8] fp8: accum=2 (scales computed once per step, buffers reused)")
    j, j_losses = run_config(device, rank, buffer_count=2, accum=2, zero_stage=3, fp8=True)
    if rank == 0 and not all(v == v for v in j_losses):
        failures.append("fp8 accum=2: NaN loss")

    # --- 9. fp8 vs bf16 must be close, not identical ------------------------
    note("[9] fp8 vs bf16: same trajectory within fp8 error")
    compare(a, g, "fp8 vs bf16", rank, failures, exact=False, tol=5e-2)

    flag = torch.tensor([len(failures)], dtype=torch.int32, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.SUM)
    if failures:
        for f in failures:
            print(f"[rank {rank}] FAIL {f}", flush=True)
    dist.barrier()
    if rank == 0:
        print(f"\nworld_size={world}: "
              f"{'ALL PASSED' if flag.item() == 0 else f'{flag.item()} FAILURES'}",
              flush=True)
    dist.destroy_process_group()
    return 1 if flag.item() else 0


if __name__ == "__main__":
    sys.exit(main())
