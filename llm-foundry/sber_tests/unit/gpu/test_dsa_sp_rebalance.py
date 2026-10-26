"""DSA sequence-parallel block-cyclic rebalance equivalence (2-GPU, mp.spawn).

The rebalanced SP path (block-cyclic query redistribution via all-to-all) must produce
the same layer output, input gradient, and parameter gradients as the plain contiguous
SP path on the same world: rebalance is a pure load-balancing permutation and must be
mathematically transparent. Checked for stage 1 (dense warm-up + dense KL) and stage 2
(fused top-k + sparse attention + top-k KL), non-varlen and packed varlen, with and
without gated attention. The contiguous SP path is additionally compared with the same
layer run on the full sequence WITHOUT SP (output / input-grad chunks, parameter grads).

Slim CI version of the 8-GPU prod_dsa_kernels/test_dsa_rebalance.py gate (world=2).
"""
import os

import pytest

import torch
if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)
if torch.cuda.get_device_capability() != (9, 0):
    pytest.skip("the DSA kernels are Hopper-only (sm90)", allow_module_level=True)
pytest.importorskip(
    "llmfoundry.models.layers.dsa_operator",
    reason="the DSA kernel packages are not installed in this image",
)
import llmfoundry.models.layers.dsa_operator  # noqa: F401

import torch.distributed as dist  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402

WORLD = 2
REL_TOL, COS_TOL = 2e-2, 0.9999          # bf16 reduction-reorder tolerance
DOCS = (700, 900, 448)                   # varlen packing for S=2048


def _worker(rank, world_size, stage, varlen, port, gated=False, heads=64, seq_len=2048):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = port
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    try:
        import composer.utils.dist as cdist
        grp = dist.new_group(list(range(world_size)))
        cdist.get_sp_group = lambda: grp              # the SP group the DSA layer collectives use
        from llmfoundry.models.giga_mix.configuration_giga_mix import GigaMixConfig
        from llmfoundry.models.layers.dsa_attention import LlamaLatentSparseAttention

        dev = f"cuda:{rank}"
        S, topk = seq_len, 128
        torch.manual_seed(0)                          # identical replicated params on every rank
        cfg = GigaMixConfig(
            hidden_size=2048, num_attention_heads=heads, num_key_value_heads=heads,
            num_krope_heads=1, qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128,
            kv_lora_rank=512, q_lora_rank=1536, max_position_embeddings=1 << 16,
            rope_theta=10000.0, norm_type="LlamaRMSNorm",
            attention_type="LlamaLatentSparseAttention", init_device=dev,
            indexer_n_heads=64, indexer_head_dim=128, indexer_topk=topk,
            aux_kl_loss_koeff=0.5, dsa_stage=stage, gated_attention=gated)
        layer = LlamaLatentSparseAttention(cfg, layer_idx=0).to(dev).train()
        layer.sp_size, layer.is_ring_attn = world_size, False
        torch.manual_seed(0)                          # bit-identical twin that runs WITHOUT SP
        ref_layer = LlamaLatentSparseAttention(cfg, layer_idx=0).to(dev).train()
        ref_layer.sp_size, ref_layer.is_ring_attn = 1, False
        for a, b in zip(layer.parameters(), ref_layer.parameters()):
            assert torch.equal(a, b)

        s_loc = S // world_size
        torch.manual_seed(1234)                       # identical GLOBAL input; rank takes its chunk
        h_global = torch.randn(1, S, cfg.hidden_size, device=dev, dtype=torch.bfloat16)
        h = h_global[:, rank * s_loc:(rank + 1) * s_loc].clone().requires_grad_(True)

        cu = position_ids = None
        if varlen:
            bounds = [0]
            for d in (*DOCS[:-1], S - sum(DOCS[:-1])):
                bounds.append(bounds[-1] + d)
            assert bounds[-1] == S
            cu = {"cu_seqlens_q": torch.tensor(bounds, device=dev, dtype=torch.int32)}
            position_ids = torch.cat([torch.arange(b - a, device=dev)
                                      for a, b in zip(bounds[:-1], bounds[1:])]).unsqueeze(0)

        def run_once(rebalance):
            layer.sp_rebalance = rebalance
            layer.zero_grad(set_to_none=True)
            h.grad = None
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = layer(h, position_ids=position_ids, cu_seqlens=cu)[0]
            out.float().sum().backward()
            pg = torch.cat([p.grad.float().flatten()
                            for p in layer.parameters() if p.grad is not None])
            return out.detach().clone(), h.grad.detach().clone(), pg

        o_ref, hg_ref, pg_ref = run_once(rebalance=False)
        o_reb, hg_reb, pg_reb = run_once(rebalance=True)
        dist.all_reduce(pg_ref)                       # true SP-summed parameter gradient
        dist.all_reduce(pg_reb)

        def check(name, a, b):
            a, b = a.float(), b.float()
            rel = ((a - b).abs().mean() / b.abs().mean().clamp_min(1e-12)).item()
            cos = torch.nn.functional.cosine_similarity(
                a.flatten(), b.flatten(), dim=0).item()
            assert rel < REL_TOL and cos > COS_TOL, (
                f"rank{rank} stage{stage} varlen={varlen} gated={gated} {name}: rel={rel:.4g} cos={cos:.6f}")

        check("output", o_reb, o_ref)
        check("h.grad", hg_reb, hg_ref)
        check("param.grad", pg_reb, pg_ref)

        # --- SP (contiguous) vs the SAME layer on the whole sequence without SP: the SP path must be
        # a pure partition of the single-GPU computation. Parameter grads: the LM loss is a plain
        # sum over all tokens on both sides, but the KL aux term is normalised by the LOCAL S_loc
        # (so that the world-wide gradient average, which includes the SP ranks, yields coeff/S):
        # SP-summed indexer grads are therefore sp_size x the single-GPU ones. Pinning that factor
        # here documents the convention.
        h_full = h_global.clone().requires_grad_(True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out_full = ref_layer(h_full, position_ids=position_ids, cu_seqlens=cu)[0]
        out_full.float().sum().backward()
        chunk = slice(rank * s_loc, (rank + 1) * s_loc)
        check("output vs single-GPU", o_ref, out_full[:, chunk].detach())
        check("h.grad vs single-GPU", hg_ref, h_full.grad[:, chunk].detach())
        layer.zero_grad(set_to_none=True); h.grad = None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = layer(h, position_ids=position_ids, cu_seqlens=cu)[0]
        out.float().sum().backward()
        for n, p in layer.named_parameters():
            if p.grad is None:
                continue
            g_sp = p.grad.float().clone()
            dist.all_reduce(g_sp)
            g_full = dict(ref_layer.named_parameters())[n].grad.float()
            factor = world_size if n.startswith("indexer.") else 1.0
            check(f"{n}.grad vs single-GPU (x{factor:g})", g_sp, factor * g_full)
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


class TestSPRebalanceEquivalence:
    @pytest.mark.unit
    @pytest.mark.requires_gpu
    @pytest.mark.slow
    @pytest.mark.world_size(WORLD)
    @pytest.mark.parametrize("heads", [32, 64])
    def test_odd_block_length(self, heads):
        """2052 / SP2 / 2 blocks = 513 lengths: the second view is unaligned."""
        if torch.cuda.device_count() < WORLD:
            pytest.skip(f"requires {WORLD} GPUs, found {torch.cuda.device_count()}")
        mp.spawn(_worker, args=(WORLD, 2, True, "12367", False, heads, 2052),
                 nprocs=WORLD, join=True)

    @pytest.mark.unit
    @pytest.mark.requires_gpu
    @pytest.mark.slow
    @pytest.mark.world_size(WORLD)
    @pytest.mark.parametrize("stage,varlen,port,gated", [
        (1, False, "12361", False),
        (2, False, "12362", False),
        (1, True, "12363", False),
        (2, True, "12364", False),
        (2, False, "12365", True),    # gated attention: un-rebalance happens BEFORE the gate
        (1, True, "12366", True),
    ])
    @pytest.mark.parametrize("heads", [32, 64])
    def test_rebalance_matches_contiguous_sp(self, stage, varlen, port, gated, heads):
        if torch.cuda.device_count() < WORLD:
            pytest.skip(f"requires {WORLD} GPUs, found {torch.cuda.device_count()}")
        os.environ.setdefault("DG_CACHE_FORCE_SKIP", "1")
        mp.spawn(_worker, args=(WORLD, stage, varlen, port, gated, heads), nprocs=WORLD, join=True)
