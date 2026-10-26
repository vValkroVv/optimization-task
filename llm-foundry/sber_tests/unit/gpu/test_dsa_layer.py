"""LlamaLatentSparseAttention layer-level contracts, both stages.

  * packed == per-document: DSA attention is document-block (no cross-doc attention,
    per-doc RoPE positions), so a packed varlen forward must equal running each doc
    separately and concatenating (exercises the whole cu_seqlens plumbing).
  * train-mode 2-batch: batch 2 after batch 1 (changed doc layout, same shape) must
    equal a fresh layer -- guards the stale rotary/cache class of bugs that eval-only
    tests miss (eval recomputes caches every call).
  * checkpoint recompute: gradients under composer's activation checkpoint must match
    the non-checkpointed run -- proves the selection replay cache hands the recompute
    the SAME top-k ids the real forward used (tl_topk tie-breaks are nondeterministic,
    so a re-scan instead of a replay could silently diverge).
  * stage-1 dense-backward cache leak: cuDNN's dense_indexer_backward_wrapper caches a
    [n_docs, max_doc, max_doc] workspace per varlen shape; the operator must clear it
    after every backward or varlen stage-1 grows ~1GB/batch until OOM (512-GPU incident).
"""
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
from llmfoundry.models.layers import dsa_operator as OP
from llmfoundry.models.layers import dsa_attention

from llmfoundry.models.giga_mix.configuration_giga_mix import GigaMixConfig  # noqa: E402

DEV = "cuda"


def make_layer(stage, train=False, **overrides):
    cfg = GigaMixConfig(
        hidden_size=2048, num_attention_heads=64, num_key_value_heads=64, num_krope_heads=1,
        qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128, kv_lora_rank=512,
        q_lora_rank=1536, max_position_embeddings=8192, rope_theta=10000.0,
        norm_type="LlamaRMSNorm", attention_type="LlamaLatentSparseAttention",
        init_device=DEV, indexer_n_heads=64, indexer_head_dim=128, indexer_topk=128,
        aux_kl_loss_koeff=0.01, dsa_stage=stage, **overrides,
    )
    layer = dsa_attention.LlamaLatentSparseAttention(cfg, layer_idx=0).to(DEV)
    return layer.train() if train else layer.eval()


def packed_inputs(docs, hidden_size, seed=7):
    s = docs[-1]
    h = torch.randn(1, s, hidden_size, device=DEV, dtype=torch.bfloat16,
                    generator=torch.Generator(DEV).manual_seed(seed))
    pos = torch.cat([torch.arange(b - a, device=DEV)
                     for a, b in zip(docs[:-1], docs[1:])])[None]
    cu = {"cu_seqlens_q": torch.tensor(docs, device=DEV, dtype=torch.int32)}
    return h, pos, cu


def rel_err(got, ref):
    got, ref = got.float().reshape(-1), ref.float().reshape(-1)
    return ((got - ref).norm() / ref.norm().clamp_min(1e-6)).item()


class TestPackedEqualsPerDocument:
    @pytest.mark.unit
    @pytest.mark.requires_gpu
    @pytest.mark.parametrize("stage", [1, 2])
    def test_varlen_packed_matches_per_doc(self, stage):
        torch.manual_seed(0)
        layer = make_layer(stage)
        docs = (0, 200, 500, 768)
        h, pos, cu = packed_inputs(docs, layer.hidden_size)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out_packed = layer(h, position_ids=pos, cu_seqlens=cu)[0]
            outs = [layer(h[:, a:b], position_ids=torch.arange(b - a, device=DEV)[None],
                          cu_seqlens=None)[0]
                    for a, b in zip(docs[:-1], docs[1:])]
        out_ref = torch.cat(outs, dim=1)
        assert torch.isfinite(out_packed).all()
        got = out_packed[0].float().flatten()
        ref = out_ref[0].float().flatten()
        cos = torch.nn.functional.cosine_similarity(got, ref, dim=0).item()
        assert cos > 0.999, f"stage {stage}: packed != per-doc (cos={cos:.5f})"


class TestTrainModeCache:
    @pytest.mark.unit
    @pytest.mark.requires_gpu
    @pytest.mark.parametrize("stage", [1, 2])
    def test_batch2_after_batch1_equals_fresh(self, stage):
        docs_a, docs_b = (0, 200, 500, 768), (0, 100, 350, 768)
        torch.manual_seed(0)
        stale = make_layer(stage, train=True)
        torch.manual_seed(0)
        fresh = make_layer(stage, train=True)
        h, pos_a, cu_a = packed_inputs(docs_a, stale.hidden_size)
        _, pos_b, cu_b = packed_inputs(docs_b, stale.hidden_size)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            stale(h, position_ids=pos_a, cu_seqlens=cu_a)          # batch 1 warms caches
            got = stale(h, position_ids=pos_b, cu_seqlens=cu_b)[0]
            ref = fresh(h, position_ids=pos_b, cu_seqlens=cu_b)[0]
        # Unsorted fused ids can change gather order and bf16 summation across runs.
        tol = 2e-3 if stage == 2 else 1e-6
        assert rel_err(got, ref) < tol, "stale per-batch cache leaked into batch 2"


class TestCheckpointRecompute:
    @pytest.mark.unit
    @pytest.mark.requires_gpu
    @pytest.mark.parametrize("offload", [False, True], ids=["cache_gpu", "cache_offload"])
    def test_stage2_grads_match_under_activation_checkpoint(self, offload):
        """Composer module-checkpoint recompute must replay the forward's selection ids,
        with the cache in both placements (dsa_sel_cache_offload false/true)."""
        from composer.trainer import torch_utils_checkpoint as ckpt_mod
        docs = (0, 200, 500, 768)
        stored, replayed = [], []

        def run(use_ckpt):
            torch.manual_seed(0)
            layer = make_layer(2, train=True)
            layer.sel_cache_offload = offload
            if use_ckpt:
                store, replay = layer._sel_store, layer._sel_replay

                def store_hook(ids, ln):
                    stored.append((ids.clone(), ln.clone()))
                    return store(ids, ln)

                def replay_hook(device):
                    out = replay(device)
                    replayed.append(out)
                    return out

                layer._sel_store, layer._sel_replay = store_hook, replay_hook
            h, pos, cu = packed_inputs(docs, layer.hidden_size)
            h = h.clone().requires_grad_(True)

            def fwd(x):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    return layer(x, position_ids=pos, cu_seqlens=cu)[0]

            out = (ckpt_mod.checkpoint(fwd, h, use_reentrant=False) if use_ckpt else fwd(h))
            out.float().square().mean().backward()
            return out.detach(), h.grad.detach().clone()

        out_ref, hg_ref = run(use_ckpt=False)
        out_ck, hg_ck = run(use_ckpt=True)

        assert replayed, "recompute never hit the replay cache"
        assert len(stored) == 1, "checkpoint recompute must not store a new selection"
        for ids, ln in replayed:
            assert torch.equal(ids, stored[0][0]) and torch.equal(ln, stored[0][1]), \
                "replay differs from the stored forward selection"

        # Independent selections can differ in tie-breaking and gather order.
        out_tol, grad_tol = (2e-3, 5e-3)
        assert rel_err(out_ck, out_ref) < out_tol
        assert rel_err(hg_ck, hg_ref) < grad_tol, "recompute diverged from forward selection"


class TestDenseBackwardCacheLeak:
    @pytest.mark.unit
    @pytest.mark.requires_gpu
    def test_stage1_varlen_backward_releases_shape_cache(self):
        """Three different varlen layouts: without the post-backward cleanup the cuDNN
        cache holds one quadratic workspace PER layout (1 -> 2 -> 3, the leak); with it
        the cache is empty after every backward."""
        cache = OP._dense_indexer_backward_cache
        assert cache is not None, "cuDNN dense-backward cache handle not found"
        cache.clear()
        torch.manual_seed(0)
        layer = make_layer(1, train=True)
        for docs in ((0, 96, 256), (0, 120, 200, 256), (0, 64, 160, 256)):
            h, pos, cu = packed_inputs(docs, layer.hidden_size)
            h = h.clone().requires_grad_(True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = layer(h, position_ids=pos, cu_seqlens=cu)[0]
            out.float().square().mean().backward()
            assert len(cache) == 0, (
                f"dense-backward cache not cleared after backward (len={len(cache)}); "
                "varlen stage-1 would leak one [n_docs, max_doc, max_doc] tensor per batch")


class TestLeftPaddedEvalRejected:
    """Left padding shifts position_ids/cu_seqlens, which DSA uses for BOTH doc
    boundaries and RoPE. MLA compensates inside rope; DSA cannot, so the flag must
    fail loudly instead of producing silently wrong attention."""

    @pytest.mark.unit
    @pytest.mark.requires_gpu
    @pytest.mark.parametrize("stage", [1, 2])
    def test_left_padded_eval_raises(self, stage):
        torch.manual_seed(0)
        layer = make_layer(stage)
        docs = (0, 200, 500, 768)
        h, pos, cu = packed_inputs(docs, layer.hidden_size)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            with pytest.raises(AssertionError, match="left-padded"):
                layer(h, position_ids=pos, cu_seqlens=cu, is_left_padded_eval=True)
            # the default path is untouched
            layer(h, position_ids=pos, cu_seqlens=cu, is_left_padded_eval=False)


class TestStage1BlockCheckpoint:
    @pytest.mark.unit
    @pytest.mark.requires_gpu
    def test_block_checkpoint_matches_plain_forward(self):
        """dsa_block_checkpoint=True re-runs the dense block on backward; outputs and indexer/input grads must match."""
        results = {}
        for flag in (False, True):
            torch.manual_seed(0)          # identical weights on both sides
            layer = make_layer(1, train=True, dsa_block_checkpoint=flag)
            h, pos, cu = packed_inputs((0, 512), layer.hidden_size)
            h = h.clone().requires_grad_(True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = layer(h, position_ids=pos, cu_seqlens=cu)[0]
            out.float().square().mean().backward()
            results[flag] = (out.detach(), h.grad.detach().clone(),
                             torch.cat([p.grad.flatten() for n, p in layer.named_parameters()
                                        if n.startswith("indexer.") and p.grad is not None]))
        for got, ref, name in zip(results[True], results[False], ("out", "h.grad", "indexer grads")):
            assert rel_err(got, ref) < 1e-2, f"{name}: block-checkpointed stage 1 diverged ({rel_err(got, ref):.3g})"
