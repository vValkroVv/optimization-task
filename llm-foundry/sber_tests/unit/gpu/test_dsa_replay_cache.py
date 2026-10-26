"""Selection replay, pinned-buffer reuse and cache lifetime during training."""
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
from composer.trainer import torch_utils_checkpoint as ckpt_mod  # noqa: E402

from llmfoundry.models.giga_mix.configuration_giga_mix import GigaMixConfig  # noqa: E402
from llmfoundry.models.layers import dsa_attention, dsa_operator  # noqa: E402

DEV = "cuda"


def make_layer(stage, **overrides):
    torch.manual_seed(0)
    kw = dict(
        hidden_size=1024, num_attention_heads=64, num_key_value_heads=64, num_krope_heads=1,
        qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128, kv_lora_rank=512,
        q_lora_rank=512, max_position_embeddings=8192, rope_theta=10000.0,
        norm_type="LlamaRMSNorm", attention_type="LlamaLatentSparseAttention",
        init_device=DEV, indexer_n_heads=64, indexer_head_dim=128, indexer_topk=128,
        aux_kl_loss_koeff=0.05, dsa_stage=stage,
    )
    kw.update(overrides)
    return dsa_attention.LlamaLatentSparseAttention(GigaMixConfig(**kw), layer_idx=0).to(DEV).train()


def inputs(s, hidden, seed=1):
    g = torch.Generator(DEV).manual_seed(seed)
    h = torch.randn(1, s, hidden, device=DEV, dtype=torch.bfloat16, generator=g).requires_grad_(True)
    return h, torch.arange(s, device=DEV)[None]


def rel(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return ((a - b).norm() / b.norm().clamp_min(1e-9)).item()


class _ScanCounter:
    def __init__(self, monkeypatch):
        self.calls = 0
        orig = dsa_operator.select_topk

        def counting(*a, **k):
            self.calls += 1
            return orig(*a, **k)
        monkeypatch.setattr(dsa_operator, "select_topk", counting)


def _ckpt_fwd_bwd(layer, h, pos):
    def fwd(x):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return layer(x, position_ids=pos)[0]
    out = ckpt_mod.checkpoint(fwd, h, use_reentrant=False)
    out.float().square().mean().backward()
    return out


class TestReplayCache:
    @pytest.mark.unit
    @pytest.mark.requires_gpu
    def test_recompute_replays_instead_of_rescanning_gpu_cache(self, monkeypatch):
        layer = make_layer(2)                       # default: selection stays on GPU
        h, pos = inputs(512, layer.hidden_size)
        counter = _ScanCounter(monkeypatch)
        _ckpt_fwd_bwd(layer, h, pos)
        assert counter.calls == 1, f"top-k scan ran {counter.calls}x: the recompute must replay the cached ids"
        ids, ln = layer._sel_cache
        assert ids.device.type == "cuda" and ln.device.type == "cuda"

    @pytest.mark.unit
    @pytest.mark.requires_gpu
    def test_recompute_replays_instead_of_rescanning_offload(self, monkeypatch):
        layer = make_layer(2, dsa_sel_cache_offload=True)
        h, pos = inputs(512, layer.hidden_size)
        counter = _ScanCounter(monkeypatch)
        _ckpt_fwd_bwd(layer, h, pos)
        assert counter.calls == 1, f"top-k scan ran {counter.calls}x: the recompute must replay the cached ids"
        ids_cpu, len_cpu, _ = layer._sel_cache
        assert ids_cpu.is_pinned() and len_cpu.is_pinned()

    @pytest.mark.unit
    @pytest.mark.requires_gpu
    def test_buffers_are_reused_and_resized(self):
        layer = make_layer(2, dsa_sel_cache_offload=True)
        h, pos = inputs(512, layer.hidden_size)
        _ckpt_fwd_bwd(layer, h, pos)
        ptr = layer._sel_cache[0].data_ptr()
        h, pos = inputs(512, layer.hidden_size, seed=2)
        _ckpt_fwd_bwd(layer, h, pos)
        assert layer._sel_cache[0].data_ptr() == ptr, "same shape must reuse the pinned buffer"
        h, pos = inputs(768, layer.hidden_size, seed=3)          # longer sequence -> new buffers
        _ckpt_fwd_bwd(layer, h, pos)
        assert layer._sel_cache[0].shape[1] == 768

    @pytest.mark.unit
    @pytest.mark.requires_gpu
    @pytest.mark.parametrize("offload", [False, True], ids=["cache_gpu", "cache_offload"])
    def test_eval_and_no_grad_do_not_populate_cache(self, offload):
        layer = make_layer(2, dsa_sel_cache_offload=offload)
        h, pos = inputs(256, layer.hidden_size)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            layer(h, position_ids=pos)
        layer.eval()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            layer(h, position_ids=pos)
        assert layer._sel_cache is None
