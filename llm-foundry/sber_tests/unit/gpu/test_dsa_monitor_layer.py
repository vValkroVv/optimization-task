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
from transformers import PretrainedConfig

from llmfoundry.models.layers import dsa_monitor
from llmfoundry.models.layers import dsa_operator
from llmfoundry.models.layers.dsa_attention import LlamaLatentSparseAttention


def _config(stage: int) -> PretrainedConfig:
    return PretrainedConfig.from_dict(
        {
            "hidden_size": 256,
            "attention_hidden_size": None,
            "num_attention_heads": 32,
            "num_key_value_heads": 32,
            "num_krope_heads": 1,
            "qk_nope_head_dim": 128,
            "qk_rope_head_dim": 64,
            "v_head_dim": 128,
            "kv_lora_rank": 512,
            "q_lora_rank": 256,
            "max_position_embeddings": 128,
            "rope_theta": 100000.0,
            "rope_origin": "custom",
            "rope_scaling": None,
            "init_device": "cuda",
            "attention_type": "LlamaLatentSparseAttention",
            "attention_bias": False,
            "attention_dropout": 0.0,
            "apply_torch_compile_to_projections": False,
            "use_custom_rotary_kernel": False,
            "apply_qk_norm": False,
            "deterministic_attention": False,
            "use_mla_scaling_factor": False,
            "gated_attention": False,
            "init_type": "deepseek",
            "norm_type": "LlamaRMSNorm",
            "rms_norm_eps": 1e-6,
            "indexer_n_heads": 64,
            "indexer_head_dim": 128,
            "indexer_topk": 128,
            "indexer_fp8_scoring": False,
            "aux_kl_loss_koeff": 1.0,
            "dsa_stage": stage,
            "dsa_sp_rebalance": True,
            "dsa_block_checkpoint": False,
        }
    )


@pytest.mark.unit
@pytest.mark.requires_gpu
@pytest.mark.parametrize("stage", [1, 2])
def test_full_dsa_layer_populates_monitor_stats_for_varlen(stage):
    layer = (
        LlamaLatentSparseAttention(_config(stage), layer_idx=3)
        .cuda()
        .to(torch.bfloat16)
        .train()
    )
    layer._compute_kl_value = True
    layer._compute_attention_mass_value = True
    layer._compute_indexer_topk_value = True
    layer.attention_mass_monitor_q_start_pos = 31
    layer.attention_mass_monitor_num_queries = 4
    layer.attention_mass_monitor_topk_values = (32, 128)
    layer.indexer_topk_monitor_q_start_pos = 31
    layer.indexer_topk_monitor_num_queries = 4
    layer.indexer_topk_monitor_topk_values = (32, 128)
    hidden_states = torch.randn(
        1,
        128,
        256,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    cu_seqlens = torch.tensor([0, 64, 128], device="cuda", dtype=torch.int32)

    output, _, _ = layer(hidden_states, cu_seqlens=cu_seqlens)
    attention_stats = layer.attention_mass_monitor_stats
    indexer_stats = layer.indexer_topk_monitor_stats

    assert output.shape == hidden_states.shape
    assert indexer_stats is not None and torch.isfinite(indexer_stats).all()
    if stage == 1:
        assert attention_stats is not None and torch.isfinite(attention_stats).all()
        assert attention_stats.shape == (2, dsa_monitor.NUM_ATTENTION_MASS_STATS)
        assert (attention_stats[:, dsa_monitor.ATTENTION_MASS_MONITORED_Q_COUNT] == 4).all()
        assert (attention_stats[:, dsa_monitor.ATTENTION_MASS_ELIGIBLE_Q_COUNT] == 64).all()
        assert (attention_stats[:, dsa_monitor.ATTENTION_MASS_TOTAL_Q_COUNT] == 128).all()
    else:
        assert attention_stats is None
    assert indexer_stats.shape == (2, dsa_monitor.NUM_INDEXER_TOPK_STATS)
    assert (indexer_stats[:, dsa_monitor.INDEXER_TOPK_MONITORED_Q_COUNT] == 4).all()
    assert (indexer_stats[:, dsa_monitor.INDEXER_TOPK_ELIGIBLE_Q_COUNT] == 64).all()
    assert (indexer_stats[:, dsa_monitor.INDEXER_TOPK_TOTAL_Q_COUNT] == 128).all()
    assert (
        indexer_stats[0, dsa_monitor.POSITION_RANGE_SUM]
        <= indexer_stats[1, dsa_monitor.POSITION_RANGE_SUM]
    )
    if stage == 1:
        assert (
            0
            <= attention_stats[0, dsa_monitor.ATTENTION_MASS_SUM]
            <= attention_stats[1, dsa_monitor.ATTENTION_MASS_SUM]
            <= 4
        )
        output.float().square().mean().backward()
        assert layer.indexer.wq.weight.grad is not None


@pytest.mark.unit
@pytest.mark.requires_gpu
@pytest.mark.parametrize("stage", [1, 2])
def test_indexer_monitor_does_not_run_attention_mass(stage, monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("attention-mass path must remain disabled")

    monkeypatch.setattr(dsa_monitor, "dense_attention_mass_stats", fail)
    layer = (
        LlamaLatentSparseAttention(_config(stage), layer_idx=3)
        .cuda()
        .to(torch.bfloat16)
        .train()
    )
    layer._compute_indexer_topk_value = True
    layer.indexer_topk_monitor_q_start_pos = 31
    layer.indexer_topk_monitor_num_queries = 4
    hidden_states = torch.randn(
        1,
        128,
        256,
        device="cuda",
        dtype=torch.bfloat16,
    )
    cu_seqlens = torch.tensor([0, 64, 128], device="cuda", dtype=torch.int32)

    output, _, _ = layer(hidden_states, cu_seqlens=cu_seqlens)

    assert output.shape == hidden_states.shape
    assert layer.attention_mass_monitor_stats is None
    assert layer.indexer_topk_monitor_stats is not None


@pytest.mark.unit
@pytest.mark.requires_gpu
def test_stage1_attention_mass_does_not_run_indexer_position_stats(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("indexer positional path must remain disabled")

    monkeypatch.setattr(dsa_monitor, "dense_indexer_topk_stats", fail)
    layer = (
        LlamaLatentSparseAttention(_config(1), layer_idx=3)
        .cuda()
        .to(torch.bfloat16)
        .train()
    )
    layer._compute_attention_mass_value = True
    layer.attention_mass_monitor_q_start_pos = 31
    layer.attention_mass_monitor_num_queries = 4
    hidden_states = torch.randn(
        1,
        128,
        256,
        device="cuda",
        dtype=torch.bfloat16,
    )
    cu_seqlens = torch.tensor([0, 64, 128], device="cuda", dtype=torch.int32)

    output, _, _ = layer(hidden_states, cu_seqlens=cu_seqlens)

    assert output.shape == hidden_states.shape
    assert layer.attention_mass_monitor_stats is not None
    assert layer.indexer_topk_monitor_stats is None


@pytest.mark.unit
@pytest.mark.requires_gpu
@pytest.mark.parametrize("stage", [1, 2])
def test_disabled_monitors_do_not_call_monitor_helpers(stage, monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("disabled monitors must not run monitor helpers")

    for function in (
        "build_query_sample",
        "dense_attention_mass_stats",
        "dense_indexer_topk_stats",
        "sparse_indexer_topk_stats",
    ):
        monkeypatch.setattr(dsa_monitor, function, fail)
    layer = (
        LlamaLatentSparseAttention(_config(stage), layer_idx=3)
        .cuda()
        .to(torch.bfloat16)
        .train()
    )
    hidden_states = torch.randn(
        1,
        128,
        256,
        device="cuda",
        dtype=torch.bfloat16,
    )
    cu_seqlens = torch.tensor([0, 64, 128], device="cuda", dtype=torch.int32)

    output, _, _ = layer(hidden_states, cu_seqlens=cu_seqlens)

    assert output.shape == hidden_states.shape
    assert layer.attention_mass_monitor_stats is None
    assert layer.indexer_topk_monitor_stats is None


@pytest.mark.unit
@pytest.mark.requires_gpu
@pytest.mark.parametrize("stage", [1, 2])
def test_monitor_stats_accumulate_across_microbatches(stage):
    """Gradient accumulation runs several forwards per optimizer step; the raw sums/counts must add
    up (the callback resets them at batch_start), not reflect only the last microbatch."""
    layer = (
        LlamaLatentSparseAttention(_config(stage), layer_idx=3)
        .cuda()
        .to(torch.bfloat16)
        .train()
    )
    layer._compute_indexer_topk_value = True
    layer.indexer_topk_monitor_num_queries = 8
    if stage == 1:
        layer._compute_attention_mass_value = True
        layer.attention_mass_monitor_num_queries = 8
    hidden_states = torch.randn(1, 128, 256, device="cuda", dtype=torch.bfloat16)
    cu_seqlens = torch.tensor([0, 48, 128], device="cuda", dtype=torch.int32)
    with torch.no_grad():
        layer(hidden_states, cu_seqlens=cu_seqlens)
        after_one = layer.indexer_topk_monitor_stats.clone()
        mass_one = layer.attention_mass_monitor_stats.clone() if stage == 1 else None
        layer(hidden_states * 1.1, cu_seqlens=cu_seqlens)
    after_two = layer.indexer_topk_monitor_stats
    q = dsa_monitor.INDEXER_TOPK_MONITORED_Q_COUNT
    assert after_two[0, q] == 2 * after_one[0, q]
    assert after_two[0, dsa_monitor.INDEXER_TOPK_ELIGIBLE_Q_COUNT] == 2 * after_one[0, dsa_monitor.INDEXER_TOPK_ELIGIBLE_Q_COUNT]
    if stage == 1:
        m = dsa_monitor.ATTENTION_MASS_MONITORED_Q_COUNT
        assert layer.attention_mass_monitor_stats[0, m] == 2 * mass_one[0, m]
    # the callback's reset (None) starts a fresh accumulation
    layer.indexer_topk_monitor_stats = None
    with torch.no_grad():
        layer(hidden_states, cu_seqlens=cu_seqlens)
    assert layer.indexer_topk_monitor_stats[0, q] == after_one[0, q]
