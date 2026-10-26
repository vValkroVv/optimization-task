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

from llmfoundry.models.layers import dsa_monitor, dsa_operator


@pytest.mark.unit
@pytest.mark.requires_gpu
def test_stage1_monitor_preserves_indexer_backward():
    device = torch.device("cuda")
    seq_len = 128
    index_heads = 64
    index_dim = 128
    attention_heads = 32
    attention_dim = 576
    attention_scale = attention_dim**-0.5
    index_scale = index_dim**-0.5

    index_q = (
        torch.randn(
            1,
            seq_len,
            index_heads,
            index_dim,
            device=device,
            dtype=torch.bfloat16,
        )
        .mul_(0.02)
        .requires_grad_()
    )
    index_k = (
        torch.randn(
            1,
            seq_len,
            1,
            index_dim,
            device=device,
            dtype=torch.bfloat16,
        )
        .mul_(0.02)
        .requires_grad_()
    )
    weights = (
        torch.randn(
            1,
            seq_len,
            index_heads,
            device=device,
            dtype=torch.bfloat16,
        )
        .mul_(0.02)
        .requires_grad_()
    )
    q_attn = torch.randn(
        1,
        seq_len,
        attention_heads,
        attention_dim,
        device=device,
        dtype=torch.bfloat16,
    ).mul_(0.02)
    kv = torch.randn(
        1,
        seq_len,
        attention_dim,
        device=device,
        dtype=torch.bfloat16,
    ).mul_(0.02)

    logits = torch.einsum("bqhd,bkd->bqhk", q_attn, kv).float()
    logits *= attention_scale
    causal = (
        torch.arange(seq_len, device=device)[None, :]
        <= torch.arange(
            seq_len,
            device=device,
        )[:, None]
    )
    logits.masked_fill_(~causal[None, :, None, :], -torch.inf)
    lse = logits.logsumexp(-1)
    monitor_rows = torch.tensor([63, 127], device=device)

    kl_loss, attention_stats, indexer_stats = dsa_operator._DSAWarmupFunction.apply(
        index_q,
        index_k,
        weights,
        q_attn.contiguous(),
        kv.contiguous(),
        lse.contiguous(),
        attention_scale,
        index_scale,
        0,
        True,
        monitor_rows,
        monitor_rows,
        (8, 16),
        (8, 16),
    )

    assert torch.isfinite(kl_loss)
    assert torch.isfinite(attention_stats).all()
    assert torch.isfinite(indexer_stats).all()
    assert attention_stats.shape == (2, dsa_monitor.NUM_ATTENTION_MASS_STATS)
    assert (attention_stats[:, dsa_monitor.ATTENTION_MASS_MONITORED_Q_COUNT] == 2).all()
    assert indexer_stats.shape == (2, dsa_monitor.NUM_INDEXER_TOPK_STATS)
    assert (indexer_stats[:, dsa_monitor.INDEXER_TOPK_MONITORED_Q_COUNT] == 2).all()
    assert (
        0
        <= attention_stats[0, dsa_monitor.ATTENTION_MASS_SUM]
        <= attention_stats[1, dsa_monitor.ATTENTION_MASS_SUM]
        <= 2
    )
    kl_loss.backward()
    assert index_q.grad is not None and torch.isfinite(index_q.grad).all()
    assert index_k.grad is not None and torch.isfinite(index_k.grad).all()
    assert weights.grad is not None and torch.isfinite(weights.grad).all()


@pytest.mark.unit
@pytest.mark.requires_gpu
def test_stage2_indexer_monitor_consumes_fused_selector_positions():
    device = torch.device("cuda")
    seq_len = 256
    index_heads = 64
    index_dim = 128

    index_q = torch.randn(
        1,
        seq_len,
        index_heads,
        index_dim,
        device=device,
        dtype=torch.bfloat16,
    ).contiguous()
    index_k = torch.randn(
        1,
        seq_len,
        1,
        index_dim,
        device=device,
        dtype=torch.bfloat16,
    ).contiguous()
    weights = torch.randn(
        1,
        seq_len,
        index_heads,
        device=device,
        dtype=torch.bfloat16,
    ).contiguous()
    positions_q = torch.arange(seq_len, device=device)
    topk_ids, topk_length = dsa_operator.select_topk(
        index_q,
        index_k,
        weights,
        index_dim**-0.5,
        K=128,
        positions_q=positions_q,
    )

    monitor_rows = torch.tensor([127, 255], device=device)
    indexer_stats = dsa_monitor.sparse_indexer_topk_stats(
        topk_ids,
        topk_length,
        positions_q,
        monitor_rows,
        (8, 32),
        index_q=index_q,
        index_k=index_k,
        weights=weights,
        index_scale=index_dim**-0.5,
    )

    assert torch.isfinite(indexer_stats).all()
    assert indexer_stats.shape == (2, dsa_monitor.NUM_INDEXER_TOPK_STATS)
    assert (indexer_stats[:, dsa_monitor.INDEXER_TOPK_MONITORED_Q_COUNT] == 2).all()
