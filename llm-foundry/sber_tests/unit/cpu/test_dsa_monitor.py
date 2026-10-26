import importlib.util
from pathlib import Path

import pytest
import torch

MODULE_PATH = (
    Path(__file__).parents[3] / "llmfoundry" / "models" / "layers" / "dsa_monitor.py"
)
SPEC = importlib.util.spec_from_file_location("dsa_monitor", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
dsa_monitor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dsa_monitor)


def test_build_query_sample_uses_document_local_positions():
    cu_seqlens = torch.tensor([0, 5, 9], dtype=torch.int32)

    sample, eligible_count, total_count = dsa_monitor.build_query_sample(
        seq_len=9,
        cu_seqlens=cu_seqlens,
        q_start_pos=1,
        num_queries=3,
        device=torch.device("cpu"),
    )

    assert sample.nonzero(as_tuple=True)[0].tolist() == [2, 3, 7]
    assert eligible_count == 5
    assert total_count == 9


def test_query_sample_is_partition_independent_for_sequence_parallelism():
    sample, eligible_count, _ = dsa_monitor.build_query_sample(
        seq_len=12,
        cu_seqlens=None,
        q_start_pos=1,
        num_queries=5,
        device=torch.device("cpu"),
    )
    rank_0_positions = torch.tensor([0, 1, 2, 6, 7, 8])
    rank_1_positions = torch.tensor([3, 4, 5, 9, 10, 11])

    rank_0_count = int(sample.index_select(0, rank_0_positions).sum())
    rank_1_count = int(sample.index_select(0, rank_1_positions).sum())

    assert eligible_count == 10
    assert rank_0_count + rank_1_count == 5


def test_dense_monitors_are_computed_independently():
    attn_score = torch.tensor(
        [
            [
                [0.2, 0.3, 0.5, 0.0],
                [0.1, 0.2, 0.3, 0.4],
            ],
        ]
    )
    attn_l1norm = torch.ones(1, 2)
    index_score = torch.tensor(
        [
            [
                [0.1, 0.9, 0.5, 99.0],
                [0.8, 0.1, 0.2, 0.7],
            ],
        ]
    )

    attention_stats = dsa_monitor.dense_attention_mass_stats(
        attn_score,
        attn_l1norm,
        index_score,
        topk_values=(1, 2, 8),
        offset=2,
        monitor_rows=torch.tensor([0, 1]),
    )
    indexer_stats = dsa_monitor.dense_indexer_topk_stats(
        index_score,
        topk_values=(1, 2),
        offset=2,
        monitor_rows=torch.tensor([0, 1]),
    )

    assert attention_stats.shape == (3, dsa_monitor.NUM_ATTENTION_MASS_STATS)
    assert float(attention_stats[0, dsa_monitor.ATTENTION_MASS_SUM]) == pytest.approx(
        0.4
    )
    assert float(attention_stats[1, dsa_monitor.ATTENTION_MASS_SUM]) == pytest.approx(
        1.3
    )
    assert float(attention_stats[2, dsa_monitor.ATTENTION_MASS_SUM]) == pytest.approx(
        2.0
    )
    assert indexer_stats.shape == (2, dsa_monitor.NUM_INDEXER_TOPK_STATS)
    assert float(indexer_stats[0, dsa_monitor.POSITION_RANGE_SUM]) == 0.0
    assert float(indexer_stats[0, dsa_monitor.POSITION_DISTANCE_SUM]) == 4.0
    assert float(indexer_stats[0, dsa_monitor.POSITION_STD_SUM]) == 0.0
    assert float(indexer_stats[1, dsa_monitor.POSITION_RANGE_SUM]) == pytest.approx(4.0)
    assert float(indexer_stats[1, dsa_monitor.POSITION_DISTANCE_SUM]) == pytest.approx(
        4.0
    )
    assert float(indexer_stats[1, dsa_monitor.POSITION_STD_SUM]) == pytest.approx(2.0)
    assert (indexer_stats[:, dsa_monitor.INDEXER_TOPK_MONITORED_Q_COUNT] == 2).all()
    assert (attention_stats[:, dsa_monitor.ATTENTION_MASS_MONITORED_Q_COUNT] == 2).all()
