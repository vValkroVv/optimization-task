import torch


ATTENTION_MASS_SUM = 0
ATTENTION_MASS_MONITORED_Q_COUNT = 1
ATTENTION_MASS_ELIGIBLE_Q_COUNT = 2
ATTENTION_MASS_TOTAL_Q_COUNT = 3
NUM_ATTENTION_MASS_STATS = 4

POSITION_RANGE_SUM = 0
POSITION_DISTANCE_SUM = 1
POSITION_STD_SUM = 2
INDEXER_TOPK_MONITORED_Q_COUNT = 3
INDEXER_TOPK_ELIGIBLE_Q_COUNT = 4
INDEXER_TOPK_TOTAL_Q_COUNT = 5
NUM_INDEXER_TOPK_STATS = 6


def _doc_ids(positions: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
    boundaries = cu_seqlens[1:-1].to(
        device=positions.device,
        dtype=positions.dtype,
    )
    return torch.bucketize(positions, boundaries, right=True)


@torch.no_grad()
def build_query_sample(
    seq_len: int,
    cu_seqlens: torch.Tensor | None,
    q_start_pos: int,
    num_queries: int | None,
    device: torch.device,
) -> tuple[torch.Tensor, int, int]:
    """Build one deterministic query sample shared by all SP ranks."""
    positions = torch.arange(seq_len, device=device)
    if cu_seqlens is None:
        local_positions = positions
    else:
        cu = cu_seqlens.to(device=device, dtype=positions.dtype)
        docs = _doc_ids(positions, cu)
        local_positions = positions - cu[:-1].index_select(0, docs)

    eligible_positions = positions[local_positions > int(q_start_pos)]
    eligible_count = eligible_positions.numel()
    if num_queries is not None and eligible_count > num_queries:
        # Integer linspace: exactly num_queries rows, spread over all eligible queries.
        sample_indices = (
            torch.arange(num_queries, device=device, dtype=torch.int64)
            * eligible_count
            // num_queries
        )
        sampled_positions = eligible_positions.index_select(0, sample_indices)
    else:
        sampled_positions = eligible_positions

    sample_mask = torch.zeros(seq_len, device=device, dtype=torch.bool)
    sample_mask[sampled_positions] = True
    return sample_mask, eligible_count, seq_len


def _select_dense_indexer_topk(
    index_score: torch.Tensor,
    topk: int,
    offset: int,
    monitor_rows: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sk = index_score.shape[-1]
    k = min(int(topk), sk)
    rows = monitor_rows.long()
    if rows.numel() == 0 or k <= 0:
        empty = torch.empty(
            (0, max(k, 0)),
            device=index_score.device,
            dtype=torch.long,
        )
        return empty, rows, rows

    q_positions = rows + int(offset)
    key_positions = torch.arange(sk, device=index_score.device)
    valid_keys = key_positions[None, :] <= q_positions[:, None]
    selected_scores = index_score[0].index_select(0, rows).float()
    selected_scores.masked_fill_(~valid_keys, -torch.inf)
    _, topk_ids = torch.topk(selected_scores, k, dim=-1)
    topk_length = torch.minimum(
        q_positions + 1,
        torch.full_like(q_positions, k),
    )
    return topk_ids, topk_length, q_positions


def _position_stats(
    topk_ids: torch.Tensor,
    topk_length: torch.Tensor,
    q_positions: torch.Tensor,
) -> torch.Tensor:
    """Return positional sums and count for non-empty top-k rows."""
    stats = torch.zeros(
        NUM_INDEXER_TOPK_STATS,
        device=topk_ids.device,
        dtype=torch.float32,
    )
    if topk_ids.numel() == 0:
        return stats

    width = topk_ids.shape[-1]
    valid = torch.arange(width, device=topk_ids.device)[None, :] < topk_length[:, None]
    positions = topk_ids.float()
    min_positions = positions.masked_fill(~valid, torch.inf).amin(-1)
    max_positions = positions.masked_fill(~valid, -torch.inf).amax(-1)
    count = topk_length.float().clamp_min(1.0)
    mean = positions.masked_fill(~valid, 0.0).sum(-1) / count
    variance = (positions - mean[:, None]).square().masked_fill(~valid, 0.0).sum(
        -1
    ) / count

    stats[POSITION_RANGE_SUM] = (max_positions - min_positions).sum()
    stats[POSITION_DISTANCE_SUM] = (q_positions.float() - min_positions).sum()
    stats[POSITION_STD_SUM] = variance.clamp_min(0.0).sqrt().sum()
    stats[INDEXER_TOPK_MONITORED_Q_COUNT] = float(topk_ids.shape[0])
    return stats


def _position_stats_for_topks(
    topk_ids: torch.Tensor,
    topk_length: torch.Tensor,
    q_positions: torch.Tensor,
    topk_values: tuple[int, ...],
) -> torch.Tensor:
    """Return one positional-stat row for each score-ranked top-k prefix."""
    if not topk_values:
        raise ValueError("topk_values must not be empty")

    width = topk_ids.shape[-1]
    return torch.stack(
        [
            _position_stats(
                topk_ids[:, : min(topk, width)],
                topk_length.clamp_max(topk),
                q_positions,
            )
            for topk in topk_values
        ]
    )


def _empty_attention_mass_stats(
    topk_values: tuple[int, ...],
    device: torch.device,
) -> torch.Tensor:
    if not topk_values:
        raise ValueError("topk_values must not be empty")
    return torch.zeros(
        (len(topk_values), NUM_ATTENTION_MASS_STATS),
        device=device,
        dtype=torch.float32,
    )


def _set_attention_mass_counts(
    stats: torch.Tensor,
    monitored_count: int,
) -> torch.Tensor:
    stats[:, ATTENTION_MASS_MONITORED_Q_COUNT] = monitored_count
    return stats


@torch.no_grad()
def dense_attention_mass_stats(
    attn_score: torch.Tensor,
    attn_l1norm: torch.Tensor,
    index_score: torch.Tensor,
    topk_values: tuple[int, ...],
    offset: int,
    monitor_rows: torch.Tensor,
) -> torch.Tensor:
    """Compute stage-1 captured mass for score-ranked indexer top-k prefixes."""
    stats = _empty_attention_mass_stats(topk_values, index_score.device)
    topk_ids, topk_length, _ = _select_dense_indexer_topk(
        index_score,
        max(topk_values),
        offset,
        monitor_rows,
    )
    if topk_ids.numel() == 0:
        return stats

    rows = monitor_rows.long()
    target = attn_score[0].index_select(0, rows).float()
    denominator = attn_l1norm[0].index_select(0, rows).float().clamp_min(1e-9)[:, None]
    selected_target = torch.gather(target, 1, topk_ids) / denominator
    width = topk_ids.shape[-1]
    for topk_index, topk in enumerate(topk_values):
        prefix_width = min(topk, width)
        valid = (
            torch.arange(prefix_width, device=index_score.device)[None, :]
            < topk_length.clamp_max(topk)[:, None]
        )
        stats[topk_index, ATTENTION_MASS_SUM] = (
            selected_target[:, :prefix_width].masked_fill(~valid, 0.0).sum()
        )
    return _set_attention_mass_counts(stats, topk_ids.shape[0])


@torch.no_grad()
def dense_indexer_topk_stats(
    index_score: torch.Tensor,
    topk_values: tuple[int, ...],
    offset: int,
    monitor_rows: torch.Tensor,
) -> torch.Tensor:
    """Compute stage-1 positional stats for score-ranked top-k prefixes."""
    topk_ids, topk_length, q_positions = _select_dense_indexer_topk(
        index_score,
        max(topk_values),
        offset,
        monitor_rows,
    )
    return _position_stats_for_topks(topk_ids, topk_length, q_positions, topk_values)


@torch.no_grad()
def sparse_indexer_topk_stats(
    topk_ids: torch.Tensor,
    topk_length: torch.Tensor,
    positions_q: torch.Tensor,
    monitor_rows: torch.Tensor,
    topk_values: tuple[int, ...],
    *,
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    weights: torch.Tensor,
    index_scale: float,
) -> torch.Tensor:
    """Compute score-ranked prefix stats from the fused selector's unsorted keys."""
    rows = monitor_rows.long()
    lengths = topk_length.index_select(0, rows).long()
    nonempty = lengths > 0
    rows = rows[nonempty]
    lengths = lengths[nonempty]
    if rows.numel() == 0:
        return torch.zeros(
            (len(topk_values), NUM_INDEXER_TOPK_STATS),
            device=topk_ids.device,
            dtype=torch.float32,
        )

    q_positions = positions_q.index_select(0, rows).long()
    if min(topk_values) >= topk_ids.shape[-1]:
        ids = topk_ids[0].index_select(0, rows).long()
        return _position_stats_for_topks(ids, lengths, q_positions, topk_values)

    stats = torch.zeros(
        (len(topk_values), NUM_INDEXER_TOPK_STATS),
        device=topk_ids.device,
        dtype=torch.float32,
    )
    # Bound the gathered keys and per-head scores even when sampling all queries.
    for start in range(0, rows.numel(), 8):
        chunk = rows[start : start + 8]
        ids = topk_ids[0].index_select(0, chunk).long()
        with torch.autocast(device_type=index_q.device.type, enabled=False):
            # Match the fused scorer's bf16 scaled queries and fp32 accumulated scores.
            q = (index_q[0].index_select(0, chunk) * index_scale).float()
            keys = index_k[0, :, 0].index_select(0, ids.clamp_min(0).flatten())
            keys = keys.view(*ids.shape, index_k.shape[-1]).float()
            scores = torch.bmm(q, keys.transpose(1, 2)).relu_()
            scores.mul_(weights[0].index_select(0, chunk).float().unsqueeze(-1))
            scores = scores.sum(1).masked_fill_(ids < 0, -torch.inf)
        ids = ids.gather(1, scores.argsort(dim=-1, descending=True))
        stats += _position_stats_for_topks(
            ids, lengths[start : start + 8], q_positions[start : start + 8], topk_values
        )
    return stats
