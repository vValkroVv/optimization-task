"""Reference of the attention row statistics that materializes the attention matrix.

Written from the metric definitions, not from the kernel: every document's
``[H, n, n]`` logits are computed in float64, the log-partition is the logsumexp
of the allowed keys, and each field is a masked reduction of ``x = s - lse``.
"""

import math
from typing import NamedTuple

import numpy as np
import torch

MIN_KEYS = 32
LOCAL_WINDOW = 128
QUANTILE_PERCENTILES = (0, 1, 5, 25, 50, 75, 95, 99, 100)

# The per-row fields, in the order of the rows of `ReferenceRows.rows`.
ROW_FIELDS = (
    "ent",
    "entn",
    "pmax",
    "lpmin",
    "mlogit",
    "sink",
    "pmin",
    "coll",
    "dist",
    "local",
)

# attn/softmax/row_stats (id 174) fields after the row count, each x H.
ROW_STATS_LAYOUT = (
    ("ent", "sum"),
    ("ent", "sumsq"),
    ("ent", "min"),
    ("ent", "max"),
    ("entn", "sum"),
    ("entn", "min"),
    ("entn", "max"),
    ("pmax", "sum"),
    ("pmax", "min"),
    ("pmax", "max"),
    ("pmin", "sum"),
    ("lpmin", "sum"),
    ("lpmin", "min"),
    ("lpmin", "max"),
    ("mlogit", "sum"),
    ("mlogit", "max"),
    ("coll", "sum"),
    ("sink", "sum"),
    ("sink", "max"),
    ("dist", "sum"),
    ("local", "sum"),
)


class ReferenceRows(NamedTuple):
    rows: torch.Tensor  # float64 [len(ROW_FIELDS), H, T]
    num_keys: torch.Tensor  # int32 [T]
    lse: torch.Tensor  # float64 [H, T]


def reference_rows(
    q: torch.Tensor, k: torch.Tensor, scale: float, cu_seqlens
) -> ReferenceRows:
    """``q``, ``k``: ``[T, H, D]``; documents ``cu_seqlens[i]:cu_seqlens[i + 1]``."""
    num_tokens, num_heads = q.shape[:2]
    rows = torch.zeros(len(ROW_FIELDS), num_heads, num_tokens, dtype=torch.float64)
    num_keys = torch.zeros(num_tokens, dtype=torch.int32)
    lse = torch.zeros(num_heads, num_tokens, dtype=torch.float64)
    bounds = [int(b) for b in cu_seqlens]
    for begin, end in zip(bounds[:-1], bounds[1:]):
        n = end - begin
        if n == 0:
            continue
        s = torch.einsum(
            "qhd,khd->hqk", q[begin:end].double().cpu(), k[begin:end].double().cpu()
        )
        s = s * scale
        i = torch.arange(n)[:, None]
        j = torch.arange(n)[None, :]
        allowed = j <= i
        doc_lse = torch.logsumexp(s.masked_fill(~allowed, -math.inf), dim=-1)
        x = s - doc_lse[..., None]
        p = torch.where(allowed, x.exp(), torch.zeros_like(x))
        ent = -torch.where(allowed, p * x, torch.zeros_like(x)).sum(-1)
        fields = {
            "ent": ent,
            "entn": ent / torch.log(torch.arange(1, n + 1, dtype=torch.float64)),
            "pmax": x.masked_fill(~allowed, -math.inf).amax(-1).exp(),
            "pmin": x.masked_fill(~allowed, math.inf).amin(-1).exp(),
            "lpmin": x.masked_fill(~allowed, math.inf).amin(-1),
            "mlogit": s.masked_fill(~allowed, -math.inf).amax(-1),
            "coll": (p * p).sum(-1),
            "sink": p[..., 0],
            "dist": (p * (i - j).double()).sum(-1),
            "local": torch.where(
                allowed & (j > i - LOCAL_WINDOW), p, torch.zeros_like(p)
            ).sum(-1),
        }
        for index, name in enumerate(ROW_FIELDS):
            rows[index, :, begin:end] = fields[name]
        num_keys[begin:end] = torch.arange(1, n + 1, dtype=torch.int32)
        lse[:, begin:end] = doc_lse
    return ReferenceRows(rows, num_keys, lse)


def reference_records(rows: torch.Tensor, num_keys: torch.Tensor):
    """Records 174-178 of per-row statistics: float32, int32, int32, float32, int32 tensors."""
    num_heads, num_tokens = rows.shape[1:]
    valid = (num_keys >= MIN_KEYS).numpy()
    values = rows.double().numpy()
    count = int(valid.sum())

    def statistic(name, kind, head):
        x = values[ROW_FIELDS.index(name), head][valid]
        if kind == "sum":
            return x.sum()
        if kind == "sumsq":
            return (x * x).sum()
        if kind == "min":
            return x.min() if x.size else math.inf
        return x.max() if x.size else -math.inf

    row_stats = [float(count)] + [
        statistic(name, kind, head)
        for name, kind in ROW_STATS_LAYOUT
        for head in range(num_heads)
    ]

    entn = values[ROW_FIELDS.index("entn")]
    finite = valid[None, :] & np.isfinite(entn)
    bins = np.floor(np.where(finite, entn, 0.0) * 32).clip(0, 31).astype(np.int64)
    head_hist = np.stack(
        [np.bincount(bins[h][finite[h]], minlength=32) for h in range(num_heads)]
    )

    population = np.sort(entn[finite])
    population_size = population.size
    if population_size:
        indices = [
            max(-(-p * population_size // 100) - 1, 0) for p in QUANTILE_PERCENTILES
        ]
        quantiles = population[indices]
    else:
        quantiles = np.full(len(QUANTILE_PERCENTILES), math.nan)
    quantile_counts = [
        population_size,
        int((valid[None, :] & np.isnan(entn)).sum()),
        int((valid[None, :] & (entn == math.inf)).sum()),
        int((valid[None, :] & (entn == -math.inf)).sum()),
        int(num_heads * (num_tokens - count)),
    ]
    return (
        torch.tensor(row_stats, dtype=torch.float64).float(),
        torch.from_numpy(head_hist.sum(0)).int(),
        torch.from_numpy(head_hist.reshape(-1)).int(),
        torch.tensor(quantiles, dtype=torch.float64).float(),
        torch.tensor(quantile_counts, dtype=torch.int32),
    )
