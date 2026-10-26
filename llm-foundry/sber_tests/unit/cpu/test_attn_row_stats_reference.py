"""The reference the attention row-statistics kernel is held to, on cases written out by hand.

`sber_tests/unit/gpu/test_attn_row_stats.py` compares the kernel with
`reference_rows` and `reference_records`; a mistake in them would fail it for the
wrong reason, so the metric definitions and the layout of the records 174-178
are pinned here without the kernel.
"""

import math

import pytest
import torch

from sber_tests.unit.attn_row_stats_reference import (
    MIN_KEYS,
    ROW_FIELDS,
    ROW_STATS_LAYOUT,
    reference_records,
    reference_rows,
)

pytestmark = [pytest.mark.unit, pytest.mark.fast]

ENTN = ROW_FIELDS.index("entn")


def _constant_rows(lengths, values):
    """Rows whose field ``f`` of head ``h`` is ``values[f][h]`` in every row."""
    num_keys = torch.cat([torch.arange(1, n + 1, dtype=torch.int32) for n in lengths])
    rows = torch.tensor(values, dtype=torch.float32)[:, :, None].repeat(
        1, 1, num_keys.numel()
    )
    return rows, num_keys


def test_uniform_attention_has_closed_form_statistics():
    seq_len, heads, dim = 300, 2, 16
    q = torch.zeros(seq_len, heads, dim)
    k = torch.randn(seq_len, heads, dim)

    rows = reference_rows(q, k, 0.5, [0, seq_len]).rows

    n = torch.arange(1, seq_len + 1, dtype=torch.float64)
    expected = {
        "ent": n.log(),
        "pmax": 1 / n,
        "pmin": 1 / n,
        "lpmin": -n.log(),
        "mlogit": torch.zeros_like(n),
        "sink": 1 / n,
        "coll": 1 / n,
        "dist": (n - 1) / 2,
        "local": n.clamp(max=128) / n,
    }
    for name, values in expected.items():
        torch.testing.assert_close(
            rows[ROW_FIELDS.index(name)],
            values.expand(heads, seq_len),
            rtol=1e-12,
            atol=1e-12,
        )
    torch.testing.assert_close(
        rows[ENTN, :, 1:],
        torch.ones(heads, seq_len - 1, dtype=torch.float64),
        rtol=1e-12,
        atol=0,
    )


def test_causal_mask_ends_at_the_document_boundary():
    # The second document's first row sees only its own key, not the first document.
    q = torch.zeros(5, 1, 4)
    k = torch.randn(5, 1, 4)
    reference = reference_rows(q, k, 1.0, [0, 3, 5])

    assert reference.num_keys.tolist() == [1, 2, 3, 1, 2]
    sink = reference.rows[ROW_FIELDS.index("sink"), 0]
    torch.testing.assert_close(
        sink, torch.tensor([1, 1 / 2, 1 / 3, 1, 1 / 2], dtype=torch.float64)
    )


def test_record_layout():
    lengths = [50, 10]
    heads = 3
    values = [
        [1.5 + 10 * field + head for head in range(heads)]
        for field in range(len(ROW_FIELDS))
    ]
    values[ENTN] = [0.1, 0.2, 0.3]
    rows, num_keys = _constant_rows(lengths, values)
    count = 50 - (MIN_KEYS - 1)  # the 10-token document has no row with 32 keys

    def expected(name, kind, head):
        v = values[ROW_FIELDS.index(name)][head]
        return {"sum": count * v, "sumsq": count * v * v, "min": v, "max": v}[kind]

    row_stats, entropy_hist, head_hist, quantiles, counts = reference_records(
        rows, num_keys
    )

    expected_row_stats = [float(count)] + [
        expected(name, kind, head)
        for name, kind in ROW_STATS_LAYOUT
        for head in range(heads)
    ]
    assert row_stats.shape == (1 + 21 * heads,)
    torch.testing.assert_close(
        row_stats,
        torch.tensor(expected_row_stats, dtype=torch.float32),
        rtol=1e-6,
        atol=0,
    )
    # floor(32 * entn): 0.1 -> 3, 0.2 -> 6, 0.3 -> 9; the bins of one head are adjacent.
    expected_head_hist = torch.zeros(heads, 32, dtype=torch.int32)
    for head, bin_index in enumerate([3, 6, 9]):
        expected_head_hist[head, bin_index] = count
    assert torch.equal(head_hist, expected_head_hist.reshape(-1))
    assert torch.equal(entropy_hist, expected_head_hist.sum(0))
    assert quantiles.shape == (9,)
    assert counts.tolist() == [heads * count, 0, 0, 0, heads * (60 - count)]


def test_rows_below_min_keys_are_excluded_whatever_they_hold():
    lengths = [31, 32, 70]
    values = [[float(f + 1), float(f + 2)] for f in range(len(ROW_FIELDS))]
    values[ENTN] = [0.25, 0.75]
    rows, num_keys = _constant_rows(lengths, values)
    clean = reference_records(rows, num_keys)

    rows[:, :, num_keys < MIN_KEYS] = torch.nan
    poisoned = reference_records(rows, num_keys)

    assert poisoned[0][0].item() == (32 - 31) + (70 - 31)
    for got, want in zip(poisoned, clean):
        torch.testing.assert_close(got, want, rtol=0, atol=0)


def test_no_row_with_enough_keys():
    heads = 2
    rows, num_keys = _constant_rows([31, 5], [[0.5] * heads for _ in ROW_FIELDS])

    row_stats, entropy_hist, head_hist, quantiles, counts = reference_records(
        rows, num_keys
    )

    assert row_stats[0].item() == 0
    for index, (_, kind) in enumerate(ROW_STATS_LAYOUT):
        expected = {"sum": 0.0, "sumsq": 0.0, "min": math.inf, "max": -math.inf}[kind]
        assert (
            row_stats[1 + index * heads : 1 + (index + 1) * heads].tolist()
            == [expected] * heads
        )
    assert entropy_hist.tolist() == [0] * 32
    assert head_hist.tolist() == [0] * (32 * heads)
    assert quantiles.isnan().all()
    assert counts.tolist() == [0, 0, 0, 0, heads * 36]


@pytest.mark.parametrize(
    "entn, bin_index",
    [
        (0.0, 0),
        (-1e-7, 0),
        (1 / 32, 1),
        (0.5, 16),
        (0.99999, 31),
        (1.0, 31),
        (1.0000002, 31),
    ],
)
def test_entropy_histogram_bins(entn, bin_index):
    values = [[entn] if name == "entn" else [0.0] for name in ROW_FIELDS]
    rows, num_keys = _constant_rows([40], values)

    _, entropy_hist, head_hist, _, _ = reference_records(rows, num_keys)

    expected = torch.zeros(32, dtype=torch.int32)
    expected[bin_index] = 40 - (MIN_KEYS - 1)
    assert torch.equal(entropy_hist, expected)
    assert torch.equal(head_hist, expected)


def test_non_finite_entropy_is_visible_but_not_binned():
    heads = 2
    values = [[1.0 + f, 2.0 + f] for f in range(len(ROW_FIELDS))]
    values[ENTN] = [0.4, 0.6]
    rows, num_keys = _constant_rows([60], values)
    clean = reference_records(rows, num_keys)
    rows[ENTN, 0, 40] = torch.nan
    rows[ENTN, 0, 41] = math.inf
    rows[ENTN, 0, 42] = -math.inf

    dirty = reference_records(rows, num_keys)

    entn_sum = 1 + 4 * heads  # the first head's entn_sum
    assert dirty[0][entn_sum].isnan()
    assert dirty[0][entn_sum + 1] == clean[0][entn_sum + 1]
    assert (dirty[0][1 : 1 + 4 * heads] == clean[0][1 : 1 + 4 * heads]).all()
    assert dirty[1].sum() == clean[1].sum() - 3
    assert torch.equal(dirty[2][32:], clean[2][32:])
    assert dirty[4].tolist() == [2 * 29 - 3, 1, 1, 1, 2 * 31]
