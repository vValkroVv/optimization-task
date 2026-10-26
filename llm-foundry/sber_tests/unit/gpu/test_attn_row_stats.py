"""Attention row statistics: Triton kernels vs the materialized float64 reference."""

import itertools
import math

import pytest
import torch
from metric_sampler.capture.kernels.triton_launch import jit_launches

from llmfoundry.models.layers import attn_row_stats
from llmfoundry.models.layers.attn_row_stats import (
    RECORD_FIELDS,
    RowField,
    attention_row_stats,
    attention_row_values,
)
from sber_tests.unit.attn_row_stats_reference import (
    MIN_KEYS,
    ROW_FIELDS,
    ROW_STATS_LAYOUT,
    reference_records,
    reference_rows,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.requires_gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
]

DEVICE = "cuda"
HEADS = 2
HEAD_DIM = 192  # MLA: 128 nope + 64 rope


@pytest.mark.parametrize("head_dim, expected", [
    (1, (16, 0, True)), (8, (16, 0, True)), (16, (16, 0, False)),
    (64, (64, 0, False)), (80, (64, 16, False)), (96, (64, 32, False)),
    (128, (128, 0, False)), (192, (128, 64, False)), (193, (256, 0, True)),
])
def test_head_dim_chunks_are_independent_of_the_qk_monitor(head_dim, expected):
    assert attn_row_stats._split_head_dim(head_dim) == expected


def _f32(value):
    """The float32 the kernel, like flash-attn, receives for a softmax scale. The
    reference must use that number: a logit of 200 differs by 1e-5 between the
    double scale and its float32 rounding, the same sign for every row."""
    return torch.tensor(value, dtype=torch.float32).item()


# Sharp enough that the entropies are far from log(n).
SCALE = _f32(3 * HEAD_DIM**-0.5)
HUGE_SCALE = _f32(2.5 * HEAD_DIM**-0.5)

# The kernel's logits are fp32, so an entropy or a log probability is exact to a
# few fp32 ulps of the largest logit (RESOLUTION_ULPS of them, a margin over the
# 2-3 ulps of summation-order noise between the kernel and the reference), not
# relative to its own size: a row with one key has a log
# probability of exactly 0. entn = ent / ln(n) carries the entropy's error
# divided by ln(n), which is largest for the first rows of a document.
RESOLUTION_ULPS = 8
ENT_ATOL = 1e-4
RELATIVE = {"rtol": 1e-4, "atol": 1e-5}


def _probability_rtol(logit_atol):
    """A probability exp(x) carries the absolute error of x as a relative one; a
    collision probability (p squared) twice that."""
    return max(RELATIVE["rtol"], 2 * logit_atol)


def _logit_atol(reference):
    largest = max(
        reference.rows[ROW_FIELDS.index("mlogit")].abs().max().item(),
        reference.lse.abs().max().item(),
        1.0,
    )
    return RESOLUTION_ULPS * torch.finfo(torch.float32).eps * largest


def _qk(total, amp=1.0, heads=HEADS, head_dim=HEAD_DIM, dtype=torch.bfloat16):
    shape = (total, heads, head_dim)
    return tuple((torch.randn(shape, device=DEVICE) * amp).to(dtype) for _ in range(2))


def _boundaries(lengths):
    return [0, *itertools.accumulate(lengths)]


def _varlen_call(q, k, reference, scale, lengths):
    """Arguments of the kernels for the documents of `lengths` packed along dim 0."""
    return (
        q[None],
        k[None],
        reference.lse.float().to(DEVICE),
        scale,
        torch.tensor(_boundaries(lengths), dtype=torch.int32, device=DEVICE),
        max(lengths),
    )


def _assert_rows_match(reference, rows):
    logit_atol = _logit_atol(reference)
    ent_atol = max(ENT_ATOL, logit_atol)
    for field in RowField:
        got = rows[field].double().cpu()
        want = reference.rows[field]
        if field is RowField.ENTN:
            # entn is undefined for the first row of a document.
            defined = reference.num_keys > 1
            scale = torch.log(reference.num_keys[defined].double())
            got, want = got[:, defined] * scale, want[:, defined] * scale
            tolerance = {"rtol": 0, "atol": ent_atol}
        elif field is RowField.ENT:
            tolerance = {"rtol": 0, "atol": ent_atol}
        elif field in (RowField.LPMIN, RowField.MLOGIT):
            tolerance = {"rtol": RELATIVE["rtol"], "atol": logit_atol}
        else:
            tolerance = {
                "rtol": _probability_rtol(logit_atol),
                "atol": RELATIVE["atol"],
            }
        torch.testing.assert_close(got, want, msg=field.name, **tolerance)


def _assert_records_close(records, want, stat_tolerance, quantile_atol):
    row_stats, hist, head_hist, quantiles, quantile_counts = (r.cpu() for r in records)
    want_stats, want_hist, want_head_hist, want_quantiles, want_counts = want
    heads = (want_stats.numel() - 1) // len(ROW_STATS_LAYOUT)
    assert row_stats.shape == want_stats.shape
    assert row_stats[0] == want_stats[0]
    for index, (name, kind) in enumerate(ROW_STATS_LAYOUT):
        columns = slice(1 + index * heads, 1 + (index + 1) * heads)
        torch.testing.assert_close(
            row_stats[columns],
            want_stats[columns],
            msg=f"{name}_{kind}",
            equal_nan=True,
            **stat_tolerance(name, kind),
        )
    # An entropy within the kernel's rounding of a bin edge may land in the neighbouring bin.
    for got, want_bins in ((hist, want_hist), (head_hist, want_head_hist)):
        assert got.sum() == want_bins.sum()
        assert (got - want_bins).abs().sum() <= 4
    torch.testing.assert_close(
        quantiles, want_quantiles, rtol=0, atol=quantile_atol, equal_nan=True
    )
    assert torch.equal(quantile_counts, want_counts)


def _assert_records_match(records, reference, sum_rtol=1e-5):
    """The records against the aggregation of the float64 reference rows."""
    logit_atol = _logit_atol(reference)
    ent_atol = max(ENT_ATOL, logit_atol)

    def tolerance(name, kind):
        if kind in ("sum", "sumsq"):
            return {"rtol": sum_rtol, "atol": 1e-6}
        if name == "ent":
            return {"rtol": 0, "atol": ent_atol}
        if name == "entn":
            # Rows that count have at least MIN_KEYS keys.
            return {"rtol": 0, "atol": ent_atol / math.log(MIN_KEYS)}
        if name in ("lpmin", "mlogit"):
            return {"rtol": RELATIVE["rtol"], "atol": logit_atol}
        return {"rtol": _probability_rtol(logit_atol), "atol": RELATIVE["atol"]}

    _assert_records_close(
        records,
        reference_records(reference.rows.float(), reference.num_keys),
        tolerance,
        ent_atol / math.log(MIN_KEYS),
    )


def _assert_records_follow_rows(records, rows, reference):
    """The records against the aggregation of the per-row values the kernel gave.

    This holds the reduction (partial records, merge, histogram, quantiles) to the
    per-row values without a float64 oracle for the logits. An oracle with an exact
    log-partition cannot share the tensor core's accumulation error of the raw dot
    product, which is systematic: the same sign and a similar size for the nearly
    one-hot rows of huge logits, so that it adds up over the rows of a sum, where
    flash-attn's own log-partition carries it too and cancels it.
    """
    _assert_records_close(
        records,
        reference_records(rows.float().cpu(), reference.num_keys),
        lambda name, kind: {"rtol": 1e-5, "atol": 1e-6},
        1e-6,
    )


def test_row_fields_follow_the_reference_order():
    assert [field.name.lower() for field in RowField] == list(ROW_FIELDS)
    assert RECORD_FIELDS[0] == "rows"
    assert [f"{name}_{kind}" for name, kind in ROW_STATS_LAYOUT] == [
        name for name in RECORD_FIELDS[1:]
    ]


@pytest.mark.parametrize("pad_lse", [False, True])
@pytest.mark.parametrize("seq_len", [1, 31, 32, 33, 127, 128, 129, 257, 700])
def test_dense(seq_len, pad_lse):
    bsz = 2
    q, k = _qk(bsz * seq_len)
    reference = reference_rows(q, k, SCALE, _boundaries([seq_len] * bsz))
    lse = (
        reference.lse.float()
        .view(HEADS, bsz, seq_len)
        .transpose(0, 1)
        .contiguous()
        .to(DEVICE)
    )
    if pad_lse:
        # flash-attn 2 pads the log-partition to a multiple of 128 along the sequence.
        lse = torch.nn.functional.pad(lse, (0, -seq_len % 128), value=math.nan)
    args = (
        q.view(bsz, seq_len, HEADS, HEAD_DIM),
        k.view(bsz, seq_len, HEADS, HEAD_DIM),
        lse,
        SCALE,
    )

    _assert_rows_match(reference, attention_row_values(*args))
    _assert_records_match(attention_row_stats(*args), reference)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "lengths", [[300], [100, 200], [5, 130, 1, 700, 64, 3], [32, 31, 33]], ids=str
)
def test_varlen(lengths, dtype):
    q, k = _qk(sum(lengths), dtype=dtype)
    reference = reference_rows(q, k, SCALE, _boundaries(lengths))
    args = _varlen_call(q, k, reference, SCALE, lengths)

    _assert_rows_match(reference, attention_row_values(*args))
    _assert_records_match(attention_row_stats(*args), reference)


def test_dense_and_varlen_launches_agree():
    bsz, seq_len = 3, 300
    q, k = _qk(bsz * seq_len)
    reference = reference_rows(q, k, SCALE, _boundaries([seq_len] * bsz))
    lse = reference.lse.float().to(DEVICE)
    q4, k4 = q.view(bsz, seq_len, HEADS, HEAD_DIM), k.view(
        bsz, seq_len, HEADS, HEAD_DIM
    )
    cu_seqlens = torch.tensor(
        _boundaries([seq_len] * bsz), dtype=torch.int32, device=DEVICE
    )

    dense = attention_row_stats(
        q4, k4, lse.view(HEADS, bsz, seq_len).transpose(0, 1), SCALE
    )
    varlen = attention_row_stats(
        q4, k4, lse, SCALE, cu_seqlens=cu_seqlens, max_seqlen=seq_len
    )

    for got, want in zip(dense, varlen):
        if got.is_floating_point():
            torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6, equal_nan=True)
        else:
            assert torch.equal(got, want)


@pytest.mark.parametrize("head_dim", [64, 96, 128])
def test_other_head_dims(head_dim):
    lengths = [300, 150]
    scale = _f32(3 * head_dim**-0.5)
    q, k = _qk(sum(lengths), head_dim=head_dim)
    reference = reference_rows(q, k, scale, _boundaries(lengths))
    args = _varlen_call(q, k, reference, scale, lengths)

    _assert_rows_match(reference, attention_row_values(*args))
    _assert_records_match(attention_row_stats(*args), reference)


def test_huge_logits():
    lengths = [500, 61]
    q, k = _qk(sum(lengths), amp=4.0)
    reference = reference_rows(q, k, HUGE_SCALE, _boundaries(lengths))
    assert reference.rows[RowField.MLOGIT].max() > 100
    args = _varlen_call(q, k, reference, HUGE_SCALE, lengths)

    rows = attention_row_values(*args)

    _assert_rows_match(reference, rows)
    _assert_records_follow_rows(attention_row_stats(*args), rows, reference)


def test_uniform_attention_has_closed_form_statistics():
    seq_len = 300
    q = torch.zeros(1, seq_len, HEADS, HEAD_DIM, device=DEVICE, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    n = torch.arange(1, seq_len + 1, dtype=torch.float64)
    lse = n.log().float().expand(1, HEADS, seq_len).contiguous().to(DEVICE)

    rows = attention_row_values(q, k, lse, SCALE).double().cpu()

    expected = {
        RowField.ENT: n.log(),
        RowField.PMAX: 1 / n,
        RowField.PMIN: 1 / n,
        RowField.LPMIN: -n.log(),
        RowField.MLOGIT: torch.zeros_like(n),
        RowField.SINK: 1 / n,
        RowField.COLL: 1 / n,
        RowField.DIST: (n - 1) / 2,
        RowField.LOCAL: n.clamp(max=128) / n,
    }
    for field, values in expected.items():
        torch.testing.assert_close(
            rows[field],
            values.expand(HEADS, seq_len),
            rtol=1e-5,
            atol=1e-5,
            msg=field.name,
        )
    torch.testing.assert_close(
        rows[RowField.ENTN, :, 1:],
        torch.ones(HEADS, seq_len - 1, dtype=torch.float64),
        rtol=1e-5,
        atol=0,
    )


def test_non_contiguous_inputs():
    # MLA concatenates the nope and rope parts, so q and k may be strided views.
    seq_len = 260
    wide = torch.randn(1, seq_len, HEADS, 2 * HEAD_DIM, device=DEVICE).to(
        torch.bfloat16
    )
    q, k = wide[..., :HEAD_DIM], wide[..., HEAD_DIM:]
    assert q.stride(-1) == 1 and not q.is_contiguous()
    reference = reference_rows(q[0], k[0], SCALE, [0, seq_len])
    args = (q, k, reference.lse.float().to(DEVICE)[None], SCALE)

    _assert_rows_match(reference, attention_row_values(*args))
    _assert_records_match(attention_row_stats(*args), reference)


@pytest.mark.parametrize(
    "amp, scale", [(1.0, SCALE), (4.0, HUGE_SCALE)], ids=["moderate", "huge"]
)
def test_with_flash_attn_log_partition(amp, scale):
    """The log-partition flash-attn returns is the normalizer the kernel expects."""
    pytest.importorskip("flash_attn")
    from llmfoundry.models.utils import flash_attn_import

    bsz, seq_len, heads = 2, 1024, 4
    # flash-attn 2 has no unequal q/k and v head dims.
    v_dim = 128 if flash_attn_import.HAS_FA3 else HEAD_DIM
    q, k = _qk(bsz * seq_len, amp=amp, heads=heads)
    q, k = q.view(bsz, seq_len, heads, HEAD_DIM), k.view(bsz, seq_len, heads, HEAD_DIM)
    v = torch.randn(bsz, seq_len, heads, v_dim, device=DEVICE).to(torch.bfloat16)
    lse = flash_attn_import._flash_attn_forward(
        q, k, v, 0.0, scale, True, -1, -1, 0.0, None, False
    )[1]
    reference = reference_rows(
        q.flatten(0, 1), k.flatten(0, 1), scale, _boundaries([seq_len] * bsz)
    )

    rows = attention_row_values(q, k, lse, scale)

    _assert_rows_match(reference, rows)
    records = attention_row_stats(q, k, lse, scale)
    if amp == 1.0:
        _assert_records_match(records, reference)
    else:
        _assert_records_follow_rows(records, rows, reference)


def test_short_documents_are_excluded():
    lengths = [31, 32, 70]
    q, k = _qk(sum(lengths), heads=3)
    reference = reference_rows(q, k, SCALE, _boundaries(lengths))

    records = attention_row_stats(*_varlen_call(q, k, reference, SCALE, lengths))

    assert records.row_stats[0].item() == (32 - 31) + (70 - 31)
    _assert_records_match(records, reference)


def test_no_row_with_enough_keys():
    lengths = [31, 5]
    q, k = _qk(sum(lengths))
    reference = reference_rows(q, k, SCALE, _boundaries(lengths))

    records = attention_row_stats(*_varlen_call(q, k, reference, SCALE, lengths))

    assert records.row_stats[0].item() == 0
    assert records.entropy_quantiles.isnan().all()
    _assert_records_match(records, reference)


def test_tokens_outside_every_document_are_masked():
    # An empty document, and a tail of tokens that belong to none.
    boundaries = [0, 100, 100, 250]
    q, k = _qk(300)
    reference = reference_rows(q, k, SCALE, boundaries)
    lse = torch.full((HEADS, 300), math.nan, device=DEVICE)
    lse[:, :250] = reference.lse.float().to(DEVICE)[:, :250]

    records = attention_row_stats(
        q[None],
        k[None],
        lse,
        SCALE,
        cu_seqlens=torch.tensor(boundaries, dtype=torch.int32, device=DEVICE),
        max_seqlen=150,
    )

    _assert_records_match(records, reference)


def test_is_deterministic():
    lengths = [300, 200]
    q, k = _qk(sum(lengths))
    reference = reference_rows(q, k, SCALE, _boundaries(lengths))
    args = _varlen_call(q, k, reference, SCALE, lengths)

    first, second = (attention_row_stats(*args) for _ in range(2))

    for a, b in zip(first, second):
        assert torch.equal(a, b)


def _config_id(config):
    kind = "elementwise" if config.kwargs["ELEMENTWISE"] else "reduced"
    return (
        f"m{config.kwargs['BLOCK_M']}n{config.kwargs['BLOCK_N']}-{kind}"
        f"-w{config.num_warps}s{config.num_stages}"
    )


@pytest.mark.parametrize(
    "config", attn_row_stats._row_stats_kernel.configs, ids=_config_id
)
def test_every_config_is_exact(config):
    # The autotuner picks one config per shape; the others must be right too.
    lengths = [300, 129, 64]
    q, k = _qk(sum(lengths))
    reference = reference_rows(q, k, SCALE, _boundaries(lengths))
    args = _varlen_call(q, k, reference, SCALE, lengths)

    rows = attn_row_stats._launch(*args, True, configs=[config]).rows
    launch = attn_row_stats._launch(*args, False, configs=[config])

    _assert_rows_match(reference, rows)
    _assert_records_match(attn_row_stats._records(launch), reference)


@pytest.mark.parametrize("backwards", [False, True], ids=["in order", "reversed"])
def test_launches_that_share_the_buffers_leave_nothing_behind(backwards):
    # The autotuner benchmarks every config on the buffers of the call that tunes,
    # which then runs the winner on them: a config that writes other partial
    # records than the next must not leave its own counted twice.
    lengths = [300, 129, 64]
    q, k = _qk(sum(lengths))
    reference = reference_rows(q, k, SCALE, _boundaries(lengths))
    args = _varlen_call(q, k, reference, SCALE, lengths)
    configs = list(attn_row_stats._row_stats_kernel.configs)

    launch = attn_row_stats._launch(
        *args, False, configs=configs[::-1] if backwards else configs
    )

    _assert_records_match(attn_row_stats._records(launch), reference)


def test_the_call_that_tunes_is_exact():
    # head_dim 80 (64 + 16) is used by no other test, so this is the first call of
    # its autotune key and runs with the benchmark runs' leftovers, if any.
    lengths = [300, 150]
    head_dim = 80
    scale = _f32(3 * head_dim**-0.5)
    q, k = _qk(sum(lengths), head_dim=head_dim)
    reference = reference_rows(q, k, scale, _boundaries(lengths))

    records = attention_row_stats(*_varlen_call(q, k, reference, scale, lengths))

    _assert_records_match(records, reference)


@pytest.mark.parametrize("head_hist", [True, False])
@pytest.mark.parametrize("offset", [0, 1], ids=["aligned", "misaligned"])
def test_records_written_into_given_tensors_equal_the_returned_ones_bit_for_bit(head_hist, offset):
    # The destinations are consecutive words of one buffer, as the recorder's slots are: at an odd word the
    # tensors' addresses are not a multiple of 16 bytes.
    lengths = [300, 200]
    q, k = _qk(sum(lengths))
    reference = reference_rows(q, k, SCALE, _boundaries(lengths))
    args = _varlen_call(q, k, reference, SCALE, lengths)
    expected = attention_row_stats(*args)
    sizes = [tensor.numel() for tensor in expected]
    words = torch.full((offset + sum(sizes),), 0x12345678, dtype=torch.int32, device=DEVICE)
    destinations = list(words[offset:].split(sizes))
    destinations[1].zero_()
    destinations[2].zero_()
    stats, hist, head, quantiles, counts = destinations
    out = (stats.view(torch.float32), hist, head if head_hist else None, quantiles.view(torch.float32), counts)

    records = attention_row_stats(*args, out=out)

    assert all(record is target for record, target in zip(records, out))
    for index, (got, want) in enumerate(zip(records, expected)):
        if index != 2 or head_hist:
            assert torch.equal(got.view(torch.int32), want.view(torch.int32)), index
    assert head_hist or torch.equal(head, torch.full_like(head, 0))


def test_directly_launched_records_equal_those_of_the_jit_call_bit_for_bit():
    # The first call of a shape tunes the kernel through the JIT call and the later ones launch its compiled
    # variant directly, on fresh buffers each time: none of them may differ from the JIT call's.
    lengths = [300, 129, 64]
    q, k = _qk(sum(lengths))
    reference = reference_rows(q, k, SCALE, _boundaries(lengths))
    args = _varlen_call(q, k, reference, SCALE, lengths)

    launched = [attention_row_stats(*args) for _ in range(3)]
    with jit_launches():
        jitted = attention_row_stats(*args)

    for records in launched:
        for got, want in zip(records, jitted):
            assert torch.equal(got.view(torch.int32), want.view(torch.int32))
