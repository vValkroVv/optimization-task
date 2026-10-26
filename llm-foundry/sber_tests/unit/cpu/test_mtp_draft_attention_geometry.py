"""Prefix-plus-diagonal draft attention: geometry and the lse merge (spec item 10).

Two plain-torch implementations of the geometry item 10 defines, and the proof
they agree: `_dense_prefix_diagonal` builds the mask, `_merged_prefix_diagonal`
merges per-step terms the way a kernel has to. Both run on CPU
and on AngelSpec's own layout, so `../../../../angelspec-conformance/compare_geom.py`
can diff them against their shipped `_mtp_dense_block0` / `_mtp_cached_merge`
without a reshape, and the CUDA path has a reference to be diffed against
(F-21 for why they live here, F-28 for why they carry AngelSpec's signature).

`valid_mask` and `doc_ids` gate the keys on top of the geometry: padding, and the
document boundary I-3 enforces on the label side.

What the tests pin, in order: the two forms against each other at every depth,
I-6 with a positive and a negative control, the padding and document gates
composing, and the reference merge's finite guard, which AngelSpec's `vmask`
mirrors -- the dense form cannot produce a NaN and the merged form can, in the
backward pass only, which `TestMergeNumericGuard` asserts on `.grad`. The shipped
`merge_prefix_diagonal` carries no such guard: it never sees a padded row.

Reference: AngelSpec @ d3412be, `angelspec/models/draft/mtp.py:46`
(`_mtp_dense_block0`) and `:103` (`_mtp_cached_merge`, the lse_terms merge).
"""

import math

import pytest
import torch

BATCH = 2
HEADS = 2
SEQ = 6
DIM = 4


def _blocked(causal, seq, valid_mask, doc_ids):
    """Key positions query `j` may *not* read: `[batch, 1, seq, seq]`.

    `causal` picks the step-0 shape (every key up to `j`) over the later steps'
    (position `j` alone). `doc_ids` is applied to both, where AngelSpec applies it
    to step 0 only: off step 0 the sole permitted key is `j` itself, always in
    `j`'s own document, so the two are the same mask.
    """
    if causal:
        blocked = torch.ones(seq, seq, dtype=torch.bool).triu(1)
    else:
        blocked = ~torch.eye(seq, dtype=torch.bool)
    blocked = blocked[None]
    if valid_mask is not None:
        blocked = blocked | ~valid_mask[:, None, :]
    if doc_ids is not None:
        blocked = blocked | (doc_ids[:, :, None] != doc_ids[:, None, :])
    return blocked[:, None]


def _step_logits(q, keys, causal, valid_mask, doc_ids):
    scale = 1.0 / math.sqrt(q.size(-1))
    logits = q @ keys.transpose(-1, -2) * scale
    return logits.masked_fill(
        _blocked(causal, q.size(-2), valid_mask, doc_ids), torch.finfo(q.dtype).min
    )


def _zero_padded_rows(out, valid_mask):
    if valid_mask is None:
        return out
    return torch.where(valid_mask[:, None, :, None], out, torch.zeros_like(out))


def _softmax_over_steps(q, cache_k, cache_v, valid_mask, doc_ids, flat):
    """One softmax over every step's keys. `flat` makes every step causal."""
    steps = cache_k.size(2)
    logits = torch.cat(
        [
            _step_logits(q, cache_k[:, :, i], flat or i == 0, valid_mask, doc_ids)
            for i in range(steps)
        ],
        dim=-1,
    )
    values = torch.cat([cache_v[:, :, i] for i in range(steps)], dim=-2)
    return _zero_padded_rows(torch.softmax(logits, dim=-1) @ values, valid_mask)


def _dense_prefix_diagonal(q, cache_k, cache_v, valid_mask=None, doc_ids=None):
    """The definition: one softmax over an explicitly built mask.

    `q` is `[batch, heads, seq, dim]` and `cache_k` / `cache_v` are
    `[batch, heads, steps, seq, dim]`; the cache's own step count is the depth,
    so there is no separate depth argument. Step 0 is causal; steps 1.. contribute
    their diagonal entry only.
    """
    return _softmax_over_steps(q, cache_k, cache_v, valid_mask, doc_ids, flat=False)


def _pad_to_neg_inf(lse, valid_mask):
    """Drop a padded *query* row out of the merge, as AngelSpec's `vmask` does."""
    if valid_mask is None:
        return lse
    return torch.where(
        valid_mask[:, None, :], lse, torch.tensor(float("-inf"), dtype=lse.dtype)
    )


def _finite(merged):
    return torch.where(torch.isfinite(merged), merged, torch.zeros_like(merged))


def _merged_prefix_diagonal(q, cache_k, cache_v, valid_mask=None, doc_ids=None):
    """The same thing, assembled the way a flash kernel forces.

    Block 0 comes back from the kernel as `(out0, lse0)` -- an output already
    normalized over its own keys, plus that row's log-sum-exp. Each later step
    contributes one key, so its "attention" is the value itself and its lse is
    the single score. Merge by renormalizing every term onto a common
    denominator::

        merged = logsumexp([lse0, lse1, ...])
        out    = out0 * exp(lse0 - merged) + sum_i v_i * exp(lse_i - merged)

    This is where an error would hide: each term is individually plausible and
    the result is still a convex combination, so it looks right while weighting
    the chain entries wrongly.

    A padded row's terms are all `-inf`, so `merged` is too and `lse - merged` is
    NaN. `_finite` replaces those rows' denominator before the subtraction; the
    output is masked to zero either way, which is why leaving it out is invisible
    until the backward pass (`TestMergeNumericGuard`).
    """
    steps = cache_k.size(2)
    scale = 1.0 / math.sqrt(q.size(-1))
    block0 = _step_logits(q, cache_k[:, :, 0], True, valid_mask, doc_ids)
    lse0 = torch.logsumexp(block0, dim=-1)
    out0 = torch.softmax(block0, dim=-1) @ cache_v[:, :, 0]

    lse_terms = [_pad_to_neg_inf(lse0, valid_mask)]
    for i in range(1, steps):
        own = (q * cache_k[:, :, i]).sum(-1) * scale
        lse_terms.append(_pad_to_neg_inf(own, valid_mask))
    merged = _finite(torch.logsumexp(torch.stack(lse_terms), dim=0))

    out = out0 * torch.exp(lse_terms[0] - merged).unsqueeze(-1)
    for i in range(1, steps):
        out = out + cache_v[:, :, i] * torch.exp(lse_terms[i] - merged).unsqueeze(-1)
    return _zero_padded_rows(out, valid_mask)


def _all_steps_causal(q, cache_k, cache_v, valid_mask=None, doc_ids=None):
    """What a single flat cache region gives instead: every step fully causal.

    This is the geometry to avoid -- chain `j` can read tokens drafted at step
    `i` for origins `< j`, which do not exist in any real request's cache. Kept
    so the tests can show the diagonal is load-bearing rather than decorative.
    """
    return _softmax_over_steps(q, cache_k, cache_v, valid_mask, doc_ids, flat=True)


def _unguarded_merge(q, cache_k, cache_v, valid_mask):
    """`_merged_prefix_diagonal` with the `_finite` call left out: the control.

    Written out rather than switched on a flag, so the reference keeps
    AngelSpec's signature and the control cannot silently pick up a fix.
    """
    steps = cache_k.size(2)
    scale = 1.0 / math.sqrt(q.size(-1))
    block0 = _step_logits(q, cache_k[:, :, 0], True, valid_mask, None)
    out0 = torch.softmax(block0, dim=-1) @ cache_v[:, :, 0]
    lse_terms = [_pad_to_neg_inf(torch.logsumexp(block0, dim=-1), valid_mask)]
    for i in range(1, steps):
        own = (q * cache_k[:, :, i]).sum(-1) * scale
        lse_terms.append(_pad_to_neg_inf(own, valid_mask))
    merged = torch.logsumexp(torch.stack(lse_terms), dim=0)

    out = out0 * torch.exp(lse_terms[0] - merged).unsqueeze(-1)
    for i in range(1, steps):
        out = out + cache_v[:, :, i] * torch.exp(lse_terms[i] - merged).unsqueeze(-1)
    return _zero_padded_rows(out, valid_mask)


def _inputs(seed=0, steps=3):
    torch.manual_seed(seed)
    shape = (BATCH, HEADS, steps, SEQ, DIM)
    return (
        torch.randn(BATCH, HEADS, SEQ, DIM),
        torch.randn(shape),
        torch.randn(shape),
    )


HOLE = 2
TAIL = SEQ - 1


def _padded():
    """Row 0 has an interior hole and a padded tail; row 1 is fully valid.

    The hole is not decoration. With padding only at the tail, causality already
    blocks every invalid key from every valid query, so the key-side half of the
    mask is unobservable and a reference that dropped it would still agree with
    AngelSpec's on every right-padded batch.
    """
    valid = torch.ones(BATCH, SEQ, dtype=torch.bool)
    valid[0, HOLE] = False
    valid[0, TAIL] = False
    return valid


def _documents():
    return torch.tensor([[0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 1, 2]])


@pytest.mark.unit
@pytest.mark.fast
class TestLseMerge:
    @pytest.mark.parametrize("steps", [1, 2, 3, 4])
    def test_merged_equals_the_dense_reference(self, steps):
        q, cache_k, cache_v = _inputs(steps=steps)
        torch.testing.assert_close(
            _merged_prefix_diagonal(q, cache_k, cache_v),
            _dense_prefix_diagonal(q, cache_k, cache_v),
            atol=1e-6,
            rtol=1e-5,
        )

    @pytest.mark.parametrize("steps", [1, 3])
    def test_merged_equals_the_dense_reference_under_padding_and_documents(self, steps):
        """The masks compose: neither form may take a short cut under either."""
        q, cache_k, cache_v = _inputs(steps=steps)
        for kwargs in (
            {"valid_mask": _padded()},
            {"doc_ids": _documents()},
            {"valid_mask": _padded(), "doc_ids": _documents()},
        ):
            torch.testing.assert_close(
                _merged_prefix_diagonal(q, cache_k, cache_v, **kwargs),
                _dense_prefix_diagonal(q, cache_k, cache_v, **kwargs),
                atol=1e-6,
                rtol=1e-5,
            )

    def test_depth_zero_is_plain_causal_attention(self):
        """Step 0 needs no diagonal term of its own -- it is already in the row."""
        q, cache_k, cache_v = _inputs(steps=1)
        plain = (
            torch.softmax(_step_logits(q, cache_k[:, :, 0], True, None, None), dim=-1)
            @ cache_v[:, :, 0]
        )
        torch.testing.assert_close(
            _merged_prefix_diagonal(q, cache_k, cache_v), plain, atol=1e-6, rtol=1e-5
        )

    def test_the_merge_is_not_a_plain_average(self):
        """Guard against a merge that happens to look right.

        Weighting the terms equally, or forgetting to renormalize block 0 by
        `exp(lse0 - merged)`, both produce a convex-looking combination that is
        wrong. Neither may reproduce the dense answer.
        """
        q, cache_k, cache_v = _inputs(steps=3)
        dense = _dense_prefix_diagonal(q, cache_k, cache_v)

        naive_mean = (
            _merged_prefix_diagonal(q, cache_k[:, :, :1], cache_v[:, :, :1])
            + cache_v[:, :, 1]
            + cache_v[:, :, 2]
        ) / 3.0
        assert not torch.allclose(naive_mean, dense, atol=1e-3)

        scale = 1.0 / math.sqrt(DIM)
        block0 = _step_logits(q, cache_k[:, :, 0], True, None, None)
        lse_terms = [torch.logsumexp(block0, dim=-1)] + [
            (q * cache_k[:, :, i]).sum(-1) * scale for i in (1, 2)
        ]
        merged = torch.logsumexp(torch.stack(lse_terms), dim=0)
        unrenormalized = torch.softmax(block0, dim=-1) @ cache_v[:, :, 0]
        for i in (1, 2):
            unrenormalized = unrenormalized + cache_v[:, :, i] * torch.exp(
                lse_terms[i] - merged
            ).unsqueeze(-1)
        assert not torch.allclose(unrenormalized, dense, atol=1e-3)


@pytest.mark.unit
@pytest.mark.fast
class TestChainIsolation:
    """Invariant I-6: chain `j` cannot read chain `j-1`'s drafted tokens."""

    @pytest.mark.parametrize("impl", [_dense_prefix_diagonal, _merged_prefix_diagonal])
    def test_perturbing_another_origins_draft_changes_nothing(self, impl):
        q, cache_k, cache_v = _inputs(steps=3)
        probe = 4
        base = impl(q, cache_k, cache_v)

        for step in (1, 2):
            for other in range(SEQ):
                if other == probe:
                    continue
                k_bumped, v_bumped = cache_k.clone(), cache_v.clone()
                k_bumped[:, :, step, other, :] = torch.randn(BATCH, HEADS, DIM) * 5.0
                v_bumped[:, :, step, other, :] = torch.randn(BATCH, HEADS, DIM) * 5.0
                moved = impl(q, k_bumped, v_bumped)
                assert torch.equal(moved[:, :, probe], base[:, :, probe]), (
                    f"query {probe} saw step {step}'s draft for origin {other}"
                )

    def test_its_own_draft_does_move_it(self):
        """The positive control: the diagonal entry must be read."""
        q, cache_k, cache_v = _inputs(steps=3)
        probe = 4
        base = _merged_prefix_diagonal(q, cache_k, cache_v)
        v_bumped = cache_v.clone()
        v_bumped[:, :, 1, probe, :] += 5.0
        moved = _merged_prefix_diagonal(q, cache_k, v_bumped)
        assert not torch.allclose(moved[:, :, probe], base[:, :, probe])

    def test_a_flat_causal_cache_breaks_isolation(self):
        """The geometry a single cache region gives, shown to be different.

        Without this the isolation tests above would pass against any
        implementation that simply ignored the later steps.
        """
        q, cache_k, cache_v = _inputs(steps=3)
        probe = 4
        base = _all_steps_causal(q, cache_k, cache_v)
        v_bumped = cache_v.clone()
        v_bumped[:, :, 1, probe - 1, :] += 5.0  # another origin's drafted token
        assert not torch.allclose(
            _all_steps_causal(q, cache_k, v_bumped)[:, :, probe], base[:, :, probe]
        ), (
            "the flat-cache reference should leak across chains; if it does not, "
            "the isolation tests above prove nothing"
        )
        assert not torch.allclose(
            base, _dense_prefix_diagonal(q, cache_k, cache_v), atol=1e-3
        )


@pytest.mark.unit
@pytest.mark.fast
class TestPaddingAndDocuments:
    """The two key gates AngelSpec's signature carries, and their controls."""

    @pytest.mark.parametrize("impl", [_dense_prefix_diagonal, _merged_prefix_diagonal])
    def test_a_padded_row_is_exactly_zero(self, impl):
        """Not "small", and not the uniform average a fully blocked softmax gives."""
        q, cache_k, cache_v = _inputs(steps=3)
        valid = _padded()
        out = impl(q, cache_k, cache_v, valid_mask=valid)
        assert torch.equal(out[0, :, [HOLE, TAIL]], torch.zeros(HEADS, 2, DIM))
        assert not torch.allclose(out[1], torch.zeros(HEADS, SEQ, DIM))

    @pytest.mark.parametrize("impl", [_dense_prefix_diagonal, _merged_prefix_diagonal])
    def test_a_padded_key_is_not_attended(self, impl):
        """A valid query must not read the interior hole, and the control shows
        the same perturbation does move it once that position is valid."""
        q, cache_k, cache_v = _inputs(steps=3)
        valid = _padded()
        after = HOLE + 1
        k_bumped, v_bumped = cache_k.clone(), cache_v.clone()
        k_bumped[0, :, 0, HOLE, :] = 5.0
        v_bumped[0, :, 0, HOLE, :] = 5.0

        assert torch.equal(
            impl(q, k_bumped, v_bumped, valid_mask=valid)[0, :, after],
            impl(q, cache_k, cache_v, valid_mask=valid)[0, :, after],
        )
        assert not torch.allclose(
            impl(q, k_bumped, v_bumped)[0, :, after],
            impl(q, cache_k, cache_v)[0, :, after],
        ), "without valid_mask the same query does read the hole"

    @pytest.mark.parametrize("impl", [_dense_prefix_diagonal, _merged_prefix_diagonal])
    def test_a_query_cannot_read_the_previous_document(self, impl):
        """The attention-side twin of item 13's label-side gate."""
        q, cache_k, cache_v = _inputs(steps=3)
        docs = _documents()
        first_of_second_doc = 3  # row 0's documents are [0,0,0,1,1,1]
        k_bumped, v_bumped = cache_k.clone(), cache_v.clone()
        k_bumped[0, :, 0, :first_of_second_doc, :] = 5.0
        v_bumped[0, :, 0, :first_of_second_doc, :] = 5.0

        base = impl(q, cache_k, cache_v, doc_ids=docs)
        assert torch.equal(
            impl(q, k_bumped, v_bumped, doc_ids=docs)[0, :, first_of_second_doc],
            base[0, :, first_of_second_doc],
        )
        assert not torch.allclose(
            impl(q, k_bumped, v_bumped)[0, :, first_of_second_doc],
            impl(q, cache_k, cache_v)[0, :, first_of_second_doc],
        ), "without doc_ids the same query does read the earlier document"


@pytest.mark.unit
@pytest.mark.fast
class TestMergeNumericGuard:
    """A padded row's merge denominator is `-inf`, and `lse - (-inf)` is NaN.

    The forward pass hides it: those rows are masked to zero afterwards either
    way. Only the gradient shows it, which is why this is tested on `.grad` and
    not on the output.
    """

    def test_the_forward_hides_it(self):
        q, cache_k, cache_v = _inputs(steps=3)
        valid = _padded()
        assert torch.isfinite(_unguarded_merge(q, cache_k, cache_v, valid)).all()

    def test_gradients_stay_finite_with_padded_rows(self):
        q, cache_k, cache_v = _inputs(steps=3)
        valid = _padded()
        guarded = q.clone().requires_grad_(True)
        _merged_prefix_diagonal(guarded, cache_k, cache_v, valid_mask=valid).sum().backward()
        assert torch.isfinite(guarded.grad).all()

        unguarded = q.clone().requires_grad_(True)
        _unguarded_merge(unguarded, cache_k, cache_v, valid).sum().backward()
        assert torch.isnan(unguarded.grad).any(), (
            "dropping the finite guard should poison the gradient; if it does "
            "not, this test is not pinning anything"
        )

    def test_an_all_padded_row_is_finite_too(self):
        """The case AngelSpec's comment names: a shard with no valid tokens."""
        q, cache_k, cache_v = _inputs(steps=3)
        none_valid = torch.zeros(BATCH, SEQ, dtype=torch.bool)
        out = _merged_prefix_diagonal(q, cache_k, cache_v, valid_mask=none_valid)
        assert torch.equal(out, torch.zeros_like(out))
