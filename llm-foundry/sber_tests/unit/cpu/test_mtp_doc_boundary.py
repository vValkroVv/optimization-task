"""No MTP supervision crosses a document boundary.

`data/packing.py:119` sets `add_on['labels'][0] = -100`, which protects exactly
one position per seam. That is enough for the main head, whose target at `j` is
`labels[j+1]`, and for nothing else: `_build_shifted_labels` rolls once more per
head, so head `h` reads `labels[j+h+1]` and the window that can fall past the
seam widens by one per head.

Ungated, head `h` is trained on `h` cross-document targets per internal seam --
`D(D+1)/2` per seam in total, 6 at `D=3`, weighted toward depth. Attention is
gated correctly (`cu_seqlens` comes from `position_ids`), so those targets are
not merely wrong, they are unreachable from the visible context.

The arithmetic these tests pin, for a row holding doc A at positions
`0 .. L-1` and doc B at `L .. `:

    ungated   head h masked at the seam for j == L-1-h only   (the -100 sentinel)
    gated     head h masked at the seam for j in [L-1-h, L-1] (h+1 positions)
    so the drop is exactly h supervised positions per seam per head

Everything here is CPU: it is label arithmetic, no kernel involved. The
production `same_doc_next_mask` / `_build_shifted_labels` are bound, not copied.

Ported from samoylovlesha/angelspec_eagle (spec item 13 / I-3 in
recurrent-mtp-spec). Reference: AngelSpec @ d3412be,
`angelspec/models/mtp.py:284` (`full_mask = full_mask * full_same_doc_next`
in the TTT shift loop).
"""

import pytest
import torch

from llmfoundry.models.giga_mix.modelling_giga_mix import GigaMixForCausalLM
from sber_tests.unit.mtp_harness import DraftLossHarness

MTP_NUM = 3
IGNORE = -100


def _LabelHarness(mtp_predictor_num: int = MTP_NUM):
    return DraftLossHarness(64, mtp_predictor_num)


def _packed_row(doc_lens, vocab_size=64):
    """One packed row of `doc_lens`, as the collator hands it to the model.

    Returns `(position_ids, shift_labels)`. Labels are the token ids, with the
    first label of every non-initial document set to `ignore_index` -- that is
    `packing.py:119`, reproduced rather than imported so the test states what it
    assumes about the collator.
    """
    total = sum(doc_lens)
    position_ids = torch.cat([torch.arange(n) for n in doc_lens]).view(1, total)
    labels = (torch.arange(total).view(1, total) % (vocab_size - 1)) + 1
    start = 0
    for n in doc_lens[:-1]:
        start += n
        labels[:, start] = IGNORE  # packing.py:119, the one protected position
    shift_labels = labels.roll(shifts=(-1,), dims=(1,)).contiguous()
    shift_labels[:, -1] = IGNORE
    return position_ids, shift_labels


@pytest.mark.unit
@pytest.mark.fast
class TestSameDocNextMask:
    def test_true_inside_a_document_false_at_every_seam_and_at_the_tail(self):
        position_ids, _ = _packed_row([5, 4, 3])
        mask = GigaMixForCausalLM.same_doc_next_mask(position_ids)

        assert mask.shape == position_ids.shape
        assert mask.dtype == torch.bool
        # Seams sit at the LAST position of each document: 4 and 8. The tail of
        # the row is masked too -- what follows it is not visible here.
        want = torch.ones(1, 12, dtype=torch.bool)
        want[:, [4, 8, 11]] = False
        assert torch.equal(mask, want)

    def test_none_position_ids_gives_none(self):
        assert GigaMixForCausalLM.same_doc_next_mask(None) is None


@pytest.mark.unit
@pytest.mark.fast
class TestGatedLabelShift:
    @staticmethod
    def _both(doc_lens, mtp_predictor_num=MTP_NUM):
        harness = _LabelHarness(mtp_predictor_num)
        position_ids, shift_labels = _packed_row(doc_lens)
        mask = harness.same_doc_next_mask(position_ids)
        return (
            harness,
            harness._build_shifted_labels(shift_labels, None),
            harness._build_shifted_labels(shift_labels, mask),
        )

    def test_no_position_ids_falls_back_to_the_ungated_shift(self):
        """`None` reaches the gate only when the batch carries no `position_ids`.

        There is no flag to turn the gate off, so this is the one route to an
        ungated chain: a loader without `varlen_input`. It must reproduce the
        plain per-depth roll exactly.
        """
        harness = _LabelHarness()
        _, shift_labels = _packed_row([5, 7])
        for gated, plain in zip(
            harness._build_shifted_labels(shift_labels, None),
            [
                shift_labels,
                *[
                    torch.cat(
                        [
                            shift_labels[:, h:],
                            torch.full((1, h), IGNORE, dtype=shift_labels.dtype),
                        ],
                        dim=1,
                    )
                    for h in range(1, MTP_NUM + 1)
                ],
            ],
        ):
            assert torch.equal(gated, plain)

    def test_head_h_has_h_plus_1_masked_positions_at_the_seam(self):
        """The Done-when condition, stated at the seam of a two-document row."""
        first = 6
        _, plain, gated = self._both([first, 7])

        for h in range(1, MTP_NUM + 1):
            seam_plain = [
                j for j in range(first) if plain[h][0, j].item() == IGNORE
            ]
            seam_gated = [
                j for j in range(first) if gated[h][0, j].item() == IGNORE
            ]
            # Ungated: only the position whose target is the -100 sentinel.
            assert seam_plain == [first - 1 - h], (
                f"head {h} ungated masks {seam_plain} inside doc A"
            )
            # Gated: that one plus every position whose target lands in doc B.
            assert seam_gated == list(range(first - 1 - h, first)), (
                f"head {h} should mask {h + 1} positions at the seam, got {seam_gated}"
            )
            assert len(seam_gated) == h + 1

    def test_gate_masks_exactly_the_cross_document_targets(self):
        """Position by position, against the definition rather than a count."""
        doc_lens = [5, 4, 6]
        harness, _, gated = self._both(doc_lens)
        bounds = torch.cumsum(torch.tensor(doc_lens), 0).tolist()

        def doc_of(j):
            return next(d for d, end in enumerate(bounds) if j < end)

        total = sum(doc_lens)
        for h in range(1, MTP_NUM + 1):
            for j in range(total):
                target = j + h + 1
                same_doc = target < total and doc_of(target) == doc_of(j)
                is_masked = gated[h][0, j].item() == IGNORE
                assert is_masked == (not same_doc), (
                    f"head {h} at j={j}: target {target} same_doc={same_doc} "
                    f"but masked={is_masked}"
                )

    def test_interior_positions_keep_their_supervision(self):
        """A larger drop than predicted would mean the gate over-gates."""
        harness, plain, gated = self._both([9, 8])
        for h in range(1, MTP_NUM + 1):
            # Everything the ungated version supervises and the gate keeps must
            # hold the same target value -- the gate only ever removes.
            kept = gated[h] != IGNORE
            assert torch.equal(gated[h][kept], plain[h][kept])
            assert bool((plain[h][kept] != IGNORE).all())

    @pytest.mark.parametrize("doc_lens", [[8, 8], [5, 5, 6], [6, 6, 6, 6]])
    def test_supervised_count_drops_by_h_per_seam_and_no_more(self, doc_lens):
        # Every document longer than D, so no seam window swallows a whole one --
        # see `test_documents_shorter_than_the_head_depth_lose_the_head` for that.
        assert min(doc_lens) > MTP_NUM
        harness, plain, gated = self._both(doc_lens)
        num_seams = len(doc_lens) - 1

        counts_plain = harness._mtp_num_tokens(
            harness._compute_valid_counts(torch.stack(plain, dim=0))
        )
        counts_gated = harness._mtp_num_tokens(
            harness._compute_valid_counts(torch.stack(gated, dim=0))
        )
        # `_mtp_num_tokens` is per head (batch summed); over all heads the
        # predicted drop is num_seams * D(D+1)/2 -- the corrupted-position count.
        want_drop = num_seams * MTP_NUM * (MTP_NUM + 1) // 2
        assert int((counts_plain - counts_gated).sum().item()) == want_drop

    def test_per_head_drop_is_h_per_seam(self):
        doc_lens = [7, 6, 5]
        harness, plain, gated = self._both(doc_lens)
        num_seams = len(doc_lens) - 1
        for h in range(1, MTP_NUM + 1):
            dropped = int(
                (plain[h] != IGNORE).sum() - (gated[h] != IGNORE).sum()
            )
            assert dropped == h * num_seams, (
                f"head {h} should lose {h} positions per seam, lost {dropped}"
            )

    def test_documents_shorter_than_the_head_depth_lose_the_head(self):
        """The edge the gate makes real: head `h` needs documents of length > h.

        A row of length-4 documents supervises head 3 nowhere -- its target is
        always in the next document. Ungated it looked supervised at 9 of 16
        positions, on targets it could not reach. Watch this if `doc_lens` ever
        approach `mtp_predictor_num`; `_mtp_num_tokens` clamps the count to 1, so
        an empty head divides by 1 rather than raising.
        """
        harness, plain, gated = self._both([4, 4, 4, 4])
        assert int((plain[MTP_NUM] != IGNORE).sum()) == 9
        assert int((gated[MTP_NUM] != IGNORE).sum()) == 0
        # Head 1 still has room: its target is one position further only.
        assert int((gated[1] != IGNORE).sum()) == 8

    def test_single_document_row_is_untouched(self):
        """No seam, no change -- the gate must not eat the sequence tail twice."""
        harness, plain, gated = self._both([12])
        for h in range(MTP_NUM + 1):
            assert torch.equal(plain[h], gated[h])


@pytest.mark.unit
@pytest.mark.fast
class TestAccLenPopulation:
    """The acc_len monitor builds its own labels -- so it takes the same gate,
    keeping its population the one the loss trains on.

    `_finalize_mtp_acc_len` stores `full_valid`: positions whose whole depth-D
    window has targets. A window that crosses a seam always contains the
    packing sentinel (`packing.py:119`), so the *intersection* population is the
    same gated or not -- the gate matters for the per-head label chains (pinned
    in `TestGatedLabelShift`), and the monitor must agree with those chains,
    not rebuild its own convention.
    """

    @staticmethod
    def _results(doc_lens, gated):
        harness = _LabelHarness()
        position_ids, shift_labels = _packed_row(doc_lens)
        mask = harness.same_doc_next_mask(position_ids) if gated else None
        # The argmaxes only feed the match counts; the population does not
        # depend on them, so anything consistently shaped will do.
        argmaxes = [torch.zeros_like(shift_labels) for _ in range(MTP_NUM + 1)]
        harness._finalize_mtp_acc_len(argmaxes, shift_labels, mask)
        return harness._mtp_acc_len_results

    def test_full_valid_is_the_document_oracle_and_gate_invariant(self):
        """`full_valid` counts positions whose depth-D window stays inside one
        document: `max(0, L - D - 1)` per document of length L. The packing
        sentinel already excludes every crossing window from the intersection,
        so the count must not depend on the gate.
        """
        doc_lens = [8, 7, 9]
        plain = self._results(doc_lens, gated=False)["full_valid"].item()
        gated = self._results(doc_lens, gated=True)["full_valid"].item()

        want = sum(max(0, n - MTP_NUM - 1) for n in doc_lens)
        assert gated == want, (gated, want)
        assert plain == want, (plain, want)

    def test_full_valid_matches_the_gated_label_chain(self):
        """The monitor's population is the label chain's, not its own."""
        doc_lens = [8, 7, 9]
        harness = _LabelHarness()
        position_ids, shift_labels = _packed_row(doc_lens)
        mask = harness.same_doc_next_mask(position_ids)
        chain = harness._build_shifted_labels(shift_labels, mask)

        full = shift_labels != IGNORE
        for h in range(1, MTP_NUM + 1):
            full &= chain[h] != IGNORE

        got = self._results(doc_lens, gated=True)["full_valid"].item()
        assert got == int(full.sum().item())
