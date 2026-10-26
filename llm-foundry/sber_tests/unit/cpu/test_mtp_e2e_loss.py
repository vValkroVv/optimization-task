"""End-to-end accepted-length loss (spec item 6), direct form.

    prefix[m] = prod_{i<=m} alpha_i        the chain survives to depth m
    E[L]      = sum_m prefix[m]
    loss      = 1 - E[L] / mtp_predictor_num                      (paper eq. 9)

`alpha` is the per-token acceptance the TV loss already produces, so this costs
no extra vocabulary pass. Three things make it more than a `cumprod`, and each
has a test here:

  1. **The frame.** The draft losses roll each head forward by `h` so one target
     projection serves every head. Row `u` of head `h` therefore belongs to chain
     origin `u - h`; a `cumprod` over the head axis as it stands would multiply
     acceptances from `D` different chains. Each head is rolled back by `-h`
     first. `test_chain_is_taken_at_one_origin` is the assertion that matters --
     it fails if the roll-back is dropped.
  2. **Chunking.** The chunk loop returns reduced sums, but the product needs
     alpha per token for the whole sequence before it can form.
  3. **The mask.** Invalid positions get alpha 1.0, not 0. A 0 would kill every
     chain downstream of a masked position; the mask is applied to the finished
     loss instead.
  4. **The population.** The chain is the DRAFT heads, so an origin counts on
     `mtp_labels[1:]` and the main head's own label is not one of the terms.
     `test_population_is_the_draft_chain_not_the_main_head` is the assertion;
     the fully-supervised rows the other cases use cannot tell the two apart.

Plus the rho floor, which is the difference between the deep heads receiving
gradient and quietly receiving none.

Ported from samoylovlesha/angelspec_eagle (spec item 6 in recurrent-mtp-spec).
Reference: AngelSpec @ d3412be, `angelspec/models/mtp.py:518` (`_e2e_loss`) and
`models/ops/loss.py:383` (`new_ones` for alpha at invalid positions).
"""

import pytest
import torch

from llmfoundry.models.giga_mix.configuration_giga_mix import GigaMixConfig
from llmfoundry.models.giga_mix.modelling_giga_mix import GigaMixForCausalLM
from sber_tests.unit.mtp_harness import DraftLossHarness, packed_labels

MTP_NUM = 3
SEQ_LEN = 12
VOCAB = 8


def _Harness(e2e_alpha_floor=0.0, trained=True):
    return DraftLossHarness(
        VOCAB,
        MTP_NUM,
        weights={"e2e_tv": 1.0 if trained else 0.0},
        mtp_position_weight_beta=1.0,
        mtp_ce_loss_weight=0.0,
        use_e2e_tv_loss=True,
        e2e_tv_loss_weight=1.0 if trained else 0.0,
        e2e_alpha_floor=e2e_alpha_floor,
    )


def _case(seed=0, e2e_alpha_floor=0.0, trained=True, scale=2.0, mask_span=None):
    torch.manual_seed(seed)
    harness = _Harness(e2e_alpha_floor=e2e_alpha_floor, trained=trained)
    shift_labels = packed_labels(SEQ_LEN, VOCAB)
    if mask_span is not None:
        shift_labels[:, mask_span[0] : mask_span[1]] = harness.ignore_index
    mtp_labels = harness._build_shifted_labels(shift_labels)
    valid_counts = harness._compute_valid_counts(torch.stack(mtp_labels, dim=0))
    combined = tuple(
        torch.randn(1, SEQ_LEN, VOCAB) * scale for _ in range(MTP_NUM + 1)
    )
    return harness, shift_labels, mtp_labels, valid_counts, combined


def _alpha_by_origin(harness, mtp_labels, combined):
    """`alpha[h - 1, t]`: head h's acceptance at position t, origin-indexed.

    Written as an explicit index loop rather than with rolls, so it states the
    pairing directly: head `h` at position `t` is judged against the target at
    `t + h`, because that is the position whose next-token distribution head `h`
    is trained to reproduce (invariant I-1). The production code reaches the same
    place by rolling the drafts forward by `h` and the alphas back by `-h`; this
    is the independent statement of where it must land.
    """
    p = torch.softmax(combined[0], dim=-1)[0]
    rows = []
    for h in range(1, MTP_NUM + 1):
        q = torch.softmax(combined[h], dim=-1)[0]
        m = torch.ones(SEQ_LEN)
        for t in range(SEQ_LEN - h):
            m[t] = torch.minimum(p[t + h], q[t]).sum()
        valid = mtp_labels[h][0] != harness.ignore_index
        rows.append(torch.where(valid, m, torch.ones_like(m)))
    return torch.stack(rows)


def _expected(harness, mtp_labels, combined, rho=0.0, first_head=1):
    """`first_head=1` is the shipped population; 0 adds the main head's labels."""
    alpha = _alpha_by_origin(harness, mtp_labels, combined)
    valid_all = mtp_labels[first_head][0] != harness.ignore_index
    for h in range(first_head + 1, MTP_NUM + 1):
        valid_all = valid_all & (mtp_labels[h][0] != harness.ignore_index)
    if rho:
        alpha = rho + (1.0 - rho) * alpha
    prefix = torch.cumprod(alpha, dim=0)
    denom = valid_all.sum().clamp_min(1).to(torch.float32)
    per_depth = 1.0 - (prefix * valid_all).sum(dim=1) / denom
    return per_depth


@pytest.mark.unit
@pytest.mark.fast
class TestE2EChainLoss:
    @pytest.mark.parametrize("chunk_size", [5, 2048])
    @pytest.mark.parametrize("rho", [0.0, 0.5])
    def test_matches_the_longhand_chain(self, chunk_size, rho):
        harness, shift_labels, mtp_labels, valid_counts, combined = _case(
            e2e_alpha_floor=rho
        )
        loss, per_depth = harness._compute_draft_losses(
            combined,
            shift_labels,
            ["e2e_tv"],
            mtp_labels,
            valid_counts,
            chunk_size=chunk_size,
        )["e2e_tv"]
        want = _expected(harness, mtp_labels, combined, rho=rho)
        torch.testing.assert_close(per_depth, want)
        torch.testing.assert_close(loss, want.mean())

    def test_population_is_the_draft_chain_not_the_main_head(self):
        """An origin counts when every DRAFT head has a target -- main head or not.

        AngelSpec's `valid_all` is a product over TTT steps, and their step `k`
        is our head `k + 1`. Item 6 has the index derivation and what the rival
        population costs; this pins the population the code ships.

        Every other case here runs a fully supervised row, where the two
        populations coincide and `mtp_labels` in place of `mtp_labels[1:]` passes
        the whole suite.
        """
        harness, shift_labels, mtp_labels, valid_counts, combined = _case(
            mask_span=(4, 7)
        )
        drafts_valid = torch.stack(
            [mtp_labels[h][0] != harness.ignore_index for h in range(1, MTP_NUM + 1)]
        ).all(dim=0)
        contested = (mtp_labels[0][0] == harness.ignore_index) & drafts_valid
        assert contested.any(), "no origin separates the two populations"

        _, per_depth = harness._compute_draft_losses(
            combined, shift_labels, ["e2e_tv"], mtp_labels, valid_counts
        )["e2e_tv"]
        torch.testing.assert_close(
            per_depth, _expected(harness, mtp_labels, combined, first_head=1)
        )
        assert not torch.allclose(
            per_depth, _expected(harness, mtp_labels, combined, first_head=0)
        )

    def _chain_grads(self, e2e_direct, rho, alpha_by_depth):
        """`d loss / d alpha` per depth, alpha given in the ORIGIN frame."""
        harness = _Harness(e2e_alpha_floor=rho)
        harness.config.e2e_direct = e2e_direct
        mtp_labels = harness._build_shifted_labels(packed_labels(SEQ_LEN, VOCAB))
        leaves = [
            torch.full((1, SEQ_LEN), v, requires_grad=True) for v in alpha_by_depth
        ]
        rolled = [leaves[h - 1].roll(shifts=(h,), dims=(1,)) for h in range(1, MTP_NUM + 1)]
        loss, _ = harness._e2e_chain_loss(rolled, mtp_labels)
        loss.backward()
        return loss.detach(), [x.grad.abs().max() for x in leaves]

    def test_detached_weights_take_the_direct_form_s_own_step(self):
        """`W_m` is `-d(direct)/d alpha_m`, so at rho=0 the two forms agree.

        Asserted on the gradient, not the loss: the two do not share a scale (a
        head accepting nothing reads 1.0 direct against 1/D here), and the point
        of keeping the branch is that the step is the same one.
        """
        alphas = [0.8, 0.55, 0.3]
        _, g_direct = self._chain_grads(True, 0.0, alphas)
        _, g_wm = self._chain_grads(False, 0.0, alphas)
        for a, b in zip(g_direct, g_wm):
            torch.testing.assert_close(a, b)

    def test_the_weights_carry_no_gradient(self):
        """The floor must reach the weights only, never the objective.

        Differentiating through `W` as well would put rho back inside the loss,
        which is the whole thing this branch exists to avoid. The negative
        control is the same arithmetic without `.detach()`: it must disagree, or
        the assertion above it proves nothing.
        """
        harness = _Harness(e2e_alpha_floor=0.5)
        harness.config.e2e_direct = False
        mtp_labels = harness._build_shifted_labels(packed_labels(SEQ_LEN, VOCAB))
        valid_all = (torch.stack(mtp_labels[1:]) != harness.ignore_index).all(dim=0)
        mask = valid_all.unsqueeze(0).float()
        denom = valid_all.sum().float().clamp_min(1.0)

        def longhand(detach):
            leaves = [torch.full((1, SEQ_LEN), v, requires_grad=True)
                      for v in (0.8, 0.55, 0.3)]
            alpha = torch.stack([x.expand(1, SEQ_LEN) for x in leaves])
            smoothed = 0.5 + 0.5 * alpha
            det = smoothed.detach() if detach else smoothed
            pre = torch.cat([torch.ones_like(det[:1]), torch.cumprod(det[:-1], dim=0)])
            s_terms = [torch.ones_like(det[0])] * MTP_NUM
            for m in range(MTP_NUM - 2, -1, -1):
                s_terms[m] = 1.0 + det[m + 1] * s_terms[m + 1]
            w = pre * torch.stack(s_terms)
            contrib = w * (1.0 - alpha)
            ((contrib * mask).sum(dim=(1, 2)) / denom).mean().backward()
            return [x.grad.abs().max() for x in leaves]

        _, produced = self._chain_grads(False, 0.5, [0.8, 0.55, 0.3])
        for a, b in zip(produced, longhand(detach=True)):
            torch.testing.assert_close(a, b)
        live = longhand(detach=False)
        assert not all(
            torch.allclose(a, b) for a, b in zip(produced, live)
        ), "detaching the weights changed nothing -- the control is vacuous"

    def test_the_floor_reweights_depths_without_moving_the_optimum(self):
        """What the branch buys, and what it leaves alone.

        Buys: the deepest head's share of the gradient rises with rho, which is
        the reason to want a floor at all. Leaves alone: a perfect chain still
        scores 0 whatever rho is, because rho never enters `1 - alpha`.
        """
        weak = [0.05, 0.6, 0.6]
        share = []
        for rho in (0.0, 0.5):
            _, g = self._chain_grads(False, rho, weak)
            share.append((g[-1] / g[0]).item())
        assert share[1] > share[0], share

        for rho in (0.0, 0.5):
            loss, _ = self._chain_grads(False, rho, [1.0, 1.0, 1.0])
            torch.testing.assert_close(loss, torch.zeros(()))

    def test_chain_is_taken_at_one_origin(self):
        """The frame check, and the one that catches a dropped roll-back.

        Perturbing head 2's hidden at position `t` must move the chain at origin
        `t` and leave every other origin alone. If the per-head roll-back is
        missing, the disturbance lands at origin `t - 2` instead.
        """
        harness, shift_labels, mtp_labels, valid_counts, combined = _case()

        def prefix_by_origin(hidden):
            alpha = _alpha_by_origin(harness, mtp_labels, hidden)
            return torch.cumprod(alpha.clamp(0.0, 1.0), dim=0)

        probe = 4
        bumped = list(combined)
        bumped[2] = combined[2].clone()
        # A single channel: adding a constant to every logit leaves the softmax
        # unchanged and the test would pass vacuously.
        bumped[2][:, probe, 0] += 12.0

        base = prefix_by_origin(combined)
        moved = prefix_by_origin(tuple(bumped))
        changed = (~torch.isclose(base, moved, atol=1e-6)).any(dim=0).nonzero().flatten()
        assert changed.tolist() == [probe], (
            f"head 2's position {probe} should move exactly origin {probe}, "
            f"moved {changed.tolist()}"
        )

        # And the production path agrees with that origin-indexed reference.
        for hidden in (combined, tuple(bumped)):
            got = harness._compute_draft_losses(
                hidden, shift_labels, ["e2e_tv"], mtp_labels, valid_counts
            )["e2e_tv"][1]
            torch.testing.assert_close(got, _expected(harness, mtp_labels, hidden))

    def test_perfect_drafts_give_zero_loss(self):
        """alpha == 1 everywhere means the chain always survives: E[L] == D.

        "Perfect" is head `h` reproducing the target at `t + h`, not at `t` --
        the pairing above. A head that copied the target verbatim would be wrong
        by exactly `h` positions and this would not come out at zero.
        """
        harness, shift_labels, mtp_labels, valid_counts, _ = _case()
        target = torch.randn(1, SEQ_LEN, VOCAB) * 2.0
        combined = (target,) + tuple(
            target.roll(shifts=(-h,), dims=(1,)) for h in range(1, MTP_NUM + 1)
        )
        loss, per_depth = harness._compute_draft_losses(
            combined, shift_labels, ["e2e_tv"], mtp_labels, valid_counts
        )["e2e_tv"]
        torch.testing.assert_close(per_depth, torch.zeros(MTP_NUM), atol=1e-5, rtol=0)
        torch.testing.assert_close(loss, torch.tensor(0.0), atol=1e-5, rtol=0)

    def test_per_depth_is_non_decreasing(self):
        """`1 - prefix[m]`: a chain cannot become more likely to survive deeper."""
        harness, shift_labels, mtp_labels, valid_counts, combined = _case()
        per_depth = harness._compute_draft_losses(
            combined, shift_labels, ["e2e_tv"], mtp_labels, valid_counts
        )["e2e_tv"][1]
        assert torch.all(per_depth[1:] >= per_depth[:-1] - 1e-6), per_depth

    def test_masked_tail_does_not_saturate_the_loss(self):
        """Every packed row has a tail that is masked at every depth.

        Note on the 1.0 fill: alpha is filled with 1.0 rather than 0 where a head
        has no target, per the reference. Under our mask that is *defensive, not
        load-bearing* -- `valid_all` is the intersection over heads, so any origin
        where some head is invalid is dropped from the loss whole, and the chain
        product never crosses a masked factor. A 0 fill produces the same numbers
        here. It would stop doing so the moment the mask became per-depth, which
        is why the safe form is kept.
        """
        harness, shift_labels, mtp_labels, valid_counts, combined = _case()
        per_depth = harness._compute_draft_losses(
            combined, shift_labels, ["e2e_tv"], mtp_labels, valid_counts
        )["e2e_tv"][1]
        assert torch.all(per_depth < 1.0), per_depth

    def test_rho_floor_bounds_the_chain_from_below(self):
        """The failure mode the floor exists for, stated as a bound.

        With drafts far from the target, alpha is small and `prefix` collapses:
        the deep heads' gradients vanish. rho keeps every factor at or above rho,
        so `prefix[m] >= rho ** (m + 1)`.
        """
        rho = 0.5
        harness, shift_labels, mtp_labels, valid_counts, combined = _case(
            e2e_alpha_floor=rho, scale=12.0
        )
        per_depth = harness._compute_draft_losses(
            combined, shift_labels, ["e2e_tv"], mtp_labels, valid_counts
        )["e2e_tv"][1]
        for m, value in enumerate(per_depth.tolist()):
            assert value <= 1.0 - rho ** (m + 1) + 1e-6, (m, value)

        # Same batch, floor off: the chain must sit strictly lower.
        plain, plain_labels, plain_mtp, plain_counts, plain_hidden = _case(
            e2e_alpha_floor=0.0, scale=12.0
        )
        unfloored = plain._compute_draft_losses(
            plain_hidden, plain_labels, ["e2e_tv"], plain_mtp, plain_counts
        )["e2e_tv"][1]
        assert torch.all(per_depth <= unfloored + 1e-6)
        assert torch.any(per_depth < unfloored - 1e-3), (
            "rho=0.5 should have changed the chain on badly matched drafts"
        )

    # `draft_loss_chunk_size` is a memory knob people lower to fit a run, and at
    # `4` the alpha of several checkpointed chunks is held with a graph across the
    # loop before the product forms. The gradient must not depend on where those
    # boundaries fall. (The value equality is pinned separately by
    # `test_matches_the_longhand_chain`, which parametrizes the same knob.)
    @pytest.mark.parametrize("chunk_size", [4, 2048])
    def test_gradient_reaches_every_depth_including_the_last(self, chunk_size):
        """The floor's real job: deep heads must not silently stop learning."""
        rho = 0.5
        harness, shift_labels, mtp_labels, valid_counts, _ = _case(
            e2e_alpha_floor=rho, scale=12.0
        )
        torch.manual_seed(1)
        combined = [
            torch.randn(1, SEQ_LEN, VOCAB, requires_grad=True)
            for _ in range(MTP_NUM + 1)
        ]
        loss = harness._compute_draft_losses(
            combined,
            shift_labels,
            ["e2e_tv"],
            mtp_labels,
            valid_counts,
            chunk_size=chunk_size,
        )["e2e_tv"][0]
        loss.backward()
        assert combined[0].grad is None, "the target must stay detached"
        for h in range(1, MTP_NUM + 1):
            assert combined[h].grad is not None, f"head {h}"
            assert torch.any(combined[h].grad != 0), f"head {h} got no gradient"

    def test_e2e_takes_no_position_weighting(self):
        """`beta` must not reach it: the depth weighting is the chain itself."""
        assert "e2e_tv" not in GigaMixForCausalLM.DRAFT_LOSS_BETA_ATTR
        harness, shift_labels, mtp_labels, valid_counts, combined = _case()
        harness.config.mtp_position_weight_beta = 0.3  # would skew a weighted mean
        loss, per_depth = harness._compute_draft_losses(
            combined, shift_labels, ["e2e_tv"], mtp_labels, valid_counts
        )["e2e_tv"]
        torch.testing.assert_close(loss, per_depth.mean())

    def test_logging_only_weight_takes_the_no_grad_path(self):
        harness, shift_labels, mtp_labels, valid_counts, _ = _case(trained=False)
        combined = tuple(
            torch.randn(1, SEQ_LEN, VOCAB, requires_grad=True)
            for _ in range(MTP_NUM + 1)
        )
        loss = harness._compute_draft_losses(
            combined, shift_labels, ["e2e_tv"], mtp_labels, valid_counts
        )["e2e_tv"][0]
        assert loss.grad_fn is None

    def test_runs_alongside_the_per_step_losses(self):
        """e2e stacks on a computed per-step loss rather than replacing it."""
        harness, shift_labels, mtp_labels, valid_counts, combined = _case()
        harness.tv_loss_weight = 1.0
        out = harness._compute_draft_losses(
            combined, shift_labels, ["tv", "e2e_tv"], mtp_labels, valid_counts
        )
        assert set(out) == {"tv", "e2e_tv"}
        torch.testing.assert_close(
            out["e2e_tv"][1], _expected(harness, mtp_labels, combined)
        )

    @pytest.mark.parametrize("bad", [-0.1, 1.0, 1.5])
    def test_out_of_range_floor_is_rejected_at_config_time(self, bad):
        with pytest.raises(ValueError, match="e2e_alpha_floor must be in"):
            GigaMixConfig(
                use_mtp=True,
                mtp_predictor_num=MTP_NUM,
                e2e_alpha_floor=bad,
                vocab_size=VOCAB,
                hidden_size=VOCAB,
                num_attention_heads=2,
                num_key_value_heads=2,
                init_device="cpu",
            )
