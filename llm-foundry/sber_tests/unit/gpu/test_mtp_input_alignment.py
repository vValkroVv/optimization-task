"""Invariant I-1 -- the MTP input alignment contract (spec item 14).

Three groups, because the invariant is enforced by code in two files and one of
the two failure modes needs a real attention kernel to exercise:

  A1. Forward wiring, CPU (`llmfoundry/models/layers/mtp.py`) --
      `LlamaDecoderMTPBlock.forward` seeds the recurrence with an UN-shifted
      draft hidden and left-shifts `input_embeds` exactly once per depth. The
      production `forward` runs verbatim; only the decoder layer is stood in, by
      a strictly causal plain-torch mixer. `TestForwardWiringOnCpu`.

  A2. The same perturbations through the real MLA layer, GPU.
      `TestDraftInputIsUnshifted`, `TestEmbedShiftIsOnePerDepth`. What A1 cannot
      reach is the causality of `flash_attn_varlen_func` itself; that is the
      only thing A2 adds, and the only reason a GPU is needed here.

  B. Loss pairing (`llmfoundry/models/giga_mix/modelling_giga_mix.py`) -- the
     forward roll in `_compute_draft_losses` pairs the same (target, draft) rows
     that `_build_shifted_labels` pairs. `TestDraftLossPairing`. CPU.

What the invariant is for, and what it costs when it breaks, is spec I-1.

Reference: AngelSpec @ d3412be, `angelspec/training/mtp_trainer.py:39`
(`align_mtp_inputs` -- its docstring is the contract) and
`angelspec/models/mtp.py:144` (`_step_loss`).


THE TWO ROLL ARITHMETICS, SIDE BY SIDE
--------------------------------------
Notation: `token[j]` is the input at position j; `labels[j] == token[j]`
(the collator hands us labels aligned to inputs, prompt positions already
`ignore_index`). D == `mtp_predictor_num`. Head index h runs 1..D; depth index
k == h-1 is the 0-based index into `mtp_layers` / `mtp_norms`.

Labels -- `modelling_giga_mix.py:1757` then `_build_shifted_labels:824`:

    shift_labels[j]   = labels[j+1]                       one roll, tail = ignore
    mtp_labels[h][j]  = shift_labels[j+h] = labels[j+h+1] h more rolls

Forward -- `mtp.py:341` inside `LlamaDecoderMTPBlock.forward`:

    depth k input_embeds[j] = embed(token[j+k+1])         k+1 cumulative rolls
    depth k input hidden    = mtp_h, seeded UN-shifted with raw_h[j]

    so head h == k+1 at position j is fed token[j+h] and predicts token[j+h+1],
    which is exactly mtp_labels[h][j].                                    (1)

Draft losses -- `_compute_draft_losses:1090`:

    target_flat[u]    = combined_hidden[0][u]             NOT rolled
    drafts_flat[h][u] = combined_hidden[h][u-h]           roll(shifts=+h)
    valids_flat[h][u] = (mtp_labels[h][u-h] != ignore)    same roll

    Substituting u = t+h: row u pairs draft position t with target position
    t+h. The main head at t+h predicts shift_labels[t+h] = labels[t+h+1];
    head h at t predicts mtp_labels[h][t] = labels[t+h+1]. Same token.      (2)

(1) and (2) agree, so the teacher distribution p that head h at position t is
matched against is the one whose argmax target is head h's own CE label. This is
AngelSpec's `target_hidden_states_padded[:, idx : idx+seq_length]` written the
other way round: they roll the target forward per depth (leaving the drafts in
place), we roll the drafts forward per depth (leaving the target in place), so a
single target projection serves every head.

Wrapped rows: `roll` is cyclic, so rows u < h carry draft position S-(h-u).
Those are dropped because `mtp_labels[h]` is `ignore_index` in its last h
positions, and the mask is rolled with the same shift.

Masking is on the PREDICTED token, not the input token: `mtp_labels[h][j]`
indexes `labels[j+h+1]`, i.e. AngelSpec's "loss_mask left-shifted TWICE" at
h = 1, one more per depth. `_finalize_mtp_acc_len`'s `full_valid:748` is
`shift_labels != ignore_index`, also a predicted position.

A third site agrees, without being asserted on here:
`_finalize_mtp_acc_len:757` takes `base_pred.roll(shifts=-h)`, so row j holds
the main head's argmax at j+h and is compared against head h's argmax at j --
the same (t, t+h) pairing as (2), reached by rolling the base backwards instead
of the draft forwards.

The two hidden tensors, by role rather than by name -- both are called
`hidden_states` in `GigaMixModel.forward`:

    :510  arg to `mtp_block`, BEFORE `self.norm`   -> draft input  (raw_h[j])
    :552  after `self.norm`, becomes
          `combined_hidden[0]`                     -> teacher / target hidden

NOT covered here (separate items, separate invariants):
  * I-3, the document-boundary widening at seams -- item 13.
  * I-2, the train/serve normalization count -- item 12. Under
    `draft_input_postnorm` the block receives the post-`model.norm` state, which
    is what serving hands the head; the shift arithmetic below is the same
    either way, because the norm is elementwise in the position axis. Item 12's
    business, not I-1's.
  * I-5/I-6, draft attention geometry -- item 10.

Item 14's "Done when" asks for this derivation as a comment in
`_compute_draft_losses`; it lives here instead, by decision, so that item 14
leaves no diff in the training path. It sits next to the tests that enforce it.
"""

import pytest
import torch
from torch import nn

from llmfoundry.models.giga_mix.configuration_giga_mix import GigaMixConfig
from llmfoundry.models.layers.mtp import LlamaDecoderMTPBlock
from llmfoundry.models.utils.flash_seqlens import get_cu_seqlens_from_pos_ids
from sber_tests.unit.mtp_harness import CausalMixer, DraftLossHarness, wiring_block

MTP_NUM = 3
SEQ_LEN = 32
PROBE = 16  # query position j; far from both ends and from the roll's wrap tail


def _perturbed(tensor: torch.Tensor, row: int) -> torch.Tensor:
    """`tensor` with row `row` of the sequence axis replaced, not nudged.

    A large replacement, so that "the output moved" cannot be confused with
    bf16 rounding in the positive controls.
    """
    out = tensor.clone()
    out[:, row, :] = torch.randn_like(out[:, row, :]) * 4.0 + 3.0
    return out


# --------------------------------------------------------------------------- #
# Part A1 -- forward wiring, no GPU.
# --------------------------------------------------------------------------- #


def _wiring_block(
    seq_len: int,
    hidden_size: int,
    mtp_recurrent: bool = False,
):
    """The production block with a causal plain-torch mixer for the decoder layer.

    Seeds are per module, so a recurrent block and an independent one are never
    accidentally the same function.
    """
    num_modules = 1 if mtp_recurrent else MTP_NUM
    return wiring_block(
        [CausalMixer(hidden_size, seq_len, seed=k) for k in range(num_modules)],
        [nn.LayerNorm(hidden_size) for _ in range(num_modules)],
        MTP_NUM,
        mtp_recurrent=mtp_recurrent,
    )


@pytest.mark.unit
@pytest.mark.fast
class TestForwardWiringOnCpu:
    """The item-14 perturbation test, minus the attention kernel."""

    @staticmethod
    def _case(hidden_size=8):
        torch.manual_seed(0)
        block = _wiring_block(SEQ_LEN, hidden_size)
        embeds = torch.randn(1, SEQ_LEN, hidden_size)
        hidden = torch.randn(1, SEQ_LEN, hidden_size)

        def run(*, input_embeds=None, hidden_states=None):
            with torch.no_grad():
                return block(
                    # Clone: the block writes `input_embeds[:, -1, :] = 0`.
                    input_embeds=(embeds if input_embeds is None else input_embeds).clone(),
                    hidden_states=(hidden if hidden_states is None else hidden_states).clone(),
                )

        return run, embeds, hidden

    def test_forward_is_reproducible(self):
        run, _, _ = self._case()
        assert all(torch.equal(a, b) for a, b in zip(run(), run()))

    @pytest.mark.parametrize("row", [PROBE + 1, PROBE + 2, SEQ_LEN - 1])
    def test_future_hidden_row_does_not_move_any_depth(self, row):
        run, _, hidden = self._case()
        base, moved = run(), run(hidden_states=_perturbed(hidden, row))
        for depth in range(MTP_NUM):
            assert torch.equal(base[depth][:, PROBE, :], moved[depth][:, PROBE, :]), (
                f"I-1 VIOLATED: perturbing hidden_states[:, {row}, :] moved the "
                f"depth-{depth} output at position {PROBE}. The draft hidden is "
                "being shifted somewhere in LlamaDecoderMTPBlock.forward."
            )

    def test_own_hidden_row_does_move_every_depth(self):
        run, _, hidden = self._case()
        base, moved = run(), run(hidden_states=_perturbed(hidden, PROBE))
        for depth in range(MTP_NUM):
            assert not torch.equal(base[depth][:, PROBE, :], moved[depth][:, PROBE, :]), (
                f"depth {depth} ignores hidden_states[:, {PROBE}, :]; the test above "
                "would pass vacuously"
            )

    @pytest.mark.parametrize("depth", range(MTP_NUM))
    def test_embed_row_reaches_exactly_the_expected_depths(self, depth):
        run, embeds, _ = self._case()
        row = PROBE + depth + 1
        base, moved = run(), run(input_embeds=_perturbed(embeds, row))
        for other in range(depth + 1):
            same = torch.equal(base[other][:, PROBE, :], moved[other][:, PROBE, :])
            assert same == (other < depth), (
                f"embed row {row} at depth {other}: "
                f"{'bit-identical' if same else 'moved'}, expected "
                f"{'bit-identical' if other < depth else 'moved'}. The per-depth "
                "embed roll is not exactly one step."
            )


# --------------------------------------------------------------------------- #
# Part A2 -- the same perturbations through the real MLA layer.  Needs a GPU:
# `LlamaLatentAttention` runs `flash_attn_varlen_func`, which has no CPU path.
# --------------------------------------------------------------------------- #


def _mtp_block_config(
    hidden_size: int = 256, num_hidden_layers: int = 4, v_head_dim: int = 128
):
    """Tiny stand-in for `sber_configs/mtp_baseline_config.yaml`.

    Arch fields that decide *which* MTP layer gets built are kept at the
    production values -- `mtp_block_type: dense`, MLA at a 192-wide q/k,
    gated attention, `layernorm_type: pre_post`, `varlen_input: true`. Only
    the sizes shrink.

    `v_head_dim` is a parameter because the two checkpoints this head trains on
    disagree about it: the gspark arms carry 128, against 192-wide q/k, and the
    converted papayaseek model carries 192 for both. The packed-attention assert
    takes either -- equal widths, or the (192, 128) pair FA3 handles natively --
    and the draft attention's Flex prefix does not treat them alike; F-45 in
    `recurrent-mtp-spec/decisions.md` has why, and what it cost to find out.

    Two deliberate departures, neither of which can hide an I-1 violation:

    * `norm_type` is `LlamaRMSNorm`, not production's `ZeroCenteredGatedNorm`.
      Every norm in `NORM_CLASS_REGISTRY` is position-local (a per-row
      statistic), so no norm can move information between positions; the
      zero-centered gated one only adds a `torch.compile` dependency to the
      test. Item 12 changes where the norms sit in the recurrence, not whether
      they mix positions.
    * `linear_attention_type` is None, so `LlamaDecoderLayer` reads
      `attention_type` directly. In production the MTP layer lands on MLA
      because `resolve_mtp_attention_anchor_layer_idx` (= num_hidden_layers-1)
      is in `full_attention_layers`; short-circuiting that lookup keeps the
      same attention class with fewer config invariants to satisfy.
    """
    return GigaMixConfig(
        # --- sizes: tiny ---
        vocab_size=256,
        hidden_size=hidden_size,
        intermediate_size=128,
        dense_intermediate_size=512,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=max(SEQ_LEN, 128),
        # --- MTP head ---
        use_mtp=True,
        mtp_predictor_num=MTP_NUM,
        mtp_block_type="dense",
        # --- attention: MLA, as production ---
        attention_type="LlamaLatentAttention",
        linear_attention_type=None,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=v_head_dim,
        kv_lora_rank=64,
        q_lora_rank=128,
        num_krope_heads=1,
        use_mla_scaling_factor=True,
        use_custom_rotary_kernel=False,
        gated_attention=True,
        attention_bias=False,
        attention_dropout=0.0,
        deterministic_attention=True,
        varlen_input=True,
        # --- norms / MLP ---
        norm_type="LlamaRMSNorm",
        layernorm_type="pre_post",
        fused_mlp=False,
        init_type="deepseek",
        # --- single process ---
        tp_size=1,
        use_cache=False,
        init_device="cpu",
    )


@pytest.fixture(scope="module")
def mtp_block_case():
    """A built `LlamaDecoderMTPBlock` plus one batch of inputs, on the GPU.

    `.train()` on purpose: `_is_varlen_active = self.training and
    self.varlen_input`, so eval mode would silently leave the packed
    attention path that training actually uses.
    """
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA: flash-attn MLA has no CPU kernel")

    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(0)

    config = _mtp_block_config()
    block = LlamaDecoderMTPBlock(config).to(device=device, dtype=dtype)
    block.train()

    batch, hidden_size = 1, config.hidden_size
    position_ids = torch.arange(SEQ_LEN, device=device).unsqueeze(0)
    cu_seqlens, max_seqlen = get_cu_seqlens_from_pos_ids(position_ids)

    inputs = dict(
        input_embeds=torch.randn(batch, SEQ_LEN, hidden_size, device=device, dtype=dtype),
        hidden_states=torch.randn(batch, SEQ_LEN, hidden_size, device=device, dtype=dtype),
        position_ids=position_ids,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
    )

    def run(*, input_embeds=None, hidden_states=None):
        # Clone: the block writes `input_embeds[:, -1, :] = 0` after each roll.
        kwargs = dict(inputs)
        kwargs["input_embeds"] = (
            inputs["input_embeds"] if input_embeds is None else input_embeds
        ).clone()
        kwargs["hidden_states"] = (
            inputs["hidden_states"] if hidden_states is None else hidden_states
        ).clone()
        with torch.no_grad():
            return block(**kwargs)

    return {"config": config, "run": run, "inputs": inputs}


class TestForwardIsDeterministic:
    """Precondition for every bit-identity assertion below."""

    @pytest.mark.unit
    @pytest.mark.requires_gpu
    def test_same_inputs_give_bit_identical_outputs(self, mtp_block_case):
        first = mtp_block_case["run"]()
        second = mtp_block_case["run"]()
        for depth, (a, b) in enumerate(zip(first, second)):
            assert torch.equal(a, b), (
                f"depth {depth} is not reproducible for identical inputs; the "
                "bit-identity checks in this file cannot be trusted until this "
                "passes (suspect a nondeterministic attention kernel)"
            )


class TestDraftInputIsUnshifted:
    """I-1, first line: `draft_h` is UN-shifted -- `raw_h[j]`, what serve provides.

    The head at position j must not be able to see the target's hidden state at
    any position after j. That forbids both a left-shift of the draft input and
    a non-causal draft attention, which are the two ways the head could read
    the answer.
    """

    @pytest.mark.unit
    @pytest.mark.requires_gpu
    @pytest.mark.parametrize("row", [PROBE + 1, PROBE + 2, SEQ_LEN - 1])
    def test_future_hidden_row_does_not_move_any_depth(self, mtp_block_case, row):
        base = mtp_block_case["run"]()
        moved = mtp_block_case["run"](
            hidden_states=_perturbed(mtp_block_case["inputs"]["hidden_states"], row)
        )

        for depth in range(MTP_NUM):
            assert torch.equal(base[depth][:, PROBE, :], moved[depth][:, PROBE, :]), (
                f"I-1 VIOLATED: perturbing hidden_states[:, {row}, :] moved the "
                f"depth-{depth} output at position {PROBE}. The draft input hidden "
                "is shifted (or the draft attention is not causal), so the head is "
                "reading the answer out of a future target state. Every acceptance "
                "number measured under this is optimistic."
            )

    @pytest.mark.unit
    @pytest.mark.requires_gpu
    def test_own_hidden_row_does_move_every_depth(self, mtp_block_case):
        """Positive control: without this the test above passes vacuously."""
        base = mtp_block_case["run"]()
        moved = mtp_block_case["run"](
            hidden_states=_perturbed(mtp_block_case["inputs"]["hidden_states"], PROBE)
        )

        for depth in range(MTP_NUM):
            assert not torch.equal(base[depth][:, PROBE, :], moved[depth][:, PROBE, :]), (
                f"depth {depth} at position {PROBE} ignores hidden_states[:, {PROBE}, :]. "
                "The perturbation test has no teeth -- fix the harness before reading "
                "the invariant checks as a pass."
            )


class TestEmbedShiftIsOnePerDepth:
    """I-1, first line: `input_ids` is left-shifted once, cumulatively per depth.

    At depth k the block has rolled `input_embeds` k+1 times, so position j is
    fed `embed(token[j+k+1])`. Perturbing embed row `j+k+1` therefore has a
    sharp per-depth fingerprint at position j: depth k sees it at its own query
    position, and every depth *below* k cannot see it at all, because for them
    the row lands at a position after j.

    That pair of assertions pins the roll to exactly one step per depth:

    * two rolls per depth -> row j+k+1 lands at j-k-1 for depth k and at
      j-1 for depth k-1, so a lower depth moves and the `same` half fails;
    * no roll at depth 0 -> row j+k+1 lands at j+1 for depth k, which a causal
      query at j cannot see, so the `not same` half fails.

    Depths *above* k are deliberately not asserted on: the row does reach them,
    but through an off-diagonal attention weight whose magnitude is not
    something this test should depend on.
    """

    @pytest.mark.unit
    @pytest.mark.requires_gpu
    @pytest.mark.parametrize("depth", range(MTP_NUM))
    def test_embed_row_reaches_exactly_the_expected_depths(self, mtp_block_case, depth):
        row = PROBE + depth + 1  # token[j+k+1] -- depth k's own input at position j
        base = mtp_block_case["run"]()
        moved = mtp_block_case["run"](
            input_embeds=_perturbed(mtp_block_case["inputs"]["input_embeds"], row)
        )

        for other in range(depth + 1):
            same = torch.equal(base[other][:, PROBE, :], moved[other][:, PROBE, :])
            if other < depth:
                # For depth `other`, embed row `row` sits at position
                # row-(other+1) > PROBE, which a causal query at PROBE cannot see.
                assert same, (
                    f"embed row {row} moved the depth-{other} output at position "
                    f"{PROBE}, but that row only enters depth-{other} input at "
                    f"position {row - other - 1}, which is after {PROBE}. The "
                    "per-depth embed roll is larger than one step, or the draft "
                    "attention is not causal."
                )
            else:
                assert not same, (
                    f"embed row {row} did NOT move the depth-{other} output at "
                    f"position {PROBE}, but depth {other} is fed token[j+{other}+1] "
                    f"at position j={PROBE}. The per-depth embed roll is missing or "
                    "smaller than one step."
                )


# --------------------------------------------------------------------------- #
# Part B -- loss pairing.  No GPU needed, only the training image (importing
# `modelling_giga_mix` pulls in flash-attn).
# --------------------------------------------------------------------------- #


def _DraftLossHarness(vocab_size: int):
    """The *production* loss helpers with a two-line model around them.

    Bound, not copied: that is the whole point of the test. Weights are non-zero
    so `_compute_draft_losses` takes its grad-carrying branch (the checkpointed
    `draft_chunk`) rather than the no_grad one.
    """
    return DraftLossHarness(
        vocab_size,
        MTP_NUM,
        weights={"lk": 1.0, "tv": 1.0, "kl": 1.0},
        mtp_position_weight_beta=0.7,
    )


def _mimicking_hidden(target: torch.Tensor, head: int, extra_shift: int = 0):
    """`combined_hidden[head]`: position t copies the target at t + head + extra.

    With `extra_shift == 0` head `head` reproduces the target distribution at
    exactly the position `_compute_draft_losses` should pair it with, so every
    distribution loss must come out at zero. Positions whose source runs off
    the end are filled with deliberate garbage -- they are the rows the roll
    wraps, and the mask has to drop them.
    """
    seq_len = target.size(1)
    out = torch.full_like(target, 7.0)  # garbage, must be masked out
    src = head + extra_shift
    if src < seq_len:
        out[:, : seq_len - src, :] = target[:, src:, :]
    return out


@pytest.mark.unit
@pytest.mark.fast
class TestDraftLossPairing:
    """`_compute_draft_losses`' roll agrees with `_build_shifted_labels`."""

    @staticmethod
    def _case(seq_len=12, batch=2, vocab_size=16):
        torch.manual_seed(0)
        harness = _DraftLossHarness(vocab_size)
        # Every label supervised, so the only masking left is the roll's own
        # tail -- which is what we want to pin.
        labels = torch.arange(batch * seq_len).view(batch, seq_len) % vocab_size
        shift_labels = labels.roll(shifts=(-1,), dims=(1,)).contiguous()
        shift_labels[:, -1] = harness.ignore_index
        mtp_labels = harness._build_shifted_labels(shift_labels)
        valid_counts = harness._compute_valid_counts(torch.stack(mtp_labels, dim=0))
        target = torch.randn(batch, seq_len, vocab_size) * 2.0
        return harness, labels, shift_labels, mtp_labels, valid_counts, target

    def test_shifted_labels_are_labels_at_j_plus_k_plus_1(self):
        harness, labels, _, mtp_labels, _, _ = self._case()
        batch, seq_len = labels.shape

        for k, level in enumerate(mtp_labels):
            for j in range(seq_len):
                source = j + k + 1
                want = (
                    labels[:, source] if source <= seq_len - 1
                    else torch.full((batch,), harness.ignore_index)
                )
                assert torch.equal(level[:, j], want.to(level.dtype)), (
                    f"mtp_labels[{k}][:, {j}] should be labels[:, {source}]"
                )

    def test_valid_count_per_head_is_seq_len_minus_h_minus_1(self):
        harness, labels, _, _, valid_counts, _ = self._case()
        batch, seq_len = labels.shape
        # mtp_labels[h][j] is valid iff j+h+1 <= seq_len-1, i.e. seq_len-1-h of them.
        want = torch.tensor(
            [batch * (seq_len - 1 - h) for h in range(1, MTP_NUM + 1)],
            dtype=torch.float32,
        )
        assert torch.equal(harness._mtp_num_tokens(valid_counts), want)

    @pytest.mark.parametrize("chunk_size", [3, 1024])
    def test_head_mimicking_target_at_t_plus_h_has_zero_loss(self, chunk_size):
        """The pairing, stated causally.

        If head h at position t is paired with the target at t+h -- as the
        docstring derivation says -- then a head that reproduces the target
        distribution at t+h has acceptance 1 and every distribution loss is 0.
        Any other pairing compares two unrelated random distributions.
        """
        harness, _, shift_labels, mtp_labels, valid_counts, target = self._case()
        combined = [target] + [
            _mimicking_hidden(target, h).requires_grad_(True)
            for h in range(1, MTP_NUM + 1)
        ]

        out = harness._compute_draft_losses(
            combined,
            shift_labels,
            ["lk", "tv", "kl"],
            mtp_labels,
            valid_counts,
            chunk_size=chunk_size,
        )

        for kind in ("lk", "tv", "kl"):
            per_head = out[kind][1]
            assert torch.allclose(
                per_head, torch.zeros_like(per_head), atol=1e-5
            ), (
                f"{kind} per-head is {per_head.tolist()}, expected ~0. Head h was "
                "built to reproduce the target distribution at t+h, so a nonzero "
                "loss means `_compute_draft_losses` pairs it with a different "
                "target position than `_build_shifted_labels` labels it with, or "
                "the wrapped rows are not being masked."
            )

    @pytest.mark.parametrize("extra_shift", [-1, 1])
    def test_off_by_one_pairing_is_detected(self, extra_shift):
        """Negative control: an off-by-one draft must NOT score zero."""
        harness, _, shift_labels, mtp_labels, valid_counts, target = self._case()
        combined = [target] + [
            _mimicking_hidden(target, h, extra_shift=extra_shift).requires_grad_(True)
            for h in range(1, MTP_NUM + 1)
        ]

        out = harness._compute_draft_losses(
            combined,
            shift_labels,
            ["tv"],
            mtp_labels,
            valid_counts,
            chunk_size=1024,
        )
        per_head = out["tv"][1]
        assert (per_head > 0.1).all(), (
            f"a draft shifted by {extra_shift} scored tv={per_head.tolist()}, which "
            "is indistinguishable from the aligned case. The zero-loss test above "
            "proves nothing until this fails loudly."
        )
