"""The two-norm MTP convention: `norm_before_chaining` + `draft_input_postnorm`.

Serving normalizes the chained draft state twice (`shared_head.norm` ->
`hnorm`), historical training once. The two flags move training onto the
serving convention; both default on. Heads fitted the other way are not
weight-compatible, so the off position must reproduce the pre-convention code
exactly -- bit for bit, forward and backward. That is what this file pins:

  - flags off: `LlamaDecoderMTPBlock.forward` equals the old loop (norms as a
    side branch after the loop, raw state chained) -- outputs AND gradients;
  - flag on: equals a longhand reference where depth `k+1` receives
    `mtp_norms[k](h_k)`;
  - the flag is not a no-op: depth 0 agrees across the pair, deeper depths move;
  - the halves are validated as a pair (`validate_mtp_config`), including after
    post-`__init__` overrides, on `GigaMixConfig` and `GigarConfig` alike.

Block tests bind the production `LlamaDecoderMTPBlock.forward`; the decoder
layer is stood in by a two-matrix toy so everything runs on CPU. The toy is
substituted via `__new__` -- the forward under test is production code.
"""

import pytest
import torch
from torch import nn

from llmfoundry.models.giga_mix.configuration_giga_mix import GigaMixConfig
from llmfoundry.models.gigar.configuration_gigar import GigarConfig
from llmfoundry.models.layers.mtp import LlamaDecoderMTPBlock

DEPTHS = 3
HIDDEN = 32
SEQ = 16
BATCH = 2


class ToyLayer(nn.Module):
    """Stand-in `LlamaDecoderMTPLayer`: fuses the two inputs, nothing else.

    The block's contract only needs the layer to consume
    `(input_embeds, hidden_states)`; attention is irrelevant to the norm
    convention, which is why this runs without a GPU.
    """

    def __init__(self, hidden, seed):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.w_embed = nn.Parameter(torch.randn(hidden, hidden, generator=gen) * 0.1)
        self.w_hidden = nn.Parameter(torch.randn(hidden, hidden, generator=gen) * 0.1)

    def forward(self, input_embeds, hidden_states, **_ignored):
        return torch.tanh(
            input_embeds @ self.w_embed + hidden_states @ self.w_hidden
        )


def _block(norm_before_chaining):
    blk = LlamaDecoderMTPBlock.__new__(LlamaDecoderMTPBlock)
    nn.Module.__init__(blk)
    blk.use_mtp = True
    blk.norm_before_chaining = norm_before_chaining
    blk.mtp_predictor_num = DEPTHS
    blk.mtp_recurrent = False
    blk.draft_prefix_diagonal = False
    blk.on_policy_ttt = False
    blk.draft_chunk_size = 2048
    blk.mtp_layers = nn.ModuleList(
        ToyLayer(HIDDEN, seed=100 + i) for i in range(DEPTHS)
    )
    blk.mtp_norms = nn.ModuleList(nn.LayerNorm(HIDDEN) for _ in range(DEPTHS))
    # Distinct affine params per depth, so a norm applied at the wrong depth
    # cannot cancel out.
    for i, norm in enumerate(blk.mtp_norms):
        with torch.no_grad():
            norm.weight.mul_(0.5 + 0.25 * i)
            norm.bias.add_(0.1 * (i + 1))
    return blk


def _old_loop(blk, input_embeds, hidden_states):
    """The pre-convention loop: raw state chained, norms after the loop."""
    outs = []
    mtp_h = hidden_states
    emb = input_embeds
    for layer in blk.mtp_layers:
        emb = torch.roll(emb, shifts=-1, dims=1).clone()
        emb[:, -1, :] = 0
        mtp_h = layer(input_embeds=emb, hidden_states=mtp_h)
        outs.append(mtp_h)
    return tuple(n(h) for n, h in zip(blk.mtp_norms, outs))


def _inputs(seed=0):
    gen = torch.Generator().manual_seed(seed)
    emb = torch.randn(BATCH, SEQ, HIDDEN, generator=gen)
    hid = torch.randn(BATCH, SEQ, HIDDEN, generator=gen)
    return emb, hid


@pytest.mark.unit
@pytest.mark.fast
class TestNormBeforeChainingOff:
    """Off = the exact pre-convention behaviour, which existing heads need."""

    def test_forward_is_bitwise_the_old_loop(self):
        blk = _block(norm_before_chaining=False)
        emb, hid = _inputs()
        new = blk(input_embeds=emb.clone(), hidden_states=hid,
                  is_left_padded_eval=False)
        old = _old_loop(blk, emb.clone(), hid)
        assert len(new) == len(old) == DEPTHS
        for k, (a, b) in enumerate(zip(new, old)):
            assert torch.equal(a, b), f"depth {k} differs from the old loop"

    def test_gradients_are_bitwise_the_old_loop(self):
        """Moving the norm call inside the loop must not change the graph:
        `norm_k(h_k)` keeps the same parents and the same two consumers of
        `h_k` (the next layer and the norm), so backward is identical too.
        """
        blk = _block(norm_before_chaining=False)
        emb, hid = _inputs()

        def run(fn):
            blk.zero_grad(set_to_none=True)
            e = emb.clone().requires_grad_(True)
            h = hid.clone().requires_grad_(True)
            outs = fn(blk, e, h) if fn is _old_loop else fn(
                input_embeds=e, hidden_states=h, is_left_padded_eval=False
            )
            loss = sum((k + 1) * o.pow(2).mean() for k, o in enumerate(outs))
            loss.backward()
            grads = {n: p.grad.clone() for n, p in blk.named_parameters()}
            return loss.detach(), grads, e.grad.clone(), h.grad.clone()

        l_new, g_new, eg_new, hg_new = run(blk)
        l_old, g_old, eg_old, hg_old = run(_old_loop)
        assert torch.equal(l_new, l_old)
        assert g_new.keys() == g_old.keys()
        for name in g_new:
            assert torch.equal(g_new[name], g_old[name]), f"grad differs: {name}"
        assert torch.equal(eg_new, eg_old)
        assert torch.equal(hg_new, hg_old)


@pytest.mark.unit
@pytest.mark.fast
class TestNormBeforeChainingOn:
    def test_forward_is_the_longhand_normed_chain(self):
        """Depth `k+1` receives `mtp_norms[k](h_k)` -- serve's chained state."""
        blk = _block(norm_before_chaining=True)
        emb, hid = _inputs()
        got = blk(input_embeds=emb.clone(), hidden_states=hid,
                  is_left_padded_eval=False)

        outs, mtp_h, e = [], hid, emb.clone()
        for k in range(DEPTHS):
            e = torch.roll(e, shifts=-1, dims=1).clone()
            e[:, -1, :] = 0
            h = blk.mtp_layers[k](input_embeds=e, hidden_states=mtp_h)
            normed = blk.mtp_norms[k](h)
            mtp_h = normed  # the convention: chain the normed state
            outs.append(normed)
        for k, (a, b) in enumerate(zip(got, outs)):
            assert torch.equal(a, b), f"depth {k} differs from longhand"

    def test_flag_is_not_a_noop_past_depth_zero(self):
        """Depth 0 has no chained-in norm yet, so it must agree across the
        pair; every deeper depth must move, or the flag rewired nothing."""
        emb, hid = _inputs()
        blk_off = _block(norm_before_chaining=False)
        blk_on = _block(norm_before_chaining=True)
        blk_on.load_state_dict(blk_off.state_dict())
        off = blk_off(input_embeds=emb.clone(), hidden_states=hid,
                      is_left_padded_eval=False)
        on = blk_on(input_embeds=emb.clone(), hidden_states=hid,
                    is_left_padded_eval=False)
        assert torch.equal(on[0], off[0])
        for k in range(1, DEPTHS):
            assert not torch.equal(on[k], off[k]), f"depth {k}: flag was a no-op"


@pytest.mark.unit
@pytest.mark.fast
class TestNormPairValidation:
    """The halves are one convention: a half-set pair normalizes the chain and
    depth 0 differently and matches serving at neither, so `validate_mtp_config`
    rejects it -- at construction and after post-`__init__` overrides (the
    teacher-student `setattr` path re-runs it)."""

    MTP = dict(use_mtp=True, mtp_predictor_num=3)

    def test_defaults_are_the_serving_convention(self):
        for cls in (GigaMixConfig, GigarConfig):
            cfg = cls(**self.MTP)
            assert cfg.norm_before_chaining is True
            assert cfg.draft_input_postnorm is True

    @pytest.mark.parametrize("chain,depth0", [(True, False), (False, True)])
    def test_half_set_pair_is_rejected_at_init(self, chain, depth0):
        for cls in (GigaMixConfig, GigarConfig):
            with pytest.raises(ValueError, match="set together"):
                cls(**self.MTP,
                    norm_before_chaining=chain, draft_input_postnorm=depth0)

    def test_half_set_pair_is_rejected_post_hoc(self):
        for cls in (GigaMixConfig, GigarConfig):
            cfg = cls(**self.MTP)
            cfg.draft_input_postnorm = False  # the setattr-override path
            with pytest.raises(ValueError, match="set together"):
                cfg.validate_mtp_config()

    def test_exempt_when_the_chain_does_not_exist(self):
        """A single head has nothing to chain into; no MTP has no heads at
        all. `norm_before_chaining` is physically a no-op there, so a
        half-set pair is legal rather than a trap."""
        GigaMixConfig(use_mtp=True, mtp_predictor_num=1,
                      norm_before_chaining=True, draft_input_postnorm=False)
        GigaMixConfig(use_mtp=False,
                      norm_before_chaining=True, draft_input_postnorm=False)

    def test_explicit_pre_norm_pair_survives_serialization(self):
        """Continuing an existing pre-norm head sets both to false; the pair
        must round-trip the way checkpoint configs are stored."""
        cfg = GigaMixConfig(**self.MTP,
                            norm_before_chaining=False,
                            draft_input_postnorm=False)
        restored = GigaMixConfig(**{
            k: v for k, v in cfg.to_dict().items()
            if k != "transformers_version"
        })
        assert restored.norm_before_chaining is False
        assert restored.draft_input_postnorm is False


@pytest.mark.unit
@pytest.mark.fast
class TestRelocatedMtpValidation:
    """The MTP checks that moved from `__init__` into `validate_mtp_config`
    still fire -- and now also fire on post-`__init__` overrides."""

    @pytest.mark.parametrize("kwargs,match", [
        (dict(mtp_position_weight_beta=0), "mtp_position_weight_beta"),
        (dict(lk_position_weight_beta=-1.0), "lk_position_weight_beta"),
        (dict(draft_loss_chunk_size=0), "draft_loss_chunk_size"),
        (dict(lk_loss_eps=0.0), "lk_loss_eps"),
        (dict(lk_loss_eps=1.0), "lk_loss_eps"),
        (dict(nu_adaptive_weight=-0.1), "nu_adaptive_weight"),
        (dict(mtp_block_type="sparse"), "mtp_block_type"),
        (dict(enable_adaptive_weight_loss=True, lk_loss_weight=0.5),
         "enable_adaptive_weight_loss"),
    ])
    def test_rejected_at_init(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            GigaMixConfig(**kwargs)

    def test_adaptive_weight_alone_is_legal(self):
        GigaMixConfig(enable_adaptive_weight_loss=True, mtp_ce_loss_weight=0.0)

    def test_post_hoc_override_is_caught(self):
        """The case `__init__` never sees: a weight overridden afterwards."""
        cfg = GigaMixConfig(enable_adaptive_weight_loss=True,
                            mtp_ce_loss_weight=0.0)
        cfg.lk_loss_weight = 0.5
        with pytest.raises(ValueError, match="enable_adaptive_weight_loss"):
            cfg.validate_mtp_config()
