"""The recurrent MTP head and the I-2 normalization convention (item 12).

Three groups, all CPU.

  1. `TestRecurrentLayout` -- the real `LlamaDecoderMTPBlock.__init__` on a tiny
     config. `mtp_recurrent` collapses `mtp_layers` / `mtp_norms` to one entry
     each, applied at every depth. NB for freeze regexes (`disable_grad`): the
     parameter names stay under index 0, so an enumerated per-depth pattern
     keeps matching under the recurrent layout but silently mis-scopes the
     moment `mtp_predictor_num` grows past what it enumerates (F-11) -- prefer
     `mtp_layers\\.\\d+\\.` / `mtp_norms\\.\\d+\\.` forms.

  2. `TestRecurrentForward` -- the production `forward`, with the decoder layer
     stood in by a plain-torch mixer (`sber_tests/unit/mtp_harness.py`,
     which is what keeps this off a GPU).
     Pins that the recurrent loop really reuses one module, that
     `norm_before_chaining` moves the normed state into the recurrence, and that
     off reproduces the pre-norm chain the head used to train under.

  3. `TestDraftInputPostnorm` -- `GigaMixModel.forward` with stand-in decoder
     layers, so the slice around the `mtp_block` call runs for real without the
     CUDA rotary kernel. Pins that depth 0 receives the post-`model.norm` hidden
     under the flag, that it is the *same tensor* the main head gets (the norm
     runs once, not twice), and that the flag off hands it the raw residual.

I-2, restated: serving normalizes the layer input twice -- `shared_head.norm`
then the next depth's `mtp_hidden_layernorm` -- where training once normalized
once. `norm_before_chaining` fixes the chain, `draft_input_postnorm` fixes
depth 0, and both default on (F-09). AngelSpec calls the result "norm-on-norm".

Ported from samoylovlesha/angelspec_eagle (spec items 10/12 in
recurrent-mtp-spec), minus the cases bound to that branch's recipe YAML
(`mtp_baseline_config.yaml` disable_grad regex) and to its gigachat-utils
`convertation` mapper (`TestExportDepthIndependence`) -- neither file ships
here. The layout half of the export claim survives as
`test_the_layout_backs_the_count_up`.

Reference: AngelSpec @ d3412be, `angelspec/models/draft/mtp.py:889` (`backbone`
-- `hnorm` at :906, `final_layernorm` at :918).
"""

import pytest
import torch
from torch import nn

from llmfoundry.models.giga_mix.configuration_giga_mix import GigaMixConfig
from llmfoundry.models.giga_mix.modelling_giga_mix import GigaMixModel
from llmfoundry.models.layers.attention import LlamaLatentAttention
from llmfoundry.models.layers.mtp import LlamaDecoderMTPBlock
from sber_tests.unit.mtp_harness import CausalMixer, stub_decoder_stack, wiring_block

MTP_NUM = 3
SEQ_LEN = 12
HIDDEN = 8
MLA = dict(
    attention_type="LlamaLatentAttention",
    qk_nope_head_dim=4,
    qk_rope_head_dim=4,
    v_head_dim=4,
    kv_lora_rank=4,
    q_lora_rank=8,
    use_custom_rotary_kernel=False,
)


def _tiny_config(**overrides):
    """Small enough to build in a test, real enough to build the real modules."""
    kwargs = dict(
        use_mtp=True,
        mtp_predictor_num=MTP_NUM,
        mtp_block_type="dense",
        vocab_size=32,
        hidden_size=HIDDEN,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        intermediate_size=16,
        dense_intermediate_size=16,
        first_k_dense_replace=2,
        attention_type="LlamaAttention",
        norm_type="ZeroCenteredRMSNorm",
        init_device="cpu",
        tp_size=1,
        max_position_embeddings=64,
        use_liger=False,
    )
    kwargs.update(overrides)
    return GigaMixConfig(**kwargs)


# --------------------------------------------------------------------------- #
# 1. Module layout.
# --------------------------------------------------------------------------- #


@pytest.mark.unit
@pytest.mark.fast
class TestRecurrentLayout:
    def test_independent_builds_one_module_per_depth(self):
        block = LlamaDecoderMTPBlock(_tiny_config())
        assert len(block.mtp_layers) == MTP_NUM
        assert len(block.mtp_norms) == MTP_NUM
        assert block.mtp_recurrent is False

    def test_recurrent_builds_exactly_one_of_each(self):
        block = LlamaDecoderMTPBlock(_tiny_config(mtp_recurrent=True, **MLA))
        assert len(block.mtp_layers) == 1
        assert len(block.mtp_norms) == 1
        assert block.mtp_recurrent is True
        assert block.mtp_predictor_num == MTP_NUM  # still drafts three tokens

    def test_init_snapshots_every_flag_the_forward_branches_on(self):
        # Every other test in this file injects these through `wiring_block`,
        # which skips `__init__` -- so a typo in one of the `getattr` names
        # would leave the suite green while a recurrent config silently trained
        # teacher-forced and pre-norm.
        block = LlamaDecoderMTPBlock(
            _tiny_config(
                mtp_recurrent=True,
                **MLA,
                # Off, so a mistyped `getattr` name reads the default and shows.
                norm_before_chaining=False,
                draft_input_postnorm=False,
                on_policy_ttt=True,
                draft_loss_chunk_size=512,
            )
        )
        assert block.mtp_recurrent is True
        assert block.norm_before_chaining is False
        assert block.on_policy_ttt is True
        assert block.draft_chunk_size == 512
        # Not a config field: the recurrent layout is what turns the draft
        # geometry on, so the two cannot be set apart (spec I-5, item 10).
        assert block.draft_prefix_diagonal is True

        defaults = LlamaDecoderMTPBlock(_tiny_config())
        assert defaults.norm_before_chaining is True
        assert defaults.on_policy_ttt is False
        assert defaults.draft_prefix_diagonal is False

    def test_recurrent_head_costs_one_depth_worth_of_parameters(self):
        independent = sum(
            p.numel() for p in LlamaDecoderMTPBlock(_tiny_config(**MLA)).parameters()
        )
        recurrent = sum(
            p.numel()
            for p in LlamaDecoderMTPBlock(_tiny_config(mtp_recurrent=True, **MLA)).parameters()
        )
        assert recurrent * MTP_NUM == independent

    def test_recurrent_parameter_names_stay_under_index_zero(self):
        block = LlamaDecoderMTPBlock(_tiny_config(mtp_recurrent=True, **MLA))
        names = [n for n, _ in block.named_parameters()]
        assert names, "the head has no parameters at all"
        for name in names:
            assert name.startswith(("mtp_layers.0.", "mtp_norms.0.")), name

    def test_recurrent_requires_mla_in_the_mtp_block(self):
        hybrid = dict(
            linear_attention_type="Qwen3NextGatedDeltaNet",
            full_attention_layers=[0],
        )
        block = LlamaDecoderMTPBlock(_tiny_config(mtp_recurrent=True, **hybrid, **MLA))
        assert isinstance(
            block.mtp_layers[0].decoder_layer.self_attn.base_attn, LlamaLatentAttention
        )
        with pytest.raises(ValueError, match="resolves to 'Qwen3NextGatedDeltaNet'"):
            _tiny_config(
                mtp_recurrent=True, use_linear_attention_in_mtp=True, **hybrid, **MLA
            )
        with pytest.raises(ValueError, match="resolves to 'LlamaAttention'"):
            _tiny_config(mtp_recurrent=True)
        _tiny_config(use_linear_attention_in_mtp=True, **hybrid)

    def test_the_layout_backs_the_count_up(self):
        """One exported head because there is exactly one to export.

        `num_nextn_predict_layers == 1` in an export would be a lie if the block
        still held per-depth modules, so assert against the module names a
        checkpoint mapper would walk (spec items 2 and 10).
        """
        block = LlamaDecoderMTPBlock(
            _tiny_config(mtp_recurrent=True, mtp_predictor_num=4, **MLA)
        )
        indices = {
            name.split(".")[1]
            for name, _ in block.named_parameters()
            if name.startswith(("mtp_layers.", "mtp_norms."))
        }
        assert indices == {"0"}


# --------------------------------------------------------------------------- #
# 2. The chain, with the decoder layer stood in.
# --------------------------------------------------------------------------- #


def _block(mtp_recurrent: bool, norm_before_chaining: bool = True):
    norms = nn.ModuleList(
        [nn.LayerNorm(HIDDEN) for _ in range(1 if mtp_recurrent else MTP_NUM)]
    )
    with torch.no_grad():
        for k, norm in enumerate(norms):  # not identity, and not all the same
            norm.weight.mul_(1.0 + 0.25 * (k + 1))
            norm.bias.add_(0.1 * (k + 1))
    layers = [CausalMixer(HIDDEN, SEQ_LEN, seed=k) for k in range(len(norms))]
    return wiring_block(
        layers,
        norms,
        MTP_NUM,
        mtp_recurrent=mtp_recurrent,
        norm_before_chaining=norm_before_chaining,
    )


def _reference_chain(block, embeds, hidden, *, recurrent, norm_before_chaining):
    """The chain written out longhand, as the spec describes it."""
    out = []
    mtp_h = hidden
    rolled = embeds
    for depth in range(MTP_NUM):
        idx = 0 if recurrent else depth
        rolled = torch.roll(rolled, shifts=-1, dims=1)
        rolled = rolled.clone()
        rolled[:, -1, :] = 0
        mtp_h = block.mtp_layers[idx](input_embeds=rolled, hidden_states=mtp_h)
        normed = block.mtp_norms[idx](mtp_h)
        if norm_before_chaining:
            mtp_h = normed
        out.append(normed)
    return out


@pytest.mark.unit
@pytest.mark.fast
class TestRecurrentForward:
    @staticmethod
    def _inputs():
        torch.manual_seed(0)
        return (
            torch.randn(1, SEQ_LEN, HIDDEN),
            torch.randn(1, SEQ_LEN, HIDDEN),
        )

    @staticmethod
    def _run(block, embeds, hidden):
        with torch.no_grad():
            # Clone: the block writes `input_embeds[:, -1, :] = 0` in place.
            return block(input_embeds=embeds.clone(), hidden_states=hidden.clone())

    @pytest.mark.parametrize("recurrent", [False, True])
    @pytest.mark.parametrize("norm_before_chaining", [False, True])
    def test_forward_matches_the_longhand_chain(self, recurrent, norm_before_chaining):
        embeds, hidden = self._inputs()
        block = _block(recurrent, norm_before_chaining)
        got = self._run(block, embeds, hidden)
        with torch.no_grad():
            want = _reference_chain(
                block,
                embeds,
                hidden,
                recurrent=recurrent,
                norm_before_chaining=norm_before_chaining,
            )
        assert len(got) == MTP_NUM
        for depth, (g, w) in enumerate(zip(got, want)):
            assert torch.equal(g, w), f"depth {depth}"

    def test_flag_off_reproduces_the_pre_norm_chain(self):
        """Off must be exactly the chain the head trained under before F-09.

        Written out as that code did it -- collect the raw chain, norm the
        collected list afterwards -- rather than by calling the new path with
        different arguments, because what the flag off is for is a checkpoint
        whose weights were fitted that way.
        """
        embeds, hidden = self._inputs()
        block = _block(mtp_recurrent=False, norm_before_chaining=False)

        with torch.no_grad():
            raw, mtp_h, rolled = [], hidden, embeds
            for layer in block.mtp_layers:
                rolled = torch.roll(rolled, shifts=-1, dims=1).clone()
                rolled[:, -1, :] = 0
                mtp_h = layer(input_embeds=rolled, hidden_states=mtp_h)
                raw.append(mtp_h)
            want = [norm(h) for norm, h in zip(block.mtp_norms, raw)]

        for depth, (g, w) in enumerate(zip(self._run(block, embeds, hidden), want)):
            assert torch.equal(g, w), f"depth {depth}"

    def test_recurrent_applies_the_same_module_at_every_depth(self):
        embeds, hidden = self._inputs()
        block = _block(mtp_recurrent=True)
        layer, norm = block.mtp_layers[0], block.mtp_norms[0]

        with torch.no_grad():
            got = self._run(block, embeds, hidden)
            mtp_h, rolled = hidden, embeds
            for depth in range(MTP_NUM):
                rolled = torch.roll(rolled, shifts=-1, dims=1).clone()
                rolled[:, -1, :] = 0
                mtp_h = norm(layer(input_embeds=rolled, hidden_states=mtp_h))
                assert torch.equal(got[depth], mtp_h), f"depth {depth}"

    def test_recurrent_and_independent_are_different_functions(self):
        """Guards against a vacuous suite: the layouts must actually diverge."""
        embeds, hidden = self._inputs()
        shared = self._run(_block(True), embeds, hidden)
        separate = self._run(_block(False), embeds, hidden)
        # Depth 0 uses module 0 in both, so it agrees by construction; the
        # divergence starts at depth 1, where the layouts pick different weights.
        assert torch.equal(shared[0], separate[0])
        for depth in range(1, MTP_NUM):
            assert not torch.allclose(shared[depth], separate[depth]), depth

    def test_norm_before_chaining_leaves_depth_0_alone_and_moves_the_rest(self):
        """The flag changes what the *next* depth receives, nothing earlier."""
        embeds, hidden = self._inputs()
        off = self._run(_block(False, norm_before_chaining=False), embeds, hidden)
        on = self._run(_block(False, norm_before_chaining=True), embeds, hidden)
        assert torch.equal(off[0], on[0])
        for depth in range(1, MTP_NUM):
            assert not torch.allclose(off[depth], on[depth]), depth


# --------------------------------------------------------------------------- #
# 3. Depth 0 -- the post-norm draft input.
# --------------------------------------------------------------------------- #


def _model_with_stub_stack(draft_input_postnorm: bool = True):
    # The two norm flags are one convention and are validated as a pair.
    model = GigaMixModel(
        _tiny_config(
            draft_input_postnorm=draft_input_postnorm,
            norm_before_chaining=draft_input_postnorm,
        )
    ).eval()
    return stub_decoder_stack(model, MTP_NUM)


@pytest.mark.unit
@pytest.mark.fast
class TestDraftInputPostnorm:
    @staticmethod
    def _ids():
        return (torch.arange(SEQ_LEN).view(1, SEQ_LEN) % 32).long()

    @classmethod
    def _run(cls, model):
        with torch.no_grad():
            outputs = model(
                input_ids=cls._ids(),
                position_ids=torch.arange(SEQ_LEN).view(1, SEQ_LEN),
                return_dict=False,
                use_cache=False,
            )
        return outputs[0]

    def test_off_feeds_depth_0_the_raw_residual(self):
        model = _model_with_stub_stack(draft_input_postnorm=False)
        main_hidden = self._run(model)
        seen = model.mtp_block.seen_hidden
        # The stub stack is the identity, so the draft input is the embedding
        # output -- pre-norm, and NOT what the main head consumes.
        assert not torch.allclose(seen, main_hidden)
        assert torch.equal(model.norm(seen), main_hidden)

    def test_depth_0_gets_the_normed_residual_not_the_raw_one(self):
        model = _model_with_stub_stack()
        main_hidden = self._run(model)
        seen = model.mtp_block.seen_hidden
        # The stub stack is the identity, so the pre-norm residual the head used
        # to receive is exactly the embedding output. It must NOT be what the
        # head sees now, and one norm must separate the two.
        with torch.no_grad():
            raw = model.embed_tokens(self._ids())
        assert not torch.allclose(seen, raw)
        assert torch.equal(model.norm(raw), seen)
        assert torch.equal(seen, main_hidden)

    def test_normalizes_once_not_twice(self):
        """The head and the main head must share one `model.norm` call.

        Two calls would be numerically equal here but would double the cost and,
        more to the point, would mean the two branches could drift apart under a
        later change. Identity, not equality.
        """
        model = _model_with_stub_stack()
        main_hidden = self._run(model)
        assert model.mtp_block.seen_hidden is main_hidden
