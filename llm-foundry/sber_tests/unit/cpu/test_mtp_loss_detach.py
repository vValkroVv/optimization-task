"""Gradient-boundary tests for ``mtp_loss_detach``.

The MTP branch has two paths back into the main model: the final decoder hidden
state and the token embeddings.  The flag must cut both paths while leaving the
MTP branch itself trainable.  The on-policy path embeds draft ids separately,
so it has a dedicated assertion as well.
"""

import pytest
import torch
from torch import nn

from llmfoundry.models.giga_mix.configuration_giga_mix import GigaMixConfig
from llmfoundry.models.giga_mix.modelling_giga_mix import GigaMixModel
from llmfoundry.models.gigar.configuration_gigar import GigarConfig
from llmfoundry.models.gigar.modelling_gigar import LlamaModel as GigarModel

HIDDEN = 8
SEQ_LEN = 12
VOCAB = 32


class TrainablePassThroughLayer(nn.Module):
    """A CPU decoder stand-in with a parameter on the main-model path."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(HIDDEN, HIDDEN, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(HIDDEN))

    def forward(self, hidden_states, **_ignored):
        return (self.proj(hidden_states), None, None, None)


class RecordingMTPBlock(nn.Module):
    """Records both branch inputs and keeps a trainable MTP-only parameter."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.seen_hidden = None
        self.seen_embeds = None

    def forward(self, input_embeds, hidden_states, **_ignored):
        self.seen_hidden = hidden_states
        self.seen_embeds = input_embeds
        return (self.proj(hidden_states + input_embeds),)


def _config_and_model(kind, detach):
    common = dict(
        use_mtp=False,
        mtp_predictor_num=1,
        mtp_loss_detach=detach,
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        intermediate_size=16,
        attention_type="LlamaAttention",
        norm_type="LlamaRMSNorm",
        init_device="cpu",
        tp_size=1,
        max_position_embeddings=64,
        use_liger=False,
    )
    if kind == "gigamix":
        config = GigaMixConfig(
            dense_intermediate_size=16,
            first_k_dense_replace=2,
            mtp_block_type="dense",
            **common,
        )
        model = GigaMixModel(config)
    else:
        config = GigarConfig(**common)
        model = GigarModel(config)

    # Building without MTP avoids constructing a real attention-backed draft
    # block in this CPU test.  The production forward and config branch below
    # are still used verbatim.
    config.use_mtp = True
    model.use_mtp = True
    model.layers = nn.ModuleList(TrainablePassThroughLayer() for _ in model.layers)
    # Keep the gradient-bearing final norm while avoiding fused GPU kernels.
    model.norm = nn.LayerNorm(HIDDEN)
    model.mtp_block = RecordingMTPBlock()
    return config, model.train()


def _ids():
    return (torch.arange(SEQ_LEN).view(1, SEQ_LEN) % VOCAB).long()


def _has_nonzero_grad(parameters):
    return any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
        for parameter in parameters
    )


@pytest.mark.unit
@pytest.mark.fast
class TestMtpLossDetachConfig:
    @pytest.mark.parametrize("config_cls", [GigaMixConfig, GigarConfig])
    def test_defaults_to_false_and_serializes(self, config_cls):
        assert config_cls().mtp_loss_detach is False
        assert config_cls(mtp_loss_detach=True).to_dict()["mtp_loss_detach"] is True


@pytest.mark.unit
@pytest.mark.fast
class TestMtpLossDetachBoundary:
    @pytest.mark.parametrize("kind", ["gigamix", "gigar"])
    @pytest.mark.parametrize("detach", [False, True])
    def test_cuts_both_main_model_paths_but_trains_mtp(self, kind, detach):
        _, model = _config_and_model(kind, detach)
        outputs = model(
            input_ids=_ids(),
            position_ids=torch.arange(SEQ_LEN).view(1, SEQ_LEN),
            return_dict=False,
            use_cache=False,
        )
        mtp_hidden = outputs[-1][0]
        mtp_hidden.square().mean().backward()

        block = model.mtp_block
        assert block.seen_hidden.requires_grad is not detach
        assert block.seen_embeds.requires_grad is not detach
        assert (block.seen_hidden.grad_fn is None) is detach
        assert (block.seen_embeds.grad_fn is None) is detach

        main_groups = (
            list(model.embed_tokens.parameters()),
            list(model.norm.parameters()),
            list(model.layers.parameters()),
        )
        for parameters in main_groups:
            assert _has_nonzero_grad(parameters) is not detach

        assert _has_nonzero_grad(block.parameters())

    @pytest.mark.parametrize("detach", [False, True])
    def test_on_policy_embeddings_obey_the_same_boundary(self, detach):
        _, model = _config_and_model("gigamix", detach)
        draft_embeds = model._embed_draft_ids(_ids())
        assert draft_embeds.requires_grad is not detach
        assert (draft_embeds.grad_fn is None) is detach
