"""Shared harnesses for the MTP tests.

The production helpers are *bound* and run verbatim -- nothing here is a copy of
what is under test. What is stood in is the model around them.

Trimmed port of samoylovlesha/angelspec_eagle's harness: the `_draft_logits`
binding comes back with the `draft_loss_topk` port.
"""

import torch
from torch import nn

from llmfoundry.models.giga_mix.configuration_giga_mix import GigaMixConfig
from llmfoundry.models.giga_mix.modelling_giga_mix import GigaMixForCausalLM
from llmfoundry.models.layers.mtp import LlamaDecoderMTPBlock

DRAFT_LOSS_KINDS = (
    "lk",
    "tv",
    "kl",
    "adaptive_weight",
    "adaptive_weight_lambda",
    "e2e_tv",
)


class DraftLossHarness(nn.Module):
    """`GigaMixForCausalLM`'s loss helpers with a two-line model around them.

    `lm_head` is the identity, so a hidden row *is* a distribution's logits and
    the expected loss can be written out in a few lines of plain torch.

    `weights` maps kind -> `<kind>_loss_weight`; the rest are 0. The loss reads
    those attributes rather than the config, which is why they are set here.
    """

    same_doc_next_mask = staticmethod(GigaMixForCausalLM.same_doc_next_mask)
    _build_shifted_labels = GigaMixForCausalLM._build_shifted_labels
    _compute_valid_counts = GigaMixForCausalLM._compute_valid_counts
    _compute_draft_losses = GigaMixForCausalLM._compute_draft_losses
    _e2e_chain_loss = GigaMixForCausalLM._e2e_chain_loss
    _e2e_chain_weights = staticmethod(GigaMixForCausalLM._e2e_chain_weights)
    _finalize_mtp_acc_len = GigaMixForCausalLM._finalize_mtp_acc_len
    _draft_log_probs = GigaMixForCausalLM._draft_log_probs
    _aggregate_per_head = GigaMixForCausalLM._aggregate_per_head
    _mtp_position_weights = GigaMixForCausalLM._mtp_position_weights
    _mtp_num_tokens = staticmethod(GigaMixForCausalLM._mtp_num_tokens)
    DRAFT_LOSS_BETA_ATTR = GigaMixForCausalLM.DRAFT_LOSS_BETA_ATTR
    DRAFT_LOSS_TERMS = GigaMixForCausalLM.DRAFT_LOSS_TERMS

    def __init__(self, vocab_size, mtp_predictor_num, weights=None, **overrides):
        super().__init__()
        self.config = GigaMixConfig(
            use_mtp=True,
            mtp_predictor_num=mtp_predictor_num,
            vocab_size=vocab_size,
            hidden_size=vocab_size,
            lm_head_logit_softcapping=None,
            draft_loss_chunk_size=2048,
            num_attention_heads=2,
            num_key_value_heads=2,
            init_device="cpu",
            tp_size=1,
            **overrides,
        )
        self.use_mtp = True
        self.ignore_index = self.config.ignore_index
        for kind in DRAFT_LOSS_KINDS:
            setattr(self, f"{kind}_loss_weight", (weights or {}).get(kind, 0.0))
        self.lm_head = nn.Linear(vocab_size, vocab_size, bias=False)
        with torch.no_grad():
            self.lm_head.weight.copy_(torch.eye(vocab_size))


def packed_labels(seq_len, vocab_size, ignore_index=-100):
    """`shift_labels` for one fully supervised row: `labels[j + 1]` at `j`."""
    labels = torch.arange(seq_len).view(1, seq_len) % vocab_size
    shift_labels = labels.roll(shifts=(-1,), dims=(1,)).contiguous()
    shift_labels[:, -1] = ignore_index
    return shift_labels


class CausalMixer(nn.Module):
    """Stand-in for `LlamaDecoderMTPLayer`: fuse the two inputs, mix causally.

    Mirrors the only two properties the block's contract depends on -- the layer
    consumes `(hidden_states, input_embeds)` position-wise, and its output at `j`
    depends on inputs at `0..j` and nothing later. `tril` mixing multiplies every
    future row by exactly 0.0, which is what makes `torch.equal` the right
    assertion rather than an approximate one.

    Substituting this for MLA is what lets these tests run without a GPU.
    """

    def __init__(self, hidden_size: int, seq_len: int, seed: int):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.w_hidden = nn.Parameter(torch.randn(hidden_size, hidden_size, generator=gen))
        self.w_embed = nn.Parameter(torch.randn(hidden_size, hidden_size, generator=gen))
        mix = torch.randn(seq_len, seq_len, generator=gen).tril()
        self.register_buffer("mix", mix / mix.abs().sum(-1, keepdim=True))

    def forward(self, input_embeds, hidden_states, **_ignored):
        fused = torch.tanh(hidden_states @ self.w_hidden + input_embeds @ self.w_embed)
        return self.mix @ fused


class PassThroughLayer(nn.Module):
    """Stand-in decoder layer: the tuple `GigaMixModel`'s loop unpacks.

    `layer_outputs[0]` is the hidden state and `[-3:]` are the three hidden
    z-loss slots, which are `None` when their coefficients are 0. Standing the
    stack in this way is what lets `GigaMixModel.forward` run without the CUDA
    rotary kernel, while the slice under test -- the `mtp_block` call and
    `self.norm` below it -- stays production code.
    """

    def forward(self, hidden_states, **_ignored):
        return (hidden_states, None, None, None)


class RecordingMTPBlock(nn.Module):
    """Stand-in `mtp_block` that records what the model's forward handed it.

    The callables start at `"unset"` rather than `None` so that "the block was
    never called" cannot pass an assertion for "no projection was passed".
    """

    def __init__(self, mtp_predictor_num):
        super().__init__()
        self.mtp_predictor_num = mtp_predictor_num
        self.seen_hidden = None
        self.embed_fn = "unset"
        self.logits_fn = "unset"

    def forward(
        self, input_embeds, hidden_states, embed_fn=None, logits_fn=None, **_ignored
    ):
        self.seen_hidden = hidden_states
        self.embed_fn = embed_fn
        self.logits_fn = logits_fn
        return tuple(hidden_states for _ in range(self.mtp_predictor_num))


def stub_decoder_stack(model, mtp_predictor_num):
    """Swap a `GigaMixModel`'s decoder stack and MTP block for the stand-ins."""
    model.layers = nn.ModuleList([PassThroughLayer() for _ in model.layers])
    model.mtp_block = RecordingMTPBlock(mtp_predictor_num)
    return model


def wiring_block(
    layers,
    norms,
    mtp_predictor_num,
    mtp_recurrent=False,
    norm_before_chaining=True,
    on_policy_ttt=False,
    draft_chunk_size=2048,
    draft_prefix_diagonal=False,
):
    """The real `LlamaDecoderMTPBlock` with `__init__` skipped.

    `forward` reads only the flat attributes set here, so populating them by hand
    runs the production roll / chain / norm code unmodified while avoiding the
    flash-attn kernel `__init__` would build. One place to update when the block
    grows another flag.
    """
    block = LlamaDecoderMTPBlock.__new__(LlamaDecoderMTPBlock)
    nn.Module.__init__(block)
    block.use_mtp = True
    block.mtp_predictor_num = mtp_predictor_num
    block.mtp_recurrent = mtp_recurrent
    block.norm_before_chaining = norm_before_chaining
    block.on_policy_ttt = on_policy_ttt
    block.draft_chunk_size = draft_chunk_size
    block.draft_prefix_diagonal = draft_prefix_diagonal
    block.mtp_layers = nn.ModuleList(layers)
    block.mtp_norms = nn.ModuleList(norms)
    return block.eval()
