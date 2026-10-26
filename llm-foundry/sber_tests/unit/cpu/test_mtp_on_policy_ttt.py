"""On-policy TTT unrolling and invariant I-4 (spec item 8).

Teacher-forced, depth `k` is fed `embed(token[j+k+1])` -- the ground-truth token
head `k` is being trained to predict. On-policy, it is fed the head's own guess
at that token instead, which is depth `k-1`'s argmax at the SAME position `j`;
no roll, because the two name the same token. Depth 0 stays teacher-forced.

I-4 is the thing that goes quietly wrong: the label chain
(`_build_shifted_labels`) and the embed chain must stay separate. Derive one
from the other and the head is trained to predict its own mistakes -- the loss
falls, acceptance does not.

Everything here is CPU.

  * `TestOnPolicyChain` -- the production `LlamaDecoderMTPBlock.forward` with the
    decoder layer stood in by a causal plain-torch mixer, plus an explicit
    `logits_fn` / `embed_fn` pair so the drafted token is a value the test
    chooses rather than a value it has to guess.
  * `TestOnPolicyPlumbing` -- `GigaMixModel.forward` with a stand-in decoder
    stack: the flag off must pass no callables, and the flag on must reach the
    block with both, since a missing one has to raise rather than silently fall
    back to teacher forcing.
  * `TestChunkedArgmax` -- the streaming projection agrees with the dense one.

Reference: AngelSpec @ d3412be, `angelspec/models/mtp.py:506` (the
`if self.on_policy:` branch) and `:376` (the `gt_ids` / `embed_ids` contract).
"""

import pytest
import torch
from torch import nn

from llmfoundry.models.giga_mix.configuration_giga_mix import GigaMixConfig
from llmfoundry.models.layers.mtp import chunked_argmax
from sber_tests.unit.mtp_harness import CausalMixer, stub_decoder_stack, wiring_block

MTP_NUM = 3
SEQ_LEN = 10
HIDDEN = 8
VOCAB = 16


def _block(on_policy_ttt: bool, chunk_size: int = 2048):
    norms = nn.ModuleList([nn.LayerNorm(HIDDEN) for _ in range(MTP_NUM)])
    with torch.no_grad():
        # Per-channel weights, deliberately: a default LayerNorm is a uniform
        # positive rescale plus a shift, which preserves the argmax -- and then
        # taking it from the un-normed state would look correct. Production
        # norms are per-channel and gated, so `normed` is the only right source.
        for k, norm in enumerate(norms):
            norm.weight.copy_(torch.linspace(0.5, 2.0, HIDDEN).roll(k))
            norm.bias.copy_(torch.linspace(-0.4, 0.4, HIDDEN).roll(k))
    return wiring_block(
        [CausalMixer(HIDDEN, SEQ_LEN, seed=k) for k in range(MTP_NUM)],
        norms,
        MTP_NUM,
        on_policy_ttt=on_policy_ttt,
        draft_chunk_size=chunk_size,
    )


class _Vocab:
    """A projection and an embedding table that are exact inverses.

    `logits_fn` one-hots the first `VOCAB` hidden channels, so the argmax of a
    hidden row is a value the test can predict; `embed_fn` maps a token id back
    to that one-hot. The point is that "which token did depth k-1 draft" is
    readable, not that the numbers resemble a real model.
    """

    def __init__(self):
        self.embed_calls = []
        self.logits_calls = 0

    def logits_fn(self, flat_hidden):
        self.logits_calls += 1
        return flat_hidden[:, :VOCAB]

    def embed_fn(self, ids):
        self.embed_calls.append(ids)
        return torch.nn.functional.one_hot(ids, num_classes=HIDDEN).to(torch.float32)


@pytest.mark.unit
@pytest.mark.fast
class TestOnPolicyChain:
    @staticmethod
    def _inputs():
        torch.manual_seed(0)
        return torch.randn(1, SEQ_LEN, HIDDEN), torch.randn(1, SEQ_LEN, HIDDEN)

    @staticmethod
    def _run(block, embeds, hidden, vocab=None):
        kwargs = {}
        if vocab is not None:
            kwargs = {"embed_fn": vocab.embed_fn, "logits_fn": vocab.logits_fn}
        with torch.no_grad():
            return block(
                input_embeds=embeds.clone(), hidden_states=hidden.clone(), **kwargs
            )

    def test_off_ignores_the_callables_entirely(self):
        embeds, hidden = self._inputs()
        vocab = _Vocab()
        plain = self._run(_block(on_policy_ttt=False), embeds, hidden)
        with_fns = self._run(_block(on_policy_ttt=False), embeds, hidden, vocab)
        for depth in range(MTP_NUM):
            assert torch.equal(plain[depth], with_fns[depth])
        assert vocab.logits_calls == 0
        assert vocab.embed_calls == []

    def test_depth_0_is_still_teacher_forced(self):
        """The fork starts at depth 1; depth 0 must be bit-identical."""
        embeds, hidden = self._inputs()
        off = self._run(_block(False), embeds, hidden)
        on = self._run(_block(True), embeds, hidden, _Vocab())
        assert torch.equal(off[0], on[0])
        for depth in range(1, MTP_NUM):
            assert not torch.allclose(off[depth], on[depth]), depth

    def test_depth_k_is_fed_the_argmax_of_depth_k_minus_1(self):
        """The contract, position by position.

        Depth `k-1`'s output at `j` names token `j+k`, which is exactly what
        depth `k` is teacher-fed. So the drafted ids must be the argmax of the
        previous depth's *normed* output -- the tensor `lm_head` sees -- taken
        at the same position, with no roll.
        """
        embeds, hidden = self._inputs()
        vocab = _Vocab()
        block = _block(True)
        got = self._run(block, embeds, hidden, vocab)

        assert len(vocab.embed_calls) == MTP_NUM - 1
        for depth in range(1, MTP_NUM):
            want_ids = got[depth - 1][..., :VOCAB].argmax(dim=-1)
            assert torch.equal(vocab.embed_calls[depth - 1], want_ids), f"depth {depth}"

    def test_drafted_embedding_carries_gradient_to_the_embedding_table(self):
        """The token is stop-grad; the lookup is not.

        `argmax` kills the gradient by itself, and the projection runs under
        `no_grad` so it stores nothing. The embedding of the drafted id stays in
        the graph, exactly as the teacher-forced path's embedding does.
        """
        embeds, hidden = self._inputs()
        table = nn.Parameter(torch.randn(VOCAB, HIDDEN))
        vocab = _Vocab()

        def embed_fn(ids):
            return table[ids]

        block = _block(True)
        out = block(
            input_embeds=embeds.clone(),
            hidden_states=hidden.clone().requires_grad_(True),
            embed_fn=embed_fn,
            logits_fn=vocab.logits_fn,
        )
        out[-1].sum().backward()
        assert table.grad is not None
        assert torch.any(table.grad != 0)

    @pytest.mark.parametrize("chunk_size", [3, 2048])
    def test_chunking_does_not_change_the_chain(self, chunk_size):
        embeds, hidden = self._inputs()
        one_shot = self._run(_block(True, chunk_size=2048), embeds, hidden, _Vocab())
        chunked = self._run(_block(True, chunk_size=chunk_size), embeds, hidden, _Vocab())
        for depth in range(MTP_NUM):
            assert torch.equal(one_shot[depth], chunked[depth]), depth


@pytest.mark.unit
@pytest.mark.fast
class TestChunkedArgmax:
    @pytest.mark.parametrize("chunk_size", [1, 7, 64, 10_000])
    def test_matches_the_dense_argmax(self, chunk_size):
        torch.manual_seed(0)
        hidden = torch.randn(2, SEQ_LEN, HIDDEN)
        weight = torch.randn(VOCAB, HIDDEN)

        def logits_fn(flat):
            return flat @ weight.T

        want = (hidden @ weight.T).argmax(dim=-1)
        got = chunked_argmax(logits_fn, hidden, chunk_size)
        assert got.shape == hidden.shape[:-1]
        assert torch.equal(got, want)


# --------------------------------------------------------------------------- #
# Plumbing: `lm_head` lives one level above the block.
# --------------------------------------------------------------------------- #


def _causal_lm(on_policy_ttt: bool):
    from llmfoundry.models.giga_mix.modelling_giga_mix import GigaMixForCausalLM

    config = GigaMixConfig(
        use_mtp=True,
        mtp_predictor_num=MTP_NUM,
        mtp_block_type="dense",
        on_policy_ttt=on_policy_ttt,
        vocab_size=VOCAB,
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
    model = GigaMixForCausalLM(config).eval()
    stub_decoder_stack(model.model, MTP_NUM)
    return model


@pytest.mark.unit
@pytest.mark.fast
class TestOnPolicyPlumbing:
    @staticmethod
    def _run(model):
        ids = torch.randint(0, VOCAB, (1, SEQ_LEN))
        with torch.no_grad():
            model.model(
                input_ids=ids,
                position_ids=torch.arange(SEQ_LEN).view(1, SEQ_LEN),
                return_dict=False,
                use_cache=False,
                lm_head_fn=model.lm_head if model.config.on_policy_ttt else None,
            )
        return model.model.mtp_block

    def test_off_passes_no_projection(self):
        block = self._run(_causal_lm(on_policy_ttt=False))
        assert block.logits_fn is None
        # The embed callable is unconditional; the block ignores it when off.
        assert callable(block.embed_fn)

    def test_on_passes_the_lm_head_itself_not_a_submodule(self):
        model = _causal_lm(on_policy_ttt=True)
        block = self._run(model)
        assert block.logits_fn is model.lm_head
        # Registering it under the block is what would double-wrap it in FSDP.
        assert "lm_head" not in dict(model.model.mtp_block.named_modules())

    def test_embed_callable_round_trips_ids_to_the_embedding_frame(self):
        model = _causal_lm(on_policy_ttt=True)
        block = self._run(model)
        ids = torch.randint(0, VOCAB, (1, SEQ_LEN))
        with torch.no_grad():
            assert torch.equal(block.embed_fn(ids), model.model.embed_tokens(ids))

    def test_tp_size_above_one_is_rejected_at_config_time(self):
        with pytest.raises(ValueError, match="on_policy_ttt is not supported"):
            GigaMixConfig(
                use_mtp=True,
                mtp_predictor_num=MTP_NUM,
                on_policy_ttt=True,
                tp_size=2,
                vocab_size=VOCAB,
                hidden_size=HIDDEN,
                num_attention_heads=2,
                num_key_value_heads=2,
                init_device="cpu",
            )
