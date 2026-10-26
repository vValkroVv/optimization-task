"""The device half of the prefix-plus-diagonal draft attention (spec item 10).

Everything about the geometry that is arithmetic is settled on CPU:
`sber_tests/unit/cpu/test_mtp_draft_attention_geometry.py` specifies it and diffs
it against AngelSpec's own functions, and
`sber_tests/unit/cpu/test_mtp_prefix_diagonal_attention.py` holds the production
merge and the cache assembly to that specification with the kernels stood in.

What no CPU test can reach is the compiled FlexAttention call itself, and it is
the part of the change carrying an assumption rather than an argument: the
log-sum-exp it returns beside the output must be (a) in natural log, (b) already
scaled by `scale`, and (c) shaped `[num_heads, seq]` rather than the other way
round. Get any of those wrong and the merge still returns a convex-looking
combination. Group A pins all three against a dense recomputation.

Group B is the integration: the real `LlamaDecoderMTPBlock` under the flag, on a
packed batch, against the same block with the flag off. Depth 0 must be
bit-identical -- it is plain causal attention either way -- and every later depth
must differ. That is the same signature the served comparison shows (item 10's
matched pair: step 1 identical, the deficit widening with depth), which is why it
is worth asserting rather than merely running.

The MLA config comes from `test_mtp_input_alignment.py` rather than a copy: it is
already the "tiny but production-shaped MLA" recipe, and two copies would drift.
"""

import math

import pytest
import torch

from llmfoundry.models.layers import attention as attention_module

from llmfoundry.models.layers.mtp import LlamaDecoderMTPBlock
from llmfoundry.models.utils.flash_seqlens import get_cu_seqlens_from_pos_ids
from sber_tests.unit.gpu.test_mtp_input_alignment import _mtp_block_config

# Over 128, because that is where the lowering stops routing FlexAttention to
# its *decoding* kernel -- which no packed training row is short enough to
# reach, and which cannot even build at a 192-wide value, its accumulator
# reshape wanting a power-of-two head width Triton will not give it. Below 128
# these tests exercised a kernel production never runs. Two mask blocks at the
# default 128 granularity, so the block mask has a full one and a partial one.
SEQ_LEN = 256
DEPTHS = 3


def _dense_causal(query, key, value, scale):
    """Plain causal attention in float32, as the definition of what the kernels do."""
    logits = torch.einsum("bqhd,bkhd->bhqk", query.float(), key.float()) * scale
    blocked = torch.ones(
        query.size(1), key.size(1), dtype=torch.bool, device=query.device
    ).triu(1)
    logits = logits.masked_fill(blocked, float("-inf"))
    out = torch.einsum("bhqk,bkhd->bqhd", torch.softmax(logits, dim=-1), value.float())
    return out, torch.logsumexp(logits, dim=-1).transpose(1, 2)


@pytest.fixture(scope="module", params=[128, 192], ids=["v128", "v192"])
def mla_case(request):
    """One built MTP block on the GPU, plus a packed single-document batch.

    Run at both value widths in the tree: 128, which the gspark arms carry
    against 192-wide q/k, and 192, which the converted papayaseek checkpoint
    carries for both. FlexAttention rounds the value width up to a power of
    two, so 192 costs a 256-wide value tile -- one more shared-memory stage per
    pipeline slot than 128 asks for, on the same 227 KB. The forward tile the
    call pins has to hold at whichever width the checkpoint brings.
    """
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA: flash-attn MLA has no CPU kernel")

    device, dtype = torch.device("cuda"), torch.bfloat16
    torch.manual_seed(0)
    config = _mtp_block_config(v_head_dim=request.param)
    # The helper sizes this for its own module's shorter rows; the draft chain
    # ropes depth `k` at `position_ids + k`, so the table has to reach past the
    # last position by the depth.
    config.max_position_embeddings = SEQ_LEN + DEPTHS
    config.mtp_predictor_num = DEPTHS
    # Recurrent on purpose: one shared layer is the configuration item 10 exists
    # for -- it is what makes `num_nextn_predict_layers` 1 and moves serving onto
    # plain EAGLE's geometry. It also means one attention object sees every depth.
    config.mtp_recurrent = True
    block = LlamaDecoderMTPBlock(config).to(device=device, dtype=dtype)
    # `_is_varlen_active = self.training and self.varlen_input`: eval mode would
    # leave the packed path that training actually uses.
    block.train()

    position_ids = torch.arange(SEQ_LEN, device=device).unsqueeze(0)
    cu_seqlens, max_seqlen = get_cu_seqlens_from_pos_ids(position_ids)
    return {
        "config": config,
        "block": block,
        "attn": block.mtp_layers[0].decoder_layer.self_attn.base_attn,
        "inputs": dict(
            input_embeds=torch.randn(
                1, SEQ_LEN, config.hidden_size, device=device, dtype=dtype
            ),
            hidden_states=torch.randn(
                1, SEQ_LEN, config.hidden_size, device=device, dtype=dtype
            ),
            position_ids=position_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        ),
    }


def _qkv(case, seed):
    """Query/key/value as `_prefix_diagonal_attention` receives them: value at
    `v_head_dim`, which is what the draft cache carries and the width the
    pinned forward tile has to fit (F-33)."""
    attn = case["attn"]
    torch.manual_seed(seed)
    device = case["inputs"]["hidden_states"].device

    def heads(width):
        return torch.randn(
            1, SEQ_LEN, attn.num_heads_per_rank, width,
            device=device, dtype=torch.bfloat16,
        )

    return heads(attn.qk_head_dim), heads(attn.qk_head_dim), heads(attn.v_head_dim)


class TestFlexReportsTheLogSumExpWeExpect:
    """Group A: the one assumption the CPU tests take on trust."""

    @pytest.mark.unit
    @pytest.mark.requires_gpu
    def test_the_returned_lse_matches_a_dense_recomputation(self, mla_case):
        attn = mla_case["attn"]
        query, key, value = _qkv(mla_case, seed=1)
        out, lse = attn._flex_attention_mla_forward(
            query, key, value, mla_case["inputs"]["cu_seqlens"]
        )
        dense_out, dense_lse = _dense_causal(query, key, value, attn.softmax_scale)
        assert lse.shape == dense_lse.shape, (
            f"expected [batch, seq, heads] {tuple(dense_lse.shape)}, got "
            f"{tuple(lse.shape)} -- FlexAttention reports [heads, seq]"
        )
        torch.testing.assert_close(out.float(), dense_out, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(lse.float(), dense_lse, atol=2e-2, rtol=2e-2)

    @pytest.mark.unit
    @pytest.mark.requires_gpu
    def test_the_lse_carries_the_softmax_scale(self, mla_case):
        """Unscaled logits would differ by a constant factor, not a shift, so a
        second reference at the wrong scale must *not* match."""
        attn = mla_case["attn"]
        query, key, value = _qkv(mla_case, seed=2)
        _, lse = attn._flex_attention_mla_forward(
            query, key, value, mla_case["inputs"]["cu_seqlens"]
        )
        _, unscaled = _dense_causal(query, key, value, 1.0)
        assert not torch.allclose(lse.float(), unscaled, atol=1e-1)

    @pytest.mark.unit
    @pytest.mark.requires_gpu
    def test_the_assembled_geometry_matches_the_dense_definition(self, mla_case):
        """`_prefix_diagonal_attention` on real kernel output, against one
        softmax over an explicitly built mask."""
        attn = mla_case["attn"]
        scale = attn.softmax_scale
        cache = [_qkv(mla_case, seed=10 + i)[1:] for i in range(DEPTHS - 1)]
        query, key, value = _qkv(mla_case, seed=99)

        draft_kv_cache = [(k, v) for k, v in cache]
        out = attn._prefix_diagonal_attention(
            query,
            key,
            value,
            draft_kv_cache,
            len(cache),
            None,
            mla_case["inputs"]["cu_seqlens"],
            mla_case["inputs"]["max_seqlen"],
        )

        steps = draft_kv_cache
        blocks, values = [], []
        blocked = torch.ones(
            SEQ_LEN, SEQ_LEN, dtype=torch.bool, device=query.device
        ).triu(1)
        eye = torch.eye(SEQ_LEN, dtype=torch.bool, device=query.device)
        for i, (step_key, step_value) in enumerate(steps):
            logits = (
                torch.einsum("bqhd,bkhd->bhqk", query.float(), step_key.float()) * scale
            )
            logits = logits.masked_fill(blocked if i == 0 else ~eye, float("-inf"))
            blocks.append(logits)
            values.append(step_value.float())
        weights = torch.softmax(torch.cat(blocks, dim=-1), dim=-1)
        dense = torch.einsum(
            "bhqk,bkhd->bqhd", weights, torch.cat(values, dim=1)
        )
        torch.testing.assert_close(out.float(), dense, atol=3e-2, rtol=3e-2)


class TestBlockUnderTheFlag:
    """Group B: the whole block, packed, flag on against flag off."""

    @pytest.mark.unit
    @pytest.mark.requires_gpu
    def test_depth_zero_agrees_and_later_depths_do_not(self, mla_case):
        block, inputs = mla_case["block"], mla_case["inputs"]

        def run(flag):
            block.draft_prefix_diagonal = flag
            kwargs = dict(inputs)
            kwargs["input_embeds"] = inputs["input_embeds"].clone()
            kwargs["hidden_states"] = inputs["hidden_states"].clone()
            with torch.no_grad():
                return block(**kwargs)

        on, off = run(True), run(False)
        block.draft_prefix_diagonal = False
        assert len(on) == len(off) == DEPTHS
        torch.testing.assert_close(on[0], off[0], atol=0.0, rtol=0.0)
        for depth in range(1, DEPTHS):
            assert not torch.allclose(on[depth], off[depth], atol=1e-3), (
                f"depth {depth} is unchanged by the flag: the cache is not "
                "reaching the attention"
            )

    @pytest.mark.unit
    @pytest.mark.requires_gpu
    def test_the_cache_holds_one_entry_per_depth(self, mla_case):
        """The block's cache is a local, so its size is observed through the
        attention: record what each call was handed."""
        block, inputs = mla_case["block"], mla_case["inputs"]
        attn = mla_case["attn"]
        seen = []
        real = attn._prefix_diagonal_attention

        def recording(query, key, value, draft_kv_cache, draft_depth, *args, **kwargs):
            seen.append(draft_depth)
            return real(
                query, key, value, draft_kv_cache, draft_depth, *args, **kwargs
            )

        attn._prefix_diagonal_attention = recording
        block.draft_prefix_diagonal = True
        try:
            kwargs = dict(inputs)
            kwargs["input_embeds"] = inputs["input_embeds"].clone()
            kwargs["hidden_states"] = inputs["hidden_states"].clone()
            with torch.no_grad():
                block(**kwargs)
        finally:
            attn._prefix_diagonal_attention = real
            block.draft_prefix_diagonal = False
        assert seen == list(range(DEPTHS)), (
            "each depth must see every earlier depth's keys and no more"
        )


class TestRopePhaseFollowsTheDraftStep:
    """The chain key of draft step `i` is served from position `j + i`, so it has
    to be roped there (F-30). `position_ids` cannot carry that: the varlen path
    passes `position_ids=None` and takes the phase from `cu_seqlens`, so the offset
    goes to the kernel as `seqlen_offsets`. Both tests exist because an offset that
    never arrives is invisible -- the run just trains the wrong phase.
    """

    @staticmethod
    def _run_recording(mla_case, force_zero=False):
        block, inputs = mla_case["block"], mla_case["inputs"]
        real = attention_module.apply_rotary_emb
        seen = []

        def recording(*args, seqlen_offsets=0, **kwargs):
            seen.append(seqlen_offsets)
            return real(
                *args, seqlen_offsets=0 if force_zero else seqlen_offsets, **kwargs
            )

        attention_module.apply_rotary_emb = recording
        block.draft_prefix_diagonal = True
        try:
            with torch.no_grad():
                outs = block(
                    input_embeds=inputs["input_embeds"].clone(),
                    hidden_states=inputs["hidden_states"].clone(),
                    position_ids=inputs["position_ids"],
                    cu_seqlens=inputs["cu_seqlens"],
                    max_seqlen=inputs["max_seqlen"],
                )
        finally:
            attention_module.apply_rotary_emb = real
            block.draft_prefix_diagonal = False
        return seen, outs

    @pytest.mark.unit
    @pytest.mark.requires_gpu
    def test_each_depth_ropes_at_its_own_offset(self, mla_case):
        seen, _ = self._run_recording(mla_case)
        assert sorted(set(seen)) == list(range(DEPTHS)), (
            f"expected one offset per depth, got {seen}"
        )
        # Query and key are roped separately, so each offset appears twice.
        for depth in range(DEPTHS):
            assert seen.count(depth) == 2, f"depth {depth} appears {seen.count(depth)}x"

    @pytest.mark.unit
    @pytest.mark.requires_gpu
    def test_the_offset_is_load_bearing(self, mla_case):
        """Forcing every offset back to 0 must change the result."""
        _, with_offset = self._run_recording(mla_case)
        _, without = self._run_recording(mla_case, force_zero=True)
        torch.testing.assert_close(with_offset[0], without[0], atol=0.0, rtol=0.0)
        for depth in range(1, DEPTHS):
            assert not torch.allclose(with_offset[depth], without[depth], atol=1e-3), (
                f"depth {depth} is unchanged by the RoPE offset"
            )


class TestGradientsFlow:
    @pytest.mark.unit
    @pytest.mark.requires_gpu
    def test_the_merged_path_is_differentiable_end_to_end(self, mla_case):
        block, inputs = mla_case["block"], mla_case["inputs"]
        block.draft_prefix_diagonal = True
        hidden = inputs["hidden_states"].clone().requires_grad_(True)
        try:
            outs = block(
                input_embeds=inputs["input_embeds"].clone(),
                hidden_states=hidden,
                position_ids=inputs["position_ids"],
                cu_seqlens=inputs["cu_seqlens"],
                max_seqlen=inputs["max_seqlen"],
            )
            outs[-1].float().sum().backward()
        finally:
            block.draft_prefix_diagonal = False
        assert hidden.grad is not None
        assert torch.isfinite(hidden.grad).all()
