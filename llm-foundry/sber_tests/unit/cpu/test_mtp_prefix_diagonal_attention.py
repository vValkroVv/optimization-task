"""The production prefix-plus-diagonal attention path (spec item 10).

The geometry itself is specified in `test_mtp_draft_attention_geometry.py`, which
diffs a pure-torch reference against AngelSpec's own functions. This file is the
other half: the production code that implements it, held to that reference.

Three things are checkable without a GPU, and all three are here.

  * `merge_prefix_diagonal` -- the merge, pure torch. Diffed against
    `_dense_prefix_diagonal`, so the shipped merge answers to the reference the
    specification proved rather than to a re-reading of it.
  * `LlamaLatentAttention._prefix_diagonal_attention` -- the assembly: which
    cache entry becomes the causal prefix, which become diagonals, and what gets
    appended. The two CUDA-only calls it makes are stood in by one dense
    equivalent.
  * `LlamaDecoderMTPBlock.forward` -- that one cache is threaded across the whole
    depth loop and grows by exactly one entry per depth.

What is *not* checkable here is the compiled FlexAttention call itself;
`sber_tests/unit/gpu/test_mtp_prefix_diagonal_gpu.py` carries that leg.
"""

import math

import pytest
import torch
from torch import nn

from llmfoundry.models.layers.attention import (
    LlamaLatentAttention,
    merge_prefix_diagonal,
)
from sber_tests.unit.cpu.test_mtp_draft_attention_geometry import (
    _dense_prefix_diagonal,
)
from sber_tests.unit.mtp_harness import wiring_block

BATCH = 2
SEQ = 6
HEADS = 2
DIM = 4
SCALE = 1.0 / math.sqrt(DIM)


def _cache(steps, seed=0):
    """A draft cache in production layout: `[batch, seq, heads, dim]` per entry."""
    torch.manual_seed(seed)
    shape = (BATCH, SEQ, HEADS, DIM)
    return [(torch.randn(shape), torch.randn(shape)) for _ in range(steps)]


def _query(seed=1):
    torch.manual_seed(seed)
    return torch.randn(BATCH, SEQ, HEADS, DIM)


def _reference(query, cache):
    """`_dense_prefix_diagonal` over the same tensors, in its own layout.

    The reference is head-major (`[b, h, s, d]`) with a stacked step axis; the
    production path is sequence-major with a list. Transposing here is the whole
    of the difference, which is worth doing in the test rather than in either
    implementation.
    """
    to_ref = lambda t: t.transpose(1, 2)
    cache_k = torch.stack([to_ref(k) for k, _ in cache], dim=2)
    cache_v = torch.stack([to_ref(v) for _, v in cache], dim=2)
    return _dense_prefix_diagonal(to_ref(query), cache_k, cache_v).transpose(1, 2)


class _DenseMLA(LlamaLatentAttention):
    """`LlamaLatentAttention` with its two CUDA-only calls stood in.

    `__init__` is skipped, as `wiring_block` skips the block's: it would build the
    flash-attn projections. `_prefix_diagonal_attention` reads only
    `softmax_scale` and those two, so all three are provided and the production
    method itself runs unmodified.
    """

    def __init__(self):
        nn.Module.__init__(self)
        self.softmax_scale = SCALE

    def _flex_attention_mla_forward(self, query, key, value, cu_seqlens=None):
        logits = torch.einsum("bqhd,bkhd->bhqk", query, key) * self.softmax_scale
        blocked = torch.ones(query.size(1), key.size(1), dtype=torch.bool).triu(1)
        logits = logits.masked_fill(blocked, torch.finfo(logits.dtype).min)
        out = torch.einsum("bhqk,bkhd->bqhd", torch.softmax(logits, dim=-1), value)
        return out, torch.logsumexp(logits, dim=-1).transpose(1, 2)

    def _flash_attention_mla_forward(
        self, query, key, value, attention_mask=None, cu_seqlens=None, max_seqlen=None
    ):
        return self._flex_attention_mla_forward(query, key, value)[0]


@pytest.mark.unit
@pytest.mark.fast
class TestMergeMatchesTheReference:
    @pytest.mark.parametrize("steps", [1, 2, 3, 4])
    def test_the_shipped_merge_equals_the_dense_reference(self, steps):
        query, cache = _query(), _cache(steps)
        attn = _DenseMLA()
        prefix_key, prefix_value = cache[0]
        out0, lse0 = attn._flex_attention_mla_forward(
            query, prefix_key, prefix_value
        )
        scores = [(query * k).sum(-1) * SCALE for k, _ in cache[1:]]
        values = [v for _, v in cache[1:]]
        torch.testing.assert_close(
            merge_prefix_diagonal(out0, lse0, scores, values),
            _reference(query, cache),
            atol=1e-6,
            rtol=1e-5,
        )

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
    def test_the_result_keeps_the_input_dtype(self, dtype):
        """FlexAttention reports its log-sum-exp in fp32 whatever the inputs are.

        Without an explicit cast the merged output is promoted to fp32 and
        `o_proj` rejects it -- a `RuntimeError` about mat1/mat2 dtypes, three
        frames away from the cause. Every other test here runs in fp32, where the
        promotion is invisible.
        """
        query, cache = _query().to(dtype), [
            (k.to(dtype), v.to(dtype)) for k, v in _cache(3)
        ]
        out0 = torch.zeros(BATCH, SEQ, HEADS, DIM, dtype=dtype)
        lse0 = torch.zeros(BATCH, SEQ, HEADS, dtype=torch.float32)
        scores = [(query * k).sum(-1).float() for k, _ in cache[1:]]
        values = [v for _, v in cache[1:]]
        assert merge_prefix_diagonal(out0, lse0, scores, values).dtype is dtype


@pytest.mark.unit
@pytest.mark.fast
class TestPrefixDiagonalAssembly:
    def test_depth_zero_is_plain_causal_and_seeds_the_cache(self):
        query, cache = _query(), _cache(1)
        key, value = cache[0]
        attn, draft_kv_cache = _DenseMLA(), []
        out = attn._prefix_diagonal_attention(query, key, value, draft_kv_cache, 0)
        torch.testing.assert_close(
            out, attn._flash_attention_mla_forward(query, key, value)
        )
        assert len(draft_kv_cache) == 1
        assert draft_kv_cache[0][0] is key and draft_kv_cache[0][1] is value

    @pytest.mark.parametrize("depth", [1, 2, 3])
    def test_a_later_depth_matches_the_reference_geometry(self, depth):
        """The production method against the specified geometry, end to end.

        The second assertion is the off-by-one guard: appending this step's keys
        *after* the merge instead of before also produces a plausible convex
        combination, one diagonal term short.
        """
        query = _query()
        cache = _cache(depth + 1)
        attn = _DenseMLA()
        draft_kv_cache = list(cache[:depth])
        out = attn._prefix_diagonal_attention(query, *cache[depth], draft_kv_cache, depth)
        torch.testing.assert_close(
            out, _reference(query, cache[: depth + 1]), atol=1e-6, rtol=1e-5
        )
        assert not torch.allclose(out, _reference(query, cache[:depth]), atol=1e-3)
        assert len(draft_kv_cache) == depth + 1

    def test_only_entry_zero_supplies_the_prefix(self):
        """A later step's off-diagonal keys are not readable (I-6), and the
        control shows entry zero's are."""
        query, cache = _query(), _cache(3)
        attn = _DenseMLA()
        probe = 4
        base = attn._prefix_diagonal_attention(query, *cache[2], list(cache[:2]), 2)

        bumped = [(k.clone(), v.clone()) for k, v in cache]
        bumped[1][0][:, probe - 1] = 5.0
        bumped[1][1][:, probe - 1] = 5.0
        moved = attn._prefix_diagonal_attention(query, *cache[2], list(bumped[:2]), 2)
        assert torch.equal(moved[:, probe], base[:, probe])

        bumped = [(k.clone(), v.clone()) for k, v in cache]
        bumped[0][0][:, probe - 1] = 5.0
        bumped[0][1][:, probe - 1] = 5.0
        moved = attn._prefix_diagonal_attention(query, *cache[2], list(bumped[:2]), 2)
        assert not torch.allclose(moved[:, probe], base[:, probe]), (
            "entry zero is the causal prefix; if perturbing it changes nothing "
            "the test above proves nothing either"
        )

    @pytest.mark.parametrize("depth", [0, 2])
    def test_writing_the_same_depth_twice_changes_nothing(self, depth):
        """Activation checkpointing runs this forward a second time, against the
        cache the first pass already wrote to.

        An append would make the recompute see a deeper cache, a different RoPE
        offset and a differently shaped cos/sin table -- which surfaces as a
        `CheckpointError` at the first optimizer step and as nothing at all before
        it. No unit test of a single forward can see that, so what is pinned here
        is the property that makes the replay safe: writing at `draft_depth` twice
        is the same as writing once.
        """
        query, cache = _query(), _cache(depth + 1)
        attn = _DenseMLA()
        first_cache = list(cache[:depth])
        first = attn._prefix_diagonal_attention(
            query, *cache[depth], first_cache, depth
        )
        again = attn._prefix_diagonal_attention(
            query, *cache[depth], first_cache, depth
        )
        torch.testing.assert_close(again, first, atol=0.0, rtol=0.0)
        assert len(first_cache) == depth + 1

    def test_a_later_steps_own_diagonal_does_move_it(self):
        query, cache = _query(), _cache(3)
        attn = _DenseMLA()
        probe = 4
        base = attn._prefix_diagonal_attention(query, *cache[2], list(cache[:2]), 2)
        bumped = [(k.clone(), v.clone()) for k, v in cache]
        bumped[1][1][:, probe] += 5.0
        moved = attn._prefix_diagonal_attention(query, *cache[2], list(bumped[:2]), 2)
        assert not torch.allclose(moved[:, probe], base[:, probe])


class _RecordingLayer(nn.Module):
    """Stands in for `LlamaDecoderMTPLayer`, recording the cache it was handed."""

    def __init__(self):
        super().__init__()
        self.seen = []

    def forward(
        self, input_embeds, hidden_states, draft_kv_cache=None, draft_depth=0, **_ignored
    ):
        self.seen.append(
            (
                draft_kv_cache,
                None if draft_kv_cache is None else len(draft_kv_cache),
                draft_depth,
            )
        )
        if draft_kv_cache is not None:
            draft_kv_cache[draft_depth:] = [(hidden_states, hidden_states)]
        return hidden_states


@pytest.mark.unit
@pytest.mark.fast
class TestBlockThreadsOneCache:
    DEPTHS = 3

    def _run(self, draft_prefix_diagonal):
        layer = _RecordingLayer()
        block = wiring_block(
            [layer],
            [nn.Identity()],
            self.DEPTHS,
            mtp_recurrent=True,
            draft_prefix_diagonal=draft_prefix_diagonal,
        )
        block(torch.randn(1, SEQ, DIM), torch.randn(1, SEQ, DIM))
        return layer.seen

    def test_one_list_is_reused_and_grows_by_one_per_depth(self):
        seen = self._run(True)
        assert [length for _, length, _ in seen] == list(range(self.DEPTHS))
        assert [depth for _, _, depth in seen] == list(range(self.DEPTHS))
        caches = {id(cache) for cache, _, _ in seen}
        assert len(caches) == 1, "each depth must see the same cache object"

    def test_the_flag_off_passes_no_cache_at_all(self):
        """Not an empty list: an empty list is what depth 0 legitimately sees.

        `draft_depth` still counts up -- it is the cache being `None` that turns
        the geometry off, and the attention keys its RoPE offset off that.
        """
        seen = self._run(False)
        assert [cache for cache, _, _ in seen] == [None] * self.DEPTHS
