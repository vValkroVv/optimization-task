"""Unit tests for DenseSplitAttnBackend (non-hybrid DSA model with
dense_attention_layers): layer routing, batch-level state delegation to the
DSA backend and the chunked-prefix hooks, with stub sub-backends."""

from types import SimpleNamespace

import pytest

from sglang.srt.layers.attention.dense_split_attn_backend import (
    DenseSplitAttnBackend,
)

pytestmark = pytest.mark.unit


class _Stub:
    def __init__(self, name, enable_chunk_kv=None):
        self.name = name
        self.token_to_kv_pool = object()
        self.req_to_token_pool = object()
        self.needs_cpu_seq_lens = False
        self.forward_metadata = None
        self.calls = []
        if enable_chunk_kv is not None:
            self.enable_chunk_kv = enable_chunk_kv

    def init_mha_chunk_metadata(self, forward_batch, disable_flashinfer_ragged=False):
        self.calls.append(("init_mha_chunk_metadata", forward_batch))

    def forward_extend(self, q, k, v, layer, forward_batch, save_kv_cache=True, **kw):
        return self.name

    def forward_decode(self, q, k, v, layer, forward_batch, save_kv_cache=True, **kw):
        return self.name


def _wrapper(dense_chunk=True):
    dsa = _Stub("dsa")
    dsa.use_mha = True
    dense = _Stub("dense", enable_chunk_kv=dense_chunk)
    return DenseSplitAttnBackend(dsa, dense, dense_layer_ids=[1, 3]), dsa, dense


def _layer(layer_id):
    return SimpleNamespace(layer_id=layer_id)


class TestRouting:
    def test_dense_layers_go_to_the_dense_backend(self):
        w, dsa, dense = _wrapper()
        assert w._backend_for(_layer(1)) is dense
        assert w._backend_for(_layer(3)) is dense
        assert w._backend_for(_layer(0)) is dsa
        assert w.forward_extend(None, None, None, _layer(3), None) == "dense"
        assert w.forward_decode(None, None, None, _layer(2), None) == "dsa"


class TestBatchLevelState:
    def test_dsa_attributes_are_delegated(self):
        w, dsa, _ = _wrapper()
        assert w.use_mha is True
        dsa.use_mha = False
        assert w.use_mha is False

    def test_forward_metadata_follows_the_dsa_backend_after_a_reset(self):
        # The runner drops stale metadata on an unpadded DP-attention idle batch
        # (attn_backend.forward_metadata = None); the next batch's DSA metadata
        # must still be visible through the wrapper.
        w, dsa, dense = _wrapper()
        dsa.forward_metadata = "batch-1"
        assert w.forward_metadata == "batch-1"
        w.forward_metadata = None
        assert dsa.forward_metadata is None and dense.forward_metadata is None
        dsa.forward_metadata = "batch-2"
        assert w.forward_metadata == "batch-2"


class TestChunkedPrefixHooks:
    def test_dense_backend_without_chunk_runner_is_skipped(self):
        w, dsa, dense = _wrapper(dense_chunk=False)
        w.init_mha_chunk_metadata("fb")
        assert dense.calls == []
        assert dsa.calls == [("init_mha_chunk_metadata", "fb")]

    def test_dense_backend_with_chunk_runner_is_planned(self):
        w, dsa, dense = _wrapper(dense_chunk=True)
        w.init_mha_chunk_metadata("fb")
        assert dense.calls == [("init_mha_chunk_metadata", "fb")]
