"""Attention backend router for a non-hybrid DSA model with ``dense_attention_layers``.

Sparse DSA layers run on the DSA backend; dense (indexer-removed) layers run
on the plain-MLA flashmla backend -- fp8-native absorbed-MLA decode, the same
kernel family the DSA layers use, so an fp8 KV cache needs no per-step
dequantization -- over the matching ``DenseSplitDSAKVPool`` sub-pool. This is
the non-hybrid counterpart of ``HybridLinearAttnBackend.dense_full_attn_backend``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.radix_attention import RadixAttention

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

# The plain-MLA backend dense layers run on: flashmla, not fa3 (fa3's fp8
# absorbed-MLA path dequantizes the whole per-layer KV buffer -> OOM).
DSA_DENSE_LAYER_BACKEND = "flashmla"

# Batch-level DSA attributes the model dispatch reads off get_attn_backend();
# forwarded to the DSA backend. Anything else raises AttributeError so a typo
# is not silently answered by the wrong backend.
_FORWARDED_TO_DSA = frozenset(
    {
        "use_mha",
        "kv_cache_dtype",
        "dsa_kv_cache_store_fp8",
        "data_type",
        "num_kv_splits",
        "kv_lora_rank",
        "qk_rope_head_dim",
        "page_size",
        "linear_attn_backend",
        "speculative_num_draft_tokens",
        "enable_chunk_kv",
        "pack_prefix_chunk_kv",
        "decode_attention_backend_str",
        "prefill_attention_backend_str",
        "forward",
        "forward_compress",
    }
)


class DenseSplitAttnBackend(AttentionBackend):
    def __init__(
        self,
        dsa_backend: AttentionBackend,
        dense_backend: AttentionBackend,
        dense_layer_ids: List[int],
    ):
        self.dsa_backend = dsa_backend
        self.dense_backend = dense_backend
        self.dense_layer_ids = set(dense_layer_ids)
        # DSA first: it decides the batch-level prefill regime (use_mha) that
        # the model's dispatch reads through this wrapper.
        self.attn_backend_list = [dsa_backend, dense_backend]
        self.token_to_kv_pool = dsa_backend.token_to_kv_pool
        self.req_to_token_pool = dsa_backend.req_to_token_pool
        self.max_context_len = getattr(dsa_backend, "max_context_len", None)
        self.needs_cpu_seq_lens = (
            dsa_backend.needs_cpu_seq_lens or dense_backend.needs_cpu_seq_lens
        )

    @property
    def forward_metadata(self):
        # Delegated to the DSA backend (a property, so a runner reset of it is
        # not shadowed by a stale attribute here).
        return self.dsa_backend.forward_metadata

    @forward_metadata.setter
    def forward_metadata(self, value):
        for backend in self.attn_backend_list:
            backend.forward_metadata = value

    def _backend_for(
        self, layer: Optional[RadixAttention], layer_id: Optional[int] = None
    ) -> AttentionBackend:
        if layer is not None:
            layer_id = layer.layer_id
        if layer_id in self.dense_layer_ids:
            return self.dense_backend
        return self.dsa_backend

    # ------------------------------------------------------------ metadata
    def init_forward_metadata(self, forward_batch: ForwardBatch):
        for backend in self.attn_backend_list:
            backend.init_forward_metadata(forward_batch)

    def init_forward_metadata_out_graph(
        self, forward_batch: ForwardBatch, in_capture: bool = False
    ):
        for backend in self.attn_backend_list:
            backend.init_forward_metadata_out_graph(
                forward_batch, in_capture=in_capture
            )

    def init_forward_metadata_in_graph(self, forward_batch: ForwardBatch):
        for backend in self.attn_backend_list:
            backend.init_forward_metadata_in_graph(forward_batch)

    def init_mha_chunk_metadata(
        self, forward_batch: ForwardBatch, disable_flashinfer_ragged: bool = False
    ):
        # Called by the first layer of the batch that prefills as MHA with a
        # prefix (one-shot or chunked), sparse or dense. A plain-MLA backend whose
        # chunked prefix cache is off has no chunk runner to plan.
        for backend in self.attn_backend_list:
            init = getattr(backend, "init_mha_chunk_metadata", None)
            if init is not None and getattr(backend, "enable_chunk_kv", True):
                init(forward_batch, disable_flashinfer_ragged)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        for backend in self.attn_backend_list:
            backend.init_cuda_graph_state(max_bs, max_num_tokens)

    def on_after_cuda_graph_warmup(self):
        for backend in self.attn_backend_list:
            backend.on_after_cuda_graph_warmup()

    def get_cuda_graph_seq_len_fill_value(self):
        return self.dsa_backend.get_cuda_graph_seq_len_fill_value()

    def get_verify_buffers_to_fill_after_draft(self):
        return self.dsa_backend.get_verify_buffers_to_fill_after_draft()

    def update_verify_buffers_to_fill_after_draft(self, spec_info, cuda_graph_bs):
        for backend in self.attn_backend_list:
            backend.update_verify_buffers_to_fill_after_draft(spec_info, cuda_graph_bs)

    def get_indexer_metadata(self, layer_id: int, forward_batch: ForwardBatch):
        return self.dsa_backend.get_indexer_metadata(layer_id, forward_batch)

    def support_triton(self):
        return all(backend.support_triton() for backend in self.attn_backend_list)

    # ------------------------------------------------------------- forward
    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        return self._backend_for(layer).forward_decode(
            q, k, v, layer, forward_batch, save_kv_cache, **kwargs
        )

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        return self._backend_for(layer).forward_extend(
            q, k, v, layer, forward_batch, save_kv_cache, **kwargs
        )

    def __getattr__(self, name: str):
        if name not in _FORWARDED_TO_DSA:
            raise AttributeError(name)
        return getattr(self.dsa_backend, name)
