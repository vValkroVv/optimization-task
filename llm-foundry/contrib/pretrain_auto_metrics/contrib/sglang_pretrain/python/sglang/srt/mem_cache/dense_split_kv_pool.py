"""KV cache for a non-hybrid DSA model whose ``dense_attention_layers`` run as
plain MLA.

Sparse DSA layers live in a ``DSATokenToKVPool`` (latent + indexer k cache,
paged-fp8 layout when the cache is fp8); dense (indexer-removed) layers live in
a plain ``MLATokenToKVPool`` (standard latent, standard fp8 with per-layer
k_scale, read natively by the flashmla backend) -- byte-for-byte the cache a
no-DSA MLA model uses. Callers address
layers by their global id; every access is routed to the owning sub-pool and
its local layer id. This is the non-hybrid counterpart of the dense sub-pool
inside ``HybridLinearKVPool``.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, List, Optional

import torch

from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.memory_pool import (
    DSATokenToKVPool,
    KVCache,
    MLATokenToKVPool,
)

if TYPE_CHECKING:
    from sglang.srt.managers.cache_controller import LayerDoneCounter


class DenseSplitDSAKVPool(KVCache):
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        dsa_layer_ids: List[int],
        dense_layer_ids: List[int],
        device: str,
        index_head_dim: int,
        enable_memory_saver: bool,
        kv_cache_dim: int,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
    ):
        assert dsa_layer_ids and dense_layer_ids, (
            "DenseSplitDSAKVPool needs both sparse DSA and dense layers; use "
            "DSATokenToKVPool / MLATokenToKVPool otherwise"
        )
        assert not (set(dsa_layer_ids) & set(dense_layer_ids))
        self.size = size
        self.page_size = page_size
        self.dtype = dtype
        self.device = device
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.index_head_dim = index_head_dim
        self.use_mla = True
        self.use_dsa = True
        self.dsa_layer_ids = sorted(dsa_layer_ids)
        self.dense_layer_ids = sorted(dense_layer_ids)
        self.layer_num = len(self.dsa_layer_ids) + len(self.dense_layer_ids)
        self.start_layer = start_layer if start_layer is not None else 0
        self.end_layer = (
            end_layer
            if end_layer is not None
            else self.start_layer + self.layer_num - 1
        )
        self.layer_transfer_counter = None
        # Sub-pools are indexed by their own local ids (start_layer=0).
        self.dsa_kv_pool = DSATokenToKVPool(
            size=size,
            page_size=page_size,
            kv_lora_rank=kv_lora_rank,
            dtype=dtype,
            qk_rope_head_dim=qk_rope_head_dim,
            layer_num=len(self.dsa_layer_ids),
            device=device,
            index_head_dim=index_head_dim,
            enable_memory_saver=enable_memory_saver,
            kv_cache_dim=kv_cache_dim,
        )
        self.dense_kv_pool = MLATokenToKVPool(
            size=size,
            page_size=page_size,
            dtype=dtype,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            layer_num=len(self.dense_layer_ids),
            device=device,
            enable_memory_saver=enable_memory_saver,
        )
        self._dsa_local = {gid: i for i, gid in enumerate(self.dsa_layer_ids)}
        self._dense_local = {gid: i for i, gid in enumerate(self.dense_layer_ids)}
        # DSA-facing attributes (the DSA backend / indexer read these).
        self.store_dtype = self.dsa_kv_pool.store_dtype
        self.kv_cache_dim = self.dsa_kv_pool.kv_cache_dim
        self.dsa_kv_cache_store_fp8 = self.dsa_kv_pool.dsa_kv_cache_store_fp8
        self.memory_saver_adapter = self.dsa_kv_pool.memory_saver_adapter
        self.custom_mem_pool = getattr(self.dsa_kv_pool, "custom_mem_pool", None)
        self.enable_custom_mem_pool = getattr(
            self.dsa_kv_pool, "enable_custom_mem_pool", False
        )
        self.cpu_offloading_chunk_size = self.dsa_kv_pool.cpu_offloading_chunk_size
        self.mem_usage = self.dsa_kv_pool.mem_usage + self.dense_kv_pool.mem_usage

    # ---------------------------------------------------------------- routing
    def _route(self, layer_id: int):
        """(sub_pool, local_layer_id) for a global layer id."""
        local = self._dense_local.get(layer_id)
        if local is not None:
            return self.dense_kv_pool, local
        local = self._dsa_local.get(layer_id)
        if local is None:
            raise ValueError(
                f"layer {layer_id} is neither a DSA layer {self.dsa_layer_ids} "
                f"nor a dense layer {self.dense_layer_ids}"
            )
        return self.dsa_kv_pool, local

    def _dsa_local_id(self, layer_id: int) -> int:
        if layer_id not in self._dsa_local:
            raise ValueError(
                f"layer {layer_id} has no DSA indexer cache (dense layer or unknown)"
            )
        return self._dsa_local[layer_id]

    @contextmanager
    def _local_layer_id(self, layer: RadixAttention):
        original = layer.layer_id
        _, layer.layer_id = self._route(original)
        try:
            yield
        finally:
            layer.layer_id = original

    def _wait_for_layer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

    # ------------------------------------------------------------- accounting
    def get_kv_size_bytes(self):
        return (
            self.dsa_kv_pool.get_kv_size_bytes()
            + self.dense_kv_pool.get_kv_size_bytes()
        )

    def get_contiguous_buf_infos(self):
        # PD-disaggregation KV transfer: the DSA and dense sub-pools have
        # different item sizes and layer order, unsupported.
        raise NotImplementedError(
            "get_contiguous_buf_infos (PD-disaggregation KV transfer) is not "
            "supported with a DSA dense/plain-MLA split KV pool"
        )

    def get_state_buf_infos(self):
        return self.dsa_kv_pool.get_state_buf_infos()

    def maybe_get_custom_mem_pool(self):
        return self.dsa_kv_pool.maybe_get_custom_mem_pool()

    def register_layer_transfer_counter(self, layer_transfer_counter: LayerDoneCounter):
        # Layer-wise waits happen here with global ids; sub-pools never wait.
        self.layer_transfer_counter = layer_transfer_counter
        self.dsa_kv_pool.register_layer_transfer_counter(None)
        self.dense_kv_pool.register_layer_transfer_counter(None)

    # -------------------------------------------------------- latent KV cache
    def get_key_buffer(self, layer_id: int):
        self._wait_for_layer(layer_id)
        pool, local = self._route(layer_id)
        return pool.get_key_buffer(local)

    def get_value_buffer(self, layer_id: int):
        self._wait_for_layer(layer_id)
        pool, local = self._route(layer_id)
        return pool.get_value_buffer(local)

    def get_kv_buffer(self, layer_id: int):
        self._wait_for_layer(layer_id)
        pool, local = self._route(layer_id)
        return pool.get_kv_buffer(local)

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
    ):
        pool, _ = self._route(layer.layer_id)
        with self._local_layer_id(layer):
            pool.set_kv_buffer(layer, loc, cache_k, cache_v)

    def set_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k_nope: torch.Tensor,
        cache_k_rope: torch.Tensor,
    ):
        pool, _ = self._route(layer.layer_id)
        with self._local_layer_id(layer):
            pool.set_mla_kv_buffer(layer, loc, cache_k_nope, cache_k_rope)

    def get_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        dst_dtype: Optional[torch.dtype] = None,
    ):
        pool, _ = self._route(layer.layer_id)
        with self._local_layer_id(layer):
            return pool.get_mla_kv_buffer(layer, loc, dst_dtype)

    def get_v_head_dim(self):
        return self.dsa_kv_pool.get_v_head_dim()

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        self.dsa_kv_pool.move_kv_cache(tgt_loc, src_loc)
        self.dense_kv_pool.move_kv_cache(tgt_loc, src_loc)

    def get_cpu_copy(self, indices, mamba_indices=None):
        raise NotImplementedError(
            "CPU offload is not supported with a dense/DSA split KV cache"
        )

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        raise NotImplementedError(
            "CPU offload is not supported with a dense/DSA split KV cache"
        )

    # ------------------------------------------------- DSA indexer key cache
    def get_index_k_with_scale_buffer(self, layer_id: int) -> torch.Tensor:
        self._wait_for_layer(layer_id)
        return self.dsa_kv_pool.get_index_k_with_scale_buffer(
            self._dsa_local_id(layer_id)
        )

    def get_index_k_continuous(self, layer_id: int, seq_len: int, page_indices):
        self._wait_for_layer(layer_id)
        return self.dsa_kv_pool.get_index_k_continuous(
            self._dsa_local_id(layer_id), seq_len, page_indices
        )

    def get_index_k_scale_continuous(self, layer_id: int, seq_len: int, page_indices):
        self._wait_for_layer(layer_id)
        return self.dsa_kv_pool.get_index_k_scale_continuous(
            self._dsa_local_id(layer_id), seq_len, page_indices
        )

    def get_index_k_scale_buffer(
        self,
        layer_id: int,
        seq_len_tensor,
        page_indices,
        seq_len_sum: int,
        max_seq_len: int,
    ):
        self._wait_for_layer(layer_id)
        return self.dsa_kv_pool.get_index_k_scale_buffer(
            self._dsa_local_id(layer_id),
            seq_len_tensor,
            page_indices,
            seq_len_sum,
            max_seq_len,
        )

    def set_index_k_scale_buffer(
        self, layer_id: int, loc, index_k, index_k_scale
    ) -> None:
        return self.dsa_kv_pool.set_index_k_scale_buffer(
            self._dsa_local_id(layer_id), loc, index_k, index_k_scale
        )

    def __getattr__(self, name: str):
        # Anything not routed above is a DSA-pool-wide property.
        if name in ("dsa_kv_pool", "dense_kv_pool"):
            raise AttributeError(name)
        return getattr(self.dsa_kv_pool, name)
