"""MLA page_first_direct HiCache loads with page size 1 use the per-layer gather kernel."""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from sglang.srt.mem_cache import memory_pool_host
from sglang.srt.mem_cache.memory_pool_host import MLATokenToKVPoolHost

pytestmark = pytest.mark.unit

WIDTH = 576


def _host(kv_buffer, *, page_size=1, pin_memory=True):
    host = MLATokenToKVPoolHost.__new__(MLATokenToKVPoolHost)
    host.layout = "page_first_direct"
    host.page_size = page_size
    host.pin_memory = pin_memory
    host.kv_buffer = kv_buffer
    host.layer_num = kv_buffer.shape[1]
    host.token_stride_size = WIDTH * kv_buffer.element_size()
    host.layout_dim = host.token_stride_size * host.layer_num
    return host


@pytest.mark.parametrize(
    "page_size,pin_memory,kernel",
    [(1, True, "mla"), (2, True, "direct"), (1, False, "direct")],
)
def test_dispatch(page_size, pin_memory, kernel):
    host = _host(
        torch.zeros((4, 2, page_size, 1, WIDTH), dtype=torch.bfloat16),
        page_size=page_size,
        pin_memory=pin_memory,
    )
    pool = SimpleNamespace(kv_buffer=[torch.zeros((8, 1, WIDTH)) for _ in range(2)])
    hi, di = torch.arange(page_size), torch.arange(page_size) + 2
    with (
        mock.patch.object(memory_pool_host, "transfer_kv_per_layer_mla_pf_lf") as mla,
        mock.patch.object(
            memory_pool_host, "transfer_kv_per_layer_direct_pf_lf"
        ) as direct,
    ):
        host.load_to_device_per_layer(pool, hi, di, 1, "direct")

    if kernel == "direct":
        mla.assert_not_called()
        direct.assert_called_once()
        return
    direct.assert_not_called()
    kwargs = mla.call_args.kwargs
    assert kwargs["src"] is host.kv_buffer and kwargs["dst"] is pool.kv_buffer[1]
    assert torch.equal(kwargs["src_indices"], hi)
    assert torch.equal(kwargs["dst_indices"], di)
    assert kwargs["layer_id"] == 1
    assert kwargs["item_size"] == WIDTH * 2
    assert kwargs["src_layout_dim"] == 2 * WIDTH * 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_matches_direct_copy_across_large_host_offsets():
    tokens, layers = 131072, 32  # ~4.8 GB pinned: host offsets cross 2 and 4 GiB
    kv_buffer = torch.empty(
        (tokens, layers, 1, 1, WIDTH), dtype=torch.bfloat16, pin_memory=True
    )
    host = _host(kv_buffer)
    rows = {0, 1, tokens - 2, tokens - 1}
    for boundary in (2**31, 2**32):
        pivot = boundary // host.layout_dim
        rows.update((pivot - 1, pivot, pivot + 1))
    hi = torch.tensor(sorted(rows), dtype=torch.int64)
    di = torch.arange(len(rows), dtype=torch.int64) * 2 + 1
    for layer in range(layers):
        kv_buffer[hi, layer] = (
            (torch.arange(WIDTH) + 17 * hi[:, None] + 7 * layer) % 251 - 125
        ).to(torch.bfloat16)[:, None, None]

    def pool():
        return SimpleNamespace(
            kv_buffer=[
                torch.full((32, 1, WIDTH), -256, dtype=torch.bfloat16, device="cuda")
                for _ in range(layers)
            ]
        )

    gathered, copied = pool(), pool()
    load_stream = torch.cuda.Stream()
    with torch.cuda.stream(load_stream):
        for layer in range(layers):
            host.load_to_device_per_layer(gathered, hi, di, layer, "direct")
            memory_pool_host.transfer_kv_per_layer_direct_pf_lf(
                src_ptrs=[kv_buffer],
                dst_ptrs=[copied.kv_buffer[layer]],
                src_indices=hi,
                dst_indices=di,
                layer_id=layer,
                page_size=1,
            )
    torch.cuda.synchronize()

    for layer in range(layers):
        assert torch.equal(gathered.kv_buffer[layer], copied.kv_buffer[layer]), layer
        assert torch.equal(gathered.kv_buffer[layer][di].cpu(), kv_buffer[hi, layer, 0])
