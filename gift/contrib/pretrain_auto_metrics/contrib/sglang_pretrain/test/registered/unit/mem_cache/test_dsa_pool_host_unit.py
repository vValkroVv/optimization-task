import inspect
import unittest

import torch

from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool
from sglang.srt.mem_cache.memory_pool_host import (
    DSAIndexerPoolHost,
    HostPoolGroup,
    MLATokenToKVPoolHost,
    PoolEntry,
)
from sglang.srt.mem_cache.pool_host.common import (
    ALLOC_MEMORY_FUNCS,
    alloc_with_pin_memory,
)
from sglang.srt.utils import is_cuda, is_hip, is_npu, is_xpu
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=9, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=9, suite="stage-b-test-1-gpu-small-amd")


class TestDSAOffloadSignatures(unittest.TestCase):
    def test_cpu_copy_methods_accept_mamba_indices(self):
        for method_name in ("get_cpu_copy", "load_cpu_copy"):
            with self.subTest(method_name=method_name):
                signature = inspect.signature(getattr(DSATokenToKVPool, method_name))
                self.assertIn("mamba_indices", signature.parameters)


class TestDSAHiCacheTransfer(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for DSA host transfer tests.")
        if is_npu() or is_xpu():
            self.skipTest("DSA host transfer tests only support CUDA/ROCm.")
        if not (is_cuda() or is_hip()):
            self.skipTest("CUDA/ROCm not available.")

    @staticmethod
    def _token_indices_for_pages(pages: torch.Tensor, page_size: int, device: str):
        parts = [
            torch.arange(
                int(page_id) * page_size,
                (int(page_id) + 1) * page_size,
                device=device,
                dtype=torch.int64,
            )
            for page_id in pages.tolist()
        ]
        return torch.cat(parts, dim=0)

    def _run_device_to_host_indexer_copy(
        self, io_backend: str, layout="layer_first", dtype=torch.bfloat16
    ):
        page_size = 1 if is_hip() else 64
        layer_num = 2
        size = page_size * 4

        device_pool = DSATokenToKVPool(
            size=size,
            page_size=page_size,
            kv_lora_rank=512 if dtype == torch.float8_e4m3fn else 128,
            dtype=dtype,
            qk_rope_head_dim=64 if dtype == torch.float8_e4m3fn else 32,
            layer_num=layer_num,
            device="cuda",
            enable_memory_saver=False,
            kv_cache_dim=656 if dtype == torch.float8_e4m3fn else 576,
            index_head_dim=128,
        )
        pin_memory = io_backend == "kernel"
        original_alloc = ALLOC_MEMORY_FUNCS["cuda"]
        if pin_memory:
            ALLOC_MEMORY_FUNCS["cuda"] = alloc_with_pin_memory
        try:
            mla_host = MLATokenToKVPoolHost(
                device_pool=device_pool,
                host_to_device_ratio=2.0,
                host_size=0,
                page_size=page_size,
                layout=layout,
                pin_memory=pin_memory,
                device="cpu",
                allocator_type="default",
                override_kv_cache_dim=device_pool.kv_cache_dim,
            )
            indexer_host = DSAIndexerPoolHost(
                device_pool=device_pool,
                anchor_host=mla_host,
                layout=layout,
                pin_memory=pin_memory,
                device="cpu",
                allocator_type="default",
            )
        finally:
            ALLOC_MEMORY_FUNCS["cuda"] = original_alloc

        for layer_id in range(layer_num):
            buf = device_pool.index_k_with_scale_buffer[layer_id]
            data = torch.arange(
                buf.numel(), device=buf.device, dtype=torch.int32
            ).view_as(buf)
            buf.copy_(((data + layer_id) % 251).to(buf.dtype))
            self.assertFalse(torch.equal(buf[1], buf[2]))
            kv_buf = device_pool.kv_buffer[layer_id]
            kv_data = torch.arange(
                kv_buf.numel(), device=kv_buf.device, dtype=torch.int32
            ).view_as(kv_buf)
            kv_buf.copy_((kv_data % 127 + 32 * layer_id).to(kv_buf.dtype))

        device_pages = torch.tensor([1, 2, 3], device="cuda", dtype=torch.int64)
        host_pages = torch.tensor(
            [0, 1, 2],
            device="cuda" if io_backend == "kernel" else "cpu",
            dtype=torch.int64,
        )
        device_indices = self._token_indices_for_pages(
            device_pages, page_size, device="cuda"
        )
        host_indices = self._token_indices_for_pages(
            host_pages,
            page_size,
            device="cuda" if io_backend == "kernel" else "cpu",
        )

        mla_host.backup_from_device_all_layer(
            device_pool, host_indices, device_indices, io_backend
        )
        indexer_host.backup_from_device_all_layer(
            device_pool, host_indices, device_indices, io_backend
        )

        for layer_id in range(layer_num):
            for host_page, device_page in zip(
                host_pages.tolist(), device_pages.tolist()
            ):
                if layout == "page_first_direct":
                    got = indexer_host.index_k_with_scale_buffer[
                        host_page, layer_id, 0
                    ].cpu()
                else:
                    got = indexer_host.index_k_with_scale_buffer[layer_id][
                        host_page
                    ].cpu()
                expected = device_pool.index_k_with_scale_buffer[layer_id][
                    device_page
                ].cpu()
                self.assertTrue(torch.equal(got, expected))
                host_start = host_page * page_size
                device_start = device_page * page_size
                if layout == "page_first_direct":
                    got_kv = mla_host.kv_buffer[host_page, layer_id].cpu()
                else:
                    got_kv = mla_host.kv_buffer[layer_id][
                        host_start : host_start + page_size
                    ].cpu()
                expected_kv = device_pool.kv_buffer[layer_id][
                    device_start : device_start + page_size
                ].cpu()
                self.assertTrue(torch.equal(got_kv, expected_kv))

        expected_kv = [buf[device_indices].clone() for buf in device_pool.kv_buffer]
        expected_indexer = [
            buf[device_pages].clone() for buf in device_pool.index_k_with_scale_buffer
        ]
        for layer_id in range(layer_num):
            device_pool.kv_buffer[layer_id][device_indices] = 0
            device_pool.index_k_with_scale_buffer[layer_id][device_pages] = 0

        group = HostPoolGroup(
            [
                PoolEntry(
                    name=PoolName.KV,
                    host_pool=mla_host,
                    device_pool=device_pool,
                    layer_mapper=lambda layer: layer,
                    is_primary_index_anchor=True,
                ),
                PoolEntry(
                    name=PoolName.INDEXER,
                    host_pool=indexer_host,
                    device_pool=device_pool,
                    layer_mapper=lambda layer: layer,
                ),
            ]
        )
        indexer_transfer = PoolTransfer(
            name=PoolName.INDEXER,
            host_indices=host_indices,
            device_indices=device_indices,
        )
        for layer_id in range(layer_num):
            group.load_to_device_per_layer(
                device_pool,
                host_indices,
                device_indices,
                layer_id,
                io_backend,
                pool_transfers=[indexer_transfer],
            )
        torch.cuda.synchronize()
        for layer_id in range(layer_num):
            self.assertTrue(
                torch.equal(
                    device_pool.kv_buffer[layer_id][device_indices],
                    expected_kv[layer_id],
                )
            )
            self.assertTrue(
                torch.equal(
                    device_pool.index_k_with_scale_buffer[layer_id][device_pages],
                    expected_indexer[layer_id],
                )
            )

    @unittest.skipIf(
        is_hip(),
        '`io_backend="kernel"` path in memory_pool_host.backup_from_device_all_layer '
        "raises ValueError on AMD (only the `direct` IO backend is wired for ROCm). "
        "The other 62 tests in this file pass on AMD.",
    )
    def test_device_to_host_indexer_kernel(self):
        self._run_device_to_host_indexer_copy(io_backend="kernel")

    @unittest.skipIf(
        is_hip(),
        "DSATokenToKVPool with page_size=1 (used on HIP) trips the ROCm 7.2.0 "
        "HIP-preshuffle assert `page_size % 16 == 0` during pool construction "
        "(seen on pr-test-amd-rocm720). The rest of this file passes on AMD.",
    )
    def test_device_to_host_indexer_direct(self):
        self._run_device_to_host_indexer_copy(io_backend="direct")

    @unittest.skipIf(is_hip(), "Packed FP8 DSA page layout is CUDA-specific.")
    def test_indexer_roundtrip_direct_fp8_page_first(self):
        self._run_device_to_host_indexer_copy(
            io_backend="direct", layout="page_first_direct", dtype=torch.float8_e4m3fn
        )


if __name__ == "__main__":
    unittest.main()
