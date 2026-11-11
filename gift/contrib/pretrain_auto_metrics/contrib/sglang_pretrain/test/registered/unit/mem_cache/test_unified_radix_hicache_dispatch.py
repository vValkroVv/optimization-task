import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.mem_cache.hicache_storage import PoolName, SidecarPoolSpec
from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler
from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
    _STRATEGIES,
    StackBuildResult,
    StackStrategy,
    _apply_stack_result,
    _DeepSeekV4Strategy,
    _DsaStrategy,
    _MambaStrategy,
    _PlainKvStrategy,
    _select_strategy,
    _SwaStrategy,
    register_stack_strategy,
)
from sglang.srt.mem_cache.unified_cache_components import ComponentType
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _mock_kvcache(cls):
    return MagicMock(spec=cls)


FULL = ComponentType.FULL
SWA = ComponentType.SWA
MAMBA = ComponentType.MAMBA


class TestHybridDsaHiCache(unittest.TestCase):
    def test_hybrid_stack_preserves_indexer_and_fp8_kv_width(self):
        from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool

        kv_pool = MagicMock(spec=DSATokenToKVPool)
        kv_pool.kv_cache_dim = 656
        params = MagicMock()
        args = SimpleNamespace(
            hicache_ratio=2,
            hicache_size=0,
            hicache_mem_layout="page_first_direct",
            hicache_storage_backend=None,
            hicache_write_policy="write_through",
            hicache_io_backend="direct",
        )
        with (
            patch.object(hybrid_pool_assembler, "build_kv_host_pool") as kv_host,
            patch.object(hybrid_pool_assembler, "MambaPoolHost"),
            patch.object(hybrid_pool_assembler, "DSAIndexerPoolHost") as indexer,
            patch.object(
                hybrid_pool_assembler,
                "HostPoolGroup",
                side_effect=lambda entries: entries,
            ),
            patch.object(hybrid_pool_assembler, "HybridCacheController"),
        ):
            entries, _ = hybrid_pool_assembler.build_hybrid_mamba_stack(
                params=params,
                server_args=args,
                kv_pool=kv_pool,
                mamba_pool=MagicMock(),
                full_layer_mapping={1: 0, 3: 1},
                mamba_layer_mapping={0: 0, 2: 1},
                page_size=64,
                tp_group=None,
                load_cache_event=None,
                storage_backend=None,
                use_mla=True,
            )
        self.assertEqual(kv_host.call_args.kwargs["override_kv_cache_dim"], 656)
        self.assertEqual(
            [entry.name for entry in entries],
            [PoolName.KV, PoolName.MAMBA, PoolName.INDEXER],
        )
        indexer.assert_called_once_with(
            kv_pool, kv_host.return_value, "page_first_direct", allocator_type=None
        )
        self.assertEqual(
            [entries[2].layer_mapper(i) for i in range(4)], [None, 0, None, 1]
        )
        self.assertIs(entries[2].device_pool, kv_pool)
        self.assertIsNone(entries[2].device_alloc_fn)
        self.assertIsNone(entries[2].device_free_fn)

    def test_plain_hybrid_stack_has_no_indexer(self):
        args = MagicMock()
        with (
            patch.object(hybrid_pool_assembler, "build_kv_host_pool"),
            patch.object(hybrid_pool_assembler, "MambaPoolHost"),
            patch.object(hybrid_pool_assembler, "DSAIndexerPoolHost") as indexer,
            patch.object(
                hybrid_pool_assembler,
                "HostPoolGroup",
                side_effect=lambda entries: entries,
            ),
            patch.object(hybrid_pool_assembler, "HybridCacheController"),
        ):
            for use_mla in (False, True):
                entries, _ = hybrid_pool_assembler.build_hybrid_mamba_stack(
                    params=MagicMock(),
                    server_args=args,
                    kv_pool=object(),
                    mamba_pool=MagicMock(),
                    full_layer_mapping={1: 0},
                    mamba_layer_mapping={0: 0},
                    page_size=64,
                    tp_group=None,
                    load_cache_event=None,
                    storage_backend=None,
                    use_mla=use_mla,
                )
                self.assertEqual(
                    [entry.name for entry in entries], [PoolName.KV, PoolName.MAMBA]
                )
        indexer.assert_not_called()

    def test_strategy_registers_sidecar_with_stage_local_layers(self):
        from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool

        pool = SimpleNamespace(
            full_kv_pool=MagicMock(spec=DSATokenToKVPool),
            dense_kv_pool=None,
            start_layer=8,
            full_attention_layer_id_mapping={9: 0, 11: 1},
            use_mla=True,
        )
        params = MagicMock()
        params.req_to_token_pool.mamba_map = {8: 0, 10: 1}
        with patch.object(
            hybrid_pool_assembler,
            "build_hybrid_mamba_stack",
            return_value=(MagicMock(), MagicMock()),
        ) as build:
            result = _MambaStrategy().build(
                cache=MagicMock(),
                kvcache=pool,
                params=params,
                server_args=MagicMock(),
                load_cache_event=None,
            )
        self.assertEqual(build.call_args.kwargs["full_layer_mapping"], {1: 0, 3: 1})
        self.assertEqual(build.call_args.kwargs["mamba_layer_mapping"], {0: 0, 2: 1})
        self.assertEqual(result.transfer_layer_num, 4)
        self.assertEqual(
            result.sidecars, [SidecarPoolSpec(PoolName.INDEXER, PoolName.KV)]
        )

    def test_unsupported_hybrid_dsa_paths_fail_before_allocation(self):
        from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool

        dsa = MagicMock(spec=DSATokenToKVPool)
        args = SimpleNamespace(
            hicache_io_backend="direct", hicache_mem_layout="page_first_direct"
        )
        kwargs = dict(
            params=None,
            server_args=args,
            kv_pool=dsa,
            mamba_pool=None,
            full_layer_mapping={},
            mamba_layer_mapping={},
            page_size=64,
            tp_group=None,
            load_cache_event=None,
            use_mla=True,
        )
        with patch.object(hybrid_pool_assembler, "build_kv_host_pool") as allocate:
            with self.assertRaisesRegex(ValueError, "storage"):
                hybrid_pool_assembler.build_hybrid_mamba_stack(
                    **kwargs, storage_backend="file"
                )
            for backend, layout in (
                ("kernel", "page_first"),
                ("direct", "layer_first"),
            ):
                args.hicache_io_backend, args.hicache_mem_layout = backend, layout
                with self.assertRaisesRegex(ValueError, "direct"):
                    hybrid_pool_assembler.build_hybrid_mamba_stack(
                        **kwargs, storage_backend=None
                    )
            allocate.assert_not_called()
        pool = SimpleNamespace(full_kv_pool=dsa, dense_kv_pool=object(), start_layer=0)
        with patch.object(hybrid_pool_assembler, "build_hybrid_mamba_stack") as build:
            with self.assertRaisesRegex(ValueError, "dense"):
                _MambaStrategy().build(
                    cache=MagicMock(),
                    kvcache=pool,
                    params=MagicMock(),
                    server_args=MagicMock(),
                    load_cache_event=None,
                )
            legacy = SimpleNamespace(hybrid_kv_cache=pool, kvcache=dsa)
            with self.assertRaisesRegex(ValueError, "SGLANG_ENABLE_UNIFIED_RADIX_TREE"):
                hybrid_pool_assembler.attach_hybrid_pool_to_mamba_cache(
                    legacy,
                    MagicMock(),
                    MagicMock(),
                    extra_config={},
                    prefetch_threshold=256,
                    load_cache_event=None,
                )
            build.assert_not_called()

    def test_hybrid_indexer_reads_wait_before_layer_remapping(self):
        from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool

        for name, args in (
            ("get_index_k_with_scale_buffer", ()),
            ("get_index_k_continuous", (1, None)),
            ("get_index_k_scale_continuous", (1, None)),
            ("get_index_k_scale_buffer", (None, None, 1, 1)),
        ):
            with self.subTest(name=name):
                events = []
                pool = object.__new__(HybridLinearKVPool)
                pool.start_layer = 8
                pool.full_attention_layer_id_mapping = {9: 0}
                pool.layer_transfer_counter = SimpleNamespace(
                    wait_until=lambda layer: events.append(("wait", layer))
                )
                pool.full_kv_pool = SimpleNamespace(
                    **{name: lambda layer, *args: events.append(("read", layer))}
                )
                getattr(pool, name)(9, *args)
                self.assertEqual(events, [("wait", 1), ("read", 0)])

    def test_hybrid_mla_gather_waits_before_layer_remapping(self):
        from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool

        events = []
        pool = object.__new__(HybridLinearKVPool)
        pool.use_mla = True
        pool.dense_kv_pool = None
        pool.start_layer = 8
        pool.full_attention_layer_id_mapping = {9: 0}
        pool.layer_transfer_counter = SimpleNamespace(
            wait_until=lambda layer: events.append(("wait", layer))
        )
        pool.full_kv_pool = SimpleNamespace(
            get_mla_kv_buffer=lambda layer, *args: events.append(
                ("read", layer.layer_id)
            )
        )
        layer = SimpleNamespace(layer_id=9)
        pool.get_mla_kv_buffer(layer, None)
        self.assertEqual(events, [("wait", 1), ("read", 0)])
        self.assertEqual(layer.layer_id, 9)


class TestUnifiedRadixHiCacheDispatch(unittest.TestCase):
    def test_strategy_registry_ordering(self):
        order = [type(s) for s in _STRATEGIES]
        # DeepSeekV4 inherits from SWAKVPool, so it must resolve before _SwaStrategy.
        self.assertLess(order.index(_DeepSeekV4Strategy), order.index(_SwaStrategy))
        self.assertEqual(order[-1], _PlainKvStrategy)

    def test_deepseek_v4_full_swa(self):
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
            DeepSeekV4TokenToKVPool,
        )

        kvcache = _mock_kvcache(DeepSeekV4TokenToKVPool)
        strategy = _select_strategy(kvcache, {FULL, SWA})
        self.assertIsInstance(strategy, _DeepSeekV4Strategy)

    def test_mamba(self):
        from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool

        kvcache = _mock_kvcache(HybridLinearKVPool)
        strategy = _select_strategy(kvcache, {FULL, MAMBA})
        self.assertIsInstance(strategy, _MambaStrategy)

    def test_swa(self):
        from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool

        kvcache = _mock_kvcache(SWAKVPool)
        strategy = _select_strategy(kvcache, {FULL, SWA})
        self.assertIsInstance(strategy, _SwaStrategy)

    def test_dsa(self):
        from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool

        kvcache = _mock_kvcache(DSATokenToKVPool)
        strategy = _select_strategy(kvcache, {FULL})
        self.assertIsInstance(strategy, _DsaStrategy)

    def test_plain_kv_fallback(self):
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

        kvcache = _mock_kvcache(MHATokenToKVPool)
        strategy = _select_strategy(kvcache, {FULL})
        self.assertIsInstance(strategy, _PlainKvStrategy)

    def test_mla_routes_to_plain(self):
        from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool

        kvcache = _mock_kvcache(MLATokenToKVPool)
        strategy = _select_strategy(kvcache, {FULL})
        self.assertIsInstance(strategy, _PlainKvStrategy)

    def test_unknown_combo_raises(self):
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
            DeepSeekV4TokenToKVPool,
        )
        from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool

        for cls in (SWAKVPool, DeepSeekV4TokenToKVPool):
            kvcache = _mock_kvcache(cls)
            with self.assertRaises(AssertionError) as cm:
                _select_strategy(kvcache, {FULL})
            self.assertIn("No matching HiCache strategy", str(cm.exception))

    def test_register_custom_strategy_takes_precedence(self):
        class _CustomStrategy(StackStrategy):
            def matches(self, kvcache, components):
                return components == {FULL}

            def build(self, **_):
                raise NotImplementedError

        custom = _CustomStrategy()
        original = list(hybrid_pool_assembler._STRATEGIES)
        try:
            register_stack_strategy(custom)
            from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

            kvcache = _mock_kvcache(MHATokenToKVPool)
            self.assertIs(_select_strategy(kvcache, {FULL}), custom)
        finally:
            hybrid_pool_assembler._STRATEGIES[:] = original


class TestApplyStackResult(unittest.TestCase):
    @staticmethod
    def _fake_cache(component_types):
        cache = MagicMock()
        cache.components = {ct: MagicMock() for ct in component_types}
        return cache

    def test_wires_components_sidecars_and_counters(self):
        full_host, swa_host, mamba_host = MagicMock(), MagicMock(), MagicMock()
        cache = self._fake_cache([FULL, SWA, MAMBA])
        kvcache = MagicMock()
        params = MagicMock()
        controller = MagicMock()
        sidecar = SidecarPoolSpec(
            pool_name=PoolName.INDEXER, indices_from_pool=PoolName.KV
        )
        result = StackBuildResult(
            host_pool_group=MagicMock(),
            cache_controller=controller,
            component_host_pools={FULL: full_host, SWA: swa_host, MAMBA: mamba_host},
            sidecars=[sidecar],
            register_req_to_token_counter=True,
            transfer_layer_num=8,
            pools_desc="KV + SWA + MAMBA",
        )

        _apply_stack_result(cache, kvcache, params, result)

        self.assertIs(cache.host_pool_group, result.host_pool_group)
        self.assertIs(cache.cache_controller, controller)
        self.assertIs(cache.full_kv_pool_host, full_host)
        self.assertIs(cache.swa_kv_pool_host, swa_host)
        self.assertIs(cache.mamba_pool_host, mamba_host)
        self.assertIs(cache.components[FULL]._full_kv_pool_host, full_host)
        self.assertIs(cache.components[SWA]._swa_kv_pool_host, swa_host)
        self.assertIs(cache.components[MAMBA]._mamba_pool_host, mamba_host)
        cache.register_sidecar_pool.assert_called_once_with(sidecar)
        kvcache.register_layer_transfer_counter.assert_called_once_with(
            controller.layer_done_counter
        )
        params.req_to_token_pool.register_layer_transfer_counter.assert_called_once_with(
            controller.layer_done_counter
        )

    def test_skips_req_to_token_counter_when_flag_false(self):
        cache = self._fake_cache([FULL])
        kvcache = MagicMock()
        params = MagicMock()
        result = StackBuildResult(
            host_pool_group=MagicMock(),
            cache_controller=MagicMock(),
            component_host_pools={FULL: MagicMock()},
            sidecars=[],
            register_req_to_token_counter=False,
            transfer_layer_num=1,
            pools_desc="KV",
        )

        _apply_stack_result(cache, kvcache, params, result)

        kvcache.register_layer_transfer_counter.assert_called_once()
        params.req_to_token_pool.register_layer_transfer_counter.assert_not_called()
        cache.register_sidecar_pool.assert_not_called()


if __name__ == "__main__":
    unittest.main()
