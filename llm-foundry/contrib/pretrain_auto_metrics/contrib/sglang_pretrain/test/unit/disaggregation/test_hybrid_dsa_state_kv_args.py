import unittest
from types import SimpleNamespace

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.utils import (
    is_mla_backend,
    setup_state_kv_args,
)
from sglang.srt.mem_cache.memory_pool import (
    DSATokenToKVPool,
    HybridLinearKVPool,
    MHATokenToKVPool,
)


def _bare(cls, **attrs):
    obj = object.__new__(cls)
    obj.__dict__.update(attrs)
    return obj


def _hybrid(full_kv_pool, use_mla=True):
    mamba_pool = SimpleNamespace(
        get_contiguous_buf_infos=lambda: ([10, 11], [100, 110], [1, 1]),
        get_state_dim_per_tensor=lambda: [7, 7],
    )
    return _bare(
        HybridLinearKVPool,
        use_mla=use_mla,
        mamba_pool=mamba_pool,
        full_kv_pool=full_kv_pool,
    )


def _dsa_pool(ptr):
    return _bare(
        DSATokenToKVPool,
        get_state_buf_infos=lambda: ([ptr], [1000], [64]),
    )


class TestHybridDsaStateKvArgs(unittest.TestCase):
    def test_hybrid_dsa_registers_mamba_and_indexer_state(self):
        kv_args = SimpleNamespace()
        setup_state_kv_args(kv_args, _hybrid(_dsa_pool(20)))

        self.assertEqual(kv_args.state_types, [StateType.MAMBA, StateType.DSA])
        self.assertEqual(kv_args.state_data_ptrs, [[10, 11], [20]])
        self.assertEqual(kv_args.state_item_lens, [[1, 1], [64]])
        self.assertEqual(kv_args.state_dim_per_tensor, [[7, 7], []])

    def test_hybrid_dsa_appends_draft_indexer_state(self):
        kv_args = SimpleNamespace()
        setup_state_kv_args(
            kv_args, _hybrid(_dsa_pool(20)), draft_token_to_kv_pool=_hybrid(_dsa_pool(30))
        )

        self.assertEqual(kv_args.state_data_ptrs[1], [20, 30])
        self.assertEqual(kv_args.state_data_lens[1], [1000, 1000])

    def test_hybrid_mha_registers_only_mamba(self):
        kv_args = SimpleNamespace()
        setup_state_kv_args(kv_args, _hybrid(_bare(MHATokenToKVPool), use_mla=False))

        self.assertEqual(kv_args.state_types, [StateType.MAMBA])

    def test_is_mla_backend_follows_hybrid_use_mla(self):
        self.assertTrue(is_mla_backend(_hybrid(_dsa_pool(20))))
        self.assertFalse(is_mla_backend(_hybrid(_bare(MHATokenToKVPool), use_mla=False)))


if __name__ == "__main__":
    unittest.main()
