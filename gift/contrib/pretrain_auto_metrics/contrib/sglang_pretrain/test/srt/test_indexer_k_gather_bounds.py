import importlib.util
import os
import sys
import types
import unittest

PAGE_SIZE = 64
BUF_NUMEL_PER_PAGE = PAGE_SIZE * (128 + 4)
INT32_MAX = 2**31 - 1

PD_PREFILL_PAGES = (18979200 + PAGE_SIZE + 1) // PAGE_SIZE
AGG_PAGES = (11400000 + PAGE_SIZE + 1) // PAGE_SIZE


def _stub(name, **attrs):
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    sys.modules[name] = module


class FakeKernel:
    def __init__(self):
        self.launched = {}

    def __getitem__(self, grid):
        def launch(**kwargs):
            self.launched.update(kwargs)

        return launch


class FakeTensor:
    def __init__(self, shape, dtype="int32"):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = "cuda:0"

    def numel(self):
        n = 1
        for d in self.shape:
            n *= d
        return n

    def to(self, dtype):
        return FakeTensor(self.shape, dtype)


def _load_accessor():
    _stub(
        "torch",
        Tensor=FakeTensor,
        uint8="uint8",
        int32="int32",
        int64="int64",
        empty=lambda *a, **k: FakeTensor((0,)),
    )
    _stub("triton", jit=lambda fn: fn)
    _stub(
        "triton.language",
        constexpr=object,
        arange=lambda *a, **k: None,
        load=lambda *a, **k: None,
        store=lambda *a, **k: None,
        sum=lambda *a, **k: None,
        program_id=lambda *a, **k: 0,
    )
    _stub(
        "sglang.srt.layers.attention.dsa.utils",
        aiter_can_use_preshuffle_paged_mqa=lambda: False,
    )
    _stub("sglang.srt.layers.quantization.fp8_kernel", is_fp8_fnuz=lambda: False)
    _stub(
        "sglang.srt.utils",
        get_bool_env_var=lambda *a, **k: False,
        is_hip=lambda: False,
    )

    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "python",
        "sglang",
        "srt",
        "layers",
        "attention",
        "dsa",
        "index_buf_accessor.py",
    )
    spec = importlib.util.spec_from_file_location("index_buf_accessor_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestPageIndexDtype(unittest.TestCase):
    def setUp(self):
        self.accessor = _load_accessor()

    def _page_indices(self, num_pages, seq_lens):
        buf = FakeTensor((num_pages, BUF_NUMEL_PER_PAGE))
        cols = max((n + PAGE_SIZE - 1) // PAGE_SIZE for n in seq_lens)
        table = FakeTensor((len(seq_lens), cols))
        return buf, table

    def test_pd_prefill_pool_needs_int64(self):
        buf, table = self._page_indices(PD_PREFILL_PAGES, [1 << 20])
        self.assertGreater(buf.numel(), INT32_MAX)
        out = self.accessor._int64_page_indices(table, buf)
        self.assertEqual(out.dtype, "int64")

    def test_aggregated_pool_stays_int32(self):
        buf, table = self._page_indices(AGG_PAGES, [1 << 20])
        self.assertLessEqual(buf.numel(), INT32_MAX)
        out = self.accessor._int64_page_indices(table, buf)
        self.assertIs(out, table)

    def test_gather_casts_before_launching_the_kernel(self):
        kernel = FakeKernel()
        self.accessor._get_k_and_s_triton_kernel = kernel
        buf, table = self._page_indices(PD_PREFILL_PAGES, [1 << 20])
        self.accessor._get_k_and_s_triton(
            buf=buf,
            page_indices=table,
            seq_lens=FakeTensor((1,)),
            seq_len_sum=1 << 20,
            max_seq_len=1 << 20,
            page_size=PAGE_SIZE,
            index_head_dim=128,
        )
        self.assertEqual(kernel.launched["page_indices_ptr"].dtype, "int64")


class TestPageTableWidth(unittest.TestCase):
    def setUp(self):
        self.accessor = _load_accessor()
        self.accessor._get_k_and_s_triton_kernel = FakeKernel()

    def _gather(self, max_seq_len, table_cols):
        return self.accessor._get_k_and_s_triton(
            buf=FakeTensor((AGG_PAGES, BUF_NUMEL_PER_PAGE)),
            page_indices=FakeTensor((1, table_cols)),
            seq_lens=FakeTensor((1,)),
            seq_len_sum=max_seq_len,
            max_seq_len=max_seq_len,
            page_size=PAGE_SIZE,
            index_head_dim=128,
        )

    def test_sequence_past_the_page_table_is_rejected(self):
        with self.assertRaises(AssertionError) as caught:
            self._gather(max_seq_len=(1 << 20) + PAGE_SIZE, table_cols=1 << 14)
        self.assertIn("page table is shorter", str(caught.exception))

    def test_exact_fit_is_allowed(self):
        self._gather(max_seq_len=1 << 20, table_cols=1 << 14)


if __name__ == "__main__":
    unittest.main()
