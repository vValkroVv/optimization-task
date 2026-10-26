import importlib.util
import os
import sys
import types
import unittest

MB = 1 << 20


def _stub(name, **attrs):
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    sys.modules[name] = module


class _Env:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value


def _load_mixin(pd_prefill_req_pool_size):
    _stub("torch")
    _stub(
        "sglang.srt.configs.model_config",
        get_dsa_index_head_dim=lambda *a, **k: 0,
        is_deepseek_dsa=lambda *a, **k: False,
        is_deepseek_v4=lambda *a, **k: False,
    )
    _stub("sglang.srt.distributed.parallel_state", get_world_group=lambda: None)
    _stub(
        "sglang.srt.environ",
        envs=types.SimpleNamespace(
            SGLANG_PD_PREFILL_REQ_POOL_SIZE=_Env(pd_prefill_req_pool_size),
        ),
    )
    _stub("sglang.srt.layers.dp_attention", get_attention_tp_size=lambda: 1)
    _stub(
        "sglang.srt.mem_cache.allocator",
        **{
            n: type(n, (), {})
            for n in ("PagedTokenToKVPoolAllocator", "TokenToKVPoolAllocator")
        },
    )
    _stub(
        "sglang.srt.mem_cache.allocator.hisparse",
        **{
            n: type(n, (), {})
            for n in (
                "DeepSeekV4HiSparseTokenToKVPoolAllocator",
                "HiSparseTokenToKVPoolAllocator",
            )
        },
    )
    _stub(
        "sglang.srt.mem_cache.allocator.swa",
        SWATokenToKVPoolAllocator=type("SWATokenToKVPoolAllocator", (), {}),
    )
    _stub(
        "sglang.srt.mem_cache.common",
        get_req_to_token_extra_context_len=lambda *a, **k: 0,
    )
    _stub(
        "sglang.srt.mem_cache.deepseek_v4_memory_pool",
        DeepSeekV4TokenToKVPool=type("DeepSeekV4TokenToKVPool", (), {}),
    )
    _stub(
        "sglang.srt.mem_cache.hisparse_memory_pool",
        HiSparseDSATokenToKVPool=type("HiSparseDSATokenToKVPool", (), {}),
    )
    _stub(
        "sglang.srt.mem_cache.memory_pool",
        **{
            n: type(n, (), {})
            for n in (
                "DSATokenToKVPool",
                "HybridLinearKVPool",
                "HybridReqToTokenPool",
                "MHATokenToKVPool",
                "MHATokenToKVPoolFP4",
                "MLATokenToKVPool",
                "MLATokenToKVPoolFP4",
                "NoOpMHATokenToKVPool",
                "ReqToTokenPool",
            )
        },
    )
    _stub("sglang.srt.mem_cache.swa_memory_pool", SWAKVPool=type("SWAKVPool", (), {}))
    _stub("sglang.srt.platforms", current_platform=types.SimpleNamespace())
    _stub(
        "sglang.srt.utils.common",
        get_available_gpu_memory=lambda *a, **k: 0,
        is_float4_e2m1fn_x2=lambda *a, **k: False,
        is_hip=lambda: False,
        is_npu=lambda: False,
    )

    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "python",
        "sglang",
        "srt",
        "model_executor",
        "model_runner_kv_cache_mixin.py",
    )
    spec = importlib.util.spec_from_file_location("kv_cache_mixin_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_server_args(**overrides):
    args = dict(
        max_running_requests=64,
        max_mamba_cache_size=50,
        disaggregation_mode="prefill",
        disable_radix_cache=False,
        mamba_radix_cache_strategy="extra_buffer",
        disable_overlap_schedule=False,
    )
    args.update(overrides)
    ns = types.SimpleNamespace(**args)
    ns.enable_mamba_extra_buffer = lambda: ns.mamba_radix_cache_strategy in (
        "extra_buffer",
        "extra_buffer_lazy",
    )
    ns.enable_mamba_extra_buffer_lazy = (
        lambda: ns.mamba_radix_cache_strategy == "extra_buffer_lazy"
    )
    return ns


def make_runner(module, server_args, *, dp_size=2, context_len=MB, mambaish=True):
    runner = module.ModelRunnerKVCacheMixin()
    runner.server_args = server_args
    runner.dp_size = dp_size
    runner.model_config = types.SimpleNamespace(context_len=context_len)
    runner.mambaish_config = object() if mambaish else None
    return runner


BF16_TOKENS = 11638912
FP8_TOKENS = 2 * BF16_TOKENS


class TestDefaultRequestSlots(unittest.TestCase):
    def setUp(self):
        self.module = _load_mixin(0)

    def resolve(self, token_capacity, **overrides):
        runner = make_runner(self.module, make_server_args(**overrides))
        return runner._resolve_max_num_reqs(token_capacity)

    def test_the_mamba_cap_is_what_binds(self):
        self.assertEqual(self.resolve(BF16_TOKENS), 10)

    def test_the_lazy_extra_buffer_gives_one_more_slot_per_four(self):
        self.assertEqual(
            self.resolve(BF16_TOKENS, mamba_radix_cache_strategy="extra_buffer_lazy"),
            12,
        )

    def test_the_kv_dtype_does_not_change_the_pool(self):
        self.assertEqual(self.resolve(FP8_TOKENS), self.resolve(BF16_TOKENS))

    def test_without_max_running_requests_the_estimate_binds(self):
        runner = make_runner(
            self.module,
            make_server_args(max_running_requests=None),
            mambaish=False,
        )
        self.assertEqual(runner._resolve_max_num_reqs(BF16_TOKENS), 4096)

    def test_a_mamba_pool_too_small_for_one_request_is_fatal(self):
        with self.assertRaises(RuntimeError):
            self.resolve(BF16_TOKENS, max_mamba_cache_size=4)


class TestPDPrefillOverride(unittest.TestCase):
    def resolve(self, pool_size, **overrides):
        module = _load_mixin(pool_size)
        runner = make_runner(module, make_server_args(**overrides))
        return runner._resolve_max_num_reqs(BF16_TOKENS)

    def test_off_by_default(self):
        self.assertEqual(self.resolve(0), 10)

    def test_it_raises_the_slots_of_a_prefill_node(self):
        self.assertEqual(self.resolve(48), 48)

    def test_it_never_lowers_them(self):
        self.assertEqual(self.resolve(4), 10)

    def test_it_is_ignored_on_a_decode_node(self):
        self.assertEqual(self.resolve(48, disaggregation_mode="decode"), 10)

    def test_it_is_ignored_without_disaggregation(self):
        self.assertEqual(self.resolve(48, disaggregation_mode="null"), 10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
