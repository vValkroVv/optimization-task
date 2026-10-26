import importlib.util
import os
import sys
import types
import unittest

PAGE = 64
CHUNK = 4096
MAX_PREFILL = 16384
SHORT = 1792
HUGE = 1 << 20


def _stub(name, **attrs):
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    sys.modules[name] = module


class _Env:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value


def _load_schedule_policy(reserve_tokens, logprob_rows=0):
    _stub("torch")
    _stub("sglang.srt.dllm.config", DllmConfig=type("DllmConfig", (), {}))
    _stub(
        "sglang.srt.managers.prefill_delayer",
        PrefillDelayerSinglePassExecutor=type(
            "PrefillDelayerSinglePassExecutor", (), {}
        ),
    )
    _stub("sglang.srt.managers.schedule_batch", Req=object, ScheduleBatch=object)
    _stub(
        "sglang.srt.mem_cache.base_prefix_cache",
        **{
            n: type(n, (), {})
            for n in (
                "BasePrefixCache",
                "InitLoadBackParams",
                "InsertParams",
                "MatchPrefixParams",
            )
        },
        zero_match_result=lambda *a, **k: None,
    )
    _stub(
        "sglang.srt.mem_cache.radix_cache",
        **{n: type(n, (), {}) for n in ("RadixCache", "RadixKey", "TreeNode")},
    )
    _stub(
        "sglang.srt.server_args",
        ServerArgs=type("ServerArgs", (), {}),
        get_global_server_args=lambda: None,
    )
    _stub("sglang.srt.utils", get_bool_env_var=lambda *a, **k: False)
    _stub(
        "sglang.srt.environ",
        envs=types.SimpleNamespace(
            SGLANG_PREFILL_RESERVE_FOR_WAITING_TOKENS=_Env(reserve_tokens),
            SGLANG_PREFILL_LOGPROB_ROWS_NEXT_TO_CHUNK=_Env(logprob_rows),
            SGLANG_RADIX_FORCE_MISS=_Env(False),
        ),
    )
    _stub(
        "sglang.srt.layers.attention.dsa.utils",
        is_dsa_prefill_cp_in_seq_split=lambda: False,
    )
    _stub(
        "sglang.srt.layers.utils.cp_utils",
        is_prefill_context_parallel_enabled=lambda: False,
    )
    _stub(
        "sglang.srt.mem_cache.allocator.swa",
        SWATokenToKVPoolAllocator=type("SWA", (), {}),
    )
    _stub(
        "sglang.srt.mem_cache.allocator.hisparse",
        DeepSeekV4HiSparseTokenToKVPoolAllocator=type("HiSparse", (), {}),
    )

    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "python",
        "sglang",
        "srt",
        "managers",
        "schedule_policy.py",
    )
    spec = importlib.util.spec_from_file_location("schedule_policy_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeTreeCache:
    disable = False

    def __init__(self):
        self.evictable = 0

    def supports_mamba(self):
        return False

    def is_tree_cache(self):
        return False

    def evictable_size(self):
        return self.evictable

    def full_evictable_size(self):
        return self.evictable

    def swa_evictable_size(self):
        return 0

    def inc_lock_ref(self, node):
        return None

    def dec_lock_ref(self, node):
        return None


class FakeAllocator:
    def __init__(self, size):
        self.size = size

    def available_size(self):
        return self.size

    def full_available_size(self):
        return self.size

    def swa_available_size(self):
        return 0


class FakeBatch:
    reqs = ()

    def batch_size(self):
        return 0


class FakeReq:
    def __init__(self, total_len, prefix_len=0, max_new_tokens=256, req_pool_idx=None):
        self.req_pool_idx = req_pool_idx
        self.origin_input_ids = [0] * total_len
        self.prefix_indices = [0] * prefix_len
        self.extend_input_len = total_len - prefix_len
        self.fill_len = 0
        self.full_untruncated_fill_ids = [0] * total_len
        self.host_hit_length = 0
        self.swa_host_hit_length = 0
        self.output_ids = []
        self.retracted_stain = False
        self.last_node = None
        self.best_match_node = None
        self.mamba_pool_idx = None
        self.sampling_params = types.SimpleNamespace(
            max_new_tokens=max_new_tokens, ignore_eos=False
        )

    def set_extend_input_len(self, value):
        self.extend_input_len = value

    def needs_host_load_back(self):
        return False


def make_adder(policy, *, max_prefill=MAX_PREFILL, kv=8 << 20, waiting=5):
    return policy.PrefillAdder(
        page_size=PAGE,
        tree_cache=FakeTreeCache(),
        token_to_kv_pool_allocator=FakeAllocator(kv),
        running_batch=FakeBatch(),
        new_token_ratio=1.0,
        rem_input_tokens=max_prefill,
        rem_chunk_tokens=CHUNK,
        waiting_queue_len=waiting,
    )


class TestPrefillWaitingAllowance(unittest.TestCase):
    def setUp(self):
        self.off = _load_schedule_policy(0)
        self.on = _load_schedule_policy(4096)

    def add_chunk(self, policy, adder):
        long_req = FakeReq(HUGE, prefix_len=CHUNK * 10)
        still_chunked = adder.add_chunked_req(long_req)
        self.assertIsNotNone(still_chunked, "the 1M request must stay chunked")
        return long_req

    def test_default_off_short_request_cannot_join_the_chunk(self):
        adder = make_adder(self.off)
        long_req = self.add_chunk(self.off, adder)
        self.assertEqual(long_req.extend_input_len, CHUNK)
        self.assertEqual(adder.rem_chunk_tokens, 0)

        res = adder.add_one_req(
            FakeReq(SHORT), has_chunked_req=True, truncation_align_size=None
        )
        self.assertEqual(res, self.off.AddReqResult.OTHER)
        self.assertEqual(len(adder.can_run_list), 1)

    def test_allowance_admits_short_requests_without_shrinking_the_chunk(self):
        adder = make_adder(self.on)
        long_req = self.add_chunk(self.on, adder)
        self.assertEqual(long_req.extend_input_len, CHUNK, "the chunk must not shrink")
        self.assertEqual(adder.rem_chunk_tokens, 4096)

        for _ in range(2):
            res = adder.add_one_req(
                FakeReq(SHORT), has_chunked_req=True, truncation_align_size=None
            )
            self.assertEqual(res, self.on.AddReqResult.CONTINUE)
        self.assertEqual(len(adder.can_run_list), 3)
        self.assertIsNone(adder.new_chunked_req)
        for req in adder.can_run_list[1:]:
            self.assertEqual(req.extend_input_len, SHORT)
        self.assertEqual(adder.rem_chunk_tokens, 4096 - 2 * SHORT)

    def test_a_request_too_big_for_the_leftover_is_not_chunked(self):
        adder = make_adder(self.on)
        self.add_chunk(self.on, adder)
        adder.rem_chunk_tokens = 1024

        res = adder.add_one_req(
            FakeReq(SHORT), has_chunked_req=True, truncation_align_size=None
        )
        self.assertEqual(res, self.on.AddReqResult.OTHER)
        self.assertIsNone(
            adder.new_chunked_req, "would trip the one-chunked-req assert"
        )
        self.assertEqual(len(adder.can_run_list), 1)

    def test_allowance_capped_by_max_prefill_tokens(self):
        adder = make_adder(self.on, max_prefill=CHUNK + 1024)
        self.add_chunk(self.on, adder)
        self.assertEqual(adder.rem_chunk_tokens, 1024)

    def test_no_allowance_without_waiting_requests(self):
        adder = make_adder(self.on, waiting=0)
        self.add_chunk(self.on, adder)
        self.assertEqual(adder.rem_chunk_tokens, 0)

    def test_final_chunk_is_not_padded_with_an_allowance(self):
        adder = make_adder(self.on)
        tail = FakeReq(CHUNK * 10 + 1024, prefix_len=CHUNK * 10)
        self.assertIsNone(
            adder.add_chunked_req(tail), "last chunk finishes the request"
        )
        self.assertEqual(adder.rem_chunk_tokens, CHUNK - 1024)


class TestOversizedWaitingRequests(unittest.TestCase):
    def walk_waiting_queue(self, policy, waiting, *, skip_oversized):
        adder = make_adder(policy)
        chunked = adder.add_chunked_req(FakeReq(HUGE, prefix_len=CHUNK * 10))
        self.assertIsNotNone(chunked)
        for req in waiting:
            if skip_oversized and adder.is_oversized_for_chunk_budget(req):
                continue
            if (
                adder.add_one_req(req, has_chunked_req=True, truncation_align_size=None)
                != policy.AddReqResult.CONTINUE
            ):
                break
        return adder.can_run_list[1:]

    def test_a_queued_long_request_blocks_the_short_ones_behind_it(self):
        policy = _load_schedule_policy(4096)
        waiting = [FakeReq(HUGE), FakeReq(SHORT), FakeReq(SHORT)]

        admitted = self.walk_waiting_queue(policy, waiting, skip_oversized=False)

        self.assertEqual(admitted, [])

    def test_skipping_it_lets_the_short_ones_through(self):
        policy = _load_schedule_policy(4096)
        waiting = [FakeReq(HUGE), FakeReq(SHORT), FakeReq(SHORT)]

        admitted = self.walk_waiting_queue(policy, waiting, skip_oversized=True)

        self.assertEqual(admitted, waiting[1:])

    def test_oversized_is_measured_against_the_remaining_budget(self):
        policy = _load_schedule_policy(4096)
        adder = make_adder(policy)
        adder.add_chunked_req(FakeReq(HUGE, prefix_len=CHUNK * 10))
        self.assertEqual(adder.rem_chunk_tokens, 4096)

        self.assertFalse(adder.is_oversized_for_chunk_budget(FakeReq(SHORT)))
        self.assertTrue(adder.is_oversized_for_chunk_budget(FakeReq(HUGE)))
        self.assertFalse(
            adder.is_oversized_for_chunk_budget(FakeReq(HUGE, prefix_len=HUGE - SHORT))
        )

        adder.rem_chunk_tokens = 0
        self.assertTrue(adder.is_oversized_for_chunk_budget(FakeReq(SHORT)))

    def test_no_chunk_budget_at_all_skips_nothing(self):
        policy = _load_schedule_policy(4096)
        adder = make_adder(policy)
        adder.rem_chunk_tokens = None
        self.assertFalse(adder.is_oversized_for_chunk_budget(FakeReq(HUGE)))


class TestMicroBatchCapNextToAChunk(unittest.TestCase):
    MICRO_BATCH_CAP = 8
    RUNNING_BS = 7

    def walk_waiting_queue(self, policy, waiting, *, chunk_allowance, pool_free=24):
        adder = make_adder(policy)
        self.assertIsNotNone(
            adder.add_chunked_req(FakeReq(HUGE, prefix_len=CHUNK * 10, req_pool_idx=3))
        )
        for req in waiting:
            if chunk_allowance:
                num_admitted = policy.num_reqs_needing_slot(adder.can_run_list)
                num_allocatable = pool_free
            else:
                num_admitted = len(adder.can_run_list)
                num_allocatable = min(self.MICRO_BATCH_CAP - self.RUNNING_BS, pool_free)
            if num_admitted >= num_allocatable:
                break
            if chunk_allowance and adder.is_oversized_for_chunk_budget(req):
                continue
            if (
                adder.add_one_req(req, has_chunked_req=True, truncation_align_size=None)
                != policy.AddReqResult.CONTINUE
            ):
                break
        return adder.can_run_list[1:]

    def test_the_micro_batch_cap_blocks_them_even_with_the_allowance(self):
        policy = _load_schedule_policy(4096)
        waiting = [FakeReq(SHORT), FakeReq(SHORT)]

        admitted = self.walk_waiting_queue(policy, waiting, chunk_allowance=False)

        self.assertEqual(admitted, [])

    def test_bounding_by_free_request_slots_lets_them_through(self):
        policy = _load_schedule_policy(4096)
        waiting = [FakeReq(HUGE), FakeReq(SHORT), FakeReq(SHORT)]

        admitted = self.walk_waiting_queue(policy, waiting, chunk_allowance=True)

        self.assertEqual(admitted, waiting[1:])

    def test_an_empty_request_pool_still_stops_the_walk(self):
        policy = _load_schedule_policy(4096)
        waiting = [FakeReq(SHORT), FakeReq(SHORT)]

        admitted = self.walk_waiting_queue(
            policy, waiting, chunk_allowance=True, pool_free=0
        )

        self.assertEqual(admitted, [])

    def test_the_last_free_slot_goes_to_a_short_request(self):
        policy = _load_schedule_policy(4096)
        waiting = [FakeReq(SHORT), FakeReq(SHORT)]

        admitted = self.walk_waiting_queue(
            policy, waiting, chunk_allowance=True, pool_free=1
        )

        self.assertEqual(admitted, waiting[:1])


def prefill_buffer_ceiling(chunk, reserve, max_prefill):
    if reserve <= 0:
        return chunk
    return max(chunk, min(chunk + reserve, max_prefill))


class TestPrefillBufferCeiling(unittest.TestCase):
    def saturate(self, policy, *, max_prefill, req_len):
        adder = make_adder(policy, max_prefill=max_prefill)
        adder.add_chunked_req(FakeReq(HUGE, prefix_len=CHUNK * 10))
        for _ in range(1000):
            res = adder.add_one_req(
                FakeReq(req_len), has_chunked_req=True, truncation_align_size=None
            )
            if res != policy.AddReqResult.CONTINUE:
                break
        else:
            self.fail("add_one_req never stopped")
        return sum(req.extend_input_len for req in adder.can_run_list)

    def test_saturated_batch_fits_the_buffer(self):
        policy = _load_schedule_policy(4096)
        ceiling = prefill_buffer_ceiling(CHUNK, 4096, MAX_PREFILL)
        self.assertEqual(ceiling, CHUNK + 4096)
        for req_len in (PAGE, 512, SHORT, 4096):
            with self.subTest(req_len=req_len):
                total = self.saturate(policy, max_prefill=MAX_PREFILL, req_len=req_len)
                self.assertGreater(total, CHUNK, "no short request got in")
                self.assertLessEqual(total, ceiling)

    def test_saturated_batch_fits_the_buffer_when_max_prefill_clamps(self):
        policy = _load_schedule_policy(8192)
        max_prefill = CHUNK + 1024
        ceiling = prefill_buffer_ceiling(CHUNK, 8192, max_prefill)
        self.assertEqual(ceiling, max_prefill)
        for req_len in (512, 4096):
            with self.subTest(req_len=req_len):
                total = self.saturate(policy, max_prefill=max_prefill, req_len=req_len)
                self.assertLessEqual(total, ceiling)

    def test_ceiling_is_unchanged_with_the_feature_off(self):
        self.assertEqual(prefill_buffer_ceiling(CHUNK, 0, MAX_PREFILL), CHUNK)


if __name__ == "__main__":
    unittest.main(verbosity=2)
