import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.disaggregation.decode import DecodePreallocQueue
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

RESERVED = 256
MAX_NEW = 256
BUDGET = 8192
SHORT_LEN = 1024
LONG_LEN = 1 << 20


def make_decode_req(rid, input_len):
    req = SimpleNamespace(
        rid=rid,
        finished_reason=None,
        return_logprob=False,
        origin_input_ids=[0] * input_len,
        output_ids=[],
        sampling_params=SimpleNamespace(max_new_tokens=MAX_NEW),
        priority=0,
        req_pool_idx=0,
        last_node=None,
        cache_protected_len=0,
        time_stats=MagicMock(),
    )
    return SimpleNamespace(
        req=req,
        waiting_for_input=True,
        kv_receiver=MagicMock(),
        metadata_buffer_index=-1,
        prefix_match=None,
    )


def make_queue(decode_reqs, budget=BUDGET):
    queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
    queue.queue = list(decode_reqs)
    queue.pending_reqs = []
    queue.retracted_queue = []
    queue.num_reserved_decode_tokens = RESERVED
    queue._resolve_pending_reqs = MagicMock()
    queue._update_handshake_waiters = MagicMock()
    queue._uses_swa_tail_prealloc = MagicMock(return_value=False)
    queue._allocatable_token_budgets = MagicMock(return_value=budget)
    queue._hicache_pending_restore_tokens = MagicMock(return_value=0)
    queue._pre_alloc = MagicMock(return_value=MagicMock())
    queue.tree_cache = MagicMock()
    queue.req_to_token_pool = MagicMock()
    queue.req_to_token_pool.available_size.return_value = 64
    queue.req_to_metadata_buffer_idx_allocator = MagicMock()
    queue.req_to_metadata_buffer_idx_allocator.available_size.return_value = 64
    queue.req_to_metadata_buffer_idx_allocator.alloc.return_value = 0
    queue.token_to_kv_pool_allocator = MagicMock(page_size=64)
    queue.token_to_kv_pool = MagicMock(page_size=64)
    queue.kv_manager = MagicMock()
    queue.kv_manager.kv_args.state_types = []
    queue.transfer_queue = MagicMock(enable_staging=False)

    scheduler = MagicMock()
    scheduler.running_batch.reqs = []
    scheduler.enable_priority_scheduling = False
    scheduler.enable_hisparse = False
    scheduler.enable_decode_hicache = False
    scheduler.server_args.disaggregation_decode_enable_radix_cache = False
    queue.scheduler = scheduler
    return queue


@patch("sglang.srt.disaggregation.decode.kv_to_page_indices", lambda *a, **k: [])
class TestDecodePreallocBypass(CustomTestCase):
    def rids(self, decode_reqs):
        return [decode_req.req.rid for decode_req in decode_reqs]

    def test_off_by_default_a_long_request_blocks_the_queue(self):
        reqs = [
            make_decode_req("long", LONG_LEN),
            make_decode_req("short-1", SHORT_LEN),
            make_decode_req("short-2", SHORT_LEN),
        ]
        queue = make_queue(reqs)

        preallocated, failed = queue.pop_preallocated()

        self.assertEqual(self.rids(preallocated), [])
        self.assertEqual(failed, [])
        self.assertEqual(self.rids(queue.queue), ["long", "short-1", "short-2"])

    def test_bypass_lets_short_requests_past_the_long_one(self):
        reqs = [
            make_decode_req("long", LONG_LEN),
            make_decode_req("short-1", SHORT_LEN),
            make_decode_req("short-2", SHORT_LEN),
        ]
        queue = make_queue(reqs)

        with envs.SGLANG_DECODE_PREALLOC_BYPASS_MAX_TOKENS.override(8192):
            preallocated, _ = queue.pop_preallocated()

        self.assertEqual(self.rids(preallocated), ["short-1", "short-2"])
        self.assertEqual(self.rids(queue.queue), ["long"])

    def test_a_short_request_that_does_not_fit_still_stops_the_pass(self):
        reqs = [
            make_decode_req("short-1", SHORT_LEN),
            make_decode_req("short-2", SHORT_LEN),
        ]
        queue = make_queue(reqs, budget=8)

        with envs.SGLANG_DECODE_PREALLOC_BYPASS_MAX_TOKENS.override(8192):
            preallocated, _ = queue.pop_preallocated()

        self.assertEqual(self.rids(preallocated), [])
        self.assertEqual(self.rids(queue.queue), ["short-1", "short-2"])

    def test_bypass_does_not_reorder_requests_that_fit(self):
        reqs = [make_decode_req(f"short-{i}", SHORT_LEN) for i in range(3)]
        queue = make_queue(reqs)

        with envs.SGLANG_DECODE_PREALLOC_BYPASS_MAX_TOKENS.override(8192):
            preallocated, _ = queue.pop_preallocated()

        self.assertEqual(self.rids(preallocated), ["short-0", "short-1", "short-2"])
        self.assertEqual(queue.queue, [])


if __name__ == "__main__":
    unittest.main()
