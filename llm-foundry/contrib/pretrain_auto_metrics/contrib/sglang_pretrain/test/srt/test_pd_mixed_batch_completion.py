import importlib.util
import os
import sys
import types
import unittest


def _stub(name, **attrs):
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    sys.modules[name] = module
    return module



def _load_prefill():
    _stub("numpy", ndarray=object, arange=None, int64=int, int32=int)
    _stub("numpy.typing", NDArray=object)
    _stub("torch", no_grad=lambda: (lambda f: f))
    _stub("sglang.srt.disaggregation.base", KVPoll=type("KVPoll", (), {}))
    _stub(
        "sglang.srt.disaggregation.base.conn",
        StateType=type("StateType", (), {}),
    )
    _stub(
        "sglang.srt.disaggregation.common.conn",
        CommonKVManager=type("CommonKVManager", (), {}),
    )
    _stub("sglang.srt.disaggregation.pp_layer", stage_layer_index=lambda *a: (0, 0))
    _stub(
        "sglang.srt.disaggregation.utils",
        FAKE_BOOTSTRAP_HOST="fake",
        DisaggregationMode=type("DisaggregationMode", (), {}),
        KVClassType=type("KVClassType", (), {}),
        MetadataBuffers=type("MetadataBuffers", (), {}),
        ReqToMetadataIdxAllocator=type("ReqToMetadataIdxAllocator", (), {}),
        TransferBackend=type("TransferBackend", (), {}),
        get_kv_class=lambda *a, **k: None,
        is_aborted=lambda req: False,
        is_mla_backend=lambda *a, **k: False,
        poll_and_all_reduce_attn_cp_tp_group=lambda *a, **k: [],
        prepare_abort=lambda *a, **k: None,
        setup_state_kv_args=lambda *a, **k: None,
    )
    _stub(
        "sglang.srt.environ",
        envs=types.SimpleNamespace(),
    )
    _stub(
        "sglang.srt.managers.schedule_batch",
        FINISH_ABORT=type("FINISH_ABORT", (), {}),
        FINISH_LENGTH=lambda **k: "finished",
        Req=object,
        ScheduleBatch=object,
    )
    _stub(
        "sglang.srt.mem_cache.common",
        kv_to_page_indices=lambda *a, **k: [],
        kv_to_page_num=lambda *a, **k: 0,
        maybe_cache_unfinished_req=lambda *a, **k: None,
        release_kv_cache=lambda *a, **k: None,
    )
    _stub(
        "sglang.srt.mem_cache.deepseek_v4_memory_pool",
        DeepSeekV4TokenToKVPool=type("DeepSeekV4TokenToKVPool", (), {}),
    )
    _stub(
        "sglang.srt.observability.req_time_stats",
        set_schedule_time_batch=lambda *a, **k: None,
    )
    _stub(
        "sglang.srt.utils.nvtx_utils",
        scheduler_nvtx_method=lambda name: (lambda f: f),
    )

    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "python",
        "sglang",
        "srt",
        "disaggregation",
        "prefill.py",
    )
    spec = importlib.util.spec_from_file_location("prefill_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeTimeStats:
    def __getattr__(self, _name):
        return lambda *a, **k: None


class FakeReq:
    def __init__(self, rid, inflight_middle_chunks=0):
        self.rid = rid
        self.inflight_middle_chunks = inflight_middle_chunks
        self.pending_bootstrap = False
        self.return_logprob = False
        self.grammar = None
        self.output_ids = []
        self.origin_input_ids = list(range(16))
        self.fill_len = 16
        self.start_send_idx = 0
        self.tmp_end_idx = 16
        self.metadata_buffer_index = 0
        self.time_stats = FakeTimeStats()


class FakeBatch:
    def __init__(self, reqs, chunked_req=None):
        self.reqs = reqs
        self.chunked_req = chunked_req
        self.spec_info = None
        self.prefill_stats = None
        self.dp_cooperation_info = None


class FakeResult:
    def __init__(self, n):
        self.logits_output = None
        self.next_token_ids = types.SimpleNamespace(tolist=lambda: [7] * n)
        self.extend_input_len_per_req = None
        self.extend_logprob_start_len_per_req = None
        self.copy_done = None
        self.routed_experts_output = None
        self.indexer_topk_output = None
        self.can_run_cuda_graph = False


class FakeScheduler:
    def __init__(self, prefill):
        self.disagg_prefill_inflight_queue = []
        self.sent_last_chunk = []
        self.sent_middle_chunk = []
        self.enable_overlap = True
        self.tree_cache = None
        self.attn_cp_cpu_group = None
        self.attn_tp_cpu_group = None
        self.spec_algorithm = types.SimpleNamespace(is_eagle=lambda: False)
        self.batch_result_processor = types.SimpleNamespace(
            move_logprobs_to_cpu=lambda **k: None
        )
        self.metrics_reporter = types.SimpleNamespace(
            report_prefill_stats=lambda **k: None
        )
        self._process = prefill.SchedulerDisaggregationPrefillMixin.process_batch_result_disagg_prefill

    def send_kv_chunk(self, req, last_chunk=False, end_idx=None):
        (self.sent_last_chunk if last_chunk else self.sent_middle_chunk).append(req.rid)

    def run(self, batch):
        self._process(self, batch, FakeResult(len(batch.reqs)))


class TestMixedBatchCompletion(unittest.TestCase):
    def setUp(self):
        self.prefill = _load_prefill()

    def _run(self, batch):
        sched = FakeScheduler(self.prefill)
        sched.run(batch)
        return sched

    def test_short_request_next_to_a_chunk_reaches_the_inflight_queue(self):
        chunk = FakeReq("long", inflight_middle_chunks=1)
        short = FakeReq("short")
        sched = self._run(FakeBatch([chunk, short], chunked_req=chunk))

        self.assertEqual([r.rid for r in sched.disagg_prefill_inflight_queue], ["short"])
        self.assertEqual(sched.sent_last_chunk, ["short"])
        self.assertEqual(sched.sent_middle_chunk, ["long"])
        self.assertEqual(chunk.inflight_middle_chunks, 0)

    def test_a_stale_counter_no_longer_swallows_the_short_request(self):
        chunk = FakeReq("long", inflight_middle_chunks=1)
        short = FakeReq("short", inflight_middle_chunks=1)
        sched = self._run(FakeBatch([chunk, short], chunked_req=chunk))

        self.assertEqual([r.rid for r in sched.disagg_prefill_inflight_queue], ["short"])
        self.assertEqual(sched.sent_last_chunk, ["short"])

    def test_the_chunks_last_batch_completes_it(self):
        chunk = FakeReq("long", inflight_middle_chunks=0)
        short = FakeReq("short")
        sched = self._run(FakeBatch([chunk, short], chunked_req=None))

        self.assertEqual(
            sorted(r.rid for r in sched.disagg_prefill_inflight_queue),
            ["long", "short"],
        )
        self.assertEqual(sched.sent_middle_chunk, [])

    def test_pure_chunk_batch_is_still_a_middle_chunk(self):
        chunk = FakeReq("long", inflight_middle_chunks=3)
        sched = self._run(FakeBatch([chunk], chunked_req=chunk))

        self.assertEqual(sched.disagg_prefill_inflight_queue, [])
        self.assertEqual(sched.sent_last_chunk, [])
        self.assertEqual(chunk.inflight_middle_chunks, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
