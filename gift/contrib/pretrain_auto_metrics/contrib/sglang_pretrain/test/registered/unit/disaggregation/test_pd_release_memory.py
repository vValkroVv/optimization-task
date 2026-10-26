import unittest
from types import SimpleNamespace

from sglang.srt.disaggregation.decode import DecodePreallocQueue, DecodeTransferQueue
from sglang.srt.disaggregation.prefill import PrefillBootstrapQueue
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class FakeConn:
    def __init__(self):
        self.calls = []

    def abort(self):
        self.calls.append("abort")

    def clear(self):
        self.calls.append("clear")


class TestReleaseMemoryEndsRequests(unittest.TestCase):
    """Dropped requests must be aborted, or keepalive keeps their rooms alive on the peer."""

    def test_decode_prealloc_aborts_queued_and_pending(self):
        queued, pending = FakeConn(), FakeConn()
        q = DecodePreallocQueue.__new__(DecodePreallocQueue)
        q.queue = [SimpleNamespace(kv_receiver=queued)]
        q.pending_reqs = [SimpleNamespace(kv_receiver=pending)]
        q.retracted_queue = []
        q.kv_manager = SimpleNamespace()

        q.release_memory_occupation()

        self.assertEqual(queued.calls, ["abort", "clear"])
        self.assertEqual(pending.calls, ["abort", "clear"])
        self.assertEqual((q.queue, q.pending_reqs), ([], []))

    def test_decode_transfer_aborts_in_flight(self):
        receiver = FakeConn()
        q = DecodeTransferQueue.__new__(DecodeTransferQueue)
        q.queue = [SimpleNamespace(kv_receiver=receiver)]

        q.release_memory_occupation()

        self.assertEqual(receiver.calls, ["abort", "clear"])
        self.assertEqual(q.queue, [])

    def test_prefill_bootstrap_aborts_senders(self):
        sender = FakeConn()
        q = PrefillBootstrapQueue.__new__(PrefillBootstrapQueue)
        q.queue = [SimpleNamespace(disagg_kv_sender=sender)]
        q.kv_manager = SimpleNamespace()

        q.release_memory_occupation()

        self.assertEqual(sender.calls, ["abort", "clear"])
        self.assertEqual(q.queue, [])


if __name__ == "__main__":
    unittest.main()
