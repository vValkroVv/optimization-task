import threading
import time
import unittest
from types import SimpleNamespace

import numpy as np

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.mooncake.conn import (
    MooncakeKVManager,
    MooncakeKVReceiver,
    MooncakeKVSender,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

ROOM = 2**63 - 5  # bootstrap rooms are random 63-bit ints


def make_kv_mgr(**kwargs):
    kv_mgr = SimpleNamespace(
        request_status={},
        room_alive_time={},
        failure_records={},
        failure_lock=threading.Lock(),
        bootstrap_timeout=300,
        waiting_timeout=300,
        max_wait=21600,
        **kwargs,
    )
    kv_mgr.record_failure = lambda room, reason: kv_mgr.failure_records.update(
        {room: reason}
    )
    kv_mgr.update_status = lambda room, status: kv_mgr.request_status.update(
        {room: status}
    )
    return kv_mgr


def make_mooncake_mgr(mode, **kwargs):
    mgr = MooncakeKVManager.__new__(MooncakeKVManager)
    mgr.disaggregation_mode = mode
    mgr.request_status = {}
    mgr.room_alive_time = {}
    mgr.keepalive_interval = 30.0
    mgr.loop_tick_time = time.time()
    mgr.room_transfer_progress = {}
    mgr.transfer_progress_lock = threading.Lock()
    mgr.live_senders = {}
    mgr._server_thread = SimpleNamespace(is_alive=lambda: True)
    for key, value in kwargs.items():
        setattr(mgr, key, value)
    return mgr


class TestTimeoutsCountSilence(unittest.TestCase):
    def _sender(self, kv_mgr):
        sender = MooncakeKVSender.__new__(MooncakeKVSender)
        sender.kv_mgr = kv_mgr
        sender.bootstrap_room = ROOM
        sender.init_time = time.time() - 1000  # waited longer than the timeout
        kv_mgr.request_status[ROOM] = KVPoll.Bootstrapping
        return sender

    def _receiver(self, kv_mgr):
        receiver = MooncakeKVReceiver.__new__(MooncakeKVReceiver)
        receiver.kv_mgr = kv_mgr
        receiver.bootstrap_room = ROOM
        receiver.init_time = time.time() - 1000
        receiver.abort_notified = True  # no prefill to notify in a unit test
        kv_mgr.request_status[ROOM] = KVPoll.WaitingForInput
        return receiver

    def test_bootstrap_waits_while_keepalive_is_fresh(self):
        kv_mgr = make_kv_mgr()
        sender = self._sender(kv_mgr)
        kv_mgr.room_alive_time[ROOM] = time.time()
        self.assertIsNone(sender._check_bootstrap_timeout())

    def test_bootstrap_fails_after_silence(self):
        kv_mgr = make_kv_mgr()
        sender = self._sender(kv_mgr)
        kv_mgr.room_alive_time[ROOM] = time.time() - 400
        self.assertEqual(sender._check_bootstrap_timeout(), KVPoll.Failed)
        self.assertIn(ROOM, kv_mgr.failure_records)

    def test_waiting_waits_while_keepalive_is_fresh(self):
        kv_mgr = make_kv_mgr()
        receiver = self._receiver(kv_mgr)
        kv_mgr.room_alive_time[ROOM] = time.time()
        self.assertIsNone(receiver._check_waiting_timeout())

    def test_bootstrap_fails_past_max_wait_despite_keepalive(self):
        kv_mgr = make_kv_mgr()
        sender = self._sender(kv_mgr)
        sender.init_time = time.time() - kv_mgr.max_wait - 1
        kv_mgr.room_alive_time[ROOM] = time.time()
        self.assertEqual(sender._check_bootstrap_timeout(), KVPoll.Failed)

    def test_waiting_fails_past_max_wait_despite_keepalive(self):
        kv_mgr = make_kv_mgr()
        receiver = self._receiver(kv_mgr)
        receiver.init_time = time.time() - kv_mgr.max_wait - 1
        kv_mgr.room_alive_time[ROOM] = time.time()
        self.assertEqual(receiver._check_waiting_timeout(), KVPoll.Failed)

    def test_waiting_fails_without_keepalive(self):
        kv_mgr = make_kv_mgr()
        receiver = self._receiver(kv_mgr)
        self.assertEqual(receiver._check_waiting_timeout(), KVPoll.Failed)


class TestKeepaliveTargets(unittest.TestCase):
    def test_prefill_sends_only_rooms_it_holds(self):
        info = SimpleNamespace(endpoint="10.0.0.1", dst_port=17000)
        mgr = make_mooncake_mgr(
            DisaggregationMode.PREFILL,
            transfer_infos={1: {"a": info}, 2: {"a": info}, 3: {"a": info}},
            live_senders={1: None, 2: None, 3: None},
        )
        mgr.request_status = {
            1: KVPoll.WaitingForInput,
            2: KVPoll.Transferring,
            3: KVPoll.Failed,
        }
        self.assertEqual(
            dict(mgr._collect_keepalive_targets()), {("10.0.0.1", 17000): [1, 2]}
        )

    def test_prefill_skips_rooms_without_sender(self):
        info = SimpleNamespace(endpoint="10.0.0.1", dst_port=17000)
        mgr = make_mooncake_mgr(
            DisaggregationMode.PREFILL,
            transfer_infos={1: {"a": info}, 2: {"a": info}},
            live_senders={1: None},
        )
        mgr.request_status = {1: KVPoll.WaitingForInput, 2: KVPoll.WaitingForInput}
        self.assertEqual(
            dict(mgr._collect_keepalive_targets()), {("10.0.0.1", 17000): [1]}
        )

    def test_decode_skips_concluded_receivers(self):
        infos = [{"rank_ip": "10.0.0.2", "rank_port": 18000}]
        mgr = make_mooncake_mgr(
            DisaggregationMode.DECODE,
            live_receivers={
                1: SimpleNamespace(conclude_state=None, bootstrap_infos=infos),
                2: SimpleNamespace(conclude_state=KVPoll.Success, bootstrap_infos=infos),
                3: SimpleNamespace(conclude_state=None),  # handshake not done yet
            },
        )
        self.assertEqual(
            dict(mgr._collect_keepalive_targets()), {("10.0.0.2", 18000): [1]}
        )

    def test_stuck_scheduler_loop_sends_nothing(self):
        info = SimpleNamespace(endpoint="10.0.0.1", dst_port=17000)
        mgr = make_mooncake_mgr(
            DisaggregationMode.PREFILL,
            transfer_infos={1: {"a": info}},
            live_senders={1: None},
        )
        mgr.request_status = {1: KVPoll.WaitingForInput}
        mgr.loop_tick_time = time.time() - 3 * mgr.keepalive_interval
        self.assertEqual(dict(mgr._collect_keepalive_targets()), {})

    def test_dead_server_thread_sends_nothing(self):
        info = SimpleNamespace(endpoint="10.0.0.1", dst_port=17000)
        mgr = make_mooncake_mgr(
            DisaggregationMode.PREFILL,
            transfer_infos={1: {"a": info}},
            live_senders={1: None},
        )
        mgr.request_status = {1: KVPoll.WaitingForInput}
        mgr._server_thread = SimpleNamespace(is_alive=lambda: False)
        self.assertEqual(dict(mgr._collect_keepalive_targets()), {})

    def test_stuck_transfer_chunk_stops_keepalive_for_its_room(self):
        info = SimpleNamespace(endpoint="10.0.0.1", dst_port=17000)
        mgr = make_mooncake_mgr(
            DisaggregationMode.PREFILL,
            transfer_infos={1: {"a": info}, 2: {"a": info}, 3: {"a": info}},
            live_senders={1: None, 2: None, 3: None},
        )
        mgr.request_status = {r: KVPoll.WaitingForInput for r in (1, 2, 3)}
        stale = time.time() - 3 * mgr.keepalive_interval
        mgr.room_transfer_progress = {1: [1, stale], 2: [1, time.time()]}
        # 1: chunk queued long ago and not sent; 2: transfer moving; 3: waits in prefill queue
        self.assertEqual(
            dict(mgr._collect_keepalive_targets()), {("10.0.0.1", 17000): [2, 3]}
        )
        mgr._mark_chunk_done(1)
        self.assertNotIn(1, mgr.room_transfer_progress)

    def test_mark_rooms_alive_round_trips_63_bit_rooms(self):
        mgr = make_mooncake_mgr(DisaggregationMode.DECODE)
        mgr.request_status = {ROOM: KVPoll.WaitingForInput}
        mgr._mark_rooms_alive(np.array([ROOM, 7], dtype=np.uint64).tobytes())
        self.assertEqual(set(mgr.room_alive_time), {ROOM})  # unknown room 7 ignored


class TestSenderClear(unittest.TestCase):
    def test_clear_drops_live_sender(self):
        kv_mgr = make_kv_mgr(
            live_senders={},
            room_transfer_progress={},
            transfer_progress_lock=threading.Lock(),
        )
        sender = MooncakeKVSender.__new__(MooncakeKVSender)
        sender.kv_mgr = kv_mgr
        sender.bootstrap_room = ROOM
        kv_mgr.live_senders[ROOM] = sender
        sender.clear()
        self.assertNotIn(ROOM, kv_mgr.live_senders)


class TestReceiverClear(unittest.TestCase):
    def test_clear_drops_live_receiver(self):
        kv_mgr = make_kv_mgr(
            live_receivers={},
            required_prefill_response_num_table={},
            prefill_response_tracker={},
        )
        receiver = MooncakeKVReceiver.__new__(MooncakeKVReceiver)
        receiver.kv_mgr = kv_mgr
        receiver.bootstrap_room = ROOM
        kv_mgr.live_receivers[ROOM] = receiver
        kv_mgr.room_alive_time[ROOM] = time.time()
        receiver.clear()
        self.assertNotIn(ROOM, kv_mgr.live_receivers)
        self.assertNotIn(ROOM, kv_mgr.room_alive_time)


if __name__ == "__main__":
    unittest.main()
