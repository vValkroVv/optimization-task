"""Queue timeout decisions must agree across every TP/DP rank."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch.distributed as dist

from sglang.test.test_utils import maybe_stub_sgl_kernel, run_distributed_test

maybe_stub_sgl_kernel()

from sglang.srt.managers.prefill_delayer import PrefillDelayer, _State


def _check_queue_timeout(rank, dp_size, attn_tp_size):
    world_size = dist.get_world_size()
    # Exercise each rank as the only expired rank, including non-TP0 ranks.
    cases = [
        (
            tuple(0.5001 if r == late else 0.4999 for r in range(world_size)),
            0.1,
            512,
            True,
        )
        for late in range(world_size)
    ]
    cases += [
        ((0.4999,) * world_size, 0.1, 512, False),
        ((0.5001,) * world_size, 0.1, 512, True),
        ((0.5001,) * world_size, None, 512, True),
        (None, 0.1, 512, False),  # No previous wait: no deadline to expire.
    ]
    if dp_size == 1:
        cases.append(((0.5001,) * world_size, 0.1, 64, False))
    observed = []
    expected = []
    with patch.dict(
        "os.environ", {"SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH": "0"}
    ), patch(
        "sglang.srt.managers.prefill_delayer.time.perf_counter", return_value=100.0
    ):
        for ages, ratio, max_running, allow in cases:
            delayer = PrefillDelayer(
                dp_size=dp_size,
                attn_tp_size=attn_tp_size,
                cpu_group=dist.group.WORLD,
                server_args=SimpleNamespace(
                    enable_dp_attention=dp_size > 1,
                    disable_overlap_schedule=False,
                    prefill_delayer_queue_min_ratio=ratio,
                    prefill_delayer_max_delay_ms=500,
                ),
                max_delay_passes=10,
                token_usage_low_watermark=None,
            )
            delayer.skip_first_delayer = False
            delayer._curr_state = (
                _State(delayed_count=2, start_time=100.0 - ages[rank])
                if ages is not None
                else None
            )
            out = delayer._negotiate_should_allow_prefill(
                local_prefillable=True,
                token_usage=0.2,
                running_batch=64,
                max_prefill_bs=16,
                max_running_requests=max_running,
                waiting_queue_len=1,
            )
            observed.append((out.output_allow, out.output_reason))
            expected.append((allow, "wait_success" if allow else "delay"))
    # Finish every collective before asserting so failures cannot strand peers.
    assert observed == expected, f"rank {rank}: {observed} != {expected}"


class TestPrefillDelayerTPTimeout(unittest.TestCase):
    def test_tp_timeout(self):
        run_distributed_test(
            _check_queue_timeout,
            world_size=2,
            backend="gloo",
            dp_size=1,
            attn_tp_size=2,
        )

    def test_dp_tp_timeout(self):
        run_distributed_test(
            _check_queue_timeout,
            world_size=4,
            backend="gloo",
            dp_size=2,
            attn_tp_size=2,
        )
