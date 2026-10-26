# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from itertools import combinations

import pytest
import torch
from torch.distributed.pipelining._schedule_visualizer import get_schedule_ops
from torch.distributed.pipelining.schedules import _Action, _ComputationType

from torchtitan.distributed.pipeline_parallel_recv_pool import (
    _assign_slots,
    _busy_intervals,
    _BWD_CONSUMERS,
    _flatten_actions,
    _FWD_CONSUMERS,
    _lease,
    _reuse_chain,
    _release,
)

RECV_F = _ComputationType.RECV_F
RECV_B = _ComputationType.RECV_B
SEND_F = _ComputationType.SEND_F
SEND_B = _ComputationType.SEND_B
FORWARD = _ComputationType.FORWARD
FULL_BACKWARD = _ComputationType.FULL_BACKWARD
BACKWARD_INPUT = _ComputationType.BACKWARD_INPUT
BACKWARD_WEIGHT = _ComputationType.BACKWARD_WEIGHT


@pytest.mark.parametrize(
    "intervals, expected",
    [
        ({0: (0, 1), 1: (2, 3), 2: (4, 5)}, 1),
        ({mb: (0, 100) for mb in range(4)}, 4),
        ({0: (0, 3), 1: (2, 5), 2: (4, 7), 3: (6, 9)}, 2),
        ({0: (0, 1), 1: (2, 9), 2: (3, 4), 3: (5, 6)}, 2),
        ({2: (0, 2), 0: (2, 3), 1: (4, 5)}, 2),
        ({2: (0, 1), 1: (2, 3), 0: (4, 5)}, 1),
    ],
)
def test_minimum_slots(intervals, expected):
    slots = _assign_slots(intervals)
    assert set(slots) == set(intervals)
    assert len(set(slots.values())) == expected
    for a, b in combinations(intervals, 2):
        if slots[a] == slots[b]:
            assert (
                intervals[a][1] < intervals[b][0] or intervals[b][1] < intervals[a][0]
            )


@pytest.mark.parametrize(
    "recv, consumers", [(RECV_F, _FWD_CONSUMERS), (RECV_B, _BWD_CONSUMERS)]
)
def test_split_backward_keeps_buffer_until_weight_backward(recv, consumers):
    actions = [
        _Action(1, recv, 0),
        _Action(1, BACKWARD_INPUT, 0),
        _Action(1, recv, 1),
        _Action(1, BACKWARD_INPUT, 1),
        _Action(1, BACKWARD_WEIGHT, 0),
        _Action(1, BACKWARD_WEIGHT, 1),
        _Action(2, recv, 0),
        _Action(2, BACKWARD_WEIGHT, 0),
    ]
    assert _busy_intervals(actions, 1, recv, consumers, {1}) == {0: (0, 4), 1: (2, 5)}


@pytest.mark.parametrize(
    "recv, consumers, source, local_end, remote_end",
    [
        (RECV_F, _FWD_CONSUMERS, 1, 3, 5),
        (RECV_B, _BWD_CONSUMERS, 3, 1, 0),
    ],
)
def test_local_alias_lifetime_stops_at_remote_handoff(
    recv, consumers, source, local_end, remote_end
):
    actions = [
        _Action(source, recv, 0),
        _Action(source, BACKWARD_WEIGHT, 0),
        _Action(source, recv, 1),
        _Action(local_end, BACKWARD_WEIGHT, 0),
        _Action(source, BACKWARD_WEIGHT, 1),
        _Action(local_end, BACKWARD_WEIGHT, 1),
        _Action(remote_end, BACKWARD_WEIGHT, 0),
    ]
    intervals = _busy_intervals(actions, source, recv, consumers, {1, 2, 3})
    assert intervals == {0: (0, 3), 1: (2, 5)}
    assert len(set(_assign_slots(intervals).values())) == 2


def test_reuse_chain_is_circular_per_slot():
    intervals = {0: (0, 3), 1: (2, 5), 2: (4, 7), 3: (6, 9)}
    slots = _assign_slots(intervals)
    assert slots == {0: 0, 1: 1, 2: 0, 3: 1}
    assert _reuse_chain(intervals, slots) == {0: 2, 2: 0, 1: 3, 3: 1}


def test_late_reader_fails_after_slot_reuse():
    slot = (torch.zeros(4, 4, requires_grad=True),)
    chain = _reuse_chain({0: (0, 1), 1: (2, 3), 2: (4, 5)}, {0: 0, 1: 0, 2: 0})
    lease: dict = {}
    weight = torch.eye(4)
    (mb0_input,) = _lease(lease, 0, slot)
    assert mb0_input is not slot[0] and mb0_input.is_leaf and mb0_input.requires_grad
    assert mb0_input.data_ptr() == slot[0].data_ptr()
    with torch.no_grad():
        slot[0].fill_(1)
    assert (mb0_input @ weight).sum() == 16
    # The slot moves on to mb1: a reader that stashed mb0's input must now
    # fail loudly instead of silently reading mb1's data.
    _release(lease, chain, 1)
    assert 0 not in lease and mb0_input.numel() == 0
    with pytest.raises(RuntimeError):
        mb0_input @ weight
    # Next step: mb0 gets a new leaf, free of the old one's autograd state.
    (again,) = _lease(lease, 0, slot)
    assert again is not mb0_input and again.numel() == 16


@pytest.mark.parametrize(
    "recv, consumers, compute, send",
    [
        (RECV_F, _FWD_CONSUMERS, FORWARD, SEND_F),
        (RECV_B, _BWD_CONSUMERS, FULL_BACKWARD, SEND_B),
    ],
)
def test_send_reads_the_recv_buffer(recv, consumers, compute, send):
    # A passthrough stage sends (an alias of) its recv buffer onward, so the
    # buffer stays busy until the SEND is posted — without SEND as a reader,
    # mb0's interval would end at the compute and the slots would collapse.
    actions = [
        _Action(1, recv, 0),
        _Action(1, compute, 0),
        _Action(1, send, 0),
        _Action(1, recv, 1),
        _Action(1, compute, 1),
        _Action(1, send, 1),
    ]
    assert _busy_intervals(actions, 1, recv, consumers, {1}) == {0: (0, 2), 1: (3, 5)}


def test_flatten_overlap_uses_only_subactions():
    fwd, bwd = _Action(1, FORWARD, 1), _Action(2, BACKWARD_INPUT, 0)
    overlap = _Action(-1, _ComputationType.OVERLAP_F_B, None, (fwd, bwd))
    recv = _Action(1, RECV_F, 1)
    assert _flatten_actions([None, recv, overlap]) == [recv, fwd, bwd]


def test_stage_without_recvs_returns_none():
    assert (
        _busy_intervals([_Action(0, FORWARD, 0)], 0, RECV_F, _FWD_CONSUMERS, {0})
        is None
    )


def test_recv_without_consumer_raises():
    with pytest.raises(ValueError, match="no consuming compute action"):
        _busy_intervals([_Action(1, RECV_F, 0)], 1, RECV_F, _FWD_CONSUMERS, {1})


@pytest.mark.parametrize(
    "schedule_name",
    [
        "Interleaved1F1B",
        "InterleavedZeroBubble",
        "ZBVZeroBubble",
        "DualPipeV",
        "LoopedBFS",
    ],
)
@pytest.mark.parametrize(
    "pp, n_mb", [(pp, n) for pp in (2, 4, 8) for n in (4, 8, 16, 32) if n >= 2 * pp]
)
def test_real_lowered_schedules(schedule_name, pp, n_mb):
    for rank_actions in get_schedule_ops(schedule_name, pp, n_mb, 2, with_comms=True):
        actions = _flatten_actions(rank_actions)
        for stage_index in {a.stage_index for a in actions}:
            for recv, consumers in [(RECV_F, _FWD_CONSUMERS), (RECV_B, _BWD_CONSUMERS)]:
                intervals = _busy_intervals(
                    actions,
                    stage_index,
                    recv,
                    consumers,
                    {a.stage_index for a in actions},
                )
                if intervals is None:
                    continue
                assert set(intervals) == set(range(n_mb))
                slots = _assign_slots(intervals)
                # Independent sweep checks that no slot is shared by live MBs
                # and that the number of allocated slots is the exact minimum.
                peak = 0
                for pos in range(len(actions)):
                    live = [
                        mb
                        for mb, (start, end) in intervals.items()
                        if start <= pos <= end
                    ]
                    assert len({slots[mb] for mb in live}) == len(live)
                    peak = max(peak, len(live))
                assert len(set(slots.values())) == peak


def _parity_worker(rank, store_path, schedule_name, passthrough):
    from datetime import timedelta

    import torch
    import torch.distributed as dist
    from torch.distributed.pipelining.schedules import get_schedule_class

    from torchtitan.distributed.pipeline_parallel_recv_pool import (
        pool_pipeline_recv_buffers,
        RecvPoolStage,
    )

    class Carry(torch.nn.Module):
        """GIFT MTP-style passthrough: in training, middle stages return their
        second input as-is, so SEND_F posts a read of the stage's own fwd recv
        buffer, and the passthrough backward accumulates the output grad into
        that leaf (SEND_B then reads the bwd recv buffer). Like GIFT, the
        carry exists only in training; eval returns a single tensor.
        """

        def __init__(self, is_first: bool, is_last: bool, identity_out: bool):
            super().__init__()
            # Same shape as a microbatch: autograd then returns the output
            # grad itself as bias.grad (no reduction, no copy).
            self.bias = torch.nn.Parameter(torch.zeros(2, 4))
            self.is_first = is_first
            self.is_last = is_last
            # Last stage returns a view of its input: the merged step output
            # aliases the fwd recv buffer and is read after the step.
            self.identity_out = identity_out

        def forward(self, value, carry=None):
            h = value + self.bias
            if not self.training:
                return value.view_as(value) if self.identity_out else h
            if self.is_first:
                carry = value + 0 * self.bias  # requires grad, like embeddings
            if self.is_last:
                return carry.view_as(carry) if self.identity_out else h + carry
            return h, carry

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{store_path}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        baseline = []
        is_v_schedule = schedule_name in ("DualPipeV", "ZBVZeroBubble")
        for pooled in (False, True):
            indices = [rank, 3 - rank] if is_v_schedule else [rank, rank + 2]
            stages = []
            for index in indices:
                torch.manual_seed(42 + index)
                model = (
                    Carry(index == 0, index == 3, passthrough == "identity")
                    if passthrough
                    else torch.nn.Sequential(
                        torch.nn.Linear(4, 4), torch.nn.Tanh(), torch.nn.Linear(4, 4)
                    )
                )
                stages.append(RecvPoolStage(model, index, 4, torch.device("cpu")))
            schedule = get_schedule_class(schedule_name)(
                stages, n_microbatches=8, loss_fn=torch.nn.MSELoss()
            )
            if pooled:
                pool_pipeline_recv_buffers(schedule)
            phase = 0

            def record_or_compare(values, phase):
                if pooled:
                    for actual, expected in zip(values, baseline[phase], strict=True):
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                else:
                    baseline.append(values)

            for step in range(3):
                # Stock _PipelineScheduleRuntime.eval() asserts
                # "can't steal_bwd_input" on V-schedule local handoffs with or
                # without pooling, so the eval round trip only runs elsewhere.
                if step == 2 and not is_v_schedule:
                    # Train -> eval -> train round trip: the mode switch
                    # re-initializes recv infra, which must reapply the pool.
                    for stage in stages:
                        stage.submod.eval()
                    torch.manual_seed(300)
                    inputs, target = torch.randn(16, 4), torch.randn(16, 4)
                    eval_losses = []
                    with torch.no_grad():
                        output = schedule.eval(
                            inputs, target=target, losses=eval_losses
                        )
                    values = [loss.detach().clone() for loss in eval_losses]
                    if output is not None:
                        values.append(output.detach().clone())
                    record_or_compare(values, phase)
                    phase += 1
                    for stage in stages:
                        stage.submod.train()
                for stage in stages:
                    stage.submod.zero_grad(set_to_none=True)
                torch.manual_seed(100 + step)
                inputs, target = torch.randn(16, 4), torch.randn(16, 4)
                losses = []
                # Step 1 discards outputs: the last stage then leases its
                # inputs instead of copying them.
                output = schedule.step(
                    inputs, target=target, losses=losses, return_outputs=step != 1
                )
                for stage in stages:
                    assert stage.chunks == 8
                    for infos, direction, recv, consumers in (
                        (stage.args_recv_info, "fwd", RECV_F, _FWD_CONSUMERS),
                        (stage.grad_recv_info, "bwd", RECV_B, _BWD_CONSUMERS),
                    ):
                        assert set(infos) == set(range(8))
                        intervals = _busy_intervals(
                            _flatten_actions(schedule.pipeline_order_with_comms[rank]),
                            stage.stage_index,
                            recv,
                            consumers,
                            set(indices),
                        )
                        if intervals is None:
                            continue
                        num_slots = len(set(_assign_slots(intervals).values()))
                        pooled_dir = (
                            stage._fwd_lease is not None
                            if direction == "fwd"
                            else stage._bwd_pooled
                        )
                        if not pooled or num_slots == 8:
                            assert not pooled_dir
                            assert len({id(info) for info in infos.values()}) == 8
                            continue
                        assert pooled_dir
                        # Shared storage: distinct buffers == slots x set size.
                        buffers = [i.buffer for t in infos.values() for i in t]
                        buffers = [b for b in buffers if b is not None]
                        assert (
                            len({id(b) for b in buffers})
                            == len(buffers) // 8 * num_slots
                        )
                        if direction == "bwd" or stage.is_last:
                            continue  # compute got copies, nothing is leased
                        # After a step exactly one owner per slot still holds
                        # a live lease; the others were cut at slot reuse.
                        slot_of = getattr(stage, f"_{direction}_recv_slots")
                        owners: dict = {}
                        for mb, leaves in stage._fwd_lease.items():
                            assert all(t.numel() > 0 for t in leaves)
                            owners[slot_of[mb]] = owners.get(slot_of[mb], 0) + 1
                        assert set(owners) == set(range(num_slots))
                        assert set(owners.values()) == {1}
                values = [loss.detach().clone() for loss in losses]
                if output is not None:
                    values.append(output.detach().clone())
                values.extend(
                    (p.grad if p.grad is not None else torch.zeros_like(p)).clone()
                    for stage in stages
                    for p in stage.submod.parameters()
                )
                record_or_compare(values, phase)
                phase += 1
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "schedule_name",
    [
        "Interleaved1F1B",
        "InterleavedZeroBubble",
        "ZBVZeroBubble",
        "DualPipeV",
        "LoopedBFS",
    ],
)
@pytest.mark.parametrize("passthrough", [False, True, "identity"])
def test_real_pipeline_gradient_parity(tmp_path, schedule_name, passthrough):
    import torch.multiprocessing as mp

    mp.spawn(
        _parity_worker,
        args=(str(tmp_path / "store"), schedule_name, passthrough),
        nprocs=2,
        join=True,
    )
