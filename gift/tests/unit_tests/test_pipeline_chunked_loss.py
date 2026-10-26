# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace

import pytest
import torch
from torch.distributed.pipelining.schedules import ScheduleInterleaved1F1B

from torchtitan.components.loss import ChunkedCELoss
from torchtitan.distributed.pipeline_parallel import _schedule_with_chunked_loss


class _RecordingLoss(ChunkedCELoss):
    def __init__(self):
        super().__init__(ChunkedCELoss.Config())
        self.synchronizations = []

    def __call__(self, pred, labels, global_valid_tokens=None, *, sync_all_reduce=True):
        self.synchronizations.append(sync_all_reduce)
        return pred.sum()


@pytest.mark.parametrize("order", [(0, 1, 2), (2, 0, 1), (1, 2, 0)])
def test_native_loss_bookkeeping_flushes_last_executed_microbatch(order):
    loss = _RecordingLoss()
    schedule_type = _schedule_with_chunked_loss(ScheduleInterleaved1F1B, loss)
    # Exercise native loss bookkeeping without process groups/model execution.
    schedule = schedule_type.__new__(schedule_type)
    schedule._n_microbatches = 3
    schedule._internal_losses = []
    schedule._loss_fn = loss
    stage = SimpleNamespace(is_last=True)
    output = torch.ones(2, requires_grad=True)
    targets = [torch.zeros(2) for _ in range(3)]
    for optimizer_step in range(2):
        for accumulation in range(2):
            loss.is_last_accumulation_step = accumulation == 1
            if optimizer_step == accumulation == 0:
                # The stage calls the loss directly to infer backward metadata.
                schedule._loss_fn(output, targets[0])
                assert loss.synchronizations.pop() is True
                assert schedule._internal_losses == []
            for mb_index in order:
                schedule._maybe_compute_loss(stage, output, targets, mb_index)
            real_losses = []
            schedule._update_losses([stage], real_losses)
            assert len(real_losses) == 3
            assert schedule._internal_losses == []
    assert loss.synchronizations == [False, False, False, False, False, True] * 2
