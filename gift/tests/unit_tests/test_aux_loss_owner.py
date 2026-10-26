# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""One owner for the step's loss decomposition: ``AuxLossTracker``.

The trainer used to reset the loss's components in one place, read them back in
another and reduce them itself, while the tracker did the same for the aux/z
splices. These pin the merged lifecycle: opening a step opens both halves, the
pipeline's shape probe is dropped through the tracker, and every term leaves
under one call.
"""

import torch

import torchtitan.components.aux_loss as aux_loss_module

from torchtitan.components.aux_loss import AuxLossTracker
from torchtitan.components.loss import CrossEntropyLoss, MTPLoss


class _SingleRankDims:
    dp_cp_enabled = False
    pp_enabled = False

    def get_optional_mesh(self, name: str):
        del name
        return None


def _tracker(loss_fn) -> AuxLossTracker:
    return AuxLossTracker(
        [],
        parallel_dims=_SingleRankDims(),
        device=torch.device("cpu"),
        loss_fn=loss_fn,
    )


def _ce_loss() -> CrossEntropyLoss:
    return CrossEntropyLoss(CrossEntropyLoss.Config())


def test_declared_components_name_what_the_loss_can_produce():
    assert _ce_loss().declared_components() == ("loss_ce",)

    one_head = MTPLoss(MTPLoss.Config(loss_weights=[0.1]))
    assert one_head.declared_components() == ("loss_ce", "loss_mtp")

    two_heads = MTPLoss(MTPLoss.Config(loss_weights=[0.1, 0.05]))
    assert two_heads.declared_components() == (
        "loss_ce",
        "loss_mtp",
        "loss_mtp1",
        "loss_mtp2",
    )

    # A head that contributes nothing is not named: a tag that is a constant
    # zero is worse than no tag.
    muted = MTPLoss(MTPLoss.Config(loss_weights=[0.1, 0.0]))
    assert muted.declared_components() == ("loss_ce", "loss_mtp", "loss_mtp1")


def test_metrics_publish_the_loss_components():
    loss_fn = _ce_loss()
    tracker = _tracker(loss_fn)
    tracker.begin_step()
    loss_fn.add_component("loss_ce", torch.tensor(2.5))

    assert tracker.metrics() == {"loss/train/loss_ce": 2.5}


def test_begin_step_opens_both_halves():
    loss_fn = _ce_loss()
    tracker = _tracker(loss_fn)
    tracker.begin_step()
    loss_fn.add_component("loss_ce", torch.tensor(2.5))

    # Without this the step's terms carry into the next one.
    tracker.begin_step()
    assert tracker.metrics() == {}


def test_keep_microbatches_drops_the_pipeline_shape_probe():
    loss_fn = _ce_loss()
    tracker = _tracker(loss_fn)
    tracker.begin_step()

    # A real microbatch, the probe (which never enters the schedule's
    # ``losses``), then another real microbatch — the order the first PP step
    # actually runs them in.
    losses = []
    for value, real in ((1.0, True), (100.0, False), (2.0, True)):
        loss_fn._begin_call()
        loss_fn.add_component("loss_ce", torch.tensor(value))
        returned = loss_fn._finish_call(torch.tensor(value))
        if real:
            losses.append(returned)

    tracker.keep_microbatches(losses)
    assert tracker.metrics() == {"loss/train/loss_ce": 3.0}


def test_mtp_metrics_batch_reduction_and_reset(monkeypatch):
    meshes = {"loss": object(), "pp": object()}

    class DataParallelDims(_SingleRankDims):
        dp_cp_enabled = True
        pp_enabled = True

        def get_optional_mesh(self, name: str):
            return meshes[name]

    calls = []

    def all_reduce(values, reduceOp, group):
        assert reduceOp == "SUM"
        assert group is meshes["loss"]
        assert not values.requires_grad
        calls.append(values.numel())
        return values * 2

    monkeypatch.setattr(aux_loss_module.funcol, "all_reduce", all_reduce)
    loss_fn = MTPLoss(MTPLoss.Config(loss_weights=[0.1, 0.05], num_chunks=2))
    loss_fn.set_lm_head(torch.nn.Linear(4, 8, bias=False))
    tracker = AuxLossTracker(
        [],
        parallel_dims=DataParallelDims(),
        device=torch.device("cpu"),
        loss_fn=loss_fn,
    )

    # The same MTP loss publishes all heads in training and just CE in eval.
    for with_mtp in (True, False):
        tracker.begin_step()
        for _ in range(2):
            shape = (3, 2, 6, 4) if with_mtp else (2, 6, 4)
            loss_fn(
                torch.randn(shape),
                torch.randint(0, 8, (2, 6)),
                torch.tensor(24.0),
            )
        names = (
            ("loss_ce", "loss_mtp", "loss_mtp1", "loss_mtp2")
            if with_mtp
            else ("loss_ce",)
        )
        produced = loss_fn.components()
        expected = {f"loss/train/{name}": float(produced[name]) * 2 for name in names}
        calls.clear()
        assert tracker.metrics() == expected
        assert calls == [len(names)]

        tracker.begin_step()
        assert tracker.metrics() == {}
        assert calls == [len(names)]
