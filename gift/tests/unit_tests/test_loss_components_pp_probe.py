# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The pipeline's shape probe is not a microbatch, wherever it lands.

On its first ``schedule.step`` the last stage runs the loss once more to learn
the gradient shapes (``stage.py::_backward_metadata_inference``), and that
value never enters the schedule's ``losses``. WHERE it runs among the real
microbatches is a torch implementation detail — currently after the first
forward, which is exactly where trimming by count kept the probe and dropped a
real microbatch (NaN loss_ce on step 1 under PP with EMA, measured). The loss
therefore stamps every call with the identity of the tensor it returned, and
``keep_microbatches`` keeps the calls whose stamp is in ``losses``.
"""

import pytest
import torch
import torch.nn as nn

from torchtitan.components.loss import ChunkedCELoss, CrossEntropyLoss


def _loss() -> CrossEntropyLoss:
    loss_fn = CrossEntropyLoss(CrossEntropyLoss.Config())
    loss_fn.reset_components()
    return loss_fn


def _batch(vocab: int = 8, tokens: int = 4):
    torch.manual_seed(0)
    return torch.randn(1, tokens, vocab), torch.randint(0, vocab, (1, tokens))


def _reference(n: int) -> torch.Tensor:
    loss_fn = _loss()
    pred, labels = _batch()
    for _ in range(n):
        loss_fn(pred, labels, torch.tensor(4.0))
    return loss_fn.components()[loss_fn.component_name]


@pytest.mark.parametrize("probe_position", [0, 1, 4, 8])
def test_the_probe_is_dropped_wherever_it_lands(probe_position):
    loss_fn = _loss()
    pred, labels = _batch()
    nan_pred = torch.full_like(pred, float("nan"))

    losses = []
    for index in range(9):
        if index == probe_position:
            loss_fn(nan_pred, labels)  # the probe; never enters ``losses``
        else:
            losses.append(loss_fn(pred, labels, torch.tensor(4.0)))
    loss_fn.keep_microbatches(losses)

    recorded = loss_fn.components()[loss_fn.component_name]
    assert torch.isfinite(recorded).all()
    assert torch.equal(recorded, _reference(8))


def test_each_call_is_its_own_slot_even_without_begin_call():
    # ChunkedCELoss (and this CE via BaseLoss) close a slot on return; a loss
    # that never opens one explicitly must still not fold two microbatches
    # into one dict, or the probe could never be taken back out.
    loss_fn = _loss()
    pred, labels = _batch()
    loss_fn(pred, labels, torch.tensor(4.0))
    loss_fn(pred, labels, torch.tensor(4.0))
    assert len(loss_fn._calls) == 2


def test_gradient_accumulation_keeps_earlier_microbatches():
    # The trainer prunes once per schedule.step; calls accepted by a previous
    # prune sit behind the watermark and must survive the next one.
    loss_fn = _loss()
    pred, labels = _batch()

    first = [loss_fn(pred, labels, torch.tensor(4.0)) for _ in range(2)]
    loss_fn(nan_pred := torch.full_like(pred, float("nan")), labels)  # probe
    loss_fn.keep_microbatches(first)
    second = [loss_fn(pred, labels, torch.tensor(4.0)) for _ in range(2)]
    loss_fn.keep_microbatches(second)

    recorded = loss_fn.components()[loss_fn.component_name]
    assert torch.isfinite(recorded).all()
    assert torch.equal(recorded, _reference(4))
    del nan_pred


def test_out_of_sync_bookkeeping_fails_loudly():
    loss_fn = _loss()
    pred, labels = _batch()
    loss_fn(pred, labels, torch.tensor(4.0))
    stranger = torch.tensor(1.0)
    with pytest.raises(RuntimeError, match="out of sync"):
        loss_fn.keep_microbatches([stranger])


# The probe must also leave no parameter gradients behind.
#
# ``ChunkedCELoss`` calls ``backward`` itself, once per sequence chunk, so the
# probe does not merely record a value -- it produces real parameter gradients
# from the uninitialized activations the schedule feeds it. Under FSDP2 those
# land in ``param.grad``; under GFSDP2 they land in a private per-group buffer
# that ``optimizer.zero_grad`` does not reach, and the step's real
# reduce-scatter adds onto them, so every parameter of the group goes nan on
# step 1. The loss therefore recognizes the probe by its target and computes
# only the gradient w.r.t. its input.
_B, _S, _D, _V = 2, 4, 8, 16


def _chunked() -> tuple[nn.Linear, ChunkedCELoss]:
    torch.manual_seed(0)
    lm_head = nn.Linear(_D, _V, bias=False)
    loss_fn = ChunkedCELoss(ChunkedCELoss.Config(num_chunks=2))
    loss_fn.lm_head = lm_head
    loss_fn.reset_components()
    return lm_head, loss_fn


def _hidden() -> torch.Tensor:
    return torch.randn(_B, _S, _D, requires_grad=True)


def _tokens() -> torch.Tensor:
    return torch.randint(0, _V, (_B, _S))


def test_probe_produces_no_parameter_gradients():
    lm_head, loss_fn = _chunked()
    labels = _tokens()
    loss_fn.set_step_targets(labels)

    hidden = _hidden()
    # A target the schedule allocated: same shape, storage of its own.
    probe_loss = loss_fn(hidden, _tokens(), torch.tensor(float(_B * _S)))
    assert lm_head.weight.grad is None

    # Still differentiable into the stage -- that is what the probe is for.
    probe_loss.backward()
    assert hidden.grad is not None
    assert lm_head.weight.grad is None


def test_microbatch_view_of_the_registered_target_still_trains():
    lm_head, loss_fn = _chunked()
    labels = _tokens()
    loss_fn.set_step_targets(labels)

    # What the schedule hands a real microbatch: a view of ``target``.
    microbatch = torch.chunk(labels, 2, dim=0)[0]
    loss = loss_fn(_hidden()[:1], microbatch, torch.tensor(float(_S)))
    assert lm_head.weight.grad is not None
    assert torch.isfinite(loss).all()


def test_unregistered_targets_disarm_the_test():
    lm_head, loss_fn = _chunked()
    # Nothing registered: the non-pipeline path, where no call is a probe.
    loss_fn(_hidden(), _tokens(), torch.tensor(float(_B * _S)))
    assert lm_head.weight.grad is not None


def test_a_second_probe_is_refused():
    _, loss_fn = _chunked()
    # Held, as the trainer holds it across the step: registering a tensor that
    # dies immediately would free its address for the next allocation to reuse,
    # and a probe target landing there would read as a microbatch.
    labels = _tokens()
    loss_fn.set_step_targets(labels)
    loss_fn(_hidden(), _tokens(), torch.tensor(float(_B * _S)))
    with pytest.raises(RuntimeError, match="metadata probe"):
        loss_fn(_hidden(), _tokens(), torch.tensor(float(_B * _S)))


def test_a_dead_targets_address_is_not_reused_as_a_microbatch():
    """Registration must survive its argument going out of scope.

    Held only by address, a registered target that dies frees that address for
    the next allocation of the same shape, and the probe landing there reads
    as a microbatch -- the protection switches itself off, silently.
    """
    _, loss_fn = _chunked()
    loss_fn.set_step_targets(_tokens())  # deliberately kept by nobody else
    # Enough same-shaped allocations that one would land on the freed address.
    # The storage test alone, so the one-probe budget is not in the way.
    assert not any(loss_fn._is_step_target(_tokens()) for _ in range(8))


def test_a_probe_after_the_first_step_is_refused():
    """Only the first ``schedule.step`` may produce a probe.

    The schedule infers gradient metadata when it initializes its stages. An
    unknown target on a later step means microbatch targets stopped being
    views of what was registered, and taking one for the probe would drop that
    microbatch's parameter gradients with nothing to show for it.
    """
    _, loss_fn = _chunked()
    first = _tokens()
    loss_fn.set_step_targets(first)
    loss_fn(_hidden(), first, torch.tensor(float(_B * _S)))
    loss_fn.set_step_targets(None)

    second = _tokens()
    loss_fn.set_step_targets(second)
    with pytest.raises(RuntimeError, match="metadata probe"):
        loss_fn(_hidden(), _tokens(), torch.tensor(float(_B * _S)))
