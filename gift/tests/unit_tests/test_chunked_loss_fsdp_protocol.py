# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""What the chunked loss says to ``lm_head``'s sharding unit, and when.

``ChunkedCELoss`` is the only place in the model that calls ``backward`` itself,
once per sequence chunk, and each of those calls ends an autograd pass. Three of
the four defects fixed in the chunked-loss MR were about what FSDP does at the
end of a pass, so the loop drives the unit through the ``FSDPModule`` protocol:

* ``set_reshard_after_forward(False)`` / ``set_reshard_after_backward(False)``
  hold ``lm_head`` gathered across every chunk instead of all-gathering it N
  times;
* ``set_requires_gradient_sync(False)`` on chunks ``0..N-2`` coalesces N
  reduce-scatters into one, and the last chunk turns it back on;
* ``set_is_last_backward(False)`` for the length of the loop stops the root
  callback from running the per-step teardown N times -- that teardown waits on
  a prefetched all-gather and throws it away, and the next chunk gathers the
  same unit again. Measured at 3b: ``mtp.layers.0`` ran 9 all-gathers a step
  where 2 would do.

None of this is visible in a loss curve, which is why it needs a test. The
order matters as much as the calls, so these assert the sequence.

The double is built the way ``fully_shard`` builds the real thing -- by
rebinding ``__class__`` on an ordinary module -- because ``FSDPModule`` cannot
be constructed directly: its ``__new__`` reaches into ``cls.__mro__`` to
instantiate the original class instead.
"""

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.distributed.fsdp import FSDPModule  # noqa: E402

from torchtitan.components.loss import ChunkedCELoss  # noqa: E402

_B, _S, _D, _V = 2, 8, 4, 16
_CHUNKS = 4


class _Recorder:
    """The ``FSDPModule`` protocol, written down instead of executed."""

    calls: list

    def set_reshard_after_forward(self, value, recurse=True):
        self.calls.append(("reshard_after_forward", value))

    def set_reshard_after_backward(self, value, recurse=True):
        self.calls.append(("reshard_after_backward", value))

    def set_requires_gradient_sync(self, value, recurse=True):
        self.calls.append(("requires_gradient_sync", value))
        for group in self.state._fsdp_param_groups:
            group.reduce_grads = value

    def set_is_last_backward(self, value):
        self.calls.append(("is_last_backward", value))
        self.state._state_ctx.is_last_backward = value

    def reshard(self):
        self.calls.append(("reshard", None))

    def _get_fsdp_state(self):
        return self.state

    def forward(self, *args, **kwargs):
        self.calls.append(("forward", None))
        return super().forward(*args, **kwargs)


def _sharded_linear(
    *, is_last_backward: bool = True, requires_gradient_sync: bool = True
) -> nn.Linear:
    """An ``nn.Linear`` that answers ``isinstance(m, FSDPModule)``.

    Class rebinding, not construction: ``fully_shard`` does exactly this to the
    module it shards, and instantiating a subclass of ``FSDPModule`` directly
    goes through its ``__new__``, which builds the original class instead. The
    flags are what a caller -- a pipeline schedule, say -- left set on entry.
    """
    torch.manual_seed(0)
    module = nn.Linear(_D, _V, bias=False)
    bases: tuple[type, ...] = (_Recorder, FSDPModule, type(module))
    module.__class__ = type("FSDPLinear", bases, {})
    module.calls = []
    module.state = SimpleNamespace(
        _state_ctx=SimpleNamespace(is_last_backward=is_last_backward),
        _fsdp_param_groups=[SimpleNamespace(reduce_grads=requires_gradient_sync)],
    )
    return module


def _loss(*, lm_head=None) -> ChunkedCELoss:
    loss_fn = ChunkedCELoss(ChunkedCELoss.Config(num_chunks=_CHUNKS))
    loss_fn.lm_head = _sharded_linear() if lm_head is None else lm_head
    loss_fn.reset_components()
    return loss_fn


def _inputs(*, requires_grad: bool = True):
    torch.manual_seed(1)
    hidden = torch.randn(_B, _S, _D, requires_grad=requires_grad)
    labels = torch.randint(0, _V, (_B, _S))
    return hidden, labels, torch.tensor(float(_B * _S))


def _run(loss_fn: ChunkedCELoss, **kwargs) -> torch.Tensor:
    hidden, labels, valid = _inputs(**kwargs)
    return loss_fn(hidden, labels, valid)


def test_the_unit_is_held_gathered_and_released_once_at_the_end():
    loss_fn = _loss()
    _run(loss_fn)
    calls = loss_fn.lm_head.calls

    assert calls[0] == ("reshard_after_forward", False)
    assert calls[1] == ("reshard_after_backward", False)
    assert calls[-1] == ("reshard", None)
    assert ("reshard_after_forward", True) in calls
    assert ("reshard_after_backward", True) in calls
    # One entry per chunk and no more: the point of holding it gathered.
    assert sum(1 for call in calls if call[0] == "forward") == _CHUNKS


def test_the_teardown_is_deferred_for_the_length_of_the_loop():
    """``set_is_last_backward(False)`` before the first chunk, ``True`` after
    the last -- not on the last chunk, because the decoder's backward has not
    run yet and it is the one that should carry the teardown."""
    loss_fn = _loss()
    _run(loss_fn)
    calls = loss_fn.lm_head.calls

    deferred = calls.index(("is_last_backward", False))
    restored = calls.index(("is_last_backward", True))
    forwards = [i for i, call in enumerate(calls) if call[0] == "forward"]

    assert sum(1 for call in calls if call[0] == "is_last_backward") == 2
    assert deferred < forwards[0]
    assert restored > forwards[-1]


def test_gradient_sync_is_coalesced_into_the_last_chunk():
    loss_fn = _loss()
    _run(loss_fn)
    calls = loss_fn.lm_head.calls

    disabled = [
        i for i, c in enumerate(calls) if c == ("requires_gradient_sync", False)
    ]
    enabled = [i for i, c in enumerate(calls) if c == ("requires_gradient_sync", True)]
    forwards = [i for i, call in enumerate(calls) if call[0] == "forward"]

    assert len(disabled) == 1, "sync is suppressed once, for the whole loop"
    assert disabled[0] < forwards[0]
    # Re-enabled before the final chunk runs, so its backward carries the one
    # reduce-scatter the others were folded into.
    assert forwards[-2] < enabled[0] < forwards[-1]


def test_a_suppressed_teardown_is_given_back_suppressed():
    """A pipeline schedule turns the teardown off for a whole stage; the loop
    borrows the flag for its chunks and must hand back what it found."""
    loss_fn = _loss(lm_head=_sharded_linear(is_last_backward=False))
    _run(loss_fn)
    head = loss_fn.lm_head

    assert ("is_last_backward", True) not in head.calls
    assert head.state._state_ctx.is_last_backward is False


def test_a_suppressed_sync_is_reenabled_for_the_last_chunk():
    """torch FSDP: a no-sync flag left by the stock PP backward must not make
    the head accumulate an unsharded fp32 gradient across microbatches; the
    last chunk reduce-scatters unconditionally."""
    loss_fn = _loss(lm_head=_sharded_linear(requires_gradient_sync=False))
    _run(loss_fn)
    head = loss_fn.lm_head

    assert ("requires_gradient_sync", True) in head.calls
    assert all(g.reduce_grads is True for g in head.state._fsdp_param_groups)


def test_a_suppressed_sync_stays_suppressed_under_gfsdp(monkeypatch):
    """gfsdp manages its buffer lifecycle outside the loss: the loss must
    stay transparent to the sync flags it finds (see ``_is_plain_fsdp``)."""
    import torchtitan.components.loss as loss_module

    monkeypatch.setattr(loss_module, "_is_plain_fsdp", lambda module: False)
    loss_fn = _loss(lm_head=_sharded_linear(requires_gradient_sync=False))
    _run(loss_fn)
    head = loss_fn.lm_head

    assert ("requires_gradient_sync", True) not in head.calls
    assert all(g.reduce_grads is False for g in head.state._fsdp_param_groups)


def test_the_flags_are_given_back_when_a_chunk_raises():
    loss_fn = _loss()

    def _boom(*args, **kwargs):
        raise RuntimeError("chunk blew up")

    loss_fn.fn = _boom
    with pytest.raises(RuntimeError, match="chunk blew up"):
        _run(loss_fn)

    head = loss_fn.lm_head
    assert head.state._state_ctx.is_last_backward is True
    assert all(g.reduce_grads is True for g in head.state._fsdp_param_groups)


def test_an_unsharded_head_is_left_alone():
    """The eager path: no FSDP, no protocol calls, same loss."""
    torch.manual_seed(0)
    plain = nn.Linear(_D, _V, bias=False)
    reference = _loss(lm_head=plain)
    hidden, labels, valid = _inputs()
    expected = reference(hidden, labels, valid)

    loss_fn = _loss()
    hidden, labels, valid = _inputs()
    actual = loss_fn(hidden, labels, valid)

    assert torch.allclose(expected, actual)


def test_the_protocol_calls_do_not_change_the_number():
    """Everything above is bookkeeping; the loss must not notice it.

    A chunked loss over N chunks is also the check that chunking itself is
    neutral -- one chunk and four must agree.
    """
    torch.manual_seed(0)
    head = nn.Linear(_D, _V, bias=False)
    hidden, labels, valid = _inputs()

    one = ChunkedCELoss(ChunkedCELoss.Config(num_chunks=1))
    one.lm_head = head
    one.reset_components()

    many = ChunkedCELoss(ChunkedCELoss.Config(num_chunks=_CHUNKS))
    many.lm_head = head
    many.reset_components()

    assert torch.allclose(
        one(hidden, labels, valid), many(hidden, labels, valid), atol=1e-6
    )


def test_validation_takes_no_backward():
    loss_fn = _loss()
    loss = _run(loss_fn, requires_grad=False)

    assert torch.isfinite(loss).all()
    assert loss_fn.lm_head.weight.grad is None
