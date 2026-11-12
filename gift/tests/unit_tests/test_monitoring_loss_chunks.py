"""Monitoring changes the head chunk count only inside the training step."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import torchtitan.trainer as trainer_module
from tests.unit_tests.test_trainer_token_count import _build_trainer
from torchtitan.callbacks.builtin.monitoring import MonitoringCallback
from torchtitan.components.loss import ChunkedCELoss, cross_entropy_loss
from torchtitan.config.schema.components.loss import ChunkedCELossConfig, MTPLossConfig


@pytest.mark.parametrize("name", ["num_chunks", "num_chunks_on_monitoring_steps"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_chunk_counts_require_positive_integers(name, value):
    with pytest.raises(ValueError, match=name):
        ChunkedCELossConfig(**{name: value})
    with pytest.raises(ValueError, match=name):
        MTPLossConfig(loss_weights=[0.3], **{name: value})


@pytest.mark.parametrize(
    "step, sources, expected", [(49, [object()], 2), (50, [object()], 32), (50, [], 2)]
)
@pytest.mark.parametrize("fail", [False, True])
def test_trainer_uses_monitoring_count_and_restores_after_failure(
    monkeypatch, step, sources, expected, fail
):
    trainer = _build_trainer()
    trainer.step = step
    trainer.gradient_accumulation_steps = 2
    trainer.parallel_dims.dp_enabled = False
    trainer.loss_fn = ChunkedCELoss(
        ChunkedCELoss.Config(num_chunks=2, num_chunks_on_monitoring_steps=32)
    )
    monitor = MonitoringCallback.__new__(MonitoringCallback)
    monitor.sources = sources
    monitor.config = SimpleNamespace(log_interval=50)
    trainer.callbacks.handler = SimpleNamespace(callbacks=[monitor])
    observed = []

    def forward_backward_step(**kwargs):
        observed.append(trainer.loss_fn.num_chunks)
        if fail:
            raise RuntimeError("forward failed")
        return torch.tensor(0.0)

    trainer.forward_backward_step = forward_backward_step
    monkeypatch.setattr(
        trainer_module.dist_utils,
        "clip_grad_norm_",
        lambda *args, **kwargs: torch.tensor(0.0),
    )
    labels = torch.tensor([[1]])
    data = iter([({"input": labels}, labels)] * 2)
    if fail:
        with pytest.raises(RuntimeError, match="forward failed"):
            trainer.train_step(data)
    else:
        trainer.train_step(data)
    assert observed == [expected] * (1 if fail else 2)
    assert trainer.loss_fn.num_chunks == 2


@pytest.mark.parametrize("num_chunks", [2, 32])
def test_chunked_head_loss_and_gradients_match_unchunked(num_chunks):
    torch.manual_seed(42)
    head = nn.Linear(8, 17, bias=False)
    reference_head = nn.Linear(8, 17, bias=False)
    reference_head.load_state_dict(head.state_dict())
    hidden = torch.randn(2, 64, 8, requires_grad=True)
    reference_hidden = hidden.detach().clone().requires_grad_(True)
    labels = torch.randint(0, 17, (2, 64))
    labels[0, ::7] = -100
    tokens = (labels != -100).sum()
    loss = ChunkedCELoss(ChunkedCELoss.Config(num_chunks=num_chunks))
    loss.set_lm_head(head)
    actual = loss(hidden, labels, tokens)
    actual.backward()
    expected = cross_entropy_loss(reference_head(reference_hidden), labels) / tokens
    expected.backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(hidden.grad, reference_hidden.grad)
    torch.testing.assert_close(head.weight.grad, reference_head.weight.grad)
