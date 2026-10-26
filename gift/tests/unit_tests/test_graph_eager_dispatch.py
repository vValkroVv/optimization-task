"""GraphTrainer honors disabled compilation before entering tracing machinery."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from torchtitan.config import load_raw_document
from torchtitan.config.build import prepare_document
from torchtitan.config.schema.experiments.graph_trainer.trainer import (
    GraphTrainerConfig,
)
from torchtitan.experiments.graph_trainer import trainer as graph_module
from torchtitan.experiments.graph_trainer.trainer import GraphTrainer
from torchtitan.trainer import Trainer


def _config():
    raw = load_raw_document("configs/experiments/graph_llama3_debug.yaml")
    config = prepare_document(raw).trainer_config
    assert isinstance(config, GraphTrainerConfig)
    return config


@pytest.mark.parametrize("mode", ["jit", "aot_fx_trace"])
def test_disabled_compilation_uses_base_forward_backward(monkeypatch, mode):
    config = _config()
    config.compile.enable = False
    config.compile.mode = mode
    config.compile.precompile_artifact_dir = "unused-artifact"
    trainer = GraphTrainer.__new__(GraphTrainer)
    trainer.config = config
    expected = torch.tensor(3.0)
    eager = Mock(return_value=expected)
    trace = Mock(side_effect=AssertionError("disabled compilation entered AOT"))
    monkeypatch.setattr(Trainer, "forward_backward_step", eager)
    monkeypatch.setattr(trainer, "_make_fx_forward_backward_step", trace)
    arguments = dict(
        input_dict={"tokens": torch.ones(2, dtype=torch.long)},
        labels=torch.zeros(2, dtype=torch.long),
        global_valid_tokens=torch.tensor(2),
    )
    assert trainer.forward_backward_step(**arguments) is expected
    eager.assert_called_once_with(**arguments)
    trace.assert_not_called()


def test_enabled_jit_still_uses_base_forward_backward(monkeypatch):
    config = _config()
    config.compile.enable = True
    config.compile.mode = "jit"
    trainer = GraphTrainer.__new__(GraphTrainer)
    trainer.config = config
    eager = Mock(return_value=torch.tensor(2.0))
    monkeypatch.setattr(Trainer, "forward_backward_step", eager)
    trainer.forward_backward_step(
        input_dict={}, labels=torch.tensor(0), global_valid_tokens=torch.tensor(1)
    )
    eager.assert_called_once()


def test_enabled_aot_keeps_traced_dispatch(monkeypatch):
    config = _config()
    config.compile.enable = True
    trainer = GraphTrainer.__new__(GraphTrainer)
    trainer.config = config
    model = torch.nn.Linear(2, 2)
    trainer.model_parts = [model]
    labels, valid = torch.tensor(0), torch.tensor(1)
    processed = (torch.ones(2), labels, {}, {})
    trainer.post_dataloading_process = Mock(return_value=processed)
    trace = Mock(return_value=torch.tensor(4.0))
    trainer._make_fx_forward_backward_step = trace
    assert (
        trainer.forward_backward_step(
            input_dict={}, labels=labels, global_valid_tokens=valid
        ).item()
        == 4.0
    )
    trace.assert_called_once()
    assert trace.call_args.args[0] is model


@pytest.mark.parametrize("enabled", [False, True])
def test_pipeline_restriction_applies_only_to_enabled_aot(monkeypatch, enabled):
    config = _config()
    config.compile.enable = enabled
    config.profiler.trace_post_processor = lambda value: value

    def initialize_base(self, received):
        self.config = received
        self.parallel_dims = SimpleNamespace(pp_enabled=True)

    monkeypatch.setattr(Trainer, "__init__", initialize_base)
    monkeypatch.setattr(graph_module, "POST_INIT_HOOKS", {})
    expectation = (
        pytest.raises(ValueError, match="does not support Pipeline")
        if enabled
        else nullcontext()
    )
    with expectation:
        GraphTrainer(config)


def test_disabled_graph_matches_actual_eager_cpu_loss_and_gradients():
    class Loss:
        def pack_targets(self, labels, extra_kwargs):
            return labels

        def __call__(self, prediction, labels, valid):
            return (
                torch.nn.functional.cross_entropy(prediction, labels, reduction="sum")
                / valid
            )

    inputs = torch.tensor([[0.2, -0.4], [0.3, 0.1]])
    labels = torch.tensor([0, 1])
    valid = torch.tensor(2)
    trainers = []
    for _ in range(2):
        trainer = GraphTrainer.__new__(GraphTrainer)
        trainer.config = _config()
        trainer.config.compile.enable = False
        model = torch.nn.Linear(2, 2)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[0.1, 0.2], [-0.1, 0.3]]))
            model.bias.zero_()
        trainer.model_parts = [model]
        trainer.parallel_dims = SimpleNamespace(pp_enabled=False)
        trainer.post_dataloading_process = lambda data, target: (
            data["tokens"],
            target,
            {},
            {},
        )
        trainer.aux_loss = SimpleNamespace(weigh=lambda **kwargs: None)
        trainer.loss_fn = Loss()
        trainer.train_context = nullcontext
        trainers.append(trainer)
    arguments = dict(
        input_dict={"tokens": inputs}, labels=labels, global_valid_tokens=valid
    )
    graph_loss = trainers[0].forward_backward_step(**arguments)
    eager_loss = Trainer.forward_backward_step(trainers[1], **arguments)
    assert torch.equal(graph_loss, eager_loss)
    for actual, expected in zip(
        trainers[0].model_parts[0].parameters(),
        trainers[1].model_parts[0].parameters(),
        strict=True,
    ):
        assert actual.grad is not None and expected.grad is not None
        assert torch.equal(actual.grad, expected.grad)
