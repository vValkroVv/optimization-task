from types import SimpleNamespace
from typing import Any, cast

import torch
import torch.distributed.checkpoint as dcp

import torchtitan.trainer as trainer_module
from torchtitan.components.aux_loss import AuxLossTracker
from torchtitan.components.loss import ChunkedCELoss
from torchtitan.models.flux.trainer import FluxTrainer
from torchtitan.callbacks.base import MutableState
from torchtitan.trainer import Trainer


def _build_trainer(trainer_type=Trainer):
    value = cast(Any, trainer_type.__new__(trainer_type))
    value.device = torch.device("cpu")
    value.step = 1
    value.ntokens_seen = torch.zeros((), dtype=torch.int64)
    value.gradient_accumulation_steps = 1
    value.model_parts = []
    value.optimizers = SimpleNamespace(
        zero_grad=lambda: None,
        step=lambda: None,
    )
    scheduler = SimpleNamespace(get_last_lr=lambda: [1.0])
    value.lr_schedulers = SimpleNamespace(
        schedulers=[scheduler],
        step=lambda: None,
    )
    value.checkpointer = SimpleNamespace(maybe_wait_for_staging=lambda: None)
    value.metrics_processor = SimpleNamespace(
        should_log=lambda step: False,
        ntokens_since_last_log=0,
        data_loading_times=[],
    )
    value.callbacks = SimpleNamespace(
        emit=lambda event, trainer, **fields: MutableState(trainer=trainer, **fields)
    )
    value.loss_fn = SimpleNamespace(
        reset_components=lambda: None, components=lambda: {}
    )
    value.parallel_dims = SimpleNamespace(
        dp_enabled=True,
        dp_cp_enabled=True,
        dp_replicate=2,
        dp_shard=4,
        cp=1,
        ep_enabled=False,
        pp_enabled=False,
        get_mesh=lambda name: name,
        get_optional_mesh=lambda name: None,
    )
    value.aux_loss = AuxLossTracker(
        value.model_parts, parallel_dims=value.parallel_dims, device=value.device
    )
    value.config = SimpleNamespace(
        training=SimpleNamespace(max_norm=1.0, seq_len=10),
        parallelism=SimpleNamespace(fsdp_backend="fsdp2"),
    )
    return value


def test_train_step_stores_global_token_count(monkeypatch):
    trainer = _build_trainer()
    seen = {}

    def forward_backward_step(**kwargs):
        seen["global_valid_tokens"] = kwargs["global_valid_tokens"]
        return torch.tensor(0.0)

    def reduce(value, mesh):
        assert value.item() == 3
        assert mesh == "batch"
        return 37

    trainer.forward_backward_step = forward_backward_step
    monkeypatch.setattr(trainer_module.dist_utils, "dist_sum", reduce)
    monkeypatch.setattr(
        trainer_module.dist_utils,
        "clip_grad_norm_",
        lambda *args, **kwargs: torch.tensor(0.0),
    )
    labels = torch.tensor([[1, 2, -100, 3]])
    data_iterator = iter([({"input": torch.ones_like(labels)}, labels)])
    trainer.train_step(data_iterator)
    assert seen["global_valid_tokens"] == 37
    assert trainer.ntokens_seen.item() == 37


def test_flux_train_step_stores_global_token_count(monkeypatch):
    trainer = _build_trainer(FluxTrainer)
    trainer.forward_backward_step = lambda **kwargs: torch.tensor(0.0)

    def reduce(value, mesh):
        assert value.item() == 20
        assert mesh == "batch"
        return 200

    monkeypatch.setattr(trainer_module.dist_utils, "dist_sum", reduce)
    monkeypatch.setattr(
        trainer_module.dist_utils,
        "clip_grad_norm_",
        lambda *args, **kwargs: torch.tensor(0.0),
    )
    labels = torch.zeros((2, 1))
    data_iterator = iter([({}, labels)])
    trainer.train_step(data_iterator)
    assert trainer.ntokens_seen.item() == 200


def test_checkpoint_token_count_is_topology_independent(tmp_path):
    source = _build_trainer()
    source.step = 9
    source.ntokens_seen.fill_(12345)
    assert source.state_dict() == {"step": 9, "ntokens_seen": 12345}
    source_path = tmp_path / "checkpoint"
    dcp.save({"train_state": source}, checkpoint_id=source_path)

    target = _build_trainer()
    target.parallel_dims = SimpleNamespace(
        dp_replicate=3,
        dp_shard=8,
        cp=2,
    )
    dcp.load({"train_state": target}, checkpoint_id=source_path)
    assert target.step == 9
    assert target.ntokens_seen.item() == 12345


def test_chunked_loss_accumulation_boundary_resets_each_optimizer_step(monkeypatch):
    trainer = _build_trainer()
    trainer.gradient_accumulation_steps = 3
    trainer.loss_fn = ChunkedCELoss(ChunkedCELoss.Config())
    observed = []
    trainer.forward_backward_step = lambda **kwargs: (
        observed.append(trainer.loss_fn.is_last_accumulation_step) or torch.tensor(0.0)
    )
    monkeypatch.setattr(trainer_module.dist_utils, "dist_sum", lambda value, mesh: 3)
    monkeypatch.setattr(
        trainer_module.dist_utils,
        "clip_grad_norm_",
        lambda *args, **kwargs: torch.tensor(0.0),
    )
    labels = torch.tensor([[1]])
    for _ in range(2):
        trainer.train_step(iter([({"input": labels}, labels)] * 3))
    assert observed == [False, False, True, False, False, True]
