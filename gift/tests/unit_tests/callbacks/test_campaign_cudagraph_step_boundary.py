import random
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch

import torchtitan.trainer as trainer_module
from torchtitan.callbacks.base import Event
from torchtitan.callbacks.manager import CallbackManager
from torchtitan.config.configurable import to_plain
from torchtitan.config.decode import decode
from torchtitan.config.schema.callbacks.base import CallbackConfig
from torchtitan.config.schema.experiments.cuda_graph_campaign.continuation import (
    CampaignRNGContinuationConfig,
)
from torchtitan.config.schema.experiments.cuda_graph_campaign.observer import (
    CampaignTimingObserverConfig,
)
from torchtitan.config.schema.experiments.cuda_graph_campaign.step_boundary import (
    CampaignCUDAGraphStepBoundaryConfig,
)
from torchtitan.config.schema.experiments.cuda_graph_campaign.trainer import (
    CampaignTrainerConfig,
)
from torchtitan.config.schema.trainer import TrainerConfig
from torchtitan.config.validation import ValidationContext
from torchtitan.experiments.cuda_graph_campaign.observer import CampaignTimingObserver
from torchtitan.trainer import Trainer


def _callback_manager(callbacks):
    return CallbackManager(callbacks, model_parts=[], sharding_backend=None)


def _trainer(callbacks):
    trainer = cast(Any, Trainer.__new__(Trainer))
    trainer.device = torch.device("cpu")
    trainer.step = 1
    trainer.ntokens_seen = torch.zeros((), dtype=torch.int64)
    trainer.gradient_accumulation_steps = 4
    trainer.model_parts = []
    trainer.optimizers = SimpleNamespace(
        optimizers=[],
        zero_grad=lambda: None,
        step=lambda: None,
    )
    trainer.lr_schedulers = SimpleNamespace(
        schedulers=[SimpleNamespace(get_last_lr=lambda: [1.0])],
        step=lambda: None,
    )
    trainer.checkpointer = SimpleNamespace(
        last_saved_path=None,
        maybe_wait_for_staging=lambda: None,
    )
    trainer.metrics_processor = SimpleNamespace(
        should_log=lambda step: False,
        ntokens_since_last_log=0,
        data_loading_times=[],
        log_extra_metrics=lambda step, metrics: None,
    )
    trainer.callbacks = callbacks
    trainer.loss_fn = SimpleNamespace(components=lambda: {})
    trainer.parallel_dims = SimpleNamespace(
        dp_enabled=False,
        dp_cp_enabled=False,
        ep_enabled=False,
        pp_enabled=False,
        get_optional_mesh=lambda name: None,
    )
    trainer.aux_loss = SimpleNamespace(begin_step=lambda: None)
    trainer.config = SimpleNamespace(
        training=SimpleNamespace(max_norm=1.0, seq_len=2),
        parallelism=SimpleNamespace(),
    )
    return trainer


def _native_config_data(callbacks):
    return {
        "dataloader": {"type": "hf_text"},
        "loss": {"type": "cross_entropy"},
        "callbacks": callbacks,
    }


def test_config_round_trips_and_callback_is_optional(monkeypatch):
    calls = []
    monkeypatch.setattr(
        torch.compiler, "cudagraph_mark_step_begin", lambda: calls.append(1)
    )

    callback_config = decode(
        CallbackConfig,
        {"type": "campaign_cudagraph_step_boundary"},
    )
    assert isinstance(callback_config, CampaignCUDAGraphStepBoundaryConfig)
    assert decode(CallbackConfig, to_plain(callback_config)) == callback_config

    manager = _callback_manager({})
    assert manager.find(type(callback_config).owner()) is None
    manager.emit(Event.batch_start, _trainer(manager))
    assert calls == []


def test_native_and_continuation_trainer_configs_accept_boundary_without_retyping():
    boundary = {"step_boundary": {"type": "campaign_cudagraph_step_boundary"}}
    native = decode(TrainerConfig, _native_config_data(boundary))
    assert isinstance(
        native.callbacks["step_boundary"], CampaignCUDAGraphStepBoundaryConfig
    )
    assert not any(
        isinstance(callback, CampaignRNGContinuationConfig)
        for callback in native.callbacks.values()
    )

    continuation = decode(
        CampaignTrainerConfig,
        _native_config_data(
            {
                "rng_continuation": {"type": "campaign_rng_continuation"},
                **boundary,
            }
        ),
    )
    assert isinstance(
        continuation.callbacks["step_boundary"], CampaignCUDAGraphStepBoundaryConfig
    )
    assert isinstance(
        continuation.callbacks["rng_continuation"], CampaignRNGContinuationConfig
    )
    assert not [
        issue
        for issue in continuation.validate(ValidationContext(stage="pre_submit"))
        if "campaign_rng_continuation" in issue.message
    ]


def test_duplicate_boundary_is_rejected_by_callback_dependency_validation():
    first = CampaignCUDAGraphStepBoundaryConfig()
    second = CampaignCUDAGraphStepBoundaryConfig()

    issues = second.validate_dependencies([first])

    assert len(issues) == 1
    assert "only once" in issues[0].message

    native = decode(
        TrainerConfig,
        _native_config_data({"first": to_plain(first), "second": to_plain(second)}),
    )
    issues = native.validate(ValidationContext(stage="pre_submit"))
    assert any(
        issue.path == "callbacks.second" and "only once" in issue.message
        for issue in issues
    )


def test_fit_start_rejects_pipeline_parallelism_without_marking(monkeypatch):
    calls = []
    monkeypatch.setattr(
        torch.compiler, "cudagraph_mark_step_begin", lambda: calls.append(1)
    )
    callback = CampaignCUDAGraphStepBoundaryConfig().build()

    callback.fit_start(SimpleNamespace(parallel_dims=SimpleNamespace(pp_enabled=False)))
    assert calls == []
    with pytest.raises(ValueError, match="pipeline parallel degree 1"):
        callback.fit_start(
            SimpleNamespace(parallel_dims=SimpleNamespace(pp_enabled=True))
        )
    assert calls == []


def test_native_train_step_marks_once_before_four_accumulation_microbatches(
    monkeypatch,
):
    events = []
    marker_calls = []

    def mark_step_begin():
        marker_calls.append(None)
        events.append("mark")

    monkeypatch.setattr(torch.compiler, "cudagraph_mark_step_begin", mark_step_begin)
    original_timing_batch_start = CampaignTimingObserver.batch_start

    def timing_batch_start(observer, state):
        events.append("timing")
        original_timing_batch_start(observer, state)

    monkeypatch.setattr(CampaignTimingObserver, "batch_start", timing_batch_start)
    callbacks = _callback_manager(
        {
            "campaign_timing": CampaignTimingObserverConfig(
                start_step=100,
                update_count=1,
            ),
            "campaign_cudagraph_step_boundary": CampaignCUDAGraphStepBoundaryConfig(),
        }
    )
    trainer = _trainer(callbacks)
    trainer.optimizers.zero_grad = lambda: events.append("zero_grad")
    trainer.optimizers.step = lambda: events.append("optimizer")
    trainer.forward_backward_step = lambda **kwargs: (
        events.append("mask_forward_backward") or torch.tensor(0.0)
    )
    monkeypatch.setattr(
        trainer_module.dist_utils,
        "clip_grad_norm_",
        lambda *args, **kwargs: events.append("clip") or torch.tensor(0.0),
    )

    def batches():
        for _ in range(4):
            events.append("fetch")
            labels = torch.ones((1, 2), dtype=torch.int64)
            yield {"input": torch.ones_like(labels)}, labels

    python_rng_before = random.getstate()
    numpy_rng_before = np.random.get_state()
    torch_rng_before = torch.random.get_rng_state().clone()
    for step in (1, 2):
        trainer.step = step
        trainer.train_step(iter(batches()))

    one_update = [
        "timing",
        "mark",
        "zero_grad",
        *("fetch",) * 4,
        *("mask_forward_backward",) * 4,
        "clip",
        "optimizer",
    ]
    assert events == one_update * 2
    assert len(marker_calls) == 2
    assert random.getstate() == python_rng_before
    numpy_rng_after = np.random.get_state()
    assert numpy_rng_after[0] == numpy_rng_before[0]
    assert np.array_equal(numpy_rng_after[1], numpy_rng_before[1])
    assert numpy_rng_after[2:] == numpy_rng_before[2:]
    assert torch.equal(torch.random.get_rng_state(), torch_rng_before)


def test_marker_exception_propagates_from_callback_manager(monkeypatch):
    def fail():
        raise RuntimeError("compiler marker failed")

    monkeypatch.setattr(torch.compiler, "cudagraph_mark_step_begin", fail)
    manager = _callback_manager(
        {"step_boundary": CampaignCUDAGraphStepBoundaryConfig()}
    )

    with pytest.raises(RuntimeError, match="compiler marker failed"):
        manager.emit(Event.batch_start, _trainer(manager))
