"""The metric sampler captures every capture_interval-th step: on the others no hook touches the recorder and the
observers the model and the monitors call on their own are detached; the activation, token-loss and expert
families each leave their slots out of the layout when switched off."""

from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit

pytest.importorskip("composer", reason="Requires the CPU training runtime")

import llmfoundry.callbacks.metric_sampler as metric_sampler_module  # noqa: E402
from llmfoundry.callbacks.metric_sampler import ActivationDumpConfigError, MetricSampler  # noqa: E402

_POINTS = (
    "residual", "attn_out", "mlp_out", "shared_expert_out", "attn_residual_contribution", "mlp_residual_contribution",
    "swiglu_in", "down_in",
)


def _metric_names(callback: MetricSampler) -> set[str]:
    callback.microbatches = 1
    requests = callback._layout_requests(
        experts=4, routed_layers={0: 0}, mtp_layers=[], streams=1, pre_reduce_keys=set(),
        activation_layers={point: [0] for point in _POINTS}, dumps_enabled=False,
    )
    return {metric.name for metric, _, _, _ in requests}


@pytest.mark.parametrize("family, prefixes", [
    ("activations", ("act/",)),
    ("token_loss", ("loss/token_", "loss/stream_", "loss/position_", "loss/sample_", "loss/worst_", "loss/nonfinite")),
    ("experts", ("router/", "expert_loss/")),
])
def test_a_family_switched_off_leaves_its_slots_out_and_the_others_in(tmp_path, family, prefixes):
    every = _metric_names(MetricSampler(str(tmp_path), exact_quantiles=True))
    without = _metric_names(MetricSampler(str(tmp_path), exact_quantiles=True, **{family: False}))

    left_out = every - without
    assert left_out and all(name.startswith(prefixes) for name in left_out)
    assert not any(name.startswith(prefixes) for name in without)
    assert {"loss/total", "time/forward_gpu_s", "memory/allocated_gb"} <= without


def _observed_sampler(tmp_path, rank_local_monitors: bool = False, **kwargs) -> MetricSampler:
    callback = MetricSampler(str(tmp_path), rank_local_monitors=rank_local_monitors, **kwargs)
    callback.causal_lm = SimpleNamespace(loss_observer=None, topk_observer=None)
    callback._routing_observers = [(SimpleNamespace(routing_observer=None), "routing")]
    callback._loss_observer = "loss"
    callback._pre_reduce_guards = [(SimpleNamespace(pre_reduce_observer=None), "pre_reduce")]
    if rank_local_monitors:
        monitors = callback.monitor_capture
        monitors._trainer_state = SimpleNamespace(rank_local_training_observer=None)
        for name in ("_lr_monitor", "_memory_monitor", "_stage_time_monitor", "_weight_monitor", "_optimizer_monitor",
                     "_training_monitor", "_optimizer_health_monitor", "_update_weight_monitor"):
            setattr(monitors, name, SimpleNamespace(rank_local_observer=None))
        monitors._speed_monitor = SimpleNamespace(rank_local_flops_observer=None)
        monitors._topk_monitor = SimpleNamespace(_enabled=True)
        monitors._balance_observers = [(SimpleNamespace(balance_observer=None), "balance")]
        monitors._expert_activation_observers = [(SimpleNamespace(rank_local_activation_observer=None), "experts")]
    return callback


def _attached(callback: MetricSampler) -> list[object]:
    attached = [
        callback._routing_observers[0][0].routing_observer,
        callback.causal_lm.loss_observer,
        callback._pre_reduce_guards[0][0].pre_reduce_observer,
    ]
    if callback.rank_local_monitors:
        monitors = callback.monitor_capture
        attached += [
            monitors._trainer_state.rank_local_training_observer,
            monitors._lr_monitor.rank_local_observer,
            monitors._memory_monitor.rank_local_observer,
            monitors._speed_monitor.rank_local_flops_observer,
            monitors._stage_time_monitor.rank_local_observer,
            monitors._weight_monitor.rank_local_observer,
            monitors._balance_observers[0][0].balance_observer,
            monitors._expert_activation_observers[0][0].rank_local_activation_observer,
            monitors._optimizer_monitor.rank_local_observer,
            monitors._training_monitor.rank_local_observer,
            monitors._optimizer_health_monitor.rank_local_observer,
            monitors._update_weight_monitor.rank_local_observer,
            callback.causal_lm.topk_observer,
        ]
    return attached


@pytest.mark.parametrize("rank_local_monitors", [False, True])
def test_the_observers_the_model_and_the_monitors_call_are_attached_only_for_captured_steps(tmp_path, rank_local_monitors):
    callback = _observed_sampler(tmp_path, rank_local_monitors)

    callback._set_step_observers(True)
    attached = _attached(callback)
    assert attached[:3] == ["routing", "loss", "pre_reduce"] and None not in attached

    callback._set_step_observers(False)
    assert set(_attached(callback)) == {None}

    callback.causal_lm.loss_observer = "set elsewhere"
    callback._set_step_observers(False)
    assert callback.causal_lm.loss_observer == "set elsewhere"


def _state(batches_done: int) -> SimpleNamespace:
    return SimpleNamespace(timestamp=SimpleNamespace(batch=SimpleNamespace(value=batches_done)))


def test_a_step_that_is_not_captured_touches_nothing_but_keeps_the_loop_time_of_its_successor(tmp_path, monkeypatch):
    callback = _observed_sampler(tmp_path, capture_interval=4)
    callback._set_step_observers(True)
    clock = iter([100.0, 107.5])
    monkeypatch.setattr(metric_sampler_module, "time", SimpleNamespace(perf_counter=lambda: next(clock)))

    # Step 3 is not captured: no hook reaches the recorder, which is absent here.
    callback.batch_start(_state(2), None)
    for hook in (callback.before_forward, callback.after_forward, callback.after_loss, callback.before_backward,
                 callback.after_backward, callback.after_train_batch):
        hook(_state(2), None)
    assert callback._capturing is False and callback.step_started == 100.0
    # Step 3 ends: step 4 is captured, so the observers stay attached for it.
    callback.batch_end(_state(3), None)
    assert None not in _attached(callback)

    # Step 5 is not captured: its observers come off when step 4 ends, which records its step through _batch_end.
    recorded = []
    monkeypatch.setattr(callback, "_batch_end", lambda state, logger: recorded.append(state.timestamp.batch.value))
    callback._capturing = True
    callback.batch_end(_state(4), None)
    assert recorded == [4] and set(_attached(callback)) == {None}

    callback.batch_start(_state(4), None)
    assert callback._capturing is False and callback.step_started == 107.5


def test_a_disabled_sampler_does_not_attach_observers_again(tmp_path):
    callback = _observed_sampler(tmp_path, capture_interval=2)
    callback._set_step_observers(False)
    callback.disabled = True

    callback.batch_end(_state(1), None)

    assert set(_attached(callback)) == {None}


@pytest.mark.parametrize("kwargs, error, message", [
    ({"capture_interval": 0}, ValueError, "must be positive"),
    ({"capture_interval": 3}, ValueError, "distribution_interval must be a multiple of capture_interval"),
    ({"capture_interval": 4, "activation_dump": {"interval": 6, "points": ["residual"], "layers": [0]}},
     ActivationDumpConfigError, "multiple of capture_interval"),
    ({"activations": False, "activation_dump": {"interval": 4, "points": ["residual"], "layers": [0]}},
     ActivationDumpConfigError, "dumps need activations"),
])
def test_capture_settings_that_cannot_work_together_fail_at_fit_start(tmp_path, kwargs, error, message):
    callback = MetricSampler(str(tmp_path), **kwargs)

    with pytest.raises(error, match=message):
        callback.fit_start(None, None)
