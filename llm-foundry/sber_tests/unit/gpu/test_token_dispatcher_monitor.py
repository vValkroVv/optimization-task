from types import SimpleNamespace

import pytest
import torch

from llmfoundry.callbacks import token_dispatcher_monitor as monitor_module


class _SupportedModel:
    """Identify a model accepted by the callback's runtime guard."""


class _State:
    """Provide the state fields consumed by monitor fit and batch hooks."""

    def __init__(self) -> None:
        """Initialize a supported model and the first logging timestamp."""
        self.model = _SupportedModel()
        self.timestamp = SimpleNamespace(batch=SimpleNamespace(value=1))

    def _get_state_metadata(self) -> dict[str, int]:
        """Return a two-microbatch training shape."""
        return {
            "train_dataloader_batch_size": 4,
            "device_train_microbatch_size": 2,
        }


class _Logger:
    """Capture metric dictionaries emitted by the production callback."""

    def __init__(self) -> None:
        """Initialize an empty metric history."""
        self.metrics: list[dict[str, object]] = []

    def log_metrics(self, metrics: dict[str, object]) -> None:
        """Retain one emitted metric mapping."""
        self.metrics.append(metrics)


class _Dispatcher:
    """Expose deterministic CUDA counters through the real callback hooks."""

    def __init__(self, values: tuple[list[int], ...], *, capacity: bool = True) -> None:
        """Initialize monitor values and lifecycle counters."""
        self.config = SimpleNamespace(
            num_routed_experts=4,
            dispatcher_balancing_strategy="fixed_capacity" if capacity else None,
            drop_by_group_capacity=False,
        )
        self.values = tuple(torch.tensor(value, device="cuda") for value in values)
        self.initialized = 0
        self.activated = 0
        self.deactivated = 0

    def _init_dispatcher_monitor(self) -> None:
        """Record monitor-buffer initialization."""
        self.initialized += 1

    def _activate_dispatcher_monitor(self) -> None:
        """Record monitor activation."""
        self.activated += 1

    def _deactivate_dispatcher_monitor(self) -> None:
        """Record monitor deactivation."""
        self.deactivated += 1

    def get_dispatcher_monitor(self) -> tuple[torch.Tensor, ...]:
        """Return local capacity and token counts in production slot order."""
        return self.values


def _dispatchers(*, capacity: bool = True) -> list[_Dispatcher]:
    """Return two layers with distinct loads for layer-maximum assertions."""
    return [
        _Dispatcher(
            ([3, 3, 3, 3], [4, 2, 0, 0], [3, 1, 0, 0], [2, 1, 0, 0]),
            capacity=capacity,
        ),
        _Dispatcher(
            ([3, 3, 3, 3], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]),
            capacity=capacity,
        ),
    ]


def _configure_groups(
    monkeypatch: pytest.MonkeyPatch,
    *,
    ep_group: object | None = None,
    tp_sp_group: object | None = None,
    ep_fsdp_group: object | None = None,
    tp_sp_group_size: int | None = None,
) -> None:
    """Configure process-group lookups used by training and evaluation hooks."""
    monkeypatch.setattr(monitor_module.dist, "get_ep_group_size", lambda: None)
    monkeypatch.setattr(monitor_module.dist, "get_ep_fsdp_group_size", lambda: None)
    monkeypatch.setattr(monitor_module.dist, "get_tp_sp_group_size", lambda: tp_sp_group_size)
    monkeypatch.setattr(monitor_module.dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(monitor_module.dist, "get_ep_group", lambda: ep_group)
    monkeypatch.setattr(monitor_module.dist, "get_tp_sp_group", lambda: tp_sp_group)
    monkeypatch.setattr(monitor_module.dist, "get_ep_fsdp_group", lambda: ep_fsdp_group)


def _start_callback(
    monkeypatch: pytest.MonkeyPatch,
    dispatchers: list[_Dispatcher],
    *,
    activate_ep_rank_metrics: bool,
) -> tuple[monitor_module.TokenDispatcherMonitor, _State, _Logger]:
    """Run the real fit and two-microbatch hooks for deterministic dispatchers."""
    monkeypatch.setattr(
        monitor_module,
        "get_moe_token_dispatchers",
        lambda state: (dispatchers, ["layer_0", "layer_1"]),
    )
    callback = monitor_module.TokenDispatcherMonitor(
        batch_log_interval=1,
        activate_ep_rank_metrics=activate_ep_rank_metrics,
    )
    callback.supported_models = [_SupportedModel]
    state = _State()
    logger = _Logger()
    callback.fit_start(state, logger)
    callback.after_forward(state, logger)
    callback.after_forward(state, logger)
    return callback, state, logger


@pytest.mark.unit
@pytest.mark.requires_gpu
def test_absent_groups_emit_local_training_and_evaluation_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing process groups must avoid collectives and retain local values."""
    _configure_groups(monkeypatch)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    def fail_reduce(*args, **kwargs) -> None:
        """Fail if a local-only hook attempts any collective."""
        raise AssertionError("absent groups must not reach all_reduce")

    monkeypatch.setattr(monitor_module.dist, "all_reduce", fail_reduce)
    dispatchers = _dispatchers()
    callback, state, logger = _start_callback(
        monkeypatch,
        dispatchers,
        activate_ep_rank_metrics=False,
    )

    assert callback.ep_group_size == 1
    assert callback.ep_fsdp_group_size == 1
    assert callback.group_size == 4
    assert all(dispatcher.initialized == 1 for dispatcher in dispatchers)
    assert all(dispatcher.activated == 1 for dispatcher in dispatchers)

    callback.eval_after_forward(state, logger)
    assert callback.eval_token_drop_step_inner_counter == 1

    callback.eval_end(state, logger)
    eval_metrics = logger.metrics[-1]
    assert eval_metrics["dispatcher_sum_metrics/eval_num_drop_steps"] == 1
    assert callback.eval_token_drop_step_inner_counter == 0

    callback.batch_end(state, logger)
    metrics = logger.metrics[-1]
    torch.testing.assert_close(
        metrics["dispatcher_sum_metrics/processed_batch_tokens"],
        torch.tensor(10.0, device="cuda"),
    )
    torch.testing.assert_close(
        metrics["dispatcher_sum_metrics/processed_batch_tokens_no_pad"],
        torch.tensor(8.0, device="cuda"),
    )
    torch.testing.assert_close(
        metrics["dispatcher_sum_metrics/processed_batch_tokens_no_pad_no_prefix"],
        torch.tensor(7.0, device="cuda"),
    )
    torch.testing.assert_close(
        metrics["dispatcher_sum_metrics/experts_max_vio_batch_avg"],
        torch.tensor(5.0 / 6.0, device="cuda"),
    )
    torch.testing.assert_close(
        metrics["dispatcher_sum_metrics/overloaded_expert_ratio"],
        torch.tensor(0.125, device="cuda"),
    )
    torch.testing.assert_close(
        metrics["dispatcher_sum_metrics/token_drop_level"],
        torch.tensor(1.0 / 24.0, device="cuda"),
    )
    torch.testing.assert_close(
        metrics["dispatcher_sum_metrics/empty_token_level"],
        torch.tensor(0.625, device="cuda"),
    )
    assert metrics["dispatcher_sum_metrics/num_drop_steps"].item() == 1
    assert not [key for key in metrics if "ep_rank" in key]
    assert all(dispatcher.deactivated == 2 for dispatcher in dispatchers)


@pytest.mark.unit
@pytest.mark.requires_gpu
def test_ep_rank_metrics_reduce_two_layer_maxima_once_after_aggregation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Enabled EP-rank metrics must MAX-reduce only final local maxima."""
    ep_fsdp_group = object()
    _configure_groups(monkeypatch, ep_fsdp_group=ep_fsdp_group)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    calls: list[tuple[str, object]] = []

    def record_reduce(
        tensor: torch.Tensor,
        reduce_operation: str,
        group: object,
        async_op: bool,
    ) -> None:
        """Record reductions without changing the single-rank reference values."""
        assert async_op is False
        calls.append((reduce_operation, group))

    monkeypatch.setattr(monitor_module.dist, "all_reduce", record_reduce)
    callback, state, logger = _start_callback(
        monkeypatch,
        _dispatchers(),
        activate_ep_rank_metrics=True,
    )

    callback.batch_end(state, logger)
    metrics = logger.metrics[-1]
    assert [call for call in calls if call[0] == "MAX"] == [
        ("MAX", ep_fsdp_group),
        ("MAX", ep_fsdp_group),
    ]
    assert all(group is ep_fsdp_group for _, group in calls)
    assert "dispatcher_sum_metrics/ep_ranks_max_vio_microbatch_avg" in metrics
    assert "dispatcher_sum_metrics/step_max_ep_rank_load" in metrics
    assert metrics["dispatcher_sum_metrics/step_max_expert_load"].item() == 4
    assert metrics["dispatcher_sum_metrics/step_max_ep_rank_load"].item() == 6
    torch.testing.assert_close(
        metrics["dispatcher_sum_metrics/max_vio_layermax"],
        torch.tensor(5.0 / 3.0, device="cuda"),
    )


@pytest.mark.unit
@pytest.mark.requires_gpu
def test_disabled_capacity_omits_capacity_metric_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dispatcher without capacity balancing must not publish drop metrics."""
    _configure_groups(monkeypatch)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(
        monitor_module.dist,
        "all_reduce",
        lambda *args, **kwargs: pytest.fail("local-only callback reduced"),
    )
    callback, state, logger = _start_callback(
        monkeypatch,
        _dispatchers(capacity=False),
        activate_ep_rank_metrics=False,
    )

    callback.batch_end(state, logger)
    metrics = logger.metrics[-1]
    assert "dispatcher_sum_metrics/overloaded_expert_ratio" not in metrics
    assert "dispatcher_sum_metrics/token_drop_level" not in metrics
    assert "dispatcher_sum_metrics/empty_token_level" not in metrics
    assert "dispatcher_sum_metrics/num_drop_steps" not in metrics


@pytest.mark.unit
@pytest.mark.requires_gpu
def test_disabled_ep_rank_metrics_ignore_available_ep_fsdp_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The disabled flag must suppress EP-rank keys and MAX reductions."""
    ep_fsdp_group = object()
    _configure_groups(monkeypatch, ep_fsdp_group=ep_fsdp_group)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    reductions: list[str] = []

    def record_reduce(
        tensor: torch.Tensor,
        reduce_operation: str,
        group: object,
        async_op: bool,
    ) -> None:
        """Record legitimate SUMs while exposing any forbidden MAX."""
        del tensor
        assert group is ep_fsdp_group
        assert async_op is False
        reductions.append(reduce_operation)

    monkeypatch.setattr(monitor_module.dist, "all_reduce", record_reduce)
    callback, state, logger = _start_callback(
        monkeypatch,
        _dispatchers(),
        activate_ep_rank_metrics=False,
    )

    callback.batch_end(state, logger)

    assert not [key for key in logger.metrics[-1] if "ep_rank" in key]
    assert reductions.count("MAX") == 1


@pytest.mark.unit
@pytest.mark.requires_gpu
def test_tp_sp_group_sums_sequence_shards_for_every_token_histogram(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Under sp every token histogram, padded or not, counts the whole microbatch."""
    tp_sp_group = object()
    _configure_groups(monkeypatch, tp_sp_group=tp_sp_group, tp_sp_group_size=2)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    calls: list[tuple[str, object]] = []

    def two_equal_shards(
        tensor: torch.Tensor,
        reduce_operation: str,
        group: object,
        async_op: bool,
    ) -> None:
        """Emulate a SUM over two sequence-parallel ranks holding identical counts."""
        assert async_op is False
        calls.append((reduce_operation, group))
        tensor.mul_(2)

    monkeypatch.setattr(monitor_module.dist, "all_reduce", two_equal_shards)
    callback, state, logger = _start_callback(
        monkeypatch,
        _dispatchers(),
        activate_ep_rank_metrics=False,
    )

    callback.batch_end(state, logger)
    metrics = logger.metrics[-1]
    # one reduce per layer for each of the three histograms
    assert calls == [("SUM", tp_sp_group)] * 6
    assert metrics["dispatcher_sum_metrics/step_max_expert_load"].item() == 8
    torch.testing.assert_close(
        metrics["dispatcher_sum_metrics/experts_max_vio_microbatch_avg"],
        torch.tensor(5.0 / 6.0, device="cuda"),
    )
    torch.testing.assert_close(
        metrics["dispatcher_sum_metrics/processed_batch_tokens"],
        torch.tensor(20.0, device="cuda"),
    )
    torch.testing.assert_close(
        metrics["dispatcher_sum_metrics/processed_batch_tokens_no_pad"],
        torch.tensor(16.0, device="cuda"),
    )
    torch.testing.assert_close(
        metrics["dispatcher_sum_metrics/processed_batch_tokens_no_pad_no_prefix"],
        torch.tensor(14.0, device="cuda"),
    )
    torch.testing.assert_close(
        metrics["dispatcher_sum_metrics/experts_max_vio_batch_avg"],
        torch.tensor(5.0 / 6.0, device="cuda"),
    )


@pytest.mark.unit
@pytest.mark.requires_gpu
def test_crossing_ep_and_tp_sp_groups_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The microbatch divisor is only defined when one group contains the other."""
    ep_group, tp_sp_group = object(), object()
    _configure_groups(
        monkeypatch,
        ep_group=ep_group,
        tp_sp_group=tp_sp_group,
        tp_sp_group_size=4,
    )
    ranks = {ep_group: [0, 1, 2, 3], tp_sp_group: [2, 3, 4, 5]}
    monkeypatch.setattr(
        torch.distributed,
        "get_process_group_ranks",
        lambda group: ranks[group],
    )

    with pytest.raises(AssertionError, match="nested"):
        _start_callback(monkeypatch, _dispatchers(), activate_ep_rank_metrics=False)

    ranks[tp_sp_group] = [0, 1]
    _start_callback(monkeypatch, _dispatchers(), activate_ep_rank_metrics=False)
