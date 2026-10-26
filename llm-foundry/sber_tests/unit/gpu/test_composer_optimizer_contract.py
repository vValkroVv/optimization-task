from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import composer
from composer.callbacks import training_metrics_monitor
from composer.optim.decoupled_weight_decay import DecoupledAdamW, DecoupledFusedAdamW, step_counts


@pytest.fixture(scope="session", autouse=True)
def assert_bundled_composer_checkout() -> None:
    """Require tests to import the composer checkout bundled with llm-foundry.

    The exact composer revision is intentionally not pinned: this suite
    passes against any llm-foundry-bundled composer revision that preserves
    the ``DecoupledAdamW`` / ``TrainingMetricsMonitor`` API it exercises.
    """
    composer_root = Path(composer.__file__).resolve().parents[1]
    assert "contrib/composer" in str(composer_root)


def _disable_real_collective(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the optimizer's rank synchronization with a local no-op."""
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda tensor: None)


@pytest.mark.unit
@pytest.mark.gpu
@pytest.mark.parametrize("optimizer_class", [DecoupledAdamW, DecoupledFusedAdamW])
def test_finite_gradient_updates_parameter_state_and_counters(
    monkeypatch: pytest.MonkeyPatch,
    optimizer_class: type[DecoupledAdamW],
) -> None:
    """A finite group must update weights, moments, steps, and total updates."""
    _disable_real_collective(monkeypatch)
    parameter = torch.nn.Parameter(torch.tensor([1.0, -1.0], device="cuda"))
    optimizer = optimizer_class([parameter], lr=0.1, weight_decay=0.01)
    parameter.grad = torch.tensor([0.5, -0.25], device="cuda")
    before = parameter.detach().clone()

    optimizer.step()

    assert not torch.equal(parameter, before)
    assert optimizer.state[parameter]["step"].item() == 1
    assert torch.count_nonzero(optimizer.state[parameter]["exp_avg"]).item() == 2
    assert torch.count_nonzero(optimizer.state[parameter]["exp_avg_sq"]).item() == 2
    assert optimizer.total_updates == 1
    assert optimizer.skip_updates == 0


@pytest.mark.unit
@pytest.mark.gpu
@pytest.mark.parametrize("optimizer_class", [DecoupledAdamW, DecoupledFusedAdamW])
@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_bad_fresh_gradient_creates_no_state(
    monkeypatch: pytest.MonkeyPatch,
    optimizer_class: type[DecoupledAdamW],
    bad_value: float,
) -> None:
    """A skipped first step must not decay weights or create optimizer state."""
    _disable_real_collective(monkeypatch)
    parameter = torch.nn.Parameter(torch.tensor([1.0], device="cuda"))
    optimizer = optimizer_class(
        [parameter],
        lr=0.1,
        weight_decay=0.1,
        amsgrad=True,
    )
    parameter.grad = torch.tensor([bad_value], device="cuda")
    before = parameter.detach().clone()

    optimizer.step()

    torch.testing.assert_close(parameter, before)
    assert parameter not in optimizer.state
    assert optimizer.total_updates == 1
    assert optimizer.skip_updates == 1


@pytest.mark.unit
@pytest.mark.gpu
@pytest.mark.parametrize("optimizer_class", [DecoupledAdamW, DecoupledFusedAdamW])
@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_bad_existing_gradient_mutates_no_optimizer_state(
    monkeypatch: pytest.MonkeyPatch,
    optimizer_class: type[DecoupledAdamW],
    bad_value: float,
) -> None:
    """A skipped later step must preserve parameters, moments, AMSGrad, and step."""
    _disable_real_collective(monkeypatch)
    parameter = torch.nn.Parameter(torch.tensor([1.0], device="cuda"))
    optimizer = optimizer_class(
        [parameter],
        lr=0.1,
        weight_decay=0.1,
        amsgrad=True,
    )
    parameter.grad = torch.tensor([0.5], device="cuda")
    optimizer.step()
    before_parameter = parameter.detach().clone()
    before_state = {
        key: value.detach().clone() for key, value in optimizer.state[parameter].items()
    }
    parameter.grad = torch.tensor([bad_value], device="cuda")

    optimizer.step()

    torch.testing.assert_close(parameter, before_parameter)
    for key, expected in before_state.items():
        torch.testing.assert_close(optimizer.state[parameter][key], expected)
    assert optimizer.total_updates == 2
    assert optimizer.skip_updates == 1


@pytest.mark.unit
@pytest.mark.gpu
@pytest.mark.parametrize("optimizer_class", [DecoupledAdamW, DecoupledFusedAdamW])
def test_finite_and_bad_groups_are_accounted_for_independently(
    monkeypatch: pytest.MonkeyPatch,
    optimizer_class: type[DecoupledAdamW],
) -> None:
    """A bad group must not prevent a separate finite group from updating."""
    _disable_real_collective(monkeypatch)
    finite = torch.nn.Parameter(torch.tensor([1.0], device="cuda"))
    bad = torch.nn.Parameter(torch.tensor([2.0], device="cuda"))
    optimizer = optimizer_class(
        [{"params": [finite]}, {"params": [bad]}],
        lr=0.1,
    )
    finite.grad = torch.tensor([0.5], device="cuda")
    bad.grad = torch.tensor([float("inf")], device="cuda")
    before_finite = finite.detach().clone()
    before_bad = bad.detach().clone()

    optimizer.step()

    assert not torch.equal(finite, before_finite)
    torch.testing.assert_close(bad, before_bad)
    assert finite in optimizer.state
    assert bad not in optimizer.state
    assert optimizer.total_updates == 2
    assert optimizer.skip_updates == 1


@pytest.mark.unit
@pytest.mark.gpu
@pytest.mark.parametrize("optimizer_class", [DecoupledAdamW, DecoupledFusedAdamW])
def test_remote_bad_rank_forces_local_skip(
    monkeypatch: pytest.MonkeyPatch,
    optimizer_class: type[DecoupledAdamW],
) -> None:
    """A remote non-finite gradient signal must skip an otherwise finite group."""
    parameter = torch.nn.Parameter(torch.tensor([1.0], device="cuda"))
    optimizer = optimizer_class([parameter], lr=0.1)
    parameter.grad = torch.tensor([0.5], device="cuda")
    before = parameter.detach().clone()

    def report_remote_failure(tensor: torch.Tensor) -> None:
        """Model one remote rank reporting a non-finite gradient."""
        tensor.fill_(1)

    monkeypatch.setattr(torch.distributed, "all_reduce", report_remote_failure)
    optimizer.step()

    torch.testing.assert_close(parameter, before)
    assert parameter not in optimizer.state
    assert optimizer.skip_updates == 1


@pytest.mark.unit
@pytest.mark.gpu
def test_training_monitor_does_not_reduce_skip_counters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Optimizer skip counters are already global and must not be reduced again."""
    optimizer = SimpleNamespace(skip_updates=2, total_updates=4)
    state = SimpleNamespace(
        timestamp=SimpleNamespace(batch=SimpleNamespace(value=10)),
        optimizers=[optimizer],
    )
    logged: list[dict[str, float]] = []
    logger = SimpleNamespace(log_metrics=lambda metrics: logged.append(metrics))
    monkeypatch.setattr(training_metrics_monitor.dist, "get_world_size", lambda: 2)

    def fail_reduce(*args, **kwargs) -> None:
        """Fail if already-global skip counters are reduced."""
        raise AssertionError("skip counters must not be reduced")

    monkeypatch.setattr(training_metrics_monitor.dist, "all_reduce", fail_reduce)
    monkeypatch.setattr(
        training_metrics_monitor.composer.utils.zero_nan_grads,
        "NUM_MODULE_GARDS_ZEROED_OUT",
        None,
    )
    monkeypatch.setattr(
        training_metrics_monitor.composer.utils.zero_nan_grads,
        "NUM_MODULE_GARDS_PROCESSED",
        None,
    )

    training_metrics_monitor.TrainingMetricsMonitor().batch_end(state, logger)

    assert logged == [
        {
            "training/skip_updates_total": 2.0,
            "training/skip_updates_ratio": 0.5,
        },
    ]
    assert optimizer.skip_updates == 0
    assert optimizer.total_updates == 0


@pytest.mark.unit
@pytest.mark.gpu
@pytest.mark.parametrize("complete_counters", [False, True])
def test_training_monitor_reduces_complete_zeroed_gradient_counters_and_resets(
    monkeypatch: pytest.MonkeyPatch,
    complete_counters: bool,
) -> None:
    """Zeroed-gradient counters reduce only as a complete pair and always reset."""
    optimizer = SimpleNamespace(skip_updates=1, total_updates=4)
    state = SimpleNamespace(
        timestamp=SimpleNamespace(batch=SimpleNamespace(value=10)),
        optimizers=[optimizer],
    )
    logged: list[dict[str, float]] = []
    logger = SimpleNamespace(log_metrics=lambda metrics: logged.append(metrics))
    monkeypatch.setattr(training_metrics_monitor.dist, "get_world_size", lambda: 2)
    zero_nan_grads = training_metrics_monitor.composer.utils.zero_nan_grads
    monkeypatch.setattr(
        zero_nan_grads,
        "NUM_MODULE_GARDS_ZEROED_OUT",
        torch.tensor(2.0, device="cuda"),
    )
    monkeypatch.setattr(
        zero_nan_grads,
        "NUM_MODULE_GARDS_PROCESSED",
        torch.tensor(4.0, device="cuda") if complete_counters else None,
    )
    reductions: list[torch.Tensor] = []

    def reduce_complete_pair(
        tensor: torch.Tensor,
        reduce_operation: str,
    ) -> None:
        """Model a second rank with identical zeroed-gradient counters."""
        assert reduce_operation == "SUM"
        reductions.append(tensor.detach().clone())
        tensor.mul_(2)

    monkeypatch.setattr(
        training_metrics_monitor.dist, "all_reduce", reduce_complete_pair
    )

    training_metrics_monitor.TrainingMetricsMonitor().batch_end(state, logger)

    metrics = logged[-1]
    if complete_counters:
        assert len(reductions) == 1
        assert metrics["training/zeroed_module_grads"] == 4.0
        assert metrics["training/zeroed_module_grads_ratio"] == 0.5
    else:
        assert reductions == []
        assert "training/zeroed_module_grads" not in metrics
        assert "training/zeroed_module_grads_ratio" not in metrics
    assert zero_nan_grads.NUM_MODULE_GARDS_ZEROED_OUT is None
    assert zero_nan_grads.NUM_MODULE_GARDS_PROCESSED is None
    assert optimizer.skip_updates == 0
    assert optimizer.total_updates == 0


@pytest.mark.unit
@pytest.mark.gpu
@pytest.mark.parametrize("optimizer_class", [DecoupledAdamW, DecoupledFusedAdamW])
@pytest.mark.parametrize("read_before_step", [False, True])
def test_host_step_is_the_device_step_through_skips_and_state_loads(
    monkeypatch: pytest.MonkeyPatch,
    optimizer_class: type[DecoupledAdamW],
    read_before_step: bool,
) -> None:
    """Metric reports read the step count without a device synchronization; it must stay the count."""
    _disable_real_collective(monkeypatch)
    parameter = torch.nn.Parameter(torch.tensor([1.0, -1.0], device="cuda"))
    optimizer = optimizer_class([parameter], lr=0.1)
    for value in (0.5, float("nan"), -0.25):
        parameter.grad = torch.full((2,), value, device="cuda")
        optimizer.step()
        assert optimizer.host_step(parameter) == optimizer.state[parameter]["step"].item()
    loaded = optimizer_class([parameter], lr=0.1)
    loaded.load_state_dict(optimizer.state_dict())
    if read_before_step:
        assert loaded.host_step(parameter) == 2
    parameter.grad = torch.full((2,), 0.5, device="cuda")

    loaded.step()

    assert loaded.host_step(parameter) == loaded.state[parameter]["step"].item() == 3
    assert step_counts(loaded) == {parameter: 3}


@pytest.mark.unit
@pytest.mark.gpu
def test_step_counts_read_device_counters_of_other_optimizers_and_skip_host_ones() -> None:
    """Without host copies, CUDA step counters are read and host counters are left to the caller."""
    on_device = torch.nn.Parameter(torch.ones(1, device="cuda"))
    on_host = torch.nn.Parameter(torch.ones(1, device="cuda"))
    optimizer = SimpleNamespace(state={
        on_device: {"step": torch.tensor(4.0, device="cuda")},
        on_host: {"step": torch.tensor(9.0)},
    })

    assert step_counts(optimizer) == {on_device: 4.0}
    assert step_counts(SimpleNamespace(state={on_host: {"step": torch.tensor(9.0)}})) == {}
