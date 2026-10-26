import importlib.util
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def _activation_monitor_path() -> Path:
    """Locate composer's activation_monitor.py.

    Prefer the in-repo submodule checkout; fall back to whatever `composer`
    resolves to on PYTHONPATH (the submodule may be un-checked-out in a
    worktree). Must run before `composer` is stubbed out in `sys.modules`.
    """
    vendored = (
        Path(__file__).parents[3]
        / "contrib" / "composer" / "composer" / "callbacks" / "activation_monitor.py"
    )
    if vendored.is_file():
        return vendored
    spec = importlib.util.find_spec("composer.callbacks.activation_monitor")
    if spec is None or spec.origin is None:
        pytest.skip("composer.callbacks.activation_monitor is not importable")
    return Path(spec.origin)


def _load_activation_monitor_module(monkeypatch):
    path = _activation_monitor_path()

    class _Callback:
        pass

    class _State:
        pass

    class _Logger:
        pass

    class _WandBLogger:
        pass

    class _GlobalActLogState:
        def __init__(self):
            self.enable_monitor = False

        def set_enable_monitor_variable(self, value):
            self.enable_monitor = value

        def get_enable_monitor_variable(self):
            return self.enable_monitor

    class _Dist:
        def __init__(self):
            self.ep_group = object()

        def get_world_size(self):
            return 1

        def get_ep_group(self):
            return self.ep_group

        def get_ep_group_size(self):
            return 1

    class _TimerType:
        CPU = "cpu"

    class _TimeUnit:
        BATCH = "ba"
        EPOCH = "ep"

    class _Time:
        def __init__(self, value=1, unit=_TimeUnit.BATCH):
            self.value = value
            self.unit = unit

        @classmethod
        def from_input(cls, value, unit):
            if isinstance(value, cls):
                return value
            if isinstance(value, int):
                return cls(value, unit)
            if isinstance(value, str):
                digits = "".join(ch for ch in value if ch.isdigit())
                return cls(int(digits or 1), unit)
            return cls(1, unit)

        @classmethod
        def from_timestring(cls, value):
            digits = "".join(ch for ch in value if ch.isdigit())
            return cls(int(digits or 1), _TimeUnit.BATCH)

        def __lt__(self, other):
            return self.value < other.value

    def _callback_timer(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator

    def _ensure_tuple(value):
        return value if isinstance(value, tuple) else (value,)

    te = types.ModuleType("transformer_engine")
    te_pytorch = types.ModuleType("transformer_engine.pytorch")
    te_tensor = types.ModuleType("transformer_engine.pytorch.tensor")
    te_quantized = types.ModuleType("transformer_engine.pytorch.tensor.quantized_tensor")
    te_quantized.QuantizedTensor = type("QuantizedTensor", (), {})
    te_quantized.QuantizedTensorBase = type("QuantizedTensorBase", (), {})

    composer = types.ModuleType("composer")
    composer_core = types.ModuleType("composer.core")
    composer_core.Callback = _Callback
    composer_core.State = _State
    composer_core.Time = _Time
    composer_core.TimeUnit = _TimeUnit
    composer_loggers = types.ModuleType("composer.loggers")
    composer_loggers.Logger = _Logger
    composer_wandb = types.ModuleType("composer.loggers.wandb_logger")
    composer_wandb.WandBLogger = _WandBLogger
    composer_utils = types.ModuleType("composer.utils")
    composer_utils.global_actlog_state = _GlobalActLogState()
    composer_utils.dist = _Dist()
    composer_utils.ensure_tuple = _ensure_tuple
    composer_utils.TimerType = _TimerType
    composer_utils.callback_timer = _callback_timer

    for name, module in {
        "transformer_engine": te,
        "transformer_engine.pytorch": te_pytorch,
        "transformer_engine.pytorch.tensor": te_tensor,
        "transformer_engine.pytorch.tensor.quantized_tensor": te_quantized,
        "composer": composer,
        "composer.core": composer_core,
        "composer.loggers": composer_loggers,
        "composer.loggers.wandb_logger": composer_wandb,
        "composer.utils": composer_utils,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    spec = importlib.util.spec_from_file_location("_composer_activation_monitor_for_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _RecordingLogger:
    destinations = []

    def __init__(self):
        self.metrics = None

    def log_metrics(self, metrics):
        self.metrics = metrics


class _FakeHandle:
    def wait(self):
        return None


@pytest.mark.unit
@pytest.mark.fast
def test_activation_monitor_make_log_uses_union_of_rank_metric_keys(monkeypatch):
    module = _load_activation_monitor_module(monkeypatch)
    monitor = object.__new__(module.ActivationMonitor)
    monitor.metrics = {"activations_amax/local": torch.tensor([2.0])}
    monitor.only_log_wandb = False
    monitor.all_gather_buffer = None
    monitor.all_gather_valid_buffer = None
    logger = _RecordingLogger()

    monkeypatch.setattr(module.dist, "get_world_size", lambda: 2)

    key_gathers = []

    def all_gather_object(output, obj, group=None):
        key_gathers.append((group, len(output), list(obj)))
        output[0] = obj
        output[1] = ["activations_amax/remote"]

    calls = []
    value_gathers = []

    def all_gather_into_tensor(output, input, group=None, async_op=False):
        value_gathers.append(group)
        assert input.numel() == 2
        assert output.numel() == 4
        calls.append(input.clone())
        if len(calls) == 1:
            remote = torch.tensor([0.0, 5.0], dtype=input.dtype, device=input.device)
        else:
            remote = torch.tensor([0.0, 1.0], dtype=input.dtype, device=input.device)
        output.copy_(torch.cat([input, remote]))
        return _FakeHandle()

    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)
    monkeypatch.setattr(torch.distributed, "all_gather_into_tensor", all_gather_into_tensor)

    monitor.make_log(logger, step=1)

    assert key_gathers == [(None, 2, ["activations_amax/local"])]
    assert value_gathers == [None, None]
    assert logger.metrics["activations_amax/local"] == 2.0
    assert logger.metrics["activations_amax/remote"] == 5.0
    assert monitor.metrics == {}


@pytest.mark.unit
@pytest.mark.fast
def test_activation_monitor_make_log_participates_with_no_local_metrics(monkeypatch):
    module = _load_activation_monitor_module(monkeypatch)
    monitor = object.__new__(module.ActivationMonitor)
    monitor.metrics = {}
    monitor.only_log_wandb = False
    monitor.all_gather_buffer = None
    monitor.all_gather_valid_buffer = None
    logger = _RecordingLogger()

    monkeypatch.setattr(module.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_backend", lambda: "gloo")

    key_gathers = []

    def all_gather_object(output, obj, group=None):
        key_gathers.append((group, len(output), list(obj)))
        output[0] = obj
        output[1] = ["activations_amax/remote"]

    calls = []
    value_gathers = []

    def all_gather_into_tensor(output, input, group=None, async_op=False):
        value_gathers.append(group)
        assert input.numel() == 1
        assert output.numel() == 2
        calls.append(input.clone())
        if len(calls) == 1:
            remote = torch.tensor([7.0], dtype=input.dtype, device=input.device)
        else:
            remote = torch.tensor([1.0], dtype=input.dtype, device=input.device)
        output.copy_(torch.cat([input, remote]))
        return _FakeHandle()

    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)
    monkeypatch.setattr(torch.distributed, "all_gather_into_tensor", all_gather_into_tensor)

    monitor.make_log(logger, step=1)

    assert key_gathers == [(None, 2, [])]
    assert value_gathers == [None, None]
    assert logger.metrics == {"activations_amax/remote": 7.0}
    assert monitor.metrics == {}


@pytest.mark.unit
@pytest.mark.fast
def test_activation_monitor_skips_empty_tensors(monkeypatch):
    module = _load_activation_monitor_module(monkeypatch)
    monitor = object.__new__(module.ActivationMonitor)
    monitor.is_forward = True
    monitor.stat_list = ["amax"]
    metrics = {}

    monitor.add_metrics(metrics, "module", "_output.0", torch.empty((0, 8)))

    assert metrics == {}
    stat = module.compute_activation_stats(torch.empty((0, 8)), "amax")
    assert stat.shape == (1,)
    assert stat.dtype == torch.float32
    assert torch.isnan(stat).item()
