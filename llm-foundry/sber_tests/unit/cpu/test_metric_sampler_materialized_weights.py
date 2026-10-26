"""Materialized weight norms of the rank-local capture: the weights of a SuperTensorModule are recorded
together, before it hands its weight buffer on, and each is reduced like in a launch of its own module."""

import sys
import types
import uuid

import pytest

pytestmark = pytest.mark.unit

pytest.importorskip("composer", reason="Requires the CPU training runtime")

import torch  # noqa: E402
from composer.callbacks.weight_monitor import WeightMonitor, cleanup_metric_name  # noqa: E402

import llmfoundry.callbacks.metric_sampler_weights as weights_module  # noqa: E402
from llmfoundry.callbacks.metric_sampler import MetricSampler  # noqa: E402
from llmfoundry.callbacks.metric_sampler_monitors import RankLocalMonitorCapture  # noqa: E402
from llmfoundry.callbacks.metric_sampler_weights import MaterializedWeights  # noqa: E402
from metric_sampler.capture.layout import Layout  # noqa: E402
from metric_sampler.capture.recorder import RankRecorder  # noqa: E402
from metric_sampler.capture.ring import ShmRing  # noqa: E402
from metric_sampler.capture.stats.helpers.norms import weight_norm_rows  # noqa: E402
from metric_sampler.contract.catalog import metric_by_name  # noqa: E402

_METRIC = "monitor/materialized_weight_local"


class _Experts(torch.nn.Module):
    def __init__(self, experts: int) -> None:
        super().__init__()
        self.num_local_experts = experts
        self.weight1 = torch.nn.Parameter(torch.randn(experts, 5, 4))
        self.weight2 = torch.nn.Parameter(torch.randn(experts, 4, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _Block(torch.nn.Module):
    def __init__(self, experts: int, empty: bool) -> None:
        super().__init__()
        self.q_proj = torch.nn.Linear(8, 8)
        self.k_proj = torch.nn.Linear(8, 4, bias=False)
        self.empty = torch.nn.Linear(8, 0, bias=False) if empty else None
        self.experts = _Experts(experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.empty is not None:
            self.empty(x)
        return self.experts(self.k_proj(self.q_proj(x)).repeat(1, 2))


class _SuperTensorModule(torch.nn.Module):
    """Calls its root module's forward directly, as GigaFSDP's does, and hands its weights on in its own
    forward hook, which the weights of the root's modules do not survive."""

    def __init__(self, root_module: torch.nn.Module) -> None:
        super().__init__()
        self.root_module = root_module
        self.register_forward_hook(self._hand_weights_on)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.root_module.forward(x)

    def _hand_weights_on(self, module, inputs, output) -> None:
        for parameter in self.root_module.parameters():
            parameter.data.fill_(float("nan"))


class _Model(torch.nn.Module):
    def __init__(self, experts: int = 3, empty: bool = False) -> None:
        super().__init__()
        self.stm = _SuperTensorModule(_Block(experts, empty))
        self.head = torch.nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.stm(x))


def _expected_rows(model: torch.nn.Module) -> dict[str, torch.Tensor | None]:
    """Each registered weight's row as a reduction of the weight alone gives it; None for an empty one."""
    rows = {}
    for name, parameter in model.named_parameters():
        name = cleanup_metric_name(name)
        if ".experts." in name:
            for expert, weight in enumerate(parameter.detach()):
                rows[f"{name}__expert_idx_{expert}"] = weight_norm_rows([weight.reshape(-1)])[0]
        else:
            rows[name] = weight_norm_rows([parameter.detach()])[0] if parameter.numel() else None
    return rows


class _Recording:
    def __init__(self, tmp_path, model: torch.nn.Module, monkeypatch) -> None:
        self.model = model
        rows = _expected_rows(model)
        self.expected = {name: row.tolist() for name, row in rows.items() if row is not None}
        self.names = sorted(rows)
        self.calls: list[tuple[int, list[int]]] = []
        self.experts: list[list[int]] = []
        real_record = weights_module.record_weight_norm_rows

        def record(recorder, rows, weights, experts, launch_numels):
            self.calls.append((sum(count or 1 for count in experts), list(launch_numels)))
            self.experts.append(list(experts))
            real_record(recorder, rows, weights, experts, launch_numels)

        monkeypatch.setattr(weights_module, "record_weight_norm_rows", record)
        monkeypatch.setitem(sys.modules, "giga_fsdp", types.SimpleNamespace(SuperTensorModule=_SuperTensorModule))
        self.sampler = sampler = MetricSampler(str(tmp_path))
        monitors = sampler.monitor_capture
        monitor = WeightMonitor(interval="10ba", stat_list=["amax", "l2norm"])
        monitor.create_module_names(model)
        monitors._parameter_indices = {name: index for index, name in enumerate(self.names)}
        layout = Layout([(metric_by_name(_METRIC), index, -1, 3) for index in range(len(self.names))])
        self.ring = ShmRing.create(f"ms_test_{uuid.uuid4().hex}", layout, slot_count=1)
        sampler.recorder = self.recorder = RankRecorder(layout, self.ring, torch.device("cpu"))
        sampler.recorder.begin_step(False)
        sampler._capture_forward_active = True
        sampler._grad_forward = True
        self.units = RankLocalMonitorCapture._super_tensor_units(model)
        self.weights = sampler.materialized_weights = MaterializedWeights(
            sampler.recorder, monitor, self.names, self.units, monitors._parameter_rows)
        monitor.rank_local_observer = monitors._observe_materialized_weights
        monitor.attach_hooks(model)
        for super_tensor_module in dict.fromkeys(self.units.values()):
            super_tensor_module.register_forward_hook(monitors._record_super_tensor_weights, prepend=True)

    def row(self, name: str) -> list[float]:
        recorder = self.sampler.recorder
        slot = recorder._layout.slot(_METRIC, self.sampler.monitor_capture._parameter_indices[name], -1)
        return recorder.float_buffer()[slot.offset : slot.offset + slot.width].tolist()

    def close(self) -> None:
        self.sampler._shutdown()
        self.recorder.close()
        self.ring.unlink()


@pytest.fixture
def recording(tmp_path, monkeypatch):
    recordings = []

    def create(model: torch.nn.Module) -> _Recording:
        recordings.append(_Recording(tmp_path, model, monkeypatch))
        return recordings[-1]

    yield create
    for created in recordings:
        created.close()


def test_super_tensor_units_map_the_modules_inside_one_to_it(recording):
    model = _Model()
    recorded = recording(model)

    units = recorded.units

    assert set(units) == set(model.stm.root_module.modules())
    assert set(units.values()) == {model.stm}


def test_weights_of_a_super_tensor_are_recorded_together_before_it_hands_them_on(recording):
    model = _Model()
    recorded = recording(model)

    model(torch.randn(2, 8))

    assert recorded.sampler.recorder._recorded.all()
    assert {name: recorded.row(name) for name in recorded.names} == recorded.expected
    # One call for the modules of the SuperTensorModule, one for the module outside it. A weight is
    # reduced in as many blocks as the largest weight of its own module needs: q_proj's bias like its
    # weight, the slices of each expert weight like each other.
    stm_call, head_call = recorded.calls
    assert stm_call == (2 + 1 + 3 + 3, [64, 64, 32] + [20] * 3 + [12] * 3)
    assert head_call == (2, [64, 64])
    # Each expert weight goes as one stack of its three experts.
    assert recorded.experts == [[0, 0, 0, 3, 3], [0, 0]]


def test_weights_of_a_microbatch_after_the_first_are_not_recorded_again(recording):
    model = _Model()
    recorded = recording(model)

    model(torch.randn(2, 8))
    model(torch.randn(2, 8))

    assert len(recorded.calls) == 2
    assert {name: recorded.row(name) for name in recorded.names} == recorded.expected
    assert recorded.weights._pending == {}


@pytest.mark.parametrize("grad_forward, training", [(False, True), (True, False)])
def test_weights_are_not_recorded_outside_a_training_forward_with_grad(recording, grad_forward, training):
    model = _Model().train(training)
    recorded = recording(model)
    recorded.sampler._grad_forward = grad_forward

    model(torch.randn(2, 8))

    assert recorded.calls == []
    assert not recorded.sampler.recorder._recorded.any()
    assert recorded.weights._pending == {}


def test_a_weight_recorded_earlier_in_the_step_is_not_recorded_again(recording):
    model = _Model()
    recorded = recording(model)
    already_recorded = "stm.k_proj.weight"
    recorded.weights._seen.add(already_recorded)

    model(torch.randn(2, 8))

    assert recorded.row(already_recorded) == [0.0, 0.0, 0.0]
    assert {name: recorded.row(name) for name in recorded.names if name != already_recorded} == {
        name: row for name, row in recorded.expected.items() if name != already_recorded
    }
    assert recorded.calls[0] == (2 + 3 + 3, [64, 64] + [20] * 3 + [12] * 3)
    assert model.stm.root_module.k_proj in recorded.weights._modules_seen


def test_an_empty_weight_stays_unrecorded_and_does_not_hold_back_the_other_modules(recording):
    model = _Model(empty=True)
    recorded = recording(model)

    model(torch.randn(2, 8))

    seen = recorded.weights._modules_seen
    assert "stm.empty.weight" not in recorded.weights._seen
    assert model.stm.root_module.empty not in seen
    assert {model.stm.root_module.q_proj, model.stm.root_module.k_proj, model.stm.root_module.experts} <= seen
    assert {name: recorded.row(name) for name in recorded.expected} == recorded.expected


def test_modules_outside_a_super_tensor_are_recorded_at_their_own_forward(recording):
    model = _Model()
    recorded = recording(model)
    recorded.weights._units = {}

    model.head(torch.randn(2, 8))

    assert recorded.calls == [(2, [64, 64])]
    assert recorded.row("head.weight") == recorded.expected["head.weight"]
    assert recorded.row("head.bias") == recorded.expected["head.bias"]
