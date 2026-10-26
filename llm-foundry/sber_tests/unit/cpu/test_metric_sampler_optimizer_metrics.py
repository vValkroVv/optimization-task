"""Rank-local optimizer metrics: whatever tensors the optimizer monitor emits, each is recorded as the
float32 scalar it holds, and a metric it did not emit as an absent one."""

import math
import uuid

import pytest

pytestmark = pytest.mark.unit

pytest.importorskip("composer", reason="Requires the CPU training runtime")

import torch  # noqa: E402

from llmfoundry.callbacks.metric_sampler import MetricSampler  # noqa: E402
from metric_sampler.capture.layout import Layout  # noqa: E402
from metric_sampler.capture.recorder import RankRecorder  # noqa: E402
from metric_sampler.capture.ring import ShmRing  # noqa: E402
from metric_sampler.contract.catalog import metric_by_name  # noqa: E402

_PREFIXES = (
    "l2_norm/grad", "l2_norm/moment", "l2_norm/moment_sq", "l2_norm/param", "l2_norm/update",
    "l2_norm/moment_bc", "l2_norm/moment_sq_bc", "amax/moment", "amax/moment_sq",
)
_NAMES = ["layers.0.q_proj", "layers.0.k_proj", "layers.0.v_proj"]


@pytest.fixture
def sampler(tmp_path):
    sampler = MetricSampler(str(tmp_path))
    monitors = sampler.monitor_capture
    monitors._optimizer_parameter_names = _NAMES
    monitors._parameter_indices = {name: index for index, name in enumerate(_NAMES)}
    monitors._optimizer_keys = tuple(tuple(f"{prefix}/{name}" for prefix in _PREFIXES) for name in _NAMES)
    layout = Layout([(metric_by_name("monitor/optimizer_local"), index, -1, 18) for index in range(len(_NAMES))])
    sampler.ring = ShmRing.create(f"ms_test_{uuid.uuid4().hex}", layout, slot_count=1)
    sampler.recorder = RankRecorder(layout, sampler.ring, torch.device("cpu"))
    sampler.recorder.begin_step(False)
    yield sampler
    sampler._shutdown()


def _recorded_rows(sampler: MetricSampler) -> list[list[float]]:
    recorder = sampler.recorder
    rows = []
    for index in range(len(_NAMES)):
        slot = recorder._layout.slot("monitor/optimizer_local", index, -1)
        rows.append(recorder.float_buffer()[slot.offset : slot.offset + slot.width].tolist())
    return rows


def _same(actual: list[float], expected: list[float]) -> bool:
    return all(a == e or (math.isnan(a) and math.isnan(e)) for a, e in zip(actual, expected, strict=True))


def test_metrics_of_every_shape_and_dtype_are_recorded_as_float32_scalars(sampler):
    emitted = [
        torch.tensor(1.5),
        torch.tensor([2.25]),
        torch.tensor(0.1, dtype=torch.bfloat16),
        torch.tensor(1 / 3, dtype=torch.float64),
        torch.tensor(7.0, requires_grad=True),
        torch.tensor([[0.2]], dtype=torch.float16),
        torch.tensor(3),
        torch.tensor(-0.0),
        torch.tensor(float("inf")),
    ]
    metrics = {f"{prefix}/{_NAMES[0]}": value for prefix, value in zip(_PREFIXES, emitted)}
    metrics[f"{_PREFIXES[0]}/{_NAMES[1]}"] = torch.tensor(float("nan"))
    metrics[f"{_PREFIXES[7]}/{_NAMES[1]}"] = torch.tensor(5.0)
    absent = float("nan")

    sampler.monitor_capture._observe_optimizer_metrics(metrics)

    first = [value.detach().float().reshape(()).item() for value in emitted]
    second = [float("nan"), absent, absent, absent, absent, absent, absent, 5.0, absent]
    rows = _recorded_rows(sampler)
    assert _same(rows[0], first + [1.0] * 9)
    assert _same(rows[1], second + [1.0, 0, 0, 0, 0, 0, 0, 1.0, 0])
    assert _same(rows[2], [absent] * 9 + [0.0] * 9)


@pytest.mark.parametrize("emitted_later", [{}, {f"{_PREFIXES[2]}/{_NAMES[2]}": torch.tensor(4.0)}])
def test_a_later_call_records_only_the_metrics_it_emits(sampler, emitted_later):
    monitors = sampler.monitor_capture
    every_metric = {f"{prefix}/{name}": torch.tensor(1.0) for prefix in _PREFIXES for name in _NAMES}
    monitors._observe_optimizer_metrics(every_metric)

    monitors._observe_optimizer_metrics(emitted_later)

    absent = float("nan")
    rows = _recorded_rows(sampler)
    assert _same(rows[0], [absent] * 9 + [0.0] * 9)
    assert _same(rows[1], [absent] * 9 + [0.0] * 9)
    if emitted_later:
        assert _same(rows[2], [absent, absent, 4.0] + [absent] * 6 + [0.0, 0.0, 1.0] + [0.0] * 6)
    else:
        assert _same(rows[2], [absent] * 9 + [0.0] * 9)


def test_an_unregistered_metric_is_rejected(sampler):
    with pytest.raises(ValueError, match="unregistered local identities"):
        sampler.monitor_capture._observe_optimizer_metrics({"l2_norm/grad/other": torch.tensor(1.0)})
