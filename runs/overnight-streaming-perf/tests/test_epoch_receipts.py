from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import measure


class _TrackingCondition:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.is_held = False

    def __enter__(self) -> _TrackingCondition:
        assert not self.is_held
        self.is_held = True
        self.events.append("condition_enter")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.is_held = False
        self.events.append("condition_exit")


class _TrackingThread:
    def __init__(self, condition: _TrackingCondition, events: list[str]) -> None:
        self.condition = condition
        self.events = events

    @property
    def name(self) -> str:
        assert self.condition.is_held
        self.events.append("thread_name")
        return "streaming-prefetch-producer"

    def is_alive(self) -> bool:
        assert self.condition.is_held
        self.events.append("thread_alive")
        return True


class _TrackingDataset:
    def __init__(self, events: list[str], next_epoch: int) -> None:
        self.events = events
        self._next_epoch = next_epoch

    @property
    def epoch_size(self) -> int:
        self.events.append("epoch_size")
        return 51_200

    @property
    def next_epoch(self) -> int:
        self.events.append("next_epoch")
        return self._next_epoch

    @next_epoch.setter
    def next_epoch(self, value: int) -> None:
        self._next_epoch = value

    def __len__(self) -> int:
        self.events.append("dataset_length")
        return 6_400


def _measurement(
    trainer: object,
    output_dir: Path,
    *,
    clock_ns=lambda: 20,
    wall_clock_ns=lambda: 10,
) -> measure.WindowMeasurement:
    return measure.WindowMeasurement(
        trainer=trainer,
        run_id="epoch-receipt-test",
        output_dir=output_dir,
        expected_steps=0,
        warmup_updates=0,
        window_updates=0,
        monitor_interval=0,
        rank=0,
        world_size=8,
        cuda_synchronize=lambda: None,
        memory_stats=lambda: {},
        capture_microbatches=False,
        clock_ns=clock_ns,
        wall_clock_ns=wall_clock_ns,
    )


def test_epoch_and_queue_receipts_precede_window_clock_and_serialize(
    tmp_path: Path, monkeypatch
) -> None:
    events: list[str] = []
    condition = _TrackingCondition(events)
    dataset = _TrackingDataset(events, next_epoch=0)
    iterator = SimpleNamespace(
        _condition=condition,
        _ready=["queued-batch-0", "queued-batch-1"],
        _capacity=32,
        _thread=_TrackingThread(condition, events),
    )
    trainer = SimpleNamespace(
        model_parts=[],
        metrics_processor=SimpleNamespace(ntokens_since_last_log=0),
        ntokens_seen=0,
        step=0,
        dataloader=SimpleNamespace(dataset=dataset, _prefetch_iterator=iterator),
    )
    monkeypatch.setattr(measure, "parameter_census", lambda model_parts: {})
    monkeypatch.setattr(measure, "host_memory_snapshot", lambda: {})

    def clock_ns() -> int:
        events.append("monotonic_clock")
        return 20

    def wall_clock_ns() -> int:
        events.append("wall_clock")
        return 10

    receipt = _measurement(
        trainer, tmp_path, clock_ns=clock_ns, wall_clock_ns=wall_clock_ns
    )
    dataset.next_epoch = 1
    events.clear()
    receipt._start_window()

    assert events.index("condition_enter") < events.index("thread_name")
    assert events.index("thread_name") < events.index("thread_alive")
    assert events.index("thread_alive") < events.index("condition_exit")
    assert events.index("condition_exit") < events.index("wall_clock")
    assert events.index("wall_clock") < events.index("monotonic_clock")

    receipt.capture_after_train()
    receipt.finish(training_succeeded=True)
    record = json.loads((tmp_path / "rank-00.json").read_text(encoding="utf-8"))
    boundaries = record["training_boundary_receipts"]
    assert boundaries["before_train"] == {
        "dataset_available": True,
        "epoch_size": 51_200,
        "length_samples_per_rank": 6_400,
        "next_epoch": 0,
    }
    assert boundaries["before_window"]["epoch"]["next_epoch"] == 1
    assert boundaries["after_train"]["next_epoch"] == 1
    assert boundaries["before_window"]["prefetch_queue"] == {
        "iterator_present": True,
        "snapshot_available": True,
        "capacity": 32,
        "ready_count": 2,
        "producer_thread_name": "streaming-prefetch-producer",
        "producer_thread_alive": True,
    }


@pytest.mark.parametrize("initial_next_epoch", [0, 1])
def test_adapter_records_after_train_epoch_after_native_return(
    tmp_path: Path, monkeypatch, initial_next_epoch: int
) -> None:
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setattr(measure, "parameter_census", lambda model_parts: {})

    class Dataset:
        epoch_size = 51_200
        next_epoch = 0

        def __len__(self) -> int:
            return 6_400

    class Profiler:
        def step(self) -> None:
            return None

    class Metrics:
        def interval_metrics(self) -> None:
            return None

    class Trainer:
        def __init__(self) -> None:
            dataset = Dataset()
            dataset.next_epoch = initial_next_epoch
            self.step = 0
            self.model_parts = []
            self.metrics_processor = SimpleNamespace(ntokens_since_last_log=0)
            self.ntokens_seen = 0
            self.dataloader = SimpleNamespace(
                dataset=dataset, _prefetch_iterator=None
            )
            self.profiler = Profiler()

        def forward_backward_step(self) -> None:
            return None

        def train(self) -> str:
            self.step = 1
            self.dataloader.dataset.next_epoch = 1
            self.profiler.step()
            return "native-result"

    adapter = measure.NativeAdapter(
        trainer_class=Trainer,
        profiler_class=Profiler,
        metrics_class=Metrics,
        metrics_module=SimpleNamespace(),
        options={
            "run_id": "epoch-adapter-test",
            "output_dir": tmp_path,
            "expected_steps": 1,
            "warmup_updates": 0,
            "window_updates": 0,
            "monitor_interval": 0,
            "capture_microbatches": False,
        },
        cuda_synchronize=lambda: None,
        memory_stats=lambda: {},
    )

    with adapter:
        assert Trainer().train() == "native-result"

    record = json.loads((tmp_path / "rank-00.json").read_text(encoding="utf-8"))
    boundaries = record["training_boundary_receipts"]
    assert boundaries["before_train"]["next_epoch"] == initial_next_epoch
    assert boundaries["after_train"]["next_epoch"] == 1


def test_receipts_tolerate_non_mosaic_trainer_without_dataset_or_queue(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(measure, "parameter_census", lambda model_parts: {})
    monkeypatch.setattr(measure, "host_memory_snapshot", lambda: {})
    trainer = SimpleNamespace(
        model_parts=[],
        metrics_processor=SimpleNamespace(ntokens_since_last_log=0),
        ntokens_seen=0,
        step=0,
        dataloader=object(),
    )
    receipt = _measurement(trainer, tmp_path)

    receipt._start_window()
    receipt.capture_after_train()
    receipt.finish(training_succeeded=True)
    record = json.loads((tmp_path / "rank-00.json").read_text(encoding="utf-8"))
    boundaries = record["training_boundary_receipts"]
    unavailable_epoch = {
        "dataset_available": False,
        "epoch_size": None,
        "length_samples_per_rank": None,
        "next_epoch": None,
    }
    assert boundaries["before_train"] == unavailable_epoch
    assert boundaries["before_window"]["epoch"] == unavailable_epoch
    assert boundaries["after_train"] == unavailable_epoch
    assert boundaries["before_window"]["prefetch_queue"] == {
        "iterator_present": False,
        "snapshot_available": False,
        "capacity": None,
        "ready_count": None,
        "producer_thread_name": None,
        "producer_thread_alive": None,
    }
