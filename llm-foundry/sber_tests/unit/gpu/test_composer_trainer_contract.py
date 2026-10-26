from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from composer import Trainer
from composer.core import Callback, Event, State
from composer.loggers import Logger
from composer.models import ComposerClassifier
from composer.optim import LinearScheduler, LinearWithWarmupScheduler
from composer.optim.scheduler import ComposerScheduler
from composer.trainer import trainer as trainer_module


class _RandomClassificationDataset(Dataset):
    """Provide deterministic-shape random samples without test-package imports."""

    def __init__(self, size: int) -> None:
        """Initialize the requested number of binary-classification samples."""
        self.features = torch.randn(size, 1, 1, 1)
        self.targets = torch.randint(0, 2, (size,))

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one feature tensor and its class index."""
        return self.features[index], self.targets[index]


class _SimpleModel(ComposerClassifier):
    """Small classification model for real Trainer contract tests."""

    def __init__(self) -> None:
        """Initialize a two-layer binary classifier."""
        module = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(1, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 2),
            torch.nn.Softmax(dim=-1),
        )
        super().__init__(module=module, num_classes=2)


@pytest.fixture(autouse=True)
def _disable_precision_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate trainer contracts from the pinned Composer autocast API."""
    monkeypatch.setattr(
        trainer_module,
        "_get_precision_context",
        lambda *args, **kwargs: nullcontext(),
    )


class _LearningRateRecorder(Callback):
    """Record the optimizer learning rate at one trainer event."""

    def __init__(self, event: Event) -> None:
        """Initialize the event and an empty learning-rate history."""
        self.event = event
        self.learning_rates: list[float] = []

    def run_event(self, event: Event, state: State, logger: Logger) -> None:
        """Record the first optimizer group's rate at the selected event."""
        del logger
        if event == self.event:
            self.learning_rates.append(state.optimizers[0].param_groups[0]["lr"])


def _build_scheduler_trainer(
    scheduler: ComposerScheduler,
    recorder: _LearningRateRecorder,
    *,
    max_duration: str,
    train_subset_num_batches: int,
    step_schedulers_every_batch: bool,
) -> Trainer:
    """Build the pinned real Trainer used by scheduler timeline tests."""
    model = _SimpleModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    dataset = _RandomClassificationDataset(size=32)
    return Trainer(
        model=model,
        train_dataloader=DataLoader(dataset, batch_size=2),
        optimizers=optimizer,
        schedulers=scheduler,
        callbacks=[recorder],
        max_duration=max_duration,
        train_subset_num_batches=train_subset_num_batches,
        step_schedulers_every_batch=step_schedulers_every_batch,
        progress_bar=False,
        log_to_console=False,
    )


class _Logger:
    """Capture trainer metrics without a logging backend."""

    def __init__(self) -> None:
        """Initialize an ordered collection of metric payloads."""
        self.payloads: list[dict[str, float]] = []

    def log_metrics(self, metrics: dict[str, float]) -> None:
        """Record one metric payload."""
        self.payloads.append(metrics)


@pytest.mark.unit
@pytest.mark.gpu
def test_sequence_stats_use_one_gather_and_log_min_average_and_max(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One gathered vector must supply every sequence statistic."""
    logger = _Logger()
    subject = SimpleNamespace(logger=logger)
    calls: list[torch.Tensor] = []

    def gather_once(local_length: torch.Tensor) -> list[torch.Tensor]:
        """Return deterministic lengths from four ranks."""
        calls.append(local_length)
        return [
            torch.tensor([8], device="cuda"),
            torch.tensor([16], device="cuda"),
            torch.tensor([12], device="cuda"),
            torch.tensor([20], device="cuda"),
        ]

    monkeypatch.setattr(trainer_module.dist, "all_gather", gather_once)

    stats = trainer_module.Trainer._calculate_sequence_stats(subject, 12)

    assert len(calls) == 1
    assert stats == {
        "min_seq_len": 8,
        "avg_seq_len": 14.0,
        "global_max_seq_len": 20,
    }
    assert logger.payloads == [
        {
            "trainer/dataloader/min_max_seq_len": 8,
            "trainer/dataloader/avg_max_seq_len": 14.0,
            "trainer/dataloader/global_max_seq_len": 20,
        },
    ]


@pytest.mark.unit
@pytest.mark.gpu
@pytest.mark.parametrize(
    ("sequence_length", "expected_sp_size"),
    [(1, 1), (1024, 1), (1025, 2), (4096, 2), (4097, 4)],
)
def test_dynamic_sp_selects_from_global_maximum_boundaries(
    sequence_length: int,
    expected_sp_size: int,
) -> None:
    """Dynamic SP selection must honor inclusive maximum-length boundaries."""
    subject = SimpleNamespace(
        dynamic_sp=[
            {"max_seq_len": 1024, "sp_size": 1},
            {"max_seq_len": 4096, "sp_size": 2},
            {"max_seq_len": 8192, "sp_size": 4},
        ],
    )

    actual = trainer_module.Trainer._get_sp_size_for_seq_len(
        subject,
        sequence_length,
    )

    assert actual == expected_sp_size


@pytest.mark.unit
@pytest.mark.gpu
def test_dynamic_sp_fails_clearly_without_local_length() -> None:
    """Dynamic SP cannot proceed when the dataloader omits local_max_seq_len."""
    subject = SimpleNamespace(dynamic_sp=[{"max_seq_len": 1024, "sp_size": 1}])

    with pytest.raises(
        Exception,
        match="local_max_seq_len is not provided but dynamic_sp is set",
    ):
        trainer_module.Trainer._handle_dynamic_sp(
            subject,
            {},
            global_max_seq_len=1024,
        )


@pytest.mark.unit
@pytest.mark.gpu
def test_fixed_sp_logs_supplied_global_statistics_without_switching(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A matching SP size must log current groups without switching."""
    logger = _Logger()
    subject = SimpleNamespace(
        dynamic_sp=[{"max_seq_len": 4096, "sp_size": 2}],
        logger=logger,
    )
    switched: list[int] = []
    subject._get_sp_size_for_seq_len = lambda sequence_length: 2
    subject._switch_sp_group = lambda size: switched.append(size)
    monkeypatch.setattr(trainer_module.dist, "get_sp_group_size", lambda: 2)
    monkeypatch.setattr(trainer_module.dist, "get_tp_sp_group_size", lambda: 4)

    trainer_module.Trainer._handle_dynamic_sp(
        subject,
        {"local_max_seq_len": 2048},
        global_max_seq_len=3072,
    )

    assert switched == []
    assert logger.payloads == [
        {
            "trainer/dynamic_sp/sp_size": 2,
            "trainer/dynamic_sp/tp_sp_size": 4,
        },
    ]


@pytest.mark.unit
@pytest.mark.gpu
def test_dynamic_sp_switch_uses_global_not_local_maximum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The switch decision must be based on the supplied global maximum."""
    logger = _Logger()
    subject = SimpleNamespace(
        dynamic_sp=[
            {"max_seq_len": 1024, "sp_size": 1},
            {"max_seq_len": 4096, "sp_size": 2},
        ],
        logger=logger,
    )
    selected_lengths: list[int] = []
    switched: list[int] = []

    def select_sp(sequence_length: int) -> int:
        """Record the selection input and return the larger SP group."""
        selected_lengths.append(sequence_length)
        return 2

    subject._get_sp_size_for_seq_len = select_sp
    subject._switch_sp_group = lambda size: switched.append(size)
    monkeypatch.setattr(trainer_module.dist, "get_sp_group_size", lambda: 1)
    monkeypatch.setattr(trainer_module.dist, "get_tp_sp_group_size", lambda: 2)

    trainer_module.Trainer._handle_dynamic_sp(
        subject,
        {"local_max_seq_len": 512},
        global_max_seq_len=3072,
    )

    assert selected_lengths == [3072]
    assert switched == [2]


@pytest.mark.unit
@pytest.mark.gpu
@pytest.mark.parametrize(
    ("scheduler", "expected_learning_rates"),
    [
        pytest.param(
            LinearScheduler(alpha_i=0.0, alpha_f=1.0, t_max="3ba"),
            [0.0, 1.0 / 3.0, 2.0 / 3.0],
            id="linear",
        ),
        pytest.param(
            LinearWithWarmupScheduler(
                t_warmup="2ba",
                alpha_i=1.0,
                alpha_f=0.0,
                t_max="4ba",
            ),
            [0.0, 0.5, 1.0, 0.5],
            id="warmup",
        ),
    ],
)
def test_batch_scheduler_uses_completed_batch_timestamp(
    scheduler: ComposerScheduler,
    expected_learning_rates: list[float],
) -> None:
    """Every real Trainer batch must start with its completed-timestamp factor."""
    recorder = _LearningRateRecorder(Event.BATCH_START)
    trainer = _build_scheduler_trainer(
        scheduler,
        recorder,
        max_duration=f"{len(expected_learning_rates)}ba",
        train_subset_num_batches=len(expected_learning_rates),
        step_schedulers_every_batch=True,
    )

    trainer.dynamic_sp = None
    trainer.fit()

    assert recorder.learning_rates == pytest.approx(expected_learning_rates)


@pytest.mark.unit
@pytest.mark.gpu
def test_epoch_scheduler_uses_completed_epoch_timestamp() -> None:
    """A real epoch scheduler must prepare the factor for the next epoch."""
    recorder = _LearningRateRecorder(Event.EPOCH_START)
    trainer = _build_scheduler_trainer(
        LinearScheduler(alpha_i=0.0, alpha_f=1.0, t_max="2ep"),
        recorder,
        max_duration="2ep",
        train_subset_num_batches=1,
        step_schedulers_every_batch=False,
    )

    trainer.dynamic_sp = None
    trainer.fit()

    assert recorder.learning_rates == pytest.approx([0.0, 0.5])
