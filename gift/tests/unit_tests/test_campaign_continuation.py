import copy
import pickle
import random
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.checkpoint import load as dcp_load
from torch.distributed.checkpoint import save as dcp_save

from torchtitan.callbacks.base import Event
from torchtitan.callbacks.manager import CallbackManager
from torchtitan.components.dataloader import DataloaderExhaustedError
from torchtitan.config import load_document, run_pre_submit
from torchtitan.config.build import prepare_document
from torchtitan.config.schema.experiments.cuda_graph_campaign.continuation import (
    CampaignRNGContinuationConfig,
)
from torchtitan.experiments.cuda_graph_campaign.continuation import (
    CampaignRNGContinuationObserver,
    RNGContinuationState,
)
from torchtitan.experiments.cuda_graph_campaign.trainer import CampaignTrainer
from torchtitan.trainer import Trainer


GIFT_ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG = GIFT_ROOT / "configs/experiments/gift_moe_3b_hybridep_fp8_compile.yaml"
CONTINUATION_CONFIG = (
    GIFT_ROOT / "configs/experiments/cuda_graph_campaign_continuation.yaml"
)


@pytest.fixture(autouse=True)
def preserve_process_rng():
    """Keep this module's deterministic cases from changing the caller's RNG."""
    python_state = random.getstate()
    numpy_state = copy.deepcopy(np.random.get_state())
    torch_cpu_state = torch.get_rng_state().clone()
    cuda_snapshot = None
    if torch.cuda.is_initialized():
        cuda_device = torch.cuda.current_device()
        cuda_snapshot = (
            cuda_device,
            torch.cuda.get_rng_state(cuda_device).clone(),
            torch.cuda.set_rng_state,
        )

    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_cpu_state)
        if cuda_snapshot is not None:
            # Tests can replace these helpers before fixture teardown.
            cuda_device, cuda_state, restore_cuda_rng = cuda_snapshot
            restore_cuda_rng(cuda_state, device=cuda_device)


class FakeCudaRNG:
    """CPU test double for the current CUDA generator only."""

    def __init__(self, device_index: int = 0):
        self.device_index = device_index
        self.state = torch.arange(16, dtype=torch.uint8)

    def current_device(self):
        return self.device_index

    def get_rng_state(self, device=None):
        if device is not None:
            assert torch.device(device).index == self.device_index
        return self.state.clone()

    def set_rng_state(self, state, device=None):
        if device is not None:
            assert torch.device(device).index == self.device_index
        self.state = state.clone()


def patch_distributed_owner(monkeypatch, *, rank=0, world_size=1):
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_rank", lambda: rank)
    monkeypatch.setattr(dist, "get_world_size", lambda: world_size)


def patch_cuda_rng(monkeypatch, fake: FakeCudaRNG):
    monkeypatch.setattr(torch.cuda, "current_device", fake.current_device)
    monkeypatch.setattr(torch.cuda, "get_rng_state", fake.get_rng_state)
    monkeypatch.setattr(torch.cuda, "set_rng_state", fake.set_rng_state)


def seed_cpu_rng(seed: int):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    torch.set_rng_state(generator.get_state())


def capture_rng(fake: FakeCudaRNG):
    return (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state().clone(),
        fake.state.clone(),
    )


def numpy_states_equal(left, right):
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def assert_rng_matches_payload(payload, fake: FakeCudaRNG):
    python_state, numpy_state = pickle.loads(payload["python_numpy_state"])
    assert random.getstate() == python_state
    assert numpy_states_equal(np.random.get_state(), numpy_state)
    assert torch.equal(torch.get_rng_state(), payload["torch_cpu_state"])
    assert torch.equal(fake.state, payload["torch_cuda_state"])


def assert_rng_snapshots_equal(left, right):
    assert left[0] == right[0]
    assert numpy_states_equal(left[1], right[1])
    assert torch.equal(left[2], right[2])
    assert torch.equal(left[3], right[3])


def make_observer():
    return CampaignRNGContinuationObserver(CampaignRNGContinuationConfig())


def make_callback_manager():
    return CallbackManager(
        {"campaign_rng_continuation": CampaignRNGContinuationConfig()},
        sharding_backend=object(),
    )


def make_fit_start_trainer(trainer_type, callbacks):
    trainer = object.__new__(trainer_type)
    trainer.callbacks = callbacks
    trainer.loss_fn = SimpleNamespace()
    trainer.model_parts = []
    trainer.parallel_dims = None
    trainer.device = torch.device("cpu")
    trainer.config = None
    trainer.step = 0
    trainer.checkpointer = SimpleNamespace(last_saved_path=None)
    trainer.ntokens_seen = 0
    trainer.gradient_accumulation_steps = 1
    trainer.optimizers = SimpleNamespace(optimizers=[])
    trainer.metrics_processor = SimpleNamespace(
        data_loading_times=[], log_extra_metrics=lambda *_: None
    )
    return trainer


def test_rng_state_stages_immutable_cpu_and_mocked_cuda_state(monkeypatch):
    fake_cuda = FakeCudaRNG()
    patch_cuda_rng(monkeypatch, fake_cuda)
    patch_distributed_owner(monkeypatch)
    random.seed(713)
    np.random.seed(713)
    seed_cpu_rng(713)
    fake_cuda.set_rng_state(torch.arange(16, dtype=torch.uint8) + 23)

    state = RNGContinuationState(
        global_rank=0, world_size=1, device=torch.device("cuda", 0)
    )
    saved = state.state_dict()
    saved_copy = copy.deepcopy(saved)
    expected_torch = saved_copy["torch_cpu_state"].clone()
    expected_cuda = saved_copy["torch_cuda_state"].clone()

    random.seed(999)
    np.random.seed(999)
    seed_cpu_rng(999)
    fake_cuda.set_rng_state(torch.full((16,), 91, dtype=torch.uint8))
    before_load = capture_rng(fake_cuda)
    state.load_state_dict(saved)
    assert_rng_snapshots_equal(capture_rng(fake_cuda), before_load)

    # The staged payload owns its tensors even if the caller mutates its copy.
    saved["torch_cpu_state"].zero_()
    saved["torch_cuda_state"].zero_()
    assert state.restore_after_first_batch()
    assert_rng_matches_payload(
        {
            **saved_copy,
            "torch_cpu_state": expected_torch,
            "torch_cuda_state": expected_cuda,
        },
        fake_cuda,
    )
    assert not state.restore_after_first_batch()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", 2),
        ("global_rank", 1),
        ("world_size", 2),
        ("device_type", "cpu"),
        ("device_index", 1),
        ("torch_cpu_state", torch.zeros(8, dtype=torch.int64)),
        ("torch_cpu_state", torch.zeros(8, dtype=torch.uint8)),
        ("torch_cuda_state", torch.zeros(16, dtype=torch.int64)),
        ("torch_cuda_state", torch.zeros(8, dtype=torch.uint8)),
        ("python_numpy_state", b"not a pickle"),
    ],
)
def test_rng_state_rejects_malformed_or_wrong_owner_payload(monkeypatch, field, value):
    fake_cuda = FakeCudaRNG()
    patch_cuda_rng(monkeypatch, fake_cuda)
    patch_distributed_owner(monkeypatch)
    state = RNGContinuationState(
        global_rank=0, world_size=1, device=torch.device("cuda", 0)
    )
    payload = state.state_dict()
    payload[field] = value
    with pytest.raises(ValueError):
        state.load_state_dict(payload)


def test_rng_state_checks_current_rank_world_and_device(monkeypatch):
    fake_cuda = FakeCudaRNG()
    patch_cuda_rng(monkeypatch, fake_cuda)
    patch_distributed_owner(monkeypatch)
    state = RNGContinuationState(
        global_rank=0, world_size=1, device=torch.device("cuda", 0)
    )
    payload = state.state_dict()

    monkeypatch.setattr(dist, "get_rank", lambda: 1)
    with pytest.raises(ValueError, match="current rank"):
        state.load_state_dict(payload)

    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    with pytest.raises(ValueError, match="current world size"):
        state.load_state_dict(payload)

    monkeypatch.setattr(dist, "get_world_size", lambda: 1)
    fake_cuda.device_index = 1
    with pytest.raises(ValueError, match="current device"):
        state.load_state_dict(payload)


def test_rng_fixture_restores_with_its_original_cuda_setter(monkeypatch):
    snapshot_cuda = FakeCudaRNG()
    patch_cuda_rng(monkeypatch, snapshot_cuda)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    expected = snapshot_cuda.state.clone()
    preservation = preserve_process_rng.__wrapped__()
    next(preservation)
    snapshot_cuda.state.zero_()

    with monkeypatch.context() as inner:
        changed_owner = FakeCudaRNG(device_index=1)
        patch_cuda_rng(inner, changed_owner)
        with pytest.raises(StopIteration):
            next(preservation)

    assert torch.equal(snapshot_cuda.state, expected)
    assert torch.equal(changed_owner.state, torch.arange(16, dtype=torch.uint8))


def _dcp_worker(rank, world_size, init_file, checkpoint_dir):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    fake_cuda = FakeCudaRNG(rank)
    with ExitStack() as stack:
        stack.enter_context(
            mock.patch.object(torch.cuda, "current_device", fake_cuda.current_device)
        )
        stack.enter_context(
            mock.patch.object(torch.cuda, "get_rng_state", fake_cuda.get_rng_state)
        )
        stack.enter_context(
            mock.patch.object(torch.cuda, "set_rng_state", fake_cuda.set_rng_state)
        )

        random.seed(400 + rank)
        np.random.seed(400 + rank)
        seed_cpu_rng(400 + rank)
        fake_cuda.set_rng_state(torch.full((16,), rank + 11, dtype=torch.uint8))
        observer = make_observer()
        states = observer.stateful_objects()
        assert list(states) == [f"continuation_rng_rank_{rank}"]
        expected = observer.rng_state.state_dict()
        dcp_save(states, checkpoint_id=checkpoint_dir)

        random.seed(900 + rank)
        np.random.seed(900 + rank)
        seed_cpu_rng(900 + rank)
        fake_cuda.set_rng_state(torch.full((16,), rank + 77, dtype=torch.uint8))
        before_load = capture_rng(fake_cuda)
        dcp_load(states, checkpoint_id=checkpoint_dir)
        assert_rng_snapshots_equal(capture_rng(fake_cuda), before_load)

        assert observer.rng_state.restore_after_first_batch()
        assert_rng_matches_payload(expected, fake_cuda)
        assert not observer.rng_state.restore_after_first_batch()

    dist.destroy_process_group()


def test_native_dcp_preserves_independent_rank_qualified_rng_payloads(tmp_path):
    world_size = 2
    init_file = tmp_path / "gloo-init"
    checkpoint_dir = tmp_path / "rank-qualified-checkpoint"
    mp.spawn(
        _dcp_worker,
        args=(world_size, str(init_file), str(checkpoint_dir)),
        nprocs=world_size,
        join=True,
    )


def clone_batch(batch):
    return ({name: value.clone() for name, value in batch[0].items()}, batch[1].clone())


def batches_equal(left, right):
    return (
        torch.equal(left[1], right[1])
        and left[0].keys() == right[0].keys()
        and all(torch.equal(left[0][name], right[0][name]) for name in left[0])
    )


def make_target_loader():
    from torchtitan.components.tokenizer import HuggingFaceTokenizer
    from torchtitan.config.schema.datasets.hf.text_datasets import (
        HuggingFaceTextDataLoaderConfig,
    )
    from torchtitan.datasets.hf.text_datasets import HuggingFaceTextDataLoader

    config = HuggingFaceTextDataLoaderConfig(
        dataset="c4_test",
        dataset_path=str(GIFT_ROOT / "tests/assets/c4_test"),
        num_workers=0,
        pin_memory=False,
        infinite=True,
    )
    tokenizer = HuggingFaceTokenizer(
        tokenizer_path=str(GIFT_ROOT / "tests/assets/tokenizer")
    )
    return HuggingFaceTextDataLoader(
        config,
        dp_world_size=8,
        dp_rank=0,
        tokenizer=tokenizer,
        seq_len=4096,
        local_batch_size=8,
    )


def test_target_loader_restore_is_after_lazy_iterator_and_first_fetch(monkeypatch):
    fake_cuda = FakeCudaRNG()
    patch_cuda_rng(monkeypatch, fake_cuda)
    patch_distributed_owner(monkeypatch)

    source_loader = make_target_loader()
    random.seed(713)
    np.random.seed(713)
    seed_cpu_rng(713)
    source_iterator = iter(source_loader)
    for _ in range(4):
        next(source_iterator)
    loader_state = copy.deepcopy(source_loader.state_dict())
    saved_observer = make_observer()
    saved_payload = saved_observer.rng_state.state_dict()

    expected = [clone_batch(next(source_iterator)) for _ in range(4)]
    expected_after = capture_rng(fake_cuda)

    resumed_loader = make_target_loader()
    resumed_loader.load_state_dict(loader_state)
    callback_manager = make_callback_manager()
    resumed_observer = callback_manager.find(CampaignRNGContinuationObserver)
    assert resumed_observer is not None
    trainer = make_fit_start_trainer(CampaignTrainer, callback_manager)
    trainer._bind_rng_continuation()
    resumed_observer.rng_state.load_state_dict(saved_payload)

    random.seed(121)
    np.random.seed(121)
    seed_cpu_rng(121)
    fake_cuda.set_rng_state(torch.full((16,), 121, dtype=torch.uint8))
    before_resume_fetch = capture_rng(fake_cuda)

    trainer.metrics_processor = SimpleNamespace(data_loading_times=[])
    resumed_batches = trainer.batch_generator(resumed_loader)
    first = clone_batch(next(resumed_batches))

    assert batches_equal(expected[0], first)
    assert not torch.equal(before_resume_fetch[2], capture_rng(fake_cuda)[2])
    assert_rng_matches_payload(saved_payload, fake_cuda)

    actual = [first] + [clone_batch(next(resumed_batches)) for _ in range(3)]
    assert all(batches_equal(left, right) for left, right in zip(expected, actual))
    assert_rng_snapshots_equal(capture_rng(fake_cuda), expected_after)
    assert len(trainer.metrics_processor.data_loading_times) == 4


def test_batch_generator_preserves_unloaded_empty_and_exception_paths(monkeypatch):
    fake_cuda = FakeCudaRNG()
    patch_cuda_rng(monkeypatch, fake_cuda)
    patch_distributed_owner(monkeypatch)
    callback_manager = make_callback_manager()
    observer = callback_manager.find(CampaignRNGContinuationObserver)
    assert observer is not None
    trainer = make_fit_start_trainer(CampaignTrainer, callback_manager)
    trainer._bind_rng_continuation()

    batch = ({"input": torch.tensor([1])}, torch.tensor([2]))
    native_trainer = object.__new__(Trainer)
    native_trainer.metrics_processor = SimpleNamespace(data_loading_times=[])
    native_batches = Trainer.batch_generator(native_trainer, [batch, batch])
    expected = [next(native_batches), next(native_batches)]

    fresh_batches = trainer.batch_generator([batch, batch])
    actual = [next(fresh_batches), next(fresh_batches)]
    assert all(batches_equal(left, right) for left, right in zip(expected, actual))
    assert observer.rng_state._pending_restore is None
    assert not observer.rng_state._restore_attempted

    empty_batches = trainer.batch_generator([])
    with pytest.raises(DataloaderExhaustedError):
        next(empty_batches)
    assert not observer.rng_state._restore_attempted

    class FailingIterable:
        def __iter__(self):
            return self

        def __next__(self):
            raise RuntimeError("fetch failed")

    failed_batches = trainer.batch_generator(FailingIterable())
    with pytest.raises(RuntimeError, match="fetch failed"):
        next(failed_batches)
    assert not observer.rng_state._restore_attempted


def test_continuation_overlay_matches_target_with_only_named_changes():
    target = run_pre_submit(str(BASE_CONFIG), world_size=8)
    overlay = run_pre_submit(str(CONTINUATION_CONFIG), world_size=8)
    assert target.ok, target.report.to_dict()
    assert overlay.ok, overlay.report.to_dict()

    target_doc, _ = load_document(str(BASE_CONFIG))
    overlay_doc, _ = load_document(str(CONTINUATION_CONFIG))
    target_config = prepare_document(target_doc).trainer_config
    overlay_config = prepare_document(overlay_doc).trainer_config
    expected = target_config.to_dict()
    expected["callbacks"] = {
        "campaign_rng_continuation": {"type": "campaign_rng_continuation"}
    }
    expected["checkpoint"].update(
        {
            "enable": True,
            "interval": 5,
            "async_mode": "disabled",
            "initial_load_model_only": False,
            "initial_load_in_hf": False,
            "last_save_model_only": False,
            "save_in_hf": False,
            "last_save_in_hf": False,
        }
    )
    expected["lr_scheduler"]["total_steps"] = 30000
    assert type(overlay_config).__name__ == "CampaignTrainerConfig"
    assert overlay_config.to_dict() == expected


def test_callback_manager_rejects_unbound_continuation_at_fit_start(monkeypatch):
    fake_cuda = FakeCudaRNG()
    patch_cuda_rng(monkeypatch, fake_cuda)
    patch_distributed_owner(monkeypatch)

    native_callbacks = make_callback_manager()
    native_trainer = make_fit_start_trainer(Trainer, native_callbacks)
    native_observer = native_callbacks.find(CampaignRNGContinuationObserver)
    assert native_observer is not None
    with pytest.raises(RuntimeError, match="requires CampaignTrainer binding"):
        native_callbacks.emit(Event.fit_start, native_trainer)
    with pytest.raises(RuntimeError, match="requires CampaignTrainer binding"):
        native_observer.restore_after_first_batch()

    campaign_callbacks = make_callback_manager()
    campaign_trainer = make_fit_start_trainer(CampaignTrainer, campaign_callbacks)
    campaign_trainer._bind_rng_continuation()
    campaign_callbacks.emit(Event.fit_start, campaign_trainer)


def test_continuation_callback_decodes_but_pairing_is_runtime_enforced():
    result = run_pre_submit(
        str(BASE_CONFIG),
        [
            (
                "callbacks.campaign_rng_continuation",
                "{type: campaign_rng_continuation}",
            )
        ],
        world_size=8,
    )
    assert result.ok, result.report.to_dict()
    assert result.decoded


def test_continuation_overlay_rejects_unverified_worker_loader():
    result = run_pre_submit(
        str(CONTINUATION_CONFIG),
        [("dataloader.num_workers", "1")],
        world_size=8,
    )
    assert not result.ok
    assert any(issue.path == "dataloader.num_workers" for issue in result.report.issues)
