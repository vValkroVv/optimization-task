"""Bounded streaming prefetch and checkpoint cursor behavior."""

import random
import threading
from contextlib import closing
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

pytest.importorskip("streaming")
pytest.importorskip("tree_sitter")

import torch
from torch.utils.data import DataLoader

from tests.unit_tests.streaming_helpers import (
    install_single_process_shm_gather,
    make_sequential_rows,
    make_text_dataloader,
    write_tokenized_mds,
)
from torchtitan.components.validate import _iter_with_loader_cleanup
from torchtitan.datasets.streaming.core import dataloader as streaming_dataloader
from torchtitan.datasets.streaming.core.prefetch import PrefetchIterator
from torchtitan.trainer import Trainer


def _take(iterator, count):
    return [next(iterator) for _ in range(count)]


def _batch_snapshot(batch):
    inputs, labels = batch
    return (
        tuple(
            (name, tensor.dtype, tuple(tensor.shape), tensor.tolist())
            for name, tensor in inputs.items()
        ),
        (labels.dtype, tuple(labels.shape), labels.tolist()),
    )


def test_prefetch_preserves_order_and_capacity():
    reached = [threading.Event() for _ in range(4)]

    def source():
        for value in range(4):
            reached[value].set()
            yield value

    iterator = PrefetchIterator(source, capacity=2)
    try:
        assert reached[1].wait(5)
        with iterator.paused() as (ready, finished):
            assert ready == [0, 1]
            assert not finished
            assert not reached[2].is_set()
        assert list(iterator) == [0, 1, 2, 3]
    finally:
        iterator.close()
        iterator.join(5)


def test_prefetch_pause_freezes_cursor_and_queue():
    cursor = 0
    produced = threading.Event()

    def source():
        nonlocal cursor
        for value in range(10):
            cursor += 1
            if cursor == 2:
                produced.set()
            yield value

    iterator = PrefetchIterator(source, capacity=2)
    try:
        assert produced.wait(5)
        with iterator.paused() as (ready, finished):
            snapshot = (cursor, ready, finished)
            assert snapshot == (2, [0, 1], False)
        assert [next(iterator) for _ in range(4)] == [0, 1, 2, 3]
        resumed = PrefetchIterator(
            lambda: iter(range(snapshot[0], 10)),
            capacity=2,
            ready=snapshot[1],
            finished=snapshot[2],
        )
        try:
            assert list(resumed) == list(range(10))
        finally:
            resumed.close()
            resumed.join(5)
    finally:
        iterator.close()
        iterator.join(5)


def test_prefetch_drains_ready_items_before_raising_source_error():
    def source():
        yield 7
        raise ValueError("source failed")

    iterator = PrefetchIterator(source, capacity=2)
    try:
        assert next(iterator) == 7
        with pytest.raises(ValueError, match="source failed"):
            next(iterator)
        with pytest.raises(ValueError, match="source failed"):
            with iterator.paused():
                pass
    finally:
        iterator.close()
        iterator.join(5)


def test_prefetch_close_releases_full_queue_and_source():
    reached = threading.Event()
    finalized = threading.Event()

    def source():
        try:
            reached.set()
            yield 7
            yield 8
        finally:
            finalized.set()

    iterator = PrefetchIterator(source, capacity=1)
    assert reached.wait(5)
    with iterator.paused() as (ready, _):
        assert ready == [7]
        iterator.close()
    iterator.join(5)
    assert finalized.is_set()


def test_prefetch_resume_at_exhaustion_drains_ready_items_only():
    def forbidden_source():
        raise AssertionError("An exhausted checkpoint must not restart the source")

    iterator = PrefetchIterator(
        forbidden_source, capacity=2, ready=[1, 2], finished=True
    )
    assert list(iterator) == [1, 2]
    iterator.join(5)


@pytest.mark.parametrize(
    "num_workers,persistent_workers",
    [(0, False), (1, False), (1, True), (2, False), (2, True)],
)
def test_prefetch_commits_delivered_cursor_and_restores_same_loader(
    tmp_path, monkeypatch, num_workers, persistent_workers
):
    install_single_process_shm_gather(monkeypatch)
    write_tokenized_mds(tmp_path, make_sequential_rows(8, row_len=4))

    def build(num_prefetch_batches):
        return make_text_dataloader(
            tmp_path,
            seq_len=4,
            local_batch_size=2,
            shuffle=True,
            infinite=True,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            num_prefetch_batches=num_prefetch_batches,
        )

    control = build(0)
    candidate = build(3)
    try:
        control_first = next(iter(control))
        iterator = iter(candidate)
        first = next(iterator)
        state = candidate.state_dict()
        assert set(state) == {"dp_rank_0"}
        assert state["dp_rank_0"]["epoch"] == 0
        assert state["dp_rank_0"]["sample_in_epoch"] == 2

        expected_loader = build(0)
        try:
            expected_loader.load_state_dict(state)
            expected = [
                _batch_snapshot(batch)
                for batch in _take(iter(expected_loader), 6)
            ]
        finally:
            expected_loader.close()

        # Move the same loader beyond the saved epoch before restoring it.
        assert len(_take(iterator, 6)) == 6
        candidate.load_state_dict(state)
        with pytest.raises(StopIteration):
            next(iterator)
        actual = [
            _batch_snapshot(batch) for batch in _take(iter(candidate), 6)
        ]
        assert _batch_snapshot(first) == _batch_snapshot(control_first)
        assert actual == expected
    finally:
        control.close()
        candidate.close()


def test_prefetch_initializes_torch_iterator_on_consumer(tmp_path, monkeypatch):
    install_single_process_shm_gather(monkeypatch)
    write_tokenized_mds(tmp_path, make_sequential_rows(8, row_len=4))
    loader = make_text_dataloader(
        tmp_path,
        seq_len=4,
        local_batch_size=2,
        infinite=True,
        num_prefetch_batches=2,
    )
    consumer_thread = threading.get_ident()
    iterator_threads = []
    producer_thread = []
    producer_started = threading.Event()

    original_iterator = DataLoader.__iter__
    original_prefetch = loader._loader.prefetch_epoch

    def tracked_iterator(dataloader):
        iterator_threads.append(threading.get_ident())
        return original_iterator(dataloader)

    def tracked_prefetch(raw_iterator):
        producer_thread.append(threading.get_ident())
        producer_started.set()
        yield from original_prefetch(raw_iterator)

    monkeypatch.setattr(DataLoader, "__iter__", tracked_iterator)
    monkeypatch.setattr(loader._loader, "prefetch_epoch", tracked_prefetch)
    try:
        iterator = iter(loader)
        assert iterator_threads == []
        next(iterator)
        assert iterator_threads == [consumer_thread]
        assert producer_started.wait(5)
        assert producer_thread[0] != consumer_thread
    finally:
        loader.close()


def test_prefetch_worker_error_reaches_consumer(tmp_path, monkeypatch):
    install_single_process_shm_gather(monkeypatch)
    write_tokenized_mds(tmp_path, make_sequential_rows(8, row_len=4))
    loader = make_text_dataloader(
        tmp_path,
        seq_len=4,
        local_batch_size=2,
        num_workers=1,
        num_prefetch_batches=2,
    )
    dataset_type = type(loader.dataset)
    original_getitem = dataset_type.__getitem__
    seen_samples = 0

    def fail_on_third_sample(dataset, index):
        nonlocal seen_samples
        seen_samples += 1
        if seen_samples == 3:
            raise RuntimeError("fixture worker failure")
        return original_getitem(dataset, index)

    monkeypatch.setattr(dataset_type, "__getitem__", fail_on_third_sample)
    iterator = iter(loader)
    try:
        next(iterator)
        with pytest.raises(RuntimeError, match="fixture worker failure"):
            next(iterator)
    finally:
        loader.close()
    assert loader._prefetch_iterator is None


def test_prefetch_join_timeout_keeps_live_source_owned(tmp_path, monkeypatch):
    install_single_process_shm_gather(monkeypatch)
    write_tokenized_mds(tmp_path, make_sequential_rows(8, row_len=4))
    loader = make_text_dataloader(
        tmp_path,
        seq_len=4,
        local_batch_size=2,
        infinite=True,
        num_prefetch_batches=2,
    )
    started = threading.Event()
    release = threading.Event()
    original_prefetch = loader._loader.prefetch_epoch

    def blocked_prefetch(raw_iterator):
        started.set()
        release.wait(5)
        yield from original_prefetch(raw_iterator)

    monkeypatch.setattr(loader._loader, "prefetch_epoch", blocked_prefetch)
    monkeypatch.setattr(
        streaming_dataloader, "_PREFETCH_JOIN_TIMEOUT_SECONDS", 0.01
    )
    iterator = iter(loader)
    try:
        next(iterator)
        assert started.wait(5)
        active = loader._prefetch_iterator
        with pytest.raises(RuntimeError, match="in-flight read"):
            loader.close_iteration()
        assert loader._prefetch_iterator is active
        assert loader._raw_iterator is not None

        release.set()
        monkeypatch.setattr(
            streaming_dataloader, "_PREFETCH_JOIN_TIMEOUT_SECONDS", 5.0
        )
        loader.close_iteration()
        assert loader._prefetch_iterator is None
        assert not active._thread.is_alive()
    finally:
        release.set()
        monkeypatch.setattr(
            streaming_dataloader, "_PREFETCH_JOIN_TIMEOUT_SECONDS", 5.0
        )
        loader.close()


def test_validation_iterator_cleanup_allows_reiteration(tmp_path, monkeypatch):
    install_single_process_shm_gather(monkeypatch)
    write_tokenized_mds(tmp_path, make_sequential_rows(8, row_len=4))
    loader = make_text_dataloader(
        tmp_path,
        seq_len=4,
        local_batch_size=2,
        infinite=True,
        num_workers=1,
        persistent_workers=True,
        num_prefetch_batches=2,
    )
    try:
        with closing(_iter_with_loader_cleanup(loader)) as batches:
            next(batches)
        assert loader._prefetch_iterator is None
        first_again = next(iter(loader))
        assert first_again[0]["input"].shape == (2, 4)
    finally:
        loader.close()


def test_trainer_batch_generator_closes_iteration_and_trainer_closes_loader():
    class Loader:
        def __iter__(self):
            return self

        def __next__(self):
            return {"input": torch.ones((1, 1))}, torch.ones((1, 1))

        def close_iteration(self):
            self.iteration_closed = True

        def close(self):
            self.loader_closed = True

    loader = Loader()
    trainer = object.__new__(Trainer)
    trainer.metrics_processor = SimpleNamespace(data_loading_times=[], close=Mock())
    batches = trainer.batch_generator(loader)
    next(batches)
    batches.close()
    assert loader.iteration_closed
    assert not hasattr(loader, "loader_closed")

    trainer.dataloader = loader
    trainer.close()
    assert loader.loader_closed


@pytest.mark.parametrize(
    "failing_resource", ["callbacks", "checkpointer", "metrics_processor"]
)
def test_trainer_close_stops_loader_when_other_cleanup_fails(failing_resource):
    close_error = RuntimeError(f"{failing_resource} close failed")
    trainer = object.__new__(Trainer)
    trainer.callbacks = SimpleNamespace(close=Mock())
    trainer.checkpointer = SimpleNamespace(close=Mock())
    trainer.metrics_processor = SimpleNamespace(close=Mock())
    trainer.dataloader = SimpleNamespace(close=Mock())
    getattr(trainer, failing_resource).close.side_effect = close_error

    with pytest.raises(RuntimeError, match=f"{failing_resource} close failed") as raised:
        trainer.close()

    assert raised.value is close_error
    trainer.dataloader.close.assert_called_once()


def test_trainer_close_propagates_loader_timeout_after_cleanup_error():
    cleanup_error = RuntimeError("callback close failed")
    join_error = RuntimeError("prefetch source join timed out")
    trainer = object.__new__(Trainer)
    trainer.callbacks = SimpleNamespace(close=Mock(side_effect=cleanup_error))
    trainer.dataloader = SimpleNamespace(close=Mock(side_effect=join_error))

    with pytest.raises(RuntimeError, match="prefetch source join timed out") as raised:
        trainer.close()

    assert raised.value is join_error
    assert raised.value.__context__ is cleanup_error
    trainer.dataloader.close.assert_called_once()


def test_prefetch_preserves_python_numpy_and_torch_rng_draw_timing(
    tmp_path, monkeypatch
):
    install_single_process_shm_gather(monkeypatch)
    write_tokenized_mds(tmp_path, make_sequential_rows(4, row_len=4))

    def observe(num_prefetch_batches):
        random.seed(17)
        np.random.seed(17)
        torch.manual_seed(17)
        loader = make_text_dataloader(
            tmp_path,
            seq_len=4,
            local_batch_size=2,
            infinite=True,
            num_prefetch_batches=num_prefetch_batches,
        )

        def draw():
            return (
                random.random(),
                float(np.random.random()),
                float(torch.rand(())),
            )

        values = [draw()]
        iterator = iter(loader)
        values.append(draw())
        next(iterator)
        values.append(draw())
        state = loader.state_dict()
        values.append(draw())
        next(iterator)
        values.append(draw())
        next(iterator)  # Consumer enters the next finite epoch.
        values.append(draw())
        loader.load_state_dict(state)
        values.append(draw())
        next(iter(loader))
        values.append(draw())
        loader.close()
        return values

    assert observe(0) == observe(3)
