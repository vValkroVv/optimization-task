"""Checkpoint and shutdown contracts for the bounded CPU producer."""

from threading import Event

import pytest

from torchtitan.datasets.streaming.core.prefetch import PrefetchIterator


def test_prefetch_preserves_order_and_capacity():
    reached = [Event() for _ in range(4)]

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


def test_prefetch_checkpoint_freezes_cursor_and_ready_items():
    cursor = 0
    produced = Event()

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


def test_prefetch_error_reaches_consumer_after_ready_items():
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


def test_prefetch_close_releases_a_full_queue_and_source():
    reached = Event()
    finalized = Event()

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


def test_prefetch_resume_at_exhaustion_only_drains_ready_items():
    def forbidden_source():
        raise AssertionError("An exhausted checkpoint must not restart the source")

    iterator = PrefetchIterator(
        forbidden_source, capacity=2, ready=[1, 2], finished=True
    )
    assert list(iterator) == [1, 2]
    iterator.join(5)
