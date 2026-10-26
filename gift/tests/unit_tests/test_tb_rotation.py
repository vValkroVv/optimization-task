# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TensorBoardLogger rotates its event file by size.

Past ``rotate_mib`` the writer is closed and the next write lazily opens a
fresh event file. The properties the rotation exists for: no logged point is
lost across rotations, closed files never change again (that is what keeps
incremental transfers cheap), and a failing size probe degrades to a warning
instead of killing training.
"""

import logging
import os
import time

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from torchtitan.components.metrics import TensorBoardLogger

BLOB = "x" * (2 * 1024 * 1024)  # one text event that alone exceeds rotate_mib=1


def _event_files(path):
    return sorted(path.glob("events.out.tfevents.*"))


def _scalar_steps(path, tag="loss"):
    """All steps of ``tag`` merged from every event file in the directory."""
    accumulator = EventAccumulator(str(path))
    accumulator.Reload()
    return sorted(event.step for event in accumulator.Scalars(tag))


def _wait_for_bytes_on_disk(path, min_bytes, timeout=10.0):
    """The event writer drains its queue on a background thread; wait for it."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        files = _event_files(path)
        if files and files[-1].stat().st_size >= min_bytes:
            return
        time.sleep(0.01)
    raise AssertionError(f"event file never reached {min_bytes} bytes on disk")


def _log_one_rotation_cycle(tb, path, first_step, num_scalars=4):
    """A blob past the size limit, then scalar logs; the first one rotates."""
    tb.log_text("blob", BLOB, step=first_step)
    _wait_for_bytes_on_disk(path, len(BLOB))
    for step in range(first_step, first_step + num_scalars):
        tb.log({"loss": float(step)}, step=step)
    return first_step + num_scalars


def test_no_step_is_lost_across_rotations(tmp_path):
    tb = TensorBoardLogger(str(tmp_path), rotate_mib=1)
    step = 0
    for _ in range(3):
        step = _log_one_rotation_cycle(tb, tmp_path, step)
    tb.close()
    assert len(_event_files(tmp_path)) >= 3
    assert _scalar_steps(tmp_path) == list(range(step))


def test_rotation_happens_on_the_next_log_call(tmp_path):
    tb = TensorBoardLogger(str(tmp_path), rotate_mib=1)
    tb.log({"loss": 0.0}, step=0)
    assert len(_event_files(tmp_path)) == 1  # tiny file: no rotation
    tb.log_text("blob", BLOB, step=0)
    _wait_for_bytes_on_disk(tmp_path, len(BLOB))
    tb.log({"loss": 1.0}, step=1)  # the probe sees the oversized file, closes
    tb.log({"loss": 2.0}, step=2)  # ...and this write opens a fresh one
    tb.close()
    assert len(_event_files(tmp_path)) == 2
    assert _scalar_steps(tmp_path) == [0, 1, 2]


def test_closed_files_never_change_after_rotation(tmp_path):
    tb = TensorBoardLogger(str(tmp_path), rotate_mib=1)
    step = _log_one_rotation_cycle(tb, tmp_path, 0)
    closed = {f: f.read_bytes() for f in _event_files(tmp_path)[:-1]}
    assert closed  # the cycle rotated at least once
    for _ in range(2):
        step = _log_one_rotation_cycle(tb, tmp_path, step)
    tb.close()
    for file, snapshot in closed.items():
        assert file.read_bytes() == snapshot


def test_rotation_disabled_keeps_a_single_complete_file(tmp_path):
    tb = TensorBoardLogger(str(tmp_path), rotate_mib=0)
    step = 0
    for _ in range(2):
        step = _log_one_rotation_cycle(tb, tmp_path, step)
    tb.close()
    assert len(_event_files(tmp_path)) == 1
    assert _scalar_steps(tmp_path) == list(range(step))


def test_probe_failure_degrades_to_a_warning(tmp_path, monkeypatch, caplog):
    tb = TensorBoardLogger(str(tmp_path), rotate_mib=1)

    def broken_getmtime(path):
        raise OSError("stale NFS handle")

    monkeypatch.setattr(os.path, "getmtime", broken_getmtime)
    step = 0
    with caplog.at_level(logging.WARNING):
        for _ in range(2):
            step = _log_one_rotation_cycle(tb, tmp_path, step)
    monkeypatch.undo()
    tb.close()
    assert any("size probe failed" in message for message in caplog.messages)
    assert len(_event_files(tmp_path)) == 1  # never rotated, never crashed
    assert _scalar_steps(tmp_path) == list(range(step))  # nothing lost
