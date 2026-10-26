# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import scipy
import torch
from torch.utils.data import DataLoader

from composer.callbacks import ActivationMonitor
import composer.callbacks.activation_monitor as activation_monitor_module
from composer.callbacks.activation_monitor import compute_activation_stats, compute_kurtosis
from composer.loggers import InMemoryLogger
from composer.optim import DecoupledAdamW
from composer.trainer import Trainer
from tests.common.datasets import RandomClassificationDataset
from tests.common.models import SimpleModel


def test_activation_monitor():

    with pytest.warns(Warning):
        activation_monitor = ActivationMonitor(interval='1ba')
    in_memory_logger = InMemoryLogger()  # track the logged metrics in the in_memory_logger
    model = SimpleModel()
    trainer = Trainer(
        model=model,
        callbacks=activation_monitor,
        loggers=in_memory_logger,
        train_dataloader=DataLoader(RandomClassificationDataset()),
        optimizers=DecoupledAdamW(model.parameters()),
        max_duration='3ba',
    )
    trainer.fit()

    num_max_calls = len(in_memory_logger.data['activations/max/_output.0'])
    num_average_calls = len(in_memory_logger.data['activations/average/_output.0'])

    assert num_max_calls == num_average_calls
    assert num_max_calls == 3

    # Checking to make sure existing keys are in the memory logger dict
    assert 'activations/l2_norm/module.0_input.0' in in_memory_logger.data.keys()
    assert 'activations/max/module.0_input.0' in in_memory_logger.data.keys()
    assert 'activations/average/module.0_input.0' in in_memory_logger.data.keys()
    assert 'activations/kurtosis/module.0_input.0' in in_memory_logger.data.keys()


def test_kurtosis_computation():
    x = torch.randn((2, 3, 4))
    res = compute_kurtosis(x)
    scipy_res = scipy.stats.kurtosis(x.numpy(), axis=-1, fisher=False).mean()
    assert np.allclose(res.numpy(), scipy_res, rtol=1e-4)


class _RecordingLogger:
    destinations = []

    def __init__(self):
        self.metrics = None

    def log_metrics(self, metrics):
        self.metrics = metrics


class _FakeHandle:
    def wait(self):
        return None


class _RecordingCollectives:
    """Records make_log's collectives and plays back canned peer data."""

    def __init__(self, monkeypatch, *, world_size, peer_keys=None, peer_value=7.0):
        self.key_gathers = []          # (group, ranks spanned, payload)
        self.value_gathers = []        # group per gather
        self.valid = None
        self._peer_keys = peer_keys or {}
        self._peer_value = peer_value
        self._values_done = False

        monkeypatch.setattr(activation_monitor_module.dist, 'get_world_size', lambda: world_size)

        def all_gather_object(output, obj, group=None):
            self.key_gathers.append((group, len(output), list(obj)))
            output[0] = list(obj)
            for i in range(1, len(output)):
                output[i] = self._peer_keys.get(i, list(obj))

        def all_gather_into_tensor(output, input, group=None, async_op=False):
            # row 0 is this rank; peers report every key as present, so a key only a
            # peer has still reaches the statistic
            self.value_gathers.append(group)
            n = output.numel() // input.numel()
            peer = torch.ones_like(input) if self._values_done else torch.full_like(input, self._peer_value)
            output.copy_(torch.cat([input.clone()] + [peer] * (n - 1)))
            if self._values_done:
                self.valid = input.clone()
            self._values_done = True
            return _FakeHandle()

        monkeypatch.setattr(torch.distributed, 'all_gather_object', all_gather_object)
        monkeypatch.setattr(torch.distributed, 'all_gather_into_tensor', all_gather_into_tensor)


def _monitor(metrics):
    monitor = ActivationMonitor(interval='10ba', only_log_wandb=False, stat_list=['amax'])
    monitor.metrics = metrics
    return monitor


def test_make_log_unions_keys_and_gathers_world_wide(monkeypatch):
    monitor = _monitor({'activations_amax/local': torch.tensor([2.0])})
    logger = _RecordingLogger()
    col = _RecordingCollectives(
        monkeypatch, world_size=2, peer_keys={1: ['activations_amax/remote']}
    )

    monitor.make_log(logger, step=1)

    assert col.key_gathers == [(None, 2, ['activations_amax/local'])]
    assert col.value_gathers == [None, None]
    assert set(logger.metrics) == {'activations_amax/local', 'activations_amax/remote'}
    assert monitor.metrics == {}


def test_make_log_participates_with_no_local_metrics(monkeypatch):
    """A rank whose experts got no tokens still joins every collective."""
    monitor = _monitor({})
    logger = _RecordingLogger()
    monkeypatch.setattr(torch.distributed, 'is_available', lambda: True)
    monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: True)
    monkeypatch.setattr(torch.distributed, 'get_backend', lambda: 'gloo')
    col = _RecordingCollectives(
        monkeypatch, world_size=2, peer_keys={1: ['activations_amax/remote']}
    )

    monitor.make_log(logger, step=1)

    assert col.key_gathers == [(None, 2, [])]
    assert col.value_gathers == [None, None]
    assert col.valid.tolist() == [0.0]          # no value here, so flagged absent
    assert logger.metrics == {'activations_amax/remote': 7.0}
    assert monitor.metrics == {}


def test_make_log_skips_a_key_no_rank_reported(monkeypatch):
    monitor = _monitor({})
    logger = _RecordingLogger()
    monkeypatch.setattr(torch.distributed, 'is_available', lambda: True)
    monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: True)
    monkeypatch.setattr(torch.distributed, 'get_backend', lambda: 'gloo')

    monkeypatch.setattr(activation_monitor_module.dist, 'get_world_size', lambda: 2)

    def all_gather_object(output, obj, group=None):
        output[0] = list(obj)
        output[1] = ['activations_amax/ghost']

    def all_gather_into_tensor(output, input, group=None, async_op=False):
        output.zero_()                           # nobody flags the key as present
        return _FakeHandle()

    monkeypatch.setattr(torch.distributed, 'all_gather_object', all_gather_object)
    monkeypatch.setattr(torch.distributed, 'all_gather_into_tensor', all_gather_into_tensor)

    monitor.make_log(logger, step=1)

    assert logger.metrics is None
    assert monitor.metrics == {}


def test_make_log_stat_covers_the_whole_world(monkeypatch):
    """The largest value lives on rank 3; the logged amax must be the global max."""
    monitor = _monitor({'activations_amax/shared': torch.tensor([1.0])})
    logger = _RecordingLogger()

    world_size = 4
    monkeypatch.setattr(activation_monitor_module.dist, 'get_world_size', lambda: world_size)

    def all_gather_object(output, obj, group=None):
        for i in range(len(output)):
            output[i] = list(obj)

    calls = []

    def all_gather_into_tensor(output, input, group=None, async_op=False):
        assert output.numel() == world_size
        calls.append(input.clone())
        if len(calls) == 1:
            output.copy_(torch.tensor([1.0, 2.0, 3.0, 42.0], dtype=input.dtype))
        else:
            output.copy_(torch.ones(world_size, dtype=input.dtype))
        return _FakeHandle()

    monkeypatch.setattr(torch.distributed, 'all_gather_object', all_gather_object)
    monkeypatch.setattr(torch.distributed, 'all_gather_into_tensor', all_gather_into_tensor)

    monitor.make_log(logger, step=1)

    assert logger.metrics['activations_amax/shared'] == 42.0


def test_make_log_single_rank_needs_no_collectives(monkeypatch):
    monitor = _monitor({'activations_amax/local': torch.tensor([5.0])})
    logger = _RecordingLogger()
    col = _RecordingCollectives(monkeypatch, world_size=1)

    monitor.make_log(logger, step=1)

    assert col.key_gathers == []
    assert col.value_gathers == []
    assert logger.metrics['activations_amax/local'] == 5.0


def test_add_metrics_skips_empty_tensors():
    monitor = ActivationMonitor(interval='10ba', only_log_wandb=False, stat_list=['amax'])
    monitor.is_forward = True
    metrics = {}

    monitor.add_metrics(metrics, 'module', '_output.0', torch.empty((0, 8)))

    assert metrics == {}
    stat = compute_activation_stats(torch.empty((0, 8)), 'amax')
    assert stat.shape == (1,) and torch.isnan(stat).item()
