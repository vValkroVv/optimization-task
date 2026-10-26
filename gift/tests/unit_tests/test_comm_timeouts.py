"""Communication timeouts must cover startup and streaming epoch preparation."""

import time
from contextlib import nullcontext
from datetime import timedelta
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from torchtitan.components.dataloader import BaseDataLoader, ReplicatedBatchDataLoader
from torchtitan.config import CommConfig, ParallelismConfig
from tests.unit_tests.config_helpers import load_experiment_config
from torchtitan.distributed import utils as dist_utils
from torchtitan.trainer import Trainer


@pytest.mark.parametrize("field", ["init_timeout_seconds", "train_timeout_seconds"])
@pytest.mark.parametrize("override", [None, 120, 7200])
def test_timeout_config(field, override):
    args = []
    if override is not None:
        args += [f"--comm.{field}", str(override)]
    config = load_experiment_config("llama3_debug.yaml", *args)
    expected = {"init_timeout_seconds": 3600, "train_timeout_seconds": 1800}
    if override is not None:
        expected[field] = override
    for name, value in expected.items():
        assert getattr(config.comm, name) == value


def _check_group_timeouts(rank, world_size, store_path, pp, tp, cp, timeout):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{store_path}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    try:
        trainer = Trainer.__new__(Trainer)
        trainer.config = load_experiment_config("llama3_debug.yaml")
        trainer.config.comm = CommConfig(init_timeout_seconds=timeout)
        trainer.config.parallelism = ParallelismConfig(
            pipeline_parallel_degree=pp,
            tensor_parallel_degree=tp,
            context_parallel_degree=cp,
        )
        with (
            mock.patch("torchtitan.distributed.parallel_dims.device_type", "cpu"),
            mock.patch(
                "torchtitan.distributed.utils.init_distributed", return_value=world_size
            ),
            mock.patch(
                "torchtitan.distributed.utils.device_module",
                SimpleNamespace(current_device=lambda: 0, synchronize=lambda: None),
            ),
        ):
            parallel_dims = trainer.init_distributed()
            groups = [dist.distributed_c10d._get_default_group()]
            groups.extend(
                mesh.get_group()
                for mesh in parallel_dims.get_all_one_dimensional_meshes().values()
            )
            for seconds in (timeout, CommConfig().train_timeout_seconds):
                if seconds != timeout:
                    dist_utils.set_pg_timeouts(
                        timedelta(seconds=seconds), parallel_dims
                    )
                for group in groups:
                    if dist.get_backend(group) == "fake":
                        continue
                    backend = group._get_backend(torch.device("cpu"))
                    assert backend.options._timeout == timedelta(seconds=seconds)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "world_size,pp,tp,cp,timeout",
    [(1, 1, 1, 1, 3600), (4, 2, 2, 1, 3600), (4, 1, 1, 2, 7200)],
)
def test_trainer_sets_world_and_mesh_timeouts(
    tmp_path, world_size, pp, tp, cp, timeout
):
    mp.spawn(
        _check_group_timeouts,
        args=(world_size, str(tmp_path / "store"), pp, tp, cp, timeout),
        nprocs=world_size,
        join=True,
    )


class _SlowEpochLoader(BaseDataLoader):
    def __init__(self, delay):
        self.delay = delay

    def __iter__(self):
        for epoch in range(3):
            if epoch:
                time.sleep(self.delay)
            tokens = torch.tensor([[epoch, epoch + 1]])
            yield {"input": tokens}, tokens + 1

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass


def _check_epoch_wait(rank, store_path, old_timeout, loaded_step, scale):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{store_path}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        trainer = Trainer.__new__(Trainer)
        trainer.config = load_experiment_config("llama3_debug.yaml")
        trainer.config.comm = (
            CommConfig(train_timeout_seconds=100) if old_timeout else CommConfig()
        )
        trainer.config.parallelism = ParallelismConfig(pipeline_parallel_degree=2)
        trainer.config.training.steps = loaded_step + 3
        trainer.config.profiler = SimpleNamespace(
            build=lambda **kwargs: nullcontext(mock.Mock())
        )
        trainer.step = loaded_step
        trainer.model_parts = []
        trainer.sharding_backend = mock.Mock()
        trainer.checkpointer = mock.Mock()
        trainer.checkpointer.save.return_value = False
        trainer.callbacks = mock.Mock()
        trainer.metrics_processor = mock.Mock(data_loading_times=[])
        trainer.gc_handler = mock.Mock()
        received = []

        def train_step(data_iterator):
            inputs, labels = next(data_iterator)
            epoch = len(received)
            expected = torch.tensor([[epoch, epoch + 1]])
            torch.testing.assert_close(inputs["input"], expected, rtol=0, atol=0)
            torch.testing.assert_close(labels, expected + 1, rtol=0, atol=0)
            received.append(epoch)

        trainer.train_step = train_step
        with (
            mock.patch("torchtitan.distributed.parallel_dims.device_type", "cpu"),
            mock.patch("torchtitan.distributed.utils.init_distributed", return_value=2),
            mock.patch(
                "torchtitan.distributed.utils.device_module",
                SimpleNamespace(current_device=lambda: 0, synchronize=lambda: None),
            ),
            # Scale both budgets and the owner delay; retain the real Trainer transition.
            mock.patch(
                "torchtitan.trainer.timedelta",
                side_effect=lambda *, seconds: timedelta(seconds=seconds * scale),
            ),
        ):
            trainer.parallel_dims = trainer.init_distributed()
            pp_mesh = trainer.parallel_dims.get_mesh("pp")
            trainer.dataloader = ReplicatedBatchDataLoader(
                _SlowEpochLoader(150 * scale) if rank == 0 else None,
                mesh=pp_mesh,
                device=torch.device("cpu"),
            )
            with mock.patch.object(
                dist_utils, "set_pg_timeouts", wraps=dist_utils.set_pg_timeouts
            ) as set_timeouts:
                if old_timeout:
                    message = (
                        "[Tt]imed out" if rank == 1 else "[Cc]onnection closed by peer"
                    )
                    with pytest.raises(RuntimeError, match=message):
                        trainer.train()
                    assert received == [0]
                else:
                    trainer.train()
                    assert received == [0, 1, 2]
                set_timeouts.assert_called_once_with(
                    timeout=timedelta(
                        seconds=trainer.config.comm.train_timeout_seconds * scale
                    ),
                    parallel_dims=trainer.parallel_dims,
                )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("loaded_step", [0, 100])
@pytest.mark.parametrize("old_timeout", [True, False], ids=["old-timeout", "default"])
def test_epoch_wait_after_first_train_step(tmp_path, old_timeout, loaded_step):
    mp.spawn(
        _check_epoch_wait,
        args=(str(tmp_path / "store"), old_timeout, loaded_step, 0.02),
        nprocs=2,
        join=True,
    )
