"""Pipeline metadata forwards must not enter dispatcher monitoring windows."""

from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.pipelining import PipelineStage, Schedule1F1B
from torch.utils.checkpoint import checkpoint

from torchtitan.callbacks.metrics.reducers import ParallelReducer
from torchtitan.callbacks.metrics.sample import MetricConfig, StatOpConfig
from torchtitan.callbacks.sources.dispatcher_source import (
    DispatcherSource,
    _UnitWindow,
    MaxVioConfig,
    MaxVioLevel,
    MaxVioWindow,
)
from torchtitan.distributed.pipeline_context import (
    is_pipeline_metadata_inference,
    pipeline_metadata_inference,
)
from torchtitan.distributed.pipeline_parallel import MetadataSilentStage
from torchtitan.protocols.moe_monitoring import DispatchObservation
from tests.unit_tests.callbacks.trainer_stub import snapshot


class _ObservedStage(torch.nn.Module):
    """A differentiable CPU stage emitting a known dispatch per microbatch."""

    def __init__(self, observe, *, checkpointed):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)
        self.observe = observe
        self.checkpointed = checkpointed

    def _forward(self, x):
        self.observe(
            DispatchObservation(
                local_tokens_per_expert=torch.tensor([2, 2]),
                global_tokens_per_local_expert=torch.tensor([1, 0]),
                routed_input=x[:1],
                topk_expert_ids_TK=torch.tensor([[0, -1], [-1, -1]]),
                requested_topk_scores=torch.tensor([[0.75, 0.25], [0.5, 0.5]]),
                capacity_keep_mask=torch.tensor([[True, False], [False, False]]),
            )
        )
        return self.linear(x).tanh()

    def forward(self, x):
        if self.checkpointed:
            return checkpoint(self._forward, x, use_reentrant=False)
        return self._forward(x)


def _run_schedule(rank, *, checkpointed, monitor_first_step, corrected):
    parallel_dims = SimpleNamespace(get_optional_mesh=lambda name: None)
    source = DispatcherSource(
        DispatcherSource.Config(
            max_vio=[
                MaxVioConfig(
                    name="max_vio_microbatch",
                    window=MaxVioWindow.global_,
                    level=MaxVioLevel.microbatch,
                )
            ],
            global_tokens_metrics=[
                MetricConfig(name="max", pipeline=[StatOpConfig(op="max")])
            ],
        )
    )
    source.reducer = ParallelReducer(parallel_dims=parallel_dims)
    unit = SimpleNamespace(
        name="layer", dispatcher=object(), router=SimpleNamespace(num_experts=2)
    )
    source.units = [unit]
    source._needs_sums = source._needs_rows = {MaxVioWindow.global_}
    source._forwards_per_batch = 4
    # The reference intentionally includes the probe in its counters.
    source._forwards_checked = not corrected
    accum = source._allocate(unit, ep_size=1, device="cpu")
    source._accum[unit.name] = accum
    source._window[unit.name] = _UnitWindow(
        sums={},
        rows={
            MaxVioWindow.global_: accum.rows[MaxVioWindow.global_].new_zeros(2, 4, 2)
        },
    )
    torch.manual_seed(42 + rank)
    model = _ObservedStage(source._observe(accum), checkpointed=checkpointed)
    stage_type = MetadataSilentStage if corrected else PipelineStage
    stage = stage_type(model, rank, 2, torch.device("cpu"))
    schedule = Schedule1F1B(stage, n_microbatches=2, loss_fn=torch.nn.MSELoss())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    numerics = []
    for step in (1, 2):
        state = snapshot(SimpleNamespace(parallel_dims=parallel_dims), step=step)
        source.active = monitor_first_step or step == 2
        source.batch_start(state)
        optimizer.zero_grad()
        losses = []
        for accumulation in range(2):
            inputs = torch.arange(12, dtype=torch.float32).reshape(4, 3) / 8
            inputs = inputs + step + accumulation
            if rank == 0:
                schedule.step(inputs)
            else:
                microbatch_losses = []
                schedule.step(target=torch.ones_like(inputs), losses=microbatch_losses)
                losses.extend(microbatch_losses)
        gradients = [p.grad.detach().clone() for p in model.parameters()]
        optimizer.step()
        numerics.extend(gradients)
        numerics.append(
            torch.linalg.vector_norm(torch.cat([g.flatten() for g in gradients]))
        )
        numerics.extend(p.detach().clone() for p in model.parameters())
        numerics.extend(loss.detach().clone() for loss in losses)
        source.after_optimizer_step(state)
        if corrected:
            # Four real forwards: two schedule calls with two microbatches each.
            torch.testing.assert_close(
                accum.sums[MaxVioWindow.global_],
                torch.tensor([4, 0], dtype=torch.int64),
                rtol=0,
                atol=0,
            )
            window = source._window["layer"]
            rows = window.batches(window.rows[MaxVioWindow.global_])
            expected_rows = 8 if step == 2 and not monitor_first_step else 4
            assert sum(block.shape[0] for block in rows) == expected_rows
        source.batch_end(state)
        metrics = {r.name: r.value for r in source.drain()}
        if corrected and source.active:
            prefix = "moe/token_dispatcher/layer/"
            expected = {
                "token_drop/requested_assignments": 16,
                "token_drop/retained_assignments": 4,
                "token_drop/dropped_assignments": 12,
                "token_drop/assignment_drop_fraction": 0.75,
                "token_drop/fully_dropped_token_fraction": 0.5,
                "token_drop/dropped_score_mass_fraction": 0.625,
                "global_tokens_per_local_expert/max": 4,
                "max_vio_microbatch": 0.5,
                "routed_tokens": 4,
            }
            for leaf, value in expected.items():
                assert metrics[prefix + leaf] == value, leaf
        elif not source.active:
            assert not metrics
    return numerics


def _pipeline_worker(rank, store_path, checkpointed):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{store_path}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        for monitor_first_step in (True, False):
            reference = _run_schedule(
                rank,
                checkpointed=checkpointed,
                monitor_first_step=monitor_first_step,
                corrected=False,
            )
            actual = _run_schedule(
                rank,
                checkpointed=checkpointed,
                monitor_first_step=monitor_first_step,
                corrected=True,
            )
            assert len(actual) == len(reference)
            for value, expected in zip(actual, reference, strict=True):
                torch.testing.assert_close(value, expected, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("checkpointed", [False, True])
def test_pipeline_probe_excluded_from_dispatcher_monitoring(tmp_path, checkpointed):
    mp.spawn(
        _pipeline_worker,
        args=(str(tmp_path / "store"), checkpointed),
        nprocs=2,
        join=True,
    )


def test_metadata_context_restored_after_nested_failure():
    assert not is_pipeline_metadata_inference()
    with pipeline_metadata_inference():
        with pytest.raises(ValueError, match="probe failed"):
            with pipeline_metadata_inference():
                raise ValueError("probe failed")
        assert is_pipeline_metadata_inference()
    assert not is_pipeline_metadata_inference()
