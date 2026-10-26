"""ModellingSource publishes the SwiGLU clamp counts of a log step.

The source hands every clamping FeedForward / GroupedExperts a count slot at
batch start, sums what each forward of the batch adds, records
``swiglu_clip/{fqn}/{w1,w3}_{count,share}`` at batch end and takes the slots
back, so an unmonitored step counts nothing.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tests.unit_tests.callbacks.trainer_stub import snapshot
from torchtitan.callbacks.metrics.reducers import ParallelReducer
from torchtitan.callbacks.sources.modelling_source import ModellingSource
from torchtitan.distributed.pipeline_parallel import metadata_probe
from torchtitan.models.common.ffn import FeedForward
from torchtitan.models.common.nn_modules import Linear

DIM, HIDDEN, LIMIT = 8, 16, 0.5


class _SingleRankDims:
    def get_optional_mesh(self, name: str):
        del name
        return None


def _ffn(limit: float) -> FeedForward:
    ffn = FeedForward.Config(
        dim=DIM,
        hidden_dim=HIDDEN,
        w1=Linear.Config(in_features=DIM, out_features=HIDDEN, bias=False),
        w2=Linear.Config(in_features=HIDDEN, out_features=DIM, bias=False),
        w3=Linear.Config(in_features=DIM, out_features=HIDDEN, bias=False),
        swiglu_limit=limit,
    ).build()
    for p in ffn.parameters():
        nn.init.normal_(p)
    return ffn


def _model() -> nn.Module:
    torch.manual_seed(0)
    model = nn.Module()
    model.layers = nn.ModuleDict({"0": nn.Module(), "1": nn.Module()})
    model.layers["0"].feed_forward = _ffn(LIMIT)
    # Does not clamp, so it has nothing to count and is not a target.
    model.layers["1"].feed_forward = _ffn(0.0)
    return model


def _state(model: nn.Module, step: int):
    return snapshot(
        SimpleNamespace(step=step, model_parts=[model], parallel_dims=_SingleRankDims())
    )


def _fitted(model: nn.Module) -> ModellingSource:
    source = ModellingSource(
        ModellingSource.Config(capture_forward=False, capture_swiglu_clips=True)
    )
    source.reducer = ParallelReducer(parallel_dims=_SingleRankDims())
    source.fit_start(_state(model, step=0))
    return source


def _clip_records(source: ModellingSource) -> dict[str, float]:
    return {
        r.name: float(r.value)
        for r in source.drain()
        if r.name.startswith("swiglu_clip/")
    }


def test_only_clamping_modules_are_targets():
    source = _fitted(_model())
    assert [fqn for fqn, _ in source._clip_targets] == ["layers.0.feed_forward"]


@pytest.mark.parametrize("probe", [False, True])
def test_a_log_step_publishes_the_batch_counts_and_shares(probe):
    model = _model()
    ffn = model.layers["0"].feed_forward
    source = _fitted(model)
    microbatches = [3 * torch.randn(4, DIM) for _ in range(2)]

    source.active = True
    source.batch_start(_state(model, step=5))
    assert ffn.swiglu_clip_counts.dtype == torch.int64
    if probe:
        with metadata_probe(model):
            ffn(microbatches[0])
        assert not source._clip_accum
    for x in microbatches:
        ffn(x)
    source.batch_end(_state(model, step=5))

    gate = torch.cat([ffn.w1(x) for x in microbatches])
    up = torch.cat([ffn.w3(x) for x in microbatches])
    w1_count = float(gate.ge(LIMIT).sum())
    w3_count = float((up.le(-LIMIT) | up.ge(LIMIT)).sum())
    prefix = "swiglu_clip/layers.0.feed_forward"
    # Recorded in float32.
    assert _clip_records(source) == pytest.approx(
        {
            f"{prefix}/w1_count": w1_count,
            f"{prefix}/w1_share": w1_count / gate.numel(),
            f"{prefix}/w3_count": w3_count,
            f"{prefix}/w3_share": w3_count / gate.numel(),
        }
    )
    # The slot is taken back: the next forward counts nothing.
    assert ffn.swiglu_clip_counts is None


def test_an_unmonitored_step_hands_out_no_slot_and_records_nothing():
    model = _model()
    ffn = model.layers["0"].feed_forward
    source = _fitted(model)

    source.active = False
    source.batch_start(_state(model, step=6))
    assert ffn.swiglu_clip_counts is None
    ffn(3 * torch.randn(4, DIM))
    source.batch_end(_state(model, step=6))

    assert _clip_records(source) == {}
