"""max_vio is computed once per step, whoever asks first.

DynamicBlockLRRescaler is a PostMutator, so by the schedule it reads
``max_vio_by_unit`` AFTER the source's own ``batch_end`` computed the window
— but the method must stay safe to call from anywhere in the step: computed
on first request, cached per step, every reader of the step sees one and the
same window, and the emitter publishes exactly those values. One ep
all-gather and loss all-reduce per step, however many callbacks ask.
"""

from types import SimpleNamespace

import torch
from torch import nn

from torchtitan.callbacks.base import ObserverState
from torchtitan.callbacks.metrics.reducers import ParallelReducer
from torchtitan.callbacks.sources.dispatcher_source import DispatcherSource
from torchtitan.config.schema.callbacks.sources.dispatcher_source import MaxVioConfig
from torchtitan.models.common.moe import MoE
from torchtitan.protocols.moe_monitoring import DispatchObservation
from tests.unit_tests.callbacks.trainer_stub import snapshot


class _SingleRankDims:
    pp_enabled = False

    def get_optional_mesh(self, name: str):
        del name
        return None


class _Dispatcher:
    dispatch_monitor = None


class _MoeLayer(MoE):
    def __init__(self):
        nn.Module.__init__(self)
        self.dispatcher = _Dispatcher()
        self.router = SimpleNamespace(num_experts=4)
        self.experts = SimpleNamespace(token_dispatcher=self.dispatcher)


def _observation(hist: list[float]) -> DispatchObservation:
    tokens = int(sum(hist))
    ids = torch.repeat_interleave(
        torch.arange(len(hist)), torch.tensor(hist, dtype=torch.long)
    ).reshape(-1, 1)
    return DispatchObservation(
        local_tokens_per_expert=torch.tensor(hist),
        global_tokens_per_local_expert=torch.tensor(hist),
        routed_input=torch.zeros(tokens, 2),
        topk_expert_ids_TK=ids,
    )


def _state(step: int) -> ObserverState:
    return snapshot(
        SimpleNamespace(
            step=step,
            model_parts=[],
            parallel_dims=_SingleRankDims(),
            gradient_accumulation_steps=2,
        )
    )


def test_max_vio_is_computed_once_per_step_whoever_asks_first():
    source = DispatcherSource(
        DispatcherSource.Config(
            max_vio=[
                MaxVioConfig(name="mb_avg", window="global", level="microbatch"),
                MaxVioConfig(name="batch_avg", window="global", level="batch"),
            ]
        )
    )
    source.reducer = ParallelReducer(parallel_dims=_SingleRankDims())
    layers = [_MoeLayer(), _MoeLayer()]
    model = nn.Module()
    model.layers = nn.ModuleDict({str(i): layer for i, layer in enumerate(layers)})
    source.fit_start(
        snapshot(
            SimpleNamespace(
                step=0,
                model_parts=[model],
                parallel_dims=_SingleRankDims(),
                gradient_accumulation_steps=2,
            )
        )
    )

    state = _state(1)
    for batch in range(2):
        state = _state(batch + 1)
        source.active = batch == 1
        source.batch_start(state)
        for forward in range(2):
            for layer in layers:
                layer.dispatcher.dispatch_monitor(
                    _observation([10.0 + batch, 20.0, 30.0, 40.0 + 10 * forward])
                )
        source.after_optimizer_step(state)

    early = source.max_vio_by_unit(state)  # the mutator's read, pre-batch_end
    assert early and all(values for values in early.values())
    assert source.max_vio_by_unit(state) is early, "not cached within the step"

    source.batch_end(state)
    published = {record.name: float(record.value) for record in source.drain()}
    for unit_name, values in early.items():
        for leaf, value in values.items():
            assert published[f"moe/token_dispatcher/{unit_name}/{leaf}"] == value
