"""include/exclude patterns select whole FQN segments and say what they did.

Pinned against the failure modes of the bare ``re.search`` they replace:
``exclude=['layers.1']`` used to drop layers 1, 10 AND 11 (substring match
plus the unescaped dot) with not a line in the log, and a typo'd pattern was
indistinguishable from one that cut a third of the model. Patterns are now
anchored to dot-segment boundaries, ``^`` pins to the top level (the way to
keep ``mtp.layers.1`` while dropping ``layers.1``), a configured filter logs
its kept/total counts, a no-match pattern warns, and DispatcherSource filters
by the same unit names instead of having no filter at all.
"""

import logging
from types import SimpleNamespace

from torch import nn

from torchtitan.callbacks.metrics.reducers import ParallelReducer
from torchtitan.callbacks.sources.dispatcher_source import DispatcherSource
from torchtitan.callbacks.sources.modelling_source import ModellingSource
from torchtitan.callbacks.sources.optimizer_source import OptimizerSource
from torchtitan.models.common.moe import MoE
from tests.unit_tests.callbacks.trainer_stub import snapshot

NUM_LAYERS, NUM_MTP = 12, 2


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


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = nn.ModuleDict(
            {"wq": nn.Linear(4, 4), "wq_norm": nn.LayerNorm(4)}
        )
        self.moe = _MoeLayer()


def _model() -> nn.Module:
    model = nn.Module()
    model.layers = nn.ModuleDict({str(i): _Block() for i in range(NUM_LAYERS)})
    mtp = nn.Module()
    mtp.layers = nn.ModuleDict({str(i): _Block() for i in range(NUM_MTP)})
    model.mtp = mtp
    return model


def _fitted(**config_kwargs) -> ModellingSource:
    source = ModellingSource(ModellingSource.Config(**config_kwargs))
    source.reducer = ParallelReducer(parallel_dims=_SingleRankDims())
    source.fit_start(
        snapshot(
            SimpleNamespace(
                step=0, model_parts=[_model()], parallel_dims=_SingleRankDims()
            )
        )
    )
    return source


def _target_fqns(source: ModellingSource) -> set[str]:
    return {fqn for fqn, _ in source._targets}


def test_a_layer_index_does_not_swallow_its_ten_prefixed_siblings():
    fqns = _target_fqns(_fitted(exclude_patterns=["layers.1"]))
    assert not any(f == "layers.1" or f.startswith("layers.1.") for f in fqns)
    assert any(f.startswith("layers.10.") for f in fqns)
    assert any(f.startswith("layers.11.") for f in fqns)


def test_patterns_match_whole_segments_not_substrings():
    fqns = _target_fqns(_fitted(exclude_patterns=["wq"]))
    assert not any(f.endswith(".wq") for f in fqns)
    assert any(f.endswith(".wq_norm") for f in fqns), "wq must not swallow wq_norm"


def test_a_bare_layer_pattern_also_hits_the_mtp_head_and_an_anchor_pins_it():
    bare = _target_fqns(_fitted(exclude_patterns=[r"layers\.1"]))
    assert not any(f.startswith("mtp.layers.1.") for f in bare)
    anchored = _target_fqns(_fitted(exclude_patterns=[r"^layers\.1"]))
    assert any(f.startswith("mtp.layers.1.") for f in anchored)
    assert not any(f.startswith("layers.1.") for f in anchored)


def test_include_filters_moe_units_and_the_log_says_so(caplog):
    from torchtitan.config.schema.callbacks.metrics.sample import (
        MetricConfig,
        ReduceConfig,
        ReduceStageConfig,
        StatOpConfig,
    )
    from torchtitan.config.schema.callbacks.sources.modelling_source import (
        MoECaptureConfig,
    )

    moe = MoECaptureConfig(
        token_metrics=[
            MetricConfig(
                name="sum",
                pipeline=[StatOpConfig(op="sum")],
                reduce=ReduceConfig(stages=[ReduceStageConfig(group="dp", op="sum")]),
            )
        ]
    )
    with caplog.at_level(logging.INFO):
        source = _fitted(include_patterns=["attention"], moe=moe)
    assert source._units == []
    assert any("0 of 14 MoE units" in message for message in caplog.messages), (
        caplog.messages
    )


def test_a_pattern_that_matches_nothing_warns(caplog):
    with caplog.at_level(logging.WARNING):
        source = _fitted(exclude_patterns=["layer_1"])
    assert any("matched none" in m and "layer_1" in m for m in caplog.messages)
    assert len(source._targets) > 0


def test_dispatcher_source_filters_by_the_same_unit_names():
    source = DispatcherSource(
        DispatcherSource.Config(exclude_patterns=[r"layers\.1", "mtp"])
    )
    source.reducer = ParallelReducer(parallel_dims=_SingleRankDims())
    model = _model()
    source.fit_start(
        snapshot(
            SimpleNamespace(
                step=0, model_parts=[model], parallel_dims=_SingleRankDims()
            )
        )
    )
    names = {unit.name for unit in source.units}
    assert "layers.1.moe" not in names and not any("mtp" in n for n in names)
    assert "layers.10.moe" in names
    # The excluded unit's dispatcher is left uninstrumented entirely.
    assert model.layers["1"].moe.dispatcher.dispatch_monitor is None
    assert model.layers["10"].moe.dispatcher.dispatch_monitor is not None


def test_optimizer_source_reports_its_parameter_filter(caplog):
    source = OptimizerSource(OptimizerSource.Config(exclude_patterns=["layers.1"]))
    source.reducer = ParallelReducer(parallel_dims=_SingleRankDims())
    with caplog.at_level(logging.INFO):
        source.fit_start(
            snapshot(
                SimpleNamespace(
                    step=0, model_parts=[_model()], parallel_dims=_SingleRankDims()
                )
            )
        )
    kept = {tag for tag, _param in source._named_params}
    assert not any(f.startswith("layers.1.") for f in kept)
    assert any(f.startswith("layers.10.") for f in kept)
    assert any("parameters" in m and "keep" in m for m in caplog.messages)


def test_no_filter_stays_silent(caplog):
    with caplog.at_level(logging.INFO):
        _fitted()
    assert not any("include/exclude" in m for m in caplog.messages)
