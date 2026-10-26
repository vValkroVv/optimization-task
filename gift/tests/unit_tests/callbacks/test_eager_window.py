"""Logged steps run the compiled blocks eagerly; every other step stays compiled.

A compiled block is opaque to the monitoring: Dynamo traces the hooks of the
block's submodules once and does not guard on them, so a hook attached to a
submodule of an already compiled block -- forward or backward -- is never
called, and a hook on the block itself is traced into the graph, where
``is_compiling()`` is True and ``_record`` drops the capture.
``MonitoringCallback`` therefore opens an ``EagerWindow`` on the steps it
reads -- only when a source hooks the model at all -- the block's compiled
slot is cleared for the step and restored after it, with the compiled code
kept, so the way back costs no recompile. The pipeline metadata probe takes
the same window, so no observer ever meets a compiled probe.
"""

from types import SimpleNamespace

import torch
from torch import nn

from torchtitan.callbacks.base import ObserverState
from torchtitan.callbacks.builtin.monitoring import MonitoringCallback
from torchtitan.callbacks.builtin.seq_length_warmup import SeqLengthWarmup
from torchtitan.callbacks.metrics.reducers import ParallelReducer
from torchtitan.callbacks.sources.modelling_source import (
    ModellingSource,
    MoECaptureConfig,
)
from torchtitan.distributed.backends.fsdp2 import FSDP2Backend
from torchtitan.distributed.compile import EagerWindow
from torchtitan.distributed.pipeline_parallel import in_metadata_probe, metadata_probe
from torchtitan.models.common.moe import MoE
from tests.unit_tests.callbacks.trainer_stub import snapshot


class _SingleRankDims:
    pp_enabled = False
    seq_len_divisor = 1

    def get_optional_mesh(self, name: str):
        del name
        return None


class _Experts(nn.Module):
    token_dispatcher = None

    def named_weight_slices(self):
        return {}

    def grouped_param_axes(self):
        return []


class _MoeLayer(MoE):
    """MoE-typed, with real modules for the router and the experts: discovery
    is by type, the router hook needs a module, the weights walk needs
    parameters."""

    def __init__(self):
        nn.Module.__init__(self)
        self.router = nn.Linear(4, 4)
        self.router.num_experts = 4
        self.experts = _Experts()


def _compiled_block() -> nn.Module:
    block = nn.Linear(4, 4)
    block.compile(backend="eager")  # Dynamo without Inductor
    return block


def _state(step: int, model: nn.Module) -> ObserverState:
    return snapshot(
        SimpleNamespace(step=step, model_parts=[model], parallel_dims=_SingleRankDims())
    )


def _modelling(config: ModellingSource.Config, model: nn.Module) -> ModellingSource:
    source = ModellingSource(config)
    source.reducer = ParallelReducer(parallel_dims=_SingleRankDims())
    source.fit_start(_state(0, model))
    return source


def test_window_lists_exactly_the_compiled_modules():
    model = nn.Sequential(_compiled_block(), nn.Linear(4, 4))
    assert len(EagerWindow([model], owner="t")) == 1
    assert len(EagerWindow([nn.Sequential(nn.Linear(4, 4))], owner="t")) == 0
    assert len(EagerWindow((), owner="t")) == 0


def test_hooks_fire_inside_the_window_and_the_compiled_code_survives_it():
    torch._dynamo.reset()
    block = _compiled_block()
    x = torch.randn(2, 4, requires_grad=True)
    block(x)  # compiled with no hooks in the trace
    compiled_slot = block._compiled_call_impl
    assert compiled_slot is not None

    # The order of a logged step: hooks go on, the window opens, the step
    # runs, the window closes, the hooks come off.
    calls: list[str] = []
    handles = [
        block.register_forward_hook(lambda m, i, o: calls.append("fwd")),
        block.register_full_backward_hook(lambda m, gi, go: calls.append("bwd")),
    ]
    window = EagerWindow([nn.Sequential(block)], owner="t")

    window.open()
    assert block._compiled_call_impl is None
    block(x).sum().backward()
    window.close()
    assert calls == ["fwd", "bwd"]
    assert block._compiled_call_impl is compiled_slot
    for handle in handles:
        handle.remove()

    # Closing twice is harmless, and the cached compiled code is still the
    # one Dynamo produced before the window: no recompile on the way back.
    window.close()
    with torch.compiler.set_stance("fail_on_recompile"):
        block(x)


def test_monitoring_opens_the_window_on_logged_steps_only():
    torch._dynamo.reset()
    block = _compiled_block()
    model = nn.Sequential(block)
    callback = MonitoringCallback(
        MonitoringCallback.Config(
            log_interval=2,
            sources={"modelling": ModellingSource.Config(capture_backward=True)},
        ),
        sharding_backend=FSDP2Backend(),
    )

    callback.fit_start(_state(0, model))
    assert len(callback._eager_window) == 1

    callback.batch_start(_state(1, model))  # not a logged step
    assert block._compiled_call_impl is not None
    callback.batch_end(_state(1, model))

    callback.batch_start(_state(2, model))  # logged step
    assert block._compiled_call_impl is None
    callback.batch_end(_state(2, model))
    assert block._compiled_call_impl is not None

    callback.batch_start(_state(4, model))
    assert block._compiled_call_impl is None
    callback.close()  # teardown mid-window restores the slot too
    assert block._compiled_call_impl is not None


def test_weight_only_monitoring_hooks_nothing_and_keeps_the_step_compiled():
    """Weights and the bias are read off the module after the step: no hook,
    so nothing needs the blocks eager. The router hook goes on only for the
    token/score families that come out of the router's forward."""
    torch._dynamo.reset()
    block = _compiled_block()
    model = nn.Sequential(block, _MoeLayer())

    weights_only = ModellingSource.Config(
        capture_forward=False,
        moe=MoECaptureConfig(expert_weights=True, bias_metrics=[]),
    )
    assert not weights_only.uses_module_hooks
    source = _modelling(weights_only, model)
    source.active = True
    source.batch_start(_state(2, model))
    assert source._handles == [], "weight-only monitoring must register no hook"
    source.close()

    routed = ModellingSource.Config(
        capture_forward=False, moe=MoECaptureConfig(per_expert_tokens=True)
    )
    assert routed.uses_module_hooks
    source = _modelling(routed, model)
    source.active = True
    source.batch_start(_state(2, model))
    assert len(source._handles) == 1, "the router hook alone"
    source.close()

    callback = MonitoringCallback(
        MonitoringCallback.Config(log_interval=2, sources={"modelling": weights_only}),
        sharding_backend=FSDP2Backend(),
    )
    callback.fit_start(_state(0, model))
    assert len(callback._eager_window) == 0
    callback.batch_start(_state(2, model))
    assert block._compiled_call_impl is not None, "no hooks, so compile stays on"
    callback.batch_end(_state(2, model))


def test_two_owners_share_the_slot_and_the_first_to_close_does_not_end_it():
    """A logged step and the metadata probe can overlap; the slot ends with the last."""
    block = nn.Linear(4, 4)
    block._compiled_call_impl = lambda *a, **k: None
    model = nn.Sequential(block)
    probe = EagerWindow([model], owner="metadata_probe")
    logged = EagerWindow([model], owner="monitoring")

    logged.open()
    probe.open()
    assert block._compiled_call_impl is None
    probe.close()  # the probe ends inside the logged step
    assert block._compiled_call_impl is None, "the logged step still needs eager"
    logged.close()
    assert block._compiled_call_impl is not None

    # A holder that re-opens every step is still released by one close.
    for _ in range(3):
        logged.open()
    assert block._compiled_call_impl is None
    logged.close()
    assert block._compiled_call_impl is not None


def test_the_warmup_runs_compiled_and_only_the_logged_step_is_eager():
    """warmup_steps=3 with log_interval=3: the warmup never touches the slot."""
    block = nn.Linear(4, 4)
    block._compiled_call_impl = lambda *a, **k: None
    model = nn.Sequential(block)

    callback = MonitoringCallback(
        MonitoringCallback.Config(
            log_interval=3, sources={"modelling": ModellingSource.Config()}
        ),
        sharding_backend=FSDP2Backend(),
    )
    warmup = SeqLengthWarmup(SeqLengthWarmup.Config(min_seq_len=2, warmup_steps=3))
    trainer = SimpleNamespace(
        parallel_dims=_SingleRankDims(),
        config=SimpleNamespace(training=SimpleNamespace(seq_len=8)),
        model_parts=[model],
    )

    warmup.fit_start(SimpleNamespace(trainer=trainer))
    callback.fit_start(_state(0, model))

    for step, eager in ((1, False), (2, False), (3, True), (4, False), (6, True)):
        callback.batch_start(_state(step, model))
        warmup.after_dataloader(
            SimpleNamespace(step=step, input_dict=None, labels=None, trainer=trainer)
        )
        assert (block._compiled_call_impl is None) is eager, f"step {step}"
        callback.batch_end(_state(step, model))
        assert block._compiled_call_impl is not None


def test_the_metadata_probe_runs_the_blocks_eager_without_a_recompile():
    torch._dynamo.reset()
    block = _compiled_block()
    model = nn.Sequential(block)
    x = torch.randn(2, 4)
    model(x)  # compiled

    assert not in_metadata_probe()
    with metadata_probe(model):
        assert in_metadata_probe()
        assert block._compiled_call_impl is None, "the probe never traces"
        model(x)
    assert not in_metadata_probe()
    assert block._compiled_call_impl is not None

    # The compiled code is still the one from before the probe.
    with torch.compiler.set_stance("fail_on_recompile"):
        model(x)
