"""The dispatch observer traces into the compiled blocks.

max_vio averages the whole logging window, so the observer must see every
step — including the compiled ones the monitoring's EagerWindow does not
cover. ``observe`` is therefore written to trace under ``fullgraph=True``:
preallocated in-place buffers, weighted-scatter masked histograms instead of
boolean indexing, the routed count read as a SymInt. These tests compile a
module whose forward reports a dispatch and pin: the numbers equal the eager
run's bit for bit, steady state never recompiles, activation checkpointing
does not double-count a dispatch (AOTAutograd applies input mutations once;
the eager recompute is filtered by ``_is_recompute``), and a forward without
a usable token mask falls back to the full histogram in both modes.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from torchtitan.callbacks.base import ObserverState
from torchtitan.callbacks.metrics.reducers import ParallelReducer
from torchtitan.callbacks.sources.dispatcher_source import (
    DispatcherSource,
    MaxVioConfig,
)
from torchtitan.models.common.moe import MoE
from torchtitan.protocols.moe_monitoring import DispatchObservation
from tests.unit_tests.callbacks.trainer_stub import snapshot

E, K, T, FORWARDS = 4, 2, 6, 2

_SERIES = [
    MaxVioConfig(name="mb_avg", window="global", level="microbatch"),
    MaxVioConfig(name="batch_avg", window="global", level="batch"),
]
_MASKED_SERIES = [
    MaxVioConfig(name="batch_avg", window="global", level="batch"),
    MaxVioConfig(name="batch_no_pad_avg", window="no_pad", level="batch"),
]


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
        # fit_start sizes the accumulators from the routing contract.
        self.router = SimpleNamespace(num_experts=E)
        self.experts = SimpleNamespace(token_dispatcher=self.dispatcher)


class _Block(nn.Module):
    """A block whose forward routes tokens and reports the dispatch."""

    def __init__(self, *, ac: bool = False):
        super().__init__()
        self.moe = _MoeLayer()
        self.lin = nn.Linear(E, E)
        self.out = nn.Linear(E, E)
        self.ac = ac

    def _forward(self, x_TD: torch.Tensor, ids_TK: torch.Tensor) -> torch.Tensor:
        hist = torch.zeros(E, device=x_TD.device).scatter_add_(
            0, ids_TK.reshape(-1), torch.ones(ids_TK.numel(), device=x_TD.device)
        )
        monitor = self.moe.dispatcher.dispatch_monitor
        if monitor is not None:
            monitor(
                DispatchObservation(
                    local_tokens_per_expert=hist,
                    global_tokens_per_local_expert=hist,
                    routed_input=x_TD.repeat_interleave(K, dim=0),
                    topk_expert_ids_TK=ids_TK,
                )
            )
        # A REAL internal activation: ``out``'s backward needs ``h_TD``, which
        # is neither an input nor the region's output, so the non-reentrant
        # checkpoint must recompute the region. Without it the AC test below
        # passes vacuously — nothing internal is saved, recompute never runs.
        h_TD = self.lin(x_TD).relu()
        return self.out(h_TD)

    def forward(self, x_TD: torch.Tensor, ids_TK: torch.Tensor) -> torch.Tensor:
        if self.ac:
            return checkpoint(self._forward, x_TD, ids_TK, use_reentrant=False)
        return self._forward(x_TD, ids_TK)


def _source(block: _Block, series=_SERIES) -> DispatcherSource:
    source = DispatcherSource(DispatcherSource.Config(max_vio=series))
    source.reducer = ParallelReducer(parallel_dims=_SingleRankDims())
    source.fit_start(
        snapshot(
            SimpleNamespace(
                step=0,
                model_parts=[block],
                parallel_dims=_SingleRankDims(),
                gradient_accumulation_steps=FORWARDS,
            )
        )
    )
    return source


def _state(step: int) -> ObserverState:
    return snapshot(SimpleNamespace(step=step, parallel_dims=_SingleRankDims()))


def _drive(source: DispatcherSource, block: _Block, *, steps: int, seed: int):
    """The real event order, logging on the last step. Returns (tags, vio)."""
    generator = torch.Generator().manual_seed(seed)
    state = _state(1)
    for step in range(1, steps + 1):
        state = _state(step)
        source.active = step == steps
        source.batch_start(state)
        stance = (
            torch.compiler.set_stance("fail_on_recompile")
            if step > 1  # buffers preallocated in fit_start: one trace, step 1
            else torch.compiler.set_stance("default")
        )
        with stance:
            for _ in range(FORWARDS):
                x = torch.randn(T, E, generator=generator)
                ids = torch.randint(0, E, (T, K), generator=generator)
                block(x, ids)
        source.after_optimizer_step(state)
    vio = source.max_vio_by_unit(state)
    source.batch_end(state)
    tags = {record.name: float(record.value) for record in source.drain()}
    return tags, vio


def test_compiled_run_matches_eager_bit_for_bit_without_recompiles():
    torch._dynamo.reset()
    eager_block = _Block()
    eager = _source(eager_block)
    compiled_block = _Block()
    compiled_block.compile(backend="eager", fullgraph=True)
    compiled = _source(compiled_block)

    eager_tags, eager_vio = _drive(eager, eager_block, steps=4, seed=7)
    compiled_tags, compiled_vio = _drive(compiled, compiled_block, steps=4, seed=7)

    assert compiled_tags == eager_tags
    assert compiled_vio == eager_vio
    assert compiled_tags["moe/token_dispatcher/moe/routed_tokens"] == T * K * FORWARDS


def test_activation_checkpointing_counts_each_dispatch_once(monkeypatch):
    torch._dynamo.reset()
    # What apply_compile sets for real runs (distributed/compile.py): without
    # it the checkpoint HOP rejects the observer's dict stores outright.
    monkeypatch.setattr(
        torch._dynamo.config, "skip_fwd_side_effects_in_bwd_under_checkpoint", True
    )
    for compile_backend in (None, "aot_eager"):
        block = _Block(ac=True)
        source = _source(block)
        runs = {"n": 0}
        if compile_backend is None:
            # Eager AC replays the python forward; prove the replay actually
            # happened, or the once-per-dispatch assert below is vacuous.
            inner = block._forward

            def counting(*args, _inner=inner, **kwargs):
                runs["n"] += 1
                return _inner(*args, **kwargs)

            block._forward = counting
        else:
            block.compile(backend=compile_backend, fullgraph=True)
        source.active = False
        source.batch_start(_state(1))
        x = torch.randn(T, E, requires_grad=True)
        ids = torch.randint(0, E, (T, K))
        block(x, ids).sum().backward()
        accum = source._accum["moe"]
        label = compile_backend or "eager"
        if compile_backend is None:
            assert runs["n"] == 2, "the checkpoint recompute never ran"
        assert int(accum.cursor) == 1, f"{label}: AC recompute double-counted"
        assert float(accum.local.sum()) == T * K, label


@pytest.mark.parametrize("delivered", ["no_mask", "wrong_size"])
def test_a_forward_without_a_usable_mask_takes_the_full_histogram(delivered):
    """K=2: the fallback mask is one entry per token, not per assignment."""
    torch._dynamo.reset()
    results = {}
    for backend in (None, "eager"):
        block = _Block()
        if backend is not None:
            block.compile(backend=backend, fullgraph=True)
        source = _source(block, series=_MASKED_SERIES)
        assert source._masks_enabled
        state = _state(1)
        source.active = True
        source.batch_start(state)
        if delivered == "wrong_size":
            # A mask built for a longer microbatch than the one dispatched.
            source._padding_mask_T = torch.ones(1, T + 3, dtype=torch.bool)
            source._prefix_mask_T = torch.ones(1, T + 3, dtype=torch.bool)
        generator = torch.Generator().manual_seed(3)
        expected = torch.zeros(E)
        for _ in range(FORWARDS):
            x = torch.randn(T, E, generator=generator)
            ids = torch.randint(0, E, (T, K), generator=generator)
            expected += torch.bincount(ids.reshape(-1), minlength=E).float()
            block(x, ids)
        accum = source._accum["moe"]
        assert torch.equal(accum.sums["no_pad"], expected)
        assert torch.equal(accum.sums["no_pad"], accum.sums["global"])
        source.after_optimizer_step(state)
        results[backend] = source.max_vio_by_unit(state)["moe"]
    assert results["eager"] == results[None]
    assert results[None]["batch_no_pad_avg"] == results[None]["batch_avg"]
