"""The dispatch observer keeps per-batch aggregates, not per-forward lists.

The observer runs on every step — max_vio averages the whole logging window —
so whatever it retains lives for a full log_interval. Per-forward clones made
that ~50k live little tensors between log steps on a production shape (58
layers x 8 microbatches x log_interval=100). The window now holds one summed
histogram per batch (for the "batch" level) and one stacked block per batch
(for the per-forward levels), the batch in progress lives in preallocated
in-place buffers — which is also what lets the observer trace into the
compiled blocks (test_dispatcher_compile.py) — and unmasked window variants
collapse into "global" instead of being stored three times. These tests drive
the real observer through the real event sequence and pin both the retention
counts and the published values.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from torchtitan.callbacks.metrics.reducers import ParallelReducer
from torchtitan.callbacks.sources.dispatcher_source import (
    DispatcherSource,
    MaxVioConfig,
)
from torchtitan.distributed.pipeline_parallel import metadata_probe
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
    """A minimal MoE-typed double: discovery finds MoE layers by type and
    names the unit by its module path."""

    def __init__(self):
        nn.Module.__init__(self)
        self.dispatcher = _Dispatcher()
        # fit_start sizes the accumulators from the routing contract.
        self.router = SimpleNamespace(num_experts=4)
        self.experts = SimpleNamespace(token_dispatcher=self.dispatcher)


_SERIES = [
    MaxVioConfig(name="mb_avg", window="global", level="microbatch"),
    MaxVioConfig(name="batch_avg", window="global", level="batch"),
    MaxVioConfig(name="batch_no_pad_avg", window="no_pad", level="batch"),
]


def _source(
    max_vio=None, forwards=2, log_interval=4
) -> tuple[DispatcherSource, list[_MoeLayer]]:
    source = DispatcherSource(
        DispatcherSource.Config(max_vio=_SERIES if max_vio is None else max_vio)
    )
    source.reducer = ParallelReducer(parallel_dims=_SingleRankDims())
    # MonitoringCallback assigns this before fit_start; it sizes the window ring.
    source.log_interval = log_interval
    layers = [_MoeLayer(), _MoeLayer()]
    model = nn.Module()
    model.layers = nn.ModuleDict({str(i): layer for i, layer in enumerate(layers)})
    state = snapshot(
        SimpleNamespace(
            step=0,
            model_parts=[model],
            parallel_dims=_SingleRankDims(),
            # Sizes the per-forward row blocks; the log step checks the
            # dispatch count against it.
            gradient_accumulation_steps=forwards,
        )
    )
    source.fit_start(state)
    return source, layers


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


def _run_batches(source, layers, batches, *, forwards=2, active_last=True):
    """Drive the real event order: batch_start -> dispatches -> optimizer."""
    state = snapshot(SimpleNamespace(step=0, parallel_dims=_SingleRankDims()))
    for batch in range(batches):
        state = snapshot(
            SimpleNamespace(step=batch + 1, parallel_dims=_SingleRankDims())
        )
        source.active = active_last and batch == batches - 1
        source.batch_start(state)
        for forward in range(forwards):
            for layer in layers:
                hist = [10.0 + batch, 20.0, 30.0, 40.0 + 10 * forward]
                layer.dispatcher.dispatch_monitor(_observation(hist))
        source.after_optimizer_step(state)
    return state


def _window_buffers(source) -> int:
    return sum(len(w.sums) + len(w.rows) for w in source._window.values())


def _window_filled(source) -> int:
    return sum(w.filled for w in source._window.values())


def test_window_holds_preallocated_ring_buffers_not_growing_lists():
    source, layers = _source(forwards=8, log_interval=10)
    before = _window_buffers(source)
    _run_batches(source, layers, batches=10, forwards=8)
    # 2 units x (one sums ring + one rows ring): the tensor count is fixed at
    # fit_start whatever the batch count. The per-batch clone list held
    # 2 x 10 x 2 = 40 tensors here; the per-forward design 480.
    # 2 units x (global sums + no_pad sums + global rows) — fixed at fit_start.
    assert before == _window_buffers(source) == 6
    assert _window_filled(source) == 20  # 10 batches folded per unit


def test_windows_are_stored_by_need_not_per_series():
    # global is always summed; the configured no_pad batch series adds ONE
    # masked sum buffer; nothing asked for masked rows.
    source, layers = _source()
    _run_batches(source, layers, batches=1)
    for window in source._window.values():
        assert set(window.sums) == {"global", "no_pad"}
        assert set(window.rows) == {"global"}


def test_batch_buffers_accumulate_in_place_and_zero_at_batch_start():
    # The buffers are what the compiled blocks trace, so they accumulate
    # whether or not the step is logged, and batch_start zeroes them in place
    # instead of rebinding — the same tensors, batch after batch.
    source, layers = _source()
    state = snapshot(SimpleNamespace(step=1, parallel_dims=_SingleRankDims()))
    source.active = False
    source.batch_start(state)
    layers[0].dispatcher.dispatch_monitor(_observation([1.0, 2.0, 3.0, 4.0]))
    layers[0].dispatcher.dispatch_monitor(_observation([1.0, 2.0, 3.0, 4.0]))
    accum = source._accum["layers.0"]
    assert torch.equal(accum.local, torch.tensor([2.0, 4.0, 6.0, 8.0]))
    assert int(accum.cursor) == 2 and int(accum.routed) == 20
    buffers = (accum.local, accum.sums["global"], accum.rows["global"])
    source.batch_start(state)
    assert source._accum["layers.0"] is accum
    assert (accum.local, accum.sums["global"], accum.rows["global"]) == buffers
    assert int(accum.cursor) == 0 and float(accum.local.sum()) == 0.0


def test_the_dispatch_path_never_allocates(monkeypatch):
    # Clones happen at the fold — one per unit per window key per batch; the
    # per-forward path only mutates buffers preallocated in fit_start,
    # because it is the part that runs inside the compiled blocks.
    source, layers = _source(forwards=8)
    state = snapshot(SimpleNamespace(step=1, parallel_dims=_SingleRankDims()))
    source.active = False
    source.batch_start(state)

    counter = {"n": 0}
    original = torch.Tensor.clone

    def counting(tensor, *args, **kwargs):
        counter["n"] += 1
        return original(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "clone", counting)
    for _ in range(8):
        layers[0].dispatcher.dispatch_monitor(_observation([1.0, 2.0, 3.0, 4.0]))
    assert counter["n"] == 0, "the per-forward path must not allocate"
    source.after_optimizer_step(state)
    # The fold copies into ring buffers preallocated at fit_start: no clones
    # anywhere on the batch path.
    assert counter["n"] == 0


def test_max_vio_values_match_hand_arithmetic():
    source, layers = _source(log_interval=2)
    state = _run_batches(source, layers, batches=2, forwards=2)

    def vio(hist):
        t = torch.tensor(hist)
        return float((t.max() - t.mean()) / t.mean().clamp_min(1.0))

    per_unit = source.max_vio_by_unit(state)
    for unit_name in ("layers.0", "layers.1"):
        got = per_unit[unit_name]
        # microbatch: violation of each of the four forwards, averaged.
        forwards = [
            [10.0, 20.0, 30.0, 40.0],
            [10.0, 20.0, 30.0, 50.0],
            [11.0, 20.0, 30.0, 40.0],
            [11.0, 20.0, 30.0, 50.0],
        ]
        assert got["mb_avg"] == pytest.approx(
            sum(vio(f) for f in forwards) / 4, rel=1e-6
        )
        # batch: the two forwards of each batch summed first, then averaged.
        batches = [
            [20.0, 40.0, 60.0, 90.0],
            [22.0, 40.0, 60.0, 90.0],
        ]
        assert got["batch_avg"] == pytest.approx(
            sum(vio(b) for b in batches) / 2, rel=1e-6
        )
        # no mask is set, so the no_pad series reads the global window.
        assert got["batch_no_pad_avg"] == got["batch_avg"]


def test_window_is_consumed_by_the_log_step():
    source, layers = _source()
    state = _run_batches(source, layers, batches=3)
    source.max_vio_by_unit(state)
    # Consumption resets the ring; the buffers themselves stay allocated.
    assert _window_filled(source) == 0
    assert _window_buffers(source) == 6


def test_validation_dispatches_do_not_touch_the_window(monkeypatch):
    # The trainer validates AFTER batch_end, so the fold and the log-step
    # reads are behind it; whatever eval dispatches add to the buffers is
    # zeroed at the next batch_start. The monitor slot stays installed
    # throughout — swapping the closure would change the identity the
    # compiled blocks traced — and eval dispatches allocate nothing.
    source, layers = _source()
    state = _run_batches(source, layers, batches=2, forwards=2)

    clones = {"n": 0}
    original = torch.Tensor.clone

    def counting(tensor, *args, **kwargs):
        clones["n"] += 1
        return original(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "clone", counting)
    observers = [layer.dispatcher.dispatch_monitor for layer in layers]
    source.validation_start(state)
    assert [layer.dispatcher.dispatch_monitor for layer in layers] == observers
    for _ in range(50):
        for layer in layers:
            layer.dispatcher.dispatch_monitor(
                _observation([999.0, 999.0, 999.0, 999.0])
            )
    source.validation_end(state)
    assert [layer.dispatcher.dispatch_monitor for layer in layers] == observers
    monkeypatch.undo()
    assert clones["n"] == 0, "validation dispatches must allocate nothing"

    polluted = source.max_vio_by_unit(state)

    clean_source, clean_layers = _source()
    clean_state = _run_batches(clean_source, clean_layers, batches=2, forwards=2)
    clean = clean_source.max_vio_by_unit(clean_state)
    assert polluted == clean, "eval data leaked into max_vio"

    # And the next batch starts from zero: nothing of the eval dispatches
    # survives batch_start.
    next_state = snapshot(SimpleNamespace(step=99, parallel_dims=_SingleRankDims()))
    source.batch_start(next_state)
    for accum in source._accum.values():
        assert int(accum.cursor) == 0 and float(accum.local.sum()) == 0.0


def test_batch_end_publishes_sums_and_routed_tokens():
    source, layers = _source()
    state = _run_batches(source, layers, batches=2, forwards=2)
    source.batch_end(state)
    got = {record.name: float(record.value) for record in source.drain()}
    prefix = "moe/token_dispatcher/layers.0"
    # 4 dispatches of (11+20+30+40) and (11+20+30+50) tokens in the last batch.
    assert got[f"{prefix}/routed_tokens"] == 101.0 + 111.0
    assert f"{prefix}/mb_avg" in got
    assert (
        got["moe/token_dispatcher/dispatcher_sum/batch_avg"]
        == got[f"{prefix}/batch_avg"]
    )


def test_masks_flow_from_the_batch_to_the_masked_windows():
    """no_pad excludes attention-masked tokens; hand arithmetic must agree."""
    source, layers = _source(forwards=1, log_interval=1)
    state = snapshot(SimpleNamespace(step=1, parallel_dims=_SingleRankDims()))
    source.active = True
    source.batch_start(state)
    # A 10-token microbatch: last 4 tokens are padding, first 2 are prefix.
    hist = [1.0, 2.0, 3.0, 4.0]  # T = 10, ids sorted by expert
    attention = torch.tensor([1, 1, 1, 1, 1, 1, 0, 0, 0, 0])
    labels = torch.tensor([-100, -100, 5, 5, 5, 5, 5, 5, 5, 5])
    source.after_dataloader(
        snapshot(
            SimpleNamespace(step=1, parallel_dims=_SingleRankDims()),
            input_dict={"attention_mask": attention},
            labels=labels,
        )
    )
    source.microbatch_start(state)
    for layer in layers:
        layer.dispatcher.dispatch_monitor(_observation(hist))
    source.after_optimizer_step(state)

    vio = source.max_vio_by_unit(state)["layers.0"]

    def hand(h):
        t = torch.tensor(h)
        return float((t.max() - t.mean()) / t.mean().clamp_min(1.0))

    # ids по гистограмме: [0, 1,1, 2,2,2, 3,3,3,3]; без последних 4 токенов
    # остаётся [1, 2, 3, 0], без префикса дополнительно минус токены 0 и 1.
    assert vio["batch_avg"] == pytest.approx(hand([1.0, 2.0, 3.0, 4.0]), rel=1e-6)
    assert vio["batch_no_pad_avg"] == pytest.approx(
        hand([1.0, 2.0, 3.0, 0.0]), rel=1e-6
    )


def test_masks_route_to_their_pp_chunks():
    """The schedule chunks a microbatch in order; each forward gets its row."""

    class _PPDims(_SingleRankDims):
        pp_enabled = True

    source = DispatcherSource(DispatcherSource.Config(max_vio=_SERIES))
    source.reducer = ParallelReducer(parallel_dims=_PPDims())
    source.log_interval = 1
    layer = _MoeLayer()
    model = nn.Module()
    model.layers = nn.ModuleDict({"0": layer})
    source.fit_start(
        snapshot(
            SimpleNamespace(
                step=0,
                model_parts=[model],
                parallel_dims=_PPDims(),
                gradient_accumulation_steps=1,
                config=SimpleNamespace(
                    training=SimpleNamespace(local_batch_size=4),
                    parallelism=SimpleNamespace(pipeline_parallel_microbatch_size=2),
                ),
            )
        )
    )
    assert source._forwards_per_batch == 2

    state = snapshot(SimpleNamespace(step=1, parallel_dims=_PPDims()))
    source.active = True
    source.batch_start(state)
    # 20 токенов = 2 pp-чанка по 10: первый без паддинга, во втором 4 падовых.
    attention = torch.cat(
        [torch.ones(10), torch.tensor([1.0, 1, 1, 1, 1, 1, 0, 0, 0, 0])]
    )
    source.after_dataloader(
        snapshot(
            SimpleNamespace(step=1, parallel_dims=_PPDims()),
            input_dict={"attention_mask": attention},
            labels=torch.full((20,), 5),
        )
    )
    source.microbatch_start(state)
    hist = [1.0, 2.0, 3.0, 4.0]
    layer.dispatcher.dispatch_monitor(_observation(hist))  # pp-чанк 0
    layer.dispatcher.dispatch_monitor(_observation(hist))  # pp-чанк 1
    source.after_optimizer_step(state)

    vio = source.max_vio_by_unit(state)["layers.0"]

    def hand(h):
        t = torch.tensor(h)
        return float((t.max() - t.mean()) / t.mean().clamp_min(1.0))

    # global: сумма чанков [2,4,6,8]; no_pad: чанк 1 теряет 4 токена → [2,4,6,4].
    assert vio["batch_avg"] == pytest.approx(hand([2.0, 4.0, 6.0, 8.0]), rel=1e-6)
    assert vio["batch_no_pad_avg"] == pytest.approx(
        hand([2.0, 4.0, 6.0, 4.0]), rel=1e-6
    )


def _pp_source(forwards_per_batch=2):
    """A PP source whose derivation expects ``forwards_per_batch`` dispatches."""

    class _PPDims(_SingleRankDims):
        pp_enabled = True

    source = DispatcherSource(
        DispatcherSource.Config(
            max_vio=[MaxVioConfig(name="mb_avg", window="global", level="microbatch")]
        )
    )
    source.reducer = ParallelReducer(parallel_dims=_PPDims())
    source.log_interval = 2
    layer = _MoeLayer()
    model = nn.Module()
    model.layers = nn.ModuleDict({"0": layer})
    source.fit_start(
        snapshot(
            SimpleNamespace(
                step=0,
                model_parts=[model],
                parallel_dims=_PPDims(),
                gradient_accumulation_steps=1,
                config=SimpleNamespace(
                    training=SimpleNamespace(local_batch_size=2 * forwards_per_batch),
                    parallelism=SimpleNamespace(pipeline_parallel_microbatch_size=2),
                ),
            )
        )
    )
    assert source._forwards_per_batch == forwards_per_batch
    return source, layer, _PPDims


def _run_batch(source, layer, step, dims, dispatches, *, probe=False):
    state = snapshot(SimpleNamespace(step=step, parallel_dims=dims()))
    source.active = False
    source.batch_start(state)
    if probe:
        # The schedule's metadata probe: a real dispatch on garbage inputs.
        with metadata_probe(nn.Module()):
            layer.dispatcher.dispatch_monitor(
                _observation([999.0, 999.0, 999.0, 999.0])
            )
    for _ in range(dispatches):
        layer.dispatcher.dispatch_monitor(_observation([1.0, 2.0, 3.0, 4.0]))
    source.after_optimizer_step(state)


def test_the_metadata_probe_leaves_no_trace_whatever_the_step_number():
    # The PP probe rides on the PROCESS's first batch — step 1, or step 101
    # after a resume — so it cannot be keyed off the step; the observer skips
    # it by the flag, and the counters hold exactly the real forwards.
    source, layer, dims = _pp_source()
    _run_batch(source, layer, step=101, dims=dims, dispatches=2, probe=True)
    accum = source._accum["layers.0"]
    real = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert int(accum.cursor) == 2 and int(accum.routed) == 20
    assert torch.equal(accum.local, 2 * real)
    assert torch.equal(accum.sums["global"], 2 * real)
    assert torch.equal(accum.rows["global"], torch.stack([real, real]))
    window = source._window["layers.0"]
    assert window.filled == 1
    assert torch.equal(window.rows["global"][0], torch.stack([real, real]))
    # The derivation was checked on that batch, and only there.
    _run_batch(source, layer, step=102, dims=dims, dispatches=3)


def test_a_wrong_derivation_raises_on_the_first_batch_probe_or_not():
    source, layer, dims = _pp_source()
    with pytest.raises(ValueError, match="forwards per batch"):
        _run_batch(source, layer, step=1, dims=dims, dispatches=3, probe=True)


def test_without_pp_the_very_first_batch_validates():
    source, layers = _source(forwards=2)
    with pytest.raises(ValueError, match="forwards per batch"):
        _run_batches(source, layers, batches=1, forwards=3)
