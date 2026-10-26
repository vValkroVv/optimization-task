"""Drop monitoring counts assignments separately from tokens and score mass."""

from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh

from torchtitan.callbacks.sources.dispatcher_source import (
    DispatcherSource,
    _UnitWindow,
    MaxVioConfig,
    MaxVioLevel,
    MaxVioWindow,
)
from torchtitan.protocols.moe_monitoring import DispatchObservation
from tests.unit_tests.callbacks.trainer_stub import snapshot


def test_drop_metrics_ignore_recompute_and_validation(monkeypatch):
    source = DispatcherSource(DispatcherSource.Config())
    source.active = True
    source._needs_sums = {MaxVioWindow.global_}
    source.units = [
        SimpleNamespace(
            name="layer", dispatcher=object(), router=SimpleNamespace(num_experts=2)
        )
    ]
    state = snapshot(
        SimpleNamespace(
            parallel_dims=SimpleNamespace(get_optional_mesh=lambda name: None)
        )
    )
    source.batch_start(state)
    scores = torch.tensor([[0.75, 0.25], [0.5, 0.5]])
    keep = torch.tensor([[True, False], [False, False]])
    observation = DispatchObservation(
        local_tokens_per_expert=torch.tensor([2, 2]),
        global_tokens_per_local_expert=torch.tensor([1, 0]),
        routed_input=torch.empty(1, 3),
        topk_expert_ids_TK=torch.tensor([[0, -1], [-1, -1]]),
        requested_topk_scores=scores,
        capacity_keep_mask=keep,
    )
    source._accum["layer"] = source._allocate(source.units[0], ep_size=1, device="cpu")
    observe = source._observe(source._accum["layer"])
    observe(observation)
    monkeypatch.setattr(source, "_is_recompute", lambda: True)
    observe(observation)
    monkeypatch.setattr(source, "_is_recompute", lambda: False)
    source.validation_start(state)
    observe(observation)
    source.validation_end(state)
    source.batch_end(state)
    metrics = {r.name.rsplit("/", 1)[-1]: r.value for r in source.drain()}
    assert metrics["requested_assignments"] == 4
    assert metrics["retained_assignments"] == 1
    assert metrics["dropped_assignments"] == 3
    assert metrics["assignment_drop_fraction"] == 0.75
    assert metrics["fully_dropped_token_fraction"] == 0.5
    assert metrics["dropped_score_mass_fraction"] == 0.625
    hist = source._masked_histogram(observation, torch.tensor([True, True]))
    torch.testing.assert_close(hist.float(), torch.tensor([1.0, 0.0]))
    source.batch_start(state)
    assert all(accum.drop is None for accum in source._accum.values())


def _source_with_drop_totals(names, totals, parallel_dims):
    source = DispatcherSource(DispatcherSource.Config())
    source.units = [
        SimpleNamespace(
            name=name, dispatcher=object(), router=SimpleNamespace(num_experts=2)
        )
        for name in names
    ]
    source._needs_sums = {MaxVioWindow.global_}
    source.active = True
    state = snapshot(SimpleNamespace(parallel_dims=parallel_dims))
    source.batch_start(state)
    # Include one observed layer without dropping and one unobserved layer.
    for unit in source.units:
        accum = source._allocate(unit, ep_size=1, device="cpu")
        source._accum[unit.name] = accum
        source._window[unit.name] = _UnitWindow(sums={}, rows={})
        if unit.name != "unobserved":
            accum.cursor.fill_(1)
        accum.drop = totals.get(unit.name)
    return source, state


def _assert_drop_metrics(source, metrics, name, totals):
    requested, retained, tokens, lost, mass, lost_mass = totals
    expected = {
        "requested_assignments": requested,
        "retained_assignments": retained,
        "dropped_assignments": requested - retained,
        "assignment_drop_fraction": (requested - retained) / max(requested, 1),
        "fully_dropped_token_fraction": lost / max(tokens, 1),
        "dropped_score_mass_fraction": lost_mass / mass if mass else 0.0,
    }
    for leaf, value in expected.items():
        assert metrics[f"{source.config.namespace}/{name}/token_drop/{leaf}"] == value


@pytest.mark.parametrize("ep_size", [1, 2])
def test_drop_metrics_reduce_and_copy_all_layers_once(monkeypatch, ep_size):
    names = ["layers.2", "layers.0", "empty", "no_drop", "unobserved"]
    local = {
        "empty": torch.zeros(6, dtype=torch.float64),
        "layers.0": torch.tensor([4, 4, 2, 0, 2, 0], dtype=torch.float64),
        "layers.2": torch.tensor([4, 1, 2, 1, 2, 1.25], dtype=torch.float64),
    }
    saved = {name: values.clone() for name, values in local.items()}
    peer = torch.tensor(
        [[8, 6, 4, 1, 4, 0.5], [4, 0, 2, 2, 2, 2], [0, 0, 0, 0, 0, 0]],
        dtype=torch.float64,
    )
    group = object()
    ep_mesh = SimpleNamespace(size=lambda: ep_size, get_group=lambda: group)
    dims = SimpleNamespace(
        get_optional_mesh=lambda axis: ep_mesh if axis == "ep" else None
    )
    source, state = _source_with_drop_totals(names, local, dims)

    # The separate histogram all-gathers are outside the drop-counter batching.
    monkeypatch.setattr(
        "torchtitan.callbacks.sources.dispatcher_source._gather_concat",
        lambda vector, *args, **kwargs: vector,
    )
    collective = Mock(side_effect=lambda tensor, group: tensor.add_(peer))
    monkeypatch.setattr(dist, "all_reduce", collective)
    copy_shapes = []
    original_tolist = torch.Tensor.tolist

    def record_tolist(tensor):
        copy_shapes.append(tuple(tensor.shape))
        return original_tolist(tensor)

    monkeypatch.setattr(torch.Tensor, "tolist", record_tolist)
    source.batch_end(state)
    assert copy_shapes == [(3, 6), (5, 2)]
    assert collective.call_count == int(ep_size > 1)
    if ep_size > 1:
        assert collective.call_args.kwargs["group"] is group
    metrics = {record.name: record.value for record in source.drain()}
    for i, name in enumerate(names[:3]):
        total = saved[name] + (peer[i] if ep_size > 1 else 0)
        _assert_drop_metrics(source, metrics, name, original_tolist(total))
        torch.testing.assert_close(
            source._accum[name].drop, saved[name], rtol=0, atol=0
        )
    assert not any("/no_drop/token_drop/" in name for name in metrics)
    assert not any("/unobserved/" in name for name in metrics)
    source.batch_start(state)
    assert all(accum.drop is None for accum in source._accum.values())


@pytest.mark.parametrize("active", [False, True])
def test_no_drop_metric_reduction_when_inactive_or_no_policy(monkeypatch, active):
    dims = SimpleNamespace(get_optional_mesh=lambda axis: None)
    totals = {} if active else {"layer": torch.ones(6, dtype=torch.float64)}
    source, state = _source_with_drop_totals(["layer"], totals, dims)
    source.active = active
    unexpected = Mock(
        side_effect=AssertionError("No drop counters should be copied or reduced")
    )
    monkeypatch.setattr(dist, "all_reduce", unexpected)
    original_tolist = torch.Tensor.tolist

    def check_tolist(tensor):
        if tensor.shape[-1] == 6:
            unexpected()
        return original_tolist(tensor)

    monkeypatch.setattr(torch.Tensor, "tolist", check_tolist)
    source.batch_end(state)
    unexpected.assert_not_called()
    assert not any("/token_drop/" in record.name for record in source.drain())


def _batched_drop_ep_pp_worker(rank, store_path):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{store_path}",
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=60),
    )
    try:
        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("pp", "ep"))
        stage = mesh["pp"].get_local_rank()
        ep_rank = mesh["ep"].get_local_rank()
        dims = SimpleNamespace(
            get_optional_mesh=lambda axis: mesh[axis] if axis in ("pp", "ep") else None
        )
        # Different stage layouts, including a stage with no drop policy. Within
        # one EP group, observation order differs and one source is empty.
        for num_drop_layers in (2 - stage, 1 - stage):
            names = [f"stage{stage}.layer{i}" for i in range(num_drop_layers)]
            base = torch.tensor([4, 1, 2, 1, 2, 1.25], dtype=torch.float64)
            totals = {
                name: base * (i + 1) * (stage + 1) * ep_rank
                for i, name in enumerate(names)
            }
            if ep_rank:
                totals = dict(reversed(list(totals.items())))
            source, state = _source_with_drop_totals(
                names + ["no_drop", "unobserved"], totals, dims
            )
            source.config.max_vio = [
                MaxVioConfig(
                    name="max_vio",
                    window=MaxVioWindow.global_,
                    level=MaxVioLevel.microbatch,
                )
            ]
            # Stage 0 has violation 1, stage 1 has violation 0: PP must still
            # average over observed layers, not over stages of unequal sizes.
            source._window.update(
                {
                    name: _UnitWindow(
                        sums={},
                        rows={
                            MaxVioWindow.global_: torch.tensor([[[2.0, 2.0 * stage]]])
                        },
                        filled=1,
                    )
                    for name in names
                }
            )
            saved = {name: value.clone() for name, value in totals.items()}
            # Establish the existing per-layer reduction as an exact reference.
            expected = {}
            for name in names:
                value = totals[name].clone()
                dist.all_reduce(value, group=mesh["ep"].get_group())
                expected[name] = value.tolist()

            with patch.object(dist, "all_reduce", wraps=dist.all_reduce) as reductions:
                source.batch_end(state)
            ep_calls = [
                call
                for call in reductions.call_args_list
                if call.kwargs.get("group") is mesh["ep"].get_group()
            ]
            assert len(ep_calls) == int(bool(names))
            if names:
                assert ep_calls[0].args[0].shape == (len(names), 6)
            metrics = {record.name: record.value for record in source.drain()}
            for name in names:
                _assert_drop_metrics(source, metrics, name, expected[name])
                torch.testing.assert_close(
                    source._accum[name].drop, saved[name], rtol=0, atol=0
                )
            assert not any("/no_drop/token_drop/" in name for name in metrics)
            full_layout = num_drop_layers == 2 - stage
            expected_vio = 2 / 3 if full_layout else 1.0
            assert (
                metrics[f"{source.config.namespace}/dispatcher_sum/max_vio"]
                == expected_vio
            )
    finally:
        dist.destroy_process_group()


def test_batched_drop_metrics_with_ep_and_unequal_pipeline_stages(tmp_path):
    mp.spawn(
        _batched_drop_ep_pp_worker, args=(str(tmp_path / "store"),), nprocs=4, join=True
    )
