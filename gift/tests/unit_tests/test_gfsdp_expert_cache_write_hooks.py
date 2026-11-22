"""Exercise the master-weight write fences with a cache-group double."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from torchtitan.components import optimizer
from torchtitan.distributed.gfsdp import _dcp
from torchtitan.distributed.gfsdp.fully_shard import GFSDPModule


class _CacheGroup:
    def __init__(self, weight, *, bf16=False, fp8=True, extra_params=()):
        self._bf16_cache_requested = bf16
        self._fp8_cache_requested = fp8
        self.requires_grad = True
        self.fsdp_params = [
            SimpleNamespace(sharded_param=param) for param in (weight, *extra_params)
        ]
        self.weight = weight
        self.events = []

    def disable_bf16_weight_cache(self):
        self._bf16_cache_requested = False

    def disable_fp8_weight_cache(self):
        self._fp8_cache_requested = False

    def invalidate_bf16_weight_cache(self):
        self.events.append(("invalidate", self.weight.item()))

    def record_weight_write(self):
        self.events.append(("write", self.weight.item()))


@pytest.mark.parametrize("bf16,fp8", [(True, False), (False, True), (True, True)])
def test_optimizer_wraps_master_update_once_for_each_group(monkeypatch, bf16, fp8):
    model = nn.Linear(1, 1, bias=False)
    nn.init.ones_(model.weight)
    group = _CacheGroup(model.weight, bf16=bf16, fp8=fp8)
    monkeypatch.setattr(optimizer, "_iter_gfsdp_param_groups", lambda _: [group])
    optimizers = optimizer.OptimizersContainer.Config(
        lr=0.25, implementation="for-loop"
    ).build(model_parts=[model])

    model.weight.grad = torch.ones_like(model.weight)
    optimizers.step()

    assert group.events == [("invalidate", 1.0), ("write", model.weight.item())]
    assert model.weight.item() < 1.0


@pytest.mark.parametrize("in_backward", [False, True])
def test_optimizer_disables_cache_when_writes_cannot_be_fenced(
    monkeypatch, in_backward
):
    model = nn.Linear(1, 1, bias=False)
    extra_params = () if in_backward else (nn.Parameter(torch.zeros(())),)
    group = _CacheGroup(model.weight, bf16=True, extra_params=extra_params)
    monkeypatch.setattr(optimizer, "_iter_gfsdp_param_groups", lambda _: [group])
    container = (
        optimizer.OptimizersInBackwardContainer
        if in_backward
        else optimizer.OptimizersContainer
    )
    container.Config(implementation="for-loop").build(model_parts=[model])

    assert not group._bf16_cache_requested
    assert not group._fp8_cache_requested


def test_apply_records_a_write_after_the_master_is_replaced():
    class ApplyModel(GFSDPModule, nn.Linear):
        def _get_fsdp_state(self):
            return self.state

        def reshard(self):
            pass

    model = nn.Linear(1, 1, bias=False)
    model.__class__ = ApplyModel
    nn.init.ones_(model.weight)
    group = _CacheGroup(model.weight)
    group._fp8_cache = object()
    group._bf16_cache_enabled = False
    group.data_buffer_ctx = object()
    group._all_gather_input_set_dtypes = {None}
    group.fsdp_params[0].reset_sharded_param = lambda: None
    model.state = SimpleNamespace(_fsdp_param_groups=[group])

    assert model._apply(lambda weight: weight + 2.0) is model

    assert group.events == [("invalidate", 1.0), ("write", 3.0)]
    assert not group._all_gather_input_set_dtypes


@pytest.mark.parametrize("fail_after_load", [False, True])
def test_dcp_fences_direct_checkpoint_writes_into_main_and_ema(
    monkeypatch, tmp_path, fail_after_load
):
    from torchtitan.protocols.sharding_backend import EMA_MODEL

    main = torch.tensor(1.0)
    ema = torch.tensor(2.0)
    group = _CacheGroup(main)
    checkpoint_id = str(tmp_path / "checkpoint")
    _dcp.dcp.save(
        {"weight": torch.tensor(3.0), EMA_MODEL: {"weight": torch.tensor(4.0)}},
        checkpoint_id=checkpoint_id,
    )

    def apply_main_state_dict(target):
        assert target["weight"] is main
        assert target[EMA_MODEL]["weight"] is ema
        assert (main.item(), ema.item()) == (3.0, 4.0)
        assert group.events == [("invalidate", 1.0)]
        if fail_after_load:
            raise RuntimeError("post-load callback failed")

    adapter = SimpleNamespace(
        _model_parts=[object()],
        build_state_dict=lambda *, load_target: {
            "weight": main,
            EMA_MODEL: {"weight": ema},
        },
        apply_main_state_dict=apply_main_state_dict,
        initialize_ema_from_main=lambda _: None,
    )
    monkeypatch.setattr(
        _dcp, "_iter_fsdp_param_groups", lambda *args, **kwargs: [group]
    )

    if fail_after_load:
        with pytest.raises(RuntimeError, match="post-load callback failed"):
            _dcp.load_gfsdp_dcp(adapter, {}, checkpoint_id, None)
    else:
        _dcp.load_gfsdp_dcp(adapter, {}, checkpoint_id, None)

    assert group.events == [("invalidate", 1.0), ("write", 3.0)]
