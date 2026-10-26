"""What the EMA callback averages, and what it does when asked to swap.

Three things are pinned here, each because it was wrong or unobservable at
some point:

* the average covers parameters and not buffers -- averaging
  ``moe.expert_bias_E`` made fsdp2 export a different EMA model than gfsdp2
  from the same run;
* ``swap_weights`` is an exchange, so calling it twice is the identity. An
  unpaired call leaves the model training on averaged weights, which is the
  kind of failure that produces plausible numbers rather than a crash;
* a restored shadow survives ``after_load``. The callback seeds its shadows
  from the live weights there, and without the ``_restored`` guard a resume
  would silently throw away everything the run had averaged.
* ``start_step`` re-seeds the shadows from the live weights before the first
  update, so the average does not carry the weights it was built from;
* on gfsdp2 the shadows live in the backend's parameter groups, so model
  parts that are not GFSDP modules are refused rather than averaged some
  other way.
"""

import pytest

pytest.importorskip("torch")

import torch
import torch.nn as nn

from torchtitan.config.configs import ParallelismConfig
from torchtitan.config.decode import decode
from torchtitan.config.loader import load_raw_document
from torchtitan.config.sections import ModelSection
from torchtitan.callbacks.builtin.ema import EMA
from torchtitan.distributed.backends import build_sharding_backend
from torchtitan.distributed.backends.fsdp2 import FSDP2Backend


class _Tiny(nn.Module):
    """One parameter and one persistent buffer, which is the distinction."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(4))
        self.register_buffer("bias_like", torch.full((4,), 5.0), persistent=True)


def _ema(decay: float = 0.5, start_step: int = 0, **kwargs) -> tuple[EMA, _Tiny]:
    model = _Tiny()
    callback = EMA.Config(
        decay=decay, update_period=1, start_step=start_step, **kwargs
    ).build(model_parts=[model], sharding_backend=FSDP2Backend())
    return callback, model


def test_weights_cover_parameters_only() -> None:
    callback, _ = _ema()
    assert set(callback.store.shadow_params) == {"weight"}
    assert "bias_like" not in callback.store.shadow_params


def test_buffer_is_excluded_under_an_activation_checkpoint_wrapper() -> None:
    """The case the bare-module test above cannot see."""
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        checkpoint_wrapper as ptd_checkpoint_wrapper,
    )

    class _Outer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.block = ptd_checkpoint_wrapper(_Tiny())

    model = _Outer()
    wrapped_names = {name for name, _ in model.named_buffers()}
    assert wrapped_names == {"block._checkpoint_wrapped_module.bias_like"}

    callback = EMA.Config(decay=0.5, update_period=1, start_step=0).build(
        model_parts=[model], sharding_backend=FSDP2Backend()
    )
    assert set(callback.store.shadow_params) == {"block.weight"}


def test_buffer_is_left_alone_by_an_update() -> None:
    callback, model = _ema()
    before = model.bias_like.clone()
    model.weight.data.fill_(3.0)
    model.bias_like.fill_(9.0)
    callback.after_load(_state(0))
    callback.after_optimizer_step(_state(1))
    # The shadow moved; the buffer is not the callback's business either way.
    assert not torch.equal(callback.store.shadow_params["weight"], torch.ones(4))
    assert torch.equal(model.bias_like, torch.full((4,), 9.0))
    assert not torch.equal(model.bias_like, before)


def test_swap_twice_is_the_identity() -> None:
    callback, model = _ema()
    callback.after_load(_state(0))
    model.weight.data.fill_(3.0)
    callback.store.shadow_params["weight"] = torch.full((4,), 7.0)

    callback.swap_weights()
    assert torch.equal(model.weight.data, torch.full((4,), 7.0))
    assert torch.equal(callback.store.shadow_params["weight"], torch.full((4,), 3.0))

    callback.swap_weights()
    assert torch.equal(model.weight.data, torch.full((4,), 3.0))
    assert torch.equal(callback.store.shadow_params["weight"], torch.full((4,), 7.0))


def test_restored_shadows_survive_after_load() -> None:
    callback, model = _ema()
    saved = {"weight": torch.full((4,), 7.0)}
    callback.store.load_state_dict(saved)
    model.weight.data.fill_(3.0)

    callback.after_load(_state(0))

    assert torch.equal(callback.store.shadow_params["weight"], torch.full((4,), 7.0)), (
        "after_load re-seeded a shadow that came from a checkpoint"
    )


def test_fresh_shadows_are_seeded_from_the_weights_the_run_starts_from() -> None:
    callback, model = _ema()
    model.weight.data.fill_(3.0)

    callback.after_load(_state(0))

    assert torch.equal(callback.store.shadow_params["weight"], torch.full((4,), 3.0))


def test_start_step_reseeds_before_the_first_update() -> None:
    callback, model = _ema(decay=0.5, start_step=3)
    callback.after_load(_state(0))  # seeded from ones

    model.weight.data.fill_(3.0)
    callback.after_optimizer_step(_state(2))
    assert torch.equal(callback.store.shadow_params["weight"], torch.ones(4))

    callback.after_optimizer_step(_state(3))
    # Without the re-seed this would be 0.5 * 1 + 0.5 * 3 = 2.
    assert torch.equal(callback.store.shadow_params["weight"], torch.full((4,), 3.0))


def test_checkpoint_namespace_matches_the_gfsdp2_adapter() -> None:
    # ``GFSDPCheckpointAdapter`` writes ``ema_model.<fqn>``; DCP flattens this
    # callback's flat state under its key into the same names.
    callback, _ = _ema()
    assert set(callback.stateful_objects()) == {"ema_model"}
    assert set(callback.store.state_dict()) == {"weight"}


def test_gfsdp2_refuses_model_parts_that_are_not_gfsdp_modules() -> None:
    with pytest.raises(ValueError, match="not GFSDP modules"):
        EMA.Config().build(
            model_parts=[_Tiny()],
            sharding_backend=build_sharding_backend(
                ParallelismConfig(fsdp_backend="gfsdp2"),
                decode(
                    ModelSection,
                    load_raw_document(
                        "configs/presets/models/gift_moe/debugmodel.yaml"
                    ).data["model"],
                ).config,
            ),
        )


def _state(step: int):
    """The two fields the callback reads off ``ObserverState``."""

    class _S:
        pass

    state = _S()
    state.step = step
    return state


def test_a_fresh_store_holds_no_average_yet() -> None:
    """Seeded from the weights, so there is nothing to publish as "the EMA"."""
    callback, _ = _ema(start_step=5)
    assert not callback.store.holds_average


def test_the_first_update_makes_it_an_average() -> None:
    callback, model = _ema(start_step=0)
    assert not callback.store.holds_average
    with torch.no_grad():
        model.weight.add_(1.0)
    callback.after_optimizer_step(_state(step=1))
    assert callback.store.holds_average


def test_a_late_start_holds_no_average_until_it_arrives() -> None:
    callback, model = _ema(start_step=3)
    callback.after_optimizer_step(_state(step=1))
    assert not callback.store.holds_average
    callback.after_optimizer_step(_state(step=3))
    assert callback.store.holds_average


def test_a_resume_before_start_step_is_not_an_average() -> None:
    """Restored shadows are the ones the run was seeded with until start_step."""
    callback, _ = _ema(start_step=5)
    shadows = callback.store.model_state_dict()
    callback.store.load_state_dict({k: v.clone() for k, v in shadows.items()})
    callback.after_load(_state(step=3))
    assert not callback.store.holds_average


def test_a_resume_after_start_step_is_an_average() -> None:
    callback, _ = _ema(start_step=5)
    shadows = callback.store.model_state_dict()
    callback.store.load_state_dict({k: v.clone() for k, v in shadows.items()})
    callback.after_load(_state(step=7))
    assert callback.store.holds_average


class _Callbacks:
    """The one method ``Trainer._use_ema_weights`` calls."""

    def __init__(self, callback: EMA) -> None:
        self.callback = callback

    def find(self, want: type):
        return self.callback if isinstance(self.callback, want) else None


def test_validation_swaps_only_when_the_run_evaluates_the_average() -> None:
    """The trainer validates the average on the same two conditions as the export."""
    from types import SimpleNamespace

    from torchtitan.trainer import Trainer

    callback, model = _ema()
    callback.after_load(_state(0))
    model.weight.data.fill_(3.0)
    callback.store.shadow_params["weight"] = torch.full((4,), 7.0)
    trainer = SimpleNamespace(callbacks=_Callbacks(callback), model_parts=[model])

    callback.store.holds_average = False
    with Trainer._use_ema_weights(trainer):
        assert torch.equal(model.weight.data, torch.full((4,), 3.0))

    callback.store.holds_average = True
    callback.store.use_ema = False
    with Trainer._use_ema_weights(trainer):
        assert torch.equal(model.weight.data, torch.full((4,), 3.0))

    callback.store.use_ema = True
    with Trainer._use_ema_weights(trainer):
        assert torch.equal(model.weight.data, torch.full((4,), 7.0))
    assert torch.equal(model.weight.data, torch.full((4,), 3.0))
