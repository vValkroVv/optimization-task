"""Checkpoint round-trip of every callback that carries state.

Three participants implement state_dict/load_state_dict: the rescaler (four
dicts with int block keys serialized through JSON strings), EMA (shadow
weights) and TrainerSource (the train-seconds clock). None of this was
covered: a bug in the rescaler's load would surface on the restart of a long
run — not as a crash, but as silently lost LR scales. Also pinned: what each
participant does when its inner keys are missing (an older checkpoint), and
that toggling ``log_throughput`` changes the set of registered stateful
objects — the top-level-key semantics on a changed set belong to DCP, which
fails loudly on a missing key.
"""

import time
from collections import deque

import torch
from torch import nn

from torchtitan.distributed.backends.fsdp2 import FSDP2Backend
from torchtitan.callbacks.builtin.dynamic_block_lr_rescaler import (
    DynamicBlockLRRescaler,
)
from torchtitan.callbacks.builtin.ema import EMA
from torchtitan.callbacks.builtin.monitoring import MonitoringCallback
from torchtitan.callbacks.sources.dispatcher_source import DispatcherSource
from torchtitan.callbacks.sources.trainer_source import TrainerSource


def _rescaler() -> DynamicBlockLRRescaler:
    monitoring = MonitoringCallback(
        MonitoringCallback.Config(
            log_interval=1, sources={"dispatcher": DispatcherSource.Config()}
        ),
        sharding_backend=FSDP2Backend(),
    )
    return DynamicBlockLRRescaler(
        DynamicBlockLRRescaler.Config(), monitoring=monitoring
    )


def test_rescaler_state_round_trips_with_int_block_keys():
    saved = _rescaler()
    saved._block_scale = {3: 0.5, 11: 0.25}
    saved._unscaled_base_lrs = {3: [1e-3, 2e-3], 11: [4e-4]}
    saved._max_vio_history = {3: deque([1.0, 2.5, 2.0], maxlen=8)}
    saved._block_last_fire = {11: 7}
    saved._capture_count = 42

    state = saved.state_dict()

    restored = _rescaler()
    restored.load_state_dict(state)
    assert restored._restored
    assert restored._loaded_state == {
        "block_scale": {3: 0.5, 11: 0.25},
        "unscaled_base_lrs": {3: [1e-3, 2e-3], 11: [4e-4]},
        "max_vio_history": {3: [1.0, 2.5, 2.0]},
        "block_last_fire": {11: 7},
        "capture_count": 42,
    }, "block keys must come back as ints, values numerically intact"


def test_rescaler_tolerates_a_checkpoint_without_its_payload():
    restored = _rescaler()
    restored.load_state_dict({})
    assert not restored._restored, "an empty load must read as a fresh run"
    assert restored._loaded_state["block_scale"] == {}
    assert restored._loaded_state["capture_count"] == 0


def _ema_store(model: nn.Module):
    """The store the callback delegates to, reached the way the callback does.

    ``EMA`` itself carries no state since the sharding backends grew their own
    stores: it holds *when* to average, and the store holds *where*. Reaching
    through ``.store`` rather than importing ``ShadowParamStore`` keeps this
    test on the object the callback actually uses.
    """
    return EMA(
        EMA.Config(decay=0.99, update_period=10),
        model_parts=[model],
        sharding_backend=FSDP2Backend(),
    ).store


def test_ema_shadow_round_trips_bitwise():
    model = nn.Linear(4, 4)
    saved = _ema_store(model)
    for tensor in saved.shadow_params.values():
        tensor.mul_(0.5)  # diverge from the live weights

    on_disk = {name: tensor.clone() for name, tensor in saved.state_dict().items()}

    restored = _ema_store(model)
    restored.load_state_dict(on_disk)
    assert restored._restored
    assert set(restored.shadow_params) == set(saved.shadow_params)
    for name, tensor in saved.shadow_params.items():
        assert torch.equal(restored.shadow_params[name], tensor)


def test_ema_names_the_shadows_a_checkpoint_did_not_carry():
    """A shadow the checkpoint lacks must not be reseeded without a trace."""
    store = _ema_store(nn.Linear(4, 4))
    expected = sorted(store.shadow_params)
    assert expected, "the store must have shadows to be missing in the first place"

    store.load_state_dict({})
    assert store._restored, "a load happened, even though it carried nothing"
    assert sorted(store.after_load()) == expected

    # Reported once: a second call must not re-announce shadows already seeded.
    assert list(store.after_load()) == []


def test_trainer_source_clock_round_trips_and_defaults():
    saved = TrainerSource(TrainerSource.Config())
    saved._prior_train_seconds = 100.0
    saved._speed_start_time = time.perf_counter()
    seconds = saved.state_dict()["train_seconds"]
    assert seconds >= 100.0

    restored = TrainerSource(TrainerSource.Config())
    restored.load_state_dict({"train_seconds": seconds})
    assert restored._prior_train_seconds == seconds

    fresh = TrainerSource(TrainerSource.Config())
    fresh.load_state_dict({})  # an older checkpoint without the key
    assert fresh._prior_train_seconds == 0.0
