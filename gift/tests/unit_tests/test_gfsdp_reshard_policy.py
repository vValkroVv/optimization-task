"""Every GFSDP unit gets the same resharding policy, resolved once.

``fsdp_reshard_after_forward`` is a global choice on this backend, not a
per-unit one: ``_init_shared_state`` deals one data buffer per group when no
unit reshards and two per process group otherwise, and inside the two-buffer
layout only the last unit in forward order may keep its buffer. A unit that
resolves the policy differently from its neighbours therefore does not merely
reshard on its own schedule -- it collides with whoever shares its buffer.

``norm`` and ``lm_head`` used to compare the raw policy string against
``"always"`` while every other unit took the resolved boolean, which agrees at
``never`` and ``always`` and parts ways at ``default``, the config's own
default.
"""

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from torchtitan.distributed.fsdp import (  # noqa: E402
    get_fsdp_reshard_after_forward_policy,
)
from tests.unit_tests.config_helpers import load_model_spec  # noqa: E402
from torchtitan.models.gift_moe._base import parallelize  # noqa: E402


class _Stub:
    """A sharded unit, as far as ``apply_gfsdp`` is concerned."""

    def set_own_grad_buffer(self):
        pass

    def set_allow_no_grad_reduce(self, enable):
        pass

    def set_shard_order(self, modules):
        pass

    def set_gradient_divide_factor(self, factor):
        pass


def _shard_calls(*, policy: str, pp_enabled: bool, monkeypatch):
    """``(module, reshard_after_forward)`` for every ``g_fully_shard`` call."""
    calls: list[tuple[object, object]] = []

    def _fake(module, **kwargs):
        calls.append((module, kwargs.get("reshard_after_forward")))
        return _Stub()

    monkeypatch.setattr(parallelize, "g_fully_shard", _fake)
    model = load_model_spec("gift_moe/debugmodel.yaml").model.build()
    parallelize.apply_gfsdp(
        model,
        None,
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        pp_enabled=pp_enabled,
        reshard_after_forward_policy=policy,
    )
    return model, calls


@pytest.mark.parametrize(
    ("policy", "pp_enabled", "expected"),
    [
        ("always", False, True),
        ("always", True, True),
        ("never", False, False),
        ("never", True, False),
        ("default", False, True),
        ("default", True, False),
    ],
)
def test_the_policy_resolves_to_zero3_without_a_pipeline(policy, pp_enabled, expected):
    assert get_fsdp_reshard_after_forward_policy(policy, pp_enabled) is expected


@pytest.mark.parametrize("policy", ["default", "always", "never"])
@pytest.mark.parametrize("pp_enabled", [False, True])
def test_every_unit_is_sharded_with_the_resolved_policy(
    policy, pp_enabled, monkeypatch
):
    model, calls = _shard_calls(
        policy=policy, pp_enabled=pp_enabled, monkeypatch=monkeypatch
    )
    expected = get_fsdp_reshard_after_forward_policy(policy, pp_enabled)

    units = [(module, value) for module, value in calls if module is not model]
    assert units, "apply_gfsdp sharded nothing"
    assert {value for _, value in units} == {expected}


def test_the_root_is_left_on_automatic(monkeypatch):
    """``None`` is what turns on the root's automatic mode, which frees the
    last unit on each data buffer from resharding."""
    model, calls = _shard_calls(
        policy="always", pp_enabled=False, monkeypatch=monkeypatch
    )

    root = [value for module, value in calls if module is model]
    assert root == [None]


def test_norm_and_lm_head_are_separate_units(monkeypatch):
    """Grouped, the chunked loss re-enters ``lm_head`` per chunk and leaves
    ``norm`` marked as never having run forward."""
    model, calls = _shard_calls(
        policy="never", pp_enabled=False, monkeypatch=monkeypatch
    )
    sharded = [module for module, _ in calls]

    assert model.norm in sharded
    assert model.lm_head in sharded
    assert not any(isinstance(module, (list, tuple)) for module in sharded)
