"""Runtime binding and parent-completed projection regression tests."""

from typing import ClassVar

import pytest

from torchtitan.config.configurable import Configurable, config
from torchtitan.config.decode import decode
from torchtitan.config.errors import ConfigSchemaError
from torchtitan.config.schema.models.common.attention.qkv import (
    FusedQKVLinearConfig,
    QKVLinearConfig,
)
from torchtitan.config.schema.models.common.ffn import FeedForwardConfig
from torchtitan.config.schema.models.common.nn_modules import LinearConfig
from torchtitan.config.schema.models.common.validation import resolve_child


@config
class ParentConfig(Configurable.Config):
    _owner_spec: ClassVar[str] = "builtins:dict"


@config
class UnboundChildConfig(ParentConfig):
    extra: int = 1


@config
class ReusingChildConfig(ParentConfig):
    _owner_spec: ClassVar[str] = ParentConfig._owner_spec


@config
class BoundChildConfig(ParentConfig):
    _owner_spec: ClassVar[str] = "builtins:list"


def test_unbound_subclass_does_not_use_parent_runtime():
    assert ParentConfig.owner() is dict
    with pytest.raises(NotImplementedError, match="UnboundChildConfig"):
        UnboundChildConfig.owner()


def test_explicit_parent_owner_reuse():
    assert ReusingChildConfig.owner() is dict


def test_child_runtime_wins_after_parent_import():
    assert ParentConfig.owner() is dict
    assert BoundChildConfig.owner() is list


def test_projection_override_keeps_bias_and_parent_geometry():
    cfg = decode(FeedForwardConfig, {"dim": 8, "hidden_dim": 16, "w1": {"bias": True}})
    assert cfg.w1 is not None
    assert cfg.w1.require_features() == (8, 16)
    assert cfg.w1.bias
    assert cfg.w2 is not None and cfg.w2.require_features() == (16, 8)


def test_partial_projection_can_wait_for_parent_width():
    cfg = decode(FeedForwardConfig, {"hidden_dim": 16, "w1": {"bias": True}})
    completed = resolve_child(cfg, FeedForwardConfig, name="ffn", dims={"dim": 8})
    assert completed.w1.require_features() == (8, 16)
    assert completed.w1.bias


def test_resolve_child_requires_without_filling():
    dims = dict(in_features=4, out_features=8)
    built = resolve_child(
        None, LinearConfig, name="p", dims=dims, require={"bias": True}
    )
    assert built.bias
    with pytest.raises(
        ValueError, match=r"p mismatches: requires bias: true \(got false\)"
    ):
        resolve_child(
            LinearConfig(), LinearConfig, name="p", dims=dims, require={"bias": True}
        )


def test_resolve_child_dims_must_be_owner_filled():
    with pytest.raises(AssertionError, match="LinearConfig.bias is not OwnerFilled"):
        resolve_child(None, LinearConfig, name="p", dims={"bias": True})


def test_projection_explicit_wrong_width_is_rejected():
    with pytest.raises(ConfigSchemaError, match="in_features"):
        decode(
            FeedForwardConfig, {"dim": 8, "hidden_dim": 16, "w1": {"in_features": 7}}
        )


@pytest.mark.parametrize("width", [0, -1, True])
def test_invalid_explicit_projection_width_is_rejected(width):
    with pytest.raises(ValueError, match="positive integer"):
        LinearConfig(in_features=width)


def test_uncompleted_standalone_projection_cannot_build():
    with pytest.raises(ValueError, match="in_features and out_features"):
        LinearConfig(bias=True).build()


@pytest.mark.parametrize("kind", ["separate", "fused"])
def test_qkv_partial_projection_overrides_complete_safely(kind):
    if kind == "separate":
        cfg = decode(
            QKVLinearConfig,
            {"head_dim": 4, "wq": {"bias": True}, "wkv": {"bias": True}},
        )
    else:
        cfg = decode(
            FusedQKVLinearConfig,
            {"head_dim": 4, "n_heads": 2, "n_kv_heads": 1, "wqkv": {"bias": True}},
        )
    completed = cfg.completed(name="qkv", dim=8, n_heads=2, n_kv_heads=1, head_dim=4)
    assert completed.in_features == 8
    assert completed.n_heads == 2
    assert completed.n_kv_heads == 1


def test_every_registered_runtime_config_declares_its_binding():
    from torchtitan.config.registry import families

    for family in families().values():
        for name, cls in family.types.items():
            if issubclass(cls, Configurable.Config):
                assert cls.__dict__.get("_owner_spec") or cls.__dict__.get("_owner"), (
                    family.name,
                    name,
                )


@pytest.mark.parametrize(
    "experiment,path",
    [
        ("llama3_debug.yaml", "model.config.blocks.dense.attention.wo.bias"),
        ("gift_4_debug.yaml", "model.config.blocks.gdn_dense.feed_forward.w1.bias"),
    ],
)
def test_projection_bias_override_passes_full_pre_submit(experiment, path):
    from torchtitan.config.pre_submit import run_pre_submit

    result = run_pre_submit(f"configs/experiments/{experiment}", [(path, "true")])
    assert result.ok, result.report.to_dict()
