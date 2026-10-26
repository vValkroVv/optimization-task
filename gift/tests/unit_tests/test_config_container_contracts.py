"""Config containers have consistent traversal, copying and completion rules."""

from dataclasses import field

import pytest

from torchtitan.config.configurable import config
from torchtitan.config.schema.components.lora import FrozenConfig
from torchtitan.config.schema.models.common.nn_modules import LinearConfig
from torchtitan.config.schema.protocols.model import BaseModelConfig
from torchtitan.config.schema.protocols.module import iter_module_configs


@config
class NestedModelConfig(BaseModelConfig):
    children: dict[str, list[LinearConfig]] = field(default_factory=dict)


@config
class TupleModelConfig(BaseModelConfig):
    children: tuple[LinearConfig, ...] = ()


def test_nested_mapping_list_is_validated():
    with pytest.raises(ValueError, match="children.group.0"):
        NestedModelConfig(children={"group": [LinearConfig()]}).materialize()


def test_nested_positions_are_copied_independently():
    child = LinearConfig(in_features=2, out_features=2)
    source = NestedModelConfig(children={"group": [child, child]})
    tree = source.materialize()
    first, second = tree.children["group"]
    assert first is not second
    first.out_features = 7
    assert second.out_features == child.out_features == 2
    assert [path for path, _ in iter_module_configs(tree)] == [
        "children.group.0",
        "children.group.1",
    ]
    assert [path for path, *_ in tree.traverse(LinearConfig)] == [
        "children.group.0",
        "children.group.1",
    ]


def test_frozen_mapping_values_do_not_hide_incomplete_children():
    source = NestedModelConfig(children={"group": [FrozenConfig(inner=LinearConfig())]})
    with pytest.raises(ValueError, match="children.group.0"):
        source.materialize()


def test_tuple_config_children_fail_explicitly():
    child = LinearConfig(in_features=2, out_features=2)
    source = TupleModelConfig(children=(child, child))
    with pytest.raises(ValueError, match="tuples are unsupported; use a list"):
        source.materialize()


def test_missing_fields_are_all_reported_in_schema_order():
    source = NestedModelConfig(children={"group": [LinearConfig(), LinearConfig()]})
    with pytest.raises(ValueError) as error:
        source.materialize()
    message = str(error.value)
    paths = [
        f"children.group.{index}.{name}"
        for index in range(2)
        for name in ("in_features", "out_features")
    ]
    positions = [message.index(path) for path in paths]
    assert positions == sorted(positions)
    assert "the config that derives it did not fill it" not in message


def test_nested_role_annotation_is_rejected_instead_of_losing_its_role():
    from torchtitan.config.configurable import (
        fields_with_role,
        OWNER_FILLED,
        OwnerFilled,
    )
    from torchtitan.config.schema.protocols.module import ModuleConfig

    @config
    class InvalidRoleConfig(ModuleConfig):
        dim: OwnerFilled[int] | None = None

    with pytest.raises(TypeError, match="outermost"):
        fields_with_role(InvalidRoleConfig, OWNER_FILLED)


def test_missing_field_order_is_stable_across_hash_seeds(monkeypatch):
    from tests.unit_tests.torch_free import run_blocked

    code = (
        "from tests.unit_tests.test_config_container_contracts import NestedModelConfig\n"
        "from torchtitan.config.schema.models.common.nn_modules import LinearConfig\n"
        "try:\n"
        "    NestedModelConfig(children={'group': [LinearConfig()]}).materialize()\n"
        "except ValueError as error:\n"
        "    print(error)\n"
    )
    messages = []
    for seed in (7, 17):
        monkeypatch.setenv("PYTHONHASHSEED", str(seed))
        messages.append(run_blocked(code).stdout)
    assert messages[0] == messages[1]
    assert "in_features" in messages[0] and "out_features" in messages[0]
