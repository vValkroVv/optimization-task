"""Strict decoder tests, one group per rule of ``torchtitan/config/decode.py``."""

import dataclasses
from collections.abc import Callable, Mapping  # noqa: F401  -- string annotations
from dataclasses import dataclass, field
from typing import Literal

import pytest

from torchtitan.config.configurable import component, component_family
from torchtitan.config.decode import checking_missing_cleared_paths, decode
from torchtitan.config.errors import ConfigSchemaError
from torchtitan.config.schema.protocols.types import MeshAxisName


def _fails(cls, data, match, path="p"):
    with pytest.raises(ConfigSchemaError, match=match):
        decode(cls, data, path=path)


# ---------------------------------------------------------------------------
# tuples
# ---------------------------------------------------------------------------


@dataclass
class _Tuples:
    pair: tuple[int, int] = (0, 0)
    mixed: tuple[int, str] = (0, "")
    many: tuple[int, ...] = ()
    items: list[int] = field(default_factory=list)


def test_fixed_tuple_checks_length_and_each_position():
    assert decode(_Tuples, {"pair": [1, 2]}, path="p").pair == (1, 2)
    assert decode(_Tuples, {"mixed": [1, "a"]}, path="p").mixed == (1, "a")
    _fails(_Tuples, {"pair": [1, 2, 3]}, r"p\.pair: expected 2 items, got 3")
    _fails(_Tuples, {"pair": [1]}, r"p\.pair: expected 2 items, got 1")
    _fails(_Tuples, {"pair": [1, "a"]}, r"p\.pair\[1\]: expected an int")
    _fails(_Tuples, {"mixed": [1, 2]}, r"p\.mixed\[1\]: expected a string")


def test_fixed_tuple_never_takes_a_lone_scalar():
    _fails(_Tuples, {"pair": 5}, r"p\.pair: expected a list, got int")
    # The variadic form and plain lists still take the one-element spelling.
    assert decode(_Tuples, {"many": 5}, path="p").many == (5,)
    assert decode(_Tuples, {"items": 5}, path="p").items == [5]


def test_window_size_is_a_pair():
    from torchtitan.config.schema.models.common.attention.softmax.kernels.flash_fa2 import (
        FlashAttention2Config,
    )

    _fails(
        FlashAttention2Config,
        {"window_size": [1, 2, 3]},
        r"attn\.window_size: expected 2 items, got 3",
        path="attn",
    )


# ---------------------------------------------------------------------------
# __post_init__ invariants
# ---------------------------------------------------------------------------


@dataclass
class _Invariant:
    v: int = 0

    def __post_init__(self):
        if self.v < 0:
            raise ValueError(f"v must be >= 0, got {self.v}")
        if self.v > 100:
            raise TypeError("a bug, not a config error")


def test_post_init_value_error_is_a_config_error_with_the_path():
    _fails(_Invariant, {"v": -1}, r"^p: v must be >= 0, got -1$")


def test_post_init_other_exceptions_propagate_as_themselves():
    with pytest.raises(TypeError, match="a bug"):
        decode(_Invariant, {"v": 101}, path="p")


def test_in_backward_optimizer_with_expert_parallel_is_reported():
    """The trainer's ``__post_init__`` error lands in the pre-submit report."""
    from tests.unit_tests.torch_free import REPO_ROOT
    from torchtitan.config import run_pre_submit

    experiment = REPO_ROOT / "configs" / "experiments" / "llama3_debug.yaml"
    result = run_pre_submit(
        str(experiment),
        [
            ("optimizer.type", "in_backward"),
            ("parallelism.expert_parallel_degree", "2"),
        ],
    )
    errors = [issue.message for issue in result.report.errors]
    assert any("not supported with Expert Parallel" in e for e in errors), errors


# ---------------------------------------------------------------------------
# unions
# ---------------------------------------------------------------------------


@dataclass
class _Arm:
    x: int = 0


@dataclass
class _Unions:
    opt_int: int | None = None
    opt_list: list[int] | None = None
    opt_dict: dict[str, int] | None = None
    opt_arm: _Arm | None = None
    several: str | int | list[str | int] | None = None


def test_single_arm_container_keeps_the_inner_path():
    _fails(_Unions, {"opt_list": ["a"]}, r"^p\.opt_list\[0\]: expected an int")
    _fails(_Unions, {"opt_dict": {"k": "a"}}, r"^p\.opt_dict\.k: expected an int")
    _fails(_Unions, {"opt_arm": {"xx": 1}}, r"^p\.opt_arm\.xx: unknown key")


def test_single_arm_scalar_keeps_the_union_summary():
    _fails(_Unions, {"opt_int": "abc"}, r"^p\.opt_int: expected int \| None, got str")
    _fails(
        _Unions,
        {"opt_list": 5.5},
        r"^p\.opt_list: expected .*list.* \| None, got float",
    )


def test_several_arms_list_value_shows_each_arm():
    _fails(
        _Unions,
        {"several": [0, 2.5]},
        r"(?s)^p\.several: expected .*got list.*as list.*p\.several\[1\]: expected",
    )


def test_stat_dim_item_error_survives():
    from torchtitan.config.schema.callbacks.metrics.sample import StatOpConfig

    _fails(
        StatOpConfig,
        {"op": "mean", "dim": [0, 2.5]},
        r"s\.dim\[1\]: expected",
        path="s",
    )


def test_sharding_single_arm_mapping_keeps_the_inner_path():
    from torchtitan.config.schema.protocols.sharding import ShardingConfig

    _fails(
        ShardingConfig,
        {"out_dst_shardings": {"tp": {"type": "shard", "dmi": 0}}},
        r"^s\.out_dst_shardings\.tp\.dmi: unknown key",
        path="s",
    )


def test_placements_are_selected_by_type():
    """A placement without ``type`` is an error, never the first union arm."""
    from torchtitan.config.schema.protocols.sharding import (
        LocalMapConfig,
        Partial,
        Replicate,
        Shard,
        ShardingConfig,
    )
    from torchtitan.config.schema.protocols.types import MeshAxisName

    def state(placement):
        decoded = decode(
            ShardingConfig, {"state_shardings": {"w": {"tp": placement}}}, path="s"
        )
        return decoded.state_shardings["w"][MeshAxisName.TP]

    assert state({"type": "replicate"}) == Replicate()
    assert state({"type": "partial"}) == Partial()
    assert state({"type": "partial", "reduce_op": "avg"}) == Partial(reduce_op="avg")
    assert state({"type": "shard", "dim": -1}) == Shard(-1)
    for ambiguous in ({}, {"dim": 0}):
        with pytest.raises(ConfigSchemaError, match="missing required key 'type'"):
            state(ambiguous)

    # A dump names each type and decodes back to the same specs.
    sharding = ShardingConfig(
        state_shardings={
            "w": {MeshAxisName.TP: Partial(), MeshAxisName.CP: Replicate()}
        }
    )
    local_map = LocalMapConfig(in_grad_placements=({MeshAxisName.TP: Shard(0)},))
    for config in (sharding, local_map):
        dumped = {
            f.name: _plain(getattr(config, f.name)) for f in dataclasses.fields(config)
        }
        assert decode(type(config), dumped, path="s") == config
    assert Partial().to_dict() == {"type": "partial", "reduce_op": "sum"}


def _plain(value):
    """What a YAML dump of a sharding field holds: placements via their own
    ``to_dict``, mesh-axis keys as plain strings, tuples as lists."""
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


# ---------------------------------------------------------------------------
# dict keys
# ---------------------------------------------------------------------------


@dataclass
class _Dicts:
    by_name: dict[str, int] = field(default_factory=dict)


def test_dict_keys_are_checked_against_the_key_type():
    assert decode(_Dicts, {"by_name": {"a": 1}}, path="p").by_name == {"a": 1}
    _fails(_Dicts, {"by_name": {1: 2}}, r"^p\.by_name\[1\]: expected a string")


def test_mesh_axis_keys_must_name_an_axis():
    from torchtitan.config.schema.protocols.sharding import Shard, ShardingConfig
    from torchtitan.config.schema.protocols.types import MeshAxisName

    good = decode(
        ShardingConfig,
        {"state_shardings": {"w": {"tp": {"type": "shard", "dim": 0}}}},
        path="s",
    )
    assert good.state_shardings == {"w": {MeshAxisName.TP: Shard(dim=0)}}
    assert good.state_shardings["w"]["tp"] == Shard(
        dim=0
    )  # StrEnum keys stay str-equal
    _fails(
        ShardingConfig,
        {"state_shardings": {"w": {"tpp": {"type": "shard", "dim": 0}}}},
        r"^s\.state_shardings\.w\.tpp: 'tpp' is not one of .*'tp'.* \(MeshAxisName\)\. Did you mean 'tp'",
        path="s",
    )


# ---------------------------------------------------------------------------
# fields a document cannot set, and the ones it can spell as a mapping
# ---------------------------------------------------------------------------


@dataclass
class _Derived:
    x: int = 0
    derived: "Nonexistent | None" = field(init=False, default=None)  # noqa: F821
    by_name: "Mapping[str, int] | None" = None


def test_init_false_field_is_not_a_yaml_key():
    """Setting it is an unknown-key error that does not suggest the field."""
    assert decode(_Derived, {"x": 1}, path="p").derived is None
    _fails(
        _Derived,
        {"derived": 1},
        r"^p\.derived: unknown key\. Valid keys for _Derived: by_name, x$",
    )


def test_misspelled_annotation_on_an_init_field_is_a_name_error():
    @dataclass
    class _Typo:
        n: "Nonexistnt" = 0  # noqa: F821

    with pytest.raises(NameError, match="Nonexistnt"):
        decode(_Typo, {"n": 1}, path="p")


def test_mapping_is_a_dict_spelling():
    assert decode(_Derived, {"by_name": {"a": 1}}).by_name == {"a": 1}
    _fails(_Derived, {"by_name": {"a": "x"}}, r"^p\.by_name\.a: expected an int")


def test_param_init_decodes_by_type():
    from torchtitan.config.schema.models.common.nn_modules import LinearConfig
    from torchtitan.config.schema.models.gift_moe.const import RenormKeepStd
    from torchtitan.config.schema.protocols.module import InitSpec, Normal, Zeros

    linear = {"in_features": 4, "out_features": 4}
    good = decode(
        LinearConfig,
        linear
        | {
            "param_init": {
                "weight": {
                    "type": "renorm_keep_std",
                    "init": {"type": "normal", "std": 0.02},
                },
                "bias": {"type": "zeros"},
            }
        },
        path="p",
    )
    assert good.param_init == {
        "weight": RenormKeepStd(init=Normal(std=0.02)),
        "bias": Zeros(),
    }
    dumped = good.param_init["weight"].to_dict()
    assert dumped == {
        "type": "renorm_keep_std",
        "init": {"type": "normal", "std": 0.02, "mean": 0.0},
    }
    assert decode(InitSpec, dumped, path="w") == good.param_init["weight"]
    # An unregistered class (the bare base here) has no ``type`` to name.
    with pytest.raises(ValueError, match="no registered YAML type"):
        InitSpec().to_dict()
    _fails(
        LinearConfig,
        linear | {"param_init": {"weight": {}}},
        r"^p\.param_init\.weight: missing required key 'type'",
    )
    _fails(
        LinearConfig,
        linear | {"param_init": {"weight": {"type": "normla", "std": 0.02}}},
        r"^p\.param_init\.weight\.type: unknown param_init type 'normla'\. Did you mean 'normal'",
    )


# ---------------------------------------------------------------------------
# a lone scalar for a list
# ---------------------------------------------------------------------------


@dataclass
class _Lists:
    names: list[Literal["a", "b"]] = field(default_factory=list)
    counts: list[int] = field(default_factory=list)


@dataclass
class _NestedLists:
    parts: list[list[str]] = field(default_factory=list)
    arms: list[_Arm] = field(default_factory=list)
    pair: tuple[int, int] = (0, 0)


def test_lone_scalar_is_promoted_only_for_scalar_items():
    """``parts: [a, b]`` means one list of two names, never two one-name parts."""
    _fails(
        _NestedLists, {"parts": ["a", "b"]}, r"^p\.parts\[0\]: expected a list, got str"
    )
    _fails(_NestedLists, {"parts": "a"}, r"^p\.parts: expected a list, got str")
    _fails(_NestedLists, {"arms": 5}, r"^p\.arms: expected a list, got int")
    assert decode(_NestedLists, {"parts": [["a", "b"]]}, path="p").parts == [["a", "b"]]


def test_tuple_length_error_bounds_the_value():
    _fails(
        _NestedLists,
        {"pair": list(range(100))},
        r"got 100 \(\[0, 1, 2, 3, 4, .*\.\.\.\]\)$",
    )


def test_lone_scalar_is_the_one_element_list_and_keeps_its_error():
    assert decode(_Lists, {"names": "a"}, path="p").names == ["a"]
    _fails(_Lists, {"names": "bad"}, r"^p\.names: 'bad' is not one of 'a', 'b'")
    _fails(_Lists, {"counts": "abc"}, r"^p\.counts: expected an int, got 'abc'")
    _fails(_Lists, {"counts": {"a": 1}}, r"^p\.counts: expected a list, got dict")
    _fails(_Lists, {"counts": None}, r"^p\.counts: expected a list, got NoneType")


# ---------------------------------------------------------------------------
# message rendering
# ---------------------------------------------------------------------------


def test_type_names_render_as_written():
    from torchtitan.config.decode import _format_type_name
    from torchtitan.config.schema.protocols.sharding import NamedPlacement

    assert _format_type_name(int | None) == "int | None"
    assert _format_type_name(dict[str, list[int]]) == "dict[str, list[int]]"
    assert _format_type_name(tuple[int, ...]) == "tuple[int, ...]"
    assert _format_type_name(Literal["a", "b"]) == "Literal['a', 'b']"
    assert _format_type_name(Callable[[int], int]) == "Callable[[int], int]"
    assert _format_type_name(NamedPlacement) == "dict[MeshAxisName, Placement]"


def test_root_errors_name_the_class():
    _fails(
        _Arm, 5, r"^_Arm: expected a mapping of _Arm fields, got int \(5\)$", path=""
    )
    _fails(_Unions, 5, r"^_Unions: expected a mapping", path="")


@dataclass
class _Required:
    a: int
    b: int


def test_missing_keys():
    _fails(_Required, {"a": 1}, r"^p\.b: missing required key$")
    _fails(_Required, {}, r"^p: missing required keys a, b$")
    _fails(_Required, {}, r"^_Required: missing required keys a, b$", path="")


def test_values_in_messages_are_bounded():
    _fails(
        _Lists,
        {"counts": ["x" * 500]},
        r"^p\.counts\[0\]: expected an int, got '.{0,90}'$",
    )
    _fails(
        _Dicts, {"by_name": list(range(100))}, r"got list \(\[0, 1, 2, .*\.\.\.\]\)$"
    )


# ---------------------------------------------------------------------------
# `type` on a field declared as one concrete class
# ---------------------------------------------------------------------------


@component_family("decode-test widget", default="plain")
@dataclass
class _WidgetBase:
    n: int = 0


@component("plain")
@dataclass
class _Plain(_WidgetBase):
    pass


@component("fancy")
@dataclass
class _Fancy(_WidgetBase):
    shine: int = 1


@component("fancier")
@dataclass
class _Fancier(_Fancy):
    pass


@dataclass
class _Holder:
    any_widget: _WidgetBase = field(default_factory=_Plain)
    fancy: _Fancy = field(default_factory=_Fancy)


def test_family_base_field_dispatches_on_type():
    assert isinstance(decode(_Holder, {"any_widget": {}}, path="p").any_widget, _Plain)
    assert isinstance(
        decode(_Holder, {"any_widget": {"type": "fancy"}}, path="p").any_widget, _Fancy
    )


def test_concrete_field_takes_its_own_type_or_a_subclass_only():
    assert decode(_Holder, {"fancy": {}}, path="p").fancy == _Fancy()
    assert decode(
        _Holder, {"fancy": {"type": "fancy", "shine": 2}}, path="p"
    ).fancy == _Fancy(shine=2)
    assert isinstance(
        decode(_Holder, {"fancy": {"type": "fancier"}}, path="p").fancy, _Fancier
    )
    _fails(
        _Holder,
        {"fancy": {"type": "plain"}},
        r"^p\.fancy\.type: this field is fixed to _Fancy; 'plain' is not a variant of it",
    )
    _fails(
        _Holder,
        {"fancy": {"type": "zz"}},
        r"^p\.fancy\.type: unknown decode-test widget type 'zz'",
    )


# ---------------------------------------------------------------------------
# a mapping is layered on the field's default
# ---------------------------------------------------------------------------


@dataclass
class _Inner:
    a: int = 0
    b: int = 0


@dataclass(frozen=True)
class _FrozenInner:
    a: int = 0
    b: int = 0


@dataclass
class _Unbuildable:
    x: int = 0

    def __post_init__(self):
        if self.x == 0:
            raise ValueError("x must be set")


@dataclass
class _Outer:
    inner: _Inner = field(default_factory=lambda: _Inner(a=1, b=2))


@dataclass
class _Layered:
    inner: _Inner = field(default_factory=lambda: _Inner(a=1, b=2))
    frozen: _FrozenInner = _FrozenInner(a=5)
    widget: _WidgetBase = field(default_factory=lambda: _Fancy(n=3, shine=7))
    outer: _Outer = field(default_factory=lambda: _Outer(inner=_Inner(a=9, b=9)))
    optional: _Inner | None = None


@dataclass
class _HoldsUnbuildable:
    unbuildable: _Unbuildable = field(default_factory=_Unbuildable)


@dataclasses.dataclass(kw_only=True)
class _Table:
    entries: Mapping[str, _Inner] = dataclasses.field(
        default_factory=lambda: {"x": _Inner(a=1, b=2), "y": _Inner(a=3, b=4)}
    )


def test_mapping_field_is_layered_on_its_default():
    """A mapping field keeps the default's entries the document leaves out,
    and an entry that is a mapping layers on the default's entry."""
    table = decode(
        _Table, {"entries": {"x": {"b": 9}, "z": {"a": 5, "b": 6}}}, path="p"
    )
    assert table.entries == {
        "x": _Inner(a=1, b=9),
        "y": _Inner(a=3, b=4),
        "z": _Inner(a=5, b=6),
    }


def test_mapping_field_null_entry_cannot_remove_a_default_entry():
    """A removed default entry could not be written back: a reload layers
    the dumped entries on the default again. A key it lacks is a typo."""
    with checking_missing_cleared_paths([("p", "entries", "y")]):
        with pytest.raises(
            ConfigSchemaError,
            match=r"^p\.entries\.y: null cannot remove an entry of the default",
        ):
            decode(_Table, {"entries": {"x": {"b": 9}}}, path="p")
    with checking_missing_cleared_paths([("p", "entries", "w")]):
        with pytest.raises(ConfigSchemaError, match=r"^p\.entries\.w: no entry"):
            decode(_Table, {"entries": {"x": {"b": 9}}}, path="p")


def test_reset_under_an_absent_default_entry_is_checked():
    """``entries`` and ``y`` exist only in the default: a reset of a field
    below them is a no-op, a misspelled one is an error."""
    with checking_missing_cleared_paths([("p", "entries", "y", "b")]):
        assert decode(_Table, {}, path="p") == _Table()
    with checking_missing_cleared_paths([("p", "entries", "y", "c")]):
        with pytest.raises(ConfigSchemaError, match=r"^p\.entries\.y\.c: unknown key"):
            decode(_Table, {}, path="p")

    data = {"entries": {"a.b": {}}}
    with checking_missing_cleared_paths([("p", "entries", "a.b", "b")]):
        assert decode(_Table, data, path="p").entries["a.b"] == _Inner()
    with checking_missing_cleared_paths([("p", "entries", "a.b", "c")]):
        with pytest.raises(ConfigSchemaError, match="unknown key"):
            decode(_Table, data, path="p")


def test_mapping_keeps_the_default_instance_values():
    layered = decode(_Layered, {"inner": {"b": 5}, "frozen": {"b": 1}}, path="p")
    assert layered.inner == _Inner(a=1, b=5)
    assert layered.frozen == _FrozenInner(a=5, b=1)
    # Two levels deep: the nested default comes from the enclosing default.
    assert decode(_Layered, {"outer": {"inner": {"a": 1}}}, path="p").outer == _Outer(
        inner=_Inner(a=1, b=9)
    )
    # No default instance to keep: the class defaults.
    assert decode(_Layered, {"optional": {"b": 5}}, path="p").optional == _Inner(b=5)


def test_family_field_keeps_the_default_class_until_type_changes_it():
    def widget(value):
        return decode(_Layered, {"widget": value}, path="p").widget

    assert widget({"n": 4}) == _Fancy(n=4, shine=7)
    assert widget({"type": "fancy", "n": 4}) == _Fancy(n=4, shine=7)
    # Another class starts from its own defaults, a subclass included.
    assert widget({"type": "plain"}) == _Plain()
    assert widget({"type": "fancier"}) == _Fancier()


def test_class_default_factory_is_not_called():
    """A class used as the factory contributes only itself; ``_Unbuildable()``
    would raise."""
    decoded = decode(_HoldsUnbuildable, {"unbuildable": {"x": 2}}, path="p")
    assert decoded.unbuildable.x == 2


def test_rope_scaling_is_a_family():
    from torchtitan.config.schema.models.common.embeddings.rope.rope import (
        ComplexRoPEConfig,
    )
    from torchtitan.config.schema.models.common.embeddings.rope.scaling import (
        NoScaling,
        YarnScaling,
    )

    rope = {"dim": 16, "max_seq_len": 64}
    assert decode(ComplexRoPEConfig, rope, path="r").scaling == NoScaling()
    assert (
        decode(ComplexRoPEConfig, rope | {"scaling": {}}, path="r").scaling
        == NoScaling()
    )
    yarn = {"type": "yarn", "rope_factor": 4.0, "original_seq_len": 16}
    assert decode(ComplexRoPEConfig, rope | {"scaling": yarn}, path="r").scaling == (
        YarnScaling(rope_factor=4.0, original_seq_len=16)
    )
    assert YarnScaling(rope_factor=4.0, original_seq_len=16).to_dict()["type"] == "yarn"
    _fails(
        ComplexRoPEConfig,
        rope | {"scaling": {"type": "Yarn"}},
        r"^r\.scaling\.type: unknown rope_scaling type 'Yarn'\. Did you mean 'yarn'",
        path="r",
    )
    _fails(
        ComplexRoPEConfig,
        rope | {"scaling": {"rope_factor": 4.0}},
        r"^r\.scaling\.rope_factor: unknown key\..*Valid keys for NoScaling",
        path="r",
    )


def test_to_plain_names_registered_types_and_decodes_back():
    """A family member leads with its YAML ``type`` (not its class name), enums are
    values, tuples are lists, and the dump decodes back to an equal config."""
    from torchtitan.config.configurable import to_plain
    from torchtitan.config.schema.models.common.embeddings.rope.rope import (
        ComplexRoPEConfig,
    )
    from torchtitan.config.schema.models.common.embeddings.rope.scaling import (
        YarnScaling,
    )
    from torchtitan.config.schema.protocols.sharding import Shard, ShardingConfig
    from torchtitan.config.schema.protocols.types import MeshAxisName

    rope = ComplexRoPEConfig(
        dim=16,
        max_seq_len=64,
        scaling=YarnScaling(rope_factor=4.0, original_seq_len=16),
    )
    dumped = to_plain(rope)
    assert dumped["type"] == "complex"
    assert dumped["scaling"]["type"] == "yarn"
    assert decode(ComplexRoPEConfig, dumped, path="r") == rope
    sharding = ShardingConfig(state_shardings={"w": {MeshAxisName.TP: Shard(0)}})
    assert to_plain(sharding)["state_shardings"] == {
        "w": {"tp": {"type": "shard", "dim": 0}}
    }
    assert to_plain((1, (2, 3))) == [1, [2, 3]]


def test_flux_tokenizer_is_fixed_to_its_class():
    from torchtitan.config.schema.models.flux.trainer import FluxTrainerConfig

    _fails(
        FluxTrainerConfig,
        {"tokenizer": {"type": "huggingface"}},
        r"^tokenizer\.type: this field is fixed to FluxTokenizerContainerConfig",
        path="",
    )


# ---------------------------------------------------------------------------
# Literal and Enum
# ---------------------------------------------------------------------------


@dataclass
class _Choices:
    mode: Literal["a", "b"] = "a"
    level: Literal[1, 2] = 1
    axis: MeshAxisName = MeshAxisName.TP


def test_literal_and_enum_share_the_not_one_of_message():
    assert decode(_Choices, {"mode": "b", "level": 2}, path="p").level == 2
    _fails(_Choices, {"mode": "c"}, r"^p\.mode: 'c' is not one of 'a', 'b'\.$")
    _fails(
        _Choices,
        {"mode": "aa"},
        r"^p\.mode: 'aa' is not one of 'a', 'b'\. Did you mean 'a'\?$",
    )
    _fails(_Choices, {"level": True}, r"^p\.level: True is not one of 1, 2\.$")
    _fails(
        _Choices,
        {"axis": "tpp"},
        r"^p\.axis: 'tpp' is not one of 'dp', .* \(MeshAxisName\)\. Did you mean 'tp'\?$",
    )
    assert decode(_Choices, {"axis": "pp"}, path="p").axis is MeshAxisName.PP
