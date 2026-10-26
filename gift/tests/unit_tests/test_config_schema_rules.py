"""Rules over every schema class a YAML document can reach.

The walk starts at each trainer Config and ``model:`` and follows field types, union
arms, container arguments and component family members.
"""

import dataclasses
import json
import types
from typing import Annotated, Any, get_args, get_origin, get_type_hints, Union

from tests.unit_tests.torch_free import run_blocked


def _reachable_classes() -> set[type]:
    from torchtitan.config.configurable import import_spec
    from torchtitan.config.decode import _get_field_types
    from torchtitan.config.registry import family_for_base, TRAINER_TYPES
    from torchtitan.config.sections import ModelSection, TrainerSection

    roots: list[Any] = [ModelSection, TrainerSection]
    roots += [import_spec(t.spec) for t in TRAINER_TYPES.values()]

    seen: set[type] = set()
    pending = list(roots)
    while pending:
        tp = pending.pop()
        origin = get_origin(tp)
        if origin is Annotated:
            pending.append(get_args(tp)[0])
        elif origin is not None:  # unions, list / tuple / dict arguments
            pending += get_args(tp)
        elif dataclasses.is_dataclass(tp) and isinstance(tp, type) and tp not in seen:
            seen.add(tp)
            family = family_for_base(tp)
            if family is not None:
                pending += family.types.values()
            pending += _get_field_types(tp).values()
            for f in dataclasses.fields(tp):
                if f.default_factory is not dataclasses.MISSING:
                    try:
                        pending.append(type(f.default_factory()))
                    except (TypeError, ValueError):
                        pass  # a default the document must override (e.g. rl's generator)
    return seen


def _admits_none(tp: Any) -> bool:
    if get_origin(tp) is Annotated:
        return _admits_none(get_args(tp)[0])
    if get_origin(tp) in (Union, types.UnionType):
        return any(_admits_none(arm) for arm in get_args(tp))
    return tp is type(None) or tp is None


def optional_fields_with_non_none_defaults() -> list[str]:
    from torchtitan.config.decode import _get_field_types

    found = []
    for cls in _reachable_classes():
        hints = _get_field_types(cls)
        for f in dataclasses.fields(cls):
            if not f.init or not _admits_none(hints[f.name]):
                continue
            if f.default_factory is not dataclasses.MISSING or (
                f.default is not dataclasses.MISSING and f.default is not None
            ):
                found.append(f"{cls.__module__}.{cls.__qualname__}.{f.name}")
    return sorted(found)


def fields_defaulting_to_abstract_family_bases() -> list[str]:
    from torchtitan.config.configurable import family_of

    found = []
    for cls in _reachable_classes():
        for f in dataclasses.fields(cls):
            if not f.init:
                continue
            if f.default_factory is not dataclasses.MISSING:
                try:
                    default = f.default_factory()
                except (TypeError, ValueError):
                    continue
            elif f.default is not dataclasses.MISSING:
                default = f.default
            else:
                continue
            family = family_of(type(default))
            if family is not None and family.default is None:
                found.append(f"{cls.__module__}.{cls.__qualname__}.{f.name}")
    return sorted(found)


def non_init_fields() -> list[str]:
    """Fields the decoder never sees (``init=False``)."""
    return sorted(
        f"{cls.__module__}.{cls.__qualname__}.{f.name}"
        for cls in _reachable_classes()
        for f in dataclasses.fields(cls)
        if not f.init
    )


def _decodable(tp: Any) -> bool:
    """Whether the decoder has a rule for the shape of *tp*."""
    import enum
    from collections.abc import Mapping
    from typing import Literal
    from torchtitan.config.value_types import ConfigValue

    if tp == ConfigValue:
        return True

    origin = get_origin(tp)
    if origin is Annotated:
        return _decodable(get_args(tp)[0])
    if origin in (Union, types.UnionType, list, tuple, dict, Mapping):
        return all(_decodable(a) for a in get_args(tp) if a is not Ellipsis)
    if origin is Literal:
        return True
    if origin is not None:  # Callable[...], Sequence[...], ...
        return False
    return (
        tp is Any
        or tp is type(None)
        or (
            isinstance(tp, type)
            and (
                tp in (bool, int, float, str, dict)  # bare dict: an open mapping
                or issubclass(tp, enum.Enum)
                or dataclasses.is_dataclass(tp)
            )
        )
    )


def _arms(tp: Any) -> list[Any]:
    if get_origin(tp) is Annotated:
        return _arms(get_args(tp)[0])
    if get_origin(tp) in (Union, types.UnionType):
        return [a for arm in get_args(tp) for a in _arms(arm)]
    return [tp]


def has_dataclass_union(tp: Any) -> bool:
    """A union of two or more dataclasses anywhere in *tp*: the field itself,
    or a ``dict`` value, ``list`` / ``tuple`` item, ``Mapping``, ``Annotated``
    or ``| None`` inside it."""
    if sum(dataclasses.is_dataclass(a) for a in _arms(tp)) > 1:
        return True
    return any(has_dataclass_union(arg) for arg in get_args(tp) if arg is not Ellipsis)


def dataclass_unions() -> list[str]:
    """Fields whose type holds a union admitting two or more dataclasses."""
    from torchtitan.config.decode import _get_field_types

    return sorted(
        f"{cls.__module__}.{cls.__qualname__}.{name}"
        for cls in _reachable_classes()
        for name, tp in _get_field_types(cls).items()
        if has_dataclass_union(tp)
    )


def undecodable_fields() -> list[str]:
    from torchtitan.config.decode import _get_field_types

    return sorted(
        f"{cls.__module__}.{cls.__qualname__}.{name}"
        for cls in _reachable_classes()
        for name, tp in _get_field_types(cls).items()
        if not _decodable(tp)
    )


def _run(function: str) -> Any:
    code = (
        "import json\n"
        f"from tests.unit_tests.test_config_schema_rules import {function}\n"
        f"print(json.dumps({function}()))\n"
    )
    return json.loads(run_blocked(code).stdout.strip().splitlines()[-1])


def test_optional_fields_default_to_none():
    """``null`` resets to the default, so ``X | None`` fields default to ``None``."""
    assert _run("optional_fields_with_non_none_defaults") == []


def test_no_field_defaults_to_an_abstract_family_base():
    """A field of a family without a default ``type`` is required, never defaulted.

    The abstract base would decode silently and fail only when the trainer builds it.
    """
    assert _run("fields_defaulting_to_abstract_family_bases") == []


def test_only_model_spec_and_documented_fp8_flags_are_set_programmatically():
    """``init=False`` takes a field off the YAML surface, so the set is pinned."""
    found = _run("non_init_fields")
    assert [f for f in found if f.endswith(".model_spec")] == [
        "torchtitan.config.schema.experiments.ft.trainer.FaultTolerantTrainerConfig.model_spec",
        "torchtitan.config.schema.experiments.graph_trainer.trainer.GraphTrainerConfig.model_spec",
        "torchtitan.config.schema.experiments.rl.grpo.RLTrainerConfig.model_spec",
        "torchtitan.config.schema.models.flux.trainer.FluxTrainerConfig.model_spec",
        "torchtitan.config.schema.trainer.TrainerConfig.model_spec",
    ]
    from torchtitan.config.configs import FP8Config
    from torchtitan.config.schema.models.common.ffn import (
        FeedForwardConfig,
    )
    from torchtitan.config.schema.models.common.moe.experts.gmm import (
        GroupedExpertsConfig,
    )

    dense_flags = {
        "use_shared_quantizer": "dense_use_shared_quantizer",
        "dense_fused_swiglu_quant": "dense_fused_swiglu_quant",
    }
    derived_flags = {
        FeedForwardConfig: dense_flags,
        GroupedExpertsConfig: {
            "use_float8_grouped_gemm": "sparse_use_grouped_gemm",
            "deep_gemm_num_sms": "sparse_deep_gemm_num_sms",
            "sparse_fused_swiglu_quant": "sparse_fused_swiglu_quant",
        },
    }
    expected = set()
    fp8_types = get_type_hints(FP8Config)
    for cls in _reachable_classes():
        sources = next(
            (
                source_fields
                for base, source_fields in derived_flags.items()
                if issubclass(cls, base)
            ),
            None,
        )
        if sources is None:
            continue
        config_fields = {field.name: field for field in dataclasses.fields(cls)}
        field_types = get_type_hints(cls)
        for name, source in sources.items():
            field = config_fields[name]
            assert not field.init
            assert field.metadata["config_source"] == f"fp8.{source}"
            assert field_types[name] == fp8_types[source]
            expected.add(f"{cls.__module__}.{cls.__qualname__}.{name}")
    # Runtime trees are separate schemas. No other hidden schema state is allowed.
    assert {name for name in found if not name.endswith(".model_spec")} == expected


def test_no_field_is_a_union_of_dataclasses():
    """A choice between config classes is a family (``type`` dispatch); a nested
    ``dict[str, A | B]`` counts too."""

    @dataclasses.dataclass
    class A:
        pass

    @dataclasses.dataclass
    class B:
        pass

    # The check itself must see through containers, not only the top level.
    assert has_dataclass_union(dict[str, tuple[A | B, ...]] | None)
    assert not has_dataclass_union(dict[str, A] | list[B])
    # Placement lists (``layers``, flux's ``double_blocks`` / ``single_blocks``)
    # take a layer group ({block | cycle, repeat}) or a block written in place.
    # The two share no key, so the first arm that decodes is the only one that
    # can; no other field may hold such a union.
    placement_fields = ("layers", "double_blocks", "single_blocks")
    unions = _run("dataclass_unions")
    assert [f for f in unions if f.rsplit(".", 1)[-1] not in placement_fields] == []


def test_every_field_annotation_is_decodable():
    """Every init field annotation resolves and has a decoder rule.

    The pinned exception, ``Function.Config.fn``, is a ``Callable`` only code sets.
    """
    assert _run("undecodable_fields") == [
        "torchtitan.config.function.Function.Config.fn"
    ]


def test_the_walk_reaches_the_schema():
    """Guards the walk itself: the rules are vacuous if it finds nothing."""
    code = (
        "from tests.unit_tests.test_config_schema_rules import _reachable_classes\n"
        "names = {c.__qualname__ for c in _reachable_classes()}\n"
        "assert len(names) > 100, len(names)\n"
        "assert {'ParallelismConfig', 'HybridEPTokenDispatcherConfig', "
        "'GraphTrainerCompileConfig', 'MetricsProcessorConfig'} <= names\n"
    )
    run_blocked(code)


def test_pre_submit_requires_loss_and_dataloader():
    """``null`` deletes a section; for ``loss`` / ``dataloader`` that leaves the
    document without it, which pre-submit reports instead of decoding an
    abstract base the trainer cannot build."""
    from tests.unit_tests.torch_free import REPO_ROOT

    experiment = REPO_ROOT / "configs" / "experiments" / "gift_moe_debug.yaml"
    for section in ("loss", "dataloader"):
        code = (
            "from torchtitan.config import run_pre_submit\n"
            f"result = run_pre_submit({str(experiment)!r}, [({section!r}, 'null')])\n"
            "print(*result.report.errors, sep='\\n')\n"
        )
        errors = run_blocked(code).stdout
        assert f"{section}: missing required key" in errors, errors


def field_role_violations() -> list[str]:
    from torchtitan.config.configurable import FieldRole, get_field_types
    from torchtitan.config.schema.protocols.module import ModuleConfig

    def contains_role(tp):
        return (
            get_origin(tp) is Annotated
            and any(isinstance(tag, FieldRole) for tag in get_args(tp)[1:])
        ) or any(contains_role(arg) for arg in get_args(tp))

    violations = []
    for cls in _reachable_classes():
        hints = get_field_types(cls)
        for item in dataclasses.fields(cls):
            if not item.init or not contains_role(hints[item.name]):
                continue
            tp = hints[item.name]
            where = f"{cls.__module__}.{cls.__qualname__}.{item.name}"
            if get_origin(tp) is not Annotated or contains_role(get_args(tp)[0]):
                violations.append(f"{where}: role must be outermost")
            if (
                item.default is not None
                or item.default_factory is not dataclasses.MISSING
            ):
                violations.append(f"{where}: role must default to None")
            if not issubclass(cls, ModuleConfig):
                violations.append(f"{where}: role belongs on a ModuleConfig")
    return sorted(violations)


def test_field_roles_have_consistent_annotations_and_defaults():
    assert _run("field_role_violations") == []
