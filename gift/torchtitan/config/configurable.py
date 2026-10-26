# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import builtins
import dataclasses
import functools
import inspect
import sys
import importlib
import logging
from collections.abc import Callable, Iterable, Iterator, Mapping
from copy import deepcopy
from dataclasses import dataclass, fields, replace
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    ClassVar,
    TypeVar,
    cast,
    get_args,
    get_origin,
)

from .provenance import ConfigMetadata, SourceMap

try:
    from typing import dataclass_transform
except ImportError:  # Python 3.10
    from typing_extensions import dataclass_transform

if TYPE_CHECKING:
    from .validation import Issue, ValidationContext

_C = TypeVar("_C", bound=type)
_T = TypeVar("_T")
_ConfigT = TypeVar("_ConfigT", bound="Configurable.Config")

logger = logging.getLogger(__name__)


def iter_container_leaves(value, parent, key, path, *, _in_tuple=False):
    """Walk mutable config containers; tuples are reserved for plain values."""
    if isinstance(value, (list, dict, tuple)):
        items = value.items() if isinstance(value, dict) else enumerate(value)
        for index, item in items:
            yield from iter_container_leaves(
                item,
                value,
                index,
                f"{path}.{index}",
                _in_tuple=_in_tuple or isinstance(value, tuple),
            )
    else:
        if _in_tuple and isinstance(value, Configurable.Config):
            raise ValueError(
                f"{path}: config children inside tuples are unsupported; use a list"
            )
        yield path, value, parent, key


def import_spec(spec: str) -> Any:
    """Import ``"module:Qual.Name"`` and return the attribute."""
    module_name, _, qualname = spec.partition(":")
    obj: Any = importlib.import_module(module_name)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


class Family:
    """One family of swappable components, stored on its base class.

    ``types`` maps each YAML ``type`` name to its config class; ``default`` is
    used when the key is absent (None makes it required).
    """

    __slots__ = ("name", "default", "base", "types")

    def __init__(self, name: str, default: str | None, base: type) -> None:
        self.name = name
        self.default = default
        self.base = base
        self.types: dict[str, type] = {}

    def resolve(self, value: dict[str, Any], path: str) -> type:
        """Pick the concrete config class for the mapping *value*."""
        from .errors import ConfigSchemaError, join_path, suggest

        key = value.get("type", self.default)
        if key is None:
            raise ConfigSchemaError(
                f"{path}: missing required key 'type'. "
                f"Available {self.name} types: {', '.join(sorted(self.types))}"
            )
        if not isinstance(key, str) or key not in self.types:
            raise ConfigSchemaError(
                f"{join_path(path, 'type')}: unknown {self.name} type {key!r}."
                f"{suggest(key, self.types)} "
                f"Available: {', '.join(sorted(self.types))}"
            )
        return self.types[key]


_FAMILY_ATTR = "__component_family__"


def family_of(cls: type) -> Family | None:
    """The :class:`Family` declared on *cls* itself (not inherited)."""
    return cls.__dict__.get(_FAMILY_ATTR)


def type_name_of(cls: type) -> str | None:
    """The YAML ``type`` *cls* is registered under, or None."""
    family = next((f for f in map(family_of, cls.__mro__) if f), None)
    if family is None:
        return None
    return next((n for n, c in family.types.items() if c is cls), None)


def to_plain(value: Any, *, require_reloadable: bool = True) -> Any:
    """Convert *value* to plain YAML / JSON data the decoder reads back.

    ``require_reloadable=False`` is for diagnostic logs, including internal
    runtime trees and converted schemas; its output is not config YAML.

    A registered family member leads with its ``type`` (``Zeros()`` would
    otherwise dump as ``{}``). Other objects dump as their ``repr``, which no
    longer decodes.
    """
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        serialization_error = getattr(type(value), "_yaml_serialization_error", None)
        if isinstance(value, ConfigMetadata) and value._get_metadata(
            "materialized", False
        ):
            serialization_error = (
                "runtime model trees are internal; save the model config with "
                "its converters"
            )
        if require_reloadable and serialization_error is not None:
            raise ValueError(f"{type(value).__qualname__}: {serialization_error}")
        type_name = type_name_of(type(value))
        family = next(
            (family_of(base) for base in type(value).__mro__ if family_of(base)), None
        )
        if require_reloadable and family is not None and type_name is None:
            raise ValueError(
                f"{type(value).__qualname__} has no registered YAML type in family "
                f"{family.name!r}; register a stable schema type or save the compact "
                "model config with its converters to recreate the runtime tree"
            )
        head = {} if type_name is None else {"type": type_name}
        if not require_reloadable and type_name is None:
            head = {"__class__": f"{type(value).__module__}.{type(value).__qualname__}"}
        result = head | {
            f.name: to_plain(
                getattr(value, f.name), require_reloadable=require_reloadable
            )
            for f in fields(value)
            if f.init
        }
        return result
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (list, tuple)):
        return [to_plain(item, require_reloadable=require_reloadable) for item in value]
    if isinstance(value, dict):
        return {
            to_plain(key, require_reloadable=require_reloadable): to_plain(
                item, require_reloadable=require_reloadable
            )
            for key, item in value.items()
        }
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if not callable(value):
        logger.warning(
            f"Config value of type {type(value).__name__} has no YAML spelling"
        )
    return repr(value)


def component_family(name: str, *, default: str | None = None):
    """Mark a class as the base of a component family selected by YAML ``type``.

    Put it *above* ``@dataclass`` (``slots=True`` returns a new class object).
    """

    def wrap(cls: _C) -> _C:
        if _FAMILY_ATTR in cls.__dict__:
            raise TypeError(f"{cls.__qualname__} already declares a family")
        setattr(cls, _FAMILY_ATTR, Family(name, default, cls))
        return cls

    return wrap


def component(type_name: str):
    """Register a class as ``type: <type_name>`` of the nearest family in its MRO.

    The MRO includes the class itself, so a base can be its own default. Put
    it *above* ``@dataclass``.
    """

    def wrap(cls: _C) -> _C:
        for base in cls.__mro__:
            family = family_of(base)
            if family is not None:
                break
        else:
            raise TypeError(f"{cls.__qualname__}: no @component_family base in its MRO")
        previous = family.types.get(type_name)
        if previous is not None and previous is not cls:
            raise TypeError(
                f"{family.name} type {type_name!r} is declared twice: "
                f"{previous.__qualname__} and {cls.__qualname__}"
            )
        family.types[type_name] = cls
        return cls

    return wrap


class Configurable:
    """Base class for all configurable components.

    Every configurable class:
    - Inherits from Configurable (or Module for nn.Module components)
    - Has a ``Config`` class attribute: a ``@config`` dataclass subclassing
      ``Configurable.Config``, either nested in the class body or bound by
      alias (``Config = MyConfig``)
    - Gets build() auto-wired via __init_subclass__ (no manual override needed)
    - Accepts ``__init__(self, config: Config)``

    build() has two modes:
    - No kwargs: ``self.owner()(config=<a shallow copy of the config>)``
    - With kwargs (runtime objects not in config): forwarded to
      ``self.owner()(config=..., **kwargs)``.  Used by non-model
      Configurables (tokenizer, dataloader, optimizer, etc.)
      that receive runtime objects at construction time.

    Declare every Config with ``@config``, never a bare ``dataclass``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(ConfigMetadata):
        """Base config class for all configurable components."""

        _yaml_serialization_error: ClassVar[str | None] = None
        """Reason this runtime-only schema cannot be persisted as typed YAML."""

        _owner: ClassVar["Callable[..., Any] | None"] = None
        _owner_spec: ClassVar[str | None] = None
        """``"module:Qual.Name"`` of a runtime constructor or factory, imported lazily."""

        @classmethod
        def derive(
            cls: type[_ConfigT],
            owner: "Configurable.Config",
            *,
            field_sources: Mapping[str, str],
            **values,
        ) -> _ConfigT:
            """Build renamed fields from an owner's fields, carrying source paths.

            This records dependencies where values are produced; matching
            scalar values cannot identify their original configuration fields.
            """
            duplicate = set(field_sources) & values.keys()
            if duplicate:
                raise ValueError(
                    f"derived fields also supplied as values: {sorted(duplicate)}"
                )
            instance = cls(
                **{
                    **{
                        target: deepcopy(getattr(owner, source))
                        for target, source in field_sources.items()
                    },
                    **values,
                }
            )
            instance._source_path = owner._source_path
            instance._source_origins = owner._source_origins
            instance._source_fields = SourceMap(
                {
                    target: owner._source_fields.get(
                        source, f"{owner._source_path}.{source}".lstrip(".")
                    )
                    for target, source in field_sources.items()
                }
            )
            return instance

        @classmethod
        def owner(cls) -> Any:
            """The runtime constructor or factory this config builds.

            Bindings belong to this exact config class. A schema-only variant
            that intentionally reuses its parent's runtime explicitly declares
            ``_owner_spec = ParentConfig._owner_spec``.
            """
            owner = cls.__dict__.get("_owner")
            if owner is None:
                owner_spec = cls.__dict__.get("_owner_spec")
                if owner_spec is not None:
                    owner = import_spec(owner_spec)
                    cls._owner = owner
                if owner is None:
                    raise NotImplementedError(
                        f"{cls.__name__} has no owner class. Bind it to a "
                        "Configurable subclass (nested Config or `Config = ...`) "
                        "or set _owner_spec."
                    )
            return owner

        def validate(self, ctx: "ValidationContext") -> "Iterable[Issue]":
            """Checks that need the stage or a sibling; local invariants belong
            in ``__post_init__``. Issue paths are relative to this node."""
            return ()

        def to_dict(self) -> dict:
            """Serialize to plain data the decoder reads back."""
            return to_plain(self)

        def tree_fields(self) -> tuple:
            """The fields tree walks (``traverse``) follow; a config that keeps
            templates beside a derived tree narrows this."""
            return fields(self)

        def traverse(
            self, config_cls: type[_T], *, _prefix: str = ""
        ) -> Iterator[tuple[str, _T, object, str | int]]:
            """Yield ``(fqn, config, parent, field_name)`` for every nested config of *config_cls*.

            Recursively traverses config fields and nested lists and dictionaries.
            Tuples may contain plain values, but not config children.
            The *fqn* mirrors the module FQN that ``build()`` would produce
            (e.g. ``"layers.0.feed_forward.w1"``).

            *parent* and *field_name* allow replacing the config in the tree::

                for fqn, cfg, parent, attr in model_config.traverse(Linear.Config):
                    setattr(parent, attr, NewConfig(...))
            """
            for f in self.tree_fields():
                val = getattr(self, f.name)
                fqn = f"{_prefix}.{f.name}" if _prefix else f.name
                for path, item, parent, key in iter_container_leaves(
                    val, self, f.name, fqn
                ):
                    if isinstance(item, config_cls):
                        yield path, item, parent, key
                    elif hasattr(item, "traverse"):
                        yield from item.traverse(config_cls, _prefix=path)

        def _copy(self: _ConfigT, **changes) -> _ConfigT:
            """``dataclasses.replace`` plus the ``init=False`` fields it resets
            (e.g. ``TrainerConfig.model_spec``)."""
            from .errors import ConfigFieldError, ConfigSchemaError

            try:
                copy = replace(self, **changes)
            except ConfigFieldError as exc:
                path = self._source_fields.get(exc.field)
                if not path:
                    raise
                source = self._source_origins.get(path)
                suffix = f" (source: {source})" if source else ""
                raise ConfigSchemaError(
                    f"{path}: {exc}{suffix}", source=source
                ) from exc
            for f in fields(self):
                if not f.init:
                    setattr(copy, f.name, getattr(self, f.name))
            copy.copy_metadata_from(self)
            return copy

        def build(self, **kwargs):
            """Construct the owning class. Auto-wired by __init_subclass__.

            Two modes:
            - No kwargs: ``self.owner()(config=<a shallow copy of self>)``
            - With kwargs (runtime objects not in config): forwarded to
              ``self.owner()(config=..., **kwargs)``.  Used by non-model
              Configurables (tokenizer, dataloader, optimizer, etc.)
              that receive runtime objects at construction time.
            """
            owner = self.owner()
            # Lazy: structured_logger imports torch; the config package must not.
            from torchtitan.observability import structured_logger as sl

            with sl.log_trace_span(f"{type(self).__qualname__}.build"):
                if not kwargs:
                    return owner(config=self._copy())

                config_fields = {f.name for f in fields(self)}
                overlap = config_fields & kwargs.keys()
                if overlap:
                    raise ValueError(
                        f"build() kwargs {overlap} overlap with config fields. "
                        "Put these values in the Config, not in build() kwargs."
                    )
                return owner(config=self._copy(), **kwargs)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "Config" in cls.__dict__:
            config_cls = cls.__dict__["Config"]
            if issubclass(config_cls, Configurable.Config):
                # Auto-wire build() to construct this class
                config_cls._owner = cls


class FieldRole:
    """What a ``None`` in a field means, carried in its annotation."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return self.name


OWNER_FILLED = FieldRole("owner_filled")
DERIVED = FieldRole("derived")
_FieldT = TypeVar("_FieldT")

OwnerFilled = Annotated[_FieldT | None, OWNER_FILLED]
"""None until the owning config fills it; explicit YAML values must agree."""
Derived = Annotated[_FieldT | None, DERIVED]
"""None means "compute it": the config or its owner does, by materialize."""


@functools.cache
def get_field_types(cls: type) -> dict[str, Any]:
    """Resolve init-field annotations, skipping runtime-only non-init types."""
    names = {f.name for f in dataclasses.fields(cls) if f.init}
    found = {}
    for base in reversed(cls.__mro__):
        namespace = vars(sys.modules[base.__module__])
        for name, tp in inspect.get_annotations(base).items():
            if name not in names:
                continue
            if isinstance(tp, str):
                tp = eval(tp, namespace, dict(vars(base)))  # noqa: S307
            found[name] = tp
    return found


def _contains_field_role(tp) -> bool:
    if get_origin(tp) is Annotated and any(
        isinstance(tag, FieldRole) for tag in get_args(tp)[1:]
    ):
        return True
    return any(_contains_field_role(arg) for arg in get_args(tp))


@functools.cache
def fields_with_role(cls: type, role: FieldRole) -> frozenset[str]:
    """Names of init fields carrying a top-level role; reject nested markers."""
    found = set()
    for name, tp in get_field_types(cls).items():
        if get_origin(tp) is Annotated:
            inner, *metadata = get_args(tp)
            if role in metadata:
                found.add(name)
        else:
            inner = tp
        if _contains_field_role(inner):
            raise TypeError(
                f"{cls.__qualname__}.{name}: a field role must be the outermost "
                "annotation; use OwnerFilled[T] or Derived[T] without '| None'"
            )
    return frozenset(found)


@dataclass_transform(kw_only_default=True)
def config(
    cls: _C | None = None,
    /,
    *,
    family: str | None = None,
    default: str | None = None,
    type: str | None = None,
) -> Any:
    """``dataclass(kw_only=True, slots=True)`` for a ``Configurable.Config`` subclass.

    ``family`` (with ``default``) makes the class a family base; ``type``
    registers it under that YAML name (:func:`component`). Both may be given.
    """

    def wrap(c: _C) -> _C:
        if not (isinstance(c, builtins.type) and issubclass(c, Configurable.Config)):
            raise TypeError(
                f"@config expects a Configurable.Config subclass, got {c!r}"
            )
        # Registration must see the final class: slots=True returns a new one.
        c = cast(_C, dataclass(kw_only=True, slots=True)(c))
        if family is not None:
            c = component_family(family, default=default)(c)
        elif default is not None:
            raise TypeError("@config: `default` needs `family`")
        if type is not None:
            c = component(type)(c)
        return c

    return wrap if cls is None else wrap(cls)


def independent_config_positions(config: _ConfigT) -> _ConfigT:
    """Copy every config position independently, including aliases within a block."""

    def separate_containers(value):
        if isinstance(value, list):
            return [separate_containers(item) for item in value]
        if isinstance(value, tuple):
            return tuple(separate_containers(item) for item in value)
        if isinstance(value, dict):
            return {key: separate_containers(item) for key, item in value.items()}
        return value

    result = deepcopy(config)
    for field in result.tree_fields():
        if field.init:
            setattr(
                result, field.name, separate_containers(getattr(result, field.name))
            )
    for _, child, parent, key in result.traverse(Configurable.Config):
        replacement = independent_config_positions(child)
        if isinstance(parent, list) and isinstance(key, int):
            parent[key] = replacement
        elif isinstance(parent, dict):
            parent[key] = replacement
        elif isinstance(key, str):
            setattr(parent, key, replacement)
        else:
            raise AssertionError("config position has no writable parent")
    return result
