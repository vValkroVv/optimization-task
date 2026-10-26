"""Strict decoding of plain dicts into typed config dataclasses.

* Unknown keys are errors; every error carries the full config path.
* No coercion except ``int`` -> ``float`` and a lone scalar -> a one-element
  list.
* A field typed as a family base (``@config(family=...)``) dispatches on the
  mapping's ``type`` key; a field typed as one concrete class accepts only
  that class or a registered subclass.
* A mapping is layered on the field's default (keys left out keep the
  default's values); a ``type`` naming another class starts from its defaults.
* A ``__post_init__`` rejects a value by raising ``ValueError``.
"""

import reprlib
import types
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import MISSING, Field, fields, is_dataclass
from enum import Enum
from typing import Annotated, Any, ForwardRef, Literal, Union, get_args, get_origin

from .configurable import (
    Configurable,
    family_of,
    fields_with_role,
    get_field_types as _get_field_types,
    OWNER_FILLED,
)
from .errors import ConfigSchemaError, join_path, suggest
from .loader import ResetPath
from .provenance import SourceMap
from .registry import family_for_base
from .value_types import ConfigValue

_NONE_TYPE = type(None)
_SOURCE_ORIGINS: ContextVar[Mapping[str, str]] = ContextVar(
    "_SOURCE_ORIGINS", default=SourceMap()
)


_SOURCE_PATHS: ContextVar[Mapping[str, str]] = ContextVar(
    "_SOURCE_PATHS", default=SourceMap()
)


@contextmanager
def decoding_origins(
    origins: Mapping[str, str], source_paths: Mapping[str, str] | None = None
) -> Iterator[None]:
    """Carry the immutable winning-source map through nested decoding."""
    paths = source_paths or SourceMap()
    token = _SOURCE_ORIGINS.set(
        SourceMap({paths.get(key, key): value for key, value in origins.items()})
    )
    path_token = _SOURCE_PATHS.set(paths)
    try:
        yield
    finally:
        _SOURCE_ORIGINS.reset(token)
        _SOURCE_PATHS.reset(path_token)


def _is_union(origin: Any) -> bool:
    return origin is Union or origin is types.UnionType


def _format_type_name(tp: Any) -> str:
    """Format *tp* as a user would write it (``A | None``, ``list[int]``)."""
    if tp is Ellipsis:
        return "..."
    if tp is _NONE_TYPE:
        return "None"
    if isinstance(tp, list):  # the parameter list of a Callable
        return f"[{', '.join(_format_type_name(a) for a in tp)}]"
    origin = get_origin(tp)
    if origin is Annotated:
        return _format_type_name(get_args(tp)[0])
    if _is_union(origin):
        return " | ".join(_format_type_name(a) for a in get_args(tp))
    if origin is Literal:
        return f"Literal[{', '.join(repr(a) for a in get_args(tp))}]"
    if origin is not None:
        return f"{_format_type_name(origin)}[{', '.join(_format_type_name(a) for a in get_args(tp))}]"
    return getattr(tp, "__name__", None) or str(tp).replace("typing.", "")


_BOUNDED_REPR = reprlib.Repr()
_BOUNDED_REPR.maxstring = 80
_BOUNDED_REPR.maxother = 80
_BOUNDED_REPR.maxlist = _BOUNDED_REPR.maxtuple = _BOUNDED_REPR.maxdict = 10


def _make_expected_error(path: str, expected: str, value: Any) -> ConfigSchemaError:
    """Make ``<path>: expected <expected>, got <type> (<value>)``."""
    return ConfigSchemaError(
        f"{path}: expected {expected}, got {type(value).__name__} "
        f"({_BOUNDED_REPR.repr(value)})"
    )


# Parent path -> the keys whose `null` deleted nothing under it. A context
# variable rather than a parameter of every decode function.
_MISSING_CLEARED_KEYS_BY_PARENT: ContextVar[dict[ResetPath, list[str | int]]] = (
    ContextVar("_MISSING_CLEARED_KEYS_BY_PARENT", default={})
)


_DELETED_KEYS_BY_PARENT: ContextVar[dict[ResetPath, list[str | int]]] = ContextVar(
    "_DELETED_KEYS_BY_PARENT", default={}
)


@contextmanager
def checking_deleted_entries(paths: Iterable[ResetPath]) -> Iterator[None]:
    """Make every ``decode`` in the block reject *paths*, entries a ``null``
    removed during the merge, where the mapping's values forbid it."""
    deleted_by_parent: dict[ResetPath, list[str | int]] = {}
    for path in paths:
        parent, key = path[:-1], path[-1]
        deleted_by_parent.setdefault(parent, []).append(key)
    token = _DELETED_KEYS_BY_PARENT.set(deleted_by_parent)
    try:
        yield
    finally:
        _DELETED_KEYS_BY_PARENT.reset(token)


@contextmanager
def checking_missing_cleared_paths(paths: Iterable[ResetPath]) -> Iterator[None]:
    """Make every ``decode`` in the block check *paths*, the ``null`` resets
    that deleted nothing: each must name a field of the class decoded at its
    parent path. Paths no decode reaches are not checked."""
    cleared_keys_by_parent: dict[ResetPath, list[str | int]] = {}
    for path in paths:
        parent, key = path[:-1], path[-1]
        cleared_keys_by_parent.setdefault(parent, []).append(key)
    token = _MISSING_CLEARED_KEYS_BY_PARENT.set(cleared_keys_by_parent)
    try:
        yield
    finally:
        _MISSING_CLEARED_KEYS_BY_PARENT.reset(token)


def decode(
    cls: type,
    data: Any,
    *,
    path: str = "",
    origins: Mapping[str, str] | None = None,
    source_paths: Mapping[str, str] | None = None,
) -> Any:
    """Decode *data* into an instance of the dataclass *cls*."""
    # Public callers start at schema sections; literal mapping keys are
    # appended during traversal, never recovered from diagnostic strings.
    token = _RESET_PATH.set(tuple(path.split(".")) if path else ())
    try:
        if origins is None:
            return _decode_value(cls, data, path)
        with decoding_origins(origins, source_paths):
            return _decode_value(cls, data, path)
    finally:
        _RESET_PATH.reset(token)


_RESET_PATH: ContextVar[ResetPath] = ContextVar("_RESET_PATH", default=())


@contextmanager
def _reset_child(key: str | int) -> Iterator[None]:
    token = _RESET_PATH.set((*_RESET_PATH.get(), key))
    try:
        yield
    finally:
        _RESET_PATH.reset(token)


def _decode_child(
    tp: Any, value: Any, path: str, key: str | int, base: Any = None
) -> Any:
    with _reset_child(key):
        return _decode_value(tp, value, join_path(path, key), base)


def _decode_value(tp: Any, value: Any, path: str, base: Any = None) -> Any:
    """*base* is the field's default a mapping is layered on: an instance, a
    class (a class ``default_factory``), or None."""
    if tp is Any:
        return value

    # Recursive open kwargs contain portable data, never runtime objects.
    if (
        tp is ConfigValue
        or tp == "ConfigValue"
        or (isinstance(tp, ForwardRef) and tp.__forward_arg__ == "ConfigValue")
    ):
        return _decode_config_value(value, path)

    origin = get_origin(tp)
    if origin is Annotated:  # the metadata is not the decoder's business
        return _decode_value(get_args(tp)[0], value, path, base)

    if _is_union(origin):
        return _decode_union(tp, value, path, base)
    if origin is Literal:
        return _decode_literal(tp, value, path)
    if origin in (list, tuple):
        return _decode_list(tp, value, path)
    if origin in (dict, Mapping):
        return _decode_dict(tp, value, path, base)
    if is_dataclass(tp):
        return _decode_dataclass(tp, value, path, base)
    return _decode_scalar(tp, value, path)


def _decode_config_value(
    value: Any, path: str, active_containers: set[int] | None = None
) -> ConfigValue:
    """Decode portable kwargs without scalar-to-list promotion or coercion."""
    if value is None or type(value) in (bool, int, float, str):
        return value
    if isinstance(value, (list, dict)):
        active_containers = set() if active_containers is None else active_containers
        identity = id(value)
        if identity in active_containers:
            raise ConfigSchemaError(f"{path}: cyclic config values are not supported")
        active_containers.add(identity)
        try:
            if isinstance(value, list):
                return [
                    _decode_config_value(item, f"{path}[{index}]", active_containers)
                    for index, item in enumerate(value)
                ]
            result: dict[str, ConfigValue] = {}
            for key, item in value.items():
                if not isinstance(key, str):
                    raise _make_expected_error(
                        path, "a mapping with string keys", value
                    )
                result[key] = _decode_config_value(
                    item, join_path(path, key), active_containers
                )
            return result
        finally:
            active_containers.remove(identity)
    raise _make_expected_error(
        path, "a scalar, null, list, or string-keyed mapping", value
    )


def _decode_union(tp: Any, value: Any, path: str, base: Any) -> Any:
    union_args = get_args(tp)
    if value is None:
        if _NONE_TYPE in union_args:
            return None
        raise ConfigSchemaError(
            f"{path or _format_type_name(tp)}: expected {_format_type_name(tp)}, got null. "
            "Use an explicit value, or remove the key to fall back to the default."
        )

    # The first arm that decodes wins. A choice between config classes is a
    # family; the one union of two dataclasses, a placement list's layer group
    # or in-place block, shares no key between its arms.
    arm_errors: list[str] = []
    arms = [arm for arm in union_args if arm is not _NONE_TYPE]
    for arm in arms:
        try:
            return _decode_value(arm, value, path, base)
        except ConfigSchemaError as exc:
            # One arm and a container: the arm's own error names the failing
            # key or item, which beats the "expected X | None" summary.
            if len(arms) == 1 and isinstance(value, (dict, list, tuple)):
                raise
            arm_errors.append(f"as {_format_type_name(arm)}: {exc}")
    arm_details = ""
    if len(arm_errors) > 1 and isinstance(value, (dict, list, tuple)):
        arm_details = "\n  " + "\n  ".join(arm_errors)
    summary_error = _make_expected_error(
        path or _format_type_name(tp), _format_type_name(tp), value
    )
    raise ConfigSchemaError(f"{summary_error}{arm_details}")


def _make_not_one_of_error(
    path: str, value: Any, options: Any, options_note: str = ""
) -> ConfigSchemaError:
    rendered_options = ", ".join(repr(option) for option in options)
    hint = suggest(value, options) if isinstance(value, str) else ""
    return ConfigSchemaError(
        f"{path}: {_BOUNDED_REPR.repr(value)} is not one of "
        f"{rendered_options}{options_note}.{hint}"
    )


def _decode_literal(tp: Any, value: Any, path: str) -> Any:
    options = get_args(tp)
    # ``==`` would let True pass for 1 (and 1 for True).
    if any(value == o and type(value) is type(o) for o in options):
        return value
    raise _make_not_one_of_error(path, value, options)


def _decode_list(tp: Any, value: Any, path: str) -> Any:
    type_args = get_args(tp)
    is_tuple = get_origin(tp) is tuple
    is_fixed_length = is_tuple and bool(type_args) and type_args[-1] is not Ellipsis
    if not isinstance(value, (list, tuple)):
        # A lone scalar is the one-element list, for scalar item types only:
        # a scalar for a list/mapping/dataclass item is a missing bracket.
        is_promotable = (
            type_args
            and not is_fixed_length
            and get_origin(type_args[0]) not in (list, tuple, dict)
            and not is_dataclass(type_args[0])
        )
        if is_promotable and not isinstance(value, dict) and value is not None:
            # The item's own error (type, Literal options) is the useful one.
            item = _decode_value(type_args[0], value, path)
            return (item,) if is_tuple else [item]
        raise _make_expected_error(path or _format_type_name(tp), "a list", value)
    if is_fixed_length:
        if len(value) != len(type_args):
            raise ConfigSchemaError(
                f"{path}: expected {len(type_args)} items, got {len(value)} "
                f"({_BOUNDED_REPR.repr(value)})"
            )
        return tuple(
            _decode_item(item_type, item, path, i)
            for i, (item_type, item) in enumerate(zip(type_args, value))
        )
    item_type = type_args[0] if type_args else Any
    decoded_items = [
        _decode_item(item_type, item, path, i) for i, item in enumerate(value)
    ]
    return tuple(decoded_items) if is_tuple else decoded_items


def _decode_item(tp: Any, item: Any, path: str, index: int) -> Any:
    """Decode one list item; scalar-to-list promotion does not apply here.

    ``parts: [a, b]`` for ``list[list[str]]`` is a missing bracket, not two
    one-name parts.
    """
    if get_origin(tp) in (list, tuple) and not isinstance(item, (list, tuple)):
        raise _make_expected_error(join_path(path, index), "a list", item)
    return _decode_child(tp, item, path, index)


def _decode_dict(tp: Any, value: Any, path: str, base: Any = None) -> dict:
    """Decode a mapping layered on *base*, the field's default entries."""
    if not isinstance(value, dict):
        raise _make_expected_error(path or _format_type_name(tp), "a mapping", value)
    key_type, value_type = get_args(tp)
    default_entries = base if isinstance(base, Mapping) else {}
    layered_entries: dict = dict(default_entries)
    for key, entry in value.items():
        decoded_key = _decode_child(key_type, key, path, key)
        layered_entries[decoded_key] = _decode_child(
            value_type,
            entry,
            path,
            key,
            layered_entries.get(decoded_key) if isinstance(entry, dict) else None,
        )
    # Removing a default entry would leave a mapping the resolved config cannot
    # record: a reload layers the dumped entries on the default again.
    if getattr(value_type, "entries_removable", True) is False:
        for key in _DELETED_KEYS_BY_PARENT.get().get(_RESET_PATH.get(), ()):
            if key not in value:
                raise ConfigSchemaError(
                    f"{join_path(path, key)}: null cannot remove an entry of an "
                    "init table; set it to {type: skip} to leave the parameter "
                    "uninitialized"
                )
    _check_dict_resets(tp, value, path, base)
    return layered_entries


def _reset_children() -> set[str | int]:
    """Actual child keys with a reset that deleted nothing below them."""
    path = _RESET_PATH.get()
    return {
        parent[len(path)]
        for parent in _MISSING_CLEARED_KEYS_BY_PARENT.get()
        if parent[: len(path)] == path and len(parent) > len(path)
    }


def _check_dict_resets(tp: Any, value: dict, path: str, base: Any) -> None:
    default_entries = base if isinstance(base, Mapping) else {}
    value_type = get_args(tp)[1]
    open_mapping = value_type is Any or value_type in ("ConfigValue", ConfigValue)
    for key in _MISSING_CLEARED_KEYS_BY_PARENT.get().get(_RESET_PATH.get(), ()):
        if key in default_entries:
            raise ConfigSchemaError(
                f"{join_path(path, key)}: null cannot remove an entry of the "
                "default mapping; set the entry to the value you want instead"
            )
        if not open_mapping:
            raise ConfigSchemaError(
                f"{join_path(path, key)}: no entry to reset."
                f"{suggest(key, [*value, *default_entries])}"
            )
    for key in _reset_children() - value.keys():
        if key not in default_entries and not open_mapping:
            raise ConfigSchemaError(
                f"{join_path(path, key)}: no entry to reset."
                f"{suggest(key, [*value, *default_entries])}"
            )
        with _reset_child(key):
            _check_absent_resets(
                value_type, default_entries.get(key), join_path(path, key)
            )


def _check_dataclass_resets(
    tp: type, value: dict, path: str, default_instance: Any
) -> None:
    init_fields = {f.name: f for f in fields(tp) if f.init}

    def unknown_key_error(key: str | int) -> ConfigSchemaError:
        return ConfigSchemaError(
            f"{join_path(path, key)}: unknown key (reset with null)."
            f"{_config_source_hint(tp, key) if isinstance(key, str) else ''}"
            f"{suggest(key, init_fields)} "
            f"Valid keys for {_format_type_name(tp)}: {_valid_keys(tp, init_fields)}"
        )

    for key in _MISSING_CLEARED_KEYS_BY_PARENT.get().get(_RESET_PATH.get(), ()):
        # A `type: null` only leaves the dispatch to its default.
        if key not in init_fields and key != "type":
            raise unknown_key_error(key)
    for key in _reset_children() - value.keys():
        if isinstance(key, str) and key in init_fields:
            with _reset_child(key):
                _check_absent_resets(
                    _get_field_types(tp)[key],
                    _get_field_default(init_fields[key], default_instance),
                    join_path(path, key),
                )
        elif path:  # the root's unknown sections are checked by the build
            raise unknown_key_error(key)


def _check_absent_resets(tp: Any, base: Any, path: str) -> None:
    """Check the resets below *path*, which the data leaves out, against the
    class a mapping there would decode as: the default's, else ``X`` of
    ``X | None``, else the family default."""
    from .schema_paths import _selected_schema  # schema_paths imports this module

    if is_dataclass(base):
        tp = base if isinstance(base, type) else type(base)
    else:
        tp = _selected_schema(tp, {})
        # TODO: an `A | B | None` field or a family without a default has no
        # class to check against, so resets below it are not checked.
        family = family_for_base(tp) if is_dataclass(tp) else None
        if family is not None and (
            family.default is None or family.types.get(family.default) is not tp
        ):
            return
    if isinstance(tp, type) and is_dataclass(tp):
        _check_dataclass_resets(tp, {}, path, base if type(base) is tp else None)
    elif get_origin(tp) in (dict, Mapping) and len(get_args(tp)) == 2:
        _check_dict_resets(tp, {}, path, base)


def _config_source_hint(tp: type, key: str) -> str:
    field = next((item for item in fields(tp) if item.name == key), None)
    source = field.metadata.get("config_source") if field and not field.init else None
    return f" Configure {source} instead." if source else ""


def _valid_keys(tp, init_fields) -> str:
    owner_filled = fields_with_role(tp, OWNER_FILLED)
    return ", ".join(
        name + (" (filled by owner)" if name in owner_filled else "")
        for name in sorted(init_fields)
    )


def _decode_dataclass(tp: type, value: Any, path: str, base: Any = None) -> Any:
    if not isinstance(value, dict):
        raise _make_expected_error(
            path or _format_type_name(tp),
            f"a mapping of {_format_type_name(tp)} fields",
            value,
        )

    # Without ``type`` the mapping keeps the default's class.
    base_class = base if isinstance(base, type) else type(base)
    type_key_consumed = False
    if base is not None and "type" not in value and issubclass(base_class, tp):
        tp = base_class
        type_key_consumed = True
    else:
        family = family_for_base(tp)
        if family is not None:
            tp = family.resolve(value, path)
            type_key_consumed = True
        elif "type" in value:
            family = next((f for f in map(family_of, tp.__mro__) if f), None)
            if family is not None:
                resolved_class = family.resolve(value, path)
                if not issubclass(resolved_class, tp):
                    raise ConfigSchemaError(
                        f"{join_path(path, 'type')}: this field is fixed to "
                        f"{_format_type_name(tp)}; {value['type']!r} is not a variant of it"
                    )
                tp = resolved_class
                type_key_consumed = True
    # Only an instance of exactly the decoded class carries values to keep.
    default_instance = base if base is not None and type(base) is tp else None

    field_types = _get_field_types(tp)
    init_fields = {f.name: f for f in fields(tp) if f.init}
    unknown_keys = [
        key
        for key in value
        if key not in init_fields and not (type_key_consumed and key == "type")
    ]
    if unknown_keys:
        first_unknown_key = unknown_keys[0]
        raise ConfigSchemaError(
            f"{join_path(path, first_unknown_key)}: unknown key."
            f"{_config_source_hint(tp, first_unknown_key)}"
            f"{suggest(first_unknown_key, init_fields)} "
            f"Valid keys for {_format_type_name(tp)}: {_valid_keys(tp, init_fields)}"
        )
    _check_dataclass_resets(tp, value, path, default_instance)

    field_values: dict[str, Any] = {}
    missing_fields: list[str] = []
    for name, f in init_fields.items():
        if name in value:
            field_values[name] = _decode_child(
                field_types[name],
                value[name],
                path,
                name,
                _get_field_default(f, default_instance)
                if isinstance(value[name], dict)
                else None,
            )
        elif default_instance is not None:
            field_values[name] = getattr(default_instance, name)
        elif f.default is MISSING and f.default_factory is MISSING:
            missing_fields.append(name)
    if len(missing_fields) == 1:
        raise ConfigSchemaError(
            f"{join_path(path, missing_fields[0])}: missing required key"
        )
    if missing_fields:
        raise ConfigSchemaError(
            f"{path or _format_type_name(tp)}: missing required keys "
            f"{', '.join(sorted(missing_fields))}"
        )

    try:
        result = tp(**field_values)
        if isinstance(result, Configurable.Config):
            result._source_path = _SOURCE_PATHS.get().get(path, path)
            result._source_fields = SourceMap(
                {
                    name: _SOURCE_PATHS.get().get(
                        join_path(path, name), join_path(path, name)
                    )
                    for name in init_fields
                }
            )
            result._source_origins = _SOURCE_ORIGINS.get()
        return result
    except ValueError as exc:
        from .errors import ConfigFieldError

        if isinstance(exc, ConfigFieldError):
            field_path = join_path(path, exc.field)
            source_path = _SOURCE_PATHS.get().get(field_path, field_path)
            source = _SOURCE_ORIGINS.get().get(field_path)
            suffix = f" (source: {source})" if source else ""
            raise ConfigSchemaError(
                f"{source_path}: {exc}{suffix}", source=source
            ) from exc
        # A ``__post_init__`` invariant. Any other exception is a bug and
        # propagates.
        raise ConfigSchemaError(f"{path or _format_type_name(tp)}: {exc}") from exc


def _get_field_default(f: Field, default_instance: Any) -> Any:
    """Return the default a mapping for field *f* is layered on.

    A class used as ``default_factory`` is returned uncalled: some classes
    cannot be built from their own defaults.
    """
    if default_instance is not None:
        return getattr(default_instance, f.name)
    if f.default is not MISSING:
        return f.default
    if f.default_factory is not MISSING:
        factory = f.default_factory
        return factory if isinstance(factory, type) else factory()
    return None


def _decode_scalar(tp: Any, value: Any, path: str) -> Any:
    if tp is bool:
        if isinstance(value, bool):
            return value
        raise ConfigSchemaError(
            f"{path}: expected a bool (true/false), got {_BOUNDED_REPR.repr(value)}"
        )
    if tp is int:
        # bool is a subclass of int; reject it so `steps: true` fails loudly.
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigSchemaError(
                f"{path}: expected an int, got {_BOUNDED_REPR.repr(value)}"
            )
        return value
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigSchemaError(
                f"{path}: expected a float, got {_BOUNDED_REPR.repr(value)}"
            )
        return float(value)
    if tp is str:
        if not isinstance(value, str):
            raise ConfigSchemaError(
                f"{path}: expected a string, got {_BOUNDED_REPR.repr(value)}; quote it ('...') "
                "if it is meant as text"
            )
        return value
    if isinstance(tp, type) and issubclass(tp, Enum):
        enum_cls: type[Enum] = tp  # narrows for the type checker
        try:
            return enum_cls(value)
        except ValueError:
            enum_values = [member.value for member in enum_cls]
            raise _make_not_one_of_error(
                path, value, enum_values, f" ({_format_type_name(tp)})"
            ) from None
    if isinstance(tp, type) and isinstance(value, tp):
        return value
    if get_origin(tp) is Callable:
        # e.g. ``Function.Config.fn``: a callable has no YAML spelling.
        raise ConfigSchemaError(
            f"{path}: a callable, set from code only, not from YAML"
        )
    raise _make_expected_error(
        path or _format_type_name(tp), _format_type_name(tp), value
    )
