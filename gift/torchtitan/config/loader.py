"""YAML loading, preset imports and deep merge.

Merge semantics (import forms, ``null`` resets, repeated imports) are described
in docs/config.md, "Imports and presets".
"""

import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import yaml
from yaml.constructor import ConstructorError

from .errors import ConfigFileError, join_path
from .provenance import SourceMap


@dataclass(frozen=True, kw_only=True)
class ImportNode:
    """One node of the resolved import graph, kept for run metadata."""

    path: str
    """Path of the imported file, relative to the current directory when shorter."""

    label: str = ""
    """Section taken by a section import; empty for a full-config import."""

    imports: tuple["ImportNode", ...] = ()
    """Direct imports, in merge order."""

    def to_dict(self) -> dict[str, Any]:
        """Return nested metadata, omitting empty labels and import lists."""
        node: dict[str, Any] = {"path": self.path}
        if self.label:
            node["label"] = self.label
        if self.imports:
            node["imports"] = [child.to_dict() for child in self.imports]
        return node


ResetPath = tuple[str | int, ...]
"""Actual mapping keys and list indices, without lossy dotted serialization."""


class ReplaceMapping(dict[str, Any]):
    """A YAML mapping that discards earlier values at the same path."""


@dataclass(frozen=True, kw_only=True)
class TypeChange:
    path: str
    old_type: str
    new_type: str
    source: str
    inherited: tuple[tuple[str, str], ...]


@dataclass(frozen=True, kw_only=True)
class RawDocument:
    """A merged, still-untyped config document."""

    data: dict[str, Any]
    """Values after import resolution; a ``null`` leaves no entry."""

    graph: ImportNode
    """Entry file and its import tree, in merge order."""

    local_keys: frozenset[str] = field(default_factory=frozenset)
    """Top-level keys the entry file declares itself."""

    type_changes: tuple[TypeChange, ...] = ()
    deleted_paths: tuple[ResetPath, ...] = ()
    """Entries a later layer's ``null`` removed from an earlier one."""
    source_paths: Mapping[str, str] = field(default_factory=SourceMap)
    """Canonical paths mapped to the original spelling in the winning layer."""

    origins: Mapping[str, str] = field(default_factory=SourceMap)
    """Winning file or override source for canonical paths, including resets."""

    missing_cleared_paths: tuple[ResetPath, ...] = ()
    """Paths whose ``null`` deleted nothing, so a schema-aware
    caller can reject a reset of an unknown key."""


def deep_merge(
    base: dict[str, Any],
    override: dict[str, Any],
    *,
    _path: ResetPath = (),
    _deleted: list[ResetPath] | None = None,
    _missing: list[ResetPath] | None = None,
) -> dict[str, Any]:
    """Merge config mappings without modifying either input.

    ``None`` in *override* means the default and is never stored: it deletes
    the key when *base* has it, otherwise its path goes to *_missing* for the
    schema check. A new mapping left empty only by such nulls is not created.
    Containers that are not merged recursively stay shared with the inputs.
    """

    def discard(path: ResetPath) -> None:
        for paths in (_deleted, _missing):
            if paths is not None:
                paths[:] = [p for p in paths if p[: len(path)] != path]

    def merge_new(value: Any, path: ResetPath) -> Any:
        if isinstance(value, dict):
            return deep_merge(
                {}, value, _path=path, _deleted=_deleted, _missing=_missing
            )
        if isinstance(value, list):
            return [merge_new(item, (*path, i)) for i, item in enumerate(value)]
        return value

    if isinstance(override, ReplaceMapping):
        discard(_path)
    merged = {} if isinstance(override, ReplaceMapping) else dict(base)
    for key, value in override.items():
        path = (*_path, key)
        if value is None:
            # A reset supersedes the resets recorded below it.
            deleted_before = _deleted is not None and path in _deleted
            discard(path)
            if key in merged or deleted_before:
                merged.pop(key, None)
                if _deleted is not None:
                    _deleted.append(path)
            elif _missing is not None:
                _missing.append(path)
        elif (
            isinstance(value, dict)
            and not isinstance(value, ReplaceMapping)
            and isinstance(merged.get(key, {}), dict)
        ):
            merged_value = deep_merge(
                merged.get(key, {}),
                value,
                _path=path,
                _deleted=_deleted,
                _missing=_missing,
            )
            # Not created when only nulls fill it; an explicit `{}` is a value.
            if merged_value or key in merged or not value:
                merged[key] = merged_value
                for paths in (_deleted, _missing):
                    if paths is not None and path in paths:
                        paths.remove(path)
        else:
            discard(path)
            merged[key] = merge_new(value, path)
    return merged


class ConfigYAMLLoader(yaml.SafeLoader):
    """``yaml.SafeLoader`` with YAML 1.2 core scalars and duplicate-key errors.

    Duplicate keys are checked before ``<<`` merge keys expand, so a merged
    key may still be overridden. A subclass, so ``yaml.safe_load`` elsewhere
    keeps PyYAML's 1.1 behaviour.
    """

    def __init__(self, stream):
        super().__init__(stream)
        self._checked_mapping_nodes: set[int] = set()

    def flatten_mapping(self, node):
        # Check explicit keys before merge expansion adds inherited ones.
        # Aliases can revisit a node after it has already been flattened.
        if id(node) not in self._checked_mapping_nodes:
            self._checked_mapping_nodes.add(id(node))
            key_marks = {}
            for key_node, _ in node.value:
                if key_node.tag in (
                    "tag:yaml.org,2002:merge",
                    "tag:yaml.org,2002:value",
                ):
                    key = key_node.value
                else:
                    key = self.construct_object(key_node)
                try:
                    previous_mark = key_marks.get(key)
                except TypeError as exc:
                    raise ConstructorError(
                        "while constructing a mapping",
                        node.start_mark,
                        "found unhashable key",
                        key_node.start_mark,
                    ) from exc
                if previous_mark is not None:
                    raise ConstructorError(
                        "first occurrence of this key",
                        previous_mark,
                        f"duplicate YAML key {key!r}",
                        key_node.start_mark,
                    )
                key_marks[key] = key_node.start_mark
        super().flatten_mapping(node)


class ConfigYAMLDumper(yaml.SafeDumper):
    """Writes what ``ConfigYAMLLoader`` and ``yaml.safe_load`` both read back:
    a string either YAML version reads bare as something else (``'1e3'``,
    ``'0o10'``, ``'yes'``) is quoted."""


# YAML 1.2 core schema (yaml.org/spec/1.2.2, 10.3.2). PyYAML implements only
# 1.1, so the loader's implicit resolvers are replaced, not extended.
_CORE_SCHEMA_RESOLVERS = (
    ("null", r"^(?:~|null|Null|NULL|)$", ["~", "n", "N", ""]),
    ("bool", r"^(?:true|True|TRUE|false|False|FALSE)$", list("tTfF")),
    # int before float: a resolver list is tried in order, and "10" matches both.
    ("int", r"^(?:[-+]?[0-9]+|0o[0-7]+|0x[0-9a-fA-F]+)$", list("-+0123456789")),
    (
        "float",
        r"""^(?:[-+]?(?:\.[0-9]+|[0-9]+(?:\.[0-9]*)?)(?:[eE][-+]?[0-9]+)?
        |[-+]?\.(?:inf|Inf|INF)|\.(?:nan|NaN|NAN))$""",
        list("-+0123456789."),
    ),
)
# Assigned on the subclass only: SafeLoader keeps its resolvers.
ConfigYAMLLoader.yaml_implicit_resolvers = {}
for _name, _pattern, _first in _CORE_SCHEMA_RESOLVERS:
    _regexp = re.compile(_pattern, re.X)
    ConfigYAMLLoader.add_implicit_resolver(
        f"tag:yaml.org,2002:{_name}", _regexp, _first
    )
    # The dumper keeps its 1.1 resolvers and adds these, so it quotes a string
    # either YAML version reads as a non-string.
    ConfigYAMLDumper.add_implicit_resolver(
        f"tag:yaml.org,2002:{_name}", _regexp, _first
    )
# 1.1-only, but harmless: `<<: *defaults` keeps working.
ConfigYAMLLoader.add_implicit_resolver(
    "tag:yaml.org,2002:merge", re.compile(r"^(?:<<)$"), ["<"]
)


def _construct_core_int(loader: yaml.SafeLoader, node: yaml.ScalarNode) -> int:
    """PyYAML's int constructor is 1.1: it reads ``010`` as octal 8."""
    text = loader.construct_scalar(node)
    base = 8 if text.startswith("0o") else 16 if text.startswith("0x") else 10
    return int(text, base)


ConfigYAMLLoader.add_constructor("tag:yaml.org,2002:int", _construct_core_int)


def _construct_replace(loader: ConfigYAMLLoader, node: yaml.Node) -> ReplaceMapping:
    if not isinstance(node, yaml.MappingNode):
        raise ConstructorError(
            None, None, "!replace requires a mapping", node.start_mark
        )
    return ReplaceMapping(loader.construct_mapping(node, deep=True))


def _construct_unknown_tag(loader: ConfigYAMLLoader, node: yaml.Node):
    raise ConstructorError(
        None,
        None,
        f"unknown YAML tag {node.tag!r}; use !replace to replace a mapping",
        node.start_mark,
    )


ConfigYAMLLoader.add_constructor("!replace", _construct_replace)
ConfigYAMLLoader.add_constructor(None, _construct_unknown_tag)
ConfigYAMLDumper.add_representer(
    ReplaceMapping, lambda dumper, data: dumper.represent_mapping("!replace", data)
)


def parse_yaml_value(text: str, *, source: str) -> Any:
    """Parse *text* as one YAML value with the config-file loader.

    Raises:
        ConfigFileError: If *text* is not valid YAML, cannot be constructed
            or holds an alias cycle; the message starts with *source*.
    """
    try:
        value = yaml.load(text, Loader=ConfigYAMLLoader)
    except (yaml.YAMLError, ValueError) as exc:
        first_line = str(exc).splitlines()[0]
        raise ConfigFileError(
            f"{source}: {text!r} is not a YAML value ({first_line})"
        ) from None
    _reject_cycles(value, source, "", set())
    return value


def _read_yaml_file(path: str) -> dict[str, Any]:
    """Read a UTF-8 YAML file; an empty document is ``{}``.

    Raises:
        ConfigFileError: If the file cannot be read or parsed, or the root is
            not a mapping of string keys.
    """
    try:
        with open(path, encoding="utf-8") as handle:
            file_data = yaml.load(handle, Loader=ConfigYAMLLoader)
    except FileNotFoundError as exc:
        raise ConfigFileError(f"config file not found: {path}") from exc
    except IsADirectoryError as exc:
        raise ConfigFileError(f"{path}: is a directory, not a config file") from exc
    except UnicodeDecodeError as exc:
        raise ConfigFileError(f"{path}: config file is not valid UTF-8") from exc
    except OSError as exc:
        raise ConfigFileError(f"{path}: cannot read config file: {exc}") from exc
    except (yaml.YAMLError, ValueError) as exc:
        # ValueError: a scalar PyYAML cannot construct (`!!int a`).
        raise ConfigFileError(f"{path}: invalid YAML: {exc}") from exc
    except RecursionError as exc:
        raise ConfigFileError(f"{path}: YAML structure nested too deep") from exc

    if file_data is None:
        return {}

    if not isinstance(file_data, dict):
        raise ConfigFileError(
            f"{path}: a config file must contain a mapping at the top level, "
            f"got {type(file_data).__name__}"
        )
    for key in file_data:
        if not isinstance(key, str):
            raise ConfigFileError(
                f"{path}: top-level config keys must be strings; got "
                f"{type(key).__name__} key {key!r}"
            )
    _reject_cycles(file_data, path, "", set())
    return file_data


def _reject_cycles(value: Any, path: str, key_path: str, active_ids: set[int]) -> None:
    """Raise if an alias makes ``value`` contain one of its own ancestors.

    PyYAML accepts ``a: &x {b: *x}``; the recursive merge passes would then
    never terminate. Only ancestors are tracked, so reusing an anchor side by
    side stays allowed.
    """
    if isinstance(value, dict):
        child_entries = (
            (f"{key_path}.{k}" if key_path else str(k), v) for k, v in value.items()
        )
    elif isinstance(value, list):
        child_entries = ((f"{key_path}[{i}]", v) for i, v in enumerate(value))
    else:
        return
    if id(value) in active_ids:
        raise ConfigFileError(
            f"{path}: recursive YAML structure (alias cycle) at {key_path or '<root>'}"
        )
    active_ids.add(id(value))
    for child_key_path, child in child_entries:
        _reject_cycles(child, path, child_key_path, active_ids)
    active_ids.discard(id(value))


_ACCEPTED_IMPORT_FORMS = (
    "`imports: path` or `imports: [path, ...]` (full-config imports, merged in "
    "order) or `imports: {section: path, ...}` (section imports: the named "
    "section of each file); the two forms cannot be mixed in one file"
)


@dataclass(frozen=True, kw_only=True)
class _ImportSpec:
    """An import declaration before its path and contents are resolved."""

    path: str
    """Absolute, or relative to the declaring file's directory."""

    section_name: str | None = None
    """Root section to extract; ``None`` for a full-config import."""


def _normalize_imports(raw_imports: Any, path: str) -> list[_ImportSpec]:
    """Validate the parsed ``imports`` value and return it as ordered specs.

    Raises:
        ConfigFileError: If the declaration has an unsupported form or an
            empty or non-string path or section name.
    """
    if raw_imports is None:
        return []
    if isinstance(raw_imports, str):
        _check_import_path(raw_imports, f"{path}: imports")
        return [_ImportSpec(path=raw_imports)]

    if isinstance(raw_imports, list):
        for index, import_path in enumerate(raw_imports):
            if not isinstance(import_path, str):
                hint = (
                    " (a section import belongs in the mapping form)"
                    if isinstance(import_path, dict)
                    else ""
                )
                raise ConfigFileError(
                    f"{path}: imports[{index}] must be a path string, got "
                    f"{type(import_path).__name__}{hint}. Accepted: {_ACCEPTED_IMPORT_FORMS}"
                )
            _check_import_path(import_path, f"{path}: imports[{index}]")
        return [_ImportSpec(path=import_path) for import_path in raw_imports]

    if isinstance(raw_imports, dict):
        import_specs = []
        for section_name, import_path in raw_imports.items():
            if not isinstance(section_name, str) or not section_name:
                raise ConfigFileError(
                    f"{path}: imports: section names must be non-empty strings, "
                    f"got {section_name!r}"
                )
            if not isinstance(import_path, str):
                raise ConfigFileError(
                    f"{path}: imports.{section_name} must be a path string, "
                    f"got {type(import_path).__name__}. Accepted: {_ACCEPTED_IMPORT_FORMS}"
                )
            _check_import_path(import_path, f"{path}: imports.{section_name}")
            import_specs.append(
                _ImportSpec(path=import_path, section_name=section_name)
            )
        return import_specs

    raise ConfigFileError(
        f"{path}: imports must be {_ACCEPTED_IMPORT_FORMS}; got {type(raw_imports).__name__}"
    )


def _check_import_path(import_path: str, location: str) -> None:
    if not import_path.strip():
        raise ConfigFileError(f"{location}: import path must not be empty")


def _strip_nulls(value: Any) -> Any:
    """Drop null mapping fields at any depth; a ``None`` list item is data.

    Dicts and lists are copied.
    """
    if isinstance(value, dict):
        return {k: _strip_nulls(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_strip_nulls(item) for item in value]
    return value


def load_raw_document(
    path: str,
    *,
    overrides: Sequence[dict[str, Any]] = (),
) -> RawDocument:
    """Load a YAML file, resolve its imports and merge everything in one pass.

    Args:
        path: Entry file, absolute or relative to the working directory.
        overrides: Extra layers merged after the entry file, in order (one per
            CLI ``--a.b value``).

    Raises:
        ConfigFileError: If a file cannot be read or parsed, an import is
            invalid, cyclic or lacks the requested section, or a repeated
            import would change the merge.
    """
    entry_document = _read_yaml_file(path)
    file_layers, import_nodes = _resolve_imports(
        path, entry_document, [os.path.realpath(path)], {}, repeat=None
    )
    type_changes: list[TypeChange] = []
    origins: dict[str, str] = {}
    deleted: list[ResetPath] = []
    missing: list[ResetPath] = []
    merged_data = _merge_layers(
        [values for values, _, _ in file_layers],
        overrides,
        sources=[source for _, _, source in file_layers],
        type_changes=type_changes,
        origins=origins,
        deleted=deleted,
        missing=missing,
    )
    repeat_descriptions = sorted(
        {repeat for _, repeat, _ in file_layers if repeat is not None}
    )
    if repeat_descriptions:
        # A repeat may merge again only while that changes nothing; otherwise
        # it would undo what a layer between the two occurrences set.
        merged_without_repeats = _merge_layers(
            [values for values, repeat, _ in file_layers if repeat is None],
            overrides,
        )
        changed_paths = _find_changed_paths(merged_data, merged_without_repeats)
        if changed_paths:
            shown_paths = ", ".join(changed_paths[:5]) + (
                ", ..." if len(changed_paths) > 5 else ""
            )
            raise ConfigFileError(
                f"{'; '.join(repeat_descriptions)}: merging it again would change "
                f"{shown_paths}, set by a file merged between the two imports. "
                "Import it once, or set those keys after the last import of it"
            )
    return RawDocument(
        data=merged_data,
        graph=ImportNode(path=_format_display_path(path), imports=tuple(import_nodes)),
        local_keys=frozenset(entry_document) - {"imports"},
        missing_cleared_paths=tuple(missing),
        deleted_paths=tuple(deleted),
        type_changes=tuple(type_changes),
        origins=SourceMap(origins),
    )


def _merge_layers(
    file_layers: Sequence[dict[str, Any]],
    override_layers: Sequence[dict[str, Any]],
    *,
    sources: Sequence[str] | None = None,
    type_changes: list[TypeChange] | None = None,
    origins: dict[str, str] | None = None,
    deleted: list[ResetPath] | None = None,
    missing: list[ResetPath] | None = None,
) -> dict[str, Any]:
    """Merge *file_layers*, then *override_layers*, left to right."""
    merged_data: dict[str, Any] = {}
    origins = {} if origins is None else origins
    layers = [*file_layers, *override_layers]
    labels = [
        *(sources or ["config import"] * len(file_layers)),
        *["CLI override"] * len(override_layers),
    ]
    for layer, source in zip(layers, labels, strict=True):
        if type_changes is not None:
            _collect_type_changes(merged_data, layer, origins, source, type_changes)
        merged_data = deep_merge(merged_data, layer, _deleted=deleted, _missing=missing)
        _record_origins(layer, origins, source)
    return merged_data


def _record_origins(
    layer: dict[str, Any] | list[Any],
    origins: dict[str, str],
    source: str,
    path: str = "",
) -> None:
    entries = layer.items() if isinstance(layer, dict) else enumerate(layer)
    for key, value in entries:
        child_path = join_path(path, key)
        for old in list(origins):
            if old == child_path or (
                (not isinstance(value, dict) or isinstance(value, ReplaceMapping))
                and old.startswith((child_path + ".", child_path + "["))
            ):
                origins.pop(old)
        origins[child_path] = source
        if isinstance(value, (dict, list)):
            _record_origins(value, origins, source, child_path)


def _collect_type_changes(
    base: dict[str, Any],
    layer: dict[str, Any],
    origins: dict[str, str],
    source: str,
    changes: list[TypeChange],
    path: str = "",
) -> None:
    if isinstance(layer, ReplaceMapping):
        changes[:] = [
            c
            for c in changes
            if not (c.path == path or not path or c.path.startswith(path + "."))
        ]
        return
    previous = next((c for c in changes if c.path == path), None)
    if previous is not None:
        # Later explicit values no longer originate in the earlier component.
        updated = TypeChange(
            path=previous.path,
            old_type=previous.old_type,
            new_type=previous.new_type,
            source=previous.source,
            inherited=tuple(
                (key, origin) for key, origin in previous.inherited if key not in layer
            ),
        )
        changes[:] = [updated if c.path == path else c for c in changes]
    old_type = previous.old_type if previous is not None else base.get("type")
    new_type = layer.get("type")
    if isinstance(old_type, str) and isinstance(new_type, str):
        changes[:] = [c for c in changes if c.path != path]
        if old_type != new_type:
            inherited = tuple(
                (key, origins.get(join_path(path, key), "config import"))
                for key in base
                if key != "type" and key not in layer
            )
            # Empty events retain the original type for a later switch back,
            # but generate no diagnostic.
            changes.append(
                TypeChange(
                    path=path,
                    old_type=old_type,
                    new_type=new_type,
                    source=source,
                    inherited=inherited,
                )
            )
    for key, value in layer.items():
        child_path = join_path(path, key)
        if value is None:
            changes[:] = [
                c
                for c in changes
                if not (c.path == child_path or c.path.startswith(child_path + "."))
            ]
        elif isinstance(value, ReplaceMapping):
            _collect_type_changes({}, value, origins, source, changes, child_path)
        elif isinstance(value, dict) and isinstance(base.get(key), dict):
            _collect_type_changes(
                base[key], value, origins, source, changes, child_path
            )


def _find_changed_paths(
    left_data: dict[str, Any], right_data: dict[str, Any], path: str = ""
) -> list[str]:
    """Find the ``join_path`` paths where two merged documents differ."""
    changed_paths = []
    for key in [*left_data, *(k for k in right_data if k not in left_data)]:
        left_value, right_value = left_data.get(key), right_data.get(key)
        if isinstance(left_value, dict) and isinstance(right_value, dict):
            changed_paths += _find_changed_paths(
                left_value, right_value, join_path(path, key)
            )
        elif key not in left_data or key not in right_data or left_value != right_value:
            changed_paths.append(join_path(path, key))
    return changed_paths


def _resolve_imports(
    config_path: str,
    file_document: dict[str, Any],
    active_import_paths: list[str],
    seen_import_paths: dict[str, str],
    *,
    repeat: str | None,
) -> tuple[list[tuple[dict[str, Any], str | None, str]], list[ImportNode]]:
    """Flatten a file's import tree into ordered merge layers.

    Returns the depth-first layers (imports before the file's own values),
    each paired with the description of the repeated import it came through
    (``None`` for a first occurrence), and the direct import nodes. Nothing is
    merged here: one fold over all layers lets a ``null`` inside an import act
    on earlier siblings of its ancestors.

    ``active_import_paths`` is the current import chain (for cycle detection);
    ``seen_import_paths`` maps every full-config import reached so far to its
    declaring file.
    """
    own_values = {k: v for k, v in file_document.items() if k != "imports"}
    if isinstance(file_document, ReplaceMapping):
        own_values = ReplaceMapping(own_values)
    import_specs = _normalize_imports(file_document.get("imports"), config_path)

    file_layers: list[tuple[dict[str, Any], str | None, str]] = []
    import_nodes: list[ImportNode] = []
    declaring_directory = os.path.dirname(os.path.abspath(config_path))

    for import_spec in import_specs:
        section_name = import_spec.section_name
        imported_path = os.path.normpath(
            os.path.join(declaring_directory, import_spec.path)
        )
        canonical_path = os.path.realpath(imported_path)
        if canonical_path in active_import_paths:
            import_chain = " -> ".join(
                _format_display_path(p) for p in active_import_paths + [imported_path]
            )
            raise ConfigFileError(f"import cycle detected: {import_chain}")
        # Section imports are exempt from repeat marking: their sub-trees are
        # projected onto one section, so a repeat inside them cannot overlap.
        import_repeat = repeat
        if section_name is None:
            if canonical_path in seen_import_paths:
                import_repeat = import_repeat or (
                    f"{_format_display_path(imported_path)} is imported by "
                    f"{seen_import_paths[canonical_path]} and again by "
                    f"{_format_display_path(config_path)}"
                )
            else:
                seen_import_paths[canonical_path] = _format_display_path(config_path)
        try:
            imported_document = _read_yaml_file(imported_path)
        except ConfigFileError as exc:
            raise ConfigFileError(
                f"{_format_display_path(config_path)}: import {import_spec.path!r} "
                f"(resolved to {imported_path}): {exc}"
            ) from exc
        imported_layers, nested_import_nodes = _resolve_imports(
            imported_path,
            imported_document,
            active_import_paths + [canonical_path],
            seen_import_paths if section_name is None else {},
            repeat=import_repeat,
        )

        if section_name:
            section_layers = [
                (
                    {
                        section_name: ReplaceMapping(layer[section_name])
                        if isinstance(layer, ReplaceMapping)
                        and isinstance(layer[section_name], dict)
                        else layer[section_name]
                    },
                    layer_repeat,
                    layer_source,
                )
                for layer, layer_repeat, layer_source in imported_layers
                if section_name in layer
            ]
            if not section_layers:
                top_level_keys = ", ".join(
                    sorted({k for layer, _, _ in imported_layers for k in layer})
                )
                raise ConfigFileError(
                    f"{_format_display_path(imported_path)} (imported under '{section_name}' by "
                    f"{_format_display_path(config_path)}) has no '{section_name}' section; top-level "
                    f"keys: {top_level_keys or 'none'}. A section import takes the named section out "
                    "of an experiment-shaped file"
                )
            imported_layers = section_layers

        file_layers.extend(imported_layers)
        import_nodes.append(
            ImportNode(
                path=_format_display_path(imported_path),
                label=section_name or "",
                imports=tuple(nested_import_nodes),
            )
        )

    file_layers.append((own_values, repeat, _format_display_path(config_path)))
    return file_layers, import_nodes


def _format_display_path(path: str) -> str:
    """Return the shorter of the absolute and the cwd-relative path."""
    absolute_path = os.path.abspath(path)
    try:
        relative_path = os.path.relpath(absolute_path)
    except ValueError:  # different drive on Windows
        return absolute_path
    return relative_path if len(relative_path) <= len(absolute_path) else absolute_path
