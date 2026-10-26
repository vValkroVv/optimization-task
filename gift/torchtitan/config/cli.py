"""The parse -> load -> decode -> preflight prologue every YAML entrypoint shares.

Importing this module is torch-free; ``load_trainer_config`` needs torch.
"""

import os
import copy
from collections.abc import Mapping, Sequence
import sys
from dataclasses import dataclass, replace
from typing import Any

from .errors import ConfigError, ConfigFileError, ConfigSchemaError, join_path
from .loader import ImportNode, load_raw_document, RawDocument
from .overrides import build_override_layers, parse_argv
from .preflight import preflight
from .sections import CONFIG_VERSION
from .validation import (
    configured_paths,
    Issue,
    merge_type_change_issues,
    RuntimeInfo,
    validate_typed_config,
    ValidationContext,
    ValidationReport,
)


def load_document(
    config_path: str, raw_overrides: list[tuple[str, str]] | None = None
) -> tuple[RawDocument, dict[str, Any]]:
    """Load YAML + imports + CLI overrides; return the document and the
    applied overrides.

    Raises:
        ConfigFileError: If the entry file does not declare ``config_version``
            itself, or a file cannot be loaded.
        ConfigSchemaError: For a malformed override.
    """
    override_layers, applied_overrides = build_override_layers(raw_overrides or [])
    return _load_document_layers(config_path, override_layers, applied_overrides)


def _load_document_layers(config_path, override_layers, applied_overrides):
    raw_document = load_raw_document(config_path, overrides=override_layers)

    if "config_version" not in raw_document.local_keys:
        raise ConfigFileError(
            f"{config_path}: every experiment config must declare "
            f"`config_version: {CONFIG_VERSION}` in the file itself, "
            "not via an import"
        )

    return raw_document, applied_overrides


@dataclass(kw_only=True)
class LoadedConfig:
    """A decoded config together with everything the run bundle records."""

    trainer_config: Any
    config_path: str
    source_document: dict[str, Any]
    """Merged raw document after imports and overrides."""

    import_graph: ImportNode
    applied_overrides: dict[str, Any]
    """``path -> value`` in argv order (a repeated path at its last position)."""

    runtime: RuntimeInfo
    preflight_report: ValidationReport


def load_config(
    config_path: str | os.PathLike[str],
    *,
    overrides: Sequence[str | Mapping[str, Any]] | None = None,
) -> LoadedConfig:
    """Load and validate a run, raising ``ConfigError`` instead of exiting.

    Override strings are CLI tokens; mappings are nested YAML layers. Mixed
    layers apply in order. No logger is required and caller mappings stay intact.
    """
    path = os.fspath(config_path)
    if not isinstance(path, str):
        raise ConfigFileError("config file path must be a string or PathLike[str]")
    layers, applied = _programmatic_override_layers(
        path, () if overrides is None else overrides
    )
    document, applied = _load_document_layers(path, layers, applied)
    return _load_config_document(document, applied, path)


def _programmatic_override_layers(config_path, overrides):
    if isinstance(overrides, (str, bytes, Mapping)):
        raise ConfigSchemaError(
            "overrides must be a sequence of CLI tokens or mappings"
        )
    layers: list[dict[str, Any]] = []
    applied: dict[str, Any] = {}
    tokens: list[str] = []

    def flush_tokens():
        if tokens:
            _, pairs = parse_argv([config_path, *tokens])
            additions, metadata = build_override_layers(pairs)
            layers.extend(additions)
            for key, value in metadata.items():
                applied.pop(key, None)
                applied[key] = value
            tokens.clear()

    def record_mapping(layer, prefix=""):
        from .loader import ReplaceMapping

        for key, value in layer.items():
            path = join_path(prefix, key)
            if (
                isinstance(value, Mapping)
                and value
                and not isinstance(value, ReplaceMapping)
            ):
                record_mapping(value, path)
            else:
                applied.pop(path, None)
                applied[path] = value

    def clone_value(value, active=None):
        from .loader import ReplaceMapping

        active = set() if active is None else active
        if isinstance(value, (Mapping, list, tuple)):
            if id(value) in active:
                raise ConfigSchemaError(
                    "overrides contain a cyclic mapping or sequence"
                )
            active.add(id(value))
            try:
                if isinstance(value, Mapping):
                    copied = {
                        key: clone_value(item, active) for key, item in value.items()
                    }
                    return (
                        ReplaceMapping(copied)
                        if isinstance(value, ReplaceMapping)
                        else copied
                    )
                items = [clone_value(item, active) for item in value]
                return tuple(items) if isinstance(value, tuple) else items
            finally:
                active.remove(id(value))
        return copy.deepcopy(value)

    for override in overrides:
        if isinstance(override, str):
            tokens.append(override)
        elif isinstance(override, Mapping):
            flush_tokens()
            layer = clone_value(override)
            assert isinstance(layer, dict), "mapping clone must remain a mapping"
            layers.append(layer)
            record_mapping(layer)
        else:
            raise ConfigSchemaError("overrides entries must be CLI strings or mappings")
    flush_tokens()
    return layers, applied


def _load_config_document(
    raw_document,
    applied_overrides,
    config_path,
    *,
    warning_sink: list[Issue] | None = None,
) -> LoadedConfig:
    from .build import _build_model_spec, prepare_document

    source_document = raw_document.data
    merge_issues = merge_type_change_issues(raw_document)
    if warning_sink is not None:
        warning_sink.extend(merge_issues)
    prepared = prepare_document(raw_document)
    trainer_config = prepared.trainer_config
    model_spec = _build_model_spec(prepared.model_section, prepared.model_config)
    trainer_config.model_spec = model_spec

    # RuntimeInfo's fallback is useful for standalone inspection, but a run must
    # not silently replace an explicitly malformed distributed environment.
    for name, minimum in (
        ("WORLD_SIZE", 1),
        ("LOCAL_WORLD_SIZE", 1),
        ("RANK", 0),
        ("LOCAL_RANK", 0),
    ):
        value = os.environ.get(name)
        if value is not None:
            try:
                valid = int(value) >= minimum
            except ValueError:
                valid = False
            if not valid:
                raise ConfigSchemaError(f"{name}: must be an integer >= {minimum}")
    runtime_info = RuntimeInfo.from_env()
    comm = getattr(trainer_config, "comm", None)
    if comm is not None and comm.mode in ("fake_backend", "local_tensor"):
        value = os.environ.get("NGPU")
        try:
            world_size = int(value) if value is not None else 0
        except ValueError:
            world_size = 0
        if world_size < 1:
            raise ConfigSchemaError(
                "NGPU: must be a positive integer when comm.mode simulates ranks"
            )
        runtime_info = replace(runtime_info, world_size=world_size)

    report = preflight(trainer_config, model_spec.model, runtime_info)
    report.issues.extend(merge_issues)
    report.issues.extend(
        validate_typed_config(
            trainer_config,
            model_spec.model,
            ValidationContext(
                stage="in_container",
                runtime=runtime_info,
                model_name=model_spec.name,
                configured_paths=configured_paths(source_document),
            ),
        ).issues
    )
    if warning_sink is not None:
        for issue in report.warnings:
            if issue not in warning_sink:
                warning_sink.append(issue)
    report.raise_if_failed()
    return LoadedConfig(
        trainer_config=trainer_config,
        config_path=config_path,
        source_document=source_document,
        import_graph=raw_document.graph,
        applied_overrides=applied_overrides,
        runtime=runtime_info,
        preflight_report=report,
    )


def load_trainer_config(argv: list[str], *, logger: Any) -> LoadedConfig:
    """CLI wrapper: log diagnostics and convert config failures to exit code 1."""
    warnings: list[Issue] = []
    try:
        config_path, raw_overrides = parse_argv(argv)
        document, applied = load_document(config_path, raw_overrides)
        loaded = _load_config_document(
            document, applied, config_path, warning_sink=warnings
        )
    except ConfigError as exc:
        for issue in warnings:
            logger.warning(f"config: {issue}")
        logger.error(f"config error: {exc}")
        raise SystemExit(1) from None
    for issue in warnings:
        logger.warning(f"config: {issue}")
    return loaded


def get_argv(argv: list[str] | None) -> list[str]:
    """Return *argv*, or ``sys.argv[1:]`` when the caller passed none."""
    return list(sys.argv[1:] if argv is None else argv)
