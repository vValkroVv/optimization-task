"""Persist the resolved config so a run can be reproduced.

Bundle contents and layout: docs/config.md, "Reproducibility".
"""

import dataclasses
import os
import platform
import secrets
import shlex
import sys
import subprocess
import time
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .cli import LoadedConfig

from .configurable import import_spec, to_plain
from .decode import decode
from .errors import ConfigError, ConfigFileError
from .loader import (
    _strip_nulls,
    ConfigYAMLDumper,
    ImportNode,
    parse_yaml_value,
    RawDocument,
)
from .registry import TRAINER_TYPES
from .sections import CONFIG_VERSION, ModelSection
from .validation import RuntimeInfo


def dump_yaml(data: Any) -> str:
    import yaml

    dumped = yaml.dump(
        data, Dumper=ConfigYAMLDumper, sort_keys=False, default_flow_style=False
    )
    assert isinstance(dumped, str)
    return dumped


def write_yaml(data: Any, path: str) -> None:
    """Write *data* to *path* atomically, creating parent directories.

    torchft replica groups share ``dump_folder``, so several processes may
    write the same file. ``open(..., "x")`` rather than ``tempfile.mkstemp``
    so the file gets the umask's permissions, not 0600.
    """
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    tmp = f"{path}.{secrets.token_hex(4)}.tmp"
    try:
        with open(tmp, "x", encoding="utf-8") as handle:
            handle.write(dump_yaml(data))
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def collect_metadata(*, command: list[str], runtime: RuntimeInfo) -> dict[str, Any]:
    """Provenance for a run: code version, invocation, environment."""
    return {
        "config_version": CONFIG_VERSION,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "hostname": platform.node(),
        "python": platform.python_version(),
        "command": shlex.join(command),
        "argv": list(command),
        "environment": {
            key: os.environ[key]
            for key in (
                "NGPU",
                "CUDA_VISIBLE_DEVICES",
                "WORLD_SIZE",
                "RANK",
                "LOCAL_RANK",
                "MASTER_ADDR",
                "MASTER_PORT",
                "TORCHELASTIC_RESTART_COUNT",
            )
            if key in os.environ
        },
        # imports.yaml records paths relative to this directory when shorter.
        "working_directory": os.getcwd(),
        "git": _get_git_info(),
        "packages": _get_package_versions(),
        "runtime": runtime.to_dict(),
    }


def _get_git_info() -> dict[str, str]:
    def run(*args: str) -> str:
        try:
            return subprocess.check_output(
                args, stderr=subprocess.DEVNULL, text=True, timeout=5
            ).strip()
        except (subprocess.SubprocessError, OSError):
            return ""

    # A status killed by the timeout mid index refresh (slow NFS) would leave
    # ``.git/index.lock`` behind; ``--no-optional-locks`` makes it read-only.
    porcelain_status = run("git", "--no-optional-locks", "status", "--porcelain")
    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": "true" if porcelain_status else "false",
    }


_TRACKED_PACKAGES = (
    "torch",
    "torchdata",
    "torchtitan",
    "datasets",
    "tokenizers",
    "transformers",
    "pyyaml",
)


def _get_package_versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    package_versions = {}
    for package_name in _TRACKED_PACKAGES:
        try:
            package_versions[package_name] = version(package_name)
        except PackageNotFoundError:
            continue
    return package_versions


LAUNCHES_DIR = "launches"


def _model_section(value: dict[str, Any] | ModelSection) -> ModelSection:
    return (
        value
        if isinstance(value, ModelSection)
        else decode(ModelSection, value, path="model")
    )


def get_typed_document(
    trainer_config: Any, raw_model_section: dict[str, Any] | ModelSection
) -> dict[str, Any]:
    """Build the full document of a decoded run, defaults included.

    The inverse of ``build_trainer_config`` up to ``model_spec``; the model is
    written in its compact form. Fields at ``None`` are left out: a ``null``
    reads as a reset to the default anyway.
    """
    trainer_type = next(
        (
            name
            for name, entry in TRAINER_TYPES.items()
            if import_spec(entry.spec) is type(trainer_config)
        ),
        None,
    )
    if trainer_type is None:
        raise ConfigFileError(
            f"Cannot serialize resolved configuration: trainer class "
            f"{type(trainer_config).__qualname__} is not registered"
        )
    model_section = _model_section(raw_model_section)
    try:
        model_data = to_plain(model_section)
        trainer_data = to_plain(trainer_config)
    except ValueError as error:
        raise ConfigFileError(
            f"Cannot serialize resolved configuration: {error}"
        ) from error
    typed_document = _strip_nulls(
        {
            "config_version": CONFIG_VERSION,
            "trainer": {"type": trainer_type},
            "model": model_data,
            **trainer_data,
        }
    )
    return typed_document


def save_loaded_run_bundle(loaded: "LoadedConfig", *, logger: Any) -> str | None:
    """Validate on every rank, then save launch provenance on rank zero.

    This runs before process-group initialization: failures stop the worker and
    its launcher must terminate sibling workers, rather than using collectives.
    """
    dest = os.path.join(loaded.trainer_config.dump_folder, "config")
    try:
        section = _model_section(loaded.source_document["model"])
        resolved = get_typed_document(
            loaded.trainer_config,
            section,
        )
        check_round_trip(
            loaded.trainer_config,
            loaded.source_document,
            resolved,
            dest,
            model_section=section,
        )
        if loaded.runtime.rank != 0:
            return None
        result = _write_run_bundle(
            loaded.source_document,
            resolved,
            dest=dest,
            command=list(sys.orig_argv),
            runtime=loaded.runtime,
            overrides=loaded.applied_overrides,
            imports=loaded.import_graph.to_dict(),
            preflight_report=loaded.preflight_report.to_dict(),
        )
    except (ConfigError, OSError) as error:
        logger.error("Cannot save run configuration: %s", error)
        raise SystemExit(1) from error
    logger.info("Saved resolved config bundle to %s", dest)
    return result


def save_run_bundle(
    trainer_config: Any,
    source_document: dict[str, Any],
    *,
    dest: str,
    command: list[str],
    runtime: RuntimeInfo,
    overrides: dict[str, Any] | None = None,
    imports: dict[str, Any] | None = None,
    preflight_report: dict[str, Any] | None = None,
) -> str:
    """Write the reproducibility bundle; return ``<dest>/resolved_config.yaml``.

    The round-trip check runs on the serialized text in memory: reading the
    file back from disk could see another replica group's copy.
    """
    section = _model_section(source_document["model"])
    resolved_document = get_typed_document(trainer_config, section)
    check_round_trip(
        trainer_config, source_document, resolved_document, dest, model_section=section
    )
    return _write_run_bundle(
        source_document,
        resolved_document,
        dest=dest,
        command=command,
        runtime=runtime,
        overrides=overrides,
        imports=imports,
        preflight_report=preflight_report,
    )


def _write_run_bundle(
    source_document,
    resolved_document,
    *,
    dest,
    command,
    runtime,
    overrides=None,
    imports=None,
    preflight_report=None,
) -> str:
    """Write an already validated bundle; all ranks validate before rank zero calls it."""

    bundle_files: dict[str, Any] = {
        "resolved_config.yaml": resolved_document,
        "source_config.yaml": source_document,
        "overrides.yaml": overrides or {},
        "imports.yaml": imports or {},
    }
    if preflight_report is not None:
        bundle_files["preflight_report.yaml"] = preflight_report
    bundle_files["metadata.yaml"] = collect_metadata(command=command, runtime=runtime)

    launch_dir = os.path.join(
        dest,
        LAUNCHES_DIR,
        f"{time.strftime('%Y%m%d-%H%M%S')}-{platform.node()}-{secrets.token_hex(3)}",
    )
    os.makedirs(launch_dir)
    for bundle_dir in (launch_dir, dest):
        for file_name, file_content in bundle_files.items():
            write_yaml(file_content, os.path.join(bundle_dir, file_name))
    return os.path.join(dest, "resolved_config.yaml")


def check_round_trip(
    trainer_config: Any,
    source_document: dict[str, Any],
    resolved_document: dict[str, Any],
    dest: str,
    *,
    model_section: ModelSection | None = None,
) -> None:
    """Check that *resolved_document* as YAML text decodes to *trainer_config*
    and to the same model Config as *source_document*."""
    from .build import prepare_document

    reloaded_data = parse_yaml_value(
        dump_yaml(resolved_document), source="resolved config"
    )
    reloaded_document = RawDocument(
        data=reloaded_data, graph=ImportNode(path="resolved config")
    )
    source_model = model_section or _model_section(source_document["model"])
    prepared = prepare_document(reloaded_document)
    # ``replace`` resets the init=False ``model_spec``, which a reload leaves unset.
    if (
        prepared.trainer_config != dataclasses.replace(trainer_config)
        or source_model != prepared.model_section
    ):
        raise ConfigFileError(
            f"{dest}: the resolved config does not decode back to the config "
            "the run uses; it would not be a reproducible artifact"
        )
