"""Pre-submit validation, and the ``config-check`` entrypoint (torch-free).

See docs/config.md, "Validation stages".
"""

import sys
from dataclasses import dataclass, field
from typing import Any

from .cli import load_document
from .errors import ConfigError, ConfigSchemaError
from .loader import ImportNode
from .overrides import parse_argv
from .sections import CONFIG_VERSION
from .validation import (
    Issue,
    RuntimeInfo,
    configured_paths,
    merge_type_change_issues,
    validate_typed_config,
    ValidationContext,
    ValidationReport,
)


@dataclass(kw_only=True)
class PreSubmitResult:
    """Everything a submitter needs to decide whether to queue the job."""

    data: dict[str, Any]
    config_path: str
    import_graph: ImportNode
    report: ValidationReport
    overrides: dict[str, Any] = field(default_factory=dict)
    decoded: bool = False
    """Whether the document decoded into the typed tree, model tree included."""

    @property
    def ok(self) -> bool:
        return self.report.ok

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_path": self.config_path,
            "config_version": CONFIG_VERSION,
            "decoded": self.decoded,
            "overrides": self.overrides,
            "imports": self.import_graph.to_dict(),
            **self.report.to_dict(),
        }


def run_pre_submit(
    config_path: str,
    raw_overrides: list[tuple[str, str]] | None = None,
    *,
    world_size: int | None = None,
) -> PreSubmitResult:
    """Load and validate a config: the typed decode and ``validate(ctx)`` pass."""
    document, overrides = load_document(config_path, raw_overrides)
    result, _ = _validate_document(
        config_path, document, overrides, world_size=world_size
    )
    return result


def _validate_document(config_path, document, overrides, *, world_size=None):
    """Validate one prepared document; keep its tree available to print modes."""
    data = document.data
    if world_size is not None and (type(world_size) is not int or world_size <= 0):
        raise ConfigSchemaError("world_size must be a positive integer")
    report = ValidationReport(stage="pre_submit")
    report.issues.extend(merge_type_change_issues(document))

    from .build import prepare_document
    from .save import check_round_trip, get_typed_document

    def report_error(exc: Exception) -> None:
        # Keep a schema error's YAML path as a structured Issue path.
        message = str(exc)
        path, separator, detail = message.partition(": ")
        if (
            isinstance(exc, ConfigSchemaError)
            and separator
            and (path.split(".", 1)[0].split("[", 1)[0] in data or path == "model")
        ):
            source = getattr(exc, "source", None)
            if source:
                detail = detail.removesuffix(f" (source: {source})")
            report.issues.append(Issue(path=path, message=detail, source=source))
        else:
            report.error("", message)

    decoded = False
    prepared = None
    # Nothing here imports a runtime module, so a ModuleNotFoundError is a
    # schema bug and propagates. ValueError: an invariant outside decode and
    # materialize().
    try:
        prepared = prepare_document(document)
    except (ConfigError, ValueError) as exc:
        report_error(exc)
    else:
        decoded = True
        config, model_config = prepared.trainer_config, prepared.model_config
        section = prepared.model_section
        try:
            resolved = get_typed_document(config, section)
            check_round_trip(config, data, resolved, config_path, model_section=section)
        except (ConfigError, ValueError) as exc:
            report_error(exc)
        ctx = ValidationContext(
            stage="pre_submit",
            runtime=RuntimeInfo(world_size=world_size)
            if world_size is not None
            else None,
            model_name=type(model_config).family,
            configured_paths=configured_paths(data),
        )
        report.issues.extend(validate_typed_config(config, model_config, ctx).issues)

    return PreSubmitResult(
        data=data,
        config_path=config_path,
        import_graph=document.graph,
        report=report,
        overrides=overrides,
        decoded=decoded,
    ), prepared


_USAGE = """\
usage: config-check <config.yaml> [--section.key value ...] [--print] [--report PATH]

Validate an experiment config before submitting it to the cluster.
Every section decodes and validates, the model architecture included, without
torch; the result does not depend on whether torch is installed.

  --print        print the merged YAML document (after imports and overrides)
  --print-merged  print the merged YAML document (--print is an alias)
  --print-resolved  print typed defaults without building runtime objects
  --print-layers  print selected components on each actual model layer
  --report PATH  write the validation report as YAML
  --world-size N check parallel degrees against the planned number of processes
"""


def main(argv: list[str] | None = None) -> int:
    """``config-check`` entrypoint. Returns a process exit code."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(_USAGE, end="")
        return 0

    print_merged = False
    print_resolved = False
    print_layers = False
    report_path: str | None = None
    world_size = None
    filtered: list[str] = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg in ("--print", "--print-merged"):
            print_merged = True
        elif arg == "--print-resolved":
            print_resolved = True
        elif arg == "--print-layers":
            print_layers = True
        elif arg == "--report":
            i += 1
            if i >= len(argv):
                print("--report requires a path", file=sys.stderr)
                return 2
            report_path = argv[i]
        elif arg.startswith("--report="):
            report_path = arg.split("=", 1)[1]
        elif arg == "--world-size" or arg.startswith("--world-size="):
            if "=" in arg:
                value = arg.split("=", 1)[1]
            else:
                i += 1
                value = argv[i] if i < len(argv) else ""
            try:
                world_size = int(value)
                if world_size <= 0:
                    raise ValueError
            except ValueError:
                print("--world-size requires a positive integer", file=sys.stderr)
                return 2
        else:
            filtered.append(arg)
        i += 1

    try:
        config_path, raw_overrides = parse_argv(filtered)
        document, overrides = load_document(config_path, raw_overrides)
        result, prepared = _validate_document(
            config_path, document, overrides, world_size=world_size
        )
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 1

    if print_merged or (print_resolved and result.ok):
        from .save import dump_yaml, get_typed_document

        data = result.data
        if print_resolved and result.ok:
            assert prepared is not None
            data = get_typed_document(prepared.trainer_config, prepared.model_section)
        print(dump_yaml(data))

    if print_layers and result.ok:
        assert prepared is not None
        from .save import dump_yaml

        summary = prepared.model_config.layer_summary()
        if summary is None:
            print("layer summary is not supported by this model type", file=sys.stderr)
            return 2
        print(dump_yaml({"layers": summary}))

    if report_path:
        from .save import write_yaml

        write_yaml(result.to_dict(), report_path)

    for issue in result.report.warnings:
        print(f"warning: {issue}", file=sys.stderr)
    for issue in result.report.errors:
        print(f"error:   {issue}", file=sys.stderr)

    if not result.ok:
        print(
            f"\n{len(result.report.errors)} error(s) in {config_path}",
            file=sys.stderr,
        )
        return 1

    model = result.data.get("model", {})
    print(
        f"{config_path}: OK "
        f"({len(result.report.warnings)} warning(s), "
        f"model={model.get('config', {}).get('type')}"
        f"/{model.get('label', 'custom')})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
