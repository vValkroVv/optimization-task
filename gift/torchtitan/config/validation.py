"""One validation engine, two stages; see docs/config.md, "Validation stages".

Issues are accumulated, so a user sees everything that is wrong in one pass.
"""

import dataclasses
from contextlib import contextmanager
from dataclasses import dataclass, field, fields
from typing import Any, Literal

from .configurable import Configurable
from .errors import ConfigError, ConfigSchemaError, join_path

Stage = Literal["pre_submit", "in_container"]
Severity = Literal["error", "warning"]


@dataclass(frozen=True, kw_only=True)
class RuntimeInfo:
    """Facts about the process the trainer is about to run in."""

    world_size: int = 1
    local_world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    hostname: str = ""
    device_count: int | None = None
    """Visible accelerators, or ``None`` when it could not be determined."""

    @classmethod
    def from_env(cls) -> "RuntimeInfo":
        import os
        import socket

        def _int(name: str, default: int) -> int:
            try:
                return int(os.environ.get(name, default))
            except ValueError:
                return default

        device_count = None
        try:
            import torch

            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
        except Exception:  # noqa: BLE001 - torch may be absent or broken
            device_count = None

        return cls(
            world_size=_int("WORLD_SIZE", 1),
            local_world_size=_int("LOCAL_WORLD_SIZE", 1),
            rank=_int("RANK", 0),
            local_rank=_int("LOCAL_RANK", 0),
            hostname=socket.gethostname(),
            device_count=device_count,
        )

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass(frozen=True, kw_only=True)
class ValidationContext:
    stage: Stage
    runtime: RuntimeInfo | None = None
    model_name: str | None = None
    """The model Config's ``family`` (``ModelSpec.name``); ``None`` without a
    model section."""
    configured_paths: frozenset[str] = frozenset()
    """Paths present in the resolved document, including values equal to defaults."""


@dataclass(frozen=True, kw_only=True)
class Issue:
    path: str
    message: str
    severity: Severity = "error"
    source: str | None = None
    """Winning YAML file or override label, when the field's origin is known."""
    runtime_path: str | None = None
    """Materialized path when ``path`` names the original compact YAML field."""

    def __str__(self) -> str:
        message = f"{self.path}: {self.message}" if self.path else self.message
        if self.source:
            message += f" (source: {self.source})"
        return message

    def to_dict(self) -> dict[str, str]:
        result = {"path": self.path, "message": self.message, "severity": self.severity}
        if self.source is not None:
            result["source"] = self.source
        if self.runtime_path is not None:
            result["runtime_path"] = self.runtime_path
        return result


def issue_for_field(
    config: "Configurable.Config",
    *,
    key: str,
    path: str,
    message: str,
    severity: Severity = "error",
) -> Issue:
    """Report a derived field through its recorded configuration dependency."""
    origins = config._source_origins
    source_path = config._source_fields.get(key)
    if not origins or not source_path:
        return Issue(path=path, message=message, severity=severity)
    return Issue(
        path=source_path,
        message=message,
        severity=severity,
        source=origins.get(source_path),
        runtime_path=path if path != source_path else None,
    )


@dataclass(kw_only=True)
class ValidationReport:
    stage: Stage
    issues: list[Issue] = field(default_factory=list)

    def error(self, path: str, message: str) -> None:
        self.issues.append(Issue(path=path, message=message, severity="error"))

    def warn(self, path: str, message: str) -> None:
        self.issues.append(Issue(path=path, message=message, severity="warning"))

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_if_failed(self) -> None:
        if self.ok:
            return
        rendered = "\n".join(f"  - {issue}" for issue in self.errors)
        raise ConfigError(
            f"{len(self.errors)} config validation error(s) "
            f"[stage={self.stage}]:\n{rendered}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "ok": self.ok,
            "issues": [issue.to_dict() for issue in self.issues],
        }


def configured_paths(data: dict[str, Any]) -> frozenset[str]:
    """Paths explicitly present after imports and overrides are resolved."""
    paths: set[str] = set()

    def _walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                child_path = join_path(path, str(key))
                paths.add(child_path)
                _walk(value, child_path)
        elif isinstance(node, list):
            for index, value in enumerate(node):
                _walk(value, join_path(path, index))

    _walk(data, "")
    return frozenset(paths)


def validate_config_tree(
    root: Any, context: ValidationContext, *, path: str = "", skip: tuple[str, ...] = ()
) -> ValidationReport:
    """Collect ``validate(context)`` issues from every Config node under *root*.

    Walks a Config's ``tree_fields()``, other dataclass fields, lists and dict
    values; node-relative issue paths are prefixed with the node's path.
    *skip* names fields of *root* not walked.
    """
    report = ValidationReport(stage=context.stage)

    def _walk(node: Any, path: str) -> None:
        if isinstance(node, Configurable.Config):
            for issue in node.validate(context):
                report.issues.append(
                    dataclasses.replace(issue, path=join_path(path, issue.path))
                    if issue.path
                    else dataclasses.replace(issue, path=path)
                )
        if dataclasses.is_dataclass(node) and not isinstance(node, type):
            walked_fields = (
                node.tree_fields()
                if isinstance(node, Configurable.Config)
                else fields(node)
            )
            for f in walked_fields:
                if node is root and f.name in skip:
                    continue
                _walk(getattr(node, f.name), join_path(path, f.name))
        elif isinstance(node, (list, tuple)):
            for i, item in enumerate(node):
                _walk(item, join_path(path, i))
        elif isinstance(node, dict):
            for key, item in node.items():
                _walk(item, join_path(path, str(key)))

    _walk(root, path)
    return report


def validate_typed_config(
    config: Any, model_config: Any, context: ValidationContext
) -> ValidationReport:
    """Run the typed pass both stages share: every Config node of *config* and
    of the materialized *model_config* (under ``model``), then the trainer's
    ``validate_model``."""
    if not getattr(model_config, "materialized", True):
        raise ValueError(
            "validate_typed_config needs the materialized model config; the "
            "compact form has templates where the checks expect layers"
        )
    report = validate_config_tree(config, context, skip=("model_spec",))
    report.issues.extend(
        validate_config_tree(model_config, context, path="model").issues
    )
    report.issues.extend(config.validate_model(model_config))
    checkpoint = getattr(config, "checkpoint", None)
    if checkpoint is not None and checkpoint.hf_save_enabled:
        # The first HF save's schedule step (``trainer.step`` is 1-based): the
        # first-step save, else the first interval save or the last save.
        steps = config.training.steps
        if not checkpoint.save_in_hf:
            step = steps - 1
        elif checkpoint.enable_first_step_checkpoint:
            step = 0
        else:
            step = min(checkpoint.interval, steps) - 1
        report.issues.extend(model_config.validate_hf_export(step=step))
    unique = {}
    for issue in report.issues:
        key = (issue.path, issue.message, issue.severity, issue.source)
        unique.setdefault(key, issue)
    report.issues = list(unique.values())
    return report


def merge_type_change_issues(document) -> list[Issue]:
    """Report retained YAML values on sibling type switches, in schema context."""
    from .configurable import family_of, import_spec
    from .registry import TRAINER_TYPES
    from .schema_paths import schema_at

    issues: list[Issue] = []
    for change in document.type_changes:
        if not change.inherited:
            continue
        old_cls: type | None = None
        new_cls: type | None = None
        if change.path == "trainer":
            old_entry, new_entry = (
                TRAINER_TYPES.get(change.old_type),
                TRAINER_TYPES.get(change.new_type),
            )
            if old_entry is not None and new_entry is not None:
                old_cls, new_cls = (
                    import_spec(old_entry.spec),
                    import_spec(new_entry.spec),
                )
        else:
            schema = schema_at(document.data, change.path)
            has_family = False
            if isinstance(schema, type):
                # A fixed variant (FT optimizer, for example) may define its own
                # family while its parent still provides the original type.
                for base in schema.__mro__:
                    family = family_of(base)
                    if family is not None:
                        has_family = True
                        old_cls = old_cls or family.types.get(change.old_type)
                        new_cls = new_cls or family.types.get(change.new_type)
            if not has_family:
                continue
        if old_cls is not None and new_cls is not None and issubclass(new_cls, old_cls):
            continue
        entries = ", ".join(f"{key} ({source})" for key, source in change.inherited)
        issues.append(
            Issue(
                path=change.path or "config",
                severity="warning",
                message=f"type changes from {change.old_type!r} to {change.new_type!r} in {change.source}; "
                f"retaining inherited keys: {entries}. Use !replace {{type: {change.new_type}, ...}} "
                "to replace this mapping, or reset it with null before a CLI replacement.",
            )
        )
    return issues


@contextmanager
def type_change_error_context(document):
    """Explain errors in inherited fields retained across a component type switch."""
    try:
        yield
    except ConfigSchemaError as error:
        message = str(error)
        path, separator, _ = message.partition(": ")
        if not separator:
            raise
        changes = {change.path: change for change in document.type_changes}
        for issue in merge_type_change_issues(document):
            if not path.startswith(issue.path + "."):
                continue
            key = path.removeprefix(issue.path + ".").split(".", 1)[0].split("[", 1)[0]
            change = changes[issue.path]
            inherited_sources = dict(change.inherited)
            if key not in inherited_sources:
                continue
            source = (
                error.source or document.origins.get(path) or inherited_sources[key]
            )
            if source:
                message = message.removesuffix(f" (source: {source})")
            suffix = f" (source: {source})" if source else ""
            raise ConfigSchemaError(
                f"{message}; {issue.message}{suffix}", source=source
            ) from error
        raise
