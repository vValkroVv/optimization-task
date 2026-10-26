"""Decode a merged YAML document into ``TrainerConfig``.

Only ``build_model_spec`` needs the training runtime (torch).
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .configurable import import_spec, type_name_of
from .registry import TRAINER_TYPES
from .decode import (
    checking_deleted_entries,
    checking_missing_cleared_paths,
    decode,
    decoding_origins,
)
from .errors import ConfigSchemaError, suggest
from .loader import load_raw_document, RawDocument
from .sections import CONFIG_VERSION, ModelSection, TrainerSection
from .schema.protocols.model import BaseModelConfig
from .validation import type_change_error_context

if TYPE_CHECKING:
    from torchtitan.config.schema.trainer import TrainerConfig
    from torchtitan.protocols.model_spec import ModelSpec


@dataclass
class PreparedConfig:
    """Decoded compact data and an independent tree owned by this preparation.

    The tree remains pristine during pure validation; runtime spec creation
    consumes it by applying converters in place.
    """

    trainer_config: "TrainerConfig"
    model_section: ModelSection
    model_config: BaseModelConfig


def build_trainer_config(
    raw_document: RawDocument, *, with_model_spec: bool = True
) -> "TrainerConfig":
    """Decode the merged *raw_document* into a ready ``TrainerConfig``.

    The whole document, not only its ``data``: its ``missing_cleared_paths``
    are checked so a misspelled reset fails like a misspelled key.
    ``with_model_spec=False`` stays torch-free and leaves ``model_spec`` unset.
    """
    prepared = prepare_document(raw_document)
    if with_model_spec:
        prepared.trainer_config.model_spec = _build_model_spec(
            prepared.model_section, prepared.model_config
        )
    return prepared.trainer_config


def prepare_document(raw_document: RawDocument) -> PreparedConfig:
    """Decode each section and materialize the model once, without runtime imports."""
    # The root has no class of its own: the section each reset starts in gets
    # the unknown-section check below.
    missing_cleared_paths = raw_document.missing_cleared_paths
    top_level_resets = list(dict.fromkeys(path[0] for path in missing_cleared_paths))
    sections_data = dict(raw_document.data)
    config_version = sections_data.pop("config_version", None)
    if type(config_version) is not int or config_version != CONFIG_VERSION:
        raise ConfigSchemaError(
            f"config_version: must be {CONFIG_VERSION} (found {config_version!r})"
        )

    with (
        type_change_error_context(raw_document),
        decoding_origins(raw_document.origins, raw_document.source_paths),
        checking_missing_cleared_paths(
            path for path in missing_cleared_paths if len(path) > 1
        ),
        checking_deleted_entries(raw_document.deleted_paths),
    ):
        trainer_section = decode(
            TrainerSection, sections_data.pop("trainer", {}), path="trainer"
        )
        trainer_type = TRAINER_TYPES.get(trainer_section.type)
        if trainer_type is None:
            raise ConfigSchemaError(
                f"trainer.type: unknown trainer type {trainer_section.type!r}."
                f"{suggest(trainer_section.type, TRAINER_TYPES)} "
                f"Available: {', '.join(sorted(TRAINER_TYPES))}"
            )

        raw_model_section = sections_data.pop("model", None)
        if raw_model_section is None:
            raise ConfigSchemaError(
                "model: section is required (model.config is the model's "
                "Config, its class picked by `type`)"
            )
        model_section = decode(ModelSection, raw_model_section, path="model")
        required_trainer = type(model_section.config).trainer_type
        if trainer_type.model_trainer_type != required_trainer:
            raise ConfigSchemaError(
                f"trainer.type: {trainer_section.type!r} accepts the "
                f"{trainer_type.model_trainer_type!r} model runtime contract; "
                f"model.config.type {type_name_of(type(model_section.config))!r} "
                f"requires {required_trainer!r}"
            )

        for section_name in [*sections_data, *top_level_resets]:
            # Catches model_spec too: derived from `model`, never set in YAML.
            if (
                section_name not in trainer_type.sections
                and section_name not in _META_SECTIONS
            ):
                reset_note = (
                    "" if section_name in sections_data else " (reset with null)"
                )
                raise ConfigSchemaError(
                    f"{section_name}: unknown top-level section{reset_note} for "
                    f"trainer {trainer_section.type!r}."
                    f"{suggest(section_name, trainer_type.sections)} "
                    f"Valid sections: {', '.join(sorted(trainer_type.sections))}"
                )

        trainer_config = decode(import_spec(trainer_type.spec), sections_data)

        tree = materialize_model_config(
            model_section.config, trainer_config=trainer_config
        )
    return PreparedConfig(trainer_config, model_section, tree)


_META_SECTIONS = ("config_version", "trainer", "model")
"""Root keys ``build_trainer_config`` consumes itself."""


def materialize_model_config(model_config, *, trainer_config=None):
    """Run ``model_config.materialize()``, its ``ValueError`` raised as a
    config error at ``model.config``."""
    try:
        from .schema.models.common.cross_section import require_model_module_fields

        if not getattr(model_config, "materialized", False):
            try:
                require_model_module_fields(model_config)
            except ValueError as exc:
                # These issues already carry complete model.config paths.
                raise ConfigSchemaError(str(exc)) from exc
        tree = model_config.materialize()
        converters = getattr(model_config, "converters", ())
        for index, converter in enumerate(converters):
            try:
                converter.validate_pipeline(tree, converters[:index], trainer_config)
            except ValueError as exc:
                raise ConfigSchemaError(
                    f"model.config.converters[{index}]: {exc}"
                ) from exc
        return tree
    except ConfigSchemaError:
        raise
    except ValueError as exc:
        raise ConfigSchemaError(f"model.config: {exc}") from exc


def build_model_spec(
    model_section: ModelSection, *, apply_converters: bool = True
) -> "ModelSpec":
    """Build a ``ModelSpec`` holding the materialized model Config.

    ``apply_converters=False`` is for checkpoint conversion, which needs the
    plain model (no LoRA wrappers)."""
    return _build_model_spec(
        model_section,
        materialize_model_config(model_section.config),
        apply_converters=apply_converters,
    )


def _build_model_spec(
    model_section: ModelSection,
    model_tree: BaseModelConfig,
    *,
    apply_converters: bool = True,
) -> "ModelSpec":
    """Consume an independently prepared tree; runtime constructors do not reconvert it."""
    from torchtitan.protocols.model_spec import ModelSpec
    from torchtitan.protocols.model import apply_model_converters

    if not model_tree.materialized:
        raise ValueError(
            f"{type(model_section.config).__name__}: build needs the materialized tree"
        )

    model_config = model_section.config
    if not model_config._runtime_spec:
        raise ConfigSchemaError("model.config: the model type has no runtime bundle")
    from torchtitan.models.common.moe.load_balance import (
        register_moe_load_balancing_hook,
    )

    model_runtime = import_spec(model_config._runtime_spec)
    family_name = type(model_config).family
    if apply_converters:
        apply_model_converters(model_tree, getattr(model_config, "converters", ()))
        try:
            model_tree.validate_materialized()
        except ValueError as exc:
            raise ConfigSchemaError(f"model.config.converters: {exc}") from exc
    return (model_runtime.spec_cls or ModelSpec)(
        name=family_name,
        # A variant preset spells its variant in the label (``gift_4/700b``).
        flavor=model_section.label,
        model=model_tree,
        parallelize_fn=model_runtime.parallelize_fn,
        pipelining_fn=model_runtime.pipelining_fn,
        post_optimizer_build_fn=register_moe_load_balancing_hook,
        state_dict_adapter=model_runtime.state_dict_adapter,
        **model_runtime.spec_kwargs,
    )


def load_model_spec(config_path: str, *, apply_converters: bool = True) -> "ModelSpec":
    """Load the ``ModelSpec`` described by the ``model:`` section of *config_path*.

    Accepts an experiment YAML, a run's ``resolved_config.yaml`` (the faithful
    choice when rebuilding a checkpoint's model) or a model preset.
    """
    raw_document = load_raw_document(config_path)
    document_data = raw_document.data
    if "model" not in document_data:
        raise ConfigSchemaError(f"{config_path}: no `model:` section")
    with (
        type_change_error_context(raw_document),
        decoding_origins(raw_document.origins, raw_document.source_paths),
        checking_missing_cleared_paths(raw_document.missing_cleared_paths),
        checking_deleted_entries(raw_document.deleted_paths),
    ):
        return build_model_spec(
            decode(ModelSection, document_data["model"], path="model"),
            apply_converters=apply_converters,
        )
