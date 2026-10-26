"""Load experiment configs and model presets the way the entrypoint does."""

from pathlib import Path
from typing import Any, TYPE_CHECKING

from torchtitan.config import deep_merge, load_document, load_raw_document, parse_argv
from torchtitan.config.build import build_model_spec, build_trainer_config
from torchtitan.config.decode import decode
from torchtitan.config.sections import ModelSection

if TYPE_CHECKING:
    from torchtitan.protocols.model_spec import ModelSpec
    from torchtitan.trainer import Trainer

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = REPO_ROOT / "configs" / "experiments"
PRESET_ROOT = REPO_ROOT / "configs" / "presets" / "models"


def load_experiment_config(experiment: str, *overrides: str) -> "Trainer.Config":
    """``configs/experiments/<experiment>`` plus CLI-style dot-path overrides.

    *overrides* are the argv tokens a user would type after the YAML path
    (``"--training.steps", "10"``; a bare flag means ``true``).
    """
    config_path, raw_overrides = parse_argv(
        [str(EXPERIMENT_ROOT / experiment), *overrides]
    )
    document, _ = load_document(config_path, raw_overrides)
    return build_trainer_config(document)


def decode_document(
    config_path: str | Path, raw_overrides: list[tuple[str, str]] | None = None
) -> dict[str, Any]:
    """``load_document`` plus the torch-free decode, as pre-submit runs them
    (resets of unknown keys included); returns the merged data."""
    document, _ = load_document(str(config_path), raw_overrides)
    build_trainer_config(document, with_model_spec=False)
    return document.data


def load_model_spec(
    preset: str,
    model_overrides: dict[str, Any] | None = None,
    *,
    apply_converters: bool = True,
) -> "ModelSpec":
    """``configs/presets/models/<preset>`` -> the built ``ModelSpec``.

    *model_overrides* merges onto the preset's ``model:`` section, as an experiment
    does; ``spec.model`` is the materialized tree.
    """
    model = deep_merge(
        load_raw_document(str(PRESET_ROOT / preset)).data["model"],
        model_overrides or {},
    )
    section = decode(ModelSection, model, path="model")
    return build_model_spec(section, apply_converters=apply_converters)


def init_policy(name: str):
    """The init policy of ``configs/presets/init/<name>.yaml``."""
    from torchtitan.config.decode import decode
    from torchtitan.config.loader import load_raw_document
    from torchtitan.config.schema.models.common.init_policy import InitPolicy

    document = load_raw_document(str(PRESET_ROOT.parent / "init" / f"{name}.yaml"))
    return decode(InitPolicy, document.data["model"]["config"]["init"], path="init")
