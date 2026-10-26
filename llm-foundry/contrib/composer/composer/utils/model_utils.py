from collections.abc import Mapping
from typing import Any, Optional


def get_composer_language_model(composer_model: Any) -> Any:
    """Return the text or omni causal language model from a Composer model."""
    outer_model = composer_model.model
    return getattr(outer_model, "language_model", outer_model)


def get_composer_lm_backbone(composer_model: Any) -> Any:
    """Return the text or omni language-model backbone from a Composer model."""
    return get_composer_language_model(composer_model).model


def get_fsdp_lm_backbone(fsdp_model: Any) -> Any:
    """Return the text or omni language-model backbone from a GigaFSDP model."""
    if hasattr(fsdp_model, "language_model"):
        return fsdp_model.language_model.model
    return fsdp_model.model


def get_lm_config(composer_model: Any) -> Optional[Any]:
    """Return the text or omni LLM config from a Composer model."""
    cfg = getattr(composer_model, "config", None)
    if cfg is None:
        return None
    if hasattr(cfg, "language_model"):
        return cfg.language_model
    return cfg


def cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Read a key from a mapping or attribute-based config."""
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)
