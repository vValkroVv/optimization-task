from pathlib import Path

from omegaconf import OmegaConf as om
from omegaconf import DictConfig


def register_relative_config_loader(base_path: Path) -> None:
    """
    Register a resolver for loading configs relative to a base path.

    Args:
        base_path: The base path to resolve paths relative to.
    Returns:
        None
    """
    base = Path(base_path).resolve()

    def rp_load(path: str):
        return om.load((base / path).resolve())

    def rp_merge(*config_paths: list[str]):
        if not config_paths:
            return om.create({})

        result_config = om.create({})

        for c_p in config_paths:
            result_config = om.merge(result_config, om.load((base / c_p).resolve()))

        return result_config

    om.register_new_resolver("rp.load", rp_load, replace=True)
    om.register_new_resolver("rp.merge", rp_merge, replace=True)


def apply_overrides(cfg: DictConfig) -> DictConfig:
    """
    Apply overrides to the config.

    Args:
        cfg: The config to apply overrides to.

    Returns:
        The config with overrides applied.
    """
    if "overrides" not in cfg or cfg.overrides is None:
        return cfg
    base = om.create(cfg)
    del base["overrides"]
    return om.merge(base, cfg.overrides)
