from torchtitan.config.configurable import config, Configurable


@config(family="rl env")
class EnvConfig(Configurable.Config):
    """Base config for RL environments."""
