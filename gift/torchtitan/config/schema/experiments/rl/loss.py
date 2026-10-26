from torchtitan.config.configurable import config, Configurable


@config(family="rl loss")
class RLLossConfig(Configurable.Config):
    """Base config for RL losses."""
