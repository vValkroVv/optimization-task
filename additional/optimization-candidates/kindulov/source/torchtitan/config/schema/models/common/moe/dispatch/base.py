from torchtitan.config.configurable import config, Configurable


@config(family="token_dispatcher")
class BaseTokenDispatcherConfig(Configurable.Config):
    """Token dispatcher; top_k and num_experts arrive with the ``dispatch`` call."""
