from torchtitan.config.schema.models.common.norms.base import BaseNormConfig

# A family, not a union: YAML picks a norm by ``type`` like any component.
NormConfig = BaseNormConfig
