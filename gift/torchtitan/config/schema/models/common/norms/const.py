from torchtitan.config.schema.protocols.types import StrEnum

DEFAULT_NORM_EPS: float = 1e-5


class GatedNormActivation(StrEnum):
    """Output-gate activations fla's ``FusedRMSNormGated`` accepts.

    ``SWISH`` and ``SILU`` select the same kernel branch.
    """

    SWISH = "swish"
    SILU = "silu"
    SIGMOID = "sigmoid"
