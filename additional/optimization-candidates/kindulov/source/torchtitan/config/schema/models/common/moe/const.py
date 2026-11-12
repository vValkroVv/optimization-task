from torchtitan.config.schema.protocols.types import StrEnum


class ScoreFunc(StrEnum):
    """Router scoring functions supported by ``TokenChoiceTopKRouter``."""

    SOFTMAX = "softmax"
    SIGMOID = "sigmoid"
