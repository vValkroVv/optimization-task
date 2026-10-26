from typing import ClassVar

from torchtitan.config.configurable import config
from torchtitan.config.schema.models.common.nn_modules import LinearConfig


@config
class FP8LinearConfig(LinearConfig):
    _owner_spec: ClassVar[str] = "torchtitan.ops.float8.cuda_gemm:FP8Linear"
