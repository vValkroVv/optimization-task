"""Runtime metadata for gpt_oss, referenced by schema ``_runtime_spec``."""

from torchtitan.config.registry import ArchRuntime
from torchtitan.models.gpt_oss.parallelize import parallelize_gptoss
from torchtitan.models.gpt_oss.state_dict_adapter import GptOssStateDictAdapter

BASE = ArchRuntime(
    parallelize_fn=parallelize_gptoss,
    pipelining_fn=None,
    state_dict_adapter=GptOssStateDictAdapter,
)
