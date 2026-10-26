"""Runtime metadata for deepseek_v3, referenced by schema ``_runtime_spec``."""

from torchtitan.config.registry import ArchRuntime
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.models.deepseek_v3.parallelize import parallelize_deepseekv3
from torchtitan.models.deepseek_v3.state_dict_adapter import DeepSeekV3StateDictAdapter

BASE = ArchRuntime(
    parallelize_fn=parallelize_deepseekv3,
    pipelining_fn=pipeline_llm,
    state_dict_adapter=DeepSeekV3StateDictAdapter,
)
