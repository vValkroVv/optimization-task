"""Runtime metadata for qwen3, referenced by schema ``_runtime_spec``."""

from torchtitan.config.registry import ArchRuntime
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.models.qwen3.parallelize import parallelize_qwen3
from torchtitan.models.qwen3.state_dict_adapter import Qwen3StateDictAdapter

BASE = ArchRuntime(
    parallelize_fn=parallelize_qwen3,
    pipelining_fn=pipeline_llm,
    state_dict_adapter=Qwen3StateDictAdapter,
)
