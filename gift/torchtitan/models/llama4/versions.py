"""Runtime metadata for llama4, referenced by schema ``_runtime_spec``."""

from torchtitan.config.registry import ArchRuntime
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.models.llama4.parallelize import parallelize_llama
from torchtitan.models.llama4.state_dict_adapter import Llama4StateDictAdapter

BASE = ArchRuntime(
    parallelize_fn=parallelize_llama,
    pipelining_fn=pipeline_llm,
    state_dict_adapter=Llama4StateDictAdapter,
)
