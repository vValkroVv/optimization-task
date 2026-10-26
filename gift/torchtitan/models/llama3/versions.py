"""Runtime metadata for llama3, referenced by schema ``_runtime_spec``."""

from torchtitan.config.registry import ArchRuntime
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.models.llama3.parallelize import parallelize_llama
from torchtitan.models.llama3.state_dict_adapter import Llama3StateDictAdapter

BASE = ArchRuntime(
    parallelize_fn=parallelize_llama,
    pipelining_fn=pipeline_llm,
    state_dict_adapter=Llama3StateDictAdapter,
)
