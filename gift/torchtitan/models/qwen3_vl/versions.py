"""Runtime metadata for qwen3_vl, referenced by schema ``_runtime_spec``."""

from torchtitan.config.registry import ArchRuntime
from torchtitan.models.qwen3_vl.parallelize import parallelize_qwen3_vl
from torchtitan.models.qwen3_vl.state_dict_adapter import Qwen3VLStateDictAdapter

BASE = ArchRuntime(
    parallelize_fn=parallelize_qwen3_vl,
    pipelining_fn=None,
    state_dict_adapter=Qwen3VLStateDictAdapter,
)
