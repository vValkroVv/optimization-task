"""Runtime metadata for GIFT presets, referenced by schema ``_runtime_spec``.

Both owners use the module-aware adapter: plain MLA stays DeepSeek-compatible,
while composed gated/sandwich/linear blocks use the existing hybrid mappings.
"""

from torchtitan.config.registry import ArchRuntime
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.models.gift_moe._base.parallelize import parallelize_gift_moe
from torchtitan.models.gift_moe.gift_3_5 import GiftHybridStateDictAdapter

BASE = ArchRuntime(
    parallelize_fn=parallelize_gift_moe,
    pipelining_fn=pipeline_llm,
    state_dict_adapter=GiftHybridStateDictAdapter,
)

HYBRID = ArchRuntime(
    parallelize_fn=parallelize_gift_moe,
    pipelining_fn=pipeline_llm,
    state_dict_adapter=GiftHybridStateDictAdapter,
)
