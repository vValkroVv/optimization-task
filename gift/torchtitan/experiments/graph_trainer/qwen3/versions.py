"""Runtime half of the graph-trainer qwen3 models: the graph parallelize
function over qwen3's pipelining and state-dict callables."""

from dataclasses import replace

from torchtitan.models.qwen3.versions import BASE as _QWEN3_BASE

from .parallelize import parallelize_qwen3

BASE = replace(_QWEN3_BASE, parallelize_fn=parallelize_qwen3)
