"""Runtime half of the graph-trainer deepseek_v3: the graph parallelize
function over deepseek_v3's pipelining, load-balancing and state-dict callables."""

from dataclasses import replace

from torchtitan.models.deepseek_v3.versions import BASE as _DEEPSEEK_V3_BASE

from . import _parallelize_fn

BASE = replace(_DEEPSEEK_V3_BASE, parallelize_fn=_parallelize_fn)
