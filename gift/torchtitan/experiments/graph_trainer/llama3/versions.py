"""Runtime half of the graph-trainer llama3: the graph parallelize function
over llama3's pipelining and state-dict callables."""

from dataclasses import replace

from torchtitan.models.llama3.versions import BASE as _LLAMA3_BASE

from . import _parallelize_fn

BASE = replace(_LLAMA3_BASE, parallelize_fn=_parallelize_fn)
