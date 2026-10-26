"""Runtime half of the fault-tolerant llama3: llama3's callables plus the
DiLoCo ``fragment_fn`` in a ``FaultTolerantModelSpec``."""

from dataclasses import replace

from torchtitan.experiments.ft.config.job_config import FaultTolerantModelSpec
from torchtitan.experiments.ft.diloco import fragment_llm
from torchtitan.models.llama3.versions import BASE as _LLAMA3_BASE

BASE = replace(
    _LLAMA3_BASE,
    spec_cls=FaultTolerantModelSpec,
    spec_kwargs={"fragment_fn": fragment_llm},
)
