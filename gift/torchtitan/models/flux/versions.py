"""Runtime metadata for flux, referenced by schema ``_runtime_spec``."""

from torchtitan.config.registry import ArchRuntime
from torchtitan.models.flux.model.state_dict_adapter import FluxStateDictAdapter
from torchtitan.models.flux.parallelize import parallelize_flux

BASE = ArchRuntime(
    parallelize_fn=parallelize_flux,
    pipelining_fn=None,
    state_dict_adapter=FluxStateDictAdapter,
)
