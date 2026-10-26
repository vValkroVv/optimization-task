from typing import ClassVar

from torchtitan.config.configurable import config, Configurable
from torchtitan.config.function import Function

PROFILE_DIR = "profiling/traces"  # Profiler.Config.save_traces_folder default
MEMORY_DIR = (
    "profiling/memory_snapshot"  # Profiler.Config.save_memory_snapshot_folder default
)


@config
class ProfilerConfig(Configurable.Config):
    _owner_spec: ClassVar[str] = "torchtitan.tools.profiler:Profiler"

    enable_profiling: bool = False

    save_traces_folder: str = PROFILE_DIR

    profile_freq: int = 10
    """In iterations."""

    profiler_repeat: int | None = None
    """``torch.profiler.schedule`` ``repeat``."""

    profiler_skip_first: int | None = None
    """``torch.profiler.schedule`` ``skip_first``."""

    profiler_skip_first_wait: int | None = None
    """``torch.profiler.schedule`` ``skip_first_wait``."""

    profiler_active: int = 1
    """``torch.profiler.schedule`` ``active``, in steps."""

    profiler_warmup: int = 3
    """``torch.profiler.schedule`` ``warmup``, in steps per cycle."""

    enable_memory_snapshot: bool = False

    save_memory_snapshot_folder: str = MEMORY_DIR

    trace_post_processor: Function.Config | None = None
    """``fn(trace_path: str) -> None`` called after each trace export.

    Set by the trainer that needs it (e.g. the graph trainer), not from YAML.
    """
