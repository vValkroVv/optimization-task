from typing import ClassVar

from torchtitan.config.configurable import config, Configurable
from torchtitan.config.errors import ConfigFieldError
from torchtitan.config.function import Function

PROFILE_DIR = "profiling/traces"  # Profiler.Config.save_traces_folder default
MEMORY_DIR = (
    "profiling/memory_snapshot"  # Profiler.Config.save_memory_snapshot_folder default
)


@config
class ProfilerConfig(Configurable.Config):
    _owner_spec: ClassVar[str] = "torchtitan.tools.profiler:Profiler"

    def __post_init__(self) -> None:
        if self.profile_ranks is None:
            return
        if not isinstance(self.profile_ranks, list):
            raise ConfigFieldError("profile_ranks", "must be a list or None")
        if not self.profile_ranks:
            raise ConfigFieldError("profile_ranks", "must not be empty")
        if any(type(rank) is not int or rank < 0 for rank in self.profile_ranks):
            raise ConfigFieldError(
                "profile_ranks", "must contain nonnegative integer global ranks"
            )
        if len(self.profile_ranks) != len(set(self.profile_ranks)):
            raise ConfigFieldError("profile_ranks", "must not contain duplicate ranks")

    enable_profiling: bool = False

    save_traces_folder: str = PROFILE_DIR

    profile_ranks: list[int] | None = None
    """Global ranks that collect traces; ``None`` enables profiling on every rank."""

    record_shapes: bool = True
    """Whether to include tensor shape metadata in profiler traces."""

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
