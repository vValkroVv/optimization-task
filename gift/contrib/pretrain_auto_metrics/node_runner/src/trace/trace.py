from .events import EventsStream, EventStatus
from .metrics import MetricsStream, NvidiaSmiSampler


class Trace:
    """Single entry point for the node's traces: the master's events, the
    node-wide metrics jsonl and the nvidia-smi sampler feeding it."""

    _nvidia_smi = NvidiaSmiSampler()

    @classmethod
    def setup(cls, config_manager) -> None:
        EventsStream.setup(config_manager)
        MetricsStream.setup(config_manager)
        cls._nvidia_smi.start()

    @classmethod
    def close(cls, status: EventStatus = "ok") -> None:
        cls._nvidia_smi.stop()
        EventsStream.close(status)
        MetricsStream.close()
