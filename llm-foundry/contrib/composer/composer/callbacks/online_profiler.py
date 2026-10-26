import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from composer.core import Callback, State
from composer.loggers import Logger
from composer.profiler.profiler_action import ProfilerAction
from composer.utils import dist

__all__ = ["OnlineProfiler"]

log = logging.getLogger(__name__)


def parse_request_file(request_file: Path) -> Optional[Dict[str, int]]:
    try:
        with request_file.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as e:
        log.warning(f"OnlineProfiler failed to read request file {str(request_file)}: {e}")
        return None

    if not isinstance(payload, dict):
        log.warning(f"OnlineProfiler request must be a JSON object: {str(request_file)}")
        return None

    try:
        warmup, active, repeat = payload["warmup"], payload["active"], payload["repeat"]
    except KeyError as e:
        log.warning(f"OnlineProfiler request has invalid file contract in {str(request_file)}: {e}")
        return None

    if (not isinstance(warmup, int) or not isinstance(active, int) or not isinstance(repeat, int)):
        log.warning(f"OnlineProfiler request must contain int fields warmup/active/repeat in {str(request_file)} but got: warmup={warmup}, active={active}, repeat={repeat}")
        return None

    if warmup < 0 or active <= 0 or repeat <= 0:
        log.warning(f"OnlineProfiler request contains invalid values in {str(request_file)}: warmup={warmup}, active={active}, repeat={repeat}")
        return None

    return {
        "wait": 0,
        "warmup": int(warmup),
        "active": int(active),
        "repeat": int(repeat),
    }


class OnlineProfiler(Callback):
    """Run on-demand profiler windows controlled by signal files.

    Requested file (JSON):

        {
          "warmup": 5,
          "active": 2,
          "repeat": 1
        }
    """

    def __init__(
        self,
        signal_dir: Optional[str],
        poll_interval_batches: int = 50,
    ) -> None:
        self.poll_interval_batches = poll_interval_batches
        self.signal_dir = signal_dir

        self._disabled = False

        self._original_schedule: Optional[Callable[[State], ProfilerAction]] = None
        self._wrapped_schedule: Optional[Callable[[State], ProfilerAction]] = None

        self._active_window: Optional[Dict[str, int]] = None
        self._active_start_batch: Optional[int] = None
        self._active_request_file: Optional[Path] = None
        self._pending_cleanup = False

    def init(self, state: State, logger: Logger) -> None:
        del logger

        if state.profiler is None:
            log.warning("OnlineProfiler is enabled but trainer profiler is disabled. Callback will be a no-op.")
            self._disabled = True
            return

        if self.signal_dir is None:
            log.warning("OnlineProfiler requires `signal_dir` in callback config. Callback will be disabled.")
            self._disabled = True
            return

        Path(self.signal_dir).mkdir(parents=True, exist_ok=True)

        self._original_schedule = state.profiler.schedule

        def wrapped_schedule(schedule_state: State) -> ProfilerAction:
            online_action = self._get_window_action(schedule_state)
            if online_action is not None:
                return online_action
            return self._original_schedule(schedule_state)

        self._wrapped_schedule = wrapped_schedule
        state.profiler.schedule = wrapped_schedule

    def batch_start(self, state: State, logger: Logger) -> None:
        # Только ранг 0 трогает файловую систему, остальные получают команды через broadcast

        del logger  # unused
        if self._disabled: return

        batch = int(state.timestamp.batch)
        if (batch + 1) % self.poll_interval_batches != 0: return

        request: Optional[Dict[str, Any]] = None
        if dist.get_global_rank() == 0:
            request = self._build_request_on_rank_zero()
        payload = [request]
        dist.broadcast_object_list(payload, src=0)
        request = payload[0]

        if request is None: return
        self._start_window(
            window=request,
            start_batch=batch,
        )

    def batch_checkpoint(self, state: State, logger: Logger) -> None:
        del state, logger  # unused
        if self._disabled or not self._pending_cleanup: return

        if dist.get_global_rank() == 0 and self._active_request_file is not None:
            try:
                self._active_request_file.unlink()
            except FileNotFoundError:
                pass
            except Exception as e:
                log.warning(f"OnlineProfiler failed to remove request file {str(self._active_request_file)}: {e}")
            else:
                log.info(f"OnlineProfiler processed request file removed: {str(self._active_request_file)}")

        self._finish_window()

    def close(self, state: State, logger: Logger) -> None:
        del logger  # unused
        if state.profiler is not None and self._original_schedule is not None:
            if self._wrapped_schedule is not None and state.profiler.schedule == self._wrapped_schedule:
                state.profiler.schedule = self._original_schedule
        self._finish_window()

    def _build_request_on_rank_zero(self) -> Optional[Dict[str, Any]]:
        signal_dir = Path(self.signal_dir)
        files = sorted(
            [p for p in signal_dir.glob("need_profiler*.json") if p.is_file()],
            key=lambda p: (p.stat().st_mtime, p.name),
        )

        if len(files) > 1:
            log.warning(
                f"OnlineProfiler found {len(files)} request files in {str(signal_dir)}.\nOnly one request can be processed at a time."
            )

        if self._active_window is not None or len(files) == 0:
            return None

        request_file = files[0]
        try:
            window = parse_request_file(request_file)
        except Exception:
            log.warning("Failed to parse OnlineProfiler request file, skipping: " + str(request_file))
            return None
            
        if window is None:
            try:
                request_file.unlink()
            except FileNotFoundError:
                pass
            except Exception as e:
                log.warning(f"OnlineProfiler failed to remove invalid request file {str(request_file)}: {e}")
            else:
                log.warning(f"OnlineProfiler removed invalid request file: {str(request_file)}")
            return None
        self._active_request_file = request_file

        return window

    def _start_window(self, window: Dict[str, int], start_batch: int) -> None:
        if self._active_window is not None:
            return
        self._active_window = window
        self._active_start_batch = start_batch
        self._pending_cleanup = False
        log.info(
            f"OnlineProfiler window started from {str(self._active_request_file)}:"
            f"wait={window['wait']} warmup={window['warmup']} active={window['active']} repeat={window['repeat']}"
            f"(start_batch={start_batch})"
        )

    def _finish_window(self) -> None:
        self._active_window = None
        self._active_start_batch = None
        self._active_request_file = None
        self._pending_cleanup = False

    def _get_window_action(self, state: State) -> Optional[ProfilerAction]:
        if self._active_window is None or self._active_start_batch is None:
            return None

        wait = self._active_window["wait"]
        warmup = self._active_window["warmup"]
        active = self._active_window["active"]
        repeat = self._active_window["repeat"]

        cycle_len = wait + warmup + active
        if cycle_len == 0:
            self._finish_window()
            return None

        batch = int(state.timestamp.batch)
        offset = batch - self._active_start_batch
        if offset < 0:
            return None

        total_window_len = cycle_len * repeat
        if offset >= total_window_len:
            return None

        position_in_cycle = offset % cycle_len
        if position_in_cycle < wait:
            return ProfilerAction.SKIP
        if position_in_cycle < wait + warmup:
            return ProfilerAction.WARMUP

        cycle_index = offset // cycle_len
        is_last_batch_in_cycle = position_in_cycle == cycle_len - 1
        is_last_cycle = cycle_index == repeat - 1

        if is_last_batch_in_cycle:
            if is_last_cycle:
                self._pending_cleanup = True
            return ProfilerAction.ACTIVE_AND_SAVE
        return ProfilerAction.ACTIVE
