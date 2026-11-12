"""Opt-in completion checkpoints for the pinned HybridEP combine backward."""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, TextIO, TypeAlias

import torch
import torch.distributed as dist

from torchtitan.callbacks.base import Observer, ObserverState
from torchtitan.config import Configurable
from torchtitan.config.schema.experiments.cuda_graph_campaign.combine_bwd_completion import (
    CampaignCombineBwdCompletionConfig,
)


_HYBRIDEP_MODULE = "torchtitan.distributed.deepep.hybridep"
_HYBRIDEP_SHA256 = "12b1921372f265ede5cca4974f5867f954f84862cc257ff8e02bb7ecafa0814d"
_HELPER_NAME = "_run_on_comm_stream"
_SELECTED_MODULE = "torchtitan.distributed.deepep.hybridep"
_SELECTED_QUALNAME = "_combine_bwd_impl.<locals>.<lambda>"
_SELECTED_LINE = 651
_FIRST_UPDATE = 21
_LAST_UPDATE = 60
_WORLD_SIZE = 8
_EXPERT_PARALLEL = 8
_PIPELINE_PARALLEL = 1
_MICROBATCHES = 4


@dataclass
class _AdapterState:
    module: ModuleType
    original: Callable[..., Any]
    observer: CampaignCombineBwdCompletion | None = None
    adapter: Callable[..., Any] | None = None
    output: TextIO | None = None
    output_path: Path | None = None
    rank: int | None = None
    world_size: int | None = None
    fit_started: bool = False
    closed: bool = False
    training_context: bool = False
    optimizer_completed: bool = False
    armed: bool = False
    step: int | None = None
    microbatch_idx: int | None = None
    next_microbatch_idx: int = 0
    global_call_ordinal: int = 0
    per_update_call_ordinal: int = 0
    microbatch_call_count: int = 0
    update_call_count: int = 0
    update_microbatch_counts: list[int] = field(default_factory=list)
    active_updates_started: list[int] = field(default_factory=list)
    active_updates_optimizer_completed: list[int] = field(default_factory=list)
    active_microbatch_zero_counts: list[dict[str, int]] = field(default_factory=list)
    active_selected_calls: int = 0
    active_completed_checkpoints: int = 0
    fit_end_seen: bool = False


_ACTIVE_STATE: _AdapterState | None = None


def _expected_hybridep_path() -> Path:
    return Path(__file__).resolve().parents[2] / "distributed/deepep/hybridep.py"


def _helper_signature_is_pinned(helper: Any) -> bool:
    if not inspect.isfunction(helper) or hasattr(helper, "__wrapped__"):
        return False
    parameters = list(inspect.signature(helper, follow_wrapped=False).parameters.values())
    return (
        len(parameters) == 2
        and parameters[0].name == "fn"
        and parameters[0].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        and parameters[1].name == "produced_on_compute"
        and parameters[1].kind is inspect.Parameter.VAR_POSITIONAL
    )


def _verify_install_target(
    module: ModuleType,
    *,
    expected_path: Path,
    expected_sha256: str,
) -> Callable[..., Any]:
    module_path = Path(getattr(module, "__file__", "")).resolve()
    if module_path != expected_path.resolve():
        raise RuntimeError(
            f"HybridEP module is not the task source: {module_path} != {expected_path}"
        )
    if hashlib.sha256(module_path.read_bytes()).hexdigest() != expected_sha256:
        raise RuntimeError("HybridEP source hash does not match the reviewed pin")

    helper = getattr(module, _HELPER_NAME, None)
    if (
        not _helper_signature_is_pinned(helper)
        or getattr(helper, "__module__", None) != module.__name__
        or getattr(helper, "__name__", None) != _HELPER_NAME
        or getattr(helper, "__qualname__", None) != _HELPER_NAME
    ):
        raise RuntimeError("HybridEP communication helper does not match the pinned function")
    helper_path = Path(inspect.getsourcefile(helper) or "").resolve()
    if helper_path != expected_path.resolve() or helper.__code__.co_firstlineno != 121:
        raise RuntimeError("HybridEP communication helper source location is not pinned")
    if getattr(module, "_buffer", None) is not None:
        raise RuntimeError("install the completion diagnostic before HybridEP buffer construction")

    source_lines = module_path.read_text(encoding="utf-8").splitlines()
    if (
        len(source_lines) < _SELECTED_LINE
        or "lambda: _buffer.combine_with_unpermute(" not in source_lines[_SELECTED_LINE - 1]
    ):
        raise RuntimeError("HybridEP combine backward selector source is not pinned")
    return helper


def _write_record(
    state: _AdapterState,
    record_type: str,
    **fields: Any,
) -> None:
    if state.output is None:
        raise RuntimeError("combine_bwd completion log is not open")
    record = {
        "schema_version": 1,
        "record_type": record_type,
        "rank": state.rank,
        "step": state.step,
        "microbatch_idx": state.microbatch_idx,
        "global_call_ordinal": state.global_call_ordinal,
        "per_update_call_ordinal": state.per_update_call_ordinal,
        **fields,
    }
    state.output.write(json.dumps(record, sort_keys=True, allow_nan=False))
    state.output.write("\n")
    state.output.flush()


def _record_error_preserving(
    state: _AdapterState,
    error: BaseException,
    record_type: str,
    **fields: Any,
) -> None:
    try:
        _write_record(
            state,
            record_type,
            exception_type=type(error).__name__,
            exception_text=str(error),
            **fields,
        )
    except BaseException as log_error:
        try:
            error.add_note(
                f"combine_bwd completion error record also failed: "
                f"{type(log_error).__name__}: {log_error}"
            )
        except AttributeError:
            pass


def _selected_function(fn: Any) -> bool:
    module_name = getattr(fn, "__module__", None)
    qualified_name = getattr(fn, "__qualname__", None)
    if module_name != _SELECTED_MODULE or qualified_name != _SELECTED_QUALNAME:
        return False
    if not inspect.isfunction(fn):
        raise RuntimeError("HybridEP combine backward binding is not a Python function")

    code = fn.__code__
    code_path = Path(code.co_filename).resolve()
    if code_path != _expected_hybridep_path().resolve() or code.co_firstlineno != _SELECTED_LINE:
        raise RuntimeError("HybridEP combine backward lambda source location changed")
    return True


def _call_adapter(state: _AdapterState, fn: Callable[[], Any], *produced_on_compute: Any) -> Any:
    try:
        selected = _selected_function(fn)
    except BaseException as error:
        _record_error_preserving(state, error, "selector_error")
        raise

    if not selected:
        return state.original(fn, *produced_on_compute)

    state.global_call_ordinal += 1
    ordinal = state.global_call_ordinal
    per_update_ordinal: int | None = None
    if state.step is not None:
        state.per_update_call_ordinal += 1
        state.update_call_count += 1
        state.microbatch_call_count += 1 if state.microbatch_idx is not None else 0
        per_update_ordinal = state.per_update_call_ordinal
    counters = {
        "step": state.step,
        "microbatch_idx": state.microbatch_idx,
        "global_call_ordinal": ordinal,
        "per_update_call_ordinal": per_update_ordinal,
    }
    if state.armed:
        state.active_selected_calls += 1

    try:
        out = state.original(fn, *produced_on_compute)
    except BaseException as error:
        _record_error_preserving(state, error, "native_call_error", **counters)
        raise

    if (
        not state.fit_started
        or state.closed
        or not state.training_context
        or state.optimizer_completed
        or state.step is None
        or state.microbatch_idx is None
    ):
        error = RuntimeError("selected combine_bwd call has no valid native training context")
        _record_error_preserving(state, error, "invalid_training_context", **counters)
        raise error

    if not state.armed:
        return out

    _write_record(state, "checkpoint_enter", **counters)
    try:
        stream = torch.cuda.current_stream()
        event = torch.cuda.Event(enable_timing=False)
        event.record(stream)
    except BaseException as error:
        _record_error_preserving(state, error, "event_record_error", **counters)
        raise

    try:
        event.synchronize()
    except BaseException as error:
        _record_error_preserving(state, error, "event_wait_error", **counters)
        raise

    _write_record(state, "checkpoint_complete", **counters)
    state.active_completed_checkpoints += 1
    return out


def install_combine_bwd_completion() -> Callable[[], None]:
    """Install the exact-pinned adapter and return its idempotent restorer."""
    global _ACTIVE_STATE

    if _ACTIVE_STATE is not None:
        raise RuntimeError("combine_bwd completion adapter is already installed")
    module = importlib.import_module(_HYBRIDEP_MODULE)
    original = _verify_install_target(
        module,
        expected_path=_expected_hybridep_path(),
        expected_sha256=_HYBRIDEP_SHA256,
    )
    state = _AdapterState(module=module, original=original)

    def adapter(fn: Callable[[], Any], *produced_on_compute: Any) -> Any:
        return _call_adapter(state, fn, *produced_on_compute)

    adapter.__name__ = _HELPER_NAME
    adapter.__qualname__ = _HELPER_NAME
    adapter.__module__ = module.__name__
    state.adapter = adapter
    module._run_on_comm_stream = adapter
    _ACTIVE_STATE = state

    restored = False

    def restore() -> None:
        nonlocal restored
        global _ACTIVE_STATE

        if restored:
            return
        restored = True
        state.closed = True
        if getattr(module, _HELPER_NAME, None) is not adapter:
            _ACTIVE_STATE = None
            raise RuntimeError(
                "HybridEP communication helper was replaced during the diagnostic; "
                "the unexpected replacement was left untouched"
            )
        module._run_on_comm_stream = original
        _ACTIVE_STATE = None

    return restore


def _active_state() -> _AdapterState:
    if _ACTIVE_STATE is None:
        raise RuntimeError(
            "campaign_combine_bwd_completion requires its dedicated training entrypoint"
        )
    return _ACTIVE_STATE


def _validate_target_runtime(state: ObserverState) -> tuple[int, int]:
    config = state.config
    parallel_dims = state.parallel_dims
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("combine_bwd completion requires an initialized process group")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != _WORLD_SIZE or not 0 <= rank < _WORLD_SIZE:
        raise RuntimeError(f"combine_bwd completion requires world size {_WORLD_SIZE}")
    if (
        int(getattr(parallel_dims, "world_size", -1)) != _WORLD_SIZE
        or int(getattr(parallel_dims, "pp", -1)) != _PIPELINE_PARALLEL
        or int(getattr(parallel_dims, "ep", -1)) != _EXPERT_PARALLEL
        or int(config.parallelism.pipeline_parallel_degree) != _PIPELINE_PARALLEL
        or int(config.parallelism.expert_parallel_degree) != _EXPERT_PARALLEL
    ):
        raise RuntimeError("combine_bwd completion requires PP1 and EP8 on eight ranks")
    if torch.device(state.device).type != "cuda":
        raise RuntimeError("combine_bwd completion requires a CUDA training device")
    if os.environ.get("TORCHINDUCTOR_CUDAGRAPHS") != "0":
        raise RuntimeError("TORCHINDUCTOR_CUDAGRAPHS must be set to 0")
    if int(state.step) != 0:
        raise RuntimeError("combine_bwd completion requires a fresh step-zero run")

    model_spec = config.model_spec
    if getattr(model_spec, "name", None) != "gift_moe" or getattr(
        model_spec, "flavor", None
    ) != "3B":
        raise RuntimeError("combine_bwd completion requires the gift_moe 3B preset")
    training = config.training
    if (
        int(training.local_batch_size) != 8
        or int(training.global_batch_size) != 256
        or int(training.seq_len) != 4096
        or int(state.gradient_accumulation_steps) != _MICROBATCHES
    ):
        raise RuntimeError("combine_bwd completion requires the fixed 3B diagnostic batch")
    if int(training.steps) != _LAST_UPDATE:
        raise RuntimeError(
            f"combine_bwd completion requires native training.steps={_LAST_UPDATE}"
        )
    if (
        getattr(config.dataloader, "dataset", None) != "c4_test"
        or getattr(config.activation_checkpoint, "mode", None) != "selective"
        or getattr(config.parallelism, "fsdp_backend", None) != "fsdp2"
        or config.compile.enable is not True
        or list(config.compile.components) != ["model", "loss"]
        or config.compile.backend != "inductor"
        or getattr(config.fp8, "dense_use_shared_quantizer", None) is not True
        or getattr(config.fp8, "sparse_use_grouped_gemm", None) is not True
        or config.checkpoint.enable is not False
        or config.comm.mode != "default"
    ):
        raise RuntimeError("combine_bwd completion requires the pinned compile/AC/data runtime")
    return rank, world_size


class CampaignCombineBwdCompletion(Configurable, Observer):
    """Rank-local event checkpoint observer for HybridEP combine backward."""

    Config: TypeAlias = CampaignCombineBwdCompletionConfig

    def __init__(self, config: Config) -> None:
        self.config = config
        self._state = _active_state()
        if self._state.observer is not None:
            raise RuntimeError("exactly one combine_bwd completion callback is allowed")
        self._state.observer = self

    def fit_start(self, state: ObserverState) -> None:
        adapter_state = self._state
        if adapter_state.observer is not self:
            raise RuntimeError("combine_bwd completion callback is not paired with its entrypoint")
        if adapter_state.fit_started or adapter_state.closed:
            raise RuntimeError("combine_bwd completion callback fit_start is out of order")
        rank, world_size = _validate_target_runtime(state)
        output_dir = Path(self.config.output_dir)
        if not output_dir.is_absolute():
            output_dir = Path(state.config.dump_folder) / output_dir
        output_path = output_dir / f"rank-{rank:05d}.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output = output_path.open("x", encoding="utf-8")

        adapter_state.rank = rank
        adapter_state.world_size = world_size
        adapter_state.output_path = output_path
        adapter_state.output = output
        adapter_state.fit_started = True
        try:
            _write_record(
                adapter_state,
                "activation",
                status="enabled",
                source={
                    "module": _HYBRIDEP_MODULE,
                    "path": str(Path(adapter_state.module.__file__).resolve()),
                    "sha256": _HYBRIDEP_SHA256,
                    "helper_definition_line": 121,
                    "selected_lambda_line": _SELECTED_LINE,
                },
                window={"first_update": _FIRST_UPDATE, "last_update": _LAST_UPDATE},
                parallelism={"world_size": world_size, "pp": 1, "ep": 8},
                preset={"model": "gift_moe", "flavor": "3B"},
            )
        except BaseException:
            adapter_state.fit_started = False
            output.close()
            adapter_state.output = None
            raise

    def batch_start(self, state: ObserverState) -> None:
        adapter_state = self._require_open()
        if adapter_state.training_context:
            raise RuntimeError("combine_bwd completion update started before previous batch_end")
        step = int(state.step)
        adapter_state.step = step
        adapter_state.microbatch_idx = None
        adapter_state.next_microbatch_idx = 0
        adapter_state.per_update_call_ordinal = 0
        adapter_state.microbatch_call_count = 0
        adapter_state.update_call_count = 0
        adapter_state.update_microbatch_counts = []
        adapter_state.optimizer_completed = False
        adapter_state.training_context = True
        adapter_state.armed = _FIRST_UPDATE <= step <= _LAST_UPDATE
        if adapter_state.armed:
            adapter_state.active_updates_started.append(step)
        _write_record(
            adapter_state,
            "update_start",
            armed=adapter_state.armed,
            selected_calls_before_update=adapter_state.global_call_ordinal,
        )

    def microbatch_start(self, state: ObserverState) -> None:
        adapter_state = self._require_training_context(state)
        if adapter_state.microbatch_idx is not None:
            self._finish_microbatch(adapter_state)
        microbatch_idx = int(state.microbatch_idx)
        if microbatch_idx != adapter_state.next_microbatch_idx or not 0 <= microbatch_idx < _MICROBATCHES:
            raise RuntimeError(
                f"combine_bwd completion expected microbatch "
                f"{adapter_state.next_microbatch_idx}, got {microbatch_idx}"
            )
        adapter_state.microbatch_idx = microbatch_idx
        adapter_state.microbatch_call_count = 0
        _write_record(
            adapter_state,
            "microbatch_start",
            selected_calls_before_microbatch=adapter_state.per_update_call_ordinal,
        )

    def after_optimizer_step(self, state: ObserverState) -> None:
        adapter_state = self._require_training_context(state)
        if adapter_state.microbatch_idx is None:
            raise RuntimeError("combine_bwd completion optimizer boundary lacks microbatches")
        self._finish_microbatch(adapter_state)
        if adapter_state.next_microbatch_idx != _MICROBATCHES:
            raise RuntimeError("combine_bwd completion did not observe all four microbatches")
        adapter_state.optimizer_completed = True
        if adapter_state.armed:
            adapter_state.active_updates_optimizer_completed.append(adapter_state.step)
        _write_record(
            adapter_state,
            "after_optimizer_step",
            selected_calls=adapter_state.update_call_count,
            microbatch_selected_calls=list(adapter_state.update_microbatch_counts),
            completion_scope="host optimizer and scheduler submission callback",
        )

    def batch_end(self, state: ObserverState) -> None:
        adapter_state = self._require_training_context(state)
        if not adapter_state.optimizer_completed:
            raise RuntimeError("combine_bwd completion batch_end preceded optimizer submission")
        _write_record(
            adapter_state,
            "update_end",
            armed=adapter_state.armed,
            selected_calls=adapter_state.update_call_count,
            microbatch_selected_calls=list(adapter_state.update_microbatch_counts),
        )
        self._disarm(adapter_state)

    def validation_start(self, state: ObserverState) -> None:
        adapter_state = self._require_open()
        _write_record(adapter_state, "validation_start")
        self._disarm(adapter_state)

    def validation_end(self, state: ObserverState) -> None:
        adapter_state = self._require_open()
        _write_record(adapter_state, "validation_end")
        self._disarm(adapter_state)

    def fit_end(self, state: ObserverState) -> None:
        adapter_state = self._require_open()
        self._disarm(adapter_state)
        expected_updates = list(range(_FIRST_UPDATE, _LAST_UPDATE + 1))
        complete = (
            adapter_state.active_updates_started == expected_updates
            and adapter_state.active_updates_optimizer_completed == expected_updates
            and not adapter_state.active_microbatch_zero_counts
            and adapter_state.active_selected_calls > 0
            and adapter_state.active_completed_checkpoints
            == adapter_state.active_selected_calls
        )
        adapter_state.fit_end_seen = True
        status = "complete" if complete else "invalid"
        _write_record(
            adapter_state,
            "diagnostic_result",
            status=status,
            expected_updates=expected_updates,
            observed_updates=adapter_state.active_updates_started,
            optimizer_completed_updates=adapter_state.active_updates_optimizer_completed,
            zero_call_microbatches=adapter_state.active_microbatch_zero_counts,
            active_selected_calls=adapter_state.active_selected_calls,
            active_completed_checkpoints=adapter_state.active_completed_checkpoints,
            peer_progress_certified=False,
        )
        if not complete:
            raise RuntimeError("combine_bwd completion activation evidence is incomplete")

    def close(self) -> None:
        adapter_state = self._state
        self._disarm(adapter_state)
        output = adapter_state.output
        if output is None:
            return
        active_error = sys.exception()
        cleanup_errors: list[BaseException] = []
        try:
            if not adapter_state.fit_end_seen:
                _write_record(
                    adapter_state,
                    "diagnostic_result",
                    status="incomplete",
                    active_selected_calls=adapter_state.active_selected_calls,
                    active_completed_checkpoints=adapter_state.active_completed_checkpoints,
                    peer_progress_certified=False,
                )
        except BaseException as error:
            cleanup_errors.append(error)
        try:
            output.flush()
        except BaseException as error:
            cleanup_errors.append(error)
        try:
            output.close()
        except BaseException as error:
            cleanup_errors.append(error)
        finally:
            adapter_state.output = None
            adapter_state.closed = True
        if cleanup_errors:
            first_error = cleanup_errors[0]
            for cleanup_error in cleanup_errors[1:]:
                try:
                    first_error.add_note(
                        "additional completion-log cleanup error: "
                        f"{type(cleanup_error).__name__}: {cleanup_error}"
                    )
                except AttributeError:
                    pass
            if active_error is not None:
                try:
                    active_error.add_note(
                        "combine_bwd completion callback cleanup failed: "
                        f"{type(first_error).__name__}: {first_error}"
                    )
                except AttributeError:
                    pass
                return
            raise first_error

    def _require_open(self) -> _AdapterState:
        adapter_state = self._state
        if not adapter_state.fit_started or adapter_state.output is None or adapter_state.closed:
            raise RuntimeError("combine_bwd completion callback is not active")
        return adapter_state

    def _require_training_context(self, state: ObserverState) -> _AdapterState:
        adapter_state = self._require_open()
        if not adapter_state.training_context or adapter_state.step != int(state.step):
            raise RuntimeError("combine_bwd completion callback received an invalid update boundary")
        return adapter_state

    @staticmethod
    def _finish_microbatch(adapter_state: _AdapterState) -> None:
        microbatch_idx = adapter_state.microbatch_idx
        if microbatch_idx is None:
            return
        adapter_state.update_microbatch_counts.append(adapter_state.microbatch_call_count)
        if adapter_state.armed and adapter_state.microbatch_call_count == 0:
            adapter_state.active_microbatch_zero_counts.append(
                {"step": adapter_state.step, "microbatch_idx": microbatch_idx}
            )
        adapter_state.next_microbatch_idx += 1
        _write_record(
            adapter_state,
            "microbatch_end",
            microbatch_idx=microbatch_idx,
            selected_calls=adapter_state.microbatch_call_count,
            selected_calls_in_update=adapter_state.update_call_count,
        )
        adapter_state.microbatch_idx = None

    @staticmethod
    def _disarm(adapter_state: _AdapterState) -> None:
        adapter_state.armed = False
        adapter_state.training_context = False
        adapter_state.microbatch_idx = None


__all__ = ["CampaignCombineBwdCompletion", "install_combine_bwd_completion"]
