"""Opt-in capture of native allocator statistics at metric reset boundaries."""

from __future__ import annotations

import functools
import importlib
import inspect
import json
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable


DEFAULT_PHASE_BOUNDARIES = {"startup_end_step": 20, "expected_end_step": 500}
_ALLOCATOR_KEYS = (
    "allocated_bytes.all.current",
    "allocated_bytes.all.peak",
    "reserved_bytes.all.current",
    "reserved_bytes.all.peak",
    "active_bytes.all.current",
    "active_bytes.all.peak",
    "num_alloc_retries",
    "num_ooms",
)
_REQUIRED_ALLOCATOR_KEYS = frozenset(_ALLOCATOR_KEYS[:4])
_ACTIVE_OBSERVER: PhaseMemoryObserver | None = None


def _parameter_names(function: Callable[..., Any]) -> tuple[str, ...]:
    return tuple(inspect.signature(function).parameters)


def _expect_signature(
    owner: Any, name: str, expected: tuple[str, ...]
) -> Callable[..., Any]:
    value = getattr(owner, name, None)
    if not callable(value) or _parameter_names(value) != expected:
        actual = _parameter_names(value) if callable(value) else None
        raise RuntimeError(f"phase-memory expected {name}{expected}, found {actual}")
    return value


def _python_int(value: Any) -> int | None:
    return value if type(value) is int else None


def _step_from_call(
    args: tuple[Any, ...], kwargs: dict[str, Any], index: int
) -> int | None:
    if "step" in kwargs:
        value = kwargs["step"]
    elif len(args) > index:
        value = args[index]
    else:
        value = None
    return _python_int(value)


def _required_identity(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"phase-memory requires environment variable {name}")
    return value


def _int_environment(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise RuntimeError(
            f"phase-memory environment variable {name} must be an integer"
        ) from exc
    if parsed < 0:
        raise RuntimeError(
            f"phase-memory environment variable {name} must be nonnegative"
        )
    return parsed


def _clock_bracket() -> dict[str, Any]:
    utc_before = time.time_ns()
    monotonic = time.perf_counter_ns()
    utc_after = time.time_ns()
    return {
        "utc_ns_bracket": [utc_before, utc_after],
        "monotonic_ns": monotonic,
        "utc_bracket_width_ns": utc_after - utc_before,
    }


class _AnchoredTime:
    """Temporarily adds UTC/monotonic anchors around one existing timer read."""

    def __init__(
        self,
        original_time: Any,
        observer: PhaseMemoryObserver,
        kind: str,
        timing_observer: Any,
    ) -> None:
        self._original_time = original_time
        self._observer = observer
        self._kind = kind
        self._timing_observer = timing_observer
        self.captured = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original_time, name)

    def perf_counter(self) -> float:
        if self.captured:
            return self._original_time.perf_counter()

        utc_before = self._original_time.time_ns()
        monotonic_before = self._original_time.perf_counter_ns()
        value = self._original_time.perf_counter()
        monotonic_after = self._original_time.perf_counter_ns()
        utc_after = self._original_time.time_ns()
        self.captured = True
        try:
            self._observer._record_timing_anchor(
                self._kind,
                self._timing_observer,
                value,
                utc_before,
                utc_after,
                monotonic_before,
                monotonic_after,
            )
        except Exception as exc:
            self._observer._add_invalid(
                f"timing anchor capture failed: {type(exc).__name__}"
            )
        return value


class PhaseMemoryObserver:
    """Collects rank-local scalar events while leaving native calls intact."""

    def __init__(
        self,
        output_dir: str | os.PathLike[str],
        phase_boundaries: Mapping[str, int],
        *,
        utils_module: Any,
        timing_module: Any,
        allocator_backend: str | None,
        allocator_backend_reason: str | None,
    ) -> None:
        resolved_output_dir = Path(output_dir).expanduser().resolve()
        self.phase_boundaries = dict(phase_boundaries)
        self.rank = _int_environment("RANK", 0)
        self.local_rank = _int_environment("LOCAL_RANK", 0)
        self.world_size = _int_environment("WORLD_SIZE", 1)
        self.output_path = resolved_output_dir / f"rank-{self.rank:05d}.json"
        self.utils_module = utils_module
        self.timing_module = timing_module
        self._timing_time_module = timing_module.time
        self._allocator_backend = allocator_backend
        self._allocator_backend_reason = allocator_backend_reason
        self._bindings: list[tuple[Any, str, Any, Any]] = []
        self._records: list[dict[str, Any]] = []
        self._invalid_reasons: list[str] = []
        self._incomplete_reasons: list[str] = []
        self._active_call: dict[str, Any] | None = None
        self._metrics_calls_seen = False
        self._reset_depth = 0
        self._generation = -1
        self._generation_valid = False
        self._generation_has_snapshot = False
        self._device_capacity_bytes: int | None = None
        self._last_reset: dict[str, Any] | None = None
        self._last_training_step: int | None = None
        self._expected_training_step = 1
        self._training_log_frequency: int | None = None
        self._timing_anchors: dict[str, dict[str, Any]] = {}
        self._timing_memory_samples: dict[str, dict[str, Any]] = {}
        self._event_count = 0
        self._restore_done = False
        self._flush_attempted = False
        self._flush_error: str | None = None
        self._flushed = False

        self.run_id = _required_identity("TORCHTITAN_PHASE_MEMORY_RUN_ID")
        self.source_id = _required_identity("TORCHTITAN_PHASE_MEMORY_SOURCE_ID")
        self.config_id = _required_identity("TORCHTITAN_PHASE_MEMORY_CONFIG_ID")
        self.device = self._device_mapping()

    def _device_mapping(self) -> dict[str, Any]:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        selector: str | None = None
        mapping_reason = None
        if visible:
            devices = [entry.strip() for entry in visible.split(",")]
            if self.local_rank < len(devices):
                selector = devices[self.local_rank]
            else:
                mapping_reason = (
                    "LOCAL_RANK exceeds the CUDA_VISIBLE_DEVICES entry count"
                )
        else:
            mapping_reason = (
                "CUDA_VISIBLE_DEVICES was not set; physical mapping is external"
            )

        uuid = selector if selector and selector.startswith(("GPU-", "MIG-")) else None
        return {
            "device_type": "cuda",
            "logical_device_index": self.local_rank,
            "logical_index_mapping": (
                "Trainer constructs cuda:LOCAL_RANK and calls device_module.set_device"
            ),
            "cuda_visible_devices": visible,
            "visible_device_selector": selector,
            "device_uuid": uuid,
            "physical_mapping_reason": mapping_reason,
        }

    def _add_invalid(self, reason: str) -> None:
        if reason not in self._invalid_reasons:
            self._invalid_reasons.append(reason)

    def _add_incomplete(self, reason: str) -> None:
        if reason not in self._incomplete_reasons:
            self._incomplete_reasons.append(reason)

    def _new_event_id(self) -> str:
        event_id = f"event-{self._event_count:05d}"
        self._event_count += 1
        return event_id

    def _mark_record_invalid(self, record: dict[str, Any], reason: str) -> None:
        record["status"] = "invalid"
        if reason not in record["reasons"]:
            record["reasons"].append(reason)
        self._add_invalid(reason)

    def _mark_context_invalid(self, context: dict[str, Any], reason: str) -> None:
        context["invalid_reasons"].append(reason)
        self._add_invalid(reason)
        if context.get("snapshot_record") is not None:
            self._mark_record_invalid(context["snapshot_record"], reason)

    def _reset_origin(self) -> dict[str, Any] | None:
        if self._last_reset is None:
            return None
        return {
            "generation": self._last_reset["generation"],
            "context": self._last_reset["context"],
            "step": self._last_reset["step"],
            "clock": self._last_reset["clock"],
        }

    def _capture_snapshot(
        self, memory_stats: Any, context: dict[str, Any]
    ) -> dict[str, Any]:
        reasons: list[str] = []
        if self._generation < 0:
            reasons.append("snapshot occurred before a recognized native reset")
        if not self._generation_valid:
            reasons.append("allocator interval has an unknown reset origin")
        if self._generation_has_snapshot:
            reasons.append(
                "allocator generation produced more than one native snapshot"
            )

        before = _clock_bracket()
        values: dict[str, int | None] = {}
        if isinstance(memory_stats, Mapping):
            for key in _ALLOCATOR_KEYS:
                values[key] = _python_int(memory_stats.get(key))
        else:
            values = {key: None for key in _ALLOCATOR_KEYS}
            reasons.append("native memory statistics were not a mapping")
        after = _clock_bracket()

        missing_required = [
            key for key in _REQUIRED_ALLOCATOR_KEYS if values.get(key) is None
        ]
        if missing_required:
            reasons.append(
                "missing required allocator counters: "
                + ", ".join(sorted(missing_required))
            )
        missing_optional = [
            key
            for key in set(_ALLOCATOR_KEYS) - _REQUIRED_ALLOCATOR_KEYS
            if values.get(key) is None
        ]

        capacity = _python_int(getattr(self.utils_module, "_capacity_bytes", None))
        if capacity is None:
            reasons.append("native device capacity was unavailable at snapshot time")
        elif self._device_capacity_bytes is None:
            self._device_capacity_bytes = capacity
        elif self._device_capacity_bytes != capacity:
            reasons.append("native device capacity changed during observation")

        step = context["step"]
        phase, window = self._phase_and_window(step, context)
        record = {
            "event_id": self._new_event_id(),
            "event_kind": "native_snapshot",
            "status": "invalid" if reasons else "pending_reset",
            "context": context["context"],
            "step": step,
            "reset_generation": self._generation,
            "reset_origin": self._reset_origin(),
            "phase_label": phase,
            "actual_window": window,
            "capture_clock": {
                "semantics": (
                    "raw stats were returned by native memory_stats before this Python-copy bracket"
                ),
                "copy_start": before,
                "copy_end": after,
            },
            "allocator_counters_bytes": values,
            "device_capacity_bytes": capacity,
            "missing_optional_keys": sorted(missing_optional),
            "reasons": reasons,
            "paired_reset_event_id": None,
        }
        self._records.append(record)
        context["snapshot_record"] = record
        self._generation_has_snapshot = True
        for reason in reasons:
            self._add_invalid(reason)
        return record

    def _phase_and_window(
        self, step: int | None, context: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        if step is None:
            return "unknown_step", {
                "label": None,
                "first_training_step": None,
                "last_training_step": None,
            }

        if context["context"] == "validation":
            return "validation", {
                "label": f"validation_at_{step}",
                "first_training_step": None,
                "last_training_step": step,
            }

        reset = self._last_reset
        if reset is None or reset["context"] == "metrics_processor_constructor":
            first_step = 1
            window_label = f"initial_to_{step}"
            origin = "metrics_processor_constructor_reset"
        elif reset["context"] == "unknown":
            first_step = None
            window_label = f"unknown_reset_to_{step}"
            origin = "unknown_reset"
        else:
            previous_step = reset["step"]
            if reset["context"] == "validation":
                previous_step = self._last_training_step
            first_step = previous_step + 1 if previous_step is not None else None
            window_label = (
                f"{first_step}_to_{step}"
                if first_step is not None
                else f"unknown_to_{step}"
            )
            origin = f"{reset['context']}_reset_after_step_{reset['step']}"

        window = {
            "label": window_label,
            "first_training_step": first_step,
            "last_training_step": step,
            "reset_origin": origin,
        }
        if step <= self.phase_boundaries["startup_end_step"]:
            return (
                f"initial_through_{self.phase_boundaries['startup_end_step']}",
                window,
            )
        return f"steady_{window_label}", window

    def _new_metrics_context(
        self, kind: str, step: int | None, owner: Any
    ) -> dict[str, Any]:
        context = {
            "context": kind,
            "step": step,
            "snapshot_record": None,
            "reset_record": None,
            "exception_type": None,
            "invalid_reasons": [],
        }
        if kind == "training":
            self._metrics_calls_seen = True
            if step is None:
                self._mark_context_invalid(
                    context, "training metrics call had no Python integer step"
                )
            elif step != self._expected_training_step:
                self._mark_context_invalid(
                    context,
                    f"training metrics cadence expected step {self._expected_training_step}, got {step}",
                )
            try:
                frequency = _python_int(owner.config.log_freq)
            except (AttributeError, TypeError):
                frequency = None
            if frequency is None or frequency < 1:
                self._mark_context_invalid(
                    context, "native metrics log_freq was unavailable or invalid"
                )
            elif self._training_log_frequency is None:
                self._training_log_frequency = frequency
            elif frequency != self._training_log_frequency:
                self._mark_context_invalid(
                    context, "native metrics log_freq changed during observation"
                )
        return context

    def _invoke_metrics_method(
        self,
        original: Callable[..., Any],
        kind: str,
        owner: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        step_index: int,
    ) -> Any:
        step = _step_from_call(args, kwargs, step_index)
        previous = self._active_call
        if previous is not None:
            reason = "nested native metrics contexts are unsupported"
            self._mark_context_invalid(previous, reason)
        context = self._new_metrics_context(kind, step, owner)
        self._active_call = context
        try:
            result = original(owner, *args, **kwargs)
        except BaseException as exc:
            context["exception_type"] = type(exc).__name__
            self._mark_context_invalid(
                context, f"native {kind} metrics call raised {type(exc).__name__}"
            )
            raise
        finally:
            self._active_call = context
            try:
                self._complete_metrics_context(context)
            except Exception as exc:
                self._mark_context_invalid(
                    context,
                    f"phase-memory context finalization failed: {type(exc).__name__}",
                )
            self._active_call = previous

        if kind == "training" and step is not None:
            self._last_training_step = step
            frequency = self._training_log_frequency
            if frequency is not None:
                self._expected_training_step = (
                    max(2, frequency) if step == 1 else step + frequency
                )
        return result

    def _complete_metrics_context(self, context: dict[str, Any]) -> None:
        snapshot = context["snapshot_record"]
        reset = context["reset_record"]
        if snapshot is None:
            reason = f"native {context['context']} metrics call at step {context['step']} had no captured raw stats"
            self._mark_context_invalid(context, reason)
            missing = {
                "event_id": self._new_event_id(),
                "event_kind": "missing_capture",
                "status": "invalid",
                "context": context["context"],
                "step": context["step"],
                "reset_generation": self._generation,
                "reasons": [reason],
            }
            self._records.append(missing)
        if reset is None:
            reason = f"native {context['context']} metrics call at step {context['step']} had no paired reset"
            self._mark_context_invalid(context, reason)
        if context["exception_type"] is not None:
            return
        if (
            snapshot is not None
            and reset is not None
            and snapshot["status"] == "pending_reset"
            and not snapshot["reasons"]
            and not context["invalid_reasons"]
        ):
            snapshot["status"] = "complete"

    def _record_reset(
        self,
        context: dict[str, Any] | None,
        *,
        nested: bool,
        exception_type: str | None = None,
    ) -> None:
        clock = _clock_bracket()
        before_generation = self._generation
        if context is None:
            if before_generation == -1 and not self._metrics_calls_seen and not nested:
                reset_context = "metrics_processor_constructor"
                reset_step = None
                allowed = True
            else:
                reset_context = "unknown"
                reset_step = None
                allowed = False
        else:
            reset_context = context["context"]
            reset_step = context["step"]
            allowed = context["snapshot_record"] is not None and not nested
            if (
                context["snapshot_record"] is not None
                and context["snapshot_record"]["reset_generation"] != self._generation
            ):
                allowed = False
                self._mark_context_invalid(
                    context,
                    "native reset did not match the captured allocator generation",
                )
            if context["reset_record"] is not None:
                allowed = False
                self._mark_context_invalid(
                    context, "multiple resets occurred in one native metrics call"
                )
            if context["snapshot_record"] is None:
                self._mark_context_invalid(
                    context, "native reset occurred without its expected stats capture"
                )

        status = "complete" if allowed and exception_type is None else "invalid"
        reason_list: list[str] = []
        if not allowed:
            if reset_context == "unknown":
                reason_list.append(
                    "reset occurred outside a recognized native metrics context"
                )
            elif context is not None and context["snapshot_record"] is None:
                reason_list.append("native reset had no preceding stats capture")
            if context is not None:
                reason_list.extend(
                    reason
                    for reason in context["invalid_reasons"]
                    if reason not in reason_list
                )
            if nested:
                reason_list.append("nested allocator reset detected")
        if exception_type is not None:
            reason_list.append(f"native reset raised {exception_type}")

        self._generation += 1
        record = {
            "event_id": self._new_event_id(),
            "event_kind": "native_reset",
            "status": status,
            "context": reset_context,
            "step": reset_step,
            "previous_generation": before_generation,
            "reset_generation": self._generation,
            "reset_completed_clock": clock,
            "paired_snapshot_event_id": None,
            "reasons": reason_list,
        }
        if context is not None:
            context["reset_record"] = record
            snapshot = context["snapshot_record"]
            if snapshot is not None:
                snapshot["paired_reset_event_id"] = record["event_id"]
                if status != "complete":
                    self._mark_record_invalid(
                        snapshot, "; ".join(reason_list) or "paired reset was invalid"
                    )
                else:
                    record["paired_snapshot_event_id"] = snapshot["event_id"]
        self._records.append(record)
        for reason in reason_list:
            self._add_invalid(reason)
        if exception_type is not None:
            self._add_invalid(f"native reset raised {exception_type}")

        self._generation_valid = allowed and exception_type is None
        self._generation_has_snapshot = False
        self._last_reset = {
            "event_id": record["event_id"],
            "context": reset_context,
            "step": reset_step,
            "generation": self._generation,
            "clock": clock,
        }

    def _timing_scope(
        self,
        kind: str,
        timing_observer: Any,
        invoke: Callable[[], Any],
    ) -> Any:
        if kind in self._timing_anchors:
            self._add_invalid(f"timing observer emitted duplicate {kind} boundary")
            return invoke()
        original_time = self.timing_module.time
        if original_time is not self._timing_time_module:
            self._add_invalid(
                "timing observer time binding changed before phase anchor"
            )
            return invoke()
        anchored_time = _AnchoredTime(original_time, self, kind, timing_observer)
        self.timing_module.time = anchored_time
        try:
            return invoke()
        finally:
            if self.timing_module.time is anchored_time:
                self.timing_module.time = original_time
            else:
                self._add_invalid(
                    "timing observer time binding was replaced during boundary capture"
                )
            if not anchored_time.captured:
                self._add_invalid(
                    f"timing observer {kind} did not call its existing perf_counter"
                )

    def _record_timing_anchor(
        self,
        kind: str,
        timing_observer: Any,
        perf_counter_seconds: float,
        utc_before: int,
        utc_after: int,
        monotonic_before: int,
        monotonic_after: int,
    ) -> None:
        step = None
        if kind == "window_open":
            step = getattr(timing_observer, "config", None)
            step = getattr(step, "start_step", None)
        else:
            step = getattr(timing_observer, "_last_boundary_step", None)
        self._timing_anchors[kind] = {
            "step": step,
            "observer_perf_counter_seconds": perf_counter_seconds,
            "utc_ns_bracket": [utc_before, utc_after],
            "monotonic_ns_bracket": [monotonic_before, monotonic_after],
            "utc_bracket_width_ns": utc_after - utc_before,
            "monotonic_bracket_width_ns": monotonic_after - monotonic_before,
            "completed_close": kind == "completed_close",
        }

    def _capture_timing_memory(self, label: str, counters: Any) -> None:
        values: dict[str, int | None] = {}
        missing: list[str] = []
        for key in (
            "current_allocated",
            "current_reserved",
            "peak_allocated",
            "peak_reserved",
        ):
            value = (
                _python_int(counters.get(key))
                if isinstance(counters, Mapping)
                else None
            )
            values[key] = value
            if value is None:
                missing.append(key)
        self._timing_memory_samples[label] = {
            "source": "existing CampaignTimingObserver memory boundary; no extra device reads",
            "relationship_to_timing_boundary": (
                "before opening perf_counter boundary"
                if label == "opening"
                else "sampled after completed-close timer and before aggregation; copied after callback return"
            ),
            "reset_generation": self._generation,
            "reset_origin": self._reset_origin(),
            "counters_bytes": values,
            "missing_keys": missing,
            "status": "complete" if not missing else "invalid",
        }
        if missing:
            self._add_invalid(
                f"timing observer {label} memory sample was missing allocator counters"
            )

    def _phase_envelopes(self) -> list[dict[str, Any]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for record in self._records:
            if (
                record.get("event_kind") != "native_snapshot"
                or record.get("context") != "training"
            ):
                continue
            grouped.setdefault(record.get("phase_label", "unknown"), []).append(record)

        envelopes: list[dict[str, Any]] = []
        for label, records in grouped.items():
            allocated = [
                record["allocator_counters_bytes"].get("allocated_bytes.all.peak")
                for record in records
                if record.get("allocator_counters_bytes") is not None
            ]
            reserved = [
                record["allocator_counters_bytes"].get("reserved_bytes.all.peak")
                for record in records
                if record.get("allocator_counters_bytes") is not None
            ]
            allocated = [value for value in allocated if value is not None]
            reserved = [value for value in reserved if value is not None]
            valid = all(record["status"] == "complete" for record in records)
            envelopes.append(
                {
                    "phase_label": label,
                    "status": "complete" if valid else "invalid",
                    "snapshot_steps": [record["step"] for record in records],
                    "actual_windows": [record["actual_window"] for record in records],
                    "max_allocated_peak_bytes": max(allocated) if allocated else None,
                    "max_reserved_peak_bytes": max(reserved) if reserved else None,
                    "partial_values_only": not valid,
                }
            )
        return envelopes

    def _document(self, primary_exception: BaseException | None) -> dict[str, Any]:
        last_step = self._last_training_step
        expected_end = self.phase_boundaries["expected_end_step"]
        if primary_exception is not None:
            self._add_incomplete(
                f"native entry exited with {type(primary_exception).__name__}"
            )
        if expected_end is not None and (last_step is None or last_step < expected_end):
            self._add_incomplete(
                f"expected training metrics through step {expected_end}; last observed step was {last_step}"
            )
        if self._generation < 0:
            self._add_invalid("metrics processor constructor reset was not observed")

        if self._invalid_reasons:
            status = "invalid"
        elif self._incomplete_reasons:
            status = "incomplete"
        else:
            status = "complete"

        return {
            "schema_version": 1,
            "record_type": "phase_memory",
            "status": status,
            "run_id": self.run_id,
            "source_id": self.source_id,
            "config_id": self.config_id,
            "rank": self.rank,
            "local_rank": self.local_rank,
            "world_size": self.world_size,
            "device": self.device,
            "allocator_backend": self._allocator_backend,
            "allocator_backend_reason": self._allocator_backend_reason,
            "device_capacity_bytes": self._device_capacity_bytes,
            "device_samples": {
                "source": "external nvidia-smi CSV: timestamp,index,memory.used",
                "timestamp_timezone": "UTC",
                "expected_sample_period_ms": 500,
                "rank_device_join": None,
                "sample_count": None,
                "first_sample_utc": None,
                "last_sample_utc": None,
                "maximum_observed_used_bytes": None,
                "maximum_sample_gap_ms": None,
                "phase_exclusions": None,
                "boundary_exclusions": None,
                "status": "pending_external_join",
                "reason": "sampler data stays external and must be joined offline using verified device mapping and clock brackets",
            },
            "phase_boundaries": self.phase_boundaries,
            "training_log_frequency": self._training_log_frequency,
            "coverage": {
                "last_training_metrics_step": last_step,
                "expected_end_step": expected_end,
                "reset_generation_count": self._generation + 1,
                "status": status,
                "invalid_reasons": self._invalid_reasons,
                "incomplete_reasons": self._incomplete_reasons,
                "known_limits": [
                    "No pre-constructor allocator snapshot was taken; whole-process bootstrap peak is unavailable.",
                    "Native stats are copied after memory_stats returns; the copy bracket does not time the allocator API itself.",
                    "The interval between native stats capture and the native reset is not sampled.",
                    "Only the imported metrics.reset_peak_stats binding is wrapped; unobserved direct torch reset calls are not detected.",
                    "A008 updates 21-23 are replay, not a capture-only phase; steady labels use reset-aligned windows.",
                ],
            },
            "phase_envelopes": self._phase_envelopes(),
            "timing_observer": {
                "anchors": self._timing_anchors,
                "existing_memory_samples": self._timing_memory_samples,
                "memory_samples_are_separate_from_native_intervals": True,
            },
            "events": self._records,
        }

    def finish_and_flush(self, primary_exception: BaseException | None = None) -> None:
        if self._flushed:
            return
        if self._flush_attempted:
            if self._flush_error is not None:
                raise RuntimeError(
                    f"phase-memory flush previously failed: {self._flush_error}"
                )
            return
        self._flush_attempted = True
        document = self._document(primary_exception)
        try:
            with self.output_path.open("x", encoding="utf-8") as output:
                json.dump(document, output, indent=2, sort_keys=True)
                output.write("\n")
        except BaseException as exc:
            self._flush_error = f"{type(exc).__name__}: {exc}"
            raise
        self._flushed = True

    def restore(self) -> None:
        global _ACTIVE_OBSERVER
        if self._restore_done:
            return
        conflicts = [
            name
            for owner, name, _original, wrapper in self._bindings
            if getattr(owner, name, None) is not wrapper
        ]
        if conflicts:
            raise RuntimeError(
                "phase-memory refused to restore replaced bindings: "
                + ", ".join(conflicts)
            )
        for owner, name, original, _wrapper in reversed(self._bindings):
            setattr(owner, name, original)
        self._restore_done = True
        if _ACTIVE_OBSERVER is self:
            _ACTIVE_OBSERVER = None


def _validate_phase_boundaries(phase_boundaries: Mapping[str, int]) -> dict[str, int]:
    if not isinstance(phase_boundaries, Mapping):
        raise TypeError("phase_boundaries must be a mapping")
    normalized = dict(phase_boundaries)
    if set(normalized) != {"startup_end_step", "expected_end_step"}:
        raise ValueError(
            "phase_boundaries must contain startup_end_step and expected_end_step"
        )
    for name, value in normalized.items():
        if type(value) is not int or value < 1:
            raise ValueError(f"phase_boundaries[{name!r}] must be a positive integer")
    if normalized["startup_end_step"] >= normalized["expected_end_step"]:
        raise ValueError("startup_end_step must be before expected_end_step")
    return normalized


def _allocator_backend() -> tuple[str | None, str | None]:
    try:
        torch = importlib.import_module("torch")
        getter = getattr(torch.cuda, "get_allocator_backend", None)
        if not callable(getter):
            return None, "torch.cuda.get_allocator_backend is unavailable"
        backend = getter()
        if type(backend) is not str or not backend:
            return None, "torch.cuda.get_allocator_backend returned no string backend"
        return backend, None
    except Exception as exc:
        return None, f"allocator backend query failed: {type(exc).__name__}"


def install_phase_memory_observer(
    output_dir: str | os.PathLike[str],
    phase_boundaries: Mapping[str, int] = DEFAULT_PHASE_BOUNDARIES,
) -> PhaseMemoryObserver:
    """Install native-stat taps and return the lifecycle handle.

    The installer runs before native Trainer construction. It only wraps
    existing module bindings; each original is invoked once with unchanged
    arguments, result, exception, and call order.
    """
    global _ACTIVE_OBSERVER
    if _ACTIVE_OBSERVER is not None:
        raise RuntimeError(
            "only one phase-memory observer may be installed per process"
        )

    boundaries = _validate_phase_boundaries(phase_boundaries)
    utils_module = importlib.import_module("torchtitan.tools.utils")
    metrics_module = importlib.import_module("torchtitan.components.metrics")
    timing_module = importlib.import_module(
        "torchtitan.experiments.cuda_graph_campaign.observer"
    )
    backend, backend_reason = _allocator_backend()
    observer = PhaseMemoryObserver(
        output_dir,
        boundaries,
        utils_module=utils_module,
        timing_module=timing_module,
        allocator_backend=backend,
        allocator_backend_reason=backend_reason,
    )
    if backend_reason is not None:
        observer._add_invalid(backend_reason)

    if observer.world_size > 1 and (
        "RANK" not in os.environ
        or "LOCAL_RANK" not in os.environ
        or "WORLD_SIZE" not in os.environ
    ):
        raise RuntimeError(
            "phase-memory requires RANK, LOCAL_RANK, and WORLD_SIZE for distributed runs"
        )

    output_path = observer.output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing_rank_files = list(output_path.parent.glob("rank-*.json"))
    if existing_rank_files:
        raise FileExistsError(
            f"phase-memory output directory already contains rank records: {output_path.parent}"
        )

    metrics_type = getattr(metrics_module, "MetricsProcessor", None)
    if not isinstance(metrics_type, type):
        raise RuntimeError(
            "phase-memory expected torchtitan.components.metrics.MetricsProcessor"
        )
    original_log = _expect_signature(
        metrics_type,
        "log",
        (
            "self",
            "step",
            "global_avg_loss",
            "global_max_loss",
            "grad_norm",
            "extra_metrics",
        ),
    )
    original_validation = _expect_signature(
        metrics_type, "log_validation", ("self", "loss", "step", "extra_metrics")
    )
    original_reset = _expect_signature(metrics_module, "reset_peak_stats", ())
    original_usage = _expect_signature(
        utils_module, "device_memory_usage", ("memory_stats",)
    )
    if getattr(utils_module, "reset_peak_stats", None) is not original_reset:
        raise RuntimeError(
            "phase-memory metrics reset binding differs from torchtitan.tools.utils"
        )
    if not all(
        callable(getattr(timing_module.time, name, None))
        for name in ("perf_counter", "perf_counter_ns", "time_ns")
    ):
        raise RuntimeError(
            "phase-memory timing observer does not expose the expected clock API"
        )

    timing_type = getattr(timing_module, "CampaignTimingObserver", None)
    if not isinstance(timing_type, type):
        raise RuntimeError("phase-memory expected CampaignTimingObserver")
    original_capture = _expect_signature(
        timing_type, "_capture_boundary", ("self", "step")
    )
    original_finish = _expect_signature(
        timing_type, "_finish_window", ("self", "state", "expected_end_step")
    )

    @functools.wraps(original_usage)
    def wrapped_usage(*args: Any, **kwargs: Any) -> Any:
        memory_stats = args[0] if args else kwargs.get("memory_stats")
        context = observer._active_call
        if context is None:
            return original_usage(*args, **kwargs)
        if context["snapshot_record"] is not None:
            observer._mark_context_invalid(
                context, "multiple raw stats captures occurred in one metrics call"
            )
            return original_usage(*args, **kwargs)
        try:
            record = observer._capture_snapshot(memory_stats, context)
        except Exception as exc:
            observer._mark_context_invalid(
                context, f"phase-memory stats copy failed: {type(exc).__name__}"
            )
            return original_usage(*args, **kwargs)
        try:
            return original_usage(*args, **kwargs)
        except BaseException as exc:
            observer._mark_record_invalid(
                record, f"native device_memory_usage raised {type(exc).__name__}"
            )
            raise

    @functools.wraps(original_reset)
    def wrapped_reset(*args: Any, **kwargs: Any) -> Any:
        context = observer._active_call
        nested = observer._reset_depth > 0
        if nested:
            observer._add_invalid("nested allocator reset detected")
        observer._reset_depth += 1
        try:
            result = original_reset(*args, **kwargs)
        except BaseException as exc:
            try:
                observer._record_reset(
                    context, nested=nested, exception_type=type(exc).__name__
                )
            except Exception as record_error:
                observer._add_invalid(
                    f"reset failure could not be recorded: {type(record_error).__name__}"
                )
            raise
        else:
            try:
                observer._record_reset(context, nested=nested)
            except Exception as record_error:
                observer._add_invalid(
                    f"reset completion could not be recorded: {type(record_error).__name__}"
                )
            return result
        finally:
            observer._reset_depth -= 1

    @functools.wraps(original_log)
    def wrapped_log(owner: Any, *args: Any, **kwargs: Any) -> Any:
        return observer._invoke_metrics_method(
            original_log, "training", owner, args, kwargs, step_index=0
        )

    @functools.wraps(original_validation)
    def wrapped_validation(owner: Any, *args: Any, **kwargs: Any) -> Any:
        return observer._invoke_metrics_method(
            original_validation, "validation", owner, args, kwargs, step_index=1
        )

    @functools.wraps(original_capture)
    def wrapped_capture(owner: Any, *args: Any, **kwargs: Any) -> Any:
        step = _step_from_call(args, kwargs, 0)
        config = getattr(owner, "config", None)
        opening_step = _python_int(getattr(config, "start_step", None))
        if step == opening_step:
            try:
                return observer._timing_scope(
                    "window_open",
                    owner,
                    lambda: original_capture(owner, *args, **kwargs),
                )
            finally:
                if "window_open" in observer._timing_anchors:
                    memory = getattr(owner, "_memory_start", None)
                    if (
                        memory is not None
                        and "opening" not in observer._timing_memory_samples
                    ):
                        try:
                            observer._capture_timing_memory("opening", memory)
                        except Exception as exc:
                            observer._add_invalid(
                                f"opening memory sample copy failed: {type(exc).__name__}"
                            )
        return original_capture(owner, *args, **kwargs)

    @functools.wraps(original_finish)
    def wrapped_finish(owner: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return observer._timing_scope(
                "completed_close",
                owner,
                lambda: original_finish(owner, *args, **kwargs),
            )
        finally:
            if "completed_close" in observer._timing_anchors:
                memory = getattr(owner, "_memory_end", None)
                if (
                    memory is not None
                    and "completed_close" not in observer._timing_memory_samples
                ):
                    try:
                        observer._capture_timing_memory("completed_close", memory)
                    except Exception as exc:
                        observer._add_invalid(
                            f"closing memory sample copy failed: {type(exc).__name__}"
                        )

    patches = (
        (utils_module, "device_memory_usage", original_usage, wrapped_usage),
        (metrics_module, "reset_peak_stats", original_reset, wrapped_reset),
        (metrics_type, "log", original_log, wrapped_log),
        (metrics_type, "log_validation", original_validation, wrapped_validation),
        (timing_type, "_capture_boundary", original_capture, wrapped_capture),
        (timing_type, "_finish_window", original_finish, wrapped_finish),
    )
    try:
        for owner, name, original, wrapper in patches:
            if getattr(owner, name, None) is not original:
                raise RuntimeError(
                    f"phase-memory binding changed before install: {name}"
                )
            setattr(owner, name, wrapper)
            observer._bindings.append((owner, name, original, wrapper))
        _ACTIVE_OBSERVER = observer
    except BaseException:
        if observer._bindings:
            observer.restore()
        raise
    return observer
