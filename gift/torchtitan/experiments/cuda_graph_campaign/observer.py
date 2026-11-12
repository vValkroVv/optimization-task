"""Opt-in, rank-local timing and activation records for the CUDA Graph campaign."""

import importlib.metadata
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any, TypeAlias

import torch
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointWrapper,
)

from torchtitan.callbacks.base import Observer, ObserverState
from torchtitan.config import Configurable
from torchtitan.config.schema.experiments.cuda_graph_campaign.observer import (
    CampaignTimingObserverConfig,
)


def _qualified_type(value: Any) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _short_inventory(entries: list[dict[str, str]]) -> dict[str, Any]:
    types = sorted({entry["type"] for entry in entries})
    return {
        "count": len(entries),
        "types": types,
        "examples": entries[:8],
    }


class CampaignTimingObserver(Configurable, Observer):
    """Measures one complete loop window without per-update device syncs.

    The observer uses ``batch_start`` as both boundaries. It records the
    already-global valid-token scalar at ``batch_end`` and never reduces token
    counts across ranks. Timing events are recorded on the current CUDA stream;
    their spans are not device-compute measurements.
    """

    Config: TypeAlias = CampaignTimingObserverConfig

    def __init__(self, config: Config) -> None:
        self.config = config
        self._activation: dict[str, Any] | None = None
        self._output_path: Path | None = None
        self._written = False
        self._started = False
        self._closed = False
        self._errors: list[str] = []
        self._token_steps: list[int] = []
        self._global_tokens_per_update: list[int] = []
        self._host_boundaries: list[float] = []
        self._host_spans: list[float] = []
        self._event_pairs: list[tuple[Any, Any]] = []
        self._last_event: Any | None = None
        self._last_boundary_time: float | None = None
        self._last_boundary_step: int | None = None
        self._memory_start: dict[str, int] | None = None
        self._memory_end: dict[str, int] | None = None

    def fit_start(self, state: ObserverState) -> None:
        self._validate_timing_device(state.device)
        self._validate_distributed_context(state.parallel_dims)
        self._output_path = self._resolve_output_path(state)
        if self._output_path.exists():
            raise FileExistsError(
                f"campaign timing output already exists: {self._output_path}"
            )
        self._activation = self._build_activation_record(state)

    def batch_start(self, state: ObserverState) -> None:
        if self._closed:
            return

        step = state.step
        start_step = self.config.start_step
        end_boundary_step = start_step + self.config.update_count

        if not self._started:
            if step < start_step:
                return
            if step != start_step:
                self._errors.append(
                    f"window start boundary expected step {start_step}, got {step}"
                )
                return
            self._synchronize_device(state.device)
            self._memory_start = self._memory_boundary(state.device)
            self._align_ranks()
            self._capture_boundary(step)
            self._started = True
            return

        expected_step = (self._last_boundary_step or start_step) + 1
        if step != expected_step:
            self._errors.append(
                f"consecutive batch_start boundary expected step {expected_step}, got {step}"
            )

        self._capture_boundary(step)
        if step >= end_boundary_step:
            self._finish_window(state, end_boundary_step)

    def batch_end(self, state: ObserverState) -> None:
        if not self._started or self._closed:
            return
        if not (
            self.config.start_step
            <= state.step
            < self.config.start_step + self.config.update_count
        ):
            return

        if self._token_steps and state.step <= self._token_steps[-1]:
            self._errors.append(
                f"duplicate or out-of-order batch_end at step {state.step}"
            )
            return

        value = state.global_valid_tokens
        if type(value) not in (int, float) or isinstance(value, bool):
            self._errors.append(
                f"step {state.step} global_valid_tokens is {type(value).__name__}, "
                "not the native Python scalar; no tensor value read was attempted"
            )
            return

        scalar = float(value)
        if not math.isfinite(scalar) or scalar < 0 or not scalar.is_integer():
            self._errors.append(
                f"step {state.step} global_valid_tokens is not a finite nonnegative integer"
            )
            return

        tokens = int(scalar)
        expected_tokens = self._activation["training_contract"][
            "expected_global_valid_tokens_per_update"
        ]
        if tokens != expected_tokens:
            self._errors.append(
                f"step {state.step} global_valid_tokens was {tokens}, "
                f"expected {expected_tokens} for the fixed-length batch contract"
            )
        self._token_steps.append(state.step)
        self._global_tokens_per_update.append(tokens)

    def fit_end(self, state: ObserverState) -> None:
        if self._written:
            return
        if self._started and not self._closed:
            reason = (
                "the following batch_start boundary was not reached; the window "
                "is incomplete and cannot produce throughput"
            )
        elif not self._started:
            reason = f"configured start step {self.config.start_step} was not reached"
        else:
            return
        self._persist(self._incomplete_record(state.step, reason))
        raise RuntimeError(f"campaign timing window is incomplete: {reason}")

    def close(self) -> None:
        if self._written or self._activation is None:
            return
        if self._started and not self._closed:
            reason = "training closed before the following batch_start boundary"
        else:
            reason = "training closed before a complete timing window"
        self._persist(self._incomplete_record(self._last_boundary_step, reason))

    def _capture_boundary(self, step: int) -> None:
        timestamp = time.perf_counter()
        event = self._new_cuda_event()
        self._record_cuda_event(event, self._device)
        if self._last_boundary_time is not None and self._last_event is not None:
            self._host_spans.append(timestamp - self._last_boundary_time)
            self._event_pairs.append((self._last_event, event))
        self._host_boundaries.append(timestamp)
        self._last_boundary_time = timestamp
        self._last_event = event
        self._last_boundary_step = step

    def _finish_window(self, state: ObserverState, expected_end_step: int) -> None:
        self._closed = True

        expected_steps = list(range(self.config.start_step, expected_end_step))
        if self._token_steps != expected_steps:
            self._errors.append(
                "batch_end token records did not cover every measured update"
            )
        if len(self._host_spans) != self.config.update_count:
            self._errors.append(
                f"expected {self.config.update_count} host spans, got {len(self._host_spans)}"
            )
        if len(self._event_pairs) != self.config.update_count:
            self._errors.append(
                f"expected {self.config.update_count} CUDA event spans, "
                f"got {len(self._event_pairs)}"
            )

        self._synchronize_device(state.device)
        completed_stop_time = time.perf_counter()
        self._memory_end = self._memory_boundary(state.device)

        local_submission_elapsed = sum(self._host_spans)
        local_completed_elapsed = completed_stop_time - self._host_boundaries[0]
        slowest_elapsed, remote_invalid = self._aggregate_window(
            local_completed_elapsed,
            bool(self._errors),
            state.device,
        )
        cuda_spans = [
            start.elapsed_time(end) / 1000.0 for start, end in self._event_pairs
        ]
        tokens_total = sum(self._global_tokens_per_update)
        invalid = bool(self._errors) or remote_invalid
        window = {
            "schema_version": 1,
            "record_type": "timing_window",
            "status": "invalid" if invalid else "complete",
            "valid": not invalid,
            "rank": self._activation["rank"],
            "start_step": self.config.start_step,
            "end_boundary_step": expected_end_step,
            "observed_end_boundary_step": self._last_boundary_step,
            "update_count": self.config.update_count,
            "host_submission_boundary_offsets_seconds": [
                value - self._host_boundaries[0] for value in self._host_boundaries
            ],
            "host_update_submission_spans_seconds": self._host_spans,
            "cuda_current_stream_boundary_spans_seconds": cuda_spans,
            "global_valid_tokens_per_update_this_rank": self._global_tokens_per_update,
            "global_valid_tokens_total_this_rank": tokens_total,
            "expected_global_valid_tokens_per_update": self._activation[
                "training_contract"
            ]["expected_global_valid_tokens_per_update"],
            "local_host_submission_elapsed_seconds": local_submission_elapsed,
            "local_completed_window_elapsed_seconds": local_completed_elapsed,
            "slowest_rank_completed_window_elapsed_seconds": slowest_elapsed,
            "global_valid_tokens_per_second": (
                tokens_total / slowest_elapsed
                if not invalid and slowest_elapsed > 0
                else None
            ),
            "slowest_rank_aggregation": "MAX over the initialized world group; local on a single process",
            "token_accounting": (
                "sum of this rank's already-global Python valid-token scalar per update; "
                "token totals are not reduced across ranks"
            ),
            "memory_boundary_bytes": {
                "start": self._memory_start,
                "end": self._memory_end,
                "peak_scope": (
                    "native PyTorch allocator high-water counters sampled at each boundary; "
                    "not reset by this observer and not a device/NVML peak"
                ),
            },
            "timing_scope": {
                "host_submission": (
                    "CPU clock spans between successive batch_start callbacks, including "
                    "native post-update metrics/checkpoint/validation cadence; the final "
                    "submission span ends before the closing device synchronization"
                ),
                "completed_window": (
                    "CPU wall time from the opening batch_start timestamp after the "
                    "pre-window device drain and rank barrier to the stop timestamp "
                    "after the closing device drain; includes queued device tail but "
                    "excludes the final elapsed/validity MAX aggregation and persistence"
                ),
                "cuda_events": (
                    "asynchronous event spans recorded on the current CUDA stream; "
                    "not device-compute time and may omit independent side-stream work"
                ),
            },
            "failure_reasons": self._errors,
            "another_rank_reported_invalid": remote_invalid and not self._errors,
        }
        self._persist(window)
        if invalid:
            raise RuntimeError(
                "campaign timing window failed validation; its JSONL record is "
                "marked invalid and has no throughput value"
            )

    def _aggregate_window(
        self, local_completed_elapsed: float, local_invalid: bool, device: Any
    ) -> tuple[float, bool]:
        """Reduce completed window elapsed time and invalid status across ranks."""
        payload = torch.tensor(
            [local_completed_elapsed, float(local_invalid)],
            dtype=torch.float64,
            device=device,
        )
        distributed = dist.is_available() and dist.is_initialized()
        if distributed and dist.get_world_size() > 1:
            dist.all_reduce(payload, op=dist.ReduceOp.MAX)
        slowest_elapsed, any_invalid = payload.tolist()
        return float(slowest_elapsed), bool(any_invalid)

    def _build_activation_record(self, state: ObserverState) -> dict[str, Any]:
        config = state.config
        training = config.training
        parallel_dims = state.parallel_dims
        configured_global_batch = int(training.global_batch_size)
        effective_global_batch = configured_global_batch
        if effective_global_batch < 0:
            batch_degree = int(getattr(parallel_dims, "dp_replicate", 1)) * int(
                getattr(parallel_dims, "dp_shard", 1)
            )
            effective_global_batch = (
                int(training.local_batch_size)
                * batch_degree
                * int(state.gradient_accumulation_steps)
            )

        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        world_size = (
            dist.get_world_size()
            if dist.is_available() and dist.is_initialized()
            else int(getattr(parallel_dims, "world_size", 1))
        )
        self._device = state.device
        source_modules = {
            name: sys.modules.get(name) for name in ("torchtitan", "torchtitan.trainer")
        }

        module_markers = self._module_markers(state.model_parts)
        loss_type = (
            f"{state.loss_fn_type.__module__}.{state.loss_fn_type.__qualname__}"
            if state.loss_fn_type is not None
            else None
        )
        dataloader = config.dataloader
        data_config = {
            "type": f"{type(dataloader).__module__}.{type(dataloader).__qualname__}",
            "dataset": getattr(dataloader, "dataset", None),
        }
        hf_text_full_length_contract = (
            type(dataloader).__name__ == "HuggingFaceTextDataLoaderConfig"
            and getattr(dataloader, "dataset", None) == "c4_test"
        )

        process_groups = {}
        if hasattr(parallel_dims, "get_all_one_dimensional_meshes"):
            for name, mesh in parallel_dims.get_all_one_dimensional_meshes().items():
                process_groups[name] = {
                    "degree": int(mesh.size()),
                    "global_ranks": [
                        int(rank_value) for rank_value in mesh.mesh.flatten().tolist()
                    ],
                }
        # The inventory above omits size-one meshes, including the real expert
        # FSDP group retained by this EP8 preset. This query reuses that mesh.
        if int(getattr(parallel_dims, "ep", 1)) > 1 and hasattr(
            parallel_dims, "get_optional_mesh"
        ):
            expert_fsdp_mesh = parallel_dims.get_optional_mesh("efsdp")
            if expert_fsdp_mesh is not None:
                process_groups["efsdp"] = {
                    "degree": int(expert_fsdp_mesh.size()),
                    "global_ranks": [
                        int(rank_value)
                        for rank_value in expert_fsdp_mesh.mesh.flatten().tolist()
                    ],
                }
        world_mesh = getattr(parallel_dims, "world_mesh", None)
        if world_mesh is not None:
            process_groups["world"] = {
                "degree": int(world_mesh.size()),
                "global_ranks": [
                    int(rank_value) for rank_value in world_mesh.mesh.flatten().tolist()
                ],
            }

        requested = {
            "compile": {
                "enabled": bool(config.compile.enable),
                "components": list(config.compile.components),
                "backend": config.compile.backend,
            },
            "activation_checkpoint": {"mode": config.activation_checkpoint.mode},
            "fp8": {
                "dense_include_linear_layers": config.fp8.dense_include_linear_layers,
                "dense_use_shared_quantizer": config.fp8.dense_use_shared_quantizer,
                "sparse_use_grouped_gemm": config.fp8.sparse_use_grouped_gemm,
            },
        }

        return {
            "schema_version": 1,
            "record_type": "activation",
            "rank": rank,
            "world_size": world_size,
            "source": {
                "observer_file": str(Path(__file__).resolve()),
                "torchtitan_package_file": getattr(
                    source_modules["torchtitan"], "__file__", None
                ),
                "trainer_file": getattr(
                    source_modules["torchtitan.trainer"], "__file__", None
                ),
            },
            "runtime": self._runtime_versions(state.device),
            "training_contract": {
                "data": data_config,
                "hf_text_c4_test_fixed_length_label_path": hf_text_full_length_contract,
                "label_contract_source": (
                    "torchtitan.datasets.hf.text_datasets.HuggingFaceTextDataset.__iter__"
                    if hf_text_full_length_contract
                    else None
                ),
                "local_batch_size": int(training.local_batch_size),
                "configured_global_batch_size": configured_global_batch,
                "effective_global_batch_size": effective_global_batch,
                "gradient_accumulation_steps": int(state.gradient_accumulation_steps),
                "sequence_length": int(training.seq_len),
                "expected_global_valid_tokens_per_update": (
                    effective_global_batch * int(training.seq_len)
                ),
                "runtime_check": (
                    "batch_end's already-global Python scalar must equal the full "
                    "fixed-length expected token count for every update"
                ),
            },
            "parallelism": {
                "world_size": int(getattr(parallel_dims, "world_size", world_size)),
                "degrees": {
                    name: int(getattr(parallel_dims, name))
                    for name in (
                        "pp",
                        "dp_replicate",
                        "dp_shard",
                        "cp",
                        "tp",
                        "ep",
                    )
                    if hasattr(parallel_dims, name)
                },
                "process_groups": process_groups,
            },
            "requested_config": requested,
            "constructed_markers": module_markers,
            "loss": {
                "constructed_type": loss_type,
                "compiled_callable": "unavailable through public ObserverState",
                "compile_execution_proof": "pending native marker/trace evidence",
            },
            "runtime_execution_proof": {
                "status": "pending",
                "cuda_graph_replay": "not observed by this callback",
                "fp8_custom_op_execution": "not inferred from constructed module presence",
            },
        }

    @staticmethod
    def _module_markers(model_parts: tuple[torch.nn.Module, ...]) -> dict[str, Any]:
        compiled = []
        checkpoint_wrappers = []
        dense_fp8 = []
        grouped_fp8 = []
        shared_quantizers = []
        dispatchers = []

        for part_index, part in enumerate(model_parts):
            for name, module in part.named_modules():
                path = f"model_parts.{part_index}" + (f".{name}" if name else "")
                module_type = _qualified_type(module)
                entry = {"path": path, "type": module_type}
                if getattr(module, "_compiled_call_impl", None) is not None:
                    compiled.append(entry)
                if isinstance(module, CheckpointWrapper):
                    checkpoint_wrappers.append(entry)
                if type(module).__name__ == "FP8Linear" and ".float8." in module_type:
                    dense_fp8.append(entry)
                if getattr(module, "_use_float8_grouped_gemm", False) is True:
                    grouped_fp8.append(entry)
                if getattr(module, "use_shared_quantizer", False) is True:
                    shared_quantizers.append(entry)
                dispatcher = getattr(module, "token_dispatcher", None)
                if (
                    dispatcher is not None
                    and type(dispatcher).__name__ == "HybridEPTokenDispatcher"
                    and ".hybridep" in _qualified_type(dispatcher)
                ):
                    capacity_config = getattr(dispatcher, "_capacity_config", None)
                    dispatchers.append(
                        {
                            "path": path + ".token_dispatcher",
                            "type": _qualified_type(dispatcher),
                            "comm_stream": getattr(
                                capacity_config, "comm_stream", None
                            ),
                        }
                    )

        hybridep_module = sys.modules.get("torchtitan.distributed.deepep.hybridep")
        global_comm_stream = (
            getattr(hybridep_module, "_COMM_STREAM_ENABLED", None)
            if hybridep_module is not None
            else None
        )
        return {
            "compiled_model_callables_installed": _short_inventory(compiled),
            "activation_checkpoint_wrappers": _short_inventory(checkpoint_wrappers),
            "dense_fp8_linear_modules": _short_inventory(dense_fp8),
            "sparse_fp8_grouped_expert_modules": _short_inventory(grouped_fp8),
            "dense_shared_quantizer_modules": _short_inventory(shared_quantizers),
            "hybridep_dispatchers": {
                "count": len(dispatchers),
                "examples": dispatchers[:8],
                "global_comm_stream_enabled": global_comm_stream,
            },
        }

    @staticmethod
    def _runtime_versions(device: Any) -> dict[str, Any]:
        versions: dict[str, Any] = {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "cuda_build": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(device),
        }
        try:
            versions["nccl"] = torch.cuda.nccl.version()
        except Exception as exc:  # noqa: BLE001 - report unavailable runtime metadata.
            versions["nccl"] = f"unavailable: {type(exc).__name__}"
        for package in ("triton", "deep_ep", "deep_gemm"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        return versions

    def _incomplete_record(
        self, observed_step: int | None, reason: str
    ) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "record_type": "timing_window",
            "status": "incomplete",
            "valid": False,
            "rank": self._activation["rank"] if self._activation else None,
            "start_step": self.config.start_step,
            "expected_end_boundary_step": self.config.start_step
            + self.config.update_count,
            "observed_step": observed_step,
            "update_count": self.config.update_count,
            "global_valid_tokens_per_second": None,
            "failure_reasons": [reason, *self._errors],
        }

    def _persist(self, window: dict[str, Any]) -> None:
        if self._written:
            return
        if self._activation is None or self._output_path is None:
            raise RuntimeError("campaign timing observer persisted before fit_start")
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        with self._output_path.open("x", encoding="utf-8") as output:
            output.write(json.dumps(self._activation, sort_keys=True, allow_nan=False))
            output.write("\n")
            output.write(json.dumps(window, sort_keys=True, allow_nan=False))
            output.write("\n")
        self._written = True

    def _resolve_output_path(self, state: ObserverState) -> Path:
        output_dir = Path(self.config.output_dir)
        if not output_dir.is_absolute():
            output_dir = Path(state.config.dump_folder) / output_dir
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        return output_dir / f"rank-{rank:05d}.jsonl"

    @staticmethod
    def _validate_timing_device(device: Any) -> None:
        if torch.device(device).type != "cuda":
            raise ValueError(
                "campaign timing observer requires a CUDA device; CPU timing "
                "doubles are supported only in unit tests"
            )

    @staticmethod
    def _validate_distributed_context(parallel_dims: Any) -> None:
        expected_world_size = int(getattr(parallel_dims, "world_size", 1))
        if not dist.is_available() or not dist.is_initialized():
            if expected_world_size > 1:
                raise RuntimeError(
                    "campaign timing observer needs an initialized distributed "
                    "process group for a multi-rank run"
                )
            return

        actual_world_size = dist.get_world_size()
        if actual_world_size != expected_world_size:
            raise RuntimeError(
                "campaign timing observer process-group world size does not match "
                f"ParallelDims: {actual_world_size} != {expected_world_size}"
            )

    def _align_ranks(self) -> None:
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            dist.barrier()

    @staticmethod
    def _new_cuda_event():
        return torch.cuda.Event(enable_timing=True)

    @staticmethod
    def _record_cuda_event(event: Any, device: Any) -> None:
        event.record(torch.cuda.current_stream(device))

    @staticmethod
    def _synchronize_device(device: Any) -> None:
        torch.cuda.synchronize(device)

    @staticmethod
    def _memory_boundary(device: Any) -> dict[str, int]:
        return {
            "current_allocated": int(torch.cuda.memory_allocated(device)),
            "current_reserved": int(torch.cuda.memory_reserved(device)),
            "peak_allocated": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved": int(torch.cuda.max_memory_reserved(device)),
        }

    @property
    def _device(self) -> Any:
        return self.__device

    @_device.setter
    def _device(self, value: Any) -> None:
        self.__device = value
