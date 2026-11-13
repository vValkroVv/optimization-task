"""Opt-in local-tensor numerical receipts for native trainer diagnostics.

This observer copies and hashes tensors to CPU. Its synchronization and transfer
cost makes it unsuitable for throughput measurements; hashes establish exact
byte equality only when the compared native executions are expected to repeat.
The separate reference mode writes the selected update's local tensors one at a
time and compares them with 1 MiB CPU blocks. Its expected payload is about
47.515 GB per rank and capture; it adds no collectives and makes no TPS claim.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
from typing import Any


_MISSING = object()


class NativeNumericalReceipt:
    """Temporarily wrap native trainer hooks and write one rank-local receipt.

    The hook classes and distributed utility module are supplied by the caller,
    so importing this diagnostic does not import or build the training runtime.
    """

    def __init__(
        self,
        *,
        trainer_class: type,
        optimizer_container_class: type,
        callback_manager_class: type,
        dist_utils_module: Any,
        output_dir: str | Path,
        rank: int,
        capture_updates: tuple[int, ...] = (1, 2, 50),
        expected_microbatches: int = 32,
        reference_output: str | Path | None = None,
        reference_input: str | Path | None = None,
        capture_update: int | None = None,
        run_id: str = "native-numerics",
    ) -> None:
        reference_mode = reference_output is not None or reference_input is not None
        if reference_output is not None and reference_input is not None:
            raise ValueError("reference_output and reference_input are mutually exclusive")
        if reference_mode:
            if capture_update not in {1, 2, 50}:
                raise ValueError("reference mode requires capture_update 1, 2, or 50")
            updates = (capture_update,)
        else:
            if capture_update is not None:
                raise ValueError("capture_update is valid only in reference mode")
            updates = tuple(capture_updates)
        if not updates or any(type(step) is not int or step < 1 for step in updates):
            raise ValueError("capture_updates must contain positive integer steps")
        if len(set(updates)) != len(updates):
            raise ValueError("capture_updates must not contain duplicates")
        if type(expected_microbatches) is not int or expected_microbatches < 1:
            raise ValueError("expected_microbatches must be a positive integer")
        if type(rank) is not int or rank < 0:
            raise ValueError("rank must be a nonnegative integer")

        self.trainer_class = trainer_class
        self.optimizer_container_class = optimizer_container_class
        self.callback_manager_class = callback_manager_class
        self.dist_utils_module = dist_utils_module
        self.output_path = Path(output_dir) / f"native-numerics-rank{rank}.json"
        self.rank = rank
        self.capture_updates = tuple(sorted(updates))
        self.reference_mode = reference_mode
        self.capture_update = capture_update
        self.prefix_updates = tuple(range(1, capture_update + 1)) if reference_mode else self.capture_updates
        self.reference_output = Path(reference_output) if reference_output is not None else None
        self.reference_input = Path(reference_input) if reference_input is not None else None
        self.run_id = run_id
        self.expected_microbatches = expected_microbatches
        self._observations: list[dict[str, Any]] = []
        self._counts = {
            step: {
                "input_batches": 0,
                "microbatches": 0,
                "summary_calls": 0,
                "clip_calls": 0,
                "optimizer_calls": 0,
                "batch_end_calls": 0,
            }
            for step in self.capture_updates
        }
        self._pending = {
            step: {"losses": [], "global_valid_tokens": None}
            for step in self.capture_updates
        }
        self._patches: list[tuple[Any, str, Any, Any, Any]] = []
        self._trainer: Any = None
        self._active_trainer: Any = None
        self._active_update: int | None = None
        self._owners: dict[int, dict[str, Any]] = {}
        self._after_load_calls = 0
        self._fit_start_calls = 0
        self._forward_wrapper_depth = 0
        self._training_metadata: dict[str, Any] = {}
        self._prefix_counts = {step: 0 for step in self.prefix_updates}
        self._callback_context: tuple[Any, str, int] | None = None
        self._emitter_instance: Any = None
        self._monitor_emitter_calls = 0
        self._monitor_record_metadata: list[dict[str, Any]] = []
        self._monitor_publish_calls: list[dict[str, Any]] = []
        self._monitor_attestation: dict[str, Any] | None = (
            {
                "status": "not_applicable",
                "reason": "selected update is not the monitor50 update",
                "capture_update": capture_update,
            }
            if reference_mode and capture_update != 50
            else None
        )
        self._payload_session: Any = None
        self._entered = False

    def __enter__(self) -> NativeNumericalReceipt:
        if self._entered:
            raise RuntimeError("a numerical receipt cannot be entered twice")
        if self.output_path.exists():
            raise FileExistsError(f"numerical receipt already exists: {self.output_path}")
        self._entered = True
        try:
            if self.reference_mode:
                from numerics_payload import NativeNumericalPayload

                if self.reference_output is not None:
                    self._payload_session = NativeNumericalPayload(
                        mode="export",
                        root=self.reference_output,
                        rank=self.rank,
                        capture_update=self.capture_update,
                        run_id=self.run_id,
                    )
                else:
                    self._payload_session = NativeNumericalPayload(
                        mode="compare",
                        root=self.reference_input,
                        rank=self.rank,
                        capture_update=self.capture_update,
                        run_id=self.run_id,
                        comparison_root=self.output_path.parent / "native-numerical-comparison",
                    )
            self._patch_method(
                self.trainer_class, "forward_backward_step", self._wrap_forward_backward
            )
            self._patch_method(
                self.optimizer_container_class, "step", self._wrap_optimizer_step
            )
            self._patch_method(
                self.callback_manager_class, "emit", self._wrap_callback_emit
            )
            self._patch_module_function(
                self.dist_utils_module, "clip_grad_norm_", self._wrap_clip_grad_norm
            )
        except BaseException:
            self._restore_wrappers()
            raise
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        self._restore_wrappers()
        if exc_type is not None:
            if self._payload_session is not None:
                self._payload_session.record_failure(
                    f"{exc_type.__name__}: {exc}"
                )
            return False
        try:
            self._normalize_calibration_observations()
            self._validate_complete()
        except BaseException as validation_error:
            if self._payload_session is not None:
                self._payload_session.record_failure(
                    f"{type(validation_error).__name__}: {validation_error}"
                )
            raise
        reference_result = None
        if self._payload_session is not None:
            try:
                reference_result = self._payload_session.finish(
                    self._observations,
                    coverage=self._counts,
                    training_metadata=self._training_metadata,
                    monitor_attestation=self._monitor_attestation,
                )
            except BaseException as payload_error:
                self._payload_session.record_failure(
                    f"{type(payload_error).__name__}: {payload_error}"
                )
                raise
        document = {
            "schema": (
                "native-numerical-calibration-v1"
                if self.reference_mode
                else "native-numerical-receipt-v1"
            ),
            "complete": True,
            "rank": self.rank,
            "capture_updates": list(self.capture_updates),
            **({"prefix_updates": list(self.prefix_updates)} if self.reference_mode else {}),
            "expected_microbatches": self.expected_microbatches,
            "hash_semantics": (
                "SHA-256 of exact local tensor bytes in logical contiguous order; floating differences are separately measured in FP64 blocks."
                if self.reference_mode
                else "SHA-256 of exact local tensor bytes in logical contiguous order; a mismatch is not a tolerance comparison."
            ),
            "clip_norm_semantics": "The returned value is the native total gradient norm; a clip coefficient is not recorded.",
            "ema_state": (
                "EMA shadow payloads and continuation state are not captured; the active crop does not update EMA through step 50."
                if self.reference_mode
                else "EMA state is not captured; this receipt does not inspect actual EMA shadow buffers."
            ),
            "callback_state_scope": "batch_end callback state is sampled after handler emission; record_count may be zero after a flush and does not attest monitoring payloads.",
            "training_metadata": self._training_metadata,
            "positions_semantics": "after_dataloader hashes producer positions; Trainer.post_dataloading_process may replace them with derived arange positions for non-block-causal Decoder batches.",
            "microbatch_capture": "CPU after_dataloader inputs are hashed before GPU forward; forward_backward_step retains detached scalar losses, one denominator reference, and Python shape/dtype/chunk metadata only. Loss scalars, tensor denominators, loss components, and auxiliary totals are hashed once before clipping; Python int/float denominators retain their exact scalar type and value.",
            "scope": "Parameter, gradient, optimizer-state and update-summary copies to CPU synchronize and are diagnostic instrumentation, not TPS.",
            "coverage": self._counts,
            "observations": self._observations,
        }
        if reference_result is not None:
            document["reference"] = reference_result
            document["monitor_attestation"] = self._monitor_attestation
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.output_path.with_name(self.output_path.name + ".tmp")
        try:
            temporary.write_text(json.dumps(document, separators=(",", ":")) + "\n")
            os.replace(temporary, self.output_path)
        finally:
            if temporary.exists():
                temporary.unlink()
        return False

    def _patch_method(self, owner: type, name: str, factory) -> None:
        original = getattr(owner, name)
        local_original = owner.__dict__.get(name, _MISSING)
        wrapper = factory(original)
        setattr(owner, name, wrapper)
        self._patches.append((owner, name, local_original, None, wrapper))

    def _patch_module_function(self, owner: Any, name: str, factory) -> None:
        original = getattr(owner, name)
        wrapper = factory(original)
        setattr(owner, name, wrapper)
        self._patches.append((owner, name, original, original, wrapper))

    def _restore_wrappers(self) -> None:
        while self._patches:
            owner, name, original, module_function, wrapper = self._patches.pop()
            if module_function is not None:
                setattr(owner, name, module_function)
            elif (
                owner is self.optimizer_container_class
                and name == "step"
                and original is not _MISSING
                and not getattr(original, "hooked", False)
                and getattr(owner, name, None) is not wrapper
                and getattr(getattr(owner, name, None), "__wrapped__", None) is wrapper
                and getattr(getattr(owner, name, None), "hooked", False)
            ):
                # Optimizer.__init__ installs its normal step-hook wrapper the
                # first time a container is built. Keep that PyTorch wrapper,
                # but make it wrap the original method rather than this probe.
                optimizer_type = _torch().optim.Optimizer
                restored = optimizer_type.profile_hook_step(original)
                restored.hooked = True
                setattr(owner, name, restored)
            elif original is _MISSING:
                delattr(owner, name)
            else:
                setattr(owner, name, original)
        self._entered = False

    def _wrap_forward_backward(self, original):
        signature = inspect.signature(original)

        @functools.wraps(original)
        def wrapped(trainer, *args, **kwargs):
            bound = signature.bind(trainer, *args, **kwargs)
            bound.apply_defaults()
            update = int(trainer.step)
            self._active_trainer = trainer
            self._active_update = update
            self._forward_wrapper_depth += 1
            try:
                loss = original(trainer, *args, **kwargs)
                if update in self._counts:
                    if not _is_tensor(loss) or loss.numel() != 1:
                        raise TypeError("captured forward_backward_step loss must be a scalar tensor")
                    counts = self._counts[update]
                    microbatch = counts["microbatches"]
                    values = bound.arguments
                    pending = self._pending[update]
                    pending["losses"].append(loss.detach())
                    if pending["global_valid_tokens"] is None:
                        pending["global_valid_tokens"] = values["global_valid_tokens"]
                    inputs = {
                        name: _tensor_shape_dtype(value)
                        for name, value in sorted(values["input_dict"].items())
                    }
                    self._observations.append(
                        {
                            "update": update,
                            "phase": "microbatch_metadata",
                            "name": "trainer.forward_backward_step",
                            "microbatch": microbatch,
                            "loss": _tensor_shape_dtype(loss),
                            "global_valid_tokens": _scalar_or_tensor_shape_dtype(
                                values["global_valid_tokens"]
                            ),
                            "labels": _tensor_shape_dtype(values["labels"]),
                            "inputs": inputs,
                            "active_num_chunks": _json_value(
                                getattr(trainer.loss_fn, "num_chunks", None)
                            ),
                        }
                    )
                    counts["microbatches"] += 1
                return loss
            finally:
                self._forward_wrapper_depth -= 1

        return wrapped

    def _wrap_clip_grad_norm(self, original):
        signature = inspect.signature(original)

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            update = self._active_update
            trainer = self._active_trainer
            if update not in self._counts or trainer is None:
                return original(*args, **kwargs)

            parameters = bound.arguments["parameters"]
            if isinstance(parameters, _torch().Tensor):
                params = [parameters]
            elif isinstance(parameters, (list, tuple)):
                params = parameters
            else:
                raise TypeError(
                    "captured clip_grad_norm_ parameters must be a tensor, list, or tuple"
                )
            self._capture_update_summary(trainer, update)
            self._capture_router_state(trainer, update, "router_preclip")
            for param in params:
                owner_extra = self._parameter_owner_extra(param) if self.reference_mode else None
                self._observe_value(
                    getattr(param, "grad", None),
                    name=self._parameter_name(param) + ".grad",
                    phase="grad_preclip",
                    update=update,
                    extra=owner_extra,
                )
            total_norm = original(*args, **kwargs)
            for param in params:
                owner_extra = self._parameter_owner_extra(param) if self.reference_mode else None
                self._observe_value(
                    getattr(param, "grad", None),
                    name=self._parameter_name(param) + ".grad",
                    phase="grad_postclip",
                    update=update,
                    extra=owner_extra,
                )
            self._observe_value(
                total_norm,
                name="clip.total_norm",
                phase="clip_result",
                update=update,
            )
            self._counts[update]["clip_calls"] += 1
            return total_norm

        return wrapped

    def _capture_cpu_batch(self, state: Any, update: int) -> None:
        if self.reference_mode:
            microbatch = self._prefix_counts[update]
        else:
            microbatch = self._counts[update]["input_batches"]
        input_dict = getattr(state, "input_dict", None)
        labels = getattr(state, "labels", None)
        if not isinstance(input_dict, dict) or labels is None:
            raise ValueError("after_dataloader must return an input dictionary and labels")
        for name, value in sorted(input_dict.items()):
            self._observe_value(
                value,
                name=f"input.{name}",
                phase="dataloader_input",
                update=update,
                microbatch=microbatch,
                cpu_only=True,
            )
        self._observe_value(
            labels,
            name="labels",
            phase="dataloader_input",
            update=update,
            microbatch=microbatch,
            cpu_only=True,
        )
        if self.reference_mode:
            self._prefix_counts[update] += 1
            if update in self._counts:
                self._counts[update]["input_batches"] += 1
        else:
            self._counts[update]["input_batches"] += 1

    def _capture_update_summary(self, trainer: Any, update: int) -> None:
        counts = self._counts[update]
        if counts["summary_calls"] != 0:
            raise RuntimeError(f"update {update} captured its preclip summary more than once")
        for name in ("microbatches", "input_batches"):
            if counts[name] != self.expected_microbatches:
                raise RuntimeError(
                    f"update {update} captured {counts[name]} {name}; "
                    f"expected {self.expected_microbatches}"
                )
        pending = self._pending[update]
        if len(pending["losses"]) != self.expected_microbatches:
            raise RuntimeError(
                f"update {update} retained {len(pending['losses'])} scalar losses; "
                f"expected {self.expected_microbatches}"
            )
        if pending["global_valid_tokens"] is None:
            raise RuntimeError(f"update {update} has no global_valid_tokens reference")

        if self._payload_session is not None:
            self._normalize_calibration_observations()
            self._payload_session.verify_prefix_inputs(self._observations)

        self._capture_scalar_losses(pending["losses"], update)
        self._observe_denominator(
            pending["global_valid_tokens"],
            name="global_valid_tokens",
            phase="update_summary",
            update=update,
        )
        self._capture_loss_components(trainer, update, None)
        self._capture_aux_state(trainer, update, None)
        pending["losses"].clear()
        pending["global_valid_tokens"] = None
        counts["summary_calls"] += 1

    def _capture_scalar_losses(self, losses: list[Any], update: int) -> None:
        torch = _torch()
        local_losses = []
        tensor_metadata = []
        with torch.no_grad():
            for loss in losses:
                local, global_shape, mesh_metadata = _local_tensor_view(loss)
                if local.numel() != 1:
                    raise TypeError("captured microbatch losses must be scalar tensors")
                if self._payload_session is not None and self._payload_session.mode == "compare":
                    metadata = {
                        "update": update,
                        "phase": "microbatch_output",
                        "name": "loss.local_normalized",
                        "microbatch": len(local_losses),
                        "present": True,
                        "rank": self.rank,
                        "dtype": str(local.dtype),
                        "source_device_type": local.device.type,
                        "global_shape": list(global_shape),
                        "local_shape": list(local.shape),
                        "local_stride": list(local.stride()),
                        "local_numel": local.numel(),
                        **mesh_metadata,
                        "owner": None,
                        "aliases": [],
                    }
                    self._payload_session.validate_tensor_metadata(metadata)
                local_losses.append(local)
                tensor_metadata.append((local, global_shape, mesh_metadata))
        first = local_losses[0]
        if any(loss.dtype != first.dtype or loss.device != first.device for loss in local_losses):
            raise ValueError("captured microbatch losses must share dtype and device")

        # Copy all detached scalars to CPU together so the diagnostic introduces
        # one update-boundary device drain instead of one drain per microbatch.
        stacked = torch.stack(local_losses)
        cpu_losses = stacked.detach().to(
            device="cpu", copy=True, memory_format=torch.contiguous_format
        )
        if not cpu_losses.is_contiguous():
            cpu_losses = cpu_losses.contiguous()
        for microbatch, cpu_loss in enumerate(cpu_losses.unbind(0)):
            digest = hashlib.sha256(
                memoryview(cpu_loss.reshape(-1).view(torch.uint8).numpy())
            ).hexdigest()
            finite = (
                bool(torch.isfinite(cpu_loss).all().item())
                if cpu_loss.is_floating_point() or cpu_loss.is_complex()
                else None
            )
            local, global_shape, mesh_metadata = tensor_metadata[microbatch]
            record = {
                    "update": update,
                    "phase": "microbatch_output",
                    "name": "loss.local_normalized",
                    "microbatch": microbatch,
                    "present": True,
                    "rank": self.rank,
                    "dtype": str(local.dtype),
                    "global_shape": list(global_shape),
                    "local_shape": list(local.shape),
                    "local_stride": list(local.stride()),
                    "local_numel": local.numel(),
                    "source_device_type": local.device.type,
                    **mesh_metadata,
                    "sha256": digest,
                    "is_finite": finite,
                }
            if self.reference_mode:
                record.update({"owner": None, "aliases": []})
            self._observations.append(record)
            if self._payload_session is not None:
                self._payload_session.observe_tensor(record, cpu_loss)
        del cpu_losses, stacked, local_losses, tensor_metadata

    def _wrap_optimizer_step(self, original):
        signature = inspect.signature(original)

        @functools.wraps(original)
        def wrapped(container, *args, **kwargs):
            bound = signature.bind(container, *args, **kwargs)
            bound.apply_defaults()
            update = self._active_update
            trainer = self._active_trainer
            result = original(container, *args, **kwargs)
            if update in self._counts and trainer is not None:
                self._capture_optimizer_state(trainer, container, update)
                self._capture_router_state(trainer, update, "router_post_optimizer")
                self._counts[update]["optimizer_calls"] += 1
            return result

        return wrapped

    def _wrap_callback_emit(self, original):
        signature = inspect.signature(original)

        @functools.wraps(original)
        def wrapped(manager, *args, **kwargs):
            bound = signature.bind(manager, *args, **kwargs)
            bound.apply_defaults()
            if self.reference_mode:
                event = getattr(bound.arguments["event"], "value", bound.arguments["event"])
                trainer = bound.arguments["trainer"]
                prior_context = self._callback_context
                self._callback_context = (trainer, str(event), int(trainer.step))
                try:
                    result = original(manager, *args, **kwargs)
                finally:
                    self._callback_context = prior_context
            else:
                result = original(manager, *args, **kwargs)
                event = getattr(bound.arguments["event"], "value", bound.arguments["event"])
                trainer = bound.arguments["trainer"]
            if event == "after_load":
                self._after_load_calls += 1
                if self._after_load_calls != 1:
                    raise RuntimeError("expected one native after_load event")
                self._bind_trainer(trainer)
                self._capture_initial_state(trainer)
            elif event == "fit_start" and self.reference_mode:
                self._fit_start_calls += 1
                if self._fit_start_calls != 1:
                    raise RuntimeError("expected one native fit_start event")
                self._capture_fit_start_rng()
                self._normalize_calibration_observations()
                self._payload_session.verify_initial_gates(
                    self._observations, self._training_metadata
                )
            elif event == "after_dataloader":
                update = int(trainer.step)
                if update in self._counts or (
                    self.reference_mode and update in self.prefix_updates
                ):
                    self._capture_cpu_batch(result, update)
            elif event == "batch_end":
                update = int(trainer.step)
                if update in self._counts:
                    counts = self._counts[update]
                    counts["batch_end_calls"] += 1
                    self._require_update_counts(update)
                    self._capture_counter_state(trainer, update, "batch_end")
                    self._capture_scheduler_state(trainer, update)
                    self._capture_router_state(trainer, update, "router_batch_end")
                    self._capture_callback_state(result, trainer, event, update)
                if trainer is self._active_trainer:
                    self._active_trainer = None
                    self._active_update = None
            return result

        return wrapped

    def _bind_trainer(self, trainer: Any) -> None:
        if self._trainer is not None and self._trainer is not trainer:
            raise RuntimeError("one numerical receipt supports one trainer instance")
        self._trainer = trainer
        self._owners = self._build_ownership(trainer)
        if self.reference_mode:
            handler = getattr(getattr(trainer, "callbacks", None), "handler", None)
            emitter = getattr(handler, "_emitter", None)
            if emitter is None or not callable(getattr(emitter, "emit", None)):
                raise ValueError("reference mode requires CallbackManager.handler._emitter")
            self._emitter_instance = emitter
            self._patch_method(type(emitter), "emit", self._wrap_metric_emitter)

    def _wrap_metric_emitter(self, original):
        signature = inspect.signature(original)

        @functools.wraps(original)
        def wrapped(emitter, *args, **kwargs):
            context = self._callback_context
            if (
                emitter is not self._emitter_instance
                or context is None
                or context[1] != "batch_end"
                or context[2] != self.capture_update
                or self.capture_update != 50
            ):
                return original(emitter, *args, **kwargs)

            bound = signature.bind(emitter, *args, **kwargs)
            bound.apply_defaults()
            self._monitor_emitter_calls += 1
            records = bound.arguments["records"]
            if not isinstance(records, list):
                raise TypeError("MetricEmitter records must be a list")
            metadata = [
                self._metric_record_metadata(record, index, records)
                for index, record in enumerate(records)
            ]
            invalid = [
                item
                for item in metadata
                if item["reduce_mode"] != "online" or not item["scalar"]
            ]
            publish_calls: list[dict[str, Any]] = []
            publish = bound.arguments["publish"]

            def observed_publish(step, metrics):
                if not isinstance(metrics, dict):
                    raise TypeError("MetricEmitter publish mapping must be a dict")
                call_index = len(publish_calls)
                metric_rows = []
                for metric_index, (name, value) in enumerate(metrics.items()):
                    if not isinstance(name, str) or type(value) not in (int, float):
                        raise TypeError("MetricEmitter publishes string keys and numeric CPU values")
                    metric_name = str(name)
                    numeric_value = float(value)
                    resource = self._is_resource_metric(metric_name)
                    record = {
                        "update": self.capture_update,
                        "phase": "monitor_publish",
                        "name": f"publish[{call_index}].{metric_name}",
                        "microbatch": None,
                        "rank": self.rank,
                        "present": True,
                        "owner": None,
                        "aliases": [],
                        "publish_call_index": call_index,
                        "publish_metric_index": metric_index,
                        "publish_step": int(step),
                        "metric_name": metric_name,
                        "value": _json_value(numeric_value),
                        "resource_observation": resource,
                    }
                    metric_rows.append(record)
                publish_calls.append(
                    {
                        "index": call_index,
                        "step": int(step),
                        "metric_names": [row["metric_name"] for row in metric_rows],
                    }
                )
                for record in metric_rows:
                    self._observations.append(record)
                return publish(step, metrics)

            if "publish" not in kwargs:
                raise RuntimeError("MetricEmitter publish must be passed by its keyword-only argument")
            forwarded = dict(kwargs)
            forwarded["publish"] = observed_publish
            result = original(emitter, *args, **forwarded)
            record_counts: dict[str, int] = {}
            for item in metadata:
                record_counts[item["name"]] = record_counts.get(item["name"], 0) + 1
            for item in metadata:
                item["multiplicity"] = record_counts[item["name"]]
                record = {
                    **item,
                    "update": self.capture_update,
                    "phase": "monitor_record",
                    "name": f"record[{item['index']}].{item['name']}",
                    "record_name": item["name"],
                    "microbatch": None,
                    "rank": self.rank,
                    "present": True,
                    "owner": None,
                    "aliases": [],
                }
                self._observations.append(record)
            self._monitor_record_metadata = metadata
            self._monitor_publish_calls = publish_calls
            self._monitor_attestation = {
                "status": "captured",
                "event": "batch_end",
                "update": self.capture_update,
                "emitter_calls": self._monitor_emitter_calls,
                "record_count": len(metadata),
                "records": metadata,
                "publish_calls": publish_calls,
            }
            if invalid:
                raise RuntimeError(
                    "monitor50 contains offline or non-scalar records outside the attested crop"
                )
            if self._monitor_emitter_calls != 1:
                raise RuntimeError("monitor50 MetricEmitter.emit was called more than once")
            return result

        return wrapped

    @staticmethod
    def _metric_record_metadata(record: Any, index: int, records: list[Any]) -> dict[str, Any]:
        value = record.value
        if _is_tensor(value):
            value_metadata = {
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "scalar": value.numel() == 1,
            }
        elif type(value) in (int, float):
            value_metadata = {
                "dtype": f"python.{type(value).__name__}",
                "shape": [],
                "scalar": True,
            }
        else:
            value_metadata = {
                "dtype": f"python.{type(value).__name__}",
                "shape": None,
                "scalar": False,
            }
        reduce = record.reduce
        reduce_mode = getattr(reduce.mode, "value", reduce.mode)
        return {
            "index": index,
            "name": str(record.name),
            "step": int(record.step),
            "reduce_mode": str(reduce_mode),
            "reduce": {
                "mode": str(reduce_mode),
                "group": _json_value(reduce.group),
                "op": str(getattr(reduce.op, "value", reduce.op)),
                "stages": [
                    {
                        "group": _json_value(stage.group),
                        "op": str(getattr(stage.op, "value", stage.op)),
                    }
                    for stage in reduce.stages
                ],
            },
            "pending_stages": [
                {
                    "group": _json_value(stage.group),
                    "op": str(getattr(stage.op, "value", stage.op)),
                }
                for stage in record.pending_stages
            ],
            "post_stats": [
                {
                    "kind": str(getattr(step.kind, "value", step.kind)),
                    "stat": None
                    if step.stat is None
                    else {
                        "op": step.stat.op,
                        "dim": _json_value(step.stat.dim),
                        "q": _json_value(step.stat.q),
                        "p": _json_value(step.stat.p),
                        "minlength": _json_value(step.stat.minlength),
                        "threshold": _json_value(step.stat.threshold),
                        "cmp": str(getattr(step.stat.cmp, "value", step.stat.cmp)),
                    },
                    "reduce": None
                    if step.reduce is None
                    else {
                        "group": _json_value(step.reduce.group),
                        "op": str(getattr(step.reduce.op, "value", step.reduce.op)),
                    },
                }
                for step in record.post_stats
            ],
            "post_axes": list(getattr(record.post_axes, "names", ())),
            **value_metadata,
        }

    @staticmethod
    def _is_resource_metric(name: str) -> bool:
        lowered = name.lower()
        return any(
            token in lowered
            for token in ("time", "tps", "throughput", "remaining", "memory", "mem")
        )

    def _capture_fit_start_rng(self) -> None:
        import random
        import struct

        import numpy as np

        torch = _torch()
        python_state = random.getstate()
        python_digest = hashlib.sha256(repr(python_state).encode("utf-8")).hexdigest()
        numpy_state = np.random.get_state()
        numpy_hash = hashlib.sha256()
        numpy_hash.update(str(numpy_state[0]).encode("utf-8"))
        numpy_hash.update(memoryview(numpy_state[1]).cast("B"))
        numpy_hash.update(struct.pack("=q?d", int(numpy_state[2]), bool(numpy_state[3]), float(numpy_state[4])))
        rng_rows = [
            ("python.random", python_digest, {"state_type": "random.getstate"}),
            (
                "numpy.random",
                numpy_hash.hexdigest(),
                {"algorithm": str(numpy_state[0]), "state_shape": list(numpy_state[1].shape)},
            ),
        ]
        torch_cpu = torch.get_rng_state()
        rng_rows.append(("torch.cpu", self._tensor_bytes_digest(torch_cpu), {"device": "cpu"}))
        device = torch.cuda.current_device()
        torch_cuda = torch.cuda.get_rng_state(device=device)
        rng_rows.append(
            (
                "torch.cuda.current_rank",
                self._tensor_bytes_digest(torch_cuda),
                {"device_index": int(device), "device_type": "cuda"},
            )
        )
        for name, digest, metadata in rng_rows:
            self._observations.append(
                {
                    "update": 0,
                    "phase": "fit_start_rng",
                    "name": name,
                    "microbatch": None,
                    "rank": self.rank,
                    "present": True,
                    "owner": None,
                    "aliases": [],
                    "dtype": "sha256",
                    "sha256": digest,
                    "metadata": metadata,
                }
            )
        self._training_metadata["deterministic_backend_flags"] = {
            "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
            "deterministic_warn_only": bool(torch.is_deterministic_algorithms_warn_only_enabled()),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        }

    @staticmethod
    def _tensor_bytes_digest(value: Any) -> str:
        torch = _torch()
        cpu = value.detach().to(device="cpu", copy=True, memory_format=torch.contiguous_format)
        return hashlib.sha256(memoryview(cpu.reshape(-1).view(torch.uint8).numpy())).hexdigest()

    def _build_ownership(self, trainer: Any) -> dict[int, dict[str, Any]]:
        names: dict[int, dict[str, Any]] = {}
        for part_index, model in enumerate(trainer.model_parts):
            for fqn, param in model.named_parameters(remove_duplicate=False):
                entry = names.get(id(param))
                if entry is None:
                    entry = {"part_index": part_index, "parameter": param, "names": []}
                    names[id(param)] = entry
                elif entry["part_index"] != part_index:
                    raise ValueError("a parameter is ambiguously owned by multiple model parts")
                entry["names"].append(fqn)

        group_owners: dict[int, dict[str, Any]] = {}
        optimizers = getattr(trainer.optimizers, "optimizers", None)
        if not optimizers:
            raise ValueError("native optimizer container has no inner optimizers")
        for optimizer_index, optimizer in enumerate(optimizers):
            for group_index, group in enumerate(optimizer.param_groups):
                for param_index, param in enumerate(group["params"]):
                    identity = id(param)
                    if identity not in names:
                        raise ValueError("optimizer parameter has no model FQN owner")
                    if identity in group_owners:
                        raise ValueError("parameter has ambiguous optimizer-group ownership")
                    group_owners[identity] = {
                        "optimizer_index": optimizer_index,
                        "group_index": group_index,
                        "param_index": param_index,
                    }

        owners: dict[int, dict[str, Any]] = {}
        for identity, entry in names.items():
            group = group_owners.get(identity)
            if entry["parameter"].requires_grad and group is None:
                raise ValueError("trainable model parameter is absent from optimizer groups")
            names_for_param = sorted(set(entry["names"]))
            if not names_for_param:
                raise ValueError("model parameter has no FQN")
            owners[identity] = {
                "part_index": entry["part_index"],
                "canonical_fqn": names_for_param[0],
                "aliases": names_for_param[1:],
                "group": group,
            }
        return owners

    def _parameter_name(self, param: Any) -> str:
        owner = self._owners.get(id(param))
        if owner is None:
            raise ValueError("captured tensor has no model FQN ownership")
        return f"model.{owner['part_index']}.{owner['canonical_fqn']}"

    def _parameter_owner_extra(self, param: Any) -> dict[str, Any]:
        owner = self._owners.get(id(param))
        if owner is None:
            raise ValueError("captured tensor has no model FQN ownership")
        return {
            "owner": {
                "part_index": owner["part_index"],
                "canonical_fqn": owner["canonical_fqn"],
                "optimizer_group": owner["group"],
            },
            "aliases": list(owner["aliases"]),
        }

    def _capture_initial_state(self, trainer: Any) -> None:
        update = int(trainer.step)
        weights = getattr(trainer.loss_fn, "loss_weights", None)
        self._training_metadata = {
            "initial_step": update,
            "loss_fn_type": f"{type(trainer.loss_fn).__module__}.{type(trainer.loss_fn).__qualname__}",
            "mtp_loss_weights": None if weights is None else _json_value(weights),
            "initial_num_chunks": _json_value(
                getattr(trainer.loss_fn, "num_chunks", None)
            ),
        }
        seen: set[int] = set()
        with _torch().no_grad():
            for part_index, model in enumerate(trainer.model_parts):
                for _, param in model.named_parameters(remove_duplicate=False):
                    identity = id(param)
                    if identity in seen:
                        continue
                    seen.add(identity)
                    owner = self._owners[identity]
                    name = f"model.{part_index}.{owner['canonical_fqn']}"
                    self._observe_value(param, name=name, phase="initial_parameter", update=update)
                    if owner["aliases"]:
                        self._observations[-1]["aliases"] = list(owner["aliases"])
                    self._observations[-1]["optimizer_owner"] = owner["group"]
                    if self.reference_mode:
                        self._observations[-1]["owner"] = {
                            "part_index": owner["part_index"],
                            "canonical_fqn": owner["canonical_fqn"],
                            "optimizer_group": owner["group"],
                        }
                        self._observations[-1]["aliases"] = list(owner["aliases"])
        self._capture_counter_state(trainer, update, "after_load")
        self._capture_router_state(trainer, update, "router_after_load")
        if self.reference_mode:
            self._capture_optimizer_inventory(trainer, update)
            self._capture_scheduler_state(trainer, update, phase="scheduler_after_load")
            self._capture_initial_buffers(trainer, update)
            owners = []
            for owner in self._owners.values():
                owners.append(
                    {
                        "part_index": owner["part_index"],
                        "canonical_fqn": owner["canonical_fqn"],
                        "aliases": list(owner["aliases"]),
                        "optimizer_group": owner["group"],
                    }
                )
            self._training_metadata["ownership"] = sorted(
                owners, key=lambda item: (item["part_index"], item["canonical_fqn"])
            )
            self._training_metadata["resolved_config_sha256"] = _config_fingerprint(
                trainer.config
            )
            self._training_metadata["config_normalizations"] = {
                "training.steps": "selected diagnostic stop update",
                "dump_folder": "run-owned output path",
            }

    def _capture_optimizer_inventory(self, trainer: Any, update: int) -> None:
        optimizers = getattr(trainer.optimizers, "optimizers", None)
        if not optimizers:
            raise ValueError("native optimizer container has no inner optimizers")
        with _torch().no_grad():
            for optimizer_index, optimizer in enumerate(optimizers):
                for group_index, group in enumerate(optimizer.param_groups):
                    parameters = [self._parameter_name(param) for param in group["params"]]
                    self._observations.append(
                        {
                            "update": update,
                            "phase": "optimizer_initial_group",
                            "name": f"optimizer.{optimizer_index}.group.{group_index}",
                            "optimizer_class": f"{type(optimizer).__module__}.{type(optimizer).__qualname__}",
                            "hyperparameters": {
                                key: _json_value(value)
                                for key, value in sorted(group.items())
                                if key != "params"
                            },
                            "parameters": parameters,
                        }
                    )
                    for param in group["params"]:
                        name = self._parameter_name(param)
                        owner_extra = self._parameter_owner_extra(param)
                        state = optimizer.state.get(param, {})
                        keys = sorted(state)
                        self._observations.append(
                            {
                                "update": update,
                                "phase": "optimizer_initial_state_keys",
                                "name": name,
                                "keys": keys,
                                **owner_extra,
                            }
                        )
                        for key in keys:
                            value = state[key]
                            state_name = f"{name}.optimizer.{key}"
                            if _is_tensor(value):
                                self._observe_value(
                                    value,
                                    name=state_name,
                                    phase="optimizer_initial_state",
                                    update=update,
                                    extra=owner_extra,
                                )
                            else:
                                self._observations.append(
                                    {
                                        "update": update,
                                        "phase": "optimizer_initial_state_value",
                                        "name": state_name,
                                        "value": _json_value(value),
                                        **owner_extra,
                                    }
                                )

    def _capture_initial_buffers(self, trainer: Any, update: int) -> None:
        for part_index, model in enumerate(trainer.model_parts):
            buffers: dict[int, dict[str, Any]] = {}
            for fqn, buffer in model.named_buffers(remove_duplicate=False):
                if buffer is None:
                    continue
                entry = buffers.get(id(buffer))
                if entry is None:
                    buffers[id(buffer)] = {"buffer": buffer, "names": [fqn]}
                else:
                    entry["names"].append(fqn)
            for entry in buffers.values():
                names = sorted(set(entry["names"]))
                self._observe_value(
                    entry["buffer"],
                    name=f"model.{part_index}.buffer.{names[0]}",
                    phase="initial_buffer",
                    update=update,
                    extra={
                        "owner": {"part_index": part_index, "canonical_fqn": names[0]},
                        "aliases": names[1:],
                    },
                )

    def _capture_loss_components(
        self, trainer: Any, update: int, microbatch: int | None
    ) -> None:
        components = trainer.loss_fn.components()
        if not isinstance(components, dict):
            raise TypeError("loss_fn.components() must return a dictionary")
        for name, value in sorted(components.items()):
            self._observe_value(
                value,
                name=f"loss.components.{name}",
                phase="loss_components",
                update=update,
                microbatch=microbatch,
            )

    def _capture_aux_state(
        self, trainer: Any, update: int, microbatch: int | None
    ) -> None:
        tracker = getattr(trainer, "aux_loss", None)
        if tracker is None:
            return
        for index, state in enumerate(getattr(tracker, "_states", ())):
            names = list(getattr(state, "names", ()))
            for field in ("totals", "scale"):
                value = getattr(state, field, None)
                if value is not None:
                    self._observe_value(
                        value,
                        name=f"aux_loss.states.{index}.{field}",
                        phase="aux_loss_state",
                        update=update,
                        microbatch=microbatch,
                        extra={"names": names} if field == "totals" else None,
                    )

    def _capture_router_state(self, trainer: Any, update: int, phase: str) -> None:
        with _torch().no_grad():
            for part_index, model in enumerate(trainer.model_parts):
                for module_fqn, module in model.named_modules(remove_duplicate=False):
                    counts = getattr(module, "tokens_per_expert_E", None)
                    if counts is None:
                        continue
                    prefix = f"model.{part_index}.{module_fqn or '<root>'}"
                    self._observe_value(
                        counts,
                        name=f"{prefix}.tokens_per_expert_E",
                        phase=phase,
                        update=update,
                        extra=(
                            {"owner": {"part_index": part_index, "module_fqn": module_fqn}, "aliases": []}
                            if self.reference_mode
                            else None
                        ),
                    )
                    bias = getattr(module, "expert_bias_E", None)
                    self._observe_value(
                        bias,
                        name=f"{prefix}.expert_bias_E",
                        phase=phase,
                        update=update,
                        extra=(
                            {"owner": {"part_index": part_index, "module_fqn": module_fqn}, "aliases": []}
                            if self.reference_mode
                            else None
                        ),
                    )

    def _capture_optimizer_state(self, trainer: Any, container: Any, update: int) -> None:
        with _torch().no_grad():
            for optimizer_index, optimizer in enumerate(container.optimizers):
                for group_index, group in enumerate(optimizer.param_groups):
                    hparams = {
                        key: _json_value(value)
                        for key, value in sorted(group.items())
                        if key != "params"
                    }
                    self._observations.append(
                        {
                            "update": update,
                            "phase": "optimizer_group",
                            "name": f"optimizer.{optimizer_index}.group.{group_index}",
                            "optimizer_class": f"{type(optimizer).__module__}.{type(optimizer).__qualname__}",
                            "hyperparameters": hparams,
                        }
                    )
                    for param in group["params"]:
                        name = self._parameter_name(param)
                        self._observe_value(
                            param,
                            name=name,
                            phase="post_optimizer_parameter",
                            update=update,
                            extra=(self._parameter_owner_extra(param) if self.reference_mode else None),
                        )
                        state = optimizer.state.get(param, {})
                        state_keys = sorted(state)
                        self._observations.append(
                            {
                                "update": update,
                                "phase": "optimizer_state_keys",
                                "name": name,
                                "keys": state_keys,
                                **(self._parameter_owner_extra(param) if self.reference_mode else {}),
                            }
                        )
                        for key in state_keys:
                            value = state[key]
                            if _is_tensor(value):
                                self._observe_value(
                                    value,
                                    name=f"{name}.optimizer.{key}",
                                    phase="optimizer_state",
                                    update=update,
                                    extra=(self._parameter_owner_extra(param) if self.reference_mode else None),
                                )
                            else:
                                self._observations.append(
                                    {
                                        "update": update,
                                        "phase": "optimizer_state_value",
                                        "name": f"{name}.optimizer.{key}",
                                        "value": _json_value(value),
                                        **(self._parameter_owner_extra(param) if self.reference_mode else {}),
                                    }
                                )

    def _capture_counter_state(self, trainer: Any, update: int, phase: str) -> None:
        self._observations.append(
            {"update": update, "phase": phase, "name": "trainer.step", "value": int(trainer.step)}
        )
        self._observe_value(
            getattr(trainer, "ntokens_seen", None),
            name="trainer.ntokens_seen",
            phase=phase,
            update=update,
        )

    def _capture_scheduler_state(
        self, trainer: Any, update: int, *, phase: str = "scheduler_state"
    ) -> None:
        schedulers = getattr(trainer, "lr_schedulers", None)
        if schedulers is None or not hasattr(schedulers, "state_dict"):
            raise ValueError("trainer has no scheduler state_dict()")
        self._observations.append(
            {
                "update": update,
                "phase": phase,
                "name": "trainer.lr_schedulers",
                "value": _json_value(schedulers.state_dict()),
            }
        )

    def _capture_callback_state(
        self, state: Any, trainer: Any, event: str, update: int
    ) -> None:
        self._observations.append(
            {
                "update": update,
                "phase": "callback_state",
                "name": "callbacks.batch_end",
                "event": getattr(getattr(state, "event", event), "value", event),
                "step": int(getattr(state, "step", trainer.step)),
                "microbatch_idx": getattr(state, "microbatch_idx", None),
                "record_count": len(getattr(state, "_records", ())),
            }
        )
        for field in ("loss", "grad_norm", "global_valid_tokens"):
            value = getattr(state, field, None)
            observe = (
                self._observe_denominator
                if field == "global_valid_tokens"
                else self._observe_value
            )
            observe(
                value,
                name=f"callbacks.batch_end.{field}",
                phase="callback_state",
                update=update,
            )
        metrics = getattr(state, "metrics", None)
        if metrics is not None:
            if not isinstance(metrics, dict):
                raise TypeError("callback metrics state must be a dictionary")
            for name, value in sorted(metrics.items()):
                if value is None or _is_tensor(value):
                    self._observe_value(
                        value,
                        name=f"callbacks.batch_end.metrics.{name}",
                        phase="callback_state",
                        update=update,
                    )
                else:
                    self._observations.append(
                        {
                            "update": update,
                            "phase": "callback_state_value",
                            "name": f"callbacks.batch_end.metrics.{name}",
                            "value": _json_value(value),
                        }
                    )

    def _require_update_counts(self, update: int) -> None:
        counts = self._counts[update]
        if counts["batch_end_calls"] != 1:
            raise RuntimeError(f"update {update} emitted batch_end more than once")
        expected = {
            "input_batches": self.expected_microbatches,
            "microbatches": self.expected_microbatches,
            "summary_calls": 1,
            "clip_calls": 1,
            "optimizer_calls": 1,
        }
        for name, count in expected.items():
            if counts[name] != count:
                raise RuntimeError(
                    f"update {update} captured {counts[name]} {name}; expected {count}"
                )

    def _validate_complete(self) -> None:
        if self._after_load_calls != 1 or self._trainer is None:
            raise RuntimeError("numerical receipt did not observe exactly one after_load event")
        if self.reference_mode and self._fit_start_calls != 1:
            raise RuntimeError("reference mode did not observe exactly one native fit_start event")
        for update in self.capture_updates:
            self._require_update_counts(update)
        if self.reference_mode:
            for update in self.prefix_updates:
                count = self._prefix_counts[update]
                if count != self.expected_microbatches:
                    raise RuntimeError(
                        f"update {update} captured {count} CPU input batches; "
                        f"expected {self.expected_microbatches}"
                    )
            if self.capture_update == 50 and (
                self._monitor_attestation is None
                or self._monitor_attestation.get("status") != "captured"
                or self._monitor_attestation.get("emitter_calls") != 1
                or self._monitor_attestation.get("record_count", 0) == 0
                or not self._monitor_attestation.get("publish_calls")
            ):
                raise RuntimeError("monitor50 actual emitter publication was not attested")

    def _normalize_calibration_observations(self) -> None:
        if not self.reference_mode:
            return
        for record in self._observations:
            record.setdefault("rank", self.rank)
            record.setdefault("microbatch", None)
            record.setdefault("owner", None)
            record.setdefault("aliases", [])
            record.setdefault("present", True)

    def _observe_value(
        self,
        value: Any,
        *,
        name: str,
        phase: str,
        update: int,
        microbatch: int | None = None,
        extra: dict[str, Any] | None = None,
        cpu_only: bool = False,
    ) -> None:
        if self._forward_wrapper_depth:
            raise RuntimeError(
                "tensor hashing inside forward_backward_step is forbidden; "
                "defer scalar hashes to the preclip boundary"
            )
        record: dict[str, Any] = {
            "update": update,
            "phase": phase,
            "name": name,
            "microbatch": microbatch,
        }
        if value is None:
            record.update(
                {"present": False, "dtype": None, "shape": None, "sha256": None, "is_finite": None}
            )
            cpu_tensor = None
        else:
            if (
                self._payload_session is not None
                and self._payload_session.mode == "compare"
                and update == self.capture_update
                and phase != "dataloader_input"
            ):
                preflight_metadata = _tensor_metadata(
                    value, self.rank, require_cpu=cpu_only
                )
                preflight_record = {**record, **preflight_metadata, "present": True}
                if extra:
                    preflight_record.update(extra)
                preflight_record.setdefault("owner", None)
                preflight_record.setdefault("aliases", [])
                self._payload_session.validate_tensor_metadata(preflight_record)
            if self.reference_mode:
                metadata, digest, finite, cpu_tensor = _tensor_snapshot(
                    value, self.rank, require_cpu=cpu_only
                )
            else:
                metadata, digest, finite = _tensor_digest(
                    value, self.rank, require_cpu=cpu_only
                )
                cpu_tensor = None
            record.update(metadata)
            record.update({"present": True, "sha256": digest, "is_finite": finite})
        if extra:
            record.update(extra)
        if self.reference_mode:
            record.setdefault("rank", self.rank)
            record.setdefault("owner", None)
            record.setdefault("aliases", [])
        self._observations.append(record)
        if cpu_tensor is not None and self._payload_session is not None:
            self._payload_session.observe_tensor(record, cpu_tensor)
            del cpu_tensor

    def _observe_denominator(
        self,
        value: Any,
        *,
        name: str,
        phase: str,
        update: int,
    ) -> None:
        if _is_tensor(value) or value is None:
            self._observe_value(
                value,
                name=name,
                phase=phase,
                update=update,
            )
            return
        if type(value) not in (int, float):
            raise TypeError(
                "global_valid_tokens must be a supported tensor, Python int, or Python float"
            )
        if self._forward_wrapper_depth:
            raise RuntimeError("denominator capture inside forward_backward_step is forbidden")
        record: dict[str, Any] = {
            "update": update,
            "phase": phase,
            "name": name,
            "microbatch": None,
            "present": True,
            "python_type": type(value).__name__,
            "dtype": f"python.{type(value).__name__}",
            "shape": [],
            "global_shape": [],
            "local_shape": [],
            "local_stride": [],
            "local_numel": 1,
            "source_device_type": None,
            "value": _json_value(value),
            "sha256": None,
            "is_finite": math.isfinite(value) if type(value) is float else None,
        }
        self._observations.append(record)


def _torch():
    import torch

    return torch


def _tensor_types():
    torch = _torch()
    from torch.distributed.tensor import DTensor

    try:
        from torchtitan.distributed.gfsdp import RaggedShardDTensor
    except ImportError:
        RaggedShardDTensor = ()
    return torch, DTensor, RaggedShardDTensor


def _is_tensor(value: Any) -> bool:
    torch, DTensor, RaggedShardDTensor = _tensor_types()
    return (
        type(value) in (torch.Tensor, torch.nn.Parameter, DTensor)
        or (RaggedShardDTensor and type(value) is RaggedShardDTensor)
    )


def _tensor_shape_dtype(value: Any) -> dict[str, Any]:
    if not _is_tensor(value):
        raise TypeError(f"unsupported tensor metadata type: {type(value)!r}")
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device_type": value.device.type,
    }


def _scalar_or_tensor_shape_dtype(value: Any) -> dict[str, Any]:
    if _is_tensor(value):
        return _tensor_shape_dtype(value)
    if type(value) in (int, float):
        return {
            "shape": [],
            "dtype": f"python.{type(value).__name__}",
            "device_type": None,
            "python_type": type(value).__name__,
            "value": _json_value(value),
        }
    raise TypeError(
        "global_valid_tokens must be a supported tensor, Python int, or Python float"
    )


def _local_tensor_view(
    value: Any, *, require_cpu: bool = False
) -> tuple[Any, tuple[int, ...], dict[str, Any]]:
    torch, DTensor, RaggedShardDTensor = _tensor_types()
    with torch.no_grad():
        if require_cpu and (
            type(value) not in (torch.Tensor, torch.nn.Parameter)
            or value.device.type != "cpu"
        ):
            raise RuntimeError(
                "after_dataloader numerical receipt requires plain CPU tensors"
            )
        if type(value) in (torch.Tensor, torch.nn.Parameter):
            local = value
            global_shape = tuple(value.shape)
            mesh_metadata = {"mesh": None, "placements": []}
        elif type(value) is DTensor or (
            RaggedShardDTensor and type(value) is RaggedShardDTensor
        ):
            local = value.to_local()
            global_shape = tuple(value.shape)
            mesh_metadata = _mesh_metadata(value)
        else:
            raise TypeError(f"unsupported tensor type for native receipt: {type(value)!r}")

        if type(local) not in (torch.Tensor, torch.nn.Parameter):
            raise TypeError(f"unsupported local tensor type for native receipt: {type(local)!r}")
        if local.layout is not torch.strided:
            raise TypeError(f"native receipt requires strided tensors, got {local.layout}")
        return local, global_shape, mesh_metadata


def _tensor_digest(
    value: Any, rank: int, *, require_cpu: bool = False
) -> tuple[dict[str, Any], str, bool | None]:
    torch = _torch()
    with torch.no_grad():
        local, global_shape, mesh_metadata = _local_tensor_view(
            value, require_cpu=require_cpu
        )
        source_device_type = local.device.type
        source = local.detach()
        cpu = source.to(device="cpu", copy=True, memory_format=torch.contiguous_format)
        if not cpu.is_contiguous():
            cpu = cpu.contiguous()
        raw = memoryview(cpu.reshape(-1).view(torch.uint8).numpy())
        digest = hashlib.sha256(raw).hexdigest()
        finite = (
            bool(torch.isfinite(cpu).all().item())
            if cpu.is_floating_point() or cpu.is_complex()
            else None
        )
        metadata = {
            "rank": rank,
            "dtype": str(local.dtype),
            "source_device_type": source_device_type,
            "global_shape": list(global_shape),
            "local_shape": list(local.shape),
            "local_stride": list(local.stride()),
            "local_numel": local.numel(),
            **mesh_metadata,
        }
        del raw, cpu, source, local
        return metadata, digest, finite


def _tensor_snapshot(
    value: Any, rank: int, *, require_cpu: bool = False
) -> tuple[dict[str, Any], str, bool | None, Any]:
    """Copy one local tensor to contiguous CPU storage once for payload mode."""
    torch = _torch()
    with torch.no_grad():
        local, global_shape, mesh_metadata = _local_tensor_view(
            value, require_cpu=require_cpu
        )
        source_device_type = local.device.type
        cpu = local.detach().to(
            device="cpu", copy=True, memory_format=torch.contiguous_format
        )
        if not cpu.is_contiguous():
            cpu = cpu.contiguous()
        raw = memoryview(cpu.reshape(-1).view(torch.uint8).numpy())
        digest = hashlib.sha256(raw).hexdigest()
        if cpu.is_floating_point() or cpu.is_complex():
            from numerics_payload import BLOCK_BYTES

            finite = _bounded_finite(cpu, BLOCK_BYTES)
        else:
            finite = None
        metadata = {
            "rank": rank,
            "dtype": str(local.dtype),
            "source_device_type": source_device_type,
            "global_shape": list(global_shape),
            "local_shape": list(local.shape),
            "local_stride": list(local.stride()),
            "local_numel": local.numel(),
            **mesh_metadata,
        }
        del raw, local
        return metadata, digest, finite, cpu


def _bounded_finite(value: Any, block_bytes: int) -> bool:
    """Scan finiteness in bounded CPU blocks, including float8 dtypes."""
    torch = _torch()
    flat = value.reshape(-1)
    block_elements = max(1, block_bytes // value.element_size())
    float8 = str(value.dtype).startswith("torch.float8_")
    for start in range(0, flat.numel(), block_elements):
        block = flat.narrow(0, start, min(block_elements, flat.numel() - start))
        if float8:
            block = block.to(torch.float32)
        if not bool(torch.isfinite(block).all().item()):
            return False
    return True


def _tensor_metadata(
    value: Any, rank: int, *, require_cpu: bool = False
) -> dict[str, Any]:
    """Read only local tensor metadata; do not materialize or transfer values."""
    with _torch().no_grad():
        local, global_shape, mesh_metadata = _local_tensor_view(
            value, require_cpu=require_cpu
        )
        return {
            "rank": rank,
            "dtype": str(local.dtype),
            "source_device_type": local.device.type,
            "global_shape": list(global_shape),
            "local_shape": list(local.shape),
            "local_stride": list(local.stride()),
            "local_numel": local.numel(),
            **mesh_metadata,
        }


def _mesh_metadata(value: Any) -> dict[str, Any]:
    mesh = value.device_mesh
    coordinate = mesh.get_coordinate()
    if coordinate is not None:
        coordinate = list(coordinate)
    placements = []
    for placement in value.placements:
        item = {"type": type(placement).__name__, "value": str(placement)}
        for field in ("local_numel", "global_offset", "shard_numels"):
            if hasattr(placement, field):
                field_value = getattr(placement, field)
                item[field] = list(field_value) if isinstance(field_value, tuple) else field_value
        placements.append(item)
    return {
        "mesh": {
            "device_type": mesh.device_type,
            "shape": list(mesh.shape),
            "axes": list(mesh.mesh_dim_names or ()),
            "coordinate": coordinate,
        },
        "placements": placements,
    }


def _json_value(value: Any) -> Any:
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            return {"float": repr(value)}
        return value
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    raise TypeError(f"unsupported JSON value in native receipt: {type(value)!r}")


def _config_fingerprint(config: Any) -> str:
    """Hash the resolved trainer config while normalizing only run stop/output."""
    from dataclasses import fields, is_dataclass
    from enum import Enum

    def encode(value: Any, path: tuple[str, ...] = ()) -> Any:
        if path == ("training", "steps"):
            return "<selected-capture-update>"
        if path == ("dump_folder",):
            return "<run-owned-output>"
        if value is None or type(value) in (str, bool, int, float):
            return _json_value(value)
        if isinstance(value, Enum):
            return encode(value.value, path)
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, type):
            return f"{value.__module__}.{value.__qualname__}"
        if is_dataclass(value) and not isinstance(value, type):
            return {
                item.name: encode(getattr(value, item.name), path + (item.name,))
                for item in fields(value)
            }
        if isinstance(value, dict):
            return {
                str(key): encode(item, path + (str(key),))
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            }
        if isinstance(value, (tuple, list)):
            return [encode(item, path + (str(index),)) for index, item in enumerate(value)]
        if isinstance(value, (set, frozenset)):
            return sorted(encode(item, path) for item in value)
        if callable(value):
            return f"{value.__module__}.{value.__qualname__}"
        raise TypeError(f"unsupported resolved config value: {type(value)!r} at {path}")

    encoded = json.dumps(encode(config), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
