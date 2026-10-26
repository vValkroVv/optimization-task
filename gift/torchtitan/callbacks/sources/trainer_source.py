from collections import deque
import re
import time

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Shard

from typing import TypeAlias

from torchtitan.config.schema.callbacks.sources.trainer_source import (
    TrainerSourceConfig,
)
from torchtitan.callbacks.base import ObserverState, OptimizerGroupView
from torchtitan.callbacks.metrics.sample import Sample, MetricRecord
from torchtitan.config.schema.callbacks.metrics.sample import (
    MetricConfig,
    ReduceStageConfig,
    ReduceOp,
    ReduceConfig,
)
from torchtitan.callbacks.sources.base import (
    MonitorSource,
    canonical_fqn,
    iter_logical_weights,
)
from torchtitan.components.loss import IGNORE_INDEX, MTPLoss
from torchtitan.distributed.pipeline_parallel import in_metadata_probe
from torchtitan.tools.logging import logger

_SECONDS_PER_HOUR = 3600.0


class TrainerSource(MonitorSource):
    """Logs training-loop progress counters, batch geometry, and token accuracy.

    Built when the preset lists it:
    ``sources={"trainer": TrainerSource.Config(...)}`` in the monitoring
    callback's config — there is no enable flag.

    Logged metrics:
        - ``time_seconds/batch/batch_total``: wall-clock of the whole train step
        - ``time/batch``: global step
        - ``time/sample``: step * local_batch_size * gradient_accumulation_steps
          * data_parallel_size (assumes a fixed batch geometry)
        - ``trainer/global_valid_tokens``: supervised label tokens in this step
          (labels != IGNORE_INDEX, already reduced by the trainer)
        - ``trainer/device_train_microbatch_size``, ``trainer/global_train_batch_size``,
          ``trainer/gradient_accumulation_steps``, ``trainer/num_gpus``: logged once
        - ``token_accuracy/top{k}``: fraction of supervised tokens whose true
          next token is within the model's top-k logits, for each configured k
    """

    # pyrefly: ignore [bad-override]
    Config: TypeAlias = TrainerSourceConfig

    config: Config

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self._estimate_start_time: float | None = None
        self._estimate_start_step: int | None = None
        self._last_loaded_step = 0
        window = config.throughput_window_size + 1
        self._history_steps: deque[int] = deque(maxlen=window)
        self._history_samples: deque[int] = deque(maxlen=window)
        self._history_tokens: deque[int] = deque(maxlen=window)
        self._history_wct: deque[float] = deque(maxlen=window)
        self._speed_start_time = 0.0
        self._prior_train_seconds = 0.0
        self._ks = sorted(set(config.topk_accuracy))
        self._logged_batch_config = False
        self._batch_start_time = 0.0
        self._logits_hooked = False
        self._param_names_by_id: dict[int, list[str]] | None = None
        self._warned_lr_conflicts = False
        self._topk_chunks: list[torch.Tensor] = []
        # Device-side [hits_k1..hits_kn, valid_tokens] accumulator: .item()
        # per microbatch would host-sync inside the training step.
        self._counts: torch.Tensor | None = None

    def fit_start(self, state: ObserverState) -> None:
        self._last_loaded_step = int(state.step)
        self._estimate_start_time = None
        self._estimate_start_step = None
        if self.config.log_throughput:
            self._speed_start_time = time.perf_counter()
            for history in (
                self._history_steps,
                self._history_samples,
                self._history_tokens,
                self._history_wct,
            ):
                history.clear()
            self._append_snapshot(state)
        if not self._ks:
            return
        self.close()
        pattern = re.compile(self.config.logits_module_pattern)
        self._logits_hooked = False
        for model_part in state.model_parts:
            for module_name, module in model_part.named_modules():
                if module_name and pattern.search(canonical_fqn(module_name)):
                    self._handles.append(
                        module.register_forward_hook(self._logits_hook)
                    )
                    self._logits_hooked = True
        if not self._logits_hooked and not state.parallel_dims.pp_enabled:
            raise ValueError(
                f"TrainerSource: topk_accuracy is configured but no module "
                f"matched logits_module_pattern={self.config.logits_module_pattern!r}."
            )

    def batch_start(self, state: ObserverState) -> None:
        super().batch_start(state)
        if (
            self.config.log_remaining_estimate
            and self._estimate_start_time is None
            and state.step - self._last_loaded_step
            > self.config.remaining_estimate_skip_steps
        ):
            self._estimate_start_time = time.perf_counter()
            self._estimate_start_step = int(state.step)
        self._batch_start_time = time.perf_counter()
        self._topk_chunks.clear()
        self._counts = None

    def _logits_hook(self, module: nn.Module, inputs: tuple, output: object) -> None:
        if in_metadata_probe():
            return
        del module, inputs
        if not self.active or torch.compiler.is_compiling() or self._is_recompute():
            return
        if isinstance(output, (torch.Tensor, DTensor)):
            with torch.no_grad():
                self._topk_chunks.append(self._global_topk_ids(output))

    def _global_topk_ids(self, logits: torch.Tensor) -> torch.Tensor:
        """Top-k vocab ids per position; merges vocab shards under loss parallel."""
        kmax = self._ks[-1]
        if not isinstance(logits, DTensor):
            return logits.detach().topk(kmax, dim=-1).indices

        vocab_axes = [
            axis
            for axis, placement in enumerate(logits.placements)
            if isinstance(placement, Shard) and placement.dim in (logits.ndim - 1, -1)
        ]
        local = logits.to_local().detach()
        if not vocab_axes:
            return local.topk(kmax, dim=-1).indices

        (axis,) = vocab_axes  # a single vocab-sharded mesh axis (loss parallel)
        group = logits.device_mesh.get_group(axis)
        group_size = dist.get_world_size(group)
        if local.shape[-1] * group_size != logits.shape[-1]:
            # Uneven vocab shards: give up on the offset math, gather instead.
            return logits.full_tensor().detach().topk(kmax, dim=-1).indices

        # The global top-k is contained in the union of per-shard top-ks.
        values, ids = local.float().topk(kmax, dim=-1)
        coord = logits.device_mesh.get_coordinate()
        # get_coordinate() is None only for ranks outside the mesh; this
        # rank just read its local shard of `logits` from that mesh.
        assert coord is not None, (
            "rank holds a DTensor shard but has no mesh coordinate"
        )
        ids = ids + coord[axis] * local.shape[-1]
        gathered_values = [torch.empty_like(values) for _ in range(group_size)]
        gathered_ids = [torch.empty_like(ids) for _ in range(group_size)]
        dist.all_gather(gathered_values, values.contiguous(), group=group)
        dist.all_gather(gathered_ids, ids.contiguous(), group=group)
        all_values = torch.cat(gathered_values, dim=-1)
        all_ids = torch.cat(gathered_ids, dim=-1)
        top = all_values.topk(kmax, dim=-1).indices
        return all_ids.gather(-1, top)

    def after_forward(self, state: ObserverState) -> None:
        if not self.active or not self._topk_chunks or state.labels is None:
            return
        with torch.no_grad():
            labels = state.labels
            if isinstance(labels, DTensor):
                labels = labels.to_local()
            labels = self._cp_local_labels(state, labels)
            is_mtp = state.loss_fn_type is not None and issubclass(
                state.loss_fn_type, MTPLoss
            )
            if is_mtp:
                if labels.dim() == 3:
                    # positions ride beside the labels as [B, 2, S]
                    # (MTPLoss.pack_targets); row 0 is the data labels.
                    labels = labels[:, 0]
            topk_ids = self._stitch_topk_chunks(labels)
            if is_mtp:
                # The shared lm_head also ran over every MTP head's hidden
                # states, in head order with the main head first; only the
                # main head's logits answer for next-token accuracy.
                num_heads = len(state.mtp_loss_weights or ()) + 1
                if topk_ids.shape[0] == labels.shape[0] * num_heads:
                    topk_ids = topk_ids[: labels.shape[0]]
            if labels.shape != topk_ids.shape[:-1]:
                raise ValueError(
                    f"TrainerSource token_accuracy: labels shape "
                    f"{tuple(labels.shape)} does not match logits positions "
                    f"{tuple(topk_ids.shape[:-1])} — the logits module or "
                    "sharding changed; update the accuracy capture."
                )
            valid = labels != IGNORE_INDEX
            matches = topk_ids.eq(labels.unsqueeze(-1))
            counts = torch.stack(
                [(matches[..., :k].any(dim=-1) & valid).sum().float() for k in self._ks]
                + [valid.sum().float()]
            )
            self._counts = counts if self._counts is None else self._counts + counts

    def _stitch_topk_chunks(self, labels: torch.Tensor) -> torch.Tensor:
        """Reassemble per-call top-k ids into (batch, seq, k) label order."""
        chunks = self._topk_chunks
        self._topk_chunks = []
        if len(chunks) == 1:
            return chunks[0]
        seq_len = labels.shape[1]
        rows: list[torch.Tensor] = []
        pending: list[torch.Tensor] = []
        pending_seq = 0
        for chunk in chunks:
            pending.append(chunk)
            pending_seq += chunk.shape[1]
            if pending_seq == seq_len:
                rows.append(
                    torch.cat(pending, dim=1) if len(pending) > 1 else pending[0]
                )
                pending, pending_seq = [], 0
            elif pending_seq > seq_len:
                break
        if pending:
            raise ValueError(
                f"TrainerSource token_accuracy: cannot stitch logits chunks "
                f"{[tuple(chunk.shape[:-1]) for chunk in chunks]} against "
                f"labels {tuple(labels.shape)} — chunk sequence lengths do "
                "not tile the label sequence; update the accuracy capture."
            )
        return torch.cat(rows, dim=0) if len(rows) > 1 else rows[0]

    def _cp_local_labels(
        self, state: ObserverState, labels: torch.Tensor
    ) -> torch.Tensor:
        """Shard full-sequence labels the way CP sharded the model inputs.

        state.labels is captured before post_dataloading_process, so under CP
        the logits cover only this rank's (load-balanced) sequence shard.
        Reusing cp_shard with the trainer's load balancer applies the same
        permutation, keeping positions aligned. Valid/hit counters are summed
        over the cp mesh afterwards, so the accuracy still describes full
        sequences.
        """
        cp_mesh = state.parallel_dims.get_optional_mesh("cp")
        if cp_mesh is None or cp_mesh.size() <= 1:
            return labels
        from torchtitan.distributed.context_parallel import cp_shard

        (labels,), _ = cp_shard(
            cp_mesh,
            (labels,),
            None,
            state.config.parallelism.context_parallel_load_balancer,
        )
        return labels

    def after_optimizer_step(self, state: ObserverState) -> None:
        """Speed series, on the former SpeedMonitor's own cadences.

        tps/tflops/mfu and the interval counters are core telemetry, owned by
        ``MetricsProcessor.interval_metrics``; this source only adds its own
        throughput windows on top.
        """
        super().after_optimizer_step(state)
        if not self.config.log_throughput and not self.config.log_remaining_estimate:
            return
        if self.config.log_throughput:
            self._append_snapshot(state)
        if state.step % self.config.throughput_log_interval != 0:
            return

        metrics: dict[str, float] = {}
        if self.config.log_throughput:
            metrics.update(
                {
                    "time/train": self._train_seconds() / _SECONDS_PER_HOUR,
                    "time/token_in_epoch": self._history_tokens[-1],
                    "time/sample_in_epoch": self._history_samples[-1],
                    "time/batch_in_epoch": self._history_steps[-1],
                }
            )
        if self.config.log_throughput and len(self._history_wct) >= 2:
            step_wct = self._history_wct[-1] - self._history_wct[-2]
            step_tokens = self._history_tokens[-1] - self._history_tokens[-2]
            if step_wct > 0:
                metrics["throughput/step/tokens_per_sec"] = step_tokens / step_wct

        window_full = self.config.log_throughput and (
            len(self._history_wct) == self.config.throughput_window_size + 1
        )
        if window_full and self._history_wct[-1] - self._history_wct[0] > 0:
            elapsed_wct = self._history_wct[-1] - self._history_wct[0]
            world_size = state.parallel_dims.world_size
            batches = (self._history_steps[-1] - self._history_steps[0]) / elapsed_wct
            samples = (
                self._history_samples[-1] - self._history_samples[0]
            ) / elapsed_wct
            tokens = (self._history_tokens[-1] - self._history_tokens[0]) / elapsed_wct
            metrics.update(
                {
                    "throughput/batches_per_sec": batches,
                    "throughput/samples_per_sec": samples,
                    "throughput/tokens_per_sec": tokens,
                    "throughput/device/batches_per_sec": batches / world_size,
                    "throughput/device/samples_per_sec": samples / world_size,
                    "throughput/device/tokens_per_sec": tokens / world_size,
                }
            )
        if (
            self.config.log_remaining_estimate
            and self._estimate_start_time is not None
            and self._estimate_start_step is not None
        ):
            completed = int(state.step) - self._estimate_start_step + 1
            remaining = max(int(state.config.training.steps) - int(state.step), 0)
            seconds_per_step = (
                time.perf_counter() - self._estimate_start_time
            ) / completed
            metrics["time/remaining_estimate"] = (
                seconds_per_step * remaining / _SECONDS_PER_HOUR
            )
        if metrics:
            state.record_all(metrics)

    def _data_parallel_size(self, state: ObserverState) -> int:
        parallel_dims = state.parallel_dims
        return parallel_dims.dp_replicate * parallel_dims.dp_shard

    def _append_snapshot(self, state: ObserverState) -> None:
        samples_per_step = (
            int(state.config.training.local_batch_size)
            * int(state.gradient_accumulation_steps)
            * self._data_parallel_size(state)
        )
        self._history_steps.append(int(state.step))
        self._history_samples.append(int(state.step) * samples_per_step)
        self._history_tokens.append(int(state.ntokens_seen))
        self._history_wct.append(time.perf_counter())

    def _train_seconds(self) -> float:
        return self._prior_train_seconds + (
            time.perf_counter() - self._speed_start_time
        )

    def stateful_objects(self) -> dict[str, object]:
        # The key predates the merge into TrainerSource: checkpoints written
        # by the standalone SpeedMonitor keep loading.
        if self.config.log_throughput:
            return {"speed_monitor": self}
        return {}

    def state_dict(self) -> dict[str, object]:
        return {"train_seconds": self._train_seconds()}

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        self._prior_train_seconds = float(state_dict.get("train_seconds", 0.0))
        self._speed_start_time = time.perf_counter()

    def before_optimizer_step(self, state: ObserverState) -> None:
        """Record the step's learning rates, one scalar record per group.

        These land in the same batch's records (cleared at batch_start,
        drained at batch_end) with no reduce stages: an LR is identical
        across the data ranks, and per-stage tags reach the logging rank
        through the emitter's pp gather like every other record.
        """
        if not self.active or not self.config.log_lr:
            return
        groups = state.optimizer_groups()
        num_parts = len({group.part_index for group in groups})
        metrics: dict[str, float] = {}
        for group in groups:
            part = "" if num_parts == 1 else f"/part{group.part_index}"
            metrics[f"lr-{group.optimizer_name}{part}/group{group.group_index}"] = (
                float(group.hparams.get("lr", 0.0))
            )
        if self.config.include_parameter_lr_tags:
            self._add_parameter_lrs(groups, self._param_names(state), metrics)
        for name, value in metrics.items():
            self._records.append(
                MetricRecord(
                    name=name, value=value, step=self._step, reduce=ReduceConfig()
                )
            )

    def _param_names(self, state: ObserverState) -> dict[int, list[str]]:
        if self._param_names_by_id is None:
            names: dict[int, list[str]] = {}
            for fqn, param, _ in iter_logical_weights(state.model_parts):
                names.setdefault(id(param), []).append(fqn)
            self._param_names_by_id = names
        return self._param_names_by_id

    def _add_parameter_lrs(
        self,
        groups: tuple[OptimizerGroupView, ...],
        param_names: dict[int, list[str]],
        metrics: dict[str, float],
    ) -> None:
        lrs: dict[str, float] = {}
        conflicts: set[str] = set()
        for group in groups:
            lr = float(group.hparams.get("lr", 0.0))
            for param in group.params:
                for fqn in param_names.get(id(param), ()):
                    if lrs.setdefault(fqn, lr) != lr:
                        conflicts.add(fqn)
        if conflicts and not self._warned_lr_conflicts:
            self._warned_lr_conflicts = True
            logger.warning(
                "TrainerSource: skipping parameter LR tags with conflicting "
                f"values after wrapper stripping: {sorted(conflicts)}"
            )
        for fqn in sorted(lrs.keys() - conflicts):
            metrics[f"lr/{fqn}"] = lrs[fqn]

    def batch_end(self, state: ObserverState) -> None:
        if not self.active:
            return
        self._record_tag(
            "time_seconds/batch/batch_total",
            time.perf_counter() - self._batch_start_time,
        )
        self._record_tag("time/batch", int(state.step))
        self._record_tag("time/sample", int(state.step) * self._samples_per_step(state))
        if state.global_valid_tokens is not None:
            self._record_tag(
                "trainer/global_valid_tokens", float(state.global_valid_tokens)
            )
        if not self._logged_batch_config:
            self._record_trainer_batch_config(state)
            self._logged_batch_config = True
        if self._logits_hooked:
            self._record_token_accuracy(state)

    def _record_token_accuracy(self, state: ObserverState) -> None:
        assert self.reducer is not None, "MonitorSource records before fit_start"
        if self._counts is None:
            counts = torch.zeros(len(self._ks) + 1, device=state.device)
        else:
            counts = self._counts
        # Hits are rank-local; sum over the data and context parallel groups so
        # the accuracy describes the global batch. Neither group spans pp, and
        # the logits hook exists uniformly within a pipeline stage.
        for group in ("dp", "cp"):
            counts = self.reducer.reduce_stage(
                counts, ReduceStageConfig(group=group, op=ReduceOp.sum)
            )
        valid = counts[-1].clamp_min(1.0)
        if counts[-1].item() > 0:
            for i, k in enumerate(self._ks):
                self._record_tag(f"token_accuracy/top{k}", counts[i] / valid)

    def _samples_per_step(self, state: ObserverState) -> int:
        parallel_dims = state.parallel_dims
        data_parallel_size = parallel_dims.dp_replicate * parallel_dims.dp_shard
        return (
            int(state.config.training.local_batch_size)
            * int(state.gradient_accumulation_steps)
            * data_parallel_size
        )

    def _record_trainer_batch_config(self, state: ObserverState) -> None:
        self._record_tag(
            "trainer/device_train_microbatch_size",
            int(state.config.training.local_batch_size),
        )
        self._record_tag(
            "trainer/global_train_batch_size", self._samples_per_step(state)
        )
        self._record_tag(
            "trainer/gradient_accumulation_steps",
            int(state.gradient_accumulation_steps),
        )
        self._record_tag("trainer/num_gpus", state.parallel_dims.world_size)

    def _record_tag(self, tag: str, value: torch.Tensor | int | float) -> None:
        sample_name, metric_name = tag.rsplit("/", 1)
        self._record(
            Sample(
                name=sample_name,
                value=value,
                metrics=[MetricConfig(name=metric_name)],
            )
        )
