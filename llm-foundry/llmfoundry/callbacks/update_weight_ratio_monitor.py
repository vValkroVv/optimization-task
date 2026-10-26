import math
from typing import NamedTuple

import torch
import triton
import triton.language as tl

from composer.core import Callback, State
from composer.functional.triton_kernels import div_ieee, float32_reciprocal, mul_ieee, slice_chunks, sqrt_ieee
from composer.loggers import Logger
from composer.optim.decoupled_weight_decay import step_counts
from composer.utils import dist

from llmfoundry.callbacks.optimizer_health_monitor import classify_param

__all__ = ["UpdateWeightRatioMonitor"]

# per-fqn accumulator layout: [numel, update_abs_sum, w_sum, w_sq_sum]
N_ACCUM_VALUES = 4

# Elements of a chunk of a slice reduced at once by _slice_partials.
_TILE = 2048

_TRITON_DTYPES = {torch.float32: tl.float32, torch.bfloat16: tl.bfloat16, torch.float16: tl.float16}


class _AdamStep(NamedTuple):
    """The applied Adam step ``step_size * m * per_fqn_lr / (sqrt(v) / sqrt_bc2 + eps)``."""

    m: torch.Tensor
    v: torch.Tensor
    per_fqn_lr: torch.Tensor | None
    sqrt_bc2: float
    eps: float
    step_size: float


@triton.jit
def _slice_partials(tensors_ptr, scalars_ptr, flags_ptr, params_ptr, starts_ptr, ends_ptr, partials_ptr,
                    W_DTYPE: tl.constexpr, STATE_DTYPE: tl.constexpr, LR_DTYPE: tl.constexpr, TILE: tl.constexpr):
    # One chunk of one slice of one parameter: [sum |update|, sum w, sum w^2, max |w|]. The chunk's
    # parameter has the addresses of its weight, moments and LR multiplier in `tensors_ptr`, its
    # [1 / sqrt(bc2), eps, step size] in `scalars_ptr` and whether it has an update and an LR
    # multiplier in `flags_ptr`. The update is rebuilt with the operations and roundings of the
    # torch expression it replaces.
    chunk = tl.program_id(0)
    param = tl.load(params_ptr + chunk)
    w_ptr = tl.load(tensors_ptr + 4 * param).to(tl.pointer_type(W_DTYPE))
    m_ptr = tl.load(tensors_ptr + 4 * param + 1).to(tl.pointer_type(STATE_DTYPE))
    v_ptr = tl.load(tensors_ptr + 4 * param + 2).to(tl.pointer_type(STATE_DTYPE))
    lr_ptr = tl.load(tensors_ptr + 4 * param + 3).to(tl.pointer_type(LR_DTYPE))
    inverse_sqrt_bc2 = tl.load(scalars_ptr + 3 * param)
    eps = tl.load(scalars_ptr + 3 * param + 1)
    step_size = tl.load(scalars_ptr + 3 * param + 2)
    has_update = tl.load(flags_ptr + 2 * param) != 0
    has_lr = tl.load(flags_ptr + 2 * param + 1) != 0
    start = tl.load(starts_ptr + chunk)
    end = tl.load(ends_ptr + chunk)
    update_sum = tl.zeros([TILE], dtype=tl.float32)
    w_sum = tl.zeros([TILE], dtype=tl.float32)
    w_sq_sum = tl.zeros([TILE], dtype=tl.float32)
    w_amax = tl.zeros([TILE], dtype=tl.float32)
    for tile in range(start, end, TILE):
        offsets = tile + tl.arange(0, TILE)
        mask = offsets < end
        w = tl.load(w_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        w_sum += w
        w_sq_sum += w * w
        w_amax = tl.maximum(w_amax, tl.abs(w))
        if has_update:
            m = tl.load(m_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            v = tl.load(v_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
            if has_lr:
                m = m * tl.load(lr_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            den = mul_ieee(sqrt_ieee(v), inverse_sqrt_bc2) + eps
            update_sum += tl.where(mask, tl.abs(div_ieee(step_size * m, den)), 0.0)
    base = partials_ptr + chunk * 4
    w_sq = tl.sum(w_sq_sum)
    tl.store(base + 0, tl.sum(update_sum))
    tl.store(base + 1, tl.sum(w_sum))
    tl.store(base + 2, w_sq)
    # tl.max drops NaN, which torch's amax propagates; the sum of squares is NaN exactly when a weight is.
    tl.store(base + 3, tl.where(w_sq != w_sq, w_sq, tl.max(w_amax)))


class UpdateWeightRatioMonitor(Callback):
    """Monitors the ratio of the applied update magnitude to the weight std.

    From the Adam optimizer state it reconstructs the actual Adam step applied
    to each weight (excluding the separate decoupled weight-decay term), matching
    ``DecoupledAdamW``: bias-corrected denom, the effective per-param LR scale
    composed by ``DynamicBlockLRRescaler``, and any packed per-FQN LR vector::

        update = (scale * lr / bc1) * per_fqn_lr * m / (sqrt(v) / sqrt(bc2) + eps)

    and logs the ratio of its mean absolute value to the standard deviation of
    the weights::

        update_over_weight_std = mean(|update|) / std(weight)

    This is the classic "update-to-weight" health signal: a well-behaved run
    typically sits around ~1e-3. Values much larger indicate steps that are
    large relative to the parameters (instability); much smaller values
    indicate the layer is effectively frozen.

    ``log_level`` selects the granularity of the emitted metrics:

    - ``'fqn'``: one metric per fully-qualified parameter name.
    - ``'group'``: aggregated over the coarse parameter groups from
      ``classify_param``.

    Note: ``'fqn'`` emits O(num_params) metrics per logging step (can be
    hundreds/thousands for a large model). Stats are accumulated into a single
    device tensor and reduced with one ``all_reduce`` / one host transfer, so
    the GPU cost is small, but keep ``batch_log_interval`` reasonably large.
    """

    def __init__(
        self,
        batch_log_interval: int = 100,
        log_level: str = "fqn",
    ):
        if batch_log_interval <= 0:
            raise ValueError(
                f"batch_log_interval must be > 0, got {batch_log_interval}"
            )
        if log_level not in ("fqn", "group"):
            raise ValueError(f"log_level must be 'fqn' or 'group', got {log_level!r}")
        self.batch_log_interval = batch_log_interval
        self.log_level = log_level
        self._param_to_group = None
        self._optimizer = None
        self.rank_local_observer = None
        # deterministic, cross-rank-consistent ordering of fqns -> row index
        self._fqn_order = None
        self._gigafsdp_slices = None
        self._step_hparams = None
        self._accumulated_slices = {}
        self._batched_slices = {}

    def fit_start(self, state: State, logger: Logger):
        optimizer = state.optimizers[0]
        self._param_to_group = {}
        for group in optimizer.param_groups:
            for p in group["params"]:
                self._param_to_group[p] = group
        self._optimizer = optimizer
        self._accumulated_slices = {}
        self._batched_slices = {}

        if state.gigafsdp_enabled:
            self._gigafsdp_slices, self._fqn_order = self._build_gigafsdp_slices(state)
        else:
            self._gigafsdp_slices = None
            # Sorted so every rank builds the same row layout for the all_reduce.
            # FSDP replicates the fqn set across ranks (only shard contents differ).
            self._fqn_order = sorted(
                name for name, p in state.model.named_parameters() if p.requires_grad
            )

    def batch_start(self, state: State, logger: Logger):
        """Snapshot the hyperparameters that the upcoming step will use.

        Composer advances batch schedulers before ``batch_end``. Reading the
        optimizer group there would therefore use the *next* step's LR.
        """
        del logger
        next_batch = state.timestamp.batch.value + 1
        if next_batch % self.batch_log_interval != 0 and self.rank_local_observer is None:
            self._step_hparams = None
            return

        optimizer = self._optimizer
        if optimizer is None:
            return
        get_scale = getattr(optimizer, "get_effective_lr_scale", None)
        get_per_fqn_lr = getattr(optimizer, "get_per_fqn_lr_multiplier", None)
        self._step_hparams = {}
        for p, group in self._param_to_group.items():
            scale = get_scale(p) if get_scale is not None else 1.0
            per_fqn_lr = get_per_fqn_lr(p) if get_per_fqn_lr is not None else None
            if torch.is_tensor(per_fqn_lr):
                per_fqn_lr = per_fqn_lr.detach().clone()
            self._step_hparams[p] = (group.get("lr", 0.0), scale, per_fqn_lr)

    def batch_end(self, state: State, logger: Logger):
        should_log = state.timestamp.batch.value % self.batch_log_interval == 0
        if not should_log and self.rank_local_observer is None:
            return

        if self._param_to_group is None or self._fqn_order is None:
            return

        optimizer = self._optimizer

        # Effective per-param LR multiplier (config `lr_scales` base composed
        # with the dynamic per-block scale). Falls back to 1.0 on optimizers
        # that don't support per-param scaling.
        get_scale = getattr(optimizer, "get_effective_lr_scale", None)
        if get_scale is None:

            def get_scale(_p):
                return 1.0

        get_per_fqn_lr = getattr(optimizer, "get_per_fqn_lr_multiplier", None)

        fqn_to_row = {name: i for i, name in enumerate(self._fqn_order)}
        device = next(state.model.parameters()).device
        stats = torch.zeros(
            len(self._fqn_order) * N_ACCUM_VALUES,
            device=device,
            dtype=torch.float64,
        )

        local_amax = (
            torch.full((len(self._fqn_order),), float("nan"), device=device, dtype=torch.float32)
            if self.rank_local_observer is not None else None
        )
        steps = step_counts(optimizer)
        accum = stats.view(-1, N_ACCUM_VALUES)
        # CUDA parameters are reduced together after the loop, in one launch per dtype combination.
        batched = {}

        for name, p in state.model.named_parameters():
            if not p.requires_grad or p.grad is None:
                continue

            group = self._param_to_group.get(p)
            if group is None:
                continue

            with torch.no_grad():
                opt_state = optimizer.state.get(p)
                update = None
                if opt_state is not None and "exp_avg" in opt_state:
                    required = {"exp_avg_sq", "step"}
                    missing = required.difference(opt_state)
                    if missing:
                        raise RuntimeError(
                            "UpdateWeightRatioMonitor supports Adam-like optimizers "
                            f"only; optimizer state is missing {sorted(missing)}"
                        )
                    eps = group.get("eps", 1e-8)
                    beta1, beta2 = group.get("betas", (0.9, 0.999))
                    m = opt_state["exp_avg"]
                    if group.get("amsgrad", False):
                        if "max_exp_avg_sq" not in opt_state:
                            raise RuntimeError(
                                "AMSGrad optimizer state is missing max_exp_avg_sq"
                            )
                        v = opt_state["max_exp_avg_sq"]
                    else:
                        v = opt_state["exp_avg_sq"]

                    # Reconstruct the ACTUAL applied step, matching
                    # DecoupledAdamW.adamw: bias-corrected denom and a per-param
                    # effective LR scale (base config `lr_scales` composed with
                    # the dynamic per-block scale from DynamicBlockLRRescaler).
                    step = steps[p] if p in steps else opt_state["step"]
                    step = step.item() if torch.is_tensor(step) else step
                    bc1 = 1 - beta1**step
                    bc2 = 1 - beta2**step
                    if self._step_hparams is not None and p in self._step_hparams:
                        lr, scale, per_fqn_lr = self._step_hparams[p]
                    else:
                        # Allows manual callback invocation while normal Composer
                        # runs use the batch_start snapshot above.
                        lr = group.get("lr", 0.0)
                        scale = get_scale(p)
                        per_fqn_lr = (
                            get_per_fqn_lr(p) if get_per_fqn_lr is not None else None
                        )

                    update = _AdamStep(m, v, per_fqn_lr, math.sqrt(bc2), eps, scale * lr / bc1)

                if self._gigafsdp_slices is None:
                    slices = [(name, 0, p.numel())]
                else:
                    slices = self._gigafsdp_slices.get(p, ())

                accumulated = self._accumulated_slices.get(p)
                if accumulated is None:
                    accumulated = self._build_accumulated_slices(slices, fqn_to_row, device)
                    self._accumulated_slices[p] = accumulated
                chunks, rows, numels = accumulated
                if chunks is not None and p.is_cuda:
                    key = (p.dtype, p.dtype if update is None else update.m.dtype,
                           p.dtype if update is None or update.per_fqn_lr is None else update.per_fqn_lr.dtype)
                    batched.setdefault(key, []).append((p, update, accumulated))
                elif chunks is not None:
                    sums, amax = self._slice_statistics(p, update, chunks)
                    accum[rows] += torch.cat((numels[:, None], sums), dim=1)
                    if self.rank_local_observer is not None:
                        local_amax[rows] = torch.fmax(local_amax[rows], amax)
                del update

        with torch.no_grad():
            for dtypes, entries in batched.items():
                sums, amax, rows, numels = self._batched_slice_statistics(dtypes, entries)
                accum[rows] += torch.cat((numels[:, None], sums), dim=1)
                if self.rank_local_observer is not None:
                    local_amax[rows] = torch.fmax(local_amax[rows], amax)
        self._step_hparams = None

        if self.rank_local_observer is not None:
            rows = stats.view(-1, N_ACCUM_VALUES)
            local_l2 = torch.where(rows[:, 0] > 0, rows[:, 3].sqrt(), float("nan")).float()
            self.rank_local_observer(self._fqn_order, rows, torch.stack((local_amax, local_l2), dim=1))
        if not should_log:
            return

        fsdp_group = dist.get_fsdp_group()
        dist.all_reduce(stats, reduce_operation="SUM", group=fsdp_group)

        if state.gigafsdp_enabled:
            hybrid_group_count = state.model.model._meta_info["hybrid_group_count"]
        else:
            hybrid_group_count = 1

        # single device -> host transfer for the whole reduced tensor
        flat = stats.tolist()

        metrics = {}
        group_accum = {}

        for name, row in fqn_to_row.items():
            offset = row * N_ACCUM_VALUES
            numel = flat[offset + 0] / hybrid_group_count
            if numel < 1:
                continue

            update_abs_sum = flat[offset + 1] / hybrid_group_count
            w_sum = flat[offset + 2] / hybrid_group_count
            w_sq_sum = flat[offset + 3] / hybrid_group_count

            if self.log_level == "fqn":
                metrics.update(
                    self._compute(
                        f"update_weight_ratio/{{}}/{name}",
                        numel,
                        update_abs_sum,
                        w_sum,
                        w_sq_sum,
                    )
                )
            else:  # 'group'
                g = group_accum.setdefault(
                    classify_param(name),
                    {
                        "numel": 0.0,
                        "update_abs_sum": 0.0,
                        "w_sum": 0.0,
                        "w_sq_sum": 0.0,
                    },
                )
                g["numel"] += numel
                g["update_abs_sum"] += update_abs_sum
                g["w_sum"] += w_sum
                g["w_sq_sum"] += w_sq_sum

        if self.log_level == "group":
            for group_name, g in group_accum.items():
                if g["numel"] < 1:
                    continue
                metrics.update(
                    self._compute(
                        f"update_weight_ratio/{{}}/{group_name}",
                        g["numel"],
                        g["update_abs_sum"],
                        g["w_sum"],
                        g["w_sq_sum"],
                    )
                )

        logger.log_metrics(metrics)

    def _batched_slice_statistics(self, dtypes: tuple, entries: list) -> tuple[torch.Tensor, ...]:
        """For CUDA parameters of one (weight, moment, LR multiplier) dtype combination with their
        updates and accumulated slices: per slice of all of them, in order, float64 [update_abs_sum,
        w_sum, w_sq_sum], float32 max |w|, and the slices' rows and element counts. One launch reads
        every chunk of every slice of every parameter; segment_reduce combines each slice's chunk
        partials on their own, so a slice's statistics do not depend on the other parameters."""
        device = entries[0][0].device
        key = tuple(p for p, _, _ in entries)
        tables = self._batched_slices.get(key)
        if tables is None:
            accumulated = [chunks_rows_numels for _, _, chunks_rows_numels in entries]
            params = torch.cat([torch.full_like(chunks[0], index) for index, (chunks, _, _) in enumerate(accumulated)])
            tables = self._batched_slices[key] = (
                params,
                *(torch.cat([chunks[part] for chunks, _, _ in accumulated]) for part in range(3)),
                torch.cat([rows for _, rows, _ in accumulated]),
                torch.cat([numels for _, _, numels in accumulated]),
            )
        params, starts, ends, chunks_per_slice, rows, numels = tables
        addresses, scalars, flags = [], [], []
        for p, update, _ in entries:
            weights = p.data.reshape(-1)
            if update is None:
                addresses.append([weights.data_ptr()] * 4)
                scalars.append([1.0, 0.0, 0.0])
                flags.append([0, 0])
            else:
                lr = weights if update.per_fqn_lr is None else update.per_fqn_lr
                addresses.append([weights.data_ptr(), update.m.data_ptr(), update.v.data_ptr(), lr.data_ptr()])
                scalars.append([float32_reciprocal(update.sqrt_bc2), update.eps, update.step_size])
                flags.append([1, int(update.per_fqn_lr is not None)])
        # Staged in pinned memory, so the copies do not wait for the queued step.
        addresses, scalars, flags = (
            torch.tensor(values, dtype=dtype, pin_memory=True).to(device, non_blocking=True)
            for values, dtype in ((addresses, torch.int64), (scalars, torch.float32), (flags, torch.int32))
        )
        partials = torch.empty(starts.numel(), 4, dtype=torch.float32, device=device)
        w_dtype, state_dtype, lr_dtype = (_TRITON_DTYPES[dtype] for dtype in dtypes)
        _slice_partials[(starts.numel(),)](
            addresses, scalars, flags, params, starts, ends, partials,
            W_DTYPE=w_dtype, STATE_DTYPE=state_dtype, LR_DTYPE=lr_dtype, TILE=_TILE,
        )
        sums = torch.segment_reduce(partials[:, :3].double(), "sum", lengths=chunks_per_slice, unsafe=True)
        amax = torch.segment_reduce(partials[:, 3], "max", lengths=chunks_per_slice, unsafe=True)
        return sums, amax, rows, numels

    @staticmethod
    def _slice_statistics(p: torch.Tensor, update: _AdamStep | None, chunks: tuple) -> tuple[torch.Tensor, torch.Tensor]:
        """Per slice: float64 [update_abs_sum, w_sum, w_sq_sum] and float32 max |w|, chunk partials
        combined in float64; the CPU counterpart of _batched_slice_statistics."""
        starts, ends, chunks_per_slice = chunks
        partials = torch.empty(starts.numel(), 4, dtype=torch.float32, device=p.device)
        w = p.data.reshape(-1).float()
        update_abs = torch.zeros_like(w)
        if update is not None:
            numerator = update.m.reshape(-1).float()
            if update.per_fqn_lr is not None:
                numerator = numerator * update.per_fqn_lr.reshape(-1).float()
            den = update.v.reshape(-1).float().sqrt() / update.sqrt_bc2 + update.eps
            update_abs = (update.step_size * numerator / den).abs()
        for chunk, (start, end) in enumerate(zip(starts.tolist(), ends.tolist())):
            w_chunk = w[start:end]
            partials[chunk] = torch.stack((
                update_abs[start:end].sum(), w_chunk.sum(), w_chunk.square().sum(), w_chunk.abs().amax(),
            ))
        sums = torch.segment_reduce(partials[:, :3].double(), "sum", lengths=chunks_per_slice, unsafe=True)
        amax = torch.segment_reduce(partials[:, 3], "max", lengths=chunks_per_slice, unsafe=True)
        return sums, amax

    @staticmethod
    def _build_accumulated_slices(slices, fqn_to_row: dict, device) -> tuple:
        """The non-empty slices with an accumulator row as ``slice_chunks`` (None if there is no
        slice), their rows and element counts."""
        kept = [(start, end, fqn_to_row[fqn]) for fqn, start, end in slices if fqn in fqn_to_row and end > start]
        chunks = None if not kept else slice_chunks([(start, end) for start, end, _ in kept], device)
        return (
            chunks,
            torch.tensor([row for _, _, row in kept], dtype=torch.int64, device=device),
            torch.tensor([end - start for start, end, _ in kept], dtype=torch.float64, device=device),
        )

    @staticmethod
    def _build_gigafsdp_slices(state: State):
        """Map each local flat shard to its real, non-padding FQN slices."""
        gigafsdp_model = state.model.model
        meta = gigafsdp_model._meta_info
        layer_to_fqns = {}
        for fqn, layer_name in meta["param_name_to_layer_name"].items():
            layer_to_fqns.setdefault(layer_name, []).append(fqn)

        giga_named_params = dict(gigafsdp_model.named_parameters())
        param_to_layer = {p: name for name, p in giga_named_params.items()}
        result = {}
        all_fqns = set()

        for p in (p for p in state.model.parameters() if p.requires_grad):
            layer_name = param_to_layer.get(p)
            if layer_name is None:
                continue
            fqns = layer_to_fqns.get(layer_name)
            if not fqns:
                continue

            if layer_name.endswith("flatten"):
                prefix = "main"
                meta_idx = meta["layer_name_to_meta_info_idx"][layer_name]
            elif layer_name.endswith("layernorm_parameter"):
                prefix = "layernorm"
                meta_idx = 0
            else:
                continue

            ordered_fqns = meta[f"{prefix}_shards_fqns"][meta_idx]
            all_fqns.update(ordered_fqns)
            numels = meta[f"{prefix}_shards_numels"][meta_idx]
            shard_size = meta[f"{prefix}_parameter_shard_bounaries"][meta_idx]
            process_group = meta.get("param_name_to_fsdp_group", {}).get(
                ordered_fqns[0]
            )
            if process_group is None:
                process_group = dist.get_fsdp_group()
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                rank = torch.distributed.get_rank(group=process_group)
            else:
                rank = 0
            shard_start = rank * shard_size
            shard_end = shard_start + shard_size

            slices = []
            fqn_start = 0
            for fqn, numel in zip(ordered_fqns, numels):
                fqn_end = fqn_start + numel
                overlap_start = max(fqn_start, shard_start)
                overlap_end = min(fqn_end, shard_end)
                if overlap_start < overlap_end:
                    slices.append(
                        (
                            fqn,
                            overlap_start - shard_start,
                            overlap_end - shard_start,
                        )
                    )
                fqn_start = fqn_end
            result[p] = slices

        # Every rank must allocate rows in the same order, including FQNs that
        # do not overlap its local shard.
        return result, sorted(all_fqns)

    @staticmethod
    def _compute(key_template, numel, update_abs_sum, w_sum, w_sq_sum):
        update_abs_mean = update_abs_sum / numel
        w_mean = w_sum / numel
        w_var = max(w_sq_sum / numel - w_mean * w_mean, 0.0)
        w_std = math.sqrt(w_var)
        ratio = update_abs_mean / w_std if w_std > 0 else float("inf")
        return {
            key_template.format("update_abs_mean"): update_abs_mean,
            key_template.format("weight_std"): w_std,
            key_template.format("ratio"): ratio,
        }
