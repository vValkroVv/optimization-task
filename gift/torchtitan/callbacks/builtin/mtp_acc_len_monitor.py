import torch
import torch.distributed as dist

from typing import TypeAlias

from torchtitan.config.schema.callbacks.builtin.mtp_acc_len_monitor import (
    MtpAccLenMonitorConfig,
)
from torchtitan.callbacks.base import Observer, ObserverState
from torchtitan.components.loss import MTPLoss
from torchtitan.config import Configurable
from torchtitan.tools.logging import logger


class MtpAccLenMonitor(Configurable, Observer):
    """Per-MTP-head acceptance and spec-decode acc_len.

    Every ``log_interval`` steps:

    - ``mtp_acc_len/head{h}_accept_vs_gt``: P(head h argmax == its shifted
      data label) — tracks the MTP training objective;
    - ``mtp_acc_len/head{h}_accept_vs_base``: P(head h argmax == the main
      head's argmax rolled to the same slot) — training-time proxy for
      serving-side speculative-decode acceptance;
    - ``mtp_acc_len/acc_len_h{H}_vs_gt`` / ``_vs_base``:
      ``1 + sum_{h<=H} P(heads 1..h all accepted)`` — expected accepted
      chain length when drafting H tokens.

    The argmaxes and per-microbatch counters are computed inside ``MTPLoss``
    from the chunk logits its loop already materializes (no second lm_head
    projection). This callback owns all monitoring state: on its steps it
    installs a list into ``MTPLoss.acceptance_sink``, collects the counters
    the loss appends there, sums them over the batch's microbatches, reduces
    once over the loss mesh and publishes.

    Disabled with a warning under tp (vocab-sharded logits) and cp (the
    roll-by-h crosses sequence shards). Under PP only the loss stage owns
    counters; other stages return before the collective, and the loss mesh
    never crosses pp, so no rank waits on a peer that skipped.
    """

    # pyrefly: ignore [bad-override]
    Config: TypeAlias = MtpAccLenMonitorConfig

    def __init__(self, config: Config) -> None:
        self.config = config
        self._active = False
        self._num_heads = 0
        self._counters: list[dict[str, torch.Tensor]] = []
        self._subscribed: list[dict[str, torch.Tensor]] | None = None

    def fit_start(self, state: ObserverState) -> None:
        loss_fn = state.loss_fn_type
        if (
            loss_fn is None
            or not issubclass(loss_fn, MTPLoss)
            or state.mtp_loss_weights is None
        ):
            logger.warning(
                "MtpAccLenMonitor: the trainer's loss is %s, not MTPLoss — "
                "there are no MTP heads to monitor, disabling.",
                getattr(loss_fn, "__name__", loss_fn),
            )
            return
        parallel_dims = state.parallel_dims
        if getattr(parallel_dims, "tp", 1) > 1:
            logger.warning(
                "MtpAccLenMonitor: tp>1 shards the vocabulary axis of the "
                "logits, argmax would need a cross-rank reduce — disabling."
            )
            return
        if getattr(parallel_dims, "cp", 1) > 1:
            logger.warning(
                "MtpAccLenMonitor: cp>1 shards the sequence, the roll-by-h "
                "comparison would cross shard boundaries — disabling."
            )
            return
        self._active = True
        self._num_heads = len(state.mtp_loss_weights)

    def _is_logging_step(self, step: int) -> bool:
        return step % self.config.log_interval == 0

    def batch_start(self, state: ObserverState) -> None:
        if not self._active:
            return
        if self._subscribed is not None:
            state.unsubscribe_acceptance(self._subscribed)
            self._subscribed = None
        self._counters = []
        if self._is_logging_step(state.step):
            state.subscribe_acceptance(self._counters)
            self._subscribed = self._counters

    def batch_end(self, state: ObserverState) -> None:
        if not self._active or not self._is_logging_step(state.step):
            return
        if self._subscribed is not None:
            state.unsubscribe_acceptance(self._subscribed)
            self._subscribed = None
        microbatch_counters, self._counters = self._counters, []
        if not microbatch_counters:
            # Validation fallback (3D pred) or a PP stage without the loss.
            return
        counters = {
            key: sum(c[key] for c in microbatch_counters)
            for key in microbatch_counters[0]
        }

        # One packed collective and one host transfer for the whole family:
        # [gt(H), base(H), valid(H), chain_gt(H), chain_base(H), main_valid].
        packed = torch.cat(
            [
                counters["per_head_correct_gt"],
                counters["per_head_correct_base"],
                counters["per_head_valid"],
                counters["chain_correct_gt"],
                counters["chain_correct_base"],
                counters["main_valid"].reshape(1),
            ]
        )
        mesh = state.parallel_dims.get_optional_mesh("loss")
        if mesh is not None and dist.is_initialized():
            dist.all_reduce(packed, group=mesh.get_group())
        values = packed.cpu().tolist()

        num_heads = self._num_heads
        gt = values[0:num_heads]
        base = values[num_heads : 2 * num_heads]
        valid = values[2 * num_heads : 3 * num_heads]
        chain_gt = values[3 * num_heads : 4 * num_heads]
        chain_base = values[4 * num_heads : 5 * num_heads]
        main_valid = values[5 * num_heads]
        if main_valid == 0:
            return

        metrics: dict[str, float] = {}
        cumulative_gt = 0.0
        cumulative_base = 0.0
        for h in range(num_heads):
            head = h + 1
            if valid[h] > 0:
                metrics[f"mtp_acc_len/head{head}_accept_vs_gt"] = gt[h] / valid[h]
                metrics[f"mtp_acc_len/head{head}_accept_vs_base"] = base[h] / valid[h]
            cumulative_gt += chain_gt[h] / main_valid
            cumulative_base += chain_base[h] / main_valid
            metrics[f"mtp_acc_len/acc_len_h{head}_vs_gt"] = 1.0 + cumulative_gt
            metrics[f"mtp_acc_len/acc_len_h{head}_vs_base"] = 1.0 + cumulative_base

        state.record_all(metrics)
