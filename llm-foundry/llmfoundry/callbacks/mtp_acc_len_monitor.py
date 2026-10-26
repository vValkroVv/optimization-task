import logging

from composer.core import Callback, State
from composer.loggers import Logger
from composer.utils import dist

from llmfoundry.callbacks.topk_accuracy_monitor import find_causal_lm

__all__ = ['MtpAccLenMonitor']

log = logging.getLogger(__name__)


class MtpAccLenMonitor(Callback):
    """Logs MTP draft acceptance rates and speculative-decoding accept length.

    Every ``batch_log_interval`` batches, for head ``h`` in 1..N
    (N = ``mtp_predictor_num``) and family ``f`` in ``{vs_gt, vs_base}``, prefixed
    ``mtp_acc_len/``:

    - ``head{h}_conditional_{f}``: P(head h accepted | heads 1..h-1 accepted).
    - ``head{h}_cumulative_{f}``: P(heads 1..h accepted).
    - ``acc_len_h{h}_{f}``: ``1 + sum_{j<=h} cumulative_j``, tokens emitted per
      verification step when drafting h.

    ``vs_gt`` compares the draft argmax against the ground-truth label -- read
    acc_len off it. ``vs_base`` compares against the base model's argmax.
    Reuses the model's argmax pass on the first microbatch of a logged batch.

    All counts share one population, ``full_valid``: positions where the main head
    and all N heads have a target.

    Disable conditions:
        - use_mtp=False (no MTP heads to monitor)
        - tp_size>1 (logits are vocab-partitioned across TP ranks)

    Supports use_liger=True via a chunked argmax-only path that re-projects
    hidden states through lm_head in small chunks (no full vocab tensor held).
    """

    _RESULT_KEYS = ('full_valid', 'chain_full_gt', 'chain_full_base')

    def __init__(
        self,
        batch_log_interval: int = 10,
    ):
        if batch_log_interval < 1:
            raise ValueError(
                f"batch_log_interval must be >= 1, got {batch_log_interval}"
            )
        self.batch_log_interval = batch_log_interval
        self._enabled = False
        self._log_this_batch = False

    def fit_start(self, state: State, logger: Logger):
        inner = find_causal_lm(state)
        if inner is None:
            log.warning("MtpAccLenMonitor: could not find inner model, disabling")
            return

        config = getattr(inner, 'config', None)
        if config is None:
            log.warning("MtpAccLenMonitor: model has no config, disabling")
            return

        if not getattr(inner, 'use_mtp', False):
            log.warning("MtpAccLenMonitor: use_mtp=False, disabling")
            return

        mtp_num = getattr(config, 'mtp_predictor_num', 0) or 0
        if mtp_num <= 0:
            log.warning("MtpAccLenMonitor: mtp_predictor_num<=0, disabling")
            return

        tp_size = getattr(config, 'tp_size', 1) or 1
        if tp_size > 1:
            log.warning(
                "MtpAccLenMonitor: tp_size>1, logits are vocab-partitioned, disabling",
            )
            return

        inner._mtp_acc_len_results = None
        inner._compute_mtp_acc_len_this_step = False
        self._inner = inner
        self._mtp_num = mtp_num
        self._enabled = True
        log.info(
            f"MtpAccLenMonitor enabled with mtp_predictor_num={mtp_num}, "
            f"target={type(inner).__name__}",
        )

    def batch_start(self, state: State, logger: Logger):
        if not self._enabled:
            return
        self._log_this_batch = (
            state.timestamp.batch.value + 1
        ) % self.batch_log_interval == 0
        if self._log_this_batch:
            self._inner._mtp_acc_len_results = None
            self._inner._compute_mtp_acc_len_this_step = True

    def batch_end(self, state: State, logger: Logger):
        if not self._log_this_batch:
            return
        self._log_this_batch = False

        results = getattr(self._inner, '_mtp_acc_len_results', None)
        self._inner._mtp_acc_len_results = None
        if results is None or any(key not in results for key in self._RESULT_KEYS):
            return

        reduce_tensors = [results[key] for key in self._RESULT_KEYS]

        sp_group = dist.get_sp_group()
        if sp_group is not None:
            for t in reduce_tensors:
                dist.all_reduce(t, reduce_operation='SUM', group=sp_group)

        fsdp_group = dist.get_fsdp_group()
        for t in reduce_tensors:
            dist.all_reduce(t, reduce_operation='SUM', group=fsdp_group)

        denom = results['full_valid'].item()
        if denom == 0:
            return

        metrics = {}
        for family, chain in (
            ('vs_gt', results['chain_full_gt']),
            ('vs_base', results['chain_full_base']),
        ):
            prev_chain = denom
            acc_len = 1.0
            for head_num, chain_h in enumerate(
                chain.tolist()[: self._mtp_num], start=1
            ):
                cumulative = chain_h / denom
                acc_len += cumulative
                metrics[f"mtp_acc_len/head{head_num}_cumulative_{family}"] = cumulative
                if prev_chain > 0:
                    metrics[f"mtp_acc_len/head{head_num}_conditional_{family}"] = chain_h / prev_chain
                metrics[f"mtp_acc_len/acc_len_h{head_num}_{family}"] = acc_len
                prev_chain = chain_h

        logger.log_metrics(metrics)
