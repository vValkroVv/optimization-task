import logging

from composer.core import Callback, State
from composer.loggers import Logger
from composer.utils import dist

__all__ = ['TopkAccuracyMonitor', 'find_causal_lm']

log = logging.getLogger(__name__)


def find_causal_lm(state: State):
    """Find the actual CausalLM below the Composer, GigaFSDP and GigaOmni wrappers.

    Callbacks must reach the unwrapped module to set attributes: GigaFSDP
    delegates `__getattr__` but not `__setattr__`.
    """
    inner = getattr(state.model, 'model', None)
    if inner is None:
        return None
    unwrapped = getattr(inner, '_module', inner)
    language_model = getattr(unwrapped, 'language_model', None)
    if language_model is not None:
        unwrapped = getattr(language_model, '_module', language_model)
    if hasattr(unwrapped, 'config') and hasattr(unwrapped, 'lm_head'):
        return unwrapped
    return inner


class TopkAccuracyMonitor(Callback):
    """Monitors top-k prediction accuracy during training.

    Computes accuracy/top1, accuracy/top5, accuracy/top20 (configurable).
    The actual top-k computation is injected into the model's forward method
    and results are read back in batch_end.

    Limitations:
    - Disabled when use_liger=True (logits are never materialized)
    - Disabled when tp_size > 1 (logits are vocab-partitioned across TP ranks)
    """

    def __init__(
        self,
        k_values: tuple[int, ...] | list[int] = (1, 5, 20),
        batch_log_interval: int = 10,
    ):
        self.k_values = sorted(k_values)
        self.batch_log_interval = batch_log_interval
        self._enabled = False

    def _find_causal_lm(self, state: State):
        return find_causal_lm(state)

    def fit_start(self, state: State, logger: Logger):
        inner = self._find_causal_lm(state)
        if inner is None:
            log.warning(
                "TopkAccuracyMonitor: could not find inner model, disabling",
            )
            return

        config = getattr(inner, 'config', None)
        if config is None:
            log.warning(
                "TopkAccuracyMonitor: model has no config, disabling",
            )
            return

        use_liger = getattr(config, 'use_liger', False)
        tp_size = getattr(config, 'tp_size', 1) or 1
        if use_liger:
            log.warning(
                "TopkAccuracyMonitor: use_liger=True, logits not available, "
                "disabling",
            )
            return
        if tp_size > 1:
            log.warning(
                "TopkAccuracyMonitor: tp_size>1, logits are vocab-partitioned,"
                " disabling",
            )
            return

        inner._topk_values = list(self.k_values)
        inner._topk_results = None
        inner._compute_topk_this_step = False
        self._inner = inner
        self._enabled = True
        log.info(
            f"TopkAccuracyMonitor enabled with k_values={self.k_values}, "
            f"target={type(inner).__name__}",
        )

    def batch_start(self, state: State, logger: Logger):
        if not self._enabled:
            return
        should_log = (state.timestamp.batch.value + 1) % self.batch_log_interval == 0
        self._inner._compute_topk_this_step = should_log

    def batch_end(self, state: State, logger: Logger):
        if not self._enabled:
            return
        if state.timestamp.batch.value % self.batch_log_interval != 0:
            return

        inner = self._inner
        results = getattr(inner, '_topk_results', None)
        if results is None:
            return

        inner._topk_results = None

        num_tokens = results['num_tokens']
        sp_group = dist.get_sp_group()
        if sp_group is not None:
            dist.all_reduce(num_tokens, reduce_operation='SUM', group=sp_group)
            for k in self.k_values:
                dist.all_reduce(results[k], reduce_operation='SUM', group=sp_group)

        fsdp_group = dist.get_fsdp_group()
        dist.all_reduce(num_tokens, reduce_operation='SUM', group=fsdp_group)
        for k in self.k_values:
            dist.all_reduce(results[k], reduce_operation='SUM', group=fsdp_group)

        total = num_tokens.item()
        if total == 0:
            return

        metrics = {}
        for k in self.k_values:
            metrics[f"token_accuracy/top{k}"] = (results[k].item() / total)

        logger.log_metrics(metrics)
