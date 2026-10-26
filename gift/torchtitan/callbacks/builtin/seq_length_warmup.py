import torch

from typing import TypeAlias

from torchtitan.config.schema.callbacks.builtin.seq_length_warmup import (
    SeqLengthWarmupConfig,
)
from torchtitan.callbacks.base import MutableState, Mutation, Mutator
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.config import Configurable


class SeqLengthWarmup(Configurable, Mutator):
    """Warms the training sequence length up from ``min_seq_len``.

    The scheduled length grows linearly from ``min_seq_len`` to the training
    sequence length (optionally capped by ``max_seq_len``) over
    ``warmup_steps`` steps. Every microbatch
    is truncated on tensor dimension 1. Pipeline parallelism is not supported: per-step shape changes break
    pipeline stage IO buffers.

    Example:
        ``callbacks["seq_length_warmup"] = SeqLengthWarmup.Config(emit_metrics=True)``

    Logged metrics when ``emit_metrics`` is enabled (once per train step):
        - ``seq_length_warmup/curr_seq_len``: scheduled sequence length
        - ``seq_length_warmup/seq_len_ratio``: curr_seq_len / training seq_len
        - ``seq_length_warmup/valid_tokens_before``: supervised label tokens in
          this rank's microbatches before truncation
        - ``seq_length_warmup/valid_tokens_after``: same, after truncation
        - ``seq_length_warmup/valid_token_ratio_after``: valid_tokens_after
          divided by the number of label positions after truncation
    """

    # Replaces input_dict/labels in after_dataloader.
    mutates = frozenset({Mutation.batch})

    # pyrefly: ignore [bad-override]
    Config: TypeAlias = SeqLengthWarmupConfig

    def __init__(self, config: Config) -> None:
        self.config = config
        self._reset_step_counters()

    def fit_start(self, state: MutableState) -> None:
        if state.trainer.parallel_dims.pp_enabled:
            raise ValueError(
                "SeqLengthWarmup does not support pipeline parallelism: "
                "changing the sequence length between steps breaks pipeline "
                "stage IO buffers."
            )
        divisor = state.trainer.parallel_dims.seq_len_divisor
        max_seq_len = self._max_seq_len(state.trainer.config.training.seq_len)

        if self.config.min_seq_len % divisor:
            raise ValueError(
                f"SeqLengthWarmup min_seq_len {self.config.min_seq_len} must be "
                f"a multiple of the TP/CP sequence length divisor {divisor}."
            )
        if max_seq_len % divisor:
            raise ValueError(
                f"SeqLengthWarmup max sequence length {max_seq_len} must be "
                f"a multiple of the TP/CP sequence length divisor {divisor}."
            )

    def _max_seq_len(self, default_seq_len: int) -> int:
        if self.config.max_seq_len is None:
            return default_seq_len
        return min(self.config.max_seq_len, default_seq_len)

    def current_seq_len(
        self,
        *,
        step: int,
        default_seq_len: int,
        seq_len_divisor: int = 1,
    ) -> int:
        max_seq_len = self._max_seq_len(default_seq_len)
        if self.config.warmup_steps <= 0 or step >= self.config.warmup_steps:
            return max_seq_len

        progress = step / self.config.warmup_steps
        grown = int(progress * (max_seq_len - self.config.min_seq_len))
        return self.config.min_seq_len + grown - grown % seq_len_divisor

    def after_dataloader(self, state: MutableState) -> None:
        if state.input_dict is None or state.labels is None:
            return
        trainer = state.trainer
        seq_len = self.current_seq_len(
            step=state.step,
            default_seq_len=trainer.config.training.seq_len,
            seq_len_divisor=trainer.parallel_dims.seq_len_divisor,
        )
        state.input_dict, state.labels = self.truncate(
            seq_len, state.input_dict, state.labels
        )

    def truncate(
        self,
        seq_len: int,
        input_dict: dict[str, torch.Tensor],
        labels: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        count_tokens = self.config.emit_metrics
        if count_tokens:
            # Device-side accumulation: .item() would host-sync per microbatch.
            self._valid_tokens_before = (
                self._valid_tokens_before + (labels != IGNORE_INDEX).sum()
            )
            self._current_seq_len = seq_len

        if seq_len < labels.shape[1]:
            input_dict = {
                key: value[:, :seq_len, ...]
                if (
                    isinstance(value, torch.Tensor)
                    and value.ndim >= 2
                    and value.shape[1] >= seq_len
                )
                else value
                for key, value in input_dict.items()
            }
            labels = labels[:, :seq_len]

        if count_tokens:
            self._valid_tokens_after = (
                self._valid_tokens_after + (labels != IGNORE_INDEX).sum()
            )
            self._label_positions_after += int(labels.numel())
        return input_dict, labels

    def batch_end(self, state: MutableState) -> None:
        if not self.config.emit_metrics or self._current_seq_len == 0:
            return
        if state.step % self.config.log_interval == 0:
            default_seq_len = state.trainer.config.training.seq_len
            valid_before, valid_after = torch.stack(
                [
                    torch.as_tensor(self._valid_tokens_before),
                    torch.as_tensor(self._valid_tokens_after),
                ]
            ).tolist()
            state.record_all(
                {
                    "seq_length_warmup/curr_seq_len": self._current_seq_len,
                    "seq_length_warmup/seq_len_ratio": self._current_seq_len
                    / default_seq_len,
                    "seq_length_warmup/valid_tokens_before": valid_before,
                    "seq_length_warmup/valid_tokens_after": valid_after,
                    "seq_length_warmup/valid_token_ratio_after": valid_after
                    / max(self._label_positions_after, 1),
                },
            )
        self._reset_step_counters()

    def _reset_step_counters(self) -> None:
        self._valid_tokens_before = 0
        self._valid_tokens_after = 0
        self._label_positions_after = 0
        self._current_seq_len = 0
