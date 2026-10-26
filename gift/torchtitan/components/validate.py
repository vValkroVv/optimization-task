# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Any, TypeAlias

import torch
import torch.nn as nn
from torch.distributed.pipelining.schedules import _PipelineSchedule

from torchtitan.config.schema.components.validate import (
    BaseValidatorConfig,
    ValidatorConfig,
)
from torchtitan.components.dataloader import (
    BaseDataLoader,
    build_replicated_dataloader,
)
from torchtitan.components.loss import IGNORE_INDEX, LossFunction
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.config import Configurable, ParallelismConfig
from torchtitan.distributed import full_dtensor, ParallelDims, utils as dist_utils
from torchtitan.distributed.context_parallel import prepare_context_parallel_input
from torchtitan.distributed.data_world import DataWorld
from torchtitan.models.common.attention import AttentionMaskKind, MaskType
from torchtitan.models.common.decoder_blocks import Decoder
from torchtitan.observability import structured_logger as sl
from torchtitan.tools import utils
from torchtitan.tools.logging import logger

ValidationContext: TypeAlias = Callable[[], AbstractContextManager[None]]


class BaseValidator(Configurable):
    Config: TypeAlias = BaseValidatorConfig

    def __init__(
        self,
        config: Config,
        **kwargs,
    ):
        self.config = config

    def validate(self, model_parts: list[nn.Module], step: int) -> None:
        raise NotImplementedError("validate method not implemented")

    def should_validate(self, step: int) -> bool:
        return step == 1 or step % self.config.freq == 0


class Validator(BaseValidator):
    """
    Simple validator focused on correctness and integration.

    Args:
        config: Validator.Config configuration
        parallelism: ParallelismConfig configuration
        dp_world_size: Data parallel world size
        dp_rank: Data parallel rank
        tokenizer: Tokenizer
        parallel_dims: Parallel dimensions
        loss_fn: Loss function to use for validation
        validation_context: Context manager for validation
        metrics_processor: Metrics processor
        pp_schedule: Pipeline schedule (optional)
        pp_has_first_stage: Whether this rank has the first PP stage (optional)
        pp_has_last_stage: Whether this rank has the last PP stage (optional)
    """

    # pyrefly: ignore [bad-override]
    Config: TypeAlias = ValidatorConfig

    # TODO: improve the constructor signature
    def __init__(
        self,
        config: Config,
        *,
        parallelism: ParallelismConfig,
        dp_world_size: int,
        dp_rank: int,
        tokenizer: BaseTokenizer,
        parallel_dims: ParallelDims,
        loss_fn: LossFunction,
        validation_context: ValidationContext,
        metrics_processor: MetricsProcessor,
        seq_len: int,
        local_batch_size: int,
        pp_schedule: _PipelineSchedule | None = None,
        pp_has_first_stage: bool | None = None,
        pp_has_last_stage: bool | None = None,
        **kwargs,
    ):
        super().__init__(config=config)
        self.parallelism = parallelism
        self.tokenizer = tokenizer
        self.parallel_dims = parallel_dims
        self.loss_fn = loss_fn
        self.dl_config = config.dataloader._copy(infinite=config.steps != -1)
        self.dp_world_size = dp_world_size
        self.dp_rank = dp_rank
        self.seq_len = seq_len
        self.local_batch_size = local_batch_size
        self.validation_context = validation_context
        self.metrics_processor = metrics_processor
        self.pp_schedule = pp_schedule
        self.pp_has_first_stage = pp_has_first_stage
        self.pp_has_last_stage = pp_has_last_stage
        self._validation_dataloader: BaseDataLoader | None = None
        self._reuse_validation_dataloader = self.dl_config.supports_reiteration
        if self._reuse_validation_dataloader:
            self.dl_config.validate_for_validation_loop()

        if config.steps == -1:
            logger.warning(
                "Setting validation steps to -1 might cause hangs because of "
                "unequal sample counts across ranks when dataset is exhausted."
            )

    def post_dataloading_process(
        self,
        input_dict: dict[str, torch.Tensor],
        labels: torch.Tensor,
        model_parts: list[nn.Module],
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
        """
        Post-processing hook after data loading and before model forward pass.

        This method processes the raw data from the dataloader and prepares it for
        the model's forward pass. It separates the main input tensor from auxiliary
        inputs and constructs additional keyword arguments (e.g., attention masks).

        Args:
            input_dict: Dictionary containing tensors from dataloader
                Has an "input" key except on PP ranks without first
                stage. May contain auxiliary inputs (e.g. position ids)
            labels: Target labels for the batch.
            model_parts: List of model parts for accessing model methods.

        Returns:
            A tuple of (inputs, labels, extra_inputs, extra_kwargs) where:
                - inputs: Main input tensor, or None when the batch has no
                  "input" (PP non-owner).
                - labels: Target labels (potentially modified by CP sharding).
                - extra_inputs: Dict of auxiliary input tensors (all keys except
                    "input" from input_dict). These are passed to the model forward
                    but are NOT forwarded across pipeline parallel stages.
                - extra_kwargs: Dict of additional keyword arguments for model forward.
                    These ARE forwarded across pipeline parallel stages. Contains
                    attention_masks if flex attention is enabled.

        Note:
            The distinction between extra_inputs and extra_kwargs is important for
            pipeline parallelism: extra_kwargs are forwarded to all pipeline stages,
            while extra_inputs are only available to the first stage.
        """
        inputs = input_dict.get("input")
        extra_inputs = {k: v for k, v in input_dict.items() if k != "input"}
        # For arguments, like attention_masks, we have to put them in a separate
        # dict as extra_inputs are not forwarded to other stages in PP, but
        # extra_kwargs are.
        extra_kwargs: dict[str, Any] = {}

        # TODO: deduplicate with Trainer.post_dataloading_process which has
        # the same logic; extract a shared function to prevent further drift.
        # Resolve positions once: per-document positions for block_causal,
        # sequential positions when CP needs them for shard indexing,
        # or None (model uses sequential RoPE slice by default).
        positions = extra_inputs.pop("positions", None)
        model = model_parts[0]
        if isinstance(model, Decoder):
            attn_config = model.attention_config

            if attn_config.mask_type == MaskType.BLOCK_CAUSAL:
                assert positions is not None, (
                    "block_causal mask requires per-document positions from the dataloader"
                )
            else:
                # labels match the input's [batch, seq] shape and exist on every PP rank
                positions = torch.arange(
                    labels.shape[1], dtype=torch.int32, device=labels.device
                ).repeat(labels.shape[0], 1)

            if attn_config.attention_mask_kind == AttentionMaskKind.BLOCK_MASK:
                extra_kwargs["attention_masks"] = model.get_attention_masks(
                    positions=positions,
                )

        extra_kwargs["positions"] = positions
        extra_kwargs["token_mask"] = labels != IGNORE_INDEX

        if self.parallel_dims.cp_enabled:
            inputs, labels, extra_kwargs = prepare_context_parallel_input(
                inputs,
                labels,
                extra_kwargs,
                self.parallel_dims.get_mesh("cp"),
                labels.device,
                self.parallelism.context_parallel_load_balancer,
            )

        if self.parallelism.full_dtensor:
            inputs, labels, extra_kwargs = full_dtensor.parallelize_inputs(
                self.parallel_dims, inputs, labels, extra_kwargs
            )

        return inputs, labels, extra_inputs, extra_kwargs

    @sl.log_trace_span("eval")
    @torch.no_grad()
    def validate(
        self,
        model_parts: list[nn.Module],
        step: int,
    ) -> None:
        sl.add_step_tag("eval")
        # Set model to eval mode
        for model in model_parts:
            model.eval()

        parallel_dims = self.parallel_dims

        accumulated_losses = []
        device_type = utils.device_type
        num_steps = 0

        # rebuilding a streaming dataset every cycle leaks its shm registration
        validation_dataloader = self._validation_dataloader
        if validation_dataloader is None:
            dataloader_kwargs = {
                "tokenizer": self.tokenizer,
                "seq_len": self.seq_len,
                "local_batch_size": self.local_batch_size,
            }
            if self.dl_config.use_data_world:
                dataloader_kwargs["data_world"] = DataWorld.from_parallel_dims(
                    parallel_dims, role="validation"
                )
            else:
                dataloader_kwargs.update(
                    dp_world_size=self.dp_world_size, dp_rank=self.dp_rank
                )
            # input is consumed only by first PP stage on owner rank
            validation_dataloader = build_replicated_dataloader(
                self.dl_config,
                lambda: self.dl_config.build(**dataloader_kwargs),
                parallel_dims=parallel_dims,
                device=torch.device(device_type),
                owner_only_keys=("input",),
            )
            if self._reuse_validation_dataloader:
                self._validation_dataloader = validation_dataloader

        for input_dict, labels in validation_dataloader:
            # pyrefly: ignore [missing-attribute, unsupported-operation]
            if self.config.steps != -1 and num_steps >= self.config.steps:
                break

            for k, v in input_dict.items():
                input_dict[k] = v.to(device_type)
            labels = labels.to(device_type)

            # Count valid tokens for this batch
            local_valid_tokens = torch.tensor(0, dtype=torch.int64, device=device_type)
            local_valid_tokens += (labels != IGNORE_INDEX).sum()

            # All-reduce token count across DP ranks to get global token count
            if parallel_dims.dp_enabled:
                batch_mesh = parallel_dims.get_mesh("batch")
                global_valid_tokens = dist_utils.dist_sum(
                    local_valid_tokens, batch_mesh, None
                )
            else:
                global_valid_tokens = local_valid_tokens.float()

            # Process data (extract inputs, handle attention masks, CP sharding)
            inputs, labels, extra_inputs, extra_kwargs = self.post_dataloading_process(
                input_dict, labels, model_parts
            )

            if parallel_dims.pp_enabled:
                assert self.pp_schedule is not None
                assert self.pp_has_first_stage is not None
                assert self.pp_has_last_stage is not None
                # Pipeline Parallel forward inside eval() call
                with self.validation_context():
                    targets, losses = (
                        (labels, []) if self.pp_has_last_stage else (None, None)
                    )
                    if self.pp_has_first_stage:
                        assert inputs is not None, (
                            "first PP stage must receive 'input' from the dataloader"
                        )
                        self.pp_schedule.eval(
                            inputs,
                            **extra_inputs,
                            **extra_kwargs,
                            target=targets,
                            losses=losses,
                        )
                    else:
                        self.pp_schedule.eval(
                            **extra_kwargs,
                            target=targets,
                            losses=losses,
                        )

                # accumulate losses across pipeline microbatches
                # TODO: PP+FSDP unexpectedly puts the loss back to the CPU
                if self.pp_has_last_stage:
                    assert losses is not None
                    # using sum because loss_fn already uses reduction='sum'
                    loss_sum = torch.sum(torch.stack(losses)).to(device_type)
                else:
                    loss_sum = torch.tensor([-1.0], device=device_type)
            else:
                with self.validation_context():
                    assert len(model_parts) == 1
                    predictions = model_parts[0](inputs, **extra_inputs, **extra_kwargs)
                    loss_sum = self.loss_fn(predictions, labels)

            accumulated_losses.append(loss_sum.detach() / global_valid_tokens)
            num_steps += 1

        # Compute average loss
        loss = torch.sum(torch.stack(accumulated_losses))
        loss /= num_steps
        if parallel_dims.dp_cp_enabled:
            global_avg_loss = dist_utils.dist_sum(
                loss, parallel_dims.get_optional_mesh("loss")
            )
        else:
            global_avg_loss = float(loss.item())

        self.metrics_processor.log_validation(loss=global_avg_loss, step=step)

        # Set model back to train mode
        for model in model_parts:
            model.train()
