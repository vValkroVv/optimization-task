# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import json
import os
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Annotated, Any, cast, ClassVar

import torch
import torch.distributed._functional_collectives as funcol
import torch.distributed.checkpoint.stateful
import tyro
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.tensor import DTensor

from torchtitan.callbacks import Callback, CallbackManager, EMA, Event
from torchtitan.distributed.backends import build_sharding_backend
from torchtitan.components.aux_loss import AuxLossTracker
from torchtitan.components.checkpoint import (
    CheckpointManager,
    resolve_checkpoint_load,
    TRAIN_STATE,
)
from torchtitan.components.dataloader import (
    BaseDataLoader,
    build_replicated_dataloader,
    DataloaderExhaustedError,
)
from torchtitan.components.loss import (
    BaseLoss,
    ChunkedCELoss,
    CompactCELoss,
    IGNORE_INDEX,
    MTPLoss,
)
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import ensure_pp_loss_visible, MetricsProcessor
from torchtitan.components.optimizer import (
    OptimizersContainer,
    OptimizersInBackwardContainer,
)
from torchtitan.components.quantization.utils import (
    has_quantization,
    log_float8_model_structure,
)
from torchtitan.components.tokenizer import BaseTokenizer, HuggingFaceTokenizer
from torchtitan.components.validate import BaseValidator, Validator
from torchtitan.config import Configurable, TORCH_DTYPE_MAP
from torchtitan.config.configs import (
    ActivationCheckpointConfig,
    CommConfig,
    CompileConfig,
    DebugConfig,
    FP8Config,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.distributed import full_dtensor, ParallelDims, utils as dist_utils
from torchtitan.distributed.context_parallel import prepare_context_parallel_input
from torchtitan.distributed.data_world import DataWorld

from torchtitan.models.common.attention import AttentionMaskKind, MaskType
from torchtitan.models.common.decoder_blocks import Decoder
from torchtitan.observability import structured_logger as sl
from torchtitan.ops.warmup.deep_gemm_fast_warmup import setup_deep_gemm_cache
from torchtitan.protocols import BaseModel
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.tools import utils
from torchtitan.tools.logging import logger
from torchtitan.tools.profiler import Profiler
from torchtitan.tools.utils import device_capacity_report, move_tensors_to_device


@torch.no_grad()
def cast_frozen_parameters(
    model: torch.nn.Module, *, dtype: torch.dtype = torch.bfloat16
) -> None:
    """Cast frozen floating-point parameters, preserving aliases and buffers.

    Call after freezing and before sharding or optimizer construction. On a meta
    model this selects the eventual allocation dtype without allocating weights.
    """
    for param in model.parameters():
        if not param.requires_grad and param.is_floating_point():
            param.data = param.data.to(dtype=dtype)


class Trainer(torch.distributed.checkpoint.stateful.Stateful, Configurable):
    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        """
        Default container for training configuration.
        """

        # NOTE: model_spec is suppressed from tyro CLI parsing and is always
        # set programmatically by the model registry before Trainer construction
        model_spec: Annotated[ModelSpec | None, tyro.conf.Suppress] = None

        hf_assets_path: str = "./tests/assets/tokenizer"
        """
        Path to HF assets folder. This folder contains local copies of Hugging Face assets,
        including model weights in .safetensors format, the model.safetensor.index.json file
        (fqn to file mapping), the config.json file, generation_config.json, and tokenizer files.
        """

        dump_folder: str = "./outputs"
        """Folder to dump job outputs"""

        profiler: Profiler.Config = field(default_factory=Profiler.Config)
        callbacks: dict[str, Callback.Config] = field(default_factory=dict)
        metrics: MetricsProcessor.Config = field(
            default_factory=MetricsProcessor.Config
        )
        tokenizer: BaseTokenizer.Config = field(
            default_factory=HuggingFaceTokenizer.Config
        )
        dataloader: BaseDataLoader.Config = field(default_factory=BaseDataLoader.Config)
        optimizer: OptimizersContainer.Config = field(
            default_factory=OptimizersContainer.Config
        )
        lr_scheduler: LRSchedulersContainer.Config = field(
            default_factory=LRSchedulersContainer.Config
        )
        training: TrainingConfig = field(default_factory=TrainingConfig)
        parallelism: ParallelismConfig = field(default_factory=ParallelismConfig)
        checkpoint: CheckpointManager.Config = field(
            default_factory=CheckpointManager.Config
        )
        activation_checkpoint: ActivationCheckpointConfig = field(
            default_factory=ActivationCheckpointConfig
        )
        compile: CompileConfig = field(default_factory=CompileConfig)
        comm: CommConfig = field(default_factory=CommConfig)
        validator: Validator.Config = field(default_factory=Validator.Config)
        debug: DebugConfig = field(default_factory=DebugConfig)
        fp8: FP8Config = field(default_factory=FP8Config)
        loss: BaseLoss.Config = field(default_factory=BaseLoss.Config)

        def __post_init__(self):
            if self.debug.batch_invariant:
                raise ValueError(
                    "Batch-invariant mode is not supported in pre-training."
                )
            if isinstance(self.optimizer, OptimizersInBackwardContainer.Config):
                if self.parallelism.expert_parallel_degree > 1:
                    raise NotImplementedError(
                        "Optimizers in backward is not supported with Expert Parallel."
                    )
                if self.parallelism.pipeline_parallel_degree > 1:
                    raise NotImplementedError(
                        "Optimizers in backward is not supported with Pipeline Parallel."
                    )
            if self.comm.mode == "torchcomms" and (
                self.dataloader.use_data_world
                or (self.validator.enable and self.validator.dataloader.use_data_world)
            ):
                raise ValueError(
                    "Dataloaders with use_data_world are not supported with "
                    "comm.mode='torchcomms': torchcomms cannot create the data mesh process group."
                )

        def maybe_log(self) -> None:
            if self.debug.print_config:
                logger.info(
                    f"Running with configs: {json.dumps(self.to_dict(), indent=2, ensure_ascii=False)}"
                )

            if self.debug.save_config_file is not None:
                config_file = os.path.join(
                    self.dump_folder, self.debug.save_config_file
                )
                if torch.distributed.is_initialized():
                    if torch.distributed.get_rank() == 0:
                        os.makedirs(os.path.dirname(config_file), exist_ok=True)
                        with open(config_file, "w") as f:
                            json.dump(self.to_dict(), f, indent=2)
                    logger.info(f"Saved job configs to {config_file}")
                else:
                    logger.warning(
                        "Job configs logging is disabled due to torch.distributed not initialized."
                    )

    # core configs
    config: Config
    parallel_dims: ParallelDims

    # swappable training components
    tokenizer: BaseTokenizer
    dataloader: BaseDataLoader
    model_config: BaseModel.Config
    # TODO: we should make this list[BaseModel / Decoder] but this will affect many components.
    # will do this in a separate PR
    model_parts: list[torch.nn.Module]
    loss_fn: BaseLoss
    optimizers: OptimizersContainer
    lr_schedulers: LRSchedulersContainer
    validator: BaseValidator
    metrics_processor: MetricsProcessor
    checkpointer: CheckpointManager
    callbacks: CallbackManager

    emits_step_callback_events: ClassVar[bool] = True
    """Does ``train_step`` emit STEP_EVENTS. A trainer that overrides
    ``train_step`` without emitting them (Flux) sets this False, and the
    callback manager then refuses callbacks that implement them."""

    # runtime utilities
    device: torch.device
    gc_handler: utils.GarbageCollection
    train_context: dist_utils.TrainContext
    gradient_accumulation_steps: int
    pp_has_first_stage: bool
    pp_has_last_stage: bool

    # additional training states
    step: int
    ntokens_seen: torch.Tensor

    # Enable debug tracing on failure: https://pytorch.org/docs/stable/elastic/errors.html
    @record
    def __init__(self, config: Config):
        torch._C._log_api_usage_once("torchtitan.train")

        self.config = config
        assert config.model_spec is not None, (
            "model_spec must be set before creating Trainer"
        )
        self.sharding_backend = build_sharding_backend(
            config.parallelism, config.model_spec.name
        )
        model_spec = config.model_spec

        device_module, device_type = utils.device_module, utils.device_type
        self.device = torch.device(  # pyrefly: ignore [read-only]
            f"{device_type}:{int(os.environ['LOCAL_RANK'])}"
        )
        # Device has to be set before creating TorchFT manager.
        device_module.set_device(self.device)

        # init distributed and build meshes
        self.parallel_dims = parallel_dims = self.init_distributed()

        # Logging needs to happen after distributed initialized
        config.maybe_log()

        dist_utils.set_pg_timeouts(
            timeout=timedelta(seconds=config.comm.init_timeout_seconds),
            parallel_dims=parallel_dims,
        )

        if parallel_dims.dp_enabled:
            batch_mesh = parallel_dims.get_mesh("batch")
            batch_degree, batch_rank = batch_mesh.size(), batch_mesh.get_local_rank()
        else:
            batch_mesh = None
            batch_degree, batch_rank = 1, 0

        # take control of garbage collection to avoid stragglers
        self.gc_handler = utils.GarbageCollection(
            gc_freq=config.training.gc_freq, debug=config.training.gc_debug
        )

        # Set random seed, and maybe enable deterministic mode
        # (mainly for debugging, expect perf loss).
        dist_utils.set_determinism(
            parallel_dims,
            self.device,
            config.debug,
            distinct_seed_mesh_dims=["pp"],
        )

        # build model (using meta init)
        model_config = model_spec.model
        # set the model args from training job configs
        model_config.update_from_config(
            config=config,
        )
        self.model_config = model_config

        if config.fp8.sparse_use_grouped_gemm:
            setup_deep_gemm_cache(
                int(os.environ["LOCAL_RANK"]),
                model_config,
                parallel_dims.ep,
                config.fp8,
            )

        logger.info(f"Building {model_spec.name} {model_spec.flavor}")

        with (
            torch.device("meta"),
            utils.set_default_dtype(TORCH_DTYPE_MAP[config.training.dtype]),
        ):
            model = model_config.build()
        cast_frozen_parameters(
            model, dtype=TORCH_DTYPE_MAP[config.training.frozen_dtype]
        )

        # Verify all submodules satisfy the Module protocol
        # TODO: move this to module validate().
        # This is current put here to verify module build and
        # converter, which should guanrantee Module protocol.
        # On the other hand, some parallelism wrappers don't
        # have this guanrantee, e.g., fully_shard.
        model.verify_module_protocol()
        if has_quantization(model_config):
            log_float8_model_structure(model)

        # calculate model size and flops per token
        (
            model_param_count,
            self.num_flops_per_token,
        ) = model_config.get_nparams_and_flops(model, config.training.seq_len)

        # metrics logging
        self.metrics_processor = config.metrics.build(
            parallel_dims=parallel_dims,
            dump_folder=config.dump_folder,
            pp_schedule=config.parallelism.pipeline_parallel_schedule,
            config_dict=config.to_dict(),
            num_flops_per_token=self.num_flops_per_token,
            has_quantization=has_quantization(model_config),
        )
        color = self.metrics_processor.color

        logger.info(
            f"{color.blue}Model {model_spec.name} {model_spec.flavor} "
            f"{color.red}size: {model_param_count:,} total parameters{color.reset}"
        )

        # move sharded model to CPU/GPU and initialize weights via DTensor
        buffer_device: torch.device | None
        if config.checkpoint.create_seed_checkpoint:
            init_device = "cpu"
            buffer_device = None
        elif config.training.enable_cpu_offload:
            init_device = "cpu"
            buffer_device = torch.device(device_type)
        else:
            init_device = device_type
            buffer_device = None

        self.loss_fn = config.loss.build(
            compile_config=config.compile,
        )
        if isinstance(self.loss_fn, CompactCELoss):
            if (
                parallel_dims.tp_enabled
                or parallel_dims.cp_enabled
                or parallel_dims.pp_enabled
                or config.parallelism.full_dtensor
            ):
                raise ValueError(
                    "CompactCELoss requires TP=CP=PP=1 and no full_dtensor."
                )
        if config.compile.enable and "model" in config.compile.components:
            if (
                config.compile.scope == "submodules"
                and config.activation_checkpoint.mode in {"selective", "memory_budget"}
            ):
                raise ValueError(
                    "Submodule compile requires activation_checkpoint.mode full or none."
                )

        # verify batch sizes
        global_batch_size = config.training.global_batch_size
        if global_batch_size < 0:
            # This global batch size results in 1 gradient accumulation
            # step.
            global_batch_size = config.training.local_batch_size * batch_degree
        assert global_batch_size > 0
        assert (
            global_batch_size % (config.training.local_batch_size * batch_degree) == 0
        ), (
            f"global batch size must be multiple of local batch size times "
            f"data-parallel degree ({global_batch_size} "
            f"% ({config.training.local_batch_size} * {batch_degree}) != 0)"
        )

        # calculate gradient accumulation steps
        self.gradient_accumulation_steps = global_batch_size // (
            config.training.local_batch_size * batch_degree
        )
        assert self.gradient_accumulation_steps > 0

        # apply parallelisms and initialization
        with sl.log_trace_span("model_parallelism_init"):
            if parallel_dims.pp_enabled:
                if not model_spec.pipelining_fn:
                    raise RuntimeError(
                        f"Pipeline Parallel is enabled but {model_spec.name} "
                        f"does not support pipelining"
                    )

                # apply both Pipeline Parallel and SPMD-style scaling techniques
                (
                    self.pp_schedule,
                    self.model_parts,
                    self.pp_has_first_stage,
                    self.pp_has_last_stage,
                ) = model_spec.pipelining_fn(
                    model,
                    parallel_dims=parallel_dims,
                    training=config.training,
                    parallelism=config.parallelism,
                    compile_config=config.compile,
                    ac_config=config.activation_checkpoint,
                    dump_folder=config.dump_folder,
                    device=self.device,
                    model_config=model_config,
                    parallelize_fn=model_spec.parallelize_fn,
                    loss_fn=self.loss_fn,
                )
                # when PP is enabled, `model` obj is no longer used after this point,
                # model_parts is used instead
                del model

                pp_rank = parallel_dims.get_mesh("pp").get_local_rank()
                assert self.pp_has_first_stage == (pp_rank == 0), (
                    f"PP schedule places first stage off PP axis rank 0 "
                    f"(rank {pp_rank}, has_first_stage={self.pp_has_first_stage}); "
                    f"dataloader owner-only 'input' optimization assumes "
                    f"stage 0 lives on PP axis rank 0"
                )

                for m in self.model_parts:
                    m.to_empty(device=init_device)
                    with torch.no_grad(), self.sharding_backend.weight_init_context(m):
                        # TODO: Change this back to init_weights once
                        # autoparallel contains the wrap_init_states
                        cast(BaseModel, m).init_weights(buffer_device=buffer_device)
                    m.train()

                # confirm that user will be able to view loss metrics on the console
                ensure_pp_loss_visible(
                    parallel_dims=parallel_dims,
                    pp_schedule=config.parallelism.pipeline_parallel_schedule,
                    color=color,
                )
            else:
                if not config.checkpoint.create_seed_checkpoint:
                    # Skip parallelize_fn for seed checkpoints — nothing from
                    # it is needed (AC, compile, nD parallelism, mixed precision, etc.).
                    model = model_spec.parallelize_fn(
                        model,
                        parallel_dims=parallel_dims,
                        training=config.training,
                        parallelism=config.parallelism,
                        compile_config=config.compile,
                        ac_config=config.activation_checkpoint,
                        dump_folder=config.dump_folder,
                    )

                model.to_empty(device=init_device)
                with torch.no_grad(), self.sharding_backend.weight_init_context(model):
                    # TODO: Change this back to init_weights once
                    # autoparallel contains the wrap_init_states
                    cast(BaseModel, model).init_weights(buffer_device=buffer_device)
                model.train()

                self.model_parts = [model]

        sd_adapter = (
            model_spec.state_dict_adapter(model_config, config.hf_assets_path)
            if model_spec.state_dict_adapter
            else None
        )
        initial_load_spec = resolve_checkpoint_load(
            config.checkpoint,
            base_folder=config.dump_folder,
            sd_adapter=sd_adapter,
            step=config.checkpoint.load_step,
        )
        if model_spec.post_model_init_fn is not None and not (
            initial_load_spec.restores_full_model
        ):
            model_spec.post_model_init_fn(
                self.model_parts,
                model_config=model_config,
            )

        # Set lm_head reference for ChunkedCELoss after model construction.
        # Non-PP: single model part always has lm_head.
        # PP: only the last stage has lm_head; non-last stages skip this.
        if isinstance(self.loss_fn, ChunkedCELoss):
            if parallel_dims.pp_enabled:
                if self.pp_has_last_stage:
                    lm_head = self.model_parts[-1].lm_head
                    assert lm_head is not None, (
                        "Last PP stage must have lm_head for ChunkedCELoss"
                    )
                    self.loss_fn.set_lm_head(
                        lm_head  # pyrefly: ignore [bad-argument-type]
                    )
                    self.model_parts[
                        -1
                    ]._skip_lm_head = True  # pyrefly: ignore [bad-argument-type]
            else:
                assert len(self.model_parts) == 1
                lm_head = self.model_parts[0].lm_head
                assert lm_head is not None, "Model must have lm_head for ChunkedCELoss"
                self.loss_fn.set_lm_head(lm_head)  # pyrefly: ignore [bad-argument-type]
                self.model_parts[
                    0
                ]._skip_lm_head = True  # pyrefly: ignore [bad-argument-type]

        # peak flops for MFU calculation, and the model's memory footprint
        gpu_peak_flops = utils.get_peak_flops(
            device_module.get_device_name(device_module.current_device())
        )
        logger.info(f"Peak FLOPS used for computing MFU: {gpu_peak_flops:.3e}")
        memory_metrics, max_reserved_gib = device_capacity_report()
        logger.info(
            f"{device_type.upper()} memory usage for model: "
            f"{max_reserved_gib:.2f}GiB"
            f"({memory_metrics['memory/max_reserved_pct']:.2f}%)"
        )

        # build optimizer after applying parallelisms to the model
        self.optimizers = config.optimizer.build(model_parts=self.model_parts)
        if model_spec.post_optimizer_build_fn is not None:
            model_spec.post_optimizer_build_fn(
                self.optimizers, self.model_parts, parallel_dims
            )
        self.lr_schedulers = config.lr_scheduler.build(
            optimizers=self.optimizers,
            training_steps=config.training.steps,
        )
        self.callbacks = CallbackManager(
            config.callbacks,
            model_parts=self.model_parts,
            sharding_backend=self.sharding_backend,
            emits_step_events=self.emits_step_callback_events,
        )
        self.metrics_processor.optimizers = self.optimizers
        self.metrics_processor.model_parts = self.model_parts

        # One object owns the per-step lifecycle of every term the loss
        # decomposes into: the grad-only aux/z splices of the model parts and
        # the loss object's own components.
        self.aux_loss = AuxLossTracker(
            self.model_parts,
            parallel_dims=parallel_dims,
            device=self.device,
            microbatch_splits=(
                config.training.local_batch_size
                // config.parallelism.pipeline_parallel_microbatch_size
                if parallel_dims.pp_enabled
                else 1
            ),
            loss_fn=self.loss_fn,
        )

        # Initialize trainer states that will be saved in checkpoint.
        # These attributes must be initialized before checkpoint loading.
        self.step = 0
        self.ntokens_seen = torch.zeros((), dtype=torch.int64, device=self.device)
        # Run-local counter: throughput windows use differences across snapshots.
        self.ninput_tokens_seen = torch.zeros_like(self.ntokens_seen)

        # build tokenizer
        self.tokenizer = config.tokenizer.build(tokenizer_path=config.hf_assets_path)

        # build dataloader
        dataloader_kwargs = {
            "tokenizer": self.tokenizer,
            "seq_len": config.training.seq_len,
            "local_batch_size": config.training.local_batch_size,
            "snapshot_every_n_steps": (
                config.checkpoint.interval * self.gradient_accumulation_steps
                if config.checkpoint.enable
                else None
            ),
        }
        if config.dataloader.use_data_world:
            dataloader_kwargs["data_world"] = DataWorld.from_parallel_dims(
                parallel_dims, role="train"
            )
        else:
            dataloader_kwargs.update(dp_world_size=batch_degree, dp_rank=batch_rank)
        # input is consumed only by first PP stage on owner rank
        self.dataloader = build_replicated_dataloader(
            config.dataloader,
            lambda: config.dataloader.build(**dataloader_kwargs),
            parallel_dims=parallel_dims,
            device=self.device,
            owner_only_keys=("input", "modality_inputs", "modality_token_indices"),
        )

        # build checkpointer; reserved-key collisions are the callback
        # manager's own check (stateful_objects raises on them).
        checkpoint_states: dict[str, Any] = {TRAIN_STATE: self}
        checkpoint_states.update(self.callbacks.stateful_objects())
        self.checkpointer = config.checkpoint.build(
            dataloader=self.dataloader,
            model_parts=self.model_parts,
            sharding_backend=self.sharding_backend,
            optimizers=self.optimizers,
            lr_schedulers=self.lr_schedulers,
            states=checkpoint_states,
            sd_adapter=sd_adapter,
            base_folder=config.dump_folder,
            initial_load_spec=initial_load_spec,
        )

        loss_parallel_enabled = (
            parallel_dims.tp_enabled and not config.parallelism.disable_loss_parallel
        )
        self.train_context = dist_utils.get_train_context(loss_parallel_enabled)

        # Build validator if validation is configured
        if config.validator.enable:
            pp_schedule, pp_has_first_stage, pp_has_last_stage = (
                (
                    self.pp_schedule,
                    self.pp_has_first_stage,
                    self.pp_has_last_stage,
                )
                if parallel_dims.pp_enabled
                else (None, None, None)
            )

            self.validator = config.validator.build(
                parallelism=config.parallelism,
                dp_world_size=batch_degree,
                dp_rank=batch_rank,
                tokenizer=self.tokenizer,
                parallel_dims=parallel_dims,
                loss_fn=self.loss_fn,
                validation_context=self.train_context,
                metrics_processor=self.metrics_processor,
                seq_len=config.training.seq_len,
                local_batch_size=config.training.local_batch_size,
                pp_schedule=pp_schedule,
                pp_has_first_stage=pp_has_first_stage,
                pp_has_last_stage=pp_has_last_stage,
            )

        logger.info(
            "Trainer is initialized with "
            f"local batch size {config.training.local_batch_size}, "
            f"global batch size {global_batch_size}, "
            f"gradient accumulation steps {self.gradient_accumulation_steps}, "
            f"sequence length {config.training.seq_len}, "
            f"total steps {config.training.steps} "
            f"(warmup {config.lr_scheduler.warmup_steps})"
        )

    @sl.log_trace_span("torch_distributed_init")
    def init_distributed(self) -> ParallelDims:
        config = self.config
        world_size = dist_utils.init_distributed(
            config.comm,
            enable_cpu_backend=config.training.enable_cpu_offload,
            base_folder=config.dump_folder,
        )

        return ParallelDims.from_config(config.parallelism, world_size)

    def batch_generator(
        self, data_iterable: Iterable[tuple[dict[str, torch.Tensor], torch.Tensor]]
    ) -> Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]]:
        """Returns an iterator that processes batches from the data iterator.

        Note: Tensors may be on CPU or already on the training device.
        """
        data_iterator = iter(data_iterable)

        while True:
            data_load_start = time.perf_counter()
            try:
                batch = next(data_iterator)
            except StopIteration as ex:
                # If data runs out during gradient accumulation, that
                # entire step will not be executed.
                raise DataloaderExhaustedError() from ex
            input_dict, labels = batch
            self.metrics_processor.data_loading_times.append(
                time.perf_counter() - data_load_start
            )

            yield input_dict, labels

    @sl.log_trace_span("post_dataloading_process")
    def post_dataloading_process(
        self, input_dict: dict[str, torch.Tensor], labels: torch.Tensor
    ) -> tuple[
        torch.Tensor | None, torch.Tensor, dict[str, torch.Tensor], dict[str, Any]
    ]:
        """
        Post-processing hook after data loading and before model forward pass.

        This method processes the raw data from the dataloader and prepares it for
        the model's forward pass. It separates the main input tensor from auxiliary
        inputs and constructs additional keyword arguments (e.g., attention masks).

        This method can be overridden in subclasses to customize data processing
        for different training strategies (e.g., converting tensors to DTensors,
        applying custom transformations, etc.).

        Args:
            input_dict: Dictionary containing tensors from dataloader.
                Has an "input" key except on PP ranks without first
                stage. May contain auxiliary inputs (e.g., position ids).
            labels: Target labels for the batch.

        Returns:
            A tuple of (inputs, labels, extra_inputs, extra_kwargs) where:
                - inputs: Main input tensor, or None when the batch has no
                  "input" (PP non-owner).
                - labels: Target labels (unchanged from input parameter).
                - extra_inputs: Dict of auxiliary input tensors from input_dict
                    (excluding "input" and "positions"). These are passed to the
                    model forward but are NOT forwarded across pipeline parallel
                    stages.
                - extra_kwargs: Dict of additional keyword arguments for model
                    forward (positions, attention_masks). These ARE forwarded
                    across all pipeline parallel stages.

        Note:
            The distinction between extra_inputs and extra_kwargs is important for
            pipeline parallelism: extra_kwargs are forwarded to all pipeline stages,
            while extra_inputs are only available to the first stage. Positions
            always go into extra_kwargs so every stage can apply RoPE correctly.
        """
        inputs = input_dict.get("input")
        extra_inputs = {k: v for k, v in input_dict.items() if k != "input"}
        # extra_kwargs are forwarded to all PP stages; extra_inputs are only
        # available to the first stage.  Positions go into extra_kwargs so
        # every stage can apply RoPE correctly.
        extra_kwargs: dict[str, Any] = {}

        positions = extra_inputs.pop("positions", None)
        attention_mask = extra_inputs.pop("attention_mask", None)

        model = self.model_parts[0]
        if isinstance(model, Decoder):
            attn_config = model.attention_config

            if attn_config.mask_type == MaskType.BLOCK_CAUSAL:
                assert positions is not None, (
                    "block_causal mask requires per-document positions from the dataloader"
                )
            elif positions is None:
                # labels match input's [batch, seq] shape and exist on every PP rank
                positions = torch.arange(
                    labels.shape[1], dtype=torch.int32, device=labels.device
                ).repeat(labels.shape[0], 1)

            if attn_config.attention_mask_kind == AttentionMaskKind.BLOCK_MASK:
                extra_kwargs["attention_masks"] = model.get_attention_masks(
                    positions=positions,
                )

        extra_kwargs["positions"] = positions
        extra_kwargs["token_mask"] = (
            labels != IGNORE_INDEX if attention_mask is None else attention_mask
        )

        if self.parallel_dims.cp_enabled:
            inputs, labels, extra_kwargs = prepare_context_parallel_input(
                inputs,
                labels,
                extra_kwargs,
                self.parallel_dims.get_mesh("cp"),
                self.device,
                self.config.parallelism.context_parallel_load_balancer,
            )

        if self.config.parallelism.full_dtensor:
            inputs, labels, extra_kwargs = full_dtensor.parallelize_inputs(
                self.parallel_dims, inputs, labels, extra_kwargs
            )

        return inputs, labels, extra_inputs, extra_kwargs

    @sl.log_trace_span("fwd_bwd")
    def forward_backward_step(
        self,
        *,
        input_dict: dict[str, torch.Tensor],
        labels: torch.Tensor,
        global_valid_tokens: torch.Tensor,
        global_loss_units: torch.Tensor | None = None,
        loss_targets: torch.Tensor | None = None,
        sample_token_counts: torch.Tensor | None = None,
        global_valid_samples: torch.Tensor
        | float
        | list[torch.Tensor | float]
        | None = None,
    ) -> torch.Tensor:
        model_parts = self.model_parts
        parallel_dims = self.parallel_dims

        inputs, labels, extra_inputs, extra_kwargs = self.post_dataloading_process(
            input_dict, labels
        )

        local = labels.to_local() if isinstance(labels, DTensor) else labels
        aux_valid_tokens = (local != IGNORE_INDEX).sum()
        aux_global_tokens = torch.as_tensor(
            global_valid_tokens, device=local.device
        ).clamp_min(1)
        if self.config.training.aux_loss_microbatch_weighting:
            if (
                not isinstance(self.loss_fn, ChunkedCELoss)
                or self.loss_fn.normalization != "samples"
                or self.config.parallelism.full_dtensor
                or parallel_dims.cp != 1
                or parallel_dims.pp_enabled
            ):
                raise ValueError(
                    "training.aux_loss_microbatch_weighting requires sample-normalized "
                    "chunked/compact/MTP CE without PP, CP or full_dtensor."
                )
            # LLMF averages regularizers over microbatches, regardless of
            # supervised answer length.
            aux_valid_tokens = local.new_ones(())
            aux_global_tokens = aux_valid_tokens * (
                self.gradient_accumulation_steps
                * parallel_dims.dp_replicate
                * parallel_dims.dp_shard
            )
        self.aux_loss.weigh(
            valid_tokens=aux_valid_tokens,
            global_valid_tokens=aux_global_tokens,
        )

        # The loss decides what of the batch rides inside the targets.
        labels = self.loss_fn.pack_targets(
            labels if loss_targets is None else loss_targets, extra_kwargs
        )
        loss_denominator = (
            global_valid_tokens if global_loss_units is None else global_loss_units
        )

        if parallel_dims.pp_enabled:
            # Pipeline Parallel forward / backward inside step() call
            loss_kwargs: dict[str, Any] = {"global_valid_tokens": loss_denominator}
            if global_valid_samples is not None:
                loss_kwargs["global_valid_samples"] = global_valid_samples
            with self.train_context():
                targets, losses = (
                    (labels, []) if self.pp_has_last_stage else (None, None)
                )
                if targets is not None:
                    # Arms the shape-probe test; see ``set_step_targets``.
                    self.loss_fn.set_step_targets(targets)
                if self.pp_has_first_stage:
                    assert inputs is not None, (
                        "first PP stage must receive 'input' from the dataloader"
                    )
                    self.pp_schedule.step(
                        inputs,
                        **extra_inputs,
                        **extra_kwargs,
                        target=targets,
                        losses=losses,
                        loss_kwargs=loss_kwargs,
                        return_outputs=False,
                    )
                else:
                    self.pp_schedule.step(
                        **extra_kwargs,
                        target=targets,
                        losses=losses,
                        loss_kwargs=loss_kwargs,
                        return_outputs=False,
                    )
                # Left armed, a call from outside the step -- validation, the
                # next step -- would read as the probe.
                self.loss_fn.set_step_targets(None)

            # accumulate losses across pipeline microbatches
            # TODO: PP+FSDP unexpectedly puts the loss back to the CPU
            if self.pp_has_last_stage:
                assert losses is not None
                # One entry per microbatch. Anything the loss accumulated
                # beyond them is the schedule's first-step gradient-shape
                # probe, which never reaches ``losses`` — matched by tensor
                # identity, since where the probe lands among the real calls
                # is a torch implementation detail.
                self.aux_loss.keep_microbatches(losses)
                # All loss classes scale by global_valid_tokens internally
                loss = torch.sum(torch.stack(losses)).to(self.device)
            else:
                loss = torch.tensor([-1.0], device=self.device)
        else:
            # Non-PP forward / backward
            assert len(model_parts) == 1
            with self.train_context():
                pred = model_parts[0](inputs, **extra_inputs, **extra_kwargs)
                # Under non-full_dtensor, labels stay as plain tensors. See
                # ``cross_entropy_loss`` for why pred must also be plain.
                # Remove once non-full_dtensor is no longer supported.
                if (
                    isinstance(pred, DTensor)
                    and not self.config.parallelism.full_dtensor
                    and self.config.parallelism.disable_loss_parallel
                ):
                    pred = pred.to_local()
                if global_valid_samples is None:
                    loss = self.loss_fn(pred, labels, loss_denominator)
                elif isinstance(self.loss_fn, CompactCELoss):
                    assert not isinstance(global_valid_samples, list)
                    loss = self.loss_fn(pred, labels, global_valid_samples)
                elif isinstance(self.loss_fn, MTPLoss):
                    loss = self.loss_fn(
                        pred, labels, loss_denominator, global_valid_samples
                    )
                else:
                    assert isinstance(self.loss_fn, ChunkedCELoss)
                    assert not isinstance(global_valid_samples, list)
                    loss = self.loss_fn(
                        pred,
                        labels,
                        loss_denominator,
                        global_valid_samples,
                        sample_token_counts,
                    )
                del pred
                loss.backward()

        # The returned loss here is local SUM loss / global_valid_tokens
        return loss

    def loss_this_pp_stage_owns(self, loss: torch.Tensor) -> torch.Tensor | None:
        """``loss`` when this pipeline stage owns it, ``None`` where it does not.

        Under pipeline parallelism a stage that does not own the loss still has
        to return a tensor for the arithmetic below, and that placeholder is a
        ``-1.0``. Handing it to a callback would have it publish -1.0 as a loss
        (``data/microbatch_loss/*`` did exactly that), so the absence is passed
        as an absence.
        """
        if self.parallel_dims.pp_enabled and not self.pp_has_last_stage:
            return None
        return loss

    def train_step(
        self, data_iterator: Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]]
    ):
        self.callbacks.emit(Event.batch_start, self)
        self.optimizers.zero_grad()
        self.aux_loss.begin_step()
        # Save the current step learning rate for logging
        lr = self.lr_schedulers.schedulers[0].get_last_lr()[0]

        # Keep these variables local to shorten the code as these are
        # the major variables that are used in the training loop.
        parallel_dims = self.parallel_dims

        # Collect all microbatches and count total valid tokens
        microbatches = []
        microbatch_valid_tokens = []
        local_input_tokens = 0
        microbatch_loss_units = []
        microbatch_valid_samples = []
        sample_normalization = (
            isinstance(self.loss_fn, ChunkedCELoss)
            and self.loss_fn.normalization == "samples"
        )
        for _microbatch in range(self.gradient_accumulation_steps):
            with sl.log_trace_span("fetching_batch"):
                self.callbacks.emit(Event.before_dataloader, self)
                input_dict, labels = next(data_iterator)
                # Batch mutators may replace these fields; the step trains on
                # what comes back.
                batch_state = self.callbacks.emit(
                    Event.after_dataloader, self, input_dict=input_dict, labels=labels
                )
                assert batch_state.input_dict is not None
                assert batch_state.labels is not None
                input_dict, labels = dict(batch_state.input_dict), batch_state.labels
                self.metrics_processor.ntokens_since_last_log += labels.numel()
                local_input_tokens += input_dict["input"].numel()
                microbatch_valid_tokens.append((labels != IGNORE_INDEX).sum())
                targets, loss_units, valid_samples, sample_token_counts = (
                    self.loss_fn.prepare_batch(
                        labels,
                        input_dict.pop("sample_token_counts", None),
                        positions=input_dict.get("positions"),
                        packing_mode=cast(str, input_dict.pop("packing_mode", "none")),
                    )
                )
                microbatch_loss_units.append(loss_units)
                if valid_samples is not None:
                    microbatch_valid_samples.append(valid_samples)
                microbatches.append((input_dict, labels, targets, sample_token_counts))
        # Reduce real token counts, compact loss units and each MTP head's
        # nonempty rows together, without a host sync in the optimizer hot path.
        count_columns = [
            torch.stack(microbatch_valid_tokens).sum().reshape(1),
            torch.stack(microbatch_loss_units).sum().reshape(1),
        ]
        if sample_normalization:
            count_columns.append(torch.stack(microbatch_valid_samples).sum(dim=0))
        # Reuse the loss-count collective; input positions include ignored labels
        # and padding, matching Composer DataSpec input_ids.numel().
        count_columns.append(torch.full_like(count_columns[0], local_input_tokens))
        local_counts = torch.cat(count_columns).to(self.device)
        local_valid_tokens, local_loss_units = local_counts[:2].unbind()

        if parallel_dims.dp_enabled:
            batch_mesh = parallel_dims.get_mesh("batch")
            global_counts = cast(
                torch.Tensor,
                cast(Any, funcol.all_reduce(local_counts, "SUM", batch_mesh)).wait(),
            )
        else:
            global_counts = local_counts
        global_input_tokens = global_counts[-1]
        global_counts = global_counts[:-1]
        local_counts = local_counts[:-1]
        global_valid_tokens, global_loss_units = global_counts[:2].unbind()

        global_valid_samples: (
            torch.Tensor | float | list[torch.Tensor | float] | None
        ) = None
        local_sample_denominators = None
        if sample_normalization:
            # Empty shifted MTP heads (and fully ignored sample batches) retain
            # the sample-loss contract: finite zero with a clamped denominator.
            if isinstance(self.loss_fn, MTPLoss):
                local_sample_denominators = local_counts[2:]
                global_valid_samples = list(
                    global_counts[2:].float().clamp_min(1).unbind()
                )
            else:
                local_sample_denominators = local_counts[2]
                global_valid_samples = global_counts[2].float().clamp_min(1)
        else:
            # Compact/token loss rejects an empty global step. Empty local
            # ranks still contribute zero to the reduction.
            torch._assert_async(
                torch.gt(global_loss_units, 0),
                "No supervised loss units in the global batch",
            )

        # Process each microbatch: move to GPU, forward/backward, then free
        accumulated_losses = []
        reduce_setters = []
        if (
            self.config.training.defer_hsdp_all_reduce
            and parallel_dims.dp_replicate > 1
            and len(microbatches) > 1
        ):
            for part in self.model_parts:
                setter = getattr(part, "set_requires_all_reduce", None)
                if not callable(setter):
                    raise ValueError(
                        "Deferred HSDP reduction requires FSDP2 model roots."
                    )
                reduce_setters.append(setter)
        try:
            for microbatch_idx, (
                input_dict,
                labels,
                targets,
                sample_token_counts,
            ) in enumerate(microbatches):
                for setter in reduce_setters:
                    setter(microbatch_idx + 1 == len(microbatches))
                self.callbacks.emit(
                    Event.microbatch_start, self, microbatch_idx=microbatch_idx
                )
                # Move every tensor in the nested batch contract to the model device.
                non_blocking = self.config.training.non_blocking_transfer
                input_dict = move_tensors_to_device(
                    input_dict, self.device, non_blocking=non_blocking
                )
                labels = labels.to(self.device, non_blocking=non_blocking)
                targets = (
                    None
                    if targets is None
                    else targets.to(self.device, non_blocking=non_blocking)
                )

                sample_token_counts = (
                    None
                    if sample_token_counts is None
                    else sample_token_counts.to(self.device, non_blocking=non_blocking)
                )
                loss = self.forward_backward_step(
                    input_dict=input_dict,
                    labels=labels,
                    global_valid_tokens=global_valid_tokens,
                    global_loss_units=global_loss_units,
                    loss_targets=targets,
                    sample_token_counts=sample_token_counts,
                    global_valid_samples=global_valid_samples,
                )
                self.callbacks.emit(
                    Event.after_forward,
                    self,
                    input_dict=input_dict,
                    labels=labels,
                    loss=self.loss_this_pp_stage_owns(loss),
                    global_valid_tokens=global_valid_tokens,
                    microbatch_idx=microbatch_idx,
                )
                accumulated_losses.append(loss.detach())
                del input_dict, labels, targets, loss
        finally:
            for setter in reduce_setters:
                setter(True)

        with sl.log_trace_span("optim"):
            self.callbacks.emit(
                Event.before_optimizer_step,
                self,
                global_valid_tokens=global_valid_tokens,
            )
            grad_norm = dist_utils.clip_grad_norm_(
                [p for m in self.model_parts for p in m.parameters()],
                self.config.training.max_norm,
                foreach=True,
                pp_mesh=parallel_dims.get_optional_mesh("pp"),
                ep_enabled=parallel_dims.ep_enabled,
            )
            self.checkpointer.maybe_wait_for_staging()
            self.optimizers.step()
            self.lr_schedulers.step()
            self.ntokens_seen.add_(global_valid_tokens)
            self.ninput_tokens_seen.add_(global_input_tokens)
            self.callbacks.emit(
                Event.after_optimizer_step,
                self,
                grad_norm=grad_norm,
                global_valid_tokens=global_valid_tokens,
            )

        # Reduce the data collected over gradient accumulation steps.
        loss = torch.sum(torch.stack(accumulated_losses))

        self.callbacks.emit(
            Event.batch_end,
            self,
            loss=self.loss_this_pp_stage_owns(loss),
            grad_norm=grad_norm,
            global_valid_tokens=global_valid_tokens,
        )

        # log metrics
        if not self.metrics_processor.should_log(self.step):
            return

        with sl.log_trace_span("collect_dist_metrics"):
            # int() forces a device sync, so both reads stay behind should_log
            sl.log_trace_scalar(
                {
                    "local_valid_tokens": int(local_valid_tokens),
                    "global_valid_tokens": int(global_valid_tokens),
                }
            )

            if parallel_dims.dp_cp_enabled:
                loss = loss.detach()
                loss_mesh = parallel_dims.get_optional_mesh("loss")

                # For global_avg_loss, we want the average loss across all ranks:
                # loss = local_loss_sum / loss_denominator
                # global_avg_loss = sum(local_loss_sum) / loss_denominator
                #                 = sum(loss)
                #
                # For global_max_loss, we want the max of local average losses across ranks:
                # local_avg_loss = local_loss_sum / local_loss_denominator
                #                = (loss * loss_denominator) / local_loss_denominator
                # global_max_loss = max(local_avg_loss)
                global_avg_loss = dist_utils.dist_sum(loss, loss_mesh)
                if sample_normalization and isinstance(self.loss_fn, MTPLoss):
                    # Each head has its own denominator, so the combined MTP
                    # objective cannot be rescaled into one exact local mean.
                    global_max_loss = None
                else:
                    if sample_normalization:
                        assert isinstance(global_valid_samples, torch.Tensor)
                        assert isinstance(local_sample_denominators, torch.Tensor)
                        loss_denominator = global_valid_samples
                        local_loss_denominator = local_sample_denominators.clamp_min(1)
                    else:
                        loss_denominator = global_loss_units
                        local_loss_denominator = local_loss_units.clamp_min(1)
                    local_avg_loss = loss * loss_denominator / local_loss_denominator
                    global_max_loss = dist_utils.dist_max(local_avg_loss, loss_mesh)
            else:
                global_avg_loss = global_max_loss = float(loss.detach().item())

            global_ntokens_seen = int(self.ntokens_seen.item())

        extra_metrics = {
            "tokens/seen": global_ntokens_seen,
            # Always published: TrainerSource's per-group lr series complement
            # this tag, they do not replace it.
            "lr": lr,
        }

        extra_metrics |= self.aux_loss.metrics()
        self.metrics_processor.log(
            self.step,
            global_avg_loss,
            global_max_loss,
            float(grad_norm.item()),
            extra_metrics=extra_metrics,
        )

    @contextlib.contextmanager
    def _use_ema_weights(self) -> Iterator[list[torch.nn.Module]]:
        """Run the body on the averaged weights when the ``EMA`` callback is on.

        The callback owns the swap on every backend; without it this is a
        no-op context over the live weights.
        """
        ema_callback = self.callbacks.find(EMA)
        if ema_callback is None:
            yield self.model_parts
            return

        try:
            ema_callback.swap_weights()
            yield self.model_parts
        finally:
            ema_callback.swap_weights()

    @record
    def train(self):
        config = self.config

        sl.log_trace_instant("training_start")

        self.sharding_backend.prepare_for_checkpoint_load(self.model_parts)
        self.checkpointer.load(step=config.checkpoint.load_step)

        if self.device.type == "cuda":
            # Release initialization cache once before training.
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()

        self.callbacks.emit(Event.after_load, self)
        # The resolved order is part of the run's record: who runs when, and
        # which of them may change the step.
        logger.info(self.callbacks.describe())
        self.metrics_processor.log_text(
            "callbacks/schedule", self.callbacks.describe(), self.step
        )
        self.callbacks.emit(Event.fit_start, self)

        # Capture loaded step for relative_step calculation.
        # After checkpoint load: self.step = restored step (e.g. 100), or 0 if fresh.
        loaded_step = self.step

        logger.info(f"Training starts at step {self.step + 1}")

        with config.profiler.build(
            global_step=self.step,
            base_folder=config.dump_folder,
        ) as profiler:
            data_iterator = self.batch_generator(self.dataloader)
            while self.should_continue_training():
                self.step += 1
                sl.set_step(self.step, relative_step=self.step - loaded_step)

                with sl.log_trace_span("step"):
                    self.gc_handler.run(self.step)

                    try:
                        self.train_step(data_iterator)
                    except DataloaderExhaustedError:
                        logger.warning("Ran out of data; last step was canceled.")
                        break

                    if self.checkpointer.save(
                        self.step,
                        last_step=(self.step == config.training.steps),
                    ):
                        self.callbacks.emit(Event.after_checkpoint_save, self)

                    # Run validation if validator is available
                    if self.config.validator.enable and self.validator.should_validate(
                        self.step
                    ):
                        self.callbacks.emit(Event.validation_start, self)
                        try:
                            # Validate the averaged weights when EMA is on;
                            # a no-op context otherwise.
                            with self._use_ema_weights() as validation_model_parts:
                                self.validator.validate(
                                    validation_model_parts, self.step
                                )
                        finally:
                            self.callbacks.emit(Event.validation_end, self)

                    # signal the profiler that the next profiling step has started
                    profiler.step()

                    # Reduce timeout after the first train step of THIS process
                    # (assuming lazy init and compilation are finished). Use the
                    # relative step so this fires on resumed runs too.
                    if self.step - loaded_step == 1:
                        dist_utils.set_pg_timeouts(
                            timeout=timedelta(
                                seconds=config.comm.train_timeout_seconds
                            ),
                            parallel_dims=self.parallel_dims,
                        )

        if torch.distributed.get_rank() == 0:
            logger.info("Sleeping 2 seconds for other ranks to complete")
            time.sleep(2)

        self.callbacks.emit(Event.fit_end, self)
        logger.info("Training completed")

    def should_continue_training(self) -> bool:
        return self.step < self.config.training.steps

    def state_dict(self) -> dict[str, Any]:
        return {"step": self.step, "ntokens_seen": int(self.ntokens_seen.item())}

    def load_state_dict(self, state_dict: dict[str, Any]):
        self.step = state_dict["step"]
        self.ntokens_seen.fill_(state_dict["ntokens_seen"])

    def close(self) -> None:
        if hasattr(self, "callbacks") and self.callbacks:
            self.callbacks.close()
        if hasattr(self, "checkpointer") and self.checkpointer:
            self.checkpointer.close()
        if hasattr(self, "metrics_processor") and self.metrics_processor:
            self.metrics_processor.close()
        if hasattr(self, "dataloader") and self.dataloader:
            self.dataloader.close()
