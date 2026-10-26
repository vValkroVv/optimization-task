# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Shared configuration dataclasses for torchtitan.

Some configs live near their owner instead of here:
  - Profiler.Config                 (in tools/profiler.py)
  - OptimizersContainer.Config      (in components/optimizer.py)
  - LRSchedulersContainer.Config    (in components/lr_scheduler.py)
  - MetricsProcessor.Config         (in components/metrics.py)
  - CheckpointManager.Config        (in components/checkpoint.py)

Configs without a clear single owner (or with circular-import constraints)
live here.
"""

from dataclasses import dataclass, field
from typing import Literal
import re

from torchtitan.config.schema.distributed.parallelism import (
    validate_expert_parallel_degree,
)


@dataclass(kw_only=True, slots=True)
class TrainingConfig:
    local_batch_size: int = 8
    """Local batch size (i.e., per-device batch size)"""

    global_batch_size: int = -1
    """
    Global batch size (defaults to `training.local_batch_size * data-parallel degree`)
    """

    seq_len: int = 2048
    """Positive sequence length."""

    max_norm: float | int = 1.0
    """Max norm for gradient clipping"""

    steps: int = 10000
    """Positive number of training steps."""

    enable_cpu_offload: bool = False
    """
    Whether to apply CPU offloading of parameters, gradients, and optimizer states in FSDP
    """

    dtype: Literal["bfloat16", "float32"] = "float32"
    """
    torch dtype for training. In contrast to mixed precision training, setting training_dtype=bfloat16 will
    put all parameters, gradients, and optimizer states in bfloat16, without an extra copy of fp32 weights.
    In the case of full bf16 training, RoPE calculations and logits will still be in fp32.
    """

    mixed_precision_param: Literal["bfloat16", "float32"] = "bfloat16"
    """
    torch dtype to use for parameters when applying mixed precision via fully_shard or torch.autocast.
    This feature takes effect via fully_shard when data_parallel_shard_degree > 1 or
    context_parallel_degree > 1; it takes effect via torch.autocast when data_replicate_degree >= 1
    and no other parallelism is enabled, i.e. under DDP or single-device training.
    """

    mixed_precision_reduce: Literal["float32"] = "float32"
    """
    torch dtype to use for reductions when applying mixed precision via FSDP.
    This feature only takes effect when data_parallel_shard_degree > 1
    """

    gc_freq: int = 50
    """Python garbage control scheduling interval, in steps"""

    gc_debug: bool = False
    """
    Enable GC debugging mode. This will perform gc.collect() at every step to
    detect if there is a reference cycle that includes a CUDA Tensor.
    Note that you may want to lower the training steps to avoid generating too
    many temporary files.
    """

    def __post_init__(self) -> None:
        from torchtitan.config.errors import ConfigFieldError

        if self.local_batch_size <= 0:
            raise ValueError("local_batch_size must be positive")
        if self.global_batch_size == 0:
            raise ConfigFieldError(
                "global_batch_size", "must be positive or negative for automatic sizing"
            )
        for name in ("steps", "seq_len"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ConfigFieldError(name, f"{name} must be a positive integer")
        if (
            self.global_batch_size > 0
            and self.global_batch_size % self.local_batch_size
        ):
            raise ValueError(
                f"global_batch_size {self.global_batch_size} is not divisible by "
                f"local_batch_size {self.local_batch_size}"
            )


@dataclass(kw_only=True, slots=True)
class ParallelismConfig:
    data_parallel_replicate_degree: int = 1
    """
    The `data_parallel_replicate_degree` argument specifies the degree of
    data parallelism for weight replication. When this value is greater
    than 1, weights will be replicated across `data_parallel_replicate_degree`
    ranks. If `data_parallel_shard_degree` is also greater than 1, the parallelism
    method used is HSDP (Hybrid Sharded Data Parallelism). Otherwise, the
    parallelism method used is DDP (Distributed Data Parallelism).
    1 means disabled.
    """

    data_parallel_shard_degree: int = -1
    """
    The `data_parallel_shard_degree` argument specifies the degree of data
    parallelism for weight sharding. When this value is greater than 1, weights
    will be sharded across `data_parallel_shard_degree` ranks. If
    `data_parallel_replicate_degree` is also greater than 1, the parallelism
    method used is HSDP (Hybrid Sharded Data Parallelism). Otherwise, the
    parallelism method used is FSDP (Fully Sharded Data Parallelism).
    -1 means leftover ranks will be used (After DP_REPLICATE/SP/PP). Note that
    only `data_parallel_shard_degree` can be negative. 1 means disabled.
    """

    fsdp_backend: Literal["fsdp2", "gfsdp2"] = "fsdp2"
    """
    Which sharding implementation `apply_fsdp` uses. `fsdp2` is torch's
    `fully_shard`; `gfsdp2` is the in-repo backend, which shards a parameter by
    flat element ranges rather than by a tensor axis and hands the whole model a
    single gradient buffer.
    """

    fsdp_reshard_after_forward: Literal["default", "always", "never"] = "default"
    """
    `reshard_after_forward` specifies the policy for applying `reshard_after_forward`
    within an FSDP setup. `reshard_after_forward` controls parameter behavior after forward,
    trading off memory and communication. See torch's `fully_shard` API for more documentation
    on `reshard_after_forward`.

    The supported policies include "default", "always" and "never":

    - "default" applies default resharding behavior, implementing "smart defaults" for known optimal
      scenarios.
    - "always" will enable `reshard_after_forward` for all forward passes.
    - "never" will disable `reshard_after_forward` for all forward passes.
    """

    gfsdp_match_fsdp2_init: bool = False
    """
    With `fsdp_backend="gfsdp2"`, initialize the weights bit for bit as `fsdp2`
    would. Off, gfsdp2 draws each rank's flat slice directly: the same
    distribution, but different numbers, so a gfsdp2 run and an fsdp2 run start
    from different models. Turn it on for paired backend comparisons; it costs
    roughly twice the RNG work at init and no communication. Ignored, with a
    warning, on `fsdp2`.
    """

    enable_fsdp_symm_mem: bool = False
    """
    Whether to enable FSDP2 symmetric-memory communication optimizations for
    all FSDP modules after `fully_shard` has been applied.
    """

    tensor_parallel_degree: int = 1
    """Tensor Parallelism degree. 1 means disabled."""

    disable_loss_parallel: bool = False
    """Whether to apply loss parallel when sequence parallel is enabled"""

    enable_async_tensor_parallel: bool = False
    """Whether to apply async tensor parallel (currently only effective when compile is enabled)"""

    enable_sequence_parallel: bool = True
    """Whether to use SequenceParallel as part of tensor parallelism. Enabled by default."""

    full_dtensor: bool = False
    """Whether to use full DTensor mode."""

    pipeline_parallel_degree: int = 1
    """
    Pipeline Parallelism degree, or number of ranks. 1 means disabled.
    If using looped schedules, this still specifies the number of physical ranks, not the number
    of stages. Stages per rank are inferred from split points degree, and schedule.
    """

    module_fqns_per_model_part: list[list[str]] | None = None
    """
    Specify a list of lists containing the FQNs (Fully Qualified Names) of modules for each model chunk.
    Each inner list represents one model chunk and contains the module names that belong to that chunk.
    e.g. [['tok_embeddings', 'layers.0'], ['layers.1', 'layers.2'], ['layers.3', 'layers.4']]
    will create 3 chunks: the first containing tok_embeddings and layers.0,
    the second containing layers.1 and layers.2, and the third containing layers.3 and layers.4.
    This provides more explicit control over which modules belong to each chunk compared to split points.
    """

    pipeline_parallel_balancing: Literal["layer_count", "flops"] = "layer_count"
    """
    How to split modules into stages when module_fqns_per_model_part is not set:
    "layer_count" spreads transformer layers evenly (honoring the *_less_layers
    knobs); "flops" minimizes the heaviest stage by analytical FLOPs cost
    (AC recompute included), keeping the last stage the lightest (it gates
    pipeline fill/drain), and ignores the *_less_layers knobs. "flops"
    optimizes compute only and may shift layers to early stages, which hold the
    most in-flight activations under 1F1B — check peak memory on tight configs.
    """

    pipeline_parallel_module_costs_ms: dict[str, float] = field(default_factory=dict)
    """
    Replace the FLOPs estimates under "flops" balancing with measured module step
    times (fwd + bwd + AC recompute; only the ratios matter). Keys match FQNs
    exactly or as dot-prefixes, first match wins; the value is assigned to every
    matched unit (e.g. "mtp" prices each MTP head); coverage must be exact both
    ways.
    """

    pipeline_parallel_first_stage_less_layers: int = 1
    """
    The number of layers to reduce in the first stage of pipeline parallelism. This is because
    the first stage has the extra overhead of the embedding layer, which is not present in the other stages.
    """

    pipeline_parallel_last_stage_less_layers: int = 1
    """
    The number of layers to reduce in the last stage of pipeline parallelism. This is because
    the last stage has the extra overhead of the output layer, which is not present in the other stages.
    """

    pipeline_parallel_num_stages_per_rank: int | None = None
    """
    Number of virtual pipeline stages per rank (total stages = this x
    pipeline_parallel_degree; single-stage schedules require 1, multi-stage
    at least 2). Mutually exclusive with pipeline_parallel_layers_per_stage,
    and the only stage-count knob allowed with
    pipeline_parallel_balancing="flops". Unset: inferred from the schedule.
    """

    pipeline_parallel_layers_per_stage: int | None = None
    """
    The number of layers per (virtual) pipeline stage. If specified, the module_fqns_per_model_part will be
    calculated from the number of layers and pipeline_parallel_degree. If not specified, the
    layers per stage will be inferred from the model, schedule, and pipeline_parallel_degree.
    Mutually exclusive with pipeline_parallel_num_stages_per_rank; rejected with
    pipeline_parallel_balancing="flops".
    """

    pipeline_parallel_schedule: str = "1F1B"
    """
    Specify the Pipeline Parallel schedule to use. The supported schedules are:
    https://github.com/pytorch/pytorch/blob/de4c2a3b4e89d96334dc678d1c3f2ae51a6630a0/torch/distributed/pipelining/schedules.py#L2161.
    The schedule must be compatible with the split points and stages_per_rank.
    Looped schedules (e.g. Interleaved1F1B) require specifying pipeline_parallel_degree = number of ranks,
    and split_points = number of stages - 1
    """

    pipeline_parallel_schedule_csv: str = ""
    """
    Specify the path to the pipeline parallel schedule csv file to use
    (empty: no csv, the schedule comes from pipeline_parallel_schedule).
    The pipeline_parallel_schedule argument must be either
    PipelineScheduleSingle, PipelineScheduleMulti, or _PipelineScheduleRuntime.
    """

    pipeline_parallel_microbatch_size: int = 1
    """
    The size of each pipeline parallel microbatch (default 1).
    This value is used to compute the total number of microbatches by dividing local_batch_size with
    pipeline_parallel_microbatch_size.
    The global training batch size must be evenly divisible by pipeline_parallel_microbatch_size.
    """

    pipeline_parallel_grad_reduce: Literal["off", "dense", "moe", "full"] = "off"
    """
    Which FSDP param groups reduce-scatter gradients inside every microbatch
    backward under PP (sharded accumulation). "off" (upstream default): one
    reduce per chunk at REDUCE_GRAD, unsharded fp32 accumulation in between.
    "full": every group; "dense"/"moe": only non-expert/expert groups, the
    rest keep the "off" behavior. For selected groups the reduce-scatter
    count grows from one per chunk to local_batch_size /
    pipeline_parallel_microbatch_size. Equivalent numerics, not bitwise.
    """

    pipeline_parallel_pool_recv_buffers: bool = False
    """
    Reuse pipeline-parallel P2P recv buffers across microbatches via a lifetime-based pool.

    torch pipelining pre-allocates one recv buffer set per microbatch and per
    direction, so recv-buffer memory grows linearly with the number of
    microbatches while the schedule only ever has a bounded number of
    microbatches in flight. Requires a _PipelineScheduleRuntime schedule
    (e.g. Interleaved1F1B); otherwise a warning is logged and buffers are
    allocated as in stock.

    Independently of this option, the stage hands the model a fresh leaf
    tensor over each recv buffer on every forward, so autograd state (module
    hooks, .grad) cannot accumulate on the persistent buffers across steps.
    """

    context_parallel_degree: int = 1
    """Context parallelism degree. 1 means disabled."""

    context_parallel_load_balancer: Literal["headtail", "ptrr", "disable"] = "headtail"
    """
    Load balancer type for context parallelism. Options:
    - "headtail": Use HeadTailLoadBalancer for SDPA
    - "ptrr": Use PTRRLoadBalancer for FlexAttention
    - "disable": Disable load balancing
    """
    # context_parallel_rotate_method (allgather / alltoall kv exchange) was
    # removed: nothing read it since upstream #2555 dropped
    # create_context_parallel_ctx. To restore it, pass it to torch's CP
    # set_rotate_method in torchtitan/distributed/context_parallel.py.

    expert_parallel_degree: int = 1
    """
    Expert parallelism degree. 1 means disabled. No effect for non-MoE models.

    Mesh constraint: the dense region (dp_shard * cp * tp) and sparse region
    (efsdp * ep) cover the same ranks, so dp_shard * cp * tp == efsdp * ep.
    EP borrows ranks from FSDP and TP: efsdp = dp_shard * cp * tp / ep.
    pp and dp_replicate are outer dimensions unaffected by this constraint.
    """

    def __post_init__(self) -> None:
        # The degree rules of ``ParallelDims._validate``, checked torch-free so a
        # bad degree fails pre-submit instead of after process-group init.
        if (
            type(self.pipeline_parallel_microbatch_size) is not int
            or self.pipeline_parallel_microbatch_size <= 0
        ):
            raise ValueError(
                "pipeline_parallel_microbatch_size must be a positive integer"
            )
        for name in (
            "data_parallel_replicate_degree",
            "tensor_parallel_degree",
            "pipeline_parallel_degree",
            "context_parallel_degree",
            "expert_parallel_degree",
        ):
            degree = getattr(self, name)
            if degree < 1:
                raise ValueError(f"{name} must be >= 1, got {degree}")
        degree = self.data_parallel_shard_degree
        if degree != -1 and degree < 1:
            raise ValueError(
                f"data_parallel_shard_degree must be -1 (all remaining ranks) "
                f"or >= 1, got {degree}"
            )
        if degree != -1:
            validate_expert_parallel_degree(
                dp_shard=degree,
                cp=self.context_parallel_degree,
                tp=self.tensor_parallel_degree,
                ep=self.expert_parallel_degree,
            )
        if (
            self.pipeline_parallel_num_stages_per_rank is not None
            and self.pipeline_parallel_layers_per_stage is not None
        ):
            raise ValueError(
                "pipeline_parallel_num_stages_per_rank and "
                "pipeline_parallel_layers_per_stage are mutually exclusive."
            )
        if (
            self.pipeline_parallel_balancing == "flops"
            and self.pipeline_parallel_layers_per_stage is not None
        ):
            raise ValueError(
                "pipeline_parallel_layers_per_stage derives the stage count from "
                "layer-count arithmetic that pipeline_parallel_balancing='flops' "
                "does not use; set pipeline_parallel_num_stages_per_rank instead."
            )

        if self.pipeline_parallel_grad_reduce != "off":
            if self.pipeline_parallel_degree == 1:
                raise ValueError(
                    "parallelism.pipeline_parallel_grad_reduce requires"
                    " pipeline_parallel_degree > 1."
                )
            if self.tensor_parallel_degree > 1:
                raise ValueError(
                    "parallelism.pipeline_parallel_grad_reduce does not"
                    " support tensor parallelism: in-backward sync"
                    " redistributes TP Partial gradients before the fp32"
                    " reduce-dtype upcast, changing reduction precision."
                    # TODO: preserve the reduce dtype for TP grads, then lift this.
                )
            if (
                self.pipeline_parallel_grad_reduce == "moe"
                and self.expert_parallel_degree == 1
            ):
                raise ValueError(
                    'parallelism.pipeline_parallel_grad_reduce="moe" requires'
                    " expert_parallel_degree > 1: without EP, expert and dense"
                    " params share one FSDP param group and cannot be reduced"
                    " separately."
                )


@dataclass(kw_only=True, slots=True)
class ActivationCheckpointConfig:
    mode: Literal["selective", "full", "memory_budget", "none"] = "selective"
    """Type of activation checkpointing to use"""

    per_op_sac_force_recompute_mm_shapes_by_fqns: list[str] = field(
        default_factory=lambda: ["moe.router.gate"]
    )
    """
    When per-op selective ac is used, this list of fully qualified names is used
    to determine which mm shapes to force recompute, rather than being considered
    by rest of the sac policy, e.g save every other mm. Only nn.Linear modules are
    supported today.

    Note: this config applies to mms not limited to those matching the specified
    fqns, e.g. if "moe.router.gate", corresponding to Linear(in, out), is specified,
    ANY mm with shape matching (*, in) x (in, out) will be force recomputed.
    """

    early_stop: bool = False
    """
    Whether to stop recomputing early when all activations have already been
    rematerialized.
    """

    memory_budget: float = 0.5
    """
    When mode is set to "memory_budget", this value determines how much
    partitioner in the compiler should trade off compute for memory.
    0.0 corresponds to the activation memory from applying
    activation checkpointing to the full compiled region, and 1.0 corresponds to
    the activation memory from the default runtime-optimized strategy. Read here:
    https://pytorch.org/blog/activation-checkpointing-techniques/
    """

    visualize_memory_budget_pareto: bool = False
    """
    This dumps out a SVG visualization of the expected runtime vs. activation
    memory tradeoffs for all memory budget values from 0 to 1 in increments of
    0.05 in {--dump_folder}/memory_budget_pareto folder. See an example here:
    https://github.com/pytorch/pytorch/pull/126320#discussion_r1625104015
    """

    preserve_rng_state: bool = True
    """
    If deterministic output compared to non-checkpointed passes is required, set
    to true. Results in stashing and restoring the RNG state during each checkpoint,
    may be slower. See https://docs.pytorch.org/docs/stable/checkpoint.html
    for details.
    """

    determinism_check: str = "default"
    """
    A string specifying the determinism function. See
    https://docs.pytorch.org/docs/stable/checkpoint.html for details.
    """

    debug: bool = False
    """
    Capture ac debug information. Will be slower. See
    https://docs.pytorch.org/docs/stable/checkpoint.html for details.
    """


@dataclass(kw_only=True, slots=True)
class CompileConfig:
    enable: bool = False
    """Whether to apply torch.compile"""

    components: list[str] = field(default_factory=lambda: ["model", "loss"])
    """Which components to compile"""

    backend: str = "inductor"


@dataclass(kw_only=True, slots=True)
class CommConfig:
    init_timeout_seconds: int = 3600
    """
    Timeout for communication operations during initialization and the first train
    step, including dataset preparation, checkpoint loading, and compilation.
    After that step, train_timeout_seconds applies.
    """

    train_timeout_seconds: int = 1800
    """Timeout after the first train step, including streaming epoch preparation."""

    trace_buf_size: int = 20000
    """Flight recorder ring buffer size, >0 means recording by default, 0 means disabled"""

    save_traces_folder: str = "comm_traces"
    """Flight recorder trace files location"""

    save_traces_file_prefix: str = "rank_"
    """Flight recorder trace files prefix"""

    mode: Literal["default", "fake_backend", "local_tensor", "torchcomms"] = "default"
    """
    Communication mode for distributed training.

    Options:
    - "default": Normal distributed training with real communication
    - "fake_backend": Fake comm backend for dry run mode only (configuration validation without GPU)
    - "local_tensor": Local tensor mode for debugging purposes. There will be only one process
      regardless of the number of GPUs. LocalTensor will simulate the computation by running one
      rank after another. While the performance will be slow, the numerics should be the same.
      This enables us to verify numerics with fewer GPUs. For example, we can directly run 5D
      parallelisms within a single node to reduce the combinations we need to use in integration tests.
    - "torchcomms": Use torchcomms-based communicators. Requires the torchcomms package to be installed.

    NOTE: local_tensor is an experimental feature and automatically uses fake_backend internally.
    """


@dataclass(kw_only=True, slots=True)
class DebugConfig:
    seed: int | None = None
    """Choose the base RNG seed used for training"""

    deterministic: bool = False
    """Use deterministic algorithms wherever possible, may be slower"""

    deterministic_warn_only: bool = False
    """Only warns about ops without deterministic implementations rather than erroring out  """

    moe_force_load_balance: bool = False
    """Sample experts with uniform random priorities, respecting group limits.
    Balanced in expectation; debugging only."""

    moe_force_first_group: bool = False
    """If True, every token is routed to experts ``0..top_k-1``, i.e. to the first
    expert-parallel rank only. Requires top_k <= num_experts / EP and an evenly
    partitioned expert set; invalid layouts raise ValueError. Exercises empty
    receivers in the dispatcher and capacity policies. Debugging only."""

    detect_anomaly: bool = False
    """Enable torch.autograd anomaly detection to help track down NaN/Inf gradients.
    Note: incurs significant overhead; for debugging only."""

    batch_invariant: bool = False
    """Enable batch-invariant mode to use batch-invariant ops in model
    forward and deterministic NCCL collective reduction order"""

    print_config: bool = False
    """Print the job configs to terminal"""

    enable_structured_logging: bool = True
    """Whether to enable the structured per-rank trace logger (see
    ``torchtitan.observability.structured_logger``). When False, all
    ``log_trace_span`` / ``log_trace_instant`` / ``log_trace_scalar`` calls
    are no-ops. Disable to fully eliminate trace overhead."""


@dataclass(kw_only=True, slots=True)
class FP8Config:
    """Single settings surface for blockwise fp8 training (dense + sparse).

    Hopper-only (SM90): requires cudacherry and DeepGEMM from the training image.
    """

    dense_include_linear_layers: str = ""
    r"""Regex matched with ``re.search`` against each ``Linear`` FQN; matching
    projections are replaced with ``FP8Linear``. Empty disables dense fp8.
    Example: ``r".*\.(w1|w2|w3|wo)$"``."""

    dense_use_shared_quantizer: bool = False
    """Share one input quantizer across w1/w3 in every fully-fp8 ``FeedForward``.
    Has no effect on FFNs that are not fully converted."""

    dense_fused_swiglu_quant: bool = True
    """Fuse SwiGLU with the quantization of w2's input in every fully-fp8
    ``FeedForward``. False runs a bf16 SwiGLU between the fp8 projections."""

    sparse_use_grouped_gemm: bool = False
    """Route MoE experts through the DeepGEMM fp8 grouped GEMM. Requires the
    ``deepep`` token dispatcher and ``score_before_experts=False``."""

    sparse_deep_gemm_num_sms: int = 116
    """SM budget handed to DeepGEMM, leaving the rest for the EP all-to-all.
    Must match the value used during cache warmup: it is part of the DeepGEMM
    JIT cache key, so a mismatch silently misses every warmed kernel."""

    deep_gemm_warmup_on_start: bool = True
    """Compile the experts' DeepGEMM kernels into the node-local JIT cache before
    building the model. False expects the cache to be warmed beforehand."""

    sparse_fused_swiglu_quant: bool = True
    """Fuse the experts' SwiGLU with the quantization of the second grouped
    GEMM's input. False runs a bf16 SwiGLU between the fp8 grouped GEMMs, which
    the routed experts' w1/w3 hidden z-loss requires."""

    def __post_init__(self) -> None:
        if self.dense_include_linear_layers:
            try:
                re.compile(self.dense_include_linear_layers)
            except re.error as exc:
                raise ValueError(
                    "fp8.dense_include_linear_layers must be a valid regex, got "
                    f"{self.dense_include_linear_layers!r}: {exc}"
                ) from exc
        if self.dense_use_shared_quantizer and not self.dense_include_linear_layers:
            raise ValueError(
                "fp8.dense_use_shared_quantizer requires "
                "fp8.dense_include_linear_layers to select the fp8 projections"
            )
        if not self.dense_fused_swiglu_quant and not self.dense_include_linear_layers:
            raise ValueError(
                "fp8.dense_fused_swiglu_quant=False requires "
                "fp8.dense_include_linear_layers to select the fp8 projections"
            )
        if not self.sparse_fused_swiglu_quant and not self.sparse_use_grouped_gemm:
            raise ValueError(
                "fp8.sparse_fused_swiglu_quant=False requires "
                "fp8.sparse_use_grouped_gemm"
            )
        if not 1 <= self.sparse_deep_gemm_num_sms <= 132:
            raise ValueError(
                "fp8.sparse_deep_gemm_num_sms must be in [1, 132] on H100, got "
                f"{self.sparse_deep_gemm_num_sms}"
            )
