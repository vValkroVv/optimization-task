# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Literal

from torchtitan.config.configs import CompileConfig
from torchtitan.config.errors import ConfigFieldError


@dataclass(kw_only=True, slots=True)
class GraphTrainerCompileConfig(CompileConfig):
    mode: Literal["jit", "aot_fx_trace"] = "aot_fx_trace"
    """
    Compilation mode. Options:
        aot_fx_trace: non-strict tracing of fwd+loss+bwd via make_fx
        jit: standard torch.compile() with custom backend (deprecated)
    ``compile.enable: false`` uses the ordinary eager forward/backward path,
    without tracing or loading precompiled artifacts, regardless of ``mode``.
    """

    backend: str = "aot_eager"

    passes: list[str] = field(default_factory=list)
    """
    Compiler pass names to apply.
    In JIT mode: applied as graph passes (e.g., auto_bucketing, transformer_block_bucketing)
    """

    enable_passes: bool = True
    """When False, skip all graph passes (both default and user-configured)."""

    disable_passes: list[str] = field(default_factory=list)
    """Pass names to selectively disable for debugging and ablation
    studies. A pass is skipped if its name exactly matches any entry.
    Example: --compile.disable_passes custom_codegen_pass,cudagraph_pass"""

    debug_graph_passes: bool = False
    """Log timing, op-count diffs, and before/after graphs for each pass to tlparse."""

    memory_policy: Literal["default", "eager", "sac_and_offload"] = "default"
    """
    Memory optimization policy for activation management (SAC, offload).
        default: SAC — save all compute-intensive ops and FSDP all_gathers.
        eager: SAC alternating mm ops between save/recompute, matching the
            eager AC policy in torchtitan.distributed.activation_checkpoint.
        sac_and_offload: SAC + CPU offload — apply default SAC first,
            then offload surviving MUST_SAVE activations to CPU within
            the cpu_offload_budget_gb budget.
    """

    pass_pipeline: str = "default"
    """Pass pipeline selection. Controls which graph pass pipeline, post-init
    hooks, and pre-train-step hooks are activated."""

    inductor_compilation: Literal["regional", "full"] = "regional"
    """
    Inductor compilation strategy. Mutually exclusive options:
        regional: compile tagged regions (e.g. FlexAttention HOPs) with
            regional_inductor while leaving the rest interpreted.
        full: compile the entire graph with inductor into optimized
            Triton kernels. Provides better performance but may change
            bitwise numerics compared to regional/interpreted execution.
    """

    numerics_changing_optim: bool = False
    """Enable passes that improve performance but may change numerics
    compared to the uncompiled path (e.g. RMSNorm Inductor fusion)."""

    cpu_offload_prefetch_n_layers: int = 1
    """Prefetch reloads this many layers ahead in the backward graph
    to overlap H2D transfers with compute."""

    cpu_offload_defer_n_layers: int = 1
    """Defer forward wait_tensor ops this many layers past the last consumer
    to overlap D2H transfers with compute."""

    cpu_offload_budget_gb: float = 100.0
    """Maximum CPU memory budget (in GB per rank) for offloaded activations.
    Tensors are selected largest-first until the budget is exhausted."""

    enable_fsdp_ag_rs_overlap: bool = False
    """When True, run ``overlap_fsdp_ag_rs_pass``. The pass moves backward
    FSDP all-gathers onto a separate CUDA stream from reduce-scatters so the
    two collectives can overlap. It is a no-op when the graph contains no
    FSDP all-gathers."""

    precompile_artifact_dir: str = ""
    """
    Directory for precompiled artifacts. Setting this enables precompile:
    precompile_main.py saves the artifact here, and training loads it from
    here to skip compilation. For multi-node setups use a shared filesystem
    path.
    """

    enable_autoparallel: bool = False
    """Use AutoParallelGraph (ILP solver-based SPMD sharding) instead of
    manual TP/FSDP/EP. Requires enabled compilation in ``aot_fx_trace`` mode."""

    def __post_init__(self) -> None:
        validate_autoparallel_config(self)


def validate_autoparallel_config(
    compile_config: GraphTrainerCompileConfig,
) -> None:
    if compile_config.enable_autoparallel:
        if not compile_config.enable:
            raise ConfigFieldError(
                "enable_autoparallel", "AutoParallel requires compile.enable=true"
            )
        if compile_config.mode != "aot_fx_trace":
            raise ConfigFieldError(
                "enable_autoparallel",
                "AutoParallel graph_trainer integration only supports "
                "--compile.mode aot_fx_trace",
            )
