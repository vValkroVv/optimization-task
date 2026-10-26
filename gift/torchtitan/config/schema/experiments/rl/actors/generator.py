import math
from dataclasses import dataclass, field
from typing import ClassVar, TYPE_CHECKING

from torchtitan.config.configs import DebugConfig, ParallelismConfig
from torchtitan.config.configurable import config, Configurable
from torchtitan.config.schema.components.checkpoint import CheckpointManagerConfig

if TYPE_CHECKING:
    from vllm.config import CompilationConfig


@dataclass(kw_only=True, slots=True)
class VLLMCudagraphConfig:
    """CUDA graph capture settings for the vLLM inference engine.

    torch.compile is configured separately via ``CompileConfig`` at the
    ``RLTrainer`` level, shared by both trainer and generator.  Only CUDA
    graph capture, which is vLLM-specific, is controlled here.

    When enabled, vLLM captures the forward pass as a single CUDA graph
    ("full" mode).  "piecewise" modes are intentionally excluded: they
    require vLLM's whole-model torch.compile to split the graph around
    non-capturable ops, which conflicts with per-layer compile.
    """

    enable: bool = True
    """Whether to enable CUDA graph capture (vLLM "full" mode)."""

    # TODO: Validate CUDA graph capture with MoE / Expert Parallelism.
    # MoE routing produces dynamic shapes that may conflict with full
    # CUDA graph capture despite being torch.compile-compatible
    # post https://github.com/pytorch/torchtitan/pull/3142

    # TODO: Explore applying CUDA graph capture on the torchtitan trainer
    # side as well (not just the vLLM generator).
    # https://github.com/pytorch/torchtitan/issues/3175

    def get_vllm_compilation_config(
        self, *, max_num_seqs: int
    ) -> "CompilationConfig | None":
        """Build a vLLM ``CompilationConfig``, or return ``None`` when
        CUDA graphs are disabled.

        ``max_num_seqs`` determines CUDA graph capture sizes: powers of
        2 from 1 up to ``max_num_seqs``, plus ``max_num_seqs`` itself
        if it isn't already a power of 2.
        """
        from vllm.config import CompilationConfig

        if not self.enable:
            return None
        if max_num_seqs <= 0:
            raise ValueError(f"max_num_seqs must be positive, got {max_num_seqs}")
        sizes = [1 << i for i in range(int(math.log2(max_num_seqs)) + 1)]
        if max_num_seqs not in sizes:
            sizes.append(max_num_seqs)
        return CompilationConfig(
            cudagraph_mode="full",
            mode=0,
            cudagraph_capture_sizes=sorted(sizes),
        )


@dataclass(kw_only=True, slots=True)
class SamplingConfig:
    """Sampling parameters passed to vLLM's SamplingParams."""

    n: int = 8
    """Number of completions to generate per prompt (vLLM SamplingParams.n)."""

    temperature: float = 0.8
    """Sampling temperature. 0.0 = greedy, higher = more random."""

    top_p: float = 0.95
    """Nucleus sampling threshold."""

    max_tokens: int = 100
    """Maximum number of tokens to generate per completion."""


@config
class VLLMGeneratorConfig(Configurable.Config):
    """Generator actor configuration.
    TODO: Expose a EngineConfig field to passing config to vLLM Engine"""

    _owner_spec: ClassVar[str] = (
        "torchtitan.experiments.rl.actors.generator:VLLMGenerator"
    )

    parallelism: ParallelismConfig = field(default_factory=ParallelismConfig)

    sampling: SamplingConfig = field(default_factory=SamplingConfig)

    model_dtype: str = "bfloat16"
    """Data type for model weights, passed directly to vLLM (auto, float16, bfloat16, float32)."""

    gpu_memory_limit: float = 0.9
    """Fraction of GPU memory to use for the vLLM engine (0.0 to 1.0)."""

    cudagraph: VLLMCudagraphConfig = field(default_factory=VLLMCudagraphConfig)

    checkpoint: CheckpointManagerConfig = field(default_factory=CheckpointManagerConfig)
    """Initial HF weight loading; keep disabled in the RL loop, where weights
    arrive from TorchStore. For standalone inference set ``enable=True`` and
    ``initial_load_in_hf=True``."""

    debug: DebugConfig = field(default_factory=DebugConfig)

    def __post_init__(self):
        # VLLMGenerator only supports TP. vLLM handles its own parallelism;
        # we only apply TP via the core parallelize function.
        p = self.parallelism
        if p.data_parallel_replicate_degree != 1:
            raise ValueError(
                f"Generator does not support data parallel replication, "
                f"got dp_replicate={p.data_parallel_replicate_degree}"
            )
        if p.pipeline_parallel_degree > 1:
            raise ValueError(
                f"Generator does not support pipeline parallelism, "
                f"got pp={p.pipeline_parallel_degree}"
            )
        if p.context_parallel_degree > 1:
            raise ValueError(
                f"Generator does not support context parallelism, "
                f"got cp={p.context_parallel_degree}"
            )
        if p.expert_parallel_degree > 1:
            raise ValueError(
                f"Generator does not support expert parallelism, "
                f"got ep={p.expert_parallel_degree}"
            )
        if p.enable_sequence_parallel:
            raise ValueError(
                "Generator does not support sequence parallelism: "
                "spmd_types erasure mode requires sequence length to be "
                "evenly divisible by TP, which doesn't hold for inference "
                "(uneven batches). Set enable_sequence_parallel=False."
            )
        if not p.disable_loss_parallel:
            raise ValueError(
                "Generator requires disable_loss_parallel=True, "
                f"got disable_loss_parallel={p.disable_loss_parallel}"
            )
