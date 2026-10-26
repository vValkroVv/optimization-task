# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import torch
import torchstore as ts
from monarch.actor import Actor, current_rank, endpoint
from typing import TypeAlias

from torchtitan.config.schema.experiments.rl.actors.generator import VLLMGeneratorConfig
from torchtitan.config.schema.experiments.rl.actors.generator import (
    SamplingConfig,
)
from torchtitan.config import (
    CompileConfig,
    Configurable,
    DebugConfig,
)
from torchtitan.distributed.utils import set_batch_invariance
from torchtitan.experiments.rl.models.vllm_registry import (
    registry_to_vllm,
    TORCHTITAN_CONFIG_FORMAT,
)
from torchtitan.experiments.rl.observability import metrics as m
from torchtitan.experiments.rl.types import Completion
from torchtitan.observability import structured_logger as sl
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.tools.logging import init_logger
from torchtitan.tools.utils import has_cuda_capability
from vllm import EngineArgs, LLMEngine, SamplingParams
from vllm.config import AttentionConfig
from vllm.outputs import RequestOutput
from vllm.sampling_params import RequestOutputKind
from vllm.v1.attention.backends.registry import AttentionBackendEnum

logger = logging.getLogger(__name__)


def _prepare_generation_request_metrics(
    output: RequestOutput, *, prefix: str
) -> list[m.Metric]:
    """Prepare vLLM metrics from a RequestOutput.

    For `[num_prompts]` submitted prompts, vLLM returns `[num_prompts]`
    per parent `RequestOutput`s (one per `add_request` call), each carrying
    a single `RequestStateStats` on `.metrics`.

    Caveat under `SamplingParams.n > 1`: vLLM stores one `RequestStateStats`
    per child request; the parent output exposes the **last-finishing**
    child's timeline. `arrival_time` is shared across siblings, but
    [`queued_ts`, `scheduled_ts`, `first_token_ts`, `last_token_ts`,
    `num_generation_tokens`] describe one specific child — not an aggregate,
    not the first sibling's. The other `n-1` siblings' stats are dropped by
    vLLM at ``output_processor._finish_request``.
    """

    # TODO: Per-request fields here come from RequestOutput.metrics
    # (RequestStateStats). Engine-aggregate stats, such as KV-cache usage,
    # prefix-cache hit rate, preemptions, and batch occupancy, live in
    # SchedulerStats / IterationStats and require registering a
    # vllm.v1.metrics.loggers.StatLoggerBase via
    # LLMEngine.from_engine_args(..., stat_loggers=[...]).

    metric_values: dict[str, float] = {}
    if output.num_cached_tokens is not None:
        metric_values[f"{prefix}/num_cached_tokens"] = output.num_cached_tokens

    stats = output.metrics
    if stats is not None:
        metric_values[f"{prefix}/queue_time_ms"] = (
            stats.scheduled_ts - stats.queued_ts
        ) * 1000

        if stats.num_generation_tokens > 0:
            metric_values[f"{prefix}/time_to_first_token_ms"] = (
                stats.first_token_latency * 1000
            )
            metric_values[f"{prefix}/prefill_time_ms"] = (
                stats.first_token_ts - stats.scheduled_ts
            ) * 1000

        if stats.num_generation_tokens > 1:
            first_to_last_token_ms = (stats.last_token_ts - stats.first_token_ts) * 1000
            metric_values[f"{prefix}/decode_time_ms"] = first_to_last_token_ms
            metric_values[f"{prefix}/inter_token_latency_ms"] = (
                first_to_last_token_ms / (stats.num_generation_tokens - 1)
            )

    # Emit each value with both Mean and Max aggregators.
    return [
        metric
        for key, value in metric_values.items()
        for metric in (m.Metric(key, m.Mean(value)), m.Metric(key, m.Max(value)))
    ]


class VLLMGenerator(Actor, Configurable):
    """
    Generates rollouts using vLLM engine.

    Maintains a vLLM engine synchronized with the Trainer via weight
    sync. ``generate()`` produces a flat list of Completions; reward
    and advantage computation live in the controller.

    Args:
        config: Generator-specific configuration.
        model_spec: TorchTitan model specification.
        model_path: Path to the HF model checkpoint.
        compile_config: Per-layer torch.compile config shared with the
            trainer so both sides compile identically.
    """

    Config: TypeAlias = VLLMGeneratorConfig

    def __init__(
        self,
        config: Config,
        *,
        model_spec: ModelSpec,
        model_path: str,
        compile_config: CompileConfig,
        max_num_seqs: int,
        output_dir: str,
    ):
        init_logger()
        sl.init_structured_logger(
            source="rl_generator",
            output_dir=output_dir,
            rank=current_rank().rank,
            enable=config.debug.enable_structured_logging,
        )
        sl.log_trace_instant("structured_logger_started")

        self.config = config
        self.model_spec = model_spec

        # max_num_seqs controls vLLM's maximum batch dimension: it sets
        # the upper bound for concurrent sequences, determines KV-cache
        # block allocation (and therefore GPU memory usage), and bounds
        # the CUDA graph capture sizes.  Always computed by the caller
        # (RLTrainer) as num_prompts_per_step * sampling.n.
        self._max_num_seqs = max_num_seqs

        # Register TorchTitan model + parser with vLLM
        registry_to_vllm(
            model_spec,
            parallelism=config.parallelism,
            compile_config=compile_config,
            checkpoint_config=config.checkpoint,
        )

        # Set vLLM environment variables from config before any vLLM initialization
        os.environ["VLLM_ATTENTION_BACKEND"] = "CUSTOM"
        os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"

        set_batch_invariance(config.debug.batch_invariant)

        self._set_determinism(config.debug)

        self.model_path = model_path

        # Build vLLM engine
        engine_kwargs = dict(
            # ``model`` is the path to the HF checkpoint directory. The
            # config is sourced from torchtitan's ModelSpec via
            # ``config_format=TORCHTITAN_CONFIG_FORMAT`` (no config.json
            # read), but vLLM still uses this path to locate the
            # tokenizer assets and the safetensors weight shards.
            model=model_path,
            trust_remote_code=True,
            # Use the torchtitan custom config parser (registered by
            # registry_to_vllm above). It builds PretrainedConfig from
            # ModelSpec instead of reading config.json from disk.
            config_format=TORCHTITAN_CONFIG_FORMAT,
            dtype=config.model_dtype,
            tensor_parallel_size=config.parallelism.tensor_parallel_degree,
            # Monarch already spawned TP workers via proc mesh. "external_launcher"
            # tells vLLM to run one worker per process (no subprocess spawning)
            distributed_executor_backend="external_launcher",
            gpu_memory_utilization=config.gpu_memory_limit,
            enforce_eager=not config.cudagraph.enable,
            attention_config=AttentionConfig(
                backend=AttentionBackendEnum.CUSTOM,
            ),
            # Enables RequestOutput.metrics, so generator metrics can be returned
            disable_log_stats=False,
        )
        engine_kwargs["max_model_len"] = model_spec.model.rope.max_seq_len
        engine_kwargs["max_num_seqs"] = self._max_num_seqs
        # FA2 requires block_size to be a multiple of 256
        if not has_cuda_capability(9, 0):
            engine_kwargs["block_size"] = 256
        vllm_compilation_config = config.cudagraph.get_vllm_compilation_config(
            max_num_seqs=self._max_num_seqs,
        )
        if vllm_compilation_config is not None:
            engine_kwargs["compilation_config"] = vllm_compilation_config
        if config.debug.seed is not None:
            engine_kwargs["seed"] = config.debug.seed
        engine_args = EngineArgs(**engine_kwargs)

        with sl.log_trace_span("vllm_init"):
            logger.info("Initializing LLMEngine from EngineArgs...")
            self._engine = LLMEngine.from_engine_args(engine_args)
            logger.info("vLLM rollout engine initialized")

        self.policy_version = 0

        logger.info("Generator initialized with vLLM engine")

    @staticmethod
    def _set_determinism(debug: DebugConfig) -> None:
        """Apply deterministic flags for the generator.

        The generator doesn't use torchtitan's ParallelDims, so we apply
        the deterministic flags directly instead of using set_determinism().
        """
        if debug.deterministic:
            torch.use_deterministic_algorithms(
                True, warn_only=debug.deterministic_warn_only
            )
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

        if debug.seed is not None:
            torch.manual_seed(debug.seed)

    def _get_model(self):
        """Access the model from the vLLM engine.
        Returns a VLLMModelWrapper instance.
        """
        return self._engine.model_executor.driver_worker.get_model()

    @endpoint
    async def sync_log_step(self, step: int, relative_step: int | None = None) -> None:
        """Sync the structured-logger step counter from the controller."""
        sl.set_step(step, relative_step=relative_step)

    @endpoint
    @sl.log_trace_span("generate")
    async def generate(
        self,
        tokenized_prompts: list[list[int]],
        *,
        sampling_config: SamplingConfig | None = None,
        metrics_prefix: str = "generator",
    ) -> tuple[list[Completion], list[m.Metric]]:
        """Generate completions and generator metrics for tokenized prompts.

        Takes ``tokenized_prompts`` as ``[num_prompts][prompt_tokens]``.
        Returns completions as ``[num_prompts * n]`` plus generator metrics,
        where ``n`` is the resolved ``SamplingConfig.n`` completions per prompt.

        Args:
            tokenized_prompts: Tokenized prompts shaped ``[num_prompts][prompt_tokens]``.
            sampling_config: Optional per-call override for the generator's
                default SamplingConfig. ``seed`` always comes from
                ``config.debug.seed`` (not part of SamplingConfig).
            metrics_prefix: Namespace prepended to every returned metric key
                (default ``"generator"``). Callers that need to keep streams
                separate, e.g. ``"validation/generator"``, can override it.
        """
        _sampling_config = (
            sampling_config if sampling_config is not None else self.config.sampling
        )

        logger.debug(
            f"{os.getpid()=} Generating start generate (policy v{self.policy_version})..."
        )

        with torch.no_grad():
            sampling_params = SamplingParams(
                temperature=_sampling_config.temperature,
                top_p=_sampling_config.top_p,
                max_tokens=_sampling_config.max_tokens,
                n=_sampling_config.n,
                seed=self.config.debug.seed,
                logprobs=1,
                output_kind=RequestOutputKind.FINAL_ONLY,
            )

            # render_cmpl is vLLM's input-pipeline entry.
            # The tokenize step is a no-op for already-tokenized prompts. The
            # lower-level alternative is vllm.inputs.tokens_input; we use the
            # high-level API to stay resilient to vLLM internal changes.
            engine_inputs = self._engine.renderer.render_cmpl(
                [{"prompt_token_ids": ids} for ids in tokenized_prompts]
            )
            for i, engine_input in enumerate(engine_inputs):
                self._engine.add_request(
                    request_id=str(i),
                    prompt=engine_input,
                    params=sampling_params,
                )

            all_outputs = []
            with sl.log_trace_span("engine_steps"):
                while self._engine.has_unfinished_requests():
                    all_outputs.extend(self._engine.step())

            # vLLM may return requests out of order; sort by the integer
            # request_id we assigned so prompt_idx lines up with the input.
            all_outputs.sort(key=lambda o: int(o.request_id))

            completions: list[Completion] = []
            generation_metrics: list[m.Metric] = []
            output_token_counts: list[int] = []
            for output in all_outputs:
                prompt_idx = int(output.request_id)
                generation_metrics.extend(
                    _prepare_generation_request_metrics(output, prefix=metrics_prefix)
                )
                for sample in output.outputs:
                    per_token_logprobs = [
                        list(logprob_dict.values())[0].logprob
                        for logprob_dict in sample.logprobs
                    ]
                    output_token_counts.append(len(sample.token_ids))
                    completions.append(
                        Completion(
                            policy_version=self.policy_version,
                            prompt_idx=prompt_idx,
                            text=sample.text,
                            token_ids=sample.token_ids,
                            token_logprobs=per_token_logprobs,
                            finish_reason=sample.finish_reason,
                        )
                    )
            generation_metrics.append(
                m.Metric(
                    f"{metrics_prefix}/output_tokens",
                    m.Sum.from_list(output_token_counts),
                )
            )

        logger.debug(
            f"{os.getpid()=} Generating finish generate (policy v{self.policy_version})..."
        )

        # TODO: consider passing metrics as an arg to Completion and Trajectory,
        # e.g. Completion(metrics=generation_metrics), so when we log the rollouts
        # to a json or database,  we can associate each rollout with its metrics
        # I am keeping like this for now for simplicity.
        return completions, generation_metrics

    @endpoint
    @sl.log_trace_span("pull_model_state_dict")
    async def pull_model_state_dict(self, version: int) -> None:
        """Pull latest weights from TorchStore.

        When ``direct_rdma=True``, weights are read directly from the
        trainer's GPU memory via one-sided RDMA, bypassing StorageVolumes.
        When ``False``, data is fetched through StorageVolumes (which may
        themselves use RDMA as their transport internally).

        See ``push_model_state_dict`` for more details on the distinction.

        Args:
            version: New policy version number.
        """
        from monarch.rdma import is_rdma_available

        model_sd = self._get_model().model.state_dict()
        await ts.get_state_dict(
            "model_state_dict",
            user_state_dict=model_sd,
            strict=False,
            direct_rdma=is_rdma_available(),
        )
        self.policy_version = version
        # Invalidate the KV prefix cache so stale values computed with the
        # old weights are never reused for new generations.
        self._engine.reset_prefix_cache()
        logger.debug(
            f"{os.getpid()=} Generator pulled model state dict for policy v{version}"
        )

    @endpoint
    async def close(self) -> None:
        """Release the vLLM engine.

        vLLM's sync ``LLMEngine`` (what we use) has no public ``shutdown``
        method; only the async ``AsyncLLM`` does. We tear it down by
        plumbing through its components in the same order ``AsyncLLM``
        uses internally:

        1. ``renderer.shutdown()`` — closes thread pools and the
           multimodal-processor cache.
        2. ``engine_core.shutdown()`` — stops the model worker and the
           scheduler.

        We intentionally skip ``cleanup_dist_env_and_memory()`` here:
        with ``external_launcher``, vLLM reuses the process group that
        Monarch created. Destroying it here would prevent Monarch's
        ``mesh.stop()`` from completing its own collective teardown.
        """
        if self._engine is not None:
            renderer = getattr(self._engine, "renderer", None)
            try:
                if renderer is not None:
                    renderer.shutdown()
            finally:
                self._engine.engine_core.shutdown()
