# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import copy
import math
import os
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager

import torch
import torch.nn as nn
from torch.distributed._mesh_layout import _MeshLayout
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.pipelining import PipelineStage

from torchtitan.models.common.regularization import suppress_aux_recording
from torch.distributed.pipelining.schedules import (
    _PipelineSchedule,
    _PipelineScheduleRuntime,
    get_schedule_class,
    PipelineScheduleMulti,
    PipelineScheduleSingle,
    ScheduleDualPipeV,
    ScheduleZBVZeroBubble,
)

from torchtitan.components.loss import ChunkedCELoss, LossFunction
from torchtitan.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.compile import EagerWindow
from torchtitan.distributed.pipeline_balance import (
    flops_balanced_module_fqns,
    model_pipeline_units,
)
from torchtitan.distributed.pipeline_context import pipeline_metadata_inference
from torchtitan.distributed.pipeline_parallel_recv_pool import (
    pool_pipeline_recv_buffers,
    RecvPoolStage,
)
from torchtitan.distributed.pipeline_grad_reduce import (
    install_per_microbatch_grad_reduce,
)
from torchtitan.protocols.model import BaseModel
from torchtitan.protocols.model_spec import ParallelizeFunction
from torchtitan.protocols.module import ModuleDict, ModuleList
from torchtitan.tools.logging import logger
from torchtitan.tools.pipeline import log_pipeline_layout

# pipeline_llm is the public entrypoint for model-specific PP setup. Helpers in
# this module are implementation details and should stay private.
__all__ = ["pipeline_llm"]


def _build_get_mesh_callback(
    parallel_dims: ParallelDims,
) -> Callable[[tuple[str, ...], _MeshLayout | None], DeviceMesh | None]:
    """Build a callback that resolves a DeviceMesh from dimension names.

    Pipeline parallelism requires an SPMD mesh during module split so that
    at runtime the current PP rank can reconstruct a DTensor after receiving
    a plain tensor from the previous PP rank. DTensors are not directly
    serializable across PP stages (because ProcessGroup is not serializable),
    so each stage uses this callback to obtain its local DeviceMesh and
    re-wrap incoming tensors as DTensors with the correct placements.
    """

    def _get_mesh(
        mesh_dim_names: tuple[str, ...], mesh_layout: _MeshLayout | None
    ) -> DeviceMesh | None:
        mesh = parallel_dims.get_mesh(list(mesh_dim_names))
        if mesh_layout is not None and mesh._layout != mesh_layout:
            return None
        return mesh

    return _get_mesh


def pipeline_llm(
    model: nn.Module,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointConfig,
    dump_folder: str,
    device: torch.device,
    model_config: BaseModel.Config,
    parallelize_fn: ParallelizeFunction,
    loss_fn: LossFunction,
) -> tuple[_PipelineSchedule, list[nn.Module], bool, bool]:
    pp_mesh = parallel_dims.get_mesh("pp")

    stem_fqns, body_items, output_fqns, output_blocks = _partition_segments(model)

    (
        num_virtual_stages,
        num_layers,
        input_weight,
        output_weight,
    ) = _get_pipeline_metadata(parallel_dims, parallelism, len(body_items))

    module_names_per_stage = parallelism.module_fqns_per_model_part
    if (
        module_names_per_stage is not None
        and parallelism.pipeline_parallel_balancing != "layer_count"
    ):
        logger.warning(
            "module_fqns_per_model_part is set: the explicit split takes "
            "precedence and pipeline_parallel_balancing="
            f"{parallelism.pipeline_parallel_balancing!r} is not applied."
        )
    block_fqns = {s.fqn for s in model.named_segments() if s.is_block}
    stage_costs = None
    if (
        module_names_per_stage is None
        and parallelism.pipeline_parallel_balancing == "flops"
    ):
        module_names_per_stage, stage_costs = flops_balanced_module_fqns(
            model,
            model_config,
            parallelism,
            training.seq_len,
            ac_config,
            num_virtual_stages,
            segments=(stem_fqns, body_items, output_fqns),
            block_fqns=block_fqns,
        )
    elif module_names_per_stage is None:
        if output_blocks:
            # The output tail carries blocks of its own (one per MTP head), so it
            # weighs more than the config's norm+lm_head weight alone.
            output_weight = _feasible_output_weight(
                num_layers,
                num_virtual_stages,
                input_weight,
                output_weight + output_blocks,
            )
        module_names_per_stage = _generate_fqn_per_model_part(
            num_virtual_stages,
            body_items,
            input_weight,
            output_weight,
            input_module_fqns=stem_fqns,
            output_module_fqns=output_fqns,
        )
    costs_are_measured_ms = stage_costs is not None and bool(
        parallelism.pipeline_parallel_module_costs_ms
    )
    if stage_costs is None:
        atoms = model_pipeline_units(
            model,
            model_config,
            training.seq_len,
            ac_config,
            segments=(stem_fqns, body_items, output_fqns),
            block_fqns=block_fqns,
        )
        unit_costs = {u.fqns[0]: u.cost for atom in atoms for u in atom}
        stage_costs = [
            sum(unit_costs.get(fqn, 0.0) for fqn in stage)
            for stage in module_names_per_stage
        ]
    stage_indices_per_rank = [
        _get_pp_rank_to_stage_indices_mapping(
            rank,
            parallel_dims.pp,
            parallelism.pipeline_parallel_schedule,
            len(module_names_per_stage),
        )
        for rank in range(parallel_dims.pp)
    ]
    log_pipeline_layout(
        module_names_per_stage=module_names_per_stage,
        stage_indices_per_rank=stage_indices_per_rank,
        pp_schedule=parallelism.pipeline_parallel_schedule,
        num_layers=num_layers,
        stage_costs=stage_costs,
        costs_are_measured_ms=costs_are_measured_ms,
        stage_blocks=[
            sum(fqn in block_fqns for fqn in stage) for stage in module_names_per_stage
        ],
        num_microbatches=(
            training.local_batch_size // parallelism.pipeline_parallel_microbatch_size
        ),
    )

    get_mesh_cb = _build_get_mesh_callback(parallel_dims)
    stages, model_parts = _pipeline_module_split(
        model,
        pp_mesh,
        parallelism.pipeline_parallel_schedule,
        device,
        module_names_per_stage,
        get_mesh=get_mesh_cb,
    )

    # For PP with looped schedules, each item in model_parts is one stage-model-chunk.
    # We need to iterate through model_parts to apply SPMD parallelisms, compilation,
    # optimizer, and checkpointing
    for i, m in enumerate(model_parts):
        # apply SPMD-style PT-D techniques
        m = parallelize_fn(
            m,
            parallel_dims=parallel_dims,
            training=training,
            parallelism=parallelism,
            compile_config=compile_config,
            ac_config=ac_config,
            dump_folder=dump_folder,
        )
        model_parts[i] = m
        # NOTE: this is to update the model in the stage
        #       in case the model is modified e.g. by torch.compile
        stages[i].submod = m

    pp_schedule = _build_pipeline_schedule(
        parallelism=parallelism,
        local_batch_size=training.local_batch_size,
        stages=stages,
        loss_fn=loss_fn,
    )

    # This is used in the train loop to determine whether to pass in the input_ids and labels
    has_first_stage = False
    has_last_stage = False
    for stage in stages:
        if stage.is_first:
            has_first_stage = True
        if stage.is_last:
            has_last_stage = True

    return pp_schedule, model_parts, has_first_stage, has_last_stage


def _partition_segments(
    model: nn.Module,
) -> tuple[list[str], list[list[str]], list[str], int]:
    """Split a model's forward-ordered segments into stem / body / output tail.

    Returns ``(stem_fqns, body_items, output_fqns, output_block_count)``:

    - **stem**: leading non-block segments, which run before any block.
    - **body**: one item per block, plus every non-block segment between blocks,
      kept with the block *before* it — such a segment is usually called from
      inside that block's forward (a tower's mergers), and for a plainly
      sequential one either side works. Hence one rule, not a per-model flag.
    - **tail**: everything from the *last* non-block run onward (``norm`` +
      ``lm_head``, plus any MTP heads). ``output_block_count`` is how many blocks
      it holds, so the splitter can weigh the last stage.

    Every FQN comes from the model's own tree, so renaming or adding a container
    needs no change here.
    """
    segments = list(model.named_segments())
    if not any(segment.is_block for segment in segments):
        raise ValueError(
            f"{type(model).__name__} exposes no repeated block to pipeline; "
            "a block class must inherit torchtitan.protocols.blocks.Block."
        )

    # Start of the last *run* of non-blocks, so the tail is ``norm`` and
    # ``lm_head`` together plus whatever follows.
    tail_start = max(i for i, s in enumerate(segments) if not s.is_block)
    while tail_start > 0 and not segments[tail_start - 1].is_block:
        tail_start -= 1
    tail = segments[tail_start:]
    lead = segments[:tail_start]

    first_block = next(i for i, s in enumerate(lead) if s.is_block)
    stem = [s.fqn for s in lead[:first_block]]

    # ``lead[first_block]`` is a block, so ``body[-1]`` always exists.
    body: list[list[str]] = []
    for segment in lead[first_block:]:
        if segment.is_block:
            body.append([segment.fqn])
        else:
            body[-1].append(segment.fqn)

    return (
        stem,
        body,
        [s.fqn for s in tail],
        sum(1 for s in tail if s.is_block),
    )


def _feasible_output_weight(
    num_layers: int, num_stages: int, input_weight: int, want: int
) -> int:
    """Largest weight up to ``want`` the splitter accepts: it caps the output
    weight at the minimum layers per stage."""
    weight = next(
        (
            w
            for w in range(want, 1, -1)
            if w <= (num_layers + input_weight + w) // num_stages
        ),
        1,
    )
    if weight != want:
        logger.warning(
            f"Output stage weighs {want} layers but only {weight} fit "
            f"{num_stages} stages over {num_layers}; it stays the heaviest."
        )
    return weight


def _get_pipeline_metadata(
    parallel_dims: ParallelDims,
    parallelism: ParallelismConfig,
    num_layers: int,
) -> tuple[int, int, int, int]:
    """Number of virtual stages, given how many blocks are splittable.

    Takes the count, not a model or config, so nothing here needs to know what a
    model calls its block container.
    """
    # Determine the number of virtual stages based on schedule type
    schedule_class = get_schedule_class(parallelism.pipeline_parallel_schedule)
    is_single_stage_schedule = issubclass(schedule_class, PipelineScheduleSingle)
    layers_per_stage = parallelism.pipeline_parallel_layers_per_stage

    # You can adjust these weights based on the computational cost of embeddings and output layers
    # Higher weights mean these modules are treated as "heavier" in the distribution
    input_weight = parallelism.pipeline_parallel_first_stage_less_layers
    output_weight = parallelism.pipeline_parallel_last_stage_less_layers

    # Calculate number of virtual stages
    if parallelism.pipeline_parallel_num_stages_per_rank is not None:
        # ParallelismConfig rejects combining this with layers_per_stage.
        num_virtual_stages = (
            parallelism.pipeline_parallel_num_stages_per_rank * parallel_dims.pp
        )
        stage_count_source = (
            "pipeline_parallel_num_stages_per_rank="
            f"{parallelism.pipeline_parallel_num_stages_per_rank}"
        )
    elif layers_per_stage is not None:
        # Calculate number of virtual stages needed (using ceiling division)
        # This allows for unequal distribution where stages can differ by at most 1 layer
        num_virtual_stages = math.ceil(
            (num_layers + input_weight + output_weight) / layers_per_stage
        )
        stage_count_source = (
            f"Model has {num_layers} layers with "
            f"pipeline_parallel_layers_per_stage={layers_per_stage}"
        )
    else:
        # Fallback to default behavior when neither knob is provided
        # For multi-stage schedules, default is 2 virtual stages per rank
        # For single-stage schedules, default is 1 virtual stage per rank
        stages_per_rank = 1 if is_single_stage_schedule else 2
        num_virtual_stages = parallel_dims.pp * stages_per_rank
        stage_count_source = None

    if stage_count_source is not None:
        # Validation: check stages per rank based on schedule type
        stage_distribution_info = (
            f"resulting in {num_virtual_stages=} across {parallel_dims.pp} PP ranks"
        )

        if num_virtual_stages % parallel_dims.pp != 0:
            raise ValueError(
                f"Number of virtual stages ({num_virtual_stages}) must be divisible by "
                f"pipeline parallel size ({parallel_dims.pp}). "
                f"{stage_count_source}. "
                f"Please adjust it to a value that results in a number of stages "
                f"divisible by {parallel_dims.pp}."
            )

        stages_per_rank = num_virtual_stages // parallel_dims.pp

        if is_single_stage_schedule and stages_per_rank != 1:
            raise ValueError(
                f"Single stage schedule requires exactly 1 stage per rank, but got {stages_per_rank} stages per rank. "
                f"{stage_count_source}, {stage_distribution_info}. "
                f"Please adjust it to achieve exactly 1 stage per rank."
            )

        if not is_single_stage_schedule and stages_per_rank < 2:
            raise ValueError(
                f"Multi-stage schedule requires at least 2 stages per rank, but got {stages_per_rank} stages per rank. "
                f"{stage_count_source}, {stage_distribution_info}. "
                f"Please adjust it to achieve at least 2 stages per rank."
            )
    return num_virtual_stages, num_layers, input_weight, output_weight


def _schedule_with_chunked_loss(
    schedule_class: type[_PipelineSchedule], loss_fn: ChunkedCELoss
) -> type[_PipelineSchedule]:
    """Subclass the schedule to pass sync policy only to real losses, never
    to torch's shape probes (which call the loss directly)."""

    class ChunkedLossSchedule(schedule_class):
        _n_microbatches: int
        _internal_losses: list[torch.Tensor]

        def _maybe_compute_loss(
            self, stage, output, target_mbs, mb_index, loss_kwargs=None
        ):
            if stage.is_last:
                loss_kwargs = dict(loss_kwargs or {})
                loss_kwargs["sync_all_reduce"] = (
                    loss_fn.is_last_accumulation_step
                    # count losses, not mb_index: CSV schedules reorder IDs
                    and len(self._internal_losses) == self._n_microbatches - 1
                )
            return super()._maybe_compute_loss(
                stage, output, target_mbs, mb_index, loss_kwargs
            )

    return ChunkedLossSchedule


def _build_pipeline_schedule(
    *,
    parallelism: ParallelismConfig,
    local_batch_size: int,
    stages: list[PipelineStage],
    loss_fn: Callable,
    backward_requires_autograd: bool = True,
) -> _PipelineSchedule:
    """Builds a pipeline schedule for the given job configuration and stages.

    Args:
        parallelism (ParallelismConfig): The parallelism configuration.
        local_batch_size (int): The local batch size for computing microbatches.
        stages (list[PipelineStage]): The stages to be scheduled.
        loss_fn (Callable): The loss function.
        backward_requires_autograd (bool): False for Graph PP, which runs
            explicit backward graphs instead of autograd.

    Returns:
        _PipelineSchedule: The pipeline schedule for the given stages.
    """
    pp_schedule_csv = parallelism.pipeline_parallel_schedule_csv

    # Validate that pp_schedule_csv is a valid path
    if pp_schedule_csv:
        if not os.path.isfile(pp_schedule_csv):
            raise FileNotFoundError(
                f"The specified path {pp_schedule_csv} does not exist or is not a file."
            )
        schedule_class = _PipelineScheduleRuntime
    else:
        schedule_class = get_schedule_class(parallelism.pipeline_parallel_schedule)

    looped_schedule = issubclass(schedule_class, PipelineScheduleMulti)
    microbatch_size = parallelism.pipeline_parallel_microbatch_size
    batch_size = local_batch_size
    # validate that the batch size is divisible by the microbatch_size otherwise we'll hang or error during training
    if batch_size % microbatch_size != 0:
        raise ValueError(
            f"Batch size {local_batch_size} must be divisible by microbatch_size {microbatch_size}. "
            "Update the config arguments for either batch_size or pipeline_parallel_microbatch_size."
        )
    n_microbatches = batch_size // microbatch_size
    # We expect that the number of local stages (`len(stages)`) is the same across all ranks
    num_total_stages = parallelism.pipeline_parallel_degree * len(stages)
    if n_microbatches < num_total_stages:
        logger.warning(
            f"Number of microbatches ({n_microbatches}) is less than the total number "
            f"of stages ({num_total_stages}) which may result in a bubble in the pipeline."
        )

    if schedule_class is PipelineScheduleSingle:
        raise ValueError(
            "PipelineScheduleSingle is an abstract base class. "
            "Use a concrete single-stage schedule such as GPipe or 1F1B."
        )
    make_schedule: Callable[..., _PipelineSchedule] = schedule_class
    if parallelism.pipeline_parallel_grad_reduce != "off":
        if not backward_requires_autograd:
            raise ValueError(
                "parallelism.pipeline_parallel_grad_reduce "
                "requires autograd backwards; Graph PP runs explicit "
                "backward graphs, so FSDP hooks never fire in backward."
            )
        if isinstance(loss_fn, ChunkedCELoss):
            make_schedule = _schedule_with_chunked_loss(schedule_class, loss_fn)
    if looped_schedule:
        schedule = make_schedule(
            stages,
            n_microbatches=n_microbatches,
            loss_fn=loss_fn,
            scale_grads=False,
            backward_requires_autograd=backward_requires_autograd,
        )
    else:
        schedule = make_schedule(
            stages[0],
            n_microbatches=n_microbatches,
            loss_fn=loss_fn,
            scale_grads=False,
        )
    logger.info(
        f"Using pipeline schedule {parallelism.pipeline_parallel_schedule} "
        f"with {n_microbatches} microbatches and {num_total_stages} stages."
    )

    if pp_schedule_csv:
        assert schedule_class in [
            PipelineScheduleSingle,
            PipelineScheduleMulti,
            _PipelineScheduleRuntime,
        ], (
            "Only PipelineScheduleSingle (single stage), PipelineScheduleMulti (multistage), "
            "and _PipelineScheduleRuntime support csv schedules"
        )
        assert isinstance(schedule, _PipelineScheduleRuntime)
        schedule._load_csv(pp_schedule_csv)

    if parallelism.pipeline_parallel_pool_recv_buffers:
        if isinstance(schedule, _PipelineScheduleRuntime):
            pool_pipeline_recv_buffers(schedule)
        else:
            logger.warning(
                "pipeline_parallel_pool_recv_buffers requires a "
                "_PipelineScheduleRuntime schedule (e.g. Interleaved1F1B); "
                "%s is not one, so recv buffer pooling is disabled.",
                type(schedule).__name__,
            )

    if parallelism.pipeline_parallel_grad_reduce != "off":
        install_per_microbatch_grad_reduce(
            schedule, scope=parallelism.pipeline_parallel_grad_reduce
        )

    return schedule


def _generate_fqn_per_model_part(
    num_stages: int,
    blocks: Sequence[str | Sequence[str]],
    input_weight: int = 1,
    output_weight: int = 1,
    *,
    input_module_fqns: Sequence[str],
    output_module_fqns: Sequence[str],
) -> list[list[str]]:
    """Distribute a model's parts across pipeline stages by block count.

    Args:
        num_stages: Number of pipeline stages.
        blocks: One entry per block, in forward order — its FQN, or a group of
            FQNs that must stay on one stage. Callers holding only a depth pass
            ``[f"layers.{i}" for i in range(n)]``.
        input_weight: Layer-equivalents the first stage's input parts cost.
        output_weight: Layer-equivalents the last stage's output parts cost.
        input_module_fqns: Parts co-located on the first stage. Required, not
            defaulted: a default would be this file guessing a model's shape.
        output_module_fqns: Parts co-located on the last stage.

    Returns:
        One list of module FQNs per stage.

    Example:
        >>> _generate_fqn_per_model_part(
        ...     2,
        ...     ["layers.0", "layers.1"],
        ...     input_module_fqns=["tok_embeddings"],
        ...     output_module_fqns=["norm", "lm_head"],
        ... )
        [['tok_embeddings', 'layers.0'], ['layers.1', 'norm', 'lm_head']]
    """
    if num_stages < 1:
        raise ValueError("Number of stages must be at least 1")

    items: list[list[str]] = [[b] if isinstance(b, str) else list(b) for b in blocks]
    num_layers = len(items)
    input_modules = list(input_module_fqns)
    output_modules = list(output_module_fqns)

    if num_stages == 1:
        return [
            input_modules + [fqn for item in items for fqn in item] + output_modules
        ]

    num_effective_layers = num_layers + input_weight + output_weight

    if num_stages > num_effective_layers:
        raise ValueError(
            f"Number of stages ({num_stages}) cannot be greater than effective layers ({num_effective_layers})"
        )

    layers_per_stage = num_effective_layers // num_stages
    extra_layers = num_effective_layers % num_stages

    # Feasibility check: Ensure at least 1 layer in each PP stage
    if layers_per_stage == 0:
        raise ValueError(
            f"Configuration would result in empty stages. "
            f"With {num_stages} stages and {num_effective_layers} effective layers "
            f"(num_layers={num_layers} + input_weight={input_weight} + output_weight={output_weight}), "
            f"each stage would get {layers_per_stage} layers on average. "
            f"Reduce num_stages or increase num_layers/weights."
        )

    # Balance check: Ensure weights don't exceed minimum layers per stage
    if input_weight > layers_per_stage:
        raise ValueError(
            f"input_weight ({input_weight}) exceeds minimum layers per stage ({layers_per_stage})."
        )
    if output_weight > layers_per_stage:
        raise ValueError(
            f"output_weight ({output_weight}) exceeds minimum layers per stage ({layers_per_stage})."
        )

    module_names_per_stage: list[list[str]] = []
    current_layer = 0

    for stage_idx in range(num_stages):
        stage_modules: list[str] = []

        effective_layers_for_stage = layers_per_stage
        if stage_idx < extra_layers:
            effective_layers_for_stage += 1

        if stage_idx == 0:
            stage_modules.extend(input_modules)
            remaining = effective_layers_for_stage - input_weight
        elif stage_idx == num_stages - 1:
            remaining = effective_layers_for_stage - output_weight
        else:
            remaining = effective_layers_for_stage

        for _ in range(remaining):
            if current_layer < num_layers:
                stage_modules.extend(items[current_layer])
                current_layer += 1

        if stage_idx == num_stages - 1:
            stage_modules.extend(output_modules)

        module_names_per_stage.append(stage_modules)

    return module_names_per_stage


_CONTAINERS = (nn.ModuleDict, nn.ModuleList, ModuleDict, ModuleList)


def _prune_to_names(module: nn.Module, keep: set[str], prefix: str = "") -> None:
    """Drop everything under ``module`` that no name in ``keep`` reaches.

    Names are full FQNs relative to the split root, so this recurses to any
    depth: a stage owning ``mtp.layers.0`` keeps ``mtp``, trims its container to
    that head, and keeps the head whole. Recursion stops at an explicitly named
    module, whose internals are never named and must not be pruned.

    Dropped plain children become ``None`` and dropped containers become empty,
    the convention the model forwards already read.
    """
    for name, child in list(module.named_children()):
        fqn = f"{prefix}.{name}" if prefix else name
        if fqn in keep:
            continue
        reached = any(name.startswith(f"{fqn}.") for name in keep)
        if isinstance(child, _CONTAINERS):
            if not reached:
                empty = (
                    ModuleDict()
                    if isinstance(child, (nn.ModuleDict, ModuleDict))
                    else ModuleList()
                )
                setattr(module, name, empty)
                continue
            wanted = {
                n[len(fqn) + 1 :].split(".", 1)[0]
                for n in keep
                if n.startswith(f"{fqn}.")
            }
            if isinstance(child, (nn.ModuleDict, ModuleDict)):
                for key in [k for k in child.keys() if k not in wanted]:
                    del child[key]
                _prune_to_names(child, keep, fqn)
            else:
                kept = [(i, m) for i, m in enumerate(child) if str(i) in wanted]
                # Prune inside each kept entry *before* rebuilding the list:
                # afterwards the entries are renumbered from 0, so their original
                # indices no longer name them and nothing would match ``keep``.
                for index, entry in kept:
                    if f"{fqn}.{index}" not in keep:
                        _prune_to_names(entry, keep, f"{fqn}.{index}")
                setattr(module, name, ModuleList([m for _, m in kept]))
        elif not reached:
            setattr(module, name, None)
        else:
            _prune_to_names(child, keep, fqn)


def _split_module(
    whole_model: nn.Module,
    module_names: list[str],
) -> nn.Module:
    """
    Splits a whole model into a module based on the specified module names.

    Args:
        whole_model: The complete model to be split
        module_names: Full FQNs (relative to the model root) to keep. Nested
            containers are supported, so a wrapper's ``decoder.layers.3`` or an
            MTP head's ``mtp.layers.0`` name works the same as ``layers.3``.

    Returns:
        The split module

    Example usage:
        module_names = ["tok_embeddings", "layers.0", "layers.1", "norm", "lm_head"]
        split_module(whole_model, module_names)
    """
    model = copy.deepcopy(whole_model)
    _prune_to_names(model, set(module_names))
    return model


_METADATA_PROBE = False


def in_metadata_probe() -> bool:
    """Whether the pipeline schedule's metadata probe is running."""
    return _METADATA_PROBE


@contextmanager
def metadata_probe(stage_model: nn.Module) -> Generator[None]:
    """Mark the metadata probe and run the stage's compiled blocks eagerly."""
    global _METADATA_PROBE
    window = EagerWindow([stage_model], owner="metadata_probe")
    try:
        _METADATA_PROBE = True
        window.open()
        yield
    finally:
        window.close()
        _METADATA_PROBE = False


class MetadataSilentStage(RecvPoolStage):
    """A stage whose metadata probe leaves no trace in the training statistics.

    The probe materializes its inputs as ``torch.empty`` and runs a real
    forward+backward through the model, so the aux/z splices would record
    that garbage as if it were a microbatch — NaN in the step-1 ``aux/*``
    totals on the loss stage — and the monitoring observers would count it
    as a microbatch. The loss's own components are protected by
    ``keep_microbatches``; everything else reads ``in_metadata_probe``.
    """

    def _forward_metadata_inference(self, *args, **kwargs):
        with pipeline_metadata_inference(), metadata_probe(self.submod):
            return super()._forward_metadata_inference(*args, **kwargs)

    def _backward_metadata_inference(self, *args, **kwargs):
        with metadata_probe(self.submod), suppress_aux_recording():
            return super()._backward_metadata_inference(*args, **kwargs)


def _get_pp_rank_to_stage_indices_mapping(
    pp_rank: int,
    pp_degree,
    pp_schedule: str,
    num_stages: int,
) -> tuple[int, ...]:
    """
    Returns a mapping from PP rank to stage indices for the given pipeline schedule.

    Args:
        pp_rank: Pipeline parallel rank
        pp_degree: Number of pipeline parallel ranks
        pp_schedule: Name of pipeline parallelism schedule
        num_stages: Number of pipeline stages

    Returns:
        Mapping from PP rank to stage indices
    """
    schedule_class = get_schedule_class(pp_schedule)
    style = (
        "v" if schedule_class in (ScheduleZBVZeroBubble, ScheduleDualPipeV) else "loop"
    )
    assert num_stages % pp_degree == 0, (
        f"num_stages {num_stages} must be evenly divisible by pp_degree {pp_degree}"
    )
    stages_per_rank = num_stages // pp_degree
    if style == "loop":
        return tuple(pp_rank + s * pp_degree for s in range(stages_per_rank))
    elif style == "v":
        assert stages_per_rank == 2, (
            f"v schedules assume 2 stages per rank, got {stages_per_rank}"
        )
        stage_v_pairs = list(
            zip(range(pp_degree), range(num_stages - 1, pp_degree - 1, -1))
        )
        return tuple(stage_v_pairs[pp_rank])
    else:
        raise ValueError(f"Unknown style {style}")


def _pipeline_module_split(
    whole_model: nn.Module,
    pp_mesh: DeviceMesh,
    pp_schedule: str,
    device: torch.device,
    module_names_per_stage: list[list[str]],
    get_mesh: Callable | None = None,
) -> tuple[list[PipelineStage], list[nn.Module]]:
    """Create pipeline stages based on specified module names for each stage.

    Also used by Graph PP to split the model into per-stage chunks before
    exporting joint forward/backward graphs for each stage.

    Some model restrictions include:
    - forward() method should tolerate deleted layers
    - weight initialization methods should tolerate deleted layers

    Args:
        whole_model: The complete model to be split
        pp_mesh: Pipeline parallel device mesh
        pp_schedule: Name of pipeline parallelism schedule
        device: Device
        module_names_per_stage: List of lists, where each inner list contains the module names
                               that should be included in that stage. Module names should be
                               dot-separated paths. Examples:
                               - "tok_embeddings" for token embeddings
                               - "layers.0", "layers.1" for specific transformer layers
                               - "norm" for the final normalization layer
                               - "lm_head" for the output projection layer

    Returns:
        Tuple of (stages, models) where stages are PipelineStage objects and models are the
        corresponding model chunks

    Example usage:
        module_names_per_stage = [
            ["tok_embeddings", "layers.0"],     # Stage 0: embeddings + first layer
            ["layers.1", "layers.2"],           # Stage 1: middle layers
            ["norm", "lm_head"]                  # Stage 2: final norm + output
        ]
    """
    pp_rank = pp_mesh.get_local_rank()
    pp_degree = pp_mesh.size()
    num_stages = len(module_names_per_stage)
    stages = []
    models = []
    pp_rank_to_stage_indices = _get_pp_rank_to_stage_indices_mapping(
        pp_rank, pp_degree, pp_schedule, num_stages
    )
    for stage_idx in pp_rank_to_stage_indices:
        module_names = module_names_per_stage[stage_idx]
        model_chunk = _split_module(whole_model, module_names)
        stage = MetadataSilentStage(
            model_chunk,
            stage_idx,
            num_stages,
            device,
            group=pp_mesh.get_group("pp"),
            get_mesh=get_mesh,
        )
        logger.info(
            f"PP rank {pp_rank} is building stage_idx {stage_idx} "
            f"with modules {module_names}"
        )
        stages.append(stage)
        models.append(model_chunk)

    return stages, models
