# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FLOPs-based pipeline stage balancing: minimize the bottleneck stage cost.

Costs come from the config ``fwd_flops_per_token`` protocol; see
``ParallelismConfig.pipeline_parallel_balancing`` for the semantics."""

import itertools
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from torch import nn

from torchtitan.config import ActivationCheckpointConfig, ParallelismConfig
from torchtitan.config.schema.protocols.module import iter_module_configs
from torchtitan.tools.logging import logger

if TYPE_CHECKING:
    from torchtitan.protocols.model import BaseModel

__all__ = [
    "PipelineUnit",
    "flops_balanced_module_fqns",
    "model_pipeline_units",
    "partition_minimax",
]

# Backward matmul cost (dL/dX + dL/dW) relative to forward.
_BWD_FACTOR = 2.0
# Extra forward recompute per AC mode, as a fraction of forward FLOPs. The 0.5
# is a calibration constant (the real fraction depends on the AC policy);
# correct systematic gaps via pipeline_parallel_module_costs_ms.
_AC_RECOMPUTE_FACTOR = {
    "none": 0.0,
    "full": 1.0,
    "selective": 0.5,
    "memory_budget": 0.5,
}

# (stem fqns, body items, tail fqns) as returned by ``_partition_segments``.
Segments = tuple[list[str], list[list[str]], list[str]]


@dataclass(frozen=True, slots=True)
class PipelineUnit:
    """An unsplittable module run with its estimated relative cost."""

    fqns: tuple[str, ...]
    cost: float


def model_pipeline_units(
    model: nn.Module,
    model_config: "BaseModel.Config",
    seq_len: int,
    ac_config: ActivationCheckpointConfig,
    *,
    segments: Segments,
    block_fqns: set[str],
) -> list[list[PipelineUnit]]:
    """Cost-annotated atoms — unit groups a stage boundary cannot cross: the
    stem run, each body item, the tail run.

    Backward and AC recompute (blocks only, mirroring ``apply_ac``) enter as
    cost multiples; the shared head is priced once per prediction head.
    """
    from torchtitan.models.utils import (
        get_blockwise_nparams_and_flops,
        NUM_FLOPS_PER_FWD_FLOP,
        prediction_head_costing,
    )
    from torchtitan.config.schema.models.common.decoder_blocks.decoder import (
        DecoderConfig,
    )

    if not isinstance(model_config, DecoderConfig) or not model_config.materialized:
        raise ValueError("Analytical pipeline balancing requires a decoder model")

    if ac_config.mode == "memory_budget":
        logger.warning(
            "pipeline balancing approximates memory_budget AC recompute cost "
            "with the selective-AC estimate"
        )
    ac_factor = _AC_RECOMPUTE_FACTOR[ac_config.mode]

    stem_fqns, body_items, tail_fqns = segments
    all_fqns = [*stem_fqns, *itertools.chain(*body_items), *tail_fqns]
    cfg_by_fqn = dict(iter_module_configs(model_config))

    def fwd_flops(fqn: str) -> float | None:
        try:
            cfg = cfg_by_fqn.get(fqn)
            return cfg.fwd_flops_per_token(seq_len) if cfg else None
        except NotImplementedError:
            return None

    priced = {fqn: fwd_flops(fqn) for fqn in all_fqns}
    if unknown := [fqn for fqn, value in priced.items() if value is None]:
        logger.info(
            f"pipeline balancing prices modules without a fwd_flops_per_token "
            f"implementation as free: {unknown}"
        )
    fwd = {fqn: value or 0.0 for fqn, value in priced.items()}

    # The segment walk sees the shared head once; the loss runs it per prediction.
    head_fqn, num_prediction_heads = prediction_head_costing(model_config)

    def cost(fqn: str) -> float:
        scale = 1.0 + _BWD_FACTOR + (ac_factor if fqn in block_fqns else 0.0)
        return fwd[fqn] * scale * (num_prediction_heads if fqn == head_fqn else 1)

    def unit(fqns: Sequence[str]) -> PipelineUnit:
        return PipelineUnit(fqns=tuple(fqns), cost=sum(cost(f) for f in fqns))

    atoms = [[unit([fqn]) for fqn in stem_fqns]] if stem_fqns else []
    atoms += [[unit(item)] for item in body_items]
    if tail_fqns:
        atoms.append([unit([fqn]) for fqn in tail_fqns])

    # The walk should re-derive the MFU total; a gap means misreported or
    # missing ``fwd_flops_per_token``.
    _, flops_per_token = get_blockwise_nparams_and_flops(model_config, model, seq_len)
    fwd_total = flops_per_token / NUM_FLOPS_PER_FWD_FLOP
    walk_total = sum(
        fwd[fqn] * (num_prediction_heads if fqn == head_fqn else 1) for fqn in all_fqns
    )
    if fwd_total > 0 and abs(fwd_total - walk_total) > 0.02 * fwd_total:
        logger.warning(
            f"pipeline balancing prices {walk_total / fwd_total:.1%} of the "
            "model's forward FLOPs from module configs; the analytical FLOPs "
            "estimates are unreliable for this model — prefer measured "
            "per-module times via pipeline_parallel_module_costs_ms."
        )
    return atoms


def flops_balanced_module_fqns(
    model: nn.Module,
    model_config: "BaseModel.Config",
    parallelism: ParallelismConfig,
    seq_len: int,
    ac_config: ActivationCheckpointConfig,
    num_stages: int,
    *,
    segments: Segments,
    block_fqns: set[str],
) -> tuple[list[list[str]], list[float]]:
    """``(module_fqns_per_model_part, per-stage costs)`` minimizing the
    costliest stage."""
    if (
        parallelism.pipeline_parallel_first_stage_less_layers != 1
        or parallelism.pipeline_parallel_last_stage_less_layers != 1
    ):
        logger.warning(
            "pipeline_parallel_first/last_stage_less_layers are ignored with "
            "pipeline_parallel_balancing='flops'."
        )
    atoms = model_pipeline_units(
        model,
        model_config,
        seq_len,
        ac_config,
        segments=segments,
        block_fqns=block_fqns,
    )
    if parallelism.pipeline_parallel_module_costs_ms:
        if ac_config.mode != "none":
            logger.info(
                "pipeline_parallel_module_costs_ms replaces the FLOPs estimates: "
                "measured costs must already include AC recompute time."
            )
        atoms = _apply_costs(atoms, parallelism.pipeline_parallel_module_costs_ms)
    elif not any(u.cost for atom in atoms for u in atom):
        raise ValueError(
            "no pipeline module reports fwd_flops_per_token, so FLOPs "
            "balancing has nothing to balance; set "
            "pipeline_parallel_module_costs_ms with measured times instead."
        )
    stages = partition_minimax(atoms, num_stages)
    unit_cost = {u.fqns[0]: u.cost for atom in atoms for u in atom}
    stage_costs = [sum(unit_cost.get(f, 0.0) for f in stage) for stage in stages]
    return stages, stage_costs


def _apply_costs(
    atoms: Sequence[Sequence[PipelineUnit]],
    costs_ms: dict[str, float],
) -> list[list[PipelineUnit]]:
    """Replace unit costs with user-measured step times (only ratios matter);
    a key matches a unit FQN exactly or as a dot-prefix, first match wins, and
    coverage must be exact both ways."""
    used: set[str] = set()
    uncovered: list[str] = []

    def cost(unit: PipelineUnit) -> float:
        for key, value in costs_ms.items():
            if any(f == key or f.startswith(key + ".") for f in unit.fqns):
                if value < 0:
                    raise ValueError(
                        f"pipeline_parallel_module_costs_ms[{key!r}] is "
                        f"negative: {value}"
                    )
                used.add(key)
                return value
        uncovered.append(unit.fqns[0])
        return 0.0

    out = [[replace(u, cost=cost(u)) for u in atom] for atom in atoms]
    if uncovered:
        raise ValueError(
            "pipeline_parallel_module_costs_ms must cover every pipeline "
            f"module; missing costs for: {uncovered}"
        )
    if unused := [key for key in costs_ms if key not in used]:
        raise ValueError(
            "pipeline_parallel_module_costs_ms keys match no pipeline module "
            f"(typo?): {unused}"
        )
    if not any(u.cost for atom in out for u in atom):
        raise ValueError("pipeline_parallel_module_costs_ms costs are all zero")
    return out


def partition_minimax(
    atoms: Sequence[Sequence[PipelineUnit]], num_stages: int
) -> list[list[str]]:
    """Split atoms into ``num_stages`` contiguous non-empty stages minimizing
    the maximum stage cost, with the last stage the lightest (it gates every
    pipeline fill and drain wave; unsatisfiable falls back to the minimal last
    stage). Ties place slack late, keeping early stages (peak in-flight
    activations under 1F1B) light."""
    if num_stages < 1:
        raise ValueError(f"num_stages must be >= 1, got {num_stages}")
    if len(atoms) < num_stages:
        raise ValueError(
            f"Cannot split {len(atoms)} unsplittable module groups "
            f"into {num_stages} non-empty stages."
        )
    costs = [sum(u.cost for u in atom) for atom in atoms]
    bounds = [0, *_partition_last_stage_lightest(costs, num_stages), len(atoms)]
    return [
        [fqn for atom in atoms[start:end] for u in atom for fqn in u.fqns]
        for start, end in itertools.pairwise(bounds)
    ]


def _partition_last_stage_lightest(
    costs: Sequence[float], num_stages: int
) -> list[int]:
    """Boundaries of the best split whose last stage is the lightest: the last
    boundary is enumerated (lightest tail first, so exact ties keep it), the
    head is split with a per-stage floor of the tail's cost."""
    n = len(costs)
    if num_stages == 1:
        return []
    # Float-sum slack so an exactly-uniform split stays feasible.
    tol = 1e-9 * max(1.0, abs(sum(costs)))
    w_last = num_stages + 1.0
    best_key, best_bounds = None, None
    tail = 0.0
    for j in range(n - 1, num_stages - 2, -1):
        tail += costs[j]
        solved = _partition_lexicographic(costs[:j], num_stages - 1, floor=tail - tol)
        if solved is None:
            # The tail only grows and the head only shrinks from here on, so
            # no shorter head can satisfy the (larger) floor either.
            break
        (head_max, head_sq, head_weighted), bounds = solved
        key = (
            max(head_max, tail),
            head_sq + tail * tail,
            head_weighted + w_last * tail,
        )
        if best_key is None or key < best_key:
            best_key, best_bounds = key, [*bounds, j]
    if best_bounds is None:
        logger.info(
            "pipeline balancing: the output tail alone outweighs the lightest "
            "achievable stage; keeping the last stage minimal."
        )
        solved = _partition_lexicographic(costs[: n - 1], num_stages - 1)
        assert solved is not None  # len(atoms) >= num_stages was checked
        best_bounds = [*solved[1], n - 1]
    return best_bounds


def _partition_lexicographic(
    costs: Sequence[float], num_stages: int, floor: float = 0.0
) -> tuple[tuple[float, float, float], list[int]] | None:
    """Best (key, boundaries) minimizing (max stage cost, sum of squared stage
    costs, position-weighted cost) lexicographically with every stage cost at
    least ``floor``; None when no such split exists. The third component only
    breaks exact ties, placing slack on late stages first."""
    prefix = [0.0, *itertools.accumulate(costs)]
    n = len(costs)
    inf = float("inf")
    unsolved = (inf, inf, inf)
    # best[k][i]: key of the best split of the first i atoms into k stages.
    best = [[unsolved] * (n + 1) for _ in range(num_stages + 1)]
    split = [[0] * (n + 1) for _ in range(num_stages + 1)]
    best[0][0] = (0.0, 0.0, 0.0)
    for k in range(1, num_stages + 1):
        # Slack preference for 1-based stage k: later stages cheapest.
        slack_weight = num_stages - k + 1.0
        # Leave enough atoms for the remaining stages on both sides.
        for i in range(k, n - (num_stages - k) + 1):
            best_key, best_j = unsolved, -1
            for j in range(k - 1, i):
                stage_cost = prefix[i] - prefix[j]
                if stage_cost < floor:
                    break  # stage_cost only shrinks as j grows
                head_max, head_sq, head_weighted = best[k - 1][j]
                if head_max == inf:
                    continue
                key = (
                    max(head_max, stage_cost),
                    head_sq + stage_cost * stage_cost,
                    head_weighted + slack_weight * stage_cost,
                )
                if key < best_key:
                    best_key, best_j = key, j
            best[k][i], split[k][i] = best_key, best_j
    if best[num_stages][n][0] == inf:
        return None
    boundaries = []
    i = n
    for k in range(num_stages, 1, -1):
        i = split[k][i]
        boundaries.append(i)
    boundaries.reverse()
    return best[num_stages][n], boundaries
