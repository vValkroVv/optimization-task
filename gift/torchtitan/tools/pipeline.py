# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence

import torch.distributed as dist

from torchtitan.tools.logging import logger


def _format_modules_with_ranges(fqns: Sequence[str]) -> str:
    """Collapse consecutive numbered FQNs: layers.0..layers.5 -> layers.0-5."""
    out = []
    i = 0
    while i < len(fqns):
        prefix, _, index = fqns[i].rpartition(".")
        j = i
        while prefix and index.isdigit() and j + 1 < len(fqns):
            nxt_prefix, _, nxt_index = fqns[j + 1].rpartition(".")
            last_index = fqns[j].rpartition(".")[2]
            if (
                nxt_prefix != prefix
                or not nxt_index.isdigit()
                or int(nxt_index) != int(last_index) + 1
            ):
                break
            j += 1
        if j > i:
            out.append(f"{prefix}.{index}-{fqns[j].rpartition('.')[2]}")
        else:
            out.append(fqns[i])
        i = j + 1
    return ", ".join(out)


def log_pipeline_layout(
    *,
    module_names_per_stage: list[list[str]],
    stage_indices_per_rank: Sequence[Sequence[int]],
    pp_schedule: str,
    num_layers: int,
    stage_costs: Sequence[float],
    stage_blocks: Sequence[int],
    num_microbatches: int,
    costs_are_measured_ms: bool = False,
) -> None:
    """Log stage/rank costs and a full-AC activation proxy on global rank 0."""
    if dist.is_initialized() and dist.get_rank() != 0:
        return

    num_stages = len(module_names_per_stage)
    mean_cost = sum(stage_costs) / num_stages
    cost_header = "cost, ms" if costs_are_measured_ms else "cost, x mean"

    def fmt_cost(cost: float) -> str:
        if costs_are_measured_ms:
            return f"{cost:.1f}"
        return f"{cost / mean_cost:.3f}" if mean_cost else "-"

    # Peak saved microbatches for single-stage schedules; other schedules differ.
    in_flight = (
        [min(num_microbatches, num_stages - i) for i in range(num_stages)]
        if pp_schedule == "1F1B"
        else [num_microbatches] * num_stages
        if pp_schedule == "GPipe"
        else None
    )
    activations = (
        [blocks * count for blocks, count in zip(stage_blocks, in_flight)]
        if in_flight is not None
        else None
    )
    headers = (
        "PP rank",
        "stage",
        "#mods",
        cost_header,
        "blocks x in-flight",
        "modules",
    )
    rows = [
        (
            str(rank),
            str(stage),
            str(len(module_names_per_stage[stage])),
            fmt_cost(stage_costs[stage]),
            str(activations[stage]) if activations is not None else "-",
            _format_modules_with_ranges(module_names_per_stage[stage]),
        )
        for rank, indices in enumerate(stage_indices_per_rank)
        for stage in indices
    ]
    widths = [max(map(len, cells)) for cells in zip(headers, *rows)]
    border = "+" + "+".join("-" * (w + 2) for w in widths) + "+"

    def fmt_row(cells: Sequence[str]) -> str:
        return (
            "|"
            + "|".join(f" {cell:<{width}} " for cell, width in zip(cells, widths))
            + "|"
        )

    lines = [
        f" schedule={pp_schedule} | pp_degree={len(stage_indices_per_rank)} | "
        f"num_virtual_stages={num_stages} | num_layers={num_layers}",
        border,
        fmt_row(headers),
        border,
    ]
    for i, row in enumerate(rows):
        if i and row[0] != rows[i - 1][0]:
            lines.append(border)
        lines.append(fmt_row(row))
    lines.append(border)
    lines.append(
        " Per-rank cost: "
        + ", ".join(
            f"rank {rank}: {fmt_cost(sum(stage_costs[i] for i in indices))}"
            for rank, indices in enumerate(stage_indices_per_rank)
        )
    )
    if activations is not None:
        lines.append(
            " Per-rank blocks x in-flight: "
            + ", ".join(
                f"rank {rank}: {sum(activations[i] for i in indices)}"
                for rank, indices in enumerate(stage_indices_per_rank)
            )
        )
    lines.append(
        " blocks x in-flight: full-AC activation proxy, not bytes; "
        "'-' for schedules other than 1F1B/GPipe."
    )
    logger.info("Pipeline parallel module layout:\n" + "\n".join(lines))
