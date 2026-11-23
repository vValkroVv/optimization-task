#!/usr/bin/env python3
"""Summarize all-rank MTP FFN boundary receipts without a numerical tolerance."""

import argparse
import json
from pathlib import Path


PATHS = {
    "suffix": (
        "backward/component_cotangent",
        "backward/final_norm.out",
        "backward/final_norm.in",
        "backward/ffn_residual_scale.out",
        "backward/ffn_residual_scale.in",
        "backward/post_ffn_norm.out",
        "backward/post_ffn_norm.in",
        "backward/ffn.out",
    ),
    "ffn_gate": (
        "backward/ffn.out",
        "backward/ffn.out.used",
        "backward/gate_gemm.out",
        "backward/ffn.in",
    ),
    "ffn_up": (
        "backward/ffn.out",
        "backward/ffn.out.used",
        "backward/up_gemm.out",
        "backward/ffn.in",
    ),
    "input_projection": ("backward/input_proj.out", "backward/input_proj.in"),
    "hidden_norm": (
        "backward/hidden_norm.out",
        "backward/hidden_norm.in",
        "backward/hidden",
    ),
    "token_norm": (
        "backward/token_norm.out",
        "backward/token_norm.in",
        "backward/token_embedding",
    ),
}


def require(condition, description):
    if not condition:
        raise ValueError(description)


def compare_rows(records):
    exact = [row["bitwise_equal"] for row in records]
    count = [row["unequal_elements"] for row in records]
    finite = [row["nonfinite_elements"] for row in records]
    require(exact == [value == 0 for value in count], "Inconsistent bitwise flags")
    absolute = [row["max_abs"] for row in records if row["max_abs"] is not None]
    return {
        "bitwise_all_ranks": all(exact),
        "different_ranks": [rank for rank, yes in enumerate(exact) if not yes],
        "unequal_elements_per_rank_minmax": [min(count), max(count)],
        "nonfinite_elements_per_rank_max": max(finite),
        "maximum_absolute_difference": max(absolute, default=None),
    }


def summarize(folder):
    receipts = []
    for rank in range(8):
        path = folder / f"rank{rank}.json"
        row = json.loads(path.read_text())["results"]
        require(row["rank"] == rank, f"Invalid rank in {path}")
        receipts.append(row)
    rank0 = receipts[0]
    keys = set(rank0["arms"]["A"]["tensors"])
    for rank, row in enumerate(receipts):
        require(
            row["source"] == rank0["source"]
            and row["phase"] == rank0["phase"]
            and row["weights_sha256"] == rank0["weights_sha256"],
            f"Rank {rank} identity differs",
        )
        arms = row["arms"]
        require(list(arms) == ["A", "AA", "B"], f"Rank {rank} comparison arms differ")
        require(
            all(set(record["tensors"]) == keys for record in arms.values()),
            f"Rank {rank} tensor coverage differs",
        )
        require(
            all(arms[key]["activation"]["split"] == (key == "B") for key in arms),
            f"Rank {rank} split activation differs",
        )
        require(
            all(
                record["input_fixture"] == arms["A"]["input_fixture"]
                for record in arms.values()
            ),
            f"Rank {rank} fixture differs between arms",
        )

    compared = {
        key: {
            arm: compare_rows([row["arms"][arm]["tensors"][key] for row in receipts])
            for arm in ("AA", "B")
        }
        for key in sorted(keys)
    }
    path_findings = {}
    for path, indices in PATHS.items():
        observed = [key for key in indices if key in compared]
        if not observed:
            continue
        first = next(
            (key for key in observed if not compared[key]["B"]["bitwise_all_ranks"]),
            None,
        )
        path_findings[path] = {
            "observed_backward_boundaries": observed,
            "first_observed_split_difference": first,
            "repeat_bitwise_at_first_difference": compared[first]["AA"][
                "bitwise_all_ranks"
            ]
            if first
            else None,
            "any_nonfinite": any(
                compared[key][arm]["nonfinite_elements_per_rank_max"]
                for key in observed
                for arm in ("AA", "B")
            ),
        }
    return {
        "phase": rank0["phase"],
        "source": rank0["source"],
        "ranks": len(receipts),
        "diagnostic": rank0["diagnostic"],
        "tensors": compared,
        "observed_backward_paths": path_findings,
        "baseline_tensor_sha256_by_rank": [
            row["arms"]["A"]["tensors"] for row in receipts
        ],
        "scope": "A/A and A/B raw comparison at observed boundaries; gaps and instrumented compilation remain",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase_dirs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence = [summarize(folder) for folder in args.phase_dirs]
    with args.output.open("x") as file:
        json.dump(evidence, file, indent=2)
        file.write("\n")


if __name__ == "__main__":
    main()
