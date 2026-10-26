from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

SEQUENCES_PER_MICROBATCH = 12
SEQUENCE_LENGTH = 4096
TOKENS_PER_MICROBATCH = SEQUENCES_PER_MICROBATCH * SEQUENCE_LENGTH
TOP_K = 8
RANKS = 8
MICROBATCHES = 2


def main(window: Path) -> None:
    registry = {metric["metric_id"]: metric["name"]
                for metric in json.loads((window.parent.parent / "metrics.json").read_text())}
    manifest = json.loads((window / "manifest.json").read_text())
    print("manifest:", {key: value for key, value in manifest.items() if key != "steps"})

    counts = pq.read_table(window / "expert_counts.parquet", partitioning=None).to_pylist()
    vectors = pq.read_table(window / "router_vectors.parquet", partitioning=None).to_pylist()
    scalars = pq.read_table(window / "scalars.parquet", partitioning=None).to_pylist()
    print("rows:", {"expert_counts": len(counts), "router_vectors": len(vectors), "scalars": len(scalars)})
    print("metrics per family:")
    for name, rows in (("expert_counts", counts), ("router_vectors", vectors), ("scalars", scalars)):
        print(" ", name, dict(Counter(registry[row["metric_id"]] for row in rows)))

    steps = sorted({row["step"] for row in counts})
    dispatcher_expected = RANKS * TOKENS_PER_MICROBATCH * TOP_K
    router_expected = RANKS * MICROBATCHES * TOKENS_PER_MICROBATCH * TOP_K
    bad_dispatcher = [row for row in counts
                      if registry[row["metric_id"]] == "dispatcher/tokens_per_local_expert"
                      and sum(row["values"]) != dispatcher_expected]
    router_rows = [row for row in counts if registry[row["metric_id"]] == "router/local_tokens_before_bias_update"]
    bad_router = [row for row in router_rows if sum(row["values"]) != router_expected]
    zero_router = [row for row in router_rows if sum(row["values"]) == 0]
    incomplete = [row for row in counts if row["contributors"] != row["expected_contributors"]]
    bias_rows = [row for row in vectors if registry[row["metric_id"]] == "router/expert_bias_before_update"]
    print(f"steps {steps[0]}..{steps[-1]} ({len(steps)})")
    print(f"dispatcher rows with sum != {dispatcher_expected}: {len(bad_dispatcher)} of "
          f"{sum(1 for row in counts if registry[row['metric_id']] == 'dispatcher/tokens_per_local_expert')}")
    print(f"router rows with sum != {router_expected}: {len(bad_router)} of {len(router_rows)}; "
          f"all-zero router rows: {len(zero_router)}")
    print(f"rows with contributors != expected: {len(incomplete)}")
    print(f"expert_bias rows not identical across ranks: {sum(1 for row in bias_rows if not row['identical'])} "
          f"of {len(bias_rows)}")
    if bad_dispatcher:
        sample = bad_dispatcher[0]
        print("  sample bad dispatcher row:", {key: value for key, value in sample.items() if key != "values"},
              "sum", sum(sample["values"]))
    losses = [row for row in scalars if registry[row["metric_id"]] == "loss/total"]
    print("loss/total rows:", len(losses), "ranks", sorted({row["rank"] for row in losses}),
          "microbatches", sorted({row["microbatch"] for row in losses}))
    microbatch_counts_by_step = {}
    for row in scalars:
        if registry[row["metric_id"]] == "sampler/microbatches":
            microbatch_counts_by_step.setdefault(row["step"], set()).add(row["values"][0])
    print("sampler/microbatches values:",
          sorted({count for step_counts in microbatch_counts_by_step.values() for count in step_counts}))
    grad = [row["values"][0] for row in scalars if registry[row["metric_id"]] == "grad/clip_total_norm"]
    print("grad/clip_total_norm rows:", len(grad), "min/max", (min(grad), max(grad)) if grad else None)
    dead = [row["values"][0] for row in scalars if registry[row["metric_id"]] == "summary/dead_experts"]
    print("summary/dead_experts rows:", len(dead), "max", max(dead) if dead else None)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
