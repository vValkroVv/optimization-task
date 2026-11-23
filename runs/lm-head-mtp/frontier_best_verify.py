"""Validate the allocator and native gather path in the full matched matrix."""

import json
from pathlib import Path
import sys

from frontier_verify import read, verify


def main():
    base = Path(sys.argv[1])
    rows = verify(base)
    for row in rows:
        n = row["chunks"]
        lane = base / f"chunks-{n}"
        for path in sorted(lane.glob("*.json")):
            if path.name.endswith(".digest.json"):
                continue
            receipt = read(path)
            best = receipt["stack_arm"] == "stack"
            assert receipt["full_arm"] == ("memory" if best else "control")
            assert receipt["expandable_segments"] is best
            gather = receipt["gather_probe"]
            assert gather["group_rank"] == 0 and gather["group_size"] == 1
            assert gather["gather_calls"] == gather["native_all_gather_into_tensor_calls"] == 24
            assert gather["distinct_bf16_gather_storages"] == (1 if best else 2)
            assert gather["input_data_ptr"] == gather["output_data_ptr"] if best else gather["input_data_ptr"] != gather["output_data_ptr"]
            assert gather["master_to_bf16_casts"] == (24 if best else 1)
            assert gather["lazy_init_allocation_delta_bytes"] == (0 if best else 1050673152)
        row["peak_total_change_pct"] = 100 * row["total_change_gib"] / row["control"]["total_gib"]
    (base / "verified-summary.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
