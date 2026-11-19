"""Compare completed four-fix native receipts and the matching training scalars."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def read_arm(receipt_file, run_root):
    receipt = json.loads(receipt_file.read_text())
    assert receipt["compiler_stable"] is True, receipt["id"]
    raw = run_root / receipt["id"]
    events = list((raw / "native/tb").glob("events.out.tfevents.*"))
    if len(events) != 1:
        raise ValueError(
            f"Expected one rank-zero training event file in {raw}: {events}"
        )
    board = EventAccumulator(str(events[0]), size_guidance={"scalars": 0}).Reload()
    series = {}
    for tag in (
        "loss/train/total",
        "l2_norm/grad/global",
        "loss/train/loss_ce",
        "loss/train/loss_mtp",
    ):
        points = board.Scalars(tag)
        assert [p.step for p in points] == list(range(1, 21)), (
            tag,
            [p.step for p in points],
        )
        series[tag] = [p.value for p in points]
        assert all(math.isfinite(v) for v in series[tag]), tag
    return {
        "receipt": receipt,
        "event_file": str(events[0].relative_to(raw)),
        "event_sha256": hashlib.sha256(events[0].read_bytes()).hexdigest(),
        "series": series,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("arms", nargs="+")
    args = parser.parse_args()
    results = {
        arm: read_arm(args.campaign / f"{arm}.json", args.run_root)
        for arm in ("baseline", *args.arms)
    }
    control = results["baseline"]
    for result in results.values():
        rec, base = result["receipt"], control["receipt"]
        for field in ("source", "gift_sha256", "normalized_config_sha256", "fixes"):
            assert rec[field] == base[field], field
        result["tps_change_pct"] = 100 * (
            rec["physical_tps"] / base["physical_tps"] - 1
        )
        result["max_absolute_difference"] = {
            name: max(
                abs(a - b) for a, b in zip(values, control["series"][name], strict=True)
            )
            for name, values in result["series"].items()
        }
    print(json.dumps(results, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
