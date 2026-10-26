from __future__ import annotations

import glob
import itertools
import statistics
import sys

WARMUP_STEPS = 100
RANK_FILES_SHOWN = 2
MILLISECONDS_PER_SECOND = 1000
PERCENT = 100


def block_medians(path: str, block: int) -> dict[str, list[float]]:
    batch_starts = []
    with open(path) as file:
        for line in file:
            step, arm, started = line.split("\t")
            batch_starts.append((int(step), arm, float(started)))
    blocks: dict[int, tuple[str, list[float]]] = {}
    for (step, arm, started), (_, _, next_started) in itertools.pairwise(batch_starts):
        if step < WARMUP_STEPS:
            continue
        blocks.setdefault(step // block, (arm, []))[1].append(next_started - started)
    medians: dict[str, list[float]] = {}
    for arm, durations in blocks.values():
        medians.setdefault(arm, []).append(statistics.median(durations))
    return medians


def main(run_dir: str, block: int) -> None:
    for path in sorted(glob.glob(f"{run_dir}/ab_batch_starts/rank*.tsv"))[:RANK_FILES_SHOWN]:
        medians = block_medians(path, block)
        baseline = statistics.mean(medians["off"])
        print(path.rsplit("/", 1)[-1],
              f"off {baseline * MILLISECONDS_PER_SECOND:.1f} ms ({len(medians['off'])} blocks)")
        for arm, arm_medians in medians.items():
            if arm == "off":
                continue
            mean = statistics.mean(arm_medians)
            standard_error = (statistics.pstdev(arm_medians) ** 2 / len(arm_medians)
                              + statistics.pstdev(medians["off"]) ** 2 / len(medians["off"])) ** 0.5
            print(f"  {arm:7s} {mean * MILLISECONDS_PER_SECOND:.1f} ms  "
                  f"diff {(mean - baseline) * MILLISECONDS_PER_SECOND:+.1f} ms "
                  f"({(mean / baseline - 1) * PERCENT:+.2f}%)  "
                  f"stderr {standard_error * MILLISECONDS_PER_SECOND:.1f} ms  blocks {len(arm_medians)}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
