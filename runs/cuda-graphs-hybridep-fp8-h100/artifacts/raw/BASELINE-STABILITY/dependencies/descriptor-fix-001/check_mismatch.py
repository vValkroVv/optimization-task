"""Map saved coordinates using inspected scheduler formulas; no GPU execution."""
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
events_path = BASE / "reproducer/003/events.jsonl"
events = [json.loads(line) for line in events_path.read_text().splitlines()]
detail = next(event for event in events if event["event"] == "mismatch_detail")
samples = detail["summaries"][0]["first_16"]
# get_num_1d_blocks_per_group and get_swizzled_block_idx, multicast-on-B.
usage = {candidate: candidate * 128 + ((116 + candidate - 1) // candidate) * 152 for candidate in (8, 16)}
width = min(usage, key=usage.get)
mapping = {}
for local in range(84):
    group, within = divmod(local, 7 * width)
    first = group * width
    effective = min(width, 12 - first)
    m, n = first + within % effective, within // effective
    assert (m, n) not in mapping
    mapping[m, n] = local
assert len(mapping) == 84
mapped = []
for sample in samples:
    local = mapping[sample["m"] // 128, sample["n"] // 152]
    global_tile = sample["g"] * 84 + local
    mapped.append({"local_tile": local, "global_tile": global_tile,
                   "cta": global_tile % 116, "iteration": global_tile // 116})
assert all(value == mapped[0] for value in mapped)
assert mapped[0] == {"local_tile": 75, "global_tile": 159, "cta": 43, "iteration": 1}


def dot(ga, gb, m, n):
    return sum(
        int((k + 3 * ga) % (5 + ga) == m % (5 + ga)) * (1 - 2 * ((m + ga) % 2))
        * int((2 * k + gb) % (7 + gb) == n % (7 + gb)) * (1 - 2 * ((n // 3 + gb) % 2))
        for k in range(128)
    )


matches = [[ga, gb] for ga in range(4) for gb in range(4)
           if all(dot(ga, gb, s["m"], s["n"]) == s["actual"] for s in samples)]
assert matches == [[3, 1]], matches
assert all(dot(1, 1, s["m"], s["n"]) == s["expected"] for s in samples)
schedule = [{"iteration": iteration, "group": tile // 84, "local_tile": tile % 84}
            for iteration, tile in enumerate(range(43, 336, 116))]
assert [entry["group"] for entry in schedule] == [0, 1, 3]
print(json.dumps({
    "result": "PASS", "events_sha256": hashlib.sha256(events_path.read_bytes()).hexdigest(),
    "scheduler_sha256": hashlib.sha256((HERE / "deep_gemm/include/deep_gemm/common/scheduler.cuh").read_bytes()).hexdigest(),
    "usage": usage, "selected_group_width": width, "mapping": mapped[0],
    "cta43_schedule": schedule, "saved_samples": len(samples),
    "matching_operand_group_pairs": matches,
    "interpretation": "All saved samples fit old group1 A-map overwritten with group3; this is not a complete tensor capture or native-fault proof.",
}, indent=2))
