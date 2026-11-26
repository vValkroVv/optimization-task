"""Check clock-limit observations in the per-rank timed calls on GPUs 0–5."""
import csv
import json
from pathlib import Path
from statistics import median

root = Path(__file__).resolve().parent
cards = [[] for _ in range(8)]
with next(root.glob("six-*.csv")).open() as stream:
    for row in csv.DictReader(stream):
        cards[int(row["gpu"])].append({k:float(v) if k=="watts" else int(v) for k,v in row.items()})
output = []
for timing in sorted((root / "replay").rglob("timing-0.json")):
    folder = timing.parent
    series = [json.loads((folder / f"timing-{i}.json").read_text()) for i in range(6)]
    assert len(series[0]) == 32 and not (folder/"timing-6.json").exists()
    wall = [max((r["stop_utc_ns"]-r["start_utc_ns"])/1e6 for r in step) for step in zip(*series, strict=True)]
    aggregate = dict(path=str(folder.relative_to(root)), median_ms=median(wall), first_five_ms=median(wall[:5]), last_six_ms=median(wall[-6:]), cards=[])
    for i,data in enumerate(cards):
        rank_series = series[i] if i < 6 else series[0]
        spans = [(r["start_utc_ns"],r["stop_utc_ns"]) for r in rank_series]
        assert data[0]["utc_ns"] <= spans[0][0] <= spans[-1][1] <= data[-1]["utc_ns"]
        inside = [r for r in data if any(lo <= r["utc_ns"] <= hi for lo,hi in spans)]
        assert inside
        lo = next(r for r in reversed(data) if r["utc_ns"] <= spans[0][0])
        hi = next(r for r in data if r["utc_ns"] >= spans[-1][1])
        aggregate["cards"].append(dict(gpu=i,samples=len(inside),max_core_c=max(r["temp_c"] for r in inside),
            max_hbm_c=max(r["memory_c"] for r in inside),median_sm_mhz=median(r["sm_mhz"] for r in inside),
            max_util=max(r["util_pct"] for r in inside),
            thermal_hits=sum(bool(r["hw_thermal"] or r["sw_thermal"]) for r in inside),
            thermal_delta_s=(hi["thermal_counter_ns"]-lo["thermal_counter_ns"])/1e9,
            brake_hits=sum(r["power_brake"] for r in inside)))
    output.append(aggregate)
assert len(output)==4
print(json.dumps(output,indent=2))
