"""Summarize thermal samples in the synchronized measured calls of each rank."""
import csv
import json
import sys
from pathlib import Path
from statistics import median

base = Path(__file__).resolve().parent
folder = Path(sys.argv[1])
out = {"cards": []}
series = [json.loads((folder / f"timing-{i}.json").read_text()) for i in range(8)]
wall = [max((r["stop_utc_ns"] - r["start_utc_ns"]) / 1e6 for r in step) for step in zip(*series, strict=True)]
assert len(wall) == 32
out["timing"] = dict(samples=len(wall), early_ms=median(wall[:5]), late_ms=median(wall[-6:]), median_ms=median(wall), worst_ms=max(wall))
reader = csv.DictReader(next(base.glob("gpu1-*.csv")).open())
cards = [[] for _ in range(8)]
for row in reader:
    cards[int(row["gpu"])].append({k: int(v) if k != "watts" else float(v) for k,v in row.items()})
for i, data in enumerate(cards):
    spans = [(r["start_utc_ns"], r["stop_utc_ns"]) for r in series[i]]
    inside = [r for r in data if any(lo <= r["utc_ns"] <= hi for lo,hi in spans)]
    assert data[0]["utc_ns"] < spans[0][0] and data[-1]["utc_ns"] > spans[-1][1]
    lo = max((r for r in data if r["utc_ns"] <= spans[0][0]),key=lambda r:r["utc_ns"])
    hi = min((r for r in data if r["utc_ns"] >= spans[-1][1]),key=lambda r:r["utc_ns"])
    out["cards"].append(dict(gpu=i, samples=len(inside), core_max=max(r["temp_c"] for r in inside),
        hbm_max=max(r["memory_c"] for r in inside), core_whole_max=max(r["temp_c"] for r in data),
        median_sm=median(r["sm_mhz"] for r in inside), min_sm=min(r["sm_mhz"] for r in inside),
        thermal_hits=sum(bool(r["sw_thermal"] or r["hw_thermal"]) for r in inside),
        hw_thermal=sum(r["hw_thermal"] for r in inside), hw_brake=sum(r["power_brake"] for r in inside),
        power_cap=sum(r["sw_power_cap"] for r in inside),
        preexisting_thermal_s=data[0]["thermal_counter_ns"]/1e9,
        thermal_delta_s=(hi["thermal_counter_ns"]-lo["thermal_counter_ns"])/1e9,
        whole_thermal_delta_s=(data[-1]["thermal_counter_ns"]-data[0]["thermal_counter_ns"])/1e9))
print(json.dumps(out,indent=2))
