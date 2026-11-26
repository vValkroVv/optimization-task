"""Align within-call clock-limit observations to rank-max host wall time."""
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

def stamp_ns(text):
    stamp = datetime.strptime(text, "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=timezone.utc)
    return round(stamp.timestamp() * 1e9)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    rows = []
    with args.csv_path.open() as stream:
        for row in csv.reader(stream, skipinitialspace=True):
            if len(row) == 10 and row[1] == "6":
                rows.append((stamp_ns(row[0]), int(row[2]), row[6] == "Active", row[8] == "Active"))
    out = {}
    for receipt in sorted(args.root.rglob("timing-0.json")):
        inv = json.loads(receipt.with_name("invocation.json").read_text())
        devices = {g["uuid"]: g["index"] for g in inv["gpu_preflight"]}
        order = [devices[u] for u in inv["settings"]["CUDA_VISIBLE_DEVICES"].split(",")]
        r = order.index(6)
        series = [json.loads(receipt.with_name(f"timing-{rank}.json").read_text()) for rank in range(8)]
        stats = []
        for number, ranks in enumerate(zip(*series, strict=True), start=1):
            start, end = (ranks[r][key] for key in ("start_utc_ns", "stop_utc_ns"))
            inside = [(sm, h, s) for at,sm,h,s in rows if start <= at <= end]
            if not inside:
                continue
            wall_ms = max(s["stop_utc_ns"]-s["start_utc_ns"] for s in ranks) / 1e6
            stats.append(dict(sample=number, profiled=ranks[r]["profiled"], rankmax_ms=wall_ms,
                              observations=len(inside), sm_mhz=median(v[0] for v in inside),
                              thermal_fraction=sum(h or s for _,h,s in inside) / len(inside),
                              hardware_thermal_samples=sum(h for _,h,_ in inside)))
        if stats:
            out[str(receipt.parent.relative_to(args.root))] = stats
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
