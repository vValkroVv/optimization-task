"""Assemble separate ReplayWriter catalogs and the evidence for all 21 cells."""
import argparse
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
GRID = HERE.parents[1]
ROOT = GRID.parent
PRIOR = GRID / "region-tuning/dw/results"
SEQUENCES = (4096, 8192, 16384, 32768, 65536, 131072, 262144)


def read(path):
    return json.loads(path.read_text())


def write(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def relative(path):
    return str(path.relative_to(ROOT))


def summary(measurement):
    keys = ("median_graph_ms", "median_submit_ms", "median_final_wait_ms", "median_wall_ms")
    return {key: measurement[key] for key in keys}


def clean_plan(plan, shape, result, chunks):
    name = result["selected"]
    assert plan["shape"] == list(shape)
    assert plan["alignments"] == result["alignments"]
    assert result["activation"][name]["tuned_calls_beta0_beta1"] == [1, chunks - 1]
    assert result["chain_bit_mismatches"][name] == 0
    assert plan["numerics"]["per_write_bit_mismatches"] == [0] * chunks
    memory = result["isolated_memory"]
    assert memory[name]["allocator_retries"] == memory["control"]["allocator_retries"] == 0
    assert memory[name]["gift_total_gib"] <= memory["control"]["gift_total_gib"]
    return plan


def verified_win(result, sections, name):
    for section in sections:
        paired = result[section]
        wall = (paired[name] if name in paired else paired)["wall_ms"]
        assert wall["beats_both_fraction"] >= 0.75
        assert max(wall["candidate_minus_control_median_ms"],
                   wall["candidate_minus_repeat_median_ms"]) < -wall["median_absolute_control_delta_ms"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=int, choices=(2, 8), help="Assemble one completed catalog")
    args = parser.parse_args()
    base_path = ROOT / "tuning/lt/results/calibrated-catalog.json"
    base = read(base_path)
    assert base["format"] == 2
    inherited = {tuple(entry["shape"]): entry for entry in base["plans"]}
    catalogs = {chunks: inherited.copy() for chunks in ((args.chunks,) if args.chunks else (2, 8))}
    cells = []
    for model, chunks in (("100b", 2), ("100b", 8), ("700b", 8)):
        if chunks not in catalogs:
            continue
        width = 4096 if model == "100b" else 7168
        for seq in SEQUENCES:
            case = f"{model}-c{chunks}-b1-s{seq}"
            shape = (2 * seq // chunks, 128256, width)
            reused = chunks == 8 and seq >= 32768
            directory = PRIOR if reused else HERE / "results"
            result_path = directory / f"{case}.json"
            plan_path = directory / f"{case}-plan.json"
            result = read(result_path)
            assert tuple(result["shape"]) == shape
            assert result["case"] == case
            assert result["torch"] == "2.15.0.dev20260822+cu130"
            assert result["cublasLt_version"] == 130101
            if not reused:
                assert result["status"] == "complete"
            cell = {
                "case": case, "model": model, "chunks": chunks, "sequence": seq, "shape": list(shape),
                "receipt": relative(result_path), "receipt_sha256": digest(result_path),
                "prior_region_tuning": reused, "control": result["control"],
                "control_isolated_memory": result["isolated_memory"]["control"],
                "finalist": result.get("finalist") if reused else next(iter(result["confirmation"]), None),
            }
            selected = result["selected"]
            if selected is None:
                cell.update(
                    route="baseline-calibrated" if shape in inherited else "native-generic",
                    selection_reason="No candidate passed both paired windows, bitwise parity and memory acceptance",
                    discovery_control=summary(result["discovery"]["control"]),
                    discovery_control_repeat=summary(result["discovery"]["control_repeat"]),
                )
            else:
                plan = clean_plan(read(plan_path), shape, result, chunks)
                verified_win(result, ("discovery_deltas", "confirmation_deltas"), selected)
                catalogs[chunks][shape] = plan
                confirmation = result["confirmation"] if reused else result["confirmation"][selected]
                deltas = result["confirmation_deltas"] if reused else result["confirmation_deltas"][selected]
                cell.update(
                    route="selected-calibrated", selected=selected, selected_plan=relative(plan_path),
                    selected_plan_sha256=digest(plan_path), confirmation_deltas=deltas,
                    control_timing=summary(confirmation["control"]),
                    control_repeat_timing=summary(confirmation["control_repeat"]),
                    selected_timing=summary(confirmation[selected]),
                    selected_isolated_memory=result["isolated_memory"][selected],
                    fp32_bit_mismatches_by_write=plan["numerics"]["per_write_bit_mismatches"],
                )
            cells.append(cell)
    outputs = {}
    for chunks, by_shape in catalogs.items():
        path = HERE / f"dw-plans.selected.c{chunks}.json"
        write(path, {"format": 2, "plans": list(by_shape.values())})
        outputs[f"C{chunks}"] = {"path": relative(path), "sha256": digest(path), "entries": len(by_shape)}
    if args.chunks is None:
        write(HERE / "manifest.json", {
            "baseline_catalog": {"path": relative(base_path), "sha256": digest(base_path)},
            "catalogs": outputs, "cells": cells,
        })


if __name__ == "__main__":
    main()
