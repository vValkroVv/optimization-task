"""Merge shape-matched linear recommendations and map all 21 B1 cells to receipts."""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent
GRID = HERE.parents[1]
MATRIX = GRID.parent
PRIOR = GRID / "region-tuning/linear"
REPEAT = {("700b", 2048), ("700b", 4096)}
PRESETS = (("100b", 2), ("100b", 8), ("700b", 8))
LENGTHS = (4096, 8192, 16384, 32768, 65536, 131072, 262144)
ROLES = ("projection", "dhidden")
WIDTH = {"100b": 4096, "700b": 7168}
VOCAB = 128256
CE_SHA256 = "d3343cd518a05a6d5bea58b0493fa33373e040f1ce3dd5864b14c03033528929"


def read(path):
    return json.loads(path.read_text())


def save(path, data):
    partial = path.with_suffix(".partial")
    partial.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    partial.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference(path):
    return str(path.relative_to(MATRIX))


def geometry(key):
    version, capability, descriptor = json.loads(key)
    m, n, k = descriptor[:3]
    rows, output = (m, n) if descriptor[10] else (n, m)
    return version, tuple(capability), rows, k, output


def expected(model, rows, role):
    k, output = (WIDTH[model], VOCAB) if role == "projection" else (VOCAB, WIDTH[model])
    return (130101, (9, 0), rows, k, output)


def check_linear(data, model, rows, role):
    assert data["cublaslt"] == 130101 and data["capability"] == [9, 0]
    assert data["model"] == model and data["batch"] == 1 and data["role"] == role
    assert rows == 2 * data["sequence"] // data["chunks"]
    _, _, _, k, output = expected(model, rows, role)
    assert data["output_shape"] == [rows, output]
    assert data["operands"] == [
        {"shape": [rows, k], "stride": [k, 1]},
        {"shape": [k, output], "stride": [1, k] if role == "projection" else [output, 1]},
    ]
    if role == "dhidden":
        assert data["dlogits"]["input_output_alias"] and data["dlogits"]["sha256"] == CE_SHA256


def summarize_metrics(data, arm):
    names = ("control", "control_repeat") if arm == "control" else (arm,)
    eager = [sample for name in names for sample in data["warmed_eager"][name]]
    isolated = [sample for name in names for sample in data["isolated_memory"][name]]
    device = [sample for name in names for sample in data["graph_ms"].get(name, [])]
    return dict(**{key: median(s[key] for s in eager) for key in
                   ("wall_ms", "event_ms", "submit_ms", "wait_ms")},
                warmed_gift_gib=max(s["gift_total_peak_gib"] for s in eager),
                isolated_gift_gib=max(s["gift_total_peak_gib"] for s in isolated),
                graph_ms=median(device) if device else None)


def merge_new(catalog, linear, model, chunks, seq, role):
    rows = 2 * seq // chunks
    case = (model, rows, role)
    directory = HERE / "results" / f"{model}-c{chunks}-s{seq}-{role}"
    path = directory / "confirm.json"
    key = expected(model, rows, role)
    if not path.is_file():
        return
    data = read(path)
    check_linear(data, *case)
    selected = data["selected"]
    initial_selected = selected
    repeat = directory / "recheck.json"
    effective = data
    if role == "projection" and (model, rows) in REPEAT:
        if not repeat.is_file():
            selected = None
        else:
            result = read(repeat)
            assert result["initial_confirm_sha256"] == digest(path) and result["bitwise"]
            if result["comparison"]["retained"]:
                effective = result
            else:
                selected = None

    if selected:
        finalist = data["finalists"][selected]
        assert finalist["bitwise"] and finalist["loaded_exact_descriptor"]
        assert finalist["activation"]["candidate"] > 0 and finalist["activation"]["fallback"] == 0
        assert finalist["compared_elements"] == rows * key[-1]
        choices_path = directory / finalist["choices"]
        choice, config = next(iter(read(choices_path).items()))
        assert geometry(choice) == key and config == finalist["config"]
        for old in tuple(catalog):
            if geometry(old) == key:
                del catalog[old]
        catalog[choice] = config

    best = selected or min(data["comparison"], key=lambda x: data["comparison"][x]["paired_wall_delta_ms"],
                           default=None)
    effective_arm = "candidate" if effective is not data else best
    comparison = (effective["comparison"] if effective is not data
                  else data["comparison"].get(best))
    linear[case] = dict(
        evidence=reference(path), recheck=reference(repeat) if repeat.is_file() else None,
        measured_shape=data["output_shape"], parity_reference_device=data["parity_reference_device"],
        initial_selection=initial_selected, selected=selected, best_tested=best,
        control_metrics=summarize_metrics(effective, "control"),
        candidate_metrics=summarize_metrics(effective, effective_arm) if best else None,
        paired_comparison=comparison, graph_errors=effective["graph_errors"],
        checked_elements=data["finalists"][best]["compared_elements"] if best else 0,
        tested_layouts=[item["problem"] for item in data["screen"]],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    prior_path = PRIOR / "linear-plans.recommended.json"
    catalog = read(prior_path)
    original_count = len(catalog)
    linear, ce = {}, {}
    for model in WIDTH:
        for seq in LENGTHS[3:]:
            for role in ROLES:
                path = PRIOR / "results" / f"{model}-s{seq}-{role}" / "confirm.json"
                data = read(path)
                check_linear(data, model, seq // 4, role)
                followup = PRIOR / "sensitivity" / f"{model}-s{seq}-{role}.json"
                linear[(model, seq // 4, role)] = {
                    "evidence": reference(path),
                    "recheck": reference(followup) if followup.is_file() else None,
                }
            path = GRID / "region-tuning/ce" / f"{model}-{seq}.json"
            data = read(path)
            arm = data["results"]["tuned"]
            assert data["fixture"]["folded_rows"] == seq // 4 and arm["sha256"] == CE_SHA256
            assert arm["dz_differences"] == arm["raw_row_loss_differences"] == arm["final_loss_differences"] == 0
            ce[(model, seq // 4)] = reference(path)

    early = MATRIX / "tuning/proj/recheck/100b-c8-confirm.json"
    early_data = read(early)
    early_screen = early.parent.parent / early_data["screen"]
    recorded_environment = read(early_screen)
    assert digest(early_screen) == early_data["screen_sha256"]
    for role in ROLES:
        choice = early_data["choices"][role]
        manifest_key = json.dumps([recorded_environment["cublaslt"], recorded_environment["capability"],
                                   choice["problem"]])
        assert choice["bitwise"] and geometry(manifest_key) == expected("100b", 2048, role)
        linear[("100b", 2048, role)] = {"evidence": reference(early)}

    for model, chunks in PRESETS:
        for seq in LENGTHS:
            rows = 2 * seq // chunks
            for role in ROLES:
                if (model, rows, role) not in linear:
                    merge_new(catalog, linear, model, chunks, seq, role)
            if (model, rows) not in ce:
                path = HERE / "ce-results" / f"{model}-c{chunks}-s{seq}" / "confirm.json"
                if path.is_file():
                    data = read(path)
                    assert 2 * data["fixture"]["logits_shape"][1] == rows
                    assert data["results"]["tuned"]["sha256"] == CE_SHA256
                    assert all(a["full_dz_bitwise"] and a["scales_checked"]
                               and a["checked_dz_elements"] == rows * VOCAB
                               and a["raw_row_loss_differences"] == a["final_loss_differences"] == 0
                               for a in data["results"].values())
                    ce[(model, rows)] = reference(path)

    choices = {geometry(k): (k, v) for k, v in catalog.items()}
    assert len(choices) == len(catalog)
    covered, uncovered = [], []
    for model, chunks in PRESETS:
        for seq in LENGTHS:
            rows = 2 * seq // chunks
            cell = dict(model=model, chunks=chunks, sequence=seq, folded_rows=rows)
            for role in ROLES:
                details = linear.get((model, rows, role))
                key, config = choices.get(expected(model, rows, role), (None, None))
                cell[role] = dict(
                    evidence=None if details is None else details["evidence"],
                    followup=None if details is None else details.get("recheck"),
                    route="calibrated cuBLASLt" if key else "ATen",
                    catalog_key=key, config=config,
                )
                if not details:
                    uncovered.append((model, chunks, seq, role))
            cell["ce"] = dict(route="tuned CUDA consuming CE", sha256=CE_SHA256,
                              evidence=ce.get((model, rows)))
            if not cell["ce"]["evidence"]:
                uncovered.append((model, chunks, seq, "ce"))
            covered.append(cell)
    if args.require_complete and uncovered:
        raise SystemExit(f"Missing component receipts: {uncovered}")
    save(HERE / "linear-plans.final-grid.json", catalog)
    save(HERE / "coverage.json", dict(
        layout="BF16, row-contiguous, allocating exact component descriptor, H100 SM90, cuBLASLt 13.1.1",
        inherited_catalog=reference(prior_path), inherited_catalog_entries=original_count,
        inherited_catalog_sha256=digest(prior_path),
        final_catalog_entries=len(catalog), final_catalog_sha256=digest(HERE / "linear-plans.final-grid.json"),
        cells=covered,
        new_linear=[dict(model=model, folded_rows=rows, role=role, **stats)
                    for (model, rows, role), stats in sorted(linear.items()) if "control_metrics" in stats],
        uncovered=uncovered,
    ))


if __name__ == "__main__":
    main()
