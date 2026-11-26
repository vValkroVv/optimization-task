"""Select warmed-eager winners and preserve the length-grid fallbacks."""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent
MODELS = ("100b", "700b")
LENGTHS = (32768, 65536, 131072, 262144)
ROLES = ("projection", "dhidden")
SENSITIVITY_CELLS = {
    ("100b", 65536, "dhidden"),
    ("700b", 32768, "projection"),
    ("700b", 65536, "projection"),
}


def read(path):
    return json.loads(path.read_text())


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def geometry(manifest_key):
    version, capability, descriptor = json.loads(manifest_key)
    m, n, k = descriptor[:3]
    lhs_rows, output_width = (m, n) if descriptor[10] else (n, m)
    return version, tuple(capability), lhs_rows, k, output_width


def metrics(run, timing_arm):
    eager = run["warmed_eager"]
    cold = run["isolated_memory"]
    graph = run["graph_ms"]
    names = ["control", "control_repeat"] if timing_arm == "control" else [timing_arm]
    samples = [sample for name in names for sample in eager[name]]
    isolated = [sample for name in names for sample in cold[name]]
    result = {key: median(s[key] for s in samples)
              for key in ("wall_ms", "event_ms", "submit_ms", "wait_ms")}
    result["warmed_gift_total_gib"] = max(s["gift_total_peak_gib"] for s in samples)
    result["isolated_gift_total_gib"] = max(s["gift_total_peak_gib"] for s in isolated)
    graph_samples = [value for name in names for value in graph.get(name, [])]
    result["graph_ms"] = median(graph_samples) if graph_samples else None
    result["samples"] = len(samples)
    return result


def add_choice(manifest, key, config):
    # prepare() checks column-major first. Remove a superseded
    # descriptor for this logical product before adding its winner.
    shape = geometry(key)
    merged = {k: v for k, v in manifest.items() if geometry(k) != shape}
    merged[key] = config
    return merged


def main(args):
    original = read(args.original_choices)
    enriched = dict(original)
    recommended = dict(original)
    evidence = []
    new = {}
    sensitivity = []
    for model in MODELS:
        for seq in LENGTHS:
            for role in ROLES:
                path = HERE / "results" / f"{model}-s{seq}-{role}" / "confirm.json"
                data = read(path)
                assert (data["model"], data["sequence"], data["role"]) == (model, seq, role)
                assert (data["batch"], data["chunks"]) == (1, 8)
                output_width = 128256 if role == "projection" else {"100b": 4096, "700b": 7168}[model]
                assert data["output_shape"] == [seq // 4, output_width]
                graph_recovery = path.parent / "graph-recovery.json"
                initial_graph_errors = data["graph_errors"]
                if graph_recovery.is_file():
                    recovery = read(graph_recovery)
                    assert recovery["original_confirm_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
                    assert recovery["cuda_visible_devices"] == data["cuda_visible_devices"]
                    data["graph_ms"] = recovery["graph_ms"]
                    data["graph_errors"] = recovery["graph_errors"]
                if role == "dhidden":
                    assert data["dlogits"]["input_output_alias"]
                    assert data["dlogits"]["sha256"] == "d3343cd518a05a6d5bea58b0493fa33373e040f1ce3dd5864b14c03033528929"
                elapsed = data["warmed_eager"]
                selected = data["selected"]
                comparison = data["comparison"]
                fastest = min(comparison, key=lambda arm: comparison[arm]["paired_wall_delta_ms"], default=None)
                reference = metrics(data, "control")
                control_noise = median(
                    abs(a["wall_ms"] - b["wall_ms"])
                    for a, b in zip(elapsed["control"], elapsed["control_repeat"], strict=True)
                )
                graph_left = data["graph_ms"].get("control", [])
                graph_right = data["graph_ms"].get("control_repeat", [])
                graph_noise = (median(abs(a - b) for a, b in zip(graph_left, graph_right, strict=True))
                               if graph_left and graph_right else None)
                item = dict(
                    model=model, sequence=seq, role=role, output_shape=data["output_shape"],
                    disposition="retain_new_plan" if selected else "keep_control",
                    selected=selected, control=data["control"], control_metrics=reference,
                    control_wall_noise_ms=control_noise, control_graph_noise_ms=graph_noise,
                    graph_errors=data["graph_errors"],
                    initial_graph_errors=initial_graph_errors,
                    measured_finalists=len(comparison),
                    evidence=str(path.relative_to(HERE)),
                )
                benchmark_arm = selected or fastest
                if benchmark_arm:
                    finalist = data["finalists"][benchmark_arm]
                    assert finalist["bitwise"] and finalist["loaded_exact_descriptor"]
                    assert finalist["activation"]["candidate"] > 0 and finalist["activation"]["fallback"] == 0
                    item["benchmark_arm"] = benchmark_arm
                    item["benchmark_choice"] = finalist
                    item["benchmark_metrics"] = metrics(data, benchmark_arm)
                    item["benchmark_comparison"] = dict(
                        comparison[benchmark_arm],
                        control_graph_ms=reference["graph_ms"],
                        candidate_graph_ms=item["benchmark_metrics"]["graph_ms"],
                    )
                if graph_recovery.is_file():
                    item["graph_recovery"] = str(graph_recovery.relative_to(HERE))
                if selected:
                    assert comparison[selected]["retained"]
                    key = json.dumps([data["cublaslt"], data["capability"], finalist["descriptor"]])
                    config = finalist["config"]
                    choices = path.parent / finalist["choices"]
                    assert read(choices) == {key: config}
                    assert key not in new
                    new[key] = config
                    enriched = add_choice(enriched, key, config)
                    reproducible = True
                    if args.sensitivity and (model, seq, role) in SENSITIVITY_CELLS:
                        sensitivity_path = HERE / "sensitivity" / f"{path.parent.name}.json"
                        rerun = read(sensitivity_path)
                        assert rerun["confirm_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
                        assert rerun["choices_sha256"] == hashlib.sha256(choices.read_bytes()).hexdigest()
                        assert rerun["same_shape_bitwise"] and rerun["loaded_exact_descriptor"]
                        assert rerun["activation"]["fallback"] == 0
                        assert rerun["compared_elements"] == finalist["compared_elements"]
                        assert rerun["cuda_visible_devices"] != data["cuda_visible_devices"]
                        reproducible = rerun["comparison"]["retained"]
                        sensitivity.append(dict(
                            model=model, sequence=seq, role=role, choice=selected,
                            disposition="recommend" if reproducible else "provisional",
                            initial=item["benchmark_comparison"],
                            repeated=rerun["comparison"],
                            evidence=str(sensitivity_path.relative_to(HERE)),
                        ))
                    if reproducible:
                        recommended = add_choice(recommended, key, config)
                evidence.append(item)
    assert len(evidence) == 16
    save(HERE / "selected.json", evidence)
    save(HERE / "linear-plans.new.json", new)
    save(HERE / "linear-plans.selected.json", enriched)
    print(f"{len(new)} new plans across {len(evidence)} cells; {len(enriched)} merged manifest entries")
    if args.sensitivity:
        assert len(sensitivity) == len(SENSITIVITY_CELLS)
        save(HERE / "sensitivity" / "summary.json", sensitivity)
        save(HERE / "linear-plans.recommended.json", recommended)
        print(f"{len(recommended)} recommended entries after {len(sensitivity)} checks")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-choices", type=Path, default=HERE.parents[2] / "tuning/linear-plans.json")
    parser.add_argument("--sensitivity", action="store_true")
    main(parser.parse_args())
