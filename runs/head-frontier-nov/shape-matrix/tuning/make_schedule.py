"""Select exact regional finalists and write an alternating whole-head replay schedule."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
# These are experimental choices derived from this campaign. The execution policy sees only
# calibrated GEMM descriptors and probes the installed native route for everything else.
finalists = {"100b-c2": "projection", "100b-c8": "projection", "700b-c8": "dhidden"}
manifest = {}
for case, operation in finalists.items():
    case_path = HERE / "proj" / (case + "-confirm")
    full = json.loads(case_path.with_suffix(".json").read_text())
    selected = full["choices"][operation]
    assert selected["bitwise"]
    for key, config in json.loads(case_path.with_suffix(".choices.json").read_text()).items():
        if json.loads(key)[2] == selected["problem"] and config == selected["config"]:
            manifest[key] = config
            break
    else:
        raise ValueError(f"Missing measured configuration for {case}")
(HERE / "linear-plans.json").write_text(json.dumps(manifest, indent=2))
trials = []
for run in ("candidate-a", "control-b", "candidate-b"):
    candidate = run.startswith("candidate")
    for model, chunks in (("700b", 8), ("100b", 2), ("100b", 8)):
        spec = dict(model=model, chunks=chunks, run=run, operations=["ce", "dw", "linear"] if candidate else [])
        if candidate:
            spec.update(choices="tuning/linear-plans.json", plan="tuning/lt/results/calibrated-catalog.json",
                        ce_path="tuning/ce/centered-tuned.so",
                        ce_sha="d3343cd518a05a6d5bea58b0493fa33373e040f1ce3dd5864b14c03033528929")
        trials.append(spec)
(HERE / "finalists.json").write_text(json.dumps(trials, indent=2))
