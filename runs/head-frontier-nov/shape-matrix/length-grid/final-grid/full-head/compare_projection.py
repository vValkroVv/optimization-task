"""Compare prior and final optimized 100B C8 / 128k heads in both orders."""

import fcntl
import os
from pathlib import Path
from statistics import median
import subprocess

from manifests import dw_plans, expected_routes, pin, read_pinned
from run_grid import HERE, legacy, provenance, run_arm
from summarise_grid import benchmark, read


ROOT = HERE / "results/projection-128k"
CASE = "100b-c8"
MODEL = "100b"
SEQ = 131072
CHUNKS = 8
PRIOR_DW = "2be234123003fad9d2583ebe3b28eb2dfb7f5351d30d0835191dbcde580ba767"
PRIOR_LINEAR = "7b2aed540a001cbd8f02f08e626c6ea1ab6905d297469a9cf4d6c4f9b247a1ca"
INSTRUMENTATION = HERE / "projection-instrumentation"


def instrument_settings(model, variant, gpus):
    settings = original_settings(model, variant, gpus)
    settings["PYTHONPATH"] = os.pathsep.join((settings["PYTHONPATH"], str(INSTRUMENTATION)))
    return settings


original_settings = legacy.settings_for


def planned_arms(matrix):
    final = next(row for row in read(matrix / "schedule.json")
                 if (row["case"], row["seq"], row["variant"]) == (CASE, SEQ, "cuda13"))
    if (final["warmup"], final["samples"]) != (10, 10):
        raise RuntimeError("Expected the matrix's 10+10 schedule")
    prior = {"linear": pin(legacy.CATALOGS[0], PRIOR_LINEAR), "dw": pin(legacy.CATALOGS[1], PRIOR_DW),
             "ce": final["manifests"]["ce"]}
    arms = {"prior": prior, "final": final["manifests"]}
    routes = {}
    for arm, manifests in arms.items():
        dw = dw_plans(read_pinned(manifests["dw"]))
        linear = read_pinned(manifests["linear"])
        expected = expected_routes(MODEL, CHUNKS, SEQ, dw, linear)
        projection = bool(expected["linear"]["projection"])
        if expected["dw"] or expected["linear"]["dhidden"] or projection != (arm == "final"):
            raise RuntimeError(f"Unexpected effective {arm} paths: {expected}")
        routes[arm] = {"dw_case": None, "linear": expected["linear"]}
    return arms, routes, final


def summarize_comparison(requests):
    result = {"case": CASE, "sequence": SEQ, "pairs": []}
    for index in (1, 2):
        pair = {"repeat": index, "order": []}
        for request in requests:
            if request["pair"] != index:
                continue
            name = request["arm"]
            pair["order"].append(name)
            arm = ROOT / name / f"{CASE}-b1-s{SEQ}" / f"pair-{index:02d}/cuda13"
            attempts = sorted(arm.glob("attempt-*/status.json"))
            statuses = [{**read(path), "path": str(path.parent.relative_to(ROOT))} for path in attempts]
            current = statuses[-1] if statuses else {"status": "pending"}
            pair[name] = {"status": current["status"], "attempts": statuses,
                          "manifests": request["manifests"]}
            if current["status"] == "success":
                path = ROOT / current["path"]
                invocation = read(path / "invocation.json")
                if any(invocation[key] != val for key, val in request.items()):
                    raise RuntimeError(f"Attempt used different settings: {path}")
                pair[name].update(benchmark(path, request))
                timings = [read(path / f"timing-{rank}.json") for rank in range(legacy.WORLD)]
                if any(len(rank) != request["samples"] for rank in timings):
                    raise RuntimeError(f"Unexpected number of timestamps: {path}")
                steps = zip(*(read(path / f"rank-{rank}.json")["measured"] for rank in range(legacy.WORLD)),
                            strict=True)
                pair[name]["rank_timings"] = timings
                pair[name]["median_rank_steps_ms"] = [median(rank["ms"] for rank in step) for step in steps]
        if all(pair[name]["status"] == "success" for name in ("prior", "final")):
            pair["throughput_gain_pct"] = 100 * (pair["prior"]["ms"] / pair["final"]["ms"] - 1)
            pair["memory_change_gib"] = pair["final"]["total_gib"] - pair["prior"]["total_gib"]
        result["pairs"].append(pair)
    gains = [pair["throughput_gain_pct"] for pair in result["pairs"] if "throughput_gain_pct" in pair]
    result["status"] = "complete" if len(gains) == 2 else "incomplete"
    if len(gains) == 2:
        result["throughput_gain_pct"] = median(gains)
    return result


def main():
    matrix = HERE / "results/final"
    status = HERE / "control/final-02.exit"
    if not status.is_file() or status.read_text().strip() != "0":
        raise RuntimeError("The full matrix must finish successfully before this comparison")
    arms, routes, final = planned_arms(matrix)
    with (HERE / ".launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        files = {f"{arm}_{kind}": pin(p["path"], p["sha256"])
                 for arm, manifests in arms.items() for kind, p in manifests.items()}
        sources = provenance({MODEL}, files)
        helpers = (Path(__file__), HERE / "run_projection_detached.sh", INSTRUMENTATION / "sitecustomize.py")
        sources["files_sha256"].update({str(path.resolve()): legacy.sha256(path) for path in helpers})
        source = legacy.CANDIDATE_SOURCE / "torchtitan/components/loss.py"
        if final["loss_source_sha256"] != sources["files_sha256"][str(source)]:
            raise RuntimeError("The tuned loss source has changed since the completed matrix")
        requests = []
        for repeat, order in ((1, ("prior", "final")), (2, ("final", "prior"))):
            for position, arm in enumerate(order, start=1):
                requests.append({**final, "arm": arm, "pair": repeat, "position": position,
                                 "manifests": arms[arm], "expected_routes": routes[arm]})
        ROOT.mkdir(parents=True, exist_ok=True)
        schedule = ROOT / "schedule.json"
        if schedule.is_file() and read(schedule) != requests:
            raise RuntimeError("The saved comparison schedule has changed")
        legacy.write_json(schedule, requests)
        legacy.settings_for = instrument_settings
        try:
            measure(requests, sources)
        finally:
            result = summarize_comparison(requests)
            legacy.write_json(HERE / "projection-128k-summary.json", result)
        if result["status"] != "complete":
            raise RuntimeError("The projection comparison did not complete all four arms")


def measure(requests, sources):
    telemetry = ROOT / f"telemetry-{legacy.now().replace(':', '-')}.csv"
    query = "--query-gpu=timestamp,index,clocks.sm,power.draw,temperature.gpu,utilization.gpu"
    env = dict(os.environ, TZ="UTC")
    with telemetry.open("x") as log:
        cmd = ["nvidia-smi", query, "--format=csv,nounits", "--loop=3"]
        with subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env) as sampler:
            try:
                for req in requests:
                    run_arm(ROOT / req["arm"], req, sources, timeout=1200)
            finally:
                if sampler.poll() is None:
                    sampler.terminate()


if __name__ == "__main__":
    main()
