"""Compare Baseline A and the final CUDA-13 CE on matched EP8 runs."""

import argparse
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import statistics
import sys

import yaml


BASELINE_CHECKOUT = Path("/workspace-SR008.nfs2/vkropotin/a10-cute-001")
FINAL_CHECKOUT = Path("/workspace-SR008.nfs2/vkropotin/lm10-c1")
BASELINE_SUMMARIZER = Path("/workspace-SR008.nfs2/vkropotin/lm10-01/summarize.py")
RAW_RELATIVE = Path("runs/production-data-abcd/artifacts/raw")
BASELINE_RAW = BASELINE_CHECKOUT / RAW_RELATIVE
FINAL_RAW = FINAL_CHECKOUT / RAW_RELATIVE
CHUNKS = (2, 8, 32)
RANKS = tuple(range(8))
STEPS = tuple(range(11, 21))
BOOTSTRAP_PREFIX = "flash_qla_bootstrap="
ANSI = re.compile(r"\x1b\[[0-9;]*m")
BASELINE_FORBIDDEN_MARKERS = (
    "lm_final_exact_ce_install=",
    "lm_final_exact_ce_complete=",
    "Stable Folded CE active:",
    "LM_HEAD_MATRIX_DW_WRITER",
    "LM_HEAD_MATRIX_DW_REDUCED",
    "LM_HEAD_MATRIX_FUSED_Z_LOSS_ACTIVE",
    "Chunked loss direct BF16 dHidden enabled",
)
INPUT_HASH_FIELDS = (
    "dataset_sha256",
    "mds_index_sha256",
    "row_manifest_sha256",
    "tokenizer_file_sha256",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text())


def load_module(name, path):
    if not path.is_file():
        raise FileNotFoundError(f"Required summarizer is unavailable: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def stable_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def fingerprint(value):
    return hashlib.sha256(stable_json(value).encode()).hexdigest()


def replace_checkout_paths(value):
    if isinstance(value, dict):
        return {key: replace_checkout_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [replace_checkout_paths(item) for item in value]
    if isinstance(value, str):
        return value.replace(str(BASELINE_CHECKOUT), "<CHECKOUT>").replace(
            str(FINAL_CHECKOUT), "<CHECKOUT>"
        )
    return value


def runtime_signature(runtime):
    normalized = replace_checkout_paths(copy.deepcopy(runtime))
    packages = normalized.get("packages", {})
    versions = {
        name: package.get("version")
        for name, package in sorted(packages.items())
        if isinstance(package, dict)
    }
    visible = {
        "python": normalized.get("python"),
        "torch": normalized.get("torch"),
        "torch_cuda": normalized.get("torch_cuda"),
        "cuda_home": normalized.get("cuda_home"),
        "cuda_device_count": normalized.get("cuda_device_count"),
        "nccl": normalized.get("nccl"),
        "gpus": normalized.get("gpus"),
        "package_versions": versions,
    }
    return fingerprint(normalized), visible, normalized


def first_differences(left, right, path="$", limit=12):
    differences = []

    def visit(a, b, key):
        if len(differences) >= limit:
            return
        if type(a) is not type(b):
            differences.append({"path": key, "left": a, "right": b})
            return
        if isinstance(a, dict):
            for child in sorted(set(a) | set(b)):
                child_path = f"{key}.{child}"
                if child not in a or child not in b:
                    differences.append({
                        "path": child_path,
                        "left": a.get(child, "<missing>"),
                        "right": b.get(child, "<missing>"),
                    })
                    if len(differences) >= limit:
                        return
                else:
                    visit(a[child], b[child], child_path)
                    if len(differences) >= limit:
                        return
        elif isinstance(a, list):
            if len(a) != len(b):
                differences.append({"path": f"{key}.length", "left": len(a), "right": len(b)})
                if len(differences) >= limit:
                    return
            for index, (left_item, right_item) in enumerate(zip(a, b)):
                visit(left_item, right_item, f"{key}[{index}]")
                if len(differences) >= limit:
                    return
        elif a != b:
            differences.append({"path": key, "left": a, "right": b})

    visit(left, right, path)
    return differences


def resolved_config(path, run_path, chunks):
    config = yaml.safe_load((path / "native/config/resolved_config.yaml").read_text())
    require(config["loss"]["num_chunks"] == chunks, f"{path}: incorrect loss chunks")
    require(config["debug"]["seed"] == 2048, f"{path}: resolved seed is not 2048")
    require(not config["profiler"]["enable_profiling"], f"{path}: profiler is enabled")
    require(not config["validator"]["enable"], f"{path}: validation is enabled")
    require(not config["checkpoint"]["enable"], f"{path}: checkpointing is enabled")

    def normalize(value):
        if isinstance(value, dict):
            return {key: normalize(item) for key, item in value.items()}
        if isinstance(value, list):
            return [normalize(item) for item in value]
        if isinstance(value, str):
            return (
                value.replace(str(run_path), "<RUN>")
                .replace(str(BASELINE_RAW), "<ARTIFACT_RAW>")
                .replace(str(FINAL_RAW), "<ARTIFACT_RAW>")
            )
        return value

    normalized = normalize(config)
    normalized["loss"]["num_chunks"] = "<CHUNKS>"
    return normalized


def stable_input_hashes(metadata, path):
    input_data = metadata["input_data"]
    missing = [field for field in INPUT_HASH_FIELDS if field not in input_data]
    require(not missing, f"{path}: missing input hashes {missing}")
    return {field: input_data[field] for field in INPUT_HASH_FIELDS}


def verify_observer(native, path):
    observer = native["compilation_observer"]
    require(observer["status"] == "observed_stable", f"{path}: compiler observer not stable")
    require(observer["observed_stable"] is True, f"{path}: observer summary is not stable")
    by_rank = observer["by_rank"]
    require(set(by_rank) == {str(rank) for rank in RANKS}, f"{path}: observer rank set differs")
    for rank, result in by_rank.items():
        require(result["observed_stable"] is True, f"{path}: observer unstable on rank {rank}")
        require(not result["errors"], f"{path}: observer errors on rank {rank}: {result['errors']}")
    return observer


def verify_baseline_activation(path):
    receipts = {}
    for rank in RANKS:
        logs = sorted(path.glob(f"logs/torchrun/*/attempt_0/{rank}/stdout.log"))
        require(len(logs) == 1, f"{path}: expected one stdout log for rank {rank}, got {logs}")
        log_path = logs[0]
        log = ANSI.sub("", log_path.read_text(errors="replace"))
        lines = [line.split(BOOTSTRAP_PREFIX, 1)[1]
                 for line in log.splitlines() if BOOTSTRAP_PREFIX in line]
        require(len(lines) == 1, f"{log_path}: expected one bridge bootstrap receipt")
        receipt = json.loads(lines[0])
        require(receipt["rank"] == rank, f"{log_path}: bridge rank mismatch")
        require(receipt["cross_entropy"] == "original", f"{log_path}: baseline CE is not original")
        require(receipt["backend_variant"] == "fla", f"{log_path}: unexpected GDN backend")
        require(receipt["routing_metadata"] == "original", f"{log_path}: routing metadata differs")
        require(receipt["capture_mode"] == "disabled", f"{log_path}: capture is enabled")
        callable_info = receipt["cross_entropy_callable"]
        require(callable_info["qualname"] == "cross_entropy_loss", f"{log_path}: CE callable differs")
        present = [marker for marker in BASELINE_FORBIDDEN_MARKERS if marker in log]
        require(not present, f"{log_path}: baseline has candidate activation markers {present}")
        receipts[rank] = {
            "rank": rank,
            "backend_variant": receipt["backend_variant"],
            "cross_entropy": receipt["cross_entropy"],
            "routing_metadata": receipt["routing_metadata"],
            "cross_entropy_callable": {
                "module": callable_info.get("module"),
                "qualname": callable_info["qualname"],
                "sha256": callable_info.get("sha256"),
            },
        }
    return receipts


def verify_final_activation(result, path):
    evidence = result["ce_evidence_by_rank"]
    require(set(evidence) == set(RANKS), f"{path}: final CE evidence rank set differs")
    for rank, receipt in evidence.items():
        require(receipt["rank"] == rank, f"{path}: final CE receipt rank mismatch")
        require(receipt["method"] == "cuda-13", f"{path}: final method is not cuda-13")
        require(receipt["backend"] == "inductor", f"{path}: final backend is not inductor")
    return evidence


def summarize_run(arm, path, chunks, baseline_summarizer, final_summarizer):
    if arm == "baseline":
        result = baseline_summarizer.summarize(path, "baseline", chunks)
        run_path = Path(result["run"])
        tokens = result["tokens_per_update"]
        tps_by_step = result["step_global_physical_tps"]
        median_tps = result["median_global_physical_tps"]
        peak_gib = result["peak_total_gpu_gib"]
    else:
        result, _ = final_summarizer.summarize(path, chunks)
        run_path = Path(result["run_path"])
        tokens = result["global_input_tokens_per_update"]
        tps_by_step = result["step_tps"]
        median_tps = result["median_global_tps"]
        peak_gib = result["peak_total_gpu_gib"]

    require(run_path.resolve() == path.resolve(), f"{path}: summarizer returned another run path")
    metadata = read_json(path / "metadata.json")
    completion = read_json(path / "completion.json")
    native = read_json(path / "summary.json")
    require(completion["status"] == "PASS", f"{path}: completion did not pass")
    for field in ("source_unchanged", "config_unchanged"):
        if field in completion:
            require(completion[field] is True, f"{path}: {field} is false")
    expected_id = path.name
    require(metadata["run_id"] == expected_id, f"{path}: metadata run id differs")
    require(metadata["seed"] == 2048, f"{path}: metadata seed is not 2048")
    for field in ("profiler", "validation", "checkpointing"):
        require(metadata[field] == "disabled", f"{path}: {field} is not disabled")

    inputs = stable_input_hashes(metadata, path)
    host = native["host"]
    require(isinstance(host, str) and host, f"{path}: missing native host identity")
    observer = verify_observer(native, path)
    runtime_hash, runtime_visible, runtime_normalized = runtime_signature(metadata["runtime"])
    config = resolved_config(path, path, chunks)
    config_hash = fingerprint(config)

    normalized_steps = {str(step): float(rate) for step, rate in tps_by_step.items()}
    require(set(normalized_steps) == {str(step) for step in STEPS}, f"{path}: measured step set differs")
    require(all(math.isfinite(rate) and rate > 0 for rate in normalized_steps.values()),
            f"{path}: invalid per-step physical rates")
    step_seconds = {step: tokens / rate for step, rate in normalized_steps.items()}
    median_seconds = statistics.median(step_seconds.values())
    require(math.isclose(median_tps, statistics.median(normalized_steps.values()), rel_tol=1e-12),
            f"{path}: reported median throughput differs from per-step rates")
    if arm == "final":
        require(math.isclose(median_seconds, result["median_s_per_update"], rel_tol=1e-12),
                f"{path}: reported median elapsed time differs from per-step rates")
        activation = verify_final_activation(result, path)
        source = result["source"]
        config_hashes = result["config_hashes"]
        observer_unverified = result["compilation_observer_unverified"]
    else:
        activation = verify_baseline_activation(path)
        source = result["source_fingerprint"]
        config_hashes = metadata.get("config_hashes_before", {})
        observer_unverified = observer["unverified"]

    row = {
        "arm": arm,
        "run_id": expected_id,
        "chunks": chunks,
        "run_path": str(path),
        "source": source,
        "config_hashes": config_hashes,
        "dataset_sha256": inputs["dataset_sha256"],
        "input_hashes": inputs,
        "seed": metadata["seed"],
        "profile_controls": {
            "profiler": metadata["profiler"],
            "validation": metadata["validation"],
            "checkpointing": metadata["checkpointing"],
        },
        "host": host,
        "runtime_fingerprint": runtime_hash,
        "runtime": runtime_visible,
        "normalized_resolved_config_sha256": config_hash,
        "tokens_per_update": tokens,
        "median_s_per_update": median_seconds,
        "median_global_tps": median_tps,
        "step_s": step_seconds,
        "step_tps": normalized_steps,
        "peak_total_gpu_gib": peak_gib,
        "peak_rank": result["peak_rank"],
        "peak_step": result["peak_step"],
        "compilation_observer_status": observer["status"],
        "compilation_observer_unverified": observer_unverified,
        "activation_receipt_by_rank": activation,
    }
    return row, config, runtime_normalized


def make_pair(baseline, final):
    throughput_ratio = final["median_global_tps"] / baseline["median_global_tps"]
    return {
        "chunks": baseline["chunks"],
        "baseline": baseline,
        "final": final,
        "final_over_baseline_tps": throughput_ratio,
        "final_tps_change_percent": 100.0 * (throughput_ratio - 1.0),
        "final_minus_baseline_median_s": (
            final["median_s_per_update"] - baseline["median_s_per_update"]
        ),
        "final_minus_baseline_peak_gib": (
            final["peak_total_gpu_gib"] - baseline["peak_total_gpu_gib"]
        ),
    }


def validate_matrix(rows):
    reference = rows[0]
    common_inputs = reference["input_hashes"]
    common_runtime = reference["_runtime_normalized"]
    common_host = reference["host"]
    common_config = reference["_config"]
    seed = reference["seed"]
    for row in rows[1:]:
        require(row["seed"] == seed == 2048, "Run matrix seed differs")
        require(row["input_hashes"] == common_inputs, "Run matrix input content hashes differ")
        require(row["host"] == common_host, "Run matrix host identity differs")
        if row["_runtime_normalized"] != common_runtime:
            differences = first_differences(common_runtime, row["_runtime_normalized"])
            raise ValueError(f"Run matrix runtime differs: {differences}")
        if row["_config"] != common_config:
            differences = first_differences(common_config, row["_config"])
            raise ValueError(f"Normalized resolved config differs: {differences}")
        require(
            row["profile_controls"] == reference["profile_controls"] == {
                "profiler": "disabled", "validation": "disabled", "checkpointing": "disabled",
            },
            "Run matrix profile controls differ or are enabled",
        )
        require(
            row["compilation_observer_status"] == "observed_stable",
            f"{row['run_id']}: compiler observer status differs",
        )

    baseline_sources = [row["source"] for row in rows if row["arm"] == "baseline"]
    final_sources = [row["source"] for row in rows if row["arm"] == "final"]
    require(len(baseline_sources) == len(final_sources) == len(CHUNKS), "Matrix arm counts differ")
    require(all(source == baseline_sources[0] for source in baseline_sources),
            "Baseline source snapshot changed across chunk counts")
    require(all(source == final_sources[0] for source in final_sources),
            "Final source snapshot changed across chunk counts")
    return {
        "seed": seed,
        "input_hashes": common_inputs,
        "host": common_host,
        "runtime_fingerprint": reference["runtime_fingerprint"],
        "normalized_resolved_config_sha256": fingerprint(common_config),
        "profile_controls": reference["profile_controls"],
        "compiler_observer": "observed_stable on all eight ranks in all six runs",
        "config_normalization": [
            "replace per-run artifact paths with <RUN>",
            "replace checkout artifact roots in config values with <ARTIFACT_RAW>",
            "replace loss.num_chunks with <CHUNKS>",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--series", default="ab01")
    args = parser.parse_args()
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.series) is not None,
            "Series may contain only letters, digits, underscores, and hyphens")

    baseline_summarizer = load_module("_ep8_baseline_summarizer", BASELINE_SUMMARIZER)
    final_summarizer = load_module("_ep8_final_summarizer", Path(__file__).with_name("summarize.py"))
    rows = []
    paired = []
    for chunks in CHUNKS:
        baseline_id = f"lhce-{args.series}-a-c{chunks}"
        final_id = f"lhce-{args.series}-b-c{chunks}"
        baseline, baseline_config, baseline_runtime = summarize_run(
            "baseline", BASELINE_RAW / baseline_id, chunks,
            baseline_summarizer, final_summarizer,
        )
        final, final_config, final_runtime = summarize_run(
            "final", FINAL_RAW / final_id, chunks,
            baseline_summarizer, final_summarizer,
        )
        require(baseline_config == final_config,
                f"C{chunks}: normalized Baseline A and final resolved configs differ: "
                f"{first_differences(baseline_config, final_config)}")
        require(baseline_runtime == final_runtime,
                f"C{chunks}: Baseline A and final runtime configurations differ: "
                f"{first_differences(baseline_runtime, final_runtime)}")
        rows.extend([
            {**baseline, "_config": baseline_config, "_runtime_normalized": baseline_runtime},
            {**final, "_config": final_config, "_runtime_normalized": final_runtime},
        ])
        paired.append(make_pair(baseline, final))

    comparability = validate_matrix(rows)
    for row in rows:
        row.pop("_config")
        row.pop("_runtime_normalized")
    output = {
        "schema_version": 1,
        "series": args.series,
        "comparability": comparability,
        "pairs": paired,
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    for pair in paired:
        print(
            f"C{pair['chunks']}: baseline={pair['baseline']['median_global_tps']:.3f} "
            f"final={pair['final']['median_global_tps']:.3f} "
            f"change={pair['final_tps_change_percent']:+.3f}% "
            f"peak_delta={pair['final_minus_baseline_peak_gib']:+.2f} GiB"
        )


if __name__ == "__main__":
    main()
