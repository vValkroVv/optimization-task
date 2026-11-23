"""Verify native 10+10 evidence and summarize elapsed training and total memory."""

import argparse
import json
from pathlib import Path
import re
from statistics import median

import yaml


STEPS = tuple(range(11, 21))
RANKS = tuple(range(8))
ANSI = re.compile(r"\x1b\[[0-9;]*m")
METRICS = re.compile(r"\bstep:\s*(\d+).*?\bmemory:\s*([0-9.]+)\s*GiB")
EXACT_CE_DONE = "lm_final_exact_ce_complete="


def read_json(path):
    return json.loads(path.read_text())


def native_peak_records(log):
    rows = []
    for line in log.splitlines():
        if match := METRICS.search(line):
            rows.append((int(match[1]), float(match[2])))
    assert [step for step, _ in rows] == list(range(1, 21)), rows
    return {step: value for step, value in rows if step in STEPS}


def summarize(path, chunks):
    completion = read_json(path / "completion.json")
    assert completion["status"] == "PASS", completion
    assert completion["source_unchanged"] and completion["config_unchanged"], completion
    native = read_json(path / "summary.json")
    assert native["status"] == "complete", path
    assert native["world_size"] == 8 and native["training_steps"] == 20, path
    meta = read_json(path / "metadata.json")
    assert meta["warmup_updates"] == meta["window_updates"] == 10, path
    assert meta["profiler"] == meta["validation"] == meta["checkpointing"] == "disabled", path

    resolved_path = path / "native/config/resolved_config.yaml"
    resolved = yaml.safe_load(resolved_path.read_text())
    assert resolved["loss"]["num_chunks"] == chunks, resolved_path
    assert resolved["loss"]["loss_weights"] == [0.3], resolved_path
    assert resolved["compile"]["enable"] and set(resolved["compile"]["components"]) == {
        "model", "loss",
    }, resolved_path
    mesh = resolved["parallelism"]
    for field, expected in {
        "data_parallel_replicate_degree": 1,
        "data_parallel_shard_degree": 8,
        "expert_parallel_degree": 8,
        "tensor_parallel_degree": 1,
        "pipeline_parallel_degree": 1,
        "context_parallel_degree": 1,
    }.items():
        assert mesh[field] == expected, (resolved_path, field)
    training = resolved["training"]
    assert (training["local_batch_size"], training["global_batch_size"], training["seq_len"]) == (
        2, 512, 4096,
    ), resolved_path
    assert "ema" in resolved["callbacks"], resolved_path

    per_rank = native["host_boundary_cadence"]["by_rank"]
    expected_calls = 20 * 32 * chunks
    memory = {}
    durations = {}
    ce_evidence = {}
    plan = {2: "9.14", 8: "5.5", 32: "5.5"}[chunks]
    for rank in RANKS:
        log_path, = path.glob(f"logs/torchrun/*/attempt_0/{rank}/stdout.log")
        log = ANSI.sub("", log_path.read_text())
        memory[rank] = native_peak_records(log)
        ce, = [json.loads(line.split(EXACT_CE_DONE, 1)[1])
               for line in log.splitlines() if EXACT_CE_DONE in line]
        assert (ce["rank"], ce["chunks"], ce["calls"]) == (rank, chunks, expected_calls), ce
        assert ce["method"] == "cuda-13" and ce["backend"] == "inductor", ce
        assert ce["incoming_fp32_scales"] == ce["prepared_fp32_scales"], ce
        assert min(ce["incoming_fp32_scales"]) > 0, ce
        ce_evidence[rank] = ce
        for marker in (
            f"LM_HEAD_MATRIX_DW_WRITER rank={rank} mode=new chunks={chunks} writer=cublasLtMatmul pair={plan}",
            "Chunked loss direct BF16 dHidden enabled",
            "Stable Folded CE active: compiled=True inplace=True early_logits_release=True",
            "LM_HEAD_MATRIX_FUSED_Z_LOSS_ACTIVE",
        ):
            assert marker in log, (log_path, marker)
        if rank == 0:
            assert f"LM_HEAD_MATRIX_DW_REDUCED rank=0 mode=new chunks={chunks}" in log, log_path
        durations[rank] = {row["step"]: row["interval_s"]
                           for row in per_rank[str(rank)]["intervals"]}
        assert set(durations[rank]) == set(STEPS) and min(durations[rank].values()) > 0, rank

    tokens = training["global_batch_size"] * training["seq_len"]
    by_step = {step: max(durations[rank][step] for rank in RANKS) for step in STEPS}
    rate = {step: tokens / by_step[step] for step in STEPS}
    peak_gib, peak_rank, peak_step = max(
        (gib, rank, step) for rank, samples in memory.items() for step, gib in samples.items()
    )
    observer = native["compilation_observer"]
    receipt = {
        "run_id": path.name,
        "run_path": str(path),
        "chunks": chunks,
        "dataset_sha256": meta["input_data"]["dataset_sha256"],
        "source": meta["source_snapshot_before"],
        "config_hashes": meta["config_hashes_before"],
        "rank0_resolved_config": str(resolved_path),
        "global_input_tokens_per_update": tokens,
        "median_s_per_update": median(by_step.values()),
        "median_global_tps": median(rate.values()),
        "tps_range": [min(rate.values()), max(rate.values())],
        "peak_total_gpu_gib": peak_gib,
        "peak_rank": peak_rank,
        "peak_step": peak_step,
        "step_s": by_step,
        "step_tps": rate,
        "memory_gib_by_rank": memory,
        "ce_evidence_by_rank": ce_evidence,
        "compilation_observer_status": observer["status"],
        "compilation_observer_unverified": observer["unverified"],
    }
    resolved.pop("dump_folder")
    resolved["loss"].pop("num_chunks")
    return receipt, resolved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("runs", nargs="+", help="each argument is CHUNKS=/path/to/native/run")
    args = parser.parse_args()
    results = []
    control_config = None
    for run in args.runs:
        chunks, path = run.split("=", 1)
        receipt, config = summarize(Path(path), int(chunks))
        if control_config is None:
            control_config = config
        assert config == control_config, "Resolved configurations differ outside loss chunks"
        results.append(receipt)
    assert all(row["source"] == results[0]["source"] for row in results)
    assert all(row["dataset_sha256"] == results[0]["dataset_sha256"] for row in results)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    for row in results:
        print(row["chunks"], row["median_global_tps"], row["median_s_per_update"],
              row["peak_total_gpu_gib"], row["compilation_observer_status"])


if __name__ == "__main__":
    main()
