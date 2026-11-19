"""Summarize one native 10+10 window and verify the selected production arms."""

import argparse
import hashlib
import json
import re
from pathlib import Path

import yaml


def read(path):
    return json.loads(path.read_text())


def extract(run, fixes):
    done = read(run / "completion.json")
    config_path = run / "native/config/resolved_config.yaml"
    assert (
        done["status"] == "PASS"
        and done["config_unchanged"]
        and done["source_unchanged"]
    )
    assert (
        hashlib.sha256(config_path.read_bytes()).hexdigest()
        == done["resolved_config_sha256"]
    )
    config = yaml.safe_load(config_path.read_text())
    config.pop("dump_folder")
    info = read(run / "metadata.json")
    window = read(run / "summary.json")
    log = re.sub(r"\x1b\[[0-9;]*m", "", (run / "logs/launcher.log").read_text())
    assert window["status"] == "complete" and not window["errors"]
    assert window["world_size"] == 8 and window["training_steps"] == 20
    assert (
        window["window"]["start_after_update"] == 10
        and window["window"]["end_after_update"] == 20
    )
    assert window["tokens"]["physical_input_tokens_global"] == 20_971_520
    assert info["profiler"] == "disabled" and info["detailed_monitoring"].startswith(
        "disabled"
    )
    groups = window["memory"]["allocator_counter_deltas_by_rank"]
    assert len(groups) == 8 and all(
        all(v == 0 for v in row.values()) for row in groups.values()
    )
    for rank in range(8):
        boot = read(run / f"measurement/flash-qla-bootstrap-rank-{rank:02d}.json")
        assert (
            boot["native_main_completed"] and boot["backend_variant"] == "external_gate"
        )
        assert boot["installer_identity"]["direct_backward"] == (
            "gdn-direct-backward" in fixes
        )
        prefix = f"[default{rank}]:kernel_fusion "
        assert prefix + "backward_arm=folded-ce" in log
        seen = set(re.findall(re.escape(prefix) + r"fix=([\w-]+)", log))
        assert seen == set(fixes), (rank, seen, fixes)
        executed = set(re.findall(rf"fusion_pack executed=([\w-]+) rank={rank}\b", log))
        assert executed == set(fixes), (rank, executed, fixes)
    step_stats = re.findall(
        r"\[default0\]:.*?step:\s*(\d+)\s+loss:\s*(\S+)\s+grad_norm:\s*(\S+)", log
    )
    memory = window["memory"]["maximum_peak_gib_across_ranks"]
    return dict(
        id=run.name,
        source=info["source_head_before"],
        gift_sha256=info["source_snapshot_before"]["tracked_and_untracked_sha256"],
        normalized_config_sha256=hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest(),
        resolved_sha256=done["resolved_config_sha256"],
        fixes=sorted(fixes),
        physical_tps=window["tokens"]["physical_input_tokens_per_second_global"],
        elapsed_s=window["window"]["primary_elapsed_s"],
        median_update_s=window["host_boundary_cadence"]["ordinary_steps"]["median_s"],
        update_seconds_by_rank={
            r: [i["interval_s"] for i in data["intervals"]]
            for r, data in window["host_boundary_cadence"]["by_rank"].items()
        },
        allocated_gib=memory["allocated_bytes.all.peak"],
        reserved_gib=memory["reserved_bytes.all.peak"],
        compiler_stable=window["compilation_observer"]["observed_stable"],
        printed_loss_and_grad_norm=step_stats,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("fix", nargs="*")
    args = parser.parse_args()
    print(json.dumps(extract(args.run, args.fix), indent=2))
