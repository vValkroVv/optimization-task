#!/usr/bin/env python3
"""Measure Expert GFSDP changes on the original A recipe, on eight H100s."""

import argparse
import json
import re
import sys
from itertools import batched
from math import isfinite
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BRIDGE_PATH = Path("runs/flash-qla/run.py")
RAW_PATH = Path("runs/overnight-streaming-perf/artifacts/raw")
REFERENCE_REV = "e9b7cdbb4cd687685cbbfdeebaf2d6bd6e19e85f"
DATA_SHA256 = "1ce62620b6dd2cf5fde6715ed5776a247d1cc1bd2df725f5c1e5f2b0effc3a70"
FLAGS = {
    "bf16-cache": "gfsdp_reuse_bf16_expert_weights",
    "fp8-cache": "gfsdp_fp8_expert_weights",
    "direct-dw": "gfsdp_direct_expert_dw",
    "tail-retention": "gfsdp_retain_tail_weights",
}
ANSI = re.compile(r"\x1b\[[0-9;]*m")
METRICS = re.compile(
    r"\bstep:\s*(\d+)\s+loss:\s*(\S+)\s+grad_norm:\s*(\S+)\s+"
    r"memory:\s*([\d.]+)\s*GiB.*?tps:\s*([\d,]+)"
)


def read_json(path):
    return json.loads(path.read_text())


def load_bridge(path, name):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def native_overrides(command):
    flags = command[command.index("--") + 2 :]
    if len(flags) % 2:
        raise RuntimeError("Native command has an override without a value")
    pairs = list(batched(flags, 2))
    overrides = dict(pairs)
    if len(overrides) != len(pairs):
        raise RuntimeError("Native command repeats a configuration override")
    overrides.pop("--dump_folder")
    return overrides


def configure_data(base):
    # The immutable original A inputs live in the original task's data directory.
    receipt = read_json(base.DATA_RECEIPT)["dataset"]
    if receipt["dataset_sha256"] != DATA_SHA256:
        raise RuntimeError("The tracked data receipt differs from the original A input")
    base.DATASET_ROOT = Path(receipt["root"])
    base.ROW_MANIFEST = Path(receipt["row_manifest"]["path"])
    inside = base._inside
    configured_path = base._configured_path
    canonical_dataset = Path("../runs/production-data-abcd/artifacts/raw/data-002")

    def allow_verified_input(root, path, label):
        if label == "MDS data path" and path == base.DATASET_ROOT:
            return path.resolve()
        return inside(root, path, label)

    def resolve_overlaid_dataset(value, *, base: Path, label):
        if label == "canonical dataset stream local" and value == str(
            canonical_dataset
        ):
            return dataset_root
        return configured_path(value, base=base, label=label)

    # The experiment imports the canonical crop and then the absolute MDS
    # fragment. Preflight must check the same effective input as the native CLI.
    dataset_root = base.DATASET_ROOT.resolve()
    base._inside = allow_verified_input
    base._configured_path = resolve_overlaid_dataset


def verify_reference(base, path):
    metadata = read_json(path / "metadata.json")
    completion = read_json(path / "completion.json")
    config_path = path / "native/config/resolved_config.yaml"
    hashes = metadata["config_hashes_before"]
    reference_input = metadata["input_data"]
    active_input = base._verify_ready_inputs(base.TASK_ROOT)
    if (
        metadata["source_head_before"] != REFERENCE_REV
        or any(
            reference_input[key] != active_input[key]
            for key in (
                "dataset_sha256",
                "mds_index_sha256",
                "row_manifest_sha256",
                "tokenizer_file_sha256",
            )
        )
        or completion["status"] != "PASS"
        or base.sha256_file(config_path) != completion["resolved_config_sha256"]
        or base.sha256_file(base.BASE_CONFIG) != hashes["canonical_experiment_sha256"]
    ):
        raise RuntimeError(
            "The supplied reference does not match the verified original A recipe"
        )
    resolved = base._load_yaml(config_path)
    resolved_stream = resolved["dataloader"]["dataset"]["streams"][0]
    if resolved_stream["local"] != reference_input["dataset_root"]:
        raise RuntimeError(
            "Original A resolved stream does not match its input receipt"
        )

    active_fragment = base._load_yaml(base.DATA_FRAGMENT)
    fragment_stream = active_fragment["dataloader"]["dataset"]["streams"][0]
    if base.sha256_file(base.DATA_FRAGMENT) != hashes["mds_fragment_sha256"]:
        experiment = path / "config/experiment.yaml"
        if base.sha256_file(experiment) != hashes["generated_experiment_sha256"]:
            raise RuntimeError(
                "Original A generated experiment differs from its receipt"
            )
        historical_path = Path(base._load_yaml(experiment)["imports"][1])
        if base.sha256_file(historical_path) != hashes["mds_fragment_sha256"]:
            raise RuntimeError("Original A MDS fragment differs from its receipt")
        historical = base._load_yaml(historical_path)
        historical_stream = historical["dataloader"]["dataset"]["streams"][0]
        if historical_stream["local"] != resolved_stream["local"]:
            raise RuntimeError(
                "Original A MDS fragment differs from its resolved input"
            )
        historical_stream["local"] = fragment_stream["local"]
        if historical != active_fragment:
            raise RuntimeError("MDS fragment changed beyond the verified input path")
    resolved_stream["local"] = fragment_stream["local"]
    return metadata["argv"], resolved


def configure_driver(base, args, base_config):
    # Let the native runner produce and record all four CLI switches.
    if args.original_a_source is None:
        base.NATIVE_FEATURE_CONTROLS += tuple(
            (field, "--" + field.replace("_", "-"), f"enable {field}")
            for name, field in FLAGS.items()
            if name != "bf16-cache"
        )
    native_args = base._training_overrides
    expect_config = base._resolved_config_expectations

    def training_overrides(parsed, paths):
        overrides = native_args(parsed, paths)
        if args.ep != 8:
            overrides.extend(("--parallelism.expert_parallel_degree", str(args.ep)))
        return overrides

    base._training_overrides = training_overrides
    base._resolved_config_expectations = lambda parsed: {
        **expect_config(parsed),
        "parallelism.expert_parallel_degree": args.ep,
    }
    if base_config is None:
        return
    expected = base_config.copy()
    expected.pop("dump_folder")
    expected["training"] = {**expected["training"], "steps": 20}
    expected["parallelism"] = {
        **expected["parallelism"],
        "expert_parallel_degree": args.ep,
    }
    if args.original_a_source is None:
        expected["parallelism"].update(
            {field: name in args.optimization for name, field in FLAGS.items()}
        )
    verify_config = base._verify_resolved_config

    def verify_resolved(paths, parsed):
        digest = verify_config(paths, parsed)
        resolved = base._load_yaml(paths["native"] / "config/resolved_config.yaml")
        resolved.pop("dump_folder")
        if resolved != expected:
            differing = sorted(
                key
                for key in resolved.keys() | expected.keys()
                if resolved.get(key) != expected.get(key)
            )
            raise RuntimeError(
                f"Resolved configuration differs from original A at {differing}"
            )
        return digest

    base._verify_resolved_config = verify_resolved


def driver(args):
    task_root = args.original_a_source or ROOT
    bridge_path = task_root / BRIDGE_PATH
    raw = task_root / RAW_PATH
    flash = load_bridge(bridge_path, "_expert_gfsdp_bridge")
    base = flash._load_base_runner()
    if args.original_a_source is not None:
        revision = base._git(task_root, "rev-parse", "HEAD").stdout.decode().strip()
        dirty_gift = base._git(
            task_root, "status", "--porcelain", "-uall", "--", "gift/"
        ).stdout
        if revision != REFERENCE_REV or dirty_gift:
            raise RuntimeError(
                "Original A requires clean GIFT at the pinned e9b7cdbb4 revision"
            )
    configure_data(base)
    supplied = set(args.optimization)
    if args.reference:
        reference_command, reference_config = verify_reference(base, args.reference)
        sms = reference_config["fp8"]["sparse_deep_gemm_num_sms"]
    else:
        plan = read_json(ROOT / "runs/production-data-abcd/campaign-plan.json")
        reference_command = plan["runs"][0]["child_torchrun_argv"]
        reference_config = None
        sms = base._load_yaml(base.BASE_CONFIG)["fp8"]["sparse_deep_gemm_num_sms"]
    if type(sms) is not int or sms <= 0:
        raise RuntimeError("Original A must provide a positive DeepGEMM SM budget")
    configure_driver(base, args, reference_config)
    expected_cli = native_overrides(reference_command)
    expected_cli["--training.steps"] = "20"
    if args.ep != 8:
        expected_cli["--parallelism.expert_parallel_degree"] = str(args.ep)
    for name in supplied:
        expected_cli[f"--parallelism.{FLAGS[name]}"] = "true"

    build = args.direct_dw_build
    build_hash = (
        base.sha256_file(build / "manifest.json")
        if build and args.mode == "run"
        else None
    )
    build_command = flash._build_command_hook

    def wrap_command(underlying):
        command = build_command(underlying)

        def patched(parsed, paths):
            argv = command(parsed, paths)
            overrides = native_overrides(argv)
            if overrides != expected_cli:
                differing = sorted(
                    key
                    for key in overrides.keys() | expected_cli.keys()
                    if overrides.get(key) != expected_cli.get(key)
                )
                raise RuntimeError(
                    f"Native CLI differs from the original A experiment at {differing}"
                )
            measure = argv.index(str(bridge_path.with_name("measure.py")))
            argv[measure : measure + 1] = [
                str(Path(__file__).resolve()),
                "rank",
                str(task_root),
                str(args.ep),
                ",".join(sorted(supplied)),
                str(build or ""),
                str(sms),
            ]
            return argv

        return patched

    flash._load_base_runner = lambda: base
    flash._build_command_hook = wrap_command
    base_config_hashes = base._config_hashes

    def config_hashes():
        hashes = base_config_hashes()
        hashes["expert_gfsdp_runner_sha256"] = base.sha256_file(Path(__file__))
        if build is not None:
            current = base.sha256_file(build / "manifest.json")
            if current != build_hash:
                raise RuntimeError("Direct dW build manifest changed")
            hashes["direct_dw_build_manifest_sha256"] = current
        return hashes

    base._config_hashes = config_hashes
    argv = [
        "--gdn-backend",
        "fla",
        "--routing-metadata",
        "original",
        "--cross-entropy",
        "original",
        "--num-prefetch-batches",
        "0",
        "--epoch-size",
        "102400",
        "--steps",
        "20",
        "--warmup",
        "10",
        "--window",
        "10",
        "--observe-compiler-cache",
        "--run-id",
        args.run_id,
        "--output-root",
        str(raw),
    ]
    argv.extend("--" + FLAGS[name].replace("_", "-") for name in sorted(supplied))
    if args.compiler_cache_root is not None:
        argv.extend(("--compiler-cache-root", str(args.compiler_cache_root)))
    if args.mode == "plan":
        bridge, native = flash._parse_bridge_args(argv)
        parsed = base._parse_args(native)
        with TemporaryDirectory(prefix="expert-gfsdp-plan-") as temporary:
            paths = {
                name: Path(temporary)
                for name in ("config", "native", "measurement", "logs")
            }
            base._write_experiment_config(paths["config"] / "experiment.yaml")
            command = wrap_command(base)(parsed, paths)
            measure = load_bridge(base.HELPER_DIR / "measure.py", "_gfsdp_plan")
            child_args = command[command.index("rank") + 6 :]
            measure._parse_args(child_args)
        print(
            json.dumps({"bridge": vars(bridge), "argv": command}, default=str, indent=2)
        )
        return 0
    # The native rank wrapper resolves its standard raw root strictly.
    raw.mkdir(parents=True, exist_ok=True)
    result = flash.main(argv)
    if result == 0:
        summarize(raw / args.run_id)
    return result


def capture_activation(measurement, ep, selected):
    from torchtitan.distributed.gfsdp.fully_shard import _iter_fsdp_param_groups

    resolved = measurement.trainer.config.parallelism
    for name, field in FLAGS.items():
        if getattr(resolved, field, False) != (name in selected):
            raise RuntimeError(
                f"Rank {measurement.rank} resolved an unexpected value for {field}"
            )
    groups = [
        group
        for part in measurement.trainer.model_parts
        for group in _iter_fsdp_param_groups(part, recurse=True)
    ]
    expert_groups = [
        group
        for group in groups
        if {p._module_info.param_name for p in group.fsdp_params}
        == {"w13_E2FD", "w2_EDF"}
    ]
    if not expert_groups:
        raise RuntimeError("No routed-expert GFSDP groups found")
    rows = []
    for group in expert_groups:
        cache = getattr(group, "_fp8_cache", None)
        row = {
            "fqn": group._module_fqn,
            "efsdp": group._all_gather_process_group.size(),
            "bf16-cache": group._bf16_cache_enabled
            and group._bf16_cache_prepared_generation >= 0,
            "fp8-cache": cache is not None
            and cache.prepared_generation >= 0
            and cache.ready_event is not None,
            "direct-dw": getattr(group, "_direct_expert_dw_slot", None) is not None,
            "tail-retention": getattr(group.comm_ctx, "retain_tail_weights", False),
        }
        expected = selected - {"bf16-cache"} if "fp8-cache" in selected else selected
        if row["efsdp"] != 8 // ep or any(
            row[name] != (name in expected) for name in FLAGS
        ):
            raise RuntimeError(f"Unexpected Expert GFSDP activation for {row}")
        rows.append(row)
    path = measurement.output_dir / f"expert-gfsdp-rank-{measurement.rank:02d}.json"
    with path.open("x") as output:
        json.dump(
            {"ep": ep, "requested": sorted(selected), "groups": rows}, output, indent=2
        )
        output.write("\n")


def run_rank(task_root, ep, selected, build, sms, native_args):
    measure = (Path(task_root) / BRIDGE_PATH).with_name("measure.py")
    flash = load_bridge(measure, "_expert_gfsdp_measure")
    native_main = flash._bootstrap_and_run
    load_base = flash._load_base_measure

    def load_measurement():
        base = load_base()

        class ExpertMeasurement(base.WindowMeasurement):
            def capture_after_train(self):
                super().capture_after_train()
                capture_activation(self, ep, selected)

            def _capture_bf16_cache_stats(self, *, update, boundary, require_active):
                super()._capture_bf16_cache_stats(
                    update=update,
                    boundary=boundary,
                    require_active=require_active and "fp8-cache" not in selected,
                )

        base.WindowMeasurement = ExpertMeasurement
        return base

    def bootstrap(argv, adapter, train):
        def configure_then_train(train_argv, training_adapter):
            if build:
                from torchtitan.ops.float8.direct_expert_dw import (
                    configure_direct_expert_dw,
                )

                component = load_bridge(
                    HERE / "direct-dw/direct_dw.py", "_expert_direct_dw_build"
                )
                extension = component.load_direct_dw(Path(build), num_sms=sms)
                configure_direct_expert_dw(
                    extension.k_grouped_fp8_gemm_nt_contiguous_rounded_f32
                )
            return train(train_argv, training_adapter)

        return native_main(argv, adapter, configure_then_train)

    flash._load_base_measure = load_measurement
    flash._bootstrap_and_run = bootstrap
    flash.main(native_args)
    return 0


def summarize(path):
    completion = read_json(path / "completion.json")
    summary = read_json(path / "summary.json")
    metadata = read_json(path / "metadata.json")
    if (
        completion["status"] != "PASS"
        or summary["status"] != "complete"
        or summary["window"]["start_after_update"] != 10
        or summary["window"]["end_after_update"] != 20
        or summary["tokens"]["physical_input_tokens_global"] != 20_971_520
    ):
        raise RuntimeError(f"Missing complete 10+10 window: {path}")
    rank_receipts = []
    per_rank_metrics = {}
    for rank in range(8):
        receipt = read_json(
            path / f"measurement/flash-qla-bootstrap-rank-{rank:02d}.json"
        )
        if not receipt["native_main_completed"] or any(
            receipt[key] != value
            for key, value in (
                ("backend_variant", "fla"),
                ("routing_metadata", "original"),
                ("cross_entropy", "original"),
            )
        ):
            raise RuntimeError(f"Unverified original A backend on rank {rank}")
        rank_receipts.append(
            read_json(path / f"measurement/expert-gfsdp-rank-{rank:02d}.json")
        )
        (stdout,) = (path / "logs/torchrun").glob(f"*/attempt_0/{rank}/stdout.log")
        rows = [
            [
                int(step),
                float(loss),
                float(grad),
                float(memory),
                int(tps.replace(",", "")),
            ]
            for step, loss, grad, memory, tps in METRICS.findall(
                ANSI.sub("", stdout.read_text())
            )
        ]
        if [row[0] for row in rows] != list(range(1, 21)):
            raise RuntimeError(f"Missing native update logs for rank {rank}")
        for update, loss, grad, _, _ in rows:
            if not (isfinite(loss) and isfinite(grad)):
                raise RuntimeError(
                    f"Non-finite loss or gradient norm at rank {rank}, update {update}"
                )
        per_rank_metrics[rank] = rows
    if len({(row["ep"], tuple(row["requested"])) for row in rank_receipts}) != 1:
        raise RuntimeError("Expert GFSDP rank activation receipts disagree")
    memory, update, rank = max(
        (row[3], row[0], rank)
        for rank, rows in per_rank_metrics.items()
        for row in rows[10:]
    )
    result = {
        "run_id": path.name,
        "ep": rank_receipts[0]["ep"],
        "requested": rank_receipts[0]["requested"],
        "total_gpu_memory_gib": memory,
        "memory_max_rank": rank,
        "memory_max_update": update,
        "window_global_tps": summary["tokens"][
            "physical_input_tokens_per_second_global"
        ],
        "median_step_global_tps": median(
            sum(rows[step][4] for rows in per_rank_metrics.values())
            for step in range(10, 20)
        ),
        "allocator_diagnostics": summary["memory"],
        "compiler_observer": summary["compilation_observer"],
        "compiler_cache_policy": metadata["compiler_cache_policy"],
        "compiler_cache_root": metadata["compiler_cache_root"],
        "gift_source_sha256": metadata["source_snapshot_before"][
            "tracked_and_untracked_sha256"
        ],
        "resolved_config_sha256": completion["resolved_config_sha256"],
        "console_fields": [
            "update",
            "loss",
            "grad_norm",
            "total_gpu_memory_gib",
            "rank_tps",
        ],
        "per_rank_console": per_rank_metrics,
        "per_rank_activation": rank_receipts,
    }
    rendered = json.dumps(result, indent=2)
    result_path = path / "expert-gfsdp-summary.json"
    if not result_path.exists():
        with result_path.open("x") as output:
            output.write(rendered + "\n")
    print(rendered)


def main():
    if sys.argv[1:2] == ["rank"]:
        task_root, ep, selected, build, sms, *native = sys.argv[2:]
        return run_rank(
            task_root, int(ep), set(selected.split(",")) - {""}, build, int(sms), native
        )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "run", "summarize"))
    parser.add_argument("run_id")
    parser.add_argument("--ep", type=int, choices=(4, 8), default=8)
    parser.add_argument("--optimization", action="append", choices=FLAGS, default=[])
    parser.add_argument("--direct-dw-build", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--original-a-source", type=Path)
    parser.add_argument("--compiler-cache-root", type=Path)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.run_id):
        parser.error("run_id must be a single artifact directory name")
    if args.original_a_source is not None:
        if args.optimization:
            parser.error(
                "the clean original A source does not support candidate switches"
            )
        args.original_a_source = args.original_a_source.resolve(strict=True)
    if args.mode == "summarize":
        summarize((args.original_a_source or ROOT) / RAW_PATH / args.run_id)
        return 0
    if len(set(args.optimization)) != len(args.optimization):
        parser.error("each optimization may be selected once")
    if args.mode == "run" and args.reference is None:
        parser.error("run requires --reference pointing to a completed original A run")
    if ("direct-dw" in args.optimization) != (args.direct_dw_build is not None):
        parser.error("--direct-dw-build is required only when direct-dw is selected")
    if args.direct_dw_build is not None:
        args.direct_dw_build = args.direct_dw_build.resolve(strict=args.mode == "run")
    return driver(args)


if __name__ == "__main__":
    raise SystemExit(main())
