"""Run the original A crop with CPU batch prefetch as the only treatment."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RAW = HERE / "artifacts/raw"
REVISION = "e9b7cdbb4cd687685cbbfdeebaf2d6bd6e19e85f"
GIFT_TREE = "0f44b971069af55d6d573628129900483ca3e1dc"
INPUT_ROOT = ROOT.parent / "optimization-task"
REFERENCE = (
    ROOT.parent / "a10-cute-001/runs/production-data-abcd/artifacts/raw"
    / "original-a-vs-cute-140-001-a"
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def read_json(path):
    return json.loads(path.read_text())


def different_keys(expected, actual, prefix=""):
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(expected.keys() | actual.keys()):
            path = f"{prefix}.{key}" if prefix else key
            if key not in expected or key not in actual:
                yield path
            else:
                yield from different_keys(expected[key], actual[key], path)
    elif expected != actual:
        yield prefix


def native_overrides(command):
    flags = command[command.index("--") + 2 :]
    require(len(flags) % 2 == 0, "Unpaired training override")
    settings = dict(zip(flags[::2], flags[1::2], strict=True))
    require(len(settings) * 2 == len(flags), "Repeated training override")
    settings.pop("--dump_folder")
    return settings


def load_bridge():
    path = ROOT / "runs/flash-qla/run.py"
    spec = importlib.util.spec_from_file_location("_prefetch_original_a", path)
    bridge = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = bridge
    spec.loader.exec_module(bridge)
    return bridge


def run(args):
    require(git("branch", "--show-current") == "prefetch-fix", "Use the prefetch-fix worktree")
    require(git("rev-parse", "HEAD:gift") == GIFT_TREE, "GIFT tree differs from original A")
    require(not git("status", "--porcelain", "-uall", "--", "gift"), "GIFT files differ from HEAD")

    # Pin the dataset and its metadata to the canonical read-only inputs.
    data = INPUT_ROOT / "runs/production-data-abcd"
    dataset = (data / "artifacts/raw/data-002").resolve(strict=True)
    reference_config = REFERENCE / "native/config/resolved_config.yaml"
    metadata = read_json(REFERENCE / "metadata.json")
    done = read_json(REFERENCE / "completion.json")
    require(
        metadata["source_head_before"] == REVISION and done["status"] == "PASS",
        "Reference is not the verified original A run",
    )
    expected_overrides = native_overrides(metadata["argv"])
    require(expected_overrides["--dataloader.num_prefetch_batches"] == "0", "Original A depth differs")
    expected_overrides["--dataloader.num_prefetch_batches"] = str(args.depth)
    expected_overrides["--training.steps"] = "20"
    if args.ep == 4:
        expected_overrides["--parallelism.expert_parallel_degree"] = "4"

    bridge = load_bridge()
    load_base = bridge._load_base_runner
    build_native_command = bridge._build_command_hook
    last_snapshot = {}

    def prepare_base():
        base = load_base()
        inside = base._inside
        configured_path = base._configured_path
        verify_inputs = base._verify_ready_inputs
        source_snapshot = base.source_snapshot
        verify_config = base._verify_resolved_config
        expectations = base._resolved_config_expectations
        make_environment = base._run_environment

        require(base.sha256_file(reference_config) == done["resolved_config_sha256"], "Reference config changed")
        base.DATASET_ROOT = dataset
        base.ROW_MANIFEST = dataset / "data-002-rows.jsonl"
        base.DATA_RECEIPT = data / "data-002.json"
        base.PREPARATION_STATUS = data / "data-002-preparation-status.json"
        base.LOADER_SUMMARY = data / "data-002-loader-summary.json"

        def allow_canonical_data(root, path, label):
            if label == "MDS data path" and path.resolve() == dataset:
                return dataset
            return inside(root, path, label)

        def resolve_canonical_data(value, *, base, label):
            path = configured_path(value, base=base, label=label)
            local_dataset = ROOT / "runs/production-data-abcd/artifacts/raw/data-002"
            if label == "canonical dataset stream local" and path == local_dataset:
                return dataset
            return path

        def check_inputs(root):
            input_receipt = verify_inputs(root)
            reference = metadata["input_data"]
            for key in ("dataset_sha256", "mds_index_sha256", "row_manifest_sha256", "tokenizer_file_sha256"):
                require(input_receipt[key] == reference[key], f"Input fingerprint differs at {key}")
            return input_receipt

        def freeze_source(root):
            result = source_snapshot(root)
            require(result["untracked_file_count"] == 0, "Unexpected untracked GIFT source")
            last_snapshot.update(result)
            return result

        def check_config(paths, parsed):
            digest = verify_config(paths, parsed)
            expected = base._load_yaml(reference_config)
            actual = base._load_yaml(paths["native"] / "config/resolved_config.yaml")
            expected["dump_folder"] = str(paths["native"].resolve())
            expected["training"]["steps"] = 20
            expected["parallelism"]["expert_parallel_degree"] = args.ep
            expected["dataloader"]["num_prefetch_batches"] = args.depth
            expected_stream = expected["dataloader"]["dataset"]["streams"][0]
            actual_stream = actual["dataloader"]["dataset"]["streams"][0]
            require(Path(actual_stream["local"]).resolve() == dataset, "Resolved stream differs")
            expected_stream["local"] = actual_stream["local"]
            differences = list(different_keys(expected, actual))
            require(not differences, f"Resolved config differs from original A: {differences}")
            return digest

        def clean_environment(paths):
            env = make_environment(paths)
            # The bridge will set its own selectors after this environment is returned.
            return {
                key: value for key, value in env.items()
                if (not key.startswith("GIFT_") or key == "GIFT_LOG_RANK")
                and key != "PYTORCH_CUDA_ALLOC_CONF"
            }

        base.RAW_ROOT.mkdir(parents=True, exist_ok=True)
        base._inside = allow_canonical_data
        base._configured_path = resolve_canonical_data
        base._verify_ready_inputs = check_inputs
        base.source_snapshot = freeze_source
        base._verify_resolved_config = check_config
        base._resolved_config_expectations = lambda parsed: expectations(parsed) | {
            "parallelism.expert_parallel_degree": args.ep,
        }
        base._run_environment = clean_environment
        return base

    def build_command(base):
        command = build_native_command(base)

        def checked_command(parsed, paths):
            argv = command(parsed, paths)
            if args.ep == 4:
                argv.extend(("--parallelism.expert_parallel_degree", "4"))
            differences = list(different_keys(expected_overrides, native_overrides(argv)))
            require(not differences, f"Native training overrides differ from original A: {differences}")
            experiment = base._load_yaml(paths["config"] / "experiment.yaml")
            expected_experiment = base._load_yaml(Path(metadata["argv"][metadata["argv"].index("--") + 1]))
            expected_experiment["imports"] = [str(base.BASE_CONFIG), str(base.DATA_FRAGMENT)]
            require(experiment == expected_experiment, "Generated experiment differs from original A")
            base._write_new_json(paths["attempt"] / "prefetch-launch.json", {
                "ep": args.ep,
                "depth": args.depth,
                "source_revision": REVISION,
                "source_sha256": last_snapshot["tracked_and_untracked_sha256"],
                "resolved_reference_sha256": done["resolved_config_sha256"],
                "adapter_sha256": base.sha256_file(Path(__file__)),
            })
            return argv

        return checked_command

    bridge._load_base_runner = prepare_base
    bridge._build_command_hook = build_command
    return bridge.main([
        "--gdn-backend", "fla", "--routing-metadata", "original",
        "--cross-entropy", "original", "--epoch-size", "102400", "--steps", "20",
        "--warmup", "10", "--window", "10", "--observe-compiler-cache",
        "--num-prefetch-batches", str(args.depth), "--run-id", args.run_id,
        "--output-root", str(RAW), "--timeout-seconds", "2400",
    ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ep", type=int, choices=(8, 4))
    parser.add_argument("depth", type=int, choices=(0, 32))
    parser.add_argument("run_id")
    raise SystemExit(run(parser.parse_args()))
