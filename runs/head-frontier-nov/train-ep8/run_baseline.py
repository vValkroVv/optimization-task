"""Run the original compiled EP8 control with the requested loss chunk count."""

import argparse
import hashlib
import sys
from pathlib import Path

import yaml

import run_arm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("chunks", type=int, choices=(2, 8, 32))
    parser.add_argument("run_id")
    parser.add_argument("--fresh-cache", action="store_true")
    args = parser.parse_args()

    checkout = Path("/workspace-SR008.nfs2/vkropotin/a10-cute-001")
    raw = checkout / "runs/production-data-abcd/artifacts/raw"
    run_arm.CHECKOUT = checkout
    bridge = run_arm.load_bridge()
    load_base = bridge._load_base_runner

    def prepare_base():
        base = load_base()
        write_config = base._write_experiment_config
        expectations = base._resolved_config_expectations
        environment = base._run_environment
        hashes = base._config_hashes

        def write_chunk_config(path):
            write_config(path)
            config = yaml.safe_load(path.read_text())
            config["loss"]["num_chunks"] = args.chunks
            path.write_text(yaml.safe_dump(config, sort_keys=False))

        def prepare_env(paths):
            env = environment(paths)
            assert env["PYTORCH_ALLOC_CONF"] == "expandable_segments:True"
            for name in (
                "PYTORCH_CUDA_ALLOC_CONF",
                "GIFT_STABLE_CE_FOLDED_BACKWARD",
                "GIFT_DIRECT_BF16_DHIDDEN",
                "GIFT_FUSED_HIDDEN_Z_LOSS",
                "GIFT_STAGED_HEAD_DW",
                "GIFT_EXPERIMENT_LM_HEAD_MATRIX",
                "GIFT_EXPERIMENT_LM_HEAD_MATRIX_CHUNKS",
                "LM_FINAL_FLASH_MEASURE",
            ):
                env.pop(name, None)
            return env

        base._write_experiment_config = write_chunk_config
        base._resolved_config_expectations = lambda parsed: {
            **expectations(parsed), "loss.num_chunks": args.chunks,
        }
        base._run_environment = prepare_env
        base._config_hashes = lambda: {
            **hashes(),
            "baseline_runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "baseline_bridge_loader_sha256": run_arm.sha(Path(run_arm.__file__)),
        }
        return base

    bridge._load_base_runner = prepare_base
    flags = [
        "--gdn-backend", "fla", "--routing-metadata", "original", "--cross-entropy", "original",
        "--run-id", args.run_id, "--steps", "20", "--warmup", "10", "--window", "10",
        "--num-prefetch-batches", "0", "--epoch-size", "102400", "--observe-compiler-cache",
        "--output-root", str(raw), "--timeout-seconds", "1800",
    ]
    if not args.fresh_cache:
        producer = "lh10-a2-001" if args.chunks == 2 else "lh10-a8-001"
        flags.extend(("--compiler-cache-root", str(raw / producer / "cache")))
    return bridge.main(flags)


if __name__ == "__main__":
    sys.exit(main())
