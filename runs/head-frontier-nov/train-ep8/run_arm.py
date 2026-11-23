"""Run the exact consuming CE on the existing 10+10 EP8 training recipe."""

import argparse
import hashlib
import importlib.util
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parent
CHECKOUT = Path("/workspace-SR008.nfs2/vkropotin/lm10-c1")
RAW = CHECKOUT / "runs/production-data-abcd/artifacts/raw"
CACHE = RAW / "lh10-n8-002/cache"
CHILD = ROOT / "final_child.py"
KERNEL = ROOT.parent / "fused-ce/centered.so"


def load_bridge():
    path = CHECKOUT / "runs/flash-qla/run.py"
    spec = importlib.util.spec_from_file_location("ep8_lm_head_bridge", path)
    bridge = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = bridge
    spec.loader.exec_module(bridge)
    return bridge


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("chunks", type=int, choices=(2, 8, 32))
    parser.add_argument("run_id")
    args = parser.parse_args()

    bridge = load_bridge()
    load_base = bridge._load_base_runner
    build_command = bridge._build_command_hook
    bridge._build_command_hook = lambda base: build_command(base, CHILD)

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
            for name in ("PYTORCH_CUDA_ALLOC_CONF", "GIFT_STAGED_HEAD_DW", "FUSED_CE_BINARY"):
                env.pop(name, None)
            env.update(
                GIFT_STABLE_CE_FOLDED_BACKWARD="1",
                GIFT_DIRECT_BF16_DHIDDEN="1",
                GIFT_FUSED_HIDDEN_Z_LOSS="1",
                GIFT_EXPERIMENT_LM_HEAD_MATRIX="new",
                GIFT_EXPERIMENT_LM_HEAD_MATRIX_CHUNKS=str(args.chunks),
                LM_FINAL_FLASH_MEASURE=str(bridge.BRIDGE_MEASURE_PATH.resolve()),
            )
            return env

        base._write_experiment_config = write_chunk_config
        base._resolved_config_expectations = lambda parsed: {
            **expectations(parsed),
            "loss.num_chunks": args.chunks,
        }
        base._run_environment = prepare_env
        base._config_hashes = lambda: {
            **hashes(),
            "final_ce_runner_sha256": sha(Path(__file__)),
            "final_ce_child_sha256": sha(CHILD),
            "final_ce_binary_sha256": sha(KERNEL),
            "final_ce_launcher_sha256": sha(KERNEL.with_name("fused.py")),
        }
        return base

    bridge._load_base_runner = prepare_base
    return bridge.main([
        "--gdn-backend", "fla", "--routing-metadata", "original", "--cross-entropy", "stable",
        "--run-id", args.run_id, "--steps", "20", "--warmup", "10", "--window", "10",
        "--num-prefetch-batches", "0", "--epoch-size", "102400", "--observe-compiler-cache",
        "--output-root", str(RAW), "--compiler-cache-root", str(CACHE),
        "--timeout-seconds", "1800",
    ])


if __name__ == "__main__":
    raise SystemExit(main())
