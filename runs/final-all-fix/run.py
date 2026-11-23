"""Start one native eight-H100 final-all-fix recipe or its one-update activation probe."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
HELPERS = ROOT / "runs/overnight-streaming-perf"
SWITCHES = (
    "GIFT_GDN_DIRECT_FLA_BACKWARD",
    "GIFT_GDN_FUSE_SHORT_CONV_BWD",
    "GIFT_GDN_FUSE_OUTPUT_NORM_FP8",
    "GIFT_MLA_FUSED_ROPE",
    "GIFT_GDN_DIRECT_OUTPUT_NORM_BWD",
    "GIFT_CUTE_NORM_FUSE_RESIDUAL",
    "GIFT_FP8_QUANTIZE_WEIGHT_TAIL",
    "GIFT_CUTE_NORM_SAC_STATS_ONLY",
    "GIFT_CUTE_NORM_BWD_SKIP",
    "GIFT_CUTE_NORM_PAIR",
    "GIFT_ROUTED_DW_BF16_OVERWRITE",
    "GIFT_STABLE_CE_FOLDED_BACKWARD",
    "GIFT_DIRECT_BF16_DHIDDEN",
    "GIFT_FUSED_HIDDEN_Z_LOSS",
)


def sha256(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def source_identity():
    """Identify tracked and new recipe bytes relative to the checkout commit."""

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT)

    scope = ("--", "gift", "runs/final-all-fix")
    names = set(
        git("diff", "--name-only", "-z", "HEAD", *scope).split(b"\0")
        + git("ls-files", "--others", "--exclude-standard", "-z", *scope).split(b"\0")
    )
    overlay = {}
    for raw in sorted(names - {b""}):
        name = os.fsdecode(raw)
        path = ROOT / name
        overlay[name] = (
            {"sha256": sha256(path), "mode": path.stat().st_mode & 0o777}
            if path.is_file()
            else None
        )
    return {"commit": git("rev-parse", "HEAD").decode().strip(), "overlay": overlay}


def main():
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ep", type=int, choices=(4, 8), required=True)
    parser.add_argument("--loss-chunks", type=int, choices=(2, 8, 32), default=2)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--direct-dw-build", type=Path, required=True)
    parser.add_argument("--ce-library", type=Path, required=True)
    parser.add_argument("--overwrite-build", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    args = parser.parse_args()
    if args.steps < 1 or (args.probe and args.steps != 1):
        parser.error("the activation probe uses one full optimizer update")
    sys.path.insert(0, str(HELPERS))
    import importlib.util

    spec = importlib.util.spec_from_file_location("original_a_run", HELPERS / "run.py")
    base = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = base
    spec.loader.exec_module(base)
    if not base.RUN_ID_RE.fullmatch(args.run_id) or args.run_id in (".", ".."):
        parser.error("invalid run ID")

    devices = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,name,uuid", "--format=csv,noheader"],
        text=True,
    ).splitlines()
    active = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip()
    if len(devices) != 8 or not all("H100" in line for line in devices) or active:
        raise RuntimeError(f"Need eight idle H100s: {devices}, active PIDs={active}")

    files = {
        args.ce_library: "41357457bb454c721ee72b5b9e4151cff7056f5e3b473a1241b39c4161925a11",
        args.direct_dw_build
        / "manifest.json": "cc287ca421603f25e1fbe80b380a1da76501141df375d11f0200d59c038f4f12",
        args.overwrite_build
        / "manifest.json": "951fd24ac5e35d81c918c93f2e7abb45a24f76acd21d2c062c3cdb2bf13aec4a",
    }
    data_receipt = json.loads(
        (ROOT / "runs/production-data-abcd/data-002.json").read_text()
    )
    dataset = data_receipt["dataset"]
    fragment = yaml.safe_load((HELPERS / "mds-dataloader.yaml").read_text())
    data_root = Path(fragment["dataloader"]["dataset"]["streams"][0]["local"])
    for item in dataset["files"]:
        files[data_root / item["path"]] = item["sha256"]
    tokenizer = data_receipt["tokenizer"]
    for name, expected in tokenizer["files"].items():
        files[Path(tokenizer["path"]) / name] = expected
    files[data_root / "data-002-rows.jsonl"] = dataset["row_manifest"]["sha256"]
    for path, expected in files.items():
        if sha256(path) != expected:
            raise RuntimeError(f"Unexpected artifact or input: {path}")

    paths = base._make_run_dirs(args.run_id, HERE / "artifacts/raw")
    args.scratch_root.mkdir(parents=True, exist_ok=True)
    # Multiprocessing uses AF_UNIX sockets below this directory.
    scratch = Path(tempfile.mkdtemp(prefix="ff-", dir=args.scratch_root.resolve()))
    if len(os.fsencode(scratch)) > 78:
        raise ValueError(f"Use a shorter personal scratch root: {scratch}")
    paths["tmp_dir"] = scratch
    env = base._run_environment(paths)
    for key in tuple(env):
        if key.startswith(("GIFT_", "TASK_FLASHQLA_", "FUSED_CE_")):
            env.pop(key)
    env.update(dict.fromkeys(SWITCHES, "1"))
    env.update(
        GIFT_LOG_RANK="all",
        TMPDIR=str(scratch),
        TMP=str(scratch),
        TEMP=str(scratch),
        PYTHONPATH=os.pathsep.join(
            [
                str(ROOT / "gift"),
                str(ROOT / "gift/contrib/flash-linear-attention"),
                str(HELPERS),
            ]
        ),
        GIFT_EXACT_FOLDED_CE_LIBRARY=str(args.ce_library.resolve()),
        GIFT_ROUTED_DW_BF16_OVERWRITE_BUILD=str(args.overwrite_build.resolve()),
        GIFT_EXPERIMENT_LM_HEAD_MATRIX="new",
        GIFT_EXPERIMENT_LM_HEAD_MATRIX_CHUNKS=str(args.loss_chunks),
    )
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc-per-node=8",
        "--max-restarts=0",
        "--redirects=3",
        "--log-dir",
        str(paths["logs"]),
        str(HERE / "rank.py"),
        "--ep",
        str(args.ep),
        "--output",
        str(paths["attempt"]),
        "--direct-dw-build",
        str(args.direct_dw_build.resolve()),
    ]
    if args.probe:
        cmd.append("--probe")
    cmd.extend(
        [
            "--",
            str(HERE / f"ep{args.ep}.yaml"),
            "--training.steps",
            str(args.steps),
            "--loss.num_chunks",
            str(args.loss_chunks),
            "--validator.enable",
            "false",
            "--dump_folder",
            str(paths["native"]),
        ]
    )
    (paths["attempt"] / "launch.json").write_text(
        json.dumps(
            {
                "argv": cmd,
                "devices": devices,
                "dataset_sha256": dataset["dataset_sha256"],
                "switches": {key: env[key] for key in env if key.startswith("GIFT_")},
                "source": source_identity(),
            },
            indent=2,
        )
        + "\n"
    )
    result = subprocess.run(cmd, cwd=ROOT / "gift", env=env)
    (paths["attempt"] / "torchrun.exit").write_text(f"{result.returncode}\n")
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
