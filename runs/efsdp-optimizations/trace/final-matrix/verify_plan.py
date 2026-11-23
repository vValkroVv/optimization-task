"""Check all four profile plans in scratch directories without running training."""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
RUNNER = Path("runs/efsdp-optimizations/run.py")
BRIDGE = Path("runs/flash-qla")
NATIVE = Path("runs/overnight-streaming-perf")
BRIDGES = (
    BRIDGE / "run.py",
    BRIDGE / "measure.py",
    NATIVE / "run.py",
    NATIVE / "measure.py",
    NATIVE / "parse_results.py",
)
INPUTS = (
    Path("gift"),
    Path("runs/production-data-abcd"),
    NATIVE / "mds-dataloader.yaml",
)
FEATURES = ("bf16-cache", "direct-dw", "fp8-cache", "tail-retention")


def stage(target, source, copies, links):
    for path in (*copies, *links):
        staged = target / path
        staged.parent.mkdir(parents=True, exist_ok=True)
        if path in copies:
            shutil.copyfile(source / path, staged)
        else:
            staged.symlink_to(
                source / path, target_is_directory=(source / path).is_dir()
            )


def git_environment(clean_checkout):
    environment = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}
    git_dir = subprocess.check_output(
        ["git", "-C", clean_checkout, "rev-parse", "--absolute-git-dir"],
        env=environment,
        text=True,
    ).strip()
    # The staged control links to GIFT. Let the original-A guard query the actual
    # checkout through Git while every bridge loads from the isolated staging tree.
    return {**environment, "GIT_DIR": git_dir, "GIT_WORK_TREE": str(clean_checkout)}


def plan(candidate, control, ep, variant, original_git, *, profile=False):
    flags = ["--ep", str(ep)]
    if variant == "A":
        flags.extend(("--original-a-source", str(control)))
    else:
        for name in FEATURES:
            flags.extend(("--optimization", name))
        flags.extend(("--direct-dw-build", str(candidate / "planned-build")))
    if profile:
        flags.append("--trace-expert")
    result = json.loads(
        subprocess.check_output(
            [sys.executable, "-B", str(RUNNER), "plan", f"ep{ep}-{variant}", *flags],
            cwd=candidate,
            env=original_git if variant == "A" else None,
        )
    )
    assert result["bridge"] == {
        "gdn_backend": "fla",
        "capture_fixtures": None,
        "routing_metadata": "original",
        "cross_entropy": "original",
    }
    child = result["argv"]
    rank, delimiter = child.index("rank"), child.index("--")
    assert child[rank + 1 : rank + 4] == [
        str(control if variant == "A" else candidate),
        str(ep),
        "" if variant == "A" else ",".join(FEATURES),
    ]
    assert Path(child[delimiter + 1]).name == "experiment.yaml"
    pairs = child[delimiter + 2 :]
    overrides = dict(zip(pairs[::2], pairs[1::2], strict=True))
    overrides.pop("--dump_folder")
    return child[rank + 6 : delimiter], overrides


def assert_options(argv, options):
    for flag, expected in options.items():
        start = argv.index(flag) + 1
        assert argv[start : start + len(expected)] == expected


def verify_matrix(clean_checkout):
    original_git = git_environment(clean_checkout)
    profile_overrides = {
        "--training.steps": "11",
        "--profiler.enable_profiling": "true",
        "--profiler.enable_memory_snapshot": "false",
        "--profiler.profile_ranks": "[0, 4]",
        "--profiler.save_traces_folder": "profiling/traces",
        "--profiler.record_shapes": "false",
        "--profiler.profile_freq": "2",
        "--profiler.profiler_warmup": "1",
        "--profiler.profiler_active": "1",
        "--profiler.profiler_repeat": "1",
        "--profiler.profiler_skip_first": "9",
        "--profiler.profiler_skip_first_wait": "0",
    }
    arms = ((8, "A"), (8, "B"), (4, "A"), (4, "B"))
    with TemporaryDirectory(prefix="expert-profile-plan-") as scratch:
        temporary = Path(scratch).resolve()
        control, candidate = temporary / "a", temporary / "b"
        stage(control, clean_checkout, BRIDGES, INPUTS)
        helper = (HERE / "bounded_profile.py").relative_to(ROOT)
        stage(candidate, ROOT, (*BRIDGES, RUNNER, helper), INPUTS)

        def get_plan(arm, *, profile=False):
            return plan(candidate, control, *arm, original_git, profile=profile)

        defaults = {arm: get_plan(arm)[1] for arm in arms}
        for checkout in (control, candidate):
            subprocess.run(
                [
                    "git",
                    "apply",
                    f"--exclude={RUNNER}",
                    str(HERE.parent / "native-profile.patch"),
                ],
                cwd=checkout,
                check=True,
            )
        subprocess.run(
            ["git", "apply", str(HERE / "expert-profile.patch")],
            cwd=candidate,
            check=True,
        )
        for ep, variant in arms:
            arm = (ep, variant)
            ordinary, overrides = get_plan(arm)
            assert overrides == defaults[arm]
            assert_options(
                ordinary, {"--steps": ["20"], "--warmup": ["10"], "--window": ["10"]}
            )
            assert all(
                flag not in ordinary for flag in ("--profile-update", "--profile-ranks")
            )
            profiled, overrides = get_plan(arm, profile=True)
            assert overrides == {**defaults[arm], **profile_overrides}
            assert_options(
                profiled,
                {
                    "--steps": ["11"],
                    "--warmup": ["10"],
                    "--window": ["0"],
                    "--profile-update": ["11"],
                    "--profile-microbatches": ["3"],
                    "--profile-ranks": ["0", "4"],
                },
            )
            print(f"PASS EP{ep} {variant}: bounded profile and unchanged 10+10 plan")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original_a_source", type=Path)
    verify_matrix(parser.parse_args().original_a_source.resolve(strict=True))
