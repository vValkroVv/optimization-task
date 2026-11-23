"""Exercise the proposed runner hooks in a temporary tree without training."""

import json
import os
import runpy
import shutil
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUNNER = Path("runs/efsdp-optimizations/run.py")
STAGED_FILES = (
    RUNNER,
    Path("runs/overnight-streaming-perf/run.py"),
    Path("runs/overnight-streaming-perf/measure.py"),
    Path("runs/overnight-streaming-perf/parse_results.py"),
    Path("runs/flash-qla/run.py"),
    Path("runs/flash-qla/measure.py"),
)
LINKED_INPUTS = (
    Path("gift"),
    Path("runs/production-data-abcd"),
    Path("runs/efsdp-optimizations/trace"),
    Path("runs/overnight-streaming-perf/mds-dataloader.yaml"),
)


def plan(stage, flags):
    command = [sys.executable, "-B", str(RUNNER), "plan"]
    return json.loads(subprocess.check_output([*command, *flags], cwd=stage))["argv"]


def overrides(argv):
    items = argv[argv.index("--") + 2 :]
    result = dict(zip(items[::2], items[1::2], strict=True))
    result.pop("--dump_folder")
    return result


def module_stubs(attributes):
    modules = {}
    for name, values in attributes.items():
        module = modules.setdefault(name, ModuleType(name))
        module.__dict__.update(values)
        while "." in name:
            parent, _, leaf = name.rpartition(".")
            module = modules.setdefault(parent, ModuleType(parent))
            setattr(module, leaf, modules[name])
            module.__path__ = []
            name = parent
    return modules


def verify_native_constructor(stage):
    entry = runpy.run_path(str(stage / RUNNER))["run_rank"]
    load = entry.__globals__["load_bridge"]
    trace = load(HERE / "bounded_ep4.py", "_trace_fixture")
    checks = []
    trace.verify_shard_group = lambda measurement: checks.append(
        (measurement.rank, measurement.profile_update, measurement.profile_microbatches)
    )
    raw = stage / "runs/overnight-streaming-perf/artifacts/raw"
    local_tmp = stage / ".tmp/run-fixture0"
    local_tmp.mkdir(parents=True)

    for rank, profile in ((0, True), (4, True), (1, True), (0, False)):
        traced = profile and rank in trace.RANKS
        steps = trace.UPDATE if profile else 1
        run_id = f"constructor-{rank}-{profile}"
        attempt = raw / run_id
        measurement = attempt / "measurement"
        measurement.mkdir(parents=True)
        triton_cache = attempt / "cache/triton"
        (triton_cache / f"rank_{rank}").mkdir(parents=True)
        deep_gemm_cache = attempt / "cache/deepgemm"
        deep_gemm_cache.mkdir()
        drains = []
        exports = []
        sync_scope = []

        class Handle:
            def __exit__(self, *_):
                exports.append(self.step_num)

        class Profiler:
            def build_torch_profiler(self):
                return Handle() if traced else None

            def step(self):
                pass

        class Optimizer:
            def step(self):
                pass

        class Trainer:
            model_parts = []
            metrics_processor = None
            optimizers = Optimizer()

            def forward_backward_step(self):
                pass

            def train_step(self):
                for _ in range(32):
                    self.forward_backward_step()
                self.optimizers.step()

            def train(self):
                profiler = Profiler()
                profiler.torch_profiler = profiler.build_torch_profiler()
                for self.step in range(1, steps + 1):
                    self.train_step()
                    profiler.step()

        def native_main(argv):
            assert argv == ["fixture.yaml"]
            sync_scope.append("train")
            Trainer().train()

        modules = module_stubs(
            {
                "torch": {
                    "cuda": SimpleNamespace(synchronize=lambda: drains.append(1)),
                    "no_grad": nullcontext,
                    "profiler": SimpleNamespace(record_function=nullcontext),
                },
                "torch._C._profiler": {"_set_cuda_sync_enabled_val": sync_scope.append},
                "torchtitan.components.metrics": {"MetricsProcessor": object},
                "torchtitan.tools.profiler": {"Profiler": Profiler},
                "torchtitan.tools.utils": {
                    "device_module": SimpleNamespace(memory_stats=dict)
                },
                "torchtitan.train": {"main": native_main},
                "torchtitan.trainer": {
                    "Trainer": Trainer,
                    "OptimizersContainer": Optimizer,
                    "dist_utils": SimpleNamespace(dist_sum=lambda: None),
                    "setup_deep_gemm_cache": lambda: None,
                },
                "torchtitan.ops.warmup": {
                    "deep_gemm_fast_warmup": SimpleNamespace(
                        _rank_cache_dir=lambda: None
                    ),
                },
                "torchtitan.datasets.streaming.core.prefetch": {
                    "PrefetchIterator": type(
                        "PrefetchIterator", (), {"__next__": lambda _: None}
                    ),
                },
                "torchtitan.datasets.streaming.core.dataloader": {
                    "_TitanStreamingDataLoader": type(
                        "Loader", (), {"begin_prefetch_epoch": lambda _: None}
                    ),
                },
                "streaming_filelock": {
                    "StreamingFileLockBindings": lambda: nullcontext(
                        SimpleNamespace(startup_metadata=dict)
                    )
                },
            }
        )

        def load_fixture(path, name):
            if path.name == "bounded_ep4.py":
                return trace
            flash = load(path, name)

            def bootstrap(argv, adapter, train):
                assert flash._measurement_dir(adapter) == measurement
                assert adapter.options["profile_ranks"] == (
                    list(trace.RANKS) if profile else [0, 1]
                )
                return train(argv, adapter)

            flash._bootstrap_and_run = bootstrap
            return flash

        options = [
            "--measurement-dir",
            str(measurement),
            "--run-id",
            run_id,
            "--steps",
            str(steps),
            "--warmup",
            str(trace.WARMUP if profile else 0),
            "--window",
            "0",
            *(trace.native_args() if profile else ()),
            "--",
            "fixture.yaml",
        ]
        env = {
            "RANK": str(rank),
            "WORLD_SIZE": "8",
            "TMPDIR": str(local_tmp),
            "TMP": str(local_tmp),
            "TEMP": str(local_tmp),
            "TASK_LOCAL_TMPDIR": str(local_tmp),
            "TRITON_CACHE_DIR": str(triton_cache),
            "TASK_DEEP_GEMM_RANK_CACHE_ROOT": str(deep_gemm_cache),
            "TASK_COMPILER_CACHE_ROOT": "",
            "ABCD_COMPONENT_FIXTURE_DIR": "",
        }
        with (
            patch.dict(
                entry.__globals__,
                {"load_bridge": load_fixture, "capture_activation": lambda *_: None},
            ),
            patch.dict(sys.modules, modules),
            patch.dict(os.environ, env),
        ):
            assert entry(stage, 4, set(), "", 116, options) == 0

        receipt = json.loads((measurement / f"rank-{rank:02d}.json").read_text())
        assert receipt["training_status"] == "complete"
        assert receipt["completed_steps"] == steps
        if profile:
            capture = receipt["profile_capture"]
            assert capture["target_update"] == trace.UPDATE
            assert capture["microbatch_calls"] == 32
            assert capture["train_step_calls"] == 1
            assert capture["optimizer_step_calls"] == 1
            assert capture["native_profiler_created"] is traced
            assert capture["collected_microbatches"] == (3 if traced else 0)
            assert capture["collection_stopped"] is traced
            assert capture["early_stop_cuda_drains"] == int(traced)
            assert capture["export_update"] == (trace.UPDATE if traced else None)
        else:
            assert "profile_capture" not in receipt
        assert len(drains) == int(traced)
        assert exports == ([trace.UPDATE] if traced else [])
        assert sync_scope == ([True, "train", False] if traced else ["train"])
    assert checks == [(rank, trace.UPDATE, 3) for rank in (0, 4, 1)]


def main():
    with TemporaryDirectory(prefix="offline-", dir=HERE) as temp:
        stage = Path(temp)
        for relative in (*STAGED_FILES, *LINKED_INPUTS):
            source = ROOT / relative
            target = stage / relative
            target.parent.mkdir(exist_ok=True, parents=True)
            if relative in STAGED_FILES:
                shutil.copyfile(source, target)
            else:
                target.symlink_to(source, target_is_directory=source.is_dir())
        subprocess.run(
            ["git", "apply", str(HERE / "native-profile.patch")], cwd=stage, check=True
        )
        flags = [
            "--ep",
            "4",
            "--optimization",
            "bf16-cache",
            "--optimization",
            "fp8-cache",
            "--optimization",
            "direct-dw",
            "--optimization",
            "tail-retention",
            "--direct-dw-build",
            str(stage / "planned-build"),
        ]
        timed = plan(stage, ["measure-ep4", *flags])
        profile = plan(stage, ["trace-ep4", "--trace-expert-pair", *flags])
        expected = {
            **overrides(timed),
            "--training.steps": "21",
            "--profiler.enable_profiling": "true",
            "--profiler.enable_memory_snapshot": "false",
            "--profiler.profile_ranks": "[0, 4]",
            "--profiler.save_traces_folder": "profiling/traces",
            "--profiler.record_shapes": "false",
            "--profiler.profile_freq": "2",
            "--profiler.profiler_warmup": "1",
            "--profiler.profiler_active": "1",
            "--profiler.profiler_repeat": "1",
            "--profiler.profiler_skip_first": "19",
            "--profiler.profiler_skip_first_wait": "0",
        }
        assert overrides(profile) == expected
        child = profile[profile.index("rank") + 6 : profile.index("--")]
        for name, expected_values in {
            "--steps": ["21"],
            "--warmup": ["20"],
            "--window": ["0"],
            "--profile-update": ["21"],
            "--profile-microbatches": ["3"],
            "--profile-ranks": ["0", "4"],
        }.items():
            start = child.index(name) + 1
            assert child[start : start + len(expected_values)] == expected_values
        timed_child = timed[timed.index("rank") + 6 : timed.index("--")]
        assert timed_child[timed_child.index("--steps") + 1] == "20"
        assert timed_child[timed_child.index("--warmup") + 1] == "10"
        assert timed_child[timed_child.index("--window") + 1] == "10"
        assert "--profile-update" not in timed_child
        wrong_topology = subprocess.run(
            [
                sys.executable,
                "-B",
                str(RUNNER),
                "plan",
                "wrong-ep",
                "--ep",
                "8",
                "--trace-expert-pair",
            ],
            cwd=stage,
            capture_output=True,
            text=True,
            check=False,
        )
        assert (
            wrong_topology.returncode == 2 and "requires EP4" in wrong_topology.stderr
        )
        verify_native_constructor(stage)
    print("PASS: original A EP4 plan, rank pair 0/4, update 21, three microbatches")
    print("PASS: same training overrides apart from update count and profiler controls")
    print("PASS: 10+10 benchmark plan is unchanged and the trace rejects EP8")
    print(
        "PASS: CUDA sync activity is enabled before train and reset after it only on traced ranks"
    )
    print(
        "PASS: FlashQLA and native measurement construct rank-scoped receipts through the rank entry point"
    )


if __name__ == "__main__":
    main()
