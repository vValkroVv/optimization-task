"""Runner plumbing for opt-in BF16 expert-cache measurement."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import measure
import run as runner
from test_measure import FakePart, _adapter


def _native_flag(native_args: list[str], name: str) -> str:
    flag = f"--{name}"
    return native_args[native_args.index(flag) + 1]


def _child(args, tmp_path: Path):
    paths = {
        "config": tmp_path / "config",
        "logs": tmp_path / "logs",
        "measurement": tmp_path / "measurement",
        "native": tmp_path / "native",
    }
    command = runner._command(args, paths)
    helper_start = command.index(str(runner.HELPER_DIR / "measure.py")) + 1
    helper_args, native_args = measure._parse_args(command[helper_start:])
    return command, helper_args, native_args


def test_exact_candidate_commands_and_opt_in_defaults(tmp_path: Path) -> None:
    default = runner._parse_args(["--run-id", "default"])
    assert default.gfsdp_reuse_bf16_expert_weights is False
    assert default.bf16_cache_counters is False
    default_overrides = runner._training_overrides(default, {"native": tmp_path / "default"})
    assert "--parallelism.gfsdp_reuse_bf16_expert_weights" not in default_overrides
    _, default_helper_args, default_native_args = _child(default, tmp_path / "default-child")
    assert default_helper_args.gfsdp_reuse_bf16_expert_weights is False
    assert "--parallelism.gfsdp_reuse_bf16_expert_weights" not in default_native_args

    common = [
        "--gfsdp-reuse-bf16-expert-weights",
        "--num-prefetch-batches",
        "32",
        "--epoch-size",
        "51200",
    ]
    timing = runner._parse_args(
        ["--run-id", "bf16-cache-tps", "--steps", "60", "--warmup", "20", "--window", "40", *common]
    )
    command, helper_args, native_args = _child(timing, tmp_path / "timing")
    assert (helper_args.steps, helper_args.warmup, helper_args.window) == (60, 20, 40)
    assert helper_args.gfsdp_reuse_bf16_expert_weights is True
    assert helper_args.bf16_cache_counters is False
    assert helper_args.capture_microbatches is False
    assert "--bf16-cache-counters" not in command
    assert "--capture-microbatches" not in command
    assert _native_flag(native_args, "parallelism.gfsdp_reuse_bf16_expert_weights") == "true"
    assert _native_flag(native_args, "dataloader.num_prefetch_batches") == "32"
    assert _native_flag(native_args, "dataloader.dataset.epoch_size") == "51200"
    assert _native_flag(native_args, "lr_scheduler.total_steps") == "65000"
    assert _native_flag(native_args, "lr_scheduler.warmup_steps") == "2000"
    assert _native_flag(native_args, "profiler.enable_profiling") == "false"

    diagnostic = runner._parse_args(
        [
            "--run-id",
            "bf16-cache-trace",
            "--steps",
            "21",
            "--warmup",
            "20",
            "--window",
            "0",
            "--profile-update",
            "21",
            "--profile-microbatches",
            "3",
            "--bf16-cache-counters",
            *common,
        ]
    )
    command, helper_args, native_args = _child(diagnostic, tmp_path / "diagnostic")
    assert (helper_args.steps, helper_args.warmup, helper_args.window) == (21, 20, 0)
    assert (helper_args.profile_update, helper_args.profile_microbatches) == (21, 3)
    assert helper_args.gfsdp_reuse_bf16_expert_weights is True
    assert helper_args.bf16_cache_counters is True
    assert helper_args.capture_microbatches is False
    assert command.count("--bf16-cache-counters") == 1
    assert "--capture-microbatches" not in command
    assert _native_flag(native_args, "profiler.profile_ranks") == "[0, 1]"
    assert _native_flag(native_args, "profiler.profiler_skip_first") == "19"
    assert _native_flag(native_args, "parallelism.gfsdp_reuse_bf16_expert_weights") == "true"


@pytest.mark.parametrize(
    "options",
    [
        ["--bf16-cache-counters"],
        ["--gfsdp-reuse-bf16-expert-weights", "--bf16-cache-counters"],
        [
            "--steps",
            "60",
            "--warmup",
            "20",
            "--window",
            "40",
            "--gfsdp-reuse-bf16-expert-weights",
            "--bf16-cache-counters",
        ],
    ],
)
def test_counter_capture_requires_a_feature_enabled_diagnostic(options: list[str]) -> None:
    with pytest.raises(SystemExit):
        runner._parse_args(["--run-id", "invalid", *options])


class CacheStatsPart(FakePart):
    def __init__(self, parameters) -> None:
        super().__init__(parameters)
        self.debug_calls = 0
        self.stats_calls = 0
        self.debug_enabled = False
        self.active = False
        self.generation = 0
        self.refreshes = 0
        self.hits = 0
        self.direct_uses = 0
        self.invalidations = 0

    def set_bf16_cache_debug(self, enabled: bool, recurse: bool = True) -> None:
        assert recurse is True
        self.debug_calls += 1
        self.debug_enabled = enabled

    def consume_cached_weight(self) -> None:
        self.active = True
        if self.generation == 0:
            self.generation = 1
            self.refreshes += 1
        self.hits += 1
        self.direct_uses += 1

    def bf16_cache_stats(self, recurse: bool = True) -> tuple[dict, ...]:
        assert recurse is True
        self.stats_calls += 1
        return (
            {
                "fqn": "layers.0.experts.cached",
                "enabled": self.active,
                "generation": self.generation,
                "refreshes": self.refreshes,
                "hits": self.hits,
                "direct_uses": self.direct_uses,
                "invalidations": self.invalidations,
            },
            {
                "fqn": "layers.0.experts.fallback",
                "enabled": False,
                "generation": 0,
                "refreshes": 0,
                "hits": 0,
                "direct_uses": 0,
                "invalidations": 0,
            },
        )


def _attach_cache_stats(
    trainer_class, monkeypatch: pytest.MonkeyPatch
) -> list[CacheStatsPart]:
    original_init = trainer_class.__init__
    original_forward_backward = trainer_class.forward_backward_step
    created: list[CacheStatsPart] = []

    def init(trainer) -> None:
        original_init(trainer)
        cache_part = CacheStatsPart(trainer.model_parts[0].parameters())
        trainer.model_parts = [cache_part]
        created.append(cache_part)

    def forward_backward(trainer, *args, **kwargs):
        trainer.model_parts[0].consume_cached_weight()
        return original_forward_backward(trainer, *args, **kwargs)

    monkeypatch.setattr(trainer_class, "__init__", init)
    monkeypatch.setattr(trainer_class, "forward_backward_step", forward_backward)
    return created


def test_diagnostic_counters_sample_update_boundaries_without_microbatch_hooks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "8")
    trainer_class, adapter, _, _ = _adapter(
        tmp_path,
        steps=2,
        warmup=0,
        window=0,
        monitor_interval=0,
        capture_microbatches=False,
    )
    created = _attach_cache_stats(trainer_class, monkeypatch)
    adapter.window_options["gfsdp_reuse_bf16_expert_weights"] = True
    adapter.window_options["bf16_cache_counters"] = True

    with adapter:
        trainer_class().train()

    cache_part = created[0]
    record = json.loads((tmp_path / "measurement/rank-00.json").read_text())
    cache = record["bf16_expert_cache"]
    assert cache["feature_configured_enabled"] is True
    assert cache["counter_capture_enabled"] is True
    assert cache_part.debug_calls == 1 and cache_part.debug_enabled
    assert cache_part.stats_calls == 3
    assert record["capture_microbatches"] is False
    assert record["microbatch_calls_by_step"] == {}
    snapshots = cache["snapshots"]
    assert [(row["update"], row["boundary"]) for row in snapshots] == [
        (0, "before_train"),
        (1, "after_optimizer_update"),
        (2, "after_optimizer_update"),
    ]
    assert [row["kind"] for row in snapshots] == ["counters"] * 3
    cached_rows = [
        next(group for group in snapshot["groups"] if group["enabled"])
        for snapshot in snapshots[1:]
    ]
    assert [row["hits"] for row in cached_rows] == [32, 64]
    assert [row["refreshes"] for row in cached_rows] == [1, 1]
    assert [row["direct_uses"] for row in cached_rows] == [32, 64]
    assert all(
        any(group["fqn"].endswith("fallback") and not group["enabled"] for group in snapshot["groups"])
        for snapshot in snapshots
    )


def test_timing_feature_records_one_activation_snapshot_with_counters_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "8")
    trainer_class, adapter, _, _ = _adapter(
        tmp_path,
        steps=2,
        warmup=1,
        window=1,
        monitor_interval=0,
        capture_microbatches=False,
    )
    created = _attach_cache_stats(trainer_class, monkeypatch)
    adapter.window_options["gfsdp_reuse_bf16_expert_weights"] = True

    with adapter:
        trainer_class().train()

    cache_part = created[0]
    record = json.loads((tmp_path / "measurement/rank-00.json").read_text())
    cache = record["bf16_expert_cache"]
    assert cache["feature_configured_enabled"] is True
    assert cache["counter_capture_enabled"] is False
    assert cache_part.debug_calls == 0 and cache_part.debug_enabled is False
    assert cache_part.stats_calls == 1
    assert record["capture_microbatches"] is False
    assert record["microbatch_calls_by_step"] == {}
    assert cache["snapshots"][0]["kind"] == "activation_metadata"
    assert cache["snapshots"][0]["boundary"] == "after_train"
    assert cache["snapshots"][0]["update"] == 2
    assert any(group["enabled"] for group in cache["snapshots"][0]["groups"])
