from __future__ import annotations

import json
import sys
from copy import deepcopy
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import compiler_stats
import measure
import parse_results
import run as runner

LOCAL_PHYSICAL_PER_UPDATE = 262_144
GLOBAL_SUPERVISED_PER_UPDATE = 2_096_640


class FakeParameter:
    def __init__(self, *, requires_grad: bool, local_numel: int, dtype: str) -> None:
        self.requires_grad = requires_grad
        self._local_numel = local_numel
        self.dtype = dtype

    def to_local(self) -> FakeParameter:
        return self

    def numel(self) -> int:
        return self._local_numel


class FakePart:
    def __init__(self, parameters: list[FakeParameter]) -> None:
        self._parameters = parameters

    def parameters(self) -> list[FakeParameter]:
        return self._parameters


class FakeScalar:
    def __init__(self) -> None:
        self.value = 0

    def item(self) -> int:
        return self.value


class FakeMetricsProcessor:
    def __init__(self) -> None:
        self.ntokens_since_last_log = 0

    def interval_metrics(self) -> int:
        value = self.ntokens_since_last_log
        self.ntokens_since_last_log = 0
        return value


class FakeProfiler:
    def step(self) -> None:
        return None


class FakeRuntime:
    def __init__(self) -> None:
        self.stats = self._reset_stats(0, 0)

    @staticmethod
    def _reset_stats(retries: int, ooms: int) -> dict[str, int]:
        return {
            "allocated_bytes.all.peak": 100,
            "active_bytes.all.peak": 200,
            "reserved_bytes.all.peak": 300,
            "num_alloc_retries": retries,
            "num_ooms": ooms,
        }

    def memory_stats(self) -> dict[str, int]:
        return dict(self.stats)

    def reset_peak_stats(self) -> None:
        retries = self.stats["num_alloc_retries"]
        ooms = self.stats["num_ooms"]
        self.stats = self._reset_stats(retries, ooms)


def _fake_classes(stop_after: int):
    runtime = FakeRuntime()
    metrics_module = SimpleNamespace(reset_peak_stats=runtime.reset_peak_stats)

    class FakeTrainer:
        def __init__(self) -> None:
            self.step = 0
            self.metrics_processor = FakeMetricsProcessor()
            self.ntokens_seen = FakeScalar()
            self.model_parts = [
                FakePart(
                    [
                        FakeParameter(
                            requires_grad=True,
                            local_numel=17,
                            dtype="float32",
                        ),
                        FakeParameter(
                            requires_grad=False,
                            local_numel=5,
                            dtype="bfloat16",
                        ),
                    ]
                )
            ]
            self.profiler = FakeProfiler()

        def forward_backward_step(self) -> None:
            return None

        def train(self) -> None:
            for step in range(1, stop_after + 1):
                self.step = step
                self.metrics_processor.ntokens_since_last_log += LOCAL_PHYSICAL_PER_UPDATE
                self.ntokens_seen.value += GLOBAL_SUPERVISED_PER_UPDATE
                self.metrics_processor.interval_metrics()
                self.runtime.stats = {
                    "allocated_bytes.all.peak": 1_000 + step,
                    "active_bytes.all.peak": 2_000 + step,
                    "reserved_bytes.all.peak": 3_000 + step,
                    "num_alloc_retries": max(0, step - 20),
                    "num_ooms": int(step >= 24),
                }
                self.metrics_module.reset_peak_stats()
                for _ in range(32):
                    self.forward_backward_step()
                self.profiler.step()

    FakeTrainer.runtime = runtime
    FakeTrainer.metrics_module = metrics_module
    return FakeTrainer, FakeProfiler, FakeMetricsProcessor, metrics_module, runtime


def _adapter(
    tmp_path: Path,
    *,
    steps: int,
    warmup: int,
    window: int,
    monitor_interval: int,
    capture_microbatches: bool,
):
    trainer_class, profiler_class, metrics_class, metrics_module, runtime = _fake_classes(steps)
    sync_calls: list[int] = []
    adapter = measure.NativeAdapter(
        trainer_class=trainer_class,
        profiler_class=profiler_class,
        metrics_class=metrics_class,
        metrics_module=metrics_module,
        options={
            "run_id": "unit-test",
            "output_dir": tmp_path / "measurement",
            "expected_steps": steps,
            "warmup_updates": warmup,
            "window_updates": window,
            "monitor_interval": monitor_interval,
            "capture_microbatches": capture_microbatches,
        },
        cuda_synchronize=lambda: sync_calls.append(1),
        memory_stats=runtime.memory_stats,
    )
    return trainer_class, adapter, runtime, sync_calls


def test_parameter_census_uses_ragged_local_guard_and_restores_grad_mode() -> None:
    import torch
    from torchtitan.distributed.gfsdp._tensor import RaggedShardDTensor

    class RaggedParameterProbe:
        # Bind the real wrapper method to metadata-only probes. Constructing a
        # distributed mesh for this guard regression would add no coverage.
        to_local = RaggedShardDTensor.to_local

        def __init__(self, *, local_tensor: torch.Tensor, requires_grad: bool) -> None:
            self._local_tensor = local_tensor
            self.requires_grad = requires_grad
            self.dtype = local_tensor.dtype

    class BrokenRaggedParameterProbe:
        to_local = RaggedShardDTensor.to_local
        requires_grad = True
        dtype = torch.float32

        @property
        def _local_tensor(self) -> torch.Tensor:
            assert not torch.is_grad_enabled()
            raise RuntimeError("sentinel local-shard failure")

    previous_grad_mode = torch.is_grad_enabled()
    torch.set_grad_enabled(True)
    try:
        trainable = RaggedParameterProbe(
            local_tensor=torch.empty(7, dtype=torch.float32), requires_grad=True
        )
        frozen = RaggedParameterProbe(
            local_tensor=torch.empty(5, dtype=torch.bfloat16), requires_grad=False
        )

        # This reproduces the old census's direct to_local() call and failure.
        def old_census(parameters: list[RaggedParameterProbe]) -> list[int]:
            return [int(parameter.to_local().numel()) for parameter in parameters]

        with pytest.raises(NotImplementedError):
            old_census([trainable])
        assert torch.is_grad_enabled()

        census = measure.parameter_census([FakePart([trainable, frozen])])
        assert census["trainable"]["local_numel"] == 7
        assert census["trainable"]["by_dtype"] == {"torch.float32": 7}
        assert census["frozen"]["local_numel"] == 5
        assert census["frozen"]["by_dtype"] == {"torch.bfloat16": 5}
        assert torch.is_grad_enabled()

        with pytest.raises(RuntimeError, match="sentinel local-shard failure"):
            measure.parameter_census([FakePart([BrokenRaggedParameterProbe()])])
        assert torch.is_grad_enabled()
    finally:
        torch.set_grad_enabled(previous_grad_mode)


def test_smoke_mode_captures_native_updates_without_adapter_syncs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "8")
    trainer_class, adapter, _, sync_calls = _adapter(
        tmp_path,
        steps=10,
        warmup=0,
        window=0,
        monitor_interval=5,
        capture_microbatches=True,
    )

    with adapter:
        trainer_class().train()

    record = json.loads((tmp_path / "measurement/rank-00.json").read_text())
    assert record["training_status"] == "complete"
    assert record["measurement_status"] == "disabled"
    assert record["measurement"] is None
    assert len(record["cadence"]) == 10
    assert record["microbatch_calls_by_step"] == {str(step): 32 for step in range(1, 11)}
    assert "profile_capture" not in record
    assert sync_calls == []


def test_full_window_uses_two_boundaries_and_tracks_native_resets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "8")
    trainer_class, adapter, _, sync_calls = _adapter(
        tmp_path,
        steps=70,
        warmup=20,
        window=50,
        monitor_interval=50,
        capture_microbatches=False,
    )

    with adapter:
        trainer_class().train()

    record = json.loads((tmp_path / "measurement/rank-00.json").read_text())
    window = record["measurement"]
    assert record["training_status"] == "complete"
    assert record["measurement_status"] == "complete"
    assert sync_calls == [1, 1]
    assert window["cuda_synchronize_calls"] == 2
    assert window["start_after_update"] == 20
    assert window["end_after_update"] == 70
    assert window["monitoring_updates"] == [50]
    assert window["physical_input_tokens_local"] == 50 * LOCAL_PHYSICAL_PER_UPDATE
    assert window["supervised_tokens_global"] == 50 * GLOBAL_SUPERVISED_PER_UPDATE
    assert window["memory"]["peak_bytes"] == {
        "allocated_bytes.all.peak": 1_070,
        "active_bytes.all.peak": 2_070,
        "reserved_bytes.all.peak": 3_070,
    }
    assert window["memory"]["allocator_counter_deltas"] == {
        "num_alloc_retries": 50,
        "num_ooms": 1,
    }
    assert record["parameter_census"]["trainable"]["local_numel"] == 17
    assert record["parameter_census"]["frozen"]["local_numel"] == 5


def test_incomplete_window_is_written_as_incomplete_and_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "8")
    short_class, short_adapter, _, short_sync_calls = _adapter(
        tmp_path / "short",
        steps=70,
        warmup=20,
        window=50,
        monitor_interval=50,
        capture_microbatches=False,
    )
    short_class.train = lambda trainer: _run_short(trainer, 40)
    with short_adapter, pytest.raises(measure.IncompleteWindowError):
        short_class().train()

    record = json.loads((tmp_path / "short/measurement/rank-00.json").read_text())
    assert record["training_status"] == "incomplete"
    assert record["measurement_status"] == "incomplete"
    assert record["measurement"] is None
    assert short_sync_calls == [1]


def test_real_deep_gemm_setup_and_share_stay_in_task_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import torchtitan.ops.warmup.deep_gemm_fast_warmup as warmup
    import torchtitan.trainer as trainer_module

    rank_root = tmp_path / "rank-cache"
    rank_root.mkdir()
    shared_root = tmp_path / "shared-cache"
    monkeypatch.setenv("DG_JIT_SHARED_CACHE_DIR", str(shared_root))
    monkeypatch.delenv("PROPAGATE_DG_CACHE", raising=False)
    monkeypatch.delenv("PROPAGATE_DG_CACHE_DIR", raising=False)

    calls: list[tuple[object, int, int, int]] = []

    def fake_warmup(model_config, ep_degree: int, num_sms: int, local_rank: int) -> None:
        calls.append((model_config, ep_degree, num_sms, local_rank))
        cache = Path(warmup._rank_cache_dir(local_rank)) / "cache"
        cache.mkdir(parents=True, exist_ok=True)
        sentinel = cache / f"kernel.rank{local_rank}"
        sentinel.mkdir()
        (sentinel / "owner.txt").write_text(str(local_rank), encoding="utf-8")

    def forbidden_cache_manager(local_rank: int) -> None:
        raise AssertionError(f"hard-coded /temp manager was called for local rank {local_rank}")

    monkeypatch.setattr(warmup, "_warmup_node", fake_warmup)
    monkeypatch.setattr(warmup, "_deepgemm_cache_manager", forbidden_cache_manager)
    original_rank_cache_dir = warmup._rank_cache_dir
    original_setup = trainer_module.setup_deep_gemm_cache
    redirect = measure.DeepGemmCacheRedirect(
        warmup, trainer_module, rank_root, task_root=tmp_path
    )
    model_config = object()
    fp8 = SimpleNamespace(
        deep_gemm_warmup_on_start=True,
        sparse_deep_gemm_num_sms=116,
    )

    with redirect:
        for local_rank in range(8):
            trainer_module.setup_deep_gemm_cache(local_rank, model_config, 8, fp8)
        for local_rank in range(8):
            warmup._share_node_kernels(local_rank, 8)

        for local_rank in range(8):
            rank_dir = rank_root / f"rank_{local_rank}" / "deepgemm" / "cache"
            assert rank_dir.is_relative_to(rank_root)
            for owner in range(8):
                marker = rank_dir / f"kernel.rank{owner}" / "owner.txt"
                assert marker.read_text(encoding="utf-8") == str(owner)

    assert calls == [(model_config, 8, 116, rank) for rank in range(8)]
    assert warmup._rank_cache_dir is original_rank_cache_dir
    assert trainer_module.setup_deep_gemm_cache is original_setup
    assert all(
        path.resolve().is_relative_to(tmp_path.resolve())
        for path in rank_root.rglob("owner.txt")
    )


def test_deep_gemm_cache_adapter_rejects_alternate_writers_before_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import torchtitan.ops.warmup.deep_gemm_fast_warmup as warmup
    import torchtitan.trainer as trainer_module

    rank_root = tmp_path / "rank-cache"
    rank_root.mkdir()
    shared_root = tmp_path / "shared-cache"
    monkeypatch.setenv("DG_JIT_SHARED_CACHE_DIR", str(shared_root))
    monkeypatch.delenv("PROPAGATE_DG_CACHE", raising=False)
    monkeypatch.delenv("PROPAGATE_DG_CACHE_DIR", raising=False)
    redirect = measure.DeepGemmCacheRedirect(
        warmup, trainer_module, rank_root, task_root=tmp_path
    )

    with redirect:
        with pytest.raises(RuntimeError, match="requires the fixed DeepGEMM warmup"):
            trainer_module.setup_deep_gemm_cache(
                0,
                object(),
                8,
                SimpleNamespace(deep_gemm_warmup_on_start=False),
            )
        assert not shared_root.exists()
        assert not any(rank_root.iterdir())

        monkeypatch.setenv("PROPAGATE_DG_CACHE", "1")
        with pytest.raises(RuntimeError, match="PROPAGATE_DG_CACHE paths"):
            trainer_module.setup_deep_gemm_cache(
                0,
                object(),
                8,
                SimpleNamespace(deep_gemm_warmup_on_start=True),
            )
        assert not shared_root.exists()
        assert not any(rank_root.iterdir())


def _run_short(trainer, stop_after: int) -> None:
    for step in range(1, stop_after + 1):
        trainer.step = step
        trainer.metrics_processor.ntokens_since_last_log += LOCAL_PHYSICAL_PER_UPDATE
        trainer.ntokens_seen.value += GLOBAL_SUPERVISED_PER_UPDATE
        trainer.metrics_processor.interval_metrics()
        trainer.runtime.stats = {
            "allocated_bytes.all.peak": 1_000 + step,
            "active_bytes.all.peak": 2_000 + step,
            "reserved_bytes.all.peak": 3_000 + step,
            "num_alloc_retries": max(0, step - 20),
            "num_ooms": int(step >= 24),
        }
        trainer.metrics_module.reset_peak_stats()
        trainer.profiler.step()


def test_profile_update_defaults_validation_cli_decode_and_resolved_config(
    tmp_path: Path,
) -> None:
    default_args = runner._parse_args(["--run-id", "default"])
    assert default_args.warmup == 10
    assert default_args.window == 10
    assert default_args.profile_update is None
    default_overrides = runner._training_overrides(
        default_args, {"native": tmp_path / "default-native"}
    )
    default_profile_pairs = [
        tuple(default_overrides[index : index + 2])
        for index in range(0, len(default_overrides), 2)
        if default_overrides[index].startswith("--profiler.")
    ]
    assert default_profile_pairs == [
        ("--profiler.enable_profiling", "false"),
        ("--profiler.enable_memory_snapshot", "false"),
    ]

    invalid_args = (
        ("--steps", "2", "--profile-update", "2"),
        ("--steps", "1", "--warmup", "1", "--profile-update", "1"),
        ("--steps", "2", "--warmup", "2", "--profile-update", "2"),
        ("--steps", "50", "--warmup", "1", "--profile-update", "50"),
        ("--steps", "8", "--warmup", "3", "--profile-update", "7"),
        ("--steps", "10", "--warmup", "1"),
        (
            "--steps",
            "57",
            "--warmup",
            "7",
            "--window",
            "50",
            "--profile-update",
            "57",
        ),
        (
            "--steps",
            "7",
            "--warmup",
            "3",
            "--monitor-interval",
            "5",
            "--window",
            "0",
            "--profile-update",
            "7",
        ),
    )
    for arguments in invalid_args:
        with pytest.raises(SystemExit):
            runner._parse_args(["--run-id", "invalid", *arguments])

    args = runner._parse_args(
        [
            "--run-id",
            "profile",
            "--steps",
            "7",
            "--warmup",
            "3",
            "--window",
            "0",
            "--profile-update",
            "7",
        ]
    )
    overrides = runner._training_overrides(
        args, {"native": tmp_path / "profile-native"}
    )
    profile_pairs = [
        (overrides[index][2:], overrides[index + 1])
        for index in range(0, len(overrides), 2)
        if overrides[index].startswith("--profiler.")
    ]

    from torchtitan.config.decode import decode
    from torchtitan.config.overrides import build_override_layers
    from torchtitan.config.schema.tools.profiler import ProfilerConfig

    profiler_layers, parsed = build_override_layers(profile_pairs)
    assert len(profiler_layers) == len(profile_pairs)
    typed_values = {
        key.removeprefix("profiler."): value
        for key, value in parsed.items()
    }
    decoded = decode(ProfilerConfig, typed_values)
    expected_values = runner._profile_config_values(args)
    assert {f"profiler.{key}": value for key, value in typed_values.items()} == expected_values
    assert decoded.enable_profiling is True
    assert decoded.enable_memory_snapshot is False
    assert decoded.profile_ranks == [0, 1]
    assert decoded.save_traces_folder == "profiling/traces"
    assert decoded.record_shapes is False
    assert decoded.profile_freq == 2
    assert decoded.profiler_warmup == 1
    assert decoded.profiler_active == 1
    assert decoded.profiler_repeat == 1
    assert decoded.profiler_skip_first == 5
    assert decoded.profiler_skip_first_wait == 0
    assert runner._profile_capture_spec(args)["warmup_updates"] == 3
    assert runner._profile_capture_spec(args)["target_update"] == 7
    paths = {
        "logs": tmp_path / "logs",
        "measurement": tmp_path / "measurement",
        "native": tmp_path / "profile-native",
        "config": tmp_path / "config",
    }
    command = runner._command(args, paths)
    helper_profile_index = command.index("--profile-update")
    assert command[helper_profile_index + 1] == "7"
    assert helper_profile_index < command.index("--")

    default_command = runner._command(
        default_args,
        {
            "logs": tmp_path / "logs-default",
            "measurement": tmp_path / "measurement-default",
            "native": tmp_path / "native-default",
            "config": tmp_path / "config-default",
        },
    )
    assert "--profile-update" not in default_command

    required_config = {
        "training.steps": args.steps,
        "training.local_batch_size": 2,
        "training.global_batch_size": 512,
        "training.seq_len": 4096,
        "lr_scheduler.total_steps": 65000,
        "lr_scheduler.warmup_steps": 2000,
        "debug.seed": 2048,
        "loss.num_chunks": 2,
        "callbacks.ema.type": "ema",
        "metrics.log_freq": 1,
        "checkpoint.enable": False,
        "validator.enable": False,
        "compile.enable": True,
        "activation_checkpoint.mode": "selective",
        "parallelism.expert_parallel_degree": 8,
        "parallelism.pipeline_parallel_degree": 1,
        "parallelism.tensor_parallel_degree": 1,
        "parallelism.context_parallel_degree": 1,
        "fp8.sparse_use_grouped_gemm": True,
        "fp8.deep_gemm_warmup_on_start": True,
        "dataloader.type": "streaming_text",
        "dump_folder": str((tmp_path / "profile-native").resolve()),
        "hf_assets_path": str(runner.TOKENIZER.resolve()),
    }

    def make_resolved_config(profile_values: dict[str, object]) -> dict[str, object]:
        config: dict[str, object] = {}
        for dotted, value in {**required_config, **profile_values}.items():
            current = config
            components = dotted.split(".")
            for component in components[:-1]:
                current = current.setdefault(component, {})  # type: ignore[assignment,union-attr]
            current[components[-1]] = value  # type: ignore[index]
        return config

    native_dir = tmp_path / "profile-native"
    config_dir = native_dir / "config"
    config_dir.mkdir(parents=True)
    resolved_path = config_dir / "resolved_config.yaml"
    import yaml

    profile_values = runner._profile_config_values(args)
    resolved_config = make_resolved_config(profile_values)
    resolved_path.write_text(yaml.safe_dump(resolved_config), encoding="utf-8")
    assert runner._verify_resolved_config({"native": native_dir}, args)

    for dotted, expected in profile_values.items():
        mutated = deepcopy(resolved_config)
        current = mutated
        components = dotted.split(".")
        for component in components[:-1]:
            current = current[component]
        old_value = current[components[-1]]
        current[components[-1]] = (
            [1, 0]
            if isinstance(expected, list)
            else (
                not expected
                if isinstance(expected, bool)
                else expected + 1
                if isinstance(expected, int)
                else f"{expected}-wrong"
            )
        )
        resolved_path.write_text(yaml.safe_dump(mutated), encoding="utf-8")
        with pytest.raises(RuntimeError, match="resolved config"):
            runner._verify_resolved_config({"native": native_dir}, args)
        current[components[-1]] = old_value


def test_measure_cli_accepts_only_the_profile_update_schedule(tmp_path: Path) -> None:
    helper_args = [
        "--measurement-dir",
        str(tmp_path / "measurement"),
        "--run-id",
        "profile",
        "--steps",
        "7",
        "--warmup",
        "3",
        "--window",
        "0",
        "--profile-update",
        "7",
        "--",
        "experiment.yaml",
    ]
    args, train_argv = measure._parse_args(helper_args)
    assert args.profile_update == args.steps == 7
    assert args.warmup == 3
    assert args.window == 0
    assert train_argv == ["experiment.yaml"]

    invalid_variants = (
        {"--warmup": "0"},
        {"--steps": "8"},
        {"--window": "50"},
    )
    for changes in invalid_variants:
        changed = list(helper_args)
        for key, value in changes.items():
            changed[changed.index(key) + 1] = value
        with pytest.raises(SystemExit):
            measure._parse_args(changed)


@pytest.mark.parametrize("rank,expected_scope_count", [(0, 34), (1, 34), (2, 0)])
def test_profile_wrappers_delegate_target_calls_and_restore(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rank: int,
    expected_scope_count: int,
) -> None:
    import torch

    monkeypatch.setenv("RANK", str(rank))
    monkeypatch.setenv("WORLD_SIZE", "8")
    scope_names: list[str] = []

    @contextmanager
    def record_scope(name: str):
        scope_names.append(name)
        yield

    monkeypatch.setattr(torch.profiler, "record_function", record_scope)
    calls: dict[str, object] = {"forward": [], "train": [], "optimizer": [], "build": 0}

    class ProfileOptimizerContainer:
        def step(self, *args, **kwargs):
            calls["optimizer"].append((args, kwargs))
            return ("optimizer-result", args, kwargs)

    class ProfileProfiler:
        def build_torch_profiler(self):
            calls["build"] += 1
            return object() if rank in (0, 1) else None

        def step(self):
            return "profiler-step"

    class ProfileTrainer:
        def __init__(self) -> None:
            self.step = 0
            self.metrics_processor = FakeMetricsProcessor()
            self.ntokens_seen = FakeScalar()
            self.model_parts = [FakePart([FakeParameter(requires_grad=True, local_numel=1, dtype="float32")])]
            self.optimizers = ProfileOptimizerContainer()
            self.profiler = ProfileProfiler()
            self.return_value = object()

        def forward_backward_step(self, *args, **kwargs):
            calls["forward"].append((self.step, args, kwargs))
            return ("forward-result", args, kwargs)

        def train_step(self, label, *, multiplier):
            calls["train"].append((self.step, label, multiplier))
            forward_results = [
                self.forward_backward_step(label, index, flag=multiplier)
                for index in range(32)
            ]
            optimizer_result = self.optimizers.step(label, update=self.step)
            return ("train-step-result", label, multiplier, forward_results[-1], optimizer_result)

        def train(self):
            self.profiler.build_torch_profiler()
            results = []
            for step in (1, 2):
                self.step = step
                results.append(self.train_step(f"update-{step}", multiplier=step))
                self.profiler.step()
            return ("native-train-result", results[-1])

    sync_calls: list[int] = []
    adapter = measure.NativeAdapter(
        trainer_class=ProfileTrainer,
        profiler_class=ProfileProfiler,
        optimizer_class=ProfileOptimizerContainer,
        metrics_class=FakeMetricsProcessor,
        metrics_module=SimpleNamespace(reset_peak_stats=lambda: None),
        options={
            "run_id": "profile-wrapper-test",
            "output_dir": tmp_path / "measurement",
            "expected_steps": 2,
            "warmup_updates": 1,
            "window_updates": 0,
            "monitor_interval": 50,
            "capture_microbatches": False,
            "profile_update": 2,
        },
        cuda_synchronize=lambda: sync_calls.append(1),
        memory_stats=lambda: {},
    )
    originals = (
        ProfileTrainer.forward_backward_step,
        ProfileTrainer.train_step,
        ProfileProfiler.build_torch_profiler,
        ProfileOptimizerContainer.step,
    )

    with adapter:
        result = ProfileTrainer().train()
    assert result[0] == "native-train-result"
    assert len(calls["forward"]) == 64
    assert len(calls["train"]) == 2
    assert len(calls["optimizer"]) == 2
    assert calls["forward"][-1] == (2, ("update-2", 31), {"flag": 2})
    assert calls["optimizer"][-1] == (("update-2",), {"update": 2})
    assert calls["build"] == 1
    assert len(scope_names) == expected_scope_count
    if rank in (0, 1):
        assert scope_names.count(f"task.profile_update.2.microbatch.rank{rank}") == 32
        assert scope_names.count(f"task.profile_update.2.train_step.rank{rank}") == 1
        assert scope_names.count(f"task.profile_update.2.optimizer_step.rank{rank}") == 1
    assert sync_calls == []
    assert (
        ProfileTrainer.forward_backward_step,
        ProfileTrainer.train_step,
        ProfileProfiler.build_torch_profiler,
        ProfileOptimizerContainer.step,
    ) == originals

    receipt = json.loads(
        (tmp_path / "measurement" / f"rank-{rank:02d}.json").read_text()
    )["profile_capture"]
    assert receipt["microbatch_calls"] == 32
    assert receipt["train_step_calls"] == 1
    assert receipt["optimizer_step_calls"] == 1
    assert receipt["native_profiler_created"] is (rank in (0, 1))


@pytest.mark.parametrize("failure", ["forward", "train", "optimizer"])
def test_profile_wrappers_preserve_exceptions_and_restore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "8")

    class ProfileOptimizerContainer:
        def step(self, *args, **kwargs):
            if failure == "optimizer":
                raise LookupError("optimizer sentinel")
            return None

    class ProfileProfiler:
        def build_torch_profiler(self):
            return object()

        def step(self):
            return None

    class ProfileTrainer:
        def __init__(self) -> None:
            self.step = 0
            self.metrics_processor = FakeMetricsProcessor()
            self.ntokens_seen = FakeScalar()
            self.model_parts = [FakePart([])]
            self.optimizers = ProfileOptimizerContainer()
            self.profiler = ProfileProfiler()

        def forward_backward_step(self, *args, **kwargs):
            if failure == "forward":
                raise LookupError("forward sentinel")
            return None

        def train_step(self, label, *, multiplier):
            if failure == "train":
                raise LookupError("train sentinel")
            for index in range(32):
                self.forward_backward_step(label, index, flag=multiplier)
            return self.optimizers.step(label, update=self.step)

        def train(self):
            self.profiler.build_torch_profiler()
            for step in (1, 2):
                self.step = step
                self.train_step(f"update-{step}", multiplier=step)
                self.profiler.step()

    adapter = measure.NativeAdapter(
        trainer_class=ProfileTrainer,
        profiler_class=ProfileProfiler,
        optimizer_class=ProfileOptimizerContainer,
        metrics_class=FakeMetricsProcessor,
        metrics_module=SimpleNamespace(reset_peak_stats=lambda: None),
        options={
            "run_id": f"profile-error-{failure}",
            "output_dir": tmp_path / failure,
            "expected_steps": 2,
            "warmup_updates": 1,
            "window_updates": 0,
            "monitor_interval": 50,
            "capture_microbatches": False,
            "profile_update": 2,
        },
        cuda_synchronize=lambda: pytest.fail("profile mode must not synchronize CUDA"),
        memory_stats=lambda: {},
    )
    originals = (
        ProfileTrainer.forward_backward_step,
        ProfileTrainer.train_step,
        ProfileProfiler.build_torch_profiler,
        ProfileOptimizerContainer.step,
    )
    expected_message = f"{failure} sentinel"
    with adapter, pytest.raises(LookupError, match=expected_message):
        ProfileTrainer().train()
    assert (
        ProfileTrainer.forward_backward_step,
        ProfileTrainer.train_step,
        ProfileProfiler.build_torch_profiler,
        ProfileOptimizerContainer.step,
    ) == originals


def test_profile_wrappers_capture_one_native_cpu_update_without_exit_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import torch
    from torchtitan.components.optimizer import OptimizersContainer
    from torchtitan.config.schema.components.optimizer import OptimizersContainerConfig
    from torchtitan.config.schema.tools.profiler import ProfilerConfig
    from torchtitan.tools.profiler import Profiler

    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    native_dir = tmp_path / "native"
    config = ProfilerConfig(
        enable_profiling=True,
        profile_ranks=[0],
        record_shapes=False,
        profile_freq=2,
        profiler_warmup=1,
        profiler_active=1,
        profiler_repeat=1,
        profiler_skip_first=5,
        profiler_skip_first_wait=0,
        enable_memory_snapshot=False,
    )

    model = torch.nn.Linear(2, 2, bias=False)
    optimizer_config = OptimizersContainerConfig(
        name="AdamW", implementation="for-loop"
    )
    optimizers = OptimizersContainer(optimizer_config, model_parts=[model])

    class ProfileTrainer:
        def __init__(self) -> None:
            self.step = 0
            self.metrics_processor = FakeMetricsProcessor()
            self.ntokens_seen = FakeScalar()
            self.model_parts = [model]
            self.optimizers = optimizers

        def forward_backward_step(self, value):
            return value

        def train_step(self, label):
            results = [self.forward_backward_step((label, index)) for index in range(32)]
            for parameter in model.parameters():
                parameter.grad = torch.ones_like(parameter)
            optimizer_result = self.optimizers.step()
            return results[-1], optimizer_result

        def train(self):
            profiler = Profiler(config, global_step=0, base_folder=str(native_dir))
            results = []
            with profiler:
                for step in range(1, 8):
                    self.step = step
                    results.append(self.train_step(f"update-{step}"))
                    profiler.step()
            return results[-1]

    sync_calls: list[int] = []
    adapter = measure.NativeAdapter(
        trainer_class=ProfileTrainer,
        profiler_class=Profiler,
        optimizer_class=OptimizersContainer,
        metrics_class=FakeMetricsProcessor,
        metrics_module=SimpleNamespace(reset_peak_stats=lambda: None),
        options={
            "run_id": "native-cpu-profile-test",
            "output_dir": tmp_path / "measurement",
            "expected_steps": 7,
            "warmup_updates": 3,
            "window_updates": 0,
            "monitor_interval": 50,
            "capture_microbatches": False,
            "profile_update": 7,
        },
        cuda_synchronize=lambda: sync_calls.append(1),
        memory_stats=lambda: {},
    )
    with adapter:
        native_result = ProfileTrainer().train()

    trace_files = sorted((native_dir / "profiling/traces").rglob("*_trace.json"))
    assert len(trace_files) == 1
    assert trace_files[0].name == "rank0_trace.json"
    assert "iteration_7" in trace_files[0].parts
    trace = json.loads(trace_files[0].read_text(encoding="utf-8"))
    names = [event.get("name") for event in trace["traceEvents"] if event.get("ph") == "X"]
    assert names.count("task.profile_update.7.microbatch.rank0") == 32
    assert names.count("task.profile_update.7.train_step.rank0") == 1
    assert names.count("task.profile_update.7.optimizer_step.rank0") == 1
    assert sync_calls == []
    assert native_result[0][0] == "update-7"
    receipt = json.loads(
        (tmp_path / "measurement/rank-00.json").read_text(encoding="utf-8")
    )["profile_capture"]
    assert receipt["native_profiler_build_calls"] == 1
    assert receipt["native_profiler_created"] is True


def _profile_parser_fixture(run_dir: Path, *, target_update: int = 2, warmup: int = 1) -> None:
    args = runner._parse_args(
        [
            "--run-id",
            "profile-parser-test",
            "--steps",
            str(target_update),
            "--warmup",
            str(warmup),
            "--window",
            "0",
            "--profile-update",
            str(target_update),
        ]
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "run_id": "profile-parser-test",
        "steps": target_update,
        "warmup_updates": warmup,
        "window_updates": 0,
        "monitor_interval": 50,
        "profile_capture": runner._profile_capture_spec(args),
        "deep_gemm_compilation_guard": compiler_stats.deep_gemm_guard_manifest(
            enabled=True, activate_after_update=warmup
        ),
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    measurement_dir = run_dir / "measurement"
    measurement_dir.mkdir()
    for rank in range(8):
        record = {
            "schema_version": 1,
            "run_id": "profile-parser-test",
            "rank": rank,
            "world_size": 8,
            "host_name": "single-host",
            "training_status": "complete",
            "expected_steps": target_update,
            "completed_steps": target_update,
            "measurement_status": "disabled",
            "measurement": None,
            "compiler_guard": {
                "environment_variable": compiler_stats.DEEP_GEMM_GUARD_ENV,
                "guarded_value": compiler_stats.DEEP_GEMM_GUARD_VALUE,
                "activation_after_update": warmup,
                "previous_value": None,
                "flag_after_activation": "0",
                "activation_sync_calls": 1,
                "end_after_update": target_update,
                "flag_at_end_boundary": "0",
                "binary_identity": {
                    "build_version": compiler_stats.DEEP_GEMM_GUARD_BUILD,
                    "extension_path": "/opt/torchtitan-venv/lib/python3.12/site-packages/deep_gemm/_C.so",
                    "sha256": compiler_stats.DEEP_GEMM_GUARD_EXTENSION_SHA256,
                },
            },
            "cadence": [
                {"step": step, "interval_s": 1.0, "monitoring_step": False}
                for step in range(1, target_update + 1)
            ],
            "parameter_census": {"trainable": {}, "frozen": {}},
            "profile_capture": {
                "target_update": target_update,
                "warmup_updates": warmup,
                "microbatch_calls": 32,
                "train_step_calls": 1,
                "optimizer_step_calls": 1,
                "native_profiler_build_calls": 1,
                "native_profiler_created": rank in (0, 1),
            },
        }
        (measurement_dir / f"rank-{rank:02d}.json").write_text(
            json.dumps(record), encoding="utf-8"
        )

    trace_dir = run_dir / "native/profiling/traces/iteration_2"
    trace_dir.mkdir(parents=True)
    for rank in (0, 1):
        names = [
            *([f"task.profile_update.2.microbatch.rank{rank}"] * 32),
            f"task.profile_update.2.train_step.rank{rank}",
            f"task.profile_update.2.optimizer_step.rank{rank}",
        ]
        events = [
            {"ph": "X", "cat": "user_annotation", "name": name, "ts": index, "dur": 1}
            for index, name in enumerate(names)
        ]
        (trace_dir / f"rank{rank}_trace.json").write_text(
            json.dumps({"traceEvents": events}), encoding="utf-8"
        )


def test_profile_parser_requires_exact_rank_traces_and_markers(tmp_path: Path) -> None:
    _profile_parser_fixture(tmp_path)
    summary = parse_results.summarize_run(tmp_path)
    assert summary["status"] == "complete"
    assert summary["mode"] == "profile_update"
    assert summary["profile_capture"]["profile_ranks"] == [0, 1]
    assert set(summary["profile_capture"]["traces_by_rank"]) == {"0", "1"}
    for trace in summary["profile_capture"]["traces_by_rank"].values():
        assert trace["size_bytes"] > 0
        assert len(trace["sha256"]) == 64
    assert "tokens" not in summary


def test_profile_parser_separates_gpu_marker_projections_from_cpu_calls(
    tmp_path: Path,
) -> None:
    _profile_parser_fixture(tmp_path)
    trace_dir = tmp_path / "native/profiling/traces/iteration_2"
    for rank in (0, 1):
        path = trace_dir / f"rank{rank}_trace.json"
        trace = json.loads(path.read_text(encoding="utf-8"))
        gpu_projections = [
            {**event, "cat": "gpu_user_annotation"}
            for event in trace["traceEvents"]
            if event["name"].endswith(("microbatch.rank" + str(rank), "train_step.rank" + str(rank)))
        ]
        trace["traceEvents"].extend(gpu_projections)
        path.write_text(json.dumps(trace), encoding="utf-8")

    summary = parse_results.summarize_run(tmp_path)

    assert summary["status"] == "complete", summary["errors"]
    for rank in (0, 1):
        trace_summary = summary["profile_capture"]["traces_by_rank"][str(rank)]
        assert trace_summary["marker_counts"] == {
            f"task.profile_update.2.microbatch.rank{rank}": 32,
            f"task.profile_update.2.train_step.rank{rank}": 1,
            f"task.profile_update.2.optimizer_step.rank{rank}": 1,
        }
        assert trace_summary["gpu_projection_counts"] == {
            f"task.profile_update.2.microbatch.rank{rank}": 32,
            f"task.profile_update.2.train_step.rank{rank}": 1,
        }


@pytest.mark.parametrize(
    "failure",
    [
        "missing",
        "extra",
        "marker_count",
        "wrong_rank",
        "wrong_update_marker",
        "wrong_rank_marker",
        "missing_category",
        "unknown_category",
        "wrong_category",
        "extra_cpu_marker",
        "unexpected_gpu_marker",
        "non_complete_marker",
        "call_count",
        "profiler_rank",
        "missing_guard",
        "bad_guard_flag",
    ],
)
def test_profile_parser_rejects_incomplete_trace_receipts(
    tmp_path: Path, failure: str
) -> None:
    _profile_parser_fixture(tmp_path)
    trace_dir = tmp_path / "native/profiling/traces/iteration_2"
    if failure == "missing":
        (trace_dir / "rank1_trace.json").unlink()
    elif failure == "extra":
        (trace_dir / "rank2_trace.json").write_text(
            json.dumps({"traceEvents": []}), encoding="utf-8"
        )
    elif failure == "marker_count":
        path = trace_dir / "rank0_trace.json"
        trace = json.loads(path.read_text(encoding="utf-8"))
        for index, event in enumerate(trace["traceEvents"]):
            if event["name"] == "task.profile_update.2.microbatch.rank0":
                del trace["traceEvents"][index]
                break
        path.write_text(json.dumps(trace), encoding="utf-8")
    elif failure == "wrong_rank":
        (trace_dir / "rank0_trace.json").rename(trace_dir / "rank2_trace.json")
        (trace_dir / "rank1_trace.json").rename(trace_dir / "rank3_trace.json")
    elif failure in {
        "wrong_update_marker",
        "wrong_rank_marker",
        "unexpected_gpu_marker",
    }:
        path = trace_dir / "rank0_trace.json"
        trace = json.loads(path.read_text(encoding="utf-8"))
        if failure == "wrong_update_marker":
            marker = "task.profile_update.1.microbatch.rank0"
            category = "user_annotation"
        elif failure == "wrong_rank_marker":
            marker = "task.profile_update.2.microbatch.rank2"
            category = "user_annotation"
        else:
            marker = "task.profile_update.2.optimizer_step.rank2"
            category = "gpu_user_annotation"
        trace["traceEvents"].append(
            {"ph": "X", "cat": category, "name": marker, "ts": 99, "dur": 1}
        )
        path.write_text(json.dumps(trace), encoding="utf-8")
    elif failure in {"missing_category", "unknown_category", "wrong_category"}:
        path = trace_dir / "rank0_trace.json"
        trace = json.loads(path.read_text(encoding="utf-8"))
        event = next(
            item
            for item in trace["traceEvents"]
            if item["name"] == "task.profile_update.2.optimizer_step.rank0"
        )
        if failure == "missing_category":
            event.pop("cat")
        elif failure == "unknown_category":
            event["cat"] = "cpu_user_annotation"
        else:
            event["cat"] = "gpu_user_annotation"
        path.write_text(json.dumps(trace), encoding="utf-8")
    elif failure in {"extra_cpu_marker", "non_complete_marker"}:
        path = trace_dir / "rank0_trace.json"
        trace = json.loads(path.read_text(encoding="utf-8"))
        if failure == "extra_cpu_marker":
            event = next(
                item
                for item in trace["traceEvents"]
                if item["name"] == "task.profile_update.2.microbatch.rank0"
            )
            trace["traceEvents"].append({**event, "ts": 99})
        else:
            event = next(
                item
                for item in trace["traceEvents"]
                if item["name"] == "task.profile_update.2.train_step.rank0"
            )
            event["ph"] = "B"
        path.write_text(json.dumps(trace), encoding="utf-8")
    elif failure == "call_count":
        path = tmp_path / "measurement/rank-04.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        record["profile_capture"]["optimizer_step_calls"] = 2
        path.write_text(json.dumps(record), encoding="utf-8")
    elif failure == "profiler_rank":
        path = tmp_path / "measurement/rank-07.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        record["profile_capture"]["native_profiler_created"] = True
        path.write_text(json.dumps(record), encoding="utf-8")
    elif failure == "missing_guard":
        path = tmp_path / "measurement/rank-07.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        record.pop("compiler_guard")
        path.write_text(json.dumps(record), encoding="utf-8")
    else:
        path = tmp_path / "measurement/rank-07.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        record["compiler_guard"]["flag_at_end_boundary"] = "1"
        path.write_text(json.dumps(record), encoding="utf-8")

    summary = parse_results.summarize_run(tmp_path)
    assert summary["status"] == "incomplete"
    assert summary["errors"]
