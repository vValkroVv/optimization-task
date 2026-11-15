from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

TASK_ROOT = Path(__file__).resolve().parents[3]


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def bridge_run() -> Any:
    return _load_module(
        "_test_flash_qla_run",
        TASK_ROOT / "runs/flash-qla/run.py",
    )


@pytest.fixture
def bridge_measure() -> Any:
    return _load_module(
        "_test_flash_qla_measure",
        TASK_ROOT / "runs/flash-qla/measure.py",
    )


def _make_task_source_tree(root: Path) -> tuple[Path, Path]:
    gift_root = root / "gift"
    backend = gift_root / "torchtitan/experiments/flash_qla/backend.py"
    backend.parent.mkdir(parents=True)
    backend.write_text("def install_backend(variant): return {\"variant\": variant}\n")
    staged_fla = gift_root / "contrib/flash-linear-attention/flash_attn/__init__.py"
    staged_fla.parent.mkdir(parents=True)
    staged_fla.write_text("")
    staged = gift_root / "contrib/flash-qla/flash_qla/__init__.py"
    staged.parent.mkdir(parents=True)
    staged.write_text("")
    return gift_root, backend


def _fake_base(root: Path, *, original_environment: Any | None = None) -> Any:
    gift_root, _ = _make_task_source_tree(root)
    helper_dir = root / "runs/overnight-streaming-perf"
    helper_dir.mkdir(parents=True)
    (helper_dir / "measure.py").write_text("# original child path\n")
    if original_environment is None:
        original_environment = lambda paths: {
            **os.environ,
            "PYTHONPATH": f"{gift_root}:{helper_dir}",
        }
    return SimpleNamespace(
        GIFT_ROOT=gift_root,
        HELPER_DIR=helper_dir,
        TASK_ROOT=root,
        _run_environment=original_environment,
        _git=lambda *args, **kwargs: SimpleNamespace(stdout=b"deadbeef\n"),
    )


def _fake_paths(root: Path, run_id: str = "test-run") -> dict[str, Path]:
    attempt = root / "runs/flash-qla/artifacts/raw" / run_id
    attempt.mkdir(parents=True)
    measurement = attempt / "measurement"
    measurement.mkdir()
    cache = attempt / "cache"
    cache.mkdir()
    return {
        "attempt": attempt,
        "measurement": measurement,
        "cache": cache,
    }


def _patch_bridge_paths(bridge_run: Any, tmp_path: Path) -> Path:
    bridge_dir = tmp_path / "runs/flash-qla"
    bridge_dir.mkdir(parents=True)
    run_path = bridge_dir / "run.py"
    measure_path = bridge_dir / "measure.py"
    capture_path = bridge_dir / "capture.py"
    run_path.write_text("bridge-run")
    measure_path.write_text("bridge-measure")
    capture_path.write_text("bridge-capture")
    bridge_run.TASK_ROOT = tmp_path
    bridge_run.BRIDGE_DIR = bridge_dir
    bridge_run.BRIDGE_MEASURE_PATH = measure_path
    return bridge_dir


def _fake_native_modules() -> tuple[Any, Any, Any, Any, Any]:
    def native_route(*args: Any) -> tuple[Any, ...]:
        return args

    def fused_route(*args: Any) -> tuple[Any, ...]:
        return args

    def native_ce(*args: Any) -> tuple[Any, ...]:
        return args

    def stable_ce_fn(*args: Any) -> tuple[Any, ...]:
        return args

    native_route.__module__ = "torchtitan.distributed.deepep.routing"
    fused_route.__module__ = "torchtitan.experiments.four_fusion.routing_metadata"
    native_ce.__module__ = "torchtitan.components.loss"
    stable_ce_fn.__module__ = "torchtitan.experiments.four_fusion.stable_ce"
    hybridep = SimpleNamespace(indices_to_map=native_route)
    original_routing = SimpleNamespace(indices_to_map=native_route)
    fused_routing = SimpleNamespace(indices_to_map=fused_route)
    stable_ce = SimpleNamespace(cross_entropy_loss=stable_ce_fn)

    class CrossEntropyLoss:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            self.fn = stable_ce.cross_entropy_loss

    class ChunkedCELoss:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            self.fn = stable_ce.cross_entropy_loss

    class MTPLoss(ChunkedCELoss):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)

    CrossEntropyLoss.__module__ = "torchtitan.components.loss"
    ChunkedCELoss.__module__ = "torchtitan.components.loss"
    MTPLoss.__module__ = "torchtitan.components.loss"
    loss_module = SimpleNamespace(
        cross_entropy_loss=native_ce,
        CrossEntropyLoss=CrossEntropyLoss,
        ChunkedCELoss=ChunkedCELoss,
        MTPLoss=MTPLoss,
    )
    return hybridep, original_routing, fused_routing, loss_module, stable_ce


def test_runner_parser_forwards_existing_arguments_unchanged(bridge_run: Any) -> None:
    native = [
        "--run-id",
        "variant-a",
        "--steps",
        "140",
        "--warmup",
        "40",
        "--window",
        "100",
        "--epoch-size",
        "102400",
        "--gfsdp-reuse-bf16-expert-weights",
    ]
    args, forwarded = bridge_run._parse_bridge_args(
        ["--gdn-backend", "external_gate", *native]
    )
    assert args.gdn_backend == "external_gate"
    assert args.capture_fixtures is None
    assert args.routing_metadata == "fused"
    assert args.cross_entropy == "stable"
    assert forwarded == native


def test_runner_parser_selects_routing_and_cross_entropy_independently(
    bridge_run: Any,
) -> None:
    args, forwarded = bridge_run._parse_bridge_args(
        [
            "--gdn-backend",
            "external_gate",
            "--routing-metadata",
            "original",
            "--cross-entropy",
            "original",
            "--run-id",
            "native-fusion-control",
        ]
    )
    assert args.routing_metadata == "original"
    assert args.cross_entropy == "original"
    assert forwarded == ["--run-id", "native-fusion-control"]


def test_runner_parser_rejects_invalid_backend_and_non_fla_capture(bridge_run: Any) -> None:
    with pytest.raises(SystemExit):
        bridge_run._parse_bridge_args(["--gdn-backend", "unknown", "--run-id", "x"])
    with pytest.raises(SystemExit):
        bridge_run._parse_bridge_args(
            [
                "--gdn-backend",
                "fused_gate",
                "--capture-fixtures",
                "/tmp/fixtures",
                "--run-id",
                "x",
            ]
        )


def test_default_output_root_uses_bridge_artifacts_without_rewriting_override(
    bridge_run: Any,
) -> None:
    defaulted = bridge_run._add_default_output_root(["--run-id", "x"], Path("/task/raw"))
    assert defaulted == ["--run-id", "x", "--output-root", "/task/raw"]
    explicit = ["--run-id", "x", "--output-root", "/custom/artifacts/raw"]
    assert bridge_run._add_default_output_root(explicit, Path("/task/raw")) == explicit


def test_measure_script_replacement_changes_only_generated_child_path(bridge_run: Any) -> None:
    original = Path("/task/runs/overnight-streaming-perf/measure.py")
    bridge = Path("/task/runs/flash-qla/measure.py")
    command = [
        "/python",
        "-m",
        "torch.distributed.run",
        str(original),
        "--measurement-dir",
        "/task/run/measurement",
        "--",
        "/task/config.yaml",
    ]
    replaced = bridge_run._replace_measure_script(command, original, bridge)
    assert replaced == ["/python", "-m", "torch.distributed.run", str(bridge), *command[4:]]
    with pytest.raises(RuntimeError, match="exactly one"):
        bridge_run._replace_measure_script([str(original), str(original)], original, bridge)


def test_config_hash_hook_tracks_bridge_sources_and_restores_after_main(
    bridge_run: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bridge_dir = _patch_bridge_paths(bridge_run, tmp_path)
    base = _fake_base(tmp_path)
    original_environment = base._run_environment
    original_command = lambda args, paths: []
    original_hashes = lambda: {"canonical": "unchanged"}
    base._command = original_command
    base._config_hashes = original_hashes
    observed: list[bool] = []

    def base_main(argv: list[str]) -> int:
        before = base._config_hashes()
        (bridge_dir / "capture.py").write_text("changed during native process")
        after = base._config_hashes()
        observed.append(before["flash_qla_bridge_sources"] != after["flash_qla_bridge_sources"])
        return 0

    base.main = base_main
    monkeypatch.setattr(bridge_run, "_load_base_runner", lambda: base)
    monkeypatch.setattr(bridge_run, "DEFAULT_OUTPUT_ROOT", tmp_path / "artifacts/raw")
    result = bridge_run.main(
        [
            "--gdn-backend",
            "fla",
            "--run-id",
            "hash-freeze",
            "--output-root",
            str(tmp_path / "artifacts/raw"),
        ]
    )
    assert result == 0
    assert observed == [True]
    assert base._run_environment is original_environment
    assert base._command is original_command
    assert base._config_hashes is original_hashes


def test_environment_hook_clears_stale_values_and_records_rank_cache_paths(
    bridge_run: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bridge_dir = _patch_bridge_paths(bridge_run, tmp_path)
    preserved_pythonpath = os.pathsep.join(("/existing/entry-one", "/existing/entry-two"))
    base = _fake_base(
        tmp_path,
        original_environment=lambda paths: {**os.environ, "PYTHONPATH": preserved_pythonpath},
    )
    paths = _fake_paths(tmp_path)
    capture_path = paths["attempt"] / "measurement" / "fixtures"
    for key, value in (
        ("TASK_FLASHQLA_BACKEND", "stale"),
        ("TASK_FLASHQLA_CAPTURE_MODE", "fixtures"),
        ("TASK_FLASHQLA_CAPTURE_DIR", "/stale/fixtures"),
        ("TASK_FLASHQLA_TILELANG_ROOT", "/stale/cache"),
        ("TASK_FLASHQLA_SOURCE_IDENTITY", "{}"),
        ("TASK_FLASHQLA_ROUTING_METADATA", "original"),
        ("TASK_FLASHQLA_CROSS_ENTROPY", "original"),
        ("TILELANG_CACHE_DIR", "/shared/home/cache"),
        ("TILELANG_TMP_DIR", "/shared/home/cache/tmp"),
    ):
        monkeypatch.setenv(key, value)

    env = bridge_run._build_environment_hook(
        base,
        backend="fla",
        capture_fixtures=capture_path,
    )(paths)

    attempt = paths["attempt"]
    tilelang_root = attempt / "cache/tilelang"
    assert env["TASK_FLASHQLA_BACKEND"] == "fla"
    assert env["TASK_FLASHQLA_CAPTURE_MODE"] == "fixtures"
    assert env["TASK_FLASHQLA_ROUTING_METADATA"] == "fused"
    assert env["TASK_FLASHQLA_CROSS_ENTROPY"] == "stable"
    assert env["TASK_FLASHQLA_CAPTURE_DIR"] == str(capture_path)
    assert capture_path.is_dir()
    assert env["TASK_FLASHQLA_TILELANG_ROOT"] == str(tilelang_root)
    assert env["TASK_COMPILER_CACHE_ROOT"] == str((attempt / "cache").resolve())
    assert env["TILELANG_CACHE_DIR"] == str(tilelang_root)
    assert env["TILELANG_TMP_DIR"] == str(tilelang_root / "tmp")
    pythonpath = env["PYTHONPATH"].split(os.pathsep)
    staged_fla = str((base.GIFT_ROOT / "contrib/flash-linear-attention").resolve())
    staged_qla = str((base.GIFT_ROOT / "contrib/flash-qla").resolve())
    assert pythonpath[:2] == [staged_fla, staged_qla]
    assert pythonpath[2:4] == ["/existing/entry-one", "/existing/entry-two"]
    assert str(base.GIFT_ROOT.resolve()) in pythonpath
    assert str(bridge_dir) in pythonpath
    assert str(base.HELPER_DIR) in pythonpath
    assert (tilelang_root / "rank_7" / "tmp").is_dir()

    receipt = json.loads((attempt / "flash-qla-bridge.json").read_text())
    assert receipt["backend_variant"] == "fla"
    assert receipt["routing_metadata"] == "fused"
    assert receipt["cross_entropy"] == "stable"
    assert receipt["capture_mode"] == "fixtures"
    assert receipt["tilelang_cache_policy"] == "fresh"
    assert receipt["tilelang_rank_directories"]["7"]["cache"] == str(
        tilelang_root / "rank_7"
    )
    assert receipt["source_identity"]["backend_module"]["sha256"]


def test_environment_hook_leaves_no_capture_override_for_full_training(
    bridge_run: Any, tmp_path: Path
) -> None:
    bridge_dir = _patch_bridge_paths(bridge_run, tmp_path)
    base = _fake_base(tmp_path)
    paths = _fake_paths(tmp_path)
    env = bridge_run._build_environment_hook(
        base,
        backend="fused_gate",
        capture_fixtures=None,
    )(paths)
    assert env["TASK_FLASHQLA_BACKEND"] == "fused_gate"
    assert env["TASK_FLASHQLA_CAPTURE_MODE"] == "disabled"
    assert env["TASK_FLASHQLA_ROUTING_METADATA"] == "fused"
    assert env["TASK_FLASHQLA_CROSS_ENTROPY"] == "stable"
    assert "TASK_FLASHQLA_CAPTURE_DIR" not in env
    assert "TILELANG_CACHE_DIR" in env
    assert "TILELANG_TMP_DIR" in env
    assert str(bridge_dir) in env["PYTHONPATH"].split(os.pathsep)


def test_selection_arguments_roundtrip_to_each_rank_environment(
    bridge_run: Any,
    bridge_measure: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_bridge_paths(bridge_run, tmp_path)
    base = _fake_base(tmp_path)
    paths = _fake_paths(tmp_path, "native-fusion-control")
    env = bridge_run._build_environment_hook(
        base,
        backend="external_gate",
        capture_fixtures=None,
        routing_metadata="original",
        cross_entropy="original",
    )(paths)

    assert env["TASK_FLASHQLA_ROUTING_METADATA"] == "original"
    assert env["TASK_FLASHQLA_CROSS_ENTROPY"] == "original"
    receipt = json.loads((paths["attempt"] / "flash-qla-bridge.json").read_text())
    assert receipt["routing_metadata"] == "original"
    assert receipt["cross_entropy"] == "original"
    for key in (
        "TASK_FLASHQLA_BACKEND",
        "TASK_FLASHQLA_CAPTURE_MODE",
        "TASK_FLASHQLA_SOURCE_IDENTITY",
        "TASK_FLASHQLA_ROUTING_METADATA",
        "TASK_FLASHQLA_CROSS_ENTROPY",
    ):
        monkeypatch.setenv(key, env[key])
    monkeypatch.delenv("TASK_FLASHQLA_CAPTURE_DIR", raising=False)
    assert bridge_measure._bridge_configuration()[-2:] == ("original", "original")


def test_environment_hook_reuses_existing_tilelang_cache_only_for_selected_root(
    bridge_run: Any, tmp_path: Path
) -> None:
    _patch_bridge_paths(bridge_run, tmp_path)
    base = _fake_base(tmp_path)
    paths = _fake_paths(tmp_path, "reuse-tilelang")
    selected_cache = tmp_path / "runs/overnight-streaming-perf/artifacts/raw/producer/cache"
    selected_cache.mkdir(parents=True)
    paths["cache"] = selected_cache
    existing_root = selected_cache / "tilelang"
    existing_root.mkdir()
    (existing_root / "tmp").mkdir()
    for rank in range(8):
        rank_root = existing_root / f"rank_{rank}"
        rank_root.mkdir()
        (rank_root / "tmp").mkdir()

    env = bridge_run._build_environment_hook(
        base, backend="external_gate", capture_fixtures=None
    )(paths)

    receipt = json.loads((paths["attempt"] / "flash-qla-bridge.json").read_text())
    assert env["TASK_COMPILER_CACHE_ROOT"] == str(selected_cache.resolve())
    assert env["TASK_FLASHQLA_TILELANG_ROOT"] == str(existing_root)
    assert receipt["tilelang_cache_policy"] == "reuse"
    assert receipt["tilelang_cache_root"] == str(existing_root)


def test_environment_hook_rejects_symlink_in_reused_tilelang_cache(
    bridge_run: Any, tmp_path: Path
) -> None:
    _patch_bridge_paths(bridge_run, tmp_path)
    base = _fake_base(tmp_path)
    paths = _fake_paths(tmp_path, "reuse-tilelang-symlink")
    selected_cache = tmp_path / "runs/overnight-streaming-perf/artifacts/raw/producer/cache"
    selected_cache.mkdir(parents=True)
    outside = tmp_path / "outside-tilelang"
    outside.mkdir()
    (selected_cache / "tilelang").symlink_to(outside, target_is_directory=True)
    paths["cache"] = selected_cache

    with pytest.raises(RuntimeError, match="must not be a symlink"):
        bridge_run._build_environment_hook(
            base, backend="external_gate", capture_fixtures=None
        )(paths)


def _measurement_attempt(root: Path, run_id: str, rank: int = 3) -> tuple[Path, Path]:
    attempt = root / "runs/flash-qla/artifacts/raw" / run_id
    measurement = attempt / "measurement"
    measurement.mkdir(parents=True)
    triton_rank = attempt / f"cache/triton/rank_{rank}"
    triton_rank.mkdir(parents=True)
    tilelang_rank = attempt / f"cache/tilelang/rank_{rank}"
    (tilelang_rank / "tmp").mkdir(parents=True)
    return attempt, measurement


def _set_rank_environment(monkeypatch: pytest.MonkeyPatch, values: dict[str, str]) -> None:
    monkeypatch.setenv("TASK_FLASHQLA_ROUTING_METADATA", "fused")
    monkeypatch.setenv("TASK_FLASHQLA_CROSS_ENTROPY", "stable")
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    tilelang_root = values.get("TASK_FLASHQLA_TILELANG_ROOT")
    if tilelang_root is not None and "TASK_COMPILER_CACHE_ROOT" not in values:
        monkeypatch.setenv("TASK_COMPILER_CACHE_ROOT", str(Path(tilelang_root).parent))


def test_startup_sets_device_and_rank_cache_before_backend_import(
    bridge_measure: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(bridge_measure, "TASK_ROOT", tmp_path)
    attempt, measurement = _measurement_attempt(tmp_path, "startup-order")
    tilelang_root = attempt / "cache/tilelang"
    _set_rank_environment(
        monkeypatch,
        {
            "TASK_FLASHQLA_BACKEND": "external_gate",
            "TASK_FLASHQLA_CAPTURE_MODE": "disabled",
            "TASK_FLASHQLA_TILELANG_ROOT": str(tilelang_root),
            "TASK_FLASHQLA_SOURCE_IDENTITY": '{"task_git_head":"deadbeef"}',
            "RANK": "3",
            "LOCAL_RANK": "3",
            "WORLD_SIZE": "8",
            "TRITON_CACHE_DIR": str(attempt / "cache/triton/rank_3"),
            "TILELANG_CACHE_DIR": str(tilelang_root),
            "TILELANG_TMP_DIR": str(tilelang_root / "tmp"),
        },
    )
    native_modules = _fake_native_modules()
    monkeypatch.setattr(
        bridge_measure, "_load_native_modules", lambda: native_modules
    )
    events: list[str] = []

    class FakeCuda:
        def set_device(self, rank: int) -> None:
            assert rank == 3
            events.append("set_device")

        def current_device(self) -> int:
            events.append("current_device")
            return 3

    fake_torch = SimpleNamespace(cuda=FakeCuda())
    adapter = SimpleNamespace(options={"output_dir": measurement})
    original_values = (
        os.environ.get("TILELANG_CACHE_DIR"),
        os.environ.get("TILELANG_TMP_DIR"),
    )

    def backend_loader() -> Any:
        events.append("backend_import")
        assert os.environ["TILELANG_CACHE_DIR"] == str(tilelang_root / "rank_3")
        assert os.environ["TILELANG_TMP_DIR"] == str(tilelang_root / "rank_3" / "tmp")
        return lambda variant: (events.append(f"install:{variant}") or {"variant": variant})

    def native_main(argv: list[str], received_adapter: Any) -> str:
        assert argv == ["config.yaml", "--model.test=true"]
        assert received_adapter is adapter
        native_modules[3].CrossEntropyLoss()
        native_modules[3].MTPLoss()
        events.append("native_main")
        return "done"

    result = bridge_measure._bootstrap_and_run(
        ["config.yaml", "--model.test=true"],
        adapter,
        native_main,
        torch_module=fake_torch,
        backend_installer_loader=backend_loader,
    )
    assert result == "done"
    assert events == [
        "set_device",
        "current_device",
        "backend_import",
        "install:external_gate",
        "native_main",
    ]
    assert os.environ.get("TILELANG_CACHE_DIR") == original_values[0]
    assert os.environ.get("TILELANG_TMP_DIR") == original_values[1]
    receipt = json.loads(
        (measurement / "flash-qla-bootstrap-rank-03.json").read_text()
    )
    assert receipt["cuda_device"] == 3
    assert receipt["installer_identity"] == {"variant": "external_gate"}
    assert receipt["tilelang_cache_dir"] == str(tilelang_root / "rank_3")
    assert receipt["routing_metadata"] == "fused"
    assert receipt["cross_entropy"] == "stable"
    assert receipt["routing_metadata_callable"]["object_id"] == id(
        native_modules[2].indices_to_map
    )
    assert receipt["cross_entropy_callable"]["object_id"] == id(
        native_modules[4].cross_entropy_loss
    )
    assert receipt["native_main_completed"] is True
    assert [item["class"].rsplit(".", maxsplit=1)[-1] for item in receipt[
        "constructed_loss_callables"
    ]] == ["CrossEntropyLoss", "MTPLoss"]
    assert all(
        item["callable"]["object_id"] == receipt["cross_entropy_callable"]["object_id"]
        for item in receipt["constructed_loss_callables"]
    )
    assert native_modules[0].indices_to_map is native_modules[1].indices_to_map
    assert native_modules[4].cross_entropy_loss.__module__ == (
        "torchtitan.experiments.four_fusion.stable_ce"
    )


def test_configure_tilelang_cache_accepts_selected_reused_root(
    bridge_measure: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(bridge_measure, "TASK_ROOT", tmp_path)
    _, measurement = _measurement_attempt(tmp_path, "selected-reuse", rank=4)
    selected_cache = tmp_path / "runs/overnight-streaming-perf/artifacts/raw/producer/cache"
    tilelang_root = selected_cache / "tilelang"
    rank_root = tilelang_root / "rank_4"
    (rank_root / "tmp").mkdir(parents=True)
    monkeypatch.setenv("TASK_COMPILER_CACHE_ROOT", str(selected_cache))
    monkeypatch.setenv("TASK_FLASHQLA_TILELANG_ROOT", str(tilelang_root))

    cache_dir, tmp_dir = bridge_measure._configure_tilelang_cache(
        SimpleNamespace(options={"output_dir": measurement}), 4
    )

    assert cache_dir == rank_root.resolve()
    assert tmp_dir == (rank_root / "tmp").resolve()
    assert os.environ["TILELANG_CACHE_DIR"] == str(rank_root.resolve())
    assert os.environ["TILELANG_TMP_DIR"] == str((rank_root / "tmp").resolve())


@pytest.mark.parametrize(
    ("routing_metadata", "cross_entropy"),
    [
        ("original", "original"),
        ("original", "stable"),
        ("fused", "original"),
        ("fused", "stable"),
    ],
)
def test_native_fusion_bindings_use_selected_callable_and_restore(
    bridge_measure: Any,
    monkeypatch: pytest.MonkeyPatch,
    routing_metadata: str,
    cross_entropy: str,
) -> None:
    native_modules = _fake_native_modules()
    monkeypatch.setattr(
        bridge_measure, "_load_native_modules", lambda: native_modules
    )
    hybridep, original_routing, fused_routing, loss_module, stable_ce = native_modules
    original_hybrid_route = hybridep.indices_to_map
    original_stable_ce = stable_ce.cross_entropy_loss
    original_initializers = {
        name: getattr(loss_module, name).__init__
        for name in ("CrossEntropyLoss", "ChunkedCELoss")
    }
    expected_route = getattr(
        original_routing if routing_metadata == "original" else fused_routing,
        "indices_to_map",
    )
    expected_ce = (
        loss_module.cross_entropy_loss
        if cross_entropy == "original"
        else original_stable_ce
    )

    with bridge_measure._bind_native_fusion_choices(
        routing_metadata, cross_entropy
    ) as selection:
        assert hybridep.indices_to_map is expected_route
        assert stable_ce.cross_entropy_loss is expected_ce
        assert selection["routing_metadata"]["callable"]["object_id"] == id(
            expected_route
        )
        assert selection["cross_entropy"]["callable"]["object_id"] == id(
            expected_ce
        )
        assert len(selection["routing_metadata"]["callable"]["sha256"]) == 64
        assert len(selection["cross_entropy"]["callable"]["sha256"]) == 64

        loss_module.CrossEntropyLoss()
        loss_module.ChunkedCELoss()
        loss_module.MTPLoss()
        assert len(selection["constructed_loss_callables"]) == 3
        assert all(
            item["callable"]["object_id"] == id(expected_ce)
            for item in selection["constructed_loss_callables"]
        )
        assert selection["constructed_loss_callables"][-1]["class"].endswith(
            ".MTPLoss"
        )

    assert hybridep.indices_to_map is original_hybrid_route
    assert stable_ce.cross_entropy_loss is original_stable_ce
    for name, initializer in original_initializers.items():
        assert getattr(loss_module, name).__init__ is initializer


def test_native_binding_unwraps_compiled_loss_callable(bridge_measure: Any) -> None:
    selected = lambda: None

    class CompiledCallable:
        def __init__(self) -> None:
            self._torchdynamo_orig_callable = selected

        def __call__(self) -> None:
            return None

    assert bridge_measure._unwrap_compiled_callable(CompiledCallable()) is selected


def test_native_binding_fails_fast_on_wrong_constructed_loss_callable(
    bridge_measure: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    native_modules = _fake_native_modules()
    monkeypatch.setattr(
        bridge_measure, "_load_native_modules", lambda: native_modules
    )
    loss_module = native_modules[3]
    original_init = loss_module.CrossEntropyLoss.__init__

    def wrong_init(self: Any) -> None:
        self.fn = loss_module.cross_entropy_loss

    loss_module.CrossEntropyLoss.__init__ = wrong_init
    try:
        with bridge_measure._bind_native_fusion_choices("fused", "stable"):
            with pytest.raises(RuntimeError, match="unexpected cross-entropy callable"):
                loss_module.CrossEntropyLoss()
    finally:
        loss_module.CrossEntropyLoss.__init__ = original_init


def test_configure_tilelang_cache_rejects_root_outside_task(
    bridge_measure: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(bridge_measure, "TASK_ROOT", tmp_path)
    _, measurement = _measurement_attempt(tmp_path, "outside-root")
    outside = tmp_path.parent / "outside-cache"
    outside.mkdir()
    monkeypatch.setenv("TASK_COMPILER_CACHE_ROOT", str(outside))
    monkeypatch.setenv("TASK_FLASHQLA_TILELANG_ROOT", str(outside / "tilelang"))

    with pytest.raises(RuntimeError, match="inside the task root"):
        bridge_measure._configure_tilelang_cache(
            SimpleNamespace(options={"output_dir": measurement}), 0
        )


def test_capture_rank1_accepts_directory_created_before_startup(
    bridge_measure: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(bridge_measure, "TASK_ROOT", tmp_path)
    attempt, measurement = _measurement_attempt(tmp_path, "capture-order", rank=1)
    tilelang_root = attempt / "cache/tilelang"
    capture_dir = measurement / "fixtures"
    # Rank 0 has already created the shared capture directory before rank 1 starts.
    capture_dir.mkdir()
    _set_rank_environment(
        monkeypatch,
        {
            "TASK_FLASHQLA_BACKEND": "fla",
            "TASK_FLASHQLA_CAPTURE_MODE": "fixtures",
            "TASK_FLASHQLA_CAPTURE_DIR": str(capture_dir),
            "TASK_FLASHQLA_TILELANG_ROOT": str(tilelang_root),
            "TASK_FLASHQLA_SOURCE_IDENTITY": '{"task_git_head":"deadbeef"}',
            "RANK": "1",
            "LOCAL_RANK": "1",
            "WORLD_SIZE": "8",
            "TRITON_CACHE_DIR": str(attempt / "cache/triton/rank_1"),
            "TILELANG_CACHE_DIR": str(tilelang_root),
            "TILELANG_TMP_DIR": str(tilelang_root / "tmp"),
        },
    )
    native_modules = _fake_native_modules()
    monkeypatch.setattr(
        bridge_measure, "_load_native_modules", lambda: native_modules
    )
    events: list[Any] = []

    class FakeCuda:
        def set_device(self, rank: int) -> None:
            events.append(("set_device", rank))

        def current_device(self) -> int:
            return 1

    def capture_loader() -> Any:
        events.append("capture_import")

        def install_capture(output_dir: Path, *, max_fixtures: int) -> dict[str, Any]:
            events.append(("capture_install", output_dir, max_fixtures))
            return {"capture": "native_fla"}

        return install_capture

    def forbidden_backend_loader() -> Any:
        pytest.fail("capture mode must not install the regular FLA backend")

    bridge_measure._bootstrap_and_run(
        [],
        SimpleNamespace(options={"output_dir": measurement}),
        lambda argv, adapter: events.append("native_main"),
        torch_module=SimpleNamespace(cuda=FakeCuda()),
        backend_installer_loader=forbidden_backend_loader,
        capture_installer_loader=capture_loader,
    )
    assert events == [
        ("set_device", 1),
        "capture_import",
        ("capture_install", capture_dir, 6),
        "native_main",
    ]
    receipt = json.loads(
        (measurement / "flash-qla-bootstrap-rank-01.json").read_text()
    )
    assert receipt["capture_mode"] == "fixtures"
    assert receipt["capture_fixture_dir"] == str(capture_dir)


def test_measure_main_restores_native_startup_hook(
    bridge_measure: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = lambda argv, adapter: None
    base = SimpleNamespace(_run_native_main=original)
    observed: list[str] = []

    def base_main(argv: list[str] | None = None) -> int:
        assert base._run_native_main is not original
        observed.append("patched")
        return 17

    base.main = base_main
    monkeypatch.setattr(bridge_measure, "_load_base_measure", lambda: base)
    assert bridge_measure.main(["--run-id", "in-process"]) == 17
    assert observed == ["patched"]
    assert base._run_native_main is original

