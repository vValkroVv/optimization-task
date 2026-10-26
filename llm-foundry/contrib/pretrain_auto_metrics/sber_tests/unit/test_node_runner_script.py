import importlib.util
import sys
import tempfile
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit


def _load_node_runner_module():
    module_path = Path(__file__).resolve().parents[2] / "node_runner" / "node_runner.py"
    src_root = module_path.parent
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    # Явно регистрируем bare-namespace-пакет `src`, чтобы `from src.X`
    # резолвился на node_runner/src/ независимо от sys.path / import-mode
    # квирков (CI vs. локальный pytest collection order может оставить `src`
    # перекрытым или не зарезолвленным, если фейковый submodule ниже
    # регистрируется без родителя).
    src_pkg_dir = src_root / "src"
    if "src" not in sys.modules:
        src_module = types.ModuleType("src")
        src_module.__path__ = [str(src_pkg_dir)]
        sys.modules["src"] = src_module

    # Avoid importing heavy convertation dependencies (transformers/scipy) in unit tests.
    if "src.convertation_manager" not in sys.modules:
        fake = types.ModuleType("src.convertation_manager")

        class _ConvertationManager:  # placeholder, never instantiated by tests
            pass

        fake.ConvertationManager = _ConvertationManager
        sys.modules["src.convertation_manager"] = fake

    spec = importlib.util.spec_from_file_location(
        "node_runner_script_test", module_path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["node_runner_script_test"] = module
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


class _FakeEngine:
    def __init__(self, events: list, unit_ranks: tuple = ((0,),)):
        self._events = events
        self._unit_ranks = unit_ranks
        self.stopped = 0
        self.attached = []

    def start(self, *args, **kwargs):
        self._events.append("engine start")

    def units(self):
        return [SimpleNamespace(ranks=ranks) for ranks in self._unit_ranks]

    def attach(self, units):
        self._events.append("engine attach")
        self.attached.append(units)

    def close(self):
        self.stopped += 1


class _FakeNvidiaSmiSampler:
    def __init__(self):
        self.started = 0
        self.stopped = 0

    def start(self):
        self.started += 1

    def stop(self):
        self.stopped += 1


class _FakeLogger:
    def __init__(self):
        self.inference_log_file = Path("/tmp/inference.log")
        self.router_log_file = Path("/tmp/router.log")
        self.obm_proxy_log_file = Path("/tmp/obm_proxy.log")
        self.convertation_log_file = Path("/tmp/convertation.log")
        self.autometrics_log_file = Path("/tmp/autometrics.log")
        self.setup_calls = []

    def setup(self, *args, **kwargs):
        self.setup_calls.append((args, kwargs))


def _build_runner(
    mode: str = "autometrics", is_master: bool = True, engine_units: tuple = ((0,),)
):
    mod = _load_node_runner_module()
    events: list[str] = []

    autometrics = SimpleNamespace(
        globals=SimpleNamespace(
            orkestration_port=29999,
            sglang_communication_port=61000,
            delete_converted_hf=True,
        ),
        router=SimpleNamespace(host="0.0.0.0", port=30000),
        inference=SimpleNamespace(tp=1),
        obm_proxy=SimpleNamespace(chat_template=None),
        convertation=SimpleNamespace(mappers=["map_fsdp_to_hf"]),
        preparation=SimpleNamespace(skip=True),
        quantization=None,
        model_info=SimpleNamespace(model_class="sft", size="lite", username="u"),
        evaluators=[],
        secondary_inference={},
    )
    cfg = SimpleNamespace(
        autometrics=autometrics,
        tensorboard_path=Path("/tmp/tb"),
        run_name="run",
        is_external_api_mode=False,
        runs_without_checkpoint=False,
        is_replay_mode=False,
        apply_defaults=lambda **kwargs: None,
    )
    config_manager = SimpleNamespace(
        setup=lambda checkpoint_path, **kwargs: None,
        job_id="test-job",
        node_name="mpimaster-0" if is_master else "mpiworker-0",
        global_nnodes=2,
        global_node_rank=0 if is_master else 1,
        is_master=is_master,
        global_dist_init_host="127.0.0.1",
        artifacts_path=Path("/tmp/artifacts"),
        checkpoint_path=Path("/tmp/checkpoint"),
        config=cfg,
        gpus_per_node=8,
        is_slurm=False,
    )

    orkestration_manager = SimpleNamespace(
        is_master=is_master,
        is_worker=not is_master,
        setup=lambda *args, **kwargs: None,
        get_registered_nodes=lambda: {0: "10.0.0.1", 1: "10.0.0.2"},
        wait_barrier=events.append,
        cleanup=lambda: None,
    )

    logger = _FakeLogger()
    engine = _FakeEngine(events, engine_units)
    inference_setup_calls = []

    def _setup_inference(**kwargs):
        events.append("inference setup")
        inference_setup_calls.append(kwargs)

    def _setup_inference_engine():
        events.append("base engine setup")
        return engine

    inference_manager = SimpleNamespace(
        setup=_setup_inference,
        setup_inference_engine=_setup_inference_engine,
        setup_calls=inference_setup_calls,
    )
    mod.InferenceManager = inference_manager

    autometrics_setup_calls = []
    cleanup_calls = []

    def _setup_auto_metrics(**kwargs):
        events.append("metrics setup")
        autometrics_setup_calls.append(kwargs)

    auto_metrics_manager = SimpleNamespace(
        setup=_setup_auto_metrics,
        run_pipeline=lambda: events.append("metrics pipeline"),
        cleanup=lambda: cleanup_calls.append(True),
        setup_calls=autometrics_setup_calls,
        cleanup_calls=cleanup_calls,
    )

    convertation_setup_calls = []
    convertation_manager = SimpleNamespace(
        setup=lambda **kwargs: convertation_setup_calls.append(kwargs),
        run_pipeline=lambda: events.append("convert"),
        hf_checkpoint_path=Path("/tmp/hf_checkpoint"),
        originaly_hf=False,
        originally_fsdp=False,
        setup_calls=convertation_setup_calls,
    )

    runner = mod.NodeRunner(
        checkpoint_path="/tmp/checkpoint",
        config_path=None,
        mode=mode,
        config_manager=config_manager,
        logger=logger,
        orkestration_manager=orkestration_manager,
        auto_metrics_manager=auto_metrics_manager,
        convertation_manager=convertation_manager,
    )
    mod.Trace._nvidia_smi = _FakeNvidiaSmiSampler()
    return mod, runner, engine, events


def _serve_forever_raises(monkeypatch: pytest.MonkeyPatch, mod) -> None:
    def _sleep(_seconds):
        raise InterruptedError()

    monkeypatch.setattr(mod.time, "sleep", _sleep)


CONVERTED = ["REGISTRATION", "convert", "inference setup"]
INFERENCE_PREFIX = CONVERTED + ["base engine setup", "INFERENCE_START"]


class TestRun:
    @pytest.mark.parametrize("is_master", [True, False])
    @pytest.mark.parametrize(
        ("mode", "expected"),
        [
            (
                "autometrics",
                CONVERTED
                + ["metrics setup", "INFERENCE_START", "metrics pipeline", "CLEANUP"],
            ),
            ("convertation", CONVERTED + ["INFERENCE_START", "CLEANUP"]),
            ("noop", CONVERTED + ["INFERENCE_START", "CLEANUP"]),
        ],
    )
    def test_only_autometrics_sets_up_and_runs_the_metrics_pipeline_between_barriers(
        self, mode, is_master, expected
    ):
        _, runner, _, events = _build_runner(mode=mode, is_master=is_master)

        runner.run()

        assert events == expected

    @pytest.mark.parametrize("is_master", [True, False])
    def test_external_api_skips_convertation_and_serves_from_artifacts(self, is_master):
        mod, runner, _, events = _build_runner(is_master=is_master)
        runner.config_manager.config.runs_without_checkpoint = True

        runner.run()

        assert events == [
            "REGISTRATION",
            "inference setup",
            "metrics setup",
            "INFERENCE_START",
            "metrics pipeline",
            "CLEANUP",
        ]
        assert runner.convertation_manager.setup_calls == []
        assert mod.InferenceManager.setup_calls[0]["hf_checkpoint_path"] == Path(
            "/tmp/artifacts"
        )

    def test_hands_the_checkpoint_and_managers_down_the_pipeline(self):
        mod, runner, _, _ = _build_runner()

        runner.run()

        [convertation_kwargs] = runner.convertation_manager.setup_calls
        assert convertation_kwargs["checkpoint_path"] == Path("/tmp/checkpoint")
        assert (convertation_kwargs["nnodes"], convertation_kwargs["node_rank"]) == (
            2,
            0,
        )
        [inference_kwargs] = mod.InferenceManager.setup_calls
        autometrics = runner.config_manager.config.autometrics
        assert inference_kwargs["inference_config"] is autometrics.inference
        assert inference_kwargs["obm_proxy_config"] is autometrics.obm_proxy
        assert inference_kwargs["hf_checkpoint_path"] == Path("/tmp/hf_checkpoint")
        [metrics_kwargs] = runner.auto_metrics_manager.setup_calls
        assert metrics_kwargs["config_manager"] is runner.config_manager
        assert metrics_kwargs["convertation_manager"] is runner.convertation_manager
        assert "base_inference_engine" not in metrics_kwargs

    def test_passes_updated_checkpoint_path_to_inference_setup(self):
        mod, runner, _, _ = _build_runner(mode="autometrics")
        fp8_path = Path("/tmp/artifacts/hf_checkpoint_fp8")

        def _run_pipeline():
            runner.convertation_manager.hf_checkpoint_path = fp8_path

        runner.convertation_manager.run_pipeline = _run_pipeline
        runner.run()
        assert mod.InferenceManager.setup_calls[0]["hf_checkpoint_path"] == fp8_path

    def test_arms_the_streams_before_the_barrier_endpoint(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """The barrier handler emits events, so the orkestration app must not start
        serving before the trace is set up."""
        mod, runner, _, _ = _build_runner()
        order = []
        monkeypatch.setattr(
            mod.Trace,
            "setup",
            classmethod(lambda cls, config_manager: order.append("trace")),
        )
        runner.orkestration_manager.setup = lambda *args: order.append("orkestration")

        runner.run()

        assert order == ["trace", "orkestration"]

    @pytest.mark.parametrize(
        "engine_units",
        [((0,),), ((0,), (1,)), ((),)],
        ids=["gpu", "two-units", "nodeless"],
    )
    def test_inference_master_starts_and_attaches_the_engine_then_serves(
        self, monkeypatch, engine_units
    ):
        mod, runner, engine, events = _build_runner(
            mode="inference", engine_units=engine_units
        )
        _serve_forever_raises(monkeypatch, mod)

        with pytest.raises(InterruptedError):
            runner.run()

        assert events == INFERENCE_PREFIX + [
            "engine start",
            "INFERENCE_READY",
            "engine attach",
            "ROUTER_START",
        ]
        assert engine.attached == [engine.units()]

    @pytest.mark.parametrize(
        ("engine_units", "boot"),
        [(((0,),), ["engine start"]), (((),), [])],
        ids=["gpu", "nodeless"],
    )
    def test_inference_worker_boots_only_gpu_engines_and_waits_for_stop(
        self, engine_units, boot
    ):
        _, runner, _, events = _build_runner(
            mode="inference", is_master=False, engine_units=engine_units
        )

        runner.run()

        assert events == INFERENCE_PREFIX + boot + [
            "INFERENCE_READY",
            "ROUTER_START",
            "INFERENCE_STOP",
        ]


class TestCleanup:
    @pytest.mark.parametrize(
        ("is_master", "expected_sleeps"),
        [(True, [5]), (False, [])],
    )
    def test_cleanup(self, monkeypatch, is_master, expected_sleeps):
        mod, runner, engine, _ = _build_runner(is_master=is_master)
        runner.base_inference_engine = engine
        sleeps = []
        monkeypatch.setattr(mod.time, "sleep", sleeps.append)
        runner.cleanup()
        assert engine.stopped == 1
        assert sleeps == expected_sleeps
        assert runner.auto_metrics_manager.cleanup_calls == [True]

    def test_ignores_base_engine_stop_errors(self, monkeypatch):
        mod, runner, _, _ = _build_runner()

        class _BrokenEngine:
            def close(self):
                raise RuntimeError("boom")

        runner.base_inference_engine = _BrokenEngine()
        monkeypatch.setattr(mod.time, "sleep", lambda _seconds: None)
        runner.cleanup()

    def test_logs_success_after_stopping_base_engine(self, monkeypatch):
        mod, runner, engine, _ = _build_runner()
        runner.base_inference_engine = engine
        infos = []
        monkeypatch.setattr(mod.Logger, "info", infos.append)
        monkeypatch.setattr(mod.time, "sleep", lambda _seconds: None)
        runner.cleanup()
        assert engine.stopped == 1
        assert any("Cleanup: inference engine stopped" in m for m in infos)

    def test_without_base_inference_engine(self, monkeypatch):
        mod, runner, _, _ = _build_runner()
        runner.base_inference_engine = None
        monkeypatch.setattr(mod.time, "sleep", lambda _seconds: None)
        runner.cleanup()

    def test_metrics_marker_lands_after_every_engine_stops(self, monkeypatch):
        """The marker claims the recording is complete, but each scraper keeps
        appending until its engine stops. An unguarded _cleanup_artifacts (rmtree
        over NFS) raising must not strand the scrapers behind the marker."""
        mod, runner, _, _ = _build_runner()
        order = []
        monkeypatch.setattr(mod.time, "sleep", lambda _seconds: None)
        monkeypatch.setattr(
            mod.Trace,
            "close",
            classmethod(lambda cls, status: order.append(f"trace:{status}")),
        )
        runner.auto_metrics_manager.cleanup = lambda: order.append("phase engines")
        runner.base_inference_engine = SimpleNamespace(
            close=lambda: order.append("base engine")
        )

        def _raise():
            raise RuntimeError("nfs is gone")

        monkeypatch.setattr(runner, "_cleanup_artifacts", _raise)

        with pytest.raises(RuntimeError, match="nfs is gone"):
            runner.cleanup()

        assert order == ["phase engines", "base engine", "trace:ok"]


class TestCleanupArtifacts:
    def test_removes_converted_hf_when_allowed(self):
        _, runner, _, _ = _build_runner()
        runner.convertation_manager.originaly_hf = False
        runner.config_manager.config.autometrics.globals.delete_converted_hf = True
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "hf"
            p.mkdir()
            runner.convertation_manager.hf_checkpoint_path = p
            runner._cleanup_artifacts()
            assert not p.exists()

    def test_early_returns_keep_directory(self, tmp_path: Path):
        _, runner, _, _ = _build_runner()
        p = tmp_path / "hf"
        p.mkdir()
        runner.convertation_manager.hf_checkpoint_path = p

        runner.convertation_manager.originaly_hf = True
        runner._cleanup_artifacts()
        assert p.exists()

        runner.convertation_manager.originaly_hf = False
        runner.config_manager.config.autometrics.globals.delete_converted_hf = False
        runner._cleanup_artifacts()
        assert p.exists()

        runner.config_manager.config.autometrics.globals.delete_converted_hf = True
        runner.mode = "convertation"
        runner._cleanup_artifacts()
        assert p.exists()

    def test_originally_fsdp_keeps_directory(self, tmp_path: Path):
        _, runner, _, _ = _build_runner()
        p = tmp_path / "hf"
        p.mkdir()
        runner.convertation_manager.hf_checkpoint_path = p
        runner.convertation_manager.originaly_hf = False
        runner.convertation_manager.originally_fsdp = True
        runner.config_manager.config.autometrics.globals.delete_converted_hf = True
        runner._cleanup_artifacts()
        assert p.exists()

    def test_worker_does_not_touch_artifacts(self, tmp_path: Path):
        _, runner, _, _ = _build_runner(is_master=False)
        p = tmp_path / "hf"
        p.mkdir()
        runner.convertation_manager.hf_checkpoint_path = p
        runner._cleanup_artifacts()
        assert p.exists()

    def test_external_api_does_not_touch_artifacts(self, tmp_path: Path):
        _, runner, _, _ = _build_runner()
        p = tmp_path / "hf"
        p.mkdir()
        runner.config_manager.config.runs_without_checkpoint = True
        runner.convertation_manager.hf_checkpoint_path = p

        runner._cleanup_artifacts()

        assert p.exists()


class TestMainEntry:
    def _patch_manager_factories(self, mod, monkeypatch):
        for name in [
            "ConfigManager",
            "FastApiOrkestrationManager",
            "AutoMetricsManager",
            "ConvertationManager",
        ]:
            if hasattr(mod, name):
                monkeypatch.setattr(
                    mod,
                    name,
                    lambda *a, **k: SimpleNamespace(setup=lambda *x, **y: None),
                )
        # Logger is used both as a factory (Logger()) and via classmethods
        # (Logger.error on crash); the stub records error messages so callers
        # can assert the crash was logged. Returns that list.
        errors: list[str] = []

        class _StubLogger:
            def __init__(self, *a, **k):
                pass

            @staticmethod
            def setup(*a, **k):
                pass

            @staticmethod
            def info(*a, **k):
                pass

            @staticmethod
            def error(message, *a, **k):
                errors.append(message)

        monkeypatch.setattr(mod, "Logger", _StubLogger)
        return errors

    def test_main_runs_then_cleans_up(self, monkeypatch):
        mod = _load_node_runner_module()
        monkeypatch.setattr(
            mod, "parse_arguments", lambda: ("/tmp", "autometrics", None)
        )

        cleaned = []

        class _MockRunner:
            def __init__(self, **kwargs):
                pass

            def run(self):
                pass

            def cleanup(self, status="ok"):
                cleaned.append(status)

        monkeypatch.setattr(mod, "NodeRunner", _MockRunner)
        self._patch_manager_factories(mod, monkeypatch)
        mod.main()

        assert cleaned == ["ok"]

    def test_main_hands_the_crash_status_to_cleanup(self, monkeypatch):
        mod = _load_node_runner_module()
        monkeypatch.setattr(
            mod, "parse_arguments", lambda: ("/tmp", "autometrics", None)
        )
        cleaned = []

        class _CrashingRunner:
            def __init__(self, **kwargs):
                pass

            def run(self):
                raise RuntimeError("run failed")

            def cleanup(self, status="ok"):
                cleaned.append(status)

        monkeypatch.setattr(mod, "NodeRunner", _CrashingRunner)
        self._patch_manager_factories(mod, monkeypatch)
        with pytest.raises(RuntimeError, match="run failed"):
            mod.main()

        assert cleaned == ["error"]

    def test_crash_in_the_metrics_pipeline_still_closes_every_engine(self, monkeypatch):
        mod, runner, _, events = _build_runner()
        monkeypatch.setattr(
            mod, "parse_arguments", lambda: ("/tmp", "autometrics", None)
        )
        monkeypatch.setattr(mod, "NodeRunner", lambda **kwargs: runner)
        monkeypatch.setattr(mod.time, "sleep", lambda _seconds: None)
        monkeypatch.setattr(mod.Logger, "error", lambda *a, **k: None)
        monkeypatch.setattr(
            mod.Trace,
            "close",
            classmethod(lambda cls, status: events.append(f"trace {status}")),
        )
        runner.config_manager.config.autometrics.globals.delete_converted_hf = False

        def _crash():
            raise RuntimeError("scheduler crashed")

        runner.auto_metrics_manager.run_pipeline = _crash

        with pytest.raises(RuntimeError, match="scheduler crashed"):
            mod.main()

        assert "CLEANUP" not in events
        assert events[-1] == "trace error"
        assert runner.auto_metrics_manager.cleanup_calls == [True]

    def test_main_reraises_run_error_even_if_cleanup_fails(self, monkeypatch):
        mod = _load_node_runner_module()
        monkeypatch.setattr(
            mod, "parse_arguments", lambda: ("/tmp", "autometrics", None)
        )

        class _FailingRunner:
            def __init__(self, **kwargs):
                pass

            def run(self):
                raise RuntimeError("run failed")

            def cleanup(self, status="ok"):
                raise RuntimeError("cleanup failed")

        monkeypatch.setattr(mod, "NodeRunner", _FailingRunner)
        errors = self._patch_manager_factories(mod, monkeypatch)
        with pytest.raises(RuntimeError, match="run failed"):
            mod.main()

        assert any("Node runner crashed" in m and "run failed" in m for m in errors)

    def test_module_main_guard_executes_main(self):
        module_path = (
            Path(__file__).resolve().parents[2] / "node_runner" / "node_runner.py"
        )
        source = module_path.read_text(encoding="utf-8")
        guard = 'if __name__ == "__main__":  # pragma: no cover\n    main()\n'
        assert guard in source

        called = {"main": 0}
        ns = {"__name__": "not_main"}
        exec(compile(source, str(module_path), "exec"), ns)
        ns["main"] = lambda: called.__setitem__("main", called["main"] + 1)
        ns["__name__"] = "__main__"

        # Preserve original file line numbers so coverage maps stay correct.
        guard_line = source.splitlines().index(
            'if __name__ == "__main__":  # pragma: no cover'
        )
        padded_guard = ("\n" * guard_line) + guard
        exec(compile(padded_guard, str(module_path), "exec"), ns)
        assert called["main"] == 1
