import json
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from node_runner.src.config_model import (
    DEFAULT_PARALLEL_PER_INFERENCE_INSTANCE,
    DreddEvaluatorConfig,
    EvaluatorsConfig,
    ExternalApiInferenceConfig,
    JudgeEndpoint,
    ModelInfoConfig as ModelInfo,
    SglangInferenceConfig,
)
from node_runner.src.inference.engine import EngineLoad, EngineUnit
from node_runner.src.job_dispatcher import JD
from node_runner.src.job_dispatcher.measurement import (
    MeasurementFiles,
    MeasurementUpload,
)
from node_runner.src.metrics import metrics_manager
from node_runner.src.metrics.evaluators import dredd as dredd_module
from node_runner.src.metrics.evaluators.acc_len import AccLenEvaluator
from node_runner.src.metrics.evaluators.dredd import (
    DreddEvaluator,
    _load_summary_from_jsonl,
)
from node_runner.src.metrics.evaluators.interface import (
    MetricsEvaluator,
    ParallelInferenceEvaluator,
    ScoringMetricsEvaluator,
)
from node_runner.src.metrics.metrics_manager import (
    EVALUATED,
    EVALUATORS,
    AutoMetricsManager,
    DynamicScheduler,
    EngineOpLog,
    EngineSlot,
)

pytestmark = pytest.mark.unit

HANG_SECONDS = 10
IDLE = EngineLoad(0, 0, capacity=10, token_usage=0.0)
MODEL_INFO = ModelInfo(model_class="sft", size="lite", username="test")


def _wait_until(condition) -> None:
    deadline = time.monotonic() + HANG_SECONDS
    while not condition():
        if time.monotonic() > deadline:
            raise TimeoutError("condition never became true")
        time.sleep(0.002)


class _Server:
    def __init__(self, journal: list):
        self.journal = journal
        self.ops: list[dict] = []
        self.arrived: dict[str, set[int]] = {}
        self.expected: dict[str, int] = {}
        self.lock = threading.Lock()


class _Node:
    def __init__(self, server: _Server, rank: int):
        self.server = server
        self.global_node_rank = rank
        self.is_master = rank == 0

    def publish_op(self, op: dict) -> int:
        with self.server.lock:
            self.server.ops.append(op)
            index = len(self.server.ops) - 1
        ranks = tuple(op.get("ranks", ()))
        self.server.journal.append(("op", op["action"], op.get("engine"), ranks))
        return index

    def fetch_op(self, index: int) -> dict:
        _wait_until(lambda: len(self.server.ops) > index)
        return self.server.ops[index]

    def wait_barrier(self, stage: str, expected: int) -> bool:
        with self.server.lock:
            self.server.arrived.setdefault(stage, set()).add(self.global_node_rank)
            self.server.expected[stage] = expected
        _wait_until(lambda: self.barrier_released(stage))
        return True

    def barrier_released(self, stage: str) -> bool:
        with self.server.lock:
            arrived = len(self.server.arrived.get(stage, ()))
            return (
                stage in self.server.expected and arrived >= self.server.expected[stage]
            )


class _FakeEngine:
    def __init__(self, key: str, node: int, layout: list[tuple], factory):
        self.key = key
        self.node = node
        self.log_profile = key
        self.obm_proxy_manager = None
        self._factory = factory
        self._units = [
            EngineUnit(ranks=ranks, worker_urls=(f"http://{key}{list(ranks)}",))
            if ranks
            else EngineUnit(ranks=(), worker_urls=())
            for ranks in layout
        ]
        self.inference_instances = factory.instances.get(key, 1)
        self.base_url = f"http://{key}"
        self.direct_base_url = f"http://direct-{key}"
        self.worker_urls = [url for unit in self._units for url in unit.worker_urls]

    def units(self, from_tail: bool = False) -> list[EngineUnit]:
        return self._units[::-1] if from_tail else list(self._units)

    def _record(self, action: str) -> None:
        self._factory.journal.append((action, self.key, self.node))

    def start(self):
        self._record("up")
        if (self.key, self.node) in self._factory.failing_starts:
            raise RuntimeError(f"{self.key} did not start")

    def stop(self):
        self._record("down")

    def close(self):
        self._record("close")
        if self.key in self._factory.failing_closes:
            raise RuntimeError(f"{self.key} did not close")

    def attach(self, units):
        pass

    def detach(self, units):
        pass

    def load(self, units):
        return self._factory.loads.get(self.key, IDLE)


class _Engines:
    def __init__(self, journal: list):
        self.journal = journal
        self.node = 0
        self.nnodes = 2
        self.layouts: dict[str, list[tuple]] = {}
        self.loads: dict[str, EngineLoad] = {}
        self.instances: dict[str, int] = {}
        self.failing_starts: set[tuple[str, int]] = set()
        self.failing_closes: set[str] = set()
        self.built: list[tuple[str, int | None]] = []

    def setup_inference_engine(self):
        self.built.append((EVALUATED, None))
        return self._engine(EVALUATED)

    def setup_preset_inference_engine(self, model_name: str, judge_index: int):
        self.built.append((model_name, judge_index))
        return self._engine(model_name)

    def _engine(self, key: str) -> _FakeEngine:
        one_unit_per_node = [(rank,) for rank in range(self.nnodes)]
        layout = self.layouts.get(key, one_unit_per_node)
        return _FakeEngine(key, self.node, layout, self)


class _Config:
    def __init__(
        self,
        journal: list,
        name: str,
        judges: tuple = (),
        fail_on: tuple = (),
        seconds: float = 0.0,
    ):
        self.journal = journal
        self.name = name
        self.secondary_model_names = list(judges)
        self.fail_on = set(fail_on)
        self.seconds = seconds
        self.applied: dict | None = None
        self.setup_args: tuple | None = None
        self.applied_before_setup: bool | None = None

    def apply_defaults(self, **kwargs):
        self.applied = kwargs


class _Shooter(MetricsEvaluator):
    def setup(self, config, artifacts_path, hf_checkpoint_path, model_info):
        self.config = config
        self.autometric_runs_config = config
        config.setup_args = (config, artifacts_path, hf_checkpoint_path, model_info)
        config.applied_before_setup = config.applied is not None
        if "setup" in config.fail_on:
            raise RuntimeError(f"{config.name} setup failed")

    def run(self):
        self._stage("run")

    def run_on_model_name(self, model_name):
        self._stage(model_name)

    def _stage(self, stage: str) -> None:
        journal = self.config.journal
        journal.append(("run", self.config.name, stage))
        time.sleep(self.config.seconds)
        if stage in self.config.fail_on:
            raise RuntimeError(f"{self.config.name} {stage} failed")
        journal.append(("ran", self.config.name, stage))

    def get_metrics(self):
        return {self.config.name: 1.0}

    def after_run(self, metrics, uploaded=None):
        self.config.journal.append(("after_run", self.config.name))
        if "after_run" in self.config.fail_on:
            raise RuntimeError(f"{self.config.name} after_run failed")


class _ScoringShooter(_Shooter, ScoringMetricsEvaluator):
    def run_phase(self, phase):
        self._stage(phase)


class _ParallelShooter(_Shooter, ParallelInferenceEvaluator):
    pass


class _AccLenShooter(_Shooter, AccLenEvaluator):
    pass


class _NoStagesShooter(_Shooter):
    def stages(self):
        return []


def _shooter(
    monkeypatch: pytest.MonkeyPatch,
    journal: list,
    name: str,
    evaluator_class: type[MetricsEvaluator] = _Shooter,
    **behaviour,
) -> tuple[str, _Config]:
    monkeypatch.setitem(EVALUATORS, name, evaluator_class)
    return name, _Config(journal, name, **behaviour)


class _FakeEvaluator(MetricsEvaluator):
    METRICS_DIR_NAME = "fake"

    def __init__(self):
        self.metrics_dir = Path("metrics/fake")
        self.hf_checkpoint_path = Path("hf_checkpoint")
        self.measurement_files = MeasurementFiles(self.metrics_dir)
        self.after_run_args = None
        self.after_run_uploaded = None
        self.metrics = {"metric/a": 1.0}

    def setup(self, *args, **kwargs):
        pass

    def run(self):
        pass

    def get_metrics(self):
        return dict(self.metrics)

    def after_run(self, metrics, uploaded=None):
        self.after_run_args = metrics
        self.after_run_uploaded = uploaded

    def measured_files_mtime_ns(self):
        return self.measurement_files.mtime_ns()


class _FakeEvaluatorsConfig(list):
    """Iterable of (name, config) mirroring the production EvaluatorsConfig."""


def _write_jd_data(artifacts_path: Path) -> None:
    artifacts_path.mkdir(parents=True, exist_ok=True)
    (artifacts_path / "jd_data.json").write_text(
        json.dumps(
            {
                "checkpoint_info": {
                    "id": "12345678-1234-5678-1234-567812345678",
                    "checkpoint_type": "hf",
                },
                "history": [],
            }
        )
    )


def _setup_manager(
    tmp_path: Path,
    is_master: bool = True,
    uses_external_api: bool = False,
    evaluators: list | None = None,
    full_metrics_recalculate: bool = False,
    orkestration_manager=None,
    tensorboard_path: str | None = "tb",
) -> AutoMetricsManager:
    mgr = AutoMetricsManager()
    inference = SglangInferenceConfig()
    if uses_external_api:
        inference = ExternalApiInferenceConfig(
            formatting="openai",
            base_url="https://openrouter.ai/api/v1",
            model="anthropic/claude-sonnet-4",
            api_key="dummy-key",
            max_concurrency=9,
        )
    autometrics = SimpleNamespace(
        globals=SimpleNamespace(
            sglang_communication_port=61000,
            full_metrics_recalculate=full_metrics_recalculate,
            send_metrics_at_end=False,
        ),
        router=SimpleNamespace(port=30000),
        inference=inference,
        evaluators=_FakeEvaluatorsConfig(evaluators or []),
        model_info=MODEL_INFO,
    )
    config_manager = SimpleNamespace(
        checkpoint_path=tmp_path / "ckpt",
        artifacts_path=tmp_path / "artifacts",
        config_path=tmp_path / "artifacts" / "pam_config.yaml",
        job_id="job-1",
        global_node_rank=0,
        global_nnodes=2,
        gpus_per_node=8,
    )
    mgr.setup(
        config_manager=config_manager,
        run_config=SimpleNamespace(
            tensorboard_path=tmp_path / tensorboard_path if tensorboard_path else None,
            run_name="run",
            autometrics=autometrics,
            max_seq_len=4096,
        ),
        orkestration_manager=orkestration_manager
        or SimpleNamespace(is_master=is_master),
        convertation_manager=SimpleNamespace(hf_checkpoint_path=tmp_path / "tokenizer"),
    )
    return mgr


def _run_all(targets: list) -> None:
    errors: list[BaseException] = []

    def _guarded(target):
        try:
            target()
        except BaseException as error:
            errors.append(error)

    threads = [
        threading.Thread(target=_guarded, args=(target,), daemon=True)
        for target in targets
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=HANG_SECONDS)
    assert not any(thread.is_alive() for thread in threads), "pipeline hung"
    assert errors == []


def _run_cluster(
    tmp_path: Path, engines: _Engines, evaluators: list, nnodes: int = 2
) -> list[dict]:
    server = _Server(engines.journal)
    engines.nnodes = nnodes
    managers = []
    for rank in range(nnodes):
        engines.node = rank
        node = _Node(server, rank)
        managers.append(
            _setup_manager(tmp_path, evaluators=evaluators, orkestration_manager=node)
        )
    _run_all([manager.run_pipeline for manager in managers])
    return server.ops


def _entries(journal: list, kind: str) -> list[tuple]:
    return [entry[1:] for entry in journal if entry[0] == kind]


def _ups(journal: list, engine: str) -> list[tuple]:
    return [entry[3] for entry in journal if entry[:3] == ("op", "up", engine)]


@pytest.fixture
def journal() -> list:
    return []


@pytest.fixture(autouse=True)
def engines(monkeypatch: pytest.MonkeyPatch, journal: list) -> _Engines:
    factory = _Engines(journal)
    monkeypatch.setattr(metrics_manager, "InferenceManager", factory)
    return factory


@pytest.fixture
def pipeline(monkeypatch: pytest.MonkeyPatch, journal: list) -> None:
    monkeypatch.setattr(DynamicScheduler, "PROBE_SECONDS", 0.01)
    monkeypatch.setattr(EngineOpLog, "POLL_SECONDS", 0.01)
    monkeypatch.setattr(EngineSlot, "COOLDOWN_SECONDS", 0)

    @contextmanager
    def shooter(name, *, phase, secondary_model=None):
        try:
            yield
        except Exception:
            journal.append(("shooter", name, phase, secondary_model, "error"))
            raise
        journal.append(("shooter", name, phase, secondary_model, "ok"))

    def save(metrics):
        journal.append(("save", *metrics))

    monkeypatch.setattr(
        metrics_manager, "EventsStream", SimpleNamespace(shooter=shooter)
    )
    monkeypatch.setattr(
        metrics_manager,
        "TensorBoardSaver",
        lambda *args: SimpleNamespace(save_metrics_to_tensorboard=save),
    )
    monkeypatch.setattr(JD, "ensure_autometrics_record", lambda **kwargs: None)
    monkeypatch.setattr(JD, "add_evaluator_metrics", lambda **kwargs: None)
    monkeypatch.setattr(
        metrics_manager.Logger, "error", lambda message: journal.append(("error",))
    )


@pytest.fixture
def jd_calls(monkeypatch: pytest.MonkeyPatch) -> list:
    calls: list = []
    monkeypatch.setattr(
        JD, "ensure_autometrics_record", lambda **kw: calls.append(("record", kw))
    )
    monkeypatch.setattr(
        JD, "add_evaluator_metrics", lambda **kw: calls.append(("metrics", kw))
    )
    monkeypatch.setattr(
        JD,
        "publish_metrics_cache_per_measurement",
        lambda **kw: calls.append(("cache", kw)),
    )
    return calls


@pytest.fixture
def capture_tensorboard(monkeypatch: pytest.MonkeyPatch) -> list:
    saved: list = []
    monkeypatch.setattr(
        "node_runner.src.metrics.metrics_manager.TensorBoardSaver",
        lambda *a: SimpleNamespace(save_metrics_to_tensorboard=saved.append),
    )
    return saved


class TestEngineSetup:
    @pytest.mark.parametrize(
        "latency,has_proxy,has_workers,replay,expected_url",
        [
            (True, True, True, False, "http://evaluated/latency"),
            (True, True, False, True, "http://evaluated/latency"),
            (True, True, False, False, "http://evaluated"),
            (True, False, True, False, "http://evaluated"),
            (False, True, True, False, "http://evaluated"),
        ],
        ids=["local", "replay", "external-api", "no-proxy", "bulk"],
    )
    def test_only_local_and_replayed_latency_shooters_use_priority_endpoint(
        self,
        monkeypatch,
        tmp_path,
        journal,
        latency,
        has_proxy,
        has_workers,
        replay,
        expected_url,
    ):
        monkeypatch.setattr(_Shooter, "LATENCY", latency)
        name, config = _shooter(monkeypatch, journal, "obm")
        manager = _setup_manager(tmp_path)
        manager.run_config.is_replay_mode = replay
        manager.run_config.autometrics.evaluators = [(name, config)]
        evaluated = manager.engines_by_key[EVALUATED]
        evaluated.obm_proxy_manager = object() if has_proxy else None
        if not has_workers:
            evaluated.worker_urls = []

        manager._build_tasks()

        assert config.applied["base_url"] == expected_url
        assert config.applied["direct_base_url"] == "http://direct-evaluated"
        assert manager.tasks[0].latency is latency

    @pytest.mark.parametrize("is_master", [True, False], ids=["master", "worker"])
    def test_every_judge_is_built_once_in_config_order_on_every_node(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        engines: _Engines,
        journal: list,
        is_master: bool,
    ):
        evaluators = [
            _shooter(
                monkeypatch, journal, "tau2", _ParallelShooter, judges=("jb", "ja")
            ),
            _shooter(monkeypatch, journal, "dredd", _ScoringShooter, judges=("jc",)),
            _shooter(
                monkeypatch, journal, "forge", _ParallelShooter, judges=("ja", "jd")
            ),
        ]

        _setup_manager(tmp_path, is_master=is_master, evaluators=evaluators)

        assert engines.built == [
            (EVALUATED, None),
            ("jb", 0),
            ("ja", 1),
            ("jc", 2),
            ("jd", 3),
        ]

    def test_every_configurable_evaluator_has_an_implementation(self):
        assert set(EVALUATORS) == set(EvaluatorsConfig.model_fields)

    def test_evaluator_is_set_up_with_the_engine_endpoints_already_applied(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        engines: _Engines,
        journal: list,
    ):
        engines.instances = {EVALUATED: 4, "j1": 2, "api:j2": 3}
        engines.layouts = {"api:j2": [()]}
        _, config = _shooter(
            monkeypatch, journal, "tau2", _ParallelShooter, judges=("j1", "api:j2")
        )

        _setup_manager(tmp_path, evaluators=[("tau2", config)])

        assert config.applied == {
            "inference_instances": 4,
            "base_url": "http://evaluated",
            "direct_base_url": "http://direct-evaluated",
            "worker_urls": ["http://evaluated[0]", "http://evaluated[1]"],
            "inference_config": SglangInferenceConfig(),
            "max_seq_len": 4096,
            "hf_checkpoint_path": tmp_path / "tokenizer",
        }
        assert config.judge_endpoints == {
            "j1": JudgeEndpoint("http://j1", 2),
            "api:j2": JudgeEndpoint("http://api:j2", 3),
        }
        assert config.setup_args == (
            config,
            tmp_path / "artifacts",
            tmp_path / "tokenizer",
            MODEL_INFO,
        )
        assert config.applied_before_setup

    def test_dredd_config_points_its_judge_at_the_judge_engine(
        self, tmp_path: Path, engines: _Engines
    ):
        engines.instances = {EVALUATED: 2, "j": 3}
        config = DreddEvaluatorConfig.model_validate({"judge": {"model_name": "j"}})

        _setup_manager(tmp_path, evaluators=[("dredd", config)])

        assert (config.model.api_base, config.model.parallel_api_calls) == (
            "http://evaluated/v1",
            2 * DEFAULT_PARALLEL_PER_INFERENCE_INSTANCE,
        )
        assert (config.judge.api_base, config.judge.parallel_api_calls) == (
            "http://j/v1",
            3 * DEFAULT_PARALLEL_PER_INFERENCE_INSTANCE,
        )

    def test_evaluator_setup_failure_aborts_the_master_setup(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, journal: list
    ):
        evaluators = [_shooter(monkeypatch, journal, "fake", fail_on=("setup",))]

        with pytest.raises(RuntimeError, match="fake setup failed"):
            _setup_manager(tmp_path, evaluators=evaluators)

    @pytest.mark.usefixtures("pipeline")
    def test_evaluator_without_stages_does_not_stop_the_others(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        engines: _Engines,
        journal: list,
    ):
        evaluators = [
            _shooter(monkeypatch, journal, "empty", _NoStagesShooter),
            _shooter(monkeypatch, journal, "fake"),
        ]

        _run_cluster(tmp_path, engines, evaluators)

        assert _entries(journal, "ran") == [("fake", "run")]

    def test_judge_named_like_the_evaluated_engine_is_rejected(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, journal: list
    ):
        evaluators = [
            _shooter(
                monkeypatch, journal, "dredd", _ScoringShooter, judges=(EVALUATED,)
            )
        ]

        with pytest.raises(ValueError, match=EVALUATED):
            _setup_manager(tmp_path, evaluators=evaluators)


class TestWorker:
    @pytest.mark.usefixtures("pipeline")
    def test_worker_follows_the_op_log_without_setting_up_any_evaluator(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        engines: _Engines,
        journal: list,
    ):
        name, config = _shooter(
            monkeypatch, journal, "tau2", _ParallelShooter, judges=("j",)
        )
        server = _Server(journal)
        _Node(server, 0).publish_op({"action": "finish"})
        engines.node = 1
        worker = _setup_manager(
            tmp_path, evaluators=[(name, config)], orkestration_manager=_Node(server, 1)
        )

        _run_all([worker.run_pipeline])

        assert config.applied is None
        assert config.setup_args is None
        assert [entry for entry in journal if entry[0] in ("up", "run")] == []

    @pytest.mark.usefixtures("pipeline")
    def test_every_node_starts_and_stops_exactly_the_units_published_for_it(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        engines: _Engines,
        journal: list,
    ):
        evaluators = [
            _shooter(monkeypatch, journal, "lm_eval", seconds=0.05),
            _shooter(monkeypatch, journal, "tau2", _ParallelShooter, judges=("j",)),
            _shooter(monkeypatch, journal, "dredd", _ScoringShooter, judges=("j",)),
        ]

        ops = _run_cluster(tmp_path, engines, evaluators)

        assert ops[-1] == {"action": "finish"}
        for rank in (0, 1):
            published = [
                (op["action"], op["engine"]) for op in ops[:-1] if rank in op["ranks"]
            ]
            executed = [
                (entry[0], entry[1])
                for entry in journal
                if entry[0] in ("up", "down") and entry[2] == rank
            ]
            assert executed == published


class TestStages:
    @pytest.mark.usefixtures("pipeline")
    def test_plain_evaluator_runs_once_and_saves_its_metrics_after(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        engines: _Engines,
        journal: list,
    ):
        evaluators = [_shooter(monkeypatch, journal, "fake")]

        _run_cluster(tmp_path, engines, evaluators)

        assert [
            e for e in journal if e[0] in ("ran", "shooter", "save", "after_run")
        ] == [
            ("ran", "fake", "run"),
            ("shooter", "fake", "single", None, "ok"),
            ("save", "fake"),
            ("after_run", "fake"),
        ]

    @pytest.mark.usefixtures("pipeline")
    def test_scoring_evaluator_frees_the_nodes_for_its_judge_after_generation(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        engines: _Engines,
        journal: list,
    ):
        engines.layouts = {EVALUATED: [(0, 1)], "j": [(0, 1)]}
        evaluators = [
            _shooter(monkeypatch, journal, "dredd", _ScoringShooter, judges=("j",))
        ]

        _run_cluster(tmp_path, engines, evaluators)

        assert [e for e in journal if e[0] in ("op", "ran", "shooter", "save")] == [
            ("op", "up", EVALUATED, (0, 1)),
            ("ran", "dredd", "generation"),
            ("shooter", "dredd", "single", None, "ok"),
            ("op", "down", EVALUATED, (0, 1)),
            ("op", "up", "j", (0, 1)),
            ("ran", "dredd", "evaluation"),
            ("shooter", "dredd", "scoring", "j", "ok"),
            ("save", "dredd"),
            ("op", "down", "j", (0, 1)),
            ("op", "finish", None, ()),
        ]

    @pytest.mark.usefixtures("pipeline")
    def test_parallel_evaluator_runs_its_judges_one_after_another(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        engines: _Engines,
        journal: list,
    ):
        evaluators = [
            _shooter(
                monkeypatch,
                journal,
                "tau2",
                _ParallelShooter,
                judges=("ja", "jb"),
                seconds=0.05,
            )
        ]

        _run_cluster(tmp_path, engines, evaluators, nnodes=4)

        assert [e for e in journal if e[0] in ("run", "ran", "shooter", "save")] == [
            ("run", "tau2", "ja"),
            ("ran", "tau2", "ja"),
            ("shooter", "tau2", "parallel", "ja", "ok"),
            ("run", "tau2", "jb"),
            ("ran", "tau2", "jb"),
            ("shooter", "tau2", "parallel", "jb", "ok"),
            ("save", "tau2"),
        ]

    @pytest.mark.usefixtures("pipeline")
    def test_shooter_events_carry_the_phase_and_the_judge(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        engines: _Engines,
        journal: list,
    ):
        evaluators = [
            _shooter(monkeypatch, journal, "plain"),
            _shooter(monkeypatch, journal, "scoring", _ScoringShooter, judges=("j",)),
            _shooter(monkeypatch, journal, "par", _ParallelShooter, judges=("k",)),
        ]

        _run_cluster(tmp_path, engines, evaluators)

        assert sorted(_entries(journal, "shooter"), key=str) == sorted(
            [
                ("plain", "single", None, "ok"),
                ("scoring", "single", None, "ok"),
                ("scoring", "scoring", "j", "ok"),
                ("par", "parallel", "k", "ok"),
            ],
            key=str,
        )

    @pytest.mark.parametrize(
        ("evaluator_class", "judges"),
        [
            pytest.param(_ScoringShooter, ("j",), id="scoring"),
            pytest.param(_ParallelShooter, ("ja", "jb"), id="parallel"),
        ],
    )
    @pytest.mark.usefixtures("pipeline")
    def test_evaluators_sharing_a_judge_run_every_stage_once_on_one_boot(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        engines: _Engines,
        journal: list,
        evaluator_class: type,
        judges: tuple,
    ):
        evaluators = [
            _shooter(monkeypatch, journal, "a", evaluator_class, judges=judges),
            _shooter(monkeypatch, journal, "b", evaluator_class, judges=judges),
        ]

        _run_cluster(tmp_path, engines, evaluators)

        ran = _entries(journal, "ran")
        assert len(ran) == len(set(ran)) == 4
        assert sorted(_entries(journal, "save")) == [("a",), ("b",)]
        for judge in judges:
            booted_units = _ups(journal, judge)
            assert len(booted_units) == len(set(booted_units))

    def test_concurrent_saves_are_serialized(
        self, tmp_path: Path, capture_tensorboard: list, jd_calls: list
    ):
        mgr = _setup_manager(tmp_path)
        entered, release = threading.Event(), threading.Event()
        first, second = _FakeEvaluator(), _FakeEvaluator()
        first_metrics = first.get_metrics

        def blocking_get_metrics():
            entered.set()
            release.wait(HANG_SECONDS)
            return first_metrics()

        first.get_metrics = blocking_get_metrics
        first_save = threading.Thread(
            target=mgr.save_evaluator_metrics, args=("first", first)
        )
        second_save = threading.Thread(
            target=mgr.save_evaluator_metrics, args=("second", second)
        )
        first_save.start()
        entered.wait(HANG_SECONDS)
        second_save.start()
        second_save.join(timeout=0.2)
        second_waited = second_save.is_alive()
        release.set()
        first_save.join(HANG_SECONDS)
        second_save.join(HANG_SECONDS)

        assert second_waited
        assert second.after_run_args == second.metrics


class TestAccLen:
    @pytest.mark.usefixtures("pipeline")
    def test_acc_len_waits_until_every_evaluated_unit_is_up(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        engines: _Engines,
        journal: list,
    ):
        evaluators = [
            _shooter(monkeypatch, journal, "acc_len", _AccLenShooter),
            _shooter(
                monkeypatch,
                journal,
                "tau2",
                _ParallelShooter,
                judges=("j",),
                seconds=0.2,
            ),
        ]

        _run_cluster(tmp_path, engines, evaluators)

        started_before_acc_len = journal[: journal.index(("run", "acc_len", "run"))]
        assert sorted(_ups(started_before_acc_len, EVALUATED)) == [(0,), (1,)]


class TestFailures:
    @pytest.mark.usefixtures("pipeline")
    def test_failing_evaluator_does_not_affect_the_others(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        engines: _Engines,
        journal: list,
    ):
        evaluators = [
            _shooter(monkeypatch, journal, "broken", fail_on=("run",)),
            _shooter(monkeypatch, journal, "fine"),
        ]

        ops = _run_cluster(tmp_path, engines, evaluators)

        assert sorted(_entries(journal, "shooter")) == [
            ("broken", "single", None, "error"),
            ("fine", "single", None, "ok"),
        ]
        assert _entries(journal, "save") == [("fine",)]
        assert _entries(journal, "error") == [()]
        assert ops[-1] == {"action": "finish"}

    @pytest.mark.usefixtures("pipeline")
    def test_failing_after_run_keeps_tensorboard_and_the_other_evaluators(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        engines: _Engines,
        journal: list,
    ):
        evaluators = [
            _shooter(monkeypatch, journal, "broken", fail_on=("after_run",)),
            _shooter(monkeypatch, journal, "fine"),
        ]

        ops = _run_cluster(tmp_path, engines, evaluators)

        assert sorted(_entries(journal, "save")) == [("broken",), ("fine",)]
        assert sorted(_entries(journal, "after_run")) == [("broken",), ("fine",)]
        assert _entries(journal, "error") == [()]
        assert ops[-1] == {"action": "finish"}

    @pytest.mark.parametrize(
        "failing_judge",
        [
            "ja",
            "jb",
            "jc",
        ],
    )
    @pytest.mark.usefixtures("pipeline")
    def test_failing_judge_stage_keeps_the_other_judges_and_their_metrics(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        engines: _Engines,
        journal: list,
        failing_judge: str,
    ):
        judges = ("ja", "jb", "jc")
        evaluators = [
            _shooter(
                monkeypatch,
                journal,
                "tau2",
                _ParallelShooter,
                judges=judges,
                fail_on=(failing_judge,),
            )
        ]

        _run_cluster(tmp_path, engines, evaluators)

        assert _entries(journal, "ran") == [
            ("tau2", judge) for judge in judges if judge != failing_judge
        ]
        assert _entries(journal, "save") == [("tau2",)]

    @pytest.mark.usefixtures("pipeline")
    def test_judges_of_a_parallel_evaluator_never_overlap_when_a_middle_one_fails(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        engines: _Engines,
        journal: list,
    ):
        engines.failing_starts = {("jb", rank) for rank in range(4)}
        evaluators = [
            _shooter(
                monkeypatch,
                journal,
                "tau2",
                _ParallelShooter,
                judges=("ja", "jb", "jc"),
                seconds=0.3,
            ),
            _shooter(monkeypatch, journal, "other", _ParallelShooter, judges=("jb",)),
        ]

        _run_cluster(tmp_path, engines, evaluators, nnodes=4)

        tau2_runs = [
            entry
            for entry in journal
            if entry[0] in ("run", "ran") and entry[1] == "tau2"
        ]
        assert tau2_runs == [
            ("run", "tau2", "ja"),
            ("ran", "tau2", "ja"),
            ("run", "tau2", "jc"),
            ("ran", "tau2", "jc"),
        ]
        assert _entries(journal, "save").count(("tau2",)) == 1

    @pytest.mark.usefixtures("pipeline")
    def test_metrics_of_earlier_judges_survive_a_last_judge_that_fails_to_start(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        engines: _Engines,
        journal: list,
    ):
        engines.failing_starts = {("jb", 1)}
        evaluators = [
            _shooter(
                monkeypatch, journal, "tau2", _ParallelShooter, judges=("ja", "jb")
            )
        ]

        _run_cluster(tmp_path, engines, evaluators)

        assert _entries(journal, "save") == [("tau2",)]

    @pytest.mark.parametrize("failing_phase", ["generation", "evaluation"])
    @pytest.mark.usefixtures("pipeline")
    def test_failed_phase_of_a_scoring_evaluator_saves_no_metrics(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        engines: _Engines,
        journal: list,
        failing_phase: str,
    ):
        evaluators = [
            _shooter(
                monkeypatch,
                journal,
                "dredd",
                _ScoringShooter,
                judges=("j",),
                fail_on=(failing_phase,),
            )
        ]

        ops = _run_cluster(tmp_path, engines, evaluators)

        assert _entries(journal, "save") == []
        assert ops[-1] == {"action": "finish"}

    @pytest.mark.usefixtures("pipeline")
    def test_failing_generation_skips_scoring_and_never_boots_the_judge(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        engines: _Engines,
        journal: list,
    ):
        evaluators = [
            _shooter(
                monkeypatch,
                journal,
                "dredd",
                _ScoringShooter,
                judges=("j",),
                fail_on=("generation",),
            )
        ]

        _run_cluster(tmp_path, engines, evaluators)

        assert _ups(journal, "j") == []
        assert ("run", "dredd", "evaluation") not in journal


class TestPipelineEnds:
    @pytest.mark.usefixtures("pipeline")
    def test_no_evaluators_publish_only_finish_and_start_nothing(
        self, tmp_path: Path, engines: _Engines, journal: list
    ):
        ops = _run_cluster(tmp_path, engines, evaluators=[])

        assert ops == [{"action": "finish"}]
        assert _entries(journal, "up") == []

    @pytest.mark.parametrize("is_master", [True, False], ids=["master", "worker"])
    @pytest.mark.usefixtures("pipeline")
    def test_cleanup_closes_every_engine_even_if_one_fails(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        engines: _Engines,
        journal: list,
        is_master: bool,
    ):
        engines.failing_closes = {"j1"}
        evaluators = [
            _shooter(
                monkeypatch, journal, "tau2", _ParallelShooter, judges=("j1", "j2")
            )
        ]
        mgr = _setup_manager(tmp_path, is_master=is_master, evaluators=evaluators)

        mgr.cleanup()

        assert _entries(journal, "close") == [(EVALUATED, 0), ("j1", 0), ("j2", 0)]
        assert _entries(journal, "error") == [()]

    def test_manager_cleanup_before_setup_is_noop(self):
        AutoMetricsManager().cleanup()

    def test_metrics_are_sent_at_the_end_even_when_the_scheduler_fails(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        mgr = _setup_manager(tmp_path)
        mgr.run_config.autometrics.globals.send_metrics_at_end = True
        posted = []
        monkeypatch.setattr(JD, "internet_is_available", lambda: True)
        monkeypatch.setattr(
            "node_runner.src.job_dispatcher.metrics.requests.post",
            lambda url, json, **kwargs: posted.append(json) or Mock(),
        )
        sent_during_run = []

        def run(self):
            JD.send_metrics({"a": 1}, None)
            sent_during_run.extend(posted)
            raise RuntimeError("scheduler failed")

        monkeypatch.setattr(DynamicScheduler, "run", run)

        with pytest.raises(RuntimeError, match="scheduler failed"):
            mgr.run_pipeline()

        assert sent_during_run == []
        assert posted == [{"a": 1}]
        assert JD._deferred_bodies is None


class TestSaveEvaluatorMetrics:
    def test_tensorboard_saved_before_failing_after_run(
        self,
        tmp_path: Path,
        capture_tensorboard: list,
    ):
        # Regression: a failing after_run (JD send / tar / symlink) must not
        # discard the TensorBoard write.
        fake = _FakeEvaluator()

        def _failing_after_run(metrics, uploaded=None):
            raise RuntimeError("after_run failed")

        fake.after_run = _failing_after_run
        mgr = _setup_manager(tmp_path)

        with pytest.raises(RuntimeError, match="after_run failed"):
            mgr.save_evaluator_metrics("fake", fake)

        assert capture_tensorboard == [fake.metrics]

    @pytest.mark.parametrize(
        ("measured", "expected_recorded", "expected_warnings", "expected_kinds"),
        [
            pytest.param(
                {"metric/a": 1.0},
                [("fake", {"metric/a": 1.0}, 0)],
                [],
                ["record", "metrics", "cache"],
                id="metrics-of-this-run-are-recorded",
            ),
            pytest.param(
                {},
                [],
                [
                    "fake produced no new metric files; "
                    "1 metric(s) not recorded to jd_data"
                ],
                ["record", "cache"],
                id="metrics-reused-from-an-earlier-run-are-not-recorded",
            ),
        ],
    )
    def test_save_evaluator_metrics_records_only_what_this_run_measured(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capture_tensorboard: list,
        measured: dict,
        expected_recorded: list,
        expected_warnings: list,
        expected_kinds: list,
        jd_calls: list,
    ):
        mgr = _setup_manager(tmp_path)
        warnings: list = []
        monkeypatch.setattr(
            "node_runner.src.metrics.metrics_manager.Logger.warning", warnings.append
        )

        fake = _FakeEvaluator()
        fake.measurement_metrics = lambda metrics: measured
        mgr.save_evaluator_metrics("fake", fake)

        recorded = [
            (kw["evaluator_name"], kw["metrics"], kw["metrics_files_mtime_ns"])
            for kind, kw in jd_calls
            if kind == "metrics"
        ]
        assert recorded == expected_recorded
        assert [kind for kind, _ in jd_calls] == expected_kinds
        assert warnings == expected_warnings
        assert capture_tensorboard == [fake.metrics]
        assert fake.after_run_args == fake.metrics

    def test_save_evaluator_metrics_hands_the_published_upload_to_after_run(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capture_tensorboard: list,
        jd_calls: list,
    ):
        mgr = _setup_manager(tmp_path)
        upload = MeasurementUpload("prefix/record-id/fake", [Path("cache/alpha.jsonl")])
        monkeypatch.setattr(
            JD, "publish_metrics_cache_per_measurement", lambda **kw: upload
        )

        fake = _FakeEvaluator()
        mgr.save_evaluator_metrics("fake", fake)

        assert fake.after_run_uploaded is upload
        assert fake.after_run_args == fake.metrics

    def test_empty_metrics_touch_neither_jd_nor_tensorboard(
        self, tmp_path: Path, capture_tensorboard: list, jd_calls: list
    ):
        mgr = _setup_manager(tmp_path)
        fake = _FakeEvaluator()
        fake.metrics = {}

        mgr.save_evaluator_metrics("fake", fake)

        assert jd_calls == []
        assert capture_tensorboard == []
        assert fake.after_run_args is None


class TestAutoMetricsManagerSetup:
    @pytest.mark.parametrize("tensorboard_path", ["tb", None])
    def test_metrics_go_to_the_configured_tensorboard(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        tensorboard_path: str | None,
        jd_calls: list,
    ):
        savers: list = []
        monkeypatch.setattr(
            metrics_manager,
            "TensorBoardSaver",
            lambda *args: (
                savers.append(args)
                or SimpleNamespace(save_metrics_to_tensorboard=lambda metrics: None)
            ),
        )
        mgr = _setup_manager(tmp_path, tensorboard_path=tensorboard_path)

        mgr.save_evaluator_metrics("fake", _FakeEvaluator())

        expected_tensorboard = tmp_path / tensorboard_path if tensorboard_path else None
        assert savers == [(tmp_path / "ckpt", expected_tensorboard, "run")]

    def test_full_metrics_recalculate_archives_previous_metrics(self, tmp_path: Path):
        old_result = tmp_path / "artifacts" / "metrics" / "bfcl" / "metrics.json"
        old_result.parent.mkdir(parents=True)
        old_result.write_text("{}")

        _setup_manager(tmp_path, full_metrics_recalculate=True)

        artifacts_children = sorted(p.name for p in (tmp_path / "artifacts").iterdir())
        assert len(artifacts_children) == 1
        (archived_name,) = artifacts_children
        assert archived_name.startswith("metrics_")
        assert (
            tmp_path / "artifacts" / archived_name / "bfcl" / "metrics.json"
        ).exists()

    @pytest.mark.parametrize(
        ("full_metrics_recalculate", "is_master"),
        [(False, True), (True, False)],
        ids=["flag_off", "worker"],
    )
    def test_previous_metrics_kept_without_flag_or_on_worker(
        self, tmp_path: Path, full_metrics_recalculate: bool, is_master: bool
    ):
        metrics_dir = tmp_path / "artifacts" / "metrics"
        metrics_dir.mkdir(parents=True)

        _setup_manager(
            tmp_path,
            is_master=is_master,
            full_metrics_recalculate=full_metrics_recalculate,
        )

        assert [p.name for p in (tmp_path / "artifacts").iterdir()] == ["metrics"]

    def test_full_metrics_recalculate_without_previous_metrics_is_noop(
        self, tmp_path: Path
    ):
        (tmp_path / "artifacts").mkdir()

        _setup_manager(tmp_path, full_metrics_recalculate=True)

        assert list((tmp_path / "artifacts").iterdir()) == []


def _build_dredd_evaluator(tmp_path: Path) -> DreddEvaluator:
    evaluator = DreddEvaluator()
    root = tmp_path / "model_ep1_ba2"
    evaluator.hf_checkpoint_path = root / "artifacts" / "hf_checkpoint"
    evaluator.hf_checkpoint_path.mkdir(parents=True, exist_ok=True)
    evaluator.metrics_dir = tmp_path / "dredd_metrics"
    evaluator.metrics_dir.mkdir(parents=True, exist_ok=True)
    evaluator.artifacts_path = tmp_path / "artifacts"
    evaluator.model_info = ModelInfo(model_class="sft", size="lite", username="u")
    evaluator.autometric_runs_config = DreddEvaluatorConfig.model_validate(
        {"judge": {"model_name": "j"}}
    )
    return evaluator


def test_load_summary_variants(tmp_path: Path):
    p = tmp_path / "summary.jsonl"
    assert _load_summary_from_jsonl(p) is None
    p.write_text("\n\n")
    assert _load_summary_from_jsonl(p) is None
    p.write_text('{"a": 1}\n{"b": 2}\n')
    assert _load_summary_from_jsonl(p) == {"a": 1}


def test_dredd_get_metrics_always_converts_to_fractions(tmp_path: Path, monkeypatch):
    evaluator = _build_dredd_evaluator(tmp_path)
    data_root = tmp_path / "data"
    questions = data_root / "arenas" / "a1.jsonl"
    questions.parent.mkdir(parents=True)
    questions.write_text(
        "".join(json.dumps({"uid": i}) + "\n" for i in range(10)), encoding="utf-8"
    )
    monkeypatch.setenv("DREDD_DATA_ROOT", str(data_root))
    d = evaluator.metrics_dir / "arenas" / "a1" / "judgments" / "b1"
    d.mkdir(parents=True, exist_ok=True)
    (d / "judgment_summary.jsonl").write_text(
        json.dumps(
            {"total": 10, "failed_judgments": 0, "win_rate": 50.0, "score_0_100": 60.0}
        )
    )
    evaluator._dredd_phase = "judgement"
    m = evaluator.get_metrics()
    assert m["dredd/a1/b1/win_rate"] == 0.5
    assert m["dredd/a1/b1/score_0_1"] == 0.6


def test_dredd_generates_first_and_then_judges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    commands: list[list[str]] = []

    class _Subprocess:
        def __init__(self, cmd, *args):
            commands.append(cmd)

        def start(self):
            pass

        def wait_for_completion(self):
            pass

        def log_error(self):
            pass

        def is_successful(self):
            return True

    monkeypatch.setattr(dredd_module, "get_dredd_submodule_path", lambda: tmp_path)
    monkeypatch.setattr(dredd_module, "SubprocessManager", _Subprocess)
    evaluator = _build_dredd_evaluator(tmp_path)

    for stage in evaluator.stages():
        stage.run()

    phases = [cmd[cmd.index("--phase") + 1] for cmd in commands]
    assert phases == ["generation", "judgement"]


class TestRecordMetricsToJdData:
    def test_save_records_the_run_and_the_metrics_of_the_evaluator(
        self, tmp_path: Path, capture_tensorboard: list, jd_calls: list
    ):
        mgr = _setup_manager(tmp_path)
        fake = _FakeEvaluator()

        mgr.save_evaluator_metrics("fake", fake)

        artifacts_path = tmp_path / "tokenizer" / "artifacts"
        assert [kind for kind, _ in jd_calls] == ["record", "metrics", "cache"]
        assert jd_calls[0][1] == {
            "artifacts_path": artifacts_path,
            "run_uuid": mgr.run_uuid,
            "checkpoint_path": tmp_path / "ckpt",
            "hf_checkpoint_path": tmp_path / "tokenizer",
            "run_name": "run",
            "job_id": "job-1",
            "pam_config_path": tmp_path / "artifacts" / "pam_config.yaml",
        }
        assert jd_calls[1][1] == {
            "artifacts_path": artifacts_path,
            "run_uuid": mgr.run_uuid,
            "evaluator_name": "fake",
            "metrics": fake.metrics,
            "metrics_files_mtime_ns": fake.measured_files_mtime_ns(),
        }

    def test_an_external_api_run_touches_no_jd_data(
        self, tmp_path: Path, capture_tensorboard: list, jd_calls: list
    ):
        mgr = _setup_manager(tmp_path, uses_external_api=True)
        fake = _FakeEvaluator()

        mgr.save_evaluator_metrics("fake", fake)

        assert jd_calls == []
        assert capture_tensorboard == [fake.metrics]

    def test_the_record_exists_for_a_shooter_without_new_metrics(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capture_tensorboard: list
    ):
        artifacts_path = tmp_path / "tokenizer" / "artifacts"
        mgr = _setup_manager(tmp_path)
        _write_jd_data(artifacts_path)
        monkeypatch.setattr(JD, "upload_pam_config", lambda *args: None)
        monkeypatch.setattr(
            JD, "publish_metrics_cache_per_measurement", lambda **kw: None
        )

        fake = _FakeEvaluator()
        fake.measurement_metrics = lambda metrics: {}
        mgr.save_evaluator_metrics("fake", fake)

        assert JD.read_measurement_keys(artifacts_path, mgr.run_uuid) is not None
        record = json.loads((artifacts_path / "jd_data.json").read_text())["history"][0]
        assert record["data"]["metrics"] == {}
        assert record["data"]["evaluators"] == []
