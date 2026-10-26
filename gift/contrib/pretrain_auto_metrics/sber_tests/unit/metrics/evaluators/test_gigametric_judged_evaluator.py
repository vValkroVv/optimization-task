"""Unit-тесты GigaMetricJudgedEvaluator.

Subprocess замокан. Контракт инжекции судьи проверяется по
материализованному per-judge YAML: судья кладётся ОДИН раз в общий слот
defaults.benchmark_config.judge_llm_config (аналог llm_judge_shooter в obm),
GigaMetric применяет его ко всем бенчам среза.
"""

import json
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from node_runner.src.config_model import (
    GigaMetricJudgedEvaluatorConfig,
    ModelInfoConfig,
)
from node_runner.src.metrics.evaluators.gigametric import GigaMetricJudgedEvaluator
from sber_tests.test_support import judge_endpoints

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def model_info() -> ModelInfoConfig:
    return ModelInfoConfig(model_class="sft", size="ultra", username="tester")


@pytest.fixture
def artifacts_path(tmp_path: Path) -> Path:
    p = tmp_path / "artifacts"
    p.mkdir()
    return p


@pytest.fixture
def base_dir(tmp_path: Path) -> Path:
    p = tmp_path / "validation_runs"
    p.mkdir()
    return p


@pytest.fixture
def hf_checkpoint(tmp_path: Path) -> Path:
    p = tmp_path / "ep1_ba100" / "artifacts" / "huggingface"
    p.mkdir(parents=True)
    return p


@pytest.fixture(autouse=True)
def _fake_validation_pam_export(monkeypatch: pytest.MonkeyPatch) -> None:
    """Фейковый ``validation.pam_export``, читающий state.json (как в тестах
    GigaMetricEvaluator — gigametric доступен только в Docker-образе)."""
    import types

    def _get_pam_metrics(validation_id: str, base: Path) -> dict[str, float]:
        state_file = Path(base) / validation_id / "state.json"
        if not state_file.exists():
            return {}
        data = json.loads(state_file.read_text(encoding="utf-8"))
        result: dict[str, float] = {}
        for task in data.get("tasks", []):
            if task.get("status") != "completed":
                continue
            mid = task.get("model_id")
            bid = task.get("benchmark_id")
            mname = task.get("primary_metric_name")
            mval = task.get("primary_metric")
            if mname is None or mval is None:
                continue
            result[f"gigametric/{mid}/{bid}/{mname}"] = mval
        return result

    fake_module = types.ModuleType("validation.pam_export")
    fake_module.get_pam_metrics = _get_pam_metrics  # type: ignore[attr-defined]
    fake_pkg = types.ModuleType("validation")
    fake_pkg.pam_export = fake_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "validation", fake_pkg)
    monkeypatch.setitem(sys.modules, "validation.pam_export", fake_module)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ALL_BENCHES = ["bbh", "hle", "ragqa", "finam_finbench", "browsecomp", "tau2"]


def _make_validation_config() -> dict[str, Any]:
    return {
        "validation_name": "judged_smoke",
        "benchmarks": {
            "bbh": {"name": "bbh-adapter"},
            "hle": {"name": "hle-adapter"},
            "ragqa": {"name": "ragqa-adapter", "parameters": {"use_llm_judge": False}},
            "finam_finbench": {"name": "finam-finbench-adapter"},
            "browsecomp": {
                "name": "browsecomp-adapter",
                "parameters": {"n_repeats": 1},
            },
            "tau2": {"name": "tau2-adapter"},
        },
        "models": [
            {
                "id": "local",
                "llm_config": {
                    "model": "openai",
                    "api_url": "http://localhost:29999/v1",
                    "api_key": "not-needed",
                },
                "benchmarks": list(ALL_BENCHES),
            }
        ],
    }


def _make_cfg(
    base_dir: Path,
    judges: dict[str, list[str]] | None = None,
    validation_config: dict[str, Any] | None = None,
    **overrides: Any,
) -> GigaMetricJudgedEvaluatorConfig:
    overrides.setdefault("local_model_id", "local")
    return GigaMetricJudgedEvaluatorConfig(
        validation_config=validation_config or _make_validation_config(),
        base_dir=base_dir,
        judges=judges
        if judges is not None
        else {"judge-a": ["bbh", "hle"], "judge-b": ["tau2"]},
        **overrides,
    )


def test_judged_config_rejects_internal_task_names(base_dir: Path) -> None:
    with pytest.raises(ValueError, match="PAPAYA aliases"):
        _make_cfg(base_dir, judges={"judge-a": ["sber_internal"]})


def _apply_defaults(
    cfg: GigaMetricJudgedEvaluatorConfig,
    secondary_base_url: str = "http://10.0.0.1:8000",
) -> None:
    cfg.judge_endpoints = judge_endpoints(
        {name: secondary_base_url for name in cfg.judges}
    )
    cfg.apply_defaults(
        inference_instances=1,
        base_url="http://localhost:29999",
    )


def _build_evaluator(
    cfg: GigaMetricJudgedEvaluatorConfig,
    artifacts_path: Path,
    hf_checkpoint: Path,
    model_info: ModelInfoConfig,
) -> GigaMetricJudgedEvaluator:
    ev = GigaMetricJudgedEvaluator()
    ev.setup(
        autometric_runs_config=cfg,
        artifacts_path=artifacts_path,
        hf_checkpoint_path=hf_checkpoint,
        model_info=model_info,
    )
    return ev


def _write_state(
    base_dir: Path, validation_id: str, tasks: list[dict[str, Any]]
) -> None:
    run_dir = base_dir / validation_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "state.json").write_text(
        json.dumps(
            {
                "validation_id": validation_id,
                "validation_name": "test",
                "config_hash": "sha256:test",
                "status": "completed_passed",
                "tasks": tasks,
            }
        ),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# secondary_model_names / формат judges (судья -> [таски], как obm_judged)
# ---------------------------------------------------------------------------


class TestGigaMetricJudgedSecondaryModelNames:
    def test_secondary_model_names_are_judge_keys(self, tmp_path):
        cfg = _make_cfg(
            tmp_path, judges={"model-a": ["bbh", "tau2"], "model-b": ["hle"]}
        )
        assert cfg.secondary_model_names == ["model-a", "model-b"]

    def test_scalar_task_normalized_to_list(self, tmp_path):
        # judges: {судья: "bench"} нормализуется в {судья: ["bench"]}.
        cfg = _make_cfg(tmp_path, judges={"model-a": "bbh"})
        assert cfg.judges == {"model-a": ["bbh"]}


# ---------------------------------------------------------------------------
# setup
# ---------------------------------------------------------------------------


class TestGigaMetricJudgedSetup:
    def test_setup_initializes_judge_tasks_and_metrics_dir(
        self, artifacts_path, base_dir, hf_checkpoint, model_info
    ):
        cfg = _make_cfg(base_dir)
        ev = _build_evaluator(cfg, artifacts_path, hf_checkpoint, model_info)

        assert ev.metrics_dir == artifacts_path / "metrics" / "gigametric_judged"
        assert set(ev._judge_tasks) == {"judge-a", "judge-b"}

    def test_setup_without_ignored_benchmarks(
        self, artifacts_path, base_dir, hf_checkpoint, model_info
    ):
        validation_config = _make_validation_config()
        validation_config["benchmarks"] = {"bbh": {"name": "bbh-adapter"}}
        validation_config["models"][0]["benchmarks"] = ["bbh"]
        cfg = _make_cfg(
            base_dir, judges={"judge-a": ["bbh"]}, validation_config=validation_config
        )
        ev = _build_evaluator(cfg, artifacts_path, hf_checkpoint, model_info)
        assert list(ev._judge_tasks) == ["judge-a"]

    def test_setup_accepts_judges_for_unknown_benchmark(
        self, artifacts_path, base_dir, hf_checkpoint, model_info
    ):
        # Как obm_judged: конфигу доверяем, setup не роняет весь замер из-за
        # judges, ссылающихся на бенч вне validation_config.benchmarks.
        cfg = _make_cfg(base_dir, judges={"judge-a": ["no_such_bench"]})
        ev = _build_evaluator(cfg, artifacts_path, hf_checkpoint, model_info)
        assert list(ev._judge_tasks) == ["judge-a"]

    @pytest.mark.parametrize(
        "bench_id, adapter",
        [
            ("financeqa", "financeqa-adapter"),
            ("finam_finbench", "finam-finbench-adapter"),
        ],
    )
    def test_setup_accepts_any_benchmark_in_config(
        self, artifacts_path, base_dir, hf_checkpoint, model_info, bench_id, adapter
    ):
        # Нет allowlist судейских адаптеров (как в obm): любой бенч из конфига
        # можно назначить судье — доверяем конфигу, судья кладётся в общий слот.
        validation_config = _make_validation_config()
        validation_config["benchmarks"][bench_id] = {"name": adapter}
        validation_config["models"][0]["benchmarks"].append(bench_id)
        cfg = _make_cfg(
            base_dir,
            judges={"judge-a": [bench_id]},
            validation_config=validation_config,
        )
        # setup не должен бросать — никакой проверки «поддерживается ли судья».
        _build_evaluator(cfg, artifacts_path, hf_checkpoint, model_info)


# ---------------------------------------------------------------------------
# _write_judged_config — судья в едином слоте defaults.benchmark_config
# ---------------------------------------------------------------------------


def _judged_config_for(
    ev: GigaMetricJudgedEvaluator, judge_model_name: str
) -> dict[str, Any]:
    path = ev._write_judged_config(judge_model_name, ev._judge_tasks[judge_model_name])
    assert path.exists()
    with open(path) as f:
        return yaml.safe_load(f)


class TestGigaMetricJudgedWriteJudgedConfig:
    # apply_defaults получает чистый base_url; judge_llm_config добавляет /v1.
    JUDGE_BASE = "http://10.0.0.1:8000"
    JUDGE_API = f"{JUDGE_BASE}/v1"

    def _evaluator(
        self, artifacts_path, base_dir, hf_checkpoint, model_info, judges
    ) -> GigaMetricJudgedEvaluator:
        cfg = _make_cfg(base_dir, judges=judges)
        _apply_defaults(cfg, self.JUDGE_BASE)
        return _build_evaluator(cfg, artifacts_path, hf_checkpoint, model_info)

    def test_judge_in_single_global_defaults_slot(
        self, artifacts_path, base_dir, hf_checkpoint, model_info
    ):
        # Судья задаётся ОДИН раз в defaults.benchmark_config.judge_llm_config
        # (аналог единого llm_judge_shooter в obm), а НЕ в каждом бенче.
        ev = self._evaluator(
            artifacts_path,
            base_dir,
            hf_checkpoint,
            model_info,
            judges={"judge-a": ["bbh", "hle"]},
        )
        data = _judged_config_for(ev, "judge-a")

        assert set(data["benchmarks"]) == {"bbh", "hle"}
        judge = data["defaults"]["benchmark_config"]["judge_llm_config"]
        assert judge["model"] == "openai"
        assert judge["name"] == "judge-a"
        assert judge["api_url"] == self.JUDGE_API
        assert judge["parameters"] == {"timeout": 21600}
        # В самих бенчах судья не дублируется и спец-флагов нет.
        for bench in ("bbh", "hle"):
            assert "judge_llm_config" not in data["benchmarks"][bench]

    def test_all_judged_benches_share_global_slot_no_allowlist(
        self, artifacts_path, base_dir, hf_checkpoint, model_info
    ):
        # Любой бенч (включая бывших «отщепенцев» browsecomp/tau2 и sbspollux)
        # получает судью через общий слот — без per-field инжекта и allowlist.
        vc = _make_validation_config()
        vc["benchmarks"]["sbspollux"] = {"name": "sbspollux-adapter"}
        vc["models"][0]["benchmarks"].append("sbspollux")
        cfg = _make_cfg(
            base_dir,
            judges={"judge-a": ["browsecomp", "tau2", "sbspollux"]},
            validation_config=vc,
        )
        _apply_defaults(cfg, self.JUDGE_BASE)
        ev = _build_evaluator(cfg, artifacts_path, hf_checkpoint, model_info)
        data = _judged_config_for(ev, "judge-a")

        assert set(data["benchmarks"]) == {"browsecomp", "tau2", "sbspollux"}
        judge = data["defaults"]["benchmark_config"]["judge_llm_config"]
        assert judge["api_url"] == self.JUDGE_API
        assert judge["name"] == "judge-a"
        # Старые per-field поля бенчей PAM не трогает.
        for bench in ("browsecomp", "tau2", "sbspollux"):
            assert "judge_llm_config" not in data["benchmarks"][bench]
        assert data["benchmarks"]["browsecomp"]["parameters"]["n_repeats"] == 1

    def test_models_filtered_to_task_subset(
        self, artifacts_path, base_dir, hf_checkpoint, model_info
    ):
        ev = self._evaluator(
            artifacts_path,
            base_dir,
            hf_checkpoint,
            model_info,
            judges={"judge-a": ["bbh"], "judge-b": ["tau2"]},
        )
        data = _judged_config_for(ev, "judge-a")

        assert set(data["benchmarks"]) == {"bbh"}
        assert data["models"][0]["benchmarks"] == ["bbh"]

    def test_no_model_references_subset_yields_empty_models(
        self, artifacts_path, base_dir, hf_checkpoint, model_info
    ):
        # Как obm_judged: конфигу доверяем, не роняем замер. Если ни одна
        # модель не ссылается на бенчи судьи — пишем пустой models, GM-сабпроцесс
        # просто не даст метрик по этому судье.
        validation_config = _make_validation_config()
        validation_config["models"][0]["benchmarks"] = ["ragqa"]
        cfg = _make_cfg(
            base_dir,
            judges={"judge-a": ["bbh"]},
            validation_config=validation_config,
        )
        _apply_defaults(cfg, self.JUDGE_BASE)
        ev = _build_evaluator(cfg, artifacts_path, hf_checkpoint, model_info)

        data = _judged_config_for(ev, "judge-a")
        assert data["models"] == []


# ---------------------------------------------------------------------------
# run_on_model_name
# ---------------------------------------------------------------------------


class TestGigaMetricJudgedRunOnModelName:
    def test_routed_into_parallel_phase(
        self, artifacts_path, base_dir, hf_checkpoint, model_info
    ):
        # Пайплайн гоняет judged-эвалуаторы только через run_on_model_name:
        # ParallelInferenceEvaluator уводит эвалуатор в параллельную фазу,
        # run() там не вызывается.
        from node_runner.src.metrics.evaluators.interface import (
            ParallelInferenceEvaluator,
        )
        from node_runner.src.metrics.metrics_manager import EVALUATORS

        cfg = _make_cfg(base_dir)
        ev = _build_evaluator(cfg, artifacts_path, hf_checkpoint, model_info)
        assert isinstance(ev, ParallelInferenceEvaluator)
        assert EVALUATORS["gigametric_judged"] is GigaMetricJudgedEvaluator

    def test_run_on_model_name_runs_subprocess_with_judged_overrides(
        self, monkeypatch, artifacts_path, base_dir, hf_checkpoint, model_info
    ):
        cfg = _make_cfg(base_dir, judges={"judge-a": ["bbh"]})
        _apply_defaults(cfg)
        ev = _build_evaluator(cfg, artifacts_path, hf_checkpoint, model_info)
        monkeypatch.setattr(ev, "_validate_runtime", lambda: None)

        captured: dict[str, object] = {"start": 0, "wait": 0, "log_error": 0}

        class _SP:
            def __init__(self, cmd, log_file, cwd=None, env=None, run_name=None):
                captured["cmd"] = cmd

            def start(self):
                captured["start"] = int(captured["start"]) + 1

            def wait_for_completion(self):
                captured["wait"] = int(captured["wait"]) + 1

            def log_error(self):
                captured["log_error"] = int(captured["log_error"]) + 1

            def is_successful(self):
                return True

        monkeypatch.setattr(
            "node_runner.src.metrics.evaluators.gigametric.SubprocessManager", _SP
        )

        ev.run_on_model_name("judge-a")

        assert captured["start"] == 1
        assert captured["wait"] == 1
        assert captured["log_error"] == 1
        cmd = captured["cmd"]
        cfg_idx = cmd.index("--config")
        assert cmd[cfg_idx + 1].endswith("_pam_judged_config_judge_a.yaml")
        vid_idx = cmd.index("--validation-id")
        assert cmd[vid_idx + 1] == f"{ev.validation_id}_judge_a"
        assert "--resume" not in cmd
        storage_idx = cmd.index("--storage-base-path")
        assert Path(cmd[storage_idx + 1]).name == f"{ev.validation_id}_judge_a"


# ---------------------------------------------------------------------------
# get_metrics — merge по judge-ранам + reshape
# ---------------------------------------------------------------------------


class TestGigaMetricJudgedGetMetrics:
    def test_merges_metrics_across_judge_runs(
        self, artifacts_path, base_dir, hf_checkpoint, model_info
    ):
        cfg = _make_cfg(
            base_dir, judges={"judge-a": ["bbh", "hle"], "judge-b": ["tau2"]}
        )
        ev = _build_evaluator(cfg, artifacts_path, hf_checkpoint, model_info)

        _write_state(
            base_dir,
            f"{ev.validation_id}_judge_a",
            [
                {
                    "status": "completed",
                    "model_id": "local",
                    "benchmark_id": "bbh",
                    "primary_metric_name": "exact_match_avg",
                    "primary_metric": 0.5,
                },
                {
                    "status": "completed",
                    "model_id": "local",
                    "benchmark_id": "hle",
                    "primary_metric_name": "exact_match_avg",
                    "primary_metric": 0.25,
                },
            ],
        )
        _write_state(
            base_dir,
            f"{ev.validation_id}_judge_b",
            [
                {
                    "status": "completed",
                    "model_id": "local",
                    "benchmark_id": "tau2",
                    "primary_metric_name": "pass^1",
                    "primary_metric": 0.75,
                },
            ],
        )

        metrics = ev.get_metrics()

        assert metrics == {
            "gigametric/bbh/exact_match_avg": 0.5,
            "gigametric/hle/exact_match_avg": 0.25,
            "gigametric/tau2/pass^1": 0.75,
        }

    def test_returns_raw_without_local_model_id(
        self, artifacts_path, base_dir, hf_checkpoint, model_info
    ):
        cfg = _make_cfg(base_dir, judges={"judge-a": ["bbh"]}, local_model_id=None)
        ev = _build_evaluator(cfg, artifacts_path, hf_checkpoint, model_info)
        _write_state(
            base_dir,
            f"{ev.validation_id}_judge_a",
            [
                {
                    "status": "completed",
                    "model_id": "local",
                    "benchmark_id": "bbh",
                    "primary_metric_name": "exact_match_avg",
                    "primary_metric": 0.5,
                },
            ],
        )
        assert ev.get_metrics() == {"gigametric/local/bbh/exact_match_avg": 0.5}


# ---------------------------------------------------------------------------
# _gigametric_version — per-judge fallback
# ---------------------------------------------------------------------------


class TestGigaMetricJudgedVersion:
    def test_reads_version_from_first_available_judge_run(
        self, artifacts_path, base_dir, hf_checkpoint, model_info
    ):
        cfg = _make_cfg(base_dir, judges={"judge-a": ["bbh"], "judge-b": ["tau2"]})
        ev = _build_evaluator(cfg, artifacts_path, hf_checkpoint, model_info)

        run_dir = base_dir / f"{ev.validation_id}_judge_b"
        run_dir.mkdir(parents=True)
        (run_dir / "gigametric_version.txt").write_text(
            "version=1.0.0\ngit_commit=abc123", encoding="utf-8"
        )

        assert ev._gigametric_version() == "version=1.0.0 | git_commit=abc123"

    def test_falls_back_to_shooter_version(
        self, artifacts_path, base_dir, hf_checkpoint, model_info
    ):
        cfg = _make_cfg(base_dir)
        ev = _build_evaluator(cfg, artifacts_path, hf_checkpoint, model_info)
        assert ev._gigametric_version() == "image-baked"

    def test_falls_back_when_version_file_unreadable(
        self, monkeypatch, artifacts_path, base_dir, hf_checkpoint, model_info
    ):
        cfg = _make_cfg(base_dir, judges={"judge-a": ["bbh"]})
        ev = _build_evaluator(cfg, artifacts_path, hf_checkpoint, model_info)

        run_dir = base_dir / f"{ev.validation_id}_judge_a"
        run_dir.mkdir(parents=True)
        (run_dir / "gigametric_version.txt").write_text("v", encoding="utf-8")

        def _raise_on_read(self: Path, *_a, **_kw) -> str:
            raise OSError("simulated read failure")

        monkeypatch.setattr(Path, "read_text", _raise_on_read)
        assert ev._gigametric_version() == "image-baked"


# ---------------------------------------------------------------------------
# validation_config_path mode
# ---------------------------------------------------------------------------


class TestGigaMetricJudgedPathMode:
    def test_loads_validation_config_from_path(
        self, tmp_path, artifacts_path, base_dir, hf_checkpoint, model_info
    ):
        config_path = tmp_path / "validation.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(_make_validation_config(), f)

        cfg = GigaMetricJudgedEvaluatorConfig(
            validation_config_path=config_path,
            base_dir=base_dir,
            local_model_id="local",
            judges={"judge-a": ["bbh"]},
        )
        ev = _build_evaluator(cfg, artifacts_path, hf_checkpoint, model_info)

        assert ev._base_validation_config["validation_name"] == "judged_smoke"
        assert list(ev._judge_tasks) == ["judge-a"]
