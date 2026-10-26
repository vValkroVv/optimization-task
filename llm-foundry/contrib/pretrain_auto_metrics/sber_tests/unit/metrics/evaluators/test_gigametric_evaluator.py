"""Unit-тесты GigaMetricEvaluator.

Subprocess замокан. On-disk-контракт (схема state.json, расположение
файла версии) проверяется передачей адаптеру заранее собранного дерева
``validation_runs/<validation_id>/``.
"""

import fcntl
import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml

from node_runner.src.job_dispatcher import JD
from node_runner.src.config_model import (
    GigaMetricEvaluatorConfig,
    GigaMetricJudgedEvaluatorConfig,
    DEFAULT_PARALLEL_PER_INFERENCE_INSTANCE,
    ModelInfoConfig,
)
from node_runner.src.metrics.evaluators.gigametric import (
    GigaMetricEvaluator,
    GigaMetricJudgedEvaluator,
)

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_evaluator(
    cfg: GigaMetricEvaluatorConfig,
    artifacts_path: Path,
    hf_checkpoint: Path,
    model_info: ModelInfoConfig,
) -> GigaMetricEvaluator:
    ev = GigaMetricEvaluator()
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
# Autouse fake validation.pam_export (gigametric — больше не submodule)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _fake_validation_pam_export(monkeypatch: pytest.MonkeyPatch) -> None:
    """Подменяет ``validation.pam_export`` фейком, читающим state.json.

    Раньше contrib/papayametric был git-submodule и реальный
    ``validation.pam_export.get_pam_metrics`` подгружался из сабмодуля.
    Сейчас gigametric кладётся только в Docker-образ → в CI/локально для
    юнит-тестов модуль недоступен. Фейк читает on-disk state.json по тому
    же контракту и отдаёт ключи ``gigametric/<model_id>/<bench>/<metric>``.
    Тесты, которым нужен свой фейк (skips_empty_bench_tail,
    finally_swallows_remove_failure), переопределяют sys.modules повторно
    в своём scope.
    """
    import types

    def _get_pam_metrics(validation_id: str, base: Path) -> dict[str, float]:
        metrics_file = Path(base) / f"{validation_id}.pam_metrics.json"
        if metrics_file.is_file():
            return json.loads(metrics_file.read_text(encoding="utf-8"))
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
# GigaMetricEvaluatorConfig validator (path/inline discriminator)
# ---------------------------------------------------------------------------


def test_config_inline_only_ok(tmp_path: Path) -> None:
    cfg = GigaMetricEvaluatorConfig(
        validation_config={"validation_name": "x", "models": []}, base_dir=tmp_path
    )
    assert cfg.validation_config_path is None
    target = tmp_path / "validation.yaml"
    cfg.materialise_validation_config(
        target=target, example_full=tmp_path / "unused.yaml", model_name="model"
    )
    loaded = yaml.safe_load(target.read_text())
    assert loaded["defaults"]["llm_config"]["parameters"]["timeout"] == 21600
    assert "timeout" not in loaded["defaults"]["llm_config"]


def test_config_path_only_ok(tmp_path: Path) -> None:
    cfg = GigaMetricEvaluatorConfig(
        validation_config_path=tmp_path / "v.yaml", base_dir=tmp_path
    )
    assert cfg.validation_config is None


def test_config_both_rejected(tmp_path: Path) -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        GigaMetricEvaluatorConfig(
            validation_config={"x": 1},
            validation_config_path=tmp_path / "v.yaml",
            base_dir=tmp_path,
        )


def test_config_neither_rejected(tmp_path: Path) -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        GigaMetricEvaluatorConfig(base_dir=tmp_path)


def test_config_benchmarks_only_ok(tmp_path: Path) -> None:
    cfg = GigaMetricEvaluatorConfig(benchmarks={"mmlu_pro": {}}, base_dir=tmp_path)
    assert cfg.validation_config is None
    assert cfg.validation_config_path is None
    assert list(cfg.benchmarks) == ["mmlu_pro"]


def test_config_benchmarks_use_runtime_defaults() -> None:
    cfg = GigaMetricEvaluatorConfig(benchmarks={"mmlu_pro": {}})

    assert cfg.base_dir is None
    assert cfg.local_model_id is None


@pytest.mark.parametrize("benchmarks", [None, ["mmlu_pro"]])
def test_config_rejects_non_mapping_benchmarks(benchmarks) -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError) as exc:
        GigaMetricEvaluatorConfig(benchmarks=benchmarks)

    assert [(error["loc"], error["type"]) for error in exc.value.errors()] == [
        (("benchmarks",), "dict_type")
    ]


def test_config_benchmarks_with_full_rejected(tmp_path: Path) -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        GigaMetricEvaluatorConfig(
            benchmarks={"mmlu_pro": {}},
            validation_config={"x": 1},
            base_dir=tmp_path,
        )


def _write_example_full(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "validation_name": "example",
                "defaults": {
                    "benchmark_config": {
                        "judge_llm_config": {
                            "api_key": "${GIGACHAT_IFT_TOKEN}",
                        },
                        "fail_fast_consecutive_item_errors": 20,
                    }
                },
                "benchmarks": {
                    "mmlu_pro": {
                        "name": "mmlu-pro-adapter",
                        "primary_metric": "exact_match_avg",
                    },
                    "bfcl_v4": {
                        "name": "bfcl-v4-adapter",
                        "primary_metric": "accuracy_overall",
                        "parameters": {
                            "num_items": 100,
                            "nested": {"kept": 1},
                        },
                    },
                    "gpqa": {
                        "name": "gpqa-adapter",
                        "primary_metric": "accuracy_5_shot_avg",
                    },
                    "sber_kib_table": {
                        "name": "sber-kib-table-adapter",
                        "primary_metric": "exact_match_avg",
                    },
                },
                "model_refs": ["leftover.yaml"],
            },
            f,
        )


def _materialise_minimal_config(
    cfg: GigaMetricEvaluatorConfig, root: Path
) -> dict[str, Any]:
    cfg.apply_defaults(inference_instances=1, base_url="http://127.0.0.1:29999")
    cfg.materialise_validation_config(
        target=root / "validation.yaml",
        example_full=root / "validation" / "configs" / "example_full.yaml",
        model_name="Qwen3-4B",
    )
    assert cfg.validation_config is not None
    return cfg.validation_config


def test_build_minimal_filters_overrides_and_injects_model(tmp_path: Path) -> None:
    _write_example_full(tmp_path / "validation" / "configs" / "example_full.yaml")
    cfg = GigaMetricEvaluatorConfig(
        benchmarks={
            "mmlu_pro": {},
            "bfcl_v4": {
                "preset": "bfcl_v4",
                "num_items": 5,
                "primary_metric": "custom_metric",
            },
        },
        base_dir=tmp_path,
        local_model_id="qwen_local",
        max_concurrent_requests=3,
    )
    built = _materialise_minimal_config(cfg, tmp_path)

    assert set(built["benchmarks"]) == {"mmlu_pro", "bfcl_v4"}
    assert built["benchmarks"]["mmlu_pro"]["primary_metric"] == "exact_match_avg"
    assert built["benchmarks"]["bfcl_v4"]["parameters"]["num_items"] == 5
    assert built["benchmarks"]["bfcl_v4"]["primary_metric"] == "custom_metric"
    assert built["benchmarks"]["bfcl_v4"]["name"] == "bfcl-v4-adapter"
    model = built["models"][0]
    assert model["id"] == "qwen_local"
    assert model["llm_config"]["api_url"] == "http://127.0.0.1:29999/v1"
    assert model["llm_config"]["name"] == "Qwen3-4B"
    assert model["llm_config"]["parameters"] == {"stream": False}
    assert model["benchmarks"] == ["mmlu_pro", "bfcl_v4"]
    assert "model_refs" not in built
    bench_defaults = built.get("defaults", {}).get("benchmark_config", {})
    assert "judge_llm_config" not in bench_defaults
    assert bench_defaults["parameters"]["max_concurrent_requests"] == 3


def test_build_minimal_defaults_local_model_id_to_model_name(tmp_path: Path) -> None:
    _write_example_full(tmp_path / "validation" / "configs" / "example_full.yaml")
    cfg = GigaMetricEvaluatorConfig(benchmarks={"mmlu_pro": {}})

    built = _materialise_minimal_config(cfg, tmp_path)

    assert cfg.local_model_id == "Qwen3-4B"
    assert built["models"][0]["id"] == "Qwen3-4B"


def test_build_minimal_resolves_papaya_preset_without_exposing_internal_id(
    tmp_path: Path,
) -> None:
    _write_example_full(tmp_path / "validation" / "configs" / "example_full.yaml")
    cfg = GigaMetricEvaluatorConfig(benchmarks={"papaya_kib_table": {}})

    built = _materialise_minimal_config(cfg, tmp_path)

    assert list(built["benchmarks"]) == ["papaya_kib_table"]
    assert built["benchmarks"]["papaya_kib_table"]["name"] == ("sber-kib-table-adapter")
    assert built["models"][0]["benchmarks"] == ["papaya_kib_table"]


@pytest.mark.parametrize(
    "benchmarks",
    [
        {"sber_kib_table": {}},
        {"kib_table": {"preset": "sber_kib_table"}},
        {"kib_table": {"parameters": {"bucket": "sber_internal"}}},
    ],
)
def test_minimal_config_rejects_internal_benchmark_names(
    benchmarks: dict[str, dict[str, Any]],
) -> None:
    with pytest.raises(ValueError, match="PAPAYA aliases"):
        GigaMetricEvaluatorConfig(benchmarks=benchmarks)


def test_setup_defaults_base_dir_inside_metrics_dir(
    artifacts_path: Path,
    hf_checkpoint: Path,
    model_info: ModelInfoConfig,
) -> None:
    cfg = GigaMetricEvaluatorConfig(validation_config={"validation_name": "x"})

    ev = _build_evaluator(cfg, artifacts_path, hf_checkpoint, model_info)

    assert ev.base_dir == Path(ev.metrics_dir) / "runs"
    assert ev.base_dir.is_dir()


def test_build_minimal_skips_unknown_benchmark(tmp_path: Path) -> None:
    _write_example_full(tmp_path / "validation" / "configs" / "example_full.yaml")
    cfg = GigaMetricEvaluatorConfig(
        benchmarks={"mmlu_pro": {}, "nonexistent": {}}, base_dir=tmp_path
    )

    with patch("node_runner.src.config_model.Logger.warning") as warning:
        built = _materialise_minimal_config(cfg, tmp_path)

    assert list(built["benchmarks"]) == ["mmlu_pro"]
    assert built["models"][0]["benchmarks"] == ["mmlu_pro"]
    message = warning.call_args.args[0]
    assert "nonexistent" in message
    assert "skipped" in message


def test_build_minimal_skips_non_string_preset(tmp_path: Path) -> None:
    _write_example_full(tmp_path / "validation" / "configs" / "example_full.yaml")
    cfg = GigaMetricEvaluatorConfig(
        benchmarks={"mmlu_pro": {}, "invalid": {"preset": 1}}, base_dir=tmp_path
    )

    with patch("node_runner.src.config_model.Logger.warning") as warning:
        built = _materialise_minimal_config(cfg, tmp_path)

    assert list(built["benchmarks"]) == ["mmlu_pro"]
    assert "preset must be a string" in warning.call_args.args[0]


def test_build_minimal_missing_example_full_does_not_abort(tmp_path: Path) -> None:
    cfg = GigaMetricEvaluatorConfig(benchmarks={"mmlu_pro": {}}, base_dir=tmp_path)

    with patch("node_runner.src.config_model.Logger.error") as error:
        built = _materialise_minimal_config(cfg, tmp_path)

    assert built["benchmarks"] == {}
    assert built["models"][0]["benchmarks"] == []
    assert "defaults not found" in error.call_args.args[0]


def test_build_minimal_non_mapping_example_full_does_not_abort(tmp_path: Path) -> None:
    bad = tmp_path / "validation" / "configs" / "example_full.yaml"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text("- just\n- a\n- list\n", encoding="utf-8")
    cfg = GigaMetricEvaluatorConfig(benchmarks={"mmlu_pro": {}}, base_dir=tmp_path)

    with patch("node_runner.src.config_model.Logger.error") as error:
        built = _materialise_minimal_config(cfg, tmp_path)

    assert built["benchmarks"] == {}
    assert built["models"][0]["benchmarks"] == []
    assert "not a mapping" in error.call_args.args[0]


def test_build_minimal_non_mapping_benchmark_catalog_does_not_abort(
    tmp_path: Path,
) -> None:
    example = tmp_path / "validation" / "configs" / "example_full.yaml"
    _write_example_full(example)
    example_config = yaml.safe_load(example.read_text())
    example_config["benchmarks"] = [{"invalid": True}]
    example.write_text(yaml.safe_dump(example_config), encoding="utf-8")
    cfg = GigaMetricEvaluatorConfig(benchmarks={"mmlu_pro": {}}, base_dir=tmp_path)

    with patch("node_runner.src.config_model.Logger.error") as error:
        built = _materialise_minimal_config(cfg, tmp_path)

    assert built["benchmarks"] == {}
    assert built["models"][0]["benchmarks"] == []
    assert "catalog is not a mapping" in error.call_args.args[0]


def test_build_minimal_entry_field_deep_merges_and_wraps_params(tmp_path: Path) -> None:
    _write_example_full(tmp_path / "validation" / "configs" / "example_full.yaml")
    cfg = GigaMetricEvaluatorConfig(
        benchmarks={
            "bfcl_v4": {"parameters": {"timeout": 60, "nested": {"added": 2}}},
            "mmlu_pro": {"temperature": 0.2},
        },
        base_dir=tmp_path,
    )
    built = _materialise_minimal_config(cfg, tmp_path)

    merged = built["benchmarks"]["bfcl_v4"]["parameters"]
    assert merged == {
        "num_items": 100,
        "timeout": 60,
        "nested": {"kept": 1, "added": 2},
    }
    assert built["benchmarks"]["mmlu_pro"]["parameters"] == {"temperature": 0.2}


def test_materialise_config_from_minimal_writes_yaml(
    tmp_path: Path,
) -> None:
    _write_example_full(tmp_path / "validation" / "configs" / "example_full.yaml")
    cfg = GigaMetricEvaluatorConfig(benchmarks={"mmlu_pro": {}}, base_dir=tmp_path)
    cfg.apply_defaults(inference_instances=1, base_url="http://127.0.0.1:29999")
    target = tmp_path / "metrics" / "gigametric" / "_pam_inline_config.yaml"
    target.parent.mkdir(parents=True)
    result = cfg.materialise_validation_config(
        target=target,
        example_full=tmp_path / "validation" / "configs" / "example_full.yaml",
        model_name="Qwen3-4B",
    )

    assert result == target
    assert target.exists()
    loaded = yaml.safe_load(target.read_text())
    assert set(loaded["benchmarks"]) == {"mmlu_pro"}
    assert loaded["defaults"]["llm_config"]["parameters"]["timeout"] == 21600
    assert "timeout" not in loaded["defaults"]["llm_config"]


def test_materialise_minimal_requires_inference_base_url(tmp_path: Path) -> None:
    _write_example_full(tmp_path / "validation" / "configs" / "example_full.yaml")
    cfg = GigaMetricEvaluatorConfig(benchmarks={"mmlu_pro": {}}, base_dir=tmp_path)

    with pytest.raises(ValueError, match="inference_base_url"):
        cfg.materialise_validation_config(
            target=tmp_path / "validation.yaml",
            example_full=tmp_path / "validation" / "configs" / "example_full.yaml",
            model_name="Qwen3-4B",
        )


# apply_defaults — semaphore = inference_instances * DEFAULT_PARALLEL_PER_INFERENCE_INSTANCE
# ---------------------------------------------------------------------------


def test_apply_defaults_sets_semaphore_from_inference_instances(tmp_path: Path) -> None:
    cfg = GigaMetricEvaluatorConfig(
        validation_config={
            "models": [{"id": "local", "llm_config": {"model": "openai"}}]
        },
        base_dir=tmp_path,
        local_model_id="local",
    )
    cfg.apply_defaults(inference_instances=4, base_url="http://127.0.0.1:29999")
    assert (
        cfg.validation_config["models"][0]["llm_config"]["semaphore"]
        == 4 * DEFAULT_PARALLEL_PER_INFERENCE_INSTANCE
    )


def test_apply_defaults_preserves_explicit_semaphore(tmp_path: Path) -> None:
    cfg = GigaMetricEvaluatorConfig(
        validation_config={"models": [{"id": "local", "llm_config": {"semaphore": 3}}]},
        base_dir=tmp_path,
        local_model_id="local",
    )
    cfg.apply_defaults(inference_instances=8, base_url="http://127.0.0.1:29999")
    assert cfg.validation_config["models"][0]["llm_config"]["semaphore"] == 3


def test_apply_defaults_skips_non_local_model(tmp_path: Path) -> None:
    cfg = GigaMetricEvaluatorConfig(
        validation_config={
            "models": [
                {"id": "local", "llm_config": {"model": "openai"}},
                {"id": "other", "llm_config": {"model": "gigachat"}},
            ]
        },
        base_dir=tmp_path,
        local_model_id="local",
    )
    cfg.apply_defaults(inference_instances=2, base_url="http://127.0.0.1:29999")
    assert (
        cfg.validation_config["models"][0]["llm_config"]["semaphore"]
        == 2 * DEFAULT_PARALLEL_PER_INFERENCE_INSTANCE
    )
    assert "semaphore" not in cfg.validation_config["models"][1]["llm_config"]


def test_apply_defaults_noop_when_local_model_id_unset(tmp_path: Path) -> None:
    cfg = GigaMetricEvaluatorConfig(
        validation_config={
            "models": [{"id": "local", "llm_config": {"model": "openai"}}]
        },
        base_dir=tmp_path,
    )
    cfg.apply_defaults(inference_instances=2, base_url="http://127.0.0.1:29999")
    assert "semaphore" not in cfg.validation_config["models"][0]["llm_config"]


def test_apply_defaults_noop_for_path_config(tmp_path: Path) -> None:
    cfg = GigaMetricEvaluatorConfig(
        validation_config_path=tmp_path / "v.yaml",
        base_dir=tmp_path,
        local_model_id="local",
    )
    cfg.apply_defaults(inference_instances=2, base_url="http://127.0.0.1:29999")
    # No raise; validation_config_path untouched.
    assert cfg.validation_config_path == tmp_path / "v.yaml"


def test_apply_defaults_skips_non_dict_entries(tmp_path: Path) -> None:
    cfg = GigaMetricEvaluatorConfig(
        validation_config={
            "models": [
                "not-a-dict",
                {"id": "local"},  # no llm_config
                {"id": "local2", "llm_config": "not-a-dict"},
                {"id": "local"},  # duplicate id, still no llm_config
            ]
        },
        base_dir=tmp_path,
        local_model_id="local",
    )
    cfg.apply_defaults(
        inference_instances=2, base_url="http://127.0.0.1:29999"
    )  # должно не упасть


# ---------------------------------------------------------------------------
# setup()
# ---------------------------------------------------------------------------


def test_setup_materialises_inline_config(
    base_dir: Path,
    artifacts_path: Path,
    hf_checkpoint: Path,
    model_info: ModelInfoConfig,
) -> None:
    cfg = GigaMetricEvaluatorConfig(
        validation_config={"validation_name": "pam_smoke", "strategy": "breadth_first"},
        base_dir=base_dir,
    )
    ev = _build_evaluator(cfg, artifacts_path, hf_checkpoint, model_info)

    materialised = ev.config_dir / "_pam_inline_config.yaml"
    assert materialised.exists()
    loaded = yaml.safe_load(materialised.read_text())
    assert loaded["validation_name"] == "pam_smoke"
    assert ev.config_path == materialised


def test_setup_uses_validation_config_path_verbatim(
    base_dir: Path,
    artifacts_path: Path,
    hf_checkpoint: Path,
    model_info: ModelInfoConfig,
    tmp_path: Path,
) -> None:
    external = tmp_path / "external_cfg.yaml"
    external.write_text("validation_name: external\n")
    cfg = GigaMetricEvaluatorConfig(validation_config_path=external, base_dir=base_dir)
    ev = _build_evaluator(cfg, artifacts_path, hf_checkpoint, model_info)
    assert ev.config_path == external.resolve()
    # В path-режиме inline-shadow-файл создаваться НЕ должен.
    assert not (
        artifacts_path / "metrics" / "gigametric" / "_pam_inline_config.yaml"
    ).exists()


def test_setup_isolates_validation_for_same_checkpoint(
    base_dir: Path,
    artifacts_path: Path,
    hf_checkpoint: Path,
    model_info: ModelInfoConfig,
) -> None:
    cfg = GigaMetricEvaluatorConfig(
        validation_config={"validation_name": "x"}, base_dir=base_dir
    )
    ev = _build_evaluator(cfg, artifacts_path, hf_checkpoint, model_info)
    assert ev.validation_id.startswith("pam_")
    ev2 = _build_evaluator(cfg, artifacts_path, hf_checkpoint, model_info)
    assert ev.validation_id != ev2.validation_id
    ev3 = _build_evaluator(
        cfg, artifacts_path, hf_checkpoint.parent / "other", model_info
    )
    assert ev.validation_id != ev3.validation_id


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------


def test_run_uses_current_no_arg_contract(
    base_dir: Path,
    artifacts_path: Path,
    hf_checkpoint: Path,
    model_info: ModelInfoConfig,
) -> None:
    cfg = GigaMetricEvaluatorConfig(
        validation_config={"validation_name": "x"}, base_dir=base_dir
    )
    ev = _build_evaluator(cfg, artifacts_path, hf_checkpoint, model_info)
    with (
        patch(
            "node_runner.src.metrics.evaluators.gigametric.SubprocessManager"
        ) as mock_sm,
        patch.object(ev, "_validate_runtime"),
    ):
        instance = MagicMock()
        mock_sm.return_value = instance
        ev.run()
    mock_sm.assert_called_once()
    instance.start.assert_called_once()


def test_run_subprocess_command_shape(
    base_dir: Path,
    artifacts_path: Path,
    hf_checkpoint: Path,
    model_info: ModelInfoConfig,
) -> None:
    cfg = GigaMetricEvaluatorConfig(
        validation_config={"validation_name": "x"}, base_dir=base_dir
    )
    ev = _build_evaluator(cfg, artifacts_path, hf_checkpoint, model_info)

    with (
        patch(
            "node_runner.src.metrics.evaluators.gigametric.SubprocessManager"
        ) as mock_sm,
        patch.object(ev, "_validate_runtime"),
    ):
        instance = MagicMock()
        mock_sm.return_value = instance
        ev.run()

    args, kwargs = mock_sm.call_args
    cmd = args[0]
    assert Path(cmd[1]) == ev._papayametric_root() / "run_validation.py"
    assert "--config" in cmd
    assert "--base-dir" in cmd
    assert "--validation-id" in cmd
    assert "--storage-base-path" in cmd
    assert not {
        "--resume",
        "--rerun-failed",
        "--recover-stale-running",
        "--allow-config-change",
        "--yes",
    }.intersection(cmd)
    assert cmd[cmd.index("--validation-id") + 1] == ev.validation_id
    storage_base = cmd[cmd.index("--storage-base-path") + 1]
    assert storage_base == str(
        Path("/home/user/papayametric_scoring") / ev.validation_id
    )
    instance.start.assert_called_once()
    instance.wait_for_completion.assert_called_once()


def test_run_subprocess_env_merges_config_env(
    base_dir: Path,
    artifacts_path: Path,
    hf_checkpoint: Path,
    model_info: ModelInfoConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("UNRELATED_VAR", "value-from-os")
    cfg = GigaMetricEvaluatorConfig(
        validation_config={"validation_name": "x"},
        base_dir=base_dir,
        env={"OPENAI_API_KEY": "sk-test-123"},
    )
    ev = _build_evaluator(cfg, artifacts_path, hf_checkpoint, model_info)

    with (
        patch(
            "node_runner.src.metrics.evaluators.gigametric.SubprocessManager"
        ) as mock_sm,
        patch.object(ev, "_validate_runtime"),
    ):
        mock_sm.return_value = MagicMock()
        ev.run()

    _args, kwargs = mock_sm.call_args
    env = kwargs["env"]
    assert env["OPENAI_API_KEY"] == "sk-test-123"
    assert env["UNRELATED_VAR"] == "value-from-os"  # os.environ passthrough
    assert env["UV_OFFLINE"] == "1"
    assert env["GZIP_ARTIFACTS"] == "1"
    assert env["ARTIFACTS_ARCHIVE_ENABLED"] == "1"


def test_build_env_adds_uv_bin_when_present(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    uv_bin = tmp_path / "gigametric_uv" / "bin"
    uv_bin.mkdir(parents=True)
    monkeypatch.setattr(GigaMetricEvaluator, "UV_BIN_PATH", uv_bin)
    monkeypatch.setenv("PATH", "/usr/bin")
    cfg = GigaMetricEvaluatorConfig(
        validation_config={"validation_name": "x"},
        base_dir=tmp_path / "base",
    )
    ev = GigaMetricEvaluator()
    ev.autometric_runs_config = cfg

    assert ev._build_env()["PATH"] == f"/usr/bin:{uv_bin}"


@pytest.mark.parametrize(
    ("pythonpath", "expected"),
    [
        (None, "/home/user/rli_sdk:/home/user/gigarli/src"),
        ("", "/home/user/rli_sdk:/home/user/gigarli/src"),
        ("/existing/src", "/home/user/rli_sdk:/home/user/gigarli/src:/existing/src"),
    ],
)
def test_build_env_preserves_pythonpath(
    pythonpath: str | None,
    expected: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PYTHONPATH", raising=False)
    if pythonpath is not None:
        monkeypatch.setenv("PYTHONPATH", pythonpath)
    ev = GigaMetricEvaluator()
    ev.autometric_runs_config = GigaMetricEvaluatorConfig(
        validation_config={"validation_name": "x"},
    )

    assert ev._build_env()["PYTHONPATH"] == expected


@pytest.mark.parametrize("exit_code", [0, 1, 2])
@pytest.mark.parametrize("export_run", [None, "current", "previous"])
def test_run_exports_metrics_after_success_or_current_run_export(
    exit_code: int,
    export_run: str | None,
    base_dir: Path,
    artifacts_path: Path,
    hf_checkpoint: Path,
    model_info: ModelInfoConfig,
) -> None:
    cfg = GigaMetricEvaluatorConfig(
        validation_config={"validation_name": "x"}, base_dir=base_dir
    )
    ev = _build_evaluator(cfg, artifacts_path, hf_checkpoint, model_info)

    with (
        patch(
            "node_runner.src.metrics.evaluators.gigametric.SubprocessManager"
        ) as mock_sm,
        patch.object(ev, "_validate_runtime"),
    ):
        instance = MagicMock()
        instance.is_successful.return_value = exit_code == 0
        mock_sm.return_value = instance
        ev.run()

    instance.log_error.assert_called_once()
    _write_state(
        base_dir,
        ev.validation_id,
        [
            {
                "status": "completed",
                "model_id": "local",
                "benchmark_id": "bbh",
                "primary_metric_name": "accuracy",
                "primary_metric": 0.5,
            }
        ],
    )
    metrics = {"gigametric/local/bbh/accuracy": 0.5}
    if export_run is not None:
        export_id = ev.validation_id if export_run == "current" else "pam_previous"
        (base_dir / f"{export_id}.pam_metrics.json").write_text(
            json.dumps(metrics), encoding="utf-8"
        )
    expected = metrics if exit_code == 0 or export_run == "current" else {}
    assert ev.get_metrics() == expected


@pytest.mark.parametrize(
    "suffixes", [(".tar.gz",), (".tar.gz.enc",), (".tar.gz", ".tar.gz.enc")]
)
@pytest.mark.parametrize("process_outcome", ["success", "failure", "exception"])
def test_run_preserves_archives_but_does_not_reuse_previous_run(
    suffixes,
    process_outcome,
    base_dir,
    artifacts_path,
    hf_checkpoint,
    model_info,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        GigaMetricEvaluator, "RUNTIME_ROOT", tmp_path / "pod" / "source"
    )
    monkeypatch.setattr(GigaMetricEvaluator, "_validate_runtime", lambda self: None)
    cfg = GigaMetricEvaluatorConfig(
        validation_config={"validation_name": "x"}, base_dir=base_dir
    )
    ev = _build_evaluator(cfg, artifacts_path, hf_checkpoint, model_info)
    archives = [
        ev._scoring_base_path().with_name(ev.validation_id + suffix)
        for suffix in suffixes
    ]

    def finish(*args):
        for archive in archives:
            archive.write_bytes(b"native scoring archive")
        if process_outcome == "exception":
            raise RuntimeError("process failed after archiving")
        return process_outcome == "success"

    monkeypatch.setattr(ev, "_run_validation_process", finish)
    if process_outcome == "exception":
        with pytest.raises(RuntimeError, match="process failed"):
            ev.run()
    else:
        ev.run()
    for archive in archives:
        assert (
            base_dir / "scoring" / archive.name
        ).read_bytes() == b"native scoring archive"
        archive.unlink()

    _write_state(
        base_dir,
        ev.validation_id,
        [
            {
                "status": "completed",
                "model_id": "local",
                "benchmark_id": "old",
                "primary_metric_name": "accuracy",
                "primary_metric": 0.5,
            }
        ],
    )
    (base_dir / ev.validation_id / "control.json").write_text('{"paused": true}')
    (base_dir / f"{ev.validation_id}.pam_metrics.json").write_text(
        '{"gigametric/local/old/accuracy": 0.5}'
    )
    restarted = _build_evaluator(cfg, artifacts_path, hf_checkpoint, model_info)

    def start_fresh(cmd, env, validation_id):
        assert "--resume" not in cmd
        assert validation_id != ev.validation_id
        assert not (base_dir / validation_id).exists()
        assert not restarted._scoring_base_path().exists()
        for archive in archives:
            assert not archive.exists()
            assert (
                not restarted._scoring_base_path()
                .with_name(validation_id + "".join(archive.suffixes))
                .exists()
            )
        return True

    monkeypatch.setattr(restarted, "_run_validation_process", start_fresh)
    restarted.run()
    assert restarted.get_metrics() == {}
    assert (base_dir / ev.validation_id / "control.json").is_file()


def test_run_refuses_concurrent_writer(
    base_dir, artifacts_path, hf_checkpoint, model_info, monkeypatch
):
    cfg = GigaMetricEvaluatorConfig(
        validation_config={"validation_name": "x"}, base_dir=base_dir
    )
    ev = _build_evaluator(cfg, artifacts_path, hf_checkpoint, model_info)
    monkeypatch.setattr(ev, "_validate_runtime", lambda: None)
    with (base_dir / f"{ev.validation_id}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with patch.object(ev, "_run_validation_process") as process:
            with pytest.raises(BlockingIOError):
                ev.run()
            process.assert_not_called()


def test_run_persists_only_anonymized_subprocess_log(
    base_dir: Path,
    artifacts_path: Path,
    hf_checkpoint: Path,
    model_info: ModelInfoConfig,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = GigaMetricEvaluatorConfig(
        validation_config={"validation_name": "x"}, base_dir=base_dir
    )
    ev = _build_evaluator(cfg, artifacts_path, hf_checkpoint, model_info)
    log_root = tmp_path / "runtime_logs"
    monkeypatch.setattr(GigaMetricEvaluator, "LOG_ROOT", log_root)
    monkeypatch.setattr(ev, "_validate_runtime", lambda: None)
    ev.log_file.write_text("previous\n", encoding="utf-8")

    class _SP:
        def __init__(self, cmd, log_file, cwd=None, env=None, run_name=None):
            self.log_file = log_file

        def start(self):
            self.log_file.write_text("SBER Giga СБЕР гига\n", encoding="utf-8")

        def wait_for_completion(self):
            pass

        def log_error(self):
            pass

        def is_successful(self):
            return True

    monkeypatch.setattr(
        "node_runner.src.metrics.evaluators.gigametric.SubprocessManager", _SP
    )

    ev.run()

    assert ev.log_file.read_text(encoding="utf-8") == (
        "previous\npapaya papaya papaya papaya\n"
    )
    assert list(log_root.iterdir()) == []
    assert not list(Path(ev.metrics_dir).glob("*.raw*"))


def test_run_anonymizes_subprocess_log_when_wait_fails(
    base_dir: Path,
    artifacts_path: Path,
    hf_checkpoint: Path,
    model_info: ModelInfoConfig,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = GigaMetricEvaluatorConfig(
        validation_config={"validation_name": "x"}, base_dir=base_dir
    )
    ev = _build_evaluator(cfg, artifacts_path, hf_checkpoint, model_info)
    log_root = tmp_path / "runtime_logs"
    monkeypatch.setattr(GigaMetricEvaluator, "LOG_ROOT", log_root)
    monkeypatch.setattr(ev, "_validate_runtime", lambda: None)

    class _SP:
        def __init__(self, cmd, log_file, cwd=None, env=None, run_name=None):
            self.log_file = log_file

        def start(self):
            self.log_file.write_text("Giga failed\n", encoding="utf-8")

        def wait_for_completion(self):
            raise RuntimeError("wait failed")

        def log_error(self):
            raise AssertionError("unreachable")

    monkeypatch.setattr(
        "node_runner.src.metrics.evaluators.gigametric.SubprocessManager", _SP
    )

    with pytest.raises(RuntimeError, match="wait failed"):
        ev.run()

    assert ev.log_file.read_text(encoding="utf-8") == "papaya failed\n"
    assert list(log_root.iterdir()) == []


# ---------------------------------------------------------------------------
# get_metrics()
# ---------------------------------------------------------------------------


def test_get_metrics_reads_state_via_pam_export(
    base_dir: Path,
    artifacts_path: Path,
    hf_checkpoint: Path,
    model_info: ModelInfoConfig,
) -> None:
    cfg = GigaMetricEvaluatorConfig(
        validation_config={"validation_name": "x"}, base_dir=base_dir
    )
    ev = _build_evaluator(cfg, artifacts_path, hf_checkpoint, model_info)

    _write_state(
        base_dir,
        ev.validation_id,
        [
            {
                "task_id": "m__b",
                "model_id": "gigachat_prod",
                "benchmark_id": "math",
                "adapter_name": "math500-adapter",
                "status": "completed",
                "primary_metric_name": "exact_match_avg",
                "primary_metric": 0.512,
            }
        ],
    )

    metrics = ev.get_metrics()
    assert metrics == {"gigametric/gigachat_prod/math/exact_match_avg": 0.512}


@pytest.mark.parametrize("local_model_id", [None, "model"])
@pytest.mark.parametrize(
    "benchmark,metric,value,expected",
    [
        ("bbac", "macro_average", 57.66, 0.5766),
        ("bbac", "mean_Accuracy_insensitive_multihop_choice", 0.0, 0.0),
        ("bbac", "mean_Accuracy_insensitive_multistep_choice", 0.5, 0.005),
        ("bbac", "mean_Accuracy_insensitive_single_call", 100.0, 1.0),
        ("bbac", "mean_Accuracy_insensitive_single_call_robustness", 75.0, 0.75),
        ("bbac", "count", 125.0, 125.0),
        ("reflectionbench", "mean_score", 126.79, 1.2679),
        ("reflectionbench", "mean_nback_score", 100.0, 1.0),
        ("reflectionbench", "mean_prlt_score", 50.0, 0.5),
        ("reflectionbench", "mean_wpt_score", -50.0, -0.5),
        ("reflectionbench", "mean_dcigt_score", 150.0, 1.5),
        ("other_benchmark", "mean_factology_mistake_count", 2.4, 2.4),
        ("other_bbac", "macro_average", 80.0, 80.0),
    ],
)
def test_get_metrics_scales_selected_metrics(
    base_dir: Path,
    artifacts_path: Path,
    hf_checkpoint: Path,
    model_info: ModelInfoConfig,
    local_model_id: str | None,
    benchmark: str,
    metric: str,
    value: float,
    expected: float,
) -> None:
    cfg = GigaMetricEvaluatorConfig(
        validation_config={"validation_name": "x"},
        base_dir=base_dir,
        local_model_id=local_model_id,
    )
    ev = _build_evaluator(cfg, artifacts_path, hf_checkpoint, model_info)
    _write_state(
        base_dir,
        ev.validation_id,
        [
            {
                "model_id": "model",
                "benchmark_id": benchmark,
                "status": "completed",
                "primary_metric_name": metric,
                "primary_metric": value,
            }
        ],
    )

    prefix = "gigametric" if local_model_id else "gigametric/model"
    expected_metrics = {f"{prefix}/{benchmark}/{metric}": expected}
    assert ev.get_metrics() == expected_metrics
    assert ev.get_metrics() == expected_metrics


@pytest.mark.parametrize("local_model_id", [None, "model"])
@pytest.mark.parametrize(
    "benchmark,raw_metrics,expected_metrics",
    [
        (
            "thinkbench",
            {
                "mean_accuracy": 0.8,
                "mean_precision": 0.7,
                "mean_recall": 0.6,
                "mean_reflection_quality": 0.5,
                "mean_tokens": 100.0,
                "mean_thought_num": 2.0,
                "mean_useful_tokens": 80.0,
                "mean_reflection_tokens": 20.0,
                "mean_token_efficiency": 0.8,
                "mean_basic_accuracy": 0.0,
                "new_metric": 0.9,
            },
            {
                "mean_accuracy": 0.8,
                "mean_precision": 0.7,
                "mean_recall": 0.6,
                "mean_reflection_quality": 0.5,
            },
        ),
        (
            "reflectionbench",
            {
                "mean_score": 75.0,
                "mean_nback_score": 100.0,
                "mean_prlt_score": 50.0,
                "mean_wpt_score": -50.0,
                "mean_dcigt_score": 200.0,
                "mean_oddball_score": 80.0,
                "new_metric": 0.9,
            },
            {
                "mean_score": 0.75,
                "mean_nback_score": 1.0,
                "mean_prlt_score": 0.5,
                "mean_wpt_score": -0.5,
                "mean_dcigt_score": 2.0,
            },
        ),
    ],
)
def test_get_metrics_applies_export_allowlist(
    base_dir: Path,
    artifacts_path: Path,
    hf_checkpoint: Path,
    model_info: ModelInfoConfig,
    local_model_id: str | None,
    benchmark: str,
    raw_metrics: dict[str, float],
    expected_metrics: dict[str, float],
) -> None:
    cfg = GigaMetricEvaluatorConfig(
        validation_config={"validation_name": "x"},
        base_dir=base_dir,
        local_model_id=local_model_id,
    )
    ev = _build_evaluator(cfg, artifacts_path, hf_checkpoint, model_info)
    raw = {
        f"gigametric/model/{benchmark}/{metric}": value
        for metric, value in raw_metrics.items()
    }
    original_raw = raw.copy()
    prefix = "gigametric" if local_model_id else "gigametric/model"
    expected = {
        f"{prefix}/{benchmark}/{metric}": value
        for metric, value in expected_metrics.items()
    }
    with patch("validation.pam_export.get_pam_metrics", return_value=raw):
        assert ev.get_metrics() == expected
        assert ev.get_metrics() == expected

    assert raw == original_raw


@pytest.mark.parametrize("local_model_id", [None, "model"])
@pytest.mark.parametrize("pass_rate", [0.0, 11 / 70, 1.0])
def test_get_metrics_exports_only_bank_agents_pass_rate(
    base_dir: Path,
    artifacts_path: Path,
    hf_checkpoint: Path,
    model_info: ModelInfoConfig,
    local_model_id: str | None,
    pass_rate: float,
) -> None:
    cfg = GigaMetricEvaluatorConfig(
        validation_config={"validation_name": "x"},
        base_dir=base_dir,
        local_model_id=local_model_id,
    )
    ev = _build_evaluator(cfg, artifacts_path, hf_checkpoint, model_info)
    benchmark = "bank_agents__ai_mentor_factology"
    raw = {
        f"gigametric/model/{benchmark}/mean_factology_pass": pass_rate,
        **{
            f"gigametric/model/{benchmark}/mean_factology_{kind}_count": 2.4
            for kind in (
                "mistake",
                "critical",
                "minor",
                "distortion",
                "fabrication",
                "omission",
                "contradiction",
            )
        },
        "gigametric/model/other_benchmark/mean_factology_mistake_count": 2.4,
    }
    prefix = "gigametric" if local_model_id else "gigametric/model"
    expected = {
        f"{prefix}/{benchmark}/mean_factology_pass": pass_rate,
        f"{prefix}/other_benchmark/mean_factology_mistake_count": 2.4,
    }
    with patch("validation.pam_export.get_pam_metrics", return_value=raw):
        metrics = ev.get_metrics()

    assert metrics == expected
    assert JD.generate_metrics_json({0: metrics}) == {0: expected}
    assert len(raw) == 9


def test_get_metrics_reshapes_for_local_model_id(
    base_dir: Path,
    artifacts_path: Path,
    hf_checkpoint: Path,
    model_info: ModelInfoConfig,
) -> None:
    cfg = GigaMetricEvaluatorConfig(
        validation_config={"validation_name": "x"},
        base_dir=base_dir,
        local_model_id="qwen_4b_local",
    )
    ev = _build_evaluator(cfg, artifacts_path, hf_checkpoint, model_info)
    _write_state(
        base_dir,
        ev.validation_id,
        [
            {
                "task_id": "m__a",
                "model_id": "qwen_4b_local",
                "benchmark_id": "math500",
                "adapter_name": "math500-adapter",
                "status": "completed",
                "primary_metric_name": "exact_match_avg",
                "primary_metric": 0.42,
            },
            {
                "task_id": "m__b",
                "model_id": "qwen_4b_local",
                "benchmark_id": "ifbench",
                "adapter_name": "ifbench-adapter",
                "status": "completed",
                "primary_metric_name": "score",
                "primary_metric": 0.9,
            },
            {
                "task_id": "m__c",
                "model_id": "gigachat_2_max_ift",
                "benchmark_id": "math500",
                "adapter_name": "math500-adapter",
                "status": "completed",
                "primary_metric_name": "exact_match_avg",
                "primary_metric": 0.7,
            },
        ],
    )
    metrics = ev.get_metrics()
    assert metrics == {
        "gigametric/math500/exact_match_avg": 0.42,
        "gigametric/ifbench/score": 0.9,
    }


def test_get_metrics_labels_toolathlon_and_sreda_no_api(
    base_dir: Path,
    artifacts_path: Path,
    hf_checkpoint: Path,
    model_info: ModelInfoConfig,
) -> None:
    cfg = GigaMetricEvaluatorConfig(
        validation_config={"validation_name": "x"},
        base_dir=base_dir,
        local_model_id="tested_model",
    )
    ev = _build_evaluator(cfg, artifacts_path, hf_checkpoint, model_info)
    raw = {
        "gigametric/tested_model/toolathlon/pass@1_TIMEOUT_2000": 0.5,
        "gigametric/tested_model/toolathlon/mean@1_finalpool_TIMEOUT_2000": 0.3,
        "gigametric/tested_model/toolathlon/pass@1_ru_office_local_TIMEOUT_2000": 0.7,
        "gigametric/tested_model/math500/exact_match_avg": 0.8,
    }
    with patch("validation.pam_export.get_pam_metrics", return_value=raw):
        metrics = ev.get_metrics()

    assert metrics == {
        "gigametric/toolathlon/pass@1_TIMEOUT_2000_no_api": 0.5,
        "gigametric/toolathlon/mean@1_finalpool_TIMEOUT_2000_no_api": 0.3,
        "gigametric/toolathlon/pass@1_ru_office_local_TIMEOUT_2000_no_api": 0.7,
        "gigametric/math500/exact_match_avg": 0.8,
    }


@pytest.mark.parametrize("timeout", [1800, 2000, 5400])
@pytest.mark.parametrize("local_model_id", [None, "tested_model"])
def test_get_metrics_uses_toolathlon_config_timeout(
    base_dir: Path,
    artifacts_path: Path,
    hf_checkpoint: Path,
    model_info: ModelInfoConfig,
    tmp_path: Path,
    timeout: int,
    local_model_id: str | None,
) -> None:
    external = tmp_path / "profile.yaml"
    external.write_text(
        yaml.safe_dump(
            {
                "benchmarks": {
                    "toolathlon": {"parameters": {"timeout_seconds": timeout}}
                },
            }
        )
    )
    cfg = GigaMetricEvaluatorConfig(
        validation_config_path=external,
        base_dir=base_dir,
        local_model_id=local_model_id,
    )
    ev = _build_evaluator(cfg, artifacts_path, hf_checkpoint, model_info)
    raw = {
        "gigametric/tested_model/toolathlon/mean_pass@3": 0.5,
        "gigametric/tested_model/toolathlon/mean_pass@1_finalpool": 0.3,
        "gigametric/tested_model/toolathlon/mean_pass^3_ru_office_local_TIMEOUT_999": 0.2,
        "gigametric/tested_model/math500/mean_exact_match": 0.8,
    }
    with patch("validation.pam_export.get_pam_metrics", return_value=raw):
        metrics = ev.get_metrics()
    prefix = "gigametric" if local_model_id else "gigametric/tested_model"
    suffix = f"_TIMEOUT_{timeout}" + ("_no_api" if local_model_id else "")
    assert metrics == {
        f"{prefix}/toolathlon/mean_pass@3{suffix}": 0.5,
        f"{prefix}/toolathlon/mean_pass@1_finalpool{suffix}": 0.3,
        f"{prefix}/toolathlon/mean_pass^3_ru_office_local{suffix}": 0.2,
        f"{prefix}/math500/mean_exact_match": 0.8,
    }


def test_setup_uses_gm_pam_catalog_and_preserves_overrides(
    base_dir: Path,
    artifacts_path: Path,
    hf_checkpoint: Path,
    model_info: ModelInfoConfig,
    tmp_path: Path,
) -> None:
    catalog = tmp_path / "gm" / "validation" / "configs" / "pam.yaml"
    catalog.parent.mkdir(parents=True)
    catalog.write_text(
        yaml.safe_dump(
            {
                "validation_name": "pam",
                "benchmarks": {
                    "toolathlon": {
                        "name": "toolathlon-adapter",
                        "aggregation": {"type": "mean", "group_by": "sub_benchmark"},
                        "parameters": {
                            "runner": "rli",
                            "num_attempts": 3,
                            "timeout_seconds": 1800,
                            "rli_tasks_from_mapping": True,
                        },
                    }
                },
            }
        )
    )
    cfg = GigaMetricEvaluatorConfig(
        benchmarks={"toolathlon": {"timeout_seconds": 5400}},
        base_dir=base_dir,
    )
    cfg.apply_defaults(inference_instances=1, base_url="http://localhost:29999")
    with patch.object(
        GigaMetricEvaluator, "_unzip_papayametric", return_value=tmp_path / "gm"
    ):
        ev = _build_evaluator(cfg, artifacts_path, hf_checkpoint, model_info)
    bench = yaml.safe_load(ev.config_path.read_text())["benchmarks"]["toolathlon"]
    assert bench["parameters"] == {
        "runner": "rli",
        "num_attempts": 3,
        "timeout_seconds": 5400,
        "rli_tasks_from_mapping": True,
    }
    assert bench["aggregation"] == {"type": "mean", "group_by": "sub_benchmark"}


def test_get_metrics_reshape_handles_empty_raw(
    base_dir: Path,
    artifacts_path: Path,
    hf_checkpoint: Path,
    model_info: ModelInfoConfig,
) -> None:
    cfg = GigaMetricEvaluatorConfig(
        validation_config={"validation_name": "x"},
        base_dir=base_dir,
        local_model_id="qwen_4b_local",
    )
    ev = _build_evaluator(cfg, artifacts_path, hf_checkpoint, model_info)
    assert ev.get_metrics() == {}


def test_get_metrics_reshape_raises_when_local_model_id_matches_none(
    base_dir: Path,
    artifacts_path: Path,
    hf_checkpoint: Path,
    model_info: ModelInfoConfig,
) -> None:
    cfg = GigaMetricEvaluatorConfig(
        validation_config={"validation_name": "x"},
        base_dir=base_dir,
        local_model_id="typo_model_id",
    )
    ev = _build_evaluator(cfg, artifacts_path, hf_checkpoint, model_info)
    _write_state(
        base_dir,
        ev.validation_id,
        [
            {
                "task_id": "m__a",
                "model_id": "qwen_4b_local",
                "benchmark_id": "math500",
                "adapter_name": "math500-adapter",
                "status": "completed",
                "primary_metric_name": "exact_match_avg",
                "primary_metric": 0.42,
            }
        ],
    )
    with pytest.raises(ValueError, match="matched 0 keys"):
        ev.get_metrics()


def test_get_metrics_reshape_keeps_multiple_metrics_per_bench(
    base_dir: Path,
    artifacts_path: Path,
    hf_checkpoint: Path,
    model_info: ModelInfoConfig,
) -> None:
    cfg = GigaMetricEvaluatorConfig(
        validation_config={"validation_name": "x"},
        base_dir=base_dir,
        local_model_id="qwen_4b_local",
    )
    ev = _build_evaluator(cfg, artifacts_path, hf_checkpoint, model_info)
    _write_state(
        base_dir,
        ev.validation_id,
        [
            {
                "task_id": "m__a",
                "model_id": "qwen_4b_local",
                "benchmark_id": "math500",
                "adapter_name": "math500-adapter",
                "status": "completed",
                "primary_metric_name": "exact_match_avg",
                "primary_metric": 0.42,
            },
            {
                "task_id": "m__b",
                "model_id": "qwen_4b_local",
                "benchmark_id": "math500",
                "adapter_name": "math500-adapter",
                "status": "completed",
                "primary_metric_name": "pass_at_1",
                "primary_metric": 0.55,
            },
        ],
    )
    metrics = ev.get_metrics()
    # Несколько метрик на один бенч сохраняются обе (ключ = bench/metric),
    # а не схлопываются по bench.
    assert metrics == {
        "gigametric/math500/exact_match_avg": 0.42,
        "gigametric/math500/pass_at_1": 0.55,
    }


def test_get_metrics_reshape_skips_empty_bench_tail(
    base_dir: Path,
    artifacts_path: Path,
    hf_checkpoint: Path,
    model_info: ModelInfoConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = GigaMetricEvaluatorConfig(
        validation_config={"validation_name": "x"},
        base_dir=base_dir,
        local_model_id="qwen_4b_local",
    )
    ev = _build_evaluator(cfg, artifacts_path, hf_checkpoint, model_info)

    def _fake_get_pam_metrics(*_args: Any, **_kwargs: Any) -> dict[str, float]:
        return {
            "gigametric/qwen_4b_local/": 0.1,  # empty tail
            "gigametric/qwen_4b_local/real/x": 0.5,
        }

    import sys as _sys
    import types as _types

    fake_module = _types.ModuleType("validation.pam_export")
    fake_module.get_pam_metrics = _fake_get_pam_metrics  # type: ignore[attr-defined]
    fake_pkg = _types.ModuleType("validation")
    monkeypatch.setitem(_sys.modules, "validation", fake_pkg)
    monkeypatch.setitem(_sys.modules, "validation.pam_export", fake_module)

    metrics = ev.get_metrics()
    assert metrics == {"gigametric/real/x": 0.5}


def test_get_metrics_returns_empty_when_no_state(
    base_dir: Path,
    artifacts_path: Path,
    hf_checkpoint: Path,
    model_info: ModelInfoConfig,
) -> None:
    cfg = GigaMetricEvaluatorConfig(
        validation_config={"validation_name": "x"}, base_dir=base_dir
    )
    ev = _build_evaluator(cfg, artifacts_path, hf_checkpoint, model_info)
    # state.json не записан → get_pam_metrics возвращает {}.
    assert ev.get_metrics() == {}


# ---------------------------------------------------------------------------
# after_run() — JD payload
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("judged", [False, True])
@pytest.mark.parametrize("uploaded", [False, True])
def test_after_run_publishes_archives_without_cache_scores(
    base_dir, artifacts_path, hf_checkpoint, model_info, monkeypatch, judged, uploaded
):
    config_kwargs = {
        "validation_config": {"validation_name": "x", "models": []},
        "base_dir": base_dir,
    }
    if judged:
        ev = GigaMetricJudgedEvaluator()
        cfg = GigaMetricJudgedEvaluatorConfig(**config_kwargs, judges={"judge": []})
    else:
        ev = GigaMetricEvaluator()
        cfg = GigaMetricEvaluatorConfig(**config_kwargs)
    ev.setup(cfg, artifacts_path, hf_checkpoint, model_info)
    archive = base_dir / "scoring" / "run.tar.gz"
    archive.parent.mkdir()
    archive.write_bytes(b"model answers archive")
    monkeypatch.setattr(
        JD, "read_checkpoint_type_and_id", lambda path: ("hf", "checkpoint-id")
    )
    uploads = []

    def upload(local_dir, prefix):
        uploads.append((local_dir, prefix))
        assert (
            local_dir / "scoring" / archive.name
        ).read_bytes() == archive.read_bytes()
        return uploaded

    monkeypatch.setattr(JD, "upload_artifacts", upload)
    send = MagicMock()
    monkeypatch.setattr(JD, "send_metrics", send)
    monkeypatch.setattr(JD, "tar_metrics_dir", MagicMock(return_value=None))
    monkeypatch.setattr(JD, "symlink_metrics_for_transfer", MagicMock())

    metrics = {"gigametric/bench/mean_accuracy": 0.7}
    ev.after_run(metrics)

    payload = send.call_args.args[0]
    assert payload["metrics"] == {100: metrics}
    assert uploads[0][0] == base_dir
    assert uploads[0][1].endswith("/gigametric/image-baked")
    assert payload.get("cache_dir_s3_path") == (
        JD.gigaeye_dir_url(uploads[0][1]) if uploaded else None
    )
    assert "cache_s3_path" not in payload
    assert archive.read_bytes() == b"model answers archive"


def test_send_metrics_to_jd_skips_without_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = GigaMetricEvaluator()
    evaluator.hf_checkpoint_path = None
    evaluator.model_info = None
    monkeypatch.setattr(
        JD,
        "send_metrics",
        lambda body, artifacts_path: pytest.fail("must not send without a checkpoint"),
    )

    evaluator._send_metrics_to_jd({"acc": 0.9})


def test_after_run_jd_payload_uses_per_run_version_file(
    base_dir: Path,
    artifacts_path: Path,
    hf_checkpoint: Path,
    model_info: ModelInfoConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = GigaMetricEvaluatorConfig(
        validation_config={"validation_name": "x"}, base_dir=base_dir
    )
    ev = _build_evaluator(cfg, artifacts_path, hf_checkpoint, model_info)

    # Кладём per-run version-файл — так делает validation/version.py.
    run_dir = base_dir / ev.validation_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "gigametric_version.txt").write_text(
        "version=1.0.0\ngit_commit=abc1234\ngit_dirty=false\n", encoding="utf-8"
    )

    captured: dict[str, Any] = {}

    def _fake_make_request_to_jd(body: dict, _artifacts) -> None:
        captured["body"] = body

    monkeypatch.setattr(JD, "send_metrics", _fake_make_request_to_jd)
    monkeypatch.setattr(JD, "tar_metrics_dir", MagicMock(return_value=None))
    monkeypatch.setattr(JD, "symlink_metrics_for_transfer", MagicMock())

    ev.after_run({"gigametric/m/b/score": 0.7})

    body = captured["body"]
    assert body["shooter_name"] == "gigametric"
    assert "1.0.0" in body["shooter_version"]
    assert body["shooter_version"] != "image-baked"  # fallback НЕ используется
    assert body["metrics"] == {100: {"gigametric/m/b/score": 0.7}}
    assert body["model_class"] == "sft"
    assert body["username"] == "tester"


def test_after_run_falls_back_to_install_time_version_file(
    base_dir: Path,
    artifacts_path: Path,
    hf_checkpoint: Path,
    model_info: ModelInfoConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = GigaMetricEvaluatorConfig(
        validation_config={"validation_name": "x"}, base_dir=base_dir
    )
    ev = _build_evaluator(cfg, artifacts_path, hf_checkpoint, model_info)
    # Per-run version-файл не пишем → фолбэк на config-поле shooter_version (default "image-baked").
    captured: dict[str, Any] = {}

    def _fake(body: dict, _artifacts) -> None:
        captured["body"] = body

    monkeypatch.setattr(JD, "send_metrics", _fake)
    monkeypatch.setattr(JD, "tar_metrics_dir", MagicMock(return_value=None))
    monkeypatch.setattr(JD, "symlink_metrics_for_transfer", MagicMock())

    ev.after_run({"gigametric/m/b/score": 0.7})
    assert captured["body"]["shooter_version"] == "image-baked"


def test_after_run_per_run_version_read_failure_falls_back(
    base_dir: Path,
    artifacts_path: Path,
    hf_checkpoint: Path,
    model_info: ModelInfoConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Если per-run version-файл существует, но чтение бросает исключение —
    адаптер логирует ошибку и фолбэчит на install-time version path
    (покрывает try/except-ветку в _gigametric_version)."""
    cfg = GigaMetricEvaluatorConfig(
        validation_config={"validation_name": "x"}, base_dir=base_dir
    )
    ev = _build_evaluator(cfg, artifacts_path, hf_checkpoint, model_info)

    run_dir = base_dir / ev.validation_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "gigametric_version.txt").write_text("ignored", encoding="utf-8")

    def _raise_on_read(self: Path, *_a: Any, **_kw: Any) -> str:
        raise OSError("simulated read failure")

    monkeypatch.setattr(Path, "read_text", _raise_on_read)

    captured: dict[str, Any] = {}

    def _fake(body: dict, _artifacts) -> None:
        captured["body"] = body

    monkeypatch.setattr(JD, "send_metrics", _fake)
    monkeypatch.setattr(JD, "tar_metrics_dir", MagicMock(return_value=None))
    monkeypatch.setattr(JD, "symlink_metrics_for_transfer", MagicMock())

    ev.after_run({"gigametric/m/b/score": 0.7})
    # Сфолбэчились на config-поле shooter_version (default "image-baked").
    assert captured["body"]["shooter_version"] == "image-baked"


# ---------------------------------------------------------------------------
# Sanity
# ---------------------------------------------------------------------------


def test_evaluator_inherits_metrics_evaluator() -> None:
    from node_runner.src.metrics.evaluators.interface import MetricsEvaluator

    assert issubclass(GigaMetricEvaluator, MetricsEvaluator)


def test_registered_in_auto_metrics_manager() -> None:
    from node_runner.src.metrics.metrics_manager import EVALUATORS

    assert EVALUATORS["gigametric"] is GigaMetricEvaluator


# ---------------------------------------------------------------------------
# get_metrics() — sys.path branches
# ---------------------------------------------------------------------------


def test_get_metrics_does_not_double_insert_papayametric_root(
    base_dir: Path,
    artifacts_path: Path,
    hf_checkpoint: Path,
    model_info: ModelInfoConfig,
) -> None:
    cfg = GigaMetricEvaluatorConfig(
        validation_config={"validation_name": "x"}, base_dir=base_dir
    )
    ev = _build_evaluator(cfg, artifacts_path, hf_checkpoint, model_info)
    papayametric_root = str(GigaMetricEvaluator._papayametric_root())
    import sys

    sys.path.insert(0, papayametric_root)
    try:
        ev.get_metrics()
        assert sys.path.count(papayametric_root) == 1
    finally:
        if papayametric_root in sys.path:
            sys.path.remove(papayametric_root)


def test_get_metrics_finally_swallows_remove_failure(
    base_dir: Path,
    artifacts_path: Path,
    hf_checkpoint: Path,
    model_info: ModelInfoConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = GigaMetricEvaluatorConfig(
        validation_config={"validation_name": "x"}, base_dir=base_dir
    )
    ev = _build_evaluator(cfg, artifacts_path, hf_checkpoint, model_info)
    import sys

    papayametric_root = str(GigaMetricEvaluator._papayametric_root())
    if papayametric_root in sys.path:
        sys.path.remove(papayametric_root)

    # Simulate concurrent mutation: pam_export removes the entry from
    # sys.path before finally has a chance to do it.
    def _fake_get_pam_metrics(validation_id: str, base: Path) -> dict[str, float]:
        if papayametric_root in sys.path:
            sys.path.remove(papayametric_root)
        return {}

    import types

    fake_module = types.ModuleType("validation.pam_export")
    fake_module.get_pam_metrics = _fake_get_pam_metrics  # type: ignore[attr-defined]
    validation_pkg = types.ModuleType("validation")
    validation_pkg.pam_export = fake_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "validation", validation_pkg)
    monkeypatch.setitem(sys.modules, "validation.pam_export", fake_module)

    ev.get_metrics()  # must not raise even though sys.path.remove fails


# ---------------------------------------------------------------------------
# _validate_runtime()
# ---------------------------------------------------------------------------


def test_validate_runtime_passes_when_all_present(
    base_dir: Path,
    artifacts_path: Path,
    hf_checkpoint: Path,
    model_info: ModelInfoConfig,
    tmp_path: Path,
) -> None:
    # runtime_python = текущий интерпретатор, run_validation.py подсовываем
    # из tmp (gigametric больше не submodule), config материализован
    # в setup() → валидация не бросает.
    cfg = GigaMetricEvaluatorConfig(
        validation_config={"validation_name": "x"},
        base_dir=base_dir,
        runtime_python=sys.executable,
    )
    ev = _build_evaluator(cfg, artifacts_path, hf_checkpoint, model_info)
    fake_root = tmp_path / "fake_giga"
    fake_root.mkdir(parents=True)
    (fake_root / "run_validation.py").write_text("# stub\n", encoding="utf-8")
    with patch.object(ev, "_papayametric_root", return_value=fake_root):
        ev._validate_runtime()


def test_validate_runtime_raises_on_missing_python(
    base_dir: Path,
    artifacts_path: Path,
    hf_checkpoint: Path,
    model_info: ModelInfoConfig,
) -> None:
    cfg = GigaMetricEvaluatorConfig(
        validation_config={"validation_name": "x"},
        base_dir=base_dir,
        runtime_python="/nonexistent/python",
    )
    ev = _build_evaluator(cfg, artifacts_path, hf_checkpoint, model_info)
    with pytest.raises(FileNotFoundError, match="runtime_python"):
        ev._validate_runtime()


def test_validate_runtime_raises_on_missing_run_validation(
    base_dir: Path,
    artifacts_path: Path,
    hf_checkpoint: Path,
    model_info: ModelInfoConfig,
    tmp_path: Path,
) -> None:
    cfg = GigaMetricEvaluatorConfig(
        validation_config={"validation_name": "x"},
        base_dir=base_dir,
        runtime_python=sys.executable,
    )
    ev = _build_evaluator(cfg, artifacts_path, hf_checkpoint, model_info)
    empty_root = tmp_path / "empty_giga"
    empty_root.mkdir()
    with patch.object(ev, "_papayametric_root", return_value=empty_root):
        with pytest.raises(FileNotFoundError, match="run_validation.py"):
            ev._validate_runtime()


def test_validate_runtime_raises_on_missing_config(
    base_dir: Path,
    artifacts_path: Path,
    hf_checkpoint: Path,
    model_info: ModelInfoConfig,
    tmp_path: Path,
) -> None:
    cfg = GigaMetricEvaluatorConfig(
        validation_config={"validation_name": "x"},
        base_dir=base_dir,
        runtime_python=sys.executable,
    )
    ev = _build_evaluator(cfg, artifacts_path, hf_checkpoint, model_info)
    ev.config_path = tmp_path / "no_such_config.yaml"
    fake_root = tmp_path / "fake_giga"
    fake_root.mkdir(parents=True)
    (fake_root / "run_validation.py").write_text("# stub\n", encoding="utf-8")
    with (
        patch.object(ev, "_papayametric_root", return_value=fake_root),
        pytest.raises(FileNotFoundError, match="config не найден"),
    ):
        ev._validate_runtime()


# ---------------------------------------------------------------------------
# _papayametric_root()
# ---------------------------------------------------------------------------


def test_papayametric_root_prefers_runtime_path_when_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Когда /home/user/papayametric/run_validation.py существует
    # (runtime-unpacked layout) — возвращаем runtime-путь, не сабмодуль.
    monkeypatch.setattr(Path, "exists", lambda self: True)
    assert GigaMetricEvaluator._papayametric_root() == Path("/home/user/papayametric")


def test_papayametric_root_falls_back_to_source_when_runtime_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Path, "exists", lambda self: False)

    assert GigaMetricEvaluator._papayametric_root() == (
        Path(__file__).resolve().parents[4] / "contrib" / "papayametric"
    )


# ---------------------------------------------------------------------------
# _unzip_papayametric()
# ---------------------------------------------------------------------------


def test_unzip_papayametric_unpacks_archive_and_cleans_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "gigametric.tar.zst"
    archive.write_bytes(b"archive")
    runtime_root = tmp_path / "runtime" / "gigametric"
    runtime_root.mkdir(parents=True)
    unzip_log = tmp_path / "unzip_papayametric.log"
    unzip_log.write_text("old log", encoding="utf-8")
    calls: list[list[str]] = []

    class _FakeSubprocessManager:
        def __init__(
            self,
            cmd: list[str],
            log_file: Path,
            cwd: Path | None,
            run_name: str,
        ) -> None:
            calls.append(cmd)
            assert log_file == unzip_log
            assert cwd is None
            assert run_name == "Unzip GigaMetric"

        def start(self) -> None:
            pass

        def wait_for_completion(self) -> None:
            unzip_log.write_text("new log", encoding="utf-8")

        def raise_for_exit_code(self) -> None:
            pass

    monkeypatch.setattr(GigaMetricEvaluator, "ARCHIVE_PATH", archive)
    monkeypatch.setattr(GigaMetricEvaluator, "RUNTIME_ROOT", runtime_root)
    monkeypatch.setattr(GigaMetricEvaluator, "UNZIP_LOG_PATH", unzip_log)
    monkeypatch.setattr(
        "node_runner.src.metrics.evaluators.gigametric.SubprocessManager",
        _FakeSubprocessManager,
    )

    assert GigaMetricEvaluator._unzip_papayametric() == runtime_root
    assert calls == [
        [
            "tar",
            "--zstd",
            "-xf",
            archive.as_posix(),
            "-C",
            runtime_root.parent.absolute().as_posix(),
        ]
    ]
    assert not unzip_log.exists()


def test_unzip_papayametric_handles_missing_runtime_and_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "gigametric.tar.zst"
    archive.write_bytes(b"archive")
    runtime_root = tmp_path / "runtime" / "gigametric"
    unzip_log = tmp_path / "unzip_papayametric.log"

    class _FakeSubprocessManager:
        def __init__(
            self,
            cmd: list[str],
            log_file: Path,
            cwd: Path | None,
            run_name: str,
        ) -> None:
            assert cmd[-2:] == ["-C", runtime_root.parent.absolute().as_posix()]
            assert log_file == unzip_log
            assert cwd is None
            assert run_name == "Unzip GigaMetric"

        def start(self) -> None:
            pass

        def wait_for_completion(self) -> None:
            pass

        def raise_for_exit_code(self) -> None:
            pass

    monkeypatch.setattr(GigaMetricEvaluator, "ARCHIVE_PATH", archive)
    monkeypatch.setattr(GigaMetricEvaluator, "RUNTIME_ROOT", runtime_root)
    monkeypatch.setattr(GigaMetricEvaluator, "UNZIP_LOG_PATH", unzip_log)
    monkeypatch.setattr(
        "node_runner.src.metrics.evaluators.gigametric.SubprocessManager",
        _FakeSubprocessManager,
    )

    assert GigaMetricEvaluator._unzip_papayametric() == runtime_root
    assert not unzip_log.exists()
