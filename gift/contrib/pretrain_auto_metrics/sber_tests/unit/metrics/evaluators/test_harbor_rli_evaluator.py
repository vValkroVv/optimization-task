import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from node_runner.src.config_model import HarborRliEvaluatorConfig, ModelInfoConfig
from node_runner.src.job_dispatcher import sync
from node_runner.src.metrics.evaluators import harbor_rli as harbor_module
from node_runner.src.metrics.evaluators.harbor_rli import HarborRliEvaluator

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "benchmark, default",
    [
        ("terminal-bench-2.0", True),
        ("terminal-bench-2.1", True),
        ("swe-bench-verified", False),
        ("swe-bench-multilingual", False),
        ("harness-bench-lite", False),
        ("harness-bench-medium", False),
    ],
)
@pytest.mark.parametrize(
    "network_config, explicit",
    [
        ({}, None),
        ({"allow-internet-access": True}, True),
        ({"allow-internet-access": False}, False),
        ({"allow_internet_access": True}, True),
        ({"allow_internet_access": False}, False),
    ],
)
def test_network_defaults_and_explicit_overrides(
    benchmark, default, network_config, explicit
):
    config = HarborRliEvaluatorConfig.model_validate(
        {"benchmark": benchmark, **network_config}
    )

    assert config.to_dict()["allow-internet-access"] is (
        default if explicit is None else explicit
    )


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    version_file = tmp_path / "harbor_rli_version.txt"
    version_file.write_text("revision-a\n")
    monkeypatch.setattr(HarborRliEvaluator, "VERSION_FILE", str(version_file))
    return version_file


@pytest.fixture
def evaluator_factory(tmp_path, runtime):
    def build(payload=None):
        payload = dict(payload or {})
        if "runs" not in payload:
            payload.setdefault("benchmark", "terminal-bench-2.0")
        config = HarborRliEvaluatorConfig.model_validate(payload)
        config.apply_defaults(inference_instances=1, base_url="http://localhost:30000")
        evaluator = HarborRliEvaluator()
        evaluator.setup(
            config,
            artifacts_path=tmp_path / "artifacts",
            hf_checkpoint_path=tmp_path / "global_step_7",
            model_info=ModelInfoConfig(model_class="sft", size="lite", username="u"),
        )
        return evaluator

    return build


def write_metrics(evaluator, *, complete=True, metrics=None):
    evaluator.metrics_path.write_text(
        json.dumps(
            {
                "complete": complete,
                "metrics": metrics
                if metrics is not None
                else {evaluator.metrics_dir.name: 0.5},
            }
        )
    )


@pytest.mark.parametrize(
    "benchmark, concurrency",
    [
        ("terminal-bench-2.0", 8),
        ("terminal-bench-2.1", 8),
        ("harness-bench-lite", 8),
        ("harness-bench-medium", 8),
        ("swe-bench-verified", 8),
        ("swe-bench-multilingual", 8),
    ],
)
def test_cli_contract_and_output_directory(evaluator_factory, benchmark, concurrency):
    evaluator = evaluator_factory(
        {
            "benchmark": benchmark,
            "agent": "mini-swe-agent",
            "venv_python": "python",
            "benchmark-kwargs": {"env-version": "revision-a"},
        }
    )
    cmd = evaluator._build_cmd()
    assert cmd[:6] == [
        "python",
        "-m",
        "harbor_rli.cli",
        "run",
        "--benchmark",
        benchmark,
    ]
    assert cmd[6:9] == ["--metrics-dir", str(evaluator.metrics_dir), "--config-json"]
    payload = json.loads(cmd[-1])
    assert payload["env-version"] == "revision-a"
    assert "task-index" not in payload
    assert "env-name" not in payload
    assert "dataset" not in payload
    assert payload["model-api-base"] == "http://localhost:30000/v1"
    assert payload["n-concurrent"] == concurrency
    assert payload["agent-kwargs"] == {}
    assert "benchmark" not in payload
    assert "benchmark-kwargs" not in payload
    assert "venv_python" not in payload
    assert (
        evaluator.metrics_dir
        == evaluator.artifacts_path / "metrics" / "harbor_rli" / benchmark
    )


@pytest.mark.parametrize("dataset", ["/custom/tasks", "custom-dataset@revision"])
def test_explicit_dataset_overrides_image_default(evaluator_factory, dataset):
    evaluator = evaluator_factory({"benchmark-kwargs": {"dataset": dataset}})
    payload = json.loads(evaluator._build_cmd()[-1])
    assert payload["dataset"] == dataset


def test_cli_omits_unset_endpoint(evaluator_factory):
    evaluator = evaluator_factory()
    evaluator.autometric_runs_config.model_api_base = None
    payload = json.loads(evaluator._build_cmd()[-1])
    assert "model-api-base" not in payload


@pytest.mark.parametrize("pythonpath", [None, "/existing/path"])
def test_runtime_environment(evaluator_factory, monkeypatch, pythonpath):
    evaluator = evaluator_factory()
    monkeypatch.delenv("PYTHONPATH", raising=False)
    if pythonpath is not None:
        monkeypatch.setenv("PYTHONPATH", pythonpath)
    monkeypatch.setattr(evaluator, "get_rli_url", lambda: "https://rli.test")
    env = evaluator._build_env()
    expected_pythonpath = "/home/user/harbor_rli_env"
    if pythonpath:
        expected_pythonpath += f":{pythonpath}"
    assert env["PYTHONPATH"] == expected_pythonpath
    assert evaluator._build_cmd()[0] == "/home/user/harbor_rli_env/bin/python"
    assert env["RLI_API_BASE"] == "https://rli.test"


def test_recursive_defaults_preserve_explicit_values():
    config = HarborRliEvaluatorConfig.model_validate(
        {
            "runs": [
                {"benchmark": "terminal-bench-2.0"},
                {
                    "benchmark": "terminal-bench-2.1",
                    "n-concurrent": 3,
                    "model-api-base": "https://external/v1",
                    "request-retries": 3,
                    "session-ready-poll-interval": 2,
                    "create-session-jitter-max-sec": 30,
                },
            ]
        }
    )
    config.apply_defaults(inference_instances=4, base_url="http://localhost:30000")
    assert [run.n_concurrent for run in config.runs] == [32, 3]
    assert [run.model_api_base for run in config.runs] == [
        "http://localhost:30000/v1",
        "https://external/v1",
    ]
    payloads = config.to_dict()["runs"]
    assert [payload["request-retries"] for payload in payloads] == [1, 3]
    assert [payload["session-ready-poll-interval"] for payload in payloads] == [1, 2]
    assert [payload["create-session-jitter-max-sec"] for payload in payloads] == [0, 30]


@pytest.mark.parametrize(
    "payload, message",
    [
        ({}, "benchmark is required"),
        ({"benchmark": "../escape"}, "String should match pattern"),
        ({"runs": []}, "at least 1 item"),
        (
            {"runs": [{"benchmark": "terminal-bench-2.0"}], "agent": "mini-swe-agent"},
            "runs cannot be combined",
        ),
        (
            {
                "runs": [
                    {"benchmark": "terminal-bench-2.0"},
                    {"benchmark": "terminal-bench-2.0"},
                ]
            },
            "different benchmark",
        ),
        ({"runs": [{"runs": [{"benchmark": "terminal-bench-2.0"}]}]}, "nested runs"),
    ],
)
def test_invalid_configs(payload, message):
    with pytest.raises(ValidationError, match=message):
        HarborRliEvaluatorConfig.model_validate(payload)


@pytest.mark.parametrize(
    "key",
    [
        "agent",
        "n-concurrent",
        "n_concurrent",
        "runs",
    ],
)
def test_adapter_kwargs_cannot_override_common_fields(key):
    with pytest.raises(ValidationError, match="duplicates common fields"):
        HarborRliEvaluatorConfig.model_validate(
            {"benchmark": "terminal-bench-2.0", "benchmark-kwargs": {key: "value"}}
        )


def test_future_benchmark_and_agent_are_adapter_contracts():
    config = HarborRliEvaluatorConfig.model_validate(
        {
            "benchmark": "custom-bench",
            "agent": "custom-agent",
            "env-version": "v1",
            "benchmark_kwargs": {"custom-mode": True},
        }
    )
    assert config.to_dict()["benchmark-kwargs"] == {"custom-mode": True}
    assert config.to_dict()["env-version"] == "v1"


@pytest.mark.parametrize(
    "benchmark, component",
    [
        ("cowork-bench", "cowork"),
        ("harness-bench-lite", "harness"),
        ("harness-bench-medium", "harness"),
        ("swe-bench-verified", "swe"),
        ("swe-bench-multilingual", "swe"),
        ("terminal-bench-2.0", "terminal20"),
        ("terminal-bench-2.1", "terminal21"),
    ],
)
def test_benchmark_versions_isolate_cache(
    evaluator_factory, runtime, benchmark, component
):
    versions = {
        "rli": "harbor-a+sdk-a",
        "benchmarks": {benchmark: f"{component}.data-a", "other-bench": "other.a"},
    }
    runtime.write_text(json.dumps(versions))
    evaluator = evaluator_factory({"benchmark": benchmark})
    assert evaluator.shooter_version == f"harbor-a+sdk-a+{component}.data-a"
    write_metrics(evaluator)
    evaluator._save_fingerprint()

    versions["benchmarks"]["other-bench"] = "other.b"
    runtime.write_text(json.dumps(versions))
    assert evaluator_factory({"benchmark": benchmark})._is_already_calculated()

    versions["benchmarks"][benchmark] = f"{component}.data-b"
    runtime.write_text(json.dumps(versions))
    assert not evaluator_factory({"benchmark": benchmark})._is_already_calculated()


@pytest.mark.parametrize(
    "contents",
    [
        "{",
        "{}",
        '{"rli": "h+g", "benchmarks": {}}',
        '{"rli": "h+g", "benchmarks": null}',
    ],
)
def test_invalid_version_skips_run_and_cached_metrics(
    evaluator_factory, runtime, process_calls, monkeypatch, contents
):
    cached = evaluator_factory()
    write_metrics(cached)
    cached._save_fingerprint()
    fingerprint = cached.fingerprint_path.read_text()
    errors = Mock()
    monkeypatch.setattr(harbor_module.Logger, "error", errors)
    runtime.write_text(contents)

    evaluator = evaluator_factory()
    send = Mock()
    archive = Mock()
    monkeypatch.setattr(evaluator, "_send_metrics_to_jd", send)
    monkeypatch.setattr(harbor_module.JD, "tar_metrics_dir", archive)
    evaluator.run()
    evaluator._save_fingerprint()
    evaluator.after_run({"score": 0.5})

    assert process_calls == []
    assert evaluator.get_metrics() == {}
    assert not evaluator._is_already_calculated()
    assert evaluator.fingerprint_path.read_text() == fingerprint
    send.assert_not_called()
    archive.assert_not_called()
    errors.assert_called_once()
    assert str(runtime) in errors.call_args.args[0]
    assert "terminal-bench-2.0" in errors.call_args.args[0]


@pytest.mark.parametrize("has_valid_run", [False, True])
def test_invalid_version_does_not_block_sibling_runs(
    evaluator_factory, runtime, monkeypatch, has_valid_run
):
    runtime.write_text(
        json.dumps(
            {
                "rli": "h+g",
                "benchmarks": {"terminal-bench-2.0": "terminal20.a"}
                if has_valid_run
                else {},
            }
        )
    )
    evaluator = evaluator_factory(
        {"runs": [{"benchmark": "terminal-bench-2.0"}, {"benchmark": "cowork-bench"}]}
    )
    for run in evaluator._runs:
        write_metrics(run, metrics={run.metrics_dir.name: 0.5})
    pool = Mock()
    monkeypatch.setattr(harbor_module, "SubprocessPoolRunner", pool)
    monkeypatch.setattr(evaluator, "_build_env", dict)
    pool.return_value.run_all.side_effect = lambda: write_metrics(evaluator._runs[0])
    evaluator.run()

    if has_valid_run:
        pool.assert_called_once()
        assert set(pool.call_args.kwargs["cmds_to_run"]) == {"terminal-bench-2.0"}
        assert evaluator.get_metrics() == {"terminal-bench-2.0": 0.5}
    else:
        pool.assert_not_called()
        assert evaluator.get_metrics() == {}


def test_image_version_changes_invalidate_cached_results(evaluator_factory, runtime):
    evaluator = evaluator_factory()
    write_metrics(evaluator)
    evaluator._save_fingerprint()
    assert evaluator._is_already_calculated()
    assert evaluator.shooter_version == "revision-a"
    assert evaluator_factory().shooter_version == evaluator.shooter_version
    runtime.write_text("revision-b\n")
    updated = evaluator_factory()
    assert updated.shooter_version == "revision-b"
    assert not updated._is_already_calculated()


@pytest.mark.parametrize(
    "change",
    [
        {"max-tokens": 123},
        {"benchmark-kwargs": {"env-version": "v2"}},
        {"pass-at-k": 2},
        {"force-recalc": True},
    ],
)
def test_config_changes_invalidate_cache(evaluator_factory, change):
    evaluator = evaluator_factory()
    write_metrics(evaluator)
    evaluator._save_fingerprint()
    assert not evaluator_factory(change)._is_already_calculated()


def test_job_name_and_endpoint_do_not_invalidate_cache(evaluator_factory):
    evaluator = evaluator_factory()
    write_metrics(evaluator)
    evaluator._save_fingerprint()
    updated = evaluator_factory(
        {"job-name": "renamed", "model-api-base": "http://other/v1"}
    )
    assert updated._is_already_calculated()


@pytest.mark.parametrize(
    "state",
    ["missing", "no-fingerprint", "incomplete", "bad-metrics", "bad-fingerprint"],
)
def test_invalid_cache_reruns(evaluator_factory, state):
    evaluator = evaluator_factory()
    if state != "missing":
        write_metrics(evaluator, complete=state != "incomplete")
    if state not in ("missing", "no-fingerprint"):
        evaluator.fingerprint_path.write_text(json.dumps(evaluator._fingerprint()))
    if state == "bad-metrics":
        evaluator.metrics_path.write_text("{")
    if state == "bad-fingerprint":
        evaluator.fingerprint_path.write_text("{")
    assert not evaluator._is_already_calculated()


@pytest.mark.parametrize(
    "contents",
    [
        None,
        "{",
        "[1]",
        "{}",
        '{"complete": true, "metrics": []}',
        '{"complete": false, "metrics": {"score": 1}}',
    ],
)
def test_missing_invalid_or_incomplete_results_are_not_published(
    evaluator_factory, contents
):
    evaluator = evaluator_factory()
    if contents is not None:
        evaluator.metrics_path.write_text(contents)
    assert evaluator.get_metrics() == {}


def test_missing_metrics_directory(evaluator_factory):
    evaluator = evaluator_factory()
    evaluator.metrics_dir = None
    assert evaluator.get_metrics() == {}


def test_only_finite_numeric_metrics_are_published(evaluator_factory):
    evaluator = evaluator_factory()
    write_metrics(
        evaluator,
        metrics={
            "score": 0.5,
            "count": 1,
            "flag": True,
            "nan": float("nan"),
            "inf": float("inf"),
            "label": "text",
        },
    )
    assert evaluator.get_metrics() == {"score": 0.5, "count": 1.0}


@pytest.mark.parametrize(
    "selector",
    [
        {"n-tasks": 1},
        {"task-name": "a"},
        {"exclude-task-name": "a"},
        {"benchmark-kwargs": {"task-id": "a"}},
        {"benchmark-kwargs": {"task_id": "a"}},
        {"benchmark-kwargs": {"tasks": ["task_01_create_hello"]}},
        {"benchmark-kwargs": {"tasks": []}},
    ],
)
@pytest.mark.parametrize(
    "benchmark", ["terminal-bench-2.0", "harness-bench-lite", "harness-bench-medium"]
)
@pytest.mark.parametrize("complete", [False, True])
def test_selector_does_not_override_adapter_completeness(
    evaluator_factory, selector, complete, benchmark
):
    evaluator = evaluator_factory({"benchmark": benchmark, **selector})
    write_metrics(evaluator, complete=complete, metrics={"score": 0.5})
    expected = {"score": 0.5} if complete else {}
    assert evaluator.get_metrics() == expected


def test_duplicate_metrics_are_logged_and_keep_last_value(
    evaluator_factory, monkeypatch
):
    evaluator = evaluator_factory(
        {
            "runs": [
                {"benchmark": "terminal-bench-2.0"},
                {"benchmark": "terminal-bench-2.1"},
            ]
        }
    )
    errors = []
    monkeypatch.setattr(harbor_module.Logger, "error", errors.append)
    write_metrics(evaluator._runs[0], metrics={"same": 0.25, "tb20": 1})
    write_metrics(evaluator._runs[1], metrics={"same": 0.75, "tb21": 1})
    assert evaluator.get_metrics() == {"same": 0.75, "tb20": 1, "tb21": 1}
    assert len(errors) == 1
    assert "harbor_rli" in errors[0]
    assert "same" in errors[0]
    assert "terminal-bench-2.1" in errors[0]


@pytest.fixture
def process_calls(monkeypatch):
    calls = []

    class FakeManager:
        def __init__(self, cmd, log, **kwargs):
            calls.append((cmd, log, kwargs))
            self.path = Path(cmd[cmd.index("--metrics-dir") + 1])

        def start(self):
            assert not (self.path / "metrics.json").exists()
            assert not (self.path / "harbor_rli_setup.json").exists()
            (self.path / "metrics.json").write_text(
                json.dumps({"complete": True, "metrics": {"score": 0.5}})
            )

        def wait_for_completion(self):
            pass

        def log_error(self):
            pass

    monkeypatch.setattr(harbor_module, "SubprocessManager", FakeManager)
    return calls


def test_single_run_saves_cache_and_skips_completed_run(
    evaluator_factory, process_calls
):
    evaluator = evaluator_factory({"n-concurrent": 3})
    assert evaluator.offered_load == 3
    evaluator.run()
    evaluator.run()
    assert len(process_calls) == 1
    assert process_calls[0][1] == evaluator.log_file
    assert process_calls[0][2]["run_name"] == "harbor_rli"
    assert evaluator.get_metrics() == {"score": 0.5}
    assert evaluator._is_already_calculated()


def test_force_recalc_clears_old_results(evaluator_factory, process_calls):
    evaluator = evaluator_factory({"force-recalc": True})
    write_metrics(evaluator)
    evaluator._save_fingerprint()
    evaluator.run()
    assert len(process_calls) == 1


def test_pool_runs_only_pending_benchmarks_and_keeps_successful_siblings(
    evaluator_factory, monkeypatch
):
    evaluator = evaluator_factory(
        {
            "runs": [
                {"benchmark": "terminal-bench-2.0", "n-concurrent": 3},
                {"benchmark": "terminal-bench-2.1", "n-concurrent": 2},
                {"benchmark": "custom-bench", "n-concurrent": 1},
            ]
        }
    )
    cached, pending, incomplete = evaluator._runs
    write_metrics(cached)
    cached._save_fingerprint()
    write_metrics(pending, complete=False)
    calls = []

    class FakePool:
        def __init__(self, cmds_to_run, max_parallel, **kwargs):
            calls.append((cmds_to_run, max_parallel, kwargs))

        def run_all(self):
            assert not pending.metrics_path.exists()
            write_metrics(pending)
            write_metrics(incomplete, complete=False)

    monkeypatch.setattr(harbor_module, "SubprocessPoolRunner", FakePool)
    assert evaluator.offered_load == 6
    evaluator.run()
    assert set(calls[0][0]) == {"terminal-bench-2.1", "custom-bench"}
    assert calls[0][1] == 2
    assert calls[0][2]["autometrics_log_file"] == evaluator.log_file
    assert evaluator.get_metrics() == {
        "terminal-bench-2.0": 0.5,
        "terminal-bench-2.1": 0.5,
    }
    assert cached._is_already_calculated()
    assert pending._is_already_calculated()
    assert not incomplete._is_already_calculated()
    write_metrics(incomplete)
    incomplete._save_fingerprint()
    evaluator.run()
    assert len(calls) == 1


@pytest.mark.parametrize("multiple", [False, True])
@pytest.mark.parametrize(
    "sync_enabled, online", [(False, False), (True, False), (True, True)]
)
def test_unfinished_runs_defer_archiving_to_metrics_pipeline(
    evaluator_factory, monkeypatch, tmp_path, multiple, sync_enabled, online
):
    evaluator = evaluator_factory(
        {
            "runs": [
                {"benchmark": "terminal-bench-2.0"},
                {"benchmark": "terminal-bench-2.1"},
            ]
        }
        if multiple
        else {}
    )
    runs = evaluator._runs if multiple else [evaluator]

    class FakeManager:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            for run in runs:
                write_metrics(run, complete=False)

        def wait_for_completion(self):
            pass

        def log_error(self):
            pass

    class FakePool(FakeManager):
        def run_all(self):
            self.start()

    monkeypatch.setattr(harbor_module, "SubprocessManager", FakeManager)
    monkeypatch.setattr(harbor_module, "SubprocessPoolRunner", FakePool)
    monkeypatch.setattr(
        harbor_module.JD, "get_jd_sync_dir", lambda: tmp_path if sync_enabled else None
    )
    monkeypatch.setattr(harbor_module.JD, "internet_is_available", lambda: online)
    archive = Mock(
        side_effect=lambda metrics_dir, *args: metrics_dir / "archive.tar.zst"
    )
    transfer = Mock()
    monkeypatch.setattr(harbor_module.JD, "tar_metrics_dir", archive)
    monkeypatch.setattr(harbor_module.JD, "symlink_metrics_for_transfer", transfer)
    evaluator.run()
    archive.assert_not_called()
    transfer.assert_not_called()
    assert evaluator.get_metrics() == {}
    assert all(run.metrics_path.exists() for run in runs)


@pytest.mark.parametrize(
    "state, selector",
    [
        ("missing", {}),
        ("invalid", {}),
        ("incomplete", {}),
        ("incomplete", {"task-name": "a"}),
        (
            "incomplete",
            {
                "benchmark": "harness-bench-lite",
                "benchmark-kwargs": {"tasks": ["task_01_create_hello"]},
            },
        ),
        ("complete", {}),
        ("complete", {"task-name": "*"}),
    ],
)
def test_after_run_publishes_only_adapter_complete_results(
    evaluator_factory, monkeypatch, state, selector
):
    evaluator = evaluator_factory(selector)
    if state == "invalid":
        evaluator.metrics_path.write_text("{")
    elif state != "missing":
        write_metrics(evaluator, complete=state == "complete")
    publish, archive, transfer = Mock(), Mock(), Mock()
    monkeypatch.setattr(evaluator, "_send_metrics_to_jd", publish)
    monkeypatch.setattr(harbor_module.JD, "tar_metrics_dir", archive)
    monkeypatch.setattr(harbor_module.JD, "symlink_metrics_for_transfer", transfer)
    evaluator.after_run(evaluator.get_metrics(), None)
    assert publish.call_count == int(state == "complete")
    assert archive.call_count == transfer.call_count == 1


@pytest.mark.parametrize("state", ["missing", "invalid", "incomplete"])
@pytest.mark.parametrize("online", [False, True])
def test_partial_runs_archive_failed_results_only_offline(
    evaluator_factory, monkeypatch, tmp_path, state, online
):
    evaluator = evaluator_factory(
        {
            "runs": [
                {"benchmark": "terminal-bench-2.0"},
                {"benchmark": "terminal-bench-2.1"},
            ]
        }
    )
    complete, failed = evaluator._runs
    write_metrics(complete)
    if state == "invalid":
        failed.metrics_path.write_text("{")
    elif state == "incomplete":
        write_metrics(failed, complete=False)
    successful_publish, failed_publish = Mock(), Mock()
    monkeypatch.setattr(complete, "_send_metrics_to_jd", successful_publish)
    monkeypatch.setattr(failed, "_send_metrics_to_jd", failed_publish)
    monkeypatch.setattr(harbor_module.JD, "internet_is_available", lambda: online)
    monkeypatch.setattr(harbor_module.JD, "get_jd_sync_dir", lambda: tmp_path / "sync")
    compressor, transfer = Mock(), Mock()
    monkeypatch.setattr(sync, "SubprocessManager", compressor)
    monkeypatch.setattr(harbor_module.JD, "symlink_metrics_for_transfer", transfer)

    metrics = evaluator.get_metrics()
    evaluator.after_run(metrics)

    assert metrics == {"terminal-bench-2.0": 0.5}
    successful_publish.assert_called_once_with(metrics)
    failed_publish.assert_not_called()
    assert compressor.call_count == (0 if online else 2)
    expected_archives = (
        [None, None]
        if online
        else [
            evaluator.artifacts_path / "terminal-bench-2.0_artifacts.tar.zst",
            evaluator.artifacts_path / "terminal-bench-2.1_artifacts.tar.zst",
        ]
    )
    assert [call.args[0] for call in transfer.call_args_list] == expected_archives


@pytest.mark.parametrize(
    "benchmark, shooter",
    [
        ("harness-bench-lite", "harness_bench"),
        ("harness-bench-medium", "harness_bench"),
        ("swe-bench-verified", "swe_bench"),
        ("swe-bench-multilingual", "swe_bench"),
        ("terminal-bench-2.0", "terminal_bench"),
        ("terminal-bench-2.1", "terminal_bench"),
        ("custom-bench", "custom_bench"),
    ],
)
def test_jd_identity_and_task_scores(
    evaluator_factory, harbor_job, monkeypatch, benchmark, shooter
):
    evaluator = evaluator_factory({"benchmark": benchmark, "pass-at-k": 2})
    path = harbor_job({"a": [1, 0], "b": [0, 0]})
    evaluator.metrics_path.write_text(
        json.dumps({"job_result_path": str(path), "total": 4})
    )
    publish = Mock()
    monkeypatch.setattr(harbor_module.JD, "can_send_metrics", lambda **kwargs: True)
    monkeypatch.setattr(harbor_module.JD, "send_evaluator_metrics_with_scores", publish)
    evaluator._send_metrics_to_jd(
        {"quality/mean": 0.25, "quality/pass_at_2": 0.5, "quality/pass@2": 0.5}
    )
    payload = publish.call_args.kwargs
    assert payload["shooter_name"] == shooter
    assert payload["shooter_version"] == f"{evaluator.shooter_version}-{benchmark}"
    assert payload["scores"].metrics == {
        "QUALITY_MEAN": 0.25,
        "QUALITY_PASS_AT_2": 0.5,
        "QUALITY_PASS@2": 0.5,
    }
    assert {
        key: [row["score"] for row in rows]
        for key, rows in payload["scores"].records.items()
    } == {
        "QUALITY_MEAN": [0.5, 0.0],
        "QUALITY_PASS_AT_2": [1.0, 0.0],
        "QUALITY_PASS@2": [1.0, 0.0],
    }


@pytest.mark.parametrize("can_send", [True, False])
def test_jd_without_task_scores(evaluator_factory, monkeypatch, can_send):
    evaluator = evaluator_factory()
    publish = Mock()
    monkeypatch.setattr(harbor_module.JD, "can_send_metrics", lambda **kwargs: can_send)
    monkeypatch.setattr(harbor_module.JD, "send_evaluator_metrics_with_scores", publish)
    evaluator._send_metrics_to_jd({"score": 0.5})
    assert publish.call_count == int(can_send)
    if can_send:
        assert publish.call_args.kwargs["scores"].records == {}


@pytest.mark.parametrize("benchmark", ["harness-bench-lite", "harness-bench-medium"])
def test_explicit_task_index_is_forwarded_to_cli(evaluator_factory, benchmark):
    evaluator = evaluator_factory(
        {
            "benchmark": benchmark,
            "benchmark-kwargs": {"task-index": "/custom/task-index.json"},
        }
    )
    payload = json.loads(evaluator._build_cmd()[-1])
    assert payload["task-index"] == "/custom/task-index.json"


@pytest.mark.parametrize("benchmark", ["harness-bench-lite", "harness-bench-medium"])
def test_benchmark_kwargs_preserve_harness_options(benchmark):
    options = {
        "recursion-limit": 80,
        "transient-attempts": 8,
        "forward-reasoning-history": False,
    }
    config = HarborRliEvaluatorConfig.model_validate(
        {"benchmark": benchmark, "benchmark_kwargs": options}
    )
    assert config.to_dict()["benchmark-kwargs"] == options


@pytest.mark.parametrize(
    "selector",
    [
        {"n-tasks": 1},
        {"task-name": "a"},
        {"exclude-task-name": "a"},
        {"benchmark-kwargs": {"task-id": "a"}},
        {"benchmark-kwargs": {"task_id": "a"}},
    ],
)
def test_filtered_results_are_available(evaluator_factory, selector):
    evaluator = evaluator_factory({"benchmark": "swe-bench-verified", **selector})
    write_metrics(evaluator)
    assert evaluator.get_metrics() == {"swe-bench-verified": 0.5}


@pytest.mark.parametrize("multiple", [False, True])
@pytest.mark.parametrize(
    "sync_enabled, online", [(False, False), (True, False), (True, True)]
)
def test_run_leaves_archiving_to_after_run(
    evaluator_factory, monkeypatch, tmp_path, multiple, sync_enabled, online
):
    evaluator = evaluator_factory(
        {
            "runs": [
                {"benchmark": "swe-bench-verified"},
                {"benchmark": "swe-bench-multilingual"},
            ]
        }
        if multiple
        else {"benchmark": "swe-bench-verified"}
    )
    runs = evaluator._runs if multiple else [evaluator]

    class FakeManager:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            for run in runs:
                write_metrics(run, complete=False)

        def wait_for_completion(self):
            pass

        def log_error(self):
            pass

    class FakePool(FakeManager):
        def run_all(self):
            self.start()

    monkeypatch.setattr(harbor_module, "SubprocessManager", FakeManager)
    monkeypatch.setattr(harbor_module, "SubprocessPoolRunner", FakePool)
    monkeypatch.setattr(
        harbor_module.JD, "get_jd_sync_dir", lambda: tmp_path if sync_enabled else None
    )
    monkeypatch.setattr(harbor_module.JD, "internet_is_available", lambda: online)
    archive = Mock(
        side_effect=lambda metrics_dir, *args: metrics_dir / "archive.tar.zst"
    )
    transfer = Mock()
    monkeypatch.setattr(harbor_module.JD, "tar_metrics_dir", archive)
    monkeypatch.setattr(harbor_module.JD, "symlink_metrics_for_transfer", transfer)
    evaluator.run()
    archive.assert_not_called()
    transfer.assert_not_called()
    assert evaluator.get_metrics() == {}
    assert all(run.metrics_path.exists() for run in runs)


@pytest.mark.parametrize("state", ["missing", "incomplete", "filtered", "complete"])
def test_after_run_publishes_complete_results(evaluator_factory, monkeypatch, state):
    payload = {"benchmark": "swe-bench-verified"}
    if state == "filtered":
        payload["task-name"] = "a"
    evaluator = evaluator_factory(payload)
    if state != "missing":
        write_metrics(evaluator, complete=state != "incomplete")
    publish, archive, transfer = Mock(), Mock(), Mock()
    monkeypatch.setattr(evaluator, "_send_metrics_to_jd", publish)
    monkeypatch.setattr(harbor_module.JD, "tar_metrics_dir", archive)
    monkeypatch.setattr(harbor_module.JD, "symlink_metrics_for_transfer", transfer)
    evaluator.after_run(evaluator.get_metrics())
    assert publish.call_count == int(state in ("filtered", "complete"))
    assert archive.call_count == transfer.call_count == 1
