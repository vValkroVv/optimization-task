from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from swe_bench_harbor_rli.cli import (
    _build_env,
    _build_harbor_cmd,
    _build_rli_config,
    _compute_pass_at_k,
    _normalized_config,
    _selected_agent_import_path,
    _summarize_job_result,
    _write_configured_pass_at_k,
    _write_reverified_metrics,
    run,
)


def _write_trial_result(
    job_dir: Path,
    trial_name: str,
    task_name: str,
    *,
    reward: int | None = None,
    exception_type: str | None = None,
) -> None:
    payload: dict[str, object] = {"task_name": task_name}
    if reward is not None:
        payload["verifier_result"] = {"rewards": {"reward": reward}}
    if exception_type is not None:
        payload["exception_info"] = {"exception_type": exception_type}
    result_path = job_dir / trial_name / "result.json"
    result_path.parent.mkdir(parents=True)
    result_path.write_text(json.dumps(payload), encoding="utf-8")


def test_normalized_config_matches_pam_swe_bench_defaults() -> None:
    expected = {
        "dataset": "/home/user/swe-bench-verified/datasets/swebench-verified",
        "agent": "mini-swe-agent",
        "agent-kwargs": {},
        "agent-timeout": None,
        "max-tokens": 32768,
        "n-concurrent": 4,
        "pass-at-k": 1,
        "env-api-base": None,
        "env-prefix": "swe-bench",
        "env-version": "1.0.2",
        "timeout-rli": 1620,
        "request-retries": 0,
        "transfer-retries": 2,
        "transfer-retry-backoff": 2.0,
        "session-ready-timeout": 600.0,
        "session-ready-poll-interval": 1.0,
        "enable-probe-warmup": True,
        "create-session-jitter-max-sec": 60.0,
        "probe-retry-backoff-sec": 2.0,
        "command-timeout": 180,
        "post-reverify": False,
        "post-reverify-grader": "rli",
    }

    config = _normalized_config("{}")

    assert {key: config[key] for key in expected} == expected


def test_normalized_config_rejects_n_attempts() -> None:
    with pytest.raises(ValueError, match="use pass-at-k"):
        _normalized_config(json.dumps({"n-attempts": 3}))


@pytest.mark.parametrize("value", [0, -1])
def test_normalized_config_rejects_invalid_pass_at_k(value: int) -> None:
    with pytest.raises(ValueError, match="pass-at-k must be a positive integer"):
        _normalized_config(json.dumps({"pass-at-k": value}))


def test_build_rli_config_includes_probe_warmup_fields() -> None:
    config = {
        "env-api-base": "https://rli.sberdevices.ru",
        "env-prefix": "swe-bench",
        "env-version": "1.0.0",
        "verify-ssl": False,
        "use-pip-proxy": True,
        "timeout": 180,
        "request-retries": 2,
        "request-retry-backoff": 2.0,
        "transfer-retries": 2,
        "transfer-retry-backoff": 2.0,
        "session-ready-timeout": 600.0,
        "session-ready-poll-interval": 1.0,
        "command-timeout": 600,
        "verifier-command-timeout": 1500,
        "enable-probe-warmup": True,
        "probe-warmup-command": "pwd",
        "create-session-jitter-max-sec": 3.0,
        "probe-retry-backoff-sec": 2.0,
    }

    rli_config = _build_rli_config(config, Path("/tmp/metrics"))

    assert rli_config["enable_probe_warmup"] is True
    assert rli_config["probe_warmup_command"] == "pwd"
    assert rli_config["create_session_jitter_max_sec"] == 3.0
    assert rli_config["probe_retry_backoff_sec"] == 2.0
    assert rli_config["verifier_command_timeout_ms"] == 1_500_000


@pytest.mark.parametrize(
    ("agent", "expected_import_path"),
    [
        (
            "terminus-2",
            "swe_bench_harbor_rli.agents.terminus.terminus_agent:RLITerminus2Agent",
        ),
        (
            "mini-swe-agent",
            "swe_bench_harbor_rli.agents.mini_swe_agent.rli_container_agent:"
            "RLIContainerMiniSweAgent",
        ),
        (
            "mini-swe-agent-proxy",
            "swe_bench_harbor_rli.agents.mini_swe_agent.mini_swe_agent:RLIMiniSweAgent",
        ),
        (
            "agentless",
            "swe_bench_harbor_rli.agents.agentless.agentless_agent:RLIAgentlessAgent",
        ),
        (
            "openhands",
            "swe_bench_harbor_rli.agents.openhands_sdk.openhands_sdk_agent:"
            "RLIOpenHandsSDKAgent",
        ),
    ],
)
def test_selected_agent_import_path_uses_supported_wrapper(
    agent: str,
    expected_import_path: str,
) -> None:
    assert _selected_agent_import_path({"agent": agent}) == expected_import_path


def test_selected_agent_import_path_rejects_custom_agent() -> None:
    with pytest.raises(ValueError, match="Unsupported swe_bench agent"):
        _selected_agent_import_path({"agent": "custom-agent"})


def test_build_harbor_cmd_does_not_inject_mini_swe_config_for_terminus(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(
        "swe_bench_harbor_rli.cli.shutil.which", lambda name: "/usr/bin/harbor"
    )

    cmd = _build_harbor_cmd(
        {
            "agent": "terminus-2",
            "dataset": "/home/user/swe-bench-verified",
            "model-name": "openai/pam-model",
            "agent-timeout": 14400,
            "n-concurrent": 1,
            "max-tokens": 16000,
            "agent-kwargs": {"record_terminal_session": False},
        },
        tmp_path,
    )

    agent = json.loads(
        (tmp_path / "harbor_job_config.json").read_text(encoding="utf-8")
    )["agents"][0]
    assert "--config" in cmd
    assert "--agent-import-path" not in cmd
    assert "--model" not in cmd
    assert "--agent-kwarg" not in cmd
    assert agent["import_path"] == (
        "swe_bench_harbor_rli.agents.terminus.terminus_agent:RLITerminus2Agent"
    )
    assert agent["model_name"] == "openai/pam-model"
    assert agent["override_timeout_sec"] == 14400
    assert agent["kwargs"]["max_tokens"] == 16000
    assert agent["kwargs"]["record_terminal_session"] is False
    assert "config_file" not in agent["kwargs"]


@pytest.mark.parametrize(
    ("payload", "expected_timeout"),
    [
        ({}, 6400),
        ({"llm-timeout": 1200}, 1200),
        ({"llm_timeout": 1300}, 1300),
        (
            {
                "llm-timeout": 1200,
                "agent-kwargs": {"llm_timeout": 900},
            },
            900,
        ),
    ],
)
def test_build_harbor_cmd_configures_llm_timeout(
    monkeypatch,
    tmp_path,
    payload,
    expected_timeout,
) -> None:
    monkeypatch.setattr(
        "swe_bench_harbor_rli.cli.shutil.which",
        lambda name: "/usr/bin/harbor",
    )
    config = _normalized_config(json.dumps(payload))

    _build_harbor_cmd(config, tmp_path)

    agent = json.loads(
        (tmp_path / "harbor_job_config.json").read_text(encoding="utf-8")
    )["agents"][0]
    assert agent["kwargs"]["llm_timeout"] == expected_timeout


def test_build_harbor_cmd_rejects_invalid_llm_timeout(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        "swe_bench_harbor_rli.cli.shutil.which",
        lambda name: "/usr/bin/harbor",
    )
    config = _normalized_config(json.dumps({"llm-timeout": 0}))

    with pytest.raises(ValueError, match="llm-timeout must be a positive integer"):
        _build_harbor_cmd(config, tmp_path)


def test_build_harbor_cmd_uses_local_rli_verifier_for_mini_agent(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        "swe_bench_harbor_rli.cli.shutil.which",
        lambda name: "/usr/bin/harbor",
    )

    cmd = _build_harbor_cmd(
        {
            "agent": "mini-swe-agent",
            "dataset": "swe-bench-verified",
            "model-name": "openai/pam-model",
            "n-concurrent": 2,
            "verifier-timeout": 2100,
            "pass-at-k": 3,
            "task-name": ["sympy", "sphinx"],
            "exclude-task-name": "django",
            "agent-kwargs": {
                "config_file": "swebench_backticks.yaml",
                "model_class": "litellm_textbased",
            },
        },
        tmp_path,
    )

    joined = " ".join(str(part) for part in cmd)
    agent = json.loads(
        (tmp_path / "harbor_job_config.json").read_text(encoding="utf-8")
    )["agents"][0]
    assert "--dataset swe-bench-verified" in joined
    assert "--verifier-timeout-multiplier 0.5833333333333334" in joined
    assert "--verifier-timeout " not in joined
    assert "--n-attempts 3" in joined
    assert joined.count("--include-task-name") == 2
    assert "--exclude-task-name django" in joined
    assert (
        "--verifier-import-path "
        "swe_bench_harbor_rli.src.verifier:SWEBenchLocalRLIVerifier"
    ) in joined
    assert agent["kwargs"]["config_file"] == "swebench_backticks.yaml"
    assert agent["kwargs"]["model_class"] == "litellm_textbased"


def test_build_harbor_cmd_omits_verifier_when_verification_disabled(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        "swe_bench_harbor_rli.cli.shutil.which",
        lambda name: "/usr/bin/harbor",
    )

    cmd = _build_harbor_cmd(
        {
            "agent": "terminus-2",
            "dataset": "swe-bench-verified",
            "disable-verification": True,
        },
        tmp_path,
    )

    assert "--disable-verification" in cmd
    assert "--verifier-import-path" not in cmd


def test_build_env_configures_model_api_and_rli_no_proxy(monkeypatch) -> None:
    monkeypatch.setenv("PYTHONPATH", "/existing/pythonpath")
    monkeypatch.setenv("NO_PROXY", "localhost")
    monkeypatch.delenv("no_proxy", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("MSWEA_API_KEY", raising=False)

    env = _build_env(
        {
            "model-api-base": "http://model-proxy:8000/v1",
            "api-key": "secret",
            "env-api-base": "https://rli.sberdevices.ru",
        }
    )

    assert env["OPENAI_API_BASE"] == "http://model-proxy:8000/v1"
    assert env["OPENAI_BASE_URL"] == "http://model-proxy:8000/v1"
    assert env["OPENAI_API_KEY"] == "secret"
    assert env["MSWEA_API_KEY"] == "secret"
    assert env["NO_PROXY"] == "localhost,rli.sberdevices.ru"
    assert env["no_proxy"] == "rli.sberdevices.ru"
    assert env["PYTHONPATH"].endswith(":/existing/pythonpath")
    assert env["PYTHONPATH"].count("/existing/pythonpath") == 1


def test_build_env_configures_output_token_budget(monkeypatch) -> None:
    monkeypatch.setenv("PAM_MAX_TOTAL_OUTPUT_TOKENS", "stale")

    env = _build_env({"max-total-output-tokens": 96_000_000})
    env_without_limit = _build_env({"max-total-output-tokens": None})

    assert env["PAM_MAX_TOTAL_OUTPUT_TOKENS"] == "96000000"
    assert "PAM_MAX_TOTAL_OUTPUT_TOKENS" not in env_without_limit


@pytest.mark.parametrize("value", [0, -1])
def test_build_env_rejects_invalid_output_token_budget(value: int) -> None:
    with pytest.raises(
        ValueError,
        match="max-total-output-tokens must be a positive integer",
    ):
        _build_env({"max-total-output-tokens": value})


def test_summarize_job_result_does_not_double_count_errors() -> None:
    result = {
        "stats": {
            "n_completed_trials": 89,
            "n_errored_trials": 24,
            "evals": {
                "terminus-2__pam-model__swe-bench-verified": {
                    "n_trials": 66,
                    "n_errors": 24,
                    "exception_stats": {
                        "AgentTimeoutError": ["agent-timeout"],
                        "ReadError": ["read-error"],
                    },
                    "metrics": [{"pass_rate": 0.22727272727272727}],
                }
            },
        }
    }

    summary = _summarize_job_result(result)

    assert summary["pass_rate"] == 15 / 89
    assert summary["pass_rate_completed"] == 15 / 66
    assert summary["passed"] == 15
    assert summary["graded"] == 66
    assert summary["completed"] == 89
    assert summary["total"] == 89
    assert summary["errors"] == 24
    assert summary["exception_types"] == ["AgentTimeoutError", "ReadError"]
    assert summary["rli_infrastructure_errors"] == ["ReadError"]


def test_compute_pass_at_k_marks_infrastructure_errors_incomplete(tmp_path) -> None:
    attempts = (
        ("task-a", 0, None),
        ("task-a", 0, None),
        ("task-a", 1, None),
        ("task-b", 1, None),
        ("task-b", 0, None),
        ("task-b", None, "ReadError"),
        ("task-c", 0, None),
        ("task-c", 0, None),
        ("task-c", None, "ReadError"),
    )
    for index, (task_name, reward, exception_type) in enumerate(attempts):
        _write_trial_result(
            tmp_path,
            f"trial-{index}",
            task_name,
            reward=reward,
            exception_type=exception_type,
        )

    assert _compute_pass_at_k(tmp_path, {"n_total_trials": 9}, 3) == {
        "k": 3,
        "score": 2 / 3,
        "passed_tasks": 2,
        "total_tasks": 3,
        "complete": False,
        "infrastructure_errors": 2,
    }


def test_compute_pass_at_k_marks_complete_run(tmp_path) -> None:
    for index, reward in enumerate((0, 0, 1)):
        _write_trial_result(
            tmp_path,
            f"trial-{index}",
            "task-a",
            reward=reward,
        )

    assert _compute_pass_at_k(tmp_path, {"n_total_trials": 3}, 3) == {
        "k": 3,
        "score": 1.0,
        "passed_tasks": 1,
        "total_tasks": 1,
        "complete": True,
        "infrastructure_errors": 0,
    }


def test_write_configured_pass_at_k_preserves_harbor_values(tmp_path) -> None:
    result_path = tmp_path / "result.json"
    result = {
        "stats": {
            "evals": {
                "agent__model__dataset": {
                    "pass_at_k": {"2": 0.75},
                }
            }
        }
    }
    result_path.write_text(json.dumps(result), encoding="utf-8")

    _write_configured_pass_at_k(
        result_path,
        result,
        {
            "k": 3,
            "score": 0.8,
            "complete": False,
            "infrastructure_errors": 1,
        },
    )

    stored = json.loads(result_path.read_text(encoding="utf-8"))
    assert stored["stats"]["evals"]["agent__model__dataset"]["pass_at_k"] == {
        "2": 0.75,
        "3": 0.8,
    }
    assert not result_path.with_suffix(".json.tmp").exists()


def test_write_reverified_metrics_unions_repeated_trials_and_grader(
    tmp_path,
) -> None:
    metrics_dir = tmp_path / "metrics"
    out_dir = tmp_path / "async"
    for trial_name, reward in (
        ("sympy__sympy-1__attempt-a", "0"),
        ("sympy__sympy-1__attempt-b", "1"),
        ("django__django-2__attempt-a", "0"),
    ):
        reward_path = metrics_dir / "run" / trial_name / "verifier" / "reward.txt"
        reward_path.parent.mkdir(parents=True, exist_ok=True)
        reward_path.write_text(reward, encoding="utf-8")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "metrics.json").write_text(
        json.dumps({"pass_rate": 0.2, "total": 10}),
        encoding="utf-8",
    )
    out_dir.mkdir()
    (out_dir / "results_async.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"instance_id": "sympy__sympy-1", "resolved": False}),
                json.dumps({"instance_id": "sphinx-doc__sphinx-2", "resolved": True}),
                json.dumps({"instance_id": "sphinx-doc__sphinx-2", "resolved": False}),
                "not-json",
            ]
        ),
        encoding="utf-8",
    )

    _write_reverified_metrics(metrics_dir, out_dir, grader_label="rli")

    payload = json.loads(
        (metrics_dir / "metrics_reverified.json").read_text(encoding="utf-8")
    )
    assert payload["resolved_instances"] == [
        "sphinx-doc__sphinx-2",
        "sympy__sympy-1",
    ]
    assert payload["in_job_resolved_count"] == 1
    assert payload["in_job_graded_count"] == 2
    assert payload["grader_resolved_count"] == 1
    assert payload["grader_graded_count"] == 2
    assert payload["pass_rate_reverified"] == 0.2
    assert payload["denominator"] == 10


def test_run_skip_primary_requires_existing_metrics(tmp_path) -> None:
    args = argparse.Namespace(
        metrics_dir=str(tmp_path),
        config_json=json.dumps({"skip-primary-run": True}),
    )

    assert run(args) == 2


def test_run_skip_primary_only_runs_post_reverify(
    monkeypatch,
    tmp_path,
) -> None:
    (tmp_path / "metrics.json").write_text("{}", encoding="utf-8")
    calls: list[tuple[dict, Path]] = []

    def fake_post_reverify(config: dict, metrics_dir: Path) -> None:
        calls.append((config, metrics_dir))

    monkeypatch.setattr(
        "swe_bench_harbor_rli.cli._run_post_reverify",
        fake_post_reverify,
    )
    args = argparse.Namespace(
        metrics_dir=str(tmp_path),
        config_json=json.dumps({"skip-primary-run": True, "post-reverify": True}),
    )

    assert run(args) == 0
    assert calls[0][0]["post-reverify"] is True
    assert calls[0][1] == tmp_path

@pytest.mark.parametrize("agent_timeout", [None, 3600])
def test_build_rli_config_passes_agent_timeout(tmp_path, agent_timeout) -> None:
    config = _normalized_config(json.dumps({"agent-timeout": agent_timeout}))

    rli_config = _build_rli_config(config, tmp_path)

    assert rli_config["agent_timeout_sec"] == agent_timeout


@pytest.mark.parametrize(
    ("option", "value", "flag"),
    [
        ("job-name", "retry-job", "--job-name"),
        (
            "environment-build-timeout-multiplier",
            4.0,
            "--environment-build-timeout-multiplier",
        ),
    ],
)
@pytest.mark.parametrize("configured", [False, True])
def test_build_harbor_cmd_passes_run_options(
    monkeypatch, tmp_path, option, value, flag, configured
) -> None:
    monkeypatch.setattr(
        "swe_bench_harbor_rli.cli.shutil.which", lambda name: "/usr/bin/harbor"
    )
    config = _normalized_config(json.dumps({option: value} if configured else {}))

    cmd = _build_harbor_cmd(config, tmp_path)

    if configured:
        assert cmd[cmd.index(flag) + 1] == str(value)
    else:
        assert flag not in cmd
