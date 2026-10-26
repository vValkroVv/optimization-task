from __future__ import annotations

import json
from pathlib import Path

import pytest

from terminal_bench.cli import (
    _build_env,
    _build_harbor_cmd,
    _build_rli_config,
    _compute_pass_at_k,
    _normalized_config,
    _selected_agent_import_path,
    _summarize_job_result,
    _write_configured_pass_at_k,
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


def test_normalized_config_defaults_pass_at_k() -> None:
    assert _normalized_config("{}")["pass-at-k"] == 1


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
        "env-prefix": "terminal-bench",
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


def test_selected_agent_import_path_uses_rli_terminus_wrapper() -> None:
    agent_import_path = _selected_agent_import_path({"agent": "terminus-2"})

    assert (
        agent_import_path
        == "terminal_bench.agents.terminus.terminus_agent:RLITerminus2Agent"
    )


def test_selected_agent_import_path_rejects_custom_agent() -> None:
    with pytest.raises(ValueError, match="Unsupported terminal_bench agent"):
        _selected_agent_import_path({"agent": "custom-agent"})


def test_build_harbor_cmd_does_not_inject_mini_swe_config_for_terminus(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(
        "terminal_bench.cli.shutil.which", lambda name: "/usr/bin/harbor"
    )

    config = _normalized_config(
        json.dumps(
            {
                "agent": "terminus-2",
                "dataset": "/home/user/terminal-bench-2",
                "model-name": "openai/pam-model",
                "agent-timeout": 14400,
                "n-concurrent": 1,
                "max-tokens": 16000,
                "agent-kwargs": {"record_terminal_session": False},
            }
        )
    )
    cmd = _build_harbor_cmd(config, tmp_path)

    joined = " ".join(str(part) for part in cmd)
    agent = json.loads(
        (tmp_path / "harbor_job_config.json").read_text(encoding="utf-8")
    )["agents"][0]
    assert "--config" in cmd
    assert "--agent-import-path" not in cmd
    assert "--model" not in cmd
    assert "--agent-kwarg" not in cmd
    assert agent["import_path"] == (
        "terminal_bench.agents.terminus.terminus_agent:RLITerminus2Agent"
    )
    assert agent["model_name"] == "openai/pam-model"
    assert agent["override_timeout_sec"] == 14400
    assert agent["kwargs"]["max_tokens"] == 16000
    assert agent["kwargs"]["record_terminal_session"] is False
    assert (
        "--verifier-import-path "
        "terminal_bench.src.verifier:TerminalBenchLocalRLIVerifier"
    ) in joined
    assert "config_file" not in agent["kwargs"]


def test_build_harbor_cmd_maps_pass_at_k_to_n_attempts(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        "terminal_bench.cli.shutil.which",
        lambda name: "/usr/bin/harbor",
    )
    config = _normalized_config(json.dumps({"pass-at-k": 3}))

    cmd = _build_harbor_cmd(config, tmp_path)

    assert "--n-attempts 3" in " ".join(str(part) for part in cmd)


def test_build_harbor_cmd_omits_verifier_when_verification_disabled(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(
        "terminal_bench.cli.shutil.which",
        lambda name: "/usr/bin/harbor",
    )

    cmd = _build_harbor_cmd(
        {
            "agent": "terminus-2",
            "dataset": "/home/user/terminal-bench-2",
            "disable-verification": True,
        },
        tmp_path,
    )

    assert "--disable-verification" in cmd
    assert "--verifier-import-path" not in cmd


def test_build_env_configures_output_token_budget(monkeypatch) -> None:
    monkeypatch.setenv("PAM_MAX_TOTAL_OUTPUT_TOKENS", "stale")

    env = _build_env({"max-total-output-tokens": 31_000_000})
    env_without_limit = _build_env({"max-total-output-tokens": None})

    assert env["PAM_MAX_TOTAL_OUTPUT_TOKENS"] == "31000000"
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
                "terminus-2__pam-model__terminal-bench-2": {
                    "n_trials": 66,
                    "n_errors": 24,
                    "metrics": [{"pass_rate": 0.22727272727272727}],
                }
            },
        }
    }

    summary = _summarize_job_result(result)

    assert summary["pass_rate"] == 0.22727272727272727
    assert summary["passed"] == 15
    assert summary["total"] == 66
    assert summary["errors"] == 24


def test_compute_pass_at_k_marks_infrastructure_errors_incomplete(tmp_path) -> None:
    attempts = (
        ("task-a", 0, None),
        ("task-a", 0, None),
        ("task-a", 1, None),
        ("task-b", 1, None),
        ("task-b", 0, None),
        ("task-b", None, "RLISessionUnavailableError"),
        ("task-c", 0, None),
        ("task-c", 0, None),
        ("task-c", None, "RLISessionUnavailableError"),
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
        "terminal_bench.cli.shutil.which", lambda name: "/usr/bin/harbor"
    )
    config = _normalized_config(json.dumps({option: value} if configured else {}))

    cmd = _build_harbor_cmd(config, tmp_path)

    if configured:
        assert cmd[cmd.index(flag) + 1] == str(value)
    else:
        assert flag not in cmd
