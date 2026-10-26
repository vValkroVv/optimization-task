import json
from pathlib import Path

import pytest

from contrib.terminal_bench import cli

pytestmark = pytest.mark.unit


def test_build_harbor_cmd_includes_rli_environment(tmp_path: Path):
    config = cli._normalized_config(
        json.dumps(
            {
                "model-api-base": "http://127.0.0.1:29999/v1",
                "max-tokens": 32768,
                "max-input-tokens": 98304,
                "task-name": "adaptive-rejection-sampler",
                "n-concurrent": 2,
            }
        )
    )

    cmd = cli._build_harbor_cmd(config, tmp_path)
    agent = json.loads(
        (tmp_path / "harbor_job_config.json").read_text(encoding="utf-8")
    )["agents"][0]

    assert "--config" in cmd
    assert "--agent-import-path" not in cmd
    assert "--agent" not in cmd
    assert agent["import_path"] == (
        "terminal_bench.agents.terminus.terminus_agent:RLITerminus2Agent"
    )
    assert agent["model_name"] == "openai/pam-model"
    assert "--environment-import-path" in cmd
    assert "terminal_bench.src.environment:RLIEnvironment" in cmd
    assert "--include-task-name" in cmd
    assert "adaptive-rejection-sampler" in cmd
    assert agent["kwargs"]["model_api_base"] == "http://127.0.0.1:29999/v1"
    assert agent["kwargs"]["max_tokens"] == 32768
    assert agent["kwargs"]["max_input_tokens"] == 98304
    kwarg = cmd[cmd.index("--environment-kwarg") + 1]
    payload = json.loads(kwarg.split("=", 1)[1])
    assert payload["api_base"] is None
    assert payload["env_prefix"] == "terminal-bench"
    assert payload["task_id"] == "adaptive-rejection-sampler"
    assert payload["request_retries"] == 2
    assert payload["request_retry_backoff"] == 2.0
    assert payload["transfer_retries"] == 2
    assert payload["transfer_retry_backoff"] == 2.0
    assert payload["use_pip_proxy"] is False
    assert payload["allow_internet_access"] is True


def test_build_harbor_cmd_accepts_mini_swe_agent_option(tmp_path: Path):
    config = cli._normalized_config(json.dumps({"agent": "mini-swe-agent"}))

    cmd = cli._build_harbor_cmd(config, tmp_path)
    agent = json.loads(
        (tmp_path / "harbor_job_config.json").read_text(encoding="utf-8")
    )["agents"][0]

    assert "--config" in cmd
    assert "--agent-import-path" not in cmd
    assert "--agent" not in cmd
    assert agent["import_path"] == (
        "terminal_bench.agents.mini_swe_agent.mini_swe_agent:RLIMiniSweAgent"
    )


def test_build_harbor_cmd_rejects_custom_agent(tmp_path: Path):
    config = cli._normalized_config(json.dumps({"agent": "custom-agent"}))

    with pytest.raises(ValueError, match="Unsupported terminal_bench agent"):
        cli._build_harbor_cmd(config, tmp_path)


def test_build_harbor_cmd_rejects_explicit_agent_import_path(tmp_path: Path):
    config = cli._normalized_config(
        json.dumps({"agent-import-path": "custom.module:Agent"})
    )

    with pytest.raises(ValueError, match="agent-import-path is not supported"):
        cli._build_harbor_cmd(config, tmp_path)


def test_summarize_job_result_from_reward_stats():
    result = {
        "n_total_trials": 3,
        "stats": {
            "n_errored_trials": 1,
            "evals": {
                "mini__model__terminal-bench": {
                    "n_trials": 3,
                    "n_errors": 1,
                    "reward_stats": {
                        "reward": {
                            "1": ["task-a", "task-b"],
                            "0": ["task-c"],
                        }
                    },
                }
            },
        },
    }

    summary = cli._summarize_job_result(result)

    assert summary["pass_rate"] == pytest.approx(2 / 3)
    assert summary["passed"] == 2
    assert summary["total"] == 3
    assert summary["errors"] == 1


def test_summarize_job_result_prefers_metrics():
    result = {
        "n_total_trials": 10,
        "stats": {
            "evals": {
                "mini__model__terminal-bench": {
                    "n_trials": 10,
                    "metrics": [{"pass_rate": 0.4}],
                }
            }
        },
    }

    summary = cli._summarize_job_result(result)

    assert summary == {"pass_rate": 0.4, "passed": 4, "total": 10, "errors": 0}


def test_load_latest_job_result_prefers_run_level_result(tmp_path: Path):
    jobs_dir = tmp_path / "harbor_jobs"
    run_dir = jobs_dir / "run-1"
    task_dir = run_dir / "task-1__abc"
    task_dir.mkdir(parents=True)

    run_result = run_dir / "result.json"
    task_result = task_dir / "result.json"
    run_result.write_text(json.dumps({"n_total_trials": 89}), encoding="utf-8")
    task_result.write_text(json.dumps({"trial_name": "task-1__abc"}), encoding="utf-8")

    path, payload = cli._load_latest_job_result(jobs_dir)

    assert path == run_result
    assert payload == {"n_total_trials": 89}


def test_summarize_job_result_from_harbor_mean_and_reward_stats():
    result = {
        "n_total_trials": 89,
        "stats": {
            "n_errored_trials": 29,
            "evals": {
                "terminus-2__pam-model__terminal-bench-2": {
                    "n_trials": 67,
                    "metrics": [{"mean": 40 / 89}],
                    "reward_stats": {
                        "reward": {
                            "1.0": [f"passed-{idx}" for idx in range(40)],
                            "0.0": [f"failed-{idx}" for idx in range(27)],
                        }
                    },
                }
            },
        },
    }

    summary = cli._summarize_job_result(result)

    assert summary["pass_rate"] == pytest.approx(40 / 89)
    assert summary["passed"] == 40
    assert summary["total"] == 89
    assert summary["errors"] == 29


def test_build_env_sets_no_proxy_for_rli_host(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "localhost")

    env = cli._build_env({"env-api-base": "https://rli.sberdevices.ru"})

    assert "localhost" in env["NO_PROXY"].split(",")
    assert "rli.sberdevices.ru" in env["NO_PROXY"].split(",")
    assert "rli.sberdevices.ru" in env["no_proxy"].split(",")


def test_build_harbor_cmd_uses_path_for_local_dataset(tmp_path: Path):
    config = cli._normalized_config(json.dumps({"dataset": str(tmp_path)}))

    cmd = cli._build_harbor_cmd(config, tmp_path)

    assert "--path" in cmd
    assert str(tmp_path) in cmd
    assert "--dataset" not in cmd


def test_tail_text_limits_output_lines():
    assert cli._tail_text("a\nb\nc", max_lines=2) == "b\nc"
