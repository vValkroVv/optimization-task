import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

# obm_proxy code imports pam_proxies as a top-level package because the proxy
# subprocess runs with cwd=node_runner/src; mirror that layout for tests.
_NODE_RUNNER_SRC = (Path(__file__).parent.parent / "node_runner" / "src").as_posix()
if _NODE_RUNNER_SRC not in sys.path:
    sys.path.insert(0, _NODE_RUNNER_SRC)


@pytest.fixture
def scalar_ci_upload(monkeypatch):
    """Build real CI artifacts while replacing the S3 publication boundary."""
    from node_runner.src.job_dispatcher import JD

    def upload(*, make_builder, metrics_dir, **kwargs):
        scores_path = make_builder().build(metrics_dir)
        return {
            "cache_s3_path": f"s3://test/{scores_path.name}",
            "cache_dir_s3_path": "s3://test/",
        }

    get_version = JD.get_shooter_version
    monkeypatch.setattr(
        JD,
        "get_shooter_version",
        lambda path: get_version(path) if Path(path).is_file() else "test-producer",
    )
    monkeypatch.setattr(JD, "upload_metrics_and_cache_scores", upload)


@pytest.fixture
def convert_single_node_config() -> dict[str, Any]:
    return {
        "input_checkpoint_path": None,
        "tokenizer_path": None,
        "output_checkpoint_path": None,
        "mappers": ["map_fsdp_to_hf"],
        "input_tp_size": 1,
        "output_tp_size": 1,
        "skip_optimizer": False,
        "print_final_model_structure": True,
    }


@pytest.fixture
def convert_multi_node_config() -> dict[str, Any]:
    return {
        "input_checkpoint_path": None,
        "tokenizer_path": None,
        "output_checkpoint_path": None,
        "mappers": ["map_fsdp_to_hf"],
        "input_tp_size": 2,
        "output_tp_size": 1,
        "skip_optimizer": False,
        "print_final_model_structure": True,
    }


@pytest.fixture
def inference_config_multi_node() -> dict[str, Any]:
    return {
        "tp": 8,
        "dp": 1,
        "dtype": "auto",
        "enable-torch-compile": False,
        "enable-dp-attention": True,
        "mem-fraction-static": 0.7,
        "trust-remote-code": True,
    }


@pytest.fixture
def inference_config_single_node() -> dict[str, Any]:
    return {
        "tp": 8,
        "dp": 1,
        "dtype": "auto",
        "enable-torch-compile": False,
        "enable-dp-attention": True,
        "mem-fraction-static": 0.7,
        "trust-remote-code": True,
    }


@pytest.fixture
def make_hf_checkpoint(tmp_path: Path) -> Callable[[], Path]:
    def _make() -> Path:
        hf_dir = Path(tmp_path, "hf_checkpoint")
        hf_dir.mkdir(parents=True, exist_ok=True)
        (hf_dir / "config.json").write_text("{}")
        (hf_dir / "model.safetensors").write_text("")
        (hf_dir / "tokenizer.json").write_text("{}")
        return hf_dir

    return _make


@pytest.fixture
def make_non_hf_checkpoint(tmp_path: Path) -> Callable[[str], Path]:
    def _make(name: str = "non_hf_checkpoint") -> Path:
        d = Path(tmp_path, name)
        d.mkdir(parents=True, exist_ok=True)
        (d / "artifacts").mkdir(parents=True, exist_ok=True)
        (d / "non-hf-checkpoint-tp_rank0").mkdir(parents=True, exist_ok=True)
        (d / "ep0-ba16000-tp_rank0-ep_rank0").mkdir(parents=True, exist_ok=True)
        return d

    return _make


@pytest.fixture
def make_artifacts_dir(tmp_path: Path) -> Callable[[str], Path]:
    def _make(name: str = "artifacts") -> Path:
        d = Path(tmp_path, name)
        d.mkdir(parents=True, exist_ok=True)
        return d

    return _make


@pytest.fixture
def fake_subprocess(monkeypatch) -> Callable[[str], dict[str, Any]]:
    """Patch a SubprocessManager at a given import path and record calls."""

    def _patch(module_path: str) -> dict[str, Any]:
        called: dict[str, Any] = {}

        class FakeSP:
            def __init__(self, cmd, log, cwd):
                called["cmd"] = cmd
                called["log"] = log
                called["cwd"] = cwd
                self.exit_code = 0

            def start(self):
                called["started"] = True

            def wait_for_completion(self):
                called["waited"] = True

            def is_successful(self):
                called["successful"] = True
                return True

            def is_alive(self):
                called["alive"] = True
                return True

            def raise_for_exit_code(self):
                called["raised"] = True
                return True

        monkeypatch.setattr(module_path, FakeSP)
        return called

    return _patch


@pytest.fixture
def fail_on_sp_instantiation(monkeypatch) -> Callable[[str, str], None]:
    def _patch(module_path: str, message: str) -> None:
        def _raise_on_init(*args, **kwargs):
            raise AssertionError(message)

        monkeypatch.setattr(module_path, _raise_on_init)

    return _patch


@pytest.fixture
def harbor_job(tmp_path):
    def write(rewards):
        run_dir = tmp_path / "harbor_job"
        run_dir.mkdir()
        trial_index = 0
        for task, values in rewards.items():
            for value in values:
                trial = run_dir / str(trial_index)
                trial.mkdir()
                (trial / "result.json").write_text(
                    json.dumps(
                        {
                            "task_name": task,
                            "verifier_result": {"rewards": {"score": value}},
                        }
                    )
                )
                trial_index += 1
        (run_dir / "result.json").write_text(
            json.dumps({"n_total_trials": trial_index})
        )
        return run_dir / "result.json"

    return write
